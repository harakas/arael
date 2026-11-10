//! The IEEE 754 arithmetic a kernel gets by default, which
//! double-single (df32) arithmetic and the solver's NaN checks rely on: NaN
//! and infinity are not assumed away, a compensated sum keeps its error
//! term, and a product rounded on its own (an `fma(a, b, 0)` pin or
//! `mul_rn`) is not fused into the add that consumes it. Contraction itself
//! stays on, as NVRTC's and hipRTC's defaults. On CUDA (feature `cuda`),
//! else on HIP (`hip`), else on Metal (`metal`), else on the CPU device
//! (`cpu`).
#![cfg(any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use arael_gpu::{Arg, Device, Launch};

#[cfg(feature = "cuda")]
type Dev = arael_gpu::cuda::CudaDevice;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
type Dev = arael_gpu::hip::HipDevice;
#[cfg(all(feature = "metal", target_vendor = "apple", not(any(feature = "cuda", feature = "hip"))))]
type Dev = arael_gpu::metal::MetalDevice;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "hip", all(feature = "metal", target_vendor = "apple")))))]
type Dev = arael_gpu::cpu::CpuDevice;

const SOURCE: &str = r#"
KERNEL void default_math(GLOBAL const float* in, GLOBAL float* out) {
    if (global_x() != 0) return;
    float nan = in[0], a = in[1], b = in[2], x = in[3], e = in[5], inf = in[6];

    out[0] = isnan(nan) ? 1.0f : 0.0f;
    out[1] = (nan != nan) ? 1.0f : 0.0f;
    out[2] = isnan(inf - inf) ? 1.0f : 0.0f;

    // two_sum: the rounding error of a + b.
    float s = a + b;
    float b_virtual = s - a;
    out[3] = (a - (s - b_virtual)) + (b - b_virtual);

    // quick_two_sum of a product pinned with fma(x, x, 0) and a smaller value.
    float p = fma(x, in[4], 0.0f);
    float sum = p + e;
    out[4] = e - (sum - p);

    // The same with the header's unfused operations.
    float q = mul_rn(x, in[4]);
    out[5] = sub_rn(e, sub_rn(add_rn(q, e), q));

    // An unpinned product used once: contraction fuses x * x + e. (A
    // product with more uses is fused or not by the compiler's own rules.)
    float fused = x * in[4] + e;
    out[6] = e - (fused - q);
}
"#;

/// `two_sum`'s error term on the host, where every operation rounds on its own.
fn two_sum_error(a: f32, b: f32) -> f32 {
    let s = a + b;
    let b_virtual = s - a;
    (a - (s - b_virtual)) + (b - b_virtual)
}

#[test]
fn kernels_keep_default_math() {
    let dev = Dev::new(0).unwrap();
    let kernel = dev.compile(SOURCE, "", "default_math").unwrap();

    let (a, b) = (1.0f32, 1e-9f32);
    // x * x is 1 + 2^-11 + 2^-24 exactly: rounding drops 2^-24, a tie, to 1 + 2^-11.
    let x = 1.0f32 + 1.0 / 4096.0;
    let e = 1.0f32 / 67_108_864.0; // 2^-26
    let input = [f32::NAN, a, b, x, x, e, f32::INFINITY];

    let p = x * x;
    let quick_two_sum_error = e - ((p + e) - p);
    // A fused x * x + e rounds once and up instead. In f64 the product and the
    // sum are exact, so one rounding to f32 is what an FMA gives.
    let fused_sum = ((x as f64) * (x as f64) + e as f64) as f32;
    assert_ne!(quick_two_sum_error, e - (fused_sum - p), "the inputs tell a fused add apart");
    assert_ne!(two_sum_error(a, b), 0.0, "the sum has a rounding error");

    let mut inp = dev.alloc(4 * input.len()).unwrap();
    dev.write(&mut inp, 0, bytemuck::cast_slice(&input)).unwrap();
    let out = dev.alloc(4 * 7).unwrap();
    dev.launch(&kernel, Launch::linear(1, 32), &[Arg::Block(&inp, 0), Arg::Block(&out, 0)]).unwrap();
    let mut actual = [0f32; 7];
    dev.read(&out, 0, bytemuck::cast_slice_mut(&mut actual)).unwrap();

    assert_eq!(actual[0], 1.0, "is_nan(NaN)");
    assert_eq!(actual[1], 1.0, "NaN != NaN");
    assert_eq!(actual[2], 1.0, "inf - inf is NaN");
    assert_eq!(actual[3], two_sum_error(a, b), "two_sum's error term");
    assert_eq!(actual[4], quick_two_sum_error, "quick_two_sum after an fma(x, x, 0) pin");
    assert_eq!(actual[5], quick_two_sum_error, "quick_two_sum with mul_rn and add_rn");
    assert_eq!(actual[6], e - (fused_sum - p), "an unpinned product is contracted, the compilers' default");
}

/// `Arch::flushes_subnormals` is what the device does: a product, a sum and
/// a subnormal input's copy through arithmetic, each subnormal on the host.
#[test]
fn the_arch_says_whether_subnormals_flush() {
    const FLUSH: &str = r#"
KERNEL void flush(GLOBAL const float* in, GLOBAL float* out) {
    if (global_x() != 0) return;
    out[0] = mul_rn(in[0], in[1]);
    out[1] = sub_rn(in[2], in[3]);
    out[2] = mul_rn(in[4], in[5]);
    out[3] = add_rn(in[4], in[6]);
}
"#;
    let dev = Dev::new(0).unwrap();
    let kernel = dev.compile(FLUSH, "", "flush").unwrap();
    let tiny = f32::MIN_POSITIVE;
    let input = [tiny, 0.25, tiny * 1.5, tiny, tiny / 8.0, 1.0, 0.0];
    let host = [input[0] * input[1], input[2] - input[3], input[4] * input[5], input[4] + input[6]];
    assert!(host.iter().all(|v| v.is_subnormal()), "the host's results are subnormal: {host:?}");
    let mut inp = dev.alloc(4 * input.len()).unwrap();
    dev.write(&mut inp, 0, bytemuck::cast_slice(&input)).unwrap();
    let out = dev.alloc(4 * 4).unwrap();
    dev.launch(&kernel, Launch::linear(1, 32), &[Arg::Block(&inp, 0), Arg::Block(&out, 0)]).unwrap();
    let mut actual = [0f32; 4];
    dev.read(&out, 0, bytemuck::cast_slice_mut(&mut actual)).unwrap();
    let want = if dev.arch().flushes_subnormals { [0.0; 4] } else { host };
    assert_eq!(actual, want, "{}: flushes_subnormals is {}", dev.arch().name, dev.arch().flushes_subnormals);
}

#[test]
fn compile_errors_are_reported() {
    let dev = Dev::new(0).unwrap();
    let err = dev.compile("KERNEL void broken(GLOBAL float* x) { x[0] = undefined_name; }", "", "broken");
    let Err(arael_gpu::GpuError::Compile { entry, log }) = err else {
        panic!("expected a compile error, got {err:?}");
    };
    assert_eq!(entry, "broken");
    assert!(log.contains("undefined_name"), "the compiler's log names the problem:\n{log}");
}
