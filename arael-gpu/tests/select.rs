//! The kernel library's select helpers, as arael-sym prints a `select` on
//! an index against them: `arael_select_index` gives the arm an index picks
//! (-1 an integer outside the arms, -2 no integer), `arael_nan_of` the NaN
//! a select gives where the CPU's panics. On CUDA (feature `cuda`), else on
//! HIP (`hip`), else on Metal (`metal`), else on the CPU device (`cpu`).
#![cfg(any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]

use arael_gpu::{Arg, Device, Launch, ASSEMBLY, WIDE};

#[cfg(feature = "cuda")]
type Dev = arael_gpu::cuda::CudaDevice;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
type Dev = arael_gpu::hip::HipDevice;
#[cfg(all(feature = "metal", target_vendor = "apple", not(any(feature = "cuda", feature = "hip"))))]
type Dev = arael_gpu::metal::MetalDevice;
#[cfg(all(feature = "cpu", not(any(feature = "cuda", feature = "hip", all(feature = "metal", target_vendor = "apple")))))]
type Dev = arael_gpu::cpu::CpuDevice;

// The helpers in float, and in double where the device has it.
const KERNEL: &str = r#"
KERNEL void select_helpers(GLOBAL const float* inf, GLOBAL int* out, GLOBAL int* isnan_out) {
    if (global_x() != 0) return;
    for (int i = 0; i < 9; i++) out[i] = arael_select_index(inf[i], 3u);
    isnan_out[0] = isnan(arael_nan_of(1.0f)) ? 1 : 0;
}
#if HAS_F64
KERNEL void select_helpers_f64(GLOBAL const double* ind, GLOBAL int* out, GLOBAL int* isnan_out) {
    if (global_x() != 0) return;
    for (int i = 0; i < 9; i++) out[i] = arael_select_index(ind[i], 3u);
    isnan_out[0] = isnan(arael_nan_of(1.0)) ? 1 : 0;
}
#endif
"#;

#[test]
fn a_select_index_is_the_arm_or_says_why_not() {
    let dev = Dev::new(0).unwrap();
    let source = format!("{WIDE}\n{ASSEMBLY}\n{KERNEL}");
    let defines = format!("{}#define SCALAR float\n", arael_gpu::test_util::wide(&dev).defines());
    let input = [0.0, 1.0, 2.0, 3.0, -1.0, 2.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    let expected = [0, 1, 2, -1, -1, -2, -2, -2, -2];
    let input_f: Vec<f32> = input.iter().map(|&v| v as f32).collect();
    // the entry `entry` on `values`: the arms it picks, and whether arael_nan_of is a NaN
    let run = |entry: &str, values: &[u8]| {
        let kernel = dev.compile(&source, &defines, entry).unwrap();
        let mut input = dev.alloc(values.len()).unwrap();
        dev.write(&mut input, 0, values).unwrap();
        let out = dev.alloc(4 * 9).unwrap();
        let nan = dev.alloc(4).unwrap();
        dev.launch(&kernel, Launch::linear(1, 32), &[Arg::Block(&input, 0), Arg::Block(&out, 0), Arg::Block(&nan, 0)]).unwrap();
        let mut actual = [0i32; 9];
        dev.read(&out, 0, bytemuck::cast_slice_mut(&mut actual)).unwrap();
        let mut is_nan = [0i32; 1];
        dev.read(&nan, 0, bytemuck::cast_slice_mut(&mut is_nan)).unwrap();
        (actual, is_nan[0])
    };
    assert_eq!(run("select_helpers", bytemuck::cast_slice(&input_f)), (expected, 1), "float");
    if arael_gpu::test_util::has_f64(&dev, "the helpers in double") {
        assert_eq!(run("select_helpers_f64", bytemuck::cast_slice(&input)), (expected, 1), "double");
    }
}
