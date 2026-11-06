//! The kernel library's select helpers, as arael-sym prints a `select` on
//! an index against them: `arael_select_index` gives the arm an index picks
//! (-1 an integer outside the arms, -2 no integer), `arael_nan_of` the NaN
//! a select gives where the CPU's panics. On CUDA (feature `cuda`), else on
//! HIP (`hip`).
#![cfg(any(feature = "cuda", feature = "hip"))]

use arael_gpu::{Arg, Device, Launch, Wide, ASSEMBLY, WIDE};

#[cfg(feature = "cuda")]
type Dev = arael_gpu::cuda::CudaDevice;
#[cfg(all(feature = "hip", not(feature = "cuda")))]
type Dev = arael_gpu::hip::HipDevice;

const KERNEL: &str = r#"
KERNEL void select_helpers(GLOBAL const float* inf, GLOBAL const double* ind, GLOBAL int* out, GLOBAL int* isnan_out) {
    if (global_x() != 0) return;
    for (int i = 0; i < 9; i++) {
        out[i] = arael_select_index(inf[i], 3u);
        out[9 + i] = arael_select_index(ind[i], 3u);
    }
    isnan_out[0] = isnan(arael_nan_of(1.0f)) ? 1 : 0;
    isnan_out[1] = isnan(arael_nan_of(1.0)) ? 1 : 0;
}
"#;

#[test]
fn a_select_index_is_the_arm_or_says_why_not() {
    let dev = Dev::new(0).unwrap();
    let source = format!("{WIDE}\n{ASSEMBLY}\n{KERNEL}");
    let defines = format!("{}#define SCALAR float\n", Wide::F64.defines());
    let kernel = dev.compile(&source, &defines, "select_helpers").unwrap();
    let input = [0.0, 1.0, 2.0, 3.0, -1.0, 2.5, f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
    let expected = [0, 1, 2, -1, -1, -2, -2, -2, -2];
    let input_f: Vec<f32> = input.iter().map(|&v| v as f32).collect();
    let mut inf = dev.alloc(4 * 9).unwrap();
    dev.write(&mut inf, 0, bytemuck::cast_slice(&input_f)).unwrap();
    let mut ind = dev.alloc(8 * 9).unwrap();
    dev.write(&mut ind, 0, bytemuck::cast_slice(&input)).unwrap();
    let out = dev.alloc(4 * 18).unwrap();
    let nan = dev.alloc(4 * 2).unwrap();
    let args = [Arg::Block(&inf, 0), Arg::Block(&ind, 0), Arg::Block(&out, 0), Arg::Block(&nan, 0)];
    dev.launch(&kernel, Launch::linear(1, 32), &args).unwrap();
    let mut actual = [0i32; 18];
    dev.read(&out, 0, bytemuck::cast_slice_mut(&mut actual)).unwrap();
    assert_eq!(actual[..9], expected, "float");
    assert_eq!(actual[9..], expected, "double");
    let mut nans = [0i32; 2];
    dev.read(&nan, 0, bytemuck::cast_slice_mut(&mut nans)).unwrap();
    assert_eq!(nans, [1, 1], "arael_nan_of is a NaN in float and double");
}
