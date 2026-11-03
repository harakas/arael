//! Expressions printed by arael-sym's C++ printer (`cpp_body_with`),
//! compiled by the native runtime (CUDA, else HIP) and run on the device,
//! agree with arael-sym's own evaluation, in double and in float.
#![cfg(any(feature = "gpu-native-cuda", feature = "gpu-native-hip"))]

use arael::sym::*;
#[cfg(feature = "gpu-native-cuda")]
use arael_gpu::native::cuda::CudaDevice as Dev;
#[cfg(all(feature = "gpu-native-hip", not(feature = "gpu-native-cuda")))]
use arael_gpu::native::hip::HipDevice as Dev;
use arael_gpu::bytemuck;
use arael_gpu::native::{Arg, Device, Launch};
use std::collections::HashMap;

/// The expressions under test, over `x` and `y`.
fn exprs() -> Vec<E> {
    let (x, y) = (symbol("x"), symbol("y"));
    vec![
        sin(x.clone()) * y.clone() + 2.0,
        pow(x.clone(), 3.0),
        pow(x.clone(), -2.0),
        pow(x.clone(), 0.5),
        pow(x.clone(), 1.7),
        atan2(y.clone(), x.clone()),
        exp(-(x.clone() * y.clone())),
        ln(x.clone()),
        log2(x.clone()),
        log10(x.clone()),
        abs(y.clone()),
        tanh(x.clone() * y.clone()),
        asin(y.clone() * 0.5),
        acos(y.clone() * 0.5),
        branch(y.clone(), sin(x.clone()), cos(x.clone())),
        heaviside(y.clone()) * x.clone(),
        clamp(y.clone(), -0.5, 0.5),
        with_wide(x.clone() * y.clone() - 1.0) * 3.0,
        (sin(x.clone() * y.clone()) / x.clone()).diff("x"),
        (sin(x.clone() * y.clone()) / x.clone()).diff("y"),
    ]
}

/// `exprs` evaluated on the device at `points` (x, y), in `scalar`.
fn on_device<F: bytemuck::Pod + Default>(dev: &Dev, scalar: &str, points: &[(f64, f64)], cast: fn(f64) -> F) -> Vec<Vec<F>> {
    let es = exprs();
    let read = |s: &str| match s {
        "x" => Some("v[0]".to_string()),
        "y" => Some("v[1]".to_string()),
        _ => None,
    };
    let body = cpp_body_with(&es, if scalar == "float" { "f32" } else { "f64" }, &read).unwrap();
    let mut source = format!(
        "KERNEL void eval(GLOBAL const SCALAR* in, GLOBAL SCALAR* out, u32 n) {{\n\
         u32 i = global_x();\nif (i >= n) return;\nGLOBAL const SCALAR* v = in + 2 * i;\n{}",
        body.stmts
    );
    for (k, o) in body.outputs.iter().enumerate() {
        source.push_str(&format!("out[{} * i + {k}] = {o};\n", es.len()));
    }
    source.push_str("}\n");
    let kernel = dev.compile(&source, &format!("#define SCALAR {scalar}\n"), "eval").unwrap();

    let input: Vec<F> = points.iter().flat_map(|&(x, y)| [cast(x), cast(y)]).collect();
    let bytes = std::mem::size_of::<F>();
    let mut inp = dev.alloc(input.len() * bytes).unwrap();
    dev.write(&mut inp, 0, bytemuck::cast_slice(&input)).unwrap();
    let out = dev.alloc(points.len() * es.len() * bytes).unwrap();
    let n = points.len() as u32;
    dev.launch(&kernel, Launch::linear(n, 64), &[Arg::Block(&inp, 0), Arg::Block(&out, 0), Arg::U32(n)]).unwrap();
    let mut flat = vec![F::default(); points.len() * es.len()];
    dev.read(&out, 0, bytemuck::cast_slice_mut(&mut flat)).unwrap();
    flat.chunks(es.len()).map(|c| c.to_vec()).collect()
}

fn points() -> Vec<(f64, f64)> {
    vec![(0.3, -0.8), (0.7, 0.25), (1.1, 0.9), (1.9, -0.05), (2.5, 0.6)]
}

fn expected() -> Vec<Vec<f64>> {
    let es = exprs();
    points()
        .iter()
        .map(|&(x, y)| {
            let vars = HashMap::from([("x", x), ("y", y)]);
            es.iter().map(|e| e.eval(&vars).unwrap()).collect()
        })
        .collect()
}

#[test]
fn printed_cpp_runs_as_evaluated_in_double() {
    let dev = Dev::new(0).unwrap();
    let got = on_device::<f64>(&dev, "double", &points(), |v| v);
    for (p, (row, want)) in got.iter().zip(expected()).enumerate() {
        for (k, (g, w)) in row.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-13 * (1.0 + w.abs()), "point {p}, expression {k}: device {g}, eval {w}");
        }
    }
}

#[test]
fn printed_cpp_runs_as_evaluated_in_float() {
    let dev = Dev::new(0).unwrap();
    let got = on_device::<f32>(&dev, "float", &points(), |v| v as f32);
    for (p, (row, want)) in got.iter().zip(expected()).enumerate() {
        for (k, (g, w)) in row.iter().zip(&want).enumerate() {
            let g = *g as f64;
            assert!((g - w).abs() <= 2e-6 * (1.0 + w.abs()), "point {p}, expression {k}: device {g}, eval {w}");
        }
    }
}
