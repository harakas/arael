// Scalar FP microbench: is f32 faster than f64 for the ops that dominate
// the loc iteration? atan2 (libm), sqrt, div, and mul-add chains. Companion
// to the LOC_TIMING=1 per-phase mode -- run both on a target core (e.g. the
// Raspberry Pi 5) to attribute an f32-vs-f64 gap to libm vs the FPU:
//
//   cargo run -r --bin fpbench
//
// Pin to a core yourself (taskset) if comparing across runs.
use std::time::Instant;

fn bench<F: FnMut() -> f64>(name: &str, mut f: F) {
    // warmup
    f();
    let mut best = f64::INFINITY;
    for _ in 0..5 {
        let t = Instant::now();
        let s = f();
        let dt = t.elapsed().as_secs_f64();
        std::hint::black_box(s);
        if dt < best { best = dt; }
    }
    println!("{:<22} {:>8.2} ns/op", name, best * 1e9 / N as f64);
}

const N: usize = 2_000_000;

fn main() {
    let xs64: Vec<f64> = (0..N).map(|i| 0.1 + (i % 997) as f64 * 0.013).collect();
    let ys64: Vec<f64> = (0..N).map(|i| -3.0 + (i % 991) as f64 * 0.0061).collect();
    let xs32: Vec<f32> = xs64.iter().map(|&v| v as f32).collect();
    let ys32: Vec<f32> = ys64.iter().map(|&v| v as f32).collect();

    bench("atan2 f64", || {
        let mut s = 0.0f64;
        for i in 0..N { s += ys64[i].atan2(xs64[i]); }
        s
    });
    bench("atan2 f32", || {
        let mut s = 0.0f32;
        for i in 0..N { s += ys32[i].atan2(xs32[i]); }
        s as f64
    });
    bench("sqrt f64", || {
        let mut s = 0.0f64;
        for i in 0..N { s += xs64[i].sqrt(); }
        s
    });
    bench("sqrt f32", || {
        let mut s = 0.0f32;
        for i in 0..N { s += xs32[i].sqrt(); }
        s as f64
    });
    bench("div f64", || {
        let mut s = 1.0f64;
        for i in 0..N { s += ys64[i] / xs64[i]; }
        s
    });
    bench("div f32", || {
        let mut s = 1.0f32;
        for i in 0..N { s += ys32[i] / xs32[i]; }
        s as f64
    });
    // dependent mul-add chain (latency-bound, like straight-line residual math)
    bench("muladd chain f64", || {
        let mut s = 0.5f64;
        for i in 0..N { s = s * 0.9999 + xs64[i]; }
        s
    });
    bench("muladd chain f32", || {
        let mut s = 0.5f32;
        for i in 0..N { s = s * 0.9999 + xs32[i]; }
        s as f64
    });
    // independent mul-adds (throughput-bound)
    bench("muladd indep f64", || {
        let (mut a, mut b, mut c, mut d) = (0.1f64, 0.2, 0.3, 0.4);
        for i in (0..N).step_by(4) {
            a = a * 0.999 + xs64[i];
            b = b * 0.999 + xs64[i + 1];
            c = c * 0.999 + xs64[i + 2];
            d = d * 0.999 + xs64[i + 3];
        }
        a + b + c + d
    });
    bench("muladd indep f32", || {
        let (mut a, mut b, mut c, mut d) = (0.1f32, 0.2, 0.3, 0.4);
        for i in (0..N).step_by(4) {
            a = a * 0.999 + xs32[i];
            b = b * 0.999 + xs32[i + 1];
            c = c * 0.999 + xs32[i + 2];
            d = d * 0.999 + xs32[i + 3];
        }
        (a + b + c + d) as f64
    });
}
