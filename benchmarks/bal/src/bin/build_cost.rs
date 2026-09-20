//! What a plain assembly pays for its block store, on a BAL scene:
//! times the plain `calc_grad_hessian_sparse` (which builds a store on
//! every call), the same assembly through a context built once, and the
//! store build alone into a fresh context, interleaved call by call, and
//! reports min and median of each in microseconds plus the build's share
//! of the plain call. One line per run, fixed format.
//!
//!     CALLS=50 BAL_DATASET=datasets/problem-372-47423-pre.txt \
//!         taskset -c 7 ./target/release/build_cost

#[path = "../bal.rs"]
mod bal;
#[path = "../arael_runner.rs"]
mod arael_runner;

use arael::simple_lm::{CooMatrix, LmProblem, LmProblemInternals, RootProblem};
use arael::threads::{Context, ParTiming};

fn us(d: std::time::Duration) -> f64 { d.as_secs_f64() * 1e6 }

/// The assembly's phase clocks, in microseconds: parameter update, the
/// store zeroing, the sweep region (both forms), the gradient gather,
/// the Hessian scatter. What is left against the wall time is the
/// gradient zeroing, the cost merge, the COO clear and the clock reads.
fn phases(t: &ParTiming) -> [f64; 5] {
    [us(t.assembly_update),
     us(t.assembly_zero),
     us(t.assembly.par.region + t.assembly.seq.region),
     us(t.gather_grad),
     us(t.scatter)]
}

/// The wall time of one sweep and the phase deltas it produced, from
/// the clocks before and after: [wall, update, zero, region, gather,
/// scatter, rest].
fn deltas(wall: f64, b: [f64; 5], a: [f64; 5]) -> [f64; 7] {
    let d: [f64; 5] = std::array::from_fn(|k| a[k] - b[k]);
    [wall, d[0], d[1], d[2], d[3], d[4], wall - d.iter().sum::<f64>()]
}

fn med(v: &mut Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let calls: usize = std::env::var("CALLS").ok().and_then(|v| v.parse().ok()).unwrap_or(50);
    let path = std::env::var("BAL_DATASET")
        .unwrap_or_else(|_| "datasets/problem-372-47423-pre.txt".to_string());
    let ds = bal::load(&path);
    let mut scene = arael_runner::build_f64(&ds);
    let mut params = Vec::new();
    RootProblem::serialize(&mut scene, &mut params);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut coo = CooMatrix::new(n);

    let mut ctx = Context::new();
    ctx.set_timing(true);
    scene.begin_with_context(&mut ctx);
    for _ in 0..3 {
        coo.clear();
        scene.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
        coo.clear();
        scene.calc_grad_hessian_sparse_with_context(&params, &mut grad, &mut coo, &mut ctx);
    }

    // Wall times of the plain call, the build and the drop; the warm and
    // the first sweep split by the assembly's own phase clocks.
    let (mut plain, mut build, mut dropped) =
        (Vec::with_capacity(calls), Vec::with_capacity(calls), Vec::with_capacity(calls));
    let mut warm: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::with_capacity(calls));
    let mut first: [Vec<f64>; 7] = std::array::from_fn(|_| Vec::with_capacity(calls));
    for _ in 0..calls {
        coo.clear();
        let t = std::time::Instant::now();
        scene.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
        plain.push(us(t.elapsed()));

        coo.clear();
        let b = phases(ctx.sweep_timing());
        let t = std::time::Instant::now();
        scene.calc_grad_hessian_sparse_with_context(&params, &mut grad, &mut coo, &mut ctx);
        let s = deltas(us(t.elapsed()), b, phases(ctx.sweep_timing()));
        for k in 0..7 { warm[k].push(s[k]); }

        let mut fresh = Context::new();
        fresh.set_timing(true);
        let t = std::time::Instant::now();
        scene.__blocks_in(&mut fresh);
        build.push(us(t.elapsed()));

        // The first sweep over a freshly built store, and the drop of a
        // built store: with the build, the three parts of a plain call.
        coo.clear();
        let b = phases(fresh.sweep_timing());
        let t = std::time::Instant::now();
        scene.calc_grad_hessian_sparse_with_context(&params, &mut grad, &mut coo, &mut fresh);
        let s = deltas(us(t.elapsed()), b, phases(fresh.sweep_timing()));
        for k in 0..7 { first[k].push(s[k]); }
        let t = std::time::Instant::now();
        drop(fresh);
        dropped.push(us(t.elapsed()));
    }
    println!("build_cost bal n={} calls={} (medians, us)", n, calls);
    println!("  plain={:.1} build={:.1} drop={:.1}", med(&mut plain), med(&mut build), med(&mut dropped));
    for (name, v) in [("warm ", &mut warm), ("first", &mut first)] {
        println!("  {} wall={:.1} update={:.1} zero={:.1} region={:.1} gather={:.1} scatter={:.1} rest={:.1}",
            name, med(&mut v[0]), med(&mut v[1]), med(&mut v[2]), med(&mut v[3]), med(&mut v[4]), med(&mut v[5]), med(&mut v[6]));
    }
}
