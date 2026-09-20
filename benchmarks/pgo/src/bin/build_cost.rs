//! What a plain assembly pays for its block store, on the pose graph:
//! times the plain `calc_grad_hessian_sparse` (which builds a store on
//! every call), the same assembly through a context built once, and the
//! store build alone into a fresh context, interleaved call by call, and
//! reports min and median of each in microseconds plus the build's share
//! of the plain call. One line per run, fixed format.
//!
//!     CALLS=100 PGO_DATASET=datasets/city10000.g2o \
//!         taskset -c 7 ./target/release/build_cost

#[path = "../g2o.rs"]
mod g2o;
#[path = "../arael_runner.rs"]
mod arael_runner;

use arael::simple_lm::{CooMatrix, LmProblem, LmProblemInternals, RootProblem};
use arael::threads::{Context, ParTiming};
use arael_runner::Graph;
use bench_harness::arael::Model as Pipeline;

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
    let calls: usize = std::env::var("CALLS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
    let path = std::env::var("PGO_DATASET").unwrap_or_else(|_| "datasets/city10000.g2o".to_string());
    let ds = g2o::load(&path, false);
    let mut g = Graph::build(&ds);
    let mut params = Vec::new();
    RootProblem::serialize(&mut g, &mut params);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut coo = CooMatrix::new(n);

    let mut ctx = Context::new();
    ctx.set_timing(true);
    g.begin_with_context(&mut ctx);
    for _ in 0..5 {
        coo.clear();
        g.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
        coo.clear();
        g.calc_grad_hessian_sparse_with_context(&params, &mut grad, &mut coo, &mut ctx);
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
        g.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
        plain.push(us(t.elapsed()));

        coo.clear();
        let b = phases(ctx.sweep_timing());
        let t = std::time::Instant::now();
        g.calc_grad_hessian_sparse_with_context(&params, &mut grad, &mut coo, &mut ctx);
        let s = deltas(us(t.elapsed()), b, phases(ctx.sweep_timing()));
        for k in 0..7 { warm[k].push(s[k]); }

        let mut fresh = Context::new();
        fresh.set_timing(true);
        let t = std::time::Instant::now();
        g.__blocks_in(&mut fresh);
        build.push(us(t.elapsed()));

        // The first sweep over a freshly built store, and the drop of a
        // built store: with the build, the three parts of a plain call.
        coo.clear();
        let b = phases(fresh.sweep_timing());
        let t = std::time::Instant::now();
        g.calc_grad_hessian_sparse_with_context(&params, &mut grad, &mut coo, &mut fresh);
        let s = deltas(us(t.elapsed()), b, phases(fresh.sweep_timing()));
        for k in 0..7 { first[k].push(s[k]); }
        let t = std::time::Instant::now();
        drop(fresh);
        dropped.push(us(t.elapsed()));
    }
    println!("build_cost pgo n={} calls={} (medians, us)", n, calls);
    println!("  plain={:.1} build={:.1} drop={:.1}", med(&mut plain), med(&mut build), med(&mut dropped));
    for (name, v) in [("warm ", &mut warm), ("first", &mut first)] {
        println!("  {} wall={:.1} update={:.1} zero={:.1} region={:.1} gather={:.1} scatter={:.1} rest={:.1}",
            name, med(&mut v[0]), med(&mut v[1]), med(&mut v[2]), med(&mut v[3]), med(&mut v[4]), med(&mut v[5]), med(&mut v[6]));
    }
}
