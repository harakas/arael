//! Sweep-only timing on the pose graph, for a before/after comparison of
//! the generated sweeps: builds the graph once, binds the steady-state
//! pattern once, then times many assemblies and cost evaluations on one
//! context and reports the sweep regions themselves (the threaded or
//! sequential sweep, without the scatter that follows it), the indexed
//! scatter and the cost sweep, as min and median over the calls in
//! microseconds. One line per run, fixed format, for a driver to collect.
//!
//!     THREADS=1 CALLS=200 PGO_DATASET=datasets/city10000.g2o \
//!         taskset -c 7 ./target/release/sweep_ab

#[path = "../g2o.rs"]
mod g2o;
#[path = "../arael_runner.rs"]
mod arael_runner;

use arael::store::HessianBinder;
use arael::simple_lm::{LmProblemInternals, RootProblem};
use arael::store::{block_partition_from_spans, csc_from_cells};
use arael::threads::{Context, ParTiming};
use arael_runner::Graph;
use bench_harness::arael::Model as Pipeline;

/// The accumulated sweep clocks, in microseconds: assembly region (both
/// forms), scatter, cost region (both forms).
fn snapshot(t: &ParTiming) -> (f64, f64, f64) {
    let us = |d: std::time::Duration| d.as_secs_f64() * 1e6;
    (us(t.assembly.par.region + t.assembly.seq.region),
     us(t.scatter),
     us(t.cost.par.region + t.cost.seq.region))
}

fn stats(v: &mut Vec<f64>) -> (f64, f64) {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[0], v[v.len() / 2])
}

fn main() {
    let threads: usize = std::env::var("THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
    let calls: usize = std::env::var("CALLS").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let path = std::env::var("PGO_DATASET").unwrap_or_else(|_| "datasets/city10000.g2o".to_string());
    let ds = g2o::load(&path, false);
    let mut g = Graph::build(&ds);
    let mut params = Vec::new();
    RootProblem::serialize(&mut g, &mut params);
    let n = params.len();

    let mut ctx = Context::new();
    ctx.set_threads(threads);
    ctx.set_timing(true);
    g.begin_with_context(&mut ctx);
    let stores = ctx.sweeps().map_or(1, |s| s.threads);

    // The steady-state route: the tile-expanded pattern, bound once.
    let mut cells = Vec::new();
    g.collect_hessian_cells(&mut cells, &mut ctx);
    let mut spans = Vec::new();
    g.collect_param_block_spans(&mut spans, &mut ctx);
    let partition = block_partition_from_spans(&spans, n);
    let (mut csc, mut resolver) = csc_from_cells::<f64>(&partition, &cells);
    let mut positions = arael::store::PositionStream::new();
    g.bind_hessian_positions(
        &mut HessianBinder::Tiled(&mut |i, j| resolver.resolve_tile(i, j)),
        &mut positions,
        &mut ctx,
    );
    let mut grad = vec![0.0; n];

    for _ in 0..5 {
        g.calc_grad_hessian_sparse_indexed(&params, &mut grad, &mut csc.vals, &positions, &mut ctx);
        g.calc_cost_with_context(&params, &mut ctx);
    }
    let (mut asm, mut scat, mut cost) = (Vec::with_capacity(calls), Vec::with_capacity(calls), Vec::with_capacity(calls));
    for _ in 0..calls {
        let b = snapshot(ctx.sweep_timing());
        g.calc_grad_hessian_sparse_indexed(&params, &mut grad, &mut csc.vals, &positions, &mut ctx);
        let a = snapshot(ctx.sweep_timing());
        asm.push(a.0 - b.0);
        scat.push(a.1 - b.1);
        let b = snapshot(ctx.sweep_timing());
        g.calc_cost_with_context(&params, &mut ctx);
        let a = snapshot(ctx.sweep_timing());
        cost.push(a.2 - b.2);
    }
    let (asm_min, asm_med) = stats(&mut asm);
    let (scat_min, scat_med) = stats(&mut scat);
    let (cost_min, cost_med) = stats(&mut cost);
    println!("sweep_ab pgo n={} stores={} calls={} asm_min={:.2} asm_med={:.2} cost_min={:.2} cost_med={:.2} scatter_min={:.2} scatter_med={:.2}",
        n, stores, calls, asm_min, asm_med, cost_min, cost_med, scat_min, scat_med);
}
