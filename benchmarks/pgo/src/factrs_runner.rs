// factrs runners (GN and LM), fed the same weighted cost function.
//
// The graph is assembled through factrs's public API: its own
// BetweenResidual<SE2> with a diagonal information noise model
// (rotation-first component ordering, matching its own g2o loader's
// permutation), and the shared unit-weight gauge prior on pose 0.
// factrs's LM starts at lambda = 1e-10 with diagonal damping --
// problem-appropriate by default under the initial-damping policy.

use crate::factrs_counting::{since, counts, CountingSolver, StepCounter};
use crate::g2o::{Dataset, PoseIn};
use factrs::assign_symbols;
use factrs::core::{BetweenResidual, GaussianNoise, GaussNewton, Graph, LevenMarquardt, PriorResidual, Values, SE2};
use factrs::dtype;
use factrs::fac;
use factrs::optimizers::{BaseOptParams, LevenParams, OptError};
use factrs::traits::Optimizer;

assign_symbols!(X: SE2);

// The dataset is f64; factrs's scalar is whatever `dtype` the crate was built
// with. The casts are what let this file compile under both, so the f32 runner
// is the same code and not a second implementation of it.
fn build(ds: &Dataset) -> (Graph, Values) {
    let mut graph = Graph::new();
    let mut values = Values::new();
    for (i, p) in ds.poses.iter().enumerate() {
        values.insert(X(i as u32), SE2::new(p.th as dtype, p.x as dtype, p.y as dtype));
    }
    for e in &ds.edges {
        let delta = SE2::new(e.dth as dtype, e.dx as dtype, e.dy as dtype);
        // Diagonal information, rotation-first (factrs SE2 tangent order).
        let inf = factrs::linalg::Vector3::new(
            (e.wr * e.wr) as dtype, (e.wt * e.wt) as dtype, (e.wt * e.wt) as dtype);
        let noise = GaussianNoise::<3>::from_vec_inf(inf.as_view());
        graph.add_factor(fac![
            BetweenResidual::new(delta),
            (X(e.a as u32), X(e.b as u32)),
            noise
        ]);
    }
    // Unit-weight gauge prior on pose 0 (same convention as every runner).
    let p0 = &ds.poses[0];
    graph.add_factor(fac![
        PriorResidual::new(SE2::new(p0.th as dtype, p0.x as dtype, p0.y as dtype)),
        X(0),
        1.0 as std
    ]);
    (graph, values)
}

pub struct RunOut {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    /// Attempts: accepted steps plus factrs's in-step damping retries, each of
    /// which costs a factorization. Recovered by counting linear solves.
    pub iterations: usize,
    pub accepted: usize,
    /// t(2 iterations), for the caller to difference against t(1). `None` when
    /// the second step was rejected, which would make the difference a retry.
    pub two_iter_ms: Option<f64>,
    pub poses: Vec<PoseIn>,
}

fn base_params(max_iterations: usize) -> BaseOptParams {
    BaseOptParams {
        max_iterations,
        // Same termination class as the other systems.
        error_tol_relative: 1e-5,
        error_tol_absolute: 1e-5,
        ..Default::default()
    }
}

// (elapsed ms, accepted steps, attempts, values)
fn optimize(ds: &Dataset, gn: bool, max_iter: usize) -> (f64, usize, usize, Values) {
    let (graph, init) = build(ds);
    let before = counts();
    let t0 = std::time::Instant::now();
    let result = if gn {
        let mut opt = GaussNewton::new(base_params(max_iter), graph);
        opt.set_solver(CountingSolver::default());
        opt.observers_mut().add(StepCounter);
        opt.optimize(init)
    } else {
        let params = LevenParams { base: base_params(max_iter), ..Default::default() };
        let mut opt = LevenMarquardt::new(params, graph);
        opt.set_solver(CountingSolver::default());
        opt.observers_mut().add(StepCounter);
        opt.optimize(init)
    };
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    // Hitting the iteration cap is a reportable outcome (the values
    // ride along in the error), not a harness failure -- the
    // validation stage judges the solution.
    let values = match result {
        Ok(v) => v,
        Err(OptError::MaxIterations(v)) => v,
        Err(e) => panic!("factrs failed: {:?}", e),
    };
    let (accepted, attempts) = since(before);
    (ms, accepted, attempts, values)
}

// Fastest of PROBE_SUBROUNDS runs of the same probe, with its accepted step
// and attempt counts.
fn probe(ds: &Dataset, gn: bool, max_iter: usize) -> (f64, usize, usize) {
    let mut best = f64::INFINITY;
    let (mut accepted, mut attempts) = (0, 0);
    for _ in 0..crate::probe::PROBE_SUBROUNDS {
        let (ms, acc, att, _) = optimize(ds, gn, max_iter);
        best = best.min(ms);
        accepted = acc;
        attempts = att;
    }
    (best, accepted, attempts)
}

fn run(ds: &Dataset, gn: bool) -> RunOut {
    // The first solve in a process pays cold allocator and cache costs the
    // later ones do not; discard it so the probes are timed on equal footing.
    let _ = optimize(ds, gn, 1);
    let (first_ms, first_accepted, first_attempts) = probe(ds, gn, 1);
    let first_iter_ms = crate::probe::first_iter_ms(first_ms, first_attempts, first_accepted);
    let (two_ms, two_accepted, _) = probe(ds, gn, 2);
    let two_iter_ms = crate::probe::two_iter_ms(two_ms, first_iter_ms, two_accepted);
    let (solve_ms, accepted, iterations, values) = optimize(ds, gn, 100);
    let poses = (0..ds.poses.len())
        .map(|i| {
            let p: &SE2 = values.get(X(i as u32)).expect("missing pose");
            PoseIn { x: p.x() as f64, y: p.y() as f64, th: p.theta() as f64 }
        })
        .collect();
    RunOut { solve_ms, first_iter_ms, iterations, accepted, two_iter_ms, poses }
}

pub fn run_gn(ds: &Dataset) -> RunOut {
    run(ds, true)
}

pub fn run_lm(ds: &Dataset) -> RunOut {
    run(ds, false)
}
