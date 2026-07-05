// factrs runners (GN and LM), fed the same weighted cost function.
//
// The graph is assembled through factrs's public API: its own
// BetweenResidual<SE2> with a diagonal information noise model
// (rotation-first component ordering, matching its own g2o loader's
// permutation), and the shared unit-weight gauge prior on pose 0.
// factrs's LM starts at lambda = 1e-10 with diagonal damping --
// problem-appropriate by default under the initial-damping policy.

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::g2o::{Dataset, PoseIn};
use factrs::assign_symbols;
use factrs::core::{BetweenResidual, GaussianNoise, GaussNewton, Graph, LevenMarquardt, PriorResidual, Values, SE2};
use factrs::fac;
use factrs::optimizers::{BaseOptParams, LevenParams, OptError, OptObserver};
use factrs::traits::Optimizer;

assign_symbols!(X: SE2);

// Iteration counting through factrs's observer hook.
static STEPS: AtomicUsize = AtomicUsize::new(0);

struct StepCounter;
impl OptObserver for StepCounter {
    fn on_step(&self, _values: &Values, _time: i64) {
        STEPS.fetch_add(1, Ordering::Relaxed);
    }
}

fn build(ds: &Dataset) -> (Graph, Values) {
    let mut graph = Graph::new();
    let mut values = Values::new();
    for (i, p) in ds.poses.iter().enumerate() {
        values.insert(X(i as u32), SE2::new(p.th, p.x, p.y));
    }
    for e in &ds.edges {
        let delta = SE2::new(e.dth, e.dx, e.dy);
        // Diagonal information, rotation-first (factrs SE2 tangent order).
        let inf = factrs::linalg::Vector3::new(e.wr * e.wr, e.wt * e.wt, e.wt * e.wt);
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
        PriorResidual::new(SE2::new(p0.th, p0.x, p0.y)),
        X(0),
        1.0 as std
    ]);
    (graph, values)
}

pub struct RunOut {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    pub iterations: usize,
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

fn run(ds: &Dataset, gn: bool) -> RunOut {
    let optimize = |max_iter: usize| -> (f64, usize, Values) {
        let (graph, init) = build(ds);
        let before = STEPS.load(Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        let result = if gn {
            let mut opt = GaussNewton::new(base_params(max_iter), graph);
            opt.observers_mut().add(StepCounter);
            opt.optimize(init)
        } else {
            let params = LevenParams { base: base_params(max_iter), ..Default::default() };
            let mut opt = LevenMarquardt::new(params, graph);
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
        (ms, STEPS.load(Ordering::Relaxed) - before, values)
    };

    let (first_iter_ms, _, _) = optimize(1);
    let (solve_ms, iterations, values) = optimize(100);
    let poses = (0..ds.poses.len())
        .map(|i| {
            let p: &SE2 = values.get(X(i as u32)).expect("missing pose");
            PoseIn { x: p.x(), y: p.y(), th: p.theta() }
        })
        .collect();
    RunOut { solve_ms, first_iter_ms, iterations, poses }
}

pub fn run_gn(ds: &Dataset) -> RunOut {
    run(ds, true)
}

pub fn run_lm(ds: &Dataset) -> RunOut {
    run(ds, false)
}
