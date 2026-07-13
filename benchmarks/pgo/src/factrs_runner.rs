// factrs runners (GN and LM), fed the same weighted cost function.
//
// The graph is assembled through factrs's public API: its own
// BetweenResidual<SE2> with a diagonal information noise model
// (rotation-first component ordering, matching its own g2o loader's
// permutation), and the shared unit-weight gauge prior on pose 0.
// factrs's LM starts at lambda = 1e-10 with diagonal damping --
// problem-appropriate by default under the initial-damping policy.

use bench_harness::factrs::{counts, since, CountingSolver, StepCounter};
use crate::g2o::{Dataset, PoseIn};
use bench_harness::table::Row;
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

pub type RunOut = Row<Vec<PoseIn>>;


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
fn optimize_counted(ds: &Dataset, gn: bool, max_iter: usize) -> (usize, usize, Values) {
    let (graph, init) = build(ds);
    let before = counts();
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
    // Hitting the iteration cap is a reportable outcome (the values
    // ride along in the error), not a harness failure -- the
    // validation stage judges the solution.
    let values = match result {
        Ok(v) => v,
        Err(OptError::MaxIterations(v)) => v,
        Err(e) => panic!("factrs failed: {:?}", e),
    };
    let (accepted, attempts) = since(before);
    (accepted, attempts, values)
}

fn solution_of(ds: &Dataset, values: &Values) -> Vec<PoseIn> {
    (0..ds.poses.len())
        .map(|i| {
            let p: &SE2 = values.get(X(i as u32)).expect("missing pose");
            PoseIn { x: p.x() as f64, y: p.y() as f64, th: p.theta() as f64 }
        })
        .collect()
}

fn run(ds: &Dataset, gn: bool) -> RunOut {
    bench_harness::solver::run(100, |max_iter| {
        let (accepted, attempts, values) = optimize_counted(ds, gn, max_iter);
        bench_harness::solver::Outcome {
            accepted,
            attempts,
            solution: solution_of(ds, &values),
        }
    })
}

pub fn run_gn(ds: &Dataset) -> RunOut {
    run(ds, true)
}

pub fn run_lm(ds: &Dataset) -> RunOut {
    run(ds, false)
}
