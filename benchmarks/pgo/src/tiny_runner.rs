// tiny-solver runners (GN and LM), fed the same weighted cost function.
//
// tiny-solver's shipped read_g2o ignores the information matrices (a
// `// todo` in its parser), so the problem is assembled here through its
// public Factor API with a weighted between factor instead. The factor
// mirrors tiny_solver::factors::BetweenFactorSE2 exactly, plus sqrt-info
// row scaling -- the solver, autodiff, and linear algebra are all
// tiny-solver's own.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::g2o::{Dataset, PoseIn};
use tiny_solver::factors::Factor;
use tiny_solver::na;
use tiny_solver::optimizer::Optimizer;

#[derive(Debug, Clone)]
struct WeightedBetweenSE2 {
    dx: f64,
    dy: f64,
    dth: f64,
    wt: f64,
    wr: f64,
}

impl<T: na::RealField> Factor<T> for WeightedBetweenSE2 {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        let p0 = &params[0];
        let p1 = &params[1];
        let se2_origin_k0 = na::Isometry2::new(
            na::Vector2::new(p0[1].clone(), p0[2].clone()),
            p0[0].clone(),
        );
        let se2_origin_k1 = na::Isometry2::new(
            na::Vector2::new(p1[1].clone(), p1[2].clone()),
            p1[0].clone(),
        );
        let se2_k0_k1 = na::Isometry2::new(
            na::Vector2::<T>::new(T::from_f64(self.dx).unwrap(), T::from_f64(self.dy).unwrap()),
            T::from_f64(self.dth).unwrap(),
        );
        let diff = se2_origin_k1.inverse() * se2_origin_k0 * se2_k0_k1;
        let wt = T::from_f64(self.wt).unwrap();
        let wr = T::from_f64(self.wr).unwrap();
        na::DVector::from_vec(vec![
            diff.translation.x.clone() * wt.clone(),
            diff.translation.y.clone() * wt,
            diff.rotation.angle() * wr,
        ])
    }
}

// Iteration counting: tiny-solver reports iterations only through its
// `log` trace lines ("iter:<n> ..."); a counting logger observes them
// without patching the crate. The check is a cheap prefix match.
static ITER_COUNT: AtomicUsize = AtomicUsize::new(0);

struct IterCounter;
impl log::Log for IterCounter {
    fn enabled(&self, meta: &log::Metadata) -> bool {
        meta.target().starts_with("tiny_solver")
    }
    fn log(&self, record: &log::Record) {
        if record.target().starts_with("tiny_solver") {
            let msg = std::fmt::format(*record.args());
            if msg.starts_with("iter:") {
                ITER_COUNT.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
    fn flush(&self) {}
}

pub fn install_iter_counter() {
    log::set_logger(&IterCounter).expect("logger already set");
    log::set_max_level(log::LevelFilter::Trace);
}

fn build(ds: &Dataset) -> (tiny_solver::Problem, HashMap<String, na::DVector<f64>>) {
    let mut problem = tiny_solver::Problem::new();
    let mut init = HashMap::new();
    for (i, p) in ds.poses.iter().enumerate() {
        init.insert(format!("x{}", i), na::DVector::from_vec(vec![p.th, p.x, p.y]));
    }
    for e in &ds.edges {
        problem.add_residual_block(
            3,
            &[&format!("x{}", e.a), &format!("x{}", e.b)],
            Box::new(WeightedBetweenSE2 {
                dx: e.dx, dy: e.dy, dth: e.dth, wt: e.wt, wr: e.wr,
            }),
            None,
        );
    }
    // Unit-weight gauge prior on pose 0 (same convention as the arael and
    // GTSAM runners).
    let p0 = &ds.poses[0];
    problem.add_residual_block(
        3,
        &["x0"],
        Box::new(tiny_solver::factors::PriorFactor { v: na::DVector::from_vec(vec![p0.th, p0.x, p0.y]) }),
        None,
    );
    (problem, init)
}

fn options(max_iteration: usize) -> tiny_solver::OptimizerOptions {
    tiny_solver::OptimizerOptions {
        max_iteration,
        ..Default::default() // 1e-5 abs / 1e-5 rel decrease, sparse Cholesky
    }
}

fn extract(ds: &Dataset, values: &HashMap<String, na::DVector<f64>>) -> Vec<PoseIn> {
    (0..ds.poses.len())
        .map(|i| {
            let v = &values[&format!("x{}", i)];
            PoseIn { x: v[1], y: v[2], th: v[0] }
        })
        .collect()
}

pub struct RunOut {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    pub iterations: usize,
    pub poses: Vec<PoseIn>,
}

fn run(ds: &Dataset, gn: bool) -> RunOut {
    let (problem, init) = build(ds);

    let optimize = |max_iter: usize| -> (f64, usize, Option<HashMap<String, na::DVector<f64>>>) {
        let before = ITER_COUNT.load(Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        let result = if gn {
            tiny_solver::GaussNewtonOptimizer::new().optimize(&problem, &init, Some(options(max_iter)))
        } else {
            tiny_solver::LevenbergMarquardtOptimizer::default().optimize(&problem, &init, Some(options(max_iter)))
        };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        (ms, ITER_COUNT.load(Ordering::Relaxed) - before, result)
    };

    let (first_iter_ms, _, _) = optimize(1);
    let (solve_ms, iterations, result) = optimize(100);
    let values = result.expect("tiny-solver returned None");
    RunOut { solve_ms, first_iter_ms, iterations, poses: extract(ds, &values) }
}

pub fn run_gn(ds: &Dataset) -> RunOut {
    run(ds, true)
}

pub fn run_lm(ds: &Dataset) -> RunOut {
    run(ds, false)
}
