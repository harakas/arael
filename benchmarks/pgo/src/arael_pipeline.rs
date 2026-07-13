// One arael pipeline, four models (2D/3D x f64/f32).
//
// The model structs have to stay separate: the arael macro bakes the scalar
// into the generated code, so the f32 and f64 models are different types, and
// SE2 and SE3 are different problems. Everything around them -- build the model,
// probe it, solve it, read the poses back -- is the same for all four, and
// keeping four copies of it is precisely how the f32 config lost the `verbose`
// flag its f64 twin had, and how a fix to the 3D probe missed the 2D one.
//
// A model supplies the four things that genuinely differ (how to build it, how
// to move parameters in and out, which solver its scalar needs); the pipeline
// below is written once.

use arael::simple_lm::{LmConfig, LmProblem, LmResult};
use arael::utils::Float;

use crate::probe::PROBE_SUBROUNDS;

pub struct RunOut<P> {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    /// Attempts: accepted steps plus damping retries, each of which costs a
    /// factorization.
    pub iterations: usize,
    pub accepted: usize,
    /// t(2 iterations), for the caller to difference against t(1). `None` when
    /// the pair cannot be differenced -- see `probe::two_iter_ms`.
    pub two_iter_ms: Option<f64>,
    pub poses: Vec<P>,
}

pub trait Model: Clone + LmProblem<Self::Scalar> {
    type Scalar: Float;
    type Dataset;
    type Pose;

    /// Problem-appropriate initial damping, before the `ARAEL_LAMBDA0`
    /// override. 2D and 3D want different values; the precisions do not.
    fn lambda0() -> f64;

    fn build(ds: &Self::Dataset) -> Self;
    fn serialize(&mut self, out: &mut Vec<Self::Scalar>);
    fn deserialize(&mut self, x: &[Self::Scalar]);
    fn poses(&self) -> Vec<Self::Pose>;

    /// The scalar picks the solver (`SparseFaer` / `SparseFaerF32`), which is
    /// the one thing a generic function cannot choose for itself.
    fn solve(
        params: &[Self::Scalar],
        model: &mut Self,
        cfg: &LmConfig<Self::Scalar>,
    ) -> LmResult<Self::Scalar>;
}

/// ARAEL_LAMBDA0 overrides the model's damping, for experiments.
pub fn lambda0<M: Model>() -> f64 {
    std::env::var("ARAEL_LAMBDA0")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(M::lambda0)
}

/// PGO_DRIVER=nielsen swaps the fixed damping schedule for the gain-ratio
/// driver: lambda then follows how well the quadratic model predicted the step
/// actually taken, instead of a fixed up/down ladder. It changes the iteration
/// count, not the cost per iteration.
pub fn nielsen() -> bool {
    std::env::var("PGO_DRIVER").as_deref() == Ok("nielsen")
}

pub fn config<M: Model>(max_iters: usize) -> LmConfig<M::Scalar> {
    // Float: num::NumCast, so a literal reaches either scalar the same way.
    let f = |v: f64| <M::Scalar as num::NumCast>::from(v).unwrap();
    let cfg = LmConfig {
        verbose: std::env::var("VERBOSE").is_ok(),
        gather_timing: true,
        // GTSAM's defaults, which Ceres and g2o are also configured to.
        abs_precision: f(1e-5),
        rel_precision: f(1e-5),
        // Stop as soon as the tolerance test says converged, the way Ceres and
        // g2o do. The library defaults (min_iters 5, patience 3) put a floor
        // under the iteration count, which on a problem that converges in four
        // iterations would time a fifth that improves the cost by 4e-14.
        patience: 1,
        min_iters: 1,
        max_iters,
        initial_lambda: f(lambda0::<M>()),
        ..Default::default()
    };
    if nielsen() {
        cfg.with_driver(arael::simple_lm::NielsenLambdaDriver::default())
    } else {
        cfg
    }
}

/// Times a probe solve on throwaway copies of the model, fastest of
/// [`PROBE_SUBROUNDS`].
///
/// A discarded warmup runs first: the first solve in a process pays cold
/// allocator and cache costs the later ones do not (the symbolic factorization
/// alone runs a fifth slower), so timing the one- and two-iteration probes as
/// they come would charge that cost to the one-iteration probe alone and
/// inflate the difference between them.
///
/// The copies keep the probe from mutating the model: re-supplying the initial
/// parameter vector does not reset parametrization state held outside it, such
/// as the reference rotation a `QuaternionParam` re-centres on after a step.
fn timed<M: Model>(
    params: &[M::Scalar],
    model: &M,
    cfg: &LmConfig<M::Scalar>,
) -> (f64, LmResult<M::Scalar>) {
    let mut r = M::solve(params, &mut model.clone(), cfg); // warmup, discarded
    let mut best = f64::INFINITY;
    for _ in 0..PROBE_SUBROUNDS {
        let t0 = std::time::Instant::now();
        r = M::solve(params, &mut model.clone(), cfg);
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    (best, r)
}

pub fn run<M: Model>(ds: &M::Dataset) -> RunOut<M::Pose> {
    let mut model = M::build(ds);
    let mut params: Vec<M::Scalar> = Vec::new();
    model.serialize(&mut params);

    // One iteration, and two, each on a fresh copy. Their difference is one
    // complete iteration with the setup cancelled out.
    let (first_ms, first) = timed::<M>(&params, &model, &config::<M>(1));
    let first_iter_ms =
        crate::probe::first_iter_ms(first_ms, first.iterations, first.accepted_iterations);
    let (two_ms, two) = timed::<M>(&params, &model, &config::<M>(2));
    let two_iter_ms =
        crate::probe::two_iter_ms(two_ms, first_iter_ms, two.accepted_iterations);

    let t0 = std::time::Instant::now();
    let result = M::solve(&params, &mut model, &config::<M>(100));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    model.deserialize(&result.x);

    RunOut {
        solve_ms,
        first_iter_ms,
        iterations: result.iterations,
        accepted: result.accepted_iterations,
        two_iter_ms,
        poses: model.poses(),
    }
}
