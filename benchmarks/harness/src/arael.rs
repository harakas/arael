// The arael pipeline: build the model, probe it, solve it, read the solution.
//
// The model structs cannot be shared -- the arael macro bakes the scalar into
// the generated code, so an f32 model and an f64 model are different types, and
// each benchmark's problem is its own. The pipeline around them is the same for
// all of them, and keeping a copy per model is how one benchmark's f32 config
// lost the verbose flag its f64 twin had.

use ::arael::simple_lm::{LmConfig, LmProblem, LmResult};
use ::arael::utils::Float;

use crate::probe::{first_iter_ms, two_iter_ms, PROBE_SUBROUNDS};
use crate::table::Row;

pub trait Model: Clone + LmProblem<Self::Scalar> {
    type Scalar: Float;
    /// Whatever the benchmark builds its model from: a parsed dataset, a
    /// generated scene.
    type Input;
    type Solution;

    /// Problem-appropriate initial damping, before the ARAEL_LAMBDA0 override.
    /// Solver benchmarks give each algorithm the damping the problem wants, not
    /// the one it ships with; see the benchmark READMEs' damping policy.
    fn lambda0() -> f64;

    fn build(input: &Self::Input) -> Self;
    fn serialize(&mut self, out: &mut Vec<Self::Scalar>);
    fn deserialize(&mut self, x: &[Self::Scalar]);
    fn solution(&self) -> Self::Solution;

    /// The scalar picks the solver (`SparseFaer` / `SparseFaerF32`, or a band
    /// solver), which is the one thing a generic function cannot choose for
    /// itself.
    fn solve(
        params: &[Self::Scalar],
        model: &mut Self,
        cfg: &LmConfig<Self::Scalar>,
    ) -> LmResult<Self::Scalar>;

    /// Anything else this benchmark wants on the config. The defaults below are
    /// the ones every benchmark has agreed on so far.
    fn tune(_cfg: &mut LmConfig<Self::Scalar>) {}
}

/// ARAEL_LAMBDA0 overrides the model's damping, for experiments.
pub fn lambda0<M: Model>() -> f64 {
    std::env::var("ARAEL_LAMBDA0")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(M::lambda0)
}

/// DRIVER=nielsen swaps the fixed damping schedule for the gain-ratio driver:
/// lambda then follows how well the quadratic model predicted the step actually
/// taken, instead of a fixed up/down ladder. It changes the iteration count, not
/// the cost per iteration.
pub fn nielsen() -> bool {
    std::env::var("DRIVER").as_deref() == Ok("nielsen")
}

pub fn config<M: Model>(max_iters: usize) -> LmConfig<M::Scalar> {
    // Float: num::NumCast, so a literal reaches either scalar the same way.
    let f = |v: f64| <M::Scalar as num::NumCast>::from(v).unwrap();
    let mut cfg = LmConfig {
        verbose: std::env::var("VERBOSE").is_ok(),
        gather_timing: true,
        // The termination class every system in these benchmarks is held to.
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
    M::tune(&mut cfg);
    if nielsen() {
        cfg.with_driver(::arael::simple_lm::NielsenLambdaDriver::default())
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
/// they come would charge that cost to the one-iteration probe alone and inflate
/// the difference between them.
///
/// The copies keep the probe from mutating the model: re-supplying the initial
/// parameter vector does not reset parametrization state held outside it, such
/// as the reference rotation a QuaternionParam re-centres on after a step. A
/// probe that ran on the real model left it half-advanced, and the solve that
/// followed inherited a warm start it never asked for.
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

pub fn run<M: Model>(input: &M::Input) -> Row<M::Solution> {
    let mut model = M::build(input);
    let mut params: Vec<M::Scalar> = Vec::new();
    model.serialize(&mut params);

    // One iteration, and two, each on a fresh copy. Their difference is one
    // complete iteration with the setup cancelled out.
    let (first_ms, first) = timed::<M>(&params, &model, &config::<M>(1));
    let first_ms = first_iter_ms(first_ms, first.iterations, first.accepted_iterations);
    let (two_ms, two) = timed::<M>(&params, &model, &config::<M>(2));
    let two_ms = two_iter_ms(two_ms, first_ms, two.accepted_iterations);

    let t0 = std::time::Instant::now();
    let result = M::solve(&params, &mut model, &config::<M>(100));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    model.deserialize(&result.x);

    Row::new(solve_ms, first_ms, result.iterations, model.solution())
        .accepted(result.accepted_iterations)
        .full_ms(two_ms)
}
