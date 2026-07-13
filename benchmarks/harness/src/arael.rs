// The arael pipeline: build the model, probe it, solve it, read the solution.
//
// The model structs cannot be shared -- the arael macro bakes the scalar into
// the generated code, so an f32 model and an f64 model are different types, and
// each benchmark's problem is its own. The pipeline around them is the same for
// all of them, and keeping a copy per model is how one benchmark's f32 config
// lost the verbose flag its f64 twin had.

use ::arael::simple_lm::{LmConfig, LmProblem, LmResult};
use ::arael::utils::Float;

use crate::table::Row;

pub trait Model: Clone + LmProblem<Self::Scalar> {
    type Scalar: Float;
    /// Whatever the benchmark builds its model from: a parsed dataset, a
    /// generated scene.
    type Input;
    type Solution;

    /// Problem-appropriate initial damping, before the ARAEL_LAMBDA0 override.
    /// Solver benchmarks give each algorithm the damping the problem wants, not
    /// the one it ships with; see the benchmark READMEs' damping policy. It sees
    /// the input because the right value can depend on the problem's size.
    fn lambda0(input: &Self::Input) -> f64;

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
pub fn lambda0<M: Model>(input: &M::Input) -> f64 {
    std::env::var("ARAEL_LAMBDA0")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| M::lambda0(input))
}

/// DRIVER=nielsen swaps the fixed damping schedule for the gain-ratio driver:
/// lambda then follows how well the quadratic model predicted the step actually
/// taken, instead of a fixed up/down ladder. It changes the iteration count, not
/// the cost per iteration.
pub fn nielsen() -> bool {
    std::env::var("DRIVER").as_deref() == Ok("nielsen")
}

pub fn config<M: Model>(input: &M::Input, max_iters: usize) -> LmConfig<M::Scalar> {
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
        initial_lambda: f(lambda0::<M>(input)),
        ..Default::default()
    };
    M::tune(&mut cfg);
    if nielsen() {
        cfg.with_driver(::arael::simple_lm::NielsenLambdaDriver::default())
    } else {
        cfg
    }
}

/// TIMING=1 prints arael's internal breakdown for every solve it runs. That
/// includes the probes, so one round of one system emits several lines: the
/// discarded warmup plus PROBE_SUBROUNDS passes of the one- and two-iteration
/// probes, then the real solve.
///
/// Only the printing is gated. `gather_timing` stays on either way, so the
/// timing numbers do not depend on whether they are being looked at.
pub fn print_timing<T>(r: &LmResult<T>) {
    if std::env::var("TIMING").is_err() {
        return;
    }
    if let Some(t) = &r.timing {
        eprintln!(
            "  [timing] total {:.1} ms = assembly {:.1} + linear solve {:.1} \
             (first assembly {:.1}), {} iters",
            t.total.as_secs_f64() * 1e3,
            t.assembly.as_secs_f64() * 1e3,
            t.linear_solve.as_secs_f64() * 1e3,
            t.first_assembly.as_secs_f64() * 1e3,
            r.iterations,
        );
    }
}

/// The probes run on throwaway copies of the model: re-supplying the initial
/// parameter vector does not reset parametrization state held outside it, such
/// as the reference rotation a QuaternionParam re-centres on after a step. A
/// probe that ran on the real model left it half-advanced, and the solve that
/// followed inherited a warm start it never asked for.
pub fn run<M: Model>(input: &M::Input) -> Row<M::Solution> {
    let mut model = M::build(input);
    let mut params: Vec<M::Scalar> = Vec::new();
    model.serialize(&mut params);

    crate::solver::run(100, |max_iters| {
        // The clone and the config are the probe's reset, not the solve: the
        // clock starts below.
        let mut m = model.clone();
        let cfg = config::<M>(input, max_iters);
        let (ms, r) = crate::solver::timed(|| M::solve(&params, &mut m, &cfg));
        print_timing(&r);
        m.deserialize(&r.x);
        crate::solver::Outcome {
            ms,
            accepted: r.accepted_iterations,
            attempts: r.iterations,
            solution: m.solution(),
        }
    })
}
