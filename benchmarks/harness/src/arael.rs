// The arael pipeline: build the model, probe it, solve it, read the solution.
//
// The model structs cannot be shared -- the arael macro bakes the scalar into
// the generated code, so an f32 model and an f64 model are different types, and
// each benchmark's problem is its own. The pipeline around them is the same for
// all of them, and keeping a copy per model is how one benchmark's f32 config
// lost the verbose flag its f64 twin had.

use ::arael::simple_lm::{LmConfig, LmProblem, LmResult, SolveFailure};
use ::arael::utils::Float;

use crate::table::Row;

pub trait Model: Clone + LmProblem<Self::Scalar> {
    type Scalar: Float;
    /// Whatever the benchmark builds its model from: a parsed dataset, a
    /// generated scene. It also carries whatever else distinguishes one row from
    /// another over the same model -- bundle adjustment runs the same camera/point
    /// model through two different linear solvers, and that choice lives here.
    type Input;
    type Solution;

    /// Problem-appropriate initial damping, before the ARAEL_LAMBDA0 override.
    /// Solver benchmarks give each algorithm the damping the problem wants, not
    /// the one it ships with; see the benchmark READMEs' damping policy. It sees
    /// the input because the right value can depend on the problem's size.
    fn lambda0(input: &Self::Input) -> f64;

    /// Whether this problem wants the gain-ratio damping driver by default. The
    /// pose graphs do not -- they are well-initialized and every step is accepted,
    /// so an adaptive schedule only over-damps. Bundle adjustment does: it is far
    /// less linear, and the fixed ladder walks into damping spirals there. DRIVER
    /// overrides either way.
    const NIELSEN: bool = false;

    fn build(input: &Self::Input) -> Self;
    fn serialize(&mut self, out: &mut Vec<Self::Scalar>);
    fn deserialize(&mut self, x: &[Self::Scalar]);
    fn solution(&self) -> Self::Solution;

    /// The scalar picks the solver (`SparseFaer` / `SparseFaerF32`, or a band
    /// solver), which is the one thing a generic function cannot choose for
    /// itself. It sees the input too, so a benchmark can run one model through
    /// several linear solvers and get a row for each.
    ///
    /// Hand the failure back rather than unwrapping it. A solve CAN fail on a
    /// real dataset -- arael's f32 rows hit a NaN Hessian diagonal on
    /// Ladybug-1723, from observations sitting on the optical centre -- and a
    /// benchmark that panics there loses every other row in the run.
    fn solve(
        input: &Self::Input,
        params: &[Self::Scalar],
        model: &mut Self,
        cfg: &LmConfig<Self::Scalar>,
    ) -> Result<LmResult<Self::Scalar>, SolveFailure<Self::Scalar>>;

    /// Anything else this benchmark wants on the config. The defaults below are
    /// the ones every benchmark has agreed on so far.
    fn tune(_cfg: &mut LmConfig<Self::Scalar>) {}

    /// Whether this row's linear solve is inexact (conjugate gradients). It
    /// reads the input because the route is what distinguishes such a row from
    /// an exact one over the same model. See [`Row::inexact`].
    fn inexact(_input: &Self::Input) -> bool { false }
}

/// ARAEL_LAMBDA0 overrides the model's damping, for experiments.
pub fn lambda0<M: Model>(input: &M::Input) -> f64 {
    std::env::var("ARAEL_LAMBDA0")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| M::lambda0(input))
}

/// Which damping driver this run uses. The gain-ratio (Nielsen) driver follows
/// how well the quadratic model predicted the step actually taken; the fixed
/// ladder walks lambda up and down by a constant factor. It changes the iteration
/// count, not the cost per iteration.
///
/// DRIVER=nielsen|fixed overrides the problem's own default
/// ([`Model::NIELSEN`]). `default` is accepted as a synonym for `fixed`:
/// the fixed ladder is arael's own `DefaultLambdaDriver`, and the headers
/// spell it that way.
pub fn nielsen<M: Model>() -> bool {
    match std::env::var("DRIVER").as_deref() {
        Ok("nielsen") => true,
        Ok("fixed") | Ok("default") => false,
        _ => M::NIELSEN,
    }
}

pub fn config<M: Model>(input: &M::Input, max_iters: usize) -> LmConfig<M::Scalar> {
    // Float: num::NumCast, so a literal reaches either scalar the same way.
    let f = |v: f64| <M::Scalar as num::NumCast>::from(v).unwrap();
    let mut cfg = LmConfig {
        verbose: std::env::var("VERBOSE").is_ok(),
        gather_timing: true,
        // The same count every other system's pool was capped at (BENCH_THREADS,
        // resolved by pin::enforce_cores), so a threaded run stays a fair race.
        // Needs arael built with the `rayon` feature, or it warns and runs
        // sequentially.
        num_threads: crate::pin::threads(),
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
    if nielsen::<M>() {
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
            "  [timing] total {:.1} ms = assembly {:.1} + analysis {:.1} + linear solve {:.1} \
             (first assembly {:.1}), {} iters",
            t.total.as_secs_f64() * 1e3,
            t.assembly.as_secs_f64() * 1e3,
            t.analysis.as_secs_f64() * 1e3,
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
///
/// `Err` carries why the solve failed, for the caller to put in the table.
/// Every probe runs the same solve, so the first failure ends the row: there is
/// no partial row worth reporting, and re-running it would only repeat itself.
pub fn run<M: Model>(input: &M::Input) -> Result<Row<M::Solution>, String> {
    let mut model = M::build(input);
    let mut params: Vec<M::Scalar> = Vec::new();
    model.serialize(&mut params);

    let mut failure: Option<String> = None;
    let row = crate::solver::run(100, |max_iters| {
        // The clone and the config are the probe's reset, not the solve: the
        // clock starts below.
        let mut m = model.clone();
        if failure.is_some() {
            // Already broken. The driver still wants its remaining probes;
            // re-running a solve that fails deterministically only costs time.
            return crate::solver::Outcome {
                ms: f64::INFINITY,
                accepted: 0,
                attempts: 1,
                solution: m.solution(),
            };
        }
        let cfg = config::<M>(input, max_iters);
        // Qualified: `M::solve` alone is ambiguous against `LmProblem::solve`
        // (the SolverKind entry point), which Model also carries.
        let (ms, r) = crate::solver::timed(|| <M as Model>::solve(input, &params, &mut m, &cfg));
        let r = match r {
            Ok(r) => r,
            Err(e) => {
                failure.get_or_insert_with(|| describe(&e));
                // The driver needs an Outcome to carry on with. Nothing reads
                // it: the caller drops the row the moment `failure` is set.
                return crate::solver::Outcome {
                    ms: f64::INFINITY,
                    accepted: 0,
                    attempts: 1,
                    solution: m.solution(),
                };
            }
        };
        print_timing(&r);
        m.deserialize(&r.x);
        crate::solver::Outcome {
            ms,
            accepted: r.accepted_iterations,
            attempts: r.iterations,
            solution: m.solution(),
        }
    });
    match failure {
        Some(why) => Err(why),
        None => Ok(row.inexact(M::inexact(input))),
    }
}

/// One line naming the fault, for a table cell: `SolveFailure`'s Display, which
/// is the short form. Its Debug carries the partial result.
pub fn describe<T>(e: &SolveFailure<T>) -> String {
    e.to_string()
}
