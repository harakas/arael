// Termination-semantics tests for the LM solver (REVIEW2 R1).
//
// A step counts as "small" when the ABSOLUTE cost decrease OR the RELATIVE
// cost decrease is below its threshold (either suffices). After `patience`
// consecutive small steps the solve stops. These tests pin that OR, plus
// max_iters and patience, on a genuinely nonlinear model -- a spring chain.
// (A linear least-squares fit converges in one step and never exercises the
// small-step tail, so it cannot test this.)

use arael::model::{Param, SelfBlock, CrossBlock};
use arael::simple_lm::{LmConfig, LmProblem, LmStatus, RootProblem};
use arael::vect::vect2d;
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, guard = self.is_anchor, {
    [point.pos.x * chain.anchor, point.pos.y * chain.anchor]
}))]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * chain.drift, d.y * chain.drift]
}))]
struct Point {
    pos: Param<vect2d>,
    is_anchor: bool,
    hb: SelfBlock<Point>,
}
#[arael::model]
#[arael(constraint(hb, {
    let d = b.pos - a.pos;
    [(d.norm() - link.rest) * chain.spring]
}))]
struct Link {
    #[arael(ref = root.points)] a: Ref<Point>,
    #[arael(ref = root.points)] b: Ref<Point>,
    rest: f64,
    hb: CrossBlock<Point, Point>,
}
#[arael::model]
#[arael(root)]
struct Chain {
    points: refs::Vec<Point>,
    links: std::vec::Vec<Link>,
    anchor: f64,
    drift: f64,
    spring: f64,
}

const N: usize = 8;

/// Zig-zag start so the springs (rest 1.0) must relax over several LM steps.
fn build_chain() -> Chain {
    let mut c = Chain {
        points: refs::Vec::new(), links: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0,
    };
    for i in 0..N {
        let pos = vect2d::new(i as f64 * 0.5, if i % 2 == 0 { 0.7 } else { -0.7 });
        c.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: SelfBlock::new() });
    }
    for i in 1..N {
        let a = c.points.ref_at(i - 1);
        let b = c.points.ref_at(i);
        c.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
    }
    c
}

#[test]
fn converges_before_max_iters() {
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() }).unwrap();
    assert!(r.iterations < 200, "should converge before the cap, took {}", r.iterations);
    assert!(r.end_cost < r.start_cost, "cost should decrease: {} -> {}", r.start_cost, r.end_cost);
    assert!(r.accepted_iterations >= 1);
}

#[test]
fn max_iters_is_respected() {
    for k in [1usize, 2, 5, 10] {
        let mut c = build_chain();
        let r = c.solve_sparse(&LmConfig { max_iters: k, min_iters: 0, ..Default::default() }).unwrap();
        assert!(r.iterations <= k, "iterations {} exceeded max_iters {}", r.iterations, k);
    }
}

#[test]
fn or_each_criterion_stops_independently() {
    // ABS arm alone: rel_precision 0 can never fire, so only the absolute
    // criterion (here always true, threshold huge) can stop the solve.
    let mut ca = build_chain();
    let ra = ca.solve_sparse(&LmConfig {
        abs_precision: 1e30, rel_precision: 0.0,
        min_iters: 0, patience: 1, max_iters: 200, ..Default::default() }).unwrap();
    // REL arm alone: abs_precision 0 can never fire.
    let mut cr = build_chain();
    let rr = cr.solve_sparse(&LmConfig {
        abs_precision: 0.0, rel_precision: 1e30,
        min_iters: 0, patience: 1, max_iters: 200, ..Default::default() }).unwrap();
    // Neither loose -> runs to real convergence.
    let mut ct = build_chain();
    let rt = ct.solve_sparse(&LmConfig {
        abs_precision: 1e-14, rel_precision: 1e-14,
        min_iters: 0, patience: 3, max_iters: 200, ..Default::default() }).unwrap();

    // Each single loose arm halts the solve well before convergence...
    assert!(ra.iterations < rt.iterations,
        "loose-abs stopped at {} iters, tight took {}", ra.iterations, rt.iterations);
    assert!(rr.iterations < rt.iterations,
        "loose-rel stopped at {} iters, tight took {}", rr.iterations, rt.iterations);
    // ...leaving more cost on the table than the fully-converged solve.
    assert!(ra.end_cost > rt.end_cost, "loose-abs should be less converged");
    assert!(rr.end_cost > rt.end_cost, "loose-rel should be less converged");
}

#[test]
fn patience_controls_stop() {
    // With every step deemed "small" (rel threshold huge), the solve stops
    // after exactly `patience` accepted small steps, so more patience runs
    // longer.
    let cfg = |patience| LmConfig {
        rel_precision: 1e30, abs_precision: 0.0,
        min_iters: 0, patience, max_iters: 200, ..Default::default() };
    let mut c2 = build_chain();
    let r2 = c2.solve_sparse(&cfg(2)).unwrap();
    let mut c8 = build_chain();
    let r8 = c8.solve_sparse(&cfg(8)).unwrap();
    assert!(r8.iterations > r2.iterations,
        "patience 8 ({}) should run longer than patience 2 ({})", r8.iterations, r2.iterations);
}

#[test]
fn status_reports_convergence() {
    // A normal solve stops via the small-step/noise-floor path (R11).
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() }).unwrap();
    assert_eq!(r.status, LmStatus::Converged, "normal solve should report Converged");
    assert!(r.final_lambda.is_finite(), "final_lambda should be finite, got {}", r.final_lambda);
}

#[test]
fn status_reports_max_iterations() {
    // A tight cap stops before convergence -> MaxIterations, not Converged.
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig { max_iters: 2, min_iters: 0, ..Default::default() }).unwrap();
    assert_eq!(r.status, LmStatus::MaxIterations, "capped solve should report MaxIterations");
    assert_eq!(r.iterations, 2);
}

// ---------------------------------------------------------------------------
// Driver-initiated termination. All three LambdaDriver hooks return Option<T>
// now, and `None` means "stop". Which status you get says whether a step
// survived: `accepted` -> DriverTerminated (the step is KEPT), `rejected` and
// `factorization_failed` -> LambdaCeiling (no step was produced).
// ---------------------------------------------------------------------------

use arael::simple_lm::{
    Dense, LambdaDriver, LambdaState, LambdaStep, LmSolver, SolveError, SolverReport,
};
use std::time::Duration;

/// Stops the solve by returning `None` from `accepted` once it has taken
/// `stop_after` good steps. Damping otherwise behaves like the default.
#[derive(Clone)]
struct StopOnAccept {
    stop_after: usize,
    seen: usize,
}

impl LambdaDriver<f64> for StopOnAccept {
    fn start(&mut self, config: &LmConfig<f64>, _s: &LambdaState<f64>) -> f64 {
        config.initial_lambda
    }
    fn accepted(&mut self, step: &LambdaStep<f64>) -> Option<f64> {
        self.seen += 1;
        if self.seen >= self.stop_after {
            return None;
        }
        Some(step.lambda * 0.2)
    }
    fn rejected(&mut self, step: &LambdaStep<f64>) -> Option<f64> {
        Some(step.lambda * 10.0)
    }
    fn factorization_failed(&mut self, lambda: f64, _s: &LambdaState<f64>) -> Option<f64> {
        Some(lambda * 10.0)
    }
}

#[test]
fn driver_can_stop_on_an_accepted_step() {
    let mut c = build_chain();
    let r = c.solve_sparse(
        &LmConfig { max_iters: 200, min_iters: 50, ..Default::default() }
            .with_driver(StopOnAccept { stop_after: 2, seen: 0 }),
    ).unwrap();
    assert_eq!(r.status, LmStatus::DriverTerminated);
    assert_eq!(r.accepted_iterations, 2, "should stop on the 2nd accepted step");
    // The driver's rule beats the config's: min_iters 50 did not hold it open.
    assert!(r.iterations < 50, "min_iters must not override the driver, took {}", r.iterations);
}

#[test]
fn stopping_on_accept_keeps_the_step() {
    // The step that triggered the stop is folded in, not thrown away: the
    // returned parameters must actually produce the returned cost, and that
    // cost must be an improvement.
    let mut c = build_chain();
    let r = c.solve_sparse(
        &LmConfig { max_iters: 200, min_iters: 0, ..Default::default() }
            .with_driver(StopOnAccept { stop_after: 1, seen: 0 }),
    ).unwrap();
    assert_eq!(r.status, LmStatus::DriverTerminated);
    assert_eq!(r.accepted_iterations, 1);
    assert!(
        r.end_cost < r.start_cost,
        "the accepted step must be kept: {} -> {}", r.start_cost, r.end_cost
    );

    // The step is not just reported -- it is IN the returned parameters. If it
    // had been discarded, x would still be the starting point.
    let mut fresh = build_chain();
    let mut x0 = std::vec::Vec::new();
    fresh.serialize(&mut x0);
    assert_eq!(x0.len(), r.x.len());
    assert!(
        core::iter::zip(&x0, &r.x).any(|(a, b)| a != b),
        "x came back unchanged, so the accepted step was thrown away"
    );

    // And stopping one step later must land strictly lower: each accepted step
    // is folded in, so the next one starts from it rather than from scratch.
    let mut c2 = build_chain();
    let r2 = c2.solve_sparse(
        &LmConfig { max_iters: 200, min_iters: 0, ..Default::default() }
            .with_driver(StopOnAccept { stop_after: 2, seen: 0 }),
    ).unwrap();
    assert_eq!(r2.accepted_iterations, 2);
    assert!(
        r2.end_cost < r.end_cost,
        "2 kept steps should beat 1: {} vs {}", r2.end_cost, r.end_cost
    );

    // NOTE: do NOT assert calc_cost(&r.x) == r.end_cost here. This model has a
    // drift term (pos - pos_value), and deserializing after the solve re-anchors
    // pos_value to the solution -- so calc_cost on the returned model is not the
    // same function that was minimized. That is `_value` semantics working as
    // designed, not a solver bug.
}

/// A backend that assembles normally but always reports the damped system as
/// not positive definite, so every attempt lands in `factorization_failed`.
/// Nothing else can drive that hook deterministically -- H = 2 J^T J is PSD and
/// damping only makes it more so.
struct NeverFactorizes(Dense);

// Dense implements LmSolver for f64 AND f32, so every delegation has to name
// which one it means.
impl LmSolver<f64> for NeverFactorizes {
    type Matrix = Vec<f64>;
    fn new_matrix(&self, n: usize) -> Vec<f64> {
        <Dense as LmSolver<f64>>::new_matrix(&self.0, n)
    }
    fn compute(
        &mut self, problem: &mut dyn LmProblem<f64>, params: &[f64],
        grad: &mut [f64], m: &mut Vec<f64>,
    ) -> Result<f64, SolveError> {
        <Dense as LmSolver<f64>>::compute(&mut self.0, problem, params, grad, m)
    }
    fn extract_diagonal(&self, m: &Vec<f64>, d: &mut [f64]) {
        <Dense as LmSolver<f64>>::extract_diagonal(&self.0, m, d)
    }
    fn solve_damped(
        &mut self, _n: usize, _m: &mut Vec<f64>, _d: &[f64], _damp: &[f64],
        _lambda: f64, _g: &[f64], _delta: &mut [f64],
    ) -> bool {
        false // "not positive definite", every time
    }
    fn matrix_nonfinite_count(&self, m: &Vec<f64>) -> usize {
        <Dense as LmSolver<f64>>::matrix_nonfinite_count(&self.0, m)
    }
    fn reset(&mut self) {
        <Dense as LmSolver<f64>>::reset(&mut self.0)
    }
    fn report(&self) -> Option<SolverReport> {
        None
    }
}

/// Returns `None` from `factorization_failed` after `tolerate` failures.
#[derive(Clone)]
struct GiveUpOnFactorization {
    tolerate: usize,
    seen: usize,
}

impl LambdaDriver<f64> for GiveUpOnFactorization {
    fn start(&mut self, config: &LmConfig<f64>, _s: &LambdaState<f64>) -> f64 {
        config.initial_lambda
    }
    fn accepted(&mut self, step: &LambdaStep<f64>) -> Option<f64> {
        Some(step.lambda * 0.2)
    }
    fn rejected(&mut self, step: &LambdaStep<f64>) -> Option<f64> {
        Some(step.lambda * 10.0)
    }
    fn factorization_failed(&mut self, lambda: f64, _s: &LambdaState<f64>) -> Option<f64> {
        self.seen += 1;
        if self.seen >= self.tolerate {
            return None; // no damping left worth trying
        }
        Some(lambda * 10.0)
    }
}

#[test]
fn driver_can_stop_on_a_factorization_failure() {
    let mut c = build_chain();
    let r = c.solve_with(
        &mut NeverFactorizes(Dense),
        &LmConfig { max_iters: 200, min_iters: 0, ..Default::default() }
            .with_driver(GiveUpOnFactorization { tolerate: 3, seen: 0 }),
    ).unwrap();
    // No step was ever produced, so the solve reports the same exhaustion
    // status as a driver that gives up on a rejection.
    assert_eq!(r.status, LmStatus::LambdaCeiling);
    assert_eq!(r.accepted_iterations, 0);
    assert_eq!(r.iterations, 3, "should stop on the 3rd failed factorization");
    assert_eq!(r.end_cost, r.start_cost, "no step was taken, so the cost cannot move");
}

#[test]
fn tolerating_factorization_failures_hits_the_retry_budget_instead() {
    // The contrast: a driver that never gives up burns the hard 20-retry cap
    // and reports RetryBudgetExhausted, not LambdaCeiling.
    let mut c = build_chain();
    let r = c.solve_with(
        &mut NeverFactorizes(Dense),
        &LmConfig { max_iters: 200, min_iters: 0, ..Default::default() }
            .with_driver(GiveUpOnFactorization { tolerate: usize::MAX, seen: 0 }),
    ).unwrap();
    assert_eq!(r.status, LmStatus::RetryBudgetExhausted);
    assert_eq!(r.accepted_iterations, 0);
}

// ---------------------------------------------------------------------------
// LmConfig::time_limit -- a hard wall-clock budget.
// ---------------------------------------------------------------------------

#[test]
fn a_spent_budget_stops_the_solve_and_overrides_min_iters() {
    // Duration::ZERO is always already spent, so this pins the mechanism with
    // no dependence on how fast the machine is. min_iters is set high on
    // purpose: a budget is a budget.
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig::<f64> {
        time_limit: Some(Duration::ZERO),
        max_iters: 200,
        min_iters: 50,
        ..Default::default()
    }).unwrap();
    assert_eq!(r.status, LmStatus::TimeLimit);
    assert_eq!(r.accepted_iterations, 0, "no time to take a step");
    assert_eq!(r.end_cost, r.start_cost, "no step taken, so the cost cannot move");
    // The first assembly still ran, so start_cost is a real number, not zero.
    assert!(r.start_cost > 0.0, "start_cost should come from the first assembly");
}

#[test]
fn a_generous_budget_does_not_change_the_answer() {
    // The limit must be inert when it cannot bind -- same status, same
    // iteration count, same cost as a solve with no limit at all.
    let cfg = LmConfig::<f64> { max_iters: 200, ..Default::default() };

    let mut a = build_chain();
    let unlimited = a.solve_sparse(&cfg).unwrap();

    let mut b = build_chain();
    let limited = b.solve_sparse(&LmConfig::<f64> {
        time_limit: Some(Duration::from_secs(60)),
        ..cfg.clone()
    }).unwrap();

    assert_eq!(limited.status, unlimited.status);
    assert_eq!(limited.status, LmStatus::Converged);
    assert_eq!(limited.iterations, unlimited.iterations);
    assert_eq!(limited.end_cost, unlimited.end_cost);
}

// ---------------------------------------------------------------------------
// LmConfig::gradient_tolerance, ::parameter_tolerance and
// ::predicted_reduction_tolerance -- all Option, all off by default. They ask
// different questions from the cost test: "am I at a stationary point", "have I
// stopped moving", and "does the model expect anything more", not "has the cost
// stopped improving".
// ---------------------------------------------------------------------------

#[test]
fn tolerances_are_off_by_default() {
    // Absent unless asked for: a default solve must be bit-identical to one
    // that names them as None.
    let mut a = build_chain();
    let base = a.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() }).unwrap();

    let mut b = build_chain();
    let explicit = b.solve_sparse(&LmConfig::<f64> {
        max_iters: 200,
        gradient_tolerance: None,
        parameter_tolerance: None,
        predicted_reduction_tolerance: None,
        ..Default::default()
    }).unwrap();
    assert_eq!(base.status, explicit.status);
    assert_eq!(base.iterations, explicit.iterations);
    assert_eq!(base.end_cost, explicit.end_cost);
    assert_eq!(base.status, LmStatus::Converged);
}

#[test]
fn gradient_tolerance_stops_the_solve() {
    // A tolerance so loose the gradient is "flat" on the first look: the solve
    // stops on it rather than on the cost, and says so.
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig::<f64> {
        gradient_tolerance: Some(1e30),
        min_iters: 0, max_iters: 200, ..Default::default()
    }).unwrap();
    assert_eq!(r.status, LmStatus::GradientTolerance);

    // And a tolerance so tight it can never fire leaves the solve alone.
    let mut c = build_chain();
    let tight = c.solve_sparse(&LmConfig::<f64> {
        gradient_tolerance: Some(0.0),
        min_iters: 0, max_iters: 200, ..Default::default()
    }).unwrap();
    assert_eq!(tight.status, LmStatus::Converged, "an impossible tolerance must not fire");
}

#[test]
fn gradient_tolerance_is_a_different_question_from_the_cost_test() {
    // Turn the cost test off entirely (thresholds it can never meet) so the
    // ONLY thing that can stop this solve short of max_iters is the gradient.
    let cfg = |gtol| LmConfig::<f64> {
        abs_precision: 0.0, rel_precision: 0.0,   // cost test can never fire
        gradient_tolerance: gtol,
        min_iters: 0, max_iters: 200, ..Default::default()
    };
    let mut a = build_chain();
    let without = a.solve_sparse(&cfg(None)).unwrap();
    let mut b = build_chain();
    let with = b.solve_sparse(&cfg(Some(1e-4))).unwrap();

    assert_ne!(without.status, LmStatus::GradientTolerance);
    assert_eq!(with.status, LmStatus::GradientTolerance);
    assert!(
        with.iterations < without.iterations,
        "the gradient test should stop it earlier: {} vs {}",
        with.iterations, without.iterations
    );
    // It stopped because the gradient was flat, so it is genuinely converged --
    // not stopped early at a worse cost.
    assert!(
        with.end_cost <= without.end_cost * 1.01,
        "gradient-stopped cost {} should be no worse than {}",
        with.end_cost, without.end_cost
    );
}

#[test]
fn parameter_tolerance_stops_the_solve() {
    // Loose enough that the first accepted step is already "negligible".
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig::<f64> {
        parameter_tolerance: Some(1e30),
        min_iters: 0, max_iters: 200, ..Default::default()
    }).unwrap();
    assert_eq!(r.status, LmStatus::ParameterTolerance);
    assert_eq!(r.accepted_iterations, 1, "should stop on the first accepted step");
    // The step is kept: it was an improvement, just a small one in x.
    assert!(r.end_cost < r.start_cost);

    // Zero can never be met (|step| <= 0 * (|x| + 0) = 0), so it must not fire.
    let mut c = build_chain();
    let tight = c.solve_sparse(&LmConfig::<f64> {
        parameter_tolerance: Some(0.0),
        min_iters: 0, max_iters: 200, ..Default::default()
    }).unwrap();
    assert_eq!(tight.status, LmStatus::Converged, "an impossible tolerance must not fire");
}

#[test]
fn parameter_tolerance_is_a_different_question_from_the_cost_test() {
    // Cost test disabled; only the step-norm test can stop this early.
    let cfg = |ptol| LmConfig::<f64> {
        abs_precision: 0.0, rel_precision: 0.0,
        parameter_tolerance: ptol,
        min_iters: 0, max_iters: 200, ..Default::default()
    };
    let mut a = build_chain();
    let without = a.solve_sparse(&cfg(None)).unwrap();
    let mut b = build_chain();
    let with = b.solve_sparse(&cfg(Some(1e-6))).unwrap();

    assert_ne!(without.status, LmStatus::ParameterTolerance);
    assert_eq!(with.status, LmStatus::ParameterTolerance);
    assert!(
        with.iterations < without.iterations,
        "the step test should stop it earlier: {} vs {}",
        with.iterations, without.iterations
    );
}

#[test]
fn predicted_reduction_stops_the_solve() {
    // A tolerance so loose the model's predicted gain is "negligible" on the
    // first accepted step: the solve stops on it and keeps that step.
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig::<f64> {
        predicted_reduction_tolerance: Some(1e30),
        min_iters: 0, max_iters: 200, ..Default::default()
    }).unwrap();
    assert_eq!(r.status, LmStatus::PredictedReduction);
    assert_eq!(r.accepted_iterations, 1, "should stop on the first accepted step");
    assert!(r.end_cost < r.start_cost, "the triggering step is kept");

    // Zero can never be met: a valid accepted step has a strictly positive
    // predicted reduction, so `predicted <= 0` never fires.
    let mut c = build_chain();
    let tight = c.solve_sparse(&LmConfig::<f64> {
        predicted_reduction_tolerance: Some(0.0),
        min_iters: 0, max_iters: 200, ..Default::default()
    }).unwrap();
    assert_eq!(tight.status, LmStatus::Converged, "an impossible tolerance must not fire");
}

#[test]
fn predicted_reduction_is_a_forward_looking_test() {
    // Cost test disabled; only the predicted-reduction test can stop this early.
    // It asks what the model expects to gain NEXT, not what the last step gained.
    let cfg = |rtol| LmConfig::<f64> {
        abs_precision: 0.0, rel_precision: 0.0,
        predicted_reduction_tolerance: rtol,
        min_iters: 0, max_iters: 200, ..Default::default()
    };
    let mut a = build_chain();
    let without = a.solve_sparse(&cfg(None)).unwrap();
    let mut b = build_chain();
    let with = b.solve_sparse(&cfg(Some(1e-8))).unwrap();

    assert_ne!(without.status, LmStatus::PredictedReduction);
    assert_eq!(with.status, LmStatus::PredictedReduction);
    assert!(
        with.iterations < without.iterations,
        "the predicted test should stop it earlier: {} vs {}",
        with.iterations, without.iterations
    );
    // It stopped because the model saw nothing worth taking, so it is genuinely
    // converged -- not stopped early at a worse cost.
    assert!(
        with.end_cost <= without.end_cost * 1.01,
        "predicted-stopped cost {} should be no worse than {}",
        with.end_cost, without.end_cost
    );
}

#[test]
fn both_tolerances_respect_min_iters() {
    // They are convergence criteria, so min_iters holds them open like the
    // others -- unlike time_limit, which is a budget and overrides it.
    for cfg in [
        LmConfig::<f64> {
            gradient_tolerance: Some(1e30), min_iters: 6, max_iters: 200,
            ..Default::default()
        },
        LmConfig::<f64> {
            parameter_tolerance: Some(1e30), min_iters: 6, max_iters: 200,
            ..Default::default()
        },
        LmConfig::<f64> {
            predicted_reduction_tolerance: Some(1e30), min_iters: 6, max_iters: 200,
            ..Default::default()
        },
    ] {
        let mut c = build_chain();
        let r = c.solve_sparse(&cfg).unwrap();
        assert!(
            r.iterations >= 6,
            "min_iters 6 not honoured, stopped at {} with {:?}", r.iterations, r.status
        );
    }
}

// ---------------------------------------------------------------------------
// LmTiming::steps -- the per-iteration timeline (C15), and arael::log (C16).
// ---------------------------------------------------------------------------

#[test]
fn steps_are_empty_unless_timing_was_asked_for() {
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() }).unwrap();
    assert!(r.timing.is_none(), "no timing unless gather_timing");
}

#[test]
fn one_step_record_per_attempt_including_damping_retries() {
    // A damping retry is an iteration here, so the timeline has one record per
    // ATTEMPT -- rejects and failed factorizations included.
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig {
        max_iters: 200, gather_timing: true, ..Default::default()
    }).unwrap();
    let t = r.timing.as_ref().expect("gather_timing was set");

    assert_eq!(
        t.steps.len(), r.iterations,
        "one record per attempt: {} records vs {} iterations",
        t.steps.len(), r.iterations
    );
    assert_eq!(
        t.steps.iter().filter(|s| s.accepted).count(), r.accepted_iterations,
        "accepted records must match accepted_iterations"
    );

    // iter is 1-based and dense; inner restarts at 0 on each new linearization.
    for (k, s) in t.steps.iter().enumerate() {
        assert_eq!(s.iter, k + 1, "iter should be 1-based and contiguous");
        assert!(s.lambda > 0.0, "lambda should be live, got {}", s.lambda);
        assert!(s.grad_max.is_finite());
        if s.factorization_failed {
            assert!(s.new_cost.is_nan(), "a failed factorization has no trial point");
            assert_eq!(s.step_norm, 0.0);
        } else {
            assert!(s.step_norm >= 0.0);
        }
    }
    // An accepted step is exactly one that lowered the cost.
    for s in t.steps.iter().filter(|s| !s.factorization_failed) {
        assert_eq!(s.accepted, s.new_cost < s.cost, "accepted iff the cost fell");
    }
    // The first attempt of a linearization has inner 0; a retry has inner > 0.
    assert_eq!(t.steps[0].inner, 0);
}

#[test]
fn each_step_carries_its_own_phase_breakdown() {
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig {
        max_iters: 200, gather_timing: true, ..Default::default()
    }).unwrap();
    let t = r.timing.as_ref().unwrap();

    for s in &t.steps {
        // Assembly is charged to the attempt that caused it and to no other. A
        // retry re-damps and re-factorizes; it does NOT re-assemble, and that
        // asymmetry is why a rejected step is cheaper than an accepted one.
        if s.inner > 0 {
            assert_eq!(
                s.assembly, std::time::Duration::ZERO,
                "retry (iter {}, inner {}) must not be charged for an assembly",
                s.iter, s.inner
            );
        }
        // Re-centering only happens on a step that was kept.
        if !s.accepted {
            assert_eq!(s.advance, std::time::Duration::ZERO);
        }
        // A failed factorization has no trial point to evaluate.
        if s.factorization_failed {
            assert_eq!(s.cost_eval, std::time::Duration::ZERO);
        }
        // The phases inside the attempt cannot exceed the attempt. (assembly is
        // excluded from `time` by construction -- it precedes the attempt.)
        assert!(
            s.linear_solve + s.cost_eval + s.advance <= s.time,
            "phases {:?}+{:?}+{:?} exceed the attempt's {:?}",
            s.linear_solve, s.cost_eval, s.advance, s.time
        );
    }

    // The per-step phases must add up to the aggregate totals the solver keeps
    // independently -- if they disagree, one of the two is lying.
    let sum = |f: fn(&arael::simple_lm::LmStep) -> std::time::Duration| {
        t.steps.iter().map(f).sum::<std::time::Duration>()
    };
    assert_eq!(sum(|s| s.assembly), t.assembly, "assembly");
    assert_eq!(sum(|s| s.analysis), t.analysis, "analysis");
    assert_eq!(sum(|s| s.linear_solve), t.linear_solve, "linear_solve");
    assert_eq!(sum(|s| s.cost_eval), t.cost_eval, "cost_eval");
    assert_eq!(sum(|s| s.advance), t.advance, "advance");

    // ...and the counts too. Count by what HAPPENED, not by a non-zero duration:
    // advance is a no-op on this model (no rotation params to re-center), so it
    // genuinely measures zero, and a zero duration does not mean it did not run.
    assert_eq!(t.steps.iter().filter(|s| s.inner == 0).count(), t.assembly_count);
    assert_eq!(t.steps.iter().filter(|s| s.accepted).count(), t.advance_count);
    assert_eq!(t.steps.iter().filter(|s| !s.factorization_failed).count(), t.cost_eval_count);
    assert_eq!(t.steps.len(), t.linear_solve_count);
}

#[test]
fn the_timeline_shows_the_cost_coming_down() {
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig {
        max_iters: 200, gather_timing: true, ..Default::default()
    }).unwrap();
    let t = r.timing.unwrap();
    let accepted: std::vec::Vec<_> = t.steps.iter().filter(|s| s.accepted).collect();
    assert!(accepted.len() >= 2);
    // Each accepted step starts where the previous one ended.
    for w in accepted.windows(2) {
        assert!(
            w[1].cost <= w[0].new_cost + 1e-12,
            "accepted steps should chain: {} -> {} then start at {}",
            w[0].cost, w[0].new_cost, w[1].cost
        );
    }
    assert_eq!(accepted[0].cost, r.start_cost);
    assert_eq!(accepted.last().unwrap().new_cost, r.end_cost);
}

#[test]
fn logging_can_be_piped_and_silenced() {
    use arael::log::{self, Level};
    use std::sync::{Arc, Mutex};

    // Pipe: a verbose solve's trace lands in our buffer, not on stderr.
    let seen: Arc<Mutex<std::vec::Vec<(Level, String)>>> = Arc::new(Mutex::new(std::vec::Vec::new()));
    let sink = Arc::clone(&seen);
    log::set_sink(move |level, msg| sink.lock().unwrap().push((level, msg.to_string())));

    let mut c = build_chain();
    let _ = c.solve_sparse(&LmConfig { max_iters: 200, verbose: true, ..Default::default() }).unwrap();

    // Tests in this binary run concurrently and the sink is global, so assert on
    // what MUST be there rather than on the buffer being pure -- a stray warn!
    // from another test is not this test's business.
    let info_lines = seen.lock().unwrap().iter().filter(|(l, _)| *l == Level::Info).count();
    assert!(info_lines > 0, "the verbose trace should have reached the sink");

    // Silence: nothing more arrives, and the level check happens before the
    // message is even formatted.
    seen.lock().unwrap().clear();
    log::silence();
    assert_eq!(log::level(), Level::Off);
    assert!(!log::enabled(Level::Error), "Off must drop even errors");

    let mut c = build_chain();
    let _ = c.solve_sparse(&LmConfig { max_iters: 200, verbose: true, ..Default::default() }).unwrap();
    assert_eq!(seen.lock().unwrap().len(), 0, "a silenced arael emits nothing");

    // Put the world back for any other test in this binary.
    log::set_level(Level::Info);
    log::reset_sink();
    assert_eq!(log::level(), Level::Info);
}

// ---------------------------------------------------------------------------
// LmResult::report / print / pretty_report / pretty_print.
// ---------------------------------------------------------------------------

use arael::simple_lm::{LmResult, LmStep, LmTiming, Style};

/// A result with one of each kind of attempt, so the timeline has all three
/// markers. Built by hand -- the spring chain is too well behaved to reject a
/// step on demand.
fn synthetic_result() -> LmResult<f64> {
    let step = |iter, inner, accepted, failed| LmStep {
        iter,
        inner,
        accepted,
        factorization_failed: failed,
        lambda: 1e-4,
        cost: 100.0,
        new_cost: if failed { f64::NAN } else { 90.0 },
        step_norm: 0.5,
        grad_max: 12.0,
        time: Duration::from_micros(300),
        assembly: if inner == 0 { Duration::from_micros(120) } else { Duration::ZERO },
        analysis: if iter == 1 { Duration::from_micros(200) } else { Duration::ZERO },
        linear_solve: Duration::from_micros(150),
        cost_eval: if failed { Duration::ZERO } else { Duration::from_micros(20) },
        advance: if accepted { Duration::from_micros(10) } else { Duration::ZERO },
    };
    LmResult {
        x: std::vec![1.0, 2.0],
        start_cost: 100.0,
        end_cost: 2.5,
        iterations: 3,
        accepted_iterations: 1,
        status: LmStatus::Converged,
        final_lambda: 1e-6,
        solver: None,
        timing: Some(LmTiming {
            total: Duration::from_micros(900),
            assembly: Duration::from_micros(120),
            first_assembly: Duration::from_micros(120),
            analysis: Duration::from_micros(200),
            linear_solve: Duration::from_micros(450),
            first_linear_solve: Duration::from_micros(150),
            cost_eval: Duration::from_micros(40),
            first_cost_eval: Duration::from_micros(20),
            advance: Duration::from_micros(10),
            first_advance: Duration::from_micros(10),
            assembly_count: 1,
            analysis_count: 1,
            linear_solve_count: 3,
            cost_eval_count: 2,
            advance_count: 1,
            steps: std::vec![
                step(1, 0, false, true),   // factorization failed
                step(2, 1, false, false),  // rejected
                step(3, 2, true, false),   // accepted
            ],
        }),
    }
}

#[test]
fn report_is_ascii_and_has_no_escapes() {
    let r = synthetic_result();
    let s = r.report();

    // print() is the ASCII version: it must be safe to put in a log or a file.
    assert!(s.is_ascii(), "report() must be pure ASCII:\n{s}");
    assert!(!s.contains('\x1b'), "report() must carry no ANSI escapes:\n{s}");

    assert!(s.contains("converged"));
    assert!(s.contains("3 iterations"));
    assert!(s.contains("1 accepted"));
    assert!(s.contains("2 retried"), "iterations - accepted = retried");
    // The three ASCII markers, one per attempt, in order: failed, rejected, ok.
    assert!(s.contains("x-+"), "the timeline should read x-+ :\n{s}");
    assert!(s.contains("97.50% down"), "cost 100 -> 2.5:\n{s}");
}

#[test]
fn pretty_report_carries_colour_and_glyphs() {
    let r = synthetic_result();
    let s = r.pretty_report();

    assert!(!s.is_ascii(), "pretty_report() is where the glyphs live");
    assert!(s.contains('\x1b'), "pretty_report() should be coloured");
    assert!(s.contains('\u{2713}'), "check mark for an accepted step");
    assert!(s.contains('\u{2717}'), "ballot X for a rejected one");
    assert!(s.contains('\u{2298}'), "circled slash for a failed factorization");
    // Failed, rejected, accepted -- in that order, once the colour is stripped.
    assert!(strip_ansi(&s).contains("\u{2298}\u{2717}\u{2713}"), "the timeline, in order");

    // Same facts, different clothes.
    assert!(s.contains("3 iterations"));
    assert!(s.contains("97.50% down"));
}

#[test]
fn render_takes_an_explicit_style() {
    let r = synthetic_result();
    assert_eq!(r.render(Style::PLAIN), r.report());
    assert_eq!(r.render(Style::PRETTY), r.pretty_report());

    // Colour without glyphs, for a terminal that cannot draw them.
    let s = r.render(Style { colour: true, unicode: false });
    assert!(s.contains('\x1b'), "coloured");
    assert!(s.is_ascii(), "but no glyphs -- ANSI escapes are themselves ASCII");
    // Each marker is wrapped in its own escape, so they are only adjacent once
    // the colour is stripped back out.
    assert!(strip_ansi(&s).contains("x-+"), "still the ASCII markers:\n{s}");
}

/// Drop ANSI colour sequences, leaving the text.
fn strip_ansi(s: &str) -> String {
    let mut out = String::new();
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c in chars.by_ref() {
                if c == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[test]
fn display_is_the_plain_report() {
    let r = synthetic_result();
    assert_eq!(format!("{r}"), r.report());
}

#[test]
fn a_real_solve_reports_itself() {
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig {
        max_iters: 200, gather_timing: true, ..Default::default()
    }).unwrap();
    let s = r.report();
    assert!(s.is_ascii());
    assert!(s.contains("converged"));
    assert!(s.contains("assembly"));
    assert!(s.contains("linear solve"));

    // Without gather_timing there is no timing block, and the report still works.
    let mut c = build_chain();
    let bare = c.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() }).unwrap();
    let s = bare.report();
    assert!(s.contains("converged"));
    assert!(!s.contains("assembly"), "no timing was gathered, so none is reported");
}

#[test]
fn an_aborted_partial_reports_aborted() {
    // The Aborted status appears only inside SolveFailure::partial; its
    // report labels it and it is never a success.
    let mut r = synthetic_result();
    r.status = LmStatus::Aborted;
    let s = r.report();
    assert!(s.contains("aborted"), "the report must label it:\n{s}");
    assert!(!r.status.is_success());
}

// ---------------------------------------------------------------------------
// LmTiming::analysis -- the backend's one-time structural work, reported apart
// from the model assembly it used to hide inside.
// ---------------------------------------------------------------------------

#[test]
fn the_structural_analysis_is_reported_apart_from_the_assembly() {
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig {
        max_iters: 200, gather_timing: true, ..Default::default()
    }).unwrap();
    let t = r.timing.as_ref().unwrap();

    // The sparse backend discovers the pattern, decides the Schur reduction,
    // picks an ordering and factorizes symbolically -- all inside the first
    // compute(). It is one-time.
    assert_eq!(t.analysis_count, 1, "the analysis runs once, not once per iteration");
    assert!(t.analysis > std::time::Duration::ZERO, "and it costs something");

    // It is charged to the attempt that caused it, and to no other.
    let with_analysis: std::vec::Vec<_> =
        t.steps.iter().filter(|s| s.analysis > std::time::Duration::ZERO).collect();
    assert_eq!(with_analysis.len(), 1);
    assert_eq!(with_analysis[0].iter, 1, "it happens on the very first attempt");
    assert_eq!(t.steps.iter().map(|s| s.analysis).sum::<std::time::Duration>(), t.analysis);

    // Disjoint from assembly: `assembly` is the model's residual and Jacobian
    // work, on every iteration including the first. If the analysis leaked into
    // it, the first assembly would tower over a steady one in every run. One
    // run cannot tell that from a cold start (fresh code and buffers on a
    // loaded machine), so the smallest first assembly over several fresh
    // solves is what stands against the steady ones.
    let mut first_min = std::time::Duration::MAX;
    let mut typical_max = std::time::Duration::ZERO;
    for _ in 0..5 {
        let mut c = build_chain();
        let r = c.solve_sparse(&LmConfig {
            max_iters: 200, gather_timing: true, ..Default::default()
        }).unwrap();
        let t = r.timing.as_ref().unwrap();
        let steady = t.steps.iter().filter(|s| s.inner == 0 && s.iter > 1);
        let typical = steady.map(|s| s.assembly).max().unwrap();
        first_min = first_min.min(t.first_assembly);
        typical_max = typical_max.max(typical);
    }
    assert!(
        first_min <= typical_max * 4,
        "the smallest first_assembly {:?} should be the same order as a steady one {:?} -- \
         if it is not, the analysis is still being charged to it",
        first_min, typical_max
    );
}

#[test]
fn a_backend_that_does_no_analysis_reports_none() {
    // Dense has no pattern to discover and no ordering to choose. Its whole
    // compute() is assembly, so the analysis phase must be empty rather than
    // picking up the trait call's own overhead on every iteration.
    let mut c = build_chain();
    let r = c.solve_dense(&LmConfig {
        max_iters: 200, gather_timing: true, ..Default::default()
    }).unwrap();
    let t = r.timing.as_ref().unwrap();
    assert_eq!(t.analysis_count, 0);
    assert_eq!(t.analysis, std::time::Duration::ZERO);
    assert!(t.steps.iter().all(|s| s.analysis == std::time::Duration::ZERO));
    // ...and assembly still accounts for every outer iteration.
    assert_eq!(t.assembly_count, t.steps.iter().filter(|s| s.inner == 0).count());
}

// A solve that starts at or below cost_threshold has nothing to improve, and
// must say so before the first attempt. The in-loop threshold test only runs
// on an ACCEPTED step, which such a solve never produces: every step is a
// rejection, so it used to run until the damping ladder gave up -- eleven
// factorizations to discover it was already finished.
#[test]
fn a_solve_that_starts_below_the_threshold_stops_before_iterating() {
    let mut c = build_chain();
    // Converge properly first, so the next solve starts at the optimum.
    c.solve_dense(&LmConfig { max_iters: 500, ..Default::default() }).unwrap();
    let settled = c.solve_dense(&LmConfig { max_iters: 500, ..Default::default() }).unwrap();
    assert!(settled.end_cost < 1e-6, "did not settle: {}", settled.end_cost);

    // Ask again with a threshold the starting cost already meets.
    let r = c.solve_dense(&LmConfig {
        max_iters: 500,
        min_iters: 8,
        cost_threshold: settled.end_cost * 10.0 + 1e-12,
        ..Default::default()
    }).unwrap();

    assert_eq!(r.iterations, 0, "iterated {} times on an already-met target", r.iterations);
    assert_eq!(r.accepted_iterations, 0);
    assert!(matches!(r.status, LmStatus::CostThreshold), "status {:?}", r.status);
    assert_eq!(r.start_cost, r.end_cost);
}
