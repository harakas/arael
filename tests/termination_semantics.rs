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
    let r = c.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() });
    assert!(r.iterations < 200, "should converge before the cap, took {}", r.iterations);
    assert!(r.end_cost < r.start_cost, "cost should decrease: {} -> {}", r.start_cost, r.end_cost);
    assert!(r.accepted_iterations >= 1);
}

#[test]
fn max_iters_is_respected() {
    for k in [1usize, 2, 5, 10] {
        let mut c = build_chain();
        let r = c.solve_sparse(&LmConfig { max_iters: k, min_iters: 0, ..Default::default() });
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
        min_iters: 0, patience: 1, max_iters: 200, ..Default::default() });
    // REL arm alone: abs_precision 0 can never fire.
    let mut cr = build_chain();
    let rr = cr.solve_sparse(&LmConfig {
        abs_precision: 0.0, rel_precision: 1e30,
        min_iters: 0, patience: 1, max_iters: 200, ..Default::default() });
    // Neither loose -> runs to real convergence.
    let mut ct = build_chain();
    let rt = ct.solve_sparse(&LmConfig {
        abs_precision: 1e-14, rel_precision: 1e-14,
        min_iters: 0, patience: 3, max_iters: 200, ..Default::default() });

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
    let r2 = c2.solve_sparse(&cfg(2));
    let mut c8 = build_chain();
    let r8 = c8.solve_sparse(&cfg(8));
    assert!(r8.iterations > r2.iterations,
        "patience 8 ({}) should run longer than patience 2 ({})", r8.iterations, r2.iterations);
}

#[test]
fn status_reports_convergence() {
    // A normal solve stops via the small-step/noise-floor path (R11).
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() });
    assert_eq!(r.status, LmStatus::Converged, "normal solve should report Converged");
    assert!(r.final_lambda.is_finite(), "final_lambda should be finite, got {}", r.final_lambda);
}

#[test]
fn status_reports_max_iterations() {
    // A tight cap stops before convergence -> MaxIterations, not Converged.
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig { max_iters: 2, min_iters: 0, ..Default::default() });
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
    Dense, LambdaDriver, LambdaState, LambdaStep, LmSolver, SolverReport,
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
    );
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
    );
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
    );
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
    ) -> f64 {
        <Dense as LmSolver<f64>>::compute(&mut self.0, problem, params, grad, m)
    }
    fn extract_diagonal(&self, m: &Vec<f64>, d: &mut [f64]) {
        <Dense as LmSolver<f64>>::extract_diagonal(&self.0, m, d)
    }
    fn solve_damped(
        &mut self, _n: usize, _m: &mut Vec<f64>, _d: &[f64],
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
    );
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
    );
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
    });
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
    let unlimited = a.solve_sparse(&cfg);

    let mut b = build_chain();
    let limited = b.solve_sparse(&LmConfig::<f64> {
        time_limit: Some(Duration::from_secs(60)),
        ..cfg.clone()
    });

    assert_eq!(limited.status, unlimited.status);
    assert_eq!(limited.status, LmStatus::Converged);
    assert_eq!(limited.iterations, unlimited.iterations);
    assert_eq!(limited.end_cost, unlimited.end_cost);
}
