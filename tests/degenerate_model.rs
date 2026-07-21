// Degenerate models -- a parameter with no Hessian diagonal entry -- must
// fail fast, not corrupt data.
//
// Structural case: a CSC column with no stored diagonal. Left unchecked,
// diag_pos[j] = 0 makes extract_diagonal/solve_damped read and overwrite
// vals[0] (an unrelated entry), surfacing as a "Cholesky failed" far away.
// to_csc / to_csc_with_map reject it with SolveError::UnconstrainedParameter,
// which a solve surfaces as LmStatus::SetupFailed.
//
// Value case: a structurally present but zero diagonal cannot be rescued by
// multiplicative damping ((1+lambda)*0 stays 0). The solver terminates the
// solve immediately with LmStatus::DegenerateDiagonal (a runtime value
// condition -- guards or saturated robustifiers can produce it from data).

use arael::simple_lm::{self, BandError, CooMatrix, CscMatrix, FnProblem, LmConfig, LmStatus, SolveError};
use arael::simple_lm::LmProblem;

// The bad-diagonal diagnostic goes through arael's process-global log sink.
// a_nan_diagonal_is_not_reported_as_a_zero installs a sink and reads it back,
// so the tests that emit that diagnostic must not run at the same time. This
// serializes them; unrelated tests stay parallel.
static SINK_GUARD: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn missing_diagonal_rejected_in_to_csc() {
    let mut coo = CooMatrix::new(2);
    coo.push(0, 0, 2.0);
    coo.push(0, 1, 1.0); // column 1: off-diagonal only, no (1,1)
    assert_eq!(coo.to_csc().err(), Some(SolveError::UnconstrainedParameter { param: 1 }));
}

#[test]
fn missing_diagonal_rejected_in_to_csc_with_map() {
    let mut coo = CooMatrix::new(2);
    coo.push(0, 0, 2.0);
    coo.push(0, 1, 1.0);
    assert_eq!(
        coo.to_csc_with_map().err(),
        Some(SolveError::UnconstrainedParameter { param: 1 })
    );
}

#[test]
fn zero_diagonal_terminates_immediately() {
    let _g = SINK_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    // Parameter 1 is never touched: gradient and Hessian row stay zero.
    let mut p = FnProblem {
        cost: |x: &[f64]| (x[0] - 1.0).powi(2),
        grad_hessian: |x: &[f64], g: &mut [f64], h: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 1.0);
            g[1] = 0.0;
            h[0] = 2.0; h[1] = 0.0; h[2] = 0.0; h[3] = 0.0;
            (x[0] - 1.0).powi(2)
        },
    };
    let result = simple_lm::solve(&[0.0, 0.0], &mut p, &LmConfig::default());
    // Terminates before spending any iterations (the old behavior burned
    // 20 inner failures per outer iteration until max_iters), returning
    // the starting point untouched.
    assert_eq!(result.iterations, 0, "must terminate before iterating");
    assert_eq!(result.x, vec![0.0, 0.0], "params must be untouched");
    assert_eq!(result.end_cost, result.start_cost);
    assert_eq!(result.start_cost, 1.0);
}

// LmConfig::min_diagonal floors the DAMPING scale, so a parameter with no
// curvature gets lambda * min_diagonal instead of (1 + lambda) * 0 = 0. The
// factorization then succeeds and that parameter simply does not move.

/// Parameter 1 is untouched: zero gradient, zero Hessian row. Parameter 0 is a
/// plain quadratic with its minimum at 1.0.
fn zero_diagonal_problem() -> FnProblem<
    impl Fn(&[f64]) -> f64,
    impl Fn(&[f64], &mut [f64], &mut [f64]) -> f64,
> {
    FnProblem {
        cost: |x: &[f64]| (x[0] - 1.0).powi(2),
        grad_hessian: |x: &[f64], g: &mut [f64], h: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 1.0);
            g[1] = 0.0;
            h[0] = 2.0; h[1] = 0.0; h[2] = 0.0; h[3] = 0.0;
            (x[0] - 1.0).powi(2)
        },
    }
}

#[test]
fn min_diagonal_lets_a_zero_diagonal_solve() {
    let mut p = zero_diagonal_problem();
    let cfg = LmConfig::<f64> {
        min_diagonal: Some(1e-6),
        max_iters: 100,
        min_iters: 0,
        ..Default::default()
    };
    let result = simple_lm::solve(&[0.0, 0.0], &mut p, &cfg);

    assert_ne!(
        result.status,
        arael::simple_lm::LmStatus::DegenerateDiagonal { param: 1 },
        "the floor should have damped through it"
    );
    assert!(result.iterations > 0, "it should actually iterate now");
    // The parameter that HAS curvature reaches its minimum...
    assert!(
        (result.x[0] - 1.0).abs() < 1e-6,
        "x0 should converge to 1.0, got {}", result.x[0]
    );
    // ...and the one that has none does not move: its gradient is zero too, so
    // the damped system leaves it exactly where it started.
    assert_eq!(result.x[1], 0.0, "a parameter with no curvature must not drift");
    assert!(result.end_cost < 1e-12, "cost should go to zero: {}", result.end_cost);
}

#[test]
fn without_the_floor_the_same_problem_still_dies() {
    let _g = SINK_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    // The contrast: None is the default and is unchanged.
    let mut p = zero_diagonal_problem();
    let result = simple_lm::solve(&[0.0, 0.0], &mut p, &LmConfig::default());
    assert_eq!(result.status, arael::simple_lm::LmStatus::DegenerateDiagonal { param: 1 });
    assert_eq!(result.iterations, 0);
}

#[test]
fn min_diagonal_does_not_rescue_a_negative_diagonal() {
    let _g = SINK_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    // J^T J's diagonal is a sum of squares, so a negative one means the assembly
    // is poisoned. Flooring it would hide the bug, so it stays fatal.
    let mut p = FnProblem {
        cost: |x: &[f64]| (x[0] - 1.0).powi(2),
        grad_hessian: |x: &[f64], g: &mut [f64], h: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 1.0);
            g[1] = 0.0;
            h[0] = 2.0; h[1] = 0.0; h[2] = 0.0; h[3] = -1.0; // impossible
            (x[0] - 1.0).powi(2)
        },
    };
    let cfg = LmConfig::<f64> { min_diagonal: Some(1e-6), ..Default::default() };
    let result = simple_lm::solve(&[0.0, 0.0], &mut p, &cfg);
    assert_eq!(
        result.status,
        arael::simple_lm::LmStatus::DegenerateDiagonal { param: 1 },
        "a negative diagonal must stay fatal even with a floor"
    );
}

/// The three fatal cases must be told apart in the LOG, not just in the status.
/// A NaN reported as a "zero" would send the reader to `min_diagonal`, which
/// floors a zero and does nothing for a NaN. Every comparison against NaN is
/// false, so a `d < 0` test lets it fall through -- which it did.
#[test]
fn a_nan_diagonal_is_not_reported_as_a_zero() {
    let _g = SINK_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    use std::sync::{Arc, Mutex};
    let seen: Arc<Mutex<std::vec::Vec<String>>> = Arc::new(Mutex::new(std::vec::Vec::new()));
    let sink = Arc::clone(&seen);
    arael::log::set_sink(move |_, msg| sink.lock().unwrap().push(msg.to_string()));

    let mut p = FnProblem {
        cost: |x: &[f64]| (x[0] - 1.0).powi(2),
        grad_hessian: |x: &[f64], g: &mut [f64], h: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 1.0);
            g[1] = 0.0;
            h[0] = 2.0; h[1] = 0.0; h[2] = 0.0; h[3] = f64::NAN;
            (x[0] - 1.0).powi(2)
        },
    };
    // No floor -- this is the arm the first version of the message got wrong.
    let result = simple_lm::solve(&[0.0, 0.0], &mut p, &LmConfig::default());
    assert_eq!(result.status, arael::simple_lm::LmStatus::DegenerateDiagonal { param: 1 });

    let msgs = seen.lock().unwrap().join("\n");
    arael::log::reset_sink();
    assert!(
        msgs.contains("not a number"),
        "a NaN diagonal must be named as one:\n{msgs}"
    );
    assert!(
        !msgs.contains("min_diagonal"),
        "and must NOT advise min_diagonal, which cannot floor a NaN:\n{msgs}"
    );
}

#[test]
fn min_diagonal_does_not_rescue_a_nan_diagonal() {
    let _g = SINK_GUARD.lock().unwrap_or_else(|e| e.into_inner());
    let mut p = FnProblem {
        cost: |x: &[f64]| (x[0] - 1.0).powi(2),
        grad_hessian: |x: &[f64], g: &mut [f64], h: &mut [f64]| {
            g[0] = 2.0 * (x[0] - 1.0);
            g[1] = 0.0;
            h[0] = 2.0; h[1] = 0.0; h[2] = 0.0; h[3] = f64::NAN;
            (x[0] - 1.0).powi(2)
        },
    };
    let cfg = LmConfig::<f64> { min_diagonal: Some(1e-6), ..Default::default() };
    let result = simple_lm::solve(&[0.0, 0.0], &mut p, &cfg);
    assert_eq!(
        result.status,
        arael::simple_lm::LmStatus::DegenerateDiagonal { param: 1 },
        "a NaN diagonal must stay fatal even with a floor"
    );
}

// Pattern drift: the indexed (cached-pattern) assembly must detect a sparsity
// pattern that changed mid-solve. The position map is built from the
// first iteration's entry sequence; a TripletBlock emitting fewer
// entries later (here: a guard flipped between assemblies) used to
// scatter every subsequent block into wrong slots -- silently wrong
// Hessian values with no error anywhere.

use arael::model::{Model, Param, SelfBlock, TripletBlock};
use arael::refs;

#[arael::model]
#[arael(constraint([hb, root.hbt], guard = self.active, {
    [(item10.a + w10.offset) * w10.isigma]
}))]
struct Item10 {
    a: Param<f64>,
    active: bool,
    hb: SelfBlock<Item10>,
}

#[arael::model]
#[arael(root)]
struct W10 {
    items: refs::Vec<Item10>,
    offset: Param<f64>,
    isigma: f64,
    hb: SelfBlock<W10>,
    hbt: TripletBlock<f64>,
}

#[test]
#[should_panic(expected = "sparsity pattern changed between iterations")]
fn pattern_drift_detected_in_indexed_assembly() {
    let mut items = refs::Vec::new();
    items.push(Item10 { a: Param::new(1.0), active: true, hb: SelfBlock::new() });
    let mut w = W10 {
        items,
        offset: Param::new(0.5),
        isigma: 2.0,
        hb: SelfBlock::new(),
        hbt: TripletBlock::new(),
    };
    let mut params = Vec::new();
    w.serialize64(&mut params);
    let n = params.len();

    // First iteration: build the pattern with the guard active.
    let mut grad = vec![0.0; n];
    let mut coo = simple_lm::CooMatrix::new(n);
    w.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
    let (csc, positions) = coo.to_csc_with_map().unwrap();

    // Mid-solve structure change: the guard flips off, the TripletBlock
    // emits nothing this iteration.
    w.items[0].active = false;

    let mut vals = vec![0.0; csc.vals.len()];
    let mut g2 = vec![0.0; n];
    let _ = w.calc_grad_hessian_sparse_indexed(&params, &mut g2, &mut vals, &positions);
}

// A structural failure that reaches the solve (not just the CSC helper) must
// surface as LmStatus::SetupFailed, with x untouched and NaN costs -- never a
// panic or a fabricated cost.

/// Band assembly reports an element outside the declared bandwidth.
struct BandOverflowProblem;
impl LmProblem<f64> for BandOverflowProblem {
    fn calc_cost(&mut self, _x: &[f64]) -> f64 { 1.0 }
    fn calc_grad_hessian_dense(&mut self, _x: &[f64], _g: &mut [f64], _h: &mut [f64]) -> f64 { 1.0 }
    fn calc_grad_hessian_band(&mut self, _x: &[f64], _g: &mut [f64], _b: &mut [f64], kd: usize) -> Result<f64, BandError> {
        Err(BandError { row: 0, col: 1, kd })
    }
    fn calc_grad_hessian_sparse(&mut self, _x: &[f64], _g: &mut [f64], _coo: &mut CooMatrix<f64>) -> f64 { unimplemented!() }
    fn calc_grad_hessian_sparse_direct(&mut self, _x: &[f64], _g: &mut [f64], _csc: &mut CscMatrix<f64>) -> f64 { unimplemented!() }
    fn calc_grad_hessian_sparse_indexed(&mut self, _x: &[f64], _g: &mut [f64], _v: &mut [f64], _p: &[usize]) -> f64 { unimplemented!() }
}

#[test]
fn band_overflow_is_a_setup_failure() {
    let mut p = BandOverflowProblem;
    let r = simple_lm::solve_band(&[0.0, 0.0], 0, &mut p, &LmConfig::default());
    assert_eq!(r.status, LmStatus::SetupFailed(SolveError::BandOverflow { row: 0, col: 1, kd: 0 }));
    assert_eq!(r.iterations, 0);
    assert_eq!(r.x, vec![0.0, 0.0], "params must be untouched");
    assert!(r.start_cost.is_nan() && r.end_cost.is_nan());
}

/// Sparse assembly leaves parameter 1 with no diagonal entry.
struct UnconstrainedSparseProblem;
impl LmProblem<f64> for UnconstrainedSparseProblem {
    fn calc_cost(&mut self, _x: &[f64]) -> f64 { 1.0 }
    fn calc_grad_hessian_dense(&mut self, _x: &[f64], _g: &mut [f64], _h: &mut [f64]) -> f64 { unimplemented!() }
    fn calc_grad_hessian_band(&mut self, _x: &[f64], _g: &mut [f64], _b: &mut [f64], _kd: usize) -> Result<f64, BandError> { unimplemented!() }
    fn calc_grad_hessian_sparse(&mut self, _x: &[f64], g: &mut [f64], coo: &mut CooMatrix<f64>) -> f64 {
        g[0] = 0.0; g[1] = 0.0;
        coo.push(0, 0, 2.0); // param 0 has a diagonal; param 1 has none
        1.0
    }
    fn calc_grad_hessian_sparse_direct(&mut self, _x: &[f64], _g: &mut [f64], _csc: &mut CscMatrix<f64>) -> f64 { unimplemented!() }
    fn calc_grad_hessian_sparse_indexed(&mut self, _x: &[f64], _g: &mut [f64], _v: &mut [f64], _p: &[usize]) -> f64 { unimplemented!() }
}

#[test]
fn unconstrained_parameter_is_a_setup_failure_through_solve() {
    let mut p = UnconstrainedSparseProblem;
    let r = simple_lm::solve_sparse(&[0.0, 0.0], &mut p, &LmConfig::default());
    assert_eq!(r.status, LmStatus::SetupFailed(SolveError::UnconstrainedParameter { param: 1 }));
    assert_eq!(r.iterations, 0);
    assert_eq!(r.x, vec![0.0, 0.0], "params must be untouched");
    assert!(r.start_cost.is_nan() && r.end_cost.is_nan());
}
