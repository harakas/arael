// Degenerate models -- a parameter with no Hessian diagonal entry -- must
// fail fast with an actionable message.
//
// B19: a CSC column without a stored diagonal left diag_pos[j] = 0, so
// extract_diagonal/solve_damped read and OVERWROTE vals[0] (an unrelated
// entry) -- silent corruption surfacing as "Cholesky failed" far away.
//
// B20: a structurally present but zero diagonal cannot be rescued by
// multiplicative damping ((1+lambda)*0 stays 0): the solver burned all 20
// inner failures per outer iteration against max_iters and returned with
// no error indication. It now terminates the solve immediately with an
// error log (a runtime value condition -- guards or saturated
// robustifiers can produce it from data -- so no panic; the structural
// missing-ENTRY case panics at CSC build).

use arael::simple_lm::{self, CooMatrix, FnProblem, LmConfig};
use arael::simple_lm::LmProblem;

#[test]
#[should_panic(expected = "no Hessian diagonal")]
fn missing_diagonal_rejected_in_to_csc() {
    let mut coo = CooMatrix::new(2);
    coo.push(0, 0, 2.0);
    coo.push(0, 1, 1.0); // column 1: off-diagonal only, no (1,1)
    let _ = coo.to_csc();
}

#[test]
#[should_panic(expected = "no Hessian diagonal")]
fn missing_diagonal_rejected_in_to_csc_with_map() {
    let mut coo = CooMatrix::new(2);
    coo.push(0, 0, 2.0);
    coo.push(0, 1, 1.0);
    let _ = coo.to_csc_with_map();
}

#[test]
fn zero_diagonal_terminates_immediately() {
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

// B10: the indexed (cached-pattern) assembly must detect a sparsity
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
    let (csc, positions) = coo.to_csc_with_map();

    // Mid-solve structure change: the guard flips off, the TripletBlock
    // emits nothing this iteration.
    w.items[refs::Ref::<Item10>::new(0)].active = false;

    let mut vals = vec![0.0; csc.vals.len()];
    let mut g2 = vec![0.0; n];
    let _ = w.calc_grad_hessian_sparse_indexed(&params, &mut g2, &mut vals, &positions);
}
