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
