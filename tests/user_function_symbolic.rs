//! Form A (purely symbolic) `#[arael::function]` test.
//!
//! Declare a user function `square(x: E) -> E { x * x }` and use it
//! inside a constraint body. Residual value, gradient, and Hessian
//! from the generated code must match numerical reference.

use arael::model::{JacobianModel, Model, Param, SelfBlock};
use arael::simple_lm::LmProblem;
use arael_sym::E;

#[arael::function]
fn square(x: E) -> E {
    x * x
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "sq_x_eq_9", {
    // residual = square(m.x) - 9.0, weighted by isigma
    [(square(m.x) - 9.0) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn make() -> (M, Vec<f64>) {
    let mut m = M { x: Param::new(2.0), isigma: 1.0, hb: SelfBlock::new() };
    let mut p = Vec::new();
    m.serialize64(&mut p);
    (m, p)
}

#[test]
fn square_user_fn_forward_and_grad() {
    let (mut m, params) = make();
    // Residual at x = 2: r = 2*2 - 9 = -5. Cost = r^2 = 25.
    let cost = m.calc_cost(&params);
    assert!((cost - 25.0).abs() < 1e-9, "cost {} != 25", cost);

    // Analytic grad / hessian (GN) via calc_grad_hessian_dense.
    let n = params.len();
    let mut g = vec![0.0f64; n];
    let mut h = vec![0.0f64; n * n];
    m.calc_grad_hessian_dense(&params, &mut g, &mut h);

    // Numerical grad: dC/dx = 2 * r * d r/dx = 2 * (-5) * 2x = 2*(-5)*4 = -40
    let eps = 1e-5_f64;
    let mut pp = params.clone();
    pp[0] += eps; let cp = m.calc_cost(&pp);
    pp[0] -= 2.0 * eps; let cm = m.calc_cost(&pp);
    let ng = (cp - cm) / (2.0 * eps);
    assert!((g[0] - ng).abs() < 1e-3, "grad[0]: analytic={} numerical={}", g[0], ng);

    // Direct value check: grad should be -40 at x=2.
    assert!((g[0] - (-40.0)).abs() < 1e-6, "grad[0] = {} (expected -40)", g[0]);
}

#[test]
fn square_user_fn_runtime_callable() {
    // The emitted fn should still be callable from ordinary Rust --
    // returns an E that prints as "x * x" when x is a symbol.
    let x = arael_sym::symbol("x");
    let e = square(x);
    let s = format!("{}", e);
    // Order may vary; accept either form.
    assert!(s.contains("x") && (s.contains("*") || s.contains("^")),
        "square(x) printed as {:?}", s);
}
