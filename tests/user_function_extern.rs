//! Form B (opaque eval + symbolic derivatives) `#[arael::function]` test.
//!
//! Declare a pair `f(x)` / `g(x)` with opaque f64 evaluators and
//! mutually-recursive derivative expressions (`df/dx = g(x)`,
//! `dg/dx = -f(x)`). Use inside a constraint; check residual + grad
//! match numerical reference.

use arael::model::{Model, Param, SelfBlock};
use arael::simple_lm::LmProblem;

// Opaque numerical evaluators -- sin and cos by another name. The
// symbolic derivatives below refer to them mutually:
//   f(x) = sin(x),  df/dx = cos(x) = g(x)
//   g(x) = cos(x),  dg/dx = -sin(x) = -f(x)

#[arael::function(f, derivs = [g(x)])]
fn f_eval(x: f64) -> f64 { x.sin() }

#[arael::function(g, derivs = [-f(x)])]
fn g_eval(x: f64) -> f64 { x.cos() }

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "sin_minus_half", {
    // Residual r = f(m.x) - 0.5 = sin(x) - 0.5.
    // At x = 0 this is -0.5 so cost = 0.25.
    // d r/dx = g(x) = cos(x); at x = 0 d r/dx = 1, so dC/dx = 2*(-0.5)*1 = -1.
    [(f(m.x) - 0.5) * m.isigma]
}))]
struct M {
    x: Param<f64>,
    isigma: f64,
    hb: SelfBlock<M>,
}

fn make() -> (M, Vec<f64>) {
    let mut m = M { x: Param::new(0.0), isigma: 1.0, hb: SelfBlock::new() };
    let mut p = Vec::new();
    m.serialize64(&mut p);
    (m, p)
}

#[test]
fn f_extern_forward_and_grad() {
    let (mut m, params) = make();
    let cost = m.calc_cost(&params);
    assert!((cost - 0.25).abs() < 1e-9, "cost {} != 0.25", cost);

    let n = params.len();
    let mut g = vec![0.0f64; n];
    let mut h = vec![0.0f64; n * n];
    m.calc_grad_hessian_dense(&params, &mut g, &mut h);

    // Analytic gradient at x = 0: dC/dx = 2 * (sin(0) - 0.5) * cos(0) = -1.
    assert!((g[0] - (-1.0)).abs() < 1e-6,
        "grad[0] = {} (expected -1 at x=0)", g[0]);

    // Numerical gradient.
    let eps = 1e-5_f64;
    let mut pp = params.clone();
    pp[0] += eps; let cp = m.calc_cost(&pp);
    pp[0] -= 2.0 * eps; let cm = m.calc_cost(&pp);
    let ng = (cp - cm) / (2.0 * eps);
    assert!((g[0] - ng).abs() < 1e-4, "grad[0]: analytic={} numerical={}", g[0], ng);
}

// Runtime callability of the emitted siblings (including mutual
// cross-references across declaration order) is covered by
// `tests/user_function_cross_ref_runtime.rs`.
