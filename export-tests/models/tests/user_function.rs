//! A crate that depends on `arael` alone can declare and use every
//! `#[arael::function]` form. The macro's output must reach arael-sym
//! through `arael::sym`, since such a crate has no `arael_sym` in scope.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmProblem, RootProblem};
use arael::sym::{symbol, E};

/// Form A, auto-differentiated.
#[arael::function]
fn square(x: E) -> E {
    x * x
}

/// Form A with an explicit derivative.
#[arael::function(derivs = [3.0 * x * x])]
fn cube(x: E) -> E {
    x * x * x
}

/// Form B: opaque eval with a symbolic derivative.
#[arael::function(half_sin, derivs = [0.5 * cos(x)])]
fn half_sin_eval(x: f64) -> f64 {
    0.5 * x.sin()
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, {
    [square(m.x) - 9.0, cube(m.x) - 8.0, half_sin(m.x)]
}))]
struct M {
    x: Param<f64>,
    hb: SelfBlock<M>,
}

#[test]
fn every_form_evaluates_and_differentiates() {
    let mut m = M { x: Param::new(2.0), hb: SelfBlock::new() };
    let mut params = Vec::new();
    m.serialize(&mut params);

    // r = (4 - 9, 8 - 8, 0.5 sin 2) at x = 2.
    let r2 = 0.5 * 2.0f64.sin();
    let cost = m.calc_cost(&params);
    let expected = 25.0 + r2 * r2;
    assert!((cost - expected).abs() < 1e-12, "cost {cost} != {expected}");

    // dC/dx = 2 r0 (2x) + 2 r1 (3x^2) + 2 r2 (0.5 cos x).
    let mut g = vec![0.0; 1];
    let mut h = vec![0.0; 1];
    m.calc_grad_hessian_dense(&params, &mut g, &mut h);
    let expected_g = 2.0 * (-5.0) * 4.0 + 2.0 * r2 * 0.5 * 2.0f64.cos();
    assert!((g[0] - expected_g).abs() < 1e-9, "grad {} != {expected_g}", g[0]);
}

#[test]
fn siblings_are_callable_at_runtime() {
    for (name, e) in [
        ("square", square(symbol("x"))),
        ("cube", cube(symbol("x"))),
        ("half_sin", half_sin(symbol("x"))),
    ] {
        let s = format!("{e}");
        assert!(s.contains('x'), "{name}(x) printed as {s:?}");
    }
}
