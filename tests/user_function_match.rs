//! `match` in a `#[arael::function]` body. The macro rewrites it to
//! `select` / `select_or` before the body reaches arael-sym, so it
//! evaluates and differentiates like the constraint-body `match`: the
//! taken arm only, an integer data field as the scrutinee, a panic on
//! an index no arm covers.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmProblem, RootProblem};
use arael::sym::{symbol, E};

/// 0 plain, 1 scaled by `k`, anything else squared.
#[arael::function]
fn pick(kind: E, s: E, k: E) -> E {
    match kind {
        0 => s,
        1 => k * s,
        _ => s * s,
    }
}

/// Nested, and no default on the outer match.
#[arael::function]
fn nested(outer: E, inner: E, a: E, b: E) -> E {
    match outer {
        0 => a,
        1 => match inner {
            0 => b,
            _ => { a * b },
        },
    }
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, {
    [pick(m.kind, m.x - m.t, m.k), nested(m.outer, m.inner, m.x, m.k)]
}))]
struct M {
    x: Param<f64>,
    t: f64,
    k: f64,
    kind: u32,
    outer: u32,
    inner: u32,
    hb: SelfBlock<M>,
}

/// Cost and d(cost)/dx at x = 3, t = 1, k = 2.
fn eval(kind: u32, outer: u32, inner: u32) -> (f64, f64) {
    let mut m = M {
        x: Param::new(3.0), t: 1.0, k: 2.0, kind, outer, inner, hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    m.serialize(&mut params);
    let cost = m.calc_cost(&params);
    let mut g = vec![0.0; 1];
    let mut h = vec![0.0; 1];
    m.calc_grad_hessian_dense(&params, &mut g, &mut h);
    (cost, g[0])
}

fn check(kind: u32, outer: u32, inner: u32, r: [f64; 2], dr: [f64; 2]) {
    let (cost, grad) = eval(kind, outer, inner);
    let want_cost = r[0] * r[0] + r[1] * r[1];
    let want_grad = 2.0 * (r[0] * dr[0] + r[1] * dr[1]);
    assert!((cost - want_cost).abs() < 1e-12,
        "kind {kind} outer {outer} inner {inner}: cost {cost} != {want_cost}");
    assert!((grad - want_grad).abs() < 1e-12,
        "kind {kind} outer {outer} inner {inner}: grad {grad} != {want_grad}");
}

#[test]
fn every_arm_takes_its_own_value_and_derivative() {
    // s = x - t = 2; pick: 0 -> s, 1 -> k s, else s^2.
    check(0, 0, 0, [2.0, 3.0], [1.0, 1.0]);
    check(1, 0, 0, [4.0, 3.0], [2.0, 1.0]);
    check(2, 0, 0, [4.0, 3.0], [4.0, 1.0]);
    check(7, 0, 0, [4.0, 3.0], [4.0, 1.0]);
    // nested: outer 0 -> x; outer 1 -> inner 0 -> k (constant), else x k.
    check(0, 1, 0, [2.0, 2.0], [1.0, 0.0]);
    check(0, 1, 1, [2.0, 6.0], [1.0, 2.0]);
    check(0, 1, 9, [2.0, 6.0], [1.0, 2.0]);
}

#[test]
#[should_panic(expected = "out of range")]
fn an_uncovered_outer_index_panics() {
    eval(0, 2, 0);
}

#[test]
fn siblings_show_the_select_form() {
    let p = format!("{}", pick(symbol("k"), symbol("s"), symbol("c")));
    assert!(p.starts_with("select_or(k, "), "{p}");
    let n = format!("{}", nested(symbol("o"), symbol("i"), symbol("a"), symbol("b")));
    assert!(n.starts_with("select(o, a, select_or(i, b, "), "{n}");
}
