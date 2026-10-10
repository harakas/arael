//! Form B with the model root as an argument, in its three shapes: a
//! function declared without a derivative, a symbolic one whose derivs
//! read the root, and a numeric one that returns its partials from the
//! same call and is told whether they are read.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem, RootProblem};
use arael::sym::symbol;
use std::cell::Cell;

thread_local! {
    static WITH_PARTIALS: Cell<usize> = const { Cell::new(0) };
    static WITHOUT_PARTIALS: Cell<usize> = const { Cell::new(0) };
}

fn counts() -> (usize, usize) {
    (WITH_PARTIALS.with(|c| c.get()), WITHOUT_PARTIALS.with(|c| c.get()))
}

fn reset_counts() {
    WITH_PARTIALS.with(|c| c.set(0));
    WITHOUT_PARTIALS.with(|c| c.set(0));
}

// Declared without a derivative: the slope the root's gain gives.
#[arael::function(slope)]
fn slope_eval(root: &Scene, x: f64) -> f64 { root.k * x.cos() }

// Symbolic: the derivative is the slope's value.
#[arael::function(gain, derivs = [slope(root, x)])]
fn gain_eval(root: &Scene, x: f64) -> f64 { root.k * x.sin() }

// Symbolic with a root field in the derivative.
#[arael::function(lin, derivs = [root.k])]
fn lin_eval(root: &Scene, x: f64) -> f64 { root.k * x }

// Numeric: value and partial from one call, and the flag counted.
#[arael::function(bowl)]
fn bowl_eval(root: &Scene, x: f64, derivs: bool) -> (f64, [f64; 1]) {
    if derivs {
        WITH_PARTIALS.with(|c| c.set(c.get() + 1));
    } else {
        WITHOUT_PARTIALS.with(|c| c.set(c.get() + 1));
    }
    (root.scale * x * x, [if derivs { 2.0 * root.scale * x } else { 0.0 }])
}

// Numeric without a root, for the runtime sibling.
#[arael::function(sq)]
fn sq_eval(x: f64, derivs: bool) -> (f64, [f64; 1]) {
    (x * x, [if derivs { 2.0 * x } else { 0.0 }])
}

#[arael::model]
#[arael(constraint(hb, name = "fit", {
    [(gain(root, item.x) - item.target) * item.isigma,
     bowl(root, item.x) - item.target2,
     lin(root, item.x) - 1.0,
     // A function without a derivative over data: its derivative is
     // never asked for.
     slope(root, item.c) * item.x - 0.5]
}))]
struct Item {
    x: Param<f64>,
    c: f64,
    target: f64,
    target2: f64,
    isigma: f64,
    hb: SelfBlock<Item>,
}

#[arael::model]
#[arael(root, jacobian)]
struct Scene {
    k: f64,
    scale: f64,
    items: arael::refs::Vec<Item>,
}

const K: f64 = 1.5;
const SCALE: f64 = 0.7;
const X: f64 = 0.3;
const C: f64 = 0.4;
const TARGET: f64 = 0.2;
const TARGET2: f64 = 0.1;
const ISIGMA: f64 = 2.0;

fn make() -> (Scene, Vec<f64>) {
    let mut items = arael::refs::Vec::new();
    items.push(Item {
        x: Param::new(X), c: C, target: TARGET, target2: TARGET2, isigma: ISIGMA,
        hb: SelfBlock::new(),
    });
    let mut s = Scene { k: K, scale: SCALE, items };
    let mut p = Vec::new();
    s.serialize(&mut p);
    (s, p)
}

// The rows and their derivatives by x, by hand.
fn rows(x: f64) -> [(f64, f64); 4] {
    [((K * x.sin() - TARGET) * ISIGMA, K * x.cos() * ISIGMA),
     (SCALE * x * x - TARGET2, 2.0 * SCALE * x),
     (K * x - 1.0, K),
     (K * C.cos() * x - 0.5, K * C.cos())]
}

#[test]
fn value_gradient_and_hessian_match_the_formulas() {
    let (mut s, params) = make();
    let cost = s.calc_cost(&params);
    let want: f64 = rows(X).iter().map(|(r, _)| r * r).sum();
    assert!((cost - want).abs() < 1e-12, "cost {cost} vs {want}");

    let mut g = vec![0.0; 1];
    let mut h = vec![0.0; 1];
    s.calc_grad_hessian_dense(&params, &mut g, &mut h);
    let grad: f64 = rows(X).iter().map(|(r, d)| 2.0 * r * d).sum();
    let hess: f64 = rows(X).iter().map(|(_, d)| 2.0 * d * d).sum();
    assert!((g[0] - grad).abs() < 1e-12, "grad {} vs {grad}", g[0]);
    assert!((h[0] - hess).abs() < 1e-12, "hessian {} vs {hess}", h[0]);

    // And against finite differences of the cost.
    let eps = 1e-6;
    let cp = s.calc_cost(&[X + eps]);
    let cm = s.calc_cost(&[X - eps]);
    assert!((g[0] - (cp - cm) / (2.0 * eps)).abs() < 1e-6);
}

#[test]
fn the_flag_says_whether_the_partials_are_read() {
    let (mut s, params) = make();
    reset_counts();
    s.calc_cost(&params);
    let (with, without) = counts();
    assert_eq!(with, 0, "the cost sweep asked for partials");
    assert!(without > 0);

    reset_counts();
    let mut g = vec![0.0; 1];
    let mut h = vec![0.0; 1];
    s.calc_grad_hessian_dense(&params, &mut g, &mut h);
    let (with, without) = counts();
    assert!(with > 0, "the assembly did not ask for partials");
    assert_eq!(without, 0, "the assembly called once more without them");
}

#[test]
fn the_model_solves() {
    let (mut s, params) = make();
    let before = s.calc_cost(&params);
    let r = s.solve_dense(&LmConfig::default()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!(r.end_cost < before);
}

#[test]
fn the_siblings_build_the_calls() {
    let scene = symbol("scene");
    let x = symbol("x");
    assert_eq!(gain(scene.clone(), x.clone()).to_rust("f64"), "gain_eval(scene, x)");
    assert_eq!(gain(scene.clone(), x.clone()).diff("x").to_rust("f64"), "slope_eval(scene, x)");
    assert_eq!(bowl(scene.clone(), x.clone()).to_rust("f64"), "bowl_eval(scene, x, true).0");
    assert_eq!(bowl(scene.clone(), x.clone()).diff("x").to_rust("f64"),
        "bowl_eval(scene, x, true).1[0]");
    // Without a root the numeric sibling evaluates through the call.
    let vars = std::collections::HashMap::from([("x", 3.0)]);
    assert_eq!(sq(x.clone()).eval(&vars).unwrap(), 9.0);
    assert_eq!(sq(x.clone()).diff("x").eval(&vars).unwrap(), 6.0);
    // With one it cannot: there is no root at runtime.
    let err = bowl(scene, x).eval(&std::collections::HashMap::from([("scene", 0.0), ("x", 3.0)]))
        .unwrap_err();
    assert!(matches!(err, arael::sym::SymError::NoEval(_)), "{err}");
}

#[test]
#[should_panic(expected = "reads the model root")]
fn a_root_taking_scalar_sibling_cannot_evaluate() {
    let vars = std::collections::HashMap::from([("scene", 0.0), ("x", 3.0)]);
    let _ = gain(symbol("scene"), symbol("x")).eval(&vars);
}
