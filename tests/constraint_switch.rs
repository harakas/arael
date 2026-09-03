// Integration test: the variadic switches inside a constraint body.
// `select` picks a residual by an integer data field and `piecewise`
// by thresholds on a scalar; both compile through the generated code
// (the `select_index()` conversion, the fused `match` / `if`) and both
// the VALUE and the DERIVATIVE follow the taken arm.

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{LmProblem, RootProblem};

#[arael::model]
#[arael(constraint(hb, {
    let e = node.x - node.target;
    // kind 0: plain error; kind 1: three times steeper; anything else panics.
    [select(node.kind, e, 3.0 * e)]
}))]
struct Node {
    x: Param<f64>,
    target: f64,
    kind: u32,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(constraint(hb, {
    let e = pw.x - pw.target;
    // e up to 0: e; up to 1: 2e; above: 3e.
    [piecewise(e, e, 0.0, 2.0 * e, 1.0, 3.0 * e)]
}))]
struct Pw {
    x: Param<f64>,
    target: f64,
    hb: SelfBlock<Pw>,
}

#[arael::model]
#[arael(root)]
struct World {
    nodes: refs::Vec<Node>,
    pws: refs::Vec<Pw>,
}

fn cost_and_grad(w: &mut World) -> (f64, Vec<f64>) {
    let mut params = Vec::new();
    w.serialize(&mut params);
    let cost = w.calc_cost(&params);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    let cost2 = w.calc_grad_hessian_dense(&params, &mut grad, &mut hess);
    assert_eq!(cost, cost2);
    (cost, grad)
}

fn node_world(x: f64, target: f64, kind: u32) -> World {
    let mut w = World { nodes: refs::Vec::new(), pws: refs::Vec::new() };
    w.nodes.push(Node { x: Param::new(x), target, kind, hb: SelfBlock::new() });
    w
}

fn pw_world(x: f64, target: f64) -> World {
    let mut w = World { nodes: refs::Vec::new(), pws: refs::Vec::new() };
    w.pws.push(Pw { x: Param::new(x), target, hb: SelfBlock::new() });
    w
}

#[test]
fn select_by_integer_field_picks_value_and_slope() {
    // e = 1. kind 0: r = e, cost 1, d cost/dx = 2 r dr/dx = 2.
    let (cost, grad) = cost_and_grad(&mut node_world(2.0, 1.0, 0));
    assert_eq!(cost, 1.0);
    assert_eq!(grad, vec![2.0]);
    // kind 1: r = 3e, cost 9, d cost/dx = 2 * 3 * 3 = 18.
    let (cost, grad) = cost_and_grad(&mut node_world(2.0, 1.0, 1));
    assert_eq!(cost, 9.0);
    assert_eq!(grad, vec![18.0]);
}

#[test]
#[should_panic(expected = "select index 2 out of range 0..2")]
fn select_out_of_range_panics() {
    let _ = cost_and_grad(&mut node_world(2.0, 1.0, 2));
}

#[test]
fn piecewise_picks_interval_value_and_slope() {
    // e = -0.5: r = e, cost 0.25, grad 2 e = -1.
    let (cost, grad) = cost_and_grad(&mut pw_world(0.5, 1.0));
    assert_eq!(cost, 0.25);
    assert_eq!(grad, vec![-1.0]);
    // e = 0.5: r = 2e = 1, cost 1, grad 2 * 1 * 2 = 4.
    let (cost, grad) = cost_and_grad(&mut pw_world(1.5, 1.0));
    assert_eq!(cost, 1.0);
    assert_eq!(grad, vec![4.0]);
    // e = 2: r = 3e = 6, cost 36, grad 2 * 6 * 3 = 36.
    let (cost, grad) = cost_and_grad(&mut pw_world(3.0, 1.0));
    assert_eq!(cost, 36.0);
    assert_eq!(grad, vec![36.0]);
    // The break belongs to the lower arm: e = 0 -> r = e = 0; e = 1 -> r = 2e = 2.
    let (cost, _) = cost_and_grad(&mut pw_world(1.0, 1.0));
    assert_eq!(cost, 0.0);
    let (cost, grad) = cost_and_grad(&mut pw_world(2.0, 1.0));
    assert_eq!(cost, 4.0);
    assert_eq!(grad, vec![8.0]);
}
