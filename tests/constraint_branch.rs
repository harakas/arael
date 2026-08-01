// Integration test: branch(q, a, b) used inside a constraint body. The residual
// is asymmetric -- it penalises negative error three times harder than positive
// (branch on the sign of the error). Verifies both that the VALUE selects the
// right side and that the DERIVATIVE selects that side's slope (branch
// differentiates each arm; the switch contributes nothing).

use arael::simple_lm::RootProblem;
use arael::model::{Model, Param, SelfBlock};
use arael::simple_lm::LmProblem;
use arael::vect::vect2d;
use arael::refs;

#[arael::model]
#[arael(constraint(hb, {
    let e = node.pos.x - node.target;
    // e >= 0 -> residual e ; e < 0 -> residual 3e (steeper penalty).
    [branch(e, e, 3.0 * e)]
}))]
struct Node {
    pos: Param<vect2d>,
    target: f64,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(root)]
struct World {
    nodes: refs::Vec<Node>,
}

fn cost_and_grad(px: f64, target: f64) -> (f64, Vec<f64>) {
    let mut w = World { nodes: refs::Vec::new() };
    w.nodes.push(Node { pos: Param::new(vect2d::new(px, 0.0)), target, hb: SelfBlock::new() });
    let mut params = Vec::new();
    w.serialize(&mut params);
    let cost = w.calc_cost(&params);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    w.calc_grad_hessian_dense(&params, &mut grad, &mut hess);
    (cost, grad)
}

#[test]
fn branch_selects_value() {
    // e = 3 - 1 = +2 -> residual e -> cost e^2 = 4.
    let (c_pos, _) = cost_and_grad(3.0, 1.0);
    assert!((c_pos - 4.0).abs() < 1e-12, "positive-side cost {}", c_pos);
    // e = 1 - 3 = -2 -> residual 3e = -6 -> cost 36.
    let (c_neg, _) = cost_and_grad(1.0, 3.0);
    assert!((c_neg - 36.0).abs() < 1e-12, "negative-side cost {}", c_neg);
}

#[test]
fn branch_derivative_selects_slope() {
    // cost = sum(r^2); grad = d(cost)/d(px).
    // Positive side: r = e, cost = e^2, grad = 2e = 4 at e = 2.
    let (_, g_pos) = cost_and_grad(3.0, 1.0);
    assert!((g_pos[0] - 4.0).abs() < 1e-9, "positive-side grad {}", g_pos[0]);
    // Negative side: r = 3e, cost = 9 e^2, grad = 18e = -36 at e = -2.
    // (If branch's derivative didn't select the 3e arm, this would be wrong.)
    let (_, g_neg) = cost_and_grad(1.0, 3.0);
    assert!((g_neg[0] - (-36.0)).abs() < 1e-9, "negative-side grad {}", g_neg[0]);
}
