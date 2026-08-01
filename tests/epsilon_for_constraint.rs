// epsilon_for as a constraint-body builtin. The residual `x - epsilon_for(x)`
// has derivative 1 (epsilon_for is constant w.r.t. its anchor), so the solver
// drives x to `epsilon_for(x)` = the machine epsilon of the MODEL's precision.
// This proves epsilon_for is (a) usable in a `#[arael(constraint(...))]` body
// and (b) resolved to the right precision (f32::EPSILON in an f32 model, not
// the f64 value a folded literal would bake in).

use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock};
use arael::simple_lm::{self, LmConfig};

// Near-Gauss-Newton, tight precision so the linear residual converges all the
// way to epsilon (the default precision would stop far short of ~1e-16).
fn tight<T: arael::utils::Float>() -> LmConfig<T> {
    LmConfig {
        initial_lambda: T::from(1e-10).unwrap(),
        abs_precision: T::from(1e-40).unwrap(),
        rel_precision: T::from(1e-14).unwrap(),
        max_iters: 50,
        ..Default::default()
    }
}

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [node.x - epsilon_for(node.x)]
}))]
struct Node {
    x: Param<f64>,
    hb: SelfBlock<Node>,
}

#[test]
fn epsilon_for_in_constraint_f64() {
    let mut node = Node { x: Param::new(1.0), hb: SelfBlock::new() };
    let mut params = Vec::new();
    node.serialize(&mut params);
    let result = simple_lm::solve(&params, &mut node, &tight::<f64>()).unwrap();
    node.deserialize(&result.x);
    assert!((node.x.value / f64::EPSILON - 1.0).abs() < 1e-3,
        "x should converge to f64::EPSILON ({:e}), got {:e}", f64::EPSILON, node.x.value);
}

#[arael::model]
#[arael(root, f32)]
#[arael(constraint(hb, {
    [nodef.x - epsilon_for(nodef.x)]
}))]
struct NodeF {
    x: Param<f32>,
    hb: SelfBlock<NodeF, f32>,
}

#[test]
fn epsilon_for_in_constraint_f32_resolves_to_f32_epsilon() {
    let mut node = NodeF { x: Param::new(1.0), hb: SelfBlock::new() };
    let mut params = Vec::new();
    node.serialize(&mut params);
    let result = simple_lm::solve_f32(&params, &mut node, &tight::<f32>()).unwrap();
    node.deserialize(&result.x);
    // The whole point: in an f32 model epsilon_for gives f32::EPSILON (~1.2e-7),
    // NOT the f64 value (~2.2e-16) that a folded literal would bake in.
    assert!(node.x.value > 1e-8, "must be f32-scale epsilon, got {:e}", node.x.value);
    assert!((node.x.value / f32::EPSILON - 1.0).abs() < 1e-2,
        "x should converge to f32::EPSILON ({:e}), got {:e}", f32::EPSILON, node.x.value);
}
