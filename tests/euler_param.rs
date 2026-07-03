// EulerAngleParam behavior through the macro and solver.

use arael::model::{Model, SelfBlock, EulerAngleParam};
use arael::simple_lm::{self, LmConfig};
use arael::vect::vect3d;
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, {
    let d = node.ea - node.target;
    [d.x * w.isigma, d.y * w.isigma, d.z * w.isigma]
}))]
struct Node {
    ea: EulerAngleParam<f64>,
    target: vect3d,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(root)]
struct W {
    nodes: refs::Vec<Node>,
    isigma: f64,
}

// A fixed EulerAngleParam in a collection must not panic in the
// generated advance(): its index is the u32::MAX sentinel, and every
// accepted LM step used to do params[u32::MAX] on it.
#[test]
fn fixed_euler_angle_param_does_not_panic_in_advance() {
    let mut w = W { nodes: refs::Vec::new(), isigma: 1.0 };
    w.nodes.push(Node {
        ea: EulerAngleParam::new(vect3d::new(0.0, 0.0, 0.0)),
        target: vect3d::new(0.3, -0.2, 0.4),
        hb: SelfBlock::new(),
    });
    let frozen = vect3d::new(0.1, 0.2, -0.3);
    w.nodes.push(Node {
        ea: EulerAngleParam::fixed(frozen),
        target: vect3d::new(1.0, 1.0, 1.0), // pulls, but the param is fixed
        hb: SelfBlock::new(),
    });

    let mut params = Vec::new();
    w.serialize64(&mut params);
    let result = simple_lm::solve(&params, &mut w, &LmConfig::default());
    w.deserialize64(&result.x);

    let free = &w.nodes[Ref::<Node>::new(0)];
    assert!((free.ea.value - vect3d::new(0.3, -0.2, 0.4)).norm() < 1e-6,
        "free EA must reach its target, got {:?}", free.ea.value);
    let fixed = &w.nodes[Ref::<Node>::new(1)];
    assert!((fixed.ea.value - frozen).norm() < 1e-12,
        "fixed EA must not move, got {:?}", fixed.ea.value);
}
