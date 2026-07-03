// EulerAngleParam behavior through the macro and solver.

use arael::model::{Model, SelfBlock, EulerAngleParam};
use arael::simple_lm::{self, LmConfig};
use arael::vect::vect3d;
use arael::matrix::matrix3d;
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

// ---------------------------------------------------------------------------
// Advance must reach EA params everywhere, not only in root collections
// ---------------------------------------------------------------------------

// A root-level EulerAngleParam pulled to a 150-degree pitch target (as a
// rotation matrix -- such a pitch has no principal euler triple). The
// delta must travel through pitch = 90; without per-step re-centering
// (advance) the delta parametrization degenerates at the gimbal and the
// solve stalls. Advance used to visit only root-level collections, so a
// root-level EA param was never re-centered.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let d = rig.att.rotation_matrix() - rig.target;
    [d[0][0] * rig.isigma, d[0][1] * rig.isigma, d[0][2] * rig.isigma,
     d[1][0] * rig.isigma, d[1][1] * rig.isigma, d[1][2] * rig.isigma,
     d[2][0] * rig.isigma, d[2][1] * rig.isigma, d[2][2] * rig.isigma]
}))]
struct Rig {
    att: EulerAngleParam<f64>,
    target: matrix3d,
    isigma: f64,
    hb: SelfBlock<Rig>,
}

#[test]
fn root_level_euler_angle_param_advances_through_gimbal() {
    let target = matrix3d::rotation_from_euler_angles(vect3d::new(0.0, 1.2, 0.0))
        * matrix3d::rotation_from_euler_angles(vect3d::new(0.0, 1.4, 0.0));
    // Combined pitch 2.6 rad (149 deg), well past the gimbal at pi/2.
    let mut rig = Rig {
        att: EulerAngleParam::new(vect3d::new(0.0, 0.0, 0.0)),
        target,
        isigma: 1.0,
        hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    rig.serialize64(&mut params);
    let result = simple_lm::solve(&params, &mut rig,
        &LmConfig { max_iters: 200, ..Default::default() });
    rig.deserialize64(&result.x);

    assert!(result.end_cost < 1e-12,
        "root-level EA must converge through the gimbal, cost={}", result.end_cost);
    let m = matrix3d::rotation_from_euler_angles(rig.att.value);
    let err: f64 = (0..3).map(|i| (0..3).map(|j| (m[i][j] - target[i][j]).abs()).sum::<f64>()).sum();
    assert!(err < 1e-6, "recomposition error {}", err);
}
