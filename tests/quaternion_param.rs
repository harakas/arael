// QuaternionParam behavior through the macro and solver. It is a gimbal-lock-free
// rotation whose reference is kept as a unit quaternion. It reuses
// EulerAngleParam's universal constraint codegen (ref_rotation * rotation(delta))
// -- only the re-centering differs (quaternion product + renormalize).

use arael::model::{Model, SelfBlock, QuaternionParam};
use arael::simple_lm::{self, LmConfig};
use arael::vect::vect3d;
use arael::matrix::matrix3d;
use arael::quatern::quaternd;

// A rotation pulled to a 149-degree pitch target (as a rotation matrix -- such a
// pitch has no principal euler triple). The delta must travel through
// pitch = 90; without per-step re-centering (advance) the delta parametrization
// degenerates at the gimbal and the solve stalls.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let d = rig.att.rotation_matrix() - rig.target;
    [d[0][0] * rig.isigma, d[0][1] * rig.isigma, d[0][2] * rig.isigma,
     d[1][0] * rig.isigma, d[1][1] * rig.isigma, d[1][2] * rig.isigma,
     d[2][0] * rig.isigma, d[2][1] * rig.isigma, d[2][2] * rig.isigma]
}))]
struct Rig {
    att: QuaternionParam<f64>,
    target: matrix3d,
    isigma: f64,
    hb: SelfBlock<Rig>,
}

#[test]
fn quaternion_param_advances_through_gimbal() {
    let target = matrix3d::rotation_from_euler_angles(vect3d::new(0.0, 1.2, 0.0))
        * matrix3d::rotation_from_euler_angles(vect3d::new(0.0, 1.4, 0.0));
    // Combined pitch 2.6 rad (149 deg), well past the gimbal at pi/2.
    let mut rig = Rig {
        att: QuaternionParam::new(quaternd::identity()),
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
        "QuaternionParam must converge through the gimbal, cost={}", result.end_cost);

    // The reference quaternion recomposes to the target rotation.
    let m = rig.att.value.rotation_matrix();
    let err: f64 = (0..3).map(|i| (0..3).map(|j| (m[i][j] - target[i][j]).abs()).sum::<f64>()).sum();
    assert!(err < 1e-6, "recomposition error {}", err);

    // The reference stays a unit quaternion (re-centering renormalizes it).
    assert!((rig.att.value.norm() - 1.0).abs() < 1e-9,
        "reference drifted off the unit sphere: |q| = {}", rig.att.value.norm());
}

// A fixed QuaternionParam contributes no parameters and must not be touched
// by the solver.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let d = fixedrig.att.rotation_matrix() - fixedrig.target;
    [d[0][0], d[1][1], d[2][2]]
}))]
struct FixedRig {
    att: QuaternionParam<f64>,
    target: matrix3d,
    hb: SelfBlock<FixedRig>,
}

#[test]
fn fixed_quaternion_param_has_no_params() {
    let q = quaternd::from_euler_angles(vect3d::new(0.1, -0.2, 0.3));
    let mut rig = FixedRig {
        att: QuaternionParam::fixed(q),
        target: q.rotation_matrix(),
        hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    rig.serialize64(&mut params);
    assert_eq!(params.len(), 0, "a fixed QuaternionParam must serialize no params");
    assert_eq!(rig.att.index(), u32::MAX);
}
