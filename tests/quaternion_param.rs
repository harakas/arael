// QuaternionParam behavior through the macro and solver. It is a gimbal-lock-free
// rotation whose reference is kept as a unit quaternion. It reuses
// EulerAngleParam's universal constraint codegen (ref_rotation * rotation(delta))
// -- only the re-centering differs (quaternion product + renormalize).

use arael::simple_lm::RootProblem;
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
    rig.serialize(&mut params);
    let result = simple_lm::solve(&params, &mut rig,
        &LmConfig { max_iters: 200, ..Default::default() }).unwrap();
    rig.deserialize(&result.x);

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

// `value` holds the initial guess during the solve; the solver works on an
// internal reference and syncs `value` only when deserialize reads back the
// result.
#[test]
fn value_syncs_only_on_deserialize() {
    let target = matrix3d::rotation_from_euler_angles(vect3d::new(0.3, -0.4, 0.9));
    let q0 = quaternd::from_euler_angles(vect3d::new(0.05, -0.02, 0.1));
    let mut rig = Rig {
        att: QuaternionParam::new(q0),
        target,
        isigma: 1.0,
        hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    rig.serialize(&mut params);
    let result = simple_lm::solve(&params, &mut rig,
        &LmConfig { max_iters: 200, ..Default::default() }).unwrap();
    assert!(result.end_cost < 1e-12, "solve must converge, cost={}", result.end_cost);

    // Before deserialize, value still holds the initial guess untouched.
    let v = rig.att.value;
    assert!(v.t == q0.t && v.v.x == q0.v.x && v.v.y == q0.v.y && v.v.z == q0.v.z,
        "value must stay at the initial guess until deserialize");

    // After deserialize, value carries the optimized orientation.
    rig.deserialize(&result.x);
    let m = rig.att.value.rotation_matrix();
    let err: f64 = (0..3).map(|i| (0..3).map(|j| (m[i][j] - target[i][j]).abs()).sum::<f64>()).sum();
    assert!(err < 1e-6, "deserialized value must recompose the result, err={}", err);
}

// Deserialize folds the handed-back delta with the same retraction advance
// uses (the rotation-vector map, not euler angles), and is idempotent.
#[test]
fn deserialize_folds_delta_via_exp_map() {
    let q0 = quaternd::from_euler_angles(vect3d::new(0.2, 0.5, -0.7));
    let mut att = QuaternionParam::new(q0);
    let mut data = Vec::<f64>::new();
    att.serialize_params(&mut data);
    assert_eq!(data.len(), 3);

    // A large delta, where the rotation-vector and euler retractions differ.
    let w = vect3d::new(0.3, -0.4, 0.5);
    data[0] = w.x; data[1] = w.y; data[2] = w.z;
    att.deserialize_params(&data);

    let expected = (q0 * quaternd::from_rotation_vector_small(w)).unit();
    let v = att.value;
    assert!((v.t - expected.t).abs() < 1e-12
        && (v.v.x - expected.v.x).abs() < 1e-12
        && (v.v.y - expected.v.y).abs() < 1e-12
        && (v.v.z - expected.v.z).abs() < 1e-12,
        "deserialize must fold via the rotation-vector retraction");

    // A second deserialize with the same data must not fold the delta twice.
    att.deserialize_params(&data);
    let v2 = att.value;
    assert!(v2.t == v.t && v2.v.x == v.v.x && v2.v.y == v.v.y && v2.v.z == v.v.z,
        "deserialize must be idempotent");
}

// update_self resets the working state from `value` (the documented Model
// contract), including on a model that was never serialized.
#[test]
fn update_self_derives_working_state_from_value() {
    let q = quaternd::from_euler_angles(vect3d::new(0.4, -0.3, 1.1));
    let mut att = QuaternionParam::new(q);
    att.update_self();
    let t = q.rotation_matrix();
    let err: f64 = (0..3).map(|i| (0..3).map(|j| (att.rotation_matrix[i][j] - t[i][j]).abs()).sum::<f64>()).sum();
    assert!(err < 1e-12, "update_self must derive the rotation from value, err={}", err);
}

// An optimizable rotation pulled to match a FIXED, non-identity rotation.
// The fixed att must evaluate at its value during the solve, not at the
// identity its default reference starts from.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let d = pairrig.free_att.rotation_matrix() - pairrig.fixed_att.rotation_matrix();
    [d[0][0], d[0][1], d[0][2],
     d[1][0], d[1][1], d[1][2],
     d[2][0], d[2][1], d[2][2]]
}))]
struct PairRig {
    free_att: QuaternionParam<f64>,
    fixed_att: QuaternionParam<f64>,
    hb: SelfBlock<PairRig>,
}

#[test]
fn fixed_quaternion_param_drives_constraints() {
    let q = quaternd::from_euler_angles(vect3d::new(0.4, -0.3, 1.1));
    let mut rig = PairRig {
        free_att: QuaternionParam::new(quaternd::identity()),
        fixed_att: QuaternionParam::fixed(q),
        hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    rig.serialize(&mut params);
    let result = simple_lm::solve(&params, &mut rig,
        &LmConfig { max_iters: 100, ..Default::default() }).unwrap();
    rig.deserialize(&result.x);

    let m = rig.free_att.value.rotation_matrix();
    let t = q.rotation_matrix();
    let err: f64 = (0..3).map(|i| (0..3).map(|j| (m[i][j] - t[i][j]).abs()).sum::<f64>()).sum();
    assert!(err < 1e-6,
        "free rotation must land on the fixed one's actual orientation, err={}", err);
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
    rig.serialize(&mut params);
    assert_eq!(params.len(), 0, "a fixed QuaternionParam must serialize no params");
    assert_eq!(rig.att.index(), u32::MAX);
}
