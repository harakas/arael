// EulerAngleParam behavior through the macro and solver.

use arael::model::{Model, SelfBlock, CrossBlock, EulerAngleParam};
use arael::simple_lm::{self, LmConfig};
use arael::vect::{vect3d, vect3f};
use arael::matrix::{matrix3d, matrix3f};
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
    let result = simple_lm::solve(&params, &mut w, &LmConfig::default()).unwrap();
    w.deserialize64(&result.x);

    let free = &w.nodes[0];
    assert!((free.ea.value - vect3d::new(0.3, -0.2, 0.4)).norm() < 1e-6,
        "free EA must reach its target, got {:?}", free.ea.value);
    let fixed = &w.nodes[1];
    assert!((fixed.ea.value - frozen).norm() < 1e-12,
        "fixed EA must not move, got {:?}", fixed.ea.value);
}

// An optimizable rotation pulled to match a FIXED, non-identity rotation.
// The fixed ea must evaluate at its value during the solve, not at the
// identity its default reference starts from.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let d = pairw.free_ea.rotation_matrix() - pairw.fixed_ea.rotation_matrix();
    [d[0][0], d[0][1], d[0][2],
     d[1][0], d[1][1], d[1][2],
     d[2][0], d[2][1], d[2][2]]
}))]
struct PairW {
    free_ea: EulerAngleParam<f64>,
    fixed_ea: EulerAngleParam<f64>,
    hb: SelfBlock<PairW>,
}

#[test]
fn fixed_euler_angle_param_drives_constraints() {
    let ea = vect3d::new(0.4, -0.3, 1.1);
    let mut w = PairW {
        free_ea: EulerAngleParam::new(vect3d::new(0.0, 0.0, 0.0)),
        fixed_ea: EulerAngleParam::fixed(ea),
        hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    w.serialize64(&mut params);
    let result = simple_lm::solve(&params, &mut w,
        &LmConfig { max_iters: 100, ..Default::default() }).unwrap();
    w.deserialize64(&result.x);

    let m = matrix3d::rotation_from_euler_angles(w.free_ea.value);
    let t = matrix3d::rotation_from_euler_angles(ea);
    let err: f64 = (0..3).map(|i| (0..3).map(|j| (m[i][j] - t[i][j]).abs()).sum::<f64>()).sum();
    assert!(err < 1e-6,
        "free rotation must land on the fixed one's actual orientation, err={}", err);
}

// update_self resets the working state from `value` (the documented Model
// contract), including on a model that was never serialized.
#[test]
fn update_self_derives_working_state_from_value() {
    let ea = vect3d::new(0.4, -0.3, 1.1);
    let mut p = EulerAngleParam::new(ea);
    p.update_self();
    let t = matrix3d::rotation_from_euler_angles(ea);
    let err: f64 = (0..3).map(|i| (0..3).map(|j| (p.rotation_matrix[i][j] - t[i][j]).abs()).sum::<f64>()).sum();
    assert!(err < 1e-12, "update_self must derive the rotation from value, err={}", err);
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
        &LmConfig { max_iters: 200, ..Default::default() }).unwrap();
    rig.deserialize64(&result.x);

    assert!(result.end_cost < 1e-12,
        "root-level EA must converge through the gimbal, cost={}", result.end_cost);
    let m = matrix3d::rotation_from_euler_angles(rig.att.value);
    let err: f64 = (0..3).map(|i| (0..3).map(|j| (m[i][j] - target[i][j]).abs()).sum::<f64>()).sum();
    assert!(err < 1e-6, "recomposition error {}", err);
}
// ---------------------------------------------------------------------------
// Aerobatics SLAM: barrel roll + Immelmann turn
// ---------------------------------------------------------------------------
//
// A pose chain whose true orientations fly nine consecutive barrel rolls
// (9 x 360 roll), an Immelmann turn (half loop -- pitch through 90 and on
// to 180 -- then a half roll back to upright), and three climbing barrel
// rolls on the way out (roll combined with a steady body-frame pitch-up,
// a corkscrew that sweeps orientation space and crosses the gimbal
// repeatedly). Relative-rotation constraints tie consecutive poses; the
// first pose is a fixed EulerAngleParam. All free poses start at
// identity, so the solver must rotate them across many gimbal crossings
// and accumulate thousands of degrees of rotation. This exercises
// exactly what EulerAngleParam is for: the delta parametrization stays
// near zero thanks to advance(), while ref_rotation accumulates
// arbitrarily large rotations.

// Body-frame incremental rotations (roll, pitch) along the maneuver:
//   9 x 24 roll steps = nine 360 deg barrel rolls,
//   12 x pitch steps  = 180 deg half loop (through the gimbal at 90),
//   12 x roll steps   = 180 deg half roll back to upright,
//   3 x 24 corkscrew steps = climbing barrel rolls (roll + pitch-up).
fn maneuver_steps() -> Vec<(f64, f64)> {
    let step = 15.0_f64.to_radians();
    let climb = 4.0_f64.to_radians();
    let mut out = Vec::new();
    for _ in 0..(9 * 24) { out.push((step, 0.0)); }
    for _ in 0..12 { out.push((0.0, step)); }
    for _ in 0..12 { out.push((step, 0.0)); }
    for _ in 0..(3 * 24) { out.push((step, climb)); }
    out
}

#[arael::model]
struct PoseE {
    ea: EulerAngleParam<f64>,
    hb: SelfBlock<PoseE>,
}

#[arael::model]
#[arael(constraint(hb, {
    let d = prev.ea.rotation_matrix().transpose() * cur.ea.rotation_matrix() - paire.delta;
    [d[0][0] * sky.isigma, d[0][1] * sky.isigma, d[0][2] * sky.isigma,
     d[1][0] * sky.isigma, d[1][1] * sky.isigma, d[1][2] * sky.isigma,
     d[2][0] * sky.isigma, d[2][1] * sky.isigma, d[2][2] * sky.isigma]
}))]
struct PairE {
    #[arael(ref = root.poses)]
    prev: Ref<PoseE>,
    #[arael(ref = root.poses)]
    cur: Ref<PoseE>,
    delta: matrix3d,
    hb: CrossBlock<PoseE, PoseE>,
}

#[arael::model]
#[arael(root)]
struct Sky {
    poses: refs::Vec<PoseE>,
    pairs: std::vec::Vec<PairE>,
    isigma: f64,
}

#[test]
fn aerobatics_slam_barrel_roll_and_immelmann() {
    let deltas: Vec<matrix3d> = maneuver_steps().iter()
        .map(|&(r, p)| matrix3d::rotation_from_euler_angles(vect3d::new(r, p, 0.0)))
        .collect();

    // Ground truth by composition.
    let mut truth: Vec<matrix3d> = vec![matrix3d::identity()];
    for d in &deltas {
        truth.push(*truth.last().unwrap() * *d);
    }

    let mut sky = Sky {
        poses: refs::Vec::new(),
        pairs: std::vec::Vec::new(),
        isigma: 10.0,
    };
    // Fixed anchor at identity; every other pose starts at identity too,
    // maximally far from the flown trajectory.
    sky.poses.push(PoseE { ea: EulerAngleParam::fixed(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    for _ in 1..truth.len() {
        sky.poses.push(PoseE { ea: EulerAngleParam::new(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    }
    for (i, d) in deltas.iter().enumerate() {
        sky.pairs.push(PairE {
            prev: sky.poses.ref_at(i),
            cur: sky.poses.ref_at(i as u32 + 1),
            delta: *d,
            hb: CrossBlock::new(),
        });
    }

    let mut params = Vec::new();
    sky.serialize64(&mut params);
    let result = simple_lm::solve_sparse(&params, &mut sky,
        // VERBOSE=1 cargo test -r --test euler_param aerobatics -- --nocapture
        // prints the LM iteration trace.
        &LmConfig {
            max_iters: 500,
            verbose: std::env::var("VERBOSE").is_ok(),
            ..Default::default()
        }).unwrap();
    sky.deserialize64(&result.x);

    assert!(result.end_cost < 1e-10,
        "aerobatics chain must converge, cost={} after {} iters",
        result.end_cost, result.iterations);
    // Iteration budget: the well-conditioned delta parametrization
    // converges in ~81 iterations; the same chain with
    // SimpleEulerAngleParam needs ~285 (rejected steps and lambda churn
    // at every near-lock passage of the corkscrew). A regression in
    // advance()/re-centering shows up here as a budget failure long
    // before it breaks convergence outright.
    assert!(result.iterations <= 150,
        "conditioning regression: {} iterations (expected ~81)",
        result.iterations);
    // Every pose's recomposed rotation must match the flown trajectory --
    // compare matrices, not euler triples, since several poses sit at or
    // beyond the gimbal where the triple is not unique.
    for (i, t) in truth.iter().enumerate() {
        let m = matrix3d::rotation_from_euler_angles(
            sky.poses[i as usize].ea.value);
        let err: f64 = (0..3).map(|r| (0..3).map(|c| (m[r][c] - t[r][c]).abs()).sum::<f64>()).sum();
        assert!(err < 1e-4, "pose {} orientation error {}", i, err);
    }
}

// ---------------------------------------------------------------------------
// The same maneuver in pure f32
// ---------------------------------------------------------------------------
//
// Same trajectory, f32 root, f32 blocks, solve_sparse_f32. Pins the
// f32 model pipeline (previously untested end to end) and the f32
// precision floor of the euler machinery: the solver satisfies every
// relative-rotation constraint to ~2 ulps of f32 (cost floor ~2e-8) and
// recovers orientations to well under 0.02 deg (measured worst 0.007 deg
// deep in the corkscrew) in essentially the same iteration count as f64.

#[arael::model]
struct PoseF {
    ea: EulerAngleParam<f32>,
    hb: SelfBlock<PoseF, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let d = prev.ea.rotation_matrix().transpose() * cur.ea.rotation_matrix() - pairf.delta;
    [d[0][0] * skyf.isigma, d[0][1] * skyf.isigma, d[0][2] * skyf.isigma,
     d[1][0] * skyf.isigma, d[1][1] * skyf.isigma, d[1][2] * skyf.isigma,
     d[2][0] * skyf.isigma, d[2][1] * skyf.isigma, d[2][2] * skyf.isigma]
}))]
struct PairF {
    #[arael(ref = root.poses)]
    prev: Ref<PoseF>,
    #[arael(ref = root.poses)]
    cur: Ref<PoseF>,
    delta: matrix3f,
    hb: CrossBlock<PoseF, PoseF, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct SkyF {
    poses: refs::Vec<PoseF>,
    pairs: std::vec::Vec<PairF>,
    isigma: f32,
}

#[test]
fn aerobatics_slam_f32() {
    let deltas: Vec<matrix3f> = maneuver_steps().iter()
        .map(|&(r, p)| matrix3f::rotation_from_euler_angles(vect3f::new(r as f32, p as f32, 0.0)))
        .collect();

    let mut truth: Vec<matrix3f> = vec![matrix3f::identity()];
    for d in &deltas {
        truth.push(*truth.last().unwrap() * *d);
    }

    let mut sky = SkyF {
        poses: refs::Vec::new(),
        pairs: std::vec::Vec::new(),
        isigma: 10.0,
    };
    sky.poses.push(PoseF { ea: EulerAngleParam::fixed(vect3f::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    for _ in 1..truth.len() {
        sky.poses.push(PoseF { ea: EulerAngleParam::new(vect3f::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    }
    for (i, d) in deltas.iter().enumerate() {
        sky.pairs.push(PairF {
            prev: sky.poses.ref_at(i),
            cur: sky.poses.ref_at(i as u32 + 1),
            delta: *d,
            hb: CrossBlock::new(),
        });
    }

    let mut params = Vec::new();
    sky.serialize32(&mut params);
    let result = simple_lm::solve_sparse_f32(&params, &mut sky, &LmConfig {
        max_iters: 500,
        verbose: std::env::var("VERBOSE").is_ok(),
        ..Default::default()
    }).unwrap();
    sky.deserialize32(&result.x);

    // f32 floor: residuals of ~2 ulps per matrix element.
    assert!(result.end_cost < 1e-7,
        "f32 aerobatics chain must converge to the f32 floor, cost={} after {} iters",
        result.end_cost, result.iterations);
    assert!(result.iterations <= 150,
        "conditioning regression: {} iterations (expected ~88)", result.iterations);
    for (i, t) in truth.iter().enumerate() {
        let m = matrix3f::rotation_from_euler_angles(
            sky.poses[i as usize].ea.value);
        let err: f32 = (0..3).map(|r| (0..3).map(|c| (m[r][c] - t[r][c]).abs()).sum::<f32>()).sum();
        assert!(err < 3e-3, "pose {} orientation error {} (f32 bound)", i, err);
    }
}
