// A remote-block constraint over an entity whose params live inside an
// #[arael(component)] field.
//
// The block is remote (`pose.hb`): the observed landmark is a constant, so
// only the pose carries derivatives and the Hessian lands on the pose's own
// self-block. That is the localization shape -- a fixed map, single-variable
// observations -- and it has to reach a component's params through the ref,
// at the same offsets a local self-block would use.

use arael::model::{Model, SelfBlock};
use arael::quatern::quaternd;
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::transform::TransformParam;
use arael::vect::vect3d;

// ---------------------------------------------------------------- remote

#[arael::model]
#[derive(Clone)]
struct Pose {
    r2w: TransformParam,
    hb: SelfBlock<Pose>,
}

#[arael::model]
#[arael(constraint(pose.hb, parent = lm, {
    let local = pose.r2w.rotation_matrix.transpose() * (lm.pos - pose.r2w.translation);
    [local.x - obs.measured.x, local.y - obs.measured.y, local.z - obs.measured.z]
}))]
#[derive(Clone)]
struct Obs {
    #[arael(ref = root.poses)]
    pose: Ref<Pose>,
    measured: vect3d,
}

#[arael::model]
#[derive(Clone)]
struct Landmark {
    pos: vect3d,
    obs: std::vec::Vec<Obs>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
struct RemoteWorld {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<Landmark>,
}

// ----------------------------------------------------------------- local
// The same problem with the observations carried on the pose itself, so the
// block is an ordinary local SelfBlock. Both must land on the same solution.

#[arael::model]
#[arael(constraint(hb, {
    let local = posel.r2w.rotation_matrix.transpose() * (posel.lm0 - posel.r2w.translation);
    let local1 = posel.r2w.rotation_matrix.transpose() * (posel.lm1 - posel.r2w.translation);
    let local2 = posel.r2w.rotation_matrix.transpose() * (posel.lm2 - posel.r2w.translation);
    [local.x - posel.m0.x, local.y - posel.m0.y, local.z - posel.m0.z,
     local1.x - posel.m1.x, local1.y - posel.m1.y, local1.z - posel.m1.z,
     local2.x - posel.m2.x, local2.y - posel.m2.y, local2.z - posel.m2.z]
}))]
#[derive(Clone)]
struct PoseL {
    r2w: TransformParam,
    lm0: vect3d, lm1: vect3d, lm2: vect3d,
    m0: vect3d, m1: vect3d, m2: vect3d,
    hb: SelfBlock<PoseL>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
struct LocalWorld {
    poses: refs::Vec<PoseL>,
}

// ------------------------------------------------------------------ data
// Three non-collinear landmarks seen from one pose: 9 residuals against the
// transform's 6 params, so the pose is over-determined and recoverable.

const LMS: [[f64; 3]; 3] = [[4.0, 0.5, 0.2], [3.0, -2.0, 1.5], [5.0, 1.0, -1.0]];

fn truth() -> (vect3d, quaternd) {
    (vect3d::new(0.4, -0.3, 0.15),
     quaternd::from_euler_angles(vect3d::new(0.05, -0.08, 0.2)))
}

/// What the pose should observe at the truth: each landmark in its frame.
fn measurements() -> Vec<vect3d> {
    let (t, q) = truth();
    let r = q.rotation_matrix();
    LMS.iter()
        .map(|l| r.transpose() * (vect3d::new(l[0], l[1], l[2]) - t))
        .collect()
}

fn start() -> TransformParam {
    TransformParam::new(vect3d::new(0.0, 0.0, 0.0), quaternd::identity())
}

#[test]
fn remote_block_reaches_component_params() {
    let m = measurements();
    let mut w = RemoteWorld { poses: refs::Vec::new(), landmarks: refs::Vec::new() };
    let pose = w.poses.push(Pose { r2w: start(), hb: SelfBlock::new() });
    for (j, l) in LMS.iter().enumerate() {
        w.landmarks.push(Landmark {
            pos: vect3d::new(l[0], l[1], l[2]),
            obs: vec![Obs { pose, measured: m[j] }],
        });
    }

    let result = w.solve_sparse(&LmConfig::default());
    assert!(result.end_cost < 1e-16, "end cost {}", result.end_cost);

    let (t, q) = truth();
    let got = &w.poses[pose].r2w;
    assert!((got.translation - t).norm() < 1e-7,
        "translation {:?} vs {:?}", got.translation, t);
    // Same rotation, up to quaternion sign.
    let dot = (got.rotation.t * q.t + got.rotation.v * q.v).abs();
    assert!((dot - 1.0).abs() < 1e-9, "rotation off, |dot| = {}", dot);
}

#[test]
fn remote_block_matches_a_local_self_block() {
    let m = measurements();

    let mut remote = RemoteWorld { poses: refs::Vec::new(), landmarks: refs::Vec::new() };
    let rp = remote.poses.push(Pose { r2w: start(), hb: SelfBlock::new() });
    for (j, l) in LMS.iter().enumerate() {
        remote.landmarks.push(Landmark {
            pos: vect3d::new(l[0], l[1], l[2]),
            obs: vec![Obs { pose: rp, measured: m[j] }],
        });
    }

    let mut local = LocalWorld { poses: refs::Vec::new() };
    let lp = local.poses.push(PoseL {
        r2w: start(),
        lm0: vect3d::new(LMS[0][0], LMS[0][1], LMS[0][2]),
        lm1: vect3d::new(LMS[1][0], LMS[1][1], LMS[1][2]),
        lm2: vect3d::new(LMS[2][0], LMS[2][1], LMS[2][2]),
        m0: m[0], m1: m[1], m2: m[2],
        hb: SelfBlock::new(),
    });

    let cfg = LmConfig::default();
    let r_remote = remote.solve_sparse(&cfg);
    let r_local = local.solve_sparse(&cfg);

    // Same problem, same parameter span, so the whole trajectory matches --
    // a wrong index would put the same residuals on different parameters.
    assert_eq!(r_remote.iterations, r_local.iterations);
    assert!((r_remote.start_cost - r_local.start_cost).abs() < 1e-12,
        "start {} vs {}", r_remote.start_cost, r_local.start_cost);
    assert!((r_remote.end_cost - r_local.end_cost).abs() < 1e-12,
        "end {} vs {}", r_remote.end_cost, r_local.end_cost);
    assert!((remote.poses[rp].r2w.translation - local.poses[lp].r2w.translation).norm() < 1e-12);
}

/// The component's params must occupy the pose's span, so a pose with a
/// 6-DOF transform reports 6 -- not the 0 a component-blind walk would give.
#[test]
fn the_component_span_is_counted() {
    let mut w = RemoteWorld { poses: refs::Vec::new(), landmarks: refs::Vec::new() };
    let pose = w.poses.push(Pose { r2w: start(), hb: SelfBlock::new() });
    let m = measurements();
    for (j, l) in LMS.iter().enumerate() {
        w.landmarks.push(Landmark {
            pos: vect3d::new(l[0], l[1], l[2]),
            obs: vec![Obs { pose, measured: m[j] }],
        });
    }
    let mut params = std::vec::Vec::new();
    w.serialize_params64(&mut params);
    assert_eq!(params.len(), 6, "transform contributes 6 params");
}

/// Freezing one half of the transform must still leave the other solvable
/// through the remote block: the span shrinks and the indices follow.
#[test]
fn a_frozen_half_still_indexes_correctly() {
    let m = measurements();
    let (t, _) = truth();
    let mut w = RemoteWorld { poses: refs::Vec::new(), landmarks: refs::Vec::new() };
    let mut p = start();
    // Hand it the true rotation and freeze it; only the translation moves.
    p.rotation = truth().1;
    p.rotation_matrix = p.rotation.rotation_matrix();
    p.optimize_rotation = false;
    let pose = w.poses.push(Pose { r2w: p, hb: SelfBlock::new() });
    for (j, l) in LMS.iter().enumerate() {
        w.landmarks.push(Landmark {
            pos: vect3d::new(l[0], l[1], l[2]),
            obs: vec![Obs { pose, measured: m[j] }],
        });
    }

    let mut params = std::vec::Vec::new();
    w.serialize_params64(&mut params);
    assert_eq!(params.len(), 3, "only the translation is free");

    let result = w.solve_sparse(&LmConfig::default());
    assert!(result.end_cost < 1e-16, "end cost {}", result.end_cost);
    assert!((w.poses[pose].r2w.translation - t).norm() < 1e-7);
}
