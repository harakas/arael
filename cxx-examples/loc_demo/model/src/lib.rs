//! The localization model of examples/loc_demo.rs, packaged for the
//! C++ interface: the model and solver are Rust; composing the
//! problem, the graduated ramp with the band solver, and the error
//! reports live in ../main.cpp. See the Rust example for the model
//! walkthrough -- the structs and constraints here are the same, with
//! pub fields and Defaults for the generated interface. The Rust
//! demo's debug-only camera ref on PointFeature is omitted (cameras
//! live on the C++ side).

use arael::model::{Param, SelfBlock, CrossBlock, SimpleEulerAngleParam};
use arael::refs::{self, Ref};
use arael::vect::{vect3f, vect2f};
use arael::matrix::matrix3f;

/// A detected point feature in camera frame. The constraint uses
/// mf2r/camera_pos/isigma; pixel is for debugging.
#[arael::model]
#[derive(Default)]
pub struct PointFeature {
    pub pixel: vect2f,
    /// Feature-to-robot rotation: col0 = view dir, col1/col2 = perp axes.
    pub mf2r: matrix3f,
    /// Camera position in robot frame.
    pub camera_pos: vect3f,
    /// 1/sigma for angular residuals (rad^-1).
    pub isigma: vect2f,
}

#[arael::model]
#[derive(Default)]
pub struct PoseInfo {
    pub delta_pos: vect3f,
    pub delta_ea: vect3f,
    pub delta_pos_cov_r: matrix3f,
    pub delta_pos_cov_isigma: vect3f,
    pub delta_ea_cov_r: matrix3f,
    pub delta_ea_cov_isigma: vect3f,
    pub tilt_roll: f32,
    pub tilt_pitch: f32,
    pub features: refs::Vec<PointFeature>,
}

/// Robot pose -- no GPS, the known landmarks provide the absolute
/// reference.
#[arael::model]
#[arael(constraint(hb_pose, {
    let pos_drift = pose.pos - pose.pos_value;
    let ea_drift = pose.ea - pose.ea_value;
    [pos_drift.x * path.drift_pos_isigma,
     pos_drift.y * path.drift_pos_isigma,
     pos_drift.z * path.drift_pos_isigma,
     ea_drift.x * path.drift_ea_isigma,
     ea_drift.y * path.drift_ea_isigma,
     ea_drift.z * path.drift_ea_isigma]
}))]
#[arael(constraint(hb_pose, {
    [(pose.ea.x - pose.info.tilt_roll) * path.tilt_isigma,
     (pose.ea.y - pose.info.tilt_pitch) * path.tilt_isigma]
}))]
#[derive(Default)]
pub struct Pose {
    pub pos: Param<vect3f>,
    pub ea: SimpleEulerAngleParam<f32>,
    pub info: PoseInfo,
    pub hb_pose: SelfBlock<Pose, f32>,
}

/// A known 3D landmark (fixed, not optimized).
#[arael::model]
#[derive(Default)]
pub struct PointLandmark {
    pub pos: vect3f,
    pub frines: std::vec::Vec<PointFrine>,
}

/// Observation linking a known landmark to a pose -- the hessian block
/// is remote (pose.hb_pose): only the pose has parameters.
#[arael::model]
#[arael(constraint(pose.hb_pose, parent=lm, {
    let gamma = path.gamma;
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - feature.camera_pos;
    let r_f = feature.mf2r.transpose() * r_r;
    let plain1 = atan2(r_f.y, r_f.x) * feature.isigma.x * path.frine_isigma_scale;
    let plain2 = atan2(r_f.z, r_f.x) * feature.isigma.y * path.frine_isigma_scale;
    let err1 = gamma * atan(plain1 / gamma);
    let err2 = gamma * atan(plain2 / gamma);
    [err1, err2]
}))]
#[derive(Default)]
pub struct PointFrine {
    #[arael(ref = root.poses)]
    pub pose: Ref<Pose>,
    #[arael(ref = pose.info.features)]
    pub feature: Ref<PointFeature>,
}

/// Odometry constraint between consecutive poses.
#[arael::model]
#[arael(constraint(hb, {
    let mr2w_prev = prev.ea.rotation_matrix();
    let pos_diff = mr2w_prev.transpose() * (cur.pos - prev.pos);
    let pos_err = pos_diff - cur.info.delta_pos;
    let pos_w = cur.info.delta_pos_cov_r.transpose() * pos_err;
    let mr2w_cur = cur.ea.rotation_matrix();
    let expected = mr2w_prev * cur.info.delta_ea.rotation_matrix();
    let error_rot = expected.transpose() * mr2w_cur;
    let ea_err = error_rot.get_euler_angles();
    let ea_w = cur.info.delta_ea_cov_r.transpose() * ea_err;
    [pos_w.x * cur.info.delta_pos_cov_isigma.x,
     pos_w.y * cur.info.delta_pos_cov_isigma.y,
     pos_w.z * cur.info.delta_pos_cov_isigma.z,
     ea_w.x * cur.info.delta_ea_cov_isigma.x,
     ea_w.y * cur.info.delta_ea_cov_isigma.y,
     ea_w.z * cur.info.delta_ea_cov_isigma.z]
}))]
#[derive(Default)]
pub struct PosePair {
    #[arael(ref = root.poses)]
    pub prev: Ref<Pose>,
    #[arael(ref = root.poses)]
    pub cur: Ref<Pose>,
    pub hb: CrossBlock<Pose, Pose, f32>,
}

#[arael::model]
#[arael(root, f32, fast_atan)]
#[derive(Default)]
pub struct Path {
    pub poses: refs::Deque<Pose>,
    pub landmarks: refs::Arena<PointLandmark>,
    pub pose_pairs: std::vec::Vec<PosePair>,
    /// Starship robustifier scale for the feature residuals.
    pub gamma: f32,
    pub drift_pos_isigma: f32,
    pub drift_ea_isigma: f32,
    pub tilt_isigma: f32,
    pub frine_isigma_scale: f32,
}
