//! The visual-inertial SLAM model of examples/slam_demo_gm.rs, packaged
//! for the C++ interface: the model and solver are Rust; composing the
//! problem, reading results, and reporting live in ../main.cpp. See the
//! Rust example for the model walkthrough -- the structs and constraints
//! here are the same, with pub fields and Defaults for the generated
//! interface. The Rust demo's debug-only camera ref on PointFeature is
//! omitted (cameras live on the C++ side); anchor_pose is a plain ref
//! field so C++ can read it back for re-anchoring.

use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::transform::TransformParam;
use arael::unitvec::UnitVecParam;
use arael::vect::{vect2d, vect3d};
use arael::matrix::matrix3d;

/// A detected point feature in camera frame. The constraint uses
/// mf2r/camera_pos/isigma; pixel is for debugging.
#[arael::model]
#[derive(Default)]
pub struct PointFeature {
    pub pixel: vect2d,
    /// Feature-to-robot rotation: col0 = view dir, col1/col2 = perp axes.
    pub mf2r: matrix3d,
    /// Camera position in robot frame.
    pub camera_pos: vect3d,
    /// 1/sigma for angular residuals (rad^-1).
    pub isigma: vect2d,
}

/// Decomposed GPS reading: position + covariance split into R and 1/sqrt(d).
#[arael::model]
#[derive(Default)]
pub struct GpsData {
    pub pos: vect3d,
    pub cov_r: matrix3d,
    pub cov_isigma: vect3d,
}

#[arael::model]
#[derive(Default)]
pub struct PoseInfo {
    pub delta_pos: vect3d,
    /// Measured relative rotation prev -> cur, as a matrix.
    pub delta_rot: matrix3d,
    pub delta_pos_cov_r: matrix3d,
    pub delta_pos_cov_isigma: vect3d,
    pub delta_rot_cov_r: matrix3d,
    pub delta_rot_cov_isigma: vect3d,
    pub gps: Option<GpsData>,
    /// Accelerometer tilt reading: the world up direction seen in the
    /// body frame (yaw-free by construction).
    pub tilt_g: vect3d,
    pub features: refs::Vec<PointFeature>,
}

/// Robot pose: one 6-DOF rigid transform. The optimized step is an
/// se(3) twist, so a rotation correction carries the translation with
/// it. GPS + odometry + tilt determine every pose at all ramp scales.
#[arael::model]
#[arael(constraint(hb_pose, name = "gps", guard = self.info.gps.is_some(),
    loss = |s| loss_geman_mcclure(s, path.gps_c2), {
    let raw = pose.r2w.translation - pose.info.gps.pos;
    let rt_raw = pose.info.gps.cov_r.transpose() * raw;
    [rt_raw.x * pose.info.gps.cov_isigma.x,
     rt_raw.y * pose.info.gps.cov_isigma.y,
     rt_raw.z * pose.info.gps.cov_isigma.z]
}))]
#[arael(constraint(hb_pose, name = "tilt", {
    // The accelerometer observes the world up direction in the body
    // frame -- the third row of the rotation. The raw difference of the
    // two unit vectors is the chord: its length equals the angular error
    // in radians to first order, so tilt_isigma whitens it directly.
    let d = pose.r2w.rotation_matrix.row(2) - pose.info.tilt_g;
    [d.x * path.tilt_isigma, d.y * path.tilt_isigma, d.z * path.tilt_isigma]
}))]
#[derive(Default)]
pub struct Pose {
    pub r2w: TransformParam<f64>,
    pub info: PoseInfo,
    pub hb_pose: SelfBlock<Pose>,
}

/// A 3D landmark, anchored inverse-depth parameterization: `anchor` is
/// a CONSTANT world point (the middlest pose observing the landmark,
/// snapshotted at build and re-snapshotted between ramp passes), `dir`
/// the unit direction from the anchor toward the landmark, `rho` the
/// inverse range along it. rho = 0 is a valid landmark at infinity.
/// The direction is pinned by its initializing measurement, so the
/// drift regularizer reduces to a weak prior on rho alone.
#[arael::model]
#[arael(constraint(hb_drift, name = "drift", {
    [(pointlandmark.rho - pointlandmark.rho_value) * path.drift_rho_isigma]
}))]
#[derive(Default)]
pub struct PointLandmark {
    pub anchor: vect3d,
    /// The observing pose the anchor is snapshotted from. Data only --
    /// no constraint reads it, so the anchor stays constant in the solve.
    #[arael(ref = root.poses)]
    pub anchor_pose: Ref<Pose>,
    pub dir: UnitVecParam<f64>,
    pub rho: Param<f64>,
    pub frines: std::vec::Vec<PointFrine>,
    pub hb_drift: SelfBlock<PointLandmark>,
}

/// Observation linking a landmark to a pose. The residual is the CHORD
/// between the predicted unit direction and the measured one: in the
/// feature frame the measurement is (1, 0, 0), so the chord is
/// (u.x - 1, u.y, u.z). No trig anywhere; smooth everywhere except a
/// landmark exactly at the camera.
#[arael::model]
#[arael(constraint(hb, parent=lm, name = "frine", loss = |s| branch(path.frine_cauchy,
    loss_cauchy(s, path.frine_c2), loss_geman_mcclure(s, path.frine_c2)), {
    let mr2w = pose.r2w.rotation_matrix;
    let cam_w = pose.r2w.translation + mr2w * feature.camera_pos;
    let ray_w = (lm.anchor - cam_w) * lm.rho + lm.dir.unit;
    let u = feature.mf2r.transpose() * (mr2w.transpose() * ray_w.unit());
    let sc = path.frine_isigma_scale;
    [(u.x - 1.0) * ((feature.isigma.x + feature.isigma.y) * 0.5) * sc,
     u.y * feature.isigma.x * sc,
     u.z * feature.isigma.y * sc]
}))]
#[derive(Default)]
pub struct PointFrine {
    #[arael(ref = root.poses)]
    pub pose: Ref<Pose>,
    #[arael(ref = pose.info.features)]
    pub feature: Ref<PointFeature>,
    pub hb: CrossBlock<PointLandmark, Pose>,
}

/// Odometry constraint between consecutive poses. The rotation residual
/// is the small-rotation vector of the error rotation -- no euler
/// angles anywhere in the residual.
#[arael::model]
#[arael(constraint(hb, name = "odometry", {
    let mr2w_prev = prev.r2w.rotation_matrix;
    let pos_diff = mr2w_prev.transpose() * (cur.r2w.translation - prev.r2w.translation);
    let pos_err = pos_diff - cur.info.delta_pos;
    let pos_w = cur.info.delta_pos_cov_r.transpose() * pos_err;
    let mr2w_cur = cur.r2w.rotation_matrix;
    let error_rot = (mr2w_prev * cur.info.delta_rot).transpose() * mr2w_cur;
    let rot_w = cur.info.delta_rot_cov_r.transpose() * error_rot.get_rotation_vector_small();
    [pos_w.x * cur.info.delta_pos_cov_isigma.x,
     pos_w.y * cur.info.delta_pos_cov_isigma.y,
     pos_w.z * cur.info.delta_pos_cov_isigma.z,
     rot_w.x * cur.info.delta_rot_cov_isigma.x,
     rot_w.y * cur.info.delta_rot_cov_isigma.y,
     rot_w.z * cur.info.delta_rot_cov_isigma.z]
}))]
#[derive(Default)]
pub struct PosePair {
    #[arael(ref = root.poses)]
    pub prev: Ref<Pose>,
    #[arael(ref = root.poses)]
    pub cur: Ref<Pose>,
    pub hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(root, jacobian)]
#[derive(Default)]
pub struct Path {
    pub poses: refs::Deque<Pose>,
    pub landmarks: refs::Arena<PointLandmark>,
    pub pose_pairs: std::vec::Vec<PosePair>,
    pub drift_rho_isigma: f64,
    pub tilt_isigma: f64,
    pub frine_isigma_scale: f64,
    /// Squared threshold for the feature blocks (GM 2.99, Cauchy 1.5).
    pub frine_c2: f64,
    /// Feature loss selector: > 0 Cauchy, else Geman-McClure.
    pub frine_cauchy: f64,
    /// Geman-McClure squared threshold for the GPS blocks
    /// (chi-square 0.95 quantile, 3 DOF).
    pub gps_c2: f64,
}
