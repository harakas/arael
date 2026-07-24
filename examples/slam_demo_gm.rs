// Synthetic visual-inertial SLAM demo.
//
// Generates an S-curve trajectory (default 60 poses, 240 point landmarks) at
// 5-30m distance. 5 cameras provide 360-degree coverage. Sensor data:
// GPS, wheel odometry, accelerometer tilt.
//
// Key points:
//
// - slam_demo with the feature (bearing) and GPS residuals robustified by
//   a Geman-McClure BLOCK loss (`loss = |s| loss_geman_mcclure(...)`)
//   instead of slam_demo's per-component gamma*atan(r/gamma) wrap.
//   SINGLE_PASS=1 skips the graduated isigma ramp (single solve at full
//   weight) -- and fails here: with half the observations wrong and
//   landmarks initialized outside their inlier basins, the ramp is what
//   carries them in.
// - The pose is a single TransformParam (6-DOF rigid transform): the
//   optimized step is an se(3) twist, so rotation corrections carry the
//   translation, and constraint bodies read `r2w.rotation_matrix` /
//   `r2w.translation` instead of composing position and euler-angle
//   params (slam_demo keeps the decoupled pos + ea form).
// - The ramp runs through one LmSession: only the non-param field
//   `frine_isigma_scale` changes between passes, so the sparsity analysis
//   is reused warm across all three solves.
// - 50% of feature associations are completely wrong (outlier_fraction=0.5)
//   with 30x pixel noise.
// - Feature residuals are direction chords, not angles: the predicted
//   unit direction minus the measured one in the feature frame, per-axis
//   whitened. First-order identical to bearing angles; no trig anywhere
//   in the model -- every residual is built from matrix products, sums,
//   and one normalization.
//
// - Constraint expressions in #[arael(constraint(...))] are symbolically
//   differentiated at compile time. No numeric Jacobians, no hand-coded
//   derivatives. All Hessian blocks (SelfBlock, CrossBlock) are generated
//   automatically from the symbolic constraint body.
//
// Gauge freedom and what each sensor fixes:
//
// Without absolute reference, visual SLAM has 6 degrees of gauge freedom
// (3 translation + 3 rotation) -- the solution can shift/rotate without
// changing feature reprojection costs.
//
// - Tilt sensor (accelerometer) observes the world up direction in the
//   body frame (the third row of the rotation, yaw-free), fixing roll and
//   pitch and making the path level. Without it the solution can tilt
//   arbitrarily.
//
// - GPS fixes translation and yaw (ea.z): relative GPS positions define
//   heading. Its role is whole-path orientation and large-scale
//   positioning -- the constraint covariance is inflated past the actual
//   error so it does not compete with odometry and features locally.
//
// - Odometry constrains relative motion between consecutive poses.
//
// - Landmarks use an anchored inverse-depth parameterization: a constant
//   anchor point (the middlest observing pose, re-snapshotted between ramp
//   passes), a UnitVecParam direction and an inverse range. The direction
//   is measurement-pinned at init and depth enters the bearing linearly,
//   so low-parallax landmarks stay well-conditioned; a weak prior on the
//   inverse range alone replaces the old 3-DOF drift regularizer. Poses
//   need none: GPS + odometry + tilt determine every pose at all ramp
//   scales.

use arael::covariance::{CovMode, Covariance};
use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::quatern::quaternf;
use arael::simple_lm::LmProblem;
use arael::transform::TransformParamF;
use arael::unitvec::UnitVecParamF;
use arael::vect::{vect3f, vect2f};
use arael::matrix::matrix3f;
use arael::refs::{self, Ref};
use arael::geometry::Camera;

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;

// ---------------------------------------------------------------------------
// Model structs
// ---------------------------------------------------------------------------

// A detected point feature in camera frame.
// The constraint uses mf2r/camera_pos/isigma; pixel and camera are for debugging.
#[arael::model]
#[allow(dead_code)]
struct PointFeature {
    pixel: vect2f,
    mf2r: matrix3f,       // feature-to-robot rotation: col0=view dir, col1/col2=perp axes
    #[arael(skip)]
    camera: Ref<Camera>,
    camera_pos: vect3f,    // camera position in robot frame
    isigma: vect2f,        // 1/sigma for angular residuals (rad^-1)
}

// Decomposed GPS reading: position + covariance split into R and 1/sqrt(d)
#[arael::model]
struct GpsData {
    pos: vect3f,
    cov_r: matrix3f,
    cov_isigma: vect3f,
}

#[arael::model]
struct PoseInfo {
    delta_pos: vect3f,
    /// Measured relative rotation prev -> cur, as a matrix.
    delta_rot: matrix3f,
    delta_pos_cov_r: matrix3f,
    delta_pos_cov_isigma: vect3f,
    delta_rot_cov_r: matrix3f,
    delta_rot_cov_isigma: vect3f,
    gps: Option<GpsData>,
    /// Accelerometer tilt reading: the world up direction seen in the
    /// body frame (yaw-free by construction).
    tilt_g: vect3f,
    features: refs::Vec<PointFeature>,
}

fn decompose_cov(cov: matrix3f) -> (matrix3f, vect3f) {
    let (r, d) = cov.symmetric_eigen();
    let isigma = vect3f::new(
        1.0 / d.x.sqrt(),
        1.0 / d.y.sqrt(),
        1.0 / d.z.sqrt(),
    );
    (r, isigma)
}

// Robot pose: one 6-DOF rigid transform (TransformParam). The optimized
// step is an se(3) twist, so a rotation correction carries the translation
// with it. slam_demo's pose drift regularizer is absent: the step params
// reset every accepted step (no solve-start value to anchor to), and every
// pose is fully determined by GPS + odometry + tilt at all ramp scales.
// The landmark drift regularizer below is the load-bearing one and stays.
#[arael::model]
#[arael(constraint(hb_pose, guard = self.info.gps.is_some(),
    loss = |s| loss_geman_mcclure(s, path.gps_c2), {
    let raw = pose.r2w.translation - pose.info.gps.pos;
    let rt_raw = pose.info.gps.cov_r.transpose() * raw;
    [rt_raw.x * pose.info.gps.cov_isigma.x,
     rt_raw.y * pose.info.gps.cov_isigma.y,
     rt_raw.z * pose.info.gps.cov_isigma.z]
}))]
#[arael(constraint(hb_pose, {
    // The accelerometer observes the world up direction in the body
    // frame -- the third row of the rotation. The raw difference of the
    // two unit vectors is the chord: its length equals the angular error
    // in radians to first order, so tilt_isigma whitens it directly.
    let d = pose.r2w.rotation_matrix.row(2) - pose.info.tilt_g;
    [d.x * path.tilt_isigma, d.y * path.tilt_isigma, d.z * path.tilt_isigma]
}))]
struct Pose {
    r2w: TransformParamF,
    info: PoseInfo,
    hb_pose: SelfBlock<Pose>,
}

// A 3D landmark, anchored inverse-depth parameterization: `anchor` is a
// CONSTANT world point (the middlest pose observing the landmark,
// snapshotted at build and re-snapshotted between ramp passes), `dir`
// the unit direction from the anchor toward the landmark, `rho` the
// inverse range along it. The world position is anchor + dir/rho, but
// no residual ever divides: bearings are scale-invariant, so the frine
// reads the ray rho*(anchor - cam) + dir, polynomial in the params.
// rho = 0 is a valid landmark at infinity. The direction is pinned by
// its initializing measurement, so the drift regularizer reduces to a
// weak prior on rho alone.
#[arael::model]
#[arael(constraint(hb_drift, {
    [(pointlandmark.rho - pointlandmark.rho_value) * path.drift_rho_isigma]
}))]
struct PointLandmark {
    anchor: vect3f,
    /// The observing pose the anchor is snapshotted from. Data only --
    /// no constraint reads it, so the anchor stays constant in the solve.
    #[arael(skip)]
    anchor_pose: Ref<Pose>,
    dir: UnitVecParamF,
    rho: Param<f32>,
    frines: std::vec::Vec<PointFrine>,
    hb_drift: SelfBlock<PointLandmark>,
}

// Observation linking a landmark to a pose. The residual is the CHORD
// between the predicted unit direction and the measured one: in the
// feature frame the measurement is (1, 0, 0), so the chord is
// (u.x - 1, u.y, u.z). The perpendicular components equal the atan2
// bearing angles to first order and carry the same per-axis whitening;
// the radial component is second-order for inliers (-theta^2/2) but
// keeps the antipode from being a spurious minimum (perpendicular
// components alone are a SINE residual, zero again at 180 degrees).
// No trig anywhere; smooth everywhere except a landmark exactly at the
// camera.
#[arael::model]
#[arael(constraint(hb, parent=lm, loss = |s| branch(path.frine_cauchy,
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
struct PointFrine {
    #[arael(ref = root.poses)]
    pose: Ref<Pose>,
    #[arael(ref = pose.info.features)]
    feature: Ref<PointFeature>,
    hb: CrossBlock<PointLandmark, Pose>,
}

// Odometry constraint between consecutive poses. The rotation residual is
// the small-rotation vector of the error rotation: vee of the antisymmetric
// part of error_rot (= sin(theta) * axis, the rotation vector to first
// order near identity) -- no euler angles anywhere in the residual.
#[arael::model]
#[arael(constraint(hb, {
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
struct PosePair {
    #[arael(ref = root.poses)]
    prev: Ref<Pose>,
    #[arael(ref = root.poses)]
    cur: Ref<Pose>,
    hb: CrossBlock<Pose, Pose>,
}

// Root
#[arael::model]
#[arael(root)]
struct Path {
    poses: refs::Deque<Pose>,
    landmarks: refs::Arena<PointLandmark>,
    pose_pairs: std::vec::Vec<PosePair>,
    drift_rho_isigma: f32,
    tilt_isigma: f32,
    frine_isigma_scale: f32,
    /// Squared threshold for the feature blocks. Half the 2-DOF 0.95
    /// quantile (5.99) for GM: the quantile logic assumes rare outliers,
    /// but half the observations here are outliers, and the tighter gate
    /// measures better (worst landmark 4.6m vs 11.2m). Cauchy prefers
    /// tighter still (1.5).
    frine_c2: f32,
    /// Feature loss selector: > 0 Cauchy, else Geman-McClure. The two
    /// viable losses at this contamination level (32-seed sweep): GM has
    /// the best mean, Cauchy the best worst-case bound; Huber (no
    /// redescent) and Tukey (hard cutoff, brittle) both fail here.
    frine_cauchy: f32,
    /// Geman-McClure squared threshold for the GPS blocks
    /// (chi-square 0.95 quantile, 3 DOF).
    gps_c2: f32,
}

// ---------------------------------------------------------------------------
// Synthetic data generation
// ---------------------------------------------------------------------------

struct SceneConfig {
    num_poses: usize,
    num_landmarks: usize,
    seed: u64,
    outlier_fraction: f32,
    outlier_scale: f32,
    // S-curve parameters
    s_amplitude: f32,
    s_frequency: f32,
    step_size: f32,
    // Noise parameters
    gps_sigma: f32,
    gps_sigma_inflate: f32,
    odo_pos_k: f32,    // position noise as fraction of distance
    odo_pos_base: f32, // base position noise (meters)
    odo_ea_k: f32,     // ea noise as fraction of rotation
    odo_ea_base: f32,  // base ea noise (radians)
    lm_visibility_range: usize, // landmark visible from anchor +- this many poses
    lm_visibility_prob: f32,    // probability of seeing landmark at any given pose in range
}

impl Default for SceneConfig {
    fn default() -> Self {
        SceneConfig {
            num_poses: 60,
            num_landmarks: 240,
            seed: 42,
            outlier_fraction: 0.5,  // 50% of feature associations are invalid/outliers
            outlier_scale: 30.0,    // outlier pixel noise is 30x normal (+-30 pixels)
            s_amplitude: 1.5,       // S-curve lateral amplitude (meters)
            s_frequency: 0.8,       // S-curve angular frequency
            step_size: 0.25,        // distance between poses (meters)
            gps_sigma: 0.3,         // GPS per-fix noise sigma (meters)
            gps_sigma_inflate: 2.0, // the constraint models gps_sigma *
                                    // inflate. GPS's role is whole-path
                                    // orientation and large-scale
                                    // positioning, not local motion
                                    // correction -- that is odometry's
                                    // and the features' job, and the
                                    // loose covariance keeps GPS from
                                    // competing with them locally
            odo_pos_k: 0.10,        // 10% of distance
            odo_pos_base: 0.03,     // 3cm base noise
            odo_ea_k: 0.01,
            odo_ea_base: 0.001,
            lm_visibility_range: 15,
            lm_visibility_prob: 0.75,
        }
    }
}

fn create_cameras() -> refs::Vec<Camera> {
    let mut cameras = refs::Vec::new();
    // 5 cameras at 72-degree intervals around the robot, looking toward horizon
    let w = 1024;
    let h = 768;
    let fov_deg = 80.0_f32;
    let fx = (w as f32 / 2.0) / (fov_deg / 2.0).to_radians().tan();
    let fy = fx;

    let n = 5;
    for i in 0..n {
        let yaw = (i as f32) * (360.0_f32 / n as f32).to_radians();
        let sy = yaw.sin();
        let cy = yaw.cos();
        // Camera looks outward from robot center, Z forward in camera = outward direction
        // mc2r rotates camera frame to robot frame: camera Z -> robot (cy, sy, 0)
        let mc2r = matrix3f::from_cols(
            vect3f::new(-sy, cy, 0.0),  // camera X -> robot left (perpendicular to view)
            vect3f::new(0.0, 0.0, -1.0), // camera Y -> robot down (image Y down)
            vect3f::new(cy, sy, 0.0),    // camera Z -> robot forward direction
        );
        cameras.push(Camera {
            fx, fy,
            cx: w as f32 / 2.0,
            cy: h as f32 / 2.0,
            width: w,
            height: h,
            camera_pos: vect3f::new(cy * 0.1, sy * 0.1, 0.3), // slight offset, 30cm high
            mc2r,
        });
    }
    cameras
}

fn generate_ground_truth_poses(cfg: &SceneConfig) -> Vec<(vect3f, vect3f)> {
    let mut poses = Vec::new();
    let mut t = 0.0_f32;
    for _ in 0..cfg.num_poses {
        let x = t;
        let y = cfg.s_amplitude * (cfg.s_frequency * t).sin();
        let pos = vect3f::new(x, y, 0.0);

        // Yaw follows tangent direction
        let dx = 1.0;
        let dy = cfg.s_amplitude * cfg.s_frequency * (cfg.s_frequency * t).cos();
        let yaw = dy.atan2(dx);
        let ea = vect3f::new(0.0, 0.0, yaw);

        poses.push((pos, ea));
        t += cfg.step_size;
    }
    poses
}

/// Returns (landmark_pos, anchor_pose_index) pairs.
fn generate_ground_truth_landmarks(cfg: &SceneConfig, rng: &mut StdRng, poses: &[(vect3f, vect3f)]) -> Vec<(vect3f, usize)> {
    let mut landmarks = Vec::new();
    for _ in 0..cfg.num_landmarks {
        loop {
            let anchor_idx = rng.random_range(0..poses.len());
            let anchor = &poses[anchor_idx].0;
            let angle = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
            let dist = 5.0 + rng.random::<f32>() * 25.0;
            let lm = vect3f::new(anchor.x + dist * angle.cos(), anchor.y + dist * angle.sin(), rng.random::<f32>() * 2.0);
            let min_dist = poses.iter()
                .map(|(p, _)| (lm - *p).norm())
                .fold(f32::MAX, f32::min);
            if min_dist >= 5.0 && min_dist <= 30.0 {
                landmarks.push((lm, anchor_idx));
                break;
            }
        }
    }
    landmarks
}

fn build_path(cfg: &SceneConfig) -> (Path, Vec<(vect3f, vect3f)>, Vec<(vect3f, usize)>) {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let normal01 = Normal::new(0.0, 1.0).unwrap();

    let gt_poses = generate_ground_truth_poses(cfg);
    let gt_landmarks = generate_ground_truth_landmarks(cfg, &mut rng, &gt_poses);
    let cameras = create_cameras();

    // Weak prior on each landmark's inverse range (1/m units): holds
    // all-outlier landmarks at their initial ray instead of letting them
    // wander; the direction needs none (pinned by its initializer).
    let drift_rho_sigma: f32 = 1.0;
    let tilt_sigma_deg: f32 = 0.25;         // accelerometer accuracy in degrees
    let tilt_sigma_rad = tilt_sigma_deg.to_radians();

    let mut path = Path {
        poses: refs::Deque::new(),
        landmarks: refs::Arena::new(),
        pose_pairs: std::vec::Vec::new(),
        drift_rho_isigma: 1.0 / drift_rho_sigma,
        tilt_isigma: 1.0 / tilt_sigma_rad,
        frine_isigma_scale: 1.0,
        frine_c2: 2.99,
        frine_cauchy: -1.0,
        gps_c2: 7.815,
    };

    // (landmark index, observing-pose index, feature ref). The pose ref is
    // resolved after all poses are built (below); storing the index here
    // avoids referencing a pose that has not been pushed yet.
    let mut frine_data: std::vec::Vec<(usize, usize, Ref<PointFeature>)> = std::vec::Vec::new();

    for (pi, &(pos, ea)) in gt_poses.iter().enumerate() {
        let mr2w = matrix3f::rotation_from_euler_angles(ea);

        // Compute odometry deltas
        let (delta_pos, delta_rot) = if pi == 0 {
            (vect3f::new(0.0, 0.0, 0.0), matrix3f::identity())
        } else {
            let (prev_pos, prev_ea) = gt_poses[pi - 1];
            let prev_mr2w = matrix3f::rotation_from_euler_angles(prev_ea);
            let prev_mw2r = prev_mr2w.transpose();
            let dp = prev_mw2r * (pos - prev_pos);
            (dp, prev_mw2r * mr2w)
        };

        // Odometry covariance proportional to motion; the rotation angle
        // comes from the trace identity cos(theta) = (tr - 1) / 2.
        let dp_norm = delta_pos.norm().max(0.01);
        let tr = delta_rot[0].x + delta_rot[1].y + delta_rot[2].z;
        let de_norm = ((tr - 1.0) * 0.5).clamp(-1.0, 1.0).acos().max(0.001);
        let pos_sigma = vect3f::new(
            cfg.odo_pos_k * dp_norm + cfg.odo_pos_base,
            (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5, // lateral less noisy than forward
            (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5,
        );
        let rot_sigma = vect3f::new(
            cfg.odo_ea_k * de_norm + cfg.odo_ea_base,
            cfg.odo_ea_k * de_norm + cfg.odo_ea_base,
            cfg.odo_ea_k * de_norm + cfg.odo_ea_base,
        );

        let delta_pos_cov = matrix3f::from_elements(
            pos_sigma.x * pos_sigma.x, 0.0, 0.0,
            0.0, pos_sigma.y * pos_sigma.y, 0.0,
            0.0, 0.0, pos_sigma.z * pos_sigma.z,
        );
        let delta_rot_cov = matrix3f::from_elements(
            rot_sigma.x * rot_sigma.x, 0.0, 0.0,
            0.0, rot_sigma.y * rot_sigma.y, 0.0,
            0.0, 0.0, rot_sigma.z * rot_sigma.z,
        );

        // Generate features (only for landmarks visible from this pose)
        let mut features: refs::Vec<PointFeature> = refs::Vec::new();
        for (li, &(lm_pos, anchor_idx)) in gt_landmarks.iter().enumerate() {
            // Skip landmarks too far from this pose (anchor-based visibility)
            let dist_to_anchor = if pi >= anchor_idx { pi - anchor_idx } else { anchor_idx - pi };
            if dist_to_anchor > cfg.lm_visibility_range { continue; }
            // Random visibility within range
            if rng.random::<f32>() > cfg.lm_visibility_prob { continue; }
            for cam_ref in cameras.refs() {
                let cam = &cameras[cam_ref];
                let p_cam = cam.world_to_camera(lm_pos, pos, mr2w);
                if p_cam.z < 0.5 { continue; } // behind camera or too close
                let pixel = cam.project(p_cam);
                if !cam.is_visible(pixel) { continue; }

                // Add pixel noise (uniform +-1 pixel, outliers get scaled up)
                let is_outlier = rng.random::<f32>() < cfg.outlier_fraction;
                let noise_scale = if is_outlier { cfg.outlier_scale } else { 1.0 };
                let noisy_pixel = vect2f::new(
                    pixel.x + noise_scale * (rng.random::<f32>() * 2.0 - 1.0),
                    pixel.y + noise_scale * (rng.random::<f32>() * 2.0 - 1.0),
                );
                // Build feature-to-robot frame (mf2r): col0 = view direction from
                // pose toward feature, col1/col2 = perpendicular axes for measuring
                // horizontal and vertical angular error.
                let dir = cam.unproject_to_robot(noisy_pixel);
                let cam_up = -(cam.mc2r.col(1));
                let up_proj = cam_up - dir * (cam_up * dir);
                let up_norm = up_proj.norm();
                if up_norm < 1e-6 { continue; }
                let col2 = up_proj * (1.0 / up_norm);
                let col1 = col2 % dir;
                let mf2r = matrix3f::from_cols(dir, col1, col2);

                let sigma = cam.pixel_angular_size(noisy_pixel);
                let isigma = vect2f::new(1.0 / sigma.x, 1.0 / sigma.y);

                let feat_ref = features.push(PointFeature {
                    pixel: noisy_pixel,
                    mf2r,
                    camera: cam_ref,
                    camera_pos: cam.camera_pos,
                    isigma,
                });
                frine_data.push((li, pi, feat_ref));
            }
        }

        // GPS: iid per-fix noise; the constraint covariance is inflated
        // past the actual error (see gps_sigma_inflate).
        let gps_pos = vect3f::new(
            pos.x + cfg.gps_sigma * rng.sample(normal01) as f32,
            pos.y + cfg.gps_sigma * rng.sample(normal01) as f32,
            pos.z + cfg.gps_sigma * rng.sample(normal01) as f32,
        );
        let ms = cfg.gps_sigma * cfg.gps_sigma_inflate;
        let gps_cov = matrix3f::from_elements(
            ms * ms, 0.0, 0.0,
            0.0, ms * ms, 0.0,
            0.0, 0.0, ms * ms,
        );

        // Noisy initial pose estimate
        let init_noise_pos = 0.1_f32;   // meters
        let init_noise_ea = 0.02_f32;   // radians (~1.1 degrees)
        let noisy_pos = vect3f::new(
            pos.x + init_noise_pos * rng.sample(normal01) as f32,
            pos.y + init_noise_pos * rng.sample(normal01) as f32,
            pos.z + init_noise_pos * rng.sample(normal01) as f32,
        );
        let noisy_ea = vect3f::new(
            ea.x + init_noise_ea * rng.sample(normal01) as f32,
            ea.y + init_noise_ea * rng.sample(normal01) as f32,
            ea.z + init_noise_ea * rng.sample(normal01) as f32,
        );

        let (delta_pos_cov_r, delta_pos_cov_isigma) = decompose_cov(delta_pos_cov);
        let (delta_rot_cov_r, delta_rot_cov_isigma) = decompose_cov(delta_rot_cov);
        let (gps_cov_r, gps_cov_isigma) = decompose_cov(gps_cov);

        path.poses.push_back(Pose {
            r2w: TransformParamF::new(noisy_pos, quaternf::from_euler_angles(noisy_ea)),
            info: PoseInfo {
                delta_pos, delta_rot,
                delta_pos_cov_r, delta_pos_cov_isigma,
                delta_rot_cov_r, delta_rot_cov_isigma,
                gps: Some(GpsData { pos: gps_pos, cov_r: gps_cov_r, cov_isigma: gps_cov_isigma }),
                tilt_g: {
                    // Sensor noise lives in angle space (roll/pitch); the
                    // reading is stored as the up direction it implies:
                    // (-sin p, cos p sin r, cos p cos r), row 2 of R.
                    let r = ea.x + tilt_sigma_rad * rng.sample(normal01) as f32;
                    let p = ea.y + tilt_sigma_rad * rng.sample(normal01) as f32;
                    vect3f::new(-p.sin(), p.cos() * r.sin(), p.cos() * r.cos())
                },
                features,
            },
            hb_pose: SelfBlock::new(),
        });
    }

    // Every pose is built; capture their handles to wire up frines and
    // odometry.
    let pose_refs: std::vec::Vec<Ref<Pose>> = path.poses.refs().collect();

    // Build landmarks with frines. The anchor is the middlest observing
    // pose (median of the observing pose indices), snapshotted at its
    // initial position; direction and inverse range initialize from the
    // noisy landmark guess.
    for (li, &(lm_pos, _)) in gt_landmarks.iter().enumerate() {
        let noisy_lm = vect3f::new(
            lm_pos.x + 0.5 * rng.sample(normal01) as f32,
            lm_pos.y + 0.5 * rng.sample(normal01) as f32,
            lm_pos.z + 0.3 * rng.sample(normal01) as f32,
        );
        let obs: std::vec::Vec<usize> = frine_data.iter()
            .filter(|(lmi, _, _)| *lmi == li)
            .map(|(_, pose_i, _)| *pose_i)
            .collect();
        if obs.is_empty() { continue; } // skip landmarks with no observations
        let frines: std::vec::Vec<PointFrine> = frine_data.iter()
            .filter(|(lmi, _, _)| *lmi == li)
            .map(|(_, pose_i, feature)| PointFrine { pose: pose_refs[*pose_i], feature: *feature, hb: CrossBlock::new() })
            .collect();
        let anchor_pose = pose_refs[obs[obs.len() / 2]];
        let anchor = path.poses[anchor_pose].r2w.translation;
        let d = noisy_lm - anchor;
        path.landmarks.push(PointLandmark {
            anchor,
            anchor_pose,
            dir: UnitVecParamF::new(d),
            rho: Param::new(1.0 / d.norm()),
            frines,
            hb_drift: SelfBlock::new(),
        });
    }

    // Build pose pairs for odometry
    for i in 1..pose_refs.len() {
        path.pose_pairs.push(PosePair {
            prev: pose_refs[i - 1],
            cur: pose_refs[i],
            hb: CrossBlock::new(),
        });
    }

    (path, gt_poses, gt_landmarks)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn print_usage() {
    eprintln!("Usage: slam_demo [OPTIONS]");
    eprintln!("  --solver <dense|faer|eigen|cholmod>  (default: faer)");
    eprintln!("  --loss <gm|cauchy>                   (default: gm)");
    eprintln!("  --poses <N>                          (default: 60)");
    eprintln!("  --landmarks <N>                      (default: 240)");
    eprintln!("  --seed <N>                           (default: 42)");
}

// Env-gated Hessian sparsity bitmap: assemble J^T J at the current estimate and
// write its nonzero (fill) pattern as an image -- one pixel per (row, col),
// black where the Hessian has an entry, mirrored to the full symmetric matrix.
// Enabled by SLAM_HESSIAN_BITMAP=<path.png> (or =1 for the default hessian.png).
fn write_hessian_bitmap(path: &mut Path, out: &str) {
    let mut params: std::vec::Vec<f64> = std::vec::Vec::new();
    path.serialize64(&mut params);
    let n = params.len();
    let mut grad = vec![0.0f64; n];
    let mut coo = arael::simple_lm::CooMatrix::new(n);
    path.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
    let mut img = image::GrayImage::from_pixel(n as u32, n as u32, image::Luma([255u8]));
    for (&r, &c) in coo.rows.iter().zip(coo.cols.iter()) {
        img.put_pixel(c, r, image::Luma([0])); // (col, row) = (x, y)
        img.put_pixel(r, c, image::Luma([0])); // mirror the upper triangle to full
    }
    match img.save(out) {
        Ok(()) => println!("Hessian fill bitmap: {n}x{n}, {} nonzeros -> {out}", coo.nnz()),
        Err(e) => eprintln!("Hessian bitmap write failed ({out}): {e}"),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut solver_name = "faer".to_string();
    let mut loss_name = "gm".to_string();
    let mut num_poses: Option<usize> = None;
    let mut num_landmarks: Option<usize> = None;
    let mut seed: Option<u64> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--solver" => { i += 1; solver_name = args.get(i).cloned().unwrap_or_default(); }
            "--loss" => { i += 1; loss_name = args.get(i).cloned().unwrap_or_default(); }
            "--poses" => { i += 1; num_poses = args.get(i).and_then(|s| s.parse().ok()); }
            "--landmarks" => { i += 1; num_landmarks = args.get(i).and_then(|s| s.parse().ok()); }
            "--seed" => { i += 1; seed = args.get(i).and_then(|s| s.parse().ok()); }
            "--help" | "-h" => { print_usage(); return; }
            other => { eprintln!("Unknown argument: {}", other); print_usage(); return; }
        }
        i += 1;
    }

    let mut cfg = SceneConfig::default();
    if let Some(p) = num_poses { cfg.num_poses = p; }
    if let Some(l) = num_landmarks { cfg.num_landmarks = l; }
    if let Some(s) = seed { cfg.seed = s; }

    println!("Solver: {}  Loss: {}  Poses: {}  Landmarks: {}  Seed: {}",
        solver_name, loss_name, cfg.num_poses, cfg.num_landmarks, cfg.seed);
    let (mut path, gt_poses, gt_landmarks) = build_path(&cfg);

    // Each family's measured-best threshold (32-seed sweep).
    match loss_name.as_str() {
        "gm" => { path.frine_cauchy = -1.0; path.frine_c2 = 2.99; }
        "cauchy" => { path.frine_cauchy = 1.0; path.frine_c2 = 1.5; }
        other => { eprintln!("Unknown loss: {}. Available: gm, cauchy", other); return; }
    }

    let mut params = std::vec::Vec::new();
    path.serialize32(&mut params);

    let n_frines: usize = path.landmarks.iter().map(|lm| lm.frines.len()).sum();
    println!("Path: {} poses, {} landmarks, {} frines, {} pose_pairs",
        path.poses.len(), path.landmarks.len(), n_frines, path.pose_pairs.len());
    println!("Parameters: {} (Pose={}, Landmark={})",
        params.len(), Pose::PARAM_COUNT, PointLandmark::PARAM_COUNT);
    println!();

    // Print a few poses
    for i in [0, cfg.num_poses / 2, cfg.num_poses - 1] {
        if i >= path.poses.len() { continue; }
        let pose = &path.poses[i];
        let (gt_p, gt_e) = gt_poses[i];
        let ea = pose.r2w.rotation.get_euler_angles();
        println!("Pose {:2}: pos=({:7.3}, {:7.3}, {:7.3}) ea=({:7.4}, {:7.4}, {:7.4})",
            i, pose.r2w.translation.x, pose.r2w.translation.y, pose.r2w.translation.z,
            ea.x, ea.y, ea.z);
        println!("      gt: pos=({:7.3}, {:7.3}, {:7.3}) ea=({:7.4}, {:7.4}, {:7.4})",
            gt_p.x, gt_p.y, gt_p.z, gt_e.x, gt_e.y, gt_e.z);
    }
    println!();

    // Hessian sparsity bitmap (eyeball the fill), env-gated.
    if let Ok(v) = std::env::var("SLAM_HESSIAN_BITMAP") {
        let out = if v == "1" || v.is_empty() { "hessian.png".to_string() } else { v };
        write_hessian_bitmap(&mut path, &out);
    }

    // Graduated optimization: start with loose feature constraints, tighten.
    // Between passes only non-param fields and param VALUES change (the
    // isigma scale, and the landmark re-anchoring), so one LmSession
    // carries the solves: the sparsity analysis from pass 1 is reused
    // warm by the later passes.
    println!("--- Optimization ---");
    let isigma_scales: std::vec::Vec<f32> = if std::env::var("SINGLE_PASS").is_ok() {
        vec![1.0]
    } else {
        vec![0.01, 0.1, 1.0]
    };

    fn run_ramp<S: arael::simple_lm::LmSolver<f64>>(
        solver: S, path: &mut Path, scales: &[f32],
    ) {
        let mut session = arael::simple_lm::LmSession::new(solver);
        for (pass, &scale) in scales.iter().enumerate() {
            path.frine_isigma_scale = scale;
            println!("\nPass {} (isigma scale={}):", pass + 1, scale);
            let config = arael::simple_lm::LmConfig::well_conditioned()
                .with_verbose(true)
                .with_rel_precision(1e-6);
            let result = session.solve(path, &config);
            println!("  {} iterations, cost {:.4} -> {:.4}",
                result.iterations, result.start_cost, result.end_cost);
            if pass + 1 < scales.len() {
                reanchor_landmarks(path);
            }
        }
    }

    // Move each landmark's anchor to its anchor pose's CURRENT position
    // and re-express direction + inverse range there, so the anchor stays
    // near the rays as the poses converge. Values only -- the session's
    // structure is untouched. Landmarks at (near) infinity keep their ray.
    fn reanchor_landmarks(path: &mut Path) {
        let (poses, landmarks) = (&path.poses, &mut path.landmarks);
        for lm in landmarks.iter_mut() {
            if lm.rho.value.abs() < 1e-4 { continue; }
            let world = lm.anchor + lm.dir.unit * (1.0 / lm.rho.value);
            let c_new = poses[lm.anchor_pose].r2w.translation;
            let d = world - c_new;
            let n = d.norm();
            if n < 1e-3 { continue; }
            lm.anchor = c_new;
            lm.dir.unit = d * (1.0 / n);
            lm.rho.value = 1.0 / n;
        }
    }

    match solver_name.as_str() {
        "dense" => run_ramp(arael::simple_lm::Dense, &mut path, &isigma_scales),
        "faer" => run_ramp(arael::simple_lm::SparseFaer::new(), &mut path, &isigma_scales),
        #[cfg(feature = "eigen")]
        "eigen" => run_ramp(arael::simple_lm::SparseEigen::new(), &mut path, &isigma_scales),
        #[cfg(not(feature = "eigen"))]
        "eigen" => { eprintln!("Eigen solver requires --features eigen"); return; }
        #[cfg(feature = "cholmod")]
        "cholmod" => run_ramp(arael::simple_lm::SparseCholmod::new(), &mut path, &isigma_scales),
        #[cfg(not(feature = "cholmod"))]
        "cholmod" => { eprintln!("CHOLMOD solver requires --features cholmod"); return; }
        _ => { eprintln!("Unknown solver: {}. Available: dense, faer, eigen, cholmod", solver_name); return; }
    }

    // Mean absolute pose error vs GT
    {
        let mut pos_err_sum = 0.0_f32;
        let mut ea_err_sum = 0.0_f32;
        let n = gt_poses.len().min(path.poses.len());
        for i in 0..n {
            let pose = &path.poses[i];
            let (gt_p, gt_e) = gt_poses[i];
            pos_err_sum += (pose.r2w.translation - gt_p).norm();
            ea_err_sum += (pose.r2w.rotation.get_euler_angles() - gt_e).norm();
        }
        let mut params64: std::vec::Vec<f64> = std::vec::Vec::new();
        path.serialize64(&mut params64);
        let cost = path.calc_cost(&params64);
        println!("\nFinal cost: {:.4}", cost);
        println!("Mean pose error vs GT: pos={:.4}m  ea={:.3}deg",
            pos_err_sum / n as f32, (ea_err_sum / n as f32).to_degrees());
    }

    // Relative pose errors: compare consecutive delta_pos in local frame
    println!("\n--- Relative pose errors ---");
    let mut dpos_errs: std::vec::Vec<f32> = std::vec::Vec::new();
    let mut dpos_rel_errs: std::vec::Vec<f32> = std::vec::Vec::new();
    let mut dea_errs_deg: std::vec::Vec<f32> = std::vec::Vec::new();
    let mut dea_rel_errs: std::vec::Vec<f32> = std::vec::Vec::new();
    for i in 1..gt_poses.len().min(path.poses.len()) {
        let prev = &path.poses[i - 1];
        let pose = &path.poses[i];
        let (gt_prev_pos, gt_prev_ea) = gt_poses[i - 1];
        let (gt_cur_pos, gt_cur_ea) = gt_poses[i];

        // GT delta_pos in previous pose's local frame
        let gt_mr2w = matrix3f::rotation_from_euler_angles(gt_prev_ea);
        let gt_delta_pos = gt_mr2w.transpose() * (gt_cur_pos - gt_prev_pos);

        // Optimized delta_pos in previous pose's local frame
        let opt_mr2w_prev = prev.r2w.rotation_matrix;
        let opt_delta_pos = opt_mr2w_prev.transpose() * (pose.r2w.translation - prev.r2w.translation);

        let dpos_err = (opt_delta_pos - gt_delta_pos).norm();
        let gt_step = gt_delta_pos.norm();
        let dpos_rel = if gt_step > 1e-6 { 100.0 * dpos_err / gt_step } else { 0.0 };

        // GT delta_ea: relative rotation from prev to cur
        let gt_mr2w_cur = matrix3f::rotation_from_euler_angles(gt_cur_ea);
        let gt_delta_ea = (gt_mr2w.transpose() * gt_mr2w_cur).get_euler_angles();

        // Optimized delta_ea
        let opt_mr2w_cur = pose.r2w.rotation_matrix;
        let opt_delta_ea = (opt_mr2w_prev.transpose() * opt_mr2w_cur).get_euler_angles();

        let dea_err = (opt_delta_ea - gt_delta_ea).norm();
        let dea_err_deg = dea_err.to_degrees();
        let gt_rot = gt_delta_ea.norm();
        let dea_rel = if gt_rot > 1e-6 { 100.0 * dea_err / gt_rot } else { 0.0 };

        println!("Pair {:2}-{:2}: dpos={:.4}m ({:.1}%)  dea={:.3}deg ({:.1}%)",
            i - 1, i, dpos_err, dpos_rel, dea_err_deg, dea_rel);
        dpos_errs.push(dpos_err);
        dpos_rel_errs.push(dpos_rel);
        dea_errs_deg.push(dea_err_deg);
        dea_rel_errs.push(dea_rel);
    }
    dpos_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dpos_rel_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dea_errs_deg.sort_by(|a, b| a.partial_cmp(b).unwrap());
    dea_rel_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !dpos_errs.is_empty() {
        let n = dpos_errs.len();
        let mean: f32 = dpos_errs.iter().sum::<f32>() / n as f32;
        println!("Delta pos: mean={:.4}m  median={:.4}m  min={:.4}m  max={:.4}m",
            mean, dpos_errs[n / 2], dpos_errs[0], dpos_errs[n - 1]);
        let mean: f32 = dpos_rel_errs.iter().sum::<f32>() / n as f32;
        println!("Delta pos: mean={:.2}%  median={:.2}%  min={:.2}%  max={:.2}%",
            mean, dpos_rel_errs[n / 2], dpos_rel_errs[0], dpos_rel_errs[n - 1]);
        let mean: f32 = dea_errs_deg.iter().sum::<f32>() / n as f32;
        println!("Delta ea:  mean={:.3}deg  median={:.3}deg  min={:.3}deg  max={:.3}deg",
            mean, dea_errs_deg[n / 2], dea_errs_deg[0], dea_errs_deg[n - 1]);
        let mean: f32 = dea_rel_errs.iter().sum::<f32>() / n as f32;
        println!("Delta ea:  mean={:.2}%  median={:.2}%  min={:.2}%  max={:.2}%",
            mean, dea_rel_errs[n / 2], dea_rel_errs[0], dea_rel_errs[n - 1]);
    }

    // Landmark uncertainty from the parameter covariance (Sigma = 2 H^-1). The
    // relative covariance Cov_rel = C_ll + C_pp - C_lp - C_pl over the landmark and
    // pose POSITION blocks cancels the shared gauge uncertainty, giving
    // uncertainty relative to the pose. Ellipsoid semi-axes = sqrt of its eigenvalues.
    let cov = match path.assemble_covariance(CovMode::AllMarginals) {
        Ok(c) => Some(c),
        Err(e) => { println!("Covariance unavailable: {e}"); None }
    };

    // Landmark errors: compare landmark-to-closest-pose vector (opt vs GT)
    println!("\n--- Landmark errors (relative to closest pose) ---");
    let mut lm_errs: std::vec::Vec<f32> = std::vec::Vec::new();
    let mut lm_rel_errs: std::vec::Vec<f32> = std::vec::Vec::new();
    let mut max_sigmas: std::vec::Vec<f64> = std::vec::Vec::new();
    for (i, (&(gt_lm, _anchor), lm)) in gt_landmarks.iter().zip(path.landmarks.iter()).enumerate() {
        // Find closest GT pose
        let (closest_idx, _) = gt_poses.iter().enumerate()
            .map(|(j, (p, _))| (j, (gt_lm - *p).norm()))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let (gt_pose_pos, gt_pose_ea) = gt_poses[closest_idx];
        let gt_mr2w = matrix3f::rotation_from_euler_angles(gt_pose_ea);
        let gt_vec = gt_mr2w.transpose() * (gt_lm - gt_pose_pos);
        // Optimized vector from closest opt pose to opt landmark in opt pose's local frame
        let opt_pose = &path.poses[closest_idx];
        let opt_mr2w = opt_pose.r2w.rotation_matrix;
        let lm_world = lm.anchor + lm.dir.unit * (1.0 / lm.rho.value);
        let opt_vec = opt_mr2w.transpose() * (lm_world - opt_pose.r2w.translation);
        let err = (opt_vec - gt_vec).norm();
        let gt_dist = gt_vec.norm();
        let rel_pct = 100.0 * err / gt_dist;

        // The landmark marginal is [dir chart (2); rho]; map it to world
        // position covariance with J = [unit_d / rho, -unit / rho^2] (the
        // anchor is constant data). Near-infinity landmarks (rho ~ 0)
        // have no finite position covariance and print without sigma.
        // The pose marginal is TransformParam's [w (rotation); d
        // (translation)], d in the reference rotation frame: the world
        // translation block is R * C_dd * R^T.
        // A query can fail (singular marginal); such a landmark prints
        // without a sigma, same as the near-infinity case.
        let sigmas = cov.as_ref().filter(|_| lm.rho.value.abs() >= 1e-4).and_then(|cov| {
            let rho = lm.rho.value as f64;
            let (u, ud) = (lm.dir.unit, lm.dir.unit_d);
            let j = nalgebra::DMatrix::from_row_slice(3, 3, &[
                ud[0].x as f64 / rho, ud[1].x as f64 / rho, -(u.x as f64) / (rho * rho),
                ud[0].y as f64 / rho, ud[1].y as f64 / rho, -(u.y as f64) / (rho * rho),
                ud[0].z as f64 / rho, ud[1].z as f64 / rho, -(u.z as f64) / (rho * rho),
            ]);
            let c_ll = &j * cov.marginal_cov(lm).ok()? * j.transpose();
            let m = opt_mr2w;
            let r = nalgebra::DMatrix::from_row_slice(3, 3, &[
                m[0].x as f64, m[0].y as f64, m[0].z as f64,
                m[1].x as f64, m[1].y as f64, m[1].z as f64,
                m[2].x as f64, m[2].y as f64, m[2].z as f64,
            ]);
            let c_dd = cov.marginal_cov(opt_pose).ok()?.view((3, 3), (3, 3)).into_owned();
            let c_pp = &r * c_dd * r.transpose();
            let c_lp = &j * cov.cross_cov(lm, opt_pose).ok()?.view((0, 3), (3, 3)).into_owned()
                * r.transpose();
            let cov_rel = &c_ll + &c_pp - &c_lp - c_lp.transpose();
            let eigen = nalgebra::SymmetricEigen::new(cov_rel);
            let mut sg = [
                eigen.eigenvalues[0].max(0.0).sqrt(),
                eigen.eigenvalues[1].max(0.0).sqrt(),
                eigen.eigenvalues[2].max(0.0).sqrt(),
            ];
            sg.sort_by(|a, b| b.partial_cmp(a).unwrap());
            Some(sg)
        });
        if let Some(sg) = sigmas {
            println!("LM {:3}: |d|={:.3}m  rel={:.2}%  dist={:.1}m  sigma=({:.3},{:.3},{:.3})m  frines={}",
                i, err, rel_pct, gt_dist, sg[0], sg[1], sg[2], lm.frines.len());
            max_sigmas.push(sg[0]);
        } else {
            println!("LM {:3}: |d|={:.3}m  rel={:.2}%  dist={:.1}m  frines={}",
                i, err, rel_pct, gt_dist, lm.frines.len());
        }

        lm_errs.push(err);
        lm_rel_errs.push(rel_pct);
    }
    lm_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    lm_rel_errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !lm_errs.is_empty() {
        let n = lm_errs.len();
        let mean: f32 = lm_errs.iter().sum::<f32>() / n as f32;
        println!("LM pos:  mean={:.3}m  median={:.3}m  min={:.3}m  max={:.3}m",
            mean, lm_errs[n / 2], lm_errs[0], lm_errs[n - 1]);
        let mean: f32 = lm_rel_errs.iter().sum::<f32>() / n as f32;
        println!("LM rel:  mean={:.2}%  median={:.2}%  min={:.2}%  max={:.2}%",
            mean, lm_rel_errs[n / 2], lm_rel_errs[0], lm_rel_errs[n - 1]);
    }
    max_sigmas.sort_by(|a, b| a.partial_cmp(b).unwrap());
    if !max_sigmas.is_empty() {
        let nm = max_sigmas.len();
        let mean: f64 = max_sigmas.iter().sum::<f64>() / nm as f64;
        println!("Max principal sigma: mean={:.3}m  median={:.3}m  min={:.3}m  max={:.3}m",
            mean, max_sigmas[nm / 2], max_sigmas[0], max_sigmas[nm - 1]);
    }

}
