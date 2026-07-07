// Synthetic visual-inertial SLAM demo.
//
// Generates an S-curve trajectory (default 60 poses, 240 point landmarks) at
// 5-30m distance. 5 cameras provide 360-degree coverage. Sensor data:
// GPS (with systematic offset), wheel odometry, accelerometer tilt.
//
// Key points:
//
// - 50% of feature associations are completely wrong (outlier_fraction=0.5)
//   with 30x pixel noise. The robust gamma*atan(r/gamma) suppression handles
//   this via graduated optimization (scaling feature isigma from 1% to 100%).
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
// - Tilt sensor (accelerometer) fixes roll (ea.x) and pitch (ea.y), making
//   the path level. Without it the solution can tilt arbitrarily.
//
// - GPS fixes translation and yaw (ea.z). Even with a systematic offset
//   (all readings biased by ~2.5m), it still constrains yaw because the
//   offset is constant while poses move -- relative GPS positions define
//   heading. The systematic offset just shifts everything, which is why
//   mean pose error vs GT roughly equals GPS offset magnitude.
//
// - Odometry constrains relative motion between consecutive poses.
//
// - Drift constraints are weak regularizers (sigma=1000m pos, 1800deg ea)
//   preventing parameters from diverging during early passes when feature
//   constraints are scaled down.

use arael::model::{Model, Param, SimpleEulerAngleParam, SelfBlock, CrossBlock};
use arael::simple_lm::LmProblem;
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
    delta_ea: vect3f,
    delta_pos_cov_r: matrix3f,
    delta_pos_cov_isigma: vect3f,
    delta_ea_cov_r: matrix3f,
    delta_ea_cov_isigma: vect3f,
    gps: Option<GpsData>,
    // Accelerometer tilt reading (roll and pitch)
    tilt_roll: f32,
    tilt_pitch: f32,
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

// Robot pose
#[arael::model]
#[arael(constraint(hb_pose, guard = self.info.gps.is_some(), {
    let gamma = path.gamma;
    let raw = pose.pos - pose.info.gps.pos;
    let rt_raw = pose.info.gps.cov_r.transpose() * raw;
    let p0 = rt_raw.x * pose.info.gps.cov_isigma.x;
    let p1 = rt_raw.y * pose.info.gps.cov_isigma.y;
    let p2 = rt_raw.z * pose.info.gps.cov_isigma.z;
    [gamma * atan(p0 / gamma),
     gamma * atan(p1 / gamma),
     gamma * atan(p2 / gamma)]
}))]
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
struct Pose {
    pos: Param<vect3f>,
    ea: SimpleEulerAngleParam<f32>,
    info: PoseInfo,
    hb_pose: SelfBlock<Pose>,
}

// A 3D landmark observed from multiple poses
#[arael::model]
#[arael(constraint(hb_drift, {
    let drift = pointlandmark.pos - pointlandmark.pos_value;
    [drift.x * path.drift_lm_isigma, drift.y * path.drift_lm_isigma, drift.z * path.drift_lm_isigma]
}))]
struct PointLandmark {
    pos: Param<vect3f>,
    frines: std::vec::Vec<PointFrine>,
    hb_drift: SelfBlock<PointLandmark>,
}

// Observation linking a landmark to a pose
#[arael::model]
#[arael(constraint(hb, parent=lm, {
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
struct PointFrine {
    #[arael(ref = root.poses)]
    pose: Ref<Pose>,
    #[arael(ref = pose.info.features)]
    feature: Ref<PointFeature>,
    hb: CrossBlock<PointLandmark, Pose>,
}

// Odometry constraint between consecutive poses
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
    gamma: f32,
    drift_pos_isigma: f32,
    drift_ea_isigma: f32,
    drift_lm_isigma: f32,
    tilt_isigma: f32,
    frine_isigma_scale: f32,
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
    gps_rel_noise: f32, // GPS relative noise (meters per meter from origin)
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
            gps_sigma: 2.5,         // GPS systematic offset sigma (meters)
            gps_rel_noise: 0.02,    // GPS relative noise (meters per meter from origin)
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

fn build_path(cfg: &SceneConfig) -> (Path, Vec<(vect3f, vect3f)>, Vec<(vect3f, usize)>, vect3f) {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let normal01 = Normal::new(0.0, 1.0).unwrap();

    // Systematic GPS offset (same for all poses, ~2.5m random direction)
    let gps_offset = vect3f::new(
        cfg.gps_sigma * rng.sample(normal01) as f32,
        cfg.gps_sigma * rng.sample(normal01) as f32,
        cfg.gps_sigma * rng.sample(normal01) as f32,
    );

    let gt_poses = generate_ground_truth_poses(cfg);
    let gt_landmarks = generate_ground_truth_landmarks(cfg, &mut rng, &gt_poses);
    let cameras = create_cameras();

    let drift_pos_sigma: f32 = 1000.0;    // meters
    let drift_ea_sigma_deg: f32 = 1800.0;  // degrees
    let drift_lm_sigma: f32 = 1000.0;      // meters
    let tilt_sigma_deg: f32 = 0.25;         // accelerometer accuracy in degrees
    let tilt_sigma_rad = tilt_sigma_deg.to_radians();

    let mut path = Path {
        poses: refs::Deque::new(),
        landmarks: refs::Arena::new(),
        pose_pairs: std::vec::Vec::new(),
        // Robust suppression: gamma*atan(r/gamma). Residuals up to ~gamma pass
        // linearly, beyond that they saturate. With 25 expected inlier residuals,
        // gamma ~= 2*sqrt(25)/pi ~= 3.18.
        gamma: 2.0 * (25.0_f32).sqrt() / std::f32::consts::PI,
        drift_pos_isigma: 1.0 / drift_pos_sigma,
        drift_ea_isigma: 1.0 / drift_ea_sigma_deg.to_radians(),
        drift_lm_isigma: 1.0 / drift_lm_sigma,
        tilt_isigma: 1.0 / tilt_sigma_rad,
        frine_isigma_scale: 1.0,
    };

    // (landmark index, observing-pose index, feature ref). The pose ref is
    // resolved after all poses are built (below); storing the index here
    // avoids referencing a pose that has not been pushed yet.
    let mut frine_data: std::vec::Vec<(usize, usize, Ref<PointFeature>)> = std::vec::Vec::new();

    for (pi, &(pos, ea)) in gt_poses.iter().enumerate() {
        let mr2w = matrix3f::rotation_from_euler_angles(ea);

        // Compute odometry deltas
        let (delta_pos, delta_ea) = if pi == 0 {
            (vect3f::new(0.0, 0.0, 0.0), vect3f::new(0.0, 0.0, 0.0))
        } else {
            let (prev_pos, prev_ea) = gt_poses[pi - 1];
            let prev_mr2w = matrix3f::rotation_from_euler_angles(prev_ea);
            let prev_mw2r = prev_mr2w.transpose();
            let dp = prev_mw2r * (pos - prev_pos);
            let de = (prev_mw2r * mr2w).get_euler_angles();
            (dp, de)
        };

        // Odometry covariance proportional to motion
        let dp_norm = delta_pos.norm().max(0.01);
        let de_norm = delta_ea.norm().max(0.001);
        let pos_sigma = vect3f::new(
            cfg.odo_pos_k * dp_norm + cfg.odo_pos_base,
            (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5, // lateral less noisy than forward
            (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5,
        );
        let ea_sigma = vect3f::new(
            cfg.odo_ea_k * de_norm + cfg.odo_ea_base,
            cfg.odo_ea_k * de_norm + cfg.odo_ea_base,
            cfg.odo_ea_k * de_norm + cfg.odo_ea_base,
        );

        let delta_pos_cov = matrix3f::from_elements(
            pos_sigma.x * pos_sigma.x, 0.0, 0.0,
            0.0, pos_sigma.y * pos_sigma.y, 0.0,
            0.0, 0.0, pos_sigma.z * pos_sigma.z,
        );
        let delta_ea_cov = matrix3f::from_elements(
            ea_sigma.x * ea_sigma.x, 0.0, 0.0,
            0.0, ea_sigma.y * ea_sigma.y, 0.0,
            0.0, 0.0, ea_sigma.z * ea_sigma.z,
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
                // horizontal and vertical angular error via atan2.
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

        // GPS: systematic offset (same for all poses) + relative noise
        let rel_noise = cfg.gps_rel_noise * (pos - gt_poses[0].0).norm();
        let gps_pos = vect3f::new(
            pos.x + gps_offset.x + rel_noise * rng.sample(normal01) as f32,
            pos.y + gps_offset.y + rel_noise * rng.sample(normal01) as f32,
            pos.z + gps_offset.z + rel_noise * rng.sample(normal01) as f32,
        );
        let gps_cov = matrix3f::from_elements(
            cfg.gps_sigma * cfg.gps_sigma, 0.0, 0.0,
            0.0, cfg.gps_sigma * cfg.gps_sigma, 0.0,
            0.0, 0.0, cfg.gps_sigma * cfg.gps_sigma,
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
        let (delta_ea_cov_r, delta_ea_cov_isigma) = decompose_cov(delta_ea_cov);
        let (gps_cov_r, gps_cov_isigma) = decompose_cov(gps_cov);

        path.poses.push_back(Pose {
            pos: Param::new(noisy_pos),
            ea: SimpleEulerAngleParam::new(noisy_ea),
            info: PoseInfo {
                delta_pos, delta_ea,
                delta_pos_cov_r, delta_pos_cov_isigma,
                delta_ea_cov_r, delta_ea_cov_isigma,
                gps: Some(GpsData { pos: gps_pos, cov_r: gps_cov_r, cov_isigma: gps_cov_isigma }),
                tilt_roll: ea.x + tilt_sigma_rad * rng.sample(normal01) as f32,
                tilt_pitch: ea.y + tilt_sigma_rad * rng.sample(normal01) as f32,
                features,
            },
            hb_pose: SelfBlock::new(),
        });
    }

    // Every pose is built; capture their handles to wire up frines and
    // odometry.
    let pose_refs: std::vec::Vec<Ref<Pose>> = path.poses.refs().collect();

    // Build landmarks with frines
    for (li, &(lm_pos, _)) in gt_landmarks.iter().enumerate() {
        let noisy_lm = vect3f::new(
            lm_pos.x + 0.5 * rng.sample(normal01) as f32,
            lm_pos.y + 0.5 * rng.sample(normal01) as f32,
            lm_pos.z + 0.3 * rng.sample(normal01) as f32,
        );
        let frines: std::vec::Vec<PointFrine> = frine_data.iter()
            .filter(|(lmi, _, _)| *lmi == li)
            .map(|(_, pose_i, feature)| PointFrine { pose: pose_refs[*pose_i], feature: *feature, hb: CrossBlock::new() })
            .collect();
        if frines.is_empty() { continue; } // skip landmarks with no observations
        path.landmarks.push(PointLandmark {
            pos: Param::new(noisy_lm),
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

    (path, gt_poses, gt_landmarks, gps_offset)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn print_usage() {
    eprintln!("Usage: slam_demo [OPTIONS]");
    eprintln!("  --solver <dense|faer|eigen|cholmod>  (default: faer)");
    eprintln!("  --poses <N>                          (default: 60)");
    eprintln!("  --landmarks <N>                      (default: 240)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut solver_name = "faer".to_string();
    let mut num_poses: Option<usize> = None;
    let mut num_landmarks: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--solver" => { i += 1; solver_name = args.get(i).cloned().unwrap_or_default(); }
            "--poses" => { i += 1; num_poses = args.get(i).and_then(|s| s.parse().ok()); }
            "--landmarks" => { i += 1; num_landmarks = args.get(i).and_then(|s| s.parse().ok()); }
            "--help" | "-h" => { print_usage(); return; }
            other => { eprintln!("Unknown argument: {}", other); print_usage(); return; }
        }
        i += 1;
    }

    let mut cfg = SceneConfig::default();
    if let Some(p) = num_poses { cfg.num_poses = p; }
    if let Some(l) = num_landmarks { cfg.num_landmarks = l; }

    println!("Solver: {}  Poses: {}  Landmarks: {}", solver_name, cfg.num_poses, cfg.num_landmarks);
    let (mut path, gt_poses, gt_landmarks, gps_offset) = build_path(&cfg);

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
        println!("Pose {:2}: pos=({:7.3}, {:7.3}, {:7.3}) ea=({:7.4}, {:7.4}, {:7.4})",
            i, pose.pos.value.x, pose.pos.value.y, pose.pos.value.z,
            pose.ea.value.x, pose.ea.value.y, pose.ea.value.z);
        println!("      gt: pos=({:7.3}, {:7.3}, {:7.3}) ea=({:7.4}, {:7.4}, {:7.4})",
            gt_p.x, gt_p.y, gt_p.z, gt_e.x, gt_e.y, gt_e.z);
    }
    println!();

    // Graduated optimization: start with loose feature constraints, tighten
    println!("--- Optimization ---");
    let isigma_scales = [0.01, 0.1, 1.0];

    for (pass, &scale) in isigma_scales.iter().enumerate() {
        path.frine_isigma_scale = scale;

        let mut params64: std::vec::Vec<f64> = std::vec::Vec::new();
        path.serialize64(&mut params64);
        let _n = params64.len();

        println!("\nPass {} (isigma scale={}):", pass + 1, scale);
        let config = arael::simple_lm::LmConfig::<f64> {
            verbose: true,
            rel_precision: 1e-6,
            ..Default::default()
        };
        let result = match solver_name.as_str() {
            "dense" => arael::simple_lm::solve(&params64, &mut path, &config),
            "faer" => arael::simple_lm::solve_sparse_faer(&params64, &mut path, &config),
            #[cfg(feature = "eigen")]
            "eigen" => arael::simple_lm::solve_sparse_eigen(&params64, &mut path, &config),
            #[cfg(not(feature = "eigen"))]
            "eigen" => { eprintln!("Eigen solver requires --features eigen"); return; }
            #[cfg(feature = "cholmod")]
            "cholmod" => arael::simple_lm::solve_sparse_cholmod(&params64, &mut path, &config),
            #[cfg(not(feature = "cholmod"))]
            "cholmod" => { eprintln!("CHOLMOD solver requires --features cholmod"); return; }
            _ => { eprintln!("Unknown solver: {}. Available: dense, faer, eigen, cholmod", solver_name); return; }
        };
        path.deserialize64(&result.x);
        println!("  {} iterations, cost {:.4} -> {:.4}", result.iterations, result.start_cost, result.end_cost);
    }

    // Mean absolute pose error vs GT (includes GPS systematic offset)
    {
        let mut pos_err_sum = 0.0_f32;
        let mut ea_err_sum = 0.0_f32;
        let n = gt_poses.len().min(path.poses.len());
        for i in 0..n {
            let pose = &path.poses[i];
            let (gt_p, gt_e) = gt_poses[i];
            pos_err_sum += (pose.pos.value - gt_p).norm();
            ea_err_sum += (pose.ea.value - gt_e).norm();
        }
        let mut params64: std::vec::Vec<f64> = std::vec::Vec::new();
        path.serialize64(&mut params64);
        let cost = path.calc_cost(&params64);
        println!("\nFinal cost: {:.4}  Simulated GPS systematic offset: ({:.3}, {:.3}, {:.3}) |{:.3}|m",
            cost, gps_offset.x, gps_offset.y, gps_offset.z, gps_offset.norm());
        println!("Mean pose error vs GT: pos={:.4}m  ea={:.3}deg  (dominated by GPS offset)",
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
        let opt_mr2w_prev = matrix3f::rotation_from_euler_angles(prev.ea.value);
        let opt_delta_pos = opt_mr2w_prev.transpose() * (pose.pos.value - prev.pos.value);

        let dpos_err = (opt_delta_pos - gt_delta_pos).norm();
        let gt_step = gt_delta_pos.norm();
        let dpos_rel = if gt_step > 1e-6 { 100.0 * dpos_err / gt_step } else { 0.0 };

        // GT delta_ea: relative rotation from prev to cur
        let gt_mr2w_cur = matrix3f::rotation_from_euler_angles(gt_cur_ea);
        let gt_delta_ea = (gt_mr2w.transpose() * gt_mr2w_cur).get_euler_angles();

        // Optimized delta_ea
        let opt_mr2w_cur = matrix3f::rotation_from_euler_angles(pose.ea.value);
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

    // Landmark uncertainty estimation via Hessian inverse.
    //
    // The Gauss-Newton Hessian H = 2*J^T*J is the information matrix (our
    // add_residual accumulates the factor of 2). The parameter covariance is
    // Cov = (J^T*J)^{-1} = 2*H^{-1}. For each landmark we extract the 3x3
    // diagonal block of Cov and compute its eigendecomposition. The square
    // roots of the eigenvalues are the semi-axis lengths of the 1-sigma
    // uncertainty ellipsoid.
    //
    // The raw diagonal blocks of Cov give GLOBAL uncertainty which includes
    // the shared gauge (GPS offset, yaw). To get useful per-landmark
    // uncertainty we compute the covariance of (landmark - closest_pose):
    // Cov_rel = C_ll - C_lp - C_pl + C_pp. This cancels the shared gauge
    // and gives uncertainty relative to the pose, matching |d| shown beside
    // it (the relative position error vs ground truth).
    let cov = {
        let mut params64: std::vec::Vec<f64> = std::vec::Vec::new();
        path.serialize64(&mut params64);
        let n = params64.len();
        let mut grad = vec![0.0_f64; n];
        let mut hessian = vec![0.0_f64; n * n];
        path.calc_grad_hessian_dense(&params64, &mut grad, &mut hessian);
        let h_mat = nalgebra::DMatrix::from_row_slice(n, n, &hessian);
        match nalgebra::linalg::Cholesky::new(h_mat) {
            Some(chol) => Some(chol.inverse() * 2.0),
            None => { println!("Hessian not positive definite -- no covariance"); None }
        }
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
        let opt_mr2w = matrix3f::rotation_from_euler_angles(opt_pose.ea.value);
        let opt_vec = opt_mr2w.transpose() * (lm.pos.value - opt_pose.pos.value);
        let err = (opt_vec - gt_vec).norm();
        let gt_dist = gt_vec.norm();
        let rel_pct = 100.0 * err / gt_dist;

        if let Some(ref cov) = cov {
            let k = lm.pos.index() as usize;
            let p = opt_pose.pos.index() as usize;
            // Cov_rel = C_ll - C_lp - C_pl + C_pp (cancels shared gauge)
            let c_ll = cov.fixed_view::<3, 3>(k, k);
            let c_pp = cov.fixed_view::<3, 3>(p, p);
            let c_lp = cov.fixed_view::<3, 3>(k, p);
            let cov_rel = (c_ll + c_pp - c_lp - c_lp.transpose()).clone_owned();
            let eigen = nalgebra::SymmetricEigen::new(cov_rel);
            let mut sigmas = [
                eigen.eigenvalues[0].max(0.0).sqrt(),
                eigen.eigenvalues[1].max(0.0).sqrt(),
                eigen.eigenvalues[2].max(0.0).sqrt(),
            ];
            sigmas.sort_by(|a, b| b.partial_cmp(a).unwrap());
            println!("LM {:3}: |d|={:.3}m  rel={:.2}%  dist={:.1}m  sigma=({:.3},{:.3},{:.3})m  frines={}",
                i, err, rel_pct, gt_dist, sigmas[0], sigmas[1], sigmas[2], lm.frines.len());
            max_sigmas.push(sigmas[0]);
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
