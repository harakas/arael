// Synthetic visual-inertial SLAM scene: the solver-agnostic problem
// every system optimizes, plus the ONE reference cost function they are
// validated against. Ported from examples/slam_demo.rs, restructured to
// emit flat data (no arael model types) so arael, tiny-solver, and any
// later competitor consume identical input.
//
// The factor graph is deliberately heterogeneous -- six factor types,
// several nonlinear (rotations, atan2 bearings, euler odometry) --
// which is the point: unlike a pure pose graph or bundle adjustment, it
// stresses per-factor-type code generation.
//
// This is the outlier-free scenario, which differs from the demo in two
// ways, both documented in README.md: (1) no outlier feature
// associations, so the bearing and GPS residuals are plain Gaussian
// (max-likelihood without outliers). The demo's robust gamma*atan kernel
// is omitted: with no outliers it has no benefit, and its saturation
// otherwise manufactures a spurious landmark-depth minimum that
// different solvers fall into. A single solve from the odometry init then
// reaches one unambiguous optimum every system agrees on. (2) fixed
// information scale (frine_isigma_scale = 1, no graduation -- graduation
// only earns its place once outliers and the robust kernel are present).
// The drift regularizers pull toward an explicit stored prior (the init),
// not arael's self-updating `_value`, so every system computes the same
// residual.

use arael::geometry::cameraf;
use arael::matrix::{matrix3d, matrix3f};
use arael::vect::{vect2f, vect3d, vect3f};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;

pub struct GpsObs {
    pub pos: vect3f,
    pub cov_r: matrix3f,    // covariance eigenvector rotation
    pub cov_isigma: vect3f, // 1 / sqrt(eigenvalues)
}

pub struct PoseData {
    pub init_pos: vect3f,
    pub init_ea: vect3f, // euler (roll, pitch, yaw)
    pub gps: Option<GpsObs>,
    pub tilt_roll: f32,
    pub tilt_pitch: f32,
}

pub struct OdoData {
    pub prev: u32,
    pub cur: u32,
    pub delta_pos: vect3f,
    pub delta_ea: vect3f,
    pub pos_cov_r: matrix3f,
    pub pos_cov_isigma: vect3f,
    pub ea_cov_r: matrix3f,
    pub ea_cov_isigma: vect3f,
}

pub struct FrineData {
    pub pose: u32,
    pub landmark: u32,
    pub mf2r: matrix3f,     // feature-to-robot frame (col0 = view dir)
    pub camera_pos: vect3f, // camera position in robot frame
    pub isigma: vect2f,     // 1 / sigma angular
}

pub struct Scene {
    pub poses: Vec<PoseData>,
    pub landmarks_init: Vec<vect3f>,
    pub odo: Vec<OdoData>,
    pub frines: Vec<FrineData>,
    pub drift_pos_isigma: f32,
    pub drift_ea_isigma: f32,
    pub drift_lm_isigma: f32,
    pub tilt_isigma: f32,
    pub frine_isigma_scale: f32,
}

/// One system's answer, evaluated by [`reference_cost`].
#[derive(Clone)]
pub struct Solution {
    pub poses: Vec<(vect3d, vect3d)>, // (pos, euler)
    pub landmarks: Vec<vect3d>,
}

/// Which path the robot drives.
///
/// On the `SCurve` the trajectory has two ends, so a landmark anchored near
/// either one is seen by fewer poses than its range allows -- its window is
/// clipped by the start or the end. `Loop` closes the path into a circle and
/// measures pose-index distance the short way round, so no window is clipped
/// and landmarks near the seam are observed from both ends of the pose list.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Trajectory {
    SCurve,
    Loop,
    /// A figure-8. The path crosses itself once, so two stretches of the
    /// trajectory that are far apart in time run through the same place.
    ///
    /// Visibility here is spatial -- a pose sees a landmark when it is within
    /// the anchor's radius on the ground -- because a pose-index rule cannot
    /// express "these two passes are at the same spot". That puts the shared
    /// landmarks in the middle of the pose ordering rather than at its ends,
    /// which is a different shape of coupling from [`Self::Loop`].
    Eight,
}

pub struct SceneConfig {
    pub trajectory: Trajectory,
    pub num_poses: usize,
    pub num_landmarks: usize,
    pub seed: u64,
    pub outlier_fraction: f32,
    pub outlier_scale: f32,
    pub s_amplitude: f32,
    pub s_frequency: f32,
    pub step_size: f32,
    pub gps_sigma: f32,
    pub gps_rel_noise: f32,
    pub odo_pos_k: f32,
    pub odo_pos_base: f32,
    pub odo_ea_k: f32,
    pub odo_ea_base: f32,
    // One-sided POSE-INDEX distance: a pose observes a landmark when it is
    // within this many poses of the landmark's anchor, so the landmark's
    // SPAN -- the poses that see it -- is 2 * range + 1.
    pub lm_visibility_range: usize,
    pub lm_visibility_prob: f32,
    // Heavy-tailed connectivity: a fraction of landmarks are "wide", spanning
    // up to half the trajectory (capped at WIDE_SPAN_CAP poses), modeling map
    // points a real trajectory revisits.
    pub wide_fraction: f32,
    // Initialization noise. The bearing factors are stiff (isigma ~600).
    // This benchmark polishes from a good init (a well-conditioned single
    // solve; per-step cost, the headline metric, is init-independent), so
    // the init noise is small.
    pub pose_init_pos_noise: f32,
    pub pose_init_ea_noise: f32,
    pub lm_init_noise: f32,
}

impl Default for SceneConfig {
    fn default() -> Self {
        SceneConfig {
            trajectory: Trajectory::SCurve,
            num_poses: 60,
            num_landmarks: 240,
            seed: 42,
            outlier_fraction: 0.0,
            outlier_scale: 30.0,
            s_amplitude: 1.5,
            s_frequency: 0.8,
            step_size: 0.25,
            gps_sigma: 2.5,
            gps_rel_noise: 0.02,
            odo_pos_k: 0.10,
            odo_pos_base: 0.03,
            odo_ea_k: 0.01,
            odo_ea_base: 0.001,
            lm_visibility_range: 15,
            lm_visibility_prob: 0.75,
            wide_fraction: 0.15,
            pose_init_pos_noise: 0.05,
            pose_init_ea_noise: 0.01,
            lm_init_noise: 0.05,
        }
    }
}

fn create_cameras() -> Vec<cameraf> {
    let mut cameras = Vec::new();
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
        let mc2r = matrix3f::from_cols(
            vect3f::new(-sy, cy, 0.0),
            vect3f::new(0.0, 0.0, -1.0),
            vect3f::new(cy, sy, 0.0),
        );
        cameras.push(cameraf {
            fx, fy,
            cx: w as f32 / 2.0,
            cy: h as f32 / 2.0,
            width: w,
            height: h,
            camera_pos: vect3f::new(cy * 0.1, sy * 0.1, 0.3),
            mc2r,
        });
    }
    cameras
}

fn ground_truth_poses(cfg: &SceneConfig) -> Vec<(vect3f, vect3f)> {
    match cfg.trajectory {
        Trajectory::SCurve => {
            let mut poses = Vec::new();
            let mut t = 0.0_f32;
            for _ in 0..cfg.num_poses {
                let x = t;
                let y = cfg.s_amplitude * (cfg.s_frequency * t).sin();
                let pos = vect3f::new(x, y, 0.0);
                let dx = 1.0;
                let dy = cfg.s_amplitude * cfg.s_frequency * (cfg.s_frequency * t).cos();
                let yaw = dy.atan2(dx);
                poses.push((pos, vect3f::new(0.0, 0.0, yaw)));
                t += cfg.step_size;
            }
            poses
        }
        Trajectory::Loop => loop_poses(cfg),
        Trajectory::Eight => eight_poses(cfg),
    }
}

/// Poses evenly spaced along a lemniscate of Gerono, `(cos t, sin t cos t)`,
/// which closes and crosses itself once at the origin.
///
/// Sized and stepped like [`loop_poses`]: the curve is scaled so its total
/// length is `num_poses * step_size`, then resampled at equal arc length, so a
/// pose advances the same distance it does on every other trajectory.
fn eight_poses(cfg: &SceneConfig) -> Vec<(vect3f, vect3f)> {
    use std::f32::consts::{PI, TAU};
    let n = cfg.num_poses;
    // Dense polyline of the unit curve, to measure arc length and resample.
    // 64 samples per pose is far finer than the spacing we read back off it.
    let dense = (n * 64).max(4096);
    let pt = |t: f32| vect3f::new(t.cos(), t.sin() * t.cos(), 0.0);
    let mut cum = Vec::with_capacity(dense + 1);
    cum.push(0.0f32);
    for k in 1..=dense {
        let (a, b) = (TAU * (k - 1) as f32 / dense as f32, TAU * k as f32 / dense as f32);
        let d = (pt(b) - pt(a)).norm();
        cum.push(cum[k - 1] + d);
    }
    let unit_len = cum[dense];
    let scale = n as f32 * cfg.step_size / unit_len;

    let mut poses = Vec::with_capacity(n);
    let mut k = 0usize;
    for i in 0..n {
        let target = unit_len * (i as f32) / (n as f32);
        while k + 1 < dense && cum[k + 1] < target {
            k += 1;
        }
        // Linear interpolation in t across the segment holding `target`.
        let seg = (cum[k + 1] - cum[k]).max(1e-20);
        let frac = ((target - cum[k]) / seg).clamp(0.0, 1.0);
        let t = TAU * (k as f32 + frac) / dense as f32;
        let p = pt(t) * scale;
        // Tangent d/dt (cos t, sin t cos t) = (-sin t, cos 2t).
        let mut yaw = (2.0 * t).cos().atan2(-t.sin());
        while yaw > PI { yaw -= TAU; }
        while yaw <= -PI { yaw += TAU; }
        poses.push((p, vect3f::new(0.0, 0.0, yaw)));
    }
    poses
}

/// Poses evenly spaced around a circle, heading along the tangent.
///
/// The radius comes from the distance the robot travels: the circumference is
/// `num_poses * step_size`, so a pose still advances `step_size` along the path
/// and the odometry deltas keep the size they have on the S-curve. The circle
/// is therefore small on a short trajectory (2.4 m at 60 poses) and grows with
/// the pose count.
fn loop_poses(cfg: &SceneConfig) -> Vec<(vect3f, vect3f)> {
    use std::f32::consts::{FRAC_PI_2, PI, TAU};
    let n = cfg.num_poses;
    let radius = n as f32 * cfg.step_size / TAU;
    let mut poses = Vec::new();
    for i in 0..n {
        let theta = TAU * (i as f32) / (n as f32);
        let pos = vect3f::new(radius * theta.cos(), radius * theta.sin(), 0.0);
        // Tangent of a counter-clockwise circle, wrapped to (-pi, pi] so the
        // stored euler angles stay in the same range the S-curve produces --
        // a yaw that ran past pi would make the drift prior and the reference
        // cost disagree with a solver that landed on the equivalent angle.
        let mut yaw = theta + FRAC_PI_2;
        while yaw > PI { yaw -= TAU; }
        poses.push((pos, vect3f::new(0.0, 0.0, yaw)));
    }
    poses
}

/// How far apart two poses are along the trajectory, counted in poses.
///
/// `wrap` closes the path: with no ends, the short way round is the distance
/// and the last pose neighbours the first. Trajectories whose visibility is
/// spatial rather than topological do not use this at all.
fn pose_index_distance(wrap: bool, a: usize, b: usize, num_poses: usize) -> usize {
    let d = a.abs_diff(b);
    if wrap { d.min(num_poses - d) } else { d }
}

/// A landmark's visibility SPAN is how many consecutive poses observe it.
/// Visibility is topological -- a pose sees a landmark when their POSE-INDEX
/// distance is within the landmark's range -- and `range` is that distance
/// measured ONE WAY from the anchor pose, so `span = 2 * range + 1` (the +1
/// is the anchor itself). Keeping the two straight matters: reading a span
/// as a range doubles the window.
fn range_for_span(span: usize) -> usize {
    span / 2
}

/// Widest span a landmark may have, whatever the trajectory length. Past
/// this a single map point couples most of the trajectory into one clique,
/// which is not what revisiting a place looks like.
const WIDE_SPAN_CAP: usize = 150;

// (landmark_pos, anchor_pose_index, visibility_range)
fn ground_truth_landmarks(cfg: &SceneConfig, rng: &mut StdRng, poses: &[(vect3f, vect3f)])
    -> Vec<(vect3f, usize, usize)> {
    // Wide landmarks span up to half the trajectory, capped at WIDE_SPAN_CAP
    // poses; the narrowest of them spans a quarter of it.
    let wide_max = range_for_span((cfg.num_poses / 2).min(WIDE_SPAN_CAP))
        .max(cfg.lm_visibility_range);
    let wide_min = range_for_span((cfg.num_poses / 4).min(WIDE_SPAN_CAP))
        .max(cfg.lm_visibility_range);
    let mut landmarks = Vec::new();
    for _ in 0..cfg.num_landmarks {
        loop {
            let anchor_idx = rng.random_range(0..poses.len());
            let anchor = &poses[anchor_idx].0;
            let angle = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
            let dist = 5.0 + rng.random::<f32>() * 25.0;
            let lm = vect3f::new(
                anchor.x + dist * angle.cos(),
                anchor.y + dist * angle.sin(),
                rng.random::<f32>() * 2.0);
            let min_dist = poses.iter()
                .map(|(p, _)| (lm - *p).norm())
                .fold(f32::MAX, f32::min);
            if min_dist >= 5.0 && min_dist <= 30.0 {
                let range = if rng.random::<f32>() < cfg.wide_fraction {
                    rng.random_range(wide_min..=wide_max)
                } else {
                    cfg.lm_visibility_range
                };
                landmarks.push((lm, anchor_idx, range));
                break;
            }
        }
    }
    landmarks
}

// symmetric_eigen decomposition into (rotation, 1/sqrt(eigenvalues)).
fn decompose_cov(cov: matrix3f) -> (matrix3f, vect3f) {
    let (r, d) = cov.symmetric_eigen();
    (r, vect3f::new(1.0 / d.x.sqrt(), 1.0 / d.y.sqrt(), 1.0 / d.z.sqrt()))
}

pub fn generate(cfg: &SceneConfig) -> Scene {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let normal01 = Normal::new(0.0, 1.0).unwrap();

    let gps_offset = vect3f::new(
        cfg.gps_sigma * rng.sample(normal01) as f32,
        cfg.gps_sigma * rng.sample(normal01) as f32,
        cfg.gps_sigma * rng.sample(normal01) as f32,
    );

    let gt_poses = ground_truth_poses(cfg);
    let gt_landmarks = ground_truth_landmarks(cfg, &mut rng, &gt_poses);
    let cameras = create_cameras();

    let drift_pos_sigma = 1000.0_f32;
    let drift_ea_sigma_deg = 1800.0_f32;
    let drift_lm_sigma = 1000.0_f32;
    let tilt_sigma_rad = 0.25_f32.to_radians();

    let mut scene = Scene {
        poses: Vec::new(),
        landmarks_init: Vec::new(),
        odo: Vec::new(),
        frines: Vec::new(),
        drift_pos_isigma: 1.0 / drift_pos_sigma,
        drift_ea_isigma: 1.0 / drift_ea_sigma_deg.to_radians(),
        drift_lm_isigma: 1.0 / drift_lm_sigma,
        tilt_isigma: 1.0 / tilt_sigma_rad,
        frine_isigma_scale: 1.0,
    };

    // Deferred so we can drop landmarks with zero observations and
    // renumber. (raw_landmark_index, pose_index, mf2r, camera_pos, isigma)
    let mut raw_frines: Vec<(usize, u32, matrix3f, vect3f, vect2f)> = Vec::new();

    for (pi, &(pos, ea)) in gt_poses.iter().enumerate() {
        let mr2w = matrix3f::rotation_from_euler_angles(ea);

        let (delta_pos, delta_ea) = if pi == 0 {
            (vect3f::new(0.0, 0.0, 0.0), vect3f::new(0.0, 0.0, 0.0))
        } else {
            let (prev_pos, prev_ea) = gt_poses[pi - 1];
            let prev_mw2r = matrix3f::rotation_from_euler_angles(prev_ea).transpose();
            (prev_mw2r * (pos - prev_pos),
             (prev_mw2r * mr2w).get_euler_angles())
        };

        let dp_norm = delta_pos.norm().max(0.01);
        let de_norm = delta_ea.norm().max(0.001);
        let pos_sigma = vect3f::new(
            cfg.odo_pos_k * dp_norm + cfg.odo_pos_base,
            (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5,
            (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5,
        );
        let ea_sigma = cfg.odo_ea_k * de_norm + cfg.odo_ea_base;
        let (pos_cov_r, pos_cov_isigma) = decompose_cov(matrix3f::from_elements(
            pos_sigma.x * pos_sigma.x, 0.0, 0.0,
            0.0, pos_sigma.y * pos_sigma.y, 0.0,
            0.0, 0.0, pos_sigma.z * pos_sigma.z));
        let (ea_cov_r, ea_cov_isigma) = decompose_cov(matrix3f::from_elements(
            ea_sigma * ea_sigma, 0.0, 0.0,
            0.0, ea_sigma * ea_sigma, 0.0,
            0.0, 0.0, ea_sigma * ea_sigma));

        // Features (only for landmarks visible from this pose).
        for (li, &(lm_pos, anchor_idx, range)) in gt_landmarks.iter().enumerate() {
            let visible = match cfg.trajectory {
                Trajectory::SCurve | Trajectory::Loop => {
                    let wrap = cfg.trajectory == Trajectory::Loop;
                    pose_index_distance(wrap, pi, anchor_idx, gt_poses.len()) <= range
                }
                // Range is a count of poses; a pose covers step_size of ground,
                // so the same window in metres is range * step_size.
                Trajectory::Eight => {
                    (pos - gt_poses[anchor_idx].0).norm() <= range as f32 * cfg.step_size
                }
            };
            if !visible { continue; }
            if rng.random::<f32>() > cfg.lm_visibility_prob { continue; }
            for cam in &cameras {
                let p_cam = cam.world_to_camera(lm_pos, pos, mr2w);
                if p_cam.z < 0.5 { continue; }
                let pixel = cam.project(p_cam);
                if !cam.is_visible(pixel) { continue; }
                let is_outlier = rng.random::<f32>() < cfg.outlier_fraction;
                let noise_scale = if is_outlier { cfg.outlier_scale } else { 1.0 };
                let noisy_pixel = vect2f::new(
                    pixel.x + noise_scale * (rng.random::<f32>() * 2.0 - 1.0),
                    pixel.y + noise_scale * (rng.random::<f32>() * 2.0 - 1.0),
                );
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
                raw_frines.push((li, pi as u32, mf2r, cam.camera_pos, isigma));
            }
        }

        let rel_noise = cfg.gps_rel_noise * (pos - gt_poses[0].0).norm();
        let gps_pos = vect3f::new(
            pos.x + gps_offset.x + rel_noise * rng.sample(normal01) as f32,
            pos.y + gps_offset.y + rel_noise * rng.sample(normal01) as f32,
            pos.z + gps_offset.z + rel_noise * rng.sample(normal01) as f32,
        );
        let (gps_cov_r, gps_cov_isigma) = decompose_cov(matrix3f::from_elements(
            cfg.gps_sigma * cfg.gps_sigma, 0.0, 0.0,
            0.0, cfg.gps_sigma * cfg.gps_sigma, 0.0,
            0.0, 0.0, cfg.gps_sigma * cfg.gps_sigma));

        let init_noise_pos = cfg.pose_init_pos_noise;
        let init_noise_ea = cfg.pose_init_ea_noise;
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

        if pi > 0 {
            scene.odo.push(OdoData {
                prev: (pi - 1) as u32,
                cur: pi as u32,
                delta_pos, delta_ea,
                pos_cov_r, pos_cov_isigma,
                ea_cov_r, ea_cov_isigma,
            });
        }
        scene.poses.push(PoseData {
            init_pos: noisy_pos,
            init_ea: noisy_ea,
            gps: Some(GpsObs { pos: gps_pos, cov_r: gps_cov_r, cov_isigma: gps_cov_isigma }),
            tilt_roll: ea.x + tilt_sigma_rad * rng.sample(normal01) as f32,
            tilt_pitch: ea.y + tilt_sigma_rad * rng.sample(normal01) as f32,
        });
    }

    // Keep only observed landmarks, renumber, attach frines.
    let mut new_index = vec![u32::MAX; gt_landmarks.len()];
    for (li, &(lm_pos, _, _)) in gt_landmarks.iter().enumerate() {
        if !raw_frines.iter().any(|(l, ..)| *l == li) { continue; }
        new_index[li] = scene.landmarks_init.len() as u32;
        let ln = cfg.lm_init_noise;
        scene.landmarks_init.push(vect3f::new(
            lm_pos.x + ln * rng.sample(normal01) as f32,
            lm_pos.y + ln * rng.sample(normal01) as f32,
            lm_pos.z + ln * rng.sample(normal01) as f32,
        ));
    }
    for (li, pose, mf2r, camera_pos, isigma) in raw_frines {
        scene.frines.push(FrineData {
            pose, landmark: new_index[li], mf2r, camera_pos, isigma,
        });
    }

    scene
}

// ---------------------------------------------------------------------------
// Reference cost -- THE function every system is validated against.
// ---------------------------------------------------------------------------

fn m3(m: &matrix3f) -> matrix3d { matrix3d::from(*m) }
fn v3(v: vect3f) -> vect3d { vect3d::from(v) }

/// Sum of squared residuals over all six factor types. This is the
/// identical objective every system optimizes; each system's final
/// solution is scored by this one function.
pub fn reference_cost(scene: &Scene, sol: &Solution) -> f64 {
    assert_eq!(sol.poses.len(), scene.poses.len());
    assert_eq!(sol.landmarks.len(), scene.landmarks_init.len());
    let mut cost = 0.0;

    // Precompute pose rotation matrices.
    let rots: Vec<matrix3d> = sol.poses.iter()
        .map(|(_, ea)| matrix3d::rotation_from_euler_angles(*ea))
        .collect();

    for (i, p) in scene.poses.iter().enumerate() {
        let (pos, ea) = sol.poses[i];
        // GPS (plain Gaussian).
        if let Some(g) = &p.gps {
            let raw = pos - v3(g.pos);
            let rt = m3(&g.cov_r).transpose() * raw;
            let isig = v3(g.cov_isigma);
            for k in 0..3 {
                let r = [rt.x, rt.y, rt.z][k] * [isig.x, isig.y, isig.z][k];
                cost += r * r;
            }
        }
        // Pose drift prior (to init).
        let pd = pos - v3(p.init_pos);
        let ed = ea - v3(p.init_ea);
        let dpi = scene.drift_pos_isigma as f64;
        let dei = scene.drift_ea_isigma as f64;
        cost += (pd.x * dpi).powi(2) + (pd.y * dpi).powi(2) + (pd.z * dpi).powi(2);
        cost += (ed.x * dei).powi(2) + (ed.y * dei).powi(2) + (ed.z * dei).powi(2);
        // Tilt.
        let ti = scene.tilt_isigma as f64;
        cost += ((ea.x - p.tilt_roll as f64) * ti).powi(2);
        cost += ((ea.y - p.tilt_pitch as f64) * ti).powi(2);
    }

    // Landmark drift prior.
    let dli = scene.drift_lm_isigma as f64;
    for (i, lm) in sol.landmarks.iter().enumerate() {
        let d = *lm - v3(scene.landmarks_init[i]);
        cost += (d.x * dli).powi(2) + (d.y * dli).powi(2) + (d.z * dli).powi(2);
    }

    // Bearing observations (plain Gaussian).
    let scale = scene.frine_isigma_scale as f64;
    for f in &scene.frines {
        let (pos, _) = sol.poses[f.pose as usize];
        let mr2w = &rots[f.pose as usize];
        let lm = sol.landmarks[f.landmark as usize];
        let lm_r = mr2w.transpose() * (lm - pos);
        let r_r = lm_r - v3(f.camera_pos);
        let r_f = m3(&f.mf2r).transpose() * r_r;
        let isig = v3f2(f.isigma);
        let e1 = r_f.y.atan2(r_f.x) * isig.0 * scale;
        let e2 = r_f.z.atan2(r_f.x) * isig.1 * scale;
        cost += e1 * e1 + e2 * e2;
    }

    // Odometry.
    for o in &scene.odo {
        let (prev_pos, _) = sol.poses[o.prev as usize];
        let (cur_pos, _) = sol.poses[o.cur as usize];
        let mr2w_prev = &rots[o.prev as usize];
        let mr2w_cur = &rots[o.cur as usize];
        let pos_diff = mr2w_prev.transpose() * (cur_pos - prev_pos);
        let pos_err = pos_diff - v3(o.delta_pos);
        let pos_w = m3(&o.pos_cov_r).transpose() * pos_err;
        let expected = *mr2w_prev * matrix3d::rotation_from_euler_angles(v3(o.delta_ea));
        let error_rot = expected.transpose() * *mr2w_cur;
        let ea_err = error_rot.get_euler_angles();
        let ea_w = m3(&o.ea_cov_r).transpose() * ea_err;
        let pi = v3(o.pos_cov_isigma);
        let ei = v3(o.ea_cov_isigma);
        cost += (pos_w.x * pi.x).powi(2) + (pos_w.y * pi.y).powi(2) + (pos_w.z * pi.z).powi(2);
        cost += (ea_w.x * ei.x).powi(2) + (ea_w.y * ei.y).powi(2) + (ea_w.z * ei.z).powi(2);
    }

    cost
}

fn v3f2(v: vect2f) -> (f64, f64) { (v.x as f64, v.y as f64) }

// ---------------------------------------------------------------------------
// Export -- flat text the C++ Ceres runner consumes (identical problem).
// ---------------------------------------------------------------------------

use std::fmt::Write;

// Values are stored f32 but scored (and read by Ceres) as f64. Write the
// exact f32->f64 upcast with full f64 precision so the C++ double equals
// the reference's f32-as-f64 -- a shortest-f32 decimal would re-parse as
// a different double and shift the cost at the 8th digit.
fn wv3(out: &mut String, v: vect3f) {
    let _ = write!(out, "{} {} {} ", v.x as f64, v.y as f64, v.z as f64);
}
fn wm3(out: &mut String, m: &matrix3f) {
    for r in 0..3 {
        let _ = write!(out, "{} {} {} ", m[r].x as f64, m[r].y as f64, m[r].z as f64);
    }
}

/// Serialize the scene to a text file for cross-language runners. Layout
/// (whitespace-separated, header first):
///   n_poses n_landmarks n_frines n_odo
///   drift_pos_isigma drift_ea_isigma drift_lm_isigma tilt_isigma frine_isigma_scale
///   per pose:      init_pos(3) init_ea(3) gps_pos(3) gps_cov_r(9) gps_cov_isigma(3) tilt_roll tilt_pitch
///   per landmark:  init_pos(3)
///   per frine:     pose landmark mf2r(9) camera_pos(3) isigma(2)
///   per odo:       prev cur delta_pos(3) delta_ea(3) pos_cov_r(9) pos_cov_isigma(3) ea_cov_r(9) ea_cov_isigma(3)
pub fn write_scene(scene: &Scene, path: &str) {
    let mut s = String::new();
    let _ = writeln!(s, "{} {} {} {}",
        scene.poses.len(), scene.landmarks_init.len(), scene.frines.len(), scene.odo.len());
    let _ = writeln!(s, "{} {} {} {} {}",
        scene.drift_pos_isigma as f64, scene.drift_ea_isigma as f64,
        scene.drift_lm_isigma as f64, scene.tilt_isigma as f64, scene.frine_isigma_scale as f64);
    for p in &scene.poses {
        let g = p.gps.as_ref().unwrap();
        wv3(&mut s, p.init_pos); wv3(&mut s, p.init_ea);
        wv3(&mut s, g.pos); wm3(&mut s, &g.cov_r); wv3(&mut s, g.cov_isigma);
        let _ = writeln!(s, "{} {}", p.tilt_roll as f64, p.tilt_pitch as f64);
    }
    for l in &scene.landmarks_init {
        wv3(&mut s, *l);
        s.push('\n');
    }
    for f in &scene.frines {
        let _ = write!(s, "{} {} ", f.pose, f.landmark);
        wm3(&mut s, &f.mf2r); wv3(&mut s, f.camera_pos);
        let _ = writeln!(s, "{} {}", f.isigma.x as f64, f.isigma.y as f64);
    }
    for o in &scene.odo {
        let _ = write!(s, "{} {} ", o.prev, o.cur);
        wv3(&mut s, o.delta_pos); wv3(&mut s, o.delta_ea);
        wm3(&mut s, &o.pos_cov_r); wv3(&mut s, o.pos_cov_isigma);
        wm3(&mut s, &o.ea_cov_r); wv3(&mut s, o.ea_cov_isigma);
        s.push('\n');
    }
    std::fs::write(path, s).unwrap();
}

/// Read a solution back (per-pose "x y z roll pitch yaw", then per-landmark
/// "x y z"), for scoring an external runner against reference_cost.
pub fn read_solution(path: &str, n_poses: usize, n_landmarks: usize) -> Solution {
    let text = std::fs::read_to_string(path).unwrap();
    let mut it = text.split_ascii_whitespace().map(|t| t.parse::<f64>().unwrap());
    let mut next = || it.next().unwrap();
    let poses = (0..n_poses).map(|_| {
        (vect3d::new(next(), next(), next()), vect3d::new(next(), next(), next()))
    }).collect();
    let landmarks = (0..n_landmarks).map(|_| vect3d::new(next(), next(), next())).collect();
    Solution { poses, landmarks }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initial_solution(scene: &Scene) -> Solution {
        Solution {
            poses: scene.poses.iter()
                .map(|p| (vect3d::from(p.init_pos), vect3d::from(p.init_ea)))
                .collect(),
            landmarks: scene.landmarks_init.iter().map(|l| vect3d::from(*l)).collect(),
        }
    }

    /// The published tables are all measured on the default scene, so a change
    /// that shifts it -- an extra rng draw, a moved visibility test -- silently
    /// invalidates every number in the README. These are the counts and the
    /// initial cost the 60-pose table was produced from.
    #[test]
    fn default_scene_is_unchanged() {
        let cfg = SceneConfig::default();
        assert_eq!(cfg.trajectory, Trajectory::SCurve, "the default scene is the S-curve");
        let scene = generate(&cfg);
        assert_eq!(scene.poses.len(), 60);
        assert_eq!(scene.landmarks_init.len(), 240);
        assert_eq!(scene.frines.len(), 5370);
        assert_eq!(scene.odo.len(), 59);
        let cost = reference_cost(&scene, &initial_solution(&scene));
        assert!((cost - 630337.3712).abs() < 1e-3, "initial reference cost {:.4}", cost);
    }

    /// Adding the loop must not perturb the S-curve's rng draw sequence.
    #[test]
    fn loop_mode_leaves_the_scurve_scene_alone() {
        let a = generate(&SceneConfig::default());
        let b = generate(&SceneConfig { trajectory: Trajectory::SCurve, ..Default::default() });
        assert_eq!(a.frines.len(), b.frines.len());
        let (sa, sb) = (initial_solution(&a), initial_solution(&b));
        assert_eq!(reference_cost(&a, &sa), reference_cost(&b, &sb));
    }

    fn loop_cfg(num_poses: usize) -> SceneConfig {
        SceneConfig { trajectory: Trajectory::Loop, num_poses,
                      num_landmarks: 4 * num_poses, ..Default::default() }
    }

    /// Every step is step_size long, including the one from the last pose back
    /// to the first: that closing step is what makes the ends meet.
    #[test]
    fn loop_trajectory_closes() {
        let cfg = loop_cfg(60);
        let poses = ground_truth_poses(&cfg);
        assert_eq!(poses.len(), 60);
        let step = |a: usize, b: usize| (poses[b].0 - poses[a].0).norm();
        // A chord is slightly shorter than the arc it subtends; at 60 poses the
        // circle is coarse enough for that to show, so allow 1%.
        for i in 1..poses.len() {
            let d = step(i - 1, i);
            assert!((d - cfg.step_size).abs() < 0.01 * cfg.step_size, "step {} is {}", i, d);
        }
        let closing = step(poses.len() - 1, 0);
        assert!((closing - cfg.step_size).abs() < 0.01 * cfg.step_size,
                "closing step is {}", closing);
    }

    /// Poses ride a circle whose circumference is the distance travelled, and
    /// the stored yaw stays in the range the S-curve produces.
    #[test]
    fn loop_geometry() {
        let cfg = loop_cfg(300);
        let poses = ground_truth_poses(&cfg);
        let radius = cfg.num_poses as f32 * cfg.step_size / std::f32::consts::TAU;
        for (pos, ea) in &poses {
            assert!((pos.norm() - radius).abs() < 1e-3, "radius {}", pos.norm());
            assert_eq!(pos.z, 0.0);
            assert!(ea.z > -std::f32::consts::PI && ea.z <= std::f32::consts::PI,
                    "yaw {} outside (-pi, pi]", ea.z);
        }
    }

    #[test]
    fn pose_index_distance_wraps_only_when_asked() {
        // 2 apart the short way round a 60-pose loop, 58 along an open path.
        assert_eq!(pose_index_distance(false, 59, 1, 60), 58);
        assert_eq!(pose_index_distance(true, 59, 1, 60), 2);
        // Half way round is the farthest two poses on a loop can be.
        assert_eq!(pose_index_distance(true, 0, 30, 60), 30);
        assert_eq!(pose_index_distance(true, 0, 0, 60), 0);
    }

    /// The point of the mode: landmarks straddle the seam, so the first and last
    /// poses share observations instead of each seeing a clipped window.
    #[test]
    fn loop_landmarks_span_the_seam() {
        let edge = 5;
        let spanning = |scene: &Scene, n: usize| {
            let mut lo = vec![false; scene.landmarks_init.len()];
            let mut hi = vec![false; scene.landmarks_init.len()];
            for f in &scene.frines {
                if (f.pose as usize) < edge { lo[f.landmark as usize] = true; }
                if (f.pose as usize) >= n - edge { hi[f.landmark as usize] = true; }
            }
            (0..lo.len()).filter(|&i| lo[i] && hi[i]).count()
        };
        let n = 60;
        let looped = generate(&loop_cfg(n));
        assert!(spanning(&looped, n) > 0, "no landmark crosses the loop seam");
        // On the open path the ends are 59 poses apart and no window is that
        // wide, so nothing is shared -- the clipping this mode removes.
        let open = generate(&SceneConfig { num_poses: n, num_landmarks: 4 * n,
                                           ..Default::default() });
        assert_eq!(spanning(&open, n), 0, "the S-curve should not share end observations");
    }

    fn eight_cfg(num_poses: usize) -> SceneConfig {
        SceneConfig { trajectory: Trajectory::Eight, num_poses,
                      num_landmarks: 4 * num_poses, ..Default::default() }
    }

    /// The lemniscate is resampled at equal arc length, so every step is
    /// step_size -- the same distance a pose covers on the other trajectories.
    #[test]
    fn eight_is_evenly_spaced_and_closes() {
        let cfg = eight_cfg(300);
        let poses = ground_truth_poses(&cfg);
        assert_eq!(poses.len(), 300);
        for i in 1..poses.len() {
            let d = (poses[i].0 - poses[i - 1].0).norm();
            assert!((d - cfg.step_size).abs() < 0.02 * cfg.step_size, "step {} is {}", i, d);
        }
        let closing = (poses[0].0 - poses[poses.len() - 1].0).norm();
        assert!((closing - cfg.step_size).abs() < 0.02 * cfg.step_size,
                "closing step is {}", closing);
    }

    /// The path crosses itself: two poses far apart in the ordering come back
    /// to the same place. That is what the mode exists to produce.
    #[test]
    fn eight_crosses_itself() {
        let poses = ground_truth_poses(&eight_cfg(300));
        let n = poses.len();
        let mut best = f32::MAX;
        for i in 0..n {
            for j in 0..n {
                // Far apart along the path, measured the short way round.
                let d = i.abs_diff(j);
                if d.min(n - d) < n / 8 { continue; }
                best = best.min((poses[i].0 - poses[j].0).norm());
            }
        }
        assert!(best < 0.5, "closest far-apart pair is {} apart", best);
    }

    /// Landmarks at the crossing are seen from both passes, coupling poses in
    /// the middle of the ordering rather than at its ends.
    #[test]
    fn eight_shares_landmarks_across_the_crossing() {
        let n = 300;
        let scene = generate(&eight_cfg(n));
        let mut lo = vec![false; scene.landmarks_init.len()];
        let mut hi = vec![false; scene.landmarks_init.len()];
        // The two passes through the origin are near 1/4 and 3/4 of the path.
        for f in &scene.frines {
            let p = f.pose as usize;
            if p.abs_diff(n / 4) < n / 16 { lo[f.landmark as usize] = true; }
            if p.abs_diff(3 * n / 4) < n / 16 { hi[f.landmark as usize] = true; }
        }
        let shared = (0..lo.len()).filter(|&i| lo[i] && hi[i]).count();
        assert!(shared > 0, "no landmark is seen from both passes of the crossing");
    }

    /// No pose sits in a visibility shadow at the ends, which is the whole
    /// reason for the mode: on the open path the first and last poses carry
    /// noticeably fewer observations than the middle ones.
    #[test]
    fn loop_has_no_observation_falloff_at_the_ends() {
        let n = 300;
        let per_pose = |scene: &Scene| {
            let mut c = vec![0usize; n];
            for f in &scene.frines { c[f.pose as usize] += 1; }
            c
        };
        let edge = 10;
        let mean = |v: &[usize]| v.iter().sum::<usize>() as f64 / v.len() as f64;

        let c = per_pose(&generate(&loop_cfg(n)));
        let ends: Vec<usize> = c[..edge].iter().chain(&c[n - edge..]).copied().collect();
        let middle = &c[n / 2 - edge..n / 2 + edge];
        assert!(mean(&ends) > 0.8 * mean(middle),
                "loop ends {:.1} vs middle {:.1}", mean(&ends), mean(middle));

        let c = per_pose(&generate(&SceneConfig { num_poses: n, num_landmarks: 4 * n,
                                                  ..Default::default() }));
        let ends: Vec<usize> = c[..edge].iter().chain(&c[n - edge..]).copied().collect();
        let middle = &c[n / 2 - edge..n / 2 + edge];
        assert!(mean(&ends) < 0.8 * mean(middle),
                "S-curve ends {:.1} vs middle {:.1}", mean(&ends), mean(middle));
    }
}
