// Synthetic localization scene: a trajectory estimated against a KNOWN
// (fixed) landmark map from bearing observations + odometry, plus the ONE
// reference cost every system is validated against. Localization, NOT SLAM:
// the landmarks are ground-truth constants (not optimized), so there is no
// gauge freedom and no GPS is needed -- the fixed map pins the frame, and
// absolute pose errors are meaningful. Ported from examples/loc_demo.rs,
// restructured to emit flat data (no arael model types) so arael,
// tiny-solver, factrs, and the C++ runners consume identical input.
//
// The factor graph is heterogeneous -- bearings (atan2), euler odometry,
// pose drift + tilt priors -- which is the point: it stresses per-factor
// code generation, not just a pose graph.
//
// This is the outlier-free scenario (see README.md): the bearing residuals
// are plain Gaussian. loc_demo's robust gamma*atan kernel is omitted --
// with no outliers it has no benefit, and its saturation manufactures a
// spurious minimum different solvers fall into. Fixed information scale
// (frine_isigma_scale = 1). The drift regularizer pulls toward an explicit
// stored prior (the init), not arael's self-updating `_value`, so every
// system computes the same residual.

use arael::geometry::Camera;
use arael::matrix::{matrix3d, matrix3f};
use arael::vect::{vect2f, vect3d, vect3f};
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;

pub struct PoseData {
    pub init_pos: vect3f,
    pub init_ea: vect3f, // euler (roll, pitch, yaw)
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
    pub landmarks: Vec<vect3f>, // FIXED ground-truth map (not optimized)
    pub odo: Vec<OdoData>,
    pub frines: Vec<FrineData>,
    pub drift_pos_isigma: f32,
    pub drift_ea_isigma: f32,
    pub tilt_isigma: f32,
    pub frine_isigma_scale: f32,
}

/// One system's answer, evaluated by [`reference_cost`]. Landmarks are fixed,
/// so the solution is the pose trajectory only.
pub struct Solution {
    pub poses: Vec<(vect3d, vect3d)>, // (pos, euler)
}

pub struct SceneConfig {
    pub num_poses: usize,
    pub num_landmarks: usize,
    pub seed: u64,
    pub outlier_fraction: f32,
    pub outlier_scale: f32,
    pub s_amplitude: f32,
    pub s_frequency: f32,
    pub step_size: f32,
    pub odo_pos_k: f32,
    pub odo_pos_base: f32,
    pub odo_ea_k: f32,
    pub odo_ea_base: f32,
    pub lm_visibility_range: usize,
    pub lm_visibility_prob: f32,
    // Heavy-tailed connectivity: a fraction of landmarks are "wide",
    // visible across up to num_poses/4 poses, modeling map points a real
    // trajectory revisits.
    pub wide_fraction: f32,
    // Initialization noise. The bearing factors are stiff (isigma ~600).
    // This benchmark polishes from a good init (per-step cost, the headline
    // metric, is init-independent), so the init noise is small.
    pub pose_init_pos_noise: f32,
    pub pose_init_ea_noise: f32,
}

impl Default for SceneConfig {
    fn default() -> Self {
        SceneConfig {
            num_poses: 60,
            num_landmarks: 240,
            seed: 42,
            outlier_fraction: 0.0,
            outlier_scale: 30.0,
            s_amplitude: 1.5,
            s_frequency: 0.8,
            step_size: 0.25,
            odo_pos_k: 0.10,
            odo_pos_base: 0.03,
            odo_ea_k: 0.01,
            odo_ea_base: 0.001,
            lm_visibility_range: 15,
            lm_visibility_prob: 0.75,
            wide_fraction: 0.15,
            pose_init_pos_noise: 0.05,
            pose_init_ea_noise: 0.01,
        }
    }
}

fn create_cameras() -> Vec<Camera> {
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
        cameras.push(Camera {
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

// (landmark_pos, anchor_pose_index, visibility_range)
fn ground_truth_landmarks(cfg: &SceneConfig, rng: &mut StdRng, poses: &[(vect3f, vect3f)])
    -> Vec<(vect3f, usize, usize)> {
    let wide_max = (cfg.num_poses / 4).max(cfg.lm_visibility_range);
    let wide_min = (cfg.num_poses / 8).max(cfg.lm_visibility_range);
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

    let gt_poses = ground_truth_poses(cfg);
    let gt_landmarks = ground_truth_landmarks(cfg, &mut rng, &gt_poses);
    let cameras = create_cameras();

    let drift_pos_sigma = 1000.0_f32;
    let drift_ea_sigma_deg = 1800.0_f32;
    let tilt_sigma_rad = 0.25_f32.to_radians();

    let mut scene = Scene {
        poses: Vec::new(),
        landmarks: Vec::new(),
        odo: Vec::new(),
        frines: Vec::new(),
        drift_pos_isigma: 1.0 / drift_pos_sigma,
        drift_ea_isigma: 1.0 / drift_ea_sigma_deg.to_radians(),
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
            let dist_to_anchor = pi.abs_diff(anchor_idx);
            if dist_to_anchor > range { continue; }
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
            tilt_roll: ea.x + tilt_sigma_rad * rng.sample(normal01) as f32,
            tilt_pitch: ea.y + tilt_sigma_rad * rng.sample(normal01) as f32,
        });
    }

    // Keep only observed landmarks (fixed ground truth, no init noise),
    // renumber, remap frine indices.
    let mut new_index = vec![u32::MAX; gt_landmarks.len()];
    for (li, &(lm_pos, _, _)) in gt_landmarks.iter().enumerate() {
        if !raw_frines.iter().any(|(l, ..)| *l == li) { continue; }
        new_index[li] = scene.landmarks.len() as u32;
        scene.landmarks.push(lm_pos);
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
fn v3f2(v: vect2f) -> (f64, f64) { (v.x as f64, v.y as f64) }

/// Sum of squared residuals over all four factor types (bearing, odometry,
/// pose drift, tilt). Landmarks are fixed, so only the poses are scored.
pub fn reference_cost(scene: &Scene, sol: &Solution) -> f64 {
    assert_eq!(sol.poses.len(), scene.poses.len());
    let mut cost = 0.0;

    let rots: Vec<matrix3d> = sol.poses.iter()
        .map(|(_, ea)| matrix3d::rotation_from_euler_angles(*ea))
        .collect();

    for (i, p) in scene.poses.iter().enumerate() {
        let (pos, ea) = sol.poses[i];
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

    // Bearing observations (plain Gaussian) against the fixed landmark map.
    let scale = scene.frine_isigma_scale as f64;
    for f in &scene.frines {
        let (pos, _) = sol.poses[f.pose as usize];
        let mr2w = &rots[f.pose as usize];
        let lm = v3(scene.landmarks[f.landmark as usize]);
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

// ---------------------------------------------------------------------------
// Export -- flat text the C++ runners consume (identical problem). Fixed
// landmarks are written as constants; the solution read back is poses only.
// ---------------------------------------------------------------------------

use std::fmt::Write;

// Values are stored f32 but scored (and read by the C++ runners) as f64.
// Write the exact f32->f64 upcast with full f64 precision so the C++ double
// equals the reference's f32-as-f64.
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
///   drift_pos_isigma drift_ea_isigma tilt_isigma frine_isigma_scale
///   per pose:      init_pos(3) init_ea(3) tilt_roll tilt_pitch
///   per landmark:  pos(3)   (FIXED, not optimized)
///   per frine:     pose landmark mf2r(9) camera_pos(3) isigma(2)
///   per odo:       prev cur delta_pos(3) delta_ea(3) pos_cov_r(9) pos_cov_isigma(3) ea_cov_r(9) ea_cov_isigma(3)
pub fn write_scene(scene: &Scene, path: &str) {
    let mut s = String::new();
    let _ = writeln!(s, "{} {} {} {}",
        scene.poses.len(), scene.landmarks.len(), scene.frines.len(), scene.odo.len());
    let _ = writeln!(s, "{} {} {} {}",
        scene.drift_pos_isigma as f64, scene.drift_ea_isigma as f64,
        scene.tilt_isigma as f64, scene.frine_isigma_scale as f64);
    for p in &scene.poses {
        wv3(&mut s, p.init_pos); wv3(&mut s, p.init_ea);
        let _ = writeln!(s, "{} {}", p.tilt_roll as f64, p.tilt_pitch as f64);
    }
    for l in &scene.landmarks {
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

/// Read a solution back (per-pose "x y z roll pitch yaw"), for scoring an
/// external runner against reference_cost. Landmarks are fixed, not read.
pub fn read_solution(path: &str, n_poses: usize) -> Solution {
    let text = std::fs::read_to_string(path).unwrap();
    let mut it = text.split_ascii_whitespace().map(|t| t.parse::<f64>().unwrap());
    let mut next = || it.next().unwrap();
    let poses = (0..n_poses).map(|_| {
        (vect3d::new(next(), next(), next()), vect3d::new(next(), next(), next()))
    }).collect();
    Solution { poses }
}
