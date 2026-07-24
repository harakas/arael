// Band vs Dense solver benchmark.
//
// Uses the same localization model as loc_demo with varying numbers of poses.
// Compares dense Cholesky, pure Rust band Cholesky, and LAPACK band Cholesky
// (when the lapack feature is enabled).
//
// Run: cargo run -r --example bench_band
// With LAPACK: cargo run -r --features lapack --example bench_band

use arael::model::{Param, SelfBlock, CrossBlock, SimpleEulerAngleParam};
use arael::vect::{vect3f, vect2f};
use arael::matrix::matrix3f;
use arael::refs::{self, Ref};
use arael::geometry::Camera;

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;

// ---------------------------------------------------------------------------
// Model structs (same as loc_demo)
// ---------------------------------------------------------------------------

#[arael::model]
#[allow(dead_code)]
struct PointFeature {
    pixel: vect2f,
    mf2r: matrix3f,
    #[arael(skip)]
    camera: Ref<Camera>,
    camera_pos: vect3f,
    isigma: vect2f,
}

#[arael::model]
struct PoseInfo {
    delta_pos: vect3f,
    delta_ea: vect3f,
    delta_pos_cov_r: matrix3f,
    delta_pos_cov_isigma: vect3f,
    delta_ea_cov_r: matrix3f,
    delta_ea_cov_isigma: vect3f,
    tilt_roll: f32,
    tilt_pitch: f32,
    features: refs::Vec<PointFeature>,
}

fn decompose_cov(cov: matrix3f) -> (matrix3f, vect3f) {
    let (r, d) = cov.symmetric_eigen();
    let isigma = vect3f::new(1.0 / d.x.sqrt(), 1.0 / d.y.sqrt(), 1.0 / d.z.sqrt());
    (r, isigma)
}

#[arael::model]
#[arael(constraint(hb_pose, {
    let pos_drift = pose.pos - pose.pos_value;
    let ea_drift = pose.ea - pose.ea_value;
    [pos_drift.x * path.drift_pos_isigma, pos_drift.y * path.drift_pos_isigma, pos_drift.z * path.drift_pos_isigma,
     ea_drift.x * path.drift_ea_isigma, ea_drift.y * path.drift_ea_isigma, ea_drift.z * path.drift_ea_isigma]
}))]
#[arael(constraint(hb_pose, {
    [(pose.ea.x - pose.info.tilt_roll) * path.tilt_isigma,
     (pose.ea.y - pose.info.tilt_pitch) * path.tilt_isigma]
}))]
struct Pose {
    pos: Param<vect3f>,
    ea: SimpleEulerAngleParam<f32>,
    info: PoseInfo,
    hb_pose: SelfBlock<Pose, f32>,
}

#[arael::model]
struct PointLandmark {
    pos: vect3f,
    frines: std::vec::Vec<PointFrine>,
}

#[arael::model]
#[arael(constraint(pose.hb_pose, parent=lm, {
    let gamma = path.gamma;
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - feature.camera_pos;
    let r_f = feature.mf2r.transpose() * r_r;
    let plain1 = atan2(r_f.y, r_f.x) * feature.isigma.x;
    let plain2 = atan2(r_f.z, r_f.x) * feature.isigma.y;
    let err1 = gamma * atan(plain1 / gamma);
    let err2 = gamma * atan(plain2 / gamma);
    [err1, err2]
}))]
struct PointFrine {
    #[arael(ref = root.poses)]
    pose: Ref<Pose>,
    #[arael(ref = pose.info.features)]
    feature: Ref<PointFeature>,
}

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
    [pos_w.x * cur.info.delta_pos_cov_isigma.x, pos_w.y * cur.info.delta_pos_cov_isigma.y, pos_w.z * cur.info.delta_pos_cov_isigma.z,
     ea_w.x * cur.info.delta_ea_cov_isigma.x, ea_w.y * cur.info.delta_ea_cov_isigma.y, ea_w.z * cur.info.delta_ea_cov_isigma.z]
}))]
struct PosePair {
    #[arael(ref = root.poses)]
    prev: Ref<Pose>,
    #[arael(ref = root.poses)]
    cur: Ref<Pose>,
    hb: CrossBlock<Pose, Pose, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct Path {
    poses: refs::Deque<Pose>,
    landmarks: refs::Arena<PointLandmark>,
    pose_pairs: std::vec::Vec<PosePair>,
    gamma: f32,
    drift_pos_isigma: f32,
    drift_ea_isigma: f32,
    tilt_isigma: f32,
}

// ---------------------------------------------------------------------------
// Scene generation (simplified from loc_demo)
// ---------------------------------------------------------------------------

struct SceneConfig {
    num_poses: usize,
    num_landmarks: usize,
    seed: u64,
    outlier_fraction: f32,
    outlier_scale: f32,
    s_amplitude: f32,
    s_frequency: f32,
    step_size: f32,
    odo_pos_k: f32,
    odo_pos_base: f32,
    odo_ea_k: f32,
    odo_ea_base: f32,
}

impl Default for SceneConfig {
    fn default() -> Self {
        SceneConfig {
            num_poses: 20, num_landmarks: 40, seed: 42,
            outlier_fraction: 0.5, outlier_scale: 30.0,
            s_amplitude: 1.5, s_frequency: 0.8, step_size: 0.25,
            odo_pos_k: 0.10, odo_pos_base: 0.03, odo_ea_k: 0.01, odo_ea_base: 0.001,
        }
    }
}

fn create_cameras() -> refs::Vec<Camera> {
    let mut cameras = refs::Vec::new();
    let w = 1024u32;
    let h = 768u32;
    let fov_deg = 80.0_f32;
    let fx = (w as f32 / 2.0) / (fov_deg / 2.0).to_radians().tan();
    let fy = fx;
    let n = 5;
    for i in 0..n {
        let yaw = (i as f32) * (360.0_f32 / n as f32).to_radians();
        let (sy, cy) = yaw.sin_cos();
        let mc2r = matrix3f::from_cols(
            vect3f::new(-sy, cy, 0.0),
            vect3f::new(0.0, 0.0, -1.0),
            vect3f::new(cy, sy, 0.0),
        );
        cameras.push(Camera {
            fx, fy, cx: w as f32 / 2.0, cy: h as f32 / 2.0,
            width: w, height: h,
            camera_pos: vect3f::new(cy * 0.1, sy * 0.1, 0.3),
            mc2r,
        });
    }
    cameras
}

fn build_path(cfg: &SceneConfig) -> Path {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let normal01 = Normal::new(0.0, 1.0).unwrap();
    let cameras = create_cameras();

    // Generate GT poses
    let mut gt_poses = Vec::new();
    let mut t = 0.0_f32;
    for _ in 0..cfg.num_poses {
        let x = t;
        let y = cfg.s_amplitude * (cfg.s_frequency * t).sin();
        let dy = cfg.s_amplitude * cfg.s_frequency * (cfg.s_frequency * t).cos();
        let yaw = dy.atan2(1.0);
        gt_poses.push((vect3f::new(x, y, 0.0), vect3f::new(0.0, 0.0, yaw)));
        t += cfg.step_size;
    }

    // Generate GT landmarks
    let mut gt_landmarks = Vec::new();
    for _ in 0..cfg.num_landmarks {
        loop {
            let anchor = &gt_poses[rng.random_range(0..gt_poses.len())].0;
            let angle = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
            let dist = 5.0 + rng.random::<f32>() * 25.0;
            let lm = vect3f::new(anchor.x + dist * angle.cos(), anchor.y + dist * angle.sin(), rng.random::<f32>() * 2.0);
            let min_dist = gt_poses.iter().map(|(p, _)| (lm - *p).norm()).fold(f32::MAX, f32::min);
            if min_dist >= 5.0 && min_dist <= 30.0 { gt_landmarks.push(lm); break; }
        }
    }

    let mut path = Path {
        poses: refs::Deque::new(),
        landmarks: refs::Arena::new(),
        pose_pairs: std::vec::Vec::new(),
        gamma: 2.0 * (25.0_f32).sqrt() / std::f32::consts::PI,
        drift_pos_isigma: 1.0 / 1000.0,
        drift_ea_isigma: 1.0 / 1800.0_f32.to_radians(),
        tilt_isigma: 1.0 / 0.25_f32.to_radians(),
    };

    let mut frine_data: Vec<(usize, usize, Ref<PointFeature>)> = Vec::new();

    for (pi, &(pos, ea)) in gt_poses.iter().enumerate() {
        let mr2w = matrix3f::rotation_from_euler_angles(ea);
        let (delta_pos, delta_ea) = if pi == 0 {
            (vect3f::default(), vect3f::default())
        } else {
            let (pp, pe) = gt_poses[pi - 1];
            let pmr = matrix3f::rotation_from_euler_angles(pe).transpose();
            (pmr * (pos - pp), (pmr * mr2w).get_euler_angles())
        };

        let dp_norm = delta_pos.norm().max(0.01);
        let de_norm = delta_ea.norm().max(0.001);
        let pos_sigma = vect3f::new(
            cfg.odo_pos_k * dp_norm + cfg.odo_pos_base,
            (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5,
            (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5);
        let ea_sigma = vect3f::new(
            cfg.odo_ea_k * de_norm + cfg.odo_ea_base,
            cfg.odo_ea_k * de_norm + cfg.odo_ea_base,
            cfg.odo_ea_k * de_norm + cfg.odo_ea_base);
        let delta_pos_cov = matrix3f::from_elements(
            pos_sigma.x.powi(2), 0.0, 0.0, 0.0, pos_sigma.y.powi(2), 0.0, 0.0, 0.0, pos_sigma.z.powi(2));
        let delta_ea_cov = matrix3f::from_elements(
            ea_sigma.x.powi(2), 0.0, 0.0, 0.0, ea_sigma.y.powi(2), 0.0, 0.0, 0.0, ea_sigma.z.powi(2));

        let mut features: refs::Vec<PointFeature> = refs::Vec::new();
        for (li, &lm_pos) in gt_landmarks.iter().enumerate() {
            for cam_ref in cameras.refs() {
                let cam = &cameras[cam_ref];
                let p_cam = cam.world_to_camera(lm_pos, pos, mr2w);
                if p_cam.z < 0.5 { continue; }
                let pixel = cam.project(p_cam);
                if !cam.is_visible(pixel) { continue; }
                let is_outlier = rng.random::<f32>() < cfg.outlier_fraction;
                let ns = if is_outlier { cfg.outlier_scale } else { 1.0 };
                let noisy_pixel = vect2f::new(pixel.x + ns * (rng.random::<f32>() * 2.0 - 1.0), pixel.y + ns * (rng.random::<f32>() * 2.0 - 1.0));
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
                let feat_ref = features.push(PointFeature { pixel: noisy_pixel, mf2r, camera: cam_ref, camera_pos: cam.camera_pos, isigma });
                frine_data.push((li, pi, feat_ref));
            }
        }

        let noisy_pos = vect3f::new(pos.x + 0.1 * rng.sample(normal01) as f32, pos.y + 0.1 * rng.sample(normal01) as f32, pos.z + 0.1 * rng.sample(normal01) as f32);
        let noisy_ea = vect3f::new(ea.x + 0.02 * rng.sample(normal01) as f32, ea.y + 0.02 * rng.sample(normal01) as f32, ea.z + 0.02 * rng.sample(normal01) as f32);
        let (dpcr, dpci) = decompose_cov(delta_pos_cov);
        let (decr, deci) = decompose_cov(delta_ea_cov);

        path.poses.push_back(Pose {
            pos: Param::new(noisy_pos), ea: SimpleEulerAngleParam::new(noisy_ea),
            info: PoseInfo {
                delta_pos, delta_ea, delta_pos_cov_r: dpcr, delta_pos_cov_isigma: dpci,
                delta_ea_cov_r: decr, delta_ea_cov_isigma: deci,
                tilt_roll: ea.x + 0.25_f32.to_radians() * rng.sample(normal01) as f32,
                tilt_pitch: ea.y + 0.25_f32.to_radians() * rng.sample(normal01) as f32,
                features,
            },
            hb_pose: SelfBlock::new(),
        });
    }

    let pose_refs: Vec<Ref<Pose>> = path.poses.refs().collect();
    for (li, &lm_pos) in gt_landmarks.iter().enumerate() {
        let frines: Vec<PointFrine> = frine_data.iter()
            .filter(|(lmi, _, _)| *lmi == li)
            .map(|(_, pose_i, feature)| PointFrine { pose: pose_refs[*pose_i], feature: *feature })
            .collect();
        if frines.is_empty() { continue; }
        path.landmarks.push(PointLandmark { pos: lm_pos, frines });
    }

    for i in 1..pose_refs.len() {
        path.pose_pairs.push(PosePair { prev: pose_refs[i - 1], cur: pose_refs[i], hb: CrossBlock::new() });
    }

    path
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

fn main() {
    println!("Band vs Dense LM solver benchmark (localization model, f32)");
    println!("Each size runs multiple times; median reported.\n");

    #[cfg(feature = "lapack")]
    println!("{:>6} {:>6} {:>6} {:>10} {:>10} {:>10} {:>7} {:>7}",
        "poses", "params", "kd", "dense(us)", "band(us)", "lapack(us)", "b/d", "l/d");
    #[cfg(not(feature = "lapack"))]
    println!("{:>6} {:>6} {:>6} {:>10} {:>10} {:>7}",
        "poses", "params", "kd", "dense(us)", "band(us)", "b/d");

    for &num_poses in &[20, 50, 100, 200, 500] {
        let cfg = SceneConfig { num_poses, num_landmarks: num_poses * 2, ..Default::default() };
        // conservative (the Default): the timing comparison wants the same
        // fixed iteration behavior on every solver, not the fastest exit.
        let solve_config = arael::simple_lm::LmConfig::conservative();
        let kd = 11;
        let runs = if num_poses <= 50 { 20 } else if num_poses <= 200 { 5 } else { 3 };

        let mut dense_times = Vec::new();
        let mut band_times = Vec::new();
        #[cfg(feature = "lapack")]
        let mut lapack_times = Vec::new();

        let mut n = 0;
        for _ in 0..runs {
            let mut path = build_path(&cfg);
            let mut params: Vec<f32> = Vec::new();
            path.serialize32(&mut params);
            n = params.len();
            let t0 = std::time::Instant::now();
            let _r = arael::simple_lm::solve_f32(&params, &mut path, &solve_config).unwrap();
            dense_times.push(t0.elapsed().as_micros() as f64);

            let mut path = build_path(&cfg);
            let mut params: Vec<f32> = Vec::new();
            path.serialize32(&mut params);
            let t0 = std::time::Instant::now();
            let _r = arael::simple_lm::solve_band_f32(&params, kd, &mut path, &solve_config).unwrap();
            band_times.push(t0.elapsed().as_micros() as f64);

            #[cfg(feature = "lapack")]
            {
                let mut path = build_path(&cfg);
                let mut params: Vec<f32> = Vec::new();
                path.serialize32(&mut params);
                let t0 = std::time::Instant::now();
                let _r = arael::simple_lm::solve_band_lapack_f32(&params, kd, &mut path, &solve_config).unwrap();
                lapack_times.push(t0.elapsed().as_micros() as f64);
            }
        }

        dense_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        band_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let dense_med = dense_times[dense_times.len() / 2];
        let band_med = band_times[band_times.len() / 2];
        let ratio_band = if band_med > 0.0 { dense_med / band_med } else { 0.0 };

        #[cfg(feature = "lapack")]
        {
            lapack_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lapack_med = lapack_times[lapack_times.len() / 2];
            let ratio_lapack = if lapack_med > 0.0 { dense_med / lapack_med } else { 0.0 };
            println!("{:>6} {:>6} {:>6} {:>10.0} {:>10.0} {:>10.0} {:>6.1}x {:>6.1}x",
                num_poses, n, kd, dense_med, band_med, lapack_med, ratio_band, ratio_lapack);
        }
        #[cfg(not(feature = "lapack"))]
        {
            println!("{:>6} {:>6} {:>6} {:>10.0} {:>10.0} {:>6.1}x",
                num_poses, n, kd, dense_med, band_med, ratio_band);
        }
    }
}
