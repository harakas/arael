// Sparse vs Dense solver benchmark using SLAM model.
//
// SLAM has landmarks creating off-diagonal fill in the hessian -- truly sparse,
// not block-tridiagonal. This benchmarks dense Cholesky vs sparse solvers.
//
// Run:
//   cargo run -r --features faer --example bench_sparse
//   OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 cargo run -r --features faer --example bench_sparse

use arael::model::{Param, SelfBlock, CrossBlock, SimpleEulerAngleParam};
use arael::vect::{vect3f, vect2f};
use arael::matrix::matrix3f;
use arael::refs::{self, Ref};
use arael::geometry::Camera;

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;

// ---------------------------------------------------------------------------
// Model structs (same as slam_demo)
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
#[arael(constraint(hb_pose, guard = self.info.gps.is_some(), {
    let gamma = path.gamma;
    let raw = pose.pos - pose.info.gps.pos;
    let rt_raw = pose.info.gps.cov_r.transpose() * raw;
    let p0 = rt_raw.x * pose.info.gps.cov_isigma.x;
    let p1 = rt_raw.y * pose.info.gps.cov_isigma.y;
    let p2 = rt_raw.z * pose.info.gps.cov_isigma.z;
    [gamma * atan(p0 / gamma), gamma * atan(p1 / gamma), gamma * atan(p2 / gamma)]
}))]
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
    hb_pose: SelfBlock<Pose>,
}

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

#[arael::model]
#[arael(constraint(hb, parent=lm, {
    let gamma = path.gamma;
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - feature.camera_pos;
    let r_f = feature.mf2r.transpose() * r_r;
    let plain1 = atan2(r_f.y, r_f.x) * feature.isigma.x;
    let plain2 = atan2(r_f.z, r_f.x) * feature.isigma.y;
    [gamma * atan(plain1 / gamma), gamma * atan(plain2 / gamma)]
}))]
struct PointFrine {
    #[arael(ref = root.poses)]
    pose: Ref<Pose>,
    #[arael(ref = pose.info.features)]
    feature: Ref<PointFeature>,
    hb: CrossBlock<PointLandmark, Pose>,
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
    hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(root)]
struct Path {
    poses: refs::Deque<Pose>,
    landmarks: refs::Vec<PointLandmark>,
    pose_pairs: std::vec::Vec<PosePair>,
    gamma: f32,
    drift_pos_isigma: f32,
    drift_ea_isigma: f32,
    drift_lm_isigma: f32,
    tilt_isigma: f32,
}

// ---------------------------------------------------------------------------
// Scene generation (from slam_demo)
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
    gps_sigma: f32,
    gps_rel_noise: f32,
    odo_pos_k: f32,
    odo_pos_base: f32,
    odo_ea_k: f32,
    odo_ea_base: f32,
    lm_visibility_range: usize,
    lm_visibility_prob: f32,
}

impl Default for SceneConfig {
    fn default() -> Self {
        SceneConfig {
            num_poses: 20, num_landmarks: 40, seed: 42,
            outlier_fraction: 0.5, outlier_scale: 30.0,
            s_amplitude: 1.5, s_frequency: 0.8, step_size: 0.25,
            gps_sigma: 2.5, gps_rel_noise: 0.02,
            odo_pos_k: 0.10, odo_pos_base: 0.03, odo_ea_k: 0.01, odo_ea_base: 0.001,
            lm_visibility_range: 15, lm_visibility_prob: 0.75,
        }
    }
}

fn create_cameras() -> refs::Vec<Camera> {
    let mut cameras = refs::Vec::new();
    let w = 1024u32; let h = 768u32;
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

    let gps_offset = vect3f::new(
        cfg.gps_sigma * rng.sample(normal01) as f32,
        cfg.gps_sigma * rng.sample(normal01) as f32,
        cfg.gps_sigma * rng.sample(normal01) as f32);

    let mut gt_poses = Vec::new();
    let mut t = 0.0_f32;
    for _ in 0..cfg.num_poses {
        let x = t;
        let y = cfg.s_amplitude * (cfg.s_frequency * t).sin();
        let dy = cfg.s_amplitude * cfg.s_frequency * (cfg.s_frequency * t).cos();
        gt_poses.push((vect3f::new(x, y, 0.0), vect3f::new(0.0, 0.0, dy.atan2(1.0))));
        t += cfg.step_size;
    }

    let mut gt_landmarks: Vec<(vect3f, usize)> = Vec::new();
    for _ in 0..cfg.num_landmarks {
        loop {
            let anchor_idx = rng.random_range(0..gt_poses.len());
            let anchor = &gt_poses[anchor_idx].0;
            let angle = rng.random::<f32>() * 2.0 * std::f32::consts::PI;
            let dist = 5.0 + rng.random::<f32>() * 25.0;
            let lm = vect3f::new(anchor.x + dist * angle.cos(), anchor.y + dist * angle.sin(), rng.random::<f32>() * 2.0);
            let min_dist = gt_poses.iter().map(|(p, _)| (lm - *p).norm()).fold(f32::MAX, f32::min);
            if min_dist >= 5.0 && min_dist <= 30.0 { gt_landmarks.push((lm, anchor_idx)); break; }
        }
    }

    let mut path = Path {
        poses: refs::Deque::new(), landmarks: refs::Vec::new(),
        pose_pairs: Vec::new(),
        gamma: 2.0 * (25.0_f32).sqrt() / std::f32::consts::PI,
        drift_pos_isigma: 1.0 / 1000.0,
        drift_ea_isigma: 1.0 / 1800.0_f32.to_radians(),
        drift_lm_isigma: 1.0 / 1000.0,
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
        let pos_sigma = vect3f::new(cfg.odo_pos_k * dp_norm + cfg.odo_pos_base, (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5, (cfg.odo_pos_k * dp_norm + cfg.odo_pos_base) * 0.5);
        let ea_sigma = vect3f::new(cfg.odo_ea_k * de_norm + cfg.odo_ea_base, cfg.odo_ea_k * de_norm + cfg.odo_ea_base, cfg.odo_ea_k * de_norm + cfg.odo_ea_base);
        let dpc = matrix3f::from_elements(pos_sigma.x.powi(2),0.0,0.0, 0.0,pos_sigma.y.powi(2),0.0, 0.0,0.0,pos_sigma.z.powi(2));
        let dec = matrix3f::from_elements(ea_sigma.x.powi(2),0.0,0.0, 0.0,ea_sigma.y.powi(2),0.0, 0.0,0.0,ea_sigma.z.powi(2));

        let mut features: refs::Vec<PointFeature> = refs::Vec::new();
        for (li, &(lm_pos, anchor_idx)) in gt_landmarks.iter().enumerate() {
            let dist_to_anchor = if pi >= anchor_idx { pi - anchor_idx } else { anchor_idx - pi };
            if dist_to_anchor > cfg.lm_visibility_range { continue; }
            if rng.random::<f32>() > cfg.lm_visibility_prob { continue; }
            for cam_ref in cameras.refs() {
                let cam = &cameras[cam_ref];
                let p_cam = cam.world_to_camera(lm_pos, pos, mr2w);
                if p_cam.z < 0.5 { continue; }
                let pixel = cam.project(p_cam);
                if !cam.is_visible(pixel) { continue; }
                let is_outlier = rng.random::<f32>() < cfg.outlier_fraction;
                let ns = if is_outlier { cfg.outlier_scale } else { 1.0 };
                let np = vect2f::new(pixel.x + ns * (rng.random::<f32>()*2.0-1.0), pixel.y + ns * (rng.random::<f32>()*2.0-1.0));
                let dir = cam.unproject_to_robot(np);
                let cam_up = -(cam.mc2r.col(1));
                let up_proj = cam_up - dir * (cam_up * dir);
                let un = up_proj.norm();
                if un < 1e-6 { continue; }
                let col2 = up_proj * (1.0/un);
                let col1 = col2 % dir;
                let mf2r = matrix3f::from_cols(dir, col1, col2);
                let sigma = cam.pixel_angular_size(np);
                let isigma = vect2f::new(1.0/sigma.x, 1.0/sigma.y);
                let fr = features.push(PointFeature { pixel: np, mf2r, camera: cam_ref, camera_pos: cam.camera_pos, isigma });
                frine_data.push((li, pi, fr));
            }
        }

        let rel_noise = cfg.gps_rel_noise * (pos - gt_poses[0].0).norm();
        let gps_pos = vect3f::new(
            pos.x + gps_offset.x + rel_noise * rng.sample(normal01) as f32,
            pos.y + gps_offset.y + rel_noise * rng.sample(normal01) as f32,
            pos.z + gps_offset.z + rel_noise * rng.sample(normal01) as f32);
        let gps_cov = matrix3f::from_elements(cfg.gps_sigma.powi(2),0.0,0.0, 0.0,cfg.gps_sigma.powi(2),0.0, 0.0,0.0,cfg.gps_sigma.powi(2));
        let np = vect3f::new(pos.x+0.1*rng.sample(normal01) as f32, pos.y+0.1*rng.sample(normal01) as f32, pos.z+0.1*rng.sample(normal01) as f32);
        let ne = vect3f::new(ea.x+0.02*rng.sample(normal01) as f32, ea.y+0.02*rng.sample(normal01) as f32, ea.z+0.02*rng.sample(normal01) as f32);
        let (dpcr, dpci) = decompose_cov(dpc);
        let (decr, deci) = decompose_cov(dec);
        let (gcr, gci) = decompose_cov(gps_cov);

        path.poses.push_back(Pose {
            pos: Param::new(np), ea: SimpleEulerAngleParam::new(ne),
            info: PoseInfo {
                delta_pos, delta_ea, delta_pos_cov_r: dpcr, delta_pos_cov_isigma: dpci,
                delta_ea_cov_r: decr, delta_ea_cov_isigma: deci,
                gps: Some(GpsData { pos: gps_pos, cov_r: gcr, cov_isigma: gci }),
                tilt_roll: ea.x + 0.25_f32.to_radians() * rng.sample(normal01) as f32,
                tilt_pitch: ea.y + 0.25_f32.to_radians() * rng.sample(normal01) as f32,
                features,
            },
            hb_pose: SelfBlock::new(),
        });
    }

    let pose_refs: Vec<Ref<Pose>> = path.poses.refs().collect();
    for (li, &(lm_pos, _)) in gt_landmarks.iter().enumerate() {
        let noisy_lm = vect3f::new(lm_pos.x + 0.5*rng.sample(normal01) as f32, lm_pos.y + 0.5*rng.sample(normal01) as f32, lm_pos.z + 0.3*rng.sample(normal01) as f32);
        let frines: Vec<PointFrine> = frine_data.iter()
            .filter(|(lmi,_,_)| *lmi == li)
            .map(|(_,pose_i,feature)| PointFrine { pose: pose_refs[*pose_i], feature: *feature, hb: CrossBlock::new() })
            .collect();
        if frines.is_empty() { continue; }
        path.landmarks.push(PointLandmark { pos: Param::new(noisy_lm), frines, hb_drift: SelfBlock::new() });
    }

    for i in 1..pose_refs.len() {
        path.pose_pairs.push(PosePair { prev: pose_refs[i-1], cur: pose_refs[i], hb: CrossBlock::new() });
    }

    path
}

// ---------------------------------------------------------------------------
// Benchmark
// ---------------------------------------------------------------------------

fn save_sparsity_png(csc: &arael::simple_lm::CscMatrix<f64>, filename: &str) {
    let n = csc.n;
    // Create n x n image: white background, black for nonzero (symmetric)
    let mut pixels = vec![255u8; n * n];
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let i = csc.row_idx[k] as usize;
            pixels[i * n + j] = 0; // upper triangle
            pixels[j * n + i] = 0; // lower triangle (symmetric)
        }
    }

    let img = image::GrayImage::from_raw(n as u32, n as u32, pixels).unwrap();
    img.save(filename).unwrap();
    println!("Saved sparsity pattern to {} ({}x{})", filename, n, n);
}

fn main() {
    unsafe {
        std::env::set_var("OMP_NUM_THREADS", "1");
        std::env::set_var("OPENBLAS_NUM_THREADS", "1");
    }

    use arael::simple_lm::LmSolver;
    use arael::simple_lm::LmProblem;

    let num_poses = 100;
    let num_landmarks = 200;
    let cfg = SceneConfig { num_poses, num_landmarks, ..Default::default() };
    let mut path = build_path(&cfg);
    let mut params: Vec<f64> = Vec::new();
    path.serialize64(&mut params);
    let n = params.len();

    // Build sparse matrix and save sparsity pattern
    let mut coo = arael::simple_lm::CooMatrix::new(n);
    let mut grad = vec![0.0f64; n];
    path.calc_grad_hessian_sparse(&params, &mut grad, &mut coo);
    let csc = coo.to_csc().unwrap();
    let nnz = csc.nnz();
    let dense_upper = n * (n + 1) / 2;
    let fill_pct = nnz as f64 / dense_upper as f64 * 100.0;
    println!("{} poses, {} params, {} landmarks", num_poses, n, num_landmarks);
    println!("nnz={} / {} upper = {:.1}% fill", nnz, dense_upper, fill_pct);

    save_sparsity_png(&csc, "sparsity.png");

    // Assembly-only benchmark: compare COO, scatter, direct, indexed, dense
    println!("\nAssembly benchmark (20 runs, median):");
    {
        let runs = 20;

        // Dense assembly
        let mut dense_times = Vec::new();
        for _ in 0..runs {
            let mut grad = vec![0.0f64; n];
            let mut hessian = vec![0.0f64; n * n];
            let t0 = std::time::Instant::now();
            path.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);
            dense_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }

        // COO + to_csc (current default)
        let mut coo_times = Vec::new();
        for _ in 0..runs {
            let mut grad = vec![0.0f64; n];
            let mut coo2 = arael::simple_lm::CooMatrix::new(n);
            let t0 = std::time::Instant::now();
            path.calc_grad_hessian_sparse(&params, &mut grad, &mut coo2);
            let _csc2 = coo2.to_csc().unwrap();
            coo_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }

        // COO + scatter (reuse CSC structure)
        let mut scatter_times = Vec::new();
        {
            let mut coo2 = arael::simple_lm::CooMatrix::new(n);
            let mut grad = vec![0.0f64; n];
            path.calc_grad_hessian_sparse(&params, &mut grad, &mut coo2);
            let mut csc2 = coo2.to_csc().unwrap();
            let scatter_map = coo2.build_scatter_map(&csc2);
            for _ in 0..runs {
                coo2.clear();
                let t0 = std::time::Instant::now();
                path.calc_grad_hessian_sparse(&params, &mut grad, &mut coo2);
                coo2.scatter_into(&mut csc2, &scatter_map);
                scatter_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
            }
        }

        // Direct binary search (no COO)
        let mut direct_times = Vec::new();
        {
            let mut coo2 = arael::simple_lm::CooMatrix::new(n);
            let mut grad = vec![0.0f64; n];
            path.calc_grad_hessian_sparse(&params, &mut grad, &mut coo2);
            let mut csc2 = coo2.to_csc().unwrap();
            for _ in 0..runs {
                let t0 = std::time::Instant::now();
                path.calc_grad_hessian_sparse_direct(&params, &mut grad, &mut csc2);
                direct_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
            }
        }

        // Indexed (precomputed position list)
        let mut indexed_times = Vec::new();
        {
            let mut coo2 = arael::simple_lm::CooMatrix::new(n);
            let mut grad = vec![0.0f64; n];
            path.calc_grad_hessian_sparse(&params, &mut grad, &mut coo2);
            let csc2 = coo2.to_csc().unwrap();
            let positions = coo2.build_scatter_map(&csc2);
            let mut vals = vec![0.0f64; csc2.nnz()];
            for _ in 0..runs {
                let t0 = std::time::Instant::now();
                path.calc_grad_hessian_sparse_indexed(&params, &mut grad, &mut vals, &positions);
                indexed_times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
            }
        }

        fn median(v: &mut Vec<f64>) -> f64 {
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        }

        println!("  {:>12}  {:>8.0}us", "dense", median(&mut dense_times));
        println!("  {:>12}  {:>8.0}us", "coo+to_csc", median(&mut coo_times));
        println!("  {:>12}  {:>8.0}us", "coo+scatter", median(&mut scatter_times));
        println!("  {:>12}  {:>8.0}us", "direct", median(&mut direct_times));
        println!("  {:>12}  {:>8.0}us", "indexed", median(&mut indexed_times));
    }

    // Single iteration timing + correctness comparison
    println!("\nSingle-iteration timing:");

    fn time_one<S: LmSolver<f64>>(
        name: &str,
        solver: &mut S,
        problem: &mut dyn LmProblem<f64>,
        x0: &[f64],
    ) -> Vec<f64> {
        let n = x0.len();
        let mut matrix = solver.new_matrix(n);
        let mut grad = vec![0.0f64; n];
        let mut diagonal = vec![0.0f64; n];
        let mut delta = vec![0.0f64; n];

        // First assembly (COO + build for sparse, flat for dense)
        let t0 = std::time::Instant::now();
        solver.compute(problem, x0, &mut grad, &mut matrix).unwrap();
        let asm1_us = t0.elapsed().as_micros();

        solver.extract_diagonal(&matrix, &mut diagonal);

        // First solve (includes symbolic factorization for sparse solvers).
        // No min_diagonal floor here, so the damping scale IS the diagonal --
        // both parameters are shared refs, so the same slice serves for both.
        let t0 = std::time::Instant::now();

        solver.solve_damped(n, &mut matrix, &diagonal, &diagonal, 0.001, &grad, &mut delta);
        let solve1_us = t0.elapsed().as_micros();

        // Second assembly (indexed for sparse, flat for dense)
        let t0 = std::time::Instant::now();
        solver.compute(problem, x0, &mut grad, &mut matrix).unwrap();
        let asm2_us = t0.elapsed().as_micros();

        solver.extract_diagonal(&matrix, &mut diagonal);

        // Second solve (numeric only, symbolic cached)
        let t0 = std::time::Instant::now();

        solver.solve_damped(n, &mut matrix, &diagonal, &diagonal, 0.001, &grad, &mut delta);
        let solve2_us = t0.elapsed().as_micros();

        println!("  {:>12}  asm1={:>7}us  asm2={:>7}us  solve1={:>7}us  solve2={:>7}us",
            name, asm1_us, asm2_us, solve1_us, solve2_us);

        delta
    }

    // Dense (reference)
    let delta_dense;
    {
        let mut path = build_path(&cfg);
        let mut params: Vec<f64> = Vec::new();
        path.serialize64(&mut params);
        delta_dense = time_one("dense", &mut arael::simple_lm::Dense, &mut path, &params);
    }

    fn compare(name: &str, delta: &[f64], reference: &[f64]) {
        let n = delta.len();
        let ref_norm: f64 = reference.iter().map(|v| v * v).sum::<f64>().sqrt();
        let mut max_abs = 0.0f64;
        let mut max_rel = 0.0f64;
        for i in 0..n {
            let d = (delta[i] - reference[i]).abs();
            max_abs = max_abs.max(d);
            if reference[i].abs() > 1e-15 {
                max_rel = max_rel.max(d / reference[i].abs());
            }
        }
        let diff_norm: f64 = delta.iter().zip(reference).map(|(a, b)| (a - b) * (a - b)).sum::<f64>().sqrt();
        let rel_norm = diff_norm / ref_norm;
        println!("  {:>12}  vs dense: ||diff||/||ref||={:.2e}  max_abs={:.2e}  max_rel={:.2e}",
            name, rel_norm, max_abs, max_rel);
    }

    // faer
        {
        let mut path = build_path(&cfg);
        let mut params: Vec<f64> = Vec::new();
        path.serialize64(&mut params);
        let delta = time_one("faer", &mut arael::simple_lm::SparseFaer::new(), &mut path, &params);
        compare("faer", &delta, &delta_dense);
    }

    // Eigen
    #[cfg(feature = "eigen")]
    {
        let mut path = build_path(&cfg);
        let mut params: Vec<f64> = Vec::new();
        path.serialize64(&mut params);
        let delta = time_one("eigen", &mut arael::simple_lm::SparseEigen::new(), &mut path, &params);
        compare("eigen", &delta, &delta_dense);
    }

    // CHOLMOD
    #[cfg(feature = "cholmod")]
    {
        let mut path = build_path(&cfg);
        let mut params: Vec<f64> = Vec::new();
        path.serialize64(&mut params);
        let delta = time_one("cholmod", &mut arael::simple_lm::SparseCholmod::new(), &mut path, &params);
        compare("cholmod", &delta, &delta_dense);
    }

}
