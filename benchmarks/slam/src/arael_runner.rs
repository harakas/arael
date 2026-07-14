// Dedicated arael model for the heterogeneous SLAM benchmark (f64) --
// a fresh implementation, not the examples/slam_demo.rs model: drift
// regularizers use explicit stored priors (not `_value`), GPS is
// unguarded (every pose has it), and there is no plotting/graduation
// machinery. Six factor types, matching scene::reference_cost exactly.

use crate::scene::{Scene, Solution};
use arael::matrix::{matrix3d, matrix3f};
use arael::model::{CrossBlock, Param, SelfBlock, SimpleEulerAngleParam};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect2f, vect3d, vect3f};

#[arael::model]
// GPS (plain Gaussian, whitened by the covariance decomposition).
#[arael(constraint(hb_pose, {
    let raw = pose.pos - pose.gps_pos;
    let rt = pose.gps_cov_r.transpose() * raw;
    [rt.x * pose.gps_cov_isigma.x,
     rt.y * pose.gps_cov_isigma.y,
     rt.z * pose.gps_cov_isigma.z]
}))]
// Pose drift prior (to the initialization).
#[arael(constraint(hb_pose, {
    let pd = pose.pos - pose.prior_pos;
    let ed = pose.ea - pose.prior_ea;
    [pd.x * path.drift_pos_isigma,
     pd.y * path.drift_pos_isigma,
     pd.z * path.drift_pos_isigma,
     ed.x * path.drift_ea_isigma,
     ed.y * path.drift_ea_isigma,
     ed.z * path.drift_ea_isigma]
}))]
// Tilt (accelerometer constrains roll + pitch).
#[arael(constraint(hb_pose, {
    [(pose.ea.x - pose.tilt_roll) * path.tilt_isigma,
     (pose.ea.y - pose.tilt_pitch) * path.tilt_isigma]
}))]
#[derive(Clone)]
struct Pose {
    pos: Param<vect3d>,
    ea: SimpleEulerAngleParam<f64>,
    prior_pos: vect3d,
    prior_ea: vect3d,
    gps_pos: vect3d,
    gps_cov_r: matrix3d,
    gps_cov_isigma: vect3d,
    tilt_roll: f64,
    tilt_pitch: f64,
    hb_pose: SelfBlock<Pose>,
}

#[arael::model]
// Landmark drift prior.
#[arael(constraint(hb_drift, {
    let d = pointlandmark.pos - pointlandmark.prior_pos;
    [d.x * path.drift_lm_isigma,
     d.y * path.drift_lm_isigma,
     d.z * path.drift_lm_isigma]
}))]
#[derive(Clone)]
struct PointLandmark {
    pos: Param<vect3d>,
    prior_pos: vect3d,
    frines: std::vec::Vec<Frine>,
    hb_drift: SelfBlock<PointLandmark>,
}

#[arael::model]
// Bearing observation (plain Gaussian), landmark <-> pose. Feature frame
// data (mf2r / camera_pos / isigma) is inlined on the frine rather than
// kept in a separate feature collection. frine_isigma_scale is the hook
// for graduated optimization (kept at 1.0 in this outlier-free problem).
#[arael(constraint(hb, parent = lm, {
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - frine.camera_pos;
    let r_f = frine.mf2r.transpose() * r_r;
    [atan2(r_f.y, r_f.x) * frine.isigma.x * path.frine_isigma_scale,
     atan2(r_f.z, r_f.x) * frine.isigma.y * path.frine_isigma_scale]
}))]
#[derive(Clone)]
struct Frine {
    #[arael(ref = root.poses)]
    pose: Ref<Pose>,
    mf2r: matrix3d,
    camera_pos: vect3d,
    isigma: vect2d,
    hb: CrossBlock<PointLandmark, Pose>,
}

#[arael::model]
// Odometry (full 6-DOF relative motion, rotation composition + euler
// extraction).
#[arael(constraint(hb, {
    let mr2w_prev = prev.ea.rotation_matrix();
    let pos_diff = mr2w_prev.transpose() * (cur.pos - prev.pos);
    let pos_err = pos_diff - posepair.delta_pos;
    let pos_w = posepair.pos_cov_r.transpose() * pos_err;
    let mr2w_cur = cur.ea.rotation_matrix();
    let expected = mr2w_prev * posepair.delta_ea.rotation_matrix();
    let error_rot = expected.transpose() * mr2w_cur;
    let ea_err = error_rot.get_euler_angles();
    let ea_w = posepair.ea_cov_r.transpose() * ea_err;
    [pos_w.x * posepair.pos_cov_isigma.x,
     pos_w.y * posepair.pos_cov_isigma.y,
     pos_w.z * posepair.pos_cov_isigma.z,
     ea_w.x * posepair.ea_cov_isigma.x,
     ea_w.y * posepair.ea_cov_isigma.y,
     ea_w.z * posepair.ea_cov_isigma.z]
}))]
#[derive(Clone)]
struct PosePair {
    #[arael(ref = root.poses)]
    prev: Ref<Pose>,
    #[arael(ref = root.poses)]
    cur: Ref<Pose>,
    delta_pos: vect3d,
    delta_ea: vect3d,
    pos_cov_r: matrix3d,
    pos_cov_isigma: vect3d,
    ea_cov_r: matrix3d,
    ea_cov_isigma: vect3d,
    hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Path {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<PointLandmark>,
    pose_pairs: std::vec::Vec<PosePair>,
    drift_pos_isigma: f64,
    drift_ea_isigma: f64,
    drift_lm_isigma: f64,
    tilt_isigma: f64,
    frine_isigma_scale: f64,
}

// ------------------------------------------------------------------ f32
// Identical model, f32 throughout (the demo's native precision).

#[arael::model]
#[arael(constraint(hb_pose, {
    let raw = posef.pos - posef.gps_pos;
    let rt = posef.gps_cov_r.transpose() * raw;
    [rt.x * posef.gps_cov_isigma.x,
     rt.y * posef.gps_cov_isigma.y,
     rt.z * posef.gps_cov_isigma.z]
}))]
#[arael(constraint(hb_pose, {
    let pd = posef.pos - posef.prior_pos;
    let ed = posef.ea - posef.prior_ea;
    [pd.x * pathf.drift_pos_isigma, pd.y * pathf.drift_pos_isigma, pd.z * pathf.drift_pos_isigma,
     ed.x * pathf.drift_ea_isigma, ed.y * pathf.drift_ea_isigma, ed.z * pathf.drift_ea_isigma]
}))]
#[arael(constraint(hb_pose, {
    [(posef.ea.x - posef.tilt_roll) * pathf.tilt_isigma,
     (posef.ea.y - posef.tilt_pitch) * pathf.tilt_isigma]
}))]
#[derive(Clone)]
struct PoseF {
    pos: Param<vect3f>,
    ea: SimpleEulerAngleParam<f32>,
    prior_pos: vect3f,
    prior_ea: vect3f,
    gps_pos: vect3f,
    gps_cov_r: matrix3f,
    gps_cov_isigma: vect3f,
    tilt_roll: f32,
    tilt_pitch: f32,
    hb_pose: SelfBlock<PoseF, f32>,
}

#[arael::model]
#[arael(constraint(hb_drift, {
    let d = pointlandmarkf.pos - pointlandmarkf.prior_pos;
    [d.x * pathf.drift_lm_isigma, d.y * pathf.drift_lm_isigma, d.z * pathf.drift_lm_isigma]
}))]
#[derive(Clone)]
struct PointLandmarkF {
    pos: Param<vect3f>,
    prior_pos: vect3f,
    frines: std::vec::Vec<FrineF>,
    hb_drift: SelfBlock<PointLandmarkF, f32>,
}

#[arael::model]
#[arael(constraint(hb, parent = lm, {
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - frinef.camera_pos;
    let r_f = frinef.mf2r.transpose() * r_r;
    [atan2(r_f.y, r_f.x) * frinef.isigma.x * pathf.frine_isigma_scale,
     atan2(r_f.z, r_f.x) * frinef.isigma.y * pathf.frine_isigma_scale]
}))]
#[derive(Clone)]
struct FrineF {
    #[arael(ref = root.poses)]
    pose: Ref<PoseF>,
    mf2r: matrix3f,
    camera_pos: vect3f,
    isigma: vect2f,
    hb: CrossBlock<PointLandmarkF, PoseF, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let mr2w_prev = prev.ea.rotation_matrix();
    let pos_diff = mr2w_prev.transpose() * (cur.pos - prev.pos);
    let pos_err = pos_diff - posepairf.delta_pos;
    let pos_w = posepairf.pos_cov_r.transpose() * pos_err;
    let mr2w_cur = cur.ea.rotation_matrix();
    let expected = mr2w_prev * posepairf.delta_ea.rotation_matrix();
    let error_rot = expected.transpose() * mr2w_cur;
    let ea_err = error_rot.get_euler_angles();
    let ea_w = posepairf.ea_cov_r.transpose() * ea_err;
    [pos_w.x * posepairf.pos_cov_isigma.x, pos_w.y * posepairf.pos_cov_isigma.y,
     pos_w.z * posepairf.pos_cov_isigma.z, ea_w.x * posepairf.ea_cov_isigma.x,
     ea_w.y * posepairf.ea_cov_isigma.y, ea_w.z * posepairf.ea_cov_isigma.z]
}))]
#[derive(Clone)]
struct PosePairF {
    #[arael(ref = root.poses)]
    prev: Ref<PoseF>,
    #[arael(ref = root.poses)]
    cur: Ref<PoseF>,
    delta_pos: vect3f,
    delta_ea: vect3f,
    pos_cov_r: matrix3f,
    pos_cov_isigma: vect3f,
    ea_cov_r: matrix3f,
    ea_cov_isigma: vect3f,
    hb: CrossBlock<PoseF, PoseF, f32>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct PathF {
    poses: refs::Vec<PoseF>,
    landmarks: refs::Vec<PointLandmarkF>,
    pose_pairs: std::vec::Vec<PosePairF>,
    drift_pos_isigma: f32,
    drift_ea_isigma: f32,
    drift_lm_isigma: f32,
    tilt_isigma: f32,
    frine_isigma_scale: f32,
}

pub fn build(scene: &Scene) -> Path {
    let mut path = Path {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        pose_pairs: std::vec::Vec::new(),
        drift_pos_isigma: scene.drift_pos_isigma as f64,
        drift_ea_isigma: scene.drift_ea_isigma as f64,
        drift_lm_isigma: scene.drift_lm_isigma as f64,
        tilt_isigma: scene.tilt_isigma as f64,
        frine_isigma_scale: scene.frine_isigma_scale as f64,
    };
    for p in &scene.poses {
        let g = p.gps.as_ref().unwrap();
        path.poses.push(Pose {
            pos: Param::new(vect3d::from(p.init_pos)),
            ea: SimpleEulerAngleParam::new(vect3d::from(p.init_ea)),
            prior_pos: vect3d::from(p.init_pos),
            prior_ea: vect3d::from(p.init_ea),
            gps_pos: vect3d::from(g.pos),
            gps_cov_r: matrix3d::from(g.cov_r),
            gps_cov_isigma: vect3d::from(g.cov_isigma),
            tilt_roll: p.tilt_roll as f64,
            tilt_pitch: p.tilt_pitch as f64,
            hb_pose: SelfBlock::new(),
        });
    }
    // Frines are grouped by landmark.
    let mut per_lm: Vec<Vec<Frine>> = (0..scene.landmarks_init.len())
        .map(|_| Vec::new()).collect();
    for f in &scene.frines {
        let pose = path.poses.ref_at(f.pose);
        per_lm[f.landmark as usize].push(Frine {
            pose,
            mf2r: matrix3d::from(f.mf2r),
            camera_pos: vect3d::from(f.camera_pos),
            isigma: vect2d::new(f.isigma.x as f64, f.isigma.y as f64),
            hb: CrossBlock::new(),
        });
    }
    for (i, init) in scene.landmarks_init.iter().enumerate() {
        path.landmarks.push(PointLandmark {
            pos: Param::new(vect3d::from(*init)),
            prior_pos: vect3d::from(*init),
            frines: std::mem::take(&mut per_lm[i]),
            hb_drift: SelfBlock::new(),
        });
    }
    for o in &scene.odo {
        let prev = path.poses.ref_at(o.prev);
        let cur = path.poses.ref_at(o.cur);
        path.pose_pairs.push(PosePair {
            prev,
            cur,
            delta_pos: vect3d::from(o.delta_pos),
            delta_ea: vect3d::from(o.delta_ea),
            pos_cov_r: matrix3d::from(o.pos_cov_r),
            pos_cov_isigma: vect3d::from(o.pos_cov_isigma),
            ea_cov_r: matrix3d::from(o.ea_cov_r),
            ea_cov_isigma: vect3d::from(o.ea_cov_isigma),
            hb: CrossBlock::new(),
        });
    }
    path
}

fn extract(path: &Path) -> Solution {
    Solution {
        poses: path.poses.iter()
            .map(|p| (p.pos.value, matrix3d::rotation_from_euler_angles(p.ea.value).get_euler_angles()))
            .collect(),
        landmarks: path.landmarks.iter().map(|l| l.pos.value).collect(),
    }
}

fn build_f32(scene: &Scene) -> PathF {
    let mut path = PathF {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        pose_pairs: std::vec::Vec::new(),
        drift_pos_isigma: scene.drift_pos_isigma,
        drift_ea_isigma: scene.drift_ea_isigma,
        drift_lm_isigma: scene.drift_lm_isigma,
        tilt_isigma: scene.tilt_isigma,
        frine_isigma_scale: scene.frine_isigma_scale,
    };
    for p in &scene.poses {
        let g = p.gps.as_ref().unwrap();
        path.poses.push(PoseF {
            pos: Param::new(p.init_pos),
            ea: SimpleEulerAngleParam::new(p.init_ea),
            prior_pos: p.init_pos,
            prior_ea: p.init_ea,
            gps_pos: g.pos,
            gps_cov_r: g.cov_r,
            gps_cov_isigma: g.cov_isigma,
            tilt_roll: p.tilt_roll,
            tilt_pitch: p.tilt_pitch,
            hb_pose: SelfBlock::new(),
        });
    }
    let mut per_lm: Vec<Vec<FrineF>> = (0..scene.landmarks_init.len()).map(|_| Vec::new()).collect();
    for f in &scene.frines {
        let pose = path.poses.ref_at(f.pose);
        per_lm[f.landmark as usize].push(FrineF {
            pose,
            mf2r: f.mf2r,
            camera_pos: f.camera_pos,
            isigma: f.isigma,
            hb: CrossBlock::new(),
        });
    }
    for (i, init) in scene.landmarks_init.iter().enumerate() {
        path.landmarks.push(PointLandmarkF {
            pos: Param::new(*init),
            prior_pos: *init,
            frines: std::mem::take(&mut per_lm[i]),
            hb_drift: SelfBlock::new(),
        });
    }
    for o in &scene.odo {
        let prev = path.poses.ref_at(o.prev);
        let cur = path.poses.ref_at(o.cur);
        path.pose_pairs.push(PosePairF {
            prev,
            cur,
            delta_pos: o.delta_pos,
            delta_ea: o.delta_ea,
            pos_cov_r: o.pos_cov_r,
            pos_cov_isigma: o.pos_cov_isigma,
            ea_cov_r: o.ea_cov_r,
            ea_cov_isigma: o.ea_cov_isigma,
            hb: CrossBlock::new(),
        });
    }
    path
}

fn extract_f32(path: &PathF) -> Solution {
    Solution {
        poses: path.poses.iter()
            .map(|p| (vect3d::from(p.pos.value),
                vect3d::from(matrix3f::rotation_from_euler_angles(p.ea.value).get_euler_angles())))
            .collect(),
        landmarks: path.landmarks.iter().map(|l| vect3d::from(l.pos.value)).collect(),
    }
}

// The plain fixed schedule with a problem-appropriate initial damping is
// the default (this well-initialized graph needs no adaptive driver -- no
// step is ever rejected, so a gain-ratio schedule only over-damps and
// inflates the step count; see benchmarks/pgo's fairness rules). The
// gain-ratio Nielsen driver is selected via SLAM_DRIVER=nielsen.
fn nielsen() -> bool {
    std::env::var("SLAM_DRIVER").map_or(false, |v| v == "nielsen")
}

fn cfg(max_iters: usize) -> arael::simple_lm::LmConfig<f64> {
    let cfg = arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        // Terminate at true convergence like the other solvers: the
        // LmConfig default min_iters is 5, which on this easy problem
        // (converges by step 2) would pad the count with noise iterations.
        min_iters: 1,
        max_iters,
        initial_lambda: std::env::var("SLAM_LAMBDA0").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-8),
        verbose: std::env::var("SLAM_VERBOSE").map_or(false, |v| v == "1"),
        gather_timing: std::env::var("SLAM_TIMING").map_or(false, |v| v == "1"),
        ..Default::default()
    };
    if nielsen() { cfg.with_driver(arael::simple_lm::NielsenLambdaDriver::default()) } else { cfg }
}

// SLAM_ARAEL_SOLVER selects the arael sparse backend: schur (default --
// Schur-complement, landmarks marginalized every damped solve), faer
// (plain sparse faer, landmarks-first ordering), eigen (Eigen
// SimplicialLLT, --features eigen), cholmod (CHOLMOD simplicial,
// --features cholmod), or cholmod_gpl (CHOLMOD supernodal, --features
// cholmod-gpl -- GPL-licensed module, see the arael Cargo.toml warning).
// The f32 row is always pure Rust: schur by default, faer on
// SLAM_ARAEL_SOLVER=faer.
fn solve64(params: &[f64], path: &mut Path, cfg: &arael::simple_lm::LmConfig<f64>)
    -> arael::simple_lm::LmResult<f64> {
    match std::env::var("SLAM_ARAEL_SOLVER").as_deref() {
        Ok("eigen") => {
            #[cfg(feature = "eigen")]
            return arael::simple_lm::solve_sparse_eigen(params, path, cfg);
            #[cfg(not(feature = "eigen"))]
            panic!("SLAM_ARAEL_SOLVER=eigen requires building with --features eigen");
        }
        Ok("cholmod") => {
            #[cfg(feature = "cholmod")]
            return arael::simple_lm::solve_sparse_cholmod(params, path, cfg);
            #[cfg(not(feature = "cholmod"))]
            panic!("SLAM_ARAEL_SOLVER=cholmod requires building with --features cholmod");
        }
        Ok("cholmod_gpl") => {
            #[cfg(feature = "cholmod-gpl")]
            return arael::simple_lm::solve_sparse_cholmod_supernodal(params, path, cfg);
            #[cfg(not(feature = "cholmod-gpl"))]
            panic!("SLAM_ARAEL_SOLVER=cholmod_gpl requires building with --features cholmod-gpl");
        }
        Ok("faer") => {
            // Plain sparse faer: the whole system, factorized as one. The
            // policy is what pins it there -- left to itself the backend
            // would find the landmarks and marginalize them, which is the
            // other row.
            arael::simple_lm::lm_solve(
                params,
                &mut arael::simple_lm::SparseFaer::new()
                    .with_policy(arael::simple_lm::SchurPolicy::Never),
                path,
                cfg,
            )
        }
        _ => {
            // Default: the backend decides for itself. It finds the
            // landmarks in the model's coupling graph and marginalizes them,
            // factorizing only the reduced pose system.
            arael::simple_lm::lm_solve(
                params,
                &mut arael::simple_lm::SparseFaer::new(),
                path,
                cfg,
            )
        }
    }
}

/// The arael model cost at the initial estimate -- for the harness to
/// cross-check against scene::reference_cost.
pub fn initial_cost(scene: &Scene) -> f64 {
    use arael::simple_lm::LmProblem;
    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize64(&mut params);
    path.calc_cost(&params)
}

/// Write the arael Hessian (J^T J) sparsity/fill pattern as a PNG at the initial
/// estimate -- one pixel per parameter, black where nonzero, mirrored to the
/// full symmetric matrix. Env-gated from main (SLAM_HESSIAN_BITMAP).
pub fn write_hessian_bitmap(scene: &Scene, out: &str) {
    use arael::simple_lm::LmProblem;
    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
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

fn cfg32(max_iters: usize, poses: usize) -> arael::simple_lm::LmConfig<f32> {
    // Problem-appropriate initial damping, per pose count. At the small
    // (60-pose) size the f32 solution lands a hair above the 1e-5 stop
    // threshold at 1e-8 and then bounces in the f32 noise floor (a
    // termination/precision interaction, not divergence); a slightly
    // heavier 1e-7 makes the last step small enough to stop cleanly. The
    // larger sizes are clean at 1e-8 (1e-7 would grind there instead --
    // there is no single value clean at every size). SLAM_LAMBDA0 overrides.
    let default_lambda = if poses <= 60 { 1e-7 } else { 1e-8 };
    let cfg = arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        min_iters: 1,
        max_iters,
        initial_lambda: std::env::var("SLAM_LAMBDA0").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(default_lambda),
        verbose: std::env::var("SLAM_VERBOSE").map_or(false, |v| v == "1"),
        gather_timing: std::env::var("SLAM_TIMING").map_or(false, |v| v == "1"),
        ..Default::default()
    };
    if nielsen() { cfg.with_driver(arael::simple_lm::NielsenLambdaDriver::default()) } else { cfg }
}

fn solve32(params: &[f32], path: &mut PathF, cfg: &arael::simple_lm::LmConfig<f32>)
    -> arael::simple_lm::LmResult<f32> {
    if std::env::var("SLAM_ARAEL_SOLVER").as_deref() == Ok("faer") {
        return arael::simple_lm::lm_solve(
            params,
            &mut arael::simple_lm::SparseFaerF32::new()
                .with_policy(arael::simple_lm::SchurPolicy::Never),
            path,
            cfg,
        );
    }
    arael::simple_lm::lm_solve(
        params, &mut arael::simple_lm::SparseFaerF32::new(), path, cfg)
}

// Capped single solve (no timing) -- used for peak-memory measurement.
pub fn run_capped(scene: &Scene, max_iters: usize) -> Solution {
    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize64(&mut params);
    let result = solve64(&params, &mut path, &cfg(max_iters));
    path.deserialize64(&result.x);
    extract(&path)
}

pub fn run_f32_capped(scene: &Scene, max_iters: usize) -> Solution {
    let mut path = build_f32(scene);
    let mut params: Vec<f32> = Vec::new();
    path.serialize32(&mut params);
    let result = solve32(&params, &mut path, &cfg32(max_iters, scene.poses.len()));
    path.deserialize32(&result.x);
    extract_f32(&path)
}


// ---------------------------------------------------------------- pipeline

// Problem-appropriate initial damping. The f32 build wants more of it at the
// small size: at 60 poses the f32 solution lands a hair above the 1e-5 stop
// threshold at 1e-8 and then bounces in the f32 noise floor -- a
// termination/precision interaction, not divergence -- and 1e-7 makes the last
// step small enough to stop cleanly. The larger sizes are clean at 1e-8 (1e-7
// would grind there instead; no single value is clean at every size).
impl bench_harness::arael::Model for Path {
    type Scalar = f64;
    type Input = Scene;
    type Solution = Solution;
    fn lambda0(_: &Scene) -> f64 { 1e-8 }
    fn build(scene: &Scene) -> Self { build(scene) }
    fn serialize(&mut self, out: &mut Vec<f64>) { self.serialize64(out); }
    fn deserialize(&mut self, x: &[f64]) { self.deserialize64(x); }
    fn solution(&self) -> Solution { extract(self) }
    fn solve(_: &Self::Input, params: &[f64], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f64>)
        -> arael::simple_lm::LmResult<f64> { solve64(params, m, cfg) }
}

impl bench_harness::arael::Model for PathF {
    type Scalar = f32;
    type Input = Scene;
    type Solution = Solution;
    fn lambda0(scene: &Scene) -> f64 {
        if scene.poses.len() <= 60 { 1e-7 } else { 1e-8 }
    }
    fn build(scene: &Scene) -> Self { build_f32(scene) }
    fn serialize(&mut self, out: &mut Vec<f32>) { self.serialize32(out); }
    fn deserialize(&mut self, x: &[f32]) { self.deserialize32(x); }
    fn solution(&self) -> Solution { extract_f32(self) }
    fn solve(_: &Self::Input, params: &[f32], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f32>)
        -> arael::simple_lm::LmResult<f32> { solve32(params, m, cfg) }
}

pub type RunOut = bench_harness::table::Row<Solution>;

pub fn run(scene: &Scene) -> RunOut { bench_harness::arael::run::<Path>(scene) }
pub fn run_f32(scene: &Scene) -> RunOut { bench_harness::arael::run::<PathF>(scene) }
