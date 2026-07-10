// Dedicated arael model for the localization benchmark (f64 + f32).
// Localization, not SLAM: landmarks are fixed constants (no Param, no drift),
// so the bearing factor carries no landmark derivatives -- its Hessian block
// is a REMOTE block on the pose (`pose.hb_pose`), not a CrossBlock. No GPS:
// the fixed map pins the frame. Drift regularizers use explicit stored priors
// (not `_value`). Four factor types, matching scene::reference_cost exactly.

use crate::scene::{Scene, Solution};
use arael::matrix::{matrix3d, matrix3f};
use arael::model::{CrossBlock, Model, Param, SelfBlock, SimpleEulerAngleParam};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect2f, vect3d, vect3f};

// ------------------------------------------------------------------ f64

#[arael::model]
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
struct Pose {
    pos: Param<vect3d>,
    ea: SimpleEulerAngleParam<f64>,
    prior_pos: vect3d,
    prior_ea: vect3d,
    tilt_roll: f64,
    tilt_pitch: f64,
    hb_pose: SelfBlock<Pose>,
}

#[arael::model]
// Bearing observation (plain Gaussian) against a FIXED landmark. The block
// is remote (pose.hb_pose) -- only the pose has parameters, so the landmark
// contributes no derivatives and the Hessian lands on the pose's self-block.
#[arael(constraint(pose.hb_pose, parent = lm, {
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - frine.camera_pos;
    let r_f = frine.mf2r.transpose() * r_r;
    [atan2(r_f.y, r_f.x) * frine.isigma.x * path.frine_isigma_scale,
     atan2(r_f.z, r_f.x) * frine.isigma.y * path.frine_isigma_scale]
}))]
struct Frine {
    #[arael(ref = root.poses)]
    pose: Ref<Pose>,
    mf2r: matrix3d,
    camera_pos: vect3d,
    isigma: vect2d,
}

#[arael::model]
// Known landmark -- fixed position, not optimized. Holds the observations of
// it; the constraint block lives on the observing pose.
struct PointLandmark {
    pos: vect3d,
    frines: std::vec::Vec<Frine>,
}

#[arael::model]
// Odometry (full 6-DOF relative motion).
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
struct Path {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<PointLandmark>,
    pose_pairs: std::vec::Vec<PosePair>,
    drift_pos_isigma: f64,
    drift_ea_isigma: f64,
    tilt_isigma: f64,
    frine_isigma_scale: f64,
}

// ------------------------------------------------------------------ f32
// Identical model, f32 throughout (the demo's native precision).

#[arael::model]
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
struct PoseF {
    pos: Param<vect3f>,
    ea: SimpleEulerAngleParam<f32>,
    prior_pos: vect3f,
    prior_ea: vect3f,
    tilt_roll: f32,
    tilt_pitch: f32,
    hb_pose: SelfBlock<PoseF, f32>,
}

#[arael::model]
#[arael(constraint(pose.hb_pose, parent = lm, {
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - frinef.camera_pos;
    let r_f = frinef.mf2r.transpose() * r_r;
    [atan2(r_f.y, r_f.x) * frinef.isigma.x * pathf.frine_isigma_scale,
     atan2(r_f.z, r_f.x) * frinef.isigma.y * pathf.frine_isigma_scale]
}))]
struct FrineF {
    #[arael(ref = root.poses)]
    pose: Ref<PoseF>,
    mf2r: matrix3f,
    camera_pos: vect3f,
    isigma: vect2f,
}

#[arael::model]
struct PointLandmarkF {
    pos: vect3f,
    frines: std::vec::Vec<FrineF>,
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
struct PathF {
    poses: refs::Vec<PoseF>,
    landmarks: refs::Vec<PointLandmarkF>,
    pose_pairs: std::vec::Vec<PosePairF>,
    drift_pos_isigma: f32,
    drift_ea_isigma: f32,
    tilt_isigma: f32,
    frine_isigma_scale: f32,
}

// ---------------------------------------------------------------- build/extract

fn build(scene: &Scene) -> Path {
    let mut path = Path {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        pose_pairs: std::vec::Vec::new(),
        drift_pos_isigma: scene.drift_pos_isigma as f64,
        drift_ea_isigma: scene.drift_ea_isigma as f64,
        tilt_isigma: scene.tilt_isigma as f64,
        frine_isigma_scale: scene.frine_isigma_scale as f64,
    };
    for p in &scene.poses {
        path.poses.push(Pose {
            pos: Param::new(vect3d::from(p.init_pos)),
            ea: SimpleEulerAngleParam::new(vect3d::from(p.init_ea)),
            prior_pos: vect3d::from(p.init_pos),
            prior_ea: vect3d::from(p.init_ea),
            tilt_roll: p.tilt_roll as f64,
            tilt_pitch: p.tilt_pitch as f64,
            hb_pose: SelfBlock::new(),
        });
    }
    let mut per_lm: Vec<Vec<Frine>> = (0..scene.landmarks.len()).map(|_| Vec::new()).collect();
    for f in &scene.frines {
        let pose = path.poses.ref_at(f.pose);
        per_lm[f.landmark as usize].push(Frine {
            pose,
            mf2r: matrix3d::from(f.mf2r),
            camera_pos: vect3d::from(f.camera_pos),
            isigma: vect2d::new(f.isigma.x as f64, f.isigma.y as f64),
        });
    }
    for (i, lm) in scene.landmarks.iter().enumerate() {
        path.landmarks.push(PointLandmark {
            pos: vect3d::from(*lm),
            frines: std::mem::take(&mut per_lm[i]),
        });
    }
    for o in &scene.odo {
        let prev = path.poses.ref_at(o.prev);
        let cur = path.poses.ref_at(o.cur);
        path.pose_pairs.push(PosePair {
            prev, cur,
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
    }
}

fn build_f32(scene: &Scene) -> PathF {
    let mut path = PathF {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        pose_pairs: std::vec::Vec::new(),
        drift_pos_isigma: scene.drift_pos_isigma,
        drift_ea_isigma: scene.drift_ea_isigma,
        tilt_isigma: scene.tilt_isigma,
        frine_isigma_scale: scene.frine_isigma_scale,
    };
    for p in &scene.poses {
        path.poses.push(PoseF {
            pos: Param::new(p.init_pos),
            ea: SimpleEulerAngleParam::new(p.init_ea),
            prior_pos: p.init_pos,
            prior_ea: p.init_ea,
            tilt_roll: p.tilt_roll,
            tilt_pitch: p.tilt_pitch,
            hb_pose: SelfBlock::new(),
        });
    }
    let mut per_lm: Vec<Vec<FrineF>> = (0..scene.landmarks.len()).map(|_| Vec::new()).collect();
    for f in &scene.frines {
        let pose = path.poses.ref_at(f.pose);
        per_lm[f.landmark as usize].push(FrineF {
            pose,
            mf2r: f.mf2r,
            camera_pos: f.camera_pos,
            isigma: f.isigma,
        });
    }
    for (i, lm) in scene.landmarks.iter().enumerate() {
        path.landmarks.push(PointLandmarkF {
            pos: *lm,
            frines: std::mem::take(&mut per_lm[i]),
        });
    }
    for o in &scene.odo {
        let prev = path.poses.ref_at(o.prev);
        let cur = path.poses.ref_at(o.cur);
        path.pose_pairs.push(PosePairF {
            prev, cur,
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
    }
}

// ---------------------------------------------------------------- solve

pub struct RunOut {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    pub iterations: usize,
    pub accepted: usize,
    pub solution: Solution,
}

fn nielsen() -> bool {
    std::env::var("LOC_DRIVER").map_or(false, |v| v == "nielsen")
}

fn cfg(max_iters: usize) -> arael::simple_lm::LmConfig<f64> {
    let cfg = arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        min_iters: 1,
        max_iters,
        initial_lambda: std::env::var("LOC_LAMBDA0").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-8),
        verbose: std::env::var("LOC_VERBOSE").map_or(false, |v| v == "1"),
        ..Default::default()
    };
    if nielsen() { cfg.with_driver(arael::simple_lm::NielsenLambdaDriver::default()) } else { cfg }
}

// Localization is block-tridiagonal: poses couple only through consecutive
// odometry, and the bearings hit FIXED landmarks so they add no pose-to-pose
// coupling. The default solver is therefore arael's band Cholesky with
// kd = 2*6 - 1 = 11 (6-DOF poses laid out consecutively), matching loc_demo.
// LOC_ARAEL_SOLVER=faer overrides with the general sparse solver.
const BAND_KD: usize = 11;

fn solve64(params: &[f64], path: &mut Path, cfg: &arael::simple_lm::LmConfig<f64>)
    -> arael::simple_lm::LmResult<f64> {
    match std::env::var("LOC_ARAEL_SOLVER").as_deref() {
        Ok("faer") => arael::simple_lm::solve_sparse_faer(params, path, cfg),
        _ => arael::simple_lm::solve_band(params, BAND_KD, path, cfg),
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

pub fn run(scene: &Scene) -> RunOut {
    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize64(&mut params);

    let t0 = std::time::Instant::now();
    let _ = solve64(&params, &mut path, &cfg(1));
    let first_iter_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = std::time::Instant::now();
    let result = solve64(&params, &mut path, &cfg(200));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    path.deserialize64(&result.x);

    RunOut {
        solve_ms, first_iter_ms,
        iterations: result.iterations,
        accepted: result.accepted_iterations,
        solution: extract(&path),
    }
}

fn cfg32(max_iters: usize, poses: usize) -> arael::simple_lm::LmConfig<f32> {
    let default_lambda = if poses <= 60 { 1e-7 } else { 1e-8 };
    let cfg = arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        min_iters: 1,
        max_iters,
        initial_lambda: std::env::var("LOC_LAMBDA0").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(default_lambda),
        verbose: std::env::var("LOC_VERBOSE").map_or(false, |v| v == "1"),
        ..Default::default()
    };
    if nielsen() { cfg.with_driver(arael::simple_lm::NielsenLambdaDriver::default()) } else { cfg }
}

fn solve32(params: &[f32], path: &mut PathF, cfg: &arael::simple_lm::LmConfig<f32>)
    -> arael::simple_lm::LmResult<f32> {
    match std::env::var("LOC_ARAEL_SOLVER").as_deref() {
        Ok("faer") => arael::simple_lm::solve_sparse_faer_f32(params, path, cfg),
        _ => arael::simple_lm::solve_band_f32(params, BAND_KD, path, cfg),
    }
}

/// One full solve with per-phase timing gathered (LOC_TIMING mode).
pub fn run_timed_once(scene: &Scene) -> arael::simple_lm::LmTiming {
    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize64(&mut params);
    let mut c = cfg(200);
    c.gather_timing = true;
    solve64(&params, &mut path, &c).timing.unwrap()
}

/// One full f32 solve with per-phase timing gathered (LOC_TIMING mode).
pub fn run_timed_once_f32(scene: &Scene) -> arael::simple_lm::LmTiming {
    let mut path = build_f32(scene);
    let mut params: Vec<f32> = Vec::new();
    path.serialize32(&mut params);
    let mut c = cfg32(200, scene.poses.len());
    c.gather_timing = true;
    solve32(&params, &mut path, &c).timing.unwrap()
}

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

pub fn run_f32(scene: &Scene) -> RunOut {
    let mut path = build_f32(scene);
    let mut params: Vec<f32> = Vec::new();
    path.serialize32(&mut params);

    let t0 = std::time::Instant::now();
    let _ = solve32(&params, &mut path, &cfg32(1, scene.poses.len()));
    let first_iter_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = std::time::Instant::now();
    let result = solve32(&params, &mut path, &cfg32(200, scene.poses.len()));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    path.deserialize32(&result.x);

    RunOut {
        solve_ms, first_iter_ms,
        iterations: result.iterations,
        accepted: result.accepted_iterations,
        solution: extract_f32(&path),
    }
}
