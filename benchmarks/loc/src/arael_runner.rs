// Dedicated arael model for the localization benchmark (f64 + f32).
// Localization, not SLAM: landmarks are fixed constants (no Param, no drift),
// so the bearing factor carries no landmark derivatives -- its Hessian block
// is a REMOTE block on the pose (`pose.hb_pose`), not a CrossBlock. No GPS:
// the fixed map pins the frame. Drift regularizers use explicit stored priors
// (not `_value`). Four factor types, matching scene::reference_cost within
// fast_atan's approximation: both roots use `fast_atan`, so the bearing
// residuals go through arael::utils::fast_atan2 (max error < 1e-6 rad) --
// the reason for the shared INITIAL_COST_RTOL in main.rs.

use crate::scene::{Scene, Solution};
use arael::matrix::{matrix3d, matrix3f};
use arael::model::{CrossBlock, Param, SelfBlock, SimpleEulerAngleParam};
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
#[derive(Clone)]
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
#[derive(Clone)]
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
#[derive(Clone)]
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
#[arael(root, fast_atan)]
#[derive(Clone)]
pub struct Path {
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
#[derive(Clone)]
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
#[derive(Clone)]
struct FrineF {
    #[arael(ref = root.poses)]
    pose: Ref<PoseF>,
    mf2r: matrix3f,
    camera_pos: vect3f,
    isigma: vect2f,
}

#[arael::model]
#[derive(Clone)]
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
#[arael(root, f32, fast_atan)]
#[derive(Clone)]
pub struct PathF {
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

// Localization is block-tridiagonal: poses couple only through consecutive
// odometry, and the bearings hit FIXED landmarks so they add no pose-to-pose
// coupling. The default solver is therefore arael's band Cholesky with
// kd = 2*6 - 1 = 11 (6-DOF poses laid out consecutively), matching loc_demo.
// LOC_ARAEL_SOLVER=faer overrides with the general sparse solver.
const BAND_KD: usize = 11;

fn faer() -> bool {
    std::env::var("LOC_ARAEL_SOLVER").as_deref() == Ok("faer")
}

/// The backend this run resolved to, for the config header.
pub fn backend() -> String {
    if faer() { "faer".to_string() } else { format!("band kd={}", BAND_KD) }
}

fn solve64(params: &[f64], path: &mut Path, cfg: &arael::simple_lm::LmConfig<f64>)
    -> arael::simple_lm::LmResult<f64> {
    if faer() {
        arael::simple_lm::solve_sparse_faer(params, path, cfg)
    } else {
        arael::simple_lm::solve_band(params, BAND_KD, path, cfg)
    }
}

fn solve32(params: &[f32], path: &mut PathF, cfg: &arael::simple_lm::LmConfig<f32>)
    -> arael::simple_lm::LmResult<f32> {
    if faer() {
        arael::simple_lm::solve_sparse_faer_f32(params, path, cfg)
    } else {
        arael::simple_lm::solve_band_f32(params, BAND_KD, path, cfg)
    }
}

// ----------------------------------------------------------- covariance

use arael::covariance::{CovMode, Covariance};

/// One covariance-scaling run: `(N, median_ms, reps)` per query count, for the
/// band-specialized TriDiagonal and the general PerQuery, plus the AllMarginals
/// bulk cost and the last-pose std dev (the localization query, a value check).
pub struct CovScaling {
    pub n_poses: usize,
    pub tridiag_pose: Vec<(usize, f64, usize)>,
    pub perquery_pose: Vec<(usize, f64, usize)>,
    /// Just the last pose (the localization query), for TriDiagonal and PerQuery.
    pub tridiag_last: (f64, usize),
    pub perquery_last: (f64, usize),
    pub allmarg_ms: f64,
    pub allmarg_reps: usize,
    pub last_pose: usize,
    pub sd_last: Vec<f64>,
}

// Solve the loc scene (f64), then time pose-covariance recovery as the query
// count scales. The map is fixed, so H is block-tridiagonal over the pose chain:
// TriDiagonal is a band forward/backward Schur pass with no factorization,
// PerQuery factors and solves, AllMarginals is the bulk selected inverse.
pub fn cov_bench(scene: &Scene, budget_s: f64, cap: usize) -> CovScaling {
    use bench_harness::cov::{cell_cap_s, query_counts, scale_counts, spread};
    use bench_harness::probe::median_ms;
    use std::hint::black_box;
    use std::time::Duration;

    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize64(&mut params);
    let cfg = bench_harness::arael::config::<Path>(scene, 200);
    let result = solve64(&params, &mut path, &cfg);
    path.deserialize64(&result.x);
    let np = path.poses.len();
    let last = np - 1;
    let budget = Duration::from_secs_f64(budget_s);
    let cap_s = cell_cap_s();

    // Validation: last-pose std dev (the localization query).
    let sd_last = {
        let cov = path.assemble_covariance(CovMode::TriDiagonal).unwrap();
        let m = cov.marginal_cov(&path.poses[last]);
        (0..6).map(|k| m[(k, k)].sqrt()).collect()
    };

    // The localization query: just the last pose. TriDiagonal gets it from the
    // forward Schur pass alone (no backward recursion), so it is the cheapest cell.
    let tridiag_last = median_ms(budget, cap, || {
        let cov = path.assemble_covariance(CovMode::TriDiagonal).unwrap();
        black_box(cov.marginal_cov(&path.poses[last]));
    });
    let perquery_last = median_ms(budget, cap, || {
        let cov = path.assemble_covariance(CovMode::PerQuery).unwrap();
        black_box(cov.marginal_cov(&path.poses[last]));
    });

    let tridiag_pose = scale_counts(query_counts(np, true), cap_s, |n| {
        let idx = spread(0, np, n);
        median_ms(budget, cap, || {
            let cov = path.assemble_covariance(CovMode::TriDiagonal).unwrap();
            for &i in &idx {
                black_box(cov.marginal_cov(&path.poses[i]));
            }
        })
    });
    let perquery_pose = scale_counts(query_counts(np, true), cap_s, |n| {
        let idx = spread(0, np, n);
        median_ms(budget, cap, || {
            let cov = path.assemble_covariance(CovMode::PerQuery).unwrap();
            for &i in &idx {
                black_box(cov.marginal_cov(&path.poses[i]));
            }
        })
    });

    // AllMarginals: bulk selected inverse over the whole band -- every pose.
    let (allmarg_ms, allmarg_reps) = median_ms(budget, cap, || {
        black_box(path.assemble_covariance(CovMode::AllMarginals).unwrap());
    });

    CovScaling {
        n_poses: np,
        tridiag_pose,
        perquery_pose,
        tridiag_last,
        perquery_last,
        allmarg_ms,
        allmarg_reps,
        last_pose: last,
        sd_last,
    }
}

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
    /// The f32 build wants heavier damping at the small size: at 60 poses the
    /// f32 solution lands a hair above the 1e-5 stop threshold at 1e-8 and then
    /// bounces in the f32 noise floor -- a termination/precision interaction,
    /// not divergence -- and 1e-7 makes the last step small enough to stop
    /// cleanly. The larger sizes are clean at 1e-8 (1e-7 would grind there
    /// instead; no single value is clean at every size).
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

/// The arael model cost at the initial estimate -- for the harness to
/// cross-check against scene::reference_cost.
pub fn initial_cost(scene: &Scene) -> f64 {
    use arael::simple_lm::LmProblem;
    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize64(&mut params);
    path.calc_cost(&params)
}

// Capped single solve (no timing) -- the peak-memory pass, which runs one
// system alone in a process of its own.
pub fn run_capped(scene: &Scene, max_iters: usize) -> Solution {
    solve_capped::<Path>(scene, max_iters)
}

pub fn run_f32_capped(scene: &Scene, max_iters: usize) -> Solution {
    solve_capped::<PathF>(scene, max_iters)
}

fn solve_capped<M: bench_harness::arael::Model<Input = Scene, Solution = Solution>>(
    scene: &Scene, max_iters: usize) -> Solution {
    let mut model = M::build(scene);
    let mut params: Vec<M::Scalar> = Vec::new();
    model.serialize(&mut params);
    let cfg = bench_harness::arael::config::<M>(scene, max_iters);
    let result = M::solve(scene, &params, &mut model, &cfg);
    model.deserialize(&result.x);
    model.solution()
}

/// One full solve, reporting arael's per-phase breakdown (LOC_PHASES mode).
pub fn run_timed_once(scene: &Scene) -> arael::simple_lm::LmTiming {
    timed_once::<Path>(scene)
}

pub fn run_timed_once_f32(scene: &Scene) -> arael::simple_lm::LmTiming {
    timed_once::<PathF>(scene)
}

fn timed_once<M: bench_harness::arael::Model<Input = Scene>>(
    scene: &Scene) -> arael::simple_lm::LmTiming {
    let mut model = M::build(scene);
    let mut params: Vec<M::Scalar> = Vec::new();
    model.serialize(&mut params);
    let cfg = bench_harness::arael::config::<M>(scene, 200);
    M::solve(scene, &params, &mut model, &cfg).timing.expect("gather_timing is on")
}
