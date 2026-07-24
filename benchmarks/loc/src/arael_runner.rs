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
use arael::matrix::matrix3;
use arael::model::{CrossBlock, Param, SelfBlock, SimpleEulerAngleParam};
use arael::refs::{self, Ref};
use arael::utils::Float;
use arael::vect::{vect2, vect3};

// The entities are generic over the scalar; the two concrete roots
// (`Path` f64, `PathF` f32) instantiate one shared model. Bodies reach
// the root's globals through the `root` alias, which resolves under
// either root.

#[arael::model]
// Pose drift prior (to the initialization).
#[arael(constraint(hb_pose, {
    let pd = pose.pos - pose.prior_pos;
    let ed = pose.ea - pose.prior_ea;
    [pd.x * root.drift_pos_isigma,
     pd.y * root.drift_pos_isigma,
     pd.z * root.drift_pos_isigma,
     ed.x * root.drift_ea_isigma,
     ed.y * root.drift_ea_isigma,
     ed.z * root.drift_ea_isigma]
}))]
// Tilt (accelerometer constrains roll + pitch).
#[arael(constraint(hb_pose, {
    [(pose.ea.x - pose.tilt_roll) * root.tilt_isigma,
     (pose.ea.y - pose.tilt_pitch) * root.tilt_isigma]
}))]
#[derive(Clone)]
struct Pose<T: Float> {
    pos: Param<vect3<T>>,
    ea: SimpleEulerAngleParam<T>,
    prior_pos: vect3<T>,
    prior_ea: vect3<T>,
    tilt_roll: T,
    tilt_pitch: T,
    hb_pose: SelfBlock<Pose<T>, T>,
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
    [atan2(r_f.y, r_f.x) * frine.isigma.x * root.frine_isigma_scale,
     atan2(r_f.z, r_f.x) * frine.isigma.y * root.frine_isigma_scale]
}))]
#[derive(Clone)]
struct Frine<T: Float> {
    #[arael(ref = root.poses)]
    pose: Ref<Pose<T>>,
    mf2r: matrix3<T>,
    camera_pos: vect3<T>,
    isigma: vect2<T>,
}

#[arael::model]
// Known landmark -- fixed position, not optimized. Holds the observations of
// it; the constraint block lives on the observing pose.
#[derive(Clone)]
struct PointLandmark<T: Float> {
    pos: vect3<T>,
    frines: std::vec::Vec<Frine<T>>,
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
struct PosePair<T: Float> {
    #[arael(ref = root.poses)]
    prev: Ref<Pose<T>>,
    #[arael(ref = root.poses)]
    cur: Ref<Pose<T>>,
    delta_pos: vect3<T>,
    delta_ea: vect3<T>,
    pos_cov_r: matrix3<T>,
    pos_cov_isigma: vect3<T>,
    ea_cov_r: matrix3<T>,
    ea_cov_isigma: vect3<T>,
    hb: CrossBlock<Pose<T>, Pose<T>, T>,
}

#[arael::model]
#[arael(root, fast_atan)]
#[derive(Clone)]
pub struct Path {
    poses: refs::Vec<Pose<f64>>,
    landmarks: refs::Vec<PointLandmark<f64>>,
    pose_pairs: std::vec::Vec<PosePair<f64>>,
    drift_pos_isigma: f64,
    drift_ea_isigma: f64,
    tilt_isigma: f64,
    frine_isigma_scale: f64,
}

#[arael::model]
#[arael(root, f32, fast_atan)]
#[derive(Clone)]
pub struct PathF {
    poses: refs::Vec<Pose<f32>>,
    landmarks: refs::Vec<PointLandmark<f32>>,
    pose_pairs: std::vec::Vec<PosePair<f32>>,
    drift_pos_isigma: f32,
    drift_ea_isigma: f32,
    tilt_isigma: f32,
    frine_isigma_scale: f32,
}

// ---------------------------------------------------------------- build/extract

/// The entity collections at any precision (`cast` is the identity at
/// f32, the exact widening at f64).
fn build_parts<T: Float>(scene: &Scene)
    -> (refs::Vec<Pose<T>>, refs::Vec<PointLandmark<T>>, std::vec::Vec<PosePair<T>>)
{
    let c = |x: f32| T::from(x).unwrap();
    let mut poses = refs::Vec::new();
    for p in &scene.poses {
        poses.push(Pose {
            pos: Param::new(p.init_pos.cast()),
            ea: SimpleEulerAngleParam::new(p.init_ea.cast()),
            prior_pos: p.init_pos.cast(),
            prior_ea: p.init_ea.cast(),
            tilt_roll: c(p.tilt_roll),
            tilt_pitch: c(p.tilt_pitch),
            hb_pose: SelfBlock::new(),
        });
    }
    let mut per_lm: Vec<Vec<Frine<T>>> = (0..scene.landmarks.len()).map(|_| Vec::new()).collect();
    for f in &scene.frines {
        let pose = poses.ref_at(f.pose);
        per_lm[f.landmark as usize].push(Frine {
            pose,
            mf2r: f.mf2r.cast(),
            camera_pos: f.camera_pos.cast(),
            isigma: f.isigma.cast(),
        });
    }
    let mut landmarks = refs::Vec::new();
    for (i, lm) in scene.landmarks.iter().enumerate() {
        landmarks.push(PointLandmark {
            pos: lm.cast(),
            frines: std::mem::take(&mut per_lm[i]),
        });
    }
    let mut pose_pairs = std::vec::Vec::new();
    for o in &scene.odo {
        pose_pairs.push(PosePair {
            prev: poses.ref_at(o.prev),
            cur: poses.ref_at(o.cur),
            delta_pos: o.delta_pos.cast(),
            delta_ea: o.delta_ea.cast(),
            pos_cov_r: o.pos_cov_r.cast(),
            pos_cov_isigma: o.pos_cov_isigma.cast(),
            ea_cov_r: o.ea_cov_r.cast(),
            ea_cov_isigma: o.ea_cov_isigma.cast(),
            hb: CrossBlock::new(),
        });
    }
    (poses, landmarks, pose_pairs)
}

fn build(scene: &Scene) -> Path {
    let (poses, landmarks, pose_pairs) = build_parts(scene);
    Path {
        poses, landmarks, pose_pairs,
        drift_pos_isigma: scene.drift_pos_isigma as f64,
        drift_ea_isigma: scene.drift_ea_isigma as f64,
        tilt_isigma: scene.tilt_isigma as f64,
        frine_isigma_scale: scene.frine_isigma_scale as f64,
    }
}

fn build_f32(scene: &Scene) -> PathF {
    let (poses, landmarks, pose_pairs) = build_parts(scene);
    PathF {
        poses, landmarks, pose_pairs,
        drift_pos_isigma: scene.drift_pos_isigma,
        drift_ea_isigma: scene.drift_ea_isigma,
        tilt_isigma: scene.tilt_isigma,
        frine_isigma_scale: scene.frine_isigma_scale,
    }
}

fn extract_parts<T: Float>(poses: &refs::Vec<Pose<T>>) -> Solution {
    Solution {
        poses: poses.iter()
            .map(|p| (p.pos.value.cast(),
                matrix3::rotation_from_euler_angles(p.ea.value).get_euler_angles().cast()))
            .collect(),
    }
}

fn extract(path: &Path) -> Solution {
    extract_parts(&path.poses)
}

fn extract_f32(path: &PathF) -> Solution {
    extract_parts(&path.poses)
}

// ---------------------------------------------------------------- solve

// Localization is block-tridiagonal: poses couple only through consecutive
// odometry, and the bearings hit FIXED landmarks so they add no pose-to-pose
// coupling. The default solver is therefore arael's band Cholesky with
// kd = 2*6 - 1 = 11 (6-DOF poses laid out consecutively), matching loc_demo.
// LOC_ARAEL_SOLVER=faer overrides with the general sparse solver.
const BAND_KD: usize = 11;

// LOC_ARAEL_SOLVER selects the backend: band (default -- scalar LAPACK-band
// Cholesky, kd=11), faer (general sparse), or narrow_band (block band Cholesky
// on the whole banded Hessian, via SparseFaer::with_narrow_band).
fn solver_kind() -> String {
    std::env::var("LOC_ARAEL_SOLVER").unwrap_or_else(|_| "band".to_string())
}

/// The backend this run resolved to, for the config header.
pub fn backend() -> String {
    match solver_kind().as_str() {
        "faer" => "faer".to_string(),
        "narrow_band" => "narrow_band (block, whole system)".to_string(),
        _ => format!("band kd={}", BAND_KD),
    }
}

fn solve64(params: &[f64], path: &mut Path, cfg: &arael::simple_lm::LmConfig<f64>)
    -> arael::simple_lm::LmResult<f64> {
    match solver_kind().as_str() {
        "faer" => arael::simple_lm::solve_sparse(params, path, cfg),
        "narrow_band" => arael::simple_lm::lm_solve(
            params, &mut arael::simple_lm::SparseFaer::new().with_narrow_band(true), path, cfg),
        _ => arael::simple_lm::solve_band(params, BAND_KD, path, cfg),
    }
}

fn solve32(params: &[f32], path: &mut PathF, cfg: &arael::simple_lm::LmConfig<f32>)
    -> arael::simple_lm::LmResult<f32> {
    match solver_kind().as_str() {
        "faer" => arael::simple_lm::solve_sparse_f32(params, path, cfg),
        "narrow_band" => arael::simple_lm::lm_solve(
            params, &mut arael::simple_lm::SparseFaerF32::new().with_narrow_band(true), path, cfg),
        _ => arael::simple_lm::solve_band_f32(params, BAND_KD, path, cfg),
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
        let m = cov.marginal_cov(&path.poses[last]).unwrap();
        (0..6).map(|k| m[(k, k)].sqrt()).collect()
    };

    // The localization query: just the last pose. TriDiagonal gets it from the
    // forward Schur pass alone (no backward recursion), so it is the cheapest cell.
    let tridiag_last = median_ms(budget, cap, || {
        let cov = path.assemble_covariance(CovMode::TriDiagonal).unwrap();
        black_box(cov.marginal_cov(&path.poses[last]).unwrap());
    });
    let perquery_last = median_ms(budget, cap, || {
        let cov = path.assemble_covariance(CovMode::PerQuery).unwrap();
        black_box(cov.marginal_cov(&path.poses[last]).unwrap());
    });

    let tridiag_pose = scale_counts(query_counts(np, true), cap_s, |n| {
        let idx = spread(0, np, n);
        median_ms(budget, cap, || {
            let cov = path.assemble_covariance(CovMode::TriDiagonal).unwrap();
            for &i in &idx {
                black_box(cov.marginal_cov(&path.poses[i]).unwrap());
            }
        })
    });
    let perquery_pose = scale_counts(query_counts(np, true), cap_s, |n| {
        let idx = spread(0, np, n);
        median_ms(budget, cap, || {
            let cov = path.assemble_covariance(CovMode::PerQuery).unwrap();
            for &i in &idx {
                black_box(cov.marginal_cov(&path.poses[i]).unwrap());
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
    let result = <M as bench_harness::arael::Model>::solve(scene, &params, &mut model, &cfg);
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
    <M as bench_harness::arael::Model>::solve(scene, &params, &mut model, &cfg)
        .timing
        .expect("gather_timing is on")
}
