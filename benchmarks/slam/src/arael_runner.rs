// Dedicated arael model for the heterogeneous SLAM benchmark -- a fresh
// implementation, not the examples/slam_demo.rs model: drift regularizers
// use explicit stored priors (not `_value`), GPS is unguarded (every pose
// has it), and there is no plotting/graduation machinery. Six factor
// types, matching scene::reference_cost exactly. The entities are generic
// over the scalar; the two concrete roots (`Path` f64, `PathF` f32)
// instantiate one shared model. Bodies reach the root's globals through
// the `root` alias, which resolves under either root.

use arael::simple_lm::RootProblem;
use crate::scene::{Scene, Solution};
use arael::matrix::matrix3;
use arael::model::{CrossBlock, Param, SelfBlock, SimpleEulerAngleParam};
use arael::refs::{self, Ref};
use arael::utils::Float;
use arael::vect::{vect2, vect3};

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
    gps_pos: vect3<T>,
    gps_cov_r: matrix3<T>,
    gps_cov_isigma: vect3<T>,
    tilt_roll: T,
    tilt_pitch: T,
    hb_pose: SelfBlock<Pose<T>, T>,
}

#[arael::model]
// Landmark drift prior.
#[arael(constraint(hb_drift, {
    let d = pointlandmark.pos - pointlandmark.prior_pos;
    [d.x * root.drift_lm_isigma,
     d.y * root.drift_lm_isigma,
     d.z * root.drift_lm_isigma]
}))]
#[derive(Clone)]
struct PointLandmark<T: Float> {
    pos: Param<vect3<T>>,
    prior_pos: vect3<T>,
    frines: std::vec::Vec<Frine<T>>,
    hb_drift: SelfBlock<PointLandmark<T>, T>,
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
    hb: CrossBlock<PointLandmark<T>, Pose<T>, T>,
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
#[arael(root)]
#[derive(Clone)]
pub struct Path {
    poses: refs::Vec<Pose<f64>>,
    landmarks: refs::Arena<PointLandmark<f64>>,
    pose_pairs: std::vec::Vec<PosePair<f64>>,
    drift_pos_isigma: f64,
    drift_ea_isigma: f64,
    drift_lm_isigma: f64,
    tilt_isigma: f64,
    frine_isigma_scale: f64,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct PathF {
    poses: refs::Vec<Pose<f32>>,
    landmarks: refs::Arena<PointLandmark<f32>>,
    pose_pairs: std::vec::Vec<PosePair<f32>>,
    drift_pos_isigma: f32,
    drift_ea_isigma: f32,
    drift_lm_isigma: f32,
    tilt_isigma: f32,
    frine_isigma_scale: f32,
}

/// The entity collections at any precision (`cast` is the identity at
/// f32, the exact widening at f64).
fn build_parts<T: Float>(scene: &Scene)
    -> (refs::Vec<Pose<T>>, refs::Arena<PointLandmark<T>>, std::vec::Vec<PosePair<T>>)
{
    let c = |x: f32| T::from(x).unwrap();
    let mut poses = refs::Vec::new();
    for p in &scene.poses {
        let g = p.gps.as_ref().unwrap();
        poses.push(Pose {
            pos: Param::new(p.init_pos.cast()),
            ea: SimpleEulerAngleParam::new(p.init_ea.cast()),
            prior_pos: p.init_pos.cast(),
            prior_ea: p.init_ea.cast(),
            gps_pos: g.pos.cast(),
            gps_cov_r: g.cov_r.cast(),
            gps_cov_isigma: g.cov_isigma.cast(),
            tilt_roll: c(p.tilt_roll),
            tilt_pitch: c(p.tilt_pitch),
            hb_pose: SelfBlock::new(),
        });
    }
    // Frines are grouped by landmark.
    let mut per_lm: Vec<Vec<Frine<T>>> = (0..scene.landmarks_init.len())
        .map(|_| Vec::new()).collect();
    for f in &scene.frines {
        let pose = poses.ref_at(f.pose);
        per_lm[f.landmark as usize].push(Frine {
            pose,
            mf2r: f.mf2r.cast(),
            camera_pos: f.camera_pos.cast(),
            isigma: f.isigma.cast(),
            hb: CrossBlock::new(),
        });
    }
    let mut landmarks = refs::Arena::new();
    for (i, init) in scene.landmarks_init.iter().enumerate() {
        landmarks.push(PointLandmark {
            pos: Param::new(init.cast()),
            prior_pos: init.cast(),
            frines: std::mem::take(&mut per_lm[i]),
            hb_drift: SelfBlock::new(),
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

pub fn build(scene: &Scene) -> Path {
    let (poses, landmarks, pose_pairs) = build_parts(scene);
    Path {
        poses, landmarks, pose_pairs,
        drift_pos_isigma: scene.drift_pos_isigma as f64,
        drift_ea_isigma: scene.drift_ea_isigma as f64,
        drift_lm_isigma: scene.drift_lm_isigma as f64,
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
        drift_lm_isigma: scene.drift_lm_isigma,
        tilt_isigma: scene.tilt_isigma,
        frine_isigma_scale: scene.frine_isigma_scale,
    }
}

fn extract_parts<T: Float>(
    poses: &refs::Vec<Pose<T>>,
    landmarks: &refs::Arena<PointLandmark<T>>,
) -> Solution {
    Solution {
        poses: poses.iter()
            .map(|p| (p.pos.value.cast(),
                matrix3::rotation_from_euler_angles(p.ea.value).get_euler_angles().cast()))
            .collect(),
        landmarks: landmarks.iter().map(|l| l.pos.value.cast()).collect(),
    }
}

fn extract(path: &Path) -> Solution {
    extract_parts(&path.poses, &path.landmarks)
}

fn extract_f32(path: &PathF) -> Solution {
    extract_parts(&path.poses, &path.landmarks)
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
    // SLAM_WC=1: use the well_conditioned preset (low lambda + gradient stop)
    // instead of the hand-tuned config, to evaluate it as the default.
    if std::env::var("SLAM_WC").map_or(false, |v| v == "1") {
        let cfg = arael::simple_lm::LmConfig::well_conditioned()
            .with_max_iters(max_iters)
            .with_verbose(std::env::var("SLAM_VERBOSE").map_or(false, |v| v == "1"))
            .with_gather_timing(std::env::var("SLAM_TIMING").map_or(false, |v| v == "1"));
        return if nielsen() { cfg.with_nielsen() } else { cfg };
    }
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
// SLAM_NARROW_BAND=1 routes a banded reduced Schur system through the narrow-band
// Cholesky instead of faer's general sparse Cholesky (opt-in, off by default).
fn narrow_band_enabled() -> bool {
    std::env::var("SLAM_NARROW_BAND").map_or(false, |v| v == "1")
}

// ARAEL_BLOCK_SUPERNODAL=1 factors with the supernodal block Cholesky
// (SparseFaer::with_block_supernodal) instead of flattening to scalar CSC for
// faer. On the default schur route it only engages where the envelope
// declines. Cross-benchmark name, like ARAEL_LAMBDA_FLOOR; see
// docs/dev/BLOCK.md.
fn block_supernodal() -> bool {
    std::env::var("ARAEL_BLOCK_SUPERNODAL").as_deref() == Ok("1")
}

// ARAEL_BLOCK_SUPERNODAL_BATCH tunes the supernodal route's update batching:
// a ratio (e.g. 1.5), or 0/off to disable. Unset keeps the library default.
// A typo is rejected rather than silently ignored.
fn block_supernodal_batch() -> Option<f64> {
    match std::env::var("ARAEL_BLOCK_SUPERNODAL_BATCH").ok().as_deref() {
        None => arael::simple_lm::SparseFaerOptions::auto().block_supernodal_batch,
        Some("0") | Some("off") => None,
        Some(v) => Some(v.parse().unwrap_or_else(|_| {
            panic!("ARAEL_BLOCK_SUPERNODAL_BATCH={}: expected a ratio, or 0/off", v)
        })),
    }
}

// SLAM_ENVELOPE=auto|always|never picks how the reduced Schur system is
// factored. A typo here would silently benchmark the other route, so an
// unknown value is an error rather than a fallback.
//
// The default is `always`, not arael's `auto`: the published tables are the
// envelope route, and a benchmark should pin what it measures rather than let
// a gate re-decide it as the gate's threshold moves. `auto` is still selectable
// to measure the gate itself.
pub fn envelope_mode() -> arael::simple_lm::EnvelopeMode {
    use arael::simple_lm::EnvelopeMode;
    match std::env::var("SLAM_ENVELOPE").as_deref() {
        Err(_) | Ok("always") => EnvelopeMode::Always,
        Ok("auto") => EnvelopeMode::Auto,
        Ok("never") => EnvelopeMode::Never,
        Ok(other) => panic!("SLAM_ENVELOPE: expected auto, always or never, got {:?}", other),
    }
}

// SLAM_PANEL_WIDTH=N sets the envelope factorization's super-panel width,
// for sweeping that curve. Unset lets arael derive it.
fn envelope_panel_width() -> Option<usize> {
    std::env::var("SLAM_PANEL_WIDTH").ok().and_then(|v| v.parse().ok())
}

type Solved<T> = Result<arael::simple_lm::LmResult<T>, arael::simple_lm::SolveFailure<T>>;

fn solve64(params: &[f64], path: &mut Path, cfg: &arael::simple_lm::LmConfig<f64>)
    -> Solved<f64> {
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
                    .with_policy(arael::simple_lm::SchurPolicy::Never)
                    .with_block_supernodal(block_supernodal())
                    .with_block_supernodal_batching(block_supernodal_batch()),
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
                &mut arael::simple_lm::SparseFaer::new().with_narrow_band(narrow_band_enabled())
                    .with_envelope_schur(envelope_mode())
                    .with_envelope_panel_width(envelope_panel_width())
                    .with_block_supernodal(block_supernodal())
                    .with_block_supernodal_batching(block_supernodal_batch()),
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
    path.serialize(&mut params);
    path.calc_cost(&params)
}

/// Write the arael Hessian (J^T J) sparsity/fill pattern as a PNG at the initial
/// estimate -- one pixel per parameter, black where nonzero, mirrored to the
/// full symmetric matrix. Env-gated from main (SLAM_HESSIAN_BITMAP).
pub fn write_hessian_bitmap(scene: &Scene, out: &str) {
    use arael::simple_lm::LmProblem;
    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize(&mut params);
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
    if std::env::var("SLAM_WC").map_or(false, |v| v == "1") {
        let cfg = arael::simple_lm::LmConfig::well_conditioned()
            .with_max_iters(max_iters)
            .with_verbose(std::env::var("SLAM_VERBOSE").map_or(false, |v| v == "1"))
            .with_gather_timing(std::env::var("SLAM_TIMING").map_or(false, |v| v == "1"));
        return if nielsen() { cfg.with_nielsen() } else { cfg };
    }
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
    -> Solved<f32> {
    if std::env::var("SLAM_ARAEL_SOLVER").as_deref() == Ok("faer") {
        return arael::simple_lm::lm_solve(
            params,
            &mut arael::simple_lm::SparseFaerF32::new()
                .with_policy(arael::simple_lm::SchurPolicy::Never)
                .with_block_supernodal(block_supernodal())
                    .with_block_supernodal_batching(block_supernodal_batch()),
            path,
            cfg,
        );
    }
    arael::simple_lm::lm_solve(
        params, &mut arael::simple_lm::SparseFaerF32::new().with_narrow_band(narrow_band_enabled())
                    .with_envelope_schur(envelope_mode())
                    .with_envelope_panel_width(envelope_panel_width())
                    .with_block_supernodal(block_supernodal())
                    .with_block_supernodal_batching(block_supernodal_batch()), path, cfg)
}

// Capped single solve (no timing) -- used for peak-memory measurement.
pub fn run_capped(scene: &Scene, max_iters: usize) -> Solution {
    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize(&mut params);
    let result = solve64(&params, &mut path, &cfg(max_iters)).expect("capped solve failed");
    path.deserialize(&result.x);
    extract(&path)
}

pub fn run_f32_capped(scene: &Scene, max_iters: usize) -> Solution {
    let mut path = build_f32(scene);
    let mut params: Vec<f32> = Vec::new();
    path.serialize(&mut params);
    let result = solve32(&params, &mut path, &cfg32(max_iters, scene.poses.len()))
        .expect("capped solve failed");
    path.deserialize(&result.x);
    extract_f32(&path)
}


// ---------------------------------------------------------------- pipeline

/// Problem-appropriate initial damping, per scene size and precision. No single
/// value is clean at every size, so each size takes the one that stops cleanly
/// there: 1e-8 at 60 and 300 poses, 3e-7 at 120, both precisions -- except f32
/// at 60, which takes 1e-7. At 1e-8 the f32 solution lands a hair above the 1e-5
/// stop threshold there and then bounces in the f32 noise floor (a
/// termination/precision interaction, not divergence); 1e-7 makes the last step
/// small enough to stop cleanly, and would grind at the large size.
///
/// ARAEL_LAMBDA0 overrides every size and both precisions with one value.
fn lambda0(poses: usize, single_precision: bool) -> f64 {
    if poses <= 60 {
        if single_precision { 1e-7 } else { 1e-8 }
    } else if poses <= 120 {
        3e-7
    } else {
        1e-8
    }
}

impl bench_harness::arael::Model for Path {
    type Scalar = f64;
    type Input = Scene;
    type Solution = Solution;
    fn lambda0(scene: &Scene) -> f64 { lambda0(scene.poses.len(), false) }
    fn build(scene: &Scene) -> Self { build(scene) }
    fn serialize(&mut self, out: &mut Vec<f64>) { arael::simple_lm::RootProblem::serialize(self, out); }
    fn deserialize(&mut self, x: &[f64]) { arael::simple_lm::RootProblem::deserialize(self, x); }
    fn solution(&self) -> Solution { extract(self) }
    fn solve(_: &Self::Input, params: &[f64], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f64>)
        -> Solved<f64> { solve64(params, m, cfg) }
    fn tune(cfg: &mut arael::simple_lm::LmConfig<f64>) {
        cfg.gradient_tolerance = std::env::var("SLAM_GTOL").ok().and_then(|v| v.parse().ok());
        cfg.predicted_reduction_tolerance = std::env::var("SLAM_PRED_TOL").ok().and_then(|v| v.parse().ok());
    }
}

impl bench_harness::arael::Model for PathF {
    type Scalar = f32;
    type Input = Scene;
    type Solution = Solution;
    fn lambda0(scene: &Scene) -> f64 { lambda0(scene.poses.len(), true) }
    fn build(scene: &Scene) -> Self { build_f32(scene) }
    fn serialize(&mut self, out: &mut Vec<f32>) { arael::simple_lm::RootProblem::serialize(self, out); }
    fn deserialize(&mut self, x: &[f32]) { arael::simple_lm::RootProblem::deserialize(self, x); }
    fn solution(&self) -> Solution { extract_f32(self) }
    fn solve(_: &Self::Input, params: &[f32], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f32>)
        -> Solved<f32> { solve32(params, m, cfg) }
    fn tune(cfg: &mut arael::simple_lm::LmConfig<f32>) {
        cfg.gradient_tolerance = std::env::var("SLAM_GTOL").ok().and_then(|v| v.parse().ok());
        cfg.predicted_reduction_tolerance = std::env::var("SLAM_PRED_TOL").ok().and_then(|v| v.parse().ok());
    }
}

/// `Err` is why the solve failed, for the table to show in place of the row.
pub type RunOut = Result<bench_harness::table::Row<Solution>, String>;

pub fn run(scene: &Scene) -> RunOut { bench_harness::arael::run::<Path>(scene) }
pub fn run_f32(scene: &Scene) -> RunOut { bench_harness::arael::run::<PathF>(scene) }

// ----------------------------------------------------------- covariance

use arael::covariance::{CovMode, Covariance};

/// One covariance-scaling run: `(N, median_ms, reps)` per query count, for poses
/// and landmarks, plus the AllMarginals bulk cost and a validation std dev.
pub struct CovScaling {
    pub n_poses: usize,
    pub n_landmarks: usize,
    pub perquery_pose: Vec<(usize, f64, usize)>,
    pub perquery_lm: Vec<(usize, f64, usize)>,
    pub allmarg_ms: f64,
    pub allmarg_reps: usize,
    pub mid_pose: usize,
    pub sd_mid_pose: Vec<f64>,
}

// Solve the slam scene (f64), then time covariance recovery as the query count
// scales, for poses (6-DOF) and landmarks. PerQuery times the full cold cost
// (assemble H + factor + query N marginals) each rep; AllMarginals is the bulk
// selected inverse over the whole factor (every pose and landmark at once).
pub fn cov_bench(scene: &Scene, budget_s: f64, cap: usize) -> CovScaling {
    use bench_harness::cov::{cell_cap_s, query_counts, scale_counts, spread};
    use bench_harness::probe::median_ms;
    use std::hint::black_box;
    use std::time::Duration;

    let mut path = build(scene);
    let mut params: Vec<f64> = Vec::new();
    path.serialize(&mut params);
    let result = solve64(&params, &mut path, &cfg(200)).expect("covariance solve failed");
    path.deserialize(&result.x);
    let (np, nl) = (path.poses.len(), path.landmarks.len());
    let budget = Duration::from_secs_f64(budget_s);
    let cap_s = cell_cap_s();

    // Validation: middle-pose std dev (a shared value-check anchor).
    let mid_pose = np / 2;
    let sd_mid_pose = path.assemble_covariance(CovMode::PerQuery).unwrap().std_dev(&path.poses[mid_pose]).unwrap();

    // PerQuery poses: 1, 2, 8, 32, all.
    let perquery_pose = scale_counts(query_counts(np, true), cap_s, |n| {
        let idx = spread(0, np, n);
        median_ms(budget, cap, || {
            let cov = path.assemble_covariance(CovMode::PerQuery).unwrap();
            for &i in &idx {
                black_box(cov.marginal_cov(&path.poses[i]).unwrap());
            }
        })
    });

    // PerQuery landmarks: 1, 2, 8, 32, all -- "all" via per-query usually hits the
    // cap (that is AllMarginals' job), which the table shows as `*`.
    // The arena has no positional indexing (a slot's position is an accident
    // of the hole layout), so query by the refs it hands out.
    let lm_refs: std::vec::Vec<_> = path.landmarks.refs().collect();
    let perquery_lm = scale_counts(query_counts(nl, true), cap_s, |n| {
        let idx = spread(0, nl, n);
        median_ms(budget, cap, || {
            let cov = path.assemble_covariance(CovMode::PerQuery).unwrap();
            for &i in &idx {
                black_box(cov.marginal_cov(&path.landmarks[lm_refs[i]]).unwrap());
            }
        })
    });

    // AllMarginals: bulk selected inverse -- every pose and landmark at once.
    let (allmarg_ms, allmarg_reps) = median_ms(budget, cap, || {
        black_box(path.assemble_covariance(CovMode::AllMarginals).unwrap());
    });

    CovScaling {
        n_poses: np,
        n_landmarks: nl,
        perquery_pose,
        perquery_lm,
        allmarg_ms,
        allmarg_reps,
        mid_pose,
        sd_mid_pose,
    }
}

#[cfg(test)]
mod tests {
    /// The published tables are measured at these values, and the README's
    /// Damping section names them. A run whose damping quietly differs from the
    /// documented one produces a table nobody can reproduce.
    #[test]
    fn documented_damping_per_scene_size() {
        for (poses, f64_lambda, f32_lambda) in
            [(60, 1e-8, 1e-7), (120, 3e-7, 3e-7), (300, 1e-8, 1e-8)]
        {
            assert_eq!(super::lambda0(poses, false), f64_lambda, "f64 at {} poses", poses);
            assert_eq!(super::lambda0(poses, true), f32_lambda, "f32 at {} poses", poses);
        }
    }
}
