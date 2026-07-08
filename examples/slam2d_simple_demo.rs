//! Minimal 2D SLAM demo for teaching: pose = (x, y, gamma).
//!
//! The problem: a robot drives in an arc. Its odometry reports how far it
//! moved between poses, but with noise that accumulates into drift. A
//! camera reports the direction (bearing) to building corners -- never the
//! distance. Neither the robot's path nor the corner positions are known.
//! SLAM (simultaneous localization and mapping) recovers both at once by
//! finding the poses and corner positions that best agree with all the
//! measurements together, in the least-squares sense.
//!
//! - World axes: x = forward (east), y = left (north), gamma = yaw.
//! - Sensor: single forward-facing camera reporting bearing (angle from the
//!   robot's forward heading) to building corners. One scalar per sighting.
//! - Odometry: 3-DOF delta (dx_local, dy_local, dgamma), diagonal covariance.
//! - All measurements are relative, so the map as a whole could slide or
//!   rotate freely. The first pose is held fixed at (0, 0, 0) -- facing
//!   east at the origin -- via `optimize = false` on its params, giving
//!   every other pose and landmark a fixed reference to be measured
//!   against.
//!
//! Naming mirrors slam_demo.rs: Path (root), Pose, PosePair (odometry edge),
//! Landmark (building corner), Frine (one landmark-to-pose bearing sighting).
//!
//! Run:
//!     cargo run -r --example slam2d_simple_demo

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::simple_lm::LmProblem;
use arael::vect::vect2f;
use arael::matrix::matrix2f;
use arael::refs::{self, Ref};

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::Normal;
use std::f32::consts::PI;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

// Each #[arael(constraint(...))] below computes one or more residuals -- the
// difference between what the model predicts and what a sensor measured. The
// same per-measurement term is a factor (GTSAM) or an edge (g2o).
//
// Solver bookkeeping declared in the structs below:
// - SelfBlock<T>: the entity's diagonal block in the Hessian matrix
//   calculated during solving.
// - CrossBlock<A, B>: the off-diagonal Hessian block coupling the two
//   entities that appear in the same constraint.
// - Ref<T>: a typed index into a collection on the root struct; which
//   collection is named by the #[arael(ref = root.poses)] attribute.

// Robot pose. Also carries the movement measured since the previous pose,
// in that pose's local frame (dx forward, dy left, dgamma = change of heading).
// The first pose has no previous -- its delta fields are unused and its
// params are held fixed (optimize = false, see build_path). The SelfBlock
// has no self-constraint; the macro still wires its parameter indices
// because Pose participates in the PosePair and Frine CrossBlocks.
#[arael::model]
struct Pose {
    pos: Param<vect2f>,             // solved for: position (x, y)
    // Heading angle: 0 = facing east (+x), counterclockwise positive.
    gamma: Param<f32>,              // solved for
    delta_pos: vect2f,              // measured: movement since the previous pose
    delta_gamma: f32,               // measured: change of heading since the previous pose
    // isigma = 1 / sigma, where sigma is the sensor's uncertainty (its
    // measurement standard deviation). Multiplying each residual by 1/sigma
    // makes an accurate sensor pull harder than a sloppy one, and the units
    // cancel so angles and meters can share one cost.
    delta_pos_isigma: f32,
    delta_gamma_isigma: f32,
    hb_pose: SelfBlock<Pose, f32>,
}

// Odometry constraint between two consecutive poses: their actual relative
// motion must match the movement measured on `cur` (delta_pos, delta_gamma).
// The heading term uses rad_diff so the residual wraps correctly across +-pi
// (plain subtraction would blow up when the two headings straddle the branch).
#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(prev.gamma).transpose() * (cur.pos - prev.pos);
    [(local.x - cur.delta_pos.x) * cur.delta_pos_isigma,
     (local.y - cur.delta_pos.y) * cur.delta_pos_isigma,
     rad_diff(cur.gamma - prev.gamma, cur.delta_gamma) * cur.delta_gamma_isigma]
}))]
struct PosePair {
    #[arael(ref = root.poses)]
    prev: Ref<Pose>,
    #[arael(ref = root.poses)]
    cur: Ref<Pose>,
    hb: CrossBlock<Pose, Pose, f32>,
}

// A building corner. The SelfBlock has no self-constraint; the macro still
// wires its parameter indices because Landmark participates in Frine's
// CrossBlock<Landmark, Pose>.
#[arael::model]
struct Landmark {
    pos: Param<vect2f>,
    frines: std::vec::Vec<Frine>,
    hb: SelfBlock<Landmark, f32>,
}

// A "frine" is this demo's (and slam_demo's) name for a single bearing
// sighting linking one landmark to one pose -- a factor (GTSAM) or edge (g2o).
//
// One bearing observation of `lm` from `pose`. The residual is the angle
// difference between the landmark's actual direction and the measured bearing
// (zero when they agree). Rotating (lm - pose) into the pose's local frame and
// then into the bearing-aligned frame collapses both rotations into one:
// R(gamma + bearing).transpose() (2D rotations commute and compose by
// addition), and atan2 reads off the leftover angle, wrapped to (-pi, pi].
#[arael::model]
#[arael(constraint(hb, parent = lm, {
    let world_angle = pose.gamma + frine.bearing;
    let aligned = matrix2sym::rotation(world_angle).transpose() * (lm.pos - pose.pos);
    [atan2(aligned.y, aligned.x) * frine.isigma]
}))]
struct Frine {
    #[arael(ref = root.poses)]
    pose: Ref<Pose>,
    bearing: f32,
    isigma: f32,
    hb: CrossBlock<Landmark, Pose, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct Path {
    poses: refs::Deque<Pose>,
    pose_pairs: std::vec::Vec<PosePair>,
    landmarks: refs::Arena<Landmark>,
}

// ---------------------------------------------------------------------------
// Synthetic data (the boring stuff, kept out of the model definition)
// ---------------------------------------------------------------------------

struct Cfg {
    n_poses: usize,
    n_landmarks: usize,
    seed: u64,
    step: f32,                 // forward distance per pose (m)
    turn: f32,                 // yaw change per pose (rad)
    fov_half: f32,             // half-FOV of the camera (rad)
    range_min: f32,
    range_max: f32,
    odo_pos_sigma: f32,        // odometry pos noise (m)
    odo_gamma_sigma: f32,      // odometry yaw noise (rad)
    bearing_sigma: f32,        // bearing noise (rad)
    init_range: f32,           // initial landmark distance guess (m)
}

impl Default for Cfg {
    fn default() -> Self {
        Cfg {
            n_poses: 20,
            n_landmarks: 30,
            seed: 42,
            step: 1.5,
            turn: 0.10,
            fov_half: 60.0_f32.to_radians(),
            range_min: 4.0,
            range_max: 50.0,
            odo_pos_sigma: 0.05,
            odo_gamma_sigma: 0.3_f32.to_radians(),
            bearing_sigma: 1.0_f32.to_radians(),
            init_range: 20.0,
        }
    }
}

// Gentle left-turning arc starting at the origin facing east.
fn truth_poses(cfg: &Cfg) -> std::vec::Vec<(vect2f, f32)> {
    let mut out = std::vec::Vec::with_capacity(cfg.n_poses);
    let mut pos = vect2f::new(0.0, 0.0);
    let mut gamma = 0.0_f32;
    out.push((pos, gamma));
    for _ in 1..cfg.n_poses {
        pos = pos + vect2f::new(cfg.step * gamma.cos(), cfg.step * gamma.sin());
        gamma += cfg.turn;
        out.push((pos, gamma));
    }
    out
}

// Corners scattered around the trajectory, at reasonable observation distance.
fn truth_landmarks(cfg: &Cfg, rng: &mut StdRng, poses: &[(vect2f, f32)]) -> std::vec::Vec<vect2f> {
    let mut out = std::vec::Vec::with_capacity(cfg.n_landmarks);
    while out.len() < cfg.n_landmarks {
        let anchor = poses[rng.random_range(0..poses.len())].0;
        let theta = rng.random::<f32>() * 2.0 * PI;
        let r = cfg.range_min + rng.random::<f32>() * (cfg.range_max - cfg.range_min);
        let lm = anchor + vect2f::new(r * theta.cos(), r * theta.sin());
        let visible = poses.iter().any(|(p, _)| {
            let d = (lm - *p).norm();
            d >= cfg.range_min && d <= cfg.range_max
        });
        if visible { out.push(lm); }
    }
    out
}

// Bearing from `(pos, gamma)` to `lm`, plus FOV / range gating.
fn observe(cfg: &Cfg, pos: vect2f, gamma: f32, lm: vect2f) -> Option<f32> {
    let d = lm - pos;
    let dist = d.norm();
    if dist < cfg.range_min || dist > cfg.range_max { return None; }
    let local = matrix2f::rotation(gamma).transpose() * d;
    let bearing = local.y.atan2(local.x);
    if bearing.abs() > cfg.fov_half { return None; }
    Some(bearing)
}

/// Returns the path, ground-truth poses, ground-truth landmarks, and a map
/// from optimized landmark index -> ground-truth landmark index (some GT
/// landmarks get filtered out when they have fewer than 2 sightings).
fn build_path(cfg: &Cfg) -> (
    Path,
    std::vec::Vec<(vect2f, f32)>,
    std::vec::Vec<vect2f>,
    std::vec::Vec<usize>,
) {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let nd = Normal::new(0.0_f32, 1.0).unwrap();

    let gt_poses = truth_poses(cfg);
    let gt_lms = truth_landmarks(cfg, &mut rng, &gt_poses);

    let mut path = Path {
        poses: refs::Deque::new(),
        pose_pairs: std::vec::Vec::new(),
        landmarks: refs::Arena::new(),
    };

    // Initial pose estimates come from dead-reckoning noisy odometry, starting
    // at (0,0,0). No GPS, no other absolute reference.
    let mut est_pos = vect2f::new(0.0, 0.0);
    let mut est_gamma = 0.0_f32;

    for (pi, &(gt_p, gt_g)) in gt_poses.iter().enumerate() {
        if pi == 0 {
            let mut first = Pose {
                pos: Param::new(vect2f::new(0.0, 0.0)),
                gamma: Param::new(0.0),
                delta_pos: vect2f::new(0.0, 0.0),
                delta_gamma: 0.0,
                delta_pos_isigma: 0.0,
                delta_gamma_isigma: 0.0,
                hb_pose: SelfBlock::new(),
            };
            // Every measurement is relative (bearings, odometry deltas), so
            // sliding or rotating the whole map changes no residual and the
            // solver would have no unique solution. Hold the first pose at
            // (0, 0, 0) as the fixed reference everything else is expressed
            // against: optimize = false keeps its params out of the
            // parameter vector, so the solver treats them as constants.
            first.pos.optimize = false;
            first.gamma.optimize = false;
            path.poses.push_back(first);
            continue;
        }

        let (prev_p, prev_g) = gt_poses[pi - 1];
        let true_delta = matrix2f::rotation(prev_g).transpose() * (gt_p - prev_p);
        let true_dg = gt_g - prev_g;

        let noisy_delta = true_delta + vect2f::new(
            cfg.odo_pos_sigma * rng.sample(nd),
            cfg.odo_pos_sigma * rng.sample(nd),
        );
        let noisy_dg = true_dg + cfg.odo_gamma_sigma * rng.sample(nd);

        // Dead-reckon the initial estimate: add up the noisy deltas. Drift
        // grows with every step -- this is what the optimizer will undo.
        est_pos = est_pos + matrix2f::rotation(est_gamma) * noisy_delta;
        est_gamma += noisy_dg;

        path.poses.push_back(Pose {
            pos: Param::new(est_pos),
            gamma: Param::new(est_gamma),
            delta_pos: noisy_delta,
            delta_gamma: noisy_dg,
            delta_pos_isigma: 1.0 / cfg.odo_pos_sigma,
            delta_gamma_isigma: 1.0 / cfg.odo_gamma_sigma,
            hb_pose: SelfBlock::new(),
        });
        let prev = path.poses.ref_at(pi - 1);
        let cur = path.poses.ref_at(pi);
        path.pose_pairs.push(PosePair { prev, cur, hb: CrossBlock::new() });
    }

    let mut lm_to_gt: std::vec::Vec<usize> = std::vec::Vec::new();
    for (gt_li, &gt_lm) in gt_lms.iter().enumerate() {
        let mut frines: std::vec::Vec<Frine> = std::vec::Vec::new();
        let mut first: Option<(usize, f32)> = None;
        for (pi, &(gt_p, gt_g)) in gt_poses.iter().enumerate() {
            let Some(true_b) = observe(cfg, gt_p, gt_g, gt_lm) else { continue; };
            let bearing = true_b + cfg.bearing_sigma * rng.sample(nd);
            if first.is_none() { first = Some((pi, bearing)); }
            frines.push(Frine {
                pose: path.poses.ref_at(pi),
                bearing,
                isigma: 1.0 / cfg.bearing_sigma,
                hb: CrossBlock::new(),
            });
        }
        if frines.len() < 2 { continue; }   // need two rays to triangulate
        let (first_pi, first_b) = first.unwrap();

        // Initialize the landmark by projecting the first observation ray to a
        // fixed distance from the *estimated* observing pose. The optimizer
        // does the actual triangulation.
        let p0 = &path.poses[first_pi];
        let world_b = p0.gamma.value + first_b;
        let init = p0.pos.value + vect2f::new(
            cfg.init_range * world_b.cos(),
            cfg.init_range * world_b.sin(),
        );
        path.landmarks.push(Landmark {
            pos: Param::new(init),
            frines,
            hb: SelfBlock::new(),
        });
        lm_to_gt.push(gt_li);
    }

    (path, gt_poses, gt_lms, lm_to_gt)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cfg = Cfg::default();
    let (mut path, gt_poses, gt_lms, lm_to_gt) = build_path(&cfg);

    let n_frines: usize = path.landmarks.iter().map(|l| l.frines.len()).sum();
    println!("Path: {} poses, {} pose_pairs, {} landmarks, {} frines",
        path.poses.len(), path.pose_pairs.len(), path.landmarks.len(), n_frines);
    // Report the flattened parameter count (serialize32 collects every
    // optimize = true param into one vector).
    let mut params: std::vec::Vec<f32> = std::vec::Vec::new();
    path.serialize32(&mut params);
    println!("Parameters: {} (Pose={}, Landmark={})\n",
        params.len(), Pose::PARAM_COUNT, Landmark::PARAM_COUNT);

    // verbose prints one line per Levenberg-Marquardt iteration: cost
    // before -> after / the improvement, and the damping lambda. Small
    // lambda = confident near-Gauss-Newton steps; lambda grows when a step
    // is rejected. With this seed you can see the cost spike and lambda
    // climb mid-run before convergence resumes.
    let lm_cfg = arael::simple_lm::LmConfig::<f32> { verbose: true, ..Default::default() };
    // One call runs the whole Levenberg-Marquardt solve (indexed sparse
    // faer backend): it flattens the params, repeatedly linearizes the
    // constraints and takes damped steps, then writes the optimized values
    // straight back into the pose/landmark structs.
    let result = path.solve_sparse(&lm_cfg);
    println!("\n{} iterations, cost {:.4} -> {:.4}",
        result.iterations, result.start_cost, result.end_cost);

    // Per-pose errors. The first pose is held at the ground-truth origin, so
    // comparing absolute positions against ground truth is meaningful.
    println!("\n-- Pose errors vs GT --");
    let mut pos_sum = 0.0_f32;
    let mut ea_sum = 0.0_f32;
    for (i, (pose, &(gt_p, gt_g))) in path.poses.iter().zip(gt_poses.iter()).enumerate() {
        let pe = (pose.pos.value - gt_p).norm();
        let ge = (pose.gamma.value - gt_g).abs();
        println!("  pose {:2}: |dp|={:.3}m  |dgamma|={:.3}deg  pos=({:7.3},{:7.3}) gamma={:7.4}",
            i, pe, ge.to_degrees(),
            pose.pos.value.x, pose.pos.value.y, pose.gamma.value);
        pos_sum += pe;
        ea_sum += ge;
    }
    let np = gt_poses.len() as f32;
    println!("  mean: pos={:.4}m  gamma={:.3}deg",
        pos_sum / np, (ea_sum / np).to_degrees());

    println!("\n-- Landmark errors vs GT --");
    let mut lm_sum = 0.0_f32;
    let n_l = path.landmarks.len();
    for (i, lm) in path.landmarks.iter().enumerate() {
        let gt = gt_lms[lm_to_gt[i]];
        let e = (lm.pos.value - gt).norm();
        println!("  lm {:2}: |d|={:.3}m  pos=({:7.3},{:7.3})  frines={}",
            i, e, lm.pos.value.x, lm.pos.value.y, lm.frines.len());
        lm_sum += e;
    }
    if n_l > 0 {
        println!("  mean: |d|={:.4}m", lm_sum / n_l as f32);
    }

    // Parameter covariance via inverse of the Gauss-Newton Hessian.
    //   H_ours = 2 * J^T J  (the factor of 2 comes from add_residual).
    //   Cov = (J^T J)^{-1} = 2 * H^{-1}.
    // The first pose is held fixed, so uncertainties are relative to a known
    // reference and the covariance is meaningful (with nothing held fixed the
    // whole map could slide or rotate freely and H would not be invertible);
    // each landmark's 2x2 diagonal block is its own positional uncertainty.
    let ellipses = compute_landmark_ellipses(&mut path);

    let out = "slam2d_simple.eps";
    write_eps(&path, &gt_poses, &gt_lms, &lm_to_gt, &ellipses, out).expect("eps write");
    println!("\nMap plotted to {}", out);
}

// Per-landmark 95% confidence ellipse: (center, semi_major, semi_minor, angle_rad).
// 95% in 2D corresponds to chi^2(0.95, df=2) = 5.991, so the semi-axes are
// sqrt(5.991 * eigenvalue) of the 2x2 position covariance block.
fn compute_landmark_ellipses(path: &mut Path) -> std::vec::Vec<(vect2f, f32, f32, f32)> {
    let mut params: std::vec::Vec<f32> = std::vec::Vec::new();
    path.serialize32(&mut params);
    let n = params.len();
    let mut grad = vec![0.0_f32; n];
    let mut hessian = vec![0.0_f32; n * n];
    path.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);

    // f32 -> f64 for stable Cholesky on the full system.
    let h64: std::vec::Vec<f64> = hessian.iter().map(|&x| x as f64).collect();
    let h_mat = nalgebra::DMatrix::from_row_slice(n, n, &h64);
    let cov = match nalgebra::linalg::Cholesky::new(h_mat) {
        Some(chol) => chol.inverse() * 2.0,
        None => {
            println!("\nHessian not positive-definite -- skipping uncertainty.");
            return std::vec::Vec::new();
        }
    };

    let chi2_95 = 5.991_f64;
    let mut out = std::vec::Vec::with_capacity(path.landmarks.len());
    for lm in path.landmarks.iter() {
        let k = lm.pos.index() as usize;
        let cll = cov.fixed_view::<2, 2>(k, k).clone_owned();
        let eig = nalgebra::SymmetricEigen::new(cll);
        // Sort descending so the major axis comes first.
        let (i0, i1) = if eig.eigenvalues[0] >= eig.eigenvalues[1] { (0, 1) } else { (1, 0) };
        let lam_major = eig.eigenvalues[i0].max(0.0);
        let lam_minor = eig.eigenvalues[i1].max(0.0);
        let semi_a = (lam_major * chi2_95).sqrt() as f32;
        let semi_b = (lam_minor * chi2_95).sqrt() as f32;
        let vx = eig.eigenvectors[(0, i0)];
        let vy = eig.eigenvectors[(1, i0)];
        let angle = (vy as f32).atan2(vx as f32);
        out.push((lm.pos.value, semi_a, semi_b, angle));
    }
    out
}

// ---------------------------------------------------------------------------
// EPS plot
// ---------------------------------------------------------------------------

// Raw PostScript -- no external deps. Layout:
//   * ground truth poses + landmarks drawn first in light gray (shadow);
//   * bearing rays from each optimized pose in the world-frame measurement
//     direction (pose.gamma + frine.bearing), capped at cfg.range_max;
//   * optimized poses chained by a dashed polyline;
//   * optimized poses as dark filled triangles pointing along gamma;
//   * optimized landmarks as red dots.
// y-up math coordinates map directly to PostScript's y-up page coordinates.
fn write_eps(
    path: &Path,
    gt_poses: &[(vect2f, f32)],
    gt_lms: &[vect2f],
    lm_to_gt: &[usize],
    ellipses: &[(vect2f, f32, f32, f32)],
    filename: &str,
) -> std::io::Result<()> {
    use std::io::Write;

    // Bounding box across everything we plan to draw.
    let mut pts: std::vec::Vec<vect2f> = std::vec::Vec::new();
    for pose in path.poses.iter() {
        pts.push(pose.pos.value);
    }
    for lm in path.landmarks.iter() {
        pts.push(lm.pos.value);
    }
    for (p, _) in gt_poses { pts.push(*p); }
    // Only show GT landmarks that ended up in the map (skip lone-sighting ones).
    for &gi in lm_to_gt { pts.push(gt_lms[gi]); }

    let xmin = pts.iter().map(|p| p.x).fold(f32::INFINITY, f32::min) - 3.0;
    let xmax = pts.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max) + 3.0;
    let ymin = pts.iter().map(|p| p.y).fold(f32::INFINITY, f32::min) - 3.0;
    let ymax = pts.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max) + 3.0;

    let page_w = 540.0_f32;   // ~7.5 in
    let page_h = 420.0_f32;   // ~5.8 in
    let pad = 18.0_f32;
    let s = ((page_w - 2.0 * pad) / (xmax - xmin))
        .min((page_h - 2.0 * pad) / (ymax - ymin));
    let dx = (page_w - s * (xmax - xmin)) * 0.5;
    let dy = (page_h - s * (ymax - ymin)) * 0.5;
    let to_pg = |p: vect2f| (dx + (p.x - xmin) * s, dy + (p.y - ymin) * s);

    let mut f = std::fs::File::create(filename)?;
    writeln!(f, "%!PS-Adobe-3.0 EPSF-3.0")?;
    writeln!(f, "%%BoundingBox: 0 0 {} {}", page_w as i32, page_h as i32)?;
    writeln!(f, "%%Creator: slam2d_simple_demo")?;
    writeln!(f, "%%EndComments")?;
    // Triangle marker -- call with `x y angle_deg size tri`. Forward tip at
    // (size, 0), back-left (-0.55 s, 0.45 s), back-right (-0.55 s, -0.45 s);
    // origin translates to (x, y) then frame rotates by angle.
    writeln!(f, "/tri {{ gsave 4 2 roll translate exch rotate \
        dup 0 moveto \
        dup -0.55 mul 1 index 0.45 mul lineto \
        dup -0.55 mul exch -0.45 mul lineto \
        closepath fill grestore }} def")?;
    // Filled circle -- call with `x y r dot`. arc consumes x y r angle1 angle2.
    writeln!(f, "/dot {{ newpath 0 360 arc fill }} def")?;

    let polyline = |f: &mut std::fs::File, pts: &[(f32, f32)]| -> std::io::Result<()> {
        write!(f, "newpath ")?;
        for (i, (x, y)) in pts.iter().enumerate() {
            if i == 0 { write!(f, "{:.2} {:.2} moveto ", x, y)?; }
            else      { write!(f, "{:.2} {:.2} lineto ", x, y)?; }
        }
        writeln!(f, "stroke")
    };

    // --- Ground-truth pose shadow (drawn first, behind everything) ---
    writeln!(f, "0.62 0.62 0.62 setrgbcolor 0.8 setlinewidth [3 2] 0 setdash")?;
    let gt_chain: std::vec::Vec<(f32, f32)> = gt_poses.iter().map(|(p, _)| to_pg(*p)).collect();
    polyline(&mut f, &gt_chain)?;
    writeln!(f, "[] 0 setdash")?;
    for (p, g) in gt_poses {
        let (x, y) = to_pg(*p);
        writeln!(f, "{:.2} {:.2} {:.2} 8 tri", x, y, g.to_degrees())?;
    }
    // GT landmark dots are drawn *after* the rays (below) so they don't get
    // buried under colored bearing bundles.

    // --- Bearing rays from each optimized pose, in world frame ---
    // Each landmark gets a distinct hue; its rays are drawn in a washed-out
    // tint of the same hue so overlapping ray bundles stay distinguishable.
    // Length = 110% of the pose->landmark distance so each ray reaches its
    // observed landmark with a small overshoot.
    writeln!(f, "0.25 setlinewidth")?;
    let n_lm = path.landmarks.len();
    for (li, lm) in path.landmarks.iter().enumerate() {
        let (r, g, b) = landmark_color(li, n_lm, true);
        writeln!(f, "{:.3} {:.3} {:.3} setrgbcolor", r, g, b)?;
        for fr in &lm.frines {
            let pose = &path.poses[fr.pose];
            let dist = (lm.pos.value - pose.pos.value).norm();
            let (px, py) = to_pg(pose.pos.value);
            let world_dir = pose.gamma.value + fr.bearing;
            let r = dist * 1.10;
            let tip = pose.pos.value
                + vect2f::new(r * world_dir.cos(), r * world_dir.sin());
            let (tx, ty) = to_pg(tip);
            writeln!(f, "newpath {:.2} {:.2} moveto {:.2} {:.2} lineto stroke",
                px, py, tx, ty)?;
        }
    }

    // --- Optimized pose chain (dashed) ---
    writeln!(f, "0.08 0.15 0.30 setrgbcolor 1.0 setlinewidth [4 2] 0 setdash")?;
    let opt_chain: std::vec::Vec<(f32, f32)> = path.poses.iter()
        .map(|pose| to_pg(pose.pos.value))
        .collect();
    polyline(&mut f, &opt_chain)?;

    // --- Optimized poses (filled triangles) ---
    writeln!(f, "[] 0 setdash 0.10 0.18 0.40 setrgbcolor")?;
    for pose in path.poses.iter() {
        let (x, y) = to_pg(pose.pos.value);
        writeln!(f, "{:.2} {:.2} {:.2} 6.5 tri",
            x, y, pose.gamma.value.to_degrees())?;
    }

    // --- 95% confidence ellipses per landmark (each in its own hue) ---
    writeln!(f, "0.6 setlinewidth")?;
    for (i, &(c, a, b, t)) in ellipses.iter().enumerate() {
        if a <= 0.0 || b <= 0.0 { continue; }
        let (r, g, bb) = landmark_color(i, n_lm, false);
        writeln!(f, "{:.3} {:.3} {:.3} setrgbcolor", r, g, bb)?;
        let segs = 48;
        let (ct, st) = (t.cos(), t.sin());
        write!(f, "newpath ")?;
        for j in 0..=segs {
            let phi = 2.0 * std::f32::consts::PI * (j as f32) / (segs as f32);
            let lx = a * phi.cos();
            let ly = b * phi.sin();
            let world = vect2f::new(c.x + ct * lx - st * ly,
                                    c.y + st * lx + ct * ly);
            let (px, py) = to_pg(world);
            if j == 0 { write!(f, "{:.2} {:.2} moveto ", px, py)?; }
            else      { write!(f, "{:.2} {:.2} lineto ", px, py)?; }
        }
        writeln!(f, "closepath stroke")?;
    }

    // --- Landmark error lines + GT landmark dots (above the ray bundles) ---
    writeln!(f, "0.55 0.55 0.55 setrgbcolor 0.5 setlinewidth")?;
    for (i, lm) in path.landmarks.iter().enumerate() {
        let opt = lm.pos.value;
        let gt = gt_lms[lm_to_gt[i]];
        let (ox, oy) = to_pg(opt);
        let (gx, gy) = to_pg(gt);
        writeln!(f, "newpath {:.2} {:.2} moveto {:.2} {:.2} lineto stroke",
            ox, oy, gx, gy)?;
    }
    for &gi in lm_to_gt {
        let (x, y) = to_pg(gt_lms[gi]);
        writeln!(f, "{:.2} {:.2} 2.2 dot", x, y)?;
    }

    // --- Optimized landmarks (one hue per landmark) ---
    for (i, lm) in path.landmarks.iter().enumerate() {
        let (x, y) = to_pg(lm.pos.value);
        let (r, g, b) = landmark_color(i, n_lm, false);
        writeln!(f, "{:.3} {:.3} {:.3} setrgbcolor {:.2} {:.2} 2.8 dot",
            r, g, b, x, y)?;
    }

    writeln!(f, "%%EOF")?;
    Ok(())
}

/// Per-landmark color. Evenly-spaced hues on an HSV wheel; rays use a
/// desaturated/lighter version of the same hue so a landmark and its
/// bearing fan visibly belong together without the rays overpowering the
/// dot.
fn landmark_color(i: usize, n: usize, ray: bool) -> (f32, f32, f32) {
    let h = if n == 0 { 0.0 } else { i as f32 / n as f32 };
    let (s, v) = if ray { (0.40, 0.97) } else { (0.85, 0.78) };
    hsv_to_rgb(h, s, v)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h6 = (h - h.floor()) * 6.0;
    let c = v * s;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    let m = v - c;
    (r + m, g + m, b + m)
}
