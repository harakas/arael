//! Multi-run merging: several robot trajectories sharing one map.
//!
//! Three robots drive slightly different arcs through the SAME environment,
//! each with its own noisy odometry. Independently dead-reckoned, their tracks
//! drift apart into three inconsistent frames. But they see the SAME building
//! corners, and those shared landmarks tie the runs together: one joint
//! least-squares solve pulls all three trajectories into a single consistent
//! map. This is multi-session SLAM / map merging.
//!
//! The model is a NESTED tree -- the merge lives in the shape:
//!
//!     Map { paths: Vec<Path>, landmarks }        // root; landmarks are SHARED
//!     Path { poses, pose_pairs, frines }         // one run
//!
//! Each run owns its poses and its own odometry/bearing constraints, but the
//! landmarks live once on the Map. A bearing sighting (`Frine`) references a
//! pose in its OWN path (`parent.poses`) and a landmark in the shared map
//! (`root.landmarks`) -- so every run's observations accumulate onto the same
//! landmark, which is exactly what merges the runs.
//!
//! Run:
//!     cargo run -r --example slam2d_multi_demo

use arael::covariance::{CovMode, Covariance};
use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::simple_lm::LmProblem;
use arael::vect::vect2f;
use arael::matrix::matrix2f;
use arael::refs::{self, Ref};

use std::f32::consts::PI;

// Shared scene + measurements (also used by slam2d_align_demo) so both demos
// run on the identical problem.
#[path = "shared/scene2d.rs"]
mod scene2d;
use scene2d::Cfg;

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

// A 2D GPS fix: an absolute position in the world frame (true position + noise).
// This is what anchors the map -- NO pose is held fixed. `Option` because a fix
// may be missing (GPS dropout); the constraint is guarded on `is_some()`,
// exactly as the 3D slam_demo does with `info.gps`.
#[arael::model]
struct GpsData {
    pos: vect2f,
    isigma: f32,
}

// Robot pose: position, heading, movement measured since the previous pose, and
// an optional GPS fix. The GPS self-constraint (guarded) whitens
// `pose.pos - gps.pos`; with fixes on multiple poses the whole trajectory --
// position AND heading (through odometry / bearings) -- is observable without
// fixing any pose.
#[arael::model]
#[arael(constraint(hb_pose, guard = self.gps.is_some(), {
    let d = pose.pos - pose.gps.pos;
    [d.x * pose.gps.isigma, d.y * pose.gps.isigma]
}))]
struct Pose {
    pos: Param<vect2f>,
    gamma: Param<f32>,
    delta_pos: vect2f,
    delta_gamma: f32,
    delta_pos_isigma: f32,
    delta_gamma_isigma: f32,
    gps: Option<GpsData>,
    hb_pose: SelfBlock<Pose, f32>,
}

// Odometry constraint between two consecutive poses OF THE SAME RUN. Both refs
// resolve against the containing Path (`parent.poses`), so a run's chain never
// reaches into another run's poses.
#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(prev.gamma).transpose() * (cur.pos - prev.pos);
    [(local.x - cur.delta_pos.x) * cur.delta_pos_isigma,
     (local.y - cur.delta_pos.y) * cur.delta_pos_isigma,
     rad_diff(cur.gamma - prev.gamma, cur.delta_gamma) * cur.delta_gamma_isigma]
}))]
struct PosePair {
    #[arael(ref = parent.poses)]
    prev: Ref<Pose>,
    #[arael(ref = parent.poses)]
    cur: Ref<Pose>,
    hb: CrossBlock<Pose, Pose, f32>,
}

// A building corner. SHARED across all runs -- it lives on the Map, not on any
// one Path. No constraint of its own; it is pinned entirely by the bearing
// sightings that reference it from every run's poses.
#[arael::model]
struct Landmark {
    pos: Param<vect2f>,
    hb: SelfBlock<Landmark, f32>,
}

// One bearing sighting: pose `pose` (in THIS run) saw landmark `lm` (in the
// shared map) at angle `bearing`. This is the cross-level link -- `pose`
// resolves against the containing Path, `lm` against the root Map -- and it is
// the same math as slam2d_simple_demo's Frine, but with the landmark reached
// by reference instead of being the parent. Because sightings from every run
// point at the same `root.landmarks`, the joint solve merges the runs.
#[arael::model]
#[arael(constraint(hb, {
    let world_angle = pose.gamma + frine.bearing;
    let aligned = matrix2sym::rotation(world_angle).transpose() * (lm.pos - pose.pos);
    [atan2(aligned.y, aligned.x) * frine.isigma]
}))]
struct Frine {
    #[arael(ref = parent.poses)]
    pose: Ref<Pose>,
    #[arael(ref = root.landmarks)]
    lm: Ref<Landmark>,
    bearing: f32,
    isigma: f32,
    hb: CrossBlock<Landmark, Pose, f32>,
}

// One run: its poses, its odometry edges, and its bearing sightings. A
// block-less grouping sub-model -- no params, no SelfBlock of its own.
#[arael::model]
struct Path {
    poses: refs::Deque<Pose>,
    pose_pairs: std::vec::Vec<PosePair>,
    frines: std::vec::Vec<Frine>,
}

#[arael::model]
#[arael(root, f32)]
struct Map {
    paths: std::vec::Vec<Path>,
    landmarks: refs::Vec<Landmark>,
}

// ---------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------

// Cfg, ground truth, and measurements all come from the shared scene2d module.

struct Gt {
    poses: std::vec::Vec<std::vec::Vec<(vect2f, f32)>>, // per run
    lms: std::vec::Vec<vect2f>,                          // shared GT landmarks
    lm_to_gt: std::vec::Vec<usize>,                      // map landmark idx -> GT idx
}

// Assemble the JOINT Map from the shared scene. Byte-identical to the previous
// inline generator (scene2d reproduces its exact RNG order).
fn build_map(cfg: &Cfg) -> (Map, Gt) {
    let scene = scene2d::generate(cfg);

    // One Path per run from the shared per-run measurements (poses + odometry +
    // GPS). No pose is held fixed -- GPS anchors every run.
    let mut paths: std::vec::Vec<Path> = scene.runs.iter().map(|rm| {
        let mut path = Path {
            poses: refs::Deque::new(),
            pose_pairs: std::vec::Vec::new(),
            frines: std::vec::Vec::new(),
        };
        for i in 0..rm.est.len() {
            path.poses.push_back(Pose {
                pos: Param::new(rm.est[i].0),
                gamma: Param::new(rm.est[i].1),
                delta_pos: rm.delta[i].0,
                delta_gamma: rm.delta[i].1,
                delta_pos_isigma: if i == 0 { 0.0 } else { 1.0 / cfg.odo_pos_sigma },
                delta_gamma_isigma: if i == 0 { 0.0 } else { 1.0 / cfg.odo_gamma_sigma },
                gps: Some(GpsData { pos: rm.gps[i], isigma: scene.gps_isigma }),
                hb_pose: SelfBlock::new(),
            });
            if i > 0 {
                let prev = path.poses.ref_at(i - 1);
                let cur = path.poses.ref_at(i);
                path.pose_pairs.push(PosePair { prev, cur, hb: CrossBlock::new() });
            }
        }
        path
    }).collect();

    // Consensus landmarks: one per GT id seen by >= 2 sightings across all runs;
    // sightings are contiguous by gt_id in the shared scene.
    let mut map = Map { paths: std::vec::Vec::new(), landmarks: refs::Vec::new() };
    let mut lm_to_gt: std::vec::Vec<usize> = std::vec::Vec::new();
    let mut i = 0;
    while i < scene.sightings.len() {
        let gid = scene.sightings[i].gt_id;
        let start = i;
        while i < scene.sightings.len() && scene.sightings[i].gt_id == gid { i += 1; }
        let group = &scene.sightings[start..i];
        if group.len() < 2 { continue; }
        let s0 = &group[0];
        let (p0, g0) = scene.runs[s0.run].est[s0.pose];
        let world_b = g0 + s0.bearing;
        let init = p0 + vect2f::new(cfg.init_range * world_b.cos(), cfg.init_range * world_b.sin());
        let map_lm_idx = map.landmarks.len() as u32;
        map.landmarks.push(Landmark { pos: Param::new(init), hb: SelfBlock::new() });
        lm_to_gt.push(gid);
        for s in group {
            let pose_ref = paths[s.run].poses.ref_at(s.pose);
            paths[s.run].frines.push(Frine {
                pose: pose_ref,
                lm: Ref::new(map_lm_idx),
                bearing: s.bearing,
                isigma: 1.0 / cfg.bearing_sigma,
                hb: CrossBlock::new(),
            });
        }
    }
    for p in paths { map.paths.push(p); }

    (map, Gt { poses: scene.gt_poses, lms: scene.gt_lms, lm_to_gt })
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let cfg = Cfg::default();
    let (mut map, gt) = build_map(&cfg);

    let n_poses: usize = map.paths.iter().map(|p| p.poses.len()).sum();
    let n_frines: usize = map.paths.iter().map(|p| p.frines.len()).sum();
    println!("Map: {} runs, {} poses total, {} shared landmarks, {} sightings",
        map.paths.len(), n_poses, map.landmarks.len(), n_frines);
    let mut params: std::vec::Vec<f32> = std::vec::Vec::new();
    map.serialize32(&mut params);
    println!("Parameters: {} (Pose={}, Landmark={})\n",
        params.len(), Pose::PARAM_COUNT, Landmark::PARAM_COUNT);

    let lm_cfg = arael::simple_lm::LmConfig::well_conditioned().with_verbose(true);
    let result = map.solve_sparse(&lm_cfg);
    println!("\n{} iterations, cost {:.4} -> {:.4}",
        result.iterations, result.start_cost, result.end_cost);

    // Errors are reported RELATIVE, not absolute (as in slam_demo). A merged
    // bearing+GPS map can still carry a tiny global rotation, and a small
    // residual yaw swings distant landmarks by metres -- inflating ABSOLUTE
    // errors without reflecting structural quality. The gauge-invariant metrics
    // below cancel the shared gauge: consecutive pose deltas expressed in the
    // previous pose's local frame, and each landmark relative to its closest
    // pose in that pose's local frame.

    // Relative pose errors: consecutive deltas in the previous pose's frame.
    let mut dpos: std::vec::Vec<f32> = std::vec::Vec::new();
    let mut dyaw_deg: std::vec::Vec<f32> = std::vec::Vec::new();
    for (r, path) in map.paths.iter().enumerate() {
        let gp = &gt.poses[r];
        let opt: std::vec::Vec<&Pose> = path.poses.iter().collect();
        for i in 1..gp.len().min(opt.len()) {
            let gt_d = matrix2f::rotation(gp[i - 1].1).transpose() * (gp[i].0 - gp[i - 1].0);
            let o_d = matrix2f::rotation(opt[i - 1].gamma.value).transpose()
                * (opt[i].pos.value - opt[i - 1].pos.value);
            dpos.push((o_d - gt_d).norm());
            let gt_dg = wrap(gp[i].1 - gp[i - 1].1);
            let o_dg = wrap(opt[i].gamma.value - opt[i - 1].gamma.value);
            dyaw_deg.push(wrap(o_dg - gt_dg).abs().to_degrees());
        }
    }
    println!("\n-- Relative pose errors (consecutive deltas, gauge-free) --");
    print_stats("  delta pos", "m", &dpos);
    print_stats("  delta yaw", "deg", &dyaw_deg);

    // Landmark errors relative to the closest pose, in that pose's local frame.
    println!("\n-- Landmark errors (relative to closest pose, gauge-free) --");
    {
        // (GT pose, optimized pose) flattened across runs for closest lookup.
        let mut flat: std::vec::Vec<((vect2f, f32), &Pose)> = std::vec::Vec::new();
        for (r, path) in map.paths.iter().enumerate() {
            for (i, pose) in path.poses.iter().enumerate() {
                flat.push((gt.poses[r][i], pose));
            }
        }
        let mut lm_errs: std::vec::Vec<f32> = std::vec::Vec::new();
        for (i, lm) in map.landmarks.iter().enumerate() {
            let gt_lm = gt.lms[gt.lm_to_gt[i]];
            let (gt_pose, opt_pose) = flat.iter()
                .min_by(|a, b| (gt_lm - a.0.0).norm().total_cmp(&(gt_lm - b.0.0).norm()))
                .unwrap();
            let gt_vec = matrix2f::rotation(gt_pose.1).transpose() * (gt_lm - gt_pose.0);
            let opt_vec = matrix2f::rotation(opt_pose.gamma.value).transpose()
                * (lm.pos.value - opt_pose.pos.value);
            lm_errs.push((opt_vec - gt_vec).norm());
        }
        print_stats("  lm pos   ", "m", &lm_errs);
    }

    // 95% confidence ellipses from the per-landmark parameter covariance. GPS
    // pins the gauge, so the raw covariance block is meaningful.
    let ellipses = compute_landmark_ellipses(&mut map);

    let out = "slam2d_multi.eps";
    write_eps(&map, &gt, &ellipses, out).expect("eps write");
    println!("\nMerged map plotted to {}", out);
}

// Wrap an angle to (-pi, pi].
fn wrap(a: f32) -> f32 {
    a.sin().atan2(a.cos())
}

// Print mean / median / min / max of a sample.
fn print_stats(label: &str, unit: &str, v: &[f32]) {
    if v.is_empty() { return; }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    let mean: f32 = s.iter().sum::<f32>() / n as f32;
    println!("{}: mean={:.4}{u}  median={:.4}{u}  min={:.4}{u}  max={:.4}{u}",
        label, mean, s[n / 2], s[0], s[n - 1], u = unit);
}

// Per-landmark 95% confidence ellipse (center, semi_major, semi_minor,
// angle_rad) from the 2x2 diagonal block of the parameter covariance. 95% in 2D
// is chi^2(0.95, df=2) = 5.991, so the semi-axes are sqrt(5.991 * eigenvalue).
fn compute_landmark_ellipses(map: &mut Map) -> std::vec::Vec<(vect2f, f32, f32, f32)> {
    let cov = match map.assemble_covariance(CovMode::AllMarginals) {
        Ok(cov) => cov,
        Err(e) => {
            println!("\n{e} -- skipping uncertainty ellipses.");
            return std::vec::Vec::new();
        }
    };

    let chi2_95 = 5.991_f64;
    let mut out = std::vec::Vec::with_capacity(map.landmarks.len());
    for lm in map.landmarks.iter() {
        let cll = cov.marginal_cov(lm);
        let eig = nalgebra::SymmetricEigen::new(cll);
        let (i0, i1) = if eig.eigenvalues[0] >= eig.eigenvalues[1] { (0, 1) } else { (1, 0) };
        let semi_a = (eig.eigenvalues[i0].max(0.0) * chi2_95).sqrt() as f32;
        let semi_b = (eig.eigenvalues[i1].max(0.0) * chi2_95).sqrt() as f32;
        let angle = (eig.eigenvectors[(1, i0)] as f32).atan2(eig.eigenvectors[(0, i0)] as f32);
        out.push((lm.pos.value, semi_a, semi_b, angle));
    }
    out
}

// ---------------------------------------------------------------------------
// EPS plot
// ---------------------------------------------------------------------------

// One distinct hue per run (poses + chain), shared landmarks in red, ground
// truth in gray shadow.
fn run_color(r: usize, n: usize) -> (f32, f32, f32) {
    let h = if n == 0 { 0.0 } else { r as f32 / n as f32 };
    hsv_to_rgb(h, 0.85, 0.72)
}

fn write_eps(map: &Map, gt: &Gt, ellipses: &[(vect2f, f32, f32, f32)], filename: &str)
    -> std::io::Result<()> {
    use std::io::Write;

    // Bounding box from the shared ground truth: all runs' poses + the merged
    // (>=2-sighting) landmarks. slam2d_align_demo frames the identical set on the
    // same scene, so the two plots share a page transform and line up. Excluding
    // barely-seen edge landmarks (never drawn) keeps the frame tight.
    let mut pts: std::vec::Vec<vect2f> = std::vec::Vec::new();
    for gp in &gt.poses { for (p, _) in gp { pts.push(*p); } }
    for &gi in &gt.lm_to_gt { pts.push(gt.lms[gi]); }

    let xmin = pts.iter().map(|p| p.x).fold(f32::INFINITY, f32::min) - 3.0;
    let xmax = pts.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max) + 3.0;
    let ymin = pts.iter().map(|p| p.y).fold(f32::INFINITY, f32::min) - 3.0;
    let ymax = pts.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max) + 3.0;

    let page_w = 620.0_f32;
    let page_h = 460.0_f32;
    let pad = 18.0_f32;
    let s = ((page_w - 2.0 * pad) / (xmax - xmin)).min((page_h - 2.0 * pad) / (ymax - ymin));
    let dx = (page_w - s * (xmax - xmin)) * 0.5;
    let dy = (page_h - s * (ymax - ymin)) * 0.5;
    let to_pg = |p: vect2f| (dx + (p.x - xmin) * s, dy + (p.y - ymin) * s);

    let mut f = std::fs::File::create(filename)?;
    writeln!(f, "%!PS-Adobe-3.0 EPSF-3.0")?;
    writeln!(f, "%%BoundingBox: 0 0 {} {}", page_w as i32, page_h as i32)?;
    writeln!(f, "%%Creator: slam2d_multi_demo")?;
    writeln!(f, "%%EndComments")?;
    writeln!(f, "/tri {{ gsave 4 2 roll translate exch rotate \
        dup 0 moveto dup -0.55 mul 1 index 0.45 mul lineto \
        dup -0.55 mul exch -0.45 mul lineto closepath fill grestore }} def")?;
    writeln!(f, "/dot {{ newpath 0 360 arc fill }} def")?;

    let polyline = |f: &mut std::fs::File, pts: &[(f32, f32)]| -> std::io::Result<()> {
        write!(f, "newpath ")?;
        for (i, (x, y)) in pts.iter().enumerate() {
            if i == 0 { write!(f, "{:.2} {:.2} moveto ", x, y)?; }
            else { write!(f, "{:.2} {:.2} lineto ", x, y)?; }
        }
        writeln!(f, "stroke")
    };

    // Ground-truth trajectories (gray shadow, dashed).
    writeln!(f, "0.66 0.66 0.66 setrgbcolor 0.7 setlinewidth [3 2] 0 setdash")?;
    for gp in &gt.poses {
        let chain: std::vec::Vec<(f32, f32)> = gp.iter().map(|(p, _)| to_pg(*p)).collect();
        polyline(&mut f, &chain)?;
    }
    writeln!(f, "[] 0 setdash")?;

    // Bearing rays (thin, run-colored) toward the shared landmarks.
    let n_runs = map.paths.len();
    writeln!(f, "0.25 setlinewidth")?;
    for (r, path) in map.paths.iter().enumerate() {
        let (cr, cg, cb) = run_color(r, n_runs);
        writeln!(f, "{:.3} {:.3} {:.3} setrgbcolor", 0.5 + 0.5 * cr, 0.5 + 0.5 * cg, 0.5 + 0.5 * cb)?;
        for fr in &path.frines {
            let pose = &path.poses[fr.pose];
            let lm = &map.landmarks[fr.lm];
            let dist = (lm.pos.value - pose.pos.value).norm();
            let (px, py) = to_pg(pose.pos.value);
            let world_dir = pose.gamma.value + fr.bearing;
            let tip = pose.pos.value + vect2f::new(dist * 1.05 * world_dir.cos(),
                                                   dist * 1.05 * world_dir.sin());
            let (tx, ty) = to_pg(tip);
            writeln!(f, "newpath {:.2} {:.2} moveto {:.2} {:.2} lineto stroke", px, py, tx, ty)?;
        }
    }

    // Optimized trajectories, one hue per run: dashed chain + filled triangles.
    for (r, path) in map.paths.iter().enumerate() {
        let (cr, cg, cb) = run_color(r, n_runs);
        writeln!(f, "{:.3} {:.3} {:.3} setrgbcolor 1.1 setlinewidth [4 2] 0 setdash", cr, cg, cb)?;
        let chain: std::vec::Vec<(f32, f32)> = path.poses.iter().map(|p| to_pg(p.pos.value)).collect();
        polyline(&mut f, &chain)?;
        writeln!(f, "[] 0 setdash")?;
        for pose in path.poses.iter() {
            let (x, y) = to_pg(pose.pos.value);
            writeln!(f, "{:.2} {:.2} {:.2} 6.0 tri", x, y, pose.gamma.value.to_degrees())?;
        }
    }

    // GT landmark dots (gray) with error lines to the optimized shared landmark.
    writeln!(f, "0.55 0.55 0.55 setrgbcolor 0.5 setlinewidth")?;
    for (i, lm) in map.landmarks.iter().enumerate() {
        let (ox, oy) = to_pg(lm.pos.value);
        let (gx, gy) = to_pg(gt.lms[gt.lm_to_gt[i]]);
        writeln!(f, "newpath {:.2} {:.2} moveto {:.2} {:.2} lineto stroke", ox, oy, gx, gy)?;
    }
    for &gi in &gt.lm_to_gt {
        let (x, y) = to_pg(gt.lms[gi]);
        writeln!(f, "{:.2} {:.2} 2.0 dot", x, y)?;
    }

    // 95% confidence ellipses per shared landmark (dark red outline).
    writeln!(f, "0.80 0.12 0.12 setrgbcolor 0.5 setlinewidth")?;
    for &(c, a, b, t) in ellipses {
        if a <= 0.0 || b <= 0.0 { continue; }
        let segs = 48;
        let (ct, st) = (t.cos(), t.sin());
        write!(f, "newpath ")?;
        for j in 0..=segs {
            let phi = 2.0 * PI * (j as f32) / (segs as f32);
            let (lx, ly) = (a * phi.cos(), b * phi.sin());
            let world = vect2f::new(c.x + ct * lx - st * ly, c.y + st * lx + ct * ly);
            let (px, py) = to_pg(world);
            if j == 0 { write!(f, "{:.2} {:.2} moveto ", px, py)?; }
            else { write!(f, "{:.2} {:.2} lineto ", px, py)?; }
        }
        writeln!(f, "closepath stroke")?;
    }

    // Shared landmarks (the merge points), dark red dots.
    writeln!(f, "0.80 0.12 0.12 setrgbcolor")?;
    for lm in map.landmarks.iter() {
        let (x, y) = to_pg(lm.pos.value);
        writeln!(f, "{:.2} {:.2} 3.0 dot", x, y)?;
    }

    writeln!(f, "%%EOF")?;
    Ok(())
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
