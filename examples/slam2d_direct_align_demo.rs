//! Building one map from several runs over the same area -- the cheapest of
//! the three ways. slam2d_multi_demo solves everything jointly;
//! slam2d_align_demo solves each run alone and then optimizes per-run
//! corrections TOGETHER with consensus landmarks. This demo drops the
//! landmarks from the optimization entirely: only the per-run corrections
//! are solved (9 parameters for three runs), and the landmarks are computed
//! afterwards in closed form.
//!
//! How it works
//! ------------
//! Stage 1 -- solve each run by itself, exactly as in slam2d_align_demo:
//! GPS anchors each run in world coordinates; keep a centre pose with its
//! uncertainty and, per landmark, its position relative to the centre plus
//! an uncertainty ellipse.
//!
//! Stage 2 -- align the paths only. For every pair of runs, every landmark
//! both saw says "my position per run A must meet my position per run B
//! once the corrections are applied". Each of those residuals couples the
//! same two correction sets, so all matches of a pair accumulate into ONE
//! shared Hessian block held by the pair (the `parent.hb` constraint form):
//! the problem stays 3 self blocks + 3 cross blocks no matter how many
//! landmarks matched. A landmark seen by k runs contributes k(k-1)/2
//! pairwise residuals of which only k-1 are independent; each is
//! down-weighted by sqrt(2/k) so the clique carries exactly the
//! independent amount of information (the derivation is at the weighting
//! code).
//!
//! A common rotation of the whole map is near-gauge (pinned only by the
//! centre priors; its 1-sigma is printed), so the plotted frame may sit a
//! fraction of a degree from slam2d_align_demo's -- the scoring
//! gauge-aligns before measuring, as both demos do.
//!
//! Stage 3 -- extract the landmarks by fusion. Matches connect landmark
//! sightings across runs; each connected component is one physical
//! landmark. Every sighting is mapped through its run's solved correction
//! and the component is fused in information form (inverse-covariance
//! weighting) -- the closed-form optimum of the landmark GIVEN the
//! alignment, i.e. exactly the position slam2d_align_demo's joint stage 2
//! would assign it with the corrections frozen. The difference between the
//! two demos is therefore precisely one thing: landmarks conditioned on
//! the alignment instead of optimized jointly with it.
//!
//! Run:  cargo run -r --example slam2d_direct_align_demo

use arael::covariance::{CovMode, Covariance};
use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::vect::{vect2f, vect3f};
use arael::matrix::{matrix2f, matrix3d, matrix3f};
use arael::refs::{self, Ref};

#[path = "shared/scene2d.rs"]
mod scene2d;
use scene2d::Cfg;

// ===========================================================================
// Stage 1 model: single-run, GPS-anchored bearing-only 2D SLAM
// (identical to slam2d_align_demo)
// ===========================================================================

#[arael::model]
struct GpsData { pos: vect2f, isigma: f32 }

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

#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(prev.gamma).transpose() * (cur.pos - prev.pos);
    [(local.x - cur.delta_pos.x) * cur.delta_pos_isigma,
     (local.y - cur.delta_pos.y) * cur.delta_pos_isigma,
     rad_diff(cur.gamma - prev.gamma, cur.delta_gamma) * cur.delta_gamma_isigma]
}))]
struct PosePair {
    #[arael(ref = root.poses)] prev: Ref<Pose>,
    #[arael(ref = root.poses)] cur: Ref<Pose>,
    hb: CrossBlock<Pose, Pose, f32>,
}

#[arael::model]
struct Landmark {
    pos: Param<vect2f>,
    frines: std::vec::Vec<Frine>,
    hb: SelfBlock<Landmark, f32>,
}

#[arael::model]
#[arael(constraint(hb, parent = landmark, {
    let world_angle = pose.gamma + frine.bearing;
    let aligned = matrix2sym::rotation(world_angle).transpose() * (landmark.pos - pose.pos);
    [atan2(aligned.y, aligned.x) * frine.isigma]
}))]
struct Frine {
    #[arael(ref = root.poses)] pose: Ref<Pose>,
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

// ===========================================================================
// Stage 2 model: rigid corrections only, matches on a shared pair block
// ===========================================================================

// One run's SE2 correction about its centre pose, with the centre pose's
// 3x3 (x, y, heading) covariance as a prior.
#[arael::model]
#[arael(constraint(hb, {
    let delta = vect3sym::from_components(pathinstance.translation.x, pathinstance.translation.y, pathinstance.rotation);
    let w = pathinstance.ccov_r.transpose() * delta;
    [w.x * pathinstance.ccov_isigma.x, w.y * pathinstance.ccov_isigma.y, w.z * pathinstance.ccov_isigma.z]
}))]
struct PathInstance {
    translation: Param<vect2f>,  // (dx, dy) shift correction, world frame
    rotation: Param<f32>,        // heading correction (radians)
    center: vect2f,       // centre pose position (world), the correction pivot
    ccov_r: matrix3f,     // eigenvectors of the centre pose 3x3 covariance
    ccov_isigma: vect3f,  // 1/sqrt(eigenvalues)
    landmarks: refs::Arena<RunLandmark>,  // this run's stage-1 map
    hb: SelfBlock<PathInstance, f32>,
}

// One landmark of a run's stage-1 map: mean relative to the run centre
// plus the stage-1 whitening. No params -- pure data, stored once per
// run and referenced by every match that uses it.
#[arael::model]
struct RunLandmark {
    mu: vect2f,          // rel. to the run's centre (world axes)
    cov_r: matrix2f,     // stage-1 conditional whitening
    cov_isigma: vect2f,
}

// One landmark seen by both runs of the pair: its corrected world position
// per run A must meet the one per run B. The entities come from the pair's
// refs (`parent = pair` names the binding); the landmark records through
// DATA refs into each run's own arena; the pair whitening (folding both
// runs' stage-1 ellipses) stays per match.
#[arael::model]
#[arael(constraint(parent.hb, parent = pair, {
    let qa = pair.a.center + pair.a.translation
        + matrix2sym::rotation(pair.a.rotation) * lm_a.mu;
    let qb = pair.b.center + pair.b.translation
        + matrix2sym::rotation(pair.b.rotation) * lm_b.mu;
    let w = pathmatch.cov_r.transpose() * (qa - qb);
    [w.x * pathmatch.cov_isigma.x, w.y * pathmatch.cov_isigma.y]
}))]
struct PathMatch {
    #[arael(ref = parent.a.landmarks)] lm_a: Ref<RunLandmark>,
    #[arael(ref = parent.b.landmarks)] lm_b: Ref<RunLandmark>,
    cov_r: matrix2f,     // eigenvectors of S_A' + S_B'
    cov_isigma: vect2f,  // 1/sqrt(eigenvalues)
}

// A pair of runs: every match accumulates into the one shared cross block.
#[arael::model]
struct PathPair {
    #[arael(ref = root.paths)] a: Ref<PathInstance>,
    #[arael(ref = root.paths)] b: Ref<PathInstance>,
    matches: std::vec::Vec<PathMatch>,
    hb: CrossBlock<PathInstance, PathInstance, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct AlignMap {
    paths: refs::Arena<PathInstance>,
    pairs: std::vec::Vec<PathPair>,
}

// ===========================================================================
// Covariance helpers (same as slam2d_align_demo)
// ===========================================================================

// Eigendecompose a covariance into whitening factors: the rotation into its
// eigenbasis and the per-axis inverse std-devs (1/sqrt of the eigenvalues).
fn whitening_factors2(c: matrix2f) -> (matrix2f, vect2f) {
    let (r, d) = c.symmetric_eigen();
    (r, vect2f::new(1.0 / d.x.max(1e-12).sqrt(), 1.0 / d.y.max(1e-12).sqrt()))
}

fn whitening_factors3(c: [[f64; 3]; 3]) -> (matrix3f, vect3f) {
    let (r, d) = matrix3d::from_array(c).symmetric_eigen();
    (r.cast::<f32>(), vect3f::new(1.0 / d.x.max(1e-12).sqrt() as f32,
                                  1.0 / d.y.max(1e-12).sqrt() as f32,
                                  1.0 / d.z.max(1e-12).sqrt() as f32))
}

// Reconstitute the 2x2 covariance from its whitening factors.
fn cov_from_whitening(r: matrix2f, isig: vect2f) -> nalgebra::Matrix2<f64> {
    let rr = nalgebra::Matrix2::new(r[0].x as f64, r[0].y as f64,
                                    r[1].x as f64, r[1].y as f64);
    let d = nalgebra::Matrix2::new(1.0 / (isig.x as f64).powi(2), 0.0,
                                   0.0, 1.0 / (isig.y as f64).powi(2));
    rr * d * rr.transpose()
}

// ===========================================================================
// Stage 1: solve one run (GPS-anchored), extract map + covariances
// (identical to slam2d_align_demo)
// ===========================================================================

struct RunMap {
    gt_ids: std::vec::Vec<usize>,
    mu: std::vec::Vec<vect2f>,         // landmark mean relative to centre (world)
    cov_r: std::vec::Vec<matrix2f>,
    cov_isigma: std::vec::Vec<vect2f>,
    center: vect2f,                    // centre pose position (world)
    ccov_r: matrix3f,                  // centre pose 3x3 covariance whitening
    ccov_isigma: vect3f,
    poses: std::vec::Vec<(vect2f, f32)>,
}

fn stage1(cfg: &Cfg, rm: &scene2d::RunMeas, gps_isigma: f32,
          scene_sightings: &[scene2d::Sighting], run: usize) -> RunMap {
    let mut path = Path {
        poses: refs::Deque::new(),
        pose_pairs: std::vec::Vec::new(),
        landmarks: refs::Arena::new(),
    };
    for i in 0..rm.est.len() {
        path.poses.push_back(Pose {
            pos: Param::new(rm.est[i].0), gamma: Param::new(rm.est[i].1),
            delta_pos: rm.delta[i].0, delta_gamma: rm.delta[i].1,
            delta_pos_isigma: if i == 0 { 0.0 } else { 1.0 / cfg.odo_pos_sigma },
            delta_gamma_isigma: if i == 0 { 0.0 } else { 1.0 / cfg.odo_gamma_sigma },
            gps: Some(GpsData { pos: rm.gps[i], isigma: gps_isigma }),
            hb_pose: SelfBlock::new(),
        });
        if i > 0 {
            path.pose_pairs.push(PosePair {
                prev: path.poses.ref_at(i - 1), cur: path.poses.ref_at(i), hb: CrossBlock::new() });
        }
    }
    // Bearings this run made, grouped by GT landmark; keep landmarks it saw >= 2.
    let mut by_gid: std::collections::BTreeMap<usize, std::vec::Vec<(usize, f32)>> = std::collections::BTreeMap::new();
    for s in scene_sightings.iter().filter(|s| s.run == run) {
        by_gid.entry(s.gt_id).or_default().push((s.pose, s.bearing));
    }
    let mut gt_ids = std::vec::Vec::new();
    for (&gid, obs) in &by_gid {
        if obs.len() < 2 { continue; }
        let (fp, fb) = obs[0];
        let p0 = &path.poses[fp];
        let wb = p0.gamma.value + fb;
        let init = p0.pos.value + vect2f::new(cfg.init_range * wb.cos(), cfg.init_range * wb.sin());
        let frines: std::vec::Vec<Frine> = obs.iter().map(|&(pi, b)| Frine {
            pose: path.poses.ref_at(pi), bearing: b, isigma: 1.0 / cfg.bearing_sigma, hb: CrossBlock::new() }).collect();
        path.landmarks.push(Landmark { pos: Param::new(init), frines, hb: SelfBlock::new() });
        gt_ids.push(gid);
    }

    path.solve_sparse(&LmConfig::well_conditioned()).unwrap();

    // Parameter covariance at the stage-1 solution.
    let cov = path.assemble_covariance(CovMode::PerQuery).expect("stage-1 Hessian not PD");

    // Centre = middle pose; its position is the correction pivot, its 3x3
    // (x, y, gamma) MARGINAL covariance is the frame prior.
    let mid = path.poses.len() / 2;
    let center = path.poses[mid].pos.value;
    let cm = cov.marginal_cov(&path.poses[mid]).unwrap();
    let mut c3 = [[0.0_f64; 3]; 3];
    for a in 0..3 { for b in 0..3 { c3[a][b] = cm[(a, b)]; } }
    let (ccov_r, ccov_isigma) = whitening_factors3(c3);

    let mut mu = std::vec::Vec::new();
    let mut cov_r = std::vec::Vec::new();
    let mut cov_isigma = std::vec::Vec::new();
    for lm in path.landmarks.iter() {
        // Landmark covariance with the poses held FIXED (conditional, not
        // marginal): the pose/frame uncertainty is carried by the centre prior.
        let c2 = cov.conditional_cov(lm).unwrap();
        let c = matrix2f::from_elements(c2[(0, 0)] as f32, c2[(0, 1)] as f32,
                                        c2[(1, 0)] as f32, c2[(1, 1)] as f32);
        let (r, isig) = whitening_factors2(c);
        mu.push(lm.pos.value - center);
        cov_r.push(r);
        cov_isigma.push(isig);
    }

    RunMap {
        gt_ids, mu, cov_r, cov_isigma, center, ccov_r, ccov_isigma,
        poses: path.poses.iter().map(|p| (p.pos.value, p.gamma.value)).collect(),
    }
}

// ===========================================================================
// Stage 3 helpers: union-find + information-form fusion
// ===========================================================================

struct UnionFind { parent: std::vec::Vec<usize> }
impl UnionFind {
    fn new(n: usize) -> Self { UnionFind { parent: (0..n).collect() } }
    fn find(&mut self, i: usize) -> usize {
        let mut r = i;
        while self.parent[r] != r { r = self.parent[r]; }
        let mut c = i;
        while self.parent[c] != r { let n = self.parent[c]; self.parent[c] = r; c = n; }
        r
    }
    fn union(&mut self, a: usize, b: usize) {
        let (ra, rb) = (self.find(a), self.find(b));
        if ra != rb { self.parent[ra] = rb; }
    }
}

// Umeyama's method: the least-squares rigid transform mapping point set b
// onto point set a (rotation angle + translation, no scale). Used only to
// remove the gauge before scoring against ground truth.
fn umeyama(a: &[vect2f], b: &[vect2f]) -> (f32, vect2f) {
    let n = a.len() as f32;
    let ca = a.iter().fold(vect2f::new(0.0, 0.0), |s, &p| s + p) * (1.0 / n);
    let cb = b.iter().fold(vect2f::new(0.0, 0.0), |s, &p| s + p) * (1.0 / n);
    let (mut h00, mut h01, mut h10, mut h11) = (0.0f64, 0.0, 0.0, 0.0);
    for (pa, pb) in a.iter().zip(b.iter()) {
        let (dax, day) = ((pa.x - ca.x) as f64, (pa.y - ca.y) as f64);
        let (dbx, dby) = ((pb.x - cb.x) as f64, (pb.y - cb.y) as f64);
        h00 += dbx * dax; h01 += dbx * day; h10 += dby * dax; h11 += dby * day;
    }
    let svd = nalgebra::Matrix2::new(h00, h01, h10, h11).svd(true, true);
    let (u, vt) = (svd.u.unwrap(), svd.v_t.unwrap());
    let mut r = vt.transpose() * u.transpose();
    if r.determinant() < 0.0 { let mut v2 = vt; v2.row_mut(1).scale_mut(-1.0); r = v2.transpose() * u.transpose(); }
    let theta = (r[(1, 0)]).atan2(r[(0, 0)]) as f32;
    (theta, ca - matrix2f::rotation(theta) * cb)
}

fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let h6 = (h - h.floor()) * 6.0;
    let c = v * s;
    let x = c * (1.0 - ((h6 % 2.0) - 1.0).abs());
    let (r, g, b) = match h6 as u32 {
        0 => (c, x, 0.0), 1 => (x, c, 0.0), 2 => (0.0, c, x),
        3 => (0.0, x, c), 4 => (x, 0.0, c), _ => (c, 0.0, x),
    };
    let m = v - c;
    (r + m, g + m, b + m)
}

// Per-run CORRECTED trajectories (one hue each) over the shared GT (gray),
// with the fused landmarks (red) and their error lines to GT -- everything
// in the solved frame.
fn write_eps(gt_poses: &[std::vec::Vec<(vect2f, f32)>], gt_lms: &[vect2f],
             run_poses: &[std::vec::Vec<(vect2f, f32)>],
             fused: &[(vect2f, usize)],
             ellipses: &[(vect2f, f32, f32, f32)], filename: &str) -> std::io::Result<()> {
    use std::io::Write;
    let mut pts: std::vec::Vec<vect2f> = std::vec::Vec::new();
    for gp in gt_poses { for (p, _) in gp { pts.push(*p); } }
    for &(_, gid) in fused { pts.push(gt_lms[gid]); }

    let xmin = pts.iter().map(|p| p.x).fold(f32::INFINITY, f32::min) - 3.0;
    let xmax = pts.iter().map(|p| p.x).fold(f32::NEG_INFINITY, f32::max) + 3.0;
    let ymin = pts.iter().map(|p| p.y).fold(f32::INFINITY, f32::min) - 3.0;
    let ymax = pts.iter().map(|p| p.y).fold(f32::NEG_INFINITY, f32::max) + 3.0;
    let (page_w, page_h, pad) = (620.0_f32, 460.0_f32, 18.0_f32);
    let s = ((page_w - 2.0 * pad) / (xmax - xmin)).min((page_h - 2.0 * pad) / (ymax - ymin));
    let dx = (page_w - s * (xmax - xmin)) * 0.5;
    let dy = (page_h - s * (ymax - ymin)) * 0.5;
    let to_pg = |p: vect2f| (dx + (p.x - xmin) * s, dy + (p.y - ymin) * s);

    let mut f = std::fs::File::create(filename)?;
    writeln!(f, "%!PS-Adobe-3.0 EPSF-3.0")?;
    writeln!(f, "%%BoundingBox: 0 0 {} {}", page_w as i32, page_h as i32)?;
    writeln!(f, "%%Creator: slam2d_direct_align_demo\n%%EndComments")?;
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

    writeln!(f, "0.66 0.66 0.66 setrgbcolor 0.7 setlinewidth [3 2] 0 setdash")?;
    for gp in gt_poses {
        let chain: std::vec::Vec<(f32, f32)> = gp.iter().map(|(p, _)| to_pg(*p)).collect();
        polyline(&mut f, &chain)?;
    }
    writeln!(f, "[] 0 setdash")?;

    let n = run_poses.len();
    for (r, rp) in run_poses.iter().enumerate() {
        let (cr, cg, cb) = hsv_to_rgb(if n == 0 { 0.0 } else { r as f32 / n as f32 }, 0.85, 0.72);
        writeln!(f, "{:.3} {:.3} {:.3} setrgbcolor 1.0 setlinewidth [4 2] 0 setdash", cr, cg, cb)?;
        let chain: std::vec::Vec<(f32, f32)> = rp.iter().map(|(p, _)| to_pg(*p)).collect();
        polyline(&mut f, &chain)?;
        writeln!(f, "[] 0 setdash")?;
        for (p, g) in rp {
            let (x, y) = to_pg(*p);
            writeln!(f, "{:.2} {:.2} {:.2} 5.0 tri", x, y, g.to_degrees())?;
        }
    }

    writeln!(f, "0.55 0.55 0.55 setrgbcolor 0.5 setlinewidth")?;
    for &(c, gid) in fused {
        let (ox, oy) = to_pg(c);
        let (gx, gy) = to_pg(gt_lms[gid]);
        writeln!(f, "newpath {:.2} {:.2} moveto {:.2} {:.2} lineto stroke", ox, oy, gx, gy)?;
    }
    for &(_, gid) in fused {
        let (x, y) = to_pg(gt_lms[gid]);
        writeln!(f, "{:.2} {:.2} 2.0 dot", x, y)?;
    }
    writeln!(f, "0.80 0.12 0.12 setrgbcolor 0.5 setlinewidth")?;
    for &(c, a, b, t) in ellipses {
        if a <= 0.0 || b <= 0.0 { continue; }
        let (ct, st) = (t.cos(), t.sin());
        write!(f, "newpath ")?;
        for j in 0..=48 {
            let phi = 2.0 * std::f32::consts::PI * (j as f32) / 48.0;
            let (lx, ly) = (a * phi.cos(), b * phi.sin());
            let (px, py) = to_pg(vect2f::new(c.x + ct * lx - st * ly, c.y + st * lx + ct * ly));
            if j == 0 { write!(f, "{:.2} {:.2} moveto ", px, py)?; }
            else { write!(f, "{:.2} {:.2} lineto ", px, py)?; }
        }
        writeln!(f, "closepath stroke")?;
    }

    writeln!(f, "0.80 0.12 0.12 setrgbcolor")?;
    for &(c, _) in fused {
        let (x, y) = to_pg(c);
        writeln!(f, "{:.2} {:.2} 3.0 dot", x, y)?;
    }
    writeln!(f, "showpage")?;
    Ok(())
}

fn main() {
    let cfg = Cfg::default();
    let scene = scene2d::generate(&cfg);

    // Stage 1: solve each run independently, GPS-anchored (world frame).
    println!("== Stage 1: independent per-run GPS SLAM ==");
    let runs: std::vec::Vec<RunMap> = (0..cfg.n_runs)
        .map(|r| stage1(&cfg, &scene.runs[r], scene.gps_isigma, &scene.sightings, r)).collect();
    for (r, rm) in runs.iter().enumerate() {
        println!("  run {}: {} landmarks mapped", r, rm.mu.len());
    }

    // Match landmarks across runs (here by gt id) and group the sightings
    // into components with union-find: each component is one physical
    // landmark. The component SIZE feeds the clique weighting below, and
    // Stage 3 fuses per component.
    let mut node_ids: std::collections::HashMap<(usize, usize), usize> = std::collections::HashMap::new();
    let mut nodes: std::vec::Vec<(usize, usize)> = std::vec::Vec::new();
    for (r, rm) in runs.iter().enumerate() {
        for i in 0..rm.mu.len() {
            node_ids.insert((r, i), nodes.len());
            nodes.push((r, i));
        }
    }
    let mut match_edges: std::vec::Vec<(usize, usize, usize, usize)> = std::vec::Vec::new();
    for ra in 0..runs.len() {
        for rb in (ra + 1)..runs.len() {
            for (ia, &gid) in runs[ra].gt_ids.iter().enumerate() {
                if let Some(ib) = runs[rb].gt_ids.iter().position(|&g| g == gid) {
                    match_edges.push((ra, ia, rb, ib));
                }
            }
        }
    }
    let mut uf = UnionFind::new(nodes.len());
    for &(ra, ia, rb, ib) in &match_edges {
        uf.union(node_ids[&(ra, ia)], node_ids[&(rb, ib)]);
    }
    let mut comps: std::collections::HashMap<usize, std::vec::Vec<(usize, usize)>> = std::collections::HashMap::new();
    for (n, &(r, i)) in nodes.iter().enumerate() {
        comps.entry(uf.find(n)).or_default().push((r, i));
    }
    let mut comp_size = vec![0usize; nodes.len()];
    for (&root, members) in &comps {
        comp_size[root] = members.len();
    }

    // Stage 2: align the paths. One PathPair per run pair; every landmark
    // both runs saw becomes a match on the pair's shared cross block.
    let mut amap = AlignMap { paths: refs::Arena::new(), pairs: std::vec::Vec::new() };
    let mut path_refs: std::vec::Vec<Ref<PathInstance>> = std::vec::Vec::new();
    let mut lm_refs: std::vec::Vec<std::vec::Vec<Ref<RunLandmark>>> = std::vec::Vec::new();
    for rm in &runs {
        let mut landmarks = refs::Arena::new();
        lm_refs.push((0..rm.mu.len()).map(|i| landmarks.push(RunLandmark {
            mu: rm.mu[i], cov_r: rm.cov_r[i], cov_isigma: rm.cov_isigma[i] })).collect());
        path_refs.push(amap.paths.push(PathInstance {
            translation: Param::new(vect2f::new(0.0, 0.0)), rotation: Param::new(0.0),
            center: rm.center, ccov_r: rm.ccov_r, ccov_isigma: rm.ccov_isigma,
            landmarks, hb: SelfBlock::new() }));
    }
    for ra in 0..runs.len() {
        for rb in (ra + 1)..runs.len() {
            let (r1, r2) = (&runs[ra], &runs[rb]);
            let mut matches = std::vec::Vec::new();
            for &(ea, ia, eb, ib) in &match_edges {
                if ea != ra || eb != rb { continue; }
                // Whitening from BOTH runs' stage-1 ellipses: the residual
                // compares two uncertain positions, so the pair covariance
                // is the sum. Corrections are small; a fixed whitening from
                // the stage-1 rotations is fine.
                //
                // Clique weighting: a landmark seen by k runs yields
                // k(k-1)/2 pairwise meet-residuals, but only k-1 of them
                // are independent (q_A - q_C is the sum of q_A - q_B and
                // q_B - q_C). Feeding the full clique as if independent
                // multiplies its information by [k(k-1)/2]/(k-1) = k/2 and
                // the alignment posterior comes out k/2 over-confident.
                // Scaling every residual of that landmark by sqrt(2/k)
                // scales each pair's information by 2/k, so the clique
                // totals k(k-1)/2 * 2/k = k-1 -- the independent count.
                // With equal sighting covariances this reproduces, block
                // for block, the information left after marginalizing the
                // landmark out of the joint problem (each diagonal becomes
                // (k-1)/k * S^-1 and each off-diagonal -S^-1/k, exactly
                // the Schur complement); with unequal ones it is the
                // matching approximation. The covariance scales by the
                // reciprocal k/2.
                let k = comp_size[uf.find(node_ids[&(ra, ia)])] as f64;
                let s = (cov_from_whitening(r1.cov_r[ia], r1.cov_isigma[ia])
                    + cov_from_whitening(r2.cov_r[ib], r2.cov_isigma[ib])) * (k / 2.0);
                let c = matrix2f::from_elements(s[(0, 0)] as f32, s[(0, 1)] as f32,
                                                s[(1, 0)] as f32, s[(1, 1)] as f32);
                let (cov_r, cov_isigma) = whitening_factors2(c);
                matches.push(PathMatch {
                    lm_a: lm_refs[ra][ia], lm_b: lm_refs[rb][ib], cov_r, cov_isigma });
            }
            if matches.is_empty() { continue; }
            println!("  pair ({}, {}): {} matches on one shared block", ra, rb, matches.len());
            amap.pairs.push(PathPair {
                a: path_refs[ra], b: path_refs[rb], matches, hb: CrossBlock::new() });
        }
    }

    println!("\n== Stage 2: align paths only ({} params) ==", 3 * runs.len());
    let result = amap.solve_sparse(&LmConfig::well_conditioned()).unwrap();
    println!("  {} iters, cost {:.4} -> {:.4}", result.iterations, result.start_cost, result.end_cost);
    for (r, fr) in amap.paths.iter().enumerate() {
        println!("  run {}: correction rotation={:+.3}deg translation=({:+.3},{:+.3})",
            r, fr.rotation.value.to_degrees(), fr.translation.value.x, fr.translation.value.y);
    }

    // Stage 3: fuse the landmarks in closed form, one component (physical
    // landmark) at a time. Solved corrections at f64 for the fusion.
    let corr: std::vec::Vec<(nalgebra::Matrix2<f64>, nalgebra::Vector2<f64>)> =
        amap.paths.iter().map(|p| {
            let th = p.rotation.value as f64;
            let rot = nalgebra::Matrix2::new(th.cos(), -th.sin(), th.sin(), th.cos());
            let t = nalgebra::Vector2::new((p.center.x + p.translation.value.x) as f64,
                                           (p.center.y + p.translation.value.y) as f64);
            (rot, t)
        }).collect();

    // Alignment posterior (2 H^-1 of the 9-param problem): the frame
    // uncertainty of the corrections, propagated into every fused
    // landmark's covariance below -- without it the ellipses would be
    // conditional on the alignment and read too confident.
    let mut sparams: std::vec::Vec<f32> = std::vec::Vec::new();
    amap.serialize(&mut sparams);
    let sn = sparams.len();
    let mut sgrad = vec![0.0_f32; sn];
    let mut shess = vec![0.0_f32; sn * sn];
    amap.calc_grad_hessian_dense(&sparams, &mut sgrad, &mut shess);
    let sh64: std::vec::Vec<f64> = shess.iter().map(|&x| x as f64).collect();
    let scov = nalgebra::linalg::Cholesky::new(nalgebra::DMatrix::from_row_slice(sn, sn, &sh64))
        .map(|c| c.inverse() * 2.0)
        .expect("stage-2 Hessian not PD");
    // Per run: parameter columns of (translation.x, translation.y, rotation).
    let pidx: std::vec::Vec<[usize; 3]> = amap.paths.iter().map(|p| {
        let t = p.translation.index() as usize;
        [t, t + 1, p.rotation.index() as usize]
    }).collect();
    // The common (gauge) rotation of the whole map is pinned only by the
    // centre priors -- report its 1-sigma so the residual frame rotation
    // in the plot reads as what it is.
    let nr = pidx.len() as f64;
    let mut common_var = 0.0;
    for a in &pidx { for b in &pidx { common_var += scov[(a[2], b[2])]; } }
    println!("  common map rotation 1-sigma: {:.3}deg (gauge, centre priors only)",
        (common_var.max(0.0).sqrt() / nr).to_degrees());

    // (fused position, gt id, fused covariance) per component. The
    // position is the closed-form optimum of the landmark given the
    // alignment; the covariance is Lambda^-1 (the conditional part) plus
    // J Sigma_c J^T, the propagated correction uncertainty, with
    // dq/dt = I and dq/dtheta = R'(theta) mu (the lever arm).
    let mut fused: std::vec::Vec<(vect2f, usize, nalgebra::Matrix2<f64>)> = std::vec::Vec::new();
    let mut multi = 0usize;
    let (mut cond_axis, mut marg_axis) = (0.0_f64, 0.0_f64);
    for members in comps.values() {
        let gid = runs[members[0].0].gt_ids[members[0].1];
        let mut lam = nalgebra::Matrix2::zeros();
        let mut eta = nalgebra::Vector2::zeros();
        let mut lams: std::vec::Vec<nalgebra::Matrix2<f64>> = std::vec::Vec::new();
        for &(r, i) in members {
            let (rot, t) = &corr[r];
            let lm = &amap.paths[path_refs[r]].landmarks[lm_refs[r][i]];
            let mu = nalgebra::Vector2::new(lm.mu.x as f64, lm.mu.y as f64);
            let q = t + rot * mu;
            let s = rot * cov_from_whitening(lm.cov_r, lm.cov_isigma) * rot.transpose();
            let li = s.try_inverse().expect("stage-1 covariance invertible");
            lam += li;
            eta += li * q;
            lams.push(li);
        }
        let cond = lam.try_inverse().expect("fused information invertible");
        let p = cond * eta;
        // Sensitivity of the fused position to the corrections: 2 x 9.
        let mut j = nalgebra::DMatrix::<f64>::zeros(2, sn);
        for (m, &(r, i)) in members.iter().enumerate() {
            let w = cond * lams[m];
            let (rot, _) = &corr[r];
            let lm = &amap.paths[path_refs[r]].landmarks[lm_refs[r][i]];
            let mu = nalgebra::Vector2::new(lm.mu.x as f64, lm.mu.y as f64);
            let dq_dth = nalgebra::Vector2::new(-rot[(1, 0)] * mu.x - rot[(0, 0)] * mu.y,
                                                 rot[(0, 0)] * mu.x - rot[(1, 0)] * mu.y);
            let wth = w * dq_dth;
            for c in 0..2 {
                for rr in 0..2 { j[(rr, pidx[r][c])] += w[(rr, c)]; }
            }
            for rr in 0..2 { j[(rr, pidx[r][2])] += wth[rr]; }
        }
        let frame = &j * &scov * j.transpose();
        let cov = cond + nalgebra::Matrix2::new(frame[(0, 0)], frame[(0, 1)],
                                                frame[(1, 0)], frame[(1, 1)]);
        cond_axis += nalgebra::SymmetricEigen::new(cond).eigenvalues.max().max(0.0).sqrt();
        marg_axis += nalgebra::SymmetricEigen::new(cov).eigenvalues.max().max(0.0).sqrt();
        let p2 = p;
        if members.len() > 1 { multi += 1; }
        fused.push((vect2f::new(p2.x as f32, p2.y as f32), gid, cov));
    }
    println!("\n== Stage 3: closed-form fusion ==");
    println!("  {} landmarks fused ({} seen by several runs)", fused.len(), multi);
    println!("  mean 1-sigma semi-major axis: conditional {:.3}m, with frame uncertainty {:.3}m",
        cond_axis / fused.len() as f64, marg_axis / fused.len() as f64);

    // Fused vs GT, gauge-aligned (the world frame carries run-0's residual
    // GPS bias, so align est -> GT before measuring).
    let est: std::vec::Vec<vect2f> = fused.iter().map(|&(p, _, _)| p).collect();
    let tru: std::vec::Vec<vect2f> = fused.iter().map(|&(_, gid, _)| scene.gt_lms[gid]).collect();
    let (ath, at) = umeyama(&tru, &est);
    let mut errs: std::vec::Vec<f32> = est.iter().zip(tru.iter())
        .map(|(&e, &g)| ((matrix2f::rotation(ath) * e + at) - g).norm()).collect();
    errs.sort_by(|a, b| a.total_cmp(b));
    let mean = errs.iter().sum::<f32>() / errs.len() as f32;
    println!("\nDIRECT-ALIGN fused landmark error vs GT: mean={:.4}m median={:.4}m max={:.4}m",
        mean, errs[errs.len() / 2], errs[errs.len() - 1]);
    println!("(compare to slam2d_align_demo's joint stage 2 and slam2d_multi_demo's full solve)");

    // 95% ellipses from the fused covariances (chi^2(0.95,2)=5.991),
    // frame uncertainty included -- comparable to slam2d_align_demo's
    // joint marginals.
    let mut plot: std::vec::Vec<(vect2f, usize)> = fused.iter().map(|&(p, g, _)| (p, g)).collect();
    plot.sort_by_key(|&(_, g)| g);
    let ellipses: std::vec::Vec<(vect2f, f32, f32, f32)> = fused.iter().map(|&(p, _, cov)| {
        let e = nalgebra::SymmetricEigen::new(cov);
        let (i0, i1) = if e.eigenvalues[0] >= e.eigenvalues[1] { (0, 1) } else { (1, 0) };
        let a = (e.eigenvalues[i0].max(0.0) * 5.991).sqrt() as f32;
        let b = (e.eigenvalues[i1].max(0.0) * 5.991).sqrt() as f32;
        let ang = (e.eigenvectors[(1, i0)] as f32).atan2(e.eigenvectors[(0, i0)] as f32);
        (p, a, b, ang)
    }).collect();

    // Plot the CORRECTED trajectories -- the frame the landmarks live in.
    let run_poses: std::vec::Vec<std::vec::Vec<(vect2f, f32)>> =
        runs.iter().zip(amap.paths.iter()).map(|(rm, fr)| {
            let rot = matrix2f::rotation(fr.rotation.value);
            rm.poses.iter().map(|&(p, g)| {
                (rm.center + fr.translation.value + rot * (p - rm.center),
                 g + fr.rotation.value)
            }).collect()
        }).collect();
    let out = "slam2d_direct_align.eps";
    write_eps(&scene.gt_poses, &scene.gt_lms, &run_poses, &plot, &ellipses, out).unwrap();
    println!("wrote {}", out);
}
