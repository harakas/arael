//! Building one map from several runs over the same area -- the cheap, two-step
//! way, compared against the all-at-once way in slam2d_multi_demo. Both use the
//! exact same data (examples/shared/scene2d.rs) so the results line up.
//!
//! The idea
//! --------
//! Three runs (three drives through the same place) each carry GPS and each spot
//! the same landmarks, seeing only their DIRECTION, not their distance. We want
//! a single map that agrees with all three runs.
//!
//! slam2d_multi_demo does it the thorough way: it throws every pose, landmark
//! and measurement into one big optimisation and solves them together. This demo
//! does it cheaply instead -- first solve each run on its own, then combine just
//! the small per-run summaries. The question it answers: is the cheap two-step
//! map as good as the big all-at-once one? (It nearly is.)
//!
//! How it works
//! ------------
//! Stage 1 -- solve each run by itself. Because each run has GPS, its map comes
//! out in real-world coordinates. From each solved run we keep a representative
//! "centre" pose (position + heading) with its uncertainty, and, for every
//! landmark, where it sits relative to that centre plus how sure the run is of
//! it (an uncertainty ellipse).
//!
//! Stage 2 -- one small optimisation that fuses the runs into a single map. It
//! solves for a merged position for each landmark and, for each run, a small
//! translation and rotation that shifts and turns that run's whole map to line
//! it up with the others.
//!
//! Two things hold those unknowns in place. The first is landmark agreement:
//! the position a run gave a landmark in Stage 1, once that run's correction is
//! applied, should land on the landmark's merged position -- and the leftover
//! gap counts for more along the directions the run measured confidently (its
//! uncertainty ellipse). The second keeps each run's correction honest: it is
//! pulled back toward "no change", as firmly as GPS pinned that run, so a
//! well-fixed run barely moves while a poorly-fixed one may slide further.
//!
//! That second part is what makes the cheap method work. It preserves the fact
//! that all of a run's landmarks move together when the run itself moves -- which
//! the per-landmark ellipses alone throw away. Keep it and the two-step map
//! matches the all-at-once solve; drop it and the runs drift apart.
//!
//! Run:  cargo run -r --example slam2d_align_demo

use arael::covariance::{CovMode, Covariance};
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
// Stage 2 model: rigid correction + consensus landmarks
// ===========================================================================

// One run's SE2 correction about its centre pose -- a small translation and
// rotation -- with the centre pose's 3x3 (x, y, heading) covariance as a prior.
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
    hb: SelfBlock<PathInstance, f32>,
}

#[arael::model]
struct MergedLandmark {
    pos: Param<vect2f>,
    hb: SelfBlock<MergedLandmark, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let theta = path.rotation;
    let e_local = landmarkinstance.mu + matrix2sym::rotation(theta).transpose() * (path.center + path.translation - landmark.pos);
    let w = landmarkinstance.cov_r.transpose() * e_local;
    [w.x * landmarkinstance.cov_isigma.x, w.y * landmarkinstance.cov_isigma.y]
}))]
struct LandmarkInstance {
    #[arael(ref = root.paths)] path: Ref<PathInstance>,
    #[arael(ref = root.landmarks)] landmark: Ref<MergedLandmark>,
    mu: vect2f,
    cov_r: matrix2f,
    cov_isigma: vect2f,
    hb: CrossBlock<PathInstance, MergedLandmark, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct Map {
    paths: refs::Arena<PathInstance>,
    landmarks: refs::Arena<MergedLandmark>,
    frines: std::vec::Vec<LandmarkInstance>,
}

// ===========================================================================
// Covariance helpers
// ===========================================================================

// Eigendecompose a covariance into whitening factors: the rotation into its
// eigenbasis and the per-axis inverse std-devs (1/sqrt of the eigenvalues).
// A residual e is then whitened by  isigma .* (rotation^T * e).
fn whitening_factors2(c: matrix2f) -> (matrix2f, vect2f) {
    let (r, d) = c.symmetric_eigen();
    (r, vect2f::new(1.0 / d.x.max(1e-12).sqrt(), 1.0 / d.y.max(1e-12).sqrt()))
}

// 3x3 counterpart of whitening_factors2, for the centre-pose (x, y, heading)
// covariance.
fn whitening_factors3(c: [[f64; 3]; 3]) -> (matrix3f, vect3f) {
    let (r, d) = matrix3d::from_array(c).symmetric_eigen();
    (r.cast::<f32>(), vect3f::new(1.0 / d.x.max(1e-12).sqrt() as f32,
                                  1.0 / d.y.max(1e-12).sqrt() as f32,
                                  1.0 / d.z.max(1e-12).sqrt() as f32))
}

// ===========================================================================
// Stage 1: solve one run (GPS-anchored), extract map + covariances
// ===========================================================================

struct RunMap {
    gt_ids: std::vec::Vec<usize>,
    mu: std::vec::Vec<vect2f>,         // landmark mean relative to centre (world)
    cov_r: std::vec::Vec<matrix2f>,
    cov_isigma: std::vec::Vec<vect2f>,
    center: vect2f,                    // centre pose position (world)
    ccov_r: matrix3f,                  // centre pose 3x3 covariance whitening
    ccov_isigma: vect3f,
    lm_world: std::vec::Vec<vect2f>,   // landmark means (world) for init/plot
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

    path.solve_sparse(&LmConfig::well_conditioned());

    // Parameter covariance at the stage-1 solution.
    let cov = path.assemble_covariance(CovMode::PerQuery).expect("stage-1 Hessian not PD");

    // Centre = middle pose; its position is the correction pivot, its 3x3
    // (x, y, gamma) MARGINAL covariance is the frame prior.
    let mid = path.poses.len() / 2;
    let center = path.poses[mid].pos.value;
    let cm = cov.marginal_cov(&path.poses[mid]);
    let mut c3 = [[0.0_f64; 3]; 3];
    for a in 0..3 { for b in 0..3 { c3[a][b] = cm[(a, b)]; } }
    let (ccov_r, ccov_isigma) = whitening_factors3(c3);

    let lm_world: std::vec::Vec<vect2f> = path.landmarks.iter().map(|l| l.pos.value).collect();
    let mut mu = std::vec::Vec::new();
    let mut cov_r = std::vec::Vec::new();
    let mut cov_isigma = std::vec::Vec::new();
    for lm in path.landmarks.iter() {
        // Landmark covariance with the poses held FIXED (conditional, not
        // marginal): the pose/frame uncertainty is carried by the centre prior,
        // so folding it back in via the marginal would double-count it and
        // inflate the ellipse isotropically.
        let c2 = cov.conditional_cov(lm);
        let c = matrix2f::from_elements(c2[(0, 0)] as f32, c2[(0, 1)] as f32,
                                        c2[(1, 0)] as f32, c2[(1, 1)] as f32);
        let (r, isig) = whitening_factors2(c);
        mu.push(lm.pos.value - center);
        cov_r.push(r);
        cov_isigma.push(isig);
    }

    RunMap {
        gt_ids, mu, cov_r, cov_isigma, center, ccov_r, ccov_isigma, lm_world,
        poses: path.poses.iter().map(|p| (p.pos.value, p.gamma.value)).collect(),
    }
}

// Umeyama's method: the least-squares rigid transform (rotation angle +
// translation) that best maps point set b onto point set a. Rigid, no scale;
// SVD of the cross-covariance with the standard reflection fix. Used only to
// remove the gauge freedom before scoring -- the decoupled map is correct up to
// a global rotation+shift (the world frame carries run 0's residual GPS bias),
// so we align the estimate to ground truth, then measure the leftover error.
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

// Decoupled fuse: per-run Stage-1 trajectories (one hue each) over the shared GT
// (gray), with the consensus landmarks (red) and their error lines to GT.
fn write_eps(gt_poses: &[std::vec::Vec<(vect2f, f32)>], gt_lms: &[vect2f],
             run_poses: &[std::vec::Vec<(vect2f, f32)>],
             consensus: &[(vect2f, usize)],
             ellipses: &[(vect2f, f32, f32, f32)], filename: &str) -> std::io::Result<()> {
    use std::io::Write;
    // Bounding box from the shared ground truth: all runs' poses + the consensus
    // landmarks' GT (identical set to slam2d_multi_demo's on the same scene), so
    // the two plots share a page transform and line up. Excluding barely-seen
    // edge landmarks (never drawn) keeps the frame tight.
    let mut pts: std::vec::Vec<vect2f> = std::vec::Vec::new();
    for gp in gt_poses { for (p, _) in gp { pts.push(*p); } }
    for &(_, gid) in consensus { pts.push(gt_lms[gid]); }

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
    writeln!(f, "%%Creator: slam2d_align_demo\n%%EndComments")?;
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

    // Ground-truth trajectories (gray dashed).
    writeln!(f, "0.66 0.66 0.66 setrgbcolor 0.7 setlinewidth [3 2] 0 setdash")?;
    for gp in gt_poses {
        let chain: std::vec::Vec<(f32, f32)> = gp.iter().map(|(p, _)| to_pg(*p)).collect();
        polyline(&mut f, &chain)?;
    }
    writeln!(f, "[] 0 setdash")?;

    // Each run's independent Stage-1 trajectory, one hue.
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

    // Consensus landmarks (red) with GT dots and error lines.
    writeln!(f, "0.55 0.55 0.55 setrgbcolor 0.5 setlinewidth")?;
    for &(c, gid) in consensus {
        let (ox, oy) = to_pg(c);
        let (gx, gy) = to_pg(gt_lms[gid]);
        writeln!(f, "newpath {:.2} {:.2} moveto {:.2} {:.2} lineto stroke", ox, oy, gx, gy)?;
    }
    for &(_, gid) in consensus {
        let (x, y) = to_pg(gt_lms[gid]);
        writeln!(f, "{:.2} {:.2} 2.0 dot", x, y)?;
    }
    // 95% confidence ellipses per consensus landmark (dark red outline).
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
    for &(c, _) in consensus {
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

    // Stage 2: build the fuse. GPS already put every run in the world frame, so
    // corrections start at identity; the centre prior + shared landmarks refine.
    let mut amap = Map { paths: refs::Arena::new(), landmarks: refs::Arena::new(), frines: std::vec::Vec::new() };
    // Keep the handle each push hands back: an arena chooses the slot.
    let path_refs: std::vec::Vec<Ref<PathInstance>> = runs.iter().map(|rm| {
        amap.paths.push(PathInstance {
            translation: Param::new(vect2f::new(0.0, 0.0)), rotation: Param::new(0.0),
            center: rm.center, ccov_r: rm.ccov_r, ccov_isigma: rm.ccov_isigma, hb: SelfBlock::new() })
    }).collect();
    // Consensus landmarks: one per GT id, init = mean of the runs' world estimates.
    let mut gid_to_lm: std::collections::HashMap<usize, Ref<MergedLandmark>> = std::collections::HashMap::new();
    let mut acc: std::collections::HashMap<Ref<MergedLandmark>, (vect2f, f32)> = std::collections::HashMap::new();
    for rm in &runs {
        for (i, &gid) in rm.gt_ids.iter().enumerate() {
            let idx = *gid_to_lm.entry(gid).or_insert_with(|| {
                amap.landmarks.push(MergedLandmark { pos: Param::new(vect2f::new(0.0, 0.0)), hb: SelfBlock::new() })
            });
            let e = acc.entry(idx).or_insert((vect2f::new(0.0, 0.0), 0.0));
            e.0 = e.0 + rm.lm_world[i]; e.1 += 1.0;
        }
    }
    for (idx, (sum, cnt)) in &acc { amap.landmarks[*idx].pos.value = *sum * (1.0 / cnt); }
    for (r, rm) in runs.iter().enumerate() {
        for (i, &gid) in rm.gt_ids.iter().enumerate() {
            amap.frines.push(LandmarkInstance {
                path: path_refs[r], landmark: gid_to_lm[&gid],
                mu: rm.mu[i], cov_r: rm.cov_r[i], cov_isigma: rm.cov_isigma[i], hb: CrossBlock::new() });
        }
    }

    println!("\n== Stage 2: fuse (corrections + centre priors + landmark obs) ==");
    println!("  {} paths, {} consensus landmarks, {} observations",
        amap.paths.len(), amap.landmarks.len(), amap.frines.len());
    let result = amap.solve_sparse(&LmConfig::well_conditioned());
    println!("  {} iters, cost {:.4} -> {:.4}", result.iterations, result.start_cost, result.end_cost);
    for (r, fr) in amap.paths.iter().enumerate() {
        println!("  run {}: correction rotation={:+.3}deg translation=({:+.3},{:+.3})",
            r, fr.rotation.value.to_degrees(), fr.translation.value.x, fr.translation.value.y);
    }

    // Consensus vs GT, gauge-aligned (the world frame carries run-0's residual
    // GPS/gauge, so align est -> GT before measuring).
    let mut est = std::vec::Vec::new();
    let mut tru = std::vec::Vec::new();
    for (&gid, &idx) in &gid_to_lm {
        est.push(amap.landmarks[idx].pos.value);
        tru.push(scene.gt_lms[gid]);
    }
    let (ath, at) = umeyama(&tru, &est);
    let mut errs: std::vec::Vec<f32> = est.iter().zip(tru.iter())
        .map(|(&e, &g)| ((matrix2f::rotation(ath) * e + at) - g).norm()).collect();
    errs.sort_by(|a, b| a.total_cmp(b));
    let mean = errs.iter().sum::<f32>() / errs.len() as f32;
    println!("\nDECOUPLED consensus landmark error vs GT: mean={:.4}m median={:.4}m max={:.4}m",
        mean, errs[errs.len() / 2], errs[errs.len() - 1]);
    println!("(compare to slam2d_multi_demo's JOINT solve on the same scene)");

    // Stage-2 consensus covariance (2 H^-1) -> 95% ellipses (chi^2(0.95,2)=5.991),
    // the decoupled counterpart to slam2d_multi's joint landmark ellipses.
    let mut sparams: std::vec::Vec<f32> = std::vec::Vec::new();
    amap.serialize32(&mut sparams);
    let sn = sparams.len();
    let mut sgrad = vec![0.0_f32; sn];
    let mut shess = vec![0.0_f32; sn * sn];
    amap.calc_grad_hessian_dense(&sparams, &mut sgrad, &mut shess);
    let sh64: std::vec::Vec<f64> = shess.iter().map(|&x| x as f64).collect();
    let scov = nalgebra::linalg::Cholesky::new(nalgebra::DMatrix::from_row_slice(sn, sn, &sh64))
        .map(|c| c.inverse() * 2.0);

    let run_poses: std::vec::Vec<_> = runs.iter().map(|rm| rm.poses.clone()).collect();
    let mut consensus: std::vec::Vec<(vect2f, usize)> = gid_to_lm.iter()
        .map(|(&gid, &idx)| (amap.landmarks[idx].pos.value, gid)).collect();
    consensus.sort_by_key(|&(_, g)| g);
    let ellipses: std::vec::Vec<(vect2f, f32, f32, f32)> = match &scov {
        None => { println!("Stage-2 Hessian not PD -- skipping ellipses."); std::vec::Vec::new() }
        Some(cov) => consensus.iter().map(|&(c, gid)| {
            let k = amap.landmarks[gid_to_lm[&gid]].pos.index() as usize;
            let e = nalgebra::SymmetricEigen::new(cov.fixed_view::<2, 2>(k, k).clone_owned());
            let (i0, i1) = if e.eigenvalues[0] >= e.eigenvalues[1] { (0, 1) } else { (1, 0) };
            let a = (e.eigenvalues[i0].max(0.0) * 5.991).sqrt() as f32;
            let b = (e.eigenvalues[i1].max(0.0) * 5.991).sqrt() as f32;
            let ang = (e.eigenvectors[(1, i0)] as f32).atan2(e.eigenvectors[(0, i0)] as f32);
            (c, a, b, ang)
        }).collect(),
    };

    let out = "slam2d_align.eps";
    write_eps(&scene.gt_poses, &scene.gt_lms, &run_poses, &consensus, &ellipses, out).unwrap();
    println!("wrote {}", out);
}
