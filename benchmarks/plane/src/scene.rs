// Plane SLAM scene: SE3 poses on a loop inside a room, plane landmarks
// (normal + distance) observed relative to each pose, odometry between
// consecutive poses. Generated in-process, deterministic; the text file
// exists only as the handoff for the external C++ runners. Also holds the
// ONE reference cost function (scene_cost) every runner is validated
// against, and the solution/ground-truth error metrics.
//
// Plane convention (g2o types/slam3d_addons/plane3d.h): coeffs (n, c) with
// |n| = 1 and plane equation n.x + c = 0; distance() = -c. Transform by an
// isometry t: n' = R n, c' = c - t.trans . n'. The observation stored for a
// pose T is the world plane transformed by T^-1.

use arael::quatern::quaternd;
use arael::vect::vect3d;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

const DEFAULT_POSES: usize = 64;
pub const SEED: u64 = 42;

/// Scene size: PLANE_POSES=N (like SLAM_POSES / LOC_POSES on the sibling
/// benchmarks). Odometry, plane and observation counts scale with the pose
/// count; PLANE_SHARED=1 switches to the fixed 6-plane degenerate mode.
fn n_poses() -> usize {
    std::env::var("PLANE_POSES").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_POSES)
}
const CEIL: f64 = 3.0;
/// Distance between consecutive poses, constant at every scene size (like
/// the slam benchmark's fixed step_size): the loop RADIUS grows with the
/// pose count, so index-based windows are also fixed spatial windows.
const POSE_STEP: f64 = 0.6;
const SIGMA_ODO_T: f64 = 0.02; // m, per axis
const SIGMA_ODO_R: f64 = 0.005; // rad, per axis
const SIGMA_OBS_ANG: f64 = 0.01; // rad, azimuth and elevation
const SIGMA_OBS_D: f64 = 0.02; // m

#[derive(Clone, Copy)]
pub struct Pose {
    pub q: quaternd,
    pub t: vect3d,
}

impl Pose {
    pub fn mul(self, o: Pose) -> Pose {
        Pose { q: (self.q * o.q).unit(), t: self.t + self.q.rotate(o.t) }
    }
    pub fn inverse(self) -> Pose {
        let qi = self.q.conj();
        Pose { q: qi, t: -qi.rotate(self.t) }
    }
}

/// Plane as (n, c), |n| = 1, n.x + c = 0.
#[derive(Clone, Copy)]
pub struct Plane {
    pub n: vect3d,
    pub c: f64,
}

impl Plane {
    pub fn normalized(n: vect3d, c: f64) -> Plane {
        let s = 1.0 / n.norm();
        Plane { n: n * s, c: c * s }
    }
    /// The g2o operator: plane transformed by the isometry `t`.
    pub fn transform(self, t: Pose) -> Plane {
        let n = t.q.rotate(self.n);
        Plane { n, c: self.c - (t.t * n) }
    }
    fn azimuth(v: vect3d) -> f64 {
        v.y.atan2(v.x)
    }
    fn elevation(v: vect3d) -> f64 {
        v.z.atan2((v.x * v.x + v.y * v.y).sqrt())
    }
    /// The g2o oplus: perturb by (d_azimuth, d_elevation, d_distance).
    pub fn oplus(self, v: [f64; 3]) -> Plane {
        let (s, c) = v[1].sin_cos();
        let n_local = vect3d::new(c * v[0].cos(), c * v[0].sin(), s);
        // rotation taking (1,0,0) to the current normal, azimuth/elevation form
        let az = Self::azimuth(self.n);
        let el = Self::elevation(self.n);
        let r = quaternd::from_axis_angle(vect3d::new(0.0, 0.0, 1.0), az)
            * quaternd::from_axis_angle(vect3d::new(0.0, 1.0, 0.0), -el);
        let n = r.rotate(n_local);
        let d = -self.c + v[2];
        Plane::normalized(n, -d)
    }
}

fn gauss(rng: &mut StdRng, sigma: f64) -> f64 {
    let x: f64 = rng.sample(StandardNormal);
    x * sigma
}

fn path_radius(n: usize) -> f64 {
    n as f64 * POSE_STEP / std::f64::consts::TAU
}

fn gt_pose(i: usize, n: usize) -> Pose {
    let th = i as f64 / n as f64 * std::f64::consts::TAU;
    let yaw = th + std::f64::consts::FRAC_PI_2;
    // A little roll/pitch wobble: with yaw-only poses the floor/ceiling
    // normals sit exactly on the azimuth chart's pole (atan2(0,0)) in the
    // sensor frame, where the observation Jacobian is undefined for every
    // implementation alike.
    let roll = 0.06 * (3.0 * th).sin();
    let pitch = 0.05 * (2.0 * th).cos();
    Pose {
        q: (quaternd::from_axis_angle(vect3d::new(0.0, 0.0, 1.0), yaw)
            * quaternd::from_axis_angle(vect3d::new(1.0, 0.0, 0.0), roll)
            * quaternd::from_axis_angle(vect3d::new(0.0, 1.0, 0.0), pitch))
        .unit(),
        t: vect3d::new(path_radius(n) * th.cos(), path_radius(n) * th.sin(), 1.0),
    }
}

/// Degenerate mode (PLANE_SHARED=1): the 6 room planes, every pose seeing
/// all of them -- a tiny landmark family shared by everyone, which makes the
/// reduced Schur system fully dense (the gate stress case). The room walls
/// sit 4 m outside the loop, whatever its radius.
fn shared_planes(n_poses: usize) -> Vec<Plane> {
    let room = path_radius(n_poses) + 4.0;
    vec![
        Plane { n: vect3d::new(-1.0, 0.0, 0.0), c: room },
        Plane { n: vect3d::new(1.0, 0.0, 0.0), c: room },
        Plane { n: vect3d::new(0.0, -1.0, 0.0), c: room },
        Plane { n: vect3d::new(0.0, 1.0, 0.0), c: room },
        Plane { n: vect3d::new(0.0, 0.0, 1.0), c: 0.0 },
        Plane { n: vect3d::new(0.0, 0.0, -1.0), c: CEIL },
    ]
}

/// Default mode: planes scale with the pose count, each visible from a
/// LIMITED window of poses (like the slam benchmark's landmark spans). One
/// anchor per ANCHOR_SPACING poses carries a triplet of planes with spanning
/// normals -- an inward wall, a tilted floor patch, a tilted side wall -- so
/// every pose window sees at least three independent orientations.
const ANCHOR_SPACING: usize = 8;
const VIS_WINDOW: usize = 6; // wrapped pose distance to the anchor

fn plane_at(point: vect3d, n: vect3d) -> Plane {
    let n = n.unit();
    Plane { n, c: -(n * point) }
}

fn scaled_planes(n_poses: usize) -> (Vec<Plane>, Vec<usize>) {
    let r = path_radius(n_poses);
    let n_anchors = (n_poses + ANCHOR_SPACING - 1) / ANCHOR_SPACING;
    let mut planes = Vec::new();
    let mut centers = Vec::new();
    for a in 0..n_anchors {
        let center = a * ANCHOR_SPACING + ANCHOR_SPACING / 2;
        let th = center as f64 / n_poses as f64 * std::f64::consts::TAU;
        let dir = vect3d::new(th.cos(), th.sin(), 0.0);
        let tang = vect3d::new(-th.sin(), th.cos(), 0.0);
        // inward wall outside the path
        planes.push(plane_at(dir * (r + 3.0), -dir));
        // tilted floor patch through the anchor's ground point (distinct
        // per anchor, and always local to the poses that see it)
        planes.push(plane_at(dir * r,
            (vect3d::new(0.0, 0.0, 1.0) + dir * 0.1).unit()));
        // tilted side wall along the tangent
        planes.push(plane_at(dir * (r - 3.0) + vect3d::new(0.0, 0.0, 1.5),
            (dir + tang * 0.3 + vect3d::new(0.0, 0.0, 0.2)).unit()));
        centers.extend([center, center, center]);
    }
    (planes, centers)
}

fn wrapped_dist(i: usize, c: usize, n: usize) -> usize {
    let d = (i as i64 - c as i64).unsigned_abs() as usize % n;
    d.min(n - d)
}

fn fmt_pose(p: &Pose) -> String {
    format!(
        "{} {} {} {} {} {} {}",
        p.t.x, p.t.y, p.t.z, p.q.t, p.q.v.x, p.q.v.y, p.q.v.z
    )
}

/// The raw problem every runner consumes: noisy initial state plus the
/// measurements with their whitening weights.
pub struct RawScene {
    pub poses: Vec<Pose>,
    pub planes: Vec<Plane>,
    pub odos: Vec<(usize, usize, Pose, f64, f64)>,          // i, j, rel, wt, wr
    pub obs: Vec<(usize, usize, Plane, f64, f64, f64)>,     // p, l, local plane, waz, wel, wd
}

/// The generated scene: the raw problem plus ground truth for the error
/// metrics.
pub struct Scene {
    pub raw: RawScene,
    pub gt_poses: Vec<Pose>,
    pub gt_planes: Vec<Plane>,
}

pub fn make_scene() -> Scene {
    make_scene_with(n_poses())
}

/// [`make_scene`] at an explicit pose count, for callers that must not depend
/// on the environment.
pub fn make_scene_with(n: usize) -> Scene {
    let mut rng = StdRng::seed_from_u64(SEED);
    let shared = std::env::var("PLANE_SHARED").is_ok();
    let (planes, centers) = if shared {
        (shared_planes(n), Vec::new())
    } else {
        scaled_planes(n)
    };
    let gt: Vec<Pose> = (0..n).map(|i| gt_pose(i, n)).collect();
    let (wt, wr) = (1.0 / SIGMA_ODO_T, 1.0 / SIGMA_ODO_R);
    let (wa, wd) = (1.0 / SIGMA_OBS_ANG, 1.0 / SIGMA_OBS_D);

    let mut raw = RawScene { poses: vec![], planes: vec![], odos: vec![], obs: vec![] };
    for i in 0..n - 1 {
        let rel = gt[i].inverse().mul(gt[i + 1]);
        let dq = quaternd::from_rotation_vector(vect3d::new(
            gauss(&mut rng, SIGMA_ODO_R),
            gauss(&mut rng, SIGMA_ODO_R),
            gauss(&mut rng, SIGMA_ODO_R)));
        let noisy = Pose {
            q: (rel.q * dq).unit(),
            t: rel.t + vect3d::new(
                gauss(&mut rng, SIGMA_ODO_T),
                gauss(&mut rng, SIGMA_ODO_T),
                gauss(&mut rng, SIGMA_ODO_T)),
        };
        raw.odos.push((i, i + 1, noisy, wt, wr));
    }
    for (i, gp) in gt.iter().enumerate() {
        for (j, pl) in planes.iter().enumerate() {
            if !shared && wrapped_dist(i, centers[j], n) > VIS_WINDOW {
                continue;
            }
            let local = pl.transform(gp.inverse()).oplus([
                gauss(&mut rng, SIGMA_OBS_ANG),
                gauss(&mut rng, SIGMA_OBS_ANG),
                gauss(&mut rng, SIGMA_OBS_D)]);
            raw.obs.push((i, j, local, wa, wa, wd));
        }
    }
    raw.poses.push(gt[0]);
    for k in 0..raw.odos.len() {
        let last = *raw.poses.last().unwrap();
        raw.poses.push(last.mul(raw.odos[k].2));
    }
    raw.planes = (0..planes.len())
        .map(|j| {
            let &(pi, _, ref local, _, _, _) =
                raw.obs.iter().find(|(_, oj, _, _, _, _)| *oj == j).unwrap();
            local.transform(raw.poses[pi])
        })
        .collect();
    Scene { raw, gt_poses: gt, gt_planes: planes }
}

pub fn write_scene_file(path: &str, sc: &Scene) {
    let mut s = String::new();
    s.push_str("planescene 1\n");
    s.push_str(&format!("counts {} {} {} {}\n",
        sc.raw.poses.len(), sc.raw.planes.len(), sc.raw.odos.len(), sc.raw.obs.len()));
    for p in &sc.raw.poses { s.push_str(&format!("pose {}\n", fmt_pose(p))); }
    for p in &sc.gt_poses { s.push_str(&format!("gtpose {}\n", fmt_pose(p))); }
    for p in &sc.raw.planes { s.push_str(&format!("plane {} {} {} {}\n", p.n.x, p.n.y, p.n.z, p.c)); }
    for p in &sc.gt_planes { s.push_str(&format!("gtplane {} {} {} {}\n", p.n.x, p.n.y, p.n.z, p.c)); }
    for &(i, j, ref rel, wt, wr) in &sc.raw.odos {
        let (it, ir) = (wt * wt, wr * wr);
        s.push_str(&format!("odom {} {} {} {} {} {} {} {} {}\n",
            i, j, fmt_pose(rel), it, it, it, ir, ir, ir));
    }
    for &(i, j, ref p, wa, we, wd) in &sc.raw.obs {
        s.push_str(&format!("obs {} {} {} {} {} {} {} {} {}\n",
            i, j, p.n.x, p.n.y, p.n.z, p.c, wa * wa, we * we, wd * wd));
    }
    std::fs::write(path, s).unwrap();
}

/// What every runner reports back: the state its solver ended at, scored by
/// [`reference_cost`] and compared across systems by pose distance.
#[derive(Clone)]
pub struct Solution {
    pub poses: Vec<Pose>,
    pub planes: Vec<Plane>,
}

/// A runner's solution file: N pose lines "tx ty tz qw qx qy qz" then M
/// plane lines "nx ny nz c".
pub fn read_solution(path: &str, n_poses: usize) -> Solution {
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<Vec<f64>> = text.lines()
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    let poses = lines[..n_poses].iter()
        .map(|f| Pose {
            t: vect3d::new(f[0], f[1], f[2]),
            q: quaternd::new(f[3], vect3d::new(f[4], f[5], f[6])).unit(),
        })
        .collect();
    let planes = lines[n_poses..].iter()
        .map(|v| Plane::normalized(vect3d::new(v[0], v[1], v[2]), v[3]))
        .collect();
    Solution { poses, planes }
}

/// The ONE reference cost every system's solution is scored by.
pub fn reference_cost(sc: &RawScene, sol: &Solution) -> f64 {
    scene_cost(sc, &sol.poses, &sol.planes)
}

/// The benchmark's cost (chi2), evaluated on any state with the shared
/// residual definitions -- the cross-runner parity meter.
pub fn scene_cost(sc: &RawScene, poses: &[Pose], planes: &[Plane]) -> f64 {
    let mut cost = 0.0;
    for &(i, j, ref m, wt, wr) in &sc.odos {
        let ra_t = poses[i].q.conj();
        let dt = ra_t.rotate(poses[j].t - poses[i].t) - m.t;
        let dr = (m.q.conj() * poses[i].q.conj() * poses[j].q).rotation_matrix();
        let r = [dt.x * wt, dt.y * wt, dt.z * wt,
            0.5 * (dr[1].z - dr[2].y) * wr,
            0.5 * (dr[2].x - dr[0].z) * wr,
            0.5 * (dr[0].y - dr[1].x) * wr];
        cost += r.iter().map(|x| x * x).sum::<f64>();
    }
    for &(p, l, ref m, waz, wel, wd) in &sc.obs {
        let nw = planes[l].n;
        let nl = poses[p].q.conj().rotate(nw);
        let cl = planes[l].c + poses[p].t * nw;
        let h = (nl.x * nl.x + nl.y * nl.y).sqrt();
        let mx = nl * m.n;
        let my = (m.n.y * nl.x - m.n.x * nl.y) / h;
        let mz = (m.n.z * (nl.x * nl.x + nl.y * nl.y) - nl.z * (nl.x * m.n.x + nl.y * m.n.y)) / h;
        let r = [my.atan2(mx) * waz,
            mz.atan2((mx * mx + my * my).sqrt()) * wel,
            (-m.c - -cl) * wd];
        cost += r.iter().map(|x| x * x).sum::<f64>();
    }
    cost
}

