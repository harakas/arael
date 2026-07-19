// Plane SLAM scene generator + solution checker.
//
// The application is g2o's plane_slam example (libg2o-doc): SE3 poses on a
// trajectory inside a room, plane landmarks (normal + distance) observed
// relative to each pose, odometry between consecutive poses. This binary
// generates a deterministic scene file both runners read, and checks a
// runner's solution against ground truth.
//
// Plane convention (g2o types/slam3d_addons/plane3d.h): coeffs (n, c) with
// |n| = 1 and plane equation n.x + c = 0; distance() = -c. Transform by an
// isometry t: n' = R n, c' = c - t.trans . n'. The observation stored for a
// pose T is the world plane transformed by T^-1.
//
//   gen <scene.txt>            write the scene
//   check <scene.txt> <sol>    pose ATE + plane errors of a solution
//
// Solution format: N lines "tx ty tz qw qx qy qz" then M lines "nx ny nz c".

use arael::quatern::quatern;
use arael::vect::vect3;

pub type Q = quatern<f64>;
pub type V3 = vect3<f64>;

const DEFAULT_POSES: usize = 64;

/// Scene size: PLANE_POSES=N (like SLAM_POSES / LOC_POSES on the sibling
/// benchmarks). Planes stay the 6 room planes; odometry and observation
/// counts scale with the pose count.
fn n_poses() -> usize {
    std::env::var("PLANE_POSES").ok().and_then(|v| v.parse().ok()).unwrap_or(DEFAULT_POSES)
}
const ROOM: f64 = 10.0; // walls at +-ROOM in x and y
const CEIL: f64 = 3.0;
const PATH_R: f64 = 6.0;
const SIGMA_ODO_T: f64 = 0.02; // m, per axis
const SIGMA_ODO_R: f64 = 0.005; // rad, per axis
const SIGMA_OBS_ANG: f64 = 0.01; // rad, azimuth and elevation
const SIGMA_OBS_D: f64 = 0.02; // m

mod factrs_runner;

#[derive(Clone, Copy)]
pub struct Pose {
    pub q: Q,
    pub t: V3,
}

impl Pose {
    fn mul(self, o: Pose) -> Pose {
        Pose { q: (self.q * o.q).unit(), t: self.t + self.q.rotate(o.t) }
    }
    fn inverse(self) -> Pose {
        let qi = self.q.conj();
        Pose { q: qi, t: -qi.rotate(self.t) }
    }
}

/// Plane as (n, c), |n| = 1, n.x + c = 0.
#[derive(Clone, Copy)]
pub struct Plane {
    pub n: V3,
    pub c: f64,
}

impl Plane {
    fn normalized(n: V3, c: f64) -> Plane {
        let s = 1.0 / n.norm();
        Plane { n: n * s, c: c * s }
    }
    /// The g2o operator: plane transformed by the isometry `t`.
    fn transform(self, t: Pose) -> Plane {
        let n = t.q.rotate(self.n);
        Plane { n, c: self.c - (t.t * n) }
    }
    fn azimuth(v: V3) -> f64 {
        v.y.atan2(v.x)
    }
    fn elevation(v: V3) -> f64 {
        v.z.atan2((v.x * v.x + v.y * v.y).sqrt())
    }
    /// The g2o oplus: perturb by (d_azimuth, d_elevation, d_distance).
    fn oplus(self, v: [f64; 3]) -> Plane {
        let (s, c) = v[1].sin_cos();
        let n_local = V3::new(c * v[0].cos(), c * v[0].sin(), s);
        // rotation taking (1,0,0) to the current normal, azimuth/elevation form
        let az = Self::azimuth(self.n);
        let el = Self::elevation(self.n);
        let r = Q::from_axis_angle(V3::new(0.0, 0.0, 1.0), az)
            * Q::from_axis_angle(V3::new(0.0, 1.0, 0.0), -el);
        let n = r.rotate(n_local);
        let d = -self.c + v[2];
        Plane::normalized(n, -d)
    }
    /// Angle between normals, plus distance difference, vs a reference.
    fn error_vs(self, gt: Plane) -> (f64, f64) {
        let dot = (self.n * gt.n).clamp(-1.0, 1.0);
        (dot.acos(), (-self.c - -gt.c).abs())
    }
}

/// Deterministic approximate gaussian (Irwin-Hall 12).
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as f64 / (1u64 << 31) as f64
    }
    fn gauss(&mut self, sigma: f64) -> f64 {
        let s: f64 = (0..12).map(|_| self.next()).sum();
        (s - 6.0) * sigma
    }
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
        q: (Q::from_axis_angle(V3::new(0.0, 0.0, 1.0), yaw)
            * Q::from_axis_angle(V3::new(1.0, 0.0, 0.0), roll)
            * Q::from_axis_angle(V3::new(0.0, 1.0, 0.0), pitch))
        .unit(),
        t: V3::new(PATH_R * th.cos(), PATH_R * th.sin(), 1.0),
    }
}

/// Degenerate mode (PLANE_SHARED=1): the 6 room planes, every pose seeing
/// all of them -- a tiny landmark family shared by everyone, which makes the
/// reduced Schur system fully dense (the gate stress case).
fn shared_planes() -> Vec<Plane> {
    vec![
        Plane { n: V3::new(-1.0, 0.0, 0.0), c: ROOM },
        Plane { n: V3::new(1.0, 0.0, 0.0), c: ROOM },
        Plane { n: V3::new(0.0, -1.0, 0.0), c: ROOM },
        Plane { n: V3::new(0.0, 1.0, 0.0), c: ROOM },
        Plane { n: V3::new(0.0, 0.0, 1.0), c: 0.0 },
        Plane { n: V3::new(0.0, 0.0, -1.0), c: CEIL },
    ]
}

/// Default mode: planes scale with the pose count, each visible from a
/// LIMITED window of poses (like the slam benchmark's landmark spans). One
/// anchor per ANCHOR_SPACING poses carries a triplet of planes with spanning
/// normals -- an inward wall, a tilted floor patch, a tilted side wall -- so
/// every pose window sees at least three independent orientations.
const ANCHOR_SPACING: usize = 8;
const VIS_WINDOW: usize = 6; // wrapped pose distance to the anchor

fn plane_at(point: V3, n: V3) -> Plane {
    let n = n.unit();
    Plane { n, c: -(n * point) }
}

fn scaled_planes(n_poses: usize) -> (Vec<Plane>, Vec<usize>) {
    let n_anchors = (n_poses + ANCHOR_SPACING - 1) / ANCHOR_SPACING;
    let mut planes = Vec::new();
    let mut centers = Vec::new();
    for a in 0..n_anchors {
        let center = a * ANCHOR_SPACING + ANCHOR_SPACING / 2;
        let th = center as f64 / n_poses as f64 * std::f64::consts::TAU;
        let dir = V3::new(th.cos(), th.sin(), 0.0);
        let tang = V3::new(-th.sin(), th.cos(), 0.0);
        // inward wall outside the path
        planes.push(plane_at(dir * (PATH_R + 3.0), -dir));
        // tilted floor patch (distinct per anchor)
        planes.push(plane_at(V3::new(0.0, 0.0, 0.0),
            (V3::new(0.0, 0.0, 1.0) + dir * 0.1).unit()));
        // tilted side wall along the tangent
        planes.push(plane_at(dir * (PATH_R - 3.0) + V3::new(0.0, 0.0, 1.5),
            (dir + tang * 0.3 + V3::new(0.0, 0.0, 0.2)).unit()));
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

/// The scene, generated in-process (deterministic; no dataset file). The
/// text file exists only as the handoff for the external C++ runners.
pub struct Scene {
    pub raw: RawScene,
    pub gt_poses: Vec<Pose>,
    pub gt_planes: Vec<Plane>,
}

fn make_scene() -> Scene {
    let n = n_poses();
    let mut rng = Rng(0x9e3779b97f4a7c15);
    let shared = std::env::var("PLANE_SHARED").is_ok();
    let (planes, centers) = if shared {
        let p = shared_planes();
        (p, Vec::new())
    } else {
        scaled_planes(n)
    };
    let gt: Vec<Pose> = (0..n).map(|i| gt_pose(i, n)).collect();
    let (wt, wr) = (1.0 / SIGMA_ODO_T, 1.0 / SIGMA_ODO_R);
    let (wa, wd) = (1.0 / SIGMA_OBS_ANG, 1.0 / SIGMA_OBS_D);

    let mut raw = RawScene { poses: vec![], planes: vec![], odos: vec![], obs: vec![] };
    for i in 0..n - 1 {
        let rel = gt[i].inverse().mul(gt[i + 1]);
        let dq = Q::from_rotation_vector(V3::new(
            rng.gauss(SIGMA_ODO_R), rng.gauss(SIGMA_ODO_R), rng.gauss(SIGMA_ODO_R)));
        let noisy = Pose {
            q: (rel.q * dq).unit(),
            t: rel.t + V3::new(
                rng.gauss(SIGMA_ODO_T), rng.gauss(SIGMA_ODO_T), rng.gauss(SIGMA_ODO_T)),
        };
        raw.odos.push((i, i + 1, noisy, wt, wr));
    }
    for (i, gp) in gt.iter().enumerate() {
        for (j, pl) in planes.iter().enumerate() {
            if !shared && wrapped_dist(i, centers[j], n) > VIS_WINDOW {
                continue;
            }
            let local = pl.transform(gp.inverse()).oplus([
                rng.gauss(SIGMA_OBS_ANG), rng.gauss(SIGMA_OBS_ANG), rng.gauss(SIGMA_OBS_D)]);
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

fn write_scene_file(path: &str, sc: &Scene) {
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

fn gen(path: &str) {
    let sc = make_scene();
    write_scene_file(path, &sc);
    println!("wrote {}: {} poses, {} planes, {} odom, {} obs",
        path, sc.raw.poses.len(), sc.raw.planes.len(), sc.raw.odos.len(), sc.raw.obs.len());
}

fn parse_pose(f: &[f64]) -> Pose {
    Pose {
        t: V3::new(f[0], f[1], f[2]),
        q: Q::new(f[3], V3::new(f[4], f[5], f[6])).unit(),
    }
}

fn parse_solution(path: &str, n_poses: usize) -> (Vec<Pose>, Vec<Plane>) {
    let text = std::fs::read_to_string(path).unwrap();
    let lines: Vec<Vec<f64>> = text.lines()
        .map(|l| l.split_whitespace().map(|x| x.parse().unwrap()).collect())
        .collect();
    let poses = lines[..n_poses].iter().map(|v| parse_pose(v)).collect();
    let planes = lines[n_poses..].iter()
        .map(|v| Plane::normalized(V3::new(v[0], v[1], v[2]), v[3]))
        .collect();
    (poses, planes)
}

fn gt_errors(gt_poses: &[Pose], gt_planes: &[Plane], poses: &[Pose], planes: &[Plane]) -> String {
    let mut ate = 0.0;
    for (p, g) in poses.iter().zip(gt_poses) {
        let d = p.t - g.t;
        ate += d * d;
    }
    let ate = (ate / gt_poses.len() as f64).sqrt();
    let mut worst_ang = 0.0f64;
    let mut worst_d = 0.0f64;
    for (p, g) in planes.iter().zip(gt_planes) {
        let (a, d) = p.error_vs(*g);
        worst_ang = worst_ang.max(a);
        worst_d = worst_d.max(d);
    }
    format!("pose ATE {:.4} m; plane worst angle {:.4} deg, worst d {:.4} m",
        ate, worst_ang.to_degrees(), worst_d)
}

fn check(scene: &str, sol: &str) {
    let text = std::fs::read_to_string(scene).unwrap();
    let mut gt_poses = Vec::new();
    let mut gt_planes = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("gtpose") => {
                let v: Vec<f64> = f[1..].iter().map(|x| x.parse().unwrap()).collect();
                gt_poses.push(parse_pose(&v));
            }
            Some("gtplane") => {
                let v: Vec<f64> = f[1..].iter().map(|x| x.parse().unwrap()).collect();
                gt_planes.push(Plane::normalized(V3::new(v[0], v[1], v[2]), v[3]));
            }
            _ => {}
        }
    }
    let (poses, planes) = parse_solution(sol, gt_poses.len());
    println!("{}", gt_errors(&gt_poses, &gt_planes, &poses, &planes));
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        None => bench(),
        Some("solve") => solve(&args[2], args.get(3).map(|s| s.as_str())),
        Some("check") => check(&args[2], &args[3]),
        _ => eprintln!("usage: plane-bench [solve <scene> [sol] | check <scene> <sol>]  (no args: run the benchmark; PLANE_POSES=N sizes the scene)"),
    }
}

// ---------------------------------------------------------------------------
// The arael runner. The S^2 plane-normal parameterization is a USER-DEFINED
// component: declared here with #[arael(component)], lifecycle via the
// Component trait, chart cached with compute=, embed via chained symbolic=
// fields -- the full macro path, nothing arael-internal.
// ---------------------------------------------------------------------------

use arael::model::{Component, CrossBlock, Param, QuaternionParam, SelfBlock};
use arael::refs::Ref;
use arael::simple_lm::{LmConfig, LmProblem};
use arael::matrix::matrix3d;
use arael::quatern::quaternd;
use arael::vect::{vect2, vect2d, vect3d};


/// Unit direction on S^2: reference quaternion chart (x-axis = direction),
/// 2-DOF body-frame delta about the frame's y/z. The embed is the rotated
/// first column of the small-rotation matrix of (1, (0, d.x, d.y)/2)
/// normalized -- exact on the sphere for every trial delta.
#[arael::model]
#[arael(component)]
struct UnitVec {
    // Concrete type names throughout: the macro classifies fields by the
    // type's last path segment, so aliases would read as opaque structs.
    ref_q: quaternd,
    #[arael(compute = self.ref_q.rotation_matrix())]
    rot: matrix3d,
    d: Param<vect2d>,
    #[arael(symbolic = 1.0 + (d.x * d.x + d.y * d.y) * 0.25)]
    s2: f64,
    #[arael(symbolic = vect3sym::from_components(
        1.0 - (d.x * d.x + d.y * d.y) / (2.0 * s2), d.y / s2, 0.0 - d.x / s2))]
    local: vect3d,
    #[arael(symbolic = rot * local)]
    unit: vect3d,
}

impl UnitVec {
    fn ex() -> V3 { V3::new(1.0, 0.0, 0.0) }
    fn new(dir: V3) -> UnitVec {
        let mut u = UnitVec {
            ref_q: Q::identity(),
            rot: matrix3d::identity(),
            d: Param::new(vect2::new(0.0, 0.0)),
            s2: 0.0,
            local: V3::new(0.0, 0.0, 0.0),
            unit: dir,
        };
        Component::start(&mut u);
        u
    }
}

impl Component for UnitVec {
    fn start(&mut self) {
        self.unit = self.unit.unit();
        self.ref_q = Q::from_two_vectors(Self::ex(), self.unit);
        self.d.value = vect2::new(0.0, 0.0);
    }
    fn update(&mut self) {
        let dq = Q::from_rotation_vector_small(V3::new(0.0, self.d.value.x, self.d.value.y));
        self.ref_q = (self.ref_q * dq).unit();
        self.d.value = vect2::new(0.0, 0.0);
    }
    fn finish(&mut self) {
        let dq = Q::from_rotation_vector_small(V3::new(0.0, self.d.value.x, self.d.value.y));
        self.unit = (self.ref_q * dq).rotate(Self::ex());
    }
}

#[arael::model]
struct PoseV {
    pos: Param<vect3d>,
    q: QuaternionParam<f64>,
    hb: SelfBlock<PoseV>,
}

#[arael::model]
struct PlaneLm {
    n: UnitVec,
    c: Param<f64>,
    hb: SelfBlock<PlaneLm>,
}

// Odometry between-residual, identical to the g2o runner's custom edge:
//   err_t = R_a^T (t_b - t_a) - t_m;  err_r = vee((R_m^T R_a^T R_b - .^T)/2)
#[arael::model]
#[arael(constraint(hb, {
    let ra = a.q.rotation_matrix();
    let rb = b.q.rotation_matrix();
    let dt = ra.transpose() * (b.pos - a.pos) - odov.tm;
    let dr = odov.rm_t * (ra.transpose() * rb);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x * odov.wt, dt.y * odov.wt, dt.z * odov.wt,
     (c2.z - c3.y) * 0.5 * odov.wr,
     (c3.x - c1.z) * 0.5 * odov.wr,
     (c1.y - c2.x) * 0.5 * odov.wr]
}, parent = odov))]
struct Odov {
    #[arael(ref = root.poses)]
    a: Ref<PoseV>,
    #[arael(ref = root.poses)]
    b: Ref<PoseV>,
    tm: vect3d,
    rm_t: matrix3d,
    wt: f64,
    wr: f64,
    hb: CrossBlock<PoseV, PoseV>,
}

// Plane observation: g2o's EdgeSE3PlaneSensorCalib error (Plane3D::ominus),
// written algebraically. Predicted local plane (n_l, c_l) from the world
// plane through the pose; error = (azimuth, elevation) of the measured
// normal in the frame aligning n_l with e1, plus the distance difference.
#[arael::model]
#[arael(constraint(hb, {
    let rp = p.q.rotation_matrix();
    let nw = l.n.unit;
    let nl = rp.transpose() * nw;
    let cl = l.c + p.pos * nw;
    let h = sqrt(nl.x * nl.x + nl.y * nl.y);
    let mx = nl * obsv.nm;
    let my = (obsv.nm.y * nl.x - obsv.nm.x * nl.y) / h;
    let mz = (obsv.nm.z * (nl.x * nl.x + nl.y * nl.y)
        - nl.z * (nl.x * obsv.nm.x + nl.y * obsv.nm.y)) / h;
    [atan2(my, mx) * obsv.waz,
     atan2(mz, sqrt(mx * mx + my * my)) * obsv.wel,
     (obsv.cm - cl) * obsv.wd]
}, parent = obsv))]
struct Obsv {
    #[arael(ref = root.poses)]
    p: Ref<PoseV>,
    #[arael(ref = root.planes)]
    l: Ref<PlaneLm>,
    nm: vect3d,
    cm: f64,
    waz: f64,
    wel: f64,
    wd: f64,
    hb: CrossBlock<PoseV, PlaneLm>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: arael::refs::Vec<PoseV>,
    planes: arael::refs::Vec<PlaneLm>,
    odos: std::vec::Vec<Odov>,
    obs: std::vec::Vec<Obsv>,
}

fn solve(scene_path: &str, sol_path: Option<&str>) {
    let raw = load_raw(scene_path);
    let (s0, e0, it, acc, ms, _, _) = solve_impl(&raw, 100, sol_path);
    println!("arael: cost {:.6} -> {:.6} in {}({}) iterations, {:.1} ms", s0, e0, it, acc, ms);
}

fn solve_impl(raw: &RawScene, max_iters: usize, sol_path: Option<&str>)
    -> (f64, f64, usize, usize, f64, Vec<Pose>, Vec<Plane>) {
    let mut world = World {
        poses: arael::refs::Vec::new(),
        planes: arael::refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    for (k, p) in raw.poses.iter().enumerate() {
        let fixed = k == 0;
        world.poses.push(PoseV {
            pos: if fixed { Param::fixed(p.t) } else { Param::new(p.t) },
            q: if fixed { QuaternionParam::fixed(p.q) } else { QuaternionParam::new(p.q) },
            hb: SelfBlock::new(),
        });
    }
    for pl in &raw.planes {
        world.planes.push(PlaneLm {
            n: UnitVec::new(pl.n),
            c: Param::new(pl.c),
            hb: SelfBlock::new(),
        });
    }
    for &(i, j, ref rel, wt, wr) in &raw.odos {
        world.odos.push(Odov {
            a: world.poses.ref_at(i as u32),
            b: world.poses.ref_at(j as u32),
            tm: rel.t,
            rm_t: rel.q.rotation_matrix().transpose(),
            wt, wr,
            hb: CrossBlock::new(),
        });
    }
    for &(p, l, ref pl, waz, wel, wd) in &raw.obs {
        world.obs.push(Obsv {
            p: world.poses.ref_at(p as u32),
            l: world.planes.ref_at(l as u32),
            nm: pl.n,
            cm: pl.c,
            waz, wel, wd,
            hb: CrossBlock::new(),
        });
    }

    let cfg = LmConfig::conservative()
        .with_max_iters(max_iters)
        .with_verbose(std::env::var("VERBOSE").is_ok());
    let t0 = std::time::Instant::now();
    let r = world.solve_sparse(&cfg);
    let ms = t0.elapsed().as_secs_f64() * 1e3;

    let poses: Vec<Pose> = world.poses.iter()
        .map(|p| Pose { q: p.q.value, t: p.pos.value }).collect();
    let planes: Vec<Plane> = world.planes.iter()
        .map(|pl| Plane { n: pl.n.unit, c: pl.c.value }).collect();
    if let Some(out) = sol_path {
        write_solution(out, &poses, &planes);
    }
    (r.start_cost, r.end_cost, r.iterations, r.accepted_iterations, ms, poses, planes)
}

// Raw scene for the factrs runner and the shared cost evaluator.
pub struct RawScene {
    pub poses: Vec<Pose>,
    pub planes: Vec<Plane>,
    pub odos: Vec<(usize, usize, Pose, f64, f64)>,          // i, j, rel, wt, wr
    pub obs: Vec<(usize, usize, Plane, f64, f64, f64)>,     // p, l, local plane, waz, wel, wd
}

fn load_raw(path: &str) -> RawScene {
    let text = std::fs::read_to_string(path).unwrap();
    let mut s = RawScene { poses: vec![], planes: vec![], odos: vec![], obs: vec![] };
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        let v = |k: usize| -> f64 { f[k].parse().unwrap() };
        match f.first().copied() {
            Some("pose") => s.poses.push(parse_pose(&f[1..].iter().map(|x| x.parse().unwrap()).collect::<Vec<f64>>())),
            Some("plane") => s.planes.push(Plane::normalized(V3::new(v(1), v(2), v(3)), v(4))),
            Some("odom") => {
                let rel = Pose { t: V3::new(v(3), v(4), v(5)), q: Q::new(v(6), V3::new(v(7), v(8), v(9))).unit() };
                s.odos.push((f[1].parse().unwrap(), f[2].parse().unwrap(), rel, v(10).sqrt(), v(13).sqrt()));
            }
            Some("obs") => {
                let pl = Plane::normalized(V3::new(v(3), v(4), v(5)), v(6));
                s.obs.push((f[1].parse().unwrap(), f[2].parse().unwrap(), pl, v(7).sqrt(), v(8).sqrt(), v(9).sqrt()));
            }
            _ => {}
        }
    }
    s
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

fn write_solution(path: &str, poses: &[Pose], planes: &[Plane]) {
    let mut s = String::new();
    for p in poses {
        s.push_str(&format!("{} {} {} {} {} {} {}\n", p.t.x, p.t.y, p.t.z, p.q.t, p.q.v.x, p.q.v.y, p.q.v.z));
    }
    for pl in planes {
        s.push_str(&format!("{} {} {} {}\n", pl.n.x, pl.n.y, pl.n.z, pl.c));
    }
    std::fs::write(path, s).unwrap();
}

/// Interleaved 1-iter / 2-iter probes, min over PLANE_PROBE_ROUNDS
/// (default 5): the minima are taken before differencing, and interleaving
/// keeps slow system drift from biasing one side of the pair.
fn probe<F1: FnMut() -> f64, F2: FnMut() -> f64>(mut f1: F1, mut f2: F2) -> (f64, f64) {
    let rounds: usize = std::env::var("PLANE_PROBE_ROUNDS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(5);
    let (mut t1, mut t2) = (f64::INFINITY, f64::INFINITY);
    // Discarded warmup, like the sibling benchmarks' harness: the first
    // solve pays cold caches and page faults for everyone after it.
    let _ = f1();
    for _ in 0..rounds {
        t1 = t1.min(f1());
        t2 = t2.min(f2());
    }
    (t1, t2)
}

fn bench() {
    let sc = make_scene();
    let raw = &sc.raw;
    println!("scene: {} poses, {} planes, {} odom, {} obs; start cost {:.6}",
        raw.poses.len(), raw.planes.len(), raw.odos.len(), raw.obs.len(),
        scene_cost(raw, &raw.poses, &raw.planes));
    let scene_file = "/tmp/plane_scene.txt";
    write_scene_file(scene_file, &sc);

    struct RowOut {
        label: &'static str,
        total: f64, t1: f64, t2: f64,
        attempts: usize, accepted: usize,
        cost: f64,
        err: String,
    }
    let mut rows: Vec<RowOut> = Vec::new();

    // arael
    {
        let (t1, t2) = probe(|| solve_impl(raw, 1, None).4, || solve_impl(raw, 2, None).4);
        let (_, e, it, acc, ms, p, pl) = solve_impl(raw, 200, None);
        rows.push(RowOut { label: "arael LM f64", total: ms, t1, t2,
            attempts: it, accepted: acc, cost: e,
            err: gt_errors(&sc.gt_poses, &sc.gt_planes, &p, &pl) });
    }
    // g2o + ceres, external
    for (label, cmd, name) in [("g2o LM", "./cpp/build/g2o_plane", "g2o"),
                               ("ceres LM", "./cpp/build/ceres_plane", "ceres")] {
        let run = |iters: usize, sol: &str| -> (f64, f64, usize, usize) {
            let out = std::process::Command::new(cmd)
                .args([scene_file, sol, &iters.to_string()]).output().expect(name);
            let text = String::from_utf8_lossy(&out.stdout);
            if let Some(l) = text.lines().find(|l| l.contains("chi2")) {
                let f: Vec<&str> = l.split_whitespace().collect();
                (f[7].parse().unwrap(), f[3].parse().unwrap(),
                 f[5].parse().unwrap(), f[5].parse().unwrap())
            } else {
                let l = text.lines().find(|l| l.contains("solve_ms")).expect("json");
                let grab = |k: &str| -> f64 {
                    let i = l.find(k).unwrap() + k.len() + 2;
                    l[i..].chars().take_while(|c| !",}".contains(*c))
                        .collect::<String>().trim().parse().unwrap()
                };
                (grab("solve_ms"), grab("end_cost"),
                 grab("iterations") as usize, grab("accepted") as usize)
            }
        };
        let sol = format!("/tmp/plane_sol_{}.txt", name);
        let (t1, t2) = probe(|| run(1, "/dev/null").0, || run(2, "/dev/null").0);
        let (ms, cost, it, acc) = run(200, &sol);
        let (p, pl) = parse_solution(&sol, raw.poses.len());
        rows.push(RowOut { label, total: ms, t1, t2, attempts: it, accepted: acc,
            cost, err: gt_errors(&sc.gt_poses, &sc.gt_planes, &p, &pl) });
    }
    // factrs
    {
        let (t1, t2) = probe(
            || factrs_runner::solve_factrs(&raw.poses, &raw.planes, &raw.odos, &raw.obs, 1).ms,
            || factrs_runner::solve_factrs(&raw.poses, &raw.planes, &raw.odos, &raw.obs, 2).ms);
        let r = factrs_runner::solve_factrs(&raw.poses, &raw.planes, &raw.odos, &raw.obs, 200);
        let cost = scene_cost(raw, &r.poses, &r.planes);
        rows.push(RowOut { label: "factrs LM", total: r.ms, t1, t2,
            attempts: r.attempts, accepted: r.accepted, cost,
            err: gt_errors(&sc.gt_poses, &sc.gt_planes, &r.poses, &r.planes) });
    }

    println!("\n{:<14} {:>10} {:>9} {:>10} {:>10} {:>12} {:>14}",
        "system", "total ms", "iters", "ms/iter", "full-iter", "1st-iter ms", "final cost");
    for r in &rows {
        let full = if r.t2 > r.t1 { format!("{:.2}", r.t2 - r.t1) } else { "-".to_string() };
        let iters = format!("{}({})", r.accepted, r.attempts);
        println!("{:<14} {:>10.1} {:>9} {:>10.2} {:>10} {:>12.2} {:>14.4}",
            r.label, r.total, iters,
            r.total / r.attempts.max(1) as f64, full, r.t1, r.cost);
    }
    println!("\nsolution vs ground truth:");
    for r in &rows { println!("{:<14} {}", r.label, r.err); }
}
