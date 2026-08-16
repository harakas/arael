// The block supernodal route against faer's scalar one, one damped step at
// a time. Same linearization, same right-hand side, same lambda: the two
// factorizations must return the same step and agree on positive
// definiteness -- a check the converged-optimum tests cannot make, since
// LM corrects a wrong-but-descent step on its own.
//
// Three systems, one per route the unit fixtures do not resemble: the M3500
// pose graph (whole Hessian under block AMD, loop closures), a nonlinear
// range-bearing scene (reduced Schur system, and the whole Hessian with the
// landmarks ordered first), and BAL problem-49 (9-wide cameras, reduced
// system under nested dissection).

use arael::matrix::matrix3d;
use arael::model::{CrossBlock, EulerAngleParam, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{
    BlockSupernodalMode, EnvelopeMode, LmProblem, LmSolver, RootProblem, SchurPolicy,
    SparseFaer, SparseFaerF32,
};
use arael::utils::Float;
use arael::vect::{vect2d, vect3d};

// ---------------------------------------------------------------------------
// M3500: 2D pose graph, as examples/m3500_demo.rs reads it.
// ---------------------------------------------------------------------------

#[arael::model]
struct Pose2 {
    pos: Param<vect2d>,
    th: Param<f64>,
    hb: SelfBlock<Pose2>,
}

#[arael::model]
#[arael(constraint(p.hb, {
    [p.pos.x - prior.pos.x,
     p.pos.y - prior.pos.y,
     p.th - prior.th]
}))]
struct Prior {
    #[arael(ref = root.poses)]
    p: Ref<Pose2>,
    pos: vect2d,
    th: f64,
}

#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(b.th).transpose()
        * (a.pos + matrix2sym::rotation(a.th) * edge.delta - b.pos);
    [local.x, local.y, rad_diff(a.th + edge.dth, b.th)]
}))]
struct Edge {
    #[arael(ref = root.poses)]
    a: Ref<Pose2>,
    #[arael(ref = root.poses)]
    b: Ref<Pose2>,
    delta: vect2d,
    dth: f64,
    hb: CrossBlock<Pose2, Pose2>,
}

#[arael::model]
#[arael(root)]
struct Graph {
    poses: refs::Vec<Pose2>,
    edges: std::vec::Vec<Edge>,
    prior: Option<Prior>,
}

fn load_m3500() -> Graph {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benchmarks/pgo/datasets/input_M3500_g2o.g2o");
    let ds = arael::g2o::Dataset2::load(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut g = Graph { poses: refs::Vec::new(), edges: std::vec::Vec::new(), prior: None };
    for p in &ds.poses {
        let r = g.poses.push(Pose2 { pos: Param::new(p.t), th: Param::new(p.th), hb: SelfBlock::new() });
        if g.prior.is_none() {
            g.prior = Some(Prior { p: r, pos: p.t, th: p.th });
        }
    }
    for d in &ds.deltas {
        g.edges.push(Edge {
            a: g.poses.ref_at(d.a),
            b: g.poses.ref_at(d.b),
            delta: d.dt,
            dth: d.dth,
            hb: CrossBlock::new(),
        });
    }
    g
}

// ---------------------------------------------------------------------------
// A nonlinear 2D range-bearing scene: poses on a line, one landmark per
// window of `win` poses. Generic so the f32 twin is the same model.
// ---------------------------------------------------------------------------

#[arael::model]
#[arael(constraint(hb, {
    [(pose.x - pose.ax) * 0.01, (pose.y - pose.ay) * 0.01]
}))]
struct Pose<T: Float> {
    x: Param<T>,
    y: Param<T>,
    ax: T,
    ay: T,
    hb: SelfBlock<Pose<T>, T>,
}

#[arael::model]
struct Landmark<T: Float> {
    x: Param<T>,
    y: Param<T>,
    hb: SelfBlock<Landmark<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, {
    [b.x - a.x - odo.dx, b.y - a.y - odo.dy]
}))]
struct Odo<T: Float> {
    #[arael(ref = root.poses)]
    a: Ref<Pose<T>>,
    #[arael(ref = root.poses)]
    b: Ref<Pose<T>>,
    dx: T,
    dy: T,
    hb: CrossBlock<Pose<T>, Pose<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, {
    let dx = l.x - p.x;
    let dy = l.y - p.y;
    [sqrt(dx * dx + dy * dy) - rangebearing.r, atan2(dy, dx) - rangebearing.b]
}))]
struct RangeBearing<T: Float> {
    #[arael(ref = root.poses)]
    p: Ref<Pose<T>>,
    #[arael(ref = root.landmarks)]
    l: Ref<Landmark<T>>,
    r: T,
    b: T,
    hb: CrossBlock<Pose<T>, Landmark<T>, T>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose<f64>>,
    landmarks: refs::Vec<Landmark<f64>>,
    odos: std::vec::Vec<Odo<f64>>,
    obs: std::vec::Vec<RangeBearing<f64>>,
}

#[arael::model]
#[arael(root, f32)]
struct WorldF {
    poses: refs::Vec<Pose<f32>>,
    landmarks: refs::Vec<Landmark<f32>>,
    odos: std::vec::Vec<Odo<f32>>,
    obs: std::vec::Vec<RangeBearing<f32>>,
}

/// The scene as plain numbers: (x, y, true x, true y) per pose, landmark
/// positions, (pose, landmark, range, bearing) observations, (a, b, dx, dy)
/// odometry. Poses start 0.1 off, landmarks 0.1 off the other way.
struct Scene {
    poses: Vec<(f64, f64, f64, f64)>,
    landmarks: Vec<(f64, f64)>,
    obs: Vec<(usize, usize, f64, f64)>,
    odos: Vec<(usize, usize, f64, f64)>,
}

fn scene(np: usize, win: usize) -> Scene {
    let pt = |i: usize| (i as f64, 0.3 * i as f64);
    let mut s = Scene { poses: Vec::new(), landmarks: Vec::new(), obs: Vec::new(), odos: Vec::new() };
    for i in 0..np {
        let (tx, ty) = pt(i);
        s.poses.push((tx + 0.1, ty - 0.1, tx, ty));
    }
    for w0 in 0..np.saturating_sub(win - 1) {
        let (lx, ly) = (w0 as f64 + 0.5, 2.0 + 0.2 * w0 as f64);
        let lm = s.landmarks.len();
        s.landmarks.push((lx - 0.1, ly + 0.1));
        for p in w0..w0 + win {
            let (px, py) = pt(p);
            let (dx, dy) = (lx - px, ly - py);
            s.obs.push((p, lm, (dx * dx + dy * dy).sqrt(), dy.atan2(dx)));
        }
    }
    for i in 1..np {
        let (ax, ay) = pt(i - 1);
        let (bx, by) = pt(i);
        s.odos.push((i - 1, i, bx - ax, by - ay));
    }
    s
}

fn world(np: usize, win: usize) -> World {
    let s = scene(np, win);
    let mut w = World { poses: refs::Vec::new(), landmarks: refs::Vec::new(), odos: Vec::new(), obs: Vec::new() };
    for &(x, y, ax, ay) in &s.poses {
        w.poses.push(Pose { x: Param::new(x), y: Param::new(y), ax, ay, hb: SelfBlock::new() });
    }
    for &(x, y) in &s.landmarks {
        w.landmarks.push(Landmark { x: Param::new(x), y: Param::new(y), hb: SelfBlock::new() });
    }
    for &(p, l, r, b) in &s.obs {
        w.obs.push(RangeBearing {
            p: w.poses.ref_at(p),
            l: w.landmarks.ref_at(l as u32),
            r,
            b,
            hb: CrossBlock::new(),
        });
    }
    for &(a, b, dx, dy) in &s.odos {
        w.odos.push(Odo { a: w.poses.ref_at(a), b: w.poses.ref_at(b), dx, dy, hb: CrossBlock::new() });
    }
    w
}

fn world_f32(np: usize, win: usize) -> WorldF {
    let s = scene(np, win);
    let mut w = WorldF { poses: refs::Vec::new(), landmarks: refs::Vec::new(), odos: Vec::new(), obs: Vec::new() };
    for &(x, y, ax, ay) in &s.poses {
        w.poses.push(Pose {
            x: Param::new(x as f32),
            y: Param::new(y as f32),
            ax: ax as f32,
            ay: ay as f32,
            hb: SelfBlock::new(),
        });
    }
    for &(x, y) in &s.landmarks {
        w.landmarks.push(Landmark { x: Param::new(x as f32), y: Param::new(y as f32), hb: SelfBlock::new() });
    }
    for &(p, l, r, b) in &s.obs {
        w.obs.push(RangeBearing {
            p: w.poses.ref_at(p),
            l: w.landmarks.ref_at(l as u32),
            r: r as f32,
            b: b as f32,
            hb: CrossBlock::new(),
        });
    }
    for &(a, b, dx, dy) in &s.odos {
        w.odos.push(Odo {
            a: w.poses.ref_at(a),
            b: w.poses.ref_at(b),
            dx: dx as f32,
            dy: dy as f32,
            hb: CrossBlock::new(),
        });
    }
    w
}

// ---------------------------------------------------------------------------
// BAL: the Snavely reprojection model, as examples/bal_demo.rs reads it.
// ---------------------------------------------------------------------------

#[arael::model]
struct Camera {
    t: Param<vect3d>,
    ea: EulerAngleParam<f64>,
    intr: Param<vect3d>,
    hb: SelfBlock<Camera>,
}

#[arael::model]
struct Point {
    pos: Param<vect3d>,
    hb: SelfBlock<Point>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [cam.intr.x * d * px - reprojection.xy.x,
     cam.intr.x * d * py - reprojection.xy.y]
}))]
struct Reprojection {
    #[arael(ref = root.cameras)]
    cam: Ref<Camera>,
    #[arael(ref = root.points)]
    pt: Ref<Point>,
    xy: vect2d,
    hb: CrossBlock<Camera, Point>,
}

#[arael::model]
#[arael(root)]
struct Bundle {
    cameras: refs::Vec<Camera>,
    points: refs::Vec<Point>,
    observations: std::vec::Vec<Reprojection>,
}

fn rodrigues_to_matrix(w: vect3d) -> matrix3d {
    let t2 = w.square();
    if t2 > 1e-24 {
        let theta = t2.sqrt();
        matrix3d::rotation_from_axis_angle(w * (1.0 / theta), theta)
    } else {
        matrix3d::from_elements(1.0, -w.z, w.y, w.z, 1.0, -w.x, -w.y, w.x, 1.0)
    }
}

/// Problem-49. With `window` set, an observation is kept only when its
/// camera is within that many cameras of the first one seeing the point,
/// and points left with fewer than two observations are dropped: the
/// reduced system then has a band instead of one dense panel, so its
/// factorization has several supernodes and real updates between them.
fn load_bal(window: Option<usize>) -> Bundle {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/benchmarks/bal/datasets/problem-49-7776-pre.txt");
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let mut it = text.split_ascii_whitespace().map(|t| t.parse::<f64>().unwrap());
    let mut next = || it.next().expect("truncated BAL file");
    let (n_cams, n_points, n_obs) = (next() as usize, next() as usize, next() as usize);
    let mut obs_raw = Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        obs_raw.push((next() as usize, next() as usize, vect2d::new(next(), next())));
    }
    let mut cameras = Vec::with_capacity(n_cams);
    for _ in 0..n_cams {
        let rodrigues = vect3d::new(next(), next(), next());
        let t = vect3d::new(next(), next(), next());
        let intr = vect3d::new(next(), next(), next());
        cameras.push((rodrigues, t, intr));
    }
    let points: Vec<vect3d> = (0..n_points).map(|_| vect3d::new(next(), next(), next())).collect();

    if let Some(w) = window {
        let mut first = vec![usize::MAX; n_points];
        for &(c, p, _) in &obs_raw {
            first[p] = first[p].min(c);
        }
        obs_raw.retain(|&(c, p, _)| c - first[p] <= w);
    }
    let mut seen = vec![0usize; n_points];
    for &(_, p, _) in &obs_raw {
        seen[p] += 1;
    }
    let mut new_index = vec![usize::MAX; n_points];
    let mut b = Bundle { cameras: refs::Vec::new(), points: refs::Vec::new(), observations: Vec::new() };
    for (rodrigues, t, intr) in cameras {
        b.cameras.push(Camera {
            t: Param::new(t),
            ea: EulerAngleParam::new(rodrigues_to_matrix(rodrigues).get_euler_angles()),
            intr: Param::new(intr),
            hb: SelfBlock::new(),
        });
    }
    for (p, pos) in points.into_iter().enumerate() {
        if seen[p] >= 2 {
            new_index[p] = b.points.len();
            b.points.push(Point { pos: Param::new(pos), hb: SelfBlock::new() });
        }
    }
    for (c, p, xy) in obs_raw {
        if new_index[p] == usize::MAX {
            continue;
        }
        let cam = b.cameras.ref_at(c);
        let pt = b.points.ref_at(new_index[p]);
        b.observations.push(Reprojection { cam, pt, xy, hb: CrossBlock::new() });
    }
    b
}

// ---------------------------------------------------------------------------
// The step comparison.
// ---------------------------------------------------------------------------

/// One linearization at `params`, then a damped step per lambda: `None`
/// where the factorization rejected the system.
fn steps<T: Float>(
    solver: &mut SparseFaer<T>,
    problem: &mut dyn LmProblem<T>,
    params: &[T],
    lambdas: &[T],
) -> Vec<Option<Vec<T>>>
where
    SparseFaer<T>: LmSolver<T>,
{
    let n = params.len();
    let mut m = solver.new_matrix(n);
    let mut grad = vec![T::zero(); n];
    solver.compute(problem, params, &mut grad, &mut m).expect("compute");
    let mut diag = vec![T::zero(); n];
    solver.extract_diagonal(&m, &mut diag);
    lambdas
        .iter()
        .map(|&lambda| {
            let mut delta = vec![T::zero(); n];
            solver.solve_damped(n, &mut m, &diag, &diag, lambda, &grad, &mut delta).then_some(delta)
        })
        .collect()
}

/// The two routes' steps agree to `tol` (relative to the largest entry of
/// the scalar route's step) and accept or reject the same lambdas.
fn assert_steps_agree<T: Float>(label: &str, lambdas: &[T], block: &[Option<Vec<T>>], scalar: &[Option<Vec<T>>], tol: f64) {
    for (i, &lambda) in lambdas.iter().enumerate() {
        match (&block[i], &scalar[i]) {
            (Some(b), Some(s)) => {
                let mut max_diff = 0.0f64;
                let mut max_val = 0.0f64;
                for (x, y) in b.iter().zip(s) {
                    max_diff = max_diff.max((x.to_f64().unwrap() - y.to_f64().unwrap()).abs());
                    max_val = max_val.max(y.to_f64().unwrap().abs());
                }
                assert!(max_val > 0.0, "{label} lambda {lambda:?}: a zero step");
                let rel = max_diff / max_val;
                assert!(rel < tol, "{label} lambda {lambda:?}: steps differ by {rel:.3e} (tolerance {tol:.0e})");
            }
            (b, s) => assert_eq!(
                b.is_some(),
                s.is_some(),
                "{label} lambda {lambda:?}: block route accepted {}, scalar route accepted {}",
                b.is_some(),
                s.is_some()
            ),
        }
    }
}

fn whole_h(mode: BlockSupernodalMode) -> SparseFaer<f64> {
    SparseFaer::new().with_policy(SchurPolicy::Never).with_block_supernodal(mode)
}

fn reduced(mode: BlockSupernodalMode) -> SparseFaer<f64> {
    SparseFaer::new()
        .with_policy(SchurPolicy::Force)
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(mode)
}

/// M3500, whole Hessian: 3-wide poses with loop closures under block AMD
/// against faer's scalar AMD. Only the gauge prior holds the system as
/// lambda goes to zero, so the agreement loosens with the conditioning.
#[test]
fn m3500_whole_hessian_steps_agree() {
    let mut g = load_m3500();
    let mut params = Vec::new();
    RootProblem::serialize(&mut g, &mut params);
    let lambdas = [1e-2, 1e-4, 1e-6, 1e-9, 0.0];
    let mut block = whole_h(BlockSupernodalMode::Always);
    let mut scalar = whole_h(BlockSupernodalMode::Never);
    let b = steps(&mut block, &mut g, &params, &lambdas);
    let s = steps(&mut scalar, &mut g, &params, &lambdas);
    assert!(block.plan().unwrap().block_supernodal);
    assert!(!scalar.plan().unwrap().block_supernodal);
    assert_steps_agree("m3500", &lambdas, &b, &s, 1e-8);
}

/// The range-bearing scene, reduced: the landmarks marginalized and the pose
/// system factored by the block supernodal against faer's scalar route.
#[test]
fn reduced_system_steps_agree() {
    let mut w = world(300, 6);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let lambdas = [1e-2, 1e-4, 1e-6, 1e-10, 0.0];
    let mut block = reduced(BlockSupernodalMode::Always);
    let mut scalar = reduced(BlockSupernodalMode::Never);
    let b = steps(&mut block, &mut w, &params, &lambdas);
    let s = steps(&mut scalar, &mut w, &params, &lambdas);
    let plan = block.plan().unwrap();
    assert!(plan.reduced && plan.block_supernodal && !plan.envelope);
    let plan = scalar.plan().unwrap();
    assert!(plan.reduced && !plan.block_supernodal);
    assert_steps_agree("reduced", &lambdas, &b, &s, 1e-10);
}

/// The range-bearing scene, whole Hessian with the landmarks ordered first
/// (the block twin of the scalar route's marginalize-first order): many
/// narrow leaves updating the pose supernodes.
#[test]
fn landmarks_first_whole_hessian_steps_agree() {
    let mut w = world(300, 6);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let lm_start = 2 * w.poses.len();
    let n = params.len();
    let lambdas = [1e-2, 1e-4, 1e-6, 1e-10, 0.0];
    let mut block = whole_h(BlockSupernodalMode::Always).with_marginalize(lm_start..n);
    let mut scalar = whole_h(BlockSupernodalMode::Never).with_marginalize(lm_start..n);
    let b = steps(&mut block, &mut w, &params, &lambdas);
    let s = steps(&mut scalar, &mut w, &params, &lambdas);
    let plan = block.plan().unwrap();
    assert!(!plan.reduced && plan.block_supernodal);
    let plan = scalar.plan().unwrap();
    assert!(!plan.reduced && !plan.block_supernodal);
    assert_steps_agree("landmarks first", &lambdas, &b, &s, 1e-10);
}

/// The f32 twins of the reduced and whole-Hessian scenes, at single
/// precision's agreement.
#[test]
fn f32_steps_agree() {
    let mut w = world_f32(300, 6);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let lambdas = [1e-2f32, 1e-4, 1e-6];

    let mut block = SparseFaerF32::new()
        .with_policy(SchurPolicy::Force)
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always);
    let mut scalar = SparseFaerF32::new()
        .with_policy(SchurPolicy::Force)
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Never);
    let b = steps(&mut block, &mut w, &params, &lambdas);
    let s = steps(&mut scalar, &mut w, &params, &lambdas);
    let plan = block.plan().unwrap();
    assert!(plan.reduced && plan.block_supernodal);
    assert!(!scalar.plan().unwrap().block_supernodal);
    assert_steps_agree("f32 reduced", &lambdas, &b, &s, 1e-3);

    let mut block = SparseFaerF32::new().with_policy(SchurPolicy::Never).with_block_supernodal(BlockSupernodalMode::Always);
    let mut scalar = SparseFaerF32::new().with_policy(SchurPolicy::Never).with_block_supernodal(BlockSupernodalMode::Never);
    let b = steps(&mut block, &mut w, &params, &lambdas);
    let s = steps(&mut scalar, &mut w, &params, &lambdas);
    let plan = block.plan().unwrap();
    assert!(!plan.reduced && plan.block_supernodal);
    assert!(!scalar.plan().unwrap().block_supernodal);
    assert_steps_agree("f32 whole", &lambdas, &b, &s, 1e-3);
}

/// BAL problem-49: the points marginalized, the 49 9-wide cameras' reduced
/// system factored by the block supernodal against faer's scalar route.
/// Gauge-free, so the small lambdas are ill conditioned. The full problem's
/// reduced system is 84% dense -- one panel, so this checks the seed and the
/// dense kernels; the windowed one has a band and several supernodes.
fn bal_steps_agree(label: &str, window: Option<usize>) {
    let mut bundle = load_bal(window);
    let mut params = Vec::new();
    RootProblem::serialize(&mut bundle, &mut params);
    let lambdas = [1e-2, 1e-4, 1e-6, 1e-8];
    let mut block = SparseFaer::new().with_envelope_schur(EnvelopeMode::Never).with_block_supernodal(BlockSupernodalMode::Always);
    let mut scalar = SparseFaer::new().with_envelope_schur(EnvelopeMode::Never).with_block_supernodal(BlockSupernodalMode::Never);
    let b = steps(&mut block, &mut bundle, &params, &lambdas);
    let s = steps(&mut scalar, &mut bundle, &params, &lambdas);
    let plan = block.plan().unwrap();
    assert!(plan.reduced && plan.block_supernodal, "{label}: reduced {} block {}", plan.reduced, plan.block_supernodal);
    let plan = scalar.plan().unwrap();
    assert!(plan.reduced && !plan.block_supernodal);
    assert_steps_agree(label, &lambdas, &b, &s, 1e-8);
}

#[test]
fn bal_reduced_system_steps_agree() {
    bal_steps_agree("bal-49", None);
}

#[test]
fn bal_windowed_reduced_system_steps_agree() {
    bal_steps_agree("bal-49 windowed", Some(6));
}
