// ScaleTransformParam: a user-defined #[arael(component)] bundling a
// TransformParam with a log-scale parameter -- the Sim(3) state for
// monocular loop closing (point action s*(R*x) + t, 7 dof).
//
// Diligence for the lightly-used component API, following the
// transform_param.rs precedent: every test has an option-B twin (a
// plain entity carrying TransformParam + Param<f64> log_s with exp()
// inline in the constraint bodies) and asserts the component
// formulation matches it digit for digit -- same costs, same optimum,
// same iteration count -- plus action correctness, per-part freezing
// (the fix_scale mode), and a serialize round trip. The ignored
// perf_report test times A vs B in release; a broken symbolic cache in
// the component would drag A behind B.

use arael::matrix::matrix3d;
use arael::model::{Component, CrossBlock, Param, SelfBlock};
use arael::quatern::quaternd;
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem, RootProblem};
use arael::transform::{ScaledTransformParam, TransformParam};
use arael::utils::Float;
use arael::vect::vect3d;

// ------------------------------------------------------------ component

/// A similarity transform: pose plus scale, acting as s*(R*x) + t.
/// The scale is optimized as its logarithm (positivity is structural);
/// `scale_factor` = exp(log_s) is what constraint bodies read.
#[arael::model]
#[arael(component)]
#[derive(Clone, Default)]
pub struct ScaleTransformParam<T: Float = f64> {
    pub pose: TransformParam<T>,
    pub log_s: Param<T>,
    #[arael(symbolic = exp(log_s))]
    pub scale_factor: T,
}

impl ScaleTransformParam {
    pub fn new(translation: vect3d, rotation: quaternd, s: f64) -> ScaleTransformParam {
        let mut st = ScaleTransformParam {
            pose: TransformParam::new(translation, rotation),
            log_s: Param::new(s.ln()),
            scale_factor: s,
        };
        Component::start(&mut st);
        st
    }
    pub fn fixed(translation: vect3d, rotation: quaternd, s: f64) -> ScaleTransformParam {
        let mut st = ScaleTransformParam {
            pose: TransformParam::fixed(translation, rotation),
            log_s: Param::fixed(s.ln()),
            scale_factor: s,
        };
        Component::start(&mut st);
        st
    }
    pub fn scale(&self) -> f64 {
        self.log_s.value.exp()
    }
}

impl<T: Float> Component for ScaleTransformParam<T> {
    fn start(&mut self) {
        self.pose.start();
        self.scale_factor = self.log_s.value.exp();
    }
    fn update(&mut self) {
        self.pose.update();
        self.scale_factor = self.log_s.value.exp();
    }
    fn finish(&mut self) {
        self.pose.finish();
        self.scale_factor = self.log_s.value.exp();
    }
}

// ------------------------------------------------- model A: the component

/// A pose with an optional triple of scaled-point observations
/// (world point xi measured as mi = s*(R*xi) + t): 9 rows against the
/// 7 dof, so a lone pose is over-determined.
#[arael::model]
#[arael(constraint(hb, name = "obs", guard = self.has_obs, {
    let r = simpose.st.pose.rotation_matrix;
    let t = simpose.st.pose.translation;
    let s = simpose.st.scale_factor;
    let p1 = r * simpose.x1 * s + t;
    let p2 = r * simpose.x2 * s + t;
    let p3 = r * simpose.x3 * s + t;
    [p1.x - simpose.m1.x, p1.y - simpose.m1.y, p1.z - simpose.m1.z,
     p2.x - simpose.m2.x, p2.y - simpose.m2.y, p2.z - simpose.m2.z,
     p3.x - simpose.m3.x, p3.y - simpose.m3.y, p3.z - simpose.m3.z]
}))]
#[derive(Default)]
pub struct SimPose {
    pub st: ScaleTransformParam<f64>,
    pub has_obs: bool,
    pub x1: vect3d,
    pub x2: vect3d,
    pub x3: vect3d,
    pub m1: vect3d,
    pub m2: vect3d,
    pub m3: vect3d,
    pub hb: SelfBlock<SimPose>,
}

/// A relative similarity constraint (the essential-graph edge shape):
/// translation error in frame a, the antisymmetric rotation-error
/// vector, and the log-scale difference, 7 rows.
#[arael::model]
#[arael(constraint(hb, {
    let ra = a.st.pose.rotation_matrix;
    let dt = ra.transpose() * (b.st.pose.translation - a.st.pose.translation)
        - simlink.dt;
    let dr = simlink.rmeas_t * (ra.transpose() * b.st.pose.rotation_matrix);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5,
     b.st.log_s - a.st.log_s - simlink.dls]
}, parent = simlink))]
#[derive(Default)]
pub struct SimLink {
    #[arael(ref = root.poses)]
    pub a: Ref<SimPose>,
    #[arael(ref = root.poses)]
    pub b: Ref<SimPose>,
    pub dt: vect3d,
    pub rmeas_t: matrix3d,
    pub dls: f64,
    pub hb: CrossBlock<SimPose, SimPose>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct WorldA {
    pub poses: refs::Vec<SimPose>,
    pub links: std::vec::Vec<SimLink>,
}

// ------------------------------------- model B: the hand-rolled twin

#[arael::model]
#[arael(constraint(hb, name = "obs", guard = self.has_obs, {
    let r = simposeb.pose.rotation_matrix;
    let t = simposeb.pose.translation;
    let s = exp(simposeb.log_s);
    let p1 = r * simposeb.x1 * s + t;
    let p2 = r * simposeb.x2 * s + t;
    let p3 = r * simposeb.x3 * s + t;
    [p1.x - simposeb.m1.x, p1.y - simposeb.m1.y, p1.z - simposeb.m1.z,
     p2.x - simposeb.m2.x, p2.y - simposeb.m2.y, p2.z - simposeb.m2.z,
     p3.x - simposeb.m3.x, p3.y - simposeb.m3.y, p3.z - simposeb.m3.z]
}))]
#[derive(Default)]
pub struct SimPoseB {
    pub pose: TransformParam<f64>,
    pub log_s: Param<f64>,
    pub has_obs: bool,
    pub x1: vect3d,
    pub x2: vect3d,
    pub x3: vect3d,
    pub m1: vect3d,
    pub m2: vect3d,
    pub m3: vect3d,
    pub hb: SelfBlock<SimPoseB>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.pose.rotation_matrix;
    let dt = ra.transpose() * (b.pose.translation - a.pose.translation)
        - simlinkb.dt;
    let dr = simlinkb.rmeas_t * (ra.transpose() * b.pose.rotation_matrix);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5,
     b.log_s - a.log_s - simlinkb.dls]
}, parent = simlinkb))]
#[derive(Default)]
pub struct SimLinkB {
    #[arael(ref = root.poses)]
    pub a: Ref<SimPoseB>,
    #[arael(ref = root.poses)]
    pub b: Ref<SimPoseB>,
    pub dt: vect3d,
    pub rmeas_t: matrix3d,
    pub dls: f64,
    pub hb: CrossBlock<SimPoseB, SimPoseB>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct WorldB {
    pub poses: refs::Vec<SimPoseB>,
    pub links: std::vec::Vec<SimLinkB>,
}

// -------------------------------------------------------------- helpers

/// Poses on a loop with drifting scale; link measurements from a
/// DIFFERENT trajectory so every residual is nonzero and the optimum is
/// a genuine compromise (the transform_param.rs trick), including a
/// closure edge.
fn scene() -> (Vec<(vect3d, quaternd, f64)>, Vec<(u32, u32, vect3d, matrix3d, f64)>) {
    scene_n(5)
}

fn scene_n(n: usize) -> (Vec<(vect3d, quaternd, f64)>, Vec<(u32, u32, vect3d, matrix3d, f64)>) {
    let mut poses = Vec::new();
    for i in 0..n {
        let th = i as f64 * 0.4;
        poses.push((
            vect3d::new(i as f64 * 1.1, 0.2 * th.sin(), -0.1 * th),
            (quaternd::from_axis_angle(vect3d::new(0.0, 0.0, 1.0), th * 0.9)
                * quaternd::from_axis_angle(vect3d::new(1.0, 0.0, 0.0), 0.15 * th))
            .unit(),
            1.0 + 0.06 * i as f64,
        ));
    }
    let mut links = Vec::new();
    for i in 0..(n as u32 - 1) {
        let rel_t = vect3d::new(1.0, 0.05 * (i as f64), 0.02);
        let rel_q = quaternd::from_axis_angle(
            vect3d::new(0.1, -0.2, 1.0).unit(), 0.35 + 0.03 * i as f64).unit();
        links.push((i, i + 1, rel_t, rel_q.rotation_matrix().transpose(), 0.04));
    }
    let closure_q = quaternd::from_axis_angle(vect3d::new(0.0, 0.1, -1.0).unit(), 1.3).unit();
    links.push((n as u32 - 1, 0, vect3d::new(-3.6, 0.4, 0.1), closure_q.rotation_matrix().transpose(), -0.2));
    (poses, links)
}

fn build_a_n(n: usize) -> WorldA {
    let (poses, links) = scene_n(n);
    let mut w = WorldA::default();
    for (k, (p, q, s)) in poses.iter().enumerate() {
        w.poses.push(SimPose {
            st: if k == 0 {
                ScaleTransformParam::fixed(*p, *q, *s)
            } else {
                ScaleTransformParam::new(*p, *q, *s)
            },
            ..Default::default()
        });
    }
    for (a, b, t, r, dls) in links {
        w.links.push(SimLink {
            a: w.poses.ref_at(a), b: w.poses.ref_at(b),
            dt: t, rmeas_t: r, dls, ..Default::default()
        });
    }
    w
}

fn build_b_n(n: usize) -> WorldB {
    let (poses, links) = scene_n(n);
    let mut w = WorldB::default();
    for (k, (p, q, s)) in poses.iter().enumerate() {
        let mut e = SimPoseB {
            pose: if k == 0 { TransformParam::fixed(*p, *q) } else { TransformParam::new(*p, *q) },
            log_s: if k == 0 { Param::fixed(s.ln()) } else { Param::new(s.ln()) },
            ..Default::default()
        };
        Component::start(&mut e.pose);
        w.poses.push(e);
    }
    for (a, b, t, r, dls) in links {
        w.links.push(SimLinkB {
            a: w.poses.ref_at(a), b: w.poses.ref_at(b),
            dt: t, rmeas_t: r, dls, ..Default::default()
        });
    }
    w
}

fn build_a() -> WorldA {
    let (poses, links) = scene();
    let mut w = WorldA::default();
    for (k, (p, q, s)) in poses.iter().enumerate() {
        w.poses.push(SimPose {
            st: if k == 0 {
                ScaleTransformParam::fixed(*p, *q, *s)
            } else {
                ScaleTransformParam::new(*p, *q, *s)
            },
            ..Default::default()
        });
    }
    for (a, b, t, r, dls) in links {
        w.links.push(SimLink {
            a: w.poses.ref_at(a),
            b: w.poses.ref_at(b),
            dt: t,
            rmeas_t: r,
            dls,
            ..Default::default()
        });
    }
    w
}

fn build_b() -> WorldB {
    let (poses, links) = scene();
    let mut w = WorldB::default();
    for (k, (p, q, s)) in poses.iter().enumerate() {
        let mut e = SimPoseB {
            pose: if k == 0 { TransformParam::fixed(*p, *q) } else { TransformParam::new(*p, *q) },
            log_s: if k == 0 { Param::fixed(s.ln()) } else { Param::new(s.ln()) },
            ..Default::default()
        };
        Component::start(&mut e.pose);
        w.poses.push(e);
    }
    for (a, b, t, r, dls) in links {
        w.links.push(SimLinkB {
            a: w.poses.ref_at(a),
            b: w.poses.ref_at(b),
            dt: t,
            rmeas_t: r,
            dls,
            ..Default::default()
        });
    }
    w
}

/// A single free pose with the 9-row observation constraint; the
/// measurements encode the target similarity (R = 0.4 rad about z,
/// t = (0.3, -0.2, 0.5), s = 1.25).
fn build_obs(s0: f64) -> WorldA {
    let rq = quaternd::from_axis_angle(vect3d::new(0.0, 0.0, 1.0), 0.4);
    let r = rq.rotation_matrix();
    let t = vect3d::new(0.3, -0.2, 0.5);
    let s = 1.25;
    let xs = [
        vect3d::new(1.0, 0.5, 2.0),
        vect3d::new(-0.7, 1.1, 3.0),
        vect3d::new(0.3, -0.9, 1.5),
    ];
    let mut w = WorldA::default();
    let mut e = SimPose {
        st: ScaleTransformParam::new(vect3d::new(0.0, 0.0, 0.0), quaternd::identity(), s0),
        has_obs: true,
        x1: xs[0],
        x2: xs[1],
        x3: xs[2],
        m1: r * xs[0] * s + t,
        m2: r * xs[1] * s + t,
        m3: r * xs[2] * s + t,
        ..Default::default()
    };
    Component::start(&mut e.st);
    w.poses.push(e);
    w
}

// ---------------------------------------------------------------- tests

/// The component's action s*(R*x) + t is right: at the encoding state
/// the observation cost is zero.
#[test]
fn action_matches_hand_calculation() {
    let mut w = build_obs(1.25);
    {
        let e = w.poses.iter_mut().next().unwrap();
        e.st.pose.translation = vect3d::new(0.3, -0.2, 0.5);
        e.st.pose.rotation = quaternd::from_axis_angle(vect3d::new(0.0, 0.0, 1.0), 0.4);
        Component::start(&mut e.st);
    }
    let cfg = LmConfig::well_conditioned();
    let r = w.solve_dense(&cfg).unwrap();
    assert!(r.start_cost < 1e-22, "cost at ground truth: {}", r.start_cost);
}

/// From a wrong scale the observation problem converges to the target
/// similarity.
#[test]
fn obs_solve_recovers_similarity() {
    let mut w = build_obs(0.7);
    let cfg = LmConfig::well_conditioned();
    let r = w.solve_dense(&cfg).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!(r.end_cost < 1e-9, "end cost {}", r.end_cost);
    let e = w.poses.iter().next().unwrap();
    assert!((e.st.scale() - 1.25).abs() < 1e-6, "scale {}", e.st.scale());
    assert!((e.st.pose.translation - vect3d::new(0.3, -0.2, 0.5)).norm() < 1e-6);
}

/// Identical initial and final costs, iteration counts, poses and
/// scales: the component twin against the hand-rolled twin.
#[test]
fn solve_parity_with_twin() {
    let cfg = LmConfig::conservative();
    let mut a = build_a();
    let mut b = build_b();
    let ra = a.solve_dense(&cfg).unwrap();
    let rb = b.solve_dense(&cfg).unwrap();

    assert!(ra.status.is_success(), "A: {:?}", ra.status);
    assert!(rb.status.is_success(), "B: {:?}", rb.status);
    assert!(
        (ra.start_cost - rb.start_cost).abs() < 1e-12 * (1.0 + ra.start_cost),
        "start cost: A {} vs B {}",
        ra.start_cost,
        rb.start_cost
    );
    assert!(ra.end_cost > 1e-6, "measurements disagree by design: {}", ra.end_cost);
    assert!(
        (ra.end_cost - rb.end_cost).abs() < 1e-12 * (1.0 + ra.end_cost),
        "end cost: A {} vs B {}",
        ra.end_cost,
        rb.end_cost
    );
    assert_eq!(ra.iterations, rb.iterations, "same damping trajectory");

    for (i, (ea, eb)) in a.poses.iter().zip(b.poses.iter()).enumerate() {
        assert!(
            (ea.st.pose.translation - eb.pose.translation).norm() < 1e-12,
            "translation[{i}]"
        );
        for r in 0..3 {
            assert!(
                (ea.st.pose.rotation_matrix[r] - eb.pose.rotation_matrix[r]).norm() < 1e-12,
                "rotation_matrix[{i}] row {r}"
            );
        }
        assert!(
            (ea.st.scale() - eb.log_s.value.exp()).abs() < 1e-12,
            "scale[{i}]: {} vs {}",
            ea.st.scale(),
            eb.log_s.value.exp()
        );
    }
}

/// fix_scale: freezing log_s holds the scale while the pose still
/// converges; a wrong frozen scale cannot be repaired.
#[test]
fn frozen_scale_holds() {
    let cfg = LmConfig::well_conditioned();

    let mut w = build_obs(1.25);
    {
        let e = w.poses.iter_mut().next().unwrap();
        e.st.log_s.optimize = false;
    }
    let r = w.solve_dense(&cfg).unwrap();
    assert!(r.end_cost < 1e-9, "end cost {}", r.end_cost);
    let e = w.poses.iter().next().unwrap();
    assert!((e.st.scale() - 1.25).abs() < 1e-15, "scale moved: {}", e.st.scale());

    let mut w2 = build_obs(0.8);
    {
        let e = w2.poses.iter_mut().next().unwrap();
        e.st.log_s.optimize = false;
    }
    let r2 = w2.solve_dense(&cfg).unwrap();
    let e2 = w2.poses.iter().next().unwrap();
    assert!((e2.st.scale() - 0.8).abs() < 1e-15, "frozen scale moved");
    assert!(r2.end_cost > 1e-6, "wrong frozen scale cannot reach zero: {}", r2.end_cost);
}

/// Serialize round trip preserves the component state.
#[test]
fn serialize_round_trip() {
    let cfg = LmConfig::well_conditioned();
    let mut a = build_a();
    let mut data = Vec::new();
    a.serialize(&mut data);
    let c0 = a.solve_dense(&cfg).unwrap().start_cost;

    let mut a2 = build_a();
    {
        for e in a2.poses.iter_mut() {
            e.st.log_s.value = 0.0;
            Component::start(&mut e.st);
        }
    }
    // indices are assigned by a serialize pass; a fresh instance must
    // run one before deserialize can address it
    let mut scratch = Vec::new();
    a2.serialize(&mut scratch);
    a2.deserialize(&data);
    let c2 = a2.solve_dense(&cfg).unwrap().start_cost;
    assert!((c0 - c2).abs() <= 1e-12 * (1.0 + c0), "{c0} vs {c2}");
}

/// Timing A vs B (release, reported not asserted): a broken symbolic
/// cache in the component would drag A behind B.
#[test]
#[ignore]
fn perf_report() {
    use std::time::{Duration, Instant};
    let cfg = LmConfig::conservative();

    // Min-of-rounds, round-robin: background contention only inflates
    // a round, so the per-engine minimum is the noise-robust estimate.
    let rounds = 6;
    let k = 1000;
    let mut min_a = Duration::MAX;
    let mut min_b = Duration::MAX;
    let mut min_c = Duration::MAX;
    let mut last = 0.0;
    for _ in 0..rounds {
        let mut acc = Duration::ZERO;
        for _ in 0..k {
            let mut a = build_a();
            let t0 = Instant::now();
            last = a.solve_dense(&cfg).unwrap().end_cost;
            acc += t0.elapsed();
        }
        min_a = min_a.min(acc);
        let mut acc = Duration::ZERO;
        for _ in 0..k {
            let mut b = build_b();
            let t0 = Instant::now();
            last = b.solve_dense(&cfg).unwrap().end_cost;
            acc += t0.elapsed();
        }
        min_b = min_b.min(acc);
        let mut acc = Duration::ZERO;
        for _ in 0..k {
            let mut c = build_c_n(5);
            let t0 = Instant::now();
            last = c.solve_dense(&cfg).unwrap().end_cost;
            acc += t0.elapsed();
        }
        min_c = min_c.min(acc);
    }
    println!(
        "solve-only min-of-{rounds} (n=5, {k}x): A(macro2) {min_a:?}  B(flat) {min_b:?}  C(builtin) {min_c:?}  A/B {:.3} C/B {:.3}  (cost {last})",
        min_a.as_secs_f64() / min_b.as_secs_f64(),
        min_c.as_secs_f64() / min_b.as_secs_f64()
    );

    // Direct walk probe: update_self is the per-evaluation model walk
    // (recursion + precompute), no solver. Attribution: if A's solve
    // excess matches its walk excess, the gap is walk depth.
    let m = 200_000usize;
    let mut a = build_a();
    let mut b = build_b();
    let mut c = build_c_n(5);
    let mut wmin_a = Duration::MAX;
    let mut wmin_b = Duration::MAX;
    let mut wmin_c = Duration::MAX;
    for _ in 0..rounds {
        let t0 = Instant::now();
        for _ in 0..m {
            arael::model::Model::update_self(&mut a);
        }
        wmin_a = wmin_a.min(t0.elapsed());
        let t0 = Instant::now();
        for _ in 0..m {
            arael::model::Model::update_self(&mut b);
        }
        wmin_b = wmin_b.min(t0.elapsed());
        let t0 = Instant::now();
        for _ in 0..m {
            arael::model::Model::update_self(&mut c);
        }
        wmin_c = wmin_c.min(t0.elapsed());
    }
    println!(
        "walk update_self min-of-{rounds} ({m}x): A {wmin_a:?}  B {wmin_b:?}  C {wmin_c:?}  A/B {:.3} C/B {:.3}",
        wmin_a.as_secs_f64() / wmin_b.as_secs_f64(),
        wmin_c.as_secs_f64() / wmin_b.as_secs_f64()
    );

    // Real scale: constraint math dominates, walk overhead drowns.
    let rounds = 3;
    let k = 5;
    let mut min_a = Duration::MAX;
    let mut min_b = Duration::MAX;
    let mut min_c = Duration::MAX;
    for _ in 0..rounds {
        let mut acc = Duration::ZERO;
        for _ in 0..k {
            let mut a = build_a_n(120);
            let t0 = Instant::now();
            last = a.solve_sparse(&cfg).unwrap().end_cost;
            acc += t0.elapsed();
        }
        min_a = min_a.min(acc);
        let mut acc = Duration::ZERO;
        for _ in 0..k {
            let mut b = build_b_n(120);
            let t0 = Instant::now();
            last = b.solve_sparse(&cfg).unwrap().end_cost;
            acc += t0.elapsed();
        }
        min_b = min_b.min(acc);
        let mut acc = Duration::ZERO;
        for _ in 0..k {
            let mut c = build_c_n(120);
            let t0 = Instant::now();
            last = c.solve_sparse(&cfg).unwrap().end_cost;
            acc += t0.elapsed();
        }
        min_c = min_c.min(acc);
    }
    println!(
        "solve-only n=120 min-of-{rounds} ({k}x): A {min_a:?}  B {min_b:?}  C {min_c:?}  A/B {:.3} C/B {:.3}  (cost {last})",
        min_a.as_secs_f64() / min_b.as_secs_f64(),
        min_c.as_secs_f64() / min_b.as_secs_f64()
    );
}

/// Probe: are nested-component scalar params serialized?
#[test]
fn serialize_probe_log_s() {
    let mut a = build_a();
    let mut data = Vec::new();
    a.serialize(&mut data);
    let mut a2 = build_a();
    for e in a2.poses.iter_mut() {
        e.st.log_s.value = 0.0;
        Component::start(&mut e.st);
    }
    let mut scratch = Vec::new();
    a2.serialize(&mut scratch);
    a2.deserialize(&data);
    let want: Vec<f64> = a.poses.iter().map(|e| e.st.scale()).collect();
    let got: Vec<f64> = a2.poses.iter().map(|e| e.st.scale()).collect();
    println!("data len {}", data.len());
    println!("want {want:?}");
    println!("got  {got:?}");
    assert_eq!(format!("{want:?}"), format!("{got:?}"));
}


// --------------------------- model C: the builtin ScaledTransformParam

#[arael::model]
#[arael(constraint(hb, name = "obs", guard = self.has_obs, {
    let r = simposec.stp.rotation_matrix;
    let t = simposec.stp.translation;
    let s = simposec.stp.scale_factor;
    let p1 = r * simposec.x1 * s + t;
    let p2 = r * simposec.x2 * s + t;
    let p3 = r * simposec.x3 * s + t;
    [p1.x - simposec.m1.x, p1.y - simposec.m1.y, p1.z - simposec.m1.z,
     p2.x - simposec.m2.x, p2.y - simposec.m2.y, p2.z - simposec.m2.z,
     p3.x - simposec.m3.x, p3.y - simposec.m3.y, p3.z - simposec.m3.z]
}))]
#[derive(Default)]
pub struct SimPoseC {
    pub stp: ScaledTransformParam<f64>,
    pub has_obs: bool,
    pub x1: vect3d,
    pub x2: vect3d,
    pub x3: vect3d,
    pub m1: vect3d,
    pub m2: vect3d,
    pub m3: vect3d,
    pub hb: SelfBlock<SimPoseC>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.stp.rotation_matrix;
    let dt = ra.transpose() * (b.stp.translation - a.stp.translation)
        - simlinkc.dt;
    let dr = simlinkc.rmeas_t * (ra.transpose() * b.stp.rotation_matrix);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5,
     b.stp.log_s - a.stp.log_s - simlinkc.dls]
}, parent = simlinkc))]
#[derive(Default)]
pub struct SimLinkC {
    #[arael(ref = root.poses)]
    pub a: Ref<SimPoseC>,
    #[arael(ref = root.poses)]
    pub b: Ref<SimPoseC>,
    pub dt: vect3d,
    pub rmeas_t: matrix3d,
    pub dls: f64,
    pub hb: CrossBlock<SimPoseC, SimPoseC>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
pub struct WorldC {
    pub poses: refs::Vec<SimPoseC>,
    pub links: std::vec::Vec<SimLinkC>,
}

fn build_c_n(n: usize) -> WorldC {
    let (poses, links) = scene_n(n);
    let mut w = WorldC::default();
    for (k, (p, q, s)) in poses.iter().enumerate() {
        w.poses.push(SimPoseC {
            stp: if k == 0 {
                ScaledTransformParam::fixed(*p, *q, *s)
            } else {
                ScaledTransformParam::new(*p, *q, *s)
            },
            ..Default::default()
        });
    }
    for (a, b, t, r, dls) in links {
        w.links.push(SimLinkC {
            a: w.poses.ref_at(a), b: w.poses.ref_at(b),
            dt: t, rmeas_t: r, dls, ..Default::default()
        });
    }
    w
}

/// The builtin against both other formulations: identical costs,
/// iteration counts and states.
#[test]
fn builtin_matches_both_twins() {
    let cfg = LmConfig::conservative();
    let mut a = build_a();
    let mut c = build_c_n(5);
    let ra = a.solve_dense(&cfg).unwrap();
    let rc = c.solve_dense(&cfg).unwrap();

    assert!(rc.status.is_success(), "C: {:?}", rc.status);
    assert!(
        (ra.start_cost - rc.start_cost).abs() < 1e-12 * (1.0 + ra.start_cost),
        "start cost: A {} vs C {}",
        ra.start_cost,
        rc.start_cost
    );
    assert!(
        (ra.end_cost - rc.end_cost).abs() < 1e-12 * (1.0 + ra.end_cost),
        "end cost: A {} vs C {}",
        ra.end_cost,
        rc.end_cost
    );
    assert_eq!(ra.iterations, rc.iterations, "same damping trajectory");

    for (i, (ea, ec)) in a.poses.iter().zip(c.poses.iter()).enumerate() {
        assert!(
            (ea.st.pose.translation - ec.stp.translation).norm() < 1e-12,
            "translation[{i}]"
        );
        assert!(
            (ea.st.scale() - ec.stp.scale).abs() < 1e-12,
            "scale[{i}]: {} vs {}",
            ea.st.scale(),
            ec.stp.scale
        );
    }
}
