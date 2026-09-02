// Transform values in constraint bodies: `tr2w * x`, `inv()`,
// composition and the named forms against the hand-written
// compositions, digit for digit -- same costs, same optimum, same
// iteration count -- for TransformParam and ScaledTransformParam.
// The operator bodies must build the very expressions the hand-written
// ones do, so every comparison here is exact, never within a tolerance.

use arael::matrix::matrix3d;
use arael::model::{CrossBlock, SelfBlock};
use arael::quatern::quaternd;
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem, RootProblem};
use arael::transform::{ScaledTransformParam, TransformParam};
use arael::vect::vect3d;

// ------------------------------------------------------------ rigid, operators

/// A robot pose r2w with observations in every form the value offers:
/// world points seen in the robot frame (inverse action), robot points
/// seen in the world (action), directions both ways (rotation only).
#[arael::model]
#[arael(constraint(hb, {
    let e1 = oppose.r2w.inv() * oppose.x1 - oppose.m1;
    let e2 = oppose.r2w.inverse_transform(oppose.x2) - oppose.m2;
    let e3 = oppose.r2w * oppose.y1 - oppose.w1;
    let e4 = oppose.r2w.transform(oppose.y2) - oppose.w2;
    let e5 = oppose.r2w.rotate(oppose.n1) - oppose.nm1;
    let e6 = oppose.r2w.inverse_rotate(oppose.nm2) - oppose.n2;
    [e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, e3.x, e3.y, e3.z,
     e4.x, e4.y, e4.z, e5.x, e5.y, e5.z, e6.x, e6.y, e6.z]
}))]
#[derive(Default)]
struct OpPose {
    r2w: TransformParam<f64>,
    x1: vect3d, m1: vect3d, x2: vect3d, m2: vect3d,
    y1: vect3d, w1: vect3d, y2: vect3d, w2: vect3d,
    n1: vect3d, nm1: vect3d, n2: vect3d, nm2: vect3d,
    hb: SelfBlock<OpPose>,
}

/// A relative-pose link through the composed inverse.
#[arael::model]
#[arael(constraint(hb, {
    let rel = a.r2w.inv() * b.r2w;
    let dt = rel.translation - oplink.dt;
    let dr = oplink.rmeas_t * rel.rotation_matrix;
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5]
}, parent = oplink))]
#[derive(Default)]
struct OpLink {
    #[arael(ref = root.poses)]
    a: Ref<OpPose>,
    #[arael(ref = root.poses)]
    b: Ref<OpPose>,
    dt: vect3d,
    rmeas_t: matrix3d,
    hb: CrossBlock<OpPose, OpPose>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
struct OpWorld {
    poses: refs::Vec<OpPose>,
    links: std::vec::Vec<OpLink>,
}

// ------------------------------------------------------------ rigid, by hand

#[arael::model]
#[arael(constraint(hb, {
    let r = hpose.r2w.rotation_matrix;
    let t = hpose.r2w.translation;
    let e1 = r.transpose() * (hpose.x1 - t) - hpose.m1;
    let e2 = r.transpose() * (hpose.x2 - t) - hpose.m2;
    let e3 = r * hpose.y1 + t - hpose.w1;
    let e4 = r * hpose.y2 + t - hpose.w2;
    let e5 = r * hpose.n1 - hpose.nm1;
    let e6 = r.transpose() * hpose.nm2 - hpose.n2;
    [e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, e3.x, e3.y, e3.z,
     e4.x, e4.y, e4.z, e5.x, e5.y, e5.z, e6.x, e6.y, e6.z]
}))]
#[derive(Default)]
struct HPose {
    r2w: TransformParam<f64>,
    x1: vect3d, m1: vect3d, x2: vect3d, m2: vect3d,
    y1: vect3d, w1: vect3d, y2: vect3d, w2: vect3d,
    n1: vect3d, nm1: vect3d, n2: vect3d, nm2: vect3d,
    hb: SelfBlock<HPose>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.r2w.rotation_matrix;
    let dt = ra.transpose() * (b.r2w.translation - a.r2w.translation) - hlink.dt;
    let dr = hlink.rmeas_t * (ra.transpose() * b.r2w.rotation_matrix);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5]
}, parent = hlink))]
#[derive(Default)]
struct HLink {
    #[arael(ref = root.poses)]
    a: Ref<HPose>,
    #[arael(ref = root.poses)]
    b: Ref<HPose>,
    dt: vect3d,
    rmeas_t: matrix3d,
    hb: CrossBlock<HPose, HPose>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
struct HWorld {
    poses: refs::Vec<HPose>,
    links: std::vec::Vec<HLink>,
}

// ------------------------------------------------------------ similarity, operators

#[arael::model]
#[arael(constraint(hb, {
    let e1 = spose.st.inv() * spose.x1 - spose.m1;
    let e2 = spose.st * spose.y1 - spose.w1;
    let e3 = spose.st.rotate(spose.n1) - spose.nm1;
    let e4 = spose.st.inverse_rotate(spose.nm2) - spose.n2;
    [e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, e3.x, e3.y, e3.z, e4.x, e4.y, e4.z]
}))]
#[derive(Default)]
struct SPose {
    st: ScaledTransformParam<f64>,
    x1: vect3d, m1: vect3d, y1: vect3d, w1: vect3d,
    n1: vect3d, nm1: vect3d, n2: vect3d, nm2: vect3d,
    hb: SelfBlock<SPose>,
}

#[arael::model]
#[arael(constraint(hb, {
    let rel = a.st.inv() * b.st;
    let dt = rel.translation - slink.dt;
    let dr = slink.rmeas_t * rel.rotation_matrix;
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5,
     rel.scale_factor - slink.ds]
}, parent = slink))]
#[derive(Default)]
struct SLink {
    #[arael(ref = root.poses)]
    a: Ref<SPose>,
    #[arael(ref = root.poses)]
    b: Ref<SPose>,
    dt: vect3d,
    rmeas_t: matrix3d,
    ds: f64,
    hb: CrossBlock<SPose, SPose>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
struct SWorld {
    poses: refs::Vec<SPose>,
    links: std::vec::Vec<SLink>,
}

// ------------------------------------------------------------ similarity, by hand

#[arael::model]
#[arael(constraint(hb, {
    let r = tpose.st.rotation_matrix;
    let t = tpose.st.translation;
    let s = tpose.st.scale_factor;
    let e1 = r.transpose() * (tpose.x1 - t) / s - tpose.m1;
    let e2 = r * tpose.y1 * s + t - tpose.w1;
    let e3 = r * tpose.n1 - tpose.nm1;
    let e4 = r.transpose() * tpose.nm2 - tpose.n2;
    [e1.x, e1.y, e1.z, e2.x, e2.y, e2.z, e3.x, e3.y, e3.z, e4.x, e4.y, e4.z]
}))]
#[derive(Default)]
struct TPose {
    st: ScaledTransformParam<f64>,
    x1: vect3d, m1: vect3d, y1: vect3d, w1: vect3d,
    n1: vect3d, nm1: vect3d, n2: vect3d, nm2: vect3d,
    hb: SelfBlock<TPose>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.st.rotation_matrix;
    let dt = ra.transpose() * (b.st.translation - a.st.translation) / a.st.scale_factor
        - tlink.dt;
    let dr = tlink.rmeas_t * (ra.transpose() * b.st.rotation_matrix);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5,
     b.st.scale_factor / a.st.scale_factor - tlink.ds]
}, parent = tlink))]
#[derive(Default)]
struct TLink {
    #[arael(ref = root.poses)]
    a: Ref<TPose>,
    #[arael(ref = root.poses)]
    b: Ref<TPose>,
    dt: vect3d,
    rmeas_t: matrix3d,
    ds: f64,
    hb: CrossBlock<TPose, TPose>,
}

#[arael::model]
#[arael(root)]
#[derive(Default)]
struct TWorld {
    poses: refs::Vec<TPose>,
    links: std::vec::Vec<TLink>,
}

// ------------------------------------------------------------ scene

/// Deterministic pseudo-noise in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
    fn v3(&mut self, amp: f64) -> vect3d {
        vect3d::new(self.next() * amp, self.next() * amp, self.next() * amp)
    }
}

struct PoseData {
    t: vect3d,
    q: quaternd,
    s: f64,
    t0: vect3d,
    q0: quaternd,
    s0: f64,
    // world point and its robot-frame measurement, robot point and its
    // world measurement, robot direction and its world measurement, and
    // the reverse
    x1: vect3d, m1: vect3d, x2: vect3d, m2: vect3d,
    y1: vect3d, w1: vect3d, y2: vect3d, w2: vect3d,
    n1: vect3d, nm1: vect3d, n2: vect3d, nm2: vect3d,
}

struct LinkData {
    a: usize,
    b: usize,
    dt: vect3d,
    rmeas_t: matrix3d,
    ds: f64,
}

/// Poses along an arc, scales near 1.25, measurements from the truth
/// with noise so the optimum is a genuine least-squares compromise,
/// and perturbed starting values.
fn scene(n: usize) -> (Vec<PoseData>, Vec<LinkData>) {
    let mut g = Lcg(0x5eed);
    let mut poses = Vec::new();
    for i in 0..n {
        let ang = 0.3 * i as f64;
        let t = vect3d::new(2.0 * ang.cos(), 2.0 * ang.sin(), 0.1 * i as f64);
        let q = quaternd::from_axis_angle(vect3d::new(0.1, -0.2, 1.0).unit(), 0.4 * i as f64 + 0.2);
        let s = 1.25 + 0.05 * i as f64;
        let r = q.rotation_matrix();
        let world = |p: vect3d| r * p * s + t;
        let robot = |p: vect3d| r.transpose() * (p - t) / s;
        let x1 = t + g.v3(1.5);
        let x2 = t + g.v3(1.5);
        let y1 = g.v3(1.0);
        let y2 = g.v3(1.0);
        let n1 = g.v3(1.0).unit();
        let n2 = g.v3(1.0).unit();
        let noise = 0.02;
        poses.push(PoseData {
            t, q, s,
            t0: t + g.v3(0.15),
            q0: (q * quaternd::from_axis_angle(g.v3(1.0).unit(), 0.1 * g.next())).unit(),
            s0: s * (1.0 + 0.05 * g.next()),
            m1: robot(x1) + g.v3(noise), x1,
            m2: robot(x2) + g.v3(noise), x2,
            w1: world(y1) + g.v3(noise), y1,
            w2: world(y2) + g.v3(noise), y2,
            nm1: r * n1 + g.v3(noise), n1,
            nm2: r * n2 + g.v3(noise), n2,
        });
    }
    let mut links = Vec::new();
    for i in 0..n.saturating_sub(1) {
        let (a, b) = (&poses[i], &poses[i + 1]);
        let ra = a.q.rotation_matrix();
        let rb = b.q.rotation_matrix();
        let rel_r = ra.transpose() * rb;
        let dt = ra.transpose() * (b.t - a.t) / a.s + g.v3(0.02);
        let rmeas = rel_r * quaternd::from_axis_angle(g.v3(1.0).unit(), 0.02 * g.next()).rotation_matrix();
        links.push(LinkData { a: i, b: i + 1, dt, rmeas_t: rmeas.transpose(), ds: b.s / a.s * (1.0 + 0.01 * g.next()) });
    }
    (poses, links)
}

macro_rules! fill_rigid {
    ($world:ty, $pose:ident, $link:ident) => {{
        let (poses, links) = scene(5);
        let mut w = <$world>::default();
        for p in &poses {
            w.poses.push($pose {
                r2w: TransformParam::new(p.t0, p.q0),
                x1: p.x1, m1: p.m1, x2: p.x2, m2: p.m2,
                y1: p.y1, w1: p.w1, y2: p.y2, w2: p.w2,
                n1: p.n1, nm1: p.nm1, n2: p.n2, nm2: p.nm2,
                hb: SelfBlock::new(),
            });
        }
        for l in &links {
            // Rigid links: the relative translation of the unscaled scene,
            // perturbed so the fit is a compromise.
            let (a, b) = (&poses[l.a], &poses[l.b]);
            let dt = a.q.rotation_matrix().transpose() * (b.t - a.t);
            w.links.push($link {
                a: w.poses.ref_at(l.a), b: w.poses.ref_at(l.b),
                dt: dt + vect3d::new(0.01, -0.02, 0.015),
                rmeas_t: l.rmeas_t,
                hb: CrossBlock::new(),
            });
        }
        w
    }};
}

macro_rules! fill_scaled {
    ($world:ty, $pose:ident, $link:ident) => {{
        let (poses, links) = scene(5);
        let mut w = <$world>::default();
        for p in &poses {
            w.poses.push($pose {
                st: ScaledTransformParam::new(p.t0, p.q0, p.s0),
                x1: p.x1, m1: p.m1, y1: p.y1, w1: p.w1,
                n1: p.n1, nm1: p.nm1, n2: p.n2, nm2: p.nm2,
                hb: SelfBlock::new(),
            });
        }
        for l in &links {
            w.links.push($link {
                a: w.poses.ref_at(l.a), b: w.poses.ref_at(l.b),
                dt: l.dt, rmeas_t: l.rmeas_t, ds: l.ds,
                hb: CrossBlock::new(),
            });
        }
        w
    }};
}

fn cfg() -> LmConfig<f64> {
    LmConfig { max_iters: 40, ..Default::default() }
}

fn q_parts(q: &quaternd) -> [f64; 4] {
    [q.t, q.v.x, q.v.y, q.v.z]
}

/// Rigid: the operator bodies and the hand-written bodies are the same
/// problem to the last bit -- costs, iterations, every parameter.
#[test]
fn rigid_operators_match_hand_written_exactly() {
    let mut op = fill_rigid!(OpWorld, OpPose, OpLink);
    let mut hand = fill_rigid!(HWorld, HPose, HLink);
    assert!(op.validate().is_clean());
    let ro = op.solve_dense(&cfg()).unwrap();
    let rh = hand.solve_dense(&cfg()).unwrap();
    assert!(ro.status.is_success(), "{:?}", ro.status);
    assert!(ro.start_cost > 1e-3, "the scene must start away from the optimum: {}", ro.start_cost);
    assert!(ro.end_cost > 1e-6, "noisy measurements never fit exactly: {}", ro.end_cost);
    assert_eq!(ro.start_cost, rh.start_cost);
    assert_eq!(ro.end_cost, rh.end_cost);
    assert_eq!(ro.iterations, rh.iterations);
    for i in 0..op.poses.len() {
        let (a, b) = (&op.poses[i].r2w, &hand.poses[i].r2w);
        assert_eq!([a.translation.x, a.translation.y, a.translation.z],
                   [b.translation.x, b.translation.y, b.translation.z], "pose {i} translation");
        assert_eq!(q_parts(&a.rotation), q_parts(&b.rotation), "pose {i} rotation");
    }
}

/// Similarity: the same, with the scale flowing through the action,
/// the inverse action and the composed relative transform.
#[test]
fn similarity_operators_match_hand_written_exactly() {
    let mut op = fill_scaled!(SWorld, SPose, SLink);
    let mut hand = fill_scaled!(TWorld, TPose, TLink);
    assert!(op.validate().is_clean());
    let ro = op.solve_dense(&cfg()).unwrap();
    let rh = hand.solve_dense(&cfg()).unwrap();
    assert!(ro.status.is_success(), "{:?}", ro.status);
    assert!(ro.start_cost > 1e-3, "the scene must start away from the optimum: {}", ro.start_cost);
    assert_eq!(ro.start_cost, rh.start_cost);
    assert_eq!(ro.end_cost, rh.end_cost);
    assert_eq!(ro.iterations, rh.iterations);
    for i in 0..op.poses.len() {
        let (a, b) = (&op.poses[i].st, &hand.poses[i].st);
        assert_eq!([a.translation.x, a.translation.y, a.translation.z],
                   [b.translation.x, b.translation.y, b.translation.z], "pose {i} translation");
        assert_eq!(q_parts(&a.rotation), q_parts(&b.rotation), "pose {i} rotation");
        assert_eq!(a.scale, b.scale, "pose {i} scale");
        assert!((a.scale - scene(5).0[i].s).abs() < 0.1, "scale {} recovered near {}", a.scale, scene(5).0[i].s);
    }
}

/// The action is what the value claims: at the ground truth, without
/// noise, every row is zero.
#[test]
fn actions_are_the_documented_maps() {
    let mut g = Lcg(7);
    let t = vect3d::new(0.3, -0.2, 0.5);
    let q = quaternd::from_axis_angle(vect3d::new(0.2, 0.5, 1.0).unit(), 0.7);
    let s = 1.3;
    let r = q.rotation_matrix();
    let x1 = g.v3(1.0);
    let y1 = g.v3(1.0);
    let n1 = g.v3(1.0).unit();
    let n2 = g.v3(1.0).unit();
    let mut w = SWorld::default();
    w.poses.push(SPose {
        st: ScaledTransformParam::new(t, q, s),
        x1, m1: r.transpose() * (x1 - t) / s,
        y1, w1: r * y1 * s + t,
        n1, nm1: r * n1,
        n2, nm2: r * n2,
        hb: SelfBlock::new(),
    });
    let c = w.solve_dense(&cfg()).unwrap().start_cost;
    assert!(c < 1e-28, "cost at the truth: {c}");
}

/// The runtime methods on the params compute what the body forms
/// compute: measurements taken with the runtime API leave the operator
/// bodies at zero cost, for both builtins and the composed link.
#[test]
fn runtime_forms_agree_with_the_body_forms() {
    let mut g = Lcg(11);
    let (t, q) = (vect3d::new(0.3, -0.2, 0.5),
                  quaternd::from_axis_angle(vect3d::new(0.2, 0.5, 1.0).unit(), 0.7));
    let (x1, x2, y1, y2) = (g.v3(1.0), g.v3(1.0), g.v3(1.0), g.v3(1.0));
    let (n1, n2) = (g.v3(1.0).unit(), g.v3(1.0).unit());
    let r2w = TransformParam::new(t, q);
    let mut w = OpWorld::default();
    w.poses.push(OpPose {
        x1, m1: r2w.inverse_transform(x1),
        x2, m2: r2w.inv() * x2,
        y1, w1: r2w.transform(y1),
        y2, w2: &r2w * y2,
        n1, nm1: r2w.rotate(n1),
        n2, nm2: r2w.rotate(n2),
        r2w,
        hb: SelfBlock::new(),
    });
    let (tb, qb) = (vect3d::new(1.1, 0.4, 0.2),
                    quaternd::from_axis_angle(vect3d::new(1.0, 0.1, -0.3).unit(), 0.4));
    let b = TransformParam::new(tb, qb);
    let rel = w.poses[0].r2w.inv() * &b;
    w.poses.push(OpPose {
        x1, m1: b.inverse_transform(x1), x2, m2: b.inverse_transform(x2),
        y1, w1: b.transform(y1), y2, w2: b.transform(y2),
        n1, nm1: b.rotate(n1), n2, nm2: b.rotate(n2),
        r2w: b, hb: SelfBlock::new(),
    });
    w.links.push(OpLink {
        a: w.poses.ref_at(0), b: w.poses.ref_at(1),
        dt: rel.translation, rmeas_t: rel.rotation_matrix.transpose(),
        hb: CrossBlock::new(),
    });
    let c = w.solve_dense(&cfg()).unwrap().start_cost;
    assert!(c < 1e-26, "rigid cost at runtime-made measurements: {c}");

    let st = ScaledTransformParam::new(t, q, 1.3);
    let mut s = SWorld::default();
    s.poses.push(SPose {
        x1, m1: st.inv() * x1, y1, w1: &st * y1,
        n1, nm1: st.rotate(n1), n2, nm2: st.rotate(n2),
        st, hb: SelfBlock::new(),
    });
    let sb = ScaledTransformParam::new(tb, qb, 0.8);
    let srel = s.poses[0].st.inv() * &sb;
    s.poses.push(SPose {
        x1, m1: sb.inverse_transform(x1), y1, w1: sb.transform(y1),
        n1, nm1: sb.rotate(n1), n2, nm2: sb.rotate(n2),
        st: sb, hb: SelfBlock::new(),
    });
    s.links.push(SLink {
        a: s.poses.ref_at(0), b: s.poses.ref_at(1),
        dt: srel.translation, rmeas_t: srel.rotation_matrix.transpose(), ds: srel.scale,
        hb: CrossBlock::new(),
    });
    let c = s.solve_dense(&cfg()).unwrap().start_cost;
    assert!(c < 1e-26, "similarity cost at runtime-made measurements: {c}");
}
