// The threaded cost and assembly sweeps over per-thread mirrors, which
// every root gets under the `rayon` feature, must agree with the
// sequential sweeps. The mirror path sums each entity's contributions
// per thread and then over the threads, so it matches to rounding, not
// to the bit; cross tiles have one writer and match exactly.
//
// The model covers every form the mirrors take: self-block constraints
// on a collection (one guarded), a flat cross constraint (one instance
// aliased to a single point), constraints nested under a parent held in
// an arena (with a robust loss), a remote block through an Option, a
// three-entity multi-cross constraint, and a constraint on the root's
// own parameter. The mirrors live in a solve `Context`, never in the
// root. A root with a form the mirrors do not cover keeps the
// sequential path at any thread count.
#![cfg(feature = "rayon")]

use arael::model::{BoxedCrossBlock, BoxedSelfBlock, CrossBlock, Param, SelfBlock, TripletBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem, RootProblem};
use arael::threads::Context;
use arael::vect::vect2d;

#[arael::model]
#[arael(constraint(hb, guard = self.is_anchor, {
    [point.pos.x * root.anchor, point.pos.y * root.anchor]
}))]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * root.drift, d.y * root.drift]
}))]
struct Point {
    pos: Param<vect2d>,
    is_anchor: bool,
    hb: BoxedSelfBlock<Point>,
}

#[arael::model]
#[arael(constraint(hb, {
    let d = b.pos - a.pos;
    [(d.x * d.x + d.y * d.y - link.rest * link.rest) * root.spring]
}))]
struct Link {
    #[arael(ref = root.points)] a: Ref<Point>,
    #[arael(ref = root.points)] b: Ref<Point>,
    rest: f64,
    hb: BoxedCrossBlock<Point, Point>,
}

#[arael::model]
#[arael(constraint(hb, {
    let d = landmark.pos - landmark.prior;
    [d.x * root.drift, d.y * root.drift]
}))]
struct Landmark {
    pos: Param<vect2d>,
    prior: vect2d,
    frines: std::vec::Vec<Frine>,
    hb: SelfBlock<Landmark>,
}

#[arael::model]
#[arael(constraint(hb, parent = lm, loss = |s| loss_huber(s, frine.k2), {
    let d = lm.pos - point.pos - frine.meas;
    [d.x * root.spring, d.y * root.spring]
}))]
struct Frine {
    #[arael(ref = root.points)] point: Ref<Point>,
    meas: vect2d,
    k2: f64,
    hb: CrossBlock<Landmark, Point>,
}

#[arael::model]
#[arael(constraint(p.hb, {
    [p.pos.x - prior.pos.x, p.pos.y - prior.pos.y]
}))]
struct Prior {
    #[arael(ref = root.points)] p: Ref<Point>,
    pos: vect2d,
}

#[arael::model]
#[arael(constraint([hb_ab, hb_ac, hb_bc], {
    let dx_ab = a.pos.x - b.pos.x;
    let dy_ab = a.pos.y - b.pos.y;
    let dx_bc = b.pos.x - c.pos.x;
    let dy_bc = b.pos.y - c.pos.y;
    let dx_ac = a.pos.x - c.pos.x;
    [(dx_ab + dx_bc - dx_ac) * root.spring,
     (dy_ab + dy_bc) * root.spring,
     (dx_ab * dy_bc) * root.spring]
}))]
struct Tri {
    #[arael(ref = root.points)] a: Ref<Point>,
    #[arael(ref = root.points)] b: Ref<Point>,
    #[arael(ref = root.points)] c: Ref<Point>,
    #[arael(cross = (a, b))] hb_ab: CrossBlock<Point, Point>,
    #[arael(cross = (a, c))] hb_ac: CrossBlock<Point, Point>,
    #[arael(cross = (b, c))] hb_bc: CrossBlock<Point, Point>,
}

#[arael::model]
#[arael(root)]
struct Web {
    points: refs::Vec<Point>,
    landmarks: refs::Arena<Landmark>,
    links: std::vec::Vec<Link>,
    tris: std::vec::Vec<Tri>,
    prior: Option<Prior>,
    anchor: f64,
    drift: f64,
    spring: f64,
}

// A root with a parameter of its own and a constraint on it: the root is
// an entity of the mirror too.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [(knob.scale - 1.5) * knob.anchor]
}))]
struct Knob {
    scale: Param<f64>,
    points: refs::Vec<Point>,
    links: std::vec::Vec<Link>,
    anchor: f64,
    drift: f64,
    spring: f64,
    hb: SelfBlock<Knob>,
}

// A TripletBlock has no per-thread copy: this root has no mirror path
// and runs the sequential sweeps whatever the thread count.
#[arael::model]
#[arael(constraint(hb, { [a.pos.x + b.pos.y + c.pos.x - 1.0] }))]
struct Loop {
    #[arael(ref = root.points)] a: Ref<Point>,
    #[arael(ref = root.points)] b: Ref<Point>,
    #[arael(ref = root.points)] c: Ref<Point>,
    hb: TripletBlock<f64>,
}

#[arael::model]
#[arael(root)]
struct Loose {
    points: refs::Vec<Point>,
    links: std::vec::Vec<Link>,
    loops: std::vec::Vec<Loop>,
    anchor: f64,
    drift: f64,
    spring: f64,
}

fn build(n: usize) -> Web {
    let mut w = Web {
        points: refs::Vec::new(),
        landmarks: refs::Arena::new(),
        links: std::vec::Vec::new(),
        tris: std::vec::Vec::new(),
        prior: None,
        anchor: 100.0,
        drift: 0.01,
        spring: 1.0,
    };
    for i in 0..n {
        let pos = vect2d::new(i as f64 * 0.5, if i % 2 == 0 { 0.7 } else { -0.7 });
        w.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: BoxedSelfBlock::new() });
    }
    for i in 1..n {
        let (a, b) = (w.points.ref_at(i - 1), w.points.ref_at(i));
        w.links.push(Link { a, b, rest: 1.0, hb: BoxedCrossBlock::new() });
    }
    // An aliased link: both slots the same point.
    let p = w.points.ref_at(n / 2);
    w.links.push(Link { a: p, b: p, rest: 0.0, hb: BoxedCrossBlock::new() });
    for i in 0..n / 3 {
        let mut frines = std::vec::Vec::new();
        for k in 0..3 {
            let j = (3 * i + k) % n;
            frines.push(Frine {
                point: w.points.ref_at(j),
                meas: vect2d::new(0.3 * k as f64, -0.2),
                k2: 0.5,
                hb: CrossBlock::new(),
            });
        }
        w.landmarks.push(Landmark {
            pos: Param::new(vect2d::new(i as f64 * 1.5, 0.1)),
            prior: vect2d::new(i as f64 * 1.5, 0.0),
            frines,
            hb: SelfBlock::new(),
        });
    }
    for i in 0..n.saturating_sub(2) {
        w.tris.push(Tri {
            a: w.points.ref_at(i),
            b: w.points.ref_at(i + 1),
            c: w.points.ref_at(i + 2),
            hb_ab: CrossBlock::new(),
            hb_ac: CrossBlock::new(),
            hb_bc: CrossBlock::new(),
        });
    }
    w.prior = Some(Prior { p: w.points.ref_at(1), pos: vect2d::new(0.4, -0.6) });
    w
}

fn build_knob(n: usize) -> Knob {
    let mut k = Knob {
        scale: Param::new(0.7),
        points: refs::Vec::new(),
        links: std::vec::Vec::new(),
        anchor: 100.0,
        drift: 0.01,
        spring: 1.0,
        hb: SelfBlock::new(),
    };
    for i in 0..n {
        let pos = vect2d::new(i as f64 * 0.5, if i % 2 == 0 { 0.7 } else { -0.7 });
        k.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: BoxedSelfBlock::new() });
    }
    for i in 1..n {
        let (a, b) = (k.points.ref_at(i - 1), k.points.ref_at(i));
        k.links.push(Link { a, b, rest: 1.0, hb: BoxedCrossBlock::new() });
    }
    k
}

/// A context for `threads`, the way the solve entries size one.
fn context(threads: usize) -> Context {
    let mut ctx = Context::new();
    ctx.set_threads(threads);
    ctx
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * (1.0 + a.abs().max(b.abs()))
}

fn assert_close(what: &str, a: &[f64], b: &[f64], rel: f64) {
    assert_eq!(a.len(), b.len());
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(close(*x, *y, rel), "{} differs at {}: {} vs {}", what, i, x, y);
    }
}

/// The dense assembly at four threads agrees with the sequential one:
/// the cost, the gradient and every Hessian entry.
#[test]
fn assembly_matches_sequential() {
    let mut seq = build(30);
    let mut par = build(30);
    let mut x = Vec::new();
    seq.serialize(&mut x);
    let mut x2 = Vec::new();
    par.serialize(&mut x2);
    assert_eq!(x, x2);
    let mut ctx = context(4);
    par.begin_with_context(&mut ctx);
    let m = ctx.mirrors::<WebMirror>().expect("the mirrors are built when the solve starts");
    assert!(m.is_on());
    assert_eq!(m.threads(), 4);
    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = seq.calc_grad_hessian_dense(&x, &mut g1, &mut h1);
    // Through the whole trial over the mirrors: the sequential form and
    // the dispatched form must agree with each other exactly.
    let c2 = par.calc_grad_hessian_dense_with_context(&x, &mut g2, &mut h2, &mut ctx);
    assert!(close(c1, c2, 1e-13), "cost {} vs {}", c1, c2);
    assert_close("grad", &g1, &g2, 1e-12);
    assert_close("hessian", &h1, &h2, 1e-12);
    for _ in 1..2 * arael::threads::Trial::SAMPLES {
        let (mut g3, mut h3) = (vec![0.0; n], vec![0.0; n * n]);
        let c3 = par.calc_grad_hessian_dense_with_context(&x, &mut g3, &mut h3, &mut ctx);
        assert_eq!(c2, c3);
        assert_eq!(g2, g3);
        assert_eq!(h2, h3);
    }
    assert!(ctx.mirrors::<WebMirror>().unwrap().assembly.measured().is_some(), "both forms timed");
}

/// The cost over the mirrors agrees with the sequential cost, in the
/// trial's sequential form, its dispatched form, and after the decision.
#[test]
fn cost_matches_sequential() {
    let mut seq = build(40);
    let mut par = build(40);
    let mut x = Vec::new();
    seq.serialize(&mut x);
    let mut x2 = Vec::new();
    par.serialize(&mut x2);
    let mut ctx = context(3);
    par.begin_with_context(&mut ctx);
    let c = seq.calc_cost(&x);
    let a = par.calc_cost_with_context(&x, &mut ctx);
    assert!(close(c, a, 1e-13), "{} vs {}", c, a);
    for _ in 1..=2 * arael::threads::Trial::SAMPLES {
        let b = par.calc_cost_with_context(&x, &mut ctx);
        assert_eq!(a, b, "the two forms sum in the same order");
    }
    assert!(ctx.mirrors::<WebMirror>().unwrap().cost.measured().is_some());
    // At one thread the context form is the model's own path.
    let mut one = context(1);
    par.begin_with_context(&mut one);
    assert!(!one.mirrors::<WebMirror>().unwrap().is_on());
    assert_eq!(par.calc_cost_with_context(&x, &mut one), par.calc_cost(&x));
}

/// A sparse solve at four threads lands where the sequential solve does.
#[test]
fn sparse_solve_matches_sequential() {
    let cfg = |t: usize| LmConfig::<f64> { max_iters: 100, num_threads: t, ..Default::default() };
    let s = build(60).solve_sparse(&cfg(1)).unwrap();
    let p = build(60).solve_sparse(&cfg(4)).unwrap();
    assert_eq!(s.status, p.status);
    assert!(close(s.end_cost, p.end_cost, 1e-9), "{} vs {}", s.end_cost, p.end_cost);
    assert_close("x", &s.x, &p.x, 1e-6);
    assert!(s.end_cost < s.start_cost);
}

/// A freed arena slot is skipped by the mirrors like it is by the
/// sequential sweep.
#[test]
fn arena_hole() {
    let mut seq = build(30);
    let mut par = build(30);
    let r = seq.landmarks.first_ref().unwrap();
    seq.landmarks.remove(r);
    let r = par.landmarks.first_ref().unwrap();
    par.landmarks.remove(r);
    let cfg = |t: usize| LmConfig::<f64> { max_iters: 50, num_threads: t, ..Default::default() };
    let s = seq.solve_sparse(&cfg(1)).unwrap();
    let p = par.solve_sparse(&cfg(4)).unwrap();
    assert!(close(s.end_cost, p.end_cost, 1e-9), "{} vs {}", s.end_cost, p.end_cost);
    assert_close("x", &s.x, &p.x, 1e-6);
}

/// A threaded solve leaves the model's own blocks as they were: an
/// evaluation without a context afterwards is bit-identical to a model
/// never threaded. A context serves a second solve with its mirrors.
#[test]
fn the_model_is_untouched_by_a_threaded_solve() {
    let mut fresh = build(20);
    let mut used = build(20);
    let mut x = Vec::new();
    fresh.serialize(&mut x);
    let mut x2 = Vec::new();
    used.serialize(&mut x2);
    let cfg = LmConfig::<f64> { max_iters: 3, num_threads: 4, ..Default::default() };
    let mut ctx = Context::new();
    let mut solver = arael::simple_lm::SparseFaer::new();
    let r = arael::simple_lm::lm_solve_with_context(&x2, &mut solver, &mut used, &cfg, &mut ctx).unwrap();
    assert!(ctx.mirrors::<WebMirror>().is_some_and(|m| m.is_on()));
    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = fresh.calc_grad_hessian_dense(&x, &mut g1, &mut h1);
    let c2 = used.calc_grad_hessian_dense(&x, &mut g2, &mut h2);
    assert_eq!(c1, c2);
    assert_eq!(g1, g2);
    assert_eq!(h1, h2);
    let r2 = arael::simple_lm::lm_solve_with_context(&x2, &mut solver, &mut used, &cfg, &mut ctx).unwrap();
    assert_eq!(r.x, r2.x, "the same solve through the same context");
}

/// `assembly_threads` gives the sweeps a count of their own, leaving the
/// linear solve on `num_threads`.
#[test]
fn assembly_threads_overrides_the_sweep_count() {
    let mut w = build(20);
    let mut x = Vec::new();
    w.serialize(&mut x);
    let mut solver = arael::simple_lm::SparseFaer::new();
    let solve = |w: &mut Web, cfg: &LmConfig<f64>, solver: &mut arael::simple_lm::SparseFaer<f64>| {
        let mut ctx = Context::new();
        arael::simple_lm::lm_solve_with_context(&x, solver, w, cfg, &mut ctx).unwrap();
        ctx
    };
    let base = LmConfig::<f64> { max_iters: 2, num_threads: 1, ..Default::default() };
    // One thread everywhere: no mirrors.
    let ctx = solve(&mut w, &base, &mut solver);
    assert!(!ctx.mirrors::<WebMirror>().is_some_and(|m| m.is_on()));
    // The sweeps threaded, the linear solve sequential.
    let cfg = LmConfig::<f64> { assembly_threads: Some(3), ..base.clone() };
    let ctx = solve(&mut w, &cfg, &mut solver);
    assert_eq!(ctx.threads(), 3);
    assert!(ctx.mirrors::<WebMirror>().is_some_and(|m| m.threads() == 3));
    // `Some(1)` pins the sweeps sequential while the linear solve threads.
    let cfg = LmConfig::<f64> { num_threads: 4, assembly_threads: Some(1), ..base.clone() };
    let ctx = solve(&mut w, &cfg, &mut solver);
    assert_eq!(ctx.threads(), 1);
    assert!(!ctx.mirrors::<WebMirror>().unwrap().is_on());
}

/// The phase timing runs only when the context asks for it; the counts
/// are kept either way.
#[test]
fn timing_is_off_unless_asked() {
    let cfg = LmConfig::<f64> { max_iters: 3, num_threads: 2, ..Default::default() };
    let mut w = build(20);
    let mut x = Vec::new();
    w.serialize(&mut x);
    let mut solver = arael::simple_lm::SparseFaer::new();
    let mut quiet = Context::new();
    arael::simple_lm::lm_solve_with_context(&x, &mut solver, &mut w, &cfg, &mut quiet).unwrap();
    let t = &quiet.mirrors::<WebMirror>().unwrap().timing;
    assert!(!t.on);
    assert_eq!(t.builds, 1);
    assert!(t.assembly.calls() > 0);
    assert_eq!(t.build, std::time::Duration::ZERO);
    assert_eq!(t.assembly.par.region + t.assembly.seq.region, std::time::Duration::ZERO);
    let mut timed = Context::new();
    timed.set_timing(true);
    arael::simple_lm::lm_solve_with_context(&x, &mut solver, &mut w, &cfg, &mut timed).unwrap();
    let t = &timed.mirrors::<WebMirror>().unwrap().timing;
    assert!(t.on);
    assert!(t.build > std::time::Duration::ZERO);
    assert!(t.assembly.par.region + t.assembly.seq.region > std::time::Duration::ZERO);
}

/// A root with a form the mirrors do not cover solves through the
/// sequential path at every thread count, and its context holds no
/// mirrors.
#[test]
fn an_uncovered_root_keeps_the_sequential_path() {
    let build = || {
        let mut l = Loose {
            points: refs::Vec::new(),
            links: std::vec::Vec::new(),
            loops: std::vec::Vec::new(),
            anchor: 100.0,
            drift: 0.01,
            spring: 1.0,
        };
        for i in 0..12 {
            let pos = vect2d::new(i as f64 * 0.5, if i % 2 == 0 { 0.7 } else { -0.7 });
            l.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: BoxedSelfBlock::new() });
        }
        for i in 1..12 {
            let (a, b) = (l.points.ref_at(i - 1), l.points.ref_at(i));
            l.links.push(Link { a, b, rest: 1.0, hb: BoxedCrossBlock::new() });
        }
        for i in 0..10 {
            l.loops.push(Loop {
                a: l.points.ref_at(i),
                b: l.points.ref_at(i + 1),
                c: l.points.ref_at(i + 2),
                hb: TripletBlock::new(),
            });
        }
        l
    };
    let cfg = |t: usize| LmConfig::<f64> { max_iters: 20, num_threads: t, ..Default::default() };
    let s = build().solve_sparse(&cfg(1)).unwrap();
    let mut model = build();
    let mut x = Vec::new();
    model.serialize(&mut x);
    let mut ctx = Context::new();
    let mut solver = arael::simple_lm::SparseFaer::new();
    let p = arael::simple_lm::lm_solve_with_context(&x, &mut solver, &mut model, &cfg(4), &mut ctx).unwrap();
    assert_eq!(s.x, p.x, "the same sequential arithmetic at four threads");
    assert_eq!(ctx.threads(), 4);
    assert!(ctx.mirrors::<WebMirror>().is_none(), "no mirrors of any root");
}

/// A context holds one root's mirrors at a time and starts another
/// root's fresh.
#[test]
fn a_context_holds_one_root_at_a_time() {
    assert_eq!(Context::new().threads(), 1);
    let mut ctx = context(2);
    let mut w = build(10);
    w.begin_with_context(&mut ctx);
    assert!(ctx.mirrors::<WebMirror>().is_some_and(|m| m.threads() == 2));
    assert!(ctx.mirrors::<KnobMirror>().is_none());
    let mut k = build_knob(10);
    k.begin_with_context(&mut ctx);
    assert!(ctx.mirrors::<WebMirror>().is_none());
    assert!(ctx.mirrors::<KnobMirror>().is_some_and(|m| m.is_on()));
    let copy = ctx.clone();
    assert!(copy.mirrors::<KnobMirror>().is_some_and(|m| m.threads() == 2));
}

/// The root's own parameter and constraint go through the mirror like
/// any entity's.
#[test]
fn root_params_match_sequential() {
    let mut seq = build_knob(20);
    let mut par = build_knob(20);
    let mut x = Vec::new();
    seq.serialize(&mut x);
    let mut x2 = Vec::new();
    par.serialize(&mut x2);
    let mut ctx = context(4);
    par.begin_with_context(&mut ctx);
    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = seq.calc_grad_hessian_dense(&x, &mut g1, &mut h1);
    let c2 = par.calc_grad_hessian_dense_with_context(&x, &mut g2, &mut h2, &mut ctx);
    assert!(close(c1, c2, 1e-13), "cost {} vs {}", c1, c2);
    assert_close("grad", &g1, &g2, 1e-12);
    assert_close("hessian", &h1, &h2, 1e-12);
    let cfg = |t: usize| LmConfig::<f64> { max_iters: 100, num_threads: t, ..Default::default() };
    let s = build_knob(20).solve_dense(&cfg(1)).unwrap();
    let p = build_knob(20).solve_dense(&cfg(4)).unwrap();
    assert!(close(s.end_cost, p.end_cost, 1e-9), "{} vs {}", s.end_cost, p.end_cost);
    assert_close("x", &s.x, &p.x, 1e-6);
    assert!(close(p.x[0], 1.5, 1e-6), "the root's scale converges: {}", p.x[0]);
}

/// Every route assembles the same Hessian over the mirrors: the dense
/// one against the COO and the indexed CSC routes.
#[test]
fn routes_agree_over_the_mirrors() {
    let mut par = build(24);
    let mut x = Vec::new();
    par.serialize(&mut x);
    let mut ctx = context(2);
    par.begin_with_context(&mut ctx);
    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = par.calc_grad_hessian_dense_with_context(&x, &mut g1, &mut h1, &mut ctx);
    let mut g2 = vec![0.0; n];
    let mut coo = arael::simple_lm::CooMatrix::new(n);
    let c2 = par.calc_grad_hessian_sparse_with_context(&x, &mut g2, &mut coo, &mut ctx);
    assert_eq!(c1, c2);
    assert_eq!(g1, g2);
    let mut dense_from_coo = vec![0.0; n * n];
    for k in 0..coo.rows.len() {
        let (i, j, v) = (coo.rows[k] as usize, coo.cols[k] as usize, coo.vals[k]);
        dense_from_coo[i * n + j] += v;
        if i != j { dense_from_coo[j * n + i] += v; }
    }
    assert_close("coo vs dense", &dense_from_coo, &h1, 1e-12);
}
