// The threaded cost and assembly sweeps over per-thread block stores,
// which a `par` root gets under the `rayon` feature, must agree with the
// sequential sweeps. A split store sums each entity's contributions in
// its own range and the gather adds the ranges, so self blocks match to
// rounding, not to the bit; cross tiles have one writer and match
// exactly.
//
// The model covers every form the sweeps take: self-block constraints
// on a collection (one guarded), a flat cross constraint (one instance
// aliased to a single point), constraints nested under a parent held in
// an arena (with a robust loss), a remote block through an Option, a
// three-entity multi-cross constraint, and a constraint on the root's
// own parameter. The stores live in a solve `Context`, never in the
// root. A root without `par` keeps one store at any thread count.
#![cfg(feature = "rayon")]

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem, RootProblem, LmProblemInternals};
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
    hb: SelfBlock<Point>,
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
    hb: CrossBlock<Point, Point>,
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
#[arael(root, par)]
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
// an entity of the store too.
#[arael::model]
#[arael(root, par)]
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

// A root that does not ask for `par` keeps one store whatever the
// thread count.
#[arael::model]
#[arael(constraint(coo, { [a.pos.x + b.pos.y + c.pos.x - 1.0] }))]
struct Loop {
    #[arael(ref = root.points)] a: Ref<Point>,
    #[arael(ref = root.points)] b: Ref<Point>,
    #[arael(ref = root.points)] c: Ref<Point>,
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

// The same shape asking for `par`. Each store keeps its own COO list, so
// this threads like any other root and must land on the sequential
// answer.
#[arael::model]
#[arael(root, par)]
struct Knot {
    points: refs::Vec<Point>,
    links: std::vec::Vec<Link>,
    loops: std::vec::Vec<Loop>,
    anchor: f64,
    drift: f64,
    spring: f64,
}

// A `par` root with extended hooks: the sweeps thread, and the hooks run
// on the calling thread on either side of the region.
#[arael::model]
#[arael(root, par, extended)]
struct ExtWeb {
    points: refs::Vec<Point>,
    links: std::vec::Vec<Link>,
    pull: f64,
    anchor: f64,
    drift: f64,
    spring: f64,
}

impl arael::model::ExtendedModel<f64> for ExtWeb {
    fn extended_compute(&mut self, params: &[f64], grad: &mut [f64],
                        _coo: &mut arael::model::Coo<f64>) {
        // One residual on the first point's x, added straight to the
        // gradient: no Hessian entry, so nothing for the COO list.
        let i = self.points.ref_at(0).index() as usize;
        grad[i] += 2.0 * (params[i] - self.pull);
    }
    fn extended_cost(&self, params: &[f64]) -> f64 {
        let i = self.points.ref_at(0).index() as usize;
        let r = params[i] - self.pull;
        r * r
    }
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
        w.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: SelfBlock::new() });
    }
    for i in 1..n {
        let (a, b) = (w.points.ref_at(i - 1), w.points.ref_at(i));
        w.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
    }
    // An aliased link: both slots the same point.
    let p = w.points.ref_at(n / 2);
    w.links.push(Link { a: p, b: p, rest: 0.0, hb: CrossBlock::new() });
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
        k.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: SelfBlock::new() });
    }
    for i in 1..n {
        let (a, b) = (k.points.ref_at(i - 1), k.points.ref_at(i));
        k.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
    }
    k
}

/// The shape a robot leaving a feature-rich area for a barren one leaves
/// behind: landmarks ordered by pose, the early ones seen many times and
/// the later ones barely at all. The imbalance is contiguous, which is
/// the worst case for a cut into contiguous ranges -- an even split by
/// landmark count hands the first store a quarter of the landmarks and
/// far more than a quarter of the observations.
fn build_skewed(rich: usize, barren: usize) -> Web {
    let mut w = build(24);
    w.landmarks = refs::Arena::new();
    let add = |w: &mut Web, i: usize, frines_n: usize| {
        let mut frines = std::vec::Vec::new();
        for k in 0..frines_n {
            frines.push(Frine {
                point: w.points.ref_at(k % 24),
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
    };
    for i in 0..rich { add(&mut w, i, 20); }
    for i in 0..barren { add(&mut w, rich + i, 1); }
    w
}

/// The constraint instances each store is given, by reading the cut the
/// solve built. Balance is in instances, not in units: a landmark is one
/// unit whether it carries twenty observations or one. Its own drift
/// constraint counts too -- the work under a landmark is that plus one
/// per observation, which is what the cut is asked to divide.
fn work_per_store(w: &Web, ctx: &arael::threads::Context, stores: usize) -> std::vec::Vec<u64> {
    let cut = ctx.cut();
    let n = w.landmarks.slot_count() as u32;
    // The landmarks container is the one whose ranges span exactly it.
    let mut which = None;
    for c in 0..cut.walks() {
        let hi = (0..stores).map(|s| cut.store(s)[c].1).max().unwrap_or(0);
        if hi == n {
            assert!(which.is_none() || which == Some(c), "two containers span {} slots", n);
            which = Some(c);
        }
    }
    let c = which.expect("the landmarks container is in the cut");
    (0..stores).map(|s| {
        let (lo, hi) = cut.store(s)[c];
        w.landmarks.iter_range(lo, hi).map(|lm| 1 + lm.frines.len() as u64).sum()
    }).collect()
}

/// A cut that divided landmarks evenly would give the first store every
/// rich one. The weight is instances under a unit, so the ranges come out
/// uneven in landmarks and even in observations.
#[test]
fn the_cut_balances_observations_not_landmarks() {
    let stores = 4;
    let mut w = build_skewed(8, 40);
    let mut ctx = context(stores);
    w.begin_with_context(&mut ctx);

    let per = work_per_store(&w, &ctx, stores);
    let total: u64 = per.iter().sum();
    assert_eq!(total, 8 * 21 + 40 * 2, "every instance lands in exactly one store");

    let share = total as f64 / stores as f64;
    let worst = per.iter().map(|&x| (x as f64 / share - 1.0).abs()).fold(0.0, f64::max);
    assert!(worst < 0.10, "work per store {:?}, share {:.1}", per, share);

    // And it is genuinely uneven in landmarks, which is the point: an even
    // split by count would have been far worse in the measure that matters.
    let cut = ctx.cut();
    let c = (0..cut.walks())
        .find(|&c| (0..stores).map(|s| cut.store(s)[c].1).max() == Some(48))
        .expect("landmarks");
    let units: std::vec::Vec<u32> =
        (0..stores).map(|s| { let (lo, hi) = cut.store(s)[c]; hi - lo }).collect();
    assert!(units.iter().max() != units.iter().min(),
        "the ranges should differ in landmark count: {:?}", units);
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
    let stores = ctx.blocks_list::<WebBlocks>().expect("the stores are built when the solve starts");
    assert_eq!(stores.len(), 4, "one store per thread");
    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = seq.calc_grad_hessian_dense(&x, &mut g1, &mut h1);
    // Every later call through the same context must agree with the
    // first exactly: the stores are reused, not rebuilt.
    let c2 = par.calc_grad_hessian_dense_with_context(&x, &mut g2, &mut h2, &mut ctx);
    assert!(close(c1, c2, 1e-13), "cost {} vs {}", c1, c2);
    assert_close("grad", &g1, &g2, 1e-12);
    assert_close("hessian", &h1, &h2, 1e-12);
    for _ in 0..3 {
        let (mut g3, mut h3) = (vec![0.0; n], vec![0.0; n * n]);
        let c3 = par.calc_grad_hessian_dense_with_context(&x, &mut g3, &mut h3, &mut ctx);
        assert_eq!(c2, c3);
        assert_eq!(g2, g3);
        assert_eq!(h2, h3);
    }

}

/// The cost over the stores agrees with the sequential cost, and
/// repeats exactly through the same context.
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
    for _ in 0..3 {
        let b = par.calc_cost_with_context(&x, &mut ctx);
        assert_eq!(a, b, "the two forms sum in the same order");
    }

    // At one thread the context form is the model's own path.
    let mut one = context(1);
    par.begin_with_context(&mut one);
    assert_eq!(one.blocks_list::<WebBlocks>().unwrap().len(), 1, "one thread is one store");
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

/// A freed arena slot is skipped by a store's range like it is by the
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
/// never threaded. A context serves a second solve with its stores.
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
    assert!(ctx.blocks_list::<WebBlocks>().is_some_and(|v| v.len() > 1));
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
    // One thread everywhere: one store.
    let ctx = solve(&mut w, &base, &mut solver);
    assert_eq!(ctx.blocks_list::<WebBlocks>().unwrap().len(), 1);
    // The sweeps threaded, the linear solve sequential.
    let cfg = LmConfig::<f64> { assembly_threads: Some(3), ..base.clone() };
    let ctx = solve(&mut w, &cfg, &mut solver);
    assert_eq!(ctx.threads(), 3);
    assert!(ctx.blocks_list::<WebBlocks>().is_some_and(|v| v.len() == 3));
    // `Some(1)` pins the sweeps sequential while the linear solve threads.
    let cfg = LmConfig::<f64> { num_threads: 4, assembly_threads: Some(1), ..base.clone() };
    let ctx = solve(&mut w, &cfg, &mut solver);
    assert_eq!(ctx.threads(), 1);
    assert_eq!(ctx.blocks_list::<WebBlocks>().unwrap().len(), 1);
}

/// The solve's report says what the threads did: the counts, the form
/// each sweep settled on, and, with `gather_timing`, where their time
/// went. A model that cannot thread says that instead.
#[test]
fn the_report_says_what_the_threads_did() {
    let cfg = LmConfig::<f64> {
        max_iters: 6, num_threads: 2, gather_timing: true, ..Default::default()
    };
    let r = build(40).solve_sparse(&cfg).unwrap();
    let text = r.report();
    assert!(text.contains("threads   sweeps 2, linear 2"), "{}", text);
    assert!(text.contains("assembly    threaded"), "{}", text);
    assert!(text.contains("cost        threaded"), "the cost sweep threads too: {}", text);
    assert!(text.contains("per assembly region"), "the assembly timing is in it: {}", text);
    assert!(text.contains("per cost    region"), "the cost timing is in it: {}", text);
    assert_eq!(r.threads.sweeps.as_ref().map(|s| s.threads), Some(2));
    assert!(!r.threads.fell_back());

    // A one-thread solve says nothing about threads.
    let quiet = build(40).solve_sparse(&LmConfig::<f64> { max_iters: 3, ..Default::default() })
        .unwrap();
    assert!(!quiet.report().contains("threads"), "{}", quiet.report());

    // A root that did not ask for `par` reports that instead.
    let mut loose = Loose {
        points: refs::Vec::new(), links: std::vec::Vec::new(), loops: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0,
    };
    for i in 0..8 {
        let pos = vect2d::new(i as f64 * 0.5, 0.3);
        loose.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: SelfBlock::new() });
    }
    for i in 1..8 {
        let (a, b) = (loose.points.ref_at(i - 1), loose.points.ref_at(i));
        loose.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
    }
    let r = loose.solve_sparse(&LmConfig::<f64> { max_iters: 3, num_threads: 4, ..Default::default() })
        .unwrap();
    assert!(r.threads.fell_back());
    assert!(r.report().contains("no threaded assembly"), "{}", r.report());
}

/// The sweep timing runs only when the context asks for it; the call
/// counts are kept either way.
#[test]
fn timing_is_off_unless_asked() {
    let cfg = LmConfig::<f64> { max_iters: 3, num_threads: 2, ..Default::default() };
    let mut w = build(20);
    let mut x = Vec::new();
    w.serialize(&mut x);
    let mut solver = arael::simple_lm::SparseFaer::new();
    let mut quiet = Context::new();
    arael::simple_lm::lm_solve_with_context(&x, &mut solver, &mut w, &cfg, &mut quiet).unwrap();
    let t = quiet.sweep_timing();
    assert!(!t.on);
    assert!(t.assembly.calls() > 0, "the calls are counted with the clocks off");
    assert!(t.cost.calls() > 0, "and so are the cost sweep's");
    assert_eq!(t.assembly.par.region + t.assembly.seq.region, std::time::Duration::ZERO);
    assert_eq!(t.cost.par.region + t.cost.seq.region, std::time::Duration::ZERO);
    assert_eq!(t.scatter, std::time::Duration::ZERO);
    let mut timed = Context::new();
    timed.set_timing(true);
    arael::simple_lm::lm_solve_with_context(&x, &mut solver, &mut w, &cfg, &mut timed).unwrap();
    let t = timed.sweep_timing();
    assert!(t.on);
    assert!(t.assembly.par.region > std::time::Duration::ZERO);
    assert!(t.assembly.par.task_max > std::time::Duration::ZERO);
    assert!(t.cost.par.region > std::time::Duration::ZERO);
    assert!(t.scatter > std::time::Duration::ZERO);
}

/// A root that did not ask for `par` solves through the sequential
/// path at every thread count, and its context keeps one store.
#[test]
fn a_root_without_par_keeps_the_sequential_path() {
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
            l.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: SelfBlock::new() });
        }
        for i in 1..12 {
            let (a, b) = (l.points.ref_at(i - 1), l.points.ref_at(i));
            l.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
        }
        for i in 0..10 {
            l.loops.push(Loop {
                a: l.points.ref_at(i),
                b: l.points.ref_at(i + 1),
                c: l.points.ref_at(i + 2),
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
    assert_eq!(ctx.blocks_list::<WebBlocks>().map_or(1, |v| v.len()), 1,
        "a root that cannot thread keeps one store");
}

/// A context holds one root's stores at a time and starts another
/// root's fresh.
#[test]
fn a_context_holds_one_root_at_a_time() {
    assert_eq!(Context::new().threads(), 1);
    let mut ctx = context(2);
    let mut w = build(10);
    w.begin_with_context(&mut ctx);
    assert!(ctx.blocks_list::<WebBlocks>().is_some_and(|v| v.len() == 2));
    assert!(ctx.blocks_list::<KnobBlocks>().is_none());
    let mut k = build_knob(10);
    k.begin_with_context(&mut ctx);
    assert!(ctx.blocks_list::<WebBlocks>().is_none());
    assert!(ctx.blocks_list::<KnobBlocks>().is_some());
    let copy = ctx.clone();
    assert!(copy.blocks_list::<KnobBlocks>().is_some_and(|v| v.len() == 2),
        "a cloned context carries the stores");
}

/// The root's own parameter and constraint go through a store like any
/// entity's.
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

/// Every route assembles the same Hessian over the stores: the dense
/// one against the COO and the indexed CSC routes.
#[test]
fn routes_agree_over_the_stores() {
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

/// Fixed parameters through the threaded assembly. Each thread writes
/// its own gradient vector at the live indices, so a fixed slot is
/// skipped at the write rather than dropped by a scatter, and an entity
/// touched from several threads still has to come out summed once.
#[test]
fn fixed_params_match_sequential() {
    let fix = |w: &mut Web| {
        // Whole entities: at a thread boundary and in the middle.
        w.points[0].pos.optimize = false;
        w.points[15].pos.optimize = false;
        // An entity every thread's leaves reach through a reference.
        w.points[29].pos.optimize = false;
    };
    let mut seq = build(30);
    fix(&mut seq);
    let mut par = build(30);
    fix(&mut par);

    let mut x = Vec::new();
    seq.serialize(&mut x);
    let mut x2 = Vec::new();
    par.serialize(&mut x2);
    assert_eq!(x, x2);
    let mut full = Vec::new();
    build(30).serialize(&mut full);
    assert_eq!(x.len() + 6, full.len(), "three fixed points, two slots each");

    let mut ctx = context(4);
    par.begin_with_context(&mut ctx);
    assert!(ctx.blocks_list::<WebBlocks>().expect("stores built").len() > 1);

    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = seq.calc_grad_hessian_dense(&x, &mut g1, &mut h1);
    let c2 = par.calc_grad_hessian_dense_with_context(&x, &mut g2, &mut h2, &mut ctx);
    assert!(close(c1, c2, 1e-13), "cost {} vs {}", c1, c2);
    assert_close("grad", &g1, &g2, 1e-12);
    assert_close("hessian", &h1, &h2, 1e-12);
}

/// The root's own parameter fixed: the root is an entity of the store
/// too, and a single-instance one.
#[test]
fn a_fixed_root_param_matches_sequential() {
    let mut seq = build_knob(24);
    seq.scale.optimize = false;
    let mut par = build_knob(24);
    par.scale.optimize = false;

    let mut x = Vec::new();
    seq.serialize(&mut x);
    let mut x2 = Vec::new();
    par.serialize(&mut x2);
    assert_eq!(x, x2);
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
}

/// A `par` root with a `coo` constraint threads like any other: each
/// store keeps its own COO list, so the sweeps split and the answer is
/// the sequential one.
#[test]
fn a_triplet_root_threads_its_sweeps() {
    let build = |n: usize| {
        let mut k = Knot {
            points: refs::Vec::new(),
            links: std::vec::Vec::new(),
            loops: std::vec::Vec::new(),
            anchor: 100.0,
            drift: 0.01,
            spring: 1.0,
        };
        for i in 0..n {
            let pos = vect2d::new(i as f64 * 0.5, if i % 2 == 0 { 0.7 } else { -0.7 });
            k.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: SelfBlock::new() });
        }
        for i in 1..n {
            let (a, b) = (k.points.ref_at(i - 1), k.points.ref_at(i));
            k.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
        }
        for i in 0..n - 2 {
            k.loops.push(Loop {
                a: k.points.ref_at(i),
                b: k.points.ref_at(i + 1),
                c: k.points.ref_at(i + 2),
            });
        }
        k
    };
    let cfg = |t: usize| LmConfig::<f64> { max_iters: 20, num_threads: t, ..Default::default() };
    let s = build(14).solve_sparse(&cfg(1)).unwrap();
    let mut model = build(14);
    let mut x = Vec::new();
    model.serialize(&mut x);
    let mut ctx = Context::new();
    let mut solver = arael::simple_lm::SparseFaer::new();
    let p = arael::simple_lm::lm_solve_with_context(&x, &mut solver, &mut model, &cfg(4), &mut ctx)
        .unwrap();
    // The COO entries a store holds are its own, summed in its own
    // range, so the two solves agree to rounding rather than the bit.
    assert_eq!(s.x.len(), p.x.len());
    for (i, (a, b)) in s.x.iter().zip(p.x.iter()).enumerate() {
        assert!((a - b).abs() < 1e-9, "x[{i}]: {a} vs {b}");
    }
    assert_eq!(ctx.blocks_list::<KnotBlocks>().map_or(1, |v| v.len()), 4,
        "a triplet root splits like any other");
    assert!(!p.threads.fell_back(), "and does not report a fallback: {}", p.report());
}

/// A `par` root with `extended` hooks: the sweeps thread over their
/// stores and the hooks run once each on the calling thread, so the
/// assembly matches the sequential one and carries the hook's cost and
/// gradient.
#[test]
fn an_extended_root_threads_its_sweeps() {
    let build = |n: usize| {
        let mut w = ExtWeb {
            points: refs::Vec::new(),
            links: std::vec::Vec::new(),
            pull: 3.0,
            anchor: 100.0,
            drift: 0.01,
            spring: 1.0,
        };
        for i in 0..n {
            let pos = vect2d::new(i as f64 * 0.5, if i % 2 == 0 { 0.7 } else { -0.7 });
            w.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: SelfBlock::new() });
        }
        for i in 1..n {
            let (a, b) = (w.points.ref_at(i - 1), w.points.ref_at(i));
            w.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
        }
        w
    };
    let mut seq = build(24);
    let mut par = build(24);
    let mut x = Vec::new();
    seq.serialize(&mut x);
    // Serializing is what wires the block slots, so both models need it.
    let mut x2 = Vec::new();
    par.serialize(&mut x2);
    assert_eq!(x, x2);
    let mut ctx = context(4);
    par.begin_with_context(&mut ctx);
    assert_eq!(ctx.blocks_list::<ExtWebBlocks>().map(|v| v.len()), Some(4),
        "`extended` does not stop the split");

    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = seq.calc_grad_hessian_dense(&x, &mut g1, &mut h1);
    let c2 = par.calc_grad_hessian_dense_with_context(&x, &mut g2, &mut h2, &mut ctx);
    assert!(close(c1, c2, 1e-13), "cost {} vs {}", c1, c2);
    assert_close("grad", &g1, &g2, 1e-12);
    assert_close("hessian", &h1, &h2, 1e-12);

    // The hook ran exactly once: its residual is in the cost, and its
    // gradient term is in the first point's x entry.
    let i = par.points.ref_at(0).index() as usize;
    let mut bare = build(24);
    let mut x3 = Vec::new();
    bare.serialize(&mut x3);
    // The same model with the hook's residual driven to zero, so the
    // difference is exactly what the hook added.
    bare.pull = x[i];
    let (mut g0, mut h0) = (vec![0.0; n], vec![0.0; n * n]);
    let c0 = bare.calc_grad_hessian_dense(&x, &mut g0, &mut h0);
    let r = x[i] - par.pull;
    assert!(close(c2 - c0, r * r, 1e-12), "the hook's cost is in there once");
    assert!(close(g2[i] - g0[i], 2.0 * r, 1e-12), "and its gradient once");

    // And a full solve through the threaded path lands where the
    // sequential one does.
    let cfg = |t: usize| LmConfig::<f64> { max_iters: 20, num_threads: t, ..Default::default() };
    let s = build(24).solve_sparse(&cfg(1)).unwrap();
    let p = build(24).solve_sparse(&cfg(4)).unwrap();
    assert!(close(s.end_cost, p.end_cost, 1e-9), "{} vs {}", s.end_cost, p.end_cost);
    assert_close("x", &s.x, &p.x, 1e-6);
}

/// Every backend assembles through the solve context, so a `par` root's
/// assembly sweeps thread on the dense route exactly as they do on the
/// sparse one.
#[test]
fn the_dense_route_threads_its_assembly() {
    let cfg = LmConfig::<f64> { max_iters: 4, num_threads: 4, ..Default::default() };
    let r = build(24).solve_dense(&cfg).unwrap();
    let rep = r.report();
    let s = r.threads.sweeps.expect("a par root reports its sweeps");
    assert!(s.cost.threaded, "the cost sweeps thread: {}", rep);
    assert!(s.assembly.threaded, "and so do the assembly sweeps: {}", rep);
}

/// Print the two reports, for eyeballing: `cargo test --features rayon
/// --test par_assembly show_reports -- --nocapture --ignored`.
#[test]
#[ignore]
fn show_reports() {
    let cfg = LmConfig::<f64> {
        max_iters: 8, num_threads: 4, gather_timing: true, ..Default::default()
    };
    println!("{}", build(400).solve_sparse(&cfg).unwrap().report());
}
