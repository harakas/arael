// The containment/block combination matrix: every collection type x
// block type x nesting shape, each verified against a hand-computed
// cost, with all assembly routes (dense, COO, indexed CSC, band)
// compared against each other, the gradient checked by finite
// differences, and validate() clean. A silently dropped sweep or a
// misfilled block fails the cost or route comparison here.

use arael::model::{Param, SelfBlock, CrossBlock, TripletBlock, BoxedSelfBlock, BoxedCrossBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{CooMatrix, LmProblem, RootProblem};

const TOL: f64 = 1e-9;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

fn densify_coo(n: usize, coo: &CooMatrix<f64>) -> Vec<f64> {
    let mut full = vec![0.0; n * n];
    for k in 0..coo.rows.len() {
        let (r, c) = (coo.rows[k] as usize, coo.cols[k] as usize);
        full[r * n + c] += coo.vals[k];
        if r != c {
            full[c * n + r] += coo.vals[k];
        }
    }
    full
}

/// The invariant battery every combination case runs.
fn check_model<P>(label: &str, m: &mut P, manual_cost: f64)
where
    P: LmProblem<f64> + RootProblem<f64>,
{
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let n = x.len();
    assert!(n > 0, "{label}: no parameters serialized");

    let cost = m.calc_cost(&x);
    assert!(close(cost, manual_cost, TOL),
        "{label}: calc_cost {} != manual {}", cost, manual_cost);

    let mut gd = vec![0.0; n];
    let mut hd = vec![0.0; n * n];
    let cd = m.calc_grad_hessian_dense(&x, &mut gd, &mut hd);
    assert!(close(cd, cost, TOL), "{label}: dense cost {} != {}", cd, cost);
    for i in 0..n {
        for j in 0..n {
            assert!(close(hd[i * n + j], hd[j * n + i], TOL),
                "{label}: dense H not symmetric at ({i},{j})");
        }
    }

    let mut gs = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    let cs = m.calc_grad_hessian_sparse(&x, &mut gs, &mut coo);
    assert!(close(cs, cost, TOL), "{label}: coo cost {} != {}", cs, cost);
    for i in 0..n {
        assert!(close(gs[i], gd[i], TOL), "{label}: coo grad[{i}] {} != dense {}", gs[i], gd[i]);
    }
    let hs = densify_coo(n, &coo);
    for i in 0..n {
        for j in 0..n {
            assert!(close(hs[i * n + j], hd[i * n + j], TOL),
                "{label}: coo H[{i},{j}] {} != dense {}", hs[i * n + j], hd[i * n + j]);
        }
    }

    let (csc, positions) = coo.to_csc_with_map().unwrap();
    let mut gi = vec![0.0; n];
    let mut vals = vec![0.0; csc.vals.len()];
    let ci = m.calc_grad_hessian_sparse_indexed(&x, &mut gi, &mut vals, &positions);
    assert!(close(ci, cost, TOL), "{label}: indexed cost {} != {}", ci, cost);
    for i in 0..n {
        assert!(close(gi[i], gd[i], TOL), "{label}: indexed grad[{i}]");
    }
    let mut hi = vec![0.0; n * n];
    for j in 0..n {
        for k in csc.col_ptr[j]..csc.col_ptr[j + 1] {
            let r = csc.row_idx[k] as usize;
            hi[r * n + j] += vals[k];
            if r != j {
                hi[j * n + r] += vals[k];
            }
        }
    }
    for i in 0..n {
        for j in 0..n {
            assert!(close(hi[i * n + j], hd[i * n + j], TOL),
                "{label}: indexed H[{i},{j}] {} != dense {}", hi[i * n + j], hd[i * n + j]);
        }
    }

    let kd = n - 1;
    let ldab = kd + 1;
    let mut gb = vec![0.0; n];
    let mut band = vec![0.0; ldab * n];
    let cb = m.calc_grad_hessian_band(&x, &mut gb, &mut band, kd)
        .unwrap_or_else(|e| panic!("{label}: band overflow at full bandwidth: {e}"));
    assert!(close(cb, cost, TOL), "{label}: band cost {} != {}", cb, cost);
    for i in 0..n {
        assert!(close(gb[i], gd[i], TOL), "{label}: band grad[{i}]");
    }
    for i in 0..n {
        for j in i..n {
            let v = band[(kd + i - j) + j * ldab];
            assert!(close(v, hd[i * n + j], TOL),
                "{label}: band H[{i},{j}] {} != dense {}", v, hd[i * n + j]);
        }
    }

    let d = m.check_gradients(&x);
    assert!(d.is_clean(), "{label}: gradient check:\n{}", d);
    let d = m.validate();
    assert!(d.is_clean(), "{label}: validate:\n{}", d);
}

// ================================================================ A:
// containment shapes for a SelfBlock entity.

// Two params with an off-diagonal coupling.
#[arael::model]
#[arael(constraint(hb, {
    [(p.x - p.t) * 2.0, (p.y - p.x) * 0.7]
}))]
struct P {
    x: Param<f64>,
    y: Param<f64>,
    t: f64,
    hb: SelfBlock<P>,
}

fn p(x: f64, y: f64, t: f64) -> P {
    P { x: Param::new(x), y: Param::new(y), t, hb: SelfBlock::new() }
}

fn p_cost(x: f64, y: f64, t: f64) -> f64 {
    ((x - t) * 2.0).powi(2) + ((y - x) * 0.7).powi(2)
}

const PDATA: [(f64, f64, f64); 3] = [(1.0, 2.0, 3.0), (-0.5, 0.3, 1.2), (4.0, -1.0, 0.5)];

fn pdata_cost() -> f64 {
    PDATA.iter().map(|&(x, y, t)| p_cost(x, y, t)).sum()
}

#[arael::model]
#[arael(root)]
struct AStdVec {
    items: std::vec::Vec<P>,
}

#[test]
fn self_block_in_std_vec() {
    let mut w = AStdVec { items: PDATA.iter().map(|&(x, y, t)| p(x, y, t)).collect() };
    check_model("std::Vec", &mut w, pdata_cost());
}

#[arael::model]
#[arael(root)]
struct ARefsVec {
    items: refs::Vec<P>,
}

#[test]
fn self_block_in_refs_vec() {
    let mut items = refs::Vec::new();
    for &(x, y, t) in &PDATA {
        items.push(p(x, y, t));
    }
    check_model("refs::Vec", &mut ARefsVec { items }, pdata_cost());
}

#[arael::model]
#[arael(root)]
struct ADeque {
    items: refs::Deque<P>,
}

#[test]
fn self_block_in_deque_with_front_pushes() {
    let mut items = refs::Deque::new();
    items.push_back(p(PDATA[0].0, PDATA[0].1, PDATA[0].2));
    items.push_back(p(PDATA[1].0, PDATA[1].1, PDATA[1].2));
    items.push_front(p(PDATA[2].0, PDATA[2].1, PDATA[2].2));
    check_model("Deque front+back", &mut ADeque { items }, pdata_cost());
}

#[arael::model]
#[arael(root)]
struct AArena {
    items: refs::Arena<P>,
}

#[test]
fn self_block_in_arena_with_a_hole() {
    // A live hole: the removed slot must contribute nothing anywhere.
    let mut items = refs::Arena::new();
    items.push(p(PDATA[0].0, PDATA[0].1, PDATA[0].2));
    let dead = items.push(p(9.0, 9.0, 9.0));
    items.push(p(PDATA[2].0, PDATA[2].1, PDATA[2].2));
    items.remove(dead).unwrap();
    let manual = p_cost(PDATA[0].0, PDATA[0].1, PDATA[0].2)
        + p_cost(PDATA[2].0, PDATA[2].1, PDATA[2].2);
    check_model("Arena hole", &mut AArena { items }, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(q.a - 1.5) * 1.1]
}))]
struct Q {
    a: Param<f64>,
    hb: SelfBlock<Q>,
}

#[arael::model]
#[arael(root)]
struct ADirect {
    one: P,
    two: Q,
}

#[test]
fn entities_as_direct_root_fields() {
    let mut w = ADirect {
        one: p(PDATA[0].0, PDATA[0].1, PDATA[0].2),
        two: Q { a: Param::new(0.2), hb: SelfBlock::new() },
    };
    let manual = p_cost(PDATA[0].0, PDATA[0].1, PDATA[0].2)
        + ((0.2f64 - 1.5) * 1.1).powi(2);
    check_model("direct fields", &mut w, manual);
}

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [(aroot.a - 2.0) * 3.0, (aroot.b - aroot.a) * 0.5]
}))]
struct ARoot {
    a: Param<f64>,
    b: Param<f64>,
    items: std::vec::Vec<P>,
    hb: SelfBlock<ARoot>,
}

#[test]
fn root_level_params_with_root_self_constraint() {
    let mut w = ARoot {
        a: Param::new(1.0),
        b: Param::new(4.0),
        items: vec![p(PDATA[0].0, PDATA[0].1, PDATA[0].2)],
        hb: SelfBlock::new(),
    };
    let manual = ((1.0f64 - 2.0) * 3.0).powi(2) + ((4.0f64 - 1.0) * 0.5).powi(2)
        + p_cost(PDATA[0].0, PDATA[0].1, PDATA[0].2);
    check_model("root self", &mut w, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(popt.x - popt.t) * 2.0, (popt.y - popt.x) * 0.7]
}))]
struct POpt {
    x: Param<f64>,
    y: Param<f64>,
    t: f64,
    hb: SelfBlock<POpt>,
}

#[arael::model]
#[arael(root)]
struct AOptional {
    maybe: Option<POpt>,
    items: std::vec::Vec<P>,
}

#[test]
fn optional_entity_some_and_none() {
    // Some: the optional entity's constraint fires like any other.
    let mut w = AOptional {
        maybe: Some(POpt {
            x: Param::new(PDATA[0].0), y: Param::new(PDATA[0].1), t: PDATA[0].2,
            hb: SelfBlock::new(),
        }),
        items: vec![p(PDATA[1].0, PDATA[1].1, PDATA[1].2)],
    };
    let manual = p_cost(PDATA[0].0, PDATA[0].1, PDATA[0].2)
        + p_cost(PDATA[1].0, PDATA[1].1, PDATA[1].2);
    check_model("Option Some", &mut w, manual);

    // None: no params, no residuals, nothing anywhere.
    let mut w = AOptional {
        maybe: None,
        items: vec![p(PDATA[1].0, PDATA[1].1, PDATA[1].2)],
    };
    check_model("Option None", &mut w, p_cost(PDATA[1].0, PDATA[1].1, PDATA[1].2));
}

// Optional frines: a cross/triplet constraint entity held in an Option
// iterates as a zero-or-one collection.

#[arael::model]
#[arael(constraint(hb, {
    [(datum.off - 0.5) * 0.9]
}))]
struct Datum {
    off: Param<f64>,
    hb: SelfBlock<Datum>,
}

#[arael::model]
#[arael(constraint(hb, parent = gpose, {
    [(gpose.v - d.off - gps.z) * 2.0]
}))]
struct Gps {
    #[arael(ref = root.datums)]
    d: Ref<Datum>,
    z: f64,
    hb: CrossBlock<GPose, Datum>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(gpose.v - gpose.t) * 0.3]
}))]
struct GPose {
    v: Param<f64>,
    t: f64,
    gps: Option<Gps>,
    hb: SelfBlock<GPose>,
}

#[arael::model]
#[arael(root)]
struct AFrineOpt {
    datums: refs::Vec<Datum>,
    poses: refs::Vec<GPose>,
}

#[test]
fn per_entity_optional_frine() {
    // One pose carries a GPS observation, the other does not.
    let mut datums = refs::Vec::new();
    let rd = datums.push(Datum { off: Param::new(0.4), hb: SelfBlock::new() });
    let mut poses = refs::Vec::new();
    poses.push(GPose { v: Param::new(1.2), t: 1.0,
        gps: Some(Gps { d: rd, z: 1.5, hb: CrossBlock::new() }), hb: SelfBlock::new() });
    poses.push(GPose { v: Param::new(2.1), t: 2.0, gps: None, hb: SelfBlock::new() });
    let mut w = AFrineOpt { datums, poses };
    let manual = ((0.4f64 - 0.5) * 0.9).powi(2)
        + ((1.2f64 - 1.0) * 0.3).powi(2) + ((2.1f64 - 2.0) * 0.3).powi(2)
        + ((1.2f64 - 0.4 - 1.5) * 2.0).powi(2);
    check_model("per-entity Option frine", &mut w, manual);
}

#[arael::model]
struct Mid {
    subs: std::vec::Vec<P>,
}

#[arael::model]
struct Outer3 {
    mids: std::vec::Vec<Mid>,
}

#[arael::model]
#[arael(root)]
struct ANested3 {
    outers: std::vec::Vec<Outer3>,
}

#[test]
fn three_level_nested_collections() {
    let mut w = ANested3 {
        outers: vec![
            Outer3 { mids: vec![
                Mid { subs: vec![p(PDATA[0].0, PDATA[0].1, PDATA[0].2)] },
                Mid { subs: vec![p(PDATA[1].0, PDATA[1].1, PDATA[1].2)] },
            ]},
            Outer3 { mids: vec![Mid { subs: vec![p(PDATA[2].0, PDATA[2].1, PDATA[2].2)] }] },
        ],
    };
    check_model("3-level nesting", &mut w, pdata_cost());
}

// ================================================================ B:
// cross-block ref shapes.

#[arael::model]
#[arael(constraint(hb, {
    [(n.v - n.t) * 0.3]
}))]
struct N {
    v: Param<f64>,
    t: f64,
    hb: SelfBlock<N>,
}

fn n(v: f64, t: f64) -> N {
    N { v: Param::new(v), t, hb: SelfBlock::new() }
}

fn n_cost(v: f64, t: f64) -> f64 {
    ((v - t) * 0.3).powi(2)
}

fn tie_cost(a: f64, b: f64, d: f64) -> f64 {
    ((b - a - d) * 1.5).powi(2)
}

#[arael::model]
#[arael(constraint(hb, {
    [(b.v - a.v - tied.d) * 1.5]
}))]
struct TieD {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    d: f64,
    hb: CrossBlock<N, N>,
}

#[arael::model]
#[arael(root)]
struct BDeque {
    nodes: refs::Deque<N>,
    ties: std::vec::Vec<TieD>,
}

#[test]
fn cross_block_into_deque_targets() {
    let mut nodes = refs::Deque::new();
    let r1 = nodes.push_back(n(1.3, 1.0));
    let r2 = nodes.push_back(n(2.2, 2.0));
    let r0 = nodes.push_front(n(0.1, 0.0));
    let ties = vec![
        TieD { a: r0, b: r1, d: 1.0, hb: CrossBlock::new() },
        TieD { a: r1, b: r2, d: 1.0, hb: CrossBlock::new() },
    ];
    let manual = n_cost(0.1, 0.0) + n_cost(1.3, 1.0) + n_cost(2.2, 2.0)
        + tie_cost(0.1, 1.3, 1.0) + tie_cost(1.3, 2.2, 1.0);
    check_model("cross Deque", &mut BDeque { nodes, ties }, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(b.v - a.v - tiea.d) * 1.5]
}))]
struct TieA {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    d: f64,
    hb: CrossBlock<N, N>,
}

#[arael::model]
#[arael(root)]
struct BArena {
    nodes: refs::Arena<N>,
    ties: std::vec::Vec<TieA>,
}

#[test]
fn cross_block_into_arena_targets_with_a_hole() {
    let mut nodes = refs::Arena::new();
    let r0 = nodes.push(n(0.1, 0.0));
    let dead = nodes.push(n(9.0, 9.0));
    let r2 = nodes.push(n(2.2, 2.0));
    nodes.remove(dead).unwrap();
    let ties = vec![TieA { a: r0, b: r2, d: 2.0, hb: CrossBlock::new() }];
    let manual = n_cost(0.1, 0.0) + n_cost(2.2, 2.0) + tie_cost(0.1, 2.2, 2.0);
    check_model("cross Arena hole", &mut BArena { nodes, ties }, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(cur.v - prev.v - tie2.d) * 1.5]
}))]
struct Tie2 {
    #[arael(ref = parent.nodes)]
    prev: Ref<N>,
    #[arael(ref = parent.nodes)]
    cur: Ref<N>,
    d: f64,
    hb: CrossBlock<N, N>,
}

#[arael::model]
struct Group {
    nodes: refs::Vec<N>,
    ties: std::vec::Vec<Tie2>,
}

#[arael::model]
#[arael(root)]
struct BParent {
    groups: std::vec::Vec<Group>,
}

#[test]
fn parent_scoped_cross_in_nested_groups() {
    let mk_group = |vals: &[(f64, f64)], d: f64| {
        let mut nodes = refs::Vec::new();
        let rs: Vec<Ref<N>> = vals.iter().map(|&(v, t)| nodes.push(n(v, t))).collect();
        let ties = rs.windows(2)
            .map(|w| Tie2 { prev: w[0], cur: w[1], d, hb: CrossBlock::new() })
            .collect();
        Group { nodes, ties }
    };
    let mut w = BParent {
        groups: vec![
            mk_group(&[(0.1, 0.0), (1.3, 1.0)], 1.0),
            mk_group(&[(5.0, 5.0), (6.4, 6.0), (7.1, 7.0)], 1.0),
        ],
    };
    let manual = n_cost(0.1, 0.0) + n_cost(1.3, 1.0)
        + n_cost(5.0, 5.0) + n_cost(6.4, 6.0) + n_cost(7.1, 7.0)
        + tie_cost(0.1, 1.3, 1.0)
        + tie_cost(5.0, 6.4, 1.0) + tie_cost(6.4, 7.1, 1.0);
    check_model("parent-scoped cross", &mut w, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(feat.q - 1.0) * 0.6]
}))]
struct Feat {
    q: Param<f64>,
    hb: SelfBlock<Feat>,
}

#[arael::model]
struct Info {
    feats: refs::Vec<Feat>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(pose6.v - pose6.t) * 0.3]
}))]
struct Pose6 {
    v: Param<f64>,
    t: f64,
    info: Info,
    hb: SelfBlock<Pose6>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(feat.q - pose.v - ob.d) * 1.2]
}))]
struct Ob {
    #[arael(ref = root.poses)]
    pose: Ref<Pose6>,
    #[arael(ref = pose.info.feats)]
    feat: Ref<Feat>,
    d: f64,
    hb: CrossBlock<Pose6, Feat>,
}

#[arael::model]
#[arael(root)]
struct BChained {
    poses: refs::Vec<Pose6>,
    obs: std::vec::Vec<Ob>,
}

#[test]
fn chained_ref_through_a_pose() {
    let mut feats = refs::Vec::new();
    let rf = feats.push(Feat { q: Param::new(0.8), hb: SelfBlock::new() });
    let mut poses = refs::Vec::new();
    let rp = poses.push(Pose6 {
        v: Param::new(0.2), t: 0.0,
        info: Info { feats },
        hb: SelfBlock::new(),
    });
    let obs = vec![Ob { pose: rp, feat: rf, d: 0.5, hb: CrossBlock::new() }];
    let manual = ((0.8f64 - 1.0) * 0.6).powi(2)
        + ((0.2f64 - 0.0) * 0.3).powi(2)
        + ((0.8f64 - 0.2 - 0.5) * 1.2).powi(2);
    check_model("chained ref", &mut BChained { poses, obs }, manual);
}

// ================================================================ C:
// triplet, boxed, and multi-containment shapes.

#[arael::model]
#[arael(constraint(hb, {
    [(a.v + b.v + c.v - tri.s) * 1.1]
}))]
struct Tri {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    #[arael(ref = root.nodes)]
    c: Ref<N>,
    s: f64,
    hb: TripletBlock<f64>,
}

#[arael::model]
#[arael(root)]
struct CTriplet {
    nodes: refs::Vec<N>,
    tris: std::vec::Vec<Tri>,
}

#[test]
fn triplet_block_three_entities() {
    let mut nodes = refs::Vec::new();
    let r0 = nodes.push(n(0.1, 0.0));
    let r1 = nodes.push(n(1.2, 1.0));
    let r2 = nodes.push(n(2.3, 2.0));
    let tris = vec![Tri { a: r0, b: r1, c: r2, s: 3.0, hb: TripletBlock::new() }];
    let manual = n_cost(0.1, 0.0) + n_cost(1.2, 1.0) + n_cost(2.3, 2.0)
        + ((0.1f64 + 1.2 + 2.3 - 3.0) * 1.1).powi(2);
    check_model("triplet", &mut CTriplet { nodes, tris }, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(bx.v - bx.t) * 0.4]
}))]
struct Bx {
    v: Param<f64>,
    t: f64,
    hb: BoxedSelfBlock<Bx>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(b.v - a.v - btie.d) * 1.3]
}))]
struct BTie {
    #[arael(ref = root.nodes)]
    a: Ref<Bx>,
    #[arael(ref = root.nodes)]
    b: Ref<Bx>,
    d: f64,
    hb: BoxedCrossBlock<Bx, Bx>,
}

#[arael::model]
#[arael(root)]
struct CBoxed {
    nodes: refs::Vec<Bx>,
    ties: std::vec::Vec<BTie>,
}

#[test]
fn boxed_blocks_through_every_route() {
    let mut nodes = refs::Vec::new();
    let r0 = nodes.push(Bx { v: Param::new(0.3), t: 0.0, hb: BoxedSelfBlock::new() });
    let r1 = nodes.push(Bx { v: Param::new(1.4), t: 1.0, hb: BoxedSelfBlock::new() });
    let ties = vec![BTie { a: r0, b: r1, d: 1.0, hb: BoxedCrossBlock::new() }];
    let manual = ((0.3f64 - 0.0) * 0.4).powi(2) + ((1.4f64 - 1.0) * 0.4).powi(2)
        + ((1.4f64 - 0.3 - 1.0) * 1.3).powi(2);
    check_model("boxed", &mut CBoxed { nodes, ties }, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(mc.v - mc.t) * 0.8]
}))]
struct Mc {
    v: Param<f64>,
    t: f64,
    hb: SelfBlock<Mc>,
}

fn mc(v: f64, t: f64) -> Mc {
    Mc { v: Param::new(v), t, hb: SelfBlock::new() }
}

fn mc_cost(v: f64, t: f64) -> f64 {
    ((v - t) * 0.8).powi(2)
}

#[arael::model]
#[arael(root)]
struct CMultiMix {
    first: refs::Deque<Mc>,
    second: refs::Arena<Mc>,
    third: std::vec::Vec<Mc>,
}

#[test]
fn multi_containment_across_collection_kinds() {
    // The same entity type in a Deque, an Arena (with a hole), and a
    // std::Vec: one sweep per collection, none dropped.
    let mut first = refs::Deque::new();
    first.push_back(mc(0.5, 0.0));
    let mut second = refs::Arena::new();
    second.push(mc(1.5, 1.0));
    let dead = second.push(mc(9.0, 9.0));
    second.remove(dead).unwrap();
    let mut w = CMultiMix {
        first,
        second,
        third: vec![mc(2.5, 2.0)],
    };
    let manual = mc_cost(0.5, 0.0) + mc_cost(1.5, 1.0) + mc_cost(2.5, 2.0);
    check_model("multi-containment mix", &mut w, manual);
}

#[arael::model]
#[arael(constraint(hb, name = "pull", {
    [(m7.v - m7.t) * 1.0]
}))]
#[arael(constraint(hb, name = "center", {
    [m7.v * 0.1]
}))]
struct M7 {
    v: Param<f64>,
    t: f64,
    hb: SelfBlock<M7>,
}

#[arael::model]
#[arael(root)]
struct CTwoAttrs {
    items: std::vec::Vec<M7>,
    empty: std::vec::Vec<N>,
}

#[test]
fn two_attrs_merge_and_empty_collections_coexist() {
    let mut w = CTwoAttrs {
        items: vec![M7 { v: Param::new(2.0), t: 1.0, hb: SelfBlock::new() }],
        empty: std::vec::Vec::new(),
    };
    let manual = ((2.0f64 - 1.0) * 1.0).powi(2) + (2.0f64 * 0.1).powi(2);
    check_model("two attrs + empty", &mut w, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(f5.a - f5.b) * 1.5, (f5.a - f5.t) * 0.5]
}))]
struct F5 {
    a: Param<f64>,
    b: Param<f64>,
    t: f64,
    hb: SelfBlock<F5>,
}

#[arael::model]
#[arael(root)]
struct CFixed {
    items: std::vec::Vec<F5>,
}

#[test]
fn fixed_params_mixed_with_free_and_a_whole_fixed_entity() {
    let mut w = CFixed {
        items: vec![
            F5 { a: Param::new(1.0), b: Param::fixed(2.0), t: 0.5, hb: SelfBlock::new() },
            F5 { a: Param::fixed(3.0), b: Param::fixed(4.0), t: 1.0, hb: SelfBlock::new() },
            F5 { a: Param::new(5.0), b: Param::new(6.0), t: 2.0, hb: SelfBlock::new() },
        ],
    };
    let f = |a: f64, b: f64, t: f64| ((a - b) * 1.5).powi(2) + ((a - t) * 0.5).powi(2);
    let manual = f(1.0, 2.0, 0.5) + f(3.0, 4.0, 1.0) + f(5.0, 6.0, 2.0);
    check_model("fixed mix", &mut w, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(b.v - a.v - lc.d) * 1.5]
}))]
struct Lc {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    d: f64,
    hb: CrossBlock<N, N>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.v + b.v + c.v - tsum.s) * 1.1]
}))]
struct TSum {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    #[arael(ref = root.nodes)]
    c: Ref<N>,
    s: f64,
    hb: TripletBlock<f64>,
}

#[arael::model]
#[arael(root)]
struct AFrineRootOpt {
    nodes: refs::Vec<N>,
    loop_closure: Option<Lc>,
    sum: Option<TSum>,
}

#[test]
fn root_level_optional_cross_and_triplet() {
    let build_nodes = || {
        let mut nodes = refs::Vec::new();
        let r0 = nodes.push(n(0.1, 0.0));
        let r1 = nodes.push(n(1.2, 1.0));
        let r2 = nodes.push(n(2.3, 2.0));
        (nodes, r0, r1, r2)
    };
    let base = n_cost(0.1, 0.0) + n_cost(1.2, 1.0) + n_cost(2.3, 2.0);

    let (nodes, r0, r1, r2) = build_nodes();
    let mut w = AFrineRootOpt {
        nodes,
        loop_closure: Some(Lc { a: r0, b: r1, d: 1.0, hb: CrossBlock::new() }),
        sum: Some(TSum { a: r0, b: r1, c: r2, s: 3.0, hb: TripletBlock::new() }),
    };
    let manual = base + ((1.2f64 - 0.1 - 1.0) * 1.5).powi(2)
        + ((0.1f64 + 1.2 + 2.3 - 3.0) * 1.1).powi(2);
    check_model("root Option frines Some", &mut w, manual);

    let (nodes, ..) = build_nodes();
    let mut w = AFrineRootOpt { nodes, loop_closure: None, sum: None };
    check_model("root Option frines None", &mut w, base);
}

// The m3500 gauge-anchor shape: ONE optional remote-block prior on the
// root instead of prior fields carried by every entity. The residuals
// write into the referenced entity's own block.
#[arael::model]
#[arael(constraint(p.hb, {
    [(p.v - rprior.z) * 2.0]
}))]
struct RPrior {
    #[arael(ref = root.nodes)]
    p: Ref<N>,
    z: f64,
}

#[arael::model]
#[arael(root)]
struct AFrineRemoteOpt {
    nodes: refs::Vec<N>,
    prior: Option<RPrior>,
}

#[test]
fn root_level_optional_remote_prior() {
    let build_nodes = || {
        let mut nodes = refs::Vec::new();
        let r0 = nodes.push(n(0.4, 0.0));
        nodes.push(n(1.3, 1.0));
        (nodes, r0)
    };
    let base = n_cost(0.4, 0.0) + n_cost(1.3, 1.0);

    let (nodes, r0) = build_nodes();
    let mut w = AFrineRemoteOpt { nodes, prior: Some(RPrior { p: r0, z: 0.1 }) };
    check_model("Option remote prior Some", &mut w, base + ((0.4f64 - 0.1) * 2.0).powi(2));

    let (nodes, _) = build_nodes();
    let mut w = AFrineRemoteOpt { nodes, prior: None };
    check_model("Option remote prior None", &mut w, base);
}

// #[arael(skip)] on an entity collection: fully excluded -- no params
// serialized AND no constraint sweep emitted. (The sweep used to run
// anyway, evaluating never-updated params: cost included phantom
// residuals at garbage values while the derivatives dropped them.)
#[arael::model]
#[arael(root)]
struct ASkip {
    live: std::vec::Vec<P>,
    #[arael(skip)]
    dead: std::vec::Vec<P>,
}

#[test]
fn skipped_collection_is_fully_excluded() {
    let mut w = ASkip {
        live: vec![p(PDATA[0].0, PDATA[0].1, PDATA[0].2)],
        dead: vec![p(9.0, 9.0, 9.0), p(8.0, 8.0, 8.0)],
    };
    let mut x = Vec::new();
    RootProblem::serialize(&mut w, &mut x);
    assert_eq!(x.len(), 2, "skipped params must not serialize");
    check_model("skip collection", &mut w,
        p_cost(PDATA[0].0, PDATA[0].1, PDATA[0].2));
}

// The same rule one level down: a skipped sub-collection inside an
// entity (registry-side discovery must not route a sweep through it).
#[arael::model]
struct SkMid {
    subs: std::vec::Vec<P>,
    #[arael(skip)]
    #[allow(dead_code)]  // never read on purpose: the point is that it is skipped
    dead: std::vec::Vec<P>,
}

#[arael::model]
#[arael(root)]
struct ASkipNested {
    mids: std::vec::Vec<SkMid>,
}

#[test]
fn skipped_nested_collection_is_fully_excluded() {
    let mut w = ASkipNested {
        mids: vec![SkMid {
            subs: vec![p(PDATA[1].0, PDATA[1].1, PDATA[1].2)],
            dead: vec![p(9.0, 9.0, 9.0)],
        }],
    };
    let mut x = Vec::new();
    RootProblem::serialize(&mut w, &mut x);
    assert_eq!(x.len(), 2, "nested skipped params must not serialize");
    check_model("skip nested collection", &mut w,
        p_cost(PDATA[1].0, PDATA[1].1, PDATA[1].2));
}

// Multi-path containment: the same SelfBlock entity type reachable
// through DIFFERENT nested paths, a duplicated intermediate
// collection, and a root-level + nested mix -- one sweep per path.
// (The nested cases used to sweep only the first path found.)

#[arael::model]
struct GroupA {
    subs: std::vec::Vec<P>,
}

#[arael::model]
struct GroupB {
    subs: std::vec::Vec<P>,
}

#[arael::model]
#[arael(root)]
struct ANestedDup {
    ga: std::vec::Vec<GroupA>,
    gb: std::vec::Vec<GroupB>,
}

#[test]
fn same_entity_under_two_nested_paths() {
    let mut w = ANestedDup {
        ga: vec![GroupA { subs: vec![p(PDATA[0].0, PDATA[0].1, PDATA[0].2)] }],
        gb: vec![GroupB { subs: vec![p(PDATA[1].0, PDATA[1].1, PDATA[1].2)] }],
    };
    let manual = p_cost(PDATA[0].0, PDATA[0].1, PDATA[0].2)
        + p_cost(PDATA[1].0, PDATA[1].1, PDATA[1].2);
    check_model("two nested paths", &mut w, manual);
}

#[arael::model]
struct GroupC {
    subs: std::vec::Vec<P>,
}

#[arael::model]
#[arael(root)]
struct ADupIntermediate {
    first: std::vec::Vec<GroupC>,
    second: std::vec::Vec<GroupC>,
}

#[test]
fn same_entity_under_a_duplicated_intermediate() {
    // GroupC itself is legally duplicated (SelfBlock-less grouping
    // struct in two collections); P below it must sweep through BOTH.
    let mut w = ADupIntermediate {
        first: vec![GroupC { subs: vec![p(PDATA[0].0, PDATA[0].1, PDATA[0].2)] }],
        second: vec![GroupC { subs: vec![p(PDATA[1].0, PDATA[1].1, PDATA[1].2)],
                    }, GroupC { subs: vec![p(PDATA[2].0, PDATA[2].1, PDATA[2].2)] }],
    };
    check_model("duplicated intermediate", &mut w, pdata_cost());
}

#[arael::model]
struct GroupD {
    subs: std::vec::Vec<P>,
}

#[arael::model]
#[arael(root)]
struct AMixedDepth {
    direct: std::vec::Vec<P>,
    groups: std::vec::Vec<GroupD>,
}

#[test]
fn same_entity_at_root_level_and_nested() {
    let mut w = AMixedDepth {
        direct: vec![p(PDATA[0].0, PDATA[0].1, PDATA[0].2)],
        groups: vec![GroupD { subs: vec![p(PDATA[1].0, PDATA[1].1, PDATA[1].2)] }],
    };
    let manual = p_cost(PDATA[0].0, PDATA[0].1, PDATA[0].2)
        + p_cost(PDATA[1].0, PDATA[1].1, PDATA[1].2);
    check_model("root + nested mix", &mut w, manual);
}

// Containment paths crossing an Option EDGE: a collection under an
// Option intermediate, and an Option entity as the nested last
// segment. Option segments iterate as zero-or-one collections, so a
// None along the path contributes nothing.

#[arael::model]
struct OptSub {
    items: std::vec::Vec<P>,
}

#[arael::model]
#[arael(root)]
struct AOptIntermediate {
    maybe: Option<OptSub>,
}

#[test]
fn collection_under_an_option_intermediate() {
    let mut w = AOptIntermediate {
        maybe: Some(OptSub { items: vec![
            p(PDATA[0].0, PDATA[0].1, PDATA[0].2),
            p(PDATA[1].0, PDATA[1].1, PDATA[1].2),
        ]}),
    };
    let manual = p_cost(PDATA[0].0, PDATA[0].1, PDATA[0].2)
        + p_cost(PDATA[1].0, PDATA[1].1, PDATA[1].2);
    check_model("Option intermediate Some", &mut w, manual);

    let mut w = AOptIntermediate { maybe: None };
    let mut x = Vec::new();
    RootProblem::serialize(&mut w, &mut x);
    assert_eq!(x.len(), 0, "None path: nothing serialized, nothing swept");
}

#[arael::model]
struct OptHolder {
    maybe_p: Option<P>,
}

#[arael::model]
#[arael(root)]
struct ANestedOption {
    groups: std::vec::Vec<OptHolder>,
}

#[test]
fn option_entity_as_nested_last_segment() {
    let mut w = ANestedOption {
        groups: vec![
            OptHolder { maybe_p: Some(p(PDATA[0].0, PDATA[0].1, PDATA[0].2)) },
            OptHolder { maybe_p: None },
            OptHolder { maybe_p: Some(p(PDATA[1].0, PDATA[1].1, PDATA[1].2)) },
        ],
    };
    let manual = p_cost(PDATA[0].0, PDATA[0].1, PDATA[0].2)
        + p_cost(PDATA[1].0, PDATA[1].1, PDATA[1].2);
    check_model("nested Option entity", &mut w, manual);
}

#[arael::model]
struct TieBundle {
    ties: std::vec::Vec<Lc>,
}

#[arael::model]
#[arael(root)]
struct AOptFrines {
    nodes: refs::Vec<N>,
    maybe: Option<TieBundle>,
}

#[test]
fn frines_under_an_option_intermediate() {
    let mut nodes = refs::Vec::new();
    let r0 = nodes.push(n(0.1, 0.0));
    let r1 = nodes.push(n(1.3, 1.0));
    let base = n_cost(0.1, 0.0) + n_cost(1.3, 1.0);

    let mut w = AOptFrines { nodes,
        maybe: Some(TieBundle { ties: vec![Lc { a: r0, b: r1, d: 1.0, hb: CrossBlock::new() }] }) };
    check_model("frines under Option Some", &mut w,
        base + ((1.3f64 - 0.1 - 1.0) * 1.5).powi(2));

    let mut nodes = refs::Vec::new();
    nodes.push(n(0.1, 0.0));
    nodes.push(n(1.3, 1.0));
    let mut w = AOptFrines { nodes, maybe: None };
    check_model("frines under Option None", &mut w, base);
}

// ---------------------------------------------------------------------------
// #[arael(skip)] on an aliased container: containers are recognized by
// literal type name, so `AliasVec<P>` is not containment -- without the
// skip the expansion rejects it (see constraint_attr_errors). The skip
// documents a deliberate out-of-model holding; the field must stay
// completely inert: not serialized, never swept.

use arael::refs::Vec as AliasVec;

#[arael::model]
#[arael(root)]
struct AAliasSkip {
    items: refs::Vec<P>,
    #[arael(skip)]
    #[allow(dead_code)]  // never read on purpose: the point is that it is skipped
    stash: AliasVec<P>,
}

// ---------------------------------------------------------------------------
// Remote-block frines (primary = a Ref target's SelfBlock) on Deque and
// Arena roots: only Vec-backed targets were exercised before.

#[arael::model]
#[arael(constraint(a.hb, {
    [(a.v - remd.m) * 0.9]
}))]
struct RemD {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    m: f64,
}

#[arael::model]
#[arael(root)]
struct CRemDeque {
    nodes: refs::Deque<N>,
    rems: std::vec::Vec<RemD>,
}

fn rem_cost(v: f64, m: f64) -> f64 {
    ((v - m) * 0.9).powi(2)
}

#[test]
fn remote_block_into_deque_targets() {
    let mut nodes = refs::Deque::new();
    let r1 = nodes.push_back(n(1.3, 1.0));
    let r0 = nodes.push_front(n(0.1, 0.0));
    let rems = vec![RemD { a: r0, m: 0.2 }, RemD { a: r1, m: 1.1 }];
    let manual = n_cost(0.1, 0.0) + n_cost(1.3, 1.0)
        + rem_cost(0.1, 0.2) + rem_cost(1.3, 1.1);
    check_model("remote Deque", &mut CRemDeque { nodes, rems }, manual);
}

#[arael::model]
#[arael(constraint(a.hb, {
    [(a.v - rema.m) * 0.9]
}))]
struct RemA {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    m: f64,
}

#[arael::model]
#[arael(root)]
struct CRemArena {
    nodes: refs::Arena<N>,
    rems: std::vec::Vec<RemA>,
}

#[test]
fn remote_block_into_arena_targets_with_a_hole() {
    let mut nodes = refs::Arena::new();
    let r0 = nodes.push(n(0.1, 0.0));
    let dead = nodes.push(n(9.0, 9.0));
    let r2 = nodes.push(n(2.2, 2.0));
    nodes.remove(dead).unwrap();
    let rems = vec![RemA { a: r0, m: 0.2 }, RemA { a: r2, m: 2.1 }];
    let manual = n_cost(0.1, 0.0) + n_cost(2.2, 2.0)
        + rem_cost(0.1, 0.2) + rem_cost(2.2, 2.1);
    check_model("remote Arena", &mut CRemArena { nodes, rems }, manual);
}

// Multi-cross frines (three refs, one CrossBlock per pair) on Deque and
// Arena roots.

fn link_cost(a: f64, b: f64, c: f64, d1: f64, d2: f64) -> f64 {
    ((b - a - d1) * 1.5).powi(2) + ((c - b - d2) * 0.8).powi(2)
}

#[arael::model]
#[arael(constraint([hb_ab, hb_ac, hb_bc], {
    [(b.v - a.v - linkd.d1) * 1.5,
     (cc.v - b.v - linkd.d2) * 0.8]
}))]
struct LinkD {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    #[arael(ref = root.nodes)]
    cc: Ref<N>,
    d1: f64,
    d2: f64,
    #[arael(cross = (a, b))]
    hb_ab: CrossBlock<N, N>,
    #[arael(cross = (a, cc))]
    hb_ac: CrossBlock<N, N>,
    #[arael(cross = (b, cc))]
    hb_bc: CrossBlock<N, N>,
}

#[arael::model]
#[arael(root)]
struct CMcDeque {
    nodes: refs::Deque<N>,
    links: std::vec::Vec<LinkD>,
}

#[test]
fn multi_cross_on_a_deque_root() {
    let mut nodes = refs::Deque::new();
    let r1 = nodes.push_back(n(1.3, 1.0));
    let r2 = nodes.push_back(n(2.2, 2.0));
    let r0 = nodes.push_front(n(0.1, 0.0));
    let links = vec![LinkD { a: r0, b: r1, cc: r2, d1: 1.0, d2: 1.0,
        hb_ab: CrossBlock::new(), hb_ac: CrossBlock::new(), hb_bc: CrossBlock::new() }];
    let manual = n_cost(0.1, 0.0) + n_cost(1.3, 1.0) + n_cost(2.2, 2.0)
        + link_cost(0.1, 1.3, 2.2, 1.0, 1.0);
    check_model("multi-cross Deque", &mut CMcDeque { nodes, links }, manual);
}

#[arael::model]
#[arael(constraint([hb_ab, hb_ac, hb_bc], {
    [(b.v - a.v - linka.d1) * 1.5,
     (cc.v - b.v - linka.d2) * 0.8]
}))]
struct LinkA {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    #[arael(ref = root.nodes)]
    cc: Ref<N>,
    d1: f64,
    d2: f64,
    #[arael(cross = (a, b))]
    hb_ab: CrossBlock<N, N>,
    #[arael(cross = (a, cc))]
    hb_ac: CrossBlock<N, N>,
    #[arael(cross = (b, cc))]
    hb_bc: CrossBlock<N, N>,
}

#[arael::model]
#[arael(root)]
struct CMcArena {
    nodes: refs::Arena<N>,
    links: std::vec::Vec<LinkA>,
}

#[test]
fn multi_cross_on_an_arena_root_with_a_hole() {
    let mut nodes = refs::Arena::new();
    let r0 = nodes.push(n(0.1, 0.0));
    let dead = nodes.push(n(9.0, 9.0));
    let r1 = nodes.push(n(1.3, 1.0));
    let r2 = nodes.push(n(2.2, 2.0));
    nodes.remove(dead).unwrap();
    let links = vec![LinkA { a: r0, b: r1, cc: r2, d1: 1.0, d2: 1.0,
        hb_ab: CrossBlock::new(), hb_ac: CrossBlock::new(), hb_bc: CrossBlock::new() }];
    let manual = n_cost(0.1, 0.0) + n_cost(1.3, 1.0) + n_cost(2.2, 2.0)
        + link_cost(0.1, 1.3, 2.2, 1.0, 1.0);
    check_model("multi-cross Arena", &mut CMcArena { nodes, links }, manual);
}

// ---------------------------------------------------------------------------
// Reads through an Option sub-struct: guarded reads never evaluate on
// None; an unguarded read panics naming the field as the body spells it
// (the contract in MODEL.md, "Guards and optional data").

#[arael::model]
struct OptData {
    off: f64,
}

#[arael::model]
#[arael(constraint(hb, guard = self.has_extra, {
    [(pg.x - pg.extra.off) * 2.0]
}))]
#[arael(constraint(hb, {
    [(pg.x - pg.t) * 0.5]
}))]
struct Pg {
    x: Param<f64>,
    t: f64,
    has_extra: bool,
    extra: Option<OptData>,
    hb: SelfBlock<Pg>,
}

#[arael::model]
#[arael(root)]
struct AOptRead {
    items: std::vec::Vec<Pg>,
}

#[test]
fn guarded_option_read_skips_none() {
    let mut w = AOptRead { items: vec![
        Pg { x: Param::new(1.0), t: 0.2, has_extra: true,
             extra: Some(OptData { off: 0.4 }), hb: SelfBlock::new() },
        Pg { x: Param::new(2.0), t: 1.1, has_extra: false,
             extra: None, hb: SelfBlock::new() },
    ]};
    let manual = ((1.0f64 - 0.4) * 2.0).powi(2) + ((1.0f64 - 0.2) * 0.5).powi(2)
        + ((2.0f64 - 1.1) * 0.5).powi(2);
    check_model("guarded Option read", &mut w, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(pu.x - pu.extra.off) * 2.0]
}))]
struct Pu {
    x: Param<f64>,
    extra: Option<OptData>,
    hb: SelfBlock<Pu>,
}

#[arael::model]
#[arael(root)]
struct AOptReadBare {
    items: std::vec::Vec<Pu>,
}

#[test]
#[should_panic(expected = "optional `pu.extra` is None -- guard the constraint")]
fn unguarded_option_read_panics_with_field_name() {
    let mut w = AOptReadBare { items: vec![
        Pu { x: Param::new(1.0), extra: None, hb: SelfBlock::new() },
    ]};
    let mut x = Vec::new();
    RootProblem::serialize(&mut w, &mut x);
    let _ = w.calc_cost(&x);
}

// A skipped stash must not leak into the root's reachable set: Nf32 has
// f32 blocks, and the precision check would reject this f64 root if the
// seed followed skipped fields (regression: it used to).
#[arael::model]
#[arael(constraint(hb, {
    [(nf32.v - nf32.t) * 0.3]
}))]
struct Nf32 {
    v: Param<f32>,
    t: f32,
    hb: SelfBlock<Nf32, f32>,
}

#[arael::model]
#[arael(root)]
struct ASkipStash {
    items: std::vec::Vec<P>,
    #[arael(skip)]
    #[allow(dead_code)]  // never read on purpose: the point is that it is skipped
    stash: std::vec::Vec<Nf32>,
}

#[test]
fn skipped_stash_stays_outside_the_reachable_set() {
    let mut w = ASkipStash {
        items: PDATA.iter().map(|&(x, y, t)| p(x, y, t)).collect(),
        stash: vec![Nf32 { v: Param::new(1.0), t: 0.5, hb: SelfBlock::new() }],
    };
    check_model("skip stash", &mut w, pdata_cost());
}

#[test]
fn skipped_alias_container_is_inert() {
    let mut items = refs::Vec::new();
    let mut stash = AliasVec::new();
    for &(x, y, t) in &PDATA {
        items.push(p(x, y, t));
        stash.push(p(x + 10.0, y - 3.0, t));
    }
    let mut w = AAliasSkip { items, stash };
    check_model("skipped alias container", &mut w, pdata_cost());
}
