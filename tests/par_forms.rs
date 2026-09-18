// The model forms the threaded sweeps used to refuse.
//
// `par` once rejected a list of shapes at compile time, because the mirror
// path re-derived the model in a second vocabulary and only covered some of
// it. The sweeps walk the model's own structure now, so there is nothing
// left to reject -- but nothing exercised these shapes threaded either.
// One root per shape here, each assembled at one store and at four and
// compared against the sequential answer.
//
// Self blocks are summed per range and then across ranges, so they match to
// rounding rather than to the bit; cross tiles have one writer and match
// exactly.
#![cfg(feature = "rayon")]

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblemInternals, RootProblem};
use arael::threads::Context;
use arael::vect::vect2d;

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * (1.0 + a.abs().max(b.abs()))
}

fn assert_close(what: &str, a: &[f64], b: &[f64], rel: f64) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(close(*x, *y, rel), "{what} differs at {i}: {x} vs {y}");
    }
}

/// Assemble `build()` sequentially and over four stores, and compare. The
/// store count is checked too: a form that silently fell back to one store
/// would pass every numeric assertion while threading nothing.
fn threaded_matches_sequential<R, B, S>(what: &str, build: B, stores: &S)
where
    R: RootProblem<f64> + LmProblemInternals<f64>,
    B: Fn() -> R,
    S: Fn(&Context) -> usize,
{
    let (mut seq, mut par) = (build(), build());
    let (mut x, mut x2) = (Vec::new(), Vec::new());
    seq.serialize(&mut x);
    par.serialize(&mut x2);
    assert_eq!(x, x2, "{what}: the two builds differ");
    assert!(!x.is_empty(), "{what}: no parameters to assemble");

    let mut ctx = Context::new();
    ctx.set_threads(4);
    par.begin_with_context(&mut ctx);
    assert_eq!(stores(&ctx), 4, "{what}: did not split into four stores");

    let n = x.len();
    let (mut g1, mut h1) = (vec![0.0; n], vec![0.0; n * n]);
    let (mut g2, mut h2) = (vec![0.0; n], vec![0.0; n * n]);
    let c1 = seq.calc_grad_hessian_dense(&x, &mut g1, &mut h1);
    let c2 = par.calc_grad_hessian_dense_with_context(&x, &mut g2, &mut h2, &mut ctx);
    assert!(close(c1, c2, 1e-12), "{what}: cost {c1} vs {c2}");
    assert_close(&format!("{what}: grad"), &g1, &g2, 1e-11);
    assert_close(&format!("{what}: hessian"), &h1, &h2, 1e-11);

    // And the same again through the context, to catch a walk that only
    // works on a freshly built store.
    let (mut g3, mut h3) = (vec![0.0; n], vec![0.0; n * n]);
    let c3 = par.calc_grad_hessian_dense_with_context(&x, &mut g3, &mut h3, &mut ctx);
    assert_eq!(c2, c3, "{what}: repeat cost");
    assert_eq!(g2, g3, "{what}: repeat grad");
    assert_eq!(h2, h3, "{what}: repeat hessian");
}

/// A full sparse solve at one thread and at four must land in the same place.
fn threaded_solve_matches<R, B>(what: &str, build: B)
where
    R: RootProblem<f64> + LmProblemInternals<f64>,
    B: Fn() -> R,
{
    let cfg = |t: usize| LmConfig::<f64> { max_iters: 60, num_threads: t, ..Default::default() };
    let s = build().solve_sparse(&cfg(1)).unwrap();
    let p = build().solve_sparse(&cfg(4)).unwrap();
    assert!(close(s.end_cost, p.end_cost, 1e-9),
        "{what}: end cost {} vs {}", s.end_cost, p.end_cost);
    assert_close(&format!("{what}: x"), &s.x, &p.x, 1e-6);
}

// ===========================================================================
// A constraint collection below the root, with `parent.` and `root.` refs
// ===========================================================================

#[arael::model]
struct Node {
    x: Param<f64>,
    hb: SelfBlock<Node>,
}

#[arael::model]
struct Anchor {
    x: Param<f64>,
    hb: SelfBlock<Anchor>,
}

// Two nodes of the SAME path: a `parent.`-relative ref, two hops from root.
#[arael::model]
#[arael(constraint(hb, {
    [(cur.x - prev.x - link.measured) * link.isigma]
}))]
struct Link {
    #[arael(ref = parent.nodes)] prev: Ref<Node>,
    #[arael(ref = parent.nodes)] cur: Ref<Node>,
    measured: f64,
    isigma: f64,
    hb: CrossBlock<Node, Node>,
}

// Cross-level: a nested node against a root-held anchor.
#[arael::model]
#[arael(constraint(hb, {
    [(anchor.x - node.x - obs.measured) * obs.isigma]
}))]
struct Obs {
    #[arael(ref = parent.nodes)] node: Ref<Node>,
    #[arael(ref = root.anchors)] anchor: Ref<Anchor>,
    measured: f64,
    isigma: f64,
    hb: CrossBlock<Anchor, Node>,
}

#[arael::model]
struct Path {
    nodes: refs::Vec<Node>,
    links: std::vec::Vec<Link>,
    obs: std::vec::Vec<Obs>,
}

#[arael::model]
#[arael(root, par)]
struct Nested {
    paths: std::vec::Vec<Path>,
    anchors: refs::Vec<Anchor>,
}

fn build_nested() -> Nested {
    let anchors_truth = [0.0_f64, 10.0];
    let mut anchors = refs::Vec::new();
    let mut anchor_refs = Vec::new();
    for &a in &anchors_truth {
        let mut anc = Anchor { x: Param::new(a), hb: SelfBlock::new() };
        anc.x.optimize = false; // fixed reference, pinning the gauge
        anchor_refs.push(anchors.push(anc));
    }
    // Enough paths that a cut into four has something to divide.
    let mut paths = std::vec::Vec::new();
    for p in 0..8 {
        let truth: Vec<f64> = (0..6).map(|i| 1.0 + 0.7 * p as f64 + 1.3 * i as f64).collect();
        let mut path = Path {
            nodes: refs::Vec::new(),
            links: std::vec::Vec::new(),
            obs: std::vec::Vec::new(),
        };
        for _ in &truth {
            path.nodes.push(Node { x: Param::new(0.0), hb: SelfBlock::new() });
        }
        for i in 1..truth.len() {
            path.links.push(Link {
                prev: path.nodes.ref_at(i - 1),
                cur: path.nodes.ref_at(i),
                measured: truth[i] - truth[i - 1],
                isigma: 1.0,
                hb: CrossBlock::new(),
            });
        }
        for (node_i, anchor_i) in [(0usize, 0usize), (5, 1)] {
            path.obs.push(Obs {
                node: path.nodes.ref_at(node_i),
                anchor: anchor_refs[anchor_i],
                measured: anchors_truth[anchor_i] - truth[node_i],
                isigma: 1.0,
                hb: CrossBlock::new(),
            });
        }
        paths.push(path);
    }
    Nested { paths, anchors }
}

#[test]
fn a_nested_collection_threads() {
    threaded_matches_sequential("nested", build_nested,
        &|c: &Context| c.blocks_list::<NestedBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("nested", build_nested);
}

// ===========================================================================
// A parent-owned cross block: the constraint carries no block of its own and
// writes the one its containing parent holds
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, { [(pnode.x - pnode.prior) * 0.05] }))]
struct PNode {
    x: Param<f64>,
    prior: f64,
    hb: SelfBlock<PNode>,
}

#[arael::model]
#[arael(constraint(parent.hb, parent = pair, guard = self.on, {
    [(pair.b.x - parent.a.x - plink.dx) * plink.w]
}))]
struct PLink {
    dx: f64,
    w: f64,
    on: bool,
}

#[arael::model]
struct PPair {
    #[arael(ref = root.pnodes)] a: Ref<PNode>,
    #[arael(ref = root.pnodes)] b: Ref<PNode>,
    links: std::vec::Vec<PLink>,
    hb: CrossBlock<PNode, PNode>,
}

#[arael::model]
#[arael(root, par)]
struct ParentCross {
    pnodes: refs::Vec<PNode>,
    pairs: std::vec::Vec<PPair>,
}

fn build_parent_cross() -> ParentCross {
    let mut m = ParentCross { pnodes: refs::Vec::new(), pairs: std::vec::Vec::new() };
    for i in 0..24 {
        m.pnodes.push(PNode {
            x: Param::new(0.3 * i as f64),
            prior: 0.25 * i as f64,
            hb: SelfBlock::new(),
        });
    }
    for i in 1..24 {
        let links = (0..3).map(|k| PLink {
            dx: 0.25 + 0.01 * k as f64,
            w: 1.0 + 0.1 * k as f64,
            // a guarded instance, so the walk skips some
            on: (i + k) % 5 != 0,
        }).collect();
        m.pairs.push(PPair {
            a: m.pnodes.ref_at(i - 1),
            b: m.pnodes.ref_at(i),
            links,
            hb: CrossBlock::new(),
        });
    }
    m
}

#[test]
fn a_parent_owned_cross_block_threads() {
    threaded_matches_sequential("parent cross", build_parent_cross,
        &|c: &Context| c.blocks_list::<ParentCrossBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("parent cross", build_parent_cross);
}

// ===========================================================================
// A `root.` SelfBlock primary: every instance writes the SAME root block, so
// all four stores accumulate into one entity and the gather adds them
// ===========================================================================

#[arael::model]
#[arael(constraint(root.hb, { [(shared.m - root.shift) * shared.w] }))]
struct Shared {
    m: f64,
    w: f64,
}

#[arael::model]
#[arael(root, par)]
struct RootSelf {
    shift: Param<f64>,
    obs: std::vec::Vec<Shared>,
    hb: SelfBlock<RootSelf>,
}

fn build_root_self() -> RootSelf {
    RootSelf {
        shift: Param::new(0.0),
        obs: (0..400).map(|k| Shared { m: 1.0 + 0.01 * k as f64, w: 1.0 + 0.001 * k as f64 }).collect(),
        hb: SelfBlock::new(),
    }
}

#[test]
fn a_root_selfblock_primary_threads() {
    threaded_matches_sequential("root selfblock", build_root_self,
        &|c: &Context| c.blocks_list::<RootSelfBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("root selfblock", build_root_self);
}

// ===========================================================================
// A `constraint_index` field: each instance is told which residual row it got
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, { [(cinode.v - cinode.target) * 0.5] }))]
struct CiNode {
    v: Param<f64>,
    target: f64,
    #[arael(constraint_index)]
    ci: u32,
    hb: SelfBlock<CiNode>,
}

#[arael::model]
#[arael(constraint(hb, { [(b.v - a.v - cilink.d) * 0.7] }))]
struct CiLink {
    #[arael(ref = root.nodes)] a: Ref<CiNode>,
    #[arael(ref = root.nodes)] b: Ref<CiNode>,
    d: f64,
    #[arael(constraint_index)]
    ci: u32,
    hb: CrossBlock<CiNode, CiNode>,
}

#[arael::model]
#[arael(root, par)]
struct CiRoot {
    nodes: refs::Vec<CiNode>,
    links: std::vec::Vec<CiLink>,
}

fn build_ci() -> CiRoot {
    let mut m = CiRoot { nodes: refs::Vec::new(), links: std::vec::Vec::new() };
    for i in 0..32 {
        m.nodes.push(CiNode {
            v: Param::new(0.2 * i as f64),
            target: 0.15 * i as f64,
            ci: 0,
            hb: SelfBlock::new(),
        });
    }
    for i in 1..32 {
        m.links.push(CiLink {
            a: m.nodes.ref_at(i - 1),
            b: m.nodes.ref_at(i),
            d: 0.15,
            ci: 0,
            hb: CrossBlock::new(),
        });
    }
    m
}

#[test]
fn a_constraint_index_threads() {
    threaded_matches_sequential("constraint_index", build_ci,
        &|c: &Context| c.blocks_list::<CiRootBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("constraint_index", build_ci);
}

// ===========================================================================
// One entity type held in two collections, with a constraint spanning them
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, { [(tnode.x - tnode.prior) * 0.1] }))]
struct TNode {
    x: Param<f64>,
    prior: f64,
    hb: SelfBlock<TNode>,
}

#[arael::model]
#[arael(constraint(hb, { [(r.x - l.x - span.d) * 0.9] }))]
struct Span {
    #[arael(ref = root.left)] l: Ref<TNode>,
    #[arael(ref = root.right)] r: Ref<TNode>,
    d: f64,
    hb: CrossBlock<TNode, TNode>,
}

#[arael::model]
#[arael(root, par)]
struct TwoColl {
    left: refs::Vec<TNode>,
    right: refs::Vec<TNode>,
    spans: std::vec::Vec<Span>,
}

fn build_two_coll() -> TwoColl {
    let node = |i: usize, o: f64| TNode {
        x: Param::new(o + 0.3 * i as f64),
        prior: o + 0.25 * i as f64,
        hb: SelfBlock::new(),
    };
    let mut m = TwoColl {
        left: refs::Vec::new(),
        right: refs::Vec::new(),
        spans: std::vec::Vec::new(),
    };
    for i in 0..20 { m.left.push(node(i, 0.0)); }
    for i in 0..20 { m.right.push(node(i, 5.0)); }
    for i in 0..20 {
        m.spans.push(Span {
            l: m.left.ref_at(i),
            r: m.right.ref_at((i + 3) % 20),
            d: 5.0,
            hb: CrossBlock::new(),
        });
    }
    m
}

#[test]
fn an_entity_in_two_collections_threads() {
    threaded_matches_sequential("two collections", build_two_coll,
        &|c: &Context| c.blocks_list::<TwoCollBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("two collections", build_two_coll);
}

// ===========================================================================
// A `parent.` SelfBlock primary: a child collection writes the self block of
// the entity that holds it
// ===========================================================================

#[arael::model]
#[arael(constraint(parent.hb, {
    [curveobs.y - (curve.m * curveobs.x + curve.c)]
}))]
struct CurveObs {
    x: f64,
    y: f64,
}

#[arael::model]
struct Curve {
    m: Param<f64>,
    c: Param<f64>,
    obs: std::vec::Vec<CurveObs>,
    hb: SelfBlock<Curve>,
}

#[arael::model]
#[arael(root, par)]
struct CurveFit {
    curves: refs::Vec<Curve>,
}

fn build_curve_fit() -> CurveFit {
    let mut curves = refs::Vec::new();
    for i in 0..20 {
        let (m, c) = (0.3 + 0.05 * i as f64, -0.2 + 0.1 * i as f64);
        let obs = (0..10).map(|k| {
            let x = 0.5 * k as f64;
            // a little structure so the fit is not exact
            CurveObs { x, y: m * x + c + 0.01 * ((i * 3 + k * 7) % 5) as f64 }
        }).collect();
        curves.push(Curve { m: Param::new(0.0), c: Param::new(0.0), obs, hb: SelfBlock::new() });
    }
    CurveFit { curves }
}

#[test]
fn a_parent_selfblock_primary_threads() {
    threaded_matches_sequential("parent selfblock", build_curve_fit,
        &|c: &Context| c.blocks_list::<CurveFitBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("parent selfblock", build_curve_fit);
}

// ===========================================================================
// A ref resolving through a chained path: the link's tag refs point into an
// arena each node owns, reached through the parent's own node refs
// ===========================================================================

#[arael::model]
struct KTag {
    off: f64,
    w: f64,
}

#[arael::model]
#[arael(constraint(hb, { [(knode.x - knode.px) * knode.pw] }))]
struct KNode {
    x: Param<f64>,
    px: f64,
    pw: f64,
    tags: refs::Arena<KTag>,
    hb: SelfBlock<KNode>,
}

#[arael::model]
#[arael(constraint(parent.hb, parent = pp, guard = self.on && ta.w > 0.0, {
    [(pp.b.x - parent.a.x - klink.d + ta.off - tb.off) * (ta.w * tb.w)]
}))]
struct KLink {
    #[arael(ref = parent.a.tags)] ta: Ref<KTag>,
    #[arael(ref = parent.b.tags)] tb: Ref<KTag>,
    d: f64,
    on: bool,
}

#[arael::model]
struct KPair {
    #[arael(ref = root.nodes)] a: Ref<KNode>,
    #[arael(ref = root.nodes)] b: Ref<KNode>,
    links: std::vec::Vec<KLink>,
    hb: CrossBlock<KNode, KNode>,
}

#[arael::model]
#[arael(root, par)]
struct ChainRoot {
    nodes: refs::Arena<KNode>,
    pairs: std::vec::Vec<KPair>,
}

fn build_chain() -> ChainRoot {
    let mut m = ChainRoot { nodes: refs::Arena::new(), pairs: std::vec::Vec::new() };
    let mut nrefs = Vec::new();
    for i in 0..24 {
        let mut tags = refs::Arena::new();
        for t in 0..3 {
            tags.push(KTag { off: 0.01 * (i + t) as f64, w: if (i + t) % 7 == 0 { -1.0 } else { 1.0 } });
        }
        nrefs.push(m.nodes.push(KNode {
            x: Param::new(0.3 * i as f64),
            px: 0.25 * i as f64,
            pw: 0.1,
            tags,
            hb: SelfBlock::new(),
        }));
    }
    for i in 1..24 {
        let (a, b) = (nrefs[i - 1], nrefs[i]);
        let ta: Vec<_> = m.nodes[a].tags.iter_refs().map(|(r, _)| r).collect();
        let tb: Vec<_> = m.nodes[b].tags.iter_refs().map(|(r, _)| r).collect();
        let links = (0..3).map(|k| KLink {
            ta: ta[k],
            tb: tb[(k + 1) % 3],
            d: 0.25,
            on: (i + k) % 5 != 0,
        }).collect();
        m.pairs.push(KPair { a, b, links, hb: CrossBlock::new() });
    }
    m
}

#[test]
fn a_ref_through_a_chained_path_threads() {
    threaded_matches_sequential("chained ref", build_chain,
        &|c: &Context| c.blocks_list::<ChainRootBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("chained ref", build_chain);
}

// ===========================================================================
// COLMAP's bundle-adjustment shape, all at once: a pose holds its images and
// an image its observations, three levels down; the observation writes a
// block list mixing its own cross blocks with one its image owns, reaches
// the pose through `parent.parent`, reads the camera through a ref its
// parent holds, and is guarded and robustified. Points and cameras are
// written from observations under many poses, so across many stores.
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, { [(ccam.f - ccam.pf) * 0.5] }))]
struct CCam {
    f: Param<f64>,
    pf: f64,
    hb: SelfBlock<CCam>,
}

#[arael::model]
#[arael(constraint(hb, { [(cpoint.p.x - cpoint.px) * 0.2, (cpoint.p.y - cpoint.py) * 0.2] }))]
struct CPoint {
    p: Param<vect2d>,
    px: f64,
    py: f64,
    hb: SelfBlock<CPoint>,
}

#[arael::model]
#[arael(constraint(hb, { [(cpose.t.x - cpose.px) * 0.3, (cpose.t.y - cpose.py) * 0.3] }))]
struct CPose {
    t: Param<vect2d>,
    px: f64,
    py: f64,
    w: f64,
    images: std::vec::Vec<CImage>,
    hb: SelfBlock<CPose>,
}

#[arael::model]
struct CImage {
    #[arael(ref = root.cams)] cam: Ref<CCam>,
    obs: std::vec::Vec<CObs>,
    hb_pose_cam: CrossBlock<CPose, CCam>,
}

#[arael::model]
#[arael(constraint([hb_point_pose, hb_point_cam, parent.hb_pose_cam],
    parent = image, parent.parent = pose,
    guard = self.on && parent.parent.w > 0.0, loss = |s| loss_huber(s, cobs.k2), {
    [image.cam.f * (point.p.x - pose.t.x) * pose.w - cobs.m.x,
     image.cam.f * (point.p.y - pose.t.y) * pose.w - cobs.m.y]
}))]
struct CObs {
    #[arael(ref = root.points)] point: Ref<CPoint>,
    m: vect2d,
    k2: f64,
    on: bool,
    hb_point_pose: CrossBlock<CPoint, CPose>,
    hb_point_cam: CrossBlock<CPoint, CCam>,
}

#[arael::model]
#[arael(root, par, marginalize(points))]
struct ColmapLike {
    poses: refs::Vec<CPose>,
    cams: refs::Vec<CCam>,
    points: refs::Vec<CPoint>,
}

fn build_colmap_like() -> ColmapLike {
    const NP: usize = 24;
    const NC: usize = 3;
    const NK: usize = 40;
    let mut cams = refs::Vec::new();
    for j in 0..NC {
        let f = [1.5, 0.8, 1.1][j];
        cams.push(CCam { f: Param::new(f * 1.1), pf: f, hb: SelfBlock::new() });
    }
    let mut points = refs::Vec::new();
    for k in 0..NK {
        let (px, py) = (1.0 + 0.3 * k as f64, 0.5 - 0.05 * k as f64);
        points.push(CPoint {
            p: Param::new(vect2d::new(px - 0.06, py + 0.04)),
            px, py,
            hb: SelfBlock::new(),
        });
    }
    let crefs: Vec<Ref<CCam>> = cams.iter_refs().map(|(r, _)| r).collect();
    let krefs: Vec<Ref<CPoint>> = points.iter_refs().map(|(r, _)| r).collect();
    let mut poses = refs::Vec::new();
    for i in 0..NP {
        let (tx, ty) = (0.4 * i as f64, -0.2 * i as f64 + 0.1);
        // one pose switched off by the guard through the grandparent
        let w = if i == 7 { -1.0 } else { 1.0 + 0.01 * i as f64 };
        let mut images = Vec::new();
        for j in 0..NC {
            if (i + j) % 11 == 0 { continue; } // some poses lack a camera
            let f = [1.5, 0.8, 1.1][j];
            // every image sees a window of the points, so a point is
            // observed from many poses and lands in many stores
            let obs = (0..12).map(|s| {
                let k = (i * 2 + j * 5 + s) % NK;
                let (px, py) = (1.0 + 0.3 * k as f64, 0.5 - 0.05 * k as f64);
                let noise = 0.05 * (((i * 5 + j * 3 + k * 7) % 9) as f64 - 4.0);
                CObs {
                    point: krefs[k],
                    m: vect2d::new(f * (px - tx) * w + noise, f * (py - ty) * w - 0.5 * noise),
                    k2: 0.09,
                    on: (i * 7 + j * 3 + k) % 5 != 0,
                    hb_point_pose: CrossBlock::new(),
                    hb_point_cam: CrossBlock::new(),
                }
            }).collect();
            images.push(CImage { cam: crefs[j], obs, hb_pose_cam: CrossBlock::new() });
        }
        poses.push(CPose {
            t: Param::new(vect2d::new(tx + 0.07, ty - 0.05)),
            px: tx, py: ty, w, images,
            hb: SelfBlock::new(),
        });
    }
    ColmapLike { poses, cams, points }
}

#[test]
fn colmaps_bundle_adjustment_shape_threads() {
    threaded_matches_sequential("colmap shape", build_colmap_like,
        &|c: &Context| c.blocks_list::<ColmapLikeBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("colmap shape", build_colmap_like);
}

// ===========================================================================
// An entity in a NESTED collection with a self-block constraint of its own:
// its block is written by its own walk, two levels down
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, { [(leaf.v - leaf.target) * 0.4] }))]
struct Leaf {
    v: Param<f64>,
    target: f64,
    hb: SelfBlock<Leaf>,
}

#[arael::model]
struct Branch {
    leaves: refs::Vec<Leaf>,
}

#[arael::model]
#[arael(root, par)]
struct Tree {
    branches: std::vec::Vec<Branch>,
}

fn build_tree() -> Tree {
    let branches = (0..16).map(|b| {
        let mut leaves = refs::Vec::new();
        for l in 0..8 {
            leaves.push(Leaf {
                v: Param::new(0.0),
                target: 0.1 * b as f64 + 0.01 * l as f64,
                hb: SelfBlock::new(),
            });
        }
        Branch { leaves }
    }).collect();
    Tree { branches }
}

#[test]
fn a_nested_entity_with_its_own_selfblock_threads() {
    threaded_matches_sequential("nested own selfblock", build_tree,
        &|c: &Context| c.blocks_list::<TreeBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("nested own selfblock", build_tree);
}

// ===========================================================================
// A `root.` SelfBlock primary written from a NESTED collection
// ===========================================================================

#[arael::model]
#[arael(constraint(root.hb, { [(nshared.m - root.shift) * nshared.w] }))]
struct NShared {
    m: f64,
    w: f64,
}

#[arael::model]
struct NGroup {
    obs: std::vec::Vec<NShared>,
}

#[arael::model]
#[arael(root, par)]
struct NRootSelf {
    shift: Param<f64>,
    groups: std::vec::Vec<NGroup>,
    hb: SelfBlock<NRootSelf>,
}

fn build_nroot_self() -> NRootSelf {
    NRootSelf {
        shift: Param::new(0.0),
        groups: (0..16).map(|g| NGroup {
            obs: (0..25).map(|k| NShared {
                m: 1.0 + 0.01 * (g * 25 + k) as f64,
                w: 1.0 + 0.001 * k as f64,
            }).collect(),
        }).collect(),
        hb: SelfBlock::new(),
    }
}

#[test]
fn a_root_selfblock_primary_from_a_nested_collection_threads() {
    threaded_matches_sequential("nested root selfblock", build_nroot_self,
        &|c: &Context| c.blocks_list::<NRootSelfBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("nested root selfblock", build_nroot_self);
}

// ===========================================================================
// COO entries: the three forms whose cross pairs cannot be tiled. Each store
// keeps its own list and claims the participants it writes, so they thread
// like every other form.
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, { [(cnode.x - cnode.prior) * 0.1] }))]
struct CNode {
    x: Param<f64>,
    prior: f64,
    hb: SelfBlock<CNode>,
}

// N-ary: three participants reached by ref, their diagonals in their own
// self blocks and the cross pairs in COO. The participants of one instance
// sit far apart in the collection, so a store sweeping `tris` writes nodes
// well outside its own node range.
#[arael::model]
#[arael(constraint(coo, { [(a.x + b.x + c.x - ctri.sum) * 0.3] }))]
struct CTri {
    #[arael(ref = root.nodes)] a: Ref<CNode>,
    #[arael(ref = root.nodes)] b: Ref<CNode>,
    #[arael(ref = root.nodes)] c: Ref<CNode>,
    sum: f64,
}

#[arael::model]
#[arael(root, par)]
struct CooNary {
    nodes: refs::Vec<CNode>,
    tris: std::vec::Vec<CTri>,
}

fn build_coo_nary() -> CooNary {
    let mut m = CooNary { nodes: refs::Vec::new(), tris: std::vec::Vec::new() };
    for k in 0..300 {
        m.nodes.push(CNode {
            x: Param::new(0.1 * k as f64), prior: 0.05 * k as f64, hb: SelfBlock::new(),
        });
    }
    for k in 0..200 {
        m.tris.push(CTri {
            a: m.nodes.ref_at(k),
            b: m.nodes.ref_at((k + 97) % 300),
            c: m.nodes.ref_at((k + 199) % 300),
            sum: 1.0 + 0.01 * k as f64,
        });
    }
    m
}

#[test]
fn an_nary_coo_constraint_threads() {
    threaded_matches_sequential("nary coo", build_coo_nary,
        &|c: &Context| c.blocks_list::<CooNaryBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("nary coo", build_coo_nary);
}

// `[hb, root.<triplet>]`: the entity's own params couple to the root's, so
// every store writes the root's diagonal as well as its own entities'.
#[arael::model]
#[arael(constraint(hb, { [(rnode.x - rnode.prior) * 0.05] }))]
#[arael(constraint([hb, coo], { [(rnode.x - root.shift) * rnode.w] }))]
struct RNode {
    x: Param<f64>,
    prior: f64,
    w: f64,
    hb: SelfBlock<RNode>,
}

#[arael::model]
#[arael(root, par)]
#[arael(constraint(hb, { [(cooroot.shift - 1.0) * 0.2] }))]
struct CooRoot {
    shift: Param<f64>,
    nodes: refs::Vec<RNode>,
    hb: SelfBlock<CooRoot>,
}

fn build_coo_root() -> CooRoot {
    let mut m = CooRoot {
        shift: Param::new(0.0),
        nodes: refs::Vec::new(),
        hb: SelfBlock::new(),
    };
    for k in 0..400 {
        m.nodes.push(RNode {
            x: Param::new(0.01 * k as f64),
            prior: 0.02 * k as f64,
            w: 1.0 + 0.001 * k as f64,
            hb: SelfBlock::new(),
        });
    }
    m
}

#[test]
fn a_root_coupled_coo_constraint_threads() {
    threaded_matches_sequential("root coo", build_coo_root,
        &|c: &Context| c.blocks_list::<CooRootBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("root coo", build_coo_root);
}

// `[hb, parent.<triplet>]`: the same, one level down -- the co-entity is the
// containing parent, and a store claims whichever parents its range reaches.
#[arael::model]
#[arael(constraint(hb, { [(bnode.x - bnode.prior) * 0.05] }))]
#[arael(constraint([hb, coo], { [(bnode.x - band.offset) * bnode.w] }))]
struct BNode {
    x: Param<f64>,
    prior: f64,
    w: f64,
    hb: SelfBlock<BNode>,
}

#[arael::model]
#[arael(constraint(hb, { [(band.offset - band.prior) * 0.2] }))]
struct Band {
    offset: Param<f64>,
    prior: f64,
    nodes: refs::Vec<BNode>,
    hb: SelfBlock<Band>,
}

#[arael::model]
#[arael(root, par)]
struct CooParent {
    bands: std::vec::Vec<Band>,
}

fn build_coo_parent() -> CooParent {
    let mut m = CooParent { bands: std::vec::Vec::new() };
    for g in 0..20 {
        let mut band = Band {
            offset: Param::new(0.1 * g as f64),
            prior: 0.05 * g as f64,
            nodes: refs::Vec::new(),
            hb: SelfBlock::new(),
        };
        for k in 0..20 {
            band.nodes.push(BNode {
                x: Param::new(0.01 * (g * 20 + k) as f64),
                prior: 0.02 * k as f64,
                w: 1.0 + 0.001 * k as f64,
                hb: SelfBlock::new(),
            });
        }
        m.bands.push(band);
    }
    m
}

#[test]
fn a_parent_coupled_coo_constraint_threads() {
    threaded_matches_sequential("parent coo", build_coo_parent,
        &|c: &Context| c.blocks_list::<CooParentBlocks>().map_or(0, |v| v.len()));
    threaded_solve_matches("parent coo", build_coo_parent);
}
