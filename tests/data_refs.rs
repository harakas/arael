// Data refs: `Ref<T>` fields whose target type has no Params are pure
// data reads -- excluded from entity/block accounting, their fields
// readable in bodies and guards. The same problem built with copied
// data and with data refs must agree exactly.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmProblem, RootProblem};

// Param-less target: pure data.
#[arael::model]
struct Tag {
    off: f64,
    w: f64,
}

#[arael::model]
#[arael(constraint(hb, {
    [(node.x - node.px) * node.pw]
}))]
struct Node {
    x: Param<f64>,
    px: f64,
    pw: f64,
    hb: SelfBlock<Node>,
}

// Plain cross constraint + root.<coll> data ref, body reads
// the target's fields.
#[arael::model]
#[arael(constraint(hb, {
    [(b.x - a.x - link.d + t.off) * t.w]
}))]
struct Link {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    #[arael(ref = root.tags)] t: Ref<Tag>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Net {
    nodes: refs::Arena<Node>,
    tags: refs::Arena<Tag>,
    links: std::vec::Vec<Link>,
}

#[test]
fn plain_cross_data_ref() {
    let mut nodes = refs::Arena::new();
    let n0 = nodes.push(Node { x: Param::new(0.1), px: 0.0, pw: 1.0, hb: SelfBlock::new() });
    let n1 = nodes.push(Node { x: Param::new(1.9), px: 2.0, pw: 0.5, hb: SelfBlock::new() });
    let mut tags = refs::Arena::new();
    let t0 = tags.push(Tag { off: 0.25, w: 1.5 });
    let mut net = Net {
        nodes, tags,
        links: vec![Link { a: n0, b: n1, t: t0, d: 2.0, hb: CrossBlock::new() }],
    };
    let mut x = Vec::new();
    RootProblem::serialize(&mut net, &mut x);
    let cost = net.calc_cost(&x);
    // manual: prior0 (0.1*1.0)^2 + prior1 (-0.1*0.5)^2 + link ((1.8-2.0+0.25)*1.5)^2
    let manual = 0.1f64.powi(2) + 0.05f64.powi(2) + (0.05 * 1.5f64).powi(2);
    assert!((cost - manual).abs() < 1e-12, "cost {} != manual {}", cost, manual);
    let d = net.check_gradients(&x);
    assert!(d.is_clean(), "gradient check:\n{}", d);
    let d = net.validate();
    assert!(d.is_clean(), "validate:\n{}", d);
}

// Chain a data ref through another ref field of the same
// struct (`a.tags`): the Node owns a per-entity tag arena.
#[arael::model]
#[arael(constraint(hb, {
    [(node2.x - node2.px) * node2.pw]
}))]
struct Node2 {
    x: Param<f64>,
    px: f64,
    pw: f64,
    tags: refs::Arena<Tag>,
    hb: SelfBlock<Node2>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(b.x - a.x - link2.d + t.off) * t.w]
}))]
struct Link2 {
    #[arael(ref = root.nodes)] a: Ref<Node2>,
    #[arael(ref = root.nodes)] b: Ref<Node2>,
    #[arael(ref = a.tags)] t: Ref<Tag>,
    d: f64,
    hb: CrossBlock<Node2, Node2>,
}

#[arael::model]
#[arael(root)]
struct Net2 {
    nodes: refs::Arena<Node2>,
    links: std::vec::Vec<Link2>,
}

#[test]
fn chained_data_ref() {
    let mut nodes = refs::Arena::new();
    let mut tags0 = refs::Arena::new();
    let t0 = tags0.push(Tag { off: 0.25, w: 1.5 });
    let n0 = nodes.push(Node2 { x: Param::new(0.1), px: 0.0, pw: 1.0, tags: tags0, hb: SelfBlock::new() });
    let n1 = nodes.push(Node2 { x: Param::new(1.9), px: 2.0, pw: 0.5, tags: refs::Arena::new(), hb: SelfBlock::new() });
    let mut net = Net2 {
        nodes,
        links: vec![Link2 { a: n0, b: n1, t: t0, d: 2.0, hb: CrossBlock::new() }],
    };
    let mut x = Vec::new();
    RootProblem::serialize(&mut net, &mut x);
    let cost = net.calc_cost(&x);
    let manual = 0.1f64.powi(2) + 0.05f64.powi(2) + (0.05 * 1.5f64).powi(2);
    assert!((cost - manual).abs() < 1e-12, "cost {} != manual {}", cost, manual);
    let d = net.check_gradients(&x);
    assert!(d.is_clean(), "gradient check:\n{}", d);
}

// ---------------------------------------------------------------------------
// Three-shape equivalence: copied-per-match data (reference) vs data
// refs on the stage-1 parent-cross form vs `parent.<ref>.<coll>` data
// refs into per-entity arenas on the parent-refs form. Guards read
// data-ref fields; per-entity arenas share index values, so agreement
// proves resolution picks the right arena.
// ---------------------------------------------------------------------------

// Reference: plain cross constraint, tag values copied onto the link.
#[arael::model]
#[arael(constraint(hb, guard = self.on && self.taw > 0.0, {
    [(b.x - a.x - rlink.d + rlink.taoff - rlink.tboff)
        * (rlink.taw * rlink.tbw)]
}))]
struct RLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
    taoff: f64,
    taw: f64,
    tboff: f64,
    tbw: f64,
    on: bool,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct NetR {
    nodes: refs::Arena<Node>,
    links: std::vec::Vec<RLink>,
}

// Stage 1: parent-cross with own entity refs plus two data refs into
// a root-level tag arena; the guard reads through a data ref.
#[arael::model]
#[arael(constraint(parent.hb, guard = self.on && ta.w > 0.0, {
    [(b.x - a.x - slink.d + ta.off - tb.off) * (ta.w * tb.w)]
}))]
struct SLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    #[arael(ref = root.tags)] ta: Ref<Tag>,
    #[arael(ref = root.tags)] tb: Ref<Tag>,
    d: f64,
    on: bool,
}

#[arael::model]
struct SPair {
    links: std::vec::Vec<SLink>,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct NetS {
    nodes: refs::Arena<Node>,
    tags: refs::Arena<Tag>,
    pairs: std::vec::Vec<SPair>,
}

// Stage 2: parent-held entity refs; the match holds data refs chained
// through them into each node's OWN tag arena.
#[arael::model]
#[arael(constraint(parent.hb, parent = pp, guard = self.on && ta.w > 0.0, {
    [(pp.b.x - parent.a.x - plink.d + ta.off - tb.off) * (ta.w * tb.w)]
}))]
struct PLink {
    #[arael(ref = parent.a.tags)] ta: Ref<Tag>,
    #[arael(ref = parent.b.tags)] tb: Ref<Tag>,
    d: f64,
    on: bool,
}

#[arael::model]
struct PPair {
    #[arael(ref = root.nodes)] a: Ref<Node3>,
    #[arael(ref = root.nodes)] b: Ref<Node3>,
    links: std::vec::Vec<PLink>,
    hb: CrossBlock<Node3, Node3>,
}

// Node with its own tag arena (the stage-2 entity).
#[arael::model]
#[arael(constraint(hb, {
    [(node3.x - node3.px) * node3.pw]
}))]
struct Node3 {
    x: Param<f64>,
    px: f64,
    pw: f64,
    tags: refs::Arena<Tag>,
    hb: SelfBlock<Node3>,
}

#[arael::model]
#[arael(root)]
struct NetP {
    nodes: refs::Arena<Node3>,
    pairs: std::vec::Vec<PPair>,
}

// One data table drives all three builds. Tag indices are PER NODE
// (every node arena has the same index range with different content).
type NodeData = (f64, f64, f64); // x, px, pw
type TagData = (f64, f64); // off, w
type MatchData = (f64, usize, usize, bool); // d, a-tag idx, b-tag idx, on
struct Data {
    nodes: Vec<NodeData>,
    tags: Vec<Vec<TagData>>, // per node
    pairs: Vec<((usize, usize), Vec<MatchData>)>,
}

fn data() -> Data {
    Data {
        nodes: vec![(0.1, 0.0, 1.0), (1.9, 2.0, 0.5), (4.2, 4.0, 0.25)],
        tags: vec![
            vec![(0.25, 1.5), (0.1, 0.6)],
            vec![(0.3, 0.8), (-0.2, 1.1)],
            vec![(0.15, -1.0), (0.05, 0.9)],
        ],
        pairs: vec![
            ((0, 1), vec![
                (2.0, 0, 1, true),
                (1.8, 1, 0, true),
                (2.1, 0, 0, false), // guarded off by `on`
            ]),
            ((1, 2), vec![(2.3, 0, 0, true)]),
            ((0, 2), vec![(4.1, 1, 1, true)]),
            ((2, 1), vec![(-2.2, 0, 1, true)]), // ta.w <= 0: guarded off
        ],
    }
}

fn build_r(d: &Data) -> NetR {
    let mut nodes = refs::Arena::new();
    let mut nrefs = Vec::new();
    for &(x, px, pw) in &d.nodes {
        nrefs.push(nodes.push(Node { x: Param::new(x), px, pw, hb: SelfBlock::new() }));
    }
    let mut links = Vec::new();
    for ((ia, ib), ms) in &d.pairs {
        for &(dd, ka, kb, on) in ms {
            let (taoff, taw) = d.tags[*ia][ka];
            let (tboff, tbw) = d.tags[*ib][kb];
            links.push(RLink {
                a: nrefs[*ia], b: nrefs[*ib],
                d: dd, taoff, taw, tboff, tbw, on, hb: CrossBlock::new(),
            });
        }
    }
    NetR { nodes, links }
}

fn build_s(d: &Data) -> NetS {
    let mut nodes = refs::Arena::new();
    let mut nrefs = Vec::new();
    for &(x, px, pw) in &d.nodes {
        nrefs.push(nodes.push(Node { x: Param::new(x), px, pw, hb: SelfBlock::new() }));
    }
    // Root tag arena: node arenas flattened, (node i, k) -> tag 2i + k.
    let mut tags = refs::Arena::new();
    let mut trefs = Vec::new();
    for per_node in &d.tags {
        for &(off, w) in per_node {
            trefs.push(tags.push(Tag { off, w }));
        }
    }
    let mut pairs = Vec::new();
    for ((ia, ib), ms) in &d.pairs {
        let links = ms.iter().map(|&(dd, ka, kb, on)| SLink {
            a: nrefs[*ia], b: nrefs[*ib],
            ta: trefs[2 * *ia + ka], tb: trefs[2 * *ib + kb],
            d: dd, on,
        }).collect();
        pairs.push(SPair { links, hb: CrossBlock::new() });
    }
    NetS { nodes, tags, pairs }
}

fn build_p(d: &Data) -> NetP {
    let mut nodes = refs::Arena::new();
    let mut nrefs = Vec::new();
    let mut trefs: Vec<Vec<Ref<Tag>>> = Vec::new();
    for (i, &(x, px, pw)) in d.nodes.iter().enumerate() {
        let mut tags = refs::Arena::new();
        trefs.push(d.tags[i].iter().map(|&(off, w)| tags.push(Tag { off, w })).collect());
        nrefs.push(nodes.push(Node3 { x: Param::new(x), px, pw, tags, hb: SelfBlock::new() }));
    }
    let mut pairs = Vec::new();
    for ((ia, ib), ms) in &d.pairs {
        let links = ms.iter().map(|&(dd, ka, kb, on)| PLink {
            ta: trefs[*ia][ka], tb: trefs[*ib][kb], d: dd, on,
        }).collect();
        pairs.push(PPair { a: nrefs[*ia], b: nrefs[*ib], links, hb: CrossBlock::new() });
    }
    NetP { nodes, pairs }
}

fn dense_of<P: LmProblem<f64> + RootProblem<f64>>(m: &mut P)
    -> (f64, Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut h = vec![0.0; n * n];
    let cost = m.calc_grad_hessian_dense(&x, &mut g, &mut h);
    (cost, x, g, h)
}

#[test]
fn data_ref_shapes_agree() {
    let d = data();
    let mut mr = build_r(&d);
    let mut ms = build_s(&d);
    let mut mp = build_p(&d);

    let (cr, xr, gr, hr) = dense_of(&mut mr);
    let (cs, xs, gs, hs) = dense_of(&mut ms);
    let (cp, xp, gp, hp) = dense_of(&mut mp);
    assert_eq!(xr.len(), xs.len());
    assert_eq!(xr.len(), xp.len());
    assert!((cr - cs).abs() < 1e-12, "cost R {} != S {}", cr, cs);
    assert!((cr - cp).abs() < 1e-12, "cost R {} != P {}", cr, cp);
    let n = xr.len();
    for i in 0..n {
        assert!((gr[i] - gs[i]).abs() < 1e-12, "g[{i}] R != S");
        assert!((gr[i] - gp[i]).abs() < 1e-12, "g[{i}] R != P");
        for j in 0..n {
            assert!((hr[i * n + j] - hs[i * n + j]).abs() < 1e-12, "H[{i},{j}] R != S");
            assert!((hr[i * n + j] - hp[i * n + j]).abs() < 1e-12, "H[{i},{j}] R != P");
        }
    }

    for (label, dd) in [("R", mr.check_gradients(&xr)), ("S", ms.check_gradients(&xs)),
                        ("P", mp.check_gradients(&xp))] {
        assert!(dd.is_clean(), "{label} gradient check:\n{dd}");
    }
    for (label, v) in [("R", mr.validate()), ("S", ms.validate()), ("P", mp.validate())] {
        assert!(v.is_clean(), "{label} validate:\n{v}");
    }
}

#[test]
fn data_ref_shapes_solve_agree() {
    use arael::simple_lm::{self, LmConfig};
    let d = data();
    let mut mr = build_r(&d);
    let mut ms = build_s(&d);
    let mut mp = build_p(&d);
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut xr = Vec::new();
    RootProblem::serialize(&mut mr, &mut xr);
    let rr = simple_lm::solve(&xr, &mut mr, &cfg).unwrap();
    let mut xs = Vec::new();
    RootProblem::serialize(&mut ms, &mut xs);
    let rs = simple_lm::solve(&xs, &mut ms, &cfg).unwrap();
    let mut xp = Vec::new();
    RootProblem::serialize(&mut mp, &mut xp);
    let rp = simple_lm::solve(&xp, &mut mp, &cfg).unwrap();

    assert!((rr.end_cost - rs.end_cost).abs() < 1e-10,
        "end cost R {} != S {}", rr.end_cost, rs.end_cost);
    assert!((rr.end_cost - rp.end_cost).abs() < 1e-10,
        "end cost R {} != P {}", rr.end_cost, rp.end_cost);
    for i in 0..rr.x.len() {
        assert!((rr.x[i] - rs.x[i]).abs() < 1e-8, "x[{i}] R != S");
        assert!((rr.x[i] - rp.x[i]).abs() < 1e-8, "x[{i}] R != P");
    }
}

// A SELF-block constraint reading a data ref: per-entity records in a
// root arena, read by the entity's own residual.
#[arael::model]
#[arael(constraint(hb, {
    [(snode.x - snode.px + t.off) * t.w]
}))]
struct SNode {
    x: Param<f64>,
    px: f64,
    #[arael(ref = root.tags)] t: Ref<Tag>,
    hb: SelfBlock<SNode>,
}

#[arael::model]
#[arael(root)]
struct NetSelf {
    nodes: refs::Arena<SNode>,
    tags: refs::Arena<Tag>,
}

#[test]
fn self_block_data_ref() {
    let mut tags = refs::Arena::new();
    let t0 = tags.push(Tag { off: 0.25, w: 1.5 });
    let t1 = tags.push(Tag { off: -0.1, w: 0.7 });
    let mut nodes = refs::Arena::new();
    nodes.push(SNode { x: Param::new(0.1), px: 0.0, t: t0, hb: SelfBlock::new() });
    nodes.push(SNode { x: Param::new(1.9), px: 2.0, t: t1, hb: SelfBlock::new() });
    let mut net = NetSelf { nodes, tags };
    let mut x = Vec::new();
    RootProblem::serialize(&mut net, &mut x);
    let cost = net.calc_cost(&x);
    let manual = ((0.1 + 0.25) * 1.5f64).powi(2) + ((-0.1 - 0.1) * 0.7f64).powi(2);
    assert!((cost - manual).abs() < 1e-12, "cost {} != manual {}", cost, manual);
    let d = net.check_gradients(&x);
    assert!(d.is_clean(), "gradient check:\n{}", d);
    let d = net.validate();
    assert!(d.is_clean(), "validate:\n{}", d);
}
