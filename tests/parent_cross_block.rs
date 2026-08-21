// `parent.<crossblock>`: a CrossBlock on a containing parent struct,
// shared by every constraint instance the parent holds. The same
// problem built with per-instance CrossBlocks and with the shared
// parent block must produce identical cost, gradient, Hessian, and
// solution; all instances under one parent must reference the same
// entity pair (checked at wiring time).

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{CooMatrix, LmConfig, LmProblem, RootProblem};

const TOL: f64 = 1e-9;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

/// Cost + all-route + FD + validate battery (same as macro_matrix.rs).
fn check_model<P>(label: &str, m: &mut P, manual_cost: f64)
where
    P: LmProblem<f64> + RootProblem<f64>,
{
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let n = x.len();
    let cost = m.calc_cost(&x);
    assert!(close(cost, manual_cost, TOL),
        "{label}: calc_cost {} != manual {}", cost, manual_cost);

    let mut gd = vec![0.0; n];
    let mut hd = vec![0.0; n * n];
    let cd = m.calc_grad_hessian_dense(&x, &mut gd, &mut hd);
    assert!(close(cd, cost, TOL), "{label}: dense cost");

    let mut gs = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    let cs = m.calc_grad_hessian_sparse(&x, &mut gs, &mut coo);
    assert!(close(cs, cost, TOL), "{label}: coo cost");
    let mut hs = vec![0.0; n * n];
    for k in 0..coo.rows.len() {
        let (r, c) = (coo.rows[k] as usize, coo.cols[k] as usize);
        hs[r * n + c] += coo.vals[k];
        if r != c { hs[c * n + r] += coo.vals[k]; }
    }
    for i in 0..n {
        assert!(close(gs[i], gd[i], TOL), "{label}: coo grad[{i}]");
        for j in 0..n {
            assert!(close(hs[i * n + j], hd[i * n + j], TOL),
                "{label}: coo H[{i},{j}] {} != dense {}", hs[i * n + j], hd[i * n + j]);
        }
    }

    let (csc, positions) = coo.to_csc_with_map().unwrap();
    let mut gi = vec![0.0; n];
    let mut vals = vec![0.0; csc.vals.len()];
    let ci = m.calc_grad_hessian_sparse_indexed(&x, &mut gi, &mut vals, &positions);
    assert!(close(ci, cost, TOL), "{label}: indexed cost");
    for i in 0..n {
        assert!(close(gi[i], gd[i], TOL), "{label}: indexed grad[{i}]");
    }

    let kd = n - 1;
    let ldab = kd + 1;
    let mut gb = vec![0.0; n];
    let mut band = vec![0.0; ldab * n];
    let cb = m.calc_grad_hessian_band(&x, &mut gb, &mut band, kd)
        .unwrap_or_else(|e| panic!("{label}: band overflow: {e}"));
    assert!(close(cb, cost, TOL), "{label}: band cost");
    for i in 0..n {
        for j in i..n {
            assert!(close(band[(kd + i - j) + j * ldab], hd[i * n + j], TOL),
                "{label}: band H[{i},{j}]");
        }
    }

    let d = m.check_gradients(&x);
    assert!(d.is_clean(), "{label}: gradient check:\n{}", d);
    let d = m.validate();
    assert!(d.is_clean(), "{label}: validate:\n{}", d);
}

// Shared entity: position with a per-node prior (anchors the solve).
#[arael::model]
#[arael(constraint(hb, {
    [(node.x - node.px) * node.pw, (node.y - node.py) * node.pw]
}))]
struct Node {
    x: Param<f64>,
    y: Param<f64>,
    px: f64,
    py: f64,
    pw: f64,
    hb: SelfBlock<Node>,
}

// Build A: classic per-instance CrossBlock.
#[arael::model]
#[arael(constraint(hb, guard = self.on, {
    [(b.x - a.x - link.dx) * link.w,
     (b.y + a.x - link.dy) * link.w]
}))]
struct Link {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    dx: f64,
    dy: f64,
    w: f64,
    on: bool,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct NetA {
    nodes: refs::Arena<Node>,
    links: std::vec::Vec<Link>,
}

// Build B: the shared parent-owned CrossBlock, one per node pair.
#[arael::model]
#[arael(constraint(parent.hb, guard = self.on, {
    [(b.x - a.x - slink.dx) * slink.w,
     (b.y + a.x - slink.dy) * slink.w]
}))]
struct SLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    dx: f64,
    dy: f64,
    w: f64,
    on: bool,
}

#[arael::model]
struct Pair {
    links: std::vec::Vec<SLink>,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct NetB {
    nodes: refs::Arena<Node>,
    pairs: std::vec::Vec<Pair>,
}

// One data table drives both builds and the manual cost.
type NodeData = (f64, f64, f64, f64, f64); // x, y, px, py, pw
type MatchData = (f64, f64, f64, bool); // dx, dy, w, on
struct Data {
    nodes: Vec<NodeData>,
    pairs: Vec<((usize, usize), Vec<MatchData>)>,
}

fn data() -> Data {
    Data {
        nodes: vec![
            (0.1, -0.3, 0.0, 0.0, 1.0),
            (1.9, 1.2, 2.0, 1.0, 0.3),
            (4.2, 0.8, 4.0, 1.0, 0.2),
        ],
        pairs: vec![
            ((0, 1), vec![
                (2.0, 1.0, 1.5, true),
                (1.8, 0.9, 0.7, true),
                (2.2, 1.1, 1.0, false), // guarded off
                (1.7, 1.05, 2.0, true),
            ]),
            ((0, 2), vec![
                (4.0, 1.0, 0.8, true),
                (3.9, 0.95, 1.1, true),
                (4.1, 1.02, 0.6, true),
            ]),
            ((1, 2), vec![
                (2.0, 0.0, 1.2, true),
                (2.1, -0.1, 0.9, true),
            ]),
            // Aliased pair: both slots the same node (matches the local
            // CrossBlock aliasing semantics).
            ((2, 2), vec![(0.0, 3.0, 0.5, true)]),
            // An empty pair: its shared block stays unwired and inert.
            ((0, 1), vec![]),
        ],
    }
}

fn nodes_of(d: &Data) -> (refs::Arena<Node>, Vec<Ref<Node>>) {
    let mut nodes = refs::Arena::new();
    let mut nrefs = Vec::new();
    for &(x, y, px, py, pw) in &d.nodes {
        nrefs.push(nodes.push(Node {
            x: Param::new(x), y: Param::new(y), px, py, pw, hb: SelfBlock::new() }));
    }
    (nodes, nrefs)
}

fn build_a(d: &Data) -> (NetA, Vec<Ref<Node>>) {
    let (nodes, nrefs) = nodes_of(d);
    let mut links = Vec::new();
    for ((ia, ib), ms) in &d.pairs {
        for &(dx, dy, w, on) in ms {
            links.push(Link {
                a: nrefs[*ia], b: nrefs[*ib],
                dx, dy, w, on, hb: CrossBlock::new(),
            });
        }
    }
    (NetA { nodes, links }, nrefs)
}

fn build_b(d: &Data) -> (NetB, Vec<Ref<Node>>) {
    let (nodes, nrefs) = nodes_of(d);
    let mut pairs = Vec::new();
    for ((ia, ib), ms) in &d.pairs {
        let links = ms.iter().map(|&(dx, dy, w, on)| SLink {
            a: nrefs[*ia], b: nrefs[*ib], dx, dy, w, on,
        }).collect();
        pairs.push(Pair { links, hb: CrossBlock::new() });
    }
    (NetB { nodes, pairs }, nrefs)
}

fn manual_cost(d: &Data) -> f64 {
    let mut c = 0.0;
    for &(x, y, px, py, pw) in &d.nodes {
        c += ((x - px) * pw).powi(2) + ((y - py) * pw).powi(2);
    }
    for ((ia, ib), ms) in &d.pairs {
        let (ax, ay) = (d.nodes[*ia].0, d.nodes[*ia].1);
        let (bx, by) = (d.nodes[*ib].0, d.nodes[*ib].1);
        let _ = ay;
        for &(dx, dy, w, on) in ms {
            if !on { continue; }
            c += ((bx - ax - dx) * w).powi(2) + ((by + ax - dy) * w).powi(2);
        }
    }
    c
}

fn dense<P: LmProblem<f64> + RootProblem<f64>>(m: &mut P) -> (f64, Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut h = vec![0.0; n * n];
    let c = m.calc_grad_hessian_dense(&x, &mut g, &mut h);
    (c, g, h)
}

#[test]
fn shared_parent_block_equals_per_instance_blocks() {
    let d = data();
    let (mut a, _) = build_a(&d);
    let (mut b, _) = build_b(&d);
    let mc = manual_cost(&d);
    check_model("per-instance", &mut a, mc);
    check_model("parent-shared", &mut b, mc);

    let (ca, ga, ha) = dense(&mut a);
    let (cb, gb, hb) = dense(&mut b);
    assert!(close(ca, cb, TOL), "cost mismatch {} vs {}", ca, cb);
    assert_eq!(ga.len(), gb.len());
    for i in 0..ga.len() {
        assert!(close(ga[i], gb[i], TOL), "grad[{i}] {} vs {}", ga[i], gb[i]);
    }
    for k in 0..ha.len() {
        assert!(close(ha[k], hb[k], TOL), "H[{k}] {} vs {}", ha[k], hb[k]);
    }
}

#[test]
fn shared_parent_block_solves_to_the_same_solution() {
    let d = data();
    let (mut a, ra) = build_a(&d);
    let (mut b, rb) = build_b(&d);
    a.solve_sparse(&LmConfig::well_conditioned()).unwrap();
    b.solve_sparse(&LmConfig::well_conditioned()).unwrap();
    for i in 0..d.nodes.len() {
        let na = a.nodes.get(ra[i]).unwrap();
        let nb = b.nodes.get(rb[i]).unwrap();
        assert!(close(na.x.value, nb.x.value, 1e-7),
            "node {i} x: {} vs {}", na.x.value, nb.x.value);
        assert!(close(na.y.value, nb.y.value, 1e-7),
            "node {i} y: {} vs {}", na.y.value, nb.y.value);
    }
}

#[test]
fn fixed_entity_on_one_side_matches() {
    let d = data();
    let (mut a, ra) = build_a(&d);
    let (mut b, rb) = build_b(&d);
    {
        let n = a.nodes.get_mut(ra[2]).unwrap();
        n.x = Param::fixed(n.x.value);
        n.y = Param::fixed(n.y.value);
        let n = b.nodes.get_mut(rb[2]).unwrap();
        n.x = Param::fixed(n.x.value);
        n.y = Param::fixed(n.y.value);
    }
    let (ca, ga, ha) = dense(&mut a);
    let (cb, gb, hb) = dense(&mut b);
    assert!(close(ca, cb, TOL));
    for i in 0..ga.len() {
        assert!(close(ga[i], gb[i], TOL), "grad[{i}]");
    }
    for k in 0..ha.len() {
        assert!(close(ha[k], hb[k], TOL), "H[{k}]");
    }
}

#[test]
#[should_panic(expected = "must reference the same entity pair")]
fn mismatched_pairs_under_one_parent_panic() {
    let d = data();
    let (mut b, rb) = build_b(&d);
    // Second match of the first pair now points at a different pair.
    b.pairs[0].links[1].b = rb[2];
    let mut x = Vec::new();
    RootProblem::serialize(&mut b, &mut x);
}
