// `parent.<field>` value reads in nested constraint bodies and guards:
// a constraint held inside a sub-model reads the containing parent's
// plain data (scalars, nested data structs) instead of carrying copies;
// where the parent is already a coupled entity, `parent` aliases that
// binding with correct derivatives.

use arael::model::{CrossBlock, JacobianModel, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{CooMatrix, LmProblem, RootProblem};

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
                "{label}: coo H[{i},{j}]");
        }
    }

    let d = m.check_gradients(&x);
    assert!(d.is_clean(), "{label}: gradient check:\n{}", d);
    let d = m.validate();
    assert!(d.is_clean(), "{label}: validate:\n{}", d);
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

// Shared entity with a per-node prior.
#[arael::model]
#[arael(constraint(hb, {
    [(node.x - node.t) * node.pw]
}))]
struct Node {
    x: Param<f64>,
    t: f64,
    pw: f64,
    hb: SelfBlock<Node>,
}

#[arael::model]
struct Cfg {
    scale: f64,
}

// ===========================================================================
// Nested cross constraint: parent data in body and guard
// ===========================================================================

// Reference: everything copied per instance, flat on the root.
#[arael::model]
#[arael(constraint(hb, guard = self.on, {
    [(b.x - a.x - rlink.d) * rlink.isigma * rlink.scale]
}))]
struct RLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
    isigma: f64,
    scale: f64,
    on: bool,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct RNet {
    nodes: refs::Arena<Node>,
    links: std::vec::Vec<RLink>,
}

// Parent values: isigma, the nested cfg.scale, and the guard live once
// on the containing group.
#[arael::model]
#[arael(constraint(hb, guard = parent.enabled, {
    [(b.x - a.x - glink.d) * parent.isigma * parent.cfg.scale]
}))]
struct GLink {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    #[arael(ref = root.nodes)] b: Ref<Node>,
    d: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
struct Group {
    isigma: f64,
    enabled: bool,
    cfg: Cfg,
    links: std::vec::Vec<GLink>,
}

#[arael::model]
#[arael(root, jacobian)]
struct GNet {
    nodes: refs::Arena<Node>,
    groups: std::vec::Vec<Group>,
}

const NODES: [(f64, f64, f64); 3] = [(0.1, 0.0, 1.0), (1.9, 2.0, 0.4), (4.2, 4.0, 0.3)];
// (group: isigma, enabled, scale), links: (a, b, d)
const GROUPS: [((f64, bool, f64), [(usize, usize, f64); 2]); 2] = [
    ((1.5, true, 0.9), [(0, 1, 2.0), (1, 2, 2.1)]),
    ((0.7, false, 1.2), [(0, 2, 4.0), (0, 1, 1.9)]),
];

fn nodes_of() -> (refs::Arena<Node>, Vec<Ref<Node>>) {
    let mut nodes = refs::Arena::new();
    let mut nrefs = Vec::new();
    for &(x, t, pw) in &NODES {
        nrefs.push(nodes.push(Node { x: Param::new(x), t, pw, hb: SelfBlock::new() }));
    }
    (nodes, nrefs)
}

fn manual_cost() -> f64 {
    let mut c = 0.0;
    for &(x, t, pw) in &NODES {
        c += ((x - t) * pw).powi(2);
    }
    for ((isigma, enabled, scale), links) in &GROUPS {
        if !enabled { continue; }
        for &(ia, ib, d) in links {
            c += ((NODES[ib].0 - NODES[ia].0 - d) * isigma * scale).powi(2);
        }
    }
    c
}

#[test]
fn parent_values_equal_per_instance_copies() {
    let (nodes, nrefs) = nodes_of();
    let mut r = RNet { nodes, links: Vec::new() };
    for ((isigma, enabled, scale), links) in &GROUPS {
        for &(ia, ib, d) in links {
            r.links.push(RLink {
                a: nrefs[ia], b: nrefs[ib], d,
                isigma: *isigma, scale: *scale, on: *enabled,
                hb: CrossBlock::new(),
            });
        }
    }
    let (nodes, nrefs) = nodes_of();
    let mut g = GNet { nodes, groups: Vec::new() };
    for ((isigma, enabled, scale), links) in &GROUPS {
        g.groups.push(Group {
            isigma: *isigma, enabled: *enabled, cfg: Cfg { scale: *scale },
            links: links.iter().map(|&(ia, ib, d)| GLink {
                a: nrefs[ia], b: nrefs[ib], d, hb: CrossBlock::new(),
            }).collect(),
        });
    }
    let mc = manual_cost();
    check_model("copied", &mut r, mc);
    check_model("parent-values", &mut g, mc);

    let (cr, gr, hr) = dense(&mut r);
    let (cg, gg, hg) = dense(&mut g);
    assert!(close(cr, cg, TOL));
    for i in 0..gr.len() {
        assert!(close(gr[i], gg[i], TOL), "grad[{i}]");
    }
    for k in 0..hr.len() {
        assert!(close(hr[k], hg[k], TOL), "H[{k}]");
    }

    // Jacobian route renders the same parent reads.
    let mut x = Vec::new();
    RootProblem::serialize(&mut g, &mut x);
    let j = g.calc_jacobian(&x);
    assert!(!j.rows.is_empty());
}

// ===========================================================================
// Nested self-block constraint: parent data in the entity's own body
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, {
    [(rnode.x - rnode.t) * rnode.pw]
}))]
struct RNode {
    x: Param<f64>,
    t: f64,
    pw: f64,
    hb: SelfBlock<RNode>,
}

#[arael::model]
struct RPod {
    nodes: refs::Arena<RNode>,
}

#[arael::model]
#[arael(root)]
struct RPods {
    pods: std::vec::Vec<RPod>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(gnode.x - gnode.t) * parent.pw]
}))]
struct GNode {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<GNode>,
}

#[arael::model]
struct GPod {
    pw: f64,
    nodes: refs::Arena<GNode>,
}

#[arael::model]
#[arael(root)]
struct GPods {
    pods: std::vec::Vec<GPod>,
}

#[test]
fn self_block_constraint_reads_parent_values() {
    let pods: [(f64, [(f64, f64); 2]); 2] =
        [(1.4, [(0.2, 0.0), (2.3, 2.0)]), (0.6, [(4.1, 4.0), (5.9, 6.0)])];
    let mut r = RPods { pods: Vec::new() };
    let mut g = GPods { pods: Vec::new() };
    let mut mc = 0.0;
    for (pw, nodes) in &pods {
        let mut rp = RPod { nodes: refs::Arena::new() };
        let mut gp = GPod { pw: *pw, nodes: refs::Arena::new() };
        for &(x, t) in nodes {
            rp.nodes.push(RNode { x: Param::new(x), t, pw: *pw, hb: SelfBlock::new() });
            gp.nodes.push(GNode { x: Param::new(x), t, hb: SelfBlock::new() });
            mc += ((x - t) * pw).powi(2);
        }
        r.pods.push(rp);
        g.pods.push(gp);
    }
    check_model("copied", &mut r, mc);
    check_model("parent-values", &mut g, mc);
    let (cr, gr, hr) = dense(&mut r);
    let (cg, gg, hg) = dense(&mut g);
    assert!(close(cr, cg, TOL));
    for i in 0..gr.len() {
        assert!(close(gr[i], gg[i], TOL), "grad[{i}]");
    }
    for k in 0..hr.len() {
        assert!(close(hr[k], hg[k], TOL), "H[{k}]");
    }
}

// ===========================================================================
// Frine-style: `parent` aliases the coupled parent entity (params too)
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, {
    [(lma.pos - a.x - fra.z) * fra.w]
}))]
struct FrA {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    z: f64,
    w: f64,
    hb: CrossBlock<LmA, Node>,
}

#[arael::model]
struct LmA {
    pos: Param<f64>,
    frines: std::vec::Vec<FrA>,
    hb: SelfBlock<LmA>,
}

#[arael::model]
#[arael(root)]
struct FNetA {
    nodes: refs::Arena<Node>,
    lms: refs::Arena<LmA>,
}

// Same residual through the `parent` alias -- identical derivatives.
#[arael::model]
#[arael(constraint(hb, {
    [(parent.pos - a.x - frb.z) * frb.w]
}))]
struct FrB {
    #[arael(ref = root.nodes)] a: Ref<Node>,
    z: f64,
    w: f64,
    hb: CrossBlock<LmB, Node>,
}

#[arael::model]
struct LmB {
    pos: Param<f64>,
    frines: std::vec::Vec<FrB>,
    hb: SelfBlock<LmB>,
}

#[arael::model]
#[arael(root)]
struct FNetB {
    nodes: refs::Arena<Node>,
    lms: refs::Arena<LmB>,
}

#[test]
fn frine_parent_alias_equals_named_binding() {
    let obs: [(f64, [(usize, f64, f64); 2]); 2] =
        [(3.0, [(0, 2.9, 1.1), (1, 1.2, 0.8)]), (5.5, [(1, 3.5, 1.3), (2, 1.4, 0.9)])];
    let (nodes, nrefs) = nodes_of();
    let mut fa = FNetA { nodes, lms: refs::Arena::new() };
    let (nodes, nrefs_b) = nodes_of();
    let mut fb = FNetB { nodes, lms: refs::Arena::new() };
    let mut mc = 0.0;
    for &(x, t, pw) in &NODES {
        mc += ((x - t) * pw).powi(2);
    }
    for (pos, frs) in &obs {
        fa.lms.push(LmA {
            pos: Param::new(*pos),
            frines: frs.iter().map(|&(ia, z, w)| FrA {
                a: nrefs[ia], z, w, hb: CrossBlock::new() }).collect(),
            hb: SelfBlock::new(),
        });
        fb.lms.push(LmB {
            pos: Param::new(*pos),
            frines: frs.iter().map(|&(ia, z, w)| FrB {
                a: nrefs_b[ia], z, w, hb: CrossBlock::new() }).collect(),
            hb: SelfBlock::new(),
        });
        for &(ia, z, w) in frs {
            mc += ((pos - NODES[ia].0 - z) * w).powi(2);
        }
    }
    check_model("named", &mut fa, mc);
    check_model("parent-alias", &mut fb, mc);
    let (ca, ga, ha) = dense(&mut fa);
    let (cb, gb, hb) = dense(&mut fb);
    assert!(close(ca, cb, TOL));
    for i in 0..ga.len() {
        assert!(close(ga[i], gb[i], TOL), "grad[{i}]");
    }
    for k in 0..ha.len() {
        assert!(close(ha[k], hb[k], TOL), "H[{k}]");
    }
}
