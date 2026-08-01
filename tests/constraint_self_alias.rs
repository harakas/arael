// Integration test: `self` in a constraint BODY refers to the constraint's own
// entity, exactly as `self` does in a guard. Covers a self-block (Node -- where
// `self` and the lowercased name `node` are interchangeable in one body, and
// `self.pos` must remain a differentiated Param) and a cross-block (Link, where
// `self.offset` / `self.w` are the constraint's own data fields).

use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::simple_lm::LmProblem;
use arael::vect::vect2d;
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, {
    // `self` and `node` name the same entity -- mixing them must resolve identically.
    [(self.pos.x - self.target) * self.w,
     (node.pos.y - node.target) * node.w]
}))]
struct Node {
    pos: Param<vect2d>,
    target: f64,
    w: f64,
    hb: SelfBlock<Node>,
}

#[arael::model]
struct Pt {
    pos: Param<vect2d>,
    hb: SelfBlock<Pt>,
}

#[arael::model]
#[arael(constraint(hb, {
    // `self.offset` / `self.w` are Link's own data; a / b are the referenced points.
    [(a.pos.x - b.pos.x - self.offset) * self.w,
     (a.pos.y - b.pos.y - self.offset) * self.w]
}))]
struct Link {
    #[arael(ref = root.points)] a: Ref<Pt>,
    #[arael(ref = root.points)] b: Ref<Pt>,
    offset: f64,
    w: f64,
    hb: CrossBlock<Pt, Pt>,
}

#[arael::model]
#[arael(root)]
struct World {
    nodes: refs::Vec<Node>,
    points: refs::Vec<Pt>,
    links: std::vec::Vec<Link>,
}

fn build() -> (World, Vec<f64>) {
    let mut w = World {
        nodes: refs::Vec::new(),
        points: refs::Vec::new(),
        links: std::vec::Vec::new(),
    };
    w.nodes.push(Node { pos: Param::new(vect2d::new(0.6, -0.8)), target: 0.3, w: 1.5, hb: SelfBlock::new() });
    w.points.push(Pt { pos: Param::new(vect2d::new(2.0, 1.0)), hb: SelfBlock::new() });
    w.points.push(Pt { pos: Param::new(vect2d::new(0.5, -0.5)), hb: SelfBlock::new() });
    w.links.push(Link { a: w.points.ref_at(0), b: w.points.ref_at(1), offset: 0.4, w: 2.0, hb: CrossBlock::new() });
    let mut params = Vec::new();
    w.serialize(&mut params);
    (w, params)
}

fn expected_cost() -> f64 {
    let rn = [(0.6 - 0.3) * 1.5, (-0.8 - 0.3) * 1.5];
    let rl = [(2.0 - 0.5 - 0.4) * 2.0, (1.0 - (-0.5) - 0.4) * 2.0];
    rn[0] * rn[0] + rn[1] * rn[1] + rl[0] * rl[0] + rl[1] * rl[1]
}

#[test]
fn self_alias_cost_matches() {
    let (mut w, params) = build();
    let cost = w.calc_cost(&params);
    let expected = expected_cost();
    assert!((cost - expected).abs() < 1e-12 * (1.0 + expected),
        "cost={} expected={}", cost, expected);
}

#[test]
fn self_alias_gradient_matches_fd() {
    let (mut w, params) = build();
    let n = params.len();
    assert_eq!(n, 6, "Node.pos + 2x Pt.pos");

    let mut grad = vec![0.0_f64; n];
    let mut hess = vec![0.0_f64; n * n];
    w.calc_grad_hessian_dense(&params, &mut grad, &mut hess);

    let eps = 1e-6;
    let mut max_abs = 0.0_f64;
    for i in 0..n {
        let mut p = params.clone();
        p[i] += eps;
        let cp = w.calc_cost(&p);
        p[i] -= 2.0 * eps;
        let cm = w.calc_cost(&p);
        let fd = (cp - cm) / (2.0 * eps);
        assert!((fd - grad[i]).abs() < 1e-4 * (1.0 + fd.abs()),
            "grad[{}]: analytic={} fd={}", i, grad[i], fd);
        max_abs = max_abs.max(grad[i].abs());
    }
    // If `self.pos` were turned into a constant instead of a Param, its
    // gradient columns would be zero.
    assert!(max_abs > 0.1, "gradient suspiciously small (self.pos not differentiated?): {}", max_abs);
}
