// `hessian_pattern_requires_compute` follows the presence of a
// TripletBlock ANYWHERE in the containment tree -- not the `extended`
// flag. An extended root without triplets keeps a static block pattern
// (extended hooks can only add Hessian entries through declared block
// fields), so the structure-based sparse routes apply; a parent-owned
// triplet (`[hb, parent.hbt]`) forces the compute-first route even
// though the root's own fields hold no triplet.

use arael::model::{ExtendedModel, Param, SelfBlock, TripletBlock};
use arael::refs;
use arael::simple_lm::{LmConfig, LmProblem, RootProblem};

// ===========================================================================
// Extended root, no TripletBlock: static pattern, fast sparse routes
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, {
    [(node.x - node.target) * 2.0]
}))]
struct Node {
    x: Param<f64>,
    target: f64,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(root, extended)]
struct UpdOnly {
    updates: u32,
    nodes: refs::Vec<Node>,
}

impl ExtendedModel<f64> for UpdOnly {
    fn extended_update(&mut self, _params: &[f64]) {
        // Derived state only; no residuals, no Hessian entries.
        self.updates += 1;
    }
}

fn upd_only() -> UpdOnly {
    let mut u = UpdOnly { updates: 0, nodes: refs::Vec::new() };
    for i in 0..4 {
        u.nodes.push(Node {
            x: Param::new(0.1 * i as f64),
            target: 1.0 + i as f64,
            hb: SelfBlock::new(),
        });
    }
    u
}

#[test]
fn extended_without_triplet_has_static_pattern() {
    let u = upd_only();
    assert!(!LmProblem::<f64>::hessian_pattern_requires_compute(&u));
}

#[test]
fn extended_without_triplet_solves_sparse() {
    let mut s = upd_only();
    let mut d = upd_only();
    let rs = s.solve_sparse(&LmConfig::default()).unwrap();
    let rd = d.solve_dense(&LmConfig::default()).unwrap();
    assert!((rs.end_cost - rd.end_cost).abs() < 1e-12);
    for (a, b) in s.nodes.iter().zip(d.nodes.iter()) {
        assert!((a.x.value - b.x.value).abs() < 1e-9);
    }
    // The extended hook ran on the structure-based route.
    assert!(s.updates > 0, "extended_update must run on the sparse route");
}

// ===========================================================================
// Parent-owned TripletBlock ([hb, parent.hbt]): compute-first route
// ===========================================================================

#[arael::model]
#[arael(constraint([hb, parent.hbt], {
    [obs.y - (curve.m * obs.x + obs.o)]
}))]
struct Obs {
    x: f64,
    y: f64,
    o: Param<f64>,
    hb: SelfBlock<Obs>,
}

#[arael::model]
struct Curve {
    m: Param<f64>,
    obs: std::vec::Vec<Obs>,
    hb: SelfBlock<Curve>,
    hbt: TripletBlock<f64>,
}

#[arael::model]
#[arael(root)]
struct Fit {
    curves: refs::Vec<Curve>,
}

fn fit() -> Fit {
    let mut f = Fit { curves: refs::Vec::new() };
    let mut c = Curve {
        m: Param::new(0.1),
        obs: std::vec::Vec::new(),
        hb: SelfBlock::new(),
        hbt: TripletBlock::new(),
    };
    c.obs.push(Obs { x: 1.0, y: 2.0, o: Param::new(0.0), hb: SelfBlock::new() });
    c.obs.push(Obs { x: 2.0, y: 3.5, o: Param::new(0.1), hb: SelfBlock::new() });
    c.obs.push(Obs { x: 3.0, y: 5.2, o: Param::new(-0.1), hb: SelfBlock::new() });
    f.curves.push(c);
    f
}

#[test]
fn nested_triplet_requires_compute() {
    let f = fit();
    assert!(LmProblem::<f64>::hessian_pattern_requires_compute(&f));
}

// Used to panic ("sparsity pattern changed between iterations"): the
// root-fields-only triplet detection missed the parent-owned triplet and
// the structure-built pattern lacked its COO entries.
#[test]
fn nested_triplet_solves_sparse() {
    let mut s = fit();
    let mut d = fit();
    let rs = s.solve_sparse(&LmConfig::default()).unwrap();
    let rd = d.solve_dense(&LmConfig::default()).unwrap();
    assert!((rs.end_cost - rd.end_cost).abs() < 1e-9,
        "sparse {} vs dense {}", rs.end_cost, rd.end_cost);
}

// ===========================================================================
// Extended root WITH a root triplet: stays compute-first
// ===========================================================================

#[arael::model]
#[arael(root, extended)]
struct ExtTriplet {
    a: Param<f64>,
    hb: SelfBlock<ExtTriplet>,
    hbt: TripletBlock<f64>,
}

impl ExtendedModel<f64> for ExtTriplet {
    fn extended_compute(&mut self, params: &[f64], grad: &mut [f64]) {
        let i = self.a.index() as usize;
        let r = params[i] - 3.0;
        self.hbt.add_residual(r, &[i as u32], &[1.0], grad);
    }
    fn extended_cost(&self, params: &[f64]) -> f64 {
        let r = params[self.a.index() as usize] - 3.0;
        r * r
    }
}

#[test]
fn extended_with_triplet_requires_compute() {
    let e = ExtTriplet { a: Param::new(0.0), hb: SelfBlock::new(), hbt: TripletBlock::new() };
    assert!(LmProblem::<f64>::hessian_pattern_requires_compute(&e));
}
