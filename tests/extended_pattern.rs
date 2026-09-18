// `hessian_pattern_requires_compute` follows the COO entries a model can
// hold: a `coo` constraint ANYWHERE in the containment tree, or an
// `extended` hook, which is handed the COO list and may push into it.

use arael::model::{Coo, ExtendedModel, Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{LmConfig, LmProblem, RootProblem, LmProblemInternals};

// ===========================================================================
// Extended root that pushes no COO entries
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
fn extended_without_coo_entries_keeps_a_static_pattern() {
    let mut u = upd_only();
    assert!(!LmProblemInternals::<f64>::hessian_pattern_requires_compute(&u));
    // The hook is handed a COO list either way; the probe the solve runs
    // before its first assembly finds this one pushes nothing.
    let mut x = Vec::new();
    u.serialize(&mut x);
    assert!(!LmProblemInternals::<f64>::extended_hook_writes_coo(&mut u, &x));
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
// Parent-coupled `coo` ([hb, coo]): compute-first route
// ===========================================================================

#[arael::model]
#[arael(constraint([hb, coo], {
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
    assert!(LmProblemInternals::<f64>::hessian_pattern_requires_compute(&f));
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
// Extended root whose hook pushes COO entries
// ===========================================================================

#[arael::model]
#[arael(root, extended)]
struct ExtTriplet {
    a: Param<f64>,
    hb: SelfBlock<ExtTriplet>,
}

impl ExtendedModel<f64> for ExtTriplet {
    fn extended_compute(&mut self, params: &[f64], grad: &mut [f64], coo: &mut Coo<f64>) {
        let i = self.a.index() as usize;
        let r = params[i] - 3.0;
        coo.add_residual(r, &[i as u32], &[1.0], grad);
    }
    fn extended_cost(&self, params: &[f64]) -> f64 {
        let r = params[self.a.index() as usize] - 3.0;
        r * r
    }
}

#[test]
fn extended_with_coo_entries_is_observed() {
    let mut e = ExtTriplet { a: Param::new(0.0), hb: SelfBlock::new() };
    // Nothing static says this model has runtime entries: its constraints
    // declare none and the hook is opaque.
    assert!(!LmProblemInternals::<f64>::hessian_pattern_requires_compute(&e));
    // Running the hook is what says so, which is what the solve does once
    // before its first assembly.
    let mut x = Vec::new();
    e.serialize(&mut x);
    assert!(LmProblemInternals::<f64>::extended_hook_writes_coo(&mut e, &x));
}

/// And the solve that follows the probe lands on the right answer -- the
/// hook's entries reach the Hessian through the discovered pattern.
#[test]
fn extended_with_coo_entries_solves_sparse() {
    let mut s = ExtTriplet { a: Param::new(0.0), hb: SelfBlock::new() };
    let rs = s.solve_sparse(&LmConfig::default()).unwrap();
    let mut d = ExtTriplet { a: Param::new(0.0), hb: SelfBlock::new() };
    let rd = d.solve_dense(&LmConfig::default()).unwrap();
    assert!((rs.end_cost - rd.end_cost).abs() < 1e-12,
        "sparse {} vs dense {}", rs.end_cost, rd.end_cost);
    assert!((s.a.value - 3.0).abs() < 1e-9, "a = {}", s.a.value);
}

// ===========================================================================
// The probe runs the hook once more per solve, alone, before the first
// assembly
// ===========================================================================

#[arael::model]
#[arael(root, extended)]
#[arael(constraint(hb, { [(logged.a - 1.0) * 0.5] }))]
struct Logged {
    a: Param<f64>,
    hb: SelfBlock<Logged>,
    /// Every hook call, in order: "update", "compute" or "cost".
    #[arael(skip)]
    log: std::cell::RefCell<std::vec::Vec<&'static str>>,
}

impl ExtendedModel<f64> for Logged {
    fn extended_update(&mut self, _params: &[f64]) {
        self.log.borrow_mut().push("update");
    }
    fn extended_compute(&mut self, params: &[f64], grad: &mut [f64], coo: &mut Coo<f64>) {
        self.log.borrow_mut().push("compute");
        let i = self.a.index() as usize;
        let r = params[i] - 3.0;
        coo.add_residual(r, &[i as u32], &[1.0], grad);
    }
    fn extended_cost(&self, params: &[f64]) -> f64 {
        self.log.borrow_mut().push("cost");
        let r = params[self.a.index() as usize] - 3.0;
        r * r
    }
}

/// An assembly runs update, compute, cost; a cost evaluation runs update,
/// cost. The probe is the one compute with no cost after it, and it comes
/// first.
#[test]
fn the_probe_runs_the_hook_once_before_the_first_assembly() {
    let mut m = Logged {
        a: Param::new(0.0), hb: SelfBlock::new(), log: std::cell::RefCell::new(Vec::new()),
    };
    m.solve_sparse(&LmConfig { max_iters: 5, ..Default::default() }).unwrap();
    let log = m.log.borrow();
    let computes: Vec<usize> = log.iter().enumerate()
        .filter(|(_, e)| **e == "compute").map(|(i, _)| i).collect();
    assert!(computes.len() >= 2, "a probe and at least one assembly: {:?}", *log);
    let bare: Vec<usize> = computes.iter().copied()
        .filter(|&i| log.get(i + 1) != Some(&"cost")).collect();
    assert_eq!(bare, vec![computes[0]], "exactly one compute has no cost after it, \
        the probe, and it is the first: {:?}", *log);
    assert_eq!(log.get(computes[0].wrapping_sub(1)), Some(&"update"),
        "the probe updates the params first: {:?}", *log);
}
