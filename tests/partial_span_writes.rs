// A row that touches part of an entity's parameters writes the block
// over the touched positions only. The gradient and Hessian equal those
// of the same residuals on entities that hold only the touched
// parameters, with the untouched rows and columns zero; a guarded
// residual's untouched partials (`branch(q, 0, 0)`) count as untouched.

use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmProblem, RootProblem};

/// Two parameters, the rows read one of them each.
#[arael::model]
struct Pair {
    a: Param<f64>,
    b: Param<f64>,
    hb: SelfBlock<Pair>,
}

/// Three parameters, a residual reads the middle one.
#[arael::model]
struct Triple {
    u: Param<f64>,
    v: Param<f64>,
    w: Param<f64>,
    hb: SelfBlock<Triple>,
}

#[arael::model]
#[arael(constraint(hb, { [branch(obs.on, pair.a * triple.v - obs.m, 0.0), pair.b - obs.n] }))]
struct Obs {
    #[arael(ref = root.pairs)]
    pair: Ref<Pair>,
    #[arael(ref = root.triples)]
    triple: Ref<Triple>,
    m: f64,
    n: f64,
    on: f64,
    hb: CrossBlock<Pair, Triple>,
}

#[arael::model]
#[arael(root)]
struct W {
    pairs: refs::Vec<Pair>,
    triples: refs::Vec<Triple>,
    obs: std::vec::Vec<Obs>,
}

/// Cost, gradient and dense Hessian at the model's current values.
fn passes(w: &mut W) -> (f64, Vec<f64>, Vec<f64>) {
    let mut params = Vec::new();
    w.serialize(&mut params);
    let cost = w.calc_cost(&params);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    let cost2 = w.calc_grad_hessian_dense(&params, &mut grad, &mut hess);
    assert_eq!(cost, cost2);
    (cost, grad, hess)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

#[test]
fn partially_touched_spans_match_the_closed_form() {
    let (a, b, u, v, w) = (1.5, -0.7, 0.3, 2.0, 4.0);
    let (m, n) = (0.25, 0.1);
    let mut world = W { pairs: refs::Vec::new(), triples: refs::Vec::new(), obs: std::vec::Vec::new() };
    world.pairs.push(Pair { a: Param::new(a), b: Param::new(b), hb: SelfBlock::new() });
    world.triples.push(Triple { u: Param::new(u), v: Param::new(v), w: Param::new(w), hb: SelfBlock::new() });
    world.obs.push(Obs {
        pair: world.pairs.ref_at(0), triple: world.triples.ref_at(0), m, n, on: 1.0, hb: CrossBlock::new(),
    });
    let (cost, grad, hess) = passes(&mut world);
    // Parameters in serialization order: a, b, u, v, w.
    let r0 = a * v - m;
    let r1 = b - n;
    assert!(close(cost, r0 * r0 + r1 * r1));
    let expect_grad = [2.0 * r0 * v, 2.0 * r1, 0.0, 2.0 * r0 * a, 0.0];
    for (i, (g, e)) in grad.iter().zip(expect_grad).enumerate() {
        assert!(close(*g, e), "grad[{i}] {g} vs {e}");
    }
    // Hessian 2 J^T J: rows a, b, u, v, w.
    let j0 = [v, 0.0, 0.0, a, 0.0];
    let j1 = [0.0, 1.0, 0.0, 0.0, 0.0];
    for i in 0..5 {
        for j in 0..5 {
            let e = 2.0 * (j0[i] * j0[j] + j1[i] * j1[j]);
            assert!(close(hess[i * 5 + j], e), "hess[{i},{j}] {} vs {e}", hess[i * 5 + j]);
        }
    }
}

#[test]
fn the_guard_off_contributes_nothing() {
    let mut world = W { pairs: refs::Vec::new(), triples: refs::Vec::new(), obs: std::vec::Vec::new() };
    world.pairs.push(Pair { a: Param::new(1.5), b: Param::new(-0.7), hb: SelfBlock::new() });
    world.triples.push(Triple { u: Param::new(0.3), v: Param::new(2.0), w: Param::new(4.0), hb: SelfBlock::new() });
    world.obs.push(Obs {
        pair: world.pairs.ref_at(0), triple: world.triples.ref_at(0), m: 0.25, n: 0.1, on: -1.0, hb: CrossBlock::new(),
    });
    let (cost, grad, hess) = passes(&mut world);
    let r1 = -0.7 - 0.1;
    assert!(close(cost, r1 * r1));
    for (i, g) in grad.iter().enumerate() {
        let e = if i == 1 { 2.0 * r1 } else { 0.0 };
        assert!(close(*g, e), "grad[{i}] {g} vs {e}");
    }
    for i in 0..5 {
        for j in 0..5 {
            let e = if i == 1 && j == 1 { 2.0 } else { 0.0 };
            assert!(close(hess[i * 5 + j], e), "hess[{i},{j}]");
        }
    }
}
