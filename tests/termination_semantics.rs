// Termination-semantics tests for the LM solver (REVIEW2 R1).
//
// A step counts as "small" when the ABSOLUTE cost decrease OR the RELATIVE
// cost decrease is below its threshold (either suffices). After `patience`
// consecutive small steps the solve stops. These tests pin that OR, plus
// max_iters and patience, on a genuinely nonlinear model -- a spring chain.
// (A linear least-squares fit converges in one step and never exercises the
// small-step tail, so it cannot test this.)

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::simple_lm::LmConfig;
use arael::vect::vect2d;
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, guard = self.is_anchor, {
    [point.pos.x * chain.anchor, point.pos.y * chain.anchor]
}))]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * chain.drift, d.y * chain.drift]
}))]
struct Point {
    pos: Param<vect2d>,
    is_anchor: bool,
    hb: SelfBlock<Point>,
}
#[arael::model]
#[arael(constraint(hb, {
    let d = b.pos - a.pos;
    [(d.norm() - link.rest) * chain.spring]
}))]
struct Link {
    #[arael(ref = root.points)] a: Ref<Point>,
    #[arael(ref = root.points)] b: Ref<Point>,
    rest: f64,
    hb: CrossBlock<Point, Point>,
}
#[arael::model]
#[arael(root)]
struct Chain {
    points: refs::Vec<Point>,
    links: std::vec::Vec<Link>,
    anchor: f64,
    drift: f64,
    spring: f64,
}

const N: usize = 8;

/// Zig-zag start so the springs (rest 1.0) must relax over several LM steps.
fn build_chain() -> Chain {
    let mut c = Chain {
        points: refs::Vec::new(), links: std::vec::Vec::new(),
        anchor: 100.0, drift: 0.01, spring: 1.0,
    };
    for i in 0..N {
        let pos = vect2d::new(i as f64 * 0.5, if i % 2 == 0 { 0.7 } else { -0.7 });
        c.points.push(Point { pos: Param::new(pos), is_anchor: i == 0, hb: SelfBlock::new() });
    }
    for i in 1..N {
        let a = c.points.ref_at(i - 1);
        let b = c.points.ref_at(i);
        c.links.push(Link { a, b, rest: 1.0, hb: CrossBlock::new() });
    }
    c
}

#[test]
fn converges_before_max_iters() {
    let mut c = build_chain();
    let r = c.solve_sparse(&LmConfig { max_iters: 200, ..Default::default() });
    assert!(r.iterations < 200, "should converge before the cap, took {}", r.iterations);
    assert!(r.end_cost < r.start_cost, "cost should decrease: {} -> {}", r.start_cost, r.end_cost);
    assert!(r.accepted_iterations >= 1);
}

#[test]
fn max_iters_is_respected() {
    for k in [1usize, 2, 5, 10] {
        let mut c = build_chain();
        let r = c.solve_sparse(&LmConfig { max_iters: k, min_iters: 0, ..Default::default() });
        assert!(r.iterations <= k, "iterations {} exceeded max_iters {}", r.iterations, k);
    }
}

#[test]
fn or_each_criterion_stops_independently() {
    // ABS arm alone: rel_precision 0 can never fire, so only the absolute
    // criterion (here always true, threshold huge) can stop the solve.
    let mut ca = build_chain();
    let ra = ca.solve_sparse(&LmConfig {
        abs_precision: 1e30, rel_precision: 0.0,
        min_iters: 0, patience: 1, max_iters: 200, ..Default::default() });
    // REL arm alone: abs_precision 0 can never fire.
    let mut cr = build_chain();
    let rr = cr.solve_sparse(&LmConfig {
        abs_precision: 0.0, rel_precision: 1e30,
        min_iters: 0, patience: 1, max_iters: 200, ..Default::default() });
    // Neither loose -> runs to real convergence.
    let mut ct = build_chain();
    let rt = ct.solve_sparse(&LmConfig {
        abs_precision: 1e-14, rel_precision: 1e-14,
        min_iters: 0, patience: 3, max_iters: 200, ..Default::default() });

    // Each single loose arm halts the solve well before convergence...
    assert!(ra.iterations < rt.iterations,
        "loose-abs stopped at {} iters, tight took {}", ra.iterations, rt.iterations);
    assert!(rr.iterations < rt.iterations,
        "loose-rel stopped at {} iters, tight took {}", rr.iterations, rt.iterations);
    // ...leaving more cost on the table than the fully-converged solve.
    assert!(ra.end_cost > rt.end_cost, "loose-abs should be less converged");
    assert!(rr.end_cost > rt.end_cost, "loose-rel should be less converged");
}

#[test]
fn patience_controls_stop() {
    // With every step deemed "small" (rel threshold huge), the solve stops
    // after exactly `patience` accepted small steps, so more patience runs
    // longer.
    let cfg = |patience| LmConfig {
        rel_precision: 1e30, abs_precision: 0.0,
        min_iters: 0, patience, max_iters: 200, ..Default::default() };
    let mut c2 = build_chain();
    let r2 = c2.solve_sparse(&cfg(2));
    let mut c8 = build_chain();
    let r8 = c8.solve_sparse(&cfg(8));
    assert!(r8.iterations > r2.iterations,
        "patience 8 ({}) should run longer than patience 2 ({})", r8.iterations, r2.iterations);
}
