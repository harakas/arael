// The RootProblem trait + the solve entry points as LmProblem default
// methods: every #[arael(root)] model implements RootProblem (serialize /
// deserialize round trip), which unlocks solve_with / solve_dense /
// solve_sparse on LmProblem -- usable generically, not just as inherent
// methods on a concrete struct.

use arael::model::{Model, Param, SelfBlock};
use arael::simple_lm::{Band, LmConfig, LmProblem, LmResult, RootProblem};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [m.x - m.tx, m.y - m.ty, m.y - m.x - 1.0]
}))]
struct M {
    x: Param<f64>,
    y: Param<f64>,
    tx: f64,
    ty: f64,
    hb: SelfBlock<M>,
}

fn model() -> M {
    M { x: Param::new(0.0), y: Param::new(0.0), tx: 2.0, ty: 3.0, hb: SelfBlock::new() }
}

// A generic helper over ANY root model -- the point of having the trait.
fn optimize<T, P>(m: &mut P, cfg: &LmConfig<T>) -> LmResult<T>
where
    T: arael::utils::Float,
    P: LmProblem<T> + RootProblem<T>,
    arael::simple_lm::Dense: arael::simple_lm::LmSolver<T>,
{
    m.solve_dense(cfg)
}

#[test]
fn solve_entry_points_via_traits() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut m = model();
    let r = m.solve_sparse(&cfg);
    assert!(r.end_cost < 1e-9, "solve_sparse: {}", r.end_cost);

    let mut m = model();
    let r = m.solve_dense(&cfg);
    assert!(r.end_cost < 1e-9, "solve_dense: {}", r.end_cost);

    let mut m = model();
    let r = m.solve_with(&mut Band::new(1), &cfg);
    assert!(r.end_cost < 1e-9, "solve_with(Band): {}", r.end_cost);

    // All three write the optimized values back into the model; the
    // system is consistent with the exact optimum at (2, 3).
    assert!((m.x.value - 2.0).abs() < 1e-6 && (m.y.value - 3.0).abs() < 1e-6,
        "optimum expected at (2, 3), got ({}, {})", m.x.value, m.y.value);
}

#[test]
fn generic_over_root_model() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let mut m = model();
    let r = optimize(&mut m, &cfg);
    assert!(r.end_cost < 1e-9, "generic optimize: {}", r.end_cost);
}

// RootProblem's round trip is the same one the suffixed inherent methods do.
#[test]
fn root_model_round_trip_matches_serialize64() {
    let mut a = model();
    let mut b = model();
    let mut va = Vec::new();
    let mut vb = Vec::new();
    RootProblem::serialize(&mut a, &mut va);
    b.serialize64(&mut vb);
    assert_eq!(va, vb);
    va[0] = 7.0;
    RootProblem::deserialize(&mut a, &va);
    b.deserialize64(&va);
    assert_eq!(a.x.value, 7.0);
    assert_eq!(a.x.value, b.x.value);
}
