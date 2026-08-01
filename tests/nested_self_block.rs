//! Stage 2 of nested-model-tree support: a block-bearing entity reachable
//! through a block-less grouping sub-model (`Root -> Vec<Group> -> Vec<Ent>`)
//! with a SELF constraint. Before nested support, `Ent`'s constraint resolved
//! to no root location and was silently dropped (the documented TODO gap);
//! here it must fire and wire its block indices two hops down.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::LmConfig;
use arael::simple_lm::LmProblem;

// A trivial entity pulled toward its own target: residual (x - target)*isigma.
// Independent per entity, so the exact minimizer is x = target and the Hessian
// is diagonal -- convergence to target proves the nested self-block both fired
// and had its parameter index wired (an unwired block contributes nothing and
// x would stay at its initial 0).
#[arael::model]
#[arael(constraint(hb, {
    [(ent.x - ent.target) * ent.isigma]
}))]
struct Ent {
    x: Param<f32>,
    target: f32,
    isigma: f32,
    hb: SelfBlock<Ent, f32>,
}

// Block-less grouping sub-model: no params, no SelfBlock, just holds entities.
#[arael::model]
struct Group {
    ents: std::vec::Vec<Ent>,
}

#[arael::model]
#[arael(root, f32)]
struct Root {
    groups: std::vec::Vec<Group>,
}

fn ent(target: f32) -> Ent {
    Ent { x: Param::new(0.0), target, isigma: 1.0, hb: SelfBlock::new() }
}

#[test]
fn nested_self_constraint_converges_dense() {
    let mut root = Root {
        groups: vec![
            Group { ents: vec![ent(3.0), ent(-2.0)] },
            Group { ents: vec![ent(5.0)] },
        ],
    };
    let cfg = LmConfig::<f32>::default();
    let result = root.solve_dense(&cfg).unwrap();
    assert!(result.end_cost < 1e-8, "cost did not converge: {}", result.end_cost);

    let expected: [&[f32]; 2] = [&[3.0, -2.0], &[5.0]];
    for (g, exp) in root.groups.iter().zip(expected.iter()) {
        assert_eq!(g.ents.len(), exp.len());
        for (e, &t) in g.ents.iter().zip(exp.iter()) {
            assert!((e.x.value - t).abs() < 1e-4,
                "nested entity x={} did not reach target {}", e.x.value, t);
        }
    }
}

#[test]
fn nested_self_constraint_converges_sparse() {
    let mut root = Root {
        groups: vec![
            Group { ents: vec![ent(1.5), ent(-4.0), ent(2.0)] },
            Group { ents: vec![ent(-1.0)] },
            Group { ents: vec![ent(0.5), ent(7.0)] },
        ],
    };
    let cfg = LmConfig::<f32>::default();
    let result = root.solve_sparse(&cfg).unwrap();
    assert!(result.end_cost < 1e-8, "cost did not converge: {}", result.end_cost);

    let expected: [&[f32]; 3] = [&[1.5, -4.0, 2.0], &[-1.0], &[0.5, 7.0]];
    for (g, exp) in root.groups.iter().zip(expected.iter()) {
        for (e, &t) in g.ents.iter().zip(exp.iter()) {
            assert!((e.x.value - t).abs() < 1e-4,
                "nested entity x={} did not reach target {}", e.x.value, t);
        }
    }
}
