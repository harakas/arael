// Two #[arael(root)] models in one crate: each root must generate a
// working solver for its own reachable constraints. The constraint
// registry used to be consumed with mem::take by the first root,
// leaving the second root an empty list -- it compiled cleanly and
// silently optimized nothing.

use arael::model::{Model, Param, SelfBlock};
use arael::simple_lm::{self, LmConfig};
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, {
    [(aitem.x - aitem.target) * alpha.isigma]
}))]
struct AItem {
    x: Param<f64>,
    target: f64,
    hb: SelfBlock<AItem>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(bitem.x - bitem.target) * beta.isigma]
}))]
struct BItem {
    x: Param<f64>,
    target: f64,
    hb: SelfBlock<BItem>,
}

// An entity shared by both roots. Its constraint must be root-agnostic
// (entity fields only): bodies name the root by its lowercase type name,
// so a root-field reference could only ever compile against one root.
#[arael::model]
#[arael(constraint(hb, {
    [(shared.x - shared.target) * shared.isigma]
}))]
struct Shared {
    x: Param<f64>,
    target: f64,
    isigma: f64,
    hb: SelfBlock<Shared>,
}

// Both roots come after both entities: the first root's registry take
// used to also consume the second root's stashed constraints.
#[arael::model]
#[arael(root)]
struct Alpha {
    items: refs::Vec<AItem>,
    shareds: refs::Vec<Shared>,
    isigma: f64,
}

#[arael::model]
#[arael(root)]
struct Beta {
    items: refs::Vec<BItem>,
    shareds: refs::Vec<Shared>,
    isigma: f64,
}

#[test]
fn shared_entity_optimizes_under_both_roots() {
    let mut alpha = Alpha { items: refs::Vec::new(), shareds: refs::Vec::new(), isigma: 1.0 };
    alpha.shareds.push(Shared { x: Param::new(0.0), target: 2.0, isigma: 1.0, hb: SelfBlock::new() });
    let mut params = Vec::new();
    alpha.serialize64(&mut params);
    let ra = simple_lm::solve(&params, &mut alpha, &LmConfig::default()).unwrap();
    assert!(ra.end_cost < 1e-12, "shared under Alpha, cost={}", ra.end_cost);

    let mut beta = Beta { items: refs::Vec::new(), shareds: refs::Vec::new(), isigma: 1.0 };
    beta.shareds.push(Shared { x: Param::new(0.0), target: -3.0, isigma: 1.0, hb: SelfBlock::new() });
    let mut params = Vec::new();
    beta.serialize64(&mut params);
    let rb = simple_lm::solve(&params, &mut beta, &LmConfig::default()).unwrap();
    assert!(rb.end_cost < 1e-12, "shared under Beta, cost={}", rb.end_cost);
    beta.deserialize64(&rb.x);
    assert!((beta.shareds[0].x.value + 3.0).abs() < 1e-6);
}

#[test]
fn first_root_optimizes() {
    let mut alpha = Alpha { items: refs::Vec::new(), shareds: refs::Vec::new(), isigma: 1.0 };
    alpha.items.push(AItem { x: Param::new(0.0), target: 5.0, hb: SelfBlock::new() });
    let mut params = Vec::new();
    alpha.serialize64(&mut params);
    let result = simple_lm::solve(&params, &mut alpha, &LmConfig::default()).unwrap();
    assert!(result.end_cost < 1e-12, "cost={}", result.end_cost);
    alpha.deserialize64(&result.x);
    assert!((alpha.items[0].x.value - 5.0).abs() < 1e-6);
}

#[test]
fn second_root_optimizes() {
    let mut beta = Beta { items: refs::Vec::new(), shareds: refs::Vec::new(), isigma: 1.0 };
    beta.items.push(BItem { x: Param::new(0.0), target: 5.0, hb: SelfBlock::new() });
    let mut params = Vec::new();
    beta.serialize64(&mut params);
    let result = simple_lm::solve(&params, &mut beta, &LmConfig::default()).unwrap();
    assert!(result.end_cost < 1e-12,
        "second root must generate a working solver, cost={}", result.end_cost);
    beta.deserialize64(&result.x);
    assert!((beta.items[0].x.value - 5.0).abs() < 1e-6);
}
