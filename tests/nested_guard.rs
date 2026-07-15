// A `guard` on a nested (parent=) remote-block constraint must be honored.
// Regression: the nested-constraint emission path built its loop from the raw
// residual statements and never wrapped them in `if <guard>`, so a guarded
// constraint fired unconditionally (found via a robust curve fit where three
// mode-guarded constraints all activated at once).

use arael::model::{Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::LmProblem;

#[arael::model]
struct Curve {
    m: Param<f64>,
    hb: SelfBlock<Curve>,
}

// One residual per observation, gated on `active`. Nested in Batch; the
// Hessian block is a remote block on the referenced Curve.
#[arael::model]
#[arael(constraint(curve.hb, parent = batch, guard = self.active, {
    [obs.target - curve.m]
}))]
struct Obs {
    #[arael(ref = root.curves)]
    curve: Ref<Curve>,
    target: f64,
    active: bool,
}

#[arael::model]
struct Batch {
    obs: std::vec::Vec<Obs>,
}

#[arael::model]
#[arael(root)]
struct World {
    curves: refs::Vec<Curve>,
    batches: refs::Vec<Batch>,
}

fn world(active: bool) -> (World, Vec<f64>) {
    let mut w = World { curves: refs::Vec::new(), batches: refs::Vec::new() };
    let cref = w.curves.push(Curve { m: Param::new(0.0), hb: SelfBlock::new() });
    // residual = target - m = 3 - 0 = 3, so an active constraint costs 9.
    let obs = vec![Obs { curve: cref, target: 3.0, active }];
    w.batches.push(Batch { obs });
    let mut params = Vec::new();
    w.serialize64(&mut params);
    (w, params)
}

#[test]
fn guard_false_filters_the_nested_constraint() {
    let (mut w, params) = world(false);
    // Guard is false: the constraint must not contribute. Cost must be 0.
    assert_eq!(w.calc_cost(&params), 0.0);
}

#[test]
fn guard_true_activates_the_nested_constraint() {
    let (mut w, params) = world(true);
    // Guard is true: residual 3, cost 9.
    assert!((w.calc_cost(&params) - 9.0).abs() < 1e-12);
}

// The gradient/Hessian sweep must honor the guard too (not just the cost-only
// path): a filtered constraint contributes nothing to the gradient.
#[test]
fn guard_false_filters_gradient_and_hessian() {
    let (mut w, params) = world(false);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    w.calc_grad_hessian_dense(&params, &mut grad, &mut hess);
    assert!(grad.iter().all(|&g| g == 0.0), "grad = {:?}", grad);
    assert!(hess.iter().all(|&h| h == 0.0), "hess = {:?}", hess);
}
