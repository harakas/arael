// Test that guards on TripletBlock constraints are applied correctly.
// This is a regression test for a bug where TripletBlock guards were
// silently ignored in cost and grad/hessian computation.

use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock, CrossBlock, TripletBlock};
use arael::simple_lm::LmProblem;
use arael::vect::vect2d;

// Minimal point entity: just position + drift
#[arael::model]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * testmodel.drift_isigma, d.y * testmodel.drift_isigma]
}))]
struct Point {
    pos: Param<vect2d>,
    pos_value: vect2d,
    hb: SelfBlock<Point>,
}

// CrossBlock constraint with guard -- guards work correctly here
#[arael::model]
#[arael(constraint(hb, guard = self.active, {
    [(a.pos.x - b.pos.x) * testmodel.constraint_isigma]
}))]
struct GuardedCross {
    #[arael(ref = root.points)]
    a: arael::refs::Ref<Point>,
    #[arael(ref = root.points)]
    b: arael::refs::Ref<Point>,
    #[arael(skip)]
    active: bool,
    hb: CrossBlock<Point, Point>,
}

// TripletBlock constraint with guard -- guards were NOT applied (bug)
#[arael::model]
#[arael(constraint(hb, guard = self.active, {
    [(a.pos.x - b.pos.x) * testmodel.constraint_isigma]
}))]
struct GuardedTriplet {
    #[arael(ref = root.points)]
    a: arael::refs::Ref<Point>,
    #[arael(ref = root.points)]
    b: arael::refs::Ref<Point>,
    #[arael(ref = root.points)]
    c: arael::refs::Ref<Point>,
    #[arael(skip)]
    active: bool,
    hb: TripletBlock<f64>,
}

// Root model with both constraint types
#[arael::model]
#[arael(root)]
struct TestModel {
    points: arael::refs::Vec<Point>,
    guarded_cross: arael::refs::Vec<GuardedCross>,
    guarded_triplet: arael::refs::Vec<GuardedTriplet>,
    drift_isigma: f64,
    constraint_isigma: f64,
}

fn make_points() -> arael::refs::Vec<Point> {
    let mut pts = arael::refs::Vec::new();
    pts.push(Point {
        pos: Param::new(vect2d::new(1.0, 0.0)),
        pos_value: vect2d::new(1.0, 0.0),
        hb: SelfBlock::new(),
    });
    pts.push(Point {
        pos: Param::new(vect2d::new(3.0, 0.0)),
        pos_value: vect2d::new(3.0, 0.0),
        hb: SelfBlock::new(),
    });
    pts.push(Point {
        pos: Param::new(vect2d::new(5.0, 0.0)),
        pos_value: vect2d::new(5.0, 0.0),
        hb: SelfBlock::new(),
    });
    pts
}

fn make_cross_model(active: bool) -> (TestModel, Vec<f64>) {
    let mut model = TestModel {
        points: make_points(),
        guarded_cross: arael::refs::Vec::new(),
        guarded_triplet: arael::refs::Vec::new(),
        drift_isigma: 0.0,
        constraint_isigma: 1000.0,
    };
    model.guarded_cross.push(GuardedCross {
        a: model.points.ref_at(0),
        b: model.points.ref_at(1),
        active,
        hb: CrossBlock::new(),
    });
    let mut params = Vec::new();
    model.serialize(&mut params);
    (model, params)
}

fn make_triplet_model(active: bool) -> (TestModel, Vec<f64>) {
    let mut model = TestModel {
        points: make_points(),
        guarded_cross: arael::refs::Vec::new(),
        guarded_triplet: arael::refs::Vec::new(),
        drift_isigma: 0.0,
        constraint_isigma: 1000.0,
    };
    model.guarded_triplet.push(GuardedTriplet {
        a: model.points.ref_at(0),
        b: model.points.ref_at(1),
        c: model.points.ref_at(2),
        active,
        hb: TripletBlock::new(),
    });
    let mut params = Vec::new();
    model.serialize(&mut params);
    (model, params)
}

#[test]
fn test_crossblock_guard_disables_constraint() {
    let (mut model, params) = make_cross_model(false);
    let cost = model.calc_cost(&params);
    assert!(cost < 1e-10, "CrossBlock guard=false should produce zero cost, got {}", cost);
}

#[test]
fn test_crossblock_guard_enables_constraint() {
    let (mut model, params) = make_cross_model(true);
    let cost = model.calc_cost(&params);
    assert!(cost > 1.0, "CrossBlock guard=true should produce nonzero cost, got {}", cost);
}

#[test]
fn test_tripletblock_guard_disables_constraint() {
    // THIS IS THE BUG: TripletBlock guard=false should produce zero cost but doesn't
    let (mut model, params) = make_triplet_model(false);
    let cost = model.calc_cost(&params);
    assert!(cost < 1e-10, "TripletBlock guard=false should produce zero cost, got {}", cost);
}

#[test]
fn test_tripletblock_guard_enables_constraint() {
    let (mut model, params) = make_triplet_model(true);
    let cost = model.calc_cost(&params);
    assert!(cost > 1.0, "TripletBlock guard=true should produce nonzero cost, got {}", cost);
}

#[test]
fn test_cross_and_triplet_guards_produce_same_hessian() {
    // Both active: same constraint math, should produce equivalent hessian contributions
    let (mut cross_model, cross_params) = make_cross_model(true);
    let (mut triplet_model, triplet_params) = make_triplet_model(true);
    let n = cross_params.len();
    let mut cross_grad = vec![0.0; n];
    let mut cross_hess = vec![0.0; n * n];
    cross_model.calc_grad_hessian_dense(&cross_params, &mut cross_grad, &mut cross_hess);
    let mut triplet_grad = vec![0.0; n];
    let mut triplet_hess = vec![0.0; n * n];
    triplet_model.calc_grad_hessian_dense(&triplet_params, &mut triplet_grad, &mut triplet_hess);
    for i in 0..n {
        assert!((cross_grad[i] - triplet_grad[i]).abs() < 1e-6,
            "grad[{}] differs: cross={} triplet={}", i, cross_grad[i], triplet_grad[i]);
    }
}
