// Integration tests for Jacobian computation via #[arael(root, jacobian)].

use arael::model::{Param, SelfBlock, CrossBlock, Model};
use arael::simple_lm::LmProblem;
use arael::vect::vect2d;

// --- Test model: points with drift + optional fix, and coincident constraints ---

#[arael::model]
struct PointConstraints {
    #[arael(skip)]
    has_fix_x: bool,
    fix_x: f64,
    #[arael(skip)]
    has_fix_y: bool,
    fix_y: f64,
}

#[arael::model]
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * testmodel.drift_isigma, d.y * testmodel.drift_isigma]
}))]
#[arael(constraint(hb, guard = self.constraints.has_fix_x, {
    [(point.pos.x - point.constraints.fix_x) * testmodel.constraint_isigma]
}))]
#[arael(constraint(hb, guard = self.constraints.has_fix_y, {
    [(point.pos.y - point.constraints.fix_y) * testmodel.constraint_isigma]
}))]
struct Point {
    pos: Param<vect2d>,
    pos_value: vect2d,
    constraints: PointConstraints,
    #[arael(constraint_index)]
    ci: u32,
    hb: SelfBlock<Point>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.pos.x - b.pos.x) * testmodel.constraint_isigma,
     (a.pos.y - b.pos.y) * testmodel.constraint_isigma]
}))]
struct Coincident {
    #[arael(ref = root.points)]
    a: arael::refs::Ref<Point>,
    #[arael(ref = root.points)]
    b: arael::refs::Ref<Point>,
    #[arael(constraint_index)]
    ci: u32,
    hb: CrossBlock<Point, Point>,
}

#[arael::model]
#[arael(root, jacobian)]
struct TestModel {
    points: arael::refs::Vec<Point>,
    coincidents: arael::refs::Vec<Coincident>,
    drift_isigma: f64,
    constraint_isigma: f64,
}

fn make_test_model() -> (TestModel, Vec<f64>) {
    let mut model = TestModel {
        points: arael::refs::Vec::new(),
        coincidents: arael::refs::Vec::new(),
        drift_isigma: 1.0,
        constraint_isigma: 10.0,
    };
    model.points.push(Point {
        pos: Param::new(vect2d::new(1.0, 2.0)),
        pos_value: vect2d::new(1.0, 2.0),
        constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
        ci: 0,
        hb: SelfBlock::new(),
    });
    model.points.push(Point {
        pos: Param::new(vect2d::new(3.0, 4.0)),
        pos_value: vect2d::new(3.0, 4.0),
        constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
        ci: 0,
        hb: SelfBlock::new(),
    });
    model.points.push(Point {
        pos: Param::new(vect2d::new(5.0, 6.0)),
        pos_value: vect2d::new(5.0, 6.0),
        constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
        ci: 0,
        hb: SelfBlock::new(),
    });
    model.coincidents.push(Coincident {
        a: arael::refs::Ref::new(0),
        b: arael::refs::Ref::new(1),
        ci: 0,
        hb: CrossBlock::new(),
    });
    let mut params = Vec::new();
    model.serialize64(&mut params);
    (model, params)
}

#[test]
fn jacobian_dimensions() {
    let (mut model, params) = make_test_model();
    let j = model.calc_jacobian(&params);

    // 3 points * 2 drift residuals = 6, 1 coincident * 2 residuals = 2 -> 8 total
    assert_eq!(j.num_residuals(), 8, "expected 8 residuals, got {}", j.num_residuals());
    // 3 points * 2 params = 6
    assert_eq!(j.num_params, 6, "expected 6 params, got {}", j.num_params);
}

#[test]
fn jacobian_cost_matches_calc_cost() {
    let (mut model, params) = make_test_model();
    let j = model.calc_jacobian(&params);

    let cost_from_jacobian: f64 = j.rows.iter().map(|r| r.residual * r.residual).sum();
    let cost_from_calc = model.calc_cost(&params);

    assert!(
        (cost_from_jacobian - cost_from_calc).abs() < 1e-12,
        "cost mismatch: jacobian={}, calc_cost={}", cost_from_jacobian, cost_from_calc
    );
}

#[test]
fn jacobian_jtj_matches_hessian() {
    // Perturb params so residuals are non-zero
    let (mut model, mut params) = make_test_model();
    params[0] += 0.1;
    params[1] += 0.2;
    params[2] -= 0.3;
    params[5] += 0.5;

    let j = model.calc_jacobian(&params);
    let dense = j.to_dense();
    let m = j.num_residuals();
    let n = j.num_params;

    // Compute J^T * J
    let mut jtj = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let mut sum = 0.0;
            for r in 0..m {
                sum += dense[r * n + i] * dense[r * n + k];
            }
            jtj[i * n + k] = sum;
        }
    }

    // Get Hessian (which accumulates 2 * J^T * J)
    let mut grad = vec![0.0f64; n];
    let mut hessian = vec![0.0f64; n * n];
    model.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);

    // Hessian should equal 2 * J^T * J
    for i in 0..n {
        for k in 0..n {
            let expected = 2.0 * jtj[i * n + k];
            let actual = hessian[i * n + k];
            assert!(
                (expected - actual).abs() < 1e-10,
                "H[{},{}] mismatch: 2*JtJ={}, H={}", i, k, expected, actual
            );
        }
    }
}

#[test]
fn jacobian_constraint_ids() {
    let (mut model, params) = make_test_model();
    let j = model.calc_jacobian(&params);

    // Points 0,1,2 get constraint IDs 0,1,2; coincident gets 3
    let point_ids: Vec<u32> = j.rows.iter().take(6).map(|r| r.constraint).collect();
    assert_eq!(point_ids, vec![0, 0, 1, 1, 2, 2], "point constraint IDs: {:?}", point_ids);

    let coinc_ids: Vec<u32> = j.rows.iter().skip(6).map(|r| r.constraint).collect();
    assert_eq!(coinc_ids, vec![3, 3], "coincident constraint IDs: {:?}", coinc_ids);

    // Verify constraint_index fields on structs match
    assert_eq!(model.points[arael::refs::Ref::new(0)].ci, 0);
    assert_eq!(model.points[arael::refs::Ref::new(1)].ci, 1);
    assert_eq!(model.points[arael::refs::Ref::new(2)].ci, 2);
    assert_eq!(model.coincidents[arael::refs::Ref::new(0)].ci, 3);
}

#[test]
fn jacobian_fixed_params() {
    let mut model = TestModel {
        points: arael::refs::Vec::new(),
        coincidents: arael::refs::Vec::new(),
        drift_isigma: 1.0,
        constraint_isigma: 10.0,
    };
    model.points.push(Point {
        pos: Param::fixed(vect2d::new(1.0, 2.0)), // FIXED
        pos_value: vect2d::new(1.0, 2.0),
        constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
        ci: 0, hb: SelfBlock::new(),
    });
    model.points.push(Point {
        pos: Param::new(vect2d::new(3.0, 4.0)),
        pos_value: vect2d::new(3.0, 4.0),
        constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
        ci: 0, hb: SelfBlock::new(),
    });
    model.coincidents.push(Coincident {
        a: arael::refs::Ref::new(0),
        b: arael::refs::Ref::new(1),
        ci: 0, hb: CrossBlock::new(),
    });
    let mut params = Vec::new();
    model.serialize64(&mut params);

    // Only 2 params (point 1 x,y), not 4
    assert_eq!(params.len(), 2);

    let j = model.calc_jacobian(&params);
    assert_eq!(j.num_params, 2);

    // Point 0 (fixed) drift residuals should have no entries
    for row in &j.rows[0..2] {
        assert!(row.entries.is_empty(), "fixed point should have no Jacobian entries, got {:?}", row.entries);
    }

    // Point 1 drift residuals should reference params 0,1
    assert!(!j.rows[2].entries.is_empty());

    // Coincident: only B's params (point 1) appear since A (point 0) is fixed
    for row in &j.rows[4..6] {
        for &(idx, _) in &row.entries {
            assert!(idx < 2, "coincident entry should reference params 0 or 1, got {}", idx);
        }
    }
}

#[test]
fn jacobian_guarded_constraints() {
    let (mut model, params) = make_test_model();

    // Without guards: 3*2 drift + 1*2 coincident = 8
    let j = model.calc_jacobian(&params);
    assert_eq!(j.num_residuals(), 8);

    // Enable fix_x on point 0
    model.points[arael::refs::Ref::new(0)].constraints.has_fix_x = true;
    model.points[arael::refs::Ref::new(0)].constraints.fix_x = 1.0;
    let j = model.calc_jacobian(&params);
    // 8 + 1 fix_x = 9
    assert_eq!(j.num_residuals(), 9);

    // Enable fix_y on point 0 too
    model.points[arael::refs::Ref::new(0)].constraints.has_fix_y = true;
    model.points[arael::refs::Ref::new(0)].constraints.fix_y = 2.0;
    let j = model.calc_jacobian(&params);
    // 8 + 1 fix_x + 1 fix_y = 10
    assert_eq!(j.num_residuals(), 10);
}

#[test]
fn jacobian_gradient_matches() {
    // Verify: gradient = 2 * J^T * r
    let (mut model, mut params) = make_test_model();
    params[0] += 0.1;
    params[1] += 0.2;

    let j = model.calc_jacobian(&params);
    let n = j.num_params;

    // Compute 2 * J^T * r
    let mut grad_from_j = vec![0.0f64; n];
    for row in &j.rows {
        for &(idx, d) in &row.entries {
            grad_from_j[idx as usize] += 2.0 * row.residual * d;
        }
    }

    // Get gradient from calc_grad_hessian_dense
    let mut grad = vec![0.0f64; n];
    let mut hessian = vec![0.0f64; n * n];
    model.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);

    for i in 0..n {
        assert!(
            (grad_from_j[i] - grad[i]).abs() < 1e-10,
            "grad[{}] mismatch: from_J={}, from_GH={}", i, grad_from_j[i], grad[i]
        );
    }
}
