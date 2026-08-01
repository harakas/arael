// Integration tests for Jacobian computation via #[arael(root, jacobian)].

use arael::simple_lm::RootProblem;
use arael::model::{CrossBlock, JacobianModel, Param, SelfBlock};
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
        a: model.points.ref_at(0),
        b: model.points.ref_at(1),
        ci: 0,
        hb: CrossBlock::new(),
    });
    let mut params = Vec::new();
    model.serialize(&mut params);
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
    assert_eq!(model.points[0].ci, 0);
    assert_eq!(model.points[1].ci, 1);
    assert_eq!(model.points[2].ci, 2);
    assert_eq!(model.coincidents[0].ci, 3);
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
        a: model.points.ref_at(0),
        b: model.points.ref_at(1),
        ci: 0, hb: CrossBlock::new(),
    });
    let mut params = Vec::new();
    model.serialize(&mut params);

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
    model.points[0].constraints.has_fix_x = true;
    model.points[0].constraints.fix_x = 1.0;
    let j = model.calc_jacobian(&params);
    // 8 + 1 fix_x = 9
    assert_eq!(j.num_residuals(), 9);

    // Enable fix_y on point 0 too
    model.points[0].constraints.has_fix_y = true;
    model.points[0].constraints.fix_y = 2.0;
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

// --- Robustified cost table: the loss is applied per block ---

#[arael::model]
#[arael(constraint(hb, name = "robust", loss = |s| loss_geman_mcclure(s, lossworld.c2), {
    [(lossitem.v - lossitem.target) * 10.0]
}))]
struct LossItem {
    v: Param<f64>,
    target: f64,
    hb: SelfBlock<LossItem>,
}

#[arael::model]
#[arael(constraint(hb, name = "anchor", {
    [lossworld.a - lossworld.a_target]
}))]
#[arael(root, jacobian)]
struct LossWorld {
    a: Param<f64>,
    a_target: f64,
    c2: f64,
    items: arael::refs::Vec<LossItem>,
    hb: SelfBlock<LossWorld>,
}

fn loss_world() -> (LossWorld, Vec<f64>) {
    let mut w = LossWorld {
        a: Param::new(1.0),
        a_target: 3.0,
        c2: 2.99,
        items: arael::refs::Vec::new(),
        hb: SelfBlock::new(),
    };
    // One inlier and one gross outlier; Geman-McClure saturates the
    // outlier's block below c2.
    w.items.push(LossItem { v: Param::new(0.01), target: 0.0, hb: SelfBlock::new() });
    w.items.push(LossItem { v: Param::new(50.0), target: 0.0, hb: SelfBlock::new() });
    let mut x = Vec::new();
    arael::simple_lm::RootProblem::serialize(&mut w, &mut x);
    (w, x)
}

/// calc_cost_table reports the ROBUSTIFIED cost per label (rho(s) per
/// block), summing to calc_cost.
#[test]
fn cost_table_applies_the_robust_loss() {
    let (mut w, x) = loss_world();
    let table = w.calc_cost_table(&x);
    let cost = w.calc_cost(&x);

    // The table is the robustified cost split by label: it sums to
    // calc_cost (up to accumulation order).
    let total: f64 = table.values().sum();
    assert!((total - cost).abs() <= 1e-12 * (1.0 + cost.abs()),
        "table total {} vs cost {}", total, cost);

    // Raw outlier mass is (50*10)^2 = 2.5e5; robustified it caps near c2.
    let robust = table["robust"];
    assert!(robust < 2.0 * 2.99, "robust label not robustified: {}", robust);

    // A label without a loss is the plain squared-residual sum.
    assert!((table["anchor"] - 4.0).abs() < 1e-12, "anchor {}", table["anchor"]);
}

/// calc_jacobian scales rows and entries by sqrt(rho'(s)), so the
/// weighted rows reproduce the assembled Gauss-Newton system
/// (2 J^T r the gradient, 2 J^T J the Hessian).
#[test]
fn jacobian_applies_the_robust_loss() {
    let (mut w, x) = loss_world();
    let j = w.calc_jacobian(&x);

    // Row-square total per label is rho'(s)*s per block.
    let weighted: f64 = j.rows.iter()
        .filter(|r| r.label == "robust")
        .map(|r| r.residual * r.residual)
        .sum();
    let rho_w = |s: f64| (2.99 / (2.99 + s)).powi(2);
    let expected = rho_w(0.01) * 0.01 + rho_w(250_000.0) * 250_000.0;
    assert!((weighted - expected).abs() < 1e-9 * (1.0 + expected),
        "weighted {} vs expected {}", weighted, expected);

    // 2 J^T r reproduces the assembled gradient and 2 J^T J the
    // assembled Gauss-Newton Hessian.
    let n = j.num_params;
    let mut gj = vec![0.0; n];
    let mut hj = vec![0.0; n * n];
    for row in &j.rows {
        for &(i, di) in &row.entries {
            gj[i as usize] += 2.0 * row.residual * di;
            for &(k, dk) in &row.entries {
                hj[i as usize * n + k as usize] += 2.0 * di * dk;
            }
        }
    }
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    w.calc_grad_hessian_dense(&x, &mut grad, &mut hess);
    for i in 0..n {
        assert!((gj[i] - grad[i]).abs() < 1e-9 * (1.0 + grad[i].abs()),
            "grad[{}]: 2 J^T r {} vs assembled {}", i, gj[i], grad[i]);
    }
    for i in 0..n * n {
        assert!((hj[i] - hess[i]).abs() < 1e-9 * (1.0 + hess[i].abs()),
            "hessian[{}]: J^T J {} vs assembled {}", i, hj[i], hess[i]);
    }
}
