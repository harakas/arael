// Root-level Params referenced from nested constraint bodies.
//
// Verifies that a constraint body can reference root-level `Param` fields
// alongside Ref-entity params, and that gradient + full Hessian match
// numerical derivatives. The root joins the constraint's entity list as
// an implicit participant when any declared CrossBlock references the
// root type.

#[allow(unused_imports)]
use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::simple_lm::LmProblem;

#[arael::model]
struct ItemA {
    a: Param<f64>,
    hb: SelfBlock<ItemA>,
}

#[arael::model]
struct ItemB {
    b: Param<f64>,
    hb: SelfBlock<ItemB>,
}

// Three-entity constraint that couples two Refs *and* the root via
// `testmodel.offset`. Three CrossBlocks -- one per unordered pair.
// Linear residuals: the Gauss-Newton Hessian J^T J equals the true
// Hessian of the sum-of-squares cost, so f.d. of the cost matches the
// analytic block Hessian exactly. Quadratic terms would introduce a
// 2*r*d^2r correction that's dropped by Gauss-Newton.
#[arael::model]
#[arael(constraint([hb_ab, hb_ar, hb_br], {
    [(a.a + b.b - testmodel.offset) * testmodel.isigma,
     (2.0 * a.a - 3.0 * b.b + testmodel.offset) * testmodel.isigma]
}))]
struct TripleCouple {
    #[arael(ref = root.items_a)] a: arael::refs::Ref<ItemA>,
    #[arael(ref = root.items_b)] b: arael::refs::Ref<ItemB>,
    hb_ab: CrossBlock<ItemA, ItemB>,
    hb_ar: CrossBlock<ItemA, TestModel>,
    hb_br: CrossBlock<ItemB, TestModel>,
}

#[arael::model]
#[arael(root)]
struct TestModel {
    items_a: arael::refs::Vec<ItemA>,
    items_b: arael::refs::Vec<ItemB>,
    couples: arael::refs::Vec<TripleCouple>,
    offset: Param<f64>,
    isigma: f64,
    hb: SelfBlock<TestModel>,
}

fn build_model() -> (TestModel, Vec<f64>) {
    let mut items_a = arael::refs::Vec::new();
    items_a.push(ItemA { a: Param::new(1.3), hb: SelfBlock::new() });
    items_a.push(ItemA { a: Param::new(-0.4), hb: SelfBlock::new() });
    let mut items_b = arael::refs::Vec::new();
    items_b.push(ItemB { b: Param::new(0.7), hb: SelfBlock::new() });
    items_b.push(ItemB { b: Param::new(2.1), hb: SelfBlock::new() });
    let mut couples = arael::refs::Vec::new();
    couples.push(TripleCouple {
        a: items_a.ref_at(0),
        b: items_b.ref_at(0),
        hb_ab: CrossBlock::new(),
        hb_ar: CrossBlock::new(),
        hb_br: CrossBlock::new(),
    });
    couples.push(TripleCouple {
        a: items_a.ref_at(1),
        b: items_b.ref_at(1),
        hb_ab: CrossBlock::new(),
        hb_ar: CrossBlock::new(),
        hb_br: CrossBlock::new(),
    });
    let mut model = TestModel {
        items_a, items_b, couples,
        offset: Param::new(0.3),
        isigma: 2.0,
        hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    model.serialize(&mut params);
    (model, params)
}

fn numerical_grad_hessian(model: &mut TestModel, params: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = params.len();
    let eps_g = 1e-6;
    let eps_h = 1e-4;
    let mut grad = vec![0.0; n];
    for i in 0..n {
        let mut pp = params.to_vec();
        pp[i] += eps_g;
        let cp = model.calc_cost(&pp);
        pp[i] -= 2.0 * eps_g;
        let cm = model.calc_cost(&pp);
        grad[i] = (cp - cm) / (2.0 * eps_g);
    }
    let mut hess = vec![0.0; n * n];
    for i in 0..n {
        for j in i..n {
            let h: f64 = if i == j {
                let c0 = model.calc_cost(params);
                let mut pp = params.to_vec();
                pp[i] += eps_h;
                let cp = model.calc_cost(&pp);
                pp[i] -= 2.0 * eps_h;
                let cm = model.calc_cost(&pp);
                (cp - 2.0 * c0 + cm) / (eps_h * eps_h)
            } else {
                let mut pp = params.to_vec();
                pp[i] += eps_h; pp[j] += eps_h;
                let cpp = model.calc_cost(&pp);
                pp[j] -= 2.0 * eps_h;
                let cpm = model.calc_cost(&pp);
                pp[i] -= 2.0 * eps_h;
                let cmm = model.calc_cost(&pp);
                pp[j] += 2.0 * eps_h;
                let cmp = model.calc_cost(&pp);
                (cpp - cpm - cmp + cmm) / (4.0 * eps_h * eps_h)
            };
            hess[i * n + j] = h;
            hess[j * n + i] = h;
        }
    }
    (grad, hess)
}

#[test]
fn test_root_params_grad_hessian_matches_numerical() {
    let (mut model, params) = build_model();
    let n = params.len();

    let mut ag = vec![0.0; n];
    let mut ah = vec![0.0; n * n];
    model.calc_grad_hessian_dense(&params, &mut ag, &mut ah);

    let (ng, nh) = numerical_grad_hessian(&mut model, &params);

    for i in 0..n {
        assert!((ag[i] - ng[i]).abs() < 1e-4,
            "grad[{}] analytic={} numerical={}", i, ag[i], ng[i]);
    }
    for i in 0..n {
        for j in 0..n {
            let idx = i * n + j;
            let d = (ah[idx] - nh[idx]).abs();
            assert!(d < 5e-3,
                "hess[{},{}] analytic={} numerical={} diff={}", i, j, ah[idx], nh[idx], d);
        }
    }
}

#[test]
fn test_root_params_cost_nonzero() {
    let (mut model, params) = build_model();
    let cost = model.calc_cost(&params);
    assert!(cost > 0.0, "cost should be nonzero: {}", cost);
    assert!(cost.is_finite(), "cost should be finite: {}", cost);
}
