// Root-owned TripletBlock + self-primary constraint.
//
// A constraint declared ON the entity struct (self-primary) can couple
// self-params with root-params via:
//   #[arael(constraint([<local_self_block>, root.<triplet>], { body }))]
//
// where `root.<triplet>` names a `TripletBlock<T>` field on the root.
// Diagonal J^T J writes land on each entity's `SelfBlock<Self>`; the
// (self, root) cross pair goes into the root's TripletBlock (COO).
//
// This test mirrors `tests/root_params_constraint.rs` in shape but
// declares the constraint on the entity rather than on a dedicated
// constraint struct, and routes cross pairs through a root-owned
// TripletBlock instead of a local CrossBlock<X, Root>. Linear
// residuals make J^T J equal the true Hessian, so analytic grad +
// Hessian must match numerical derivatives exactly (up to f.d. noise).

#[allow(unused_imports)]
use arael::simple_lm::RootProblem;
use arael::model::{Param, SelfBlock, TripletBlock};
use arael::simple_lm::LmProblem;

#[arael::model]
#[arael(constraint([hb, root.hbt], {
    [(item.a + testmodel.offset) * testmodel.isigma,
     (2.0 * item.a - 3.0 * testmodel.offset) * testmodel.isigma]
}))]
struct Item {
    a: Param<f64>,
    hb: SelfBlock<Item>,
}

#[arael::model]
#[arael(root)]
struct TestModel {
    items: arael::refs::Vec<Item>,
    offset: Param<f64>,
    isigma: f64,
    hb: SelfBlock<TestModel>,
    hbt: TripletBlock<f64>,
}

fn build_model() -> (TestModel, Vec<f64>) {
    let mut items = arael::refs::Vec::new();
    items.push(Item { a: Param::new(1.3), hb: SelfBlock::new() });
    items.push(Item { a: Param::new(-0.4), hb: SelfBlock::new() });
    items.push(Item { a: Param::new(0.7), hb: SelfBlock::new() });
    let mut model = TestModel {
        items,
        offset: Param::new(0.3),
        isigma: 2.0,
        hb: SelfBlock::new(),
        hbt: TripletBlock::new(),
    };
    let mut params = Vec::new();
    model.serialize(&mut params);
    (model, params)
}

#[test]
fn root_triplet_self_cost_nonzero() {
    let (mut m, params) = build_model();
    let cost = m.calc_cost(&params);
    assert!(cost > 0.0, "cost should be > 0 at initial params, got {}", cost);
}

#[test]
fn root_triplet_self_grad_hessian_matches_numerical() {
    let (mut m, params) = build_model();
    let n = params.len();
    let mut ag = vec![0.0_f64; n];
    let mut ah = vec![0.0_f64; n * n];
    m.calc_grad_hessian_dense(&params, &mut ag, &mut ah);

    // Numerical gradient via central difference of the cost.
    let eps = 1e-5_f64;
    let mut ng = vec![0.0_f64; n];
    for i in 0..n {
        let mut pp = params.clone();
        pp[i] += eps;
        let cp = m.calc_cost(&pp);
        pp[i] -= 2.0 * eps;
        let cm = m.calc_cost(&pp);
        ng[i] = (cp - cm) / (2.0 * eps);
    }
    for i in 0..n {
        assert!((ag[i] - ng[i]).abs() < 1e-4,
            "grad[{}]: analytic={} numerical={}", i, ag[i], ng[i]);
    }

    // Numerical Hessian via second-order central difference of the cost.
    // Linear residuals => Gauss-Newton J^T J == full Hessian of sum-of-
    // squares cost, so f.d. of the cost matches the analytic block
    // Hessian exactly up to f.d. noise.
    let eps2 = 1e-3_f64;
    let c0 = m.calc_cost(&params);
    for i in 0..n {
        for j in i..n {
            let mut pp = params.clone();
            pp[i] += eps2; pp[j] += eps2;
            let cpp = m.calc_cost(&pp);
            pp[j] -= 2.0 * eps2;
            let cpm = m.calc_cost(&pp);
            pp[i] -= 2.0 * eps2;
            let cmm = m.calc_cost(&pp);
            pp[j] += 2.0 * eps2;
            let cmp = m.calc_cost(&pp);
            let _ = c0;
            let nh_ij = (cpp - cpm - cmp + cmm) / (4.0 * eps2 * eps2);
            // calc_grad_hessian_dense writes 2*J^T J into `ah`, which
            // is exactly d2 cost for linear residuals, so compare
            // directly to the numerical second-difference.
            let a_ij = ah[i * n + j];
            assert!((a_ij - nh_ij).abs() < 1e-2,
                "hess[{},{}]: analytic={} numerical(d2 cost)={}",
                i, j, a_ij, nh_ij);
        }
    }
}
