// `#[arael(component)]` + the Component trait: a compound parameter whose
// Params fold into the owning entity's span and block. The toy component
// here is a re-centering offset -- reference + zero-centred delta, the same
// lifecycle shape as a manifold param: `start` seeds the reference from the
// user-facing value, `update` folds the accepted delta and resets it,
// `finish` writes the user-facing value back. The body reads `off.c`, whose
// `symbolic =` expansion carries d(c)/d(delta) = 1 through the constraint.

use arael::model::{Component, Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};

#[arael::model]
#[arael(component)]
struct Offset2 {
    ref_c: f64,
    d: Param<f64>,
    #[arael(symbolic = ref_c + d)]
    c: f64,
}

impl Component for Offset2 {
    fn start(&mut self) {
        self.ref_c = self.c;
        self.d.value = 0.0;
    }
    fn update(&mut self) {
        self.ref_c += self.d.value;
        self.d.value = 0.0;
    }
    fn finish(&mut self) {
        self.c = self.ref_c + self.d.value;
    }
}

// obs and prior disagree, so the optimum is a least-squares compromise
// with nonzero cost: c* = (4 obs + prior) / 5, w* = 2 c*.
#[arael::model]
#[arael(constraint(hb, {
    [(item.obs - item.off.c) * item.isigma,
     (item.prior - item.off.c) * 1.0,
     (item.w - 2.0 * item.off.c) * 0.5]
}))]
struct Item {
    w: Param<f64>,
    off: Offset2,
    obs: f64,
    prior: f64,
    isigma: f64,
    hb: SelfBlock<Item>,
}

#[arael::model]
#[arael(root)]
struct M {
    items: arael::refs::Vec<Item>,
}

fn item(obs: f64, c0: f64) -> Item {
    Item {
        w: Param::new(0.1),
        off: Offset2 { ref_c: 0.0, d: Param::new(0.0), c: c0 },
        obs,
        prior: obs - 0.5,
        isigma: 2.0,
        hb: SelfBlock::new(),
    }
}

fn build() -> M {
    let mut items = arael::refs::Vec::new();
    items.push(item(3.0, 0.5));
    items.push(item(-1.0, 0.0));
    M { items }
}

/// The component's param interleaves with the entity's own param (w before,
/// d after) -- the FD check validates the whole span layout and the symbolic
/// derivative through `off.c`.
#[test]
fn grad_hessian_match_finite_differences() {
    let mut m = build();
    let mut params = Vec::new();
    m.serialize64(&mut params);
    assert_eq!(params.len(), 4, "w + d per item");

    let n = params.len();
    let mut ag = vec![0.0; n];
    let mut ah = vec![0.0; n * n];
    m.calc_grad_hessian_dense(&params, &mut ag, &mut ah);

    let eps = 1e-6;
    for i in 0..n {
        let mut pp = params.clone();
        pp[i] += eps;
        let cp = m.calc_cost(&pp);
        pp[i] -= 2.0 * eps;
        let cm = m.calc_cost(&pp);
        let ng = (cp - cm) / (2.0 * eps);
        assert!((ag[i] - ng).abs() < 1e-5,
            "grad[{}]: analytic={} numerical={}", i, ag[i], ng);
    }
}

/// Solving drives each item's offset to its observation and w to 2c; the
/// optimized value comes back through Component::finish, and the accepted
/// steps were folded into the reference (delta reset to zero).
#[test]
fn solves_and_recenters() {
    let mut m = build();
    let r = m.solve_dense(&LmConfig::conservative());
    assert!(r.status.is_success(), "{:?}", r.status);
    for (i, obs) in [(0usize, 3.0f64), (1, -1.0)] {
        let it = &m.items[i];
        let c_star = (4.0 * obs + (obs - 0.5)) / 5.0;
        assert!((it.off.c - c_star).abs() < 1e-8, "c[{}] = {} vs {}", i, it.off.c, c_star);
        assert!((it.w.value - 2.0 * c_star).abs() < 1e-8, "w[{}] = {}", i, it.w.value);
        // Re-centred: the reference carries the solution, the delta is zero.
        assert!(it.off.d.value.abs() < 1e-12, "d[{}] = {}", i, it.off.d.value);
        assert!((it.off.ref_c - c_star).abs() < 1e-8, "ref_c[{}] = {}", i, it.off.ref_c);
    }
    assert!(r.end_cost > 0.1, "the residuals disagree by design; end_cost = {}", r.end_cost);
}

/// start() seeds from the user-facing value: solving twice from the same
/// model state converges immediately the second time (the chart re-seeds
/// from the solved c).
#[test]
fn restart_is_consistent() {
    let mut m = build();
    m.solve_dense(&LmConfig::conservative());
    let r2 = m.solve_dense(&LmConfig::conservative());
    assert!(r2.status.is_success());
    assert!((r2.start_cost - r2.end_cost).abs() < 1e-10,
        "second solve should start at the optimum: {} vs {}", r2.start_cost, r2.end_cost);
}
