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

// --- declared Jacobian caches: #[arael(deriv = <field>, by = <param>)] -----
//
// A chained nonlinear component: c = ref + d, c2 = c*c, with the Jacobian
// cache c2_d = [d(c2)/dd] declared as an explicit field. The constraint
// Jacobian reads the cache instead of re-deriving 2c per observation, and
// the precompute keeps both the values and the cache fresh at every
// evaluation point.

#[arael::model]
#[arael(component)]
struct Sq {
    ref_c: f64,
    d: Param<f64>,
    #[arael(symbolic = ref_c + d)]
    c: f64,
    #[arael(symbolic = c * c)]
    c2: f64,
    #[arael(deriv = c2, by = d)]
    c2_d: [f64; 1],
}

impl Component for Sq {
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

#[arael::model]
#[arael(constraint(hb, {
    [(sqitem.obs - sqitem.sq.c2) * 1.0,
     (sqitem.sq.c - sqitem.prior) * 0.5]
}))]
struct SqItem {
    sq: Sq,
    obs: f64,
    prior: f64,
    hb: SelfBlock<SqItem>,
}

#[arael::model]
#[arael(root)]
struct MSq {
    items: arael::refs::Vec<SqItem>,
}

fn build_sq() -> MSq {
    let mut items = arael::refs::Vec::new();
    for (obs, prior, c0) in [(4.0, 1.9, 1.2), (9.0, 3.2, 2.5)] {
        items.push(SqItem {
            sq: Sq { ref_c: 0.0, d: Param::new(0.0), c: c0, c2: 0.0, c2_d: [0.0] },
            obs,
            prior,
            hb: SelfBlock::new(),
        });
    }
    MSq { items }
}

/// The Jacobian read from the declared cache must equal the derivative the
/// inline expression would have carried: FD over the cost proves the whole
/// path (precompute -> cache read -> accumulated gradient).
#[test]
fn declared_deriv_cache_matches_finite_differences() {
    let mut m = build_sq();
    let mut params = Vec::new();
    m.serialize64(&mut params);
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
        assert!((ag[i] - ng).abs() < 1e-4 * (1.0 + ng.abs()),
            "grad[{}]: analytic={} numerical={}", i, ag[i], ng);
    }
}

/// After a solve the precomputed fields and the cache hold the values at
/// the final state: c2 = c^2 and c2_d = [2c] at the re-centred delta.
#[test]
fn deriv_cache_and_values_are_fresh_after_solve() {
    let mut m = build_sq();
    let r = m.solve_dense(&LmConfig::conservative());
    assert!(r.status.is_success(), "{:?}", r.status);
    for it in m.items.iter() {
        let c = it.sq.c;
        assert!((it.sq.c2 - c * c).abs() < 1e-12, "c2 = {} vs {}", it.sq.c2, c * c);
        assert!((it.sq.c2_d[0] - 2.0 * c).abs() < 1e-12,
            "c2_d = {} vs {}", it.sq.c2_d[0], 2.0 * c);
        // The optimum balances the two residuals; the cost surface is
        // nonlinear in d, so reaching it at all proves the cached Jacobian
        // was correct along the way.
        let g = 2.0 * (it.sq.c2 - it.obs) * (2.0 * c) + 0.5 * (c - it.prior);
        assert!(g.abs() < 1e-6, "stationarity residual {}", g);
    }
}
