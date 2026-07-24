// The `root.<selfblock>` primary block: a constraint on a data-only entity
// writes the ROOT's own SelfBlock -- the "one shared parameter set, many
// observations" shape. Linear residuals make J^T J the true Hessian, so the
// analytic grad + Hessian must match finite differences exactly (up to f.d.
// noise); the fits must recover the generating line; and the equivalent
// forms (`root.` alias vs lowercased type name, `root.hb` vs the param-less
// `[hb, root.hbt]` triplet spelling) must agree on the optimum.

use arael::model::{Param, SelfBlock, TripletBlock};
use arael::simple_lm::{LmConfig, LmProblem};

// --- the primary form: root.hb, body via the `root` alias ---

#[arael::model]
#[arael(constraint(root.hb, { [e.y - root.a * e.x - root.b] }))]
struct E {
    x: f64,
    y: f64,
}

#[arael::model]
#[arael(root)]
struct Fit {
    a: Param<f64>,
    b: Param<f64>,
    hb: SelfBlock<Fit>,
    data: std::vec::Vec<E>,
}

fn points() -> Vec<(f64, f64)> {
    (0..20).map(|i| { let x = i as f64 * 0.1; (x, 2.0 * x + 1.0) }).collect()
}

fn build() -> Fit {
    Fit {
        a: Param::new(0.3),
        b: Param::new(-0.2),
        hb: SelfBlock::new(),
        data: points().into_iter().map(|(x, y)| E { x, y }).collect(),
    }
}

#[test]
fn grad_hessian_match_finite_differences() {
    let mut m = build();
    let mut params = Vec::new();
    m.serialize64(&mut params);
    let n = params.len();
    assert_eq!(n, 2, "only the root's params serialize");

    let mut ag = vec![0.0_f64; n];
    let mut ah = vec![0.0_f64; n * n];
    m.calc_grad_hessian_dense(&params, &mut ag, &mut ah);

    let eps = 1e-5_f64;
    for i in 0..n {
        let mut pp = params.clone();
        pp[i] += eps;
        let cp = m.calc_cost(&pp);
        pp[i] -= 2.0 * eps;
        let cm = m.calc_cost(&pp);
        let ng = (cp - cm) / (2.0 * eps);
        assert!((ag[i] - ng).abs() < 1e-4,
            "grad[{}]: analytic={} numerical={}", i, ag[i], ng);
    }

    // Linear residuals: 2*J^T J is exactly d2 cost.
    let eps2 = 1e-3_f64;
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
            let nh = (cpp - cpm - cmp + cmm) / (4.0 * eps2 * eps2);
            assert!((ah[i * n + j] - nh).abs() < 1e-2,
                "hess[{},{}]: analytic={} numerical={}", i, j, ah[i * n + j], nh);
        }
    }
}

#[test]
fn solves_to_the_exact_line() {
    let mut m = build();
    let r = m.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!((m.a.value - 2.0).abs() < 1e-8, "a = {}", m.a.value);
    assert!((m.b.value - 1.0).abs() < 1e-8, "b = {}", m.b.value);
    assert!(r.end_cost < 1e-20, "end_cost = {}", r.end_cost);
}

// --- the same model spelled with the lowercased root type name ---

#[arael::model]
#[arael(constraint(root.hb, { [e2.y - fit2.a * e2.x - fit2.b] }))]
struct E2 {
    x: f64,
    y: f64,
}

#[arael::model]
#[arael(root)]
struct Fit2 {
    a: Param<f64>,
    b: Param<f64>,
    hb: SelfBlock<Fit2>,
    data: std::vec::Vec<E2>,
}

/// `root.a` and `<root_lc>.a` are the same param: both spellings produce the
/// same fit from the same start.
#[test]
fn root_alias_and_type_name_agree() {
    let mut m1 = build();
    let mut m2 = Fit2 {
        a: Param::new(0.3),
        b: Param::new(-0.2),
        hb: SelfBlock::new(),
        data: points().into_iter().map(|(x, y)| E2 { x, y }).collect(),
    };
    let r1 = m1.solve_dense(&LmConfig::conservative()).unwrap();
    let r2 = m2.solve_dense(&LmConfig::conservative()).unwrap();
    assert_eq!(r1.x, r2.x, "identical models must take identical steps");
    assert_eq!(r1.iterations, r2.iterations);
}

// --- the param-less [hb, root.hbt] triplet spelling (formerly a macro
// panic: index out of bounds building the entity spans) ---

#[arael::model]
#[arael(constraint([hb, root.hbt], { [e3.y - fit3.a * e3.x - fit3.b] }))]
struct E3 {
    x: f64,
    y: f64,
    hb: SelfBlock<E3>,
}

#[arael::model]
#[arael(root)]
struct Fit3 {
    a: Param<f64>,
    b: Param<f64>,
    hb: SelfBlock<Fit3>,
    hbt: TripletBlock<f64>,
    data: std::vec::Vec<E3>,
}

/// A param-less entity in the triplet form degenerates to root-only writes
/// (no self diagonal, no cross pairs) and reaches the same optimum.
#[test]
fn paramless_triplet_form_matches_root_selfblock() {
    let mut m1 = build();
    let mut m3 = Fit3 {
        a: Param::new(0.3),
        b: Param::new(-0.2),
        hb: SelfBlock::new(),
        hbt: TripletBlock::new(),
        data: points().into_iter().map(|(x, y)| E3 { x, y, hb: SelfBlock::new() }).collect(),
    };
    let r1 = m1.solve_dense(&LmConfig::conservative()).unwrap();
    let r3 = m3.solve_dense(&LmConfig::conservative()).unwrap();
    assert_eq!(r1.x, r3.x, "same equations, same steps");
}

// --- guard + block loss on the new form ---

#[arael::model]
#[arael(constraint(root.hb, guard = e4.ok, loss = |s| loss_huber(s, 0.01), {
    [e4.y - root.a * e4.x - root.b]
}))]
struct E4 {
    x: f64,
    y: f64,
    ok: bool,
}

#[arael::model]
#[arael(root)]
struct Fit4 {
    a: Param<f64>,
    b: Param<f64>,
    hb: SelfBlock<Fit4>,
    data: std::vec::Vec<E4>,
}

#[test]
fn guard_and_loss_apply() {
    let mut data: Vec<E4> = points().into_iter()
        .map(|(x, y)| E4 { x, y, ok: true }).collect();
    data[7].y = 50.0;   // outlier: Huber caps its pull
    data[9].ok = false; // invalid: guard removes it entirely
    data[9].y = -999.0;
    let mut m = Fit4 { a: Param::new(0.0), b: Param::new(0.0), hb: SelfBlock::new(), data };
    let r = m.solve_dense(&LmConfig::conservative().with_max_iters(200)).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!((m.a.value - 2.0).abs() < 0.05, "a = {} (outlier not suppressed?)", m.a.value);
    assert!((m.b.value - 1.0).abs() < 0.05, "b = {}", m.b.value);
}

// --- f32 root ---

#[arael::model]
#[arael(constraint(root.hb, { [e5.y - root.a * e5.x - root.b] }))]
struct E5 {
    x: f32,
    y: f32,
}

#[arael::model]
#[arael(root, f32)]
struct Fit5 {
    a: Param<f32>,
    b: Param<f32>,
    hb: SelfBlock<Fit5, f32>,
    data: std::vec::Vec<E5>,
}

#[test]
fn f32_root_selfblock_solves() {
    let mut m = Fit5 {
        a: Param::new(0.0f32),
        b: Param::new(0.0f32),
        hb: SelfBlock::new(),
        data: (0..20).map(|i| {
            let x = i as f32 * 0.1;
            E5 { x, y: 2.0 * x + 1.0 }
        }).collect(),
    };
    let r = m.solve_dense(&LmConfig::<f32>::conservative()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!((m.a.value - 2.0).abs() < 1e-4, "a = {}", m.a.value);
    assert!((m.b.value - 1.0).abs() < 1e-4, "b = {}", m.b.value);
}
