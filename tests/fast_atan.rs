// The `fast_atan` root keyword: generated code must call
// arael::utils::fast_atan / fast_atan2 instead of libm atan / atan2,
// in both the cost path and the grad/hessian path, while derivatives
// stay the exact rational forms (solves still converge).

use arael::model::{Model, Param, SelfBlock};
use arael::simple_lm::{self, LmConfig, LmProblem};
use arael::utils::{fast_atan, fast_atan2};

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [atan2(m.a, m.b) - m.t,
     atan(m.a) * 0.5]
}))]
struct M {
    a: Param<f64>,
    b: Param<f64>,
    t: f64,
    hb: SelfBlock<M>,
}

#[arael::model]
#[arael(root, fast_atan)]
#[arael(constraint(hb, {
    [atan2(mf.a, mf.b) - mf.t,
     (atan(mf.a) - mf.s) * 0.5]
}))]
struct Mf {
    a: Param<f64>,
    b: Param<f64>,
    t: f64,
    s: f64,
    hb: SelfBlock<Mf>,
}

// The two models compute their cost through different atan routes: the
// plain one matches libm exactly, the fast_atan one matches the fast
// approximations exactly. Evaluated at a point where the approximation
// error is visible, this pins which call the codegen emitted.
#[test]
fn cost_routes_through_the_selected_atan() {
    let (a, b, t) = (0.83_f64, -1.7_f64, 0.4_f64);
    let expected_std = {
        let r1 = a.atan2(b) - t;
        let r2 = a.atan() * 0.5;
        r1 * r1 + r2 * r2
    };
    let expected_fast = {
        let r1 = fast_atan2(a, b) - t;
        let r2 = fast_atan(a) * 0.5;
        r1 * r1 + r2 * r2
    };

    let mut m = M { a: Param::new(a), b: Param::new(b), t, hb: SelfBlock::new() };
    let mut params = Vec::new();
    m.serialize64(&mut params);
    assert_eq!(m.calc_cost(&params), expected_std);

    let mut mf = Mf { a: Param::new(a), b: Param::new(b), t, s: 0.0, hb: SelfBlock::new() };
    let mut params = Vec::new();
    mf.serialize64(&mut params);
    assert_eq!(mf.calc_cost(&params), expected_fast);

    // The two routes genuinely differ at this point (otherwise the two
    // asserts above would not distinguish them).
    assert!(expected_std != expected_fast);
}

// fast_atan / fast_atan2 are registered scalar functions, so a constraint
// body can call them directly -- per call site, without the root keyword.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [fast_atan2(md.a, md.b) - md.t,
     fast_atan(md.a) * 0.5]
}))]
struct Md {
    a: Param<f64>,
    b: Param<f64>,
    t: f64,
    hb: SelfBlock<Md>,
}

#[test]
fn fast_atan_callable_directly_in_constraints() {
    let (a, b, t) = (0.83_f64, -1.7_f64, 0.4_f64);
    let expected = {
        let r1 = fast_atan2(a, b) - t;
        let r2 = fast_atan(a) * 0.5;
        r1 * r1 + r2 * r2
    };
    let mut md = Md { a: Param::new(a), b: Param::new(b), t, hb: SelfBlock::new() };
    let mut params = Vec::new();
    md.serialize64(&mut params);
    assert_eq!(md.calc_cost(&params), expected);
}

// The arael-sym eval port and the arael::utils runtime implementation
// stay in lockstep: same folds, same constants, same values.
#[test]
fn sym_eval_matches_utils_runtime() {
    use std::collections::HashMap;
    for i in 0..100 {
        let x = -8.0 + i as f64 * 0.161;
        let vars: HashMap<&str, f64> = [("x", x)].into();
        let sym_val = arael::sym::fast_atan(arael::sym::symbol("x")).eval(&vars).unwrap();
        assert_eq!(sym_val, fast_atan(x), "diverged at x = {}", x);
    }
}

// The grad/hessian path works with the fast calls in the residuals: the
// solve closes both residuals under the FAST atan (the derivatives are
// the exact rational forms -- close enough for full convergence).
#[test]
fn fast_model_solves_to_the_optimum() {
    let (t, s) = (0.9_f64, 0.35_f64);
    let mut mf = Mf {
        a: Param::new(0.8),
        b: Param::new(1.1),
        t, s,
        hb: SelfBlock::new(),
    };
    let mut params = Vec::new();
    mf.serialize64(&mut params);
    let result = simple_lm::solve(&params, &mut mf,
        &LmConfig { max_iters: 100, ..Default::default() });
    mf.deserialize64(&result.x);

    assert!(result.end_cost < 1e-14, "fast model must converge, cost={}", result.end_cost);
    assert!((fast_atan(mf.a.value) - s).abs() < 1e-7,
        "fast_atan residual not closed: {}", fast_atan(mf.a.value));
    assert!((fast_atan2(mf.a.value, mf.b.value) - t).abs() < 1e-7,
        "fast_atan2 residual not closed: {}", fast_atan2(mf.a.value, mf.b.value));
    // And the solution is within the approximation error of the exact one.
    assert!((mf.a.value.atan() - s).abs() < 2e-6, "a off: {}", mf.a.value.atan());
}
