// `ln_1p` and `exp_m1` in constraint bodies: the generated calls into
// arael::utils compile at both precisions and solve to the inverse.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};

#[arael::model]
#[arael(constraint(hb, { [ln_1p(l.x) - l.t] }))]
struct L { x: Param<f64>, t: f64, hb: SelfBlock<L> }

#[arael::model]
#[arael(constraint(hb, { [exp_m1(e.x) - e.t] }))]
struct E { x: Param<f64>, t: f64, hb: SelfBlock<E> }

#[arael::model]
#[arael(root)]
struct R { l: std::vec::Vec<L>, e: std::vec::Vec<E> }

#[arael::model]
#[arael(constraint(hb, { [ln_1p(l32.x) - l32.t] }))]
struct L32 { x: Param<f32>, t: f32, hb: SelfBlock<L32, f32> }

#[arael::model]
#[arael(constraint(hb, { [exp_m1(e32.x) - e32.t] }))]
struct E32 { x: Param<f32>, t: f32, hb: SelfBlock<E32, f32> }

#[arael::model]
#[arael(root, f32)]
struct R32 { l: std::vec::Vec<L32>, e: std::vec::Vec<E32> }

/// ln(1 + x) = t at x = exp(t) - 1; exp(x) - 1 = t at x = ln(1 + t).
#[test]
fn solves_to_the_inverse_f64() {
    let mut r = R {
        l: vec![L { x: Param::new(0.0), t: 0.3, hb: SelfBlock::new() }],
        e: vec![E { x: Param::new(0.0), t: 0.3, hb: SelfBlock::new() }],
    };
    let res = r.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(res.status.is_success(), "{:?}", res.status);
    assert!((r.l[0].x.value - 0.3f64.exp_m1()).abs() < 1e-9, "l: {}", r.l[0].x.value);
    assert!((r.e[0].x.value - 0.3f64.ln_1p()).abs() < 1e-9, "e: {}", r.e[0].x.value);
}

#[test]
fn solves_to_the_inverse_f32() {
    let mut r = R32 {
        l: vec![L32 { x: Param::new(0.0), t: 0.3, hb: SelfBlock::new() }],
        e: vec![E32 { x: Param::new(0.0), t: 0.3, hb: SelfBlock::new() }],
    };
    let res = r.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(res.status.is_success(), "{:?}", res.status);
    assert!((r.l[0].x.value - 0.3f32.exp_m1()).abs() < 1e-5, "l: {}", r.l[0].x.value);
    assert!((r.e[0].x.value - 0.3f32.ln_1p()).abs() < 1e-5, "e: {}", r.e[0].x.value);
}
