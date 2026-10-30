// A constant as the first argument of `atan2` and `pow`. The generated code
// used to write it as the receiver of a method call, `1.0.atan2(x)`, which
// rustc rejects (an ambiguous numeric type); it now calls the free function
// in arael::utils, which takes its precision from the other argument.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};

#[arael::model]
#[arael(constraint(hb, { [atan2(1.0, a.x) - a.t] }))]
struct A {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<A>,
}

#[arael::model]
#[arael(constraint(hb, { [atan2(pi, b.x) - b.t] }))]
struct B {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<B>,
}

#[arael::model]
#[arael(constraint(hb, { [pow(2.0, c.x) - c.t] }))]
struct C {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<C>,
}

#[arael::model]
#[arael(root)]
struct R {
    a: std::vec::Vec<A>,
    b: std::vec::Vec<B>,
    c: std::vec::Vec<C>,
}

#[arael::model]
#[arael(constraint(hb, { [atan2(1.0, a32.x) - a32.t] }))]
struct A32 {
    x: Param<f32>,
    t: f32,
    hb: SelfBlock<A32, f32>,
}

#[arael::model]
#[arael(constraint(hb, { [pow(2.0, c32.x) - c32.t] }))]
struct C32 {
    x: Param<f32>,
    t: f32,
    hb: SelfBlock<C32, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct R32 {
    a: std::vec::Vec<A32>,
    c: std::vec::Vec<C32>,
}

/// atan2(1, x) = 0.5 at x = 1/tan(0.5); atan2(pi, x) = 0.5 at
/// x = pi/tan(0.5); 2^x = 3 at x = log2(3).
#[test]
fn constant_first_argument_f64() {
    let mut r = R {
        a: vec![A { x: Param::new(1.0), t: 0.5, hb: SelfBlock::new() }],
        b: vec![B { x: Param::new(1.0), t: 0.5, hb: SelfBlock::new() }],
        c: vec![C { x: Param::new(1.0), t: 3.0, hb: SelfBlock::new() }],
    };
    let res = r.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(res.status.is_success(), "{:?}", res.status);
    let pi = std::f64::consts::PI;
    assert!((r.a[0].x.value - 1.0 / 0.5f64.tan()).abs() < 1e-6, "a: {}", r.a[0].x.value);
    assert!((r.b[0].x.value - pi / 0.5f64.tan()).abs() < 1e-6, "b: {}", r.b[0].x.value);
    assert!((r.c[0].x.value - 3.0f64.log2()).abs() < 1e-6, "c: {}", r.c[0].x.value);
}

#[test]
fn constant_first_argument_f32() {
    let mut r = R32 {
        a: vec![A32 { x: Param::new(1.0), t: 0.5, hb: SelfBlock::new() }],
        c: vec![C32 { x: Param::new(1.0), t: 3.0, hb: SelfBlock::new() }],
    };
    let res = r.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(res.status.is_success(), "{:?}", res.status);
    assert!((r.a[0].x.value - 1.0 / 0.5f32.tan()).abs() < 1e-4, "a: {}", r.a[0].x.value);
    assert!((r.c[0].x.value - 3.0f32.log2()).abs() < 1e-4, "c: {}", r.c[0].x.value);
}
