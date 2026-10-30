// A robust loss at a scale far above the residuals is plain least squares
// to rounding: rho(s) = s (1 - O(s/c2)). The Cauchy, Tukey and soft-L1
// costs used to cancel there (`ln(1 + s/c2)`, `1 - (1 - s/c2)^3`,
// `sqrt(1 + s/k2) - 1`), off by eps / (s/c2) relative and exactly 0 once
// s/c2 is below eps -- in f32 already at s/c2 = 1e-8.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmProblem, RootProblem};

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_cauchy(s, 1e12), { [pc.x - pc.t] }))]
struct Pc { x: Param<f64>, t: f64, hb: SelfBlock<Pc> }

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_tukey(s, 1e12), { [pt.x - pt.t] }))]
struct Pt { x: Param<f64>, t: f64, hb: SelfBlock<Pt> }

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_soft_l1(s, 1e12), { [ps.x - ps.t] }))]
struct Ps { x: Param<f64>, t: f64, hb: SelfBlock<Ps> }

#[arael::model]
#[arael(root)]
struct R {
    c: std::vec::Vec<Pc>,
    t: std::vec::Vec<Pt>,
    s: std::vec::Vec<Ps>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_cauchy(s, 1e12), { [pc32.x - pc32.t] }))]
struct Pc32 { x: Param<f32>, t: f32, hb: SelfBlock<Pc32, f32> }

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_tukey(s, 1e12), { [pt32.x - pt32.t] }))]
struct Pt32 { x: Param<f32>, t: f32, hb: SelfBlock<Pt32, f32> }

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_soft_l1(s, 1e12), { [ps32.x - ps32.t] }))]
struct Ps32 { x: Param<f32>, t: f32, hb: SelfBlock<Ps32, f32> }

#[arael::model]
#[arael(root, f32)]
struct R32 {
    c: std::vec::Vec<Pc32>,
    t: std::vec::Vec<Pt32>,
    s: std::vec::Vec<Ps32>,
}

/// Residual 1 under each kernel at c2 = 1e12: s = 1, x = s/c2 = 1e-12.
#[test]
fn loss_far_above_the_residuals_is_least_squares_f64() {
    let x = 1e-12f64;
    for (which, expect) in [
        (0, 1.0 - x / 2.0 + x * x / 3.0),
        (1, 1.0 - x + x * x / 3.0),
        (2, 1.0 - x / 4.0 + x * x / 8.0),
    ] {
        let mut r = R { c: vec![], t: vec![], s: vec![] };
        match which {
            0 => r.c.push(Pc { x: Param::new(0.0), t: 1.0, hb: SelfBlock::new() }),
            1 => r.t.push(Pt { x: Param::new(0.0), t: 1.0, hb: SelfBlock::new() }),
            _ => r.s.push(Ps { x: Param::new(0.0), t: 1.0, hb: SelfBlock::new() }),
        }
        let mut params = Vec::new();
        r.serialize(&mut params);
        let cost = r.calc_cost(&params);
        assert!(((cost - expect) / expect).abs() < 1e-15, "kernel {which}: {cost:e} vs {expect:e}");
    }
}

#[test]
fn loss_far_above_the_residuals_is_least_squares_f32() {
    for which in 0..3 {
        let mut r = R32 { c: vec![], t: vec![], s: vec![] };
        match which {
            0 => r.c.push(Pc32 { x: Param::new(0.0), t: 1.0, hb: SelfBlock::new() }),
            1 => r.t.push(Pt32 { x: Param::new(0.0), t: 1.0, hb: SelfBlock::new() }),
            _ => r.s.push(Ps32 { x: Param::new(0.0), t: 1.0, hb: SelfBlock::new() }),
        }
        let mut params = Vec::new();
        r.serialize(&mut params);
        let cost = r.calc_cost(&params);
        assert!((cost - 1.0).abs() < 1e-6, "kernel {which}: {cost:e}");
    }
}
