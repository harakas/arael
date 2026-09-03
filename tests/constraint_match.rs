// `match` in constraint bodies: runtime kernel selection in a `loss =`
// clause, arm selection in a residual body, and the same in a fit.
//
// Checks:
//   - a constraint whose loss is `match kind { .. }` gives the same cost,
//     gradient and Hessian as the constraint using that kernel directly,
//     for every kind, inlier and outlier;
//   - a `match` in a residual body follows the taken arm's value and
//     slope, and a kind no arm covers panics;
//   - a fit with a `match` loss agrees with the direct-kernel fits.

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{LmProblem, RootProblem};

// --- Dispatching constraint: one body, four kernels by `kind` ---

#[arael::model]
#[arael(constraint(hb, loss = |s| match ptd.kind {
    0 => s,
    1 => loss_huber(s, ptd.k2),
    2 => loss_cauchy(s, ptd.k2),
    _ => loss_tukey(s, ptd.k2),
}, {
    [ptd.x - ptd.mx, ptd.y - ptd.my]
}))]
struct PtD {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    k2: f64,
    kind: u32,
    hb: SelfBlock<PtD>,
}

// --- The same residual with each kernel written directly ---

#[arael::model]
#[arael(constraint(hb, loss = |s| s, {
    [pti.x - pti.mx, pti.y - pti.my]
}))]
struct PtI {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    hb: SelfBlock<PtI>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_huber(s, pth.k2), {
    [pth.x - pth.mx, pth.y - pth.my]
}))]
struct PtH {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    k2: f64,
    hb: SelfBlock<PtH>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_cauchy(s, ptc.k2), {
    [ptc.x - ptc.mx, ptc.y - ptc.my]
}))]
struct PtC {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    k2: f64,
    hb: SelfBlock<PtC>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_tukey(s, ptt.k2), {
    [ptt.x - ptt.mx, ptt.y - ptt.my]
}))]
struct PtT {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    k2: f64,
    hb: SelfBlock<PtT>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_soft_l1(s, ptl.k2), {
    [ptl.x - ptl.mx, ptl.y - ptl.my]
}))]
struct PtL {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    k2: f64,
    hb: SelfBlock<PtL>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_geman_mcclure(s, ptg.k2), {
    [ptg.x - ptg.mx, ptg.y - ptg.my]
}))]
struct PtG {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    k2: f64,
    hb: SelfBlock<PtG>,
}

// --- The shorthand: all six kernels by `kind` ---

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_select(pts.kind, s, pts.k2), {
    [pts.x - pts.mx, pts.y - pts.my]
}))]
struct PtS {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    k2: f64,
    kind: u32,
    hb: SelfBlock<PtS>,
}

// --- `match` in a residual body ---

#[arael::model]
#[arael(constraint(hb, {
    let e = ptb.x - ptb.target;
    // kind 0: plain error; kind 1: three times steeper; no default.
    [match ptb.kind {
        0 => e,
        1 => 3.0 * e,
    }]
}))]
struct PtB {
    x: Param<f64>,
    target: f64,
    kind: u32,
    hb: SelfBlock<PtB>,
}

#[arael::model]
#[arael(root)]
struct W {
    d: refs::Vec<PtD>,
    i: refs::Vec<PtI>,
    h: refs::Vec<PtH>,
    c: refs::Vec<PtC>,
    t: refs::Vec<PtT>,
    l: refs::Vec<PtL>,
    g: refs::Vec<PtG>,
    s: refs::Vec<PtS>,
    b: refs::Vec<PtB>,
}

fn empty() -> W {
    W {
        d: refs::Vec::new(), i: refs::Vec::new(), h: refs::Vec::new(),
        c: refs::Vec::new(), t: refs::Vec::new(), l: refs::Vec::new(),
        g: refs::Vec::new(), s: refs::Vec::new(), b: refs::Vec::new(),
    }
}

/// Cost, gradient and dense Hessian at the model's current values.
fn passes<P: LmProblem<f64> + RootProblem<f64>>(p: &mut P) -> (f64, Vec<f64>, Vec<f64>) {
    let mut params = Vec::new();
    p.serialize(&mut params);
    let cost = p.calc_cost(&params);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    let cost2 = p.calc_grad_hessian_dense(&params, &mut grad, &mut hess);
    assert_eq!(cost, cost2);
    (cost, grad, hess)
}

fn close(a: f64, b: f64) -> bool {
    (a - b).abs() <= 1e-12 * (1.0 + a.abs().max(b.abs()))
}

fn assert_same(what: &str, a: (f64, Vec<f64>, Vec<f64>), b: (f64, Vec<f64>, Vec<f64>)) {
    assert!(close(a.0, b.0), "{what}: cost {} vs {}", a.0, b.0);
    assert_eq!(a.1.len(), b.1.len());
    for (i, (x, y)) in a.1.iter().zip(&b.1).enumerate() {
        assert!(close(*x, *y), "{what}: grad[{i}] {x} vs {y}");
    }
    for (i, (x, y)) in a.2.iter().zip(&b.2).enumerate() {
        assert!(close(*x, *y), "{what}: hess[{i}] {x} vs {y}");
    }
}

// Two data points: an inlier (s = 0.25 < k2) and an outlier (s = 9 > k2).
const K2: f64 = 1.0;
const CASES: [(f64, f64, f64, f64); 2] = [
    (0.5, -0.3, 0.1, 0.1),   // s = 0.16 + 0.16
    (3.0, 0.0, 0.0, 0.0),    // s = 9
];

fn dispatch(kind: u32, (x, y, mx, my): (f64, f64, f64, f64)) -> W {
    let mut w = empty();
    w.d.push(PtD { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, kind, hb: SelfBlock::new() });
    w
}

#[test]
fn match_loss_agrees_with_each_kernel_written_directly() {
    for case in CASES {
        let (x, y, mx, my) = case;
        let mut w = empty();
        w.i.push(PtI { x: Param::new(x), y: Param::new(y), mx, my, hb: SelfBlock::new() });
        assert_same("kind 0 / identity", passes(&mut dispatch(0, case)), passes(&mut w));

        let mut w = empty();
        w.h.push(PtH { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, hb: SelfBlock::new() });
        assert_same("kind 1 / huber", passes(&mut dispatch(1, case)), passes(&mut w));

        let mut w = empty();
        w.c.push(PtC { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, hb: SelfBlock::new() });
        assert_same("kind 2 / cauchy", passes(&mut dispatch(2, case)), passes(&mut w));

        // The `_` arm: any other kind is Tukey.
        for kind in [3, 7] {
            let mut w = empty();
            w.t.push(PtT { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, hb: SelfBlock::new() });
            assert_same("kind 3+ / tukey", passes(&mut dispatch(kind, case)), passes(&mut w));
        }
    }
}

fn shorthand(kind: u32, (x, y, mx, my): (f64, f64, f64, f64)) -> W {
    let mut w = empty();
    w.s.push(PtS { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, kind, hb: SelfBlock::new() });
    w
}

#[test]
fn loss_select_agrees_with_each_kernel_written_directly() {
    for case in CASES {
        let (x, y, mx, my) = case;
        let mut w = empty();
        w.i.push(PtI { x: Param::new(x), y: Param::new(y), mx, my, hb: SelfBlock::new() });
        assert_same("select 0 / identity", passes(&mut shorthand(0, case)), passes(&mut w));
        let mut w = empty();
        w.h.push(PtH { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, hb: SelfBlock::new() });
        assert_same("select 1 / huber", passes(&mut shorthand(1, case)), passes(&mut w));
        let mut w = empty();
        w.l.push(PtL { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, hb: SelfBlock::new() });
        assert_same("select 2 / soft_l1", passes(&mut shorthand(2, case)), passes(&mut w));
        let mut w = empty();
        w.c.push(PtC { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, hb: SelfBlock::new() });
        assert_same("select 3 / cauchy", passes(&mut shorthand(3, case)), passes(&mut w));
        let mut w = empty();
        w.g.push(PtG { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, hb: SelfBlock::new() });
        assert_same("select 4 / geman_mcclure", passes(&mut shorthand(4, case)), passes(&mut w));
        let mut w = empty();
        w.t.push(PtT { x: Param::new(x), y: Param::new(y), mx, my, k2: K2, hb: SelfBlock::new() });
        assert_same("select 5 / tukey", passes(&mut shorthand(5, case)), passes(&mut w));
    }
}

#[test]
#[should_panic(expected = "select index 6 out of range 0..6")]
fn loss_select_panics_on_unknown_kind() {
    let _ = passes(&mut shorthand(6, CASES[0]));
}

#[test]
fn match_in_residual_body_picks_arm_value_and_slope() {
    // e = 1. kind 0: r = e, cost 1, d cost/dx = 2. kind 1: r = 3e, cost 9, grad 18.
    let mut w = empty();
    w.b.push(PtB { x: Param::new(2.0), target: 1.0, kind: 0, hb: SelfBlock::new() });
    let (cost, grad, _) = passes(&mut w);
    assert_eq!(cost, 1.0);
    assert_eq!(grad, vec![2.0]);
    let mut w = empty();
    w.b.push(PtB { x: Param::new(2.0), target: 1.0, kind: 1, hb: SelfBlock::new() });
    let (cost, grad, _) = passes(&mut w);
    assert_eq!(cost, 9.0);
    assert_eq!(grad, vec![18.0]);
}

#[test]
#[should_panic(expected = "select index 2 out of range 0..2")]
fn match_without_default_panics_on_unknown_kind() {
    let mut w = empty();
    w.b.push(PtB { x: Param::new(2.0), target: 1.0, kind: 2, hb: SelfBlock::new() });
    let _ = passes(&mut w);
}

// --- The same in a fit ---

#[arael::model]
#[derive(Clone, Copy)]
struct XY { x: f64, y: f64 }

#[arael::model]
#[arael(fit64(data, |e| a * e.x + b - e.y, loss = |s| match kind {
    0 => s,
    _ => loss_cauchy(s, k),
}))]
struct LineM {
    a: Param<f64>,
    b: Param<f64>,
    data: Vec<XY>,
    k: f64,
    kind: u32,
}

#[arael::model]
#[arael(fit64(data, |e| a * e.x + b - e.y, loss = |s| s))]
struct LineI {
    a: Param<f64>,
    b: Param<f64>,
    data: Vec<XY>,
}

#[arael::model]
#[arael(fit64(data, |e| a * e.x + b - e.y, loss = |s| loss_cauchy(s, k)))]
struct LineC {
    a: Param<f64>,
    b: Param<f64>,
    data: Vec<XY>,
    k: f64,
}

fn fit_passes<P: LmProblem<f64>>(p: &mut P, params: &[f64]) -> (f64, Vec<f64>, Vec<f64>) {
    let cost = p.calc_cost(params);
    let n = params.len();
    let mut grad = vec![0.0; n];
    let mut hess = vec![0.0; n * n];
    let cost2 = p.calc_grad_hessian_dense(params, &mut grad, &mut hess);
    assert_eq!(cost, cost2);
    (cost, grad, hess)
}

#[test]
fn fit_match_loss_agrees_with_direct_kernels() {
    let data: Vec<XY> = (0..8).map(|i| {
        let x = i as f64 * 0.5 - 2.0;
        XY { x, y: 2.0 * x - 1.0 + if i == 5 { 20.0 } else { 0.1 * (i % 3) as f64 } }
    }).collect();
    let at = [1.5, -0.5];
    let mut m0 = LineM { a: Param::new(at[0]), b: Param::new(at[1]), data: data.clone(), k: 0.25, kind: 0 };
    let mut i = LineI { a: Param::new(at[0]), b: Param::new(at[1]), data: data.clone() };
    assert_same("fit kind 0 / identity", fit_passes(&mut m0, &at), fit_passes(&mut i, &at));
    let mut m1 = LineM { a: Param::new(at[0]), b: Param::new(at[1]), data: data.clone(), k: 0.25, kind: 1 };
    let mut c = LineC { a: Param::new(at[0]), b: Param::new(at[1]), data, k: 0.25 };
    assert_same("fit kind 1 / cauchy", fit_passes(&mut m1, &at), fit_passes(&mut c, &at));
}
