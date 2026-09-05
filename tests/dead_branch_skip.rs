//! A constraint whose rows all sit under one `branch` with zero on the
//! other side writes nothing on that side. The values on both sides must
//! match the closed form through every assembly route, with and without a
//! loss, and a branch whose other side is not zero, or rows under two
//! different conditions, must keep their writes.

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{CooMatrix, LmProblem, RootProblem};

const TOL: f64 = 1e-9;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

/// Cost + all-route + FD + validate battery; returns the dense (g, H).
fn check_model<P>(label: &str, m: &mut P, manual_cost: f64) -> (Vec<f64>, Vec<f64>)
where
    P: LmProblem<f64> + RootProblem<f64>,
{
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let n = x.len();
    let cost = m.calc_cost(&x);
    assert!(close(cost, manual_cost, TOL),
        "{label}: calc_cost {} != manual {}", cost, manual_cost);

    let mut gd = vec![0.0; n];
    let mut hd = vec![0.0; n * n];
    let cd = m.calc_grad_hessian_dense(&x, &mut gd, &mut hd);
    assert!(close(cd, cost, TOL), "{label}: dense cost");

    let mut gs = vec![0.0; n];
    let mut coo = CooMatrix::new(n);
    let cs = m.calc_grad_hessian_sparse(&x, &mut gs, &mut coo);
    assert!(close(cs, cost, TOL), "{label}: coo cost");
    let mut hs = vec![0.0; n * n];
    for k in 0..coo.rows.len() {
        let (r, c) = (coo.rows[k] as usize, coo.cols[k] as usize);
        hs[r * n + c] += coo.vals[k];
        if r != c { hs[c * n + r] += coo.vals[k]; }
    }
    for i in 0..n {
        assert!(close(gs[i], gd[i], TOL), "{label}: coo grad[{i}]");
        for j in 0..n {
            assert!(close(hs[i * n + j], hd[i * n + j], TOL),
                "{label}: coo H[{i},{j}] {} != dense {}", hs[i * n + j], hd[i * n + j]);
        }
    }

    let (csc, positions) = coo.to_csc_with_map().unwrap();
    let mut gi = vec![0.0; n];
    let mut vals = vec![0.0; csc.vals.len()];
    let ci = m.calc_grad_hessian_sparse_indexed(&x, &mut gi, &mut vals, &positions);
    assert!(close(ci, cost, TOL), "{label}: indexed cost");
    for i in 0..n {
        assert!(close(gi[i], gd[i], TOL), "{label}: indexed grad[{i}]");
    }

    let kd = n - 1;
    let ldab = kd + 1;
    let mut gb = vec![0.0; n];
    let mut band = vec![0.0; ldab * n];
    let cb = m.calc_grad_hessian_band(&x, &mut gb, &mut band, kd)
        .unwrap_or_else(|e| panic!("{label}: band overflow: {e}"));
    assert!(close(cb, cost, TOL), "{label}: band cost");
    for i in 0..n {
        for j in i..n {
            assert!(close(band[(kd + i - j) + j * ldab], hd[i * n + j], TOL),
                "{label}: band H[{i},{j}]");
        }
    }

    let d = m.check_gradients(&x);
    assert!(d.is_clean(), "{label}: gradient check:\n{}", d);
    (gd, hd)
}

/// One-sided spring `k * max(c - v, 0)` as a branch with zero on the
/// other side: the skip applies.
#[arael::model]
#[arael(constraint(hb, name = "spring", {
    [spring.k * branch(spring.c - spring.v, spring.c - spring.v, 0.0)]
}))]
struct Spring {
    v: Param<f64>,
    c: f64,
    k: f64,
    hb: SelfBlock<Spring>,
}

/// The same spring under a Cauchy loss, `rho(0) = 0`: the skip applies.
#[arael::model]
#[arael(constraint(hb, name = "robust", loss = |s| loss_cauchy(s, robustspring.k2), {
    [robustspring.k * branch(robustspring.c - robustspring.v,
        robustspring.c - robustspring.v, 0.0)]
}))]
struct RobustSpring {
    v: Param<f64>,
    c: f64,
    k: f64,
    k2: f64,
    hb: SelfBlock<RobustSpring>,
}

/// A branch whose other side is one, not zero: no skip, the row is 1
/// there.
#[arael::model]
#[arael(constraint(hb, name = "step", {
    [branch(step.c - step.v, step.c - step.v, 1.0)]
}))]
struct Step {
    v: Param<f64>,
    c: f64,
    hb: SelfBlock<Step>,
}

/// Two rows under two different conditions: no skip.
#[arael::model]
#[arael(constraint(hb, name = "pair", {
    [branch(pair.c - pair.v, pair.c - pair.v, 0.0),
     branch(pair.v - pair.d, pair.v - pair.d, 0.0)]
}))]
struct Pair {
    v: Param<f64>,
    c: f64,
    d: f64,
    hb: SelfBlock<Pair>,
}

#[arael::model]
#[arael(root)]
struct World {
    springs: refs::Vec<Spring>,
    robust: refs::Vec<RobustSpring>,
    steps: refs::Vec<Step>,
    pairs: refs::Vec<Pair>,
}

const C: f64 = 2.0;
const D: f64 = 4.0;
const K: f64 = 3.0;
const K2: f64 = 0.5;

fn world(v: f64) -> World {
    let mut w = World {
        springs: refs::Vec::new(),
        robust: refs::Vec::new(),
        steps: refs::Vec::new(),
        pairs: refs::Vec::new(),
    };
    w.springs.push(Spring { v: Param::new(v), c: C, k: K, hb: SelfBlock::new() });
    w.robust.push(RobustSpring { v: Param::new(v), c: C, k: K, k2: K2, hb: SelfBlock::new() });
    w.steps.push(Step { v: Param::new(v), c: C, hb: SelfBlock::new() });
    w.pairs.push(Pair { v: Param::new(v), c: C, d: D, hb: SelfBlock::new() });
    w
}

fn cauchy(s: f64) -> f64 {
    K2 * (1.0 + s / K2).ln()
}

#[test]
fn engaged_side_matches_closed_form() {
    let v = 1.0;
    let r = K * (C - v);
    let mut w = world(v);
    let cost = r * r + cauchy(r * r) + (C - v) * (C - v) + (C - v) * (C - v);
    let (_, h) = check_model("engaged", &mut w, cost);
    // The spring's Hessian diagonal is 2 k^2.
    assert!(close(h[0], 2.0 * K * K, TOL), "spring H = {}", h[0]);
}

#[test]
fn skipped_side_writes_nothing() {
    let v = 3.0;
    let mut w = world(v);
    // Only the step's other side is left: the constant 1.
    let (g, h) = check_model("skipped", &mut w, 1.0);
    let n = 4;
    for i in 0..n {
        assert!(g[i] == 0.0, "grad[{i}] = {}", g[i]);
        for j in 0..n {
            assert!(h[i * n + j] == 0.0, "H[{i},{j}] = {}", h[i * n + j]);
        }
    }
}

#[test]
fn second_condition_keeps_its_row() {
    let v = 5.0;
    let mut w = world(v);
    // Springs and the pair's first row are off; the pair's second row is
    // v - d = 1, and the step's other side is 1.
    let (_, h) = check_model("second", &mut w, 2.0);
    let n = 4;
    assert!(close(h[3 * n + 3], 2.0, TOL), "pair H = {}", h[3 * n + 3]);
}
