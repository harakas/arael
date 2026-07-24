// `parent.<selfblock>`: a data-only entity nested in a parameter-bearing
// entity writes into THAT entity's SelfBlock -- the non-root analog of
// `root.<selfblock>`, the "one shared parameter set, many observations"
// shape without a container struct or Ref indirection.

use arael::model::{Param, SelfBlock};
use arael::refs;
use arael::simple_lm::{CooMatrix, LmConfig, LmProblem, RootProblem};

const TOL: f64 = 1e-9;

fn close(a: f64, b: f64, tol: f64) -> bool {
    (a - b).abs() <= tol * (1.0 + a.abs().max(b.abs()))
}

/// Cost + all-route + FD + validate battery (same as macro_matrix.rs).
fn check_model<P>(label: &str, m: &mut P, manual_cost: f64)
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
    let d = m.validate();
    assert!(d.is_clean(), "{label}: validate:\n{}", d);
}

#[arael::model]
#[arael(constraint(parent.hb, {
    [obs.y - (curve.m * obs.x + curve.c)]
}))]
struct Obs {
    x: f64,
    y: f64,
}

#[arael::model]
struct Curve {
    m: Param<f64>,
    c: Param<f64>,
    obs: std::vec::Vec<Obs>,
    hb: SelfBlock<Curve>,
}

#[arael::model]
#[arael(root)]
struct Fit {
    curves: refs::Vec<Curve>,
}

#[test]
fn observations_write_their_parents_block() {
    let mut curves = refs::Vec::new();
    curves.push(Curve {
        m: Param::new(0.3), c: Param::new(0.1),
        obs: vec![
            Obs { x: 0.0, y: 1.0 },
            Obs { x: 1.0, y: 3.1 },
            Obs { x: 2.0, y: 4.9 },
        ],
        hb: SelfBlock::new(),
    });
    curves.push(Curve {
        m: Param::new(-0.2), c: Param::new(0.5),
        obs: vec![Obs { x: 1.0, y: -0.4 }, Obs { x: 2.0, y: -0.9 }],
        hb: SelfBlock::new(),
    });
    let mut w = Fit { curves };
    let r1 = |x: f64, y: f64| y - (0.3 * x + 0.1);
    let r2 = |x: f64, y: f64| y - (-0.2 * x + 0.5);
    let manual = r1(0.0, 1.0).powi(2) + r1(1.0, 3.1).powi(2) + r1(2.0, 4.9).powi(2)
        + r2(1.0, -0.4).powi(2) + r2(2.0, -0.9).powi(2);
    check_model("parent.hb", &mut w, manual);

    // Each curve solves to ITS OWN least-squares line: per-parent sweeps,
    // per-parent blocks.
    let r = w.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    // Closed form for curve 1: x = [0,1,2], y = [1.0, 3.1, 4.9].
    assert!((w.curves[0].m.value - 1.95).abs() < 1e-9, "m1 {}", w.curves[0].m.value);
    assert!((w.curves[0].c.value - 1.05).abs() < 1e-9, "c1 {}", w.curves[0].c.value);
    // Curve 2: x = [1,2], y = [-0.4,-0.9] -- exact line through both.
    assert!((w.curves[1].m.value - (-0.5)).abs() < 1e-9, "m2 {}", w.curves[1].m.value);
    assert!((w.curves[1].c.value - 0.1).abs() < 1e-9, "c2 {}", w.curves[1].c.value);
}

// The parent itself nested below the root: the sweep wraps the whole
// containment path.
#[arael::model]
struct Group {
    curves: refs::Vec<Curve>,
}

#[arael::model]
#[arael(root)]
struct FitDeep {
    groups: std::vec::Vec<Group>,
}

#[test]
fn parents_nested_below_the_root() {
    let mk = |m: f64, c: f64, data: &[(f64, f64)]| {
        let mut curves = refs::Vec::new();
        curves.push(Curve {
            m: Param::new(m), c: Param::new(c),
            obs: data.iter().map(|&(x, y)| Obs { x, y }).collect(),
            hb: SelfBlock::new(),
        });
        Group { curves }
    };
    let mut w = FitDeep {
        groups: vec![
            mk(0.3, 0.1, &[(0.0, 1.0), (1.0, 3.1), (2.0, 4.9)]),
            mk(-0.2, 0.5, &[(1.0, -0.4), (2.0, -0.9)]),
        ],
    };
    let r1 = |x: f64, y: f64| y - (0.3 * x + 0.1);
    let r2 = |x: f64, y: f64| y - (-0.2 * x + 0.5);
    let manual = r1(0.0, 1.0).powi(2) + r1(1.0, 3.1).powi(2) + r1(2.0, 4.9).powi(2)
        + r2(1.0, -0.4).powi(2) + r2(2.0, -0.9).powi(2);
    check_model("deep parent.hb", &mut w, manual);
}
