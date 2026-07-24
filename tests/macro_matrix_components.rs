// Component params (`#[arael(component)]`) in the constraint forms
// that used to reject them: TripletBlock, multi-CrossBlock,
// `root.<selfblock>`, and `[hb, root.<triplet>]`. Each case runs the
// full invariant battery -- hand-computed cost, dense/COO/indexed/band
// assembly agreement, FD gradients, clean validate() -- and the
// registration shape also solves.

use arael::model::{Component, Param, SelfBlock, CrossBlock, TripletBlock};
use arael::refs::{self, Ref};
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
    assert!(n > 0, "{label}: no parameters serialized");

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

// The toy component: a re-centering offset, params fold into the
// owner's span, the symbolic embed carries d(c)/d(delta) = 1.
#[arael::model]
#[arael(component)]
struct Off {
    ref_c: f64,
    d: Param<f64>,
    #[arael(symbolic = ref_c + d)]
    c: f64,
}

impl Component for Off {
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

fn off(c: f64) -> Off {
    Off { ref_c: 0.0, d: Param::new(0.0), c }
}

// ------------------------------------------------ triplet participants

#[arael::model]
#[arael(constraint(hb, {
    [(n.off.c - n.t) * 0.3]
}))]
struct N {
    off: Off,
    t: f64,
    hb: SelfBlock<N>,
}

fn n(c: f64, t: f64) -> N {
    N { off: off(c), t, hb: SelfBlock::new() }
}

fn n_cost(c: f64, t: f64) -> f64 {
    ((c - t) * 0.3).powi(2)
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.off.c + b.off.c + cc.off.c - tri.s) * 1.1]
}))]
struct Tri {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    #[arael(ref = root.nodes)]
    cc: Ref<N>,
    s: f64,
    hb: TripletBlock<f64>,
}

#[arael::model]
#[arael(root)]
struct WTri {
    nodes: refs::Vec<N>,
    tris: std::vec::Vec<Tri>,
}

#[test]
fn component_params_in_a_triplet() {
    let mut nodes = refs::Vec::new();
    let r0 = nodes.push(n(0.1, 0.0));
    let r1 = nodes.push(n(1.2, 1.0));
    let r2 = nodes.push(n(2.3, 2.0));
    let tris = vec![Tri { a: r0, b: r1, cc: r2, s: 3.0, hb: TripletBlock::new() }];
    let mut w = WTri { nodes, tris };
    let manual = n_cost(0.1, 0.0) + n_cost(1.2, 1.0) + n_cost(2.3, 2.0)
        + ((0.1f64 + 1.2 + 2.3 - 3.0) * 1.1).powi(2);
    check_model("component triplet", &mut w, manual);
}

// -------------------------------------------- multi-cross participants

#[arael::model]
#[arael(constraint([hb_ab, hb_ac, hb_bc], {
    [(b.off.c - a.off.c - link.d1) * 1.5,
     (cc.off.c - b.off.c - link.d2) * 0.8]
}))]
struct Link {
    #[arael(ref = root.nodes)]
    a: Ref<N>,
    #[arael(ref = root.nodes)]
    b: Ref<N>,
    #[arael(ref = root.nodes)]
    cc: Ref<N>,
    d1: f64,
    d2: f64,
    #[arael(cross = (a, b))]
    hb_ab: CrossBlock<N, N>,
    #[arael(cross = (a, cc))]
    hb_ac: CrossBlock<N, N>,
    #[arael(cross = (b, cc))]
    hb_bc: CrossBlock<N, N>,
}

#[arael::model]
#[arael(root)]
struct WMc {
    nodes: refs::Vec<N>,
    links: std::vec::Vec<Link>,
}

#[test]
fn component_params_in_a_multi_cross() {
    let mut nodes = refs::Vec::new();
    let r0 = nodes.push(n(0.1, 0.0));
    let r1 = nodes.push(n(1.2, 1.0));
    let r2 = nodes.push(n(2.3, 2.0));
    let links = vec![Link { a: r0, b: r1, cc: r2, d1: 1.0, d2: 1.0,
        hb_ab: CrossBlock::new(), hb_ac: CrossBlock::new(), hb_bc: CrossBlock::new() }];
    let mut w = WMc { nodes, links };
    let manual = n_cost(0.1, 0.0) + n_cost(1.2, 1.0) + n_cost(2.3, 2.0)
        + ((1.2f64 - 0.1 - 1.0) * 1.5).powi(2)
        + ((2.3f64 - 1.2 - 1.0) * 0.8).powi(2);
    check_model("component multi-cross", &mut w, manual);
}

// ------------------------- root.<selfblock> with a component-param root
// (the registration shape: one shared parameter set, many observations)

#[arael::model]
#[arael(constraint(root.hb, {
    [(obs.m - root.shift.c) * 2.0]
}))]
struct Obs {
    m: f64,
}

#[arael::model]
#[arael(root)]
struct WReg {
    shift: Off,
    obs: std::vec::Vec<Obs>,
    hb: SelfBlock<WReg>,
}

#[test]
fn component_root_in_root_selfblock() {
    let mut w = WReg {
        shift: off(0.2),
        obs: vec![Obs { m: 1.0 }, Obs { m: 1.4 }, Obs { m: 0.9 }],
        hb: SelfBlock::new(),
    };
    let manual = ((1.0f64 - 0.2) * 2.0).powi(2)
        + ((1.4f64 - 0.2) * 2.0).powi(2)
        + ((0.9f64 - 0.2) * 2.0).powi(2);
    check_model("component root.<selfblock>", &mut w, manual);

    // And it solves: the shared offset lands on the observation mean.
    let r = w.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    let mean = (1.0 + 1.4 + 0.9) / 3.0;
    assert!((w.shift.c - mean).abs() < 1e-9, "shift {} vs mean {}", w.shift.c, mean);
}

// ---------------------- [hb, root.<triplet>] with a component-param root

#[arael::model]
#[arael(constraint([hb, root.hbt], {
    [(e.x - root.bias.c - e.t) * 1.2]
}))]
struct E {
    x: Param<f64>,
    t: f64,
    hb: SelfBlock<E>,
}

#[arael::model]
#[arael(root)]
struct WJoin {
    bias: Off,
    items: std::vec::Vec<E>,
    hb: SelfBlock<WJoin>,
    hbt: TripletBlock<f64>,
}

#[test]
fn component_root_in_root_triplet() {
    let mut w = WJoin {
        bias: off(0.3),
        items: vec![
            E { x: Param::new(1.0), t: 0.5, hb: SelfBlock::new() },
            E { x: Param::new(2.0), t: 1.8, hb: SelfBlock::new() },
        ],
        hb: SelfBlock::new(),
        hbt: TripletBlock::new(),
    };
    let manual = ((1.0f64 - 0.3 - 0.5) * 1.2).powi(2)
        + ((2.0f64 - 0.3 - 1.8) * 1.2).powi(2);
    check_model("component [hb, root.hbt]", &mut w, manual);
}
