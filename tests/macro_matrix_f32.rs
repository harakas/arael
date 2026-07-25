// The f32 side of the combination matrix: SelfBlock, CrossBlock,
// TripletBlock, and the boxed block variants on an #[arael(root, f32)]
// root -- hand-computed cost, dense-vs-COO assembly agreement, FD
// gradient, validate. (TripletBlock<f32> and the boxed f32 blocks had
// no test coverage anywhere before this.)

use arael::model::{Component, Param, SelfBlock, CrossBlock, TripletBlock, BoxedSelfBlock, BoxedCrossBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{CooMatrix, LmProblem, RootProblem};

fn check_f32<P: LmProblem<f32> + RootProblem<f32>>(label: &str, m: &mut P, manual: f32) {
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let n = x.len();
    let cost = m.calc_cost(&x);
    assert!((cost - manual).abs() <= 1e-5 * (1.0 + manual.abs()),
        "{label}: cost {} != manual {}", cost, manual);
    let mut gd = vec![0.0f32; n];
    let mut hd = vec![0.0f32; n * n];
    m.calc_grad_hessian_dense(&x, &mut gd, &mut hd);
    let mut gs = vec![0.0f32; n];
    let mut coo = CooMatrix::new(n);
    m.calc_grad_hessian_sparse(&x, &mut gs, &mut coo);
    let mut hs = vec![0.0f32; n * n];
    for k in 0..coo.rows.len() {
        let (r, c) = (coo.rows[k] as usize, coo.cols[k] as usize);
        hs[r * n + c] += coo.vals[k];
        if r != c {
            hs[c * n + r] += coo.vals[k];
        }
    }
    for i in 0..n {
        assert!((gs[i] - gd[i]).abs() <= 1e-4 * (1.0 + gd[i].abs()), "{label}: grad[{i}]");
        for j in 0..n {
            assert!((hs[i * n + j] - hd[i * n + j]).abs() <= 1e-3 * (1.0 + hd[i * n + j].abs()),
                "{label}: H[{i},{j}] coo {} dense {}", hs[i * n + j], hd[i * n + j]);
        }
    }
    let d = m.check_gradients(&x);
    assert!(d.is_clean(), "{label}: gradients:\n{}", d);
    let d = m.validate();
    assert!(d.is_clean(), "{label}: validate:\n{}", d);
}

#[arael::model]
#[arael(constraint(hb, {
    [(nf.v - nf.t) * 0.3]
}))]
struct Nf {
    v: Param<f32>,
    t: f32,
    hb: SelfBlock<Nf, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(b.v - a.v - tf.d) * 1.5]
}))]
struct Tf {
    #[arael(ref = root.nodes)]
    a: Ref<Nf>,
    #[arael(ref = root.nodes)]
    b: Ref<Nf>,
    d: f32,
    hb: CrossBlock<Nf, Nf, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.v + b.v + c.v - trif.s) * 1.1]
}))]
struct Trif {
    #[arael(ref = root.nodes)]
    a: Ref<Nf>,
    #[arael(ref = root.nodes)]
    b: Ref<Nf>,
    #[arael(ref = root.nodes)]
    c: Ref<Nf>,
    s: f32,
    hb: TripletBlock<f32>,
}

#[arael::model]
#[arael(root, f32)]
struct WF {
    nodes: refs::Vec<Nf>,
    ties: std::vec::Vec<Tf>,
    tris: std::vec::Vec<Trif>,
}

#[test]
fn f32_self_cross_and_triplet_blocks() {
    let mut nodes = refs::Vec::new();
    let r0 = nodes.push(Nf { v: Param::new(0.1), t: 0.0, hb: SelfBlock::new() });
    let r1 = nodes.push(Nf { v: Param::new(1.3), t: 1.0, hb: SelfBlock::new() });
    let r2 = nodes.push(Nf { v: Param::new(2.2), t: 2.0, hb: SelfBlock::new() });
    let ties = vec![Tf { a: r0, b: r1, d: 1.0, hb: CrossBlock::new() }];
    let tris = vec![Trif { a: r0, b: r1, c: r2, s: 3.0, hb: TripletBlock::new() }];
    let mut w = WF { nodes, ties, tris };
    let nc = |v: f32, t: f32| ((v - t) * 0.3f32).powi(2);
    let manual = nc(0.1, 0.0) + nc(1.3, 1.0) + nc(2.2, 2.0)
        + ((1.3f32 - 0.1 - 1.0) * 1.5).powi(2)
        + ((0.1f32 + 1.3 + 2.2 - 3.0) * 1.1).powi(2);
    check_f32("f32 self+cross+triplet", &mut w, manual);
}

#[arael::model]
#[arael(constraint(hb, {
    [(bxf.v - bxf.t) * 0.4]
}))]
struct Bxf {
    v: Param<f32>,
    t: f32,
    hb: BoxedSelfBlock<Bxf, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(b.v - a.v - btf.d) * 1.3]
}))]
struct Btf {
    #[arael(ref = root.nodes)]
    a: Ref<Bxf>,
    #[arael(ref = root.nodes)]
    b: Ref<Bxf>,
    d: f32,
    hb: BoxedCrossBlock<Bxf, Bxf, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct WBF {
    nodes: refs::Vec<Bxf>,
    ties: std::vec::Vec<Btf>,
}

#[test]
fn f32_boxed_blocks() {
    let mut nodes = refs::Vec::new();
    let r0 = nodes.push(Bxf { v: Param::new(0.3), t: 0.0, hb: BoxedSelfBlock::new() });
    let r1 = nodes.push(Bxf { v: Param::new(1.4), t: 1.0, hb: BoxedSelfBlock::new() });
    let ties = vec![Btf { a: r0, b: r1, d: 1.0, hb: BoxedCrossBlock::new() }];
    let mut w = WBF { nodes, ties };
    let manual = (0.3f32 * 0.4).powi(2) + (0.4f32 * 0.4).powi(2)
        + ((1.4f32 - 0.3 - 1.0) * 1.3).powi(2);
    check_f32("f32 boxed", &mut w, manual);
}

// ---------------------------------------------------------------- new forms
// f32 smoke tests: the emission is shared with f64, one battery run each.

// `parent.<selfblock>` -- data-only observations under an f32 parent.
#[arael::model]
#[arael(constraint(parent.hb, {
    [opf.y - (curvepf.m * opf.x + curvepf.c)]
}))]
struct Opf {
    x: f32,
    y: f32,
}

#[arael::model]
struct CurvePf {
    m: Param<f32>,
    c: Param<f32>,
    obs: std::vec::Vec<Opf>,
    hb: SelfBlock<CurvePf, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct WPf {
    curves: std::vec::Vec<CurvePf>,
}

#[test]
fn f32_parent_selfblock() {
    let mut w = WPf { curves: vec![CurvePf {
        m: Param::new(0.3), c: Param::new(0.1),
        obs: vec![Opf { x: 0.0, y: 1.0 }, Opf { x: 1.0, y: 3.1 }],
        hb: SelfBlock::new(),
    }]};
    let r = |x: f32, y: f32| y - (0.3 * x + 0.1);
    check_f32("f32 parent.hb", &mut w, r(0.0, 1.0).powi(2) + r(1.0, 3.1).powi(2));
}

// `[hb, parent.<triplet>]` -- own-params observations coupling to the parent.
#[arael::model]
#[arael(constraint([hb, parent.hbt], {
    [otf.y - (curvetf.m * otf.x + otf.o),
     otf.o * 3.0]
}))]
struct Otf {
    x: f32,
    y: f32,
    o: Param<f32>,
    hb: SelfBlock<Otf, f32>,
}

#[arael::model]
struct CurveTf {
    m: Param<f32>,
    obs: std::vec::Vec<Otf>,
    hb: SelfBlock<CurveTf, f32>,
    hbt: TripletBlock<f32>,
}

#[arael::model]
#[arael(root, f32)]
struct WTf {
    curves: std::vec::Vec<CurveTf>,
}

#[test]
fn f32_parent_triplet() {
    let mut w = WTf { curves: vec![CurveTf {
        m: Param::new(1.8),
        obs: vec![
            Otf { x: 1.0, y: 2.0, o: Param::new(0.01), hb: SelfBlock::new() },
            Otf { x: 2.0, y: 3.7, o: Param::new(-0.02), hb: SelfBlock::new() },
        ],
        hb: SelfBlock::new(), hbt: TripletBlock::new(),
    }]};
    let r = |x: f32, y: f32, o: f32| (y - (1.8 * x + o)).powi(2) + (o * 3.0f32).powi(2);
    check_f32("f32 [hb, parent.hbt]", &mut w, r(1.0, 2.0, 0.01) + r(2.0, 3.7, -0.02));
}

// Frines in an Option, directly on the root and under an Option
// intermediate.
#[arael::model]
#[arael(root, f32)]
struct WOptF {
    nodes: refs::Vec<Nf>,
    lc: Option<Tf>,
}

#[test]
fn f32_option_frine() {
    let nc = |v: f32, t: f32| ((v - t) * 0.3f32).powi(2);
    let mut nodes = refs::Vec::new();
    let r0 = nodes.push(Nf { v: Param::new(0.1), t: 0.0, hb: SelfBlock::new() });
    let r1 = nodes.push(Nf { v: Param::new(1.3), t: 1.0, hb: SelfBlock::new() });
    let base = nc(0.1, 0.0) + nc(1.3, 1.0);
    let mut w = WOptF { nodes, lc: Some(Tf { a: r0, b: r1, d: 1.0, hb: CrossBlock::new() }) };
    check_f32("f32 Option frine Some", &mut w, base + ((1.3f32 - 0.1 - 1.0) * 1.5).powi(2));

    let mut nodes = refs::Vec::new();
    nodes.push(Nf { v: Param::new(0.1), t: 0.0, hb: SelfBlock::new() });
    nodes.push(Nf { v: Param::new(1.3), t: 1.0, hb: SelfBlock::new() });
    let mut w = WOptF { nodes, lc: None };
    check_f32("f32 Option frine None", &mut w, base);
}

#[arael::model]
struct BundleF {
    ties: std::vec::Vec<Tf>,
}

#[arael::model]
#[arael(root, f32)]
struct WEdgeF {
    nodes: refs::Vec<Nf>,
    maybe: Option<BundleF>,
}

#[test]
fn f32_frines_under_an_option_intermediate() {
    let nc = |v: f32, t: f32| ((v - t) * 0.3f32).powi(2);
    let mut nodes = refs::Vec::new();
    let r0 = nodes.push(Nf { v: Param::new(0.1), t: 0.0, hb: SelfBlock::new() });
    let r1 = nodes.push(Nf { v: Param::new(1.3), t: 1.0, hb: SelfBlock::new() });
    let base = nc(0.1, 0.0) + nc(1.3, 1.0);
    let mut w = WEdgeF { nodes,
        maybe: Some(BundleF { ties: vec![Tf { a: r0, b: r1, d: 1.0, hb: CrossBlock::new() }] }) };
    check_f32("f32 Option edge Some", &mut w, base + ((1.3f32 - 0.1 - 1.0) * 1.5).powi(2));

    let mut nodes = refs::Vec::new();
    nodes.push(Nf { v: Param::new(0.1), t: 0.0, hb: SelfBlock::new() });
    nodes.push(Nf { v: Param::new(1.3), t: 1.0, hb: SelfBlock::new() });
    let mut w = WEdgeF { nodes, maybe: None };
    check_f32("f32 Option edge None", &mut w, base);
}

// Component params inside an f32 triplet.
#[arael::model]
#[arael(component)]
struct OffF {
    ref_c: f32,
    d: Param<f32>,
    #[arael(symbolic = ref_c + d)]
    c: f32,
}

impl Component for OffF {
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
    [(ncf.off.c - ncf.t) * 0.3]
}))]
struct Ncf {
    off: OffF,
    t: f32,
    hb: SelfBlock<Ncf, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.off.c + b.off.c - tricf.s) * 1.1]
}))]
struct Tricf {
    #[arael(ref = root.cnodes)]
    a: Ref<Ncf>,
    #[arael(ref = root.cnodes)]
    b: Ref<Ncf>,
    s: f32,
    hb: TripletBlock<f32>,
}

#[arael::model]
#[arael(root, f32)]
struct WCf {
    cnodes: refs::Vec<Ncf>,
    tris: std::vec::Vec<Tricf>,
}

#[test]
fn f32_component_params_in_a_triplet() {
    let mk = |c: f32, t: f32| Ncf {
        off: OffF { ref_c: 0.0, d: Param::new(0.0), c }, t, hb: SelfBlock::new() };
    let mut cnodes = refs::Vec::new();
    let r0 = cnodes.push(mk(0.1, 0.0));
    let r1 = cnodes.push(mk(1.2, 1.0));
    let tris = vec![Tricf { a: r0, b: r1, s: 1.5, hb: TripletBlock::new() }];
    let mut w = WCf { cnodes, tris };
    let nc = |c: f32, t: f32| ((c - t) * 0.3f32).powi(2);
    let manual = nc(0.1, 0.0) + nc(1.2, 1.0) + ((0.1f32 + 1.2 - 1.5) * 1.1).powi(2);
    check_f32("f32 component triplet", &mut w, manual);
}
