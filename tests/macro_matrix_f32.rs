// The f32 side of the combination matrix: SelfBlock, CrossBlock,
// TripletBlock, and the boxed block variants on an #[arael(root, f32)]
// root -- hand-computed cost, dense-vs-COO assembly agreement, FD
// gradient, validate. (TripletBlock<f32> and the boxed f32 blocks had
// no test coverage anywhere before this.)

use arael::model::{Param, SelfBlock, CrossBlock, TripletBlock, BoxedSelfBlock, BoxedCrossBlock};
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
