// Block-level robust loss (`loss = |s| ...` on a constraint). The loss wraps
// the block's squared residual norm s = |r|^2: it contributes rho(s) to the
// cost and scales the block's gradient and Hessian by the weight w = rho'(s).
//
// Checks:
//   - `loss = |s| s` is bit-identical to no loss (w == 1).
//   - Cauchy scales grad and Hessian by w = c^2/(c^2+s), cost = rho(s).
//   - Tukey redescends: a gross outlier gets w == 0 and drops out.
//   - The same holds across a CrossBlock (cross Hessian scaled too).

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::LmProblem;

// --- Self-block models: identical residual, different loss ---

#[arael::model]
#[arael(constraint(hb, {
    [pt.x - pt.mx, pt.y - pt.my]
}))]
struct Pt {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    hb: SelfBlock<Pt>,
}

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
#[arael(constraint(hb, loss = |s| loss_cauchy(s, ptc.c2), {
    [ptc.x - ptc.mx, ptc.y - ptc.my]
}))]
struct PtC {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    c2: f64,
    hb: SelfBlock<PtC>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_tukey(s, ptt.c2), {
    [ptt.x - ptt.mx, ptt.y - ptt.my]
}))]
struct PtT {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    c2: f64,
    hb: SelfBlock<PtT>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_geman_mcclure(s, ptg.c2), {
    [ptg.x - ptg.mx, ptg.y - ptg.my]
}))]
struct PtG {
    x: Param<f64>,
    y: Param<f64>,
    mx: f64,
    my: f64,
    c2: f64,
    hb: SelfBlock<PtG>,
}

/// Mixed precision, the slam_demo shape: f32 storage, f64 solve and
/// blocks (no `f32` root keyword). The loss reads an f32 field, so the
/// block accumulator must take the rows' precision -- casts happen at
/// the weight/cost boundaries.
#[arael::model]
#[arael(constraint(hb, loss = |s| loss_geman_mcclure(s, ptm.c2), {
    [ptm.x - ptm.mx, ptm.y - ptm.my]
}))]
struct PtM {
    x: Param<f32>,
    y: Param<f32>,
    mx: f32,
    my: f32,
    c2: f32,
    hb: SelfBlock<PtM>,
}

#[arael::model]
#[arael(root)]
struct W {
    plain: refs::Vec<Pt>,
    ident: refs::Vec<PtI>,
    cauchy: refs::Vec<PtC>,
    tukey: refs::Vec<PtT>,
    gm: refs::Vec<PtG>,
    mixed: refs::Vec<PtM>,
}

// Shared residual for plain/ident/cauchy; tukey gets a gross outlier.
const X: f64 = 0.5;
const Y: f64 = -0.3;
const MX: f64 = 0.1;
const MY: f64 = 0.2;
const C2: f64 = 1.0; // squared threshold (chi2 units)
// s = (X-MX)^2 + (Y-MY)^2
fn s_inlier() -> f64 {
    (X - MX) * (X - MX) + (Y - MY) * (Y - MY)
}

fn build_self() -> (W, Vec<f64>) {
    let mut w = W {
        plain: refs::Vec::new(),
        ident: refs::Vec::new(),
        cauchy: refs::Vec::new(),
        tukey: refs::Vec::new(),
        gm: refs::Vec::new(),
        mixed: refs::Vec::new(),
    };
    w.plain.push(Pt { x: Param::new(X), y: Param::new(Y), mx: MX, my: MY, hb: SelfBlock::new() });
    w.ident.push(PtI { x: Param::new(X), y: Param::new(Y), mx: MX, my: MY, hb: SelfBlock::new() });
    w.cauchy.push(PtC { x: Param::new(X), y: Param::new(Y), mx: MX, my: MY, c2: C2, hb: SelfBlock::new() });
    // Outlier: residual (3, 0), s = 9 >> c2 = 1, so Tukey rejects it.
    w.tukey.push(PtT { x: Param::new(3.0), y: Param::new(0.0), mx: 0.0, my: 0.0, c2: C2, hb: SelfBlock::new() });
    w.gm.push(PtG { x: Param::new(X), y: Param::new(Y), mx: MX, my: MY, c2: C2, hb: SelfBlock::new() });
    w.mixed.push(PtM { x: Param::new(X as f32), y: Param::new(Y as f32),
        mx: MX as f32, my: MY as f32, c2: C2 as f32, hb: SelfBlock::new() });
    let mut params = Vec::new();
    w.serialize64(&mut params);
    (w, params)
}

// Param layout: plain(0,1), ident(2,3), cauchy(4,5), tukey(6,7), gm(8,9), mixed(10,11).
fn gh(root: &mut W, params: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = params.len();
    let mut grad = vec![0.0_f64; n];
    let mut hess = vec![0.0_f64; n * n];
    root.calc_grad_hessian_dense(params, &mut grad, &mut hess);
    (grad, hess)
}

#[test]
fn identity_loss_is_bit_identical_to_no_loss() {
    let (mut w, params) = build_self();
    assert_eq!(params.len(), 12);
    let (grad, hess) = gh(&mut w, &params);
    let n = 12;
    // ident block (2,3) must equal plain block (0,1) exactly.
    assert_eq!(grad[2], grad[0]);
    assert_eq!(grad[3], grad[1]);
    for (a, b) in [(0usize, 2usize), (1, 3)] {
        for (c, d) in [(0usize, 2usize), (1, 3)] {
            assert_eq!(hess[(b) * n + d], hess[a * n + c],
                "hessian ident[{},{}] != plain[{},{}]", b, d, a, c);
        }
    }
}

#[test]
fn cauchy_scales_gradient_and_hessian_by_weight() {
    let (mut w, params) = build_self();
    let (grad, hess) = gh(&mut w, &params);
    let n = 12;
    let s = s_inlier();
    let weight = C2 / (C2 + s); // rho'(s) for Cauchy
    // cauchy block (4,5) == weight * plain block (0,1).
    assert!((grad[4] - weight * grad[0]).abs() < 1e-12, "grad_x {} vs {}", grad[4], weight * grad[0]);
    assert!((grad[5] - weight * grad[1]).abs() < 1e-12, "grad_y {} vs {}", grad[5], weight * grad[1]);
    assert!((hess[4 * n + 4] - weight * hess[0 * n + 0]).abs() < 1e-12);
    assert!((hess[5 * n + 5] - weight * hess[1 * n + 1]).abs() < 1e-12);
}

#[test]
fn geman_mcclure_scales_gradient_and_hessian_by_weight() {
    let (mut w, params) = build_self();
    let (grad, hess) = gh(&mut w, &params);
    let n = 12;
    let s = s_inlier();
    let c2 = C2;
    let weight = (c2 / (c2 + s)) * (c2 / (c2 + s)); // rho'(s) for GM
    // gm block (8,9) == weight * plain block (0,1).
    assert!((grad[8] - weight * grad[0]).abs() < 1e-12, "grad_x {} vs {}", grad[8], weight * grad[0]);
    assert!((grad[9] - weight * grad[1]).abs() < 1e-12, "grad_y {} vs {}", grad[9], weight * grad[1]);
    assert!((hess[8 * n + 8] - weight * hess[0 * n + 0]).abs() < 1e-12);
    assert!((hess[9 * n + 9] - weight * hess[1 * n + 1]).abs() < 1e-12);
}

#[test]
fn mixed_precision_gm_weight_applies() {
    // f32 rows and an f32 scale field under an f64 solve: the same GM
    // weight as the all-f64 block, to f32 accuracy.
    let (mut w, params) = build_self();
    let (grad, _hess) = gh(&mut w, &params);
    let s = s_inlier();
    let weight = (C2 / (C2 + s)) * (C2 / (C2 + s));
    assert!((grad[10] - weight * grad[0]).abs() < 1e-5,
        "grad_x {} vs {}", grad[10], weight * grad[0]);
    assert!((grad[11] - weight * grad[1]).abs() < 1e-5,
        "grad_y {} vs {}", grad[11], weight * grad[1]);
}

#[test]
fn tukey_rejects_a_gross_outlier() {
    let (mut w, params) = build_self();
    let (grad, hess) = gh(&mut w, &params);
    let n = 12;
    // Outlier has s = 9 > c2 = 1, weight rho'(s) = 0: it leaves grad and
    // Hessian untouched.
    assert_eq!(grad[6], 0.0);
    assert_eq!(grad[7], 0.0);
    assert_eq!(hess[6 * n + 6], 0.0);
    assert_eq!(hess[7 * n + 7], 0.0);
}

#[test]
fn total_cost_sums_the_robustified_blocks() {
    let (mut w, params) = build_self();
    let cost = w.calc_cost(&params);
    let s = s_inlier();
    let rho_cauchy = C2 * (1.0 + s / (C2)).ln();
    let rho_tukey = C2 / 3.0; // fully redescended (capped)
    let rho_gm = C2 * s / (C2 + s);
    // plain s + ident s + cauchy rho + tukey rho + gm rho + mixed gm rho
    let expected = s + s + rho_cauchy + rho_tukey + rho_gm + rho_gm;
    // The mixed block computes its rho in f32, so the sum agrees only to
    // single precision.
    assert!((cost - expected).abs() < 1e-6, "cost {} vs {}", cost, expected);
}

// --- Cross-block models: identical residual, plain vs Cauchy ---

#[arael::model]
struct Node {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(constraint(hb, {
    [b.x - a.x - lp.dx, b.y - a.y - lp.dy]
}))]
struct Lp {
    #[arael(ref = root.nodes)]
    a: Ref<Node>,
    #[arael(ref = root.nodes)]
    b: Ref<Node>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_cauchy(s, lc.c2), {
    [b.x - a.x - lc.dx, b.y - a.y - lc.dy]
}))]
struct Lc {
    #[arael(ref = root.nodes)]
    a: Ref<Node>,
    #[arael(ref = root.nodes)]
    b: Ref<Node>,
    dx: f64,
    dy: f64,
    c2: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct CrossPlain {
    nodes: refs::Vec<Node>,
    links: std::vec::Vec<Lp>,
}

#[arael::model]
#[arael(root)]
struct CrossCauchy {
    nodes: refs::Vec<Node>,
    links: std::vec::Vec<Lc>,
}

#[test]
fn cross_block_loss_scales_the_cross_hessian() {
    // Two nodes with residual (0.4, -0.5) -> s = 0.41, same as the self case.
    let n0 = (0.0_f64, 0.0_f64);
    let n1 = (0.4_f64, -0.5_f64);

    let mut plain = CrossPlain { nodes: refs::Vec::new(), links: std::vec::Vec::new() };
    plain.nodes.push(Node { x: Param::new(n0.0), y: Param::new(n0.1), hb: SelfBlock::new() });
    plain.nodes.push(Node { x: Param::new(n1.0), y: Param::new(n1.1), hb: SelfBlock::new() });
    plain.links.push(Lp { a: plain.nodes.ref_at(0), b: plain.nodes.ref_at(1), dx: 0.0, dy: 0.0, hb: CrossBlock::new() });
    let mut pp = Vec::new();
    plain.serialize64(&mut pp);

    let mut cauchy = CrossCauchy { nodes: refs::Vec::new(), links: std::vec::Vec::new() };
    cauchy.nodes.push(Node { x: Param::new(n0.0), y: Param::new(n0.1), hb: SelfBlock::new() });
    cauchy.nodes.push(Node { x: Param::new(n1.0), y: Param::new(n1.1), hb: SelfBlock::new() });
    cauchy.links.push(Lc { a: cauchy.nodes.ref_at(0), b: cauchy.nodes.ref_at(1), dx: 0.0, dy: 0.0, c2: C2, hb: CrossBlock::new() });
    let mut cp = Vec::new();
    cauchy.serialize64(&mut cp);

    let n = pp.len();
    assert_eq!(n, 4);
    let mut gp = vec![0.0; n];
    let mut hp = vec![0.0; n * n];
    plain.calc_grad_hessian_dense(&pp, &mut gp, &mut hp);
    let mut gc = vec![0.0; n];
    let mut hc = vec![0.0; n * n];
    cauchy.calc_grad_hessian_dense(&cp, &mut gc, &mut hc);

    let s = 0.4 * 0.4 + 0.5 * 0.5;
    let weight = C2 / (C2 + s);
    // Every gradient entry and every Hessian entry (including the a-b cross
    // terms) is the plain value scaled by the loss weight.
    for i in 0..n {
        assert!((gc[i] - weight * gp[i]).abs() < 1e-12, "grad[{}] {} vs {}", i, gc[i], weight * gp[i]);
        for j in 0..n {
            let (a, b) = (hc[i * n + j], weight * hp[i * n + j]);
            assert!((a - b).abs() < 1e-12, "hess[{},{}] {} vs {}", i, j, a, b);
        }
    }
    // At least one off-diagonal cross term is non-zero (the test would pass
    // vacuously if the cross Hessian were empty).
    assert!(hp[0 * n + 2].abs() > 1e-9, "expected a non-zero a-b cross term");

    let rho = C2 * (1.0 + s / (C2)).ln();
    assert!((cauchy.calc_cost(&cp) - rho).abs() < 1e-12);
}
