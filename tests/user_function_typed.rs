//! Form C (typed) `#[arael::function]` test.
//!
//! A typed function takes vectors and scalars, binds `let`s and returns
//! a tuple; a constraint body destructures it, another returns it as
//! the residual rows through a second typed function. Both models must
//! produce the cost, gradient and Hessian of the residual written
//! inline, and the emitted fns stay callable from Rust.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmProblem, RootProblem};
use arael::sym::{symbol, vect2sym, vect3sym, E};
use arael::vect::{vect2, vect3};

/// Pinhole projection of a camera-frame point `p` at focal `f` and
/// principal point `c`; zero below machine-epsilon depth.
#[arael::function]
fn project(p: vect3sym, f: E, c: vect2sym) -> (E, E) {
    let ok = p.z - epsilon_for(p.z);
    let u = p.x / p.z;
    let v = p.y / p.z;
    (branch(ok, f * u + c.x, 0.0), branch(ok, f * v + c.y, 0.0))
}

/// The pixel residual of an observation `obs` of `p`.
#[arael::function]
fn pixel_residual(p: vect3sym, f: E, c: vect2sym, obs: vect2sym) -> [E; 2] {
    let (px, py) = project(p, f, c);
    [px - obs.x, py - obs.y]
}

/// A residual under a per-observation kind: 0 as is, anything else
/// halved.
#[arael::function]
fn scaled(r: E, kind: E) -> E {
    let half = r * 0.5;
    match kind { 0 => r, _ => half }
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "pixel", {
    let (px, py) = project(dest.p, dest.f, dest.c);
    [scaled(px - dest.obs.x, dest.kind), scaled(py - dest.obs.y, dest.kind)]
}))]
struct Dest {
    p: Param<vect3<f64>>,
    f: Param<f64>,
    c: vect2<f64>,
    obs: vect2<f64>,
    kind: u32,
    hb: SelfBlock<Dest>,
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "pixel", {
    pixel_residual(ret.p, ret.f, ret.c, ret.obs)
}))]
struct Ret {
    p: Param<vect3<f64>>,
    f: Param<f64>,
    c: vect2<f64>,
    obs: vect2<f64>,
    hb: SelfBlock<Ret>,
}

#[arael::model]
#[arael(root, jacobian)]
#[arael(constraint(hb, name = "pixel", {
    let ok = plain.p.z - epsilon_for(plain.p.z);
    [branch(ok, plain.f * (plain.p.x / plain.p.z) + plain.c.x, 0.0) - plain.obs.x,
     branch(ok, plain.f * (plain.p.y / plain.p.z) + plain.c.y, 0.0) - plain.obs.y]
}))]
struct Plain {
    p: Param<vect3<f64>>,
    f: Param<f64>,
    c: vect2<f64>,
    obs: vect2<f64>,
    hb: SelfBlock<Plain>,
}

const P: [f64; 3] = [0.3, -0.2, 2.0];
const F: f64 = 500.0;
const C: [f64; 2] = [320.0, 240.0];
const OBS: [f64; 2] = [390.0, 195.0];

/// Cost, gradient and Gauss-Newton Hessian of a model at its stored
/// parameters.
fn evaluate<M: RootProblem<f64> + LmProblem<f64>>(
    m: &mut M,
) -> (Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    let mut params = Vec::new();
    m.serialize(&mut params);
    let cost = m.calc_cost(&params);
    let n = params.len();
    let mut g = vec![0.0; n];
    let mut h = vec![0.0; n * n];
    m.calc_grad_hessian_dense(&params, &mut g, &mut h);
    (params, cost, g, h)
}

fn assert_same(label: &str, a: &[f64], b: &[f64]) {
    assert_eq!(a.len(), b.len(), "{}: lengths differ", label);
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= 1e-12 * x.abs().max(1.0),
            "{}[{}]: {} != {}", label, i, x, y);
    }
}

#[test]
fn typed_fn_matches_inline_residual() {
    let mut dest = Dest {
        p: Param::new(vect3::new(P[0], P[1], P[2])),
        f: Param::new(F),
        c: vect2::new(C[0], C[1]),
        obs: vect2::new(OBS[0], OBS[1]),
        kind: 0,
        hb: SelfBlock::new(),
    };
    let mut ret = Ret {
        p: Param::new(vect3::new(P[0], P[1], P[2])),
        f: Param::new(F),
        c: vect2::new(C[0], C[1]),
        obs: vect2::new(OBS[0], OBS[1]),
        hb: SelfBlock::new(),
    };
    let mut plain = Plain {
        p: Param::new(vect3::new(P[0], P[1], P[2])),
        f: Param::new(F),
        c: vect2::new(C[0], C[1]),
        obs: vect2::new(OBS[0], OBS[1]),
        hb: SelfBlock::new(),
    };
    let (params, cost_plain, g_plain, h_plain) = evaluate(&mut plain);
    // r = (500 * 0.15 + 320 - 390, 500 * -0.1 + 240 - 195) = (5, -5).
    assert!((cost_plain - 50.0).abs() < 1e-9, "cost {} != 50", cost_plain);

    let (_, cost_dest, g_dest, h_dest) = evaluate(&mut dest);
    assert_same("cost", &[cost_dest], &[cost_plain]);
    assert_same("gradient", &g_dest, &g_plain);
    assert_same("hessian", &h_dest, &h_plain);

    let (_, cost_ret, g_ret, h_ret) = evaluate(&mut ret);
    assert_same("cost", &[cost_ret], &[cost_plain]);
    assert_same("gradient", &g_ret, &g_plain);
    assert_same("hessian", &h_ret, &h_plain);

    // The gradient against central differences on the inline model.
    let eps = 1e-6;
    for i in 0..params.len() {
        let mut pp = params.clone();
        pp[i] += eps;
        let cp = plain.calc_cost(&pp);
        pp[i] -= 2.0 * eps;
        let cm = plain.calc_cost(&pp);
        let ng = (cp - cm) / (2.0 * eps);
        assert!((g_plain[i] - ng).abs() < 1e-3 * ng.abs().max(1.0),
            "grad[{}]: analytic {} numerical {}", i, g_plain[i], ng);
    }
}

#[test]
fn typed_fn_kind_picks_the_arm() {
    let mut halved = Dest {
        p: Param::new(vect3::new(P[0], P[1], P[2])),
        f: Param::new(F),
        c: vect2::new(C[0], C[1]),
        obs: vect2::new(OBS[0], OBS[1]),
        kind: 1,
        hb: SelfBlock::new(),
    };
    let (_, cost, _, _) = evaluate(&mut halved);
    assert!((cost - 12.5).abs() < 1e-9, "halved cost {} != 12.5", cost);
}

#[test]
fn typed_fn_callable_from_rust() {
    // The emitted fns build expression trees over the sym types.
    let (px, py) = project(vect3sym::new("p"), symbol("f"), vect2sym::new("c"));
    let sx = format!("{}", px);
    let sy = format!("{}", py);
    assert!(sx.contains("p.x") && sx.contains("f") && sx.contains("c.x"), "px printed as {:?}", sx);
    assert!(sy.contains("p.y") && sy.contains("f") && sy.contains("c.y"), "py printed as {:?}", sy);

    let [rx, ry] = pixel_residual(
        vect3sym::new("p"), symbol("f"), vect2sym::new("c"), vect2sym::new("obs"));
    assert!(format!("{}", rx).contains("obs.x"));
    assert!(format!("{}", ry).contains("obs.y"));

    let s = format!("{}", scaled(symbol("r"), symbol("kind")));
    assert!(s.contains("r") && s.contains("kind"), "scaled printed as {:?}", s);
}
