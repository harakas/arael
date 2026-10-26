//! A body `match` whose arms are vectors, matrices, quaternions, tuples
//! or arrays: one select per scalar component.
//!
//! Each model is checked against the same residual written another
//! way: per scalar with scalar matches, or with each arm written
//! directly. Equal cost, gradient and Hessian for every arm, the `_`
//! default included. A typed function with a tuple-valued match and a
//! block arm is used in a body and called from Rust.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmProblem, RootProblem};
use arael::sym::{symbol, vect2sym, E};
use arael::vect::{vect2, vect3};

// --- A (vector, scalar) tuple against per-scalar matches ---

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let (d, s) = match vt.kind {
        0 => (vt.p, 1.0),
        1 => (vt.p * 2.0 + vt.q, vt.w),
        _ => (vt.q - vt.p, 0.5),
    };
    [d.x * s - vt.m.x, d.y * s - vt.m.y, d.z * s - vt.m.z]
}))]
struct Vt {
    p: Param<vect3<f64>>,
    q: Param<vect3<f64>>,
    w: Param<f64>,
    m: vect3<f64>,
    kind: u32,
    hb: SelfBlock<Vt>,
}

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let dx = match vs.kind { 0 => vs.p.x, 1 => vs.p.x * 2.0 + vs.q.x, _ => vs.q.x - vs.p.x };
    let dy = match vs.kind { 0 => vs.p.y, 1 => vs.p.y * 2.0 + vs.q.y, _ => vs.q.y - vs.p.y };
    let dz = match vs.kind { 0 => vs.p.z, 1 => vs.p.z * 2.0 + vs.q.z, _ => vs.q.z - vs.p.z };
    let s = match vs.kind { 0 => 1.0, 1 => vs.w, _ => 0.5 };
    [dx * s - vs.m.x, dy * s - vs.m.y, dz * s - vs.m.z]
}))]
struct Vs {
    p: Param<vect3<f64>>,
    q: Param<vect3<f64>>,
    w: Param<f64>,
    m: vect3<f64>,
    kind: u32,
    hb: SelfBlock<Vs>,
}

// --- The residual rows themselves as the match value ---

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    match ra.kind {
        0 => [ra.x - 1.0, ra.y],
        _ => [ra.y * ra.x, ra.x + 2.0],
    }
}))]
struct Ra {
    x: Param<f64>,
    y: Param<f64>,
    kind: u32,
    hb: SelfBlock<Ra>,
}

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    [match rs.kind { 0 => rs.x - 1.0, _ => rs.y * rs.x },
     match rs.kind { 0 => rs.y, _ => rs.x + 2.0 }]
}))]
struct Rs {
    x: Param<f64>,
    y: Param<f64>,
    kind: u32,
    hb: SelfBlock<Rs>,
}

// --- A matrix and a quaternion, against each arm written directly ---

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let (r, q) = match mq.kind {
        0 => (matrix2sym::rotation(mq.a), quaternsym::from_euler_angles(mq.ea)),
        _ => (matrix2sym::rotation(mq.a * 2.0).transpose(), quaternsym::identity()),
    };
    let v = r * mq.v;
    let u = q.rotate(mq.u);
    [v.x - 1.0, v.y, u.x - mq.u.y, u.y, u.z * mq.a]
}))]
struct Mq {
    a: Param<f64>,
    ea: Param<vect3<f64>>,
    v: vect2<f64>,
    u: vect3<f64>,
    kind: u32,
    hb: SelfBlock<Mq>,
}

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let v = matrix2sym::rotation(mq0.a) * mq0.v;
    let u = quaternsym::from_euler_angles(mq0.ea).rotate(mq0.u);
    [v.x - 1.0, v.y, u.x - mq0.u.y, u.y, u.z * mq0.a]
}))]
struct Mq0 {
    a: Param<f64>,
    ea: Param<vect3<f64>>,
    v: vect2<f64>,
    u: vect3<f64>,
    hb: SelfBlock<Mq0>,
}

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let v = matrix2sym::rotation(mq1.a * 2.0).transpose() * mq1.v;
    let u = quaternsym::identity().rotate(mq1.u);
    [v.x - 1.0, v.y, u.x - mq1.u.y, u.y, u.z * mq1.a]
}))]
struct Mq1 {
    a: Param<f64>,
    ea: Param<vect3<f64>>,
    v: vect2<f64>,
    u: vect3<f64>,
    hb: SelfBlock<Mq1>,
}

// --- A typed function with a tuple-valued match and a block arm ---

/// A point `n` through one of three distortions, and a scale: as is,
/// radial with coefficient `k`, or scaled by `k`.
#[arael::function]
fn distort(n: vect2sym, model: E, k: E) -> (vect2sym, E) {
    match model {
        0 => (n, 1.0),
        1 => {
            let r2 = n.x * n.x + n.y * n.y;
            (n * (1.0 + k * r2), 2.0)
        }
        _ => (n * k, k),
    }
}

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let n = vect2sym::from_components(tf.p.x / tf.p.z, tf.p.y / tf.p.z);
    let (d, s) = distort(n, tf.model, tf.k);
    [d.x * s - tf.obs.x, d.y * s - tf.obs.y]
}))]
struct Tf {
    p: Param<vect3<f64>>,
    k: Param<f64>,
    obs: vect2<f64>,
    model: u32,
    hb: SelfBlock<Tf>,
}

#[arael::model]
#[arael(root)]
#[arael(constraint(hb, {
    let u = ts.p.x / ts.p.z;
    let v = ts.p.y / ts.p.z;
    let r2 = u * u + v * v;
    let dx = match ts.model { 0 => u, 1 => u * (1.0 + ts.k * r2), _ => u * ts.k };
    let dy = match ts.model { 0 => v, 1 => v * (1.0 + ts.k * r2), _ => v * ts.k };
    let s = match ts.model { 0 => 1.0, 1 => 2.0, _ => ts.k };
    [dx * s - ts.obs.x, dy * s - ts.obs.y]
}))]
struct Ts {
    p: Param<vect3<f64>>,
    k: Param<f64>,
    obs: vect2<f64>,
    model: u32,
    hb: SelfBlock<Ts>,
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
    assert!(a.0 > 0.0, "{what}: a zero cost tests nothing");
    assert!(close(a.0, b.0), "{what}: cost {} vs {}", a.0, b.0);
    assert_eq!(a.1.len(), b.1.len());
    for (i, (x, y)) in a.1.iter().zip(&b.1).enumerate() {
        assert!(close(*x, *y), "{what}: grad[{i}] {x} vs {y}");
    }
    for (i, (x, y)) in a.2.iter().zip(&b.2).enumerate() {
        assert!(close(*x, *y), "{what}: hess[{i}] {x} vs {y}");
    }
}

#[test]
fn a_tuple_of_a_vector_and_a_scalar_switches_like_per_scalar_matches() {
    let (p, q, m) = (vect3::new(0.4, -0.7, 1.3), vect3::new(-0.2, 0.9, 0.5), vect3::new(0.1, 0.2, -0.3));
    for kind in [0, 1, 2, 5] {
        let mut a = Vt { p: Param::new(p), q: Param::new(q), w: Param::new(1.7), m, kind, hb: SelfBlock::new() };
        let mut b = Vs { p: Param::new(p), q: Param::new(q), w: Param::new(1.7), m, kind, hb: SelfBlock::new() };
        assert_same(&format!("kind {kind}"), passes(&mut a), passes(&mut b));
    }
}

#[test]
fn the_residual_rows_switch_as_an_array() {
    for kind in [0, 1, 3] {
        let mut a = Ra { x: Param::new(0.6), y: Param::new(-1.1), kind, hb: SelfBlock::new() };
        let mut b = Rs { x: Param::new(0.6), y: Param::new(-1.1), kind, hb: SelfBlock::new() };
        assert_same(&format!("kind {kind}"), passes(&mut a), passes(&mut b));
    }
}

#[test]
fn a_matrix_and_a_quaternion_switch_to_the_taken_arm() {
    let (a, ea, v, u) = (0.3, vect3::new(0.2, -0.1, 0.4), vect2::new(1.5, -0.5), vect3::new(0.7, 0.2, -0.9));
    let mq = |kind| Mq { a: Param::new(a), ea: Param::new(ea), v, u, kind, hb: SelfBlock::new() };
    let mut m0 = Mq0 { a: Param::new(a), ea: Param::new(ea), v, u, hb: SelfBlock::new() };
    let mut m1 = Mq1 { a: Param::new(a), ea: Param::new(ea), v, u, hb: SelfBlock::new() };
    assert_same("kind 0", passes(&mut mq(0)), passes(&mut m0));
    assert_same("kind 1", passes(&mut mq(1)), passes(&mut m1));
    assert_same("kind 4", passes(&mut mq(4)), passes(&mut m1));
}

#[test]
fn a_typed_function_switches_a_tuple_with_a_block_arm() {
    let (p, obs) = (vect3::new(0.3, -0.2, 2.0), vect2::new(0.1, 0.05));
    for model in [0, 1, 2] {
        let mut a = Tf { p: Param::new(p), k: Param::new(0.8), obs, model, hb: SelfBlock::new() };
        let mut b = Ts { p: Param::new(p), k: Param::new(0.8), obs, model, hb: SelfBlock::new() };
        assert_same(&format!("model {model}"), passes(&mut a), passes(&mut b));
    }
}

#[test]
fn a_typed_function_with_a_tuple_valued_match_is_callable_from_rust() {
    let (d, s) = distort(vect2sym::new("n"), symbol("m"), symbol("k"));
    assert_eq!(format!("{}", s), "select_or(m, 1, 2, k)");
    assert_eq!(format!("{}", d.x), "select_or(m, n.x, n.x * (k * (n.x^2 + n.y^2) + 1), k * n.x)");
}
