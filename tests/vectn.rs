// Constraint-body integration of `vect<T, N>` / `matrix<T, R, C>`:
// N-dof params, matrix-data residuals, narrowing of dim-2 results into
// the fixed-type ops, against a hand-expanded per-component reference.

use arael::matrix::{matrixd, matrixf};
use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{CooMatrix, LmConfig, LmProblem, RootProblem};
use arael::vect::{vectd, vectf};

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
                "{label}: coo H[{i},{j}]");
        }
    }

    let d = m.check_gradients(&x);
    assert!(d.is_clean(), "{label}: gradient check:\n{}", d);
    let d = m.validate();
    assert!(d.is_clean(), "{label}: validate:\n{}", d);
}

fn dense<P: LmProblem<f64> + RootProblem<f64>>(m: &mut P) -> (f64, Vec<f64>, Vec<f64>) {
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let n = x.len();
    let mut g = vec![0.0; n];
    let mut h = vec![0.0; n * n];
    let c = m.calc_grad_hessian_dense(&x, &mut g, &mut h);
    (c, g, h)
}

// ===========================================================================
// N-dimensional model
// ===========================================================================

// Entity: a 5-dof parameter vector with a per-component prior.
#[arael::model]
#[arael(constraint(hb, {
    let d = state.v - state.t;
    [d[0] * state.w, d[1] * state.w, d[2] * state.w,
     d[3] * state.w, d[4] * state.w]
}))]
struct State {
    v: Param<vectd<5>>,
    t: vectd<5>,
    w: f64,
    hb: SelfBlock<State>,
}

// Cross constraint: a 2x5 data matrix projects the state difference; the
// dim-2 result narrows to Vec2 and feeds the fixed-type rotation.
#[arael::model]
#[arael(constraint(hb, {
    let p = link.h * (b.v - a.v);
    let q = matrix2sym::rotation(link.phi) * p;
    [(q.x - link.z0) * link.iw, (q.y - link.z1) * link.iw]
}))]
struct Link {
    #[arael(ref = root.states)] a: Ref<State>,
    #[arael(ref = root.states)] b: Ref<State>,
    h: matrixd<2, 5>,
    phi: f64,
    z0: f64,
    z1: f64,
    iw: f64,
    hb: CrossBlock<State, State>,
}

#[arael::model]
#[arael(root)]
struct Net {
    states: refs::Arena<State>,
    links: std::vec::Vec<Link>,
}

// ===========================================================================
// Per-component reference: identical math, plain scalar params
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, {
    [(rstate.f0 - rstate.t0) * rstate.w, (rstate.f1 - rstate.t1) * rstate.w,
     (rstate.f2 - rstate.t2) * rstate.w, (rstate.f3 - rstate.t3) * rstate.w,
     (rstate.f4 - rstate.t4) * rstate.w]
}))]
struct RState {
    f0: Param<f64>, f1: Param<f64>, f2: Param<f64>, f3: Param<f64>, f4: Param<f64>,
    t0: f64, t1: f64, t2: f64, t3: f64, t4: f64,
    w: f64,
    hb: SelfBlock<RState>,
}

#[arael::model]
#[arael(constraint(hb, {
    let p0 = rlink.h00 * (b.f0 - a.f0) + rlink.h01 * (b.f1 - a.f1)
        + rlink.h02 * (b.f2 - a.f2) + rlink.h03 * (b.f3 - a.f3)
        + rlink.h04 * (b.f4 - a.f4);
    let p1 = rlink.h10 * (b.f0 - a.f0) + rlink.h11 * (b.f1 - a.f1)
        + rlink.h12 * (b.f2 - a.f2) + rlink.h13 * (b.f3 - a.f3)
        + rlink.h14 * (b.f4 - a.f4);
    let c = cos(rlink.phi);
    let s = sin(rlink.phi);
    [(c * p0 - s * p1 - rlink.z0) * rlink.iw,
     (s * p0 + c * p1 - rlink.z1) * rlink.iw]
}))]
struct RLink {
    #[arael(ref = root.states)] a: Ref<RState>,
    #[arael(ref = root.states)] b: Ref<RState>,
    h00: f64, h01: f64, h02: f64, h03: f64, h04: f64,
    h10: f64, h11: f64, h12: f64, h13: f64, h14: f64,
    phi: f64,
    z0: f64,
    z1: f64,
    iw: f64,
    hb: CrossBlock<RState, RState>,
}

#[arael::model]
#[arael(root)]
struct RNet {
    states: refs::Arena<RState>,
    links: std::vec::Vec<RLink>,
}

// ===========================================================================
// Shared data
// ===========================================================================

const STATES: [([f64; 5], [f64; 5], f64); 3] = [
    ([0.1, -0.2, 0.3, 0.4, -0.5], [0.0, 0.0, 0.5, 0.5, -0.4], 1.0),
    ([1.0, 1.1, 0.9, -0.3, 0.2], [1.2, 1.0, 1.0, -0.2, 0.0], 0.6),
    ([-0.4, 0.7, 0.0, 0.8, 0.3], [-0.5, 0.5, 0.1, 1.0, 0.4], 0.8),
];
const H: [[f64; 5]; 2] = [[1.0, 0.5, -0.3, 0.2, 0.0], [0.0, 1.0, 0.4, -0.2, 0.7]];
// (a, b, phi, z0, z1, iw)
const LINKS: [(usize, usize, f64, f64, f64, f64); 3] = [
    (0, 1, 0.3, 0.8, -0.2, 1.4),
    (1, 2, -0.5, -0.4, 0.3, 0.9),
    (0, 2, 0.0, 0.1, 0.6, 1.1),
];

fn build() -> (Net, Vec<Ref<State>>) {
    let mut net = Net { states: refs::Arena::new(), links: std::vec::Vec::new() };
    let refs: Vec<Ref<State>> = STATES.iter().map(|&(v, t, w)| {
        net.states.push(State {
            v: Param::new(vectd::new(v)), t: vectd::new(t), w, hb: SelfBlock::new() })
    }).collect();
    for &(a, b, phi, z0, z1, iw) in &LINKS {
        net.links.push(Link {
            a: refs[a], b: refs[b],
            h: matrixd::from_array(H), phi, z0, z1, iw, hb: CrossBlock::new(),
        });
    }
    (net, refs)
}

fn build_ref() -> (RNet, Vec<Ref<RState>>) {
    let mut net = RNet { states: refs::Arena::new(), links: std::vec::Vec::new() };
    let refs: Vec<Ref<RState>> = STATES.iter().map(|&(v, t, w)| {
        net.states.push(RState {
            f0: Param::new(v[0]), f1: Param::new(v[1]), f2: Param::new(v[2]),
            f3: Param::new(v[3]), f4: Param::new(v[4]),
            t0: t[0], t1: t[1], t2: t[2], t3: t[3], t4: t[4],
            w, hb: SelfBlock::new() })
    }).collect();
    for &(a, b, phi, z0, z1, iw) in &LINKS {
        net.links.push(RLink {
            a: refs[a], b: refs[b],
            h00: H[0][0], h01: H[0][1], h02: H[0][2], h03: H[0][3], h04: H[0][4],
            h10: H[1][0], h11: H[1][1], h12: H[1][2], h13: H[1][3], h14: H[1][4],
            phi, z0, z1, iw, hb: CrossBlock::new(),
        });
    }
    (net, refs)
}

fn manual_cost() -> f64 {
    let mut c = 0.0;
    for &(v, t, w) in &STATES {
        for k in 0..5 { c += ((v[k] - t[k]) * w).powi(2); }
    }
    for &(a, b, phi, z0, z1, iw) in &LINKS {
        let (va, vb) = (STATES[a].0, STATES[b].0);
        let mut p = [0.0f64; 2];
        for r in 0..2 {
            for k in 0..5 { p[r] += H[r][k] * (vb[k] - va[k]); }
        }
        let (s, co) = phi.sin_cos();
        c += ((co * p[0] - s * p[1] - z0) * iw).powi(2);
        c += ((s * p[0] + co * p[1] - z1) * iw).powi(2);
    }
    c
}

#[test]
fn vectn_battery() {
    let (mut net, _) = build();
    check_model("vectn", &mut net, manual_cost());
}

#[test]
fn vectn_matches_per_component_reference() {
    let (mut net, _) = build();
    let (mut rnet, _) = build_ref();
    let mc = manual_cost();
    check_model("reference", &mut rnet, mc);
    let (ca, ga, ha) = dense(&mut net);
    let (cb, gb, hb) = dense(&mut rnet);
    assert!(close(ca, cb, TOL));
    assert_eq!(ga.len(), gb.len());
    for i in 0..ga.len() {
        assert!(close(ga[i], gb[i], TOL), "grad[{i}] {} vs {}", ga[i], gb[i]);
    }
    for k in 0..ha.len() {
        assert!(close(ha[k], hb[k], TOL), "H[{k}] {} vs {}", ha[k], hb[k]);
    }
}

#[test]
fn vectn_solves_like_reference() {
    let (mut net, nr) = build();
    let (mut rnet, rr) = build_ref();
    let a = net.solve_sparse(&LmConfig::well_conditioned()).unwrap();
    let b = rnet.solve_sparse(&LmConfig::well_conditioned()).unwrap();
    assert!(close(a.end_cost, b.end_cost, 1e-8));
    for i in 0..STATES.len() {
        let sv = net.states.get(nr[i]).unwrap().v.value;
        let r = rnet.states.get(rr[i]).unwrap();
        let rv = [r.f0.value, r.f1.value, r.f2.value, r.f3.value, r.f4.value];
        for k in 0..5 {
            assert!(close(sv[k], rv[k], 1e-7), "state {i} comp {k}: {} vs {}", sv[k], rv[k]);
        }
    }
}

// ===========================================================================
// f32 twin
// ===========================================================================

#[arael::model]
#[arael(constraint(hb, {
    let d = fstate.v - fstate.t;
    [d[0] * fstate.w, d[1] * fstate.w, d[2] * fstate.w,
     d[3] * fstate.w, d[4] * fstate.w]
}))]
struct FState {
    v: Param<vectf<5>>,
    t: vectf<5>,
    w: f32,
    hb: SelfBlock<FState, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let p = flink.h * (b.v - a.v);
    let q = matrix2sym::rotation(flink.phi) * p;
    [(q.x - flink.z0) * flink.iw, (q.y - flink.z1) * flink.iw]
}))]
struct FLink {
    #[arael(ref = root.states)] a: Ref<FState>,
    #[arael(ref = root.states)] b: Ref<FState>,
    h: matrixf<2, 5>,
    phi: f32,
    z0: f32,
    z1: f32,
    iw: f32,
    hb: CrossBlock<FState, FState, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct FNet {
    states: refs::Arena<FState>,
    links: std::vec::Vec<FLink>,
}

#[test]
fn vectn_f32_solves() {
    let mut net = FNet { states: refs::Arena::new(), links: std::vec::Vec::new() };
    let refs: Vec<Ref<FState>> = STATES.iter().map(|&(v, t, w)| {
        net.states.push(FState {
            v: Param::new(vectd::new(v).cast()),
            t: vectd::new(t).cast(),
            w: w as f32, hb: SelfBlock::new() })
    }).collect();
    for &(a, b, phi, z0, z1, iw) in &LINKS {
        net.links.push(FLink {
            a: refs[a], b: refs[b],
            h: matrixd::from_array(H).cast(),
            phi: phi as f32, z0: z0 as f32, z1: z1 as f32, iw: iw as f32,
            hb: CrossBlock::new(),
        });
    }
    let r = net.solve_sparse(&LmConfig::well_conditioned()).unwrap();
    let (mut d64, _) = build();
    let r64 = d64.solve_sparse(&LmConfig::well_conditioned()).unwrap();
    assert!((r.end_cost as f64 - r64.end_cost).abs() < 1e-3,
        "f32 {} vs f64 {}", r.end_cost, r64.end_cost);
}
