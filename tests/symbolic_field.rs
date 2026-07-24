// `#[arael(symbolic = <expr>)]`: a data field whose constraint-body reads
// expand to a derivative-carrying expression of the struct's own fields.
// The canonical use is a reparameterization: here `scale` means `exp(w)`,
// so the solve optimizes w freely while the residual sees a positive scale
// -- and the gradient/Hessian must carry d(exp(w))/dw, which the FD check
// verifies.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem};

#[arael::model]
#[arael(constraint(hb, { [(curve.target - curve.base * curve.scale) * curve.isigma] }))]
struct Curve {
    w: Param<f64>,
    base: f64,
    target: f64,
    isigma: f64,
    #[arael(symbolic = exp(w))]
    scale: f64,
    hb: SelfBlock<Curve>,
}

#[arael::model]
#[arael(root)]
struct M {
    curves: arael::refs::Vec<Curve>,
}

fn build() -> M {
    let mut curves = arael::refs::Vec::new();
    // target = base * e^w has the exact solution w = ln(target/base).
    curves.push(Curve {
        w: Param::new(0.2),
        base: 2.0,
        target: 5.0,
        isigma: 1.5,
        scale: 0.0,
        hb: SelfBlock::new(),
    });
    curves.push(Curve {
        w: Param::new(-0.3),
        base: 4.0,
        target: 1.0,
        isigma: 0.7,
        scale: 0.0,
        hb: SelfBlock::new(),
    });
    M { curves }
}

#[test]
fn gradient_carries_the_symbolic_derivative() {
    let mut m = build();
    let mut params = Vec::new();
    m.serialize64(&mut params);
    let n = params.len();

    let mut ag = vec![0.0; n];
    let mut ah = vec![0.0; n * n];
    m.calc_grad_hessian_dense(&params, &mut ag, &mut ah);

    let eps = 1e-6;
    for i in 0..n {
        let mut pp = params.clone();
        pp[i] += eps;
        let cp = m.calc_cost(&pp);
        pp[i] -= 2.0 * eps;
        let cm = m.calc_cost(&pp);
        let ng = (cp - cm) / (2.0 * eps);
        assert!((ag[i] - ng).abs() < 1e-5,
            "grad[{}]: analytic={} numerical={}", i, ag[i], ng);
        // A zero gradient here would mean the symbolic expression was read
        // as a plain constant (the pre-feature behavior).
        assert!(ag[i].abs() > 1e-6, "grad[{}] is zero -- exp(w) not differentiated", i);
    }
}

#[test]
fn solves_the_reparameterized_fit() {
    let mut m = build();
    let r = m.solve_dense(&LmConfig::conservative()).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert!((m.curves[0].w.value - (5.0f64 / 2.0).ln()).abs() < 1e-8,
        "w0 = {}", m.curves[0].w.value);
    assert!((m.curves[1].w.value - (1.0f64 / 4.0).ln()).abs() < 1e-8,
        "w1 = {}", m.curves[1].w.value);
    assert!(r.end_cost < 1e-20, "end_cost = {}", r.end_cost);
}
