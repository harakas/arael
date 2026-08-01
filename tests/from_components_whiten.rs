// Integration test for the covariance-whitening path used by the alignment
// demo's centre prior: build a vect3 from mixed scalar components
// (vect2sym::from_components / vect3sym::from_components) and whiten it by a 3x3
// matrix's transpose, producing a 3-element residual. Checks the analytic cost
// against the runtime vector math and the gradient against finite differences.

use arael::simple_lm::RootProblem;
use arael::model::{Model, Param, SelfBlock};
use arael::simple_lm::LmProblem;
use arael::vect::{vect2d, vect3d};
use arael::matrix::matrix3d;
use arael::refs;

// r = diag(isig) * cov^T * from_components(d.x, d.y, rot)
#[arael::model]
#[arael(constraint(hb, {
    let v = vect3sym::from_components(wht.d.x, wht.d.y, wht.rot);
    let e = wht.cov.transpose() * v;
    [e.x * wht.isig.x, e.y * wht.isig.y, e.z * wht.isig.z]
}))]
struct Wht {
    d: Param<vect2d>,
    rot: Param<f64>,
    cov: matrix3d,
    isig: vect3d,
    hb: SelfBlock<Wht>,
}

#[arael::model]
#[arael(root)]
struct Root {
    whts: refs::Vec<Wht>,
}

const D: (f64, f64) = (0.3, -0.7);
const ROT: f64 = 0.15;
const ISIG: (f64, f64, f64) = (1.2, 0.8, 2.0);

// A non-symmetric 3x3 so cov^T actually differs from cov.
fn cov() -> matrix3d {
    matrix3d::from_elements(0.9, 0.2, -0.1, 0.05, 1.1, 0.3, -0.2, 0.15, 0.8)
}

fn build() -> (Root, Vec<f64>) {
    let mut root = Root { whts: refs::Vec::new() };
    root.whts.push(Wht {
        d: Param::new(vect2d::new(D.0, D.1)),
        rot: Param::new(ROT),
        cov: cov(),
        isig: vect3d::new(ISIG.0, ISIG.1, ISIG.2),
        hb: SelfBlock::new(),
    });
    let mut params = Vec::new();
    root.serialize(&mut params);
    (root, params)
}

fn expected_residuals() -> Vec<f64> {
    let v = vect3d::new(D.0, D.1, ROT); // from_components(d.x, d.y, rot)
    let e = cov().transpose() * v;
    vec![e.x * ISIG.0, e.y * ISIG.1, e.z * ISIG.2]
}

#[test]
fn cost_matches_runtime_vector_math() {
    let (mut root, params) = build();
    let expected: f64 = expected_residuals().iter().map(|r| r * r).sum();
    let cost = root.calc_cost(&params);
    assert!((cost - expected).abs() < 1e-10 * (1.0 + expected),
        "cost={} expected={}", cost, expected);
}

#[test]
fn gradient_matches_finite_differences() {
    let (mut root, params) = build();
    let n = params.len();
    assert_eq!(n, 3, "d (vect2) + rot (scalar)");

    let mut grad = vec![0.0_f64; n];
    let mut hessian = vec![0.0_f64; n * n];
    root.calc_grad_hessian_dense(&params, &mut grad, &mut hessian);

    let eps = 1e-6;
    let mut max_abs = 0.0_f64;
    for i in 0..n {
        let mut p = params.clone();
        p[i] += eps;
        let cp = root.calc_cost(&p);
        p[i] -= 2.0 * eps;
        let cm = root.calc_cost(&p);
        let fd = (cp - cm) / (2.0 * eps);
        assert!((fd - grad[i]).abs() < 1e-4 * (1.0 + fd.abs()),
            "grad[{}]: analytic={} fd={}", i, grad[i], fd);
        max_abs = max_abs.max(grad[i].abs());
    }
    assert!(max_abs > 0.1, "gradient suspiciously small: {}", max_abs);
}
