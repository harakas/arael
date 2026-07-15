// Parameter covariance recovery. A point pinned by one 2-vector residual
// scaled by `isig` has Gauss-Newton Hessian H = 2 isig^2 I, so the covariance
// Sigma = 2 H^-1 = (1/isig^2) I -- an analytic value to check against.

use arael::covariance::{CovError, Covariance};
use arael::model::{Param, SelfBlock};
use arael::refs;

#[arael::model]
#[arael(constraint(hb, {
    [(pt.x - pt.ax) * pt.isig, (pt.y - pt.ay) * pt.isig]
}))]
struct Pt {
    x: Param<f64>,
    y: Param<f64>,
    ax: f64,
    ay: f64,
    isig: f64,
    hb: SelfBlock<Pt>,
}

#[arael::model]
#[arael(root)]
struct W {
    pts: refs::Vec<Pt>,
}

fn world(isigs: &[f64]) -> W {
    let mut w = W { pts: refs::Vec::new() };
    for &isig in isigs {
        w.pts.push(Pt { x: Param::new(0.0), y: Param::new(0.0), ax: 1.0, ay: 2.0, isig, hb: SelfBlock::new() });
    }
    w
}

#[test]
fn marginal_cov_matches_analytic() {
    // isig = 0.5 -> Sigma = (1/0.25) I = diag(4, 4).
    let mut w = world(&[0.5]);
    let cov = w.assemble_covariance().unwrap();
    let c = cov.marginal_cov(&w.pts[0]);
    assert_eq!((c.nrows(), c.ncols()), (2, 2));
    assert!((c[(0, 0)] - 4.0).abs() < 1e-10, "c00 = {}", c[(0, 0)]);
    assert!((c[(1, 1)] - 4.0).abs() < 1e-10, "c11 = {}", c[(1, 1)]);
    assert!(c[(0, 1)].abs() < 1e-12, "off-diagonal should be 0");
    // std_dev = sqrt of the diagonal.
    let sd = cov.std_dev(&w.pts[0]);
    assert!((sd[0] - 2.0).abs() < 1e-10 && (sd[1] - 2.0).abs() < 1e-10, "sd = {:?}", sd);

    // A single isolated entity shares no factor, so H is block-diagonal and its
    // conditional covariance equals its marginal.
    let cc = cov.conditional_cov(&w.pts[0]);
    assert!((cc[(0, 0)] - 4.0).abs() < 1e-10 && (cc[(1, 1)] - 4.0).abs() < 1e-10, "cc = {}", cc);
}

#[test]
fn independent_points_have_zero_cross_covariance() {
    // isig = 1 -> each point Sigma = I; the two points share no measurement.
    let mut w = world(&[1.0, 1.0]);
    let cov = w.assemble_covariance().unwrap();

    let c0 = cov.marginal_cov(&w.pts[0]);
    assert!((c0[(0, 0)] - 1.0).abs() < 1e-10 && (c0[(1, 1)] - 1.0).abs() < 1e-10);

    let cross = cov.cross_cov(&w.pts[0], &w.pts[1]);
    assert_eq!((cross.nrows(), cross.ncols()), (2, 2));
    assert!(cross.iter().all(|&v| v.abs() < 1e-12), "cross = {}", cross);

    // Querying the whole collection is the joint over all its entities: 4x4,
    // block-diagonal here (independent points).
    let joint = cov.marginal_cov(&w.pts);
    assert_eq!((joint.nrows(), joint.ncols()), (4, 4));
    assert!((joint[(0, 0)] - 1.0).abs() < 1e-10 && (joint[(2, 2)] - 1.0).abs() < 1e-10);
    assert!(joint[(0, 2)].abs() < 1e-12, "cross block should be 0");
}

#[test]
fn empty_model_is_an_error() {
    let mut w = W { pts: refs::Vec::new() };
    assert_eq!(w.assemble_covariance().err(), Some(CovError::Empty));
}
