// Parameter covariance recovery. A point pinned by one 2-vector residual
// scaled by `isig` has Gauss-Newton Hessian H = 2 isig^2 I, so the covariance
// Sigma = 2 H^-1 = (1/isig^2) I -- an analytic value to check against.

use arael::covariance::{CovError, Covariance};
use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};

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

// --- Coupled model: two 1-DOF nodes, each with a prior toward 0 (isig p) and
// tied to the other by a difference residual (isig t). This makes H_01 nonzero,
// so the sparse factor has off-diagonal fill and the solve must handle the
// fill-reducing permutation -- unlike the block-diagonal cases above.
//
//   J^T J = [[p^2+t^2, -t^2], [-t^2, p^2+t^2]],  H = 2 J^T J,  Sigma = (J^T J)^-1.
// With p = t = 1:  Sigma = (1/3)[[2, 1], [1, 2]].
//   marginal(node)   = 2/3
//   cross(n0, n1)    = 1/3
//   conditional(node) = 2 (H_ee)^-1 = 2 / (2 (p^2+t^2)) = 1/2  (node1 held fixed)

#[arael::model]
#[arael(constraint(hb, { [node.v * node.prior_isig] }))]
struct Node {
    v: Param<f64>,
    prior_isig: f64,
    hb: SelfBlock<Node>,
}

#[arael::model]
#[arael(constraint(hb, { [(a.v - b.v) * tie.isig] }))]
struct Tie {
    #[arael(ref = root.nodes)]
    a: Ref<Node>,
    #[arael(ref = root.nodes)]
    b: Ref<Node>,
    isig: f64,
    hb: CrossBlock<Node, Node>,
}

#[arael::model]
#[arael(root)]
struct Chain {
    nodes: refs::Vec<Node>,
    ties: std::vec::Vec<Tie>,
}

#[test]
fn coupled_marginals_match_analytic() {
    let mut c = Chain { nodes: refs::Vec::new(), ties: std::vec::Vec::new() };
    c.nodes.push(Node { v: Param::new(0.0), prior_isig: 1.0, hb: SelfBlock::new() });
    c.nodes.push(Node { v: Param::new(0.0), prior_isig: 1.0, hb: SelfBlock::new() });
    c.ties.push(Tie { a: Ref::new(0), b: Ref::new(1), isig: 1.0, hb: CrossBlock::new() });

    let cov = c.assemble_covariance().unwrap();

    // Marginal of each node folds in the coupling to the other: 2/3.
    let m0 = cov.marginal_cov(&c.nodes[0]);
    let m1 = cov.marginal_cov(&c.nodes[1]);
    assert!((m0[(0, 0)] - 2.0 / 3.0).abs() < 1e-9, "m0 = {}", m0[(0, 0)]);
    assert!((m1[(0, 0)] - 2.0 / 3.0).abs() < 1e-9, "m1 = {}", m1[(0, 0)]);

    // The nodes are correlated through the tie: cross = 1/3 (positive).
    let x = cov.cross_cov(&c.nodes[0], &c.nodes[1]);
    assert!((x[(0, 0)] - 1.0 / 3.0).abs() < 1e-9, "cross = {}", x[(0, 0)]);

    // Joint 2x2 over the whole collection: [[2/3, 1/3], [1/3, 2/3]].
    let j = cov.marginal_cov(&c.nodes);
    assert_eq!((j.nrows(), j.ncols()), (2, 2));
    assert!((j[(0, 0)] - 2.0 / 3.0).abs() < 1e-9 && (j[(0, 1)] - 1.0 / 3.0).abs() < 1e-9, "joint = {}", j);

    // Conditional (node1 held fixed) is smaller than the marginal: 1/2 < 2/3.
    let cc = cov.conditional_cov(&c.nodes[0]);
    assert!((cc[(0, 0)] - 0.5).abs() < 1e-9, "conditional = {}", cc[(0, 0)]);
}

#[test]
fn empty_model_is_an_error() {
    let mut w = W { pts: refs::Vec::new() };
    assert_eq!(w.assemble_covariance().err(), Some(CovError::Empty));
}
