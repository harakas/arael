// Parameter covariance recovery. A point pinned by one 2-vector residual
// scaled by `isig` has Gauss-Newton Hessian H = 2 isig^2 I, so the covariance
// Sigma = 2 H^-1 = (1/isig^2) I -- an analytic value to check against.

use arael::covariance::{CovError, CovMode, CovOptions, CovOrdering, Covariance};
use arael::simple_lm::BlockSupernodalMode;
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
    let cov = w.assemble_covariance(CovMode::PerQuery).unwrap();
    let c = cov.marginal_cov(&w.pts[0]).unwrap();
    assert_eq!((c.nrows(), c.ncols()), (2, 2));
    assert!((c[(0, 0)] - 4.0).abs() < 1e-10, "c00 = {}", c[(0, 0)]);
    assert!((c[(1, 1)] - 4.0).abs() < 1e-10, "c11 = {}", c[(1, 1)]);
    assert!(c[(0, 1)].abs() < 1e-12, "off-diagonal should be 0");
    // std_dev = sqrt of the diagonal.
    let sd = cov.std_dev(&w.pts[0]).unwrap();
    assert!((sd[0] - 2.0).abs() < 1e-10 && (sd[1] - 2.0).abs() < 1e-10, "sd = {:?}", sd);

    // A single isolated entity shares no factor, so H is block-diagonal and its
    // conditional covariance equals its marginal.
    let cc = cov.conditional_cov(&w.pts[0]).unwrap();
    assert!((cc[(0, 0)] - 4.0).abs() < 1e-10 && (cc[(1, 1)] - 4.0).abs() < 1e-10, "cc = {}", cc);
}

#[test]
fn independent_points_have_zero_cross_covariance() {
    // isig = 1 -> each point Sigma = I; the two points share no measurement.
    let mut w = world(&[1.0, 1.0]);
    let cov = w.assemble_covariance(CovMode::PerQuery).unwrap();

    let c0 = cov.marginal_cov(&w.pts[0]).unwrap();
    assert!((c0[(0, 0)] - 1.0).abs() < 1e-10 && (c0[(1, 1)] - 1.0).abs() < 1e-10);

    let cross = cov.cross_cov(&w.pts[0], &w.pts[1]).unwrap();
    assert_eq!((cross.nrows(), cross.ncols()), (2, 2));
    assert!(cross.iter().all(|&v| v.abs() < 1e-12), "cross = {}", cross);

    // Querying the whole collection is the joint over all its entities: 4x4,
    // block-diagonal here (independent points).
    let joint = cov.marginal_cov(&w.pts).unwrap();
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
    c.ties.push(Tie { a: c.nodes.ref_at(0), b: c.nodes.ref_at(1), isig: 1.0, hb: CrossBlock::new() });

    let cov = c.assemble_covariance(CovMode::PerQuery).unwrap();

    // Marginal of each node folds in the coupling to the other: 2/3.
    let m0 = cov.marginal_cov(&c.nodes[0]).unwrap();
    let m1 = cov.marginal_cov(&c.nodes[1]).unwrap();
    assert!((m0[(0, 0)] - 2.0 / 3.0).abs() < 1e-9, "m0 = {}", m0[(0, 0)]);
    assert!((m1[(0, 0)] - 2.0 / 3.0).abs() < 1e-9, "m1 = {}", m1[(0, 0)]);

    // The nodes are correlated through the tie: cross = 1/3 (positive).
    let x = cov.cross_cov(&c.nodes[0], &c.nodes[1]).unwrap();
    assert!((x[(0, 0)] - 1.0 / 3.0).abs() < 1e-9, "cross = {}", x[(0, 0)]);

    // Joint 2x2 over the whole collection: [[2/3, 1/3], [1/3, 2/3]].
    let j = cov.marginal_cov(&c.nodes).unwrap();
    assert_eq!((j.nrows(), j.ncols()), (2, 2));
    assert!((j[(0, 0)] - 2.0 / 3.0).abs() < 1e-9 && (j[(0, 1)] - 1.0 / 3.0).abs() < 1e-9, "joint = {}", j);

    // Conditional (node1 held fixed) is smaller than the marginal: 1/2 < 2/3.
    let cc = cov.conditional_cov(&c.nodes[0]).unwrap();
    assert!((cc[(0, 0)] - 0.5).abs() < 1e-9, "conditional = {}", cc[(0, 0)]);
}

// A path of `n` nodes: each node has a prior toward 0, consecutive nodes are
// tied. H is banded (tridiagonal in the node blocks), so its factor carries real
// fill -- a non-trivial pattern for the selected inverse to walk.
fn path_chain(n: usize) -> Chain {
    let mut c = Chain { nodes: refs::Vec::new(), ties: std::vec::Vec::new() };
    for _ in 0..n {
        c.nodes.push(Node { v: Param::new(0.0), prior_isig: 1.0, hb: SelfBlock::new() });
    }
    for i in 0..n - 1 {
        c.ties.push(Tie { a: c.nodes.ref_at(i), b: c.nodes.ref_at(i + 1), isig: 1.0, hb: CrossBlock::new() });
    }
    c
}

#[test]
fn precompute_matches_analytic_coupled() {
    // The two-node analytic case, now via the selected inverse.
    let mut c = Chain { nodes: refs::Vec::new(), ties: std::vec::Vec::new() };
    c.nodes.push(Node { v: Param::new(0.0), prior_isig: 1.0, hb: SelfBlock::new() });
    c.nodes.push(Node { v: Param::new(0.0), prior_isig: 1.0, hb: SelfBlock::new() });
    c.ties.push(Tie { a: c.nodes.ref_at(0), b: c.nodes.ref_at(1), isig: 1.0, hb: CrossBlock::new() });

    let cov = c.assemble_covariance(CovMode::AllMarginals).unwrap();

    assert!((cov.marginal_cov(&c.nodes[0]).unwrap()[(0, 0)] - 2.0 / 3.0).abs() < 1e-9);
    // The tie couples the nodes, so the cross entry is inside the pattern.
    assert!((cov.cross_cov(&c.nodes[0], &c.nodes[1]).unwrap()[(0, 0)] - 1.0 / 3.0).abs() < 1e-9);
}

#[test]
fn precompute_selected_inverse_matches_solve() {
    let mut c = path_chain(6);

    // Reference: per-query solves.
    let solved = c.assemble_covariance(CovMode::PerQuery).unwrap();
    let ref_marg: std::vec::Vec<f64> = (0..6).map(|i| solved.marginal_cov(&c.nodes[i]).unwrap()[(0, 0)]).collect();
    let ref_adjacent = solved.cross_cov(&c.nodes[2], &c.nodes[3]).unwrap()[(0, 0)];
    let ref_distant = solved.cross_cov(&c.nodes[0], &c.nodes[5]).unwrap()[(0, 0)];

    // Selected inverse: every marginal is a cache lookup.
    let pc = c.assemble_covariance(CovMode::AllMarginals).unwrap();
    for i in 0..6 {
        let m = pc.marginal_cov(&c.nodes[i]).unwrap()[(0, 0)];
        assert!((m - ref_marg[i]).abs() < 1e-9, "node {}: sel {} vs solve {}", i, m, ref_marg[i]);
    }
    // Adjacent nodes are connected in the factor -> in-pattern lookup.
    let adj = pc.cross_cov(&c.nodes[2], &c.nodes[3]).unwrap()[(0, 0)];
    assert!((adj - ref_adjacent).abs() < 1e-9, "adjacent cross: sel {} vs solve {}", adj, ref_adjacent);
    // The endpoints share no factor entry -> out of pattern -> solve fallback,
    // which must still return the correct (nonzero) coupling.
    let dist = pc.cross_cov(&c.nodes[0], &c.nodes[5]).unwrap()[(0, 0)];
    assert!((dist - ref_distant).abs() < 1e-9, "distant cross: sel {} vs solve {}", dist, ref_distant);
    assert!(ref_distant.abs() > 1e-6, "endpoints should still be correlated");
}

#[test]
fn precompute_selected_inverse_denser_fill() {
    // 20 nodes, each tied to its next two neighbours: a pentadiagonal H whose
    // factor fills in beyond the input band. Every marginal from the selected
    // inverse must still match the per-query solve.
    let n = 20;
    let mut c = Chain { nodes: refs::Vec::new(), ties: std::vec::Vec::new() };
    for _ in 0..n {
        c.nodes.push(Node { v: Param::new(0.0), prior_isig: 1.0, hb: SelfBlock::new() });
    }
    for i in 0..n {
        for step in [1usize, 2] {
            if i + step < n {
                c.ties.push(Tie {
                    a: c.nodes.ref_at(i),
                    b: c.nodes.ref_at(i + step),
                    isig: 1.0,
                    hb: CrossBlock::new(),
                });
            }
        }
    }

    let solved = c.assemble_covariance(CovMode::PerQuery).unwrap();
    let ref_marg: std::vec::Vec<f64> = (0..n).map(|i| solved.marginal_cov(&c.nodes[i]).unwrap()[(0, 0)]).collect();

    let pc = c.assemble_covariance(CovMode::AllMarginals).unwrap();
    for i in 0..n {
        let m = pc.marginal_cov(&c.nodes[i]).unwrap()[(0, 0)];
        assert!((m - ref_marg[i]).abs() < 1e-9, "node {}: sel {} vs solve {}", i, m, ref_marg[i]);
    }
}

#[test]
fn tridiagonal_matches_solve() {
    // A path chain is block-tridiagonal. Every marginal from the forward/backward
    // Schur sweeps must match the per-query solve -- the last node forward-only,
    // interior nodes via the (lazily triggered) backward pass.
    let mut c = path_chain(8);
    let solved = c.assemble_covariance(CovMode::PerQuery).unwrap();
    let refm: std::vec::Vec<f64> = (0..8).map(|i| solved.marginal_cov(&c.nodes[i]).unwrap()[(0, 0)]).collect();

    let band = c.assemble_covariance(CovMode::TriDiagonal).unwrap();
    // Last node: the forward-only path.
    let last = band.marginal_cov(&c.nodes[7]).unwrap()[(0, 0)];
    assert!((last - refm[7]).abs() < 1e-9, "last: band {} vs solve {}", last, refm[7]);
    // Interior nodes: trigger and reuse the backward pass.
    for i in 0..8 {
        let m = band.marginal_cov(&c.nodes[i]).unwrap()[(0, 0)];
        assert!((m - refm[i]).abs() < 1e-9, "node {}: band {} vs solve {}", i, m, refm[i]);
    }
}

#[test]
fn tridiagonal_rejects_singular_hessian() {
    // An untied first node with a zero-information prior contributes a
    // zero diagonal block: the forward sweep cannot invert it when it
    // reaches node 1, and the error must be NotPositiveDefinite, not a
    // NaN-poisoned covariance.
    let mut c = Chain { nodes: refs::Vec::new(), ties: std::vec::Vec::new() };
    c.nodes.push(Node { v: Param::new(0.0), prior_isig: 0.0, hb: SelfBlock::new() });
    for _ in 0..3 {
        c.nodes.push(Node { v: Param::new(0.0), prior_isig: 1.0, hb: SelfBlock::new() });
    }
    for i in 1..3 {
        c.ties.push(Tie { a: c.nodes.ref_at(i), b: c.nodes.ref_at(i + 1),
            isig: 1.0, hb: CrossBlock::new() });
    }
    assert_eq!(c.assemble_covariance(CovMode::TriDiagonal).err(),
        Some(CovError::NotPositiveDefinite));
}

#[test]
fn tridiagonal_rejects_non_band() {
    // A long-range tie (0 <-> 4) puts an off-band block into H.
    let mut c = path_chain(6);
    c.ties.push(Tie { a: c.nodes.ref_at(0), b: c.nodes.ref_at(4), isig: 1.0, hb: CrossBlock::new() });
    assert_eq!(c.assemble_covariance(CovMode::TriDiagonal).err(), Some(CovError::NotTriDiagonal));
}

// A 2-DOF pose whose prior couples x and y (full 2x2 diagonal block), tied to its
// neighbour so the coupling block H[i,i+1] is NON-symmetric -- which a 1-DOF chain
// cannot exercise, so it guards the transpose orientation in the Schur sweeps.
#[arael::model]
#[arael(constraint(hb, { [pose2.x * pose2.pi, (pose2.x + pose2.y) * pose2.pi] }))]
struct Pose2 {
    x: Param<f64>,
    y: Param<f64>,
    pi: f64,
    hb: SelfBlock<Pose2>,
}

#[arael::model]
#[arael(constraint(hb, { [(a.x - b.x) * tie2.t, (a.y - b.x) * tie2.t] }))]
struct Tie2 {
    #[arael(ref = root.poses)]
    a: Ref<Pose2>,
    #[arael(ref = root.poses)]
    b: Ref<Pose2>,
    t: f64,
    hb: CrossBlock<Pose2, Pose2>,
}

#[arael::model]
#[arael(root)]
struct Chain2 {
    poses: refs::Vec<Pose2>,
    ties: std::vec::Vec<Tie2>,
}

#[test]
fn tridiagonal_2dof_matches_solve() {
    let n = 6;
    let mut c = Chain2 { poses: refs::Vec::new(), ties: std::vec::Vec::new() };
    for _ in 0..n {
        c.poses.push(Pose2 { x: Param::new(0.0), y: Param::new(0.0), pi: 1.0, hb: SelfBlock::new() });
    }
    for i in 0..n - 1 {
        c.ties.push(Tie2 { a: c.poses.ref_at(i), b: c.poses.ref_at(i + 1), t: 1.0, hb: CrossBlock::new() });
    }

    let solved = c.assemble_covariance(CovMode::PerQuery).unwrap();
    let band = c.assemble_covariance(CovMode::TriDiagonal).unwrap();
    for i in 0..n {
        let s = solved.marginal_cov(&c.poses[i]).unwrap();
        let b = band.marginal_cov(&c.poses[i]).unwrap();
        assert_eq!((b.nrows(), b.ncols()), (2, 2));
        for r in 0..2 {
            for cc in 0..2 {
                assert!((s[(r, cc)] - b[(r, cc)]).abs() < 1e-9,
                    "pose {} [{},{}]: solve {} vs band {}", i, r, cc, s[(r, cc)], b[(r, cc)]);
            }
        }
    }
}

#[test]
fn all_marginals_dense_matches_solve() {
    // A denser 2-DOF graph (each pose tied to its next three) fills the factor
    // in, forming real supernodes -- exercises the block selected inverse and its
    // Sigma_RR gather. Every marginal and a coupled cross block must match the
    // per-query solve.
    let n = 10;
    let mut c = Chain2 { poses: refs::Vec::new(), ties: std::vec::Vec::new() };
    for _ in 0..n {
        c.poses.push(Pose2 { x: Param::new(0.0), y: Param::new(0.0), pi: 1.0, hb: SelfBlock::new() });
    }
    for i in 0..n {
        for step in [1usize, 2, 3] {
            if i + step < n {
                c.ties.push(Tie2 { a: c.poses.ref_at(i), b: c.poses.ref_at(i + step), t: 1.0, hb: CrossBlock::new() });
            }
        }
    }
    let solved = c.assemble_covariance(CovMode::PerQuery).unwrap();
    let allm = c.assemble_covariance(CovMode::AllMarginals).unwrap();
    for i in 0..n {
        let s = solved.marginal_cov(&c.poses[i]).unwrap();
        let a = allm.marginal_cov(&c.poses[i]).unwrap();
        for r in 0..2 {
            for cc in 0..2 {
                assert!((s[(r, cc)] - a[(r, cc)]).abs() < 1e-9,
                    "pose {} [{},{}]: solve {} vs allmarg {}", i, r, cc, s[(r, cc)], a[(r, cc)]);
            }
        }
    }
    // Coupled cross block: in the factor pattern, so AllMarginals answers it from
    // the cache; it must match the solve.
    let sc = solved.cross_cov(&c.poses[0], &c.poses[2]).unwrap();
    let ac = allm.cross_cov(&c.poses[0], &c.poses[2]).unwrap();
    for r in 0..2 {
        for cc in 0..2 {
            assert!((sc[(r, cc)] - ac[(r, cc)]).abs() < 1e-9, "cross [{},{}]: {} vs {}", r, cc, sc[(r, cc)], ac[(r, cc)]);
        }
    }
}

#[test]
fn empty_model_is_an_error() {
    let mut w = W { pts: refs::Vec::new() };
    assert_eq!(w.assemble_covariance(CovMode::PerQuery).err(), Some(CovError::Empty));
}

// A 2-DOF chain whose last pose leaves y unobservable: its prior weights y at
// zero (piy = 0) and the ties couple only x. Every other pose is fully anchored,
// so build_band's forward pass succeeds -- but the last raw diagonal block is
// singular, so an interior marginal's backward pass inverts a singular block. It
// must not panic (pseudo-inverse carries it), interior marginals stay finite,
// and the unobservable direction reads INFINITY.
#[arael::model]
#[arael(constraint(hb, { [posey.x * posey.pix, posey.y * posey.piy] }))]
struct PoseY {
    x: Param<f64>,
    y: Param<f64>,
    pix: f64,
    piy: f64,
    hb: SelfBlock<PoseY>,
}

#[arael::model]
#[arael(constraint(hb, { [(a.x - b.x) * tiey.t] }))]
struct TieY {
    #[arael(ref = root.poses)]
    a: Ref<PoseY>,
    #[arael(ref = root.poses)]
    b: Ref<PoseY>,
    t: f64,
    hb: CrossBlock<PoseY, PoseY>,
}

#[arael::model]
#[arael(root)]
struct ChainY {
    poses: refs::Vec<PoseY>,
    ties: std::vec::Vec<TieY>,
}

#[test]
fn tridiagonal_singular_backward_block_does_not_panic() {
    let n = 4;
    let mut c = ChainY { poses: refs::Vec::new(), ties: std::vec::Vec::new() };
    for i in 0..n {
        // Last pose: no y information anywhere, so its diagonal block is singular.
        let piy = if i == n - 1 { 0.0 } else { 1.0 };
        c.poses.push(PoseY { x: Param::new(0.0), y: Param::new(0.0), pix: 1.0, piy, hb: SelfBlock::new() });
    }
    for i in 0..n - 1 {
        c.ties.push(TieY { a: c.poses.ref_at(i), b: c.poses.ref_at(i + 1), t: 1.0, hb: CrossBlock::new() });
    }

    let band = c.assemble_covariance(CovMode::TriDiagonal).unwrap();
    // An interior marginal triggers the backward pass, which inverts the singular
    // last block. Before the fix this panicked; now it stays finite.
    let m = band.marginal_cov(&c.poses[1]).unwrap();
    assert_eq!((m.nrows(), m.ncols()), (2, 2));
    assert!(m[(0, 0)].is_finite() && m[(1, 1)].is_finite(), "interior pose stays finite: {:?}", m);
    // The last pose's y is unobservable: its marginal has no finite
    // answer and reports the singularity instead of an INFINITY fill.
    assert_eq!(band.marginal_cov(&c.poses[n - 1]).err(),
        Some(CovError::NotPositiveDefinite));
}

#[test]
fn tridiagonal_cross_cov_is_unsupported() {
    let mut c = path_chain(6);
    let band = c.assemble_covariance(CovMode::TriDiagonal).unwrap();
    // The band backend stores no off-diagonal blocks: a typed error, not
    // a sentinel.
    assert_eq!(band.cross_cov(&c.nodes[0], &c.nodes[1]).err(),
        Some(CovError::UnsupportedQuery { op: "cross_cov" }));
}

#[test]
fn tridiagonal_multiblock_query_is_unsupported() {
    let mut c = path_chain(6);
    let band = c.assemble_covariance(CovMode::TriDiagonal).unwrap();
    // Querying the whole root spans every block; the band backend cannot
    // answer that as a single marginal.
    assert_eq!(band.marginal_cov(&c).err(),
        Some(CovError::UnsupportedQuery { op: "marginal_cov" }));
    assert_eq!(band.std_dev(&c).err(),
        Some(CovError::UnsupportedQuery { op: "std_dev" }));
}

// --- 3-DOF chain. The models above are 1- and 2-DOF; a block ordering has
// nothing to divide out at one scalar per block, so this is the model that
// exercises CovOrdering::Auto's block path.

#[arael::model]
#[arael(constraint(hb, {
    [n3.x * n3.prior_isig, n3.y * n3.prior_isig, n3.z * n3.prior_isig]
}))]
struct N3 {
    x: Param<f64>,
    y: Param<f64>,
    z: Param<f64>,
    prior_isig: f64,
    hb: SelfBlock<N3>,
}

#[arael::model]
#[arael(constraint(hb, {
    [(a.x - b.x) * tie3.isig, (a.y - b.y) * tie3.isig, (a.z - b.z) * tie3.isig]
}))]
struct Tie3 {
    #[arael(ref = root.nodes)]
    a: Ref<N3>,
    #[arael(ref = root.nodes)]
    b: Ref<N3>,
    isig: f64,
    hb: CrossBlock<N3, N3>,
}

#[arael::model]
#[arael(root)]
struct Chain3 {
    nodes: refs::Vec<N3>,
    ties: std::vec::Vec<Tie3>,
}

fn chain3(n: usize) -> Chain3 {
    let mut c = Chain3 { nodes: refs::Vec::new(), ties: std::vec::Vec::new() };
    for _ in 0..n {
        c.nodes.push(N3 {
            x: Param::new(0.0), y: Param::new(0.0), z: Param::new(0.0),
            prior_isig: 1.0, hb: SelfBlock::new(),
        });
    }
    for i in 0..n - 1 {
        c.ties.push(Tie3 {
            a: c.nodes.ref_at(i), b: c.nodes.ref_at(i + 1), isig: 1.0, hb: CrossBlock::new(),
        });
    }
    c
}

#[test]
fn block_factorization_matches_the_scalar_one() {
    // The block supernodal Cholesky and faer's scalar one factor the same H.
    // Every query has to come back the same, whichever produced the factor.
    let n = 40;
    let mut c = chain3(n);
    let block = c.assemble_covariance_with(CovMode::PerQuery, &CovOptions::auto()).unwrap();
    let scalar = c
        .assemble_covariance_with(
            CovMode::PerQuery,
            &CovOptions::auto().with_block_supernodal(BlockSupernodalMode::Never),
        )
        .unwrap();
    assert!(block.took_block_route(), "3-DOF blocks should take the block route");
    assert!(!scalar.took_block_route(), "Never must stay on the scalar factor");

    for i in 0..n {
        let b = block.marginal_cov(&c.nodes[i]).unwrap();
        let s = scalar.marginal_cov(&c.nodes[i]).unwrap();
        for r in 0..3 {
            for k in 0..3 {
                assert!((b[(r, k)] - s[(r, k)]).abs() < 1e-10,
                    "marginal node {i} [{r},{k}]: block {} vs scalar {}", b[(r, k)], s[(r, k)]);
            }
        }
        // conditional_cov reads H itself -- out of the block matrix on one
        // route and out of the scalar CSC on the other.
        let bc = block.conditional_cov(&c.nodes[i]).unwrap();
        let sc = scalar.conditional_cov(&c.nodes[i]).unwrap();
        for r in 0..3 {
            for k in 0..3 {
                assert!((bc[(r, k)] - sc[(r, k)]).abs() < 1e-10,
                    "conditional node {i} [{r},{k}]: block {} vs scalar {}", bc[(r, k)], sc[(r, k)]);
            }
        }
    }

    // A coupled pair and a distant one: in and out of the factor's pattern.
    for (a, b_) in [(5usize, 6usize), (0, 39)] {
        let x = block.cross_cov(&c.nodes[a], &c.nodes[b_]).unwrap()[(0, 0)];
        let y = scalar.cross_cov(&c.nodes[a], &c.nodes[b_]).unwrap()[(0, 0)];
        assert!((x - y).abs() < 1e-10, "cross {a}-{b_}: block {x} vs scalar {y}");
    }

    // std_dev goes through the same solve.
    let bs = block.std_dev(&c.nodes[7]).unwrap();
    let ss = scalar.std_dev(&c.nodes[7]).unwrap();
    for (x, y) in bs.iter().zip(&ss) {
        assert!((x - y).abs() < 1e-10, "std_dev: block {x} vs scalar {y}");
    }
}

#[test]
fn all_marginals_declines_the_block_route() {
    // The selected inverse reads faer's supernode panels, so AllMarginals
    // stays on the scalar factor however loudly it is asked otherwise.
    let mut c = chain3(8);
    let a = c
        .assemble_covariance_with(
            CovMode::AllMarginals,
            &CovOptions::auto().with_block_supernodal(BlockSupernodalMode::Always),
        )
        .unwrap();
    assert!(!a.took_block_route());
    // And it still answers correctly.
    let s = c
        .assemble_covariance_with(
            CovMode::PerQuery,
            &CovOptions::auto().with_block_supernodal(BlockSupernodalMode::Never),
        )
        .unwrap();
    for i in 0..8 {
        let x = a.marginal_cov(&c.nodes[i]).unwrap()[(0, 0)];
        let y = s.marginal_cov(&c.nodes[i]).unwrap()[(0, 0)];
        assert!((x - y).abs() < 1e-10, "node {i}: {x} vs {y}");
    }
}

#[test]
fn one_scalar_per_block_declines_the_block_route() {
    // A 1-DOF chain: the block partition is the scalar one, so there is
    // nothing for the block route to divide out.
    let mut c = path_chain(8);
    let a = c
        .assemble_covariance_with(
            CovMode::PerQuery,
            &CovOptions::auto().with_block_supernodal(BlockSupernodalMode::Always),
        )
        .unwrap();
    assert!(!a.took_block_route());
}

#[test]
fn ordering_does_not_change_the_covariance() {
    // The ordering decides how much fill the factor carries, never the answer.
    let n = 40;
    let mut c = chain3(n);
    for mode in [CovMode::PerQuery, CovMode::AllMarginals] {
        let auto = c.assemble_covariance_with(mode, &CovOptions::auto()).unwrap();
        let amd = c
            .assemble_covariance_with(mode, &CovOptions::auto().with_ordering(CovOrdering::Amd))
            .unwrap();
        let nat = c
            .assemble_covariance_with(mode, &CovOptions::auto().with_ordering(CovOrdering::Natural))
            .unwrap();
        let nd = c
            .assemble_covariance_with(
                mode,
                &CovOptions::auto().with_ordering(CovOrdering::NestedDissection),
            )
            .unwrap();

        for i in 0..n {
            let a = auto.marginal_cov(&c.nodes[i]).unwrap();
            for (label, other) in [("amd", &amd), ("natural", &nat), ("nd", &nd)] {
                let b = other.marginal_cov(&c.nodes[i]).unwrap();
                for r in 0..3 {
                    for k in 0..3 {
                        assert!((a[(r, k)] - b[(r, k)]).abs() < 1e-10,
                            "{mode:?} {label} node {i} [{r},{k}]: {} vs {}", a[(r, k)], b[(r, k)]);
                    }
                }
            }
        }

        // Neighbours are coupled, so this cross block is inside the factor's
        // pattern -- the selected inverse reads it there rather than solving.
        let a = auto.cross_cov(&c.nodes[5], &c.nodes[6]).unwrap()[(0, 0)];
        let b = nat.cross_cov(&c.nodes[5], &c.nodes[6]).unwrap()[(0, 0)];
        assert!(a.abs() > 1e-6, "{mode:?}: neighbours should be correlated");
        assert!((a - b).abs() < 1e-10, "{mode:?}: cross {a} vs {b}");
    }

    // The plain entry point is the Auto path.
    let plain = c.assemble_covariance(CovMode::PerQuery).unwrap();
    let auto = c.assemble_covariance_with(CovMode::PerQuery, &CovOptions::auto()).unwrap();
    assert_eq!(
        plain.marginal_cov(&c.nodes[0]).unwrap()[(0, 0)],
        auto.marginal_cov(&c.nodes[0]).unwrap()[(0, 0)],
    );
}
