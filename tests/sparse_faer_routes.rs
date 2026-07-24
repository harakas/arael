// SparseFaer is one backend with two routes: marginalize the model's
// landmark-like blocks and factorize what is left, or factorize the whole
// system. It picks for itself; SchurPolicy and FaerOrdering override the
// pick. Every route must reach the same optimum -- they are the same
// equations, solved in a different order.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{SolveFailureKind,
    lm_solve, CooMatrix, CscMatrix, FaerOrdering, LmConfig, LmProblem, RootProblem,
    SchurPolicy, SolveError, SolverReport, SparseFaer,
};

// --- a model with marginalizable blocks: poses seeing shared landmarks ---

#[arael::model]
#[arael(constraint(hb, {
    [(pose.x - pose.ax) * 0.1, (pose.y - pose.ay) * 0.1]
}))]
struct Pose {
    x: Param<f64>,
    y: Param<f64>,
    ax: f64,
    ay: f64,
    hb: SelfBlock<Pose>,
}

#[arael::model]
struct Landmark {
    x: Param<f64>,
    y: Param<f64>,
    hb: SelfBlock<Landmark>,
}

// Odometry couples consecutive poses to each other -- which is exactly what
// makes the poses NOT marginalizable, and the landmarks the only family that
// is. Without it both families would be legal (the bundle-adjustment shape).
#[arael::model]
#[arael(constraint(hb, {
    [b.x - a.x - odo.dx, b.y - a.y - odo.dy]
}))]
struct Odo {
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Pose, Pose>,
}

#[arael::model]
#[arael(constraint(hb, {
    [l.x - p.x - obs.dx, l.y - p.y - obs.dy]
}))]
struct Obs {
    #[arael(ref = root.poses)]
    p: Ref<Pose>,
    #[arael(ref = root.landmarks)]
    l: Ref<Landmark>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Pose, Landmark>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<Landmark>,
    odos: std::vec::Vec<Odo>,
    obs: std::vec::Vec<Obs>,
}

const N_POSES: usize = 6;
const N_LANDMARKS: usize = 8;

fn build(off: f64) -> World {
    let mut w = World {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    let pose_true = |i: usize| (i as f64, 0.5 * i as f64);
    let lm_true = |j: usize| (j as f64, 2.0 + j as f64);
    for i in 0..N_POSES {
        let (tx, ty) = pose_true(i);
        w.poses.push(Pose {
            x: Param::new(tx + off),
            y: Param::new(ty - off),
            ax: tx,
            ay: ty,
            hb: SelfBlock::new(),
        });
    }
    for j in 0..N_LANDMARKS {
        let (tx, ty) = lm_true(j);
        w.landmarks.push(Landmark {
            x: Param::new(tx - off),
            y: Param::new(ty + off),
            hb: SelfBlock::new(),
        });
    }
    for i in 1..N_POSES {
        let (ax, ay) = pose_true(i - 1);
        let (bx, by) = pose_true(i);
        w.odos.push(Odo {
            a: w.poses.ref_at((i - 1)),
            b: w.poses.ref_at(i),
            dx: bx - ax,
            dy: by - ay,
            hb: CrossBlock::new(),
        });
    }
    // Every landmark is seen from every pose: landmarks couple to poses,
    // never to each other, which is what makes them marginalizable.
    for i in 0..N_POSES {
        for j in 0..N_LANDMARKS {
            let (px, py) = pose_true(i);
            let (lx, ly) = lm_true(j);
            w.obs.push(Obs {
                p: w.poses.ref_at(i),
                l: w.landmarks.ref_at(j),
                dx: lx - px,
                dy: ly - py,
                hb: CrossBlock::new(),
            });
        }
    }
    w
}

fn solved_params(solver: &mut SparseFaer<f64>) -> (Vec<f64>, f64) {
    let cfg = LmConfig { max_iters: 60, ..Default::default() };
    let mut w = build(0.05);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let r = lm_solve(&params, solver, &mut w, &cfg).unwrap();
    let mut out = Vec::new();
    RootProblem::serialize(&mut w, &mut out);
    (out, r.end_cost)
}

/// Left alone, the backend finds the landmarks and marginalizes them.
#[test]
fn auto_marginalizes_what_it_finds() {
    let mut solver = SparseFaer::new();
    let (_, cost) = solved_params(&mut solver);
    assert!(cost < 1e-14, "end_cost {}", cost);
    let plan = solver.plan().expect("a plan");
    assert!(plan.reduced, "the landmarks should have been marginalized");
    assert_eq!(plan.eliminated_blocks, N_LANDMARKS);
    assert_eq!(plan.kept_params, 2 * N_POSES);
}

/// Never means never: the whole system, factorized as one. Same optimum --
/// it is the same set of equations either way.
#[test]
fn never_solves_the_whole_system_to_the_same_answer() {
    let (reduced, c_reduced) = solved_params(&mut SparseFaer::new());

    let mut solver = SparseFaer::new().with_policy(SchurPolicy::Never);
    let (whole, c_whole) = solved_params(&mut solver);

    let plan = solver.plan().expect("a plan");
    assert!(!plan.reduced, "Never must not marginalize");
    assert_eq!(plan.eliminated_blocks, 0);
    assert_eq!(plan.kept_params, 2 * (N_POSES + N_LANDMARKS));

    assert!(c_whole < 1e-14, "end_cost {}", c_whole);
    assert!((c_whole - c_reduced).abs() < 1e-12);
    for (i, (a, b)) in std::iter::zip(&whole, &reduced).enumerate() {
        assert!((a - b).abs() < 1e-8, "param {}: {} vs {}", i, a, b);
    }
}

/// Force marginalizes without weighing it: no fill comparison, no cheap
/// filter, nothing in the plan but the reduction itself.
#[test]
fn force_skips_the_analysis() {
    let mut solver = SparseFaer::new().with_policy(SchurPolicy::Force);
    let (_, cost) = solved_params(&mut solver);
    assert!(cost < 1e-14, "end_cost {}", cost);
    let plan = solver.plan().expect("a plan");
    assert!(plan.reduced);
    assert!(plan.fill_ratio.is_none() && plan.flop_ratio.is_none());
}

/// Every ordering of the whole system reaches the same optimum. They differ
/// in fill and speed, never in the answer.
#[test]
fn every_ordering_of_the_whole_system_agrees() {
    let (reference, _) = solved_params(&mut SparseFaer::new());
    for ordering in [
        FaerOrdering::Auto,
        FaerOrdering::Amd,
        FaerOrdering::MarginalizeFirst,
        FaerOrdering::Natural,
    ] {
        let mut solver = SparseFaer::new()
            .with_policy(SchurPolicy::Never)
            .with_ordering(ordering)
            // Named, so MarginalizeFirst has something to put first.
            .with_marginalize(2 * N_POSES..2 * (N_POSES + N_LANDMARKS));
        let (got, cost) = solved_params(&mut solver);
        assert!(cost < 1e-14, "{:?}: end_cost {}", ordering, cost);
        for (i, (a, b)) in std::iter::zip(&got, &reference).enumerate() {
            assert!((a - b).abs() < 1e-8, "{:?}: param {}: {} vs {}", ordering, i, a, b);
        }
    }
}

/// Naming blocks that couple to each other is a mistake in the model
/// description, not a slow path: marginalizing them is not defined. The solve
/// fails at setup with a coupled-marginalization error rather than a panic.
#[test]
fn naming_coupled_blocks_is_rejected() {
    // Odometry joins consecutive poses directly, so the poses are not a legal
    // marginalize set.
    let mut solver = SparseFaer::new()
        .with_policy(SchurPolicy::Force)
        .with_marginalize(0..2 * N_POSES);
    let cfg = LmConfig { max_iters: 60, ..Default::default() };
    let mut w = build(0.05);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let e = lm_solve(&params, &mut solver, &mut w, &cfg)
        .expect_err("coupled marginalization must fail at setup");
    assert!(
        matches!(e.kind, SolveFailureKind::Setup(SolveError::CoupledMarginalization { .. })),
        "expected a coupled-marginalization setup failure, got {:?}",
        e.kind
    );
    // The solve did not run: there is no partial state.
    assert!(e.partial.is_none());
}

// --- a pose graph: every block couples to another one ---

#[arael::model]
#[arael(root)]
struct Chain {
    poses: refs::Vec<Pose>,
    odos: std::vec::Vec<Odo>,
}

/// A pose graph has nothing marginalizable in it -- odometry couples every
/// pose to its neighbours, so no set of blocks is mutually uncoupled. The
/// backend must see that and take the plain route, without building any of
/// the block machinery. This is what makes the reduction free to have around
/// for models that cannot use it.
#[test]
fn a_pose_graph_is_never_reduced() {
    let mut c = Chain { poses: refs::Vec::new(), odos: std::vec::Vec::new() };
    for i in 0..5 {
        let (tx, ty) = (i as f64, 0.5 * i as f64);
        c.poses.push(Pose {
            x: Param::new(tx + 0.05),
            y: Param::new(ty - 0.05),
            ax: tx,
            ay: ty,
            hb: SelfBlock::new(),
        });
    }
    for i in 1..5 {
        c.odos.push(Odo {
            a: c.poses.ref_at((i - 1)),
            b: c.poses.ref_at(i),
            dx: 1.0,
            dy: 0.5,
            hb: CrossBlock::new(),
        });
    }
    assert!(
        LmProblem::marginalize_candidates(&c).is_empty(),
        "a pose chain offers nothing to marginalize"
    );

    let cfg = LmConfig { max_iters: 40, ..Default::default() };
    let mut params = Vec::new();
    RootProblem::serialize(&mut c, &mut params);
    let mut solver = SparseFaer::new();
    let r = lm_solve(&params, &mut solver, &mut c, &cfg).unwrap();
    assert!(r.end_cost < 1e-14, "end_cost {}", r.end_cost);

    let plan = solver.plan().expect("a plan");
    assert!(!plan.reduced);
    assert_eq!(plan.eliminated_blocks, 0);
    // The cheap filter never ran either: there was nothing to filter.
    assert!(plan.flop_ratio.is_none() && plan.fill_ratio.is_none());
}

// --- a hand-built problem: no macro, no blocks, no structure to walk ---

/// Fits `y = a * x + b` to three points, with a COO Hessian assembled by
/// hand. There is no model, so no block structure and no marginalize
/// candidates -- the backend has to discover the pattern by a COO pass and
/// factorize the whole system, which is the only route open to it.
struct LineFit {
    pts: Vec<(f64, f64)>,
}

impl LmProblem<f64> for LineFit {
    fn calc_cost(&mut self, p: &[f64]) -> f64 {
        self.pts.iter().map(|&(x, y)| (p[0] * x + p[1] - y).powi(2)).sum::<f64>() * 0.5
    }

    fn calc_grad_hessian_dense(&mut self, p: &[f64], grad: &mut [f64], h: &mut [f64]) -> f64 {
        grad.fill(0.0);
        h.fill(0.0);
        for &(x, y) in &self.pts {
            let r = p[0] * x + p[1] - y;
            let j = [x, 1.0];
            for a in 0..2 {
                grad[a] += j[a] * r;
                for b in 0..2 {
                    h[a + b * 2] += j[a] * j[b];
                }
            }
        }
        self.calc_cost(p)
    }

    fn calc_grad_hessian_band(
        &mut self, _p: &[f64], _g: &mut [f64], _b: &mut [f64], _kd: usize,
    ) -> Result<f64, arael::simple_lm::BandError> {
        unimplemented!("dense/sparse only")
    }

    fn calc_grad_hessian_sparse(
        &mut self, p: &[f64], grad: &mut [f64], coo: &mut CooMatrix<f64>,
    ) -> f64 {
        grad.fill(0.0);
        for &(x, y) in &self.pts {
            let r = p[0] * x + p[1] - y;
            let j = [x, 1.0];
            for a in 0..2 {
                grad[a] += j[a] * r;
                // Upper triangle only -- the storage convention every sparse
                // backend here reads.
                for b in a..2 {
                    coo.push(a as u32, b as u32, j[a] * j[b]);
                }
            }
        }
        self.calc_cost(p)
    }

    fn calc_grad_hessian_sparse_direct(
        &mut self, _p: &[f64], _g: &mut [f64], _csc: &mut CscMatrix<f64>,
    ) -> f64 {
        unimplemented!("not used by SparseFaer")
    }

    fn calc_grad_hessian_sparse_indexed(
        &mut self, p: &[f64], grad: &mut [f64], vals: &mut [f64], positions: &[usize],
    ) -> f64 {
        grad.fill(0.0);
        vals.fill(0.0);
        let mut k = 0;
        for &(x, y) in &self.pts {
            let r = p[0] * x + p[1] - y;
            let j = [x, 1.0];
            for a in 0..2 {
                grad[a] += j[a] * r;
                for b in a..2 {
                    vals[positions[k]] += j[a] * j[b];
                    k += 1;
                }
            }
        }
        self.calc_cost(p)
    }
}

/// The default backend has to work for a problem built by hand -- no macro,
/// no block structure, nothing to marginalize and nothing to detect it from.
#[test]
fn hand_built_problem_solves_through_the_plain_route() {
    let mut p = LineFit { pts: vec![(0.0, 1.0), (1.0, 3.0), (2.0, 5.0)] };
    let cfg = LmConfig { max_iters: 40, ..Default::default() };
    let mut solver = SparseFaer::new();
    let r = lm_solve(&[0.0, 0.0], &mut solver, &mut p, &cfg).unwrap();

    // y = 2x + 1 fits the three points exactly.
    assert!((r.x[0] - 2.0).abs() < 1e-10, "slope {}", r.x[0]);
    assert!((r.x[1] - 1.0).abs() < 1e-10, "intercept {}", r.x[1]);
    assert!(r.end_cost < 1e-20, "end_cost {}", r.end_cost);

    let Some(SolverReport::Schur(plan)) = r.solver else {
        panic!("expected a plan, got {:?}", r.solver);
    };
    assert!(!plan.reduced, "there is nothing to marginalize here");
    assert_eq!(plan.kept_params, 2);
}

/// ... and asking it to marginalize such a problem anyway is an error, not a
/// silent no-op.
#[test]
#[should_panic(expected = "no block structure")]
fn forcing_a_reduction_on_a_hand_built_problem_is_rejected() {
    let mut p = LineFit { pts: vec![(0.0, 1.0), (1.0, 3.0)] };
    let cfg = LmConfig { max_iters: 5, ..Default::default() };
    let mut solver = SparseFaer::new().with_policy(SchurPolicy::Force);
    lm_solve(&[0.0, 0.0], &mut solver, &mut p, &cfg).unwrap();
}
