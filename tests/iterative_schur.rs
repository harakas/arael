// Iterative Schur: conjugate gradients on the reduced system instead of
// factorizing it (`SchurSolve::Iterative`). The step it produces is inexact
// by construction, so what these tests pin is that the SOLVE still lands on
// the same optimum as the factorizing route, that the route is selected
// explicitly and never by accident, and that asking for it where there is no
// reduced system is an error rather than a silent fallback.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{
    lm_solve, CgOptions, LmConfig, LmProblem, RootProblem, SchurPolicy, SolveError,
    SolveFailureKind, SolverReport, SparseFaer,
};

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
#[arael(root, marginalize(landmarks))]
struct World {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<Landmark>,
    odos: std::vec::Vec<Odo>,
    obs: std::vec::Vec<Obs>,
}

const N_POSES: usize = 4;
const N_LANDMARKS: usize = 6;

/// Consistent mini-SLAM: poses on a line, landmarks above it, exact
/// odometry and observation measurements, weak pose priors fixing the
/// gauge. Everything starts at the true optimum shifted by `off`.
fn build(off: f64) -> World {
    let mut w = World {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    let pose_true = |i: usize| (i as f64, 0.0);
    let lm_true = |j: usize| (j as f64 * 0.6, 1.0);
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
            x: Param::new(tx + off),
            y: Param::new(ty + off),
            hb: SelfBlock::new(),
        });
    }
    for i in 1..N_POSES {
        let (ax, ay) = pose_true(i - 1);
        let (bx, by) = pose_true(i);
        w.odos.push(Odo {
            a: w.poses.ref_at(i - 1),
            b: w.poses.ref_at(i),
            dx: bx - ax,
            dy: by - ay,
            hb: CrossBlock::new(),
        });
    }
    // Each landmark observed from two nearby poses.
    for j in 0..N_LANDMARKS {
        let (lx, ly) = lm_true(j);
        for pi in [j % N_POSES, (j + 1) % N_POSES] {
            let (px, py) = pose_true(pi);
            w.obs.push(Obs {
                p: w.poses.ref_at(pi),
                l: w.landmarks.ref_at(j),
                dx: lx - px,
                dy: ly - py,
                hb: CrossBlock::new(),
            });
        }
    }
    w
}

/// The model's parameter vector, which `serialize` fills in place.
fn x0_of(w: &mut World) -> std::vec::Vec<f64> {
    let mut x = std::vec::Vec::new();
    w.serialize(&mut x);
    x
}

fn cg(tol: f64) -> CgOptions {
    CgOptions { tol, ..Default::default() }
}

/// The reduced system solved iteratively reaches the same optimum as the one
/// solved by Cholesky. A loose inner tolerance is deliberate: it is the outer
/// solve that has to land, not each inner one.
#[test]
fn iterative_matches_the_factorizing_route() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut wf = build(0.05);
    let rf = lm_solve(
        &x0_of(&mut wf),
        &mut SparseFaer::new().with_policy(SchurPolicy::Force),
        &mut wf,
        &cfg,
    )
    .unwrap();
    wf.deserialize(&rf.x);

    let mut wi = build(0.05);
    let ri = lm_solve(
        &x0_of(&mut wi),
        &mut SparseFaer::new()
            .with_policy(SchurPolicy::Force)
            .with_iterative_schur(cg(1e-10)),
        &mut wi,
        &cfg,
    )
    .unwrap();
    wi.deserialize(&ri.x);

    assert!(ri.end_cost < 1e-12, "iterative end_cost {}", ri.end_cost);
    for j in 0..N_LANDMARKS {
        let (a, b) = (&wf.landmarks[j], &wi.landmarks[j]);
        assert!((a.x.value - b.x.value).abs() < 1e-6, "landmark {} x", j);
        assert!((a.y.value - b.y.value).abs() < 1e-6, "landmark {} y", j);
    }
    for i in 0..N_POSES {
        let (a, b) = (&wf.poses[i], &wi.poses[i]);
        assert!((a.x.value - b.x.value).abs() < 1e-6, "pose {} x", i);
        assert!((a.y.value - b.y.value).abs() < 1e-6, "pose {} y", i);
    }
}

/// The plan reports the CG work, and only on the iterative route -- a
/// factorizing solve has no CG iterations rather than zero of them.
#[test]
fn cg_iterations_are_reported() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut wi = build(0.05);
    let ri = lm_solve(
        &x0_of(&mut wi),
        &mut SparseFaer::new()
            .with_policy(SchurPolicy::Force)
            .with_iterative_schur(cg(1e-10)),
        &mut wi,
        &cfg,
    )
    .unwrap();
    let Some(SolverReport::Schur(plan)) = ri.solver else {
        panic!("no Schur plan reported");
    };
    assert!(plan.reduced);
    let iters = plan.cg_iterations.expect("iterative route reports CG iterations");
    assert!(iters > 0, "no CG iterations recorded");

    let mut wf = build(0.05);
    let rf = lm_solve(
        &x0_of(&mut wf),
        &mut SparseFaer::new().with_policy(SchurPolicy::Force),
        &mut wf,
        &cfg,
    )
    .unwrap();
    let Some(SolverReport::Schur(plan)) = rf.solver else {
        panic!("no Schur plan reported");
    };
    assert_eq!(plan.cg_iterations, None, "factorizing route reported CG work");
}

/// CG never factorizes the reduced system, so nothing orders it either. A
/// reported ordering would mean the symbolic analysis -- and the factor buffers
/// sized from it -- were built and thrown away.
#[test]
fn iterative_orders_nothing() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut wi = build(0.05);
    let ri = lm_solve(
        &x0_of(&mut wi),
        &mut SparseFaer::new()
            .with_policy(SchurPolicy::Force)
            .with_iterative_schur(cg(1e-10)),
        &mut wi,
        &cfg,
    )
    .unwrap();
    let Some(SolverReport::Schur(plan)) = ri.solver else {
        panic!("no Schur plan reported");
    };
    assert!(plan.reduced);
    assert_eq!(plan.ordering, None, "iterative route ordered the reduced system");
    assert!(!plan.narrow_band, "iterative route took the band factorization");

    // The factorizing route does order it -- so the assertion above is about
    // the route, not about the plan never carrying an ordering.
    let mut wf = build(0.05);
    let rf = lm_solve(
        &x0_of(&mut wf),
        &mut SparseFaer::new().with_policy(SchurPolicy::Force),
        &mut wf,
        &cfg,
    )
    .unwrap();
    let Some(SolverReport::Schur(plan)) = rf.solver else {
        panic!("no Schur plan reported");
    };
    assert!(plan.ordering.is_some(), "factorizing route reported no ordering");
}

/// No reduction, nothing to run CG on. Factorizing the whole system instead
/// would answer a different question without saying so, so it is an error.
#[test]
fn iterative_without_a_reduction_is_an_error() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let mut w = build(0.05);
    let err = lm_solve(
        &x0_of(&mut w),
        &mut SparseFaer::new()
            .with_policy(SchurPolicy::Never)
            .with_iterative_schur(cg(1e-10)),
        &mut w,
        &cfg,
    )
    .unwrap_err();
    assert!(
        matches!(
            err.kind,
            SolveFailureKind::Setup(SolveError::IterativeSchurWithoutReduction)
        ),
        "wrong failure: {:?}",
        err.kind
    );
}

/// A tight tolerance costs more inner iterations than a loose one, and both
/// still land the outer solve. This is the knob the caller trades with.
#[test]
fn tolerance_trades_inner_work() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let mut counts = std::vec::Vec::new();
    for tol in [1e-2, 1e-12] {
        let mut w = build(0.05);
        let r = lm_solve(
            &x0_of(&mut w),
            &mut SparseFaer::new()
                .with_policy(SchurPolicy::Force)
                .with_iterative_schur(cg(tol)),
            &mut w,
            &cfg,
        )
        .unwrap();
        assert!(r.end_cost < 1e-10, "tol {}: end_cost {}", tol, r.end_cost);
        let Some(SolverReport::Schur(plan)) = r.solver else {
            panic!("no Schur plan");
        };
        counts.push(plan.cg_iterations.unwrap());
    }
    assert!(counts[1] >= counts[0],
        "tighter tolerance did not cost at least as much: {:?}", counts);
}
