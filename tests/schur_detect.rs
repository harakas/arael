// Automatic detection of the marginalizable blocks, on a deliberately
// heterogeneous model: TWO pose sequences and TWO landmark types of
// different block sizes, interleaved in the root's field order so the
// eliminated parameters are not one contiguous run.
//
// The detector reasons over the model's coupling graph -- one node per
// entity type, one edge per CrossBlock<A, B>, a self-loop when A == B --
// and a set is marginalizable exactly when no edge joins two of its
// members. Here: Pose self-couples (odometry), so it is out; Point and
// Line couple only to Pose, so {Point, Line} is the answer. No hint.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem, RootProblem, SchurPolicy, SparseFaer};

// 2-parameter pose
#[arael::model]
#[arael(constraint(hb, {
    [(pose.x - pose.px) * 0.05, (pose.y - pose.py) * 0.05]
}))]
struct Pose {
    x: Param<f64>,
    y: Param<f64>,
    px: f64,
    py: f64,
    hb: SelfBlock<Pose>,
}

// 2-parameter landmark
#[arael::model]
#[arael(constraint(hb, {
    [(point.x - point.px) * 0.02, (point.y - point.py) * 0.02]
}))]
struct Point {
    x: Param<f64>,
    y: Param<f64>,
    px: f64,
    py: f64,
    hb: SelfBlock<Point>,
}

// 3-parameter landmark -- a DIFFERENT block size from Point, so the
// eliminated set spans two widths.
#[arael::model]
#[arael(constraint(hb, {
    [(line.a - line.pa) * 0.02, (line.b - line.pb) * 0.02, (line.c - line.pc) * 0.02]
}))]
struct Line {
    a: Param<f64>,
    b: Param<f64>,
    c: Param<f64>,
    pa: f64,
    pb: f64,
    pc: f64,
    hb: SelfBlock<Line>,
}

// odometry: Pose <-> Pose. This self-loop is what disqualifies Pose.
#[arael::model]
#[arael(constraint(hb, {
    [b.x - a.x - odo.dx, b.y - a.y - odo.dy]
}))]
struct Odo {
    #[arael(ref = root.poses_a)]
    a: Ref<Pose>,
    #[arael(ref = root.poses_a)]
    b: Ref<Pose>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Pose, Pose>,
}

// point observation: Pose <-> Point
#[arael::model]
#[arael(constraint(hb, {
    [p.x - pose.x - pointobs.dx, p.y - pose.y - pointobs.dy]
}))]
struct PointObs {
    #[arael(ref = root.poses_a)]
    pose: Ref<Pose>,
    #[arael(ref = root.points)]
    p: Ref<Point>,
    dx: f64,
    dy: f64,
    hb: CrossBlock<Pose, Point>,
}

// line observation: Pose <-> Line
#[arael::model]
#[arael(constraint(hb, {
    [l.a * pose.x + l.b * pose.y + l.c - lineobs.d]
}))]
struct LineObs {
    #[arael(ref = root.poses_b)]
    pose: Ref<Pose>,
    #[arael(ref = root.lines)]
    l: Ref<Line>,
    d: f64,
    hb: CrossBlock<Pose, Line>,
}

// Root fields INTERLEAVED: the eliminable parameters (points, lines) are
// split by a kept collection (poses_b) sitting between them.
#[arael::model]
#[arael(root)]
struct World {
    poses_a: refs::Vec<Pose>,
    points: refs::Vec<Point>,
    poses_b: refs::Vec<Pose>,
    lines: refs::Vec<Line>,
    odos: std::vec::Vec<Odo>,
    pobs: std::vec::Vec<PointObs>,
    lobs: std::vec::Vec<LineObs>,
}

const N_A: usize = 5; // poses in sequence A
const N_B: usize = 4; // poses in sequence B
const N_POINTS: usize = 6;
const N_LINES: usize = 3;

fn build() -> World {
    let mut w = World {
        poses_a: refs::Vec::new(),
        points: refs::Vec::new(),
        poses_b: refs::Vec::new(),
        lines: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        pobs: std::vec::Vec::new(),
        lobs: std::vec::Vec::new(),
    };
    for i in 0..N_A {
        w.poses_a.push(Pose {
            x: Param::new(i as f64 + 0.1),
            y: Param::new(0.2 * i as f64),
            px: i as f64,
            py: 0.0,
            hb: SelfBlock::new(),
        });
    }
    for i in 0..N_B {
        w.poses_b.push(Pose {
            x: Param::new(2.0 * i as f64 - 0.1),
            y: Param::new(1.0 + 0.1 * i as f64),
            px: 2.0 * i as f64,
            py: 1.0,
            hb: SelfBlock::new(),
        });
    }
    for j in 0..N_POINTS {
        w.points.push(Point {
            x: Param::new(0.5 * j as f64),
            y: Param::new(1.5 - 0.1 * j as f64),
            px: 0.5 * j as f64,
            py: 1.5,
            hb: SelfBlock::new(),
        });
    }
    for k in 0..N_LINES {
        w.lines.push(Line {
            a: Param::new(0.3 + 0.1 * k as f64),
            b: Param::new(1.0 - 0.1 * k as f64),
            c: Param::new(0.2 * k as f64),
            pa: 0.3,
            pb: 1.0,
            pc: 0.0,
            hb: SelfBlock::new(),
        });
    }
    // odometry chains sequence A only (sequence B is held by its priors
    // and its line observations -- two genuinely different pose groups)
    for i in 1..N_A {
        w.odos.push(Odo {
            a: w.poses_a.ref_at(i - 1),
            b: w.poses_a.ref_at(i),
            dx: 1.0,
            dy: 0.0,
            hb: CrossBlock::new(),
        });
    }
    // every point seen from two poses of sequence A
    for j in 0..N_POINTS {
        for pi in [j % N_A, (j + 2) % N_A] {
            w.pobs.push(PointObs {
                pose: w.poses_a.ref_at(pi),
                p: w.points.ref_at(j),
                dx: 0.4,
                dy: 0.9,
                hb: CrossBlock::new(),
            });
        }
    }
    // every line seen from two poses of sequence B
    for k in 0..N_LINES {
        for pi in [k % N_B, (k + 1) % N_B] {
            w.lobs.push(LineObs {
                pose: w.poses_b.ref_at(pi),
                l: w.lines.ref_at(k),
                d: 1.2,
                hb: CrossBlock::new(),
            });
        }
    }
    w
}

/// The macro must offer exactly one candidate set -- {Point, Line} -- and
/// its ranges must be the points' and lines' parameters, nothing else.
#[test]
fn detects_both_landmark_types() {
    let w = build();
    let candidates = LmProblem::marginalize_candidates(&w);
    assert_eq!(candidates.len(), 1, "expected one maximal candidate set, got {:?}", candidates);

    // serialize order: poses_a | points | poses_b | lines
    let pa = 2 * N_A;
    let pts = 2 * N_POINTS;
    let pb = 2 * N_B;
    let lns = 3 * N_LINES;
    assert_eq!(candidates[0], vec![pa..pa + pts, pa + pts + pb..pa + pts + pb + lns]);

    // and the eliminated parameters really are split by a kept run
    let eliminated: usize = candidates[0].iter().map(|r| r.len()).sum();
    assert_eq!(eliminated, pts + lns);
    assert_eq!(pa + pts + pb + lns, 2 * (N_A + N_B) + 2 * N_POINTS + 3 * N_LINES);
}

/// With no hint at all, the solver detects the landmarks, marginalizes both
/// types at once (2- and 3-parameter blocks in one reduction) and lands on
/// the dense solve's optimum.
#[test]
fn auto_schur_matches_dense() {
    let cfg = LmConfig { max_iters: 60, ..Default::default() };

    let mut wd = build();
    let rd = wd.solve_dense(&cfg).unwrap();

    let mut wq = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut wq, &mut params);
    let mut solver = SparseFaer::new(); // no hint, no policy
    let rq = wq.solve_with(&mut solver, &cfg).unwrap();

    let plan = solver.plan().expect("the first compute must record a plan");
    assert!(plan.reduced, "the reduction should have been kept: {:?}", plan);
    assert_eq!(plan.eliminated_blocks, N_POINTS + N_LINES);
    assert_eq!(plan.eliminated_params, 2 * N_POINTS + 3 * N_LINES);
    assert_eq!(plan.kept_params, 2 * (N_A + N_B));

    assert!(
        (rd.end_cost - rq.end_cost).abs() <= 1e-10 * (1.0 + rd.end_cost),
        "dense {} vs auto-schur {}",
        rd.end_cost,
        rq.end_cost
    );
    for j in 0..N_POINTS {
        let (a, b) = (&wd.points[j as usize], &wq.points[j as usize]);
        assert!((a.x.value - b.x.value).abs() < 1e-6, "point {} x", j);
        assert!((a.y.value - b.y.value).abs() < 1e-6, "point {} y", j);
    }
    // The two routes must reach the same OPTIMUM -- that is the invariant, and
    // it holds to machine precision. The line parameters themselves are only
    // determined to about 1e-7: a line here is weakly observed along one
    // direction, so two factorizations with different arithmetic order land at
    // slightly different points in the same flat valley. Asserting 1e-8 on them
    // was asserting more than the problem determines.
    assert!(
        (rd.end_cost - rq.end_cost).abs() < 1e-12 * (1.0 + rd.end_cost.abs()),
        "the routes reached different optima: dense {} vs schur {}",
        rd.end_cost,
        rq.end_cost
    );
    for k in 0..N_LINES {
        let (a, b) = (&wd.lines[k as usize], &wq.lines[k as usize]);
        assert!((a.a.value - b.a.value).abs() < 1e-6, "line {} a", k);
        assert!((a.c.value - b.c.value).abs() < 1e-6, "line {} c", k);
    }
    for i in 0..N_B {
        let (a, b) = (&wd.poses_b[i as usize], &wq.poses_b[i as usize]);
        assert!((a.x.value - b.x.value).abs() < 1e-6, "pose_b {} x", i);
    }
}

/// Forced marginalization skips the analysis entirely and must reach the
/// same optimum.
#[test]
fn forced_schur_matches_dense() {
    let cfg = LmConfig { max_iters: 60, ..Default::default() };

    let mut wd = build();
    let rd = wd.solve_dense(&cfg).unwrap();

    let mut wq = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut wq, &mut params);
    let mut solver = SparseFaer::new().with_policy(SchurPolicy::Force);
    let rq = wq.solve_with(&mut solver, &cfg).unwrap();

    let plan = solver.plan().unwrap();
    assert!(plan.reduced);
    assert_eq!(plan.fill_ratio, None, "Force must not run the analysis");
    assert_eq!(plan.eliminated_blocks, N_POINTS + N_LINES);
    assert!((rd.end_cost - rq.end_cost).abs() <= 1e-10 * (1.0 + rd.end_cost));
}

/// Declining the reduction (an impossible fill ratio) must fall back to
/// factorizing the full system -- same optimum, nothing marginalized.
#[test]
fn declined_schur_falls_back_to_full_system() {
    let cfg = LmConfig { max_iters: 60, ..Default::default() };

    let mut wd = build();
    let rd = wd.solve_dense(&cfg).unwrap();

    let mut wq = build();
    let mut params = Vec::new();
    RootProblem::serialize(&mut wq, &mut params);
    let mut solver =
        SparseFaer::new().with_policy(SchurPolicy::Auto {
            flop_margin: 0.0,
            obvious_flop_ratio: 0.0, // never short-circuit: force the comparison
        });
    let rq = wq.solve_with(&mut solver, &cfg).unwrap();

    let plan = solver.plan().unwrap();
    assert!(!plan.reduced, "the reduction should have been declined: {:?}", plan);
    assert_eq!(plan.eliminated_blocks, 0);
    assert_eq!(plan.kept_params, params.len(), "the full system stays");
    assert!(plan.fill_ratio.is_some(), "Auto must record its evidence");
    assert!(
        (rd.end_cost - rq.end_cost).abs() <= 1e-10 * (1.0 + rd.end_cost),
        "dense {} vs declined-schur {}",
        rd.end_cost,
        rq.end_cost
    );
}

/// The solve's own report: what the backend decided must reach the caller
/// through LmResult, including through the convenience entry points that
/// own their solver and would otherwise drop it.
#[test]
fn solve_reports_what_the_backend_did() {
    use arael::simple_lm::SolverReport;
    let cfg = LmConfig { max_iters: 60, ..Default::default() };

    let mut w = build();
    let r = w.solve_sparse(&cfg).unwrap(); // constructs and drops its own solver

    let Some(SolverReport::Schur(plan)) = r.solver else {
        panic!("solve_sparse must report what it did, got {:?}", r.solver);
    };
    assert!(plan.reduced);
    assert_eq!(plan.eliminated_blocks, N_POINTS + N_LINES);
    assert_eq!(plan.eliminated_params, 2 * N_POINTS + 3 * N_LINES);
    assert_eq!(plan.kept_params, 2 * (N_A + N_B));
    // No hint, so the Auto policy ran. This model's reduced system is tiny
    // (18 pose parameters), so the cheap test settles it and the ordering
    // comparison is skipped -- the plan says exactly that.
    let flop = plan.flop_ratio.expect("Auto records its cheap statistic");
    assert!(flop < 15.0, "expected an obvious reduction, got flop ratio {}", flop);
    assert_eq!(
        plan.fill_ratio, None,
        "the ordering comparison must be skipped when the reduction is obvious"
    );

    // and the backend that has nothing to say says nothing
    let mut wd = build();
    assert!(wd.solve_dense(&cfg).unwrap().solver.is_none());
}
