// The marginalize root keyword: `#[arael(root, marginalize(field))]` marks
// landmark-style fields (small blocks coupled to other parameters but never
// to each other). The macro generates RootProblem::marginalize_hint with the
// fields' parameter ranges -- computed by the same field walk serialize uses,
// so fixed params shift the ranges correctly -- and SparseFaer reads the hint
// off the model itself.
//
// A named set is marginalized when that pays, and ordered first in the
// factorization when it does not. SchurPolicy::Never pins the second
// behaviour, which is what the ordering cases below exercise.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{lm_solve, LmConfig, LmProblem, RootProblem, SchurPolicy, SparseFaer};

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

/// The generated hint covers exactly the landmarks' parameter range,
/// offset by the pose parameters serialized before them.
#[test]
fn hint_range_matches_layout() {
    let w = build(0.0);
    let hint = RootProblem::marginalize_hint(&w);
    assert_eq!(hint, vec![2 * N_POSES..2 * (N_POSES + N_LANDMARKS)]);
}

/// Fixed params shrink the serialized vector; the hint must follow the
/// serialize walk, not the field declaration.
#[test]
fn hint_range_respects_fixed_params() {
    let mut w = build(0.0);
    w.poses[0].x = Param::fixed(0.0);
    w.landmarks[0].y = Param::fixed(1.0);
    let hint = RootProblem::marginalize_hint(&w);
    let pose_params = 2 * N_POSES - 1;
    let lm_params = 2 * N_LANDMARKS - 1;
    assert_eq!(hint, vec![pose_params..pose_params + lm_params]);
}

/// solve_sparse (which auto-wires the hint) reaches the same optimum as
/// the dense solver on the identical problem.
#[test]
fn hinted_sparse_matches_dense() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };

    let mut wd = build(0.05);
    let rd = wd.solve_dense(&cfg).unwrap();
    assert!(rd.end_cost < 1e-14, "dense end_cost {}", rd.end_cost);

    let mut ws = build(0.05);
    let rs = ws.solve_sparse(&cfg).unwrap();
    assert!(rs.end_cost < 1e-14, "sparse end_cost {}", rs.end_cost);

    for j in 0..N_LANDMARKS {
        let (a, b) = (
            &wd.landmarks[j as usize],
            &ws.landmarks[j as usize],
        );
        assert!((a.x.value - b.x.value).abs() < 1e-8, "landmark {} x", j);
        assert!((a.y.value - b.y.value).abs() < 1e-8, "landmark {} y", j);
    }
}

/// The explicit solver-side path, including a deliberately wrong hint
/// (poses instead of landmarks) and an out-of-bounds range: the ordering
/// is used as given, and an elimination order affects speed, never the
/// solution -- every variant must reach the same optimum.
#[test]
fn explicit_hints_are_safe() {
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let n_pose = 2 * N_POSES;
    let n = 2 * (N_POSES + N_LANDMARKS);

    for ranges in [
        vec![n_pose..n],           // the right hint
        vec![0..n_pose],           // wrong: poses first
        vec![n_pose..n + 7],       // out of bounds: clamped
        vec![n_pose..n, n_pose..n], // duplicate: deduplicated
        vec![0..n],                // everything: natural order
    ] {
        let mut w = build(0.05);
        let mut params = Vec::new();
        RootProblem::serialize(&mut w, &mut params);
        let mut solver = SparseFaer::new().with_policy(SchurPolicy::Never);
        for r in ranges.clone() {
            solver = solver.with_marginalize(r);
        }
        let result = lm_solve(&params, &mut solver, &mut w, &cfg).unwrap();
        assert!(result.end_cost < 1e-14,
            "hint {:?}: end_cost {}", ranges, result.end_cost);
    }
}
