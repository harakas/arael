// SchurPolicy::Auto has a cheap pre-filter that waves an obviously-good
// reduction through without pricing both routes, because the exact comparison
// needs a fill-reducing ordering of the whole matrix.
//
// The filter's flop ratio measures the reduced route against a floor no
// whole-system factorization can beat. That floor is a bound, not a cost: a
// route within the bar of it can still be several times worse than what the
// whole system actually factors for. So the filter also requires the reduction
// to remove a real share of the parameters -- its entire benefit is factoring
// something smaller, and one that keeps nearly everything cannot be obviously
// right whatever the ratio says.
//
// Measured on benchmarks/plane, whose planes are few next to its poses: the
// reduction kept 84% of the parameters and the filter waved it through, and the
// route it picked ran about twice the whole-system one at every scene size.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{lm_solve, LmConfig, RootProblem, SparseFaer};

#[arael::model]
#[arael(constraint(hb, {
    [(pose.x - pose.ax) * 0.01, (pose.y - pose.ay) * 0.01]
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
#[arael(root)]
struct World {
    poses: refs::Vec<Pose>,
    landmarks: refs::Vec<Landmark>,
    odos: std::vec::Vec<Odo>,
    obs: std::vec::Vec<Obs>,
}

const N_POSES: usize = 60;
const WIN: usize = 4; // poses each landmark is seen from

fn pose_true(i: usize) -> (f64, f64) {
    (i as f64, 0.3 * i as f64)
}

/// A trajectory with `n_lm` landmarks spread along it, each observed from a
/// window of `WIN` consecutive poses. `n_lm` sets how much of the system the
/// reduction removes, which is what the gate is being tested on.
fn build(n_lm: usize) -> World {
    let mut w = World {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    for i in 0..N_POSES {
        let (tx, ty) = pose_true(i);
        w.poses.push(Pose {
            x: Param::new(tx + 0.1),
            y: Param::new(ty - 0.1),
            ax: tx,
            ay: ty,
            hb: SelfBlock::new(),
        });
    }
    for k in 0..n_lm {
        // Spread the landmarks evenly so every one keeps a local window.
        let s = k * (N_POSES - WIN) / n_lm.max(1);
        let (lx, ly) = (s as f64 + 0.5, 2.0 + 0.2 * s as f64);
        w.landmarks.push(Landmark {
            x: Param::new(lx - 0.1),
            y: Param::new(ly + 0.1),
            hb: SelfBlock::new(),
        });
        for p in s..s + WIN {
            let (px, py) = pose_true(p);
            w.obs.push(Obs {
                p: w.poses.ref_at(p),
                l: w.landmarks.ref_at(k as u32),
                dx: lx - px,
                dy: ly - py,
                hb: CrossBlock::new(),
            });
        }
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
    w
}

/// Returns (the plan, the end cost).
fn solve(n_lm: usize) -> (arael::simple_lm::SchurPlan, f64) {
    let mut w = build(n_lm);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let mut solver = SparseFaer::<f64>::new();
    let cfg = LmConfig { max_iters: 60, ..Default::default() };
    let r = lm_solve(&params, &mut solver, &mut w, &cfg).unwrap();
    (solver.plan().expect("a plan"), r.end_cost)
}

/// Few landmarks: the reduction removes almost nothing, so the routes must be
/// priced exactly rather than waved through on the flop ratio.
#[test]
fn a_reduction_that_keeps_the_system_is_priced_exactly() {
    let (plan, cost) = solve(6);
    let kept = plan.kept_params as f64 / (plan.kept_params + plan.eliminated_params) as f64;
    assert!(kept > 0.5, "this scene should keep most of the parameters, kept {:.2}", kept);
    assert!(plan.fill_ratio.is_some(),
            "the routes must be priced exactly when the reduction keeps {:.0}% of \
             the system, not waved through", 100.0 * kept);
    assert!(cost.is_finite());
}

/// Many landmarks: the reduction removes most of the system, which is what the
/// pre-filter exists to wave through -- the fix must not have disabled it.
#[test]
fn a_reduction_that_removes_the_system_still_skips_the_pricing() {
    let (plan, cost) = solve(200);
    let kept = plan.kept_params as f64 / (plan.kept_params + plan.eliminated_params) as f64;
    assert!(kept < 0.5, "this scene should remove most of the parameters, kept {:.2}", kept);
    assert!(plan.reduced, "a reduction this favourable must be taken");
    assert!(plan.fill_ratio.is_none(),
            "pricing both routes is not worth it when the reduction removes \
             {:.0}% of the system", 100.0 * (1.0 - kept));
    assert!(cost.is_finite());
}

/// Whichever route the gate picks, the answer is the same.
#[test]
fn both_gate_outcomes_reach_the_same_optimum() {
    for n_lm in [6, 200] {
        let (_, auto) = solve(n_lm);
        let mut w = build(n_lm);
        let mut params = Vec::new();
        RootProblem::serialize(&mut w, &mut params);
        let mut forced = SparseFaer::<f64>::new()
            .with_policy(arael::simple_lm::SchurPolicy::Never);
        let cfg = LmConfig { max_iters: 60, ..Default::default() };
        let whole = lm_solve(&params, &mut forced, &mut w, &cfg).unwrap().end_cost;
        assert!((auto - whole).abs() < 1e-9 * whole.abs().max(1.0),
                "{} landmarks: auto {} vs whole-system {}", n_lm, auto, whole);
    }
}
