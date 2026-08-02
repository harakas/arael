// The supernodal block Cholesky is an alternative factorization of whichever
// system the solve factors -- the reduced Schur system or the whole Hessian --
// run directly on the block form under a block-level ordering. Enabled with
// with_block_supernodal(true), it must reach the same optimum as the default
// scalar route, on both systems and at both precisions.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::simple_lm::{
    lm_solve, EnvelopeMode, LmConfig, RootProblem, SchurPolicy, SparseFaer,
};
use arael::refs::{self, Ref};

#[arael::model]
#[arael(constraint(hb, {
    [(pose.x - pose.ax) * 0.01, (pose.y - pose.ay) * 0.01]
}))]
struct Pose<T: arael::utils::Float> {
    x: Param<T>,
    y: Param<T>,
    ax: T,
    ay: T,
    hb: SelfBlock<Pose<T>, T>,
}

#[arael::model]
struct Landmark<T: arael::utils::Float> {
    x: Param<T>,
    y: Param<T>,
    hb: SelfBlock<Landmark<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, {
    [b.x - a.x - odo.dx, b.y - a.y - odo.dy]
}))]
struct Odo<T: arael::utils::Float> {
    #[arael(ref = root.poses)]
    a: Ref<Pose<T>>,
    #[arael(ref = root.poses)]
    b: Ref<Pose<T>>,
    dx: T,
    dy: T,
    hb: CrossBlock<Pose<T>, Pose<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, {
    [l.x - p.x - obs.dx, l.y - p.y - obs.dy]
}))]
struct Obs<T: arael::utils::Float> {
    #[arael(ref = root.poses)]
    p: Ref<Pose<T>>,
    #[arael(ref = root.landmarks)]
    l: Ref<Landmark<T>>,
    dx: T,
    dy: T,
    hb: CrossBlock<Pose<T>, Landmark<T>, T>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose<f64>>,
    landmarks: refs::Vec<Landmark<f64>>,
    odos: std::vec::Vec<Odo<f64>>,
    obs: std::vec::Vec<Obs<f64>>,
}

#[arael::model]
#[arael(root, f32)]
struct WorldF {
    poses: refs::Vec<Pose<f32>>,
    landmarks: refs::Vec<Landmark<f32>>,
    odos: std::vec::Vec<Odo<f32>>,
    obs: std::vec::Vec<Obs<f32>>,
}

const N_POSES: usize = 60;

fn pose_true(i: usize) -> (f64, f64) {
    (i as f64, 0.3 * i as f64)
}

/// The scene as plain numbers: poses, landmarks (one per window of `win`
/// consecutive poses, with its observations), odometry links.
struct Scene {
    poses: Vec<(f64, f64, f64, f64)>,
    landmarks: Vec<(f64, f64)>,
    obs: Vec<(usize, usize, f64, f64)>,
    odos: Vec<(usize, usize, f64, f64)>,
}

fn scene(off: f64, win: usize) -> Scene {
    let mut s = Scene { poses: Vec::new(), landmarks: Vec::new(), obs: Vec::new(), odos: Vec::new() };
    for i in 0..N_POSES {
        let (tx, ty) = pose_true(i);
        s.poses.push((tx + off, ty - off, tx, ty));
    }
    for w0 in 0..N_POSES.saturating_sub(win - 1) {
        let (lx, ly) = (w0 as f64 + 0.5, 2.0 + 0.2 * w0 as f64);
        let lm = s.landmarks.len();
        s.landmarks.push((lx - off, ly + off));
        for p in w0..w0 + win {
            let (px, py) = pose_true(p);
            s.obs.push((p, lm, lx - px, ly - py));
        }
    }
    for i in 1..N_POSES {
        let (ax, ay) = pose_true(i - 1);
        let (bx, by) = pose_true(i);
        s.odos.push((i - 1, i, bx - ax, by - ay));
    }
    s
}

fn build(off: f64, win: usize) -> World {
    let s = scene(off, win);
    let mut w = World {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    for &(x, y, ax, ay) in &s.poses {
        w.poses.push(Pose { x: Param::new(x), y: Param::new(y), ax, ay, hb: SelfBlock::new() });
    }
    for &(x, y) in &s.landmarks {
        w.landmarks.push(Landmark { x: Param::new(x), y: Param::new(y), hb: SelfBlock::new() });
    }
    for &(p, l, dx, dy) in &s.obs {
        w.obs.push(Obs {
            p: w.poses.ref_at(p),
            l: w.landmarks.ref_at(l as u32),
            dx,
            dy,
            hb: CrossBlock::new(),
        });
    }
    for &(a, b, dx, dy) in &s.odos {
        w.odos.push(Odo {
            a: w.poses.ref_at(a),
            b: w.poses.ref_at(b),
            dx,
            dy,
            hb: CrossBlock::new(),
        });
    }
    w
}

fn build_f32(off: f64, win: usize) -> WorldF {
    let s = scene(off, win);
    let mut w = WorldF {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    for &(x, y, ax, ay) in &s.poses {
        w.poses.push(Pose {
            x: Param::new(x as f32),
            y: Param::new(y as f32),
            ax: ax as f32,
            ay: ay as f32,
            hb: SelfBlock::new(),
        });
    }
    for &(x, y) in &s.landmarks {
        w.landmarks.push(Landmark {
            x: Param::new(x as f32),
            y: Param::new(y as f32),
            hb: SelfBlock::new(),
        });
    }
    for &(p, l, dx, dy) in &s.obs {
        w.obs.push(Obs {
            p: w.poses.ref_at(p),
            l: w.landmarks.ref_at(l as u32),
            dx: dx as f32,
            dy: dy as f32,
            hb: CrossBlock::new(),
        });
    }
    for &(a, b, dx, dy) in &s.odos {
        w.odos.push(Odo {
            a: w.poses.ref_at(a),
            b: w.poses.ref_at(b),
            dx: dx as f32,
            dy: dy as f32,
            hb: CrossBlock::new(),
        });
    }
    w
}

fn solve(solver: &mut SparseFaer<f64>, win: usize) -> (Vec<f64>, f64) {
    let cfg = LmConfig { max_iters: 60, ..Default::default() };
    let mut w = build(0.1, win);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let r = lm_solve(&params, solver, &mut w, &cfg).unwrap();
    let mut out = Vec::new();
    RootProblem::serialize(&mut w, &mut out);
    (out, r.end_cost)
}

/// The supernodal route on the reduced Schur system (envelope switched off
/// so the scalar route is what it replaces) reaches the default's optimum,
/// and the plan says it ran.
#[test]
fn supernodal_schur_route_matches_the_default() {
    let mut default = SparseFaer::new();
    let (x_def, c_def) = solve(&mut default, 2);

    let mut sn = SparseFaer::new()
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(true);
    let (x_sn, c_sn) = solve(&mut sn, 2);
    let plan = sn.plan().expect("a plan");
    assert!(plan.reduced, "the landmarks should marginalize");
    assert!(plan.block_supernodal, "the supernodal route should have run");
    assert!(!plan.envelope);

    assert!(c_def < 1e-12 && c_sn < 1e-12, "{} {}", c_def, c_sn);
    for (a, b) in std::iter::zip(&x_def, &x_sn) {
        assert!((a - b).abs() < 1e-9, "routes disagree: {} vs {}", a, b);
    }
}

/// The whole-Hessian flavor: no reduction, the block Hessian factored
/// directly under block-AMD.
#[test]
fn supernodal_whole_system_matches_the_scalar_route() {
    let mut scalar = SparseFaer::new().with_policy(SchurPolicy::Never);
    let (x_sc, c_sc) = solve(&mut scalar, 2);
    assert!(!scalar.plan().expect("a plan").block_supernodal);

    let mut sn = SparseFaer::new()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(true);
    let (x_sn, c_sn) = solve(&mut sn, 2);
    let plan = sn.plan().expect("a plan");
    assert!(!plan.reduced);
    assert!(plan.block_supernodal, "the supernodal route should have run");

    assert!(c_sc < 1e-12 && c_sn < 1e-12, "{} {}", c_sc, c_sn);
    for (a, b) in std::iter::zip(&x_sc, &x_sn) {
        assert!((a - b).abs() < 1e-9, "routes disagree: {} vs {}", a, b);
    }
}

/// SparseFaerOptions carries the flag into the solver it builds.
#[test]
fn the_options_struct_carries_block_supernodal() {
    use arael::simple_lm::SparseFaerOptions;
    let mut sn = SparseFaer::from_options(
        &SparseFaerOptions::auto()
            .with_envelope_schur(EnvelopeMode::Never)
            .with_block_supernodal(true),
    );
    let (_, c) = solve(&mut sn, 2);
    assert!(c < 1e-12, "end_cost {}", c);
    assert!(sn.plan().expect("a plan").block_supernodal);
}

/// A named marginalize set under SchurPolicy::Never orders those blocks
/// first in the whole-Hessian supernodal factorization (the block twin
/// of the scalar route's marginalize-first rule) -- and the answer must
/// not depend on the ordering.
#[test]
fn whole_system_orders_a_named_set_first() {
    let mut w = build(0.1, 2);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let n = params.len();
    let lm_start = 2 * N_POSES;

    let mut amd = SparseFaer::new()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(true);
    let (x_amd, c_amd) = solve(&mut amd, 2);

    let mut mf = SparseFaer::new()
        .with_policy(SchurPolicy::Never)
        .with_marginalize(lm_start..n)
        .with_block_supernodal(true);
    let (x_mf, c_mf) = solve(&mut mf, 2);
    let plan = mf.plan().expect("a plan");
    assert!(!plan.reduced, "SchurPolicy::Never must not reduce");
    assert!(plan.block_supernodal);

    assert!(c_amd < 1e-12 && c_mf < 1e-12, "{} {}", c_amd, c_mf);
    for (a, b) in std::iter::zip(&x_amd, &x_mf) {
        assert!((a - b).abs() < 1e-9, "orderings disagree: {} vs {}", a, b);
    }
}

/// Update batching is a user knob: disabling it must change nothing about
/// the answer, only how the updates are applied.
#[test]
fn batching_can_be_disabled_without_changing_the_answer() {
    let mut on = SparseFaer::new()
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(true);
    let (x_on, c_on) = solve(&mut on, 2);

    let mut off = SparseFaer::new()
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(true)
        .with_block_supernodal_batching(None);
    let (x_off, c_off) = solve(&mut off, 2);
    assert!(off.plan().expect("a plan").block_supernodal);

    assert!(c_on < 1e-12 && c_off < 1e-12, "{} {}", c_on, c_off);
    for (a, b) in std::iter::zip(&x_on, &x_off) {
        assert!((a - b).abs() < 1e-9, "batching changed the answer: {} vs {}", a, b);
    }
}

/// The f32 twin takes the same route and lands on the optimum at its own
/// precision.
#[test]
fn f32_supernodal_schur_route_solves() {
    let cfg = LmConfig::<f32> { max_iters: 60, ..Default::default() };
    let mut w = build_f32(0.1, 2);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let mut solver = arael::simple_lm::SparseFaerF32::new()
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(true);
    let r = lm_solve(&params, &mut solver, &mut w, &cfg).unwrap();
    assert!(solver.plan().expect("a plan").block_supernodal);
    assert!(r.end_cost < 1e-6, "end_cost {}", r.end_cost);
}
