// The supernodal block Cholesky factorizes whichever system the solve
// factors -- the reduced Schur system or the whole Hessian -- directly on
// the block form under a block-level ordering. It is the default where the
// scalar route would otherwise run (BlockSupernodalMode::Auto, sequential
// solves); every mode must reach the same optimum as the scalar route, on
// both systems and at both precisions.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::simple_lm::{
    BlockSupernodalMode,
    lm_solve, EnvelopeMode, LmConfig, MixedPrecision, RootProblem, SchurPolicy, SparseFaer,
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

    // Force, not Auto: this is about the supernodal factorization of a
    // REDUCED system, and Auto prices the whole system under nested
    // dissection too -- on this scene it wins, leaving nothing reduced to
    // factorize.
    let mut sn = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Force)
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always));
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

/// The default is now the supernodal wherever the scalar factorization
/// would run on a sequential solve: a plain solver on this model takes
/// it without being asked.
#[test]
fn auto_default_takes_the_supernodal_route() {
    let mut auto = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default().with_policy(SchurPolicy::Never));
    let (_, c) = solve(&mut auto, 2);
    assert!(c < 1e-12, "end_cost {}", c);
    assert!(
        auto.plan().expect("a plan").block_supernodal,
        "Auto on a sequential solve must take the supernodal route",
    );
}

/// Auto takes the supernodal route at any thread count, and the threaded
/// run agrees with the sequential one: the dense kernels split their
/// output, not their reduction, and the assembly over the per-thread
/// mirrors sums each entity per thread and then across the threads, so
/// it matches to rounding rather than to the bit. (Meaningful only with
/// the rayon feature -- without it num_threads collapses to sequential.)
#[cfg(feature = "threads")]
#[test]
fn threads_do_not_change_the_supernodal_route_or_its_answer() {
    let solve = |num_threads| {
        let cfg = LmConfig { max_iters: 60, num_threads, ..Default::default() };
        let mut w = build(0.1, 2);
        let mut params = Vec::new();
        RootProblem::serialize(&mut w, &mut params);
        let mut solver = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default().with_policy(SchurPolicy::Never));
        let r = lm_solve(&params, &mut solver, &mut w, &cfg).unwrap();
        let took = solver.plan().expect("a plan").block_supernodal;
        (r.end_cost, r.x, took)
    };
    let (c1, x1, took1) = solve(1);
    let (c4, x4, took4) = solve(4);
    assert!(took1 && took4, "Auto takes the supernodal route at any thread count");
    assert!(c1 < 1e-12, "end_cost {}", c1);
    assert!((c1 - c4).abs() <= 1e-12, "same cost to rounding: {} vs {}", c1, c4);
    assert_eq!(x1.len(), x4.len());
    for (a, b) in x1.iter().zip(&x4) {
        assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs()), "same parameters to rounding: {} vs {}", a, b);
    }
}

/// The mixed route at several thread counts: the f32 factor, the threaded
/// solve and the threaded residual agree with the sequential run to
/// rounding, the plan still reporting the f32 factor and its refinement.
#[cfg(feature = "threads")]
#[test]
fn threads_do_not_change_the_mixed_precision_answer() {
    let solve = |num_threads| {
        let cfg = LmConfig { max_iters: 60, num_threads, ..Default::default() };
        let mut w = build(0.1, 2);
        let mut params = Vec::new();
        RootProblem::serialize(&mut w, &mut params);
        let mut solver = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
            .with_policy(SchurPolicy::Never)
            .with_block_supernodal(BlockSupernodalMode::Always)
            .with_mixed_precision(Some(MixedPrecision::default())));
        let r = lm_solve(&params, &mut solver, &mut w, &cfg).unwrap();
        let plan = solver.plan().expect("a plan");
        assert!(plan.block_supernodal && plan.f32_factor, "the f32 factor at {} threads", num_threads);
        assert!(plan.refinements.expect("refinement counted") > 0);
        (r.end_cost, r.x)
    };
    let (c1, x1) = solve(1);
    assert!(c1 < 1e-12, "end_cost {}", c1);
    for threads in [2usize, 4] {
        let (c, x) = solve(threads);
        assert!((c1 - c).abs() <= 1e-12, "same cost to rounding at {} threads: {} vs {}", threads, c1, c);
        assert_eq!(x1.len(), x.len());
        for (a, b) in x1.iter().zip(&x) {
            assert!((a - b).abs() <= 1e-6 * (1.0 + a.abs()),
                "same parameters to rounding at {} threads: {} vs {}", threads, a, b);
        }
    }
}

/// A DECLINED reduction (Auto weighed it and said no) must still land on
/// the supernodal for the whole system -- the decline path is a different
/// branch from "nothing to marginalize", and it used to fall through to
/// the scalar route. A zero flop margin forces the decline.
#[test]
fn a_declined_reduction_still_takes_the_supernodal() {
    let mut auto = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default().with_policy(SchurPolicy::Auto {
        flop_margin: 0.0,
        obvious_flop_ratio: 0.0,
    }));
    let (x_a, c_a) = solve(&mut auto, 2);
    let plan = auto.plan().expect("a plan");
    assert!(!plan.reduced, "a zero margin must decline the reduction");
    assert!(
        plan.block_supernodal,
        "the declined-reduction path must reach the supernodal route",
    );

    let mut scalar = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Auto { flop_margin: 0.0, obvious_flop_ratio: 0.0 })
        .with_block_supernodal(BlockSupernodalMode::Never));
    let (x_s, c_s) = solve(&mut scalar, 2);
    assert!(!scalar.plan().expect("a plan").block_supernodal);

    assert!(c_a < 1e-12 && c_s < 1e-12, "{} {}", c_a, c_s);
    for (a, b) in std::iter::zip(&x_a, &x_s) {
        assert!((a - b).abs() < 1e-9, "routes disagree: {} vs {}", a, b);
    }
}

/// The whole-Hessian flavor: no reduction, the block Hessian factored
/// directly. BlockSupernodalMode::Never restores the scalar route.
#[test]
fn supernodal_whole_system_matches_the_scalar_route() {
    let mut scalar = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(BlockSupernodalMode::Never));
    let (x_sc, c_sc) = solve(&mut scalar, 2);
    assert!(!scalar.plan().expect("a plan").block_supernodal);

    let mut sn = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(BlockSupernodalMode::Always));
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
            .with_block_supernodal(BlockSupernodalMode::Always),
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

    let mut amd = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(BlockSupernodalMode::Always));
    let (x_amd, c_amd) = solve(&mut amd, 2);

    let mut mf = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_marginalize(lm_start..n)
        .with_block_supernodal(BlockSupernodalMode::Always));
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
    let mut on = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always));
    let (x_on, c_on) = solve(&mut on, 2);

    let mut off = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always)
        .with_block_supernodal_batching(None));
    let (x_off, c_off) = solve(&mut off, 2);
    assert!(off.plan().expect("a plan").block_supernodal);

    assert!(c_on < 1e-12 && c_off < 1e-12, "{} {}", c_on, c_off);
    for (a, b) in std::iter::zip(&x_on, &x_off) {
        assert!((a - b).abs() < 1e-9, "batching changed the answer: {} vs {}", a, b);
    }
}

/// The memory-lean amalgamation is a user knob: it changes how much the
/// supernodes pad, never the answer.
#[test]
fn memory_lean_changes_nothing_about_the_answer() {
    let mut dflt = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always));
    let (x_d, c_d) = solve(&mut dflt, 2);

    let mut lean = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always)
        .with_block_supernodal_memory_lean(true));
    let (x_l, c_l) = solve(&mut lean, 2);
    assert!(lean.plan().expect("a plan").block_supernodal);

    assert!(c_d < 1e-12 && c_l < 1e-12, "{} {}", c_d, c_l);
    for (a, b) in std::iter::zip(&x_d, &x_l) {
        assert!((a - b).abs() < 1e-9, "lean changed the answer: {} vs {}", a, b);
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
    let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always));
    let r = lm_solve(&params, &mut solver, &mut w, &cfg).unwrap();
    assert!(solver.plan().expect("a plan").block_supernodal);
    assert!(r.end_cost < 1e-6, "end_cost {}", r.end_cost);
}

/// On an f32 solver the factor is single precision already; the setting
/// adds the refinement, in f64 over the f32 matrix, and the route still
/// lands on the optimum at the solver's precision with the plan reporting
/// the factor and the steps.
#[test]
fn f32_solver_with_mixed_precision_refines_and_solves() {
    let run = |mixed: Option<MixedPrecision>| {
        let cfg = LmConfig::<f32> { max_iters: 60, ..Default::default() };
        let mut w = build_f32(0.1, 2);
        let mut params = Vec::new();
        RootProblem::serialize(&mut w, &mut params);
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&arael::simple_lm::SparseFaerOptions::default()
            .with_envelope_schur(EnvelopeMode::Never)
            .with_block_supernodal(BlockSupernodalMode::Always)
            .with_mixed_precision(mixed));
        let r = lm_solve(&params, &mut solver, &mut w, &cfg).unwrap();
        let plan = solver.plan().expect("a plan");
        assert!(plan.block_supernodal);
        (r.end_cost, r.x, plan.f32_factor, plan.refinements)
    };
    let (c_plain, x_plain, f_plain, steps_plain) = run(None);
    assert!(!f_plain && steps_plain.is_none());
    let (c_mixed, x_mixed, f_mixed, steps_mixed) = run(Some(MixedPrecision::default()));
    assert!(f_mixed, "the plan reports the f32 factor on an f32 solver too");
    assert!(steps_mixed.expect("refinement counted") > 0);
    assert!(c_plain < 1e-6 && c_mixed < 1e-6, "{} {}", c_plain, c_mixed);
    assert_eq!(x_plain.len(), x_mixed.len());
    for (a, b) in x_plain.iter().zip(&x_mixed) {
        assert!((a - b).abs() <= 1e-3 * (1.0 + a.abs()), "refinement moved an f32 parameter: {} vs {}", a, b);
    }
}

/// The f32 factor with refinement, on the reduced Schur system: the f64
/// optimum, and the plan says the factor was single precision and how
/// many refinement steps the solve took.
#[test]
fn mixed_precision_schur_route_reaches_the_f64_optimum() {
    let mut default = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Force)
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always));
    let (x_64, c_64) = solve(&mut default, 2);
    assert!(!default.plan().expect("a plan").f32_factor);
    assert_eq!(default.plan().unwrap().refinements, None);

    let mut mixed = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Force)
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always)
        .with_mixed_precision(Some(MixedPrecision::default())));
    let (x_32, c_32) = solve(&mut mixed, 2);
    let plan = mixed.plan().expect("a plan");
    assert!(plan.reduced && plan.block_supernodal && plan.f32_factor);
    let steps = plan.refinements.expect("refinement counted");
    assert!(steps > 0, "an f32 solve needs refinement to reach 1e-10");
    assert_eq!(plan.refinement_shortfalls, Some(0), "well conditioned: every solve refines to tolerance");

    assert!(c_64 < 1e-12 && c_32 < 1e-12, "{} {}", c_64, c_32);
    for (a, b) in std::iter::zip(&x_64, &x_32) {
        assert!((a - b).abs() < 1e-9, "mixed precision changed the answer: {} vs {}", a, b);
    }
}

/// The same on the whole system (no reduction), against the scalar route.
#[test]
fn mixed_precision_whole_system_reaches_the_f64_optimum() {
    let mut scalar = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(BlockSupernodalMode::Never));
    let (x_s, c_s) = solve(&mut scalar, 2);

    let mut mixed = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(BlockSupernodalMode::Always)
        .with_mixed_precision(Some(MixedPrecision::default())));
    let (x_m, c_m) = solve(&mut mixed, 2);
    let plan = mixed.plan().expect("a plan");
    assert!(!plan.reduced && plan.block_supernodal && plan.f32_factor);
    assert!(plan.refinements.expect("refinement counted") > 0);

    assert!(c_s < 1e-12 && c_m < 1e-12, "{} {}", c_s, c_m);
    for (a, b) in std::iter::zip(&x_s, &x_m) {
        assert!((a - b).abs() < 1e-9, "mixed precision changed the answer: {} vs {}", a, b);
    }
}

/// With the step cap at zero the single-precision solution is taken as it
/// is: an inexact step the LM loop still converges on, with no refinement
/// counted. The scalar route ignores the setting altogether.
#[test]
fn mixed_precision_without_refinement_and_off_the_block_route() {
    let mp = MixedPrecision { max_refinements: 0, tol: 1e-10 };
    let mut raw = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(BlockSupernodalMode::Always)
        .with_mixed_precision(Some(mp)));
    let (_, c) = solve(&mut raw, 2);
    let plan = raw.plan().expect("a plan");
    assert!(plan.f32_factor);
    assert_eq!(plan.refinements, Some(0));
    assert_eq!(plan.refinement_shortfalls, Some(0));
    assert!(c < 1e-10, "an unrefined f32 step still converges: {}", c);

    let mut scalar = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(BlockSupernodalMode::Never)
        .with_mixed_precision(Some(MixedPrecision::default())));
    let (_, c) = solve(&mut scalar, 2);
    let plan = scalar.plan().expect("a plan");
    assert!(!plan.block_supernodal && !plan.f32_factor);
    assert_eq!(plan.refinements, None);
    assert!(c < 1e-12);
}

/// The GPUs compiled in (features `gpu-cuda`, `gpu-hip`), device 0, in f64
/// wide values and, with `df32`, in df32 too.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]
fn gpu_devices(df32: bool) -> Vec<arael::simple_lm::GpuDevice> {
    let mut base = Vec::new();
    #[cfg(feature = "gpu-cuda")]
    base.push(arael::simple_lm::GpuDevice::cuda(0));
    #[cfg(feature = "gpu-hip")]
    base.push(arael::simple_lm::GpuDevice::hip(0));
    let wides: &[arael::simple_lm::GpuWide] = if df32 { &[arael::simple_lm::GpuWide::F64, arael::simple_lm::GpuWide::Df32] } else { &[arael::simple_lm::GpuWide::F64] };
    base.iter().flat_map(|d| wides.iter().map(|&w| d.clone().with_wide(w))).collect()
}

/// The supernodal block route factorized and solved on a GPU: the f64
/// optimum on the whole system and on the reduced one, in f64 and with the
/// f32 device factor refined, the plan saying which, in either wide scalar.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]
#[test]
fn gpu_factor_reaches_the_f64_optimum() {
    let mut cpu = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(BlockSupernodalMode::Always));
    let (x_c, c_c) = solve(&mut cpu, 2);
    assert!(c_c < 1e-12);
    for device in gpu_devices(true) {
        for (policy, reduced) in [(SchurPolicy::Never, false), (SchurPolicy::Force, true)] {
            for mixed in [None, Some(MixedPrecision::default())] {
                let mut g = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
                    .with_policy(policy)
                    .with_envelope_schur(EnvelopeMode::Never)
                    .with_block_supernodal(BlockSupernodalMode::Always)
                    .with_mixed_precision(mixed)
                    .with_gpu(Some(device.clone())));
                let (x_g, c_g) = solve(&mut g, 2);
                let plan = g.plan().expect("a plan");
                assert!(plan.block_supernodal && plan.gpu_factor, "the plan reports the device factor");
                assert_eq!(plan.reduced, reduced);
                assert_eq!(plan.f32_factor, mixed.is_some());
                if mixed.is_some() {
                    assert!(plan.refinements.expect("refinement counted") > 0);
                } else {
                    assert_eq!(plan.refinements, None);
                }
                assert!(c_g < 1e-12, "{:?}: {} (reduced {}, mixed {})", device.backend, c_g, reduced, mixed.is_some());
                for (a, b) in std::iter::zip(&x_c, &x_g) {
                    assert!((a - b).abs() < 1e-9, "{:?}: the GPU changed the answer: {} vs {}", device.backend, a, b);
                }
            }
        }
    }
}

/// An f32 solver on the device route: the reduction is formed in the wide
/// scalar from the f32 Hessian on the device, the factor is f32 there, and
/// the solve lands on the CPU f32 route's optimum; in df32 nothing on the
/// device runs in f64.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]
#[test]
fn gpu_reduced_route_on_an_f32_solver() {
    let run = |gpu: Option<&arael::simple_lm::GpuDevice>| {
        let cfg = LmConfig::<f32> { max_iters: 60, ..Default::default() };
        let mut w = build_f32(0.1, 2);
        let mut params = Vec::new();
        RootProblem::serialize(&mut w, &mut params);
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&arael::simple_lm::SparseFaerOptions::default()
            .with_policy(SchurPolicy::Force)
            .with_envelope_schur(EnvelopeMode::Never)
            .with_block_supernodal(BlockSupernodalMode::Always)
            .with_gpu(gpu.cloned()));
        let r = lm_solve(&params, &mut solver, &mut w, &cfg).unwrap();
        let plan = solver.plan().expect("a plan");
        assert!(plan.reduced && plan.block_supernodal);
        assert_eq!(plan.gpu_factor, gpu.is_some());
        (r.end_cost, r.x)
    };
    let (c_cpu, x_cpu) = run(None);
    for device in gpu_devices(true) {
        let (c_gpu, x_gpu) = run(Some(&device));
        assert!(c_cpu < 1e-6 && c_gpu < 1e-6, "{:?}: {} {}", device.backend, c_cpu, c_gpu);
        for (a, b) in x_cpu.iter().zip(&x_gpu) {
            assert!((a - b).abs() <= 1e-3 * (1.0 + a.abs()), "{:?}: the device moved an f32 parameter: {} vs {}", device.backend, a, b);
        }
    }
}

/// A device whose runtime is not compiled in is a setup error.
#[cfg(all(feature = "gpu", not(all(feature = "gpu-cuda", feature = "gpu-hip"))))]
#[test]
fn a_runtime_not_compiled_in_is_unavailable() {
    #[cfg(not(feature = "gpu-hip"))]
    let device = arael::simple_lm::GpuDevice::hip(0);
    #[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
    let device = arael::simple_lm::GpuDevice::cuda(0);
    let cfg = LmConfig { max_iters: 60, ..Default::default() };
    let mut w = build(0.1, 2);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let mut solver = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Force)
        .with_envelope_schur(EnvelopeMode::Never)
        .with_block_supernodal(BlockSupernodalMode::Always)
        .with_gpu(Some(device)));
    let err = lm_solve(&params, &mut solver, &mut w, &cfg).err().expect("unavailable");
    assert!(
        matches!(err.kind, arael::simple_lm::SolveFailureKind::Setup(arael::simple_lm::SetupError::SolverUnavailable { solver: "gpu", .. })),
        "{:?}",
        err.kind
    );
}

/// Solvers built from clones of one `GpuDevice` share its device state:
/// after the first solves, solving again (the whole system, then the
/// reduced route of the same problem, then the whole system again) creates
/// no device buffer the first solves did not.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]
#[test]
fn gpu_device_clones_share_their_buffers() {
    for gpu in gpu_devices(false) {
        let fresh = || gpu_devices(false).into_iter().find(|d| d.backend == gpu.backend).unwrap();
        assert_eq!(gpu.buffers_created(), 0);
        let run = |policy: SchurPolicy| {
            let mut s = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
                .with_policy(policy)
                .with_envelope_schur(EnvelopeMode::Never)
                .with_block_supernodal(BlockSupernodalMode::Always)
                .with_mixed_precision(Some(MixedPrecision::default()))
                .with_gpu(Some(gpu.clone())));
            let (_, c) = solve(&mut s, 2);
            assert!(c < 1e-12 && s.plan().unwrap().gpu_factor);
        };
        run(SchurPolicy::Never);
        run(SchurPolicy::Force);
        let created = gpu.buffers_created();
        assert!(created > 0 && gpu.reserved_bytes() > 0);
        run(SchurPolicy::Never);
        run(SchurPolicy::Force);
        assert_eq!(gpu.buffers_created(), created, "{:?}: the repeated solves took every buffer from the shared state", gpu.backend);
        // a separately created device has a state of its own
        assert_eq!(fresh().buffers_created(), 0);
    }
}

/// The GPU takes the factorizing route only: with conjugate gradients on the
/// reduced system the solve stays on the CPU, and the plan says so.
#[cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]
#[test]
fn gpu_option_leaves_conjugate_gradients_on_the_cpu() {
    for device in gpu_devices(false) {
        let mut cg = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
            .with_policy(SchurPolicy::Force)
            .with_iterative_schur(Default::default())
            .with_gpu(Some(device)));
        let (_, c) = solve(&mut cg, 2);
        let plan = cg.plan().expect("a plan");
        assert!(plan.reduced && !plan.gpu_factor && plan.cg_iterations.is_some());
        assert!(c < 1e-8, "{}", c);
    }
}

/// Without the `gpu` feature the option is a setup error before anything
/// runs, like the other compiled-out backends.
#[cfg(not(feature = "gpu"))]
#[test]
fn gpu_option_without_the_feature_is_unavailable() {
    let cfg = LmConfig { max_iters: 60, ..Default::default() };
    let mut w = build(0.1, 2);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let mut solver = SparseFaer::from_options(&arael::simple_lm::SparseFaerOptions::default()
        .with_policy(SchurPolicy::Never)
        .with_block_supernodal(BlockSupernodalMode::Always)
        .with_gpu(Some(arael::simple_lm::GpuDevice::cuda(0))));
    let err = lm_solve(&params, &mut solver, &mut w, &cfg).err().expect("unavailable");
    assert!(
        matches!(err.kind, arael::simple_lm::SolveFailureKind::Setup(arael::simple_lm::SetupError::SolverUnavailable { solver: "gpu", .. })),
        "{:?}",
        err.kind
    );
}
