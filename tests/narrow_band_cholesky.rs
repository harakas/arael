// The narrow-band Cholesky is an alternative factorization of a BANDED reduced
// Schur system: same equations as faer's general sparse route, factorized in
// block form with fill confined to the band. Enabled with with_narrow_band(true), it
// must reach the same optimum as the default route.
//
// The scene is a trajectory whose landmarks are locally covisible -- each seen
// from a short window of consecutive poses -- so eliminating them leaves a
// banded pose system, which is the case the narrow-band route is for.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{
    lm_solve, EnvelopeMode, LmConfig, ReducedOrdering, RootProblem, SparseFaer,
};

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

fn pose_true(i: usize) -> (f64, f64) {
    (i as f64, 0.3 * i as f64)
}

/// A trajectory where each landmark is seen only from a window of `win`
/// consecutive poses, so the reduced pose system is banded with a half-width
/// set by `win` (not by the trajectory length).
fn build(off: f64, win: usize) -> World {
    let mut w = World {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
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
    // One landmark per window start, seen from poses [s, s+win).
    let mut lm = 0u32;
    for s in 0..N_POSES.saturating_sub(win - 1) {
        let (lx, ly) = (s as f64 + 0.5, 2.0 + 0.2 * s as f64);
        w.landmarks.push(Landmark {
            x: Param::new(lx - off),
            y: Param::new(ly + off),
            hb: SelfBlock::new(),
        });
        for p in s..s + win {
            let (px, py) = pose_true(p);
            w.obs.push(Obs {
                p: w.poses.ref_at(p),
                l: w.landmarks.ref_at(lm),
                dx: lx - px,
                dy: ly - py,
                hb: CrossBlock::new(),
            });
        }
        lm += 1;
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

/// The reduced system here is banded, so it is naturally ordered and the
/// envelope route takes it -- which is the default.
#[test]
fn envelope_route_is_the_default_when_naturally_ordered() {
    let mut solver = SparseFaer::new();
    let (_, cost) = solve(&mut solver, 2);
    let plan = solver.plan().expect("a plan");
    assert!(plan.reduced, "the landmarks should marginalize");
    assert_eq!(plan.ordering, Some(ReducedOrdering::NaturalBanded));
    assert!(plan.envelope, "the envelope route is on by default");
    assert!(cost < 1e-12, "end_cost {}", cost);
}

/// Auto prices the envelope against the ordered sparse factor and takes it
/// only with a margin. This scene is banded, which is what the envelope is
/// for, so Auto must take it -- and reach the same answer as forcing it.
#[test]
fn auto_takes_the_envelope_on_a_banded_reduction() {
    let mut auto = SparseFaer::new().with_envelope_schur(EnvelopeMode::Auto);
    let (x_auto, c_auto) = solve(&mut auto, 2);
    assert!(auto.plan().expect("a plan").envelope,
            "a banded reduction is what the envelope route is for");

    let mut always = SparseFaer::new().with_envelope_schur(EnvelopeMode::Always);
    let (x_always, c_always) = solve(&mut always, 2);
    assert!(c_auto < 1e-12 && c_always < 1e-12, "{} {}", c_auto, c_always);
    for (a, b) in std::iter::zip(&x_auto, &x_always) {
        assert!((a - b).abs() < 1e-9, "Auto and Always disagree: {} vs {}", a, b);
    }
}

/// Whatever Auto decides, the answer must not depend on it.
#[test]
fn every_envelope_mode_reaches_the_same_optimum() {
    let modes = [EnvelopeMode::Auto, EnvelopeMode::Always, EnvelopeMode::Never];
    let mut first: Option<Vec<f64>> = None;
    for m in modes {
        let mut s = SparseFaer::new().with_envelope_schur(m);
        let (x, c) = solve(&mut s, 2);
        assert!(c < 1e-12, "{:?} end_cost {}", m, c);
        match &first {
            None => first = Some(x),
            Some(f) => for (a, b) in std::iter::zip(f, &x) {
                assert!((a - b).abs() < 1e-9, "{:?} differs: {} vs {}", m, a, b);
            },
        }
    }
}

/// EnvelopeMode::Never puts the reduced system back on faer, and both routes
/// reach the same optimum.
#[test]
fn envelope_route_can_be_switched_off() {
    let mut off = SparseFaer::new().with_envelope_schur(EnvelopeMode::Never);
    let (x_faer, c_faer) = solve(&mut off, 2);
    let plan = off.plan().expect("a plan");
    assert!(plan.reduced);
    assert!(!plan.envelope, "EnvelopeMode::Never must not take it");

    let mut on = SparseFaer::new().with_envelope_schur(EnvelopeMode::Always);
    let (x_env, c_env) = solve(&mut on, 2);
    assert!(on.plan().expect("a plan").envelope);

    assert!(c_faer < 1e-12 && c_env < 1e-12, "{} {}", c_faer, c_env);
    for (a, b) in std::iter::zip(&x_faer, &x_env) {
        assert!((a - b).abs() < 1e-9, "routes disagree: {} vs {}", a, b);
    }
}

/// SparseFaerOptions carries the envelope fields into the solver it builds,
/// so SolverKind::Sparse can select the route.
#[test]
fn the_options_struct_carries_the_envelope_mode() {
    use arael::simple_lm::SparseFaerOptions;
    let mut on = SparseFaer::from_options(
        &SparseFaerOptions::auto().with_envelope_schur(EnvelopeMode::Always));
    let (x_on, c_on) = solve(&mut on, 2);
    assert!(on.plan().expect("a plan").envelope,
            "Always through the options struct must take the envelope");

    let mut off = SparseFaer::from_options(
        &SparseFaerOptions::auto().with_envelope_schur(EnvelopeMode::Never));
    let (x_off, c_off) = solve(&mut off, 2);
    assert!(!off.plan().expect("a plan").envelope,
            "Never through the options struct must not take it");

    assert!(c_on < 1e-12 && c_off < 1e-12, "{} {}", c_on, c_off);
    for (a, b) in std::iter::zip(&x_on, &x_off) {
        assert!((a - b).abs() < 1e-9, "routes disagree: {} vs {}", a, b);
    }
}

/// with_narrow_band takes the narrow-band Cholesky and reaches the same optimum.
#[test]
fn band_route_matches_faer_route() {
    for win in [2usize, 3, 5] {
        let (faer, c_faer) = solve(&mut SparseFaer::new(), win);

        let mut band_solver = SparseFaer::new().with_narrow_band(true);
        let (band, c_band) = solve(&mut band_solver, win);

        let plan = band_solver.plan().expect("a plan");
        assert!(plan.reduced, "win {}: expected a reduction", win);
        assert!(plan.envelope, "win {}: expected the narrow-band route to be taken", win);
        assert_eq!(plan.ordering, Some(ReducedOrdering::NaturalBanded));

        assert!(c_band < 1e-12, "win {}: band end_cost {}", win, c_band);
        assert!(
            (c_band - c_faer).abs() < 1e-12,
            "win {}: costs differ, band {} vs faer {}",
            win, c_band, c_faer,
        );
        for (i, (a, b)) in std::iter::zip(&band, &faer).enumerate() {
            assert!(
                (a - b).abs() < 1e-8,
                "win {}: param {} differs, band {} vs faer {}",
                win, i, a, b,
            );
        }
    }
}

/// A wider covisibility window widens the band (more than one block of
/// coupling); the narrow-band route must still agree with faer.
#[test]
fn wider_band_still_agrees() {
    let win = 5;
    let (faer, _) = solve(&mut SparseFaer::new(), win);
    let mut band_solver = SparseFaer::new().with_narrow_band(true);
    let (band, cost) = solve(&mut band_solver, win);
    let plan = band_solver.plan().expect("a plan");
    assert!(plan.envelope && plan.kept_bandwidth >= 2, "bandwidth {}", plan.kept_bandwidth);
    assert!(cost < 1e-12, "end_cost {}", cost);
    for (a, b) in std::iter::zip(&band, &faer) {
        assert!((a - b).abs() < 1e-8);
    }
}

/// A pose chain with no landmarks has nothing to marginalize, so the WHOLE
/// (banded) Hessian is factorized -- with_narrow_band routes that through the
/// block band Cholesky too, not only the reduced Schur system.
fn build_chain(off: f64) -> World {
    let mut w = World {
        poses: refs::Vec::new(),
        landmarks: refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
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

fn solve_chain(solver: &mut SparseFaer<f64>) -> (Vec<f64>, f64) {
    let cfg = LmConfig { max_iters: 60, ..Default::default() };
    let mut w = build_chain(0.1);
    let mut params = Vec::new();
    RootProblem::serialize(&mut w, &mut params);
    let r = lm_solve(&params, solver, &mut w, &cfg).unwrap();
    let mut out = Vec::new();
    RootProblem::serialize(&mut w, &mut out);
    (out, r.end_cost)
}

/// The report has to say which way the reduced system was factored. A reader
/// cannot infer it: a naturally-ordered system goes either way depending on
/// [`EnvelopeMode`], and the ordering line looks identical for both.
#[test]
fn the_report_says_how_the_reduced_system_was_factored() {
    let mut on = SparseFaer::new().with_envelope_schur(EnvelopeMode::Always);
    let (_, _) = solve(&mut on, 2);
    let mut off = SparseFaer::new().with_envelope_schur(EnvelopeMode::Never);
    let (_, _) = solve(&mut off, 2);

    let on_txt = format!("{:?}", on.plan().expect("a plan"));
    let off_txt = format!("{:?}", off.plan().expect("a plan"));
    assert!(on_txt.contains("envelope: true"), "{}", on_txt);
    assert!(off_txt.contains("envelope: false"), "{}", off_txt);
}

/// The two band routes are separate features that happen to share internals:
/// `with_narrow_band` factors the WHOLE Hessian, `EnvelopeMode` factors the
/// REDUCED system. Turning the second off must not touch the first.
#[test]
fn envelope_mode_does_not_reach_the_whole_system_band_route() {
    for mode in [EnvelopeMode::Auto, EnvelopeMode::Always, EnvelopeMode::Never] {
        let mut s = SparseFaer::new()
            .with_narrow_band(true)
            .with_envelope_schur(mode);
        let (_, cost) = solve_chain(&mut s);
        let plan = s.plan().expect("a plan");
        assert!(!plan.reduced, "a pose chain marginalizes nothing");
        assert!(plan.envelope,
                "{:?} must not disable the whole-system band route", mode);
        assert!(cost < 1e-12, "{:?} end_cost {}", mode, cost);
    }
}

#[test]
fn whole_system_band_route_matches_faer() {
    let (faer, c_faer) = solve_chain(&mut SparseFaer::new());

    let mut band_solver = SparseFaer::new().with_narrow_band(true);
    let (band, c_band) = solve_chain(&mut band_solver);

    let plan = band_solver.plan().expect("a plan");
    assert!(!plan.reduced, "a pose chain marginalizes nothing");
    assert!(plan.envelope, "the whole banded Hessian should take the band route");
    assert!(c_band < 1e-12, "band end_cost {}", c_band);
    assert!((c_band - c_faer).abs() < 1e-12, "band {} vs faer {}", c_band, c_faer);
    for (i, (a, b)) in std::iter::zip(&band, &faer).enumerate() {
        assert!((a - b).abs() < 1e-8, "param {}: band {} vs faer {}", i, a, b);
    }
}
