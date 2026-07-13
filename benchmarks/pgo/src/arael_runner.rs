// arael runners: identical model in f64 and f32 (separate structs --
// the precision is a compile-time property of the generated code).

use crate::g2o::{Dataset, PoseIn};
use crate::probe::PROBE_SUBROUNDS;
use arael::model::{Param, SelfBlock, CrossBlock};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect2f};

// ---------------------------------------------------------------- f64

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    [pose2.pos.x - pose2.prior.x,
     pose2.pos.y - pose2.prior.y,
     pose2.th - pose2.prior_th]
}))]
#[derive(Clone)]
struct Pose2 {
    pos: Param<vect2d>,
    th: Param<f64>,
    prior: vect2d,
    prior_th: f64,
    has_prior: bool,
    hb: SelfBlock<Pose2>,
}

#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(b.th).transpose()
        * (a.pos + matrix2sym::rotation(a.th) * edge.delta - b.pos);
    [local.x * edge.wt,
     local.y * edge.wt,
     rad_diff(a.th + edge.dth, b.th) * edge.wr]
}))]
#[derive(Clone)]
struct Edge {
    #[arael(ref = root.poses)]
    a: Ref<Pose2>,
    #[arael(ref = root.poses)]
    b: Ref<Pose2>,
    delta: vect2d,
    dth: f64,
    wt: f64,
    wr: f64,
    hb: CrossBlock<Pose2, Pose2>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
struct Graph {
    poses: refs::Vec<Pose2>,
    edges: std::vec::Vec<Edge>,
}

// ---------------------------------------------------------------- f32

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    [pose2f.pos.x - pose2f.prior.x,
     pose2f.pos.y - pose2f.prior.y,
     pose2f.th - pose2f.prior_th]
}))]
#[derive(Clone)]
struct Pose2F {
    pos: Param<vect2f>,
    th: Param<f32>,
    prior: vect2f,
    prior_th: f32,
    has_prior: bool,
    hb: SelfBlock<Pose2F, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let local = matrix2sym::rotation(b.th).transpose()
        * (a.pos + matrix2sym::rotation(a.th) * edgef.delta - b.pos);
    [local.x * edgef.wt,
     local.y * edgef.wt,
     rad_diff(a.th + edgef.dth, b.th) * edgef.wr]
}))]
#[derive(Clone)]
struct EdgeF {
    #[arael(ref = root.poses)]
    a: Ref<Pose2F>,
    #[arael(ref = root.poses)]
    b: Ref<Pose2F>,
    delta: vect2f,
    dth: f32,
    wt: f32,
    wr: f32,
    hb: CrossBlock<Pose2F, Pose2F, f32>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
struct GraphF {
    poses: refs::Vec<Pose2F>,
    edges: std::vec::Vec<EdgeF>,
}

// ---------------------------------------------------------------- runners

fn build_f64(ds: &Dataset) -> Graph {
    let mut g = Graph { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        g.poses.push(Pose2 {
            pos: Param::new(vect2d::new(p.x, p.y)),
            th: Param::new(p.th),
            prior: vect2d::new(p.x, p.y),
            prior_th: p.th,
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(Edge {
            a,
            b,
            delta: vect2d::new(e.dx, e.dy),
            dth: e.dth,
            wt: e.wt,
            wr: e.wr,
            hb: CrossBlock::new(),
        });
    }
    g
}

fn build_f32(ds: &Dataset) -> GraphF {
    let mut g = GraphF { poses: refs::Vec::new(), edges: std::vec::Vec::new() };
    for (i, p) in ds.poses.iter().enumerate() {
        g.poses.push(Pose2F {
            pos: Param::new(vect2f::new(p.x as f32, p.y as f32)),
            th: Param::new(p.th as f32),
            prior: vect2f::new(p.x as f32, p.y as f32),
            prior_th: p.th as f32,
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    for e in &ds.edges {
        let a = g.poses.ref_at(e.a);
        let b = g.poses.ref_at(e.b);
        g.edges.push(EdgeF {
            a,
            b,
            delta: vect2f::new(e.dx as f32, e.dy as f32),
            dth: e.dth as f32,
            wt: e.wt as f32,
            wr: e.wr as f32,
            hb: CrossBlock::new(),
        });
    }
    g
}

/// PGO_ORDERING=nd factorizes the whole system under a nested-dissection
/// ordering instead of AMD. On parking-garage -- a 3D pose graph dense enough
/// that AMD's ordering leaves faer no supernodes worth having -- it is worth a
/// lot; on the sparser graphs it is not. Default stays AMD.
pub(crate) fn ordering() -> arael::simple_lm::FaerOrdering {
    if std::env::var("PGO_ORDERING").as_deref() == Ok("nd") {
        arael::simple_lm::FaerOrdering::NestedDissection
    } else {
        arael::simple_lm::FaerOrdering::Auto
    }
}

/// TIMING=1 prints arael's internal breakdown for every solve it runs. That
/// includes the probes, so one round of one system emits several lines: the
/// discarded warmup plus PROBE_SUBROUNDS passes of the one- and two-iteration
/// probes, then the real solve.
///
/// Only the printing is gated. `gather_timing` stays on either way, so the
/// timing numbers do not depend on whether they are being looked at.
fn timing_enabled() -> bool {
    std::env::var("TIMING").is_ok()
}

fn print_timing<T>(r: &arael::simple_lm::LmResult<T>) {
    if !timing_enabled() {
        return;
    }
    if let Some(t) = &r.timing {
        eprintln!(
            "  [timing] total {:.1} ms = assembly {:.1} + linear solve {:.1} (first assembly {:.1}), {} iters",
            t.total.as_secs_f64() * 1e3,
            t.assembly.as_secs_f64() * 1e3,
            t.linear_solve.as_secs_f64() * 1e3,
            t.first_assembly.as_secs_f64() * 1e3,
            r.iterations,
        );
    }
}

pub fn solve_f64<P: arael::simple_lm::LmProblem<f64>>(
    params: &[f64],
    p: &mut P,
    cfg: &arael::simple_lm::LmConfig<f64>,
) -> arael::simple_lm::LmResult<f64> {
    let mut solver = arael::simple_lm::SparseFaer::new().with_ordering(ordering());
    let r = arael::simple_lm::lm_solve(params, &mut solver, p, cfg);
    print_timing(&r);
    r
}

/// Times a probe solve on throwaway copies of the model, fastest of
/// [`PROBE_SUBROUNDS`].
///
/// A discarded warmup runs first: the first solve in a process pays cold
/// allocator and cache costs the later ones do not (the symbolic factorization
/// alone runs a fifth slower), so timing the one- and two-iteration probes as
/// they come would charge that cost to the one-iteration probe alone and
/// inflate the difference between them.
///
/// The copies keep the probe from mutating the model: re-supplying the initial
/// parameter vector does not reset parametrization state held outside it, such
/// as the reference rotation an `EulerAngleParam` re-centres on after a step.
pub fn timed_f64<P: arael::simple_lm::LmProblem<f64> + Clone>(
    params: &[f64],
    p: &P,
    cfg: &arael::simple_lm::LmConfig<f64>,
) -> (f64, arael::simple_lm::LmResult<f64>) {
    let mut r = solve_f64(params, &mut p.clone(), cfg); // warmup, discarded
    let mut best = f64::INFINITY;
    for _ in 0..PROBE_SUBROUNDS {
        let t0 = std::time::Instant::now();
        r = solve_f64(params, &mut p.clone(), cfg);
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    (best, r)
}

/// See [`timed_f64`].
pub fn timed_f32<P: arael::simple_lm::LmProblem<f32> + Clone>(
    params: &[f32],
    p: &P,
    cfg: &arael::simple_lm::LmConfig<f32>,
) -> (f64, arael::simple_lm::LmResult<f32>) {
    let mut r = solve_f32(params, &mut p.clone(), cfg); // warmup, discarded
    let mut best = f64::INFINITY;
    for _ in 0..PROBE_SUBROUNDS {
        let t0 = std::time::Instant::now();
        r = solve_f32(params, &mut p.clone(), cfg);
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
    }
    (best, r)
}

pub fn solve_f32<P: arael::simple_lm::LmProblem<f32>>(
    params: &[f32],
    p: &mut P,
    cfg: &arael::simple_lm::LmConfig<f32>,
) -> arael::simple_lm::LmResult<f32> {
    let mut solver = arael::simple_lm::SparseFaerF32::new().with_ordering(ordering());
    let r = arael::simple_lm::lm_solve(params, &mut solver, p, cfg);
    print_timing(&r);
    r
}

pub struct RunOut {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    pub iterations: usize,
    /// Accepted (cost-decreasing) steps; `iterations` additionally counts
    /// damping retries. Other systems report only their outer iteration
    /// count, so their tables carry a single number.
    pub accepted: usize,
    /// Wall clock of a fresh TWO-iteration solve. One complete iteration is
    /// this minus `first_iter_ms` -- both pay the same setup, so it cancels --
    /// but the subtraction must happen on the MINIMA over rounds, not on a
    /// single pair: differencing two noisy cold runs can even come out
    /// negative. `None` if the second step was rejected.
    pub two_iter_ms: Option<f64>,
    pub poses: Vec<PoseIn>,
}

// Termination: same criterion class and thresholds as the tiny-solver and
// GTSAM defaults (stop when a step improves the cost by less than 1e-5
// absolute or 1e-5 relative). patience = 1 so ONE small step terminates,
// matching how both other systems check it.
// Initial damping, problem-appropriate for well-initialized pose graphs
// (the LmConfig docs recommend small initial_lambda for these); see the
// README's initial-damping policy. Env-overridable for experiments.
pub(crate) fn lambda0() -> f64 {
    std::env::var("ARAEL_LAMBDA0").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-8)
}

pub(crate) fn cfg64(max_iters: usize) -> arael::simple_lm::LmConfig<f64> {
    cfg64_with_lambda(max_iters, lambda0())
}

/// PGO_DRIVER=nielsen swaps the fixed damping schedule for the gain-ratio
/// driver: lambda then follows how well the quadratic model predicted the step
/// actually taken, instead of a fixed up/down ladder. It changes the iteration
/// count, not the cost per iteration -- so it moves the total, and the
/// per-iteration comparison stays honest either way.
pub(crate) fn nielsen() -> bool {
    std::env::var("PGO_DRIVER").as_deref() == Ok("nielsen")
}

pub(crate) fn cfg64_with_lambda(max_iters: usize, initial_lambda: f64) -> arael::simple_lm::LmConfig<f64> {
    let cfg = arael::simple_lm::LmConfig {
        verbose: std::env::var("VERBOSE").is_ok(),
        gather_timing: true,
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        // Stop as soon as the tolerance test says converged, the way Ceres and
        // g2o do. The library defaults (min_iters 5, patience 3) put a floor
        // under the iteration count, which on a problem that converges in four
        // iterations would time a fifth that improves the cost by 4e-14.
        patience: 1,
        min_iters: 1,
        max_iters,
        initial_lambda,
        ..Default::default()
    };
    if nielsen() {
        cfg.with_driver(arael::simple_lm::NielsenLambdaDriver::default())
    } else {
        cfg
    }
}

pub(crate) fn cfg32(max_iters: usize) -> arael::simple_lm::LmConfig<f32> {
    cfg32_with_lambda(max_iters, lambda0() as f32)
}

pub(crate) fn cfg32_with_lambda(max_iters: usize, initial_lambda: f32) -> arael::simple_lm::LmConfig<f32> {
    let cfg = arael::simple_lm::LmConfig {
        verbose: std::env::var("VERBOSE").is_ok(),
        gather_timing: true,
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        // Stop as soon as the tolerance test says converged, the way Ceres and
        // g2o do. The library defaults (min_iters 5, patience 3) put a floor
        // under the iteration count, which on a problem that converges in four
        // iterations would time a fifth that improves the cost by 4e-14.
        patience: 1,
        min_iters: 1,
        max_iters,
        initial_lambda,
        ..Default::default()
    };
    if nielsen() {
        cfg.with_driver(arael::simple_lm::NielsenLambdaDriver::default())
    } else {
        cfg
    }
}

pub fn run_f64(ds: &Dataset) -> RunOut {
    let mut g = build_f64(ds);
    let mut params: Vec<f64> = Vec::new();
    g.serialize64(&mut params);

    // First-iteration time: a fresh solve capped at one iteration
    // (setup + first assembly + symbolic + numeric factorization + step).
    let (first_ms, first) = timed_f64(&params, &g, &cfg64(1));
    let first_iter_ms = crate::probe::first_iter_ms(
        first_ms, first.iterations, first.accepted_iterations);

    // ... and a fresh two-iteration solve. The difference is one complete
    // iteration with the setup cancelled out; see second_iter_ms.
    let (two_ms, two) = timed_f64(&params, &g, &cfg64(2));
    let two_iter_ms = crate::probe::two_iter_ms(two_ms, first_iter_ms, two.accepted_iterations);

    let t0 = std::time::Instant::now();
    let result = solve_f64(&params, &mut g, &cfg64(100));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize64(&result.x);
    let poses = g.poses.iter()
        .map(|p| PoseIn { x: p.pos.value.x, y: p.pos.value.y, th: p.th.value })
        .collect();
    RunOut {
        solve_ms, first_iter_ms,
        iterations: result.iterations,
        accepted: result.accepted_iterations,
        two_iter_ms,
        poses,
    }
}

pub fn run_f32(ds: &Dataset) -> RunOut {
    let mut g = build_f32(ds);
    let mut params: Vec<f32> = Vec::new();
    g.serialize32(&mut params);

    let (first_ms, first) = timed_f32(&params, &g, &cfg32(1));
    let first_iter_ms = crate::probe::first_iter_ms(
        first_ms, first.iterations, first.accepted_iterations);

    // ... and a fresh two-iteration solve. The difference is one complete
    // iteration with the setup cancelled out; see second_iter_ms.
    let (two_ms, two) = timed_f32(&params, &g, &cfg32(2));
    let two_iter_ms = crate::probe::two_iter_ms(two_ms, first_iter_ms, two.accepted_iterations);

    let t0 = std::time::Instant::now();
    let result = solve_f32(&params, &mut g, &cfg32(100));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    g.deserialize32(&result.x);
    let poses = g.poses.iter()
        .map(|p| PoseIn { x: p.pos.value.x as f64, y: p.pos.value.y as f64, th: p.th.value as f64 })
        .collect();
    RunOut {
        solve_ms, first_iter_ms,
        iterations: result.iterations,
        accepted: result.accepted_iterations,
        two_iter_ms,
        poses,
    }
}
