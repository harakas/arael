// SO(3) parameterization conditioning benchmark: the aerobatics maneuver.
//
// A pose chain flies nine 360-degree barrel rolls, an Immelmann half-loop
// (pitch through the gimbal at 90 and on to 180), a half-roll back to upright,
// and three climbing barrel rolls -- a corkscrew that sweeps orientation space
// and crosses the euler gimbal repeatedly. Consecutive poses are tied by a
// relative-rotation constraint; the first pose is fixed at identity and EVERY
// free pose starts at identity, maximally far from the flown trajectory. The
// solver must rotate them across many gimbal crossings and accumulate
// thousands of degrees of rotation.
//
// This is a worst case, chosen to expose how the SO(3) parameterization
// conditions the problem -- it is NOT a typical solve (see README). The three
// parameterizations reach the same trajectory; what differs is the LM
// iteration count, and with it the total time.
//
//   cargo run --release          # median of 20 rounds
//   ROUNDS=50 cargo run --release

use arael::model::{Model, SelfBlock, CrossBlock, EulerAngleParam, SimpleEulerAngleParam, QuaternionParam};
use arael::simple_lm::{self, LmConfig, LmProblem};
use arael::vect::vect3d;
use arael::matrix::matrix3d;
use arael::quatern::quaternd;
use arael::refs::{self, Ref};
use std::time::Instant;

// 9x360 barrel rolls, a 180 half-loop (through the gimbal at 90), a 180
// half-roll back, then 3 climbing barrel rolls (roll + steady pitch-up).
fn maneuver_steps() -> Vec<(f64, f64)> {
    let step = 15.0_f64.to_radians();
    let climb = 4.0_f64.to_radians();
    let mut out = Vec::new();
    for _ in 0..(9 * 24) { out.push((step, 0.0)); }
    for _ in 0..12 { out.push((0.0, step)); }
    for _ in 0..12 { out.push((step, 0.0)); }
    for _ in 0..(3 * 24) { out.push((step, climb)); }
    out
}

fn deltas_and_truth() -> (Vec<matrix3d>, Vec<matrix3d>) {
    let deltas: Vec<matrix3d> = maneuver_steps().iter()
        .map(|&(r, p)| matrix3d::rotation_from_euler_angles(vect3d::new(r, p, 0.0)))
        .collect();
    let mut truth: Vec<matrix3d> = vec![matrix3d::identity()];
    for d in &deltas { truth.push(*truth.last().unwrap() * *d); }
    (deltas, truth)
}

fn cfg() -> LmConfig<f64> {
    LmConfig { max_iters: 500, ..Default::default() }
}

fn solve_timed<P: LmProblem<f64>>(path: &mut P, params: &[f64]) -> (usize, usize, f64, f64) {
    let t0 = Instant::now();
    let r = simple_lm::solve_sparse(params, path, &cfg());
    let ms = t0.elapsed().as_secs_f64() * 1e3;
    (r.iterations, r.accepted_iterations, r.end_cost, ms)
}

// ---------------------------------------------------------------------------
// SimpleEulerAngleParam (angles optimized directly, no re-centering)
// ---------------------------------------------------------------------------
#[arael::model]
struct PoseS { ea: SimpleEulerAngleParam<f64>, hb: SelfBlock<PoseS> }

#[arael::model]
#[arael(constraint(hb, {
    let d = prev.ea.rotation_matrix().transpose() * cur.ea.rotation_matrix() - pairs.delta;
    [d[0][0] * skys.isigma, d[0][1] * skys.isigma, d[0][2] * skys.isigma,
     d[1][0] * skys.isigma, d[1][1] * skys.isigma, d[1][2] * skys.isigma,
     d[2][0] * skys.isigma, d[2][1] * skys.isigma, d[2][2] * skys.isigma]
}))]
struct PairS {
    #[arael(ref = root.poses)] prev: Ref<PoseS>,
    #[arael(ref = root.poses)] cur: Ref<PoseS>,
    delta: matrix3d,
    hb: CrossBlock<PoseS, PoseS>,
}

#[arael::model]
#[arael(root)]
struct SkyS { poses: refs::Vec<PoseS>, pairs: std::vec::Vec<PairS>, isigma: f64 }

fn build_s(deltas: &[matrix3d], n: usize) -> (SkyS, Vec<f64>) {
    let mut sky = SkyS { poses: refs::Vec::new(), pairs: std::vec::Vec::new(), isigma: 10.0 };
    sky.poses.push(PoseS { ea: SimpleEulerAngleParam::fixed(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    for _ in 1..n { sky.poses.push(PoseS { ea: SimpleEulerAngleParam::new(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() }); }
    for (i, d) in deltas.iter().enumerate() {
        sky.pairs.push(PairS { prev: sky.poses.ref_at(i), cur: sky.poses.ref_at(i + 1), delta: *d, hb: CrossBlock::new() });
    }
    let mut params = Vec::new();
    sky.serialize64(&mut params);
    (sky, params)
}

// ---------------------------------------------------------------------------
// EulerAngleParam (euler-angle delta around a re-centered matrix reference)
// ---------------------------------------------------------------------------
#[arael::model]
struct PoseE { ea: EulerAngleParam<f64>, hb: SelfBlock<PoseE> }

#[arael::model]
#[arael(constraint(hb, {
    let d = prev.ea.rotation_matrix().transpose() * cur.ea.rotation_matrix() - paire.delta;
    [d[0][0] * skye.isigma, d[0][1] * skye.isigma, d[0][2] * skye.isigma,
     d[1][0] * skye.isigma, d[1][1] * skye.isigma, d[1][2] * skye.isigma,
     d[2][0] * skye.isigma, d[2][1] * skye.isigma, d[2][2] * skye.isigma]
}))]
struct PairE {
    #[arael(ref = root.poses)] prev: Ref<PoseE>,
    #[arael(ref = root.poses)] cur: Ref<PoseE>,
    delta: matrix3d,
    hb: CrossBlock<PoseE, PoseE>,
}

#[arael::model]
#[arael(root)]
struct SkyE { poses: refs::Vec<PoseE>, pairs: std::vec::Vec<PairE>, isigma: f64 }

fn build_e(deltas: &[matrix3d], n: usize) -> (SkyE, Vec<f64>) {
    let mut sky = SkyE { poses: refs::Vec::new(), pairs: std::vec::Vec::new(), isigma: 10.0 };
    sky.poses.push(PoseE { ea: EulerAngleParam::fixed(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    for _ in 1..n { sky.poses.push(PoseE { ea: EulerAngleParam::new(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() }); }
    for (i, d) in deltas.iter().enumerate() {
        sky.pairs.push(PairE { prev: sky.poses.ref_at(i), cur: sky.poses.ref_at(i + 1), delta: *d, hb: CrossBlock::new() });
    }
    let mut params = Vec::new();
    sky.serialize64(&mut params);
    (sky, params)
}

// ---------------------------------------------------------------------------
// QuaternionParam (rotation-vector delta around a re-centered quaternion)
// ---------------------------------------------------------------------------
#[arael::model]
struct PoseQ { ea: QuaternionParam<f64>, hb: SelfBlock<PoseQ> }

#[arael::model]
#[arael(constraint(hb, {
    let d = prev.ea.rotation_matrix().transpose() * cur.ea.rotation_matrix() - pairq.delta;
    [d[0][0] * skyq.isigma, d[0][1] * skyq.isigma, d[0][2] * skyq.isigma,
     d[1][0] * skyq.isigma, d[1][1] * skyq.isigma, d[1][2] * skyq.isigma,
     d[2][0] * skyq.isigma, d[2][1] * skyq.isigma, d[2][2] * skyq.isigma]
}))]
struct PairQ {
    #[arael(ref = root.poses)] prev: Ref<PoseQ>,
    #[arael(ref = root.poses)] cur: Ref<PoseQ>,
    delta: matrix3d,
    hb: CrossBlock<PoseQ, PoseQ>,
}

#[arael::model]
#[arael(root)]
struct SkyQ { poses: refs::Vec<PoseQ>, pairs: std::vec::Vec<PairQ>, isigma: f64 }

fn build_q(deltas: &[matrix3d], n: usize) -> (SkyQ, Vec<f64>) {
    let mut sky = SkyQ { poses: refs::Vec::new(), pairs: std::vec::Vec::new(), isigma: 10.0 };
    sky.poses.push(PoseQ { ea: QuaternionParam::fixed(quaternd::identity()), hb: SelfBlock::new() });
    for _ in 1..n { sky.poses.push(PoseQ { ea: QuaternionParam::new(quaternd::identity()), hb: SelfBlock::new() }); }
    for (i, d) in deltas.iter().enumerate() {
        sky.pairs.push(PairQ { prev: sky.poses.ref_at(i), cur: sky.poses.ref_at(i + 1), delta: *d, hb: CrossBlock::new() });
    }
    let mut params = Vec::new();
    sky.serialize64(&mut params);
    (sky, params)
}

// Accumulates one parameterization's results across the timed rounds.
struct Row {
    name: &'static str,
    iters: usize,
    accepted: usize,
    cost: f64,
    ms: Vec<f64>,
}

impl Row {
    fn new(name: &'static str) -> Self { Row { name, iters: 0, accepted: 0, cost: 0.0, ms: Vec::new() } }
    fn record(&mut self, r: (usize, usize, f64, f64)) {
        self.iters = r.0; self.accepted = r.1; self.cost = r.2; self.ms.push(r.3);
    }
    fn median_ms(&self) -> f64 {
        let mut v = self.ms.clone();
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    }
}

fn main() {
    let rounds = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let (deltas, truth) = deltas_and_truth();
    let n = truth.len();
    println!("aerobatics maneuver: {} poses, {} relative-rotation constraints", n, deltas.len());

    // Warm up all three.
    { let (mut p, pp) = build_s(&deltas, n); solve_timed(&mut p, &pp); }
    { let (mut p, pp) = build_e(&deltas, n); solve_timed(&mut p, &pp); }
    { let (mut p, pp) = build_q(&deltas, n); solve_timed(&mut p, &pp); }

    let mut se = Row::new("SimpleEulerAngleParam");
    let mut eu = Row::new("EulerAngleParam");
    let mut qu = Row::new("QuaternionParam");
    for _ in 0..rounds {
        { let (mut p, pp) = build_s(&deltas, n); se.record(solve_timed(&mut p, &pp)); }
        { let (mut p, pp) = build_e(&deltas, n); eu.record(solve_timed(&mut p, &pp)); }
        { let (mut p, pp) = build_q(&deltas, n); qu.record(solve_timed(&mut p, &pp)); }
    }

    println!("\n{:<28} {:>6} {:>9} {:>12} {:>9} {:>12}",
        "parameterization", "iters", "accepted", "total ms", "ms/iter", "final cost");
    for r in [&se, &eu, &qu] {
        let ms = r.median_ms();
        println!("{:<28} {:>6} {:>9} {:>12.3} {:>9.3} {:>12.2e}",
            r.name, r.iters, r.accepted, ms, ms / r.iters as f64, r.cost);
    }
}
