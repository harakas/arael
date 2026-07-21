// Compare the three SO(3) parameterizations on the aerobatics maneuver from
// euler_param.rs (nine barrel rolls + an Immelmann half-loop through the
// gimbal + a climbing corkscrew). Every free pose starts at identity, so the
// solver must sweep orientation space and cross the euler gimbal repeatedly.
// This is where the parameterization matters: SimpleEulerAngleParam optimizes
// the angles directly and degenerates at every near-lock passage, while
// EulerAngleParam and QuaternionParam keep an always-near-zero delta around a
// re-centered reference. Reports iteration counts per parameterization.
//
//   cargo test -r --test rot_param_compare -- --nocapture

use arael::model::{Model, SelfBlock, CrossBlock, EulerAngleParam, SimpleEulerAngleParam, QuaternionParam};
use arael::simple_lm::{self, LmConfig, LmProblem};
use arael::vect::vect3d;
use arael::matrix::matrix3d;
use arael::quatern::quaternd;
use arael::refs::{self, Ref};

// Same maneuver as euler_param.rs: 9x360 barrel rolls, a 180 half-loop
// (through the gimbal at 90), a 180 half-roll back, then 3 climbing barrel
// rolls (roll + steady pitch-up).
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

fn solve_once<P: LmProblem<f64>>(path: &mut P, params: &[f64]) -> arael::simple_lm::LmResult<f64> {
    simple_lm::solve_sparse_faer(params, path, &cfg())
}

// Largest per-pose recomposition error vs the flown trajectory (matrices, not
// euler triples, since poses sit at/beyond the gimbal).
fn max_pose_err(rots: &[matrix3d], truth: &[matrix3d]) -> f64 {
    rots.iter().zip(truth).map(|(m, t)| {
        (0..3).map(|r| (0..3).map(|c| (m[r][c] - t[r][c]).abs()).sum::<f64>()).sum::<f64>()
    }).fold(0.0, f64::max)
}

// ---------------------------------------------------------------------------
// SimpleEulerAngleParam (angles optimized directly)
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

fn run_simple() -> (usize, usize, f64, f64) {
    let (deltas, truth) = deltas_and_truth();
    let mut sky = SkyS { poses: refs::Vec::new(), pairs: std::vec::Vec::new(), isigma: 10.0 };
    sky.poses.push(PoseS { ea: SimpleEulerAngleParam::fixed(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    for _ in 1..truth.len() {
        sky.poses.push(PoseS { ea: SimpleEulerAngleParam::new(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    }
    for (i, d) in deltas.iter().enumerate() {
        sky.pairs.push(PairS { prev: sky.poses.ref_at(i), cur: sky.poses.ref_at(i as u32 + 1), delta: *d, hb: CrossBlock::new() });
    }
    let mut params = Vec::new();
    sky.serialize64(&mut params);
    let r = solve_once(&mut sky, &params);
    sky.deserialize64(&r.x);
    let rots: Vec<matrix3d> = (0..truth.len())
        .map(|i| matrix3d::rotation_from_euler_angles(sky.poses[i as usize].ea.value)).collect();
    (r.iterations, r.accepted_iterations, r.end_cost, max_pose_err(&rots, &truth))
}

// ---------------------------------------------------------------------------
// EulerAngleParam (euler-angle delta around a matrix reference)
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

fn run_euler() -> (usize, usize, f64, f64) {
    let (deltas, truth) = deltas_and_truth();
    let mut sky = SkyE { poses: refs::Vec::new(), pairs: std::vec::Vec::new(), isigma: 10.0 };
    sky.poses.push(PoseE { ea: EulerAngleParam::fixed(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    for _ in 1..truth.len() {
        sky.poses.push(PoseE { ea: EulerAngleParam::new(vect3d::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    }
    for (i, d) in deltas.iter().enumerate() {
        sky.pairs.push(PairE { prev: sky.poses.ref_at(i), cur: sky.poses.ref_at(i as u32 + 1), delta: *d, hb: CrossBlock::new() });
    }
    let mut params = Vec::new();
    sky.serialize64(&mut params);
    let r = solve_once(&mut sky, &params);
    sky.deserialize64(&r.x);
    let rots: Vec<matrix3d> = (0..truth.len())
        .map(|i| matrix3d::rotation_from_euler_angles(sky.poses[i as usize].ea.value)).collect();
    (r.iterations, r.accepted_iterations, r.end_cost, max_pose_err(&rots, &truth))
}

// ---------------------------------------------------------------------------
// QuaternionParam (rotation-vector delta around a quaternion reference)
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

fn run_quat() -> (usize, usize, f64, f64) {
    let (deltas, truth) = deltas_and_truth();
    let mut sky = SkyQ { poses: refs::Vec::new(), pairs: std::vec::Vec::new(), isigma: 10.0 };
    sky.poses.push(PoseQ { ea: QuaternionParam::fixed(quaternd::identity()), hb: SelfBlock::new() });
    for _ in 1..truth.len() {
        sky.poses.push(PoseQ { ea: QuaternionParam::new(quaternd::identity()), hb: SelfBlock::new() });
    }
    for (i, d) in deltas.iter().enumerate() {
        sky.pairs.push(PairQ { prev: sky.poses.ref_at(i), cur: sky.poses.ref_at(i as u32 + 1), delta: *d, hb: CrossBlock::new() });
    }
    let mut params = Vec::new();
    sky.serialize64(&mut params);
    let r = solve_once(&mut sky, &params);
    sky.deserialize64(&r.x);
    let rots: Vec<matrix3d> = (0..truth.len())
        .map(|i| sky.poses[i as usize].ea.value.rotation_matrix()).collect();
    (r.iterations, r.accepted_iterations, r.end_cost, max_pose_err(&rots, &truth))
}

#[test]
fn aerobatics_iteration_comparison() {
    let poses = maneuver_steps().len() + 1;
    let (s_it, s_ac, s_cost, s_err) = run_simple();
    let (e_it, e_ac, e_cost, e_err) = run_euler();
    let (q_it, q_ac, q_cost, q_err) = run_quat();

    println!("\naerobatics maneuver: {} poses, {} relative-rotation constraints", poses, poses - 1);
    println!("{:<28} {:>6} {:>9} {:>13} {:>12}", "parameterization", "iters", "accepted", "final cost", "max pose err");
    println!("{:<28} {:>6} {:>9} {:>13.3e} {:>12.2e}", "SimpleEulerAngleParam", s_it, s_ac, s_cost, s_err);
    println!("{:<28} {:>6} {:>9} {:>13.3e} {:>12.2e}", "EulerAngleParam", e_it, e_ac, e_cost, e_err);
    println!("{:<28} {:>6} {:>9} {:>13.3e} {:>12.2e}", "QuaternionParam", q_it, q_ac, q_cost, q_err);

    // All three must reach the flown trajectory.
    for (name, cost, err) in [("simple", s_cost, s_err), ("euler", e_cost, e_err), ("quat", q_cost, q_err)] {
        assert!(cost < 1e-9, "{name} did not converge, cost={cost}");
        assert!(err < 1e-4, "{name} orientation error {err}");
    }
}
