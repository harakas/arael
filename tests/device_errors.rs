//! A device failure ends a solve with `SolveFailureKind::Device`, never a
//! panic (feature `gpu-cuda`, `gpu-hip`, `gpu-metal` or `gpu-cpu`): every device operation of a
//! solve fails in turn (`arael_gpu::fault`), on each route, and each solve
//! returns the failure; its partial result, when it has one, holds
//! parameters and a cost that agree with the model as the failed solve
//! left it; and the next solve on the same device reaches the optimum.
//! In an f64 world on a device with f64, and in an f32 world on every
//! device.
//! The fault injector is process-wide, so this file is a binary of its
//! own and its tests take turns ([`one_at_a_time`]).
#![cfg(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu", gpu_metal))]

use arael::__gpu::fault;
use arael::model::{CrossBlock, MixedEulerAngleParam, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{GpuDevice, LmConfig, LmProblemInternals, LmSolver, LmStatus, RootProblem, SolveFailureKind, SolveResult, SparseFaer, SparseFaerOptions};
use arael::utils::Float;
use arael::vect::{vect2, vect3, vect3d};

#[arael::model]
#[arael(constraint(hb, {
    [(cam.t.x - cam.prior.x) * 0.1, (cam.t.y - cam.prior.y) * 0.1, (cam.t.z - cam.prior.z) * 0.1]
}))]
#[derive(Clone)]
pub struct Cam<T: Float> {
    pub t: Param<vect3<T>>,
    pub ea: MixedEulerAngleParam<T>,
    pub prior: vect3<T>,
    pub hb: SelfBlock<Cam<T>, T>,
}

#[arael::model]
#[derive(Clone)]
pub struct Pt<T: Float> {
    pub pos: Param<vect3<T>>,
    pub hb: SelfBlock<Pt<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    [100.0 * pc.x / pc.z - obs.xy.x, 100.0 * pc.y / pc.z - obs.xy.y]
}))]
#[derive(Clone)]
pub struct Obs<T: Float> {
    #[arael(ref = root.cams)]
    pub cam: Ref<Cam<T>>,
    #[arael(ref = root.pts)]
    pub pt: Ref<Pt<T>>,
    pub xy: vect2<T>,
    pub hb: CrossBlock<Cam<T>, Pt<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct World {
    pub cams: refs::Vec<Cam<f64>>,
    pub pts: refs::Vec<Pt<f64>>,
    pub obs: std::vec::Vec<Obs<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct WorldF {
    pub cams: refs::Vec<Cam<f32>>,
    pub pts: refs::Vec<Pt<f32>>,
    pub obs: std::vec::Vec<Obs<f32>>,
}

type Parts<T> = (refs::Vec<Cam<T>>, refs::Vec<Pt<T>>, std::vec::Vec<Obs<T>>);

/// Observations of true cameras and points, then the cameras turned and
/// moved and the points moved off the truth.
fn parts<T: Float>() -> Parts<T> {
    let mut cams = refs::Vec::new();
    let mut truth = Vec::new();
    for k in 0..4 {
        let t = vect3d::new(0.3 * k as f64, -0.1 * k as f64, 0.0);
        let ea = vect3d::new(0.02 * k as f64, -0.03, 0.05 * k as f64);
        truth.push((t, ea));
        cams.push(Cam {
            t: Param::new((t + vect3d::new(0.02, -0.01, 0.015)).cast()),
            ea: MixedEulerAngleParam::new((ea + vect3d::new(0.01, -0.008, 0.012)).cast()),
            prior: t.cast(),
            hb: SelfBlock::new(),
        });
    }
    let mut pts = refs::Vec::new();
    let mut ptrue = Vec::new();
    for p in 0..12 {
        let q = vect3d::new(0.4 * (p % 4) as f64 - 0.6, 0.3 * (p / 4) as f64 - 0.3, 4.0 + 0.2 * (p % 3) as f64);
        ptrue.push(q);
        pts.push(Pt { pos: Param::new((q + vect3d::new(0.03, -0.02, 0.05)).cast()), hb: SelfBlock::new() });
    }
    let mut obs = Vec::new();
    for (p, q) in ptrue.iter().enumerate() {
        for (k, (t, ea)) in truth.iter().enumerate() {
            if (p + k) % 4 == 3 {
                continue;
            }
            let r = arael::matrix::matrix3d::rotation_from_euler_angles(*ea);
            let pc = r * *q + *t;
            let xy = vect2::new(100.0 * pc.x / pc.z, 100.0 * pc.y / pc.z);
            obs.push(Obs { cam: cams.ref_at(k), pt: pts.ref_at(p), xy: xy.cast(), hb: CrossBlock::new() });
        }
    }
    (cams, pts, obs)
}

fn world() -> World {
    let (cams, pts, obs) = parts();
    World { cams, pts, obs }
}

fn world_f32() -> WorldF {
    let (cams, pts, obs) = parts();
    WorldF { cams, pts, obs }
}

#[cfg(feature = "gpu-cuda")]
fn gpu() -> GpuDevice {
    GpuDevice::cuda(0)
}
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
fn gpu() -> GpuDevice {
    GpuDevice::hip(0)
}
#[cfg(all(gpu_metal, not(any(feature = "gpu-cuda", feature = "gpu-hip"))))]
fn gpu() -> GpuDevice {
    GpuDevice::metal(0)
}
#[cfg(all(feature = "gpu-cpu", not(any(feature = "gpu-cuda", feature = "gpu-hip", gpu_metal))))]
fn gpu() -> GpuDevice {
    GpuDevice::cpu()
}

/// The routes a GPU solve takes: the reduced system with the device
/// assembly (the LM loop on the device), the reduced system assembled on
/// the CPU (a device damped solve per attempt), the whole system on the
/// device.
fn routes() -> Vec<(&'static str, SparseFaerOptions)> {
    use arael::simple_lm::{BlockSupernodalMode, EnvelopeMode, SchurPolicy};
    let reduced = || {
        SparseFaerOptions::default()
            .with_policy(SchurPolicy::Force)
            .with_envelope_schur(EnvelopeMode::Never)
            .with_block_supernodal(BlockSupernodalMode::Always)
            .with_gpu(Some(gpu()))
    };
    vec![
        ("reduced, device assembly", reduced().with_gpu_assembly(true)),
        ("reduced, CPU assembly", reduced().with_gpu_assembly(false)),
        (
            "whole system",
            SparseFaerOptions::default().with_policy(SchurPolicy::Never).with_block_supernodal(BlockSupernodalMode::Always).with_gpu(Some(gpu())),
        ),
    ]
}

/// The tests' turn at the process-wide fault injector.
fn one_at_a_time() -> std::sync::MutexGuard<'static, ()> {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    TURN.lock().unwrap_or_else(|p| p.into_inner())
}

/// Whether `gpu()` solves the f64 world; without f64 (Metal) it prints that
/// `what` in f64 is skipped (its f32 twin runs).
fn f64_world(what: &str) -> bool {
    let has = gpu().has_f64();
    if !has {
        eprintln!("{what}: the device has no f64, the f64 world skipped");
    }
    has
}

fn config<T: Float>() -> LmConfig<T> {
    LmConfig::<T> { max_iters: 30, ..Default::default() }
}

fn solve<T: Float, R: LmProblemInternals<T> + RootProblem<T>>(w: &mut R, options: &SparseFaerOptions) -> SolveResult<T>
where
    SparseFaer<T>: LmSolver<T>,
{
    let mut x = Vec::new();
    w.serialize(&mut x);
    arael::simple_lm::lm_solve(&x, &mut SparseFaer::<T>::from_options(options), w, &config())
}

/// Two costs of the world equal to its scalar's rounding: to `rel` of `b`,
/// plus `abs` of the start's cost (the optimum's cost is rounding of zero).
struct Near {
    rel: f64,
    abs: f64,
}

impl Near {
    fn f64() -> Self {
        Near { rel: 1e-9, abs: 1e-20 }
    }
    /// The f32 world's costs: summed in f32 on the CPU, wide on the device,
    /// and the device's reduction order not fixed.
    fn f32() -> Self {
        Near { rel: 1e-5, abs: 1e-7 }
    }
    fn holds(&self, a: f64, b: f64, start: f64) -> bool {
        (a - b).abs() <= self.rel * b.abs() + self.abs * start
    }
}

fn every_failure_is_reported<T: Float, R: LmProblemInternals<T> + RootProblem<T>>(make: impl Fn() -> R, near: Near, world: &str)
where
    SparseFaer<T>: LmSolver<T>,
{
    for (route, options) in routes() {
        let route = format!("{world}, {route}");
        let reference = solve(&mut make(), &options).unwrap_or_else(|f| panic!("{route}: {f}"));
        assert!(reference.accepted_iterations > 1, "{route}: the solve takes steps");
        let (start, end) = (reference.start_cost.to_f64().unwrap(), reference.end_cost.to_f64().unwrap());
        fault::arm(0);
        solve(&mut make(), &options).unwrap();
        let operations = fault::disarm();
        assert!(operations > 0, "{route}: device operations counted");
        let mut with_partial = 0;
        for k in 1..=operations {
            let mut w = make();
            fault::arm(k);
            let r = solve(&mut w, &options);
            fault::disarm();
            let failure = match r {
                Ok(_) => panic!("{route}: operation {k} of {operations} failed, the solve succeeded"),
                Err(f) => f,
            };
            let SolveFailureKind::Device(d) = &failure.kind else {
                panic!("{route}: operation {k}: {failure}");
            };
            assert!(d.message.contains(&format!("injected failure of operation {k} ")), "{route}: operation {k}: {failure}");
            if let Some(p) = &failure.partial {
                with_partial += 1;
                assert_eq!(p.status, LmStatus::Aborted, "{route}: operation {k}");
                let cost = w.calc_cost(&p.x).to_f64().unwrap();
                let partial = p.end_cost.to_f64().unwrap();
                assert!(near.holds(cost, partial, start), "{route}: operation {k}: the partial result's cost {partial} against {cost} at its parameters");
            }
            let next = solve(&mut make(), &options).unwrap_or_else(|f| panic!("{route}: after operation {k}: {f}"));
            let next = next.end_cost.to_f64().unwrap();
            assert!(near.holds(next, end, start), "{route}: after operation {k}: {next} against {end}");
        }
        // failures in the setup (no partial result) and in the loop (one)
        assert!(with_partial > 0 && with_partial < operations, "{route}: {with_partial} of {operations} failures with a partial result");
        eprintln!("{route}: {operations} device operations, each failure reported, {with_partial} with a partial result");
    }
}

#[test]
fn every_device_failure_in_a_solve_is_reported_and_the_next_solve_succeeds() {
    let _turn = one_at_a_time();
    if f64_world("every device failure") {
        every_failure_is_reported(world, Near::f64(), "f64");
    }
}

#[test]
fn every_device_failure_in_an_f32_solve_is_reported_and_the_next_solve_succeeds() {
    let _turn = one_at_a_time();
    every_failure_is_reported(world_f32, Near::f32(), "f32");
}

fn a_session_recovers<T: Float, R: LmProblemInternals<T> + RootProblem<T> + arael::model::Model>(make: impl Fn() -> R, near: Near, world: &str)
where
    SparseFaer<T>: LmSolver<T>,
{
    use arael::simple_lm::LmSession;
    let options = routes().remove(0).1;
    let reference = solve(&mut make(), &options).unwrap();
    let (start, end) = (reference.start_cost.to_f64().unwrap(), reference.end_cost.to_f64().unwrap());
    fault::arm(0);
    let mut session = LmSession::new(SparseFaer::<T>::from_options(&options));
    session.solve(&mut make(), &config()).unwrap();
    session.solve(&mut make(), &config()).unwrap();
    let operations = fault::disarm();
    for k in 1..=operations {
        let mut session = LmSession::new(SparseFaer::<T>::from_options(&options));
        fault::arm(k);
        // the first solve, or the warm second
        let failed = session.solve(&mut make(), &config()).is_err() || session.solve(&mut make(), &config()).is_err();
        fault::disarm();
        assert!(failed, "{world}: operation {k} of {operations}: no solve failed");
        let next = session.solve(&mut make(), &config()).unwrap_or_else(|f| panic!("{world}: after operation {k}: {f}"));
        let next = next.end_cost.to_f64().unwrap();
        assert!(near.holds(next, end, start), "{world}: after operation {k}: {next} against {end}");
    }
    eprintln!("{world} session: {operations} device operations over two solves, each failure recovered from");
}

/// A session keeps its device state between solves: a solve that fails at
/// any operation leaves the session able to solve again.
#[test]
fn a_session_solves_again_after_any_device_failure() {
    let _turn = one_at_a_time();
    if f64_world("a session after a device failure") {
        a_session_recovers(world, Near::f64(), "f64");
    }
}

#[test]
fn an_f32_session_solves_again_after_any_device_failure() {
    let _turn = one_at_a_time();
    a_session_recovers(world_f32, Near::f32(), "f32");
}
