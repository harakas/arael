//! A device failure ends a solve with `SolveFailureKind::Device`, never a
//! panic (feature `gpu-cuda` or `gpu-hip`): every device operation of a
//! solve fails in turn (`arael_gpu::fault`), on each route, and each solve
//! returns the failure; its partial result, when it has one, holds
//! parameters and a cost that agree with the model as the failed solve
//! left it; and the next solve on the same device reaches the optimum.
//! The fault injector is process-wide, so this file is a binary of its
//! own and its tests take turns ([`one_at_a_time`]).
#![cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]

use arael::__gpu::fault;
use arael::model::{CrossBlock, MixedEulerAngleParam, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{GpuDevice, LmConfig, LmProblem, LmStatus, RootProblem, SolveFailureKind, SolveResult, SparseFaer, SparseFaerOptions};
use arael::vect::{vect2, vect3, vect3d};

#[arael::model]
#[arael(constraint(hb, {
    [(cam.t.x - cam.prior.x) * 0.1, (cam.t.y - cam.prior.y) * 0.1, (cam.t.z - cam.prior.z) * 0.1]
}))]
#[derive(Clone)]
pub struct Cam {
    pub t: Param<vect3<f64>>,
    pub ea: MixedEulerAngleParam<f64>,
    pub prior: vect3<f64>,
    pub hb: SelfBlock<Cam>,
}

#[arael::model]
#[derive(Clone)]
pub struct Pt {
    pub pos: Param<vect3<f64>>,
    pub hb: SelfBlock<Pt>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    [100.0 * pc.x / pc.z - obs.xy.x, 100.0 * pc.y / pc.z - obs.xy.y]
}))]
#[derive(Clone)]
pub struct Obs {
    #[arael(ref = root.cams)]
    pub cam: Ref<Cam>,
    #[arael(ref = root.pts)]
    pub pt: Ref<Pt>,
    pub xy: vect2<f64>,
    pub hb: CrossBlock<Cam, Pt>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct World {
    pub cams: refs::Vec<Cam>,
    pub pts: refs::Vec<Pt>,
    pub obs: std::vec::Vec<Obs>,
}

/// Observations of true cameras and points, then the cameras turned and
/// moved and the points moved off the truth.
fn world() -> World {
    let mut cams = refs::Vec::new();
    let mut truth = Vec::new();
    for k in 0..4 {
        let t = vect3d::new(0.3 * k as f64, -0.1 * k as f64, 0.0);
        let ea = vect3d::new(0.02 * k as f64, -0.03, 0.05 * k as f64);
        truth.push((t, ea));
        cams.push(Cam {
            t: Param::new(t + vect3d::new(0.02, -0.01, 0.015)),
            ea: MixedEulerAngleParam::new(ea + vect3d::new(0.01, -0.008, 0.012)),
            prior: t,
            hb: SelfBlock::new(),
        });
    }
    let mut pts = refs::Vec::new();
    let mut ptrue = Vec::new();
    for p in 0..12 {
        let q = vect3d::new(0.4 * (p % 4) as f64 - 0.6, 0.3 * (p / 4) as f64 - 0.3, 4.0 + 0.2 * (p % 3) as f64);
        ptrue.push(q);
        pts.push(Pt { pos: Param::new(q + vect3d::new(0.03, -0.02, 0.05)), hb: SelfBlock::new() });
    }
    let mut obs = Vec::new();
    for (p, q) in ptrue.iter().enumerate() {
        for (k, (t, ea)) in truth.iter().enumerate() {
            if (p + k) % 4 == 3 {
                continue;
            }
            let r = arael::matrix::matrix3d::rotation_from_euler_angles(*ea);
            let pc = r * *q + *t;
            obs.push(Obs { cam: cams.ref_at(k), pt: pts.ref_at(p), xy: vect2::new(100.0 * pc.x / pc.z, 100.0 * pc.y / pc.z), hb: CrossBlock::new() });
        }
    }
    World { cams, pts, obs }
}

#[cfg(feature = "gpu-cuda")]
fn gpu() -> GpuDevice {
    GpuDevice::cuda(0)
}
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
fn gpu() -> GpuDevice {
    GpuDevice::hip(0)
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

fn solve(w: &mut World, options: &SparseFaerOptions) -> SolveResult<f64> {
    let mut x = Vec::new();
    w.serialize(&mut x);
    let cfg = LmConfig::<f64> { max_iters: 30, ..Default::default() };
    arael::simple_lm::lm_solve(&x, &mut SparseFaer::<f64>::from_options(options), w, &cfg)
}

#[test]
fn every_device_failure_in_a_solve_is_reported_and_the_next_solve_succeeds() {
    let _turn = one_at_a_time();
    for (route, options) in routes() {
        let reference = solve(&mut world(), &options).unwrap_or_else(|f| panic!("{route}: {f}"));
        assert!(reference.accepted_iterations > 1, "{route}: the solve takes steps");
        fault::arm(0);
        solve(&mut world(), &options).unwrap();
        let operations = fault::disarm();
        assert!(operations > 0, "{route}: device operations counted");
        let mut with_partial = 0;
        for k in 1..=operations {
            let mut w = world();
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
                let cost = w.calc_cost(&p.x);
                assert!((cost - p.end_cost).abs() <= 1e-9 * p.end_cost, "{route}: operation {k}: the partial result's cost {} against {cost} at its parameters", p.end_cost);
            }
            let next = solve(&mut world(), &options).unwrap_or_else(|f| panic!("{route}: after operation {k}: {f}"));
            assert!((next.end_cost - reference.end_cost).abs() <= 1e-9 * reference.end_cost + 1e-20, "{route}: after operation {k}: {} against {}", next.end_cost, reference.end_cost);
        }
        // failures in the setup (no partial result) and in the loop (one)
        assert!(with_partial > 0 && with_partial < operations, "{route}: {with_partial} of {operations} failures with a partial result");
        eprintln!("{route}: {operations} device operations, each failure reported, {with_partial} with a partial result");
    }
}

/// A session keeps its device state between solves: a solve that fails at
/// any operation leaves the session able to solve again.
#[test]
fn a_session_solves_again_after_any_device_failure() {
    use arael::simple_lm::LmSession;
    let _turn = one_at_a_time();
    let options = routes().remove(0).1;
    let cfg = LmConfig::<f64> { max_iters: 30, ..Default::default() };
    let reference = solve(&mut world(), &options).unwrap();
    fault::arm(0);
    let mut session = LmSession::new(SparseFaer::<f64>::from_options(&options));
    session.solve(&mut world(), &cfg).unwrap();
    session.solve(&mut world(), &cfg).unwrap();
    let operations = fault::disarm();
    for k in 1..=operations {
        let mut session = LmSession::new(SparseFaer::<f64>::from_options(&options));
        fault::arm(k);
        // the first solve, or the warm second
        let failed = session.solve(&mut world(), &cfg).is_err() || session.solve(&mut world(), &cfg).is_err();
        fault::disarm();
        assert!(failed, "operation {k} of {operations}: no solve failed");
        let next = session.solve(&mut world(), &cfg).unwrap_or_else(|f| panic!("after operation {k}: {f}"));
        assert!((next.end_cost - reference.end_cost).abs() <= 1e-9 * reference.end_cost + 1e-20, "after operation {k}: {} against {}", next.end_cost, reference.end_cost);
    }
    eprintln!("session: {operations} device operations over two solves, each failure recovered from");
}
