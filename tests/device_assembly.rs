//! The device assembly (feature `gpu-cuda`): a root whose constraints all
//! have a device form assembles its gradient, Hessian tiles and cost on
//! the GPU to what the CPU assembly gives against the same pattern.
#![cfg(feature = "gpu-cuda")]

use arael::model::{CrossBlock, MixedEulerAngleParam, Param, QuaternionParam, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmProblem, LmProblemInternals, RootProblem};
use arael::utils::Float;
use arael::vect::{vect2, vect3};

/// The bundle-adjustment camera of benchmarks/bal: translation, rotation,
/// (f, k1, k2), and the depth under which an observation is dropped.
#[arael::model]
#[derive(Clone)]
pub struct Camera<T: Float> {
    pub t: Param<vect3<T>>,
    pub ea: MixedEulerAngleParam<T>,
    pub intr: Param<vect3<T>>,
    pub zdrop: T,
    pub hb: SelfBlock<Camera<T>, T>,
}

#[arael::model]
#[derive(Clone)]
pub struct Point<T: Float> {
    pub pos: Param<vect3<T>>,
    pub hb: SelfBlock<Point<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let keep = -pc.z - cam.zdrop;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [branch(keep, with_wide(cam.intr.x * d * px - obs.xy.x), 0.0),
     branch(keep, with_wide(cam.intr.x * d * py - obs.xy.y), 0.0)]
}))]
#[derive(Clone)]
pub struct Obs<T: Float> {
    #[arael(ref = root.cameras)]
    pub cam: Ref<Camera<T>>,
    #[arael(ref = root.points)]
    pub pt: Ref<Point<T>>,
    pub xy: vect2<T>,
    pub hb: CrossBlock<Camera<T>, Point<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Scene {
    pub cameras: refs::Vec<Camera<f64>>,
    pub points: refs::Vec<Point<f64>>,
    pub observations: std::vec::Vec<Obs<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct SceneF {
    pub cameras: refs::Vec<Camera<f32>>,
    pub points: refs::Vec<Point<f32>>,
    pub observations: std::vec::Vec<Obs<f32>>,
}

/// A point prior: a self-block constraint, which has no device form yet.
#[arael::model]
#[arael(constraint(hb, { [priorpoint.pos.x - 1.0, priorpoint.pos.y, priorpoint.pos.z] }))]
#[derive(Clone)]
pub struct PriorPoint {
    pub pos: Param<vect3<f64>>,
    pub hb: SelfBlock<PriorPoint>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Priors {
    pub points: refs::Vec<PriorPoint>,
}

/// A scene's cameras, points and observations.
type Parts<T> = (refs::Vec<Camera<T>>, refs::Vec<Point<T>>, std::vec::Vec<Obs<T>>);

struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// Six cameras looking down -z at forty points about five units away,
/// each point seen by three or four cameras; camera 2's intrinsics fixed,
/// camera 4's observations dropped by its depth guard.
fn scene<T: Float>() -> Parts<T> {
    let mut rng = Lcg(7);
    let c = |v: f64| T::from(v).unwrap();
    let mut cameras = refs::Vec::new();
    for k in 0..6 {
        let ea = arael::vect::vect3d::new(0.1 * rng.uniform(), 0.1 * rng.uniform(), 0.3 * rng.uniform());
        let intr = vect3::new(c(500.0 + 10.0 * k as f64), c(0.01 * rng.uniform()), c(0.001 * rng.uniform()));
        cameras.push(Camera {
            t: Param::new(vect3::new(c(rng.uniform()), c(rng.uniform()), c(0.2 * rng.uniform()))),
            ea: MixedEulerAngleParam::new(ea),
            intr: if k == 2 { Param::fixed(intr) } else { Param::new(intr) },
            zdrop: c(if k == 4 { 1e30 } else { -1e30 }),
            hb: SelfBlock::new(),
        });
    }
    let mut points = refs::Vec::new();
    for _ in 0..40 {
        points.push(Point { pos: Param::new(vect3::new(c(2.0 * rng.uniform()), c(2.0 * rng.uniform()), c(-5.0 + rng.uniform()))), hb: SelfBlock::new() });
    }
    let mut observations = std::vec::Vec::new();
    for p in 0..40 {
        for j in 0..(3 + p % 2) {
            let cam = (p + 2 * j) % 6;
            observations.push(Obs {
                cam: cameras.ref_at(cam),
                pt: points.ref_at(p),
                xy: vect2::new(c(40.0 * rng.uniform()), c(40.0 * rng.uniform())),
                hb: CrossBlock::new(),
            });
        }
    }
    (cameras, points, observations)
}

/// The parameters, moved off the serialized point so the rotations' steps
/// are not zero.
fn perturbed<T: Float>(x: &[T]) -> Vec<T> {
    x.iter().enumerate().map(|(i, &v)| v + T::from(1e-3 * ((i % 7) as f64 - 3.0)).unwrap()).collect()
}

fn gpu() -> arael::__gpu::GpuContext {
    arael::__gpu::GpuContext::new(arael::__gpu::GpuBackend::Cuda, 0).unwrap()
}

#[test]
fn the_f64_bundle_adjustment_assembles_on_the_device_as_on_the_cpu() {
    let (cameras, points, observations) = scene::<f64>();
    let mut s = Scene { cameras, points, observations };
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x = perturbed(&x);
    let chk = arael::__gpu::device_check(&mut s, &x, &gpu());
    assert!(chk.values > 0);
    assert!((chk.device_cost - chk.cpu_cost).abs() <= 1e-12 * chk.cpu_cost, "{chk:?}");
    assert!(chk.grad_err < 1e-12 && chk.hess_err < 1e-12, "{chk:?}");
}

#[test]
fn the_f32_bundle_adjustment_assembles_on_the_device_as_on_the_cpu() {
    let (cameras, points, observations) = scene::<f32>();
    let mut s = SceneF { cameras, points, observations };
    assert_eq!(LmProblemInternals::<f32>::device_unsupported(&s), None);
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x = perturbed(&x);
    let chk = arael::__gpu::device_check(&mut s, &x, &gpu());
    // the CPU sums its f32 products in f32, the device in f64
    assert!((chk.device_cost - chk.cpu_cost).abs() <= 1e-5 * chk.cpu_cost, "{chk:?}");
    assert!(chk.grad_err < 1e-4 && chk.hess_err < 1e-4, "{chk:?}");
}

/// A scene whose observations are the projections of true cameras and
/// points plus half a pixel of noise, the points and translations then
/// moved off the truth: what a solve starts from.
fn consistent_scene<T: Float>() -> Parts<T> {
    let mut rng = Lcg(11);
    let c = |v: f64| T::from(v).unwrap();
    let mut cams = Vec::new();
    for k in 0..6 {
        let ea = arael::vect::vect3d::new(0.05 * rng.uniform(), 0.05 * rng.uniform(), 0.2 * rng.uniform());
        let t = arael::vect::vect3d::new(0.5 * rng.uniform(), 0.5 * rng.uniform(), 0.1 * rng.uniform());
        cams.push((ea, t, 500.0 + 10.0 * k as f64, 0.01 * rng.uniform(), 0.001 * rng.uniform()));
    }
    let pts: Vec<arael::vect::vect3d> = (0..60).map(|_| arael::vect::vect3d::new(2.0 * rng.uniform(), 2.0 * rng.uniform(), -5.0 + rng.uniform())).collect();
    let mut cameras = refs::Vec::new();
    for &(ea, t, f, k1, k2) in &cams {
        let moved = t + arael::vect::vect3d::new(0.01 * rng.uniform(), 0.01 * rng.uniform(), 0.01 * rng.uniform());
        cameras.push(Camera {
            t: Param::new(vect3::new(c(moved.x), c(moved.y), c(moved.z))),
            ea: MixedEulerAngleParam::new(ea),
            intr: Param::new(vect3::new(c(f), c(k1), c(k2))),
            zdrop: c(-1e30),
            hb: SelfBlock::new(),
        });
    }
    let mut points = refs::Vec::new();
    for p in &pts {
        let q = *p + arael::vect::vect3d::new(0.02 * rng.uniform(), 0.02 * rng.uniform(), 0.02 * rng.uniform());
        points.push(Point { pos: Param::new(vect3::new(c(q.x), c(q.y), c(q.z))), hb: SelfBlock::new() });
    }
    let mut observations = std::vec::Vec::new();
    for (pi, p) in pts.iter().enumerate() {
        for j in 0..(3 + pi % 2) {
            let ci = (pi + 2 * j) % 6;
            let (ea, t, f, k1, k2) = cams[ci];
            let pc = arael::matrix::matrix3d::rotation_from_euler_angles(ea) * *p + t;
            let (px, py) = (-pc.x / pc.z, -pc.y / pc.z);
            let r2 = px * px + py * py;
            let d = 1.0 + r2 * (k1 + k2 * r2);
            observations.push(Obs {
                cam: cameras.ref_at(ci),
                pt: points.ref_at(pi),
                xy: vect2::new(c(f * d * px + 0.5 * rng.uniform()), c(f * d * py + 0.5 * rng.uniform())),
                hb: CrossBlock::new(),
            });
        }
    }
    (cameras, points, observations)
}

fn options(gpu: bool, assembly: bool) -> arael::simple_lm::SparseFaerOptions {
    arael::simple_lm::SparseFaerOptions::default()
        .with_policy(arael::simple_lm::SchurPolicy::Force)
        .with_envelope_schur(arael::simple_lm::EnvelopeMode::Never)
        .with_block_supernodal(arael::simple_lm::BlockSupernodalMode::Always)
        .with_gpu(gpu.then(|| arael::simple_lm::GpuDevice::cuda(0)))
        .with_gpu_assembly(assembly)
}

/// The LM solve through the device assembly (H, the gradient and the
/// trial costs on the GPU, the rotations re-centred there after every
/// accepted step) reaches the CPU route's optimum, as the GPU route with
/// the assembly on the CPU does.
#[test]
fn a_solve_through_the_device_assembly_reaches_the_cpu_optimum() {
    let run = |gpu: bool, assembly: bool| {
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&options(gpu, assembly));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        let plan = solver.plan().expect("a plan");
        assert!(plan.reduced && plan.gpu_factor == gpu);
        assert_eq!(plan.gpu_assembly, gpu && assembly);
        (r.start_cost, r.end_cost, r.iterations)
    };
    let (start, cpu, _) = run(false, false);
    let (_, uploaded, _) = run(true, false);
    let (_, device, iters) = run(true, true);
    assert!(cpu < 0.05 * start, "the scene converges to its noise: {start} to {cpu}");
    assert!(iters > 2, "accepted steps re-centre the rotations: {iters} iterations");
    for c in [uploaded, device] {
        assert!((c - cpu).abs() <= 1e-9 * cpu, "{c} against the CPU's {cpu}");
    }
}

/// The same for an f32 root: the device's sums in f64 land on the CPU f32
/// route's optimum to its precision.
#[test]
fn an_f32_solve_through_the_device_assembly_reaches_the_cpu_optimum() {
    let run = |gpu: bool| {
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&options(gpu, true));
        let cfg = arael::simple_lm::LmConfig::<f32> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu);
        r.end_cost
    };
    let (cpu, device) = (run(false), run(true));
    assert!((device - cpu).abs() <= 1e-4 * cpu, "{device} against the CPU's {cpu}");
}

#[test]
fn a_root_with_a_constraint_without_a_device_form_says_which() {
    let mut points = refs::Vec::new();
    points.push(PriorPoint { pos: Param::new(vect3::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    let s = Priors { points };
    let why = LmProblemInternals::<f64>::device_unsupported(&s).expect("a self-block constraint has no device form yet");
    assert!(why.contains("PriorPoint"), "{why}");
}

/// A camera turned by a quaternion: its rotation has no device form (the
/// device composes Euler-angle steps).
#[arael::model]
#[derive(Clone)]
pub struct QCamera {
    pub t: Param<vect3<f64>>,
    pub q: QuaternionParam<f64>,
    pub hb: SelfBlock<QCamera>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.q.rotation_matrix() * pt.pos + cam.t;
    [-pc.x / pc.z - qobs.xy.x, -pc.y / pc.z - qobs.xy.y]
}))]
#[derive(Clone)]
pub struct QObs {
    #[arael(ref = root.cameras)]
    pub cam: Ref<QCamera>,
    #[arael(ref = root.points)]
    pub pt: Ref<Point<f64>>,
    pub xy: vect2<f64>,
    pub hb: CrossBlock<QCamera, Point<f64>>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct QScene {
    pub cameras: refs::Vec<QCamera>,
    pub points: refs::Vec<Point<f64>>,
    pub observations: std::vec::Vec<QObs>,
}

#[test]
fn a_quaternion_rotation_has_no_device_form() {
    let s = QScene { cameras: refs::Vec::new(), points: refs::Vec::new(), observations: std::vec::Vec::new() };
    let why = LmProblemInternals::<f64>::device_unsupported(&s).expect("a quaternion rotation has no device form");
    assert!(why.contains("cam.q") && why.contains("EulerAngleParam"), "{why}");
}

/// The parameters' values as the model holds them after a solve.
fn state(s: &Scene) -> Vec<f64> {
    let mut v = Vec::new();
    for c in s.cameras.iter() {
        for p in [c.t.value, c.ea.value, c.intr.value] {
            v.extend([p.x, p.y, p.z]);
        }
    }
    for p in s.points.iter() {
        v.extend([p.pos.value.x, p.pos.value.y, p.pos.value.z]);
    }
    v
}

fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len());
    let err = a.iter().zip(b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    assert!(err <= tol, "{what}: the largest difference is {err}");
}

/// The device solve leaves the model where the CPU's does (the reference
/// rotations come back from the device), and a session's next solve on
/// changed data reads the model again.
#[test]
fn a_device_solve_writes_the_model_back_and_a_warm_one_reads_it_again() {
    use arael::simple_lm::{LmConfig, LmSession, SparseFaer};
    let cfg = LmConfig::<f64> { max_iters: 50, ..Default::default() };
    let cpu = |s: &Scene| {
        let mut s = s.clone();
        let r = s.solve_with(&mut SparseFaer::<f64>::from_options(&options(false, false)), &cfg).unwrap();
        (r.end_cost, state(&s))
    };
    let (cameras, points, observations) = consistent_scene::<f64>();
    let mut s = Scene { cameras, points, observations };
    let (cpu_cost, cpu_state) = cpu(&s);
    let mut session = LmSession::new(SparseFaer::<f64>::from_options(&options(true, true)));
    let r = session.solve(&mut s, &cfg).unwrap();
    assert!(session.solver().plan().unwrap().gpu_assembly);
    assert!((r.end_cost - cpu_cost).abs() <= 1e-9 * cpu_cost, "{} against the CPU's {cpu_cost}", r.end_cost);
    assert_close(&state(&s), &cpu_state, 1e-7, "the solved model");
    // other data: an observation, a camera's rotation, a point
    s.observations[5].xy.x += 3.0;
    s.cameras[1].ea.value.z += 0.02;
    s.points[7].pos.value.y -= 0.05;
    let (cpu_cost, cpu_state) = cpu(&s);
    let r = session.solve(&mut s, &cfg).unwrap();
    assert!((r.end_cost - cpu_cost).abs() <= 1e-9 * cpu_cost, "{} against the CPU's {cpu_cost}", r.end_cost);
    assert_close(&state(&s), &cpu_state, 1e-7, "the warm solve's model");
}

type Seen = std::sync::Arc<std::sync::Mutex<Vec<(bool, Vec<f64>)>>>;

/// A lambda driver that records what it is handed: the vectors' lengths
/// and the predicted reduction, set and as the step reports it.
#[derive(Clone)]
struct Probe {
    inner: arael::simple_lm::DefaultLambdaDriver<f64>,
    seen: std::sync::Arc<std::sync::Mutex<Vec<(usize, usize, usize, Option<f64>, f64)>>>,
}

impl arael::simple_lm::LambdaDriver<f64> for Probe {
    fn start(&mut self, config: &arael::simple_lm::LmConfig<f64>, state: &arael::simple_lm::LambdaState<f64>) -> f64 {
        self.seen.lock().unwrap().push((state.grad.len(), state.diagonal.len(), 0, None, 0.0));
        self.inner.start(config, state)
    }
    fn accepted(&mut self, step: &arael::simple_lm::LambdaStep<f64>) -> Option<f64> {
        self.seen.lock().unwrap().push((step.grad.len(), step.diagonal.len(), step.delta.len(), step.predicted, step.predicted_reduction()));
        self.inner.accepted(step)
    }
    fn rejected(&mut self, step: &arael::simple_lm::LambdaStep<f64>) -> Option<f64> {
        self.seen.lock().unwrap().push((step.grad.len(), step.diagonal.len(), step.delta.len(), step.predicted, step.predicted_reduction()));
        self.inner.rejected(step)
    }
    fn factorization_failed(&mut self, lambda: f64, state: &arael::simple_lm::LambdaState<f64>) -> Option<f64> {
        self.inner.factorization_failed(lambda, state)
    }
}

/// On the device the observer gets the parameters and the driver the
/// vectors only when the config asks; the predicted reduction always.
#[test]
fn the_device_solve_hands_out_vectors_when_asked() {
    use arael::simple_lm::{LmConfig, LmIter, SparseFaer};
    let run = |gpu: bool, asked: bool| {
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let seen: Seen = Default::default();
        let log = seen.clone();
        let probe = Probe { inner: Default::default(), seen: Default::default() };
        let driven = probe.seen.clone();
        let cfg = LmConfig::<f64> { max_iters: 20, ..Default::default() }
            .with_driver(probe)
            .with_observer(move |it: &LmIter<f64>| {
                log.lock().unwrap().push((it.accepted, it.params.to_vec()));
                std::ops::ControlFlow::Continue(())
            })
            .with_device_observer_params(asked)
            .with_device_driver_vectors(asked);
        let mut solver = SparseFaer::<f64>::from_options(&options(gpu, true));
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        let seen = seen.lock().unwrap().clone();
        let driven = driven.lock().unwrap().clone();
        (x.len(), r, seen, driven)
    };
    let (n, cpu, seen, driven) = run(false, false);
    assert!(seen.iter().all(|(_, p)| p.len() == n));
    assert!(driven.iter().skip(1).all(|d| d.0 == n && d.1 == n && d.2 == n && d.3.is_none()));
    for asked in [false, true] {
        let (_, r, seen, driven) = run(true, asked);
        assert!(r.accepted_iterations > 2);
        assert!((r.end_cost - cpu.end_cost).abs() <= 1e-9 * cpu.end_cost);
        let want = if asked { n } else { 0 };
        assert!(seen.iter().all(|(_, p)| p.len() == want), "asked {asked}");
        assert!(driven.iter().all(|d| d.0 == want && d.1 == want));
        for &(_, _, delta, predicted, reduction) in driven.iter().skip(1) {
            assert_eq!(delta, want);
            assert_eq!(predicted, Some(reduction));
            assert!(reduction > 0.0);
        }
        if asked {
            // the last accepted step's parameters are the result
            let last = seen.iter().rev().find(|(a, _)| *a).unwrap();
            assert_close(&last.1, &r.x, 0.0, "the observer's last parameters");
        }
    }
}

/// The device recomputes the predicted reduction the host computes from
/// the vectors it hands out.
#[test]
fn the_device_predicted_reduction_is_the_host_formula() {
    use arael::simple_lm::{LambdaStep, LmConfig, SparseFaer};
    #[derive(Clone)]
    struct Check(std::sync::Arc<std::sync::Mutex<Vec<(f64, f64)>>>);
    impl arael::simple_lm::LambdaDriver<f64> for Check {
        fn start(&mut self, config: &LmConfig<f64>, _state: &arael::simple_lm::LambdaState<f64>) -> f64 {
            config.initial_lambda
        }
        fn accepted(&mut self, step: &LambdaStep<f64>) -> Option<f64> {
            let host = LambdaStep { predicted: None, ..*step }.predicted_reduction();
            self.0.lock().unwrap().push((step.predicted.unwrap(), host));
            Some(step.lambda * 0.2)
        }
        fn rejected(&mut self, step: &LambdaStep<f64>) -> Option<f64> {
            Some(step.lambda * 10.0)
        }
        fn factorization_failed(&mut self, lambda: f64, _state: &arael::simple_lm::LambdaState<f64>) -> Option<f64> {
            Some(lambda * 10.0)
        }
    }
    let (cameras, points, observations) = consistent_scene::<f64>();
    let mut s = Scene { cameras, points, observations };
    let mut x = Vec::new();
    s.serialize(&mut x);
    let pairs = Check(Default::default());
    let cfg = LmConfig::<f64> { max_iters: 10, ..Default::default() }.with_driver(pairs.clone()).with_device_driver_vectors(true);
    arael::simple_lm::lm_solve(&x, &mut SparseFaer::<f64>::from_options(&options(true, true)), &mut s, &cfg).unwrap();
    let pairs = pairs.0.lock().unwrap();
    assert!(pairs.len() > 2);
    for &(device, host) in pairs.iter() {
        assert!((device - host).abs() <= 1e-12 * host.abs(), "{device} against {host}");
    }
}

/// The gradient test on a device linearization, made with its first
/// attempt's read, stops where the host's does.
#[test]
fn the_device_gradient_test_stops_where_the_host_one_does() {
    use arael::simple_lm::{LmConfig, LmStatus, SparseFaer};
    let run = |assembly: bool| {
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let cfg = LmConfig::<f64> { max_iters: 100, gradient_tolerance: Some(1e-3), abs_precision: 0.0, rel_precision: 0.0, ..Default::default() };
        let mut solver = SparseFaer::<f64>::from_options(&options(true, assembly));
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, assembly);
        (r.status, r.iterations, r.accepted_iterations)
    };
    let host = run(false);
    assert_eq!(host.0, LmStatus::GradientTolerance);
    assert!(host.2 > 2);
    assert_eq!(run(true), host);
}

/// A slider pulled to 1 and, while the slider is below 0.5, a follower
/// pulled to 2: the first step takes the slider past 0.5, so no
/// curvature reaches the followers at the second linearization.
#[arael::model]
#[derive(Clone)]
pub struct Slider {
    pub s: Param<vect2<f64>>,
    pub hb: SelfBlock<Slider>,
}

#[arael::model]
#[derive(Clone)]
pub struct Follower {
    pub f: Param<vect2<f64>>,
    pub hb: SelfBlock<Follower>,
}

#[arael::model]
#[arael(constraint(hb, {
    [sl.s.x - 1.0, sl.s.y, branch(0.5 - sl.s.x, fo.f.x - 2.0, 0.0), branch(0.5 - sl.s.x, fo.f.y, 0.0)]
}))]
#[derive(Clone)]
pub struct Link {
    #[arael(ref = root.sliders)]
    pub sl: Ref<Slider>,
    #[arael(ref = root.followers)]
    pub fo: Ref<Follower>,
    pub hb: CrossBlock<Slider, Follower>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Switch {
    pub sliders: refs::Vec<Slider>,
    pub followers: refs::Vec<Follower>,
    pub links: std::vec::Vec<Link>,
}

/// A bad diagonal found on a device linearization ends the solve as the
/// host's check does: the same parameter, the same partial state.
#[test]
fn a_bad_diagonal_on_the_device_ends_the_solve_as_on_the_host() {
    use arael::simple_lm::{DiagonalFault, LmConfig, SolveFailureKind, SparseFaer};
    let run = |gpu: bool| {
        let mut sliders = refs::Vec::new();
        for _ in 0..2 {
            sliders.push(Slider { s: Param::new(vect2::new(0.0, 0.0)), hb: SelfBlock::new() });
        }
        let mut followers = refs::Vec::new();
        for _ in 0..3 {
            followers.push(Follower { f: Param::new(vect2::new(0.0, 0.0)), hb: SelfBlock::new() });
        }
        let mut links = std::vec::Vec::new();
        for k in 0..6 {
            links.push(Link { sl: sliders.ref_at(k % 2), fo: followers.ref_at(k % 3), hb: CrossBlock::new() });
        }
        let mut s = Switch { sliders, followers, links };
        assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = SparseFaer::<f64>::from_options(&options(gpu, true));
        let e = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &LmConfig::<f64>::default()).unwrap_err();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu);
        let p = e.partial.expect("a partial state");
        (e.kind, p.iterations, p.accepted_iterations, p.x)
    };
    let host = run(false);
    assert!(matches!(host.0, SolveFailureKind::DegenerateDiagonal { param: 4, fault: DiagonalFault::Zero }), "{:?}", host.0);
    assert_eq!((host.1, host.2), (1, 1));
    let device = run(true);
    assert_eq!(format!("{:?}", device.0), format!("{:?}", host.0));
    assert_eq!((device.1, device.2), (host.1, host.2));
    assert_close(&device.3, &host.3, 1e-12, "the partial state");
}

/// `GpuDevice::shared` hands every caller the same device state: a later
/// call finds the buffers a solve through an earlier one left, a separately
/// created device has none.
#[test]
fn a_shared_device_is_one_state_per_process() {
    use arael::simple_lm::{GpuBackend, GpuDevice, LmConfig, SparseFaer};
    let (cameras, points, observations) = consistent_scene::<f64>();
    let mut s = Scene { cameras, points, observations };
    let opts = options(true, true).with_gpu(Some(GpuDevice::shared(GpuBackend::Cuda, 0)));
    s.solve_with(&mut SparseFaer::<f64>::from_options(&opts), &LmConfig::<f64>::default()).unwrap();
    assert!(GpuDevice::shared(GpuBackend::Cuda, 0).reserved_bytes() > 0);
    assert_eq!(GpuDevice::cuda(0).reserved_bytes(), 0);
}
