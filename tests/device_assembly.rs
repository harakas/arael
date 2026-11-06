//! The device assembly (feature `gpu-cuda` or `gpu-hip`): a root whose
//! constraints all have a device form assembles its gradient, Hessian
//! tiles and cost on the GPU to what the CPU assembly gives against the
//! same pattern.
#![cfg(any(feature = "gpu-cuda", feature = "gpu-hip"))]

use arael::df32;
use arael::angle::AngleParam;
use arael::model::{CrossBlock, MixedEulerAngleParam, Param, QuaternionParam, SelfBlock, SimpleEulerAngleParam, VarSelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{GpuDevice, LmProblem, LmProblemInternals, RootProblem};
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

/// The camera with its rotation's state in df32, for an f32 root with
/// `wide = df32` (the macro takes one type parameter per entity, so a
/// concrete copy).
#[arael::model]
#[derive(Clone)]
pub struct CameraW {
    pub t: Param<vect3<f32>>,
    pub ea: MixedEulerAngleParam<f32, df32>,
    pub intr: Param<vect3<f32>>,
    pub zdrop: f32,
    pub hb: SelfBlock<CameraW, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let keep = -pc.z - cam.zdrop;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [branch(keep, with_wide(cam.intr.x * d * px - obsw.xy.x), 0.0),
     branch(keep, with_wide(cam.intr.x * d * py - obsw.xy.y), 0.0)]
}))]
#[derive(Clone)]
pub struct ObsW {
    #[arael(ref = root.cameras)]
    pub cam: Ref<CameraW>,
    #[arael(ref = root.points)]
    pub pt: Ref<Point<f32>>,
    pub xy: vect2<f32>,
    pub hb: CrossBlock<CameraW, Point<f32>, f32>,
}

#[arael::model]
#[arael(root, f32, wide = df32)]
#[derive(Clone)]
pub struct SceneW {
    pub cameras: refs::Vec<CameraW>,
    pub points: refs::Vec<Point<f32>>,
    pub observations: std::vec::Vec<ObsW>,
}

/// A point prior whose loss an index picks, which has no device form yet.
#[arael::model]
#[arael(constraint(hb, loss = |s| loss_select(priorpoint.kind, s, 1.0), { [priorpoint.pos.x - 1.0, priorpoint.pos.y, priorpoint.pos.z] }))]
#[derive(Clone)]
pub struct PriorPoint {
    pub pos: Param<vect3<f64>>,
    pub kind: u32,
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

/// A scene whose observations are the projections of true cameras and
/// points plus half a pixel of noise, the points and translations then
/// moved off the truth: what a solve starts from.
fn consistent_scene<T: Float>() -> Parts<T> {
    consistent_scene_of(6, 60, 11)
}

/// [`consistent_scene`] with `nc` cameras and `np` points, from `seed`.
fn consistent_scene_of<T: Float>(nc: usize, np: usize, seed: u64) -> Parts<T> {
    let mut rng = Lcg(seed);
    let c = |v: f64| T::from(v).unwrap();
    let mut cams = Vec::new();
    for k in 0..nc {
        let ea = arael::vect::vect3d::new(0.05 * rng.uniform(), 0.05 * rng.uniform(), 0.2 * rng.uniform());
        let t = arael::vect::vect3d::new(0.5 * rng.uniform(), 0.5 * rng.uniform(), 0.1 * rng.uniform());
        cams.push((ea, t, 500.0 + 10.0 * k as f64, 0.01 * rng.uniform(), 0.001 * rng.uniform()));
    }
    let pts: Vec<arael::vect::vect3d> = (0..np).map(|_| arael::vect::vect3d::new(2.0 * rng.uniform(), 2.0 * rng.uniform(), -5.0 + rng.uniform())).collect();
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
            let ci = (pi + 2 * j) % nc;
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

/// The GPU runtimes compiled in, on device 0, in both wide scalars.
fn devices() -> Vec<GpuDevice> {
    let mut v = Vec::new();
    #[cfg(feature = "gpu-cuda")]
    v.extend([GpuDevice::cuda(0), GpuDevice::cuda(0).with_wide(arael::simple_lm::GpuWide::Df32)]);
    #[cfg(feature = "gpu-hip")]
    v.extend([GpuDevice::hip(0), GpuDevice::hip(0).with_wide(arael::simple_lm::GpuWide::Df32)]);
    v
}

/// A device's runtime and wide scalar, as the verbose log names them.
fn label(d: &GpuDevice) -> String {
    let runtime = match d.backend {
        arael::simple_lm::GpuBackend::Cuda => "cuda",
        arael::simple_lm::GpuBackend::Hip => "hip",
    };
    match d.wide() {
        arael::simple_lm::GpuWide::F64 => runtime.to_string(),
        arael::simple_lm::GpuWide::Df32 => format!("{runtime} df32"),
    }
}

/// The reduced supernodal route, on `gpu` when it is given.
fn options(gpu: Option<&GpuDevice>, assembly: bool) -> arael::simple_lm::SparseFaerOptions {
    arael::simple_lm::SparseFaerOptions::default()
        .with_policy(arael::simple_lm::SchurPolicy::Force)
        .with_envelope_schur(arael::simple_lm::EnvelopeMode::Never)
        .with_block_supernodal(arael::simple_lm::BlockSupernodalMode::Always)
        .with_gpu(gpu.cloned())
        .with_gpu_assembly(assembly)
}

/// The LM solve through the device assembly (H, the gradient and the
/// trial costs on the GPU, the rotations re-centred there after every
/// accepted step) reaches the CPU route's optimum, as the GPU route with
/// the assembly on the CPU does, on each runtime.
#[test]
fn a_solve_through_the_device_assembly_reaches_the_cpu_optimum() {
    let run = |gpu: Option<&GpuDevice>, assembly: bool| {
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&options(gpu, assembly));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        let plan = solver.plan().expect("a plan");
        assert!(plan.reduced && plan.gpu_factor == gpu.is_some());
        assert_eq!(plan.gpu_assembly, gpu.is_some() && assembly);
        (r.start_cost, r.end_cost, r.iterations)
    };
    let (start, cpu, _) = run(None, false);
    assert!(cpu < 0.05 * start, "the scene converges to its noise: {start} to {cpu}");
    for d in devices() {
        let (_, uploaded, _) = run(Some(&d), false);
        let (_, device, iters) = run(Some(&d), true);
        assert!(iters > 2, "{}: accepted steps re-centre the rotations: {iters} iterations", label(&d));
        for c in [uploaded, device] {
            assert!((c - cpu).abs() <= 1e-9 * cpu, "{}: {c} against the CPU's {cpu}", label(&d));
        }
    }
}

/// The same for an f32 root: the device's sums in f64 land on the CPU f32
/// route's optimum to its precision.
#[test]
fn an_f32_solve_through_the_device_assembly_reaches_the_cpu_optimum() {
    let run = |gpu: Option<&GpuDevice>| {
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&options(gpu, true));
        let cfg = arael::simple_lm::LmConfig::<f32> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        r.end_cost
    };
    let cpu = run(None);
    for d in devices() {
        let device = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-4 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
    }
}

#[test]
fn a_root_with_a_constraint_without_a_device_form_says_which() {
    let mut points = refs::Vec::new();
    points.push(PriorPoint { pos: Param::new(vect3::new(0.0, 0.0, 0.0)), kind: 1, hb: SelfBlock::new() });
    let s = Priors { points };
    let why = LmProblemInternals::<f64>::device_unsupported(&s).expect("a loss picked by an index has no device form yet");
    assert!(why.contains("PriorPoint") && why.contains("select"), "{why}");
}

/// A camera turned by a `QuaternionParam`: the device composes its
/// reference quaternion with its rotation-vector step.
#[arael::model]
#[derive(Clone)]
pub struct QCamera {
    pub t: Param<vect3<f64>>,
    pub q: QuaternionParam<f64>,
    pub intr: Param<vect3<f64>>,
    pub hb: SelfBlock<QCamera>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.q.rotation_matrix() * pt.pos + cam.t;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [cam.intr.x * d * px - qobs.xy.x, cam.intr.x * d * py - qobs.xy.y]
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

/// [`consistent_scene`] with the cameras turned by quaternions.
fn quaternion_scene() -> QScene {
    let (cams, points, obs) = consistent_scene::<f64>();
    let mut cameras = refs::Vec::new();
    for c in cams.iter() {
        cameras.push(QCamera { t: c.t.clone(), q: QuaternionParam::from_euler_angles(c.ea.value), intr: c.intr.clone(), hb: SelfBlock::new() });
    }
    let observations = obs
        .iter()
        .map(|o| QObs { cam: cameras.ref_at(o.cam.index() as usize), pt: points.ref_at(o.pt.index() as usize), xy: o.xy, hb: CrossBlock::new() })
        .collect();
    QScene { cameras, points, observations }
}

/// The parameter values of a [`QScene`], the rotations as quaternions.
fn quaternion_state(s: &QScene) -> Vec<f64> {
    let mut v = Vec::new();
    for c in s.cameras.iter() {
        let q = c.q.value;
        v.extend([c.t.value.x, c.t.value.y, c.t.value.z, q.t, q.v.x, q.v.y, q.v.z, c.intr.value.x, c.intr.value.y, c.intr.value.z]);
    }
    for p in s.points.iter() {
        v.extend([p.pos.value.x, p.pos.value.y, p.pos.value.z]);
    }
    v
}

/// Cameras turned by quaternions assemble on the device as on the CPU, and
/// the solve re-centres them there and leaves the model where the CPU's
/// does.
#[test]
fn a_quaternion_scene_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    use arael::simple_lm::{LmConfig, SparseFaer};
    let mut s = quaternion_scene();
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for (wide, tol) in [(Wide::F64, 1e-11), (Wide::Df32, 1e-10)] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
    }
    let cfg = LmConfig::<f64> { max_iters: 50, ..Default::default() };
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = quaternion_scene();
        let mut solver = SparseFaer::<f64>::from_options(&options(gpu, true));
        let r = s.solve_with(&mut solver, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        (r.end_cost, r.iterations, quaternion_state(&s))
    };
    let (cpu, iters, cpu_state) = run(None);
    assert!(iters > 2, "accepted steps re-centre the rotations: {iters} iterations");
    for d in devices() {
        let (device, _, state) = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
        assert_close(&state, &cpu_state, 1e-7, &format!("{}: the solved model", label(&d)));
    }
}

/// `parts` as a [`SceneW`]: the rotations' state moved into df32.
fn wide_scene((cams, points, obs): Parts<f32>) -> SceneW {
    let mut cameras = refs::Vec::new();
    for c in cams.iter() {
        let v = c.ea.value;
        let d = df32::from_f64;
        cameras.push(CameraW {
            t: c.t.clone(),
            ea: MixedEulerAngleParam::new(vect3::new(d(v.x), d(v.y), d(v.z))),
            intr: c.intr.clone(),
            zdrop: c.zdrop,
            hb: SelfBlock::new(),
        });
    }
    let observations = obs
        .iter()
        .map(|o| ObsW { cam: cameras.ref_at(o.cam.index() as usize), pt: points.ref_at(o.pt.index() as usize), xy: o.xy, hb: CrossBlock::new() })
        .collect();
    SceneW { cameras, points, observations }
}

/// An f32 root whose rotations keep their state in df32 (`wide = df32`)
/// solves on the device to the CPU's optimum and leaves the model where the
/// CPU does: the pack walk reads the df32 steps and reference rotations,
/// and the device's re-centred rotations come back into them.
#[test]
fn a_df32_wide_root_solves_on_the_device_as_on_the_cpu() {
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = wide_scene(consistent_scene::<f32>());
        assert_eq!(LmProblemInternals::<f32>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&options(gpu, true));
        let cfg = arael::simple_lm::LmConfig::<f32> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        let angles: Vec<f64> = s.cameras.iter().flat_map(|c| [c.ea.value.x, c.ea.value.y, c.ea.value.z]).map(|a| a.to_f64()).collect();
        (r.end_cost, angles)
    };
    let (cpu, cpu_angles) = run(None);
    for d in devices() {
        let (device, angles) = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-4 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
        assert_close(&angles, &cpu_angles, 1e-4, &format!("{}: the rotations", label(&d)));
    }
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

/// `a` within `tol` of `b`, entry by entry, relative to the entry's size
/// past 1: the scene's focal lengths are near 500, its angles near 0.1.
fn assert_close(a: &[f64], b: &[f64], tol: f64, what: &str) {
    assert_eq!(a.len(), b.len());
    let rel = |x: f64, y: f64| (x - y).abs() / y.abs().max(1.0);
    let (err, at) = a.iter().zip(b).enumerate().fold((0.0f64, 0), |m, (k, (&x, &y))| if rel(x, y) > m.0 { (rel(x, y), k) } else { m });
    assert!(err <= tol, "{what}: the largest difference is {err}, at {at} ({} against {})", a[at], b[at]);
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
        let r = s.solve_with(&mut SparseFaer::<f64>::from_options(&options(None, false)), &cfg).unwrap();
        (r.end_cost, state(&s))
    };
    for d in devices() {
        let name = label(&d);
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let (cpu_cost, cpu_state) = cpu(&s);
        let mut session = LmSession::new(SparseFaer::<f64>::from_options(&options(Some(&d), true)));
        let r = session.solve(&mut s, &cfg).unwrap();
        assert!(session.solver().plan().unwrap().gpu_assembly);
        assert!((r.end_cost - cpu_cost).abs() <= 1e-9 * cpu_cost, "{name}: {} against the CPU's {cpu_cost}", r.end_cost);
        assert_close(&state(&s), &cpu_state, 1e-7, &format!("{name}: the solved model"));
        // other data: an observation, a camera's rotation, a point
        s.observations[5].xy.x += 3.0;
        s.cameras[1].ea.value.z += 0.02;
        s.points[7].pos.value.y -= 0.05;
        let (cpu_cost, cpu_state) = cpu(&s);
        let r = session.solve(&mut s, &cfg).unwrap();
        assert!((r.end_cost - cpu_cost).abs() <= 1e-9 * cpu_cost, "{name}: {} against the CPU's {cpu_cost}", r.end_cost);
        assert_close(&state(&s), &cpu_state, 1e-7, &format!("{name}: the warm solve's model"));
    }
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
    let run = |gpu: Option<&GpuDevice>, asked: bool| {
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
    let (n, cpu, seen, driven) = run(None, false);
    assert!(seen.iter().all(|(_, p)| p.len() == n));
    assert!(driven.iter().skip(1).all(|d| d.0 == n && d.1 == n && d.2 == n && d.3.is_none()));
    for d in devices() {
        let name = label(&d);
        for asked in [false, true] {
            let (_, r, seen, driven) = run(Some(&d), asked);
            assert!(r.accepted_iterations > 2);
            assert!((r.end_cost - cpu.end_cost).abs() <= 1e-9 * cpu.end_cost, "{name}");
            let want = if asked { n } else { 0 };
            assert!(seen.iter().all(|(_, p)| p.len() == want), "{name}: asked {asked}");
            assert!(driven.iter().all(|d| d.0 == want && d.1 == want), "{name}");
            for &(_, _, delta, predicted, reduction) in driven.iter().skip(1) {
                assert_eq!(delta, want);
                assert_eq!(predicted, Some(reduction));
                assert!(reduction > 0.0);
            }
            if asked {
                // the last accepted step's parameters are the result
                let last = seen.iter().rev().find(|(a, _)| *a).unwrap();
                assert_close(&last.1, &r.x, 0.0, &format!("{name}: the observer's last parameters"));
            }
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
    for d in devices() {
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let pairs = Check(Default::default());
        let cfg = LmConfig::<f64> { max_iters: 10, ..Default::default() }.with_driver(pairs.clone()).with_device_driver_vectors(true);
        arael::simple_lm::lm_solve(&x, &mut SparseFaer::<f64>::from_options(&options(Some(&d), true)), &mut s, &cfg).unwrap();
        let pairs = pairs.0.lock().unwrap();
        assert!(pairs.len() > 2);
        for &(device, host) in pairs.iter() {
            assert!((device - host).abs() <= 1e-12 * host.abs(), "{}: {device} against {host}", label(&d));
        }
    }
}

/// The gradient test on a device linearization, made with its first
/// attempt's read, stops where the host's does.
#[test]
fn the_device_gradient_test_stops_where_the_host_one_does() {
    use arael::simple_lm::{LmConfig, LmStatus, SparseFaer};
    let run = |d: &GpuDevice, assembly: bool| {
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let cfg = LmConfig::<f64> { max_iters: 100, gradient_tolerance: Some(1e-3), abs_precision: 0.0, rel_precision: 0.0, ..Default::default() };
        let mut solver = SparseFaer::<f64>::from_options(&options(Some(d), assembly));
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, assembly);
        (r.status, r.iterations, r.accepted_iterations)
    };
    for d in devices() {
        let host = run(&d, false);
        assert_eq!(host.0, LmStatus::GradientTolerance);
        assert!(host.2 > 2);
        assert_eq!(run(&d, true), host, "{}", label(&d));
    }
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
    let run = |gpu: Option<&GpuDevice>| {
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
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        let p = e.partial.expect("a partial state");
        (e.kind, p.iterations, p.accepted_iterations, p.x)
    };
    let host = run(None);
    assert!(matches!(host.0, SolveFailureKind::DegenerateDiagonal { param: 4, fault: DiagonalFault::Zero }), "{:?}", host.0);
    assert_eq!((host.1, host.2), (1, 1));
    for d in devices() {
        let name = label(&d);
        let device = run(Some(&d));
        assert_eq!(format!("{:?}", device.0), format!("{:?}", host.0), "{name}");
        assert_eq!((device.1, device.2), (host.1, host.2), "{name}");
        assert_close(&device.3, &host.3, 1e-12, &format!("{name}: the partial state"));
    }
}

/// `GpuDevice::shared` hands every caller the same device state: a later
/// call finds the memory a solve through an earlier one left, a separately
/// created device has none.
#[test]
fn a_shared_device_is_one_state_per_process() {
    use arael::simple_lm::{LmConfig, SparseFaer};
    let backend = devices()[0].backend;
    let (cameras, points, observations) = consistent_scene::<f64>();
    let mut s = Scene { cameras, points, observations };
    let opts = options(Some(&GpuDevice::shared(backend, 0)), true);
    s.solve_with(&mut SparseFaer::<f64>::from_options(&opts), &LmConfig::<f64>::default()).unwrap();
    assert!(GpuDevice::shared(backend, 0).reserved_bytes() > 0);
    assert_eq!(devices()[0].reserved_bytes(), 0);
}

/// Solve after solve through one device, each with a solver of its own or
/// through one session, on scenes of other structures a few percent smaller
/// or larger than the first: the device allocates for the first and reuses
/// its memory for the rest.
#[test]
fn solves_of_slightly_changed_structures_allocate_nothing() {
    use arael::simple_lm::{LmConfig, LmSession, SparseFaer};
    let sizes = [(8usize, 80usize, 3u64), (8, 84, 4), (8, 77, 5), (8, 82, 6), (8, 80, 7)];
    let cfg = LmConfig::<f64> { max_iters: 8, ..Default::default() };
    for d in devices() {
        let name = label(&d);
        let mut made = Vec::new();
        for &(nc, np, seed) in &sizes {
            let (cameras, points, observations) = consistent_scene_of::<f64>(nc, np, seed);
            let mut s = Scene { cameras, points, observations };
            let mut solver = SparseFaer::<f64>::from_options(&options(Some(&d), true));
            assert!(s.solve_with(&mut solver, &cfg).unwrap().end_cost.is_finite(), "{name}");
            made.push(d.buffers_created());
        }
        assert!(made[0] > 0 && made.iter().all(|&m| m == made[0]), "{name}: buffers after each solve {made:?}");
        // a session told of each structural change, its solver kept
        let mut session = LmSession::new(SparseFaer::<f64>::from_options(&options(Some(&d), true)));
        for &(nc, np, seed) in &sizes {
            let (cameras, points, observations) = consistent_scene_of::<f64>(nc, np, seed + 10);
            let mut s = Scene { cameras, points, observations };
            session.invalidate();
            assert!(session.solve(&mut s, &cfg).unwrap().end_cost.is_finite(), "{name}");
            assert_eq!(d.buffers_created(), made[0], "{name}: a session's solve of another structure");
        }
    }
}

/// With a device assembly the first linearization is assembled on the
/// device too (the CPU only brings the model's working copies to the
/// start, which the layout reads), and the solve still lands on the CPU
/// route's optimum.
#[test]
fn the_first_linearization_is_assembled_on_the_device() {
    use arael::simple_lm::{LmConfig, SparseFaer};
    let run = |gpu: Option<&GpuDevice>, verbose: bool| {
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let cfg = LmConfig::<f64> { max_iters: 50, verbose, ..Default::default() };
        s.solve_with(&mut SparseFaer::<f64>::from_options(&options(gpu, true)), &cfg).unwrap()
    };
    let cpu = run(None, false);
    for d in devices() {
        let lines: std::sync::Arc<std::sync::Mutex<Vec<String>>> = Default::default();
        let sink = lines.clone();
        arael::log::set_sink(move |_, m| sink.lock().unwrap().push(m.to_string()));
        let device = run(Some(&d), true);
        arael::log::reset_sink();
        let lines = lines.lock().unwrap();
        let name = label(&d);
        assert!(lines.iter().any(|l| l.starts_with("gpu:") && l.contains("its first assembly")), "{name}: {lines:#?}");
        assert!(lines.iter().any(|l| l.starts_with(&format!("gpu schur ({name},"))), "{name}: {lines:#?}");
        assert!((device.start_cost - cpu.start_cost).abs() <= 1e-12 * cpu.start_cost, "{name}");
        assert!((device.end_cost - cpu.end_cost).abs() <= 1e-9 * cpu.end_cost, "{name}");
    }
}

/// A device: CUDA when it is compiled in, else HIP.
#[cfg(feature = "gpu-cuda")]
fn device() -> std::sync::Arc<arael::__gpu::cuda::CudaDevice> {
    std::sync::Arc::new(arael::__gpu::cuda::CudaDevice::new(0).unwrap())
}
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
fn device() -> std::sync::Arc<arael::__gpu::hip::HipDevice> {
    std::sync::Arc::new(arael::__gpu::hip::HipDevice::new(0).unwrap())
}

/// The device assembly matches the CPU's in both wide scalars: in df32 the
/// sums keep about 48 bits, against f64's 53. Some cameras see a point
/// twice, two observations with cross blocks of their own on one tile of H,
/// which the device sums as one.
#[test]
fn the_f64_bundle_adjustment_assembles_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let (_, _, observations) = scene::<f64>();
    let pairs: std::collections::HashSet<(u32, u32)> = observations.iter().map(|o| (o.cam.index(), o.pt.index())).collect();
    assert!(pairs.len() < observations.len(), "the scene observes some pair twice");
    for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
        let (cameras, points, observations) = scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let x1 = perturbed(&x);
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!(chk.values > 0);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
    }
}

#[test]
fn the_f32_bundle_adjustment_assembles_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    for wide in [Wide::F64, Wide::Df32] {
        let (cameras, points, observations) = scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        assert_eq!(LmProblemInternals::<f32>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let x1 = perturbed(&x);
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        // the CPU sums its f32 products in f32, the device in its wide scalar
        assert!((chk.device_cost - chk.cpu_cost).abs() <= 1e-5 * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < 1e-4 && chk.hess_err < 1e-4, "{wide:?}: {chk:?}");
    }
}

/// A solver with its GPU state moves to another thread, as a CPU one does,
/// and solves there.
#[test]
fn a_gpu_solver_moves_between_threads() {
    use arael::simple_lm::{LmConfig, SparseFaer};
    fn send<S: Send>(_: &S) {}
    for d in devices() {
        let mut solver = SparseFaer::<f64>::from_options(&options(Some(&d), true));
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        s.solve_with(&mut solver, &LmConfig::<f64>::default()).unwrap();
        send(&solver);
        let cost = std::thread::spawn(move || s.solve_with(&mut solver, &LmConfig::<f64>::default()).unwrap().end_cost).join().unwrap();
        assert!(cost.is_finite(), "{}", label(&d));
    }
}

/// A 2D pose graph (benchmarks/pgo's): poses of a position and a heading,
/// relative-pose edges, and a prior on one pose, a root-level constraint of
/// one entity written into its own block.
#[arael::model]
#[derive(Clone)]
pub struct Pose2D<T: Float> {
    pub pos: Param<vect2<T>>,
    pub th: Param<T>,
    pub hb: SelfBlock<Pose2D<T>, T>,
}

#[arael::model]
#[arael(constraint(p.hb, {
    [p.pos.x - prior2d.pos.x, p.pos.y - prior2d.pos.y, p.th - prior2d.th]
}))]
#[derive(Clone)]
pub struct Prior2D<T: Float> {
    #[arael(ref = root.poses)]
    pub p: Ref<Pose2D<T>>,
    pub pos: vect2<T>,
    pub th: T,
}

#[arael::model]
#[arael(constraint(hb, {
    let c = cos(a.th);
    let s = sin(a.th);
    let dx = b.pos.x - a.pos.x;
    let dy = b.pos.y - a.pos.y;
    [(c * dx + s * dy - edge2d.dx) * edge2d.w,
     (c * dy - s * dx - edge2d.dy) * edge2d.w,
     rad_diff(b.th - a.th, edge2d.dth) * edge2d.w]
}))]
#[derive(Clone)]
pub struct Edge2D<T: Float> {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose2D<T>>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose2D<T>>,
    pub dx: T,
    pub dy: T,
    pub dth: T,
    pub w: T,
    pub hb: CrossBlock<Pose2D<T>, Pose2D<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Graph2D {
    pub poses: refs::Vec<Pose2D<f64>>,
    pub edges: std::vec::Vec<Edge2D<f64>>,
    pub prior: Option<Prior2D<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct Graph2DF {
    pub poses: refs::Vec<Pose2D<f32>>,
    pub edges: std::vec::Vec<Edge2D<f32>>,
    pub prior: Option<Prior2D<f32>>,
}

type Graph2DParts<T> = (refs::Vec<Pose2D<T>>, std::vec::Vec<Edge2D<T>>, Option<Prior2D<T>>);

/// A loop of `n` poses on a circle, odometry between neighbours and a loop
/// closure every fifth pose, the measurements the truth plus noise, the
/// poses moved off it; the prior holds the first pose where it starts.
fn pose_graph<T: Float>(n: usize, seed: u64) -> Graph2DParts<T> {
    let mut rng = Lcg(seed);
    let c = |v: f64| T::from(v).unwrap();
    let truth: Vec<(f64, f64, f64)> = (0..n)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            (5.0 * a.cos(), 5.0 * a.sin(), a + std::f64::consts::FRAC_PI_2)
        })
        .collect();
    let mut poses = refs::Vec::new();
    for (k, &(x, y, th)) in truth.iter().enumerate() {
        let off = if k == 0 { 0.0 } else { 1.0 };
        poses.push(Pose2D {
            pos: Param::new(vect2::new(c(x + off * 0.1 * rng.uniform()), c(y + off * 0.1 * rng.uniform()))),
            th: Param::new(c(th + off * 0.05 * rng.uniform())),
            hb: SelfBlock::new(),
        });
    }
    let mut edges = std::vec::Vec::new();
    let mut edge = |i: usize, j: usize, rng: &mut Lcg| {
        let ((xa, ya, ta), (xb, yb, tb)) = (truth[i], truth[j]);
        let (dx, dy) = (xb - xa, yb - ya);
        edges.push(Edge2D {
            a: poses.ref_at(i),
            b: poses.ref_at(j),
            dx: c(ta.cos() * dx + ta.sin() * dy + 0.01 * rng.uniform()),
            dy: c(ta.cos() * dy - ta.sin() * dx + 0.01 * rng.uniform()),
            dth: c(tb - ta + 0.005 * rng.uniform()),
            w: c(10.0),
            hb: CrossBlock::new(),
        });
    };
    for k in 0..n {
        edge(k, (k + 1) % n, &mut rng);
        if k % 5 == 0 {
            edge(k, (k + n / 2) % n, &mut rng);
        }
    }
    let (x0, y0, t0) = truth[0];
    let prior = Some(Prior2D { p: poses.ref_at(0), pos: vect2::new(c(x0), c(y0)), th: c(t0) });
    (poses, edges, prior)
}

/// The whole-system supernodal route, on `gpu` when it is given.
fn whole_options(gpu: Option<&GpuDevice>) -> arael::simple_lm::SparseFaerOptions {
    arael::simple_lm::SparseFaerOptions::default()
        .with_policy(arael::simple_lm::SchurPolicy::Never)
        .with_block_supernodal(arael::simple_lm::BlockSupernodalMode::Always)
        .with_gpu(gpu.cloned())
}

/// A pose graph with a prior, a root-level constraint of one entity, has a
/// device assembly, and it matches the CPU's, in f64 and f32 and both wide
/// scalars. Some pairs of poses are joined both ways, two edges of opposite
/// orientation on one cross tile of H.
#[test]
fn a_pose_graph_with_a_prior_assembles_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let (_, edges, _) = pose_graph::<f64>(40, 3);
    let pairs: std::collections::HashSet<(u32, u32)> = edges.iter().map(|e| (e.a.index(), e.b.index())).collect();
    assert!(pairs.iter().any(|&(a, b)| pairs.contains(&(b, a))), "the graph joins some pair both ways");
    for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
        let (poses, edges, prior) = pose_graph::<f64>(40, 3);
        let mut s = Graph2D { poses, edges, prior };
        assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let x1 = perturbed(&x);
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!(chk.values > 0);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        let (poses, edges, prior) = pose_graph::<f32>(40, 3);
        let mut s = Graph2DF { poses, edges, prior };
        assert_eq!(LmProblemInternals::<f32>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let x1 = perturbed(&x);
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= 1e-5 * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < 1e-4 && chk.hess_err < 1e-4, "{wide:?}: {chk:?}");
    }
}

/// The pose graph's whole-system solve through the device assembly reaches
/// the CPU route's optimum.
#[test]
fn a_pose_graph_solves_through_the_device_assembly_as_on_the_cpu() {
    let run = |gpu: Option<&GpuDevice>| {
        let (poses, edges, prior) = pose_graph::<f64>(40, 3);
        let mut s = Graph2D { poses, edges, prior };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        let plan = solver.plan().expect("a plan");
        assert!(!plan.reduced && plan.gpu_factor == gpu.is_some());
        assert_eq!(plan.gpu_assembly, gpu.is_some());
        (r.start_cost, r.end_cost)
    };
    let (start, cpu) = run(None);
    assert!(cpu < 0.05 * start, "the graph converges to its noise: {start} to {cpu}");
    for d in devices() {
        let (_, device) = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
    }
}

/// The pose graph of benchmarks/pgo, the heading an `AngleParam`: the
/// constraints read its rotation matrix, whose sin and cos the CPU caches
/// per iteration and the device computes from the angle.
#[arael::model]
#[derive(Clone)]
pub struct PoseA<T: Float> {
    pub pos: Param<vect2<T>>,
    pub rot: AngleParam<T>,
    pub hb: SelfBlock<PoseA<T>, T>,
}

#[arael::model]
#[arael(constraint(p.hb, {
    [p.pos.x - priora.pos.x, p.pos.y - priora.pos.y, p.rot.angle - priora.th]
}))]
#[derive(Clone)]
pub struct PriorA<T: Float> {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseA<T>>,
    pub pos: vect2<T>,
    pub th: T,
}

#[arael::model]
#[arael(constraint(hb, {
    let local = b.rot.rotation_matrix.transpose()
        * (a.pos + a.rot.rotation_matrix * edgea.delta - b.pos);
    [local.x * edgea.wt,
     local.y * edgea.wt,
     rad_diff(a.rot.angle + edgea.dth, b.rot.angle) * edgea.wr]
}))]
#[derive(Clone)]
pub struct EdgeA<T: Float> {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseA<T>>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseA<T>>,
    pub delta: vect2<T>,
    pub dth: T,
    pub wt: T,
    pub wr: T,
    pub hb: CrossBlock<PoseA<T>, PoseA<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct GraphA {
    pub poses: refs::Vec<PoseA<f64>>,
    pub edges: std::vec::Vec<EdgeA<f64>>,
    pub prior: Option<PriorA<f64>>,
}

/// [`pose_graph`] with `AngleParam` headings.
fn angle_graph(n: usize, seed: u64) -> GraphA {
    let (poses, edges, prior) = pose_graph::<f64>(n, seed);
    let mut pa = refs::Vec::new();
    for p in poses.iter() {
        pa.push(PoseA { pos: p.pos.clone(), rot: AngleParam::new(p.th.value), hb: SelfBlock::new() });
    }
    let edges = edges
        .iter()
        .map(|e| {
            // the edge's delta in a's frame and the heading change, as pgo's
            EdgeA {
                a: pa.ref_at(e.a.index() as usize),
                b: pa.ref_at(e.b.index() as usize),
                delta: vect2::new(e.dx, e.dy),
                dth: e.dth,
                wt: e.w,
                wr: e.w,
                hb: CrossBlock::new(),
            }
        })
        .collect();
    let prior = prior.map(|q| PriorA { p: pa.ref_at(q.p.index() as usize), pos: q.pos, th: q.th });
    GraphA { poses: pa, edges, prior }
}

/// The `AngleParam` pose graph assembles on the device as on the CPU at a
/// point away from the one its caches were computed for, and solves there
/// to the CPU's optimum.
#[test]
fn an_angle_pose_graph_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let mut s = angle_graph(40, 3);
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
    }
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = angle_graph(40, 3);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        r.end_cost
    };
    let cpu = run(None);
    for d in devices() {
        let device = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
    }
}

/// The 2D pose graph with its poses' blocks stored at their live sizes
/// (`VarSelfBlock`), one pose's position fixed and one pose fixed whole.
#[arael::model]
#[derive(Clone)]
pub struct PoseV<T: Float> {
    pub pos: Param<vect2<T>>,
    pub th: Param<T>,
    pub hb: VarSelfBlock<PoseV<T>, T>,
}

#[arael::model]
#[arael(constraint(p.hb, {
    [p.pos.x - priorv.pos.x, p.pos.y - priorv.pos.y, p.th - priorv.th]
}))]
#[derive(Clone)]
pub struct PriorV<T: Float> {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseV<T>>,
    pub pos: vect2<T>,
    pub th: T,
}

#[arael::model]
#[arael(constraint(hb, {
    let c = cos(a.th);
    let s = sin(a.th);
    let dx = b.pos.x - a.pos.x;
    let dy = b.pos.y - a.pos.y;
    [(c * dx + s * dy - edgev.dx) * edgev.w,
     (c * dy - s * dx - edgev.dy) * edgev.w,
     rad_diff(b.th - a.th, edgev.dth) * edgev.w]
}))]
#[derive(Clone)]
pub struct EdgeV<T: Float> {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseV<T>>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseV<T>>,
    pub dx: T,
    pub dy: T,
    pub dth: T,
    pub w: T,
    pub hb: CrossBlock<PoseV<T>, PoseV<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct GraphV {
    pub poses: refs::Vec<PoseV<f64>>,
    pub edges: std::vec::Vec<EdgeV<f64>>,
    pub prior: Option<PriorV<f64>>,
}

/// [`pose_graph`] over `PoseV`, pose 7's position and pose 11 fixed.
fn var_graph(n: usize, seed: u64) -> GraphV {
    let (poses, edges, prior) = pose_graph::<f64>(n, seed);
    let mut pv = refs::Vec::new();
    for (k, p) in poses.iter().enumerate() {
        pv.push(PoseV {
            pos: if k == 7 || k == 11 { Param::fixed(p.pos.value) } else { p.pos.clone() },
            th: if k == 11 { Param::fixed(p.th.value) } else { p.th.clone() },
            hb: VarSelfBlock::new(),
        });
    }
    let edges = edges
        .iter()
        .map(|e| EdgeV {
            a: pv.ref_at(e.a.index() as usize),
            b: pv.ref_at(e.b.index() as usize),
            dx: e.dx,
            dy: e.dy,
            dth: e.dth,
            w: e.w,
            hb: CrossBlock::new(),
        })
        .collect();
    let prior = prior.map(|q| PriorV { p: pv.ref_at(q.p.index() as usize), pos: q.pos, th: q.th });
    GraphV { poses: pv, edges, prior }
}

/// Entities with blocks at their live sizes, some parameters fixed,
/// assemble on the device as on the CPU and solve there to the CPU's
/// optimum.
#[test]
fn a_var_self_block_pose_graph_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let mut s = var_graph(40, 3);
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
    let mut x = Vec::new();
    s.serialize(&mut x);
    assert_eq!(x.len(), 3 * 40 - 2 - 3);
    let x1 = perturbed(&x);
    for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
    }
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = var_graph(40, 3);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        r.end_cost
    };
    let cpu = run(None);
    for d in devices() {
        let device = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
    }
}

/// A 3D pose graph (benchmarks/pgo's): poses of a position and a
/// `QuaternionParam` rotation, the first held by a prior written as the
/// pose's own constraint behind a guard over its data, relative-pose edges
/// with the quaternion-vector rotation residual.
#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    let rerr = pose3.prior_rot_t * pose3.ea.rotation_matrix();
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    [pose3.pos.x - pose3.prior.x,
     pose3.pos.y - pose3.prior.y,
     pose3.pos.z - pose3.prior.z,
     s[2].y / denom,
     s[0].z / denom,
     s[1].x / denom]
}))]
#[derive(Clone)]
pub struct Pose3 {
    pub pos: Param<vect3<f64>>,
    pub ea: QuaternionParam<f64>,
    pub prior: vect3<f64>,
    pub prior_rot_t: arael::matrix::matrix3<f64>,
    pub has_prior: bool,
    pub hb: SelfBlock<Pose3>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.ea.rotation_matrix();
    let rerr = edge3.rmeas_t * (ra.transpose() * b.ea.rotation_matrix());
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let terr = ra.transpose() * (b.pos - a.pos) - edge3.dt;
    [terr.x * edge3.wt, terr.y * edge3.wt, terr.z * edge3.wt,
     s[2].y / denom * edge3.wr, s[0].z / denom * edge3.wr, s[1].x / denom * edge3.wr]
}))]
#[derive(Clone)]
pub struct Edge3 {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose3>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose3>,
    pub dt: vect3<f64>,
    pub rmeas_t: arael::matrix::matrix3<f64>,
    pub wt: f64,
    pub wr: f64,
    pub hb: CrossBlock<Pose3, Pose3>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Graph3 {
    pub poses: refs::Vec<Pose3>,
    pub edges: std::vec::Vec<Edge3>,
}

/// A loop of `n` poses on a tilted circle, each turned along it,
/// odometry between neighbours and a closure every fifth pose, the
/// measurements the truth plus noise, the poses moved off it.
fn pose_graph3(n: usize, seed: u64) -> Graph3 {
    use arael::matrix::matrix3d;
    use arael::vect::vect3d;
    let mut rng = Lcg(seed);
    let truth: Vec<(vect3d, matrix3d)> = (0..n)
        .map(|k| {
            let a = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            (vect3d::new(5.0 * a.cos(), 5.0 * a.sin(), a.sin()), matrix3d::rotation_from_euler_angles(vect3d::new(0.1 * a.cos(), 0.2 * a.sin(), a)))
        })
        .collect();
    let mut poses = refs::Vec::new();
    for (k, (t, r)) in truth.iter().enumerate() {
        let off = if k == 0 { 0.0 } else { 1.0 };
        let dr = matrix3d::rotation_from_euler_angles(vect3d::new(0.03 * rng.uniform(), 0.03 * rng.uniform(), 0.03 * rng.uniform()) * off);
        poses.push(Pose3 {
            pos: Param::new(*t + vect3d::new(0.1 * rng.uniform(), 0.1 * rng.uniform(), 0.1 * rng.uniform()) * off),
            ea: QuaternionParam::from_rotation_matrix(*r * dr),
            prior: *t,
            prior_rot_t: r.transpose(),
            has_prior: k == 0,
            hb: SelfBlock::new(),
        });
    }
    let mut edges = std::vec::Vec::new();
    let mut edge = |i: usize, j: usize, rng: &mut Lcg| {
        let ((ta, ra), (tb, rb)) = (truth[i], truth[j]);
        let noise = matrix3d::rotation_from_euler_angles(vect3d::new(0.005 * rng.uniform(), 0.005 * rng.uniform(), 0.005 * rng.uniform()));
        edges.push(Edge3 {
            a: poses.ref_at(i),
            b: poses.ref_at(j),
            dt: ra.transpose() * (tb - ta) + vect3d::new(0.01 * rng.uniform(), 0.01 * rng.uniform(), 0.01 * rng.uniform()),
            rmeas_t: (ra.transpose() * rb * noise).transpose(),
            wt: 10.0,
            wr: 20.0,
            hb: CrossBlock::new(),
        });
    };
    for k in 0..n {
        edge(k, (k + 1) % n, &mut rng);
        if k % 5 == 0 {
            edge(k, (k + n / 2) % n, &mut rng);
        }
    }
    Graph3 { poses, edges }
}

/// The 3D pose graph assembles on the device as on the CPU, its prior
/// packed for the one pose its guard keeps, and solves there to the CPU's
/// optimum, the model's rotations where the CPU leaves them.
#[test]
fn a_3d_pose_graph_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let mut s = pose_graph3(40, 5);
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for (wide, tol) in [(Wide::F64, 1e-11), (Wide::Df32, 1e-10)] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
    }
    let state = |s: &Graph3| -> Vec<f64> {
        s.poses.iter().flat_map(|p| [p.pos.value.x, p.pos.value.y, p.pos.value.z, p.ea.value.t, p.ea.value.v.x, p.ea.value.v.y, p.ea.value.v.z]).collect()
    };
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = pose_graph3(40, 5);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        s.deserialize(&r.x);
        (r.start_cost, r.end_cost, state(&s))
    };
    let (start, cpu, cpu_state) = run(None);
    assert!(cpu < 0.05 * start, "the graph converges to its noise: {start} to {cpu}");
    for d in devices() {
        let (_, device, st) = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
        assert_close(&st, &cpu_state, 1e-7, &format!("{}: the solved model", label(&d)));
    }
}

/// The 2D pose graph with guards on all three forms: a heading prior on
/// the pose itself guarded by the pose's own parameter, a position prior
/// through a ref guarded by the root's data, edges guarded by their own
/// data. A guard does not change during a solve (docs/MODEL.md), so the
/// device evaluates each once and gives the instances it drops no row.
#[arael::model]
#[arael(constraint(hb, guard = self.th.value > 3.063, { [(poseg.th - poseg.heading) * 0.5] }))]
#[derive(Clone)]
pub struct PoseG {
    pub pos: Param<vect2<f64>>,
    pub th: Param<f64>,
    pub heading: f64,
    pub hb: SelfBlock<PoseG>,
}

#[arael::model]
#[arael(constraint(p.hb, guard = root.use_prior, {
    [p.pos.x - priorg.pos.x, p.pos.y - priorg.pos.y, p.th - priorg.th]
}))]
#[derive(Clone)]
pub struct PriorG {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseG>,
    pub pos: vect2<f64>,
    pub th: f64,
}

#[arael::model]
#[arael(constraint(hb, guard = self.on, {
    let c = cos(a.th);
    let s = sin(a.th);
    let dx = b.pos.x - a.pos.x;
    let dy = b.pos.y - a.pos.y;
    [(c * dx + s * dy - edgeg.dx) * edgeg.w,
     (c * dy - s * dx - edgeg.dy) * edgeg.w,
     rad_diff(b.th - a.th, edgeg.dth) * edgeg.w * root.edge_scale]
}))]
#[derive(Clone)]
pub struct EdgeG {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseG>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseG>,
    pub dx: f64,
    pub dy: f64,
    pub dth: f64,
    pub w: f64,
    pub on: bool,
    pub hb: CrossBlock<PoseG, PoseG>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct GraphG {
    pub poses: refs::Vec<PoseG>,
    pub edges: std::vec::Vec<EdgeG>,
    pub priors: std::vec::Vec<PriorG>,
    pub use_prior: bool,
    /// Read by every edge's body: one value in the root's record.
    pub edge_scale: f64,
}

/// [`pose_graph`] with guards: the closures every tenth edge off, the
/// heading prior on the poses whose heading starts above 3.063 (half way
/// between two of the loop's headings, so no pose crosses it in a solve).
fn guarded_graph(n: usize, seed: u64, use_prior: bool) -> GraphG {
    let (poses, edges, prior) = pose_graph::<f64>(n, seed);
    let mut pg = refs::Vec::new();
    for p in poses.iter() {
        pg.push(PoseG { pos: p.pos.clone(), th: p.th.clone(), heading: p.th.value + 0.01, hb: SelfBlock::new() });
    }
    let edges = edges
        .iter()
        .enumerate()
        .map(|(k, e)| EdgeG {
            a: pg.ref_at(e.a.index() as usize),
            b: pg.ref_at(e.b.index() as usize),
            dx: e.dx,
            dy: e.dy,
            dth: e.dth,
            w: e.w,
            on: k % 10 != 3,
            hb: CrossBlock::new(),
        })
        .collect();
    let priors = prior.iter().map(|q| PriorG { p: pg.ref_at(q.p.index() as usize), pos: q.pos, th: q.th }).collect();
    GraphG { poses: pg, edges, priors, use_prior, edge_scale: 0.8 }
}

/// Guards over a parameter, the root's data and an instance's data: the
/// device keeps the instances the CPU keeps, assembles as the CPU does and
/// solves to its optimum.
#[test]
fn guards_on_every_form_keep_on_the_device_what_they_keep_on_the_cpu() {
    use arael::__gpu::Wide;
    let s0 = guarded_graph(40, 3, true);
    let headed = s0.poses.iter().filter(|p| p.th.value > 3.063).count();
    assert!(headed > 0 && headed < 40, "some poses have the heading prior: {headed}");
    assert!(s0.edges.iter().any(|e| !e.on));
    // the heading in the pose's record, the edge scale once in the root's
    let shape = arael::__gpu::DeviceRoot::<f64>::device_shape(&s0);
    assert_eq!((shape.entity_data, shape.root_data), (vec![16], 16));
    for use_prior in [true, false] {
        let mut s = guarded_graph(40, 3, use_prior);
        assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let x1 = perturbed(&x);
        for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{use_prior} {wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{use_prior} {wide:?}: {chk:?}");
        }
    }
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = guarded_graph(40, 3, true);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        r.end_cost
    };
    let cpu = run(None);
    for d in devices() {
        let device = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
    }
}

/// The 2D pose graph with a robust loss on every form: a Cauchy heading
/// prior on the pose itself, a soft-L1 position prior through a ref at the
/// root's scale, Huber odometry and Tukey loop closures at their own
/// scales, and Geman-McClure observations of landmarks, nested in them
/// (`parent = lm`).
#[arael::model]
#[arael(constraint(hb, loss = |s| loss_cauchy(s, 0.04), { [(posel.th - posel.heading) * 2.0] }))]
#[derive(Clone)]
pub struct PoseL {
    pub pos: Param<vect2<f64>>,
    pub th: Param<f64>,
    pub heading: f64,
    pub hb: SelfBlock<PoseL>,
}

#[arael::model]
#[arael(constraint(p.hb, loss = |s| loss_soft_l1(s, root.prior_k2), {
    [p.pos.x - priorl.pos.x, p.pos.y - priorl.pos.y, p.th - priorl.th]
}))]
#[derive(Clone)]
pub struct PriorL {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseL>,
    pub pos: vect2<f64>,
    pub th: f64,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_huber(s, odol.k2), {
    let cs = cos(a.th);
    let sn = sin(a.th);
    let dx = b.pos.x - a.pos.x;
    let dy = b.pos.y - a.pos.y;
    [(cs * dx + sn * dy - odol.dx) * 10.0, (cs * dy - sn * dx - odol.dy) * 10.0, rad_diff(b.th - a.th, odol.dth) * 10.0]
}))]
#[derive(Clone)]
pub struct OdoL {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseL>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseL>,
    pub dx: f64,
    pub dy: f64,
    pub dth: f64,
    pub k2: f64,
    pub hb: CrossBlock<PoseL, PoseL>,
}

#[arael::model]
#[arael(constraint(hb, loss = |s| loss_tukey(s, loopl.k2), {
    let cs = cos(a.th);
    let sn = sin(a.th);
    let dx = b.pos.x - a.pos.x;
    let dy = b.pos.y - a.pos.y;
    [(cs * dx + sn * dy - loopl.dx) * 10.0, (cs * dy - sn * dx - loopl.dy) * 10.0, rad_diff(b.th - a.th, loopl.dth) * 10.0]
}))]
#[derive(Clone)]
pub struct LoopL {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseL>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseL>,
    pub dx: f64,
    pub dy: f64,
    pub dth: f64,
    pub k2: f64,
    pub hb: CrossBlock<PoseL, PoseL>,
}

#[arael::model]
#[derive(Clone)]
pub struct LmL {
    pub pos: Param<vect2<f64>>,
    pub sights: std::vec::Vec<SightL>,
    pub hb: SelfBlock<LmL>,
}

#[arael::model]
#[arael(constraint(hb, parent = lm, loss = |s| loss_geman_mcclure(s, 1.0), {
    let cs = cos(pose.th);
    let sn = sin(pose.th);
    let dx = lm.pos.x - pose.pos.x;
    let dy = lm.pos.y - pose.pos.y;
    [(cs * dx + sn * dy - sightl.x) * 10.0, (cs * dy - sn * dx - sightl.y) * 10.0]
}))]
#[derive(Clone)]
pub struct SightL {
    #[arael(ref = root.poses)]
    pub pose: Ref<PoseL>,
    pub x: f64,
    pub y: f64,
    pub hb: CrossBlock<LmL, PoseL>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct GraphL {
    pub poses: refs::Vec<PoseL>,
    pub landmarks: refs::Vec<LmL>,
    pub odometry: std::vec::Vec<OdoL>,
    pub loops: std::vec::Vec<LoopL>,
    pub priors: std::vec::Vec<PriorL>,
    pub prior_k2: f64,
}

/// [`pose_graph`] under losses: every third loop closure and the odometry
/// out of pose 7 metres off, far past their scales, as is every fourth
/// landmark's first sighting; the landmarks on a ring outside the poses',
/// each seen by the poses within a twelfth of a turn of it.
fn lossy_graph(n: usize, m: usize, seed: u64) -> GraphL {
    let (poses, edges, prior) = pose_graph::<f64>(n, seed);
    let mut rng = Lcg(seed + 1);
    let mut pl = refs::Vec::new();
    for p in poses.iter() {
        pl.push(PoseL { pos: p.pos.clone(), th: p.th.clone(), heading: p.th.value + 0.01, hb: SelfBlock::new() });
    }
    let (mut odometry, mut loops) = (std::vec::Vec::new(), std::vec::Vec::new());
    for e in edges.iter() {
        let (a, b) = (e.a.index() as usize, e.b.index() as usize);
        if b == (a + 1) % n {
            let off = if a == 7 { 1.0 } else { 0.0 };
            odometry.push(OdoL { a: pl.ref_at(a), b: pl.ref_at(b), dx: e.dx + off, dy: e.dy, dth: e.dth, k2: 0.25, hb: CrossBlock::new() });
        } else {
            let off = if loops.len() % 3 == 1 { 1.0 } else { 0.0 };
            loops.push(LoopL { a: pl.ref_at(a), b: pl.ref_at(b), dx: e.dx + off, dy: e.dy - off, dth: e.dth, k2: 25.0, hb: CrossBlock::new() });
        }
    }
    let mut landmarks = refs::Vec::new();
    for j in 0..m {
        let a = 2.0 * std::f64::consts::PI * (j as f64 + 0.5) / m as f64;
        let l = (7.0 * a.cos(), 7.0 * a.sin());
        let mut sights = std::vec::Vec::new();
        for k in 0..n {
            let t = 2.0 * std::f64::consts::PI * k as f64 / n as f64;
            let turn = (t - a).rem_euclid(2.0 * std::f64::consts::PI);
            if turn.min(2.0 * std::f64::consts::PI - turn) > std::f64::consts::PI / 6.0 {
                continue;
            }
            let (x, y, th) = (5.0 * t.cos(), 5.0 * t.sin(), t + std::f64::consts::FRAC_PI_2);
            let (dx, dy) = (l.0 - x, l.1 - y);
            let off = if j % 4 == 0 && sights.is_empty() { 1.0 } else { 0.0 };
            sights.push(SightL {
                pose: pl.ref_at(k),
                x: th.cos() * dx + th.sin() * dy + 0.01 * rng.uniform() + off,
                y: th.cos() * dy - th.sin() * dx + 0.01 * rng.uniform(),
                hb: CrossBlock::new(),
            });
        }
        landmarks.push(LmL { pos: Param::new(vect2::new(l.0 + 0.1 * rng.uniform(), l.1 + 0.1 * rng.uniform())), sights, hb: SelfBlock::new() });
    }
    let priors = prior.iter().map(|q| PriorL { p: pl.ref_at(q.p.index() as usize), pos: q.pos, th: q.th }).collect();
    GraphL { poses: pl, landmarks, odometry, loops, priors, prior_k2: 0.01 }
}

/// Losses on every form, some residuals far in their tails (Tukey's weight
/// zero there): the device assembles as the CPU does and solves to its
/// optimum.
#[test]
fn losses_on_every_form_assemble_and_solve_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let mut s = lossy_graph(40, 12, 3);
    assert!(s.landmarks.iter().all(|l| l.sights.len() >= 2));
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
    }
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = lossy_graph(40, 12, 3);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        s.deserialize(&r.x);
        let st: Vec<f64> = s.poses.iter().flat_map(|p| [p.pos.value.x, p.pos.value.y, p.th.value]).collect();
        (r.start_cost, r.end_cost, st)
    };
    let (start, cpu, cpu_state) = run(None);
    // the outliers keep about 38 of the cost: Tukey's c2 / 3 per closure,
    // Huber's 2 sqrt(k2 s) - k2, Geman-McClure's c2 s / (c2 + s)
    assert!(cpu < 40.0 && cpu < 0.6 * start, "the graph converges past its outliers: {start} to {cpu}");
    for d in devices() {
        let (_, device, st) = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
        assert_close(&st, &cpu_state, 1e-7, &format!("{}: the solved poses", label(&d)));
    }
}

/// The bundle-adjustment observation under a Huber loss of two pixels.
#[arael::model]
#[arael(constraint(hb, loss = |s| loss_huber(s, obsr.k2), {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let keep = -pc.z - cam.zdrop;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [branch(keep, with_wide(cam.intr.x * d * px - obsr.xy.x), 0.0),
     branch(keep, with_wide(cam.intr.x * d * py - obsr.xy.y), 0.0)]
}))]
#[derive(Clone)]
pub struct ObsR<T: Float> {
    #[arael(ref = root.cameras)]
    pub cam: Ref<Camera<T>>,
    #[arael(ref = root.points)]
    pub pt: Ref<Point<T>>,
    pub xy: vect2<T>,
    pub k2: T,
    pub hb: CrossBlock<Camera<T>, Point<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct SceneR {
    pub cameras: refs::Vec<Camera<f64>>,
    pub points: refs::Vec<Point<f64>>,
    pub observations: std::vec::Vec<ObsR<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct SceneRF {
    pub cameras: refs::Vec<Camera<f32>>,
    pub points: refs::Vec<Point<f32>>,
    pub observations: std::vec::Vec<ObsR<f32>>,
}

type RobustParts<T> = (refs::Vec<Camera<T>>, refs::Vec<Point<T>>, std::vec::Vec<ObsR<T>>);

/// [`consistent_scene`] under the Huber loss: every fifth observation
/// thirty pixels off, camera 4's observations of points less than five
/// units deep dropped by its depth guard (the loss skipped with the rows).
fn robust_scene<T: Float>() -> RobustParts<T> {
    let (mut cameras, points, observations) = consistent_scene::<T>();
    cameras[4usize].zdrop = T::from(5.0).unwrap();
    let observations = observations
        .into_iter()
        .enumerate()
        .map(|(k, o)| {
            let off = T::from(if k % 5 == 2 { 30.0 } else { 0.0 }).unwrap();
            ObsR { cam: o.cam, pt: o.pt, xy: vect2::new(o.xy.x + off, o.xy.y), k2: T::from(4.0).unwrap(), hb: CrossBlock::new() }
        })
        .collect();
    (cameras, points, observations)
}

/// A loss behind a guard of the body (the depth's `branch`): the device
/// assembles as the CPU does in f64 and f32, and solves to its optimum.
#[test]
fn a_robust_bundle_adjustment_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let (cameras, points, observations) = robust_scene::<f64>();
    let c4 = &cameras[4usize];
    let depths: Vec<f64> = observations
        .iter()
        .filter(|o| o.cam.index() == 4)
        .map(|o| -(arael::matrix::matrix3d::rotation_from_euler_angles(c4.ea.value) * points[o.pt].pos.value + c4.t.value).z)
        .collect();
    assert!(depths.iter().any(|&d| d < 5.0) && depths.iter().any(|&d| d > 5.0), "camera 4 keeps some observations: {depths:?}");
    for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
        let (cameras, points, observations) = robust_scene::<f64>();
        let mut s = SceneR { cameras, points, observations };
        assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let x1 = perturbed(&x);
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        let (cameras, points, observations) = robust_scene::<f32>();
        let mut s = SceneRF { cameras, points, observations };
        assert_eq!(LmProblemInternals::<f32>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let x1 = perturbed(&x);
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= 1e-5 * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < 1e-4 && chk.hess_err < 1e-4, "{wide:?}: {chk:?}");
    }
    let run = |gpu: Option<&GpuDevice>| {
        let (cameras, points, observations) = robust_scene::<f64>();
        let mut s = SceneR { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&options(gpu, true));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        (r.start_cost, r.end_cost)
    };
    let (start, cpu) = run(None);
    assert!(cpu < 0.5 * start, "the scene converges past its outliers: {start} to {cpu}");
    for d in devices() {
        let (_, device) = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
    }
}

/// A scale prior reading a `ScaledTransformParam`'s scale, which the CPU
/// computes per iteration from the parameters and the device does not
/// compute.
#[arael::model]
#[arael(constraint(hb, { [scaled.s.scale_factor - 2.0] }))]
#[derive(Clone)]
pub struct Scaled {
    pub s: arael::transform::ScaledTransformParam,
    pub hb: SelfBlock<Scaled>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct ScaledRoot {
    pub items: refs::Vec<Scaled>,
}

#[test]
fn a_constraint_reading_state_the_device_does_not_compute_has_no_device_form() {
    let s = ScaledRoot { items: refs::Vec::new() };
    let why = LmProblemInternals::<f64>::device_unsupported(&s).expect("a read of the scale has no device form");
    assert!(why.contains("Scaled") && why.contains("scaled.s"), "{why}");
}

/// The plane SLAM model of benchmarks/plane: `TransformParam` poses,
/// planes of a `UnitVecParam` normal and a distance, odometry and plane
/// observations.
#[arael::model]
#[derive(Clone)]
pub struct PoseT {
    pub r2w: arael::transform::TransformParam,
    pub hb: SelfBlock<PoseT>,
}

#[arael::model]
#[derive(Clone)]
pub struct PlaneT {
    pub n: arael::unitvec::UnitVecParam,
    pub c: Param<f64>,
    pub hb: SelfBlock<PlaneT>,
}

#[arael::model]
#[arael(constraint(hb, {
    let rel = a.r2w.inv() * b.r2w;
    let dt = rel.translation - odot.measured_translation;
    let dr = odot.measured_rotation_transposed * rel.rotation_matrix;
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x * odot.wt, dt.y * odot.wt, dt.z * odot.wt,
     (c2.z - c3.y) * 0.5 * odot.wr, (c3.x - c1.z) * 0.5 * odot.wr, (c1.y - c2.x) * 0.5 * odot.wr]
}, parent = odot))]
#[derive(Clone)]
pub struct OdoT {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseT>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseT>,
    pub measured_translation: vect3<f64>,
    pub measured_rotation_transposed: arael::matrix::matrix3<f64>,
    pub wt: f64,
    pub wr: f64,
    pub hb: CrossBlock<PoseT, PoseT>,
}

#[arael::model]
#[arael(constraint(hb, {
    let rp = p.r2w.rotation_matrix;
    let nw = l.n.unit;
    let nl = rp.transpose() * nw;
    let cl = l.c + p.r2w.translation * nw;
    let h = sqrt(nl.x * nl.x + nl.y * nl.y);
    let mx = nl * obst.measured_normal;
    let my = (obst.measured_normal.y * nl.x - obst.measured_normal.x * nl.y) / h;
    let mz = (obst.measured_normal.z * (nl.x * nl.x + nl.y * nl.y)
        - nl.z * (nl.x * obst.measured_normal.x + nl.y * obst.measured_normal.y)) / h;
    [atan2(my, mx) * obst.wa, atan2(mz, sqrt(mx * mx + my * my)) * obst.wa, (obst.measured_c - cl) * obst.wd]
}, parent = obst))]
#[derive(Clone)]
pub struct ObsT {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseT>,
    #[arael(ref = root.planes)]
    pub l: Ref<PlaneT>,
    pub measured_normal: vect3<f64>,
    pub measured_c: f64,
    pub wa: f64,
    pub wd: f64,
    pub hb: CrossBlock<PoseT, PlaneT>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct WorldT {
    pub poses: refs::Vec<PoseT>,
    pub planes: refs::Vec<PlaneT>,
    pub odos: std::vec::Vec<OdoT>,
    pub obs: std::vec::Vec<ObsT>,
}

/// `n` poses along a curve looking at 8 planes, odometry between
/// neighbours, each pose observing every plane, the measurements the truth
/// plus noise, the poses and planes moved off it. Pose 0 is fixed, pose 3
/// holds its rotation, plane 5 holds its normal.
fn plane_world(n: usize, seed: u64) -> WorldT {
    use arael::matrix::matrix3d;
    use arael::quatern::quatern;
    use arael::vect::vect3d;
    let mut rng = Lcg(seed);
    let truth: Vec<(vect3d, matrix3d)> = (0..n)
        .map(|k| {
            let a = 0.3 * k as f64;
            (vect3d::new(2.0 * a.cos(), 2.0 * a.sin(), 0.2 * a), matrix3d::rotation_from_euler_angles(vect3d::new(0.05 * a.sin(), 0.05 * a.cos(), a)))
        })
        .collect();
    let planes_truth: Vec<(vect3d, f64)> = (0..8)
        .map(|k| {
            let a = std::f64::consts::PI * k as f64 / 4.0;
            let nrm = vect3d::new(a.cos(), a.sin(), 0.3 * (k as f64 - 3.5) / 3.5);
            (nrm * (1.0 / (nrm * nrm).sqrt()), -6.0 - 0.5 * k as f64)
        })
        .collect();
    let mut poses = refs::Vec::new();
    for (k, (t, r)) in truth.iter().enumerate() {
        let off = if k == 0 { 0.0 } else { 1.0 };
        let dr = matrix3d::rotation_from_euler_angles(vect3d::new(0.02 * rng.uniform(), 0.02 * rng.uniform(), 0.02 * rng.uniform()) * off);
        let tt = *t + vect3d::new(0.05 * rng.uniform(), 0.05 * rng.uniform(), 0.05 * rng.uniform()) * off;
        let mut r2w = arael::transform::TransformParam::new(tt, quatern::from_rotation_matrix(*r * dr));
        if k == 0 {
            r2w = arael::transform::TransformParam::fixed(tt, quatern::from_rotation_matrix(*r));
        }
        if k == 3 {
            r2w.optimize_rotation = false;
        }
        poses.push(PoseT { r2w, hb: SelfBlock::new() });
    }
    let mut planes = refs::Vec::new();
    for (k, (nrm, c)) in planes_truth.iter().enumerate() {
        let moved = *nrm + vect3d::new(0.02 * rng.uniform(), 0.02 * rng.uniform(), 0.02 * rng.uniform());
        let n = if k == 5 { arael::unitvec::UnitVecParam::fixed(*nrm) } else { arael::unitvec::UnitVecParam::new(moved) };
        planes.push(PlaneT { n, c: Param::new(c + 0.05 * rng.uniform()), hb: SelfBlock::new() });
    }
    let mut odos = std::vec::Vec::new();
    for k in 1..n {
        let ((ta, ra), (tb, rb)) = (truth[k - 1], truth[k]);
        let noise = matrix3d::rotation_from_euler_angles(vect3d::new(0.002 * rng.uniform(), 0.002 * rng.uniform(), 0.002 * rng.uniform()));
        odos.push(OdoT {
            a: poses.ref_at(k - 1),
            b: poses.ref_at(k),
            measured_translation: ra.transpose() * (tb - ta) + vect3d::new(0.005 * rng.uniform(), 0.005 * rng.uniform(), 0.005 * rng.uniform()),
            measured_rotation_transposed: (ra.transpose() * rb * noise).transpose(),
            wt: 20.0,
            wr: 50.0,
            hb: CrossBlock::new(),
        });
    }
    let mut obs = std::vec::Vec::new();
    for (k, (t, r)) in truth.iter().enumerate() {
        for (j, (nrm, c)) in planes_truth.iter().enumerate() {
            let nl = r.transpose() * *nrm + vect3d::new(0.003 * rng.uniform(), 0.003 * rng.uniform(), 0.003 * rng.uniform());
            obs.push(ObsT {
                p: poses.ref_at(k),
                l: planes.ref_at(j),
                measured_normal: nl * (1.0 / (nl * nl).sqrt()),
                measured_c: c + *t * *nrm + 0.01 * rng.uniform(),
                wa: 30.0,
                wd: 10.0,
                hb: CrossBlock::new(),
            });
        }
    }
    WorldT { poses, planes, odos, obs }
}

/// The plane model assembles on the device as on the CPU, the poses' and
/// the normals' references kept there, and solves there to the CPU's
/// optimum, the model where the CPU leaves it.
#[test]
fn a_plane_world_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let mut s = plane_world(30, 9);
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for (wide, tol) in [(Wide::F64, 1e-11), (Wide::Df32, 1e-10)] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
    }
    let state = |s: &WorldT| -> Vec<f64> {
        let mut v = Vec::new();
        for p in s.poses.iter() {
            let (t, q) = (p.r2w.translation, p.r2w.rotation);
            v.extend([t.x, t.y, t.z, q.t, q.v.x, q.v.y, q.v.z]);
        }
        for l in s.planes.iter() {
            v.extend([l.n.unit.x, l.n.unit.y, l.n.unit.z, l.c.value]);
        }
        v
    };
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = plane_world(30, 9);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        s.deserialize(&r.x);
        (r.start_cost, r.end_cost, r.iterations, state(&s))
    };
    let (start, cpu, iters, cpu_state) = run(None);
    assert!(cpu < 0.05 * start && iters > 2, "the world converges to its noise: {start} to {cpu} in {iters} iterations");
    for d in devices() {
        let (_, device, _, st) = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
        assert_close(&st, &cpu_state, 1e-7, &format!("{}: the solved model", label(&d)));
    }
}

/// The device assembly's image staged in ordinary memory: the copies go
/// through the runtime's own staging, and a session's solves (the first
/// image, then values only) reach the CPU's optimum as with pinned memory.
#[test]
fn an_image_staged_in_pageable_memory_solves_as_a_pinned_one() {
    use arael::simple_lm::{GpuStaging, LmConfig, LmSession, SparseFaer};
    let cfg = LmConfig::<f64> { max_iters: 50, ..Default::default() };
    let cpu = |s: &Scene| {
        let mut s = s.clone();
        let r = s.solve_with(&mut SparseFaer::<f64>::from_options(&options(None, false)), &cfg).unwrap();
        (r.end_cost, state(&s))
    };
    for d in devices() {
        let d = d.with_staging(GpuStaging::Pageable);
        let name = label(&d);
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let mut session = LmSession::new(SparseFaer::<f64>::from_options(&options(Some(&d), true)));
        for round in 0..2 {
            let (cpu_cost, cpu_state) = cpu(&s);
            let r = session.solve(&mut s, &cfg).unwrap();
            assert!(session.solver().plan().unwrap().gpu_assembly);
            assert!((r.end_cost - cpu_cost).abs() <= 1e-9 * cpu_cost, "{name}, solve {round}: {} against the CPU's {cpu_cost}", r.end_cost);
            assert_close(&state(&s), &cpu_state, 1e-7, &format!("{name}, solve {round}: the solved model"));
            s.observations[5].xy.x += 3.0;
        }
    }
}

/// A covariance after a solve that assembled on the device: the solve bound
/// no positions into H, so the covariance binds them itself, under the
/// solve's pattern and symbolic analysis (none built again), and matches
/// the covariance after the CPU's solve.
#[test]
fn a_covariance_after_a_device_solve_reuses_its_analysis() {
    use arael::covariance::CovMode;
    let run = |gpu: Option<&GpuDevice>| {
        let (poses, edges, prior) = pose_graph::<f64>(40, 3);
        let mut s = Graph2D { poses, edges, prior };
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, covariance: Some(CovMode::PerQuery), ..Default::default() };
        let r = s.solve_with(&mut solver, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        let cov = r.covariance.as_ref().expect("the covariance");
        assert!(cov.took_block_route());
        assert_eq!(cov.plan().symbolics_built, 0, "the covariance factors under the solve's analysis");
        cov.marginal_cov(&s.poses[7]).unwrap()
    };
    let cpu = run(None);
    for d in devices() {
        let dev = run(Some(&d));
        let scale = cpu.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let diff = dev.iter().zip(cpu.iter()).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(diff <= 1e-7 * scale, "{}: the marginal covariance differs by {diff} of {scale}", label(&d));
    }
}

/// The SLAM model of benchmarks/slam: poses of a position and
/// `SimpleEulerAngleParam` angles with three constraints on themselves
/// reading root scalars, landmarks in an arena with a drift prior, bearing
/// observations nested in their landmark (`parent = lm`, the landmark and
/// the observing pose), odometry between poses.
#[arael::model]
#[arael(constraint(hb_pose, {
    let raw = spose.pos - spose.gps_pos;
    let rt = spose.gps_cov_r.transpose() * raw;
    [rt.x * spose.gps_isigma.x, rt.y * spose.gps_isigma.y, rt.z * spose.gps_isigma.z]
}))]
#[arael(constraint(hb_pose, {
    let pd = spose.pos - spose.prior_pos;
    let ed = spose.ea - spose.prior_ea;
    [pd.x * root.drift_pos_isigma, pd.y * root.drift_pos_isigma, pd.z * root.drift_pos_isigma,
     ed.x * root.drift_ea_isigma, ed.y * root.drift_ea_isigma, ed.z * root.drift_ea_isigma]
}))]
#[arael(constraint(hb_pose, {
    [(spose.ea.x - spose.tilt_roll) * root.tilt_isigma, (spose.ea.y - spose.tilt_pitch) * root.tilt_isigma]
}))]
#[derive(Clone)]
pub struct SPose {
    pub pos: Param<vect3<f64>>,
    pub ea: SimpleEulerAngleParam<f64>,
    pub prior_pos: vect3<f64>,
    pub prior_ea: vect3<f64>,
    pub gps_pos: vect3<f64>,
    pub gps_cov_r: arael::matrix::matrix3<f64>,
    pub gps_isigma: vect3<f64>,
    pub tilt_roll: f64,
    pub tilt_pitch: f64,
    pub hb_pose: SelfBlock<SPose>,
}

#[arael::model]
#[arael(constraint(hb_drift, {
    let d = slm.pos - slm.prior_pos;
    [d.x * root.drift_lm_isigma, d.y * root.drift_lm_isigma, d.z * root.drift_lm_isigma]
}))]
#[derive(Clone)]
pub struct SLm {
    pub pos: Param<vect3<f64>>,
    pub prior_pos: vect3<f64>,
    pub frines: std::vec::Vec<SFrine>,
    pub hb_drift: SelfBlock<SLm>,
}

#[arael::model]
#[arael(constraint(hb, parent = lm, {
    let mr2w = pose.ea.rotation_matrix();
    let lm_r = mr2w.transpose() * (lm.pos - pose.pos);
    let r_r = lm_r - sfrine.camera_pos;
    let r_f = sfrine.mf2r.transpose() * r_r;
    [atan2(r_f.y, r_f.x) * sfrine.isigma.x * root.frine_isigma_scale,
     atan2(r_f.z, r_f.x) * sfrine.isigma.y * root.frine_isigma_scale]
}))]
#[derive(Clone)]
pub struct SFrine {
    #[arael(ref = root.poses)]
    pub pose: Ref<SPose>,
    pub mf2r: arael::matrix::matrix3<f64>,
    pub camera_pos: vect3<f64>,
    pub isigma: vect2<f64>,
    pub hb: CrossBlock<SLm, SPose>,
}

#[arael::model]
#[arael(constraint(hb, {
    let mr2w_prev = prev.ea.rotation_matrix();
    let pos_diff = mr2w_prev.transpose() * (cur.pos - prev.pos);
    let pos_err = pos_diff - spair.delta_pos;
    let pos_w = spair.pos_cov_r.transpose() * pos_err;
    let mr2w_cur = cur.ea.rotation_matrix();
    let expected = mr2w_prev * spair.delta_ea.rotation_matrix();
    let error_rot = expected.transpose() * mr2w_cur;
    let ea_err = error_rot.get_euler_angles();
    let ea_w = spair.ea_cov_r.transpose() * ea_err;
    [pos_w.x * spair.pos_isigma.x, pos_w.y * spair.pos_isigma.y, pos_w.z * spair.pos_isigma.z,
     ea_w.x * spair.ea_isigma.x, ea_w.y * spair.ea_isigma.y, ea_w.z * spair.ea_isigma.z]
}))]
#[derive(Clone)]
pub struct SPair {
    #[arael(ref = root.poses)]
    pub prev: Ref<SPose>,
    #[arael(ref = root.poses)]
    pub cur: Ref<SPose>,
    pub delta_pos: vect3<f64>,
    pub delta_ea: vect3<f64>,
    pub pos_cov_r: arael::matrix::matrix3<f64>,
    pub pos_isigma: vect3<f64>,
    pub ea_cov_r: arael::matrix::matrix3<f64>,
    pub ea_isigma: vect3<f64>,
    pub hb: CrossBlock<SPose, SPose>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct SPath {
    pub poses: refs::Vec<SPose>,
    pub landmarks: refs::Arena<SLm>,
    pub pairs: std::vec::Vec<SPair>,
    pub drift_pos_isigma: f64,
    pub drift_ea_isigma: f64,
    pub drift_lm_isigma: f64,
    pub tilt_isigma: f64,
    pub frine_isigma_scale: f64,
}

/// `n` poses along a winding path and `m` landmarks beside it, each seen
/// by the poses within four steps of it, the measurements the truth plus
/// noise, the poses and landmarks moved off it.
fn slam_path(n: usize, m: usize, seed: u64) -> SPath {
    use arael::matrix::matrix3d;
    use arael::quatern::quatern;
    use arael::vect::vect3d;
    let mut rng = Lcg(seed);
    let truth: Vec<(vect3d, vect3d)> = (0..n)
        .map(|k| {
            let t = k as f64;
            (vect3d::new(t, 2.0 * (0.3 * t).sin(), 0.1 * t), vect3d::new(0.02 * t.sin(), 0.03 * t.cos(), 0.3 * (0.3 * t).cos()))
        })
        .collect();
    let lms: Vec<vect3d> = (0..m)
        .map(|j| vect3d::new(j as f64 * n as f64 / m as f64, if j % 2 == 0 { 5.0 } else { -5.0 }, 1.0 + (j % 3) as f64))
        .collect();
    let jitter = |rng: &mut Lcg, s: f64| vect3d::new(s * rng.uniform(), s * rng.uniform(), s * rng.uniform());
    let mut poses = refs::Vec::new();
    for (k, &(p, ea)) in truth.iter().enumerate() {
        let off = if k == 0 { 0.0 } else { 1.0 };
        let (p0, ea0) = (p + jitter(&mut rng, 0.05 * off), ea + jitter(&mut rng, 0.01 * off));
        poses.push(SPose {
            pos: Param::new(p0),
            ea: SimpleEulerAngleParam::new(ea0),
            prior_pos: p0,
            prior_ea: ea0,
            gps_pos: p + jitter(&mut rng, 0.1),
            gps_cov_r: matrix3d::identity(),
            gps_isigma: vect3d::new(2.0, 2.0, 1.0),
            tilt_roll: ea.x + 0.002 * rng.uniform(),
            tilt_pitch: ea.y + 0.002 * rng.uniform(),
            hb_pose: SelfBlock::new(),
        });
    }
    let mut landmarks = refs::Arena::new();
    for (j, &l) in lms.iter().enumerate() {
        let mut frines = std::vec::Vec::new();
        for (k, &(p, ea)) in truth.iter().enumerate() {
            if (k as f64 - l.x).abs() > 4.0 {
                continue;
            }
            let r = matrix3d::rotation_from_euler_angles(ea).transpose() * (l - p);
            let d = r * (1.0 / (r * r).sqrt());
            let noise = matrix3d::rotation_from_euler_angles(jitter(&mut rng, 0.002));
            frines.push(SFrine {
                pose: poses.ref_at(k),
                mf2r: quatern::from_two_vectors(vect3d::new(1.0, 0.0, 0.0), d).rotation_matrix() * noise,
                camera_pos: vect3d::new(0.0, 0.0, 0.0),
                isigma: vect2::new(100.0, 100.0),
                hb: CrossBlock::new(),
            });
        }
        let _ = j;
        let l0 = l + jitter(&mut rng, 0.1);
        landmarks.push(SLm { pos: Param::new(l0), prior_pos: l0, frines, hb_drift: SelfBlock::new() });
    }
    let mut pairs = std::vec::Vec::new();
    for k in 1..n {
        let ((pa, ea), (pb, eb)) = (truth[k - 1], truth[k]);
        let (ra, rb) = (matrix3d::rotation_from_euler_angles(ea), matrix3d::rotation_from_euler_angles(eb));
        pairs.push(SPair {
            prev: poses.ref_at(k - 1),
            cur: poses.ref_at(k),
            delta_pos: ra.transpose() * (pb - pa) + jitter(&mut rng, 0.01),
            delta_ea: (ra.transpose() * rb).get_euler_angles() + jitter(&mut rng, 0.002),
            pos_cov_r: matrix3d::identity(),
            pos_isigma: vect3d::new(20.0, 20.0, 20.0),
            ea_cov_r: matrix3d::identity(),
            ea_isigma: vect3d::new(50.0, 50.0, 50.0),
            hb: CrossBlock::new(),
        });
    }
    SPath { poses, landmarks, pairs, drift_pos_isigma: 0.01, drift_ea_isigma: 0.01, drift_lm_isigma: 0.01, tilt_isigma: 10.0, frine_isigma_scale: 1.0 }
}

/// The SLAM model assembles on the device as on the CPU (nested
/// observations, simple Euler angles, root scalars) and solves there to
/// the CPU's optimum.
#[test]
fn a_slam_path_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let mut s = slam_path(24, 30, 13);
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for (wide, tol) in [(Wide::F64, 1e-11), (Wide::Df32, 1e-10)] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
    }
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = slam_path(24, 30, 13);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        (r.start_cost, r.end_cost)
    };
    let (start, cpu) = run(None);
    assert!(cpu < 0.05 * start, "the path converges to its noise: {start} to {cpu}");
    for d in devices() {
        let (_, device) = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
    }
}
