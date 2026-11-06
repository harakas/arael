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

/// A point held near the origin.
#[arael::model]
#[arael(constraint(hb, { [priorpoint.pos.x - 1.0, priorpoint.pos.y, priorpoint.pos.z] }))]
#[derive(Clone)]
pub struct PriorPoint {
    pub pos: Param<vect3<f64>>,
    pub hb: SelfBlock<PriorPoint>,
}

/// Two points a unit apart, a `coo` constraint, which has no device form
/// yet.
#[arael::model]
#[arael(constraint(coo, { [b.pos.x - a.pos.x - 1.0] }))]
#[derive(Clone)]
pub struct Apart {
    #[arael(ref = root.points)]
    pub a: Ref<PriorPoint>,
    #[arael(ref = root.points)]
    pub b: Ref<PriorPoint>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Priors {
    pub points: refs::Vec<PriorPoint>,
    pub aparts: std::vec::Vec<Apart>,
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
    for _ in 0..2 {
        points.push(PriorPoint { pos: Param::new(vect3::new(0.0, 0.0, 0.0)), hb: SelfBlock::new() });
    }
    let aparts = vec![Apart { a: points.ref_at(0), b: points.ref_at(1) }];
    let s = Priors { points, aparts };
    let why = LmProblemInternals::<f64>::device_unsupported(&s).expect("a `coo` constraint has no device form yet");
    assert!(why.contains("Apart") && why.contains("coo"), "{why}");
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
        assert_eq!(chk.leaf_bits, 53, "an f64 model's leaves");
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
        assert_eq!(chk.leaf_bits, 24, "an f32 model's leaves");
        // the CPU sums its f32 products in f32, the device in its wide scalar
        assert!((chk.device_cost - chk.cpu_cost).abs() <= 1e-5 * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < 1e-4 && chk.hess_err < 1e-4, "{wide:?}: {chk:?}");
    }
}

/// A point of f32 parameters and data in an f64 root.
#[arael::model]
#[arael(constraint(hb, { [(narrowpt.x - narrowpt.mx) * 2.0, narrowpt.x * narrowpt.x - narrowpt.mx] }))]
#[derive(Clone)]
pub struct NarrowPt {
    pub x: Param<f32>,
    pub mx: f32,
    pub hb: SelfBlock<NarrowPt>,
}

/// A point of f64 parameters and data.
#[arael::model]
#[arael(constraint(hb, { [(widept.x - widept.mx) * 2.0] }))]
#[derive(Clone)]
pub struct WidePt {
    pub x: Param<f64>,
    pub mx: f64,
    pub hb: SelfBlock<WidePt>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct NarrowRoot {
    pub narrow: refs::Vec<NarrowPt>,
    pub wide: refs::Vec<WidePt>,
}

/// An f32 entity in an f64 root: the CPU computes its rows in f32, the
/// device in f64, and the device check reads the narrower leaves' bits.
#[test]
fn an_f32_entity_in_an_f64_root_reports_its_leaves_bits() {
    use arael::__gpu::Wide;
    let mut s = NarrowRoot { narrow: refs::Vec::new(), wide: refs::Vec::new() };
    for k in 0..4 {
        let t = 0.3 + 0.2 * k as f64;
        s.narrow.push(NarrowPt { x: Param::new(t as f32), mx: (t + 0.1) as f32, hb: SelfBlock::new() });
        s.wide.push(WidePt { x: Param::new(t), mx: t - 0.1, hb: SelfBlock::new() });
    }
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for wide in [Wide::F64, Wide::Df32] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert_eq!(chk.leaf_bits, 24, "{wide:?}: the f32 entity's leaves");
        let tol = arael::__gpu::check_tolerance(chk.leaf_bits);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
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

/// Edges whose two refs name one pose: the cross block lands on the pose's
/// own tile, each symmetric pair of entries in one cell and the diagonal
/// twice, as the CPU writes an aliased cross block.
#[test]
fn an_edge_from_a_pose_to_itself_assembles_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
        let (poses, mut edges, prior) = pose_graph::<f64>(20, 3);
        for k in [3usize, 11, 11] {
            edges.push(Edge2D { a: poses.ref_at(k), b: poses.ref_at(k), dx: 0.01, dy: -0.02, dth: 0.03, w: 5.0, hb: CrossBlock::new() });
        }
        let mut s = Graph2D { poses, edges, prior };
        assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let x1 = perturbed(&x);
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
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

/// A point held near where it was seen.
#[arael::model]
#[arael(constraint(hb, { [(ptm.pos.x - ptm.prior.x) * 0.1, (ptm.pos.y - ptm.prior.y) * 0.1] }))]
#[derive(Clone)]
pub struct PtM {
    pub pos: Param<vect2<f64>>,
    pub prior: vect2<f64>,
    pub hb: SelfBlock<PtM>,
}

/// A line `n . p = d`, its normal held near unit length.
#[arael::model]
#[arael(constraint(hb, { [(lnm.n.x * lnm.n.x + lnm.n.y * lnm.n.y - 1.0) * 10.0] }))]
#[derive(Clone)]
pub struct LnM {
    pub n: Param<vect2<f64>>,
    pub d: Param<f64>,
    pub hb: SelfBlock<LnM>,
}

/// A segment of a line: the midpoint of `a` and `b` on the line, `b - a`
/// along it at length `len`. Three entities, a cross block per pair.
#[arael::model]
#[arael(constraint([hb_la, hb_lb, hb_ab], guard = self.on, loss = |s| loss_cauchy(s, 4.0), {
    let mx = (a.pos.x + b.pos.x) * 0.5;
    let my = (a.pos.y + b.pos.y) * 0.5;
    let dx = b.pos.x - a.pos.x;
    let dy = b.pos.y - a.pos.y;
    [(ln.n.x * mx + ln.n.y * my - ln.d) * mid.w,
     (sqrt(dx * dx + dy * dy) - mid.len) * mid.w,
     (ln.n.x * dx + ln.n.y * dy) * mid.w]
}))]
#[derive(Clone)]
pub struct Mid {
    #[arael(ref = root.lines)]
    pub ln: Ref<LnM>,
    #[arael(ref = root.points)]
    pub a: Ref<PtM>,
    #[arael(ref = root.points)]
    pub b: Ref<PtM>,
    pub len: f64,
    pub w: f64,
    pub on: bool,
    #[arael(cross = (ln, a))]
    pub hb_la: CrossBlock<LnM, PtM>,
    #[arael(cross = (ln, b))]
    pub hb_lb: CrossBlock<LnM, PtM>,
    #[arael(cross = (a, b))]
    pub hb_ab: CrossBlock<PtM, PtM>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct MidWorld {
    pub lines: refs::Vec<LnM>,
    pub points: refs::Vec<PtM>,
    pub mids: std::vec::Vec<Mid>,
}

/// Five lines with six segments each; every third segment the one before
/// it with its points swapped (a pair of points on one cross tile both
/// ways round), every seventh guarded off, one point fixed; the points and
/// lines moved off the truth.
fn mid_world(seed: u64) -> MidWorld {
    let mut rng = Lcg(seed);
    let mut lines = refs::Vec::new();
    let mut points = refs::Vec::new();
    let mut mids = std::vec::Vec::new();
    for l in 0..5 {
        let th = 0.6 * l as f64 + 0.2;
        let (n, t) = ((th.cos(), th.sin()), (-th.sin(), th.cos()));
        let d = 1.0 + l as f64;
        let line = lines.push(LnM {
            n: Param::new(vect2::new(n.0 + 0.05 * rng.uniform(), n.1 + 0.05 * rng.uniform())),
            d: Param::new(d + 0.05 * rng.uniform()),
            hb: SelfBlock::new(),
        });
        for k in 0..6 {
            let (a, b, len) = if k % 3 == 2 {
                let prev: &Mid = mids.last().unwrap();
                (prev.b, prev.a, prev.len)
            } else {
                let s = k as f64 - 2.5;
                let len = 1.0 + 0.1 * k as f64;
                let c = (n.0 * d + t.0 * s, n.1 * d + t.1 * s);
                let mut pt = |sign: f64, rng: &mut Lcg| {
                    let p = vect2::new(c.0 + sign * t.0 * len / 2.0 + 0.05 * rng.uniform(), c.1 + sign * t.1 * len / 2.0 + 0.05 * rng.uniform());
                    let pos = if points.len() == 3 { Param::fixed(p) } else { Param::new(p) };
                    points.push(PtM { pos, prior: p, hb: SelfBlock::new() })
                };
                (pt(-1.0, &mut rng), pt(1.0, &mut rng), len)
            };
            let on = mids.len() % 7 != 4;
            mids.push(Mid { ln: line, a, b, len, w: 3.0, on, hb_la: CrossBlock::new(), hb_lb: CrossBlock::new(), hb_ab: CrossBlock::new() });
        }
    }
    MidWorld { lines, points, mids }
}

/// A constraint of three entities with a cross block per pair (a guard, a
/// loss): one side per entity, one reducer per pair, the device assembles
/// as the CPU does and solves to its optimum.
#[test]
fn a_multi_cross_constraint_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    let mut s = mid_world(5);
    assert!(s.mids.iter().any(|m| !m.on));
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None);
    let shape = arael::__gpu::DeviceRoot::<f64>::device_shape(&s);
    let mid = shape.constraints.iter().find(|k| k.sides.len() == 3).expect("the segment's three sides");
    assert_eq!(mid.pairs.len(), 3);
    assert_eq!(mid.sides[1], mid.sides[2], "both points in one entity table");
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
    }
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = mid_world(5);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        (r.start_cost, r.end_cost, r.x.iter().copied().collect::<Vec<f64>>())
    };
    let (start, cpu, cpu_x) = run(None);
    assert!(cpu < 0.05 * start, "the segments converge: {start} to {cpu}");
    for d in devices() {
        let (_, device, dx) = run(Some(&d));
        assert!((device - cpu).abs() <= 1e-9 * cpu, "{}: {device} against the CPU's {cpu}", label(&d));
        assert_close(&dx, &cpu_x, 1e-7, &format!("{}: the solved parameters", label(&d)));
    }
}

/// A node of the shared cross tile tests, held near where it was seen.
#[arael::model]
#[arael(constraint(hb, { [(nodep.x - nodep.px) * 0.3, (nodep.y - nodep.py) * 0.3] }))]
#[derive(Clone)]
pub struct NodeP {
    pub x: Param<f64>,
    pub y: Param<f64>,
    pub px: f64,
    pub py: f64,
    pub hb: SelfBlock<NodeP>,
}

/// A link with refs of its own, written into its parent's cross block: one
/// tile for every link the parent holds.
#[arael::model]
#[arael(constraint(parent.hb, guard = self.on, {
    [(b.x - a.x - shlink.dx) * shlink.w, (b.y - a.y - shlink.dy) * shlink.w]
}))]
#[derive(Clone)]
pub struct ShLink {
    #[arael(ref = root.nodes)]
    pub a: Ref<NodeP>,
    #[arael(ref = root.nodes)]
    pub b: Ref<NodeP>,
    pub dx: f64,
    pub dy: f64,
    pub w: f64,
    pub on: bool,
}

#[arael::model]
#[derive(Clone)]
pub struct ShPair {
    pub links: std::vec::Vec<ShLink>,
    pub hb: CrossBlock<NodeP, NodeP>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct ShNet {
    pub nodes: refs::Vec<NodeP>,
    pub pairs: std::vec::Vec<ShPair>,
}

/// A link of no refs: its parent's refs are its entities, its parent's
/// cross block its tile, its parent's gain a factor.
#[arael::model]
#[arael(constraint(parent.hb, guard = self.on, {
    [(parent.b.x - parent.a.x - prlink.dx) * prlink.w * parent.gain,
     (parent.b.y - parent.a.y - prlink.dy) * prlink.w * parent.gain]
}))]
#[derive(Clone)]
pub struct PrLink {
    pub dx: f64,
    pub dy: f64,
    pub w: f64,
    pub on: bool,
}

#[arael::model]
#[derive(Clone)]
pub struct PrPair {
    #[arael(ref = root.nodes)]
    pub a: Ref<NodeP>,
    #[arael(ref = root.nodes)]
    pub b: Ref<NodeP>,
    pub gain: f64,
    pub links: std::vec::Vec<PrLink>,
    pub hb: CrossBlock<NodeP, NodeP>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct PrNet {
    pub nodes: refs::Vec<NodeP>,
    pub pairs: std::vec::Vec<PrPair>,
}

/// A link with refs and a cross block of its own, held below the root.
#[arael::model]
#[arael(constraint(hb, guard = self.on, {
    [(b.x - a.x - ownlink.dx) * ownlink.w, (b.y - a.y - ownlink.dy) * ownlink.w]
}))]
#[derive(Clone)]
pub struct OwnLink {
    #[arael(ref = root.nodes)]
    pub a: Ref<NodeP>,
    #[arael(ref = root.nodes)]
    pub b: Ref<NodeP>,
    pub dx: f64,
    pub dy: f64,
    pub w: f64,
    pub on: bool,
    pub hb: CrossBlock<NodeP, NodeP>,
}

#[arael::model]
#[derive(Clone)]
pub struct OwnPair {
    pub links: std::vec::Vec<OwnLink>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct OwnNet {
    pub nodes: refs::Vec<NodeP>,
    pub pairs: std::vec::Vec<OwnPair>,
}

/// One table for both nets: per pair its nodes, its gain and its links
/// (dx, dy, w, on).
type NetPairs = Vec<((usize, usize), f64, Vec<(f64, f64, f64, bool)>)>;

/// Twelve nodes on a grid, moved off it; pairs of them with one to four
/// links each (the measurements the truth plus noise, every fifth link
/// guarded off), two pairs joining the same nodes in opposite orders, one
/// pair joining a node to itself, one pair with no links.
fn net_data(seed: u64) -> (Vec<(f64, f64)>, NetPairs) {
    let mut rng = Lcg(seed);
    let truth: Vec<(f64, f64)> = (0..12).map(|i| ((i % 4) as f64, (i / 4) as f64)).collect();
    let nodes = truth.iter().map(|&(x, y)| (x + 0.1 * rng.uniform(), y + 0.1 * rng.uniform())).collect();
    let mut ends: Vec<(usize, usize)> = (0..16).map(|k| (k % 12, (k * 5 + 3) % 12)).filter(|(i, j)| i != j).collect();
    ends.extend([(1, 2), (2, 1), (5, 5), (0, 1)]);
    let mut n = 0;
    let pairs = ends
        .iter()
        .enumerate()
        .map(|(k, &(i, j))| {
            let count = if k + 1 == ends.len() { 0 } else { 1 + k % 4 };
            let links = (0..count)
                .map(|m| {
                    n += 1;
                    let (dx, dy) = (truth[j].0 - truth[i].0, truth[j].1 - truth[i].1);
                    (dx + 0.01 * rng.uniform(), dy + 0.01 * rng.uniform(), 1.0 + 0.2 * m as f64, n % 5 != 3)
                })
                .collect();
            ((i, j), 0.5 + 0.1 * k as f64, links)
        })
        .collect();
    (nodes, pairs)
}

fn net_nodes(nodes: &[(f64, f64)]) -> refs::Vec<NodeP> {
    let mut v = refs::Vec::new();
    for &(x, y) in nodes {
        v.push(NodeP { x: Param::new(x), y: Param::new(y), px: x, py: y, hb: SelfBlock::new() });
    }
    v
}

fn shared_net(seed: u64) -> ShNet {
    let (nodes, pairs) = net_data(seed);
    let nodes = net_nodes(&nodes);
    let pairs = pairs
        .into_iter()
        .map(|((i, j), _, links)| ShPair {
            links: links.into_iter().map(|(dx, dy, w, on)| ShLink { a: nodes.ref_at(i), b: nodes.ref_at(j), dx, dy, w, on }).collect(),
            hb: CrossBlock::new(),
        })
        .collect();
    ShNet { nodes, pairs }
}

fn parent_ref_net(seed: u64) -> PrNet {
    let (nodes, pairs) = net_data(seed);
    let nodes = net_nodes(&nodes);
    let pairs = pairs
        .into_iter()
        .map(|((i, j), gain, links)| PrPair {
            a: nodes.ref_at(i),
            b: nodes.ref_at(j),
            gain,
            links: links.into_iter().map(|(dx, dy, w, on)| PrLink { dx, dy, w, on }).collect(),
            hb: CrossBlock::new(),
        })
        .collect();
    PrNet { nodes, pairs }
}

fn own_block_net(seed: u64) -> OwnNet {
    let (nodes, pairs) = net_data(seed);
    let nodes = net_nodes(&nodes);
    let pairs = pairs
        .into_iter()
        .map(|((i, j), _, links)| OwnPair {
            links: links.into_iter().map(|(dx, dy, w, on)| OwnLink { a: nodes.ref_at(i), b: nodes.ref_at(j), dx, dy, w, on, hb: CrossBlock::new() }).collect(),
        })
        .collect();
    OwnNet { nodes, pairs }
}

/// `make`'s model assembles on the device as on the CPU in both wide
/// scalars, and its whole-system solve on every device reaches the CPU's
/// optimum.
fn assembles_and_solves_as_on_the_cpu<P: LmProblemInternals<f64> + RootProblem<f64>>(make: impl Fn() -> P, what: &str) {
    use arael::__gpu::Wide;
    let mut s = make();
    assert_eq!(LmProblemInternals::<f64>::device_unsupported(&s), None, "{what}");
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
        let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
        assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{what} {wide:?}: {chk:?}");
        assert!(chk.grad_err < tol && chk.hess_err < tol, "{what} {wide:?}: {chk:?}");
    }
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = make();
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some(), "{what}");
        (r.start_cost, r.end_cost, r.x.to_vec())
    };
    let (start, cpu, cpu_x) = run(None);
    assert!(cpu < 0.05 * start, "{what} converges: {start} to {cpu}");
    for d in devices() {
        let (_, device, dx) = run(Some(&d));
        // relative to the optimum's cost, or to the start's where the
        // optimum's is rounding of zero
        assert!((device - cpu).abs() <= 1e-9 * cpu + 1e-15 * start, "{what} {}: {device} against the CPU's {cpu}", label(&d));
        assert_close(&dx, &cpu_x, 1e-7, &format!("{what} {}: the solved parameters", label(&d)));
    }
}

/// Links writing into their parent's cross block, with refs of their own
/// and with their parent's: every link under a parent sums into one tile,
/// two parents of opposite orders into one tile of H, a node joined to
/// itself into its own tile.
#[test]
fn parent_owned_cross_blocks_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(|| shared_net(9), "own refs");
    assembles_and_solves_as_on_the_cpu(|| parent_ref_net(9), "parent refs");
}

/// The same links with cross blocks of their own, in collections below the
/// root: walked through their parents' loops.
#[test]
fn cross_constraints_below_the_root_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(|| own_block_net(9), "own blocks");
}

/// The mixed form's camera (a focal length) and point, each held near
/// where it was seen.
#[arael::model]
#[arael(constraint(hb, { [(camm.f - camm.pf) * 0.5] }))]
#[derive(Clone)]
pub struct CamM {
    pub f: Param<f64>,
    pub pf: f64,
    pub hb: SelfBlock<CamM>,
}

#[arael::model]
#[arael(constraint(hb, { [(pointm.p.x - pointm.px) * 0.2, (pointm.p.y - pointm.py) * 0.2] }))]
#[derive(Clone)]
pub struct PointM {
    pub p: Param<vect2<f64>>,
    pub px: f64,
    pub py: f64,
    pub hb: SelfBlock<PointM>,
}

#[arael::model]
#[arael(constraint(hb, { [(posem.t.x - posem.px) * 0.3, (posem.t.y - posem.py) * 0.3] }))]
#[derive(Clone)]
pub struct PoseM {
    pub t: Param<vect2<f64>>,
    pub px: f64,
    pub py: f64,
    pub hb: SelfBlock<PoseM>,
}

/// An observation of a point in an image: its own (point, pose) and
/// (point, camera) tiles, the image's (pose, camera) tile shared by all
/// the image's observations, the pose and the camera the image's refs.
#[arael::model]
#[arael(constraint([hb_point_pose, hb_point_cam, parent.hb_pose_cam], parent = image, guard = self.on,
    loss = |s| loss_huber(s, 0.09), {
    [image.cam.f * (point.p.x - image.pose.t.x) - imgobs.m.x,
     image.cam.f * (point.p.y - image.pose.t.y) - imgobs.m.y]
}))]
#[derive(Clone)]
pub struct ImgObs {
    #[arael(ref = root.points)]
    pub point: Ref<PointM>,
    pub m: vect2<f64>,
    pub on: bool,
    pub hb_point_pose: CrossBlock<PointM, PoseM>,
    pub hb_point_cam: CrossBlock<PointM, CamM>,
}

#[arael::model]
#[derive(Clone)]
pub struct ImageM {
    #[arael(ref = root.poses)]
    pub pose: Ref<PoseM>,
    #[arael(ref = root.cams)]
    pub cam: Ref<CamM>,
    pub obs: std::vec::Vec<ImgObs>,
    pub hb_pose_cam: CrossBlock<PoseM, CamM>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct ImgWorld {
    pub poses: refs::Vec<PoseM>,
    pub cams: refs::Vec<CamM>,
    pub points: refs::Vec<PointM>,
    pub images: std::vec::Vec<ImageM>,
}

/// A pose holding its images: the observation reaches it two levels up.
#[arael::model]
#[arael(constraint(hb, { [(poseh.t.x - poseh.px) * 0.3, (poseh.t.y - poseh.py) * 0.3] }))]
#[derive(Clone)]
pub struct PoseH {
    pub t: Param<vect2<f64>>,
    pub px: f64,
    pub py: f64,
    pub w: f64,
    pub images: std::vec::Vec<ImageH>,
    pub hb: SelfBlock<PoseH>,
}

#[arael::model]
#[derive(Clone)]
pub struct ImageH {
    #[arael(ref = root.cams)]
    pub cam: Ref<CamM>,
    pub obs: std::vec::Vec<ObsH>,
    pub hb_pose_cam: CrossBlock<PoseH, CamM>,
}

#[arael::model]
#[arael(constraint([hb_point_pose, hb_point_cam, parent.hb_pose_cam], parent = image, parent.parent = pose,
    guard = self.on, {
    [image.cam.f * (point.p.x - pose.t.x) * pose.w - obsh.m.x,
     image.cam.f * (point.p.y - pose.t.y) * pose.w - obsh.m.y]
}))]
#[derive(Clone)]
pub struct ObsH {
    #[arael(ref = root.points)]
    pub point: Ref<PointM>,
    pub m: vect2<f64>,
    pub on: bool,
    pub hb_point_pose: CrossBlock<PointM, PoseH>,
    pub hb_point_cam: CrossBlock<PointM, CamM>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct PoseWorld {
    pub poses: refs::Vec<PoseH>,
    pub cams: refs::Vec<CamM>,
    pub points: refs::Vec<PointM>,
}

/// A 2D position.
type P2 = (f64, f64);

/// An image of the mixed scenes: its pose, its camera, and per observation
/// its point, its measurement and whether its guard keeps it.
type MixedImage = (usize, usize, Vec<(usize, P2, bool)>);

/// The truth of the mixed scenes: six poses, three cameras, twelve points.
fn mixed_truth() -> (Vec<P2>, Vec<f64>, Vec<P2>) {
    let poses = (0..6).map(|i| (0.4 * i as f64, -0.2 * i as f64 + 0.1)).collect();
    let cams = vec![1.5, 0.8, 1.2];
    let points = (0..12).map(|k| (1.0 + 0.3 * (k % 4) as f64, 0.5 - 0.25 * (k / 4) as f64)).collect();
    (poses, cams, points)
}

/// The images of the mixed scenes, (pose, camera) pairs; pose 4 through
/// camera 1 an image of no observations. Every image sees every point but
/// every seventh, the measurements the truth plus noise, some past the
/// Huber knee; every fifth observation guarded off.
fn mixed_images() -> Vec<MixedImage> {
    let (tp, tc, tk) = mixed_truth();
    let mut rng = Lcg(21);
    let mut n = 0;
    let mut images = Vec::new();
    for (i, &(px, py)) in tp.iter().enumerate() {
        for j in [i % 3, (i + 1) % 3] {
            let obs = (0..tk.len())
                .filter(|_| !(i == 4 && j == 1))
                .filter_map(|k| {
                    n += 1;
                    (n % 7 != 0).then(|| {
                        let m = (tc[j] * (tk[k].0 - px) + 0.05 * rng.uniform(), tc[j] * (tk[k].1 - py) + 0.05 * rng.uniform());
                        (k, m, n % 5 != 2)
                    })
                })
                .collect();
            images.push((i, j, obs));
        }
    }
    images
}

fn img_world() -> ImgWorld {
    let (tp, tc, tk) = mixed_truth();
    let mut poses = refs::Vec::new();
    for (i, &(x, y)) in tp.iter().enumerate() {
        poses.push(PoseM { t: Param::new(vect2::new(x + 0.07, y - 0.05 * i as f64)), px: x, py: y, hb: SelfBlock::new() });
    }
    let mut cams = refs::Vec::new();
    for &f in &tc {
        cams.push(CamM { f: Param::new(f * 1.1), pf: f, hb: SelfBlock::new() });
    }
    let mut points = refs::Vec::new();
    for (k, &(x, y)) in tk.iter().enumerate() {
        points.push(PointM { p: Param::new(vect2::new(x - 0.06, y + 0.04 * k as f64)), px: x, py: y, hb: SelfBlock::new() });
    }
    let images = mixed_images()
        .into_iter()
        .map(|(i, j, obs)| ImageM {
            pose: poses.ref_at(i),
            cam: cams.ref_at(j),
            obs: obs
                .into_iter()
                .map(|(k, m, on)| ImgObs { point: points.ref_at(k), m: vect2::new(m.0, m.1), on, hb_point_pose: CrossBlock::new(), hb_point_cam: CrossBlock::new() })
                .collect(),
            hb_pose_cam: CrossBlock::new(),
        })
        .collect();
    ImgWorld { poses, cams, points, images }
}

fn pose_world() -> PoseWorld {
    let (tp, tc, tk) = mixed_truth();
    let mut cams = refs::Vec::new();
    for &f in &tc {
        cams.push(CamM { f: Param::new(f * 1.1), pf: f, hb: SelfBlock::new() });
    }
    let mut points = refs::Vec::new();
    for (k, &(x, y)) in tk.iter().enumerate() {
        points.push(PointM { p: Param::new(vect2::new(x - 0.06, y + 0.04 * k as f64)), px: x, py: y, hb: SelfBlock::new() });
    }
    let mut poses = refs::Vec::new();
    for (i, &(x, y)) in tp.iter().enumerate() {
        poses.push(PoseH { t: Param::new(vect2::new(x + 0.07, y - 0.05 * i as f64)), px: x, py: y, w: 1.0, images: std::vec::Vec::new(), hb: SelfBlock::new() });
    }
    for (i, j, obs) in mixed_images() {
        let image = ImageH {
            cam: cams.ref_at(j),
            obs: obs
                .into_iter()
                .map(|(k, m, on)| ObsH { point: points.ref_at(k), m: vect2::new(m.0, m.1), on, hb_point_pose: CrossBlock::new(), hb_point_cam: CrossBlock::new() })
                .collect(),
            hb_pose_cam: CrossBlock::new(),
        };
        poses[i].images.push(image);
    }
    PoseWorld { poses, cams, points }
}

/// The mixed form: own cross blocks per observation, the image's shared
/// one, the entities through the instance's ref, the image's refs and the
/// pose two levels up.
#[test]
fn the_mixed_form_assembles_and_solves_on_the_device_as_on_the_cpu() {
    let s = img_world();
    let shape = arael::__gpu::DeviceRoot::<f64>::device_shape(&s);
    let obs = shape.constraints.iter().find(|k| k.sides.len() == 3).expect("the observation's three sides");
    assert_eq!(obs.pairs.len(), 3);
    assembles_and_solves_as_on_the_cpu(img_world, "the image's refs");
    assembles_and_solves_as_on_the_cpu(pose_world, "the pose two levels up");
}

/// A sample of a line `y = a x + b` whose parameters are the root's: the
/// sample writes the root's self block, the root the one entity of its
/// table.
#[arael::model]
#[arael(constraint(root.hb, guard = self.ok, loss = |s| loss_huber(s, 0.01), { [sample.y - root.a * sample.x - root.b] }))]
#[derive(Clone)]
pub struct Sample {
    pub x: f64,
    pub y: f64,
    pub ok: bool,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct LineFit {
    pub a: Param<f64>,
    pub b: Param<f64>,
    pub hb: SelfBlock<LineFit>,
    pub samples: std::vec::Vec<Sample>,
}

/// A sample of its curve, which holds the parameters: the sample writes
/// its parent's self block.
#[arael::model]
#[arael(constraint(parent.hb, { [csample.y - (curve.m * csample.x + curve.c)] }))]
#[derive(Clone)]
pub struct CSample {
    pub x: f64,
    pub y: f64,
}

#[arael::model]
#[derive(Clone)]
pub struct Curve {
    pub m: Param<f64>,
    pub c: Param<f64>,
    pub samples: std::vec::Vec<CSample>,
    pub hb: SelfBlock<Curve>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Curves {
    pub curves: refs::Vec<Curve>,
}

/// One self-constrained type held in two root collections: one entity
/// table over both, walked in their order.
#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct TwoHomes {
    pub near: refs::Vec<NodeP>,
    pub far: refs::Vec<NodeP>,
}

fn line_fit() -> LineFit {
    let mut rng = Lcg(31);
    let samples = (0..40)
        .map(|i| {
            let x = 0.1 * i as f64;
            let out = if i % 9 == 4 { 1.0 } else { 0.0 };
            Sample { x, y: 2.0 * x + 1.0 + 0.02 * rng.uniform() + out, ok: i % 11 != 7 }
        })
        .collect();
    LineFit { a: Param::new(0.3), b: Param::new(-0.2), hb: SelfBlock::new(), samples }
}

fn curves() -> Curves {
    let mut rng = Lcg(32);
    let mut curves = refs::Vec::new();
    for k in 0..5 {
        let (m, c) = (0.5 * k as f64 - 1.0, 0.3 * k as f64);
        let samples = (0..8).map(|i| CSample { x: 0.2 * i as f64, y: m * 0.2 * i as f64 + c + 0.01 * rng.uniform() }).collect();
        curves.push(Curve { m: Param::new(m + 0.2), c: Param::new(c - 0.1), samples, hb: SelfBlock::new() });
    }
    Curves { curves }
}

fn two_homes() -> TwoHomes {
    let mut rng = Lcg(33);
    let (mut near, mut far) = (refs::Vec::new(), refs::Vec::new());
    for k in 0..20 {
        let (px, py) = (k as f64, 0.5 * k as f64);
        let p = NodeP { x: Param::new(px + 0.1 * rng.uniform()), y: Param::new(py + 0.1 * rng.uniform()), px, py, hb: SelfBlock::new() };
        if k % 3 == 0 { far.push(p) } else { near.push(p) };
    }
    TwoHomes { near, far }
}

/// A pose held near where it was seen.
#[arael::model]
#[arael(constraint(hb_pose, { [(poser.pos.x - poser.px) * 0.5, (poser.pos.y - poser.py) * 0.5] }))]
#[derive(Clone)]
pub struct PoseR {
    pub pos: Param<vect2<f64>>,
    pub px: f64,
    pub py: f64,
    pub hb_pose: SelfBlock<PoseR>,
}

/// A sighting of a landmark's anchor from a pose, through the root's
/// offset: the pose's self block is the primary, the (pose, root) pair the
/// sighting's own cross block, the landmark's anchor parent data.
#[arael::model]
#[arael(constraint([pose.hb_pose, hb_root], parent = lm, {
    let d = lm.anchor - pose.pos;
    [(d.x + offsetroot.offset - frineo.dx) * offsetroot.isigma,
     (d.y + offsetroot.offset - frineo.dy) * offsetroot.isigma]
}))]
#[derive(Clone)]
pub struct FrineO {
    #[arael(ref = root.poses)]
    pub pose: Ref<PoseR>,
    pub dx: f64,
    pub dy: f64,
    pub hb_root: CrossBlock<PoseR, OffsetRoot>,
}

#[arael::model]
#[derive(Clone)]
pub struct LandmarkO {
    pub anchor: vect2<f64>,
    pub frines: std::vec::Vec<FrineO>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct OffsetRoot {
    pub poses: refs::Vec<PoseR>,
    pub landmarks: refs::Vec<LandmarkO>,
    pub offset: Param<f64>,
    pub isigma: f64,
    pub hb: SelfBlock<OffsetRoot>,
}

fn offset_root() -> OffsetRoot {
    let mut rng = Lcg(34);
    let mut poses = refs::Vec::new();
    let truth: Vec<(f64, f64)> = (0..8).map(|k| (k as f64, (0.4 * k as f64).sin())).collect();
    for &(x, y) in &truth {
        poses.push(PoseR { pos: Param::new(vect2::new(x + 0.1 * rng.uniform(), y + 0.1 * rng.uniform())), px: x, py: y, hb_pose: SelfBlock::new() });
    }
    let mut landmarks = refs::Vec::new();
    for j in 0..4 {
        let anchor = vect2::new(2.0 * j as f64 + 0.5, 3.0);
        let frines = (0..8)
            .filter(|k| (k + j) % 3 != 0)
            .map(|k| FrineO {
                pose: poses.ref_at(k),
                dx: anchor.x - truth[k].0 + 0.25 + 0.01 * rng.uniform(),
                dy: anchor.y - truth[k].1 + 0.25 + 0.01 * rng.uniform(),
                hb_root: CrossBlock::new(),
            })
            .collect();
        landmarks.push(LandmarkO { anchor, frines });
    }
    OffsetRoot { poses, landmarks, offset: Param::new(0.0), isigma: 3.0, hb: SelfBlock::new() }
}

/// Data-only instances writing the root's self block and a parent's, and
/// an entity type held in two collections: the device assembles as the CPU
/// does and solves to its optimum.
#[test]
fn root_and_parent_self_blocks_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assert!(line_fit().samples.iter().any(|s| !s.ok));
    assembles_and_solves_as_on_the_cpu(line_fit, "the root's self block");
    assembles_and_solves_as_on_the_cpu(curves, "the parent's self block");
    assembles_and_solves_as_on_the_cpu(two_homes, "a type in two collections");
}

/// A remote primary block with an own cross block to the root, the root a
/// side of the constraint (its offset a parameter), the parent's data read
/// by name.
#[test]
fn the_root_as_a_side_assembles_and_solves_on_the_device_as_on_the_cpu() {
    let s = offset_root();
    let shape = arael::__gpu::DeviceRoot::<f64>::device_shape(&s);
    assert!(shape.constraints.iter().any(|k| k.sides.len() == 2 && k.pairs.len() == 1), "the sighting's pose and root");
    assembles_and_solves_as_on_the_cpu(offset_root, "the root as a side");
}

/// A point held near where it was seen.
#[arael::model]
#[arael(constraint(hb, { [(ownpt.x - ownpt.px) * 2.0] }))]
#[derive(Clone)]
pub struct OwnPt {
    pub x: Param<f64>,
    pub px: f64,
    pub hb: SelfBlock<OwnPt>,
}

/// A sample of the line the root's scale and offset make, written into
/// the root's self block.
#[arael::model]
#[arael(constraint(root.hb, { [(ownsample.y - root.scale * ownsample.x - root.offset) * 2.0] }))]
#[derive(Clone)]
pub struct OwnSample {
    pub x: f64,
    pub y: f64,
}

/// A root with constraints of its own: a prior on its offset reading its
/// own data, and two rows over both its parameters behind a guard on its
/// own field; beside an entity collection and data instances writing its
/// self block.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [(ownroot.offset - ownroot.prior) * ownroot.isigma] }))]
#[arael(constraint(hb, guard = self.use_scale, {
    [(ownroot.scale * ownroot.scale - 2.0) * 0.5, (ownroot.scale - 1.4) * sin(ownroot.offset + 1.0)]
}))]
#[derive(Clone)]
pub struct OwnRoot {
    pub pts: refs::Vec<OwnPt>,
    pub samples: std::vec::Vec<OwnSample>,
    pub offset: Param<f64>,
    pub scale: Param<f64>,
    pub prior: f64,
    pub isigma: f64,
    pub use_scale: bool,
    pub hb: SelfBlock<OwnRoot>,
}

fn own_root(use_scale: bool) -> OwnRoot {
    let mut rng = Lcg(35);
    let mut pts = refs::Vec::new();
    for k in 0..5 {
        pts.push(OwnPt { x: Param::new(k as f64 + 0.1 * rng.uniform()), px: k as f64, hb: SelfBlock::new() });
    }
    let samples = (0..9)
        .map(|k| {
            let x = 0.5 * k as f64 - 2.0;
            OwnSample { x, y: 1.4 * x + 0.3 + 0.01 * rng.uniform() }
        })
        .collect();
    OwnRoot { pts, samples, offset: Param::new(0.0), scale: Param::new(1.0), prior: 0.25, isigma: 3.0, use_scale, hb: SelfBlock::new() }
}

/// The root's own constraints, the root its own instance (no containment
/// path), with its guard on and off: the device assembles as the CPU does
/// and solves to its optimum.
#[test]
fn constraints_on_the_root_itself_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(|| own_root(true), "the root's own constraints");
    assembles_and_solves_as_on_the_cpu(|| own_root(false), "the root's own constraints, the guarded one off");
}

/// A pose pulled toward its measurement and held near where the solve
/// started through its parameters' stored values (`<param>_value`): a
/// scalar's, a vector's and a `SimpleEulerAngleParam`'s.
#[arael::model]
#[arael(constraint(hb, {
    let d = driftp.p - driftp.p_value;
    let e = driftp.ea - driftp.ea_value;
    [(driftp.s - driftp.s_value) * 0.3, d.x * 0.25, d.y * 0.25, e.x * 0.2, e.y * 0.2, e.z * 0.2]
}))]
#[arael(constraint(hb, {
    [(driftp.s - driftp.ms) * 2.0, (driftp.p.x - driftp.mp.x) * 2.0, (driftp.p.y - driftp.mp.y) * 2.0,
     (driftp.ea.x + driftp.ea.y - driftp.me) * 2.0, (driftp.ea.z - driftp.me) * 1.5]
}))]
#[derive(Clone)]
pub struct DriftP {
    pub s: Param<f64>,
    pub p: Param<vect2<f64>>,
    pub ea: SimpleEulerAngleParam<f64>,
    pub ms: f64,
    pub mp: vect2<f64>,
    pub me: f64,
    pub hb: SelfBlock<DriftP>,
}

/// A root whose own constraint holds its offset near its stored value
/// while pulling it to a target.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [(driftroot.offset - driftroot.offset_value) * 0.2, (driftroot.offset - driftroot.target) * 1.5] }))]
#[derive(Clone)]
pub struct DriftRoot {
    pub poses: refs::Vec<DriftP>,
    pub offset: Param<f64>,
    pub target: f64,
    pub hb: SelfBlock<DriftRoot>,
}

fn drift_root() -> DriftRoot {
    let mut rng = Lcg(36);
    let mut poses = refs::Vec::new();
    for k in 0..6 {
        let t = 0.3 * k as f64;
        poses.push(DriftP {
            s: Param::new(t),
            p: Param::new(vect2::new(t, 1.0 - t)),
            ea: SimpleEulerAngleParam::new(vect3::new(0.1 * t, -0.2 * t, 0.05)),
            ms: t + 0.4 + 0.1 * rng.uniform(),
            mp: vect2::new(t - 0.3, 1.2 - t),
            me: 0.2 + 0.1 * rng.uniform(),
            hb: SelfBlock::new(),
        });
    }
    DriftRoot { poses, offset: Param::new(0.0), target: 0.8, hb: SelfBlock::new() }
}

/// The parameters' values as a drift root holds them.
fn drift_state(s: &DriftRoot) -> Vec<f64> {
    let mut v = vec![s.offset.value];
    for p in s.poses.iter() {
        v.extend([p.s.value, p.p.value.x, p.p.value.y, p.ea.value.x, p.ea.value.y, p.ea.value.z]);
    }
    v
}

/// Reads of the parameters' stored values, an entity's and the root's,
/// are data the device packs: it assembles as the CPU does at parameters
/// away from those values and solves to the CPU's optimum.
#[test]
fn stored_parameter_values_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(drift_root, "the stored values");
}

/// A solve writes the optimum into the stored values; the next solve
/// through the same session holds the parameters near the new ones, on
/// the device as on the CPU.
#[test]
fn a_warm_solve_reads_the_stored_values_the_last_one_wrote() {
    use arael::simple_lm::{LmConfig, LmSession, SparseFaer};
    let cfg = LmConfig::<f64> { max_iters: 50, ..Default::default() };
    let cpu = |s: &DriftRoot| {
        let mut s = s.clone();
        let r = s.solve_with(&mut SparseFaer::<f64>::from_options(&whole_options(None)), &cfg).unwrap();
        (r.end_cost, s)
    };
    for d in devices() {
        let name = label(&d);
        let mut s = drift_root();
        let mut session = LmSession::new(SparseFaer::<f64>::from_options(&whole_options(Some(&d))));
        for solve in ["the first solve", "the warm solve"] {
            let (cpu_cost, cpu_model) = cpu(&s);
            let before = drift_state(&s);
            let r = session.solve(&mut s, &cfg).unwrap();
            assert!(session.solver().plan().unwrap().gpu_assembly, "{name}: {solve}");
            assert!((r.end_cost - cpu_cost).abs() <= 1e-9 * cpu_cost, "{name}: {solve}: {} against the CPU's {cpu_cost}", r.end_cost);
            assert_close(&drift_state(&s), &drift_state(&cpu_model), 1e-7, &format!("{name}: {solve}'s model"));
            assert!(drift_state(&s).iter().zip(&before).any(|(a, b)| (a - b).abs() > 1e-3), "{name}: {solve} moves the model");
        }
    }
}

/// A one-coordinate point held near where it was seen, and again under the
/// loss its `kind` picks of `loss_select`'s six.
#[arael::model]
#[arael(constraint(hb, { [(ptk.x - ptk.px) * 0.5] }))]
#[arael(constraint(hb, loss = |s| loss_select(ptk.kind, s, ptk.k2), { [(ptk.x - ptk.px) * 2.0] }))]
#[derive(Clone)]
pub struct PtK {
    pub x: Param<f64>,
    pub px: f64,
    pub kind: u32,
    pub k2: f64,
    pub hb: SelfBlock<PtK>,
}

/// A camera of three parameters, each held near where it was seen; its
/// `model` picks which of them an observation reads.
#[arael::model]
#[arael(constraint(hb, { [(camk.f - camk.pf) * 0.3, (camk.k - camk.pk) * 0.3, (camk.c - camk.pc) * 0.3] }))]
#[derive(Clone)]
pub struct CamK {
    pub model: u32,
    pub f: Param<f64>,
    pub k: Param<f64>,
    pub c: Param<f64>,
    pub pf: f64,
    pub pk: f64,
    pub pc: f64,
    pub hb: SelfBlock<CamK>,
}

/// An observation whose camera's model picks the residual (model 0 reads
/// f, 1 f and k, any other c: arms writing apart), behind a depth
/// `branch`, under the loss its own `kind` picks.
#[arael::model]
#[arael(constraint(hb, loss = |s| match obsk.kind { 0 => s, _ => loss_cauchy(s, obsk.k2) }, {
    [branch(pt.x, match cam.model {
        0 => pt.x * cam.f - obsk.m,
        1 => pt.x * cam.f * (1.0 + cam.k * pt.x * pt.x) - obsk.m,
        _ => pt.x + cam.c - obsk.m,
    }, 0.0)]
}))]
#[derive(Clone)]
pub struct ObsK {
    #[arael(ref = root.pts)]
    pub pt: Ref<PtK>,
    #[arael(ref = root.cams)]
    pub cam: Ref<CamK>,
    pub m: f64,
    pub k2: f64,
    pub kind: u32,
    pub hb: CrossBlock<PtK, CamK>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct SelWorld {
    pub pts: refs::Vec<PtK>,
    pub cams: refs::Vec<CamK>,
    pub obs: std::vec::Vec<ObsK>,
}

/// Four cameras of models 0, 1, 2 and 7 (7 outside the arms: the default),
/// twelve points (two behind the depth `branch`), every camera seeing
/// every point, the measurements the truth plus noise.
fn sel_world() -> SelWorld {
    let mut rng = Lcg(41);
    let cam_truth = [(0u32, 1.5, 0.1, 0.4), (1, 0.8, 0.2, -0.3), (2, 1.1, -0.1, 0.7), (7, 1.3, 0.05, -0.6)];
    let mut cams = refs::Vec::new();
    for &(model, f, k, c) in &cam_truth {
        cams.push(CamK { model, f: Param::new(f * 1.1), k: Param::new(k + 0.05), c: Param::new(c - 0.1), pf: f, pk: k, pc: c, hb: SelfBlock::new() });
    }
    let pt_truth: Vec<f64> = (0..12).map(|i| if i % 6 == 5 { -0.5 - 0.1 * i as f64 } else { 0.5 + 0.15 * i as f64 }).collect();
    let mut pts = refs::Vec::new();
    for (i, &x) in pt_truth.iter().enumerate() {
        pts.push(PtK { x: Param::new(x + 0.05 * rng.uniform()), px: x, kind: (i % 6) as u32, k2: 0.04, hb: SelfBlock::new() });
    }
    let mut obs = std::vec::Vec::new();
    for (i, &x) in pt_truth.iter().enumerate() {
        for (j, &(model, f, k, c)) in cam_truth.iter().enumerate() {
            let pred = match model {
                0 => x * f,
                1 => x * f * (1.0 + k * x * x),
                _ => x + c,
            };
            obs.push(ObsK { pt: pts.ref_at(i), cam: cams.ref_at(j), m: pred + 0.01 * rng.uniform(), k2: 0.25, kind: ((i + j) % 2) as u32, hb: CrossBlock::new() });
        }
    }
    SelWorld { pts, cams, obs }
}

/// Index selects in a body (arms writing apart, a default arm taken by a
/// model outside the arms) and picking losses, from instance and entity
/// data: the device assembles as the CPU does and solves to its optimum.
#[test]
fn index_selects_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(sel_world, "index selects");
}

/// A point whose select has no default.
#[arael::model]
#[arael(constraint(hb, { [select(selnan.k, selnan.x - 1.0, selnan.x * 2.0 - 1.0)] }))]
#[derive(Clone)]
pub struct SelNan {
    pub x: Param<f64>,
    pub k: u32,
    pub hb: SelfBlock<SelNan>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct SelNanRoot {
    pub items: refs::Vec<SelNan>,
}

/// An index outside a select's arms, with no default: the CPU's solve
/// panics; the device's values turn NaN, and its solve fails with a NaN
/// diagonal at that instance's parameter.
#[test]
fn a_select_index_outside_its_arms_fails_the_device_solve_at_its_parameter() {
    use arael::simple_lm::{DiagonalFault, LmConfig, SolveFailureKind};
    let model = || {
        let mut items = refs::Vec::new();
        for k in [0u32, 1, 7, 0] {
            items.push(SelNan { x: Param::new(0.3), k, hb: SelfBlock::new() });
        }
        SelNanRoot { items }
    };
    let solve = |gpu: Option<&GpuDevice>| {
        let mut s = model();
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &LmConfig::<f64>::default())
    };
    assert!(std::panic::catch_unwind(|| solve(None)).is_err(), "the CPU's select panics");
    for d in devices() {
        let failure = solve(Some(&d)).expect_err("the device solve fails");
        assert_eq!(failure.kind, SolveFailureKind::DegenerateDiagonal { param: 2, fault: DiagonalFault::Nan }, "{}: {failure}", label(&d));
    }
}

/// The kernel library's fast atan and atan2 against arael::utils' on the
/// CPU, over signs, zeros, the folds' boundaries and wide magnitudes: the
/// same folds, polynomial and constants, equal to rounding (the device may
/// contract a product and an add into one rounding), in f32 and f64.
#[test]
fn the_device_fast_atan_is_the_cpus() {
    use arael::__gpu::{Arg, Device, Launch, Wide, ASSEMBLY, WIDE};
    const KERNEL: &str = r#"
KERNEL void fast_atans(GLOBAL const double* in, GLOBAL double* out_d, GLOBAL float* out_f, u32 n) {
    const u32 i = global_x();
    if (i >= n) return;
    const double y = in[2 * i], x = in[2 * i + 1];
    out_d[2 * i] = arael_fast_atan(y);
    out_d[2 * i + 1] = arael_fast_atan2(y, x);
    out_f[2 * i] = arael_fast_atan((float)y);
    out_f[2 * i + 1] = arael_fast_atan2((float)y, (float)x);
}
"#;
    let dev = device();
    let source = format!("{WIDE}\n{ASSEMBLY}\n{KERNEL}");
    let kernel = dev.compile(&source, &format!("{}#define SCALAR double\n", Wide::F64.defines()), "fast_atans").unwrap();
    let mags = [0.0, 1e-8, 0.1, 0.2679491924311227, 0.268, 0.5, 0.57735, 1.0, 1.0000001, 3.7320508, 10.0, 1e6];
    let vals: Vec<f64> = mags.iter().flat_map(|&m| [m, -m]).collect();
    let pairs: Vec<(f64, f64)> = vals.iter().flat_map(|&y| vals.iter().map(move |&x| (y, x))).collect();
    let n = pairs.len();
    let input: Vec<f64> = pairs.iter().flat_map(|&(y, x)| [y, x]).collect();
    let mut inp = dev.alloc(8 * input.len()).unwrap();
    dev.write(&mut inp, 0, arael::__gpu::bytemuck::cast_slice(&input)).unwrap();
    let (out_d, out_f) = (dev.alloc(16 * n).unwrap(), dev.alloc(8 * n).unwrap());
    let args = [Arg::Block(&inp, 0), Arg::Block(&out_d, 0), Arg::Block(&out_f, 0), Arg::U32(n as u32)];
    dev.launch(&kernel, Launch::linear(n as u32, 256), &args).unwrap();
    let (mut d, mut f) = (vec![0f64; 2 * n], vec![0f32; 2 * n]);
    dev.read(&out_d, 0, arael::__gpu::bytemuck::cast_slice_mut(&mut d)).unwrap();
    dev.read(&out_f, 0, arael::__gpu::bytemuck::cast_slice_mut(&mut f)).unwrap();
    for (k, &(y, x)) in pairs.iter().enumerate() {
        let (yf, xf) = (y as f32, x as f32);
        let cpu_d = [arael::utils::fast_atan(y), arael::utils::fast_atan2(y, x)];
        let cpu_f = [arael::utils::fast_atan(yf), arael::utils::fast_atan2(yf, xf)];
        for j in 0..2 {
            assert!((d[2 * k + j] - cpu_d[j]).abs() <= 8.0 * f64::EPSILON, "f64 ({y}, {x}) [{j}]: {} vs {}", d[2 * k + j], cpu_d[j]);
            assert!((f[2 * k + j] - cpu_f[j]).abs() <= 8.0 * f32::EPSILON, "f32 ({y}, {x}) [{j}]: {} vs {}", f[2 * k + j], cpu_f[j]);
        }
    }
}

/// A purely symbolic user function (Form A), inlined into the body.
#[arael::function]
fn sq_dev(x: arael::sym::E) -> arael::sym::E {
    x * x
}

/// A typed user function (Form C), inlined into the body.
#[arael::function]
fn offset_dev(p: arael::sym::vect2sym, l: arael::sym::vect2sym) -> arael::sym::vect2sym {
    l - p
}

/// A 2D pose held near where it was seen.
#[arael::model]
#[arael(constraint(hb, { [(poseb.pos.x - poseb.px) * 0.2, (poseb.pos.y - poseb.py) * 0.2, (poseb.th - poseb.pth) * 0.2] }))]
#[derive(Clone)]
pub struct PoseB {
    pub pos: Param<vect2<f64>>,
    pub th: Param<f64>,
    pub px: f64,
    pub py: f64,
    pub pth: f64,
    pub hb: SelfBlock<PoseB>,
}

/// A landmark at a known place seen from a pose by bearing and squared
/// range, through the user functions; the root's atan2 the fast one.
#[arael::model]
#[arael(constraint(p.hb, {
    let d = offset_dev(p.pos, bearing.l);
    [rad_diff(atan2(d.y, d.x) - p.th, bearing.b) * 20.0, (sq_dev(d.x) + sq_dev(d.y) - bearing.r2) * 0.5]
}))]
#[derive(Clone)]
pub struct Bearing {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseB>,
    pub l: vect2<f64>,
    pub b: f64,
    pub r2: f64,
}

#[arael::model]
#[arael(root, fast_atan)]
#[derive(Clone)]
pub struct BearingWorld {
    pub poses: refs::Vec<PoseB>,
    pub bearings: std::vec::Vec<Bearing>,
}

/// Six poses on a circle seeing five landmarks, the measurements the truth
/// plus noise, the poses moved off it.
fn bearing_world() -> BearingWorld {
    let mut rng = Lcg(61);
    let truth: Vec<(f64, f64, f64)> = (0..6).map(|k| {
        let a = k as f64;
        (3.0 * a.cos(), 3.0 * a.sin(), a + 1.2)
    }).collect();
    let marks = [(0.5, 0.3), (-1.0, 1.5), (2.0, -1.0), (-2.5, -2.0), (4.5, 1.0)];
    let mut poses = refs::Vec::new();
    for &(x, y, th) in &truth {
        poses.push(PoseB {
            pos: Param::new(vect2::new(x + 0.1 * rng.uniform(), y + 0.1 * rng.uniform())),
            th: Param::new(th + 0.05 * rng.uniform()),
            px: x,
            py: y,
            pth: th,
            hb: SelfBlock::new(),
        });
    }
    let mut bearings = std::vec::Vec::new();
    for (k, &(x, y, th)) in truth.iter().enumerate() {
        for &(lx, ly) in &marks {
            let (dx, dy) = (lx - x, ly - y);
            bearings.push(Bearing {
                p: poses.ref_at(k),
                l: vect2::new(lx, ly),
                b: arael::utils::rad_diff(dy.atan2(dx), th) + 0.002 * rng.uniform(),
                r2: dx * dx + dy * dy + 0.01 * rng.uniform(),
            });
        }
    }
    BearingWorld { poses, bearings }
}

/// A root's fast atan2 and user functions with a symbolic body (inlined)
/// assemble on the device as on the CPU and solve to its optimum.
#[test]
fn fast_atan_and_symbolic_user_functions_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(bearing_world, "fast atan2 and user functions");
}

/// A user function whose value is Rust code (Form B): no C++ form.
#[arael::function(opaque_dev, derivs = [x * 0.0 + 1.0])]
fn opaque_dev_eval(x: f64) -> f64 {
    x
}

#[arael::model]
#[arael(constraint(hb, { [opaque_dev(opaquee.x) - 1.0] }))]
#[derive(Clone)]
pub struct OpaqueE {
    pub x: Param<f64>,
    pub hb: SelfBlock<OpaqueE>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct OpaqueRoot {
    pub items: refs::Vec<OpaqueE>,
}

#[test]
fn a_user_function_of_rust_code_has_no_device_form() {
    let s = OpaqueRoot { items: refs::Vec::new() };
    let why = LmProblemInternals::<f64>::device_unsupported(&s).expect("a Rust-coded user function has no device form");
    assert!(why.contains("OpaqueE") && why.contains("opaque_dev_eval") && why.contains("no C++ form"), "{why}");
}

/// Data fields whose paths look alike once spelled as record fields:
/// `p.x` and `p_x` of an entity, `q.x` and `q_x` of the root.
#[arael::model]
#[arael(constraint(hb, { [(clash.x - clash.p.x - clash.p_x) * 2.0, (clash.x - clash.p.x - clash.p_x) * (root.q.x - root.q_x)] }))]
#[derive(Clone)]
pub struct Clash {
    pub x: Param<f64>,
    pub p: vect2<f64>,
    pub p_x: f64,
    pub hb: SelfBlock<Clash>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct ClashRoot {
    pub items: refs::Vec<Clash>,
    pub q: vect2<f64>,
    pub q_x: f64,
}

fn clash_root() -> ClashRoot {
    let mut items = refs::Vec::new();
    for k in 0..5 {
        let t = 0.4 * k as f64;
        items.push(Clash { x: Param::new(t + 0.2), p: vect2::new(t, -1.0), p_x: 0.1 * t, hb: SelfBlock::new() });
    }
    ClashRoot { items, q: vect2::new(0.7, 9.0), q_x: -0.2 }
}

/// Paths that spell alike as record fields (`p.x`, `p_x`) take a field
/// each; the model builds (it did not, with or without the `gpu`
/// feature) and assembles on the device as on the CPU.
#[test]
fn data_fields_of_alike_names_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(clash_root, "alike names");
}

/// Optional data of an entity, read under a guard on it.
#[arael::model]
#[derive(Clone)]
pub struct Extra {
    pub off: f64,
}

#[arael::model]
#[arael(constraint(hb, { [(optent.x - optent.px) * 0.5] }))]
#[arael(constraint(hb, guard = self.has_extra, { [(optent.x - optent.extra.off) * 2.0] }))]
#[arael(constraint(hb, guard = root.has_bias, { [(optent.x - optent.px) * root.bias.off] }))]
#[derive(Clone)]
pub struct OptEnt {
    pub x: Param<f64>,
    pub px: f64,
    pub extra: Option<Extra>,
    pub has_extra: bool,
    pub hb: SelfBlock<OptEnt>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct OptRoot {
    pub items: refs::Vec<OptEnt>,
    pub bias: Option<Extra>,
    pub has_bias: bool,
}

/// Six entities, every other one with its optional data; the root's
/// optional data present or not.
fn opt_root(bias: bool) -> OptRoot {
    let mut items = refs::Vec::new();
    for k in 0..6 {
        let some = k % 2 == 0;
        let px = 0.3 * k as f64;
        items.push(OptEnt {
            x: Param::new(px + 0.2),
            px,
            extra: some.then(|| Extra { off: px }),
            has_extra: some,
            hb: SelfBlock::new(),
        });
    }
    OptRoot { items, bias: bias.then(|| Extra { off: 0.4 }), has_bias: bias }
}

/// Optional data of entities and of the root read under guards on it:
/// the device writes what a `None` leaves as NaN, which no kept instance
/// reads (it panicked writing it), and assembles as the CPU does.
#[test]
fn optional_data_read_under_a_guard_assembles_and_solves_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(|| opt_root(true), "the root's data present");
    assembles_and_solves_as_on_the_cpu(|| opt_root(false), "the root's data absent");
}

/// A component whose value follows a reference its `update` moves after
/// every accepted step (`ref_c`), which the device cannot run: its caches
/// are computed from data, not from parameters alone.
#[arael::model]
#[arael(component)]
#[derive(Clone)]
pub struct Moved {
    pub ref_c: f64,
    pub d: Param<f64>,
    #[arael(symbolic = ref_c + d)]
    pub c: f64,
}

impl arael::model::Component for Moved {
    fn start(&mut self) {
        self.ref_c = self.c;
        self.d.value = 0.0;
    }
    fn update(&mut self) {
        self.ref_c += self.d.value;
        self.d.value = 0.0;
    }
    fn finish(&mut self) {
        self.c = self.ref_c + self.d.value;
    }
}

#[arael::model]
#[arael(constraint(hb, { [movedholder.m.c - 2.0] }))]
#[derive(Clone)]
pub struct MovedHolder {
    pub m: Moved,
    pub hb: SelfBlock<MovedHolder>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct MovedRoot {
    pub items: refs::Vec<MovedHolder>,
}

/// A `compute =` field read by a body: Rust code reassigns it on every
/// update, which the device cannot run.
#[arael::model]
#[arael(constraint(hb, { [computed.c - 1.0] }))]
#[derive(Clone)]
pub struct Computed {
    pub w: Param<f64>,
    #[arael(compute = w.exp())]
    pub c: f64,
    pub hb: SelfBlock<Computed>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct ComputedRoot {
    pub items: refs::Vec<Computed>,
}

/// A stored parameter value inside a component read by a body: the
/// component's update stores it between steps (`Moved::update` zeroes
/// `d.value`), which the device cannot follow.
#[arael::model]
#[arael(constraint(hb, { [(storedholder.m.d - storedholder.m.d_value) * 2.0] }))]
#[derive(Clone)]
pub struct StoredHolder {
    pub m: Moved,
    pub hb: SelfBlock<StoredHolder>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct StoredRoot {
    pub items: refs::Vec<StoredHolder>,
}

#[test]
fn a_constraint_reading_state_the_device_does_not_compute_has_no_device_form() {
    let s = MovedRoot { items: refs::Vec::new() };
    let why = LmProblemInternals::<f64>::device_unsupported(&s).expect("a cache over moved data has no device form");
    assert!(why.contains("MovedHolder") && why.contains("m.c") && why.contains("m.ref_c"), "{why}");
    let s = ComputedRoot { items: refs::Vec::new() };
    let why = LmProblemInternals::<f64>::device_unsupported(&s).expect("a `compute =` field has no device form");
    assert!(why.contains("Computed") && why.contains("computed.c") && why.contains("compute ="), "{why}");
    let s = StoredRoot { items: refs::Vec::new() };
    let why = LmProblemInternals::<f64>::device_unsupported(&s).expect("a stored value inside a component has no device form");
    assert!(why.contains("StoredHolder") && why.contains("storedholder.m.d.value") && why.contains("inside the component"), "{why}");
}

/// A Sim(3) pose (`ScaledTransformParam`): its pose a transform's reference
/// kind, its scale `exp(log_s)` a cache over an absolute parameter. Three
/// known points seen from it.
#[arael::model]
#[arael(constraint(hb, {
    let r = simpose.stp.rotation_matrix;
    let t = simpose.stp.translation;
    let s = simpose.stp.scale_factor;
    let p1 = r * simpose.x1 * s + t;
    let p2 = r * simpose.x2 * s + t;
    let p3 = r * simpose.x3 * s + t;
    [p1.x - simpose.m1.x, p1.y - simpose.m1.y, p1.z - simpose.m1.z,
     p2.x - simpose.m2.x, p2.y - simpose.m2.y, p2.z - simpose.m2.z,
     p3.x - simpose.m3.x, p3.y - simpose.m3.y, p3.z - simpose.m3.z]
}))]
#[derive(Clone)]
pub struct SimPose {
    pub stp: arael::transform::ScaledTransformParam<f64>,
    pub x1: vect3<f64>,
    pub x2: vect3<f64>,
    pub x3: vect3<f64>,
    pub m1: vect3<f64>,
    pub m2: vect3<f64>,
    pub m3: vect3<f64>,
    pub hb: SelfBlock<SimPose>,
}

/// The relative translation, rotation and log scale between two Sim(3)
/// poses.
#[arael::model]
#[arael(constraint(hb, {
    let ra = a.stp.rotation_matrix;
    let dt = ra.transpose() * (b.stp.translation - a.stp.translation) - simlink.dt;
    let dr = simlink.rmeas_t * (ra.transpose() * b.stp.rotation_matrix);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x, dt.y, dt.z,
     (c2.z - c3.y) * 0.5, (c3.x - c1.z) * 0.5, (c1.y - c2.x) * 0.5,
     b.stp.log_s - a.stp.log_s - simlink.dls]
}))]
#[derive(Clone)]
pub struct SimLink {
    #[arael(ref = root.poses)]
    pub a: Ref<SimPose>,
    #[arael(ref = root.poses)]
    pub b: Ref<SimPose>,
    pub dt: vect3<f64>,
    pub rmeas_t: arael::matrix::matrix3<f64>,
    pub dls: f64,
    pub hb: CrossBlock<SimPose, SimPose>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct SimWorld {
    pub poses: refs::Vec<SimPose>,
    pub links: std::vec::Vec<SimLink>,
}

/// Six Sim(3) poses along a turning path, each seeing three known points,
/// links between neighbours and one closing the loop, the measurements the
/// truth plus noise, the poses moved off it; the first pose fixed.
fn sim_world() -> SimWorld {
    use arael::quatern::quaternd;
    use arael::vect::vect3d;
    let mut rng = Lcg(51);
    let truth: Vec<(vect3d, quaternd, f64)> = (0..6)
        .map(|i| {
            let th = i as f64 * 0.4;
            let q = (quaternd::from_axis_angle(vect3d::new(0.0, 0.0, 1.0), th * 0.9) * quaternd::from_axis_angle(vect3d::new(1.0, 0.0, 0.0), 0.15 * th)).unit();
            (vect3d::new(i as f64 * 1.1, 0.2 * th.sin(), -0.1 * th), q, 1.0 + 0.06 * i as f64)
        })
        .collect();
    let jitter = |rng: &mut Lcg, s: f64| vect3d::new(s * rng.uniform(), s * rng.uniform(), s * rng.uniform());
    let xs = [vect3d::new(1.0, 0.0, 0.0), vect3d::new(0.0, 1.0, 0.5), vect3d::new(0.3, -0.4, 1.0)];
    let mut poses = refs::Vec::new();
    for (k, &(t, q, s)) in truth.iter().enumerate() {
        let r = q.rotation_matrix();
        let m: Vec<vect3d> = xs.iter().map(|&x| r * x * s + t + jitter(&mut rng, 0.01)).collect();
        let stp = if k == 0 {
            arael::transform::ScaledTransformParam::fixed(t, q, s)
        } else {
            let dq = quaternd::from_rotation_vector(jitter(&mut rng, 0.05));
            arael::transform::ScaledTransformParam::new(t + jitter(&mut rng, 0.05), (q * dq).unit(), s * 1.05)
        };
        poses.push(SimPose { stp, x1: xs[0], x2: xs[1], x3: xs[2], m1: m[0], m2: m[1], m3: m[2], hb: SelfBlock::new() });
    }
    let mut links = std::vec::Vec::new();
    for (a, b) in (0..6).map(|a| (a, (a + 1) % 6)) {
        let ((ta, qa, sa), (tb, qb, sb)) = (truth[a], truth[b]);
        let ra = qa.rotation_matrix();
        links.push(SimLink {
            a: poses.ref_at(a),
            b: poses.ref_at(b),
            dt: ra.transpose() * (tb - ta) + jitter(&mut rng, 0.01),
            rmeas_t: (ra.transpose() * qb.rotation_matrix()).transpose(),
            dls: sb.ln() - sa.ln() + 0.002 * rng.uniform(),
            hb: CrossBlock::new(),
        });
    }
    SimWorld { poses, links }
}

/// A `#[arael(symbolic = exp(w))]` field of an entity: the body reads its
/// cache, which the device computes from the parameter as it goes (the
/// stored field follows the parameter between steps).
#[arael::model]
#[arael(constraint(hb, { [(curvee.target - curvee.base * curvee.scale) * curvee.isigma] }))]
#[derive(Clone)]
pub struct CurveE {
    pub w: Param<f64>,
    pub base: f64,
    pub target: f64,
    pub isigma: f64,
    #[arael(symbolic = exp(w))]
    pub scale: f64,
    pub hb: SelfBlock<CurveE>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct CurvesE {
    pub curves: refs::Vec<CurveE>,
}

fn curves_e() -> CurvesE {
    let mut curves = refs::Vec::new();
    for k in 0..6 {
        let (base, w) = (1.0 + 0.5 * k as f64, 0.3 * k as f64 - 0.7);
        curves.push(CurveE { w: Param::new(w + 0.3), base, target: base * w.exp(), isigma: 1.5, scale: 0.0, hb: SelfBlock::new() });
    }
    CurvesE { curves }
}

/// A user component over parameters alone: its caches, values and a
/// declared derivative, computed by the device from the parameters.
#[arael::model]
#[arael(component)]
#[derive(Clone)]
pub struct Polar {
    pub r: Param<f64>,
    pub th: Param<f64>,
    #[arael(symbolic = r * cos(th))]
    pub x: f64,
    #[arael(symbolic = r * sin(th))]
    pub y: f64,
    #[arael(deriv = x, by = th)]
    pub x_th: [f64; 1],
}

impl arael::model::Component for Polar {}

#[arael::model]
#[arael(constraint(hb, { [(beacon.polar.x - beacon.mx) * 2.0, (beacon.polar.y - beacon.my) * 2.0, (beacon.polar.r - beacon.mr) * 0.5] }))]
#[derive(Clone)]
pub struct Beacon {
    pub polar: Polar,
    pub mx: f64,
    pub my: f64,
    pub mr: f64,
    pub hb: SelfBlock<Beacon>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Beacons {
    pub beacons: refs::Vec<Beacon>,
}

fn beacons() -> Beacons {
    let mut rng = Lcg(52);
    let mut beacons = refs::Vec::new();
    for k in 0..6 {
        let (r, th) = (1.0 + 0.4 * k as f64, 0.5 * k as f64 - 1.0);
        beacons.push(Beacon {
            polar: Polar { r: Param::new(r + 0.1), th: Param::new(th - 0.1), x: 0.0, y: 0.0, x_th: [0.0] },
            mx: r * th.cos() + 0.01 * rng.uniform(),
            my: r * th.sin() + 0.01 * rng.uniform(),
            mr: r,
            hb: SelfBlock::new(),
        });
    }
    Beacons { beacons }
}

/// `symbolic =` and `deriv =` caches over parameters, on an entity and in a
/// user component, and the `ScaledTransformParam` (a transform's pose and
/// a scale cache): the device computes them, assembles as the CPU does and
/// solves to its optimum.
#[test]
fn caches_over_parameters_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(curves_e, "a symbolic field of an entity");
    assembles_and_solves_as_on_the_cpu(beacons, "a user component");
    assembles_and_solves_as_on_the_cpu(sim_world, "Sim(3) poses");
    // the device's re-centred references handed back to the model
    let solved = |gpu: Option<&GpuDevice>| {
        let mut s = sim_world();
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(gpu));
        let cfg = arael::simple_lm::LmConfig::<f64> { max_iters: 50, ..Default::default() };
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        s.deserialize(&r.x);
        s.poses
            .iter()
            .flat_map(|p| {
                let (t, q) = (p.stp.translation, p.stp.rotation);
                [t.x, t.y, t.z, q.t, q.v.x, q.v.y, q.v.z, p.stp.scale]
            })
            .collect::<Vec<f64>>()
    };
    let cpu = solved(None);
    for d in devices() {
        assert_close(&solved(Some(&d)), &cpu, 1e-7, &format!("{}: the Sim(3) poses written back", label(&d)));
    }
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
/// The root's kernels name what each piece of the C++ is: the root's,
/// the entity tables' and the constraints' source lines, a constraint's
/// attribute, the model path of each data record field and of each leaf
/// a kernel reads (an instance's field by the attribute's name, not the
/// walk's); the code indented by its braces.
#[test]
fn a_roots_kernels_are_commented_with_the_model() {
    use arael::__gpu::DeviceRoot;
    let cpp = DeviceRoot::<f64>::device_cpp(&slam_path(2, 2, 1));
    let me = include_str!("device_assembly.rs");
    let line = |text: &str| me.lines().position(|l| l == text).unwrap() + 1;
    let path = "tests/device_assembly.rs";
    for want in [
        format!("// Device assembly of the root `SPath` ({path}:{}), generated by #[arael::model].", line("pub struct SPath {")),
        format!("//   E0: `SPose` ({path}:{}), self block `hb_pose`, parameters: pos.work().x, pos.work().y, pos.work().z, ea.work().x, ea.work().y, ea.work().z", line("pub struct SPose {")),
        "    F gps_ucov_ur_l0_r_dx; // gps_cov_r[0].x".to_string(),
        format!("// C4: a constraint of `SFrine` ({path}:{}), 2 rows", line("#[arael(constraint(hb, parent = lm, {")),
        "//       let r_f = sfrine.mf2r.transpose() * r_r;".to_string(),
        "// Instance data D5: sfrine.mf2r[0].x".to_string(),
        "        const F __l15 = COL(B, F, C0_EV0)[s0 * 6u + 0u]; // spose.pos.work().x".to_string(),
    ] {
        assert!(cpp.lines().any(|l| l == want), "no line `{want}` in\n{cpp}");
    }
    assert!(!cpp.contains("__frine."), "an instance field by the walk's name");
}

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
