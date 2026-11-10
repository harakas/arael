//! The device assembly (feature `gpu-cuda`, `gpu-hip`, `gpu-metal` or
//! `gpu-cpu`): a root whose constraints all have a device form assembles
//! its gradient, Hessian tiles and cost on the GPU to what the CPU assembly
//! gives against the same pattern.
#![cfg(any(feature = "gpu-cuda", feature = "gpu-hip", feature = "gpu-cpu", gpu_metal))]

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
// The cost summed in f64: the solver compares costs, and f32 sums leave
// differences of rounding noise that can stop a solve short of its optimum.
#[arael(root, f32, cost_f64)]
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

/// The GPU runtimes compiled in that solve an f64 model as written, on
/// device 0, in both wide scalars. Metal, without f64, solves the f32 twins
/// ([`f32_devices`]).
fn devices() -> Vec<GpuDevice> {
    #[allow(unused_mut)]
    let mut v = Vec::new();
    #[cfg(feature = "gpu-cuda")]
    v.extend([GpuDevice::cuda(0), GpuDevice::cuda(0).with_wide(arael::simple_lm::GpuWide::Df32)]);
    #[cfg(feature = "gpu-hip")]
    v.extend([GpuDevice::hip(0), GpuDevice::hip(0).with_wide(arael::simple_lm::GpuWide::Df32)]);
    #[cfg(feature = "gpu-cpu")]
    v.extend([GpuDevice::cpu(), GpuDevice::cpu().with_wide(arael::simple_lm::GpuWide::Df32)]);
    v
}

/// Every GPU runtime compiled in, for f32 models: [`devices`] and Metal (in
/// df32 alone).
fn f32_devices() -> Vec<GpuDevice> {
    #[allow(unused_mut)]
    let mut v = devices();
    #[cfg(gpu_metal)]
    v.push(GpuDevice::metal(0));
    assert!(!v.is_empty(), "no GPU runtime compiled in");
    v
}

/// A device's runtime and wide scalar, as the verbose log names them.
fn label(d: &GpuDevice) -> String {
    let runtime = match d.backend {
        arael::simple_lm::GpuBackend::Cuda => "cuda",
        arael::simple_lm::GpuBackend::Hip => "hip",
        arael::simple_lm::GpuBackend::Cpu => "cpu",
        arael::simple_lm::GpuBackend::Metal => "metal",
    };
    match d.wide() {
        arael::simple_lm::GpuWide::F64 => runtime.to_string(),
        arael::simple_lm::GpuWide::Df32 => format!("{runtime} df32"),
    }
}

/// Every backend but Metal computes in f64; Metal's wide values are df32.
#[test]
fn a_device_says_whether_it_has_f64() {
    use arael::simple_lm::GpuWide;
    for d in [GpuDevice::cuda(0), GpuDevice::hip(0), GpuDevice::cpu()] {
        assert!(d.has_f64() && d.backend.has_f64(), "{}", label(&d));
        assert_eq!(d.wide(), GpuWide::F64, "{}", label(&d));
    }
    let metal = GpuDevice::metal(0);
    assert!(!metal.has_f64() && !metal.backend.has_f64());
    assert_eq!(metal.wide(), GpuWide::Df32);
    assert!(devices().iter().all(|d| d.has_f64()));
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

/// What runs a solve of the consistent scene to its optimum: the scene
/// descends a long shallow valley (about 200 iterations), so no small-step
/// stop short of it, and the damping floored for the bundle adjustment's
/// gauge freedom.
fn to_the_optimum<T: Float>() -> arael::simple_lm::LmConfig<T> {
    arael::simple_lm::LmConfig::<T> {
        max_iters: 1000,
        rel_precision: T::from(1e-8).unwrap(),
        abs_precision: T::from(1e-9).unwrap(),
        lambda_floor: T::from(1e-6).unwrap(),
        ..Default::default()
    }
}

/// The optimum of the consistent scene, solved as an f64 model on the CPU.
fn consistent_optimum() -> f64 {
    let (cameras, points, observations) = consistent_scene::<f64>();
    let mut s = Scene { cameras, points, observations };
    let mut x = Vec::new();
    s.serialize(&mut x);
    let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(&options(None, false));
    arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &to_the_optimum()).unwrap().end_cost
}

/// The same for an f32 root, run to the optimum: the device route, its sums
/// wide, lands on the f64 model's optimum to 1e-4; the CPU's f32 route, its
/// gradient and Hessian summed in f32, to 1e-3.
#[test]
fn an_f32_solve_through_the_device_assembly_reaches_the_optimum() {
    let run = |gpu: Option<&GpuDevice>| {
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&options(gpu, true));
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &to_the_optimum()).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        r.end_cost as f64
    };
    let optimum = consistent_optimum();
    let off = |c: f64| (c - optimum).abs() / optimum;
    let cpu = run(None);
    assert!(off(cpu) <= 1e-3, "the CPU's f32 route: {cpu} against the optimum {optimum}");
    for d in f32_devices() {
        let device = run(Some(&d));
        assert!(off(device) <= 1e-4, "{}: {device} against the optimum {optimum}", label(&d));
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
pub struct QCamera<T: Float> {
    pub t: Param<vect3<T>>,
    pub q: QuaternionParam<T>,
    pub intr: Param<vect3<T>>,
    pub hb: SelfBlock<QCamera<T>, T>,
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
pub struct QObs<T: Float> {
    #[arael(ref = root.cameras)]
    pub cam: Ref<QCamera<T>>,
    #[arael(ref = root.points)]
    pub pt: Ref<Point<T>>,
    pub xy: vect2<T>,
    pub hb: CrossBlock<QCamera<T>, Point<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct QScene {
    pub cameras: refs::Vec<QCamera<f64>>,
    pub points: refs::Vec<Point<f64>>,
    pub observations: std::vec::Vec<QObs<f64>>,
}

#[arael::model]
#[arael(root, f32, cost_f64)]
#[derive(Clone)]
pub struct QSceneF {
    pub cameras: refs::Vec<QCamera<f32>>,
    pub points: refs::Vec<Point<f32>>,
    pub observations: std::vec::Vec<QObs<f32>>,
}

type QParts<T> = (refs::Vec<QCamera<T>>, refs::Vec<Point<T>>, std::vec::Vec<QObs<T>>);

/// [`consistent_scene`] with the cameras turned by quaternions.
fn quaternion_parts<T: Float>() -> QParts<T> {
    let (cams, points, obs) = consistent_scene::<T>();
    let mut cameras = refs::Vec::new();
    for c in cams.iter() {
        let ea = c.ea.value;
        let ea = vect3::new(T::from(ea.x).unwrap(), T::from(ea.y).unwrap(), T::from(ea.z).unwrap());
        cameras.push(QCamera { t: c.t.clone(), q: QuaternionParam::from_euler_angles(ea), intr: c.intr.clone(), hb: SelfBlock::new() });
    }
    let observations = obs
        .iter()
        .map(|o| QObs { cam: cameras.ref_at(o.cam.index() as usize), pt: points.ref_at(o.pt.index() as usize), xy: o.xy, hb: CrossBlock::new() })
        .collect();
    (cameras, points, observations)
}

fn quaternion_scene() -> QScene {
    let (cameras, points, observations) = quaternion_parts::<f64>();
    QScene { cameras, points, observations }
}

fn quaternion_scene_f32() -> QSceneF {
    let (cameras, points, observations) = quaternion_parts::<f32>();
    QSceneF { cameras, points, observations }
}

/// The model `s` a solve (`r`) left behind costs what the solve ended at:
/// the solve wrote its parameters (re-centred references included) back.
/// To 1e-5 of the end cost, plus 1e-7 of the start's for the f32 rounding
/// of residuals small against their terms.
fn left_behind_costs<P: LmProblem<f32> + RootProblem<f32>>(s: &mut P, r: &arael::simple_lm::LmResult<f32>, what: &str) {
    let mut x = Vec::new();
    s.serialize(&mut x);
    let left = s.calc_cost(&x) as f64;
    let (start, end) = (r.start_cost as f64, r.end_cost as f64);
    assert!((left - end).abs() <= 1e-5 * end + 1e-7 * start, "{what}: the model left behind costs {left}, the solve ended at {end}");
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
    if checks_f64("the quaternion scene") {
        for (wide, tol) in [(Wide::F64, 1e-11), (Wide::Df32, 1e-10)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        }
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

/// The f32 twin: the quaternion cameras assemble on the device as on the
/// CPU, solve on every device to the f64 scene's optimum, and the solve
/// writes the re-centred rotations back into the model.
#[test]
fn an_f32_quaternion_scene_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::simple_lm::SparseFaer;
    let what = "the f32 quaternion scene";
    f32_twin_assembles_and_solves(quaternion_scene, quaternion_scene_f32, |gpu| options(gpu, true), what);
    for d in f32_devices() {
        let mut s = quaternion_scene_f32();
        let start: Vec<f32> = s.cameras.iter().flat_map(|c| [c.q.value.t, c.q.value.v.x, c.q.value.v.y, c.q.value.v.z]).collect();
        let r = s.solve_with(&mut SparseFaer::<f32>::from_options(&options(Some(&d), true)), &to_the_optimum()).unwrap();
        let moved = s.cameras.iter().flat_map(|c| [c.q.value.t, c.q.value.v.x, c.q.value.v.y, c.q.value.v.z]).zip(&start).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(moved > 1e-4, "{what} {}: the rotations moved by {moved}", label(&d));
        left_behind_costs(&mut s, &r, &format!("{what} {}", label(&d)));
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

/// An f32 root whose rotations keep their state in df32 (`wide = df32`),
/// run to the optimum, lands on the f64 model's optimum to 1e-4 on the
/// device as on the CPU, and the solve writes the model back: its rotations
/// moved, and the model left behind costs what the solve ended at (the pack
/// walk reads the df32 steps and reference rotations, and the device's
/// re-centred rotations come back into them). The routes' rotations
/// themselves differ along the scene's gauge freedom and are not compared.
#[test]
fn a_df32_wide_root_solves_on_the_device_as_on_the_cpu() {
    let optimum = consistent_optimum();
    let run = |gpu: Option<&GpuDevice>| {
        let name = gpu.map_or("the CPU".to_string(), label);
        let mut s = wide_scene(consistent_scene::<f32>());
        assert_eq!(LmProblemInternals::<f32>::device_unsupported(&s), None);
        let angles = |s: &SceneW| -> Vec<f64> { s.cameras.iter().flat_map(|c| [c.ea.value.x, c.ea.value.y, c.ea.value.z]).map(|a| a.to_f64()).collect() };
        let start = angles(&s);
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&options(gpu, true));
        let r = s.solve_with(&mut solver, &to_the_optimum()).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        let moved = angles(&s).iter().zip(&start).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
        assert!(moved > 1e-3, "{name}: the solved rotations back in the model: moved by {moved}");
        let mut x = Vec::new();
        s.serialize(&mut x);
        let left = arael::simple_lm::LmProblem::<f32>::calc_cost(&mut s, &x) as f64;
        let end = r.end_cost as f64;
        assert!((left - end).abs() <= 1e-5 * end, "{name}: the model left behind costs {left}, the solve ended at {end}");
        assert!((end - optimum).abs() <= 1e-4 * optimum, "{name}: {end} against the optimum {optimum}");
    };
    run(None);
    for d in f32_devices() {
        run(Some(&d));
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

/// The data the warm solves below change: an observation, a camera's
/// rotation, a point.
fn change_scene<T: Float>(cameras: &mut refs::Vec<Camera<T>>, points: &mut refs::Vec<Point<T>>, observations: &mut [Obs<T>]) {
    observations[5].xy.x = observations[5].xy.x + T::from(3.0).unwrap();
    cameras[1usize].ea.value.z += 0.02;
    points[7usize].pos.value.y = points[7usize].pos.value.y - T::from(0.05).unwrap();
}

/// The f32 twin: a device solve writes the model back, and after its data
/// changed a warm solve through the same session reads it again; each
/// lands on the optimum of the f64 scene as it then is.
#[test]
fn an_f32_device_solve_writes_the_model_back_and_a_warm_one_reads_it_again() {
    use arael::simple_lm::{LmSession, SparseFaer};
    let (optimum, changed) = {
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let first = s.solve_with(&mut SparseFaer::<f64>::from_options(&options(None, false)), &to_the_optimum()).unwrap().end_cost;
        change_scene(&mut s.cameras, &mut s.points, &mut s.observations);
        (first, s.solve_with(&mut SparseFaer::<f64>::from_options(&options(None, false)), &to_the_optimum()).unwrap().end_cost)
    };
    for d in f32_devices() {
        let name = label(&d);
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let mut session = LmSession::new(SparseFaer::<f32>::from_options(&options(Some(&d), true)));
        let r = session.solve(&mut s, &to_the_optimum()).unwrap();
        assert!(session.solver().plan().unwrap().gpu_assembly);
        assert!((r.end_cost as f64 - optimum).abs() <= 1e-4 * optimum, "{name}: {} against the optimum {optimum}", r.end_cost);
        left_behind_costs(&mut s, &r, &format!("{name}: the solved model"));
        change_scene(&mut s.cameras, &mut s.points, &mut s.observations);
        let r = session.solve(&mut s, &to_the_optimum()).unwrap();
        assert!((r.end_cost as f64 - changed).abs() <= 1e-4 * changed, "{name}: the warm solve's {} against the optimum {changed}", r.end_cost);
        left_behind_costs(&mut s, &r, &format!("{name}: the warm solve's model"));
    }
}

type Seen = std::sync::Arc<std::sync::Mutex<Vec<(bool, Vec<f64>)>>>;

/// A lambda driver that records what it is handed: the vectors' lengths
/// and the predicted reduction, set and as the step reports it.
#[derive(Clone)]
struct Probe<T: Float> {
    inner: arael::simple_lm::DefaultLambdaDriver<T>,
    seen: std::sync::Arc<std::sync::Mutex<Vec<(usize, usize, usize, Option<f64>, f64)>>>,
}

impl<T: Float> Probe<T> {
    fn record(&self, step: &arael::simple_lm::LambdaStep<T>) {
        let predicted = step.predicted.map(|p| p.to_f64().unwrap());
        self.seen.lock().unwrap().push((step.grad.len(), step.diagonal.len(), step.delta.len(), predicted, step.predicted_reduction().to_f64().unwrap()));
    }
}

impl<T: Float> arael::simple_lm::LambdaDriver<T> for Probe<T> {
    fn start(&mut self, config: &arael::simple_lm::LmConfig<T>, state: &arael::simple_lm::LambdaState<T>) -> T {
        self.seen.lock().unwrap().push((state.grad.len(), state.diagonal.len(), 0, None, 0.0));
        self.inner.start(config, state)
    }
    fn accepted(&mut self, step: &arael::simple_lm::LambdaStep<T>) -> Option<T> {
        self.record(step);
        self.inner.accepted(step)
    }
    fn rejected(&mut self, step: &arael::simple_lm::LambdaStep<T>) -> Option<T> {
        self.record(step);
        self.inner.rejected(step)
    }
    fn factorization_failed(&mut self, lambda: T, state: &arael::simple_lm::LambdaState<T>) -> Option<T> {
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

/// The f32 twin: on every device the observer gets the parameters and the
/// driver the vectors only when the config asks; the predicted reduction
/// always.
#[test]
fn an_f32_device_solve_hands_out_vectors_when_asked() {
    use arael::simple_lm::{LmConfig, LmIter, SparseFaer};
    let run = |gpu: Option<&GpuDevice>, asked: bool| {
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let seen: Seen = Default::default();
        let log = seen.clone();
        let probe = Probe::<f32> { inner: Default::default(), seen: Default::default() };
        let driven = probe.seen.clone();
        let cfg = LmConfig::<f32> { max_iters: 20, ..Default::default() }
            .with_driver(probe)
            .with_observer(move |it: &LmIter<f32>| {
                log.lock().unwrap().push((it.accepted, it.params.iter().map(|&p| p as f64).collect()));
                std::ops::ControlFlow::Continue(())
            })
            .with_device_observer_params(asked)
            .with_device_driver_vectors(asked);
        let mut solver = SparseFaer::<f32>::from_options(&options(gpu, true));
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        let seen = seen.lock().unwrap().clone();
        let driven = driven.lock().unwrap().clone();
        (x.len(), r, seen, driven)
    };
    let (n, _, seen, driven) = run(None, false);
    assert!(seen.iter().all(|(_, p)| p.len() == n));
    assert!(driven.iter().skip(1).all(|d| d.0 == n && d.1 == n && d.2 == n && d.3.is_none()));
    for d in f32_devices() {
        let name = label(&d);
        for asked in [false, true] {
            let (_, r, seen, driven) = run(Some(&d), asked);
            assert!(r.accepted_iterations > 2, "{name}");
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
                let x: Vec<f64> = r.x.iter().map(|&v| v as f64).collect();
                assert_close(&last.1, &x, 0.0, &format!("{name}: the observer's last parameters"));
            }
        }
    }
}

/// A lambda driver that records each accepted step's predicted reduction
/// as the step carries it (the device's, with the vectors handed out) and as
/// the host's formula gives it from the vectors.
#[derive(Clone)]
struct Check<T>(std::sync::Arc<std::sync::Mutex<Vec<(f64, f64)>>>, core::marker::PhantomData<T>);

impl<T: Float> arael::simple_lm::LambdaDriver<T> for Check<T> {
    fn start(&mut self, config: &arael::simple_lm::LmConfig<T>, _state: &arael::simple_lm::LambdaState<T>) -> T {
        config.initial_lambda
    }
    fn accepted(&mut self, step: &arael::simple_lm::LambdaStep<T>) -> Option<T> {
        let host = arael::simple_lm::LambdaStep { predicted: None, ..*step }.predicted_reduction();
        self.0.lock().unwrap().push((step.predicted.unwrap().to_f64().unwrap(), host.to_f64().unwrap()));
        Some(step.lambda * T::from(0.2).unwrap())
    }
    fn rejected(&mut self, step: &arael::simple_lm::LambdaStep<T>) -> Option<T> {
        Some(step.lambda * T::from(10.0).unwrap())
    }
    fn factorization_failed(&mut self, lambda: T, _state: &arael::simple_lm::LambdaState<T>) -> Option<T> {
        Some(lambda * T::from(10.0).unwrap())
    }
}

/// The device recomputes the predicted reduction the host computes from
/// the vectors it hands out.
#[test]
fn the_device_predicted_reduction_is_the_host_formula() {
    use arael::simple_lm::{LmConfig, SparseFaer};
    for d in devices() {
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s = Scene { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let pairs = Check::<f64>(Default::default(), Default::default());
        let cfg = LmConfig::<f64> { max_iters: 10, ..Default::default() }.with_driver(pairs.clone()).with_device_driver_vectors(true);
        arael::simple_lm::lm_solve(&x, &mut SparseFaer::<f64>::from_options(&options(Some(&d), true)), &mut s, &cfg).unwrap();
        let pairs = pairs.0.lock().unwrap();
        assert!(pairs.len() > 2);
        for &(device, host) in pairs.iter() {
            assert!((device - host).abs() <= 1e-12 * host.abs(), "{}: {device} against {host}", label(&d));
        }
    }
}

/// The f32 twin: on every device the predicted reduction the device
/// computes in its wide scalar is the host's formula over the f32 vectors
/// it hands out, to f32 rounding.
#[test]
fn an_f32_device_predicted_reduction_is_the_host_formula() {
    use arael::simple_lm::{LmConfig, SparseFaer};
    for d in f32_devices() {
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let pairs = Check::<f32>(Default::default(), Default::default());
        let cfg = LmConfig::<f32> { max_iters: 10, ..Default::default() }.with_driver(pairs.clone()).with_device_driver_vectors(true);
        arael::simple_lm::lm_solve(&x, &mut SparseFaer::<f32>::from_options(&options(Some(&d), true)), &mut s, &cfg).unwrap();
        let pairs = pairs.0.lock().unwrap();
        assert!(pairs.len() > 2);
        for &(device, host) in pairs.iter() {
            assert!((device - host).abs() <= 1e-4 * host.abs(), "{}: {device} against {host}", label(&d));
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

/// The f32 twin: the gradient test on a device linearization stops the
/// solve, as on the host's linearization, on every device.
#[test]
fn an_f32_device_gradient_test_stops_the_solve() {
    use arael::simple_lm::{LmConfig, LmStatus, SparseFaer};
    let run = |d: &GpuDevice, assembly: bool| {
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let cfg = LmConfig::<f32> { max_iters: 100, gradient_tolerance: Some(1e-3), abs_precision: 0.0, rel_precision: 0.0, ..Default::default() };
        let mut solver = SparseFaer::<f32>::from_options(&options(Some(d), assembly));
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &cfg).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, assembly);
        (r.status, r.accepted_iterations)
    };
    for d in f32_devices() {
        for assembly in [false, true] {
            let (status, accepted) = run(&d, assembly);
            assert_eq!(status, LmStatus::GradientTolerance, "{} assembly {assembly}", label(&d));
            assert!(accepted > 2, "{} assembly {assembly}", label(&d));
        }
    }
}

/// A slider pulled to 1 and, while the slider is below 0.5, a follower
/// pulled to 2: the first step takes the slider past 0.5, so no
/// curvature reaches the followers at the second linearization.
#[arael::model]
#[derive(Clone)]
pub struct Slider<T: Float> {
    pub s: Param<vect2<T>>,
    pub hb: SelfBlock<Slider<T>, T>,
}

#[arael::model]
#[derive(Clone)]
pub struct Follower<T: Float> {
    pub f: Param<vect2<T>>,
    pub hb: SelfBlock<Follower<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, {
    [sl.s.x - 1.0, sl.s.y, branch(0.5 - sl.s.x, fo.f.x - 2.0, 0.0), branch(0.5 - sl.s.x, fo.f.y, 0.0)]
}))]
#[derive(Clone)]
pub struct Link<T: Float> {
    #[arael(ref = root.sliders)]
    pub sl: Ref<Slider<T>>,
    #[arael(ref = root.followers)]
    pub fo: Ref<Follower<T>>,
    pub hb: CrossBlock<Slider<T>, Follower<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Switch {
    pub sliders: refs::Vec<Slider<f64>>,
    pub followers: refs::Vec<Follower<f64>>,
    pub links: std::vec::Vec<Link<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct SwitchF {
    pub sliders: refs::Vec<Slider<f32>>,
    pub followers: refs::Vec<Follower<f32>>,
    pub links: std::vec::Vec<Link<f32>>,
}

type SwitchParts<T> = (refs::Vec<Slider<T>>, refs::Vec<Follower<T>>, std::vec::Vec<Link<T>>);

/// Two sliders and three followers, each slider linked to each follower.
fn switch_parts<T: Float>() -> SwitchParts<T> {
    let zero = || Param::new(vect2::new(T::zero(), T::zero()));
    let mut sliders = refs::Vec::new();
    for _ in 0..2 {
        sliders.push(Slider { s: zero(), hb: SelfBlock::new() });
    }
    let mut followers = refs::Vec::new();
    for _ in 0..3 {
        followers.push(Follower { f: zero(), hb: SelfBlock::new() });
    }
    let mut links = std::vec::Vec::new();
    for k in 0..6 {
        links.push(Link { sl: sliders.ref_at(k % 2), fo: followers.ref_at(k % 3), hb: CrossBlock::new() });
    }
    (sliders, followers, links)
}

/// A bad diagonal found on a device linearization ends the solve as the
/// host's check does: the same parameter, the same partial state.
#[test]
fn a_bad_diagonal_on_the_device_ends_the_solve_as_on_the_host() {
    use arael::simple_lm::{DiagonalFault, LmConfig, SolveFailureKind, SparseFaer};
    let run = |gpu: Option<&GpuDevice>| {
        let (sliders, followers, links) = switch_parts::<f64>();
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

/// The f32 twin: on every device a bad diagonal ends the solve as the
/// host's check does, the partial state to f32 rounding.
#[test]
fn an_f32_bad_diagonal_on_the_device_ends_the_solve_as_on_the_host() {
    use arael::simple_lm::{DiagonalFault, LmConfig, SolveFailureKind, SparseFaer};
    let run = |gpu: Option<&GpuDevice>| {
        let (sliders, followers, links) = switch_parts::<f32>();
        let mut s = SwitchF { sliders, followers, links };
        assert_eq!(LmProblemInternals::<f32>::device_unsupported(&s), None);
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = SparseFaer::<f32>::from_options(&options(gpu, true));
        let e = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &LmConfig::<f32>::default()).unwrap_err();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some());
        let p = e.partial.expect("a partial state");
        (e.kind, p.iterations, p.accepted_iterations, p.x.iter().map(|&v| v as f64).collect::<Vec<f64>>())
    };
    let host = run(None);
    assert!(matches!(host.0, SolveFailureKind::DegenerateDiagonal { param: 4, fault: DiagonalFault::Zero }), "{:?}", host.0);
    assert_eq!((host.1, host.2), (1, 1));
    for d in f32_devices() {
        let name = label(&d);
        let device = run(Some(&d));
        assert_eq!(format!("{:?}", device.0), format!("{:?}", host.0), "{name}");
        assert_eq!((device.1, device.2), (host.1, host.2), "{name}");
        assert_close(&device.3, &host.3, 1e-6, &format!("{name}: the partial state"));
    }
}

/// `GpuDevice::shared` hands every caller the same device state: a later
/// call finds the memory a solve through an earlier one left, a separately
/// created device has none. On each runtime, through the f32 scene.
#[test]
fn a_shared_device_is_one_state_per_process() {
    use arael::simple_lm::{LmConfig, SparseFaer};
    let mut backends = Vec::new();
    for d in f32_devices() {
        if backends.contains(&d.backend) {
            continue;
        }
        backends.push(d.backend);
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let opts = options(Some(&GpuDevice::shared(d.backend, 0)), true);
        s.solve_with(&mut SparseFaer::<f32>::from_options(&opts), &LmConfig::<f32>::default()).unwrap();
        assert!(GpuDevice::shared(d.backend, 0).reserved_bytes() > 0, "{}", label(&d));
        assert_eq!(d.reserved_bytes(), 0, "{}", label(&d));
    }
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

/// The f32 twin: on every device, solves of the f32 scene's slightly
/// changed structures reuse the first one's memory.
#[test]
fn f32_solves_of_slightly_changed_structures_allocate_nothing() {
    use arael::simple_lm::{LmConfig, LmSession, SparseFaer};
    let sizes = [(8usize, 80usize, 3u64), (8, 84, 4), (8, 77, 5), (8, 82, 6), (8, 80, 7)];
    let cfg = LmConfig::<f32> { max_iters: 8, ..Default::default() };
    for d in f32_devices() {
        let name = label(&d);
        let mut made = Vec::new();
        for &(nc, np, seed) in &sizes {
            let (cameras, points, observations) = consistent_scene_of::<f32>(nc, np, seed);
            let mut s = SceneF { cameras, points, observations };
            let mut solver = SparseFaer::<f32>::from_options(&options(Some(&d), true));
            assert!(s.solve_with(&mut solver, &cfg).unwrap().end_cost.is_finite(), "{name}");
            made.push(d.buffers_created());
        }
        assert!(made[0] > 0 && made.iter().all(|&m| m == made[0]), "{name}: buffers after each solve {made:?}");
        let mut session = LmSession::new(SparseFaer::<f32>::from_options(&options(Some(&d), true)));
        for &(nc, np, seed) in &sizes {
            let (cameras, points, observations) = consistent_scene_of::<f32>(nc, np, seed + 10);
            let mut s = SceneF { cameras, points, observations };
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

/// The f32 twin: on every device the first linearization is assembled on
/// the device, the start cost is the CPU's to f32 rounding of the
/// residuals, and the solve lands on the optimum.
#[test]
fn an_f32_first_linearization_is_assembled_on_the_device() {
    use arael::simple_lm::SparseFaer;
    let run = |gpu: Option<&GpuDevice>, verbose: bool| {
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let cfg = arael::simple_lm::LmConfig::<f32> { verbose, ..to_the_optimum() };
        s.solve_with(&mut SparseFaer::<f32>::from_options(&options(gpu, true)), &cfg).unwrap()
    };
    let optimum = consistent_optimum();
    let cpu = run(None, false);
    for d in f32_devices() {
        let lines: std::sync::Arc<std::sync::Mutex<Vec<String>>> = Default::default();
        let sink = lines.clone();
        arael::log::set_sink(move |_, m| sink.lock().unwrap().push(m.to_string()));
        let device = run(Some(&d), true);
        arael::log::reset_sink();
        let lines = lines.lock().unwrap();
        let name = label(&d);
        assert!(lines.iter().any(|l| l.starts_with("gpu:") && l.contains("its first assembly")), "{name}: {lines:#?}");
        assert!(lines.iter().any(|l| l.starts_with(&format!("gpu schur ({name},"))), "{name}: {lines:#?}");
        let (start, cpu_start) = (device.start_cost as f64, cpu.start_cost as f64);
        assert!((start - cpu_start).abs() <= 1e-5 * cpu_start, "{name}: start {start} against {cpu_start}");
        let end = device.end_cost as f64;
        assert!((end - optimum).abs() <= 1e-4 * optimum, "{name}: {end} against the optimum {optimum}");
    }
}

/// A device: CUDA when it is compiled in, else HIP, else Metal, else the
/// CPU device.
#[cfg(feature = "gpu-cuda")]
fn device() -> std::sync::Arc<arael::__gpu::cuda::CudaDevice> {
    std::sync::Arc::new(arael::__gpu::cuda::CudaDevice::new(0).unwrap())
}
#[cfg(all(feature = "gpu-hip", not(feature = "gpu-cuda")))]
fn device() -> std::sync::Arc<arael::__gpu::hip::HipDevice> {
    std::sync::Arc::new(arael::__gpu::hip::HipDevice::new(0).unwrap())
}
#[cfg(all(gpu_metal, not(any(feature = "gpu-cuda", feature = "gpu-hip"))))]
fn device() -> std::sync::Arc<arael::__gpu::metal::MetalDevice> {
    std::sync::Arc::new(arael::__gpu::metal::MetalDevice::new(0).unwrap())
}
#[cfg(all(feature = "gpu-cpu", not(any(feature = "gpu-cuda", feature = "gpu-hip", gpu_metal))))]
fn device() -> std::sync::Arc<arael::__gpu::cpu::CpuDevice> {
    std::sync::Arc::new(arael::__gpu::cpu::CpuDevice::new(0).unwrap())
}

/// The wide scalars a device check runs in on `device()`: both, or df32
/// alone on a device without f64 (Metal).
fn check_wides() -> Vec<arael::__gpu::Wide> {
    use arael::__gpu::{Device, Wide};
    if device().arch().has_f64 { vec![Wide::F64, Wide::Df32] } else { vec![Wide::Df32] }
}

/// Whether `device()` assembles f64 models (Metal does not); without, it
/// prints that the f64 device checks of `what` are skipped (its f32 twin
/// checks in f32).
fn checks_f64(what: &str) -> bool {
    use arael::__gpu::Device;
    let has = device().arch().has_f64;
    if !has {
        eprintln!("{what}: the device has no f64, its f64 device checks skipped");
    }
    has
}

/// The device check of an f32 model against its CPU assembly, at its start
/// and at a point off it, in each wide scalar of `device()`: the cost to
/// `cost_tol`, the gradient and the Hessian to `deriv_tol` (the CPU sums its
/// f32 terms in f32, the device in its wide scalar).
fn checks_f32<P: LmProblemInternals<f32> + RootProblem<f32>>(s: &mut P, cost_tol: f64, deriv_tol: f64, what: &str) {
    assert_eq!(LmProblemInternals::<f32>::device_unsupported(s), None, "{what}");
    let mut x = Vec::new();
    s.serialize(&mut x);
    let x1 = perturbed(&x);
    for wide in check_wides() {
        let chk = arael::__gpu::device_check(s, &x, &x1, &device(), wide);
        assert!(chk.values > 0, "{what}");
        assert!((chk.device_cost - chk.cpu_cost).abs() <= cost_tol * chk.cpu_cost, "{what} {wide:?}: {chk:?}");
        assert!(chk.grad_err < deriv_tol && chk.hess_err < deriv_tol, "{what} {wide:?}: {chk:?}");
    }
}

/// The cost an f64 model's solve through `opts` on the CPU runs to
/// ([`to_the_optimum`]): what its f32 twin is measured against.
fn optimum_of<P: LmProblemInternals<f64> + RootProblem<f64>>(mut s: P, opts: &arael::simple_lm::SparseFaerOptions) -> f64 {
    let mut x = Vec::new();
    s.serialize(&mut x);
    let mut solver = arael::simple_lm::SparseFaer::<f64>::from_options(opts);
    arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &to_the_optimum()).unwrap().end_cost
}

/// An f32 model solved to its optimum ([`to_the_optimum`]) through `opts`,
/// a fresh `make()` each, on the CPU's f32 route and on every device of
/// [`f32_devices`]: the device routes, their sums wide, land on `optimum`
/// (the f64 twin's) to `tol`, the CPU's f32 route to 1e-3; relative to the
/// optimum's cost, or to the start's where the optimum's is rounding of
/// zero.
fn f32_solves_reach<P: LmProblemInternals<f32> + RootProblem<f32>>(
    make: impl Fn() -> P,
    opts: impl Fn(Option<&GpuDevice>) -> arael::simple_lm::SparseFaerOptions,
    optimum: f64,
    tol: f64,
    what: &str,
) {
    let run = |gpu: Option<&GpuDevice>| {
        let mut s = make();
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&opts(gpu));
        let r = arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &to_the_optimum()).unwrap();
        assert_eq!(solver.plan().unwrap().gpu_assembly, gpu.is_some(), "{what}");
        (r.start_cost as f64, r.end_cost as f64)
    };
    let (start, cpu) = run(None);
    let near = |c: f64, tol: f64| (c - optimum).abs() <= tol * optimum + 1e-7 * start;
    assert!(near(cpu, 1e-3), "{what}: the CPU's f32 route ends at {cpu}, the optimum is {optimum}");
    for d in f32_devices() {
        let (_, device) = run(Some(&d));
        assert!(near(device, tol), "{what} {}: {device}, the optimum is {optimum}", label(&d));
    }
}

/// An f64 model's f32 twin (`make32`) assembles on the device as on the
/// CPU ([`checks_f32`]) and solves through `opts` on every device to the
/// f64 model's (`make64`) optimum ([`f32_solves_reach`], to 1e-4).
fn f32_twin_assembles_and_solves<P64, P32>(
    make64: impl Fn() -> P64,
    make32: impl Fn() -> P32,
    opts: impl Fn(Option<&GpuDevice>) -> arael::simple_lm::SparseFaerOptions,
    what: &str,
) where
    P64: LmProblemInternals<f64> + RootProblem<f64>,
    P32: LmProblemInternals<f32> + RootProblem<f32>,
{
    checks_f32(&mut make32(), 1e-5, 1e-4, what);
    let optimum = optimum_of(make64(), &opts(None));
    f32_solves_reach(make32, opts, optimum, 1e-4, what);
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
    if !checks_f64("the f64 bundle adjustment") {
        return;
    }
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
    for wide in check_wides() {
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
    if !checks_f64("an f32 entity in an f64 root") {
        return;
    }
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

/// The f32 twin, on every device.
#[test]
fn an_f32_gpu_solver_moves_between_threads() {
    use arael::simple_lm::{LmConfig, SparseFaer};
    fn send<S: Send>(_: &S) {}
    for d in f32_devices() {
        let mut solver = SparseFaer::<f32>::from_options(&options(Some(&d), true));
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        s.solve_with(&mut solver, &LmConfig::<f32>::default()).unwrap();
        send(&solver);
        let cost = std::thread::spawn(move || s.solve_with(&mut solver, &LmConfig::<f32>::default()).unwrap().end_cost).join().unwrap();
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
    if checks_f64("the pose graph") {
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
        }
    }
    let (poses, edges, prior) = pose_graph::<f32>(40, 3);
    checks_f32(&mut Graph2DF { poses, edges, prior }, 1e-5, 1e-4, "the f32 pose graph");
}

/// Edges from a pose to itself, in [`pose_graph`] of `n` poses.
fn self_edges<T: Float>(n: usize) -> Graph2DParts<T> {
    let c = |v: f64| T::from(v).unwrap();
    let (poses, mut edges, prior) = pose_graph::<T>(n, 3);
    for k in [3usize, 11, 11] {
        edges.push(Edge2D { a: poses.ref_at(k), b: poses.ref_at(k), dx: c(0.01), dy: c(-0.02), dth: c(0.03), w: c(5.0), hb: CrossBlock::new() });
    }
    (poses, edges, prior)
}

/// Edges whose two refs name one pose: the cross block lands on the pose's
/// own tile, each symmetric pair of entries in one cell and the diagonal
/// twice, as the CPU writes an aliased cross block.
#[test]
fn an_edge_from_a_pose_to_itself_assembles_on_the_device_as_on_the_cpu() {
    use arael::__gpu::Wide;
    if checks_f64("an edge from a pose to itself") {
        for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
            let (poses, edges, prior) = self_edges::<f64>(20);
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
    let (poses, edges, prior) = self_edges::<f32>(20);
    checks_f32(&mut Graph2DF { poses, edges, prior }, 1e-5, 1e-4, "an f32 edge from a pose to itself");
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

/// The f32 twin: the pose graph's whole-system solve reaches the f64
/// graph's optimum on every device.
#[test]
fn an_f32_pose_graph_solves_through_the_device_assembly_to_the_optimum() {
    let (poses, edges, prior) = pose_graph::<f64>(40, 3);
    let optimum = optimum_of(Graph2D { poses, edges, prior }, &whole_options(None));
    let make = || {
        let (poses, edges, prior) = pose_graph::<f32>(40, 3);
        Graph2DF { poses, edges, prior }
    };
    f32_solves_reach(make, whole_options, optimum, 1e-4, "the f32 pose graph");
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

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct GraphAF {
    pub poses: refs::Vec<PoseA<f32>>,
    pub edges: std::vec::Vec<EdgeA<f32>>,
    pub prior: Option<PriorA<f32>>,
}

type GraphAParts<T> = (refs::Vec<PoseA<T>>, std::vec::Vec<EdgeA<T>>, Option<PriorA<T>>);

fn angle_graph(n: usize, seed: u64) -> GraphA {
    let (poses, edges, prior) = angle_parts::<f64>(n, seed);
    GraphA { poses, edges, prior }
}

fn angle_graph_f32(n: usize, seed: u64) -> GraphAF {
    let (poses, edges, prior) = angle_parts::<f32>(n, seed);
    GraphAF { poses, edges, prior }
}

/// [`pose_graph`] with `AngleParam` headings.
fn angle_parts<T: Float>(n: usize, seed: u64) -> GraphAParts<T> {
    let (poses, edges, prior) = pose_graph::<T>(n, seed);
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
    (pa, edges, prior)
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
    if checks_f64("the angle pose graph") {
        for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        }
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

#[test]
fn an_f32_angle_pose_graph_assembles_and_solves_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(|| angle_graph(40, 3), || angle_graph_f32(40, 3), whole_options, "the f32 angle pose graph");
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

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct GraphVF {
    pub poses: refs::Vec<PoseV<f32>>,
    pub edges: std::vec::Vec<EdgeV<f32>>,
    pub prior: Option<PriorV<f32>>,
}

type GraphVParts<T> = (refs::Vec<PoseV<T>>, std::vec::Vec<EdgeV<T>>, Option<PriorV<T>>);

fn var_graph(n: usize, seed: u64) -> GraphV {
    let (poses, edges, prior) = var_parts::<f64>(n, seed);
    GraphV { poses, edges, prior }
}

fn var_graph_f32(n: usize, seed: u64) -> GraphVF {
    let (poses, edges, prior) = var_parts::<f32>(n, seed);
    GraphVF { poses, edges, prior }
}

/// [`pose_graph`] over `PoseV`, pose 7's position and pose 11 fixed.
fn var_parts<T: Float>(n: usize, seed: u64) -> GraphVParts<T> {
    let (poses, edges, prior) = pose_graph::<T>(n, seed);
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
    (pv, edges, prior)
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
    if checks_f64("the var self block pose graph") {
        for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        }
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

#[test]
fn an_f32_var_self_block_pose_graph_assembles_and_solves_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(|| var_graph(40, 3), || var_graph_f32(40, 3), whole_options, "the f32 var self block pose graph");
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
pub struct Pose3<T: Float> {
    pub pos: Param<vect3<T>>,
    pub ea: QuaternionParam<T>,
    pub prior: vect3<T>,
    pub prior_rot_t: arael::matrix::matrix3<T>,
    pub has_prior: bool,
    pub hb: SelfBlock<Pose3<T>, T>,
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
pub struct Edge3<T: Float> {
    #[arael(ref = root.poses)]
    pub a: Ref<Pose3<T>>,
    #[arael(ref = root.poses)]
    pub b: Ref<Pose3<T>>,
    pub dt: vect3<T>,
    pub rmeas_t: arael::matrix::matrix3<T>,
    pub wt: T,
    pub wr: T,
    pub hb: CrossBlock<Pose3<T>, Pose3<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Graph3 {
    pub poses: refs::Vec<Pose3<f64>>,
    pub edges: std::vec::Vec<Edge3<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct Graph3F {
    pub poses: refs::Vec<Pose3<f32>>,
    pub edges: std::vec::Vec<Edge3<f32>>,
}

fn pose_graph3(n: usize, seed: u64) -> Graph3 {
    let (poses, edges) = pose_graph3_parts::<f64>(n, seed);
    Graph3 { poses, edges }
}

fn pose_graph3_f32(n: usize, seed: u64) -> Graph3F {
    let (poses, edges) = pose_graph3_parts::<f32>(n, seed);
    Graph3F { poses, edges }
}

/// A loop of `n` poses on a tilted circle, each turned along it,
/// odometry between neighbours and a closure every fifth pose, the
/// measurements the truth plus noise, the poses moved off it.
fn pose_graph3_parts<T: Float>(n: usize, seed: u64) -> (refs::Vec<Pose3<T>>, std::vec::Vec<Edge3<T>>) {
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
            pos: Param::new((*t + vect3d::new(0.1 * rng.uniform(), 0.1 * rng.uniform(), 0.1 * rng.uniform()) * off).cast()),
            ea: QuaternionParam::from_rotation_matrix((*r * dr).cast()),
            prior: t.cast(),
            prior_rot_t: r.transpose().cast(),
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
            dt: (ra.transpose() * (tb - ta) + vect3d::new(0.01 * rng.uniform(), 0.01 * rng.uniform(), 0.01 * rng.uniform())).cast(),
            rmeas_t: (ra.transpose() * rb * noise).transpose().cast(),
            wt: T::from(10.0).unwrap(),
            wr: T::from(20.0).unwrap(),
            hb: CrossBlock::new(),
        });
    };
    for k in 0..n {
        edge(k, (k + 1) % n, &mut rng);
        if k % 5 == 0 {
            edge(k, (k + n / 2) % n, &mut rng);
        }
    }
    (poses, edges)
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
    if checks_f64("the 3d pose graph") {
        for (wide, tol) in [(Wide::F64, 1e-11), (Wide::Df32, 1e-10)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        }
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

/// The f32 twin: the 3D pose graph assembles on the device as on the CPU,
/// solves on every device to the f64 graph's optimum, and the solve writes
/// the re-centred rotations back into the model.
#[test]
fn an_f32_3d_pose_graph_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::simple_lm::SparseFaer;
    let what = "the f32 3d pose graph";
    f32_twin_assembles_and_solves(|| pose_graph3(40, 5), || pose_graph3_f32(40, 5), whole_options, what);
    for d in f32_devices() {
        let mut s = pose_graph3_f32(40, 5);
        let r = s.solve_with(&mut SparseFaer::<f32>::from_options(&whole_options(Some(&d))), &to_the_optimum()).unwrap();
        left_behind_costs(&mut s, &r, &format!("{what} {}", label(&d)));
    }
}

/// The 2D pose graph with guards on all three forms: a heading prior on
/// the pose itself guarded by the pose's own parameter, a position prior
/// through a ref guarded by the root's data, edges guarded by their own
/// data. A guard does not change during a solve (docs/MODEL.md), so the
/// device evaluates each once and gives the instances it drops no row.
#[arael::model]
#[arael(constraint(hb, guard = f64::from(self.th.value) > 3.063, { [(poseg.th - poseg.heading) * 0.5] }))]
#[derive(Clone)]
pub struct PoseG<T: Float> {
    pub pos: Param<vect2<T>>,
    pub th: Param<T>,
    pub heading: T,
    pub hb: SelfBlock<PoseG<T>, T>,
}

#[arael::model]
#[arael(constraint(p.hb, guard = root.use_prior, {
    [p.pos.x - priorg.pos.x, p.pos.y - priorg.pos.y, p.th - priorg.th]
}))]
#[derive(Clone)]
pub struct PriorG<T: Float> {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseG<T>>,
    pub pos: vect2<T>,
    pub th: T,
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
pub struct EdgeG<T: Float> {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseG<T>>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseG<T>>,
    pub dx: T,
    pub dy: T,
    pub dth: T,
    pub w: T,
    pub on: bool,
    pub hb: CrossBlock<PoseG<T>, PoseG<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct GraphG {
    pub poses: refs::Vec<PoseG<f64>>,
    pub edges: std::vec::Vec<EdgeG<f64>>,
    pub priors: std::vec::Vec<PriorG<f64>>,
    pub use_prior: bool,
    /// Read by every edge's body: one value in the root's record.
    pub edge_scale: f64,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct GraphGF {
    pub poses: refs::Vec<PoseG<f32>>,
    pub edges: std::vec::Vec<EdgeG<f32>>,
    pub priors: std::vec::Vec<PriorG<f32>>,
    pub use_prior: bool,
    pub edge_scale: f32,
}

type GraphGParts<T> = (refs::Vec<PoseG<T>>, std::vec::Vec<EdgeG<T>>, std::vec::Vec<PriorG<T>>);

fn guarded_graph(n: usize, seed: u64, use_prior: bool) -> GraphG {
    let (poses, edges, priors) = guarded_parts::<f64>(n, seed);
    GraphG { poses, edges, priors, use_prior, edge_scale: 0.8 }
}

fn guarded_graph_f32(n: usize, seed: u64, use_prior: bool) -> GraphGF {
    let (poses, edges, priors) = guarded_parts::<f32>(n, seed);
    GraphGF { poses, edges, priors, use_prior, edge_scale: 0.8 }
}

/// [`pose_graph`] with guards: the closures every tenth edge off, the
/// heading prior on the poses whose heading starts above 3.063 (half way
/// between two of the loop's headings, so no pose crosses it in a solve).
fn guarded_parts<T: Float>(n: usize, seed: u64) -> GraphGParts<T> {
    let (poses, edges, prior) = pose_graph::<T>(n, seed);
    let mut pg = refs::Vec::new();
    for p in poses.iter() {
        pg.push(PoseG { pos: p.pos.clone(), th: p.th.clone(), heading: p.th.value + T::from(0.01).unwrap(), hb: SelfBlock::new() });
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
    (pg, edges, priors)
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
        if !checks_f64("guards on every form") {
            break;
        }
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

/// The f32 twin: the guards keep on every device what they keep on the
/// CPU, with the root's prior on and off.
#[test]
fn f32_guards_on_every_form_keep_on_the_device_what_they_keep_on_the_cpu() {
    let s0 = guarded_graph_f32(40, 3, true);
    let headed = s0.poses.iter().filter(|p| p.th.value > 3.063).count();
    assert_eq!(headed, guarded_graph(40, 3, true).poses.iter().filter(|p| p.th.value > 3.063).count());
    checks_f32(&mut guarded_graph_f32(40, 3, false), 1e-5, 1e-4, "f32 guards, the prior off");
    f32_twin_assembles_and_solves(|| guarded_graph(40, 3, true), || guarded_graph_f32(40, 3, true), whole_options, "f32 guards");
}

/// The 2D pose graph with a robust loss on every form: a Cauchy heading
/// prior on the pose itself, a soft-L1 position prior through a ref at the
/// root's scale, Huber odometry and Tukey loop closures at their own
/// scales, and Geman-McClure observations of landmarks, nested in them
/// (`parent = lm`).
#[arael::model]
#[arael(constraint(hb, loss = |s| loss_cauchy(s, 0.04), { [(posel.th - posel.heading) * 2.0] }))]
#[derive(Clone)]
pub struct PoseL<T: Float> {
    pub pos: Param<vect2<T>>,
    pub th: Param<T>,
    pub heading: T,
    pub hb: SelfBlock<PoseL<T>, T>,
}

#[arael::model]
#[arael(constraint(p.hb, loss = |s| loss_soft_l1(s, root.prior_k2), {
    [p.pos.x - priorl.pos.x, p.pos.y - priorl.pos.y, p.th - priorl.th]
}))]
#[derive(Clone)]
pub struct PriorL<T: Float> {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseL<T>>,
    pub pos: vect2<T>,
    pub th: T,
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
pub struct OdoL<T: Float> {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseL<T>>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseL<T>>,
    pub dx: T,
    pub dy: T,
    pub dth: T,
    pub k2: T,
    pub hb: CrossBlock<PoseL<T>, PoseL<T>, T>,
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
pub struct LoopL<T: Float> {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseL<T>>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseL<T>>,
    pub dx: T,
    pub dy: T,
    pub dth: T,
    pub k2: T,
    pub hb: CrossBlock<PoseL<T>, PoseL<T>, T>,
}

#[arael::model]
#[derive(Clone)]
pub struct LmL<T: Float> {
    pub pos: Param<vect2<T>>,
    pub sights: std::vec::Vec<SightL<T>>,
    pub hb: SelfBlock<LmL<T>, T>,
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
pub struct SightL<T: Float> {
    #[arael(ref = root.poses)]
    pub pose: Ref<PoseL<T>>,
    pub x: T,
    pub y: T,
    pub hb: CrossBlock<LmL<T>, PoseL<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct GraphL {
    pub poses: refs::Vec<PoseL<f64>>,
    pub landmarks: refs::Vec<LmL<f64>>,
    pub odometry: std::vec::Vec<OdoL<f64>>,
    pub loops: std::vec::Vec<LoopL<f64>>,
    pub priors: std::vec::Vec<PriorL<f64>>,
    pub prior_k2: f64,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct GraphLF {
    pub poses: refs::Vec<PoseL<f32>>,
    pub landmarks: refs::Vec<LmL<f32>>,
    pub odometry: std::vec::Vec<OdoL<f32>>,
    pub loops: std::vec::Vec<LoopL<f32>>,
    pub priors: std::vec::Vec<PriorL<f32>>,
    pub prior_k2: f32,
}

type GraphLParts<T> = (refs::Vec<PoseL<T>>, refs::Vec<LmL<T>>, std::vec::Vec<OdoL<T>>, std::vec::Vec<LoopL<T>>, std::vec::Vec<PriorL<T>>);

fn lossy_graph(n: usize, m: usize, seed: u64) -> GraphL {
    let (poses, landmarks, odometry, loops, priors) = lossy_parts::<f64>(n, m, seed);
    GraphL { poses, landmarks, odometry, loops, priors, prior_k2: 0.01 }
}

fn lossy_graph_f32(n: usize, m: usize, seed: u64) -> GraphLF {
    let (poses, landmarks, odometry, loops, priors) = lossy_parts::<f32>(n, m, seed);
    GraphLF { poses, landmarks, odometry, loops, priors, prior_k2: 0.01 }
}

/// [`pose_graph`] under losses: every third loop closure and the odometry
/// out of pose 7 metres off, far past their scales, as is every fourth
/// landmark's first sighting; the landmarks on a ring outside the poses',
/// each seen by the poses within a twelfth of a turn of it.
fn lossy_parts<T: Float>(n: usize, m: usize, seed: u64) -> GraphLParts<T> {
    let c = |v: f64| T::from(v).unwrap();
    let (poses, edges, prior) = pose_graph::<T>(n, seed);
    let mut rng = Lcg(seed + 1);
    let mut pl = refs::Vec::new();
    for p in poses.iter() {
        pl.push(PoseL { pos: p.pos.clone(), th: p.th.clone(), heading: p.th.value + c(0.01), hb: SelfBlock::new() });
    }
    let (mut odometry, mut loops) = (std::vec::Vec::new(), std::vec::Vec::new());
    for e in edges.iter() {
        let (a, b) = (e.a.index() as usize, e.b.index() as usize);
        if b == (a + 1) % n {
            let off = c(if a == 7 { 1.0 } else { 0.0 });
            odometry.push(OdoL { a: pl.ref_at(a), b: pl.ref_at(b), dx: e.dx + off, dy: e.dy, dth: e.dth, k2: c(0.25), hb: CrossBlock::new() });
        } else {
            let off = c(if loops.len() % 3 == 1 { 1.0 } else { 0.0 });
            loops.push(LoopL { a: pl.ref_at(a), b: pl.ref_at(b), dx: e.dx + off, dy: e.dy - off, dth: e.dth, k2: c(25.0), hb: CrossBlock::new() });
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
                x: c(th.cos() * dx + th.sin() * dy + 0.01 * rng.uniform() + off),
                y: c(th.cos() * dy - th.sin() * dx + 0.01 * rng.uniform()),
                hb: CrossBlock::new(),
            });
        }
        landmarks.push(LmL { pos: Param::new(vect2::new(c(l.0 + 0.1 * rng.uniform()), c(l.1 + 0.1 * rng.uniform()))), sights, hb: SelfBlock::new() });
    }
    let priors = prior.iter().map(|q| PriorL { p: pl.ref_at(q.p.index() as usize), pos: q.pos, th: q.th }).collect();
    (pl, landmarks, odometry, loops, priors)
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
    if checks_f64("losses on every form") {
        for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        }
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

#[test]
fn f32_losses_on_every_form_assemble_and_solve_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(|| lossy_graph(40, 12, 3), || lossy_graph_f32(40, 12, 3), whole_options, "f32 losses on every form");
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
    if checks_f64("the robust bundle adjustment") {
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
        }
    }
    let (cameras, points, observations) = robust_scene::<f32>();
    checks_f32(&mut SceneRF { cameras, points, observations }, 1e-5, 1e-4, "the f32 robust bundle adjustment");
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

/// The f32 twin: the robust scene solves on every device to the f64
/// scene's optimum.
#[test]
fn an_f32_robust_bundle_adjustment_solves_on_the_device_to_the_optimum() {
    let (cameras, points, observations) = robust_scene::<f64>();
    let optimum = optimum_of(SceneR { cameras, points, observations }, &options(None, true));
    let make = || {
        let (cameras, points, observations) = robust_scene::<f32>();
        SceneRF { cameras, points, observations }
    };
    f32_solves_reach(make, |gpu| options(gpu, true), optimum, 1e-4, "the f32 robust bundle adjustment");
}

/// A point held near where it was seen.
#[arael::model]
#[arael(constraint(hb, { [(ptm.pos.x - ptm.prior.x) * 0.1, (ptm.pos.y - ptm.prior.y) * 0.1] }))]
#[derive(Clone)]
pub struct PtM<T: Float> {
    pub pos: Param<vect2<T>>,
    pub prior: vect2<T>,
    pub hb: SelfBlock<PtM<T>, T>,
}

/// A line `n . p = d`, its normal held near unit length.
#[arael::model]
#[arael(constraint(hb, { [(lnm.n.x * lnm.n.x + lnm.n.y * lnm.n.y - 1.0) * 10.0] }))]
#[derive(Clone)]
pub struct LnM<T: Float> {
    pub n: Param<vect2<T>>,
    pub d: Param<T>,
    pub hb: SelfBlock<LnM<T>, T>,
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
pub struct Mid<T: Float> {
    #[arael(ref = root.lines)]
    pub ln: Ref<LnM<T>>,
    #[arael(ref = root.points)]
    pub a: Ref<PtM<T>>,
    #[arael(ref = root.points)]
    pub b: Ref<PtM<T>>,
    pub len: T,
    pub w: T,
    pub on: bool,
    #[arael(cross = (ln, a))]
    pub hb_la: CrossBlock<LnM<T>, PtM<T>, T>,
    #[arael(cross = (ln, b))]
    pub hb_lb: CrossBlock<LnM<T>, PtM<T>, T>,
    #[arael(cross = (a, b))]
    pub hb_ab: CrossBlock<PtM<T>, PtM<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct MidWorld {
    pub lines: refs::Vec<LnM<f64>>,
    pub points: refs::Vec<PtM<f64>>,
    pub mids: std::vec::Vec<Mid<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct MidWorldF {
    pub lines: refs::Vec<LnM<f32>>,
    pub points: refs::Vec<PtM<f32>>,
    pub mids: std::vec::Vec<Mid<f32>>,
}

fn mid_world(seed: u64) -> MidWorld {
    let (lines, points, mids) = mid_parts::<f64>(seed);
    MidWorld { lines, points, mids }
}

fn mid_world_f32(seed: u64) -> MidWorldF {
    let (lines, points, mids) = mid_parts::<f32>(seed);
    MidWorldF { lines, points, mids }
}

/// Five lines with six segments each; every third segment the one before
/// it with its points swapped (a pair of points on one cross tile both
/// ways round), every seventh guarded off, one point fixed; the points and
/// lines moved off the truth.
fn mid_parts<T: Float>(seed: u64) -> (refs::Vec<LnM<T>>, refs::Vec<PtM<T>>, std::vec::Vec<Mid<T>>) {
    let f = |v: f64| T::from(v).unwrap();
    let mut rng = Lcg(seed);
    let mut lines = refs::Vec::new();
    let mut points = refs::Vec::new();
    let mut mids = std::vec::Vec::new();
    for l in 0..5 {
        let th = 0.6 * l as f64 + 0.2;
        let (n, t) = ((th.cos(), th.sin()), (-th.sin(), th.cos()));
        let d = 1.0 + l as f64;
        let line = lines.push(LnM {
            n: Param::new(vect2::new(f(n.0 + 0.05 * rng.uniform()), f(n.1 + 0.05 * rng.uniform()))),
            d: Param::new(f(d + 0.05 * rng.uniform())),
            hb: SelfBlock::new(),
        });
        for k in 0..6 {
            let (a, b, len) = if k % 3 == 2 {
                let prev: &Mid<T> = mids.last().unwrap();
                (prev.b, prev.a, prev.len)
            } else {
                let s = k as f64 - 2.5;
                let len = 1.0 + 0.1 * k as f64;
                let c = (n.0 * d + t.0 * s, n.1 * d + t.1 * s);
                let mut pt = |sign: f64, rng: &mut Lcg| {
                    let p = vect2::new(f(c.0 + sign * t.0 * len / 2.0 + 0.05 * rng.uniform()), f(c.1 + sign * t.1 * len / 2.0 + 0.05 * rng.uniform()));
                    let pos = if points.len() == 3 { Param::fixed(p) } else { Param::new(p) };
                    points.push(PtM { pos, prior: p, hb: SelfBlock::new() })
                };
                (pt(-1.0, &mut rng), pt(1.0, &mut rng), f(len))
            };
            let on = mids.len() % 7 != 4;
            mids.push(Mid { ln: line, a, b, len, w: f(3.0), on, hb_la: CrossBlock::new(), hb_lb: CrossBlock::new(), hb_ab: CrossBlock::new() });
        }
    }
    (lines, points, mids)
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
    if checks_f64("the multi cross constraint") {
        for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        }
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

#[test]
fn an_f32_multi_cross_constraint_assembles_and_solves_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(|| mid_world(5), || mid_world_f32(5), whole_options, "the f32 multi cross constraint");
}

/// A node of the shared cross tile tests, held near where it was seen.
#[arael::model]
#[arael(constraint(hb, { [(nodep.x - nodep.px) * 0.3, (nodep.y - nodep.py) * 0.3] }))]
#[derive(Clone)]
pub struct NodeP<T: Float> {
    pub x: Param<T>,
    pub y: Param<T>,
    pub px: T,
    pub py: T,
    pub hb: SelfBlock<NodeP<T>, T>,
}

/// A link with refs of its own, written into its parent's cross block: one
/// tile for every link the parent holds.
#[arael::model]
#[arael(constraint(parent.hb, guard = self.on, {
    [(b.x - a.x - shlink.dx) * shlink.w, (b.y - a.y - shlink.dy) * shlink.w]
}))]
#[derive(Clone)]
pub struct ShLink<T: Float> {
    #[arael(ref = root.nodes)]
    pub a: Ref<NodeP<T>>,
    #[arael(ref = root.nodes)]
    pub b: Ref<NodeP<T>>,
    pub dx: T,
    pub dy: T,
    pub w: T,
    pub on: bool,
}

#[arael::model]
#[derive(Clone)]
pub struct ShPair<T: Float> {
    pub links: std::vec::Vec<ShLink<T>>,
    pub hb: CrossBlock<NodeP<T>, NodeP<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct ShNet {
    pub nodes: refs::Vec<NodeP<f64>>,
    pub pairs: std::vec::Vec<ShPair<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct ShNetF {
    pub nodes: refs::Vec<NodeP<f32>>,
    pub pairs: std::vec::Vec<ShPair<f32>>,
}

/// A link of no refs: its parent's refs are its entities, its parent's
/// cross block its tile, its parent's gain a factor.
#[arael::model]
#[arael(constraint(parent.hb, guard = self.on, {
    [(parent.b.x - parent.a.x - prlink.dx) * prlink.w * parent.gain,
     (parent.b.y - parent.a.y - prlink.dy) * prlink.w * parent.gain]
}))]
#[derive(Clone)]
pub struct PrLink<T: Float> {
    pub dx: T,
    pub dy: T,
    pub w: T,
    pub on: bool,
}

#[arael::model]
#[derive(Clone)]
pub struct PrPair<T: Float> {
    #[arael(ref = root.nodes)]
    pub a: Ref<NodeP<T>>,
    #[arael(ref = root.nodes)]
    pub b: Ref<NodeP<T>>,
    pub gain: T,
    pub links: std::vec::Vec<PrLink<T>>,
    pub hb: CrossBlock<NodeP<T>, NodeP<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct PrNet {
    pub nodes: refs::Vec<NodeP<f64>>,
    pub pairs: std::vec::Vec<PrPair<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct PrNetF {
    pub nodes: refs::Vec<NodeP<f32>>,
    pub pairs: std::vec::Vec<PrPair<f32>>,
}

/// A link with refs and a cross block of its own, held below the root.
#[arael::model]
#[arael(constraint(hb, guard = self.on, {
    [(b.x - a.x - ownlink.dx) * ownlink.w, (b.y - a.y - ownlink.dy) * ownlink.w]
}))]
#[derive(Clone)]
pub struct OwnLink<T: Float> {
    #[arael(ref = root.nodes)]
    pub a: Ref<NodeP<T>>,
    #[arael(ref = root.nodes)]
    pub b: Ref<NodeP<T>>,
    pub dx: T,
    pub dy: T,
    pub w: T,
    pub on: bool,
    pub hb: CrossBlock<NodeP<T>, NodeP<T>, T>,
}

#[arael::model]
#[derive(Clone)]
pub struct OwnPair<T: Float> {
    pub links: std::vec::Vec<OwnLink<T>>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct OwnNet {
    pub nodes: refs::Vec<NodeP<f64>>,
    pub pairs: std::vec::Vec<OwnPair<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct OwnNetF {
    pub nodes: refs::Vec<NodeP<f32>>,
    pub pairs: std::vec::Vec<OwnPair<f32>>,
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

fn net_nodes<T: Float>(nodes: &[(f64, f64)]) -> refs::Vec<NodeP<T>> {
    let mut v = refs::Vec::new();
    for &(x, y) in nodes {
        let (x, y) = (T::from(x).unwrap(), T::from(y).unwrap());
        v.push(NodeP { x: Param::new(x), y: Param::new(y), px: x, py: y, hb: SelfBlock::new() });
    }
    v
}

fn shared_parts<T: Float>(seed: u64) -> (refs::Vec<NodeP<T>>, std::vec::Vec<ShPair<T>>) {
    let f = |v: f64| T::from(v).unwrap();
    let (nodes, pairs) = net_data(seed);
    let nodes = net_nodes::<T>(&nodes);
    let pairs = pairs
        .into_iter()
        .map(|((i, j), _, links)| ShPair {
            links: links.into_iter().map(|(dx, dy, w, on)| ShLink { a: nodes.ref_at(i), b: nodes.ref_at(j), dx: f(dx), dy: f(dy), w: f(w), on }).collect(),
            hb: CrossBlock::new(),
        })
        .collect();
    (nodes, pairs)
}

fn shared_net(seed: u64) -> ShNet {
    let (nodes, pairs) = shared_parts(seed);
    ShNet { nodes, pairs }
}

fn shared_net_f32(seed: u64) -> ShNetF {
    let (nodes, pairs) = shared_parts(seed);
    ShNetF { nodes, pairs }
}

fn parent_ref_parts<T: Float>(seed: u64) -> (refs::Vec<NodeP<T>>, std::vec::Vec<PrPair<T>>) {
    let f = |v: f64| T::from(v).unwrap();
    let (nodes, pairs) = net_data(seed);
    let nodes = net_nodes::<T>(&nodes);
    let pairs = pairs
        .into_iter()
        .map(|((i, j), gain, links)| PrPair {
            a: nodes.ref_at(i),
            b: nodes.ref_at(j),
            gain: f(gain),
            links: links.into_iter().map(|(dx, dy, w, on)| PrLink { dx: f(dx), dy: f(dy), w: f(w), on }).collect(),
            hb: CrossBlock::new(),
        })
        .collect();
    (nodes, pairs)
}

fn parent_ref_net(seed: u64) -> PrNet {
    let (nodes, pairs) = parent_ref_parts(seed);
    PrNet { nodes, pairs }
}

fn parent_ref_net_f32(seed: u64) -> PrNetF {
    let (nodes, pairs) = parent_ref_parts(seed);
    PrNetF { nodes, pairs }
}

fn own_block_parts<T: Float>(seed: u64) -> (refs::Vec<NodeP<T>>, std::vec::Vec<OwnPair<T>>) {
    let f = |v: f64| T::from(v).unwrap();
    let (nodes, pairs) = net_data(seed);
    let nodes = net_nodes::<T>(&nodes);
    let pairs = pairs
        .into_iter()
        .map(|((i, j), _, links)| OwnPair {
            links: links
                .into_iter()
                .map(|(dx, dy, w, on)| OwnLink { a: nodes.ref_at(i), b: nodes.ref_at(j), dx: f(dx), dy: f(dy), w: f(w), on, hb: CrossBlock::new() })
                .collect(),
        })
        .collect();
    (nodes, pairs)
}

fn own_block_net(seed: u64) -> OwnNet {
    let (nodes, pairs) = own_block_parts(seed);
    OwnNet { nodes, pairs }
}

fn own_block_net_f32(seed: u64) -> OwnNetF {
    let (nodes, pairs) = own_block_parts(seed);
    OwnNetF { nodes, pairs }
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
    if checks_f64(what) {
        for (wide, tol) in [(Wide::F64, 1e-12), (Wide::Df32, 1e-11)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{what} {wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{what} {wide:?}: {chk:?}");
        }
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

#[test]
fn f32_parent_owned_cross_blocks_assemble_and_solve_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(|| shared_net(9), || shared_net_f32(9), whole_options, "f32 own refs");
    f32_twin_assembles_and_solves(|| parent_ref_net(9), || parent_ref_net_f32(9), whole_options, "f32 parent refs");
}

/// The same links with cross blocks of their own, in collections below the
/// root: walked through their parents' loops.
#[test]
fn cross_constraints_below_the_root_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(|| own_block_net(9), "own blocks");
}

#[test]
fn f32_cross_constraints_below_the_root_assemble_and_solve_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(|| own_block_net(9), || own_block_net_f32(9), whole_options, "f32 own blocks");
}

/// The mixed form's camera (a focal length) and point, each held near
/// where it was seen.
#[arael::model]
#[arael(constraint(hb, { [(camm.f - camm.pf) * 0.5] }))]
#[derive(Clone)]
pub struct CamM<T: Float> {
    pub f: Param<T>,
    pub pf: T,
    pub hb: SelfBlock<CamM<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, { [(pointm.p.x - pointm.px) * 0.2, (pointm.p.y - pointm.py) * 0.2] }))]
#[derive(Clone)]
pub struct PointM<T: Float> {
    pub p: Param<vect2<T>>,
    pub px: T,
    pub py: T,
    pub hb: SelfBlock<PointM<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, { [(posem.t.x - posem.px) * 0.3, (posem.t.y - posem.py) * 0.3] }))]
#[derive(Clone)]
pub struct PoseM<T: Float> {
    pub t: Param<vect2<T>>,
    pub px: T,
    pub py: T,
    pub hb: SelfBlock<PoseM<T>, T>,
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
pub struct ImgObs<T: Float> {
    #[arael(ref = root.points)]
    pub point: Ref<PointM<T>>,
    pub m: vect2<T>,
    pub on: bool,
    pub hb_point_pose: CrossBlock<PointM<T>, PoseM<T>, T>,
    pub hb_point_cam: CrossBlock<PointM<T>, CamM<T>, T>,
}

#[arael::model]
#[derive(Clone)]
pub struct ImageM<T: Float> {
    #[arael(ref = root.poses)]
    pub pose: Ref<PoseM<T>>,
    #[arael(ref = root.cams)]
    pub cam: Ref<CamM<T>>,
    pub obs: std::vec::Vec<ImgObs<T>>,
    pub hb_pose_cam: CrossBlock<PoseM<T>, CamM<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct ImgWorld {
    pub poses: refs::Vec<PoseM<f64>>,
    pub cams: refs::Vec<CamM<f64>>,
    pub points: refs::Vec<PointM<f64>>,
    pub images: std::vec::Vec<ImageM<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct ImgWorldF {
    pub poses: refs::Vec<PoseM<f32>>,
    pub cams: refs::Vec<CamM<f32>>,
    pub points: refs::Vec<PointM<f32>>,
    pub images: std::vec::Vec<ImageM<f32>>,
}

/// A pose holding its images: the observation reaches it two levels up.
#[arael::model]
#[arael(constraint(hb, { [(poseh.t.x - poseh.px) * 0.3, (poseh.t.y - poseh.py) * 0.3] }))]
#[derive(Clone)]
pub struct PoseH<T: Float> {
    pub t: Param<vect2<T>>,
    pub px: T,
    pub py: T,
    pub w: T,
    pub images: std::vec::Vec<ImageH<T>>,
    pub hb: SelfBlock<PoseH<T>, T>,
}

#[arael::model]
#[derive(Clone)]
pub struct ImageH<T: Float> {
    #[arael(ref = root.cams)]
    pub cam: Ref<CamM<T>>,
    pub obs: std::vec::Vec<ObsH<T>>,
    pub hb_pose_cam: CrossBlock<PoseH<T>, CamM<T>, T>,
}

#[arael::model]
#[arael(constraint([hb_point_pose, hb_point_cam, parent.hb_pose_cam], parent = image, parent.parent = pose,
    guard = self.on, {
    [image.cam.f * (point.p.x - pose.t.x) * pose.w - obsh.m.x,
     image.cam.f * (point.p.y - pose.t.y) * pose.w - obsh.m.y]
}))]
#[derive(Clone)]
pub struct ObsH<T: Float> {
    #[arael(ref = root.points)]
    pub point: Ref<PointM<T>>,
    pub m: vect2<T>,
    pub on: bool,
    pub hb_point_pose: CrossBlock<PointM<T>, PoseH<T>, T>,
    pub hb_point_cam: CrossBlock<PointM<T>, CamM<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct PoseWorld {
    pub poses: refs::Vec<PoseH<f64>>,
    pub cams: refs::Vec<CamM<f64>>,
    pub points: refs::Vec<PointM<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct PoseWorldF {
    pub poses: refs::Vec<PoseH<f32>>,
    pub cams: refs::Vec<CamM<f32>>,
    pub points: refs::Vec<PointM<f32>>,
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

/// The mixed scenes' cameras and points, moved off the truth.
fn mixed_cams_points<T: Float>(tc: &[f64], tk: &[P2]) -> (refs::Vec<CamM<T>>, refs::Vec<PointM<T>>) {
    let f = |v: f64| T::from(v).unwrap();
    let mut cams = refs::Vec::new();
    for &c in tc {
        cams.push(CamM { f: Param::new(f(c * 1.1)), pf: f(c), hb: SelfBlock::new() });
    }
    let mut points = refs::Vec::new();
    for (k, &(x, y)) in tk.iter().enumerate() {
        points.push(PointM { p: Param::new(vect2::new(f(x - 0.06), f(y + 0.04 * k as f64))), px: f(x), py: f(y), hb: SelfBlock::new() });
    }
    (cams, points)
}

type ImgParts<T> = (refs::Vec<PoseM<T>>, refs::Vec<CamM<T>>, refs::Vec<PointM<T>>, std::vec::Vec<ImageM<T>>);

fn img_parts<T: Float>() -> ImgParts<T> {
    let f = |v: f64| T::from(v).unwrap();
    let (tp, tc, tk) = mixed_truth();
    let mut poses = refs::Vec::new();
    for (i, &(x, y)) in tp.iter().enumerate() {
        poses.push(PoseM { t: Param::new(vect2::new(f(x + 0.07), f(y - 0.05 * i as f64))), px: f(x), py: f(y), hb: SelfBlock::new() });
    }
    let (cams, points) = mixed_cams_points::<T>(&tc, &tk);
    let images = mixed_images()
        .into_iter()
        .map(|(i, j, obs)| ImageM {
            pose: poses.ref_at(i),
            cam: cams.ref_at(j),
            obs: obs
                .into_iter()
                .map(|(k, m, on)| ImgObs { point: points.ref_at(k), m: vect2::new(f(m.0), f(m.1)), on, hb_point_pose: CrossBlock::new(), hb_point_cam: CrossBlock::new() })
                .collect(),
            hb_pose_cam: CrossBlock::new(),
        })
        .collect();
    (poses, cams, points, images)
}

fn img_world() -> ImgWorld {
    let (poses, cams, points, images) = img_parts();
    ImgWorld { poses, cams, points, images }
}

fn img_world_f32() -> ImgWorldF {
    let (poses, cams, points, images) = img_parts();
    ImgWorldF { poses, cams, points, images }
}

fn pose_world_parts<T: Float>() -> (refs::Vec<PoseH<T>>, refs::Vec<CamM<T>>, refs::Vec<PointM<T>>) {
    let f = |v: f64| T::from(v).unwrap();
    let (tp, tc, tk) = mixed_truth();
    let (cams, points) = mixed_cams_points::<T>(&tc, &tk);
    let mut poses = refs::Vec::new();
    for (i, &(x, y)) in tp.iter().enumerate() {
        poses.push(PoseH { t: Param::new(vect2::new(f(x + 0.07), f(y - 0.05 * i as f64))), px: f(x), py: f(y), w: T::one(), images: std::vec::Vec::new(), hb: SelfBlock::new() });
    }
    for (i, j, obs) in mixed_images() {
        let image = ImageH {
            cam: cams.ref_at(j),
            obs: obs
                .into_iter()
                .map(|(k, m, on)| ObsH { point: points.ref_at(k), m: vect2::new(f(m.0), f(m.1)), on, hb_point_pose: CrossBlock::new(), hb_point_cam: CrossBlock::new() })
                .collect(),
            hb_pose_cam: CrossBlock::new(),
        };
        poses[i].images.push(image);
    }
    (poses, cams, points)
}

fn pose_world() -> PoseWorld {
    let (poses, cams, points) = pose_world_parts();
    PoseWorld { poses, cams, points }
}

fn pose_world_f32() -> PoseWorldF {
    let (poses, cams, points) = pose_world_parts();
    PoseWorldF { poses, cams, points }
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

#[test]
fn the_f32_mixed_form_assembles_and_solves_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(img_world, img_world_f32, whole_options, "f32, the image's refs");
    f32_twin_assembles_and_solves(pose_world, pose_world_f32, whole_options, "f32, the pose two levels up");
}

/// A sample of a line `y = a x + b` whose parameters are the root's: the
/// sample writes the root's self block, the root the one entity of its
/// table.
#[arael::model]
#[arael(constraint(root.hb, guard = self.ok, loss = |s| loss_huber(s, 0.01), { [sample.y - root.a * sample.x - root.b] }))]
#[derive(Clone)]
pub struct Sample<T: Float> {
    pub x: T,
    pub y: T,
    pub ok: bool,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct LineFit {
    pub a: Param<f64>,
    pub b: Param<f64>,
    pub hb: SelfBlock<LineFit>,
    pub samples: std::vec::Vec<Sample<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct LineFitF {
    pub a: Param<f32>,
    pub b: Param<f32>,
    pub hb: SelfBlock<LineFitF, f32>,
    pub samples: std::vec::Vec<Sample<f32>>,
}

/// A sample of its curve, which holds the parameters: the sample writes
/// its parent's self block.
#[arael::model]
#[arael(constraint(parent.hb, { [csample.y - (curve.m * csample.x + curve.c)] }))]
#[derive(Clone)]
pub struct CSample<T: Float> {
    pub x: T,
    pub y: T,
}

#[arael::model]
#[derive(Clone)]
pub struct Curve<T: Float> {
    pub m: Param<T>,
    pub c: Param<T>,
    pub samples: std::vec::Vec<CSample<T>>,
    pub hb: SelfBlock<Curve<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Curves {
    pub curves: refs::Vec<Curve<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct CurvesF {
    pub curves: refs::Vec<Curve<f32>>,
}

/// One self-constrained type held in two root collections: one entity
/// table over both, walked in their order.
#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct TwoHomes {
    pub near: refs::Vec<NodeP<f64>>,
    pub far: refs::Vec<NodeP<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct TwoHomesF {
    pub near: refs::Vec<NodeP<f32>>,
    pub far: refs::Vec<NodeP<f32>>,
}

fn line_samples<T: Float>() -> std::vec::Vec<Sample<T>> {
    let mut rng = Lcg(31);
    (0..40)
        .map(|i| {
            let x = 0.1 * i as f64;
            let out = if i % 9 == 4 { 1.0 } else { 0.0 };
            Sample { x: T::from(x).unwrap(), y: T::from(2.0 * x + 1.0 + 0.02 * rng.uniform() + out).unwrap(), ok: i % 11 != 7 }
        })
        .collect()
}

fn line_fit() -> LineFit {
    LineFit { a: Param::new(0.3), b: Param::new(-0.2), hb: SelfBlock::new(), samples: line_samples() }
}

fn line_fit_f32() -> LineFitF {
    LineFitF { a: Param::new(0.3), b: Param::new(-0.2), hb: SelfBlock::new(), samples: line_samples() }
}

fn curve_list<T: Float>() -> refs::Vec<Curve<T>> {
    let f = |v: f64| T::from(v).unwrap();
    let mut rng = Lcg(32);
    let mut curves = refs::Vec::new();
    for k in 0..5 {
        let (m, c) = (0.5 * k as f64 - 1.0, 0.3 * k as f64);
        let samples = (0..8).map(|i| CSample { x: f(0.2 * i as f64), y: f(m * 0.2 * i as f64 + c + 0.01 * rng.uniform()) }).collect();
        curves.push(Curve { m: Param::new(f(m + 0.2)), c: Param::new(f(c - 0.1)), samples, hb: SelfBlock::new() });
    }
    curves
}

fn curves() -> Curves {
    Curves { curves: curve_list() }
}

fn curves_f32() -> CurvesF {
    CurvesF { curves: curve_list() }
}

fn two_homes_parts<T: Float>() -> (refs::Vec<NodeP<T>>, refs::Vec<NodeP<T>>) {
    let f = |v: f64| T::from(v).unwrap();
    let mut rng = Lcg(33);
    let (mut near, mut far) = (refs::Vec::new(), refs::Vec::new());
    for k in 0..20 {
        let (px, py) = (k as f64, 0.5 * k as f64);
        let p = NodeP { x: Param::new(f(px + 0.1 * rng.uniform())), y: Param::new(f(py + 0.1 * rng.uniform())), px: f(px), py: f(py), hb: SelfBlock::new() };
        if k % 3 == 0 { far.push(p) } else { near.push(p) };
    }
    (near, far)
}

fn two_homes() -> TwoHomes {
    let (near, far) = two_homes_parts();
    TwoHomes { near, far }
}

fn two_homes_f32() -> TwoHomesF {
    let (near, far) = two_homes_parts();
    TwoHomesF { near, far }
}

/// A pose held near where it was seen.
#[arael::model]
#[arael(constraint(hb_pose, { [(poser.pos.x - poser.px) * 0.5, (poser.pos.y - poser.py) * 0.5] }))]
#[derive(Clone)]
pub struct PoseR<T: Float> {
    pub pos: Param<vect2<T>>,
    pub px: T,
    pub py: T,
    pub hb_pose: SelfBlock<PoseR<T>, T>,
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
    pub pose: Ref<PoseR<f64>>,
    pub dx: f64,
    pub dy: f64,
    pub hb_root: CrossBlock<PoseR<f64>, OffsetRoot>,
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
    pub poses: refs::Vec<PoseR<f64>>,
    pub landmarks: refs::Vec<LandmarkO>,
    pub offset: Param<f64>,
    pub isigma: f64,
    pub hb: SelfBlock<OffsetRoot>,
}

/// [`FrineO`] in an f32 root (its cross block names the root type, so a
/// concrete copy).
#[arael::model]
#[arael(constraint([pose.hb_pose, hb_root], parent = lm, {
    let d = lm.anchor - pose.pos;
    [(d.x + offsetrootf.offset - frineof.dx) * offsetrootf.isigma,
     (d.y + offsetrootf.offset - frineof.dy) * offsetrootf.isigma]
}))]
#[derive(Clone)]
pub struct FrineOF {
    #[arael(ref = root.poses)]
    pub pose: Ref<PoseR<f32>>,
    pub dx: f32,
    pub dy: f32,
    pub hb_root: CrossBlock<PoseR<f32>, OffsetRootF, f32>,
}

#[arael::model]
#[derive(Clone)]
pub struct LandmarkOF {
    pub anchor: vect2<f32>,
    pub frines: std::vec::Vec<FrineOF>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct OffsetRootF {
    pub poses: refs::Vec<PoseR<f32>>,
    pub landmarks: refs::Vec<LandmarkOF>,
    pub offset: Param<f32>,
    pub isigma: f32,
    pub hb: SelfBlock<OffsetRootF, f32>,
}

/// The offset root's poses moved off the truth, and per landmark its
/// anchor and sightings (pose, dx, dy).
type OffsetData = (Vec<(f64, f64, f64, f64)>, Vec<((f64, f64), Vec<(usize, f64, f64)>)>);

fn offset_data() -> OffsetData {
    let mut rng = Lcg(34);
    let truth: Vec<(f64, f64)> = (0..8).map(|k| (k as f64, (0.4 * k as f64).sin())).collect();
    let poses = truth.iter().map(|&(x, y)| (x + 0.1 * rng.uniform(), y + 0.1 * rng.uniform(), x, y)).collect();
    let landmarks = (0..4)
        .map(|j| {
            let anchor = (2.0 * j as f64 + 0.5, 3.0);
            let frines = (0..8)
                .filter(|k| (k + j) % 3 != 0)
                .map(|k| (k, anchor.0 - truth[k].0 + 0.25 + 0.01 * rng.uniform(), anchor.1 - truth[k].1 + 0.25 + 0.01 * rng.uniform()))
                .collect();
            (anchor, frines)
        })
        .collect();
    (poses, landmarks)
}

fn offset_poses<T: Float>(poses: &[(f64, f64, f64, f64)]) -> refs::Vec<PoseR<T>> {
    let f = |v: f64| T::from(v).unwrap();
    let mut v = refs::Vec::new();
    for &(x, y, px, py) in poses {
        v.push(PoseR { pos: Param::new(vect2::new(f(x), f(y))), px: f(px), py: f(py), hb_pose: SelfBlock::new() });
    }
    v
}

fn offset_root() -> OffsetRoot {
    let (poses, marks) = offset_data();
    let poses = offset_poses(&poses);
    let mut landmarks = refs::Vec::new();
    for (anchor, frines) in marks {
        let frines = frines.into_iter().map(|(k, dx, dy)| FrineO { pose: poses.ref_at(k), dx, dy, hb_root: CrossBlock::new() }).collect();
        landmarks.push(LandmarkO { anchor: vect2::new(anchor.0, anchor.1), frines });
    }
    OffsetRoot { poses, landmarks, offset: Param::new(0.0), isigma: 3.0, hb: SelfBlock::new() }
}

fn offset_root_f32() -> OffsetRootF {
    let (poses, marks) = offset_data();
    let poses = offset_poses(&poses);
    let mut landmarks = refs::Vec::new();
    for (anchor, frines) in marks {
        let frines = frines.into_iter().map(|(k, dx, dy)| FrineOF { pose: poses.ref_at(k), dx: dx as f32, dy: dy as f32, hb_root: CrossBlock::new() }).collect();
        landmarks.push(LandmarkOF { anchor: vect2::new(anchor.0 as f32, anchor.1 as f32), frines });
    }
    OffsetRootF { poses, landmarks, offset: Param::new(0.0), isigma: 3.0, hb: SelfBlock::new() }
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

#[test]
fn f32_root_and_parent_self_blocks_assemble_and_solve_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(line_fit, line_fit_f32, whole_options, "f32, the root's self block");
    f32_twin_assembles_and_solves(curves, curves_f32, whole_options, "f32, the parent's self block");
    f32_twin_assembles_and_solves(two_homes, two_homes_f32, whole_options, "f32, a type in two collections");
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

#[test]
fn f32_the_root_as_a_side_assembles_and_solves_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(offset_root, offset_root_f32, whole_options, "f32, the root as a side");
}

/// A point held near where it was seen.
#[arael::model]
#[arael(constraint(hb, { [(ownpt.x - ownpt.px) * 2.0] }))]
#[derive(Clone)]
pub struct OwnPt<T: Float> {
    pub x: Param<T>,
    pub px: T,
    pub hb: SelfBlock<OwnPt<T>, T>,
}

/// A sample of the line the root's scale and offset make, written into
/// the root's self block.
#[arael::model]
#[arael(constraint(root.hb, { [(ownsample.y - root.scale * ownsample.x - root.offset) * 2.0] }))]
#[derive(Clone)]
pub struct OwnSample<T: Float> {
    pub x: T,
    pub y: T,
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
    pub pts: refs::Vec<OwnPt<f64>>,
    pub samples: std::vec::Vec<OwnSample<f64>>,
    pub offset: Param<f64>,
    pub scale: Param<f64>,
    pub prior: f64,
    pub isigma: f64,
    pub use_scale: bool,
    pub hb: SelfBlock<OwnRoot>,
}

/// [`OwnRoot`] in f32 (its constraints name the root type, so a concrete
/// copy).
#[arael::model]
#[arael(root, f32)]
#[arael(constraint(hb, { [(ownrootf.offset - ownrootf.prior) * ownrootf.isigma] }))]
#[arael(constraint(hb, guard = self.use_scale, {
    [(ownrootf.scale * ownrootf.scale - 2.0) * 0.5, (ownrootf.scale - 1.4) * sin(ownrootf.offset + 1.0)]
}))]
#[derive(Clone)]
pub struct OwnRootF {
    pub pts: refs::Vec<OwnPt<f32>>,
    pub samples: std::vec::Vec<OwnSample<f32>>,
    pub offset: Param<f32>,
    pub scale: Param<f32>,
    pub prior: f32,
    pub isigma: f32,
    pub use_scale: bool,
    pub hb: SelfBlock<OwnRootF, f32>,
}

fn own_parts<T: Float>() -> (refs::Vec<OwnPt<T>>, std::vec::Vec<OwnSample<T>>) {
    let f = |v: f64| T::from(v).unwrap();
    let mut rng = Lcg(35);
    let mut pts = refs::Vec::new();
    for k in 0..5 {
        pts.push(OwnPt { x: Param::new(f(k as f64 + 0.1 * rng.uniform())), px: f(k as f64), hb: SelfBlock::new() });
    }
    let samples = (0..9)
        .map(|k| {
            let x = 0.5 * k as f64 - 2.0;
            OwnSample { x: f(x), y: f(1.4 * x + 0.3 + 0.01 * rng.uniform()) }
        })
        .collect();
    (pts, samples)
}

fn own_root(use_scale: bool) -> OwnRoot {
    let (pts, samples) = own_parts();
    OwnRoot { pts, samples, offset: Param::new(0.0), scale: Param::new(1.0), prior: 0.25, isigma: 3.0, use_scale, hb: SelfBlock::new() }
}

fn own_root_f32(use_scale: bool) -> OwnRootF {
    let (pts, samples) = own_parts();
    OwnRootF { pts, samples, offset: Param::new(0.0), scale: Param::new(1.0), prior: 0.25, isigma: 3.0, use_scale, hb: SelfBlock::new() }
}

/// The root's own constraints, the root its own instance (no containment
/// path), with its guard on and off: the device assembles as the CPU does
/// and solves to its optimum.
#[test]
fn constraints_on_the_root_itself_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(|| own_root(true), "the root's own constraints");
    assembles_and_solves_as_on_the_cpu(|| own_root(false), "the root's own constraints, the guarded one off");
}

#[test]
fn f32_constraints_on_the_root_itself_assemble_and_solve_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(|| own_root(true), || own_root_f32(true), whole_options, "f32, the root's own constraints");
    f32_twin_assembles_and_solves(|| own_root(false), || own_root_f32(false), whole_options, "f32, the root's own constraints, the guarded one off");
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
pub struct DriftP<T: Float> {
    pub s: Param<T>,
    pub p: Param<vect2<T>>,
    pub ea: SimpleEulerAngleParam<T>,
    pub ms: T,
    pub mp: vect2<T>,
    pub me: T,
    pub hb: SelfBlock<DriftP<T>, T>,
}

/// A root whose own constraint holds its offset near its stored value
/// while pulling it to a target.
#[arael::model]
#[arael(root)]
#[arael(constraint(hb, { [(driftroot.offset - driftroot.offset_value) * 0.2, (driftroot.offset - driftroot.target) * 1.5] }))]
#[derive(Clone)]
pub struct DriftRoot {
    pub poses: refs::Vec<DriftP<f64>>,
    pub offset: Param<f64>,
    pub target: f64,
    pub hb: SelfBlock<DriftRoot>,
}

/// [`DriftRoot`] in f32 (its constraint names the root type, so a concrete
/// copy).
#[arael::model]
#[arael(root, f32)]
#[arael(constraint(hb, { [(driftrootf.offset - driftrootf.offset_value) * 0.2, (driftrootf.offset - driftrootf.target) * 1.5] }))]
#[derive(Clone)]
pub struct DriftRootF {
    pub poses: refs::Vec<DriftP<f32>>,
    pub offset: Param<f32>,
    pub target: f32,
    pub hb: SelfBlock<DriftRootF, f32>,
}

fn drift_poses<T: Float>() -> refs::Vec<DriftP<T>> {
    let f = |v: f64| T::from(v).unwrap();
    let mut rng = Lcg(36);
    let mut poses = refs::Vec::new();
    for k in 0..6 {
        let t = 0.3 * k as f64;
        poses.push(DriftP {
            s: Param::new(f(t)),
            p: Param::new(vect2::new(f(t), f(1.0 - t))),
            ea: SimpleEulerAngleParam::new(vect3::new(f(0.1 * t), f(-0.2 * t), f(0.05))),
            ms: f(t + 0.4 + 0.1 * rng.uniform()),
            mp: vect2::new(f(t - 0.3), f(1.2 - t)),
            me: f(0.2 + 0.1 * rng.uniform()),
            hb: SelfBlock::new(),
        });
    }
    poses
}

fn drift_root() -> DriftRoot {
    DriftRoot { poses: drift_poses(), offset: Param::new(0.0), target: 0.8, hb: SelfBlock::new() }
}

fn drift_root_f32() -> DriftRootF {
    DriftRootF { poses: drift_poses(), offset: Param::new(0.0), target: 0.8, hb: SelfBlock::new() }
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

#[test]
fn f32_stored_parameter_values_assemble_and_solve_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(drift_root, drift_root_f32, whole_options, "f32 stored values");
}

/// The f32 twin: on every device each solve through one session lands
/// where the CPU's f32 solve of the same model does, the model too (the
/// stored values it reads are the ones the last solve wrote), and moves the
/// model.
#[test]
fn an_f32_warm_solve_reads_the_stored_values_the_last_one_wrote() {
    use arael::simple_lm::{LmSession, SparseFaer};
    let state = |s: &DriftRootF| -> Vec<f64> {
        let mut v = vec![s.offset.value as f64];
        for p in s.poses.iter() {
            v.extend([p.s.value, p.p.value.x, p.p.value.y, p.ea.value.x, p.ea.value.y, p.ea.value.z].map(|x| x as f64));
        }
        v
    };
    for d in f32_devices() {
        let name = label(&d);
        let mut s = drift_root_f32();
        let mut session = LmSession::new(SparseFaer::<f32>::from_options(&whole_options(Some(&d))));
        for solve in ["the first solve", "the warm solve"] {
            let mut cpu_model = s.clone();
            let cpu = cpu_model.solve_with(&mut SparseFaer::<f32>::from_options(&whole_options(None)), &to_the_optimum()).unwrap();
            let before = state(&s);
            let r = session.solve(&mut s, &to_the_optimum()).unwrap();
            assert!(session.solver().plan().unwrap().gpu_assembly, "{name}: {solve}");
            let (end, cpu_end) = (r.end_cost as f64, cpu.end_cost as f64);
            assert!((end - cpu_end).abs() <= 1e-5 * cpu_end + 1e-7 * r.start_cost as f64, "{name}: {solve}: {end} against the CPU's {cpu_end}");
            assert_close(&state(&s), &state(&cpu_model), 1e-4, &format!("{name}: {solve}'s model"));
            assert!(state(&s).iter().zip(&before).any(|(a, b)| (a - b).abs() > 1e-3), "{name}: {solve} moves the model");
        }
    }
}

/// A one-coordinate point held near where it was seen, and again under the
/// loss its `kind` picks of `loss_select`'s six.
#[arael::model]
#[arael(constraint(hb, { [(ptk.x - ptk.px) * 0.5] }))]
#[arael(constraint(hb, loss = |s| loss_select(ptk.kind, s, ptk.k2), { [(ptk.x - ptk.px) * 2.0] }))]
#[derive(Clone)]
pub struct PtK<T: Float> {
    pub x: Param<T>,
    pub px: T,
    pub kind: u32,
    pub k2: T,
    pub hb: SelfBlock<PtK<T>, T>,
}

/// A camera of three parameters, each held near where it was seen; its
/// `model` picks which of them an observation reads.
#[arael::model]
#[arael(constraint(hb, { [(camk.f - camk.pf) * 0.3, (camk.k - camk.pk) * 0.3, (camk.c - camk.pc) * 0.3] }))]
#[derive(Clone)]
pub struct CamK<T: Float> {
    pub model: u32,
    pub f: Param<T>,
    pub k: Param<T>,
    pub c: Param<T>,
    pub pf: T,
    pub pk: T,
    pub pc: T,
    pub hb: SelfBlock<CamK<T>, T>,
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
pub struct ObsK<T: Float> {
    #[arael(ref = root.pts)]
    pub pt: Ref<PtK<T>>,
    #[arael(ref = root.cams)]
    pub cam: Ref<CamK<T>>,
    pub m: T,
    pub k2: T,
    pub kind: u32,
    pub hb: CrossBlock<PtK<T>, CamK<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct SelWorld {
    pub pts: refs::Vec<PtK<f64>>,
    pub cams: refs::Vec<CamK<f64>>,
    pub obs: std::vec::Vec<ObsK<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct SelWorldF {
    pub pts: refs::Vec<PtK<f32>>,
    pub cams: refs::Vec<CamK<f32>>,
    pub obs: std::vec::Vec<ObsK<f32>>,
}

type SelParts<T> = (refs::Vec<PtK<T>>, refs::Vec<CamK<T>>, std::vec::Vec<ObsK<T>>);

/// Four cameras of models 0, 1, 2 and 7 (7 outside the arms: the default),
/// twelve points (two behind the depth `branch`), every camera seeing
/// every point, the measurements the truth plus noise.
fn sel_parts<T: Float>() -> SelParts<T> {
    let v = |x: f64| T::from(x).unwrap();
    let mut rng = Lcg(41);
    let cam_truth = [(0u32, 1.5, 0.1, 0.4), (1, 0.8, 0.2, -0.3), (2, 1.1, -0.1, 0.7), (7, 1.3, 0.05, -0.6)];
    let mut cams = refs::Vec::new();
    for &(model, f, k, c) in &cam_truth {
        cams.push(CamK { model, f: Param::new(v(f * 1.1)), k: Param::new(v(k + 0.05)), c: Param::new(v(c - 0.1)), pf: v(f), pk: v(k), pc: v(c), hb: SelfBlock::new() });
    }
    let pt_truth: Vec<f64> = (0..12).map(|i| if i % 6 == 5 { -0.5 - 0.1 * i as f64 } else { 0.5 + 0.15 * i as f64 }).collect();
    let mut pts = refs::Vec::new();
    for (i, &x) in pt_truth.iter().enumerate() {
        pts.push(PtK { x: Param::new(v(x + 0.05 * rng.uniform())), px: v(x), kind: (i % 6) as u32, k2: v(0.04), hb: SelfBlock::new() });
    }
    let mut obs = std::vec::Vec::new();
    for (i, &x) in pt_truth.iter().enumerate() {
        for (j, &(model, f, k, c)) in cam_truth.iter().enumerate() {
            let pred = match model {
                0 => x * f,
                1 => x * f * (1.0 + k * x * x),
                _ => x + c,
            };
            obs.push(ObsK { pt: pts.ref_at(i), cam: cams.ref_at(j), m: v(pred + 0.01 * rng.uniform()), k2: v(0.25), kind: ((i + j) % 2) as u32, hb: CrossBlock::new() });
        }
    }
    (pts, cams, obs)
}

fn sel_world() -> SelWorld {
    let (pts, cams, obs) = sel_parts();
    SelWorld { pts, cams, obs }
}

fn sel_world_f32() -> SelWorldF {
    let (pts, cams, obs) = sel_parts();
    SelWorldF { pts, cams, obs }
}

/// Index selects in a body (arms writing apart, a default arm taken by a
/// model outside the arms) and picking losses, from instance and entity
/// data: the device assembles as the CPU does and solves to its optimum.
#[test]
fn index_selects_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(sel_world, "index selects");
}

#[test]
fn f32_index_selects_assemble_and_solve_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(sel_world, sel_world_f32, whole_options, "f32 index selects");
}

/// A point whose select has no default.
#[arael::model]
#[arael(constraint(hb, { [select(selnan.k, selnan.x - 1.0, selnan.x * 2.0 - 1.0)] }))]
#[derive(Clone)]
pub struct SelNan<T: Float> {
    pub x: Param<T>,
    pub k: u32,
    pub hb: SelfBlock<SelNan<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct SelNanRoot {
    pub items: refs::Vec<SelNan<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct SelNanRootF {
    pub items: refs::Vec<SelNan<f32>>,
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

/// The f32 twin, on every device.
#[test]
fn an_f32_select_index_outside_its_arms_fails_the_device_solve_at_its_parameter() {
    use arael::simple_lm::{DiagonalFault, LmConfig, SolveFailureKind};
    let solve = |gpu: Option<&GpuDevice>| {
        let mut items = refs::Vec::new();
        for k in [0u32, 1, 7, 0] {
            items.push(SelNan { x: Param::new(0.3f32), k, hb: SelfBlock::new() });
        }
        let mut s = SelNanRootF { items };
        let mut x = Vec::new();
        s.serialize(&mut x);
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&whole_options(gpu));
        arael::simple_lm::lm_solve(&x, &mut solver, &mut s, &LmConfig::<f32>::default())
    };
    assert!(std::panic::catch_unwind(|| solve(None)).is_err(), "the CPU's select panics");
    for d in f32_devices() {
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
    if !checks_f64("the device fast atan") {
        return;
    }
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

/// The f32 twin, in a kernel of floats alone (every device's).
#[test]
fn the_device_fast_atan_is_the_cpus_in_f32() {
    use arael::__gpu::{Arg, Device, Launch, Wide, ASSEMBLY, WIDE};
    const KERNEL: &str = r#"
KERNEL void fast_atans(GLOBAL const float* in, GLOBAL float* out, u32 n) {
    const u32 i = global_x();
    if (i >= n) return;
    const float y = in[2 * i], x = in[2 * i + 1];
    out[2 * i] = arael_fast_atan(y);
    out[2 * i + 1] = arael_fast_atan2(y, x);
}
"#;
    let dev = device();
    let source = format!("{WIDE}\n{ASSEMBLY}\n{KERNEL}");
    let kernel = dev.compile(&source, &format!("{}#define SCALAR float\n", Wide::Df32.defines()), "fast_atans").unwrap();
    let mags = [0.0f32, 1e-8, 0.1, 0.2679492, 0.268, 0.5, 0.57735, 1.0, 1.0000001, 3.7320508, 10.0, 1e6];
    let vals: Vec<f32> = mags.iter().flat_map(|&m| [m, -m]).collect();
    let pairs: Vec<(f32, f32)> = vals.iter().flat_map(|&y| vals.iter().map(move |&x| (y, x))).collect();
    let n = pairs.len();
    let input: Vec<f32> = pairs.iter().flat_map(|&(y, x)| [y, x]).collect();
    let mut inp = dev.alloc(4 * input.len()).unwrap();
    dev.write(&mut inp, 0, arael::__gpu::bytemuck::cast_slice(&input)).unwrap();
    let out = dev.alloc(8 * n).unwrap();
    let args = [Arg::Block(&inp, 0), Arg::Block(&out, 0), Arg::U32(n as u32)];
    dev.launch(&kernel, Launch::linear(n as u32, 256), &args).unwrap();
    let mut f = vec![0f32; 2 * n];
    dev.read(&out, 0, arael::__gpu::bytemuck::cast_slice_mut(&mut f)).unwrap();
    for (k, &(y, x)) in pairs.iter().enumerate() {
        let cpu = [arael::utils::fast_atan(y), arael::utils::fast_atan2(y, x)];
        for j in 0..2 {
            assert!((f[2 * k + j] - cpu[j]).abs() <= 8.0 * f32::EPSILON, "({y}, {x}) [{j}]: {} vs {}", f[2 * k + j], cpu[j]);
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
pub struct PoseB<T: Float> {
    pub pos: Param<vect2<T>>,
    pub th: Param<T>,
    pub px: T,
    pub py: T,
    pub pth: T,
    pub hb: SelfBlock<PoseB<T>, T>,
}

/// A landmark at a known place seen from a pose by bearing and squared
/// range, through the user functions; the root's atan2 the fast one.
#[arael::model]
#[arael(constraint(p.hb, {
    let d = offset_dev(p.pos, bearing.l);
    [rad_diff(atan2(d.y, d.x) - p.th, bearing.b) * 20.0, (sq_dev(d.x) + sq_dev(d.y) - bearing.r2) * 0.5]
}))]
#[derive(Clone)]
pub struct Bearing<T: Float> {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseB<T>>,
    pub l: vect2<T>,
    pub b: T,
    pub r2: T,
}

#[arael::model]
#[arael(root, fast_atan)]
#[derive(Clone)]
pub struct BearingWorld {
    pub poses: refs::Vec<PoseB<f64>>,
    pub bearings: std::vec::Vec<Bearing<f64>>,
}

#[arael::model]
#[arael(root, f32, fast_atan)]
#[derive(Clone)]
pub struct BearingWorldF {
    pub poses: refs::Vec<PoseB<f32>>,
    pub bearings: std::vec::Vec<Bearing<f32>>,
}

/// Six poses on a circle seeing five landmarks, the measurements the truth
/// plus noise, the poses moved off it.
fn bearing_parts<T: Float>() -> (refs::Vec<PoseB<T>>, std::vec::Vec<Bearing<T>>) {
    let f = |v: f64| T::from(v).unwrap();
    let mut rng = Lcg(61);
    let truth: Vec<(f64, f64, f64)> = (0..6).map(|k| {
        let a = k as f64;
        (3.0 * a.cos(), 3.0 * a.sin(), a + 1.2)
    }).collect();
    let marks = [(0.5, 0.3), (-1.0, 1.5), (2.0, -1.0), (-2.5, -2.0), (4.5, 1.0)];
    let mut poses = refs::Vec::new();
    for &(x, y, th) in &truth {
        poses.push(PoseB {
            pos: Param::new(vect2::new(f(x + 0.1 * rng.uniform()), f(y + 0.1 * rng.uniform()))),
            th: Param::new(f(th + 0.05 * rng.uniform())),
            px: f(x),
            py: f(y),
            pth: f(th),
            hb: SelfBlock::new(),
        });
    }
    let mut bearings = std::vec::Vec::new();
    for (k, &(x, y, th)) in truth.iter().enumerate() {
        for &(lx, ly) in &marks {
            let (dx, dy) = (lx - x, ly - y);
            bearings.push(Bearing {
                p: poses.ref_at(k),
                l: vect2::new(f(lx), f(ly)),
                b: f(arael::utils::rad_diff(dy.atan2(dx), th) + 0.002 * rng.uniform()),
                r2: f(dx * dx + dy * dy + 0.01 * rng.uniform()),
            });
        }
    }
    (poses, bearings)
}

fn bearing_world() -> BearingWorld {
    let (poses, bearings) = bearing_parts();
    BearingWorld { poses, bearings }
}

fn bearing_world_f32() -> BearingWorldF {
    let (poses, bearings) = bearing_parts();
    BearingWorldF { poses, bearings }
}

/// A root's fast atan2 and user functions with a symbolic body (inlined)
/// assemble on the device as on the CPU and solve to its optimum.
#[test]
fn fast_atan_and_symbolic_user_functions_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(bearing_world, "fast atan2 and user functions");
}

#[test]
fn f32_fast_atan_and_symbolic_user_functions_assemble_and_solve_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(bearing_world, bearing_world_f32, whole_options, "f32 fast atan2 and user functions");
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
pub struct Clash<T: Float> {
    pub x: Param<T>,
    pub p: vect2<T>,
    pub p_x: T,
    pub hb: SelfBlock<Clash<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct ClashRoot {
    pub items: refs::Vec<Clash<f64>>,
    pub q: vect2<f64>,
    pub q_x: f64,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct ClashRootF {
    pub items: refs::Vec<Clash<f32>>,
    pub q: vect2<f32>,
    pub q_x: f32,
}

fn clash_items<T: Float>() -> refs::Vec<Clash<T>> {
    let f = |v: f64| T::from(v).unwrap();
    let mut items = refs::Vec::new();
    for k in 0..5 {
        let t = 0.4 * k as f64;
        items.push(Clash { x: Param::new(f(t + 0.2)), p: vect2::new(f(t), f(-1.0)), p_x: f(0.1 * t), hb: SelfBlock::new() });
    }
    items
}

fn clash_root() -> ClashRoot {
    ClashRoot { items: clash_items(), q: vect2::new(0.7, 9.0), q_x: -0.2 }
}

fn clash_root_f32() -> ClashRootF {
    ClashRootF { items: clash_items(), q: vect2::new(0.7, 9.0), q_x: -0.2 }
}

/// Paths that spell alike as record fields (`p.x`, `p_x`) take a field
/// each; the model builds (it did not, with or without the `gpu`
/// feature) and assembles on the device as on the CPU.
#[test]
fn data_fields_of_alike_names_assemble_and_solve_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(clash_root, "alike names");
}

#[test]
fn f32_data_fields_of_alike_names_assemble_and_solve_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(clash_root, clash_root_f32, whole_options, "f32 alike names");
}

/// Optional data of an entity, read under a guard on it.
#[arael::model]
#[derive(Clone)]
pub struct Extra<T: Float> {
    pub off: T,
}

#[arael::model]
#[arael(constraint(hb, { [(optent.x - optent.px) * 0.5] }))]
#[arael(constraint(hb, guard = self.has_extra, { [(optent.x - optent.extra.off) * 2.0] }))]
#[arael(constraint(hb, guard = root.has_bias, { [(optent.x - optent.px) * root.bias.off] }))]
#[derive(Clone)]
pub struct OptEnt<T: Float> {
    pub x: Param<T>,
    pub px: T,
    pub extra: Option<Extra<T>>,
    pub has_extra: bool,
    pub hb: SelfBlock<OptEnt<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct OptRoot {
    pub items: refs::Vec<OptEnt<f64>>,
    pub bias: Option<Extra<f64>>,
    pub has_bias: bool,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct OptRootF {
    pub items: refs::Vec<OptEnt<f32>>,
    pub bias: Option<Extra<f32>>,
    pub has_bias: bool,
}

/// Six entities, every other one with its optional data.
fn opt_items<T: Float>() -> refs::Vec<OptEnt<T>> {
    let mut items = refs::Vec::new();
    for k in 0..6 {
        let some = k % 2 == 0;
        let px = T::from(0.3 * k as f64).unwrap();
        items.push(OptEnt {
            x: Param::new(px + T::from(0.2).unwrap()),
            px,
            extra: some.then(|| Extra { off: px }),
            has_extra: some,
            hb: SelfBlock::new(),
        });
    }
    items
}

/// [`opt_items`], the root's optional data present or not.
fn opt_root(bias: bool) -> OptRoot {
    OptRoot { items: opt_items(), bias: bias.then(|| Extra { off: 0.4 }), has_bias: bias }
}

fn opt_root_f32(bias: bool) -> OptRootF {
    OptRootF { items: opt_items(), bias: bias.then(|| Extra { off: 0.4 }), has_bias: bias }
}

/// Optional data of entities and of the root read under guards on it:
/// the device writes what a `None` leaves as NaN, which no kept instance
/// reads (it panicked writing it), and assembles as the CPU does.
#[test]
fn optional_data_read_under_a_guard_assembles_and_solves_on_the_device_as_on_the_cpu() {
    assembles_and_solves_as_on_the_cpu(|| opt_root(true), "the root's data present");
    assembles_and_solves_as_on_the_cpu(|| opt_root(false), "the root's data absent");
}

#[test]
fn f32_optional_data_read_under_a_guard_assembles_and_solves_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(|| opt_root(true), || opt_root_f32(true), whole_options, "f32, the root's data present");
    f32_twin_assembles_and_solves(|| opt_root(false), || opt_root_f32(false), whole_options, "f32, the root's data absent");
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
pub struct SimPose<T: Float> {
    pub stp: arael::transform::ScaledTransformParam<T>,
    pub x1: vect3<T>,
    pub x2: vect3<T>,
    pub x3: vect3<T>,
    pub m1: vect3<T>,
    pub m2: vect3<T>,
    pub m3: vect3<T>,
    pub hb: SelfBlock<SimPose<T>, T>,
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
pub struct SimLink<T: Float> {
    #[arael(ref = root.poses)]
    pub a: Ref<SimPose<T>>,
    #[arael(ref = root.poses)]
    pub b: Ref<SimPose<T>>,
    pub dt: vect3<T>,
    pub rmeas_t: arael::matrix::matrix3<T>,
    pub dls: T,
    pub hb: CrossBlock<SimPose<T>, SimPose<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct SimWorld {
    pub poses: refs::Vec<SimPose<f64>>,
    pub links: std::vec::Vec<SimLink<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct SimWorldF {
    pub poses: refs::Vec<SimPose<f32>>,
    pub links: std::vec::Vec<SimLink<f32>>,
}

fn sim_world() -> SimWorld {
    let (poses, links) = sim_parts();
    SimWorld { poses, links }
}

fn sim_world_f32() -> SimWorldF {
    let (poses, links) = sim_parts();
    SimWorldF { poses, links }
}

/// Six Sim(3) poses along a turning path, each seeing three known points,
/// links between neighbours and one closing the loop, the measurements the
/// truth plus noise, the poses moved off it; the first pose fixed.
fn sim_parts<T: Float>() -> (refs::Vec<SimPose<T>>, std::vec::Vec<SimLink<T>>) {
    use arael::quatern::quaternd;
    use arael::vect::vect3d;
    let f = |v: f64| T::from(v).unwrap();
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
            arael::transform::ScaledTransformParam::fixed(t.cast(), q.cast(), f(s))
        } else {
            let dq = quaternd::from_rotation_vector(jitter(&mut rng, 0.05));
            arael::transform::ScaledTransformParam::new((t + jitter(&mut rng, 0.05)).cast(), (q * dq).unit().cast(), f(s * 1.05))
        };
        let (x, m) = (|k: usize| xs[k].cast(), |k: usize| m[k].cast());
        poses.push(SimPose { stp, x1: x(0), x2: x(1), x3: x(2), m1: m(0), m2: m(1), m3: m(2), hb: SelfBlock::new() });
    }
    let mut links = std::vec::Vec::new();
    for (a, b) in (0..6).map(|a| (a, (a + 1) % 6)) {
        let ((ta, qa, sa), (tb, qb, sb)) = (truth[a], truth[b]);
        let ra = qa.rotation_matrix();
        links.push(SimLink {
            a: poses.ref_at(a),
            b: poses.ref_at(b),
            dt: (ra.transpose() * (tb - ta) + jitter(&mut rng, 0.01)).cast(),
            rmeas_t: (ra.transpose() * qb.rotation_matrix()).transpose().cast(),
            dls: f(sb.ln() - sa.ln() + 0.002 * rng.uniform()),
            hb: CrossBlock::new(),
        });
    }
    (poses, links)
}

/// A `#[arael(symbolic = exp(w))]` field of an entity: the body reads its
/// cache, which the device computes from the parameter as it goes (the
/// stored field follows the parameter between steps).
#[arael::model]
#[arael(constraint(hb, { [(curvee.target - curvee.base * curvee.scale) * curvee.isigma] }))]
#[derive(Clone)]
pub struct CurveE<T: Float> {
    pub w: Param<T>,
    pub base: T,
    pub target: T,
    pub isigma: T,
    #[arael(symbolic = exp(w))]
    pub scale: T,
    pub hb: SelfBlock<CurveE<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct CurvesE {
    pub curves: refs::Vec<CurveE<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct CurvesEF {
    pub curves: refs::Vec<CurveE<f32>>,
}

fn curve_e_list<T: Float>() -> refs::Vec<CurveE<T>> {
    let f = |v: f64| T::from(v).unwrap();
    let mut curves = refs::Vec::new();
    for k in 0..6 {
        let (base, w) = (1.0 + 0.5 * k as f64, 0.3 * k as f64 - 0.7);
        curves.push(CurveE { w: Param::new(f(w + 0.3)), base: f(base), target: f(base * w.exp()), isigma: f(1.5), scale: T::zero(), hb: SelfBlock::new() });
    }
    curves
}

fn curves_e() -> CurvesE {
    CurvesE { curves: curve_e_list() }
}

fn curves_e_f32() -> CurvesEF {
    CurvesEF { curves: curve_e_list() }
}

/// A user component over parameters alone: its caches, values and a
/// declared derivative, computed by the device from the parameters.
#[arael::model]
#[arael(component)]
#[derive(Clone)]
pub struct Polar<T: Float> {
    pub r: Param<T>,
    pub th: Param<T>,
    #[arael(symbolic = r * cos(th))]
    pub x: T,
    #[arael(symbolic = r * sin(th))]
    pub y: T,
    #[arael(deriv = x, by = th)]
    pub x_th: [T; 1],
}

impl<T: Float> arael::model::Component for Polar<T> {}

#[arael::model]
#[arael(constraint(hb, { [(beacon.polar.x - beacon.mx) * 2.0, (beacon.polar.y - beacon.my) * 2.0, (beacon.polar.r - beacon.mr) * 0.5] }))]
#[derive(Clone)]
pub struct Beacon<T: Float> {
    pub polar: Polar<T>,
    pub mx: T,
    pub my: T,
    pub mr: T,
    pub hb: SelfBlock<Beacon<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Beacons {
    pub beacons: refs::Vec<Beacon<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct BeaconsF {
    pub beacons: refs::Vec<Beacon<f32>>,
}

fn beacon_list<T: Float>() -> refs::Vec<Beacon<T>> {
    let f = |v: f64| T::from(v).unwrap();
    let mut rng = Lcg(52);
    let mut beacons = refs::Vec::new();
    for k in 0..6 {
        let (r, th) = (1.0 + 0.4 * k as f64, 0.5 * k as f64 - 1.0);
        beacons.push(Beacon {
            polar: Polar { r: Param::new(f(r + 0.1)), th: Param::new(f(th - 0.1)), x: T::zero(), y: T::zero(), x_th: [T::zero()] },
            mx: f(r * th.cos() + 0.01 * rng.uniform()),
            my: f(r * th.sin() + 0.01 * rng.uniform()),
            mr: f(r),
            hb: SelfBlock::new(),
        });
    }
    beacons
}

fn beacons() -> Beacons {
    Beacons { beacons: beacon_list() }
}

fn beacons_f32() -> BeaconsF {
    BeaconsF { beacons: beacon_list() }
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

/// The f32 twin: the caches and the Sim(3) poses on every device, the
/// device's re-centred references written back into the model.
#[test]
fn f32_caches_over_parameters_assemble_and_solve_on_the_device_as_on_the_cpu() {
    use arael::simple_lm::SparseFaer;
    f32_twin_assembles_and_solves(curves_e, curves_e_f32, whole_options, "f32, a symbolic field of an entity");
    f32_twin_assembles_and_solves(beacons, beacons_f32, whole_options, "f32, a user component");
    f32_twin_assembles_and_solves(sim_world, sim_world_f32, whole_options, "f32 Sim(3) poses");
    for d in f32_devices() {
        let mut s = sim_world_f32();
        let r = s.solve_with(&mut SparseFaer::<f32>::from_options(&whole_options(Some(&d))), &to_the_optimum()).unwrap();
        left_behind_costs(&mut s, &r, &format!("{}: the Sim(3) poses written back", label(&d)));
    }
}

/// The plane SLAM model of benchmarks/plane: `TransformParam` poses,
/// planes of a `UnitVecParam` normal and a distance, odometry and plane
/// observations.
#[arael::model]
#[derive(Clone)]
pub struct PoseT<T: Float> {
    pub r2w: arael::transform::TransformParam<T>,
    pub hb: SelfBlock<PoseT<T>, T>,
}

#[arael::model]
#[derive(Clone)]
pub struct PlaneT<T: Float> {
    pub n: arael::unitvec::UnitVecParam<T>,
    pub c: Param<T>,
    pub hb: SelfBlock<PlaneT<T>, T>,
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
pub struct OdoT<T: Float> {
    #[arael(ref = root.poses)]
    pub a: Ref<PoseT<T>>,
    #[arael(ref = root.poses)]
    pub b: Ref<PoseT<T>>,
    pub measured_translation: vect3<T>,
    pub measured_rotation_transposed: arael::matrix::matrix3<T>,
    pub wt: T,
    pub wr: T,
    pub hb: CrossBlock<PoseT<T>, PoseT<T>, T>,
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
pub struct ObsT<T: Float> {
    #[arael(ref = root.poses)]
    pub p: Ref<PoseT<T>>,
    #[arael(ref = root.planes)]
    pub l: Ref<PlaneT<T>>,
    pub measured_normal: vect3<T>,
    pub measured_c: T,
    pub wa: T,
    pub wd: T,
    pub hb: CrossBlock<PoseT<T>, PlaneT<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct WorldT {
    pub poses: refs::Vec<PoseT<f64>>,
    pub planes: refs::Vec<PlaneT<f64>>,
    pub odos: std::vec::Vec<OdoT<f64>>,
    pub obs: std::vec::Vec<ObsT<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct WorldTF {
    pub poses: refs::Vec<PoseT<f32>>,
    pub planes: refs::Vec<PlaneT<f32>>,
    pub odos: std::vec::Vec<OdoT<f32>>,
    pub obs: std::vec::Vec<ObsT<f32>>,
}

type PlaneParts<T> = (refs::Vec<PoseT<T>>, refs::Vec<PlaneT<T>>, std::vec::Vec<OdoT<T>>, std::vec::Vec<ObsT<T>>);

fn plane_world(n: usize, seed: u64) -> WorldT {
    let (poses, planes, odos, obs) = plane_parts(n, seed);
    WorldT { poses, planes, odos, obs }
}

fn plane_world_f32(n: usize, seed: u64) -> WorldTF {
    let (poses, planes, odos, obs) = plane_parts(n, seed);
    WorldTF { poses, planes, odos, obs }
}

/// `n` poses along a curve looking at 8 planes, odometry between
/// neighbours, each pose observing every plane, the measurements the truth
/// plus noise, the poses and planes moved off it. Pose 0 is fixed, pose 3
/// holds its rotation, plane 5 holds its normal.
fn plane_parts<T: Float>(n: usize, seed: u64) -> PlaneParts<T> {
    use arael::matrix::matrix3d;
    use arael::quatern::quatern;
    use arael::vect::vect3d;
    let f = |v: f64| T::from(v).unwrap();
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
        let mut r2w = arael::transform::TransformParam::new(tt.cast(), quatern::from_rotation_matrix(*r * dr).cast());
        if k == 0 {
            r2w = arael::transform::TransformParam::fixed(tt.cast(), quatern::from_rotation_matrix(*r).cast());
        }
        if k == 3 {
            r2w.optimize_rotation = false;
        }
        poses.push(PoseT { r2w, hb: SelfBlock::new() });
    }
    let mut planes = refs::Vec::new();
    for (k, (nrm, c)) in planes_truth.iter().enumerate() {
        let moved = *nrm + vect3d::new(0.02 * rng.uniform(), 0.02 * rng.uniform(), 0.02 * rng.uniform());
        let n = if k == 5 { arael::unitvec::UnitVecParam::fixed(nrm.cast()) } else { arael::unitvec::UnitVecParam::new(moved.cast()) };
        planes.push(PlaneT { n, c: Param::new(f(c + 0.05 * rng.uniform())), hb: SelfBlock::new() });
    }
    let mut odos = std::vec::Vec::new();
    for k in 1..n {
        let ((ta, ra), (tb, rb)) = (truth[k - 1], truth[k]);
        let noise = matrix3d::rotation_from_euler_angles(vect3d::new(0.002 * rng.uniform(), 0.002 * rng.uniform(), 0.002 * rng.uniform()));
        odos.push(OdoT {
            a: poses.ref_at(k - 1),
            b: poses.ref_at(k),
            measured_translation: (ra.transpose() * (tb - ta) + vect3d::new(0.005 * rng.uniform(), 0.005 * rng.uniform(), 0.005 * rng.uniform())).cast(),
            measured_rotation_transposed: (ra.transpose() * rb * noise).transpose().cast(),
            wt: f(20.0),
            wr: f(50.0),
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
                measured_normal: (nl * (1.0 / (nl * nl).sqrt())).cast(),
                measured_c: f(c + *t * *nrm + 0.01 * rng.uniform()),
                wa: f(30.0),
                wd: f(10.0),
                hb: CrossBlock::new(),
            });
        }
    }
    (poses, planes, odos, obs)
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
    if checks_f64("the plane world") {
        for (wide, tol) in [(Wide::F64, 1e-11), (Wide::Df32, 1e-10)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        }
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

/// The f32 twin: the plane world on every device, the poses' and the
/// normals' references written back into the model.
#[test]
fn an_f32_plane_world_assembles_and_solves_on_the_device_as_on_the_cpu() {
    use arael::simple_lm::SparseFaer;
    let what = "the f32 plane world";
    f32_twin_assembles_and_solves(|| plane_world(30, 9), || plane_world_f32(30, 9), whole_options, what);
    for d in f32_devices() {
        let mut s = plane_world_f32(30, 9);
        let r = s.solve_with(&mut SparseFaer::<f32>::from_options(&whole_options(Some(&d))), &to_the_optimum()).unwrap();
        left_behind_costs(&mut s, &r, &format!("{what} {}", label(&d)));
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

/// The f32 twin: on every device, a session's solves through pageable
/// staging land on the f64 scene's optimum as it is at each solve.
#[test]
fn an_f32_image_staged_in_pageable_memory_solves_to_the_optimum() {
    use arael::simple_lm::{GpuStaging, LmSession, SparseFaer};
    let optimum = |s: &Scene| s.clone().solve_with(&mut SparseFaer::<f64>::from_options(&options(None, false)), &to_the_optimum()).unwrap().end_cost;
    for d in f32_devices() {
        let d = d.with_staging(GpuStaging::Pageable);
        let name = label(&d);
        let (cameras, points, observations) = consistent_scene::<f64>();
        let mut s64 = Scene { cameras, points, observations };
        let (cameras, points, observations) = consistent_scene::<f32>();
        let mut s = SceneF { cameras, points, observations };
        let mut session = LmSession::new(SparseFaer::<f32>::from_options(&options(Some(&d), true)));
        for round in 0..2 {
            let opt = optimum(&s64);
            let r = session.solve(&mut s, &to_the_optimum()).unwrap();
            assert!(session.solver().plan().unwrap().gpu_assembly);
            assert!((r.end_cost as f64 - opt).abs() <= 1e-4 * opt, "{name}, solve {round}: {} against the optimum {opt}", r.end_cost);
            left_behind_costs(&mut s, &r, &format!("{name}, solve {round}"));
            s.observations[5].xy.x += 3.0;
            s64.observations[5].xy.x += 3.0;
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

/// The f32 twin: on every device the covariance after an f32 device solve
/// factors under the solve's analysis and matches the f64 CPU solve's to
/// f32 precision. Pose 7's marginal rests on H's smallest eigenvalue (one
/// prior, at the loop's start): in f32 it is good to about 1% at any
/// solve's point, the CPU's f32 route's included.
#[test]
fn an_f32_covariance_after_a_device_solve_reuses_its_analysis() {
    use arael::covariance::CovMode;
    let reference = {
        let (poses, edges, prior) = pose_graph::<f64>(40, 3);
        let mut s = Graph2D { poses, edges, prior };
        let cfg = arael::simple_lm::LmConfig::<f64> { covariance: Some(CovMode::PerQuery), ..to_the_optimum() };
        let r = s.solve_with(&mut arael::simple_lm::SparseFaer::<f64>::from_options(&whole_options(None)), &cfg).unwrap();
        r.covariance.as_ref().expect("the covariance").marginal_cov(&s.poses[7]).unwrap()
    };
    let scale = reference.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    for d in f32_devices() {
        let (poses, edges, prior) = pose_graph::<f32>(40, 3);
        let mut s = Graph2DF { poses, edges, prior };
        let mut solver = arael::simple_lm::SparseFaer::<f32>::from_options(&whole_options(Some(&d)));
        let cfg = arael::simple_lm::LmConfig::<f32> { covariance: Some(CovMode::PerQuery), ..to_the_optimum() };
        let r = s.solve_with(&mut solver, &cfg).unwrap();
        assert!(solver.plan().unwrap().gpu_assembly, "{}", label(&d));
        let cov = r.covariance.as_ref().expect("the covariance");
        assert!(cov.took_block_route());
        assert_eq!(cov.plan().symbolics_built, 0, "{}: the covariance factors under the solve's analysis", label(&d));
        let dev = cov.marginal_cov(&s.poses[7]).unwrap();
        let diff = dev.iter().zip(reference.iter()).fold(0.0f64, |m, (&a, &b)| m.max((a as f64 - b).abs()));
        assert!(diff <= 3e-2 * scale, "{}: the marginal covariance differs by {diff} of {scale}", label(&d));
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
pub struct SPose<T: Float> {
    pub pos: Param<vect3<T>>,
    pub ea: SimpleEulerAngleParam<T>,
    pub prior_pos: vect3<T>,
    pub prior_ea: vect3<T>,
    pub gps_pos: vect3<T>,
    pub gps_cov_r: arael::matrix::matrix3<T>,
    pub gps_isigma: vect3<T>,
    pub tilt_roll: T,
    pub tilt_pitch: T,
    pub hb_pose: SelfBlock<SPose<T>, T>,
}

#[arael::model]
#[arael(constraint(hb_drift, {
    let d = slm.pos - slm.prior_pos;
    [d.x * root.drift_lm_isigma, d.y * root.drift_lm_isigma, d.z * root.drift_lm_isigma]
}))]
#[derive(Clone)]
pub struct SLm<T: Float> {
    pub pos: Param<vect3<T>>,
    pub prior_pos: vect3<T>,
    pub frines: std::vec::Vec<SFrine<T>>,
    pub hb_drift: SelfBlock<SLm<T>, T>,
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
pub struct SFrine<T: Float> {
    #[arael(ref = root.poses)]
    pub pose: Ref<SPose<T>>,
    pub mf2r: arael::matrix::matrix3<T>,
    pub camera_pos: vect3<T>,
    pub isigma: vect2<T>,
    pub hb: CrossBlock<SLm<T>, SPose<T>, T>,
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
pub struct SPair<T: Float> {
    #[arael(ref = root.poses)]
    pub prev: Ref<SPose<T>>,
    #[arael(ref = root.poses)]
    pub cur: Ref<SPose<T>>,
    pub delta_pos: vect3<T>,
    pub delta_ea: vect3<T>,
    pub pos_cov_r: arael::matrix::matrix3<T>,
    pub pos_isigma: vect3<T>,
    pub ea_cov_r: arael::matrix::matrix3<T>,
    pub ea_isigma: vect3<T>,
    pub hb: CrossBlock<SPose<T>, SPose<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct SPath {
    pub poses: refs::Vec<SPose<f64>>,
    pub landmarks: refs::Arena<SLm<f64>>,
    pub pairs: std::vec::Vec<SPair<f64>>,
    pub drift_pos_isigma: f64,
    pub drift_ea_isigma: f64,
    pub drift_lm_isigma: f64,
    pub tilt_isigma: f64,
    pub frine_isigma_scale: f64,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct SPathF {
    pub poses: refs::Vec<SPose<f32>>,
    pub landmarks: refs::Arena<SLm<f32>>,
    pub pairs: std::vec::Vec<SPair<f32>>,
    pub drift_pos_isigma: f32,
    pub drift_ea_isigma: f32,
    pub drift_lm_isigma: f32,
    pub tilt_isigma: f32,
    pub frine_isigma_scale: f32,
}

type SPathParts<T> = (refs::Vec<SPose<T>>, refs::Arena<SLm<T>>, std::vec::Vec<SPair<T>>);

fn slam_path(n: usize, m: usize, seed: u64) -> SPath {
    let (poses, landmarks, pairs) = slam_parts(n, m, seed);
    SPath { poses, landmarks, pairs, drift_pos_isigma: 0.01, drift_ea_isigma: 0.01, drift_lm_isigma: 0.01, tilt_isigma: 10.0, frine_isigma_scale: 1.0 }
}

fn slam_path_f32(n: usize, m: usize, seed: u64) -> SPathF {
    let (poses, landmarks, pairs) = slam_parts(n, m, seed);
    SPathF { poses, landmarks, pairs, drift_pos_isigma: 0.01, drift_ea_isigma: 0.01, drift_lm_isigma: 0.01, tilt_isigma: 10.0, frine_isigma_scale: 1.0 }
}

/// `n` poses along a winding path and `m` landmarks beside it, each seen
/// by the poses within four steps of it, the measurements the truth plus
/// noise, the poses and landmarks moved off it.
fn slam_parts<T: Float>(n: usize, m: usize, seed: u64) -> SPathParts<T> {
    use arael::matrix::matrix3d;
    use arael::quatern::quatern;
    use arael::vect::vect3d;
    let f = |v: f64| T::from(v).unwrap();
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
            pos: Param::new(p0.cast()),
            ea: SimpleEulerAngleParam::new(ea0.cast()),
            prior_pos: p0.cast(),
            prior_ea: ea0.cast(),
            gps_pos: (p + jitter(&mut rng, 0.1)).cast(),
            gps_cov_r: matrix3d::identity().cast(),
            gps_isigma: vect3d::new(2.0, 2.0, 1.0).cast(),
            tilt_roll: f(ea.x + 0.002 * rng.uniform()),
            tilt_pitch: f(ea.y + 0.002 * rng.uniform()),
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
                mf2r: (quatern::from_two_vectors(vect3d::new(1.0, 0.0, 0.0), d).rotation_matrix() * noise).cast(),
                camera_pos: vect3::new(f(0.0), f(0.0), f(0.0)),
                isigma: vect2::new(f(100.0), f(100.0)),
                hb: CrossBlock::new(),
            });
        }
        let _ = j;
        let l0 = (l + jitter(&mut rng, 0.1)).cast();
        landmarks.push(SLm { pos: Param::new(l0), prior_pos: l0, frines, hb_drift: SelfBlock::new() });
    }
    let mut pairs = std::vec::Vec::new();
    for k in 1..n {
        let ((pa, ea), (pb, eb)) = (truth[k - 1], truth[k]);
        let (ra, rb) = (matrix3d::rotation_from_euler_angles(ea), matrix3d::rotation_from_euler_angles(eb));
        pairs.push(SPair {
            prev: poses.ref_at(k - 1),
            cur: poses.ref_at(k),
            delta_pos: (ra.transpose() * (pb - pa) + jitter(&mut rng, 0.01)).cast(),
            delta_ea: ((ra.transpose() * rb).get_euler_angles() + jitter(&mut rng, 0.002)).cast(),
            pos_cov_r: matrix3d::identity().cast(),
            pos_isigma: vect3d::new(20.0, 20.0, 20.0).cast(),
            ea_cov_r: matrix3d::identity().cast(),
            ea_isigma: vect3d::new(50.0, 50.0, 50.0).cast(),
            hb: CrossBlock::new(),
        });
    }
    (poses, landmarks, pairs)
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
        format!("//   E0: `SPose` ({path}:{}), self block `hb_pose`, parameters: pos.work().x, pos.work().y, pos.work().z, ea.work().x, ea.work().y, ea.work().z", line("pub struct SPose<T: Float> {")),
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
    if checks_f64("the slam path") {
        for (wide, tol) in [(Wide::F64, 1e-11), (Wide::Df32, 1e-10)] {
            let chk = arael::__gpu::device_check(&mut s, &x, &x1, &device(), wide);
            assert!((chk.device_cost - chk.cpu_cost).abs() <= tol * chk.cpu_cost, "{wide:?}: {chk:?}");
            assert!(chk.grad_err < tol && chk.hess_err < tol, "{wide:?}: {chk:?}");
        }
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

#[test]
fn an_f32_slam_path_assembles_and_solves_on_the_device_as_on_the_cpu() {
    f32_twin_assembles_and_solves(|| slam_path(24, 30, 13), || slam_path_f32(24, 30, 13), whole_options, "the f32 slam path");
}

