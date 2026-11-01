//! The device assembly (feature `gpu-cuda`): a root whose constraints all
//! have a device form assembles its gradient, Hessian tiles and cost on
//! the GPU to what the CPU assembly gives against the same pattern.
#![cfg(feature = "gpu-cuda")]

use arael::model::{CrossBlock, MixedEulerAngleParam, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{LmProblemInternals, RootProblem};
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
