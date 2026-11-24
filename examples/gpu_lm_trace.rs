//! Counts the CUDA launches, waits and copies of each attempt of a bundle
//! adjustment solved on the GPU. Run it under `scripts/gpu_trace.py`, which
//! needs NVIDIA's Nsight Systems (`nsys`); build it with
//! `--features gpu-cuda` and run it on an NVIDIA GPU.
//!
//!   cargo build -r --features gpu-cuda --example gpu_lm_trace
//!   scripts/gpu_trace.py -- target/release/examples/gpu_lm_trace
//!
//! The problem is a synthetic bundle adjustment (300 cameras, 30000
//! points), solved with the points eliminated (Schur complement) on CUDA
//! device 0. After each damped attempt it writes a marker copy, which the
//! script uses to split the trace into attempts; the marker's own copy and
//! wait are counted too.
//! The markers go through a CUDA device of their own. `--f32-factor`
//! factors in f32 and refines the steps.

use arael::model::{CrossBlock, EulerAngleParam, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{GpuDevice, LmConfig, LmIter, LmProblem, MixedPrecision, SchurPolicy, SparseFaer, SparseFaerOptions};
use arael::vect::{vect2, vect3, vect3d};

#[arael::model]
pub struct Camera {
    pub t: Param<vect3<f64>>,
    pub ea: EulerAngleParam<f64>,
    pub hb: SelfBlock<Camera>,
}

#[arael::model]
pub struct Point {
    pub pos: Param<vect3<f64>>,
    pub hb: SelfBlock<Point>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    [500.0 * pc.x / pc.z - obs.xy.x, 500.0 * pc.y / pc.z - obs.xy.y]
}))]
pub struct Obs {
    #[arael(ref = root.cameras)]
    pub cam: Ref<Camera>,
    #[arael(ref = root.points)]
    pub pt: Ref<Point>,
    pub xy: vect2<f64>,
    pub hb: CrossBlock<Camera, Point>,
}

#[arael::model]
#[arael(root)]
pub struct Scene {
    pub cameras: refs::Vec<Camera>,
    pub points: refs::Vec<Point>,
    pub observations: std::vec::Vec<Obs>,
}

/// Marker copies are `MARKER + id` bytes; the trace script knows the range.
const MARKER: usize = 1_000_003;

struct Lcg(u64);
impl Lcg {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 - 0.5
    }
}

/// `nc` cameras in a row at z = 10 looking down at `np` points near the
/// ground, each point seen by four neighbouring cameras, observations a
/// pixel of noise off the truth, the start moved off it.
fn scene(nc: usize, np: usize) -> Scene {
    let mut rng = Lcg(5);
    let mut cameras = refs::Vec::new();
    let mut truth = Vec::new();
    for k in 0..nc {
        let ea = vect3d::new(std::f64::consts::PI + 0.02 * rng.uniform(), 0.02 * rng.uniform(), 0.1 * rng.uniform());
        let t = vect3d::new(-(k as f64), 0.0, 10.0);
        truth.push((ea, t));
        cameras.push(Camera {
            t: Param::new(t + vect3d::new(0.05 * rng.uniform(), 0.05 * rng.uniform(), 0.05 * rng.uniform())),
            ea: EulerAngleParam::new(ea + vect3d::new(0.002 * rng.uniform(), 0.002 * rng.uniform(), 0.002 * rng.uniform())),
            hb: SelfBlock::new(),
        });
    }
    let mut points = refs::Vec::new();
    let mut observations = std::vec::Vec::new();
    for p in 0..np {
        let first = p % (nc - 3);
        let pos = vect3d::new(first as f64 + 1.5 + 2.0 * rng.uniform(), 4.0 * rng.uniform(), 0.5 * rng.uniform());
        points.push(Point { pos: Param::new(pos + vect3d::new(0.05 * rng.uniform(), 0.05 * rng.uniform(), 0.05 * rng.uniform())), hb: SelfBlock::new() });
        for c in first..first + 4 {
            let (ea, t) = truth[c];
            let pc = arael::matrix::matrix3d::rotation_from_euler_angles(ea) * pos + t;
            observations.push(Obs {
                cam: cameras.ref_at(c),
                pt: points.ref_at(p),
                xy: vect2::new(500.0 * pc.x / pc.z + rng.uniform(), 500.0 * pc.y / pc.z + rng.uniform()),
                hb: CrossBlock::new(),
            });
        }
    }
    Scene { cameras, points, observations }
}

/// Writes the marker of an id and waits for it.
type Mark = std::sync::Arc<dyn Fn(usize) + Send + Sync>;

/// The device, and markers written through a device of their own.
#[cfg(feature = "gpu-cuda")]
fn marks() -> (GpuDevice, Mark) {
    use arael::__gpu::{Device, cuda::CudaDevice};
    let dev = CudaDevice::new(0).expect("a CUDA device");
    let buf = std::sync::Mutex::new(dev.alloc(MARKER + 4096).unwrap());
    let host = vec![0u8; MARKER + 4096];
    let mark = move |id: usize| {
        dev.write(&mut buf.lock().unwrap(), 0, &host[..MARKER + id]).unwrap();
        dev.sync().unwrap();
    };
    (GpuDevice::cuda(0), std::sync::Arc::new(mark))
}

#[cfg(not(feature = "gpu-cuda"))]
fn marks() -> (GpuDevice, Mark) {
    panic!("the trace needs the gpu-cuda feature");
}

fn main() {
    let f32_factor = std::env::args().any(|a| a == "--f32-factor");
    let mut s = scene(300, 30000);
    let (gpu, mark) = marks();
    let opts = SparseFaerOptions::default()
        .with_policy(SchurPolicy::Force)
        .with_mixed_precision(f32_factor.then(MixedPrecision::default))
        .with_gpu(Some(gpu));
    let mut solver = SparseFaer::<f64>::from_options(&opts);
    let cfg = LmConfig::<f64> { max_iters: 30, ..Default::default() }.with_observer(move |it: &LmIter<f64>| {
        mark(it.iter.min(4095));
        std::ops::ControlFlow::Continue(())
    });
    let r = s.solve_with(&mut solver, &cfg).unwrap();
    println!(
        "cost {:.4} -> {:.4}, {} iterations ({} accepted), {:?}, on the device: {:?}",
        r.start_cost, r.end_cost, r.iterations, r.accepted_iterations, r.status, r.gpu.as_ref().map(|g| &g.work)
    );
}
