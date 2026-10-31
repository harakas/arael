//! The cost and the floor of the `with_wide` scope's two wide scalars on
//! the BAL models, on the CPU: the f32 model with its scope in f64 against
//! the same model with its scope in `df32` (two f32s), both against the
//! f64 model. At the dataset's initial point, so no solve is needed:
//!
//! 1. the residuals of the two f32 models against the f64 model's (the
//!    f64 model's observations rounded to f32 so the three stand at one
//!    point) and against each other;
//! 2. the time of `calc_cost` (residual values only, where the scope is
//!    most of the work) and of `calc_grad_hessian_sparse` (the whole
//!    assembly) for the three models, the median of several runs.
//!
//! The macro takes one type parameter per entity, so the df32 model is a
//! concrete copy of `arael_runner`'s three structs with `wide = df32`.
//!
//! Usage: `wide_cost <dataset> [repeats]`

#[path = "../bal.rs"]
mod bal;
#[path = "../arael_runner.rs"]
mod arael_runner;

use std::time::Instant;

use arael::df32;
use arael::model::{CrossBlock, JacobianModel, MixedEulerAngleParam, Param, SelfBlock};
use arael::refs::{self, Ref};
use arael::simple_lm::{CooMatrix, LmProblem, RootProblem};
use arael::vect::{vect2, vect3, vect3d};

use arael_runner::{Camera, Obs, Point};
use bal::Dataset;

// The f64 model and the f32 model with the f64 scope, with Jacobians.

#[arael::model]
#[arael(root, jacobian)]
struct SceneJ {
    cameras: refs::Vec<Camera<f64>>,
    points: refs::Vec<Point<f64>>,
    observations: std::vec::Vec<Obs<f64>>,
}

#[arael::model]
#[arael(root, f32, jacobian)]
struct SceneFJ {
    cameras: refs::Vec<Camera<f32>>,
    points: refs::Vec<Point<f32>>,
    observations: std::vec::Vec<Obs<f32>>,
}

// The f32 model with the df32 scope: arael_runner's structs, concrete.

#[arael::model]
pub struct CameraD {
    pub t: Param<vect3<f32>>,
    pub ea: MixedEulerAngleParam<f32, df32>,
    pub intr: Param<vect3<f32>>,
    pub zdrop: f32,
    pub hb: SelfBlock<CameraD, f32>,
}

#[arael::model]
pub struct PointD {
    pub pos: Param<vect3<f32>>,
    pub hb: SelfBlock<PointD, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let keep = -pc.z - cam.zdrop;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [branch(keep, with_wide(cam.intr.x * d * px - obsd.xy.x), 0.0),
     branch(keep, with_wide(cam.intr.x * d * py - obsd.xy.y), 0.0)]
}))]
pub struct ObsD {
    #[arael(ref = root.cameras)]
    pub cam: Ref<CameraD>,
    #[arael(ref = root.points)]
    pub pt: Ref<PointD>,
    pub xy: vect2<f32>,
    pub hb: CrossBlock<CameraD, PointD, f32>,
}

#[arael::model]
#[arael(root, f32, jacobian, wide = df32)]
struct SceneFJD {
    cameras: refs::Vec<CameraD>,
    points: refs::Vec<PointD>,
    observations: std::vec::Vec<ObsD>,
}

fn build_d(ds: &Dataset) -> SceneFJD {
    let c3 = |v: vect3d| -> vect3<f32> { v.cast() };
    let zdrop = bal::guard_z().unwrap_or(-1e30) as f32;
    let mut cameras = refs::Vec::new();
    for c in &ds.cameras {
        let ea = c.rot().get_euler_angles();
        cameras.push(CameraD {
            t: Param::new(c3(c.t)),
            ea: MixedEulerAngleParam::new(vect3::<df32>::new(
                df32::from_f64(ea.x), df32::from_f64(ea.y), df32::from_f64(ea.z))),
            intr: Param::new(c3(vect3d::new(c.f, c.k1, c.k2))),
            zdrop,
            hb: SelfBlock::new(),
        });
    }
    let mut points = refs::Vec::new();
    for p in &ds.points {
        points.push(PointD { pos: Param::new(c3(*p)), hb: SelfBlock::new() });
    }
    let mut observations = std::vec::Vec::new();
    for o in &ds.observations {
        observations.push(ObsD {
            cam: cameras.ref_at(o.cam),
            pt: points.ref_at(o.point),
            xy: o.xy.cast(),
            hb: CrossBlock::new(),
        });
    }
    SceneFJD { cameras, points, observations }
}

/// The residuals at the model's initial point.
fn residuals<M, T>(m: &mut M) -> Vec<f64>
where
    M: RootProblem<T> + JacobianModel<T>,
    T: arael::utils::Float,
{
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    m.calc_jacobian(&x).rows.iter().map(|r| r.residual.to_f64().unwrap()).collect()
}

fn rms(d: &[f64]) -> f64 {
    (d.iter().map(|v| v * v).sum::<f64>() / d.len() as f64).sqrt()
}

/// The median wall time of `f` over `repeats` runs, in milliseconds.
fn median_ms(repeats: usize, mut f: impl FnMut()) -> f64 {
    let mut times: Vec<f64> = (0..repeats).map(|_| {
        let t = Instant::now();
        f();
        t.elapsed().as_secs_f64() * 1e3
    }).collect();
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    times[times.len() / 2]
}

/// `calc_cost` and `calc_grad_hessian_sparse` times of one model.
fn timings<M, T>(m: &mut M, repeats: usize) -> (f64, f64)
where
    M: RootProblem<T> + LmProblem<T>,
    T: arael::utils::Float,
{
    let mut x = Vec::new();
    RootProblem::serialize(m, &mut x);
    let n = x.len();
    let cost = median_ms(repeats, || { m.calc_cost(&x); });
    let mut g = vec![T::zero(); n];
    let mut coo = CooMatrix::new(n);
    let gh = median_ms(repeats, || { m.calc_grad_hessian_sparse(&x, &mut g, &mut coo); });
    (cost, gh)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: wide_cost <dataset> [repeats]");
        std::process::exit(2);
    }
    let ds = bal::load(&args[1]);
    let repeats: usize = args.get(2).map_or(7, |v| v.parse().expect("repeats"));
    println!("== {}: {} cameras, {} points, {} observations; {} repeats",
        args[1], ds.cameras.len(), ds.points.len(), ds.observations.len(), repeats);

    // The f64 model at the f32 models' point: observations and parameters
    // rounded to f32 (the rotations are wide in every model).
    let (cams, pts, mut obs) = arael_runner::build_parts::<f64>(&ds);
    for o in obs.iter_mut() { o.xy = o.xy.cast::<f32>().cast(); }
    let mut s64 = SceneJ { cameras: cams, points: pts, observations: obs };
    let mut x = Vec::new();
    RootProblem::serialize(&mut s64, &mut x);
    let x32: Vec<f64> = x.iter().map(|v| *v as f32 as f64).collect();
    RootProblem::deserialize(&mut s64, &x32);
    let (cams, pts, obs) = arael_runner::build_parts::<f32>(&ds);
    let mut s32 = SceneFJ { cameras: cams, points: pts, observations: obs };
    let mut s32d = build_d(&ds);

    println!("-- residuals at the initial point");
    let r64 = residuals(&mut s64);
    let r32 = residuals(&mut s32);
    let r32d = residuals(&mut s32d);
    assert_eq!(r64.len(), r32.len());
    assert_eq!(r64.len(), r32d.len());
    let d32: Vec<f64> = r32.iter().zip(&r64).map(|(a, b)| a - b).collect();
    let d32d: Vec<f64> = r32d.iter().zip(&r64).map(|(a, b)| a - b).collect();
    let dd: Vec<f64> = r32d.iter().zip(&r32).map(|(a, b)| a - b).collect();
    let differing = dd.iter().filter(|v| **v != 0.0).count();
    println!("  rms residual {:.4} px", rms(&r64));
    println!("  f32 model, f64 scope,  against f64: rms {:.3e} px, max {:.3e} px",
        rms(&d32), d32.iter().fold(0.0f64, |m, v| m.max(v.abs())));
    println!("  f32 model, df32 scope, against f64: rms {:.3e} px, max {:.3e} px",
        rms(&d32d), d32d.iter().fold(0.0f64, |m, v| m.max(v.abs())));
    println!("  df32 scope against f64 scope:        rms {:.3e} px, max {:.3e} px, {} of {} residuals differ",
        rms(&dd), dd.iter().fold(0.0f64, |m, v| m.max(v.abs())), differing, dd.len());

    println!("-- time, median of {} (ms): cost | gradient and Hessian (coo)", repeats);
    let (c64, g64) = timings(&mut s64, repeats);
    let (c32, g32) = timings(&mut s32, repeats);
    let (c32d, g32d) = timings(&mut s32d, repeats);
    println!("  f64 model:              {:8.2} | {:8.2}", c64, g64);
    println!("  f32 model, f64 scope:   {:8.2} | {:8.2}", c32, g32);
    println!("  f32 model, df32 scope:  {:8.2} | {:8.2}   ({:+.1}% | {:+.1}% against the f64 scope)",
        c32d, g32d, (c32d / c32 - 1.0) * 100.0, (g32d / g32 - 1.0) * 100.0);
}
