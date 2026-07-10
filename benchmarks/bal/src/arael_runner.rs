// arael BAL runners: identical model in f64 and f32.
//
// Camera: 9 parameters -- world-to-camera rotation as an
// EulerAngleParam (delta composed with a reference, re-centered per
// accepted step; initialized from the file's Rodrigues vector),
// translation, and (f, k1, k2) as one vect3 Param. Point: 3 parameters.
// The constraint body is the Snavely reprojection residual (see
// bal.rs). No gauge prior -- BAL problems are benchmarked with the
// 7-DOF gauge left to LM damping, as Ceres runs them.

use crate::bal::{CameraIn, Dataset};
use arael::model::{CrossBlock, EulerAngleParam, Param, SelfBlock};
use arael::quatern::{quaternd, quaternf};
use arael::refs::{self, Ref};
use arael::vect::{vect2d, vect2f, vect3d, vect3f};

// ---------------------------------------------------------------- f64

#[arael::model]
struct Camera {
    t: Param<vect3d>,
    ea: EulerAngleParam<f64>, // world-to-camera
    intr: Param<vect3d>,      // (f, k1, k2)
    hb: SelfBlock<Camera>,
}

#[arael::model]
struct Point {
    pos: Param<vect3d>,
    hb: SelfBlock<Point>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [cam.intr.x * d * px - obs.xy.x,
     cam.intr.x * d * py - obs.xy.y]
}))]
struct Obs {
    #[arael(ref = root.cameras)]
    cam: Ref<Camera>,
    #[arael(ref = root.points)]
    pt: Ref<Point>,
    xy: vect2d,
    hb: CrossBlock<Camera, Point>,
}

#[arael::model]
#[arael(root)]
struct Scene {
    cameras: refs::Vec<Camera>,
    points: refs::Vec<Point>,
    observations: std::vec::Vec<Obs>,
}

// ---------------------------------------------------------------- f32

#[arael::model]
struct CameraF {
    t: Param<vect3f>,
    ea: EulerAngleParam<f32>,
    intr: Param<vect3f>,
    hb: SelfBlock<CameraF, f32>,
}

#[arael::model]
struct PointF {
    pos: Param<vect3f>,
    hb: SelfBlock<PointF, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let pc = cam.ea.rotation_matrix() * pt.pos + cam.t;
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.intr.y + cam.intr.z * r2);
    [cam.intr.x * d * px - obsf.xy.x,
     cam.intr.x * d * py - obsf.xy.y]
}))]
struct ObsF {
    #[arael(ref = root.cameras)]
    cam: Ref<CameraF>,
    #[arael(ref = root.points)]
    pt: Ref<PointF>,
    xy: vect2f,
    hb: CrossBlock<CameraF, PointF, f32>,
}

#[arael::model]
#[arael(root, f32)]
struct SceneF {
    cameras: refs::Vec<CameraF>,
    points: refs::Vec<PointF>,
    observations: std::vec::Vec<ObsF>,
}

// ---------------------------------------------------------------- runners

fn build_f64(ds: &Dataset) -> Scene {
    let mut s = Scene {
        cameras: refs::Vec::new(),
        points: refs::Vec::new(),
        observations: std::vec::Vec::new(),
    };
    for c in &ds.cameras {
        s.cameras.push(Camera {
            t: Param::new(c.t),
            ea: EulerAngleParam::new(c.rot().get_euler_angles()),
            intr: Param::new(vect3d::new(c.f, c.k1, c.k2)),
            hb: SelfBlock::new(),
        });
    }
    for p in &ds.points {
        s.points.push(Point { pos: Param::new(*p), hb: SelfBlock::new() });
    }
    for o in &ds.observations {
        let cam = s.cameras.ref_at(o.cam);
        let pt = s.points.ref_at(o.point);
        s.observations.push(Obs {
            cam,
            pt,
            xy: o.xy,
            hb: CrossBlock::new(),
        });
    }
    s
}

fn build_f32(ds: &Dataset) -> SceneF {
    let mut s = SceneF {
        cameras: refs::Vec::new(),
        points: refs::Vec::new(),
        observations: std::vec::Vec::new(),
    };
    for c in &ds.cameras {
        s.cameras.push(CameraF {
            t: Param::new(vect3f::from(c.t)),
            ea: EulerAngleParam::new(vect3f::from(c.rot().get_euler_angles())),
            intr: Param::new(vect3f::new(c.f as f32, c.k1 as f32, c.k2 as f32)),
            hb: SelfBlock::new(),
        });
    }
    for p in &ds.points {
        s.points.push(PointF { pos: Param::new(vect3f::from(*p)), hb: SelfBlock::new() });
    }
    for o in &ds.observations {
        let cam = s.cameras.ref_at(o.cam);
        let pt = s.points.ref_at(o.point);
        s.observations.push(ObsF {
            cam,
            pt,
            xy: vect2f::from(o.xy),
            hb: CrossBlock::new(),
        });
    }
    s
}

pub struct RunOut {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    pub iterations: usize,
    pub accepted: usize,
    /// Cost of one FULL accepted iteration -- the solver's steady-state
    /// per-phase means (assembly + damped solve + trial cost + advance,
    /// first calls excluded: they carry one-time structure costs) summed.
    /// Undiluted by rejected attempts, which skip the re-linearization.
    /// 0.0 when a phase has no steady-state sample (fewer than 2 calls).
    pub full_iter_ms: f64,
    pub cameras: Vec<CameraIn>,
    pub points: Vec<vect3d>,
}

fn full_iter_ms(timing: Option<&arael::simple_lm::LmTiming>) -> f64 {
    let Some(t) = timing else { return 0.0 };
    match (t.mean_assembly(), t.mean_linear_solve(), t.mean_cost_eval(), t.mean_advance()) {
        (Some(a), Some(l), Some(c), Some(adv)) => (a + l + c + adv).as_secs_f64() * 1e3,
        _ => 0.0,
    }
}

// Initial damping: bundle adjustment is far less linear than the pose
// graphs, so the pose-graph values (1e-8 .. 1e-10) are not appropriate;
// 1e-4 is in the range trust-region defaults were tuned for on BAL.
// Env-overridable for experiments.
fn lambda0() -> f64 {
    std::env::var("ARAEL_LAMBDA0").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-4)
}

// Damping floor (env ARAEL_LAMBDA_FLOOR, library default 1e-12). Under
// the fixed schedule bundle adjustment needed a raised floor against
// gauge-driven Cholesky failure spirals; the Nielsen driver's gain
// ratio never marches lambda into that regime, so the floor is moot
// (measured identical at 1e-12 and 1e-6) and stays at the default.
fn lambda_floor() -> f64 {
    std::env::var("ARAEL_LAMBDA_FLOOR").ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-12)
}

// ARAEL_VERBOSE=1 prints the solver's per-iteration trace (cost,
// lambda, accepted/rejected) to stderr.
fn verbose() -> bool {
    std::env::var("ARAEL_VERBOSE").map_or(false, |v| v == "1")
}

// The benchmark defaults to the gain-ratio NielsenLambdaDriver -- on
// bundle adjustment it eliminates the fixed schedule's damping
// rejections and the Ladybug-138 gauge spiral outright.
// ARAEL_DRIVER=fixed selects the classic fixed-multiplier schedule.
fn nielsen() -> bool {
    std::env::var("ARAEL_DRIVER").map_or(true, |v| v != "fixed")
}

fn solve64(params: &[f64], s: &mut Scene, cfg: &arael::simple_lm::LmConfig<f64>) -> arael::simple_lm::LmResult<f64> {
    arael::simple_lm::solve_sparse_faer(params, s, cfg)
}

fn solve32(params: &[f32], s: &mut SceneF, cfg: &arael::simple_lm::LmConfig<f32>) -> arael::simple_lm::LmResult<f32> {
    arael::simple_lm::solve_sparse_faer_f32(params, s, cfg)
}

fn cfg64(max_iters: usize) -> arael::simple_lm::LmConfig<f64> {
    let cfg = arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        max_iters,
        initial_lambda: lambda0(),
        lambda_floor: lambda_floor(),
        verbose: verbose(),
        gather_timing: true, // per-phase times for the full-iteration column
        ..Default::default()
    };
    if nielsen() { cfg.with_driver(arael::simple_lm::NielsenLambdaDriver::default()) } else { cfg }
}

fn cfg32(max_iters: usize) -> arael::simple_lm::LmConfig<f32> {
    let cfg = arael::simple_lm::LmConfig {
        abs_precision: 1e-5,
        rel_precision: 1e-5,
        patience: 1,
        max_iters,
        initial_lambda: lambda0() as f32,
        lambda_floor: lambda_floor() as f32,
        verbose: verbose(),
        gather_timing: true,
        ..Default::default()
    };
    if nielsen() { cfg.with_driver(arael::simple_lm::NielsenLambdaDriver::default()) } else { cfg }
}

fn rodrigues_of(m: arael::matrix::matrix3d) -> vect3d {
    let (axis, angle) = quaternd::from_rotation_matrix(m).get_axis_angle();
    axis * angle
}

type Solve64 = fn(&[f64], &mut Scene, &arael::simple_lm::LmConfig<f64>)
    -> arael::simple_lm::LmResult<f64>;

fn run_f64_with(ds: &Dataset, solve: Solve64) -> RunOut {
    let mut s = build_f64(ds);
    let mut params: Vec<f64> = Vec::new();
    s.serialize64(&mut params);

    let t0 = std::time::Instant::now();
    let _ = solve(&params, &mut s, &cfg64(1));
    let first_iter_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = std::time::Instant::now();
    let result = solve(&params, &mut s, &cfg64(100));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    s.deserialize64(&result.x);
    let cameras = s.cameras.iter()
        .map(|c| CameraIn {
            rodrigues: rodrigues_of(arael::matrix::matrix3d::rotation_from_euler_angles(c.ea.value)),
            t: c.t.value,
            f: c.intr.value.x,
            k1: c.intr.value.y,
            k2: c.intr.value.z,
        })
        .collect();
    let points = s.points.iter().map(|p| p.pos.value).collect();
    RunOut {
        solve_ms, first_iter_ms,
        iterations: result.iterations,
        accepted: result.accepted_iterations,
        full_iter_ms: full_iter_ms(result.timing.as_ref()),
        cameras, points,
    }
}

pub fn run_f64(ds: &Dataset) -> RunOut {
    run_f64_with(ds, solve64)
}

/// The CHOLMOD-supernodal row (GPL-licensed module; see the cholmod-gpl
/// feature warning in the arael Cargo.toml).
#[cfg(feature = "cholmod-gpl")]
pub fn run_f64_supernodal(ds: &Dataset) -> RunOut {
    run_f64_with(ds, |p, s, cfg| arael::simple_lm::solve_sparse_cholmod_supernodal(p, s, cfg))
}

// Single capped solves for the peak-memory measurement (no warm-up pass;
// peak fill-in is reached in the first factorization).

pub fn run_f64_capped(ds: &Dataset, max_iters: usize) -> Vec<f64> {
    let mut s = build_f64(ds);
    let mut params: Vec<f64> = Vec::new();
    s.serialize64(&mut params);
    solve64(&params, &mut s, &cfg64(max_iters)).x
}

#[cfg(feature = "cholmod-gpl")]
pub fn run_f64_supernodal_capped(ds: &Dataset, max_iters: usize) -> Vec<f64> {
    let mut s = build_f64(ds);
    let mut params: Vec<f64> = Vec::new();
    s.serialize64(&mut params);
    arael::simple_lm::solve_sparse_cholmod_supernodal(&params, &mut s, &cfg64(max_iters)).x
}

pub fn run_f32_capped(ds: &Dataset, max_iters: usize) -> Vec<f32> {
    let mut s = build_f32(ds);
    let mut params: Vec<f32> = Vec::new();
    s.serialize32(&mut params);
    solve32(&params, &mut s, &cfg32(max_iters)).x
}

pub fn run_f32(ds: &Dataset) -> RunOut {
    let mut s = build_f32(ds);
    let mut params: Vec<f32> = Vec::new();
    s.serialize32(&mut params);

    let t0 = std::time::Instant::now();
    let _ = solve32(&params, &mut s, &cfg32(1));
    let first_iter_ms = t0.elapsed().as_secs_f64() * 1e3;

    let t0 = std::time::Instant::now();
    let result = solve32(&params, &mut s, &cfg32(100));
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;
    s.deserialize32(&result.x);
    let cameras = s.cameras.iter()
        .map(|c| {
            let m = arael::matrix::matrix3f::rotation_from_euler_angles(c.ea.value);
            let (axis, angle) = quaternf::from_rotation_matrix(m).get_axis_angle();
            CameraIn {
                rodrigues: vect3d::from(axis * angle),
                t: vect3d::from(c.t.value),
                f: c.intr.value.x as f64,
                k1: c.intr.value.y as f64,
                k2: c.intr.value.z as f64,
            }
        })
        .collect();
    let points = s.points.iter().map(|p| vect3d::from(p.pos.value)).collect();
    RunOut {
        solve_ms, first_iter_ms,
        iterations: result.iterations,
        accepted: result.accepted_iterations,
        full_iter_ms: full_iter_ms(result.timing.as_ref()),
        cameras, points,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bal::reference_cost;

    // The arael model's cost must equal the reference cost on the real
    // vendored dataset -- this pins the symbolic residual (rotation
    // parameterization, negative-z convention, distortion, intrinsics
    // layout) to the one function every system is judged by.
    #[test]
    fn arael_cost_matches_reference() {
        use arael::simple_lm::LmProblem;
        let ds = crate::bal::load("datasets/problem-49-7776-pre.txt");
        let reference = reference_cost(&ds, &ds.cameras, &ds.points);
        assert!(reference > 1.0, "unexpectedly small initial cost: {}", reference);

        let mut s = build_f64(&ds);
        let mut params: Vec<f64> = Vec::new();
        s.serialize64(&mut params);
        let arael_cost = s.calc_cost(&params);
        assert!(((arael_cost - reference) / reference).abs() < 1e-9,
            "arael {} vs reference {}", arael_cost, reference);
    }
}
