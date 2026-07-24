// arael BAL runners: one generic model, instantiated by the f64 and
// f32 roots.
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
use arael::utils::Float;
use arael::vect::{vect2, vect3, vect3d};

#[arael::model]
#[derive(Clone)]
struct Camera<T: Float> {
    t: Param<vect3<T>>,
    ea: EulerAngleParam<T>, // world-to-camera
    intr: Param<vect3<T>>,  // (f, k1, k2)
    hb: SelfBlock<Camera<T>, T>,
}

#[arael::model]
#[derive(Clone)]
struct Point<T: Float> {
    pos: Param<vect3<T>>,
    hb: SelfBlock<Point<T>, T>,
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
#[derive(Clone)]
struct Obs<T: Float> {
    #[arael(ref = root.cameras)]
    cam: Ref<Camera<T>>,
    #[arael(ref = root.points)]
    pt: Ref<Point<T>>,
    xy: vect2<T>,
    hb: CrossBlock<Camera<T>, Point<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Scene {
    cameras: refs::Vec<Camera<f64>>,
    points: refs::Vec<Point<f64>>,
    observations: std::vec::Vec<Obs<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct SceneF {
    cameras: refs::Vec<Camera<f32>>,
    points: refs::Vec<Point<f32>>,
    observations: std::vec::Vec<Obs<f32>>,
}

// ---------------------------------------------------------------- runners

fn build_parts<T: Float>(ds: &Dataset)
    -> (refs::Vec<Camera<T>>, refs::Vec<Point<T>>, std::vec::Vec<Obs<T>>)
{
    let c3 = |v: vect3d| -> vect3<T> { v.cast() };
    let mut cameras = refs::Vec::new();
    for c in &ds.cameras {
        cameras.push(Camera {
            t: Param::new(c3(c.t)),
            ea: EulerAngleParam::new(c3(c.rot().get_euler_angles())),
            intr: Param::new(c3(vect3d::new(c.f, c.k1, c.k2))),
            hb: SelfBlock::new(),
        });
    }
    let mut points = refs::Vec::new();
    for p in &ds.points {
        points.push(Point { pos: Param::new(c3(*p)), hb: SelfBlock::new() });
    }
    let mut observations = std::vec::Vec::new();
    for o in &ds.observations {
        observations.push(Obs {
            cam: cameras.ref_at(o.cam),
            pt: points.ref_at(o.point),
            xy: o.xy.cast(),
            hb: CrossBlock::new(),
        });
    }
    (cameras, points, observations)
}

pub fn build_f64(ds: &Dataset) -> Scene {
    let (cameras, points, observations) = build_parts(ds);
    Scene { cameras, points, observations }
}

fn build_f32(ds: &Dataset) -> SceneF {
    let (cameras, points, observations) = build_parts(ds);
    SceneF { cameras, points, observations }
}

/// What the benchmark hands the pipeline: the problem, the damping it wants, and
/// WHICH linear solver this row is measuring. The route belongs here because it
/// is what distinguishes one arael row from another over the identical model --
/// the same cameras and points, factorized two different ways.
pub struct Problem {
    /// Shared, not copied: the four arael rows are the same dataset factorized
    /// four ways, and Ladybug-1723 is 156k points.
    pub ds: std::rc::Rc<Dataset>,
    /// Per-dataset initial damping (see the table in main.rs). ARAEL_LAMBDA0
    /// overrides it.
    pub lambda0: f64,
    pub route: Route,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Route {
    /// The whole camera+point system, factorized as one. No elimination hint:
    /// ordering the points first was measured to HURT on BAL, unlike slam.
    Sparse,
    /// The points marginalized on every damped solve, factorizing only the
    /// reduced camera system.
    Schur,
    /// CHOLMOD's supernodal factorization of the Schur-reduced system (GPL
    /// module; see the cholmod-gpl feature warning in the arael Cargo.toml).
    #[cfg(feature = "cholmod-gpl")]
    CholmodGpl,
}

impl Route {
    /// The row label this route runs under.
    pub fn label(self, precision: &str) -> String {
        let route = match self {
            Route::Sparse => "sparse",
            Route::Schur => "schur",
            #[cfg(feature = "cholmod-gpl")]
            Route::CholmodGpl => "cholmod-gpl",
        };
        format!("arael LM {} {}", precision, route)
    }
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

// The two arael linear-solver routes the benchmark compares.
//
// `sparse` factorizes the full camera+point system with faer (no
// elimination hint: ordering the points first was measured to HURT on
// BAL, unlike the slam benchmark). `schur` marginalizes the points on
// every damped solve and factorizes only the camera system. Which wins
// depends on the camera count -- see the README.
fn solve64(params: &[f64], s: &mut Scene, cfg: &arael::simple_lm::LmConfig<f64>) -> arael::simple_lm::LmResult<f64> {
    // The plain row: the whole system, no reduction. Without the policy the
    // backend would marginalize the points itself -- that is the other row.
    let mut solver = arael::simple_lm::SparseFaer::new()
        .with_policy(arael::simple_lm::SchurPolicy::Never);
    arael::simple_lm::lm_solve(params, &mut solver, s, cfg).unwrap()
}

fn solve64_schur(params: &[f64], s: &mut Scene, cfg: &arael::simple_lm::LmConfig<f64>) -> arael::simple_lm::LmResult<f64> {
    // No hint: the eliminable blocks are detected from the model's coupling
    // graph. Forced, so the benchmark measures the reduction itself rather
    // than the policy's verdict about it.
    // BAL_SCHUR_POLICY=auto gates the reduction on the fill analysis
    // instead of forcing it (see SchurPolicy).
    let policy = if std::env::var("BAL_SCHUR_POLICY").as_deref() == Ok("auto") {
        arael::simple_lm::SchurPolicy::default()
    } else {
        arael::simple_lm::SchurPolicy::Force
    };
    // Nested dissection on the reduced camera system: a 3D point makes a
    // clique of the cameras that see it, and AMD drowns in cliques. At
    // Ladybug-1723 it factorizes S in 1508 ms against AMD's 4730.
    // BAL_ORDERING=amd goes back to AMD for comparison.
    let ordering = if std::env::var("BAL_ORDERING").as_deref() == Ok("amd") {
        arael::simple_lm::FaerOrdering::Auto
    } else {
        arael::simple_lm::FaerOrdering::NestedDissection
    };
    let mut solver = arael::simple_lm::SparseFaer::new()
        .with_policy(policy)
        .with_ordering(ordering);
    let r = arael::simple_lm::lm_solve(params, &mut solver, s, cfg).unwrap();
    if std::env::var("BAL_SCHUR_PLAN").is_ok() {
        if let Some(p) = solver.plan() {
            eprintln!("schur plan: {:?}", p);
        }
    }
    r
}

fn solve32(params: &[f32], s: &mut SceneF, cfg: &arael::simple_lm::LmConfig<f32>) -> arael::simple_lm::LmResult<f32> {
    // The plain row: the whole system, no reduction. Without the policy the
    // backend would marginalize the points itself -- that is the other row.
    let mut solver = arael::simple_lm::SparseFaerF32::new()
        .with_policy(arael::simple_lm::SchurPolicy::Never);
    arael::simple_lm::lm_solve(params, &mut solver, s, cfg).unwrap()
}

fn solve32_schur(params: &[f32], s: &mut SceneF, cfg: &arael::simple_lm::LmConfig<f32>) -> arael::simple_lm::LmResult<f32> {
    let ordering = if std::env::var("BAL_ORDERING").as_deref() == Ok("amd") {
        arael::simple_lm::FaerOrdering::Auto
    } else {
        arael::simple_lm::FaerOrdering::NestedDissection
    };
    let mut solver = arael::simple_lm::SparseFaerF32::new()
        .with_policy(arael::simple_lm::SchurPolicy::Force)
        .with_ordering(ordering);
    arael::simple_lm::lm_solve(params, &mut solver, s, cfg).unwrap()
}

fn rodrigues_of(m: arael::matrix::matrix3d) -> vect3d {
    let (axis, angle) = quaternd::from_rotation_matrix(m).get_axis_angle();
    axis * angle
}

/// One system's answer: the cameras and the points. Scored by
/// [`crate::bal::reference_cost`], and compared to the best solution by
/// similarity-aligned camera centres -- bundle adjustment has a 7-DOF gauge, so
/// absolute positions are not comparable across systems.
#[derive(Clone)]
pub struct Solution {
    pub cameras: Vec<CameraIn>,
    pub points: Vec<vect3d>,
}

pub type RunOut = bench_harness::table::Row<Solution>;

impl bench_harness::arael::Model for Scene {
    type Scalar = f64;
    type Input = Problem;
    type Solution = Solution;
    // Bundle adjustment is far less linear than a pose graph, and the fixed
    // ladder walks into damping spirals on it (Ladybug-138 gauge spiral). The
    // gain-ratio driver does not.
    const NIELSEN: bool = true;
    fn lambda0(p: &Problem) -> f64 { p.lambda0 }
    fn build(p: &Problem) -> Self { build_f64(&p.ds) }
    fn serialize(&mut self, out: &mut Vec<f64>) { self.serialize64(out); }
    fn deserialize(&mut self, x: &[f64]) { self.deserialize64(x); }
    fn tune(cfg: &mut arael::simple_lm::LmConfig<f64>) { cfg.lambda_floor = lambda_floor(); }
    fn solution(&self) -> Solution {
        Solution {
            cameras: self.cameras.iter()
                .map(|c| CameraIn {
                    rodrigues: rodrigues_of(
                        arael::matrix::matrix3d::rotation_from_euler_angles(c.ea.value)),
                    t: c.t.value,
                    f: c.intr.value.x,
                    k1: c.intr.value.y,
                    k2: c.intr.value.z,
                })
                .collect(),
            points: self.points.iter().map(|p| p.pos.value).collect(),
        }
    }
    fn solve(p: &Problem, params: &[f64], m: &mut Self,
             cfg: &arael::simple_lm::LmConfig<f64>) -> arael::simple_lm::LmResult<f64> {
        match p.route {
            Route::Sparse => solve64(params, m, cfg),
            Route::Schur => solve64_schur(params, m, cfg),
            #[cfg(feature = "cholmod-gpl")]
            Route::CholmodGpl =>
                arael::simple_lm::solve_sparse_cholmod_supernodal(params, m, cfg),
        }
    }
}

/// One covariance-scaling run: `(N, median_ms, reps)` per query count.
pub struct CovScaling {
    pub perquery_cam: Vec<(usize, f64, usize)>,
    pub perquery_point: Vec<(usize, f64, usize)>,
    pub allmarg_ms: f64,
    pub allmarg_reps: usize,
    pub sd_cam2: [f64; 6],
}

// Solve the gauge-fixed problem, then time covariance recovery as the query count
// scales. Known calibration: intrinsics (f,k1,k2) held constant, so camera
// marginals are 6-DOF poses; points are 3-DOF. PerQuery times the full cold cost
// (assemble H + factor + query N marginals) each rep; AllMarginals is the bulk
// selected inverse over the whole factor (every camera and point at once).
pub fn cov_bench(problem: &Problem) -> CovScaling {
    use arael::covariance::{CovMode, Covariance};
    use bench_harness::cov::{query_counts, scale_counts, spread};
    use bench_harness::probe::median_ms;
    use std::hint::black_box;
    use std::time::Duration;

    let mut scene = build_f64(&problem.ds);
    // Known calibration: intrinsics are near-unconstrained when a camera's points
    // cluster near the image center, so holding them constant keeps H positive
    // definite and recovers 6-DOF pose covariance.
    for c in &mut scene.cameras {
        c.intr.optimize = false;
    }
    // Gauge fix (BAL is a similarity): hold cameras 0 and 1 fully constant.
    scene.cameras[0].t.optimize = false;
    scene.cameras[0].ea.optimize = false;
    if scene.cameras.len() > 1 {
        scene.cameras[1].t.optimize = false;
        scene.cameras[1].ea.optimize = false;
    }
    let mut params: Vec<f64> = Vec::new();
    scene.serialize64(&mut params);
    let cfg = bench_harness::arael::config::<Scene>(problem, 100);
    let result = solve64_schur(&params, &mut scene, &cfg);
    scene.deserialize64(&result.x);

    let (ncam, npt) = (scene.cameras.len(), scene.points.len());
    let free = ncam - 2; // cameras 0 and 1 are the fixed gauge
    let budget = Duration::from_secs_f64(
        std::env::var("COV_BUDGET_S").ok().and_then(|v| v.parse().ok()).unwrap_or(5.0));
    let cap: usize = std::env::var("COV_CAP").ok().and_then(|v| v.parse().ok()).unwrap_or(200);
    let cap_s = bench_harness::cov::cell_cap_s();

    // Validation: camera 2's 6-DOF pose std dev (translation, then rotation).
    let sd_cam2 = {
        let cov = scene.assemble_covariance(CovMode::PerQuery).expect("gauge-fixed H is PD");
        let m = cov.marginal_cov(&scene.cameras[2]).unwrap();
        std::array::from_fn(|d| m[(d, d)].sqrt())
    };

    // PerQuery camera poses: 1, 2, 8, 32, all.
    let perquery_cam = scale_counts(query_counts(free, true), cap_s, |n| {
        let idx = spread(2, free, n);
        median_ms(budget, cap, || {
            let cov = scene.assemble_covariance(CovMode::PerQuery).unwrap();
            for &i in &idx {
                black_box(cov.marginal_cov(&scene.cameras[i]).unwrap());
            }
        })
    });

    // PerQuery points: 1, 2, 8, 32, all -- "all" via per-query usually hits the
    // cap (that is AllMarginals' job), which the table shows as `*`.
    let perquery_point = scale_counts(query_counts(npt, true), cap_s, |n| {
        let idx = spread(0, npt, n);
        median_ms(budget, cap, || {
            let cov = scene.assemble_covariance(CovMode::PerQuery).unwrap();
            for &i in &idx {
                black_box(cov.marginal_cov(&scene.points[i]).unwrap());
            }
        })
    });

    // AllMarginals: bulk selected inverse -- every camera and point at once.
    let (allmarg_ms, allmarg_reps) = median_ms(budget, cap, || {
        black_box(scene.assemble_covariance(CovMode::AllMarginals).unwrap());
    });

    CovScaling { perquery_cam, perquery_point, allmarg_ms, allmarg_reps, sd_cam2 }
}

impl bench_harness::arael::Model for SceneF {
    type Scalar = f32;
    type Input = Problem;
    type Solution = Solution;
    const NIELSEN: bool = true;
    fn lambda0(p: &Problem) -> f64 { p.lambda0 }
    fn build(p: &Problem) -> Self { build_f32(&p.ds) }
    fn serialize(&mut self, out: &mut Vec<f32>) { self.serialize32(out); }
    fn deserialize(&mut self, x: &[f32]) { self.deserialize32(x); }
    fn tune(cfg: &mut arael::simple_lm::LmConfig<f32>) { cfg.lambda_floor = lambda_floor() as f32; }
    fn solution(&self) -> Solution {
        Solution {
            cameras: self.cameras.iter()
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
                .collect(),
            points: self.points.iter().map(|p| vect3d::from(p.pos.value)).collect(),
        }
    }
    fn solve(p: &Problem, params: &[f32], m: &mut Self,
             cfg: &arael::simple_lm::LmConfig<f32>) -> arael::simple_lm::LmResult<f32> {
        match p.route {
            Route::Sparse => solve32(params, m, cfg),
            Route::Schur => solve32_schur(params, m, cfg),
            // CHOLMOD's supernodal module is double-precision only.
            #[cfg(feature = "cholmod-gpl")]
            Route::CholmodGpl => unreachable!("cholmod-gpl is an f64-only row"),
        }
    }
}

pub fn run_f64(p: &Problem) -> RunOut { bench_harness::arael::run::<Scene>(p) }
pub fn run_f32(p: &Problem) -> RunOut { bench_harness::arael::run::<SceneF>(p) }

// Capped single solves (no probes) -- the peak-memory pass, which runs one
// system alone in a process of its own. The peak fill-in is reached in the first
// factorization, so a few iterations measure the same high-water mark.
pub fn run_f64_capped(p: &Problem, max_iters: usize) -> Solution {
    solve_capped::<Scene>(p, max_iters)
}

pub fn run_f32_capped(p: &Problem, max_iters: usize) -> Solution {
    solve_capped::<SceneF>(p, max_iters)
}

fn solve_capped<M: bench_harness::arael::Model<Input = Problem, Solution = Solution>>(
    p: &Problem, max_iters: usize) -> Solution {
    probe_capped::<M>(p, max_iters).solution
}

/// What one capped solve did, for the damping probe: no warmup, no sub-rounds,
/// no full solve. The point is to see whether the first iterations are CLEAN --
/// one attempt each, accepted -- because a rejected step there is what denies the
/// benchmark its per-iteration number.
pub struct Probe {
    pub ms: f64,
    pub accepted: usize,
    pub attempts: usize,
    pub solution: Solution,
}

pub fn probe_f64(p: &Problem, max_iters: usize) -> Probe {
    probe_capped::<Scene>(p, max_iters)
}

pub fn probe_f32(p: &Problem, max_iters: usize) -> Probe {
    probe_capped::<SceneF>(p, max_iters)
}

fn probe_capped<M: bench_harness::arael::Model<Input = Problem, Solution = Solution>>(
    p: &Problem, max_iters: usize) -> Probe {
    let mut model = M::build(p);
    let mut params: Vec<M::Scalar> = Vec::new();
    model.serialize(&mut params);
    let cfg = bench_harness::arael::config::<M>(p, max_iters);
    // The build is the reset, not the solve -- the clock starts here, as it does
    // everywhere else in the harness.
    let (ms, result) = bench_harness::solver::timed(|| {
        <M as bench_harness::arael::Model>::solve(p, &params, &mut model, &cfg)
    });
    model.deserialize(&result.x);
    Probe {
        ms,
        accepted: result.accepted_iterations,
        attempts: result.iterations,
        solution: model.solution(),
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
