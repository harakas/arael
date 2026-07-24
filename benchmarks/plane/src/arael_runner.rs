// The arael runner. Poses use arael's builtin TransformParam (translation
// and rotation optimized together, stepping in twist coordinates) and plane normals its
// UnitVecParam (the S^2 direction component);
// examples/plane_slam_demo.rs carries the same model with the direction
// component spelled out as a user-defined #[arael(component)] -- the two
// are equivalent by construction and by test
// (tests/unitvec_param.rs::builtin_matches_the_macro_component).
//
// The entities are generic over the scalar; the two concrete roots
// (`World` f64, `WorldF` f32) instantiate one shared model.

use arael::model::{CrossBlock, Param, SelfBlock};
use arael::refs::Ref;
use arael::transform::TransformParam;
use arael::unitvec::UnitVecParam;
use arael::simple_lm::{lm_solve, LmConfig, LmResult, SparseFaer};
use arael::matrix::matrix3;
use arael::utils::Float;
use arael::vect::vect3;

use crate::scene::{Plane, Pose, RawScene, Solution};

#[arael::model]
#[derive(Clone)]
struct PoseV<T: Float> {
    /// The pose: reference (robot) frame into world.
    r2w: TransformParam<T>,
    /// This pose's Hessian tile (gradient + diagonal block of J^T J).
    hb: SelfBlock<PoseV<T>, T>,
}

#[arael::model]
#[derive(Clone)]
struct PlaneLm<T: Float> {
    /// Unit normal of the plane (2-DOF component).
    n: UnitVecParam<T>,
    /// Distance coefficient: the plane is n.x + c = 0, distance = -c.
    c: Param<T>,
    /// This plane's Hessian tile.
    hb: SelfBlock<PlaneLm<T>, T>,
}

// Odometry between-residual, identical to the g2o runner's custom edge.
// Translation: err_t = R_a^T (b.r2w.translation - a.r2w.translation) - measured_translation.
// Rotation: the error rotation dR = R_m^T R_a^T R_b (measured relative
// rotation inverted, composed with the estimated one) should be identity;
// the residual is its skew part read as a vector,
//   err_r = vee((dR - dR^T)/2) = sin(angle) * axis,
// zero exactly when measurement and estimate agree. vee() maps a
// skew-symmetric matrix to its 3-vector: vee(M) = (M[2][1], M[0][2],
// M[1][0]) -- the c1/c2/c3 column arithmetic in the body.
#[arael::model]
#[arael(constraint(hb, {
    let ra = a.r2w.rotation_matrix;
    let rb = b.r2w.rotation_matrix;
    let dt = ra.transpose() * (b.r2w.translation - a.r2w.translation) - odov.measured_translation;
    let dr = odov.measured_rotation_transposed * (ra.transpose() * rb);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x * odov.translation_weight, dt.y * odov.translation_weight, dt.z * odov.translation_weight,
     (c2.z - c3.y) * 0.5 * odov.rotation_weight,
     (c3.x - c1.z) * 0.5 * odov.rotation_weight,
     (c1.y - c2.x) * 0.5 * odov.rotation_weight]
}, parent = odov))]
#[derive(Clone)]
struct Odov<T: Float> {
    /// The earlier pose: the measurement is expressed in ITS frame.
    #[arael(ref = root.poses)]
    a: Ref<PoseV<T>>,
    /// The later pose the measurement leads to.
    #[arael(ref = root.poses)]
    b: Ref<PoseV<T>>,
    /// Measured relative translation: where odometry says `b` sits in
    /// `a`'s frame; compared against R_a^T (b.r2w.translation - a.r2w.translation).
    measured_translation: vect3<T>,
    /// TRANSPOSE of the measured relative rotation, R_m^T -- stored
    /// pre-transposed because the residual only ever uses it that way
    /// (dR = R_m^T R_a^T R_b).
    measured_rotation_transposed: matrix3<T>,
    /// Whitening weight (1/sigma, per axis) of the translation residual.
    translation_weight: T,
    /// Whitening weight (1/sigma, per axis) of the rotation residual.
    rotation_weight: T,
    /// The a-b coupling tile of J^T J this constraint accumulates into
    /// (named as the primary block in the constraint attribute above).
    hb: CrossBlock<PoseV<T>, PoseV<T>, T>,
}

// Plane observation: g2o's EdgeSE3PlaneSensorCalib error (Plane3D::ominus),
// written algebraically. Predicted local plane (n_l, c_l) from the world
// plane through the pose; error = (azimuth, elevation) of the measured
// normal in the frame aligning n_l with e1, plus the distance difference.
#[arael::model]
#[arael(constraint(hb, {
    let rp = p.r2w.rotation_matrix;
    let nw = l.n.unit;
    let nl = rp.transpose() * nw;
    let cl = l.c + p.r2w.translation * nw;
    let h = sqrt(nl.x * nl.x + nl.y * nl.y);
    let mx = nl * obsv.measured_normal;
    let my = (obsv.measured_normal.y * nl.x - obsv.measured_normal.x * nl.y) / h;
    let mz = (obsv.measured_normal.z * (nl.x * nl.x + nl.y * nl.y)
        - nl.z * (nl.x * obsv.measured_normal.x + nl.y * obsv.measured_normal.y)) / h;
    [atan2(my, mx) * obsv.azimuth_weight,
     atan2(mz, sqrt(mx * mx + my * my)) * obsv.elevation_weight,
     (obsv.measured_c - cl) * obsv.distance_weight]
}, parent = obsv))]
#[derive(Clone)]
struct Obsv<T: Float> {
    /// The observing pose.
    #[arael(ref = root.poses)]
    p: Ref<PoseV<T>>,
    /// The observed plane landmark.
    #[arael(ref = root.planes)]
    l: Ref<PlaneLm<T>>,
    /// Measured plane normal (unit) in the sensor frame.
    measured_normal: vect3<T>,
    /// Measured distance coefficient of the local plane (n.x + c = 0).
    measured_c: T,
    /// Whitening weight (1/sigma) of the azimuth residual.
    azimuth_weight: T,
    /// Whitening weight (1/sigma) of the elevation residual.
    elevation_weight: T,
    /// Whitening weight (1/sigma) of the distance residual.
    distance_weight: T,
    /// The pose-plane coupling tile of J^T J.
    hb: CrossBlock<PoseV<T>, PlaneLm<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct World {
    poses: arael::refs::Vec<PoseV<f64>>,
    planes: arael::refs::Vec<PlaneLm<f64>>,
    odos: std::vec::Vec<Odov<f64>>,
    obs: std::vec::Vec<Obsv<f64>>,
}

/// Termination class for this benchmark, shared by every system in the
/// table (see the C++ runners' matching constants).
///
/// Tighter than the harness default of 1e-5, because these costs are
/// large: 1e-5 RELATIVE at a cost of 12000 means "stop once a step gains
/// less than 0.12", which leaves a solve short of the table's 5 cm
/// agreement gate. PLANE_TOL overrides.
pub fn tolerance() -> f64 {
    std::env::var("PLANE_TOL").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-7)
}

/// The same class for single-precision rows, which cannot be held to the
/// double-precision one: f32 epsilon is 1.2e-7, so a relative test at or
/// below that is measuring rounding noise rather than progress. A solver
/// asked to chase it spends its time on rejected steps -- at 900 poses
/// SymForce took 30 attempts for 6 accepted steps at 1e-7, and 7 for 7
/// here.
pub fn tolerance_f32() -> f64 {
    std::env::var("PLANE_TOL_F32").ok().and_then(|v| v.parse().ok()).unwrap_or(1e-5)
}

/// The entity collections at any precision (`cast` is the identity at
/// f64 here -- the scene is f64 -- and the rounding conversion at f32).
/// Quaternions are re-normalized after the cast, as the f32 build always
/// did.
fn build_parts<T: Float>(raw: &RawScene) -> (
    arael::refs::Vec<PoseV<T>>,
    arael::refs::Vec<PlaneLm<T>>,
    std::vec::Vec<Odov<T>>,
    std::vec::Vec<Obsv<T>>,
) {
    let c = |x: f64| T::from(x).unwrap();
    let mut poses = arael::refs::Vec::new();
    for (k, p) in raw.poses.iter().enumerate() {
        let (t, q) = (p.t.cast(), p.q.cast().unit());
        poses.push(PoseV {
            r2w: if k == 0 { TransformParam::fixed(t, q) } else { TransformParam::new(t, q) },
            hb: SelfBlock::new(),
        });
    }
    let mut planes = arael::refs::Vec::new();
    for pl in &raw.planes {
        planes.push(PlaneLm {
            n: UnitVecParam::new(pl.n.cast()),
            c: Param::new(c(pl.c)),
            hb: SelfBlock::new(),
        });
    }
    let mut odos = std::vec::Vec::new();
    for &(i, j, ref rel, translation_weight, rotation_weight) in &raw.odos {
        odos.push(Odov {
            a: poses.ref_at(i as u32),
            b: poses.ref_at(j as u32),
            measured_translation: rel.t.cast(),
            measured_rotation_transposed: rel.q.cast().unit().rotation_matrix().transpose(),
            translation_weight: c(translation_weight),
            rotation_weight: c(rotation_weight),
            hb: CrossBlock::new(),
        });
    }
    let mut obs = std::vec::Vec::new();
    for &(p, l, ref pl, azimuth_weight, elevation_weight, distance_weight) in &raw.obs {
        obs.push(Obsv {
            p: poses.ref_at(p as u32),
            l: planes.ref_at(l as u32),
            measured_normal: pl.n.cast(),
            measured_c: c(pl.c),
            azimuth_weight: c(azimuth_weight),
            elevation_weight: c(elevation_weight),
            distance_weight: c(distance_weight),
            hb: CrossBlock::new(),
        });
    }
    (poses, planes, odos, obs)
}

fn build(raw: &RawScene) -> World {
    let (poses, planes, odos, obs) = build_parts(raw);
    World { poses, planes, odos, obs }
}

fn build_f32(raw: &RawScene) -> WorldF {
    let (poses, planes, odos, obs) = build_parts(raw);
    WorldF { poses, planes, odos, obs }
}

fn extract_parts<T: Float>(
    poses: &arael::refs::Vec<PoseV<T>>,
    planes: &arael::refs::Vec<PlaneLm<T>>,
) -> Solution {
    Solution {
        poses: poses.iter()
            .map(|p| Pose {
                q: p.r2w.rotation.cast().unit(),
                t: p.r2w.translation.cast(),
            })
            .collect(),
        planes: planes.iter()
            .map(|pl| Plane::normalized(pl.n.unit.cast(), pl.c.value.to_f64().unwrap()))
            .collect(),
    }
}

fn extract(world: &World) -> Solution {
    extract_parts(&world.poses, &world.planes)
}

fn extract_f32(world: &WorldF) -> Solution {
    extract_parts(&world.poses, &world.planes)
}

/// The arael model cost at the initial estimate -- for the harness to
/// cross-check against scene::reference_cost.
pub fn initial_cost(raw: &RawScene) -> f64 {
    use arael::simple_lm::LmProblem;
    let mut world = build(raw);
    let mut params: Vec<f64> = Vec::new();
    world.serialize64(&mut params);
    world.calc_cost(&params)
}

impl bench_harness::arael::Model for World {
    type Scalar = f64;
    type Input = RawScene;
    type Solution = Solution;
    // Near-Gauss-Newton start: clean Gaussian noise from a good odometry
    // init, same policy as the other pose benchmarks.
    fn lambda0(_: &RawScene) -> f64 { 1e-8 }
    // The unanchored loop has a slow global bending mode; the fixed ladder
    // oscillates around its optimal damping in that tail, the gain-ratio
    // driver holds it.
    const NIELSEN: bool = true;
    fn build(raw: &RawScene) -> Self { build(raw) }
    fn serialize(&mut self, out: &mut Vec<f64>) { self.serialize64(out); }
    fn deserialize(&mut self, x: &[f64]) { self.deserialize64(x); }
    fn solution(&self) -> Solution { extract(self) }
    fn solve(_: &RawScene, params: &[f64], m: &mut Self, cfg: &LmConfig<f64>)
        -> LmResult<f64> {
        lm_solve(params, &mut SparseFaer::<f64>::new(), m, cfg).unwrap()
    }
    fn tune(cfg: &mut LmConfig<f64>) {
        cfg.abs_precision = tolerance();
        cfg.rel_precision = tolerance();
    }
}

pub type RunOut = bench_harness::table::Row<Solution>;

pub fn run(raw: &RawScene) -> RunOut { bench_harness::arael::run::<World>(raw) }
pub fn run_f32(raw: &RawScene) -> RunOut { bench_harness::arael::run::<WorldF>(raw) }

// Capped single solves (no timing) -- for the peak-memory pass.
pub fn run_capped(raw: &RawScene, max_iters: usize) -> Solution {
    let mut world = build(raw);
    let mut params: Vec<f64> = Vec::new();
    world.serialize64(&mut params);
    let cfg = bench_harness::arael::config::<World>(raw, max_iters);
    let r = lm_solve(&params, &mut SparseFaer::<f64>::new(), &mut world, &cfg).unwrap();
    world.deserialize64(&r.x);
    extract(&world)
}

pub fn run_f32_capped(raw: &RawScene, max_iters: usize) -> Solution {
    let mut world = build_f32(raw);
    let mut params: Vec<f32> = Vec::new();
    world.serialize32(&mut params);
    let cfg = bench_harness::arael::config::<WorldF>(raw, max_iters);
    let r = lm_solve(&params, &mut SparseFaer::<f32>::new(), &mut world, &cfg).unwrap();
    world.deserialize32(&r.x);
    extract_f32(&world)
}

// ------------------------------------------------------------ the f32 root

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct WorldF {
    poses: arael::refs::Vec<PoseV<f32>>,
    planes: arael::refs::Vec<PlaneLm<f32>>,
    odos: std::vec::Vec<Odov<f32>>,
    obs: std::vec::Vec<Obsv<f32>>,
}

impl bench_harness::arael::Model for WorldF {
    type Scalar = f32;
    type Input = RawScene;
    type Solution = Solution;
    fn lambda0(_: &RawScene) -> f64 { 1e-8 }
    const NIELSEN: bool = true;
    fn build(raw: &RawScene) -> Self { build_f32(raw) }
    fn serialize(&mut self, out: &mut Vec<f32>) { self.serialize32(out); }
    fn deserialize(&mut self, x: &[f32]) { self.deserialize32(x); }
    fn solution(&self) -> Solution { extract_f32(self) }
    fn solve(_: &RawScene, params: &[f32], m: &mut Self, cfg: &LmConfig<f32>)
        -> LmResult<f32> {
        lm_solve(params, &mut SparseFaer::<f32>::new(), m, cfg).unwrap()
    }
    fn tune(cfg: &mut LmConfig<f32>) {
        cfg.abs_precision = tolerance_f32() as f32;
        cfg.rel_precision = tolerance_f32() as f32;
    }
}
