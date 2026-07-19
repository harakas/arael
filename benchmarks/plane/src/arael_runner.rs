// The arael runner. The S^2 plane-normal parameterization is a USER-DEFINED
// component: declared here with #[arael(component)], lifecycle via the
// Component trait, chart cached with compute=, embed via chained symbolic=
// fields -- the full macro path, nothing arael-internal.
//
// The f32 model is a type-for-type twin (the macro bakes the scalar into the
// generated code, so the two are different types), sharing the constraint
// bodies verbatim.

use arael::model::{Component, CrossBlock, Param, QuaternionParam, SelfBlock};
use arael::refs::Ref;
use arael::simple_lm::{lm_solve, LmConfig, LmResult, SparseFaer};
use arael::matrix::{matrix3d, matrix3f};
use arael::quatern::{quaternd, quaternf};
use arael::vect::{vect2d, vect2f, vect3d, vect3f};

use crate::scene::{Plane, Pose, RawScene, Solution};

/// Unit direction on S^2: reference quaternion chart (x-axis = direction),
/// 2-DOF body-frame delta about the frame's y/z. The embed is the rotated
/// first column of the small-rotation matrix of (1, (0, d.x, d.y)/2)
/// normalized -- exact on the sphere for every trial delta.
#[arael::model]
#[arael(component)]
#[derive(Clone)]
struct UnitVec {
    // Rotation that takes unit vector (1, 0, 0) into "unit".
    ref_q: quaternd,
    #[arael(compute = self.ref_q.rotation_matrix())]
    rot: matrix3d,
    // d forms a rotation vector axis*angle = (0, dx, dy). We can construct a first-order rotation quaternion
    //    q = (1, 0, d.x/2, d.y/2) / sqrt(s2)
    // where the normalization coefficient is
    //    s2 = 1 + d.x^2 / 4 + d.y^2 / 4
    // we derive the rotation matrix for q -- but rotation of (1, 0, 0) is equal to its first column:
    //    [1 - 2*(y^2+z^2), 2*(x*y+ w*z), 2*(x*z-w*y)] -- we never have to calculate 1/sqrt(s2) as it is always squared
    d: Param<vect2d>,
    #[arael(symbolic = {
        let s2 = 1.0 + (d.x * d.x + d.y * d.y) * 0.25;
        let local = vect3sym::from_components(
            1.0 - (d.x * d.x + d.y * d.y) / (2.0 * s2), d.y / s2, 0.0 - d.x / s2);
        rot * local
    })]
    unit: vect3d,
    #[arael(deriv = unit, by = d)]
    unit_d: [vect3d; 2],
}

impl UnitVec {
    fn ex() -> vect3d {
        vect3d::new(1.0, 0.0, 0.0)
    }
    fn new(dir: vect3d) -> UnitVec {
        let mut u = UnitVec {
            ref_q: quaternd::identity(),
            rot: matrix3d::identity(),
            d: Param::new(vect2d::new(0.0, 0.0)),
            unit: dir,
            unit_d: [vect3d::new(0.0, 0.0, 0.0); 2],
        };
        Component::start(&mut u);
        u
    }
}

impl Component for UnitVec {
    fn start(&mut self) {
        self.unit = self.unit.unit();
        self.ref_q = quaternd::from_two_vectors(Self::ex(), self.unit);
        self.d.value = vect2d::new(0.0, 0.0);
    }
    fn update(&mut self) {
        let dq = quaternd::from_rotation_vector_small(
            vect3d::new(0.0, self.d.value.x, self.d.value.y));
        self.ref_q = (self.ref_q * dq).unit();
        self.d.value = vect2d::new(0.0, 0.0);
    }
    fn finish(&mut self) {
        let dq = quaternd::from_rotation_vector_small(
            vect3d::new(0.0, self.d.value.x, self.d.value.y));
        self.unit = (self.ref_q * dq).rotate(Self::ex());
    }
}

#[arael::model]
#[derive(Clone)]
struct PoseV {
    /// Position in the world frame.
    pos: Param<vect3d>,
    /// Rotation body-to-world.
    q: QuaternionParam<f64>,
    /// This pose's Hessian tile (gradient + diagonal block of J^T J).
    hb: SelfBlock<PoseV>,
}

#[arael::model]
#[derive(Clone)]
struct PlaneLm {
    /// Unit normal of the plane (2-DOF component).
    n: UnitVec,
    /// Distance coefficient: the plane is n.x + c = 0, distance = -c.
    c: Param<f64>,
    /// This plane's Hessian tile.
    hb: SelfBlock<PlaneLm>,
}

// Odometry between-residual, identical to the g2o runner's custom edge.
// Translation: err_t = R_a^T (b.pos - a.pos) - measured_translation.
// Rotation: the error rotation dR = R_m^T R_a^T R_b (measured relative
// rotation inverted, composed with the estimated one) should be identity;
// the residual is its skew part read as a vector,
//   err_r = vee((dR - dR^T)/2) = sin(angle) * axis,
// zero exactly when measurement and estimate agree. vee() maps a
// skew-symmetric matrix to its 3-vector: vee(M) = (M[2][1], M[0][2],
// M[1][0]) -- the c1/c2/c3 column arithmetic in the body.
#[arael::model]
#[arael(constraint(hb, {
    let ra = a.q.rotation_matrix();
    let rb = b.q.rotation_matrix();
    let dt = ra.transpose() * (b.pos - a.pos) - odov.measured_translation;
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
struct Odov {
    /// The earlier pose: the measurement is expressed in ITS frame.
    #[arael(ref = root.poses)]
    a: Ref<PoseV>,
    /// The later pose the measurement leads to.
    #[arael(ref = root.poses)]
    b: Ref<PoseV>,
    /// Measured relative translation: where odometry says `b` sits in
    /// `a`'s frame; compared against R_a^T (b.pos - a.pos).
    measured_translation: vect3d,
    /// TRANSPOSE of the measured relative rotation, R_m^T -- stored
    /// pre-transposed because the residual only ever uses it that way
    /// (dR = R_m^T R_a^T R_b).
    measured_rotation_transposed: matrix3d,
    /// Whitening weight (1/sigma, per axis) of the translation residual.
    translation_weight: f64,
    /// Whitening weight (1/sigma, per axis) of the rotation residual.
    rotation_weight: f64,
    /// The a-b coupling tile of J^T J this constraint accumulates into
    /// (named as the primary block in the constraint attribute above).
    hb: CrossBlock<PoseV, PoseV>,
}

// Plane observation: g2o's EdgeSE3PlaneSensorCalib error (Plane3D::ominus),
// written algebraically. Predicted local plane (n_l, c_l) from the world
// plane through the pose; error = (azimuth, elevation) of the measured
// normal in the frame aligning n_l with e1, plus the distance difference.
#[arael::model]
#[arael(constraint(hb, {
    let rp = p.q.rotation_matrix();
    let nw = l.n.unit;
    let nl = rp.transpose() * nw;
    let cl = l.c + p.pos * nw;
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
struct Obsv {
    /// The observing pose.
    #[arael(ref = root.poses)]
    p: Ref<PoseV>,
    /// The observed plane landmark.
    #[arael(ref = root.planes)]
    l: Ref<PlaneLm>,
    /// Measured plane normal (unit) in the sensor frame.
    measured_normal: vect3d,
    /// Measured distance coefficient of the local plane (n.x + c = 0).
    measured_c: f64,
    /// Whitening weight (1/sigma) of the azimuth residual.
    azimuth_weight: f64,
    /// Whitening weight (1/sigma) of the elevation residual.
    elevation_weight: f64,
    /// Whitening weight (1/sigma) of the distance residual.
    distance_weight: f64,
    /// The pose-plane coupling tile of J^T J.
    hb: CrossBlock<PoseV, PlaneLm>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct World {
    poses: arael::refs::Vec<PoseV>,
    planes: arael::refs::Vec<PlaneLm>,
    odos: std::vec::Vec<Odov>,
    obs: std::vec::Vec<Obsv>,
}

fn build(raw: &RawScene) -> World {
    let mut world = World {
        poses: arael::refs::Vec::new(),
        planes: arael::refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    for (k, p) in raw.poses.iter().enumerate() {
        let fixed = k == 0;
        world.poses.push(PoseV {
            pos: if fixed { Param::fixed(p.t) } else { Param::new(p.t) },
            q: if fixed { QuaternionParam::fixed(p.q) } else { QuaternionParam::new(p.q) },
            hb: SelfBlock::new(),
        });
    }
    for pl in &raw.planes {
        world.planes.push(PlaneLm {
            n: UnitVec::new(pl.n),
            c: Param::new(pl.c),
            hb: SelfBlock::new(),
        });
    }
    for &(i, j, ref rel, translation_weight, rotation_weight) in &raw.odos {
        world.odos.push(Odov {
            a: world.poses.ref_at(i as u32),
            b: world.poses.ref_at(j as u32),
            measured_translation: rel.t,
            measured_rotation_transposed: rel.q.rotation_matrix().transpose(),
            translation_weight,
            rotation_weight,
            hb: CrossBlock::new(),
        });
    }
    for &(p, l, ref pl, azimuth_weight, elevation_weight, distance_weight) in &raw.obs {
        world.obs.push(Obsv {
            p: world.poses.ref_at(p as u32),
            l: world.planes.ref_at(l as u32),
            measured_normal: pl.n,
            measured_c: pl.c,
            azimuth_weight, elevation_weight, distance_weight,
            hb: CrossBlock::new(),
        });
    }
    world
}

fn extract(world: &World) -> Solution {
    Solution {
        poses: world.poses.iter()
            .map(|p| Pose { q: p.q.value, t: p.pos.value }).collect(),
        planes: world.planes.iter()
            .map(|pl| Plane { n: pl.n.unit, c: pl.c.value }).collect(),
    }
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
        lm_solve(params, &mut SparseFaer::<f64>::new(), m, cfg)
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
    let r = lm_solve(&params, &mut SparseFaer::<f64>::new(), &mut world, &cfg);
    world.deserialize64(&r.x);
    extract(&world)
}

pub fn run_f32_capped(raw: &RawScene, max_iters: usize) -> Solution {
    let mut world = build_f32(raw);
    let mut params: Vec<f32> = Vec::new();
    world.serialize32(&mut params);
    let cfg = bench_harness::arael::config::<WorldF>(raw, max_iters);
    let r = lm_solve(&params, &mut SparseFaer::<f32>::new(), &mut world, &cfg);
    world.deserialize32(&r.x);
    extract_f32(&world)
}

// ------------------------------------------------------------ the f32 twin

#[arael::model]
#[arael(component)]
#[derive(Clone)]
struct UnitVecF {
    ref_q: quaternf,
    #[arael(compute = self.ref_q.rotation_matrix())]
    rot: matrix3f,
    d: Param<vect2f>,
    #[arael(symbolic = {
        let s2 = 1.0 + (d.x * d.x + d.y * d.y) * 0.25;
        let local = vect3sym::from_components(
            1.0 - (d.x * d.x + d.y * d.y) / (2.0 * s2), d.y / s2, 0.0 - d.x / s2);
        rot * local
    })]
    unit: vect3f,
    #[arael(deriv = unit, by = d)]
    unit_d: [vect3f; 2],
}

impl UnitVecF {
    fn ex() -> vect3f {
        vect3f::new(1.0, 0.0, 0.0)
    }
    fn new(dir: vect3f) -> UnitVecF {
        let mut u = UnitVecF {
            ref_q: quaternf::identity(),
            rot: matrix3f::identity(),
            d: Param::new(vect2f::new(0.0, 0.0)),
            unit: dir,
            unit_d: [vect3f::new(0.0, 0.0, 0.0); 2],
        };
        Component::start(&mut u);
        u
    }
}

impl Component for UnitVecF {
    fn start(&mut self) {
        self.unit = self.unit.unit();
        self.ref_q = quaternf::from_two_vectors(Self::ex(), self.unit);
        self.d.value = vect2f::new(0.0, 0.0);
    }
    fn update(&mut self) {
        let dq = quaternf::from_rotation_vector_small(
            vect3f::new(0.0, self.d.value.x, self.d.value.y));
        self.ref_q = (self.ref_q * dq).unit();
        self.d.value = vect2f::new(0.0, 0.0);
    }
    fn finish(&mut self) {
        let dq = quaternf::from_rotation_vector_small(
            vect3f::new(0.0, self.d.value.x, self.d.value.y));
        self.unit = (self.ref_q * dq).rotate(Self::ex());
    }
}

#[arael::model]
#[derive(Clone)]
struct PoseVF {
    pos: Param<vect3f>,
    q: QuaternionParam<f32>,
    hb: SelfBlock<PoseVF, f32>,
}

#[arael::model]
#[derive(Clone)]
struct PlaneLmF {
    n: UnitVecF,
    c: Param<f32>,
    hb: SelfBlock<PlaneLmF, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.q.rotation_matrix();
    let rb = b.q.rotation_matrix();
    let dt = ra.transpose() * (b.pos - a.pos) - odovf.measured_translation;
    let dr = odovf.measured_rotation_transposed * (ra.transpose() * rb);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x * odovf.translation_weight, dt.y * odovf.translation_weight, dt.z * odovf.translation_weight,
     (c2.z - c3.y) * 0.5 * odovf.rotation_weight,
     (c3.x - c1.z) * 0.5 * odovf.rotation_weight,
     (c1.y - c2.x) * 0.5 * odovf.rotation_weight]
}, parent = odovf))]
#[derive(Clone)]
struct OdovF {
    #[arael(ref = root.poses)]
    a: Ref<PoseVF>,
    #[arael(ref = root.poses)]
    b: Ref<PoseVF>,
    measured_translation: vect3f,
    measured_rotation_transposed: matrix3f,
    translation_weight: f32,
    rotation_weight: f32,
    hb: CrossBlock<PoseVF, PoseVF, f32>,
}

#[arael::model]
#[arael(constraint(hb, {
    let rp = p.q.rotation_matrix();
    let nw = l.n.unit;
    let nl = rp.transpose() * nw;
    let cl = l.c + p.pos * nw;
    let h = sqrt(nl.x * nl.x + nl.y * nl.y);
    let mx = nl * obsvf.measured_normal;
    let my = (obsvf.measured_normal.y * nl.x - obsvf.measured_normal.x * nl.y) / h;
    let mz = (obsvf.measured_normal.z * (nl.x * nl.x + nl.y * nl.y)
        - nl.z * (nl.x * obsvf.measured_normal.x + nl.y * obsvf.measured_normal.y)) / h;
    [atan2(my, mx) * obsvf.azimuth_weight,
     atan2(mz, sqrt(mx * mx + my * my)) * obsvf.elevation_weight,
     (obsvf.measured_c - cl) * obsvf.distance_weight]
}, parent = obsvf))]
#[derive(Clone)]
struct ObsvF {
    #[arael(ref = root.poses)]
    p: Ref<PoseVF>,
    #[arael(ref = root.planes)]
    l: Ref<PlaneLmF>,
    measured_normal: vect3f,
    measured_c: f32,
    azimuth_weight: f32,
    elevation_weight: f32,
    distance_weight: f32,
    hb: CrossBlock<PoseVF, PlaneLmF, f32>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
pub struct WorldF {
    poses: arael::refs::Vec<PoseVF>,
    planes: arael::refs::Vec<PlaneLmF>,
    odos: std::vec::Vec<OdovF>,
    obs: std::vec::Vec<ObsvF>,
}

fn v3f(v: vect3d) -> vect3f {
    vect3f::new(v.x as f32, v.y as f32, v.z as f32)
}
fn qf(q: quaternd) -> quaternf {
    quaternf::new(q.t as f32, v3f(q.v)).unit()
}

fn build_f32(raw: &RawScene) -> WorldF {
    let mut world = WorldF {
        poses: arael::refs::Vec::new(),
        planes: arael::refs::Vec::new(),
        odos: std::vec::Vec::new(),
        obs: std::vec::Vec::new(),
    };
    for (k, p) in raw.poses.iter().enumerate() {
        let fixed = k == 0;
        world.poses.push(PoseVF {
            pos: if fixed { Param::fixed(v3f(p.t)) } else { Param::new(v3f(p.t)) },
            q: if fixed { QuaternionParam::fixed(qf(p.q)) } else { QuaternionParam::new(qf(p.q)) },
            hb: SelfBlock::new(),
        });
    }
    for pl in &raw.planes {
        world.planes.push(PlaneLmF {
            n: UnitVecF::new(v3f(pl.n)),
            c: Param::new(pl.c as f32),
            hb: SelfBlock::new(),
        });
    }
    for &(i, j, ref rel, translation_weight, rotation_weight) in &raw.odos {
        world.odos.push(OdovF {
            a: world.poses.ref_at(i as u32),
            b: world.poses.ref_at(j as u32),
            measured_translation: v3f(rel.t),
            measured_rotation_transposed: qf(rel.q).rotation_matrix().transpose(),
            translation_weight: translation_weight as f32,
            rotation_weight: rotation_weight as f32,
            hb: CrossBlock::new(),
        });
    }
    for &(p, l, ref pl, azimuth_weight, elevation_weight, distance_weight) in &raw.obs {
        world.obs.push(ObsvF {
            p: world.poses.ref_at(p as u32),
            l: world.planes.ref_at(l as u32),
            measured_normal: v3f(pl.n),
            measured_c: pl.c as f32,
            azimuth_weight: azimuth_weight as f32,
            elevation_weight: elevation_weight as f32,
            distance_weight: distance_weight as f32,
            hb: CrossBlock::new(),
        });
    }
    world
}

fn extract_f32(world: &WorldF) -> Solution {
    Solution {
        poses: world.poses.iter()
            .map(|p| Pose {
                q: quaternd::new(p.q.value.t as f64,
                    vect3d::new(p.q.value.v.x as f64, p.q.value.v.y as f64,
                        p.q.value.v.z as f64)).unit(),
                t: vect3d::new(p.pos.value.x as f64, p.pos.value.y as f64,
                    p.pos.value.z as f64),
            })
            .collect(),
        planes: world.planes.iter()
            .map(|pl| Plane::normalized(
                vect3d::new(pl.n.unit.x as f64, pl.n.unit.y as f64, pl.n.unit.z as f64),
                pl.c.value as f64))
            .collect(),
    }
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
        lm_solve(params, &mut SparseFaer::<f32>::new(), m, cfg)
    }
}
