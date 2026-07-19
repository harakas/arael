// Plane SLAM with a USER-DEFINED component.
//
// The subject here is the component API: `UnitVec`, a 2-DOF unit direction
// on the sphere, is defined IN THIS FILE from public building blocks --
// `#[arael(component)]`, the `Component` lifecycle trait, a `compute =`
// cached field, a `symbolic =` embed with `let` intermediates, and a
// declared `deriv =` Jacobian cache. Nothing about it is arael-internal;
// a user crate can define its own manifold parameterizations the same way.
//
// The problem it is put to work on: SE3 poses on a loop with odometry
// between neighbours, and plane landmarks (unit normal + distance
// coefficient) observed relative to each pose -- the application of g2o's
// plane_slam example, and the same model the plane benchmark
// (benchmarks/plane) races against g2o, Ceres, GTSAM, SymForce and factrs.
//
// Run with: cargo run --release --example plane_slam_demo

use arael::model::{Component, CrossBlock, Param, QuaternionParam, SelfBlock};
use arael::matrix::matrix3d;
use arael::quatern::quaternd;
use arael::refs::{self, Ref};
use arael::simple_lm::{LmConfig, LmProblem};
use arael::vect::{vect2d, vect3d};

use rand::prelude::*;
use rand::rngs::StdRng;
use rand_distr::StandardNormal;

// ---------------------------------------------------------------------------
// The user-defined component: a unit direction on S^2
// ---------------------------------------------------------------------------

/// Unit direction on S^2 with 2 degrees of freedom.
///
/// The chart is a reference quaternion whose x-axis IS the direction; the
/// solver sees a 2-DOF body-frame rotation delta about the frame's y/z
/// axes. Rotating about the direction itself changes nothing -- that third
/// axis is simply not a parameter, so the unobservable dimension never
/// reaches the solver. Every accepted step folds the delta into the
/// reference and re-centres it at zero (`Component::update`), the same
/// contract as arael's own rotation parameters.
#[arael::model]
#[arael(component)]
struct UnitVec {
    /// Rotation that takes the unit vector (1, 0, 0) into `unit`.
    ref_q: quaternd,
    /// Chart matrix, cached: refreshed from `ref_q` whenever the reference
    /// moves; a per-iteration CONSTANT in generated constraint code.
    #[arael(compute = self.ref_q.rotation_matrix())]
    rot: matrix3d,
    /// The 2-DOF tangent delta. `d` forms a rotation vector
    /// axis*angle = (0, d.x, d.y); the first-order rotation quaternion is
    ///    q = (1, 0, d.x/2, d.y/2) / sqrt(s2),  s2 = 1 + (d.x^2 + d.y^2)/4.
    /// Normalizing makes q a genuine rotation for EVERY delta, so a trial
    /// step of any size stays on the sphere.
    d: Param<vect2d>,
    /// What constraint bodies read. The expression is the first column of
    /// q's rotation matrix, [1 - 2(y^2+z^2), 2(xy + wz), 2(xz - wy)],
    /// rotated by the chart -- 1/sqrt(s2) never appears because every term
    /// is a product of two components. Precomputed per entity per
    /// evaluation; bodies read the field, not the formula.
    #[arael(symbolic = {
        let s2 = 1.0 + (d.x * d.x + d.y * d.y) * 0.25;
        let local = vect3sym::from_components(
            1.0 - (d.x * d.x + d.y * d.y) / (2.0 * s2), d.y / s2, 0.0 - d.x / s2);
        rot * local
    })]
    unit: vect3d,
    /// Declared Jacobian cache: [d(unit)/d(d.x), d(unit)/d(d.y)], filled by
    /// the generated precompute, read by constraint Jacobians instead of
    /// re-deriving the embed at every observation.
    #[arael(deriv = unit, by = d)]
    unit_d: [vect3d; 2],
}

impl UnitVec {
    fn ex() -> vect3d {
        vect3d::new(1.0, 0.0, 0.0)
    }
    fn new(direction: vect3d) -> UnitVec {
        let mut u = UnitVec {
            ref_q: quaternd::identity(),
            rot: matrix3d::identity(),
            d: Param::new(vect2d::new(0.0, 0.0)),
            unit: direction,
            unit_d: [vect3d::new(0.0, 0.0, 0.0); 2],
        };
        Component::start(&mut u);
        u
    }
}

impl Component for UnitVec {
    /// User-facing value in: seed the chart from `unit`, zero the delta.
    fn start(&mut self) {
        self.unit = self.unit.unit();
        self.ref_q = quaternd::from_two_vectors(Self::ex(), self.unit);
        self.d.value = vect2d::new(0.0, 0.0);
    }
    /// Accepted step: fold the delta into the reference, re-centre at zero.
    fn update(&mut self) {
        let dq = quaternd::from_rotation_vector_small(
            vect3d::new(0.0, self.d.value.x, self.d.value.y));
        self.ref_q = (self.ref_q * dq).unit();
        self.d.value = vect2d::new(0.0, 0.0);
    }
    /// Solve done: optimized direction out.
    fn finish(&mut self) {
        let dq = quaternd::from_rotation_vector_small(
            vect3d::new(0.0, self.d.value.x, self.d.value.y));
        self.unit = (self.ref_q * dq).rotate(Self::ex());
    }
}

// ---------------------------------------------------------------------------
// The model: poses, plane landmarks, and the two constraints
// ---------------------------------------------------------------------------

#[arael::model]
struct Pose {
    /// Position in the world frame.
    pos: Param<vect3d>,
    /// Rotation body-to-world.
    q: QuaternionParam<f64>,
    /// This pose's Hessian tile.
    hb: SelfBlock<Pose>,
}

#[arael::model]
struct PlaneLandmark {
    /// Unit normal of the plane -- the component above.
    normal: UnitVec,
    /// Distance coefficient: the plane is n.x + c = 0, distance = -c.
    c: Param<f64>,
    /// This plane's Hessian tile.
    hb: SelfBlock<PlaneLandmark>,
}

// Odometry between-residual.
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
    let dt = ra.transpose() * (b.pos - a.pos) - odometry.measured_translation;
    let dr = odometry.measured_rotation_transposed * (ra.transpose() * rb);
    let c1 = dr * vect3sym::from_components(1.0, 0.0, 0.0);
    let c2 = dr * vect3sym::from_components(0.0, 1.0, 0.0);
    let c3 = dr * vect3sym::from_components(0.0, 0.0, 1.0);
    [dt.x * odometry.translation_weight,
     dt.y * odometry.translation_weight,
     dt.z * odometry.translation_weight,
     (c2.z - c3.y) * 0.5 * odometry.rotation_weight,
     (c3.x - c1.z) * 0.5 * odometry.rotation_weight,
     (c1.y - c2.x) * 0.5 * odometry.rotation_weight]
}, parent = odometry))]
struct Odometry {
    /// The earlier pose: the measurement is expressed in ITS frame.
    #[arael(ref = root.poses)]
    a: Ref<Pose>,
    /// The later pose the measurement leads to.
    #[arael(ref = root.poses)]
    b: Ref<Pose>,
    /// Measured relative translation: where odometry says `b` sits in
    /// `a`'s frame.
    measured_translation: vect3d,
    /// TRANSPOSE of the measured relative rotation, stored pre-transposed
    /// because the residual only ever uses it that way.
    measured_rotation_transposed: matrix3d,
    /// Whitening weight (1/sigma, per axis) of the translation residual.
    translation_weight: f64,
    /// Whitening weight (1/sigma, per axis) of the rotation residual.
    rotation_weight: f64,
    /// The a-b coupling tile of J^T J.
    hb: CrossBlock<Pose, Pose>,
}

// Plane observation (g2o's Plane3D::ominus, written algebraically).
// Predict the plane in the observing pose's frame: normal
// n_l = R_p^T n_world, distance coefficient c_l = c + pos . n_world. The
// error is the (azimuth, elevation) of the measured normal expressed in a
// frame that aligns the predicted normal with the x-axis, plus the
// distance difference -- three numbers, zero when prediction and
// measurement coincide.
#[arael::model]
#[arael(constraint(hb, {
    let rp = p.q.rotation_matrix();
    let nw = l.normal.unit;
    let nl = rp.transpose() * nw;
    let cl = l.c + p.pos * nw;
    let h = sqrt(nl.x * nl.x + nl.y * nl.y);
    let mx = nl * observation.measured_normal;
    let my = (observation.measured_normal.y * nl.x - observation.measured_normal.x * nl.y) / h;
    let mz = (observation.measured_normal.z * (nl.x * nl.x + nl.y * nl.y)
        - nl.z * (nl.x * observation.measured_normal.x + nl.y * observation.measured_normal.y)) / h;
    [atan2(my, mx) * observation.azimuth_weight,
     atan2(mz, sqrt(mx * mx + my * my)) * observation.elevation_weight,
     (observation.measured_c - cl) * observation.distance_weight]
}, parent = observation))]
struct Observation {
    /// The observing pose.
    #[arael(ref = root.poses)]
    p: Ref<Pose>,
    /// The observed plane landmark.
    #[arael(ref = root.planes)]
    l: Ref<PlaneLandmark>,
    /// Measured plane normal (unit) in the sensor frame.
    measured_normal: vect3d,
    /// Measured distance coefficient of the local plane.
    measured_c: f64,
    /// Whitening weights (1/sigma) of the three residual components.
    azimuth_weight: f64,
    elevation_weight: f64,
    distance_weight: f64,
    /// The pose-plane coupling tile of J^T J.
    hb: CrossBlock<Pose, PlaneLandmark>,
}

#[arael::model]
#[arael(root)]
struct World {
    poses: refs::Vec<Pose>,
    planes: refs::Vec<PlaneLandmark>,
    odometry: std::vec::Vec<Odometry>,
    observations: std::vec::Vec<Observation>,
}

// ---------------------------------------------------------------------------
// A small synthetic scene
// ---------------------------------------------------------------------------

/// A rigid pose (rotation + position) for scene bookkeeping.
#[derive(Clone, Copy)]
struct SE3 {
    q: quaternd,
    pos: vect3d,
}

impl SE3 {
    fn compose(self, o: SE3) -> SE3 {
        SE3 { q: (self.q * o.q).unit(), pos: self.pos + self.q.rotate(o.pos) }
    }
    fn inverse(self) -> SE3 {
        let qi = self.q.conj();
        SE3 { q: qi, pos: -qi.rotate(self.pos) }
    }
}

/// A plane as (unit normal, c), n.x + c = 0. Transforming by a pose (the
/// g2o Plane3D convention): n' = R n, c' = c - pos . n'.
#[derive(Clone, Copy)]
struct ScenePlane {
    normal: vect3d,
    c: f64,
}

impl ScenePlane {
    fn through(point: vect3d, normal: vect3d) -> ScenePlane {
        let normal = normal.unit();
        ScenePlane { normal, c: -(normal * point) }
    }
    fn transform(self, t: SE3) -> ScenePlane {
        let normal = t.q.rotate(self.normal);
        ScenePlane { normal, c: self.c - (t.pos * normal) }
    }
}

const N_POSES: usize = 48;
const PATH_RADIUS: f64 = 5.0;
const VIS_WINDOW: usize = 5; // wrapped pose distance from a plane's anchor
const SIGMA_ODO_T: f64 = 0.02; // m, per axis
const SIGMA_ODO_R: f64 = 0.005; // rad, per axis
const SIGMA_OBS_ANG: f64 = 0.01; // rad, normal direction
const SIGMA_OBS_D: f64 = 0.02; // m, distance coefficient

fn ground_truth_pose(i: usize) -> SE3 {
    let th = i as f64 / N_POSES as f64 * std::f64::consts::TAU;
    // A little roll/pitch wobble keeps the floor normals off the azimuth
    // chart's pole in the sensor frame.
    let roll = 0.06 * (3.0 * th).sin();
    let pitch = 0.05 * (2.0 * th).cos();
    SE3 {
        q: (quaternd::from_axis_angle(vect3d::new(0.0, 0.0, 1.0), th + std::f64::consts::FRAC_PI_2)
            * quaternd::from_axis_angle(vect3d::new(1.0, 0.0, 0.0), roll)
            * quaternd::from_axis_angle(vect3d::new(0.0, 1.0, 0.0), pitch))
        .unit(),
        pos: vect3d::new(PATH_RADIUS * th.cos(), PATH_RADIUS * th.sin(), 1.0),
    }
}

/// Plane triplets anchored around the loop, each visible from a window of
/// poses: an inward-facing wall, a tilted floor patch, a tilted side wall
/// -- three independent orientations per window.
fn ground_truth_planes() -> (Vec<ScenePlane>, Vec<usize>) {
    let mut planes = Vec::new();
    let mut anchors = Vec::new();
    for a in 0..6 {
        let center = a * N_POSES / 6 + 4;
        let th = center as f64 / N_POSES as f64 * std::f64::consts::TAU;
        let dir = vect3d::new(th.cos(), th.sin(), 0.0);
        let tang = vect3d::new(-th.sin(), th.cos(), 0.0);
        planes.push(ScenePlane::through(dir * (PATH_RADIUS + 3.0), -dir));
        planes.push(ScenePlane::through(dir * PATH_RADIUS,
            (vect3d::new(0.0, 0.0, 1.0) + dir * 0.1).unit()));
        planes.push(ScenePlane::through(dir * (PATH_RADIUS - 3.0) + vect3d::new(0.0, 0.0, 1.5),
            (dir + tang * 0.3 + vect3d::new(0.0, 0.0, 0.2)).unit()));
        anchors.extend([center, center, center]);
    }
    (planes, anchors)
}

fn wrapped_distance(i: usize, j: usize) -> usize {
    let d = (i as i64 - j as i64).unsigned_abs() as usize % N_POSES;
    d.min(N_POSES - d)
}

fn gauss(rng: &mut StdRng, sigma: f64) -> f64 {
    let x: f64 = rng.sample(StandardNormal);
    x * sigma
}

fn build_world() -> (World, Vec<SE3>, Vec<ScenePlane>) {
    let mut rng = StdRng::seed_from_u64(7);
    let gt_poses: Vec<SE3> = (0..N_POSES).map(ground_truth_pose).collect();
    let (gt_planes, anchors) = ground_truth_planes();

    // Noisy odometry measurements, and the initial guess integrated from
    // them (so the initial poses drift away from ground truth).
    let mut measurements = Vec::new();
    let mut init_poses = vec![gt_poses[0]];
    for i in 0..N_POSES - 1 {
        let rel = gt_poses[i].inverse().compose(gt_poses[i + 1]);
        let noise_q = quaternd::from_rotation_vector(vect3d::new(
            gauss(&mut rng, SIGMA_ODO_R),
            gauss(&mut rng, SIGMA_ODO_R),
            gauss(&mut rng, SIGMA_ODO_R)));
        let noisy = SE3 {
            q: (rel.q * noise_q).unit(),
            pos: rel.pos + vect3d::new(
                gauss(&mut rng, SIGMA_ODO_T),
                gauss(&mut rng, SIGMA_ODO_T),
                gauss(&mut rng, SIGMA_ODO_T)),
        };
        measurements.push((i, noisy));
        let last = *init_poses.last().unwrap();
        init_poses.push(last.compose(noisy));
    }

    // Noisy local plane observations from every pose that sees the plane.
    let mut observations = Vec::new();
    for (i, gt) in gt_poses.iter().enumerate() {
        for (j, plane) in gt_planes.iter().enumerate() {
            if wrapped_distance(i, anchors[j]) > VIS_WINDOW {
                continue;
            }
            let local = plane.transform(gt.inverse());
            let tilt = quaternd::from_rotation_vector(vect3d::new(
                gauss(&mut rng, SIGMA_OBS_ANG),
                gauss(&mut rng, SIGMA_OBS_ANG),
                gauss(&mut rng, SIGMA_OBS_ANG)));
            observations.push((i, j, ScenePlane {
                normal: tilt.rotate(local.normal),
                c: local.c + gauss(&mut rng, SIGMA_OBS_D),
            }));
        }
    }

    // Initial plane guesses: each plane's first observation, carried back
    // to the world frame through the (drifted) initial pose that saw it.
    let init_planes: Vec<ScenePlane> = (0..gt_planes.len())
        .map(|j| {
            let &(i, _, local) = observations.iter().find(|&&(_, oj, _)| oj == j).unwrap();
            local.transform(init_poses[i])
        })
        .collect();

    let mut world = World {
        poses: refs::Vec::new(),
        planes: refs::Vec::new(),
        odometry: std::vec::Vec::new(),
        observations: std::vec::Vec::new(),
    };
    for (k, p) in init_poses.iter().enumerate() {
        // The first pose is the gauge: fixed, everything else is relative
        // to it.
        let fixed = k == 0;
        world.poses.push(Pose {
            pos: if fixed { Param::fixed(p.pos) } else { Param::new(p.pos) },
            q: if fixed { QuaternionParam::fixed(p.q) } else { QuaternionParam::new(p.q) },
            hb: SelfBlock::new(),
        });
    }
    for plane in &init_planes {
        world.planes.push(PlaneLandmark {
            normal: UnitVec::new(plane.normal),
            c: Param::new(plane.c),
            hb: SelfBlock::new(),
        });
    }
    for &(i, rel) in &measurements {
        world.odometry.push(Odometry {
            a: world.poses.ref_at(i as u32),
            b: world.poses.ref_at((i + 1) as u32),
            measured_translation: rel.pos,
            measured_rotation_transposed: rel.q.rotation_matrix().transpose(),
            translation_weight: 1.0 / SIGMA_ODO_T,
            rotation_weight: 1.0 / SIGMA_ODO_R,
            hb: CrossBlock::new(),
        });
    }
    for &(i, j, local) in &observations {
        world.observations.push(Observation {
            p: world.poses.ref_at(i as u32),
            l: world.planes.ref_at(j as u32),
            measured_normal: local.normal,
            measured_c: local.c,
            azimuth_weight: 1.0 / SIGMA_OBS_ANG,
            elevation_weight: 1.0 / SIGMA_OBS_ANG,
            distance_weight: 1.0 / SIGMA_OBS_D,
            hb: CrossBlock::new(),
        });
    }
    (world, gt_poses, gt_planes)
}

// ---------------------------------------------------------------------------

fn main() {
    let (mut world, gt_poses, gt_planes) = build_world();
    println!(
        "scene: {} poses, {} planes, {} odometry pairs, {} observations",
        world.poses.len(), world.planes.len(),
        world.odometry.len(), world.observations.len(),
    );

    let drift: f64 = world.poses.iter().zip(&gt_poses)
        .map(|(p, gt)| { let d = p.pos.value - gt.pos; d * d })
        .sum::<f64>();
    println!("initial pose RMS vs ground truth: {:.3} m (odometry drift)",
        (drift / N_POSES as f64).sqrt());

    let cfg = LmConfig::well_conditioned().with_verbose(true);
    let result = world.solve_sparse(&cfg);
    println!("cost {:.4} -> {:.4} in {} iterations ({:?})",
        result.start_cost, result.end_cost, result.iterations, result.status);

    let rms: f64 = world.poses.iter().zip(&gt_poses)
        .map(|(p, gt)| { let d = p.pos.value - gt.pos; d * d })
        .sum::<f64>();
    println!("final pose RMS vs ground truth: {:.3} m",
        (rms / N_POSES as f64).sqrt());

    // The optimized directions come back through Component::finish: read
    // `normal.unit` like any other field.
    let mut worst_angle: f64 = 0.0;
    let mut worst_c: f64 = 0.0;
    for (lm, gt) in world.planes.iter().zip(&gt_planes) {
        let dot = (lm.normal.unit * gt.normal).clamp(-1.0, 1.0);
        worst_angle = worst_angle.max(dot.acos().to_degrees());
        worst_c = worst_c.max((lm.c.value - gt.c).abs());
    }
    println!("plane landmarks: worst normal error {:.3} deg, worst distance error {:.3} m",
        worst_angle, worst_c);
}
