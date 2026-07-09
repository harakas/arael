// factrs runner for the localization benchmark. Four factor types are
// custom Residual1/Residual2 impls (factrs's dual-number ForwardProp
// autodiff). Poses are VectorVar<6> = [x, y, z, roll, pitch, yaw] (additive
// retraction, no manifold), matching arael's SimpleEulerAngleParam exactly.
// Landmarks are FIXED constants baked into the bearing residual (single
// variable over the pose), so every residual equals scene::reference_cost.
// Whitening is baked into the residual (plain Gaussian, no robust kernel).

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::scene::{Scene, Solution};
use arael::matrix::matrix3f;
use arael::vect::{vect3d, vect3f};
use factrs::assign_symbols;
use factrs::core::{Graph, LevenMarquardt, Values};
use factrs::dtype;
use factrs::fac;
use factrs::linalg::{Const, ForwardProp, Matrix3, Numeric, Vector3, VectorX};
use factrs::optimizers::{BaseOptParams, LevenParams, OptError, OptObserver};
use factrs::residuals::{Residual1, Residual2};
use factrs::traits::Optimizer;
use factrs::variables::{VectorVar, VectorVar6};

assign_symbols!(P: VectorVar6);

fn m2a(m: &matrix3f) -> [[f64; 3]; 3] {
    [[m[0].x as f64, m[0].y as f64, m[0].z as f64],
     [m[1].x as f64, m[1].y as f64, m[1].z as f64],
     [m[2].x as f64, m[2].y as f64, m[2].z as f64]]
}
fn v2a(v: vect3f) -> [f64; 3] { [v.x as f64, v.y as f64, v.z as f64] }

fn mat<T: Numeric>(a: &[[f64; 3]; 3]) -> Matrix3<T> {
    Matrix3::new(
        T::from(a[0][0]), T::from(a[0][1]), T::from(a[0][2]),
        T::from(a[1][0]), T::from(a[1][1]), T::from(a[1][2]),
        T::from(a[2][0]), T::from(a[2][1]), T::from(a[2][2]))
}
fn vec3<T: Numeric>(a: &[f64; 3]) -> Vector3<T> {
    Vector3::new(T::from(a[0]), T::from(a[1]), T::from(a[2]))
}

fn euler_to_rot<T: Numeric>(roll: T, pitch: T, yaw: T) -> Matrix3<T> {
    let (sx, cx) = (roll.sin(), roll.cos());
    let (sy, cy) = (pitch.sin(), pitch.cos());
    let (sz, cz) = (yaw.sin(), yaw.cos());
    Matrix3::new(
        cy * cz,            -cx * sz + cz * sx * sy,  cx * cz * sy + sx * sz,
        cy * sz,             cx * cz + sx * sy * sz,  cx * sy * sz - cz * sx,
        -sy,                 cy * sx,                 cx * cy)
}
fn rot_to_euler<T: Numeric>(m: &Matrix3<T>) -> [T; 3] {
    [m[(2, 1)].atan2(m[(2, 2)]), -m[(2, 0)].asin(), m[(1, 0)].atan2(m[(0, 0)])]
}
fn pose_rot<T: Numeric>(p: &VectorVar<6, T>) -> Matrix3<T> {
    euler_to_rot(p.0[3], p.0[4], p.0[5])
}
fn pose_pos<T: Numeric>(p: &VectorVar<6, T>) -> Vector3<T> {
    Vector3::new(p.0[0], p.0[1], p.0[2])
}

// -- the four residuals --

#[derive(Clone, Debug)]
struct DriftRes { prior: [f64; 6], pi: f64, ei: f64 }
#[factrs::mark]
impl Residual1 for DriftRes {
    type DimIn = Const<6>;
    type DimOut = Const<6>;
    type V1 = VectorVar6;
    type Differ = ForwardProp<Const<6>>;
    fn residual1<T: Numeric>(&self, p: VectorVar<6, T>) -> VectorX<T> {
        VectorX::from_vec(vec![
            (p.0[0] - T::from(self.prior[0])) * T::from(self.pi),
            (p.0[1] - T::from(self.prior[1])) * T::from(self.pi),
            (p.0[2] - T::from(self.prior[2])) * T::from(self.pi),
            (p.0[3] - T::from(self.prior[3])) * T::from(self.ei),
            (p.0[4] - T::from(self.prior[4])) * T::from(self.ei),
            (p.0[5] - T::from(self.prior[5])) * T::from(self.ei)])
    }
}

#[derive(Clone, Debug)]
struct TiltRes { roll: f64, pitch: f64, isigma: f64 }
#[factrs::mark]
impl Residual1 for TiltRes {
    type DimIn = Const<6>;
    type DimOut = Const<2>;
    type V1 = VectorVar6;
    type Differ = ForwardProp<Const<6>>;
    fn residual1<T: Numeric>(&self, p: VectorVar<6, T>) -> VectorX<T> {
        VectorX::from_vec(vec![
            (p.0[3] - T::from(self.roll)) * T::from(self.isigma),
            (p.0[4] - T::from(self.pitch)) * T::from(self.isigma)])
    }
}

// Bearing to a FIXED landmark -- single variable (the pose).
#[derive(Clone, Debug)]
struct BearingRes { lm: [f64; 3], mf2r: [[f64; 3]; 3], camera_pos: [f64; 3], isigma: [f64; 2], scale: f64 }
#[factrs::mark]
impl Residual1 for BearingRes {
    type DimIn = Const<6>;
    type DimOut = Const<2>;
    type V1 = VectorVar6;
    type Differ = ForwardProp<Const<6>>;
    fn residual1<T: Numeric>(&self, pose: VectorVar<6, T>) -> VectorX<T> {
        let mr2w = pose_rot(&pose);
        let lm_v = vec3::<T>(&self.lm);
        let lm_r = mr2w.transpose() * (lm_v - pose_pos(&pose));
        let r_r = lm_r - vec3::<T>(&self.camera_pos);
        let r_f = mat::<T>(&self.mf2r).transpose() * r_r;
        let plain1 = r_f[1].atan2(r_f[0]) * T::from(self.isigma[0] * self.scale);
        let plain2 = r_f[2].atan2(r_f[0]) * T::from(self.isigma[1] * self.scale);
        VectorX::from_vec(vec![plain1, plain2])
    }
}

#[derive(Clone, Debug)]
struct OdoRes {
    delta_pos: [f64; 3], delta_ea: [f64; 3],
    pos_cov_r: [[f64; 3]; 3], pos_isigma: [f64; 3],
    ea_cov_r: [[f64; 3]; 3], ea_isigma: [f64; 3],
}
#[factrs::mark]
impl Residual2 for OdoRes {
    type DimIn = Const<12>;
    type DimOut = Const<6>;
    type V1 = VectorVar6;
    type V2 = VectorVar6;
    type Differ = ForwardProp<Const<12>>;
    fn residual2<T: Numeric>(&self, prev: VectorVar<6, T>, cur: VectorVar<6, T>) -> VectorX<T> {
        let mr2w_prev = pose_rot(&prev);
        let mr2w_cur = pose_rot(&cur);
        let pos_diff = mr2w_prev.transpose() * (pose_pos(&cur) - pose_pos(&prev));
        let pos_err = pos_diff - vec3::<T>(&self.delta_pos);
        let pos_w = mat::<T>(&self.pos_cov_r).transpose() * pos_err;
        let d = &self.delta_ea;
        let expected = mr2w_prev * euler_to_rot::<T>(T::from(d[0]), T::from(d[1]), T::from(d[2]));
        let error_rot = expected.transpose() * mr2w_cur;
        let ea_err = rot_to_euler(&error_rot);
        let ea_w = mat::<T>(&self.ea_cov_r).transpose() * Vector3::new(ea_err[0], ea_err[1], ea_err[2]);
        let pi = vec3::<T>(&self.pos_isigma);
        let ei = vec3::<T>(&self.ea_isigma);
        VectorX::from_vec(vec![
            pos_w[0] * pi[0], pos_w[1] * pi[1], pos_w[2] * pi[2],
            ea_w[0] * ei[0], ea_w[1] * ei[1], ea_w[2] * ei[2]])
    }
}

static STEPS: AtomicUsize = AtomicUsize::new(0);
struct StepCounter;
impl OptObserver for StepCounter {
    fn on_step(&self, _v: &Values, _t: i64) { STEPS.fetch_add(1, Ordering::Relaxed); }
}

fn pv(pos: vect3f, ea: vect3f) -> VectorVar6 {
    VectorVar(factrs::linalg::Vector::<6, dtype>::new(
        pos.x as f64, pos.y as f64, pos.z as f64, ea.x as f64, ea.y as f64, ea.z as f64))
}

fn build(scene: &Scene) -> (Graph, Values) {
    let mut graph = Graph::new();
    let mut values = Values::new();
    for (i, p) in scene.poses.iter().enumerate() {
        values.insert(P(i as u32), pv(p.init_pos, p.init_ea));
        graph.add_factor(fac![DriftRes {
            prior: [p.init_pos.x as f64, p.init_pos.y as f64, p.init_pos.z as f64,
                    p.init_ea.x as f64, p.init_ea.y as f64, p.init_ea.z as f64],
            pi: scene.drift_pos_isigma as f64, ei: scene.drift_ea_isigma as f64 }, P(i as u32)]);
        graph.add_factor(fac![TiltRes { roll: p.tilt_roll as f64, pitch: p.tilt_pitch as f64,
            isigma: scene.tilt_isigma as f64 }, P(i as u32)]);
    }
    for f in &scene.frines {
        graph.add_factor(fac![BearingRes { lm: v2a(scene.landmarks[f.landmark as usize]),
            mf2r: m2a(&f.mf2r), camera_pos: v2a(f.camera_pos),
            isigma: [f.isigma.x as f64, f.isigma.y as f64], scale: scene.frine_isigma_scale as f64 }, P(f.pose)]);
    }
    for o in &scene.odo {
        graph.add_factor(fac![OdoRes { delta_pos: v2a(o.delta_pos), delta_ea: v2a(o.delta_ea),
            pos_cov_r: m2a(&o.pos_cov_r), pos_isigma: v2a(o.pos_cov_isigma),
            ea_cov_r: m2a(&o.ea_cov_r), ea_isigma: v2a(o.ea_cov_isigma) }, (P(o.prev), P(o.cur))]);
    }
    (graph, values)
}

fn extract(scene: &Scene, v: &Values) -> Solution {
    Solution {
        poses: (0..scene.poses.len()).map(|i| {
            let p: &VectorVar6 = v.get(P(i as u32)).unwrap();
            (vect3d::new(p.0[0], p.0[1], p.0[2]), vect3d::new(p.0[3], p.0[4], p.0[5]))
        }).collect(),
    }
}

pub struct RunOut {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    pub iterations: usize,
    pub solution: Solution,
}

fn base(max_iterations: usize) -> BaseOptParams {
    BaseOptParams {
        max_iterations,
        error_tol_relative: 1e-5,
        error_tol_absolute: 1e-5,
        ..Default::default()
    }
}

/// Sum of squared residuals at the init -- harness cross-check.
pub fn initial_cost(scene: &Scene) -> f64 {
    let sq = |r: VectorX<f64>| r.iter().map(|x| x * x).sum::<f64>();
    let mut cost = 0.0;
    for p in &scene.poses {
        let pb = pv(p.init_pos, p.init_ea);
        cost += sq(DriftRes {
            prior: [p.init_pos.x as f64, p.init_pos.y as f64, p.init_pos.z as f64,
                    p.init_ea.x as f64, p.init_ea.y as f64, p.init_ea.z as f64],
            pi: scene.drift_pos_isigma as f64, ei: scene.drift_ea_isigma as f64 }.residual1(pb.clone()));
        cost += sq(TiltRes { roll: p.tilt_roll as f64, pitch: p.tilt_pitch as f64,
            isigma: scene.tilt_isigma as f64 }.residual1(pb));
    }
    for f in &scene.frines {
        let pd = &scene.poses[f.pose as usize];
        let pb = pv(pd.init_pos, pd.init_ea);
        cost += sq(BearingRes { lm: v2a(scene.landmarks[f.landmark as usize]),
            mf2r: m2a(&f.mf2r), camera_pos: v2a(f.camera_pos),
            isigma: [f.isigma.x as f64, f.isigma.y as f64], scale: scene.frine_isigma_scale as f64 }.residual1(pb));
    }
    for o in &scene.odo {
        let mk = |pd: &crate::scene::PoseData| VectorVar(factrs::linalg::Vector::<6, dtype>::new(
            pd.init_pos.x as f64, pd.init_pos.y as f64, pd.init_pos.z as f64,
            pd.init_ea.x as f64, pd.init_ea.y as f64, pd.init_ea.z as f64));
        cost += sq(OdoRes { delta_pos: v2a(o.delta_pos), delta_ea: v2a(o.delta_ea),
            pos_cov_r: m2a(&o.pos_cov_r), pos_isigma: v2a(o.pos_cov_isigma),
            ea_cov_r: m2a(&o.ea_cov_r), ea_isigma: v2a(o.ea_cov_isigma) }
            .residual2(mk(&scene.poses[o.prev as usize]), mk(&scene.poses[o.cur as usize])));
    }
    cost
}

pub fn run(scene: &Scene) -> RunOut {
    let optimize = |max_iter: usize| -> (f64, usize, Values) {
        let (graph, init) = build(scene);
        let before = STEPS.load(Ordering::Relaxed);
        let params = LevenParams { base: base(max_iter), ..Default::default() };
        let mut opt = LevenMarquardt::new(params, graph);
        opt.observers_mut().add(StepCounter);
        let t0 = std::time::Instant::now();
        let result = opt.optimize(init);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        let values = match result {
            Ok(v) => v,
            Err(OptError::MaxIterations(v)) => v,
            Err(e) => panic!("factrs failed: {:?}", e),
        };
        (ms, STEPS.load(Ordering::Relaxed) - before, values)
    };
    let (first_iter_ms, _, _) = optimize(1);
    let (solve_ms, iterations, values) = optimize(200);
    RunOut { solve_ms, first_iter_ms, iterations, solution: extract(scene, &values) }
}
