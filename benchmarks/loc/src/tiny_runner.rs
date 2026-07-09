// tiny-solver runner for the localization benchmark. All four factor types
// are built through tiny-solver's public Factor<T> API (dual-number autodiff
// over T: RealField). Poses are plain 6-vectors [x, y, z, roll, pitch, yaw]
// with the rotation built from the euler angles inside each residual,
// matching arael's SimpleEulerAngleParam exactly. Landmarks are FIXED
// constants baked into the bearing factor (single-variable over the pose),
// so every residual equals scene::reference_cost.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::scene::{Scene, Solution};
use arael::matrix::matrix3f;
use arael::vect::{vect3d, vect3f};
use tiny_solver::factors::Factor;
use tiny_solver::na;
use tiny_solver::optimizer::Optimizer;

fn m2a(m: &matrix3f) -> [[f64; 3]; 3] {
    [[m[0].x as f64, m[0].y as f64, m[0].z as f64],
     [m[1].x as f64, m[1].y as f64, m[1].z as f64],
     [m[2].x as f64, m[2].y as f64, m[2].z as f64]]
}
fn v2a(v: vect3f) -> [f64; 3] { [v.x as f64, v.y as f64, v.z as f64] }

fn tf<T: na::RealField>(x: f64) -> T { T::from_f64(x).unwrap() }

fn mat<T: na::RealField>(a: &[[f64; 3]; 3]) -> na::Matrix3<T> {
    na::Matrix3::new(
        tf(a[0][0]), tf(a[0][1]), tf(a[0][2]),
        tf(a[1][0]), tf(a[1][1]), tf(a[1][2]),
        tf(a[2][0]), tf(a[2][1]), tf(a[2][2]))
}
fn vec3<T: na::RealField>(a: &[f64; 3]) -> na::Vector3<T> {
    na::Vector3::new(tf(a[0]), tf(a[1]), tf(a[2]))
}

fn euler_to_rot<T: na::RealField>(roll: T, pitch: T, yaw: T) -> na::Matrix3<T> {
    let (sx, cx) = (roll.clone().sin(), roll.cos());
    let (sy, cy) = (pitch.clone().sin(), pitch.cos());
    let (sz, cz) = (yaw.clone().sin(), yaw.cos());
    na::Matrix3::new(
        cy.clone() * cz.clone(),
        -cx.clone() * sz.clone() + cz.clone() * sx.clone() * sy.clone(),
        cx.clone() * cz.clone() * sy.clone() + sx.clone() * sz.clone(),
        cy.clone() * sz.clone(),
        cx.clone() * cz.clone() + sx.clone() * sy.clone() * sz.clone(),
        cx.clone() * sy.clone() * sz.clone() - cz.clone() * sx.clone(),
        -sy.clone(),
        cy.clone() * sx,
        cx * cy)
}

fn rot_to_euler<T: na::RealField>(m: &na::Matrix3<T>) -> [T; 3] {
    let pitch = -m[(2, 0)].clone().asin();
    let roll = m[(2, 1)].clone().atan2(m[(2, 2)].clone());
    let yaw = m[(1, 0)].clone().atan2(m[(0, 0)].clone());
    [roll, pitch, yaw]
}

fn pose_rot<T: na::RealField>(p: &na::DVector<T>) -> na::Matrix3<T> {
    euler_to_rot(p[3].clone(), p[4].clone(), p[5].clone())
}
fn pose_pos<T: na::RealField>(p: &na::DVector<T>) -> na::Vector3<T> {
    na::Vector3::new(p[0].clone(), p[1].clone(), p[2].clone())
}

// -- the four factors --

#[derive(Clone)]
struct DriftFactor { prior_pos: [f64; 3], prior_ea: [f64; 3], pi: f64, ei: f64 }
impl<T: na::RealField> Factor<T> for DriftFactor {
    fn residual_func(&self, p: &[na::DVector<T>]) -> na::DVector<T> {
        let pi: T = tf(self.pi);
        let ei: T = tf(self.ei);
        na::DVector::from_vec(vec![
            (p[0][0].clone() - tf(self.prior_pos[0])) * pi.clone(),
            (p[0][1].clone() - tf(self.prior_pos[1])) * pi.clone(),
            (p[0][2].clone() - tf(self.prior_pos[2])) * pi,
            (p[0][3].clone() - tf(self.prior_ea[0])) * ei.clone(),
            (p[0][4].clone() - tf(self.prior_ea[1])) * ei.clone(),
            (p[0][5].clone() - tf(self.prior_ea[2])) * ei])
    }
}

#[derive(Clone)]
struct TiltFactor { roll: f64, pitch: f64, isigma: f64 }
impl<T: na::RealField> Factor<T> for TiltFactor {
    fn residual_func(&self, p: &[na::DVector<T>]) -> na::DVector<T> {
        let ti: T = tf(self.isigma);
        na::DVector::from_vec(vec![
            (p[0][3].clone() - tf(self.roll)) * ti.clone(),
            (p[0][4].clone() - tf(self.pitch)) * ti])
    }
}

// params: [pose(6)]; the landmark is a fixed constant.
#[derive(Clone)]
struct BearingFactor { lm: [f64; 3], mf2r: [[f64; 3]; 3], camera_pos: [f64; 3], isigma: [f64; 2], scale: f64 }
impl<T: na::RealField> Factor<T> for BearingFactor {
    fn residual_func(&self, p: &[na::DVector<T>]) -> na::DVector<T> {
        let lm = vec3::<T>(&self.lm);
        let mr2w = pose_rot(&p[0]);
        let lm_r = mr2w.transpose() * (lm - pose_pos(&p[0]));
        let r_r = lm_r - vec3::<T>(&self.camera_pos);
        let r_f = mat::<T>(&self.mf2r).transpose() * r_r;
        let scale: T = tf(self.scale);
        let plain1 = r_f[1].clone().atan2(r_f[0].clone()) * tf::<T>(self.isigma[0]) * scale.clone();
        let plain2 = r_f[2].clone().atan2(r_f[0].clone()) * tf::<T>(self.isigma[1]) * scale;
        na::DVector::from_vec(vec![plain1, plain2])
    }
}

// params: [prev_pose(6), cur_pose(6)]
#[derive(Clone)]
struct OdoFactor {
    delta_pos: [f64; 3], delta_ea: [f64; 3],
    pos_cov_r: [[f64; 3]; 3], pos_isigma: [f64; 3],
    ea_cov_r: [[f64; 3]; 3], ea_isigma: [f64; 3],
}
impl<T: na::RealField> Factor<T> for OdoFactor {
    fn residual_func(&self, p: &[na::DVector<T>]) -> na::DVector<T> {
        let mr2w_prev = pose_rot(&p[0]);
        let mr2w_cur = pose_rot(&p[1]);
        let pos_diff = mr2w_prev.transpose() * (pose_pos(&p[1]) - pose_pos(&p[0]));
        let pos_err = pos_diff - vec3::<T>(&self.delta_pos);
        let pos_w = mat::<T>(&self.pos_cov_r).transpose() * pos_err;
        let dea = &self.delta_ea;
        let expected = mr2w_prev * euler_to_rot::<T>(tf(dea[0]), tf(dea[1]), tf(dea[2]));
        let error_rot = expected.transpose() * mr2w_cur;
        let ea_err = rot_to_euler(&error_rot);
        let ea_w = mat::<T>(&self.ea_cov_r).transpose()
            * na::Vector3::new(ea_err[0].clone(), ea_err[1].clone(), ea_err[2].clone());
        let pi = vec3::<T>(&self.pos_isigma);
        let ei = vec3::<T>(&self.ea_isigma);
        na::DVector::from_vec(vec![
            pos_w[0].clone() * pi[0].clone(),
            pos_w[1].clone() * pi[1].clone(),
            pos_w[2].clone() * pi[2].clone(),
            ea_w[0].clone() * ei[0].clone(),
            ea_w[1].clone() * ei[1].clone(),
            ea_w[2].clone() * ei[2].clone()])
    }
}

// Iteration counting via tiny-solver's log trace.
static ITER_COUNT: AtomicUsize = AtomicUsize::new(0);
struct IterCounter;
impl log::Log for IterCounter {
    fn enabled(&self, m: &log::Metadata) -> bool { m.target().starts_with("tiny_solver") }
    fn log(&self, r: &log::Record) {
        if r.target().starts_with("tiny_solver")
            && std::fmt::format(*r.args()).starts_with("iter:") {
            ITER_COUNT.fetch_add(1, Ordering::Relaxed);
        }
    }
    fn flush(&self) {}
}
pub fn install_iter_counter() {
    log::set_logger(&IterCounter).expect("logger already set");
    log::set_max_level(log::LevelFilter::Trace);
}

fn pv(pos: vect3f, ea: vect3f) -> na::DVector<f64> {
    na::DVector::from_vec(vec![
        pos.x as f64, pos.y as f64, pos.z as f64,
        ea.x as f64, ea.y as f64, ea.z as f64])
}

fn build(scene: &Scene) -> (tiny_solver::Problem, HashMap<String, na::DVector<f64>>) {
    let mut problem = tiny_solver::Problem::new();
    let mut init = HashMap::new();

    for (i, p) in scene.poses.iter().enumerate() {
        let name = format!("p{}", i);
        init.insert(name.clone(), pv(p.init_pos, p.init_ea));
        problem.add_residual_block(6, &[&name], Box::new(DriftFactor {
            prior_pos: v2a(p.init_pos), prior_ea: v2a(p.init_ea),
            pi: scene.drift_pos_isigma as f64, ei: scene.drift_ea_isigma as f64,
        }), None);
        problem.add_residual_block(2, &[&name], Box::new(TiltFactor {
            roll: p.tilt_roll as f64, pitch: p.tilt_pitch as f64, isigma: scene.tilt_isigma as f64,
        }), None);
    }
    for f in &scene.frines {
        problem.add_residual_block(2,
            &[&format!("p{}", f.pose)],
            Box::new(BearingFactor {
                lm: v2a(scene.landmarks[f.landmark as usize]),
                mf2r: m2a(&f.mf2r), camera_pos: v2a(f.camera_pos),
                isigma: [f.isigma.x as f64, f.isigma.y as f64],
                scale: scene.frine_isigma_scale as f64,
            }), None);
    }
    for o in &scene.odo {
        problem.add_residual_block(6,
            &[&format!("p{}", o.prev), &format!("p{}", o.cur)],
            Box::new(OdoFactor {
                delta_pos: v2a(o.delta_pos), delta_ea: v2a(o.delta_ea),
                pos_cov_r: m2a(&o.pos_cov_r), pos_isigma: v2a(o.pos_cov_isigma),
                ea_cov_r: m2a(&o.ea_cov_r), ea_isigma: v2a(o.ea_cov_isigma),
            }), None);
    }
    (problem, init)
}

fn extract(scene: &Scene, values: &HashMap<String, na::DVector<f64>>) -> Solution {
    Solution {
        poses: (0..scene.poses.len()).map(|i| {
            let v = &values[&format!("p{}", i)];
            (vect3d::new(v[0], v[1], v[2]), vect3d::new(v[3], v[4], v[5]))
        }).collect(),
    }
}

/// Sum of squared residuals at the initial estimate -- the harness asserts
/// this equals scene::reference_cost.
pub fn initial_cost(scene: &Scene) -> f64 {
    let sq = |r: na::DVector<f64>| r.iter().map(|x| x * x).sum::<f64>();
    let mut cost = 0.0;
    for p in &scene.poses {
        let pb = pv(p.init_pos, p.init_ea);
        cost += sq(DriftFactor { prior_pos: v2a(p.init_pos), prior_ea: v2a(p.init_ea),
            pi: scene.drift_pos_isigma as f64, ei: scene.drift_ea_isigma as f64 }
            .residual_func(&[pb.clone()]));
        cost += sq(TiltFactor { roll: p.tilt_roll as f64, pitch: p.tilt_pitch as f64,
            isigma: scene.tilt_isigma as f64 }.residual_func(&[pb]));
    }
    for f in &scene.frines {
        let pd = &scene.poses[f.pose as usize];
        let pb = pv(pd.init_pos, pd.init_ea);
        cost += sq(BearingFactor { lm: v2a(scene.landmarks[f.landmark as usize]),
            mf2r: m2a(&f.mf2r), camera_pos: v2a(f.camera_pos),
            isigma: [f.isigma.x as f64, f.isigma.y as f64],
            scale: scene.frine_isigma_scale as f64 }.residual_func(&[pb]));
    }
    for o in &scene.odo {
        let a = &scene.poses[o.prev as usize];
        let b = &scene.poses[o.cur as usize];
        cost += sq(OdoFactor { delta_pos: v2a(o.delta_pos), delta_ea: v2a(o.delta_ea),
            pos_cov_r: m2a(&o.pos_cov_r), pos_isigma: v2a(o.pos_cov_isigma),
            ea_cov_r: m2a(&o.ea_cov_r), ea_isigma: v2a(o.ea_cov_isigma) }
            .residual_func(&[pv(a.init_pos, a.init_ea), pv(b.init_pos, b.init_ea)]));
    }
    cost
}

pub struct RunOut {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    pub iterations: usize,
    pub solution: Solution,
}

fn options(max_iteration: usize) -> tiny_solver::OptimizerOptions {
    tiny_solver::OptimizerOptions {
        max_iteration,
        min_abs_error_decrease_threshold: 0.0,
        min_rel_error_decrease_threshold: 1e-5,
        ..Default::default()
    }
}

pub fn run_lm(scene: &Scene) -> RunOut {
    let (problem, init) = build(scene);
    let radius = std::env::var("TINY_RADIUS0").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(1e12);
    let optimize = |max_iter: usize| -> (f64, usize, HashMap<String, na::DVector<f64>>) {
        let before = ITER_COUNT.load(Ordering::Relaxed);
        let t0 = std::time::Instant::now();
        let result = tiny_solver::LevenbergMarquardtOptimizer::new(1e-6, 1e32, radius)
            .optimize(&problem, &init, Some(options(max_iter)));
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        (ms, ITER_COUNT.load(Ordering::Relaxed) - before, result.expect("tiny returned None"))
    };
    let max_iter = std::env::var("TINY_MAXITER").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(200);
    let (first_iter_ms, _, _) = optimize(1);
    let (solve_ms, iterations, values) = optimize(max_iter);
    RunOut { solve_ms, first_iter_ms, iterations, solution: extract(scene, &values) }
}
