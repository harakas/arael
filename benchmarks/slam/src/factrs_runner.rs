// factrs runner for the heterogeneous SLAM benchmark. All six factor
// types are custom Residual1/Residual2 impls (factrs's dual-number
// ForwardProp autodiff); its solver and linear algebra remain factrs's
// own. Poses are VectorVar<6> = [x, y, z, roll, pitch, yaw] (additive
// retraction, no manifold), matching arael's SimpleEulerAngleParam
// exactly; landmarks are VectorVar<3>. The rotation convention and the
// odometry euler extraction reproduce arael's matrix3 twins, so every
// residual equals scene::reference_cost. Whitening is baked into the
// residual (plain Gaussian, no robust kernel); factors carry unit noise.


use crate::scene::{Scene, Solution};
use arael::matrix::matrix3f;
use arael::vect::{vect3d, vect3f};
use bench_harness::factrs::{counts, since, CountingSolver, StepCounter};
use factrs::assign_symbols;
use factrs::core::{Graph, LevenMarquardt, Values};
use factrs::dtype;
use factrs::fac;
use factrs::linalg::{Const, ForwardProp, Matrix3, Numeric, Vector3, VectorX};
use factrs::optimizers::{BaseOptParams, LevenParams, OptError};
use factrs::residuals::{Residual1, Residual2};
use factrs::traits::Optimizer;
use factrs::variables::{VectorVar, VectorVar3, VectorVar6};

assign_symbols!(P: VectorVar6; L: VectorVar3);

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

// -- the six residuals --

#[derive(Clone, Debug)]
struct GpsRes { pos: [f64; 3], cov_r: [[f64; 3]; 3], isigma: [f64; 3] }
#[factrs::mark]
impl Residual1 for GpsRes {
    type DimIn = Const<6>;
    type DimOut = Const<3>;
    type V1 = VectorVar6;
    type Differ = ForwardProp<Const<6>>;
    fn residual1<T: Numeric>(&self, p: VectorVar<6, T>) -> VectorX<T> {
        let raw = pose_pos(&p) - vec3(&self.pos);
        let rt = mat::<T>(&self.cov_r).transpose() * raw;
        VectorX::from_vec(vec![
            rt[0] * T::from(self.isigma[0]),
            rt[1] * T::from(self.isigma[1]),
            rt[2] * T::from(self.isigma[2])])
    }
}

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

#[derive(Clone, Debug)]
struct LmDriftRes { prior: [f64; 3], isigma: f64 }
#[factrs::mark]
impl Residual1 for LmDriftRes {
    type DimIn = Const<3>;
    type DimOut = Const<3>;
    type V1 = VectorVar3;
    type Differ = ForwardProp<Const<3>>;
    fn residual1<T: Numeric>(&self, l: VectorVar<3, T>) -> VectorX<T> {
        VectorX::from_vec(vec![
            (l.0[0] - T::from(self.prior[0])) * T::from(self.isigma),
            (l.0[1] - T::from(self.prior[1])) * T::from(self.isigma),
            (l.0[2] - T::from(self.prior[2])) * T::from(self.isigma)])
    }
}

#[derive(Clone, Debug)]
struct BearingRes { mf2r: [[f64; 3]; 3], camera_pos: [f64; 3], isigma: [f64; 2], scale: f64 }
#[factrs::mark]
impl Residual2 for BearingRes {
    type DimIn = Const<9>;
    type DimOut = Const<2>;
    type V1 = VectorVar3;
    type V2 = VectorVar6;
    type Differ = ForwardProp<Const<9>>;
    fn residual2<T: Numeric>(&self, lm: VectorVar<3, T>, pose: VectorVar<6, T>) -> VectorX<T> {
        let mr2w = pose_rot(&pose);
        let lm_v = Vector3::new(lm.0[0], lm.0[1], lm.0[2]);
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


fn pv(pos: vect3f, ea: vect3f) -> VectorVar6 {
    VectorVar(factrs::linalg::Vector::<6, dtype>::new(
        pos.x as f64, pos.y as f64, pos.z as f64, ea.x as f64, ea.y as f64, ea.z as f64))
}

fn build(scene: &Scene) -> (Graph, Values) {
    let mut graph = Graph::new();
    let mut values = Values::new();
    for (i, p) in scene.poses.iter().enumerate() {
        values.insert(P(i as u32), pv(p.init_pos, p.init_ea));
        let g = p.gps.as_ref().unwrap();
        graph.add_factor(fac![GpsRes { pos: v2a(g.pos), cov_r: m2a(&g.cov_r),
            isigma: v2a(g.cov_isigma) }, P(i as u32)]);
        graph.add_factor(fac![DriftRes {
            prior: [p.init_pos.x as f64, p.init_pos.y as f64, p.init_pos.z as f64,
                    p.init_ea.x as f64, p.init_ea.y as f64, p.init_ea.z as f64],
            pi: scene.drift_pos_isigma as f64, ei: scene.drift_ea_isigma as f64 }, P(i as u32)]);
        graph.add_factor(fac![TiltRes { roll: p.tilt_roll as f64, pitch: p.tilt_pitch as f64,
            isigma: scene.tilt_isigma as f64 }, P(i as u32)]);
    }
    for (i, init) in scene.landmarks_init.iter().enumerate() {
        values.insert(L(i as u32), VectorVar3::new(init.x as f64, init.y as f64, init.z as f64));
        graph.add_factor(fac![LmDriftRes { prior: v2a(*init),
            isigma: scene.drift_lm_isigma as f64 }, L(i as u32)]);
    }
    for f in &scene.frines {
        graph.add_factor(fac![BearingRes { mf2r: m2a(&f.mf2r), camera_pos: v2a(f.camera_pos),
            isigma: [f.isigma.x as f64, f.isigma.y as f64], scale: scene.frine_isigma_scale as f64 }, (L(f.landmark), P(f.pose))]);
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
        landmarks: (0..scene.landmarks_init.len()).map(|i| {
            let l: &VectorVar3 = v.get(L(i as u32)).unwrap();
            vect3d::new(l.0[0], l.0[1], l.0[2])
        }).collect(),
    }
}

pub type RunOut = bench_harness::table::Row<Solution>;


fn base(max_iterations: usize) -> BaseOptParams {
    // 1e-5 relative/absolute -- the shared termination class. factrs's
    // default LM damping (lambda starts 1e-10) is already near-Gauss-Newton
    // and problem-appropriate for this well-initialized graph.
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
        let pb = VectorVar(factrs::linalg::Vector::<6, dtype>::new(
            p.init_pos.x as f64, p.init_pos.y as f64, p.init_pos.z as f64,
            p.init_ea.x as f64, p.init_ea.y as f64, p.init_ea.z as f64));
        let g = p.gps.as_ref().unwrap();
        cost += sq(GpsRes { pos: v2a(g.pos), cov_r: m2a(&g.cov_r), isigma: v2a(g.cov_isigma) }.residual1(pb.clone()));
        cost += sq(DriftRes {
            prior: [p.init_pos.x as f64, p.init_pos.y as f64, p.init_pos.z as f64,
                    p.init_ea.x as f64, p.init_ea.y as f64, p.init_ea.z as f64],
            pi: scene.drift_pos_isigma as f64, ei: scene.drift_ea_isigma as f64 }.residual1(pb.clone()));
        cost += sq(TiltRes { roll: p.tilt_roll as f64, pitch: p.tilt_pitch as f64,
            isigma: scene.tilt_isigma as f64 }.residual1(pb));
    }
    for init in &scene.landmarks_init {
        let lb = VectorVar3::new(init.x as f64, init.y as f64, init.z as f64);
        cost += sq(LmDriftRes { prior: v2a(*init), isigma: scene.drift_lm_isigma as f64 }.residual1(lb));
    }
    for f in &scene.frines {
        let li = scene.landmarks_init[f.landmark as usize];
        let lb = VectorVar3::new(li.x as f64, li.y as f64, li.z as f64);
        let pd = &scene.poses[f.pose as usize];
        let pb = VectorVar(factrs::linalg::Vector::<6, dtype>::new(
            pd.init_pos.x as f64, pd.init_pos.y as f64, pd.init_pos.z as f64,
            pd.init_ea.x as f64, pd.init_ea.y as f64, pd.init_ea.z as f64));
        cost += sq(BearingRes { mf2r: m2a(&f.mf2r), camera_pos: v2a(f.camera_pos),
            isigma: [f.isigma.x as f64, f.isigma.y as f64], scale: scene.frine_isigma_scale as f64 }.residual2(lb, pb));
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
    bench_harness::solver::run(200, |max_iter| {
        // Building the graph is the probe's reset, not the solve -- the clock
        // starts at the optimize() below, the same boundary the C++ runners draw.
        let (graph, init) = build(scene);
        let before = counts();
        let (ms, result) = bench_harness::solver::timed(|| {
            let params = LevenParams { base: base(max_iter), ..Default::default() };
            let mut opt = LevenMarquardt::new(params, graph);
            // factrs keeps its damping retries inside step(): a rejected step
            // multiplies lambda and re-solves the damped system -- another full
            // factorization -- without ever returning. Counting the linear solves
            // recovers them; its observer sees accepted steps only.
            opt.set_solver(CountingSolver::default());
            opt.observers_mut().add(StepCounter);
            opt.optimize(init)
        });
        let values = match result {
            Ok(v) => v,
            Err(OptError::MaxIterations(v)) => v,
            Err(e) => panic!("factrs failed: {:?}", e),
        };
        let (accepted, attempts) = since(before);
        bench_harness::solver::Outcome {
            ms,
            accepted,
            attempts,
            solution: extract(scene, &values),
        }
    })
}
