// factrs runner. Poses are factrs SE3; a plane is a VectorVar<3> holding
// (dy, dz, c): a fixed tangent chart around the plane's INITIAL normal
// (anchor rotation baked into each observation residual) plus the absolute
// distance coefficient. Residuals are the benchmark's shared definitions,
// whitened, through factrs's ForwardProp dual-number autodiff. factrs has
// no fixed variables, so the gauge is a strong prior on pose 0 (zero at the
// start point, so start costs stay comparable).

use bench_harness::factrs::{counts, since, CountingSolver, StepCounter};
use factrs::assign_symbols;
use factrs::core::{Graph, LevenMarquardt, Values};
use factrs::dtype;
use factrs::fac;
use factrs::linalg::{Const, ForwardProp, Numeric, Vector3, VectorX};
use factrs::optimizers::{BaseOptParams, LevenParams, OptError};
use factrs::residuals::Residual2;
use factrs::traits::Optimizer;
use factrs::variables::{VectorVar3, SE3, SO3};

use crate::scene::{Plane, Pose, RawScene, Solution};
use arael::quatern::quaternd;
use arael::vect::vect3d;

assign_symbols!(A: SE3; B: VectorVar3);

fn quat_mul<T: Numeric>(a: &[T; 4], b: &[T; 4]) -> [T; 4] {
    [
        a[3] * b[0] + a[0] * b[3] + a[1] * b[2] - a[2] * b[1],
        a[3] * b[1] - a[0] * b[2] + a[1] * b[3] + a[2] * b[0],
        a[3] * b[2] + a[0] * b[1] - a[1] * b[0] + a[2] * b[3],
        a[3] * b[3] - a[0] * b[0] - a[1] * b[1] - a[2] * b[2],
    ]
}
fn quat_of<T: Numeric>(r: &SO3<T>) -> [T; 4] {
    [r.x(), r.y(), r.z(), r.w()]
}
fn quat_conj<T: Numeric>(q: &[T; 4]) -> [T; 4] {
    [-q[0], -q[1], -q[2], q[3]]
}
fn quat_rotate<T: Numeric>(q: &[T; 4], v: [T; 3]) -> [T; 3] {
    let p = [v[0], v[1], v[2], T::from(0.0)];
    let r = quat_mul(&quat_mul(q, &p), &quat_conj(q));
    [r[0], r[1], r[2]]
}

#[derive(Clone, Debug)]
pub struct OdoRes {
    pub tm: [dtype; 3],
    pub qm: [dtype; 4], // x,y,z,w
    pub wt: dtype,
    pub wr: dtype,
}

#[factrs::mark]
impl Residual2 for OdoRes {
    type Differ = ForwardProp<<Self as Residual2>::DimIn>;
    type V1 = SE3;
    type V2 = SE3;
    type DimIn = Const<12>;
    type DimOut = Const<6>;

    fn residual2<T: Numeric>(&self, a: SE3<T>, b: SE3<T>) -> VectorX<T> {
        let qa = quat_of(a.rot());
        let qb = quat_of(b.rot());
        let qm = [T::from(self.qm[0]), T::from(self.qm[1]), T::from(self.qm[2]), T::from(self.qm[3])];
        let d = [
            b.xyz()[0] - a.xyz()[0],
            b.xyz()[1] - a.xyz()[1],
            b.xyz()[2] - a.xyz()[2],
        ];
        let local = quat_rotate(&quat_conj(&qa), d);
        // err_r = vee((dR - dR^T)/2) = 2 w v of the error quaternion.
        let qe = quat_mul(&quat_conj(&qm), &quat_mul(&quat_conj(&qa), &qb));
        let two_w = T::from(2.0) * qe[3];
        let mut out = VectorX::zeros(6);
        out[0] = (local[0] - T::from(self.tm[0])) * T::from(self.wt);
        out[1] = (local[1] - T::from(self.tm[1])) * T::from(self.wt);
        out[2] = (local[2] - T::from(self.tm[2])) * T::from(self.wt);
        out[3] = two_w * qe[0] * T::from(self.wr);
        out[4] = two_w * qe[1] * T::from(self.wr);
        out[5] = two_w * qe[2] * T::from(self.wr);
        out
    }
}

#[derive(Clone, Debug)]
pub struct ObsRes {
    pub anchor: [[dtype; 3]; 3], // R_anchor columns' rows: anchor[r][c]
    pub nm: [dtype; 3],
    pub cm: dtype,
    pub w: [dtype; 3], // waz, wel, wd
}

#[factrs::mark]
impl Residual2 for ObsRes {
    type Differ = ForwardProp<<Self as Residual2>::DimIn>;
    type V1 = SE3;
    type V2 = VectorVar3;
    type DimIn = Const<9>;
    type DimOut = Const<3>;

    fn residual2<T: Numeric>(&self, p: SE3<T>, x: VectorVar3<T>) -> VectorX<T> {
        let (dy, dz, c) = (x.0[0], x.0[1], x.0[2]);
        // chart: first column of the small-rotation matrix, in the anchor frame
        let s2 = T::from(1.0) + (dy * dy + dz * dz) * T::from(0.25);
        let l = [
            T::from(1.0) - (dy * dy + dz * dz) / (T::from(2.0) * s2),
            dz / s2,
            -dy / s2,
        ];
        let mut nw = [T::from(0.0); 3];
        for r in 0..3 {
            nw[r] = T::from(self.anchor[r][0]) * l[0]
                + T::from(self.anchor[r][1]) * l[1]
                + T::from(self.anchor[r][2]) * l[2];
        }
        let qp = quat_of(p.rot());
        let nl = quat_rotate(&quat_conj(&qp), nw);
        let cl = c + p.xyz()[0] * nw[0] + p.xyz()[1] * nw[1] + p.xyz()[2] * nw[2];
        let h = (nl[0] * nl[0] + nl[1] * nl[1]).sqrt();
        let nm = [T::from(self.nm[0]), T::from(self.nm[1]), T::from(self.nm[2])];
        let mx = nl[0] * nm[0] + nl[1] * nm[1] + nl[2] * nm[2];
        let my = (nm[1] * nl[0] - nm[0] * nl[1]) / h;
        let mz = (nm[2] * (nl[0] * nl[0] + nl[1] * nl[1])
            - nl[2] * (nl[0] * nm[0] + nl[1] * nm[1])) / h;
        let mut out = VectorX::zeros(3);
        out[0] = my.atan2(mx) * T::from(self.w[0]);
        out[1] = mz.atan2((mx * mx + my * my).sqrt()) * T::from(self.w[1]);
        out[2] = (T::from(self.cm) - cl) * T::from(self.w[2]);
        out
    }
}

// Strong 6-DOF prior fixing the gauge at pose 0 (zero at the start point).
#[derive(Clone, Debug)]
pub struct PriorSE3Res {
    pub t: [dtype; 3],
    pub q: [dtype; 4],
    pub w: dtype,
}

#[factrs::mark]
impl factrs::residuals::Residual1 for PriorSE3Res {
    type Differ = ForwardProp<<Self as factrs::residuals::Residual1>::DimIn>;
    type V1 = SE3;
    type DimIn = Const<6>;
    type DimOut = Const<6>;

    fn residual1<T: Numeric>(&self, a: SE3<T>) -> VectorX<T> {
        let qa = quat_of(a.rot());
        let qm = [T::from(self.q[0]), T::from(self.q[1]), T::from(self.q[2]), T::from(self.q[3])];
        let qe = quat_mul(&quat_conj(&qm), &qa);
        let two_w = T::from(2.0) * qe[3];
        let mut out = VectorX::zeros(6);
        for k in 0..3 {
            out[k] = (a.xyz()[k] - T::from(self.t[k])) * T::from(self.w);
            out[3 + k] = two_w * qe[k] * T::from(self.w);
        }
        out
    }
}

fn se3_of(p: &Pose) -> SE3 {
    SE3::from_rot_trans(
        SO3::from_xyzw(p.q.v.x, p.q.v.y, p.q.v.z, p.q.t),
        Vector3::new(p.t.x, p.t.y, p.t.z),
    )
}

/// Plane chart anchors: the initial normal's frame, baked into each
/// observation residual (and needed again to read the solution back).
fn anchors(raw: &RawScene) -> Vec<quaternd> {
    raw.planes.iter()
        .map(|pl| quaternd::from_two_vectors(vect3d::new(1.0, 0.0, 0.0), pl.n.unit()))
        .collect()
}

fn odo_res(rel: &Pose, wt: f64, wr: f64) -> OdoRes {
    OdoRes {
        tm: [rel.t.x, rel.t.y, rel.t.z],
        qm: [rel.q.v.x, rel.q.v.y, rel.q.v.z, rel.q.t],
        wt, wr,
    }
}

fn obs_res(anchor: &quaternd, pl: &Plane, waz: f64, wel: f64, wd: f64) -> ObsRes {
    let r = anchor.rotation_matrix();
    let s = 1.0 / pl.n.norm();
    ObsRes {
        anchor: [[r[0].x, r[0].y, r[0].z], [r[1].x, r[1].y, r[1].z], [r[2].x, r[2].y, r[2].z]],
        nm: [pl.n.x * s, pl.n.y * s, pl.n.z * s],
        cm: pl.c * s,
        w: [waz, wel, wd],
    }
}

fn prior_res(p0: &Pose) -> PriorSE3Res {
    PriorSE3Res {
        t: [p0.t.x, p0.t.y, p0.t.z],
        q: [p0.q.v.x, p0.q.v.y, p0.q.v.z, p0.q.t],
        w: 1e6,
    }
}

/// The factrs residuals' cost at the initial estimate -- for the harness to
/// cross-check against scene::reference_cost. The gauge prior sits at the
/// start pose, so it contributes zero here.
pub fn initial_cost(raw: &RawScene) -> f64 {
    use factrs::residuals::Residual1;
    let sq = |r: VectorX<f64>| r.iter().map(|x| x * x).sum::<f64>();
    let anch = anchors(raw);
    let mut cost = 0.0;
    for &(i, j, ref rel, wt, wr) in &raw.odos {
        cost += sq(odo_res(rel, wt, wr)
            .residual2(se3_of(&raw.poses[i]), se3_of(&raw.poses[j])));
    }
    for &(p, l, ref pl, waz, wel, wd) in &raw.obs {
        let s = 1.0 / raw.planes[l].n.norm();
        cost += sq(obs_res(&anch[l], pl, waz, wel, wd)
            .residual2(se3_of(&raw.poses[p]), VectorVar3::new(0.0, 0.0, raw.planes[l].c * s)));
    }
    cost += sq(prior_res(&raw.poses[0]).residual1(se3_of(&raw.poses[0])));
    cost
}

fn build(raw: &RawScene) -> (Graph, Values) {
    let mut graph = Graph::new();
    let mut init = Values::new();
    let anch = anchors(raw);
    for (i, p) in raw.poses.iter().enumerate() {
        init.insert(A(i as u32), se3_of(p));
    }
    for (j, pl) in raw.planes.iter().enumerate() {
        let s = 1.0 / pl.n.norm();
        init.insert(B(j as u32), VectorVar3::new(0.0, 0.0, pl.c * s));
    }
    for &(i, j, ref rel, wt, wr) in &raw.odos {
        graph.add_factor(fac![odo_res(rel, wt, wr), (A(i as u32), A(j as u32)), 1.0 as std]);
    }
    for &(p, l, ref pl, waz, wel, wd) in &raw.obs {
        graph.add_factor(fac![obs_res(&anch[l], pl, waz, wel, wd),
            (A(p as u32), B(l as u32)), 1.0 as std]);
    }
    graph.add_factor(fac![prior_res(&raw.poses[0]), A(0), 1.0 as std]);
    (graph, init)
}

fn extract(raw: &RawScene, values: &Values) -> Solution {
    let anch = anchors(raw);
    let poses: Vec<Pose> = (0..raw.poses.len())
        .map(|i| {
            let p: &SE3 = values.get(A(i as u32)).expect("pose");
            let r = p.rot();
            Pose {
                q: quaternd::new(r.w(), vect3d::new(r.x(), r.y(), r.z())).unit(),
                t: vect3d::new(p.xyz()[0], p.xyz()[1], p.xyz()[2]),
            }
        })
        .collect();
    let planes: Vec<Plane> = (0..raw.planes.len())
        .map(|j| {
            let x: &VectorVar3 = values.get(B(j as u32)).expect("plane");
            let (dy, dz, c) = (x.0[0], x.0[1], x.0[2]);
            let dq = quaternd::from_rotation_vector_small(vect3d::new(0.0, dy, dz));
            let n = (anch[j] * dq).rotate(vect3d::new(1.0, 0.0, 0.0));
            Plane { n, c }
        })
        .collect();
    Solution { poses, planes }
}

pub fn run(raw: &RawScene) -> crate::arael_runner::RunOut {
    bench_harness::solver::run(200, |max_iter| {
        // Building the graph is the probe's reset, not the solve -- the clock
        // starts at the optimize() below, the same boundary the C++ runners
        // draw.
        let (graph, init) = build(raw);
        let before = counts();
        let (ms, result) = bench_harness::solver::timed(|| {
            let params = LevenParams {
                base: BaseOptParams {
                    max_iterations: max_iter,
                    // The benchmark's shared termination class.
                    error_tol_relative: crate::arael_runner::tolerance(),
                    error_tol_absolute: crate::arael_runner::tolerance(),
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut opt = LevenMarquardt::new(params, graph);
            // factrs keeps its damping retries inside step(): counting the
            // linear solves recovers them; its observer sees accepted steps
            // only.
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
            solution: extract(raw, &values),
        }
    })
}
