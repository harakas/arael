// factrs 3D (SE3) runners, fed the same canonical cost function.
//
// factrs's native BetweenResidual<SE3> minimizes the full SE3 log map
// (a different objective than the benchmark's canonical quaternion-
// vector convention, see g2o3.rs), so the residual is implemented here
// through factrs's public custom-residual API instead -- its
// dual-number ForwardProp autodiff, optimizers, and linear algebra all
// stay factrs's own. The full 6x6 sqrt-information factor is folded
// into the residual (whitened residual + unit noise) rather than
// expressed as a GaussianNoise, keeping the weighting bit-identical to
// the reference cost.

use core::fmt;
use bench_harness::factrs::{counts, since, CountingSolver, StepCounter};
use crate::g2o3::{Dataset3, Pose3In};
use arael::vect::vect3d;
use bench_harness::table::Row;
use factrs::assign_symbols;
use factrs::core::{GaussNewton, Graph, LevenMarquardt, Values};
use factrs::dtype;
use factrs::fac;
use factrs::linalg::{Const, ForwardProp, Numeric, Vector3, VectorX};
use factrs::optimizers::{BaseOptParams, LevenParams, OptError};
use factrs::residuals::{Residual1, Residual2};
use factrs::traits::Optimizer;
use factrs::variables::{SE3, SO3};

assign_symbols!(Y: SE3);


// Hamilton product (a * b) on (x, y, z, w) component tuples.
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

// Rotate v by the quaternion q (unit): q * (0, v) * q^-1.
fn quat_rotate<T: Numeric>(q: &[T; 4], v: &Vector3<T>) -> Vector3<T> {
    let p = [v[0], v[1], v[2], T::from(0.0)];
    let r = quat_mul(&quat_mul(q, &p), &quat_conj(q));
    Vector3::new(r[0], r[1], r[2])
}

// 2 * vec(q_err) on the qw >= 0 branch -- identical to the reference
// rot_residual. The sign flip is locally constant, so dual-number
// autodiff through it is exact.
fn quat_vec_residual<T: Numeric>(q: &[T; 4]) -> [T; 3] {
    let two = if q[3] >= T::from(0.0) {
        T::from(2.0)
    } else {
        T::from(-2.0)
    };
    [q[0] * two, q[1] * two, q[2] * two]
}

#[derive(Clone, Debug)]
pub struct CanonicalBetweenSE3 {
    dt: [dtype; 3],
    dq: [dtype; 4], // (x, y, z, w)
    u: [[dtype; 6]; 6],
}

impl fmt::Display for CanonicalBetweenSE3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CanonicalBetweenSE3(dt={:?}, dq={:?})", self.dt, self.dq)
    }
}

#[factrs::mark]
impl Residual2 for CanonicalBetweenSE3 {
    type Differ = ForwardProp<<Self as Residual2>::DimIn>;
    type V1 = SE3;
    type V2 = SE3;
    type DimIn = Const<12>;
    type DimOut = Const<6>;

    fn residual2<T: Numeric>(&self, a: SE3<T>, b: SE3<T>) -> VectorX<T> {
        let qa = quat_of(a.rot());
        let qb = quat_of(b.rot());
        let dq = [
            T::from(self.dq[0]), T::from(self.dq[1]),
            T::from(self.dq[2]), T::from(self.dq[3]),
        ];
        let diff = Vector3::new(
            b.xyz()[0] - a.xyz()[0],
            b.xyz()[1] - a.xyz()[1],
            b.xyz()[2] - a.xyz()[2],
        );
        let local = quat_rotate(&quat_conj(&qa), &diff);
        let terr = [
            local[0] - T::from(self.dt[0]),
            local[1] - T::from(self.dt[1]),
            local[2] - T::from(self.dt[2]),
        ];
        let qerr = quat_mul(&quat_conj(&dq), &quat_mul(&quat_conj(&qa), &qb));
        let [rx, ry, rz] = quat_vec_residual(&qerr);
        let r = [terr[0], terr[1], terr[2], rx, ry, rz];
        let mut out = VectorX::zeros(6);
        for i in 0..6 {
            let mut w = T::from(0.0);
            for j in i..6 {
                if self.u[i][j] != 0.0 {
                    w += T::from(self.u[i][j]) * r[j];
                }
            }
            out[i] = w;
        }
        out
    }
}

// Unit-weight gauge prior on pose 0, same convention.
#[derive(Clone, Debug)]
pub struct CanonicalPriorSE3 {
    t: [dtype; 3],
    q: [dtype; 4],
}

impl fmt::Display for CanonicalPriorSE3 {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CanonicalPriorSE3(t={:?}, q={:?})", self.t, self.q)
    }
}

#[factrs::mark]
impl Residual1 for CanonicalPriorSE3 {
    type Differ = ForwardProp<<Self as Residual1>::DimIn>;
    type V1 = SE3;
    type DimIn = Const<6>;
    type DimOut = Const<6>;

    fn residual1<T: Numeric>(&self, p: SE3<T>) -> VectorX<T> {
        let q0 = [
            T::from(self.q[0]), T::from(self.q[1]),
            T::from(self.q[2]), T::from(self.q[3]),
        ];
        let qerr = quat_mul(&quat_conj(&q0), &quat_of(p.rot()));
        let [rx, ry, rz] = quat_vec_residual(&qerr);
        let mut out = VectorX::zeros(6);
        out[0] = p.xyz()[0] - T::from(self.t[0]);
        out[1] = p.xyz()[1] - T::from(self.t[1]);
        out[2] = p.xyz()[2] - T::from(self.t[2]);
        out[3] = rx;
        out[4] = ry;
        out[5] = rz;
        out
    }
}

// The dataset is f64; factrs's scalar is whatever `dtype` the crate was
// built with. The casts are what let this file compile under both, so the
// f32 runner is the same code and not a second implementation of it.
fn se3_of(p: &Pose3In) -> SE3 {
    SE3::from_rot_trans(
        SO3::from_xyzw(
            p.q[0] as dtype, p.q[1] as dtype, p.q[2] as dtype, p.q[3] as dtype),
        Vector3::new(p.t.x as dtype, p.t.y as dtype, p.t.z as dtype),
    )
}

fn build(ds: &Dataset3) -> (Graph, Values) {
    let mut graph = Graph::new();
    let mut values = Values::new();
    for (i, p) in ds.poses.iter().enumerate() {
        values.insert(Y(i as u32), se3_of(p));
    }
    for e in &ds.edges {
        // The sqrt-info is inside the residual; unit noise here.
        graph.add_factor(fac![
            CanonicalBetweenSE3 {
                dt: [e.dt.x as dtype, e.dt.y as dtype, e.dt.z as dtype],
                dq: e.dq.map(|v| v as dtype),
                u: e.u.map(|row| row.map(|v| v as dtype)),
            },
            (Y(e.a), Y(e.b)),
            1.0 as std
        ]);
    }
    let p0 = &ds.poses[0];
    graph.add_factor(fac![
        CanonicalPriorSE3 {
            t: [p0.t.x as dtype, p0.t.y as dtype, p0.t.z as dtype],
            q: p0.q.map(|v| v as dtype),
        },
        Y(0),
        1.0 as std
    ]);
    (graph, values)
}

pub type RunOut3 = Row<Vec<Pose3In>>;


fn base_params(max_iterations: usize) -> BaseOptParams {
    BaseOptParams {
        max_iterations,
        error_tol_relative: 1e-5,
        error_tol_absolute: 1e-5,
        ..Default::default()
    }
}

// (elapsed ms, accepted steps, attempts, values)
fn optimize_counted(ds: &Dataset3, gn: bool, max_iter: usize) -> (usize, usize, Values) {
    let (graph, init) = build(ds);
    let before = counts();
    let result = if gn {
        let mut opt = GaussNewton::new(base_params(max_iter), graph);
        opt.set_solver(CountingSolver::default());
        opt.observers_mut().add(StepCounter);
        opt.optimize(init)
    } else {
        let params = LevenParams { base: base_params(max_iter), ..Default::default() };
        let mut opt = LevenMarquardt::new(params, graph);
        opt.set_solver(CountingSolver::default());
        opt.observers_mut().add(StepCounter);
        opt.optimize(init)
    };
    let values = match result {
        Ok(v) => v,
        Err(OptError::MaxIterations(v)) => v,
        Err(e) => panic!("factrs failed: {:?}", e),
    };
    let (accepted, attempts) = since(before);
    (accepted, attempts, values)
}

fn solution_of(ds: &Dataset3, values: &Values) -> Vec<Pose3In> {
    (0..ds.poses.len())
        .map(|i| {
            let p: &SE3 = values.get(Y(i as u32)).expect("missing pose");
            let r = p.rot();
            Pose3In {
                t: vect3d::new(
                    p.xyz()[0] as f64, p.xyz()[1] as f64, p.xyz()[2] as f64),
                q: [r.x() as f64, r.y() as f64, r.z() as f64, r.w() as f64],
            }
        })
        .collect()
}

fn run(ds: &Dataset3, gn: bool) -> RunOut3 {
    bench_harness::solver::run(100, |max_iter| {
        let (accepted, attempts, values) = optimize_counted(ds, gn, max_iter);
        bench_harness::solver::Outcome {
            accepted,
            attempts,
            solution: solution_of(ds, &values),
        }
    })
}

pub fn run_gn(ds: &Dataset3) -> RunOut3 {
    run(ds, true)
}

pub fn run_lm(ds: &Dataset3) -> RunOut3 {
    run(ds, false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::g2o3::reference_cost3;
    use factrs::traits::Variable;

    // The factrs residuals must reproduce the reference cost exactly.
    #[test]
    fn factrs_cost_matches_reference() {
        let ds = crate::arael_runner3::tests::synthetic();
        let reference = reference_cost3(&ds, &ds.poses);
        let mut cost = 0.0;
        for e in &ds.edges {
            let f = CanonicalBetweenSE3 {
                dt: [e.dt.x, e.dt.y, e.dt.z],
                dq: e.dq,
                u: e.u,
            };
            let r: VectorX<f64> = f.residual2(
                se3_of(&ds.poses[e.a as usize]).cast(),
                se3_of(&ds.poses[e.b as usize]).cast(),
            );
            cost += r.norm_squared();
        }
        let p0 = &ds.poses[0];
        let prior = CanonicalPriorSE3 { t: [p0.t.x, p0.t.y, p0.t.z], q: p0.q };
        let r: VectorX<f64> = prior.residual1(se3_of(p0).cast());
        cost += r.norm_squared();
        assert!(((cost - reference) / reference).abs() < 1e-12,
            "factrs {} vs reference {}", cost, reference);
    }
}
