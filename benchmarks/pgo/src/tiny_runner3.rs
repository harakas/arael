// tiny-solver 3D (SE3) runners, fed the same canonical cost function.
//
// tiny-solver's native BetweenFactorSE3 minimizes the full SE3 log map
// (a different objective than the benchmark's canonical quaternion-
// vector convention, see g2o3.rs) and its g2o reader drops the
// information matrices -- so the problem is assembled through its
// public Factor API with the canonical residual instead. The solver,
// autodiff, manifold (its own SE3Manifold), and linear algebra are all
// tiny-solver's own.

use std::collections::HashMap;
use std::sync::Arc;

use crate::g2o3::{Dataset3, Pose3In};
use arael::vect::vect3d;
use tiny_solver::factors::Factor;
use tiny_solver::manifold::se3::SE3Manifold;
use tiny_solver::na;
use tiny_solver::optimizer::Optimizer;

// Block layout is tiny-solver's SE3 convention: [qx, qy, qz, qw, x, y, z].
fn split_block<T: na::RealField>(p: &na::DVector<T>) -> (na::UnitQuaternion<T>, na::Vector3<T>) {
    let q = na::Quaternion::new(p[3].clone(), p[0].clone(), p[1].clone(), p[2].clone());
    (
        na::UnitQuaternion::from_quaternion(q),
        na::Vector3::new(p[4].clone(), p[5].clone(), p[6].clone()),
    )
}

// 2 * vec(q_err) on the qw >= 0 branch -- identical to the reference
// rot_residual. The sign flip is locally constant, so autodiff through
// it is exact.
fn quat_vec_residual<T: na::RealField>(q: &na::UnitQuaternion<T>) -> [T; 3] {
    let two = if q.w >= T::zero() {
        T::from_f64(2.0).unwrap()
    } else {
        T::from_f64(-2.0).unwrap()
    };
    [q.i.clone() * two.clone(), q.j.clone() * two.clone(), q.k.clone() * two]
}

fn apply_sqrt_info<T: na::RealField>(u: &[[f64; 6]; 6], r: [T; 6]) -> na::DVector<T> {
    let mut out = Vec::with_capacity(6);
    for i in 0..6 {
        let mut w = T::zero();
        for j in i..6 {
            if u[i][j] != 0.0 {
                w += T::from_f64(u[i][j]).unwrap() * r[j].clone();
            }
        }
        out.push(w);
    }
    na::DVector::from_vec(out)
}

#[derive(Debug, Clone)]
struct CanonicalBetweenSE3 {
    dt: [f64; 3],
    dq: [f64; 4], // (x, y, z, w)
    u: [[f64; 6]; 6],
}

impl<T: na::RealField> Factor<T> for CanonicalBetweenSE3 {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        let (qa, ta) = split_block(&params[0]);
        let (qb, tb) = split_block(&params[1]);
        let dq = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
            T::from_f64(self.dq[3]).unwrap(),
            T::from_f64(self.dq[0]).unwrap(),
            T::from_f64(self.dq[1]).unwrap(),
            T::from_f64(self.dq[2]).unwrap(),
        ));
        let dt = na::Vector3::new(
            T::from_f64(self.dt[0]).unwrap(),
            T::from_f64(self.dt[1]).unwrap(),
            T::from_f64(self.dt[2]).unwrap(),
        );
        let terr = qa.inverse_transform_vector(&(tb - ta)) - dt;
        let qerr = dq.inverse() * qa.inverse() * qb;
        let [rx, ry, rz] = quat_vec_residual(&qerr);
        apply_sqrt_info(
            &self.u,
            [
                terr[0].clone(), terr[1].clone(), terr[2].clone(),
                rx, ry, rz,
            ],
        )
    }
}

// Unit-weight gauge prior on pose 0, same convention.
#[derive(Debug, Clone)]
struct CanonicalPriorSE3 {
    t: [f64; 3],
    q: [f64; 4],
}

impl<T: na::RealField> Factor<T> for CanonicalPriorSE3 {
    fn residual_func(&self, params: &[na::DVector<T>]) -> na::DVector<T> {
        let (qp, tp) = split_block(&params[0]);
        let q0 = na::UnitQuaternion::from_quaternion(na::Quaternion::new(
            T::from_f64(self.q[3]).unwrap(),
            T::from_f64(self.q[0]).unwrap(),
            T::from_f64(self.q[1]).unwrap(),
            T::from_f64(self.q[2]).unwrap(),
        ));
        let qerr = q0.inverse() * qp;
        let [rx, ry, rz] = quat_vec_residual(&qerr);
        na::DVector::from_vec(vec![
            tp[0].clone() - T::from_f64(self.t[0]).unwrap(),
            tp[1].clone() - T::from_f64(self.t[1]).unwrap(),
            tp[2].clone() - T::from_f64(self.t[2]).unwrap(),
            rx, ry, rz,
        ])
    }
}

fn build(ds: &Dataset3) -> (tiny_solver::Problem, HashMap<String, na::DVector<f64>>) {
    let mut problem = tiny_solver::Problem::new();
    let mut init = HashMap::new();
    for (i, p) in ds.poses.iter().enumerate() {
        let name = format!("x{}", i);
        problem.set_variable_manifold(&name, Arc::new(SE3Manifold));
        init.insert(name, na::DVector::from_vec(vec![
            p.q[0], p.q[1], p.q[2], p.q[3], p.t.x, p.t.y, p.t.z,
        ]));
    }
    for e in &ds.edges {
        problem.add_residual_block(
            6,
            &[&format!("x{}", e.a), &format!("x{}", e.b)],
            Box::new(CanonicalBetweenSE3 {
                dt: [e.dt.x, e.dt.y, e.dt.z],
                dq: e.dq,
                u: e.u,
            }),
            None,
        );
    }
    let p0 = &ds.poses[0];
    problem.add_residual_block(
        6,
        &["x0"],
        Box::new(CanonicalPriorSE3 { t: [p0.t.x, p0.t.y, p0.t.z], q: p0.q }),
        None,
    );
    (problem, init)
}

fn options(max_iteration: usize) -> tiny_solver::OptimizerOptions {
    tiny_solver::OptimizerOptions {
        max_iteration,
        ..Default::default() // 1e-5 abs / 1e-5 rel decrease, sparse Cholesky
    }
}

fn extract(ds: &Dataset3, values: &HashMap<String, na::DVector<f64>>) -> Vec<Pose3In> {
    (0..ds.poses.len())
        .map(|i| {
            let v = &values[&format!("x{}", i)];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2] + v[3] * v[3]).sqrt();
            Pose3In {
                t: vect3d::new(v[4], v[5], v[6]),
                q: [v[0] / n, v[1] / n, v[2] / n, v[3] / n],
            }
        })
        .collect()
}

pub struct RunOut3 {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    pub iterations: usize,
    pub poses: Vec<Pose3In>,
}

fn run(ds: &Dataset3, gn: bool) -> RunOut3 {
    let (problem, init) = build(ds);

    let optimize = |max_iter: usize| -> (f64, usize, Option<HashMap<String, na::DVector<f64>>>) {
        let before = crate::tiny_runner::iter_count();
        let t0 = std::time::Instant::now();
        let result = if gn {
            tiny_solver::GaussNewtonOptimizer::new().optimize(&problem, &init, Some(options(max_iter)))
        } else {
            let radius = std::env::var("TINY_RADIUS0").ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(1e12);
            tiny_solver::LevenbergMarquardtOptimizer::new(1e-6, 1e32, radius)
                .optimize(&problem, &init, Some(options(max_iter)))
        };
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        (ms, crate::tiny_runner::iter_count() - before, result)
    };

    let (first_iter_ms, _, _) = optimize(1);
    let (solve_ms, iterations, result) = optimize(100);
    let values = result.expect("tiny-solver returned None");
    RunOut3 { solve_ms, first_iter_ms, iterations, poses: extract(ds, &values) }
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
    use tiny_solver::factors::Factor;

    // The tiny-solver factors must reproduce the reference cost exactly:
    // sum of squared residuals over all blocks == reference_cost3.
    #[test]
    fn tiny_cost_matches_reference() {
        let ds = crate::arael_runner3::tests::synthetic();
        let reference = reference_cost3(&ds, &ds.poses);
        let block = |p: &Pose3In| {
            na::DVector::from_vec(vec![p.q[0], p.q[1], p.q[2], p.q[3], p.t.x, p.t.y, p.t.z])
        };
        let mut cost = 0.0;
        for e in &ds.edges {
            let f = CanonicalBetweenSE3 {
                dt: [e.dt.x, e.dt.y, e.dt.z],
                dq: e.dq,
                u: e.u,
            };
            let r: na::DVector<f64> = f.residual_func(&[
                block(&ds.poses[e.a as usize]),
                block(&ds.poses[e.b as usize]),
            ]);
            cost += r.norm_squared();
        }
        let p0 = &ds.poses[0];
        let prior = CanonicalPriorSE3 { t: [p0.t.x, p0.t.y, p0.t.z], q: p0.q };
        let r: na::DVector<f64> = prior.residual_func(&[block(p0)]);
        cost += r.norm_squared();
        assert!(((cost - reference) / reference).abs() < 1e-12,
            "tiny {} vs reference {}", cost, reference);
    }
}
