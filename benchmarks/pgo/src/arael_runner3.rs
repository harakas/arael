// arael 3D (SE3) runners: one generic model, instantiated by the f64
// and f32 roots.
//
// Pose rotation is a QuaternionParam: a rotation-vector delta composed with
// a reference rotation that re-centers after every accepted step, so
// the parameterization never leaves its small-angle sweet spot even on
// arbitrarily oriented graphs (the sphere). The constraint body is the
// canonical quaternion-vector between residual (see g2o3.rs), weighted
// by the edge's upper-triangular 6x6 sqrt-information blocks.

use bench_harness::arael::{run, Model as Pipeline};
use bench_harness::table::Row;
use crate::g2o3::{Dataset3, Pose3In};
use arael::matrix::matrix3;
use arael::model::{CrossBlock, Param, QuaternionParam, SelfBlock};
use arael::quatern::quatern;
use arael::refs::{self, Ref};
use arael::utils::Float;
use arael::vect::vect3;

#[arael::model]
#[arael(constraint(hb, guard = self.has_prior, {
    let rerr = pose3.prior_rot_t * pose3.ea.rotation_matrix();
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    [pose3.pos.x - pose3.prior.x,
     pose3.pos.y - pose3.prior.y,
     pose3.pos.z - pose3.prior.z,
     s[2].y / denom,
     s[0].z / denom,
     s[1].x / denom]
}))]
#[derive(Clone)]
struct Pose3<T: Float> {
    pos: Param<vect3<T>>,
    ea: QuaternionParam<T>,
    prior: vect3<T>,
    prior_rot_t: matrix3<T>,
    has_prior: bool,
    hb: SelfBlock<Pose3<T>, T>,
}

#[arael::model]
#[arael(constraint(hb, {
    let ra = a.ea.rotation_matrix();
    let rerr = edge3.rmeas_t * (ra.transpose() * b.ea.rotation_matrix());
    let s = rerr - rerr.transpose();
    let denom = safe_sqrt(rerr[0].x + rerr[1].y + rerr[2].z + 1.0);
    let rrot = vect3sym::from_components(s[2].y / denom, s[0].z / denom, s[1].x / denom);
    let terr = ra.transpose() * (b.pos - a.pos) - edge3.dt;
    let wt = edge3.u_tt * terr + edge3.u_tr * rrot;
    let wr = edge3.u_rr * rrot;
    [wt.x, wt.y, wt.z, wr.x, wr.y, wr.z]
}))]
#[derive(Clone)]
struct Edge3<T: Float> {
    #[arael(ref = root.poses)]
    a: Ref<Pose3<T>>,
    #[arael(ref = root.poses)]
    b: Ref<Pose3<T>>,
    dt: vect3<T>,
    rmeas_t: matrix3<T>, // measured rotation, transposed
    u_tt: matrix3<T>,    // sqrt-info blocks: [ u_tt u_tr ; 0 u_rr ]
    u_tr: matrix3<T>,
    u_rr: matrix3<T>,
    hb: CrossBlock<Pose3<T>, Pose3<T>, T>,
}

#[arael::model]
#[arael(root)]
#[derive(Clone)]
pub struct Graph3 {
    poses: refs::Vec<Pose3<f64>>,
    edges: std::vec::Vec<Edge3<f64>>,
}

#[arael::model]
#[arael(root, f32)]
#[derive(Clone)]
struct Graph3F {
    poses: refs::Vec<Pose3<f32>>,
    edges: std::vec::Vec<Edge3<f32>>,
}

// ---------------------------------------------------------------- runners

fn build_parts<T: Float>(ds: &Dataset3)
    -> (refs::Vec<Pose3<T>>, std::vec::Vec<Edge3<T>>)
{
    let mut poses = refs::Vec::new();
    for (i, p) in ds.poses.iter().enumerate() {
        let rot = p.rot();
        poses.push(Pose3 {
            pos: Param::new(p.t.cast()),
            ea: QuaternionParam::new(quatern::from_rotation_matrix(rot.cast())),
            prior: p.t.cast(),
            prior_rot_t: rot.transpose().cast(),
            has_prior: i == 0,
            hb: SelfBlock::new(),
        });
    }
    let mut edges = std::vec::Vec::new();
    for e in &ds.edges {
        let (u_tt, u_tr, u_rr) = e.u_blocks();
        edges.push(Edge3 {
            a: poses.ref_at(e.a),
            b: poses.ref_at(e.b),
            dt: e.dt.cast(),
            rmeas_t: crate::g2o3::quat_to_matrix(e.dq).transpose().cast(),
            u_tt: u_tt.cast(),
            u_tr: u_tr.cast(),
            u_rr: u_rr.cast(),
            hb: CrossBlock::new(),
        });
    }
    (poses, edges)
}

pub(crate) fn build_f64(ds: &Dataset3) -> Graph3 {
    let (poses, edges) = build_parts(ds);
    Graph3 { poses, edges }
}

fn build_f32(ds: &Dataset3) -> Graph3F {
    let (poses, edges) = build_parts(ds);
    Graph3F { poses, edges }
}

fn solution_parts<T: Float>(poses: &refs::Vec<Pose3<T>>) -> Vec<Pose3In> {
    let f = |x: T| x.to_f64().unwrap();
    poses.iter()
        .map(|p| Pose3In {
            t: p.pos.value.cast(),
            q: [f(p.ea.value.v.x), f(p.ea.value.v.y), f(p.ea.value.v.z), f(p.ea.value.t)],
        })
        .collect()
}

// Initial damping for the 3D graphs. parking-garage's weak information matrices
// leave 1e-8 over-damped -- it converges with damping rejections and an early
// plateau stop 6.5 cm short of the optimum; 1e-10 converges cleanly. sphere2500
// is insensitive to the choice.
pub const LAMBDA0_3D: f64 = 1e-10;

impl Pipeline for Graph3 {
    type Scalar = f64;
    type Input = Dataset3;
    type Solution = Vec<Pose3In>;
    fn lambda0(_: &Dataset3) -> f64 { LAMBDA0_3D }
    fn build(ds: &Dataset3) -> Self { build_f64(ds) }
    fn serialize(&mut self, out: &mut Vec<f64>) { self.serialize64(out); }
    fn deserialize(&mut self, x: &[f64]) { self.deserialize64(x); }
    fn solution(&self) -> Vec<Pose3In> { solution_parts(&self.poses) }
    fn solve(_: &Self::Input, params: &[f64], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f64>)
        -> arael::simple_lm::LmResult<f64> {
        crate::arael_runner::solve_f64(params, m, cfg)
    }
}

impl Pipeline for Graph3F {
    type Scalar = f32;
    type Input = Dataset3;
    type Solution = Vec<Pose3In>;
    fn lambda0(_: &Dataset3) -> f64 { LAMBDA0_3D }
    fn build(ds: &Dataset3) -> Self { build_f32(ds) }
    fn serialize(&mut self, out: &mut Vec<f32>) { self.serialize32(out); }
    fn deserialize(&mut self, x: &[f32]) { self.deserialize32(x); }
    fn solution(&self) -> Vec<Pose3In> { solution_parts(&self.poses) }
    fn solve(_: &Self::Input, params: &[f32], m: &mut Self, cfg: &arael::simple_lm::LmConfig<f32>)
        -> arael::simple_lm::LmResult<f32> {
        crate::arael_runner::solve_f32(params, m, cfg)
    }
}

pub type RunOut3 = Row<Vec<Pose3In>>;

pub fn run_f64(ds: &Dataset3) -> RunOut3 { run::<Graph3>(ds) }
pub fn run_f32(ds: &Dataset3) -> RunOut3 { run::<Graph3F>(ds) }

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::g2o3::{aligned_rmse3, reference_cost3, Edge3In};
    use arael::quatern::quaternd;
    use arael::vect::vect3d;

    // A 5-pose loop with varied rotations. Measurements are taken from a
    // DIFFERENT (perturbed) trajectory so every residual is nonzero and
    // the weighting actually matters.
    pub(crate) fn synthetic() -> Dataset3 {
        let pose = |t: (f64, f64, f64), ea: (f64, f64, f64)| {
            let q = quaternd::from_euler_angles(vect3d::new(ea.0, ea.1, ea.2));
            Pose3In { t: vect3d::new(t.0, t.1, t.2), q: [q.v.x, q.v.y, q.v.z, q.t] }
        };
        let poses = vec![
            pose((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
            pose((1.0, 0.2, -0.1), (0.1, -0.2, 0.4)),
            pose((1.8, 1.1, 0.3), (-0.3, 0.1, 1.2)),
            pose((1.2, 2.0, 0.7), (0.2, 0.4, 2.1)),
            pose((0.1, 1.4, 0.2), (-0.1, -0.3, -2.8)),
        ];
        // Full 6x6 sqrt-info with off-diagonal coupling, different per edge.
        let u_for = |k: usize| {
            let mut u = [[0.0f64; 6]; 6];
            for i in 0..6 {
                u[i][i] = 1.0 + 0.2 * ((i + k) % 3) as f64;
                for j in (i + 1)..6 {
                    u[i][j] = 0.05 * ((i + 2 * j + k) % 5) as f64;
                }
            }
            u
        };
        let edge = |k: usize, a: usize, b: usize, poses: &[Pose3In]| {
            let ra = poses[a].rot();
            // Perturbed relative measurement.
            let dt = ra.transpose() * (poses[b].t - poses[a].t)
                + vect3d::new(0.03, -0.02, 0.01) * (1.0 + k as f64 * 0.3);
            let dq_mat = ra.transpose() * poses[b].rot()
                * arael::matrix::matrix3d::rotation_from_euler_angles(
                    vect3d::new(0.02, 0.01, -0.03) * (1.0 + k as f64 * 0.2));
            let q = quaternd::from_rotation_matrix(dq_mat);
            Edge3In {
                a: a as u32, b: b as u32,
                dt,
                dq: [q.v.x, q.v.y, q.v.z, q.t],
                u: u_for(k),
            }
        };
        let edges = (0..5).map(|k| edge(k, k, (k + 1) % 5, &poses)).collect();
        Dataset3 { poses, edges }
    }

    // The arael model's cost must be exactly the reference cost -- this
    // pins the whole symbolic pipeline (residual convention, sqrt-info
    // blocks, prior) to the one function every system is judged by.
    #[test]
    fn arael_cost_matches_reference() {
        use arael::simple_lm::LmProblem;
        let ds = synthetic();
        let reference = reference_cost3(&ds, &ds.poses);
        assert!(reference > 1e-3, "synthetic residuals unexpectedly small: {}", reference);

        let mut g = build_f64(&ds);
        let mut params: Vec<f64> = Vec::new();
        g.serialize64(&mut params);
        let arael_cost = g.calc_cost(&params);
        assert!(((arael_cost - reference) / reference).abs() < 1e-12,
            "arael {} vs reference {}", arael_cost, reference);
    }

    // Consistent measurements => the ground truth is the global optimum
    // (cost 0). Solving from a perturbed init must recover it.
    #[test]
    fn arael_solves_consistent_loop() {
        let mut ds = synthetic();
        for e in &mut ds.edges {
            let (a, b) = (&ds.poses[e.a as usize], &ds.poses[e.b as usize]);
            let ra = a.rot();
            e.dt = ra.transpose() * (b.t - a.t);
            let q = quaternd::from_rotation_matrix(ra.transpose() * b.rot());
            e.dq = [q.v.x, q.v.y, q.v.z, q.t];
        }
        let truth = ds.poses.clone();
        assert!(reference_cost3(&ds, &truth) < 1e-20);
        // Perturb every pose except the anchored one.
        for (i, p) in ds.poses.iter_mut().enumerate().skip(1) {
            p.t = p.t + vect3d::new(0.05, -0.08, 0.06) * (1.0 + (i % 3) as f64);
            let rot = p.rot()
                * arael::matrix::matrix3d::rotation_from_euler_angles(
                    vect3d::new(0.06, -0.05, 0.08) * (1.0 + (i % 2) as f64));
            let q = quaternd::from_rotation_matrix(rot);
            p.q = [q.v.x, q.v.y, q.v.z, q.t];
        }
        assert!(reference_cost3(&ds, &ds.poses) > 1e-2);

        let out = run_f64(&ds);
        assert!(reference_cost3(&ds, &out.solution) < 1e-8,
            "f64 did not reach the optimum: {}", reference_cost3(&ds, &out.solution));
        assert!(aligned_rmse3(&out.solution, &truth) < 1e-4);

        let out32 = run_f32(&ds);
        assert!(reference_cost3(&ds, &out32.solution) < 1e-4,
            "f32 did not reach the optimum: {}", reference_cost3(&ds, &out32.solution));
    }
}
