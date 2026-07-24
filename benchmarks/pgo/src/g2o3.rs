// SE3 g2o loader and the 3D reference cost function every system is
// validated against.
//
// Canonical 3D residual convention: the Ceres pose_graph_3d / g2o
// EdgeSE3 form,
//
//   r = U * [ R_a^T (t_b - t_a) - t_meas ; rot_residual(R_err) ]
//   R_err = R_meas^T R_a^T R_b
//   rot_residual(R) = vee(R - R^T) / sqrt(1 + trace(R))
//
// The rotation part equals 2 * vec(q_err) -- twice the vector part of
// the error quaternion on its qw >= 0 branch (2 sin(theta/2) * axis).
// It agrees with the SO3 log map to first order, is algebraically
// smooth (no acos, no 0/0 at zero error), and only degenerates at a
// 180-degree error, far outside any converging trajectory. U is the
// upper Cholesky factor of the edge's full 6x6 information matrix
// (info = U^T U), rows ordered like the file: (x y z qx qy qz).

use arael::matrix::matrix3d;
use arael::quatern::quaternd;
use arael::vect::vect3d;

/// One pose: translation + unit quaternion (x, y, z, w).
#[derive(Clone, Copy, Debug)]
pub struct Pose3In {
    pub t: vect3d,
    pub q: [f64; 4],
}

impl Pose3In {
    pub fn rot(&self) -> matrix3d {
        quat_to_matrix(self.q)
    }
}

/// One relative SE3 measurement with its upper-triangular sqrt
/// information factor (info = u^T u), row order (x y z qx qy qz).
#[derive(Clone, Copy, Debug)]
pub struct Edge3In {
    pub a: u32,
    pub b: u32,
    pub dt: vect3d,
    pub dq: [f64; 4],
    pub u: [[f64; 6]; 6],
}

impl Edge3In {
    /// The three 3x3 blocks of the upper-triangular sqrt-info factor:
    /// [ u_tt u_tr ; 0 u_rr ] -- the shapes the arael model stores.
    pub fn u_blocks(&self) -> (matrix3d, matrix3d, matrix3d) {
        let b = |r0: usize, c0: usize| {
            matrix3d::from_elements(
                self.u[r0][c0], self.u[r0][c0 + 1], self.u[r0][c0 + 2],
                self.u[r0 + 1][c0], self.u[r0 + 1][c0 + 1], self.u[r0 + 1][c0 + 2],
                self.u[r0 + 2][c0], self.u[r0 + 2][c0 + 1], self.u[r0 + 2][c0 + 2],
            )
        };
        (b(0, 0), b(0, 3), b(3, 3))
    }
}

pub struct Dataset3 {
    pub poses: Vec<Pose3In>,
    pub edges: Vec<Edge3In>,
}

/// (x, y, z, w) component array to the runtime quaternion type.
pub fn quat_to_matrix(q: [f64; 4]) -> matrix3d {
    quaternd::new(q[3], vect3d::new(q[0], q[1], q[2])).rotation_matrix()
}

/// Rotation matrix to (x, y, z, w) components via the runtime
/// quaternion type.
#[allow(dead_code)] // used by the tests and by runners that hold rotations as matrices
pub fn matrix_to_quat(m: matrix3d) -> [f64; 4] {
    let q = quaternd::from_rotation_matrix(m);
    [q.v.x, q.v.y, q.v.z, q.t]
}

/// Load via `arael::g2o` and resolve each measurement's information
/// matrix into its upper sqrt factor. Quaternions become (x, y, z, w)
/// arrays -- the shape the external-system adapters consume.
pub fn load3(path: &str) -> Dataset3 {
    let ds = arael::g2o::Dataset3::load(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
    Dataset3 {
        poses: ds.poses.iter().map(|p| Pose3In {
            t: p.t,
            q: [p.q.v.x, p.q.v.y, p.q.v.z, p.q.t],
        }).collect(),
        edges: ds.deltas.iter().map(|d| Edge3In {
            a: d.a,
            b: d.b,
            dt: d.dt,
            dq: [d.dq.v.x, d.dq.v.y, d.dq.v.z, d.dq.t],
            u: d.sqrt_info_upper(),
        }).collect(),
    }
}

/// The smooth quaternion-vector rotation residual: 2 sin(theta/2) * axis
/// of the rotation error. The max() guards the exact-180-degree pole
/// (1 + trace = 0) with a finite denominator; it changes nothing in the
/// smooth region.
pub fn rot_residual(r: matrix3d) -> vect3d {
    let tr = r[0].x + r[1].y + r[2].z;
    let denom = (1.0 + tr).max(1e-12).sqrt();
    vect3d::new(
        (r[2].y - r[1].z) / denom,
        (r[0].z - r[2].x) / denom,
        (r[1].x - r[0].y) / denom,
    )
}

/// Unweighted canonical residual of one edge given the two poses.
pub fn edge_residual(e: &Edge3In, a: &Pose3In, b: &Pose3In) -> [f64; 6] {
    let ra = a.rot();
    let rerr = quat_to_matrix(e.dq).transpose() * (ra.transpose() * b.rot());
    let terr = ra.transpose() * (b.t - a.t) - e.dt;
    let rrot = rot_residual(rerr);
    [terr.x, terr.y, terr.z, rrot.x, rrot.y, rrot.z]
}

/// Sum of squared weighted residuals (no 1/2 factor) over the between
/// factors plus the unit-weight gauge prior on pose 0. This is THE cost
/// every system optimizes; each system's result is evaluated with this
/// one function.
pub fn reference_cost3(ds: &Dataset3, sol: &[Pose3In]) -> f64 {
    assert_eq!(sol.len(), ds.poses.len());
    let mut cost = 0.0;
    for e in &ds.edges {
        let r = edge_residual(e, &sol[e.a as usize], &sol[e.b as usize]);
        for i in 0..6 {
            let mut w = 0.0;
            for j in i..6 {
                w += e.u[i][j] * r[j];
            }
            cost += w * w;
        }
    }
    let p = &sol[0];
    let g = &ds.poses[0];
    let terr = p.t - g.t;
    let rrot = rot_residual(g.rot().transpose() * p.rot());
    cost += terr.square() + rrot.square();
    cost
}

#[cfg(test)]
mod tests {
    use super::*;

    // Vendored-file sanity: counts, unit quaternions, PD information
    // (the loader asserts PD internally), and sqrt-info roundtrip
    // u^T u == info for a spot-checked edge.
    #[test]
    fn loads_vendored_3d_datasets() {
        for (path, n_poses, n_edges) in [
            ("datasets/sphere2500.g2o", 2500usize, 4949usize),
            ("datasets/parking-garage.g2o", 1661, 6275),
        ] {
            let ds = load3(path);
            assert_eq!(ds.poses.len(), n_poses);
            assert_eq!(ds.edges.len(), n_edges);
            for e in &ds.edges {
                assert!((e.a as usize) < n_poses && (e.b as usize) < n_poses);
            }
            let q = ds.poses[1].q;
            let n = q.iter().map(|v| v * v).sum::<f64>();
            assert!((n - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn sqrt_info_roundtrip() {
        let ds = load3("datasets/sphere2500.g2o");
        let e = &ds.edges[0];
        // Rebuild info = u^T u and compare against the file's entries.
        let text = std::fs::read_to_string("datasets/sphere2500.g2o").unwrap();
        let line = text.lines().find(|l| l.starts_with("EDGE_SE3:QUAT")).unwrap();
        let v: Vec<f64> = line.split_whitespace().skip(10)
            .map(|t| t.parse().unwrap()).collect();
        let mut k = 0;
        for i in 0..6 {
            for j in i..6 {
                let mut info_ij = 0.0;
                for r in 0..6 {
                    info_ij += e.u[r][i] * e.u[r][j];
                }
                assert!((info_ij - v[k]).abs() < 1e-9 * (1.0 + v[k].abs()),
                    "info[{}][{}]: {} vs {}", i, j, info_ij, v[k]);
                k += 1;
            }
        }
    }

    // rot_residual equals 2 * vec(q) (qw >= 0 branch) for a spread of
    // rotations.
    #[test]
    fn rot_residual_is_twice_quat_vec() {
        for ea in [
            vect3d::new(0.1, -0.2, 0.3),
            vect3d::new(1.2, 0.7, -2.0),
            vect3d::new(-0.01, 0.02, 0.005),
        ] {
            let q = quaternd::from_euler_angles(ea);
            let q = if q.t >= 0.0 { q } else { -q };
            let r = rot_residual(q.rotation_matrix());
            assert!((r.x - 2.0 * q.v.x).abs() < 1e-12);
            assert!((r.y - 2.0 * q.v.y).abs() < 1e-12);
            assert!((r.z - 2.0 * q.v.z).abs() < 1e-12);
        }
    }
}

/// RMSE between two solutions' translations after the best rigid
/// (rotation + translation) alignment -- gauge-independent geometric
/// agreement. Rotation via the polar decomposition of the covariance
/// (equivalent to the Umeyama SVD solution when the covariance is
/// nonsingular with positive determinant, always the case for two
/// solutions of the same graph).
pub fn aligned_rmse3(a: &[Pose3In], b: &[Pose3In]) -> f64 {
    let n = a.len() as f64;
    let mut ma = vect3d::new(0.0, 0.0, 0.0);
    let mut mb = vect3d::new(0.0, 0.0, 0.0);
    for (pa, pb) in a.iter().zip(b.iter()) {
        ma = ma + pa.t;
        mb = mb + pb.t;
    }
    let (ma, mb) = (ma * (1.0 / n), mb * (1.0 / n));
    // c = sum (a_i - ma)(b_i - mb)^T; the maximizing rotation of
    // trace(R c) is R = c^T (c c^T)^{-1/2}.
    let mut c = matrix3d::zero_matrix();
    for (pa, pb) in a.iter().zip(b.iter()) {
        let x = pa.t - ma;
        let y = pb.t - mb;
        for i in 0..3 {
            let xi = [x.x, x.y, x.z][i];
            c[i] = c[i] + y * xi;
        }
    }
    let cct = c * c.transpose();
    let (v, d) = cct.symmetric_eigen();
    assert!(d.x > 0.0 && d.y > 0.0 && d.z > 0.0, "degenerate covariance in aligned_rmse3");
    let inv_sqrt = v
        * matrix3d::from_elements(
            1.0 / d.x.sqrt(), 0.0, 0.0,
            0.0, 1.0 / d.y.sqrt(), 0.0,
            0.0, 0.0, 1.0 / d.z.sqrt(),
        )
        * v.transpose();
    let r = c.transpose() * inv_sqrt;
    assert!(r.det() > 0.0, "reflection in aligned_rmse3 -- solutions are mirrored");
    let mut err = 0.0;
    for (pa, pb) in a.iter().zip(b.iter()) {
        let e = r * (pa.t - ma) - (pb.t - mb);
        err += e.square();
    }
    (err / n).sqrt()
}
