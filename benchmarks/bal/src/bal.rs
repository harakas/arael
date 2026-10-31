// BAL (Bundle Adjustment in the Large) loader and the reference cost
// function every system is validated against.
//
// File format (one problem per file):
//   <num_cameras> <num_points> <num_observations>
//   <camera_index> <point_index> <x> <y>          (per observation)
//   <9 values per camera>   R as Rodrigues axis-angle (world-to-camera),
//                           t, f, k1, k2 -- one value per line
//   <3 values per point>
//
// Camera model (the Snavely reprojection convention): X_cam = R X + t,
// perspective divide with NEGATIVE z (BAL cameras look down -z),
// radial distortion 1 + k1 r^2 + k2 r^4, scale by f. The residual is
// (predicted - observed) in pixels, unit weight; the reference cost is
// the plain sum of squared residuals (no 1/2 factor).

use arael::matrix::matrix3d;
use arael::vect::{vect2d, vect3d};

#[derive(Clone, Copy, Debug)]
pub struct CameraIn {
    pub rodrigues: vect3d, // world-to-camera rotation, axis-angle
    pub t: vect3d,
    pub f: f64,
    pub k1: f64,
    pub k2: f64,
}

impl CameraIn {
    /// World-to-camera rotation matrix from the Rodrigues vector.
    pub fn rot(&self) -> matrix3d {
        rodrigues_to_matrix(self.rodrigues)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ObservationIn {
    pub cam: u32,
    pub point: u32,
    pub xy: vect2d,
}

pub struct Dataset {
    pub cameras: Vec<CameraIn>,
    pub points: Vec<vect3d>,
    pub observations: Vec<ObservationIn>,
}

/// Rodrigues axis-angle to rotation matrix (runtime code; the branch at
/// small angles is the standard Taylor guard).
pub fn rodrigues_to_matrix(w: vect3d) -> matrix3d {
    let theta2 = w.square();
    if theta2 > 1e-24 {
        let theta = theta2.sqrt();
        let axis = w * (1.0 / theta);
        matrix3d::rotation_from_axis_angle(axis, theta)
    } else {
        // First-order: I + skew(w).
        matrix3d::from_elements(
            1.0, -w.z, w.y,
            w.z, 1.0, -w.x,
            -w.y, w.x, 1.0,
        )
    }
}

pub fn load(path: &str) -> Dataset {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    let mut it = text.split_ascii_whitespace().map(|t| {
        t.parse::<f64>().unwrap_or_else(|e| panic!("bad number {}: {}", t, e))
    });
    let mut next = || it.next().expect("truncated BAL file");
    let n_cams = next() as usize;
    let n_points = next() as usize;
    let n_obs = next() as usize;
    let mut observations = Vec::with_capacity(n_obs);
    for _ in 0..n_obs {
        observations.push(ObservationIn {
            cam: next() as u32,
            point: next() as u32,
            xy: vect2d::new(next(), next()),
        });
    }
    let mut cameras = Vec::with_capacity(n_cams);
    for _ in 0..n_cams {
        cameras.push(CameraIn {
            rodrigues: vect3d::new(next(), next(), next()),
            t: vect3d::new(next(), next(), next()),
            f: next(),
            k1: next(),
            k2: next(),
        });
    }
    let mut points = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        points.push(vect3d::new(next(), next(), next()));
    }
    assert!(it.next().is_none(), "trailing data in BAL file");
    for o in &observations {
        assert!((o.cam as usize) < n_cams && (o.point as usize) < n_points);
    }
    Dataset { cameras, points, observations }
}

/// One observation's 2-component reprojection residual.
pub fn residual(cam: &CameraIn, rot: &matrix3d, point: vect3d, xy: vect2d) -> [f64; 2] {
    let pc = *rot * point + cam.t;
    // BAL convention: cameras look down -z.
    let px = -pc.x / pc.z;
    let py = -pc.y / pc.z;
    let r2 = px * px + py * py;
    let d = 1.0 + r2 * (cam.k1 + cam.k2 * r2);
    [cam.f * d * px - xy.x, cam.f * d * py - xy.y]
}

/// Sum of squared reprojection residuals (no 1/2 factor). THE cost every
/// system optimizes; each system's result is evaluated with this one
/// function.
pub fn reference_cost(ds: &Dataset, cams: &[CameraIn], points: &[vect3d]) -> f64 {
    assert_eq!(cams.len(), ds.cameras.len());
    assert_eq!(points.len(), ds.points.len());
    let rots: Vec<matrix3d> = cams.iter().map(|c| c.rot()).collect();
    let mut cost = 0.0;
    for o in &ds.observations {
        let c = &cams[o.cam as usize];
        let r = residual(c, &rots[o.cam as usize], points[o.point as usize], o.xy);
        cost += r[0] * r[0] + r[1] * r[1];
    }
    cost
}

/// Reprojection error percentiles in pixels: the median and the 95th
/// percentile of the per-observation error norm. The quality figure of a
/// bundle adjustment result -- two solutions whose costs differ in the fifth
/// digit are the same reconstruction when these agree.
pub fn reprojection_percentiles(ds: &Dataset, cams: &[CameraIn], points: &[vect3d]) -> (f64, f64) {
    assert_eq!(cams.len(), ds.cameras.len());
    assert_eq!(points.len(), ds.points.len());
    let rots: Vec<matrix3d> = cams.iter().map(|c| c.rot()).collect();
    let mut errs: Vec<f64> = ds.observations.iter().map(|o| {
        let r = residual(&cams[o.cam as usize], &rots[o.cam as usize], points[o.point as usize], o.xy);
        (r[0] * r[0] + r[1] * r[1]).sqrt()
    }).collect();
    (percentile(&mut errs, 0.5), percentile(&mut errs, 0.95))
}

/// The `q` quantile (0 to 1) of `v` by the nearest-rank rule: the smallest
/// value that at least `q` of the values are at or below. Reorders `v`.
pub fn percentile(v: &mut [f64], q: f64) -> f64 {
    assert!(!v.is_empty(), "percentile of nothing");
    let k = ((q * v.len() as f64).ceil() as usize).clamp(1, v.len()) - 1;
    let (_, x, _) = v.select_nth_unstable_by(k, |a, b| a.partial_cmp(b).expect("NaN error"));
    *x
}

/// Camera optical centers (-R^T t) -- the geometric anchor for
/// validation. Landmark positions are NOT a usable gate: BAL scenes
/// contain weakly-observed points (two rays at tiny parallax) that
/// slide along their rays almost cost-free, so converged solutions
/// with costs within 0.01% can differ by several percent in all-points
/// RMSE. Camera centers are observed by hundreds of residuals each.
pub fn camera_centers(cams: &[CameraIn]) -> Vec<vect3d> {
    cams.iter().map(|c| -(c.rot().transpose() * c.t)).collect()
}

/// RMSE between two solutions' camera centers after the best SIMILARITY
/// (scale + rotation + translation) alignment, relative to the scene
/// extent. Bundle adjustment has a 7-DOF gauge freedom (BAL fixes no
/// gauge), so positions are only comparable up to a similarity
/// transform, and BAL units are arbitrary, so the threshold must be
/// relative.
pub fn aligned_relative_rmse(a: &[vect3d], b: &[vect3d]) -> f64 {
    let n = a.len() as f64;
    let mut ma = vect3d::new(0.0, 0.0, 0.0);
    let mut mb = vect3d::new(0.0, 0.0, 0.0);
    for (pa, pb) in a.iter().zip(b.iter()) {
        ma = ma + *pa;
        mb = mb + *pb;
    }
    let (ma, mb) = (ma * (1.0 / n), mb * (1.0 / n));
    let mut c = matrix3d::zero_matrix();
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for (pa, pb) in a.iter().zip(b.iter()) {
        let x = *pa - ma;
        let y = *pb - mb;
        var_a += x.square();
        var_b += y.square();
        for i in 0..3 {
            let xi = [x.x, x.y, x.z][i];
            c[i] = c[i] + y * xi;
        }
    }
    // Rotation via polar decomposition (as in the pose-graph benchmark),
    // scale from the variance ratio -- the Umeyama solution when the
    // covariance is nonsingular with positive determinant.
    let cct = c * c.transpose();
    let (v, d) = cct.symmetric_eigen();
    assert!(d.x > 0.0 && d.y > 0.0 && d.z > 0.0, "degenerate covariance");
    let inv_sqrt = v
        * matrix3d::from_elements(
            1.0 / d.x.sqrt(), 0.0, 0.0,
            0.0, 1.0 / d.y.sqrt(), 0.0,
            0.0, 0.0, 1.0 / d.z.sqrt(),
        )
        * v.transpose();
    let r = c.transpose() * inv_sqrt;
    assert!(r.det() > 0.0, "reflection in aligned_relative_rmse");
    let s = (var_b / var_a).sqrt();
    let mut err = 0.0;
    for (pa, pb) in a.iter().zip(b.iter()) {
        let e = r * (*pa - ma) * s - (*pb - mb);
        err += e.square();
    }
    // Relative to the aligned scene's RMS extent.
    (err / n).sqrt() / (var_b / n).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_vendored_dataset() {
        let ds = load("datasets/problem-49-7776-pre.txt");
        assert_eq!(ds.cameras.len(), 49);
        assert_eq!(ds.points.len(), 7776);
        assert_eq!(ds.observations.len(), 31843);
        // The initial reference cost of Ladybug-49 is a known quantity
        // (Ceres's own example prints half of it: 8.509125e5).
        let c = reference_cost(&ds, &ds.cameras, &ds.points);
        assert!((c - 1701824.921362).abs() < 1e-3, "initial cost {}", c);
    }

    /// Nearest rank: the median of five values is the third, the 95th
    /// percentile the fifth; of one hundred the 95th is the 95th smallest.
    #[test]
    fn percentiles_by_nearest_rank() {
        let mut v = [5.0, 1.0, 4.0, 2.0, 3.0];
        assert_eq!(percentile(&mut v, 0.5), 3.0);
        assert_eq!(percentile(&mut v, 0.95), 5.0);
        assert_eq!(percentile(&mut v, 0.0), 1.0);
        let mut hundred: Vec<f64> = (1..=100).rev().map(|i| i as f64).collect();
        assert_eq!(percentile(&mut hundred, 0.95), 95.0);
        assert_eq!(percentile(&mut hundred, 0.5), 50.0);
    }

    /// One identity camera at unit focal length looking at points on the
    /// plane z = -1 reprojects each point to its own (x, y); observations
    /// displaced by 1..=20 pixels along x have those errors.
    #[test]
    fn reprojection_percentiles_are_pixel_errors() {
        let cam = CameraIn { rodrigues: vect3d::new(0.0, 0.0, 0.0), t: vect3d::new(0.0, 0.0, 0.0),
                             f: 1.0, k1: 0.0, k2: 0.0 };
        let points: Vec<vect3d> = (0..20).map(|i| vect3d::new(0.1 * i as f64, 0.2, -1.0)).collect();
        let observations = (0..20).map(|i| ObservationIn {
            cam: 0, point: i as u32, xy: vect2d::new(0.1 * i as f64 + (i + 1) as f64, 0.2),
        }).collect();
        let ds = Dataset { cameras: vec![cam.clone()], points: points.clone(), observations };
        let (p50, p95) = reprojection_percentiles(&ds, &[cam], &points);
        assert!((p50 - 10.0).abs() < 1e-12, "median {}", p50);
        assert!((p95 - 19.0).abs() < 1e-12, "95th {}", p95);
    }

    #[test]
    fn camera_center_roundtrip() {
        let ds = load("datasets/problem-49-7776-pre.txt");
        let centers = camera_centers(&ds.cameras);
        // X_cam = R * center + t must be zero.
        for (c, ctr) in ds.cameras.iter().zip(centers.iter()) {
            let x = c.rot() * *ctr + c.t;
            assert!(x.square() < 1e-18);
        }
    }
}
