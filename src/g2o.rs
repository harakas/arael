// .g2o pose-graph file I/O: the SE2 and SE3:QUAT subset.
//
// Records handled: VERTEX_SE2, EDGE_SE2, VERTEX_SE3:QUAT, EDGE_SE3:QUAT.
// Unknown record types are skipped. Vertex ids must be dense and ordered
// (0, 1, 2, ...), so measurements reference poses by plain index.
// Information matrices are kept as read; how to weight them is the
// caller's decision.

use crate::matrix::matrix3d;
use crate::quatern::quaternd;
use crate::vect::{vect2d, vect3d};

/// Failure to read or parse a .g2o file.
#[derive(Debug)]
pub enum G2oError {
    /// The file could not be read.
    Io(std::io::Error),
    /// A record could not be parsed; `line` is 1-based.
    Parse {
        /// 1-based line number of the offending record.
        line: usize,
        /// What was wrong with it.
        why: String,
    },
}

impl std::fmt::Display for G2oError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            G2oError::Io(e) => write!(f, "cannot read g2o file: {}", e),
            G2oError::Parse { line, why } => write!(f, "g2o line {}: {}", line, why),
        }
    }
}

impl std::error::Error for G2oError {}

impl From<std::io::Error> for G2oError {
    fn from(e: std::io::Error) -> Self {
        G2oError::Io(e)
    }
}

/// One 2D pose from a VERTEX_SE2 record.
#[derive(Clone, Copy, Debug)]
pub struct Pose2 {
    /// Position.
    pub t: vect2d,
    /// Heading.
    pub th: f64,
}

/// One relative SE2 measurement from an EDGE_SE2 record: pose `b` seen
/// from pose `a`'s body frame.
#[derive(Clone, Copy, Debug)]
pub struct DeltaPose2 {
    /// Index of the observing pose.
    pub a: u32,
    /// Index of the observed pose.
    pub b: u32,
    /// Measured translation in `a`'s frame.
    pub dt: vect2d,
    /// Measured heading change.
    pub dth: f64,
    /// Information matrix upper triangle in file order:
    /// I11 I12 I13 I22 I23 I33, rows ordered (x y theta).
    pub info: [f64; 6],
}

impl DeltaPose2 {
    /// Sqrt-information row weights `(wt, wr)` when the information
    /// matrix is diagonal with equal translation entries -- the shape of
    /// the canonical 2D datasets. `None` when it is anything else.
    pub fn iso_sqrt_info(&self) -> Option<(f64, f64)> {
        let [i11, i12, i13, i22, i23, i33] = self.info;
        if i12.abs() < 1e-9 && i13.abs() < 1e-9 && i23.abs() < 1e-9
            && (i11 - i22).abs() < 1e-9
        {
            Some((i11.sqrt(), i33.sqrt()))
        } else {
            None
        }
    }

    /// Eigen factors `(r, w)` of the information matrix for exact
    /// whitening of any symmetric information matrix:
    /// `info = r * diag(w)^2 * r^T`, so the weighted residual is
    /// `diag(w) * r^T * res`. Eigenvalues below zero (numerically
    /// indefinite input) clamp to zero weight, so near-singular
    /// matrices lose the degenerate direction instead of failing a
    /// factorization.
    pub fn eigen_sqrt_info(&self) -> (matrix3d, vect3d) {
        let [i11, i12, i13, i22, i23, i33] = self.info;
        let m = matrix3d::from_array([
            [i11, i12, i13],
            [i12, i22, i23],
            [i13, i23, i33],
        ]);
        let (r, d) = m.symmetric_eigen();
        (r, vect3d::new(
            d.x.max(0.0).sqrt(),
            d.y.max(0.0).sqrt(),
            d.z.max(0.0).sqrt(),
        ))
    }
}

/// A 2D pose graph: poses and the relative measurements between them.
#[derive(Clone, Debug, Default)]
pub struct Dataset2 {
    /// Poses, indexed by vertex id.
    pub poses: Vec<Pose2>,
    /// Relative measurements between pose pairs.
    pub deltas: Vec<DeltaPose2>,
}

/// One 3D pose from a VERTEX_SE3:QUAT record.
#[derive(Clone, Copy, Debug)]
pub struct Pose3 {
    /// Position.
    pub t: vect3d,
    /// Orientation (unit quaternion; normalized on load).
    pub q: quaternd,
}

impl Pose3 {
    /// The pose's rotation matrix.
    pub fn rot(&self) -> matrix3d {
        self.q.rotation_matrix()
    }
}

/// One relative SE3 measurement from an EDGE_SE3:QUAT record: pose `b`
/// seen from pose `a`'s body frame.
#[derive(Clone, Copy, Debug)]
pub struct DeltaPose3 {
    /// Index of the observing pose.
    pub a: u32,
    /// Index of the observed pose.
    pub b: u32,
    /// Measured translation in `a`'s frame.
    pub dt: vect3d,
    /// Measured rotation change (unit quaternion; normalized on load).
    pub dq: quaternd,
    /// Full symmetric 6x6 information matrix, rows ordered
    /// (x y z qx qy qz).
    pub info: [[f64; 6]; 6],
}

impl DeltaPose3 {
    /// Upper Cholesky factor `u` of the information matrix
    /// (`info = u^T u`). Panics when the matrix is not positive
    /// definite -- that is a data error, not something to paper over.
    pub fn sqrt_info_upper(&self) -> [[f64; 6]; 6] {
        // Lower Cholesky info = l l^T, returned transposed.
        let a = &self.info;
        let mut l = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..=i {
                let mut s = a[i][j];
                for k in 0..j {
                    s -= l[i][k] * l[j][k];
                }
                if i == j {
                    assert!(s > 0.0, "information matrix not positive definite (pivot {} = {})", i, s);
                    l[i][j] = s.sqrt();
                } else {
                    l[i][j] = s / l[j][j];
                }
            }
        }
        let mut u = [[0.0f64; 6]; 6];
        for i in 0..6 {
            for j in 0..6 {
                u[i][j] = l[j][i];
            }
        }
        u
    }

    /// The three 3x3 blocks of the upper-triangular sqrt-info factor:
    /// `[ u_tt u_tr ; 0 u_rr ]`.
    pub fn u_blocks(&self) -> (matrix3d, matrix3d, matrix3d) {
        let u = self.sqrt_info_upper();
        let b = |r0: usize, c0: usize| {
            matrix3d::from_elements(
                u[r0][c0], u[r0][c0 + 1], u[r0][c0 + 2],
                u[r0 + 1][c0], u[r0 + 1][c0 + 1], u[r0 + 1][c0 + 2],
                u[r0 + 2][c0], u[r0 + 2][c0 + 1], u[r0 + 2][c0 + 2],
            )
        };
        (b(0, 0), b(0, 3), b(3, 3))
    }
}

/// A 3D pose graph: poses and the relative measurements between them.
#[derive(Clone, Debug, Default)]
pub struct Dataset3 {
    /// Poses, indexed by vertex id.
    pub poses: Vec<Pose3>,
    /// Relative measurements between pose pairs.
    pub deltas: Vec<DeltaPose3>,
}

// ------------------------------------------------------------- parsing

/// One whitespace-split record with error context.
struct Rec<'a> {
    f: Vec<&'a str>,
    line: usize,
}

impl<'a> Rec<'a> {
    fn err(&self, why: impl Into<String>) -> G2oError {
        G2oError::Parse { line: self.line, why: why.into() }
    }

    fn need(&self, n: usize) -> Result<(), G2oError> {
        if self.f.len() < n {
            Err(self.err(format!("{} needs {} fields, got {}", self.f[0], n, self.f.len())))
        } else {
            Ok(())
        }
    }

    fn f64(&self, i: usize) -> Result<f64, G2oError> {
        self.f[i].parse().map_err(|_| self.err(format!("bad number {:?} (field {})", self.f[i], i)))
    }

    fn u32(&self, i: usize) -> Result<u32, G2oError> {
        self.f[i].parse().map_err(|_| self.err(format!("bad id {:?} (field {})", self.f[i], i)))
    }

    fn dense_id(&self, i: usize, len: usize) -> Result<(), G2oError> {
        if self.u32(i)? as usize != len {
            Err(self.err(format!("vertex ids must be dense and ordered (id {} at count {})", self.f[i], len)))
        } else {
            Ok(())
        }
    }
}

fn records(text: &str) -> impl Iterator<Item = Rec<'_>> {
    text.lines().enumerate().filter_map(|(i, l)| {
        let f: Vec<&str> = l.split_whitespace().collect();
        if f.is_empty() { None } else { Some(Rec { f, line: i + 1 }) }
    })
}

fn check_indices(n_poses: usize, pairs: impl Iterator<Item = (u32, u32)>) -> Result<(), G2oError> {
    for (a, b) in pairs {
        if a as usize >= n_poses || b as usize >= n_poses {
            return Err(G2oError::Parse {
                line: 0,
                why: format!("measurement references pose {} / {} of {}", a, b, n_poses),
            });
        }
    }
    Ok(())
}

impl Dataset2 {
    /// Read a 2D pose graph from a .g2o file.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Dataset2, G2oError> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// Parse a 2D pose graph from .g2o text.
    pub fn parse(text: &str) -> Result<Dataset2, G2oError> {
        let mut ds = Dataset2::default();
        for r in records(text) {
            match r.f[0] {
                "VERTEX_SE2" => {
                    // VERTEX_SE2 id x y theta
                    r.need(5)?;
                    r.dense_id(1, ds.poses.len())?;
                    ds.poses.push(Pose2 {
                        t: vect2d::new(r.f64(2)?, r.f64(3)?),
                        th: r.f64(4)?,
                    });
                }
                "EDGE_SE2" => {
                    // EDGE_SE2 a b dx dy dth I11 I12 I13 I22 I23 I33
                    r.need(12)?;
                    let mut info = [0.0; 6];
                    for (k, v) in info.iter_mut().enumerate() {
                        *v = r.f64(6 + k)?;
                    }
                    ds.deltas.push(DeltaPose2 {
                        a: r.u32(1)?,
                        b: r.u32(2)?,
                        dt: vect2d::new(r.f64(3)?, r.f64(4)?),
                        dth: r.f64(5)?,
                        info,
                    });
                }
                _ => {}
            }
        }
        check_indices(ds.poses.len(), ds.deltas.iter().map(|d| (d.a, d.b)))?;
        Ok(ds)
    }

    /// Write the pose graph back out as .g2o.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_g2o())
    }

    /// Render the pose graph as .g2o text.
    pub fn to_g2o(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        for (i, p) in self.poses.iter().enumerate() {
            writeln!(out, "VERTEX_SE2 {} {} {} {}", i, p.t.x, p.t.y, p.th).unwrap();
        }
        for d in &self.deltas {
            write!(out, "EDGE_SE2 {} {} {} {} {}", d.a, d.b, d.dt.x, d.dt.y, d.dth).unwrap();
            for v in d.info {
                write!(out, " {}", v).unwrap();
            }
            writeln!(out).unwrap();
        }
        out
    }
}

impl Dataset3 {
    /// Read a 3D pose graph from a .g2o file.
    pub fn load(path: impl AsRef<std::path::Path>) -> Result<Dataset3, G2oError> {
        Self::parse(&std::fs::read_to_string(path)?)
    }

    /// Parse a 3D pose graph from .g2o text.
    pub fn parse(text: &str) -> Result<Dataset3, G2oError> {
        let quat = |r: &Rec, i: usize| -> Result<quaternd, G2oError> {
            // File order qx qy qz qw.
            let v = vect3d::new(r.f64(i)?, r.f64(i + 1)?, r.f64(i + 2)?);
            let q = quaternd::new(r.f64(i + 3)?, v);
            let n = (q.t * q.t + q.v.square()).sqrt();
            if n < 1e-12 {
                return Err(r.err("zero-length quaternion"));
            }
            Ok(quaternd::new(q.t / n, q.v * (1.0 / n)))
        };
        let mut ds = Dataset3::default();
        for r in records(text) {
            match r.f[0] {
                "VERTEX_SE3:QUAT" => {
                    // VERTEX_SE3:QUAT id x y z qx qy qz qw
                    r.need(9)?;
                    r.dense_id(1, ds.poses.len())?;
                    ds.poses.push(Pose3 {
                        t: vect3d::new(r.f64(2)?, r.f64(3)?, r.f64(4)?),
                        q: quat(&r, 5)?,
                    });
                }
                "EDGE_SE3:QUAT" => {
                    // EDGE_SE3:QUAT a b dx dy dz qx qy qz qw then the 21
                    // upper-triangular information entries, row-major,
                    // rows ordered (x y z qx qy qz).
                    r.need(31)?;
                    let mut info = [[0.0f64; 6]; 6];
                    let mut k = 10;
                    for i in 0..6 {
                        for j in i..6 {
                            let v = r.f64(k)?;
                            info[i][j] = v;
                            info[j][i] = v;
                            k += 1;
                        }
                    }
                    ds.deltas.push(DeltaPose3 {
                        a: r.u32(1)?,
                        b: r.u32(2)?,
                        dt: vect3d::new(r.f64(3)?, r.f64(4)?, r.f64(5)?),
                        dq: quat(&r, 6)?,
                        info,
                    });
                }
                _ => {}
            }
        }
        check_indices(ds.poses.len(), ds.deltas.iter().map(|d| (d.a, d.b)))?;
        Ok(ds)
    }

    /// Write the pose graph back out as .g2o.
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_g2o())
    }

    /// Render the pose graph as .g2o text.
    pub fn to_g2o(&self) -> String {
        use std::fmt::Write as _;
        let mut out = String::new();
        for (i, p) in self.poses.iter().enumerate() {
            writeln!(out, "VERTEX_SE3:QUAT {} {} {} {} {} {} {} {}",
                i, p.t.x, p.t.y, p.t.z, p.q.v.x, p.q.v.y, p.q.v.z, p.q.t).unwrap();
        }
        for d in &self.deltas {
            write!(out, "EDGE_SE3:QUAT {} {} {} {} {} {} {} {} {}",
                d.a, d.b, d.dt.x, d.dt.y, d.dt.z, d.dq.v.x, d.dq.v.y, d.dq.v.z, d.dq.t).unwrap();
            for i in 0..6 {
                for j in i..6 {
                    write!(out, " {}", d.info[i][j]).unwrap();
                }
            }
            writeln!(out).unwrap();
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEXT2: &str = "\
VERTEX_SE2 0 0.0 0.0 0.0
VERTEX_SE2 1 1.0 0.5 0.1
FIX 0
EDGE_SE2 0 1 1.0 0.5 0.1 100 0 0 100 0 400
";

    #[test]
    fn parses_2d() {
        let ds = Dataset2::parse(TEXT2).unwrap();
        assert_eq!(ds.poses.len(), 2);
        assert_eq!(ds.deltas.len(), 1);
        assert_eq!((ds.poses[1].t.x, ds.poses[1].t.y), (1.0, 0.5));
        assert_eq!(ds.poses[1].th, 0.1);
        let d = &ds.deltas[0];
        assert_eq!((d.a, d.b), (0, 1));
        assert_eq!((d.dt.x, d.dt.y), (1.0, 0.5));
        assert_eq!(d.info, [100.0, 0.0, 0.0, 100.0, 0.0, 400.0]);
        assert_eq!(d.iso_sqrt_info(), Some((10.0, 20.0)));
    }

    #[test]
    fn non_iso_info_is_reported() {
        let mut d = Dataset2::parse(TEXT2).unwrap().deltas[0];
        d.info[1] = 1.0; // off-diagonal
        assert_eq!(d.iso_sqrt_info(), None);
        d.info[1] = 0.0;
        d.info[3] = 50.0; // anisotropic translation
        assert_eq!(d.iso_sqrt_info(), None);
    }

    #[test]
    fn eigen_sqrt_info_reconstructs_correlated_info() {
        let mut d = Dataset2::parse(TEXT2).unwrap().deltas[0];
        // MIT-like: anisotropic with off-diagonal couplings
        d.info = [1.78, 0.027, 0.0, 3.85, 0.0, 388.7];
        let (r, w) = d.eigen_sqrt_info();
        // info == r * diag(w)^2 * r^T
        let full = [
            [d.info[0], d.info[1], d.info[2]],
            [d.info[1], d.info[3], d.info[4]],
            [d.info[2], d.info[4], d.info[5]],
        ];
        for i in 0..3 {
            for j in 0..3 {
                let mut v = 0.0;
                for k in 0..3 {
                    v += r[i][k] * w[k] * w[k] * r[j][k];
                }
                assert!((v - full[i][j]).abs() < 1e-9, "({i},{j}): {v} vs {}", full[i][j]);
            }
        }
    }

    #[test]
    fn eigen_sqrt_info_clamps_indefinite() {
        let mut d = Dataset2::parse(TEXT2).unwrap().deltas[0];
        // numerically indefinite (off-diagonal above the PSD bound)
        d.info = [1.0, 1.001, 0.0, 1.0, 0.0, 4.0];
        let (_, w) = d.eigen_sqrt_info();
        assert!(w.x >= 0.0 && w.y >= 0.0 && w.z >= 0.0);
        assert!(w.x == 0.0, "negative eigenvalue must clamp to zero weight");
    }

    #[test]
    fn roundtrips_2d() {
        let ds = Dataset2::parse(TEXT2).unwrap();
        let back = Dataset2::parse(&ds.to_g2o()).unwrap();
        assert_eq!(back.poses.len(), ds.poses.len());
        for (p, q) in ds.poses.iter().zip(back.poses.iter()) {
            assert_eq!((p.t.x, p.t.y, p.th), (q.t.x, q.t.y, q.th));
        }
        for (d, e) in ds.deltas.iter().zip(back.deltas.iter()) {
            assert_eq!((d.a, d.b, d.dt.x, d.dt.y, d.dth), (e.a, e.b, e.dt.x, e.dt.y, e.dth));
            assert_eq!(d.info, e.info);
        }
    }

    #[test]
    fn rejects_sparse_ids() {
        let e = Dataset2::parse("VERTEX_SE2 1 0 0 0\n").unwrap_err();
        assert!(matches!(e, G2oError::Parse { line: 1, .. }), "{:?}", e);
        assert!(e.to_string().contains("dense"), "{}", e);
    }

    #[test]
    fn rejects_bad_number_with_line() {
        let e = Dataset2::parse("VERTEX_SE2 0 0 0 0\nVERTEX_SE2 1 x 0 0\n").unwrap_err();
        assert!(matches!(e, G2oError::Parse { line: 2, .. }), "{:?}", e);
    }

    #[test]
    fn rejects_short_record() {
        let e = Dataset2::parse("EDGE_SE2 0 1 1 0 0 100 0 0 100 0\n").unwrap_err();
        assert!(e.to_string().contains("12 fields"), "{}", e);
    }

    #[test]
    fn rejects_out_of_range_reference() {
        let e = Dataset2::parse("VERTEX_SE2 0 0 0 0\nEDGE_SE2 0 3 1 0 0 100 0 0 100 0 400\n")
            .unwrap_err();
        assert!(e.to_string().contains("references pose"), "{}", e);
    }

    // 21 info entries: a diagonal 6x6 (2 2 2 4 4 4) in upper-tri row-major.
    const TEXT3: &str = "\
VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1
VERTEX_SE3:QUAT 1 1 0 0 0 0 0.09983341664682815 0.9950041652780258
EDGE_SE3:QUAT 0 1 1 0 0 0 0 0.09983341664682815 0.9950041652780258 2 0 0 0 0 0 2 0 0 0 0 2 0 0 0 4 0 0 4 0 4
";

    #[test]
    fn parses_3d() {
        let ds = Dataset3::parse(TEXT3).unwrap();
        assert_eq!(ds.poses.len(), 2);
        assert_eq!(ds.deltas.len(), 1);
        // The quaternion is sin/cos of a 0.2 rad yaw about z.
        let q = ds.poses[1].q;
        assert!((q.t * q.t + q.v.square() - 1.0).abs() < 1e-12);
        let r = ds.poses[1].rot();
        assert!((r[0].x - 0.2f64.cos()).abs() < 1e-12);
        let d = &ds.deltas[0];
        assert_eq!(d.info[0][0], 2.0);
        assert_eq!(d.info[3][3], 4.0);
        assert_eq!(d.info[0][3], 0.0);
    }

    #[test]
    fn normalizes_quaternions() {
        let text = "VERTEX_SE3:QUAT 0 0 0 0 0 0 0.2 2.0\n";
        let q = Dataset3::parse(text).unwrap().poses[0].q;
        assert!((q.t * q.t + q.v.square() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn sqrt_info_roundtrips() {
        let mut d = Dataset3::parse(TEXT3).unwrap().deltas[0];
        // A denser SPD matrix: diag + coupling.
        for i in 0..6 {
            for j in 0..6 {
                d.info[i][j] = if i == j { 4.0 + i as f64 } else { 0.5 / (1.0 + (i as f64 - j as f64).abs()) };
            }
        }
        let u = d.sqrt_info_upper();
        for i in 0..6 {
            for j in 0..6 {
                let mut s = 0.0;
                for r in 0..6 {
                    s += u[r][i] * u[r][j];
                }
                assert!((s - d.info[i][j]).abs() < 1e-12, "info[{}][{}]", i, j);
            }
        }
        // u_blocks are the corners of u.
        let (utt, utr, urr) = d.u_blocks();
        assert_eq!(utt[0].x, u[0][0]);
        assert_eq!(utr[0].x, u[0][3]);
        assert_eq!(urr[2].z, u[5][5]);
    }

    #[test]
    fn roundtrips_3d() {
        let ds = Dataset3::parse(TEXT3).unwrap();
        let back = Dataset3::parse(&ds.to_g2o()).unwrap();
        assert_eq!(back.poses.len(), ds.poses.len());
        let (p, q) = (&ds.deltas[0], &back.deltas[0]);
        assert_eq!((p.dt.x, p.dt.y, p.dt.z), (q.dt.x, q.dt.y, q.dt.z));
        assert_eq!((p.dq.t, p.dq.v.x, p.dq.v.y, p.dq.v.z), (q.dq.t, q.dq.v.x, q.dq.v.y, q.dq.v.z));
        assert_eq!(p.info, q.info);
    }
}
