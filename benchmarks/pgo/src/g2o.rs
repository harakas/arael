// g2o loader and the reference cost function every system is validated
// against.

/// One pose: (x, y, theta).
#[derive(Clone, Copy, Debug)]
pub struct PoseIn {
    pub x: f64,
    pub y: f64,
    pub th: f64,
}

/// One relative SE2 measurement with sqrt-information row weights.
/// Only diagonal information matrices with I11 == I22 are accepted --
/// true for the canonical M3500 and city10000 files.
#[derive(Clone, Copy, Debug)]
pub struct EdgeIn {
    pub a: u32,
    pub b: u32,
    pub dx: f64,
    pub dy: f64,
    pub dth: f64,
    pub wt: f64, // sqrt(I11) == sqrt(I22)
    pub wr: f64, // sqrt(I33)
}

pub struct Dataset {
    pub poses: Vec<PoseIn>,
    pub edges: Vec<EdgeIn>,
}

/// `unit_weights` replaces the file's information matrices with identity
/// (the configuration tiny-solver's shipped benchmark runs, since its g2o
/// reader drops the info matrices).
pub fn load(path: &str, unit_weights: bool) -> Dataset {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read {}: {}", path, e));
    let mut poses = Vec::new();
    let mut edges = Vec::new();
    for line in text.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        match f.first().copied() {
            Some("VERTEX_SE2") => {
                let id: usize = f[1].parse().unwrap();
                assert_eq!(id, poses.len(), "vertices must be dense and ordered");
                poses.push(PoseIn {
                    x: f[2].parse().unwrap(),
                    y: f[3].parse().unwrap(),
                    th: f[4].parse().unwrap(),
                });
            }
            Some("EDGE_SE2") => {
                let i11: f64 = f[6].parse().unwrap();
                let i12: f64 = f[7].parse().unwrap();
                let i13: f64 = f[8].parse().unwrap();
                let i22: f64 = f[9].parse().unwrap();
                let i23: f64 = f[10].parse().unwrap();
                let i33: f64 = f[11].parse().unwrap();
                assert!(
                    i12.abs() < 1e-9 && i13.abs() < 1e-9 && i23.abs() < 1e-9,
                    "non-diagonal information matrix in {}", path
                );
                assert!((i11 - i22).abs() < 1e-9, "anisotropic translation info in {}", path);
                edges.push(EdgeIn {
                    a: f[1].parse().unwrap(),
                    b: f[2].parse().unwrap(),
                    dx: f[3].parse().unwrap(),
                    dy: f[4].parse().unwrap(),
                    dth: f[5].parse().unwrap(),
                    wt: if unit_weights { 1.0 } else { i11.sqrt() },
                    wr: if unit_weights { 1.0 } else { i33.sqrt() },
                });
            }
            _ => {}
        }
    }
    Dataset { poses, edges }
}

/// Sum of squared weighted residuals (no 1/2 factor), over the between
/// factors plus the unit-weight gauge prior on pose 0. This is THE cost
/// every system optimizes; each system's result is evaluated with this
/// one function.
pub fn reference_cost(ds: &Dataset, sol: &[PoseIn]) -> f64 {
    assert_eq!(sol.len(), ds.poses.len());
    let mut cost = 0.0;
    for e in &ds.edges {
        let a = sol[e.a as usize];
        let b = sol[e.b as usize];
        let (sa, ca) = a.th.sin_cos();
        let (sb, cb) = b.th.sin_cos();
        let gx = a.x + ca * e.dx - sa * e.dy - b.x;
        let gy = a.y + sa * e.dx + ca * e.dy - b.y;
        let r0 = (cb * gx + sb * gy) * e.wt;
        let r1 = (-sb * gx + cb * gy) * e.wt;
        let r2 = arael::utils::rad_diff(a.th + e.dth, b.th) * e.wr;
        cost += r0 * r0 + r1 * r1 + r2 * r2;
    }
    let p0 = sol[0];
    let g0 = ds.poses[0];
    cost += (p0.x - g0.x).powi(2) + (p0.y - g0.y).powi(2) + (p0.th - g0.th).powi(2);
    cost
}

/// RMSE between two solutions after the best rigid (rotation +
/// translation) alignment -- gauge-independent geometric agreement.
pub fn aligned_rmse(a: &[PoseIn], b: &[PoseIn]) -> f64 {
    let n = a.len() as f64;
    let (max, may) = a.iter().fold((0.0, 0.0), |(x, y), p| (x + p.x, y + p.y));
    let (mbx, mby) = b.iter().fold((0.0, 0.0), |(x, y), p| (x + p.x, y + p.y));
    let (max, may, mbx, mby) = (max / n, may / n, mbx / n, mby / n);
    let (mut sxx, mut sxy, mut syx, mut syy) = (0.0, 0.0, 0.0, 0.0);
    for (pa, pb) in a.iter().zip(b.iter()) {
        let (ax, ay) = (pa.x - max, pa.y - may);
        let (bx, by) = (pb.x - mbx, pb.y - mby);
        sxx += ax * bx;
        sxy += ax * by;
        syx += ay * bx;
        syy += ay * by;
    }
    let th = (sxy - syx).atan2(sxx + syy);
    let (s, c) = th.sin_cos();
    let mut err = 0.0;
    for (pa, pb) in a.iter().zip(b.iter()) {
        let (ax, ay) = (pa.x - max, pa.y - may);
        let (bx, by) = (pb.x - mbx, pb.y - mby);
        let (rx, ry) = (c * ax - s * ay, s * ax + c * ay);
        err += (rx - bx).powi(2) + (ry - by).powi(2);
    }
    (err / n).sqrt()
}
