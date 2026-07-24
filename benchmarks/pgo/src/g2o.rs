// The 2D dataset the runners consume, and the reference cost function
// every system is validated against. Parsing is arael::g2o; this adapter
// resolves the per-edge sqrt-information row weights up front. Only
// diagonal information matrices with I11 == I22 are accepted -- true for
// the canonical M3500 and city10000 files.

pub use arael::g2o::Pose2 as PoseIn;

/// One relative SE2 measurement with resolved sqrt-information row
/// weights.
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

#[derive(Default)]
pub struct Dataset {
    pub poses: Vec<PoseIn>,
    pub edges: Vec<EdgeIn>,
}

/// `unit_weights` replaces the file's information matrices with identity
/// (the configuration tiny-solver's shipped benchmark runs, since its g2o
/// reader drops the info matrices).
pub fn load(path: &str, unit_weights: bool) -> Dataset {
    let ds = arael::g2o::Dataset2::load(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
    let edges = ds.deltas.iter().map(|d| {
        let (wt, wr) = if unit_weights {
            (1.0, 1.0)
        } else {
            d.iso_sqrt_info()
                .unwrap_or_else(|| panic!("non-isotropic information matrix in {}", path))
        };
        EdgeIn { a: d.a, b: d.b, dx: d.dt.x, dy: d.dt.y, dth: d.dth, wt, wr }
    }).collect();
    Dataset { poses: ds.poses, edges }
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
        let gx = a.t.x + ca * e.dx - sa * e.dy - b.t.x;
        let gy = a.t.y + sa * e.dx + ca * e.dy - b.t.y;
        let r0 = (cb * gx + sb * gy) * e.wt;
        let r1 = (-sb * gx + cb * gy) * e.wt;
        let r2 = arael::utils::rad_diff(a.th + e.dth, b.th) * e.wr;
        cost += r0 * r0 + r1 * r1 + r2 * r2;
    }
    let p0 = sol[0];
    let g0 = ds.poses[0];
    cost += (p0.t.x - g0.t.x).powi(2) + (p0.t.y - g0.t.y).powi(2) + (p0.th - g0.th).powi(2);
    cost
}

/// RMSE between two solutions after the best rigid (rotation +
/// translation) alignment -- gauge-independent geometric agreement.
pub fn aligned_rmse(a: &[PoseIn], b: &[PoseIn]) -> f64 {
    let n = a.len() as f64;
    let (max, may) = a.iter().fold((0.0, 0.0), |(x, y), p| (x + p.t.x, y + p.t.y));
    let (mbx, mby) = b.iter().fold((0.0, 0.0), |(x, y), p| (x + p.t.x, y + p.t.y));
    let (max, may, mbx, mby) = (max / n, may / n, mbx / n, mby / n);
    let (mut sxx, mut sxy, mut syx, mut syy) = (0.0, 0.0, 0.0, 0.0);
    for (pa, pb) in a.iter().zip(b.iter()) {
        let (ax, ay) = (pa.t.x - max, pa.t.y - may);
        let (bx, by) = (pb.t.x - mbx, pb.t.y - mby);
        sxx += ax * bx;
        sxy += ax * by;
        syx += ay * bx;
        syy += ay * by;
    }
    let th = (sxy - syx).atan2(sxx + syy);
    let (s, c) = th.sin_cos();
    let mut err = 0.0;
    for (pa, pb) in a.iter().zip(b.iter()) {
        let (ax, ay) = (pa.t.x - max, pa.t.y - may);
        let (bx, by) = (pb.t.x - mbx, pb.t.y - mby);
        let (rx, ry) = (c * ax - s * ay, s * ax + c * ay);
        err += (rx - bx).powi(2) + (ry - by).powi(2);
    }
    (err / n).sqrt()
}
