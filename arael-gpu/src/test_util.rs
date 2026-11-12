//! Synthetic matrices for the tests: a grid graph of blocks of mixed width,
//! random values, diagonal dominance, and a dense twin for residuals. And
//! what a test runs in on its device: f64 and df32, or df32 alone on a
//! device without f64 (Metal).

use arael_faer::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};

use crate::plan::SparseIndex;
use crate::{Device, GpuPrecision, Wide};

/// Whether `dev` computes in f64; without, it prints that `what` is
/// skipped, so a test run says what it left out.
pub fn has_f64<D: Device>(dev: &D, what: &str) -> bool {
    let has = dev.arch().has_f64;
    if !has {
        eprintln!("{} has no f64: {what} skipped", dev.arch().name);
    }
    has
}

/// The wide scalar of a test about something else: f64, or df32 on a
/// device without f64.
pub fn wide<D: Device>(dev: &D) -> Wide {
    if dev.arch().has_f64 { Wide::F64 } else { Wide::Df32 }
}

/// The wide scalars `dev` computes in.
pub fn wides<D: Device>(dev: &D) -> Vec<Wide> {
    if has_f64(dev, "Wide::F64") { vec![Wide::F64, Wide::Df32] } else { vec![Wide::Df32] }
}

/// The factor scalars `dev` computes in.
pub fn precisions<D: Device>(dev: &D) -> Vec<GpuPrecision> {
    if has_f64(dev, "the f64 factor") { vec![GpuPrecision::F64, GpuPrecision::F32] } else { vec![GpuPrecision::F32] }
}

pub struct Lcg(pub u64);

impl Lcg {
    pub fn next_bits(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f64 {
        self.next_bits() as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// A `k x k` grid of blocks, block `(r, c)` adjacent to its right and lower
/// neighbours, block widths cycling through `widths`; stored upper, the
/// diagonal tiles upper-only. Returns the matrix, its dense twin (column
/// major, full) and a right-hand side.
pub fn grid_spd(k: usize, widths: &[usize], seed: u64) -> (SparseBlockColMat<SparseIndex, f64>, Vec<f64>, Vec<f64>) {
    let (a, rhs) = grid_spd_sparse(k, widths, seed);
    let dense = dense_of(&a);
    (a, dense, rhs)
}

/// [`grid_spd`] without the dense twin, for grids too large for one.
pub fn grid_spd_sparse(k: usize, widths: &[usize], seed: u64) -> (SparseBlockColMat<SparseIndex, f64>, Vec<f64>) {
    let nblk = k * k;
    let mut part = vec![0u32];
    for b in 0..nblk {
        part.push(part[b] + widths[b % widths.len()] as u32);
    }
    let n = part[nblk] as usize;
    let mut cells: Vec<(usize, usize)> = Vec::new();
    for r in 0..k {
        for c in 0..k {
            let b = r * k + c;
            cells.push((b, b));
            if c + 1 < k {
                cells.push((b, b + 1));
            }
            if r + 1 < k {
                cells.push((b, b + k));
            }
        }
    }
    let (sym, _) = SymbolicSparseBlockColMat::<SparseIndex>::from_scalar_coords(part.clone(), part.clone(), cells.len(), |i| {
        (part[cells[i].0] as usize, part[cells[i].1] as usize)
    });
    let mut a = SparseBlockColMat::<SparseIndex, f64>::zeroed(sym);
    let asym = a.symbolic().clone();
    let mut rng = Lcg(seed);
    for j in 0..asym.nblk_cols() {
        for b in asym.col_range(j) {
            let i = asym.blk_row(b);
            let (wi, wj) = asym.block_dims(b);
            let base = asym.val_range(b).start;
            for cc in 0..wj {
                for rr in 0..wi {
                    if i == j && rr > cc {
                        continue;
                    }
                    let v = rng.unit();
                    a.vals_mut()[base + rr + cc * wi] = v;
                }
            }
        }
    }
    // dominance: five neighbouring blocks of the widest width at most
    let boost = 4.0 * *widths.iter().max().unwrap() as f64 * 5.0;
    for j in 0..asym.nblk_cols() {
        let b = asym.col_range(j).find(|&b| asym.blk_row(b) == j).unwrap();
        let (w, _) = asym.block_dims(b);
        let base = asym.val_range(b).start;
        for kk in 0..w {
            a.vals_mut()[base + kk * (w + 1)] += boost;
        }
    }
    let rhs: Vec<f64> = (0..n).map(|_| rng.unit() * 2.0).collect();
    (a, rhs)
}

/// The dense twin of upper-stored `a`: column major, full.
pub fn dense_of(a: &SparseBlockColMat<SparseIndex, f64>) -> Vec<f64> {
    let asym = a.symbolic();
    let n = asym.ncols();
    let mut dense = vec![0.0f64; n * n];
    for j in 0..asym.nblk_cols() {
        let cj = asym.col_span(j).start;
        for b in asym.col_range(j) {
            let i = asym.blk_row(b);
            let ri = asym.row_span(i).start;
            let (wi, wj) = asym.block_dims(b);
            let base = asym.val_range(b).start;
            for cc in 0..wj {
                for rr in 0..wi {
                    if i == j && rr > cc {
                        continue;
                    }
                    let v = a.vals()[base + rr + cc * wi];
                    dense[(ri + rr) + (cj + cc) * n] = v;
                    dense[(cj + cc) + (ri + rr) * n] = v;
                }
            }
        }
    }
    dense
}

/// `|A x - b| / |b|` through the dense twin.
pub fn rel_residual(dense: &[f64], n: usize, x: &[f64], rhs: &[f64]) -> f64 {
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for r in 0..n {
        let mut ax = 0.0;
        for c in 0..n {
            ax += dense[r + c * n] * x[c];
        }
        num += (ax - rhs[r]) * (ax - rhs[r]);
        den += rhs[r] * rhs[r];
    }
    (num / den).sqrt()
}

/// A bundle-adjustment-like block system: `np` poses of width 6 in a chain
/// (odometry couplings), `nl` landmarks of width 3 (every seventh of width
/// 2), each observed by two to four consecutive poses. Half the landmarks
/// get block ids before the poses, so coupling tiles come in both storage
/// orientations. Returns the matrix (upper stored), its dense twin, a
/// right-hand side, and the landmark block ids to eliminate (ascending).
pub fn ba_like(np: usize, nl: usize, seed: u64) -> (SparseBlockColMat<SparseIndex, f64>, Vec<f64>, Vec<f64>, Vec<usize>) {
    let half = nl / 2;
    // block ids: landmarks 0..half, poses half..half+np, landmarks half+np..
    let nblk = np + nl;
    let is_landmark = |b: usize| b < half || b >= half + np;
    let landmark_index = |b: usize| if b < half { b } else { b - np };
    let pose_id = |p: usize| half + p;
    let width = |b: usize| -> usize {
        if is_landmark(b) { if landmark_index(b) % 7 == 3 { 2 } else { 3 } } else { 6 }
    };
    let mut part = vec![0u32];
    for b in 0..nblk {
        part.push(part[b] + width(b) as u32);
    }
    let n = part[nblk] as usize;
    let mut rng = Lcg(seed);
    let mut cells: Vec<(usize, usize)> = Vec::new();
    for b in 0..nblk {
        cells.push((b, b));
    }
    for p in 0..np.saturating_sub(1) {
        cells.push((pose_id(p), pose_id(p + 1)));
    }
    for b in 0..nblk {
        if !is_landmark(b) {
            continue;
        }
        let l = landmark_index(b);
        let first = (l * np / nl.max(1)).min(np - 1);
        let count = 2 + (rng.next_bits() % 3) as usize;
        for p in first..(first + count).min(np) {
            let (r, c) = if b < pose_id(p) { (b, pose_id(p)) } else { (pose_id(p), b) };
            cells.push((r, c));
        }
    }
    cells.sort_unstable();
    cells.dedup();
    let (sym, _) = SymbolicSparseBlockColMat::<SparseIndex>::from_scalar_coords(part.clone(), part.clone(), cells.len(), |i| {
        (part[cells[i].0] as usize, part[cells[i].1] as usize)
    });
    let mut a = SparseBlockColMat::<SparseIndex, f64>::zeroed(sym);
    let asym = a.symbolic().clone();
    let mut dense = vec![0.0f64; n * n];
    for j in 0..asym.nblk_cols() {
        let cj = asym.col_span(j).start;
        for b in asym.col_range(j) {
            let i = asym.blk_row(b);
            let ri = asym.row_span(i).start;
            let (wi, wj) = asym.block_dims(b);
            let base = asym.val_range(b).start;
            for cc in 0..wj {
                for rr in 0..wi {
                    if i == j && rr > cc {
                        continue;
                    }
                    let v = rng.unit();
                    a.vals_mut()[base + rr + cc * wi] = v;
                    dense[(ri + rr) + (cj + cc) * n] = v;
                    dense[(cj + cc) + (ri + rr) * n] = v;
                }
            }
        }
    }
    // dominance: a pose sees up to 4 * nl / np landmarks of width 3 plus two poses
    let boost = 6.0 * (4.0 * nl as f64 / np.max(1) as f64 * 3.0 + 12.0 + 6.0);
    for j in 0..asym.nblk_cols() {
        let b = asym.col_range(j).find(|&b| asym.blk_row(b) == j).unwrap();
        let (w, _) = asym.block_dims(b);
        let base = asym.val_range(b).start;
        let cj = asym.col_span(j).start;
        for kk in 0..w {
            a.vals_mut()[base + kk * (w + 1)] += boost;
            dense[(cj + kk) * (n + 1)] += boost;
        }
    }
    let rhs: Vec<f64> = (0..n).map(|_| rng.unit() * 2.0).collect();
    let eliminated: Vec<usize> = (0..nblk).filter(|&b| is_landmark(b)).collect();
    (a, dense, rhs, eliminated)
}

/// A bundle adjustment's Gauss-Newton Hessian `J^T J` on [`ba_like`]'s
/// structure, its diagonal damped by `1 + lambda`. An observation is two
/// residual rows over its pose and its landmark, an odometry coupling six
/// over its two poses. A landmark's rows scale one direction of its columns
/// by `eps`, as for a point seen along nearly parallel rays, so its block's
/// condition number is about `min(1 / eps^2, 1 / lambda)`. Returns the
/// matrix (upper stored), the right-hand side `J^T r` for random residuals
/// `r`, and the landmark block ids.
pub fn ba_ill_conditioned(np: usize, nl: usize, eps: f64, lambda: f64, seed: u64) -> (SparseBlockColMat<SparseIndex, f64>, Vec<f64>, Vec<usize>) {
    let (mut a, _, mut rhs, eliminated) = ba_like(np, nl, seed);
    let sym = a.symbolic().clone();
    let nblk = sym.nblk_cols();
    let mut landmark = vec![false; nblk];
    for &b in &eliminated {
        landmark[b] = true;
    }
    let width = |b: usize| sym.col_span(b).len();
    let tile = |r: usize, c: usize| sym.val_range(sym.col_range(c).find(|&b| sym.blk_row(b) == r).expect("a stored tile")).start;
    let mut rng = Lcg(seed ^ 0x9e37_79b9_7f4a_7c15);
    // per landmark, I + (eps - 1) u u^T for a random unit u: its columns'
    // scaling, column-major
    let mut scale: Vec<Vec<f64>> = vec![Vec::new(); nblk];
    for &b in &eliminated {
        let w = width(b);
        let mut u: Vec<f64> = (0..w).map(|_| rng.unit()).collect();
        let norm = u.iter().map(|v| v * v).sum::<f64>().sqrt();
        u.iter_mut().for_each(|v| *v /= norm);
        scale[b] = (0..w * w).map(|k| f64::from(k % w == k / w) + (eps - 1.0) * u[k % w] * u[k / w]).collect();
    }
    // J_r^T J_c into the tile (r, c), r <= c; each J column-major, m rows
    let add = |a: &mut SparseBlockColMat<SparseIndex, f64>, r: usize, jr: &[f64], c: usize, jc: &[f64], m: usize| {
        let (wr, wc, base) = (width(r), width(c), tile(r, c));
        for cc in 0..wc {
            for rr in 0..wr {
                if r == c && rr > cc {
                    continue;
                }
                let v: f64 = (0..m).map(|k| jr[k + rr * m] * jc[k + cc * m]).sum();
                a.vals_mut()[base + rr + cc * wr] += v;
            }
        }
    };
    a.vals_mut().fill(0.0);
    rhs.fill(0.0);
    for c in 0..nblk {
        for b in sym.col_range(c) {
            let r = sym.blk_row(b);
            if r == c {
                continue;
            }
            let m = if landmark[r] || landmark[c] { 2 } else { 6 };
            let mut jac = |blk: usize| -> Vec<f64> {
                let g: Vec<f64> = (0..m * width(blk)).map(|_| rng.unit()).collect();
                if !landmark[blk] {
                    return g;
                }
                let (w, s) = (width(blk), &scale[blk]);
                (0..m * w).map(|k| (0..w).map(|i| g[k % m + i * m] * s[i + (k / m) * w]).sum()).collect()
            };
            let (jr, jc) = (jac(r), jac(c));
            add(&mut a, r, &jr, r, &jr, m);
            add(&mut a, c, &jc, c, &jc, m);
            add(&mut a, r, &jr, c, &jc, m);
            let res: Vec<f64> = (0..m).map(|_| rng.unit()).collect();
            for (blk, j) in [(r, &jr), (c, &jc)] {
                let off = sym.col_span(blk).start;
                for k in 0..width(blk) {
                    rhs[off + k] += (0..m).map(|i| j[i + k * m] * res[i]).sum::<f64>();
                }
            }
        }
    }
    for b in 0..nblk {
        let (w, base) = (width(b), tile(b, b));
        for k in 0..w {
            a.vals_mut()[base + k * (w + 1)] *= 1.0 + lambda;
        }
    }
    (a, rhs, eliminated)
}
