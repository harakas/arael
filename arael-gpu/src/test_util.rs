//! Synthetic matrices for the tests: a grid graph of blocks of mixed width,
//! random values, diagonal dominance, and a dense twin for residuals.

use arael_faer::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};

use crate::plan::SparseIndex;

pub struct Lcg(pub u64);

impl Lcg {
    pub fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    /// Uniform in [-1, 1).
    pub fn unit(&mut self) -> f64 {
        self.next() as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// A `k x k` grid of blocks, block `(r, c)` adjacent to its right and lower
/// neighbours, block widths cycling through `widths`; stored upper, the
/// diagonal tiles upper-only. Returns the matrix, its dense twin (column
/// major, full) and a right-hand side.
pub fn grid_spd(k: usize, widths: &[usize], seed: u64) -> (SparseBlockColMat<SparseIndex, f64>, Vec<f64>, Vec<f64>) {
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
    let mut dense = vec![0.0f64; n * n];
    let mut rng = Lcg(seed);
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
    // dominance: five neighbouring blocks of the widest width at most
    let boost = 4.0 * *widths.iter().max().unwrap() as f64 * 5.0;
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
    (a, dense, rhs)
}

/// `|A x - b| / |b|` through the dense twin.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
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
