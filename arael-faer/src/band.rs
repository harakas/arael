//! Block-band (envelope) Cholesky for a symmetric positive-definite
//! block-CSC matrix in its natural order.
//!
//! The reduced Schur system of a trajectory is banded: block `(i, j)` is
//! nonzero only when some eliminated block couples the two, and in natural
//! (trajectory) order that reach is a short span. A band factorization
//! exploits this directly -- no fill-reducing ordering, no symbolic
//! analysis, no scalar-CSC round trip. Fill is confined to each column's
//! envelope by construction (George-Liu: Cholesky preserves the envelope,
//! so column `j` fills only rows `top(j)..=j`, where `top(j)` is the
//! topmost stored block-row of `S`'s column `j`).
//!
//! Convention: upper factor `R` with `R^T R = S`, matching the block-CSC
//! upper-triangle storage of `S`. Left-looking by block-column: column `j`
//! is computed from the finished columns to its left, so every tile read
//! during a column's work is already final.
//!
//! Tile kernels are shared with [`crate::schur`]: the update
//! `R_ij -= R_ki^T R_kj` goes through [`SchurReal::gemm_sub_nano`]; the
//! diagonal Cholesky and the triangular tile solves are the small
//! hand-rolled routines below.

use crate::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};
use crate::{value_index, ValueIndex};
use crate::schur::SchurReal;

/// Marks a factor tile whose `S` source is a structural zero (an envelope
/// position `S` does not store; it is filled in during factorization).
const NO_SRC: ValueIndex = ValueIndex::MAX;

/// Failure of a band factorization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BandError {
    /// A diagonal tile was not positive definite (the reduced system is
    /// singular or indefinite at this damping).
    NotPositiveDefinite,
}

/// One-time structural analysis of a banded block-CSC matrix `S`: the
/// factor's envelope pattern and the map from each factor tile back to
/// `S`'s value buffer. Reused across damped solves; only the values change.
#[derive(Clone, Debug)]
pub struct BandSymbolic {
    /// scalar dimension of the (square, symmetric) system
    n: usize,
    /// scalar partition: block `j` spans `part[j]..part[j+1]`
    part: Vec<usize>,
    /// topmost stored block-row of `S`'s column `j` (the envelope top)
    top: Vec<usize>,
    /// factor pattern: column `j` holds the contiguous rows `top[j]..=j`
    factor: SymbolicSparseBlockColMat<crate::SparseIndex>,
    /// per factor tile, the start offset of the matching `S` tile in `S`'s
    /// value buffer, or [`NO_SRC`] if `S` has no tile there
    s_src: Vec<ValueIndex>,
    /// largest `row_width * col_width` over all factor tiles (scratch size)
    max_tile: usize,
}

impl BandSymbolic {
    /// Analyzes `s` (the symbolic structure of a symmetric block-CSC matrix
    /// stored as its upper block triangle, in natural order) and builds the
    /// envelope factor pattern.
    pub fn new(s: &SymbolicSparseBlockColMat<crate::SparseIndex>) -> Self {
        let nb = s.nblk_cols();
        assert_eq!(nb, s.nblk_rows(), "band Cholesky needs a square matrix");
        let n = s.ncols();

        let mut part = Vec::with_capacity(nb + 1);
        for j in 0..nb {
            part.push(s.col_span(j).start);
        }
        part.push(n);

        let mut top = vec![0usize; nb];
        for j in 0..nb {
            let col = s.col_range(j);
            // block-CSC columns are ascending by block-row, so the first
            // stored tile is the topmost. A column always has its diagonal.
            top[j] = s.blk_row(col.start);
            assert!(top[j] <= j, "S must be stored as its upper block triangle");
        }

        let mut blk_col_ptr = Vec::with_capacity(nb + 1);
        blk_col_ptr.push(0);
        let mut blk_row_idx = Vec::new();
        let mut val_ptr = Vec::new();
        val_ptr.push(0usize);
        let mut s_src = Vec::new();
        let mut max_tile = 0usize;

        for j in 0..nb {
            let wj = part[j + 1] - part[j];
            // walk S's stored rows of this column alongside the envelope
            // rows top[j]..=j; both are ascending, so one merge pass maps
            // each envelope tile to its S source (or NO_SRC).
            let mut sb = s.col_range(j).peekable();
            for i in top[j]..=j {
                let wi = part[i + 1] - part[i];
                let mut src = NO_SRC;
                while let Some(&b) = sb.peek() {
                    let br = s.blk_row(b);
                    if br < i {
                        sb.next();
                    } else {
                        if br == i {
                            src = value_index(s.val_range(b).start);
                            sb.next();
                        }
                        break;
                    }
                }
                blk_row_idx.push(i);
                let end = val_ptr.last().unwrap() + wi * wj;
                val_ptr.push(end);
                s_src.push(src);
                max_tile = max_tile.max(wi * wj);
            }
            blk_col_ptr.push(blk_row_idx.len());
        }

        let idx = |v: &[usize]| -> Vec<crate::SparseIndex> {
            v.iter().map(|&x| x as crate::SparseIndex).collect()
        };
        let factor = SymbolicSparseBlockColMat::new_checked(
            idx(&part),
            idx(&part),
            idx(&blk_col_ptr),
            idx(&blk_row_idx),
            idx(&val_ptr),
        );

        Self { n, part, top, factor, s_src, max_tile }
    }

    /// number of scalar values the factor buffer must hold
    #[inline]
    pub fn factor_val_count(&self) -> usize {
        self.factor.val_count()
    }

    /// scalar dimension of the system
    #[inline]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// number of block columns
    #[inline]
    pub fn nblocks(&self) -> usize {
        self.part.len() - 1
    }

    /// factor value-buffer offset of tile `(row i, column j)`; both must lie
    /// in the envelope (`top[j] <= i <= j`).
    #[inline]
    fn tile_off(&self, i: usize, j: usize) -> usize {
        let b = self.factor.col_range(j).start + (i - self.top[j]);
        self.factor.val_range(b).start
    }

    /// factor value-buffer offset of the tile at factor block index `b`.
    #[inline]
    fn tile_val(&self, b: usize) -> usize {
        self.factor.val_range(b).start
    }
}

/// Numeric factorization `R^T R = S` into `factor` (laid out per
/// [`BandSymbolic::factor_val_count`]). `s` must have the exact structure
/// `sym` was built from. Reuses `sym` across damped solves.
pub fn band_factorize<T: SchurReal>(
    sym: &BandSymbolic,
    s: &SparseBlockColMat<crate::SparseIndex, T>,
    factor: &mut [T],
) -> Result<(), BandError> {
    assert_eq!(factor.len(), sym.factor_val_count());
    let nb = sym.nblocks();
    let part = &sym.part;
    let top = &sym.top;
    let s_vals = s.vals();
    let mut scratch = vec![T::ZERO; sym.max_tile];

    for j in 0..nb {
        let wj = part[j + 1] - part[j];
        // Block index of column j's first tile; its tiles run top[j]..=j, one
        // per envelope row, so tile (i, j) is block `cj + (i - top[j])`.
        let cj = sym.factor.col_range(j).start;
        for i in top[j]..=j {
            let wi = part[i + 1] - part[i];
            let dst_len = wi * wj;
            let buf = &mut scratch[..dst_len];

            // seed with S(i, j) (a structural zero when S has no tile here)
            let src = sym.s_src[cj + (i - top[j])];
            if src == NO_SRC {
                buf.fill(T::ZERO);
            } else {
                let src = src as usize;
                buf.clone_from_slice(&s_vals[src..src + dst_len]);
            }

            // R_ij -= sum_k R_ki^T R_kj over rows k shared by both columns'
            // envelopes and above i (already-final factor tiles). gemm_sub
            // takes the unrolled kernel for pose-pose-pose shapes (6x6x6,
            // 3x3x3), else nano-gemm.
            let ci = sym.factor.col_range(i).start;
            let kstart = top[i].max(top[j]);
            for k in kstart..i {
                let wk = part[k + 1] - part[k];
                let rki = sym.tile_val(ci + (k - top[i]));
                let rkj = sym.tile_val(cj + (k - top[j]));
                crate::schur::gemm_sub(
                    buf,
                    &factor[rki..rki + wk * wi],
                    true,
                    wi,
                    wk,
                    &factor[rkj..rkj + wk * wj],
                    wj,
                );
            }

            let off = sym.tile_val(cj + (i - top[j]));
            if i < j {
                // R_ij = R_ii^{-T} buf
                let rii = sym.tile_val(ci + (i - top[i]));
                trsm_upper_transpose(&factor[rii..rii + wi * wi], buf, wi, wj);
                factor[off..off + dst_len].clone_from_slice(buf);
            } else {
                // R_jj: upper Cholesky of the accumulated diagonal tile
                if !chol_upper_in_place(&mut buf[..wj * wj], wj) {
                    return Err(BandError::NotPositiveDefinite);
                }
                factor[off..off + dst_len].clone_from_slice(buf);
            }
        }
    }
    Ok(())
}

/// Solves `S x = rhs` in place given a factor produced by
/// [`band_factorize`]. `rhs` has length [`BandSymbolic::dim`].
pub fn band_solve<T: SchurReal>(sym: &BandSymbolic, factor: &[T], rhs: &mut [T]) {
    assert_eq!(rhs.len(), sym.dim());
    let nb = sym.nblocks();
    let part = &sym.part;
    let top = &sym.top;

    // forward: R^T y = rhs (R^T is lower block triangular).
    // R_jj^T y_j = rhs_j - sum_{i in [top[j]..j)} R_ij^T y_i
    for j in 0..nb {
        let (js, je) = (part[j], part[j + 1]);
        let wj = je - js;
        for i in top[j]..j {
            let wi = part[i + 1] - part[i];
            let off = sym.tile_off(i, j);
            let rij = &factor[off..off + wi * wj];
            let is = part[i];
            // y_j -= R_ij^T y_i : (R_ij^T)[c, r] = R_ij[r, c]
            for c in 0..wj {
                let mut acc = T::ZERO;
                for r in 0..wi {
                    acc = acc + rij[r + c * wi] * rhs[is + r];
                }
                rhs[js + c] = rhs[js + c] - acc;
            }
        }
        let djj = sym.tile_off(j, j);
        solve_upper_transpose(&factor[djj..djj + wj * wj], &mut rhs[js..je], wj);
    }

    // backward: R x = y (R is upper block triangular), column-oriented so
    // only column tiles are read: solve x_j, then push into rows above.
    for j in (0..nb).rev() {
        let (js, je) = (part[j], part[j + 1]);
        let wj = je - js;
        let djj = sym.tile_off(j, j);
        solve_upper(&factor[djj..djj + wj * wj], &mut rhs[js..je], wj);
        for i in top[j]..j {
            let wi = part[i + 1] - part[i];
            let off = sym.tile_off(i, j);
            let rij = &factor[off..off + wi * wj];
            let is = part[i];
            // rhs_i -= R_ij x_j
            for r in 0..wi {
                let mut acc = T::ZERO;
                for c in 0..wj {
                    acc = acc + rij[r + c * wi] * rhs[js + c];
                }
                rhs[is + r] = rhs[is + r] - acc;
            }
        }
    }
}

/// In-place upper Cholesky of a `w x w` column-major tile: overwrites the
/// upper triangle with `R` such that `R^T R = A` (only the upper triangle
/// of `A` is read). Returns false if `A` is not positive definite.
fn chol_upper_in_place<T: SchurReal>(a: &mut [T], w: usize) -> bool {
    for j in 0..w {
        let mut d = a[j + j * w];
        for k in 0..j {
            let r = a[k + j * w];
            d = d - r * r;
        }
        if !(d > T::ZERO) {
            return false;
        }
        let djj = d.sqrt();
        a[j + j * w] = djj;
        for i in j + 1..w {
            let mut s = a[j + i * w];
            for k in 0..j {
                s = s - a[k + j * w] * a[k + i * w];
            }
            a[j + i * w] = s / djj;
        }
    }
    true
}

/// Solves `R^T X = B` in place on a `w x m` column-major panel, `R` the
/// upper factor tile (only its upper triangle is read). `R^T` is lower, so
/// this is forward substitution per column.
fn trsm_upper_transpose<T: SchurReal>(r: &[T], panel: &mut [T], w: usize, m: usize) {
    for c in 0..m {
        for row in 0..w {
            let mut s = panel[row + c * w];
            for k in 0..row {
                s = s - r[k + row * w] * panel[k + c * w];
            }
            panel[row + c * w] = s / r[row + row * w];
        }
    }
}

/// Solves `R^T x = b` in place for a single `w`-vector (forward
/// substitution; `R` upper factor tile, upper triangle read).
fn solve_upper_transpose<T: SchurReal>(r: &[T], x: &mut [T], w: usize) {
    for row in 0..w {
        let mut s = x[row];
        for k in 0..row {
            s = s - r[k + row * w] * x[k];
        }
        x[row] = s / r[row + row * w];
    }
}

/// Solves `R x = b` in place for a single `w`-vector (back substitution;
/// `R` upper factor tile, upper triangle read).
fn solve_upper<T: SchurReal>(r: &[T], x: &mut [T], w: usize) {
    for row in (0..w).rev() {
        let mut s = x[row];
        for k in row + 1..w {
            s = s - r[row + k * w] * x[k];
        }
        x[row] = s / r[row + row * w];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bsc::SymbolicSparseBlockColMat;

    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// Random SPD symmetric block matrix over `part` with the given upper
    /// block cells (diagonal blocks added automatically). Returns the
    /// upper-stored block-CSC `S`, the full (mirrored) dense matrix, and a
    /// random rhs. Strict diagonal dominance guarantees positive
    /// definiteness for any cell pattern.
    fn build_banded(
        part: &[usize],
        cells: &[(usize, usize)],
        seed: u64,
    ) -> (SparseBlockColMat<crate::SparseIndex, f64>, Vec<f64>, Vec<f64>) {
        let n = *part.last().unwrap();
        let nb = part.len() - 1;
        let mut all: Vec<(usize, usize)> = (0..nb).map(|b| (b, b)).collect();
        for &(bi, bj) in cells {
            assert!(bi <= bj, "cells must be upper block triangle");
            if bi != bj {
                all.push((bi, bj));
            }
        }

        let mut rng = Lcg(seed);
        let mut full = vec![0.0f64; n * n];
        for &(bi, bj) in &all {
            for c in part[bj]..part[bj + 1] {
                for r in part[bi]..part[bi + 1] {
                    if bi == bj && r > c {
                        continue;
                    }
                    let v = rng.next();
                    full[r + c * n] = v;
                    full[c + r * n] = v;
                }
            }
        }
        for i in 0..n {
            let mut s = 0.0;
            for k in 0..n {
                if k != i {
                    s += full[i + k * n].abs();
                }
            }
            full[i + i * n] = s + 1.0;
        }

        // coordinates of the stored upper tiles (diagonal blocks: upper only)
        let mut coords = Vec::new();
        for &(bi, bj) in &all {
            for c in part[bj]..part[bj + 1] {
                for r in part[bi]..part[bi + 1] {
                    if bi == bj && r > c {
                        continue;
                    }
                    coords.push((r, c));
                }
            }
        }
        let idx: Vec<crate::SparseIndex> =
            part.iter().map(|&p| p as crate::SparseIndex).collect();
        let (sym, pos) = SymbolicSparseBlockColMat::from_scalar_coords(
            idx.clone(),
            idx,
            coords.len(),
            |k| coords[k],
        );
        let mut s = SparseBlockColMat::<crate::SparseIndex, f64>::zeroed(sym);
        for (k, &(r, c)) in coords.iter().enumerate() {
            s.vals_mut()[pos[k] as usize] = full[r + c * n];
        }
        let rhs: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        (s, full, rhs)
    }

    /// relative residual ||full * x - rhs|| / ||rhs||
    fn rel_resid(full: &[f64], n: usize, x: &[f64], rhs: &[f64]) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            let mut ax = 0.0;
            for k in 0..n {
                ax += full[i + k * n] * x[k];
            }
            num += (ax - rhs[i]) * (ax - rhs[i]);
            den += rhs[i] * rhs[i];
        }
        (num / den).sqrt()
    }

    fn solve_f64(s: &SparseBlockColMat<crate::SparseIndex, f64>, rhs: &[f64]) -> Vec<f64> {
        let sym = BandSymbolic::new(s.symbolic());
        let mut factor = vec![0.0f64; sym.factor_val_count()];
        band_factorize(&sym, s, &mut factor).unwrap();
        let mut x = rhs.to_vec();
        band_solve(&sym, &factor, &mut x);
        x
    }

    #[test]
    fn block_tridiagonal_uniform() {
        // width-3 blocks, block-tridiagonal (block half-bandwidth 1)
        let part: Vec<usize> = (0..=8).map(|i| i * 3).collect();
        let cells: Vec<(usize, usize)> = (0..7).map(|i| (i, i + 1)).collect();
        let (s, full, rhs) = build_banded(&part, &cells, 1);
        let x = solve_f64(&s, &rhs);
        assert!(rel_resid(&full, part[8], &x, &rhs) < 1e-10);
    }

    #[test]
    fn wide_band_uniform() {
        // width-6 blocks (SE3 poses), block half-bandwidth 3
        let part: Vec<usize> = (0..=10).map(|i| i * 6).collect();
        let mut cells = Vec::new();
        for j in 0usize..10 {
            for i in j.saturating_sub(3)..j {
                cells.push((i, j));
            }
        }
        let (s, full, rhs) = build_banded(&part, &cells, 7);
        let x = solve_f64(&s, &rhs);
        assert!(rel_resid(&full, *part.last().unwrap(), &x, &rhs) < 1e-10);
    }

    #[test]
    fn mixed_widths() {
        let part = vec![0, 2, 5, 6, 8, 11, 13, 16];
        let cells = vec![(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (1, 3), (3, 5)];
        let (s, full, rhs) = build_banded(&part, &cells, 13);
        let x = solve_f64(&s, &rhs);
        assert!(rel_resid(&full, *part.last().unwrap(), &x, &rhs) < 1e-10);
    }

    #[test]
    fn gappy_band_fills_envelope() {
        // column 3 reaches back to row 0 but skips rows 1,2: the envelope
        // [0..=3] contains tiles S does not store, which factorization
        // fills. The factor must have more tiles than S.
        let part: Vec<usize> = (0..=5).map(|i| i * 2).collect();
        let cells = vec![(0, 1), (1, 2), (2, 3), (3, 4), (0, 3)];
        let (s, full, rhs) = build_banded(&part, &cells, 21);
        let sym = BandSymbolic::new(s.symbolic());
        let mut factor = vec![0.0f64; sym.factor_val_count()];
        band_factorize(&sym, &s, &mut factor).unwrap();
        // factor stores strictly more tiles than S (envelope fill of (1,3),(2,3))
        assert!(sym.factor.nblocks() > s.symbolic().nblocks());
        let mut x = rhs.clone();
        band_solve(&sym, &factor, &mut x);
        assert!(rel_resid(&full, *part.last().unwrap(), &x, &rhs) < 1e-10);
    }

    #[test]
    fn dense_matches() {
        // fully dense (every upper cell present): band route must still be exact
        let part = vec![0, 2, 4, 6, 9];
        let nb = 4;
        let mut cells = Vec::new();
        for i in 0..nb {
            for j in i + 1..nb {
                cells.push((i, j));
            }
        }
        let (s, full, rhs) = build_banded(&part, &cells, 99);
        let x = solve_f64(&s, &rhs);
        assert!(rel_resid(&full, *part.last().unwrap(), &x, &rhs) < 1e-10);
    }

    #[test]
    fn f32_wide_band() {
        let part: Vec<usize> = (0..=8).map(|i| i * 6).collect();
        let mut cells = Vec::new();
        for j in 0usize..8 {
            for i in j.saturating_sub(2)..j {
                cells.push((i, j));
            }
        }
        let (s64, full, rhs64) = build_banded(&part, &cells, 5);
        let s32 = SparseBlockColMat::<crate::SparseIndex, f32>::new(
            s64.symbolic().clone(),
            s64.vals().iter().map(|&v| v as f32).collect(),
        );
        let sym = BandSymbolic::new(s32.symbolic());
        let mut factor = vec![0.0f32; sym.factor_val_count()];
        band_factorize(&sym, &s32, &mut factor).unwrap();
        let mut x32 = rhs64.iter().map(|&v| v as f32).collect::<Vec<_>>();
        band_solve(&sym, &factor, &mut x32);
        let x: Vec<f64> = x32.iter().map(|&v| v as f64).collect();
        assert!(rel_resid(&full, *part.last().unwrap(), &x, &rhs64) < 1e-3);
    }

    #[test]
    fn rejects_non_pd() {
        let part: Vec<usize> = (0..=4).map(|i| i * 2).collect();
        let cells = vec![(0, 1), (1, 2), (2, 3)];
        let (mut s, _full, _rhs) = build_banded(&part, &cells, 3);
        // wreck a diagonal scalar: make block 2's first diagonal negative
        let sym = s.symbolic().clone();
        let diag_b = sym.col_range(2).find(|&b| sym.blk_row(b) == 2).unwrap();
        let off = sym.val_range(diag_b).start;
        s.vals_mut()[off] = -50.0;
        let bsym = BandSymbolic::new(s.symbolic());
        let mut factor = vec![0.0f64; bsym.factor_val_count()];
        assert_eq!(
            band_factorize(&bsym, &s, &mut factor),
            Err(BandError::NotPositiveDefinite)
        );
    }
}
