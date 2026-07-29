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

/// `-1` in the scalar type, for the GEMM that subtracts the left updates.
#[inline]
fn minus_one<T: SchurReal>() -> T {
    let one = <T as faer::traits::ComplexField>::one_impl();
    T::ZERO - one
}

/// Marks a factor tile whose `S` source is a structural zero (an envelope
/// position `S` does not store; it is filled in during factorization).
const NO_SRC: ValueIndex = ValueIndex::MAX;

/// Widest a super-panel may get, in scalar columns.
///
/// The update GEMM gains efficiency with width and saturates around here; past
/// that only the fill keeps growing, since a panel `w` wide takes its envelope
/// top as the deepest of its columns and so factorizes `n * (b + w)` values
/// against a true envelope of `n * b`. Below ~30 the GEMM is too narrow to
/// pay; past ~90 the fill dominates and it is slower at every size measured.
///
/// Between those the curve is not smooth: it moves a few percent either way
/// with no relation to width, and which width is best changes with the
/// problem -- 72 beat 48 at one size and lost to it at two others. That is
/// the panel's stride landing well or badly in cache, not arithmetic, so
/// there is nothing to derive here. 48 is chosen for being at or ahead of the
/// general sparse route at every size measured rather than best at any; a
/// caller who cares can measure their own case through
/// [`BandSymbolic::with_panel_width`].
///
/// The panel takes whole block columns, so the width it reaches is this
/// rounded DOWN to a multiple of the block size: 48 is exactly 8 six-wide
/// poses, where 64 would reach only 60 and waste the rest of its budget.
const PANEL_WIDTH_MAX: usize = 48;

/// Narrowest worth grouping at all: below this the GEMM is no better than
/// the per-column panels it replaces.
const PANEL_WIDTH_MIN: usize = 8;

/// Grouping is abandoned when it would store this much more than the exact
/// envelope -- which is what happens on a genuinely narrow band, where the
/// envelope is the whole point and widening it is pure loss.
const PANEL_FILL_SLACK: f64 = 4.0;

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
    /// Super-panel grouping: block columns `sup_start[S]..sup_start[S + 1]`
    /// share one panel, so the factorization's GEMMs are ~[`PANEL_WIDTH`]
    /// wide on both sides instead of one block. `sup_of` maps a block column
    /// to its super-panel, `sup_top` the super-block its rows start at (the
    /// envelope top, snapped down to a super boundary so every operand is a
    /// whole row-block).
    sup_start: Vec<usize>,
    sup_top: Vec<usize>,
    sup_off: Vec<usize>,
    sup_rows: Vec<usize>,
    /// Where block column `j`'s envelope panel starts in the factor buffer.
    ///
    /// A column is ONE dense column-major panel of `col_rows[j]` scalar rows
    /// (covering `part[top[j]]..part[j + 1]`) by `wj` columns, not a run of
    /// separate tiles. Same bytes either way, but it makes a column's tiles
    /// a strided submatrix, so the whole `sum_k R_ki^T R_kj` update is one
    /// GEMM over the shared row range rather than one per `k`.
    col_off: Vec<usize>,
    /// scalar rows in column `j`'s panel
    col_rows: Vec<usize>,
    /// scalar row column `j`'s panel starts at -- its SUPER-panel's envelope
    /// top, which reaches at or above `top[j]`
    col_r0: Vec<usize>,
    /// whether the columns were grouped at all (false = one column per panel,
    /// the exact envelope; true = grouped, storing a little more)
    grouped: bool,
    /// per (column `j`, envelope row `i`) the matching `S` tile offset, or
    /// [`NO_SRC`]; column `j`'s run starts at `src_ptr[j]`
    s_src: Vec<ValueIndex>,
    src_ptr: Vec<usize>,
    /// scalars in the whole factor buffer
    n_vals: usize,
    /// largest `row_width * col_width` over all factor tiles (scratch size)
    max_tile: usize,
}

impl BandSymbolic {
    /// Analyzes `s` (the symbolic structure of a symmetric block-CSC matrix
    /// stored as its upper block triangle, in natural order) and builds the
    /// envelope factor pattern.
    pub fn new(s: &SymbolicSparseBlockColMat<crate::SparseIndex>) -> Self {
        Self::with_panel_width(s, None)
    }

    /// [`new`](Self::new) with the super-panel width chosen by the caller
    /// rather than from the envelope.
    ///
    /// `None` derives it: half the mean envelope height, bounded by
    /// [`PANEL_WIDTH_MIN`] and [`PANEL_WIDTH_MAX`]. That is the right shape --
    /// a wide panel pays for itself only while it stays well under the band it
    /// sits on -- and it is what a caller should normally leave alone. An
    /// explicit width is for measuring the curve, and is clamped to a width
    /// that can actually be grouped.
    pub fn with_panel_width(
        s: &SymbolicSparseBlockColMat<crate::SparseIndex>,
        panel_width: Option<usize>,
    ) -> Self {
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

        // Exact per-column envelope cost, the yardstick the grouping must not
        // stray far from.
        let exact: usize = (0..nb)
            .map(|j| (part[j + 1] - part[top[j]]) * (part[j + 1] - part[j]))
            .sum();

        // Greedy grouping by width, then measure what it actually costs. The
        // panels' tops snap down to a super boundary, so a group is only
        // affordable when its columns reach back to roughly the same row.
        let group = |width: usize| -> (Vec<usize>, Vec<usize>, Vec<usize>, usize) {
            let mut sup_start = vec![0usize];
            let mut j0 = 0usize;
            while j0 < nb {
                let mut j1 = j0 + 1;
                while j1 < nb && part[j1 + 1] - part[j0] <= width {
                    j1 += 1;
                }
                sup_start.push(j1);
                j0 = j1;
            }
            let ns = sup_start.len() - 1;
            let mut sup_of = vec![0usize; nb];
            for sidx in 0..ns {
                for j in sup_start[sidx]..sup_start[sidx + 1] {
                    sup_of[j] = sidx;
                }
            }
            // Panel top: the lowest envelope top any of the group's columns
            // reaches, snapped down to the super-block holding it.
            let mut sup_top = vec![0usize; ns];
            let mut cost = 0usize;
            for sidx in 0..ns {
                let (c0, c1) = (sup_start[sidx], sup_start[sidx + 1]);
                let t = (c0..c1).map(|j| top[j]).min().unwrap();
                sup_top[sidx] = sup_of[t];
                let rows = part[c1] - part[sup_start[sup_of[t]]];
                cost += rows * (part[c1] - part[c0]);
            }
            (sup_start, sup_of, sup_top, cost)
        };

        // Never group wider than the envelope is tall. A panel `w` wide over
        // an envelope `h` tall does `w * h` useful work and carries about
        // `w^2 / 2` of fill from snapping its top, so grouping pays only
        // while `w` stays well under `h` -- and a narrow band is exactly
        // where it does not. Measured across a landmark-span sweep: at a
        // half-bandwidth of 18 scalars a 32-wide panel costs 18% of the
        // solve, an 18-wide one wins.
        let mean_height = exact / n.max(1);
        let pw = match panel_width {
            Some(w) => w.max(1),
            None => (mean_height / 2).clamp(PANEL_WIDTH_MIN, PANEL_WIDTH_MAX),
        };
        let (sup_start, sup_of, sup_top, cost) = {
            let wide = group(pw);
            if (wide.3 as f64) <= PANEL_FILL_SLACK * exact as f64 {
                wide
            } else {
                // Narrow band: grouping would widen the envelope for nothing.
                group(1)
            }
        };
        let grouped = sup_start.len() - 1 != nb;
        let _ = cost;

        // Panel layout, one dense column-major block per super-panel.
        let ns = sup_start.len() - 1;
        let mut sup_off = Vec::with_capacity(ns);
        let mut sup_rows = Vec::with_capacity(ns);
        let mut acc = 0usize;
        let mut max_tile = 0usize;
        for sidx in 0..ns {
            let (c0, c1) = (sup_start[sidx], sup_start[sidx + 1]);
            let w = part[c1] - part[c0];
            let rows = part[c1] - part[sup_start[sup_top[sidx]]];
            sup_off.push(acc);
            sup_rows.push(rows);
            acc += rows * w;
            max_tile = max_tile.max(w * w);
        }

        // Per (column j, envelope row i), where S's matching tile lives.
        let mut src_ptr = Vec::with_capacity(nb);
        let mut s_src = Vec::new();
        for j in 0..nb {
            src_ptr.push(s_src.len());
            let mut sb = s.col_range(j).peekable();
            for i in top[j]..=j {
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
                s_src.push(src);
            }
        }

        // Per-column views into the super-panels, for the solve.
        let mut col_off = Vec::with_capacity(nb);
        let mut col_rows = Vec::with_capacity(nb);
        let mut col_r0 = Vec::with_capacity(nb);
        for j in 0..nb {
            let sidx = sup_of[j];
            let rows = sup_rows[sidx];
            col_off.push(sup_off[sidx] + (part[j] - part[sup_start[sidx]]) * rows);
            col_rows.push(rows);
            col_r0.push(part[sup_start[sup_top[sidx]]]);
        }

        Self {
            n, part, top, sup_start, sup_top, sup_off, sup_rows,
            col_off, col_rows, col_r0, grouped, s_src, src_ptr, n_vals: acc, max_tile,
        }
    }

    /// number of scalar values the factor buffer must hold
    #[inline]
    pub fn factor_val_count(&self) -> usize {
        self.n_vals
    }

    /// scalar dimension of the system
    #[inline]
    pub fn dim(&self) -> usize {
        self.n
    }

    /// Whether the columns were grouped into wide panels. False means one
    /// panel per block column and the exact envelope, which is what a narrow
    /// band gets -- grouping there would store a multiple of the envelope for
    /// no gain.
    #[inline]
    pub fn grouped(&self) -> bool {
        self.grouped
    }

    /// number of block columns
    #[inline]
    pub fn nblocks(&self) -> usize {
        self.part.len() - 1
    }

    /// factor value-buffer offset of tile `(row i, column j)`; both must lie
    /// in the envelope (`top[j] <= i <= j`). The tile is column-major with
    /// column stride [`col_stride`](Self::col_stride), NOT its own row count.
    #[inline]
    fn tile_off(&self, i: usize, j: usize) -> usize {
        self.col_off[j] + (self.part[i] - self.col_r0[j])
    }

    /// column stride of every tile in block column `j`
    #[inline]
    fn col_stride(&self, j: usize) -> usize {
        self.col_rows[j]
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
    let part = &sym.part;
    let top = &sym.top;
    let s_vals = s.vals();
    let ns = sym.sup_start.len() - 1;
    let mut scratch = vec![T::ZERO; sym.max_tile];

    // Scalar start of super-panel `S`, and of its first stored row.
    let cstart = |sidx: usize| part[sym.sup_start[sidx]];
    let cend = |sidx: usize| part[sym.sup_start[sidx + 1]];
    let rstart = |sidx: usize| part[sym.sup_start[sym.sup_top[sidx]]];

    for jsup in 0..ns {
        let wj = cend(jsup) - cstart(jsup);
        let hj = sym.sup_rows[jsup];
        let base_j = sym.sup_off[jsup];
        let r0j = rstart(jsup);

        // Seed the panel: zero, then drop in every S tile of its columns.
        factor[base_j..base_j + hj * wj].fill(T::ZERO);
        for j in sym.sup_start[jsup]..sym.sup_start[jsup + 1] {
            let wjb = part[j + 1] - part[j];
            let cj = part[j] - cstart(jsup);
            for i in top[j]..=j {
                let src = sym.s_src[sym.src_ptr[j] + (i - top[j])];
                if src == NO_SRC {
                    continue;
                }
                let (src, wi) = (src as usize, part[i + 1] - part[i]);
                let ri = part[i] - r0j;
                for c in 0..wjb {
                    let dst = base_j + (cj + c) * hj + ri;
                    factor[dst..dst + wi]
                        .clone_from_slice(&s_vals[src + c * wi..src + c * wi + wi]);
                }
            }
        }

        for isup in sym.sup_top[jsup]..=jsup {
            let wi = cend(isup) - cstart(isup);
            let hi = sym.sup_rows[isup];
            let base_i = sym.sup_off[isup];
            let r0i = rstart(isup);
            // Row block `isup` sits this far down panel jsup.
            let dst_off = base_j + (cstart(isup) - r0j);

            // R_IJ -= sum_K R_KI^T R_KJ over the row blocks both panels
            // share above I -- one GEMM, ~PANEL_WIDTH on each side.
            let kstart = sym.sup_top[isup].max(sym.sup_top[jsup]);
            let krows = cstart(isup) - cstart(kstart);
            if krows > 0 {
                let a_off = base_i + (cstart(kstart) - r0i);
                let b_off = base_j + (cstart(kstart) - r0j);
                let a = unsafe {
                    faer::MatRef::from_raw_parts(
                        factor.as_ptr().add(a_off), krows, wi, 1, hi as isize,
                    )
                };
                let b = unsafe {
                    faer::MatRef::from_raw_parts(
                        factor.as_ptr().add(b_off), krows, wj, 1, hj as isize,
                    )
                };
                let d = unsafe {
                    faer::MatMut::from_raw_parts_mut(
                        factor.as_mut_ptr().add(dst_off), wi, wj, 1, hj as isize,
                    )
                };
                faer::linalg::matmul::matmul(
                    d, faer::Accum::Add, a.transpose(), b, minus_one::<T>(), faer::Par::Seq,
                );
            }

            if isup < jsup {
                // R_IJ = R_II^{-T} R_IJ: R_II is upper, so its transpose is
                // the lower triangle faer solves against, in place.
                let rii = base_i + (cstart(isup) - r0i);
                let l = unsafe {
                    faer::MatRef::from_raw_parts(
                        factor.as_ptr().add(rii), wi, wi, 1, hi as isize,
                    )
                };
                let x = unsafe {
                    faer::MatMut::from_raw_parts_mut(
                        factor.as_mut_ptr().add(dst_off), wi, wj, 1, hj as isize,
                    )
                };
                faer::linalg::triangular_solve::solve_lower_triangular_in_place(
                    l.transpose(), x, faer::Par::Seq,
                );
            } else {
                // R_JJ: upper Cholesky of the diagonal block. O(w^3) once per
                // panel, so it runs on a contiguous copy.
                let buf = &mut scratch[..wj * wj];
                for c in 0..wj {
                    for r in 0..wj {
                        buf[r + c * wj] = factor[dst_off + r + c * hj];
                    }
                }
                if !chol_upper_in_place(buf, wj) {
                    return Err(BandError::NotPositiveDefinite);
                }
                for c in 0..wj {
                    for r in 0..wj {
                        factor[dst_off + r + c * hj] = buf[r + c * wj];
                    }
                }
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
        let hj = sym.col_stride(j);
        for i in top[j]..j {
            let wi = part[i + 1] - part[i];
            let off = sym.tile_off(i, j);
            let is = part[i];
            // y_j -= R_ij^T y_i : (R_ij^T)[c, r] = R_ij[r, c]
            for c in 0..wj {
                let mut acc = T::ZERO;
                for r in 0..wi {
                    acc = acc + factor[off + r + c * hj] * rhs[is + r];
                }
                rhs[js + c] = rhs[js + c] - acc;
            }
        }
        let djj = sym.tile_off(j, j);
        solve_upper_transpose_strided(&factor[djj..], &mut rhs[js..je], wj, hj);
    }

    // backward: R x = y (R is upper block triangular), column-oriented so
    // only column tiles are read: solve x_j, then push into rows above.
    for j in (0..nb).rev() {
        let (js, je) = (part[j], part[j + 1]);
        let wj = je - js;
        let hj = sym.col_stride(j);
        let djj = sym.tile_off(j, j);
        solve_upper_strided(&factor[djj..], &mut rhs[js..je], wj, hj);
        for i in top[j]..j {
            let wi = part[i + 1] - part[i];
            let off = sym.tile_off(i, j);
            let is = part[i];
            // rhs_i -= R_ij x_j
            for r in 0..wi {
                let mut acc = T::ZERO;
                for c in 0..wj {
                    acc = acc + factor[off + r + c * hj] * rhs[js + c];
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


/// [`solve_upper_transpose`] on a tile whose column stride is the whole
/// panel's height rather than its own width.
fn solve_upper_transpose_strided<T: SchurReal>(r: &[T], x: &mut [T], w: usize, stride: usize) {
    for row in 0..w {
        let mut s = x[row];
        for k in 0..row {
            s = s - r[k + row * stride] * x[k];
        }
        x[row] = s / r[row + row * stride];
    }
}

/// [`solve_upper`] on a tile with a panel column stride.
fn solve_upper_strided<T: SchurReal>(r: &[T], x: &mut [T], w: usize, stride: usize) {
    for row in (0..w).rev() {
        let mut s = x[row];
        for k in row + 1..w {
            s = s - r[row + k * stride] * x[k];
        }
        x[row] = s / r[row + row * stride];
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
        // the factor covers strictly more than S: the envelope fills in
        // (1,3) and (2,3), which S does not store
        assert!(sym.factor_val_count() > s.vals().len());
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

    /// Several super-panels, which is the case every other test misses: they
    /// are all narrower than PANEL_WIDTH, so they exercise one panel and
    /// nothing about the grouping. A wrong panel row-origin shows up here
    /// and nowhere else.
    #[test]
    fn multi_panel_dense() {
        // 40 blocks of 6 = 240 scalars, so ~4 panels at PANEL_WIDTH.
        let part: Vec<usize> = (0..=40).map(|i| i * 6).collect();
        let nb = part.len() - 1;
        // dense upper block triangle: every column reaches row 0
        let mut cells = Vec::new();
        for j in 0..nb {
            for i in 0..j {
                cells.push((i, j));
            }
        }
        let (s, full, rhs) = build_banded(&part, &cells, 11);
        let sym = BandSymbolic::new(s.symbolic());
        assert!(sym.sup_start.len() - 1 > 1, "test needs more than one panel");
        let mut factor = vec![0.0f64; sym.factor_val_count()];
        band_factorize(&sym, &s, &mut factor).unwrap();
        let mut x = rhs.clone();
        band_solve(&sym, &factor, &mut x);
        assert!(rel_resid(&full, *part.last().unwrap(), &x, &rhs) < 1e-10);
    }

    /// An explicit panel width overrides the derived one, and a wider panel
    /// stores strictly more: it snaps its envelope top to a panel boundary, so
    /// every column in it reaches as deep as the deepest. That growth is what
    /// bounds how wide a panel is worth making.
    #[test]
    fn a_wider_panel_stores_more() {
        let part: Vec<usize> = (0..=40).map(|i| i * 6).collect();
        let nb = part.len() - 1;
        let mut cells = Vec::new();
        for j in 0..nb {
            for i in j.saturating_sub(8)..j {
                cells.push((i, j));
            }
        }
        let (s, full, rhs) = build_banded(&part, &cells, 17);
        let sizes: Vec<usize> = [8usize, 16, 32, 64]
            .iter()
            .map(|&w| BandSymbolic::with_panel_width(s.symbolic(), Some(w)).factor_val_count())
            .collect();
        for pair in sizes.windows(2) {
            assert!(pair[1] > pair[0], "widening a panel must store more: {:?}", sizes);
        }
        // and every width still factorizes the same matrix exactly
        for &w in &[8usize, 64] {
            let sym = BandSymbolic::with_panel_width(s.symbolic(), Some(w));
            let mut factor = vec![0.0f64; sym.factor_val_count()];
            band_factorize(&sym, &s, &mut factor).unwrap();
            let mut x = rhs.clone();
            band_solve(&sym, &factor, &mut x);
            assert!(rel_resid(&full, *part.last().unwrap(), &x, &rhs) < 1e-10, "width {}", w);
        }
    }

    /// Multiple panels over a NARROW band: grouping is allowed (it is matched
    /// to the envelope height), but the factor must stay a band -- nowhere
    /// near the dense triangle -- and the solve must still be exact.
    #[test]
    fn multi_panel_narrow_band() {
        let part: Vec<usize> = (0..=40).map(|i| i * 6).collect();
        let nb = part.len() - 1;
        let mut cells = Vec::new();
        for j in 0..nb {
            for i in j.saturating_sub(2)..j {
                cells.push((i, j));
            }
        }
        let (s, full, rhs) = build_banded(&part, &cells, 13);
        let sym = BandSymbolic::new(s.symbolic());
        // The envelope must still be exploited: a 240-wide system whose band
        // is 2 blocks deep has a dense triangle of 28920 values, and the
        // factor must stay a small fraction of it.
        let n = *part.last().unwrap();
        let dense_triangle = n * (n + 1) / 2;
        assert!(
            sym.factor_val_count() * 4 < dense_triangle,
            "narrow band factor {} is not a band against a dense triangle of {}",
            sym.factor_val_count(),
            dense_triangle,
        );
        let mut factor = vec![0.0f64; sym.factor_val_count()];
        band_factorize(&sym, &s, &mut factor).unwrap();
        let mut x = rhs.clone();
        band_solve(&sym, &factor, &mut x);
        assert!(rel_resid(&full, *part.last().unwrap(), &x, &rhs) < 1e-10);
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
