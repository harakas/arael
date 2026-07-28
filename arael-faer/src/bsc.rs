//! variable-block sparse column-major matrix storage ("block CSC").
//!
//! the scalar matrix is partitioned into rectangular tiles by two
//! partition arrays (variable block widths). each STORED block is a
//! dense column-major tile, viewable as a [`MatRef`] at zero cost.
//!
//! layout, mirroring scalar CSC one level up:
//! - `row_part`/`col_part` hold SCALAR OFFSETS: block-row r spans
//!   scalar rows `row_part[r]..row_part[r+1]` (variable widths).
//! - `blk_col_ptr` holds INDICES INTO `blk_row_idx` AND `val_ptr`:
//!   those are all block-columns concatenated, and `blk_col_ptr[j]` is
//!   where block-column j's segment begins in both.
//! - `blk_row_idx[b]` and `vals[val_ptr[b]..val_ptr[b + 1]]` are a
//!   pair: "there is a dense tile at block-row `blk_row_idx[b]` of
//!   this block-column, with these column-major values".
//!
//! Symmetric matrices store the upper block triangle by convention
//! (like faer's `Side::Upper` elsewhere); the type itself is a
//! general rectangular matrix.
//!
//! Uses only faer's public API; written as an upstream candidate for
//! a future `faer::sparse::bsc` module.

use crate::{value_index, ValueIndex};
use core::iter;
use faer::sparse::{SparseColMat, SparseColMatRef, SymbolicSparseColMat};
use faer::traits::ComplexField;
use faer::traits::math_utils::zero;
use faer::{Index, Mat, MatMut, MatRef};

/// structure (pattern + partitions) of a variable-block sparse
/// column-major matrix. see the [module docs](self) for the layout.
#[derive(Clone, Debug)]
pub struct SymbolicSparseBlockColMat<I> {
    row_part: Vec<I>,
    col_part: Vec<I>,
    blk_col_ptr: Vec<I>,
    blk_row_idx: Vec<I>,
    val_ptr: Vec<I>,
}

/// variable-block sparse column-major matrix: structure + one
/// contiguous value buffer holding every tile's column-major payload.
#[derive(Clone, Debug)]
pub struct SparseBlockColMat<I, T> {
    symbolic: SymbolicSparseBlockColMat<I>,
    vals: Vec<T>,
}

impl<I: Index> SymbolicSparseBlockColMat<I> {
    /// creates the symbolic structure after checking every invariant:
    /// partitions monotone starting at 0; `blk_col_ptr` monotone
    /// covering `blk_row_idx`; block rows in range, strictly ascending
    /// within each block-column; `val_ptr` gaps equal to each stored
    /// tile's `row_width * col_width`, with nonzero dimensions.
    #[track_caller]
    pub fn new_checked(
        row_part: Vec<I>,
        col_part: Vec<I>,
        blk_col_ptr: Vec<I>,
        blk_row_idx: Vec<I>,
        val_ptr: Vec<I>,
    ) -> Self {
        let monotone_from_zero = |p: &[I]| {
            assert!(!p.is_empty());
            assert!(p[0].zx() == 0);
            for w in p.windows(2) {
                assert!(w[0].zx() <= w[1].zx());
            }
        };
        monotone_from_zero(&row_part);
        monotone_from_zero(&col_part);
        monotone_from_zero(&blk_col_ptr);
        monotone_from_zero(&val_ptr);

        let nblk_rows = row_part.len() - 1;
        let nblk_cols = col_part.len() - 1;
        let nblocks = blk_row_idx.len();
        assert!(blk_col_ptr.len() == nblk_cols + 1);
        assert!(blk_col_ptr[nblk_cols].zx() == nblocks);
        assert!(val_ptr.len() == nblocks + 1);

        for j in 0..nblk_cols {
            let col_width = col_part[j + 1].zx() - col_part[j].zx();
            let begin = blk_col_ptr[j].zx();
            let end = blk_col_ptr[j + 1].zx();
            for b in begin..end {
                let r = blk_row_idx[b].zx();
                assert!(r < nblk_rows);
                if b > begin {
                    // strictly ascending: sorted, no duplicate tiles
                    assert!(blk_row_idx[b - 1].zx() < r);
                }
                let row_width = row_part[r + 1].zx() - row_part[r].zx();
                assert!(row_width > 0);
                assert!(col_width > 0);
                let len = val_ptr[b + 1].zx() - val_ptr[b].zx();
                assert!(len == row_width * col_width);
            }
        }

        Self {
            row_part,
            col_part,
            blk_col_ptr,
            blk_row_idx,
            val_ptr,
        }
    }

    /// number of block-rows
    #[inline]
    pub fn nblk_rows(&self) -> usize {
        self.row_part.len() - 1
    }
    /// number of block-columns
    #[inline]
    pub fn nblk_cols(&self) -> usize {
        self.col_part.len() - 1
    }
    /// number of stored blocks
    #[inline]
    pub fn nblocks(&self) -> usize {
        self.blk_row_idx.len()
    }
    /// number of scalar rows
    #[inline]
    pub fn nrows(&self) -> usize {
        self.row_part[self.nblk_rows()].zx()
    }
    /// number of scalar columns
    #[inline]
    pub fn ncols(&self) -> usize {
        self.col_part[self.nblk_cols()].zx()
    }
    /// total number of stored scalar values
    #[inline]
    pub fn val_count(&self) -> usize {
        self.val_ptr[self.nblocks()].zx()
    }

    /// scalar row range of block-row `r`
    #[inline]
    pub fn row_span(&self, r: usize) -> core::ops::Range<usize> {
        self.row_part[r].zx()..self.row_part[r + 1].zx()
    }
    /// scalar column range of block-column `j`
    #[inline]
    pub fn col_span(&self, j: usize) -> core::ops::Range<usize> {
        self.col_part[j].zx()..self.col_part[j + 1].zx()
    }
    /// indices (into `blk_row_idx`/`val_ptr`/payloads) of the blocks
    /// stored in block-column `j`
    #[inline]
    pub fn col_range(&self, j: usize) -> core::ops::Range<usize> {
        self.blk_col_ptr[j].zx()..self.blk_col_ptr[j + 1].zx()
    }
    /// block-row of stored block `b`
    #[inline]
    pub fn blk_row(&self, b: usize) -> usize {
        self.blk_row_idx[b].zx()
    }
    /// (row_width, col_width) of stored block `b`. the column width is
    /// recovered from the payload length, so no column lookup is needed.
    #[inline]
    pub fn block_dims(&self, b: usize) -> (usize, usize) {
        let r = self.blk_row(b);
        let row_width = self.row_part[r + 1].zx() - self.row_part[r].zx();
        let len = self.val_ptr[b + 1].zx() - self.val_ptr[b].zx();
        (row_width, len / row_width)
    }
    /// payload range of stored block `b` in the value buffer
    #[inline]
    pub fn val_range(&self, b: usize) -> core::ops::Range<usize> {
        self.val_ptr[b].zx()..self.val_ptr[b + 1].zx()
    }

    /// Builds the block structure covering a stream of scalar
    /// coordinates, plus each coordinate's scatter POSITION into the
    /// value buffer -- the block-level analog of a COO -> CSC
    /// conversion with a position map. A tile is stored iff any
    /// coordinate falls inside it.
    ///
    /// `coord(k)` returns the k-th scalar `(row, col)`; it is called
    /// twice per coordinate (structure pass, then position pass), so it
    /// must be pure and cheap (e.g. an indexed read of COO arrays).
    /// Duplicate coordinates are fine and map to the same position
    /// (accumulation semantics are the caller's).
    ///
    /// Positions follow the tile layout: `val_ptr[b] + local_col *
    /// row_width + local_row` (column-major within the tile), matching
    /// [`SparseBlockColMat::block`].
    pub fn from_scalar_coords(
        row_part: Vec<I>,
        col_part: Vec<I>,
        n_coords: usize,
        coord: impl Fn(usize) -> (usize, usize),
    ) -> (Self, Vec<ValueIndex>) {
        // scalar index -> block index lookup tables, O(n) once
        let of = |part: &[I]| {
            let nblk = part.len() - 1;
            let mut of = vec![0u32; part[nblk].zx()];
            for b in 0..nblk {
                for i in part[b].zx()..part[b + 1].zx() {
                    of[i] = b as u32;
                }
            }
            of
        };
        let blk_row_of = of(&row_part);
        let blk_col_of = of(&col_part);

        // structure pass: the set of touched cells, as (block_col,
        // block_row) packed for one sort. After sort + dedup the vector
        // IS the block enumeration in storage order (columns ascending,
        // rows ascending within a column). Contributions arrive
        // block-object by block-object, so consecutive coordinates
        // usually share a cell: run-compressing before the sort shrinks
        // it from one key per scalar to roughly one per contributing
        // block object.
        let key = |i: usize, j: usize| {
            ((blk_col_of[j] as u64) << 32) | blk_row_of[i] as u64
        };
        let mut cells: Vec<u64> = Vec::with_capacity(1024);
        let mut last = u64::MAX;
        for k in 0..n_coords {
            let (i, j) = coord(k);
            let c = key(i, j);
            if c != last {
                cells.push(c);
                last = c;
            }
        }
        cells.sort_unstable();
        cells.dedup();

        let nblk_cols = col_part.len() - 1;
        let nblocks = cells.len();
        let mut blk_col_ptr = Vec::with_capacity(nblk_cols + 1);
        let mut blk_row_idx = Vec::with_capacity(nblocks);
        let mut val_ptr = Vec::with_capacity(nblocks + 1);
        blk_col_ptr.push(I::truncate(0));
        val_ptr.push(I::truncate(0));
        let mut val_total = 0usize;
        let mut col = 0usize;
        for &cell in &cells {
            let (bc, br) = ((cell >> 32) as usize, cell as u32 as usize);
            while col < bc {
                blk_col_ptr.push(I::truncate(blk_row_idx.len()));
                col += 1;
            }
            blk_row_idx.push(I::truncate(br));
            let row_w = row_part[br + 1].zx() - row_part[br].zx();
            let col_w = col_part[bc + 1].zx() - col_part[bc].zx();
            val_total += row_w * col_w;
            val_ptr.push(I::truncate(val_total));
        }
        while col < nblk_cols {
            blk_col_ptr.push(I::truncate(blk_row_idx.len()));
            col += 1;
        }

        let this = Self::new_checked(
            row_part, col_part, blk_col_ptr, blk_row_idx, val_ptr,
        );

        // position pass: block index recovered by binary search in the
        // sorted cell keys (block numbering == sorted order), memoized
        // per run of same-cell coordinates -- one search per block
        // object, arithmetic only for the scalars inside it
        let mut positions = Vec::with_capacity(n_coords);
        let mut last = u64::MAX;
        let (mut base, mut row_w, mut row_start, mut col_start) = (0usize, 0usize, 0usize, 0usize);
        for k in 0..n_coords {
            let (i, j) = coord(k);
            let c = key(i, j);
            if c != last {
                last = c;
                let b = cells.binary_search(&c).unwrap();
                let br = this.blk_row(b);
                let bc = blk_col_of[j] as usize;
                base = this.val_ptr[b].zx();
                row_w = this.row_span(br).len();
                row_start = this.row_span(br).start;
                col_start = this.col_span(bc).start;
            }
            positions.push(value_index(base + (j - col_start) * row_w + (i - row_start)));
        }

        (this, positions)
    }

    /// row partition: block-row `r` spans scalar rows
    /// `row_part()[r]..row_part()[r + 1]`
    #[inline]
    pub fn row_part(&self) -> &[I] {
        &self.row_part
    }

    /// column partition: block-column `j` spans scalar columns
    /// `col_part()[j]..col_part()[j + 1]`
    #[inline]
    pub fn col_part(&self) -> &[I] {
        &self.col_part
    }

    /// scalar nonzeros of the tile expansion's upper triangle: the stored
    /// slots minus the strictly-lower half of each stored diagonal block.
    /// Diagonal tiles are stored square, so [`Self::val_count`] (and the
    /// [`Self::csc_pattern`] length) overcount the triangle by exactly
    /// those halves.
    pub fn scalar_upper_nnz(&self) -> usize {
        let mut extra = 0usize;
        for j in 0..self.nblk_cols() {
            if self.col_range(j).any(|b| self.blk_row(b) == j) {
                let w = self.col_span(j).len();
                extra += w * (w - 1) / 2;
            }
        }
        self.val_count() - extra
    }

    /// scalar-CSC pattern of the tile expansion (the structure half of
    /// [`SparseBlockColMat::to_csc`]): `(col_ptr, row_idx)`, rows
    /// sorted within each column. blocks are sorted by block-row and
    /// their scalar rows are contiguous, so sortedness comes free.
    /// pair with [`SparseBlockColMat::csc_vals_into`] to refill values
    /// each iteration without rebuilding the pattern.
    pub fn csc_pattern(&self) -> (Vec<I>, Vec<I>) {
        let mut col_ptr = Vec::with_capacity(self.ncols() + 1);
        let mut row_idx = Vec::with_capacity(self.val_count());
        col_ptr.push(I::truncate(0));
        for j in 0..self.nblk_cols() {
            for _ in 0..self.col_span(j).len() {
                for b in self.col_range(j) {
                    for i in self.row_span(self.blk_row(b)) {
                        row_idx.push(I::truncate(i));
                    }
                }
                col_ptr.push(I::truncate(row_idx.len()));
            }
        }
        (col_ptr, row_idx)
    }

    /// partition and pattern arrays: `(row_part, col_part, blk_col_ptr,
    /// blk_row_idx, val_ptr)`
    #[inline]
    pub fn parts(&self) -> (&[I], &[I], &[I], &[I], &[I]) {
        (
            &self.row_part,
            &self.col_part,
            &self.blk_col_ptr,
            &self.blk_row_idx,
            &self.val_ptr,
        )
    }
}

impl<I: Index, T> SparseBlockColMat<I, T> {
    /// wraps a value buffer (length [`val_count`](SymbolicSparseBlockColMat::val_count))
    /// around a symbolic structure
    #[track_caller]
    pub fn new(symbolic: SymbolicSparseBlockColMat<I>, vals: Vec<T>) -> Self {
        assert!(vals.len() == symbolic.val_count());
        Self { symbolic, vals }
    }

    /// the symbolic structure
    #[inline]
    pub fn symbolic(&self) -> &SymbolicSparseBlockColMat<I> {
        &self.symbolic
    }
    /// the raw value buffer
    #[inline]
    pub fn vals(&self) -> &[T] {
        &self.vals
    }
    /// the raw value buffer, mutable (numeric refill between solves)
    #[inline]
    pub fn vals_mut(&mut self) -> &mut [T] {
        &mut self.vals
    }

    /// stored block `b` as a dense column-major view. O(1).
    #[inline]
    pub fn block(&self, b: usize) -> MatRef<'_, T> {
        let (nrows, ncols) = self.symbolic.block_dims(b);
        MatRef::from_column_major_slice(
            &self.vals[self.symbolic.val_range(b)],
            nrows,
            ncols,
        )
    }

    /// stored block `b` as a mutable dense column-major view. O(1).
    #[inline]
    pub fn block_mut(&mut self, b: usize) -> MatMut<'_, T> {
        let (nrows, ncols) = self.symbolic.block_dims(b);
        let range = self.symbolic.val_range(b);
        MatMut::from_column_major_slice_mut(
            &mut self.vals[range],
            nrows,
            ncols,
        )
    }

    /// iterates block-column `j`'s stored blocks as
    /// `(block_index, block_row, tile)`
    pub fn col_blocks(
        &self,
        j: usize,
    ) -> impl Iterator<Item = (usize, usize, MatRef<'_, T>)> {
        self.symbolic
            .col_range(j)
            .map(move |b| (b, self.symbolic.blk_row(b), self.block(b)))
    }

    /// tile at (block-row `r`, block-column `j`), if stored.
    /// O(log of the column's block count).
    pub fn get_block(&self, r: usize, j: usize) -> Option<MatRef<'_, T>> {
        let range = self.symbolic.col_range(j);
        let idx = &self.symbolic.blk_row_idx[range.clone()];
        idx.binary_search_by_key(&r, |i| i.zx())
            .ok()
            .map(|k| self.block(range.start + k))
    }
}

impl<I: Index, T: ComplexField> SparseBlockColMat<I, T> {
    /// a zero-filled matrix over the given structure (the target of an
    /// indexed scatter fill)
    pub fn zeroed(symbolic: SymbolicSparseBlockColMat<I>) -> Self {
        let n = symbolic.val_count();
        Self::new(symbolic, vec![zero::<T>(); n])
    }


    /// expands to a scalar CSC matrix with the same values (rows sorted
    /// within each column; only stored tiles contribute entries)
    pub fn to_csc(&self) -> SparseColMat<I, T> {
        let sym = &self.symbolic;
        let (col_ptr, row_idx) = sym.csc_pattern();
        let mut vals = vec![zero::<T>(); sym.val_count()];
        self.csc_vals_into(&mut vals);
        SparseColMat::new(
            SymbolicSparseColMat::new_checked(
                sym.nrows(), sym.ncols(), col_ptr, None, row_idx,
            ),
            vals,
        )
    }

    /// refills a scalar-CSC value buffer laid out exactly as
    /// [`to_csc`](Self::to_csc) produces (`out.len()` must equal
    /// [`val_count`](SymbolicSparseBlockColMat::val_count)). the
    /// per-iteration companion to a one-time `to_csc`: the pattern is
    /// fixed, so later iterations only need the values re-gathered.
    pub fn csc_vals_into(&self, out: &mut [T]) {
        let sym = &self.symbolic;
        assert_eq!(out.len(), sym.val_count());
        let mut k = 0;
        for j in 0..sym.nblk_cols() {
            let col_width = sym.col_span(j).len();
            for local_col in 0..col_width {
                for b in sym.col_range(j) {
                    let rows = sym.row_span(sym.blk_row(b)).len();
                    let payload = &self.vals[sym.val_range(b)];
                    let col_slice = &payload[local_col * rows..][..rows];
                    out[k..k + rows].clone_from_slice(col_slice);
                    k += rows;
                }
            }
        }
    }

    /// expands to a dense matrix (missing tiles stay zero). literal
    /// expansion of the storage: upper-only symmetric storage is NOT
    /// mirrored -- callers mirror themselves if they need the full
    /// matrix.
    pub fn to_dense(&self) -> Mat<T> {
        let sym = &self.symbolic;
        let mut out = Mat::<T>::zeros(sym.nrows(), sym.ncols());
        for j in 0..sym.nblk_cols() {
            let cols = sym.col_span(j);
            for b in sym.col_range(j) {
                let rows = sym.row_span(sym.blk_row(b));
                let blk = self.block(b);
                for (jj, cj) in cols.clone().enumerate() {
                    for (ii, ri) in rows.clone().enumerate() {
                        out[(ri, cj)] = blk[(ii, jj)].clone();
                    }
                }
            }
        }
        out
    }

    /// gathers a scalar CSC matrix into block form under the given
    /// partitions. a tile is stored iff any scalar entry falls inside
    /// it; unset scalars within a stored tile are zero.
    #[track_caller]
    pub fn from_csc(
        csc: SparseColMatRef<'_, I, T>,
        row_part: Vec<I>,
        col_part: Vec<I>,
    ) -> Self {
        let nrows = csc.nrows();
        let ncols = csc.ncols();
        assert!(row_part[row_part.len() - 1].zx() == nrows);
        assert!(col_part[col_part.len() - 1].zx() == ncols);
        let nblk_rows = row_part.len() - 1;
        let nblk_cols = col_part.len() - 1;

        // scalar index -> block index maps (both directions cheap;
        // build once, O(n))
        let row_of = |part: &[I], nblk: usize| {
            let mut of = vec![0usize; part[nblk].zx()];
            for r in 0..nblk {
                for i in part[r].zx()..part[r + 1].zx() {
                    of[i] = r;
                }
            }
            of
        };
        let blk_row_of = row_of(&row_part, nblk_rows);

        // pass 1: which tiles exist in each block-column
        let mut blk_col_ptr = Vec::with_capacity(nblk_cols + 1);
        let mut blk_row_idx = Vec::new();
        let mut val_ptr = Vec::new();
        let mut seen = vec![usize::MAX; nblk_rows];
        blk_col_ptr.push(I::truncate(0));
        val_ptr.push(I::truncate(0));
        let mut val_total = 0usize;
        for jb in 0..nblk_cols {
            let begin = blk_row_idx.len();
            for j in col_part[jb].zx()..col_part[jb + 1].zx() {
                for i in csc.row_idx_of_col(j) {
                    let rb = blk_row_of[i];
                    if seen[rb] != jb {
                        seen[rb] = jb;
                        blk_row_idx.push(I::truncate(rb));
                    }
                }
            }
            blk_row_idx[begin..].sort_unstable_by_key(|i: &I| i.zx());
            let col_width = col_part[jb + 1].zx() - col_part[jb].zx();
            for k in begin..blk_row_idx.len() {
                let rb = blk_row_idx[k].zx();
                let row_width = row_part[rb + 1].zx() - row_part[rb].zx();
                val_total += row_width * col_width;
                val_ptr.push(I::truncate(val_total));
            }
            blk_col_ptr.push(I::truncate(blk_row_idx.len()));
        }

        let symbolic = SymbolicSparseBlockColMat::new_checked(
            row_part,
            col_part,
            blk_col_ptr,
            blk_row_idx,
            val_ptr,
        );

        // pass 2: scatter values into zero-filled payloads
        let mut this = SparseBlockColMat::new(
            symbolic,
            vec![zero::<T>(); val_total],
        );
        for jb in 0..this.symbolic.nblk_cols() {
            let col_start = this.symbolic.col_span(jb).start;
            for j in this.symbolic.col_span(jb) {
                for (i, v) in
                    iter::zip(csc.row_idx_of_col(j), csc.val_of_col(j))
                {
                    let rb = blk_row_of[i];
                    let range = this.symbolic.col_range(jb);
                    let idx = &this.symbolic.blk_row_idx[range.clone()];
                    let k = idx
                        .binary_search_by_key(&rb, |x| x.zx())
                        .unwrap();
                    let b = range.start + k;
                    let rows = this.symbolic.row_span(rb);
                    let row_width = rows.len();
                    let local_row = i - rows.start;
                    let local_col = j - col_start;
                    let at = this.symbolic.val_range(b).start
                        + local_col * row_width
                        + local_row;
                    this.vals[at] = v.clone();
                }
            }
        }
        this
    }
}

/// One off-diagonal tile applied both ways: `y_rows += A x_cols` and
/// `y_cols += A^T x_rows`, in a single pass so the tile is read once.
///
/// The row width is a const parameter AND the row-spanning slices arrive as
/// arrays. Both matter: a constant trip count over a runtime-length slice
/// still bounds-checks every element, because nothing tells the compiler the
/// slice is `NR` long. Converting once per tile moves that check out of the
/// inner loop, which is the whole point of specializing.
#[inline(always)]
fn tile_both_ways<T, const NR: usize>(
    payload: &[T],
    xr: &[T; NR],
    yr: &mut [T; NR],
    xc: &[T],
    yc: &mut [T],
    ncols: usize,
) where
    T: ComplexField + Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
{
    for lc in 0..ncols {
        let xcj = xc[lc];
        let col: &[T; NR] = payload[lc * NR..lc * NR + NR].try_into().unwrap();
        let mut acc = zero::<T>();
        for lr in 0..NR {
            let a = col[lr];
            yr[lr] = yr[lr] + a * xcj;
            acc = acc + a * xr[lr];
        }
        yc[lc] = yc[lc] + acc;
    }
}

/// [`tile_both_ways`] for a width with no specialization.
#[inline]
fn tile_both_ways_dyn<T>(
    payload: &[T],
    xr: &[T],
    yr: &mut [T],
    xc: &[T],
    yc: &mut [T],
    ncols: usize,
    nr: usize,
) where
    T: ComplexField + Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
{
    for lc in 0..ncols {
        let xcj = xc[lc];
        let col = &payload[lc * nr..lc * nr + nr];
        let mut acc = zero::<T>();
        for lr in 0..nr {
            let a = col[lr];
            yr[lr] = yr[lr] + a * xcj;
            acc = acc + a * xr[lr];
        }
        yc[lc] = yc[lc] + acc;
    }
}

impl<I: Index, T> SparseBlockColMat<I, T>
where
    T: ComplexField + Copy + core::ops::Add<Output = T> + core::ops::Mul<Output = T>,
{
    /// `y = A * x` for a symmetric matrix held as the upper block triangle.
    ///
    /// Each stored off-diagonal tile is applied twice, once as itself and
    /// once transposed, since its mirror is not stored. A DIAGONAL tile is
    /// read as its upper scalar triangle only, and mirrored the same way:
    /// producers fill the upper triangle (the scalar CSC this expands to
    /// goes to faer as `Side::Upper`), so the strictly-lower half of a
    /// diagonal tile is the zero it was allocated with, not the transpose.
    /// Reading the whole tile there would silently drop half of every
    /// diagonal block's contribution.
    ///
    /// Requires the symmetric convention: square, with equal row and column
    /// partitions. `y` is overwritten, not accumulated into.
    pub fn mul_symmetric_upper(&self, x: &[T], y: &mut [T]) {
        let sym = &self.symbolic;
        assert_eq!(x.len(), sym.ncols(), "x length must match the column count");
        assert_eq!(y.len(), sym.nrows(), "y length must match the row count");
        debug_assert_eq!(sym.row_part(), sym.col_part(), "not a symmetric partition");
        for v in y.iter_mut() {
            *v = zero();
        }
        for j in 0..sym.nblk_cols() {
            let cols = sym.col_span(j);
            for b in sym.col_range(j) {
                let r = sym.blk_row(b);
                let rows = sym.row_span(r);
                let nr = rows.len();
                let payload = &self.vals[sym.val_range(b)];
                if r == j {
                    for (lc, cj) in cols.clone().enumerate() {
                        for lr in 0..=lc {
                            let a = payload[lc * nr + lr];
                            let ri = rows.start + lr;
                            y[ri] = y[ri] + a * x[cj];
                            if lr != lc {
                                y[cj] = y[cj] + a * x[ri];
                            }
                        }
                    }
                } else {
                    // Both directions in one pass over the tile. The column's
                    // x entry is loop-invariant and its y entry is a pure
                    // reduction, so both stay in registers and the inner loop
                    // is a straight walk of three contiguous slices -- which
                    // is what lets it vectorize and drops the bounds checks
                    // an indexed form pays twice per element.
                    //
                    // r < j for a stored tile, so the row span ends at or
                    // before the column span starts and the two output slices
                    // can be split apart.
                    let ncols = cols.len();
                    let (head, tail) = y.split_at_mut(cols.start);
                    let yr = &mut head[rows.clone()];
                    let yc = &mut tail[..ncols];
                    let xr = &x[rows.clone()];
                    let xc = &x[cols.clone()];
                    macro_rules! fixed {
                        ($w:expr) => {
                            tile_both_ways::<T, $w>(
                                payload,
                                xr.try_into().unwrap(),
                                (&mut yr[..]).try_into().unwrap(),
                                xc,
                                yc,
                                ncols,
                            )
                        };
                    }
                    match nr {
                        3 => fixed!(3),
                        6 => fixed!(6),
                        9 => fixed!(9),
                        _ => tile_both_ways_dyn(payload, xr, yr, xc, yc, ncols, nr),
                    }
                }
            }
        }
    }
}

/// Memoized scalar-coordinate -> value-buffer position resolver over a
/// built structure: one block lookup per RUN of same-cell coordinates
/// (contributions arrive block-object by block-object), arithmetic only
/// inside a run. The consumer for building indexed position maps
/// without a COO pass.
pub struct PositionResolver<'a, I: Index> {
    sym: &'a SymbolicSparseBlockColMat<I>,
    blk_row_of: Vec<u32>,
    blk_col_of: Vec<u32>,
    last_key: u64,
    base: usize,
    row_w: usize,
    row_start: usize,
    col_start: usize,
}

impl<'a, I: Index> PositionResolver<'a, I> {
    pub fn new(sym: &'a SymbolicSparseBlockColMat<I>) -> Self {
        let of = |part: &[I]| {
            let nblk = part.len() - 1;
            let mut of = vec![0u32; part[nblk].zx()];
            for b in 0..nblk {
                for i in part[b].zx()..part[b + 1].zx() {
                    of[i] = b as u32;
                }
            }
            of
        };
        let (row_part, col_part, _, _, _) = sym.parts();
        Self {
            blk_row_of: of(row_part),
            blk_col_of: of(col_part),
            sym,
            last_key: u64::MAX,
            base: 0,
            row_w: 0,
            row_start: 0,
            col_start: 0,
        }
    }

    /// Position of scalar (i, j) in the value buffer. Panics if the
    /// coordinate's cell is not part of the structure.
    #[inline]
    pub fn resolve(&mut self, i: usize, j: usize) -> usize {
        let bc = self.blk_col_of[j] as usize;
        let br = self.blk_row_of[i] as usize;
        let key = ((bc as u64) << 32) | br as u64;
        if key != self.last_key {
            self.last_key = key;
            let range = self.sym.col_range(bc);
            let idx = &self.sym.parts().3[range.clone()];
            let k = idx
                .binary_search_by_key(&br, |x| x.zx())
                .expect("coordinate outside the built block structure");
            let b = range.start + k;
            self.base = self.sym.val_range(b).start;
            self.row_w = self.sym.row_span(br).len();
            self.row_start = self.sym.row_span(br).start;
            self.col_start = self.sym.col_span(bc).start;
        }
        self.base + (j - self.col_start) * self.row_w + (i - self.row_start)
    }

    /// Position of scalar (i, j) plus the column stride of its tile, so a
    /// caller holding the tile origin can derive every other position in the
    /// tile as `origin + (c - col_start) * stride + (r - row_start)`.
    #[inline]
    pub fn resolve_tile(&mut self, i: usize, j: usize) -> (usize, usize) {
        let pos = self.resolve(i, j);
        (pos, self.row_w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // the SCHUR.md doodle: poses P1, P2 (width 2), landmarks L1, L2,
    // L3 (width 1); upper block triangle of
    //     [ P1  .   A  C  .  ]
    //     [ .   P2  B  .  D  ]
    //     [         L1       ]
    //     [            L2    ]
    //     [               L3 ]
    // block-rows/cols: [P1, P2, L1, L2, L3] -> partitions [0,2,4,5,6,7]
    fn doodle() -> SparseBlockColMat<usize, f64> {
        let part = vec![0usize, 2, 4, 5, 6, 7];
        // per block-column (sorted block rows):
        //   col0: {0: P1}          col1: {1: P2}
        //   col2: {0: A, 2: L1}    col3: {0: C, 3: L2}
        //   col4: {1: D, 4: L3}
        let blk_col_ptr = vec![0usize, 1, 2, 4, 6, 8];
        let blk_row_idx = vec![0usize, 1, 0, 2, 0, 3, 1, 4];
        // payload sizes: P1 2x2=4, P2 2x2=4, A 2x1=2, L1 1, C 2, L2 1,
        // D 2, L3 1
        let val_ptr = vec![0usize, 4, 8, 10, 11, 13, 14, 16, 17];
        let symbolic = SymbolicSparseBlockColMat::new_checked(
            part.clone(),
            part,
            blk_col_ptr,
            blk_row_idx,
            val_ptr,
        );
        // values chosen so every scalar cell is identifiable:
        // column-major within each tile
        let vals = vec![
            11., 21., 12., 22., // P1
            33., 43., 34., 44., // P2
            15., 25., // A
            55., // L1
            16., 26., // C
            66., // L2
            37., 47., // D
            77., // L3
        ];
        SparseBlockColMat::new(symbolic, vals)
    }

    #[test]
    fn block_access() {
        let m = doodle();
        assert_eq!(m.symbolic().nrows(), 7);
        assert_eq!(m.symbolic().ncols(), 7);
        assert_eq!(m.symbolic().nblocks(), 8);

        // A = block at (blk_row 0, blk_col 2), 2x1
        let a = m.get_block(0, 2).unwrap();
        assert_eq!(a.nrows(), 2);
        assert_eq!(a.ncols(), 1);
        assert_eq!(a[(0, 0)], 15.);
        assert_eq!(a[(1, 0)], 25.);

        // P2 diagonal tile, column-major
        let p2 = m.get_block(1, 1).unwrap();
        assert_eq!(p2[(0, 1)], 34.);
        assert_eq!(p2[(1, 0)], 43.);

        // absent tile
        assert!(m.get_block(1, 2).is_none());

        // iteration order within a block-column
        let got: Vec<usize> =
            m.col_blocks(3).map(|(_, r, _)| r).collect();
        assert_eq!(got, vec![0, 3]);
    }

    #[test]
    fn csc_vals_refill_matches_to_csc() {
        let mut m = doodle();
        let csc0 = m.to_csc();
        // perturb the values, refill, and compare against a fresh to_csc
        for (k, v) in m.vals_mut().iter_mut().enumerate() {
            *v += k as f64;
        }
        let mut vals = vec![0.0; csc0.val().len()];
        m.csc_vals_into(&mut vals);
        assert_eq!(vals, m.to_csc().val());
        assert_ne!(vals, csc0.val());
    }

    #[test]
    fn to_dense_matches_csc_expansion() {
        let m = doodle();
        let dense = m.to_dense();
        assert_eq!(dense.nrows(), m.symbolic().nrows());
        assert_eq!(dense.ncols(), m.symbolic().ncols());
        // reference: expand through the scalar CSC
        let csc = m.to_csc();
        let mut full = Mat::<f64>::zeros(csc.nrows(), csc.ncols());
        for j in 0..csc.ncols() {
            for (i, v) in iter::zip(csc.row_idx_of_col(j), csc.val_of_col(j)) {
                full[(i, j)] = *v;
            }
        }
        assert_eq!(dense, full);
        // a coordinate inside a stored tile and one outside any tile
        assert_eq!(dense[(0, 4)], 15.);
        assert_eq!(dense[(4, 0)], 0.);
    }

    #[test]
    fn scalar_coords_builder() {
        // rebuild the doodle's structure from its scalar coordinates
        // and scatter the same values through the position map
        let m = doodle();
        let csc = m.to_csc();
        let mut coords = Vec::new();
        for j in 0..csc.ncols() {
            for i in csc.row_idx_of_col(j) {
                coords.push((i, j));
            }
        }
        let part = vec![0usize, 2, 4, 5, 6, 7];
        let (sym, positions) = SymbolicSparseBlockColMat::from_scalar_coords(
            part.clone(),
            part,
            coords.len(),
            |k| coords[k],
        );
        assert_eq!(sym.nblocks(), m.symbolic().nblocks());
        assert_eq!(sym.parts().2, m.symbolic().parts().2);
        assert_eq!(sym.parts().3, m.symbolic().parts().3);
        assert_eq!(sym.parts().4, m.symbolic().parts().4);

        let mut rebuilt = SparseBlockColMat::<usize, f64>::zeroed(sym);
        let mut k = 0;
        for j in 0..csc.ncols() {
            for (_, v) in
                iter::zip(csc.row_idx_of_col(j), csc.val_of_col(j))
            {
                rebuilt.vals_mut()[positions[k] as usize] += *v;
                k += 1;
            }
        }
        assert_eq!(rebuilt.vals(), m.vals());
    }

    #[test]
    fn csc_roundtrip() {
        let m = doodle();
        let csc = m.to_csc();
        assert_eq!(csc.nrows(), 7);
        assert_eq!(csc.ncols(), 7);
        // scalar column 2 (P2's first column): rows 2,3 from P2
        assert_eq!(csc.row_idx_of_col(2).collect::<Vec<_>>(), vec![2, 3]);
        // scalar column 4 (the L1 block-column): A's rows then L1's diagonal
        assert_eq!(csc.row_idx_of_col(4).collect::<Vec<_>>(), vec![0, 1, 4]);

        // value spot checks against the dense picture
        let dense = |csc: &SparseColMat<usize, f64>, i: usize, j: usize| {
            let mut out = 0.0;
            for (r, v) in iter::zip(csc.row_idx_of_col(j), csc.val_of_col(j)) {
                if r == i {
                    out = *v;
                }
            }
            out
        };
        assert_eq!(dense(&csc, 0, 4), 15.); // A[0,0]
        assert_eq!(dense(&csc, 1, 4), 25.); // A[1,0]
        assert_eq!(dense(&csc, 4, 4), 55.); // L1
        assert_eq!(dense(&csc, 3, 6), 47.); // D[1,0] -> scalar (3, 6)
        assert_eq!(dense(&csc, 2, 4), 0.0); // structural zero

        // roundtrip: from_csc under the same partitions reproduces
        // the block matrix exactly
        let part = vec![0usize, 2, 4, 5, 6, 7];
        let back = SparseBlockColMat::from_csc(
            csc.as_ref(),
            part.clone(),
            part,
        );
        assert_eq!(back.symbolic().nblocks(), m.symbolic().nblocks());
        assert_eq!(back.vals(), m.vals());
    }

    // The doodle's structure under the SYMMETRIC convention: producers write
    // the upper triangle only, so the strictly-lower half of each diagonal
    // tile stays zero (P1's 21. and P2's 43. become 0.).
    fn doodle_symmetric() -> SparseBlockColMat<usize, f64> {
        let mut m = doodle();
        let p1 = m.symbolic().val_range(0);
        m.vals_mut()[p1.start + 1] = 0.;
        let p2 = m.symbolic().val_range(1);
        m.vals_mut()[p2.start + 1] = 0.;
        m
    }

    /// The full symmetric matrix the storage stands for: every stored tile
    /// mirrored, diagonal tiles mirrored from their upper triangle.
    fn mirrored_dense(m: &SparseBlockColMat<usize, f64>) -> Vec<Vec<f64>> {
        let sym = m.symbolic();
        let n = sym.nrows();
        let mut out = vec![vec![0.0; n]; n];
        for j in 0..sym.nblk_cols() {
            let cols = sym.col_span(j);
            for b in sym.col_range(j) {
                let rows = sym.row_span(sym.blk_row(b));
                let nr = rows.len();
                let payload = &m.vals()[sym.val_range(b)];
                for (lc, cj) in cols.clone().enumerate() {
                    for (lr, ri) in rows.clone().enumerate() {
                        if sym.blk_row(b) == j && lr > lc {
                            continue; // not written by the convention
                        }
                        let v = payload[lc * nr + lr];
                        out[ri][cj] = v;
                        out[cj][ri] = v;
                    }
                }
            }
        }
        out
    }

    #[test]
    fn symmetric_matvec_matches_dense() {
        let m = doodle_symmetric();
        let dense = mirrored_dense(&m);
        let n = m.symbolic().nrows();
        // a vector with no zeros or repeats, so a dropped term shows up
        let x: Vec<f64> = (0..n).map(|i| 1.0 + i as f64 * 0.5).collect();

        let mut want = vec![0.0; n];
        for i in 0..n {
            for k in 0..n {
                want[i] += dense[i][k] * x[k];
            }
        }
        let mut got = vec![0.0; n];
        m.mul_symmetric_upper(&x, &mut got);
        for i in 0..n {
            assert!((got[i] - want[i]).abs() < 1e-12,
                "row {}: got {}, want {}", i, got[i], want[i]);
        }
    }

    /// Widths 9, 3 and 5: the first two take a specialized inner loop, the
    /// last the dynamic fallback, and the off-diagonal tiles cover every
    /// dispatch arm. Guards the specializations against disagreeing with the
    /// dense reference or with each other.
    #[test]
    fn symmetric_matvec_over_specialized_widths() {
        let part = vec![0usize, 9, 12, 17];
        // col 0: {0}; col 1: {0,1}; col 2: {0,1,2}
        let blk_col_ptr = vec![0usize, 1, 3, 6];
        let blk_row_idx = vec![0usize, 0, 1, 0, 1, 2];
        let val_ptr = vec![0usize, 81, 108, 117, 162, 177, 202];
        let symbolic = SymbolicSparseBlockColMat::new_checked(
            part.clone(), part, blk_col_ptr, blk_row_idx, val_ptr,
        );
        // Distinct, non-zero, non-repeating values so a dropped or
        // double-counted term cannot cancel.
        let vals: Vec<f64> = (0..202).map(|k| 1.0 + (k % 37) as f64 * 0.25).collect();
        let mut m = SparseBlockColMat::new(symbolic, vals);
        // Apply the storage convention: diagonal tiles keep only their upper
        // triangle, the rest is the zero it was allocated with.
        for (b, w) in [(0usize, 9usize), (2, 3), (5, 5)] {
            let r = m.symbolic().val_range(b);
            for lc in 0..w {
                for lr in (lc + 1)..w {
                    m.vals_mut()[r.start + lc * w + lr] = 0.0;
                }
            }
        }

        let dense = mirrored_dense(&m);
        let n = m.symbolic().nrows();
        let x: Vec<f64> = (0..n).map(|i| 0.5 + i as f64 * 0.125).collect();
        let mut want = vec![0.0; n];
        for i in 0..n {
            for k in 0..n {
                want[i] += dense[i][k] * x[k];
            }
        }
        let mut got = vec![0.0; n];
        m.mul_symmetric_upper(&x, &mut got);
        for i in 0..n {
            assert!((got[i] - want[i]).abs() < 1e-9,
                "row {}: got {}, want {}", i, got[i], want[i]);
        }
    }

    // A diagonal tile's strictly-lower half is allocated zero and never
    // written, so reading the whole tile there loses the off-diagonal
    // coupling. Guards against a "simplification" back to a full-tile read.
    #[test]
    fn symmetric_matvec_mirrors_diagonal_tiles() {
        let m = doodle_symmetric();
        let n = m.symbolic().nrows();
        // e1 hits P1's (0,1) entry, whose mirror at (1,0) is only reachable
        // by mirroring the upper triangle.
        let mut x = vec![0.0; n];
        x[1] = 1.0;
        let mut got = vec![0.0; n];
        m.mul_symmetric_upper(&x, &mut got);
        assert_eq!(got[0], 12.); // P1[0,1]
        assert_eq!(got[1], 22.); // P1[1,1]

        let mut x = vec![0.0; n];
        x[0] = 1.0;
        let mut got = vec![0.0; n];
        m.mul_symmetric_upper(&x, &mut got);
        assert_eq!(got[0], 11.); // P1[0,0]
        assert_eq!(got[1], 12.); // the mirror, NOT the stored zero
    }
}
