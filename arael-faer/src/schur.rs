//! blocked Schur complement over [`bsc`](crate::bsc) matrices:
//! eliminate a set of mutually-uncoupled block variables (typically
//! landmarks) from a symmetric block system stored upper, producing
//! the reduced system `S = Hkk - Hke Hee^-1 Hek` and the reduced
//! right-hand side `bk' = bk - Hke Hee^-1 be`.
//!
//! symbolic/numeric split in faer's house style: [`schur_symbolic`]
//! analyzes the structure once -- S's block pattern (kept-kept tiles
//! plus one observer clique per eliminated block), a copy map for the
//! kept-kept part, per-eliminated coupling-tile lists, and the target
//! position of every observer-pair contribution. [`schur_reduce`] is
//! the per-iteration numeric pass: indexed arithmetic only, no
//! allocation once the [`SchurContext`] workspaces have grown to size.
//! Blocks arrive pre-damped; the module is lambda-free.
//!
//! The eliminated set must be internally uncoupled (no stored tile
//! joining two eliminated blocks) so that `Hee` is block-diagonal --
//! [`schur_symbolic`] rejects anything else.
//!
//! Storage convention: symmetric matrices store the upper block
//! triangle, and diagonal tiles carry only their scalar upper triangle
//! (the strictly-lower part is zero, as the indexed assembly leaves
//! it). Diagonal tiles are read as symmetric from the upper part, and
//! S comes back in the same convention.

use crate::bsc::{SparseBlockColMat, SymbolicSparseBlockColMat};
use crate::{value_index, ValueIndex};
use faer::Index;
use faer::traits::ComplexField;

mod sealed {
    pub trait Sealed {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

/// scalar operations the hand-rolled tile kernels need. the kernels
/// run on raw column-major slices (tile sizes are single digits, where
/// plain loops beat dispatching into a general GEMM). Implemented for
/// `f32` and `f64`; sealed.
pub trait SchurReal:
    sealed::Sealed
    + ComplexField
    + Copy
    + Send
    + Sync
    + PartialOrd
    + core::ops::Add<Output = Self>
    + core::ops::Sub<Output = Self>
    + core::ops::Mul<Output = Self>
    + core::ops::Div<Output = Self>
{
    const ZERO: Self;
    fn sqrt(self) -> Self;

    /// Widen to f64. Iterative solvers accumulate their reductions here
    /// whatever the storage type: a dot product over the whole system is
    /// where single precision loses its digits first, and widening it costs
    /// O(n) against an O(nnz) matrix-vector product.
    fn to_f64(self) -> f64;
    /// Narrow back from f64, for a scalar computed in the widened form.
    fn from_f64(v: f64) -> Self;

    /// `dst -= C_a * Z_b` for a tile shape with no unrolled kernel (see
    /// [`FIXED_SHAPES`]). nano-gemm takes the widths at run time, so it covers every
    /// shape, and on the shapes it is asked for it is faster than the plain
    /// loop it replaces (`--example gemmbench`). It does NOT beat the unrolled
    /// kernels, which is why those still take the shapes they cover: nano-gemm
    /// reaches its microkernel through a function pointer, and at these sizes that
    /// indirect call costs more than the arithmetic it dispatches.
    ///
    /// A transposed `C_a` is transposed here, into a stack buffer, rather than handed
    /// to nano-gemm as a strided lhs -- see the body for why that matters, and
    /// `TRANS_PACK_MAX` for the size at which it stops being worth it.
    ///
    /// The plan is built per call and NOT cached. It depends only on the shape, so
    /// caching it looks obvious -- but a plan is four lookups into a const
    /// microkernel table plus a branch chain, into a struct that stays on the stack,
    /// and a cache has to be searched before it can save that. A thread-local
    /// hash map costs more to search than the plan costs to build, and a
    /// thread-local linear scan over the few live shapes only comes out level
    /// with rebuilding. Level is not worth the state, so there is none.
    fn gemm_sub_nano(
        dst: &mut [Self],
        ca: &[Self],
        trans: bool,
        wa: usize,
        we: usize,
        zb: &[Self],
        wb: usize,
    );
}

/// How big a transposed `C_a` tile [`SchurReal::gemm_sub_nano`] will transpose into
/// a stack buffer rather than hand to nano-gemm as a strided lhs. 12x12 -- one
/// dimension past the widest entity anything here builds (a 9-dof BAL camera), and
/// 1152 bytes of f64, so the buffer is a couple of cache lines and not a stack
/// problem.
const TRANS_PACK_MAX: usize = 12 * 12;

/// The [`SchurReal`] impl for one scalar. nano-gemm's constructors are inherent
/// methods on the concrete scalar, not a trait, so the impl is generated per type.
macro_rules! impl_schur_real {
    ($t:ty, $colmajor:ident, $strided:ident) => {
        impl SchurReal for $t {
            const ZERO: Self = 0.0;
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn to_f64(self) -> f64 {
                self as f64
            }
            fn from_f64(v: f64) -> Self {
                v as $t
            }

            fn gemm_sub_nano(
                dst: &mut [Self],
                ca: &[Self],
                trans: bool,
                wa: usize,
                we: usize,
                zb: &[Self],
                wb: usize,
            ) {
                // nano-gemm's whole execution API is unsafe -- it takes raw
                // pointers and strides and trusts them. These three assertions are
                // what make the call below sound, so they are not debug_assert:
                // the reduction's tile slices come from a symbolic structure, and
                // a structure/width mismatch would otherwise be a buffer overrun
                // rather than a panic.
                assert_eq!(dst.len(), wa * wb, "dst is not the wa x wb tile");
                assert_eq!(ca.len(), wa * we, "C_a is not the wa x we tile");
                assert_eq!(zb.len(), we * wb, "Z_b is not the we x wb tile");

                // The lhs is C_a (wa x we). Stored directly it is column-major and
                // nano-gemm reads it in place. Stored transposed the tile holds
                // C_a^T (we x wa), so C_a[i, k] is at ca[k + i * we]: expressing
                // that as a row stride of `we` is correct, and it is a trap.
                // nano-gemm answers ANY lhs with a row stride other than 1 by
                // packing it -- `copy_millikernel` declares two 64 KB stack buffers
                // and copies into them. That cost is flat, so on a small tile it is
                // all there is: on x86 f64 it is ~60 ns whatever the widths, which
                // is more than the whole GEMM. So transpose it here instead, into a
                // buffer that fits in a cache line or two, and hand nano-gemm a
                // column-major lhs it can read in place.
                //
                // Above the cap the strided plan stands: packing is O(wa * we)
                // against O(wa * we * wb) of arithmetic, so a flat cost stops
                // mattering once the tile is big enough to amortize it.
                // MaybeUninit, not [0.0; TRANS_PACK_MAX]: zeroing the whole buffer
                // is 1152 bytes of memset on every call, which measured as a flat
                // ~64 ns -- worse than the packing it was meant to avoid. Uninit
                // costs nothing; it is stack space and no instructions.
                let mut packed = [const { core::mem::MaybeUninit::<$t>::uninit() }; TRANS_PACK_MAX];
                let pack = trans && wa * we <= TRANS_PACK_MAX;
                if pack {
                    for i in 0..wa {
                        for k in 0..we {
                            packed[i + k * wa].write(ca[k + i * we]);
                        }
                    }
                }

                let (lhs, lhs_rs, lhs_cs): (*const $t, isize, isize) = if !trans {
                    (ca.as_ptr(), 1, wa as isize)
                } else if pack {
                    (packed.as_ptr().cast::<$t>(), 1, wa as isize)
                } else {
                    (ca.as_ptr(), we as isize, 1)
                };

                // dst is always column-major, and so is the lhs unless it is an
                // oversized transposed tile -- only that case needs general strides.
                let plan = if lhs_rs == 1 {
                    nano_gemm::Plan::<$t>::$colmajor(wa, wb, we)
                } else {
                    nano_gemm::Plan::<$t>::$strided(wa, wb, we)
                };

                // SAFETY: nano-gemm requires (a) the plan's (m, n, k) to equal the
                // ones passed here, (b) any strides the plan pinned to equal the ones
                // passed here, and (c) every element it reads or writes to be in
                // bounds of the three buffers.
                //
                // (a) The plan is built two lines up from the same wa, wb, we, with
                //     m = wa, n = wb, k = we in both places. It cannot disagree.
                // (b) The `colmajor` constructor pins dst and lhs to column-major
                //     (rs = 1, cs = nrows), and it is chosen exactly when lhs_rs = 1,
                //     which is what is then passed: dst (1, wa), lhs (1, wa). The
                //     `strided` constructor pins nothing, and it takes the only case
                //     that is not column-major, the oversized transposed tile.
                // (c) The assertions above fix the lengths of dst, ca and zb. The
                //     largest offset touched is then:
                //       dst: 1*(wa-1) + wa*(wb-1)       = wa*wb - 1 < dst.len()
                //       zb : 1*(we-1) + we*(wb-1)       = we*wb - 1 < zb.len()
                //       lhs (rs=1, cs=wa):  1*(wa-1) + wa*(we-1) = wa*we - 1
                //       lhs (rs=we, cs=1): we*(wa-1) +  1*(we-1) = wa*we - 1
                //     `lhs` points at either ca, which the assertion fixes at wa * we
                //     long, or at `packed`, which is TRANS_PACK_MAX long and is only
                //     used when wa * we <= TRANS_PACK_MAX. Either way the reads land
                //     inside it.
                // (d) Every element read from `packed` is initialized. With rs = 1
                //     and cs = wa, nano-gemm reads exactly the offsets
                //     { i + k*wa : i < wa, k < we }, and the pack loop writes exactly
                //     that set. Nothing outside it is read, so the untouched tail of
                //     the buffer stays uninit and unobserved.
                //
                // All three pointers are derived from live locals, and dst is a &mut
                // so it does not alias lhs or zb -- `packed` is a fresh local, so it
                // cannot alias anything.
                //
                // nano-gemm computes dst = alpha*dst + beta*(lhs*rhs); ours is
                // dst -= C_a * Z_b, so alpha = 1 and beta = -1.
                unsafe {
                    plan.execute_unchecked(
                        wa,
                        wb,
                        we,
                        dst.as_mut_ptr(),
                        1,
                        wa as isize,
                        lhs,
                        lhs_rs,
                        lhs_cs,
                        zb.as_ptr(),
                        1,
                        we as isize,
                        1.0,
                        -1.0,
                        false,
                        false,
                    );
                }
            }
        }
    };
}

impl_schur_real!(f32, new_colmajor_lhs_and_dst_f32, new_f32);
impl_schur_real!(f64, new_colmajor_lhs_and_dst_f64, new_f64);

/// Why a Schur reduction cannot proceed: the eliminated set is not
/// eliminable, or a diagonal tile failed to factor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SchurError {
    /// a stored tile couples two eliminated blocks: `Hee` is not
    /// block-diagonal and the per-block elimination is invalid
    CoupledEliminated { row: usize, col: usize },
    /// an eliminated block has no stored diagonal tile
    MissingDiagonal { block: usize },
    /// `eliminated` was not strictly ascending or held an id out of range
    BadEliminatedSet,
    /// a diagonal tile was not positive definite during factorization
    /// (with LM damping applied upstream this indicates a modeling bug)
    NotPositiveDefinite { block: usize },
}

impl core::fmt::Display for SchurError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SchurError::CoupledEliminated { row, col } => {
                write!(f, "eliminated blocks {row} and {col} are coupled by a stored tile")
            }
            SchurError::MissingDiagonal { block } => {
                write!(f, "eliminated block {block} has no diagonal tile")
            }
            SchurError::BadEliminatedSet => {
                f.write_str("the eliminated set is not strictly ascending or names a block out of range")
            }
            SchurError::NotPositiveDefinite { block } => {
                write!(f, "the diagonal tile of eliminated block {block} is not positive definite")
            }
        }
    }
}

impl std::error::Error for SchurError {}

/// one-time structural analysis of a Schur reduction (see
/// [`schur_symbolic`]); consumed by [`schur_reduce`] every iteration.
#[derive(Clone, Debug)]
pub struct SchurSymbolic<I: Index> {
    /// kept id -> original block id, ascending
    pub kept: Vec<I>,
    /// original block id -> kept id (unspecified for eliminated blocks)
    kept_of: Vec<I>,
    /// structure of S over the kept partition
    pub s: SymbolicSparseBlockColMat<I>,
    /// kept-kept tiles: block index in H -> block index in S
    copy_src: Vec<I>,
    copy_dst: Vec<I>,
    /// per eliminated block, in `eliminated` order: H block index of
    /// its diagonal tile, its width, its range of the obs_* arrays and
    /// the panel width (observer columns, without the rhs column)
    elim_diag: Vec<I>,
    elim_w: Vec<I>,
    elim_obs_ptr: Vec<I>,
    elim_ncols: Vec<I>,
    /// per eliminated block, the observer width shared by all of its
    /// observers, or 0 when they differ -- 0 also when they differ in
    /// storage orientation, which [`gemm_row`] needs constant too.
    /// `elim_utrans` is that shared orientation.
    elim_uw: Vec<I>,
    elim_utrans: Vec<bool>,
    /// flattened observer lists. Everything the numeric passes read per
    /// observer is resolved here, so their inner loops index flat arrays
    /// instead of walking H's and S's block structure:
    ///
    /// * `obs_trans` -- the coupling tile is stored transposed, i.e. as
    ///   (elim, kept) rather than (kept, elim)
    /// * `obs_ca_off` -- where its coupling tile starts in `h.vals()`
    /// * `obs_w` -- its block width
    /// * `obs_panel_col` -- its column offset inside the solve panel
    /// * `obs_kept_off` -- where its span starts in the kept (S) scalar
    ///   numbering, for the reduction's rhs update
    /// * `obs_orig_off` -- where its span starts in the original scalar
    ///   numbering, for back-substitution
    obs_trans: Vec<bool>,
    obs_ca_off: Vec<I>,
    obs_w: Vec<I>,
    obs_panel_col: Vec<I>,
    obs_kept_off: Vec<I>,
    obs_orig_off: Vec<I>,
    /// * `obs_kept` -- its kept block id, which the numeric pass turns
    ///   into the target tile through a table of the column at hand
    obs_kept: Vec<I>,
    /// observer-pair products per reduction
    pairs: usize,
    /// per kept column, where the tile of each row from the column's
    /// topmost stored row (`col_top`) down to the diagonal starts in S's
    /// values, `ValueIndex::MAX` where the column has no such tile:
    /// `col_at[col_at_ptr[b] + (a - col_top[b])]` is the tile `(a, b)`
    col_at_ptr: Vec<usize>,
    col_top: Vec<I>,
    col_at: Vec<ValueIndex>,
    /// pair products before each kept column, `nk + 1` entries: what a
    /// column costs, for cutting the columns between threads
    col_pairs: Vec<usize>,
    /// where each kept column's entries of `copy_src` / `copy_dst`
    /// start, `nk + 1` entries
    copy_ptr: Vec<I>,
    /// workspace sizing: max panel elements / max eliminated width
    max_panel: usize,
    max_ew: usize,
    /// every GEMM tile shape this reduction needs, with the number of calls
    /// carried by each -- see [`SchurSymbolic::gemm_shapes`]
    shapes: Vec<((usize, usize, usize), usize)>,
    /// flops one [`schur_reduce`] costs: the observer-pair GEMMs, which
    /// dominate it. a caller weighing the reduction against factorizing the
    /// whole system needs this, and it is free to accumulate here.
    reduce_flops: f64,
}

impl<I: Index> SchurSymbolic<I> {
    /// allocate a zeroed S with this structure (reused across iterations)
    pub fn alloc_s<T: ComplexField>(&self) -> SparseBlockColMat<I, T> {
        SparseBlockColMat::zeroed(self.s.clone())
    }
    /// number of observer-pair contributions per reduction (the flop
    /// driver: quadratic in observers-per-eliminated-block)
    pub fn pair_count(&self) -> usize {
        self.pairs
    }
    /// The GEMM tile shapes this reduction needs, `((wa, we, wb), calls)`, in
    /// no particular order. A shape not in [`FIXED_SHAPES`] goes to the
    /// nano-gemm fallback, which costs more than the unrolled kernel, so a
    /// caller that cares about the last of it should look here -- and the call
    /// count says how much of the reduction is off the unrolled path.
    ///
    /// Covers every stage: the pair GEMMs, the reduction's `wb == 1` rhs
    /// update, and [`schur_backsub`]'s `wb == 1` update -- the last two run
    /// once per observer per eliminated block.
    pub fn gemm_shapes(&self) -> &[((usize, usize, usize), usize)] {
        &self.shapes
    }
    /// flops one [`schur_reduce`] costs (its observer-pair GEMMs). Free to
    /// read; the symbolic pass accumulates it.
    pub fn reduce_flops(&self) -> f64 {
        self.reduce_flops
    }
    /// scalar size of the reduced system
    pub fn kept_size(&self) -> usize {
        self.s.nrows()
    }

    /// Half-bandwidth of the reduced system: the largest scalar distance
    /// between a stored tile's first row and its last column. A banded
    /// system's factor cannot spill outside the band, so this bounds what
    /// factorizing it can cost -- far tighter than "it might be dense" when
    /// the eliminated blocks only couple nearby kept ones, which is the norm
    /// for a trajectory (a landmark is seen from a bounded stretch of it).
    /// O(tiles), no factorization.
    pub fn kept_bandwidth(&self) -> usize {
        let mut b = 0usize;
        for j in 0..self.s.nblk_cols() {
            let col_end = self.s.col_span(j).end;
            if let Some(first) = self.s.col_range(j).next() {
                let row_start = self.s.row_span(self.s.blk_row(first)).start;
                b = b.max(col_end - row_start);
            }
        }
        b
    }
    /// kept id of an original block id (must be kept)
    pub fn kept_of(&self, orig: usize) -> usize {
        self.kept_of[orig].zx()
    }
}

/// The chunk width [`SchurContext`] starts with, in kept block columns:
/// wide enough that a coupling tile is read few times, narrow enough
/// that a chunk's tiles of S stay in cache while its products land. On a
/// landmark trajectory and on bundle adjustment the curve is flat from
/// about half of this to twice it.
pub const DEFAULT_CHUNK_COLUMNS: usize = 128;

/// reusable numeric workspaces for [`schur_reduce`]; grows to the
/// symbolic sizes on first use, no allocation afterwards.
pub struct SchurContext<T> {
    /// lower-LLT factor of the current diagonal tile, column-major
    dwork: Vec<T>,
    /// solve panel `Z = D^-1 [C^T | b_e]`, column-major, width
    /// `sum(observer widths) + 1`, for the implicit routes
    panel: Vec<T>,
    /// every eliminated block's lower-LLT factor, concatenated in
    /// `elim_diag` order, written by [`schur_reduce`]. Kept so that
    /// [`schur_backsub`] -- which by construction runs on the same `h` and
    /// context, since it consumes the solution of the system the reduction
    /// produced -- does not factor the same tiles a second time.
    efactors: Vec<T>,
    /// where each eliminated block's factor starts in `efactors`
    efactor_at: Vec<usize>,
    /// every eliminated block's `z = D_e^-1 b_e`, concatenated in
    /// `elim_diag` order, and the blocks [`schur_backsub`] recovers,
    /// laid out the same way
    z: Vec<T>,
    xe: Vec<T>,
    /// where each eliminated block's `z` and `xe` start
    z_at: Vec<usize>,
    /// one scratch per thread
    workers: Vec<ColumnScratch<T>>,
    /// threads for [`schur_reduce`] and [`schur_backsub`]
    threads: usize,
    /// kept columns a thread's range is walked in at a time, 0 for the
    /// whole range
    chunk: usize,
    /// S block index -> H block index for the kept-kept tiles, built lazily
    /// by the implicit product ([`schur_apply`]), which reads H through S's
    /// geometry. `usize::MAX` where S has a tile the elimination created.
    s_to_h: Vec<usize>,
    /// per-stage breakdown of the last [`schur_reduce`] call, gathered
    /// only when enabled (the clock is never read otherwise)
    timing: Option<SchurTiming>,
}

impl<T> Default for SchurContext<T> {
    fn default() -> Self {
        Self {
            dwork: Vec::new(),
            panel: Vec::new(),
            efactors: Vec::new(),
            efactor_at: Vec::new(),
            z: Vec::new(),
            xe: Vec::new(),
            z_at: Vec::new(),
            workers: Vec::new(),
            threads: 1,
            chunk: DEFAULT_CHUNK_COLUMNS,
            s_to_h: Vec::new(),
            timing: None,
        }
    }
}

impl<T> SchurContext<T> {
    pub fn new() -> Self {
        Self::default()
    }
    /// Run [`schur_reduce`] and [`schur_backsub`] on `n` threads of
    /// [`crate::pool`]; 1 (the default) runs them on the calling
    /// thread. 0 counts as 1.
    pub fn set_threads(&mut self, n: usize) {
        self.threads = n.max(1);
    }
    /// The thread count [`set_threads`](Self::set_threads) gave.
    pub fn threads(&self) -> usize {
        self.threads
    }
    /// How many kept columns of S a thread of [`schur_reduce`] forms at
    /// a time: its range in chunks of `n` columns, the eliminated blocks
    /// walked once per chunk. 0 is the whole range; the default is
    /// [`DEFAULT_CHUNK_COLUMNS`]. A narrow chunk keeps the writes local, a
    /// wide one reads every coupling tile fewer times. The result is the
    /// same at any width.
    pub fn set_chunk_columns(&mut self, n: usize) {
        self.chunk = n;
    }
    /// The chunk width [`set_chunk_columns`](Self::set_chunk_columns) set.
    pub fn chunk_columns(&self) -> usize {
        self.chunk
    }
    /// gather a per-stage [`SchurTiming`] on every subsequent
    /// [`schur_reduce`] call (a few clock reads per call)
    pub fn enable_timing(&mut self) {
        self.timing = Some(SchurTiming::default());
    }
    /// stage breakdown of the last [`schur_reduce`] call, if gathering
    /// was enabled
    pub fn timing(&self) -> Option<&SchurTiming> {
        self.timing.as_ref()
    }
}

/// where a [`schur_reduce`] call spent its time (see that function's
/// stage-by-stage description; enable via
/// [`SchurContext::enable_timing`])
#[derive(Clone, Debug, Default)]
pub struct SchurTiming {
    /// stage 1: the dense Cholesky of every eliminated diagonal tile and
    /// its `z = D_e^-1 b_e`
    pub factor: std::time::Duration,
    /// stage 2: the kept columns, dispatch to join
    pub columns: std::time::Duration,
}

impl SchurTiming {
    /// `factor` plus `columns`.
    pub fn total(&self) -> std::time::Duration {
        self.factor + self.columns
    }
}

/// accumulates wall time between laps into stage counters; a no-op
/// (single branch, no clock read) when timing is disabled
struct Stopwatch {
    last: Option<std::time::Instant>,
}

impl Stopwatch {
    fn new(on: bool) -> Self {
        Self { last: on.then(std::time::Instant::now) }
    }
    #[inline]
    fn lap(&mut self, acc: &mut std::time::Duration) {
        if let Some(last) = &mut self.last {
            let now = std::time::Instant::now();
            *acc += now - *last;
            *last = now;
        }
    }
}

/// analyze the Schur reduction of `h` (symmetric, upper block triangle
/// stored, square partition) eliminating the given block ids (strictly
/// ascending). Errors if the eliminated set is internally coupled or an
/// eliminated block lacks its diagonal tile.
///
/// # Panics
///
/// When `h`'s block partition is not square.
pub fn schur_symbolic<I: Index>(
    h: &SymbolicSparseBlockColMat<I>,
    eliminated: &[usize],
) -> Result<SchurSymbolic<I>, SchurError> {
    let nblk = h.nblk_cols();
    assert_eq!(h.nblk_rows(), nblk, "square block partition required");

    // eliminated bitmap + per-eliminated slot id
    let mut elim_slot = vec![usize::MAX; nblk];
    let mut prev = None;
    for (slot, &e) in eliminated.iter().enumerate() {
        if e >= nblk || prev.is_some_and(|p| p >= e) {
            return Err(SchurError::BadEliminatedSet);
        }
        prev = Some(e);
        elim_slot[e] = slot;
    }
    let ne = eliminated.len();

    // kept renumbering and kept scalar partition
    let mut kept: Vec<I> = Vec::with_capacity(nblk - ne);
    let mut kept_of = vec![I::truncate(0); nblk];
    let mut kept_part: Vec<usize> = Vec::with_capacity(nblk - ne + 1);
    kept_part.push(0);
    for b in 0..nblk {
        if elim_slot[b] == usize::MAX {
            kept_of[b] = I::truncate(kept.len());
            kept.push(I::truncate(b));
            kept_part.push(kept_part.last().unwrap() + h.col_span(b).len());
        }
    }

    // single scan over all stored tiles: classify each as kept-kept
    // (copy), eliminated diagonal, or coupling tile of one eliminated
    // block. observers of one eliminated block arrive in ascending
    // block order (rows of its own column first, then transposed tiles
    // from later columns), so the per-block lists come out sorted.
    let mut diag: Vec<Option<I>> = vec![None; ne];
    let mut obs: Vec<Vec<(I, bool, I)>> = vec![Vec::new(); ne];
    let mut copy_kk: Vec<(I, usize, usize)> = Vec::new(); // (hblk, kr, kc)
    for c in 0..nblk {
        for b in h.col_range(c) {
            let r = h.blk_row(b);
            let (re, ce) = (elim_slot[r], elim_slot[c]);
            match (re != usize::MAX, ce != usize::MAX) {
                (true, true) => {
                    if r == c {
                        diag[ce] = Some(I::truncate(b));
                    } else {
                        return Err(SchurError::CoupledEliminated { row: r, col: c });
                    }
                }
                (false, true) => obs[ce].push((I::truncate(b), false, I::truncate(r))),
                (true, false) => obs[re].push((I::truncate(b), true, I::truncate(c))),
                (false, false) => {
                    copy_kk.push((I::truncate(b), kept_of[r].zx(), kept_of[c].zx()));
                }
            }
        }
    }

    // per-eliminated flat lists (observer order = ascending block id)
    let mut elim_diag = Vec::with_capacity(ne);
    let mut elim_w = Vec::with_capacity(ne);
    let mut elim_obs_ptr = Vec::with_capacity(ne + 1);
    let mut elim_ncols = Vec::with_capacity(ne);
    let mut elim_uw = Vec::with_capacity(ne);
    let mut elim_utrans = Vec::with_capacity(ne);
    // one entry per observation, and the count is already known: reserving
    // keeps eight arrays from growing by doubling through it
    let n_obs: usize = obs.iter().map(|l| l.len()).sum();
    let mut obs_trans = Vec::with_capacity(n_obs);
    let mut obs_block = Vec::with_capacity(n_obs);
    let mut obs_ca_off = Vec::with_capacity(n_obs);
    let mut obs_w = Vec::with_capacity(n_obs);
    let mut obs_panel_col = Vec::with_capacity(n_obs);
    let mut obs_kept_off = Vec::with_capacity(n_obs);
    let mut obs_orig_off = Vec::with_capacity(n_obs);
    let mut obs_kept = Vec::with_capacity(n_obs);
    let mut max_panel = 0usize;
    let mut max_ew = 0usize;
    let mut total_pairs = 0usize;
    let mut reduce_flops = 0.0f64;
    let mut shapes: Vec<((usize, usize, usize), usize)> = Vec::new();
    elim_obs_ptr.push(I::truncate(0));
    for slot in 0..ne {
        let e = eliminated[slot];
        let d = diag[slot].ok_or(SchurError::MissingDiagonal { block: e })?;
        elim_diag.push(d);
        let list = &obs[slot];
        let mut panel_cols = 0usize;
        for &(hb, tr, oblk) in list {
            let span = h.col_span(oblk.zx());
            obs_trans.push(tr);
            obs_block.push(oblk);
            obs_ca_off.push(I::truncate(h.val_range(hb.zx()).start));
            obs_w.push(I::truncate(span.len()));
            obs_panel_col.push(I::truncate(panel_cols));
            obs_orig_off.push(I::truncate(span.start));
            obs_kept_off.push(I::truncate(kept_part[kept_of[oblk.zx()].zx()]));
            obs_kept.push(kept_of[oblk.zx()]);
            panel_cols += span.len();
        }
        elim_ncols.push(I::truncate(panel_cols));
        panel_cols += 1; // + the rhs column
        let uniform = list
            .first()
            .map(|&(_, tr, ob)| (h.col_span(ob.zx()).len(), tr))
            .filter(|&(w, tr)| {
                list.iter().all(|&(_, t, o)| t == tr && h.col_span(o.zx()).len() == w)
            });
        elim_uw.push(I::truncate(uniform.map_or(0, |(w, _)| w)));
        elim_utrans.push(uniform.is_some_and(|(_, tr)| tr));
        total_pairs += list.len() * (list.len() + 1) / 2;
        let ew = h.col_span(e).len();
        // Which GEMM shapes this problem actually needs, and how many
        // contributions each carries. The widths are in hand here, so the
        // census is free; a caller uses it to see whether the reduction is
        // running on unrolled kernels or on the nano-gemm fallback. Every
        // stage counts: the pair GEMMs, the reduction's one-column rhs
        // update, and back-substitution's one-column update with the widths
        // the other way round.
        let mut count_shape = |wa: usize, we: usize, wb: usize| {
            match shapes.iter_mut().find(|(k, _)| *k == (wa, we, wb)) {
                Some((_, n)) => *n += 1,
                None => shapes.push(((wa, we, wb), 1)),
            }
        };
        for (bi, &(_, _, ob)) in list.iter().enumerate() {
            let wb = h.col_span(ob.zx()).len();
            for &(_, _, oa) in list.iter().take(bi + 1) {
                count_shape(h.col_span(oa.zx()).len(), ew, wb);
            }
        }
        for &(_, _, oa) in list.iter() {
            let wa = h.col_span(oa.zx()).len();
            count_shape(wa, ew, 1);
            count_shape(ew, wa, 1);
        }
        {
            // sum over pairs a <= b of 2 * w_a * ew * w_b, in closed form
            let mut sum_w = 0.0f64;
            let mut sum_w2 = 0.0f64;
            for &(_, _, oblk) in list {
                let w = h.col_span(oblk.zx()).len() as f64;
                sum_w += w;
                sum_w2 += w * w;
            }
            reduce_flops += ew as f64 * (sum_w * sum_w + sum_w2);
        }
        max_ew = max_ew.max(ew);
        max_panel = max_panel.max(ew * panel_cols);
        elim_w.push(I::truncate(ew));
        elim_obs_ptr.push(I::truncate(obs_trans.len()));
    }

    // S structure and every target position in one column-major pass,
    // no sorting: a stamp array unions each kept column's rows -- its
    // kept-kept tiles (already sorted by the scan) plus each clique's
    // {a <= b} for every observer b living in this column -- and the
    // read-back scan of 0..=kc emits them ascending. blk_at resolves
    // rows to S block indices for the copy map and pair targets, which
    // scatter straight into their b-major slots. everything is O(1)
    // per contribution; nothing is searched.
    let nk = kept.len();
    // inverted clique index: per kept column, the (slot, bi) of every
    // observer b in that column (CSR layout)
    let mut land_ptr = vec![0usize; nk + 1];
    for b in &obs_block {
        land_ptr[kept_of[b.zx()].zx() + 1] += 1;
    }
    for k in 0..nk {
        land_ptr[k + 1] += land_ptr[k];
    }
    let mut land_ent = vec![(0u32, 0u32); *land_ptr.last().unwrap()];
    {
        let mut cursor = land_ptr.clone();
        for slot in 0..ne {
            let start = elim_obs_ptr[slot].zx();
            for (bi, o) in (start..elim_obs_ptr[slot + 1].zx()).enumerate() {
                let kc = kept_of[obs_block[o].zx()].zx();
                land_ent[cursor[kc]] = (slot as u32, bi as u32);
                cursor[kc] += 1;
            }
        }
    }

    let mut mark = vec![0u32; nk];
    let mut blk_at = vec![0u32; nk];
    let mut blk_col_ptr: Vec<I> = Vec::with_capacity(nk + 1);
    let mut blk_row_idx: Vec<I> = Vec::new();
    let mut val_ptr: Vec<I> = Vec::new();
    let mut copy_dst = vec![I::truncate(0); copy_kk.len()];
    let mut copy_ptr = Vec::with_capacity(nk + 1);
    let mut col_pairs = Vec::with_capacity(nk + 1);
    col_pairs.push(0usize);
    blk_col_ptr.push(I::truncate(0));
    val_ptr.push(I::truncate(0));
    let mut vals_end = 0usize;
    let mut copy_cursor = 0usize;
    // Rows touched in the current column, gathered rather than searched
    // for. Reading the marks back with a `for kr in 0..=kc` scan would be
    // O(kept^2) overall -- invisible when the kept system is a handful of
    // cameras, ruinous when it is most of the model (six seconds at
    // 158k kept blocks). Gathering and sorting is O(nnz log w) instead.
    let mut touched: Vec<usize> = Vec::new();
    for kc in 0..nk {
        let stamp = kc as u32 + 1;
        let copy_begin = copy_cursor;
        copy_ptr.push(I::truncate(copy_begin));
        touched.clear();
        let touch = |kr: usize, mark: &mut Vec<u32>, touched: &mut Vec<usize>| {
            if mark[kr] != stamp {
                mark[kr] = stamp;
                touched.push(kr);
            }
        };
        while copy_cursor < copy_kk.len() && copy_kk[copy_cursor].2 == kc {
            touch(copy_kk[copy_cursor].1, &mut mark, &mut touched);
            copy_cursor += 1;
        }
        for &(slot, bi) in &land_ent[land_ptr[kc]..land_ptr[kc + 1]] {
            let start = elim_obs_ptr[slot as usize].zx();
            for o in start..=start + bi as usize {
                touch(kept_of[obs_block[o].zx()].zx(), &mut mark, &mut touched);
            }
        }
        touched.sort_unstable();
        let colw = kept_part[kc + 1] - kept_part[kc];
        for &kr in &touched {
            blk_at[kr] = blk_row_idx.len() as u32;
            blk_row_idx.push(I::truncate(kr));
            vals_end += (kept_part[kr + 1] - kept_part[kr]) * colw;
            val_ptr.push(I::truncate(vals_end));
        }
        blk_col_ptr.push(I::truncate(blk_row_idx.len()));
        for ci in copy_begin..copy_cursor {
            copy_dst[ci] = I::truncate(blk_at[copy_kk[ci].1] as usize);
        }
        let mut pairs = col_pairs[kc];
        for &(_, bi) in &land_ent[land_ptr[kc]..land_ptr[kc + 1]] {
            pairs += bi as usize + 1;
        }
        col_pairs.push(pairs);
    }
    debug_assert_eq!(*col_pairs.last().unwrap(), total_pairs);
    copy_ptr.push(I::truncate(copy_cursor));
    let kp: Vec<I> = kept_part.iter().map(|&x| I::truncate(x)).collect();
    let s = SymbolicSparseBlockColMat::new_checked(
        kp.clone(),
        kp,
        blk_col_ptr,
        blk_row_idx,
        val_ptr,
    );

    // The tile table of every kept column, top row to diagonal.
    let mut col_at_ptr = Vec::with_capacity(nk + 1);
    let mut col_top = Vec::with_capacity(nk);
    let mut col_at: Vec<ValueIndex> = Vec::new();
    for kc in 0..nk {
        let base = col_at.len();
        col_at_ptr.push(base);
        let col = s.col_range(kc);
        let top = if col.is_empty() { kc } else { s.blk_row(col.start) };
        col_top.push(I::truncate(top));
        col_at.resize(base + (kc - top + 1), ValueIndex::MAX);
        for b in col {
            col_at[base + s.blk_row(b) - top] = value_index(s.val_range(b).start);
        }
    }
    col_at_ptr.push(col_at.len());
    let copy_src: Vec<I> = copy_kk.iter().map(|&(hb, _, _)| hb).collect();

    Ok(SchurSymbolic {
        kept,
        kept_of,
        s,
        copy_src,
        copy_dst,
        elim_diag,
        elim_w,
        elim_obs_ptr,
        elim_ncols,
        elim_uw,
        elim_utrans,
        obs_trans,
        obs_ca_off,
        obs_w,
        obs_panel_col,
        obs_kept_off,
        obs_orig_off,
        obs_kept,
        pairs: total_pairs,
        col_at_ptr,
        col_top,
        col_at,
        col_pairs,
        copy_ptr,
        shapes,
        max_panel,
        max_ew,
        reduce_flops,
    })
}

/// in-place lower-Cholesky of a `w x w` column-major tile; false if a
/// pivot is not strictly positive
pub(crate) fn llt_in_place<T: SchurReal>(a: &mut [T], w: usize) -> bool {
    for k in 0..w {
        let mut d = a[k + k * w];
        for p in 0..k {
            let l = a[k + p * w];
            d = d - l * l;
        }
        if !(d > T::ZERO) {
            return false;
        }
        let dk = d.sqrt();
        a[k + k * w] = dk;
        for i in k + 1..w {
            let mut s = a[i + k * w];
            for p in 0..k {
                s = s - a[i + p * w] * a[k + p * w];
            }
            a[i + k * w] = s / dk;
        }
    }
    true
}

/// solve `(L L^T) Z = P` in place on a `w x m` column-major panel,
/// `L` lower from [`llt_in_place`]
pub(crate) fn llt_solve_panel<T: SchurReal>(l: &[T], panel: &mut [T], w: usize, m: usize) {
    for c in 0..m {
        let col = &mut panel[c * w..(c + 1) * w];
        for i in 0..w {
            let mut s = col[i];
            for p in 0..i {
                s = s - l[i + p * w] * col[p];
            }
            col[i] = s / l[i + i * w];
        }
        for i in (0..w).rev() {
            let mut s = col[i];
            for p in i + 1..w {
                s = s - l[p + i * w] * col[p];
            }
            col[i] = s / l[i + i * w];
        }
    }
}

/// [`gemm_sub`] with compile-time dimensions: constant trip counts let
/// the compiler fully unroll and vectorize.
#[inline]
fn gemm_sub_fixed<T: SchurReal, const WA: usize, const WE: usize, const WB: usize>(
    dst: &mut [T],
    ca: &[T],
    zb: &[T],
) {
    note_fixed_kernel();
    for c in 0..WB {
        for k in 0..WE {
            let z = zb[k + c * WE];
            let dcol = &mut dst[c * WA..(c + 1) * WA];
            let acol = &ca[k * WA..(k + 1) * WA];
            for i in 0..WA {
                dcol[i] = dcol[i] - acol[i] * z;
            }
        }
    }
}

/// Is it cheaper to transpose a `WA x WE` tile than to multiply through it in place?
///
/// [`gemm_sub_fixed`] is an axpy down a contiguous column, which vectorizes.
/// [`gemm_sub_fixed_trans`] is a horizontal dot product per output element, which does
/// not. So for a transposed tile there is a choice: multiply in place with the worse
/// loop shape, or pay a `WA x WE` transpose and then use the better one. The transpose
/// wins once there is enough of a column to vectorize (`WA` at least 5) and few
/// enough columns to copy (`WE` at most 4): at `WE` of 6 the copy stops paying, at
/// `WA` of 3 there is no column to vectorize.
///
/// `WB` is the amortization: the copy is `WA * WE` elements and is paid once for all
/// `WB` output columns. The rhs update is a single column, where the copy alone equals
/// the whole multiply, so it never transposes.
const fn transpose_first(wa: usize, we: usize, wb: usize) -> bool {
    wa >= 5 && we <= 4 && wb > 1
}

/// [`gemm_sub_fixed`] for a transposed-stored lhs: `ca` holds `C_a^T`
/// (`WE x WA` column-major), so each output element is a WE-term dot -- unless
/// [`transpose_first`] says the tile is worth transposing, in which case it is, and the
/// direct kernel's loop runs instead. The predicate is const, so only one of the two
/// bodies is ever generated for a given shape.
#[inline]
fn gemm_sub_fixed_trans<T: SchurReal, const WA: usize, const WE: usize, const WB: usize>(
    dst: &mut [T],
    ca: &[T],
    zb: &[T],
) {
    note_fixed_kernel();
    if transpose_first(WA, WE, WB) {
        // C_a^T is WE x WA, so C_a[i, k] is at ca[k + i * WE]. `a[k]` is then the
        // k-th column of C_a, contiguous, which is what the axpy below wants.
        //
        // `[[T; WA]; WE]` needs no generic_const_exprs -- it is nested arrays, not an
        // array of length WA * WE. The zero-init is free: every element is written
        // before it is read, so the stores are dead and get dropped (measured
        // identical to a MaybeUninit version).
        let mut a = [[T::ZERO; WA]; WE];
        for i in 0..WA {
            for k in 0..WE {
                a[k][i] = ca[k + i * WE];
            }
        }
        for c in 0..WB {
            for k in 0..WE {
                let z = zb[k + c * WE];
                let dcol = &mut dst[c * WA..(c + 1) * WA];
                let acol = &a[k];
                for i in 0..WA {
                    dcol[i] = dcol[i] - acol[i] * z;
                }
            }
        }
        return;
    }
    for c in 0..WB {
        for i in 0..WA {
            let arow = &ca[i * WE..(i + 1) * WE];
            let zcol = &zb[c * WE..(c + 1) * WE];
            let mut s = T::ZERO;
            for k in 0..WE {
                s = s + arow[k] * zcol[k];
            }
            dst[i + c * WA] = dst[i + c * WA] - s;
        }
    }
}

/// The tile shapes that have a fully unrolled GEMM kernel. Every other shape
/// works, through the nano-gemm fallback, which takes the widths at run time and
/// costs more than the unrolled kernel on these sizes (`--example
/// gemmbench`).
///
/// Two families, both `(wa, we, wb)`:
///
/// * `wb > 1` -- the pair GEMMs, `(observer, marginalized, observer)`.
/// * `wb == 1` -- the one-column updates, one per observer per eliminated
///   block: the reduction's rhs `b'_a -= C_a z` at `(wa, we, 1)`, and
///   [`schur_backsub`]'s `t -= C_a^T x_a` at the widths the other way round,
///   `(we, wa, 1)`. Both run as often as there are observations, so they need
///   kernels just as much: on the fallback each pays a nano-gemm plan for 27
///   flops.
///
/// This is the same list the `fixed_shapes!` macro dispatches on; a test walks every
/// shape up to 9x9x9 and checks the two agree, so they cannot drift apart.
/// [`SchurSymbolic::gemm_shapes`] reports what a given problem actually needs,
/// which is how a caller finds out it is on the fallback.
/// The widths are the ones SLAM systems actually use. Observers: 3 (a 2D
/// pose), 6 (a 3D pose), 7 (a similarity, for scale-aware loop closure), 9 (a
/// camera with intrinsics, which is also what a BAL camera and a NavState
/// are). Marginalized: 1 (inverse depth), 2 (a 2D point, or a bearing), 3 (a
/// 3D point), 4 (a 3D line, or a 2D segment). Cross-checked against g2o's
/// vertex dimensions and GTSAM's variable dimensions.
pub const FIXED_SHAPES: &[(usize, usize, usize)] = &[
    // -- 2D --
    (3, 2, 3), // 2D pose (x, y, theta) through a 2D point. The slam2d demos,
    // g2o VertexSE2 + VertexPointXY, Victoria-Park-style range-bearing SLAM
    (3, 3, 3), // 2D pose through an oriented landmark (a fiducial marker)
    (3, 4, 3), // 2D pose through a segment landmark (g2o VertexSegment2D)
    (2, 3, 2), // the mirror: a 3-wide family marginalized, seen from 2-wide
    // ones (map alignment -- a few path corrections, many landmarks)
    // -- 3D --
    (6, 1, 6), // 3D pose through an inverse-depth point (monocular SLAM/VIO)
    (6, 2, 6), // 3D pose through a bearing / direction landmark (GTSAM Unit3)
    (6, 3, 6), // 3D pose through a 3D point. g2o VertexSE3Expmap +
    // VertexPointXYZ, GTSAM Pose3 + Point3: the workhorse
    (6, 4, 6), // 3D pose through a line (g2o VertexLine3D, Plucker) or a plane
    (6, 6, 6), // 3D pose through a marginalized 6-dof entity (marker, object)
    // -- larger observers --
    (7, 3, 7), // similarity pose through a 3D point (g2o VertexSim3Expmap:
    // scale-aware loop closure)
    (9, 3, 9), // camera-with-intrinsics through a 3D point. BAL, GTSAM
    // SfmCamera = PinholeCamera<Cal3Bundler>, and a 9-dof NavState
    // -- the reduction's rhs update: the same (wa, we), one column wide --
    (3, 2, 1),
    (3, 3, 1),
    (3, 4, 1),
    (2, 3, 1),
    (6, 1, 1),
    (6, 2, 1),
    (6, 3, 1),
    (6, 4, 1),
    (6, 6, 1),
    (7, 3, 1),
    (9, 3, 1),
    // -- back-substitution: the widths the other way round. The ones the list
    // above already covers ((2,3,1), (3,3,1), (3,2,1), (6,6,1)) are not
    // repeated.
    (4, 3, 1),
    (1, 6, 1),
    (2, 6, 1),
    (3, 6, 1),
    (4, 6, 1),
    (3, 7, 1),
    (3, 9, 1),
    // -- bundle adjustment with the camera its own entity: a 6-dof pose and
    // a camera of 2, 4, 8 or 10 parameters (intrinsics, or intrinsics with
    // a rig pose) through a 3D point, in every pairing, and 5 for
    // a pose with one translation axis frozen by the gauge --
    (6, 3, 2),
    (2, 3, 6),
    (6, 3, 4),
    (4, 3, 6),
    (6, 3, 8),
    (8, 3, 6),
    (6, 3, 10),
    (10, 3, 6),
    (4, 3, 4),
    (8, 3, 8),
    (10, 3, 10),
    (5, 3, 6),
    (6, 3, 5),
    (5, 3, 5),
    (5, 3, 2),
    (2, 3, 5),
    (5, 3, 8),
    (8, 3, 5),
    // two cameras of different kinds through one point (a rig's reference
    // camera and one of its other cameras)
    (2, 3, 8),
    (8, 3, 2),
    (2, 3, 4),
    (4, 3, 2),
    (4, 3, 8),
    (8, 3, 4),
    (2, 3, 10),
    (10, 3, 2),
    (4, 3, 10),
    (10, 3, 4),
    (8, 3, 10),
    (10, 3, 8),
    // their one-column updates ((2, 3, 1) and (4, 3, 1) are above) and
    // back-substitutions ((3, 2, 1), (3, 4, 1) and (3, 6, 1) are above)
    (8, 3, 1),
    (10, 3, 1),
    (5, 3, 1),
    (3, 8, 1),
    (3, 10, 1),
    (3, 5, 1),
];

/// Does this tile shape have an unrolled kernel, or does it fall to nano-gemm?
/// See [`FIXED_SHAPES`].
pub fn has_fixed_kernel(wa: usize, we: usize, wb: usize) -> bool {
    FIXED_SHAPES.contains(&(wa, we, wb))
}

// Counts calls that reached an unrolled kernel. Whether the dispatch fires is
// invisible in the output -- the fallback computes the same thing -- so without
// this a broken match arm would silently take the slower path and no test would
// notice. Compiled out of every non-test build.
#[cfg(test)]
thread_local! {
    static FIXED_KERNEL_HITS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
#[inline]
fn note_fixed_kernel() {
    FIXED_KERNEL_HITS.with(|c| c.set(c.get() + 1));
}

#[cfg(not(test))]
#[inline(always)]
fn note_fixed_kernel() {}

// Counts observer runs that took the uniform path (`gemm_row`). Like
// `note_fixed_kernel`, the output cannot tell the two routes apart, so a
// missing dispatch arm would silently cost speed and no test would notice.
// Compiled out of every non-test build.
#[cfg(test)]
thread_local! {
    static UNIFORM_RUN_HITS: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
#[inline]
fn note_uniform_run() {
    UNIFORM_RUN_HITS.with(|c| c.set(c.get() + 1));
}

#[cfg(not(test))]
#[inline(always)]
fn note_uniform_run() {}

/// The tile shapes that get a fully unrolled kernel, as
/// `(observer, marginalized, observer)` widths. Every other shape falls to the
/// nano-gemm fallback ([`SchurReal::gemm_sub_nano`]), which is correct for any
/// shape but reaches its microkernel through a function pointer -- at these
/// sizes that indirect call costs more than the arithmetic it dispatches.
///
/// The list is what arael's own models produce:
///
/// * `(3, 2, 3)` -- 2D SLAM: a pose is `(x, y, theta)`, a landmark `(x, y)`.
/// * `(2, 3, 2)` -- the same models the other way up, when the 3-wide family
///   is the one worth marginalizing (map alignment: a few path corrections
///   seen from many landmarks, rather than the reverse).
/// * `(3, 3, 3)` -- 2D SLAM with oriented landmarks (fiducial markers), which
///   are pose-shaped themselves.
/// * `(6, 3, 6)` -- 3D SLAM: a 6-dof pose through a 3D point.
/// * `(6, 4, 6)` -- 3D SLAM with a 4-parameter landmark.
/// * `(6, 6, 6)` -- 3D with a marginalized 6-dof entity (a marker, an object).
/// * `(9, 3, 9)` -- bundle adjustment: a 9-parameter camera through a point.
/// * `(6, 3, 2)`, `(6, 3, 8)`, `(8, 3, 8)`, ... -- bundle adjustment with the
///   camera its own entity: a 6-dof pose and a camera of 2, 4, 8 or 10
///   parameters through a 3D point, in every pairing, plus the
///   5-wide pose the gauge leaves.
///
/// The three widths are packed into one integer (a nibble each) and matched
/// as a single value, so the compiler emits ONE switch. Matching the tuple
/// `(wa, we, wb)` directly compiles to a chain of comparisons instead, and
/// every shape then pays for the shapes listed ahead of it. With the pack,
/// growing the list costs the existing shapes nothing, which is what makes
/// the list free to grow.
macro_rules! fixed_shapes {
    ($kernel:ident, $dst:expr, $ca:expr, $zb:expr, $wa:expr, $we:expr, $wb:expr) => {
        // widths are tile dimensions, far below 16
        match ($wa << 8) | ($we << 4) | $wb {
            0x323 => return $kernel::<T, 3, 2, 3>($dst, $ca, $zb),
            0x333 => return $kernel::<T, 3, 3, 3>($dst, $ca, $zb),
            0x343 => return $kernel::<T, 3, 4, 3>($dst, $ca, $zb),
            0x232 => return $kernel::<T, 2, 3, 2>($dst, $ca, $zb),
            0x616 => return $kernel::<T, 6, 1, 6>($dst, $ca, $zb),
            0x626 => return $kernel::<T, 6, 2, 6>($dst, $ca, $zb),
            0x636 => return $kernel::<T, 6, 3, 6>($dst, $ca, $zb),
            0x646 => return $kernel::<T, 6, 4, 6>($dst, $ca, $zb),
            0x666 => return $kernel::<T, 6, 6, 6>($dst, $ca, $zb),
            0x737 => return $kernel::<T, 7, 3, 7>($dst, $ca, $zb),
            0x939 => return $kernel::<T, 9, 3, 9>($dst, $ca, $zb),
            // the reduction's rhs update, one column wide
            0x321 => return $kernel::<T, 3, 2, 1>($dst, $ca, $zb),
            0x331 => return $kernel::<T, 3, 3, 1>($dst, $ca, $zb),
            0x341 => return $kernel::<T, 3, 4, 1>($dst, $ca, $zb),
            0x231 => return $kernel::<T, 2, 3, 1>($dst, $ca, $zb),
            0x611 => return $kernel::<T, 6, 1, 1>($dst, $ca, $zb),
            0x621 => return $kernel::<T, 6, 2, 1>($dst, $ca, $zb),
            0x631 => return $kernel::<T, 6, 3, 1>($dst, $ca, $zb),
            0x641 => return $kernel::<T, 6, 4, 1>($dst, $ca, $zb),
            0x661 => return $kernel::<T, 6, 6, 1>($dst, $ca, $zb),
            0x731 => return $kernel::<T, 7, 3, 1>($dst, $ca, $zb),
            0x931 => return $kernel::<T, 9, 3, 1>($dst, $ca, $zb),
            // back-substitution, the widths the other way round
            0x431 => return $kernel::<T, 4, 3, 1>($dst, $ca, $zb),
            0x161 => return $kernel::<T, 1, 6, 1>($dst, $ca, $zb),
            0x261 => return $kernel::<T, 2, 6, 1>($dst, $ca, $zb),
            0x361 => return $kernel::<T, 3, 6, 1>($dst, $ca, $zb),
            0x461 => return $kernel::<T, 4, 6, 1>($dst, $ca, $zb),
            0x371 => return $kernel::<T, 3, 7, 1>($dst, $ca, $zb),
            0x391 => return $kernel::<T, 3, 9, 1>($dst, $ca, $zb),
            // bundle adjustment with the camera its own entity: a 6-dof pose
            // and a camera of 2, 4, 8 or 10 parameters through a 3D point,
            // in every pairing, and 5 for a pose with one translation axis
            // frozen by the gauge
            0x632 => return $kernel::<T, 6, 3, 2>($dst, $ca, $zb),
            0x236 => return $kernel::<T, 2, 3, 6>($dst, $ca, $zb),
            0x634 => return $kernel::<T, 6, 3, 4>($dst, $ca, $zb),
            0x436 => return $kernel::<T, 4, 3, 6>($dst, $ca, $zb),
            0x638 => return $kernel::<T, 6, 3, 8>($dst, $ca, $zb),
            0x836 => return $kernel::<T, 8, 3, 6>($dst, $ca, $zb),
            0x63A => return $kernel::<T, 6, 3, 10>($dst, $ca, $zb),
            0xA36 => return $kernel::<T, 10, 3, 6>($dst, $ca, $zb),
            0x434 => return $kernel::<T, 4, 3, 4>($dst, $ca, $zb),
            0x838 => return $kernel::<T, 8, 3, 8>($dst, $ca, $zb),
            0xA3A => return $kernel::<T, 10, 3, 10>($dst, $ca, $zb),
            0x536 => return $kernel::<T, 5, 3, 6>($dst, $ca, $zb),
            0x635 => return $kernel::<T, 6, 3, 5>($dst, $ca, $zb),
            0x535 => return $kernel::<T, 5, 3, 5>($dst, $ca, $zb),
            0x532 => return $kernel::<T, 5, 3, 2>($dst, $ca, $zb),
            0x235 => return $kernel::<T, 2, 3, 5>($dst, $ca, $zb),
            0x538 => return $kernel::<T, 5, 3, 8>($dst, $ca, $zb),
            0x835 => return $kernel::<T, 8, 3, 5>($dst, $ca, $zb),
            // two cameras of different kinds through one point (a rig's
            // reference camera and one of its other cameras)
            0x238 => return $kernel::<T, 2, 3, 8>($dst, $ca, $zb),
            0x832 => return $kernel::<T, 8, 3, 2>($dst, $ca, $zb),
            0x234 => return $kernel::<T, 2, 3, 4>($dst, $ca, $zb),
            0x432 => return $kernel::<T, 4, 3, 2>($dst, $ca, $zb),
            0x438 => return $kernel::<T, 4, 3, 8>($dst, $ca, $zb),
            0x834 => return $kernel::<T, 8, 3, 4>($dst, $ca, $zb),
            0x23A => return $kernel::<T, 2, 3, 10>($dst, $ca, $zb),
            0xA32 => return $kernel::<T, 10, 3, 2>($dst, $ca, $zb),
            0x43A => return $kernel::<T, 4, 3, 10>($dst, $ca, $zb),
            0xA34 => return $kernel::<T, 10, 3, 4>($dst, $ca, $zb),
            0x83A => return $kernel::<T, 8, 3, 10>($dst, $ca, $zb),
            0xA38 => return $kernel::<T, 10, 3, 8>($dst, $ca, $zb),
            // their one-column updates and back-substitutions
            0x831 => return $kernel::<T, 8, 3, 1>($dst, $ca, $zb),
            0xA31 => return $kernel::<T, 10, 3, 1>($dst, $ca, $zb),
            0x531 => return $kernel::<T, 5, 3, 1>($dst, $ca, $zb),
            0x381 => return $kernel::<T, 3, 8, 1>($dst, $ca, $zb),
            0x3A1 => return $kernel::<T, 3, 10, 1>($dst, $ca, $zb),
            0x351 => return $kernel::<T, 3, 5, 1>($dst, $ca, $zb),
            _ => {}
        }
    };
}

/// The pair shapes, as `(observer, marginalized)` -- the third width is the
/// observer width again. What [`gemm_row`] dispatches on, and the same list as
/// the `wb > 1` half of [`FIXED_SHAPES`]; a test checks every one of those
/// reaches the uniform path, so the two cannot drift apart.
macro_rules! uniform_pair_shapes {
    ($run:ident, $w:expr, $we:expr, $($arg:expr),*) => {
        // widths are tile dimensions, far below 16
        match ($w << 4) | $we {
            0x32 => return $run!(3, 2, $($arg),*),
            0x33 => return $run!(3, 3, $($arg),*),
            0x34 => return $run!(3, 4, $($arg),*),
            0x23 => return $run!(2, 3, $($arg),*),
            0x61 => return $run!(6, 1, $($arg),*),
            0x62 => return $run!(6, 2, $($arg),*),
            0x63 => return $run!(6, 3, $($arg),*),
            0x64 => return $run!(6, 4, $($arg),*),
            0x66 => return $run!(6, 6, $($arg),*),
            0x73 => return $run!(7, 3, $($arg),*),
            0x93 => return $run!(9, 3, $($arg),*),
            // bundle adjustment cameras seen alone (fixed images), and the
            // gauge-frozen pose
            0x43 => return $run!(4, 3, $($arg),*),
            0x83 => return $run!(8, 3, $($arg),*),
            0xA3 => return $run!(10, 3, $($arg),*),
            0x53 => return $run!(5, 3, $($arg),*),
            _ => {}
        }
    };
}

/// The run of pair products of one observer `b` of an eliminated block
/// whose observers all have the same width and storage orientation -- the
/// ordinary case, since the entity marginalized out is usually seen by one
/// kind of entity (poses through a landmark, cameras through a point). One
/// dispatch covers the run instead of one per pair, and the tile shape is a
/// compile-time constant throughout.
///
/// `zb` is `Z_b`, `kept` the run's observers `a <= b` as kept block ids,
/// ascending, `ca_off` their coupling tiles in `h_vals`, and `at` the
/// column's tile table from row `top` down. The targets are given from
/// `vbase`: `s_vals` is the slice of S that starts there.
#[inline]
fn gemm_row<T: SchurReal, I: Index, const W: usize, const WE: usize, const TRANS: bool>(
    s_vals: &mut [T],
    vbase: usize,
    h_vals: &[T],
    zb: &[T],
    at: &[ValueIndex],
    top: usize,
    kept: &[I],
    ca_off: &[I],
) {
    note_uniform_run();
    for (&k, &a) in core::iter::zip(kept, ca_off) {
        let d = at[k.zx() - top] as usize - vbase;
        let a = a.zx();
        if TRANS {
            gemm_sub_fixed_trans::<T, W, WE, W>(
                &mut s_vals[d..d + W * W],
                &h_vals[a..a + W * WE],
                zb,
            );
        } else {
            gemm_sub_fixed::<T, W, WE, W>(
                &mut s_vals[d..d + W * W],
                &h_vals[a..a + W * WE],
                zb,
            );
        }
    }
}

/// [`gemm_row`] for a width and orientation only known at run time. Returns
/// false when the shape has no unrolled kernel, leaving the caller to run the
/// pair-at-a-time loop.
#[allow(clippy::too_many_arguments)]
fn gemm_row_dispatch<T: SchurReal, I: Index>(
    w: usize,
    we: usize,
    trans: bool,
    s_vals: &mut [T],
    vbase: usize,
    h_vals: &[T],
    zb: &[T],
    at: &[ValueIndex],
    top: usize,
    kept: &[I],
    ca_off: &[I],
) -> bool {
    macro_rules! run {
        ($w:literal, $we:literal, $tr:expr) => {{
            if $tr {
                gemm_row::<T, I, $w, $we, true>(s_vals, vbase, h_vals, zb, at, top, kept, ca_off);
            } else {
                gemm_row::<T, I, $w, $we, false>(s_vals, vbase, h_vals, zb, at, top, kept, ca_off);
            }
            true
        }};
    }
    uniform_pair_shapes!(run, w, we, trans);
    false
}

/// `dst -= C_a * Z_b` where `dst` is `wa x wb` column-major, `Z_b` is
/// `we x wb` column-major, and `C_a` is `wa x we` -- stored directly
/// (`trans == false`) or as its transpose `we x wa` (`trans == true`).
/// Dispatches an unrolled kernel for [`FIXED_SHAPES`], else nano-gemm.
#[inline]
pub(crate) fn gemm_sub<T: SchurReal>(
    dst: &mut [T],
    ca: &[T],
    trans: bool,
    wa: usize,
    we: usize,
    zb: &[T],
    wb: usize,
) {
    if !trans {
        fixed_shapes!(gemm_sub_fixed, dst, ca, zb, wa, we, wb);
    } else {
        fixed_shapes!(gemm_sub_fixed_trans, dst, ca, zb, wa, we, wb);
    }
    // No unrolled kernel for this shape: nano-gemm, which takes the widths at
    // run time. Both macros above `return` on a hit, so reaching here IS the
    // fallback.
    T::gemm_sub_nano(dst, ca, trans, wa, we, zb, wb);
}

/// Factor every eliminated block's diagonal tile into the context, without
/// reducing anything. [`schur_reduce`] does this as its first stage; the
/// implicit product needs the same factors but never forms S, so it calls
/// this once per damped solve and then multiplies.
pub fn schur_factor_eliminated<I: Index, T: SchurReal>(
    sym: &SchurSymbolic<I>,
    h: &SparseBlockColMat<I, T>,
    ctx: &mut SchurContext<T>,
) -> Result<(), SchurError> {
    let hs = h.symbolic();
    ctx.dwork.resize(sym.max_ew * sym.max_ew, T::ZERO);
    size_efactors(sym, ctx);
    for slot in 0..sym.elim_diag.len() {
        let d_blk = sym.elim_diag[slot].zx();
        let e = hs.blk_row(d_blk);
        let we = hs.col_span(e).len();
        let dwork = &mut ctx.dwork[..we * we];
        read_symmetric_tile(&h.vals()[hs.val_range(d_blk)], dwork, we);
        if !llt_in_place(dwork, we) {
            return Err(SchurError::NotPositiveDefinite { block: e });
        }
        ctx.efactors[ctx.efactor_at[slot]..ctx.efactor_at[slot + 1]]
            .copy_from_slice(dwork);
    }
    Ok(())
}

/// `y = (B - E C^-1 E^T) x` over the KEPT numbering, without forming S.
///
/// The same operator [`schur_reduce`] builds explicitly, applied instead. It
/// trades one reduction for one pass per call, so it pays only when a solve
/// takes few enough products -- see docs/dev/CG.md.
///
/// Requires a preceding [`schur_factor_eliminated`] (or [`schur_reduce`]) on
/// this `h` and context, for the `C^-1`.
///
/// # Panics
///
/// When `x` or `y` is not the reduced system's dimension long.
pub fn schur_apply<I: Index, T: SchurReal>(
    sym: &SchurSymbolic<I>,
    h: &SparseBlockColMat<I, T>,
    ctx: &mut SchurContext<T>,
    x: &[T],
    y: &mut [T],
) {
    let hs = h.symbolic();
    assert_eq!(x.len(), sym.s.nrows());
    assert_eq!(y.len(), sym.s.nrows());
    build_s_to_h(sym, ctx);
    y.iter_mut().for_each(|v| *v = T::ZERO);

    // B x -- the kept-kept tiles, read out of H but placed by S's geometry.
    // Same upper-triangle convention as bsc::mul_symmetric_upper: a diagonal
    // tile carries only its upper half, so it is mirrored rather than read
    // whole.
    for j in 0..sym.s.nblk_cols() {
        let cols = sym.s.col_span(j);
        for b in sym.s.col_range(j) {
            let hb = ctx.s_to_h[b];
            if hb == usize::MAX {
                continue; // pure fill: exists in S, has no counterpart in H
            }
            let r = sym.s.blk_row(b);
            let rows = sym.s.row_span(r);
            let nr = rows.len();
            let payload = &h.vals()[hs.val_range(hb)];
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
                for (lc, cj) in cols.clone().enumerate() {
                    let xcj = x[cj];
                    let mut acc = T::ZERO;
                    for lr in 0..nr {
                        let a = payload[lc * nr + lr];
                        let ri = rows.start + lr;
                        y[ri] = y[ri] + a * xcj;
                        acc = acc + a * x[ri];
                    }
                    y[cj] = y[cj] + acc;
                }
            }
        }
    }

    // - E C^-1 E^T x, one eliminated block at a time
    ctx.panel.resize(sym.max_panel.max(sym.max_ew), T::ZERO);
    for slot in 0..sym.elim_diag.len() {
        let d_blk = sym.elim_diag[slot].zx();
        let we = hs.col_span(hs.blk_row(d_blk)).len();
        let orange = sym.elim_obs_ptr[slot].zx()..sym.elim_obs_ptr[slot + 1].zx();

        // t = -sum_a C_a^T x_a, then negated: gemm_sub only subtracts.
        let t = &mut ctx.panel[..we];
        t.iter_mut().for_each(|v| *v = T::ZERO);
        for o in orange.clone() {
            let wa = sym.obs_w[o].zx();
            let ca = sym.obs_ca_off[o].zx();
            let off = sym.obs_kept_off[o].zx();
            gemm_sub(t, &h.vals()[ca..ca + wa * we], !sym.obs_trans[o], we, wa,
                     &x[off..off + wa], 1);
        }
        for v in t.iter_mut() {
            *v = T::ZERO - *v;
        }
        let f = &ctx.efactors[ctx.efactor_at[slot]..ctx.efactor_at[slot + 1]];
        llt_solve_panel(f, t, we, 1);

        // y_a -= C_a u
        for o in orange {
            let wa = sym.obs_w[o].zx();
            let ca = sym.obs_ca_off[o].zx();
            let off = sym.obs_kept_off[o].zx();
            gemm_sub(&mut y[off..off + wa], &h.vals()[ca..ca + wa * we],
                     sym.obs_trans[o], wa, we, &ctx.panel[..we], 1);
        }
    }
}

/// Everything the implicit route needs that [`schur_apply`] does not supply:
/// the reduced right-hand side CG solves for, and S's diagonal blocks, which
/// its preconditioner factors. Neither requires S.
///
/// Also leaves the eliminated blocks' Cholesky factors in `ctx`, so a
/// [`schur_apply`] that follows needs no separate
/// [`schur_factor_eliminated`].
///
/// `diag` is filled with each kept block's `S_aa`, FULL and symmetric,
/// column-major, concatenated in kept order, and `spans` with its
/// `(scalar offset, width)` -- together the layout
/// `cg::BlockJacobi::from_diagonal_blocks` reads.
///
/// # Panics
///
/// When `rhs_kept` is not the reduced system's dimension long.
pub fn schur_prepare_implicit<I: Index, T: SchurReal>(
    sym: &SchurSymbolic<I>,
    h: &SparseBlockColMat<I, T>,
    rhs: &[T],
    ctx: &mut SchurContext<T>,
    rhs_kept: &mut [T],
    diag: &mut std::vec::Vec<T>,
    spans: &mut std::vec::Vec<(usize, usize)>,
) -> Result<(), SchurError> {
    let hs = h.symbolic();
    assert_eq!(rhs_kept.len(), sym.s.nrows());
    ctx.dwork.resize(sym.max_ew * sym.max_ew, T::ZERO);
    ctx.panel.resize(sym.max_panel, T::ZERO);
    size_efactors(sym, ctx);

    // Seed: the kept rhs is b_kept, and S's diagonal starts as H's.
    spans.clear();
    diag.clear();
    let mut diag_at = std::vec::Vec::with_capacity(sym.kept.len());
    for k in 0..sym.kept.len() {
        let orig = sym.kept[k].zx();
        let hspan = hs.col_span(orig);
        let kspan = sym.s.col_span(k);
        let w = kspan.len();
        rhs_kept[kspan.clone()].copy_from_slice(&rhs[hspan]);
        spans.push((kspan.start, w));
        diag_at.push(diag.len());
        let base = diag.len();
        diag.resize(base + w * w, T::ZERO);
        if let Some(b) = hs.col_range(orig).find(|&b| hs.blk_row(b) == orig) {
            let tile = &h.vals()[hs.val_range(b)];
            read_symmetric_tile(tile, &mut diag[base..base + w * w], w);
        }
    }

    // Subtract each eliminated block's contribution: the same panel solve the
    // reduction does, but only the a == a pairs and the rhs column are used.
    for slot in 0..sym.elim_diag.len() {
        let d_blk = sym.elim_diag[slot].zx();
        let e = hs.blk_row(d_blk);
        let we = hs.col_span(e).len();

        let dwork = &mut ctx.dwork[..we * we];
        read_symmetric_tile(&h.vals()[hs.val_range(d_blk)], dwork, we);
        if !llt_in_place(dwork, we) {
            return Err(SchurError::NotPositiveDefinite { block: e });
        }
        ctx.efactors[ctx.efactor_at[slot]..ctx.efactor_at[slot + 1]]
            .copy_from_slice(dwork);

        let orange = sym.elim_obs_ptr[slot].zx()..sym.elim_obs_ptr[slot + 1].zx();
        let col = sym.elim_ncols[slot].zx();
        for o in orange.clone() {
            let wo = sym.obs_w[o].zx();
            let ca = sym.obs_ca_off[o].zx();
            let tile = &h.vals()[ca..ca + wo * we];
            let ocol = sym.obs_panel_col[o].zx();
            let dst = &mut ctx.panel[ocol * we..(ocol + wo) * we];
            if sym.obs_trans[o] {
                dst.copy_from_slice(tile);
            } else {
                for cc in 0..wo {
                    for rr in 0..we {
                        dst[rr + cc * we] = tile[cc + rr * wo];
                    }
                }
            }
        }
        ctx.panel[col * we..col * we + we].copy_from_slice(&rhs[hs.col_span(e)]);
        llt_solve_panel(dwork, &mut ctx.panel[..(col + 1) * we], we, col + 1);

        for o in orange {
            let wa = sym.obs_w[o].zx();
            let ca = sym.obs_ca_off[o].zx();
            let tile = &h.vals()[ca..ca + wa * we];
            let trans = sym.obs_trans[o];
            let kept_off = sym.obs_kept_off[o].zx();
            // Kept ids are ascending in `spans`, and an observer's kept offset
            // IS the start of its span, so this locates its diagonal block.
            let ki = spans.binary_search_by_key(&kept_off, |&(s, _)| s).unwrap();
            let base = diag_at[ki];
            let ocol = sym.obs_panel_col[o].zx();
            let z_a = &ctx.panel[ocol * we..(ocol + wa) * we];
            gemm_sub(&mut diag[base..base + wa * wa], tile, trans, wa, we, z_a, wa);
            // b_kept(a) -= C_a z_b
            let z_b = &ctx.panel[col * we..col * we + we];
            gemm_sub(&mut rhs_kept[kept_off..kept_off + wa], tile, trans, wa, we, z_b, 1);
        }
    }
    Ok(())
}

/// Read a tile stored upper-only into a full symmetric `w x w` scratch.
#[inline]
fn read_symmetric_tile<T: SchurReal>(tile: &[T], out: &mut [T], w: usize) {
    for j in 0..w {
        for i in 0..=j {
            let v = tile[i + j * w];
            out[i + j * w] = v;
            out[j + i * w] = v;
        }
    }
}

/// Size the eliminated-block buffers (the factors, `z`, the recovered
/// blocks) and their offsets, and one scratch per thread. Widths come from
/// the structure, so this settles on the first call.
fn size_efactors<I: Index, T: SchurReal>(sym: &SchurSymbolic<I>, ctx: &mut SchurContext<T>) {
    let ne = sym.elim_diag.len();
    if ctx.efactor_at.len() != ne + 1 || ctx.z_at.len() != ne + 1 {
        ctx.efactor_at.clear();
        ctx.z_at.clear();
        let (mut at, mut zat) = (0usize, 0usize);
        for slot in 0..ne {
            ctx.efactor_at.push(at);
            ctx.z_at.push(zat);
            let we = sym.elim_w[slot].zx();
            at += we * we;
            zat += we;
        }
        ctx.efactor_at.push(at);
        ctx.z_at.push(zat);
        ctx.efactors.resize(at, T::ZERO);
        ctx.z.resize(zat, T::ZERO);
        ctx.xe.resize(zat, T::ZERO);
    }
    ctx.workers.resize_with(ctx.threads, ColumnScratch::default);
    for w in ctx.workers.iter_mut() {
        w.panel.resize(sym.max_panel.max(sym.max_ew), T::ZERO);
    }
}

/// One thread's scratch: the solve panel `Z` of the eliminated block at
/// hand, `we x (sum of its observers' widths in the range)`.
struct ColumnScratch<T> {
    panel: Vec<T>,
}

impl<T> Default for ColumnScratch<T> {
    fn default() -> Self {
        ColumnScratch { panel: Vec::new() }
    }
}

/// Where kept column `j` of S starts in its value buffer; `j == nk` is
/// the end.
#[inline]
fn col_vals<I: Index>(s: &SymbolicSparseBlockColMat<I>, j: usize) -> usize {
    let (_, _, blk_col_ptr, _, val_ptr) = s.parts();
    val_ptr[blk_col_ptr[j].zx()].zx()
}

/// Cut `0..n` into at most `threads` ranges of about equal weight, where
/// `weight[i]` is the weight before item `i` (`n + 1` entries). Returns
/// the boundaries, `n` last; empty ranges are left out.
fn cut_by_weight(weight: &[usize], threads: usize) -> Vec<usize> {
    let n = weight.len() - 1;
    let total = weight[n];
    let mut bounds = vec![0usize];
    let mut c = 0usize;
    for k in 1..threads {
        let target = total * k / threads;
        while c < n && weight[c] < target {
            c += 1;
        }
        if c > *bounds.last().unwrap() && c < n {
            bounds.push(c);
        }
    }
    bounds.push(n);
    bounds
}

/// Cut `0..n` into at most `threads` ranges of about equal count.
fn cut_by_count(n: usize, threads: usize) -> Vec<usize> {
    let mut bounds = vec![0usize];
    for k in 1..threads {
        let c = n * k / threads;
        if c > *bounds.last().unwrap() && c < n {
            bounds.push(c);
        }
    }
    bounds.push(n);
    bounds
}

/// Splits `buf` at the given ascending offsets, the first of which is
/// where `buf` starts.
fn split_at_offsets<'a, T>(mut buf: &'a mut [T], offsets: &[usize]) -> Vec<&'a mut [T]> {
    let mut parts = Vec::with_capacity(offsets.len() - 1);
    for w in offsets.windows(2) {
        let (head, tail) = buf.split_at_mut(w[1] - w[0]);
        parts.push(head);
        buf = tail;
    }
    parts
}

/// One thread's share of the columns stage of [`schur_reduce`]: a range
/// of kept columns, the values of S and the reduced rhs they cover, and
/// the thread's panel scratch.
struct ColumnTask<'a, T> {
    cols: core::ops::Range<usize>,
    vbase: usize,
    rbase: usize,
    s_vals: &'a mut [T],
    rhs_out: &'a mut [T],
    scratch: &'a mut ColumnScratch<T>,
}

/// S block index -> H block index for the kept-kept tiles, `usize::MAX` where
/// S has a tile that the elimination created and H never had.
fn build_s_to_h<I: Index, T>(sym: &SchurSymbolic<I>, ctx: &mut SchurContext<T>) {
    if ctx.s_to_h.len() == sym.s.nblocks() {
        return;
    }
    ctx.s_to_h.clear();
    ctx.s_to_h.resize(sym.s.nblocks(), usize::MAX);
    for i in 0..sym.copy_dst.len() {
        ctx.s_to_h[sym.copy_dst[i].zx()] = sym.copy_src[i].zx();
    }
}

/// numeric Schur reduction: fills `s` (allocated via
/// [`SchurSymbolic::alloc_s`]) with `S = Hkk - Hke Hee^-1 Hek` and
/// `rhs_out` (length `s.nrows()`, the kept blocks compacted in order)
/// with `bk - Hke Hee^-1 be`, from pre-damped `h` and `rhs`. `h` must
/// have the exact symbolic structure `sym` was built from.
///
/// The work runs in two stages:
///
/// 1. **factor** -- for every eliminated block `e`, the dense Cholesky
///    `D_e = L L^T` of its `w_e x w_e` diagonal tile (read
///    symmetric-from-upper) and `z_e = D_e^-1 b_e`, both kept in the
///    context: the columns below read them, and so does
///    [`schur_backsub`], which would otherwise redo the factor.
///
/// 2. **columns** -- the kept block columns of S, cut into one range per
///    thread ([`SchurContext::set_threads`]) by pair count. A range's
///    tiles are one range of S's values and its blocks one range of
///    `rhs_out`, so the threads write disjoint ranges and nothing is
///    locked. Each thread walks every eliminated block `e` in order and
///    takes the observers `b` of `e` that lie in its range, writing
///    `C_a = H(a, e)` for the coupling tile to observer `a`:
///
///    - the range of S and of `rhs_out` is zeroed first;
///    - the panel `Z = D_e^-1 [C_b^T ..]` over those observers, one
///      forward and one backward triangular solve per column -- that pair
///      of solves IS the `D_e^-1` application; no inverse is ever formed;
///    - `S(a, b) -= C_a Z_b` for every observer `a <= b` of `e`, the
///      target found through the column's tile table, and
///      `rhs_out(b) -= C_b z_e`;
///    - afterwards the range's tiles of `Hkk` and its slice of the kept
///      rhs are folded in, after the coupling terms, so f32 sums the many
///      small contributions among themselves before meeting the large
///      diagonal once, instead of rounding each contribution against it;
///    - the `a == b` products wrote the diagonal tiles in full, so their
///      strictly-lower parts are re-zeroed to restore the
///      upper-only-within-tile convention.
///
/// Every tile receives its contributions in eliminated-block order
/// whatever the cut, so the result is the same at any thread count.
/// Per-stage wall time lands in [`SchurContext::timing`] when enabled.
///
/// # Panics
///
/// When `rhs` is not `h`'s dimension long, or `rhs_out` not the reduced
/// system's.
pub fn schur_reduce<I: Index, T: SchurReal>(
    sym: &SchurSymbolic<I>,
    h: &SparseBlockColMat<I, T>,
    rhs: &[T],
    ctx: &mut SchurContext<T>,
    s: &mut SparseBlockColMat<I, T>,
    rhs_out: &mut [T],
) -> Result<(), SchurError> {
    let hs = h.symbolic();
    assert_eq!(rhs.len(), hs.nrows());
    assert_eq!(rhs_out.len(), sym.s.nrows());
    ctx.dwork.resize(sym.max_ew * sym.max_ew, T::ZERO);
    size_efactors(sym, ctx);
    let gather = ctx.timing.is_some();
    let mut t = SchurTiming::default();
    let mut sw = Stopwatch::new(gather);

    // stage 1: D_e = L L^T and z_e = D_e^-1 b_e for every eliminated block
    // (diagonal tiles are stored upper-only within the tile, so read
    // symmetric from the upper triangle)
    for slot in 0..sym.elim_diag.len() {
        let d_blk = sym.elim_diag[slot].zx();
        let e = hs.blk_row(d_blk);
        let we = sym.elim_w[slot].zx();
        let dwork = &mut ctx.dwork[..we * we];
        read_symmetric_tile(&h.vals()[hs.val_range(d_blk)], dwork, we);
        if !llt_in_place(dwork, we) {
            return Err(SchurError::NotPositiveDefinite { block: e });
        }
        ctx.efactors[ctx.efactor_at[slot]..ctx.efactor_at[slot + 1]].copy_from_slice(dwork);
        let z = &mut ctx.z[ctx.z_at[slot]..ctx.z_at[slot + 1]];
        z.copy_from_slice(&rhs[hs.col_span(e)]);
        llt_solve_panel(dwork, z, we, 1);
    }
    sw.lap(&mut t.factor);

    // stage 2: the kept columns, one range per thread
    {
        let SchurContext { efactors, efactor_at, z, z_at, workers, threads, chunk, .. } =
            &mut *ctx;
        let (efactors, efactor_at, z, z_at, chunk) = (&*efactors, &*efactor_at, &*z, &*z_at, *chunk);
        let bounds = cut_by_weight(&sym.col_pairs, *threads);
        let vbounds: Vec<usize> = bounds.iter().map(|&c| col_vals(&sym.s, c)).collect();
        let rbounds: Vec<usize> = bounds.iter().map(|&c| sym.s.col_part()[c].zx()).collect();
        let s_parts = split_at_offsets(s.vals_mut(), &vbounds);
        let r_parts = split_at_offsets(rhs_out, &rbounds);
        let mut tasks: Vec<ColumnTask<'_, T>> = Vec::with_capacity(bounds.len() - 1);
        for (((w, s_vals), rhs_out), scratch) in
            bounds.windows(2).zip(s_parts).zip(r_parts).zip(workers.iter_mut())
        {
            tasks.push(ColumnTask {
                cols: w[0]..w[1],
                vbase: col_vals(&sym.s, w[0]),
                rbase: sym.s.col_part()[w[0]].zx(),
                s_vals,
                rhs_out,
                scratch,
            });
        }
        let h_vals = h.vals();
        let nk = sym.kept.len();
        let run = |task: &mut ColumnTask<'_, T>| {
            let (vbase, rbase) = (task.vbase, task.rbase);
            let (c0, c1) = (task.cols.start, task.cols.end);
            let s_vals: &mut [T] = &mut *task.s_vals;
            let rhs_out: &mut [T] = &mut *task.rhs_out;
            let panel: &mut [T] = &mut task.scratch.panel;
            s_vals.fill(T::ZERO);
            rhs_out.fill(T::ZERO);
            let width = if chunk == 0 { c1 - c0 } else { chunk }.max(1);

            let mut k0 = c0;
            while k0 < c1 {
            let k1 = (k0 + width).min(c1);
            let whole = k0 == 0 && k1 == nk;
            for slot in 0..sym.elim_diag.len() {
                let start = sym.elim_obs_ptr[slot].zx();
                let end = sym.elim_obs_ptr[slot + 1].zx();
                let kept = &sym.obs_kept[start..end];
                // The block's observers in this chunk: they are ascending,
                // so one contiguous run of its list.
                let (lo, hi) = if whole {
                    (0, kept.len())
                } else if kept.is_empty() || kept[0].zx() >= k1 || kept[kept.len() - 1].zx() < k0 {
                    continue;
                } else {
                    (
                        kept.partition_point(|&k| k.zx() < k0),
                        kept.partition_point(|&k| k.zx() < k1),
                    )
                };
                if lo == hi {
                    continue;
                }
                let we = sym.elim_w[slot].zx();

                // Z = D_e^-1 [C_b^T for every observer b in the range]
                let mut pc = 0usize;
                for o in start + lo..start + hi {
                    let wo = sym.obs_w[o].zx();
                    let ca = sym.obs_ca_off[o].zx();
                    let tile = &h_vals[ca..ca + wo * we];
                    let dst = &mut panel[pc * we..(pc + wo) * we];
                    if sym.obs_trans[o] {
                        // stored (e, kept): the tile IS C^T (we x wo)
                        dst.copy_from_slice(tile);
                    } else {
                        // stored (kept, e) as wo x we: transpose into the panel
                        for cc in 0..wo {
                            for rr in 0..we {
                                dst[rr + cc * we] = tile[cc + rr * wo];
                            }
                        }
                    }
                    pc += wo;
                }
                let f = &efactors[efactor_at[slot]..efactor_at[slot + 1]];
                llt_solve_panel(f, &mut panel[..pc * we], we, pc);
                let z = &z[z_at[slot]..z_at[slot + 1]];
                let uw = sym.elim_uw[slot].zx();
                let utrans = sym.elim_utrans[slot];

                let mut pc = 0usize;
                for bi in lo..hi {
                    let ob = start + bi;
                    let wb = sym.obs_w[ob].zx();
                    let b = kept[bi].zx();
                    let zb = &panel[pc * we..(pc + wb) * we];
                    let at = &sym.col_at[sym.col_at_ptr[b]..sym.col_at_ptr[b + 1]];
                    let top = sym.col_top[b].zx();

                    // S(a, b) -= C_a Z_b for every observer a <= b of e
                    let krun = &sym.obs_kept[start..=ob];
                    let cas = &sym.obs_ca_off[start..=ob];
                    let uniform = uw != 0
                        && gemm_row_dispatch(
                            uw, we, utrans, s_vals, vbase, h_vals, zb, at, top, krun, cas,
                        );
                    if !uniform {
                        for (ai, (&k, &ca)) in core::iter::zip(krun, cas).enumerate() {
                            let oa = start + ai;
                            let wa = sym.obs_w[oa].zx();
                            let d = at[k.zx() - top] as usize - vbase;
                            let ca = ca.zx();
                            gemm_sub(
                                &mut s_vals[d..d + wa * wb],
                                &h_vals[ca..ca + wa * we],
                                sym.obs_trans[oa],
                                wa,
                                we,
                                zb,
                                wb,
                            );
                        }
                    }

                    // rhs_out(b) -= C_b z_e
                    let ca = sym.obs_ca_off[ob].zx();
                    let off = sym.obs_kept_off[ob].zx() - rbase;
                    gemm_sub(
                        &mut rhs_out[off..off + wb],
                        &h_vals[ca..ca + wb * we],
                        sym.obs_trans[ob],
                        wb,
                        we,
                        z,
                        1,
                    );
                    pc += wb;
                }
            }
            k0 = k1;
            }

            // Hkk's tiles and the kept rhs of the range's columns, on top of
            // the accumulated coupling terms. Diagonal tiles are upper-only
            // in H, so only S's upper triangle is touched.
            for ci in sym.copy_ptr[c0].zx()..sym.copy_ptr[c1].zx() {
                let src = hs.val_range(sym.copy_src[ci].zx());
                let dst = sym.s.val_range(sym.copy_dst[ci].zx());
                let sdst = &mut s_vals[dst.start - vbase..dst.end - vbase];
                for (d, &v) in core::iter::zip(sdst.iter_mut(), &h_vals[src]) {
                    *d = *d + v;
                }
            }
            for kc in c0..c1 {
                let src = hs.col_span(sym.kept[kc].zx());
                let dst = sym.s.col_span(kc);
                let w = dst.len();
                let rdst = &mut rhs_out[dst.start - rbase..dst.end - rbase];
                for (d, &v) in core::iter::zip(rdst.iter_mut(), &rhs[src]) {
                    *d = *d + v;
                }
                // The diagonal tile is the column's last, when it has one.
                let col = sym.s.col_range(kc);
                if col.end > col.start && sym.s.blk_row(col.end - 1) == kc {
                    let range = sym.s.val_range(col.end - 1);
                    let tile = &mut s_vals[range.start - vbase..range.end - vbase];
                    for j in 0..w {
                        for i in j + 1..w {
                            tile[i + j * w] = T::ZERO;
                        }
                    }
                }
            }
        };
        if tasks.len() == 1 {
            run(&mut tasks[0]);
        } else {
            crate::pool::run_over(&mut tasks, |_, task| run(task));
        }
    }
    sw.lap(&mut t.columns);
    if gather {
        ctx.timing = Some(t);
    }
    Ok(())
}

/// back-substitution after the reduced solve: recovers the eliminated
/// blocks and scatters everything into full-length coordinates.
///
/// With `x_kept` solving `S x_kept = bk'` (kept-compacted order), each
/// eliminated block `e` is recovered independently as
///
/// ```text
/// x_e = D_e^-1 (b_e - sum_a C_a^T x_a)
/// ```
///
/// over its observers `a` -- the same coupling tiles as the reduction,
/// applied in the OPPOSITE orientation (the forward pass applies
/// `C_a`, back-substitution applies `C_a^T`, so the storage-orientation
/// flag simply flips). `rhs` is the ORIGINAL full right-hand side;
/// `x_full` (length `h.nrows()`) receives the kept slices of `x_kept`
/// and the recovered eliminated blocks. `D_e^-1` is applied on the
/// factor the reduction computed -- a reduction on this `h` and context
/// always precedes: `x_kept` is the solution of the system it produced.
///
/// The eliminated blocks are recovered in ranges, one per thread
/// ([`SchurContext::set_threads`]), each into its own span of a context
/// buffer, and copied into `x_full` afterwards.
///
/// # Panics
///
/// When `rhs` or `x_full` is not `h`'s dimension long, or `x_kept` not
/// the reduced system's.
pub fn schur_backsub<I: Index, T: SchurReal>(
    sym: &SchurSymbolic<I>,
    h: &SparseBlockColMat<I, T>,
    rhs: &[T],
    x_kept: &[T],
    ctx: &mut SchurContext<T>,
    x_full: &mut [T],
) -> Result<(), SchurError> {
    let hs = h.symbolic();
    assert_eq!(rhs.len(), hs.nrows());
    assert_eq!(x_full.len(), hs.nrows());
    assert_eq!(x_kept.len(), sym.s.nrows());
    size_efactors(sym, ctx);

    // kept blocks scatter back to their original spans first: the
    // eliminated recovery below reads them out of x_full
    for (k, &orig) in sym.kept.iter().enumerate() {
        let dst = hs.col_span(orig.zx());
        let src = sym.s.col_span(k);
        x_full[dst].copy_from_slice(&x_kept[src]);
    }

    let ne = sym.elim_diag.len();
    {
        let SchurContext { efactors, efactor_at, xe, z_at, workers, threads, .. } = &mut *ctx;
        let (efactors, efactor_at, z_at) = (&*efactors, &*efactor_at, &*z_at);
        let bounds = cut_by_count(ne, *threads);
        let xbounds: Vec<usize> = bounds.iter().map(|&s| z_at[s]).collect();
        let x_parts = split_at_offsets(xe, &xbounds);
        let mut tasks: Vec<(core::ops::Range<usize>, &mut [T], &mut [T])> =
            Vec::with_capacity(bounds.len() - 1);
        for ((w, out), t) in bounds.windows(2).zip(x_parts).zip(workers.iter_mut()) {
            tasks.push((w[0]..w[1], out, &mut t.panel));
        }
        let h_vals = h.vals();
        let x_full = &*x_full;
        let run = |task: &mut (core::ops::Range<usize>, &mut [T], &mut [T])| {
            let (slots, out, scratch) = task;
            let xbase = z_at[slots.start];
            for slot in slots.clone() {
                let e = hs.blk_row(sym.elim_diag[slot].zx());
                let we = sym.elim_w[slot].zx();

                // t = b_e - sum_a C_a^T x_a
                let t = &mut scratch[..we];
                t.copy_from_slice(&rhs[hs.col_span(e)]);
                for o in sym.elim_obs_ptr[slot].zx()..sym.elim_obs_ptr[slot + 1].zx() {
                    let wa = sym.obs_w[o].zx();
                    let ca = sym.obs_ca_off[o].zx();
                    let xoff = sym.obs_orig_off[o].zx();
                    gemm_sub(
                        t,
                        &h_vals[ca..ca + wa * we],
                        !sym.obs_trans[o],
                        we,
                        wa,
                        &x_full[xoff..xoff + wa],
                        1,
                    );
                }

                // x_e = D_e^-1 t: the same two triangular solves as the
                // reduction's, on its factor
                let f = &efactors[efactor_at[slot]..efactor_at[slot + 1]];
                llt_solve_panel(f, t, we, 1);
                out[z_at[slot] - xbase..z_at[slot + 1] - xbase].copy_from_slice(t);
            }
        };
        if tasks.len() == 1 {
            run(&mut tasks[0]);
        } else {
            crate::pool::run_over(&mut tasks, |_, task| run(task));
        }
    }
    for slot in 0..ne {
        let e = hs.blk_row(sym.elim_diag[slot].zx());
        x_full[hs.col_span(e)].copy_from_slice(&ctx.xe[ctx.z_at[slot]..ctx.z_at[slot + 1]]);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // deterministic LCG so fixtures need no rand dependency
    struct Lcg(u64);
    impl Lcg {
        fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }

    /// upper-stored symmetric block matrix with mixed widths.
    /// widths [2, 3, 1, 2, 3, 2]; tiles:
    ///   diag on every block; couplings (0,1) (0,3) (1,2) (1,3) (1,5)
    ///   (2,3) (3,5) (0,4) (2,4) (4,5)
    /// eliminating [1, 4] exercises both storage orientations for the
    /// coupling tiles: (0,1)/(0,4)/(2,4) are stored (kept, elim);
    /// (1,2)/(1,3)/(1,5)/(4,5) are stored (elim, kept).
    fn fixture() -> (SparseBlockColMat<usize, f64>, Vec<f64>) {
        build_upper(
            &[0, 2, 5, 6, 8, 11, 13],
            &[
                (0, 0),
                (0, 1), (1, 1),
                (1, 2), (2, 2),
                (0, 3), (1, 3), (2, 3), (3, 3),
                (0, 4), (2, 4), (4, 4),
                (1, 5), (3, 5), (4, 5), (5, 5),
            ],
            42,
        )
    }

    /// random upper-stored symmetric block matrix + rhs over the given
    /// partition and block cells. diagonal tiles are upper-only within
    /// the tile (the assembly convention) and strongly diagonally
    /// dominant so every LLT (eliminated tiles AND the dense
    /// reference) succeeds.
    fn build_upper(
        part: &[usize],
        cells: &[(usize, usize)],
        seed: u64,
    ) -> (SparseBlockColMat<usize, f64>, Vec<f64>) {
        let anchors: Vec<(usize, usize)> =
            cells.iter().map(|&(r, c)| (part[r], part[c])).collect();
        let (sym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
            part.to_vec(),
            part.to_vec(),
            anchors.len(),
            |k| anchors[k],
        );
        let mut m = SparseBlockColMat::<usize, f64>::zeroed(sym);
        let mut rng = Lcg(seed);
        for v in m.vals_mut() {
            *v = rng.next();
        }
        for b in 0..part.len() - 1 {
            let w = part[b + 1] - part[b];
            let blk = m.symbolic().val_range(
                m.symbolic().col_range(b).find(|&i| m.symbolic().blk_row(i) == b).unwrap(),
            );
            let vals = &mut m.vals_mut()[blk];
            for i in 0..w {
                for j in 0..i {
                    vals[i + j * w] = 0.0;
                }
                vals[i + i * w] = 10.0 + vals[i + i * w].abs();
            }
        }
        let n = *part.last().unwrap();
        let rhs: Vec<f64> = (0..n).map(|_| rng.next()).collect();
        (m, rhs)
    }

    /// mirror the upper-stored block matrix into a full dense one
    fn full_dense(m: &SparseBlockColMat<usize, f64>) -> Vec<f64> {
        let n = m.symbolic().nrows();
        let d = m.to_dense();
        let mut full = vec![0.0; n * n];
        for j in 0..n {
            for i in 0..n {
                let v = if d[(i, j)] != 0.0 { d[(i, j)] } else { d[(j, i)] };
                full[i + j * n] = v;
            }
        }
        full
    }

    /// dense reference: S = Hkk - Hke Hee^-1 Hek and
    /// bk' = bk - Hke Hee^-1 be, via the module's own LLT on the full
    /// (block-diagonal) eliminated subsystem
    fn dense_reference(
        full: &[f64],
        n: usize,
        keep: &[usize],
        elim: &[usize],
        rhs: &[f64],
    ) -> (Vec<f64>, Vec<f64>) {
        let (nk, ne) = (keep.len(), elim.len());
        let mut hee = vec![0.0; ne * ne];
        for (a, &i) in elim.iter().enumerate() {
            for (b, &j) in elim.iter().enumerate() {
                hee[a + b * ne] = full[i + j * n];
            }
        }
        assert!(llt_in_place(&mut hee, ne));
        // panel [Hek | be] -> Hee^-1 [Hek | bk]
        let m = nk + 1;
        let mut panel = vec![0.0; ne * m];
        for (c, &j) in keep.iter().enumerate() {
            for (r, &i) in elim.iter().enumerate() {
                panel[r + c * ne] = full[i + j * n];
            }
        }
        for (r, &i) in elim.iter().enumerate() {
            panel[r + nk * ne] = rhs[i];
        }
        llt_solve_panel(&hee, &mut panel, ne, m);
        let mut s = vec![0.0; nk * nk];
        for a in 0..nk {
            for b in 0..nk {
                let mut acc = full[keep[a] + keep[b] * n];
                for r in 0..ne {
                    acc -= full[keep[a] + elim[r] * n] * panel[r + b * ne];
                }
                s[a + b * nk] = acc;
            }
        }
        let mut rk = vec![0.0; nk];
        for a in 0..nk {
            let mut acc = rhs[keep[a]];
            for r in 0..ne {
                acc -= full[keep[a] + elim[r] * n] * panel[r + nk * ne];
            }
            rk[a] = acc;
        }
        (s, rk)
    }

    fn run_and_compare(elim_blocks: &[usize]) {
        let (h, rhs) = fixture();
        check_vs_dense(&h, &rhs, &[0, 2, 5, 6, 8, 11, 13], elim_blocks);
    }

    fn check_vs_dense(
        h: &SparseBlockColMat<usize, f64>,
        rhs: &[f64],
        part: &[usize],
        elim_blocks: &[usize],
    ) {
        let nblk = part.len() - 1;
        let n = *part.last().unwrap();
        let sym = schur_symbolic(h.symbolic(), elim_blocks).unwrap();
        let mut s = sym.alloc_s::<f64>();
        let mut ctx = SchurContext::new();
        let mut rk = vec![0.0; s.symbolic().nrows()];
        schur_reduce(&sym, h, rhs, &mut ctx, &mut s, &mut rk).unwrap();

        // scalar keep/elim index lists
        let is_elim = |b: usize| elim_blocks.contains(&b);
        let keep_idx: Vec<usize> =
            (0..nblk).filter(|&b| !is_elim(b)).flat_map(|b| part[b]..part[b + 1]).collect();
        let elim_idx: Vec<usize> =
            (0..nblk).filter(|&b| is_elim(b)).flat_map(|b| part[b]..part[b + 1]).collect();
        let full = full_dense(h);
        let (s_ref, rk_ref) = dense_reference(&full, n, &keep_idx, &elim_idx, rhs);

        let nk = keep_idx.len();
        let sd = s.to_dense();
        for j in 0..nk {
            for i in 0..=j {
                let got = sd[(i, j)];
                let want = s_ref[i + j * nk];
                assert!(
                    (got - want).abs() <= 1e-11 * (1.0 + want.abs()),
                    "S[{}, {}]: {} vs {}",
                    i, j, got, want
                );
            }
        }
        for i in 0..nk {
            assert!(
                (rk[i] - rk_ref[i]).abs() <= 1e-11 * (1.0 + rk_ref[i].abs()),
                "rhs[{}]: {} vs {}",
                i, rk[i], rk_ref[i]
            );
        }

        // full-solve identity: reduce -> solve S -> backsub must
        // reproduce the direct full-system solve
        let mut x_ref = rhs.to_vec();
        let mut hfull = full.clone();
        assert!(llt_in_place(&mut hfull, n));
        llt_solve_panel(&hfull, &mut x_ref, n, 1);

        let mut xk = rk.clone();
        let mut sfull = s_ref.clone();
        assert!(llt_in_place(&mut sfull, nk));
        llt_solve_panel(&sfull, &mut xk, nk, 1);

        let mut x_full = vec![0.0; n];
        schur_backsub(&sym, h, rhs, &xk, &mut ctx, &mut x_full).unwrap();
        for i in 0..n {
            assert!(
                (x_full[i] - x_ref[i]).abs() <= 1e-9 * (1.0 + x_ref[i].abs()),
                "x[{}]: {} vs {}",
                i, x_full[i], x_ref[i]
            );
        }
    }

    #[test]
    fn matches_dense_reference() {
        run_and_compare(&[1, 4]);
    }

    #[test]
    fn single_block_and_widths() {
        run_and_compare(&[4]);
        run_and_compare(&[2]); // width-1 eliminated block
        run_and_compare(&[0]); // eliminated first, all couplings transposed
        run_and_compare(&[5]); // eliminated last, all couplings direct
    }


    /// FIXED_SHAPES advertises which shapes are fast; fixed_shapes! decides
    /// which ones actually are. Nothing in the OUTPUT distinguishes them --
    /// the fallback computes the same values -- so a match arm that never
    /// fires, or a list that promises a kernel nobody wrote, would cost speed
    /// in silence. Walk every shape up to 9x9x9 and demand the two agree, in
    /// both orientations. This is the only test that can see the property.
    #[test]
    fn the_shape_list_and_the_dispatch_agree() {
        let hits = || FIXED_KERNEL_HITS.with(|c| c.get());
        for wa in 1..=10usize {
            for we in 1..=10usize {
                for wb in 1..=10usize {
                    for trans in [false, true] {
                        let ca = vec![0.5f64; wa * we];
                        let zb = vec![0.5f64; we * wb];
                        let mut dst = vec![0.0f64; wa * wb];
                        FIXED_KERNEL_HITS.with(|c| c.set(0));
                        gemm_sub(&mut dst, &ca, trans, wa, we, &zb, wb);
                        let unrolled = hits() == 1;
                        assert_eq!(
                            unrolled,
                            has_fixed_kernel(wa, we, wb),
                            "({}, {}, {}) trans={}: dispatch says unrolled={}, \
                             FIXED_SHAPES says {}",
                            wa, we, wb, trans, unrolled,
                            has_fixed_kernel(wa, we, wb)
                        );
                    }
                }
            }
        }
    }

    /// Every shape with an unrolled kernel must agree with the generic
    /// loop, in both orientations. A wrong const-generic arm would be
    /// invisible on the models that do not use that shape.
    #[test]
    fn fixed_shape_kernels_match_the_generic_loop() {
        // Every shape up to 9x9x9, not just the listed ones: an unrolled kernel
        // with a wrong arm and the nano-gemm fallback with wrong strides are both
        // invisible until someone checks the values. A transposed lhs is where that
        // bites -- it is the kind of thing that is right for square tiles and wrong
        // for the rest.
        //
        // The tail crosses TRANS_PACK_MAX. A transposed tile at or under the cap is
        // transposed into a stack buffer and passed column-major; over the cap it
        // stays put and is passed with a row stride. Two code paths, one of which
        // the 9x9x9 sweep never reaches (81 < 144), so the boundary is walked here:
        // 143 and 144 pack, 145 and up do not.
        let shapes: Vec<(usize, usize, usize)> = (1..=9)
            .flat_map(|wa| (1..=9).flat_map(move |we| (1..=9).map(move |wb| (wa, we, wb))))
            .chain([
                (12, 12, 3), // 144: the cap exactly, still packed
                (11, 13, 3), // 143: just under
                (13, 12, 3), // 156: just over, strided
                (12, 13, 2), // 156: just over the other way
                (16, 16, 4), // 256: well over
                (20, 3, 5),  // 60: wide but under the cap
                (3, 20, 5),  // 60: tall but under the cap
            ])
            .collect();
        for (wa, we, wb) in shapes {
            for trans in [false, true] {
                let mut rng = 12345u64;
                let mut next = || {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    ((rng >> 33) as f64) / (u32::MAX as f64) - 0.5
                };
                // C_a is wa x we, stored as itself or as its transpose
                let ca: Vec<f64> = (0..wa * we).map(|_| next()).collect();
                let zb: Vec<f64> = (0..we * wb).map(|_| next()).collect();
                let dst0: Vec<f64> = (0..wa * wb).map(|_| next()).collect();

                // reference: dst -= C_a * Z_b, read straight from the
                // definition, whatever the storage orientation
                let mut want = dst0.clone();
                for c in 0..wb {
                    for i in 0..wa {
                        let mut acc = 0.0;
                        for k in 0..we {
                            let a = if trans { ca[k + i * we] } else { ca[i + k * wa] };
                            acc += a * zb[k + c * we];
                        }
                        want[i + c * wa] -= acc;
                    }
                }

                let mut got = dst0.clone();
                gemm_sub(&mut got, &ca, trans, wa, we, &zb, wb);
                for (i, (g, w)) in core::iter::zip(&got, &want).enumerate() {
                    assert!(
                        (g - w).abs() <= 1e-12 * (1.0 + w.abs()),
                        "({}, {}, {}) trans={}: element {}: {} vs {}",
                        wa, we, wb, trans, i, g, w
                    );
                }
            }
        }
    }

    /// The band bound the reduce/decline heuristic leans on: a landmark
    /// seen only from nearby poses leaves a narrow band in S, and one
    /// that reaches across the trajectory widens it to the whole system.
    #[test]
    fn kept_bandwidth_tracks_how_far_landmarks_reach() {
        let part = [0, 2, 4, 6, 8, 10, 12];
        // blocks 0..3 are a pose chain (odometry couples neighbours);
        // blocks 4 and 5 are landmarks, each seen from two poses.
        let chain = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 3), (3, 3)];

        // near: landmark 4 sees poses 0-1, landmark 5 sees poses 2-3.
        let mut cells = chain.to_vec();
        cells.extend([(0, 4), (1, 4), (4, 4), (2, 5), (3, 5), (5, 5)]);
        let (h, rhs) = build_upper(&part, &cells, 7);
        let sym = schur_symbolic(h.symbolic(), &[4, 5]).unwrap();
        assert_eq!(sym.kept_size(), 8);
        // S couples only neighbouring poses: 2 scalar columns of pose,
        // reaching back over one more pose block.
        assert_eq!(sym.kept_bandwidth(), 4);
        check_vs_dense(&h, &rhs, &part, &[4, 5]);

        // far: landmark 5 now sees poses 0 and 3, the ends of the chain.
        let mut cells = chain.to_vec();
        cells.extend([(0, 4), (1, 4), (4, 4), (0, 5), (3, 5), (5, 5)]);
        let (h, rhs) = build_upper(&part, &cells, 7);
        let sym = schur_symbolic(h.symbolic(), &[4, 5]).unwrap();
        assert_eq!(sym.kept_size(), 8);
        // eliminating it couples pose 0 to pose 3, so the band spans all
        // 8 kept parameters -- no better than calling S dense.
        assert_eq!(sym.kept_bandwidth(), 8);
        check_vs_dense(&h, &rhs, &part, &[4, 5]);
    }

    #[test]
    fn mixed_width_eliminated_set() {
        // Several entity TYPES eliminated together -- e.g. 3-parameter
        // points and 6-parameter lines in one reduction. Blocks 2 and 5
        // have widths 1 and 2 and are uncoupled (no (2, 5) tile), so they
        // are a legal eliminated set with differing block sizes.
        run_and_compare(&[2, 5]);
        // and with a third, wider one (block 0, width 2; no (0, 2) or
        // (0, 5) tile either)
        run_and_compare(&[0, 2, 5]);
    }

    #[test]
    fn wide_blocks_both_orientations() {
        // widths [6, 3, 6], eliminate the middle block: observer 0
        // couples non-transposed (tile (0, 1)) and observer 2
        // transposed (tile (1, 2)), so the (6, 3, 6)-shaped pairs hit
        // both the fixed-size fast path and the transposed fallback
        let part = [0usize, 6, 9, 15];
        let cells = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)];
        let (h, rhs) = build_upper(&part, &cells, 7);
        check_vs_dense(&h, &rhs, &part, &[1]);
    }

    /// A pair shape with one observer width on both sides must reach
    /// [`gemm_row`], not just the pair-at-a-time loop: the two compute the
    /// same values, so nothing in the output would show a missing arm in
    /// `uniform_pair_shapes!`. A mixed pair shape (two observer widths) has
    /// no uniform run; it must reach the unrolled kernel through the pair
    /// loop and match the dense reference.
    #[test]
    fn every_pair_shape_reaches_the_uniform_run() {
        for &(wa, we, wb) in FIXED_SHAPES.iter() {
            if wb == 1 {
                continue; // a one-column update, not a pair shape
            }
            if wa != wb {
                // observers of two widths around the eliminated block
                let part = [0usize, wa, wa + we, wa + we + wb];
                let cells = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)];
                let (h, rhs) = build_upper(&part, &cells, 5);
                UNIFORM_RUN_HITS.with(|c| c.set(0));
                FIXED_KERNEL_HITS.with(|c| c.set(0));
                check_vs_dense(&h, &rhs, &part, &[1]);
                assert_eq!(UNIFORM_RUN_HITS.with(|c| c.get()), 0,
                    "({}, {}, {}) took the uniform run with two observer widths", wa, we, wb);
                assert!(FIXED_KERNEL_HITS.with(|c| c.get()) > 0,
                    "({}, {}, {}) never reached an unrolled kernel", wa, we, wb);
                continue;
            }
            // Eliminating the LAST block leaves both coupling tiles stored
            // (kept, elim); eliminating the FIRST one stores both the other
            // way. Either way all observers agree, which is what the uniform
            // path needs.
            for elim_first in [false, true] {
                let part = if elim_first {
                    [0, we, we + wa, we + 2 * wa]
                } else {
                    [0, wa, 2 * wa, 2 * wa + we]
                };
                let elim = if elim_first { 0 } else { 2 };
                let cells = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
                let (h, rhs) = build_upper(&part, &cells, 3);
                UNIFORM_RUN_HITS.with(|c| c.set(0));
                check_vs_dense(&h, &rhs, &part, &[elim]);
                // One run per observer: the block has two.
                assert_eq!(
                    UNIFORM_RUN_HITS.with(|c| c.get()),
                    2,
                    "({}, {}, {}) elim_first={} did not take the uniform run",
                    wa, we, wb, elim_first
                );
            }
        }
    }

    /// A trajectory: `poses` poses of width 3, each landmark of width 2 seen
    /// from `span` consecutive poses, so the kept columns carry unequal
    /// pair counts for the cut to balance.
    fn trajectory(poses: usize, landmarks: usize, span: usize) -> (SparseBlockColMat<usize, f64>, Vec<f64>, Vec<usize>) {
        let mut part = vec![0usize];
        for _ in 0..poses {
            part.push(part.last().unwrap() + 3);
        }
        for _ in 0..landmarks {
            part.push(part.last().unwrap() + 2);
        }
        let mut cells: Vec<(usize, usize)> = (0..poses + landmarks).map(|b| (b, b)).collect();
        for p in 1..poses {
            cells.push((p - 1, p));
        }
        for l in 0..landmarks {
            let first = (l * (poses - span)) / landmarks.max(1);
            for p in first..first + span {
                cells.push((p, poses + l));
            }
        }
        let (h, rhs) = build_upper(&part, &cells, 17);
        (h, rhs, (poses..poses + landmarks).collect())
    }

    /// The columns run in the same order whatever the thread count, so a
    /// threaded reduction and back-substitution match the sequential ones
    /// to the bit -- with more threads than columns too, where the cut
    /// leaves ranges out.
    #[test]
    fn every_thread_count_matches_the_sequential_reduction_to_the_bit() {
        let (small, small_rhs) = fixture();
        let (traj, traj_rhs, traj_elim) = trajectory(12, 30, 4);
        for (h, rhs, elim) in [(small, small_rhs, vec![1usize, 4]), (traj, traj_rhs, traj_elim)] {
            let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
            let reduce = |threads: usize, chunk: usize| {
                let mut s = sym.alloc_s::<f64>();
                let mut ctx = SchurContext::new();
                ctx.set_threads(threads);
                ctx.set_chunk_columns(chunk);
                let mut rk = vec![0.0; s.symbolic().nrows()];
                schur_reduce(&sym, &h, &rhs, &mut ctx, &mut s, &mut rk).unwrap();
                // Any kept solution serves the back-substitution comparison.
                let xk: Vec<f64> = rk.iter().map(|v| v * 0.5).collect();
                let mut x_full = vec![0.0; h.symbolic().nrows()];
                schur_backsub(&sym, &h, &rhs, &xk, &mut ctx, &mut x_full).unwrap();
                (s.vals().to_vec(), rk, x_full)
            };
            let base = reduce(1, 0);
            for (threads, chunk) in [(2, 0), (3, 0), (4, 0), (16, 0), (1, 1), (1, 3), (4, 2), (3, 5)] {
                let got = reduce(threads, chunk);
                assert_eq!(got.0, base.0, "S at {} threads, chunk {}", threads, chunk);
                assert_eq!(got.1, base.1, "rhs at {} threads, chunk {}", threads, chunk);
                assert_eq!(got.2, base.2, "x at {} threads, chunk {}", threads, chunk);
            }
        }
        // And the trajectory's reduction is right at all.
        let (h, rhs, elim) = trajectory(12, 30, 4);
        let part: Vec<usize> = h.symbolic().col_part().iter().map(|&p| p).collect();
        check_vs_dense(&h, &rhs, &part, &elim);
    }

    #[test]
    fn mixed_observer_widths_skip_the_uniform_run() {
        // widths [6, 3, 9]: eliminate the middle one and the two observers
        // disagree, so the pair loop must run shape by shape and still match
        // the dense reference.
        let part = [0usize, 6, 9, 18];
        let cells = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)];
        let (h, rhs) = build_upper(&part, &cells, 11);
        UNIFORM_RUN_HITS.with(|c| c.set(0));
        check_vs_dense(&h, &rhs, &part, &[1]);
        assert_eq!(UNIFORM_RUN_HITS.with(|c| c.get()), 0);
    }

    #[test]
    fn shape_census_covers_the_one_column_updates() {
        // widths [6, 3, 6], eliminate the middle one: observers 0 and 2 give
        // three pair GEMMs of (6, 3, 6), one reduction rhs update each at
        // (6, 3, 1), and one back-substitution update each at (3, 6, 1).
        let part = [0usize, 6, 9, 15];
        let cells = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2), (0, 2)];
        let (h, _) = build_upper(&part, &cells, 7);
        let sym = schur_symbolic(h.symbolic(), &[1]).unwrap();
        let mut got = sym.gemm_shapes().to_vec();
        got.sort();
        assert_eq!(got, vec![((3, 6, 1), 2), ((6, 3, 1), 2), ((6, 3, 6), 3)]);
        // The one-column updates run once per observation, so they need a
        // kernel as much as the pair GEMMs do. A census that skipped them
        // would let those stages sit on the fallback unseen.
        for &((wa, we, wb), _) in sym.gemm_shapes() {
            assert!(has_fixed_kernel(wa, we, wb), "no kernel for ({}, {}, {})", wa, we, wb);
        }
    }

    /// The implicit product must agree with the explicit one it replaces:
    /// same operator, one formed and multiplied, the other multiplied
    /// directly. Checked over a basis so every column of S is compared, not
    /// just one direction.
    #[test]
    fn implicit_apply_matches_the_reduced_system() {
        // Uncoupled sets only -- the symbolic rejects anything else, and it
        // has its own test for that.
        for elim in [&[2usize][..], &[1, 4][..], &[0][..], &[5][..]] {
            let (h, rhs) = fixture();
            let sym = schur_symbolic(h.symbolic(), elim).unwrap();
            let nk = sym.s.nrows();

            // explicit: form S, multiply by it
            let mut s = sym.alloc_s::<f64>();
            let mut ctx = SchurContext::new();
            let mut rk = vec![0.0; nk];
            schur_reduce(&sym, &h, &rhs, &mut ctx, &mut s, &mut rk).unwrap();

            // implicit: factor the eliminated blocks, then apply
            let mut ictx = SchurContext::new();
            schur_factor_eliminated(&sym, &h, &mut ictx).unwrap();

            for k in 0..nk {
                let mut e = vec![0.0; nk];
                e[k] = 1.0;
                let mut want = vec![0.0; nk];
                s.mul_symmetric_upper(&e, &mut want);
                let mut got = vec![0.0; nk];
                schur_apply(&sym, &h, &mut ictx, &e, &mut got);
                for i in 0..nk {
                    assert!(
                        (got[i] - want[i]).abs() < 1e-9,
                        "elim {:?}, column {}, row {}: implicit {} vs explicit {}",
                        elim, k, i, got[i], want[i]
                    );
                }
            }
        }
    }

    /// The implicit route's right-hand side and diagonal blocks must be the
    /// ones the reduction produces -- the rhs it solves for, and the blocks
    /// its preconditioner factors.
    #[test]
    fn implicit_rhs_and_diagonal_match_the_reduction() {
        for elim in [&[2usize][..], &[1, 4][..], &[0][..]] {
            let (h, rhs) = fixture();
            let sym = schur_symbolic(h.symbolic(), elim).unwrap();
            let nk = sym.s.nrows();

            let mut s = sym.alloc_s::<f64>();
            let mut ctx = SchurContext::new();
            let mut rk = vec![0.0; nk];
            schur_reduce(&sym, &h, &rhs, &mut ctx, &mut s, &mut rk).unwrap();

            let mut ictx = SchurContext::new();
            let mut irk = vec![0.0; nk];
            let mut diag = std::vec::Vec::new();
            let mut spans = std::vec::Vec::new();
            schur_prepare_implicit(&sym, &h, &rhs, &mut ictx, &mut irk,
                                   &mut diag, &mut spans).unwrap();

            for i in 0..nk {
                assert!((irk[i] - rk[i]).abs() < 1e-9,
                    "elim {:?}, rhs {}: implicit {} vs explicit {}", elim, i, irk[i], rk[i]);
            }

            // Every diagonal block, against the same block read out of S.
            let mut at = 0usize;
            for (k, &(start, w)) in spans.iter().enumerate() {
                assert_eq!(start, sym.s.col_span(k).start);
                let tile = s.get_block(k, k).expect("S has no diagonal tile");
                for c in 0..w {
                    for r in 0..w {
                        // S stores the upper triangle only; diag is full.
                        let (lo, hi) = if r <= c { (r, c) } else { (c, r) };
                        let want = tile[(lo, hi)];
                        let got = diag[at + c * w + r];
                        assert!((got - want).abs() < 1e-9,
                            "elim {:?}, block {} ({},{}) : {} vs {}",
                            elim, k, r, c, got, want);
                    }
                }
                at += w * w;
            }
        }
    }

    /// The factors the implicit path computes are the ones the reduction
    /// computes -- so a reduction can hand them over, and vice versa.
    #[test]
    fn standalone_factoring_matches_the_reduction() {
        let (h, rhs) = fixture();
        let sym = schur_symbolic(h.symbolic(), &[1, 4]).unwrap();
        let mut s = sym.alloc_s::<f64>();
        let mut rk = vec![0.0; s.symbolic().nrows()];
        let mut a = SchurContext::new();
        schur_reduce(&sym, &h, &rhs, &mut a, &mut s, &mut rk).unwrap();
        let mut b = SchurContext::new();
        schur_factor_eliminated(&sym, &h, &mut b).unwrap();
        assert_eq!(a.efactor_at, b.efactor_at);
        for i in 0..a.efactors.len() {
            assert!((a.efactors[i] - b.efactors[i]).abs() < 1e-12, "factor {}", i);
        }
    }

    #[test]
    fn observer_pair_bookkeeping() {
        let (h, _) = fixture();
        let sym = schur_symbolic(h.symbolic(), &[2]).unwrap();
        // block 2 (width 1) couples to blocks 1, 3, 4 -> 3 observers,
        // 3 * 4 / 2 = 6 pairs
        assert_eq!(sym.pair_count(), 6);
    }

    #[test]
    fn clique_structure_present() {
        let (h, _) = fixture();
        let sym = schur_symbolic(h.symbolic(), &[1, 4]).unwrap();
        // eliminating block 1 couples its observers {0, 2, 3, 5}
        // pairwise (kept ids 0, 1, 2, 3). tile (2, 5) -> kept (1, 3)
        // is clique-only: H does not store it
        let s = &sym.s;
        let has = |kr: usize, kc: usize| s.col_range(kc).any(|b| s.blk_row(b) == kr);
        assert!(has(1, 3), "clique tile (2,5) must exist in S");
        // all four observers share eliminated block 1, so the kept
        // upper triangle of S is complete
        for kc in 0..4 {
            for kr in 0..=kc {
                assert!(has(kr, kc), "S({}, {}) missing", kr, kc);
            }
        }
    }

    #[test]
    fn coupled_eliminated_rejected() {
        let (h, _) = fixture();
        // blocks 1 and 2 are coupled by tile (1, 2)
        assert_eq!(
            schur_symbolic(h.symbolic(), &[1, 2]).unwrap_err(),
            SchurError::CoupledEliminated { row: 1, col: 2 }
        );
    }

    #[test]
    fn bad_set_rejected() {
        let (h, _) = fixture();
        assert_eq!(
            schur_symbolic(h.symbolic(), &[4, 1]).unwrap_err(),
            SchurError::BadEliminatedSet
        );
        assert_eq!(
            schur_symbolic(h.symbolic(), &[6]).unwrap_err(),
            SchurError::BadEliminatedSet
        );
    }

    #[test]
    fn missing_diagonal_rejected() {
        // structure without a diagonal tile on block 1
        let part = vec![0usize, 2, 4, 6];
        let cells = [(0usize, 0usize), (0, 1), (1, 2), (2, 2)];
        let anchors: Vec<(usize, usize)> =
            cells.iter().map(|&(r, c)| (part[r], part[c])).collect();
        let (sym, _) = SymbolicSparseBlockColMat::from_scalar_coords(
            part.clone(),
            part,
            anchors.len(),
            |k| anchors[k],
        );
        assert_eq!(
            schur_symbolic(&sym, &[1]).unwrap_err(),
            SchurError::MissingDiagonal { block: 1 }
        );
    }

    #[test]
    fn not_positive_definite_reported() {
        let (mut h, rhs) = fixture();
        // wreck eliminated block 4's diagonal tile
        let sym4 = h.symbolic().clone();
        let d4 = sym4.col_range(4).find(|&b| sym4.blk_row(b) == 4).unwrap();
        for v in &mut h.vals_mut()[sym4.val_range(d4)] {
            *v = -1.0;
        }
        let sym = schur_symbolic(h.symbolic(), &[4]).unwrap();
        let mut s = sym.alloc_s::<f64>();
        let mut rk = vec![0.0; s.symbolic().nrows()];
        assert_eq!(
            schur_reduce(&sym, &h, &rhs, &mut SchurContext::new(), &mut s, &mut rk).unwrap_err(),
            SchurError::NotPositiveDefinite { block: 4 }
        );
    }
}
