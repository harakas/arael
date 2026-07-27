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
use faer::Index;
use faer::traits::ComplexField;

/// scalar operations the hand-rolled tile kernels need. the kernels
/// run on raw column-major slices (tile sizes are single digits, where
/// plain loops beat dispatching into a general GEMM).
pub trait SchurReal:
    ComplexField
    + Copy
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
    /// shape, and on the shapes it is asked for it is up to 4x faster than the plain
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
    /// and a cache has to be searched before it can save that. Measured: a
    /// thread-local hash map costs ~25 ns against a GEMM that takes 15, which made
    /// this fallback SLOWER than the plain loop it replaces; a thread-local linear
    /// scan over the few live shapes comes out level with just rebuilding. Level is
    /// not worth the state, so there is none.
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

#[derive(Debug, PartialEq, Eq)]
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
    /// its diagonal tile, plus ranges into the obs_* / pair_off arrays
    /// and the panel width (observer columns, without the rhs column)
    elim_diag: Vec<I>,
    elim_obs_ptr: Vec<I>,
    elim_pair_ptr: Vec<I>,
    elim_ncols: Vec<I>,
    /// per eliminated block, the observer width shared by all of its
    /// observers, or 0 when they differ -- 0 also when they differ in
    /// storage orientation, which [`gemm_tri`] needs constant too.
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
    /// where every observer-pair target starts in S's value array: per
    /// eliminated block, for each observer b (ascending) all pairs
    /// (a, b) with a <= b, a ascending -- the numeric pass consumes it
    /// in this exact order
    pair_off: Vec<I>,
    /// S block indices of the kept diagonal tiles (for the final
    /// zero-lower pass -- diagonal tiles are upper-only by convention)
    s_diag: Vec<I>,
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
        self.pair_off.len()
    }
    /// The GEMM tile shapes this reduction needs, `((wa, we, wb), calls)`, in
    /// no particular order. A shape not in [`FIXED_SHAPES`] goes to the
    /// nano-gemm fallback, which costs about 1.2-1.4x the unrolled kernel, so a
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

/// reusable numeric workspaces for [`schur_reduce`]; grows to the
/// symbolic sizes on first use, no allocation afterwards.
pub struct SchurContext<T> {
    /// lower-LLT factor of the current diagonal tile, column-major
    dwork: Vec<T>,
    /// solve panel `Z = D^-1 [C^T | b_e]`, column-major, width
    /// `sum(observer widths) + 1`
    panel: Vec<T>,
    /// per-stage breakdown of the last [`schur_reduce`] call, gathered
    /// only when enabled (the clock is never read otherwise)
    timing: Option<SchurTiming>,
}

impl<T> Default for SchurContext<T> {
    fn default() -> Self {
        Self { dwork: Vec::new(), panel: Vec::new(), timing: None }
    }
}

impl<T> SchurContext<T> {
    pub fn new() -> Self {
        Self::default()
    }
    /// gather a per-stage [`SchurTiming`] on every subsequent
    /// [`schur_reduce`] call (costs a few clock reads per eliminated
    /// block; leave off for production solves)
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
    /// stage 1: zero S, copy the Hkk tiles and the kept rhs slices
    pub seed: std::time::Duration,
    /// stage 2a: dense Cholesky of every eliminated diagonal tile
    pub factor: std::time::Duration,
    /// stage 2b: gather each panel and triangular-solve it
    /// (`Z = D_e^-1 [C^T | b_e]`)
    pub panel: std::time::Duration,
    /// stage 2c: the observer-pair GEMMs into S (the flop driver)
    pub gemm: std::time::Duration,
    /// stage 2d: the per-observer rhs updates
    pub rhs: std::time::Duration,
    /// stage 3: re-zero the strictly-lower part of S's diagonal tiles
    pub finish: std::time::Duration,
}

impl SchurTiming {
    pub fn total(&self) -> std::time::Duration {
        self.seed + self.factor + self.panel + self.gemm + self.rhs + self.finish
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
    let mut elim_obs_ptr = Vec::with_capacity(ne + 1);
    let mut elim_pair_ptr = Vec::with_capacity(ne + 1);
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
    let mut max_panel = 0usize;
    let mut max_ew = 0usize;
    let mut total_pairs = 0usize;
    let mut reduce_flops = 0.0f64;
    let mut shapes: Vec<((usize, usize, usize), usize)> = Vec::new();
    elim_obs_ptr.push(I::truncate(0));
    elim_pair_ptr.push(I::truncate(0));
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
        elim_obs_ptr.push(I::truncate(obs_trans.len()));
        elim_pair_ptr.push(I::truncate(total_pairs));
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
    let mut pair_off = vec![I::truncate(0); total_pairs];
    let mut s_diag = Vec::with_capacity(nk);
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
            if kr == kc {
                s_diag.push(I::truncate(blk_row_idx.len()));
            }
            blk_row_idx.push(I::truncate(kr));
            vals_end += (kept_part[kr + 1] - kept_part[kr]) * colw;
            val_ptr.push(I::truncate(vals_end));
        }
        blk_col_ptr.push(I::truncate(blk_row_idx.len()));
        for ci in copy_begin..copy_cursor {
            copy_dst[ci] = I::truncate(blk_at[copy_kk[ci].1] as usize);
        }
        for &(slot, bi) in &land_ent[land_ptr[kc]..land_ptr[kc + 1]] {
            let start = elim_obs_ptr[slot as usize].zx();
            let base =
                elim_pair_ptr[slot as usize].zx() + (bi as usize) * (bi as usize + 1) / 2;
            for ai in 0..=bi as usize {
                let ka = kept_of[obs_block[start + ai].zx()].zx();
                // the numeric pass wants the target's scalar start, not its
                // block index. Every block of this column has its val_ptr
                // entry by now, so resolve it here rather than in a second
                // pass over all the pairs.
                pair_off[base + ai] = val_ptr[blk_at[ka] as usize];
            }
        }
    }
    let kp: Vec<I> = kept_part.iter().map(|&x| I::truncate(x)).collect();
    let s = SymbolicSparseBlockColMat::new_checked(
        kp.clone(),
        kp,
        blk_col_ptr,
        blk_row_idx,
        val_ptr,
    );
    let copy_src: Vec<I> = copy_kk.iter().map(|&(hb, _, _)| hb).collect();

    Ok(SchurSymbolic {
        kept,
        kept_of,
        s,
        copy_src,
        copy_dst,
        elim_diag,
        elim_obs_ptr,
        elim_pair_ptr,
        elim_ncols,
        elim_uw,
        elim_utrans,
        obs_trans,
        obs_ca_off,
        obs_w,
        obs_panel_col,
        obs_kept_off,
        obs_orig_off,
        pair_off,
        s_diag,
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
/// the compiler fully unroll and vectorize (measured 2x on the slam
/// pair GEMMs vs the runtime-dimension loop)
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
/// wins once there is enough of a column to vectorize (`WA`) and few enough columns to
/// copy (`WE`).
///
/// Measured with `--example gemmbench` on aarch64/NEON, which is the tighter
/// constraint -- on x86/AVX2 transposing first wins on every shape tested, so anything
/// that pays there pays here. Both scalars agree. Transposed, `fixed` -> `fixed+T`:
///
/// ```text
///                     f64 (aarch64)   f32 (aarch64)
///   (6,1,6)  in         1.87x           2.11x
///   (6,2,6)  in         1.75x           1.99x
///   (6,3,6)  in         1.31x           1.20x      <- the SLAM workhorse
///   (6,4,6)  in         1.13x           1.21x
///   (9,3,9)  in         1.22x           1.69x      <- BAL
///   (7,3,7)  in         1.34x           1.10x
///   (6,6,6)  out        1.01x           0.99x      <- WE = 6: the copy stops paying
///   (9,6,9)  out        0.95x           0.98x
///   (3,3,3)  out        0.85x           0.90x      <- WA = 3: no column to vectorize
///   (3,4,3)  out        0.83x           0.79x
///   (2,3,2)  out        0.64x           0.65x
/// ```
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
/// costs about 1.2-1.4x the unrolled kernel on these sizes (`--example
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
/// which is how a caller finds out it is paying the 2x.
/// The widths are the ones SLAM systems actually use. Observers: 3 (a 2D
/// pose), 6 (a 3D pose), 7 (a similarity, for scale-aware loop closure), 9 (a
/// camera with intrinsics, which is also what a BAL camera and a NavState
/// are). Marginalized: 1 (inverse depth), 2 (a 2D point, or a bearing), 3 (a
/// 3D point), 4 (a 3D line, or a 2D segment). Cross-checked against g2o's
/// vertex dimensions and GTSAM's variable dimensions.
pub const FIXED_SHAPES: [(usize, usize, usize); 29] = [
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
];

/// Does this tile shape have an unrolled kernel, or does it fall to nano-gemm?
/// See [`FIXED_SHAPES`].
pub fn has_fixed_kernel(wa: usize, we: usize, wb: usize) -> bool {
    FIXED_SHAPES.contains(&(wa, we, wb))
}

/// Counts calls that reached an unrolled kernel. Whether the dispatch fires is
/// invisible in the output -- the fallback computes the same thing -- so without
/// this a broken match arm would silently take the slower path and no test would
/// notice. Compiled out of every non-test build.
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

/// Counts eliminated blocks that took the uniform triangular path. Like
/// [`note_fixed_kernel`], the output cannot tell the two routes apart, so a
/// missing dispatch arm would silently cost speed and no test would notice.
/// Compiled out of every non-test build.
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
///
/// The three widths are packed into one integer (a nibble each) and matched
/// as a single value, so the compiler emits ONE switch. Matching the tuple
/// `(wa, we, wb)` directly compiles to a chain of comparisons instead, and
/// every shape then pays for the shapes listed ahead of it: measured on a
/// 7-arm chain, the reduce got 5.5% slower at (6, 4, 6) and 2.0% slower at
/// (6, 3, 6) purely from their position in the list. With the pack, growing
/// the list to 11 shapes costs the existing ones nothing (measured again).
/// It is not a micro-optimization -- it is what makes the list free to grow.
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
            _ => {}
        }
    };
}

/// The pair shapes, as `(observer, marginalized)` -- the third width is the
/// observer width again. What [`gemm_tri`] dispatches on, and the same list as
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
            _ => {}
        }
    };
}

/// The whole triangular pair loop of one eliminated block whose observers all
/// have the same width and storage orientation -- the ordinary case, since the
/// entity marginalized out is usually seen by one kind of entity (poses through
/// a landmark, cameras through a point). One dispatch covers every pair instead
/// of one per pair, and the tile shape is a compile-time constant throughout.
///
/// `pair_off`, `ca_off` and `panel_col` are this block's slices: the pair
/// targets in `s_vals` (b-major, a ascending, a <= b -- the symbolic emission
/// order), and per observer its coupling tile in `h_vals` and its column in
/// `panel`.
#[inline]
fn gemm_tri<T: SchurReal, I: Index, const W: usize, const WE: usize, const TRANS: bool>(
    s_vals: &mut [T],
    h_vals: &[T],
    panel: &[T],
    pair_off: &[I],
    ca_off: &[I],
    panel_col: &[I],
) {
    note_uniform_run();
    let mut pair = 0usize;
    for bi in 0..ca_off.len() {
        let pc = panel_col[bi].zx() * WE;
        let zb = &panel[pc..pc + WE * W];
        for &off in &ca_off[..=bi] {
            let d = pair_off[pair].zx();
            let a = off.zx();
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
            pair += 1;
        }
    }
}

/// [`gemm_tri`] for a width and orientation only known at run time. Returns
/// false when the shape has no unrolled kernel, leaving the caller to run the
/// pair-at-a-time loop.
fn gemm_tri_dispatch<T: SchurReal, I: Index>(
    w: usize,
    we: usize,
    trans: bool,
    s_vals: &mut [T],
    h_vals: &[T],
    panel: &[T],
    pair_off: &[I],
    ca_off: &[I],
    panel_col: &[I],
) -> bool {
    macro_rules! run {
        ($w:literal, $we:literal, $tr:expr) => {{
            if $tr {
                gemm_tri::<T, I, $w, $we, true>(s_vals, h_vals, panel, pair_off, ca_off, panel_col);
            } else {
                gemm_tri::<T, I, $w, $we, false>(s_vals, h_vals, panel, pair_off, ca_off, panel_col);
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

/// numeric Schur reduction: fills `s` (allocated via
/// [`SchurSymbolic::alloc_s`]) with `S = Hkk - Hke Hee^-1 Hek` and
/// `rhs_out` (length `s.nrows()`, the kept blocks compacted in order)
/// with `bk - Hke Hee^-1 be`, from pre-damped `h` and `rhs`. `h` must
/// have the exact symbolic structure `sym` was built from.
///
/// The work runs in three stages:
///
/// 1. **seed** -- S is zeroed, the kept-kept tiles of H are copied in
///    (S starts as `Hkk`), and the kept slices of `rhs` are copied to
///    `rhs_out`.
///
/// 2. **per eliminated block `e`** (independent of every other one,
///    because `Hee` is block-diagonal). Writing `C_a = H(a, e)` for
///    the coupling tile to observer `a` and `D_e = H(e, e)`:
///
///    - **factor**: dense Cholesky `D_e = L L^T` of the `w_e x w_e`
///      diagonal tile (read symmetric-from-upper into `ctx.dwork`).
///    - **panel**: the block's whole `Hee^-1`-application happens
///      here, once, as `Z = D_e^-1 [C^T | b_e]`. The coupling tiles'
///      transposes and the block's rhs slice are gathered into one
///      `w_e x (sum w_a + 1)` panel, and each panel column gets a
///      forward solve against `L` then a backward solve against
///      `L^T` -- that pair of triangular solves IS the `D_e^-1`
///      application; no inverse is ever formed. Afterwards panel
///      column group `b` holds `Z_b = D_e^-1 C_b^T` and the last
///      column holds `z = D_e^-1 b_e`.
///    - **gemm**: for every observer pair `a <= b`,
///      `S(a, b) -= C_a * Z_b` (i.e. `C_a D_e^-1 C_b^T`), the target
///      tile found via the precomputed `pair_dst` map -- consumed
///      sequentially, so the loop order must stay b-major exactly as
///      the symbolic pass emitted it.
///    - **rhs**: for every observer `a`, `rhs_out(a) -= C_a * z`.
///
/// 3. **finish** -- diagonal-pair GEMMs wrote full tiles; the
///    strictly-lower part of S's kept diagonal tiles is re-zeroed to
///    restore the upper-only-within-tile convention.
///
/// Per-stage wall time lands in [`SchurContext::timing`] when enabled.
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
    ctx.panel.resize(sym.max_panel, T::ZERO);
    let gather = ctx.timing.is_some();
    let mut t = SchurTiming::default();
    let mut sw = Stopwatch::new(gather);

    // stage 1: zero S and rhs_out. Hkk and the kept rhs are folded in at
    // stage 2.5, after the coupling terms have accumulated, so f32 sums the
    // many small contributions among themselves before meeting the large
    // diagonal once, instead of rounding each contribution against it.
    s.vals_mut().iter_mut().for_each(|v| *v = T::ZERO);
    rhs_out.iter_mut().for_each(|v| *v = T::ZERO);
    sw.lap(&mut t.seed);

    // stage 2: one eliminated block at a time
    for slot in 0..sym.elim_diag.len() {
        let d_blk = sym.elim_diag[slot].zx();
        let e = hs.blk_row(d_blk);
        let we = hs.col_span(e).len();

        // 2a factor: D_e = L L^T (diagonal tiles are stored upper-only
        // within the tile, so read symmetric from the upper triangle)
        let dwork = &mut ctx.dwork[..we * we];
        let dtile = &h.vals()[hs.val_range(d_blk)];
        for j in 0..we {
            for i in 0..=j {
                let v = dtile[i + j * we];
                dwork[i + j * we] = v;
                dwork[j + i * we] = v;
            }
        }
        if !llt_in_place(dwork, we) {
            return Err(SchurError::NotPositiveDefinite { block: e });
        }
        sw.lap(&mut t.factor);

        // 2b panel: gather [C^T | b_e], then Z = D_e^-1 [C^T | b_e]
        // via one forward + one backward triangular solve per column
        let orange = sym.elim_obs_ptr[slot].zx()..sym.elim_obs_ptr[slot + 1].zx();
        let col = sym.elim_ncols[slot].zx();
        for o in orange.clone() {
            let wo = sym.obs_w[o].zx();
            let ca = sym.obs_ca_off[o].zx();
            let tile = &h.vals()[ca..ca + wo * we];
            let ocol = sym.obs_panel_col[o].zx();
            let dst = &mut ctx.panel[ocol * we..(ocol + wo) * we];
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
        }
        ctx.panel[col * we..col * we + we].copy_from_slice(&rhs[hs.col_span(e)]);
        llt_solve_panel(dwork, &mut ctx.panel[..(col + 1) * we], we, col + 1);
        sw.lap(&mut t.panel);

        // 2c gemm: S(a, b) -= C_a * Z_b for every pair a <= b, b-major
        // (pair_off is consumed sequentially in the symbolic emission
        // order)
        let mut pair = sym.elim_pair_ptr[slot].zx();
        let s_vals = s.vals_mut();
        let uw = sym.elim_uw[slot].zx();
        let uniform = uw != 0
            && gemm_tri_dispatch(
                uw,
                we,
                sym.elim_utrans[slot],
                s_vals,
                h.vals(),
                &ctx.panel,
                &sym.pair_off[pair..pair + orange.len() * (orange.len() + 1) / 2],
                &sym.obs_ca_off[orange.clone()],
                &sym.obs_panel_col[orange.clone()],
            );
        if !uniform {
            for o_b in orange.clone() {
                let wb = sym.obs_w[o_b].zx();
                let bcol = sym.obs_panel_col[o_b].zx();
                let zb = &ctx.panel[bcol * we..(bcol + wb) * we];
                for o_a in orange.start..=o_b {
                    let wa = sym.obs_w[o_a].zx();
                    let ca = sym.obs_ca_off[o_a].zx();
                    let dst = sym.pair_off[pair].zx();
                    gemm_sub(
                        &mut s_vals[dst..dst + wa * wb],
                        &h.vals()[ca..ca + wa * we],
                        sym.obs_trans[o_a],
                        wa,
                        we,
                        zb,
                        wb,
                    );
                    pair += 1;
                }
            }
        }
        sw.lap(&mut t.gemm);

        // 2d rhs: b'_a -= C_a * z for every observer
        let z = &ctx.panel[col * we..col * we + we];
        for o_a in orange.clone() {
            let wa = sym.obs_w[o_a].zx();
            let ca = sym.obs_ca_off[o_a].zx();
            let out = sym.obs_kept_off[o_a].zx();
            gemm_sub(
                &mut rhs_out[out..out + wa],
                &h.vals()[ca..ca + wa * we],
                sym.obs_trans[o_a],
                wa,
                we,
                z,
                1,
            );
        }
        sw.lap(&mut t.rhs);
    }

    // stage 2.5: fold Hkk and the kept rhs on top of the accumulated pot.
    // The (large) diagonal blocks land once, after the (small) coupling
    // contributions have summed among themselves. Diagonal tiles are
    // upper-only in H, so only S's upper triangle is touched; stage 3 clears
    // the lower.
    for (hb, sb) in core::iter::zip(&sym.copy_src, &sym.copy_dst) {
        let src = hs.val_range(hb.zx());
        let dst = sym.s.val_range(sb.zx());
        let hsrc = &h.vals()[src];
        let sdst = &mut s.vals_mut()[dst];
        for (d, &v) in core::iter::zip(sdst.iter_mut(), hsrc) {
            *d = *d + v;
        }
    }
    for (k, &orig) in sym.kept.iter().enumerate() {
        let src = hs.col_span(orig.zx());
        let dst = sym.s.col_span(k);
        for (d, &v) in core::iter::zip(rhs_out[dst].iter_mut(), &rhs[src]) {
            *d = *d + v;
        }
    }
    sw.lap(&mut t.seed);

    // stage 3: diagonal-pair GEMMs wrote full tiles; restore the
    // upper-only-within-tile convention on S's kept diagonal
    for sb in &sym.s_diag {
        let range = sym.s.val_range(sb.zx());
        let w = sym.s.block_dims(sb.zx()).0;
        let tile = &mut s.vals_mut()[range];
        for j in 0..w {
            for i in j + 1..w {
                tile[i + j * w] = T::ZERO;
            }
        }
    }
    sw.lap(&mut t.finish);
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
/// and the recovered eliminated blocks. `D_e` is re-factored here --
/// caching the reduce-time factors in [`SchurContext`] is a deferred
/// optimization (the factor stage costs ~1% of a reduction).
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
    ctx.dwork.resize(sym.max_ew * sym.max_ew, T::ZERO);
    ctx.panel.resize(sym.max_panel.max(sym.max_ew), T::ZERO);

    // kept blocks scatter back to their original spans first: the
    // eliminated recovery below reads them out of x_full
    for (k, &orig) in sym.kept.iter().enumerate() {
        let dst = hs.col_span(orig.zx());
        let src = sym.s.col_span(k);
        x_full[dst].copy_from_slice(&x_kept[src]);
    }

    for slot in 0..sym.elim_diag.len() {
        let d_blk = sym.elim_diag[slot].zx();
        let e = hs.blk_row(d_blk);
        let we = hs.col_span(e).len();

        // t = b_e - sum_a C_a^T x_a
        let t = &mut ctx.panel[..we];
        t.copy_from_slice(&rhs[hs.col_span(e)]);
        for o in sym.elim_obs_ptr[slot].zx()..sym.elim_obs_ptr[slot + 1].zx() {
            let wa = sym.obs_w[o].zx();
            let ca = sym.obs_ca_off[o].zx();
            let xoff = sym.obs_orig_off[o].zx();
            gemm_sub(
                t,
                &h.vals()[ca..ca + wa * we],
                !sym.obs_trans[o],
                we,
                wa,
                &x_full[xoff..xoff + wa],
                1,
            );
        }

        // x_e = D_e^-1 t (same two triangular solves as the reduction)
        let dwork = &mut ctx.dwork[..we * we];
        let dtile = &h.vals()[hs.val_range(d_blk)];
        for j in 0..we {
            for i in 0..=j {
                let v = dtile[i + j * we];
                dwork[i + j * we] = v;
                dwork[j + i * we] = v;
            }
        }
        if !llt_in_place(dwork, we) {
            return Err(SchurError::NotPositiveDefinite { block: e });
        }
        llt_solve_panel(dwork, t, we, 1);
        x_full[hs.col_span(e)].copy_from_slice(t);
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
    /// fires, or a list that promises a kernel nobody wrote, would cost 2x in
    /// silence. Walk every shape up to 9x9x9 and demand the two agree, in
    /// both orientations. This is the only test that can see the property.
    #[test]
    fn the_shape_list_and_the_dispatch_agree() {
        let hits = || FIXED_KERNEL_HITS.with(|c| c.get());
        for wa in 1..=9usize {
            for we in 1..=9usize {
                for wb in 1..=9usize {
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

    /// Every pair shape must reach [`gemm_tri`], not just the pair-at-a-time
    /// loop: the two compute the same values, so nothing in the output would
    /// show a missing arm in `uniform_pair_shapes!`.
    #[test]
    fn every_pair_shape_reaches_the_uniform_run() {
        for &(wa, we, wb) in FIXED_SHAPES.iter() {
            if wb == 1 {
                continue; // a one-column update, not a pair shape
            }
            assert_eq!(wa, wb, "a pair shape has the observer width on both sides");
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
                assert_eq!(
                    UNIFORM_RUN_HITS.with(|c| c.get()),
                    1,
                    "({}, {}, {}) elim_first={} did not take the uniform run",
                    wa, we, wb, elim_first
                );
            }
        }
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
