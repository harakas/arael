//! [`GpuSchur`]: the Schur reduction of a damped system on the device, and
//! the solve around it.
//!
//! The reduction splits the parameter blocks into eliminated blocks (in
//! bundle adjustment the points) and kept blocks (the cameras). An observer
//! is a coupling tile `C_o` of H between an eliminated block `e` and a kept
//! block `k`. S, the reduced system, is H over the kept blocks less the
//! eliminated blocks' part. A solve from H's tile values and the gradient
//! `b`, all on the device:
//!
//! 1. each eliminated block's diagonal tile `D_e` is factored (Cholesky),
//!    and `z_e = D_e^-1 b_e`;
//! 2. per observer, `W_o = C_o D_e^-1`;
//! 3. each tile of S is H's tile less `W_oa C_ob^T` over the pairs of
//!    observers `(oa, ob)` of one eliminated block that touch it;
//! 4. the reduced right-hand side is `b_k` less `C_o z_e` over the kept
//!    block's observers;
//! 5. S is factored and solved for the kept blocks' step `x_k` (a
//!    [`GpuFactor`], or [`GpuCg`](crate::GpuCg)), and the solution may be
//!    refined against S (for an f32 factor);
//! 6. the eliminated blocks' step is `x_e = D_e^-1 (b_e - sum of C_o^T x_k)`
//!    over the block's observers.
//!
//! `D_e^-1` is applied by two triangular solves with `D_e`'s factor. H and
//! the gradient are in the solve's [`Values`]; S and its vectors in the
//! wide scalar ([`Wide`]), or in f32 with
//! [`GpuSchurParams::f32_reduction`]. Blocks are at most 64 columns wide.
//!
//! The refinement decides on the device whether its next step is due, so a
//! solve waits once: [`GpuSchur::solve`] reads the step, the pivot flags and
//! the refinement's state in one copy.

// D_e^-1 is never formed: multiplying by a rounded inverse loses digits in
// proportion to D_e's condition number, high for a point seen along nearly
// parallel rays.

use std::sync::Arc;

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::{SchurReal, SchurSymbolic};

use crate::plan::SparseIndex;
use crate::supernodal::GpuFactor;
use crate::{Arg, Device, GpuError, Launch, Layout, Scope, Section, Values, WIDE, Wide, define_u32s};

/// No kept-kept tile behind a tile of S.
const NONE: u32 = u32::MAX;

// The header of the reduction's plan buffer: where each table and array
// starts.
/// The eliminated blocks: a record of [`SE_REC`] words each (the `SE_*`
/// fields), and one more at the end for the end of the last block's
/// observers.
pub(crate) const SE: usize = 0;
/// The observers (an eliminated block's coupling to a kept block), in the
/// order of their eliminated blocks: a record of [`SO_REC`] words each (the
/// `SO_*` fields).
pub(crate) const SO: usize = 1;
pub(crate) const ST_VAL_OFF: usize = 2;
pub(crate) const ST_WR: usize = 3;
pub(crate) const ST_WC: usize = 4;
pub(crate) const ST_A: usize = 5;
pub(crate) const ST_B: usize = 6;
pub(crate) const ST_COPY_OFF: usize = 7;
/// Per tile of S: its first pair (two observers of one eliminated block:
/// a term of the tile), the next tile's the end; the pairs themselves are
/// written on the device (`k_schur_pairs`).
pub(crate) const ST_PAIR_PTR: usize = 8;
/// Per observer: its kept block.
pub(crate) const OBS_KEPT: usize = 9;
pub(crate) const SK_OBS_PTR: usize = 10;
pub(crate) const SK_OBS: usize = 11;
pub(crate) const SK_S_OFF: usize = 12;
pub(crate) const SK_ORIG_OFF: usize = 13;
pub(crate) const SK_W: usize = 14;
/// The blocks of `k_schur_tiles` ([`tile_blocks`]), added by
/// [`GpuSchur`]'s build.
pub(crate) const SB_ITEMS: usize = 15;
/// Per kept block: its first tile of S, the next block's the end.
pub(crate) const SK_T_PTR: usize = 16;
/// Per eliminated block: its first pair in `k_schur_pairs`' threads, the
/// next block's the end (m observers make m (m + 1) / 2 pairs).
pub(crate) const SE_PAIR_PTR: usize = 17;
pub(crate) const SHEADER: usize = 32;

// The words of a pair's expanded record, what `k_schur_tiles` needs of the
// pair's observers (oa of the tile's row block, ob of its column block),
// grouped by tile (ST_PAIR_PTR), a tile's pairs in no fixed order: written
// on the device once per structure (`k_schur_pairs`, from the observers'
// records) into a block of their own, so the tile kernel reads a pair in
// one step and the host neither builds nor uploads the pairs.
/// The value offset of W_oa in w.
pub(crate) const PX_W_OFF: usize = 0;
/// The value offset of ob's coupling tile in h.
pub(crate) const PX_C_OFF: usize = 1;
/// ob's sizes ([`so_dims`]).
pub(crate) const PX_DIMS: usize = 2;
pub(crate) const PX_REC: usize = 3;

// The words of a kept block's observer's expanded record, what
// `k_schur_rhs` needs of the observer and its eliminated block, one per
// entry of [`SK_OBS`] in its order: written on the device once per
// structure (`k_schur_kobs`) after the pairs' records, so the kernel reads
// an observer in one step, not through its record and its eliminated
// block's.
/// The observer's sizes ([`so_dims`]).
pub(crate) const KO_DIMS: usize = 0;
/// The value offset of its coupling tile in h.
pub(crate) const KO_C_OFF: usize = 1;
/// The value offset of its eliminated block's z.
pub(crate) const KO_Z_OFF: usize = 2;
pub(crate) const KO_REC: usize = 3;

// The words of a kept block's tile entry's expanded record, what
// `k_symv_sub` needs of a tile of S touching the block, one per entry of
// `kt` in its order: written on the device once per structure
// (`k_schur_kt`) after the kept blocks' observers' records, so the kernel
// reads a tile in one step, not through the tile arrays (`ST_*`) and the
// other block's offset (`SK_S_OFF`).
/// The tile's value offset in s.
pub(crate) const KT_VAL_OFF: usize = 0;
/// The tile's sizes and how the block uses it ([`kt_dims`]).
pub(crate) const KT_DIMS: usize = 1;
/// The other block's offset in the reduced vector: the tile's column
/// block's, or its row block's when transposed.
pub(crate) const KT_X_OFF: usize = 2;
pub(crate) const KT_REC: usize = 3;

// The words of an eliminated block's record ([`SE`]). A kernel that walks
// the blocks reads one table, not an array per field.
/// The value offset of its diagonal tile in h.
pub(crate) const SE_DIAG_OFF: usize = 0;
pub(crate) const SE_W: usize = 1;
/// Its scalar offset in the gradient and the step.
pub(crate) const SE_ORIG_OFF: usize = 2;
/// Its first observer; the next record's is the end of its observers.
pub(crate) const SE_OBS_PTR: usize = 3;
/// The value offset of its square in dfac.
pub(crate) const SE_DFAC_OFF: usize = 4;
/// Its value offset in z.
pub(crate) const SE_Z_OFF: usize = 5;
pub(crate) const SE_REC: usize = 6;

// The words of an observer's record ([`SO`]). Five words: `k_schur_backsub`
// and `k_schur_rhs` in f64 slow down with the record's width.
/// The value offset of W_o in w.
pub(crate) const SO_W_OFF: usize = 0;
/// Its sizes ([`so_dims`]).
pub(crate) const SO_DIMS: usize = 1;
/// The value offset of its coupling tile in h.
pub(crate) const SO_CA_OFF: usize = 2;
/// Its eliminated block (a record of [`SE`]).
pub(crate) const SO_E: usize = 3;
/// The scalar offset of its kept block in the step.
pub(crate) const SO_ORIG_OFF: usize = 4;
pub(crate) const SO_REC: usize = 5;

/// An observer's sizes in one word: the width of its kept block (bits 0 to
/// 7), of its eliminated block (bits 8 to 15), and `trans` (bit 16): whether
/// its coupling tile is stored (eliminated, kept). The kernels' `DIMS_WO`,
/// `DIMS_WE` and `DIMS_TRANS` take it apart.
///
/// ```ignore
/// assert_eq!(so_dims(9, 3, true), 0x10309);
/// ```
fn so_dims(wo: u32, we: u32, trans: bool) -> u32 {
    assert!(wo < 256 && we < 256, "a block of {} columns does not fit an observer's sizes word", wo.max(we));
    wo | we << 8 | (trans as u32) << 16
}

/// A kept block's tile entry's sizes word ([`KT_DIMS`]), as `k_schur_kt`
/// packs it: the tile's rows and columns (at most 64, [`round_cap`]),
/// whether the block is its column block (the tile then transposed gives
/// the block's rows), and whether it is a diagonal tile (stored
/// upper-only). The kernel's `KT_WR`, `KT_WC`, `KT_TRANS` and `KT_DIAG`
/// take it apart.
///
/// ```ignore
/// assert_eq!(kt_dims(9, 3, true, false), 0x10309);
/// ```
#[cfg(test)]
fn kt_dims(wr: u32, wc: u32, trans: bool, diag: bool) -> u32 {
    wr | wc << 8 | (trans as u32) << 16 | (diag as u32) << 17
}

/// The settings of a [`GpuSchur`] ([`GpuSchur::with_params`],
/// [`GpuSchur::whole_with_params`]): how the forming of S is cut over the
/// device, the scalar of the reduction, and the warps a kept block takes.
///
/// S is formed a tile at a time, each tile by a unit of threads of the
/// device's lockstep width ([`Arch::warp`], at least 32), which sums the
/// tile's pairs: the pairs of observers of one eliminated block, a term of
/// the tile each. When the kernel's shared memory would not fit the device
/// ([`Arch::shared_bytes`]), a default `tile_warps` is lowered first, then
/// a default `tile_groups`; a value set here that does not fit makes the
/// constructor return [`GpuError::Limit`].
///
/// ```
/// use arael_gpu::GpuSchurParams;
/// // one lane group per unit, the rest as the device has it
/// let params = GpuSchurParams { tile_groups: Some(1), ..Default::default() };
/// assert_eq!(params.split_pairs, 128);
/// ```
///
/// [`Arch::warp`]: crate::Arch::warp
/// [`Arch::shared_bytes`]: crate::Arch::shared_bytes
#[derive(Clone, Copy, Debug)]
pub struct GpuSchurParams {
    /// Units per block of the forming of S, at least 1. With several, a
    /// tile of more than [`split_pairs`](Self::split_pairs) pairs takes a
    /// whole block, split over its units, and the other tiles a unit each.
    /// `None` (the default) takes the device's value
    /// ([`Self::tile_warps_on`]).
    pub tile_warps: Option<u32>,
    /// The most pairs a tile may have and still take one unit of a block of
    /// several ([`tile_warps`](Self::tile_warps)); a tile with more takes a
    /// whole block. Default 128.
    pub split_pairs: u32,
    /// Lane groups per unit: 1, 2, 4 or 8. Each group sums different pairs
    /// of the tile, and the groups' sums are added at the tile's end. More
    /// groups leave fewer lanes idle on small tiles (a 6 x 6 tile on 32
    /// lanes keeps 36 of 64 lane-products busy with one group, 36 of 48
    /// with two) and take more registers and shared memory. `None` (the
    /// default) takes two when they issue more than 10% fewer products over
    /// the reduction's tiles than one, else one.
    pub tile_groups: Option<u32>,
    /// Whether the reduction computes in f32, as the host's f32 solve does:
    /// the eliminated blocks' factors, W, S, the reduced right-hand side,
    /// the kept blocks' solution and the eliminated blocks' step, from the
    /// inputs rounded to f32 (an f32 model's [`Values`] already are), all
    /// stored in f32 (the step in the solve's values). For a solve whose
    /// factor is f32 and that does not refine against S (a refined solve
    /// panics), or by conjugate gradients in f32. Default `false`: the
    /// reduction computes in the wide scalar.
    pub f32_reduction: bool,
    /// Warps (32 threads) a kept block takes in the kernels with a lane per
    /// row of the block: the reduced right-hand side (over the block's
    /// observers) and the refinement's residual (over its tiles of S). 1 to
    /// 32. `None` (the default) takes, per kernel, the power of two that
    /// gives each lane group about 8 of the average kept block's items, at
    /// most 32 for the right-hand side and 8 for the residual. Kept blocks
    /// wider than 32 columns take one warp.
    pub kept_warps: Option<u32>,
    /// Host threads of arael's worker pool ([`arael_faer::pool`]) that
    /// build the reduction's plan, at least 1. The plan does not depend on
    /// it. Default 1.
    pub threads: usize,
}

/// The most warps a kept block takes in a kernel whose lanes are its rows:
/// a block of 1024 threads.
pub(crate) const MAX_KEPT_WARPS: u32 = 32;

/// The warps (32 threads) a kept block takes in a kernel whose lanes are
/// its rows, from the items (observers, or tiles of S) the `blocks` kept
/// blocks have in all and their width rounded up (`kw_cap`, at most 32): a
/// power of two, 1 to `most`, that gives a group of lanes about 8 items. A
/// warp is `32 / kw_cap` groups of lanes, a lane a row of the block, and a
/// group takes an item at a time.
///
/// ```ignore
/// // 372 cameras of 9 parameters (two groups a warp), 147 tiles of S each
/// assert_eq!(kept_warps_for(54692, 372, 16, 8), 8);
/// ```
pub(crate) fn kept_warps_for(items: usize, blocks: usize, kw_cap: usize, most: u32) -> u32 {
    let groups = (PLANE as usize / kw_cap).max(1);
    let warps = items / (blocks.max(1) * 8 * groups);
    // the power of two at or under it
    (1u32 << warps.max(1).ilog2()).min(most)
}

impl GpuSchurParams {
    /// The [`tile_warps`](Self::tile_warps) used on the device `arch`: the
    /// value set, else 4 on a discrete GPU, 1 on an integrated GPU and the
    /// CPU device. [`GpuSchur::tile_warps`] gives the value a reduction
    /// took after the fit to the device's shared memory.
    pub fn tile_warps_on(&self, arch: &crate::Arch) -> u32 {
        self.tile_warps.unwrap_or(if arch.integrated { 1 } else { 4 })
    }
}

impl Default for GpuSchurParams {
    fn default() -> Self {
        GpuSchurParams { tile_warps: None, split_pairs: 128, tile_groups: None, f32_reduction: false, kept_warps: None, threads: 1 }
    }
}

// The refinement's state on the device: f64 words and u32 flags, read back
// with the step.
pub(crate) const ST_BNORM: usize = 0;
pub(crate) const ST_TOL: usize = 1;
pub(crate) const ST_RNORM: usize = 2;
/// A refinement step is due; the gate of the step's kernels.
pub(crate) const F_ACTIVE: usize = 0;
pub(crate) const F_STEPS: usize = 1;
/// The last step did not reduce the residual and is undone.
pub(crate) const F_REVERT: usize = 2;
/// An eliminated block's slot plus one when its diagonal is not definite.
pub(crate) const F_ELIM: usize = 3;
/// Nonzero when S's factorization met a pivot that is not positive.
pub(crate) const F_FACTOR: usize = 4;
pub(crate) const N_FLAGS: usize = 8;
/// Bytes from the state's start to the flags'.
pub(crate) const FLAGS_AT: usize = 64;
/// Bytes of a solve's status at [`GpuSchur::status_view`]: the
/// refinement's state and the flags, for [`GpuSchur::outcome`].
pub const STATUS_BYTES: usize = FLAGS_AT + 4 * N_FLAGS;

/// What the refinement of a solve took ([`GpuSchur::solve`]); all zero for
/// a solve without one.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefineStats {
    /// Refinement steps run, a step undone for not reducing the residual
    /// among them.
    pub steps: usize,
    /// The norm of the residual `b - S x` of the reduced system at the end.
    pub rnorm: f64,
    /// The norm of the reduced right-hand side `b`.
    pub bnorm: f64,
    /// Whether `rnorm` stayed above the tolerance; a norm that is not a
    /// number does.
    pub short: bool,
}

/// What a plan reads of a reduction: the eliminated blocks and their
/// observers (in [`SchurSymbolic`]'s layout), the kept blocks, S's
/// structure, the kept-kept tiles copied from H, the pair count.
struct Elimination<'a> {
    elim_diag: &'a [SparseIndex],
    elim_w: &'a [SparseIndex],
    elim_obs_ptr: &'a [SparseIndex],
    obs_w: &'a [SparseIndex],
    obs_ca_off: &'a [SparseIndex],
    obs_trans: &'a [bool],
    obs_orig_off: &'a [SparseIndex],
    obs_kept: &'a [SparseIndex],
    kept: std::borrow::Cow<'a, [SparseIndex]>,
    s: &'a SymbolicSparseBlockColMat<SparseIndex>,
    copy_src: std::borrow::Cow<'a, [SparseIndex]>,
    copy_dst: std::borrow::Cow<'a, [SparseIndex]>,
    pairs: usize,
    /// Per tile of S, its pairs ([`SchurSymbolic::tile_pairs`]).
    tile_pairs: std::borrow::Cow<'a, [u32]>,
}

/// Observers a thread of a reduction's plan takes at least.
const PLAN_MIN_OBSERVERS: usize = 1 << 15;

/// A range of eliminated blocks of a reduction's plan: its sums, and the
/// offsets it starts at (the earlier ranges' sums).
#[derive(Default)]
struct PlanRange {
    slots: core::ops::Range<usize>,
    sum_dfac: u32,
    sum_z: u32,
    sum_w: u32,
    sum_pairs: u32,
    ew_max: usize,
    kw_max: usize,
    dfac: u32,
    z: u32,
    w: u32,
    pairs: u32,
}

/// A range's slices of the plan's words: the eliminated blocks' and the
/// observers' records, the observers' kept blocks, and per eliminated
/// block its block of H and its pairs' end.
struct PlanSlices<'a> {
    se: &'a mut [u32],
    so: &'a mut [u32],
    obs_kept: &'a mut [u32],
    elim_blk: &'a mut [u32],
    e_pair_ptr: &'a mut [u32],
}

impl PlanRange {
    /// The range's sums: the factors' and the solves' lengths, W's
    /// length, the pairs, the widest eliminated and kept blocks.
    fn sum(&mut self, sym: &Elimination) {
        let p = sym.elim_obs_ptr;
        (self.ew_max, self.kw_max) = (1, 1);
        for slot in self.slots.clone() {
            let we = sym.elim_w[slot];
            self.ew_max = self.ew_max.max(we as usize);
            self.sum_dfac += we * we;
            self.sum_z += we;
            for o in p[slot] as usize..p[slot + 1] as usize {
                let wo = sym.obs_w[o];
                self.sum_w += wo * we;
                self.kw_max = self.kw_max.max(wo as usize);
            }
            let m = p[slot + 1] - p[slot];
            self.sum_pairs += m * (m + 1) / 2;
        }
    }

    /// The range's records from its first offsets.
    fn fill(&self, sym: &Elimination, hsym: &SymbolicSparseBlockColMat<SparseIndex>, sl: &mut PlanSlices<'_>) {
        let (p, hpart) = (sym.elim_obs_ptr, hsym.col_part());
        let (mut dfac, mut z, mut w, mut pairs) = (self.dfac, self.z, self.w, self.pairs);
        let (s0, o0) = (self.slots.start, p[self.slots.start] as usize);
        for slot in self.slots.clone() {
            let ls = slot - s0;
            let r = &mut sl.se[SE_REC * ls..][..SE_REC];
            let dblk = sym.elim_diag[slot] as usize;
            let e = hsym.blk_row(dblk);
            let we = sym.elim_w[slot];
            (r[SE_DIAG_OFF], r[SE_W], r[SE_ORIG_OFF], r[SE_OBS_PTR], r[SE_DFAC_OFF], r[SE_Z_OFF]) = (hsym.val_range(dblk).start as u32, we, hpart[e], p[slot], dfac, z);
            sl.elim_blk[ls] = e as u32;
            dfac += we * we;
            z += we;
            for o in p[slot] as usize..p[slot + 1] as usize {
                let wo = sym.obs_w[o];
                let r = &mut sl.so[SO_REC * (o - o0)..][..SO_REC];
                (r[SO_W_OFF], r[SO_DIMS], r[SO_CA_OFF], r[SO_E], r[SO_ORIG_OFF]) = (w, so_dims(wo, we, sym.obs_trans[o]), sym.obs_ca_off[o], slot as u32, sym.obs_orig_off[o]);
                w += wo * we;
                sl.obs_kept[o - o0] = sym.obs_kept[o];
            }
            let m = p[slot + 1] - p[slot];
            pairs += m * (m + 1) / 2;
            sl.e_pair_ptr[ls] = pairs;
        }
    }
}

/// Where the status starts in a solve's output, after the step's `n`
/// values of `bytes` bytes.
pub(crate) fn status_at(n: usize, bytes: usize) -> usize {
    (n * bytes).div_ceil(256) * 256
}

/// The outcome of a solve with refinement `refine` from its status bytes,
/// `elim_blk` giving the block of each eliminated slot.
pub(crate) fn decode_status(status: &[u8], elim_blk: &[u32], refine: Option<(usize, f64)>) -> Result<RefineStats, GpuSchurError> {
    let word = |at: usize| bytemuck::pod_read_unaligned::<u32>(&status[at..at + 4]);
    let real = |at: usize| bytemuck::pod_read_unaligned::<f64>(&status[at..at + 8]);
    let flag = |f: usize| word(FLAGS_AT + 4 * f);
    let elim = flag(F_ELIM);
    if elim != 0 {
        return Err(GpuSchurError::Eliminated { block: elim_blk[elim as usize - 1] as usize });
    }
    if flag(F_FACTOR) != 0 {
        return Err(GpuSchurError::Reduced);
    }
    let mut stats = RefineStats::default();
    if refine.is_some_and(|r| r.0 > 0) {
        let st = |i: usize| real(8 * i);
        // a norm that is not a number is short of any tolerance
        stats = RefineStats { steps: flag(F_STEPS) as usize, rnorm: st(ST_RNORM), bnorm: st(ST_BNORM), short: !(st(ST_RNORM) <= st(ST_TOL)) };
    }
    Ok(stats)
}

fn round_cap(w: usize) -> usize {
    [4usize, 8, 16, 32, 64].into_iter().find(|&c| c >= w).unwrap_or_else(|| panic!("a block of {w} columns is wider than the reduction's kernels take (64)"))
}

/// The host side of a reduction: the plan words the kernels read (the
/// header's offsets relative to word 0) and the sizes of the reduction's
/// arrays.
pub(crate) struct SchurPlan {
    pub words: Vec<u32>,
    pub ne: u32,
    pub no: u32,
    pub nt: u32,
    pub nk: u32,
    /// The Hessian's scalar dimension, and the reduced system's.
    pub n: usize,
    pub nks: usize,
    /// Values of S, of the eliminated blocks' Cholesky factors, of z and of
    /// W.
    pub s_len: usize,
    pub dfac_len: usize,
    pub z_len: usize,
    pub w_len: usize,
    /// The widest eliminated block, and it and the widest kept block
    /// rounded up as the kernels take them.
    pub ew_max: usize,
    pub ew_cap: usize,
    pub kw_cap: usize,
    /// The widest kept block: a tile of S has at most kw_max^2 entries.
    pub kw_max: usize,
    /// Per eliminated slot, its block id (for the error).
    pub elim_blk: Vec<u32>,
    /// The pairs.
    pub np: usize,
}

impl SchurPlan {
    /// The plan of `sym`, the reduction of a matrix with the block structure `hsym`.
    /// Built on up to `threads` threads of the worker pool
    /// ([`arael_faer::pool`]), each taking at least `min_obs` observers;
    /// the plan does not depend on either.
    pub fn reduction(sym: &SchurSymbolic<SparseIndex>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, threads: usize, min_obs: usize) -> Self {
        let (src, dst) = sym.copy_tiles();
        let elim = Elimination {
            elim_diag: sym.elim_diag(),
            elim_w: sym.elim_w(),
            elim_obs_ptr: sym.elim_obs_ptr(),
            obs_w: sym.obs_w(),
            obs_ca_off: sym.obs_ca_off(),
            obs_trans: sym.obs_trans(),
            obs_orig_off: sym.obs_orig_off(),
            obs_kept: sym.obs_kept(),
            kept: std::borrow::Cow::Borrowed(&sym.kept),
            s: &sym.s,
            copy_src: std::borrow::Cow::Borrowed(src),
            copy_dst: std::borrow::Cow::Borrowed(dst),
            pairs: sym.pair_count(),
            tile_pairs: std::borrow::Cow::Borrowed(sym.tile_pairs()),
        };
        Self::build(&elim, hsym, threads, min_obs)
    }

    /// The plan with nothing eliminated: S is H.
    pub fn whole(hsym: &SymbolicSparseBlockColMat<SparseIndex>) -> Self {
        let tiles: Vec<SparseIndex> = (0..hsym.nblocks() as SparseIndex).collect();
        Self::build(
            &Elimination {
                elim_diag: &[],
                elim_w: &[],
                elim_obs_ptr: &[0],
                obs_w: &[],
                obs_ca_off: &[],
                obs_trans: &[],
                obs_orig_off: &[],
                obs_kept: &[],
                kept: std::borrow::Cow::Owned((0..hsym.nblk_cols() as SparseIndex).collect()),
                s: hsym,
                copy_src: std::borrow::Cow::Borrowed(&tiles),
                copy_dst: std::borrow::Cow::Borrowed(&tiles),
                pairs: 0,
                tile_pairs: std::borrow::Cow::Owned(vec![0; hsym.nblocks()]),
            },
            hsym,
            1,
            PLAN_MIN_OBSERVERS,
        )
    }

    fn build(sym: &Elimination, hsym: &SymbolicSparseBlockColMat<SparseIndex>, threads: usize, min_obs: usize) -> Self {
        let ne = sym.elim_diag.len();
        let no = sym.obs_w.len();
        let nk = sym.kept.len();
        let n = hsym.ncols();
        let nks = sym.s.nrows();
        let hpart = hsym.col_part();
        let ssym = sym.s;
        let spart = ssym.col_part();
        let nt = ssym.nblocks();
        let e_obs_ptr = sym.elim_obs_ptr;
        // The plan's words: the header, then the eliminated blocks' and the
        // observers' records, then a word per tile in six arrays and the
        // tiles' pair pointers, the observers' kept blocks and the kept
        // blocks' observers, the kept blocks' tables and the eliminated
        // blocks' pair pointers. Laid out once here, every region written
        // in place.
        let (at_se, at_so) = (SHEADER, SHEADER + SE_REC * (ne + 1));
        let packed: [(usize, usize); 15] = [
            (ST_VAL_OFF, nt),
            (ST_WR, nt),
            (ST_WC, nt),
            (ST_A, nt),
            (ST_B, nt),
            (ST_COPY_OFF, nt),
            (ST_PAIR_PTR, nt + 1),
            (OBS_KEPT, no),
            (SK_OBS_PTR, nk + 1),
            (SK_OBS, no),
            (SK_S_OFF, nk),
            (SK_ORIG_OFF, nk),
            (SK_W, nk),
            (SK_T_PTR, nk + 1),
            (SE_PAIR_PTR, ne + 1),
        ];
        let words = at_so + SO_REC * no + packed.iter().map(|p| p.1).sum::<usize>();
        let mut buf: Vec<u32> = vec![0; words];
        (buf[SE], buf[SO]) = (at_se as u32, at_so as u32);
        let mut at = at_so + SO_REC * no;
        for &(slot, len) in &packed {
            buf[slot] = at as u32;
            at += len;
        }
        let rest = &mut buf[at_se..];
        let (se, rest) = rest.split_at_mut(SE_REC * (ne + 1));
        let (so, rest) = rest.split_at_mut(SO_REC * no);
        let (t_val, rest) = rest.split_at_mut(nt);
        let (t_wr, rest) = rest.split_at_mut(nt);
        let (t_wc, rest) = rest.split_at_mut(nt);
        let (t_a, rest) = rest.split_at_mut(nt);
        let (t_b, rest) = rest.split_at_mut(nt);
        let (t_copy, rest) = rest.split_at_mut(nt);
        let (t_pair_ptr, rest) = rest.split_at_mut(nt + 1);
        let (obs_kept, rest) = rest.split_at_mut(no);
        let (k_obs_ptr, rest) = rest.split_at_mut(nk + 1);
        let (k_obs, rest) = rest.split_at_mut(no);
        let (k_s_off, rest) = rest.split_at_mut(nk);
        let (k_orig, rest) = rest.split_at_mut(nk);
        let (k_w, rest) = rest.split_at_mut(nk);
        let (k_t_ptr, e_pair_ptr) = rest.split_at_mut(nk + 1);

        // The eliminated blocks by ranges of about equal observers, on up
        // to `threads` threads: first each range's sums, from which every
        // range's first offsets follow, then each range's records.
        let parts = threads.min(no / min_obs.max(1)).clamp(1, ne.max(1));
        let mut cuts = vec![0usize];
        for t in 1..parts {
            let at = e_obs_ptr.partition_point(|&o| (o as usize) < no * t / parts).min(ne);
            cuts.push(at.max(cuts[t - 1]));
        }
        cuts.push(ne);
        let mut ranges: Vec<PlanRange> = (0..parts).map(|t| PlanRange { slots: cuts[t]..cuts[t + 1], ..Default::default() }).collect();
        let sum = |_: usize, r: &mut PlanRange| r.sum(sym);
        let over = |ranges: &mut Vec<PlanRange>, f: &(dyn Fn(usize, &mut PlanRange) + Sync)| {
            if let [one] = &mut ranges[..] {
                f(0, one);
            } else {
                arael_faer::pool::run_over(ranges, f);
            }
        };
        over(&mut ranges, &sum);
        let (mut dfac_len, mut z_len, mut w_len, mut pairs) = (0u32, 0u32, 0u32, 0u32);
        let (mut ew_max, mut kw_max) = (1usize, 1usize);
        for r in ranges.iter_mut() {
            (r.dfac, r.z, r.w, r.pairs) = (dfac_len, z_len, w_len, pairs);
            dfac_len += r.sum_dfac;
            z_len += r.sum_z;
            w_len += r.sum_w;
            pairs += r.sum_pairs;
            ew_max = ew_max.max(r.ew_max);
            kw_max = kw_max.max(r.kw_max);
        }
        let mut elim_blk = vec![0u32; ne];
        {
            // each range's slices of the records, the observers' kept
            // blocks, the eliminated blocks and their pair pointers
            let obs_cut = |c: usize| e_obs_ptr[c] as usize;
            let mut se = arael_faer::cut_slices(&mut se[..SE_REC * ne], &cuts.iter().map(|&c| SE_REC * c).collect::<Vec<_>>()).into_iter();
            let mut so = arael_faer::cut_slices(so, &cuts.iter().map(|&c| SO_REC * obs_cut(c)).collect::<Vec<_>>()).into_iter();
            let mut ok = arael_faer::cut_slices(obs_kept, &cuts.iter().map(|&c| obs_cut(c)).collect::<Vec<_>>()).into_iter();
            let mut eb = arael_faer::cut_slices(&mut elim_blk, &cuts).into_iter();
            let mut ep = arael_faer::cut_slices(&mut e_pair_ptr[1..], &cuts).into_iter();
            let mut fills: Vec<(&PlanRange, PlanSlices<'_>)> = ranges
                .iter()
                .map(|r| (r, PlanSlices { se: se.next().unwrap(), so: so.next().unwrap(), obs_kept: ok.next().unwrap(), elim_blk: eb.next().unwrap(), e_pair_ptr: ep.next().unwrap() }))
                .collect();
            let fill = |_: usize, (r, sl): &mut (&PlanRange, PlanSlices<'_>)| r.fill(sym, hsym, sl);
            if let [one] = &mut fills[..] {
                fill(0, one);
            } else {
                arael_faer::pool::run_over(&mut fills, fill);
            }
        }
        // the end of the last eliminated block's observers
        se[SE_REC * ne + SE_OBS_PTR] = e_obs_ptr[ne];

        // the tiles of S, in storage order, with their kept-kept source
        t_copy.fill(NONE);
        for (&src, &dst) in sym.copy_src.iter().zip(sym.copy_dst.iter()) {
            t_copy[dst as usize] = hsym.val_range(src as usize).start as u32;
        }
        // a tile's blocks' offsets in the reduced vector are the kept blocks' (SK_S_OFF of ST_A and ST_B)
        for bcol in 0..ssym.nblk_cols() {
            for t in ssym.col_range(bcol) {
                let (wr, wc) = ssym.block_dims(t);
                kw_max = kw_max.max(wr).max(wc);
                (t_val[t], t_wr[t], t_wc[t], t_a[t], t_b[t]) = (ssym.val_range(t).start as u32, wr as u32, wc as u32, ssym.blk_row(t) as u32, bcol as u32);
            }
        }
        // the kept blocks' observer lists, by the ranges: each counts its
        // observers per kept block, then places them after the earlier
        // ranges' of the block
        let kept_of_obs = sym.obs_kept;
        let obs_range = |t: usize| e_obs_ptr[cuts[t]] as usize..e_obs_ptr[cuts[t + 1]] as usize;
        let mut kat: Vec<Vec<u32>> = vec![Vec::new(); parts];
        let count = |t: usize, at: &mut Vec<u32>| {
            at.resize(nk, 0);
            for &k in &kept_of_obs[obs_range(t)] {
                at[k as usize] += 1;
            }
        };
        if let [one] = &mut kat[..] {
            count(0, one);
        } else {
            arael_faer::pool::run_over(&mut kat, count);
        }
        for k in 0..nk {
            let mut m = 0u32;
            for at in kat.iter_mut() {
                let c = at[k];
                at[k] = k_obs_ptr[k] + m;
                m += c;
            }
            k_obs_ptr[k + 1] = k_obs_ptr[k] + m;
        }
        {
            let out = arael_faer::pool::SharedMut(k_obs.as_mut_ptr());
            let place = |t: usize, at: &mut Vec<u32>| {
                for o in obs_range(t) {
                    let k = kept_of_obs[o] as usize;
                    // A (range, kept block) pair owns the entries from its
                    // offset above to the next range's: no two writes
                    // meet, and every write is below no.
                    unsafe { *out.ptr().add(at[k] as usize) = o as u32 };
                    at[k] += 1;
                }
            };
            if let [one] = &mut kat[..] {
                place(0, one);
            } else {
                arael_faer::pool::run_over(&mut kat, place);
            }
        }
        for k in 0..nk {
            (k_s_off[k], k_orig[k], k_w[k], k_t_ptr[k]) = (spart[k], hpart[sym.kept[k] as usize], spart[k + 1] - spart[k], ssym.col_range(k).start as u32);
        }
        k_t_ptr[nk] = nt as u32;
        // each tile's first pair, from the symbolic's count of each tile's
        // pairs (every (oa, ob) of one eliminated block with kept(oa) <=
        // kept(ob)); the device writes the pairs (`k_schur_pairs`)
        let np = sym.pairs;
        assert_eq!(sym.tile_pairs.len(), nt, "a pair count per tile of S");
        for t in 0..nt {
            t_pair_ptr[t + 1] = t_pair_ptr[t] + sym.tile_pairs[t];
        }
        assert_eq!(t_pair_ptr[nt] as usize, np, "the reduction counted its pairs");
        let ew_cap = round_cap(ew_max);
        let kw_cap = round_cap(kw_max);
        SchurPlan {
            words: buf,
            ne: ne as u32,
            no: no as u32,
            nt: nt as u32,
            nk: nk as u32,
            n,
            nks,
            s_len: ssym.val_count(),
            dfac_len: dfac_len as usize,
            z_len: z_len as usize,
            w_len: w_len as usize,
            ew_max,
            ew_cap,
            kw_cap,
            kw_max,
            elim_blk,
            np,
        }
    }
}

/// What the kernels over a reduction's plan share: the plan's header, the
/// reduction's scalar, the records' sizes words. Compiled before
/// [`SOURCE`], and before the conjugate gradients' kernels.
pub(crate) const SHARED: &str = include_str!("../kernels/schur.h");
/// The kernels, compiled per block-width class.
const SOURCE: &str = include_str!("../kernels/schur.cpp");

/// Threads per block of the kernels over entries and blocks.
pub(crate) const THREADS: u32 = 256;
/// Threads per block of the kernels over the tiles of S and the kept blocks.
pub(crate) const PLANE: u32 = 32;

/// The next reduction's identity ([`GpuSchur::id`]).
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Why a [`GpuSchur`] solve failed.
#[derive(Clone, Debug)]
pub enum GpuSchurError {
    /// The diagonal tile of the eliminated block `block` is not positive
    /// definite; the caller raises the damping.
    Eliminated { block: usize },
    /// The reduced system is not positive definite; the caller raises the
    /// damping.
    Reduced,
    /// A device operation failed.
    Device(GpuError),
}

impl From<GpuError> for GpuSchurError {
    fn from(e: GpuError) -> Self {
        GpuSchurError::Device(e)
    }
}

impl core::fmt::Display for GpuSchurError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpuSchurError::Eliminated { block } => write!(f, "the diagonal tile of eliminated block {block} is not positive definite"),
            GpuSchurError::Reduced => write!(f, "the reduced system is not positive definite"),
            GpuSchurError::Device(e) => e.fmt(f),
        }
    }
}

impl std::error::Error for GpuSchurError {}

/// What a captured attempt depends on beyond the reduction's own blocks:
/// the factor whose kernels it recorded and the refinement's launches and
/// tolerance.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct GraphKey {
    factor: u64,
    max_steps: usize,
    tol_bits: u64,
}

struct Kernels<K> {
    zero_u32: K,
    elim: K,
    w: K,
    tiles: K,
    rhs: K,
    scatter_kept: K,
    backsub: K,
    symv_sub: K,
    copy: K,
    refine_start: K,
    refine_apply: K,
    refine_judge: K,
    refine_revert: K,
    pairs: K,
    kobs: K,
    kt: K,
}

/// Byte offsets of the work block's sections.
struct Sections {
    /// The Hessian's tile values and the gradient, in the route's values.
    h: usize,
    rhs: usize,
    dfac: usize,
    z: usize,
    w: usize,
    s: usize,
    rhs_kept: usize,
    x_kept: usize,
    x_prev: usize,
    r: usize,
    /// What a solve reads back, in one copy: the full step (`n` values),
    /// then at `status_at` from here the refinement's state and the flags.
    out: usize,
    bytes: usize,
}

/// The reduced system of a [`GpuSchur`] on its device, for a solver of it
/// that is not a factor: where S, the right-hand side and the solution
/// are, and the kept blocks' records. Offsets are bytes into their block.
pub(crate) struct ReducedSystem<'a, D: Device> {
    /// The reduction's kernels' #defines: the plan's and the records'
    /// words, the block-width caps, the scalars.
    pub defines: &'a str,
    /// The plan's words, and its header as the kernels take it.
    pub plan: &'a D::Block,
    pub header: &'a [u32; SHEADER],
    /// In `plan`, per kept block: where its tile entries start.
    pub kt_ptr: usize,
    /// The expanded records, the kept blocks' tile entries ([`KT_REC`]
    /// words each) from `kt` on.
    pub expanded: &'a D::Block,
    pub kt: usize,
    /// The work block: S's tile values, the reduced right-hand side, the
    /// kept blocks' solution (what the back-substitution reads) and a
    /// residual vector, all in the reduction's scalar; the solve's flags
    /// ([`N_FLAGS`] words).
    pub work: &'a D::Block,
    pub s: usize,
    pub b: usize,
    pub x: usize,
    pub r: usize,
    pub flags: usize,
    /// Kept blocks, the reduced system's scalar dimension, the widest kept
    /// block's columns and them rounded up to the kernels' cap (`KW`), and
    /// the kept blocks' tile entries in all.
    pub nk: u32,
    pub nks: usize,
    pub kw_max: usize,
    pub kw_cap: usize,
    pub kt_entries: usize,
}

/// The Schur reduction of one sparsity structure on a [`Device`], and the
/// damped solves around it ([module docs](self)). Build it once per
/// structure, then call [`solve`](Self::solve) once per damped attempt:
/// from H's tile values and the gradient on the host to the step, with one
/// wait. Every device buffer is made in [`new`](Self::new).
///
/// The parts of a solve can also be queued one by one, without a wait, for
/// a caller whose H and gradient are already on the device (written at
/// [`h_target`](Self::h_target) and [`rhs_target`](Self::rhs_target)) or
/// that solves S another way: [`launch_reduce`](Self::launch_reduce), a
/// solve of S ([`launch_factored`](Self::launch_factored), or
/// [`GpuCg`](crate::GpuCg)), [`launch_backsub`](Self::launch_backsub).
/// The step is then at [`step_view`](Self::step_view) and the status at
/// [`status_view`](Self::status_view), to read together and hand to
/// [`outcome`](Self::outcome).
///
/// ```ignore
/// use arael_gpu::{GpuFactor, GpuPrecision, GpuSchur, GpuSchurError, Wide, schur::STATUS_BYTES};
///
/// let mut schur = GpuSchur::new(&dev, &schur_sym, h.symbolic(), Wide::F64)?;
/// let mut factor = GpuFactor::new(&dev, GpuPrecision::F32, &sn_of_s, &schur_sym.s, &Default::default(), Wide::F64)?;
/// let refine = Some((4, 1e-10));
/// match schur.solve(&mut factor, &hvals, &rhs, refine, &mut delta) {
///     Ok(stats) => {}                                                          // delta holds the step
///     Err(GpuSchurError::Eliminated { .. } | GpuSchurError::Reduced) => {}     // raise the damping
///     Err(GpuSchurError::Device(e)) => return Err(e.into()),
/// }
///
/// // the same by parts
/// schur.upload(&hvals, &rhs)?;
/// schur.launch_reduce()?;
/// schur.launch_factored(&factor, refine)?;
/// schur.launch_backsub()?;
/// let (mut step, mut status) = (vec![0u8; schur.values().bytes() * schur.dim()], [0u8; STATUS_BYTES]);
/// let ((sb, sa), (tb, ta)) = (schur.step_view(), schur.status_view());
/// dev.read_many(&mut [(sb, sa, &mut step[..]), (tb, ta, &mut status[..])])?;  // the one wait
/// let stats = schur.outcome(&status, refine)?;
/// ```
pub struct GpuSchur<D: Device> {
    /// The last attempt's launches as a graph, with what it was captured
    /// against; first, so it is dropped before the blocks it records.
    graph: Option<(GraphKey, D::Graph)>,
    graphs: bool,
    replayed: bool,
    dev: Arc<D>,
    /// [`id`](Self::id).
    id: u64,
    /// The kernels' #defines, for another unit over the same plan (the
    /// conjugate gradients').
    defines: String,
    /// The widest kept block's columns, and them rounded up to the
    /// kernels' cap (`KW`).
    kw_max: usize,
    kw_cap: usize,
    /// The kept blocks' tile entries: a tile of S once for its row block
    /// and, off the diagonal, once for its column block.
    kt_entries: usize,
    /// Threads a block of the reduced right-hand side's kernel and of the
    /// residual's.
    rhs_threads: u32,
    symv_threads: u32,
    values: Values,
    wide: Wide,
    /// [`GpuSchurParams::f32_reduction`].
    f32_reduction: bool,
    k: Kernels<D::Kernel>,
    /// The plan words (at 0, as the plan's header offsets require), then a
    /// word holding 1, at `on`: the gate of kernels that always run; then
    /// per kept block the tiles of S touching it, `kt_ptr` (byte offset)
    /// and their list, for the residual.
    ints: D::Block,
    /// The plan's header, as the kernels take it: a launch parameter.
    header: [u32; SHEADER],
    /// The pairs' expanded records ([`PX_REC`] words each), for
    /// `k_schur_tiles`; from byte `kobs_at` on the kept blocks'
    /// observers' ([`KO_REC`] words each), for `k_schur_rhs`; from byte
    /// `kt_at` on the kept blocks' tile entries' ([`KT_REC`] words each),
    /// for `k_symv_sub`.
    expanded: D::Block,
    kobs_at: usize,
    kt_at: usize,
    on: usize,
    kt_ptr: usize,
    work: D::Block,
    sec: Sections,
    status_at: usize,
    ne: u32,
    no: u32,
    /// Threads per unit of `k_schur_tiles`, its units per block, its lane
    /// groups per unit, its blocks and its shared memory in bytes.
    lanes: u32,
    tile_warps: u32,
    tile_groups: u32,
    tile_blocks: u32,
    tile_shared: u32,
    /// Threads per observer in `k_schur_w`.
    w_rows: u32,
    nk: u32,
    n: usize,
    nks: usize,
    nh: usize,
    s_len: usize,
    elim_blk: Vec<u32>,
    /// The stamps [`solve`](Self::solve) makes ([`stamp_solves`](Self::stamp_solves)).
    stamps: Option<(usize, usize)>,
    /// The inputs encoded in the route's values.
    conv: Vec<u8>,
    /// The step and the status as a solve reads them.
    host_out: Vec<u8>,
    step: Vec<f64>,
}

impl<D: Device> GpuSchur<D> {
    /// The reduction on `dev` of a matrix of block structure `hsym` (H's),
    /// as arael-faer's symbolic Schur analysis `sym` of it splits the
    /// blocks, for a solve of `values` (a [`Wide`] for an f64 model): the
    /// plan uploaded, the device memory allocated, the kernels compiled.
    ///
    /// # Errors
    ///
    /// [`GpuError::Unavailable`] for [`Wide::F64`] on a device without f64;
    /// [`GpuError::Limit`] when the reduction's records do not fit 32-bit
    /// word offsets, or the forming of S needs more shared memory than the
    /// device has; a compile or driver error.
    ///
    /// # Panics
    ///
    /// When a block is wider than 64 columns.
    pub fn new(dev: &Arc<D>, sym: &SchurSymbolic<SparseIndex>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, values: impl Into<Values>) -> Result<Self, GpuError> {
        Self::with_params(dev, sym, hsym, values, &GpuSchurParams::default())
    }

    /// [`new`](Self::new) with the settings `params`. A `tile_warps` or
    /// `tile_groups` set that does not fit the device's shared memory is
    /// [`GpuError::Limit`].
    ///
    /// ```ignore
    /// let params = GpuSchurParams { tile_groups: Some(1), ..Default::default() };
    /// let g = GpuSchur::with_params(&dev, &sym, h.symbolic(), Wide::F64, &params)?;
    /// ```
    ///
    /// # Panics
    ///
    /// As [`new`](Self::new), and for `params` out of range: `tile_warps`
    /// 0, `tile_groups` not 1, 2, 4 or 8, `kept_warps` not 1 to 32.
    pub fn with_params(dev: &Arc<D>, sym: &SchurSymbolic<SparseIndex>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, values: impl Into<Values>, params: &GpuSchurParams) -> Result<Self, GpuError> {
        let plan = SchurPlan::reduction(sym, hsym, params.threads, PLAN_MIN_OBSERVERS);
        Self::build(dev, plan, hsym.val_count(), values.into(), params)
    }

    /// [`new`](Self::new) with nothing eliminated: S is H, and a solve is
    /// the whole system's. Errors and panics as [`new`](Self::new).
    pub fn whole(dev: &Arc<D>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, values: impl Into<Values>) -> Result<Self, GpuError> {
        Self::whole_with_params(dev, hsym, values, &GpuSchurParams::default())
    }

    /// [`whole`](Self::whole) with the settings `params`, as
    /// [`with_params`](Self::with_params) takes them: with
    /// [`GpuSchurParams::f32_reduction`], S (H) and the solve's vectors in
    /// f32, as the host's f32 solve keeps them.
    ///
    /// ```ignore
    /// let params = GpuSchurParams { f32_reduction: true, ..Default::default() };
    /// let g = GpuSchur::whole_with_params(&dev, h.symbolic(), Values::f32(Wide::Df32), &params)?;
    /// ```
    pub fn whole_with_params(dev: &Arc<D>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, values: impl Into<Values>, params: &GpuSchurParams) -> Result<Self, GpuError> {
        Self::build(dev, SchurPlan::whole(hsym), hsym.val_count(), values.into(), params)
    }

    fn build(dev: &Arc<D>, p: SchurPlan, nh: usize, values: Values, params: &GpuSchurParams) -> Result<Self, GpuError> {
        crate::check_f64(dev.arch(), values.wide() == Wide::F64, "Wide::F64")?;
        Self::build_lanes(dev, p, nh, values, dev.arch().warp.max(PLANE), params)
    }

    /// [`build`](Self::build) with `lanes` threads per unit of
    /// `k_schur_tiles` (the device's lockstep width in `build`).
    fn build_lanes(dev: &Arc<D>, p: SchurPlan, nh: usize, values: Values, lanes: u32, params: &GpuSchurParams) -> Result<Self, GpuError> {
        // k_schur_w's threads per observer, a row of W each
        let w_rows = p.kw_cap as u32;
        // a unit wider than the device's lockstep width takes a block alone:
        // its barrier is the block's
        let warps = if lanes == dev.arch().warp { params.tile_warps_on(dev.arch()) } else { 1 };
        let groups = params.tile_groups.unwrap_or_else(|| tile_groups(&p.words, p.nt as usize, lanes));
        let elem = if params.f32_reduction { 4 } else { 8 };
        let widths = TileWidths { kw_cap: p.kw_cap, ew_cap: p.ew_cap, kw_max: p.kw_max, ew_max: p.ew_max, lanes, elem };
        let set = (params.tile_warps.is_some() && lanes == dev.arch().warp, params.tile_groups.is_some());
        let (tile_warps, tile_groups, tile_shared) = fit_tiles(&widths, (warps, groups), set, dev.arch().shared_bytes)?;
        let (kt_ptr_words, kt_words) = tiles_by_kept_block(&p);
        // the warps a kept block takes in the kernels whose lanes are its
        // rows: the reduced right-hand side over its observers, the
        // residual over its tile entries
        let kept_warps = |items: usize, most: u32| {
            let warps = params.kept_warps.unwrap_or_else(|| kept_warps_for(items, p.nk as usize, p.kw_cap, most));
            assert!((1..=MAX_KEPT_WARPS).contains(&warps), "{warps} warps a kept block, 1 to {MAX_KEPT_WARPS}");
            // a block wider than a plane's lanes takes one warp, its lanes over the items
            if p.kw_cap <= PLANE as usize { warps } else { 1 }
        };
        let (rhs_warps, symv_warps) = (kept_warps(p.no as usize, MAX_KEPT_WARPS), kept_warps(kt_words.len(), 8));
        let defines = format!(
            "{}#define SCHUR_F32 {}\n#define RWARPS {rhs_warps}u\n#define SWARPS {symv_warps}u\n{}",
            values.defines(),
            params.f32_reduction as u32,
            defines(&p, lanes, tile_warps, tile_groups)
        );
        let mut words = p.words;
        let blocks = tile_blocks(&words, p.nt as usize, tile_warps as usize, params.split_pairs);
        let tile_blocks = (blocks.len() / (tile_warps as usize + 1)) as u32;
        // the tile blocks, a word holding 1 and the kept blocks' tile lists
        // follow the plan's words, each from a 256-byte boundary, in a tail
        // written apart from them: appending to the plan's words would
        // reallocate them
        let base = words.len();
        let mut tail: Vec<u32> = Vec::new();
        let mut put = |v: &[u32]| {
            let at = (base + tail.len()).div_ceil(64) * 64;
            tail.resize(at - base, 0);
            tail.extend_from_slice(v);
            at
        };
        let at_blocks = put(&blocks);
        let on = 4 * put(&[1]);
        let kt_ptr = 4 * put(&kt_ptr_words);
        let kt_list = 4 * put(&kt_words);
        words[SB_ITEMS] = at_blocks as u32;
        let mut ints = dev.alloc(4 * (base + tail.len()))?;
        dev.write(&mut ints, 0, bytemuck::cast_slice(&words))?;
        dev.write(&mut ints, 4 * base, bytemuck::cast_slice(&tail))?;
        let status_at = status_at(p.n, values.bytes());
        let mut l = Layout::new();
        let sec = Sections {
            h: l.take(nh * values.bytes()),
            rhs: l.take(p.n * values.bytes()),
            dfac: l.take(p.dfac_len * elem),
            z: l.take(p.z_len * elem),
            w: l.take(p.w_len * elem),
            s: l.take(p.s_len * elem),
            rhs_kept: l.take(p.nks * elem),
            x_kept: l.take(p.nks * elem),
            x_prev: l.take(p.nks * elem),
            r: l.take(p.nks * elem),
            out: l.take(status_at + STATUS_BYTES),
            bytes: l.bytes(),
        };
        let work = dev.alloc(sec.bytes)?;
        let source = format!("{WIDE}\n{SHARED}\n{SOURCE}");
        let entries = ["k_zero_u32", "k_schur_elim", "k_schur_w", "k_schur_tiles", "k_schur_rhs", "k_scatter_kept", "k_schur_backsub", "k_symv_sub", "k_copy_r", "k_refine_start", "k_refine_apply", "k_refine_judge", "k_refine_revert", "k_schur_pairs", "k_schur_kobs", "k_schur_kt"];
        let [zero_u32, elim, w, tiles, rhs, scatter_kept, backsub, symv_sub, copy, refine_start, refine_apply, refine_judge, refine_revert, pairs, kobs, kt]: [D::Kernel; 16] =
            dev.compile_many(&source, &defines, &entries)?.try_into().ok().expect("a kernel per entry point");
        let k = Kernels { zero_u32, elim, w, tiles, rhs, scatter_kept, backsub, symv_sub, copy, refine_start, refine_apply, refine_judge, refine_revert, pairs, kobs, kt };
        let header: [u32; SHEADER] = words[..SHEADER].try_into().expect("the plan's header");
        // the expanded records of the pairs, the kept blocks' observers and
        // the kept blocks' tile entries, written on the device from the
        // plan, and the tiles' pair counters the pairs' writing takes
        let (no, nkt, nt) = (p.no as usize, kt_words.len(), p.nt as usize);
        // the kernels address these records' words in u32
        let record_words = PX_REC * p.np + KO_REC * no + KT_REC * nkt + nt;
        if record_words > u32::MAX as usize {
            return Err(GpuError::Limit(format!("the reduction's {} pairs take {record_words} words of records, more than u32 addresses", p.np)));
        }
        let kobs_at = 4 * PX_REC * p.np;
        let kt_at = kobs_at + 4 * KO_REC * no;
        let cur_at = kt_at + 4 * KT_REC * nkt;
        let expanded = dev.alloc((PX_REC * p.np + KO_REC * no + KT_REC * nkt + nt).max(1) * 4)?;
        let hd = Arg::Bytes(bytemuck::cast_slice(&header));
        if p.np > 0 {
            dev.launch(&k.zero_u32, Launch::linear(nt as u32, PLANE), &[Arg::Block(&expanded, cur_at), Arg::U32(nt as u32)])?;
            let args = [Arg::Block(&ints, 0), hd, Arg::Block(&expanded, cur_at), Arg::Block(&expanded, 0), Arg::U32(p.np as u32), Arg::U32(p.ne)];
            dev.launch(&k.pairs, Launch::linear(p.np as u32, THREADS), &args)?;
        }
        if no > 0 {
            let args = [Arg::Block(&ints, 0), hd, Arg::Block(&expanded, kobs_at), Arg::U32(no as u32)];
            dev.launch(&k.kobs, Launch::linear(no as u32, THREADS), &args)?;
        }
        if nkt > 0 {
            let args = [Arg::Block(&ints, 0), hd, Arg::Block(&ints, kt_list), Arg::Block(&expanded, kt_at), Arg::U32(nkt as u32)];
            dev.launch(&k.kt, Launch::linear(nkt as u32, THREADS), &args)?;
        }
        dev.describe(&expanded, "schur.expanded", &|| {
            vec![
                Section::u32s("pairs", 0, PX_REC * p.np).record(PX_REC).doc(
                    "per pair, by tile (ST_PAIR_PTR), a tile's in no fixed order: W's value offset in w of its observer of the tile's row \
                     block, the coupling tile's value offset in h and the sizes of its observer of the column block",
                ),
                Section::u32s("kept_obs", kobs_at, KO_REC * no).record(KO_REC).doc(
                    "per entry of SK_OBS (the observers by kept block): the observer's sizes, its coupling tile's value offset in h, its \
                     eliminated block's value offset in z",
                ),
                Section::u32s("kept_tiles", kt_at, KT_REC * nkt).record(KT_REC).doc(
                    "per entry of kt (the tiles of S by kept block): the tile's value offset in s, its sizes (transposed when the block is the \
                     tile's column), the other block's offset in the reduced vector",
                ),
                Section::u32s("pair_counts", cur_at, nt).doc("per tile of S: its pairs written, the counter k_schur_pairs takes a pair's place from"),
            ]
        });
        dev.describe(&ints, "schur.plan", &|| {
            let mut v = plan_sections(&words);
            v.extend([
                Section::u32s("tile_blocks", 4 * at_blocks, blocks.len()).doc("k_schur_tiles' blocks: per block its units' tiles (and split parts)"),
                Section::u32s("on", on, 1).doc("a word holding 1: the gate of kernels that always run"),
                Section::u32s("kt_ptr", kt_ptr, kt_ptr_words.len()).doc("per kept block: where its tiles of S start in kt"),
                Section::u32s("kt", kt_list, kt_words.len()).doc("the tiles of S touching each kept block (tile index * 2, +1 when the block is the tile's column)"),
            ]);
            v
        });
        dev.describe(&work, "schur.work", &|| {
            let reduced = if params.f32_reduction { "f32" } else { values.wide().name() };
            let r = |name: &str, at: usize, count: usize| Section::new(name, at, count * elem, elem, reduced).role("R");
            vec![
                Section::values("h", sec.h, nh, values).doc("H's tile values: assembled or uploaded, its diagonal damped in place"),
                Section::values("rhs", sec.rhs, p.n, values).doc("the gradient"),
                r("dfac", sec.dfac, p.dfac_len).doc("the eliminated blocks' Cholesky factors"),
                r("z", sec.z, p.z_len).doc("per eliminated block: z = D^-1 b"),
                r("w", sec.w, p.w_len).doc("per observer: W = C D^-1"),
                r("s", sec.s, p.s_len).doc("S's tile values: the reduced system"),
                r("rhs_kept", sec.rhs_kept, p.nks).doc("the reduced right-hand side"),
                r("x_kept", sec.x_kept, p.nks).doc("the kept blocks' solution"),
                r("x_prev", sec.x_prev, p.nks).doc("the refinement's previous iterate"),
                r("r", sec.r, p.nks).doc("the refinement's residual"),
                Section::values("step", sec.out, p.n, values).doc("the full step, what a solve reads back"),
                Section::wide("state", sec.out + status_at, FLAGS_AT / 8, values.wide()).doc("the refinement's state: norms and tolerance"),
                Section::u32s("flags", sec.out + status_at + FLAGS_AT, N_FLAGS).doc("the solve's flags: refinement active, steps, revert, failed eliminated block, failed pivot"),
            ]
        });
        Ok(GpuSchur {
            graph: None,
            graphs: true,
            replayed: false,
            dev: dev.clone(),
            id: NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            defines,
            kw_max: p.kw_max,
            kw_cap: p.kw_cap,
            kt_entries: nkt,
            rhs_threads: rhs_warps * PLANE,
            symv_threads: symv_warps * PLANE,
            values,
            wide: values.wide(),
            f32_reduction: params.f32_reduction,
            k,
            ints,
            header,
            expanded,
            kobs_at,
            kt_at,
            on,
            kt_ptr,
            work,
            sec,
            status_at,
            ne: p.ne,
            no: p.no,
            lanes,
            tile_warps,
            tile_groups,
            tile_blocks,
            tile_shared,
            w_rows,
            nk: p.nk,
            n: p.n,
            nks: p.nks,
            nh,
            s_len: p.s_len,
            elim_blk: p.elim_blk,
            stamps: None,
            conv: Vec::new(),
            host_out: vec![0; status_at + STATUS_BYTES],
            step: vec![0.0; p.n],
        })
    }

    /// The lane groups per unit of the forming of S
    /// ([`GpuSchurParams::tile_groups`]), as given or as the plan chose,
    /// lowered to fit the device's shared memory.
    pub fn tile_groups(&self) -> u32 {
        self.tile_groups
    }

    /// The units per block of the forming of S
    /// ([`GpuSchurParams::tile_warps`]), as given or the device's, lowered
    /// to fit its shared memory.
    pub fn tile_warps(&self) -> u32 {
        self.tile_warps
    }

    /// The device the reduction runs on; its factor runs there too.
    pub fn device(&self) -> &Arc<D> {
        &self.dev
    }

    /// The reduction's identity, unique in the process: what was built
    /// over this reduction's blocks (a graph, the conjugate gradients)
    /// runs only with it.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// The reduced system as a solver of it other than a factor reads it.
    pub(crate) fn reduced_system(&self) -> ReducedSystem<'_, D> {
        ReducedSystem {
            defines: &self.defines,
            plan: &self.ints,
            header: &self.header,
            kt_ptr: self.kt_ptr,
            expanded: &self.expanded,
            kt: self.kt_at,
            work: &self.work,
            s: self.sec.s,
            b: self.sec.rhs_kept,
            x: self.sec.x_kept,
            r: self.sec.r,
            flags: self.flag_at(0),
            nk: self.nk,
            nks: self.nks,
            kw_max: self.kw_max,
            kw_cap: self.kw_cap,
            kt_entries: self.kt_entries,
        }
    }

    /// The Hessian's scalar dimension.
    pub fn dim(&self) -> usize {
        self.n
    }

    /// The reduced system's scalar dimension.
    pub fn kept_size(&self) -> usize {
        self.nks
    }

    /// S's values on the device (block, byte offset) and their count, in the
    /// storage order of the symbolic's `s`, as the last reduction left them,
    /// in [`reduced`](Self::reduced).
    pub fn s_values(&self) -> ((&D::Block, usize), usize) {
        ((&self.work, self.sec.s), self.s_len)
    }

    /// The route's wide scalar: the refinement's state's, and the
    /// reduction's without [`GpuSchurParams::f32_reduction`].
    pub fn wide(&self) -> Wide {
        self.wide
    }

    /// The route's values: H's, the gradient's and the step's scalar.
    pub fn values(&self) -> Values {
        self.values
    }

    /// Whether the reduction computes in f32
    /// ([`GpuSchurParams::f32_reduction`]).
    pub fn f32_reduction(&self) -> bool {
        self.f32_reduction
    }

    /// The reduced right-hand side on the device, [`kept_size`](Self::kept_size)
    /// values of [`reduced`](Self::reduced).
    pub fn rhs_kept(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.rhs_kept)
    }

    /// Where [`launch`](Self::launch) reads the Hessian's tile values: the
    /// structure's value count, in [`values`](Self::values), for a device
    /// assembly to write.
    pub fn h_target(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.h)
    }

    /// Where [`launch`](Self::launch) reads the gradient: [`dim`](Self::dim) values.
    pub fn rhs_target(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.rhs)
    }

    /// The step of the last solve: [`dim`](Self::dim) values of
    /// [`values`](Self::values).
    pub fn step_view(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.out)
    }

    /// The last solve's status ([`STATUS_BYTES`] bytes), to read and hand to
    /// [`outcome`](Self::outcome).
    pub fn status_view(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.out + self.status_at)
    }

    /// The Hessian's tile values as the last upload or device assembly left
    /// them, read back.
    #[cfg(test)]
    pub(crate) fn h_values(&self) -> Result<Vec<f64>, GpuError> {
        let mut bytes = vec![0u8; self.values.bytes() * self.nh];
        self.dev.read(&self.work, self.sec.h, &mut bytes)?;
        let mut v = vec![0.0f64; self.nh];
        self.values.decode(&bytes, &mut v);
        Ok(v)
    }

    /// S's values and the reduced right-hand side read back.
    #[cfg(test)]
    pub(crate) fn download(&self) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        Ok((self.read_reduced(self.sec.s, self.s_len)?, self.read_reduced(self.sec.rhs_kept, self.nks)?))
    }

    /// The scalar the reduction computes and keeps S and its vectors in: f32
    /// with [`GpuSchurParams::f32_reduction`], else the wide scalar.
    pub fn reduced(&self) -> Values {
        Values::of(if self.f32_reduction { 4 } else { 8 }, self.wide)
    }

    /// `n` values of [`reduced`](Self::reduced) of the work block from byte
    /// `at` on, read back.
    #[cfg(test)]
    fn read_reduced(&self, at: usize, n: usize) -> Result<Vec<f64>, GpuError> {
        let r = self.reduced();
        let mut bytes = vec![0u8; r.bytes() * n];
        self.dev.read(&self.work, at, &mut bytes)?;
        let mut v = vec![0.0f64; n];
        r.decode(&bytes, &mut v);
        Ok(v)
    }

    /// Byte offset of flag `f` in the work block.
    fn flag_at(&self, f: usize) -> usize {
        self.sec.out + self.status_at + FLAGS_AT + 4 * f
    }

    /// Writes the Hessian's tile values `hvals`, in the storage order of the
    /// structure the reduction was built for, and the gradient `rhs` at
    /// [`h_target`](Self::h_target) and [`rhs_target`](Self::rhs_target),
    /// converted to [`values`](Self::values), for the launches after it
    /// (no wait).
    ///
    /// # Panics
    ///
    /// When `hvals` is not the structure's value count or `rhs` not
    /// [`dim`](Self::dim) values.
    pub fn upload<M: SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) -> Result<(), GpuError> {
        assert_eq!(hvals.len(), self.nh, "the Hessian's values, in the structure the reduction was built for");
        assert_eq!(rhs.len(), self.n);
        let values = self.values;
        self.conv.clear();
        values.extend(hvals.iter().map(|v| v.to_f64()), &mut self.conv);
        self.dev.write(&mut self.work, self.sec.h, &self.conv)?;
        self.conv.clear();
        values.extend(rhs.iter().map(|v| v.to_f64()), &mut self.conv);
        self.dev.write(&mut self.work, self.sec.rhs, &self.conv)
    }

    /// `kernel` over `blocks` blocks of `threads`; nothing when there are no blocks.
    fn run(&self, kernel: &D::Kernel, blocks: u32, threads: u32, args: &[Arg<'_, D::Block>]) -> Result<(), GpuError> {
        self.run_shared(kernel, blocks, threads, 0, args)
    }

    /// [`run`](Self::run) with `shared_bytes` of dynamic shared memory a block.
    fn run_shared(&self, kernel: &D::Kernel, blocks: u32, threads: u32, shared_bytes: u32, args: &[Arg<'_, D::Block>]) -> Result<(), GpuError> {
        self.dev.launch(kernel, Launch { grid: [blocks, 1, 1], block: [threads, 1, 1], shared_bytes }, args)
    }

    /// The plan's header as a launch's argument.
    fn header_arg(&self) -> Arg<'_, D::Block> {
        Arg::Bytes(bytemuck::cast_slice(&self.header))
    }

    /// The launches of the reduction, the first stage of a damped solve:
    /// the flags zeroed, then S and the reduced right-hand side formed from
    /// the damped Hessian's values and the gradient on the device (at
    /// [`h_target`](Self::h_target) and [`rhs_target`](Self::rhs_target)).
    /// [`launch`](Self::launch) runs the three stages with a factor; a
    /// caller with another solver of the reduced system launches it between
    /// this and [`launch_backsub`](Self::launch_backsub).
    ///
    /// ```ignore
    /// g.launch_reduce()?;
    /// g.launch_factored(&factor, None)?;
    /// g.launch_backsub()?;
    /// ```
    pub fn launch_reduce(&self) -> Result<(), GpuError> {
        let _scope = Scope::new(&*self.dev, "reduce");
        let (s, plan, hd) = (&self.sec, Arg::Block(&self.ints, 0), self.header_arg());
        let w = |at: usize| Arg::Block(&self.work, at);
        self.run(&self.k.zero_u32, 1, PLANE, &[w(self.flag_at(0)), Arg::U32(N_FLAGS as u32)])?;
        let elim = [plan, hd, w(s.h), w(s.rhs), w(s.dfac), w(s.z), w(self.flag_at(F_ELIM)), Arg::U32(self.ne)];
        self.run(&self.k.elim, self.ne.div_ceil(THREADS), THREADS, &elim)?;
        self.run(&self.k.w, (self.no * self.w_rows).div_ceil(THREADS), THREADS, &[plan, hd, w(s.h), w(s.dfac), w(s.w), Arg::U32(self.no)])?;
        let (pairs, kobs) = (Arg::Block(&self.expanded, 0), Arg::Block(&self.expanded, self.kobs_at));
        self.run_shared(&self.k.tiles, self.tile_blocks, self.lanes * self.tile_warps, self.tile_shared, &[plan, hd, pairs, w(s.h), w(s.w), w(s.s)])?;
        self.run(&self.k.rhs, self.nk, self.rhs_threads, &[plan, hd, kobs, w(s.h), w(s.rhs), w(s.z), w(s.rhs_kept)])
    }

    /// Forms S and the reduced right-hand side from the damped Hessian's
    /// tile values `hvals` and the gradient `rhs` (as [`upload`](Self::upload)
    /// takes them), and waits for the eliminated blocks' pivot flags: the
    /// first stage of [`solve`](Self::solve) on its own, for inspection.
    /// [`GpuSchurError::Eliminated`] when an eliminated block's diagonal
    /// tile is not positive definite.
    pub fn reduce<M: SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) -> Result<(), GpuSchurError> {
        self.upload(hvals, rhs)?;
        self.launch_reduce()?;
        let mut failed = [0u32];
        self.dev.read(&self.work, self.flag_at(F_ELIM), bytemuck::cast_slice_mut(&mut failed))?;
        if failed[0] != 0 { Err(GpuSchurError::Eliminated { block: self.elim_blk[failed[0] as usize - 1] as usize }) } else { Ok(()) }
    }

    /// Turns graphs on (the default) or off for [`solve`](Self::solve). With
    /// graphs, the first solve with a factor and a refinement setting runs
    /// its kernels and records them as a graph, and the solves after it
    /// with the same factor and setting replay the recording, one launch
    /// instead of hundreds.
    pub fn use_graphs(&mut self, on: bool) {
        self.graphs = on;
        if !on {
            self.graph = None;
        }
    }

    /// Whether solves capture and replay graphs ([`use_graphs`](Self::use_graphs)).
    pub fn graphs(&self) -> bool {
        self.graphs
    }

    /// Whether the last [`solve`](Self::solve) replayed a captured graph.
    pub fn replayed(&self) -> bool {
        self.replayed
    }

    /// The launches of one damped solve after the uploads: the reduction,
    /// the factorization, the solve, `max_steps` refinement steps, the
    /// back-substitution. The same sequence every time for the same factor
    /// and refinement, which is what makes it a graph.
    fn launch_attempt(&self, factor: &GpuFactor<D>, max_steps: usize, tol_rel: f64) -> Result<(), GpuError> {
        self.launch_reduce()?;
        self.launch_solve(factor, max_steps, tol_rel)?;
        self.launch_backsub()
    }

    /// The launches that solve the reduced system with `factor`, the stage
    /// between [`launch_reduce`](Self::launch_reduce) and
    /// [`launch_backsub`](Self::launch_backsub): S factorized, the kept
    /// blocks solved for, and the solution refined against S under `refine`
    /// (`(max_steps, tolerance)`, as [`solve`](Self::solve) takes it).
    ///
    /// # Panics
    ///
    /// When `factor` is not on this reduction's device (the same `Arc`),
    /// and for a refinement of a reduction with
    /// [`GpuSchurParams::f32_reduction`].
    pub fn launch_factored(&self, factor: &GpuFactor<D>, refine: Option<(usize, f64)>) -> Result<(), GpuError> {
        assert!(Arc::ptr_eq(&self.dev, factor.device()), "the factor runs on the reduction's device, so S never leaves it");
        let (max_steps, tol_rel) = refine.filter(|r| r.0 > 0).unwrap_or((0, 0.0));
        self.launch_solve(factor, max_steps, tol_rel)
    }

    /// [`launch_factored`](Self::launch_factored) with the refinement's
    /// steps and tolerance as they are launched.
    fn launch_solve(&self, factor: &GpuFactor<D>, max_steps: usize, tol_rel: f64) -> Result<(), GpuError> {
        assert!(!(self.f32_reduction && max_steps > 0), "a refinement measures its residuals against S, which GpuSchurParams::f32_reduction forms in f32");
        let s = &self.sec;
        let w = |at: usize| Arg::Block(&self.work, at);
        {
            let _scope = Scope::new(&*self.dev, "factorize");
            if self.f32_reduction {
                factor.factorize_device_f32((&self.work, s.s), (&self.work, self.flag_at(F_FACTOR)))?;
            } else {
                factor.factorize_device((&self.work, s.s), (&self.work, self.flag_at(F_FACTOR)))?;
            }
        }
        let nks = self.nks as u32;
        let per_entry = nks.div_ceil(THREADS);
        let on = Arg::Block(&self.ints, self.on);
        // the flags, and the one that gates a refinement step's kernels
        let flags = w(self.flag_at(0));
        let gate = w(self.flag_at(F_ACTIVE));
        let state = w(s.out + self.status_at);
        let copy = |gate, dst: usize, src: usize| self.run(&self.k.copy, per_entry, THREADS, &[gate, w(dst), w(src), Arg::U32(nks)]);
        let residual = |gate| self.launch_residual(gate);
        {
            let _scope = Scope::new(&*self.dev, "solve");
            copy(on, s.x_kept, s.rhs_kept)?;
            if self.f32_reduction {
                factor.solve_device_f32((&self.work, s.x_kept))?;
            } else {
                factor.solve_device((&self.work, s.x_kept))?;
            }
        }
        if max_steps > 0 {
            let _scope = Scope::new(&*self.dev, "refine");
            residual(on)?;
            self.run(&self.k.refine_start, 1, THREADS, &[w(s.rhs_kept), w(s.r), state, flags, Arg::U32(nks), self.wide.arg(tol_rel)])?;
            for _ in 0..max_steps {
                factor.solve_device_if((&self.work, s.r), (&self.work, self.flag_at(F_ACTIVE)))?;
                self.run(&self.k.refine_apply, per_entry, THREADS, &[flags, w(s.x_kept), w(s.x_prev), w(s.r), Arg::U32(nks)])?;
                residual(gate)?;
                self.run(&self.k.refine_judge, 1, THREADS, &[w(s.r), state, flags, Arg::U32(nks)])?;
            }
            self.run(&self.k.refine_revert, per_entry, THREADS, &[flags, w(s.x_kept), w(s.x_prev), Arg::U32(nks)])?;
        }
        Ok(())
    }

    /// The launches of the residual of the kept blocks' solution, `r = b -
    /// S x` (the reduced right-hand side, S, the solution and `r` in the
    /// work block), when `gate[0]` is nonzero.
    pub(crate) fn launch_residual(&self, gate: Arg<'_, D::Block>) -> Result<(), GpuError> {
        let (s, plan) = (&self.sec, Arg::Block(&self.ints, 0));
        let w = |at: usize| Arg::Block(&self.work, at);
        let nks = self.nks as u32;
        self.run(&self.k.copy, nks.div_ceil(THREADS), THREADS, &[gate, w(s.r), w(s.rhs_kept), Arg::U32(nks)])?;
        let (kt_ptr, kt) = (Arg::Block(&self.ints, self.kt_ptr), Arg::Block(&self.expanded, self.kt_at));
        self.run(&self.k.symv_sub, self.nk, self.symv_threads, &[gate, plan, self.header_arg(), kt_ptr, kt, w(s.s), w(s.x_kept), w(s.r)])
    }

    /// The launches of the back-substitution, the last stage of a damped
    /// solve: the kept blocks' solution into the full step (at
    /// [`step_view`](Self::step_view)), then the eliminated blocks'.
    pub fn launch_backsub(&self) -> Result<(), GpuError> {
        let _scope = Scope::new(&*self.dev, "backsub");
        let (s, plan, hd) = (&self.sec, Arg::Block(&self.ints, 0), self.header_arg());
        let w = |at: usize| Arg::Block(&self.work, at);
        self.run(&self.k.scatter_kept, self.nk.div_ceil(THREADS), THREADS, &[plan, hd, w(s.x_kept), w(s.out), Arg::U32(self.nk)])?;
        self.run(&self.k.backsub, self.ne.div_ceil(THREADS), THREADS, &[plan, hd, w(s.h), w(s.rhs), w(s.dfac), w(s.out), Arg::U32(self.ne)])
    }

    /// Have [`solve`](Self::solve) stamp the device's queue
    /// ([`Device::stamp`]) before its upload (`stamps.0`) and after its
    /// launches (`stamps.1`), so the span between them, read after the
    /// solve, is the device's time for it; `None` (the default) stamps
    /// nothing.
    ///
    /// ```ignore
    /// g.stamp_solves(Some((0, 1)));
    /// g.solve(&mut factor, &hvals, &rhs, None, &mut delta)?;
    /// let seconds = dev.span(0, 1)?;
    /// ```
    pub fn stamp_solves(&mut self, stamps: Option<(usize, usize)>) {
        self.stamps = stamps;
    }

    /// Solves the damped system for the step into `delta`, on the device,
    /// with one wait. The input is the damped Hessian's tile values `hvals`
    /// and the gradient `rhs`, as [`upload`](Self::upload) takes them;
    /// `factor` factorizes and solves S. `refine` is `Some((max_steps,
    /// tol))` to refine the solution against S (for an f32 factor), `tol`
    /// relative to the norm of the reduced right-hand side; its
    /// [`RefineStats`] come back.
    ///
    /// The refinement decides on the device: `max_steps` steps are
    /// launched, and those after it has converged or stalled do nothing. A
    /// failed pivot is found with the result, so a failing solve runs to the
    /// end; `delta` is then left as it was. The launches are captured as a
    /// graph and replayed by the solves after it (see
    /// [`use_graphs`](Self::use_graphs)).
    ///
    /// ```ignore
    /// let stats = schur.solve(&mut factor, h.vals(), &rhs, Some((4, 1e-10)), &mut delta)?;
    /// if stats.short {
    ///     // the refinement stopped above the tolerance
    /// }
    /// ```
    ///
    /// # Errors
    ///
    /// [`GpuSchurError::Eliminated`] or [`GpuSchurError::Reduced`] when a
    /// matrix is not positive definite (raise the damping and solve again);
    /// [`GpuSchurError::Device`] when a device operation fails.
    ///
    /// # Panics
    ///
    /// As [`upload`](Self::upload) and
    /// [`launch_factored`](Self::launch_factored), and when `delta` is not
    /// [`dim`](Self::dim) values.
    pub fn solve<M: SchurReal + 'static>(
        &mut self,
        factor: &mut GpuFactor<D>,
        hvals: &[M],
        rhs: &[M],
        refine: Option<(usize, f64)>,
        delta: &mut [M],
    ) -> Result<RefineStats, GpuSchurError> {
        self.start_solve(hvals, rhs)?;
        self.enqueue(factor, refine)?;
        self.finish_solve(refine, delta)
    }

    /// The start of a damped solve from host values: the stamp before it
    /// ([`stamp_solves`](Self::stamp_solves)) and the upload of the
    /// Hessian's tile values and the gradient; the solve's launches follow.
    pub(crate) fn start_solve<M: SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) -> Result<(), GpuError> {
        if let Some((before, _)) = self.stamps {
            self.dev.stamp(before)?;
        }
        self.upload(hvals, rhs)
    }

    /// The end of a damped solve, after its launches: the stamp after it,
    /// one read of the step and the status (the wait for the solve), the
    /// outcome under the refinement `refine`, and the step into `delta`
    /// (left as it was by a failed solve).
    pub(crate) fn finish_solve<M: SchurReal + 'static>(&mut self, refine: Option<(usize, f64)>, delta: &mut [M]) -> Result<RefineStats, GpuSchurError> {
        assert_eq!(delta.len(), self.n);
        if let Some((_, after)) = self.stamps {
            self.dev.stamp(after)?;
        }
        self.dev.read(&self.work, self.sec.out, &mut self.host_out)?;
        let stats = self.outcome(&self.host_out[self.status_at..], refine)?;
        self.values.decode(&self.host_out[..self.values.bytes() * self.n], &mut self.step);
        for (d, &v) in delta.iter_mut().zip(&self.step) {
            *d = M::from_f64(v);
        }
        Ok(stats)
    }

    /// The attempt's launches, replayed when a graph of them is captured,
    /// without waiting; with graphs on and none captured for this factor and
    /// refinement, the launches are recorded right after they are queued, so
    /// the recording overlaps their run.
    fn enqueue(&mut self, factor: &GpuFactor<D>, refine: Option<(usize, f64)>) -> Result<(), GpuError> {
        assert!(Arc::ptr_eq(&self.dev, factor.device()), "the factor runs on the reduction's device, so S never leaves it");
        let (max_steps, tol_rel) = refine.filter(|r| r.0 > 0).unwrap_or((0, 0.0));
        let key = self.graph_key(factor, refine);
        let replay = self.graphs && self.graph.as_ref().is_some_and(|g| g.0 == key);
        self.replayed = replay;
        if replay {
            return self.dev.replay(&self.graph.as_ref().unwrap().1);
        }
        self.launch_attempt(factor, max_steps, tol_rel)?;
        if self.graphs {
            self.graph = None;
            let this = &*self;
            let graph = self.dev.capture(|| this.launch_attempt(factor, max_steps, tol_rel))?;
            self.graph = Some((key, graph));
        }
        Ok(())
    }

    /// The launches of [`solve`](Self::solve) and nothing else, from the
    /// damped Hessian's values and the gradient already on the device (at
    /// [`h_target`](Self::h_target) and [`rhs_target`](Self::rhs_target)):
    /// the step lands at [`step_view`](Self::step_view), the status at
    /// [`status_view`](Self::status_view), for the caller to read with its
    /// own results and hand to [`outcome`](Self::outcome). Launches only, so
    /// a caller can capture them in a graph of its own (keyed by
    /// [`graph_key`](Self::graph_key)). Panics as
    /// [`launch_factored`](Self::launch_factored).
    pub fn launch(&self, factor: &GpuFactor<D>, refine: Option<(usize, f64)>) -> Result<(), GpuError> {
        assert!(Arc::ptr_eq(&self.dev, factor.device()), "the factor runs on the reduction's device, so S never leaves it");
        let (max_steps, tol_rel) = refine.filter(|r| r.0 > 0).unwrap_or((0, 0.0));
        self.launch_attempt(factor, max_steps, tol_rel)
    }

    /// What a captured [`launch`](Self::launch) depends on beyond the
    /// reduction's own blocks: the factor and the refinement. A graph
    /// replays only under its key.
    pub fn graph_key(&self, factor: &GpuFactor<D>, refine: Option<(usize, f64)>) -> GraphKey {
        let (max_steps, tol_rel) = refine.filter(|r| r.0 > 0).unwrap_or((0, 0.0));
        GraphKey { factor: factor.id(), max_steps, tol_bits: tol_rel.to_bits() }
    }

    /// The outcome of a solve with refinement `refine` from its status bytes
    /// (as read from [`status_view`](Self::status_view)).
    pub fn outcome(&self, status: &[u8], refine: Option<(usize, f64)>) -> Result<RefineStats, GpuSchurError> {
        // the state's words as f64, the flags after them as they are
        let mut f64_status = status[..STATUS_BYTES].to_vec();
        for c in f64_status[..FLAGS_AT].as_chunks_mut::<8>().0 {
            *c = self.wide.value(u64::from_le_bytes(*c)).to_le_bytes();
        }
        Ok(decode_status(&f64_status, &self.elim_blk, refine)?)
    }
}

/// The plan words' sections, for [`Device::describe`]: the header, then
/// the table or array of each of its slots.
fn plan_sections(words: &[u32]) -> Vec<Section> {
    let slots = [
        (
            SE,
            "SE",
            "per eliminated block, a record: the value offset of its diagonal tile in h, its width, its scalar offset in the gradient and the step, \
             its first observer (the next record's is their end; one record more than blocks), the value offset of its square in dfac, its value offset in z",
        ),
        (
            SO,
            "SO",
            "per observer, a record: the value offset of W_o in w, its sizes (the width of its kept block, of its eliminated block times 256, \
             65536 when its coupling tile is stored (eliminated, kept)), the value offset of its coupling tile in h, its eliminated block, \
             the scalar offset of its kept block in the step",
        ),
        (ST_VAL_OFF, "ST_VAL_OFF", "per tile of S: its value offset in s"),
        (ST_WR, "ST_WR", "per tile of S: its rows"),
        (ST_WC, "ST_WC", "per tile of S: its columns"),
        (ST_A, "ST_A", "per tile of S: its row kept block"),
        (ST_B, "ST_B", "per tile of S: its column kept block"),
        (ST_COPY_OFF, "ST_COPY_OFF", "per tile of S: the value offset in h of the kept-kept tile behind it, or NONE"),
        (ST_PAIR_PTR, "ST_PAIR_PTR", "per tile of S: its first pair"),
        (OBS_KEPT, "OBS_KEPT", "per observer: its kept block"),
        (SK_OBS_PTR, "SK_OBS_PTR", "per kept block: its first entry in SK_OBS"),
        (SK_OBS, "SK_OBS", "the observers grouped by kept block"),
        (SK_S_OFF, "SK_S_OFF", "per kept block: its scalar offset in the reduced vector"),
        (SK_ORIG_OFF, "SK_ORIG_OFF", "per kept block: its scalar offset in the step"),
        (SK_W, "SK_W", "per kept block: its width"),
        (SK_T_PTR, "SK_T_PTR", "per kept block: its first tile of S, its column's"),
        (SE_PAIR_PTR, "SE_PAIR_PTR", "per eliminated block: its first pair in k_schur_pairs' threads"),
    ];
    let mut sections = crate::plan_sections(words, SHEADER, &slots);
    for s in &mut sections {
        match s.name.as_str() {
            "SE" => s.record = SE_REC,
            "SO" => s.record = SO_REC,
            _ => {}
        }
    }
    sections
}

/// Per kept block, the tiles of S touching it: `(ptr, tiles)` with block
/// k's at `tiles[ptr[k]..ptr[k + 1]]`, each a tile index times 2, plus 1
/// when k is the tile's column block and not its row block. The device
/// expands each into its record for `k_symv_sub` (`k_schur_kt`, [`KT_REC`]).
fn tiles_by_kept_block(p: &SchurPlan) -> (Vec<u32>, Vec<u32>) {
    let (nt, nk) = (p.nt as usize, p.nk as usize);
    let array = |slot: usize| &p.words[p.words[slot] as usize..][..nt];
    let (ta, tb) = (array(ST_A), array(ST_B));
    let mut ptr = vec![0u32; nk + 1];
    for t in 0..nt {
        ptr[ta[t] as usize + 1] += 1;
        if ta[t] != tb[t] {
            ptr[tb[t] as usize + 1] += 1;
        }
    }
    for k in 0..nk {
        ptr[k + 1] += ptr[k];
    }
    let mut fill = ptr.clone();
    let mut tiles = vec![0u32; ptr[nk] as usize];
    for t in 0..nt {
        let a = ta[t] as usize;
        tiles[fill[a] as usize] = 2 * t as u32;
        fill[a] += 1;
        if ta[t] != tb[t] {
            let b = tb[t] as usize;
            tiles[fill[b] as usize] = 2 * t as u32 + 1;
            fill[b] += 1;
        }
    }
    (ptr, tiles)
}

/// The blocks of `k_schur_tiles` of `units` units over the plan's `nt`
/// tiles (`words`, its pair pointers at `ST_PAIR_PTR`), `units` + 1 words
/// each: a tile of more than `split_pairs` pairs alone on a block (word 0
/// one), the others `units` to a block (word 0 zero, idle units NONE), the
/// tiles by pair count, most first, so the longest work starts first and a
/// block's units finish together. One unit: a block per tile, in storage
/// order.
fn tile_blocks(words: &[u32], nt: usize, units: usize, split_pairs: u32) -> Vec<u32> {
    let ptr = &words[words[ST_PAIR_PTR] as usize..][..nt + 1];
    let pairs = |t: u32| ptr[t as usize + 1] - ptr[t as usize];
    let mut order: Vec<u32> = (0..nt as u32).collect();
    if units == 1 {
        // a block per tile, in storage order: nothing to split or group
        return order.iter().flat_map(|&t| [0, t]).collect();
    }
    order.sort_by_key(|&t| std::cmp::Reverse(pairs(t)));
    let split = order.iter().take_while(|&&t| pairs(t) > split_pairs).count();
    let mut out = Vec::with_capacity((split + (nt - split).div_ceil(units)) * (units + 1));
    for &t in &order[..split] {
        out.extend_from_slice(&[1, t]);
        out.extend(std::iter::repeat_n(NONE, units - 1));
    }
    for chunk in order[split..].chunks(units) {
        out.push(0);
        out.extend_from_slice(chunk);
        out.extend(std::iter::repeat_n(NONE, units - chunk.len()));
    }
    out
}

/// The entries of a tile of S per lane of `k_schur_tiles`, at most, for
/// kept blocks of at most `kw_max` columns, `groups` groups of a unit of
/// `lanes` lanes.
fn tile_acc(kw_max: usize, lanes: u32, groups: u32) -> usize {
    (kw_max * kw_max).div_ceil((lanes / groups) as usize)
}

/// The block widths `k_schur_tiles` is compiled for: the kept and the
/// eliminated blocks' rounded up (`KW`, `EW`), the widest of each, and a
/// unit's lanes.
struct TileWidths {
    kw_cap: usize,
    ew_cap: usize,
    kw_max: usize,
    ew_max: usize,
    lanes: u32,
    /// Bytes of a value as the reduction computes: 8 in the wide scalar
    /// (f64 or df32), 4 in f32 ([`GpuSchurParams::f32_reduction`]).
    elem: usize,
}

/// `k_schur_tiles`' shared memory a block, in bytes, at `warps` units of
/// `groups` lane groups: the groups' W_oa and C_ob (`KW x EW` each) and the
/// units' sums of a split tile, `elem` bytes each.
///
/// Kept blocks of 9 columns, eliminated of 3 (`KW` 16, `EW` 4), on 32
/// lanes, 1 unit of 1 group, in the wide scalar: 8 (2 x 64 + 3 x 32) = 1792.
fn tiles_shared_bytes(w: &TileWidths, warps: u32, groups: u32) -> usize {
    let unit = 2 * groups as usize * w.kw_cap * w.ew_cap + tile_acc(w.kw_max, w.lanes, groups) * (w.lanes / groups) as usize;
    w.elem * warps as usize * unit
}

/// The units per block and lane groups of `k_schur_tiles` that fit
/// `limit` bytes of shared memory, and its bytes: `want` lowered, the units
/// first, then the groups (halved, staying 1, 2, 4 or 8), where it is the
/// default; [`GpuError::Limit`] where a value set (`set`: the units', the
/// groups') does not fit, or one unit of one group does not.
fn fit_tiles(w: &TileWidths, want: (u32, u32), set: (bool, bool), limit: u32) -> Result<(u32, u32, u32), GpuError> {
    let (mut warps, mut groups) = want;
    assert!(warps >= 1, "tile_warps {warps} is at least 1");
    assert!(matches!(groups, 1 | 2 | 4 | 8), "tile_groups {groups} is 1, 2, 4 or 8");
    let fits = |warps, groups| tiles_shared_bytes(w, warps, groups) <= limit as usize;
    while !fits(warps, groups) && !set.0 && warps > 1 {
        warps -= 1;
    }
    while !fits(warps, groups) && !set.1 && groups > 1 {
        groups /= 2;
    }
    let bytes = tiles_shared_bytes(w, warps, groups);
    if bytes > limit as usize {
        let what = match set {
            (true, _) | (_, true) => format!("GpuSchurParams' tile_warps {warps} and tile_groups {groups}"),
            _ => "one unit of one lane group".to_string(),
        };
        return Err(GpuError::Limit(format!(
            "the forming of S (k_schur_tiles) takes {bytes} bytes of shared memory for kept blocks of {} and eliminated blocks of {} columns at {what}, the device {limit}",
            w.kw_max, w.ew_max
        )));
    }
    Ok((warps, groups, bytes as u32))
}

/// [`tile_groups`] picks two lane groups per unit when they issue fewer
/// products than one by more than this share.
const TWO_GROUPS_SAVING: f64 = 0.1;

/// The lane groups per unit of `k_schur_tiles`
/// ([`GpuSchurParams::tile_groups`]) for the plan's `nt` tiles (`words`)
/// on units of `lanes` lanes: two when they issue more than
/// [`TWO_GROUPS_SAVING`] fewer products than one ([`issued_products`]),
/// else one. Measured on an NVIDIA discrete and an AMD integrated GPU: on
/// SLAM's 6 x 6 tiles two groups save 24% of the products and take 20% and
/// 11% less time than one; on bundle adjustment's 9 x 9 tiles they save 2%
/// and take 3% and 15% more; four groups are slower than two on both.
fn tile_groups(words: &[u32], nt: usize, lanes: u32) -> u32 {
    let (one, two) = (issued_products(words, nt, lanes, 1), issued_products(words, nt, lanes, 2));
    if (two as f64) < (1.0 - TWO_GROUPS_SAVING) * one as f64 { 2 } else { 1 }
}

/// The products `k_schur_tiles` issues over the plan's `nt` tiles
/// (`words`) with `groups` groups of a unit of `lanes` lanes, per column of
/// the pairs' eliminated blocks: per tile, its pairs over the groups times
/// its entries over a group's lanes, both rounded up.
///
/// A 6 x 6 tile of 4 pairs on 32 lanes: 2 x 3 with 2 groups against 4 x 2
/// with one; of 1 pair, 1 x 3 against 1 x 2.
fn issued_products(words: &[u32], nt: usize, lanes: u32, groups: u32) -> u64 {
    let at = |slot: usize| &words[words[slot] as usize..][..nt];
    let (wr, wc, a, b) = (at(ST_WR), at(ST_WC), at(ST_A), at(ST_B));
    let ptr = &words[words[ST_PAIR_PTR] as usize..][..nt + 1];
    let group_lanes = (lanes / groups) as u64;
    (0..nt)
        .map(|t| {
            let (r, c) = (wr[t] as u64, wc[t] as u64);
            let entries = if a[t] == b[t] { r * (r + 1) / 2 } else { r * c };
            ((ptr[t + 1] - ptr[t]) as u64).div_ceil(groups as u64) * entries.div_ceil(group_lanes)
        })
        .sum()
}

/// The `#define`s of `schur.cpp`: the block sizes, the width classes, and
/// the plan header's and the status's slots of [`crate::schur`].
fn defines(p: &SchurPlan, lanes: u32, tile_warps: u32, tile_groups: u32) -> String {
    let mut s = String::new();
    define_u32s(
        &mut s,
        &[
            ("THREADS", THREADS as usize),
            ("TILE_LANES", lanes as usize),
            ("TILE_WARPS", tile_warps as usize),
            ("TILE_GROUPS", tile_groups as usize),
            ("EWX", p.ew_max),
            ("EW", p.ew_cap),
            ("KW", p.kw_cap),
            ("ACC", tile_acc(p.kw_max, lanes, tile_groups)),
            ("SHEADER", SHEADER),
            ("SE", SE),
            ("SE_REC", SE_REC),
            ("SE_DIAG_OFF", SE_DIAG_OFF),
            ("SE_W", SE_W),
            ("SE_ORIG_OFF", SE_ORIG_OFF),
            ("SE_OBS_PTR", SE_OBS_PTR),
            ("SE_DFAC_OFF", SE_DFAC_OFF),
            ("SE_Z_OFF", SE_Z_OFF),
            ("SO", SO),
            ("SO_REC", SO_REC),
            ("SO_W_OFF", SO_W_OFF),
            ("SO_DIMS", SO_DIMS),
            ("SO_CA_OFF", SO_CA_OFF),
            ("SO_E", SO_E),
            ("SO_ORIG_OFF", SO_ORIG_OFF),
            ("ST_VAL_OFF", ST_VAL_OFF),
            ("ST_WR", ST_WR),
            ("ST_WC", ST_WC),
            ("ST_A", ST_A),
            ("ST_B", ST_B),
            ("ST_COPY_OFF", ST_COPY_OFF),
            ("ST_PAIR_PTR", ST_PAIR_PTR),
            ("OBS_KEPT", OBS_KEPT),
            ("SK_T_PTR", SK_T_PTR),
            ("SE_PAIR_PTR", SE_PAIR_PTR),
            ("PX_REC", PX_REC),
            ("PX_W_OFF", PX_W_OFF),
            ("PX_C_OFF", PX_C_OFF),
            ("PX_DIMS", PX_DIMS),
            ("KO_REC", KO_REC),
            ("KO_DIMS", KO_DIMS),
            ("KO_C_OFF", KO_C_OFF),
            ("KO_Z_OFF", KO_Z_OFF),
            ("KT_REC", KT_REC),
            ("KT_VAL_OFF", KT_VAL_OFF),
            ("KT_DIMS", KT_DIMS),
            ("KT_X_OFF", KT_X_OFF),
            ("SK_OBS_PTR", SK_OBS_PTR),
            ("SK_OBS", SK_OBS),
            ("SK_S_OFF", SK_S_OFF),
            ("SK_ORIG_OFF", SK_ORIG_OFF),
            ("SK_W", SK_W),
            ("SB_ITEMS", SB_ITEMS),
            ("R_BNORM", ST_BNORM),
            ("R_TOL", ST_TOL),
            ("R_RNORM", ST_RNORM),
            ("F_ACTIVE", F_ACTIVE),
            ("F_STEPS", F_STEPS),
            ("F_REVERT", F_REVERT),
        ],
    );
    s
}

#[cfg(all(test, any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple"))))]
mod tests {
    use super::*;
    use crate::GpuPrecision;
    use crate::TestDevice;
    use crate::test_util::{ba_ill_conditioned, ba_like, precisions, rel_residual, wide, wides};
    use arael_faer::schur::{SchurContext, schur_reduce, schur_symbolic};
    use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};

    type Matrix = arael_faer::bsc::SparseBlockColMat<SparseIndex, f64>;
    type Matrix32 = arael_faer::bsc::SparseBlockColMat<SparseIndex, f32>;

    fn device() -> Arc<TestDevice> {
        Arc::new(TestDevice::new(0).unwrap())
    }

    fn err_max(a: &[f64], b: &[f64]) -> f64 {
        let m = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0.0f64, |e, (x, y)| e.max((x - y).abs())) / m
    }

    /// The factor scalars `dev` computes in, each with its refinement and
    /// the residual a solve reaches: f64 unrefined, f32 refined.
    fn solve_settings(dev: &TestDevice) -> Vec<(GpuPrecision, Option<(usize, f64)>, f64)> {
        precisions(dev)
            .into_iter()
            .map(|p| match p {
                GpuPrecision::F64 => (p, None, 1e-10),
                GpuPrecision::F32 => (p, Some((4usize, 1e-10f64)), 1e-8),
            })
            .collect()
    }

    fn factor(dev: &Arc<TestDevice>, precision: GpuPrecision, sn: &SupernodalSymbolic, s: &SymbolicSparseBlockColMat<SparseIndex>, wide: Wide) -> GpuFactor<TestDevice> {
        GpuFactor::new(dev, precision, sn, s, &Default::default(), wide).unwrap()
    }

    /// A BA-like system, its Schur symbolic and the supernodal analysis of S.
    fn setup(np: usize, nl: usize, seed: u64) -> (Matrix, Vec<f64>, Vec<f64>, Vec<usize>, SchurSymbolic<SparseIndex>, SupernodalSymbolic) {
        let (h, dense, rhs, elim) = ba_like(np, nl, seed);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        (h, dense, rhs, elim, sym, sn)
    }

    #[test]
    fn the_reduction_matches_the_cpu() {
        for wide in wides(&*device()) {
            the_reduction_matches_the_cpu_in(wide);
        }
    }

    /// The forming of S fits the shared memory: its default units per block
    /// lowered first, then its lane groups; a value set that does not fit,
    /// or one unit of one group that does not, is a limit error.
    #[test]
    fn the_tiles_fit_the_shared_memory_or_say_why_not() {
        let ba = TileWidths { kw_cap: 16, ew_cap: 4, kw_max: 9, ew_max: 3, lanes: 32, elem: 8 };
        assert_eq!(tiles_shared_bytes(&ba, 1, 1), 1792);
        // f32 products: half
        assert_eq!(tiles_shared_bytes(&TileWidths { elem: 4, ..ba }, 1, 1), 896);
        let b = |w: &TileWidths, warps, groups| tiles_shared_bytes(w, warps, groups) as u32;
        assert_eq!(fit_tiles(&ba, (4, 2), (false, false), 32 * 1024).unwrap(), (4, 2, b(&ba, 4, 2)), "fits as wanted");
        // kept blocks of 20 columns, eliminated of 32: a unit of two groups
        // over 32 KB, of one under it
        let wide = TileWidths { kw_cap: 32, ew_cap: 32, kw_max: 20, ew_max: 20, lanes: 32, elem: 8 };
        assert!(b(&wide, 1, 2) > 32 * 1024 && b(&wide, 1, 1) < 32 * 1024);
        assert_eq!(fit_tiles(&wide, (4, 2), (false, false), 32 * 1024).unwrap(), (1, 1, b(&wide, 1, 1)), "units, then groups");
        assert_eq!(fit_tiles(&wide, (4, 2), (false, false), 48 * 1024).unwrap(), (1, 2, b(&wide, 1, 2)), "units alone");
        assert_eq!(fit_tiles(&wide, (2, 2), (false, false), 100 * 1024).unwrap(), (2, 2, b(&wide, 2, 2)));
        // set values are not lowered
        let e = fit_tiles(&wide, (4, 2), (true, false), 48 * 1024).unwrap_err();
        assert!(matches!(&e, GpuError::Limit(m) if m.contains("tile_warps 4 and tile_groups 1")), "{e}");
        let e = fit_tiles(&wide, (4, 2), (false, true), 32 * 1024).unwrap_err();
        assert!(matches!(&e, GpuError::Limit(m) if m.contains("tile_warps 1 and tile_groups 2")), "{e}");
        // blocks of 40 columns: one unit of one group over 64 KB
        let widest = TileWidths { kw_cap: 64, ew_cap: 64, kw_max: 40, ew_max: 40, lanes: 32, elem: 8 };
        let e = fit_tiles(&widest, (1, 1), (false, false), 64 * 1024).unwrap_err();
        assert!(
            matches!(&e, GpuError::Limit(m) if m.contains("one unit of one lane group") && m.contains("kept blocks of 40 and eliminated blocks of 40 columns")),
            "{e}"
        );
    }

    /// A checkerboard grid of `k x k` blocks (`k` odd), the even blocks `ew`
    /// columns wide and eliminated, the odd ones `kw` wide and kept: no two
    /// eliminated blocks touch.
    fn checkerboard(k: usize, ew: usize, kw: usize) -> (Matrix, Vec<f64>, SchurSymbolic<SparseIndex>) {
        assert!(k % 2 == 1, "an odd side keeps the block index's parity the cell's");
        let (h, _, rhs) = crate::test_util::grid_spd(k, &[ew, kw], 7);
        let elim: Vec<usize> = (0..k * k).step_by(2).collect();
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        (h, rhs, sym)
    }

    /// Kept blocks of 20 columns against eliminated ones of 10 (`KW` 32,
    /// `EW` 16) reduce as the CPU does, in the shared memory the device
    /// has; units per block set past it are a limit error.
    #[test]
    fn wide_blocks_reduce_in_the_shared_memory_the_device_has() {
        let (h, rhs, sym) = checkerboard(5, 10, 20);
        let mut s_cpu = sym.alloc_s::<f64>();
        let mut rhs_cpu = vec![0.0f64; sym.kept_size()];
        schur_reduce(&sym, &h, &rhs, &mut SchurContext::new(), &mut s_cpu, &mut rhs_cpu).unwrap();
        let dev = device();
        let limit = dev.arch().shared_bytes;
        for wide in wides(&*dev) {
            let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            assert!(g.tile_shared <= limit, "{} bytes, the device {limit}", g.tile_shared);
            g.reduce(h.vals(), &rhs).unwrap();
            let es = err_max(&g.download().unwrap().0, s_cpu.vals());
            assert!(es < 1e-12, "{wide:?}, {} units of {} groups: S err/max {es}", g.tile_warps(), g.tile_groups());
        }
        let w = TileWidths { kw_cap: 32, ew_cap: 16, kw_max: 20, ew_max: 10, lanes: dev.arch().warp.max(PLANE), elem: 8 };
        if w.lanes == dev.arch().warp {
            let warps = limit / tiles_shared_bytes(&w, 1, 1) as u32 + 1;
            let params = GpuSchurParams { tile_warps: Some(warps), tile_groups: Some(1), ..Default::default() };
            let e = GpuSchur::with_params(&dev, &sym, h.symbolic(), wide(&*dev), &params).err().expect("a limit error");
            assert!(matches!(&e, GpuError::Limit(m) if m.contains(&format!("tile_warps {warps}"))), "{e}");
        }
    }

    /// Blocks of 40 columns (`KW` and `EW` 64): one unit of one group takes
    /// 78336 bytes, a limit error on a device with less shared memory (a
    /// Metal GPU, the CPU device), else a reduction as the CPU's.
    #[test]
    fn the_widest_blocks_reduce_or_are_a_limit_error() {
        let (h, rhs, sym) = checkerboard(3, 40, 40);
        let dev = device();
        let w = TileWidths { kw_cap: 64, ew_cap: 64, kw_max: 40, ew_max: 40, lanes: dev.arch().warp.max(PLANE), elem: 8 };
        let need = tiles_shared_bytes(&w, 1, 1);
        match GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)) {
            Err(GpuError::Limit(m)) => assert!(need > dev.arch().shared_bytes as usize, "{m}"),
            Err(e) => panic!("{e}"),
            Ok(mut g) => {
                assert!(need <= dev.arch().shared_bytes as usize);
                let mut s_cpu = sym.alloc_s::<f64>();
                let mut rhs_cpu = vec![0.0f64; sym.kept_size()];
                schur_reduce(&sym, &h, &rhs, &mut SchurContext::new(), &mut s_cpu, &mut rhs_cpu).unwrap();
                g.reduce(h.vals(), &rhs).unwrap();
                let es = err_max(&g.download().unwrap().0, s_cpu.vals());
                assert!(es < 1e-12, "S err/max {es}");
            }
        }
    }

    fn the_reduction_matches_the_cpu_in(wide: Wide) {
        let (h, _, rhs, elim) = ba_like(40, 300, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let mut s_cpu = sym.alloc_s::<f64>();
        let mut rhs_cpu = vec![0.0f64; sym.kept_size()];
        let mut ctx = SchurContext::new();
        schur_reduce(&sym, &h, &rhs, &mut ctx, &mut s_cpu, &mut rhs_cpu).unwrap();
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        g.reduce(h.vals(), &rhs).unwrap();
        let (s_gpu, rhs_gpu) = g.download().unwrap();
        assert_eq!(s_gpu.len(), s_cpu.vals().len());
        let es = err_max(&s_gpu, s_cpu.vals());
        let er = err_max(&rhs_gpu, &rhs_cpu);
        assert!(es < 1e-12 && er < 1e-12, "S err/max {es}, rhs err/max {er}");
        assert!(sym.obs_trans().iter().any(|&t| t) && sym.obs_trans().iter().any(|&t| !t), "both tile orientations in the fixture");
        // the same from f32 inputs, widened on the host
        let (h32, r32): (Vec<f32>, Vec<f32>) = (h.vals().iter().map(|&v| v as f32).collect(), rhs.iter().map(|&v| v as f32).collect());
        g.reduce(&h32, &r32).unwrap();
        let (s32, _) = g.download().unwrap();
        let h64: Vec<f64> = h32.iter().map(|&v| v as f64).collect();
        let r64: Vec<f64> = r32.iter().map(|&v| v as f64).collect();
        g.reduce(&h64, &r64).unwrap();
        let s64 = g.download().unwrap().0;
        assert_eq!(s32, s64, "f32 inputs are their f64 twins");
        // f32 values (an f32 root's) reduced in the wide scalar: as f32
        // inputs widened, and held as f32
        let mut g32 = GpuSchur::new(&dev, &sym, h.symbolic(), Values::f32(wide)).unwrap();
        g32.reduce(&h32, &r32).unwrap();
        assert_eq!(g32.h_values().unwrap(), h64, "{wide:?}: H in f32");
        let es = err_max(&g32.download().unwrap().0, &s64);
        assert!(es < 1e-13, "{wide:?}: S from f32 values, err/max {es} against the wide inputs'");
    }

    /// Point blocks of condition numbers near 1e8, a direction of each
    /// almost unobserved (as for a point seen along nearly parallel rays)
    /// and bounded by the damping: S and the reduced right-hand side are
    /// the CPU's, and the back-substitution's points solve their rows, to
    /// the wide scalar's rounding.
    #[test]
    fn ill_conditioned_point_blocks_reduce_and_solve_as_on_the_cpu() {
        let dev = device();
        let (h, rhs, elim) = ba_ill_conditioned(30, 200, 1e-8, 1e-8, 3);
        let hs = h.symbolic();
        let sym = schur_symbolic(hs, &elim).unwrap();
        let sn = SupernodalSymbolic::new(&sym.s, Some(&nd_block_order(&sym.s)), &SupernodalParams::default()).unwrap();
        let mut s_cpu = sym.alloc_s::<f64>();
        let mut rhs_cpu = vec![0.0f64; sym.kept_size()];
        schur_reduce(&sym, &h, &rhs, &mut SchurContext::new(), &mut s_cpu, &mut rhs_cpu).unwrap();
        for wide in wides(&*dev) {
            // multiplying by D's rounded inverse: 1e-9 in f64, 1e-7 in df32
            let tol = if wide == Wide::F64 { 1e-14 } else { 1e-12 };
            let mut g = GpuSchur::new(&dev, &sym, hs, wide).unwrap();
            g.reduce(h.vals(), &rhs).unwrap();
            let (s, r) = g.download().unwrap();
            let (es, er) = (err_max(&s, s_cpu.vals()), err_max(&r, &rhs_cpu));
            let (precision, refine, _) = solve_settings(&dev)[0];
            let mut f = factor(&dev, precision, &sn, &sym.s, wide);
            let mut delta = vec![0.0f64; hs.ncols()];
            g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap();
            let eb = point_residual(&h, &sym, &elim, &rhs, &delta);
            assert!(es < tol && er < tol && eb < tol, "{wide:?}: S err/max {es}, rhs err/max {er}, points' residual {eb}");
        }
    }

    /// [`ill_conditioned_point_blocks_reduce_and_solve_as_on_the_cpu`] in
    /// f32, at f32's scales (condition numbers near 1e4): the f32 reduction
    /// is the CPU's f32 reduction's to its rounding and solves.
    #[test]
    fn ill_conditioned_point_blocks_reduce_in_f32_as_on_the_cpu() {
        let dev = device();
        let (h, rhs, elim) = ba_ill_conditioned(30, 200, 1e-3, 1e-4, 3);
        let hs = h.symbolic();
        let sym = schur_symbolic(hs, &elim).unwrap();
        let sn = SupernodalSymbolic::new(&sym.s, Some(&nd_block_order(&sym.s)), &SupernodalParams::default()).unwrap();
        let h32 = Matrix32::new(hs.clone(), h.vals().iter().map(|&v| v as f32).collect());
        let r32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        let mut s_cpu32 = sym.alloc_s::<f32>();
        let mut rhs_cpu32 = vec![0.0f32; sym.kept_size()];
        schur_reduce(&sym, &h32, &r32, &mut SchurContext::new(), &mut s_cpu32, &mut rhs_cpu32).unwrap();
        let (s_cpu32, rhs_cpu32): (Vec<f64>, Vec<f64>) = (s_cpu32.vals().iter().map(|&v| v as f64).collect(), rhs_cpu32.iter().map(|&v| v as f64).collect());
        let h64 = Matrix::new(hs.clone(), h32.vals().iter().map(|&v| v as f64).collect());
        let r64: Vec<f64> = r32.iter().map(|&v| v as f64).collect();
        for wide in wides(&*dev) {
            let params = GpuSchurParams { f32_reduction: true, ..Default::default() };
            let mut g = GpuSchur::with_params(&dev, &sym, hs, wide, &params).unwrap();
            g.reduce(h32.vals(), &r32).unwrap();
            let (s, r) = g.download().unwrap();
            let (es, er) = (err_max(&s, &s_cpu32), err_max(&r, &rhs_cpu32));
            let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
            let mut delta = vec![0.0f64; hs.ncols()];
            g.solve(&mut f, h64.vals(), &r64, None, &mut delta).unwrap();
            let eb = point_residual(&h64, &sym, &elim, &r64, &delta);
            assert!(es < 2e-5 && er < 2e-5 && eb < 1e-6, "{wide:?}: S err/max {es}, rhs err/max {er}, points' residual {eb}");
        }
    }

    /// The worst point's `|D_e x_e - t_e| / (|D_e| |x_e| + |t_e|)` (max
    /// norms), `t_e = b_e - sum C_o^T x_o` over its observers: how well the
    /// step's points solve their rows given its kept blocks.
    fn point_residual(h: &Matrix, sym: &SchurSymbolic<SparseIndex>, elim: &[usize], rhs: &[f64], x: &[f64]) -> f64 {
        let (hs, hv) = (h.symbolic(), h.vals());
        let mut worst = 0.0f64;
        for (slot, &e) in elim.iter().enumerate() {
            let we = sym.elim_w()[slot] as usize;
            let off = hs.col_span(e).start;
            let mut t: Vec<f64> = rhs[off..off + we].to_vec();
            for o in sym.elim_obs_ptr()[slot] as usize..sym.elim_obs_ptr()[slot + 1] as usize {
                let (wo, ca, xo) = (sym.obs_w()[o] as usize, sym.obs_ca_off()[o] as usize, sym.obs_orig_off()[o] as usize);
                for (k, tk) in t.iter_mut().enumerate() {
                    for i in 0..wo {
                        let c = if sym.obs_trans()[o] { hv[ca + k + i * we] } else { hv[ca + i + k * wo] };
                        *tk -= c * x[xo + i];
                    }
                }
            }
            let diag = hs.val_range(sym.elim_diag()[slot] as usize).start;
            let d = |i: usize, j: usize| if i <= j { hv[diag + i + j * we] } else { hv[diag + j + i * we] };
            let xe = &x[off..off + we];
            let (mut res, mut dn) = (0.0f64, 0.0f64);
            for i in 0..we {
                let row: f64 = (0..we).map(|j| d(i, j) * xe[j]).sum();
                res = res.max((row - t[i]).abs());
                dn = dn.max((0..we).map(|j| d(i, j).abs()).sum());
            }
            let xn = xe.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let tn = t.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            worst = worst.max(res / (dn * xn + tn));
        }
        worst
    }

    /// The f32 reduction is the CPU's f32 route: from f32 inputs, S and the
    /// reduced right-hand side are f32 values, the CPU's f32 reduction's to
    /// its rounding and the exact reduction's to single precision, on split
    /// and grouped tiles and on wide blocks, in either wide scalar, the
    /// inputs held in it or in f32; a solve through them solves the system
    /// to single precision.
    #[test]
    fn the_f32_reduction_reduces_as_the_cpu_in_f32() {
        let dev = device();
        let (h, dense, rhs, elim) = ba_like(40, 300, 5);
        let n = h.symbolic().ncols();
        let ba = (h.clone(), rhs, schur_symbolic(h.symbolic(), &elim).unwrap());
        for (fixture, (h, rhs, sym)) in [ba, checkerboard(5, 10, 20)].into_iter().enumerate() {
            let h32 = Matrix32::new(h.symbolic().clone(), h.vals().iter().map(|&v| v as f32).collect());
            let r32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
            let mut s_cpu32 = sym.alloc_s::<f32>();
            let mut rhs_cpu32 = vec![0.0f32; sym.kept_size()];
            schur_reduce(&sym, &h32, &r32, &mut SchurContext::new(), &mut s_cpu32, &mut rhs_cpu32).unwrap();
            let (s_cpu32, rhs_cpu32): (Vec<f64>, Vec<f64>) = (s_cpu32.vals().iter().map(|&v| v as f64).collect(), rhs_cpu32.iter().map(|&v| v as f64).collect());
            // the exact reduction of the f32 inputs
            let h64 = Matrix::new(h.symbolic().clone(), h32.vals().iter().map(|&v| v as f64).collect());
            let r64: Vec<f64> = r32.iter().map(|&v| v as f64).collect();
            let mut s_exact = sym.alloc_s::<f64>();
            let mut rhs_exact = vec![0.0f64; sym.kept_size()];
            schur_reduce(&sym, &h64, &r64, &mut SchurContext::new(), &mut s_exact, &mut rhs_exact).unwrap();
            // the inputs in the wide scalar or in f32 (an f32 root's values)
            let values = wides(&*dev).into_iter().flat_map(|w| [Values::from(w), Values::f32(w)]);
            for values in values {
                let wide = values.wide();
                for (groups, split_pairs) in [(1, 128), (2, 128), (1, 1)] {
                    let params = GpuSchurParams { f32_reduction: true, tile_groups: Some(groups), split_pairs, ..Default::default() };
                    let mut g = GpuSchur::with_params(&dev, &sym, h.symbolic(), values, &params).unwrap();
                    g.reduce(h32.vals(), &r32).unwrap();
                    let (s, r) = g.download().unwrap();
                    let at = (fixture, values, groups, split_pairs);
                    assert!(s.iter().chain(&r).all(|&v| v as f32 as f64 == v), "{at:?}: S and the right-hand side hold f32 values");
                    let (es, er) = (err_max(&s, &s_cpu32), err_max(&r, &rhs_cpu32));
                    assert!(es < 1e-6 && er < 1e-6, "{at:?}: against the CPU's f32 reduction, S err/max {es}, rhs err/max {er}");
                    let (es, er) = (err_max(&s, s_exact.vals()), err_max(&r, &rhs_exact));
                    assert!(es < 1e-6 && er < 1e-6, "{at:?}: against the exact reduction, S err/max {es}, rhs err/max {er}");
                    if fixture == 0 {
                        let sn = SupernodalSymbolic::new(&sym.s, Some(&nd_block_order(&sym.s)), &SupernodalParams::default()).unwrap();
                        let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
                        let mut delta = vec![0.0f64; n];
                        g.solve(&mut f, h64.vals(), &r64, None, &mut delta).unwrap();
                        let res = rel_residual(&dense, n, &delta, &rhs);
                        assert!(res < 1e-5, "{at:?}: residual {res}");
                        if values.is_f32() {
                            assert!(delta.iter().all(|&v| v as f32 as f64 == v), "{at:?}: the step in f32");
                        }
                    }
                }
            }
        }
    }

    /// A refinement measures its residuals against S: refining against an
    /// S formed in f32 is refused.
    #[test]
    #[should_panic(expected = "f32_reduction")]
    fn a_refined_solve_through_the_f32_reduction_panics() {
        let (h, _, rhs, _, sym, sn) = setup(10, 30, 2);
        let dev = device();
        let wide = wide(&*dev);
        let params = GpuSchurParams { f32_reduction: true, ..Default::default() };
        let mut g = GpuSchur::with_params(&dev, &sym, h.symbolic(), wide, &params).unwrap();
        let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
        let mut delta = vec![0.0f64; h.symbolic().ncols()];
        let _ = g.solve(&mut f, h.vals(), &rhs, Some((4, 1e-10)), &mut delta);
    }

    /// Tiles split over a block's units (every tile of more than one pair)
    /// and tiles on one unit each, on 1, 2, 4 and 8 lane groups a unit,
    /// reduce as the CPU does, in both wide scalars. The fixture's
    /// eliminated blocks are 2 and 3 wide, so a step's groups take pairs of
    /// different widths.
    #[test]
    fn split_and_grouped_tiles_reduce_as_the_cpu() {
        let (h, _, rhs, elim) = ba_like(40, 300, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        assert!(sym.elim_w().contains(&2) && sym.elim_w().contains(&3), "eliminated blocks of two widths");
        let mut s_cpu = sym.alloc_s::<f64>();
        let mut rhs_cpu = vec![0.0f64; sym.kept_size()];
        schur_reduce(&sym, &h, &rhs, &mut SchurContext::new(), &mut s_cpu, &mut rhs_cpu).unwrap();
        let dev = device();
        for wide in wides(&*device()) {
            for split_pairs in [1, u32::MAX] {
                for groups in [1, 2, 4, 8] {
                    let params = GpuSchurParams { split_pairs, tile_groups: Some(groups), ..Default::default() };
                    let mut g = GpuSchur::with_params(&dev, &sym, h.symbolic(), wide, &params).unwrap();
                    g.reduce(h.vals(), &rhs).unwrap();
                    let es = err_max(&g.download().unwrap().0, s_cpu.vals());
                    assert!(es < 1e-12, "{wide:?}, tiles over {split_pairs} pairs split, {groups} groups: S err/max {es}");
                }
            }
        }
    }

    /// The device writes every pair's expanded record once per structure,
    /// grouped by tile of S in storage order: a tile (a, b) holds every
    /// pair (oa, ob) of one eliminated block's observers with kept(oa) = a
    /// and kept(ob) = b, in no fixed order, as W's offset of oa, the
    /// coupling tile's offset of ob and ob's sizes. The plan counts the
    /// pairs by tile (ST_PAIR_PTR) and names each observer's kept block
    /// (OBS_KEPT), each column's first tile (SK_T_PTR) and each eliminated
    /// block's first pair (SE_PAIR_PTR).
    #[test]
    fn the_device_writes_each_tiles_pairs() {
        for (poses, landmarks, seed) in [(10, 30, 2), (12, 60, 4), (6, 40, 5)] {
            let (h, _, _, elim) = ba_like(poses, landmarks, seed);
            let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
            let p = SchurPlan::reduction(&sym, h.symbolic(), 1, PLAN_MIN_OBSERVERS);
            let w = &p.words;
            let (ssym, kept, eptr, so) = (&sym.s, sym.obs_kept(), sym.elim_obs_ptr(), w[SO] as usize);
            let rec = |oa: usize, ob: usize| [w[so + SO_REC * oa + SO_W_OFF], w[so + SO_REC * ob + SO_CA_OFF], w[so + SO_REC * ob + SO_DIMS]];
            let mut want: std::collections::HashMap<(usize, usize), Vec<[u32; PX_REC]>> = std::collections::HashMap::new();
            for slot in 0..eptr.len() - 1 {
                for i in eptr[slot] as usize..eptr[slot + 1] as usize {
                    for j in i..eptr[slot + 1] as usize {
                        want.entry((kept[i] as usize, kept[j] as usize)).or_default().push(rec(i, j));
                    }
                }
            }
            let at = |slot: usize, n: usize| &w[w[slot] as usize..][..n];
            assert!(at(OBS_KEPT, p.no as usize).iter().zip(kept).all(|(&a, &b)| a == b as u32));
            let t_ptr: Vec<u32> = (0..ssym.nblk_cols()).map(|b| ssym.col_range(b).start as u32).chain([p.nt]).collect();
            assert_eq!(at(SK_T_PTR, p.nk as usize + 1), &t_ptr[..]);
            let e_ptr: Vec<u32> = std::iter::once(0)
                .chain((0..p.ne as usize).scan(0u32, |s, e| {
                    let m = eptr[e + 1] - eptr[e];
                    *s += m * (m + 1) / 2;
                    Some(*s)
                }))
                .collect();
            assert_eq!(at(SE_PAIR_PTR, p.ne as usize + 1), &e_ptr[..]);
            let ptr = at(ST_PAIR_PTR, p.nt as usize + 1);
            assert!(p.np > 100 && ptr[p.nt as usize] as usize == p.np, "pairs: {}", p.np);
            let dev = device();
            let g = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
            let mut px = vec![0u32; PX_REC * p.np];
            dev.read(&g.expanded, 0, bytemuck::cast_slice_mut(&mut px)).unwrap();
            for b in 0..ssym.nblk_cols() {
                for t in ssym.col_range(b) {
                    let mut got: Vec<[u32; PX_REC]> = px[PX_REC * ptr[t] as usize..PX_REC * ptr[t + 1] as usize].chunks(PX_REC).map(|r| [r[0], r[1], r[2]]).collect();
                    let mut expect = want.remove(&(ssym.blk_row(t), b)).unwrap_or_default();
                    got.sort_unstable();
                    expect.sort_unstable();
                    assert_eq!(got, expect, "tile {t} ({}, {b})", ssym.blk_row(t));
                }
            }
            assert!(want.is_empty(), "every pair's tile is a tile of S");
        }
    }

    /// The device expands every entry of the kept blocks' observer lists
    /// once per structure: the observer's sizes, its coupling tile's offset,
    /// its eliminated block's z offset.
    #[test]
    fn the_device_expands_each_kept_observer_into_its_words() {
        let (h, _, _, elim) = ba_like(10, 30, 2);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let p = SchurPlan::reduction(&sym, h.symbolic(), 1, PLAN_MIN_OBSERVERS);
        let w = &p.words;
        let so = w[SO] as usize;
        let se = w[SE] as usize;
        let kobs = &w[w[SK_OBS] as usize..][..p.no as usize];
        let want_kobs: Vec<u32> = kobs
            .iter()
            .flat_map(|&o| {
                let r = so + SO_REC * o as usize;
                [w[r + SO_DIMS], w[r + SO_CA_OFF], w[se + SE_REC * w[r + SO_E] as usize + SE_Z_OFF]]
            })
            .collect();
        let dev = device();
        let g = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
        let mut got = vec![0u32; KO_REC * p.no as usize];
        dev.read(&g.expanded, g.kobs_at, bytemuck::cast_slice_mut(&mut got)).unwrap();
        assert_eq!(got, want_kobs);
    }

    /// The device expands a kept block's tile entries once per structure:
    /// each tile of S touching the block, once, its value offset, sizes and
    /// the other block's offset in the reduced vector, transposed when the
    /// block is the tile's column block, a diagonal tile marked.
    #[test]
    fn the_device_expands_a_kept_blocks_tile_entries_into_records() {
        let (h, _, _, elim) = ba_like(10, 30, 2);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let p = SchurPlan::reduction(&sym, h.symbolic(), 1, PLAN_MIN_OBSERVERS);
        let (nt, nk) = (p.nt as usize, p.nk as usize);
        let at = |slot: usize, i: usize| p.words[p.words[slot] as usize + i];
        let (ptr, _) = tiles_by_kept_block(&p);
        let dev = device();
        let g = GpuSchur::new(&dev, &sym, h.symbolic(), wide(&*dev)).unwrap();
        let mut kt = vec![0u32; KT_REC * ptr[nk] as usize];
        dev.read(&g.expanded, g.kt_at, bytemuck::cast_slice_mut(&mut kt)).unwrap();
        let mut seen = vec![0u32; nt];
        for k in 0..nk {
            for q in ptr[k] as usize..ptr[k + 1] as usize {
                let r = &kt[KT_REC * q..][..KT_REC];
                let t = (0..nt).find(|&t| at(ST_VAL_OFF, t) == r[KT_VAL_OFF]).expect("a tile's value offset");
                let (a, b) = (at(ST_A, t) as usize, at(ST_B, t) as usize);
                let trans = k != a;
                assert_eq!(k, if trans { b } else { a });
                assert_eq!(r[KT_DIMS], kt_dims(at(ST_WR, t), at(ST_WC, t), trans, a == b));
                assert_eq!(r[KT_X_OFF], at(SK_S_OFF, if trans { a } else { b }));
                seen[t] += 1;
            }
        }
        let diag = (0..nt).filter(|&t| at(ST_A, t) == at(ST_B, t)).count();
        assert!(diag == nk && nt > 2 * nk, "tiles {nt}, diagonal {diag}");
        for t in 0..nt {
            assert_eq!(seen[t], if at(ST_A, t) == at(ST_B, t) { 1 } else { 2 }, "tile {t}");
        }
    }

    /// The plan built on threads, the eliminated blocks cut into ranges
    /// of a few observers, is the plan one thread builds.
    #[test]
    fn the_plan_on_threads_is_the_plan_on_one() {
        let (h, _, _, elim) = ba_like(40, 3000, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let one = SchurPlan::reduction(&sym, h.symbolic(), 1, PLAN_MIN_OBSERVERS);
        let key = |p: &SchurPlan| (p.words.clone(), p.elim_blk.clone(), p.dfac_len, p.z_len, p.w_len, p.ew_max, p.kw_max, p.np);
        for (threads, min_obs) in [(2, 1), (3, 100), (8, 1), (64, 1)] {
            let many = SchurPlan::reduction(&sym, h.symbolic(), threads, min_obs);
            assert!(key(&many) == key(&one), "{threads} threads, {min_obs} observers at least");
        }
    }

    /// The plan's tables: a record per eliminated block and one more for
    /// the end of the last one's observers; a record per observer, its
    /// sizes in one word; per tile of S the count of its pairs (two
    /// observers of one eliminated block, of its row and column blocks).
    #[test]
    fn the_plan_holds_a_record_per_block_and_observer_and_counts_the_pairs() {
        assert_eq!(so_dims(9, 3, true), 0x10309);
        assert_eq!(so_dims(6, 3, false), 0x306);
        let (h, _, _, elim) = ba_like(10, 30, 2);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let p = SchurPlan::reduction(&sym, h.symbolic(), 1, PLAN_MIN_OBSERVERS);
        let (ne, no, nt, np) = (p.ne as usize, p.no as usize, p.nt as usize, sym.pair_count());
        let w = &p.words;
        // the words were reserved for, not grown into
        assert_eq!(w.capacity(), w.len());
        let whole = SchurPlan::whole(h.symbolic());
        assert_eq!(whole.words.capacity(), whole.words.len());
        let se = &w[w[SE] as usize..][..SE_REC * (ne + 1)];
        let so = &w[w[SO] as usize..][..SO_REC * no];
        for (e, r) in se.chunks_exact(SE_REC).enumerate() {
            assert_eq!(r[SE_OBS_PTR], sym.elim_obs_ptr()[e]);
            if e < ne {
                assert_eq!(r[SE_W], sym.elim_w()[e]);
            }
        }
        assert_eq!(se[SE_REC * ne + SE_OBS_PTR] as usize, no, "the record past the last block: the observers' end");
        let mut w_len = 0;
        for (o, r) in so.chunks_exact(SO_REC).enumerate() {
            let e = r[SO_E] as usize;
            assert!((se[SE_REC * e + SE_OBS_PTR]..se[SE_REC * (e + 1) + SE_OBS_PTR]).contains(&(o as u32)), "observer {o} is one of block {e}'s");
            assert_eq!(r[SO_DIMS], so_dims(sym.obs_w()[o], sym.elim_w()[e], sym.obs_trans()[o]));
            assert_eq!((r[SO_CA_OFF], r[SO_ORIG_OFF], r[SO_W_OFF]), (sym.obs_ca_off()[o], sym.obs_orig_off()[o], w_len));
            w_len += sym.obs_w()[o] * sym.elim_w()[e];
        }
        let at = |slot: usize| &w[w[slot] as usize..];
        let (ptr, a, b) = (at(ST_PAIR_PTR), at(ST_A), at(ST_B));
        assert_eq!((ptr[0], ptr[nt] as usize), (0, np));
        let (kept, eptr) = (sym.obs_kept(), sym.elim_obs_ptr());
        let mut count: std::collections::HashMap<(u32, u32), u32> = std::collections::HashMap::new();
        for e in 0..ne {
            for i in eptr[e] as usize..eptr[e + 1] as usize {
                for j in i..eptr[e + 1] as usize {
                    *count.entry((kept[i] as u32, kept[j] as u32)).or_default() += 1;
                }
            }
        }
        for t in 0..nt {
            assert_eq!(ptr[t + 1] - ptr[t], count.get(&(a[t], b[t])).copied().unwrap_or(0), "tile {t}'s pairs");
        }
    }

    /// Every tile of S is in one block of `k_schur_tiles`: those of more
    /// than the split's pairs alone on a block, the others on one unit, the
    /// tiles by pair count, most first; with one unit a block, a block per
    /// tile in storage order.
    #[test]
    fn the_tile_blocks_hold_every_tile_once() {
        let (h, _, _, elim) = ba_like(40, 300, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let p = SchurPlan::reduction(&sym, h.symbolic(), 1, PLAN_MIN_OBSERVERS);
        let ptr = &p.words[p.words[ST_PAIR_PTR] as usize..][..p.nt as usize + 1];
        let pairs = |t: u32| ptr[t as usize + 1] - ptr[t as usize];
        let split_pairs = 20;
        let units = 4;
        let blocks = tile_blocks(&p.words, p.nt as usize, units, split_pairs);
        let mut seen = vec![0u32; p.nt as usize];
        let mut order = Vec::new();
        for item in blocks.chunks(units + 1) {
            if item[0] != 0 {
                assert!(pairs(item[1]) > split_pairs && item[2..].iter().all(|&t| t == NONE), "split item {item:?}");
                seen[item[1] as usize] += 1;
                order.push(pairs(item[1]));
            } else {
                for &t in item[1..].iter().filter(|&&t| t != NONE) {
                    assert!(pairs(t) <= split_pairs, "tile {t} of {} pairs on one unit", pairs(t));
                    seen[t as usize] += 1;
                    order.push(pairs(t));
                }
            }
        }
        assert!(seen.iter().all(|&c| c == 1), "every tile once");
        assert!(order.windows(2).all(|x| x[0] >= x[1]), "tiles by pair count, most first");
        assert!(blocks.chunks(units + 1).any(|it| it[0] != 0), "a split tile in the fixture");
        // one unit: a block per tile, in storage order
        let one = tile_blocks(&p.words, p.nt as usize, 1, split_pairs);
        assert_eq!(one, (0..p.nt).flat_map(|t| [0, t]).collect::<Vec<u32>>());
    }

    /// The plan words [`issued_products`] reads, for tiles of (rows,
    /// columns, diagonal, pairs).
    fn tiles_plan(tiles: &[(u32, u32, bool, u32)]) -> Vec<u32> {
        let mut words = vec![0u32; SHEADER];
        let mut ptr = vec![0u32];
        for t in tiles {
            ptr.push(ptr.last().unwrap() + t.3);
        }
        let arrays: [(usize, Vec<u32>); 5] = [
            (ST_WR, tiles.iter().map(|t| t.0).collect()),
            (ST_WC, tiles.iter().map(|t| t.1).collect()),
            (ST_A, (0..tiles.len() as u32).collect()),
            (ST_B, tiles.iter().zip(0u32..).map(|(t, i)| if t.2 { i } else { i + 1 }).collect()),
            (ST_PAIR_PTR, ptr),
        ];
        for (slot, v) in arrays {
            words[slot] = words.len() as u32;
            words.extend(v);
        }
        words
    }

    /// A tile's issued products: its pairs over the groups times its
    /// entries over a group's lanes, both rounded up; a diagonal tile's
    /// entries its upper ones.
    #[test]
    fn the_issued_products_count_steps_and_lane_passes() {
        let issued = |tile, groups| issued_products(&tiles_plan(&[tile]), 1, 32, groups);
        assert_eq!(issued((6, 6, false, 4), 1), 4 * 2);
        assert_eq!(issued((6, 6, false, 4), 2), 2 * 3);
        assert_eq!(issued((6, 6, false, 1), 1), 2);
        assert_eq!(issued((6, 6, false, 1), 2), 3);
        // 21 upper entries
        assert_eq!(issued((6, 6, true, 4), 1), 4);
        assert_eq!(issued((6, 6, true, 4), 2), 2 * 2);
        assert_eq!(issued((6, 6, false, 0), 2), 0);
    }

    /// Two lane groups for 6 x 6 tiles of many pairs (SLAM), one for 9 x 9
    /// tiles (bundle adjustment) and for tiles of one pair.
    #[test]
    fn two_groups_take_small_tiles_of_many_pairs() {
        let groups = |tile| tile_groups(&tiles_plan(&[tile; 10]), 10, 32);
        assert_eq!(groups((6, 6, false, 40)), 2);
        assert_eq!(groups((9, 9, false, 40)), 1);
        assert_eq!(groups((6, 6, false, 1)), 1);
        // no pairs, as in a whole system's plan
        assert_eq!(groups((6, 6, false, 0)), 1);
    }

    /// A tile of S on 64 lanes, as on a 64-lane wavefront, is the tile on
    /// 32, with one lane group a unit and with several, to rounding: each
    /// build orders a tile's pairs its own way.
    #[test]
    fn a_tile_on_64_lanes_is_the_tile_on_32() {
        let (h, _, rhs, elim) = ba_like(40, 300, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let dev = device();
        for wide in wides(&*device()) {
            for groups in [1, 4] {
                let reduce = |lanes| {
                    let p = SchurPlan::reduction(&sym, h.symbolic(), 1, PLAN_MIN_OBSERVERS);
                    let params = GpuSchurParams { tile_groups: Some(groups), ..Default::default() };
                    let mut g = GpuSchur::build_lanes(&dev, p, h.symbolic().val_count(), wide.into(), lanes, &params).unwrap();
                    g.reduce(h.vals(), &rhs).unwrap();
                    g.download().unwrap()
                };
                let ((s64, r64), (s32, r32)) = (reduce(64), reduce(32));
                let (es, er) = (err_max(&s64, &s32), err_max(&r64, &r32));
                assert!(es < 1e-12 && er < 1e-12, "{wide:?}, {groups} groups: S {es}, rhs {er}");
            }
        }
    }

    #[test]
    fn the_reduced_solve_solves_the_whole_system() {
        for wide in wides(&*device()) {
            the_reduced_solve_solves_the_whole_system_in(wide);
        }
    }

    fn the_reduced_solve_solves_the_whole_system_in(wide: Wide) {
        let (h, dense, rhs, _, sym, sn) = setup(40, 300, 9);
        let n = h.symbolic().ncols();
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        for (precision, refine, tol) in solve_settings(&dev) {
            let mut f = factor(&dev, precision, &sn, &sym.s, wide);
            let mut delta = vec![0.0f64; n];
            let stats = g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap();
            let res = rel_residual(&dense, n, &delta, &rhs);
            assert!(res < tol, "{precision:?}: residual {res}");
            if refine.is_some() {
                assert!(stats.steps > 0 && !stats.short && stats.rnorm <= 1e-10 * stats.bnorm, "{stats:?}");
            } else {
                assert_eq!(stats.steps, 0);
            }
            // a second solve of the same structure
            let mut delta2 = vec![0.0f64; n];
            g.solve(&mut f, h.vals(), &rhs, refine, &mut delta2).unwrap();
            assert!(rel_residual(&dense, n, &delta2, &rhs) < tol);
        }
    }

    /// With nothing eliminated S is H: the reduction solves the whole
    /// system, refinement included.
    #[test]
    fn nothing_eliminated_solves_the_whole_system() {
        for wide in wides(&*device()) {
            nothing_eliminated_solves_the_whole_system_in(wide);
        }
    }

    /// An f32 root's whole system (f32 values, the f32 reduction): S is H
    /// in f32, and the f32 factor solves it to single precision, the step
    /// in f32.
    #[test]
    fn an_f32_whole_system_solves_in_f32() {
        let (h, dense, rhs, _) = ba_like(30, 200, 5);
        let n = h.symbolic().ncols();
        let h32: Vec<f32> = h.vals().iter().map(|&v| v as f32).collect();
        let r32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
        let sn = SupernodalSymbolic::new(h.symbolic(), Some(&nd_block_order(h.symbolic())), &SupernodalParams::default()).unwrap();
        let dev = device();
        for wide in wides(&*dev) {
            let params = GpuSchurParams { f32_reduction: true, ..Default::default() };
            let mut g = GpuSchur::whole_with_params(&dev, h.symbolic(), Values::f32(wide), &params).unwrap();
            assert!(g.f32_reduction() && g.reduced().is_f32());
            g.reduce(&h32, &r32).unwrap();
            let (s, r) = g.download().unwrap();
            assert!(s.iter().chain(&r).all(|&v| v as f32 as f64 == v), "{wide:?}: S and the right-hand side in f32");
            assert_eq!(r, r32.iter().map(|&v| v as f64).collect::<Vec<_>>(), "{wide:?}: the right-hand side is the gradient");
            let mut f = factor(&dev, GpuPrecision::F32, &sn, h.symbolic(), wide);
            let mut delta = vec![0.0f32; n];
            g.solve(&mut f, &h32, &r32, None, &mut delta).unwrap();
            let delta: Vec<f64> = delta.iter().map(|&v| v as f64).collect();
            let res = rel_residual(&dense, n, &delta, &rhs);
            assert!(res < 1e-5, "{wide:?}: residual {res}");
        }
    }

    fn nothing_eliminated_solves_the_whole_system_in(wide: Wide) {
        let (h, dense, rhs, _) = ba_like(30, 200, 5);
        let n = h.symbolic().ncols();
        let order = nd_block_order(h.symbolic());
        let sn = SupernodalSymbolic::new(h.symbolic(), Some(&order), &SupernodalParams::default()).unwrap();
        let dev = device();
        let mut g = GpuSchur::whole(&dev, h.symbolic(), wide).unwrap();
        for (precision, refine, tol) in solve_settings(&dev) {
            let mut f = factor(&dev, precision, &sn, h.symbolic(), wide);
            for _ in 0..2 {
                let mut delta = vec![0.0f64; n];
                let stats = g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap();
                let res = rel_residual(&dense, n, &delta, &rhs);
                assert!(res < tol, "{precision:?}: residual {res}");
                if refine.is_some() {
                    assert!(stats.steps > 0 && !stats.short, "{stats:?}");
                }
            }
        }
    }

    /// The refinement decides on the device: it stops once below the
    /// tolerance, launched steps after that doing nothing; with a tolerance
    /// it cannot reach it stops at the first step that does not reduce the
    /// residual, and undoes that step.
    #[test]
    fn the_refinement_stops_on_the_device() {
        for wide in wides(&*device()) {
            the_refinement_stops_on_the_device_in(wide);
        }
    }

    /// A residual norm that is not a number is not a converged solve: a
    /// right-hand side with a NaN leaves the refinement short of its
    /// tolerance, whatever the comparisons with a NaN say.
    #[test]
    fn a_nan_residual_is_not_a_converged_refinement() {
        for wide in wides(&*device()) {
            let (h, _, mut rhs, _, sym, sn) = setup(40, 300, 9);
            let n = h.symbolic().ncols();
            let dev = device();
            let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
            rhs[n / 2] = f64::NAN;
            let mut delta = vec![0.0f64; n];
            let stats = g.solve(&mut f, h.vals(), &rhs, Some((12, 1e-10)), &mut delta).unwrap();
            assert!(stats.short && stats.rnorm.is_nan(), "{wide:?}: {stats:?}");
        }
    }

    fn the_refinement_stops_on_the_device_in(wide: Wide) {
        let (h, dense, rhs, _, sym, sn) = setup(40, 300, 9);
        let n = h.symbolic().ncols();
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
        let mut delta = vec![0.0f64; n];
        let reached = g.solve(&mut f, h.vals(), &rhs, Some((12, 1e-10)), &mut delta).unwrap();
        assert!(reached.steps > 0 && reached.steps < 12 && !reached.short, "{reached:?}");
        let res = rel_residual(&dense, n, &delta, &rhs);
        assert!(res < 1e-8, "residual {res}");
        let mut delta0 = vec![0.0f64; n];
        let stalled = g.solve(&mut f, h.vals(), &rhs, Some((40, 0.0)), &mut delta0).unwrap();
        assert!(stalled.short && stalled.steps > reached.steps && stalled.steps < 40, "{stalled:?}");
        assert!(stalled.rnorm <= reached.rnorm, "{stalled:?} against {reached:?}");
        let res0 = rel_residual(&dense, n, &delta0, &rhs);
        assert!(res0 < 1e-8, "residual {res0} after the undone step");
    }

    /// The kernels whose lanes are a kept block's rows give the same sums
    /// over any number of warps a block: the reduced right-hand side is the
    /// CPU's, and a refined solve (its residuals by `k_symv_sub`) solves
    /// the system, with 1 warp, with a count that is no power of two, and
    /// with as many as leave most warps without an item.
    #[test]
    fn the_kept_blocks_kernels_sum_the_same_over_any_warps() {
        let (h, dense, rhs, _, sym, sn) = setup(40, 300, 9);
        let n = h.symbolic().ncols();
        let mut s_cpu = sym.alloc_s::<f64>();
        let mut rhs_cpu = vec![0.0f64; sym.kept_size()];
        schur_reduce(&sym, &h, &rhs, &mut SchurContext::new(), &mut s_cpu, &mut rhs_cpu).unwrap();
        let dev = device();
        let wide = wide(&*dev);
        for warps in [1u32, 3, 8, 32] {
            let params = GpuSchurParams { kept_warps: Some(warps), ..Default::default() };
            let mut g = GpuSchur::with_params(&dev, &sym, h.symbolic(), wide, &params).unwrap();
            g.reduce(h.vals(), &rhs).unwrap();
            let er = err_max(&g.download().unwrap().1, &rhs_cpu);
            assert!(er < 1e-12, "{warps} warps: rhs err/max {er}");
            let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
            let mut delta = vec![0.0f64; n];
            let stats = g.solve(&mut f, h.vals(), &rhs, Some((12, 1e-10)), &mut delta).unwrap();
            assert!(stats.steps > 0 && !stats.short, "{warps} warps: {stats:?}");
            let res = rel_residual(&dense, n, &delta, &rhs);
            assert!(res < 1e-8, "{warps} warps: residual {res}");
        }
    }

    /// The warps a kept block takes: about 8 items a group of lanes, a
    /// power of two from 1 to the most asked.
    #[test]
    fn the_kept_blocks_warps_follow_their_items() {
        // 372 cameras of 9 parameters (two groups a warp): 147 tiles of S
        // and 550 observers a block
        assert_eq!(kept_warps_for(54692, 372, 16, 8), 8);
        assert_eq!(kept_warps_for(550 * 372, 372, 16, 32), 32);
        assert_eq!(kept_warps_for(394 * 1723, 1723, 16, 32), 16);
        // a SLAM map's 900 poses of 6 (four groups a warp), 145 observers each
        assert_eq!(kept_warps_for(130823, 900, 8, 32), 4);
        assert_eq!(kept_warps_for(40 * 100, 100, 16, 8), 2);
        assert_eq!(kept_warps_for(5 * 100, 100, 8, 8), 1);
        assert_eq!(kept_warps_for(0, 0, 4, 8), 1);
    }

    /// `h2` and `dense2`: the diagonal of every diagonal tile scaled by 1.5.
    fn damped(h: &Matrix, dense: &[f64]) -> (Matrix, Vec<f64>) {
        let hs = h.symbolic().clone();
        let n = hs.ncols();
        let mut h2 = h.clone();
        for blk in 0..hs.nblk_cols() {
            let t = hs.col_range(blk).find(|&b| hs.blk_row(b) == blk).unwrap();
            let (w, _) = hs.block_dims(t);
            let base = hs.val_range(t).start;
            for i in 0..w {
                h2.vals_mut()[base + i * w + i] *= 1.5;
            }
        }
        let mut dense2 = dense.to_vec();
        for i in 0..n {
            dense2[i + i * n] *= 1.5;
        }
        (h2, dense2)
    }

    /// The second solve with a factor and a refinement setting replays the
    /// graph the first captured, against new values, and lands exactly where
    /// the launched kernels land; a new factor or another refinement setting
    /// is captured anew.
    #[test]
    fn replayed_solves_match_launched_ones() {
        for wide in wides(&*device()) {
            replayed_solves_match_launched_ones_in(wide);
        }
    }

    fn replayed_solves_match_launched_ones_in(wide: Wide) {
        let (h, dense, rhs, _, sym, sn) = setup(40, 300, 9);
        let n = h.symbolic().ncols();
        let dev = device();
        // another damping: the replay must see the values written before it
        let (h2, dense2) = damped(&h, &dense);
        for precision in precisions(&*dev) {
            let refine = (precision == GpuPrecision::F32).then_some((4, 1e-10));
            let mut launched = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            launched.use_graphs(false);
            let mut fl = factor(&dev, precision, &sn, &sym.s, wide);
            let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            let mut f = factor(&dev, precision, &sn, &sym.s, wide);
            for (round, (hm, dm)) in [(&h, &dense), (&h2, &dense2), (&h, &dense)].into_iter().enumerate() {
                let (mut a, mut b) = (vec![0.0f64; n], vec![0.0f64; n]);
                let sa = launched.solve(&mut fl, hm.vals(), &rhs, refine, &mut a).unwrap();
                let sb = g.solve(&mut f, hm.vals(), &rhs, refine, &mut b).unwrap();
                assert!(!launched.replayed());
                assert_eq!(g.replayed(), round > 0, "{precision:?} round {round}");
                assert_eq!(sa.steps, sb.steps);
                let res = rel_residual(dm, n, &b, &rhs);
                assert!(res < 1e-8, "{precision:?} round {round}: residual {res}");
                let dx = a.iter().zip(&b).fold(0.0f64, |m, (x, y)| m.max((x - y).abs())) / a.iter().fold(0.0f64, |m, x| m.max(x.abs()));
                assert!(dx < 1e-12, "{precision:?} round {round}: replay against launches {dx}");
            }
            // another factor of the same structure, then another refinement setting
            let mut f2 = factor(&dev, precision, &sn, &sym.s, wide);
            let mut d = vec![0.0f64; n];
            g.solve(&mut f2, h.vals(), &rhs, refine, &mut d).unwrap();
            assert!(!g.replayed());
            g.solve(&mut f2, h.vals(), &rhs, Some((2, 1e-10)), &mut d).unwrap();
            assert!(!g.replayed());
            g.solve(&mut f2, h.vals(), &rhs, Some((2, 1e-10)), &mut d).unwrap();
            assert!(g.replayed());
            assert!(rel_residual(&dense, n, &d, &rhs) < 1e-8);
        }
    }

    /// The launches alone, from values written at the targets, read back
    /// through the views, give what `solve` gives; and a graph the caller
    /// captures of them replays to the same; so do its three stages
    /// launched one after another.
    #[test]
    fn the_launches_alone_match_solve() {
        for wide in wides(&*device()) {
            the_launches_alone_match_solve_in(wide);
        }
    }

    fn the_launches_alone_match_solve_in(wide: Wide) {
        let (h, _, rhs, _, sym, sn) = setup(30, 250, 4);
        let n = h.symbolic().ncols();
        let dev = device();
        let refine = Some((4usize, 1e-10f64));
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        g.use_graphs(false);
        let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
        let mut want = vec![0.0f64; n];
        let stats = g.solve(&mut f, h.vals(), &rhs, refine, &mut want).unwrap();
        // fresh blocks, the inputs written at the targets
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        let f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
        g.upload(h.vals(), &rhs).unwrap();
        let held: Vec<f64> = h.vals().iter().map(|&v| wide.value(wide.bits(v))).collect();
        assert_eq!(g.h_values().unwrap(), held, "the values as the wide scalar holds them");
        let read_result = |g: &GpuSchur<TestDevice>| {
            let mut step = vec![0.0f64; n];
            let mut bytes = vec![0u8; 8 * n];
            let mut status = vec![0u8; STATUS_BYTES];
            let (b, at) = g.step_view();
            dev.read(b, at, &mut bytes).unwrap();
            wide.decode(&bytes, &mut step);
            let (b, at) = g.status_view();
            dev.read(b, at, &mut status).unwrap();
            (step, g.outcome(&status, refine).unwrap())
        };
        g.launch(&f, refine).unwrap();
        let (step, got) = read_result(&g);
        assert_eq!(got.steps, stats.steps);
        let dx = err_max(&step, &want);
        assert!(dx < 1e-12, "launches against solve: {dx}");
        let graph = dev.capture(|| g.launch(&f, refine)).unwrap();
        dev.replay(&graph).unwrap();
        let (step2, got2) = read_result(&g);
        assert_eq!(got2.steps, stats.steps);
        assert!(err_max(&step2, &want) < 1e-12);
        assert_eq!(g.graph_key(&f, refine), g.graph_key(&f, Some((4, 1e-10))));
        assert_ne!(g.graph_key(&f, refine), g.graph_key(&f, Some((3, 1e-10))));
        // the three stages one after another are the attempt: on fresh
        // blocks, so nothing of the launches above is read
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        let f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
        g.upload(h.vals(), &rhs).unwrap();
        g.launch_reduce().unwrap();
        g.launch_factored(&f, refine).unwrap();
        g.launch_backsub().unwrap();
        let (step3, got3) = read_result(&g);
        assert_eq!(got3.steps, stats.steps);
        assert!(err_max(&step3, &want) < 1e-12, "the stages against solve");
    }

    /// Solves, their captures and replays included, allocate nothing: the
    /// blocks are made when the reduction and the factor are (from the
    /// device's pool once an earlier structure has filled it).
    #[test]
    fn solves_allocate_nothing() {
        for wide in wides(&*device()) {
            solves_allocate_nothing_in(wide);
        }
    }

    fn solves_allocate_nothing_in(wide: Wide) {
        let dev = device();
        for (np, nl, seed) in [(30, 400, 1u64), (60, 900, 2), (20, 200, 3)] {
            let (h, dense, rhs, _, sym, sn) = setup(np, nl, seed);
            let n = h.symbolic().ncols();
            let before = dev.allocations();
            let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
            let fresh = dev.allocations() - before;
            // the plan block, the expanded records and the work block
            if before == 0 {
                assert_eq!(fresh, 3);
            } else {
                assert!(fresh <= 3, "{fresh}");
            }
            let mut f = factor(&dev, GpuPrecision::F32, &sn, &sym.s, wide);
            let made = dev.allocations();
            let mut delta = vec![0.0f64; n];
            for _ in 0..3 {
                g.solve(&mut f, h.vals(), &rhs, Some((4, 1e-10)), &mut delta).unwrap();
                assert!(rel_residual(&dense, n, &delta, &rhs) < 1e-8);
            }
            assert!(g.replayed());
            assert_eq!(dev.allocations(), made, "no block made by a solve");
        }
    }

    /// A pivot that fails, in an eliminated block or in S, comes back with
    /// the solve's result and leaves the step as it was; the next solve of
    /// the same objects starts clean.
    #[test]
    fn bad_pivots_are_reported_with_the_result() {
        for wide in wides(&*device()) {
            bad_pivots_are_reported_with_the_result_in(wide);
        }
    }

    fn bad_pivots_are_reported_with_the_result_in(wide: Wide) {
        let (good, dense, rhs, elim, sym, sn) = setup(10, 30, 2);
        let n = good.symbolic().ncols();
        let asym = good.symbolic().clone();
        let diag0 = |blk: usize| asym.val_range(asym.col_range(blk).find(|&b| asym.blk_row(b) == blk).unwrap()).start;
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, &asym, wide).unwrap();
        for precision in precisions(&*dev) {
            let mut f = factor(&dev, precision, &sn, &sym.s, wide);
            let refine = (precision == GpuPrecision::F32).then_some((4, 1e-10));
            // an eliminated block
            let e = elim[3];
            let mut h = good.clone();
            h.vals_mut()[diag0(e)] = -1.0;
            assert!(matches!(g.reduce(h.vals(), &rhs), Err(GpuSchurError::Eliminated { block }) if block == e));
            let mut delta = vec![7.0f64; n];
            let err = g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap_err();
            assert!(matches!(err, GpuSchurError::Eliminated { block } if block == e), "{err:?}");
            assert!(delta.iter().all(|&v| v == 7.0), "the step is left as it was");
            // a kept block: S(k, k) is H(k, k) less a semidefinite sum
            let k = sym.kept[0] as usize;
            let mut h = good.clone();
            h.vals_mut()[diag0(k)] = -1e8;
            let err = g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap_err();
            assert!(matches!(err, GpuSchurError::Reduced), "{err:?}");
            // and the good matrix again
            g.solve(&mut f, good.vals(), &rhs, refine, &mut delta).unwrap();
            assert!(rel_residual(&dense, n, &delta, &rhs) < 1e-8);
        }
    }

    /// At BA widths (eliminated blocks of 3, kept of 6) the per-thread
    /// arrays stay in registers: no kernel uses local memory (a GPU's), on
    /// 1, 2, 4 and 8 lane groups a unit.
    #[test]
    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn the_kernels_keep_their_arrays_in_registers() {
        for wide in wides(&*device()) {
            the_kernels_keep_their_arrays_in_registers_in(wide);
        }
    }

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn the_kernels_keep_their_arrays_in_registers_in(wide: Wide) {
        let (h, _, _, _, sym, _) = setup(10, 30, 2);
        for groups in [1, 2, 4, 8] {
            let params = GpuSchurParams { tile_groups: Some(groups), ..Default::default() };
            let g = GpuSchur::with_params(&device(), &sym, h.symbolic(), wide, &params).unwrap();
            let k = &g.k;
            for (name, kernel) in [("elim", &k.elim), ("tiles", &k.tiles), ("rhs", &k.rhs), ("backsub", &k.backsub), ("pairs", &k.pairs), ("kobs", &k.kobs), ("kt", &k.kt)] {
                assert_eq!(kernel.local_bytes().unwrap(), 0, "{wide:?} k_schur_{name}, {groups} tile groups: {} registers", kernel.registers().unwrap());
            }
        }
    }

}
