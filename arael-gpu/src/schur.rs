//! [`GpuSchur`]: the Schur reduction on the device, generic over block
//! widths, with the reduced system kept on the device for the factorization,
//! the refinement and the back-substitution.
//!
//! From H and the gradient in the route's [`Values`] (f32 for an f32 root,
//! else the wide scalar), formed in the route's wide scalar ([`Wide`]:
//! `f64` or df32), as an f32 factor's refinement measures its residuals
//! against S, or in f32 for an f32 solve that does not refine, as the CPU's
//! f32 route ([`GpuSchurParams::f32_reduction`]).
//! Per reduction, from the Hessian's tile values and the gradient:
//!
//!   elim      one unit per eliminated block: the Cholesky factor of `D_e`,
//!             and `z_e = D_e^-1 b_e`
//!   w         one unit per observer: `W_o = C_o D_e^-1`
//!   tiles     one plane per tile of S: `S(a, b) = H(a, b) - sum over the
//!             pairs (oa, ob) of its list of W_oa C_ob^T`, the pair lists
//!             built once on the host; no shared write
//!   rhs       one plane per kept block: `b_k - sum over its observers of
//!             C_o z_e`
//!   backsub   one unit per eliminated block: `x_e = D_e^-1 (b_e - sum_a C_a^T x_a)`
//!
//! `D_e^-1` is applied by two triangular solves with its factor, as on the
//! CPU, never as an explicit inverse: multiplying by a rounded inverse loses
//! digits in proportion to `D_e`'s condition number, high for a point seen
//! along nearly parallel rays.
//!
//! A block symv over the tiles of S gives the refinement its residual on the
//! device, and one-block norm kernels decide there whether the next step is
//! due, so a solve with an f32 factor waits only for its result either:
//! [`GpuSchur::solve`] reads the step, the pivot flags and the refinement's
//! state in one copy.

use std::sync::Arc;

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::{SchurReal, SchurSymbolic};

use crate::plan::SparseIndex;
use crate::supernodal::{GpuFactor, Packer};
use crate::{Arg, Device, GpuError, Launch, Layout, Values, WIDE, Wide, define_u32s};

/// No kept-kept tile behind a tile of S.
const NONE: u32 = u32::MAX;

// The header of the reduction's plan buffer: where each array starts.
pub(crate) const SE_DIAG_OFF: usize = 0;
pub(crate) const SE_W: usize = 1;
pub(crate) const SE_ORIG_OFF: usize = 2;
pub(crate) const SE_OBS_PTR: usize = 3;
pub(crate) const SE_DFAC_OFF: usize = 4;
pub(crate) const SE_Z_OFF: usize = 5;
pub(crate) const SO_CA_OFF: usize = 6;
pub(crate) const SO_TRANS: usize = 7;
pub(crate) const SO_W: usize = 8;
pub(crate) const SO_E: usize = 9;
pub(crate) const SO_ORIG_OFF: usize = 10;
pub(crate) const SO_W_OFF: usize = 11;
pub(crate) const ST_VAL_OFF: usize = 12;
pub(crate) const ST_WR: usize = 13;
pub(crate) const ST_WC: usize = 14;
pub(crate) const ST_A: usize = 15;
pub(crate) const ST_B: usize = 16;
pub(crate) const ST_A_OFF: usize = 17;
pub(crate) const ST_B_OFF: usize = 18;
pub(crate) const ST_COPY_OFF: usize = 19;
pub(crate) const ST_PAIR_PTR: usize = 20;
pub(crate) const SP_OA: usize = 21;
pub(crate) const SP_OB: usize = 22;
pub(crate) const SK_OBS_PTR: usize = 23;
pub(crate) const SK_OBS: usize = 24;
pub(crate) const SK_S_OFF: usize = 25;
pub(crate) const SK_ORIG_OFF: usize = 26;
pub(crate) const SK_W: usize = 27;
/// The blocks of `k_schur_tiles` ([`tile_blocks`]), added by
/// [`GpuSchur`]'s build.
pub(crate) const SB_ITEMS: usize = 28;
pub(crate) const SHEADER: usize = 32;

/// How [`GpuSchur`] cuts the forming of S (`k_schur_tiles`), whose
/// lockstep units (the device's lockstep width of threads each) take a tile
/// of S each. The defaults are the measured ones, lowered where the
/// kernel's shared memory would not fit the device ([`Arch::shared_bytes`]:
/// units per block first, then lane groups); a value set here that does
/// not fit is [`GpuError::Limit`].
///
/// [`Arch::shared_bytes`]: crate::Arch::shared_bytes
#[derive(Clone, Copy, Debug)]
pub struct GpuSchurParams {
    /// Units per block; `None` takes the device's choice
    /// ([`Self::tile_warps_on`]).
    pub tile_warps: Option<u32>,
    /// The most pairs of a tile that one unit sums, with more than one unit
    /// per block: a tile with more takes a whole block.
    pub split_pairs: u32,
    /// Lane groups per unit, 1, 2, 4 or 8: each group takes a different pair
    /// of the tile, and the groups' sums are added at the tile's end. More
    /// groups leave fewer lanes idle on small tiles (a 6 x 6 tile on 32
    /// lanes: 36 of 64 lane-products busy with one group, 36 of 48 with
    /// two) and take more registers and shared memory. `None`: two groups
    /// when they issue more than 10% fewer products over the plan's tiles
    /// than one, else one (measured: two on SLAM, one on bundle
    /// adjustment; four slower than two on both).
    pub tile_groups: Option<u32>,
    /// Whether the reduction computes in f32, as the CPU's f32 route: the
    /// eliminated blocks' factors, W, S, the reduced right-hand side, the
    /// kept blocks' solution and the eliminated blocks' step, from the
    /// inputs rounded to f32 (an f32 root's [`Values`] already are), all
    /// stored in f32 (the step in the route's values). For a solve whose
    /// factor is f32 and that does not refine against S: a refined solve
    /// panics. Off: in the wide scalar.
    pub f32_reduction: bool,
}

impl GpuSchurParams {
    /// [`tile_warps`](Self::tile_warps) on the device `arch`, unset: 4 on a
    /// discrete GPU, 1 on an integrated one. Measured on bundle adjustment
    /// and SLAM: on an NVIDIA discrete GPU, 4 units, heavy tiles split over
    /// a block, cut the reduction's tail (`k_schur_tiles` 36% faster on
    /// Ladybug-372, 18% on slam); on an AMD integrated GPU, with nothing to
    /// fill, a block per tile in storage order measured fastest. Discrete
    /// AMD GPUs are not measured.
    ///
    /// ```ignore
    /// let units = GpuSchurParams::default().tile_warps_on(dev.arch());
    /// ```
    pub fn tile_warps_on(&self, arch: &crate::Arch) -> u32 {
        self.tile_warps.unwrap_or(if arch.integrated { 1 } else { 4 })
    }
}

impl Default for GpuSchurParams {
    fn default() -> Self {
        GpuSchurParams { tile_warps: None, split_pairs: 128, tile_groups: None, f32_reduction: false }
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

/// What a refined solve took: steps, the final and the right-hand side's
/// norm, and whether it stayed above the tolerance.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefineStats {
    pub steps: usize,
    pub rnorm: f64,
    pub bnorm: f64,
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
        stats = RefineStats { steps: flag(F_STEPS) as usize, rnorm: st(ST_RNORM), bnorm: st(ST_BNORM), short: st(ST_RNORM) > st(ST_TOL) };
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
}

impl SchurPlan {
    /// The plan of `sym`, the reduction of a matrix with the block structure `hsym`.
    pub fn reduction(sym: &SchurSymbolic<SparseIndex>, hsym: &SymbolicSparseBlockColMat<SparseIndex>) -> Self {
        let (src, dst) = sym.copy_tiles();
        Self::build(
            &Elimination {
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
            },
            hsym,
        )
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
            },
            hsym,
        )
    }

    fn build(sym: &Elimination, hsym: &SymbolicSparseBlockColMat<SparseIndex>) -> Self {
        let ne = sym.elim_diag.len();
        let no = sym.obs_w.len();
        let nk = sym.kept.len();
        let n = hsym.ncols();
        let nks = sym.s.nrows();
        let hpart = hsym.col_part();
        // the eliminated blocks
        let (mut e_diag, mut e_w, mut e_orig, mut e_dfac, mut e_z, mut elim_blk) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut dfac_len, mut z_len) = (0u32, 0u32);
        let mut ew_max = 1usize;
        for slot in 0..ne {
            let dblk = sym.elim_diag[slot] as usize;
            let e = hsym.blk_row(dblk);
            let we = sym.elim_w[slot] as usize;
            ew_max = ew_max.max(we);
            e_diag.push(hsym.val_range(dblk).start as u32);
            e_w.push(we as u32);
            e_orig.push(hpart[e]);
            e_dfac.push(dfac_len);
            e_z.push(z_len);
            elim_blk.push(e as u32);
            dfac_len += (we * we) as u32;
            z_len += we as u32;
        }
        let e_obs_ptr: Vec<u32> = sym.elim_obs_ptr.to_vec();
        // the observers, with their eliminated slot and their W offset
        let mut o_e = vec![0u32; no];
        for slot in 0..ne {
            o_e[e_obs_ptr[slot] as usize..e_obs_ptr[slot + 1] as usize].fill(slot as u32);
        }
        let o_ca: Vec<u32> = sym.obs_ca_off.to_vec();
        let o_trans: Vec<u32> = sym.obs_trans.iter().map(|&t| t as u32).collect();
        let o_w: Vec<u32> = sym.obs_w.to_vec();
        let o_orig: Vec<u32> = sym.obs_orig_off.to_vec();
        let mut o_woff = Vec::with_capacity(no);
        let mut w_len = 0u32;
        let mut kw_max = 1usize;
        for o in 0..no {
            o_woff.push(w_len);
            w_len += o_w[o] * e_w[o_e[o] as usize];
            kw_max = kw_max.max(o_w[o] as usize);
        }
        // the tiles of S, in storage order, with their kept-kept source
        let ssym = sym.s;
        let spart = ssym.col_part();
        let nt = ssym.nblocks();
        let mut copy_of = vec![NONE; nt];
        let (csrc, cdst) = (&sym.copy_src[..], &sym.copy_dst[..]);
        for (&src, &dst) in csrc.iter().zip(cdst) {
            copy_of[dst as usize] = hsym.val_range(src as usize).start as u32;
        }
        let (mut t_val, mut t_wr, mut t_wc, mut t_a, mut t_b, mut t_aoff, mut t_boff, mut t_copy) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for bcol in 0..ssym.nblk_cols() {
            for t in ssym.col_range(bcol) {
                // a tile's index in these arrays is its block index in S
                debug_assert_eq!(t, t_val.len());
                let a = ssym.blk_row(t);
                let (wr, wc) = ssym.block_dims(t);
                kw_max = kw_max.max(wr).max(wc);
                t_val.push(ssym.val_range(t).start as u32);
                t_wr.push(wr as u32);
                t_wc.push(wc as u32);
                t_a.push(a as u32);
                t_b.push(bcol as u32);
                t_aoff.push(spart[a]);
                t_boff.push(spart[bcol]);
                t_copy.push(copy_of[t]);
            }
        }
        // The pairs: every (oa, ob) of one eliminated block with kept(oa) <=
        // kept(ob), grouped by their tile of S in a counting sort. The tile
        // (a, b) is found by row in column b, whose rows are ascending.
        let kept_of_obs = sym.obs_kept;
        let tile_of = |a: usize, b: usize| -> u32 {
            let r = ssym.col_range(b);
            let (mut lo, mut hi) = (r.start, r.end);
            while lo < hi {
                let mid = (lo + hi) / 2;
                if ssym.blk_row(mid) < a {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            assert!(lo < r.end && ssym.blk_row(lo) == a, "every observer pair has a tile in S");
            lo as u32
        };
        let np = sym.pairs;
        let mut pair_tile: Vec<u32> = Vec::with_capacity(np);
        let mut t_pair_ptr = vec![0u32; nt + 1];
        for slot in 0..ne {
            let (q0, q1) = (e_obs_ptr[slot] as usize, e_obs_ptr[slot + 1] as usize);
            for i in q0..q1 {
                for j in i..q1 {
                    let t = tile_of(kept_of_obs[i] as usize, kept_of_obs[j] as usize);
                    pair_tile.push(t);
                    t_pair_ptr[t as usize + 1] += 1;
                }
            }
        }
        assert_eq!(pair_tile.len(), np, "the reduction counted its pairs");
        for t in 0..nt {
            t_pair_ptr[t + 1] += t_pair_ptr[t];
        }
        // the kept blocks' observer lists
        let mut k_obs_ptr = vec![0u32; nk + 1];
        for &k in kept_of_obs {
            k_obs_ptr[k as usize + 1] += 1;
        }
        for k in 0..nk {
            k_obs_ptr[k + 1] += k_obs_ptr[k];
        }
        let mut fill = k_obs_ptr.clone();
        let mut k_obs = vec![0u32; no];
        for (o, &k) in kept_of_obs.iter().enumerate() {
            k_obs[fill[k as usize] as usize] = o as u32;
            fill[k as usize] += 1;
        }
        let k_s_off: Vec<u32> = (0..nk).map(|k| spart[k]).collect();
        let k_orig: Vec<u32> = sym.kept.iter().map(|&orig| hpart[orig as usize]).collect();
        let k_w: Vec<u32> = (0..nk).map(|k| spart[k + 1] - spart[k]).collect();
        // pack, the pairs' observers written in place by their tile
        let mut buf: Vec<u32> = vec![0; SHEADER];
        let arrays: [(usize, &[u32]); 21] = [
            (SE_DIAG_OFF, &e_diag),
            (SE_W, &e_w),
            (SE_ORIG_OFF, &e_orig),
            (SE_OBS_PTR, &e_obs_ptr),
            (SE_DFAC_OFF, &e_dfac),
            (SE_Z_OFF, &e_z),
            (SO_CA_OFF, &o_ca),
            (SO_TRANS, &o_trans),
            (SO_W, &o_w),
            (SO_E, &o_e),
            (SO_ORIG_OFF, &o_orig),
            (SO_W_OFF, &o_woff),
            (ST_VAL_OFF, &t_val),
            (ST_WR, &t_wr),
            (ST_WC, &t_wc),
            (ST_A, &t_a),
            (ST_B, &t_b),
            (ST_A_OFF, &t_aoff),
            (ST_B_OFF, &t_boff),
            (ST_COPY_OFF, &t_copy),
            (ST_PAIR_PTR, &t_pair_ptr),
        ];
        let tail: [(usize, &[u32]); 5] = [(SK_OBS_PTR, &k_obs_ptr), (SK_OBS, &k_obs), (SK_S_OFF, &k_s_off), (SK_ORIG_OFF, &k_orig), (SK_W, &k_w)];
        buf.reserve(arrays.iter().chain(&tail).map(|a| a.1.len()).sum::<usize>() + 2 * np + 64);
        for (slot, v) in arrays {
            buf[slot] = buf.len() as u32;
            buf.extend_from_slice(v);
        }
        let (at_oa, at_ob) = (buf.len(), buf.len() + np);
        buf[SP_OA] = at_oa as u32;
        buf[SP_OB] = at_ob as u32;
        buf.resize(at_ob + np, 0);
        let mut next = t_pair_ptr.clone();
        let mut k = 0usize;
        for slot in 0..ne {
            let (q0, q1) = (e_obs_ptr[slot] as usize, e_obs_ptr[slot + 1] as usize);
            for i in q0..q1 {
                for j in i..q1 {
                    let t = pair_tile[k] as usize;
                    k += 1;
                    let at = next[t] as usize;
                    next[t] += 1;
                    buf[at_oa + at] = i as u32;
                    buf[at_ob + at] = j as u32;
                }
            }
        }
        drop(pair_tile);
        for (slot, v) in tail {
            buf[slot] = buf.len() as u32;
            buf.extend_from_slice(v);
        }
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
        }
    }
}

/// The kernels, compiled per block-width class.
const SOURCE: &str = include_str!("../kernels/schur.cpp");

/// Threads per block of the kernels over entries and blocks.
const THREADS: u32 = 256;
/// Threads per block of the kernels over the tiles of S and the kept blocks.
const PLANE: u32 = 32;

/// Why a [`GpuSchur`] solve failed.
#[derive(Clone, Debug)]
pub enum GpuSchurError {
    /// The diagonal tile of the eliminated block `block` is not positive
    /// definite; the caller raises the damping.
    Eliminated { block: usize },
    /// The reduced system is not positive definite; the caller raises the
    /// damping.
    Reduced,
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

/// The device plan of a [`SchurSymbolic`] and the device state of one
/// reduction on a [`Device`]: the Hessian's values and gradient as
/// uploaded, the eliminated blocks' Cholesky factors, W, S and the reduced
/// right-hand side, the
/// solution. Every device buffer is made in [`new`](Self::new): the plan in
/// one block, uploaded in one copy, and the rest in a work block.
pub struct GpuSchur<D: Device> {
    /// The last attempt's launches as a graph, with what it was captured
    /// against; first, so it is dropped before the blocks it records.
    graph: Option<(GraphKey, D::Graph)>,
    graphs: bool,
    replayed: bool,
    dev: Arc<D>,
    values: Values,
    wide: Wide,
    /// [`GpuSchurParams::f32_reduction`].
    f32_reduction: bool,
    k: Kernels<D::Kernel>,
    /// The plan words (at 0, as the plan's header offsets require), then a
    /// word holding 1, at `on`: the gate of kernels that always run; then
    /// per kept block the tiles of S touching it, `kt_ptr` and `kt` (byte
    /// offsets), for the residual.
    ints: D::Block,
    on: usize,
    kt_ptr: usize,
    kt: usize,
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
    /// The inputs encoded in the route's values.
    conv: Vec<u8>,
    /// The step and the status as a solve reads them.
    host_out: Vec<u8>,
    step: Vec<f64>,
}

impl<D: Device> GpuSchur<D> {
    /// The device plan of `sym`, the reduction of a matrix with the block
    /// structure `hsym`, uploaded to `dev`, with the reduction's blocks and
    /// kernels, for a route of `values` (a [`Wide`] for an f64 root's).
    pub fn new(dev: &Arc<D>, sym: &SchurSymbolic<SparseIndex>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, values: impl Into<Values>) -> Result<Self, GpuError> {
        Self::with_params(dev, sym, hsym, values, &GpuSchurParams::default())
    }

    /// [`new`](Self::new) with the work cut by `params`.
    ///
    /// ```ignore
    /// let params = GpuSchurParams { tile_groups: Some(1), ..Default::default() };
    /// let g = GpuSchur::with_params(&dev, &sym, h.symbolic(), Wide::F64, &params)?;
    /// ```
    pub fn with_params(dev: &Arc<D>, sym: &SchurSymbolic<SparseIndex>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, values: impl Into<Values>, params: &GpuSchurParams) -> Result<Self, GpuError> {
        Self::build(dev, SchurPlan::reduction(sym, hsym), hsym.val_count(), values.into(), params)
    }

    /// [`new`](Self::new) with nothing eliminated: S is H, and a solve is
    /// the whole system's.
    pub fn whole(dev: &Arc<D>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, values: impl Into<Values>) -> Result<Self, GpuError> {
        Self::whole_with_params(dev, hsym, values, &GpuSchurParams::default())
    }

    /// [`whole`](Self::whole) with `params`: with
    /// [`GpuSchurParams::f32_reduction`], S (H) and the solve's vectors in
    /// f32, as the CPU's f32 route.
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
        let defines = format!("{}#define SCHUR_F32 {}\n{}", values.defines(), params.f32_reduction as u32, defines(&p, lanes, tile_warps, tile_groups));
        let (kt_ptr_words, kt_words) = tiles_by_kept_block(&p);
        let mut pk = Packer { words: p.words };
        let blocks = tile_blocks(&pk.words, p.nt as usize, tile_warps as usize, params.split_pairs);
        let tile_blocks = (blocks.len() / (tile_warps as usize + 1)) as u32;
        let at_blocks = pk.push(&blocks);
        pk.words[SB_ITEMS] = at_blocks as u32;
        let on = 4 * pk.push(&[1]);
        let kt_ptr = 4 * pk.push(&kt_ptr_words);
        let kt = 4 * pk.push(&kt_words);
        let mut ints = dev.alloc(pk.words.len() * 4)?;
        dev.write(&mut ints, 0, bytemuck::cast_slice(&pk.words))?;
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
        let source = format!("{WIDE}\n{SOURCE}");
        let entries = ["k_zero_u32", "k_schur_elim", "k_schur_w", "k_schur_tiles", "k_schur_rhs", "k_scatter_kept", "k_schur_backsub", "k_symv_sub", "k_copy_r", "k_refine_start", "k_refine_apply", "k_refine_judge", "k_refine_revert"];
        let [zero_u32, elim, w, tiles, rhs, scatter_kept, backsub, symv_sub, copy, refine_start, refine_apply, refine_judge, refine_revert]: [D::Kernel; 13] =
            dev.compile_many(&source, &defines, &entries)?.try_into().ok().expect("a kernel per entry point");
        let k = Kernels { zero_u32, elim, w, tiles, rhs, scatter_kept, backsub, symv_sub, copy, refine_start, refine_apply, refine_judge, refine_revert };
        Ok(GpuSchur {
            graph: None,
            graphs: true,
            replayed: false,
            dev: dev.clone(),
            values,
            wide: values.wide(),
            f32_reduction: params.f32_reduction,
            k,
            ints,
            on,
            kt_ptr,
            kt,
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
    /// them, read back (for diagnostics).
    pub fn h_values(&self) -> Result<Vec<f64>, GpuError> {
        let mut bytes = vec![0u8; self.values.bytes() * self.nh];
        self.dev.read(&self.work, self.sec.h, &mut bytes)?;
        let mut v = vec![0.0f64; self.nh];
        self.values.decode(&bytes, &mut v);
        Ok(v)
    }

    /// S's values and the reduced right-hand side read back (for tests).
    pub fn download(&self) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        Ok((self.read_reduced(self.sec.s, self.s_len)?, self.read_reduced(self.sec.rhs_kept, self.nks)?))
    }

    /// The scalar the reduction computes and keeps S and its vectors in: f32
    /// with [`GpuSchurParams::f32_reduction`], else the wide scalar.
    pub fn reduced(&self) -> Values {
        Values::of(if self.f32_reduction { 4 } else { 8 }, self.wide)
    }

    /// `n` values of [`reduced`](Self::reduced) of the work block from byte
    /// `at` on, read back.
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

    /// Write the Hessian's tile values (`hsym`'s value count) and the
    /// gradient at [`h_target`](Self::h_target) and
    /// [`rhs_target`](Self::rhs_target), in [`values`](Self::values), for
    /// [`launch`](Self::launch) (no wait).
    pub fn upload<M: SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) -> Result<(), GpuError> {
        assert_eq!(hvals.len(), self.nh, "the Hessian's values, in the structure the reduction was built for");
        assert_eq!(rhs.len(), self.n);
        let values = self.values;
        self.conv.clear();
        hvals.iter().for_each(|v| values.push(v.to_f64(), &mut self.conv));
        self.dev.write(&mut self.work, self.sec.h, &self.conv)?;
        self.conv.clear();
        rhs.iter().for_each(|v| values.push(v.to_f64(), &mut self.conv));
        self.dev.write(&mut self.work, self.sec.rhs, &self.conv)
    }

    /// `kernel` over `blocks` blocks of `threads`; nothing when there are no blocks.
    fn run(&self, kernel: &D::Kernel, blocks: u32, threads: u32, args: &[Arg<'_, D::Block>]) -> Result<(), GpuError> {
        self.run_shared(kernel, blocks, threads, 0, args)
    }

    /// [`run`](Self::run) with `shared_bytes` of dynamic shared memory a block.
    fn run_shared(&self, kernel: &D::Kernel, blocks: u32, threads: u32, shared_bytes: u32, args: &[Arg<'_, D::Block>]) -> Result<(), GpuError> {
        if blocks == 0 {
            return Ok(());
        }
        self.dev.launch(kernel, Launch { grid: [blocks, 1, 1], block: [threads, 1, 1], shared_bytes }, args)
    }

    /// Zero the flags and launch the reduction, from the uploaded values.
    fn launch_reduce(&self) -> Result<(), GpuError> {
        let (s, plan) = (&self.sec, Arg::Block(&self.ints, 0));
        let w = |at: usize| Arg::Block(&self.work, at);
        self.run(&self.k.zero_u32, 1, PLANE, &[w(self.flag_at(0)), Arg::U32(N_FLAGS as u32)])?;
        let elim = [plan, w(s.h), w(s.rhs), w(s.dfac), w(s.z), w(self.flag_at(F_ELIM)), Arg::U32(self.ne)];
        self.run(&self.k.elim, self.ne.div_ceil(THREADS), THREADS, &elim)?;
        self.run(&self.k.w, (self.no * self.w_rows).div_ceil(THREADS), THREADS, &[plan, w(s.h), w(s.dfac), w(s.w), Arg::U32(self.no)])?;
        self.run_shared(&self.k.tiles, self.tile_blocks, self.lanes * self.tile_warps, self.tile_shared, &[plan, w(s.h), w(s.w), w(s.s)])?;
        self.run(&self.k.rhs, self.nk, PLANE, &[plan, w(s.h), w(s.rhs), w(s.z), w(s.rhs_kept)])
    }

    /// Form S and the reduced right-hand side from the damped Hessian's tile
    /// values `hvals` (in `hsym`'s storage order) and the gradient `rhs`, and
    /// wait for the pivot flags (for inspection: [`solve`](Self::solve)
    /// reduces as its first stage).
    pub fn reduce<M: SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) -> Result<(), GpuSchurError> {
        self.upload(hvals, rhs)?;
        self.launch_reduce()?;
        let mut failed = [0u32];
        self.dev.read(&self.work, self.flag_at(F_ELIM), bytemuck::cast_slice_mut(&mut failed))?;
        if failed[0] != 0 { Err(GpuSchurError::Eliminated { block: self.elim_blk[failed[0] as usize - 1] as usize }) } else { Ok(()) }
    }

    /// Whether [`solve`](Self::solve) captures its launches as a graph and
    /// replays it (the default): the first solve with a factor and a
    /// refinement setting runs the kernels and records them, and the solves
    /// after it replay the recording, one launch instead of hundreds.
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
        assert!(!(self.f32_reduction && max_steps > 0), "a refinement measures its residuals against S, which GpuSchurParams::f32_reduction forms in f32");
        self.launch_reduce()?;
        let (s, plan) = (&self.sec, Arg::Block(&self.ints, 0));
        let w = |at: usize| Arg::Block(&self.work, at);
        if self.f32_reduction {
            factor.factorize_device_f32((&self.work, s.s), (&self.work, self.flag_at(F_FACTOR)))?;
        } else {
            factor.factorize_device((&self.work, s.s), (&self.work, self.flag_at(F_FACTOR)))?;
        }
        let nks = self.nks as u32;
        let per_entry = nks.div_ceil(THREADS);
        let on = Arg::Block(&self.ints, self.on);
        // the flags, and the one that gates a refinement step's kernels
        let flags = w(self.flag_at(0));
        let gate = w(self.flag_at(F_ACTIVE));
        let state = w(s.out + self.status_at);
        let copy = |gate, dst: usize, src: usize| self.run(&self.k.copy, per_entry, THREADS, &[gate, w(dst), w(src), Arg::U32(nks)]);
        // r = b - S x
        let residual = |gate| {
            copy(gate, s.r, s.rhs_kept)?;
            let (kt_ptr, kt) = (Arg::Block(&self.ints, self.kt_ptr), Arg::Block(&self.ints, self.kt));
            self.run(&self.k.symv_sub, self.nk, PLANE, &[gate, plan, kt_ptr, kt, w(s.s), w(s.x_kept), w(s.r)])
        };
        copy(on, s.x_kept, s.rhs_kept)?;
        if self.f32_reduction {
            factor.solve_device_f32((&self.work, s.x_kept))?;
        } else {
            factor.solve_device((&self.work, s.x_kept))?;
        }
        if max_steps > 0 {
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
        // the kept blocks into the full vector, then the eliminated ones
        self.run(&self.k.scatter_kept, self.nk.div_ceil(THREADS), THREADS, &[plan, w(s.x_kept), w(s.out), Arg::U32(self.nk)])?;
        self.run(&self.k.backsub, self.ne.div_ceil(THREADS), THREADS, &[plan, w(s.h), w(s.rhs), w(s.dfac), w(s.out), Arg::U32(self.ne)])
    }

    /// One damped solve, entirely on the device, waiting once: form S and
    /// the reduced right-hand side from the Hessian's tile values `hvals`
    /// (in `hsym`'s storage order) and the gradient `rhs`, factorize S with
    /// `factor`, solve for the kept blocks, refine the solution against S
    /// when `refine` is `Some((max_steps, tol))` (an f32 factor; `tol`
    /// relative to the right-hand side), recover the eliminated blocks, and
    /// read the full step into `delta` with the pivot flags and the
    /// refinement's state.
    ///
    /// The refinement decides on the device: `max_steps` steps are
    /// launched, and those after it has converged or stalled do nothing. A
    /// failed pivot is found with the result, so a failing solve runs to the
    /// end; `delta` is then left as it was. The launches are captured as a
    /// graph and replayed by the solves after it (see
    /// [`use_graphs`](Self::use_graphs)).
    pub fn solve<M: SchurReal + 'static>(
        &mut self,
        factor: &mut GpuFactor<D>,
        hvals: &[M],
        rhs: &[M],
        refine: Option<(usize, f64)>,
        delta: &mut [M],
    ) -> Result<RefineStats, GpuSchurError> {
        assert_eq!(delta.len(), self.n);
        self.upload(hvals, rhs)?;
        self.enqueue(factor, refine)?;
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
    /// [`graph_key`](Self::graph_key)).
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

/// Per kept block, the tiles of S touching it, for `k_symv_sub`: `(ptr, tiles)`
/// with block k's at `tiles[ptr[k]..ptr[k + 1]]`, each a tile index times 2,
/// plus 1 when k is the tile's column block and not its row block.
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
            ("SE_DIAG_OFF", SE_DIAG_OFF),
            ("SE_W", SE_W),
            ("SE_ORIG_OFF", SE_ORIG_OFF),
            ("SE_OBS_PTR", SE_OBS_PTR),
            ("SE_DFAC_OFF", SE_DFAC_OFF),
            ("SE_Z_OFF", SE_Z_OFF),
            ("SO_CA_OFF", SO_CA_OFF),
            ("SO_TRANS", SO_TRANS),
            ("SO_W", SO_W),
            ("SO_E", SO_E),
            ("SO_ORIG_OFF", SO_ORIG_OFF),
            ("SO_W_OFF", SO_W_OFF),
            ("ST_VAL_OFF", ST_VAL_OFF),
            ("ST_WR", ST_WR),
            ("ST_WC", ST_WC),
            ("ST_A", ST_A),
            ("ST_B", ST_B),
            ("ST_A_OFF", ST_A_OFF),
            ("ST_B_OFF", ST_B_OFF),
            ("ST_COPY_OFF", ST_COPY_OFF),
            ("ST_PAIR_PTR", ST_PAIR_PTR),
            ("SP_OA", SP_OA),
            ("SP_OB", SP_OB),
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

    /// Every tile of S is in one block of `k_schur_tiles`: those of more
    /// than the split's pairs alone on a block, the others on one unit, the
    /// tiles by pair count, most first; with one unit a block, a block per
    /// tile in storage order.
    #[test]
    fn the_tile_blocks_hold_every_tile_once() {
        let (h, _, _, elim) = ba_like(40, 300, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let p = SchurPlan::reduction(&sym, h.symbolic());
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
    /// 32, with one lane group a unit and with several (a group's pairs and
    /// the order of the groups' sums do not depend on the lanes).
    #[test]
    fn a_tile_on_64_lanes_is_the_tile_on_32() {
        let (h, _, rhs, elim) = ba_like(40, 300, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let dev = device();
        for wide in wides(&*device()) {
            for groups in [1, 4] {
                let reduce = |lanes| {
                    let p = SchurPlan::reduction(&sym, h.symbolic());
                    let params = GpuSchurParams { tile_groups: Some(groups), ..Default::default() };
                    let mut g = GpuSchur::build_lanes(&dev, p, h.symbolic().val_count(), wide.into(), lanes, &params).unwrap();
                    g.reduce(h.vals(), &rhs).unwrap();
                    g.download().unwrap()
                };
                assert_eq!(reduce(64), reduce(32), "{wide:?}, {groups} groups");
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
    /// captures of them replays to the same.
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
            // the plan block and the work block
            if before == 0 {
                assert_eq!(fresh, 2);
            } else {
                assert!(fresh <= 2, "{fresh}");
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
            for (name, kernel) in [("elim", &k.elim), ("tiles", &k.tiles), ("rhs", &k.rhs), ("backsub", &k.backsub)] {
                assert_eq!(kernel.local_bytes().unwrap(), 0, "{wide:?} k_schur_{name}, {groups} tile groups: {} registers", kernel.registers().unwrap());
            }
        }
    }

}
