//! [`GpuSchur`]: the Schur reduction on the device, generic over block
//! widths, with the reduced system kept on the device for the factorization,
//! the refinement and the back-substitution.
//!
//! Formed in the route's wide scalar ([`Wide`]: `f64` or df32) whatever the
//! matrix's scalar (docs/dev/GPU.md, E2: `f32` cannot form S, the
//! cancelling sums lose their operands' digits). Per reduction, from the
//! Hessian's tile values and the gradient:
//!
//!   elim      one unit per eliminated block: `D_e^-1` through its Cholesky
//!             factor, and `z_e = D_e^-1 b_e`
//!   w         one unit per observer: `W_o = C_o D_e^-1`
//!   tiles     one plane per tile of S: `S(a, b) = H(a, b) - sum over the
//!             pairs (oa, ob) of its list of W_oa C_ob^T`, the pair lists
//!             built once on the host; no shared write
//!   rhs       one plane per kept block: `b_k - sum over its observers of
//!             C_o z_e`
//!   backsub   one unit per eliminated block: `x_e = D_e^-1 (b_e - sum_a C_a^T x_a)`
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
use crate::{Arg, Device, GpuError, Launch, Layout, WIDE, Wide, define_u32s};

/// No kept-kept tile behind a tile of S.
const NONE: u32 = u32::MAX;

// The header of the reduction's plan buffer: where each array starts.
pub(crate) const SE_DIAG_OFF: usize = 0;
pub(crate) const SE_W: usize = 1;
pub(crate) const SE_ORIG_OFF: usize = 2;
pub(crate) const SE_OBS_PTR: usize = 3;
pub(crate) const SE_DINV_OFF: usize = 4;
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
pub(crate) const SHEADER: usize = 32;

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

/// Where the status starts in a solve's output, after the step's `n` f64.
pub(crate) fn status_at(n: usize) -> usize {
    (n * core::mem::size_of::<f64>()).div_ceil(256) * 256
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
    /// Values of S, of the block inverses, of z and of W.
    pub s_len: usize,
    pub dinv_len: usize,
    pub z_len: usize,
    pub w_len: usize,
    /// The widest eliminated and kept blocks, rounded up as the kernels take them.
    pub ew_cap: usize,
    pub kw_cap: usize,
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
        let (mut e_diag, mut e_w, mut e_orig, mut e_dinv, mut e_z, mut elim_blk) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut dinv_len, mut z_len) = (0u32, 0u32);
        let mut ew_max = 1usize;
        for slot in 0..ne {
            let dblk = sym.elim_diag[slot] as usize;
            let e = hsym.blk_row(dblk);
            let we = sym.elim_w[slot] as usize;
            ew_max = ew_max.max(we);
            e_diag.push(hsym.val_range(dblk).start as u32);
            e_w.push(we as u32);
            e_orig.push(hpart[e]);
            e_dinv.push(dinv_len);
            e_z.push(z_len);
            elim_blk.push(e as u32);
            dinv_len += (we * we) as u32;
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
            (SE_DINV_OFF, &e_dinv),
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
            dinv_len: dinv_len as usize,
            z_len: z_len as usize,
            w_len: w_len as usize,
            ew_cap,
            kw_cap,
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
    /// The Hessian's tile values and the gradient, f64.
    h: usize,
    rhs: usize,
    dinv: usize,
    z: usize,
    w: usize,
    s: usize,
    rhs_kept: usize,
    x_kept: usize,
    x_prev: usize,
    r: usize,
    /// What a solve reads back, in one copy: the full step (`n` f64), then
    /// at `status_at` from here the refinement's state and the flags.
    out: usize,
    bytes: usize,
}

/// The device plan of a [`SchurSymbolic`] and the device state of one
/// reduction on a [`Device`]: the Hessian's values and gradient as
/// uploaded, the block inverses, W, S and the reduced right-hand side, the
/// solution. Every device buffer is made in [`new`](Self::new): the plan in
/// one block, uploaded in one copy, and the rest in a work block.
pub struct GpuSchur<D: Device> {
    /// The scalar of H, the gradient, S and the reduction's vectors.
    /// The last attempt's launches as a graph, with what it was captured
    /// against; first, so it is dropped before the blocks it records.
    graph: Option<(GraphKey, D::Graph)>,
    graphs: bool,
    replayed: bool,
    dev: Arc<D>,
    wide: Wide,
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
    nt: u32,
    /// Threads per tile of S in `k_schur_tiles`.
    lanes: u32,
    nk: u32,
    n: usize,
    nks: usize,
    nh: usize,
    s_len: usize,
    elim_blk: Vec<u32>,
    /// The inputs encoded in the wide scalar.
    conv: Vec<u8>,
    /// The step and the status as a solve reads them.
    host_out: Vec<u8>,
    step: Vec<f64>,
}

impl<D: Device> GpuSchur<D> {
    /// The device plan of `sym`, the reduction of a matrix with the block
    /// structure `hsym`, uploaded to `dev`, with the reduction's blocks and
    /// kernels, in `wide`.
    pub fn new(dev: &Arc<D>, sym: &SchurSymbolic<SparseIndex>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, wide: Wide) -> Result<Self, GpuError> {
        Self::build(dev, SchurPlan::reduction(sym, hsym), hsym.val_count(), wide)
    }

    /// [`new`](Self::new) with nothing eliminated: S is H, and a solve is
    /// the whole system's.
    pub fn whole(dev: &Arc<D>, hsym: &SymbolicSparseBlockColMat<SparseIndex>, wide: Wide) -> Result<Self, GpuError> {
        Self::build(dev, SchurPlan::whole(hsym), hsym.val_count(), wide)
    }

    fn build(dev: &Arc<D>, p: SchurPlan, nh: usize, wide: Wide) -> Result<Self, GpuError> {
        let lanes = if D::RUNTIME == "cuda" { PLANE } else { dev.arch().warp.max(PLANE) };
        Self::build_lanes(dev, p, nh, wide, lanes)
    }

    /// [`build`](Self::build) with `lanes` threads per tile of S in
    /// `k_schur_tiles`: the device's whole wavefront where the kernels stage
    /// their blocks (not on CUDA, see `SCHUR_STAGED` in schur.cpp), else a
    /// plane.
    fn build_lanes(dev: &Arc<D>, p: SchurPlan, nh: usize, wide: Wide, lanes: u32) -> Result<Self, GpuError> {
        assert!(lanes == PLANE || D::RUNTIME != "cuda", "CUDA's k_schur_tiles takes a plane per tile");
        let defines = format!("{}{}", wide.defines(), defines(&p, lanes));
        let (kt_ptr_words, kt_words) = tiles_by_kept_block(&p);
        let mut pk = Packer { words: p.words };
        let on = 4 * pk.push(&[1]);
        let kt_ptr = 4 * pk.push(&kt_ptr_words);
        let kt = 4 * pk.push(&kt_words);
        let mut ints = dev.alloc(pk.words.len() * 4)?;
        dev.write(&mut ints, 0, bytemuck::cast_slice(&pk.words))?;
        let status_at = status_at(p.n);
        let f = core::mem::size_of::<f64>();
        let mut l = Layout::new();
        let sec = Sections {
            h: l.take(nh * f),
            rhs: l.take(p.n * f),
            dinv: l.take(p.dinv_len * f),
            z: l.take(p.z_len * f),
            w: l.take(p.w_len * f),
            s: l.take(p.s_len * f),
            rhs_kept: l.take(p.nks * f),
            x_kept: l.take(p.nks * f),
            x_prev: l.take(p.nks * f),
            r: l.take(p.nks * f),
            out: l.take(status_at + STATUS_BYTES),
            bytes: l.bytes(),
        };
        let work = dev.alloc(sec.bytes)?;
        let source = format!("{WIDE}\n{SOURCE}");
        let entries = ["k_zero_u32", "k_schur_elim", "k_schur_w", "k_schur_tiles", "k_schur_rhs", "k_scatter_kept", "k_schur_backsub", "k_symv_sub", "k_copy_f64", "k_refine_start", "k_refine_apply", "k_refine_judge", "k_refine_revert"];
        let [zero_u32, elim, w, tiles, rhs, scatter_kept, backsub, symv_sub, copy, refine_start, refine_apply, refine_judge, refine_revert]: [D::Kernel; 13] =
            dev.compile_many(&source, &defines, &entries)?.try_into().ok().expect("a kernel per entry point");
        let k = Kernels { zero_u32, elim, w, tiles, rhs, scatter_kept, backsub, symv_sub, copy, refine_start, refine_apply, refine_judge, refine_revert };
        Ok(GpuSchur {
            graph: None,
            graphs: true,
            replayed: false,
            dev: dev.clone(),
            wide,
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
            nt: p.nt,
            lanes,
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
    /// storage order of the symbolic's `s`, as the last reduction left them.
    pub fn s_values(&self) -> ((&D::Block, usize), usize) {
        ((&self.work, self.sec.s), self.s_len)
    }

    /// The scalar of H, the gradient, S and the reduction's vectors.
    pub fn wide(&self) -> Wide {
        self.wide
    }

    /// The reduced right-hand side on the device, [`kept_size`](Self::kept_size) values.
    pub fn rhs_kept(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.rhs_kept)
    }

    /// Where [`launch`](Self::launch) reads the Hessian's tile values: the
    /// structure's value count, in [`wide`](Self::wide), for a device
    /// assembly to write.
    pub fn h_target(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.h)
    }

    /// Where [`launch`](Self::launch) reads the gradient: [`dim`](Self::dim) values.
    pub fn rhs_target(&self) -> (&D::Block, usize) {
        (&self.work, self.sec.rhs)
    }

    /// The step of the last solve: [`dim`](Self::dim) values.
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
        self.read_values(self.sec.h, self.nh)
    }

    /// S's values and the reduced right-hand side read back (for tests).
    pub fn download(&self) -> Result<(Vec<f64>, Vec<f64>), GpuError> {
        Ok((self.read_values(self.sec.s, self.s_len)?, self.read_values(self.sec.rhs_kept, self.nks)?))
    }

    /// `n` values of the work block from byte `at` on, read back.
    fn read_values(&self, at: usize, n: usize) -> Result<Vec<f64>, GpuError> {
        let mut bytes = vec![0u8; 8 * n];
        self.dev.read(&self.work, at, &mut bytes)?;
        let mut v = vec![0.0f64; n];
        self.wide.decode(&bytes, &mut v);
        Ok(v)
    }

    /// Byte offset of flag `f` in the work block.
    fn flag_at(&self, f: usize) -> usize {
        self.sec.out + self.status_at + FLAGS_AT + 4 * f
    }

    /// Write the Hessian's tile values (`hsym`'s value count) and the
    /// gradient at [`h_target`](Self::h_target) and
    /// [`rhs_target`](Self::rhs_target), in [`wide`](Self::wide), for
    /// [`launch`](Self::launch) (no wait).
    pub fn upload<M: SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) -> Result<(), GpuError> {
        assert_eq!(hvals.len(), self.nh, "the Hessian's values, in the structure the reduction was built for");
        assert_eq!(rhs.len(), self.n);
        let wide = self.wide;
        self.conv.clear();
        self.conv.extend(hvals.iter().flat_map(|v| wide.bits(v.to_f64()).to_le_bytes()));
        self.dev.write(&mut self.work, self.sec.h, &self.conv)?;
        self.conv.clear();
        self.conv.extend(rhs.iter().flat_map(|v| wide.bits(v.to_f64()).to_le_bytes()));
        self.dev.write(&mut self.work, self.sec.rhs, &self.conv)
    }

    /// `kernel` over `blocks` blocks of `threads`; nothing when there are no blocks.
    fn run(&self, kernel: &D::Kernel, blocks: u32, threads: u32, args: &[Arg<'_, D::Block>]) -> Result<(), GpuError> {
        if blocks == 0 {
            return Ok(());
        }
        self.dev.launch(kernel, Launch { grid: [blocks, 1, 1], block: [threads, 1, 1], shared_bytes: 0 }, args)
    }

    /// Zero the flags and launch the reduction, from the uploaded values.
    fn launch_reduce(&self) -> Result<(), GpuError> {
        let (s, plan) = (&self.sec, Arg::Block(&self.ints, 0));
        let w = |at: usize| Arg::Block(&self.work, at);
        self.run(&self.k.zero_u32, 1, PLANE, &[w(self.flag_at(0)), Arg::U32(N_FLAGS as u32)])?;
        let elim = [plan, w(s.h), w(s.rhs), w(s.dinv), w(s.z), w(self.flag_at(F_ELIM)), Arg::U32(self.ne)];
        self.run(&self.k.elim, self.ne.div_ceil(THREADS), THREADS, &elim)?;
        self.run(&self.k.w, self.no.div_ceil(THREADS), THREADS, &[plan, w(s.h), w(s.dinv), w(s.w), Arg::U32(self.no)])?;
        self.run(&self.k.tiles, self.nt, self.lanes, &[plan, w(s.h), w(s.w), w(s.s)])?;
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
        self.launch_reduce()?;
        let (s, plan) = (&self.sec, Arg::Block(&self.ints, 0));
        let w = |at: usize| Arg::Block(&self.work, at);
        factor.factorize_device((&self.work, s.s), (&self.work, self.flag_at(F_FACTOR)))?;
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
        factor.solve_device((&self.work, s.x_kept))?;
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
        self.run(&self.k.backsub, self.ne.div_ceil(THREADS), THREADS, &[plan, w(s.h), w(s.rhs), w(s.dinv), w(s.out), Arg::U32(self.ne)])
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
        self.wide.decode(&self.host_out[..8 * self.n], &mut self.step);
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

/// The `#define`s of `schur.cpp`: the block sizes, the width classes, and
/// the plan header's and the status's slots of [`crate::schur`].
fn defines(p: &SchurPlan, lanes: u32) -> String {
    let mut s = String::new();
    define_u32s(
        &mut s,
        &[
            ("THREADS", THREADS as usize),
            ("TILE_LANES", lanes as usize),
            ("EW", p.ew_cap),
            ("KW", p.kw_cap),
            // the entries of a tile of S per lane
            ("ACC", (p.kw_cap * p.kw_cap).div_ceil(lanes as usize)),
            ("SE_DIAG_OFF", SE_DIAG_OFF),
            ("SE_W", SE_W),
            ("SE_ORIG_OFF", SE_ORIG_OFF),
            ("SE_OBS_PTR", SE_OBS_PTR),
            ("SE_DINV_OFF", SE_DINV_OFF),
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

#[cfg(all(test, any(feature = "cuda", feature = "hip", feature = "cpu")))]
mod tests {
    use super::*;
    use crate::GpuPrecision;
    use crate::TestDevice;
    use crate::test_util::{ba_like, rel_residual};
    use arael_faer::schur::{SchurContext, schur_reduce, schur_symbolic};
    use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};

    type Matrix = arael_faer::bsc::SparseBlockColMat<SparseIndex, f64>;

    fn device() -> Arc<TestDevice> {
        Arc::new(TestDevice::new(0).unwrap())
    }

    fn err_max(a: &[f64], b: &[f64]) -> f64 {
        let m = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0.0f64, |e, (x, y)| e.max((x - y).abs())) / m
    }

    /// Both wide scalars: the reduction's tests run in each.
    const WIDES: [Wide; 2] = [Wide::F64, Wide::Df32];

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
        for wide in WIDES {
            the_reduction_matches_the_cpu_in(wide);
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
        assert_eq!(s32, g.download().unwrap().0, "f32 inputs are their f64 twins");
    }

    /// A tile of S on 64 lanes, as on a 64-lane wavefront, is the tile on
    /// 32.
    #[test]
    fn a_tile_on_64_lanes_is_the_tile_on_32() {
        if <TestDevice as Device>::RUNTIME == "cuda" {
            // CUDA's k_schur_tiles takes a plane per tile (SCHUR_STAGED)
            return;
        }
        let (h, _, rhs, elim) = ba_like(40, 300, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let dev = device();
        for wide in WIDES {
            let reduce = |lanes| {
                let p = SchurPlan::reduction(&sym, h.symbolic());
                let mut g = GpuSchur::build_lanes(&dev, p, h.symbolic().val_count(), wide, lanes).unwrap();
                g.reduce(h.vals(), &rhs).unwrap();
                g.download().unwrap()
            };
            assert_eq!(reduce(64), reduce(32), "{wide:?}");
        }
    }

    #[test]
    fn the_reduced_solve_solves_the_whole_system() {
        for wide in WIDES {
            the_reduced_solve_solves_the_whole_system_in(wide);
        }
    }

    fn the_reduced_solve_solves_the_whole_system_in(wide: Wide) {
        let (h, dense, rhs, _, sym, sn) = setup(40, 300, 9);
        let n = h.symbolic().ncols();
        let dev = device();
        let mut g = GpuSchur::new(&dev, &sym, h.symbolic(), wide).unwrap();
        for (precision, refine, tol) in [(GpuPrecision::F64, None, 1e-10), (GpuPrecision::F32, Some((4usize, 1e-10f64)), 1e-8)] {
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
        for wide in WIDES {
            nothing_eliminated_solves_the_whole_system_in(wide);
        }
    }

    fn nothing_eliminated_solves_the_whole_system_in(wide: Wide) {
        let (h, dense, rhs, _) = ba_like(30, 200, 5);
        let n = h.symbolic().ncols();
        let order = nd_block_order(h.symbolic());
        let sn = SupernodalSymbolic::new(h.symbolic(), Some(&order), &SupernodalParams::default()).unwrap();
        let dev = device();
        let mut g = GpuSchur::whole(&dev, h.symbolic(), wide).unwrap();
        for (precision, refine, tol) in [(GpuPrecision::F64, None, 1e-10), (GpuPrecision::F32, Some((4usize, 1e-10f64)), 1e-8)] {
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
        for wide in WIDES {
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
        for wide in WIDES {
            replayed_solves_match_launched_ones_in(wide);
        }
    }

    fn replayed_solves_match_launched_ones_in(wide: Wide) {
        let (h, dense, rhs, _, sym, sn) = setup(40, 300, 9);
        let n = h.symbolic().ncols();
        let dev = device();
        // another damping: the replay must see the values written before it
        let (h2, dense2) = damped(&h, &dense);
        for precision in [GpuPrecision::F64, GpuPrecision::F32] {
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
        for wide in WIDES {
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
        for wide in WIDES {
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
        for wide in WIDES {
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
        for precision in [GpuPrecision::F64, GpuPrecision::F32] {
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
    /// arrays stay in registers: no kernel uses local memory (a GPU's).
    #[test]
    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn the_kernels_keep_their_arrays_in_registers() {
        for wide in WIDES {
            the_kernels_keep_their_arrays_in_registers_in(wide);
        }
    }

    #[cfg(any(feature = "cuda", feature = "hip"))]
    fn the_kernels_keep_their_arrays_in_registers_in(wide: Wide) {
        let (h, _, _, _, sym, _) = setup(10, 30, 2);
        let g = GpuSchur::new(&device(), &sym, h.symbolic(), wide).unwrap();
        let k = &g.k;
        for (name, kernel) in [("elim", &k.elim), ("tiles", &k.tiles), ("rhs", &k.rhs), ("backsub", &k.backsub)] {
            assert_eq!(kernel.local_bytes().unwrap(), 0, "k_schur_{name}: {} registers", kernel.registers().unwrap());
        }
    }

}
