//! [`GpuSchur`]: the Schur reduction on the device, generic over block
//! widths, with the reduced system kept on the device for the factorization,
//! the refinement and the back-substitution.
//!
//! Formed in `f64` whatever the matrix's scalar (docs/dev/GPU.md, E2: `f32`
//! cannot form S, the cancelling sums lose their operands' digits). Per
//! reduction, from the Hessian's tile values and the gradient:
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
//! device, and one-cube norm kernels decide there whether the next step is
//! due, so a solve with an f32 factor waits only for its result either:
//! [`GpuSchur::solve`] reads the step, the pivot flags and the refinement's
//! state in one copy.

// Cube code, as kernels.rs says.
#![allow(clippy::unnecessary_cast, clippy::manual_div_ceil, clippy::neg_cmp_op_on_partial_ord, clippy::assign_op_pattern, clippy::too_many_arguments, clippy::manual_memcpy)]

use arael_faer::bsc::SymbolicSparseBlockColMat;
use arael_faer::schur::{SchurError, SchurSymbolic};
use cubecl::prelude::*;
use cubecl::server::Handle;

use crate::context::{GpuContext, Lease};
use crate::host::Upload;
use crate::supernodal::Packer;
use crate::plan::SparseIndex;
use crate::GpuFactor;

/// No kept-kept tile behind a tile of S.
const NONE: u32 = u32::MAX;

// The header of the reduction's plan buffer: where each array starts.
const SE_DIAG_OFF: usize = 0;
const SE_W: usize = 1;
const SE_ORIG_OFF: usize = 2;
const SE_OBS_PTR: usize = 3;
const SE_DINV_OFF: usize = 4;
const SE_Z_OFF: usize = 5;
const SO_CA_OFF: usize = 6;
const SO_TRANS: usize = 7;
const SO_W: usize = 8;
const SO_E: usize = 9;
const SO_ORIG_OFF: usize = 10;
const SO_W_OFF: usize = 11;
const ST_VAL_OFF: usize = 12;
const ST_WR: usize = 13;
const ST_WC: usize = 14;
const ST_A: usize = 15;
const ST_B: usize = 16;
const ST_A_OFF: usize = 17;
const ST_B_OFF: usize = 18;
const ST_COPY_OFF: usize = 19;
const ST_PAIR_PTR: usize = 20;
const SP_OA: usize = 21;
const SP_OB: usize = 22;
const SK_OBS_PTR: usize = 23;
const SK_OBS: usize = 24;
const SK_S_OFF: usize = 25;
const SK_ORIG_OFF: usize = 26;
const SK_W: usize = 27;
const SHEADER: usize = 32;

// ---------------------------------------------------------------- kernels

/// `C[i][k]` of an observer's coupling tile at `ca`, `wo x we`, in either
/// storage orientation.
#[cube]
fn coupling(h: &[f64], ca: usize, trans: u32, wo: usize, we: usize, i: usize, k: usize) -> f64 {
    let mut v = h[ca + i + k * wo];
    if trans != 0u32 {
        v = h[ca + k + i * we];
    }
    v
}

/// Stage 1, one unit per eliminated block: the inverse of its damped
/// diagonal tile (upper-stored) through the Cholesky factor, column-major
/// `we x we` into `dinv`, and `z = D^-1 b`. A pivot that is not positive
/// sets `fail[0]` to the block's slot plus one.
#[cube(launch_unchecked)]
pub fn k_schur_elim(plan: &[u32], h: &[f64], rhs: &[f64], dinv: &mut [f64], z: &mut [f64], fail: &mut [u32], ne: u32, #[comptime] ew: usize) {
    let o_diag = plan[SE_DIAG_OFF] as usize;
    let o_w = plan[SE_W] as usize;
    let o_orig = plan[SE_ORIG_OFF] as usize;
    let o_dinv = plan[SE_DINV_OFF] as usize;
    let o_z = plan[SE_Z_OFF] as usize;
    let e = ABSOLUTE_POS;
    if e < ne as usize {
        let w = plan[o_w + e] as usize;
        let diag = plan[o_diag + e] as usize;
        let orig = plan[o_orig + e] as usize;
        let doff = plan[o_dinv + e] as usize;
        let zoff = plan[o_z + e] as usize;
        // the tile, symmetric from its upper triangle, then its lower factor in place
        let mut a = Array::<f64>::new(ew * ew);
        for j in 0..w {
            for i in 0..w {
                let mut v = h[diag + i + j * w];
                if i > j {
                    v = h[diag + j + i * w];
                }
                a[i + j * ew] = v;
            }
        }
        for j in 0..w {
            let mut d = a[j + j * ew];
            for k in 0..j {
                d -= a[j + k * ew] * a[j + k * ew];
            }
            if !(d > 0.0f64) {
                fail[0] = e as u32 + 1u32;
                d = 1.0f64;
            }
            let ljj = f64::sqrt(d);
            a[j + j * ew] = ljj;
            let mut i = j + 1usize;
            while i < w {
                let mut v = a[i + j * ew];
                for k in 0..j {
                    v -= a[i + k * ew] * a[j + k * ew];
                }
                a[i + j * ew] = v / ljj;
                i += 1usize;
            }
        }
        // L^-1 (lower) by forward substitution, column by column
        let mut li = Array::<f64>::new(ew * ew);
        for c in 0..w {
            for r in 0..w {
                let mut rhs_v = 0.0f64;
                if r == c {
                    rhs_v = 1.0f64;
                }
                if r >= c {
                    let mut v = rhs_v;
                    for k in c..r {
                        v -= a[r + k * ew] * li[k + c * ew];
                    }
                    li[r + c * ew] = v / a[r + r * ew];
                } else {
                    li[r + c * ew] = 0.0f64;
                }
            }
        }
        // D^-1 = L^-T L^-1
        for j in 0..w {
            for i in 0..w {
                let mut v = 0.0f64;
                for k in 0..w {
                    v += li[k + i * ew] * li[k + j * ew];
                }
                dinv[doff + i + j * w] = v;
            }
        }
        for i in 0..w {
            let mut v = 0.0f64;
            for j in 0..w {
                v += dinv[doff + i + j * w] * rhs[orig + j];
            }
            z[zoff + i] = v;
        }
    }
}

/// Stage 2, one unit per observer: `W_o = C_o D_e^-1`, row-major `wo x we`.
#[cube(launch_unchecked)]
pub fn k_schur_w(plan: &[u32], h: &[f64], dinv: &[f64], w: &mut [f64], no: u32) {
    let o_ew = plan[SE_W] as usize;
    let o_dinv = plan[SE_DINV_OFF] as usize;
    let o_ca = plan[SO_CA_OFF] as usize;
    let o_trans = plan[SO_TRANS] as usize;
    let o_ow = plan[SO_W] as usize;
    let o_e = plan[SO_E] as usize;
    let o_woff = plan[SO_W_OFF] as usize;
    let o = ABSOLUTE_POS;
    if o < no as usize {
        let wo = plan[o_ow + o] as usize;
        let e = plan[o_e + o] as usize;
        let we = plan[o_ew + e] as usize;
        let ca = plan[o_ca + o] as usize;
        let trans = plan[o_trans + o];
        let doff = plan[o_dinv + e] as usize;
        let woff = plan[o_woff + o] as usize;
        for i in 0..wo {
            for l in 0..we {
                let mut v = 0.0f64;
                for k in 0..we {
                    v += coupling(h, ca, trans, wo, we, i, k) * dinv[doff + k + l * we];
                }
                w[woff + i * we + l] = v;
            }
        }
    }
}

/// Stage 3, one plane of 32 units per tile of S: the kept-kept tile of H
/// (or nothing) minus `W_oa C_ob^T` over the tile's pairs, unit `lane`
/// owning the entries `lane, lane + 32, ...` (`acc_cap` of them at most);
/// a diagonal tile upper-only.
#[cube(launch_unchecked)]
pub fn k_schur_tiles(plan: &[u32], h: &[f64], w: &[f64], s: &mut [f64], #[comptime] acc_cap: usize) {
    let o_ew = plan[SE_W] as usize;
    let o_ca = plan[SO_CA_OFF] as usize;
    let o_trans = plan[SO_TRANS] as usize;
    let o_ow = plan[SO_W] as usize;
    let o_e = plan[SO_E] as usize;
    let o_woff = plan[SO_W_OFF] as usize;
    let o_val = plan[ST_VAL_OFF] as usize;
    let o_wr = plan[ST_WR] as usize;
    let o_wc = plan[ST_WC] as usize;
    let o_a = plan[ST_A] as usize;
    let o_b = plan[ST_B] as usize;
    let o_copy = plan[ST_COPY_OFF] as usize;
    let o_pp = plan[ST_PAIR_PTR] as usize;
    let o_oa = plan[SP_OA] as usize;
    let o_ob = plan[SP_OB] as usize;
    let t = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let val = plan[o_val + t] as usize;
    let wr = plan[o_wr + t] as usize;
    let wc = plan[o_wc + t] as usize;
    let a = plan[o_a + t];
    let b = plan[o_b + t];
    let copy = plan[o_copy + t];
    let p0 = plan[o_pp + t] as usize;
    let p1 = plan[o_pp + t + 1] as usize;
    let mut acc = Array::<f64>::new(acc_cap);
    #[unroll]
    for m in 0..acc_cap {
        acc[m] = 0.0f64;
    }
    for p in p0..p1 {
        let oa = plan[o_oa + p] as usize;
        let ob = plan[o_ob + p] as usize;
        let e = plan[o_e + oa] as usize;
        let we = plan[o_ew + e] as usize;
        let woff = plan[o_woff + oa] as usize;
        let cb = plan[o_ca + ob] as usize;
        let tb = plan[o_trans + ob];
        let wb = plan[o_ow + ob] as usize;
        #[unroll]
        for m in 0..acc_cap {
            let ent = lane + 32 * m;
            if ent < wr * wc {
                let j = ent / wr;
                let i = ent - j * wr;
                let mut v = 0.0f64;
                for k in 0..we {
                    v += w[woff + i * we + k] * coupling(h, cb, tb, wb, we, j, k);
                }
                acc[m] -= v;
            }
        }
    }
    #[unroll]
    for m in 0..acc_cap {
        let ent = lane + 32 * m;
        if ent < wr * wc {
            let j = ent / wr;
            let i = ent - j * wr;
            let mut v = acc[m];
            if copy != 4294967295u32 {
                v += h[copy as usize + i + j * wr];
            }
            if a == b && i > j {
                v = 0.0f64;
            }
            s[val + i + j * wr] = v;
        }
    }
}

/// Stage 4, one plane per kept block: `rhs_kept = b_k - sum over the block's
/// observers of C_o z_e`, the lanes over the observers, a plane sum per row.
#[cube(launch_unchecked)]
pub fn k_schur_rhs(plan: &[u32], h: &[f64], rhs: &[f64], z: &[f64], out: &mut [f64], #[comptime] kw_cap: usize) {
    let o_ew = plan[SE_W] as usize;
    let o_z = plan[SE_Z_OFF] as usize;
    let o_ca = plan[SO_CA_OFF] as usize;
    let o_trans = plan[SO_TRANS] as usize;
    let o_e = plan[SO_E] as usize;
    let o_kptr = plan[SK_OBS_PTR] as usize;
    let o_kobs = plan[SK_OBS] as usize;
    let o_soff = plan[SK_S_OFF] as usize;
    let o_korig = plan[SK_ORIG_OFF] as usize;
    let o_kw = plan[SK_W] as usize;
    let k = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let s_off = plan[o_soff + k] as usize;
    let orig = plan[o_korig + k] as usize;
    let wk = plan[o_kw + k] as usize;
    let q0 = plan[o_kptr + k] as usize;
    let q1 = plan[o_kptr + k + 1] as usize;
    let mut acc = Array::<f64>::new(kw_cap);
    for i in 0..kw_cap {
        acc[i] = 0.0f64;
    }
    let mut q = q0 + lane;
    while q < q1 {
        let o = plan[o_kobs + q] as usize;
        let e = plan[o_e + o] as usize;
        let we = plan[o_ew + e] as usize;
        let ca = plan[o_ca + o] as usize;
        let trans = plan[o_trans + o];
        let zoff = plan[o_z + e] as usize;
        for i in 0..wk {
            let mut v = 0.0f64;
            for kk in 0..we {
                v += coupling(h, ca, trans, wk, we, i, kk) * z[zoff + kk];
            }
            acc[i] += v;
        }
        q += 32usize;
    }
    for i in 0..wk {
        let tot = plane_sum(acc[i]);
        if lane == 0usize {
            out[s_off + i] = rhs[orig + i] - tot;
        }
    }
}

/// The kept blocks of the reduced solution into the full vector, one unit
/// per kept block.
#[cube(launch_unchecked)]
pub fn k_scatter_kept(plan: &[u32], xk: &[f64], x: &mut [f64], nk: u32) {
    let o_soff = plan[SK_S_OFF] as usize;
    let o_korig = plan[SK_ORIG_OFF] as usize;
    let o_kw = plan[SK_W] as usize;
    let k = ABSOLUTE_POS;
    if k < nk as usize {
        let s_off = plan[o_soff + k] as usize;
        let orig = plan[o_korig + k] as usize;
        let wk = plan[o_kw + k] as usize;
        for i in 0..wk {
            x[orig + i] = xk[s_off + i];
        }
    }
}

/// Back-substitution, one unit per eliminated block: `x_e = D_e^-1 (b_e -
/// sum_a C_a^T x_a)` over its observers, the kept slices of `x` in place.
#[cube(launch_unchecked)]
pub fn k_schur_backsub(plan: &[u32], h: &[f64], rhs: &[f64], dinv: &[f64], x: &mut [f64], ne: u32, #[comptime] ew: usize) {
    let o_w = plan[SE_W] as usize;
    let o_orig = plan[SE_ORIG_OFF] as usize;
    let o_optr = plan[SE_OBS_PTR] as usize;
    let o_dinv = plan[SE_DINV_OFF] as usize;
    let o_ca = plan[SO_CA_OFF] as usize;
    let o_trans = plan[SO_TRANS] as usize;
    let o_ow = plan[SO_W] as usize;
    let o_oorig = plan[SO_ORIG_OFF] as usize;
    let e = ABSOLUTE_POS;
    if e < ne as usize {
        let w = plan[o_w + e] as usize;
        let orig = plan[o_orig + e] as usize;
        let doff = plan[o_dinv + e] as usize;
        let q0 = plan[o_optr + e] as usize;
        let q1 = plan[o_optr + e + 1] as usize;
        let mut t = Array::<f64>::new(ew);
        for i in 0..w {
            t[i] = rhs[orig + i];
        }
        for o in q0..q1 {
            let wa = plan[o_ow + o] as usize;
            let ca = plan[o_ca + o] as usize;
            let trans = plan[o_trans + o];
            let xo = plan[o_oorig + o] as usize;
            for kk in 0..w {
                let mut v = 0.0f64;
                for i in 0..wa {
                    v += coupling(h, ca, trans, wa, w, i, kk) * x[xo + i];
                }
                t[kk] -= v;
            }
        }
        for i in 0..w {
            let mut v = 0.0f64;
            for j in 0..w {
                v += dinv[doff + i + j * w] * t[j];
            }
            x[orig + i] = v;
        }
    }
}

/// `r -= S x` over the tiles of S (upper block storage, diagonal tiles
/// upper-only), one plane per tile, the lanes over the tile's rows for the
/// product and over its columns for the transposed one; nothing unless
/// `gate[0]` is nonzero.
#[cube(launch_unchecked)]
pub fn k_symv_sub(gate: &[u32], plan: &[u32], s: &[f64], x: &[f64], r: &mut [Atomic<f64>]) {
    if gate[0] == 0u32 {
        terminate!();
    }
    let o_val = plan[ST_VAL_OFF] as usize;
    let o_wr = plan[ST_WR] as usize;
    let o_wc = plan[ST_WC] as usize;
    let o_a = plan[ST_A] as usize;
    let o_b = plan[ST_B] as usize;
    let o_aoff = plan[ST_A_OFF] as usize;
    let o_boff = plan[ST_B_OFF] as usize;
    let t = CUBE_POS as usize;
    let lane = UNIT_POS as usize;
    let val = plan[o_val + t] as usize;
    let wr = plan[o_wr + t] as usize;
    let wc = plan[o_wc + t] as usize;
    let a = plan[o_a + t];
    let b = plan[o_b + t];
    let a_off = plan[o_aoff + t] as usize;
    let b_off = plan[o_boff + t] as usize;
    if a == b {
        let mut i = lane;
        while i < wr {
            let mut acc = 0.0f64;
            for j in 0..wc {
                let mut v = s[val + i + j * wr];
                if i > j {
                    v = s[val + j + i * wr];
                }
                acc += v * x[b_off + j];
            }
            r[a_off + i].fetch_add(0.0f64 - acc);
            i += 32usize;
        }
    } else {
        let mut i = lane;
        while i < wr {
            let mut acc = 0.0f64;
            for j in 0..wc {
                acc += s[val + i + j * wr] * x[b_off + j];
            }
            r[a_off + i].fetch_add(0.0f64 - acc);
            i += 32usize;
        }
        let mut j = lane;
        while j < wc {
            let mut acc = 0.0f64;
            for i2 in 0..wr {
                acc += s[val + i2 + j * wr] * x[a_off + i2];
            }
            r[b_off + j].fetch_add(0.0f64 - acc);
            j += 32usize;
        }
    }
}

/// `dst = src` when `gate[0]` is nonzero.
#[cube(launch_unchecked)]
pub fn k_copy_f64(gate: &[u32], dst: &mut [f64], src: &[f64], n: u32) {
    if gate[0] != 0u32 && ABSOLUTE_POS < n as usize {
        dst[ABSOLUTE_POS] = src[ABSOLUTE_POS];
    }
}

// The refinement's state on the device: f64 words and u32 flags, read back
// with the step.
const ST_BNORM: usize = 0;
const ST_TOL: usize = 1;
const ST_RNORM: usize = 2;
/// A refinement step is due; the gate of the step's kernels.
const F_ACTIVE: usize = 0;
const F_STEPS: usize = 1;
/// The last step did not reduce the residual and is undone.
const F_REVERT: usize = 2;
/// An eliminated block's slot plus one when its diagonal is not definite.
const F_ELIM: usize = 3;
/// Nonzero when S's factorization met a pivot that is not positive.
const F_FACTOR: usize = 4;
const N_FLAGS: usize = 8;
/// Bytes from the state's start to the flags', and the state's size.
const FLAGS_AT: usize = 64;
const STATUS_BYTES: usize = FLAGS_AT + 4 * N_FLAGS;

/// The sum of squares of `x[0..n]` by one cube of 256 units, returned to
/// every unit.
#[cube]
fn cube_sumsq(x: &[f64], n: usize) -> f64 {
    let t = UNIT_POS as usize;
    let mut acc = 0.0f64;
    let mut i = t;
    while i < n {
        acc += x[i] * x[i];
        i += 256usize;
    }
    let mut sh = Shared::<[f64]>::new_slice(256usize);
    // a previous call's readers are done with the shared words first
    sync_cube();
    sh[t] = acc;
    sync_cube();
    let mut half = 128usize;
    while half > 0usize {
        if t < half {
            sh[t] += sh[t + half];
        }
        sync_cube();
        half = half / 2usize;
    }
    sh[0]
}

/// The refinement's start, one cube of 256 units, after the first solve's
/// residual `r` of the right-hand side `b`: the norms, the tolerance
/// `tol_rel ||b||`, and whether a step is due.
#[cube(launch_unchecked)]
pub fn k_refine_start(b: &[f64], r: &[f64], st: &mut [f64], flags: &mut [u32], n: u32, tol_rel: f64) {
    let bn = f64::sqrt(cube_sumsq(b, n as usize));
    let rn = f64::sqrt(cube_sumsq(r, n as usize));
    if UNIT_POS == 0u32 {
        let tol = tol_rel * bn;
        st[ST_BNORM] = bn;
        st[ST_TOL] = tol;
        st[ST_RNORM] = rn;
        flags[F_STEPS] = 0u32;
        flags[F_REVERT] = 0u32;
        if rn > tol {
            flags[F_ACTIVE] = 1u32;
        } else {
            flags[F_ACTIVE] = 0u32;
        }
    }
}

/// A refinement step's update, one unit per entry, when the step is due:
/// `x` kept in `x_prev`, the correction `d` added to it.
#[cube(launch_unchecked)]
pub fn k_refine_apply(flags: &[u32], x: &mut [f64], x_prev: &mut [f64], d: &[f64], n: u32) {
    let i = ABSOLUTE_POS;
    if flags[F_ACTIVE] != 0u32 && i < n as usize {
        x_prev[i] = x[i];
        x[i] += d[i];
    }
}

/// A due refinement step's judgement, one cube of 256 units, from the
/// residual `r` of the updated iterate: a step that did not reduce the
/// residual's norm is to be undone and ends the refinement; otherwise the
/// norm is kept and the refinement goes on while it is above the tolerance.
#[cube(launch_unchecked)]
pub fn k_refine_judge(r: &[f64], st: &mut [f64], flags: &mut [u32], n: u32) {
    if flags[F_ACTIVE] == 0u32 {
        terminate!();
    }
    let next = f64::sqrt(cube_sumsq(r, n as usize));
    if UNIT_POS == 0u32 {
        flags[F_STEPS] += 1u32;
        if next >= st[ST_RNORM] {
            // the factor is too far off for the correction to help
            flags[F_REVERT] = 1u32;
            flags[F_ACTIVE] = 0u32;
        } else {
            st[ST_RNORM] = next;
            if !(next > st[ST_TOL]) {
                flags[F_ACTIVE] = 0u32;
            }
        }
    }
}

/// The iterate before a rejected step back, one unit per entry.
#[cube(launch_unchecked)]
pub fn k_refine_revert(flags: &[u32], x: &mut [f64], x_prev: &[f64], n: u32) {
    let i = ABSOLUTE_POS;
    if flags[F_REVERT] != 0u32 && i < n as usize {
        x[i] = x_prev[i];
    }
}

// ---------------------------------------------------------------- host

/// What a refined solve took: steps, the final and the right-hand side's
/// norm, and whether it stayed above the tolerance.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefineStats {
    pub steps: usize,
    pub rnorm: f64,
    pub bnorm: f64,
    pub short: bool,
}

/// Why a [`GpuSchur::solve`] failed; the caller raises the damping.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuSchurError {
    /// The diagonal tile of the eliminated block `block` is not positive
    /// definite.
    Eliminated { block: usize },
    /// The reduced system is not positive definite.
    Reduced,
}

/// The device plan of a [`SchurSymbolic`], and the device state of one
/// reduction: the Hessian's values and gradient as uploaded, the block
/// inverses, W, S and the reduced right-hand side, the solution.
pub struct GpuSchur {
    /// The last attempt's launches as a graph, with what it was captured
    /// against; first, so it is dropped before the buffers it records.
    graph: Option<Captured>,
    graphs: bool,
    replayed: bool,
    ctx: GpuContext,
    plan: Lease,
    plan_len: usize,
    /// A word holding 1, in the plan's buffer: the gate of kernels that
    /// always run.
    d_on: Handle,
    ne: u32,
    no: u32,
    nt: u32,
    nk: u32,
    n: usize,
    nks: usize,
    s_len: usize,
    ew_cap: usize,
    kw_cap: usize,
    acc_cap: usize,
    /// per eliminated slot, its block id (for the error)
    elim_blk: Vec<u32>,
    /// The Hessian's tile values and the gradient as last uploaded, in f64.
    h: Upload<f64>,
    h_len: usize,
    rhs: Upload<f64>,
    dinv: Lease,
    z: Lease,
    w: Lease,
    s: Lease,
    rhs_kept: Lease,
    x_kept: Lease,
    x_prev: Lease,
    r: Lease,
    /// What a solve reads back, in one copy: the full step (`n` f64), then
    /// at `status_at` the refinement's state and the flags.
    out: Lease,
    status_at: usize,
}

/// What a captured attempt depends on beyond the reduction's own buffers:
/// the factor whose kernels it recorded, the refinement's launches and
/// tolerance, and the buffers the uploads land in.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct GraphKey {
    factor: u64,
    max_steps: usize,
    tol_bits: u64,
    h: u64,
    h_len: usize,
    rhs: u64,
}

struct Captured {
    key: GraphKey,
    graph: cubecl_runtime::client::Graph,
}

fn round_cap(w: usize) -> usize {
    [4usize, 8, 16, 32, 64].into_iter().find(|&c| c >= w).unwrap_or_else(|| panic!("a block of {w} columns is wider than the reduction's kernels take (64)"))
}

impl GpuSchur {
    /// Build the device plan of `sym`, the reduction of a matrix with the
    /// block structure `hsym`, and lease the reduction's buffers from `ctx`
    /// (returned on drop). The plan goes up in one copy.
    pub fn new(ctx: &GpuContext, sym: &SchurSymbolic<SparseIndex>, hsym: &SymbolicSparseBlockColMat<SparseIndex>) -> Self {
        let ne = sym.elim_diag().len();
        let no = sym.obs_w().len();
        let nk = sym.kept.len();
        let n = hsym.ncols();
        let nks = sym.s.nrows();
        let hpart = hsym.col_part();
        // the eliminated blocks
        let (mut e_diag, mut e_w, mut e_orig, mut e_dinv, mut e_z, mut elim_blk) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let (mut dinv_len, mut z_len) = (0u32, 0u32);
        let mut ew_max = 1usize;
        for slot in 0..ne {
            let dblk = sym.elim_diag()[slot] as usize;
            let e = hsym.blk_row(dblk);
            let we = sym.elim_w()[slot] as usize;
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
        let e_obs_ptr: Vec<u32> = sym.elim_obs_ptr().to_vec();
        // the observers, with their eliminated slot and their W offset
        let mut o_e = vec![0u32; no];
        for slot in 0..ne {
            o_e[e_obs_ptr[slot] as usize..e_obs_ptr[slot + 1] as usize].fill(slot as u32);
        }
        let o_ca: Vec<u32> = sym.obs_ca_off().to_vec();
        let o_trans: Vec<u32> = sym.obs_trans().iter().map(|&t| t as u32).collect();
        let o_w: Vec<u32> = sym.obs_w().to_vec();
        let o_orig: Vec<u32> = sym.obs_orig_off().to_vec();
        let mut o_woff = Vec::with_capacity(no);
        let mut w_len = 0u32;
        let mut kw_max = 1usize;
        for o in 0..no {
            o_woff.push(w_len);
            w_len += o_w[o] * e_w[o_e[o] as usize];
            kw_max = kw_max.max(o_w[o] as usize);
        }
        // the tiles of S, in storage order, with their kept-kept source
        let ssym = &sym.s;
        let spart = ssym.col_part();
        let nt = ssym.nblocks();
        let mut copy_of = vec![NONE; nt];
        let (csrc, cdst) = sym.copy_tiles();
        for (&src, &dst) in csrc.iter().zip(cdst) {
            copy_of[dst as usize] = hsym.val_range(src as usize).start as u32;
        }
        let (mut t_val, mut t_wr, mut t_wc, mut t_a, mut t_b, mut t_aoff, mut t_boff, mut t_copy) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut tile_index = std::collections::HashMap::<u32, u32>::new(); // S value offset -> tile index
        for bcol in 0..ssym.nblk_cols() {
            for t in ssym.col_range(bcol) {
                let a = ssym.blk_row(t);
                let (wr, wc) = ssym.block_dims(t);
                kw_max = kw_max.max(wr).max(wc);
                let v = ssym.val_range(t).start as u32;
                tile_index.insert(v, t_val.len() as u32);
                t_val.push(v);
                t_wr.push(wr as u32);
                t_wc.push(wc as u32);
                t_a.push(a as u32);
                t_b.push(bcol as u32);
                t_aoff.push(spart[a]);
                t_boff.push(spart[bcol]);
                t_copy.push(copy_of[t]);
            }
        }
        // the pairs, by tile: every (oa, ob) of one eliminated block with kept(oa) <= kept(ob)
        let kept_of_obs = sym.obs_kept();
        let mut pairs: Vec<(u32, u32, u32)> = Vec::with_capacity(sym.pair_count());
        for slot in 0..ne {
            let (q0, q1) = (e_obs_ptr[slot] as usize, e_obs_ptr[slot + 1] as usize);
            for i in q0..q1 {
                for j in i..q1 {
                    let (ka, kb) = (kept_of_obs[i] as usize, kept_of_obs[j] as usize);
                    let v = sym.s_tile(ka, kb).expect("every observer pair has a tile in S");
                    pairs.push((tile_index[&v], i as u32, j as u32));
                }
            }
        }
        pairs.sort_unstable_by_key(|p| p.0);
        let mut t_pair_ptr = vec![0u32; nt + 1];
        for p in &pairs {
            t_pair_ptr[p.0 as usize + 1] += 1;
        }
        for t in 0..nt {
            t_pair_ptr[t + 1] += t_pair_ptr[t];
        }
        let p_oa: Vec<u32> = pairs.iter().map(|p| p.1).collect();
        let p_ob: Vec<u32> = pairs.iter().map(|p| p.2).collect();
        drop(pairs);
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
        // pack
        let mut buf: Vec<u32> = vec![0; SHEADER];
        let arrays: [(usize, &[u32]); 28] = [
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
            (SP_OA, &p_oa),
            (SP_OB, &p_ob),
            (SK_OBS_PTR, &k_obs_ptr),
            (SK_OBS, &k_obs),
            (SK_S_OFF, &k_s_off),
            (SK_ORIG_OFF, &k_orig),
            (SK_W, &k_w),
        ];
        for (slot, v) in arrays {
            buf[slot] = buf.len() as u32;
            buf.extend_from_slice(v);
        }
        let plan_len = buf.len();
        let mut pk = Packer::default();
        pk.push(&buf);
        let at_on = pk.push(&[1]);
        let plan = pk.upload(ctx);
        let d_on = Packer::view(plan.handle(), at_on);
        let s_len = ssym.val_count();
        let f64s = core::mem::size_of::<f64>();
        let ew_cap = round_cap(ew_max);
        let kw_cap = round_cap(kw_max);
        let empty = |n: usize| ctx.lease(n.max(1) * f64s);
        let status_at = (n * f64s).div_ceil(256) * 256;
        GpuSchur {
            dinv: empty(dinv_len as usize),
            z: empty(z_len as usize),
            w: empty(w_len as usize),
            s: empty(s_len),
            rhs_kept: empty(nks),
            x_kept: empty(nks),
            x_prev: empty(nks),
            r: empty(nks),
            out: ctx.lease(status_at + STATUS_BYTES),
            status_at,
            graph: None,
            graphs: true,
            replayed: false,
            ctx: ctx.clone(),
            plan,
            plan_len,
            d_on,
            ne: ne as u32,
            no: no as u32,
            nt: nt as u32,
            nk: nk as u32,
            n,
            nks,
            s_len,
            ew_cap,
            kw_cap,
            acc_cap: (kw_cap * kw_cap).div_ceil(32),
            elim_blk,
            h: Upload::default(),
            h_len: 0,
            rhs: Upload::default(),
        }
    }

    /// The reduced system's scalar dimension.
    pub fn kept_size(&self) -> usize {
        self.nks
    }

    /// S's values on the device, in the storage order of the symbolic's `s`,
    /// as the last reduction left them.
    pub fn s_values(&self) -> (&Handle, usize) {
        (self.s.handle(), self.s_len)
    }

    /// The reduced right-hand side on the device.
    pub fn rhs_kept(&self) -> &Handle {
        self.rhs_kept.handle()
    }

    /// A view of the flags from flag `f` on.
    fn flags(&self, f: usize) -> Handle {
        self.out.handle().clone().offset_start((self.status_at + FLAGS_AT + 4 * f) as u64)
    }

    /// A view of the refinement's f64 state.
    fn state(&self) -> Handle {
        self.out.handle().clone().offset_start(self.status_at as u64)
    }

    /// Upload the Hessian's values and the gradient, in place, without
    /// waiting.
    ///
    /// # Safety
    ///
    /// As [`Upload::write`]: `hvals` and `rhs` stay borrowed and unchanged
    /// until a blocking read on the context's client has returned.
    unsafe fn upload<M: arael_faer::schur::SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) {
        assert_eq!(rhs.len(), self.n);
        let ctx = self.ctx.clone();
        unsafe {
            self.h.write(&ctx, hvals, |v| v.to_f64());
            self.rhs.write(&ctx, rhs, |v| v.to_f64());
        }
        self.h_len = hvals.len();
    }

    /// Zero the flags and launch the reduction, from the uploaded values.
    fn launch_reduce(&self) {
        let c = self.ctx.client();
        let arr = |h: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(h.clone(), len.max(1)) };
        let d256 = CubeDim::new_1d(256);
        let d32 = CubeDim::new_1d(32);
        let count = |n: u32| CubeCount::Static(n.max(1), 1, 1);
        let (h, r) = (self.h.handle(), self.rhs.handle());
        let plen = self.plan_len;
        unsafe {
            crate::kernels::k_zero_u32::launch_unchecked(c, count(1), d32, arr(&self.flags(0), N_FLAGS), N_FLAGS as u32);
            k_schur_elim::launch_unchecked(c, count(self.ne.div_ceil(256)), d256, arr(self.plan.handle(), plen), arr(h, self.h_len), arr(r, self.n), arr(self.dinv.handle(), 1), arr(self.z.handle(), 1), arr(&self.flags(F_ELIM), 1), self.ne, self.ew_cap);
            k_schur_w::launch_unchecked(c, count(self.no.div_ceil(256)), d256, arr(self.plan.handle(), plen), arr(h, self.h_len), arr(self.dinv.handle(), 1), arr(self.w.handle(), 1), self.no);
            k_schur_tiles::launch_unchecked(c, count(self.nt), d32, arr(self.plan.handle(), plen), arr(h, self.h_len), arr(self.w.handle(), 1), arr(self.s.handle(), self.s_len), self.acc_cap);
            k_schur_rhs::launch_unchecked(c, count(self.nk), d32, arr(self.plan.handle(), plen), arr(h, self.h_len), arr(r, self.n), arr(self.z.handle(), 1), arr(self.rhs_kept.handle(), self.nks), self.kw_cap);
        }
    }

    /// Form S and the reduced right-hand side from the damped Hessian's tile
    /// values `hvals` (in `hsym`'s storage order) and the gradient `rhs`, and
    /// wait for them (for inspection: [`solve`](Self::solve) reduces as its
    /// first stage). `Err` when an eliminated block's diagonal is not
    /// positive definite.
    pub fn reduce<M: arael_faer::schur::SchurReal + 'static>(&mut self, hvals: &[M], rhs: &[M]) -> Result<(), SchurError> {
        let ctx = self.ctx.clone();
        let flight = crate::host::InFlight::new(ctx.client());
        // SAFETY: `hvals` and `rhs` stay borrowed until the read below.
        unsafe { self.upload(hvals, rhs) };
        self.launch_reduce();
        let flags = ctx.client().read_one(self.out.view(self.status_at + STATUS_BYTES)).expect("read the flags");
        flight.done();
        let failed = bytemuck::pod_read_unaligned::<u32>(&flags[self.status_at + FLAGS_AT + 4 * F_ELIM..][..4]);
        if failed != 0 { Err(SchurError::NotPositiveDefinite { block: self.elim_blk[failed as usize - 1] as usize }) } else { Ok(()) }
    }

    /// Whether [`solve`](Self::solve) captures its launches as a graph and
    /// replays it (the default): the first solve with a factor and a
    /// refinement setting runs the kernels and then records them, and the
    /// solves after it replay the recording, one launch instead of hundreds.
    pub fn use_graphs(&mut self, on: bool) {
        self.graphs = on;
        if !on {
            self.graph = None;
        }
    }

    /// Whether the last [`solve`](Self::solve) replayed a captured graph.
    pub fn replayed(&self) -> bool {
        self.replayed
    }

    /// The launches of one damped solve after the uploads, all of them
    /// kernels: the reduction, the factorization, the solve, `max_steps`
    /// refinement steps, the back-substitution. The same sequence every
    /// time for the same factor and refinement, which is what makes it a
    /// graph.
    fn launch_attempt(&self, factor: &mut GpuFactor, max_steps: usize, tol_rel: f64) {
        self.launch_reduce();
        let c = self.ctx.client();
        factor.factorize_device(self.s.handle(), self.s_len, &self.flags(F_FACTOR));
        let arr = |h: &Handle, len: usize| unsafe { BufferArg::from_raw_parts(h.clone(), len.max(1)) };
        let d256 = CubeDim::new_1d(256);
        let d32 = CubeDim::new_1d(32);
        let count = |n: u32| CubeCount::Static(n.max(1), 1, 1);
        let plen = self.plan_len;
        let nks = self.nks as u32;
        let per_entry = count(nks.div_ceil(256));
        let (on, gate, state) = (self.d_on.clone(), self.flags(F_ACTIVE), self.state());
        let copy = |gate: &Handle, dst: &Handle, src: &Handle| unsafe { k_copy_f64::launch_unchecked(c, per_entry.clone(), d256, arr(gate, 1), arr(dst, self.nks), arr(src, self.nks), nks) };
        // r = b - S x
        let residual = |gate: &Handle| unsafe {
            copy(gate, self.r.handle(), self.rhs_kept.handle());
            k_symv_sub::launch_unchecked(c, count(self.nt), d32, arr(gate, 1), arr(self.plan.handle(), plen), arr(self.s.handle(), self.s_len), arr(self.x_kept.handle(), self.nks), arr(self.r.handle(), self.nks));
        };
        copy(&on, self.x_kept.handle(), self.rhs_kept.handle());
        factor.solve_device(self.x_kept.handle());
        if max_steps > 0 {
            residual(&on);
            unsafe { k_refine_start::launch_unchecked(c, count(1), d256, arr(self.rhs_kept.handle(), self.nks), arr(self.r.handle(), self.nks), arr(&state, 4), arr(&gate, N_FLAGS), nks, tol_rel) };
            for _ in 0..max_steps {
                factor.solve_device_if(self.r.handle(), &gate);
                unsafe { k_refine_apply::launch_unchecked(c, per_entry.clone(), d256, arr(&gate, N_FLAGS), arr(self.x_kept.handle(), self.nks), arr(self.x_prev.handle(), self.nks), arr(self.r.handle(), self.nks), nks) };
                residual(&gate);
                unsafe { k_refine_judge::launch_unchecked(c, count(1), d256, arr(self.r.handle(), self.nks), arr(&state, 4), arr(&gate, N_FLAGS), nks) };
            }
            unsafe { k_refine_revert::launch_unchecked(c, per_entry.clone(), d256, arr(&gate, N_FLAGS), arr(self.x_kept.handle(), self.nks), arr(self.x_prev.handle(), self.nks), nks) };
        }
        // the kept blocks into the full vector, then the eliminated ones
        unsafe {
            k_scatter_kept::launch_unchecked(c, count(self.nk.div_ceil(256)), d256, arr(self.plan.handle(), plen), arr(self.x_kept.handle(), self.nks), arr(self.out.handle(), self.n), self.nk);
            k_schur_backsub::launch_unchecked(c, count(self.ne.div_ceil(256)), d256, arr(self.plan.handle(), plen), arr(self.h.handle(), self.h_len), arr(self.rhs.handle(), self.n), arr(self.dinv.handle(), 1), arr(self.out.handle(), self.n), self.ne, self.ew_cap);
        }
    }

    /// One damped solve, entirely on the device, waiting once: form S and
    /// the reduced right-hand side from the Hessian's tile values `hvals`
    /// (in `hsym`'s storage order) and the gradient `rhs`, factorize S with
    /// `factor`, solve for the kept blocks, refine the solution against S
    /// when `refine` is `Some((max_steps, tol))` (an f32 factor; `tol`
    /// relative to the right-hand side, floored by the caller), recover the
    /// eliminated blocks, and read the full step into `delta` with the
    /// pivot flags and the refinement's state.
    ///
    /// The refinement decides on the device: `max_steps` steps are
    /// launched, and those after it has converged or stalled do nothing. A
    /// failed pivot is found with the result, so a failing solve runs to
    /// the end; `delta` is then left as it was. The launches are captured as
    /// a graph and replayed by the solves after it (see
    /// [`use_graphs`](Self::use_graphs)).
    pub fn solve<M: arael_faer::schur::SchurReal + 'static>(
        &mut self,
        factor: &mut GpuFactor,
        hvals: &[M],
        rhs: &[M],
        refine: Option<(usize, f64)>,
        delta: &mut [M],
    ) -> Result<RefineStats, GpuSchurError> {
        assert_eq!(delta.len(), self.n);
        assert_eq!(rhs.len(), self.n);
        assert!(self.ctx.same(factor.context()), "the factor runs on the reduction's context, so S never leaves its stream");
        let ctx = self.ctx.clone();
        let c = ctx.client();
        let (max_steps, tol_rel) = refine.filter(|r| r.0 > 0).unwrap_or((0, 0.0));
        // the buffers the uploads land in are fixed first: a graph replays
        // against the addresses it was captured with
        let key = GraphKey {
            factor: factor.id(),
            max_steps,
            tol_bits: tol_rel.to_bits(),
            h: self.h.reserve(&ctx, hvals.len()).id(),
            h_len: hvals.len(),
            rhs: self.rhs.reserve(&ctx, rhs.len()).id(),
        };
        let replay = self.graphs && self.graph.as_ref().is_some_and(|g| g.key == key);
        let capture = self.graphs && !replay;
        if capture {
            self.graph = None;
            // before the run that warms the pools up for the recording
            c.graph_prepare().expect("prepare the attempt's graph capture");
        }
        let flight = crate::host::InFlight::new(c);
        // SAFETY: `hvals` and `rhs` stay borrowed until the read below.
        unsafe { self.upload(hvals, rhs) };
        if replay {
            // SAFETY: the graph's buffers are this reduction's leases, the
            // factor's (its id is in the key) and the uploads' (theirs too),
            // all alive; the context's client is pinned to the stream the
            // graph was captured on, which orders the uploads before and the
            // read after it.
            unsafe { self.graph.as_ref().unwrap().graph.replay() }.expect("replay the attempt's graph");
        } else {
            self.launch_attempt(factor, max_steps, tol_rel);
        }
        self.replayed = replay;
        let bytes = c.read_one(self.out.view(self.status_at + STATUS_BYTES)).expect("read the step");
        flight.done();
        // the attempt is done: rotating CubeCL's drop queue returns the
        // staging its uploads held to the pinned pool, which otherwise grows
        // (the read above has waited, so the rotation's fence wait is free)
        c.flush().expect("device flush");
        if capture {
            // The run above warmed the pools: record the same launches, which
            // do not execute now, for the solves after this one. CubeCL opens
            // CUDA's capture in global mode, where a sync from any stream of
            // the process fails while it is open, so the window runs on the
            // device's thread with nothing else scheduled against it.
            let this = &*self;
            let graph = c
                .exclusive(move || {
                    c.start_capture().expect("start the attempt's graph capture");
                    this.launch_attempt(factor, max_steps, tol_rel);
                    c.stop_capture().expect("capture the attempt's graph")
                })
                .expect("run the graph capture exclusively");
            self.graph = Some(Captured { key, graph });
        }
        let word = |at: usize| bytemuck::pod_read_unaligned::<u32>(&bytes[at..at + 4]);
        let real = |at: usize| bytemuck::pod_read_unaligned::<f64>(&bytes[at..at + 8]);
        let flag = |f: usize| word(self.status_at + FLAGS_AT + 4 * f);
        let elim = flag(F_ELIM);
        if elim != 0 {
            return Err(GpuSchurError::Eliminated { block: self.elim_blk[elim as usize - 1] as usize });
        }
        if flag(F_FACTOR) != 0 {
            return Err(GpuSchurError::Reduced);
        }
        for (d, &v) in delta.iter_mut().zip(f64::from_bytes(&bytes[..self.n * 8])) {
            *d = M::from_f64(v);
        }
        let mut stats = RefineStats::default();
        if max_steps > 0 {
            let st = |i: usize| real(self.status_at + 8 * i);
            stats = RefineStats { steps: flag(F_STEPS) as usize, rnorm: st(ST_RNORM), bnorm: st(ST_BNORM), short: st(ST_RNORM) > st(ST_TOL) };
        }
        Ok(stats)
    }

    /// The reduced right-hand side and S's values read back (for tests).
    pub fn download(&self) -> (Vec<f64>, Vec<f64>) {
        crate::sync(self.ctx.client());
        let s = f64::from_bytes(&self.ctx.client().read_one(self.s.view(self.s_len * 8)).expect("read S")).to_vec();
        let r = f64::from_bytes(&self.ctx.client().read_one(self.rhs_kept.view(self.nks * 8)).expect("read rhs")).to_vec();
        (s, r)
    }
}

impl GpuFactor {
    /// As [`crate::GpuSupernodal::factorize_device`].
    pub fn factorize_device(&mut self, vals: &Handle, nvals: usize, fail: &Handle) {
        match self {
            GpuFactor::F32(g) => g.factorize_device(vals, nvals, fail),
            GpuFactor::F64(g) => g.factorize_device(vals, nvals, fail),
        }
    }

    /// As [`crate::GpuSupernodal::solve_device`].
    pub fn solve_device(&mut self, x: &Handle) {
        match self {
            GpuFactor::F32(g) => g.solve_device(x),
            GpuFactor::F64(g) => g.solve_device(x),
        }
    }

    /// As [`crate::GpuSupernodal::solve_device_if`].
    pub fn solve_device_if(&mut self, x: &Handle, gate: &Handle) {
        match self {
            GpuFactor::F32(g) => g.solve_device_if(x, gate),
            GpuFactor::F64(g) => g.solve_device_if(x, gate),
        }
    }
}


#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::test_util::{ba_like, rel_residual};
    use crate::{GpuBackend, GpuContext, GpuPrecision, GpuSupernodal};
    use arael_faer::schur::{SchurContext, schur_reduce, schur_symbolic};
    use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};

    fn err_max(a: &[f64], b: &[f64]) -> f64 {
        let m = b.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        a.iter().zip(b).fold(0.0f64, |e, (x, y)| e.max((x - y).abs())) / m
    }

    #[test]
    fn the_reduction_matches_the_cpu() {
        let (h, _, rhs, elim) = ba_like(40, 300, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let mut s_cpu = sym.alloc_s::<f64>();
        let mut rhs_cpu = vec![0.0f64; sym.kept_size()];
        let mut ctx = SchurContext::new();
        schur_reduce(&sym, &h, &rhs, &mut ctx, &mut s_cpu, &mut rhs_cpu).unwrap();
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let mut g = GpuSchur::new(&ctx, &sym, h.symbolic());
        g.reduce(h.vals(), &rhs).unwrap();
        let (s_gpu, rhs_gpu) = g.download();
        assert_eq!(s_gpu.len(), s_cpu.vals().len());
        let es = err_max(&s_gpu, s_cpu.vals());
        let er = err_max(&rhs_gpu, &rhs_cpu);
        assert!(es < 1e-12 && er < 1e-12, "S err/max {es}, rhs err/max {er}");
        assert!(sym.obs_trans().iter().any(|&t| t) && sym.obs_trans().iter().any(|&t| !t), "both tile orientations in the fixture");
    }

    #[test]
    fn the_reduced_solve_solves_the_whole_system() {
        let (h, dense, rhs, elim) = ba_like(40, 300, 9);
        let n = h.symbolic().ncols();
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let mut g = GpuSchur::new(&ctx, &sym, h.symbolic());
        for (precision, refine, tol) in [(GpuPrecision::F64, None, 1e-10), (GpuPrecision::F32, Some((4usize, 1e-10f64)), 1e-8)] {
            let mut f = GpuFactor::new(&ctx, precision, &sn, &sym.s, &Default::default());
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
        let _ = GpuSupernodal::<f64>::dim;
    }

    /// The refinement decides on the device: it stops once below the
    /// tolerance, launched steps after that doing nothing; with a tolerance
    /// it cannot reach it stops at the first step that does not reduce the
    /// residual, and undoes that step.
    #[test]
    fn the_refinement_stops_on_the_device() {
        let (h, dense, rhs, elim) = ba_like(40, 300, 9);
        let n = h.symbolic().ncols();
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let mut g = GpuSchur::new(&ctx, &sym, h.symbolic());
        let mut f = GpuFactor::new(&ctx, GpuPrecision::F32, &sn, &sym.s, &Default::default());
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

    /// The second solve with a factor and a refinement setting replays the
    /// graph the first captured, against new values, and lands where the
    /// launched kernels land; a new factor or another refinement setting is
    /// captured anew.
    #[test]
    fn replayed_solves_match_launched_ones() {
        let (h, dense, rhs, elim) = ba_like(40, 300, 9);
        let n = h.symbolic().ncols();
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        // another damping: the replay must see the values written before it
        let hs = h.symbolic().clone();
        let mut h2 = h.clone();
        for blk in 0..hs.nblk_cols() {
            let t = hs.col_range(blk).find(|&b| hs.blk_row(b) == blk).unwrap();
            let (w, _) = hs.block_dims(t);
            let base = hs.val_range(t).start;
            for i in 0..w {
                h2.vals_mut()[base + i * w + i] *= 1.5;
            }
        }
        let mut dense2 = dense.clone();
        for i in 0..n {
            dense2[i + i * n] *= 1.5;
        }
        for precision in [GpuPrecision::F64, GpuPrecision::F32] {
            let refine = (precision == GpuPrecision::F32).then_some((4, 1e-10));
            let mut launched = GpuSchur::new(&ctx, &sym, h.symbolic());
            launched.use_graphs(false);
            let mut fl = GpuFactor::new(&ctx, precision, &sn, &sym.s, &Default::default());
            let mut g = GpuSchur::new(&ctx, &sym, h.symbolic());
            let mut f = GpuFactor::new(&ctx, precision, &sn, &sym.s, &Default::default());
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
            let mut f2 = GpuFactor::new(&ctx, precision, &sn, &sym.s, &Default::default());
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

    /// A capture on one context while another context reads on another
    /// thread: CUDA's global capture mode fails any sync of the process while
    /// a capture is open, so the capture has to keep the device to itself.
    #[test]
    fn captures_leave_other_contexts_alone() {
        let (h, dense, rhs, elim) = ba_like(30, 200, 4);
        let n = h.symbolic().ncols();
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        let stop = std::sync::atomic::AtomicBool::new(false);
        std::thread::scope(|scope| {
            let reader = scope.spawn(|| {
                let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
                let mut g = GpuSchur::new(&ctx, &sym, h.symbolic());
                g.use_graphs(false);
                let mut f = GpuFactor::new(&ctx, GpuPrecision::F32, &sn, &sym.s, &Default::default());
                let mut solves = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) || solves < 20 {
                    let mut d = vec![0.0f64; n];
                    g.solve(&mut f, h.vals(), &rhs, Some((4, 1e-10)), &mut d).unwrap();
                    assert!(rel_residual(&dense, n, &d, &rhs) < 1e-8);
                    solves += 1;
                }
                solves
            });
            let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
            let mut g = GpuSchur::new(&ctx, &sym, h.symbolic());
            for _ in 0..10 {
                // a new factor each time: every first solve captures
                let mut f = GpuFactor::new(&ctx, GpuPrecision::F32, &sn, &sym.s, &Default::default());
                for _ in 0..2 {
                    let mut d = vec![0.0f64; n];
                    g.solve(&mut f, h.vals(), &rhs, Some((4, 1e-10)), &mut d).unwrap();
                    assert!(rel_residual(&dense, n, &d, &rhs) < 1e-8);
                }
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            assert!(reader.join().unwrap() >= 20);
        });
    }

    /// Solve after solve on one context, structures going up and down: once
    /// the largest has been seen, a new solver's setup leases every buffer
    /// from the arena and creates none.
    #[test]
    fn repeated_solves_on_one_context_stop_creating_buffers() {
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let systems: Vec<_> = [(30, 400, 1u64), (60, 900, 2), (20, 200, 3), (60, 900, 4)]
            .iter()
            .map(|&(np, nl, seed)| {
                let (h, dense, rhs, elim) = ba_like(np, nl, seed);
                let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
                let order = nd_block_order(&sym.s);
                let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
                (h, dense, rhs, sym, sn)
            })
            .collect();
        let mut created_after_largest = None;
        for round in 0..2 {
            for (i, (h, dense, rhs, sym, sn)) in systems.iter().enumerate() {
                let n = h.symbolic().ncols();
                let mut g = GpuSchur::new(&ctx, sym, h.symbolic());
                let mut f = GpuFactor::new(&ctx, GpuPrecision::F32, sn, &sym.s, &Default::default());
                let mut delta = vec![0.0f64; n];
                g.solve(&mut f, h.vals(), rhs, Some((4, 1e-10)), &mut delta).unwrap();
                assert!(rel_residual(dense, n, &delta, rhs) < 1e-8);
                if round == 0 && i == 1 {
                    created_after_largest = Some(ctx.buffers_created());
                }
            }
        }
        assert_eq!(Some(ctx.buffers_created()), created_after_largest, "no buffer created after the largest structure");
    }

    /// A pivot that fails, in an eliminated block or in S, comes back with
    /// the solve's result and leaves the step as it was; the next solve of
    /// the same objects starts clean.
    #[test]
    fn bad_pivots_are_reported_with_the_result() {
        let (good, dense, rhs, elim) = ba_like(10, 30, 2);
        let n = good.symbolic().ncols();
        let sym = schur_symbolic(good.symbolic(), &elim).unwrap();
        let asym = good.symbolic().clone();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        let diag0 = |blk: usize| asym.val_range(asym.col_range(blk).find(|&b| asym.blk_row(b) == blk).unwrap()).start;
        let ctx = GpuContext::new(GpuBackend::Cuda, 0).unwrap();
        let mut g = GpuSchur::new(&ctx, &sym, &asym);
        for precision in [GpuPrecision::F64, GpuPrecision::F32] {
            let mut f = GpuFactor::new(&ctx, precision, &sn, &sym.s, &Default::default());
            let refine = (precision == GpuPrecision::F32).then_some((4, 1e-10));
            // an eliminated block
            let e = elim[3];
            let mut h = good.clone();
            h.vals_mut()[diag0(e)] = -1.0;
            assert_eq!(g.reduce(h.vals(), &rhs), Err(SchurError::NotPositiveDefinite { block: e }));
            let mut delta = vec![7.0f64; n];
            assert_eq!(g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap_err(), GpuSchurError::Eliminated { block: e });
            assert!(delta.iter().all(|&v| v == 7.0), "the step is left as it was");
            // a kept block: S(k, k) is H(k, k) less a semidefinite sum
            let k = sym.kept[0] as usize;
            let mut h = good.clone();
            h.vals_mut()[diag0(k)] = -1e8;
            assert_eq!(g.solve(&mut f, h.vals(), &rhs, refine, &mut delta).unwrap_err(), GpuSchurError::Reduced);
            // and the good matrix again
            g.solve(&mut f, good.vals(), &rhs, refine, &mut delta).unwrap();
            assert!(rel_residual(&dense, n, &delta, &rhs) < 1e-8);
        }
    }
}
