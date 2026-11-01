//! The device kernels of the block supernodal Cholesky and its solve, one
//! launch per level unless said (docs/dev/GPU.md, E3 and E5 step 7).
//!
//! Loops with a run-time trip count are written unrolled where the loads
//! matter: LLVM's NVPTX path leaves them rolled, and each iteration then
//! waits for its own load.

// Cube code, not Rust: the casts fix the types the `#[cube]` macro sees,
// the DSL has no `div_ceil` or `partial_cmp`, and a kernel's parameters are
// its buffers.
#![allow(
    clippy::unnecessary_cast,
    clippy::manual_div_ceil,
    clippy::neg_cmp_op_on_partial_ord,
    clippy::assign_op_pattern,
    clippy::too_many_arguments
)]

use cubecl::prelude::*;

use crate::plan::{GROUP, H_CMAP, H_DIMS, H_PAIR_KW_PTR, H_PR_CMAP, H_PR_D, H_PR_R0, H_PR_RMAP, H_PR_S, H_RMAP, H_ROW_SC, H_ROW_SC_OFF, H_SC_START, H_SUP_BEGIN, H_VAL_PTR, NB, TB, TC, TK, TR, XR_CH};

#[cube(launch_unchecked)]
pub fn k_zero<F: Float>(buf: &mut [F], n: u32) {
    if ABSOLUTE_POS < n as usize {
        buf[ABSOLUTE_POS] = F::new(0.0f32);
    }
}

#[cube(launch_unchecked)]
pub fn k_zero_u32(buf: &mut [u32], n: u32) {
    if ABSOLUTE_POS < n as usize {
        buf[ABSOLUTE_POS] = 0u32;
    }
}

/// The seed: one unit per stored tile, scattered into its panel; `tiles` has
/// 6 entries per tile (destination, stride, transposed, value offset, rows,
/// columns). An upper tile lands transposed, a diagonal one as is.
#[cube(launch_unchecked)]
pub fn k_seed<F: Float>(tiles: &[u32], vals: &[F], factor: &mut [F], ntiles: u32) {
    let t = ABSOLUTE_POS;
    if t < ntiles as usize {
        let dst = tiles[6 * t] as usize;
        let stride = tiles[6 * t + 1] as usize;
        let trans = tiles[6 * t + 2];
        let off = tiles[6 * t + 3] as usize;
        let wr = tiles[6 * t + 4] as usize;
        let wc = tiles[6 * t + 5] as usize;
        if trans == 0u32 {
            for c in 0..wc {
                for r in 0..wr {
                    factor[dst + c * stride + r] = vals[off + c * wr + r];
                }
            }
        } else {
            for c in 0..wr {
                for r in 0..wc {
                    factor[dst + c * stride + r] = vals[off + c + r * wr];
                }
            }
        }
    }
}

/// The small descendants' updates: one cube per (target, TR x TC tile,
/// chunk), a group of GROUP units per rectangle with the units over the
/// rectangle's elements, accumulated in shared memory, subtracted into the
/// panel atomically (a tile may be several chunks). `factor` and `out` are
/// the same buffer: the descendants are read, the target written.
#[cube(launch_unchecked)]
pub fn k_tile_update<F: Float + CubePrimitive<Scalar: AtomicNumeric>>(
    items: &[u32],
    rects: &[u32],
    plan: &[u32],
    factor: &[F],
    out: &mut [Atomic<F>],
    ) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let o_pr_d = plan[H_PR_D] as usize;
    let o_pr_r0 = plan[H_PR_R0] as usize;
    let item = CUBE_POS as usize;
    let s = items[5 * item] as usize;
    let row0 = items[5 * item + 1] as usize;
    let col0 = items[5 * item + 2] as usize;
    let r0 = items[5 * item + 3] as usize;
    let r1 = items[5 * item + 4] as usize;
    let q = plan[o_dims + 2 * s] as usize;
    let h = plan[o_dims + 2 * s + 1] as usize;
    let u = Shared::<[Atomic<F>]>::new_slice(TR * TC);
    let t = UNIT_POS as usize;
    let nt = CUBE_DIM as usize;
    let mut e = t;
    while e < TR * TC {
        u[e].store(F::new(0.0f32));
        e += nt;
    }
    sync_cube();
    let g = t / GROUP;
    let lane = t % GROUP;
    let ngroups = nt / GROUP;
    let mut rr = r0 + g;
    while rr < r1 {
        let rb = 4 * rr;
        let pi = rects[rb] as usize;
        let geo = rects[rb + 1];
        let dra = rects[rb + 2] as usize;
        let dca = rects[rb + 3] as usize;
        let i0 = (geo & 0xffu32) as usize;
        let ni = ((geo >> 8u32) & 0xffu32) as usize;
        let j0 = ((geo >> 16u32) & 0xffu32) as usize;
        let nj = (geo >> 24u32) as usize;
        let d = plan[o_pr_d + pi] as usize;
        let pr0 = plan[o_pr_r0 + pi] as usize;
        let dq = plan[o_dims + 2 * d] as usize;
        let dh = plan[o_dims + 2 * d + 1] as usize;
        let dp = plan[o_val_ptr + d] as usize + pr0;
        let mut e2 = lane;
        while e2 < ni * nj {
            let jj = e2 / ni;
            let ii = e2 - jj * ni;
            let mut acc = F::new(0.0f32);
            for k in 0..dq {
                acc += factor[dp + k * dh + dra + ii] * factor[dp + k * dh + dca + jj];
            }
            u[(j0 + jj) * TR + i0 + ii].fetch_add(acc);
            e2 += GROUP;
        }
        rr += ngroups;
    }
    sync_cube();
    let panel = plan[o_val_ptr + s] as usize;
    let mut nrows = h - row0;
    if nrows > TR {
        nrows = TR;
    }
    let mut ncols = q - col0;
    if ncols > TC {
        ncols = TC;
    }
    let mut e3 = t;
    while e3 < TR * TC {
        let j = e3 / TR;
        let i = e3 - j * TR;
        if i < nrows && j < ncols && row0 + i >= col0 + j {
            let v = u[e3].load();
            out[panel + (col0 + j) * h + row0 + i].fetch_add(F::new(0.0f32) - v);
        }
        e3 += nt;
    }
}

/// The big descendants' updates: one cube per (target, TB x TB tile) looping
/// over the rectangles that land in the tile with a shared-memory GEMM core
/// (TK-wide k slabs, 4 x 4 outputs per unit) into a shared tile, the scatter
/// through the pair's row and column maps, one subtraction.
#[cube(launch_unchecked)]
pub fn k_big_update<F: Float>(items: &[u32], rects: &[u32], plan: &[u32], factor: &mut [F]) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let o_pr_d = plan[H_PR_D] as usize;
    let o_pr_r0 = plan[H_PR_R0] as usize;
    let o_pr_rmap = plan[H_PR_RMAP] as usize;
    let o_pr_cmap = plan[H_PR_CMAP] as usize;
    let o_rmap = plan[H_RMAP] as usize;
    let o_cmap = plan[H_CMAP] as usize;
    let item = CUBE_POS as usize;
    let s = items[5 * item] as usize;
    let row0 = items[5 * item + 1] as usize;
    let col0 = items[5 * item + 2] as usize;
    let r0 = items[5 * item + 3] as usize;
    let r1 = items[5 * item + 4] as usize;
    let q = plan[o_dims + 2 * s] as usize;
    let h = plan[o_dims + 2 * s + 1] as usize;
    let mut u = Shared::<[F]>::new_slice(TB * TB);
    let mut sa = Shared::<[F]>::new_slice(TK * TB);
    let mut sb = Shared::<[F]>::new_slice(TK * TB);
    let t = UNIT_POS as usize;
    let nt = CUBE_DIM as usize;
    let tx = t % 16;
    let ty = t / 16;
    let mut e = t;
    while e < TB * TB {
        u[e] = F::new(0.0f32);
        e += nt;
    }
    let mut acc = Array::<F>::new(16usize);
    for rr in r0..r1 {
        let pi = rects[5 * rr] as usize;
        let da = rects[5 * rr + 1] as usize;
        let db = rects[5 * rr + 2] as usize;
        let ca = rects[5 * rr + 3] as usize;
        let cb = rects[5 * rr + 4] as usize;
        let d = plan[o_pr_d + pi] as usize;
        let pr0 = plan[o_pr_r0 + pi] as usize;
        let dq = plan[o_dims + 2 * d] as usize;
        let dh = plan[o_dims + 2 * d + 1] as usize;
        let dp = plan[o_val_ptr + d] as usize + pr0;
        let nr = db - da;
        let nc = cb - ca;
        #[unroll]
        for z in 0..16 {
            acc[z] = F::new(0.0f32);
        }
        let mut k0 = 0usize;
        while k0 < dq {
            sync_cube();
            let mut e2 = t;
            while e2 < TK * TB {
                let kk = e2 / TB;
                let i = e2 - kk * TB;
                let k = k0 + kk;
                let mut va = F::new(0.0f32);
                let mut vb = F::new(0.0f32);
                if k < dq && i < nr {
                    va = factor[dp + k * dh + da + i];
                }
                if k < dq && i < nc {
                    vb = factor[dp + k * dh + ca + i];
                }
                sa[e2] = va;
                sb[e2] = vb;
                e2 += nt;
            }
            sync_cube();
            #[unroll]
            for kk in 0..TK {
                let a0 = sa[kk * TB + tx * 4];
                let a1 = sa[kk * TB + tx * 4 + 1];
                let a2 = sa[kk * TB + tx * 4 + 2];
                let a3 = sa[kk * TB + tx * 4 + 3];
                let b0 = sb[kk * TB + ty * 4];
                let b1 = sb[kk * TB + ty * 4 + 1];
                let b2 = sb[kk * TB + ty * 4 + 2];
                let b3 = sb[kk * TB + ty * 4 + 3];
                acc[0] += a0 * b0;
                acc[1] += a0 * b1;
                acc[2] += a0 * b2;
                acc[3] += a0 * b3;
                acc[4] += a1 * b0;
                acc[5] += a1 * b1;
                acc[6] += a1 * b2;
                acc[7] += a1 * b3;
                acc[8] += a2 * b0;
                acc[9] += a2 * b1;
                acc[10] += a2 * b2;
                acc[11] += a2 * b3;
                acc[12] += a3 * b0;
                acc[13] += a3 * b1;
                acc[14] += a3 * b2;
                acc[15] += a3 * b3;
            }
            k0 += TK;
        }
        let rm = plan[o_pr_rmap + pi] as usize + da;
        let cm = plan[o_pr_cmap + pi] as usize + ca;
        #[unroll]
        for r in 0..4 {
            #[unroll]
            for c in 0..4 {
                let i = tx * 4 + r;
                let j = ty * 4 + c;
                if i < nr && j < nc {
                    let ui = (plan[o_cmap + cm + j] as usize - col0) * TB + (plan[o_rmap + rm + i] as usize - row0);
                    u[ui] += acc[r * 4 + c];
                }
            }
        }
    }
    sync_cube();
    let panel = plan[o_val_ptr + s] as usize;
    let mut nrows = h - row0;
    if nrows > TB {
        nrows = TB;
    }
    let mut ncols = q - col0;
    if ncols > TB {
        ncols = TB;
    }
    let mut e3 = t;
    while e3 < TB * TB {
        let j = e3 / TB;
        let i = e3 - j * TB;
        if i < nrows && j < ncols && row0 + i >= col0 + j {
            factor[panel + (col0 + j) * h + row0 + i] -= u[e3];
        }
        e3 += nt;
    }
}

/// A small supernode's diagonal Cholesky in shared memory and the triangular
/// solve of its rows below, in place; one cube per supernode of `list`.
/// `qq` is the largest small q, squared. A pivot that is not positive sets
/// `fail[0]` to the supernode plus one and continues with 1.
#[cube(launch_unchecked)]
pub fn k_small_factor<F: Float>(list: &[u32], plan: &[u32], factor: &mut [F], fail: &mut [u32], #[comptime] qq: usize) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let s = list[CUBE_POS as usize] as usize;
    let q = plan[o_dims + 2 * s] as usize;
    let h = plan[o_dims + 2 * s + 1] as usize;
    let panel = plan[o_val_ptr + s] as usize;
    let t = UNIT_POS as usize;
    let nt = CUBE_DIM as usize;
    let mut l = Shared::<[F]>::new_slice(qq);
    let mut e = t;
    while e < q * q {
        let c = e / q;
        let r = e - c * q;
        let mut v = F::new(0.0f32);
        if r >= c {
            v = factor[panel + c * h + r];
        }
        l[e] = v;
        e += nt;
    }
    sync_cube();
    for j in 0..q {
        if t == 0usize {
            let mut d = l[j * q + j];
            if !(d > F::new(0.0f32)) {
                fail[0] = s as u32 + 1u32;
                d = F::new(1.0f32);
            }
            l[j * q + j] = F::sqrt(d);
        }
        sync_cube();
        let ljj = l[j * q + j];
        let mut i = j + 1usize + t;
        while i < q {
            l[j * q + i] = l[j * q + i] / ljj;
            i += nt;
        }
        sync_cube();
        let w = q - j - 1usize;
        let mut e2 = t;
        while e2 < w * w {
            let kk = j + 1usize + e2 / w;
            let ii = j + 1usize + e2 % w;
            if ii >= kk {
                let d = l[j * q + ii] * l[j * q + kk];
                l[kk * q + ii] -= d;
            }
            e2 += nt;
        }
        sync_cube();
    }
    let mut e3 = t;
    while e3 < q * q {
        let c = e3 / q;
        let r = e3 - c * q;
        if r >= c {
            factor[panel + c * h + r] = l[e3];
        }
        e3 += nt;
    }
    sync_cube();
    let mut r = q + t;
    while r < h {
        for c in 0..q {
            let mut v = factor[panel + c * h + r];
            for kk in 0..c {
                v -= factor[panel + kk * h + r] * l[kk * q + c];
            }
            factor[panel + c * h + r] = v / l[c * q + c];
        }
        r += nt;
    }
}

/// Panel step, part one: the NB x NB diagonal tile at (j, j) of every big
/// panel of the level, one cube per panel, row-major `l[r * NB + c]` in
/// shared memory.
#[cube(launch_unchecked)]
pub fn k_potf2<F: Float>(panels: &[u32], j: u32, plan: &[u32], factor: &mut [F], fail: &mut [u32]) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let s = panels[CUBE_POS as usize] as usize;
    let q = plan[o_dims + 2 * s] as usize;
    let h = plan[o_dims + 2 * s + 1] as usize;
    let j = j as usize;
    if j < q {
        let mut nb = q - j;
        if nb > NB {
            nb = NB;
        }
        let mut l = Shared::<[F]>::new_slice(NB * NB);
        let tile = plan[o_val_ptr + s] as usize + j * h + j;
        let t = UNIT_POS as usize;
        let nt = CUBE_DIM as usize;
        let mut e = t;
        while e < NB * NB {
            let r = e / NB;
            let c = e - r * NB;
            let mut v = F::new(0.0f32);
            if r < nb && c <= r {
                v = factor[tile + c * h + r];
            }
            l[e] = v;
            e += nt;
        }
        sync_cube();
        for k in 0..nb {
            if t == 0usize {
                let mut d = l[k * NB + k];
                if !(d > F::new(0.0f32)) {
                    fail[0] = s as u32 + 1u32;
                    d = F::new(1.0f32);
                }
                l[k * NB + k] = F::sqrt(d);
            }
            sync_cube();
            let lkk = l[k * NB + k];
            let mut i = k + 1usize + t;
            while i < nb {
                l[i * NB + k] = l[i * NB + k] / lkk;
                i += nt;
            }
            sync_cube();
            let w = nb - k - 1usize;
            let mut e2 = t;
            while e2 < w * w {
                let c = k + 1usize + e2 / w;
                let i = k + 1usize + e2 % w;
                if i >= c {
                    let d = l[i * NB + k] * l[c * NB + k];
                    l[i * NB + c] -= d;
                }
                e2 += nt;
            }
            sync_cube();
        }
        let mut e3 = t;
        while e3 < NB * NB {
            let r = e3 / NB;
            let c = e3 - r * NB;
            if r < nb && c <= r {
                factor[tile + c * h + r] = l[e3];
            }
            e3 += nt;
        }
    }
}

/// Panel step, part two: the rows below the step's diagonal tile,
/// right-solved against it (L21 = A21 L11^-T); TB rows per cube of TB units,
/// the cube index walking the level's panels through their row-chunk counts.
#[cube(launch_unchecked)]
pub fn k_trsm<F: Float>(panels: &[u32], npanels: u32, j: u32, plan: &[u32], factor: &mut [F]) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let j = j as usize;
    let mut idx = CUBE_POS as usize;
    let mut p = 0usize;
    let mut found = false;
    let mut s = 0usize;
    let mut q = 0usize;
    let mut h = 0usize;
    while p < npanels as usize && !found {
        s = panels[p] as usize;
        q = plan[o_dims + 2 * s] as usize;
        h = plan[o_dims + 2 * s + 1] as usize;
        let mut rows = 0usize;
        if j < q {
            let mut nb = q - j;
            if nb > NB {
                nb = NB;
            }
            rows = h - j - nb;
        }
        let chunks = (rows + TB - 1usize) / TB;
        if idx < chunks {
            found = true;
        } else {
            idx -= chunks;
            p += 1usize;
        }
    }
    if found {
        let mut nb = q - j;
        if nb > NB {
            nb = NB;
        }
        let panel = plan[o_val_ptr + s] as usize;
        let t = UNIT_POS as usize;
        let nt = CUBE_DIM as usize;
        let mut l = Shared::<[F]>::new_slice(NB * NB);
        let mut e = t;
        while e < NB * NB {
            let r = e / NB;
            let c = e - r * NB;
            let mut v = F::new(0.0f32);
            if r < nb && c <= r {
                v = factor[panel + (j + c) * h + j + r];
            }
            l[e] = v;
            e += nt;
        }
        sync_cube();
        let i = j + nb + idx * TB + t;
        if i < h {
            let mut v = Array::<F>::new(NB);
            for c in 0..nb {
                v[c] = factor[panel + (j + c) * h + i];
            }
            for c in 0..nb {
                let mut x = v[c];
                for k in 0..c {
                    x -= v[k] * l[c * NB + k];
                }
                v[c] = x / l[c * NB + c];
            }
            for c in 0..nb {
                factor[panel + (j + c) * h + i] = v[c];
            }
        }
    }
}

/// Panel step, part three: the rank-nb update of the trailing lower
/// trapezoid by the step's solved block column, TB x TB tiles; tiles above
/// the diagonal are skipped and the straddling tile clipped. The cube index
/// walks the panels' tile counts.
#[cube(launch_unchecked)]
pub fn k_syrk<F: Float>(panels: &[u32], npanels: u32, j: u32, plan: &[u32], factor: &mut [F]) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let j = j as usize;
    let mut idx = CUBE_POS as usize;
    let mut p = 0usize;
    let mut found = false;
    let mut s = 0usize;
    let mut q = 0usize;
    let mut h = 0usize;
    let mut nct = 0usize;
    while p < npanels as usize && !found {
        s = panels[p] as usize;
        q = plan[o_dims + 2 * s] as usize;
        h = plan[o_dims + 2 * s + 1] as usize;
        let mut nb = 0usize;
        if j < q {
            nb = q - j;
            if nb > NB {
                nb = NB;
            }
        }
        let mut rows = 0usize;
        let mut cols = 0usize;
        if j + nb < q {
            rows = h - j - nb;
            cols = q - j - nb;
        }
        nct = (cols + TB - 1usize) / TB;
        let tiles = ((rows + TB - 1usize) / TB) * nct;
        if idx < tiles {
            found = true;
        } else {
            idx -= tiles;
            p += 1usize;
        }
    }
    if found {
        let ti = idx / nct;
        let tj = idx - ti * nct;
        if ti >= tj {
            let mut nb = q - j;
            if nb > NB {
                nb = NB;
            }
            let row0 = j + nb + ti * TB;
            let col0 = j + nb + tj * TB;
            let mut nrows = h - row0;
            if nrows > TB {
                nrows = TB;
            }
            let mut ncols = q - col0;
            if ncols > TB {
                ncols = TB;
            }
            let panel = plan[o_val_ptr + s] as usize;
            let t = UNIT_POS as usize;
            let nt = CUBE_DIM as usize;
            let tx = t % 16;
            let ty = t / 16;
            let mut sa = Shared::<[F]>::new_slice(NB * TB);
            let mut sb = Shared::<[F]>::new_slice(NB * TB);
            let mut e = t;
            while e < NB * TB {
                let k = e / TB;
                let i = e - k * TB;
                let mut va = F::new(0.0f32);
                let mut vb = F::new(0.0f32);
                if k < nb && i < nrows {
                    va = factor[panel + (j + k) * h + row0 + i];
                }
                if k < nb && i < ncols {
                    vb = factor[panel + (j + k) * h + col0 + i];
                }
                sa[e] = va;
                sb[e] = vb;
                e += nt;
            }
            sync_cube();
            let mut acc = Array::<F>::new(16usize);
            #[unroll]
            for z in 0..16 {
                acc[z] = F::new(0.0f32);
            }
            for kk in 0..nb {
                let a0 = sa[kk * TB + tx * 4];
                let a1 = sa[kk * TB + tx * 4 + 1];
                let a2 = sa[kk * TB + tx * 4 + 2];
                let a3 = sa[kk * TB + tx * 4 + 3];
                let b0 = sb[kk * TB + ty * 4];
                let b1 = sb[kk * TB + ty * 4 + 1];
                let b2 = sb[kk * TB + ty * 4 + 2];
                let b3 = sb[kk * TB + ty * 4 + 3];
                acc[0] += a0 * b0;
                acc[1] += a0 * b1;
                acc[2] += a0 * b2;
                acc[3] += a0 * b3;
                acc[4] += a1 * b0;
                acc[5] += a1 * b1;
                acc[6] += a1 * b2;
                acc[7] += a1 * b3;
                acc[8] += a2 * b0;
                acc[9] += a2 * b1;
                acc[10] += a2 * b2;
                acc[11] += a2 * b3;
                acc[12] += a3 * b0;
                acc[13] += a3 * b1;
                acc[14] += a3 * b2;
                acc[15] += a3 * b3;
            }
            #[unroll]
            for r in 0..4 {
                #[unroll]
                for c in 0..4 {
                    let i = tx * 4 + r;
                    let jj = ty * 4 + c;
                    if i < nrows && jj < ncols && row0 + i >= col0 + jj {
                        factor[panel + (col0 + jj) * h + row0 + i] -= acc[r * 4 + c];
                    }
                }
            }
        }
    }
}

/// Forward solve, part one: the gather. `items` has 4 entries per item
/// (pairs [pa, pb), mid rows [e0, e1) counted from pair pa's first); every
/// mid row subtracts its descendant row's dot product with the descendant's
/// final y from its target's x, atomically. `x` and `xa` are the same buffer.
#[cube(launch_unchecked)]
pub fn k_fwd_gather<F: Float + CubePrimitive<Scalar: AtomicNumeric>>(
    items: &[u32],
    plan: &[u32],
    factor: &[F],
    x: &[F],
    xa: &mut [Atomic<F>],
    ) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let o_sup_begin = plan[H_SUP_BEGIN] as usize;
    let o_sc_start = plan[H_SC_START] as usize;
    let o_pr_d = plan[H_PR_D] as usize;
    let o_pr_r0 = plan[H_PR_R0] as usize;
    let o_pr_s = plan[H_PR_S] as usize;
    let o_pr_cmap = plan[H_PR_CMAP] as usize;
    let o_cmap = plan[H_CMAP] as usize;
    let o_pair_kw_ptr = plan[H_PAIR_KW_PTR] as usize;
    let item = CUBE_POS as usize;
    let pa = items[4 * item] as usize;
    let pb = items[4 * item + 1] as usize;
    let e0 = items[4 * item + 2] as usize;
    let e1 = items[4 * item + 3] as usize;
    let base = plan[o_pair_kw_ptr + pa] as usize;
    let t = UNIT_POS as usize;
    let nt = CUBE_DIM as usize;
    let mut e = e0 + t;
    while e < e1 {
        let mut lo = pa;
        let mut hi = pb;
        while hi - lo > 1usize {
            let mid = (lo + hi) / 2usize;
            if plan[o_pair_kw_ptr + mid] as usize - base <= e {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        let pi = lo;
        let i = e - (plan[o_pair_kw_ptr + pi] as usize - base);
        let s = plan[o_pr_s + pi] as usize;
        let d = plan[o_pr_d + pi] as usize;
        let r0 = plan[o_pr_r0 + pi] as usize;
        let dq = plan[o_dims + 2 * d] as usize;
        let dh = plan[o_dims + 2 * d + 1] as usize;
        let dcol0 = plan[o_sc_start + plan[o_sup_begin + d] as usize] as usize;
        let col0 = plan[o_sc_start + plan[o_sup_begin + s] as usize] as usize;
        let dp = plan[o_val_ptr + d] as usize + r0 + i;
        let mut a0 = F::new(0.0f32);
        let mut a1 = F::new(0.0f32);
        let mut a2 = F::new(0.0f32);
        let mut a3 = F::new(0.0f32);
        let mut k = 0usize;
        while k + 4usize <= dq {
            a0 += factor[dp + k * dh] * x[dcol0 + k];
            a1 += factor[dp + (k + 1usize) * dh] * x[dcol0 + k + 1usize];
            a2 += factor[dp + (k + 2usize) * dh] * x[dcol0 + k + 2usize];
            a3 += factor[dp + (k + 3usize) * dh] * x[dcol0 + k + 3usize];
            k += 4usize;
        }
        while k < dq {
            a0 += factor[dp + k * dh] * x[dcol0 + k];
            k += 1usize;
        }
        let tc = plan[o_cmap + plan[o_pr_cmap + pi] as usize + i] as usize;
        xa[col0 + tc].fetch_add(F::new(0.0f32) - ((a0 + a1) + (a2 + a3)));
        e += nt;
    }
}

/// A lane's share of sum_i a[a0 + i] b[b0 + i] over i = lane, lane + 32, ...:
/// `u` loads in flight per iteration into two alternating accumulators, then
/// the one-wide remainder.
#[cube]
pub fn lane_dot<F: Float>(a: &[F], a0: usize, b: &Shared<[F]>, b0: usize, n: usize, lane: usize, #[comptime] u: usize) -> F {
    let mut p0 = F::new(0.0f32);
    let mut p1 = F::new(0.0f32);
    let mut ii = lane;
    while ii + (u - 1) * 32usize < n {
        let mut va = Array::<F>::new(u);
        let mut vb = Array::<F>::new(u);
        #[unroll]
        for k in 0..u {
            va[k] = a[a0 + ii + k * 32usize];
            vb[k] = b[b0 + ii + k * 32usize];
        }
        #[unroll]
        for k in 0..u / 2 {
            p0 += va[2 * k] * vb[2 * k];
            p1 += va[2 * k + 1] * vb[2 * k + 1];
        }
        ii += u * 32usize;
    }
    while ii < n {
        p0 += a[a0 + ii] * b[b0 + ii];
        ii += 32usize;
    }
    p0 + p1
}

/// The 32 x 32 lower tile at (j0, j0) into row-major `ls` (coalesced along
/// the column), and the reciprocal of its diagonal for this unit's lane. Four
/// elements per unit of a cube of 256, unrolled.
#[cube]
pub fn load_tile32<F: Float>(factor: &[F], panel: usize, h: usize, j0: usize, nb: usize, ls: &mut Shared<[F]>) -> F {
    let t = UNIT_POS as usize;
    #[unroll]
    for i in 0..4 {
        let e = t + i * 256usize;
        let c = e / 32usize;
        let r = e - c * 32usize;
        let mut v = F::new(0.0f32);
        if r < nb && c <= r {
            v = factor[panel + (j0 + c) * h + j0 + r];
        }
        ls[r * 32usize + c] = v;
    }
    sync_cube();
    let lane = t % 32usize;
    let mut rd = F::new(0.0f32);
    if lane < nb {
        rd = F::new(1.0f32) / ls[lane * 32usize + lane];
    }
    rd
}

/// Forward solve, part two: the trsv, one cube of 256 units per supernode.
/// In 32-column blocks: the diagonal tile's 32 sequential steps by the first
/// plane on shared data, the rows below updated by all units with the full
/// block's 32 loads issued before its FMAs. `max_q`: the widest supernode.
#[cube(launch_unchecked)]
pub fn k_fwd_trsv<F: Float>(list: &[u32], plan: &[u32], factor: &[F], x: &mut [F], #[comptime] max_q: usize) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let o_sup_begin = plan[H_SUP_BEGIN] as usize;
    let o_sc_start = plan[H_SC_START] as usize;
    let s = list[CUBE_POS as usize] as usize;
    let q = plan[o_dims + 2 * s] as usize;
    let h = plan[o_dims + 2 * s + 1] as usize;
    let col0 = plan[o_sc_start + plan[o_sup_begin + s] as usize] as usize;
    let panel = plan[o_val_ptr + s] as usize;
    let t = UNIT_POS as usize;
    let nt = CUBE_DIM as usize;
    let lane = t % 32usize;
    let mut xs = Shared::<[F]>::new_slice(max_q);
    let mut ls = Shared::<[F]>::new_slice(1024usize);
    let mut c0 = t;
    while c0 < q {
        xs[c0] = x[col0 + c0];
        c0 += nt;
    }
    let mut j0 = 0usize;
    while j0 < q {
        let mut nb = q - j0;
        if nb > 32usize {
            nb = 32usize;
        }
        sync_cube();
        let rd = load_tile32::<F>(factor, panel, h, j0, nb, &mut ls);
        if t < 32usize {
            for c in 0..nb {
                if lane == c {
                    xs[j0 + c] *= rd;
                }
                sync_plane();
                let v = xs[j0 + c];
                if lane > c && lane < nb {
                    xs[j0 + lane] -= ls[lane * 32usize + c] * v;
                }
                sync_plane();
            }
        }
        sync_cube();
        let mut i = j0 + nb + t;
        while i < q {
            let mut acc = F::new(0.0f32);
            if nb == 32usize {
                let mut v = Array::<F>::new(32usize);
                #[unroll]
                for c in 0..32 {
                    v[c] = factor[panel + (j0 + c) * h + i];
                }
                #[unroll]
                for c in 0..32 {
                    acc += v[c] * xs[j0 + c];
                }
            } else {
                for c in 0..nb {
                    acc += factor[panel + (j0 + c) * h + i] * xs[j0 + c];
                }
            }
            xs[i] -= acc;
            i += nt;
        }
        j0 += 32usize;
    }
    sync_cube();
    let mut c1 = t;
    while c1 < q {
        x[col0 + c1] = xs[c1];
        c1 += nt;
    }
}

/// Backward solve, part one: x_s -= L_below^T x for the columns [c0, c1) of
/// a supernode (`items`, 3 per item); the rows' x gathered into shared memory
/// XR_CH at a time, a plane per column with the lanes over the rows. `lu`:
/// loads in flight per lane iteration.
#[cube(launch_unchecked)]
pub fn k_bwd_gemv<F: Float + CubePrimitive<Scalar: PlaneNumeric>>(items: &[u32], plan: &[u32], factor: &[F], x: &mut [F], #[comptime] lu: usize) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let o_sup_begin = plan[H_SUP_BEGIN] as usize;
    let o_sc_start = plan[H_SC_START] as usize;
    let o_row_sc = plan[H_ROW_SC] as usize;
    let o_row_sc_off = plan[H_ROW_SC_OFF] as usize;
    let item = CUBE_POS as usize;
    let s = items[3 * item] as usize;
    let c0 = items[3 * item + 1] as usize;
    let c1 = items[3 * item + 2] as usize;
    let q = plan[o_dims + 2 * s] as usize;
    let h = plan[o_dims + 2 * s + 1] as usize;
    let col0 = plan[o_sc_start + plan[o_sup_begin + s] as usize] as usize;
    let panel = plan[o_val_ptr + s] as usize;
    let rsc = plan[o_row_sc_off + s] as usize;
    let t = UNIT_POS as usize;
    let nt = CUBE_DIM as usize;
    let lane = t % 32usize;
    let warp = t / 32usize;
    let nwarps = nt / 32usize;
    let mut xr = Shared::<[F]>::new_slice(XR_CH);
    let mut i0 = 0usize;
    while i0 < h - q {
        let mut n = h - q - i0;
        if n > XR_CH {
            n = XR_CH;
        }
        sync_cube();
        let mut i = t;
        while i < n {
            xr[i] = x[plan[o_row_sc + rsc + i0 + i] as usize];
            i += nt;
        }
        sync_cube();
        let mut c = c0 + warp;
        while c < c1 {
            let pc = panel + c * h + q + i0;
            let part = plane_sum(lane_dot::<F>(factor, pc, &xr, 0usize, n, lane, lu));
            if lane == 0usize {
                x[col0 + c] -= part;
            }
            c += nwarps;
        }
        i0 += XR_CH;
    }
}

/// Backward solve, part two: the transposed trsv, one cube of 256 units per
/// supernode; 32-column blocks from the last, the block's columns against
/// the rows below it (a plane per column), then the diagonal tile's steps by
/// the first plane on shared data.
#[cube(launch_unchecked)]
pub fn k_bwd_trsv<F: Float + CubePrimitive<Scalar: PlaneNumeric>>(
    list: &[u32],
    plan: &[u32],
    factor: &[F],
    x: &mut [F],
        #[comptime] max_q: usize,
    #[comptime] lu: usize,
) {
    let o_dims = plan[H_DIMS] as usize;
    let o_val_ptr = plan[H_VAL_PTR] as usize;
    let o_sup_begin = plan[H_SUP_BEGIN] as usize;
    let o_sc_start = plan[H_SC_START] as usize;
    let s = list[CUBE_POS as usize] as usize;
    let q = plan[o_dims + 2 * s] as usize;
    let h = plan[o_dims + 2 * s + 1] as usize;
    let col0 = plan[o_sc_start + plan[o_sup_begin + s] as usize] as usize;
    let panel = plan[o_val_ptr + s] as usize;
    let t = UNIT_POS as usize;
    let nt = CUBE_DIM as usize;
    let lane = t % 32usize;
    let warp = t / 32usize;
    let nwarps = nt / 32usize;
    let mut xs = Shared::<[F]>::new_slice(max_q);
    let mut ls = Shared::<[F]>::new_slice(1024usize);
    let mut c0 = t;
    while c0 < q {
        xs[c0] = x[col0 + c0];
        c0 += nt;
    }
    sync_cube();
    let mut bi = (q + 31usize) / 32usize;
    while bi > 0usize {
        bi -= 1usize;
        let j0 = bi * 32usize;
        let mut nb = q - j0;
        if nb > 32usize {
            nb = 32usize;
        }
        let n = q - j0 - nb;
        let mut c = warp;
        while c < nb {
            let pc = panel + (j0 + c) * h + j0 + nb;
            let part = plane_sum(lane_dot::<F>(factor, pc, &xs, j0 + nb, n, lane, lu));
            if lane == 0usize {
                xs[j0 + c] -= part;
            }
            c += nwarps;
        }
        sync_cube();
        let rd = load_tile32::<F>(factor, panel, h, j0, nb, &mut ls);
        if t < 32usize {
            let mut c2 = nb;
            while c2 > 0usize {
                c2 -= 1usize;
                if lane == c2 {
                    xs[j0 + c2] *= rd;
                }
                sync_plane();
                let v = xs[j0 + c2];
                if lane < c2 {
                    xs[j0 + lane] -= ls[c2 * 32usize + lane] * v;
                }
                sync_plane();
            }
        }
        sync_cube();
    }
    let mut c1 = t;
    while c1 < q {
        x[col0 + c1] = xs[c1];
        c1 += nt;
    }
}
