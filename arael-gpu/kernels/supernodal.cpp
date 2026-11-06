// supernodal.cpp: the block supernodal Cholesky and its level-set solve
// (docs/dev/GPU.md, E3 and E5 step 7), launched by
// arael-gpu/src/supernodal.rs over the plan of arael-gpu/src/plan.rs.
//
// Compiled per SCALAR (float or double), the wide scalar W of the vectors
// it solves (wide.h, in front of this) and LU; the tile sizes (NB, TB, TK,
// TR, TC, GROUP, XR_CH) and the plan header's slots (H_*) come from plan.rs
// as #defines, THREADS (the block size of every kernel but k_trsm, which
// runs TB) from supernodal.rs. One launch per level unless said.
//
// The solve's kernels take a `gate`: they do nothing unless gate[0] is
// nonzero, so a refinement can stop on the device and the steps launched
// after it cost only their launches.

#if ARAEL_WARP != 32
#error "supernodal.cpp: the trsv and GEMV kernels are written for 32-lane warps"
#endif
#if THREADS % 32 != 0 || 1024 % THREADS != 0
#error "supernodal.cpp: THREADS is a multiple of 32 dividing 1024"
#endif

typedef SCALAR F;

KERNEL_THREADS(THREADS) void k_zero(GLOBAL F* RESTRICT buf, u32 n) {
    u32 i = global_x();
    if (i < n) buf[i] = F(0);
}

KERNEL_THREADS(THREADS) void k_zero_u32(GLOBAL u32* RESTRICT buf, u32 n) {
    u32 i = global_x();
    if (i < n) buf[i] = 0;
}

// Stored tile `t` scattered into its panel, its values cast to F: `tiles` has
// 6 entries per tile (destination, stride, transposed, value offset, rows,
// columns). An upper tile lands transposed, a diagonal one as is.
template <typename M>
DEVICE void seed_tile(GLOBAL const u32* tiles, GLOBAL const M* vals, GLOBAL F* factor, u32 t) {
    u32 dst = tiles[6 * t];
    u32 stride = tiles[6 * t + 1];
    u32 trans = tiles[6 * t + 2];
    u32 off = tiles[6 * t + 3];
    u32 wr = tiles[6 * t + 4];
    u32 wc = tiles[6 * t + 5];
    if (trans == 0) {
        for (u32 c = 0; c < wc; c++)
            for (u32 r = 0; r < wr; r++) factor[dst + c * stride + r] = (F)vals[off + c * wr + r];
    } else {
        for (u32 c = 0; c < wr; c++)
            for (u32 r = 0; r < wc; r++) factor[dst + c * stride + r] = (F)vals[off + c + r * wr];
    }
}

// The seed from f32 values: one thread per stored tile.
KERNEL_THREADS(THREADS) void k_seed_f32(GLOBAL const u32* RESTRICT tiles, GLOBAL const float* RESTRICT vals, GLOBAL F* RESTRICT factor, u32 ntiles) {
    u32 t = global_x();
    if (t < ntiles) seed_tile<float>(tiles, vals, factor, t);
}

// The seed from f64 values.
KERNEL_THREADS(THREADS) void k_seed_f64(GLOBAL const u32* RESTRICT tiles, GLOBAL const double* RESTRICT vals, GLOBAL F* RESTRICT factor, u32 ntiles) {
    u32 t = global_x();
    if (t < ntiles) seed_tile<double>(tiles, vals, factor, t);
}

// The seed from values of the route's wide scalar (wide.h): S formed on the device.
KERNEL_THREADS(THREADS) void k_seed_wide(GLOBAL const u32* RESTRICT tiles, GLOBAL const W* RESTRICT vals, GLOBAL F* RESTRICT factor, u32 ntiles) {
    u32 t = global_x();
    if (t < ntiles) seed_tile<W>(tiles, vals, factor, t);
}

// The small descendants' updates: one block per (target, TR x TC tile,
// chunk), a group of GROUP threads per rectangle with the threads over the
// rectangle's elements, accumulated in shared memory, subtracted into the
// panel atomically (a tile may be several chunks). `src` and `dst` are the
// factor: the descendants are read through `src`, the target written
// through `dst`.
KERNEL_THREADS(THREADS) void k_tile_update(GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT rects, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT src, GLOBAL F* RESTRICT dst) {
    SHARED F u[TR * TC];
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 o_pr_d = plan[H_PR_D];
    u32 o_pr_r0 = plan[H_PR_R0];
    u32 item = block_x();
    u32 s = items[5 * item];
    u32 row0 = items[5 * item + 1];
    u32 col0 = items[5 * item + 2];
    u32 r0 = items[5 * item + 3];
    u32 r1 = items[5 * item + 4];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    u32 t = thread_x();
    const u32 nt = THREADS;
    for (u32 e = t; e < TR * TC; e += nt) u[e] = F(0);
    BARRIER();
    u32 g = t / GROUP;
    u32 lane = t % GROUP;
    u32 ngroups = nt / GROUP;
    for (u32 rr = r0 + g; rr < r1; rr += ngroups) {
        u32 rb = 4 * rr;
        u32 pi = rects[rb];
        u32 geo = rects[rb + 1];
        u32 dra = rects[rb + 2];
        u32 dca = rects[rb + 3];
        u32 i0 = geo & 0xffu;
        u32 ni = (geo >> 8) & 0xffu;
        u32 j0 = (geo >> 16) & 0xffu;
        u32 nj = geo >> 24;
        u32 d = plan[o_pr_d + pi];
        u32 pr0 = plan[o_pr_r0 + pi];
        u32 dq = plan[o_dims + 2 * d];
        u32 dh = plan[o_dims + 2 * d + 1];
        u32 dp = plan[o_val_ptr + d] + pr0;
        for (u32 e2 = lane; e2 < ni * nj; e2 += GROUP) {
            u32 jj = e2 / ni;
            u32 ii = e2 - jj * ni;
            F acc = F(0);
            for (u32 k = 0; k < dq; k++) acc += src[dp + k * dh + dra + ii] * src[dp + k * dh + dca + jj];
            atomic_add(&u[(j0 + jj) * TR + i0 + ii], acc);
        }
    }
    BARRIER();
    u32 panel = plan[o_val_ptr + s];
    u32 nrows = min(h - row0, TR);
    u32 ncols = min(q - col0, TC);
    for (u32 e3 = t; e3 < TR * TC; e3 += nt) {
        u32 j = e3 / TR;
        u32 i = e3 - j * TR;
        if (i < nrows && j < ncols && row0 + i >= col0 + j) {
            F v = u[e3];
            atomic_add(&dst[panel + (col0 + j) * h + row0 + i], F(0) - v);
        }
    }
}

// The big descendants' updates: one block per item, a (target, TB x TB
// tile) and some of the rectangles that land in it (all, unless the plan
// cut the tile's work to fill the device), looping over them with a
// shared-memory GEMM core (TK-wide k slabs, 4 x 4 outputs per thread) into a
// shared tile, the scatter through the pair's row and column maps, one
// subtraction: atomic when the tile is shared with other items. An item of
// one rectangle may cover a range [k0, k1) of its k (k1 = 0: all of it).
KERNEL_THREADS(THREADS) void k_big_update(GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT rects, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor) {
    SHARED F u[TB * TB];
    SHARED F sa[TK * TB];
    SHARED F sb[TK * TB];
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 o_pr_d = plan[H_PR_D];
    u32 o_pr_r0 = plan[H_PR_R0];
    u32 o_pr_rmap = plan[H_PR_RMAP];
    u32 o_pr_cmap = plan[H_PR_CMAP];
    u32 o_rmap = plan[H_RMAP];
    u32 o_cmap = plan[H_CMAP];
    u32 item = block_x();
    u32 s = items[BIG_ITEM * item];
    u32 row0 = items[BIG_ITEM * item + 1];
    u32 col0 = items[BIG_ITEM * item + 2];
    u32 r0 = items[BIG_ITEM * item + 3];
    u32 r1 = items[BIG_ITEM * item + 4];
    u32 kb = items[BIG_ITEM * item + 5];
    u32 ke = items[BIG_ITEM * item + 6];
    u32 shared_tile = items[BIG_ITEM * item + 7];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    u32 t = thread_x();
    const u32 nt = THREADS;
    u32 tx = t % 16;
    u32 ty = t / 16;
    for (u32 e = t; e < TB * TB; e += nt) u[e] = F(0);
    F acc[16];
    for (u32 rr = r0; rr < r1; rr++) {
        u32 pi = rects[5 * rr];
        u32 da = rects[5 * rr + 1];
        u32 db = rects[5 * rr + 2];
        u32 ca = rects[5 * rr + 3];
        u32 cb = rects[5 * rr + 4];
        u32 d = plan[o_pr_d + pi];
        u32 pr0 = plan[o_pr_r0 + pi];
        u32 dq = plan[o_dims + 2 * d];
        u32 dh = plan[o_dims + 2 * d + 1];
        u32 dp = plan[o_val_ptr + d] + pr0;
        u32 nr = db - da;
        u32 nc = cb - ca;
        u32 k_end = ke == 0 ? dq : ke;
#pragma unroll
        for (u32 z = 0; z < 16; z++) acc[z] = F(0);
        for (u32 k0 = kb; k0 < k_end; k0 += TK) {
            BARRIER();
            for (u32 e2 = t; e2 < TK * TB; e2 += nt) {
                u32 kk = e2 / TB;
                u32 i = e2 - kk * TB;
                u32 k = k0 + kk;
                F va = F(0);
                F vb = F(0);
                if (k < k_end && i < nr) va = factor[dp + k * dh + da + i];
                if (k < k_end && i < nc) vb = factor[dp + k * dh + ca + i];
                sa[e2] = va;
                sb[e2] = vb;
            }
            BARRIER();
#pragma unroll
            for (u32 kk = 0; kk < TK; kk++) {
                F a0 = sa[kk * TB + tx * 4];
                F a1 = sa[kk * TB + tx * 4 + 1];
                F a2 = sa[kk * TB + tx * 4 + 2];
                F a3 = sa[kk * TB + tx * 4 + 3];
                F b0 = sb[kk * TB + ty * 4];
                F b1 = sb[kk * TB + ty * 4 + 1];
                F b2 = sb[kk * TB + ty * 4 + 2];
                F b3 = sb[kk * TB + ty * 4 + 3];
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
        }
        u32 rm = plan[o_pr_rmap + pi] + da;
        u32 cm = plan[o_pr_cmap + pi] + ca;
#pragma unroll
        for (u32 r = 0; r < 4; r++) {
#pragma unroll
            for (u32 c = 0; c < 4; c++) {
                u32 i = tx * 4 + r;
                u32 j = ty * 4 + c;
                if (i < nr && j < nc) {
                    u32 ui = (plan[o_cmap + cm + j] - col0) * TB + (plan[o_rmap + rm + i] - row0);
                    u[ui] += acc[r * 4 + c];
                }
            }
        }
    }
    BARRIER();
    u32 panel = plan[o_val_ptr + s];
    u32 nrows = min(h - row0, TB);
    u32 ncols = min(q - col0, TB);
    for (u32 e3 = t; e3 < TB * TB; e3 += nt) {
        u32 j = e3 / TB;
        u32 i = e3 - j * TB;
        if (i < nrows && j < ncols && row0 + i >= col0 + j) {
            GLOBAL F* dst = &factor[panel + (col0 + j) * h + row0 + i];
            if (shared_tile != 0u) atomic_add(dst, -u[e3]);
            else *dst -= u[e3];
        }
    }
}

// A small supernode's diagonal Cholesky in shared memory (q * q values of
// dynamic shared memory) and the triangular solve of its rows below, in
// place; one block per supernode of `list`. A pivot that is not positive
// sets fail[0] to the supernode plus one and continues with 1.
KERNEL_THREADS(THREADS) void k_small_factor(GLOBAL const u32* RESTRICT list, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor, GLOBAL u32* RESTRICT fail) {
    DYNAMIC_SHARED(F, l);
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 s = list[block_x()];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    u32 panel = plan[o_val_ptr + s];
    u32 t = thread_x();
    const u32 nt = THREADS;
    for (u32 e = t; e < q * q; e += nt) {
        u32 c = e / q;
        u32 r = e - c * q;
        F v = F(0);
        if (r >= c) v = factor[panel + c * h + r];
        l[e] = v;
    }
    BARRIER();
    for (u32 j = 0; j < q; j++) {
        if (t == 0) {
            F d = l[j * q + j];
            if (!(d > F(0))) {
                fail[0] = s + 1;
                d = F(1);
            }
            l[j * q + j] = sqrt(d);
        }
        BARRIER();
        F ljj = l[j * q + j];
        for (u32 i = j + 1 + t; i < q; i += nt) l[j * q + i] = l[j * q + i] / ljj;
        BARRIER();
        u32 w = q - j - 1;
        for (u32 e2 = t; e2 < w * w; e2 += nt) {
            u32 kk = j + 1 + e2 / w;
            u32 ii = j + 1 + e2 % w;
            if (ii >= kk) {
                F d = l[j * q + ii] * l[j * q + kk];
                l[kk * q + ii] -= d;
            }
        }
        BARRIER();
    }
    for (u32 e3 = t; e3 < q * q; e3 += nt) {
        u32 c = e3 / q;
        u32 r = e3 - c * q;
        if (r >= c) factor[panel + c * h + r] = l[e3];
    }
    BARRIER();
    for (u32 r = q + t; r < h; r += nt) {
        for (u32 c = 0; c < q; c++) {
            F v = factor[panel + c * h + r];
            for (u32 kk = 0; kk < c; kk++) v -= factor[panel + kk * h + r] * l[kk * q + c];
            factor[panel + c * h + r] = v / l[c * q + c];
        }
    }
}

// Panel step, part one: the NB x NB diagonal tile at (j, j) of every big
// panel of the level, one block per panel, row-major l[r * NB + c] in
// shared memory.
KERNEL_THREADS(THREADS) void k_potf2(GLOBAL const u32* RESTRICT panels, u32 j, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor, GLOBAL u32* RESTRICT fail) {
    SHARED F l[NB * NB];
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 s = panels[block_x()];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    if (j >= q) return;
    u32 nb = min(q - j, NB);
    u32 tile = plan[o_val_ptr + s] + j * h + j;
    u32 t = thread_x();
    const u32 nt = THREADS;
    for (u32 e = t; e < NB * NB; e += nt) {
        u32 r = e / NB;
        u32 c = e - r * NB;
        F v = F(0);
        if (r < nb && c <= r) v = factor[tile + c * h + r];
        l[e] = v;
    }
    BARRIER();
    for (u32 k = 0; k < nb; k++) {
        if (t == 0) {
            F d = l[k * NB + k];
            if (!(d > F(0))) {
                fail[0] = s + 1;
                d = F(1);
            }
            l[k * NB + k] = sqrt(d);
        }
        BARRIER();
        F lkk = l[k * NB + k];
        for (u32 i = k + 1 + t; i < nb; i += nt) l[i * NB + k] = l[i * NB + k] / lkk;
        BARRIER();
        u32 w = nb - k - 1;
        for (u32 e2 = t; e2 < w * w; e2 += nt) {
            u32 c = k + 1 + e2 / w;
            u32 i = k + 1 + e2 % w;
            if (i >= c) {
                F d = l[i * NB + k] * l[c * NB + k];
                l[i * NB + c] -= d;
            }
        }
        BARRIER();
    }
    for (u32 e3 = t; e3 < NB * NB; e3 += nt) {
        u32 r = e3 / NB;
        u32 c = e3 - r * NB;
        if (r < nb && c <= r) factor[tile + c * h + r] = l[e3];
    }
}

// Panel step, part two: the rows below the step's diagonal tile,
// right-solved against it (L21 = A21 L11^-T); TB rows per block of TB
// threads, the block index walking the level's panels through their
// row-chunk counts.
KERNEL_THREADS(TB) void k_trsm(GLOBAL const u32* RESTRICT panels, u32 npanels, u32 j, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor) {
    SHARED F l[NB * NB];
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 idx = block_x();
    u32 p = 0;
    bool found = false;
    u32 s = 0;
    u32 q = 0;
    u32 h = 0;
    while (p < npanels && !found) {
        s = panels[p];
        q = plan[o_dims + 2 * s];
        h = plan[o_dims + 2 * s + 1];
        u32 rows = 0;
        if (j < q) rows = h - j - min(q - j, NB);
        u32 chunks = (rows + TB - 1) / TB;
        if (idx < chunks) {
            found = true;
        } else {
            idx -= chunks;
            p += 1;
        }
    }
    if (!found) return;
    u32 nb = min(q - j, NB);
    u32 panel = plan[o_val_ptr + s];
    u32 t = thread_x();
    const u32 nt = TB;
    for (u32 e = t; e < NB * NB; e += nt) {
        u32 r = e / NB;
        u32 c = e - r * NB;
        F v = F(0);
        if (r < nb && c <= r) v = factor[panel + (j + c) * h + j + r];
        l[e] = v;
    }
    BARRIER();
    u32 i = j + nb + idx * TB + t;
    if (i < h) {
        // The loops run to the constant NB and skip the columns past nb, so
        // they unroll fully, every index into v is a constant and the row
        // stays in registers. With nb as the bound, v went to local memory
        // and each step of the solve waited on a load.
        F v[NB];
#pragma unroll
        for (u32 c = 0; c < NB; c++) v[c] = c < nb ? factor[panel + (j + c) * h + i] : F(0);
#pragma unroll
        for (u32 c = 0; c < NB; c++) {
            if (c < nb) {
                F x = v[c];
#pragma unroll
                for (u32 k = 0; k < c; k++) x -= v[k] * l[c * NB + k];
                v[c] = x / l[c * NB + c];
            }
        }
#pragma unroll
        for (u32 c = 0; c < NB; c++) {
            if (c < nb) factor[panel + (j + c) * h + i] = v[c];
        }
    }
}

// Panel step, part three: the rank-nb update of the trailing lower
// trapezoid by the step's solved block column, TB x TB tiles; tiles above
// the diagonal are skipped and the straddling tile clipped. The block index
// walks the panels' tile counts.
KERNEL_THREADS(THREADS) void k_syrk(GLOBAL const u32* RESTRICT panels, u32 npanels, u32 j, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor) {
    SHARED F sa[NB * TB];
    SHARED F sb[NB * TB];
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 idx = block_x();
    u32 p = 0;
    bool found = false;
    u32 s = 0;
    u32 q = 0;
    u32 h = 0;
    u32 nct = 0;
    while (p < npanels && !found) {
        s = panels[p];
        q = plan[o_dims + 2 * s];
        h = plan[o_dims + 2 * s + 1];
        u32 nb = 0;
        if (j < q) nb = min(q - j, NB);
        u32 rows = 0;
        u32 cols = 0;
        if (j + nb < q) {
            rows = h - j - nb;
            cols = q - j - nb;
        }
        nct = (cols + TB - 1) / TB;
        u32 tiles = ((rows + TB - 1) / TB) * nct;
        if (idx < tiles) {
            found = true;
        } else {
            idx -= tiles;
            p += 1;
        }
    }
    if (!found) return;
    u32 ti = idx / nct;
    u32 tj = idx - ti * nct;
    if (ti < tj) return;
    u32 nb = min(q - j, NB);
    u32 row0 = j + nb + ti * TB;
    u32 col0 = j + nb + tj * TB;
    u32 nrows = min(h - row0, TB);
    u32 ncols = min(q - col0, TB);
    u32 panel = plan[o_val_ptr + s];
    u32 t = thread_x();
    const u32 nt = THREADS;
    u32 tx = t % 16;
    u32 ty = t / 16;
    for (u32 e = t; e < NB * TB; e += nt) {
        u32 k = e / TB;
        u32 i = e - k * TB;
        F va = F(0);
        F vb = F(0);
        if (k < nb && i < nrows) va = factor[panel + (j + k) * h + row0 + i];
        if (k < nb && i < ncols) vb = factor[panel + (j + k) * h + col0 + i];
        sa[e] = va;
        sb[e] = vb;
    }
    BARRIER();
    F acc[16];
#pragma unroll
    for (u32 z = 0; z < 16; z++) acc[z] = F(0);
    for (u32 kk = 0; kk < nb; kk++) {
        F a0 = sa[kk * TB + tx * 4];
        F a1 = sa[kk * TB + tx * 4 + 1];
        F a2 = sa[kk * TB + tx * 4 + 2];
        F a3 = sa[kk * TB + tx * 4 + 3];
        F b0 = sb[kk * TB + ty * 4];
        F b1 = sb[kk * TB + ty * 4 + 1];
        F b2 = sb[kk * TB + ty * 4 + 2];
        F b3 = sb[kk * TB + ty * 4 + 3];
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
#pragma unroll
    for (u32 r = 0; r < 4; r++) {
#pragma unroll
        for (u32 c = 0; c < 4; c++) {
            u32 i = tx * 4 + r;
            u32 jj = ty * 4 + c;
            if (i < nrows && jj < ncols && row0 + i >= col0 + jj) factor[panel + (col0 + jj) * h + row0 + i] -= acc[r * 4 + c];
        }
    }
}

// Forward solve, part one: the gather. `items` has 4 entries per item
// (pairs [pa, pb), mid rows [e0, e1) counted from pair pa's first); every
// mid row subtracts its descendant row's dot product with the descendant's
// final y from its target's x, atomically. `x` and `xa` are the same vector:
// the descendants' y read through `x`, the targets written through `xa`.
KERNEL_THREADS(THREADS) void k_fwd_gather(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL const F* RESTRICT x, GLOBAL F* RESTRICT xa) {
    if (gate[0] == 0) return;
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 o_sup_begin = plan[H_SUP_BEGIN];
    u32 o_sc_start = plan[H_SC_START];
    u32 o_pr_d = plan[H_PR_D];
    u32 o_pr_r0 = plan[H_PR_R0];
    u32 o_pr_s = plan[H_PR_S];
    u32 o_pr_cmap = plan[H_PR_CMAP];
    u32 o_cmap = plan[H_CMAP];
    u32 o_pair_kw_ptr = plan[H_PAIR_KW_PTR];
    u32 item = block_x();
    u32 pa = items[4 * item];
    u32 pb = items[4 * item + 1];
    u32 e0 = items[4 * item + 2];
    u32 e1 = items[4 * item + 3];
    u32 base = plan[o_pair_kw_ptr + pa];
    u32 t = thread_x();
    const u32 nt = THREADS;
    for (u32 e = e0 + t; e < e1; e += nt) {
        u32 lo = pa;
        u32 hi = pb;
        while (hi - lo > 1) {
            u32 mid = (lo + hi) / 2;
            if (plan[o_pair_kw_ptr + mid] - base <= e) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        u32 pi = lo;
        u32 i = e - (plan[o_pair_kw_ptr + pi] - base);
        u32 s = plan[o_pr_s + pi];
        u32 d = plan[o_pr_d + pi];
        u32 r0 = plan[o_pr_r0 + pi];
        u32 dq = plan[o_dims + 2 * d];
        u32 dh = plan[o_dims + 2 * d + 1];
        u32 dcol0 = plan[o_sc_start + plan[o_sup_begin + d]];
        u32 col0 = plan[o_sc_start + plan[o_sup_begin + s]];
        u32 dp = plan[o_val_ptr + d] + r0 + i;
        F a0 = F(0);
        F a1 = F(0);
        F a2 = F(0);
        F a3 = F(0);
        u32 k = 0;
        while (k + 4 <= dq) {
            a0 += factor[dp + k * dh] * x[dcol0 + k];
            a1 += factor[dp + (k + 1) * dh] * x[dcol0 + k + 1];
            a2 += factor[dp + (k + 2) * dh] * x[dcol0 + k + 2];
            a3 += factor[dp + (k + 3) * dh] * x[dcol0 + k + 3];
            k += 4;
        }
        while (k < dq) {
            a0 += factor[dp + k * dh] * x[dcol0 + k];
            k += 1;
        }
        u32 tc = plan[o_cmap + plan[o_pr_cmap + pi] + i];
        atomic_add(&xa[col0 + tc], F(0) - ((a0 + a1) + (a2 + a3)));
    }
}

// A lane's share of sum_i a[a0 + i] b[b0 + i] over i = lane, lane + 32, ...:
// LU loads in flight per iteration into two alternating accumulators, then
// the one-wide remainder.
DEVICE F lane_dot(GLOBAL const F* a, u32 a0, SHARED_PTR const F* b, u32 b0, u32 n, u32 lane) {
    F p0 = F(0);
    F p1 = F(0);
    u32 ii = lane;
    while (ii + (LU - 1) * 32 < n) {
        F va[LU];
        F vb[LU];
#pragma unroll
        for (u32 k = 0; k < LU; k++) {
            va[k] = a[a0 + ii + k * 32];
            vb[k] = b[b0 + ii + k * 32];
        }
#pragma unroll
        for (u32 k = 0; k < LU / 2; k++) {
            p0 += va[2 * k] * vb[2 * k];
            p1 += va[2 * k + 1] * vb[2 * k + 1];
        }
        ii += LU * 32;
    }
    while (ii < n) {
        p0 += a[a0 + ii] * b[b0 + ii];
        ii += 32;
    }
    return p0 + p1;
}

// The 32 x 32 lower tile at (j0, j0) into row-major `ls` (coalesced along
// the column), and the reciprocal of its diagonal for this thread's lane.
// 1024 / THREADS elements per thread.
DEVICE F load_tile32(GLOBAL const F* factor, u32 panel, u32 h, u32 j0, u32 nb, SHARED_PTR F* ls) {
    u32 t = thread_x();
#pragma unroll
    for (u32 i = 0; i < 1024 / THREADS; i++) {
        u32 e = t + i * THREADS;
        u32 c = e / 32;
        u32 r = e - c * 32;
        F v = F(0);
        if (r < nb && c <= r) v = factor[panel + (j0 + c) * h + j0 + r];
        ls[r * 32 + c] = v;
    }
    BARRIER();
    u32 lane = t % 32;
    F rd = F(0);
    if (lane < nb) rd = F(1) / ls[lane * 32 + lane];
    return rd;
}

// Forward solve, part two: the trsv, one block of 256 threads per supernode.
// In 32-column blocks: the diagonal tile's 32 sequential steps by the first
// warp on shared data, the rows below updated by all threads with the full
// block's 32 loads issued before its FMAs. Dynamic shared memory: 1024 + q
// values for the widest supernode q of the launch.
KERNEL_THREADS(THREADS) void k_fwd_trsv(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT list, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    DYNAMIC_SHARED(F, smem);
    SHARED_PTR F* ls = smem;
    SHARED_PTR F* xs = smem + 1024;
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 o_sup_begin = plan[H_SUP_BEGIN];
    u32 o_sc_start = plan[H_SC_START];
    u32 s = list[block_x()];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    u32 col0 = plan[o_sc_start + plan[o_sup_begin + s]];
    u32 panel = plan[o_val_ptr + s];
    u32 t = thread_x();
    const u32 nt = THREADS;
    u32 lane = t % 32;
    for (u32 c0 = t; c0 < q; c0 += nt) xs[c0] = x[col0 + c0];
    for (u32 j0 = 0; j0 < q; j0 += 32) {
        u32 nb = min(q - j0, 32u);
        BARRIER();
        F rd = load_tile32(factor, panel, h, j0, nb, ls);
        if (t < 32) {
            for (u32 c = 0; c < nb; c++) {
                if (lane == c) xs[j0 + c] *= rd;
                WARP_BARRIER();
                F v = xs[j0 + c];
                if (lane > c && lane < nb) xs[j0 + lane] -= ls[lane * 32 + c] * v;
                WARP_BARRIER();
            }
        }
        BARRIER();
        for (u32 i = j0 + nb + t; i < q; i += nt) {
            F acc = F(0);
            if (nb == 32) {
                F v[32];
#pragma unroll
                for (u32 c = 0; c < 32; c++) v[c] = factor[panel + (j0 + c) * h + i];
#pragma unroll
                for (u32 c = 0; c < 32; c++) acc += v[c] * xs[j0 + c];
            } else {
                for (u32 c = 0; c < nb; c++) acc += factor[panel + (j0 + c) * h + i] * xs[j0 + c];
            }
            xs[i] -= acc;
        }
    }
    BARRIER();
    for (u32 c1 = t; c1 < q; c1 += nt) x[col0 + c1] = xs[c1];
}

// Backward solve, part one: x_s -= L_below^T x for the columns [c0, c1) of
// a supernode (`items`, 3 per item); the rows' x gathered into shared memory
// XR_CH at a time, a warp per column with the lanes over the rows.
KERNEL_THREADS(THREADS) void k_bwd_gemv(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    SHARED F xr[XR_CH];
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 o_sup_begin = plan[H_SUP_BEGIN];
    u32 o_sc_start = plan[H_SC_START];
    u32 o_row_sc = plan[H_ROW_SC];
    u32 o_row_sc_off = plan[H_ROW_SC_OFF];
    u32 item = block_x();
    u32 s = items[3 * item];
    u32 c0 = items[3 * item + 1];
    u32 c1 = items[3 * item + 2];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    u32 col0 = plan[o_sc_start + plan[o_sup_begin + s]];
    u32 panel = plan[o_val_ptr + s];
    u32 rsc = plan[o_row_sc_off + s];
    u32 t = thread_x();
    const u32 nt = THREADS;
    u32 lane = t % 32;
    u32 warp = t / 32;
    u32 nwarps = nt / 32;
    for (u32 i0 = 0; i0 < h - q; i0 += XR_CH) {
        u32 n = min(h - q - i0, XR_CH);
        BARRIER();
        for (u32 i = t; i < n; i += nt) xr[i] = x[plan[o_row_sc + rsc + i0 + i]];
        BARRIER();
        for (u32 c = c0 + warp; c < c1; c += nwarps) {
            u32 pc = panel + c * h + q + i0;
            F part = warp_sum(lane_dot(factor, pc, xr, 0, n, lane));
            if (lane == 0) x[col0 + c] -= part;
        }
    }
}

// Backward solve, part two: the transposed trsv, one block of 256 threads
// per supernode; 32-column blocks from the last, the block's columns against
// the rows below it (a warp per column), then the diagonal tile's steps by
// the first warp on shared data. Dynamic shared memory as k_fwd_trsv's.
KERNEL_THREADS(THREADS) void k_bwd_trsv(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT list, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    DYNAMIC_SHARED(F, smem);
    SHARED_PTR F* ls = smem;
    SHARED_PTR F* xs = smem + 1024;
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 o_sup_begin = plan[H_SUP_BEGIN];
    u32 o_sc_start = plan[H_SC_START];
    u32 s = list[block_x()];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    u32 col0 = plan[o_sc_start + plan[o_sup_begin + s]];
    u32 panel = plan[o_val_ptr + s];
    u32 t = thread_x();
    const u32 nt = THREADS;
    u32 lane = t % 32;
    u32 warp = t / 32;
    u32 nwarps = nt / 32;
    for (u32 c0 = t; c0 < q; c0 += nt) xs[c0] = x[col0 + c0];
    BARRIER();
    u32 bi = (q + 31) / 32;
    while (bi > 0) {
        bi -= 1;
        u32 j0 = bi * 32;
        u32 nb = min(q - j0, 32u);
        u32 n = q - j0 - nb;
        for (u32 c = warp; c < nb; c += nwarps) {
            u32 pc = panel + (j0 + c) * h + j0 + nb;
            F part = warp_sum(lane_dot(factor, pc, xs, j0 + nb, n, lane));
            if (lane == 0) xs[j0 + c] -= part;
        }
        BARRIER();
        F rd = load_tile32(factor, panel, h, j0, nb, ls);
        if (t < 32) {
            u32 c2 = nb;
            while (c2 > 0) {
                c2 -= 1;
                if (lane == c2) xs[j0 + c2] *= rd;
                WARP_BARRIER();
                F v = xs[j0 + c2];
                if (lane < c2) xs[j0 + lane] -= ls[c2 * 32 + lane] * v;
                WARP_BARRIER();
            }
        }
        BARRIER();
    }
    for (u32 c1 = t; c1 < q; c1 += nt) x[col0 + c1] = xs[c1];
}

// The right-hand side into the factor's ordering and scalar: one thread per
// permuted block.
KERNEL_THREADS(THREADS) void k_perm_in(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT plan, GLOBAL const W* RESTRICT x, GLOBAL F* RESTRICT xb, u32 nb) {
    if (gate[0] == 0) return;
    u32 k = global_x();
    if (k < nb) {
        u32 old = plan[plan[H_ORDER] + k];
        u32 w = plan[plan[H_BLK_WIDTH] + k];
        u32 dst = plan[plan[H_SC_START] + k];
        u32 src = plan[plan[H_PART] + old];
        for (u32 i = 0; i < w; i++) xb[dst + i] = (F)x[src + i];
    }
}

// The solution back into the matrix's ordering, in the wide scalar.
KERNEL_THREADS(THREADS) void k_perm_out(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT xb, GLOBAL W* RESTRICT x, u32 nb) {
    if (gate[0] == 0) return;
    u32 k = global_x();
    if (k < nb) {
        u32 old = plan[plan[H_ORDER] + k];
        u32 w = plan[plan[H_BLK_WIDTH] + k];
        u32 src = plan[plan[H_SC_START] + k];
        u32 dst = plan[plan[H_PART] + old];
        for (u32 i = 0; i < w; i++) x[dst + i] = (W)xb[src + i];
    }
}
