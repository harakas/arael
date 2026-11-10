// supernodal.cpp: the block supernodal Cholesky and its level-set solve
// (docs/dev/GPU.md, E3 and E5 step 7), launched by
// arael-gpu/src/supernodal.rs over the plan of arael-gpu/src/plan.rs.
//
// Compiled per SCALAR (float or double), the wide scalar W of the vectors
// it solves (wide.h, in front of this) and LU; the tile sizes (NB, TB, TK,
// PAIR_*, XR_CH) and the plan header's slots (H_*) come from plan.rs
// as #defines, THREADS (the block size of every kernel but k_trsm, which
// runs TRSM_ROWS, and k_tile_inv), SCALAR_BYTES and the device's sizes
// (SYRK_BLOCK, SYRK_TB, TRSM_ROWS, POTF2_UNROLL) from supernodal.rs. One
// launch per level unless said.
//
// The solve's kernels take a `gate`: they do nothing unless gate[0] is
// nonzero, so a refinement can stop on the device and the steps launched
// after it cost only their launches.

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

// Stored tile `t` scattered into its panel by the SEED_LANES threads of a
// block, its values cast to F: `tiles` has 6 entries per tile (destination,
// stride, transposed, value offset, rows, columns). An upper tile lands
// transposed, a diagonal one as is. Lane `lane` takes the elements lane,
// lane + SEED_LANES, ... in the destination's order, so neighbouring lanes
// write neighbouring rows; the tile's entries are the same in every lane.
template <typename M>
DEVICE void seed_tile_lanes(GLOBAL const u32* tiles, GLOBAL const M* vals, GLOBAL F* factor, u32 t, u32 lane) {
    u32 dst = tiles[6 * t];
    u32 stride = tiles[6 * t + 1];
    u32 trans = tiles[6 * t + 2];
    u32 off = tiles[6 * t + 3];
    u32 wr = tiles[6 * t + 4];
    u32 wc = tiles[6 * t + 5];
    u32 n = wr * wc;
    if (trans == 0) {
        for (u32 e = lane; e < n; e += SEED_LANES) {
            u32 c = e / wr;
            u32 r = e - c * wr;
            factor[dst + c * stride + r] = (F)vals[off + e];
        }
    } else {
        for (u32 e = lane; e < n; e += SEED_LANES) {
            u32 c = e / wc;
            u32 r = e - c * wc;
            factor[dst + c * stride + r] = (F)vals[off + c + r * wr];
        }
    }
}

// The seed from f32 values: a block of SEED_LANES threads per stored tile.
KERNEL_THREADS(SEED_LANES) void k_seed_f32(GLOBAL const u32* RESTRICT tiles, GLOBAL const float* RESTRICT vals, GLOBAL F* RESTRICT factor, u32 ntiles) {
    u32 t = block_x();
    if (t < ntiles) seed_tile_lanes<float>(tiles, vals, factor, t, thread_x());
}

// The seed from f64 values (a device without f64 seeds from f32 values the
// host rounded).
#if HAS_F64
KERNEL_THREADS(SEED_LANES) void k_seed_f64(GLOBAL const u32* RESTRICT tiles, GLOBAL const double* RESTRICT vals, GLOBAL F* RESTRICT factor, u32 ntiles) {
    u32 t = block_x();
    if (t < ntiles) seed_tile_lanes<double>(tiles, vals, factor, t, thread_x());
}
#endif

// The seed from values of the route's wide scalar (wide.h): S formed on the device.
KERNEL_THREADS(SEED_LANES) void k_seed_wide(GLOBAL const u32* RESTRICT tiles, GLOBAL const W* RESTRICT vals, GLOBAL F* RESTRICT factor, u32 ntiles) {
    u32 t = block_x();
    if (t < ntiles) seed_tile_lanes<W>(tiles, vals, factor, t, thread_x());
}

// The small descendant updates: one block per piece of `items` (PAIR_ITEM
// words each, plan.rs's LevelWork::pair_items): rows [i0, i1) of a pair
// against its columns [j0, j1), element (i, j) the product of the
// descendant's rows i and j (counted from the pair's first) over its dq
// columns, the lower part (i >= j) subtracted from the target through the
// pair's row and column maps, atomically: other pieces and pairs land on
// the same entries. Both ranges of rows are staged in shared memory PAIR_KC
// columns at a time, threads below ni loading the rows, the next nj the
// columns' rows; thread t owns elements t + u THREADS (u < PAIR_EPT) of the
// piece, row fastest. `src` and `dst` are the factor: the descendant read
// through `src`, the target written through `dst`.
KERNEL_THREADS(THREADS) void k_pair_update(GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT src, GLOBAL F* RESTRICT dst) {
    SHARED F sa[PAIR_KC * PAIR_NI];
    SHARED F sb[PAIR_KC * PAIR_NJ];
    SHARED u32 tr[PAIR_NI];
    SHARED u32 tc[PAIR_NJ];
    GLOBAL const u32* it = items + PAIR_ITEM * block_x();
    u32 t = thread_x();
    u32 base = it[0];
    u32 dh = it[1];
    u32 dq = it[2];
    u32 panel = it[3];
    u32 hs = it[4];
    u32 i0 = it[7];
    u32 j0 = it[9];
    u32 ni = it[8] - i0;
    u32 nj = it[10] - j0;
    if (t < ni) tr[t] = plan[plan[H_RMAP] + it[5] + i0 + t];
    else if (t < ni + nj) tc[t - ni] = plan[plan[H_CMAP] + it[6] + j0 + t - ni];
    u32 ii[PAIR_EPT];
    u32 jj[PAIR_EPT];
    F acc[PAIR_EPT];
#pragma unroll
    for (u32 u = 0; u < PAIR_EPT; u++) {
        u32 e = t + u * THREADS;
        jj[u] = e / ni;
        ii[u] = e - jj[u] * ni;
        acc[u] = F(0);
    }
    for (u32 k0 = 0; k0 < dq; k0 += PAIR_KC) {
        u32 kc = min(dq - k0, PAIR_KC);
        // the previous pass's reads done
        BARRIER();
        if (t < ni) {
            for (u32 k = 0; k < kc; k++) sa[k * ni + t] = src[base + (k0 + k) * dh + i0 + t];
        } else if (t < ni + nj) {
            for (u32 k = 0; k < kc; k++) sb[k * nj + t - ni] = src[base + (k0 + k) * dh + j0 + t - ni];
        }
        BARRIER();
#pragma unroll
        for (u32 u = 0; u < PAIR_EPT; u++) {
            if (jj[u] < nj) {
                F a = acc[u];
                for (u32 k = 0; k < kc; k++) a += sa[k * ni + ii[u]] * sb[k * nj + jj[u]];
                acc[u] = a;
            }
        }
    }
#pragma unroll
    for (u32 u = 0; u < PAIR_EPT; u++) {
        if (jj[u] < nj && i0 + ii[u] >= j0 + jj[u]) atomic_add(&dst[panel + tc[jj[u]] * hs + tr[ii[u]]], F(0) - acc[u]);
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

#if NB != WARP
#error "supernodal.cpp: a panel step's diagonal rows are one warp"
#endif

// The loops below that walk the columns of a row held in registers (an
// array of NB) are not unrolled: each step works on a[0], the row's
// current column, and then shifts the row down one column (a[c - 1] =
// a[c]), so the indices into the array stay constant and the array in
// registers. Unrolled over the columns, a kernel is tens of thousands of
// instructions, and a block of a small level spends its time fetching
// them.
//
// A step's unrolled loop over the row's later columns runs to a constant
// width W of 32, 24, 16 or 8 columns, the narrowest that holds the
// columns left (`left`), the columns past them predicated off: a step
// issues at most 7 columns it does not need (an f64 multiply-add takes 16
// issue cycles where f64 is slow, predicated off or not), and no branch
// splits the loop (which would expose the latency of each column's
// shuffle or load).

// The width of the loop over `left` columns: 32, 24, 16 or 8.
DEVICE u32 step_width(u32 left) {
    return left > 24 ? 32 : left > 16 ? 24 : left > 8 ? 16 : 8;
}

// The row's first W columns shifted down one.
template <u32 W>
DEVICE_INLINE void shift_down(THREAD F (&a)[NB]) {
#pragma unroll
    for (u32 c = 1; c < W; c++) a[c - 1] = a[c];
}

// potf2_regs' pivots k while the columns left (nb - k) fit width W.
template <u32 W, typename Out>
DEVICE_INLINE void potf2_steps(THREAD F (&a)[NB], u32 lane, u32 nb, THREAD u32& k, THREAD F& dr, THREAD bool& bad, THREAD Out& out) {
    for (; k < nb && step_width(nb - k) == W; k++) {
        F d = shfl(a[0], k);
        if (!(d > F(0))) {
            bad = true;
            d = F(1);
        }
        // one reciprocal square root per pivot (where f64 is slow, its
        // correctly rounded square root and division take twice as long)
        F rk = rsqrt(d);
        F lkk = d * rk;
        if (lane == k) {
            a[0] = lkk;
            dr = rk;
        } else if (lane > k) {
            a[0] = a[0] * rk;
        }
        out.put(k, a[0]);
        // column k + c less L[r][k] L[k + c][k]; a source lane past the
        // warp wraps, and its column is past nb
#pragma unroll
        for (u32 c = 1; c < W; c++) {
            F lck = shfl(a[0], k + c);
            if (k + c < nb && lane >= k + c) a[c] -= a[0] * lck;
        }
        shift_down<W>(a);
    }
}

// potf2_regs on a full tile (nb == NB): the 32 pivots unrolled, so every
// index is a constant and row r keeps column c in a[c], without shifts;
// column k reaches the lanes through `col` (2 NB values of shared memory,
// its halves taken by turns, so one warp barrier a step) instead of a
// shuffle per column. The multiply-adds are potf2_steps', in its order.
template <typename Out>
DEVICE_INLINE bool potf2_full(THREAD F (&a)[NB], u32 lane, THREAD F& dr, THREAD Out& out, SHARED_PTR F* col) {
    bool bad = false;
#pragma unroll
    for (u32 k = 0; k < NB; k++) {
        F d = shfl(a[k], k);
        if (!(d > F(0))) {
            bad = true;
            d = F(1);
        }
        F rk = rsqrt(d);
        F lkk = d * rk;
        if (lane == k) {
            a[k] = lkk;
            dr = rk;
        } else if (lane > k) {
            a[k] = a[k] * rk;
        }
        out.put(k, a[k]);
        SHARED_PTR F* ck = col + (k % 2) * NB;
        ck[lane] = a[k];
        WARP_BARRIER();
#pragma unroll
        for (u32 c = k + 1; c < NB; c++) {
            if (lane >= c) a[c] -= a[k] * ck[c];
        }
    }
    return bad;
}

// A warp's 32 rows of an nb-column panel factored in place in registers:
// lane r holds row r from the current column on (a[0]), reading the other
// rows' entries by shuffles. The top nb x nb block becomes its Cholesky
// factor, the rows below it are solved against it (L21 = A21 L11^-T),
// right-looking. Column k of L is handed over as out.put(k, value) on
// every lane (lane r: L[r][k], meaningful for r >= k); `dr` gets
// 1 / L[r][r] on the lanes below nb. A full tile takes potf2_full when
// given `col` (2 NB values of shared memory); without (null) the loop, as
// in k_small_factor, where potf2_full measured slower in f32. Returns
// whether a pivot was not positive; it continues with 1. `a` is undefined
// after.
template <typename Out>
DEVICE_INLINE bool potf2_regs(THREAD F (&a)[NB], u32 lane, u32 nb, THREAD F& dr, THREAD Out& out, SHARED_PTR F* col) {
    if (nb == NB && col != nullptr) return potf2_full(a, lane, dr, out, col);
    bool bad = false;
    u32 k = 0;
    potf2_steps<32>(a, lane, nb, k, dr, bad, out);
    potf2_steps<24>(a, lane, nb, k, dr, bad, out);
    potf2_steps<16>(a, lane, nb, k, dr, bad, out);
    potf2_steps<8>(a, lane, nb, k, dr, bad, out);
    return bad;
}

// potf2_regs' column output into the panel at `tile` (column stride h):
// lanes below nb their lower part, the others (rows below the tile, which
// k_small_factor's last step has in the warp) every column, rows before
// `rows` only.
struct ColsOut {
    GLOBAL F* factor;
    u32 tile;
    u32 h;
    u32 lane;
    u32 nb;
    u32 rows;
    DEVICE void put(u32 k, F v) {
        if (lane < rows && (lane >= nb || lane >= k)) factor[tile + k * h + lane] = v;
    }
};

// The nb x nb lower diagonal tile at `tile` (column stride h) of
// supernode s factored in place by one warp (potf2_regs; with
// POTF2_UNROLL, a full tile by potf2_full, `col` its shared memory). A
// pivot that is not positive sets fail[0] to s plus one and continues
// with 1.
DEVICE_INLINE void potf2_warp(GLOBAL F* factor, u32 tile, u32 h, u32 nb, u32 lane, GLOBAL u32* fail, u32 s, SHARED_PTR F* col) {
    F a[NB];
#pragma unroll
    for (u32 c = 0; c < NB; c++) a[c] = (lane < nb && c <= lane) ? factor[tile + c * h + lane] : F(0);
    F dr = F(1);
    ColsOut out = {factor, tile, h, lane, nb, nb};
    bool bad = potf2_regs(a, lane, nb, dr, out, POTF2_UNROLL ? col : (SHARED_PTR F*)nullptr);
    if (bad && lane == 0) fail[0] = s + 1;
}

// k_small_factor's diagonal warp output: the panel, and for the diagonal
// block's rows the shared copies, `l` (the diagonal block, column-major, q
// rows) and `l11` (the step's tile, NB rows).
struct SmallOut {
    ColsOut cols;
    SHARED_PTR F* l;
    SHARED_PTR F* l11;
    u32 q;
    u32 j;
    u32 r;
    DEVICE void put(u32 k, F v) {
        cols.put(k, v);
        if (r < q && cols.lane >= k) {
            l[(j + k) * q + r] = v;
            l11[k * NB + cols.lane] = v;
        }
    }
};

// The columns of k_small_factor's rows the plain loops below load first:
// 8 in f32, 4 in f64, whose rows take twice the registers.
CONSTANT constexpr u32 SMALL_KU = sizeof(F) == 4 ? 8 : 4;

// small_take's products, the loop over the row's nb columns W wide; L[r][k]
// comes through a ring of SMALL_KU loads ahead (`lk`, loaded).
template <u32 W>
DEVICE_INLINE void small_take_cols(THREAD F (&v)[NB], THREAD F (&lk)[SMALL_KU], GLOBAL const F* factor, u32 panel, u32 h, SHARED_PTR const F* l, u32 q, u32 j, u32 nb, u32 r) {
    for (u32 k = 0; k < j; k++) {
        F x = lk[0];
#pragma unroll
        for (u32 u = 1; u < SMALL_KU; u++) lk[u - 1] = lk[u];
        lk[SMALL_KU - 1] = k + SMALL_KU < j ? factor[panel + (k + SMALL_KU) * h + r] : F(0);
#pragma unroll
        for (u32 c = 0; c < W; c++) {
            if (c < nb) v[c] -= x * l[k * q + j + c];
        }
    }
}

// k_small_factor's row r at the step at column j: its nb values less the
// earlier columns' products, v[c] = A[r][j + c] - sum over k < j of
// L[r][k] L[j + c][k], L's diagonal block (column-major, q rows) in `l`.
DEVICE_INLINE void small_take(THREAD F (&v)[NB], GLOBAL const F* factor, u32 panel, u32 h, SHARED_PTR const F* l, u32 q, u32 j, u32 nb, u32 r) {
#pragma unroll
    for (u32 c = 0; c < NB; c++) v[c] = c < nb ? factor[panel + (j + c) * h + r] : F(0);
    F lk[SMALL_KU];
#pragma unroll
    for (u32 u = 0; u < SMALL_KU; u++) lk[u] = u < j ? factor[panel + u * h + r] : F(0);
    switch (step_width(nb)) {
    case 32: small_take_cols<32>(v, lk, factor, panel, h, l, q, j, nb, r); break;
    case 24: small_take_cols<24>(v, lk, factor, panel, h, l, q, j, nb, r); break;
    case 16: small_take_cols<16>(v, lk, factor, panel, h, l, q, j, nb, r); break;
    default: small_take_cols<8>(v, lk, factor, panel, h, l, q, j, nb, r); break;
    }
}

// small_solve's columns k while the columns left (nb - k) fit width W.
template <u32 W>
DEVICE_INLINE void small_solve_steps(THREAD F (&v)[NB], GLOBAL F* factor, u32 panel, u32 h, SHARED_PTR F* l, SHARED_PTR const F* l11, SHARED_PTR const F* d11, u32 q, u32 j, u32 nb, u32 r, THREAD u32& k) {
    for (; k < nb && step_width(nb - k) == W; k++) {
        F x = v[0] * d11[k];
        factor[panel + (j + k) * h + r] = x;
        if (r < q) l[(j + k) * q + r] = x;
#pragma unroll
        for (u32 c = 1; c < W; c++) {
            if (k + c < nb) v[c] -= x * l11[k * NB + k + c];
        }
        shift_down<W>(v);
    }
}

// k_small_factor's row r below the step's diagonal tile solved against it
// column by column (the tile column-major in `l11`, NB rows, the
// reciprocals of its diagonal in `d11`) and stored; a row of the diagonal
// block also into `l`.
DEVICE_INLINE void small_solve(THREAD F (&v)[NB], GLOBAL F* factor, u32 panel, u32 h, SHARED_PTR F* l, SHARED_PTR const F* l11, SHARED_PTR const F* d11, u32 q, u32 j, u32 nb, u32 r) {
    u32 k = 0;
    small_solve_steps<32>(v, factor, panel, h, l, l11, d11, q, j, nb, r, k);
    small_solve_steps<24>(v, factor, panel, h, l, l11, d11, q, j, nb, r, k);
    small_solve_steps<16>(v, factor, panel, h, l, l11, d11, q, j, nb, r, k);
    small_solve_steps<8>(v, factor, panel, h, l, l11, d11, q, j, nb, r, k);
}

// A small supernode's Cholesky, supernode s by the block (nt threads,
// thread t), in NB-column steps; thread r owns the panel's row r, and rows r +
// nt, r + 2 nt, ... for a block of nt threads (q <= nt: the diagonal
// block's rows are the first). A step at column j: every row from j on
// takes its nb values less the earlier columns' products, the diagonal
// block's rows of those columns read from shared memory (q * q + min(q, NB)
// * NB + NB values of dynamic shared memory: the diagonal block of L,
// column-major; the step's diagonal tile, its columns NB apart, so at
// constant offsets, and its diagonal's reciprocals); the warp
// holding rows j to j + 31 factors them in registers (potf2_regs), the
// other rows solve against the step's diagonal tile. A pivot that is not
// positive sets fail[0] to the supernode plus one and continues with 1. The
// caller has the block wait before `l` is used again.
DEVICE_INLINE void small_factor(u32 s, GLOBAL const u32* plan, GLOBAL F* factor, GLOBAL u32* fail, SHARED_PTR F* l, u32 t, u32 nt) {
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    u32 panel = plan[o_val_ptr + s];
    u32 lane = t % NB;
    SHARED_PTR F* l11 = l + q * q;
    SHARED_PTR F* d11 = l11 + min(q, NB) * NB;
    bool bad = false;
    for (u32 j = 0; j < q; j += NB) {
        u32 nb = min(q - j, NB);
        F v[NB];
        if (t >= j && t < h) {
            small_take(v, factor, panel, h, l, q, j, nb, t);
        } else {
#pragma unroll
            for (u32 c = 0; c < NB; c++) v[c] = F(0);
        }
        bool diag_warp = t / NB == j / NB;
        if (diag_warp) {
            // the step's diagonal rows and, in a last step narrower than
            // the warp, the rows below them
            F dr = F(1);
            SmallOut out = {{factor, panel + j * h + j, h, lane, nb, h - j}, l, l11, q, j, t};
            bad = potf2_regs(v, lane, nb, dr, out, (SHARED_PTR F*)nullptr) || bad;
            if (lane < nb) d11[lane] = dr;
        }
        BARRIER();
        if (!diag_warp && t >= j + NB && t < h) small_solve(v, factor, panel, h, l, l11, d11, q, j, nb, t);
        for (u32 r = t + nt; r < h; r += nt) {
            small_take(v, factor, panel, h, l, q, j, nb, r);
            small_solve(v, factor, panel, h, l, l11, d11, q, j, nb, r);
        }
        // the step's rows of the diagonal block, read by the next step, and
        // its tile, rewritten by the next step
        if (j + NB < q) BARRIER();
    }
    if (bad) fail[0] = s + 1;
}

// The small supernodes' factors, one block per supernode of `list`
// (small_factor).
KERNEL_THREADS_REGS(THREADS) void k_small_factor(GLOBAL const u32* RESTRICT list, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor, GLOBAL u32* RESTRICT fail) {
    DYNAMIC_SHARED(F, l);
    small_factor(list[block_x()], plan, factor, fail, l, thread_x(), block_dim_x());
}

// k_chain's update piece at `it` (as k_pair_update's) by one warp (lane
// `lane`): the piece's elements 4 a lane at a time, their products straight
// from the factor.
DEVICE void warp_piece(GLOBAL const u32* it, GLOBAL const u32* plan, GLOBAL F* factor, u32 lane) {
    u32 base = it[0];
    u32 dh = it[1];
    u32 dq = it[2];
    u32 panel = it[3];
    u32 hs = it[4];
    u32 rm = plan[H_RMAP] + it[5];
    u32 cm = plan[H_CMAP] + it[6];
    u32 i0 = it[7];
    u32 j0 = it[9];
    u32 ni = it[8] - i0;
    u32 n = ni * (it[10] - j0);
    for (u32 e0 = 0; e0 < n; e0 += 4 * NB) {
        u32 ii[4];
        u32 jj[4];
        F acc[4];
#pragma unroll
        for (u32 u = 0; u < 4; u++) {
            u32 e = min(e0 + lane + u * NB, n - 1);
            jj[u] = e / ni;
            ii[u] = e - jj[u] * ni;
            acc[u] = F(0);
        }
        for (u32 k = 0; k < dq; k++) {
#pragma unroll
            for (u32 u = 0; u < 4; u++) acc[u] += factor[base + k * dh + i0 + ii[u]] * factor[base + k * dh + j0 + jj[u]];
        }
#pragma unroll
        for (u32 u = 0; u < 4; u++) {
            if (e0 + lane + u * NB < n && i0 + ii[u] >= j0 + jj[u]) atomic_add(&factor[panel + plan[cm + j0 + jj[u]] * hs + plan[rm + i0 + ii[u]]], F(0) - acc[u]);
        }
    }
}

// A chain of levels factored by one block in one launch: per level
// (`levels`, 4 words: the word offsets into `plan` of its small update
// pieces and of its small targets, and their counts), the pieces a warp
// each, then the targets' factors in turn. For the top of a tree, small
// levels one above the other, whose launches would cost more than their
// work. The levels read what the levels before them wrote in the same
// launch, so `factor` is one plain pointer (not the read-only path); `l`
// is the factors' dynamic shared memory, for the widest of the chain's
// targets.
KERNEL_THREADS_REGS(THREADS) void k_chain(GLOBAL const u32* RESTRICT levels, u32 nlevels, GLOBAL const u32* RESTRICT plan, GLOBAL F* factor, GLOBAL u32* RESTRICT fail) {
    DYNAMIC_SHARED(F, l);
    u32 t = thread_x();
    for (u32 lv = 0; lv < nlevels; lv++) {
        u32 pieces = levels[4 * lv];
        u32 npieces = levels[4 * lv + 1];
        u32 targets = levels[4 * lv + 2];
        u32 ntargets = levels[4 * lv + 3];
        for (u32 p = t / NB; p < npieces; p += THREADS / NB) warp_piece(plan + pieces + PAIR_ITEM * p, plan, factor, t % NB);
        for (u32 g = 0; g < ntargets; g++) {
            // the updates in; the previous factor's reads of `l` done
            BARRIER();
            small_factor(plan[targets + g], plan, factor, fail, l, t, THREADS);
        }
        // the level's factors in, read by the next level's updates
        BARRIER();
    }
}

// Panel step, part one, for a level's first step: the NB x NB diagonal
// tile at (0, 0) of every big panel of the level, one warp per panel. The
// later steps' diagonal tiles are factored by the step before them, at the
// end of k_syrk.
KERNEL_THREADS_REGS(NB) void k_potf2(GLOBAL const u32* RESTRICT panels, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor, GLOBAL u32* RESTRICT fail) {
    SHARED F col[2 * NB];
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 s = panels[block_x()];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    potf2_warp(factor, plan[o_val_ptr + s], h, min(q, NB), thread_x(), fail, s, col);
}

#if NB * NB % TRSM_ROWS != 0 || TRSM_ROWS < NB
#error "supernodal.cpp: TRSM_ROWS divides NB * NB and is at least NB"
#endif

// Panel step, part two: the rows below the step's diagonal tile,
// right-solved against it (L21 = A21 L11^-T); TRSM_ROWS rows per block of
// TRSM_ROWS threads, the block index walking the level's panels through
// their row-chunk counts. Right-looking: column k is finished, then taken
// off the columns after it, so a value gets the same multiply-adds in the
// same order as in a left-looking solve, but the dependent chain is two
// operations per column instead of one per column before it.
KERNEL_THREADS(TRSM_ROWS) void k_trsm(GLOBAL const u32* RESTRICT panels, u32 npanels, u32 j, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor) {
    // the diagonal tile's lower part, column-major: column k at lc[k NB]
    SHARED F lc[NB * NB];
    // the diagonal's reciprocals: one division per column, not per row
    SHARED F dinv[NB];
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
        u32 chunks = (rows + TRSM_ROWS - 1) / TRSM_ROWS;
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
    // consecutive threads read consecutive rows of a column; the passes are
    // independent loads, all in flight at once
#pragma unroll
    for (u32 pass = 0; pass < NB * NB / TRSM_ROWS; pass++) {
        u32 e = pass * TRSM_ROWS + t;
        u32 c = e / NB;
        u32 r = e - c * NB;
        F v = F(0);
        if (r < nb && c <= r) v = factor[panel + (j + c) * h + j + r];
        lc[e] = v;
    }
    BARRIER();
    if (t < nb) dinv[t] = F(1) / lc[t * NB + t];
    BARRIER();
    u32 i = j + nb + idx * TRSM_ROWS + t;
    if (i < h) {
        // The loops run to the constant NB and skip the columns past nb, so
        // they unroll fully, every index into v is a constant and the row
        // stays in registers. With nb as the bound, v went to local memory
        // and each step of the solve waited on a load.
        F v[NB];
#pragma unroll
        for (u32 c = 0; c < NB; c++) v[c] = c < nb ? factor[panel + (j + c) * h + i] : F(0);
#pragma unroll
        for (u32 k = 0; k < NB; k++) {
            if (k < nb) {
                v[k] *= dinv[k];
                // the columns past nb hold zeros and stay zero
#pragma unroll
                for (u32 c = k + 1; c < NB; c++) v[c] -= v[k] * lc[k * NB + c];
            }
        }
#pragma unroll
        for (u32 c = 0; c < NB; c++) {
            if (c < nb) factor[panel + (j + c) * h + i] = v[c];
        }
    }
}

#if SYRK_BLOCK % NB != 0 || SYRK_BLOCK < NB
#error "supernodal.cpp: SYRK_BLOCK is a multiple of NB"
#endif

#if SYRK_TB != 32 && SYRK_TB != 64
#error "supernodal.cpp: SYRK_TB is 32 or 64"
#endif
// A k_syrk thread's accumulator rows (and columns) in its tile, 16 apart.
#define SYRK_R (SYRK_TB / 16)

// One slab of nk staged columns of a 16 R x 16 R tile into a thread's
// accumulators: rows tx + 16 r of sa against columns ty + 16 c of sb (each
// staged column 16 R values), acc[R r + c]. On a diagonal tile (DIAG) the
// accumulators with r < c lie wholly above the diagonal, which the tile
// does not write, and are skipped.
template <u32 R, bool DIAG> DEVICE_INLINE void tile_slab(THREAD F (&acc)[R * R], SHARED_PTR const F* sa, SHARED_PTR const F* sb, u32 nk, u32 tx, u32 ty) {
    for (u32 kk = 0; kk < nk; kk++) {
        F a[R];
        F b[R];
#pragma unroll
        for (u32 r = 0; r < R; r++) a[r] = sa[kk * 16 * R + tx + 16 * r];
#pragma unroll
        for (u32 c = 0; c < R; c++) b[c] = sb[kk * 16 * R + ty + 16 * c];
#pragma unroll
        for (u32 r = 0; r < R; r++) {
#pragma unroll
            for (u32 c = 0; c < R; c++) {
                if (!DIAG || r >= c) acc[r * R + c] += a[r] * b[c];
            }
        }
    }
}

// Panel step, part three: the update of the trailing lower trapezoid by
// the solved columns [k0, k1), k1 = min(k0 + kn, q), over rows [k1, h) and
// columns [k1, c1), c1 = min(cend, q), in SYRK_TB x SYRK_TB tiles
// (plan::syrk_update picks k0, kn and cend per step for blocks of
// SYRK_BLOCK columns: a step inside a block updates the rest of its block,
// the block's last step everything past it); tiles above the diagonal are
// skipped and the straddling tile clipped. The columns stream through
// shared memory NB at a time into the tile's accumulators, so a tile is
// read and written once per update; the loop's trip count is constant, so
// with SYRK_BLOCK = NB it is a single pass. A thread takes the
// rows tx + 16 r and columns ty + 16 c, so neighbouring threads read and
// write neighbouring rows. The block index walks the panels' tile counts.
// The block of a panel's tile (0, 0), which alone writes the next diagonal
// tile (at k1, SYRK_TB >= NB), then factors it (potf2_warp). In f64 with
// POTF2_UNROLL the kernel is held to two blocks a multiprocessor:
// potf2_full would take it to 168 registers, one block; held, it spills a
// little in the tail, which measured faster on bundle adjustment's wide
// steps and 1.5% slower on SLAM's narrow ones.
#if SCALAR_BYTES == 8 && POTF2_UNROLL
#define SYRK_ENTRY KERNEL_THREADS_BLOCKS(THREADS, 2)
#else
#define SYRK_ENTRY KERNEL_THREADS(THREADS)
#endif
SYRK_ENTRY void k_syrk(GLOBAL const u32* RESTRICT panels, u32 npanels, u32 k0, u32 kn, u32 cend, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor, GLOBAL u32* RESTRICT fail) {
    SHARED F sa[NB * SYRK_TB];
    SHARED F sb[NB * SYRK_TB];
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 idx = block_x();
    u32 p = 0;
    bool found = false;
    u32 s = 0;
    u32 q = 0;
    u32 h = 0;
    u32 k1 = 0;
    u32 c1 = 0;
    u32 nct = 0;
    while (p < npanels && !found) {
        s = panels[p];
        q = plan[o_dims + 2 * s];
        h = plan[o_dims + 2 * s + 1];
        u32 rows = 0;
        u32 cols = 0;
        if (k0 < q) {
            k1 = min(k0 + kn, q);
            c1 = min(cend, q);
            if (k1 < c1) {
                rows = h - k1;
                cols = c1 - k1;
            }
        }
        nct = (cols + SYRK_TB - 1) / SYRK_TB;
        u32 tiles = ((rows + SYRK_TB - 1) / SYRK_TB) * nct;
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
    u32 row0 = k1 + ti * SYRK_TB;
    u32 col0 = k1 + tj * SYRK_TB;
    u32 nrows = min(h - row0, SYRK_TB);
    u32 ncols = min(c1 - col0, SYRK_TB);
    u32 panel = plan[o_val_ptr + s];
    u32 t = thread_x();
    const u32 nt = THREADS;
    u32 tx = t % 16;
    u32 ty = t / 16;
    F acc[SYRK_R * SYRK_R];
#pragma unroll
    for (u32 z = 0; z < SYRK_R * SYRK_R; z++) acc[z] = F(0);
    for (u32 sl = 0; sl < SYRK_BLOCK / NB; sl++) {
        u32 kb = k0 + sl * NB;
        if (kb >= k1) break;
        u32 nk = min(k1 - kb, NB);
        // the previous slab's reads are done
        if (sl > 0) BARRIER();
        for (u32 e = t; e < NB * SYRK_TB; e += nt) {
            u32 k = e / SYRK_TB;
            u32 i = e - k * SYRK_TB;
            F va = F(0);
            F vb = F(0);
            if (k < nk && i < nrows) va = factor[panel + (kb + k) * h + row0 + i];
            if (k < nk && i < ncols) vb = factor[panel + (kb + k) * h + col0 + i];
            sa[e] = va;
            sb[e] = vb;
        }
        BARRIER();
        // the branch is uniform: a block's tile is diagonal or not
        if (ti == tj) {
            tile_slab<SYRK_R, true>(acc, sa, sb, nk, tx, ty);
        } else {
            tile_slab<SYRK_R, false>(acc, sa, sb, nk, tx, ty);
        }
    }
#pragma unroll
    for (u32 r = 0; r < SYRK_R; r++) {
#pragma unroll
        for (u32 c = 0; c < SYRK_R; c++) {
            u32 i = tx + 16 * r;
            u32 jj = ty + 16 * c;
            if (i < nrows && jj < ncols && row0 + i >= col0 + jj) factor[panel + (col0 + jj) * h + row0 + i] -= acc[r * SYRK_R + c];
        }
    }
    if (ti == 0 && tj == 0) {
        // the block's writes seen by its first warp, and its reads of `sa`
        // done (potf2_warp's column)
        BARRIER();
        if (t < NB) potf2_warp(factor, panel + k1 * h + k1, h, min(q - k1, NB), t, fail, s, sa);
    }
}

#if TB % 16 != 0
#error "supernodal.cpp: TB is a multiple of 16"
#endif

// The big descendants' updates tiled in each pair's own rows and columns
// (GpuSupernodalParams::pair_tiles): one block per item (PAIR_TILE words:
// the target s, the pair, the tile's first row i0 and column j0 of the
// pair's m x kw block, its rows and columns, and the k range [k0, k1), k1
// 0: every k). The tile's products over the k range stream through shared
// memory TK columns at a time (tile_slab: rows tx + 16 r, columns
// ty + 16 c, the last slab to the k range's end); then each element at or
// below the pair's diagonal (pair row i >= column j) is subtracted from
// the target through the pair's maps, atomically, as other items reach
// the same element. A warp's atomics take 16 consecutive pair rows, mostly
// consecutive target rows.
KERNEL_THREADS(THREADS) void k_big_pair(GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor) {
    SHARED F sa[TK * TB];
    SHARED F sb[TK * TB];
    GLOBAL const u32* it = items + PAIR_TILE * block_x();
    u32 s = it[0];
    u32 pi = it[1];
    u32 i0 = it[2];
    u32 j0 = it[3];
    u32 nr = it[4];
    u32 nc = it[5];
    u32 kb = it[6];
    u32 ke = it[7];
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 d = plan[plan[H_PR_D] + pi];
    u32 dh = plan[o_dims + 2 * d + 1];
    u32 dp = plan[o_val_ptr + d] + plan[plan[H_PR_R0] + pi];
    u32 k_end = ke == 0 ? plan[o_dims + 2 * d] : ke;
    u32 t = thread_x();
    const u32 nt = THREADS;
    u32 tx = t % 16;
    u32 ty = t / 16;
    F acc[(TB / 16) * (TB / 16)];
#pragma unroll
    for (u32 z = 0; z < (TB / 16) * (TB / 16); z++) acc[z] = F(0);
    for (u32 k0 = kb; k0 < k_end; k0 += TK) {
        u32 nk = min(k_end - k0, TK);
        // the previous slab's reads are done
        BARRIER();
        for (u32 e = t; e < TK * TB; e += nt) {
            u32 kk = e / TB;
            u32 i = e - kk * TB;
            F va = F(0);
            F vb = F(0);
            if (kk < nk && i < nr) va = factor[dp + (k0 + kk) * dh + i0 + i];
            if (kk < nk && i < nc) vb = factor[dp + (k0 + kk) * dh + j0 + i];
            sa[e] = va;
            sb[e] = vb;
        }
        BARRIER();
        // the branch is uniform: a block's tile is diagonal or not
        if (i0 == j0) {
            tile_slab<TB / 16, true>(acc, sa, sb, nk, tx, ty);
        } else {
            tile_slab<TB / 16, false>(acc, sa, sb, nk, tx, ty);
        }
    }
    u32 rm = plan[H_RMAP] + plan[plan[H_PR_RMAP] + pi] + i0;
    u32 cm = plan[H_CMAP] + plan[plan[H_PR_CMAP] + pi] + j0;
    u32 panel = plan[o_val_ptr + s];
    u32 h = plan[o_dims + 2 * s + 1];
#pragma unroll
    for (u32 r = 0; r < TB / 16; r++) {
#pragma unroll
        for (u32 c = 0; c < TB / 16; c++) {
            u32 i = tx + 16 * r;
            u32 j = ty + 16 * c;
            if (i < nr && j < nc && i0 + i >= j0 + j) atomic_add(&factor[panel + plan[cm + j] * h + plan[rm + i]], -acc[r * (TB / 16) + c]);
        }
    }
}

// Forward solve, part one: the gather of `n` mid rows from `rows` (their
// GATHER_ROW fields `stride` apart, plan.rs's LevelWork::g_rows: the
// descendant's row at its column 0, the target's x index, the descendant,
// whose height, width and first scalar column are in its solve header) by
// nt threads; every mid row subtracts its descendant row's dot product with
// the descendant's final y from its target's x, atomically. `x` and `xa`
// are the same vector: the descendants' y read through `x`, the targets
// written through `xa`.
DEVICE_INLINE void fwd_gather_rows(GLOBAL const u32* plan, GLOBAL const u32* rows, u32 n, u32 stride, GLOBAL const F* factor, GLOBAL const F* x, GLOBAL F* xa, u32 t, u32 nt) {
    GLOBAL const u32* heads = plan + plan[H_SOLVE_HEAD];
    for (u32 e = t; e < n; e += nt) {
        u32 dp = rows[e];
        u32 tgt = rows[stride + e];
        GLOBAL const u32* head = heads + SOLVE_HEAD * rows[2 * stride + e];
        u32 dq = head[0];
        u32 dh = head[1];
        u32 dcol0 = head[2];
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
        atomic_add(&xa[tgt], F(0) - ((a0 + a1) + (a2 + a3)));
    }
}

// The forward gather, one block per item of `items` (3 words: the word
// offset into `plan` of its first row, its rows, the level's rows, which
// part the fields; fwd_gather_rows).
KERNEL_THREADS(THREADS) void k_fwd_gather(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL const F* RESTRICT x, GLOBAL F* RESTRICT xa) {
    if (gate[0] == 0) return;
    GLOBAL const u32* it = items + 3 * block_x();
    fwd_gather_rows(plan, plan + it[0], it[1], it[2], factor, x, xa, thread_x(), THREADS);
}

// A lane's share of sum_i a[a0 + i] b[b0 + i] over i = lane, lane + 32, ...:
// LU loads in flight per iteration into two alternating accumulators, then
// the one-wide remainder.
DEVICE_INLINE F lane_dot(GLOBAL const F* a, u32 a0, SHARED_PTR const F* b, u32 b0, u32 n, u32 lane) {
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

#if INV_TILE != 2 * 32 * 32
#error "supernodal.cpp: an inverted tile is two 32 x 32 copies"
#endif

// The inverse X of every 32 x 32 diagonal tile L of the supernodes wider
// than NB (the plan's list at H_INV_TILES: the supernode and the tile's
// first column), a warp per tile, into its INV_TILE values at `inv`: X
// column-major (the forward solve's), then row-major (the backward
// solve's), so a lane of either solve reads its row of X, or of X^T,
// coalesced. Lane c solves L x = e_c by forward substitution; past the
// tile's nb columns X is zero.
KERNEL_THREADS(32) void k_tile_inv(GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* RESTRICT inv) {
    // L's lower part, column k at ls[33 k]; then X, column c at ls[33 c]
    SHARED F ls[32 * 33];
    u32 tile = block_x();
    GLOBAL const u32* it = plan + plan[H_INV_TILES] + 2 * tile;
    u32 s = it[0];
    u32 j0 = it[1];
    GLOBAL const u32* head = plan + plan[H_SOLVE_HEAD] + SOLVE_HEAD * s;
    u32 q = head[0];
    u32 h = head[1];
    u32 panel = head[3];
    u32 nb = min(q - j0, 32u);
    u32 lane = thread_x();
    for (u32 k = 0; k < 32; k++) ls[k * 33 + lane] = (k < nb && lane < nb && lane >= k) ? factor[panel + (j0 + k) * h + j0 + lane] : F(0);
    WARP_BARRIER();
    F b[32];
#pragma unroll
    for (u32 r = 0; r < 32; r++) b[r] = (r == lane && lane < nb) ? F(1) : F(0);
#pragma unroll
    for (u32 k = 0; k < 32; k++) {
        if (k < nb) {
            b[k] = b[k] / ls[k * 33 + k];
#pragma unroll
            for (u32 r = k + 1; r < 32; r++) b[r] -= ls[k * 33 + r] * b[k];
        }
    }
    // every lane's reads of L done
    WARP_BARRIER();
#pragma unroll
    for (u32 r = 0; r < 32; r++) ls[lane * 33 + r] = b[r];
    WARP_BARRIER();
    GLOBAL F* out = inv + tile * INV_TILE;
    // X[r][c] at ls[33 c + r]: column-major at c 32 + r, row-major at 1024 + r 32 + c
    for (u32 c = 0; c < 32; c++) out[c * 32 + lane] = ls[c * 33 + lane];
    for (u32 r = 0; r < 32; r++) out[1024 + r * 32 + lane] = ls[lane * 33 + r];
}

// A diagonal tile's solve as a product with its inverse, by one warp:
// lane r's y_r = sum over c < nb of M[r][c] t_c, M at `xi` with M[r][c] at
// c 32 + r (the forward copy of k_tile_inv for M = L^-1, the backward copy
// for M = L^-T); t and then y at `xs`.
DEVICE_INLINE void tile_inv_apply(GLOBAL const F* xi, SHARED_PTR F* xs, u32 nb, u32 lane) {
    F v[32];
#pragma unroll
    for (u32 c = 0; c < 32; c++) v[c] = xi[c * 32 + lane];
    F y = F(0);
#pragma unroll
    for (u32 c = 0; c < 32; c++) {
        if (c < nb) y += v[c] * xs[c];
    }
    // every lane's reads of t done
    WARP_BARRIER();
    if (lane < nb) xs[lane] = y;
}

// The forward steps k of trsv_fwd_warp while the columns left (q - k) fit
// width W: lane k's y_k = b_k / L_kk, the lanes below less L_rk y_k.
template <u32 W>
DEVICE_INLINE void trsv_fwd_steps(THREAD F (&a)[NB], THREAD F& b, F rd, u32 lane, u32 q, THREAD u32& k) {
    for (; k < q && step_width(q - k) == W; k++) {
        if (lane == k) b = b * rd;
        F y = shfl(b, k);
        if (lane > k) b -= a[0] * y;
        shift_down<W>(a);
    }
}

// L y = b for a diagonal block of q <= 32 columns at `panel` (column stride
// h) by one warp in registers: lane r holds row r of L from the current
// column on (shifted down one a step, as potf2_regs) and b_r, and returns
// y_r.
DEVICE_INLINE F trsv_fwd_warp(GLOBAL const F* factor, u32 panel, u32 h, u32 q, F b, u32 lane) {
    F a[NB];
#pragma unroll
    for (u32 c = 0; c < NB; c++) a[c] = (lane < q && c <= lane) ? factor[panel + c * h + lane] : F(0);
    F rd = lane < q ? F(1) / factor[panel + lane * h + lane] : F(0);
    u32 k = 0;
    trsv_fwd_steps<32>(a, b, rd, lane, q, k);
    trsv_fwd_steps<24>(a, b, rd, lane, q, k);
    trsv_fwd_steps<16>(a, b, rd, lane, q, k);
    trsv_fwd_steps<8>(a, b, rd, lane, q, k);
    return b;
}

// The backward steps k of trsv_bwd_warp (column c = q - 1 - k) while the
// rows left fit width W: lane c's x_c = b_c / L_cc, the lanes above less
// L_cr x_c.
template <u32 W>
DEVICE_INLINE void trsv_bwd_steps(THREAD F (&a)[NB], THREAD F& b, F rd, u32 lane, u32 q, THREAD u32& k) {
    for (; k < q && step_width(q - k) == W; k++) {
        u32 c = q - 1 - k;
        if (lane == c) b = b * rd;
        F xc = shfl(b, c);
        if (lane < c) b -= a[0] * xc;
        shift_down<W>(a);
    }
}

// L^T x = b for a diagonal block of q <= 32 columns by one warp in
// registers: lane r holds column r of L from the last row up (a[i] =
// L[q - 1 - i][r], shifted down one a step) and b_r, and returns x_r.
DEVICE_INLINE F trsv_bwd_warp(GLOBAL const F* factor, u32 panel, u32 h, u32 q, F b, u32 lane) {
    F a[NB];
#pragma unroll
    for (u32 i = 0; i < NB; i++) a[i] = (lane < q && i < q && q - 1 - i > lane) ? factor[panel + lane * h + q - 1 - i] : F(0);
    F rd = lane < q ? F(1) / factor[panel + lane * h + lane] : F(0);
    u32 k = 0;
    trsv_bwd_steps<32>(a, b, rd, lane, q, k);
    trsv_bwd_steps<24>(a, b, rd, lane, q, k);
    trsv_bwd_steps<16>(a, b, rd, lane, q, k);
    trsv_bwd_steps<8>(a, b, rd, lane, q, k);
    return b;
}

// Row i's share of a forward step: L[i][j0 + c] y[c] over the step's nb
// columns (at `panel`, column stride h), the full step's 32 loads issued
// before its FMAs; the one-block and the many-block forward solves both
// take a row's steps through it.
DEVICE_INLINE F trsv_row_step(GLOBAL const F* factor, u32 panel, u32 h, u32 j0, u32 nb, u32 i, SHARED_PTR const F* y) {
    F acc = F(0);
    if (nb == 32) {
        F v[32];
#pragma unroll
        for (u32 c = 0; c < 32; c++) v[c] = factor[panel + (j0 + c) * h + i];
#pragma unroll
        for (u32 c = 0; c < 32; c++) acc += v[c] * y[c];
    } else {
        for (u32 c = 0; c < nb; c++) acc += factor[panel + (j0 + c) * h + i] * y[c];
    }
    return acc;
}

// The forward trsv of the supernode at `head` (its solve header) by the
// block over its columns [jlo, jend) (multiples of 32, or q), held in `xs`
// from column jlo: the columns before js are solved and only taken off the
// rows from js on (not off each other); from js on, each 32-column step's
// diagonal tile is solved by the first warp as a product with its inverse
// (`inv`, k_tile_inv) and taken off the rows after it, up to jend. `t` is
// the thread.
DEVICE_INLINE void fwd_trsv_steps(GLOBAL const u32* head, GLOBAL const F* factor, GLOBAL const F* inv, SHARED_PTR F* xs, u32 jlo, u32 js, u32 jend, u32 t) {
    u32 q = head[0];
    u32 h = head[1];
    u32 panel = head[3];
    u32 tile0 = head[5];
    const u32 nt = THREADS;
    u32 lane = t % 32;
    for (u32 j0 = jlo; j0 < jend; j0 += 32) {
        u32 nb = min(q - j0, 32u);
        // the step's x in: loaded, or updated by the steps before it
        BARRIER();
        if (j0 >= js && t < 32) tile_inv_apply(inv + (tile0 + j0 / 32) * INV_TILE, xs + (j0 - jlo), nb, lane);
        BARRIER();
        for (u32 i = max(j0 + nb, js) + t; i < jend; i += nt) xs[i - jlo] -= trsv_row_step(factor, panel, h, j0, nb, i, xs + (j0 - jlo));
    }
}

// Forward solve, part two: the trsv of supernode s by the block of THREADS
// threads; a supernode of at most 32 columns by the first warp in
// registers (trsv_fwd_warp), a wider one in 32-column steps
// (fwd_trsv_steps). `xs`: q values (dynamic shared memory for the widest
// supernode q of the launch). `t` is the thread.
DEVICE_INLINE void fwd_trsv_one(u32 s, GLOBAL const u32* plan, GLOBAL const F* factor, GLOBAL const F* inv, GLOBAL F* x, SHARED_PTR F* xs, u32 t) {
    GLOBAL const u32* head = plan + plan[H_SOLVE_HEAD] + SOLVE_HEAD * s;
    u32 q = head[0];
    u32 h = head[1];
    u32 col0 = head[2];
    u32 panel = head[3];
    const u32 nt = THREADS;
    u32 lane = t % 32;
    if (q <= NB) {
        // one 32-column block: the first warp in registers
        if (t < NB) {
            F b = trsv_fwd_warp(factor, panel, h, q, lane < q ? x[col0 + lane] : F(0), lane);
            if (lane < q) x[col0 + lane] = b;
        }
        return;
    }
    for (u32 c0 = t; c0 < q; c0 += nt) xs[c0] = x[col0 + c0];
    fwd_trsv_steps(head, factor, inv, xs, 0, 0, q, t);
    BARRIER();
    for (u32 c1 = t; c1 < q; c1 += nt) x[col0 + c1] = xs[c1];
}

// The forward trsv, one block per supernode of `list` (fwd_trsv_one).
KERNEL_THREADS(THREADS) void k_fwd_trsv(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT list, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL const F* RESTRICT inv, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    DYNAMIC_SHARED(F, smem);
    fwd_trsv_one(list[block_x()], plan, factor, inv, x, smem, thread_x());
}

// Backward solve, part one: x_s -= L_below^T x for the columns [c0, c1) of
// supernode s by the block of THREADS threads; the rows' x gathered into
// `xr` XR_CH at a time, a warp per column with the lanes over the rows. A
// column's sums do not depend on the range it is in. `t` is the thread.
DEVICE_INLINE void bwd_gemv_cols(u32 s, u32 c0, u32 c1, GLOBAL const u32* plan, GLOBAL const F* factor, GLOBAL F* x, SHARED_PTR F* xr, u32 t) {
    u32 o_row_sc = plan[H_ROW_SC];
    GLOBAL const u32* head = plan + plan[H_SOLVE_HEAD] + SOLVE_HEAD * s;
    u32 q = head[0];
    u32 h = head[1];
    u32 col0 = head[2];
    u32 panel = head[3];
    u32 rsc = head[4];
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

// The GEMV item at `it`, 3 words: its supernode and columns [c0, c1).
DEVICE_INLINE void bwd_gemv_item(GLOBAL const u32* it, GLOBAL const u32* plan, GLOBAL const F* factor, GLOBAL F* x, SHARED_PTR F* xr, u32 t) {
    bwd_gemv_cols(it[0], it[1], it[2], plan, factor, x, xr, t);
}

// The backward GEMV, one block per item (bwd_gemv_item).
KERNEL_THREADS(THREADS) void k_bwd_gemv(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    SHARED F xr[XR_CH];
    bwd_gemv_item(items + 3 * block_x(), plan, factor, x, xr, thread_x());
}

// The transposed trsv of the supernode at `head` by the block over its
// columns [cbegin, cend) against its rows up to rend, `xs` holding
// [cbegin, rend) from cbegin (the rows from cend on solved): 32-column
// steps from the last, each step's columns taken the rows after the step
// off (a warp per column), then its diagonal tile solved by the first warp
// as a product with its inverse transposed (`inv`, k_tile_inv). `t` is the
// thread.
DEVICE_INLINE void bwd_trsv_steps(GLOBAL const u32* head, GLOBAL const F* factor, GLOBAL const F* inv, SHARED_PTR F* xs, u32 cbegin, u32 cend, u32 rend, u32 t) {
    u32 q = head[0];
    u32 h = head[1];
    u32 panel = head[3];
    u32 tile0 = head[5];
    u32 lane = t % 32;
    u32 warp = t / 32;
    u32 nwarps = THREADS / 32;
    u32 bi = (cend + 31) / 32;
    while (bi > cbegin / 32) {
        bi -= 1;
        u32 j0 = bi * 32;
        u32 nb = min(q - j0, 32u);
        u32 n = rend - j0 - nb;
        for (u32 c = warp; c < nb; c += nwarps) {
            u32 pc = panel + (j0 + c) * h + j0 + nb;
            F part = warp_sum(lane_dot(factor, pc, xs, j0 + nb - cbegin, n, lane));
            if (lane == 0) xs[j0 + c - cbegin] -= part;
        }
        BARRIER();
        if (t < 32) tile_inv_apply(inv + (tile0 + bi) * INV_TILE + 1024, xs + (j0 - cbegin), nb, lane);
        BARRIER();
    }
}

// Backward solve, part two: the transposed trsv of supernode s by the block
// of THREADS threads; a supernode of at most 32 columns by the first warp
// in registers (trsv_bwd_warp), a wider one in 32-column steps from the
// last (bwd_trsv_steps). `xs` and `t` as fwd_trsv_one's.
DEVICE_INLINE void bwd_trsv_one(u32 s, GLOBAL const u32* plan, GLOBAL const F* factor, GLOBAL const F* inv, GLOBAL F* x, SHARED_PTR F* xs, u32 t) {
    GLOBAL const u32* head = plan + plan[H_SOLVE_HEAD] + SOLVE_HEAD * s;
    u32 q = head[0];
    u32 h = head[1];
    u32 col0 = head[2];
    u32 panel = head[3];
    const u32 nt = THREADS;
    u32 lane = t % 32;
    if (q <= NB) {
        // one 32-column block: the first warp in registers
        if (t < NB) {
            F b = trsv_bwd_warp(factor, panel, h, q, lane < q ? x[col0 + lane] : F(0), lane);
            if (lane < q) x[col0 + lane] = b;
        }
        return;
    }
    for (u32 c0 = t; c0 < q; c0 += nt) xs[c0] = x[col0 + c0];
    BARRIER();
    bwd_trsv_steps(head, factor, inv, xs, 0, q, q, t);
    for (u32 c1 = t; c1 < q; c1 += nt) x[col0 + c1] = xs[c1];
}

// The backward trsv, one block per supernode of `list` (bwd_trsv_one); its
// registers uncapped, or trsv_bwd_warp's row spills in f64.
KERNEL_THREADS_REGS(THREADS) void k_bwd_trsv(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT list, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL const F* RESTRICT inv, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    DYNAMIC_SHARED(F, smem);
    bwd_trsv_one(list[block_x()], plan, factor, inv, x, smem, thread_x());
}

#if TRSV_BLOCK % 32 != 0 || TRSV_BLOCK == 0
#error "supernodal.cpp: TRSV_BLOCK is a multiple of 32"
#endif

// The forward trsv of a wide supernode over many blocks, a launch per
// block j of TRSV_BLOCK columns (GpuSupernodalParams::trsv_block): the
// launch's items, 4 words each (the supernode, j, and rows [a, b)), a
// block each. a == b: block j takes block j - 1 off its rows and solves
// itself (fwd_trsv_steps). Otherwise rows [a, b) past block j, a row a
// thread, take block j - 1 off in its 32-column steps (trsv_row_step). A
// launch writes each row in one item; the launches order the blocks.
KERNEL_THREADS(THREADS) void k_fwd_trsv_wide(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL const F* RESTRICT inv, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    SHARED F xs[2 * TRSV_BLOCK];
    GLOBAL const u32* it = items + TRSV_ITEM * block_x();
    u32 s = it[0];
    u32 j = it[1];
    u32 a = it[2];
    u32 b = it[3];
    GLOBAL const u32* head = plan + plan[H_SOLVE_HEAD] + SOLVE_HEAD * s;
    u32 q = head[0];
    u32 h = head[1];
    u32 col0 = head[2];
    u32 panel = head[3];
    u32 t = thread_x();
    const u32 nt = THREADS;
    if (a == b) {
        u32 jlo = j > 0 ? (j - 1) * TRSV_BLOCK : 0;
        u32 js = j * TRSV_BLOCK;
        u32 jend = min(js + TRSV_BLOCK, q);
        for (u32 c = t; c < jend - jlo; c += nt) xs[c] = x[col0 + jlo + c];
        fwd_trsv_steps(head, factor, inv, xs, jlo, js, jend, t);
        BARRIER();
        for (u32 c = js - jlo + t; c < jend - jlo; c += nt) x[col0 + jlo + c] = xs[c];
    } else {
        // block j - 1 is whole: its steps are 32 columns
        u32 y0 = (j - 1) * TRSV_BLOCK;
        for (u32 c = t; c < TRSV_BLOCK; c += nt) xs[c] = x[col0 + y0 + c];
        BARRIER();
        u32 i = a + t;
        if (i < b) {
            F xi = x[col0 + i];
            for (u32 j0 = y0; j0 < y0 + TRSV_BLOCK; j0 += 32) xi -= trsv_row_step(factor, panel, h, j0, 32, i, xs + (j0 - y0));
            x[col0 + i] = xi;
        }
    }
}

// The backward trsv of a wide supernode over many blocks, a launch per
// block of TRSV_BLOCK columns from the last: the launch's items, 4 words
// each (the supernode, the block k, and columns [a, b)), a block each.
// a == b: block k takes block k + 1 (solved) off its columns and solves
// itself (bwd_trsv_steps). Otherwise columns [a, b) before block k take
// block k + 1's rows off, a warp per column, the lanes over the rows. A
// launch writes each column in one item; the launches order the blocks.
KERNEL_THREADS(THREADS) void k_bwd_trsv_wide(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL const F* RESTRICT inv, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    SHARED F xs[2 * TRSV_BLOCK];
    GLOBAL const u32* it = items + TRSV_ITEM * block_x();
    u32 s = it[0];
    u32 k = it[1];
    u32 a = it[2];
    u32 b = it[3];
    GLOBAL const u32* head = plan + plan[H_SOLVE_HEAD] + SOLVE_HEAD * s;
    u32 q = head[0];
    u32 h = head[1];
    u32 col0 = head[2];
    u32 panel = head[3];
    u32 t = thread_x();
    const u32 nt = THREADS;
    if (a == b) {
        u32 cbegin = k * TRSV_BLOCK;
        u32 cend = min(cbegin + TRSV_BLOCK, q);
        u32 rend = min(cbegin + 2 * TRSV_BLOCK, q);
        for (u32 r = t; r < rend - cbegin; r += nt) xs[r] = x[col0 + cbegin + r];
        BARRIER();
        bwd_trsv_steps(head, factor, inv, xs, cbegin, cend, rend, t);
        for (u32 r = t; r < cend - cbegin; r += nt) x[col0 + cbegin + r] = xs[r];
    } else {
        u32 r0 = (k + 1) * TRSV_BLOCK;
        u32 r1 = min(r0 + TRSV_BLOCK, q);
        for (u32 r = t; r < r1 - r0; r += nt) xs[r] = x[col0 + r0 + r];
        BARRIER();
        u32 lane = t % 32;
        for (u32 c = a + t / 32; c < b; c += THREADS / 32) {
            F part = warp_sum(lane_dot(factor, panel + c * h + r0, xs, 0, r1 - r0, lane));
            if (lane == 0) x[col0 + c] -= part;
        }
    }
}

// A chain of the solve's levels, one supernode each, by one block in one
// launch: per level (`levels`, 5 words: the word offset into `plan` of its
// gather rows and their count, its supernode, the word offset of its GEMV
// items and their count), forward its gather rows and its trsv, level
// after level. The levels read what the levels before them wrote, so `x` is
// one plain pointer (not the read-only path); `smem` is the trsvs' dynamic
// shared memory, for the chain's widest supernode; `inv` their inverted
// diagonal tiles.
KERNEL_THREADS_REGS(THREADS) void k_fwd_chain(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT levels, u32 nlevels, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL const F* RESTRICT inv, GLOBAL F* x) {
    if (gate[0] == 0) return;
    DYNAMIC_SHARED(F, smem);
    u32 t = thread_x();
    for (u32 lv = 0; lv < nlevels; lv++) {
        GLOBAL const u32* rec = levels + 5 * lv;
        fwd_gather_rows(plan, plan + rec[0], rec[1], rec[1], factor, x, x, t, THREADS);
        // the gathers in; the previous trsv's reads of `smem` done
        BARRIER();
        fwd_trsv_one(rec[2], plan, factor, inv, x, smem, t);
        // its x, read by the next level's gathers
        BARRIER();
    }
}

// The backward half of k_fwd_chain's chain: its levels from the last, each
// its GEMV items and its transposed trsv.
KERNEL_THREADS_REGS(THREADS) void k_bwd_chain(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT levels, u32 nlevels, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL const F* RESTRICT inv, GLOBAL F* x) {
    if (gate[0] == 0) return;
    SHARED F xr[XR_CH];
    DYNAMIC_SHARED(F, smem);
    u32 t = thread_x();
    u32 lv = nlevels;
    while (lv > 0) {
        lv -= 1;
        GLOBAL const u32* rec = levels + 5 * lv;
        // the level's GEMV items cover its one supernode's columns in
        // order: run as one range, the rows' x gathered once
        u32 ng = rec[4];
        if (ng > 0) {
            GLOBAL const u32* it = plan + rec[3];
            bwd_gemv_cols(it[0], it[1], it[3 * (ng - 1) + 2], plan, factor, x, xr, t);
        }
        // the GEMVs' x in; the previous trsv's reads of `smem` done
        BARRIER();
        bwd_trsv_one(rec[2], plan, factor, inv, x, smem, t);
        // its x, read by the next (lower) level's GEMVs
        BARRIER();
    }
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
