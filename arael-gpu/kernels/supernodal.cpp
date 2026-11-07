// supernodal.cpp: the block supernodal Cholesky and its level-set solve
// (docs/dev/GPU.md, E3 and E5 step 7), launched by
// arael-gpu/src/supernodal.rs over the plan of arael-gpu/src/plan.rs.
//
// Compiled per SCALAR (float or double), the wide scalar W of the vectors
// it solves (wide.h, in front of this) and LU; the tile sizes (NB, TB, TK,
// PAIR_*, XR_CH) and the plan header's slots (H_*) come from plan.rs
// as #defines, THREADS (the block size of every kernel but k_trsm, which
// runs TB) from supernodal.rs. One launch per level unless said.
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

// The seed from f64 values.
KERNEL_THREADS(SEED_LANES) void k_seed_f64(GLOBAL const u32* RESTRICT tiles, GLOBAL const double* RESTRICT vals, GLOBAL F* RESTRICT factor, u32 ntiles) {
    u32 t = block_x();
    if (t < ntiles) seed_tile_lanes<double>(tiles, vals, factor, t, thread_x());
}

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
DEVICE_INLINE void shift_down(F (&a)[NB]) {
#pragma unroll
    for (u32 c = 1; c < W; c++) a[c - 1] = a[c];
}

// potf2_regs' pivots k while the columns left (nb - k) fit width W.
template <u32 W, typename Out>
DEVICE_INLINE void potf2_steps(F (&a)[NB], u32 lane, u32 nb, u32& k, F& dr, bool& bad, Out& out) {
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

// A warp's 32 rows of an nb-column panel factored in place in registers:
// lane r holds row r from the current column on (a[0]), reading the other
// rows' entries by shuffles, so no barrier and no shared memory. The top
// nb x nb block becomes its Cholesky factor, the rows below it are solved
// against it (L21 = A21 L11^-T), right-looking. Column k of L is handed
// over as out.put(k, value) on every lane (lane r: L[r][k], meaningful for
// r >= k); `dr` gets 1 / L[r][r] on the lanes below nb. Returns whether a
// pivot was not positive; it continues with 1.
template <typename Out>
DEVICE_INLINE bool potf2_regs(F (&a)[NB], u32 lane, u32 nb, F& dr, Out& out) {
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
// supernode s factored in place by one warp (potf2_regs). A pivot that is
// not positive sets fail[0] to s plus one and continues with 1.
DEVICE_INLINE void potf2_warp(GLOBAL F* factor, u32 tile, u32 h, u32 nb, u32 lane, GLOBAL u32* fail, u32 s) {
    F a[NB];
#pragma unroll
    for (u32 c = 0; c < NB; c++) a[c] = (lane < nb && c <= lane) ? factor[tile + c * h + lane] : F(0);
    F dr = F(1);
    ColsOut out = {factor, tile, h, lane, nb, nb};
    bool bad = potf2_regs(a, lane, nb, dr, out);
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
constexpr u32 SMALL_KU = sizeof(F) == 4 ? 8 : 4;

// small_take's products, the loop over the row's nb columns W wide; L[r][k]
// comes through a ring of SMALL_KU loads ahead (`lk`, loaded).
template <u32 W>
DEVICE_INLINE void small_take_cols(F (&v)[NB], F (&lk)[SMALL_KU], GLOBAL const F* factor, u32 panel, u32 h, SHARED_PTR const F* l, u32 q, u32 j, u32 nb, u32 r) {
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
DEVICE_INLINE void small_take(F (&v)[NB], GLOBAL const F* factor, u32 panel, u32 h, SHARED_PTR const F* l, u32 q, u32 j, u32 nb, u32 r) {
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
DEVICE_INLINE void small_solve_steps(F (&v)[NB], GLOBAL F* factor, u32 panel, u32 h, SHARED_PTR F* l, SHARED_PTR const F* l11, SHARED_PTR const F* d11, u32 q, u32 j, u32 nb, u32 r, u32& k) {
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
DEVICE_INLINE void small_solve(F (&v)[NB], GLOBAL F* factor, u32 panel, u32 h, SHARED_PTR F* l, SHARED_PTR const F* l11, SHARED_PTR const F* d11, u32 q, u32 j, u32 nb, u32 r) {
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
            bad = potf2_regs(v, lane, nb, dr, out) || bad;
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
    u32 o_dims = plan[H_DIMS];
    u32 o_val_ptr = plan[H_VAL_PTR];
    u32 s = panels[block_x()];
    u32 q = plan[o_dims + 2 * s];
    u32 h = plan[o_dims + 2 * s + 1];
    potf2_warp(factor, plan[o_val_ptr + s], h, min(q, NB), thread_x(), fail, s);
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

// One slab of nk staged columns into a thread's accumulators: rows
// tx + 16 r of sa against columns ty + 16 c of sb, acc[4 r + c]. On a
// diagonal tile (DIAG) the accumulators with r < c lie wholly above the
// diagonal, which the tile does not write, and are skipped.
template <bool DIAG> DEVICE_INLINE void syrk_slab(F (&acc)[16], SHARED_PTR const F* sa, SHARED_PTR const F* sb, u32 nk, u32 tx, u32 ty) {
    for (u32 kk = 0; kk < nk; kk++) {
        F a[4];
        F b[4];
#pragma unroll
        for (u32 r = 0; r < 4; r++) a[r] = sa[kk * TB + tx + 16 * r];
#pragma unroll
        for (u32 c = 0; c < 4; c++) b[c] = sb[kk * TB + ty + 16 * c];
#pragma unroll
        for (u32 r = 0; r < 4; r++) {
#pragma unroll
            for (u32 c = 0; c < 4; c++) {
                if (!DIAG || r >= c) acc[r * 4 + c] += a[r] * b[c];
            }
        }
    }
}

// Panel step, part three: the update of the trailing lower trapezoid by
// the solved columns [k0, k1), k1 = min(k0 + kn, q), over rows [k1, h) and
// columns [k1, c1), c1 = min(cend, q), in TB x TB tiles (plan::syrk_update
// picks k0, kn and cend per step for blocks of SYRK_BLOCK columns: a step
// inside a block updates the rest of its block, the block's last step
// everything past it); tiles above the diagonal are skipped and the
// straddling tile clipped. The columns stream through shared memory NB at
// a time into the tile's accumulators, so a tile is read and written once
// per update; the loop's trip count is constant, so with SYRK_BLOCK = NB
// it is a single pass. A thread takes the rows tx + 16 r and columns
// ty + 16 c, so neighbouring threads read and write neighbouring rows. The
// block index walks the panels' tile counts. The block of a panel's tile
// (0, 0), which alone writes the next diagonal tile (at k1), then factors
// it (potf2_warp).
KERNEL_THREADS(THREADS) void k_syrk(GLOBAL const u32* RESTRICT panels, u32 npanels, u32 k0, u32 kn, u32 cend, GLOBAL const u32* RESTRICT plan, GLOBAL F* RESTRICT factor, GLOBAL u32* RESTRICT fail) {
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
    u32 row0 = k1 + ti * TB;
    u32 col0 = k1 + tj * TB;
    u32 nrows = min(h - row0, TB);
    u32 ncols = min(c1 - col0, TB);
    u32 panel = plan[o_val_ptr + s];
    u32 t = thread_x();
    const u32 nt = THREADS;
    u32 tx = t % 16;
    u32 ty = t / 16;
    F acc[16];
#pragma unroll
    for (u32 z = 0; z < 16; z++) acc[z] = F(0);
    for (u32 sl = 0; sl < SYRK_BLOCK / NB; sl++) {
        u32 kb = k0 + sl * NB;
        if (kb >= k1) break;
        u32 nk = min(k1 - kb, NB);
        // the previous slab's reads are done
        if (sl > 0) BARRIER();
        for (u32 e = t; e < NB * TB; e += nt) {
            u32 k = e / TB;
            u32 i = e - k * TB;
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
            syrk_slab<true>(acc, sa, sb, nk, tx, ty);
        } else {
            syrk_slab<false>(acc, sa, sb, nk, tx, ty);
        }
    }
#pragma unroll
    for (u32 r = 0; r < 4; r++) {
#pragma unroll
        for (u32 c = 0; c < 4; c++) {
            u32 i = tx + 16 * r;
            u32 jj = ty + 16 * c;
            if (i < nrows && jj < ncols && row0 + i >= col0 + jj) factor[panel + (col0 + jj) * h + row0 + i] -= acc[r * 4 + c];
        }
    }
    if (ti == 0 && tj == 0) {
        // the block's writes seen by its first warp
        BARRIER();
        if (t < NB) potf2_warp(factor, panel + k1 * h + k1, h, min(q - k1, NB), t, fail, s);
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

// The 32 x 32 lower tile at (j0, j0) into row-major `ls` (coalesced along
// the column), and the reciprocal of its diagonal for this thread's lane.
// 1024 / THREADS elements per thread.
DEVICE_INLINE F load_tile32(GLOBAL const F* factor, u32 panel, u32 h, u32 j0, u32 nb, SHARED_PTR F* ls) {
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

// The forward steps k of trsv_fwd_warp while the columns left (q - k) fit
// width W: lane k's y_k = b_k / L_kk, the lanes below less L_rk y_k.
template <u32 W>
DEVICE_INLINE void trsv_fwd_steps(F (&a)[NB], F& b, F rd, u32 lane, u32 q, u32& k) {
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
DEVICE_INLINE void trsv_bwd_steps(F (&a)[NB], F& b, F rd, u32 lane, u32 q, u32& k) {
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

// Forward solve, part two: the trsv of supernode s by the block of THREADS
// threads; a supernode of at most 32 columns by the first warp in
// registers (trsv_fwd_warp). Wider, in 32-column blocks: the diagonal
// tile's 32 sequential steps by the first warp on shared data, the rows
// below updated by all threads with the full block's 32 loads issued
// before its FMAs. `smem`: 1024 + q values (dynamic shared memory for the
// widest supernode q of the launch).
DEVICE_INLINE void fwd_trsv_one(u32 s, GLOBAL const u32* plan, GLOBAL const F* factor, GLOBAL F* x, SHARED_PTR F* smem) {
    SHARED_PTR F* ls = smem;
    SHARED_PTR F* xs = smem + 1024;
    GLOBAL const u32* head = plan + plan[H_SOLVE_HEAD] + SOLVE_HEAD * s;
    u32 q = head[0];
    u32 h = head[1];
    u32 col0 = head[2];
    u32 panel = head[3];
    u32 t = thread_x();
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

// The forward trsv, one block per supernode of `list` (fwd_trsv_one).
KERNEL_THREADS(THREADS) void k_fwd_trsv(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT list, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    DYNAMIC_SHARED(F, smem);
    fwd_trsv_one(list[block_x()], plan, factor, x, smem);
}

// Backward solve, part one: x_s -= L_below^T x for the columns [c0, c1) of
// supernode s by the block of THREADS threads; the rows' x gathered into
// `xr` XR_CH at a time, a warp per column with the lanes over the rows. A
// column's sums do not depend on the range it is in.
DEVICE_INLINE void bwd_gemv_cols(u32 s, u32 c0, u32 c1, GLOBAL const u32* plan, GLOBAL const F* factor, GLOBAL F* x, SHARED_PTR F* xr) {
    u32 o_row_sc = plan[H_ROW_SC];
    GLOBAL const u32* head = plan + plan[H_SOLVE_HEAD] + SOLVE_HEAD * s;
    u32 q = head[0];
    u32 h = head[1];
    u32 col0 = head[2];
    u32 panel = head[3];
    u32 rsc = head[4];
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

// The GEMV item at `it`, 3 words: its supernode and columns [c0, c1).
DEVICE_INLINE void bwd_gemv_item(GLOBAL const u32* it, GLOBAL const u32* plan, GLOBAL const F* factor, GLOBAL F* x, SHARED_PTR F* xr) {
    bwd_gemv_cols(it[0], it[1], it[2], plan, factor, x, xr);
}

// The backward GEMV, one block per item (bwd_gemv_item).
KERNEL_THREADS(THREADS) void k_bwd_gemv(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT items, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    SHARED F xr[XR_CH];
    bwd_gemv_item(items + 3 * block_x(), plan, factor, x, xr);
}

// Backward solve, part two: the transposed trsv of supernode s by the block
// of THREADS threads; a supernode of at most 32 columns by the first warp
// in registers (trsv_bwd_warp). Wider, 32-column blocks from the last, the
// block's columns against the rows below it (a warp per column), then the
// diagonal tile's steps by the first warp on shared data. `smem` as
// fwd_trsv_one's.
DEVICE_INLINE void bwd_trsv_one(u32 s, GLOBAL const u32* plan, GLOBAL const F* factor, GLOBAL F* x, SHARED_PTR F* smem) {
    SHARED_PTR F* ls = smem;
    SHARED_PTR F* xs = smem + 1024;
    GLOBAL const u32* head = plan + plan[H_SOLVE_HEAD] + SOLVE_HEAD * s;
    u32 q = head[0];
    u32 h = head[1];
    u32 col0 = head[2];
    u32 panel = head[3];
    u32 t = thread_x();
    const u32 nt = THREADS;
    u32 lane = t % 32;
    u32 warp = t / 32;
    u32 nwarps = nt / 32;
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

// The backward trsv, one block per supernode of `list` (bwd_trsv_one).
KERNEL_THREADS(THREADS) void k_bwd_trsv(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT list, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* RESTRICT x) {
    if (gate[0] == 0) return;
    DYNAMIC_SHARED(F, smem);
    bwd_trsv_one(list[block_x()], plan, factor, x, smem);
}

// A chain of the solve's levels, one supernode each, by one block in one
// launch: per level (`levels`, 5 words: the word offset into `plan` of its
// gather rows and their count, its supernode, the word offset of its GEMV
// items and their count), forward its gather rows and its trsv, level
// after level. The levels read what the levels before them wrote, so `x` is
// one plain pointer (not the read-only path); `smem` is the trsvs' dynamic
// shared memory, for the chain's widest supernode.
KERNEL_THREADS_REGS(THREADS) void k_fwd_chain(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT levels, u32 nlevels, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* x) {
    if (gate[0] == 0) return;
    DYNAMIC_SHARED(F, smem);
    u32 t = thread_x();
    for (u32 lv = 0; lv < nlevels; lv++) {
        GLOBAL const u32* rec = levels + 5 * lv;
        fwd_gather_rows(plan, plan + rec[0], rec[1], rec[1], factor, x, x, t, THREADS);
        // the gathers in; the previous trsv's reads of `smem` done
        BARRIER();
        fwd_trsv_one(rec[2], plan, factor, x, smem);
        // its x, read by the next level's gathers
        BARRIER();
    }
}

// The backward half of k_fwd_chain's chain: its levels from the last, each
// its GEMV items and its transposed trsv.
KERNEL_THREADS_REGS(THREADS) void k_bwd_chain(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT levels, u32 nlevels, GLOBAL const u32* RESTRICT plan, GLOBAL const F* RESTRICT factor, GLOBAL F* x) {
    if (gate[0] == 0) return;
    SHARED F xr[XR_CH];
    DYNAMIC_SHARED(F, smem);
    u32 lv = nlevels;
    while (lv > 0) {
        lv -= 1;
        GLOBAL const u32* rec = levels + 5 * lv;
        // the level's GEMV items cover its one supernode's columns in
        // order: run as one range, the rows' x gathered once
        u32 ng = rec[4];
        if (ng > 0) {
            GLOBAL const u32* it = plan + rec[3];
            bwd_gemv_cols(it[0], it[1], it[3 * (ng - 1) + 2], plan, factor, x, xr);
        }
        // the GEMVs' x in; the previous trsv's reads of `smem` done
        BARRIER();
        bwd_trsv_one(rec[2], plan, factor, x, smem);
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
