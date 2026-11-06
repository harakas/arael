// schur.cpp: the Schur reduction on the device and the refinement of its
// reduced solve (docs/dev/GPU.md, E2 and E9), launched by
// arael-gpu/src/schur.rs over its plan.
//
// In W, the route's wide scalar (wide.h). Compiled per EW (the widest eliminated block, rounded up to 4,
// 8, 16, 32 or 64), KW (the widest kept block, rounded likewise),
// TILE_LANES (k_schur_tiles' threads per tile) and ACC (KW * KW /
// TILE_LANES rounded up, the entries of a tile of S per lane); the plan
// header's slots (SE_*, SO_*, ST_*, SP_*, SK_*), the refinement state's
// words (R_*) and flags (F_*) come from schur.rs as #defines. Kernels
// launched with THREADS threads per block, or PLANE where said.
//
// Per-thread arrays are indexed only by constants once their loops unroll:
// the loops run to EW or KW and skip what is past the block's width, so the
// arrays stay in registers (arael_gpu.h's rule, from k_trsm).

// STAGED (lib.rs, `staged`; not on CUDA until measured there, TODO.md):
// k_schur_w and k_schur_tiles read their small blocks once, into
// registers and shared memory, instead of from global memory at every use;
// k_schur_w takes a thread per row of W, k_schur_tiles a wavefront per
// tile (TILE_LANES threads, 64 on AMD GCN and CDNA).

// One plane: the threads of the tile, rhs and symv kernels.
#define PLANE 32u
#if PLANE != WARP
#error "schur.cpp: the rhs and symv kernels sum over a plane with warp_sum"
#endif
// No kept-kept tile of H behind a tile of S.
#define NONE 0xffffffffu

KERNEL_THREADS(PLANE) void k_zero_u32(GLOBAL u32* RESTRICT buf, u32 n) {
    u32 i = global_x();
    if (i < n) buf[i] = 0;
}

// C[i][k] of an observer's coupling tile at `ca`, wo x we, in either storage
// orientation.
DEVICE W coupling(GLOBAL const W* h, u32 ca, u32 trans, u32 wo, u32 we, u32 i, u32 k) {
    return trans != 0 ? h[ca + k + i * we] : h[ca + i + k * wo];
}

// Stage 1, one thread per eliminated block: the inverse of its damped
// diagonal tile (upper-stored) through the Cholesky factor, column-major
// we x we into `dinv`, and z = D^-1 b. A pivot that is not positive sets
// fail[0] to the block's slot plus one and continues with 1.
KERNEL_THREADS_REGS(THREADS) void k_schur_elim(GLOBAL const u32* RESTRICT plan, GLOBAL const W* RESTRICT h, GLOBAL const W* RESTRICT rhs, GLOBAL W* RESTRICT dinv, GLOBAL W* RESTRICT z, GLOBAL u32* RESTRICT fail, u32 ne) {
    u32 e = global_x();
    if (e >= ne) return;
    u32 w = plan[plan[SE_W] + e];
    u32 diag = plan[plan[SE_DIAG_OFF] + e];
    u32 orig = plan[plan[SE_ORIG_OFF] + e];
    u32 doff = plan[plan[SE_DINV_OFF] + e];
    u32 zoff = plan[plan[SE_Z_OFF] + e];
    // the tile, symmetric from its upper triangle, then its lower factor in place
    W a[EW * EW];
#pragma unroll
    for (u32 j = 0; j < EW; j++)
#pragma unroll
        for (u32 i = 0; i < EW; i++)
            if (i < w && j < w) a[i + j * EW] = i > j ? h[diag + j + i * w] : h[diag + i + j * w];
#pragma unroll
    for (u32 j = 0; j < EW; j++) {
        if (j < w) {
            W d = a[j + j * EW];
#pragma unroll
            for (u32 k = 0; k < j; k++) d -= a[j + k * EW] * a[j + k * EW];
            if (!(d > 0.0f)) {
                fail[0] = e + 1;
                d = 1.0f;
            }
            W ljj = sqrt(d);
            a[j + j * EW] = ljj;
#pragma unroll
            for (u32 i = j + 1; i < EW; i++) {
                if (i < w) {
                    W v = a[i + j * EW];
#pragma unroll
                    for (u32 k = 0; k < j; k++) v -= a[i + k * EW] * a[j + k * EW];
                    a[i + j * EW] = v / ljj;
                }
            }
        }
    }
    // L^-1 (lower) by forward substitution, column by column
    W li[EW * EW];
#pragma unroll
    for (u32 c = 0; c < EW; c++) {
#pragma unroll
        for (u32 r = 0; r < EW; r++) {
            if (c < w && r < w) {
                if (r >= c) {
                    W v = r == c ? 1.0f : 0.0f;
#pragma unroll
                    for (u32 k = c; k < r; k++) v -= a[r + k * EW] * li[k + c * EW];
                    li[r + c * EW] = v / a[r + r * EW];
                } else {
                    li[r + c * EW] = 0.0f;
                }
            }
        }
    }
    // D^-1 = L^-T L^-1, kept in `a` for z
#pragma unroll
    for (u32 j = 0; j < EW; j++) {
#pragma unroll
        for (u32 i = 0; i < EW; i++) {
            if (i < w && j < w) {
                W v = 0.0f;
#pragma unroll
                for (u32 k = 0; k < EW; k++)
                    if (k < w) v = wide_fma(v, li[k + i * EW], li[k + j * EW]);
                a[i + j * EW] = v;
                dinv[doff + i + j * w] = v;
            }
        }
    }
#pragma unroll
    for (u32 i = 0; i < EW; i++) {
        if (i < w) {
            W v = 0.0f;
#pragma unroll
            for (u32 j = 0; j < EW; j++)
                if (j < w) v = wide_fma(v, a[i + j * EW], rhs[orig + j]);
            z[zoff + i] = v;
        }
    }
}

#if STAGED
// Stage 2, KW threads per observer, one per row of W_o (those past wo
// idle): row i of W_o = C_o D_e^-1, row-major wo x we, the row of C_o read
// once into registers. Neighbouring threads read and write neighbouring
// rows.
KERNEL_THREADS(THREADS) void k_schur_w(GLOBAL const u32* RESTRICT plan, GLOBAL const W* RESTRICT h, GLOBAL const W* RESTRICT dinv, GLOBAL W* RESTRICT w, u32 no) {
    u32 x = global_x();
    u32 o = x / KW;
    u32 i = x - o * KW;
    if (o >= no) return;
    u32 wo = plan[plan[SO_W] + o];
    if (i >= wo) return;
    u32 e = plan[plan[SO_E] + o];
    u32 we = plan[plan[SE_W] + e];
    u32 ca = plan[plan[SO_CA_OFF] + o];
    u32 trans = plan[plan[SO_TRANS] + o];
    u32 doff = plan[plan[SE_DINV_OFF] + e];
    u32 woff = plan[plan[SO_W_OFF] + o];
    W c[EW];
#pragma unroll
    for (u32 k = 0; k < EW; k++) c[k] = k < we ? coupling(h, ca, trans, wo, we, i, k) : W(0.0f);
#pragma unroll
    for (u32 l = 0; l < EW; l++) {
        if (l < we) {
            W v = 0.0f;
#pragma unroll
            for (u32 k = 0; k < EW; k++)
                if (k < we) v = wide_fma(v, c[k], dinv[doff + k + l * we]);
            w[woff + i * we + l] = v;
        }
    }
}
#else
// Stage 2, one thread per observer: W_o = C_o D_e^-1, row-major wo x we.
KERNEL_THREADS(THREADS) void k_schur_w(GLOBAL const u32* RESTRICT plan, GLOBAL const W* RESTRICT h, GLOBAL const W* RESTRICT dinv, GLOBAL W* RESTRICT w, u32 no) {
    u32 o = global_x();
    if (o >= no) return;
    u32 wo = plan[plan[SO_W] + o];
    u32 e = plan[plan[SO_E] + o];
    u32 we = plan[plan[SE_W] + e];
    u32 ca = plan[plan[SO_CA_OFF] + o];
    u32 trans = plan[plan[SO_TRANS] + o];
    u32 doff = plan[plan[SE_DINV_OFF] + e];
    u32 woff = plan[plan[SO_W_OFF] + o];
    for (u32 i = 0; i < wo; i++) {
        for (u32 l = 0; l < we; l++) {
            W v = 0.0f;
            for (u32 k = 0; k < we; k++) v = wide_fma(v, coupling(h, ca, trans, wo, we, i, k), dinv[doff + k + l * we]);
            w[woff + i * we + l] = v;
        }
    }
}
#endif

#if STAGED
// Stage 3, one block of TILE_LANES threads per tile of S (the wavefront):
// the kept-kept tile of H (or nothing) minus W_oa C_ob^T over the tile's
// pairs, thread `lane` owning the entries lane, lane + TILE_LANES, ... (ACC
// of them at most); a diagonal tile upper-only. A pair's W_oa (wr x we,
// row-major) and C_ob (wc x we, row j at j EW) are read once into shared
// memory. The tile is the block's, so its plan words are the same in every
// lane (scalar loads on AMD).
KERNEL_THREADS(TILE_LANES) void k_schur_tiles(GLOBAL const u32* RESTRICT plan, GLOBAL const W* RESTRICT h, GLOBAL const W* RESTRICT w, GLOBAL W* RESTRICT s) {
    SHARED W ws[KW * EW];
    SHARED W cs[KW * EW];
    u32 t = block_x();
    u32 lane = thread_x();
    u32 o_ew = plan[SE_W];
    u32 o_ca = plan[SO_CA_OFF];
    u32 o_trans = plan[SO_TRANS];
    u32 o_ow = plan[SO_W];
    u32 o_e = plan[SO_E];
    u32 o_woff = plan[SO_W_OFF];
    u32 o_oa = plan[SP_OA];
    u32 o_ob = plan[SP_OB];
    u32 val = plan[plan[ST_VAL_OFF] + t];
    u32 wr = plan[plan[ST_WR] + t];
    u32 wc = plan[plan[ST_WC] + t];
    u32 a = plan[plan[ST_A] + t];
    u32 b = plan[plan[ST_B] + t];
    u32 copy = plan[plan[ST_COPY_OFF] + t];
    u32 p0 = plan[plan[ST_PAIR_PTR] + t];
    u32 p1 = plan[plan[ST_PAIR_PTR] + t + 1];
    W acc[ACC];
#pragma unroll
    for (u32 m = 0; m < ACC; m++) acc[m] = 0.0f;
    for (u32 p = p0; p < p1; p++) {
        u32 oa = plan[o_oa + p];
        u32 ob = plan[o_ob + p];
        u32 e = plan[o_e + oa];
        u32 we = plan[o_ew + e];
        u32 woff = plan[o_woff + oa];
        u32 cb = plan[o_ca + ob];
        u32 tb = plan[o_trans + ob];
        u32 wb = plan[o_ow + ob];
        // the previous pair's reads are done (the block is one wavefront)
        BARRIER();
        for (u32 x = lane; x < wr * we; x += TILE_LANES) ws[x] = w[woff + x];
        for (u32 x = lane; x < wc * we; x += TILE_LANES) {
            u32 j = x / we;
            u32 k = x - j * we;
            cs[j * EW + k] = coupling(h, cb, tb, wb, we, j, k);
        }
        BARRIER();
#pragma unroll
        for (u32 m = 0; m < ACC; m++) {
            u32 ent = lane + TILE_LANES * m;
            if (ent < wr * wc) {
                u32 j = ent / wr;
                u32 i = ent - j * wr;
                W v = 0.0f;
                for (u32 k = 0; k < we; k++) v = wide_fma(v, ws[i * we + k], cs[j * EW + k]);
                acc[m] -= v;
            }
        }
    }
#pragma unroll
    for (u32 m = 0; m < ACC; m++) {
        u32 ent = lane + TILE_LANES * m;
        if (ent < wr * wc) {
            u32 j = ent / wr;
            u32 i = ent - j * wr;
            W v = acc[m];
            if (copy != NONE) v += h[copy + i + j * wr];
            if (a == b && i > j) v = 0.0f;
            s[val + i + j * wr] = v;
        }
    }
}
#else
// Stage 3, one block of PLANE threads per tile of S: the kept-kept tile of H
// (or nothing) minus W_oa C_ob^T over the tile's pairs, thread `lane` owning
// the entries lane, lane + 32, ... (ACC of them at most); a diagonal tile
// upper-only.
KERNEL_THREADS(PLANE) void k_schur_tiles(GLOBAL const u32* RESTRICT plan, GLOBAL const W* RESTRICT h, GLOBAL const W* RESTRICT w, GLOBAL W* RESTRICT s) {
    u32 o_ew = plan[SE_W];
    u32 o_ca = plan[SO_CA_OFF];
    u32 o_trans = plan[SO_TRANS];
    u32 o_ow = plan[SO_W];
    u32 o_e = plan[SO_E];
    u32 o_woff = plan[SO_W_OFF];
    u32 o_oa = plan[SP_OA];
    u32 o_ob = plan[SP_OB];
    u32 t = block_x();
    u32 lane = thread_x();
    u32 val = plan[plan[ST_VAL_OFF] + t];
    u32 wr = plan[plan[ST_WR] + t];
    u32 wc = plan[plan[ST_WC] + t];
    u32 a = plan[plan[ST_A] + t];
    u32 b = plan[plan[ST_B] + t];
    u32 copy = plan[plan[ST_COPY_OFF] + t];
    u32 p0 = plan[plan[ST_PAIR_PTR] + t];
    u32 p1 = plan[plan[ST_PAIR_PTR] + t + 1];
    W acc[ACC];
#pragma unroll
    for (u32 m = 0; m < ACC; m++) acc[m] = 0.0f;
    for (u32 p = p0; p < p1; p++) {
        u32 oa = plan[o_oa + p];
        u32 ob = plan[o_ob + p];
        u32 e = plan[o_e + oa];
        u32 we = plan[o_ew + e];
        u32 woff = plan[o_woff + oa];
        u32 cb = plan[o_ca + ob];
        u32 tb = plan[o_trans + ob];
        u32 wb = plan[o_ow + ob];
#pragma unroll
        for (u32 m = 0; m < ACC; m++) {
            u32 ent = lane + PLANE * m;
            if (ent < wr * wc) {
                u32 j = ent / wr;
                u32 i = ent - j * wr;
                W v = 0.0f;
                for (u32 k = 0; k < we; k++) v = wide_fma(v, w[woff + i * we + k], coupling(h, cb, tb, wb, we, j, k));
                acc[m] -= v;
            }
        }
    }
#pragma unroll
    for (u32 m = 0; m < ACC; m++) {
        u32 ent = lane + PLANE * m;
        if (ent < wr * wc) {
            u32 j = ent / wr;
            u32 i = ent - j * wr;
            W v = acc[m];
            if (copy != NONE) v += h[copy + i + j * wr];
            if (a == b && i > j) v = 0.0f;
            s[val + i + j * wr] = v;
        }
    }
}
#endif

// Stage 4, one block of PLANE threads per kept block: rhs_kept = b_k - sum
// over the block's observers of C_o z_e, the lanes over the observers, a
// warp sum per row.
KERNEL_THREADS(PLANE) void k_schur_rhs(GLOBAL const u32* RESTRICT plan, GLOBAL const W* RESTRICT h, GLOBAL const W* RESTRICT rhs, GLOBAL const W* RESTRICT z, GLOBAL W* RESTRICT out) {
    u32 o_ew = plan[SE_W];
    u32 o_z = plan[SE_Z_OFF];
    u32 o_ca = plan[SO_CA_OFF];
    u32 o_trans = plan[SO_TRANS];
    u32 o_e = plan[SO_E];
    u32 o_kobs = plan[SK_OBS];
    u32 k = block_x();
    u32 lane = thread_x();
    u32 s_off = plan[plan[SK_S_OFF] + k];
    u32 orig = plan[plan[SK_ORIG_OFF] + k];
    u32 wk = plan[plan[SK_W] + k];
    u32 q0 = plan[plan[SK_OBS_PTR] + k];
    u32 q1 = plan[plan[SK_OBS_PTR] + k + 1];
    W acc[KW];
#pragma unroll
    for (u32 i = 0; i < KW; i++) acc[i] = 0.0f;
    for (u32 q = q0 + lane; q < q1; q += PLANE) {
        u32 o = plan[o_kobs + q];
        u32 e = plan[o_e + o];
        u32 we = plan[o_ew + e];
        u32 ca = plan[o_ca + o];
        u32 trans = plan[o_trans + o];
        u32 zoff = plan[o_z + e];
#pragma unroll
        for (u32 i = 0; i < KW; i++) {
            if (i < wk) {
                W v = 0.0f;
                for (u32 kk = 0; kk < we; kk++) v = wide_fma(v, coupling(h, ca, trans, wk, we, i, kk), z[zoff + kk]);
                acc[i] += v;
            }
        }
    }
    // wk is the block's, so the whole warp takes each sum
#pragma unroll
    for (u32 i = 0; i < KW; i++) {
        if (i < wk) {
            W tot = warp_sum(acc[i]);
            if (lane == 0) out[s_off + i] = rhs[orig + i] - tot;
        }
    }
}

// The kept blocks of the reduced solution into the full vector, one thread
// per kept block.
KERNEL_THREADS(THREADS) void k_scatter_kept(GLOBAL const u32* RESTRICT plan, GLOBAL const W* RESTRICT xk, GLOBAL W* RESTRICT x, u32 nk) {
    u32 k = global_x();
    if (k >= nk) return;
    u32 s_off = plan[plan[SK_S_OFF] + k];
    u32 orig = plan[plan[SK_ORIG_OFF] + k];
    u32 wk = plan[plan[SK_W] + k];
    for (u32 i = 0; i < wk; i++) x[orig + i] = xk[s_off + i];
}

// Back-substitution, one thread per eliminated block: x_e = D_e^-1 (b_e -
// sum_a C_a^T x_a) over its observers, the kept slices of `x` in place.
KERNEL_THREADS(THREADS) void k_schur_backsub(GLOBAL const u32* RESTRICT plan, GLOBAL const W* RESTRICT h, GLOBAL const W* RESTRICT rhs, GLOBAL const W* RESTRICT dinv, GLOBAL W* RESTRICT x, u32 ne) {
    u32 e = global_x();
    if (e >= ne) return;
    u32 o_ca = plan[SO_CA_OFF];
    u32 o_trans = plan[SO_TRANS];
    u32 o_ow = plan[SO_W];
    u32 o_oorig = plan[SO_ORIG_OFF];
    u32 w = plan[plan[SE_W] + e];
    u32 orig = plan[plan[SE_ORIG_OFF] + e];
    u32 doff = plan[plan[SE_DINV_OFF] + e];
    u32 q0 = plan[plan[SE_OBS_PTR] + e];
    u32 q1 = plan[plan[SE_OBS_PTR] + e + 1];
    W t[EW];
#pragma unroll
    for (u32 i = 0; i < EW; i++)
        if (i < w) t[i] = rhs[orig + i];
    for (u32 o = q0; o < q1; o++) {
        u32 wa = plan[o_ow + o];
        u32 ca = plan[o_ca + o];
        u32 trans = plan[o_trans + o];
        u32 xo = plan[o_oorig + o];
#pragma unroll
        for (u32 kk = 0; kk < EW; kk++) {
            if (kk < w) {
                W v = 0.0f;
                for (u32 i = 0; i < wa; i++) v = wide_fma(v, coupling(h, ca, trans, wa, w, i, kk), x[xo + i]);
                t[kk] -= v;
            }
        }
    }
#pragma unroll
    for (u32 i = 0; i < EW; i++) {
        if (i < w) {
            W v = 0.0f;
#pragma unroll
            for (u32 j = 0; j < EW; j++)
                if (j < w) v = wide_fma(v, dinv[doff + i + j * w], t[j]);
            x[orig + i] = v;
        }
    }
}

// r -= S x over the tiles of S (upper block storage, diagonal tiles
// upper-only), one block of PLANE threads per kept block k owning rows k of
// r: the lanes over the tiles touching k (kt[kt_ptr[k]..kt_ptr[k + 1]], each
// a tile index times 2 plus 1 when k is its column block, whose transpose
// then gives the rows), a warp sum per row; nothing unless gate[0] is
// nonzero. No two blocks write the same entry.
KERNEL_THREADS(PLANE) void k_symv_sub(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT plan, GLOBAL const u32* RESTRICT kt_ptr, GLOBAL const u32* RESTRICT kt, GLOBAL const W* RESTRICT s, GLOBAL const W* RESTRICT x, GLOBAL W* RESTRICT r) {
    if (gate[0] == 0) return;
    u32 k = block_x();
    u32 lane = thread_x();
    u32 k_off = plan[plan[SK_S_OFF] + k];
    u32 wk = plan[plan[SK_W] + k];
    W acc[KW];
#pragma unroll
    for (u32 i = 0; i < KW; i++) acc[i] = 0.0f;
    for (u32 q = kt_ptr[k] + lane; q < kt_ptr[k + 1]; q += PLANE) {
        u32 t = kt[q] >> 1;
        u32 transposed = kt[q] & 1u;
        u32 val = plan[plan[ST_VAL_OFF] + t];
        u32 wr = plan[plan[ST_WR] + t];
        u32 wc = plan[plan[ST_WC] + t];
        u32 a = plan[plan[ST_A] + t];
        u32 b = plan[plan[ST_B] + t];
        if (transposed == 0) {
            // rows of (k, b): S_kb x_b, a diagonal tile read from its upper triangle
            u32 b_off = plan[plan[ST_B_OFF] + t];
#pragma unroll
            for (u32 i = 0; i < KW; i++) {
                if (i < wk) {
                    W v = 0.0f;
                    for (u32 j = 0; j < wc; j++) v = wide_fma(v, a == b && i > j ? s[val + j + i * wr] : s[val + i + j * wr], x[b_off + j]);
                    acc[i] += v;
                }
            }
        } else {
            // rows of (a, k)^T: S_ak^T x_a
            u32 a_off = plan[plan[ST_A_OFF] + t];
#pragma unroll
            for (u32 j = 0; j < KW; j++) {
                if (j < wk) {
                    W v = 0.0f;
                    for (u32 i2 = 0; i2 < wr; i2++) v = wide_fma(v, s[val + i2 + j * wr], x[a_off + i2]);
                    acc[j] += v;
                }
            }
        }
    }
    // wk is the block's, so the whole warp takes each sum
#pragma unroll
    for (u32 i = 0; i < KW; i++) {
        if (i < wk) {
            W tot = warp_sum(acc[i]);
            if (lane == 0) r[k_off + i] -= tot;
        }
    }
}

// dst = src when gate[0] is nonzero.
KERNEL_THREADS(THREADS) void k_copy_f64(GLOBAL const u32* RESTRICT gate, GLOBAL W* RESTRICT dst, GLOBAL const W* RESTRICT src, u32 n) {
    u32 i = global_x();
    if (gate[0] != 0 && i < n) dst[i] = src[i];
}

// The sum of squares of x[0..n) by one block of THREADS threads, returned to
// every thread; `sh` is THREADS words of shared memory.
DEVICE W block_sumsq(GLOBAL const W* x, u32 n, SHARED_PTR W* sh) {
    u32 t = thread_x();
    W acc = 0.0f;
    for (u32 i = t; i < n; i += THREADS) acc = wide_fma(acc, x[i], x[i]);
    // a previous call's readers are done with the shared words first
    BARRIER();
    sh[t] = acc;
    BARRIER();
    for (u32 half = THREADS / 2; half > 0; half /= 2) {
        if (t < half) sh[t] += sh[t + half];
        BARRIER();
    }
    return sh[0];
}

// The refinement's start, one block, after the first solve's residual r of
// the right-hand side b: the norms, the tolerance tol_rel ||b||, and whether
// a step is due.
KERNEL_THREADS(THREADS) void k_refine_start(GLOBAL const W* RESTRICT b, GLOBAL const W* RESTRICT r, GLOBAL W* RESTRICT st, GLOBAL u32* RESTRICT flags, u32 n, W tol_rel) {
    SHARED W sh[THREADS];
    W bn = sqrt(block_sumsq(b, n, sh));
    W rn = sqrt(block_sumsq(r, n, sh));
    if (thread_x() == 0) {
        W tol = tol_rel * bn;
        st[R_BNORM] = bn;
        st[R_TOL] = tol;
        st[R_RNORM] = rn;
        flags[F_STEPS] = 0;
        flags[F_REVERT] = 0;
        flags[F_ACTIVE] = rn > tol ? 1u : 0u;
    }
}

// A refinement step's update, one thread per entry, when the step is due:
// x kept in x_prev, the correction d added to it.
KERNEL_THREADS(THREADS) void k_refine_apply(GLOBAL const u32* RESTRICT flags, GLOBAL W* RESTRICT x, GLOBAL W* RESTRICT x_prev, GLOBAL const W* RESTRICT d, u32 n) {
    u32 i = global_x();
    if (flags[F_ACTIVE] != 0 && i < n) {
        x_prev[i] = x[i];
        x[i] += d[i];
    }
}

// A due refinement step's judgement, one block, from the residual r of the
// updated iterate: a step that did not reduce the residual's norm is to be
// undone and ends the refinement; otherwise the norm is kept and the
// refinement goes on while it is above the tolerance.
KERNEL_THREADS(THREADS) void k_refine_judge(GLOBAL const W* RESTRICT r, GLOBAL W* RESTRICT st, GLOBAL u32* RESTRICT flags, u32 n) {
    SHARED W sh[THREADS];
    // every thread reads the flag before the first barrier, thread 0 writes
    // it after the last
    if (flags[F_ACTIVE] == 0) return;
    W next = sqrt(block_sumsq(r, n, sh));
    if (thread_x() == 0) {
        flags[F_STEPS] += 1;
        if (next >= st[R_RNORM]) {
            // the factor is too far off for the correction to help
            flags[F_REVERT] = 1;
            flags[F_ACTIVE] = 0;
        } else {
            st[R_RNORM] = next;
            if (!(next > st[R_TOL])) flags[F_ACTIVE] = 0;
        }
    }
}

// The iterate before a rejected step back, one thread per entry.
KERNEL_THREADS(THREADS) void k_refine_revert(GLOBAL const u32* RESTRICT flags, GLOBAL W* RESTRICT x, GLOBAL const W* RESTRICT x_prev, u32 n) {
    u32 i = global_x();
    if (flags[F_REVERT] != 0 && i < n) x[i] = x_prev[i];
}
