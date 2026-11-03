// schur.cpp: the Schur reduction on the device and the refinement of its
// reduced solve (docs/dev/GPU.md, E2 and E9), launched by
// arael-gpu/src/native/schur.rs over the plan of arael-gpu/src/schur.rs.
//
// All f64. Compiled per EW (the widest eliminated block, rounded up to 4,
// 8, 16, 32 or 64), KW (the widest kept block, rounded likewise) and ACC
// (KW * KW / 32, the entries of a tile of S per lane); the plan header's
// slots (SE_*, SO_*, ST_*, SP_*, SK_*), the refinement state's words
// (R_*) and flags (F_*) come from schur.rs as #defines. Kernels launched
// with THREADS threads per block, or PLANE where said.
//
// Per-thread arrays are indexed only by constants once their loops unroll:
// the loops run to EW or KW and skip what is past the block's width, so the
// arrays stay in registers (arael_gpu.h's rule, from k_trsm).

#if ARAEL_WARP != 32
#error "schur.cpp: the plane kernels are written for 32-lane warps"
#endif

// One plane: the threads of the tile, rhs and symv kernels.
#define PLANE 32u
// No kept-kept tile of H behind a tile of S.
#define NONE 0xffffffffu

KERNEL_THREADS(PLANE) void k_zero_u32(GLOBAL u32* RESTRICT buf, u32 n) {
    u32 i = global_x();
    if (i < n) buf[i] = 0;
}

// C[i][k] of an observer's coupling tile at `ca`, wo x we, in either storage
// orientation.
DEVICE double coupling(GLOBAL const double* h, u32 ca, u32 trans, u32 wo, u32 we, u32 i, u32 k) {
    return trans != 0 ? h[ca + k + i * we] : h[ca + i + k * wo];
}

// Stage 1, one thread per eliminated block: the inverse of its damped
// diagonal tile (upper-stored) through the Cholesky factor, column-major
// we x we into `dinv`, and z = D^-1 b. A pivot that is not positive sets
// fail[0] to the block's slot plus one and continues with 1.
KERNEL_THREADS(THREADS) void k_schur_elim(GLOBAL const u32* RESTRICT plan, GLOBAL const double* RESTRICT h, GLOBAL const double* RESTRICT rhs, GLOBAL double* RESTRICT dinv, GLOBAL double* RESTRICT z, GLOBAL u32* RESTRICT fail, u32 ne) {
    u32 e = global_x();
    if (e >= ne) return;
    u32 w = plan[plan[SE_W] + e];
    u32 diag = plan[plan[SE_DIAG_OFF] + e];
    u32 orig = plan[plan[SE_ORIG_OFF] + e];
    u32 doff = plan[plan[SE_DINV_OFF] + e];
    u32 zoff = plan[plan[SE_Z_OFF] + e];
    // the tile, symmetric from its upper triangle, then its lower factor in place
    double a[EW * EW];
#pragma unroll
    for (u32 j = 0; j < EW; j++)
#pragma unroll
        for (u32 i = 0; i < EW; i++)
            if (i < w && j < w) a[i + j * EW] = i > j ? h[diag + j + i * w] : h[diag + i + j * w];
#pragma unroll
    for (u32 j = 0; j < EW; j++) {
        if (j < w) {
            double d = a[j + j * EW];
#pragma unroll
            for (u32 k = 0; k < j; k++) d -= a[j + k * EW] * a[j + k * EW];
            if (!(d > 0.0)) {
                fail[0] = e + 1;
                d = 1.0;
            }
            double ljj = sqrt(d);
            a[j + j * EW] = ljj;
#pragma unroll
            for (u32 i = j + 1; i < EW; i++) {
                if (i < w) {
                    double v = a[i + j * EW];
#pragma unroll
                    for (u32 k = 0; k < j; k++) v -= a[i + k * EW] * a[j + k * EW];
                    a[i + j * EW] = v / ljj;
                }
            }
        }
    }
    // L^-1 (lower) by forward substitution, column by column
    double li[EW * EW];
#pragma unroll
    for (u32 c = 0; c < EW; c++) {
#pragma unroll
        for (u32 r = 0; r < EW; r++) {
            if (c < w && r < w) {
                if (r >= c) {
                    double v = r == c ? 1.0 : 0.0;
#pragma unroll
                    for (u32 k = c; k < r; k++) v -= a[r + k * EW] * li[k + c * EW];
                    li[r + c * EW] = v / a[r + r * EW];
                } else {
                    li[r + c * EW] = 0.0;
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
                double v = 0.0;
#pragma unroll
                for (u32 k = 0; k < EW; k++)
                    if (k < w) v += li[k + i * EW] * li[k + j * EW];
                a[i + j * EW] = v;
                dinv[doff + i + j * w] = v;
            }
        }
    }
#pragma unroll
    for (u32 i = 0; i < EW; i++) {
        if (i < w) {
            double v = 0.0;
#pragma unroll
            for (u32 j = 0; j < EW; j++)
                if (j < w) v += a[i + j * EW] * rhs[orig + j];
            z[zoff + i] = v;
        }
    }
}

// Stage 2, one thread per observer: W_o = C_o D_e^-1, row-major wo x we.
KERNEL_THREADS(THREADS) void k_schur_w(GLOBAL const u32* RESTRICT plan, GLOBAL const double* RESTRICT h, GLOBAL const double* RESTRICT dinv, GLOBAL double* RESTRICT w, u32 no) {
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
            double v = 0.0;
            for (u32 k = 0; k < we; k++) v += coupling(h, ca, trans, wo, we, i, k) * dinv[doff + k + l * we];
            w[woff + i * we + l] = v;
        }
    }
}

// Stage 3, one block of PLANE threads per tile of S: the kept-kept tile of H
// (or nothing) minus W_oa C_ob^T over the tile's pairs, thread `lane` owning
// the entries lane, lane + 32, ... (ACC of them at most); a diagonal tile
// upper-only.
KERNEL_THREADS(PLANE) void k_schur_tiles(GLOBAL const u32* RESTRICT plan, GLOBAL const double* RESTRICT h, GLOBAL const double* RESTRICT w, GLOBAL double* RESTRICT s) {
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
    double acc[ACC];
#pragma unroll
    for (u32 m = 0; m < ACC; m++) acc[m] = 0.0;
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
                double v = 0.0;
                for (u32 k = 0; k < we; k++) v += w[woff + i * we + k] * coupling(h, cb, tb, wb, we, j, k);
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
            double v = acc[m];
            if (copy != NONE) v += h[copy + i + j * wr];
            if (a == b && i > j) v = 0.0;
            s[val + i + j * wr] = v;
        }
    }
}

// Stage 4, one block of PLANE threads per kept block: rhs_kept = b_k - sum
// over the block's observers of C_o z_e, the lanes over the observers, a
// warp sum per row.
KERNEL_THREADS(PLANE) void k_schur_rhs(GLOBAL const u32* RESTRICT plan, GLOBAL const double* RESTRICT h, GLOBAL const double* RESTRICT rhs, GLOBAL const double* RESTRICT z, GLOBAL double* RESTRICT out) {
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
    double acc[KW];
#pragma unroll
    for (u32 i = 0; i < KW; i++) acc[i] = 0.0;
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
                double v = 0.0;
                for (u32 kk = 0; kk < we; kk++) v += coupling(h, ca, trans, wk, we, i, kk) * z[zoff + kk];
                acc[i] += v;
            }
        }
    }
    // wk is the block's, so the whole warp takes each sum
#pragma unroll
    for (u32 i = 0; i < KW; i++) {
        if (i < wk) {
            double tot = warp_sum(acc[i]);
            if (lane == 0) out[s_off + i] = rhs[orig + i] - tot;
        }
    }
}

// The kept blocks of the reduced solution into the full vector, one thread
// per kept block.
KERNEL_THREADS(THREADS) void k_scatter_kept(GLOBAL const u32* RESTRICT plan, GLOBAL const double* RESTRICT xk, GLOBAL double* RESTRICT x, u32 nk) {
    u32 k = global_x();
    if (k >= nk) return;
    u32 s_off = plan[plan[SK_S_OFF] + k];
    u32 orig = plan[plan[SK_ORIG_OFF] + k];
    u32 wk = plan[plan[SK_W] + k];
    for (u32 i = 0; i < wk; i++) x[orig + i] = xk[s_off + i];
}

// Back-substitution, one thread per eliminated block: x_e = D_e^-1 (b_e -
// sum_a C_a^T x_a) over its observers, the kept slices of `x` in place.
KERNEL_THREADS(THREADS) void k_schur_backsub(GLOBAL const u32* RESTRICT plan, GLOBAL const double* RESTRICT h, GLOBAL const double* RESTRICT rhs, GLOBAL const double* RESTRICT dinv, GLOBAL double* RESTRICT x, u32 ne) {
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
    double t[EW];
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
                double v = 0.0;
                for (u32 i = 0; i < wa; i++) v += coupling(h, ca, trans, wa, w, i, kk) * x[xo + i];
                t[kk] -= v;
            }
        }
    }
#pragma unroll
    for (u32 i = 0; i < EW; i++) {
        if (i < w) {
            double v = 0.0;
#pragma unroll
            for (u32 j = 0; j < EW; j++)
                if (j < w) v += dinv[doff + i + j * w] * t[j];
            x[orig + i] = v;
        }
    }
}

// r -= S x over the tiles of S (upper block storage, diagonal tiles
// upper-only), one block of PLANE threads per tile, the lanes over the
// tile's rows for the product and over its columns for the transposed one;
// nothing unless gate[0] is nonzero.
KERNEL_THREADS(PLANE) void k_symv_sub(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT plan, GLOBAL const double* RESTRICT s, GLOBAL const double* RESTRICT x, GLOBAL double* RESTRICT r) {
    if (gate[0] == 0) return;
    u32 t = block_x();
    u32 lane = thread_x();
    u32 val = plan[plan[ST_VAL_OFF] + t];
    u32 wr = plan[plan[ST_WR] + t];
    u32 wc = plan[plan[ST_WC] + t];
    u32 a = plan[plan[ST_A] + t];
    u32 b = plan[plan[ST_B] + t];
    u32 a_off = plan[plan[ST_A_OFF] + t];
    u32 b_off = plan[plan[ST_B_OFF] + t];
    if (a == b) {
        for (u32 i = lane; i < wr; i += PLANE) {
            double acc = 0.0;
            for (u32 j = 0; j < wc; j++) acc += (i > j ? s[val + j + i * wr] : s[val + i + j * wr]) * x[b_off + j];
            atomic_add(&r[a_off + i], 0.0 - acc);
        }
    } else {
        for (u32 i = lane; i < wr; i += PLANE) {
            double acc = 0.0;
            for (u32 j = 0; j < wc; j++) acc += s[val + i + j * wr] * x[b_off + j];
            atomic_add(&r[a_off + i], 0.0 - acc);
        }
        for (u32 j = lane; j < wc; j += PLANE) {
            double acc = 0.0;
            for (u32 i2 = 0; i2 < wr; i2++) acc += s[val + i2 + j * wr] * x[a_off + i2];
            atomic_add(&r[b_off + j], 0.0 - acc);
        }
    }
}

// dst = src when gate[0] is nonzero.
KERNEL_THREADS(THREADS) void k_copy_f64(GLOBAL const u32* RESTRICT gate, GLOBAL double* RESTRICT dst, GLOBAL const double* RESTRICT src, u32 n) {
    u32 i = global_x();
    if (gate[0] != 0 && i < n) dst[i] = src[i];
}

// The sum of squares of x[0..n) by one block of THREADS threads, returned to
// every thread; `sh` is THREADS words of shared memory.
DEVICE double block_sumsq(GLOBAL const double* x, u32 n, SHARED_PTR double* sh) {
    u32 t = thread_x();
    double acc = 0.0;
    for (u32 i = t; i < n; i += THREADS) acc += x[i] * x[i];
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
KERNEL_THREADS(THREADS) void k_refine_start(GLOBAL const double* RESTRICT b, GLOBAL const double* RESTRICT r, GLOBAL double* RESTRICT st, GLOBAL u32* RESTRICT flags, u32 n, double tol_rel) {
    SHARED double sh[THREADS];
    double bn = sqrt(block_sumsq(b, n, sh));
    double rn = sqrt(block_sumsq(r, n, sh));
    if (thread_x() == 0) {
        double tol = tol_rel * bn;
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
KERNEL_THREADS(THREADS) void k_refine_apply(GLOBAL const u32* RESTRICT flags, GLOBAL double* RESTRICT x, GLOBAL double* RESTRICT x_prev, GLOBAL const double* RESTRICT d, u32 n) {
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
KERNEL_THREADS(THREADS) void k_refine_judge(GLOBAL const double* RESTRICT r, GLOBAL double* RESTRICT st, GLOBAL u32* RESTRICT flags, u32 n) {
    SHARED double sh[THREADS];
    // every thread reads the flag before the first barrier, thread 0 writes
    // it after the last
    if (flags[F_ACTIVE] == 0) return;
    double next = sqrt(block_sumsq(r, n, sh));
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
KERNEL_THREADS(THREADS) void k_refine_revert(GLOBAL const u32* RESTRICT flags, GLOBAL double* RESTRICT x, GLOBAL const double* RESTRICT x_prev, u32 n) {
    u32 i = global_x();
    if (flags[F_REVERT] != 0 && i < n) x[i] = x_prev[i];
}
