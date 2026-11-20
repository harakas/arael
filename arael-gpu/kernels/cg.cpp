// cg.cpp: preconditioned conjugate gradients on the reduced system of a
// Schur reduction, launched by arael-gpu/src/cg.rs. Compiled after wide.h
// and schur.h, with the reduction's #defines (schur.cpp lists them) and
// cg.rs's: the words of a kept block's record (KB_*, KB_REC of them), the
// values of a block's preconditioner factor (MSTRIDE: the widest kept
// block's square), the state's words (CG_*) and the flags (CF_*), and the
// reduction's flags read here (F_ELIM, F_FACTOR).
//
// S x = b with S's tiles, b and x in the reduction's work block, in R; the
// vectors r, z, p, q in R; the dot products, as partial sums per kept block
// and as the state, in W. The preconditioner is block Jacobi: the Cholesky
// factor of each diagonal tile of S.
//
// The iteration is arael-faer's cg::solve:
//
//   x = 0, r = b, z = M^-1 r, p = z, rho = r.z
//   until rho <= tol rho0, or the cap:
//     q = S p, alpha = rho / p.q        (p.q not positive ends it)
//     x += alpha p, r -= alpha q
//     z = M^-1 r, rho' = r.z
//     p = z + (rho' / rho) p, rho = rho'
//
// The number of iterations is data, so the kernels of an iteration are
// launched a fixed number of times and return at once when CF_ACTIVE is
// clear: the host reads the flags and launches more while it is set.

// The sum of v[0..n) by one block of THREADS threads, returned to every
// thread; `sh` is THREADS words of shared memory, t the thread.
DEVICE W block_sum(GLOBAL const W* v, u32 n, SHARED_PTR W* sh, u32 t) {
    W acc = 0.0f;
    for (u32 i = t; i < n; i += THREADS) acc += v[i];
    // a previous call's readers are done with the shared words first
    BARRIER();
    sh[t] = acc;
    BARRIER();
    for (u32 span = THREADS / 2; span > 0; span /= 2) {
        if (t < span) sh[t] += sh[t + span];
        BARRIER();
    }
    return sh[0];
}

// z = M^-1 r over a kept block's w rows, by two triangular solves with the
// block's factor m (below), in place in z; returns the block's part of r.z.
DEVICE W precond_block(GLOBAL const R* m, u32 w, GLOBAL const R* r, GLOBAL R* z) {
    for (u32 j = 0; j < w; j++) {
        R v = r[j];
        for (u32 c = 0; c < j; c++) v = r_fma(v, -m[j + c * w], z[c]);
        z[j] = v * m[j + j * w];
    }
    for (u32 jj = 0; jj < w; jj++) {
        u32 j = w - 1 - jj;
        R v = z[j];
        for (u32 c = j + 1; c < w; c++) v = r_fma(v, -m[c + j * w], z[c]);
        z[j] = v * m[j + j * w];
    }
    W dot = 0.0f;
    for (u32 i = 0; i < w; i++) dot = wide_fma(dot, W(r[i]), W(z[i]));
    return dot;
}

// Each kept block's record (KB_REC words): its offset in the reduced
// vector, its width, and the value offset in s of its diagonal tile, found
// among its tile entries (NONE when S has none). Once per structure, a
// thread a block.
KERNEL_THREADS(THREADS) void k_cg_blocks(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const u32* RESTRICT kt_ptr, GLOBAL const u32* RESTRICT kt, GLOBAL u32* RESTRICT kb, u32 nk) {
    u32 k = global_x();
    if (k >= nk) return;
    u32 diag = NONE;
    for (u32 q = kt_ptr[k]; q < kt_ptr[k + 1]; q++)
        if (KT_DIAG(kt[KT_REC * q + KT_DIMS]) != 0) diag = kt[KT_REC * q + KT_VAL_OFF];
    kb[KB_REC * k + KB_OFF] = plan[H.w[SK_S_OFF] + k];
    kb[KB_REC * k + KB_W] = plan[H.w[SK_W] + k];
    kb[KB_REC * k + KB_DIAG] = diag;
}

// The preconditioner, one thread per kept block: the lower Cholesky factor
// of the block's diagonal tile of S (stored upper) into its MSTRIDE values
// of mfac, column-major w x w, the reciprocals of the factor's diagonal on
// the diagonal. In place in mfac, the loops over the block's own width
// (kept blocks are up to 64 wide). A pivot that is not positive, or a
// block without a diagonal tile, sets fail[0]: S is not positive definite.
KERNEL_THREADS(THREADS) void k_cg_precond(GLOBAL const u32* RESTRICT kb, GLOBAL const R* RESTRICT s, GLOBAL R* RESTRICT mfac, GLOBAL u32* RESTRICT fail, u32 nk) {
    u32 k = global_x();
    if (k >= nk) return;
    u32 w = kb[KB_REC * k + KB_W];
    u32 diag = kb[KB_REC * k + KB_DIAG];
    if (diag == NONE) {
        fail[0] = 1;
        return;
    }
    GLOBAL R* m = mfac + MSTRIDE * k;
    for (u32 j = 0; j < w; j++) {
        R d = s[diag + j + j * w];
        for (u32 c = 0; c < j; c++) d = r_fma(d, -m[j + c * w], m[j + c * w]);
        if (!(d > 0.0f)) {
            fail[0] = 1;
            d = 1.0f;
        }
        R inv = R(1.0f) / sqrt(d);
        m[j + j * w] = inv;
        for (u32 i = j + 1; i < w; i++) {
            // S(j, i) of the upper triangle is L(i, j)'s entry
            R v = s[diag + j + i * w];
            for (u32 c = 0; c < j; c++) v = r_fma(v, -m[i + c * w], m[j + c * w]);
            m[i + j * w] = v * inv;
        }
    }
}

// The start, one thread per kept block: x = 0, r = b, z = M^-1 r, p = z,
// and the block's part of r.z.
KERNEL_THREADS(THREADS) void k_cg_start(GLOBAL const u32* RESTRICT kb, GLOBAL const R* RESTRICT mfac, GLOBAL const R* RESTRICT b, GLOBAL R* RESTRICT x, GLOBAL R* RESTRICT r, GLOBAL R* RESTRICT z, GLOBAL R* RESTRICT p, GLOBAL W* RESTRICT part, u32 nk) {
    u32 k = global_x();
    if (k >= nk) return;
    u32 off = kb[KB_REC * k + KB_OFF];
    u32 w = kb[KB_REC * k + KB_W];
    for (u32 i = 0; i < w; i++) {
        x[off + i] = 0.0f;
        r[off + i] = b[off + i];
    }
    part[k] = precond_block(mfac + MSTRIDE * k, w, r + off, z + off);
    for (u32 i = 0; i < w; i++) p[off + i] = z[off + i];
}

// The start's sum, one block: rho = rho0 = r.z, the target tol rho0, the
// flags. Nothing to iterate when the reduction or the preconditioner failed
// (failed[F_ELIM], failed[F_FACTOR]), when rho0 is not positive (solved:
// converged in no iteration) or is already at the target.
KERNEL_THREADS(THREADS) void k_cg_begin(GLOBAL const W* RESTRICT part, GLOBAL W* RESTRICT st, GLOBAL u32* RESTRICT fl, GLOBAL const u32* RESTRICT failed, u32 nk, W tol) {
    SHARED W sh[THREADS];
    W rho = block_sum(part, nk, sh, thread_x());
    if (thread_x() == 0) {
        W target = tol * rho;
        st[CG_RHO] = rho;
        st[CG_RHO0] = rho;
        st[CG_TARGET] = target;
        u32 bad = failed[F_ELIM] | failed[F_FACTOR];
        u32 go = bad == 0 && rho > 0.0f && !(rho <= target) ? 1u : 0u;
        fl[CF_ITERS] = 0;
        fl[CF_BREAKDOWN] = 0;
        fl[CF_RESTART] = 0;
        fl[CF_CONVERGED] = bad == 0 && go == 0 ? 1u : 0u;
        fl[CF_ACTIVE] = go;
    }
}

// q = S p over the tiles of S (upper block storage, diagonal tiles
// upper-only) and the parts of p.q, a block of threads per kept block k
// owning rows k of q; no two blocks write the same entry. Each of k's tile
// entries gives its rows: a tile (k, b) as stored, a tile (a, k) transposed.
#if KW <= PLANE
// The lanes over a tile's rows: MWARPS warps a block, each warp MGROUPS
// groups of KW lanes (schur.h). A group takes every (MWARPS MGROUPS)-th of
// k's tile entries, its lane `row` the tile's contribution to row `row` of
// q, so the lanes of a group read neighbouring values of S at every step
// and the same value of p. The groups' sums are added within the warp, the
// warps' through shared memory, and warp 0 writes q and the block's part of
// p.q.
KERNEL_THREADS(MWARPS * PLANE) void k_cg_matvec(GLOBAL const u32* RESTRICT fl, GLOBAL const u32* RESTRICT kt_ptr, GLOBAL const u32* RESTRICT kt, GLOBAL const u32* RESTRICT kb, GLOBAL const R* RESTRICT s, GLOBAL const R* RESTRICT p, GLOBAL R* RESTRICT q, GLOBAL W* RESTRICT part) {
#if MWARPS > 1
    SHARED R sh[MWARPS * KW];
#endif
    if (fl[CF_ACTIVE] == 0) return;
    u32 k = block_x();
    u32 warp = thread_x() / PLANE;
    u32 lane = thread_x() % PLANE;
    u32 row = lane % KW;
    u32 k_off = kb[KB_REC * k + KB_OFF];
    u32 wk = kb[KB_REC * k + KB_W];
    u32 end = kt_ptr[k + 1];
    R acc = 0.0f;
    for (u32 e = kt_ptr[k] + warp * MGROUPS + lane / KW; e < end; e += MWARPS * MGROUPS) {
        u32 val = kt[KT_REC * e + KT_VAL_OFF];
        u32 dims = kt[KT_REC * e + KT_DIMS];
        u32 xo = kt[KT_REC * e + KT_X_OFF];
        u32 wr = KT_WR(dims);
        if (row < wk) {
            if (KT_TRANS(dims) == 0) {
                // row `row` of (k, b): S_kb p_b, a diagonal tile read from its upper triangle
                u32 wc = KT_WC(dims);
                u32 diag = KT_DIAG(dims);
                for (u32 j = 0; j < wc; j++) acc = r_fma(acc, diag && row > j ? s[val + j + row * wr] : s[val + row + j * wr], p[xo + j]);
            } else {
                // row `row` of (a, k)^T: S_ak^T p_a
                for (u32 i2 = 0; i2 < wr; i2++) acc = r_fma(acc, s[val + i2 + row * wr], p[xo + i2]);
            }
        }
    }
    // a row's sums of the warp's groups: lanes row, row + KW, ...
#pragma unroll
    for (u32 m = KW; m < PLANE; m *= 2) acc += shfl_xor(acc, m);
#if MWARPS > 1
    // and of the block's warps
    if (lane < KW) sh[warp * KW + lane] = acc;
    BARRIER();
    if (warp != 0) return;
    if (lane < KW) {
        acc = sh[lane];
#pragma unroll
        for (u32 w = 1; w < MWARPS; w++) acc += sh[w * KW + lane];
    }
#endif
    W dot = 0.0f;
    if (lane < wk) {
        q[k_off + lane] = acc;
        dot = wide_fma(dot, acc, p[k_off + lane]);
    }
    dot = warp_sum(dot);
    if (lane == 0) part[k] = dot;
}
#else
// Kept blocks wider than a plane has lanes: one warp a block, the lanes
// over k's tile entries (k_symv_sub's walk), each lane all the rows of its
// tiles, and a warp sum per row.
KERNEL_THREADS(PLANE) void k_cg_matvec(GLOBAL const u32* RESTRICT fl, GLOBAL const u32* RESTRICT kt_ptr, GLOBAL const u32* RESTRICT kt, GLOBAL const u32* RESTRICT kb, GLOBAL const R* RESTRICT s, GLOBAL const R* RESTRICT p, GLOBAL R* RESTRICT q, GLOBAL W* RESTRICT part) {
    if (fl[CF_ACTIVE] == 0) return;
    u32 k = block_x();
    u32 lane = thread_x();
    u32 k_off = kb[KB_REC * k + KB_OFF];
    u32 wk = kb[KB_REC * k + KB_W];
    R acc[KW];
#pragma unroll
    for (u32 i = 0; i < KW; i++) acc[i] = 0.0f;
    for (u32 e = kt_ptr[k] + lane; e < kt_ptr[k + 1]; e += PLANE) {
        u32 val = kt[KT_REC * e + KT_VAL_OFF];
        u32 dims = kt[KT_REC * e + KT_DIMS];
        u32 xo = kt[KT_REC * e + KT_X_OFF];
        u32 wr = KT_WR(dims);
        if (KT_TRANS(dims) == 0) {
            // rows of (k, b): S_kb p_b, a diagonal tile read from its upper triangle
            u32 wc = KT_WC(dims);
            u32 diag = KT_DIAG(dims);
#pragma unroll
            for (u32 i = 0; i < KW; i++) {
                if (i < wk) {
                    R v = 0.0f;
                    for (u32 j = 0; j < wc; j++) v = r_fma(v, diag && i > j ? s[val + j + i * wr] : s[val + i + j * wr], p[xo + j]);
                    acc[i] += v;
                }
            }
        } else {
            // rows of (a, k)^T: S_ak^T p_a
#pragma unroll
            for (u32 j = 0; j < KW; j++) {
                if (j < wk) {
                    R v = 0.0f;
                    for (u32 i2 = 0; i2 < wr; i2++) v = r_fma(v, s[val + i2 + j * wr], p[xo + i2]);
                    acc[j] += v;
                }
            }
        }
    }
    // wk is the block's, so the whole warp takes each sum
    W dot = 0.0f;
#pragma unroll
    for (u32 i = 0; i < KW; i++) {
        if (i < wk) {
            R tot = warp_sum(acc[i]);
            if (lane == 0) {
                q[k_off + i] = tot;
                dot = wide_fma(dot, W(tot), W(p[k_off + i]));
            }
        }
    }
    if (lane == 0) part[k] = dot;
}
#endif

// The step length, one block: alpha = rho / p.q from the parts of p.q. A
// p.q that is not positive (S not positive definite along p, or not a
// number) ends the iteration with the iterate it has. With `restart`
// nonzero, CF_RESTART is set on the iterations that recompute the residual:
// every restart-th.
KERNEL_THREADS(THREADS) void k_cg_alpha(GLOBAL u32* RESTRICT fl, GLOBAL const W* RESTRICT part, GLOBAL W* RESTRICT st, u32 nk, u32 restart) {
    SHARED W sh[THREADS];
    // every thread reads the flag before the first barrier, thread 0 writes
    // it after the last
    if (fl[CF_ACTIVE] == 0) return;
    W pq = block_sum(part, nk, sh, thread_x());
    if (thread_x() == 0) {
        if (!(pq > 0.0f)) {
            fl[CF_BREAKDOWN] = 1;
            fl[CF_ACTIVE] = 0;
        } else {
            st[CG_ALPHA] = st[CG_RHO] / pq;
            fl[CF_RESTART] = restart != 0 && (fl[CF_ITERS] + 1) % restart == 0 ? 1u : 0u;
        }
    }
}

// The update, one thread per kept block: x += alpha p, r -= alpha q,
// z = M^-1 r, and the block's part of r.z.
KERNEL_THREADS(THREADS) void k_cg_update(GLOBAL const u32* RESTRICT fl, GLOBAL const W* RESTRICT st, GLOBAL const u32* RESTRICT kb, GLOBAL const R* RESTRICT mfac, GLOBAL const R* RESTRICT p, GLOBAL const R* RESTRICT q, GLOBAL R* RESTRICT x, GLOBAL R* RESTRICT r, GLOBAL R* RESTRICT z, GLOBAL W* RESTRICT part, u32 nk) {
    u32 k = global_x();
    if (fl[CF_ACTIVE] == 0 || k >= nk) return;
    u32 off = kb[KB_REC * k + KB_OFF];
    u32 w = kb[KB_REC * k + KB_W];
    R alpha = (R)st[CG_ALPHA];
    for (u32 i = 0; i < w; i++) {
        x[off + i] = r_fma(x[off + i], alpha, p[off + i]);
        r[off + i] = r_fma(r[off + i], -alpha, q[off + i]);
    }
    part[k] = precond_block(mfac + MSTRIDE * k, w, r + off, z + off);
}

// z = M^-1 r and the parts of r.z again, one thread per kept block, when
// gate[0] is nonzero: after the residual was recomputed.
KERNEL_THREADS(THREADS) void k_cg_apply(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT kb, GLOBAL const R* RESTRICT mfac, GLOBAL const R* RESTRICT r, GLOBAL R* RESTRICT z, GLOBAL W* RESTRICT part, u32 nk) {
    u32 k = global_x();
    if (gate[0] == 0 || k >= nk) return;
    u32 off = kb[KB_REC * k + KB_OFF];
    part[k] = precond_block(mfac + MSTRIDE * k, kb[KB_REC * k + KB_W], r + off, z + off);
}

// The iteration's end, one block: rho' = r.z from its parts and the
// iteration counted. At or under the target the solve has converged;
// otherwise beta = rho' / rho for the next direction, and at `cap`
// iterations the solve stops with the iterate it has.
KERNEL_THREADS(THREADS) void k_cg_rho(GLOBAL u32* RESTRICT fl, GLOBAL const W* RESTRICT part, GLOBAL W* RESTRICT st, u32 nk, u32 cap) {
    SHARED W sh[THREADS];
    // every thread reads the flag before the first barrier, thread 0 writes
    // it after the last
    if (fl[CF_ACTIVE] == 0) return;
    W rho = block_sum(part, nk, sh, thread_x());
    if (thread_x() == 0) {
        u32 iters = fl[CF_ITERS] + 1;
        fl[CF_ITERS] = iters;
        fl[CF_RESTART] = 0;
        if (rho <= st[CG_TARGET]) {
            fl[CF_CONVERGED] = 1;
            fl[CF_ACTIVE] = 0;
        } else {
            st[CG_BETA] = rho / st[CG_RHO];
            if (iters >= cap) fl[CF_ACTIVE] = 0;
        }
        st[CG_RHO] = rho;
    }
}

// The next direction, one thread per entry: p = z + beta p.
KERNEL_THREADS(THREADS) void k_cg_direction(GLOBAL const u32* RESTRICT fl, GLOBAL const W* RESTRICT st, GLOBAL const R* RESTRICT z, GLOBAL R* RESTRICT p, u32 n) {
    u32 i = global_x();
    if (fl[CF_ACTIVE] == 0 || i >= n) return;
    p[i] = r_fma(z[i], (R)st[CG_BETA], p[i]);
}
