// schur.cpp: the Schur reduction on the device and the refinement of its
// reduced solve (docs/dev/GPU.md, E2 and E9), launched by
// arael-gpu/src/schur.rs over its plan.
//
// H, the gradient and the step are in V, the route's values (wide.h); the
// reduction computes in R (below) and keeps its vectors in R; the
// refinement's norms and state are in W, the route's wide scalar. Compiled per EWX (the widest
// eliminated block), EW (it rounded up to 4, 8, 16, 32 or 64), KW (the
// widest kept block, rounded likewise), TILE_LANES (k_schur_tiles' threads
// per unit: the device's lockstep width, 32 or 64), TILE_WARPS (its units
// per block), TILE_GROUPS (a unit's lane groups) and ACC (the widest kept
// block's square over a group's lanes, rounded up: the entries of a tile of
// S per lane);
// the plan header's slots (SE_*, SO_*, ST_*, SP_*, SK_*, SB_*), the
// refinement state's words (R_*) and flags (F_*) come from schur.rs as
// #defines. Kernels launched with THREADS threads per block, or PLANE where
// said.
//
// Per-thread arrays are indexed only by constants once their loops unroll:
// the loops run to EW or KW and skip what is past the block's width, so the
// arrays stay in registers (arael_gpu.h's rule, from k_trsm).

// One plane: the threads of the tile, rhs and symv kernels.
#define PLANE 32u
#if PLANE != WARP
#error "schur.cpp: the rhs and symv kernels sum over a plane with warp_sum"
#endif
// No kept-kept tile of H behind a tile of S.
#define NONE 0xffffffffu

// R: the scalar the reduction computes in. W itself with SCHUR_F32 0; f32
// with SCHUR_F32 1, as the CPU's f32 route, for a solve whose factor is f32
// and that does not refine against S: the inputs (H, the gradient, the kept
// blocks' solution) are rounded to f32 as read (an f32 root's H and
// gradient already are), and the eliminated blocks' factors, z, W, S, the
// reduced right-hand side and the kept blocks' solution are kept in f32.
#if SCHUR_F32
typedef float R;
#define R_DF32 0
DEVICE R r_fma(R acc, R x, R y) {
    return fma(x, y, acc);
}
#else
typedef W R;
#define R_DF32 ARAEL_WIDE_DF32
DEVICE R r_fma(R acc, R x, R y) {
    return wide_fma(acc, x, y);
}
#endif
// v in R: a value of H or the gradient (V), or of a vector (W).
template <typename T> DEVICE R to_r(T v) {
    return (R)v;
}

KERNEL_THREADS(PLANE) void k_zero_u32(GLOBAL u32* RESTRICT buf, u32 n) {
    u32 i = global_x();
    if (i < n) buf[i] = 0;
}

// C[i][k] of an observer's coupling tile at `ca`, wo x we, in either storage
// orientation.
DEVICE V coupling(GLOBAL const V* h, u32 ca, u32 trans, u32 wo, u32 we, u32 i, u32 k) {
    return trans != 0 ? h[ca + k + i * we] : h[ca + i + k * wo];
}

// v = D^-1 v in place (w <= EW entries) by two triangular solves with D's
// factor at `f`, column-major w x w: D = U P U^T, U unit lower (D's
// Cholesky factor L with each column divided by its diagonal entry) below
// the diagonal, P = diag(L_jj^2) as its reciprocals on the diagonal. The
// CPU's llt_solve_panel up to rounding, at one product a term: solves keep
// S the reduction of a nearby positive definite system where multiplying
// by D's rounded inverse does not (its error grows with D's condition
// number, high for a point seen along nearly parallel rays). wide_fma's
// rounding, relative to its terms, keeps the solves backward stable.
DEVICE void ldlt_solve(GLOBAL const R* f, u32 w, THREAD R* v) {
#pragma unroll
    for (u32 k = 0; k < EW; k++) {
        if (k < w) {
            R s = v[k];
#pragma unroll
            for (u32 j = 0; j < k; j++) s = r_fma(s, -f[k + j * w], v[j]);
            v[k] = s;
        }
    }
#pragma unroll
    for (u32 k = 0; k < EW; k++)
        if (k < w) v[k] = v[k] * f[k + k * w];
#pragma unroll
    for (u32 kk = 0; kk < EW; kk++) {
        u32 k = EW - 1 - kk;
        if (k < w) {
            R s = v[k];
#pragma unroll
            for (u32 j = k + 1; j < EW; j++)
                if (j < w) s = r_fma(s, -f[j + k * w], v[j]);
            v[k] = s;
        }
    }
}

// Stage 1, one thread per eliminated block: the factor of its damped
// diagonal tile (upper-stored) into `dfac` (see ldlt_solve), from its
// Cholesky factor, and z = D^-1 b. A pivot that is not positive sets
// fail[0] to the block's slot plus one and continues with 1.
KERNEL_THREADS_REGS(THREADS) void k_schur_elim(GLOBAL const u32* RESTRICT plan, GLOBAL const V* RESTRICT h, GLOBAL const V* RESTRICT rhs, GLOBAL R* RESTRICT dfac, GLOBAL R* RESTRICT z, GLOBAL u32* RESTRICT fail, u32 ne) {
    u32 e = global_x();
    if (e >= ne) return;
    u32 w = plan[plan[SE_W] + e];
    u32 diag = plan[plan[SE_DIAG_OFF] + e];
    u32 orig = plan[plan[SE_ORIG_OFF] + e];
    u32 doff = plan[plan[SE_DFAC_OFF] + e];
    u32 zoff = plan[plan[SE_Z_OFF] + e];
    // the tile, symmetric from its upper triangle, then its lower factor in
    // place and its pivots L_jj^2
    R a[EW * EW];
    R piv[EW];
#pragma unroll
    for (u32 j = 0; j < EW; j++)
#pragma unroll
        for (u32 i = 0; i < EW; i++)
            if (i < w && j < w) a[i + j * EW] = to_r(i > j ? h[diag + j + i * w] : h[diag + i + j * w]);
#pragma unroll
    for (u32 j = 0; j < EW; j++) {
        if (j < w) {
            R d = a[j + j * EW];
#pragma unroll
            for (u32 k = 0; k < j; k++) d -= a[j + k * EW] * a[j + k * EW];
            if (!(d > 0.0f)) {
                fail[0] = e + 1;
                d = 1.0f;
            }
            piv[j] = d;
            R ljj = sqrt(d);
            a[j + j * EW] = ljj;
#pragma unroll
            for (u32 i = j + 1; i < EW; i++) {
                if (i < w) {
                    R v = a[i + j * EW];
#pragma unroll
                    for (u32 k = 0; k < j; k++) v -= a[i + k * EW] * a[j + k * EW];
                    a[i + j * EW] = v / ljj;
                }
            }
        }
    }
#pragma unroll
    for (u32 j = 0; j < EW; j++)
#pragma unroll
        for (u32 i = 0; i < EW; i++)
            if (i < w && j < w && i >= j) dfac[doff + i + j * w] = i == j ? R(1.0f) / piv[j] : a[i + j * EW] / a[j + j * EW];
    R t[EW];
#pragma unroll
    for (u32 i = 0; i < EW; i++) t[i] = i < w ? to_r(rhs[orig + i]) : R(0.0f);
    ldlt_solve(dfac + doff, w, t);
#pragma unroll
    for (u32 i = 0; i < EW; i++)
        if (i < w) z[zoff + i] = t[i];
}

// Stage 2, KW threads per observer, one per row of W_o (those past wo
// idle): row i of W_o = C_o D_e^-1, row-major wo x we, the row of C_o
// solved with D's factor (ldlt_solve). Neighbouring threads read and write
// neighbouring rows.
KERNEL_THREADS(THREADS) void k_schur_w(GLOBAL const u32* RESTRICT plan, GLOBAL const V* RESTRICT h, GLOBAL const R* RESTRICT dfac, GLOBAL R* RESTRICT w, u32 no) {
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
    u32 doff = plan[plan[SE_DFAC_OFF] + e];
    u32 woff = plan[plan[SO_W_OFF] + o];
    R c[EW];
#pragma unroll
    for (u32 k = 0; k < EW; k++) c[k] = k < we ? to_r(coupling(h, ca, trans, wo, we, i, k)) : R(0.0f);
    ldlt_solve(dfac + doff, we, c);
#pragma unroll
    for (u32 l = 0; l < EW; l++)
        if (l < we) w[woff + i * we + l] = c[l];
}

// Stage 3, blocks of TILE_WARPS lockstep units of TILE_LANES threads, the
// plan's block items (SB_ITEMS, TILE_WARPS + 1 words each): word 0 zero, a
// tile of S per unit (NONE: idle); nonzero, one heavy tile on the whole
// block, its units taking every TILE_WARPS-th step and summing their
// accumulators in shared memory at the end. A tile: the kept-kept tile of
// H (or nothing) minus W_oa C_ob^T over its pairs, the sum accumulated and
// subtracted once at the end (a sign in the products' loop is an f64
// instruction of its own where the selects below take it). A unit's lanes
// form TILE_GROUPS groups of GROUP_LANES, lane `lane` in group lane %
// TILE_GROUPS; a step of the unit takes TILE_GROUPS consecutive pairs, one
// per group, and the groups' sums are added at the tile's end. Lane `l` of
// a group owns the entries l, l + GROUP_LANES, ... (ACC of them at most); a
// diagonal tile its upper entries only, zeros below. A pair's W_oa and C_ob
// are copied once into its group's shared memory as they lie in memory; in
// f64 the products go straight into the accumulators. A unit's tile is the
// same in all its lanes, so are its plan words (scalar loads on AMD).
// A unit's barrier: its lockstep width's, or, when a unit is wider than the
// device's lockstep width (TILE_WARPS 1), the block's.
#if TILE_WARPS == 1
#define UNIT_BARRIER() BARRIER()
#else
#define UNIT_BARRIER() WARP_BARRIER()
#endif
#define GROUP_LANES (TILE_LANES / TILE_GROUPS)

// v + a c, a and c zeros where not `on` (past a group's pair's width;
// always on with one group).
DEVICE R tile_fma(R v, bool on, R a, R c) {
    on = TILE_GROUPS == 1 || on;
    R zero = 0.0f;
    return r_fma(v, on ? a : zero, on ? c : zero);
}

// A step's products: acc[m] plus W_oa(i, k) C_ob(j, k) over the columns k,
// (i, j) the entry of ij[m], W_oa(i, k) at ws[i we + k] and C_ob(j, k) at
// cs[j cj + k ck], zeros past the group's pair's width `we`. The columns
// to K, a constant: unrolled, without conditions; with K 0 to kn, uniform:
// a loop (unrolled to EW behind guards, the compiler turns the guards into
// predicates and all EW products issue). The loop indexes shared memory
// only. An accumulator past the tile's n entries in every lane is skipped.
template <u32 K>
DEVICE_INLINE void tile_products(THREAD R* acc, THREAD const u32* ij, u32 n, u32 kn, u32 we, u32 cj, u32 ck, SHARED_PTR const R* ws, SHARED_PTR const R* cs) {
#pragma unroll
    for (u32 m = 0; m < ACC; m++) {
        if (GROUP_LANES * m < n) {
            SHARED_PTR const R* a = ws + (ij[m] & 0xffffu) * we;
            SHARED_PTR const R* c = cs + (ij[m] >> 16) * cj;
#if R_DF32
            // df32: the pair's sum first, then one accurate addition
            R v = 0.0f;
            if constexpr (K != 0) {
#pragma unroll
                for (u32 k = 0; k < K; k++) v = tile_fma(v, k < we, a[k], c[k * ck]);
            } else {
#pragma unroll 1
                for (u32 k = 0; k < kn; k++) v = tile_fma(v, k < we, a[k], c[k * ck]);
            }
            acc[m] += v;
#else
            if constexpr (K != 0) {
#pragma unroll
                for (u32 k = 0; k < K; k++) acc[m] = tile_fma(acc[m], k < we, a[k], c[k * ck]);
            } else {
#pragma unroll 1
                for (u32 k = 0; k < kn; k++) acc[m] = tile_fma(acc[m], k < we, a[k], c[k * ck]);
            }
#endif
        }
    }
}

// df32: registers uncapped, as capped to fit more blocks the accumulators of
// several groups spill (CUDA); f64 and f32 run faster capped.
#if R_DF32
#define TILES_ENTRY KERNEL_THREADS_REGS(TILE_LANES * TILE_WARPS)
#else
#define TILES_ENTRY KERNEL_THREADS(TILE_LANES * TILE_WARPS)
#endif
TILES_ENTRY void k_schur_tiles(GLOBAL const u32* RESTRICT plan, GLOBAL const V* RESTRICT h, GLOBAL const R* RESTRICT w, GLOBAL R* RESTRICT s) {
    // the launch's shared memory (schur.rs, tiles_shared_bytes): the groups'
    // W_oa, their C_ob, the units' sums of a split tile, in R
    DYNAMIC_SHARED(R, ws_all);
    SHARED_PTR R* cs_all = ws_all + TILE_WARPS * TILE_GROUPS * KW * EW;
    SHARED_PTR R* red = cs_all + TILE_WARPS * TILE_GROUPS * KW * EW;
    u32 lane = thread_x() % TILE_LANES;
    // the groups interleaved, so each 32-lane warp holds all of them and
    // their sums are shuffles
    u32 g = lane % TILE_GROUPS;
    u32 l = lane / TILE_GROUPS;
    // the unit, and so its tile, uniform in the unit: the pair loop's
    // guards on the tile's size stay branches
    u32 unit = warp_uniform(thread_x() / TILE_LANES);
    GLOBAL const u32* item = plan + plan[SB_ITEMS] + (TILE_WARPS + 1) * block_x();
    bool split = item[0] != 0;
    u32 t = warp_uniform(item[1 + (split ? 0 : unit)]);
    // a unit without a tile ends here; a split block has no idle unit, and
    // only it waits for the whole block
    if (t == NONE) return;
    SHARED_PTR R* ws = ws_all + (unit * TILE_GROUPS + g) * KW * EW;
    SHARED_PTR R* cs = cs_all + (unit * TILE_GROUPS + g) * KW * EW;
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
    bool diag = a == b;
    // the entries computed: a diagonal tile's upper ones, column by column
    u32 n = diag ? wr * (wr + 1) / 2 : wr * wc;
    R acc[ACC];
    // each accumulator's entry of the tile, i | j << 16, found once; past
    // the tile's entries, entry (0, 0), computed and not written, so the
    // products need no condition of the lane's own
    u32 ij[ACC];
#pragma unroll
    for (u32 m = 0; m < ACC; m++) {
        acc[m] = 0.0f;
        u32 ent = l + GROUP_LANES * m;
        u32 i = 0;
        u32 j = 0;
        if (ent < n) {
            if (diag) {
                while ((j + 1) * (j + 2) / 2 <= ent) j++;
                i = ent - j * (j + 1) / 2;
            } else {
                j = ent / wr;
                i = ent - j * wr;
            }
        }
        ij[m] = i | (j << 16);
    }
    // a unit's steps, every TILE_WARPS-th of a split tile
    u32 stride = (split ? TILE_WARPS : 1u) * TILE_GROUPS;
    for (u32 q = p0 + (split ? unit * TILE_GROUPS : 0u); q < p1; q += stride) {
        // the group's pair; none past the tile's last
        u32 p = q + g;
        u32 we = 0u;
        u32 woff = 0u;
        u32 cb = 0u;
        u32 tb = 0u;
        u32 wb = 0u;
        if (p < p1) {
            u32 oa = plan[o_oa + p];
            u32 ob = plan[o_ob + p];
            we = plan[o_ew + plan[o_e + oa]];
            woff = plan[o_woff + oa];
            cb = plan[o_ca + ob];
            tb = plan[o_trans + ob];
            wb = plan[o_ow + ob];
        }
        // the step's widest pair, uniform in the unit (one group's too, for
        // the compiler): the products' columns
        u32 wu = we;
#pragma unroll
        for (u32 o = 1; o < TILE_GROUPS; o <<= 1) {
            u32 x = shfl_xor(wu, o);
            wu = x > wu ? x : wu;
        }
        wu = warp_uniform(wu);
        // the unit's previous step's reads are done
        UNIT_BARRIER();
        // W_oa (row-major wr x we) and C_ob as H holds it (coupling(): row j
        // at j we when transposed, else column k at k wb), both contiguous:
        // one pass of coalesced loads, no index division
        for (u32 x = l; x < wr * we; x += GROUP_LANES) ws[x] = w[woff + x];
        for (u32 x = l; x < wc * we; x += GROUP_LANES) cs[x] = to_r(h[cb + x]);
        UNIT_BARRIER();
        // entry (j, k) of C_ob at cs[j cj + k ck]
        u32 cj = tb != 0 ? we : 1u;
        u32 ck = tb != 0 ? 1u : wb;
        if (wu == EWX) tile_products<EWX>(acc, ij, n, wu, we, cj, ck, ws, cs);
        else tile_products<0>(acc, ij, n, wu, we, cj, ck, ws, cs);
    }
    // the groups' sums, in every lane
#pragma unroll
    for (u32 m = 0; m < ACC; m++) {
#pragma unroll
        for (u32 o = 1; o < TILE_GROUPS; o <<= 1) acc[m] += shfl_xor(acc[m], o);
    }
    if (split) {
        // the units' sums, added in unit order by the first unit
        if (g == 0) {
#pragma unroll
            for (u32 m = 0; m < ACC; m++) red[(unit * ACC + m) * GROUP_LANES + l] = acc[m];
        }
        BARRIER();
        if (unit != 0) return;
#pragma unroll
        for (u32 m = 0; m < ACC; m++) {
            for (u32 u = 1; u < TILE_WARPS; u++) acc[m] += red[(u * ACC + m) * GROUP_LANES + l];
        }
    }
    if (g == 0) {
#pragma unroll
        for (u32 m = 0; m < ACC; m++) {
            if (l + GROUP_LANES * m < n) {
                u32 i = ij[m] & 0xffffu;
                u32 j = ij[m] >> 16;
                R zero = 0.0f;
                R v = (copy != NONE ? to_r(h[copy + i + j * wr]) : zero) - acc[m];
                s[val + i + j * wr] = v;
            }
        }
    }
    if (diag) {
        for (u32 x = lane; x < wr * wr; x += TILE_LANES) {
            u32 j = x / wr;
            u32 i = x - j * wr;
            if (i > j) s[val + x] = R(0.0f);
        }
    }
}

// Stage 4, one block of PLANE threads per kept block: rhs_kept = b_k - sum
// over the block's observers of C_o z_e, the lanes over the observers, a
// warp sum per row.
KERNEL_THREADS(PLANE) void k_schur_rhs(GLOBAL const u32* RESTRICT plan, GLOBAL const V* RESTRICT h, GLOBAL const V* RESTRICT rhs, GLOBAL const R* RESTRICT z, GLOBAL R* RESTRICT out) {
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
    R acc[KW];
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
                R v = 0.0f;
                for (u32 kk = 0; kk < we; kk++) v = r_fma(v, to_r(coupling(h, ca, trans, wk, we, i, kk)), z[zoff + kk]);
                acc[i] += v;
            }
        }
    }
    // wk is the block's, so the whole warp takes each sum
#pragma unroll
    for (u32 i = 0; i < KW; i++) {
        if (i < wk) {
            R tot = warp_sum(acc[i]);
            if (lane == 0) out[s_off + i] = to_r(rhs[orig + i]) - tot;
        }
    }
}

// The kept blocks of the reduced solution (R) into the full vector (V), one
// thread per kept block.
KERNEL_THREADS(THREADS) void k_scatter_kept(GLOBAL const u32* RESTRICT plan, GLOBAL const R* RESTRICT xk, GLOBAL V* RESTRICT x, u32 nk) {
    u32 k = global_x();
    if (k >= nk) return;
    u32 s_off = plan[plan[SK_S_OFF] + k];
    u32 orig = plan[plan[SK_ORIG_OFF] + k];
    u32 wk = plan[plan[SK_W] + k];
    for (u32 i = 0; i < wk; i++) x[orig + i] = (V)xk[s_off + i];
}

// Back-substitution, one thread per eliminated block: x_e = D_e^-1 (b_e -
// sum_a C_a^T x_a) over its observers (ldlt_solve), the kept slices of `x`
// in place.
KERNEL_THREADS(THREADS) void k_schur_backsub(GLOBAL const u32* RESTRICT plan, GLOBAL const V* RESTRICT h, GLOBAL const V* RESTRICT rhs, GLOBAL const R* RESTRICT dfac, GLOBAL V* RESTRICT x, u32 ne) {
    u32 e = global_x();
    if (e >= ne) return;
    u32 o_ca = plan[SO_CA_OFF];
    u32 o_trans = plan[SO_TRANS];
    u32 o_ow = plan[SO_W];
    u32 o_oorig = plan[SO_ORIG_OFF];
    u32 w = plan[plan[SE_W] + e];
    u32 orig = plan[plan[SE_ORIG_OFF] + e];
    u32 doff = plan[plan[SE_DFAC_OFF] + e];
    u32 q0 = plan[plan[SE_OBS_PTR] + e];
    u32 q1 = plan[plan[SE_OBS_PTR] + e + 1];
    R t[EW];
#pragma unroll
    for (u32 i = 0; i < EW; i++)
        if (i < w) t[i] = to_r(rhs[orig + i]);
    for (u32 o = q0; o < q1; o++) {
        u32 wa = plan[o_ow + o];
        u32 ca = plan[o_ca + o];
        u32 trans = plan[o_trans + o];
        u32 xo = plan[o_oorig + o];
#pragma unroll
        for (u32 kk = 0; kk < EW; kk++) {
            if (kk < w) {
                R v = 0.0f;
                for (u32 i = 0; i < wa; i++) v = r_fma(v, to_r(coupling(h, ca, trans, wa, w, i, kk)), to_r(x[xo + i]));
                t[kk] -= v;
            }
        }
    }
    ldlt_solve(dfac + doff, w, t);
#pragma unroll
    for (u32 i = 0; i < EW; i++)
        if (i < w) x[orig + i] = (V)t[i];
}

// r -= S x over the tiles of S (upper block storage, diagonal tiles
// upper-only), one block of PLANE threads per kept block k owning rows k of
// r: the lanes over the tiles touching k (kt[kt_ptr[k]..kt_ptr[k + 1]], each
// a tile index times 2 plus 1 when k is its column block, whose transpose
// then gives the rows), a warp sum per row; nothing unless gate[0] is
// nonzero. No two blocks write the same entry.
KERNEL_THREADS(PLANE) void k_symv_sub(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT plan, GLOBAL const u32* RESTRICT kt_ptr, GLOBAL const u32* RESTRICT kt, GLOBAL const R* RESTRICT s, GLOBAL const R* RESTRICT x, GLOBAL R* RESTRICT r) {
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
                    for (u32 j = 0; j < wc; j++) v = wide_fma(v, W(a == b && i > j ? s[val + j + i * wr] : s[val + i + j * wr]), W(x[b_off + j]));
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
                    for (u32 i2 = 0; i2 < wr; i2++) v = wide_fma(v, W(s[val + i2 + j * wr]), W(x[a_off + i2]));
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
            if (lane == 0) r[k_off + i] = (R)(W(r[k_off + i]) - tot);
        }
    }
}

// dst = src (R) when gate[0] is nonzero.
KERNEL_THREADS(THREADS) void k_copy_r(GLOBAL const u32* RESTRICT gate, GLOBAL R* RESTRICT dst, GLOBAL const R* RESTRICT src, u32 n) {
    u32 i = global_x();
    if (gate[0] != 0 && i < n) dst[i] = src[i];
}

// The sum of squares of x[0..n) (R) in W by one block of THREADS threads,
// returned to every thread; `sh` is THREADS words of shared memory, t the
// thread.
DEVICE W block_sumsq(GLOBAL const R* x, u32 n, SHARED_PTR W* sh, u32 t) {
    W acc = 0.0f;
    for (u32 i = t; i < n; i += THREADS) acc = wide_fma(acc, W(x[i]), W(x[i]));
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

// The refinement's start, one block, after the first solve's residual r of
// the right-hand side b: the norms, the tolerance tol_rel ||b||, and whether
// a step is due.
KERNEL_THREADS(THREADS) void k_refine_start(GLOBAL const R* RESTRICT b, GLOBAL const R* RESTRICT r, GLOBAL W* RESTRICT st, GLOBAL u32* RESTRICT flags, u32 n, W tol_rel) {
    SHARED W sh[THREADS];
    W bn = sqrt(block_sumsq(b, n, sh, thread_x()));
    W rn = sqrt(block_sumsq(r, n, sh, thread_x()));
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
KERNEL_THREADS(THREADS) void k_refine_apply(GLOBAL const u32* RESTRICT flags, GLOBAL R* RESTRICT x, GLOBAL R* RESTRICT x_prev, GLOBAL const R* RESTRICT d, u32 n) {
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
KERNEL_THREADS(THREADS) void k_refine_judge(GLOBAL const R* RESTRICT r, GLOBAL W* RESTRICT st, GLOBAL u32* RESTRICT flags, u32 n) {
    SHARED W sh[THREADS];
    // every thread reads the flag before the first barrier, thread 0 writes
    // it after the last
    if (flags[F_ACTIVE] == 0) return;
    W next = sqrt(block_sumsq(r, n, sh, thread_x()));
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
KERNEL_THREADS(THREADS) void k_refine_revert(GLOBAL const u32* RESTRICT flags, GLOBAL R* RESTRICT x, GLOBAL const R* RESTRICT x_prev, u32 n) {
    u32 i = global_x();
    if (flags[F_REVERT] != 0 && i < n) x[i] = x_prev[i];
}
