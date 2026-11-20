// schur.cpp: the Schur reduction on the device and the refinement of its
// reduced solve (docs/dev/GPU.md, E2 and E9), launched by
// arael-gpu/src/schur.rs over its plan. Compiled after schur.h, which
// declares the plan's header, R and the functions shared with cg.cpp.
//
// H, the gradient and the step are in V, the route's values (wide.h); the
// reduction computes in R (schur.h) and keeps its vectors in R; the
// refinement's norms and state are in W, the route's wide scalar. Compiled per EWX (the widest
// eliminated block), EW (it rounded up to 4, 8, 16, 32 or 64), KW (the
// widest kept block, rounded likewise), TILE_LANES (k_schur_tiles' threads
// per unit: the device's lockstep width, 32 or 64), TILE_WARPS (its units
// per block), TILE_GROUPS (a unit's lane groups), ACC (the widest kept
// block's square over a group's lanes, rounded up: the entries of a tile of
// S per lane), and RWARPS and SWARPS (the warps a kept block takes in
// k_schur_rhs and in k_symv_sub);
// the plan header's slots (SE, SO, ST_*, OBS_KEPT, SK_*, SB_*; SHEADER of
// them), the words of
// an eliminated block's record (SE_*, SE_REC of them, in the table at SE),
// of an observer's (SO_*, SO_REC, at SO) and of the expanded records of a
// pair, of a kept block's observer
// and of a kept block's tile entry (PX_*, PX_REC, KO_*, KO_REC, KT_*,
// KT_REC, in a block of their own), the refinement
// state's words (R_*) and flags (F_*) come from schur.rs as
// #defines. Kernels launched with THREADS threads per block, or PLANE where
// said.
//
// Per-thread arrays are indexed only by constants once their loops unroll:
// the loops run to EW or KW and skip what is past the block's width, so the
// arrays stay in registers (arael_gpu.h's rule, from k_trsm).

KERNEL_THREADS(PLANE) void k_zero_u32(GLOBAL u32* RESTRICT buf, u32 n) {
    u32 i = global_x();
    if (i < n) buf[i] = 0;
}

// Stage 1, one thread per eliminated block: the factor of its damped
// diagonal tile (upper-stored) into `dfac` (see ldlt_solve), from its
// Cholesky factor, and z = D^-1 b. A pivot that is not positive sets
// fail[0] to the block's slot plus one and continues with 1.
KERNEL_THREADS_REGS(THREADS) void k_schur_elim(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const V* RESTRICT h, GLOBAL const V* RESTRICT rhs, GLOBAL R* RESTRICT dfac, GLOBAL R* RESTRICT z, GLOBAL u32* RESTRICT fail, u32 ne) {
    u32 e = global_x();
    if (e >= ne) return;
    GLOBAL const u32* se = plan + H.w[SE] + SE_REC * e;
    u32 w = se[SE_W];
    u32 diag = se[SE_DIAG_OFF];
    u32 orig = se[SE_ORIG_OFF];
    u32 doff = se[SE_DFAC_OFF];
    u32 zoff = se[SE_Z_OFF];
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
KERNEL_THREADS(THREADS) void k_schur_w(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const V* RESTRICT h, GLOBAL const R* RESTRICT dfac, GLOBAL R* RESTRICT w, u32 no) {
    u32 x = global_x();
    u32 o = x / KW;
    u32 i = x - o * KW;
    if (o >= no) return;
    GLOBAL const u32* so = plan + H.w[SO] + SO_REC * o;
    u32 dims = so[SO_DIMS];
    u32 wo = DIMS_WO(dims);
    if (i >= wo) return;
    u32 we = DIMS_WE(dims);
    u32 trans = DIMS_TRANS(dims);
    u32 ca = so[SO_CA_OFF];
    u32 doff = plan[H.w[SE] + SE_REC * so[SO_E] + SE_DFAC_OFF];
    u32 woff = so[SO_W_OFF];
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

// Each pair's expanded record (PX_REC words) for k_schur_tiles, grouped by
// tile of S from the tile's first pair (ST_PAIR_PTR): of its observer of
// the tile's row block (oa) the value offset of W in w, of its observer of
// the column block (ob) the value offset of its coupling tile in h and its
// sizes. Once per structure, a thread a pair: its eliminated block found
// among the blocks' first pairs (SE_PAIR_PTR), within the block its k-th
// pair, by ob then oa: (oa, ob) = (first + i, first + j) with k = j (j + 1)
// / 2 + i, i <= j (so kept(oa) <= kept(ob)); the tile of row kept(oa)
// found in column kept(ob)'s tiles (SK_T_PTR, their rows ascending in
// ST_A), the pair's place in the tile taken from the tile's counter (cnt,
// zeroed): a tile's pairs in no fixed order.
KERNEL_THREADS(THREADS) void k_schur_pairs(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL u32* RESTRICT cnt, GLOBAL u32* RESTRICT px, u32 np, u32 ne) {
    SHARED u32 first_e[1];
    GLOBAL const u32* pp = plan + H.w[SE_PAIR_PTR];
    // the block's first pair's eliminated block: the last whose first pair
    // is at or before it, by one thread; the threads walk on from it
    if (thread_x() == 0) {
        u32 p0 = block_x() * THREADS;
        u32 lo = 0, hi = ne;
        while (hi - lo > 1) {
            u32 mid = (lo + hi) / 2;
            if (pp[mid] <= p0) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        first_e[0] = lo;
    }
    BARRIER();
    u32 p = global_x();
    if (p >= np) return;
    u32 e = first_e[0];
    while (pp[e + 1] <= p) e++;
    u32 k = p - pp[e];
    u32 j = u32((sqrt(8.0f * float(k) + 1.0f) - 1.0f) * 0.5f);
    while ((j + 1) * (j + 2) / 2 <= k) j++;
    while (j * (j + 1) / 2 > k) j--;
    u32 first = plan[H.w[SE] + SE_REC * e + SE_OBS_PTR];
    u32 oa = first + k - j * (j + 1) / 2;
    u32 ob = first + j;
    u32 a = plan[H.w[OBS_KEPT] + oa];
    u32 b = plan[H.w[OBS_KEPT] + ob];
    // the column's first tile of a row not above a: row a's
    GLOBAL const u32* rows = plan + H.w[ST_A];
    u32 lo = plan[H.w[SK_T_PTR] + b], end = plan[H.w[SK_T_PTR] + b + 1];
    while (lo < end) {
        u32 mid = (lo + end) / 2;
        if (rows[mid] < a) {
            lo = mid + 1;
        } else {
            end = mid;
        }
    }
    GLOBAL const u32* rb = plan + H.w[SO] + SO_REC * ob;
    u32 at = plan[H.w[ST_PAIR_PTR] + lo] + atomic_add(cnt + lo, 1u);
    px[PX_REC * at + PX_W_OFF] = plan[H.w[SO] + SO_REC * oa + SO_W_OFF];
    px[PX_REC * at + PX_C_OFF] = rb[SO_CA_OFF];
    px[PX_REC * at + PX_DIMS] = rb[SO_DIMS];
}

// df32: registers uncapped, as capped to fit more blocks the accumulators of
// several groups spill (CUDA); f64 and f32 run faster capped.
#if R_DF32
#define TILES_ENTRY KERNEL_THREADS_REGS(TILE_LANES * TILE_WARPS)
#else
#define TILES_ENTRY KERNEL_THREADS(TILE_LANES * TILE_WARPS)
#endif
TILES_ENTRY void k_schur_tiles(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const u32* RESTRICT pairs, GLOBAL const V* RESTRICT h, GLOBAL const R* RESTRICT w, GLOBAL R* RESTRICT s) {
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
    GLOBAL const u32* item = plan + H.w[SB_ITEMS] + (TILE_WARPS + 1) * block_x();
    bool split = item[0] != 0;
    u32 t = warp_uniform(item[1 + (split ? 0 : unit)]);
    // a unit without a tile ends here; a split block has no idle unit, and
    // only it waits for the whole block
    if (t == NONE) return;
    SHARED_PTR R* ws = ws_all + (unit * TILE_GROUPS + g) * KW * EW;
    SHARED_PTR R* cs = cs_all + (unit * TILE_GROUPS + g) * KW * EW;
    u32 val = plan[H.w[ST_VAL_OFF] + t];
    u32 wr = plan[H.w[ST_WR] + t];
    u32 wc = plan[H.w[ST_WC] + t];
    u32 a = plan[H.w[ST_A] + t];
    u32 b = plan[H.w[ST_B] + t];
    u32 copy = plan[H.w[ST_COPY_OFF] + t];
    u32 p0 = plan[H.w[ST_PAIR_PTR] + t];
    u32 p1 = plan[H.w[ST_PAIR_PTR] + t + 1];
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
        // ob observes the tile's column block: C_ob has wc rows
        u32 wb = 0u;
        if (p < p1) {
            // the pair's expanded record (k_schur_pairs): W_oa's place, C_ob's
            // and ob's sizes (both observe the pair's eliminated block)
            u32 pr = PX_REC * p;
            woff = pairs[pr + PX_W_OFF];
            cb = pairs[pr + PX_C_OFF];
            u32 db = pairs[pr + PX_DIMS];
            we = DIMS_WE(db);
            tb = DIMS_TRANS(db);
            wb = wc;
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

// Each entry of the kept blocks' observer lists (SK_OBS) expanded for
// k_schur_rhs (KO_REC words, in the list's order): the observer's sizes, its
// coupling tile's value offset in h, its eliminated block's value offset in
// z. Once per structure, a thread an entry.
KERNEL_THREADS(THREADS) void k_schur_kobs(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL u32* RESTRICT ko, u32 n) {
    u32 q = global_x();
    if (q >= n) return;
    u32 ro = H.w[SO] + SO_REC * plan[H.w[SK_OBS] + q];
    ko[KO_REC * q + KO_DIMS] = plan[ro + SO_DIMS];
    ko[KO_REC * q + KO_C_OFF] = plan[ro + SO_CA_OFF];
    ko[KO_REC * q + KO_Z_OFF] = plan[H.w[SE] + SE_REC * plan[ro + SO_E] + SE_Z_OFF];
}

// Stage 4, a block of threads per kept block k: rhs_kept = b_k - sum over
// the block's observers of C_o z_e (each observer's expanded record).
#if KW <= PLANE
// The lanes over the block's rows: RWARPS warps a block, each warp MGROUPS
// groups of KW lanes (schur.h). A group takes every (RWARPS MGROUPS)-th of
// k's observers, its lane `row` the observer's contribution to row `row`,
// so the lanes of a group read neighbouring values of the coupling tile at
// every step and the same value of z.
KERNEL_THREADS(RWARPS * PLANE) void k_schur_rhs(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const u32* RESTRICT kobs, GLOBAL const V* RESTRICT h, GLOBAL const V* RESTRICT rhs, GLOBAL const R* RESTRICT z, GLOBAL R* RESTRICT out) {
#if RWARPS > 1
    SHARED R sh[RWARPS * KW];
#endif
    u32 k = block_x();
    u32 warp = thread_x() / PLANE;
    u32 lane = thread_x() % PLANE;
    u32 row = lane % KW;
    u32 s_off = plan[H.w[SK_S_OFF] + k];
    u32 orig = plan[H.w[SK_ORIG_OFF] + k];
    u32 wk = plan[H.w[SK_W] + k];
    u32 q1 = plan[H.w[SK_OBS_PTR] + k + 1];
    R acc = 0.0f;
    for (u32 q = plan[H.w[SK_OBS_PTR] + k] + warp * MGROUPS + lane / KW; q < q1; q += RWARPS * MGROUPS) {
        // the observer's expanded record (k_schur_kobs): its sizes, its
        // coupling tile's offset, its eliminated block's z offset
        u32 ro = KO_REC * q;
        u32 dims = kobs[ro + KO_DIMS];
        u32 we = DIMS_WE(dims);
        u32 ca = kobs[ro + KO_C_OFF];
        u32 trans = DIMS_TRANS(dims);
        u32 zoff = kobs[ro + KO_Z_OFF];
        if (row < wk)
            for (u32 kk = 0; kk < we; kk++) acc = r_fma(acc, to_r(coupling(h, ca, trans, wk, we, row, kk)), z[zoff + kk]);
    }
    // a row's sums of the warp's groups (lanes row, row + KW, ...), then of
    // the block's warps
#pragma unroll
    for (u32 m = KW; m < PLANE; m *= 2) acc += shfl_xor(acc, m);
#if RWARPS > 1
    if (lane < KW) sh[warp * KW + lane] = acc;
    BARRIER();
    if (warp != 0) return;
    if (lane < KW) {
        acc = sh[lane];
#pragma unroll
        for (u32 w = 1; w < RWARPS; w++) acc += sh[w * KW + lane];
    }
#endif
    if (lane < wk) out[s_off + lane] = to_r(rhs[orig + lane]) - acc;
}
#else
// Kept blocks wider than a plane has lanes: one warp a block, the lanes
// over the observers, each lane all the rows, and a warp sum per row.
KERNEL_THREADS(PLANE) void k_schur_rhs(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const u32* RESTRICT kobs, GLOBAL const V* RESTRICT h, GLOBAL const V* RESTRICT rhs, GLOBAL const R* RESTRICT z, GLOBAL R* RESTRICT out) {
    u32 k = block_x();
    u32 lane = thread_x();
    u32 s_off = plan[H.w[SK_S_OFF] + k];
    u32 orig = plan[H.w[SK_ORIG_OFF] + k];
    u32 wk = plan[H.w[SK_W] + k];
    u32 q0 = plan[H.w[SK_OBS_PTR] + k];
    u32 q1 = plan[H.w[SK_OBS_PTR] + k + 1];
    R acc[KW];
#pragma unroll
    for (u32 i = 0; i < KW; i++) acc[i] = 0.0f;
    for (u32 q = q0 + lane; q < q1; q += PLANE) {
        // the observer's expanded record (k_schur_kobs): its sizes, its
        // coupling tile's offset, its eliminated block's z offset
        u32 ro = KO_REC * q;
        u32 dims = kobs[ro + KO_DIMS];
        u32 we = DIMS_WE(dims);
        u32 ca = kobs[ro + KO_C_OFF];
        u32 trans = DIMS_TRANS(dims);
        u32 zoff = kobs[ro + KO_Z_OFF];
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
#endif

// The kept blocks of the reduced solution (R) into the full vector (V), one
// thread per kept block.
KERNEL_THREADS(THREADS) void k_scatter_kept(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const R* RESTRICT xk, GLOBAL V* RESTRICT x, u32 nk) {
    u32 k = global_x();
    if (k >= nk) return;
    u32 s_off = plan[H.w[SK_S_OFF] + k];
    u32 orig = plan[H.w[SK_ORIG_OFF] + k];
    u32 wk = plan[H.w[SK_W] + k];
    for (u32 i = 0; i < wk; i++) x[orig + i] = (V)xk[s_off + i];
}

// Back-substitution, one thread per eliminated block: x_e = D_e^-1 (b_e -
// sum_a C_a^T x_a) over its observers (ldlt_solve), the kept slices of `x`
// in place.
KERNEL_THREADS(THREADS) void k_schur_backsub(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const V* RESTRICT h, GLOBAL const V* RESTRICT rhs, GLOBAL const R* RESTRICT dfac, GLOBAL V* RESTRICT x, u32 ne) {
    u32 e = global_x();
    if (e >= ne) return;
    // the block's record; its observers end where the next block's start
    GLOBAL const u32* se = plan + H.w[SE] + SE_REC * e;
    GLOBAL const u32* so = plan + H.w[SO];
    u32 w = se[SE_W];
    u32 orig = se[SE_ORIG_OFF];
    u32 doff = se[SE_DFAC_OFF];
    u32 q0 = se[SE_OBS_PTR];
    u32 q1 = se[SE_REC + SE_OBS_PTR];
    R t[EW];
#pragma unroll
    for (u32 i = 0; i < EW; i++)
        if (i < w) t[i] = to_r(rhs[orig + i]);
    for (u32 o = q0; o < q1; o++) {
        GLOBAL const u32* r = so + SO_REC * o;
        u32 dims = r[SO_DIMS];
        u32 wa = DIMS_WO(dims);
        u32 ca = r[SO_CA_OFF];
        u32 trans = DIMS_TRANS(dims);
        u32 xo = r[SO_ORIG_OFF];
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

// Each entry of the kept blocks' tile lists (kt: a tile index times 2, plus
// 1 when the block is the tile's column block) expanded for k_symv_sub
// (KT_REC words, in the list's order): the tile's value offset in s, its
// sizes word, the other block's offset in the reduced vector. Once per
// structure, a thread an entry.
KERNEL_THREADS(THREADS) void k_schur_kt(GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const u32* RESTRICT kt, GLOBAL u32* RESTRICT out, u32 n) {
    u32 q = global_x();
    if (q >= n) return;
    u32 t = kt[q] >> 1;
    u32 trans = kt[q] & 1u;
    u32 a = plan[H.w[ST_A] + t];
    u32 b = plan[H.w[ST_B] + t];
    out[KT_REC * q + KT_VAL_OFF] = plan[H.w[ST_VAL_OFF] + t];
    out[KT_REC * q + KT_DIMS] = KT_PACK(plan[H.w[ST_WR] + t], plan[H.w[ST_WC] + t], trans, a == b ? 1u : 0u);
    out[KT_REC * q + KT_X_OFF] = plan[H.w[SK_S_OFF] + (trans ? a : b)];
}

// r -= S x over the tiles of S (upper block storage, diagonal tiles
// upper-only), a block of threads per kept block k owning rows k of r, over
// k's tile entries (kt_ptr[k]..kt_ptr[k + 1], KT_REC words each: the
// tile's value offset, its sizes word, the other block's offset in x; a
// tile whose column block is k transposed gives the rows); nothing unless
// gate[0] is nonzero. No two blocks write the same entry. The sums are in W.
#if KW <= PLANE
// The lanes over the block's rows: SWARPS warps a block, each warp MGROUPS
// groups of KW lanes (schur.h). A group takes every (SWARPS MGROUPS)-th of
// k's tile entries, its lane `row` the tile's contribution to row `row`,
// so the lanes of a group read neighbouring values of S at every step and
// the same value of x.
KERNEL_THREADS(SWARPS * PLANE) void k_symv_sub(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const u32* RESTRICT kt_ptr, GLOBAL const u32* RESTRICT kt, GLOBAL const R* RESTRICT s, GLOBAL const R* RESTRICT x, GLOBAL R* RESTRICT r) {
#if SWARPS > 1
    SHARED W sh[SWARPS * KW];
#endif
    if (gate[0] == 0) return;
    u32 k = block_x();
    u32 warp = thread_x() / PLANE;
    u32 lane = thread_x() % PLANE;
    u32 row = lane % KW;
    u32 k_off = plan[H.w[SK_S_OFF] + k];
    u32 wk = plan[H.w[SK_W] + k];
    u32 end = kt_ptr[k + 1];
    W acc = 0.0f;
    for (u32 q = kt_ptr[k] + warp * MGROUPS + lane / KW; q < end; q += SWARPS * MGROUPS) {
        u32 val = kt[KT_REC * q + KT_VAL_OFF];
        u32 dims = kt[KT_REC * q + KT_DIMS];
        u32 xo = kt[KT_REC * q + KT_X_OFF];
        u32 wr = KT_WR(dims);
        if (row < wk) {
            if (KT_TRANS(dims) == 0) {
                // row `row` of (k, b): S_kb x_b, a diagonal tile read from its upper triangle
                u32 wc = KT_WC(dims);
                u32 diag = KT_DIAG(dims);
                for (u32 j = 0; j < wc; j++) acc = wide_fma(acc, W(diag && row > j ? s[val + j + row * wr] : s[val + row + j * wr]), W(x[xo + j]));
            } else {
                // row `row` of (a, k)^T: S_ak^T x_a
                for (u32 i2 = 0; i2 < wr; i2++) acc = wide_fma(acc, W(s[val + i2 + row * wr]), W(x[xo + i2]));
            }
        }
    }
    // a row's sums of the warp's groups (lanes row, row + KW, ...), then of
    // the block's warps
#pragma unroll
    for (u32 m = KW; m < PLANE; m *= 2) acc += shfl_xor(acc, m);
#if SWARPS > 1
    if (lane < KW) sh[warp * KW + lane] = acc;
    BARRIER();
    if (warp != 0) return;
    if (lane < KW) {
        acc = sh[lane];
#pragma unroll
        for (u32 w = 1; w < SWARPS; w++) acc += sh[w * KW + lane];
    }
#endif
    if (lane < wk) r[k_off + lane] = (R)(W(r[k_off + lane]) - acc);
}
#else
// Kept blocks wider than a plane has lanes: one warp a block, the lanes
// over k's tile entries, each lane all the rows, and a warp sum per row.
KERNEL_THREADS(PLANE) void k_symv_sub(GLOBAL const u32* RESTRICT gate, GLOBAL const u32* RESTRICT plan, TABLE_PARAM(SchurHeader) H, GLOBAL const u32* RESTRICT kt_ptr, GLOBAL const u32* RESTRICT kt, GLOBAL const R* RESTRICT s, GLOBAL const R* RESTRICT x, GLOBAL R* RESTRICT r) {
    if (gate[0] == 0) return;
    u32 k = block_x();
    u32 lane = thread_x();
    u32 k_off = plan[H.w[SK_S_OFF] + k];
    u32 wk = plan[H.w[SK_W] + k];
    W acc[KW];
#pragma unroll
    for (u32 i = 0; i < KW; i++) acc[i] = 0.0f;
    for (u32 q = kt_ptr[k] + lane; q < kt_ptr[k + 1]; q += PLANE) {
        u32 val = kt[KT_REC * q + KT_VAL_OFF];
        u32 dims = kt[KT_REC * q + KT_DIMS];
        u32 xo = kt[KT_REC * q + KT_X_OFF];
        u32 wr = KT_WR(dims);
        if (KT_TRANS(dims) == 0) {
            // rows of (k, b): S_kb x_b, a diagonal tile read from its upper triangle
            u32 wc = KT_WC(dims);
            u32 diag = KT_DIAG(dims);
#pragma unroll
            for (u32 i = 0; i < KW; i++) {
                if (i < wk) {
                    W v = 0.0f;
                    for (u32 j = 0; j < wc; j++) v = wide_fma(v, W(diag && i > j ? s[val + j + i * wr] : s[val + i + j * wr]), W(x[xo + j]));
                    acc[i] += v;
                }
            }
        } else {
            // rows of (a, k)^T: S_ak^T x_a
#pragma unroll
            for (u32 j = 0; j < KW; j++) {
                if (j < wk) {
                    W v = 0.0f;
                    for (u32 i2 = 0; i2 < wr; i2++) v = wide_fma(v, W(s[val + i2 + j * wr]), W(x[xo + i2]));
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
#endif

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
// a step is due. A norm that is not a number starts no step; the host reads
// it as short of the tolerance.
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
// updated iterate: a step that did not reduce the residual's norm (a norm
// that is not a number included) is to be undone and ends the refinement;
// otherwise the norm is kept and the refinement goes on while it is above
// the tolerance.
KERNEL_THREADS(THREADS) void k_refine_judge(GLOBAL const R* RESTRICT r, GLOBAL W* RESTRICT st, GLOBAL u32* RESTRICT flags, u32 n) {
    SHARED W sh[THREADS];
    // every thread reads the flag before the first barrier, thread 0 writes
    // it after the last
    if (flags[F_ACTIVE] == 0) return;
    W next = sqrt(block_sumsq(r, n, sh, thread_x()));
    if (thread_x() == 0) {
        flags[F_STEPS] += 1;
        if (!(next < st[R_RNORM])) {
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
