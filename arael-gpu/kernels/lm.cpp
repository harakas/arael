// lm.cpp: the Levenberg-Marquardt loop's vector work on the device, launched
// by arael-gpu/src/native/lm.rs (the passes of arael-gpu/src/lm.rs).
//
// In W, the route's wide scalar (wide.h). Every kernel runs THREADS threads per block. The scalars slots
// (S_*), the partials per block (LIN_PARTIALS, ATTEMPT_PARTIALS), NO_FAULT
// and FIXED come from lm.rs as #defines. A pass leaves partials per block,
// which the combine kernels fold, in one block, into the scalars.

#define OP_SUM 0
#define OP_MAX 1
#define OP_MIN 2

template <int OP> DEVICE W combine(W a, W b) {
    if (OP == OP_SUM) return a + b;
    if (OP == OP_MAX) return b > a ? b : a;
    return b < a ? b : a;
}

// `v` combined over the block by OP, the result in every thread; `sh` is
// THREADS words of shared memory.
template <int OP> DEVICE W block_reduce(W v, SHARED_PTR W* sh) {
    u32 t = thread_x();
    // a previous call's readers are done with the shared words first
    BARRIER();
    sh[t] = v;
    BARRIER();
    for (u32 half = THREADS / 2; half > 0; half /= 2) {
        if (t < half) sh[t] = combine<OP>(sh[t], sh[t + half]);
        BARRIER();
    }
    return sh[0];
}

// H's diagonal gathered: out[i] = h[pos[i]], one thread per scalar.
KERNEL_THREADS(THREADS) void k_gather_diag(GLOBAL const W* RESTRICT h, GLOBAL const u32* RESTRICT pos, GLOBAL W* RESTRICT out, u32 n) {
    u32 i = global_x();
    if (i < n) out[i] = h[pos[i]];
}

// One thread per parameter: H's diagonal diag[i] = h[pos[i]], the damping
// scale (max(diag, floor) when has_floor, else the diagonal), and per block
// the gradient's largest magnitude, the largest |g| / sqrt(d) over d > 0,
// and the first faulty entry as 4 i + kind (1 NaN, 2 negative, 3 zero; with
// a floor a NaN or negative diagonal, without one also a zero).
KERNEL_THREADS(THREADS) void k_lin_pass(GLOBAL const W* RESTRICT g, GLOBAL const W* RESTRICT h, GLOBAL const u32* RESTRICT pos, GLOBAL W* RESTRICT diag, GLOBAL W* RESTRICT damp, W floor, u32 has_floor, GLOBAL W* RESTRICT partials, u32 n) {
    SHARED W sh[THREADS];
    u32 i = global_x();
    W gm = 0.0f;
    W gs = 0.0f;
    W fault = NO_FAULT;
    if (i < n) {
        W d = h[pos[i]];
        diag[i] = d;
        W gi = fabs(g[i]);
        gm = gi;
        if (d > 0.0f) gs = gi / sqrt(d);
        bool bad = !(d > 0.0f);
        if (has_floor != 0) {
            bad = !(d >= 0.0f);
            damp[i] = d > floor ? d : floor;
        } else {
            damp[i] = d;
        }
        if (bad) {
            u32 kind = 3;
            if (isnan(d)) {
                kind = 1;
            } else if (d < 0.0f) {
                kind = 2;
            }
            fault = wide_u32(4u * i + kind);
        }
    }
    gm = block_reduce<OP_MAX>(gm, sh);
    gs = block_reduce<OP_MAX>(gs, sh);
    fault = block_reduce<OP_MIN>(fault, sh);
    if (thread_x() == 0) {
        u32 c = block_x() * LIN_PARTIALS;
        partials[c] = gm;
        partials[c + 1] = gs;
        partials[c + 2] = fault;
    }
}

// The damped diagonal written into H's values: h[pos[i]] = diag[i] +
// lambda[0] damp[i], one thread per scalar.
KERNEL_THREADS(THREADS) void k_damp(GLOBAL W* RESTRICT h, GLOBAL const u32* RESTRICT pos, GLOBAL const W* RESTRICT diag, GLOBAL const W* RESTRICT damp, GLOBAL const W* RESTRICT lambda, u32 n) {
    u32 i = global_x();
    if (i < n) h[pos[i]] = diag[i] + lambda[0] * damp[i];
}

// One thread per parameter: the trial point x_try = x - delta, and per block
// the sums of delta^2, x_try^2 and delta (g + lambda[0] d delta) (twice the
// predicted reduction).
KERNEL_THREADS(THREADS) void k_attempt_pass(GLOBAL const W* RESTRICT x, GLOBAL const W* RESTRICT delta, GLOBAL const W* RESTRICT g, GLOBAL const W* RESTRICT diag, GLOBAL const W* RESTRICT lambda, GLOBAL W* RESTRICT x_try, GLOBAL W* RESTRICT partials, u32 n) {
    SHARED W sh[THREADS];
    u32 i = global_x();
    W s2 = 0.0f;
    W t2 = 0.0f;
    W pr = 0.0f;
    if (i < n) {
        W d = delta[i];
        W t = x[i] - d;
        x_try[i] = t;
        s2 = d * d;
        t2 = t * t;
        pr = d * (g[i] + lambda[0] * diag[i] * d);
    }
    s2 = block_reduce<OP_SUM>(s2, sh);
    t2 = block_reduce<OP_SUM>(t2, sh);
    pr = block_reduce<OP_SUM>(pr, sh);
    if (thread_x() == 0) {
        u32 c = block_x() * ATTEMPT_PARTIALS;
        partials[c] = s2;
        partials[c + 1] = t2;
        partials[c + 2] = pr;
    }
}

// Entry k of the partials of `blocks` blocks (PER a block) folded by OP over
// the threads of one block, the result in every thread.
template <int OP, u32 PER, u32 K> DEVICE W fold_partials(GLOBAL const W* partials, u32 blocks, SHARED_PTR W* sh) {
    W acc = OP == OP_MIN ? NO_FAULT : W(0.0f);
    for (u32 c = thread_x(); c < blocks; c += THREADS) acc = combine<OP>(acc, partials[c * PER + K]);
    return block_reduce<OP>(acc, sh);
}

// k_lin_pass's partials folded into the scalars, one block.
KERNEL_THREADS(THREADS) void k_combine_lin(GLOBAL const W* RESTRICT partials, u32 blocks, GLOBAL W* RESTRICT out) {
    SHARED W sh[THREADS];
    W gm = fold_partials<OP_MAX, LIN_PARTIALS, 0>(partials, blocks, sh);
    W gs = fold_partials<OP_MAX, LIN_PARTIALS, 1>(partials, blocks, sh);
    W fault = fold_partials<OP_MIN, LIN_PARTIALS, 2>(partials, blocks, sh);
    if (thread_x() == 0) {
        out[S_GRAD_MAX] = gm;
        out[S_GRAD_SCALED] = gs;
        out[S_FAULT] = fault;
    }
}

// k_attempt_pass's partials folded into the scalars, one block; the
// predicted reduction is half its sum.
KERNEL_THREADS(THREADS) void k_combine_attempt(GLOBAL const W* RESTRICT partials, u32 blocks, GLOBAL W* RESTRICT out) {
    SHARED W sh[THREADS];
    W s2 = fold_partials<OP_SUM, ATTEMPT_PARTIALS, 0>(partials, blocks, sh);
    W t2 = fold_partials<OP_SUM, ATTEMPT_PARTIALS, 1>(partials, blocks, sh);
    W pr = fold_partials<OP_SUM, ATTEMPT_PARTIALS, 2>(partials, blocks, sh);
    if (thread_x() == 0) {
        out[S_STEP2] = s2;
        out[S_TRY2] = t2;
        out[S_PREDICTED] = 0.5f * pr;
    }
}

// dst = src, n values.
KERNEL_THREADS(THREADS) void k_copy(GLOBAL W* RESTRICT dst, GLOBAL const W* RESTRICT src, u32 n) {
    u32 i = global_x();
    if (i < n) dst[i] = src[i];
}

// The rotation steps of x zeroed, one thread per entity: the three global
// indices at idx[at[e]..], a fixed one left alone; what re-centring does to
// the parameters after an accepted step.
KERNEL_THREADS(THREADS) void k_zero_steps(GLOBAL W* RESTRICT x, GLOBAL const u32* RESTRICT idx, GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
#pragma unroll
    for (u32 k = 0; k < 3; k++) {
        u32 g = idx[a + k];
        if (g != FIXED) x[g] = 0.0f;
    }
}
