// lm.cpp: the Levenberg-Marquardt loop's vector work on the device, launched
// by arael-gpu/src/lm.rs.
//
// H, the gradient and the loop's vectors (the parameters, the trial point,
// the step, H's diagonal, the damping scale and lambda) are in V, the
// route's values (wide.h), as the CPU's route keeps them; the partials and
// the scalars are words of W, its wide scalar, and the step's and the trial
// point's sums of squares are summed in W, as the CPU sums them in f64.
// Every kernel runs THREADS threads per block. The scalars slots (S_*), the
// partials per block (LIN_PARTIALS, ATTEMPT_PARTIALS), NO_FAULT and FIXED
// come from lm.rs as #defines. A pass leaves partials per block, which the
// combine kernels fold, in one block, into the scalars.

#define OP_SUM 0
#define OP_MAX 1
#define OP_MIN 2

template <int OP, typename T> DEVICE T combine(T a, T b) {
    if (OP == OP_SUM) return a + b;
    if (OP == OP_MAX) return b > a ? b : a;
    return b < a ? b : a;
}

// `v` combined over the block by OP, the result in every thread; `sh` is
// THREADS words of shared memory, t the thread.
template <int OP, typename T> DEVICE T block_reduce(T v, SHARED_PTR T* sh, u32 t) {
    // a previous call's readers are done with the shared words first
    BARRIER();
    sh[t] = v;
    BARRIER();
    for (u32 span = THREADS / 2; span > 0; span /= 2) {
        if (t < span) sh[t] = combine<OP>(sh[t], sh[t + span]);
        BARRIER();
    }
    return sh[0];
}

// H's diagonal gathered: out[i] = h[pos[i]], one thread per scalar.
KERNEL_THREADS(THREADS) void k_gather_diag(GLOBAL const V* RESTRICT h, GLOBAL const u32* RESTRICT pos, GLOBAL V* RESTRICT out, u32 n) {
    u32 i = global_x();
    if (i < n) out[i] = h[pos[i]];
}

// One thread per parameter: H's diagonal diag[i] = h[pos[i]], the damping
// scale (max(diag, floor) when has_floor, else the diagonal), and per block
// the gradient's largest magnitude, the largest |g| / sqrt(d) over d > 0,
// and the first faulty entry as 4 i + kind (1 NaN, 2 negative, 3 zero; with
// a floor a NaN or negative diagonal, without one also a zero).
KERNEL_THREADS(THREADS) void k_lin_pass(GLOBAL const V* RESTRICT g, GLOBAL const V* RESTRICT h, GLOBAL const u32* RESTRICT pos, GLOBAL V* RESTRICT diag, GLOBAL V* RESTRICT damp, V floor, u32 has_floor, GLOBAL W* RESTRICT partials, u32 n) {
    SHARED V shv[THREADS];
    SHARED W shw[THREADS];
    u32 i = global_x();
    V gm = 0.0f;
    V gs = 0.0f;
    W fault = NO_FAULT;
    if (i < n) {
        V d = h[pos[i]];
        diag[i] = d;
        V gi = fabs(g[i]);
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
    gm = block_reduce<OP_MAX>(gm, shv, thread_x());
    gs = block_reduce<OP_MAX>(gs, shv, thread_x());
    fault = block_reduce<OP_MIN>(fault, shw, thread_x());
    if (thread_x() == 0) {
        u32 c = block_x() * LIN_PARTIALS;
        partials[c] = W(gm);
        partials[c + 1] = W(gs);
        partials[c + 2] = fault;
    }
}

// The damped diagonal written into H's values: h[pos[i]] = diag[i] +
// lambda[0] damp[i], one thread per scalar.
KERNEL_THREADS(THREADS) void k_damp(GLOBAL V* RESTRICT h, GLOBAL const u32* RESTRICT pos, GLOBAL const V* RESTRICT diag, GLOBAL const V* RESTRICT damp, GLOBAL const V* RESTRICT lambda, u32 n) {
    u32 i = global_x();
    if (i < n) h[pos[i]] = diag[i] + lambda[0] * damp[i];
}

// One thread per parameter: the trial point x_try = x - delta, and per block
// the sums of delta^2 and x_try^2 (in W) and of delta (g + lambda[0] d
// delta) (twice the predicted reduction, in V).
KERNEL_THREADS(THREADS) void k_attempt_pass(GLOBAL const V* RESTRICT x, GLOBAL const V* RESTRICT delta, GLOBAL const V* RESTRICT g, GLOBAL const V* RESTRICT diag, GLOBAL const V* RESTRICT lambda, GLOBAL V* RESTRICT x_try, GLOBAL W* RESTRICT partials, u32 n) {
    SHARED V shv[THREADS];
    SHARED W shw[THREADS];
    u32 i = global_x();
    W s2 = 0.0f;
    W t2 = 0.0f;
    V pr = 0.0f;
    if (i < n) {
        V d = delta[i];
        V t = x[i] - d;
        x_try[i] = t;
        s2 = W(d) * W(d);
        t2 = W(t) * W(t);
        pr = d * (g[i] + lambda[0] * diag[i] * d);
    }
    s2 = block_reduce<OP_SUM>(s2, shw, thread_x());
    t2 = block_reduce<OP_SUM>(t2, shw, thread_x());
    pr = block_reduce<OP_SUM>(pr, shv, thread_x());
    if (thread_x() == 0) {
        u32 c = block_x() * ATTEMPT_PARTIALS;
        partials[c] = s2;
        partials[c + 1] = t2;
        partials[c + 2] = W(pr);
    }
}

// Entry K of the partials of `blocks` blocks (PER a block) folded by OP in
// T from `acc`, OP's identity, over the threads of one block (t the
// thread), the result in every thread.
template <int OP, u32 PER, u32 K, typename T> DEVICE T fold_partials(GLOBAL const W* partials, u32 blocks, SHARED_PTR T* sh, u32 t, T acc) {
    for (u32 c = t; c < blocks; c += THREADS) acc = combine<OP>(acc, (T)partials[c * PER + K]);
    return block_reduce<OP>(acc, sh, t);
}

// k_lin_pass's partials folded into the scalars, one block (the maxima are
// exact in W).
KERNEL_THREADS(THREADS) void k_combine_lin(GLOBAL const W* RESTRICT partials, u32 blocks, GLOBAL W* RESTRICT out) {
    SHARED W sh[THREADS];
    W zero = 0.0f;
    W gm = fold_partials<OP_MAX, LIN_PARTIALS, 0>(partials, blocks, sh, thread_x(), zero);
    W gs = fold_partials<OP_MAX, LIN_PARTIALS, 1>(partials, blocks, sh, thread_x(), zero);
    W fault = fold_partials<OP_MIN, LIN_PARTIALS, 2>(partials, blocks, sh, thread_x(), W(NO_FAULT));
    if (thread_x() == 0) {
        out[S_GRAD_MAX] = gm;
        out[S_GRAD_SCALED] = gs;
        out[S_FAULT] = fault;
    }
}

// k_attempt_pass's partials folded into the scalars, one block; the
// predicted reduction is half its sum, in V.
KERNEL_THREADS(THREADS) void k_combine_attempt(GLOBAL const W* RESTRICT partials, u32 blocks, GLOBAL W* RESTRICT out) {
    SHARED V shv[THREADS];
    SHARED W shw[THREADS];
    W s2 = fold_partials<OP_SUM, ATTEMPT_PARTIALS, 0>(partials, blocks, shw, thread_x(), W(0.0f));
    W t2 = fold_partials<OP_SUM, ATTEMPT_PARTIALS, 1>(partials, blocks, shw, thread_x(), W(0.0f));
    V pr = fold_partials<OP_SUM, ATTEMPT_PARTIALS, 2>(partials, blocks, shv, thread_x(), V(0.0f));
    if (thread_x() == 0) {
        out[S_STEP2] = s2;
        out[S_TRY2] = t2;
        out[S_PREDICTED] = W(V(0.5f) * pr);
    }
}

// dst = src, n values.
KERNEL_THREADS(THREADS) void k_copy(GLOBAL V* RESTRICT dst, GLOBAL const V* RESTRICT src, u32 n) {
    u32 i = global_x();
    if (i < n) dst[i] = src[i];
}

// The rotation steps of x zeroed, one thread per entity: the three global
// indices at idx[at[e]..], a fixed one left alone; what re-centring does to
// the parameters after an accepted step.
KERNEL_THREADS(THREADS) void k_zero_steps(GLOBAL V* RESTRICT x, GLOBAL const u32* RESTRICT idx, GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
#pragma unroll
    for (u32 k = 0; k < 3; k++) {
        u32 g = idx[a + k];
        if (g != FIXED) x[g] = 0.0f;
    }
}
