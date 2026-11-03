// lm.cpp: the Levenberg-Marquardt loop's vector work on the device, launched
// by arael-gpu/src/native/lm.rs (the passes of arael-gpu/src/lm.rs).
//
// All f64. Every kernel runs THREADS threads per block. The scalars slots
// (S_*), the partials per block (LIN_PARTIALS, ATTEMPT_PARTIALS), NO_FAULT
// and FIXED come from lm.rs as #defines. A pass leaves partials per block,
// which the combine kernels fold, in one block, into the scalars.

#define OP_SUM 0
#define OP_MAX 1
#define OP_MIN 2

template <int OP> DEVICE double combine(double a, double b) {
    if (OP == OP_SUM) return a + b;
    if (OP == OP_MAX) return b > a ? b : a;
    return b < a ? b : a;
}

// `v` combined over the block by OP, the result in every thread; `sh` is
// THREADS words of shared memory.
template <int OP> DEVICE double block_reduce(double v, SHARED_PTR double* sh) {
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
KERNEL_THREADS(THREADS) void k_gather_diag(GLOBAL const double* RESTRICT h, GLOBAL const u32* RESTRICT pos, GLOBAL double* RESTRICT out, u32 n) {
    u32 i = global_x();
    if (i < n) out[i] = h[pos[i]];
}

// One thread per parameter: H's diagonal diag[i] = h[pos[i]], the damping
// scale (max(diag, floor) when has_floor, else the diagonal), and per block
// the gradient's largest magnitude, the largest |g| / sqrt(d) over d > 0,
// and the first faulty entry as 4 i + kind (1 NaN, 2 negative, 3 zero; with
// a floor a NaN or negative diagonal, without one also a zero).
KERNEL_THREADS(THREADS) void k_lin_pass(GLOBAL const double* RESTRICT g, GLOBAL const double* RESTRICT h, GLOBAL const u32* RESTRICT pos, GLOBAL double* RESTRICT diag, GLOBAL double* RESTRICT damp, double floor, u32 has_floor, GLOBAL double* RESTRICT partials, u32 n) {
    SHARED double sh[THREADS];
    u32 i = global_x();
    double gm = 0.0;
    double gs = 0.0;
    double fault = NO_FAULT;
    if (i < n) {
        double d = h[pos[i]];
        diag[i] = d;
        double gi = fabs(g[i]);
        gm = gi;
        if (d > 0.0) gs = gi / sqrt(d);
        bool bad = !(d > 0.0);
        if (has_floor != 0) {
            bad = !(d >= 0.0);
            damp[i] = d > floor ? d : floor;
        } else {
            damp[i] = d;
        }
        if (bad) {
            double kind = 3.0;
            if (d != d) {
                kind = 1.0;
            } else if (d < 0.0) {
                kind = 2.0;
            }
            fault = 4.0 * (double)i + kind;
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
KERNEL_THREADS(THREADS) void k_damp(GLOBAL double* RESTRICT h, GLOBAL const u32* RESTRICT pos, GLOBAL const double* RESTRICT diag, GLOBAL const double* RESTRICT damp, GLOBAL const double* RESTRICT lambda, u32 n) {
    u32 i = global_x();
    if (i < n) h[pos[i]] = diag[i] + lambda[0] * damp[i];
}

// One thread per parameter: the trial point x_try = x - delta, and per block
// the sums of delta^2, x_try^2 and delta (g + lambda[0] d delta) (twice the
// predicted reduction).
KERNEL_THREADS(THREADS) void k_attempt_pass(GLOBAL const double* RESTRICT x, GLOBAL const double* RESTRICT delta, GLOBAL const double* RESTRICT g, GLOBAL const double* RESTRICT diag, GLOBAL const double* RESTRICT lambda, GLOBAL double* RESTRICT x_try, GLOBAL double* RESTRICT partials, u32 n) {
    SHARED double sh[THREADS];
    u32 i = global_x();
    double s2 = 0.0;
    double t2 = 0.0;
    double pr = 0.0;
    if (i < n) {
        double d = delta[i];
        double t = x[i] - d;
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
template <int OP, u32 PER, u32 K> DEVICE double fold_partials(GLOBAL const double* partials, u32 blocks, SHARED_PTR double* sh) {
    double acc = OP == OP_MIN ? 1e300 : 0.0;
    for (u32 c = thread_x(); c < blocks; c += THREADS) acc = combine<OP>(acc, partials[c * PER + K]);
    return block_reduce<OP>(acc, sh);
}

// k_lin_pass's partials folded into the scalars, one block.
KERNEL_THREADS(THREADS) void k_combine_lin(GLOBAL const double* RESTRICT partials, u32 blocks, GLOBAL double* RESTRICT out) {
    SHARED double sh[THREADS];
    double gm = fold_partials<OP_MAX, LIN_PARTIALS, 0>(partials, blocks, sh);
    double gs = fold_partials<OP_MAX, LIN_PARTIALS, 1>(partials, blocks, sh);
    double fault = fold_partials<OP_MIN, LIN_PARTIALS, 2>(partials, blocks, sh);
    if (thread_x() == 0) {
        out[S_GRAD_MAX] = gm;
        out[S_GRAD_SCALED] = gs;
        out[S_FAULT] = fault;
    }
}

// k_attempt_pass's partials folded into the scalars, one block; the
// predicted reduction is half its sum.
KERNEL_THREADS(THREADS) void k_combine_attempt(GLOBAL const double* RESTRICT partials, u32 blocks, GLOBAL double* RESTRICT out) {
    SHARED double sh[THREADS];
    double s2 = fold_partials<OP_SUM, ATTEMPT_PARTIALS, 0>(partials, blocks, sh);
    double t2 = fold_partials<OP_SUM, ATTEMPT_PARTIALS, 1>(partials, blocks, sh);
    double pr = fold_partials<OP_SUM, ATTEMPT_PARTIALS, 2>(partials, blocks, sh);
    if (thread_x() == 0) {
        out[S_STEP2] = s2;
        out[S_TRY2] = t2;
        out[S_PREDICTED] = 0.5 * pr;
    }
}

// dst = src, n values.
KERNEL_THREADS(THREADS) void k_copy(GLOBAL double* RESTRICT dst, GLOBAL const double* RESTRICT src, u32 n) {
    u32 i = global_x();
    if (i < n) dst[i] = src[i];
}

// The rotation steps of x zeroed, one thread per entity: the three global
// indices at idx[at[e]..], a fixed one left alone; what re-centring does to
// the parameters after an accepted step.
KERNEL_THREADS(THREADS) void k_zero_steps(GLOBAL double* RESTRICT x, GLOBAL const u32* RESTRICT idx, GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
#pragma unroll
    for (u32 k = 0; k < 3; k++) {
        u32 g = idx[a + k];
        if (g != FIXED) x[g] = 0.0;
    }
}
