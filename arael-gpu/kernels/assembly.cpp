// The pieces of a device assembly that every model shares: the kernels a
// constraint kernel reads from (the entities' parameter values and rotation
// caches) and the device functions it writes with (a residual row's rank-one
// update of an entity's tile and of a cross tile of the Hessian, its
// gradient terms, and the cost). The constraint kernels themselves are
// generated per model by arael's macros and compiled after this source.
//
// H and the gradient are double whatever the model's scalar F, and every
// write is an atomic add into buffers zeroed first. An entity's tile holds
// the upper block triangle, entry (i, j) of the entity's live parameters at
// base + (g_j - start) * stride + (g_i - start), g the global parameter
// indices and start the first live one; a cross tile holds the
// lower-numbered entity's span on its rows.
//
// Defines: SCALAR (the model's scalar). Every kernel runs blocks of 256.

typedef SCALAR F;

// A fixed parameter's global index.
#define FIXED 0xffffffffu

// An entity's parameter values from x (double) in F, one thread per value:
// ev[i] = x[idx[i]], or fixed[i] for a fixed parameter.
KERNEL_THREADS(256) void k_gather_params(GLOBAL const double* RESTRICT x, GLOBAL const u32* RESTRICT idx,
                                         GLOBAL const F* RESTRICT fixed, GLOBAL F* RESTRICT ev, u32 n) {
    u32 i = global_x();
    if (i >= n) return;
    u32 g = idx[i];
    ev[i] = g == FIXED ? fixed[i] : (F)x[g];
}

// R(delta) of an Euler-angle step, row-major: roll, pitch, yaw, with
// R = rot(z) rot(y) rot(x).
DEVICE void euler_matrix(double dx, double dy, double dz, double* d) {
    double sx = sin(dx), cx = cos(dx), sy = sin(dy), cy = cos(dy), sz = sin(dz), cz = cos(dz);
    d[0] = cy * cz;
    d[1] = sx * sy * cz - cx * sz;
    d[2] = cx * sy * cz + sx * sz;
    d[3] = cy * sz;
    d[4] = cx * cz + sx * sy * sz;
    d[5] = cx * sy * sz - sx * cz;
    d[6] = -sy;
    d[7] = cy * sx;
    d[8] = cx * cy;
}

// dR/d roll, pitch, yaw of the same step, k-major (27 values).
DEVICE void euler_derivatives(double dx, double dy, double dz, double* dd) {
    double sx = sin(dx), cx = cos(dx), sy = sin(dy), cy = cos(dy), sz = sin(dz), cz = cos(dz);
    dd[0] = 0.0;
    dd[1] = sx * sz + cz * cx * sy;
    dd[2] = cx * sz - sx * cz * sy;
    dd[3] = 0.0;
    dd[4] = cx * sy * sz - sx * cz;
    dd[5] = -sx * sy * sz - cz * cx;
    dd[6] = 0.0;
    dd[7] = cy * cx;
    dd[8] = -sx * cy;
    dd[9] = -sy * cz;
    dd[10] = cz * sx * cy;
    dd[11] = cx * cz * cy;
    dd[12] = -sy * sz;
    dd[13] = sx * cy * sz;
    dd[14] = cx * cy * sz;
    dd[15] = -cy;
    dd[16] = -sy * sx;
    dd[17] = -cx * sy;
    dd[18] = -cy * sz;
    dd[19] = -cx * cz - sz * sx * sy;
    dd[20] = sx * cz - cx * sz * sy;
    dd[21] = cy * cz;
    dd[22] = sx * sy * cz - cx * sz;
    dd[23] = cx * sy * cz + sz * sx;
    dd[24] = 0.0;
    dd[25] = 0.0;
    dd[26] = 0.0;
}

// The composed rotation R_ref R(delta) of one entity, in double (rot_wide)
// and F (rot), from its reference rotation refr (row-major, double) and its
// step (three values of ev at at[e]).
DEVICE void euler_compose(GLOBAL const double* RESTRICT refr, GLOBAL const F* RESTRICT ev, u32 a, u32 e,
                          GLOBAL double* RESTRICT rot_wide, GLOBAL F* RESTRICT rot) {
    double d[9];
    euler_matrix((double)ev[a], (double)ev[a + 1], (double)ev[a + 2], d);
    u32 r0 = e * 9;
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
#pragma unroll
        for (u32 j = 0; j < 3; j++) {
            double v = 0.0;
#pragma unroll
            for (u32 k = 0; k < 3; k++) v += refr[r0 + i * 3 + k] * d[k * 3 + j];
            rot_wide[r0 + i * 3 + j] = v;
            rot[r0 + i * 3 + j] = (F)v;
        }
    }
}

// One Euler-angle rotation per thread, values only: what a cost evaluation
// reads.
KERNEL_THREADS(256) void k_euler_values(GLOBAL const double* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                        GLOBAL const u32* RESTRICT at, GLOBAL double* RESTRICT rot_wide,
                                        GLOBAL F* RESTRICT rot, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    euler_compose(refr, ev, at[e], e, rot_wide, rot);
}

// One Euler-angle rotation per thread with its three derivatives
// R_ref dR(delta)/d delta_k in F (drot, k-major): what a linearization reads.
KERNEL_THREADS(256) void k_euler_rotations(GLOBAL const double* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                           GLOBAL const u32* RESTRICT at, GLOBAL double* RESTRICT rot_wide,
                                           GLOBAL F* RESTRICT rot, GLOBAL F* RESTRICT drot, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    euler_compose(refr, ev, a, e, rot_wide, rot);
    double dd[27];
    euler_derivatives((double)ev[a], (double)ev[a + 1], (double)ev[a + 2], dd);
    u32 r0 = e * 9, q0 = e * 27;
#pragma unroll
    for (u32 m = 0; m < 3; m++) {
#pragma unroll
        for (u32 i = 0; i < 3; i++) {
#pragma unroll
            for (u32 j = 0; j < 3; j++) {
                double v = 0.0;
#pragma unroll
                for (u32 k = 0; k < 3; k++) v += refr[r0 + i * 3 + k] * dd[m * 9 + k * 3 + j];
                drot[q0 + m * 9 + i * 3 + j] = (F)v;
            }
        }
    }
}

// The reference rotations moved by their steps, one thread per entity:
// refr = refr R(delta), as EulerAngleParam::advance does after an accepted
// step.
KERNEL_THREADS(256) void k_euler_advance(GLOBAL double* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                         GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    double d[9];
    euler_matrix((double)ev[a], (double)ev[a + 1], (double)ev[a + 2], d);
    u32 r0 = e * 9;
    double m[9];
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
#pragma unroll
        for (u32 j = 0; j < 3; j++) {
            double v = 0.0;
#pragma unroll
            for (u32 k = 0; k < 3; k++) v += refr[r0 + i * 3 + k] * d[k * 3 + j];
            m[i * 3 + j] = v;
        }
    }
#pragma unroll
    for (u32 i = 0; i < 9; i++) refr[r0 + i] = m[i];
}

// n doubles of x zeroed: an assembly's targets before the atomic adds.
KERNEL_THREADS(256) void k_zero_f64(GLOBAL double* RESTRICT x, u32 n) {
    u32 i = global_x();
    if (i < n) x[i] = 0.0;
}

// The rotation steps of x zeroed, one thread per entity: the three global
// indices at idx[at[e]..], a fixed one left alone; what re-centring does to
// the parameters after an accepted step.
KERNEL_THREADS(256) void k_zero_steps(GLOBAL double* RESTRICT x, GLOBAL const u32* RESTRICT idx,
                                      GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
#pragma unroll
    for (u32 k = 0; k < 3; k++) {
        u32 g = idx[a + k];
        if (g != FIXED) x[g] = 0.0;
    }
}

// H's diagonal gathered: out[i] = h[pos[i]], one thread per scalar.
KERNEL_THREADS(256) void k_gather_diag(GLOBAL const double* RESTRICT h, GLOBAL const u32* RESTRICT pos,
                                       GLOBAL double* RESTRICT out, u32 n) {
    u32 i = global_x();
    if (i >= n) return;
    out[i] = h[pos[i]];
}

// One residual row's update of an entity's tile and of the gradient:
// H += s dr dr^T over the upper triangle of the entity's live parameters
// and g[g_i] += s r dr_i, s being 2 (2w under a loss). idx[at..at + N] are
// the entity's global parameter indices; its tile is at base with columns
// stride apart, from parameter start.
template <u32 N>
DEVICE void self_rank1(GLOBAL double* h, GLOBAL double* g, GLOBAL const u32* RESTRICT idx, u32 at, u32 base,
                       u32 stride, u32 start, double s, F r, const F (&dr)[N]) {
#pragma unroll
    for (u32 i = 0; i < N; i++) {
        u32 gi = idx[at + i];
        if (gi != FIXED) {
            double di = (double)dr[i];
            atomic_add(&g[gi], s * (double)r * di);
            u32 ri = gi - start;
#pragma unroll
            for (u32 j = i; j < N; j++) {
                u32 gj = idx[at + j];
                if (gj != FIXED) atomic_add(&h[base + (gj - start) * stride + ri], s * di * (double)dr[j]);
            }
        }
    }
}

// One residual row's update of a cross tile: H += s dra drb^T between the
// live parameters of entities a (idx_a[at_a..], first live start_a) and b.
// The tile holds the lower-numbered entity on its rows: flip 0 when that is
// a, 1 when it is b.
template <u32 NA, u32 NB>
DEVICE void cross_rank1(GLOBAL double* h, GLOBAL const u32* RESTRICT idx_a, u32 at_a, u32 start_a,
                        GLOBAL const u32* RESTRICT idx_b, u32 at_b, u32 start_b, u32 base, u32 stride, u32 flip,
                        double s, const F (&dra)[NA], const F (&drb)[NB]) {
#pragma unroll
    for (u32 i = 0; i < NA; i++) {
        u32 gi = idx_a[at_a + i];
        if (gi != FIXED) {
            double di = s * (double)dra[i];
            u32 ra = gi - start_a;
#pragma unroll
            for (u32 j = 0; j < NB; j++) {
                u32 gj = idx_b[at_b + j];
                if (gj != FIXED) {
                    u32 cb = gj - start_b;
                    double v = di * (double)drb[j];
                    atomic_add(flip == 0 ? &h[base + cb * stride + ra] : &h[base + ra * stride + cb], v);
                }
            }
        }
    }
}

// The block's sum of its threads' v added to cost[0]: a shared-memory tree
// over a block of 256 threads, then one atomic add per block. Every thread
// of the block calls it.
DEVICE void cost_add(GLOBAL double* cost, double v) {
    SHARED double sh[256];
    u32 t = thread_x();
    sh[t] = v;
    BARRIER();
    for (u32 half = 128; half > 0; half >>= 1) {
        if (t < half) sh[t] += sh[t + half];
        BARRIER();
    }
    if (t == 0) atomic_add(cost, sh[0]);
}

// A column of the layout table: block base plus the byte offset in slot k
// of L, as T. The generated constraint kernels read every column this way.
#define COL(base, T, k) ((GLOBAL const T*)((base) + L[ARAEL_COL_##k]))
