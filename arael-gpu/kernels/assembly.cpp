// The pieces of a device assembly that every model shares: the kernels a
// constraint kernel reads from (the entities' parameter values and rotation
// caches) and the device functions the generated kernels write and sum with.
// The constraint kernels themselves are generated per model by arael's
// macros and compiled after this source, with wide.h in front of it.
//
// H, the gradient and the cost are in W, the route's wide scalar, and no
// write is atomic. A constraint's instance kernel writes, per instance, its
// residual rows and their derivative rows (a record of RECW values of F,
// row after row: the residual, the A entity's derivatives, the B entity's),
// and per block of 256 instances its part of the cost. Then one kernel per
// entity side sums the records of each entity's instances into its tile and
// gradient, one per cross tile its instances' products, and one block the
// cost's parts: each output has one writer per launch, and the launches run
// one after another, so constraints that share an output add into it in turn.
//
// An entity's tile holds the upper block triangle, entry (i, j) of the
// entity's live parameters at base + (g_j - start) * stride + (g_i - start),
// g the global parameter indices and start the first live one; a cross tile
// holds the lower-numbered entity's span on its rows.
//
// Defines: SCALAR (the model's scalar). Every kernel runs blocks of 256.

typedef SCALAR F;

// A fixed parameter's global index.
#define FIXED 0xffffffffu

// The helpers of arael's external functions (arael-sym's EXTERN_CPP), as
// arael::utils computes them, for float and double.

// An angle in [-pi, pi]: one turn per 2 pi taken off outside it.
template <typename T>
DEVICE T arael_rad2rad(T v) {
    const T pi = (T)3.14159265358979323846;
    const T two_pi = (T)6.28318530717958647692;
    if (v < -pi || v > pi) return v - two_pi * floor(v / two_pi + (T)0.5);
    return v;
}

// The difference of two angles in [-pi, pi].
DEVICE float arael_rad_diff(float a, float b) { return arael_rad2rad(a - b); }
DEVICE double arael_rad_diff(double a, double b) { return arael_rad2rad(a - b); }

// The sum of two angles in [-pi, pi].
DEVICE float arael_rad_sum(float a, float b) { return arael_rad2rad(a + b); }
DEVICE double arael_rad_sum(double a, double b) { return arael_rad2rad(a + b); }

// sqrt(v), 0 for v at or below 0.
DEVICE float arael_safe_sqrt(float v) { return v <= 0.0f ? 0.0f : sqrt(v); }
DEVICE double arael_safe_sqrt(double v) { return v <= 0.0 ? 0.0 : sqrt(v); }

// The machine epsilon of the argument's type, its value ignored.
DEVICE float arael_epsilon_for(float) { return 1.1920928955078125e-7f; }
DEVICE double arael_epsilon_for(double) { return 2.220446049250313e-16; }

// An entity's parameter values from x (W) in F, one thread per value:
// ev[i] = x[idx[i]], or fixed[i] for a fixed parameter.
KERNEL_THREADS(256) void k_gather_params(GLOBAL const W* RESTRICT x, GLOBAL const u32* RESTRICT idx,
                                         GLOBAL const F* RESTRICT fixed, GLOBAL F* RESTRICT ev, u32 n) {
    u32 i = global_x();
    if (i >= n) return;
    u32 g = idx[i];
    ev[i] = g == FIXED ? fixed[i] : (F)x[g];
}

// R(delta) of an Euler-angle step, row-major: roll, pitch, yaw, with
// R = rot(z) rot(y) rot(x).
DEVICE void euler_matrix(W dx, W dy, W dz, W* d) {
    WidePair ex = wide_sin_cos(dx), ey = wide_sin_cos(dy), ez = wide_sin_cos(dz);
    W sx = ex.s, cx = ex.c, sy = ey.s, cy = ey.c, sz = ez.s, cz = ez.c;
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
DEVICE void euler_derivatives(W dx, W dy, W dz, W* dd) {
    WidePair ex = wide_sin_cos(dx), ey = wide_sin_cos(dy), ez = wide_sin_cos(dz);
    W sx = ex.s, cx = ex.c, sy = ey.s, cy = ey.c, sz = ez.s, cz = ez.c;
    W zero = 0.0f;
    dd[0] = zero;
    dd[1] = sx * sz + cz * cx * sy;
    dd[2] = cx * sz - sx * cz * sy;
    dd[3] = zero;
    dd[4] = cx * sy * sz - sx * cz;
    dd[5] = -sx * sy * sz - cz * cx;
    dd[6] = zero;
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
    dd[24] = zero;
    dd[25] = zero;
    dd[26] = zero;
}

// The composed rotation R_ref R(delta) of one entity, in W (rot_wide) and F
// (rot), from its reference rotation refr (row-major, W) and its step (three
// values of ev at at[e]).
DEVICE void euler_compose(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev, u32 a, u32 e,
                          GLOBAL W* RESTRICT rot_wide, GLOBAL F* RESTRICT rot) {
    W d[9];
    euler_matrix((W)ev[a], (W)ev[a + 1], (W)ev[a + 2], d);
    u32 r0 = e * 9;
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
#pragma unroll
        for (u32 j = 0; j < 3; j++) {
            W v = 0.0f;
#pragma unroll
            for (u32 k = 0; k < 3; k++) v += refr[r0 + i * 3 + k] * d[k * 3 + j];
            rot_wide[r0 + i * 3 + j] = v;
            rot[r0 + i * 3 + j] = (F)v;
        }
    }
}

// One Euler-angle rotation per thread, values only: what a cost evaluation
// reads.
KERNEL_THREADS(256) void k_euler_values(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                        GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT rot_wide,
                                        GLOBAL F* RESTRICT rot, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    euler_compose(refr, ev, at[e], e, rot_wide, rot);
}

// One Euler-angle rotation per thread with its three derivatives
// R_ref dR(delta)/d delta_k in F (drot, k-major): what a linearization reads.
KERNEL_THREADS(256) void k_euler_derivatives(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                             GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT rot_wide,
                                             GLOBAL F* RESTRICT rot, GLOBAL F* RESTRICT drot, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    euler_compose(refr, ev, a, e, rot_wide, rot);
    W dd[27];
    euler_derivatives((W)ev[a], (W)ev[a + 1], (W)ev[a + 2], dd);
    u32 r0 = e * 9, q0 = e * 27;
#pragma unroll
    for (u32 m = 0; m < 3; m++) {
#pragma unroll
        for (u32 i = 0; i < 3; i++) {
#pragma unroll
            for (u32 j = 0; j < 3; j++) {
                W v = 0.0f;
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
KERNEL_THREADS(256) void k_euler_advance(GLOBAL W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                         GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    W d[9];
    euler_matrix((W)ev[a], (W)ev[a + 1], (W)ev[a + 2], d);
    u32 r0 = e * 9;
    W m[9];
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
#pragma unroll
        for (u32 j = 0; j < 3; j++) {
            W v = 0.0f;
#pragma unroll
            for (u32 k = 0; k < 3; k++) v += refr[r0 + i * 3 + k] * d[k * 3 + j];
            m[i * 3 + j] = v;
        }
    }
#pragma unroll
    for (u32 i = 0; i < 9; i++) refr[r0 + i] = m[i];
}

// The rotation matrix of the unit quaternion q (w, x, y, z), row-major, as
// quatern::rotation_matrix.
DEVICE void quat_matrix(GLOBAL const W* RESTRICT q, W* m) {
    W w = q[0], x = q[1], y = q[2], z = q[3];
    W one = 1.0f, two = 2.0f;
    m[0] = one - two * (y * y + z * z);
    m[1] = two * (x * y - z * w);
    m[2] = two * (x * z + y * w);
    m[3] = two * (x * y + z * w);
    m[4] = one - two * (x * x + z * z);
    m[5] = two * (y * z - x * w);
    m[6] = two * (x * z - y * w);
    m[7] = two * (y * z + x * w);
    m[8] = one - two * (x * x + y * y);
}

// m v, m row-major.
DEVICE void mat_vec(const W* m, const W* v, W* out) {
#pragma unroll
    for (u32 i = 0; i < 3; i++) out[i] = m[i * 3] * v[0] + m[i * 3 + 1] * v[1] + m[i * 3 + 2] * v[2];
}

// a x b.
DEVICE void cross3(const W* a, const W* b, W* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

// m R(v) in W (wide) and F (val), 9 values each, and with DERIVS
// m dR(v)/dv_k in F (der, 27 values, k-major). R(v) is
// matrix3::from_rotation_vector_small: with (x, y, z) = v / 2 and
// s = 2 / (1 + x^2 + y^2 + z^2), R(v) = I + s A and
// dR/dv_k = (s dA/dk - k s^2 A) / 2.
template <bool DERIVS>
DEVICE void rotvec_apply(const W* m, W vx, W vy, W vz, GLOBAL W* RESTRICT wide, GLOBAL F* RESTRICT val,
                         GLOBAL F* RESTRICT der) {
    W half = 0.5f, one = 1.0f, two = 2.0f, zero = 0.0f;
    W x = vx * half, y = vy * half, z = vz * half;
    W s = two / (one + x * x + y * y + z * z);
    W A[9] = {-(y * y + z * z), x * y - z, x * z + y,
              x * y + z, -(x * x + z * z), y * z - x,
              x * z - y, y * z + x, -(x * x + y * y)};
    W d[9];
#pragma unroll
    for (u32 i = 0; i < 9; i++) d[i] = s * A[i];
    d[0] += one;
    d[4] += one;
    d[8] += one;
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
#pragma unroll
        for (u32 j = 0; j < 3; j++) {
            W v = 0.0f;
#pragma unroll
            for (u32 k = 0; k < 3; k++) v += m[i * 3 + k] * d[k * 3 + j];
            wide[i * 3 + j] = v;
            val[i * 3 + j] = (F)v;
        }
    }
    if (!DERIVS) return;
    W m2x = -(two * x), m2y = -(two * y), m2z = -(two * z), m1 = -one;
    // dA/dx, dA/dy, dA/dz
    W dA[27] = {zero, y, z, y, m2x, m1, z, one, m2x,
                m2y, x, one, x, zero, z, m1, z, m2y,
                m2z, m1, x, one, m2z, y, x, y, zero};
    W kv[3] = {x, y, z};
    W sh = s * half, s2h = s * s * half;
#pragma unroll
    for (u32 c = 0; c < 3; c++) {
        W dd[9];
#pragma unroll
        for (u32 i = 0; i < 9; i++) dd[i] = sh * dA[c * 9 + i] - kv[c] * s2h * A[i];
#pragma unroll
        for (u32 i = 0; i < 3; i++) {
#pragma unroll
            for (u32 j = 0; j < 3; j++) {
                W v = 0.0f;
#pragma unroll
                for (u32 k = 0; k < 3; k++) v += m[i * 3 + k] * dd[k * 3 + j];
                der[c * 9 + i * 3 + j] = (F)v;
            }
        }
    }
}

// q = unit(q unit(1, v / 2)): a reference quaternion moved by a
// rotation-vector step, as QuaternionParam::advance and the components'
// updates do.
DEVICE void quat_advance(GLOBAL W* RESTRICT q, W vx, W vy, W vz) {
    W half = 0.5f, one = 1.0f;
    W sx = vx * half, sy = vy * half, sz = vz * half;
    W inv = one / sqrt(one + sx * sx + sy * sy + sz * sz);
    W sw = inv;
    sx *= inv;
    sy *= inv;
    sz *= inv;
    W w = q[0], x = q[1], y = q[2], z = q[3];
    W nw = w * sw - (x * sx + y * sy + z * sz);
    W nx = sx * w + x * sw + (y * sz - z * sy);
    W ny = sy * w + y * sw + (z * sx - x * sz);
    W nz = sz * w + z * sw + (x * sy - y * sx);
    W r = one / sqrt(nw * nw + nx * nx + ny * ny + nz * nz);
    q[0] = nw * r;
    q[1] = nx * r;
    q[2] = ny * r;
    q[3] = nz * r;
}

// A QuaternionParam: its reference a unit quaternion (4 values), its step
// a rotation vector, its values R(q_ref) R(v) (9) and their derivatives
// (27).
KERNEL_THREADS(256) void k_quat_values(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                       GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                       GLOBAL F* RESTRICT val, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    W m[9];
    quat_matrix(refr + e * 4, m);
    rotvec_apply<false>(m, (W)ev[a], (W)ev[a + 1], (W)ev[a + 2], wide + e * 9, val + e * 9, val);
}

KERNEL_THREADS(256) void k_quat_derivatives(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                            GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                            GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    W m[9];
    quat_matrix(refr + e * 4, m);
    rotvec_apply<true>(m, (W)ev[a], (W)ev[a + 1], (W)ev[a + 2], wide + e * 9, val + e * 9, der + e * 27);
}

KERNEL_THREADS(256) void k_quat_advance(GLOBAL W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                        GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    quat_advance(refr + e * 4, (W)ev[a], (W)ev[a + 1], (W)ev[a + 2]);
}

// carry(w, d) = d + (w x d) / 2 + (w x (w x d)) / 6, the translation of a
// twist step (crate::twist::carry).
DEVICE void carry3(const W* w, const W* d, W* t) {
    W half = 0.5f, sixth = (W)1.0f / (W)6.0f;
    W wd[3], wwd[3];
    cross3(w, d, wd);
    cross3(w, wd, wwd);
#pragma unroll
    for (u32 i = 0; i < 3; i++) t[i] = d[i] + wd[i] * half + wwd[i] * sixth;
}

// A TransformParam, as TransformParam::__precompute_symbolic: its reference
// the rotation's unit quaternion and the translation (7 values), its step
// w (rotation vector) then d (translation rate); its values the rotation
// R(q_ref) R(w) and the translation t_ref + R(q_ref) carry(w, d) (12), its
// derivatives the rotation's by w (27), the translation's by d (9) and by
// w (9), k-major.
template <bool DERIVS>
DEVICE void transform_compose(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev, u32 a, u32 e,
                              GLOBAL W* RESTRICT wide, GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der) {
    GLOBAL const W* r = refr + e * 7;
    W m[9];
    quat_matrix(r, m);
    W w[3] = {(W)ev[a], (W)ev[a + 1], (W)ev[a + 2]};
    W d[3] = {(W)ev[a + 3], (W)ev[a + 4], (W)ev[a + 5]};
    GLOBAL W* ow = wide + e * 12;
    GLOBAL F* ov = val + e * 12;
    GLOBAL F* od = der + e * 45;
    rotvec_apply<DERIVS>(m, w[0], w[1], w[2], ow, ov, od);
    W c[3], mc[3];
    carry3(w, d, c);
    mat_vec(m, c, mc);
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
        W t = r[4 + i] + mc[i];
        ow[9 + i] = t;
        ov[9 + i] = (F)t;
    }
    if (!DERIVS) return;
    W half = 0.5f, sixth = (W)1.0f / (W)6.0f, zero = 0.0f, one = 1.0f;
    W wd[3];
    cross3(w, d, wd);
#pragma unroll
    for (u32 k = 0; k < 3; k++) {
        W ek[3] = {k == 0 ? one : zero, k == 1 ? one : zero, k == 2 ? one : zero};
        // d(carry)/dd_k = carry(w, e_k)
        W t[3], mt[3];
        carry3(w, ek, t);
        mat_vec(m, t, mt);
#pragma unroll
        for (u32 i = 0; i < 3; i++) od[27 + k * 3 + i] = (F)mt[i];
        // d(carry)/dw_k = (e_k x d) / 2 + (e_k x (w x d) + w x (e_k x d)) / 6
        W ed[3], ewd[3], wed[3];
        cross3(ek, d, ed);
        cross3(ek, wd, ewd);
        cross3(w, ed, wed);
#pragma unroll
        for (u32 i = 0; i < 3; i++) t[i] = ed[i] * half + (ewd[i] + wed[i]) * sixth;
        mat_vec(m, t, mt);
#pragma unroll
        for (u32 i = 0; i < 3; i++) od[36 + k * 3 + i] = (F)mt[i];
    }
}

KERNEL_THREADS(256) void k_transform_values(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                            GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                            GLOBAL F* RESTRICT val, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    transform_compose<false>(refr, ev, at[e], e, wide, val, val);
}

KERNEL_THREADS(256) void k_transform_derivatives(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                                 GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                                 GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    transform_compose<true>(refr, ev, at[e], e, wide, val, der);
}

// The reference moved by the step, as TransformParam's update: the
// translation by R(q_ref) carry(w, d) (the rotation before the move), then
// the rotation by w.
KERNEL_THREADS(256) void k_transform_advance(GLOBAL W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                             GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    GLOBAL W* r = refr + e * 7;
    W m[9];
    quat_matrix(r, m);
    W w[3] = {(W)ev[a], (W)ev[a + 1], (W)ev[a + 2]};
    W d[3] = {(W)ev[a + 3], (W)ev[a + 4], (W)ev[a + 5]};
    W c[3], mc[3];
    carry3(w, d, c);
    mat_vec(m, c, mc);
#pragma unroll
    for (u32 i = 0; i < 3; i++) r[4 + i] += mc[i];
    quat_advance(r, w[0], w[1], w[2]);
}

// A UnitVecParam, as UnitVecParam::__precompute_symbolic: its reference a
// unit quaternion (4 values), its step d (2), its value the direction
// R(q_ref) local(d) (3) with local(d) = (1 - u / (2 s2), d.y / s2,
// -d.x / s2), u = |d|^2 and s2 = 1 + u / 4, its derivatives by d.x and d.y
// (6).
template <bool DERIVS>
DEVICE void unitvec_compose(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev, u32 a, u32 e,
                            GLOBAL W* RESTRICT wide, GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der) {
    W m[9];
    quat_matrix(refr + e * 4, m);
    W dx = (W)ev[a], dy = (W)ev[a + 1];
    W one = 1.0f, two = 2.0f, quarter = 0.25f, half = 0.5f;
    W u = dx * dx + dy * dy;
    W s2 = one + u * quarter;
    W local[3] = {one - u / (two * s2), dy / s2, -dx / s2};
    W o[3];
    mat_vec(m, local, o);
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
        wide[e * 3 + i] = o[i];
        val[e * 3 + i] = (F)o[i];
    }
    if (!DERIVS) return;
    W inv2 = one / (s2 * s2);
    W lx[3] = {-dx * inv2, -(dx * dy * half) * inv2, -(s2 - dx * dx * half) * inv2};
    W ly[3] = {-dy * inv2, (s2 - dy * dy * half) * inv2, (dx * dy * half) * inv2};
    W ox[3], oy[3];
    mat_vec(m, lx, ox);
    mat_vec(m, ly, oy);
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
        der[e * 6 + i] = (F)ox[i];
        der[e * 6 + 3 + i] = (F)oy[i];
    }
}

KERNEL_THREADS(256) void k_unitvec_values(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                          GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                          GLOBAL F* RESTRICT val, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    unitvec_compose<false>(refr, ev, at[e], e, wide, val, val);
}

KERNEL_THREADS(256) void k_unitvec_derivatives(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                               GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                               GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    unitvec_compose<true>(refr, ev, at[e], e, wide, val, der);
}

// The reference moved by the step, as UnitVecParam's update: q_ref by the
// rotation vector (0, d.x, d.y).
KERNEL_THREADS(256) void k_unitvec_advance(GLOBAL W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                           GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    quat_advance(refr + e * 4, (W)0.0f, (W)ev[a], (W)ev[a + 1]);
}

// n values of x zeroed: an assembly's targets before the sums add into them.
KERNEL_THREADS(256) void k_zero_wide(GLOBAL W* RESTRICT x, u32 n) {
    u32 i = global_x();
    if (i < n) x[i] = 0.0f;
}

// The steps of x zeroed, one thread per entity: the len global indices at
// idx[at[e]..], a fixed one left alone; what re-centring does to the
// parameters after an accepted step.
KERNEL_THREADS(256) void k_zero_steps(GLOBAL W* RESTRICT x, GLOBAL const u32* RESTRICT idx,
                                      GLOBAL const u32* RESTRICT at, u32 len, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    for (u32 k = 0; k < len; k++) {
        u32 g = idx[a + k];
        if (g != FIXED) x[g] = 0.0f;
    }
}

// The block's sum of its threads' v into part[block]: a shared-memory tree
// over a block of 256 threads. Every thread of the block calls it.
DEVICE void cost_partial(GLOBAL W* part, W v) {
    SHARED W sh[256];
    u32 t = thread_x();
    sh[t] = v;
    BARRIER();
    for (u32 half = 128; half > 0; half >>= 1) {
        if (t < half) sh[t] += sh[t + half];
        BARRIER();
    }
    if (t == 0) part[block_x()] = sh[0];
}

// cost[0] += the sum of part[0..m), by one block of 256 threads.
DEVICE void cost_sum(GLOBAL const W* part, u32 m, GLOBAL W* cost) {
    SHARED W sh[256];
    u32 t = thread_x();
    W acc = 0.0f;
    for (u32 i = t; i < m; i += 256) acc += part[i];
    sh[t] = acc;
    BARRIER();
    for (u32 half = 128; half > 0; half >>= 1) {
        if (t < half) sh[t] += sh[t + half];
        BARRIER();
    }
    if (t == 0) cost[0] += sh[0];
}

// One entity side's sums, one thread per (entity slot, entry) over `slots`
// slots: an entity of N parameters has N (N + 1) / 2 entries of its tile's
// upper triangle, then N of the gradient. Entry (i, j) of the tile sums
// 2 d_i d_j, entry i of the gradient 2 r d_i, over the rows of the slot's
// instances inst[off[s]..off[s + 1]], its derivatives at OFF in each
// row of STRIDE values of a record of R rows. idx and tile are the entity
// array's global indices and tiles.
template <u32 N, u32 R, u32 STRIDE, u32 OFF>
DEVICE void reduce_entity(GLOBAL const F* RESTRICT rec, GLOBAL const u32* RESTRICT off, GLOBAL const u32* RESTRICT inst,
                          GLOBAL const u32* RESTRICT idx, GLOBAL const u32* RESTRICT tile, GLOBAL W* RESTRICT h,
                          GLOBAL W* RESTRICT g, u32 slots) {
    const u32 nt = N * (N + 1) / 2;
    const u32 ne = nt + N;
    u32 t = global_x();
    if (t >= slots * ne) return;
    u32 s = t / ne;
    u32 e = t - s * ne;
    // the entry's (i, j), i <= j, row by row of the upper triangle
    u32 i = 0, j = 0;
    bool grad = e >= nt;
    if (grad) {
        i = e - nt;
    } else {
        u32 left = e;
        while (left >= N - i) {
            left -= N - i;
            i++;
        }
        j = i + left;
    }
    u32 gi = idx[s * N + i];
    if (gi == FIXED) return;
    u32 gj = grad ? 0u : idx[s * N + j];
    if (!grad && gj == FIXED) return;
    W acc = 0.0f;
    for (u32 q = off[s]; q < off[s + 1]; q++) {
        GLOBAL const F* r = rec + inst[q] * (R * STRIDE);
#pragma unroll
        for (u32 row = 0; row < R; row++) {
            GLOBAL const F* rr = r + row * STRIDE;
            acc = wide_fma(acc, rr[OFF + i], grad ? rr[0] : rr[OFF + j]);
        }
    }
    acc += acc;
    if (grad) {
        g[gi] += acc;
    } else {
        u32 start = tile[s * 3 + 2];
        h[tile[s * 3] + (gj - start) * tile[s * 3 + 1] + (gi - start)] += acc;
    }
}

// One cross array's sums, one thread per (cross slot, i, j) over `slots`
// slots: 2 dA_i dB_j summed over the rows of the slot's instances
// inst[off[x]..off[x + 1]] (one pair of entities, sa and sb those of the
// first, per slot), into the cross tile `tile_x` of the slot, between the
// live parameters of the entities (global indices idx_a, idx_b; first live
// ones in tile_a, tile_b). An instance of the pair the other way round (its
// A the first's B: two entities of one array, NA = NB) adds dA_j dB_i.
template <u32 NA, u32 NB, u32 R, u32 STRIDE>
DEVICE void reduce_cross(GLOBAL const F* RESTRICT rec, GLOBAL const u32* RESTRICT off, GLOBAL const u32* RESTRICT inst,
                         GLOBAL const u32* RESTRICT slot_a, GLOBAL const u32* RESTRICT slot_b,
                         GLOBAL const u32* RESTRICT idx_a, GLOBAL const u32* RESTRICT tile_a,
                         GLOBAL const u32* RESTRICT idx_b, GLOBAL const u32* RESTRICT tile_b,
                         GLOBAL const u32* RESTRICT tile_x, GLOBAL W* RESTRICT h, u32 slots) {
    u32 t = global_x();
    if (t >= slots * NA * NB) return;
    u32 x = t / (NA * NB);
    u32 e = t - x * (NA * NB);
    u32 i = e / NB;
    u32 j = e - i * NB;
    u32 q0 = off[x], q1 = off[x + 1];
    if (q0 == q1) return;
    u32 sa = slot_a[inst[q0]], sb = slot_b[inst[q0]];
    u32 gi = idx_a[sa * NA + i];
    u32 gj = idx_b[sb * NB + j];
    if (gi == FIXED || gj == FIXED) return;
    W acc = 0.0f;
    for (u32 q = q0; q < q1; q++) {
        GLOBAL const F* r = rec + inst[q] * (R * STRIDE);
        bool reversed = NA == NB && slot_a[inst[q]] != sa;
        u32 ia = reversed ? 1 + j : 1 + i;
        u32 jb = reversed ? 1 + NA + i : 1 + NA + j;
#pragma unroll
        for (u32 row = 0; row < R; row++) {
            GLOBAL const F* rr = r + row * STRIDE;
            acc = wide_fma(acc, rr[ia], rr[jb]);
        }
    }
    acc += acc;
    u32 ra = gi - tile_a[sa * 3 + 2];
    u32 cb = gj - tile_b[sb * 3 + 2];
    u32 base = tile_x[x * 3], stride = tile_x[x * 3 + 1], flip = tile_x[x * 3 + 2];
    h[flip == 0 ? base + cb * stride + ra : base + ra * stride + cb] += acc;
}

// A column of the layout table: block base plus the byte offset in slot k
// of L, as T (COL_MUT: writable). The generated constraint kernels read
// every column this way.
#define COL(base, T, k) ((GLOBAL const T*)((base) + L[ARAEL_COL_##k]))
#define COL_MUT(base, T, k) ((GLOBAL T*)((base) + L[ARAEL_COL_##k]))
