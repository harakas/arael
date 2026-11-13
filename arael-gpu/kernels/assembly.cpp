// The pieces of a device assembly that every model shares: the kernels a
// constraint kernel reads from (the entities' parameter values and rotation
// caches) and the device functions the generated kernels write and sum with.
// The constraint kernels themselves are generated per model by arael's
// macros and compiled after this source, with wide.h in front of it.
//
// The parameters, H and the gradient are in V, the root's scalar (wide.h:
// f32 for an f32 root, as the CPU's route sums and keeps them), the cost's
// terms and sums in the root's CACC (V, or W with `cost_f64`), the cost
// itself and the references in W, the route's wide scalar, and no write is
// atomic. A constraint's instance kernel writes, per instance, its
// residual rows and their derivative rows (a record of RECW values of F,
// row after row: the residual, then each of its entities' derivatives in
// side order), and per block of 256 instances its part of the cost. Then
// one kernel per side sums the records of each entity's instances into its
// tile and gradient, one per pair of sides with a cross tile its instances'
// products, and one block the
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
#if HAS_F64
DEVICE double arael_rad_diff(double a, double b) { return arael_rad2rad(a - b); }
#endif

// The sum of two angles in [-pi, pi].
DEVICE float arael_rad_sum(float a, float b) { return arael_rad2rad(a + b); }
#if HAS_F64
DEVICE double arael_rad_sum(double a, double b) { return arael_rad2rad(a + b); }
#endif

// sqrt(v), 0 for v at or below 0.
DEVICE float arael_safe_sqrt(float v) { return v <= 0.0f ? 0.0f : sqrt(v); }
#if HAS_F64
DEVICE double arael_safe_sqrt(double v) { return v <= 0.0 ? 0.0 : sqrt(v); }
#endif

// The machine epsilon of the argument's type, its value ignored.
DEVICE float arael_epsilon_for(float) { return 1.1920928955078125e-7f; }
#if HAS_F64
DEVICE double arael_epsilon_for(double) { return 2.220446049250313e-16; }
#endif

// arael::utils' fast atan (max error < 1e-6 radians), with its folds, its
// constants and its operations in its order: the argument's magnitude
// folded below 1, then below tan(pi / 12) by the pi / 6 shift, a rational
// polynomial there.
template <typename T>
DEVICE T arael_fast_atan_t(T v) {
    const T half_pi = (T)1.5707963267948966;
    const T sixth_pi = (T)5.235987755982988e-1;
    const T tan_sixth_pi = (T)5.773502691896257e-1;
    const T tan_twelfth_pi = (T)2.679491924311227e-1;
    const T c1 = (T)1.6867629106, c2 = (T)0.4378497304, c3 = (T)1.6867633134;
    const bool negative = v < (T)0;
    T x = negative ? -v : v;
    const bool inverted = x > (T)1;
    if (inverted) x = (T)1 / x;
    T y;
    if (x > tan_twelfth_pi) {
        x = (x - tan_sixth_pi) / ((T)1 + tan_sixth_pi * x);
        const T x2 = x * x;
        y = x * (c1 + x2 * c2) / (c3 + x2);
        y += sixth_pi;
    } else {
        const T x2 = x * x;
        y = x * (c1 + x2 * c2) / (c3 + x2);
    }
    if (inverted) y = half_pi - y;
    return negative ? -y : y;
}
DEVICE float arael_fast_atan(float v) { return arael_fast_atan_t(v); }
#if HAS_F64
DEVICE double arael_fast_atan(double v) { return arael_fast_atan_t(v); }
#endif

// arael::utils' fast atan2 over its fast atan: 0 at the origin, +-pi / 2
// on the y axis, the quadrant by x's and y's signs.
template <typename T>
DEVICE T arael_fast_atan2_t(T y, T x) {
    const T pi = (T)3.14159265358979323846;
    if (x > (T)0) return arael_fast_atan_t(y / x);
    if (x == (T)0) {
        if (y == (T)0) return (T)0;
        return copysign((T)0.5 * pi, y);
    }
    if (y >= (T)0) return arael_fast_atan_t(y / x) + pi;
    return arael_fast_atan_t(y / x) - pi;
}
DEVICE float arael_fast_atan2(float y, float x) { return arael_fast_atan2_t(y, x); }
#if HAS_F64
DEVICE double arael_fast_atan2(double y, double x) { return arael_fast_atan2_t(y, x); }
#endif

// The arm of n a select's index x picks (arael::utils::SelectIndex): the
// integer in [0, n) x is, -1 for an integer outside them (the default
// arm), -2 for a NaN, an infinity or a fraction (where the CPU panics).
template <typename T>
DEVICE int arael_select_index(T x, u32 n) {
    if (isnan(x) || isinf(x) || x != floor(x)) return -2;
    if (x < (T)0 || x >= (T)n) return -1;
    return (int)x;
}

// A quiet NaN of the argument's type, its value ignored: a select's value
// where the CPU's select panics.
DEVICE float arael_nan_of(float) { return __int_as_float(0x7fc00000); }
#if HAS_F64
DEVICE double arael_nan_of(double) { return __longlong_as_double(0x7ff8000000000000LL); }
#endif
#ifdef ARAEL_WIDE_DF32
DEVICE df32 arael_nan_of(df32) { return df32(arael_nan_of(0.0f)); }
#endif

// An entity's parameter values from x (V) in F, one thread per value:
// ev[i] = x[idx[i]], or fixed[i] for a fixed parameter.
KERNEL_THREADS(256) void k_gather_params(GLOBAL const V* RESTRICT x, GLOBAL const u32* RESTRICT idx,
                                         GLOBAL const F* RESTRICT fixed, GLOBAL F* RESTRICT ev, u32 n) {
    u32 i = global_x();
    if (i >= n) return;
    u32 g = idx[i];
    ev[i] = g == FIXED ? fixed[i] : (F)x[g];
}

// R(delta) of an Euler-angle step, row-major: roll, pitch, yaw, with
// R = rot(z) rot(y) rot(x).
template <typename T>
DEVICE void euler_matrix(T dx, T dy, T dz, THREAD T* d) {
    SinCos<T> ex = sin_cos(dx), ey = sin_cos(dy), ez = sin_cos(dz);
    T sx = ex.s, cx = ex.c, sy = ey.s, cy = ey.c, sz = ez.s, cz = ez.c;
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
template <typename T>
DEVICE void euler_derivatives(T dx, T dy, T dz, THREAD T* dd) {
    SinCos<T> ex = sin_cos(dx), ey = sin_cos(dy), ez = sin_cos(dz);
    T sx = ex.s, cx = ex.c, sy = ey.s, cy = ey.c, sz = ez.s, cz = ez.c;
    T zero = 0.0f;
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

// The composed rotation R_ref R(delta) of one entity, computed in T, the
// reference's scalar, into W (rot_wide) and F (rot), from its reference
// rotation refr (row-major) and its step (three values of ev at at[e]).
template <typename T>
DEVICE void euler_compose(GLOBAL const T* RESTRICT refr, GLOBAL const F* RESTRICT ev, u32 a, u32 e,
                          GLOBAL W* RESTRICT rot_wide, GLOBAL F* RESTRICT rot) {
    T d[9];
    euler_matrix((T)ev[a], (T)ev[a + 1], (T)ev[a + 2], d);
    u32 r0 = e * 9;
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
#pragma unroll
        for (u32 j = 0; j < 3; j++) {
            T v = 0.0f;
#pragma unroll
            for (u32 k = 0; k < 3; k++) v += refr[r0 + i * 3 + k] * d[k * 3 + j];
            rot_wide[r0 + i * 3 + j] = W(v);
            rot[r0 + i * 3 + j] = (F)v;
        }
    }
}

// One Euler-angle rotation per thread, values only: what a cost evaluation
// reads. In T, the reference's scalar: V for an EulerAngleParam (the
// root's), W for a MixedEulerAngleParam (its wide state); e the thread's
// entity.
template <typename T>
DEVICE void euler_values(GLOBAL const T* RESTRICT refr, GLOBAL const F* RESTRICT ev, GLOBAL const u32* RESTRICT at,
                         GLOBAL W* RESTRICT rot_wide, GLOBAL F* RESTRICT rot, u32 n, u32 e) {
    if (e >= n) return;
    euler_compose(refr, ev, at[e], e, rot_wide, rot);
}

// One Euler-angle rotation per thread with its three derivatives
// R_ref dR(delta)/d delta_k in F (drot, k-major): what a linearization
// reads; T as euler_values'.
template <typename T>
DEVICE void euler_derivs(GLOBAL const T* RESTRICT refr, GLOBAL const F* RESTRICT ev, GLOBAL const u32* RESTRICT at,
                         GLOBAL W* RESTRICT rot_wide, GLOBAL F* RESTRICT rot, GLOBAL F* RESTRICT drot, u32 n, u32 e) {
    if (e >= n) return;
    u32 a = at[e];
    euler_compose(refr, ev, a, e, rot_wide, rot);
    T dd[27];
    euler_derivatives((T)ev[a], (T)ev[a + 1], (T)ev[a + 2], dd);
    u32 r0 = e * 9, q0 = e * 27;
#pragma unroll
    for (u32 m = 0; m < 3; m++) {
#pragma unroll
        for (u32 i = 0; i < 3; i++) {
#pragma unroll
            for (u32 j = 0; j < 3; j++) {
                T v = 0.0f;
#pragma unroll
                for (u32 k = 0; k < 3; k++) v += refr[r0 + i * 3 + k] * dd[m * 9 + k * 3 + j];
                drot[q0 + m * 9 + i * 3 + j] = (F)v;
            }
        }
    }
}

// The reference rotations moved by their steps, one thread per entity:
// refr = refr R(delta), as EulerAngleParam::advance does after an accepted
// step; T as euler_values'.
template <typename T>
DEVICE void euler_advance(GLOBAL T* RESTRICT refr, GLOBAL const F* RESTRICT ev, GLOBAL const u32* RESTRICT at, u32 n, u32 e) {
    if (e >= n) return;
    u32 a = at[e];
    T d[9];
    euler_matrix((T)ev[a], (T)ev[a + 1], (T)ev[a + 2], d);
    u32 r0 = e * 9;
    T m[9];
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
#pragma unroll
        for (u32 j = 0; j < 3; j++) {
            T v = 0.0f;
#pragma unroll
            for (u32 k = 0; k < 3; k++) v += refr[r0 + i * 3 + k] * d[k * 3 + j];
            m[i * 3 + j] = v;
        }
    }
#pragma unroll
    for (u32 i = 0; i < 9; i++) refr[r0 + i] = m[i];
}

KERNEL_THREADS(256) void k_euler_values(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                        GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT rot_wide,
                                        GLOBAL F* RESTRICT rot, u32 n) {
    euler_values(refr, ev, at, rot_wide, rot, n, global_x());
}
KERNEL_THREADS(256) void k_euler_derivatives(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                             GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT rot_wide,
                                             GLOBAL F* RESTRICT rot, GLOBAL F* RESTRICT drot, u32 n) {
    euler_derivs(refr, ev, at, rot_wide, rot, drot, n, global_x());
}
KERNEL_THREADS(256) void k_euler_advance(GLOBAL V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                         GLOBAL const u32* RESTRICT at, u32 n) {
    euler_advance(refr, ev, at, n, global_x());
}
KERNEL_THREADS(256) void k_mixed_euler_values(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                              GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT rot_wide,
                                              GLOBAL F* RESTRICT rot, u32 n) {
    euler_values(refr, ev, at, rot_wide, rot, n, global_x());
}
KERNEL_THREADS(256) void k_mixed_euler_derivatives(GLOBAL const W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                                   GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT rot_wide,
                                                   GLOBAL F* RESTRICT rot, GLOBAL F* RESTRICT drot, u32 n) {
    euler_derivs(refr, ev, at, rot_wide, rot, drot, n, global_x());
}
KERNEL_THREADS(256) void k_mixed_euler_advance(GLOBAL W* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                               GLOBAL const u32* RESTRICT at, u32 n) {
    euler_advance(refr, ev, at, n, global_x());
}

// A SimpleEulerAngleParam: no reference, its parameters the angles
// themselves; its values R(angles) (9) and their derivatives by the angles
// (27, k-major), as the parameter's own precompute.
KERNEL_THREADS(256) void k_simple_euler_values(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                               GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                               GLOBAL F* RESTRICT val, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    V d[9];
    euler_matrix((V)ev[a], (V)ev[a + 1], (V)ev[a + 2], d);
#pragma unroll
    for (u32 i = 0; i < 9; i++) {
        wide[e * 9 + i] = W(d[i]);
        val[e * 9 + i] = (F)d[i];
    }
}

KERNEL_THREADS(256) void k_simple_euler_derivatives(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                                    GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                                    GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    V d[9];
    euler_matrix((V)ev[a], (V)ev[a + 1], (V)ev[a + 2], d);
#pragma unroll
    for (u32 i = 0; i < 9; i++) {
        wide[e * 9 + i] = W(d[i]);
        val[e * 9 + i] = (F)d[i];
    }
    V dd[27];
    euler_derivatives((V)ev[a], (V)ev[a + 1], (V)ev[a + 2], dd);
#pragma unroll
    for (u32 i = 0; i < 27; i++) der[e * 27 + i] = (F)dd[i];
}

// The rotation matrix of the unit quaternion q (w, x, y, z), row-major, as
// quatern::rotation_matrix.
template <typename T>
DEVICE void quat_matrix(GLOBAL const T* RESTRICT q, THREAD T* m) {
    T w = q[0], x = q[1], y = q[2], z = q[3];
    T one = 1.0f, two = 2.0f;
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
template <typename T>
DEVICE void mat_vec(THREAD const T* m, THREAD const T* v, THREAD T* out) {
#pragma unroll
    for (u32 i = 0; i < 3; i++) out[i] = m[i * 3] * v[0] + m[i * 3 + 1] * v[1] + m[i * 3 + 2] * v[2];
}

// a x b.
template <typename T>
DEVICE void cross3(THREAD const T* a, THREAD const T* b, THREAD T* c) {
    c[0] = a[1] * b[2] - a[2] * b[1];
    c[1] = a[2] * b[0] - a[0] * b[2];
    c[2] = a[0] * b[1] - a[1] * b[0];
}

// m R(v), computed in T, into W (wide) and F (val), 9 values each, and with DERIVS
// m dR(v)/dv_k in F (der, 27 values, k-major). R(v) is
// matrix3::from_rotation_vector_small: with (x, y, z) = v / 2 and
// s = 2 / (1 + x^2 + y^2 + z^2), R(v) = I + s A and
// dR/dv_k = (s dA/dk - k s^2 A) / 2.
template <bool DERIVS, typename T>
DEVICE void rotvec_apply(THREAD const T* m, T vx, T vy, T vz, GLOBAL W* RESTRICT wide, GLOBAL F* RESTRICT val,
                         GLOBAL F* RESTRICT der) {
    T one_half = 0.5f, one = 1.0f, two = 2.0f, zero = 0.0f;
    T x = vx * one_half, y = vy * one_half, z = vz * one_half;
    T s = two / (one + x * x + y * y + z * z);
    T A[9] = {-(y * y + z * z), x * y - z, x * z + y,
              x * y + z, -(x * x + z * z), y * z - x,
              x * z - y, y * z + x, -(x * x + y * y)};
    T d[9];
#pragma unroll
    for (u32 i = 0; i < 9; i++) d[i] = s * A[i];
    d[0] += one;
    d[4] += one;
    d[8] += one;
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
#pragma unroll
        for (u32 j = 0; j < 3; j++) {
            T v = 0.0f;
#pragma unroll
            for (u32 k = 0; k < 3; k++) v += m[i * 3 + k] * d[k * 3 + j];
            wide[i * 3 + j] = W(v);
            val[i * 3 + j] = (F)v;
        }
    }
    if (!DERIVS) return;
    T m2x = -(two * x), m2y = -(two * y), m2z = -(two * z), m1 = -one;
    // dA/dx, dA/dy, dA/dz
    T dA[27] = {zero, y, z, y, m2x, m1, z, one, m2x,
                m2y, x, one, x, zero, z, m1, z, m2y,
                m2z, m1, x, one, m2z, y, x, y, zero};
    T kv[3] = {x, y, z};
    T sh = s * one_half, s2h = s * s * one_half;
#pragma unroll
    for (u32 c = 0; c < 3; c++) {
        T dd[9];
#pragma unroll
        for (u32 i = 0; i < 9; i++) dd[i] = sh * dA[c * 9 + i] - kv[c] * s2h * A[i];
#pragma unroll
        for (u32 i = 0; i < 3; i++) {
#pragma unroll
            for (u32 j = 0; j < 3; j++) {
                T v = 0.0f;
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
template <typename T>
DEVICE void quat_advance(GLOBAL T* RESTRICT q, T vx, T vy, T vz) {
    T one_half = 0.5f, one = 1.0f;
    T sx = vx * one_half, sy = vy * one_half, sz = vz * one_half;
    T inv = one / sqrt(one + sx * sx + sy * sy + sz * sz);
    T sw = inv;
    sx *= inv;
    sy *= inv;
    sz *= inv;
    T w = q[0], x = q[1], y = q[2], z = q[3];
    T nw = w * sw - (x * sx + y * sy + z * sz);
    T nx = sx * w + x * sw + (y * sz - z * sy);
    T ny = sy * w + y * sw + (z * sx - x * sz);
    T nz = sz * w + z * sw + (x * sy - y * sx);
    T r = one / sqrt(nw * nw + nx * nx + ny * ny + nz * nz);
    q[0] = nw * r;
    q[1] = nx * r;
    q[2] = ny * r;
    q[3] = nz * r;
}

// A QuaternionParam: its reference a unit quaternion (4 values), its step
// a rotation vector, its values R(q_ref) R(v) (9) and their derivatives
// (27).
KERNEL_THREADS(256) void k_quat_values(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                       GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                       GLOBAL F* RESTRICT val, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    V m[9];
    quat_matrix(refr + e * 4, m);
    rotvec_apply<false>(m, (V)ev[a], (V)ev[a + 1], (V)ev[a + 2], wide + e * 9, val + e * 9, val);
}

KERNEL_THREADS(256) void k_quat_derivatives(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                            GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                            GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    V m[9];
    quat_matrix(refr + e * 4, m);
    rotvec_apply<true>(m, (V)ev[a], (V)ev[a + 1], (V)ev[a + 2], wide + e * 9, val + e * 9, der + e * 27);
}

KERNEL_THREADS(256) void k_quat_advance(GLOBAL V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                        GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    quat_advance(refr + e * 4, (V)ev[a], (V)ev[a + 1], (V)ev[a + 2]);
}

// carry(w, d) = d + (w x d) / 2 + (w x (w x d)) / 6, the translation of a
// twist step (crate::twist::carry).
template <typename T>
DEVICE void carry3(THREAD const T* w, THREAD const T* d, THREAD T* t) {
    T one_half = 0.5f, sixth = (T)1.0f / (T)6.0f;
    T wd[3], wwd[3];
    cross3(w, d, wd);
    cross3(w, wd, wwd);
#pragma unroll
    for (u32 i = 0; i < 3; i++) t[i] = d[i] + wd[i] * one_half + wwd[i] * sixth;
}

// A TransformParam, as TransformParam::__precompute_symbolic: its reference
// the rotation's unit quaternion and the translation (7 values), its step
// w (rotation vector) then d (translation rate); its values the rotation
// R(q_ref) R(w) and the translation t_ref + R(q_ref) carry(w, d) (12), its
// derivatives the rotation's by w (27), the translation's by d (9) and by
// w (9), k-major.
template <bool DERIVS, typename T>
DEVICE void transform_compose(GLOBAL const T* RESTRICT refr, GLOBAL const F* RESTRICT ev, u32 a, u32 e,
                              GLOBAL W* RESTRICT wide, GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der) {
    GLOBAL const T* r = refr + e * 7;
    T m[9];
    quat_matrix(r, m);
    T w[3] = {(T)ev[a], (T)ev[a + 1], (T)ev[a + 2]};
    T d[3] = {(T)ev[a + 3], (T)ev[a + 4], (T)ev[a + 5]};
    GLOBAL W* ow = wide + e * 12;
    GLOBAL F* ov = val + e * 12;
    GLOBAL F* od = der + e * 45;
    rotvec_apply<DERIVS>(m, w[0], w[1], w[2], ow, ov, od);
    T c[3], mc[3];
    carry3(w, d, c);
    mat_vec(m, c, mc);
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
        T t = r[4 + i] + mc[i];
        ow[9 + i] = W(t);
        ov[9 + i] = (F)t;
    }
    if (!DERIVS) return;
    T one_half = 0.5f, sixth = (T)1.0f / (T)6.0f, zero = 0.0f, one = 1.0f;
    T wd[3];
    cross3(w, d, wd);
#pragma unroll
    for (u32 k = 0; k < 3; k++) {
        T ek[3] = {k == 0 ? one : zero, k == 1 ? one : zero, k == 2 ? one : zero};
        // d(carry)/dd_k = carry(w, e_k)
        T t[3], mt[3];
        carry3(w, ek, t);
        mat_vec(m, t, mt);
#pragma unroll
        for (u32 i = 0; i < 3; i++) od[27 + k * 3 + i] = (F)mt[i];
        // d(carry)/dw_k = (e_k x d) / 2 + (e_k x (w x d) + w x (e_k x d)) / 6
        T ed[3], ewd[3], wed[3];
        cross3(ek, d, ed);
        cross3(ek, wd, ewd);
        cross3(w, ed, wed);
#pragma unroll
        for (u32 i = 0; i < 3; i++) t[i] = ed[i] * one_half + (ewd[i] + wed[i]) * sixth;
        mat_vec(m, t, mt);
#pragma unroll
        for (u32 i = 0; i < 3; i++) od[36 + k * 3 + i] = (F)mt[i];
    }
}

KERNEL_THREADS(256) void k_transform_values(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                            GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                            GLOBAL F* RESTRICT val, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    transform_compose<false>(refr, ev, at[e], e, wide, val, val);
}

KERNEL_THREADS(256) void k_transform_derivatives(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                                 GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                                 GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    transform_compose<true>(refr, ev, at[e], e, wide, val, der);
}

// The reference moved by the step, as TransformParam's update: the
// translation by R(q_ref) carry(w, d) (the rotation before the move), then
// the rotation by w.
KERNEL_THREADS(256) void k_transform_advance(GLOBAL V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                             GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    GLOBAL V* r = refr + e * 7;
    V m[9];
    quat_matrix(r, m);
    V w[3] = {(V)ev[a], (V)ev[a + 1], (V)ev[a + 2]};
    V d[3] = {(V)ev[a + 3], (V)ev[a + 4], (V)ev[a + 5]};
    V c[3], mc[3];
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
template <bool DERIVS, typename T>
DEVICE void unitvec_compose(GLOBAL const T* RESTRICT refr, GLOBAL const F* RESTRICT ev, u32 a, u32 e,
                            GLOBAL W* RESTRICT wide, GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der) {
    T m[9];
    quat_matrix(refr + e * 4, m);
    T dx = (T)ev[a], dy = (T)ev[a + 1];
    T one = 1.0f, two = 2.0f, quarter = 0.25f, one_half = 0.5f;
    T u = dx * dx + dy * dy;
    T s2 = one + u * quarter;
    T local[3] = {one - u / (two * s2), dy / s2, -dx / s2};
    T o[3];
    mat_vec(m, local, o);
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
        wide[e * 3 + i] = W(o[i]);
        val[e * 3 + i] = (F)o[i];
    }
    if (!DERIVS) return;
    T inv2 = one / (s2 * s2);
    T lx[3] = {-dx * inv2, -(dx * dy * one_half) * inv2, -(s2 - dx * dx * one_half) * inv2};
    T ly[3] = {-dy * inv2, (s2 - dy * dy * one_half) * inv2, (dx * dy * one_half) * inv2};
    T ox[3], oy[3];
    mat_vec(m, lx, ox);
    mat_vec(m, ly, oy);
#pragma unroll
    for (u32 i = 0; i < 3; i++) {
        der[e * 6 + i] = (F)ox[i];
        der[e * 6 + 3 + i] = (F)oy[i];
    }
}

KERNEL_THREADS(256) void k_unitvec_values(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                          GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                          GLOBAL F* RESTRICT val, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    unitvec_compose<false>(refr, ev, at[e], e, wide, val, val);
}

KERNEL_THREADS(256) void k_unitvec_derivatives(GLOBAL const V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                               GLOBAL const u32* RESTRICT at, GLOBAL W* RESTRICT wide,
                                               GLOBAL F* RESTRICT val, GLOBAL F* RESTRICT der, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    unitvec_compose<true>(refr, ev, at[e], e, wide, val, der);
}

// The reference moved by the step, as UnitVecParam's update: q_ref by the
// rotation vector (0, d.x, d.y).
KERNEL_THREADS(256) void k_unitvec_advance(GLOBAL V* RESTRICT refr, GLOBAL const F* RESTRICT ev,
                                           GLOBAL const u32* RESTRICT at, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    quat_advance(refr + e * 4, (V)0.0f, (V)ev[a], (V)ev[a + 1]);
}

// n values of x zeroed: an assembly's targets before the sums add into them
// (the cost in W; H and the gradient in V). k_zero_values, for H's
// millions, takes 4 values a thread, i, i + m, i + 2m and i + 3m with
// m = ceil(n / 4) (launched over m): one a thread left an Apple GPU
// writing f32 at under half its bandwidth.
KERNEL_THREADS(256) void k_zero_wide(GLOBAL W* RESTRICT x, u32 n) {
    u32 i = global_x();
    if (i < n) x[i] = 0.0f;
}
KERNEL_THREADS(256) void k_zero_values(GLOBAL V* RESTRICT x, u32 n) {
    u32 m = (n + 3) / 4;
    u32 i = global_x();
    if (i >= m) return;
#pragma unroll
    for (u32 k = 0; k < 4; k++)
        if (i + k * m < n) x[i + k * m] = 0.0f;
}

// The steps of x zeroed, one thread per entity: the len global indices at
// idx[at[e]..], a fixed one left alone; what re-centring does to the
// parameters after an accepted step.
KERNEL_THREADS(256) void k_zero_steps(GLOBAL V* RESTRICT x, GLOBAL const u32* RESTRICT idx,
                                      GLOBAL const u32* RESTRICT at, u32 len, u32 n) {
    u32 e = global_x();
    if (e >= n) return;
    u32 a = at[e];
    for (u32 k = 0; k < len; k++) {
        u32 g = idx[a + k];
        if (g != FIXED) x[g] = 0.0f;
    }
}

// The block's sum of its threads' v into part[b], in C (the root's CACC):
// a shared-memory tree over a block of 256 threads, in `sh` (256 values of
// the kernel's shared memory); t is the thread, b the block. Every thread
// of the block calls it.
template <typename C> DEVICE void cost_partial(GLOBAL C* part, C v, SHARED_PTR C* sh, u32 t, u32 b) {
    sh[t] = v;
    BARRIER();
    for (u32 span = 128; span > 0; span >>= 1) {
        if (t < span) sh[t] += sh[t + span];
        BARRIER();
    }
    if (t == 0) part[b] = sh[0];
}

// cost[0] += the sum of part[0..m), in C, by one block of 256 threads,
// `sh` and t as cost_partial's: the cost (W) as C sums it, constraint after
// constraint.
template <typename C> DEVICE void cost_sum(GLOBAL const C* part, u32 m, GLOBAL W* cost, SHARED_PTR C* sh, u32 t) {
    C acc = 0.0f;
    for (u32 i = t; i < m; i += 256) acc += part[i];
    sh[t] = acc;
    BARRIER();
    for (u32 span = 128; span > 0; span >>= 1) {
        if (t < span) sh[t] += sh[t + span];
        BARRIER();
    }
    if (t == 0) cost[0] = W(C(cost[0]) + sh[0]);
}

// One entity side's sums, one thread per (entity slot, entry) over `slots`
// slots: an entity of N parameters has N (N + 1) / 2 entries of its tile's
// upper triangle, then N of the gradient. Entry (i, j) of the tile sums
// 2 d_i d_j, entry i of the gradient 2 r d_i, over the rows of the slot's
// instances inst[off[s]..off[s + 1]], its derivatives at OFF in each
// row of STRIDE values of a record of R rows. idx and tile are the entity
// array's global indices and tiles; t is the thread's global index.
template <u32 N, u32 R, u32 STRIDE, u32 OFF>
DEVICE void reduce_entity(GLOBAL const F* RESTRICT rec, GLOBAL const u32* RESTRICT off, GLOBAL const u32* RESTRICT inst,
                          GLOBAL const u32* RESTRICT idx, GLOBAL const u32* RESTRICT tile, GLOBAL V* RESTRICT h,
                          GLOBAL V* RESTRICT g, u32 slots, u32 t) {
    const u32 nt = N * (N + 1) / 2;
    const u32 ne = nt + N;
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
    V acc = 0.0f;
    for (u32 q = off[s]; q < off[s + 1]; q++) {
        GLOBAL const F* r = rec + inst[q] * (R * STRIDE);
#pragma unroll
        for (u32 row = 0; row < R; row++) {
            GLOBAL const F* rr = r + row * STRIDE;
            acc = v_fma(acc, rr[OFF + i], grad ? rr[0] : rr[OFF + j]);
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
// ones in tile_a, tile_b). A's derivatives sit at OFFA in each row, B's at
// OFFB. An instance of the pair the other way round (its A the first's B:
// two entities of one array, NA = NB) adds dA_j dB_i. A slot whose two
// sides are one entity (aliased refs) sums into that entity's own tile,
// the upper triangle: (i, j) and (j, i) land in one cell, which the thread
// of i <= j sums, dA_i dB_j + dA_j dB_i (2 dA_i dB_i on the diagonal). t
// is the thread's global index.
template <u32 NA, u32 NB, u32 R, u32 STRIDE, u32 OFFA, u32 OFFB>
DEVICE void reduce_cross(GLOBAL const F* RESTRICT rec, GLOBAL const u32* RESTRICT off, GLOBAL const u32* RESTRICT inst,
                         GLOBAL const u32* RESTRICT slot_a, GLOBAL const u32* RESTRICT slot_b,
                         GLOBAL const u32* RESTRICT idx_a, GLOBAL const u32* RESTRICT tile_a,
                         GLOBAL const u32* RESTRICT idx_b, GLOBAL const u32* RESTRICT tile_b,
                         GLOBAL const u32* RESTRICT tile_x, GLOBAL V* RESTRICT h, u32 slots, u32 t) {
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
    u32 start_a = tile_a[sa * 3 + 2], start_b = tile_b[sb * 3 + 2];
    const bool alias = NA == NB && start_a == start_b;
    if (alias && i > j) return;
    V acc = 0.0f;
    u32 base = tile_x[x * 3], stride = tile_x[x * 3 + 1], flip = tile_x[x * 3 + 2];
    if (alias) {
        for (u32 q = q0; q < q1; q++) {
            GLOBAL const F* r = rec + inst[q] * (R * STRIDE);
#pragma unroll
            for (u32 row = 0; row < R; row++) {
                GLOBAL const F* rr = r + row * STRIDE;
                acc = v_fma(acc, rr[OFFA + i], rr[OFFB + j]);
                acc = v_fma(acc, rr[OFFA + j], rr[OFFB + i]);
            }
        }
        acc += acc;
        u32 lo = (gi < gj ? gi : gj) - start_a, hi = (gi < gj ? gj : gi) - start_a;
        h[base + hi * stride + lo] += acc;
        return;
    }
    for (u32 q = q0; q < q1; q++) {
        GLOBAL const F* r = rec + inst[q] * (R * STRIDE);
        bool reversed = NA == NB && slot_a[inst[q]] != sa;
        u32 ia = reversed ? OFFA + j : OFFA + i;
        u32 jb = reversed ? OFFB + i : OFFB + j;
#pragma unroll
        for (u32 row = 0; row < R; row++) {
            GLOBAL const F* rr = r + row * STRIDE;
            acc = v_fma(acc, rr[ia], rr[jb]);
        }
    }
    acc += acc;
    u32 ra = gi - start_a;
    u32 cb = gj - start_b;
    h[flip == 0 ? base + cb * stride + ra : base + ra * stride + cb] += acc;
}

// A column of the layout table: block base plus the byte offset in slot k
// of L, as T (COL_MUT: writable). The generated constraint kernels read
// every column this way.
#define COL(base, T, k) ((GLOBAL const T*)((base) + L[ARAEL_COL_##k]))
#define COL_MUT(base, T, k) ((GLOBAL T*)((base) + L[ARAEL_COL_##k]))
