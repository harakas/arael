// wide.h: W, the wide scalar of a GPU solve: what an f64 model's values
// (H, the gradient, the step, the LM loop's vectors, the references), the
// cost and the Schur reduction are kept and summed in (an f32 model's values
// are f32: V and sin_cos, at the end). `double`, or with ARAEL_WIDE_DF32
// defined `df32`, a value as the sum of two floats (about 48 bits of
// significand, f32's exponent range), for devices whose f64 is slow (most
// consumer cards) or absent (Metal).
//
// A kernel is written once over W, with:
//   W(x)                    an f32 (or, with HAS_F64, an f64 explicitly)
//   (float)w                the value rounded to f32
//   ARAEL_WIDE(d, hi, lo)   a constant: d as a double literal, hi and lo its
//                           two-float split (the host's df32::from_f64)
//   + - * / and unary -, += -= *= /=, < > <= >= == !=
//   sqrt, fabs, isnan, isinf, isfinite, floor, copysign,
//   exp, expm1, log, log1p, log2, log10, pow,
//   sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh,
//   wide_max, wide_min (a NaN loses to a number),
//   wide_sin_cos, wide_fma(acc, x, y) (acc + x y, for sums of products)
//
// Both forms are 8 bytes, so a W array has the layout of a double array; a
// df32 is (hi, lo) as two floats.
//
// df32 is arael's host df32 (src/twofloat.rs), operation for operation:
// every add and product rounded on its own (add_rn, mul_rn: never fused),
// the product's error by fma, so a df32 result matches the host's bit for bit
// (log and atan2, which start from the f32 library's value, to df32's
// precision).

#ifdef ARAEL_WIDE_DF32

// a + b as the sum and its exact rounding error (Knuth).
DEVICE void two_sum(float a, float b, THREAD float& s, THREAD float& e) {
    s = add_rn(a, b);
    float bb = sub_rn(s, a);
    e = add_rn(sub_rn(a, sub_rn(s, bb)), sub_rn(b, bb));
}

// a + b as the sum and its error, for |a| >= |b| or a == 0.
DEVICE void quick_two_sum(float a, float b, THREAD float& s, THREAD float& e) {
    s = add_rn(a, b);
    e = sub_rn(b, sub_rn(s, a));
}

// a * b as the product and its exact rounding error.
DEVICE void two_prod(float a, float b, THREAD float& p, THREAD float& e) {
    p = mul_rn(a, b);
    e = fma(a, b, -p);
}

struct df32 {
    float hi, lo;
#if ARAEL_METAL
    // Metal constructs and converts an object in its address space: a
    // df32 in shared or global memory needs members of that space.
    df32() thread = default;
    df32() threadgroup = default;
    df32() device = default;
#else
    df32() = default;
#endif
    // the parts as they are: a constant's split, or a normalized pair
    DEVICE constexpr df32(float h, float l) : hi(h), lo(l) {}
    DEVICE df32(float x) : hi(x), lo(0.0f) {}
#if HAS_F64
    DEVICE explicit df32(double x) {
        hi = (float)x;
        lo = isfinite(hi) ? (float)(x - (double)hi) : 0.0f;
    }
    DEVICE explicit operator double() const { return (double)hi + (double)lo; }
#endif
#if ARAEL_METAL
    DEVICE explicit operator float() const thread { return add_rn(hi, lo); }
    DEVICE explicit operator float() const threadgroup { return add_rn(hi, lo); }
    DEVICE explicit operator float() const device { return add_rn(hi, lo); }
    DEVICE explicit operator float() const constant { return add_rn(hi, lo); }
#else
    DEVICE explicit operator float() const { return add_rn(hi, lo); }
#endif
};

// A normalized (hi, lo); a non-finite hi carries no trailing part.
DEVICE df32 df32_pack(float hi, float lo) { return isfinite(hi) ? df32(hi, lo) : df32(hi, 0.0f); }

DEVICE df32 operator+(df32 a, df32 b) {
    float s1, s2, t1, t2;
    two_sum(a.hi, b.hi, s1, s2);
    if (!isfinite(s1)) return df32(s1);
    two_sum(a.lo, b.lo, t1, t2);
    s2 = add_rn(s2, t1);
    quick_two_sum(s1, s2, s1, s2);
    s2 = add_rn(s2, t2);
    float hi, lo;
    quick_two_sum(s1, s2, hi, lo);
    return df32_pack(hi, lo);
}

DEVICE df32 operator-(df32 a) { return df32(-a.hi, -a.lo); }
DEVICE df32 operator-(df32 a, df32 b) { return a + (-b); }

DEVICE df32 operator*(df32 a, df32 b) {
    float p, e;
    two_prod(a.hi, b.hi, p, e);
    if (!isfinite(p)) return df32(p);
    e = add_rn(e, add_rn(mul_rn(a.hi, b.lo), mul_rn(a.lo, b.hi)));
    float hi, lo;
    quick_two_sum(p, e, hi, lo);
    return df32_pack(hi, lo);
}

// a * b for an f32 b.
DEVICE df32 df32_mul_f32(df32 a, float b) {
    float p, e;
    two_prod(a.hi, b, p, e);
    if (!isfinite(p)) return df32(p);
    e = add_rn(e, mul_rn(a.lo, b));
    float hi, lo;
    quick_two_sum(p, e, hi, lo);
    return df32_pack(hi, lo);
}

// a * a.
DEVICE df32 df32_sqr(df32 a) {
    float p, e;
    two_prod(a.hi, a.hi, p, e);
    if (!isfinite(p)) return df32(p);
    e = add_rn(e, mul_rn(mul_rn(2.0f, a.hi), a.lo));
    float hi, lo;
    quick_two_sum(p, e, hi, lo);
    return df32_pack(hi, lo);
}

DEVICE df32 operator/(df32 a, df32 b) {
    float q1 = a.hi / b.hi;
    // f32's answer for infinities, NaN and division by zero
    if (!isfinite(a.hi) || !isfinite(b.hi) || b.hi == 0.0f || !isfinite(q1)) return df32(q1);
    df32 r = a - df32_mul_f32(b, q1);
    float q2 = r.hi / b.hi;
    r = r - df32_mul_f32(b, q2);
    float q3 = r.hi / b.hi;
    float s, e;
    quick_two_sum(q1, q2, s, e);
    return df32(s, e) + df32(q3);
}

// The compound assignments, on Metal for a df32 in each address space.
#define ARAEL_DF32_ASSIGN(SPACE)                                                      \
    DEVICE SPACE df32& operator+=(SPACE df32& a, df32 b) { a = a + b; return a; } \
    DEVICE SPACE df32& operator-=(SPACE df32& a, df32 b) { a = a - b; return a; } \
    DEVICE SPACE df32& operator*=(SPACE df32& a, df32 b) { a = a * b; return a; } \
    DEVICE SPACE df32& operator/=(SPACE df32& a, df32 b) { a = a / b; return a; }
#if ARAEL_METAL
ARAEL_DF32_ASSIGN(thread)
ARAEL_DF32_ASSIGN(threadgroup)
ARAEL_DF32_ASSIGN(device)
#else
ARAEL_DF32_ASSIGN()
#endif
#undef ARAEL_DF32_ASSIGN

// Ordered by hi, then lo; anything with a NaN is unordered.
DEVICE bool operator<(df32 a, df32 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
DEVICE bool operator>(df32 a, df32 b) { return b < a; }
DEVICE bool operator<=(df32 a, df32 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo); }
DEVICE bool operator>=(df32 a, df32 b) { return b <= a; }
DEVICE bool operator==(df32 a, df32 b) { return a.hi == b.hi && a.lo == b.lo; }
DEVICE bool operator!=(df32 a, df32 b) { return !(a == b); }

DEVICE bool isnan(df32 a) { return isnan(a.hi); }
DEVICE bool isfinite(df32 a) { return isfinite(a.hi); }
DEVICE df32 fabs(df32 a) { return a.hi < 0.0f ? -a : a; }

// Karp's trick: one correction of an f32 square root.
DEVICE df32 sqrt(df32 a) {
    if (a.hi <= 0.0f || !isfinite(a.hi)) return df32(sqrt(a.hi));
    float x = 1.0f / sqrt(a.hi);
    float ax = mul_rn(a.hi, x);
    float diff = (a - df32_sqr(df32(ax))).hi;
    float hi, lo;
    quick_two_sum(ax, mul_rn(diff, mul_rn(x, 0.5f)), hi, lo);
    return df32_pack(hi, lo);
}

// `v` of lane (this lane XOR mask), for warp_sum. Every lane of the warp calls it.
DEVICE df32 shfl_xor(df32 v, u32 mask) { return df32(shfl_xor(v.hi, mask), shfl_xor(v.lo, mask)); }

DEVICE df32 wide_max(df32 a, df32 b) { return isnan(a) ? b : isnan(b) ? a : a >= b ? a : b; }
DEVICE df32 wide_min(df32 a, df32 b) { return isnan(a) ? b : isnan(b) ? a : a <= b ? a : b; }

// The host df32's constants (df32::from_f64 of the f64 ones).
#define DF32_PI df32(3.141592741e+00f, -8.742277657e-08f)
#define DF32_FRAC_PI_2 df32(1.570796371e+00f, -4.371138829e-08f)
#define DF32_LN_2 df32(6.931471825e-01f, -1.904654212e-09f)
#define DF32_LN_10 df32(2.302585125e+00f, -3.197543563e-08f)
#define DF32_NAN df32(__int_as_float(0x7fc00000))
#define DF32_INFINITY df32(__int_as_float(0x7f800000))
#define DF32_NEG_INFINITY df32(-__int_as_float(0x7f800000))

// a - k c for an integer-valued k and a constant c given as a df32 and a
// third f32 part: both products exact, so the remainder keeps c to about
// 72 bits times k. The argument reduction of exp and sin_cos.
DEVICE df32 df32_reduce(df32 a, float k, df32 c, float c3) {
    float p1, e1, p2, e2;
    two_prod(c.hi, k, p1, e1);
    two_prod(c.lo, k, p2, e2);
    return ((((a - df32(p1)) - df32(e1)) - df32(p2)) - df32(e2)) - df32(mul_rn(c3, k));
}

// sin and cos: x = k pi/2 + r with |r| <= pi/4, the constant in three f32
// parts so the remainder keeps its digits, then the Taylor series of r.
DEVICE df32 df32_sin_taylor(df32 r) {
    const df32 c[9] = {df32(1.000000000e+00f, 0.000000000e+00f), df32(-1.666666716e-01f, 4.967053879e-09f),
                       df32(8.333333768e-03f, -4.346172033e-10f), df32(-1.984127011e-04f, 2.725596875e-12f),
                       df32(2.755731884e-06f, 3.793571224e-14f), df32(-2.505210794e-08f, -4.417623045e-16f),
                       df32(1.605904437e-10f, -5.352526512e-18f), df32(-7.647163610e-13f, -1.220071047e-20f),
                       df32(2.811457359e-15f, -1.046208474e-22f)};
    df32 x2 = df32_sqr(r);
    df32 s = c[8];
#pragma unroll
    for (int i = 7; i >= 0; i--) s = s * x2 + c[i];
    return s * r;
}

DEVICE df32 df32_cos_taylor(df32 r) {
    const df32 c[9] = {df32(1.000000000e+00f, 0.000000000e+00f), df32(-5.000000000e-01f, 0.000000000e+00f),
                       df32(4.166666791e-02f, -1.241763470e-09f), df32(-1.388888923e-03f, 3.363109444e-11f),
                       df32(2.480158764e-05f, -3.406996094e-13f), df32(-2.755731998e-07f, 7.575112209e-15f),
                       df32(2.087675588e-09f, 1.108283981e-16f), df32(-1.147074536e-11f, -2.372207689e-19f),
                       df32(4.779477256e-14f, 7.625444044e-22f)};
    df32 x2 = df32_sqr(r);
    df32 s = c[8];
#pragma unroll
    for (int i = 7; i >= 0; i--) s = s * x2 + c[i];
    return s;
}

struct WidePair {
    df32 s, c;
};

DEVICE WidePair wide_sin_cos(df32 x) {
    if (!isfinite(x.hi)) {
        float nan = sub_rn(x.hi, x.hi);
        return WidePair{df32(nan), df32(nan)};
    }
    float k = round(mul_rn(x.hi, 6.366197467e-01f));
    df32 r = df32_reduce(x, k, DF32_FRAC_PI_2, -1.776356839e-15f);
    df32 s = df32_sin_taylor(r);
    df32 c = df32_cos_taylor(r);
    // the quadrant, k mod 4: exact for the integer k at any magnitude (an
    // integer conversion of k overflows past 2^63)
    float m = fmod(k, 4.0f);
    int q = (int)(m < 0.0f ? add_rn(m, 4.0f) : m);
    if (q == 0) return WidePair{s, c};
    if (q == 1) return WidePair{c, -s};
    if (q == 2) return WidePair{-s, -c};
    return WidePair{-c, s};
}

// The rest of the host df32's functions, as src/twofloat.rs computes them,
// for a `with_wide` scope computed in df32. log and atan2 start from the
// f32 library's value, which may differ from the host's in its last bit;
// their Newton steps take both to df32's precision.

DEVICE bool isinf(df32 a) { return isinf(a.hi); }

// a with b's sign.
DEVICE df32 copysign(df32 a, df32 b) {
    return (copysign(1.0f, a.hi) < 0.0f) == (copysign(1.0f, b.hi) < 0.0f) ? a : -a;
}

DEVICE df32 floor(df32 a) {
    float hi = floor(a.hi);
    if (hi != a.hi) return df32(hi);
    float h, l;
    quick_two_sum(hi, floor(a.lo), h, l);
    return df32_pack(h, l);
}

// 2^k, exactly, for k in [-126, 127].
DEVICE float df32_pow2(int k) { return __int_as_float((k + 127) << 23); }

// a 2^k, exact while the result stays normal, for |k| up to 252.
DEVICE df32 df32_scale_pow2(df32 a, int k) {
    int k1 = k / 2;
    float f1 = df32_pow2(k1), f2 = df32_pow2(k - k1);
    return df32_pack(mul_rn(mul_rn(a.hi, f1), f2), mul_rn(mul_rn(a.lo, f1), f2));
}

// a as m 2^e with m in [0.5, 1); zero gives (0, 0).
DEVICE df32 df32_frexp(df32 a, THREAD int& e) {
    e = 0;
    if (a.hi == 0.0f) return a;
    int shift = 0;
    if (fabs(a.hi) < 1.175494351e-38f) {
        a = df32_scale_pow2(a, 48);
        shift = -48;
    }
    int x = ((__float_as_int(fabs(a.hi)) >> 23) & 0xff) - 126;
    e = x + shift;
    return df32_scale_pow2(a, -x);
}

// exp(r) for |r| <= 0.18 by its Taylor series to degree 11.
DEVICE df32 df32_exp_taylor(df32 r) {
    df32 p = df32(2.505210794e-08f, 4.417623045e-16f);
    p = p * r + df32(2.755731998e-07f, -7.575112209e-15f);
    p = p * r + df32(2.755731884e-06f, 3.793571224e-14f);
    p = p * r + df32(2.480158764e-05f, -3.406996094e-13f);
    p = p * r + df32(1.984127011e-04f, -2.725596875e-12f);
    p = p * r + df32(1.388888923e-03f, -3.363109444e-11f);
    p = p * r + df32(8.333333768e-03f, -4.346172033e-10f);
    p = p * r + df32(4.166666791e-02f, -1.241763470e-09f);
    p = p * r + df32(1.666666716e-01f, -4.967053879e-09f);
    p = p * r + df32(0.5f);
    p = p * r + df32(1.0f);
    return p * r + df32(1.0f);
}

DEVICE df32 exp(df32 a) {
    if (isnan(a.hi)) return DF32_NAN;
    if (a.hi > 88.8f) return DF32_INFINITY;
    if (a.hi < -104.0f) return df32(0.0f);
    // a = k ln2 + r, |r| <= 0.35: the series on r / 2, squared once
    float k = round(a.hi / 6.931471825e-01f);
    df32 r = df32_scale_pow2(df32_reduce(a, k, DF32_LN_2, -1.110223025e-16f), -1);
    return df32_scale_pow2(df32_sqr(df32_exp_taylor(r)), (int)k);
}

DEVICE df32 log(df32 a) {
    if (isnan(a.hi) || a.hi < 0.0f) return DF32_NAN;
    if (a.hi == 0.0f) return DF32_NEG_INFINITY;
    if (isinf(a.hi)) return DF32_INFINITY;
    // log(m 2^e) = log m + e log 2, m in [0.5, 1): two Newton steps on exp
    // from the f32 logarithm
    int e;
    df32 m = df32_frexp(a, e);
    df32 y = df32(log(m.hi));
    y = (y + m * exp(-y)) - df32(1.0f);
    y = (y + m * exp(-y)) - df32(1.0f);
    return y + df32_mul_f32(DF32_LN_2, (float)e);
}

DEVICE df32 log2(df32 a) { return log(a) / DF32_LN_2; }
DEVICE df32 log10(df32 a) { return log(a) / DF32_LN_10; }

// a^n by squaring.
DEVICE df32 df32_powi(df32 a, int n) {
    if (n == 0) return df32(1.0f);
    df32 r = df32(1.0f), base = a;
    u32 e = n < 0 ? (u32)(-(i64)n) : (u32)n;
    while (e > 0u) {
        if (e & 1u) r = r * base;
        e >>= 1;
        if (e > 0u) base = df32_sqr(base);
    }
    return n < 0 ? df32(1.0f) / r : r;
}

DEVICE df32 pow(df32 a, df32 n) {
    // an integer n below 2^31 by squaring, hi + lo exact
    if (isfinite(n.hi) && floor(n.hi) == n.hi && floor(n.lo) == n.lo && fabs(n.hi) < 2147483648.0f)
        return df32_powi(a, (int)((i64)n.hi + (i64)n.lo));
    if (a.hi == 0.0f) return n.hi > 0.0f ? df32(0.0f) : DF32_INFINITY;
    if (a.hi < 0.0f) return DF32_NAN;
    return exp(n * log(a));
}

DEVICE df32 sin(df32 a) { return wide_sin_cos(a).s; }
DEVICE df32 cos(df32 a) { return wide_sin_cos(a).c; }
DEVICE df32 tan(df32 a) {
    WidePair p = wide_sin_cos(a);
    return p.s / p.c;
}

DEVICE df32 atan2(df32 y, df32 x) {
    if (!(isfinite(x.hi) && isfinite(y.hi))) return df32(atan2(y.hi, x.hi));
    if (x.hi == 0.0f) {
        // both zero: +-0 from +0, +-pi from -0, exactly, as the host's
        if (y.hi == 0.0f) {
            if (copysign(1.0f, x.hi) > 0.0f) return df32(y.hi);
            return copysign(1.0f, y.hi) < 0.0f ? -DF32_PI : DF32_PI;
        }
        return y.hi > 0.0f ? DF32_FRAC_PI_2 : -DF32_FRAC_PI_2;
    }
    if (y.hi == 0.0f) {
        if (x.hi > 0.0f) return df32(atan2(y.hi, x.hi));
        return copysign(1.0f, y.hi) < 0.0f ? -DF32_PI : DF32_PI;
    }
    // Newton on sin and cos from the f32 angle, twice: the direction
    // (xx, yy) is unit, so the step reads straight off the residual
    df32 r = sqrt(df32_sqr(x) + df32_sqr(y));
    df32 xx = x / r, yy = y / r;
    df32 z = df32(atan2((float)y, (float)x));
    for (int i = 0; i < 2; i++) {
        WidePair p = wide_sin_cos(z);
        if (fabs(xx.hi) > fabs(yy.hi)) z = z + (yy - p.s) / p.c;
        else z = z - (xx - p.c) / p.s;
    }
    return z;
}

DEVICE df32 atan(df32 a) { return atan2(a, df32(1.0f)); }

// (1 - a)(1 + a) rather than 1 - a^2: exact factors near a = +-1.
DEVICE df32 asin(df32 a) {
    if (fabs(a.hi) > 1.0f || isnan(a.hi)) return DF32_NAN;
    if (fabs(a.hi) == 1.0f && a.lo == 0.0f) return a.hi > 0.0f ? DF32_FRAC_PI_2 : -DF32_FRAC_PI_2;
    return atan2(a, sqrt((df32(1.0f) - a) * (df32(1.0f) + a)));
}

DEVICE df32 acos(df32 a) {
    if (fabs(a.hi) > 1.0f || isnan(a.hi)) return DF32_NAN;
    if (fabs(a.hi) == 1.0f && a.lo == 0.0f) return a.hi > 0.0f ? df32(0.0f) : DF32_PI;
    return atan2(sqrt((df32(1.0f) - a) * (df32(1.0f) + a)), a);
}

// exp(a) - 1; below |a| = 0.5 by its series to degree 15.
DEVICE df32 expm1(df32 a) {
    if (fabs(a.hi) >= 0.5f || !isfinite(a.hi)) return exp(a) - df32(1.0f);
    df32 p = df32(7.647163610e-13f, 1.220071047e-20f);
    p = p * a + df32(1.147074536e-11f, 2.372207689e-19f);
    p = p * a + df32(1.605904437e-10f, -5.352526512e-18f);
    p = p * a + df32(2.087675588e-09f, 1.108283981e-16f);
    p = p * a + df32(2.505210794e-08f, 4.417623045e-16f);
    p = p * a + df32(2.755731998e-07f, -7.575112209e-15f);
    p = p * a + df32(2.755731884e-06f, 3.793571224e-14f);
    p = p * a + df32(2.480158764e-05f, -3.406996094e-13f);
    p = p * a + df32(1.984127011e-04f, -2.725596875e-12f);
    p = p * a + df32(1.388888923e-03f, -3.363109444e-11f);
    p = p * a + df32(8.333333768e-03f, -4.346172033e-10f);
    p = p * a + df32(4.166666791e-02f, -1.241763470e-09f);
    p = p * a + df32(1.666666716e-01f, -4.967053879e-09f);
    p = p * a + df32(0.5f);
    p = p * a + df32(1.0f);
    return p * a;
}

// log(1 + a); below |a| = 0.1 as 2 atanh(u), u = a / (2 + a), odd powers
// of u to u^13.
DEVICE df32 log1p(df32 a) {
    if (fabs(a.hi) >= 0.1f || !isfinite(a.hi)) return log(df32(1.0f) + a);
    df32 u = a / (df32(2.0f) + a);
    df32 u2 = df32_sqr(u);
    df32 s = df32(7.692307979e-02f, -2.865607973e-09f);
    s = s * u2 + df32(9.090909362e-02f, -2.709302116e-09f);
    s = s * u2 + df32(1.111111119e-01f, -8.278422947e-10f);
    s = s * u2 + df32(1.428571492e-01f, -6.386212004e-09f);
    s = s * u2 + df32(2.000000030e-01f, -2.980232283e-09f);
    s = s * u2 + df32(3.333333433e-01f, -9.934107759e-09f);
    s = s * u2 + df32(1.0f);
    return (s * u) * df32(2.0f);
}

DEVICE df32 sinh(df32 a) {
    if (!isfinite(a.hi)) return a;
    if (fabs(a.hi) < 0.05f) {
        df32 x2 = df32_sqr(a);
        df32 s = df32(2.755731884e-06f, 3.793571224e-14f);
        s = s * x2 + df32(1.984127011e-04f, -2.725596875e-12f);
        s = s * x2 + df32(8.333333768e-03f, -4.346172033e-10f);
        s = s * x2 + df32(1.666666716e-01f, -4.967053879e-09f);
        s = s * x2 + df32(1.0f);
        return s * a;
    }
    // odd, from exp(|x|) >= 1: exp of a large negative x is tiny, its low
    // part subnormal (flushed on Metal); as the host's
    df32 e = exp(fabs(a));
    df32 s = (e - df32(1.0f) / e) * df32(0.5f);
    return a.hi < 0.0f ? -s : s;
}

DEVICE df32 cosh(df32 a) {
    if (!isfinite(a.hi)) return fabs(a);
    // even, from exp(|x|) >= 1, as sinh
    df32 e = exp(fabs(a));
    return (e + df32(1.0f) / e) * df32(0.5f);
}

DEVICE df32 tanh(df32 a) {
    if (isnan(a.hi)) return a;
    if (fabs(a.hi) > 20.0f) return df32(a.hi > 0.0f ? 1.0f : -1.0f);
    df32 e = expm1(a * df32(2.0f));
    return e / (e + df32(2.0f));
}

// acc + x y for a sum of products: the product split by fma, added by the
// "sloppy" two-float add (QD's), whose error is relative to |acc| + |x y|
// (about 2^-45 of it) rather than to the sum, as an f64 sum's is to its
// terms (2^-53). No branches: a non-finite term makes the sum non-finite
// (hi infinite or NaN, lo perhaps NaN), which is all a sum's checks read.
// Not the host's df32 add: for sums only.
DEVICE df32 wide_fma(df32 acc, df32 x, df32 y) {
    float p, e;
    two_prod(x.hi, y.hi, p, e);
    e = add_rn(e, add_rn(mul_rn(x.hi, y.lo), mul_rn(x.lo, y.hi)));
    float s, t;
    two_sum(acc.hi, p, s, t);
    t = add_rn(t, add_rn(acc.lo, e));
    float hi, lo;
    quick_two_sum(s, t, hi, lo);
    return df32(hi, lo);
}

// acc + x y for two f32 factors: the product exact (two_prod).
DEVICE df32 wide_fma(df32 acc, float x, float y) {
    float p, e;
    two_prod(x, y, p, e);
    float s, t;
    two_sum(acc.hi, p, s, t);
    t = add_rn(t, add_rn(acc.lo, e));
    float hi, lo;
    quick_two_sum(s, t, hi, lo);
    return df32(hi, lo);
}

#if HAS_F64
DEVICE df32 wide_fma(df32 acc, double x, double y) { return wide_fma(acc, df32(x), df32(y)); }
#endif

// v exactly: the f32 nearest it and what that left, |lo| <= 256.
DEVICE df32 wide_u32(u32 v) {
    float hi = (float)v;
    return df32(hi, (float)((i64)v - (i64)hi));
}

typedef df32 W;
#define ARAEL_WIDE(d, hi, lo) df32(hi, lo)

#else

typedef double W;
#define ARAEL_WIDE(d, hi, lo) (d)

struct WidePair {
    double s, c;
};

DEVICE WidePair wide_sin_cos(double x) { return WidePair{sin(x), cos(x)}; }
DEVICE double wide_u32(u32 v) { return (double)v; }
DEVICE double wide_fma(double acc, double x, double y) { return fma(x, y, acc); }
DEVICE double wide_max(double a, double b) { return isnan(a) ? b : isnan(b) ? a : a >= b ? a : b; }
DEVICE double wide_min(double a, double b) { return isnan(a) ? b : isnan(b) ? a : a <= b ? a : b; }

#endif

// sin_cos(x): x's sine and cosine in x's scalar, f32 or W (wide_sin_cos).
template <typename T> struct SinCos {
    T s, c;
};
DEVICE SinCos<float> sin_cos(float x) { return SinCos<float>{sin(x), cos(x)}; }
DEVICE SinCos<W> sin_cos(W x) {
    WidePair p = wide_sin_cos(x);
    return SinCos<W>{p.s, p.c};
}

// V: the scalar of a solve's values (H and the gradient), the model's as
// the host keeps them: f32 with VALUES_F32 1 (an f32 model's solve),
// else W. v_fma(acc, x, y): acc + x y in V.
#ifndef VALUES_F32
#define VALUES_F32 0
#endif
#if VALUES_F32
typedef float V;
DEVICE float v_fma(float acc, float x, float y) { return fma(x, y, acc); }
#else
typedef W V;
template <typename X, typename Y> DEVICE W v_fma(W acc, X x, Y y) { return wide_fma(acc, x, y); }
#endif
