// wide.h: W, the wide scalar of a device route: what H, the gradient, the
// cost, the Schur reduction and the LM loop's vectors are kept and summed
// in. `double`, or with ARAEL_WIDE_DF32 defined `df32`, a value as the sum
// of two floats (about 48 bits of significand, f32's exponent range), for
// devices without fast f64 (consumer cards, where f64 runs at 1/32 to 1/64
// of f32) or without f64 at all (Metal).
//
// A kernel is written once over W, with:
//   W(x)                    an f32 (or, with HAS_F64, an f64 explicitly)
//   (float)w                the value rounded to f32
//   ARAEL_WIDE(d, hi, lo)   a constant: d as a double literal, hi and lo its
//                           two-float split (the host's df32::from_f64)
//   + - * / and unary -, += -= *= /=, < > <= >= == !=
//   sqrt, fabs, isnan, wide_max, wide_min (a NaN loses to a number),
//   wide_sin_cos, wide_fma(acc, x, y) (acc + x y, for sums of products)
//
// Both forms are 8 bytes, so a W array has the layout of a double array; a
// df32 is (hi, lo) as two floats.
//
// df32 is arael's host df32 (src/twofloat.rs), operation for operation:
// every add and product rounded on its own (add_rn, mul_rn: never fused),
// the product's error by fma, so a df32 result matches the host's bit for bit.

#ifdef ARAEL_WIDE_DF32

// a + b as the sum and its exact rounding error (Knuth).
DEVICE void two_sum(float a, float b, float& s, float& e) {
    s = add_rn(a, b);
    float bb = sub_rn(s, a);
    e = add_rn(sub_rn(a, sub_rn(s, bb)), sub_rn(b, bb));
}

// a + b as the sum and its error, for |a| >= |b| or a == 0.
DEVICE void quick_two_sum(float a, float b, float& s, float& e) {
    s = add_rn(a, b);
    e = sub_rn(b, sub_rn(s, a));
}

// a * b as the product and its exact rounding error.
DEVICE void two_prod(float a, float b, float& p, float& e) {
    p = mul_rn(a, b);
    e = fmaf(a, b, -p);
}

struct df32 {
    float hi, lo;
    df32() = default;
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
    DEVICE explicit operator float() const { return add_rn(hi, lo); }
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

DEVICE df32& operator+=(df32& a, df32 b) { a = a + b; return a; }
DEVICE df32& operator-=(df32& a, df32 b) { a = a - b; return a; }
DEVICE df32& operator*=(df32& a, df32 b) { a = a * b; return a; }
DEVICE df32& operator/=(df32& a, df32 b) { a = a / b; return a; }

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
    if (a.hi <= 0.0f || !isfinite(a.hi)) return df32(sqrtf(a.hi));
    float x = 1.0f / sqrtf(a.hi);
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
    float k = roundf(mul_rn(x.hi, 6.366197467e-01f));
    // x - k pi/2, both products exact
    float p1, e1, p2, e2;
    two_prod(1.570796371e+00f, k, p1, e1);
    two_prod(-4.371138829e-08f, k, p2, e2);
    df32 r = ((((x - df32(p1)) - df32(e1)) - df32(p2)) - df32(e2)) - df32(mul_rn(-1.776356839e-15f, k));
    df32 s = df32_sin_taylor(r);
    df32 c = df32_cos_taylor(r);
    long long q = ((long long)k % 4 + 4) % 4;
    if (q == 0) return WidePair{s, c};
    if (q == 1) return WidePair{c, -s};
    if (q == 2) return WidePair{-s, -c};
    return WidePair{-c, s};
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
    return df32(hi, (float)((long long)v - (long long)hi));
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
