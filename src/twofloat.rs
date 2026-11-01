//! `df32`: a two-float scalar, about 48 bits of significand in f32
//! arithmetic.
//!
//! A `df32` is `hi + lo`, two f32 with `|lo| <= ulp(hi) / 2`. Every
//! operation is f32 arithmetic with fused multiply-add, so the type runs
//! on a target without f64 and gives the same results there: it is the
//! wide scalar of a `with_wide` scope and of a `MixedEulerAngleParam`
//! for an f32 model on such a target (`#[arael(root, f32, wide = df32)]`,
//! docs/MODEL.md). Relative precision is about 2^-46 per operation; the
//! exponent range is f32's.
//!
//! ```
//! use arael::df32;
//! let a = df32::from_f64(1.0 + 1e-10);
//! let b = df32::from_f32(1.0);
//! // The difference of two near-equal values is exact, which f32 loses.
//! assert_eq!((a - b).to_f64(), a.to_f64() - 1.0);
//! assert!((a.sqrt().to_f64() - (1.0f64 + 1e-10).sqrt()).abs() < 1e-14);
//! ```
//!
//! Conversions from and to f64 split and join the two halves exactly;
//! they are how the host writes constants and reads results, not
//! arithmetic the type depends on. The transcendental functions are
//! series and Newton steps in `df32` arithmetic (the algorithms of the QD
//! library's `dd_real`, sized for 48 bits); `sin_cos` reduces its
//! argument by `pi/2` with a three-part constant and keeps about 1e-13
//! absolute for arguments within some hundred radians.

#![allow(non_camel_case_types)]

use std::cmp::Ordering;
use std::fmt;
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};

/// A two-float scalar: `hi + lo`, with `|lo| <= ulp(hi) / 2`.
#[derive(Clone, Copy, Default, PartialEq)]
pub struct df32 {
    /// The leading f32: the value rounded to f32.
    pub hi: f32,
    /// The trailing f32: what the rounding to `hi` lost.
    pub lo: f32,
}

// --- the two-float primitives -------------------------------------------

/// `a + b` as a sum and its exact rounding error (Knuth).
#[inline(always)]
const fn two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let bb = s - a;
    let e = (a - (s - bb)) + (b - bb);
    (s, e)
}

/// `a + b` as a sum and its error, for `|a| >= |b|` or `a == 0`.
#[inline(always)]
const fn quick_two_sum(a: f32, b: f32) -> (f32, f32) {
    let s = a + b;
    let e = b - (s - a);
    (s, e)
}

/// `a * b` as a product and its exact rounding error.
#[inline(always)]
fn two_prod(a: f32, b: f32) -> (f32, f32) {
    let p = a * b;
    let e = a.mul_add(b, -p);
    (p, e)
}

/// A normalized `(hi, lo)` as a `df32`; a non-finite `hi` carries no
/// trailing part (the two-sums leave NaN residuals behind infinities).
#[inline(always)]
const fn pack(hi: f32, lo: f32) -> df32 {
    if hi.is_finite() { df32 { hi, lo } } else { df32 { hi, lo: 0.0 } }
}

impl df32 {
    pub const ZERO: df32 = df32 { hi: 0.0, lo: 0.0 };
    pub const ONE: df32 = df32 { hi: 1.0, lo: 0.0 };
    pub const TWO: df32 = df32 { hi: 2.0, lo: 0.0 };
    pub const HALF: df32 = df32 { hi: 0.5, lo: 0.0 };
    pub const NAN: df32 = df32 { hi: f32::NAN, lo: 0.0 };
    pub const INFINITY: df32 = df32 { hi: f32::INFINITY, lo: 0.0 };
    pub const NEG_INFINITY: df32 = df32 { hi: f32::NEG_INFINITY, lo: 0.0 };
    pub const PI: df32 = df32::from_f64(std::f64::consts::PI);
    pub const FRAC_PI_2: df32 = df32::from_f64(std::f64::consts::FRAC_PI_2);
    pub const FRAC_PI_4: df32 = df32::from_f64(std::f64::consts::FRAC_PI_4);
    pub const E: df32 = df32::from_f64(std::f64::consts::E);
    pub const LN_2: df32 = df32::from_f64(std::f64::consts::LN_2);
    pub const LN_10: df32 = df32::from_f64(std::f64::consts::LN_10);
    /// The relative precision of one operation, 2^-46.
    pub const EPSILON: df32 = df32 { hi: 1.4210854715202004e-14, lo: 0.0 };

    /// The third f32 of `pi/2` beyond `FRAC_PI_2`'s two, for the argument
    /// reduction of `sin_cos`.
    const FRAC_PI_2_P3: f32 = (std::f64::consts::FRAC_PI_2
        - (df32::FRAC_PI_2.hi as f64 + df32::FRAC_PI_2.lo as f64)) as f32;
    /// The third f32 of `ln 2`, for the argument reduction of `exp`.
    const LN_2_P3: f32 = (std::f64::consts::LN_2
        - (df32::LN_2.hi as f64 + df32::LN_2.lo as f64)) as f32;

    /// A value from its two parts, normalized.
    pub const fn new(hi: f32, lo: f32) -> df32 {
        let (s, e) = two_sum(hi, lo);
        pack(s, e)
    }

    /// An f32 as a `df32`, exactly.
    pub const fn from_f32(x: f32) -> df32 {
        df32 { hi: x, lo: 0.0 }
    }

    /// An f64 split into its two f32 halves: exact when the f64 has at
    /// most 48 significant bits, otherwise rounded there.
    pub const fn from_f64(x: f64) -> df32 {
        let hi = x as f32;
        if !hi.is_finite() { return df32 { hi, lo: 0.0 }; }
        let lo = (x - hi as f64) as f32;
        df32 { hi, lo }
    }

    /// The value as an f64, exactly.
    #[inline(always)]
    pub fn to_f64(self) -> f64 {
        self.hi as f64 + self.lo as f64
    }

    /// The value rounded to f32.
    #[inline(always)]
    pub fn to_f32(self) -> f32 {
        self.hi + self.lo
    }

    /// `self * self`.
    #[inline]
    pub fn sqr(self) -> df32 {
        let (p, e) = two_prod(self.hi, self.hi);
        if !p.is_finite() { return df32::from_f32(p); }
        let e = e + 2.0 * self.hi * self.lo;
        let (hi, lo) = quick_two_sum(p, e);
        pack(hi, lo)
    }

    /// `self * b` for an f32 `b`.
    #[inline]
    pub fn mul_f32(self, b: f32) -> df32 {
        let (p, e) = two_prod(self.hi, b);
        if !p.is_finite() { return df32::from_f32(p); }
        let e = e + self.lo * b;
        let (hi, lo) = quick_two_sum(p, e);
        pack(hi, lo)
    }

    /// `self - k * c` for an integer-valued f32 `k` and a constant `c`
    /// given as a `df32` and a third f32 part: the argument reduction of
    /// `exp` and `sin_cos`. Both products are exact (two_prod), and the
    /// leading subtraction cancels exactly, so the remainder carries the
    /// constant to about 72 bits times `k`.
    #[inline]
    fn reduce(self, k: f32, c: df32, c3: f32) -> df32 {
        let (p1, e1) = two_prod(c.hi, k);
        let (p2, e2) = two_prod(c.lo, k);
        ((((self - df32::from_f32(p1)) - df32::from_f32(e1)) - df32::from_f32(p2))
            - df32::from_f32(e2)) - df32::from_f32(c3 * k)
    }

    /// `self * 2^k`, exact while the result stays normal.
    pub fn scale_pow2(self, k: i32) -> df32 {
        // Two steps keep every factor a normal f32 for |k| up to 252.
        let k1 = k / 2;
        let k2 = k - k1;
        let f1 = 2.0f32.powi(k1);
        let f2 = 2.0f32.powi(k2);
        pack(self.hi * f1 * f2, self.lo * f1 * f2)
    }

    /// `self` as `m * 2^e` with `m` in `[0.5, 1)`, `(m, e)`; zero gives
    /// `(0, 0)`.
    fn frexp(self) -> (df32, i32) {
        if self.hi == 0.0 { return (self, 0); }
        let mut x = self;
        let mut shift = 0;
        if x.hi.abs() < f32::MIN_POSITIVE {
            x = x.scale_pow2(48);
            shift = -48;
        }
        let e = ((x.hi.abs().to_bits() >> 23) & 0xff) as i32 - 126;
        (x.scale_pow2(-e), e + shift)
    }

    /// Whether the value is an integer.
    fn is_integral(self) -> bool {
        self.hi.is_finite() && self.hi.fract() == 0.0 && self.lo.fract() == 0.0
    }

    /// `sin` of `|r| <= pi/4` by its Taylor series.
    fn sin_taylor(r: df32) -> df32 {
        // (-1)^n / (2n+1)!, n = 0..=8: the term after is below 1e-19.
        const C: [df32; 9] = [
            df32::from_f64(1.0),
            df32::from_f64(-1.0 / 6.0),
            df32::from_f64(1.0 / 120.0),
            df32::from_f64(-1.0 / 5040.0),
            df32::from_f64(1.0 / 362880.0),
            df32::from_f64(-1.0 / 39916800.0),
            df32::from_f64(1.0 / 6227020800.0),
            df32::from_f64(-1.0 / 1307674368000.0),
            df32::from_f64(1.0 / 355687428096000.0),
        ];
        let x2 = r.sqr();
        let mut s = C[8];
        for c in C[..8].iter().rev() {
            s = s * x2 + *c;
        }
        s * r
    }

    /// `cos` of `|r| <= pi/4` by its Taylor series.
    fn cos_taylor(r: df32) -> df32 {
        // (-1)^n / (2n)!, n = 0..=8: the term after is below 1e-17.
        const C: [df32; 9] = [
            df32::from_f64(1.0),
            df32::from_f64(-1.0 / 2.0),
            df32::from_f64(1.0 / 24.0),
            df32::from_f64(-1.0 / 720.0),
            df32::from_f64(1.0 / 40320.0),
            df32::from_f64(-1.0 / 3628800.0),
            df32::from_f64(1.0 / 479001600.0),
            df32::from_f64(-1.0 / 87178291200.0),
            df32::from_f64(1.0 / 20922789888000.0),
        ];
        let x2 = r.sqr();
        let mut s = C[8];
        for c in C[..8].iter().rev() {
            s = s * x2 + *c;
        }
        s
    }

    /// `exp(r)` for `|r| <= 0.18` by its Taylor series to degree 11.
    fn exp_taylor(r: df32) -> df32 {
        const C: [df32; 12] = [
            df32::from_f64(1.0),
            df32::from_f64(1.0),
            df32::from_f64(1.0 / 2.0),
            df32::from_f64(1.0 / 6.0),
            df32::from_f64(1.0 / 24.0),
            df32::from_f64(1.0 / 120.0),
            df32::from_f64(1.0 / 720.0),
            df32::from_f64(1.0 / 5040.0),
            df32::from_f64(1.0 / 40320.0),
            df32::from_f64(1.0 / 362880.0),
            df32::from_f64(1.0 / 3628800.0),
            df32::from_f64(1.0 / 39916800.0),
        ];
        let mut p = C[11];
        for c in C[..11].iter().rev() {
            p = p * r + *c;
        }
        p
    }
}

// --- the methods of a primitive float --------------------------------------
//
// Generated constraint code and user code call `.sin()`, `.powi(n)` and the
// rest on a scalar without a trait in scope, as they would on an f32; so a
// df32 has them as inherent methods, each the `num::Float` one.

macro_rules! forward_unary {
    ($($name:ident),* $(,)?) => { $(
        #[inline]
        pub fn $name(self) -> df32 { <df32 as num::Float>::$name(self) }
    )* };
}

impl df32 {
    forward_unary!(floor, ceil, round, trunc, fract, abs, signum, recip, sqrt, cbrt,
        exp, exp2, exp_m1, ln, ln_1p, log2, log10, sin, cos, tan, asin, acos, atan,
        sinh, cosh, tanh, asinh, acosh, atanh);
    #[inline] pub fn is_nan(self) -> bool { self.hi.is_nan() }
    #[inline] pub fn is_finite(self) -> bool { self.hi.is_finite() }
    #[inline] pub fn is_infinite(self) -> bool { self.hi.is_infinite() }
    #[inline] pub fn powi(self, n: i32) -> df32 { <df32 as num::Float>::powi(self, n) }
    #[inline] pub fn powf(self, n: df32) -> df32 { <df32 as num::Float>::powf(self, n) }
    #[inline] pub fn log(self, base: df32) -> df32 { <df32 as num::Float>::log(self, base) }
    #[inline] pub fn atan2(self, x: df32) -> df32 { <df32 as num::Float>::atan2(self, x) }
    #[inline] pub fn hypot(self, o: df32) -> df32 { <df32 as num::Float>::hypot(self, o) }
    #[inline] pub fn max(self, o: df32) -> df32 { <df32 as num::Float>::max(self, o) }
    #[inline] pub fn min(self, o: df32) -> df32 { <df32 as num::Float>::min(self, o) }
    #[inline] pub fn mul_add(self, a: df32, b: df32) -> df32 { <df32 as num::Float>::mul_add(self, a, b) }
    #[inline] pub fn sin_cos(self) -> (df32, df32) { <df32 as num::Float>::sin_cos(self) }
    #[inline] pub fn heaviside(self) -> df32 { <df32 as crate::utils::Float>::heaviside(self) }
    #[inline] pub fn clamp(self, low: df32, high: df32) -> df32 { <df32 as crate::utils::Float>::clamp(self, low, high) }
    #[inline] pub fn select_index(self) -> i64 { <df32 as crate::utils::SelectIndex>::select_index(self) }
}

// --- conversions ----------------------------------------------------------

impl From<f32> for df32 {
    fn from(x: f32) -> df32 { df32::from_f32(x) }
}

impl From<f64> for df32 {
    fn from(x: f64) -> df32 { df32::from_f64(x) }
}

impl From<df32> for f64 {
    fn from(x: df32) -> f64 { x.to_f64() }
}

impl fmt::Display for df32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(&self.to_f64(), f)
    }
}

impl fmt::Debug for df32 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "df32({})", self.to_f64())
    }
}

// --- arithmetic -------------------------------------------------------------

impl Add for df32 {
    type Output = df32;
    #[inline]
    fn add(self, b: df32) -> df32 {
        let (s1, s2) = two_sum(self.hi, b.hi);
        if !s1.is_finite() { return df32::from_f32(s1); }
        let (t1, t2) = two_sum(self.lo, b.lo);
        let s2 = s2 + t1;
        let (s1, s2) = quick_two_sum(s1, s2);
        let s2 = s2 + t2;
        let (hi, lo) = quick_two_sum(s1, s2);
        pack(hi, lo)
    }
}

impl Neg for df32 {
    type Output = df32;
    #[inline]
    fn neg(self) -> df32 { df32 { hi: -self.hi, lo: -self.lo } }
}

impl Sub for df32 {
    type Output = df32;
    #[inline]
    fn sub(self, b: df32) -> df32 { self + (-b) }
}

impl Mul for df32 {
    type Output = df32;
    #[inline]
    fn mul(self, b: df32) -> df32 {
        let (p, e) = two_prod(self.hi, b.hi);
        if !p.is_finite() { return df32::from_f32(p); }
        let e = e + (self.hi * b.lo + self.lo * b.hi);
        let (hi, lo) = quick_two_sum(p, e);
        pack(hi, lo)
    }
}

impl Div for df32 {
    type Output = df32;
    fn div(self, b: df32) -> df32 {
        let q1 = self.hi / b.hi;
        if !self.hi.is_finite() || !b.hi.is_finite() || b.hi == 0.0 || !q1.is_finite() {
            // f32's answer for infinities, NaN and division by zero.
            return df32::from_f32(q1);
        }
        let r = self - b.mul_f32(q1);
        let q2 = r.hi / b.hi;
        let r = r - b.mul_f32(q2);
        let q3 = r.hi / b.hi;
        let (q1, q2) = quick_two_sum(q1, q2);
        df32 { hi: q1, lo: q2 } + df32::from_f32(q3)
    }
}

impl Rem for df32 {
    type Output = df32;
    fn rem(self, b: df32) -> df32 {
        self - (self / b).trunc() * b
    }
}

impl AddAssign for df32 { fn add_assign(&mut self, b: df32) { *self = *self + b; } }
impl SubAssign for df32 { fn sub_assign(&mut self, b: df32) { *self = *self - b; } }
impl MulAssign for df32 { fn mul_assign(&mut self, b: df32) { *self = *self * b; } }
impl DivAssign for df32 { fn div_assign(&mut self, b: df32) { *self = *self / b; } }
impl RemAssign for df32 { fn rem_assign(&mut self, b: df32) { *self = *self % b; } }

impl PartialOrd for df32 {
    fn partial_cmp(&self, o: &df32) -> Option<Ordering> {
        if self.hi.is_nan() || o.hi.is_nan() { return None; }
        match self.hi.partial_cmp(&o.hi)? {
            Ordering::Equal => self.lo.partial_cmp(&o.lo),
            ord => Some(ord),
        }
    }
}

// --- num traits --------------------------------------------------------------

impl num::Zero for df32 {
    fn zero() -> df32 { df32::ZERO }
    fn is_zero(&self) -> bool { self.hi == 0.0 }
}

impl num::One for df32 {
    fn one() -> df32 { df32::ONE }
}

impl num::Num for df32 {
    type FromStrRadixErr = num::traits::ParseFloatError;
    fn from_str_radix(s: &str, radix: u32) -> Result<df32, Self::FromStrRadixErr> {
        <f64 as num::Num>::from_str_radix(s, radix).map(df32::from_f64)
    }
}

impl num::ToPrimitive for df32 {
    fn to_i64(&self) -> Option<i64> { num::ToPrimitive::to_i64(&df32::to_f64(*self)) }
    fn to_u64(&self) -> Option<u64> { num::ToPrimitive::to_u64(&df32::to_f64(*self)) }
    fn to_f64(&self) -> Option<f64> { Some(df32::to_f64(*self)) }
    fn to_f32(&self) -> Option<f32> { Some(df32::to_f32(*self)) }
}

impl num::NumCast for df32 {
    fn from<T: num::ToPrimitive>(n: T) -> Option<df32> {
        n.to_f64().map(df32::from_f64)
    }
}

impl num::Float for df32 {
    fn nan() -> df32 { df32::NAN }
    fn infinity() -> df32 { df32::INFINITY }
    fn neg_infinity() -> df32 { df32::NEG_INFINITY }
    fn neg_zero() -> df32 { df32 { hi: -0.0, lo: 0.0 } }
    fn min_value() -> df32 { df32::from_f32(f32::MIN) }
    fn min_positive_value() -> df32 { df32::from_f32(f32::MIN_POSITIVE) }
    fn epsilon() -> df32 { df32::EPSILON }
    fn max_value() -> df32 { df32::from_f32(f32::MAX) }

    fn is_nan(self) -> bool { self.hi.is_nan() }
    fn is_infinite(self) -> bool { self.hi.is_infinite() }
    fn is_finite(self) -> bool { self.hi.is_finite() }
    fn is_normal(self) -> bool { self.hi.is_normal() }
    fn classify(self) -> FpCategory { self.hi.classify() }

    fn floor(self) -> df32 {
        let hi = self.hi.floor();
        if hi == self.hi {
            let (h, l) = quick_two_sum(hi, self.lo.floor());
            pack(h, l)
        } else {
            df32::from_f32(hi)
        }
    }
    fn ceil(self) -> df32 {
        let hi = self.hi.ceil();
        if hi == self.hi {
            let (h, l) = quick_two_sum(hi, self.lo.ceil());
            pack(h, l)
        } else {
            df32::from_f32(hi)
        }
    }
    fn round(self) -> df32 {
        // Half away from zero, as f32's. `d` is the exact distance of
        // `hi` from the integer f32 picked.
        let hi = self.hi.round();
        let d = self.hi - hi;
        if d == 0.0 {
            // An integral hi: lo, at most half an ulp of it, decides.
            let n = if hi >= 0.0 {
                if self.lo >= 0.5 { hi + 1.0 } else if self.lo < -0.5 { hi - 1.0 } else { hi }
            } else if self.lo <= -0.5 { hi - 1.0 } else if self.lo > 0.5 { hi + 1.0 } else { hi };
            df32::from_f32(n)
        } else if d.abs() == 0.5 {
            // f32 saw a tie and went away from zero; lo tells the true side.
            if d < 0.0 && self.lo < 0.0 { df32::from_f32(hi - 1.0) }
            else if d > 0.0 && self.lo > 0.0 { df32::from_f32(hi + 1.0) }
            else { df32::from_f32(hi) }
        } else {
            df32::from_f32(hi)
        }
    }
    fn trunc(self) -> df32 {
        if self.hi >= 0.0 { self.floor() } else { self.ceil() }
    }
    fn fract(self) -> df32 { self - self.trunc() }

    fn abs(self) -> df32 { if self.hi < 0.0 { -self } else { self } }
    fn signum(self) -> df32 { df32::from_f32(self.hi.signum()) }
    fn is_sign_positive(self) -> bool { self.hi.is_sign_positive() }
    fn is_sign_negative(self) -> bool { self.hi.is_sign_negative() }

    fn mul_add(self, a: df32, b: df32) -> df32 { self * a + b }
    fn recip(self) -> df32 { df32::ONE / self }

    fn powi(self, n: i32) -> df32 {
        if n == 0 { return df32::ONE; }
        let mut r = df32::ONE;
        let mut base = self;
        let mut e = n.unsigned_abs();
        while e > 0 {
            if e & 1 == 1 { r = r * base; }
            e >>= 1;
            if e > 0 { base = base.sqr(); }
        }
        if n < 0 { df32::ONE / r } else { r }
    }
    fn powf(self, n: df32) -> df32 {
        if n.is_integral() && n.hi.abs() < 2147483648.0 {
            return self.powi(n.to_f64() as i32);
        }
        if self.hi == 0.0 {
            return if n.hi > 0.0 { df32::ZERO } else { df32::INFINITY };
        }
        if self.hi < 0.0 { return df32::NAN; }
        (n * self.ln()).exp()
    }
    fn sqrt(self) -> df32 {
        if self.hi <= 0.0 || !self.hi.is_finite() {
            // Zero of either sign, a negative (NaN), an infinity, NaN.
            return df32::from_f32(self.hi.sqrt());
        }
        // Karp's trick: one correction of an f32 square root.
        let x = 1.0 / self.hi.sqrt();
        let ax = self.hi * x;
        let diff = (self - df32::from_f32(ax).sqr()).hi;
        let (hi, lo) = quick_two_sum(ax, diff * (x * 0.5));
        pack(hi, lo)
    }

    fn exp(self) -> df32 {
        if self.hi.is_nan() { return df32::NAN; }
        if self.hi > 88.8 { return df32::INFINITY; }
        if self.hi < -104.0 { return df32::ZERO; }
        // x = k ln2 + r, |r| <= 0.35; the series on r/2, squared once.
        let k = (self.hi / std::f32::consts::LN_2).round();
        let r = self.reduce(k, df32::LN_2, df32::LN_2_P3).scale_pow2(-1);
        df32::exp_taylor(r).sqr().scale_pow2(k as i32)
    }
    fn exp2(self) -> df32 { (self * df32::LN_2).exp() }
    fn ln(self) -> df32 {
        if self.hi.is_nan() || self.hi < 0.0 { return df32::NAN; }
        if self.hi == 0.0 { return df32::NEG_INFINITY; }
        if self.hi.is_infinite() { return df32::INFINITY; }
        // ln(m 2^e) = ln m + e ln 2, with m in [0.5, 1) well inside the
        // range of exp; two Newton steps on exp from the f32 logarithm.
        let (m, e) = self.frexp();
        let mut y = df32::from_f32(m.hi.ln());
        y = y + m * (-y).exp() - df32::ONE;
        y = y + m * (-y).exp() - df32::ONE;
        y + df32::LN_2.mul_f32(e as f32)
    }
    fn log(self, base: df32) -> df32 { self.ln() / base.ln() }
    fn log2(self) -> df32 { self.ln() / df32::LN_2 }
    fn log10(self) -> df32 { self.ln() / df32::LN_10 }

    fn max(self, o: df32) -> df32 {
        if self.hi.is_nan() { o } else if o.hi.is_nan() { self } else if self >= o { self } else { o }
    }
    fn min(self, o: df32) -> df32 {
        if self.hi.is_nan() { o } else if o.hi.is_nan() { self } else if self <= o { self } else { o }
    }
    fn abs_sub(self, o: df32) -> df32 {
        if self <= o { df32::ZERO } else { self - o }
    }
    fn cbrt(self) -> df32 {
        if self.hi == 0.0 || !self.hi.is_finite() { return df32::from_f32(self.hi.cbrt()); }
        let mut y = df32::from_f32(self.hi.cbrt());
        for _ in 0..2 {
            y = (y * df32::TWO + self / y.sqr()) / df32::from_f32(3.0);
        }
        y
    }
    fn hypot(self, o: df32) -> df32 { (self.sqr() + o.sqr()).sqrt() }

    fn sin(self) -> df32 { self.sin_cos().0 }
    fn cos(self) -> df32 { self.sin_cos().1 }
    fn tan(self) -> df32 { let (s, c) = self.sin_cos(); s / c }
    fn asin(self) -> df32 {
        if self.hi.abs() > 1.0 || self.hi.is_nan() { return df32::NAN; }
        if self.hi.abs() == 1.0 && self.lo == 0.0 {
            return if self.hi > 0.0 { df32::FRAC_PI_2 } else { -df32::FRAC_PI_2 };
        }
        // (1 - x)(1 + x) rather than 1 - x^2: exact factors near x = +-1.
        self.atan2(((df32::ONE - self) * (df32::ONE + self)).sqrt())
    }
    fn acos(self) -> df32 {
        if self.hi.abs() > 1.0 || self.hi.is_nan() { return df32::NAN; }
        if self.hi.abs() == 1.0 && self.lo == 0.0 {
            return if self.hi > 0.0 { df32::ZERO } else { df32::PI };
        }
        ((df32::ONE - self) * (df32::ONE + self)).sqrt().atan2(self)
    }
    fn atan(self) -> df32 { self.atan2(df32::ONE) }
    fn atan2(self, x: df32) -> df32 {
        let y = self;
        if !(x.hi.is_finite() && y.hi.is_finite()) {
            return df32::from_f32(y.hi.atan2(x.hi));
        }
        if x.hi == 0.0 {
            return if y.hi == 0.0 { df32::from_f32(y.hi.atan2(x.hi)) }
                else if y.hi > 0.0 { df32::FRAC_PI_2 } else { -df32::FRAC_PI_2 };
        }
        if y.hi == 0.0 {
            return if x.hi > 0.0 { df32::from_f32(y.hi.atan2(x.hi)) }
                else if y.hi.is_sign_negative() { -df32::PI } else { df32::PI };
        }
        // Newton on sin and cos from the f32 angle, twice: the direction
        // (xx, yy) is unit, so the step reads straight off the residual.
        let r = x.hypot(y);
        let xx = x / r;
        let yy = y / r;
        let mut z = df32::from_f32(y.to_f32().atan2(x.to_f32()));
        for _ in 0..2 {
            let (sz, cz) = z.sin_cos();
            if xx.hi.abs() > yy.hi.abs() {
                z = z + (yy - sz) / cz;
            } else {
                z = z - (xx - cz) / sz;
            }
        }
        z
    }
    fn sin_cos(self) -> (df32, df32) {
        if !self.hi.is_finite() { return (df32::NAN, df32::NAN); }
        // x = k pi/2 + r, |r| <= pi/4, the constant in three f32 parts.
        let k = (self.hi * std::f32::consts::FRAC_2_PI).round();
        let r = self.reduce(k, df32::FRAC_PI_2, df32::FRAC_PI_2_P3);
        let s = df32::sin_taylor(r);
        let c = df32::cos_taylor(r);
        match ((k as i64) % 4 + 4) % 4 {
            0 => (s, c),
            1 => (c, -s),
            2 => (-s, -c),
            _ => (-c, s),
        }
    }
    fn exp_m1(self) -> df32 {
        if self.hi.abs() >= 0.5 || !self.hi.is_finite() {
            return self.exp() - df32::ONE;
        }
        // x + x^2/2! + ... to degree 15: the term after is below 1e-18.
        const C: [df32; 15] = [
            df32::from_f64(1.0),
            df32::from_f64(1.0 / 2.0),
            df32::from_f64(1.0 / 6.0),
            df32::from_f64(1.0 / 24.0),
            df32::from_f64(1.0 / 120.0),
            df32::from_f64(1.0 / 720.0),
            df32::from_f64(1.0 / 5040.0),
            df32::from_f64(1.0 / 40320.0),
            df32::from_f64(1.0 / 362880.0),
            df32::from_f64(1.0 / 3628800.0),
            df32::from_f64(1.0 / 39916800.0),
            df32::from_f64(1.0 / 479001600.0),
            df32::from_f64(1.0 / 6227020800.0),
            df32::from_f64(1.0 / 87178291200.0),
            df32::from_f64(1.0 / 1307674368000.0),
        ];
        let mut p = C[14];
        for c in C[..14].iter().rev() {
            p = p * self + *c;
        }
        p * self
    }
    fn ln_1p(self) -> df32 {
        if self.hi.abs() >= 0.1 || !self.hi.is_finite() {
            return (df32::ONE + self).ln();
        }
        // 2 atanh(u), u = x / (2 + x), |u| < 0.053: odd powers to u^13.
        let u = self / (df32::TWO + self);
        let u2 = u.sqr();
        let mut s = df32::from_f64(1.0 / 13.0);
        for n in (0..6).rev() {
            s = s * u2 + df32::from_f64(1.0 / (2 * n + 1) as f64);
        }
        s * u * df32::TWO
    }
    fn sinh(self) -> df32 {
        if !self.hi.is_finite() { return self; }
        if self.hi.abs() < 0.05 {
            // x + x^3/3! + ... + x^9/9!: the term after is below 1e-22.
            let x2 = self.sqr();
            let mut s = df32::from_f64(1.0 / 362880.0);
            s = s * x2 + df32::from_f64(1.0 / 5040.0);
            s = s * x2 + df32::from_f64(1.0 / 120.0);
            s = s * x2 + df32::from_f64(1.0 / 6.0);
            s = s * x2 + df32::ONE;
            return s * self;
        }
        let e = self.exp();
        (e - df32::ONE / e) * df32::HALF
    }
    fn cosh(self) -> df32 {
        if !self.hi.is_finite() { return self.abs(); }
        let e = self.exp();
        (e + df32::ONE / e) * df32::HALF
    }
    fn tanh(self) -> df32 {
        if self.hi.is_nan() { return self; }
        if self.hi.abs() > 20.0 { return df32::from_f32(self.hi.signum()); }
        let e = (self * df32::TWO).exp_m1();
        e / (e + df32::TWO)
    }
    fn asinh(self) -> df32 {
        if !self.hi.is_finite() { return self; }
        let a = self.abs();
        let r = (a + (a.sqr() + df32::ONE).sqrt()).ln();
        if self.hi < 0.0 { -r } else { r }
    }
    fn acosh(self) -> df32 {
        if self.hi < 1.0 || self.hi.is_nan() { return df32::NAN; }
        // (x - 1)(x + 1) rather than x^2 - 1: exact factors near x = 1.
        (self + ((self - df32::ONE) * (self + df32::ONE)).sqrt()).ln()
    }
    fn atanh(self) -> df32 {
        if self.hi.abs() > 1.0 || self.hi.is_nan() { return df32::NAN; }
        if self.hi.abs() == 1.0 && self.lo == 0.0 {
            return if self.hi > 0.0 { df32::INFINITY } else { df32::NEG_INFINITY };
        }
        // (ln(1 + x) - ln(1 - x)) / 2: both arguments exact near x = +-1,
        // where 2x / (1 - x) would lose its digits.
        (self.ln_1p() - (-self).ln_1p()) * df32::HALF
    }

    fn integer_decode(self) -> (u64, i16, i8) { self.to_f64().integer_decode() }
}

// --- arael traits -----------------------------------------------------------

impl crate::utils::Float for df32 {
    fn pi() -> df32 { df32::PI }
    fn two() -> df32 { df32::TWO }
    fn four() -> df32 { df32::from_f32(4.0) }
    fn half() -> df32 { df32::HALF }
}

impl crate::utils::SelectIndex for df32 {
    fn select_index(self) -> i64 {
        let v = self.to_f64();
        if !v.is_finite() || v.fract() != 0.0 {
            panic!("select index {} is not an integer", v);
        }
        v as i64
    }
}

impl crate::model::ParamType for df32 {
    const SIZE: usize = 1;
    const SUFFIXES: &'static [&'static str] = &[""];
    fn write_to<F: crate::utils::Float>(&self, dst: &mut [F]) { dst[0] = F::from(self.to_f64()).unwrap(); }
    fn read_from<F: crate::utils::Float>(src: &[F]) -> Self { df32::from_f64(src[0].to_f64().unwrap()) }
}

impl crate::utils::WideScalar for df32 {
    #[inline(always)]
    fn from_f64(v: f64) -> df32 { df32::from_f64(v) }
}

impl crate::utils::Wide<f32> for df32 {
    #[inline(always)]
    fn widen(v: f32) -> df32 { df32::from_f32(v) }
    #[inline(always)]
    fn narrow(self) -> f32 { self.to_f32() }
}

impl serde::Serialize for df32 {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_f64(self.to_f64())
    }
}

impl<'de> serde::Deserialize<'de> for df32 {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<df32, D::Error> {
        f64::deserialize(d).map(df32::from_f64)
    }
}

// --- tests --------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// A small xorshift generator: the tests need spread, not quality.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x << 13; x ^= x >> 7; x ^= x << 17;
            self.0 = x;
            x
        }
        /// Uniform in [0, 1).
        fn unit(&mut self) -> f64 { (self.next() >> 11) as f64 / (1u64 << 53) as f64 }
        /// A signed value with a random magnitude in [10^-lo, 10^hi).
        fn wide(&mut self, lo: f64, hi: f64) -> f64 {
            let mag = 10f64.powf(-lo + (lo + hi) * self.unit());
            if self.next() & 1 == 0 { mag } else { -mag }
        }
        /// A df32 whose value is exactly representable.
        fn df(&mut self, lo: f64, hi: f64) -> df32 { df32::from_f64(self.wide(lo, hi)) }
    }

    fn rel(a: f64, b: f64) -> f64 { (a - b).abs() / b.abs().max(f64::MIN_POSITIVE) }

    /// The unit in the last place of `x`: the spacing of f32 values at
    /// `|x|`, a subnormal spacing below 2^-103.
    fn ulp(x: f32) -> f32 {
        let a = x.abs();
        if a < f32::MIN_POSITIVE { return f32::from_bits(1); }
        let p = ((a.to_bits() >> 23) & 0xff) as i32 - 127 - 23;
        if p >= -126 { f32::from_bits(((p + 127) as u32) << 23) } else { f32::from_bits(1u32 << (p + 149)) }
    }

    /// `|lo| <= ulp(hi) / 2`, the normalized form.
    fn is_normalized(x: df32) -> bool {
        !x.hi.is_finite() || x.lo.abs() <= ulp(x.hi) * 0.5
    }

    const TOL: f64 = 2.9e-14; // 2^-45

    #[test]
    fn the_split_keeps_48_bits_and_the_join_is_exact() {
        let mut rng = Rng(0x9e3779b97f4a7c15);
        for _ in 0..10000 {
            let x = rng.wide(30.0, 30.0);
            let d = df32::from_f64(x);
            assert!(is_normalized(d), "{d:?}");
            assert!(rel(d.to_f64(), x) <= 2f64.powi(-48), "{x} -> {d:?}");
            // The join is exact: splitting it again changes nothing.
            assert_eq!(df32::from_f64(d.to_f64()), d);
        }
        assert_eq!(df32::from_f32(1.5).to_f64(), 1.5);
        assert_eq!(df32::new(1.0, 1e-10).to_f64(), 1.0 + 1e-10f32 as f64);
        assert!(df32::from_f64(1e300).hi.is_infinite() && df32::from_f64(1e300).lo == 0.0);
    }

    #[test]
    fn arithmetic_against_f64() {
        let mut rng = Rng(12345);
        let mut worst = [0.0f64; 5];
        for _ in 0..100000 {
            let a = rng.df(8.0, 8.0);
            let b = rng.df(8.0, 8.0);
            let (fa, fb) = (a.to_f64(), b.to_f64());
            let cases = [
                (a + b, fa + fb), (a - b, fa - fb), (a * b, fa * fb), (a / b, fa / fb),
                (a.abs().sqrt(), fa.abs().sqrt()),
            ];
            for (i, (got, want)) in cases.iter().enumerate() {
                assert!(is_normalized(*got), "{got:?}");
                // The f64 reference itself rounds at 2^-53 and the exact
                // result of two 48-bit operands may not fit 48 bits, so the
                // tolerance is one df32 rounding.
                let e = rel(got.to_f64(), *want);
                worst[i] = worst[i].max(e);
                assert!(e <= TOL, "case {i}: {a:?} {b:?} -> {got:?}, want {want}, rel {e:e}");
            }
        }
        assert!(worst.iter().all(|w| *w > 0.0), "no rounding at all is suspicious: {worst:?}");
    }

    #[test]
    fn cancellation_is_exact() {
        let a = df32::from_f64(1.0 + 1e-10);
        let b = df32::from_f32(1.0);
        assert_eq!((a - b).to_f64(), a.to_f64() - 1.0);
        let c = df32::from_f64(12345.678901234);
        let d = df32::from_f64(12345.678901233);
        assert_eq!((c - d).to_f64(), c.to_f64() - d.to_f64());
    }

    #[test]
    fn special_values_and_order() {
        let nan = df32::NAN;
        let one = df32::ONE;
        assert!((nan + one).is_nan() && (one / df32::ZERO).is_infinite());
        assert!((df32::INFINITY - df32::INFINITY).is_nan());
        assert_eq!((df32::INFINITY * df32::from_f32(-2.0)).hi, f32::NEG_INFINITY);
        assert!((df32::INFINITY * df32::TWO).lo == 0.0);
        assert!(nan.partial_cmp(&one).is_none() && !(nan < one) && !(nan == one));
        let a = df32::from_f64(1.0 + 1e-10);
        assert!(a > one && one < a && a >= a && one != a);
        assert_eq!(one.max(nan), one);
        assert_eq!(nan.min(one), one);
        assert_eq!(df32::from_f32(-3.0).signum().hi, -1.0);
        assert_eq!(df32::from_f32(-3.0).abs().hi, 3.0);
    }

    #[test]
    fn rounding_functions() {
        let v = |x: f64| df32::from_f64(x);
        assert_eq!(v(2.5).round().to_f64(), 3.0);
        assert_eq!(v(-2.5).round().to_f64(), -3.0);
        assert_eq!(v(2.5 - 1e-9).round().to_f64(), 2.0);
        assert_eq!(v(-2.5 + 1e-9).round().to_f64(), -2.0);
        assert_eq!(v(2.5 + 1e-9).round().to_f64(), 3.0);
        assert_eq!(v(1e7 + 0.5).round().to_f64(), 1e7 + 1.0);
        assert_eq!(v(-7.25).floor().to_f64(), -8.0);
        assert_eq!(v(-7.25).ceil().to_f64(), -7.0);
        assert_eq!(v(-7.25).trunc().to_f64(), -7.0);
        assert_eq!(v(-7.25).fract().to_f64(), -0.25);
        assert_eq!(v(1e7 + 0.25).floor().to_f64(), 1e7);
        assert_eq!(v(1e7 + 0.25).ceil().to_f64(), 1e7 + 1.0);
        assert_eq!(v(1e7 + 0.25).fract().to_f64(), 0.25);
        assert_eq!((v(7.5) % v(2.0)).to_f64(), 1.5);
        assert_eq!(v(2.0).powi(10).to_f64(), 1024.0);
        assert!(rel(v(2.0).powi(-3).to_f64(), 0.125) == 0.0);
    }

    fn check(name: &str, got: df32, want: f64, tol: f64) {
        assert!(is_normalized(got), "{name}: {got:?} not normalized");
        let e = (got.to_f64() - want).abs() / want.abs().max(1.0);
        assert!(e <= tol, "{name}: got {got:?}, want {want}, err {e:e}");
    }

    #[test]
    fn exp_and_ln_against_f64() {
        let mut rng = Rng(777);
        for _ in 0..5000 {
            let x = rng.wide(3.0, 1.9); // |x| up to about 80
            let d = df32::from_f64(x);
            let xe = d.to_f64();
            check("exp", d.exp(), xe.exp(), 1e-13);
            check("exp_m1", d.exp_m1(), xe.exp_m1(), 1e-13);
            let p = df32::from_f64(xe.abs() * 1e-10 + 1e-30);
            check("ln", p.ln(), p.to_f64().ln(), 1e-13);
        }
        for _ in 0..5000 {
            let x = rng.wide(30.0, 30.0).abs();
            let d = df32::from_f64(x);
            check("ln wide", d.ln(), d.to_f64().ln(), 1e-13);
            check("log2", d.log2(), d.to_f64().log2(), 1e-13);
            check("log10", d.log10(), d.to_f64().log10(), 1e-13);
            check("sqrt", d.sqrt(), d.to_f64().sqrt(), TOL);
            check("cbrt", d.cbrt(), d.to_f64().cbrt(), 1e-13);
        }
        for _ in 0..5000 {
            let x = rng.wide(12.0, -1.0); // small arguments
            let d = df32::from_f64(x);
            check("ln_1p", d.ln_1p(), d.to_f64().ln_1p(), 1e-13);
            check("exp_m1 small", d.exp_m1(), d.to_f64().exp_m1(), 1e-13);
            check("sinh small", d.sinh(), d.to_f64().sinh(), 1e-13);
            check("tanh small", d.tanh(), d.to_f64().tanh(), 1e-13);
        }
        assert!(df32::from_f32(100.0).exp().is_infinite());
        assert_eq!(df32::from_f32(-120.0).exp(), df32::ZERO);
        assert_eq!(df32::ZERO.ln(), df32::NEG_INFINITY);
        assert!(df32::from_f32(-1.0).ln().is_nan());
        check("exp2", df32::from_f64(10.5).exp2(), 10.5f64.exp2(), 1e-13);
        check("powf", df32::from_f64(2.5).powf(df32::from_f64(3.3)), 2.5f64.powf(3.3), 1e-13);
        check("powf int", df32::from_f64(-2.5).powf(df32::from_f64(3.0)), (-2.5f64).powi(3), TOL);
    }

    #[test]
    fn trigonometry_against_f64() {
        let mut rng = Rng(4242);
        for _ in 0..5000 {
            let x = rng.wide(6.0, 1.5); // |x| up to about 30
            let d = df32::from_f64(x);
            let xe = d.to_f64();
            let (s, c) = d.sin_cos();
            check("sin", s, xe.sin(), 2e-13);
            check("cos", c, xe.cos(), 2e-13);
            check("sin^2 + cos^2", s.sqr() + c.sqr(), 1.0, 1e-13);
            check("tan", d.tan(), xe.tan(), 1e-12 * (1.0 + xe.tan().abs()));
            check("atan", d.atan(), xe.atan(), 1e-13);
            let y = rng.wide(6.0, 1.5);
            let dy = df32::from_f64(y);
            check("atan2", dy.atan2(d), dy.to_f64().atan2(xe), 1e-13);
            check("hypot", d.hypot(dy), xe.hypot(dy.to_f64()), TOL);
        }
        for _ in 0..5000 {
            let x = rng.unit() * 2.0 - 1.0;
            let d = df32::from_f64(x);
            check("asin", d.asin(), d.to_f64().asin(), 1e-13);
            check("acos", d.acos(), d.to_f64().acos(), 1e-13);
            check("atanh", d.atanh(), d.to_f64().atanh(), 1e-13 * (1.0 + d.to_f64().atanh().abs()));
            let big = df32::from_f64(x * 40.0);
            check("sinh", big.sinh(), big.to_f64().sinh(), 1e-13);
            check("cosh", big.cosh(), big.to_f64().cosh(), 1e-13);
            check("tanh", big.tanh(), big.to_f64().tanh(), 1e-13);
            check("asinh", big.asinh(), big.to_f64().asinh(), 1e-13);
            check("acosh", (big.abs() + df32::ONE).acosh(), (big.to_f64().abs() + 1.0).acosh(), 1e-13);
        }
        // The exact points.
        assert_eq!(df32::ZERO.sin_cos(), (df32::ZERO, df32::ONE));
        check("sin pi/2", df32::FRAC_PI_2.sin(), 1.0, 1e-14);
        check("cos pi", df32::PI.cos(), -1.0, 1e-14);
        check("sin pi", df32::PI.sin(), 0.0, 1e-14);
        check("asin 1", df32::ONE.asin(), std::f64::consts::FRAC_PI_2, 1e-14);
        check("acos -1", (-df32::ONE).acos(), std::f64::consts::PI, 1e-14);
        check("atan2(1, 0)", df32::ONE.atan2(df32::ZERO), std::f64::consts::FRAC_PI_2, 1e-14);
        check("atan2(0, -1)", df32::ZERO.atan2(-df32::ONE), std::f64::consts::PI, 1e-14);
        assert!(df32::INFINITY.sin().is_nan());
    }

    /// The use the type exists for: a rotation built from angles and read
    /// back, in df32 arithmetic through the generic matrix code.
    #[test]
    fn euler_angles_round_trip_through_a_rotation() {
        use crate::matrix::matrix3;
        use crate::vect::vect3;
        let mut rng = Rng(99);
        for _ in 0..200 {
            let ea = vect3::<f64>::new(rng.unit() * 2.0 - 1.0, (rng.unit() - 0.5) * 2.8, (rng.unit() * 2.0 - 1.0) * 3.0);
            let d = vect3::<df32>::new(df32::from_f64(ea.x), df32::from_f64(ea.y), df32::from_f64(ea.z));
            let r = matrix3::<df32>::rotation_from_euler_angles(d);
            let r64 = matrix3::<f64>::rotation_from_euler_angles(ea);
            for i in 0..3 {
                assert!((r[i].x.to_f64() - r64[i].x).abs() < 1e-13, "entry {i}.x");
                assert!((r[i].y.to_f64() - r64[i].y).abs() < 1e-13, "entry {i}.y");
                assert!((r[i].z.to_f64() - r64[i].z).abs() < 1e-13, "entry {i}.z");
            }
            let back = r.get_euler_angles();
            assert!((back.x.to_f64() - ea.x).abs() < 1e-12, "roll {} vs {}", back.x, ea.x);
            assert!((back.y.to_f64() - ea.y).abs() < 1e-12, "pitch {} vs {}", back.y, ea.y);
            assert!((back.z.to_f64() - ea.z).abs() < 1e-12, "yaw {} vs {}", back.z, ea.z);
            // Orthonormal to df32 precision.
            let e = r * r.transpose() - matrix3::<df32>::identity();
            for i in 0..3 {
                assert!(e[i].x.to_f64().abs() < 1e-13 && e[i].y.to_f64().abs() < 1e-13 && e[i].z.to_f64().abs() < 1e-13);
            }
        }
    }

    #[test]
    fn the_arael_traits() {
        use crate::utils::{Wide, WideScalar, SelectIndex};
        use crate::model::ParamType;
        assert_eq!(<df32 as Wide<f32>>::widen(0.1f32).to_f64(), 0.1f32 as f64);
        assert_eq!(<df32 as Wide<f32>>::narrow(df32::from_f64(0.1)), 0.1f32);
        assert_eq!(<df32 as WideScalar>::from_f64(0.1).to_f64(), df32::from_f64(0.1).to_f64());
        assert_eq!(df32::from_f32(3.0).select_index(), 3);
        let mut slot = [0.0f32];
        df32::from_f64(0.1).write_to(&mut slot);
        assert_eq!(slot[0], 0.1f32);
        let back = <df32 as ParamType>::read_from(&[0.1f64]);
        assert_eq!(back, df32::from_f64(0.1));
        assert_eq!(<df32 as num::NumCast>::from(2.5f64).unwrap().to_f64(), 2.5);
        assert_eq!(<df32 as crate::utils::Float>::pi(), df32::PI);
        assert_eq!(format!("{}", df32::from_f32(1.5)), "1.5");
        assert_eq!(format!("{:?}", df32::from_f32(1.5)), "df32(1.5)");
        let json = serde_json::to_string(&df32::from_f64(0.1)).unwrap();
        let d: df32 = serde_json::from_str(&json).unwrap();
        assert_eq!(d, df32::from_f64(0.1));
    }
}
