//! Numeric traits and angle utilities.

/// Floating-point trait used throughout arael for generic f32/f64 code.
///
/// Extends `num::Float` with degree/radian conversions, angle normalization,
/// numeric constants, and a fast approximate atan.
pub trait Float : num::Float + std::fmt::Debug + num::NumCast + std::ops::AddAssign + std::ops::MulAssign + crate::model::ParamType + 'static {
    /// Convert degrees to radians.
    fn deg2rad(self) -> Self {
        self.to_radians()
    }

    /// Convert radians to degrees.
    fn rad2deg(self) -> Self {
        self.to_degrees()
    }

    /// Pi constant.
    fn pi() -> Self;
    /// Constant 2.
    fn two() -> Self;
    /// Constant 4.
    fn four() -> Self;
    /// Constant 0.5.
    fn half()-> Self;
    /// 2*pi.
    fn two_pi() -> Self { Self::two() * Self::pi() }
    /// pi/2.
    fn half_pi() -> Self { Self::half() * Self::pi() }

    /// Normalises radians to the range [-pi, pi]
    fn rad2rad(self) -> Self {
        if self < -Self::pi() || self > Self::pi() {
            self.rad2similar(Self::zero())
        } else {
            self
        }
    }

    /// Normalises radians to the range [similar-pi, similar+pi]
    fn rad2similar(self, similar: Self) -> Self {
        let delta = self - similar;
        // could use Self::div_euclid instead?
        self - Self::two_pi() * (delta / Self::two_pi() + Self::half()).floor()
    }

    /// Rollover-safe difference of two radian angles, result in [-pi, pi].
    fn rad_diff(self, other: Self) -> Self {
        (self - other).rad2rad()
    }

    /// Rollover-safe sum of two radian angles, result in [-pi, pi].
    fn rad_sum(self, other: Self) -> Self {
        (self + other).rad2rad()
    }

    /// Safe asin that does not diverge on numerical noise across applicable range
    fn safe_asin(self) -> Self {
        if self >= Self::one() {
            if self > Self::from(1.01).unwrap() {
                warn!("safe_asin: applied on divergent argument {:?}", self);
            }
            Self::half_pi()
        } else if self <= -Self::one() {
            if self < Self::from(-1.01).unwrap() {
                warn!("safe_asin: applied on divergent argument {:?}", self)
            }
            -Self::half_pi()
        } else {
            self.asin()
        }
    }

    /// Safe acos that does not diverge on numerical noise across applicable range
    fn safe_acos(self) -> Self {
        if self >= Self::one() {
            if self > Self::from(1.01).unwrap() {
                warn!("safe_acos: applied on divergent argument {:?}", self);
            }
            Self::zero()
        } else if self <= -Self::one() {
            if self < Self::from(-1.01).unwrap() {
                warn!("safe_acos: applied on divergent argument {:?}", self)
            }
            Self::pi()
        } else {
            self.acos()
        }
    }

    /// Fast polynomial approximation of atan. Max error < 1e-6 radians.
    fn fast_atan(self) -> Self {
      let half_pi = Self::from(std::f64::consts::FRAC_PI_2).unwrap();
      let sixth_pi = Self::from(5.235_987_755_982_988e-1).unwrap();
      let tan_sixth_pi = Self::from(5.773_502_691_896_257e-1).unwrap();
      let tan_twelfth_pi = Self::from(2.679_491_924_311_227e-1).unwrap();
      let c1 = Self::from(1.6867629106).unwrap();
      let c2 = Self::from(0.4378497304).unwrap();
      let c3 = Self::from(1.6867633134).unwrap();
      let mut x = self;
      let mut y;

      if x < Self::zero() {
          x = -x;
          if x > Self::one() {
              x = x.recip();
              if x > tan_twelfth_pi {
                  x = (x - tan_sixth_pi) / (Self::one() + tan_sixth_pi * x);
                  let x2 = x * x;
                  y = x * (c1 + x2 * c2) / (c3 + x2);
                  y += sixth_pi;
              } else {
                  let x2 = x * x;
                  y = x * (c1 + x2 * c2) / (c3 + x2);
              }
              y = half_pi - y;
          } else {
              if x > tan_twelfth_pi {
                  x = (x - tan_sixth_pi) / (Self::one() + tan_sixth_pi * x);
                  let x2 = x * x;
                  y = x * (c1 + x2 * c2) / (c3 + x2);
                  y += sixth_pi;
              } else {
                  let x2 = x * x;
                  y = x * (c1 + x2 * c2) / (c3 + x2);
                }
            }
          -y
      } else {
          if x > Self::one() {
              x = x.recip();
              if x > tan_twelfth_pi {
                  x = (x - tan_sixth_pi) / (Self::one() + tan_sixth_pi * x);
                  let x2 = x * x;
                  y = x * (c1 + x2 * c2) / (c3 + x2);
                  y += sixth_pi;
              } else {
                  let x2 = x * x;
                  y = x * (c1 + x2 * c2) / (c3 + x2);
              }
              y = half_pi - y;
          } else {
              if x > tan_twelfth_pi {
                  x = (x - tan_sixth_pi) / (Self::one() + tan_sixth_pi * x);
                  let x2 = x * x;
                  y = x * (c1 + x2 * c2) / (c3 + x2);
                  y += sixth_pi;
              } else {
                  let x2 = x * x;
                  y = x * (c1 + x2 * c2) / (c3 + x2);
              }
          }
          y
        }
    }

    /// Heaviside step function: 0 if x < 0, 1 if x >= 0.
    fn heaviside(self) -> Self {
        if self >= Self::zero() { Self::one() } else { Self::zero() }
    }

    /// Clamp value to the range [low, high].
    fn clamp(self, low: Self, high: Self) -> Self {
        if self < low {
            low
        } else if self > high {
            high
        } else {
            self
        }
    }
}

impl Float for f32 {
    fn pi() -> f32 { std::f32::consts::PI }
    fn two() -> f32 { 2.0f32 }
    fn four() -> f32 { 4.0f32 }
    fn half() -> f32 { 0.5f32 }
}

impl Float for f64 {
    fn pi() -> f64 { std::f64::consts::PI }
    fn two() -> f64 { 2.0f64 }
    fn four() -> f64 { 4.0f64 }
    fn half() -> f64 { 0.5f64 }
}

/// Convert degrees to radians.
pub fn deg2rad<T: Float>(v: T) -> T { v.deg2rad() }
/// Convert radians to degrees.
pub fn rad2deg<T: Float>(v: T) -> T { v.rad2deg() }
/// Normalize radians to [-pi, pi].
pub fn rad2rad<T: Float>(v: T) -> T { v.rad2rad() }
/// Rollover-safe difference of two radian angles, result in [-pi, pi].
pub fn rad_diff<T: Float>(a: T, b: T) -> T { a.rad_diff(b) }
/// Rollover-safe sum of two radian angles, result in [-pi, pi].
pub fn rad_sum<T: Float>(a: T, b: T) -> T { a.rad_sum(b) }

/// Machine epsilon of the anchor value's type (`f32::EPSILON` / `f64::EPSILON`).
///
/// The anchor value itself is ignored; only its TYPE is used, so the symbolic
/// codegen can emit `arael::utils::epsilon_for(x)` in a type-inferred context
/// and have `T` resolved from a nearby concrete value. A bare `T::epsilon()`
/// call would give an unpinnable inference variable; passing an anchor pins it,
/// so f32 constraint code gets `f32::EPSILON` and f64 code `f64::EPSILON`
/// (a folded literal would bake in one precision). Used by the `safe_asin` /
/// `safe_acos` derivative guards.
///
/// ```
/// assert_eq!(arael::utils::epsilon_for(1.0_f32), f32::EPSILON);
/// assert_eq!(arael::utils::epsilon_for(1.0_f64), f64::EPSILON);
/// ```
#[inline(always)]
pub fn epsilon_for<T: Float>(_anchor: T) -> T { <T as num::Float>::epsilon() }

/// Safe asin that clamps input to [-1, 1].
pub fn safe_asin<T: Float>(v: T) -> T { v.safe_asin() }
/// Safe acos that clamps input to [-1, 1].
pub fn safe_acos<T: Float>(v: T) -> T { v.safe_acos() }

/// Safe square root: asserts the input is not significantly negative, clamps
/// noise-level negatives to zero.
pub fn safe_sqrt<T: Float>(v: T) -> T {
    debug_assert!(v >= T::from(-1e-6).unwrap(),
        "safe_sqrt called with significantly negative value: {:?}", v);
    if v <= T::zero() { T::zero() } else { v.sqrt() }
}

/// Standard atan2(y, x).
pub fn atan2<T: Float>(y: T, x: T) -> T { y.atan2(x) }

/// Fast approximate atan. Max error < 1e-6 radians.
pub fn fast_atan<T: Float>(x: T) -> T { x.fast_atan() }

/// Fast approximate atan2. Max error < 1e-6 radians. Does not handle atan2(+-inf, +-inf).
pub fn fast_atan2<T: Float>(y: T, x: T) -> T {
    if x > T::zero() {
        (y / x).fast_atan()
    } else if x == T::zero() {
        if y == T::zero() {
            return T::zero();
        }
        T::half_pi().copysign(y)
    } else { // x < 0
        if y >= T::zero() {
            (y / x).fast_atan() + T::pi()
        } else {
            (y / x).fast_atan() - T::pi()
        }
    }
}

macro_rules! left_side_scalar_multiplication {
    ($right_side_type:ty, $scalar_type:ty) => {
        impl ops::Mul<$right_side_type> for $scalar_type
        {
            type Output = $right_side_type;
            fn mul(self, _rhs: $right_side_type) -> $right_side_type {
                _rhs * self
            }
        }
    }
}

pub(crate) use left_side_scalar_multiplication;


#[cfg(test)]
mod tests {
    use super::*;

    fn equal<T: Float>(a: T, b: T) -> bool {
        (a - b).abs() < T::from(10).unwrap() * (a.abs() + b.abs() + T::epsilon()) * T::epsilon()
    }

    #[test]
    fn test() {
        // sanity of our equal function
        assert!( equal(1.0, 1.000000000000001));
        assert!(!equal(1.0, 1.00000000000001));
        // deg2rad and rad2deg
        assert_eq!(deg2rad(180.0), f64::pi());
        assert_eq!(deg2rad(-180.0), -f64::pi());
        assert_eq!(rad2deg(f64::pi()), 180.0);
        assert_eq!(rad2deg(-f64::pi()), -180.0);
        // rad2similar
        assert_eq!(-0.25.rad2similar(0.01), -0.25);
        assert_eq!(1.1.rad2similar(100.0), 1.1 + 16.0 * f64::two_pi());
        assert_eq!(1.1.rad2similar(-100.0), 1.1 - 16.0 * f64::two_pi());
        // rad2rad
        assert!(equal((2.0*f64::pi() + 0.12).rad2rad(), 0.12));
        assert!(equal((-2.0*f64::pi() - 0.12).rad2rad(), -0.12));
        assert!(equal(rad2rad(7.0 * f64::pi() + 0.5), -f64::pi() + 0.5));
        assert!(equal(rad2rad(0.0), 0.0));
        // rad_diff — rollover safe
        assert!(equal(rad_diff(0.1, 2.0 * f64::pi() + 0.1), 0.0));
        assert!(equal(rad_diff(0.1, -0.1), 0.2));
        assert!(equal(rad_diff(f64::pi() - 0.1, -f64::pi() + 0.1), -0.2));  // across +-pi boundary
        assert!(equal(rad_diff(0.0, f64::pi()), -f64::pi()));
        // rad_sum — rollover safe
        assert!(equal(rad_sum(f64::pi() - 0.1, 0.2), -f64::pi() + 0.1));  // wraps past pi
        assert!(equal(rad_sum(-f64::pi() + 0.1, -0.2), f64::pi() - 0.1));  // wraps past -pi
        assert!(equal(rad_sum(0.0, 0.0), 0.0));
        assert!(equal(rad_sum(1.0, -1.0), 0.0));
        // constants
        assert_eq!(f64::two(), 2.0);
        assert_eq!(f64::four(), 4.0);
        assert_eq!(f64::half(), 0.5);
        // safe_acos
        assert_eq!(1.1.safe_acos(), 0.0);
        assert_eq!(0.0.safe_acos(), f64::half_pi());
        assert_eq!((-1.1).safe_acos(), f64::pi());
        // safe_asin
        assert_eq!(1.1.safe_asin(), f64::half_pi());
        assert!(equal(0.5.safe_asin(), 0.5_f64.asin()));
        assert_eq!((-1.1).safe_asin(), -f64::half_pi());
    }

    #[test]
    fn fast_atan() {
        let max_value = 10000.0;
        let iters = 100000;
        let mut max_err : f64 = 0.0;
        let mut worst_x : f64 = 0.0;
        for n in -iters..=iters {
            let x = n as f64 / iters as f64 * max_value;
            let err = x.atan() - x.fast_atan();
            if err.abs() > max_err.abs() {
                max_err = err;
                worst_x = x;
            }
        }
        println!("fast_atan max error: atan({:?}) err={:?} degrees", worst_x, max_err.rad2deg());
        assert!(max_err.abs() < 1e-6);
    }

    #[test]
    fn test_fast_atan2() {
        let max_value = 10.0;
        let iters = 10;
        let mut max_err : f64 = 0.0;
        let mut worst_x : f64 = 0.0;
        let mut worst_y : f64 = 0.0;
        for m in -iters..=iters {
            let y = m as f64 / iters as f64 * max_value;
            for n in -iters..=iters {
                if n == 0 && m == 0 {
                    continue;
                }
                let x = n as f64 / iters as f64 * max_value;
                let err = atan2(y, x) - fast_atan2(y, x);
                if err.abs() > max_err.abs() {
                    max_err = err;
                    worst_x = x;
                    worst_y = y;
                }
            }
        }
        println!("fast_atan2 max error: atan2({:?}, {:?}) is {:?} degrees", worst_y, worst_x, max_err.rad2deg());
        assert!(max_err.abs() < 1e-6);
    }
}


