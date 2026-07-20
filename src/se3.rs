//! Compact form of a rigid transform: translation and rotation in six
//! numbers.
//!
//! A twist holds a rate of translation `d` and a rate of rotation `w`,
//! both applied together over one unit step. That is a translation and a
//! rotation in another form, and you can convert either way:
//!
//! ```ignore
//! let (t, q) = twist.translation_rotation();
//! let twist  = se3d::from_translation_rotation(t, q);
//! ```
//!
//! The round trip is exact. Two things about the form are worth knowing.
//!
//! **The translation is not `d`.** The frame turns while it moves, so the
//! path curves: you travel `d` carried through that rotation -- the arc a
//! turning body traces, not the chord. Turning 90 degrees while running at
//! rate 1 moves you 0.9, not 1.0. With no rotation the two are the same.
//!
//! **It is a simplified se(3).** The rotation uses the small-angle form,
//! turning by `2 atan(|w| / 2)` rather than `|w|`, and the translation
//! carry keeps the first three terms of its series. This costs no accuracy
//! in use: both are exact, invertible maps -- a different chart, not a
//! lossy one -- and the round trip above holds to 1e-12 far outside the
//! small-angle regime. What it buys is arithmetic free of trigonometry,
//! so the derivatives arael generates from it stay clean, and a rotation
//! step identical to [`QuaternionParam`](crate::model::QuaternionParam)'s.
//!
//! To combine two twists, convert to translation and rotation, compose
//! there, and convert back -- there is no direct formula for composing
//! them.
//!
//! [`SE3Param`](crate::se3param::SE3Param) steps a pose in these
//! coordinates: it holds a reference frame and folds each accepted step
//! into it, so the step is always expressed in the pose's own frame.

use crate::quatern::quatern;
use crate::utils::Float;
use crate::vect::vect3;

/// A translation and a rotation, held as the rates that produce them over
/// one unit step. See the module docs.
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct se3<T: Float> {
    /// Rate of translation. The distance actually travelled is not this
    /// -- the frame turns as it moves; see the module docs.
    pub d: vect3<T>,
    /// Rate of rotation: axis times `2 tan(angle / 2)`. Note the factor --
    /// this is the small-angle form, not axis times angle, so a quarter
    /// turn is `|w| = 2`, not `pi/2`.
    pub w: vect3<T>,
}

#[allow(non_camel_case_types)]
pub type se3f = se3<f32>;
#[allow(non_camel_case_types)]
pub type se3d = se3<f64>;

fn c<T: Float>(x: f64) -> T {
    T::from(x).unwrap()
}

impl<T: Float> se3<T> {
    /// A twist from its linear and angular rates.
    pub fn new(d: vect3<T>, w: vect3<T>) -> se3<T> {
        se3 { d, w }
    }

    /// The transform that changes nothing.
    pub fn zero() -> se3<T> {
        let z = vect3::new(T::zero(), T::zero(), T::zero());
        se3 { d: z, w: z }
    }

    /// This transform as a translation and a rotation. Inverse of
    /// [`from_translation_rotation`](se3::from_translation_rotation).
    pub fn translation_rotation(self) -> (vect3<T>, quatern<T>) {
        (self.translation(), self.rotation())
    }

    /// The translation: the rate carried through the rotation happening
    /// alongside it, so a turning step follows the arc.
    pub fn translation(self) -> vect3<T> {
        carry(self.w, self.d)
    }

    /// The rotation.
    pub fn rotation(self) -> quatern<T> {
        quatern::from_rotation_vector_small(self.w)
    }

    /// This translation and rotation in twist form. Inverse of
    /// [`translation_rotation`](se3::translation_rotation).
    ///
    /// Undefined for a half turn, where the rotation's scalar part
    /// vanishes -- the same limit the pose parameter's chart has, and one
    /// its re-centring keeps a solver far away from.
    pub fn from_translation_rotation(t: vect3<T>, q: quatern<T>) -> se3<T> {
        let q = q.unit();
        let w = q.v * (c::<T>(2.0) / q.t);
        se3 { d: carry_inverse(w, t), w }
    }
}

impl<T: Float> Default for se3<T> {
    fn default() -> Self {
        se3::zero()
    }
}

/// The translation a twist produces: `d` carried through the rotation
/// happening at the same time, keeping the first three terms of the
/// series. Polynomial, so derivatives stay free of the removable
/// singularity the closed form carries at zero rotation.
pub(crate) fn carry<T: Float>(w: vect3<T>, d: vect3<T>) -> vect3<T> {
    d + (w % d) * c::<T>(0.5) + (w % (w % d)) * c::<T>(1.0 / 6.0)
}

/// The exact inverse of [`carry`] for the same `w`: recovers the linear
/// rate from the translation it produced.
///
/// `carry` is `I + K/2 + K^2/6` applied to `d`, with `K` the cross-product
/// matrix of `w`. Because `K^3 = -|w|^2 K`, the inverse has the same shape
/// -- `I + a K + b K^2` -- with coefficients solvable in closed form, so no
/// matrix inversion and no singularity (the denominator is a sum of
/// squares that cannot both vanish).
fn carry_inverse<T: Float>(w: vect3<T>, t: vect3<T>) -> vect3<T> {
    let th2 = w * w;
    let u = T::one() - th2 * c::<T>(1.0 / 6.0);
    let b = (c::<T>(1.0 / 3.0) + th2 * c::<T>(1.0 / 9.0))
        / (c::<T>(4.0) * u * u + th2);
    let a = -(c::<T>(2.0) * b * u) - c::<T>(1.0 / 3.0);
    t + (w % t) * a + (w % (w % t)) * b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Converting to translation and rotation and back returns the
    /// same twist, well past the small-angle regime.
    #[test]
    fn round_trips() {
        for (d, w) in [
            (vect3::new(0.0, 0.0, 0.0), vect3::new(0.0, 0.0, 0.0)),
            (vect3::new(0.4, -0.2, 0.1), vect3::new(0.0, 0.0, 0.0)),
            (vect3::new(0.0, 0.0, 0.0), vect3::new(0.13, -0.07, 0.21)),
            (vect3::new(1.3, 0.5, -0.9), vect3::new(0.31, 0.22, -0.44)),
            (vect3::new(2.0, -1.0, 0.5), vect3::new(1.2, -0.8, 0.6)),
        ] {
            let twist = se3d::new(d, w);
            let (t, q) = twist.translation_rotation();
            let back = se3d::from_translation_rotation(t, q);
            assert!((back.d - d).norm() < 1e-12, "d: {:?} vs {:?}",
                (d.x, d.y, d.z), (back.d.x, back.d.y, back.d.z));
            assert!((back.w - w).norm() < 1e-12, "w: {:?} vs {:?}",
                (w.x, w.y, w.z), (back.w.x, back.w.y, back.w.z));
        }
    }

    /// A quarter turn while running one unit forward traces a quarter
    /// circle: one radius forward and one radius left, displaced less than
    /// the rate. With no rotation the translation IS the rate.
    #[test]
    fn a_turning_step_traces_the_arc() {
        let turning = se3d::new(vect3::new(1.0, 0.0, 0.0),
            vect3::new(0.0, 0.0, std::f64::consts::FRAC_PI_2));
        let t = turning.translation();
        assert!(t.x > 0.55 && t.x < 0.75, "{:?}", (t.x, t.y));
        assert!(t.y > 0.55 && t.y < 0.85, "{:?}", (t.x, t.y));
        assert!(t.norm() < 1.0, "the arc displaces less than the chord");

        let straight = se3d::new(vect3::new(1.0, 2.0, -3.0),
            vect3::new(0.0, 0.0, 0.0));
        assert!((straight.translation() - vect3::new(1.0, 2.0, -3.0)).norm() < 1e-15);
    }

    /// The rotation is the same retraction the rotation params use, so a
    /// pure-rotation twist agrees with the quaternion form exactly.
    #[test]
    fn rotation_matches_the_rotation_retraction() {
        for w in [vect3::new(0.0, 0.0, 0.0),
                  vect3::new(0.05, -0.02, 0.03),
                  vect3::new(0.4, 0.25, -0.3)] {
            let a = se3d::new(vect3::new(0.0, 0.0, 0.0), w).rotation();
            let b = quatern::from_rotation_vector_small(w);
            assert!((a.v - b.v).norm() < 1e-15 && (a.t - b.t).abs() < 1e-15);
        }
    }
}
