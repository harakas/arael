//! Quaternion type for 3D rotations.

#![allow(non_camel_case_types)]

use std::ops;
use std::fmt;
use crate::utils::{left_side_scalar_multiplication};
use crate::utils::Float;
use crate::vect::vect3;
use crate::vect::Similar;
use crate::matrix::matrix3;

/// Quaternion with scalar part `t` and vector part `v` (t + xi + yj + zk).
///
/// Supports addition, subtraction, negation, scalar multiplication, and quaternion
/// multiplication (`*` operator). Unit quaternions represent 3D rotations.
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct quatern<T : Float> {
    pub t : T,
    pub v : vect3<T>
}

/// Quaternion with f32 components.
pub type quaternf = quatern<f32>;
/// Quaternion with f64 components.
pub type quaternd = quatern<f64>;

impl<T: Float> fmt::Debug for quatern<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Q({:?}, V({:?}, {:?}, {:?}))", self.t, self.v.x, self.v.y, self.v.z)
    }
}

impl<T: Float> ops::Add<quatern<T>> for quatern<T> {
    type Output = quatern<T>;
    fn add(self, _rhs: quatern<T>) -> quatern<T> {
        quatern::<T> {t: self.t + _rhs.t, v: self.v + _rhs.v}
    }
}

impl<T: Float> ops::Sub<quatern<T>> for quatern<T> {
    type Output = quatern<T>;
    fn sub(self, _rhs: quatern<T>) -> quatern<T> {
        quatern::<T> {t: self.t - _rhs.t, v: self.v - _rhs.v}
    }
}

impl<T: Float> ops::Mul<T> for quatern<T> {
    type Output = quatern<T>;
    fn mul(self, _rhs: T) -> quatern<T> {
        quatern::<T> {t: self.t * _rhs, v: self.v * _rhs}
    }
}

left_side_scalar_multiplication!(quaternf,f32);
left_side_scalar_multiplication!(quaternd,f64);

impl<T: Float> ops::Neg for quatern<T> {
    type Output = quatern<T>;
    fn neg(self) -> quatern<T> {
        quatern::<T> {t: -self.t, v: -self.v}
    }
}

impl<T: Float> ops::Mul<quatern<T>> for quatern<T> {
    type Output = quatern<T>;
    fn mul(self, _rhs: quatern<T>) -> quatern<T> {
        quatern::<T> {t: self.t * _rhs.t - self.v * _rhs.v, v: _rhs.v * self.t + self.v * _rhs.t + self.v % _rhs.v}
    }
}

impl<T: Float> quatern<T> {
    /// Constructs a quaternion from a scalar part and a vector part.
    pub fn new(t: T, v: vect3<T>) -> quatern<T> {
        quatern::<T> { t, v }
    }

    /// Returns the identity quaternion (1 + 0i + 0j + 0k), representing no rotation.
    pub fn identity() -> quatern<T> {
        quatern::<T>::new(T::one(), vect3::<T>::new(T::zero(), T::zero(), T::zero()))
    }

    /// Returns the dot product of two quaternions.
    pub fn dot(self, q: quatern<T>) -> T {
        self.t * q.t + self.v * q.v
    }

    /// Returns the norm (magnitude) of the quaternion.
    pub fn norm(self) -> T {
        self.dot(self).sqrt()
    }

    /// Returns the unit (normalized) quaternion.
    pub fn unit(self) -> quatern<T> {
        self * self.norm().recip()
    }

    /// Returns the conjugate (negated vector part).
    pub fn conj(self) -> quatern<T> {
        quatern::<T>::new(self.t, -self.v)
    }

    /// Constructs a `quatern<T>` by converting from a `quatern<K>` of a different float type.
    pub fn from<K: Float>(q: quatern<K>) -> quatern<T> {
        quatern::<T>::new(T::from(q.t).unwrap(), vect3::<T>::from(q.v))
    }

    /// Converts this quaternion to a `quatern<K>` of a different float type.
    pub fn cast<K: Float>(self) -> quatern<K> {
        quatern::<K>::new(K::from(self.t).unwrap(), self.v.cast::<K>())
    }

    /// Logs an error if this quaternion is not approximately unit length.
    pub fn assert_unit_length(self) {
        if (self.dot(self) - T::one()).abs() > T::from(1e5).unwrap() * T::epsilon() {
            error!("arael::quatern: not unit length, norm = {:?}", self.norm());
        }
    }

    /// Rotates a 3D vector by this unit quaternion: q * v * q'.
    pub fn rotate(self, v: vect3<T>) -> vect3<T> {
        (self * quatern::<T>::new(T::zero(), v) * self.conj()).v
    }

    /// Converts this unit quaternion to the equivalent 3x3 rotation matrix.
    pub fn rotation_matrix(self) -> matrix3<T> {
        let (v, t) = (self.v, self.t);
        let x2 = v.x * v.x;
        let y2 = v.y * v.y;
        let z2 = v.z * v.z;
        matrix3::<T>::from_rows(
            vect3::<T>::new(T::one() - T::two() * (y2 + z2),      T::two()*(v.x*v.y - v.z*t),      T::two()*(v.x*v.z + v.y*t)),
            vect3::<T>::new(     T::two()*(v.x*v.y + v.z*t), T::one() - T::two() * (x2 + z2),      T::two()*(v.y*v.z - v.x*t)),
            vect3::<T>::new(     T::two()*(v.x*v.z - v.y*t),      T::two()*(v.y*v.z + v.x*t), T::one() - T::two() * (x2 + y2))
        )
    }

    /// Extracts the rotation axis and angle from a unit quaternion.
    /// Returns (axis, angle) where axis is a unit vector and angle is in radians.
    pub fn get_axis_angle(self) -> (vect3<T>, T) {
        let angle = (T::two() * self.t.safe_acos()).rad2rad();
        let s2 = T::one() - self.t * self.t;
        if s2 > T::epsilon() * T::epsilon() {
            (self.v * s2.sqrt().recip(), angle)
        } else {
            (vect3::<T>::new(T::one(), T::zero(), T::zero()), T::zero())
        }
    }

    /// Extracts Euler angles (x=roll, y=pitch, z=yaw) from a unit quaternion.
    pub fn get_euler_angles(self) -> vect3<T> {
        let ea_x = T::atan2(T::two() * (self.t * self.v.x + self.v.y * self.v.z), T::one() - T::two() * (self.v.x * self.v.x + self.v.y * self.v.y));
        let ea_z = T::atan2(T::two() * (self.t * self.v.z + self.v.x * self.v.y), T::one() - T::two() * (self.v.y * self.v.y + self.v.z * self.v.z));

        let s = T::two() * (self.t * self.v.y - self.v.z * self.v.x);
        if s >= T::one() {
            vect3::<T>::new(ea_x, T::half_pi(), ea_z)
        } else if s <= -T::one() {
            vect3::<T>::new(ea_x, -T::half_pi(), ea_z)
        } else {
            vect3::<T>::new(ea_x, s.asin(), ea_z)
        }
    }

    /// Constructs a unit quaternion from Euler angles (x=roll, y=pitch, z=yaw).
    pub fn from_euler_angles(ea: vect3<T>) -> quatern<T> {
        let ha = ea * T::half();
        let (shax, chax) = ha.x.sin_cos();
        let (shay, chay) = ha.y.sin_cos();
        let (shaz, chaz) = ha.z.sin_cos();
        quatern::<T>::new(
            chax * chay * chaz + shax * shay * shaz,
            vect3::<T>::new(
                shax * chay * chaz - chax * shay * shaz,
                chax * shay * chaz + shax * chay * shaz,
                chax * chay * shaz - shax * shay * chaz
            )
        )
    }

    /// Constructs a unit quaternion from a rotation axis (must be unit) and angle in radians.
    pub fn from_axis_angle(normal: vect3<T>, angle: T) -> quatern<T> {
        let half_angle = T::half() * angle;
        let (sin_half_angle, cos_half_angle) = half_angle.sin_cos();
        quatern::<T>::new(cos_half_angle, normal * sin_half_angle)
    }

    /// Constructs a unit quaternion from a rotation vector `v = axis * angle`
    /// (the exp map of so(3)): `q = [cos(|v|/2), (v/|v|) sin(|v|/2)]`. The
    /// coefficients are even functions of `|v|`, computed exactly for
    /// `|v|^2 >= 1e-8` and by their 2-term Taylor below it, so there is no
    /// axis-normalization singularity at `v = 0`. Matches the symbolic
    /// `quaternsym::from_rotation_vector`.
    pub fn from_rotation_vector(v: vect3<T>) -> quatern<T> {
        let s = v.x * v.x + v.y * v.y + v.z * v.z;
        let theta = s.sqrt();
        let half = T::half() * theta;
        let eps = T::from(1e-8).unwrap();
        let (q_t, scale) = if s >= eps {
            (half.cos(), half.sin() / theta)
        } else {
            (T::one() - s * T::from(0.125).unwrap() + s * s * T::from(1.0 / 384.0).unwrap(),
             T::half() - s * T::from(1.0 / 48.0).unwrap())
        };
        quatern::<T>::new(q_t, v * scale)
    }

    /// First-order (small-angle) retraction of a rotation vector to a unit
    /// quaternion: `normalize(1, v/2)`. Branch-free and smooth for all v (the
    /// normalizing denominator `sqrt(1 + |v|^2/4)` is always >= 1). Agrees with
    /// the exact exp map `from_rotation_vector` to first order. Matches the
    /// symbolic `quaternsym::from_rotation_vector_small`.
    pub fn from_rotation_vector_small(v: vect3<T>) -> quatern<T> {
        quatern::<T>::new(T::one(), v * T::half()).unit()
    }

    /// Constructs a unit quaternion from a rotation matrix (Shepperd's
    /// method: the largest of the four squared components is computed
    /// first, so no branch divides by a small number).
    pub fn from_rotation_matrix(m: matrix3<T>) -> quatern<T> {
        let two = T::one() + T::one();
        let quarter = T::half() * T::half();
        let tr = m[0].x + m[1].y + m[2].z;
        let q = if tr > T::zero() {
            let s = (tr + T::one()).sqrt() * two;
            quatern::<T>::new(
                quarter * s,
                vect3::<T>::new(
                    (m[2].y - m[1].z) / s,
                    (m[0].z - m[2].x) / s,
                    (m[1].x - m[0].y) / s,
                ),
            )
        } else if m[0].x > m[1].y && m[0].x > m[2].z {
            let s = (T::one() + m[0].x - m[1].y - m[2].z).sqrt() * two;
            quatern::<T>::new(
                (m[2].y - m[1].z) / s,
                vect3::<T>::new(
                    quarter * s,
                    (m[0].y + m[1].x) / s,
                    (m[0].z + m[2].x) / s,
                ),
            )
        } else if m[1].y > m[2].z {
            let s = (T::one() + m[1].y - m[0].x - m[2].z).sqrt() * two;
            quatern::<T>::new(
                (m[0].z - m[2].x) / s,
                vect3::<T>::new(
                    (m[0].y + m[1].x) / s,
                    quarter * s,
                    (m[1].z + m[2].y) / s,
                ),
            )
        } else {
            let s = (T::one() + m[2].z - m[0].x - m[1].y).sqrt() * two;
            quatern::<T>::new(
                (m[1].x - m[0].y) / s,
                vect3::<T>::new(
                    (m[0].z + m[2].x) / s,
                    (m[1].z + m[2].y) / s,
                    quarter * s,
                ),
            )
        };
        q.unit()
    }

    /// Raises this unit quaternion to a scalar power. Scales the rotation angle by `f`.
    pub fn pow(self, f: T) -> quatern<T> {
        let (axis, angle) = self.get_axis_angle();
        Self::from_axis_angle(axis, f * angle)
    }

    /// Quaternion logarithm. Returns a pure quaternion whose vector part is axis * angle.
    pub fn log(self) -> quatern<T> {
        let (axis, angle) = self.get_axis_angle();
        Self::new(T::zero(), axis * angle)
    }

    /// Quaternion exponential. Inverse of `log()` -- converts a pure quaternion back
    /// to a unit rotation quaternion.
    pub fn exp(self) -> quatern<T> {
        let angle = self.v.norm();
        if angle < T::epsilon() {
            return Self::identity();
        }
        let axis = self.v * angle.recip();
        Self::from_axis_angle(axis, angle)
    }

    /// Constructs the shortest-arc unit quaternion that rotates unit vector `from` to
    /// unit vector `to`. Both arguments must be unit length.
    pub fn from_two_vectors(from: vect3<T>, to: vect3<T>) -> quatern<T> {
        from.assert_unit_length();
        to.assert_unit_length();

        // find a vector towards the middle of the rotation arc
        // this makes it at half angle -- we use this natural occurence
        // to our advantage to directly construct the quaternion
        let mut mid = (from + to) * T::half();

        // special case: if the two vectors are parallel but opposed
        // then we have full 180 degree rotation with any vector that is crossed to from/to
        let mid_len2 = mid * mid;
        if mid_len2 < T::epsilon() {
            return Self::from_axis_angle(from.across(), T::pi())
        }

        // normalize the mid vector
        mid = mid * mid_len2.sqrt().recip();

        Self::new(mid * to, mid % to)
    }

    /// Spherical linear interpolation between two unit quaternions. `f=0` returns
    /// `from`, `f=1` returns `to`.
    pub fn slerp(from: quatern<T>, to: quatern<T>, f: T) -> quatern<T> {
        // q1 = q0 * dq => dq = q0' * q1; dm = q0 * dq**f = q0 * (q0' * q1)**f
        from * (from.conj() * to).pow(f)
    }

    /// Returns true if `self` and `other` are approximately equal within floating-point tolerance.
    pub fn similar(self, other: quatern<T>) -> bool {
        (self.t - other.t).abs() < T::from(10).unwrap() * (self.t.abs() + other.t.abs() + T::epsilon()) * T::epsilon() && self.v.similar(other.v)
    }
}

// Re-export symbolic companion type from arael-sym
pub use arael_sym::quaternsym;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vect::vect3d;
    use crate::matrix::matrix3d;

    #[test]
    fn test_from_rotation_matrix() {
        // Angles chosen so the four Shepperd pivot branches are all hit:
        // near-identity (trace pivot) and three near-180-degree rotations
        // about each axis (the per-axis diagonal pivots).
        let cases = [
            vect3d::new(0.1, -0.2, 0.3),
            vect3d::new(3.0, 0.05, -0.02),
            vect3d::new(0.04, 3.0, 0.03),
            vect3d::new(0.01, -0.06, 3.1),
            vect3d::new(-1.4, 0.9, 2.2),
        ];
        for ea in cases {
            let q = quaternd::from_euler_angles(ea);
            let r = quaternd::from_rotation_matrix(q.rotation_matrix());
            // q and -q encode the same rotation; compare the matrices.
            assert!(r.rotation_matrix().similar(q.rotation_matrix()),
                "roundtrip failed for {:?}", ea);
            assert!((r.norm() - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn from_rotation_vector_runtime_exp_map() {
        // Zero rotation vector -> identity quaternion.
        let q0 = quaternd::from_rotation_vector(vect3d::new(0.0, 0.0, 0.0));
        assert!((q0.t - 1.0).abs() < 1e-12);
        assert!(q0.v.x.abs() < 1e-12 && q0.v.y.abs() < 1e-12 && q0.v.z.abs() < 1e-12);
        // v = axis*angle matches from_axis_angle(v/|v|, |v|) and is unit.
        for v in [vect3d::new(0.0, 0.0, std::f64::consts::FRAC_PI_2),
                  vect3d::new(0.3, -0.5, 0.2), vect3d::new(1.2, 0.0, 0.0)] {
            let theta = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
            let expected = quaternd::from_axis_angle(v * (1.0 / theta), theta);
            let q = quaternd::from_rotation_vector(v);
            assert!((q.t - expected.t).abs() < 1e-12
                && (q.v.x - expected.v.x).abs() < 1e-12
                && (q.v.y - expected.v.y).abs() < 1e-12
                && (q.v.z - expected.v.z).abs() < 1e-12, "exp map mismatch at {:?}", v);
            assert!((q.norm() - 1.0).abs() < 1e-12, "not unit at {:?}", v);
        }
    }

    #[test]
    fn from_rotation_vector_small_runtime_retraction() {
        // Always unit; equals normalize(1, v/2) by definition.
        for v in [vect3d::new(0.0, 0.0, 0.0), vect3d::new(0.3, -0.5, 0.2), vect3d::new(2.0, 1.0, -3.0)] {
            let q = quaternd::from_rotation_vector_small(v);
            assert!((q.norm() - 1.0).abs() < 1e-12, "not unit at {:?}", v);
            let expected = quaternd::new(1.0, v * 0.5).unit();
            assert!((q.t - expected.t).abs() < 1e-12
                && (q.v.x - expected.v.x).abs() < 1e-12
                && (q.v.y - expected.v.y).abs() < 1e-12
                && (q.v.z - expected.v.z).abs() < 1e-12, "!= normalize(1, v/2) at {:?}", v);
        }
        // Zero -> identity.
        let q0 = quaternd::from_rotation_vector_small(vect3d::new(0.0, 0.0, 0.0));
        assert!((q0.t - 1.0).abs() < 1e-12 && q0.v.x.abs() < 1e-12);
        // Agrees with the exact exp map to first order (small v).
        let sv = vect3d::new(1e-3, -2e-3, 5e-4);
        let (qr, qe) = (quaternd::from_rotation_vector_small(sv), quaternd::from_rotation_vector(sv));
        assert!((qr.t - qe.t).abs() < 1e-8
            && (qr.v.x - qe.v.x).abs() < 1e-8
            && (qr.v.y - qe.v.y).abs() < 1e-8
            && (qr.v.z - qe.v.z).abs() < 1e-8, "retraction should match exp to first order");
    }

    #[test]
    fn quatern_serde_roundtrip() {
        let q = quaternd::from_euler_angles(vect3d::new(0.1, -0.2, 0.3));
        let json = serde_json::to_string(&q).expect("serialize");
        let back: quaternd = serde_json::from_str(&json).expect("deserialize");
        assert!((q.t - back.t).abs() < 1e-12
            && (q.v.x - back.v.x).abs() < 1e-12
            && (q.v.y - back.v.y).abs() < 1e-12
            && (q.v.z - back.v.z).abs() < 1e-12);
    }

    #[test]
    fn test() {
        let q1 = quaternd::new(1.0, vect3d::new(4.0, 2.0, 2.0));
        let q2 = quaternd::new(3.0, vect3d::new(-3.0, 1.0, 0.0));
        assert_eq!(q1.t, 1.0); assert_eq!(q1.v.x, 4.0); assert_eq!(q1.v.y, 2.0); assert_eq!(q1.v.z, 2.0);
        // norm
        assert_eq!(q1.norm(), 5.0);
        // neg
        assert!((-q1).similar(quaternd::new(-1.0, vect3d::new(-4.0, -2.0, -2.0))));
        // unit
        assert!(q1.unit().similar(quaternd::new(1.0 / 5.0, vect3d::new(4.0 / 5.0, 2.0 / 5.0, 2.0 / 5.0))));
        // scalar multiplication
        assert!((2.0 * q1).similar(quaternd::new(2.0, vect3d::new(8.0, 4.0, 4.0))));
        assert!((q1 * 2.0).similar(quaternd::new(2.0, vect3d::new(8.0, 4.0, 4.0))));
        // conj
        assert!(q1.conj().similar(quaternd::new(1.0, vect3d::new(-4.0, -2.0, -2.0))));
        // cast sanity
        assert!(q1.cast::<f32>().cast::<f64>().similar(q1));
        // adding
        assert!((q1 + q2).similar(quaternd::new(4.0, vect3d::new(1.0, 3.0, 2.0))));
        // substracting
        assert!((q1 - q2).similar(quaternd::new(-2.0, vect3d::new(7.0, 1.0, 2.0))));
        // dot product
        assert_eq!(q1.dot(q2), 3.0 - 12.0 + 2.0 + 0.0);
        // outer product
        assert!((q1 * q2).similar(quaternd::new(13.0, vect3d::new(7.0, 1.0, 16.0))));
        // euler angles sanity and comparison with matrix3d
        let v = vect3d::new(30.0, -10.0, 20.0);
        let ea = vect3d::new(0.4, -0.26, 1.1);
        assert!(quaternd::from_euler_angles(ea).rotate(v).similar(matrix3d::rotation_from_euler_angles(ea) * v));
        assert!(quaternd::from_euler_angles(ea).get_euler_angles().similar(ea));
        assert!((quaternd::from_euler_angles(vect3d::new(0.0, 0.0, ea.z)) * quaternd::from_euler_angles(vect3d::new(0.0, ea.y, 0.0)) * quaternd::from_euler_angles(vect3d::new(ea.x, 0.0, 0.0))).similar(quaternd::from_euler_angles(ea)));
        // axis angle sanity
        let axis = vect3d::new(12.0, -1.0, 3.0).unit();
        let angle = 0.234;
        let (ret_axis, ret_angle) = quaternd::from_axis_angle(axis, angle).get_axis_angle();
        assert!(axis.similar(ret_axis));
        assert!((angle - ret_angle).abs() < 10.0*f64::EPSILON);
        assert!((quaternd::from_axis_angle(axis, angle).rotate(v) - vect3d::new(30.282, -12.606, 18.002)).norm() < 0.001);
    }

    #[test]
    fn test_exp_zero_vector() {
        // exp of zero vector part should be identity, not NaN
        let q = quaternd::new(0.0, vect3d::new(0.0, 0.0, 0.0));
        let r = q.exp();
        assert!(r.t.is_finite());
        assert!(r.v.is_finite());
        assert!(r.similar(quaternd::identity()));
    }

    #[test]
    fn test_log_exp_roundtrip() {
        let axis = vect3d::new(1.0, 2.0, 3.0).unit();
        let angle = 0.8;
        let q = quaternd::from_axis_angle(axis, angle);
        let roundtrip = q.log().exp();
        assert!(q.similar(roundtrip));
    }

    #[test]
    fn test_slerp_endpoints() {
        let q1 = quaternd::from_euler_angles(vect3d::new(0.1, 0.2, 0.3));
        let q2 = quaternd::from_euler_angles(vect3d::new(0.5, -0.1, 1.0));
        assert!(quaternd::slerp(q1, q2, 0.0).similar(q1));
        assert!(quaternd::slerp(q1, q2, 1.0).similar(q2));
    }

    #[test]
    fn test_slerp_midpoint() {
        let axis = vect3d::new(0.0, 0.0, 1.0);
        let q1 = quaternd::from_axis_angle(axis, 0.0);
        let q2 = quaternd::from_axis_angle(axis, 1.0);
        let mid = quaternd::slerp(q1, q2, 0.5);
        let (_, mid_angle) = mid.get_axis_angle();
        assert!((mid_angle - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_pow_identity() {
        let axis = vect3d::new(1.0, 0.0, 0.0);
        let q = quaternd::from_axis_angle(axis, 0.6);
        assert!(q.pow(1.0).similar(q));
        assert!(q.pow(0.0).similar(quaternd::identity()));
    }

    #[test]
    fn test_from_two_vectors_opposite() {
        // 180-degree rotation (edge case)
        let a = vect3d::new(1.0, 0.0, 0.0);
        let b = vect3d::new(-1.0, 0.0, 0.0);
        let q = quaternd::from_two_vectors(a, b);
        let rotated = q.rotate(a);
        assert!(rotated.similar(b));
    }

    #[test]
    fn test_from_two_vectors_same() {
        let a = vect3d::new(0.0, 1.0, 0.0);
        let q = quaternd::from_two_vectors(a, a);
        assert!(q.rotate(a).similar(a));
    }
}

