//! 2D and 3D vector types with standard math operations.

#![allow(non_camel_case_types)]

use std::ops;
use std::fmt;
use crate::utils::{deg2rad, rad2deg, left_side_scalar_multiplication};
use crate::utils::Float;

/// Trait for approximate floating-point equality comparisons.
pub trait Similar {
    /// Returns true if `self` and `other` are approximately equal within floating-point tolerance.
    fn similar(self, other: Self) -> bool;
}

/// 3D vector with x, y, z components.
///
/// Supports addition, subtraction, negation, scalar multiplication, dot product
/// (`*` operator), and cross product (`%` operator). Indexable by `usize` (0=x, 1=y, 2=z).
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct vect3<T : Float>
{
    pub x : T,
    pub y : T,
    pub z : T,
}

/// 3D vector with f32 components.
pub type vect3f = vect3<f32>;
/// 3D vector with f64 components.
pub type vect3d = vect3<f64>;

impl<T: Float> Default for vect3<T> {
    fn default() -> Self { vect3 { x: T::zero(), y: T::zero(), z: T::zero() } }
}

impl<T: Float> fmt::Debug for vect3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}, {:?}, {:?}]", self.x, self.y, self.z)
    }
}

impl<T: Float> ops::Add<vect3<T>> for vect3<T>
{
    type Output = vect3<T>;
    fn add(self, _rhs: vect3<T>) -> vect3<T> {
        vect3::<T> {x: self.x + _rhs.x, y: self.y + _rhs.y, z: self.z + _rhs.z}
    }
}

impl<T: Float> ops::Sub<vect3<T>> for vect3<T>
{
    type Output = vect3<T>;
    fn sub(self, _rhs: vect3<T>) -> vect3<T> {
        vect3::<T> {x: self.x - _rhs.x, y: self.y - _rhs.y, z: self.z - _rhs.z}
    }
}

impl<T: Float> ops::Mul<vect3<T>> for vect3<T>
{
    type Output = T;
    fn mul(self, _rhs: vect3<T>) -> T {
        self.x * _rhs.x + self.y * _rhs.y + self.z * _rhs.z
    }
}

impl<T: Float> ops::Mul<T> for vect3<T>
{
    type Output = vect3<T>;
    fn mul(self, _rhs: T) -> vect3<T> {
        vect3::<T> {x: self.x * _rhs, y: self.y * _rhs, z: self.z * _rhs}
    }
}

left_side_scalar_multiplication!(vect3f,f32);
left_side_scalar_multiplication!(vect3d,f64);

impl<T: Float> ops::Neg for vect3<T>
{
    type Output = vect3<T>;
    fn neg(self) -> vect3<T> {
        vect3::<T> {x: -self.x, y: -self.y, z: -self.z}
    }
}

impl<T: Float> ops::Rem for vect3<T>
{
    type Output = vect3<T>;
    fn rem(self, _rhs: vect3<T>) -> vect3<T> {
        vect3::<T> {x: self.y * _rhs.z - self.z * _rhs.y, y: self.z * _rhs.x - self.x * _rhs.z, z: self.x * _rhs.y - self.y * _rhs.x}
    }
}

impl<T: Float> ops::Div<T> for vect3<T>
{
    type Output = vect3<T>;
    fn div(self, _rhs: T) -> vect3<T> {
        vect3::<T> {x: self.x / _rhs, y: self.y / _rhs, z: self.z / _rhs}
    }
}

impl<T: Float> vect3<T> {
    /// Constructs a 3D vector from components.
    pub fn new(x: T, y: T, z: T) -> vect3<T> {
        vect3::<T> { x, y, z}
    }

    /// Returns the squared magnitude (dot product with itself).
    pub fn square(self) -> T {
        self * self
    }

    /// Returns the Euclidean norm (length).
    pub fn norm(self) -> T {
        self.square().sqrt()
    }

    /// Returns the unit (normalized) vector.
    pub fn unit(self) -> vect3<T> {
        self * self.norm().recip()
    }

    /// Returns a unit vector perpendicular to `self`.
    pub fn across(self) -> vect3<T> {
        if self.y.abs() < self.x.abs() {
            vect3::<T>::new(-self.z, T::zero(), self.x).unit()
        } else {
            vect3::<T>::new(T::zero(), self.z, -self.y).unit()
        }
    }

    /// Returns true if all components are finite (not NaN or infinity).
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Converts all components from degrees to radians.
    pub fn deg2rad(self) -> vect3<T> {
        vect3::<T>::new(deg2rad(self.x), deg2rad(self.y), deg2rad(self.z))
    }

    /// Converts all components from radians to degrees.
    pub fn rad2deg(self) -> vect3<T> {
        vect3::<T>::new(rad2deg(self.x), rad2deg(self.y), rad2deg(self.z))
    }

    /// Constructs a `vect3<T>` by converting from a `vect3<K>` of a different float type.
    pub fn from<K: Float>(v: vect3<K>) -> vect3<T> {
        vect3::<T>::new(T::from(v.x).unwrap(), T::from(v.y).unwrap(), T::from(v.z).unwrap())
    }

    /// Converts this vector to a `vect3<K>` of a different float type.
    pub fn cast<K: Float>(self) -> vect3<K> {
        vect3::<K>::new(K::from(self.x).unwrap(), K::from(self.y).unwrap(), K::from(self.z).unwrap())
    }

    /// Logs an error if this vector is not approximately unit length.
    pub fn assert_unit_length(self) {
        if (self.square() - T::one()).abs() > T::from(1e5).unwrap() * T::epsilon() {
            error!("arael::vect: not unit length, norm = {:?}, error = {:?}", self.norm(), T::one() - self.norm());
        }
    }

    /// Builds a 3x3 rotation matrix from this vector interpreted as Euler angles
    /// (x=roll, y=pitch, z=yaw).
    pub fn rotation_matrix(self) -> crate::matrix::matrix3<T> {
        crate::matrix::matrix3::rotation_from_euler_angles(self)
    }

    /// Computes element-wise sin and cos. Returns `(sin_vec, cos_vec)`.
    pub fn sincos(self) -> (vect3<T>, vect3<T>) {
        let (sin_x, cos_x) = self.x.sin_cos();
        let (sin_y, cos_y) = self.y.sin_cos();
        let (sin_z, cos_z) = self.z.sin_cos();
        (vect3::<T>::new(sin_x, sin_y, sin_z), vect3::<T>::new(cos_x, cos_y, cos_z))
    }
}

impl<T: Float> Similar for vect3<T> {
    fn similar(self, other: vect3<T>) -> bool {
        (self - other).norm() < T::from(10).unwrap() * (self.norm() + other.norm() + T::epsilon()) * T::epsilon()
    }
}

impl<T: Float> ops::Index<usize> for vect3<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            2 => &self.z,
            _ => panic!("arael::vect: index {} out of bounds", index)
        }
    }
}

impl<T: Float> ops::IndexMut<usize> for vect3<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            2 => &mut self.z,
            _ => panic!("arael::vect: index {} out of bounds", index)
        }
    }
}

/// 2D vector with x, y components.
///
/// Supports addition, subtraction, negation, scalar multiplication, and dot product
/// (`*` operator). Indexable by `usize` (0=x, 1=y).
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct vect2<T : Float>
{
    pub x : T,
    pub y : T,
}

/// 2D vector with f32 components.
pub type vect2f = vect2<f32>;
/// 2D vector with f64 components.
pub type vect2d = vect2<f64>;

impl<T: Float> Default for vect2<T> {
    fn default() -> Self { vect2 { x: T::zero(), y: T::zero() } }
}

impl<T: Float> fmt::Debug for vect2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}, {:?}]", self.x, self.y)
    }
}

impl<T: Float> ops::Add<vect2<T>> for vect2<T>
{
    type Output = vect2<T>;
    fn add(self, _rhs: vect2<T>) -> vect2<T> {
        vect2::<T> {x: self.x + _rhs.x, y: self.y + _rhs.y}
    }
}

impl<T: Float> ops::Sub<vect2<T>> for vect2<T>
{
    type Output = vect2<T>;
    fn sub(self, _rhs: vect2<T>) -> vect2<T> {
        vect2::<T> {x: self.x - _rhs.x, y: self.y - _rhs.y}
    }
}

impl<T: Float> ops::Mul<vect2<T>> for vect2<T>
{
    type Output = T;
    fn mul(self, _rhs: vect2<T>) -> T {
        self.x * _rhs.x + self.y * _rhs.y
    }
}

impl<T: Float> ops::Mul<T> for vect2<T>
{
    type Output = vect2<T>;
    fn mul(self, _rhs: T) -> vect2<T> {
        vect2::<T> {x: self.x * _rhs, y: self.y * _rhs}
    }
}

left_side_scalar_multiplication!(vect2f,f32);
left_side_scalar_multiplication!(vect2d,f64);

impl<T: Float> ops::Neg for vect2<T>
{
    type Output = vect2<T>;
    fn neg(self) -> vect2<T> {
        vect2::<T> {x: -self.x, y: -self.y}
    }
}

impl<T: Float> ops::Div<T> for vect2<T>
{
    type Output = vect2<T>;
    fn div(self, _rhs: T) -> vect2<T> {
        vect2::<T> {x: self.x / _rhs, y: self.y / _rhs}
    }
}

impl<T: Float> vect2<T> {
    /// Constructs a 2D vector from components.
    pub fn new(x: T, y: T) -> vect2<T> {
        vect2::<T> { x, y}
    }

    /// Returns the squared magnitude (dot product with itself).
    pub fn square(self) -> T {
        self * self
    }

    /// Returns the Euclidean norm (length).
    pub fn norm(self) -> T {
        self.square().sqrt()
    }

    /// Returns the unit (normalized) vector.
    pub fn unit(self) -> vect2<T> {
        self * self.norm().recip()
    }

    /// Returns a unit vector perpendicular to `self` (90-degree counter-clockwise rotation).
    pub fn across(self) -> vect2<T> {
        vect2::<T>::new(-self.y, self.x)
    }

    /// Returns the 2D cross product (determinant): `self.x * rhs.y - self.y * rhs.x`.
    pub fn cross(self, rhs: vect2<T>) -> T {
        self.x * rhs.y - self.y * rhs.x
    }

    /// Returns true if all components are finite (not NaN or infinity).
    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite()
    }

    /// Converts all components from degrees to radians.
    pub fn deg2rad(self) -> vect2<T> {
        vect2::<T>::new(deg2rad(self.x), deg2rad(self.y))
    }

    /// Converts all components from radians to degrees.
    pub fn rad2deg(self) -> vect2<T> {
        vect2::<T>::new(rad2deg(self.x), rad2deg(self.y))
    }

    /// Constructs a `vect2<T>` by converting from a `vect2<K>` of a different float type.
    pub fn from<K: Float>(v: vect2<K>) -> vect2<T> {
        vect2::<T>::new(T::from(v.x).unwrap(), T::from(v.y).unwrap())
    }

    /// Converts this vector to a `vect2<K>` of a different float type.
    pub fn cast<K: Float>(self) -> vect2<K> {
        vect2::<K>::new(K::from(self.x).unwrap(), K::from(self.y).unwrap())
    }

    /// Logs an error if this vector is not approximately unit length.
    pub fn assert_unit_length(self) {
        if (self.square() - T::one()).abs() > T::from(1e5).unwrap() * T::epsilon() {
            error!("arael::vect: not unit length, norm = {:?}, error = {:?}", self.norm(), T::one() - self.norm());
        }
    }

    /// Computes element-wise sin and cos. Returns `(sin_vec, cos_vec)`.
    pub fn sincos(self) -> (vect2<T>, vect2<T>) {
        let (sin_x, cos_x) = self.x.sin_cos();
        let (sin_y, cos_y) = self.y.sin_cos();
        (vect2::<T>::new(sin_x, sin_y), vect2::<T>::new(cos_x, cos_y))
    }
}

impl<T: Float> Similar for vect2<T> {
    fn similar(self, other: vect2<T>) -> bool {
        (self - other).norm() < T::from(10).unwrap() * (self.norm() + other.norm() + T::epsilon()) * T::epsilon()
    }
}

impl<T: Float> ops::Index<usize> for vect2<T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        match index {
            0 => &self.x,
            1 => &self.y,
            _ => panic!("arael::vect: index {} out of bounds", index)
        }
    }
}

impl<T: Float> ops::IndexMut<usize> for vect2<T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        match index {
            0 => &mut self.x,
            1 => &mut self.y,
            _ => panic!("arael::vect: index {} out of bounds", index)
        }
    }
}

// Re-export symbolic companion types from arael-sym
pub use arael_sym::vect3sym;
pub use arael_sym::vect2sym;

#[cfg(test)]
mod tests {
    use super::*;

    // compare two vectors taking numerical noise into account
    fn equal<O: Similar>(a: O, b: O) -> bool {
        a.similar(b)
    }

    #[test]
    fn test() {
        let v1 = vect3d::new(2.0, 3.0, 6.0);
        let v2 = vect3d::new(5.0, 3.0, 4.0);
        // see if our equal actually works
        assert!(!equal(v1, v2));
        // construction method
        assert!(equal(v1, vect3d {x: 2.0, y: 3.0, z: 6.0}));
        // individual components
        assert_eq!(v1.x, 2.0); assert_eq!(v1.y, 3.0); assert_eq!(v1.z, 6.0);
        assert_eq!(v1[0], 2.0); assert_eq!(v1[1], 3.0); assert_eq!(v1[2], 6.0);
        // square
        assert_eq!(v1.square(), 49.0);
        // norm
        assert_eq!(v1.norm(), 7.0);
        // neg
        assert!(equal(-v1, vect3d::new(-2.0, -3.0, -6.0)));
        // multiplication by scalar
        assert!(equal(2.0*v1, vect3d::new(4.0, 6.0, 12.0)));
        assert!(equal(v1*2.0, vect3d::new(4.0, 6.0, 12.0)));
        // across sanity
        assert_eq!(v1.across() * v1, 0.0);
        // unit
        assert!((v1.unit().norm() - 1.0).abs() < 10.0 * f64::EPSILON);
        assert!(equal(v1.unit(), vect3d::new(2.0 / 7.0, 3.0 / 7.0, 6.0 / 7.0)));
        // cast sanity
        assert!(v1.cast::<f32>().cast::<f64>().similar(v1));
        // adding
        assert!(equal(v1 + v2, vect3d::new(7.0, 6.0, 10.0)));
        // substracting
        assert!(equal(v1 - v2, vect3d::new(-3.0, 0.0, 2.0)));
        // dot product
        assert_eq!(v1 * v1, 49.0);
        assert_eq!(v1 * v2, 43.0);
        // cross product
        assert!(equal(v1 % v2, vect3d::new(-6.0, 22.0, -9.0)));
        assert!(equal(v1 % (5.0*v1), vect3d::new(0.0, 0.0, 0.0)));
        // division by scalar
        assert!(equal(v1 / 2.0, vect3d::new(1.0, 1.5, 3.0)));
    }

    #[test]
    fn vect2_div_and_cross() {
        let a = vect2d::new(3.0, -4.0);
        let b = vect2d::new(1.0, 2.0);
        assert!(equal(a / 2.0, vect2d::new(1.5, -2.0)));
        // 2D cross product (determinant): 3*2 - (-4)*1 = 10
        assert_eq!(a.cross(b), 10.0);
        assert_eq!(b.cross(a), -10.0);
        // cross with the 90-degree rotation of self equals the square
        assert_eq!(a.cross(a.across()), a.square());
    }

    #[test]
    fn test_vect2() {
        let v1 = vect2d::new(3.0, 4.0);
        let v2 = vect2d::new(5.0, 3.0);
        // see if our equal actually works
        assert!(!equal(v1, v2));
        // construction method
        assert!(equal(v1, vect2d {x: 3.0, y: 4.0}));
        // individual components
        assert_eq!(v1.x, 3.0); assert_eq!(v1.y, 4.0);
        assert_eq!(v1[0], 3.0); assert_eq!(v1[1], 4.0);
        // square
        assert_eq!(v1.square(), 25.0);
        // norm
        assert_eq!(v1.norm(), 5.0);
        // neg
        assert!(equal(-v1, vect2::new(-3.0, -4.0)));
        // multiplication by scalar
        assert!(equal(2.0*v1, vect2d::new(6.0, 8.0)));
        assert!(equal(v1*2.0, vect2d::new(6.0, 8.0)));
        // across sanity
        assert_eq!(v1.across() * v1, 0.0);
        // unit
        assert!((v1.unit().norm() - 1.0).abs() < 10.0 * f64::EPSILON);
        assert!(equal(v1.unit(), vect2d::new(3.0 / 5.0, 4.0 / 5.0)));
        // cast sanity
        assert!(v1.cast::<f32>().cast::<f64>().similar(v1));
        // adding
        assert!(equal(v1 + v2, vect2d::new(8.0, 7.0)));
        // substracting
        assert!(equal(v1 - v2, vect2d::new(-2.0, 1.0)));
        // dot product
        assert_eq!(v1 * v1, 25.0);
        assert_eq!(v1 * v2, 27.0);
    }

    #[test]
    fn test_vect3_unit_is_unit_length() {
        let v = vect3d::new(3.0, -4.0, 12.0);
        let u = v.unit();
        assert!((u.square() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_vect3_cross_product_perpendicular() {
        let a = vect3d::new(1.0, 0.0, 0.0);
        let b = vect3d::new(0.0, 1.0, 0.0);
        let c = a % b;
        assert!(c.similar(vect3d::new(0.0, 0.0, 1.0)));
        // cross product is perpendicular to both
        assert!((c * a).abs() < 1e-14);
        assert!((c * b).abs() < 1e-14);
    }

    #[test]
    fn test_vect3_is_finite() {
        assert!(vect3d::new(1.0, 2.0, 3.0).is_finite());
        assert!(!vect3d::new(f64::INFINITY, 0.0, 0.0).is_finite());
        assert!(!vect3d::new(0.0, f64::NAN, 0.0).is_finite());
    }

    #[test]
    fn test_vect3_deg_rad_roundtrip() {
        let v = vect3d::new(30.0, 45.0, 90.0);
        assert!(v.deg2rad().rad2deg().similar(v));
    }
}

