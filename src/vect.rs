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
pub use arael_sym::vectsym;

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

// ---------------------------------------------------------------------------
// vect<T, N> -- fixed-size N-dimensional vector
// ---------------------------------------------------------------------------

/// N-dimensional vector over `[T; N]` -- the generic sibling of
/// [`vect2`] / [`vect3`]. Addition, subtraction, negation, scalar
/// multiplication, dot product (`*` operator), `Index<usize>`. The
/// dimension lives in the const generic; `From` converts to and from
/// the fixed types and nalgebra's `SVector`.
#[derive(Clone, Copy, PartialEq)]
pub struct vect<T: Float, const N: usize> {
    pub e: [T; N],
}

// serde has no blanket impls for const-generic arrays; serialize as a
// sequence of exactly N components.
impl<T: Float + serde::Serialize, const N: usize> serde::Serialize for vect<T, N> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut seq = s.serialize_seq(Some(N))?;
        for v in &self.e { seq.serialize_element(v)?; }
        seq.end()
    }
}

impl<'de, T, const N: usize> serde::Deserialize<'de> for vect<T, N>
where
    T: Float + serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct V<T, const N: usize>(std::marker::PhantomData<T>);
        impl<'de, T, const N: usize> serde::de::Visitor<'de> for V<T, N>
        where
            T: Float + serde::Deserialize<'de>,
        {
            type Value = vect<T, N>;
            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "a sequence of {} numbers", N)
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self, mut seq: A,
            ) -> Result<Self::Value, A::Error> {
                let mut e = [T::zero(); N];
                for (i, slot) in e.iter_mut().enumerate() {
                    *slot = seq.next_element()?.ok_or_else(|| {
                        serde::de::Error::invalid_length(i, &self)
                    })?;
                }
                Ok(vect { e })
            }
        }
        d.deserialize_seq(V::<T, N>(std::marker::PhantomData))
    }
}

/// N-dimensional vector with f32 components.
pub type vectf<const N: usize> = vect<f32, N>;
/// N-dimensional vector with f64 components.
pub type vectd<const N: usize> = vect<f64, N>;

impl<T: Float, const N: usize> vect<T, N> {
    /// Create from components.
    pub fn new(e: [T; N]) -> Self { vect { e } }
    /// Create from an array (same as [`new`](Self::new), named for
    /// symmetry with the matrix constructor).
    pub fn from_array(e: [T; N]) -> Self { vect { e } }
    /// The zero vector.
    pub fn zeros() -> Self { vect { e: [T::zero(); N] } }
    /// Number of components.
    pub const fn len(&self) -> usize { N }
    /// True when N == 0.
    pub const fn is_empty(&self) -> bool { N == 0 }
    /// Squared Euclidean norm.
    pub fn norm_squared(self) -> T {
        let mut s = T::zero();
        for i in 0..N { s = s + self.e[i] * self.e[i]; }
        s
    }
    /// Euclidean norm.
    pub fn norm(self) -> T { self.norm_squared().sqrt() }
    /// Component-wise cast to another float type.
    pub fn cast<U: Float>(self) -> vect<U, N> {
        vect { e: std::array::from_fn(|i| U::from(self.e[i]).unwrap()) }
    }
}

impl<T: Float, const N: usize> Default for vect<T, N> {
    fn default() -> Self { Self::zeros() }
}

impl<T: Float, const N: usize> fmt::Debug for vect<T, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_list().entries(self.e.iter()).finish()
    }
}

impl<T: Float, const N: usize> ops::Index<usize> for vect<T, N> {
    type Output = T;
    fn index(&self, i: usize) -> &T { &self.e[i] }
}

impl<T: Float, const N: usize> ops::IndexMut<usize> for vect<T, N> {
    fn index_mut(&mut self, i: usize) -> &mut T { &mut self.e[i] }
}

impl<T: Float, const N: usize> ops::Add for vect<T, N> {
    type Output = vect<T, N>;
    fn add(self, rhs: Self) -> Self {
        vect { e: std::array::from_fn(|i| self.e[i] + rhs.e[i]) }
    }
}

impl<T: Float, const N: usize> ops::Sub for vect<T, N> {
    type Output = vect<T, N>;
    fn sub(self, rhs: Self) -> Self {
        vect { e: std::array::from_fn(|i| self.e[i] - rhs.e[i]) }
    }
}

impl<T: Float, const N: usize> ops::Neg for vect<T, N> {
    type Output = vect<T, N>;
    fn neg(self) -> Self {
        vect { e: std::array::from_fn(|i| -self.e[i]) }
    }
}

/// Dot product.
impl<T: Float, const N: usize> ops::Mul for vect<T, N> {
    type Output = T;
    fn mul(self, rhs: Self) -> T {
        let mut s = T::zero();
        for i in 0..N { s = s + self.e[i] * rhs.e[i]; }
        s
    }
}

impl<T: Float, const N: usize> ops::Mul<T> for vect<T, N> {
    type Output = vect<T, N>;
    fn mul(self, s: T) -> Self {
        vect { e: std::array::from_fn(|i| self.e[i] * s) }
    }
}

impl<T: Float, const N: usize> ops::Div<T> for vect<T, N> {
    type Output = vect<T, N>;
    fn div(self, s: T) -> Self {
        vect { e: std::array::from_fn(|i| self.e[i] / s) }
    }
}

impl<const N: usize> ops::Mul<vect<f32, N>> for f32 {
    type Output = vect<f32, N>;
    fn mul(self, rhs: vect<f32, N>) -> vect<f32, N> { rhs * self }
}

impl<const N: usize> ops::Mul<vect<f64, N>> for f64 {
    type Output = vect<f64, N>;
    fn mul(self, rhs: vect<f64, N>) -> vect<f64, N> { rhs * self }
}

// Conversions to and from the fixed types.
impl<T: Float> From<vect2<T>> for vect<T, 2> {
    fn from(v: vect2<T>) -> Self { vect { e: [v.x, v.y] } }
}
impl<T: Float> From<vect<T, 2>> for vect2<T> {
    fn from(v: vect<T, 2>) -> Self { vect2 { x: v.e[0], y: v.e[1] } }
}
impl<T: Float> From<vect3<T>> for vect<T, 3> {
    fn from(v: vect3<T>) -> Self { vect { e: [v.x, v.y, v.z] } }
}
impl<T: Float> From<vect<T, 3>> for vect3<T> {
    fn from(v: vect<T, 3>) -> Self { vect3 { x: v.e[0], y: v.e[1], z: v.e[2] } }
}

// Mixed dot products with the fixed types.
impl<T: Float> ops::Mul<vect2<T>> for vect<T, 2> {
    type Output = T;
    fn mul(self, rhs: vect2<T>) -> T { self.e[0] * rhs.x + self.e[1] * rhs.y }
}
impl<T: Float> ops::Mul<vect<T, 2>> for vect2<T> {
    type Output = T;
    fn mul(self, rhs: vect<T, 2>) -> T { self.x * rhs.e[0] + self.y * rhs.e[1] }
}
impl<T: Float> ops::Mul<vect3<T>> for vect<T, 3> {
    type Output = T;
    fn mul(self, rhs: vect3<T>) -> T {
        self.e[0] * rhs.x + self.e[1] * rhs.y + self.e[2] * rhs.z
    }
}
impl<T: Float> ops::Mul<vect<T, 3>> for vect3<T> {
    type Output = T;
    fn mul(self, rhs: vect<T, 3>) -> T {
        self.x * rhs.e[0] + self.y * rhs.e[1] + self.z * rhs.e[2]
    }
}

// Conversions to and from nalgebra, for callers doing heavy math on
// their own side of the model boundary.
impl<T: Float + nalgebra::Scalar, const N: usize> From<vect<T, N>>
    for nalgebra::SVector<T, N>
{
    fn from(v: vect<T, N>) -> Self { nalgebra::SVector::from(v.e) }
}
impl<T: Float + nalgebra::Scalar, const N: usize> From<nalgebra::SVector<T, N>>
    for vect<T, N>
{
    fn from(v: nalgebra::SVector<T, N>) -> Self {
        vect { e: std::array::from_fn(|i| v[i]) }
    }
}

