//! 2x2 and 3x3 matrix types with rotation, transpose, and linear algebra operations.

#![allow(non_camel_case_types)]

use std::ops;
use std::fmt;
use crate::vect::vect2;
use crate::vect::vect3;
use crate::vect::Similar;
use crate::utils::left_side_scalar_multiplication;
use crate::utils::Float;
use crate::utils::atan2;

/// 3x3 matrix stored as 3 row vectors.
///
/// Supports addition, subtraction, negation, scalar multiplication, matrix-matrix
/// multiplication, and matrix-vector multiplication. Indexable by `usize` to get rows.
#[derive(Clone, Copy)]
pub struct matrix3<T : Float>
{
    pub rows : [vect3<T>; 3]
}

/// 3x3 matrix with f32 elements.
pub type matrix3f = matrix3<f32>;
/// 3x3 matrix with f64 elements.
pub type matrix3d = matrix3<f64>;

impl<T: Float> ops::Index<usize> for matrix3<T> {
    type Output = vect3<T>;
    fn index(&self, index: usize) -> &vect3<T> {
        &self.rows[index]
    }
}

impl<T: Float> ops::IndexMut<usize> for matrix3<T> {
    fn index_mut(&mut self, index: usize) -> &mut vect3<T> {
        &mut self.rows[index]
    }
}

impl<T : Float> fmt::Debug for matrix3<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}; {:?}; {:?}]", self[0], self[1], self[2])
    }
}

impl<T: Float> ops::Add<matrix3<T>> for matrix3<T>
{
    type Output = matrix3<T>;
    fn add(self, _rhs: matrix3<T>) -> matrix3<T> {
        matrix3::<T>::from_rows(self[0] + _rhs[0], self[1] + _rhs[1], self[2] + _rhs[2])
    }
}

impl<T: Float> ops::Sub<matrix3<T>> for matrix3<T>
{
    type Output = matrix3<T>;
    fn sub(self, _rhs: matrix3<T>) -> matrix3<T> {
        matrix3::<T>::from_rows(self[0] - _rhs[0], self[1] - _rhs[1], self[2] - _rhs[2])
    }
}

impl<T: Float> ops::Neg for matrix3<T>
{
    type Output = matrix3<T>;
    fn neg(self) -> matrix3<T> {
        matrix3::<T>::from_rows(-self[0], -self[1], -self[2])
    }
}

impl<T: Float> ops::Mul<vect3<T>> for matrix3<T>
{
    type Output = vect3<T>;
    fn mul(self, _rhs: vect3<T>) -> vect3<T> {
        vect3::<T>::new(self[0] * _rhs, self[1] * _rhs, self[2] * _rhs)
    }
}

impl<T: Float> ops::Mul<matrix3<T>> for vect3<T>
{
    type Output = vect3<T>;
    fn mul(self, _rhs: matrix3<T>) -> vect3<T> {
        vect3::<T>::new(self * _rhs.col(0), self * _rhs.col(1), self * _rhs.col(2))
    }
}

impl<T: Float> ops::Mul<matrix3<T>> for matrix3<T>
{
    type Output = matrix3<T>;
    fn mul(self, _rhs: matrix3<T>) -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::new(
                self.row(0) * _rhs.col(0),
                self.row(0) * _rhs.col(1),
                self.row(0) * _rhs.col(2)
            ),
            vect3::<T>::new(
                self.row(1) * _rhs.col(0),
                self.row(1) * _rhs.col(1),
                self.row(1) * _rhs.col(2)
            ),
            vect3::<T>::new(
                self.row(2) * _rhs.col(0),
                self.row(2) * _rhs.col(1),
                self.row(2) * _rhs.col(2)
            )
        )
    }
}

impl<T: Float> ops::Mul<T> for matrix3<T>
{
    type Output = matrix3<T>;
    fn mul(self, _rhs: T) -> matrix3<T> {
        matrix3::<T>::from_rows(
            self[0] * _rhs,
            self[1] * _rhs,
            self[2] * _rhs
        )
    }
}

left_side_scalar_multiplication!(matrix3f,f32);
left_side_scalar_multiplication!(matrix3d,f64);

impl<T: Float> matrix3<T>
{
    /// Constructs a matrix from three row vectors.
    pub fn from_rows(row0 : vect3<T>, row1 : vect3<T>, row2 : vect3<T>) -> matrix3<T> {
        matrix3::<T> { rows: [row0, row1, row2] }
    }

    /// Constructs a matrix from three column vectors.
    pub fn from_cols(col0 : vect3<T>, col1 : vect3<T>, col2 : vect3<T>) -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::new(col0.x, col1.x, col2.x),
            vect3::<T>::new(col0.y, col1.y, col2.y),
            vect3::<T>::new(col0.z, col1.z, col2.z),
        )
    }

    /// Constructs a matrix from 9 individual elements in row-major order.
    pub fn from_elements(a00: T, a01: T, a02: T, a10: T, a11: T, a12: T, a20: T, a21: T, a22: T) -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::new(a00, a01, a02),
            vect3::<T>::new(a10, a11, a12),
            vect3::<T>::new(a20, a21, a22)
        )
    }

    /// Constructs a matrix from a 9-element array in row-major order.
    pub fn from_slice(slice: &[T; 9]) -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::new(slice[0], slice[1], slice[2]),
            vect3::<T>::new(slice[3], slice[4], slice[5]),
            vect3::<T>::new(slice[6], slice[7], slice[8])
        )
    }

    /// Returns the 3x3 zero matrix.
    pub fn zero_matrix() -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::new(T::zero(), T::zero(), T::zero()),
            vect3::<T>::new(T::zero(), T::zero(), T::zero()),
            vect3::<T>::new(T::zero(), T::zero(), T::zero())
        )
    }

    /// Returns the 3x3 identity matrix.
    pub fn identity() -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::new( T::one(), T::zero(), T::zero()),
            vect3::<T>::new(T::zero(),  T::one(), T::zero()),
            vect3::<T>::new(T::zero(), T::zero(),  T::one())
        )
    }

    /// Returns the column at the given index as a vector.
    pub fn col(self, index: usize) -> vect3<T> {
        match index {
            0 => vect3::<T>::new(self[0].x, self[1].x, self[2].x),
            1 => vect3::<T>::new(self[0].y, self[1].y, self[2].y),
            2 => vect3::<T>::new(self[0].z, self[1].z, self[2].z),
            _ => panic!("arael::matrix3: column index {} out of bounds (0..3)", index)
        }
    }

    /// Returns the row at the given index as a vector.
    pub fn row(self, index: usize) -> vect3<T> {
        self[index]
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(self) -> matrix3<T> {
        matrix3::<T>::from_rows(self.col(0), self.col(1), self.col(2))
    }

    /// Creates a rotation matrix into frame where unit vector n forms the z axis
    pub fn null_space(n: vect3<T>) -> matrix3<T> {
        n.assert_unit_length();
        let z = n;
        let x = n.across();
        let y = z % x;
        matrix3::<T>::from_cols(x, y, z)
    }

    /// Builds a rotation matrix from Euler angles (x=roll, y=pitch, z=yaw).
    pub fn rotation_from_euler_angles(euler_angles: vect3<T>) -> matrix3<T> {
        let (sin_euler_angles, cos_euler_angles) = euler_angles.sincos();
        matrix3::<T>::rotation_from_euler_angles_sincos(sin_euler_angles, cos_euler_angles)
    }

    /// Builds a rotation matrix from pre-computed sin/cos of Euler angles.
    pub fn rotation_from_euler_angles_sincos(sin_euler_angles: vect3<T>, cos_euler_angles: vect3<T>) -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::new(
                cos_euler_angles.y*cos_euler_angles.z,
                -cos_euler_angles.x*sin_euler_angles.z + cos_euler_angles.z*sin_euler_angles.x*sin_euler_angles.y,
                cos_euler_angles.x*cos_euler_angles.z*sin_euler_angles.y + sin_euler_angles.x*sin_euler_angles.z
            ),
            vect3::<T>::new(
                cos_euler_angles.y*sin_euler_angles.z,
                cos_euler_angles.x*cos_euler_angles.z + sin_euler_angles.x*sin_euler_angles.y*sin_euler_angles.z,
                cos_euler_angles.x*sin_euler_angles.y*sin_euler_angles.z - cos_euler_angles.z*sin_euler_angles.x
            ),
            vect3::<T>::new(
                -sin_euler_angles.y,
                cos_euler_angles.y*sin_euler_angles.x,
                cos_euler_angles.x*cos_euler_angles.y
            )
        )
    }

    /// Builds a rotation matrix from an axis (must be unit) and angle in radians.
    pub fn rotation_from_axis_angle(axis: vect3<T>, phi: T) -> matrix3<T> {
        matrix3::<T>::rotation_from_axis_angle_sincos(axis, phi.sin_cos())
    }

    /// Builds a rotation matrix from an axis and pre-computed (sin, cos) of the angle.
    pub fn rotation_from_axis_angle_sincos(axis: vect3<T>, (sin_phi, cos_phi): (T, T)) -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::new(
                cos_phi + axis.x * axis.x * (T::one() - cos_phi),
                axis.x * axis.y * (T::one() - cos_phi) - axis.z * sin_phi,
                axis.x * axis.z * (T::one() - cos_phi) + axis.y * sin_phi
            ),
            vect3::<T>::new(
                axis.y * axis.x * (T::one() - cos_phi) + axis.z * sin_phi,
                cos_phi + axis.y * axis.y * (T::one() - cos_phi),
                axis.y * axis.z * (T::one() - cos_phi) - axis.x * sin_phi
            ),
            vect3::<T>::new(
                axis.z * axis.x * (T::one() - cos_phi) - axis.y * sin_phi,
                axis.z * axis.y * (T::one() - cos_phi) + axis.x * sin_phi,
                cos_phi + axis.z * axis.z * (T::one() - cos_phi)
            )
        )
    }

    /// Extracts Euler angles (x=roll, y=pitch, z=yaw) from a rotation matrix.
    /// Diverges near |pitch| = pi/2 (gimbal lock).
    pub fn get_euler_angles(self) -> vect3<T> {
        let y = -self[2][0].asin();
        if y.abs() < T::pi() / T::from(2).unwrap() {
            vect3::<T>::new(
                atan2(self[2][1], self[2][2]),
                y,
                atan2(self[1][0], self[0][0])
            )
        } else {
            vect3::<T>::new(
                atan2(-self[2][1], -self[2][2]),
                y,
                atan2(-self[1][0], -self[0][0])
            )
        }
    }

    /// Constructs a `matrix3<T>` by converting from a `matrix3<K>` of a different float type.
    pub fn from<K: Float>(v: matrix3<K>) -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::from(v[0]),
            vect3::<T>::from(v[1]),
            vect3::<T>::from(v[2])
        )
    }

    /// Converts this matrix to a `matrix3<K>` of a different float type.
    pub fn cast<K: Float>(self) -> matrix3<K> {
        matrix3::<K>::from_rows(
            self[0].cast::<K>(),
            self[1].cast::<K>(),
            self[2].cast::<K>()
        )
    }

    /// Returns the determinant of the matrix.
    pub fn det(self) -> T {
        self[0][0] * self[1][1] * self[2][2] +
        self[0][2] * self[1][0] * self[2][1] +
        self[0][1] * self[1][2] * self[2][0] -
        self[0][0] * self[1][2] * self[2][1] -
        self[0][1] * self[1][0] * self[2][2] -
        self[0][2] * self[1][1] * self[2][0]
    }

    /// Symmetric eigendecomposition: self = R * diag(d) * R^T
    ///
    /// The matrix must be symmetric. Returns (R, d) where R is the rotation
    /// matrix of eigenvectors (columns) and d contains the eigenvalues.
    /// Eigenvalues are sorted in ascending order.
    pub fn symmetric_eigen(self) -> (matrix3<T>, vect3<T>) {
        // Convert to nalgebra for eigendecomposition
        let na_mat = nalgebra::Matrix3::new(
            self[0][0].to_f64().unwrap(), self[0][1].to_f64().unwrap(), self[0][2].to_f64().unwrap(),
            self[1][0].to_f64().unwrap(), self[1][1].to_f64().unwrap(), self[1][2].to_f64().unwrap(),
            self[2][0].to_f64().unwrap(), self[2][1].to_f64().unwrap(), self[2][2].to_f64().unwrap(),
        );
        let eigen = na_mat.symmetric_eigen();
        // eigen.eigenvalues: Vector3, eigen.eigenvectors: Matrix3 (columns are eigenvectors)

        // Sort by eigenvalue (ascending)
        let mut idx = [0usize, 1, 2];
        idx.sort_by(|&a, &b| eigen.eigenvalues[a].partial_cmp(&eigen.eigenvalues[b]).unwrap());

        let d = vect3::<T>::new(
            T::from(eigen.eigenvalues[idx[0]]).unwrap(),
            T::from(eigen.eigenvalues[idx[1]]).unwrap(),
            T::from(eigen.eigenvalues[idx[2]]).unwrap(),
        );
        let r = matrix3::<T>::from_cols(
            vect3::<T>::new(
                T::from(eigen.eigenvectors[(0, idx[0])]).unwrap(),
                T::from(eigen.eigenvectors[(1, idx[0])]).unwrap(),
                T::from(eigen.eigenvectors[(2, idx[0])]).unwrap(),
            ),
            vect3::<T>::new(
                T::from(eigen.eigenvectors[(0, idx[1])]).unwrap(),
                T::from(eigen.eigenvectors[(1, idx[1])]).unwrap(),
                T::from(eigen.eigenvectors[(2, idx[1])]).unwrap(),
            ),
            vect3::<T>::new(
                T::from(eigen.eigenvectors[(0, idx[2])]).unwrap(),
                T::from(eigen.eigenvectors[(1, idx[2])]).unwrap(),
                T::from(eigen.eigenvectors[(2, idx[2])]).unwrap(),
            ),
        );
        (r, d)
    }
}

impl<T: Float> Similar for matrix3<T> {
    fn similar(self, other: matrix3<T>) -> bool {
        self[0].similar(other[0]) && self[1].similar(other[1]) && self[2].similar(other[2])
    }
}

/// 2x2 matrix stored as 2 row vectors.
///
/// Supports addition, subtraction, negation, scalar multiplication, matrix-matrix
/// multiplication, and matrix-vector multiplication. Indexable by `usize` to get rows.
#[derive(Clone, Copy)]
pub struct matrix2<T : Float>
{
    pub rows : [vect2<T>; 2]
}

/// 2x2 matrix with f32 elements.
pub type matrix2f = matrix2<f32>;
/// 2x2 matrix with f64 elements.
pub type matrix2d = matrix2<f64>;

impl<T: Float> ops::Index<usize> for matrix2<T> {
    type Output = vect2<T>;
    fn index(&self, index: usize) -> &vect2<T> {
        &self.rows[index]
    }
}

impl<T: Float> ops::IndexMut<usize> for matrix2<T> {
    fn index_mut(&mut self, index: usize) -> &mut vect2<T> {
        &mut self.rows[index]
    }
}

impl<T : Float> fmt::Debug for matrix2<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{:?}; {:?}]", self[0], self[1])
    }
}

impl<T: Float> ops::Add<matrix2<T>> for matrix2<T>
{
    type Output = matrix2<T>;
    fn add(self, _rhs: matrix2<T>) -> matrix2<T> {
        matrix2::<T>::from_rows(self[0] + _rhs[0], self[1] + _rhs[1])
    }
}

impl<T: Float> ops::Sub<matrix2<T>> for matrix2<T>
{
    type Output = matrix2<T>;
    fn sub(self, _rhs: matrix2<T>) -> matrix2<T> {
        matrix2::<T>::from_rows(self[0] - _rhs[0], self[1] - _rhs[1])
    }
}

impl<T: Float> ops::Neg for matrix2<T>
{
    type Output = matrix2<T>;
    fn neg(self) -> matrix2<T> {
        matrix2::<T>::from_rows(-self[0], -self[1])
    }
}

impl<T: Float> ops::Mul<vect2<T>> for matrix2<T>
{
    type Output = vect2<T>;
    fn mul(self, _rhs: vect2<T>) -> vect2<T> {
        vect2::<T>::new(self[0] * _rhs, self[1] * _rhs)
    }
}

impl<T: Float> ops::Mul<matrix2<T>> for vect2<T>
{
    type Output = vect2<T>;
    fn mul(self, _rhs: matrix2<T>) -> vect2<T> {
        vect2::<T>::new(self * _rhs.col(0), self * _rhs.col(1))
    }
}

impl<T: Float> ops::Mul<matrix2<T>> for matrix2<T>
{
    type Output = matrix2<T>;
    fn mul(self, _rhs: matrix2<T>) -> matrix2<T> {
        matrix2::<T>::from_rows(
            vect2::<T>::new(
                self.row(0) * _rhs.col(0),
                self.row(0) * _rhs.col(1)
            ),
            vect2::<T>::new(
                self.row(1) * _rhs.col(0),
                self.row(1) * _rhs.col(1)
            )
        )
    }
}

impl<T: Float> ops::Mul<T> for matrix2<T>
{
    type Output = matrix2<T>;
    fn mul(self, _rhs: T) -> matrix2<T> {
        matrix2::<T>::from_rows(
            self[0] * _rhs,
            self[1] * _rhs
        )
    }
}

left_side_scalar_multiplication!(matrix2f,f32);
left_side_scalar_multiplication!(matrix2d,f64);

impl<T: Float> matrix2<T>
{
    /// Constructs a matrix from two row vectors.
    pub fn from_rows(row0 : vect2<T>, row1 : vect2<T>) -> matrix2<T> {
        matrix2::<T> { rows: [row0, row1] }
    }

    /// Constructs a matrix from two column vectors.
    pub fn from_cols(col0 : vect2<T>, col1 : vect2<T>) -> matrix2<T> {
        matrix2::<T>::from_rows(
            vect2::<T>::new(col0.x, col1.x),
            vect2::<T>::new(col0.y, col1.y)
        )
    }

    /// Constructs a matrix from 4 individual elements in row-major order.
    pub fn from_elements(a00: T, a01: T, a10: T, a11: T) -> matrix2<T> {
        matrix2::<T>::from_rows(
            vect2::<T>::new(a00, a01),
            vect2::<T>::new(a10, a11),
        )
    }

    /// Constructs a matrix from a 4-element array in row-major order.
    pub fn from_slice(slice: &[T; 4]) -> matrix2<T> {
        matrix2::<T>::from_rows(
            vect2::<T>::new(slice[0], slice[1]),
            vect2::<T>::new(slice[2], slice[3])
        )
    }

    /// Returns the 2x2 zero matrix.
    pub fn zero_matrix() -> matrix2<T> {
        matrix2::<T>::from_rows(
            vect2::<T>::new(T::zero(), T::zero()),
            vect2::<T>::new(T::zero(), T::zero())
        )
    }

    /// Returns the 2x2 identity matrix.
    pub fn identity() -> matrix2<T> {
        matrix2::<T>::from_rows(
            vect2::<T>::new( T::one(), T::zero()),
            vect2::<T>::new(T::zero(),  T::one())
        )
    }

    /// Returns the column at the given index as a vector.
    pub fn col(self, index: usize) -> vect2<T> {
        match index {
            0 => vect2::<T>::new(self[0].x, self[1].x),
            1 => vect2::<T>::new(self[0].y, self[1].y),
            _ => panic!("arael::matrix2: column index {} out of bounds (0..2)", index)
        }
    }

    /// Returns the row at the given index as a vector.
    pub fn row(self, index: usize) -> vect2<T> {
        self[index]
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(self) -> matrix2<T> {
        matrix2::<T>::from_rows(self.col(0), self.col(1))
    }

    /// Builds a 2D rotation matrix for the given angle in radians.
    pub fn rotation(angle: T) -> matrix2<T> {
        let (sin_angle, cos_angle) = angle.sin_cos();
        matrix2::<T>::rotation_from_sincos(sin_angle, cos_angle)
    }

    /// Builds a 2D rotation matrix from pre-computed sin and cos values.
    pub fn rotation_from_sincos(sin_angle: T, cos_angle: T) -> matrix2<T> {
        matrix2::<T>::from_rows(
            vect2::<T>::new(cos_angle, -sin_angle),
            vect2::<T>::new(sin_angle, cos_angle)
        )
    }

    /// Extracts the rotation angle from a 2D rotation matrix.
    pub fn get_rotation_angle(self) -> T {
        crate::utils::atan2(self[1][0], self[0][0])
    }

    /// Constructs a `matrix2<T>` by converting from a `matrix2<K>` of a different float type.
    pub fn from<K: Float>(v: matrix2<K>) -> matrix2<T> {
        matrix2::<T>::from_rows(
            vect2::<T>::from(v[0]),
            vect2::<T>::from(v[1])
        )
    }

    /// Converts this matrix to a `matrix2<K>` of a different float type.
    pub fn cast<K: Float>(self) -> matrix2<K> {
        matrix2::<K>::from_rows(
            self[0].cast::<K>(),
            self[1].cast::<K>()
        )
    }

    /// Returns the determinant of the matrix.
    pub fn det(self) -> T {
        self[0][0] * self[1][1] - self[0][1] * self[1][0]
    }
}

impl<T: Float> Similar for matrix2<T> {
    fn similar(self, other: matrix2<T>) -> bool {
        self[0].similar(other[0]) && self[1].similar(other[1])
    }
}

// Re-export symbolic companion types from arael-sym
pub use arael_sym::matrix3sym;
pub use arael_sym::matrix2sym;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vect::{vect2d, vect3d};
    use crate::quatern::quaternd;

    // compare two vectors taking numerical noise into account
    fn equal<O: Similar>(a: O, b: O) -> bool {
        a.similar(b)
    }

    #[test]
    fn test() {
        let a = matrix3d::from_rows(
            vect3d::new(2.0, 1.0, 3.0),
            vect3d::new(7.0, 5.0, 6.0),
            vect3d::new(-5.0, 0.0, 1.0)
        );
        let b = matrix3d::from_rows(
            vect3d::new(-1.0, 3.0, 1.0),
            vect3d::new(2.0, 2.0, 3.0),
            vect3d::new(1.0, 5.0, 6.0)
        );
        let v = vect3d::new(2.0, -1.0, 5.0);
        // sanity of testing function
        assert!(equal(a, a));
        assert!(!equal(a, b));
        // construction methods
        assert!(equal(a, matrix3d::from_elements(2.0, 1.0, 3.0, 7.0, 5.0, 6.0, -5.0, 0.0, 1.0)));
        assert!(equal(a, matrix3d::from_slice(&[2.0, 1.0, 3.0, 7.0, 5.0, 6.0, -5.0, 0.0, 1.0])));
        assert!(equal(a, matrix3d::from_cols(vect3d::new(2.0, 7.0, -5.0), vect3d::new(1.0, 5.0, 0.0), vect3d::new(3.0, 6.0, 1.0))));
        assert!(equal(matrix3d::identity(), matrix3d::from_elements(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)));
        assert!(equal(matrix3d::zero_matrix(), matrix3d::from_elements(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)));
        // accessors
        assert_eq!(a[0][0], 2.0); assert_eq!(a[0][1], 1.0); assert_eq!(a[0][2], 3.0);
        assert_eq!(a[1][0], 7.0); assert_eq!(a[1][1], 5.0); assert_eq!(a[1][2], 6.0);
        assert_eq!(a[2][0],-5.0); assert_eq!(a[2][1], 0.0); assert_eq!(a[2][2], 1.0);
        assert!(a.row(0).similar(vect3d::new(2.0, 1.0, 3.0)));
        assert!(a.row(1).similar(vect3d::new(7.0, 5.0, 6.0)));
        assert!(a.row(2).similar(vect3d::new(-5.0, 0.0, 1.0)));
        assert!(a.col(0).similar(vect3d::new(2.0, 7.0, -5.0)));
        assert!(a.col(1).similar(vect3d::new(1.0, 5.0, 0.0)));
        assert!(a.col(2).similar(vect3d::new(3.0, 6.0, 1.0)));
        // neg and adding and zero_matrix sanity
        assert!(equal(-a + a, matrix3d::zero_matrix()));
        // determinant
        assert_eq!(a.det(), 48.0);
        // transpose
        assert!(equal(a.transpose(), matrix3d::from_elements(2.0, 7.0, -5.0, 1.0, 5.0, 0.0, 3.0, 6.0, 1.0)));
        // scalar multiplication
        assert!(equal(2.0 * a, matrix3d::from_elements(4.0, 2.0, 6.0, 14.0, 10.0, 12.0, -10.0, 0.0, 2.0)));
        assert!(equal(a * 2.0, matrix3d::from_elements(4.0, 2.0, 6.0, 14.0, 10.0, 12.0, -10.0, 0.0, 2.0)));
        // cast sanity
        assert!(a.cast::<f32>().cast::<f64>().similar(a));
        // adding
        assert!(equal(a + b, matrix3d::from_elements(1.0, 4.0, 4.0, 9.0, 7.0, 9.0, -4.0, 5.0, 7.0)));
        // substracting
        assert!(equal(a - b, matrix3d::from_elements(3.0, -2.0, 2.0, 5.0, 3.0, 3.0, -6.0, -5.0, -5.0)));
        // multiplication
        assert!(equal(a * b, matrix3d::from_elements(3.0, 23.0, 23.0, 9.0, 61.0, 58.0, 6.0, -10.0, 1.0)));
        assert!(equal(matrix3d::identity() * a * matrix3d::identity(), a));
        // multiplication with a vector
        assert!((a * v).similar(vect3d::new(18.0, 39.0, -5.0)));
        // null_space sanity
        assert!((matrix3d::null_space(v.unit()) * vect3d::new(0.0, 0.0, 1.0)).similar(v.unit()));
        assert!((matrix3d::null_space(v.unit()).det() - 1.0).abs() < f64::EPSILON);
        // euler angles rotation back to angles sanity
        let ea = vect3d::new(1.0, 0.1, -2.4);
        assert!(matrix3d::rotation_from_euler_angles(ea).get_euler_angles().similar(ea));
        // rotation from axis angle
        let axis = vect3d::new(1.0, 2.0, 3.0).unit();
        let angle = 1.2;
        assert!((matrix3d::rotation_from_axis_angle(axis, angle) * v).similar(quaternd::from_axis_angle(axis, angle).rotate(v)));
    }

    #[test]
    fn test_matrix2() {
        let r = matrix2d::rotation(f64::half_pi());
        assert!(equal(r * vect2d::new(1.0, 0.0), vect2d::new(0.0, 1.0)));
    }

    #[test]
    fn test_matrix2_rotation_angle() {
        let angle = 1.23;
        let r = matrix2d::rotation(angle);
        assert!((r.get_rotation_angle() - angle).abs() < 1e-12);
    }

    #[test]
    fn test_matrix3_identity_det() {
        assert_eq!(matrix3d::identity().det(), 1.0);
        assert_eq!(matrix3d::zero_matrix().det(), 0.0);
    }

    #[test]
    fn test_matrix3_transpose_twice() {
        let a = matrix3d::from_elements(2.0, 1.0, 3.0, 7.0, 5.0, 6.0, -5.0, 0.0, 1.0);
        assert!(equal(a.transpose().transpose(), a));
    }

    #[test]
    fn test_rotation_matrix_is_orthogonal() {
        let ea = vect3d::new(0.7, -0.3, 1.5);
        let r = matrix3d::rotation_from_euler_angles(ea);
        // R * R^T = I for orthogonal matrix
        assert!((r * r.transpose()).similar(matrix3d::identity()));
        // det = 1 for proper rotation
        assert!((r.det() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_symmetric_eigen() {
        // Build a symmetric matrix K = R * diag(d) * R^T with known values
        let ea = vect3d::new(0.3, -0.5, 1.2);
        let r_orig = matrix3d::rotation_from_euler_angles(ea);
        let d_orig = vect3d::new(1.0, 3.0, 7.0);
        let diag = matrix3d::from_elements(
            d_orig.x, 0.0, 0.0,
            0.0, d_orig.y, 0.0,
            0.0, 0.0, d_orig.z,
        );
        let k = r_orig * diag * r_orig.transpose();

        // Decompose
        let (r, d) = k.symmetric_eigen();

        // Eigenvalues should match (already sorted ascending)
        assert!((d.x - 1.0).abs() < 1e-10, "d.x={}", d.x);
        assert!((d.y - 3.0).abs() < 1e-10, "d.y={}", d.y);
        assert!((d.z - 7.0).abs() < 1e-10, "d.z={}", d.z);

        // R should be orthogonal
        assert!((r * r.transpose()).similar(matrix3d::identity()));

        // Reconstruct: R * diag(d) * R^T should equal K
        let diag_rec = matrix3d::from_elements(
            d.x, 0.0, 0.0,
            0.0, d.y, 0.0,
            0.0, 0.0, d.z,
        );
        let k_rec = r * diag_rec * r.transpose();
        assert!(k_rec.similar(k));
    }
}

