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
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct matrix3<T : Float>
{
    pub rows : [vect3<T>; 3]
}

/// 3x3 matrix with f32 elements.
pub type matrix3f = matrix3<f32>;
/// 3x3 matrix with f64 elements.
pub type matrix3d = matrix3<f64>;

impl<T: Float> Default for matrix3<T> {
    fn default() -> Self { matrix3::zero_matrix() }
}

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

    /// Constructs a matrix from a row-major nested array.
    pub fn from_array(a: [[T; 3]; 3]) -> matrix3<T> {
        matrix3::<T>::from_rows(
            vect3::<T>::new(a[0][0], a[0][1], a[0][2]),
            vect3::<T>::new(a[1][0], a[1][1], a[1][2]),
            vect3::<T>::new(a[2][0], a[2][1], a[2][2]),
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

    /// Vee of the antisymmetric part, `(M - M^T)/2`: for a rotation matrix
    /// exactly `sin(theta) * axis`, the rotation vector to first order near
    /// identity. The extraction companion of
    /// [`Self::from_rotation_vector_small`]; use on error rotations, not
    /// full attitudes.
    pub fn get_rotation_vector_small(self) -> vect3<T> {
        let half = T::half();
        vect3::<T>::new(
            (self[2][1] - self[1][2]) * half,
            (self[0][2] - self[2][0]) * half,
            (self[1][0] - self[0][1]) * half,
        )
    }

    /// Returns true if all elements are finite (not NaN or infinity).
    pub fn is_finite(self) -> bool {
        self.rows[0].is_finite() && self.rows[1].is_finite() && self.rows[2].is_finite()
    }

    /// Returns the row at the given index as a vector.
    pub fn row(self, index: usize) -> vect3<T> {
        self[index]
    }

    /// Returns the transpose of this matrix.
    pub fn transpose(self) -> matrix3<T> {
        matrix3::<T>::from_rows(self.col(0), self.col(1), self.col(2))
    }

    /// Builds an orthonormal frame whose THIRD column is the unit vector
    /// `n`: the returned matrix maps frame coordinates to world
    /// coordinates (`M * e_z == n`); its transpose maps world into the
    /// frame. The first two columns span the plane perpendicular to `n`
    /// (i.e. the null space of `n^T` -- the historical reason for the
    /// name, which is otherwise misleading: the matrix is a full
    /// rotation/frame, not a null-space projector).
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

    /// Derivatives of [`Self::rotation_from_euler_angles_sincos`] w.r.t. each euler
    /// angle: returns `[dR/dea.x, dR/dea.y, dR/dea.z]` (roll, pitch, yaw). Each
    /// entry is the derivative of the corresponding entry above, using
    /// `d(sin)/dangle = cos` and `d(cos)/dangle = -sin`; it consumes only the
    /// supplied sin/cos, so it adds no trig. Used to precompute the pose
    /// rotation Jacobian for constraints that differentiate through the
    /// euler-angle rotation.
    pub fn rotation_from_euler_angles_sincos_deriv(
        sin_euler_angles: vect3<T>, cos_euler_angles: vect3<T>,
    ) -> [matrix3<T>; 3] {
        let (sx, sy, sz) = (sin_euler_angles.x, sin_euler_angles.y, sin_euler_angles.z);
        let (cx, cy, cz) = (cos_euler_angles.x, cos_euler_angles.y, cos_euler_angles.z);
        let z = T::zero();
        // d/d(roll): sx -> cx, cx -> -sx
        let dx = matrix3::<T>::from_rows(
            vect3::<T>::new(z, sx*sz + cz*cx*sy,  -sx*cz*sy + cx*sz),
            vect3::<T>::new(z, -sx*cz + cx*sy*sz, -sx*sy*sz - cz*cx),
            vect3::<T>::new(z, cy*cx,             -sx*cy),
        );
        // d/d(pitch): sy -> cy, cy -> -sy
        let dy = matrix3::<T>::from_rows(
            vect3::<T>::new(-sy*cz, cz*sx*cy, cx*cz*cy),
            vect3::<T>::new(-sy*sz, sx*cy*sz, cx*cy*sz),
            vect3::<T>::new(-cy,    -sy*sx,   -cx*sy),
        );
        // d/d(yaw): sz -> cz, cz -> -sz
        let dz = matrix3::<T>::from_rows(
            vect3::<T>::new(-cy*sz, -cx*cz - sz*sx*sy, -cx*sz*sy + sx*cz),
            vect3::<T>::new(cy*cz,  -cx*sz + sx*sy*cz, cx*sy*cz + sz*sx),
            vect3::<T>::new(z,      z,                 z),
        );
        [dx, dy, dz]
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

    /// Rotation matrix of the small-angle retraction `normalize(1, v/2)`,
    /// computed sqrt-free from the unnormalized quaternion `(1, v/2)`:
    /// `M(1, v/2)` scaled by `s = 2/|q|^2`, `|q|^2 = 1 + |v|^2/4 >= 1`.
    /// Algebraically identical to
    /// `quatern::from_rotation_vector_small(v).rotation_matrix()` but with no
    /// normalization sqrt; matches the symbolic `matrix3sym::from_rotation_vector_small`.
    pub fn from_rotation_vector_small(v: vect3<T>) -> matrix3<T> {
        let half = T::half();
        let (x, y, z) = (v.x * half, v.y * half, v.z * half);
        let (x2, y2, z2) = (x * x, y * y, z * z);
        let s = T::two() / (T::one() + x2 + y2 + z2);
        let one = T::one();
        matrix3::<T>::from_rows(
            vect3::<T>::new(one - s * (y2 + z2), s * (x * y - z), s * (x * z + y)),
            vect3::<T>::new(s * (x * y + z), one - s * (x2 + z2), s * (y * z - x)),
            vect3::<T>::new(s * (x * z - y), s * (y * z + x), one - s * (x2 + y2)),
        )
    }

    /// Derivatives of [`Self::from_rotation_vector_small`] w.r.t. each component of
    /// `v`: returns `[dR/dv.x, dR/dv.y, dR/dv.z]`. With `x,y,z = v/2` and
    /// `s = 2/(1 + x^2+y^2+z^2)`, the only non-trivial part is `ds/d(v.k) =
    /// -half * k_half * s^2` (each entry is `+-s*P` with polynomial `P`).
    /// Used to precompute the pose rotation Jacobian for constraints that
    /// differentiate through the retraction.
    pub fn from_rotation_vector_small_deriv(v: vect3<T>) -> [matrix3<T>; 3] {
        let half = T::half();
        let two = T::two();
        let (x, y, z) = (v.x * half, v.y * half, v.z * half);
        let (x2, y2, z2) = (x * x, y * y, z * z);
        let s = two / (T::one() + x2 + y2 + z2);
        let s2 = s * s;
        let n = |e: T| e * half; // d/d(v.k) = half * d/d(k_half)
        let dx = matrix3::<T>::from_rows(
            vect3::<T>::new(n(x*s2*(y2+z2)),           n(-x*s2*(x*y-z) + s*y),     n(-x*s2*(x*z+y) + s*z)),
            vect3::<T>::new(n(-x*s2*(x*y+z) + s*y),    n(x*s2*(x2+z2) - two*s*x),  n(-x*s2*(y*z-x) - s)),
            vect3::<T>::new(n(-x*s2*(x*z-y) + s*z),    n(-x*s2*(y*z+x) + s),       n(x*s2*(x2+y2) - two*s*x)),
        );
        let dy = matrix3::<T>::from_rows(
            vect3::<T>::new(n(y*s2*(y2+z2) - two*s*y), n(-y*s2*(x*y-z) + s*x),     n(-y*s2*(x*z+y) + s)),
            vect3::<T>::new(n(-y*s2*(x*y+z) + s*x),    n(y*s2*(x2+z2)),            n(-y*s2*(y*z-x) + s*z)),
            vect3::<T>::new(n(-y*s2*(x*z-y) - s),      n(-y*s2*(y*z+x) + s*z),     n(y*s2*(x2+y2) - two*s*y)),
        );
        let dz = matrix3::<T>::from_rows(
            vect3::<T>::new(n(z*s2*(y2+z2) - two*s*z), n(-z*s2*(x*y-z) - s),       n(-z*s2*(x*z+y) + s*x)),
            vect3::<T>::new(n(-z*s2*(x*y+z) + s),      n(z*s2*(x2+z2) - two*s*z),  n(-z*s2*(y*z-x) + s*y)),
            vect3::<T>::new(n(-z*s2*(x*z-y) + s*x),    n(-z*s2*(y*z+x) + s*y),     n(z*s2*(x2+y2))),
        );
        [dx, dy, dz]
    }

    /// Extracts Euler angles (x=roll, y=pitch, z=yaw) from a rotation matrix.
    /// At and near gimbal lock (|pitch| within ~sqrt(eps) of pi/2) only
    /// roll -+ yaw is determined; the roll = 0 convention is used and yaw
    /// carries the combined angle. The recomposition error is bounded by
    /// ~sqrt(eps) everywhere -- the information-theoretic floor for
    /// extracting euler angles from a float matrix.
    pub fn get_euler_angles(self) -> vect3<T> {
        // safe_asin: float noise can push a valid rotation's entry just
        // past +-1, where raw asin returns NaN and poisons all angles.
        let y = -self[2][0].safe_asin();
        // The roll/yaw split lives in entries scaled by cos(pitch); this
        // sum of squares is cos(pitch)^2 for a consistent matrix, and the
        // squared signal magnitude for one that is orthonormal only to
        // solver tolerance. Below eps (i.e. |cos(pitch)| below sqrt(eps))
        // the split is noise: sqrt(eps) is the crossover where the main
        // branch's eps/cos(pitch) amplification equals the lock branch's
        // cos(pitch) truncation.
        let cp2 = self[2][1] * self[2][1] + self[2][2] * self[2][2];
        if cp2 > T::epsilon() {
            vect3::<T>::new(
                atan2(self[2][1], self[2][2]),
                y,
                atan2(self[1][0], self[0][0])
            )
        } else {
            // Gimbal lock: m01/m11 stay well-conditioned and hold the
            // combined angle in both hemispheres:
            // m01 = -+sin(roll -+ yaw), m11 = cos(roll -+ yaw).
            vect3::<T>::new(
                T::zero(),
                y,
                atan2(-self[0][1], self[1][1])
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
    /// The matrix must be symmetric. Returns (R, d) where R holds the
    /// eigenvectors as columns and d the eigenvalues, sorted ascending.
    /// R is orthonormal but not necessarily a rotation: eigenvector signs
    /// and ordering are arbitrary, so det(R) may be -1 (a reflection).
    /// This does not affect uses that rely only on self = R * diag(d) * R^T
    /// (e.g. covariance whitening); if a proper rotation is needed, negate
    /// one column when det(R) < 0.
    ///
    /// Non-finite input propagates: NaN in yields NaN out.
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
        idx.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));

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
#[derive(Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct matrix2<T : Float>
{
    pub rows : [vect2<T>; 2]
}

/// 2x2 matrix with f32 elements.
pub type matrix2f = matrix2<f32>;
/// 2x2 matrix with f64 elements.
pub type matrix2d = matrix2<f64>;

impl<T: Float> Default for matrix2<T> {
    fn default() -> Self { matrix2::zero_matrix() }
}

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
    /// Returns true if all elements are finite (not NaN or infinity).
    pub fn is_finite(self) -> bool {
        self.rows[0].is_finite() && self.rows[1].is_finite()
    }

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

    /// Symmetric eigendecomposition: self = R * diag(d) * R^T
    ///
    /// The matrix must be symmetric. Returns (R, d) where R holds the
    /// eigenvectors as columns and d the eigenvalues, sorted ascending.
    /// R is orthonormal but not necessarily a rotation: eigenvector signs
    /// and ordering are arbitrary, so det(R) may be -1 (a reflection).
    /// This does not affect uses that rely only on self = R * diag(d) * R^T
    /// (e.g. covariance whitening); if a proper rotation is needed, negate
    /// one column when det(R) < 0.
    ///
    /// Non-finite input propagates: NaN in yields NaN out.
    pub fn symmetric_eigen(self) -> (matrix2<T>, vect2<T>) {
        let na_mat = nalgebra::Matrix2::new(
            self[0][0].to_f64().unwrap(), self[0][1].to_f64().unwrap(),
            self[1][0].to_f64().unwrap(), self[1][1].to_f64().unwrap(),
        );
        let eigen = na_mat.symmetric_eigen();

        // Sort by eigenvalue (ascending)
        let mut idx = [0usize, 1];
        idx.sort_by(|&a, &b| eigen.eigenvalues[a].total_cmp(&eigen.eigenvalues[b]));

        let d = vect2::<T>::new(
            T::from(eigen.eigenvalues[idx[0]]).unwrap(),
            T::from(eigen.eigenvalues[idx[1]]).unwrap(),
        );
        let r = matrix2::<T>::from_cols(
            vect2::<T>::new(
                T::from(eigen.eigenvectors[(0, idx[0])]).unwrap(),
                T::from(eigen.eigenvectors[(1, idx[0])]).unwrap(),
            ),
            vect2::<T>::new(
                T::from(eigen.eigenvectors[(0, idx[1])]).unwrap(),
                T::from(eigen.eigenvectors[(1, idx[1])]).unwrap(),
            ),
        );
        (r, d)
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

    #[test]
    fn test_default_is_zero() {
        let m3 = matrix3d::default();
        let m2 = matrix2d::default();
        for r in 0..3 {
            for c in 0..3 {
                assert_eq!(m3[r][c], 0.0);
            }
        }
        for r in 0..2 {
            for c in 0..2 {
                assert_eq!(m2[r][c], 0.0);
            }
        }
    }

    #[test]
    fn test_get_rotation_vector_small_round_trip() {
        // For small w, from_rotation_vector_small and the vee extraction
        // agree to O(|w|^3) (the extraction returns sin(theta) * axis).
        let w = vect3d::new(0.01, -0.02, 0.015);
        let r = matrix3d::from_rotation_vector_small(w);
        let back = r.get_rotation_vector_small();
        assert!((back - w).norm() < 1e-5, "back={:?}", back);
        // Identity has a zero rotation vector.
        let z = matrix3d::identity().get_rotation_vector_small();
        assert_eq!(z.norm(), 0.0);
    }

    #[test]
    fn test_get_euler_angles_noisy_gimbal_boundary() {
        // Regression: a rotation matrix whose (2,0) entry drifted past
        // -1 by float noise fed raw asin -> NaN poisoned all three
        // angles. safe_asin clamps, so pitch saturates at +pi/2.
        let base = matrix3d::rotation_from_euler_angles(
            vect3d::new(0.0, f64::half_pi(), 0.0));
        let noisy = matrix3d::from_rows(
            base[0],
            base[1],
            vect3d::new(-1.0 - 1e-10, base[2].y, base[2].z),
        );
        let ea = noisy.get_euler_angles();
        assert!(ea.x.is_finite() && ea.y.is_finite() && ea.z.is_finite(),
            "angles must be finite, got {:?}", ea);
        assert!((ea.y - f64::half_pi()).abs() < 1e-6, "pitch={}", ea.y);

        // And the other direction (entry past +1 -> pitch -pi/2).
        let noisy2 = matrix3d::from_rows(
            base[0], base[1],
            vect3d::new(1.0 + 1e-10, base[2].y, base[2].z),
        );
        let ea2 = noisy2.get_euler_angles();
        assert!(ea2.y.is_finite());
        assert!((ea2.y + f64::half_pi()).abs() < 1e-6, "pitch={}", ea2.y);
    }

    #[test]
    fn test_get_euler_angles_exact_gimbal_lock() {
        // At exact gimbal lock (cos(pitch) == 0, entries literally zero)
        // only roll -+ yaw is determined; the extractor must use the
        // roll = 0 convention and recover the combined angle from
        // m01/m11, which stay well-conditioned at lock. A hand-written
        // lock matrix (e.g. an axis-aligned 90 degree rotation) is the
        // realistic trigger -- computed rotations never have exact zeros.
        let d: f64 = 0.4;

        // pitch = +pi/2: m01 = sin(roll - yaw), m11 = cos(roll - yaw)
        let m = matrix3d::from_rows(
            vect3d::new(0.0, d.sin(), d.cos()),
            vect3d::new(0.0, d.cos(), -d.sin()),
            vect3d::new(-1.0, 0.0, 0.0),
        );
        let ea = m.get_euler_angles();
        assert_eq!(ea.x, 0.0, "roll must follow the roll=0 convention");
        assert!((ea.y - f64::half_pi()).abs() < 1e-12, "pitch={}", ea.y);
        assert!(matrix3d::rotation_from_euler_angles(ea).similar(m),
            "recomposition mismatch, ea={:?}", ea);

        // pitch = -pi/2: m01 = -sin(roll + yaw), m11 = cos(roll + yaw)
        let m2 = matrix3d::from_rows(
            vect3d::new(0.0, -d.sin(), -d.cos()),
            vect3d::new(0.0, d.cos(), -d.sin()),
            vect3d::new(1.0, 0.0, 0.0),
        );
        let ea2 = m2.get_euler_angles();
        assert_eq!(ea2.x, 0.0);
        assert!((ea2.y + f64::half_pi()).abs() < 1e-12, "pitch={}", ea2.y);
        assert!(matrix3d::rotation_from_euler_angles(ea2).similar(m2),
            "recomposition mismatch, ea={:?}", ea2);

        // Round trip through a lock rotation built from angles: sin of
        // the f64 nearest pi/2 rounds to exactly 1, so this also lands
        // in the lock branch (with tiny nonzero off-entries).
        let src = vect3d::new(0.3, f64::half_pi(), 0.7);
        let m3 = matrix3d::rotation_from_euler_angles(src);
        let ea3 = m3.get_euler_angles();
        assert!(matrix3d::rotation_from_euler_angles(ea3).similar(m3),
            "recomposition mismatch, ea={:?}", ea3);
    }

    #[test]
    fn test_symmetric_eigen_nan_propagates() {
        // NaN input must propagate to the output, not panic in the
        // eigenvalue sort (partial_cmp().unwrap() used to).
        let nan = f64::NAN;
        let m = matrix3d::from_rows(
            vect3d::new(nan, 0.0, 0.0),
            vect3d::new(0.0, 2.0, 0.0),
            vect3d::new(0.0, 0.0, 1.0),
        );
        let (_r, d) = m.symmetric_eigen();
        assert!(d.x.is_nan() || d.y.is_nan() || d.z.is_nan(),
            "NaN input must yield NaN eigenvalues, got {:?}", d);
    }

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

    #[test]
    fn test_symmetric_eigen_2x2() {
        // Build a symmetric 2x2 K = R * diag(d) * R^T with known values.
        let r_orig = matrix2d::rotation(0.7);
        let d_orig = vect2d::new(2.0, 9.0);
        let diag = matrix2d::from_elements(d_orig.x, 0.0, 0.0, d_orig.y);
        let k = r_orig * diag * r_orig.transpose();

        let (r, d) = k.symmetric_eigen();

        // Eigenvalues match (sorted ascending).
        assert!((d.x - 2.0).abs() < 1e-10, "d.x={}", d.x);
        assert!((d.y - 9.0).abs() < 1e-10, "d.y={}", d.y);
        // R orthogonal.
        assert!((r * r.transpose()).similar(matrix2d::identity()));
        // Reconstruct.
        let diag_rec = matrix2d::from_elements(d.x, 0.0, 0.0, d.y);
        assert!((r * diag_rec * r.transpose()).similar(k));
    }

    #[test]
    fn test_symmetric_eigen_2x2_nan_propagates() {
        // NaN input must propagate to the output, not panic in the
        // eigenvalue sort.
        let nan = f64::NAN;
        let m = matrix2d::from_rows(vect2d::new(nan, 0.0), vect2d::new(0.0, 2.0));
        let (_r, d) = m.symmetric_eigen();
        assert!(d.x.is_nan() || d.y.is_nan(),
            "NaN input must yield NaN eigenvalues, got {:?}", d);
    }

    #[test]
    fn test_from_array() {
        let a = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        assert!(matrix3d::from_array(a).similar(
            matrix3d::from_elements(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)));
    }

    #[test]
    fn from_rotation_vector_small_matches_quaternion_and_is_rotation() {
        use crate::quatern::quaternd;
        for v in [
            vect3d::new(0.0, 0.0, 0.0),
            vect3d::new(0.3, -0.5, 0.2),
            vect3d::new(1.2, 0.0, 0.0),
            vect3d::new(-0.8, 0.9, 1.5),
        ] {
            let m = matrix3d::from_rotation_vector_small(v);
            // The sqrt-free matrix equals the retraction quaternion's rotation matrix.
            assert!(m.similar(quaternd::from_rotation_vector_small(v).rotation_matrix()),
                "sqrt-free != quaternion.rotation_matrix() at v={:?}", v);
            // And it is a proper rotation.
            assert!((m * m.transpose()).similar(matrix3d::identity()),
                "not orthonormal at v={:?}", v);
            assert!((m.det() - 1.0).abs() < 1e-12, "det != 1 at v={:?}", v);
        }
    }

    #[test]
    fn from_rotation_vector_small_deriv_matches_finite_diff() {
        let eps = 1e-7;
        let get = |m: &matrix3d, i: usize, j: usize| match (i, j) {
            (0,0)=>m.rows[0].x,(0,1)=>m.rows[0].y,(0,2)=>m.rows[0].z,
            (1,0)=>m.rows[1].x,(1,1)=>m.rows[1].y,(1,2)=>m.rows[1].z,
            (2,0)=>m.rows[2].x,(2,1)=>m.rows[2].y,_=>m.rows[2].z,
        };
        for v in [vect3d::new(0.0,0.0,0.0), vect3d::new(0.3,-0.5,0.2),
                  vect3d::new(1.2,0.0,-0.4), vect3d::new(-0.7,0.9,0.4)] {
            let d = matrix3d::from_rotation_vector_small_deriv(v);
            let r0 = matrix3d::from_rotation_vector_small(v);
            for k in 0..3 {
                let mut vp = v;
                match k { 0=>vp.x+=eps, 1=>vp.y+=eps, _=>vp.z+=eps }
                let rp = matrix3d::from_rotation_vector_small(vp);
                for i in 0..3 { for j in 0..3 {
                    let fd = (get(&rp,i,j) - get(&r0,i,j)) / eps;
                    assert!((get(&d[k],i,j) - fd).abs() < 1e-4,
                        "d/dv[{k}] [{i}][{j}] at v={v:?}: analytic {} vs fd {}",
                        get(&d[k],i,j), fd);
                }}
            }
        }
    }

    #[test]
    fn rotation_from_euler_angles_sincos_deriv_matches_finite_diff() {
        let eps = 1e-7;
        let get = |m: &matrix3d, i: usize, j: usize| match (i, j) {
            (0,0)=>m.rows[0].x,(0,1)=>m.rows[0].y,(0,2)=>m.rows[0].z,
            (1,0)=>m.rows[1].x,(1,1)=>m.rows[1].y,(1,2)=>m.rows[1].z,
            (2,0)=>m.rows[2].x,(2,1)=>m.rows[2].y,_=>m.rows[2].z,
        };
        for ea in [vect3d::new(0.0,0.0,0.0), vect3d::new(0.3,-0.5,0.2),
                   vect3d::new(1.2,0.7,-0.4), vect3d::new(-0.7,0.9,2.1)] {
            let (s, c) = ea.sincos();
            let d = matrix3d::rotation_from_euler_angles_sincos_deriv(s, c);
            let r0 = matrix3d::rotation_from_euler_angles(ea);
            for k in 0..3 {
                let mut ep = ea;
                match k { 0=>ep.x+=eps, 1=>ep.y+=eps, _=>ep.z+=eps }
                let rp = matrix3d::rotation_from_euler_angles(ep);
                for i in 0..3 { for j in 0..3 {
                    let fd = (get(&rp,i,j) - get(&r0,i,j)) / eps;
                    assert!((get(&d[k],i,j) - fd).abs() < 1e-4,
                        "dR/dea[{k}] [{i}][{j}] at ea={ea:?}: analytic {} vs fd {}",
                        get(&d[k],i,j), fd);
                }}
            }
        }
    }
}

