//! Symbolic companion types for geometric primitives (vectors, matrices, quaternions).

#![allow(non_camel_case_types)]

use std::ops;
use crate::{E, symbol, sin, cos};

// ---------------------------------------------------------------------------
// vect3sym
// ---------------------------------------------------------------------------

/// Symbolic 3D vector with x, y, z components.
///
/// Convention: x = forward, y = left, z = up.
#[derive(Clone)]
pub struct vect3sym {
    /// Forward component.
    pub x: E,
    /// Left component.
    pub y: E,
    /// Up component.
    pub z: E,
}

impl vect3sym {
    /// Create a symbolic 3D vector whose component symbols are named
    /// `{base}.x`, `{base}.y`, `{base}.z`.
    pub fn new(base: &str) -> Self {
        vect3sym {
            x: symbol(&format!("{}.x", base)),
            y: symbol(&format!("{}.y", base)),
            z: symbol(&format!("{}.z", base)),
        }
    }

    /// Compute element-wise (sin, cos) of this vector, returning two vectors.
    pub fn sincos(&self) -> (vect3sym, vect3sym) {
        (
            vect3sym {
                x: sin(self.x.clone()),
                y: sin(self.y.clone()),
                z: sin(self.z.clone()),
            },
            vect3sym {
                x: cos(self.x.clone()),
                y: cos(self.y.clone()),
                z: cos(self.z.clone()),
            },
        )
    }

    /// Build a 3x3 rotation matrix from Euler angles (x=roll, y=pitch, z=yaw).
    ///
    /// Uses the intrinsic ZYX (yaw-pitch-roll) rotation convention.
    pub fn rotation_matrix(&self) -> matrix3sym {
        let (s, c) = self.sincos();
        matrix3sym {
            rows: [
                vect3sym {
                    x: c.y.clone() * c.z.clone(),
                    y: -c.x.clone() * s.z.clone() + c.z.clone() * s.x.clone() * s.y.clone(),
                    z: c.x.clone() * c.z.clone() * s.y.clone() + s.x.clone() * s.z.clone(),
                },
                vect3sym {
                    x: c.y.clone() * s.z.clone(),
                    y: c.x.clone() * c.z.clone() + s.x.clone() * s.y.clone() * s.z.clone(),
                    z: c.x.clone() * s.y.clone() * s.z.clone() - c.z.clone() * s.x.clone(),
                },
                vect3sym {
                    x: -s.y.clone(),
                    y: c.y.clone() * s.x.clone(),
                    z: c.x.clone() * c.y.clone(),
                },
            ],
        }
    }
}

impl ops::Add<vect3sym> for vect3sym {
    type Output = vect3sym;
    fn add(self, rhs: vect3sym) -> vect3sym {
        vect3sym { x: self.x + rhs.x, y: self.y + rhs.y, z: self.z + rhs.z }
    }
}

impl ops::Sub<vect3sym> for vect3sym {
    type Output = vect3sym;
    fn sub(self, rhs: vect3sym) -> vect3sym {
        vect3sym { x: self.x - rhs.x, y: self.y - rhs.y, z: self.z - rhs.z }
    }
}

impl ops::Neg for vect3sym {
    type Output = vect3sym;
    fn neg(self) -> vect3sym {
        vect3sym { x: -self.x, y: -self.y, z: -self.z }
    }
}

impl ops::Mul<E> for vect3sym {
    type Output = vect3sym;
    fn mul(self, rhs: E) -> vect3sym {
        vect3sym { x: self.x * rhs.clone(), y: self.y * rhs.clone(), z: self.z * rhs }
    }
}

impl ops::Mul<vect3sym> for E {
    type Output = vect3sym;
    fn mul(self, rhs: vect3sym) -> vect3sym {
        vect3sym { x: self.clone() * rhs.x, y: self.clone() * rhs.y, z: self * rhs.z }
    }
}

impl ops::Mul<vect3sym> for vect3sym {
    type Output = E;
    fn mul(self, rhs: vect3sym) -> E {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}

impl ops::Div<E> for vect3sym {
    type Output = vect3sym;
    fn div(self, rhs: E) -> vect3sym {
        vect3sym { x: self.x / rhs.clone(), y: self.y / rhs.clone(), z: self.z / rhs }
    }
}

impl ops::Rem<vect3sym> for vect3sym {
    type Output = vect3sym;
    /// Cross product (mirrors the runtime `vect3` `%` operator).
    fn rem(self, rhs: vect3sym) -> vect3sym {
        self.cross(&rhs)
    }
}

impl vect3sym {
    /// Squared length (dot product with self).
    pub fn square(&self) -> E {
        self.x.clone() * self.x.clone()
            + self.y.clone() * self.y.clone()
            + self.z.clone() * self.z.clone()
    }
    /// Length (Euclidean norm).
    pub fn norm(&self) -> E {
        crate::sqrt(self.square())
    }
    /// Unit (normalized) vector.
    pub fn unit(self) -> vect3sym {
        let n = self.norm();
        self / n
    }
    /// Cross product.
    pub fn cross(&self, rhs: &vect3sym) -> vect3sym {
        vect3sym {
            x: self.y.clone() * rhs.z.clone() - self.z.clone() * rhs.y.clone(),
            y: self.z.clone() * rhs.x.clone() - self.x.clone() * rhs.z.clone(),
            z: self.x.clone() * rhs.y.clone() - self.y.clone() * rhs.x.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// vect2sym
// ---------------------------------------------------------------------------

/// Symbolic 2D vector with x, y components.
#[derive(Clone)]
pub struct vect2sym {
    /// X component.
    pub x: E,
    /// Y component.
    pub y: E,
}

impl vect2sym {
    /// Create a symbolic 2D vector whose component symbols are named
    /// `{base}.x`, `{base}.y`.
    pub fn new(base: &str) -> Self {
        vect2sym {
            x: symbol(&format!("{}.x", base)),
            y: symbol(&format!("{}.y", base)),
        }
    }
}

impl ops::Sub<vect2sym> for vect2sym {
    type Output = vect2sym;
    fn sub(self, rhs: vect2sym) -> vect2sym {
        vect2sym { x: self.x - rhs.x, y: self.y - rhs.y }
    }
}

impl ops::Add<vect2sym> for vect2sym {
    type Output = vect2sym;
    fn add(self, rhs: vect2sym) -> vect2sym {
        vect2sym { x: self.x + rhs.x, y: self.y + rhs.y }
    }
}

impl ops::Neg for vect2sym {
    type Output = vect2sym;
    fn neg(self) -> vect2sym {
        vect2sym { x: -self.x, y: -self.y }
    }
}

impl ops::Mul<E> for vect2sym {
    type Output = vect2sym;
    fn mul(self, rhs: E) -> vect2sym {
        vect2sym { x: self.x * rhs.clone(), y: self.y * rhs }
    }
}

impl ops::Mul<vect2sym> for E {
    type Output = vect2sym;
    fn mul(self, rhs: vect2sym) -> vect2sym {
        vect2sym { x: self.clone() * rhs.x, y: self * rhs.y }
    }
}

impl ops::Mul<vect2sym> for vect2sym {
    type Output = E;
    /// Dot product.
    fn mul(self, rhs: vect2sym) -> E {
        self.x * rhs.x + self.y * rhs.y
    }
}

impl ops::Div<E> for vect2sym {
    type Output = vect2sym;
    fn div(self, rhs: E) -> vect2sym {
        vect2sym { x: self.x / rhs.clone(), y: self.y / rhs }
    }
}

impl vect2sym {
    /// Squared length (dot product with self).
    pub fn square(&self) -> E {
        self.x.clone() * self.x.clone() + self.y.clone() * self.y.clone()
    }
    /// Length (Euclidean norm).
    pub fn norm(&self) -> E {
        crate::sqrt(self.square())
    }
    /// Unit (normalized) vector.
    pub fn unit(self) -> vect2sym {
        let n = self.norm();
        self / n
    }
    /// Perpendicular vector (90-degree counter-clockwise rotation).
    pub fn across(self) -> vect2sym {
        vect2sym { x: -self.y, y: self.x }
    }
    /// 2D cross product (determinant): self.x * rhs.y - self.y * rhs.x.
    pub fn cross(&self, rhs: &vect2sym) -> E {
        self.x.clone() * rhs.y.clone() - self.y.clone() * rhs.x.clone()
    }
}

// ---------------------------------------------------------------------------
// matrix3sym
// ---------------------------------------------------------------------------

/// Symbolic 3x3 matrix, stored as three row vectors.
#[derive(Clone)]
pub struct matrix3sym {
    /// The three row vectors.
    pub rows: [vect3sym; 3],
}

impl matrix3sym {
    /// Create a symbolic 3x3 matrix. Row symbols are `{base}[0]`, `{base}[1]`,
    /// `{base}[2]`, with each row having `.x`, `.y`, `.z` components.
    pub fn new(base: &str) -> Self {
        matrix3sym {
            rows: [
                vect3sym::new(&format!("{}[0]", base)),
                vect3sym::new(&format!("{}[1]", base)),
                vect3sym::new(&format!("{}[2]", base)),
            ],
        }
    }

    /// Extract Euler angles (x=roll, y=pitch, z=yaw) from this rotation matrix.
    pub fn get_euler_angles(&self) -> vect3sym {
        vect3sym {
            x: crate::atan2(self.rows[2].y.clone(), self.rows[2].z.clone()),
            // safe_asin mirrors the runtime matrix3::get_euler_angles:
            // clamp noise past +-1 instead of emitting NaN.
            y: -crate::safe_asin(self.rows[2].x.clone()),
            z: crate::atan2(self.rows[1].x.clone(), self.rows[0].x.clone()),
        }
    }

    /// Return the transpose of this 3x3 matrix.
    pub fn transpose(&self) -> matrix3sym {
        matrix3sym {
            rows: [
                vect3sym { x: self.rows[0].x.clone(), y: self.rows[1].x.clone(), z: self.rows[2].x.clone() },
                vect3sym { x: self.rows[0].y.clone(), y: self.rows[1].y.clone(), z: self.rows[2].y.clone() },
                vect3sym { x: self.rows[0].z.clone(), y: self.rows[1].z.clone(), z: self.rows[2].z.clone() },
            ],
        }
    }

    /// Identity matrix (constant entries).
    pub fn identity() -> matrix3sym {
        let o = || crate::constant(1.0);
        let z = || crate::constant(0.0);
        matrix3sym {
            rows: [
                vect3sym { x: o(), y: z(), z: z() },
                vect3sym { x: z(), y: o(), z: z() },
                vect3sym { x: z(), y: z(), z: o() },
            ],
        }
    }

    /// Build a matrix from three row vectors.
    pub fn from_rows(r0: vect3sym, r1: vect3sym, r2: vect3sym) -> matrix3sym {
        matrix3sym { rows: [r0, r1, r2] }
    }

    /// Build a matrix from three column vectors.
    pub fn from_cols(c0: vect3sym, c1: vect3sym, c2: vect3sym) -> matrix3sym {
        matrix3sym {
            rows: [
                vect3sym { x: c0.x, y: c1.x, z: c2.x },
                vect3sym { x: c0.y, y: c1.y, z: c2.y },
                vect3sym { x: c0.z, y: c1.z, z: c2.z },
            ],
        }
    }

    /// Build a matrix from nine scalar elements in row-major order.
    #[allow(clippy::too_many_arguments)]
    pub fn from_elements(m00: E, m01: E, m02: E,
                         m10: E, m11: E, m12: E,
                         m20: E, m21: E, m22: E) -> matrix3sym {
        matrix3sym {
            rows: [
                vect3sym { x: m00, y: m01, z: m02 },
                vect3sym { x: m10, y: m11, z: m12 },
                vect3sym { x: m20, y: m21, z: m22 },
            ],
        }
    }

    /// Determinant.
    pub fn det(&self) -> E {
        let m = &self.rows;
        m[0].x.clone() * m[1].y.clone() * m[2].z.clone()
            + m[0].z.clone() * m[1].x.clone() * m[2].y.clone()
            + m[0].y.clone() * m[1].z.clone() * m[2].x.clone()
            - m[0].x.clone() * m[1].z.clone() * m[2].y.clone()
            - m[0].y.clone() * m[1].x.clone() * m[2].z.clone()
            - m[0].z.clone() * m[1].y.clone() * m[2].x.clone()
    }

    /// Rotation matrix from Euler angles (x=roll, y=pitch, z=yaw), intrinsic
    /// ZYX convention. Mirrors `matrix3::rotation_from_euler_angles`.
    pub fn rotation_from_euler_angles(ea: &vect3sym) -> matrix3sym {
        ea.rotation_matrix()
    }

    /// Rotation matrix from a unit axis and a symbolic angle. Mirrors
    /// `matrix3::rotation_from_axis_angle`.
    pub fn rotation_from_axis_angle(axis: &vect3sym, phi: E) -> matrix3sym {
        let s = sin(phi.clone());
        let c = cos(phi);
        let one = crate::constant(1.0);
        let k = one - c.clone(); // 1 - cos
        let (ax, ay, az) = (axis.x.clone(), axis.y.clone(), axis.z.clone());
        matrix3sym {
            rows: [
                vect3sym {
                    x: c.clone() + ax.clone() * ax.clone() * k.clone(),
                    y: ax.clone() * ay.clone() * k.clone() - az.clone() * s.clone(),
                    z: ax.clone() * az.clone() * k.clone() + ay.clone() * s.clone(),
                },
                vect3sym {
                    x: ay.clone() * ax.clone() * k.clone() + az.clone() * s.clone(),
                    y: c.clone() + ay.clone() * ay.clone() * k.clone(),
                    z: ay.clone() * az.clone() * k.clone() - ax.clone() * s.clone(),
                },
                vect3sym {
                    x: az.clone() * ax.clone() * k.clone() - ay.clone() * s.clone(),
                    y: az.clone() * ay.clone() * k.clone() + ax.clone() * s.clone(),
                    z: c + az.clone() * az * k,
                },
            ],
        }
    }
}

impl ops::Add<matrix3sym> for matrix3sym {
    type Output = matrix3sym;
    fn add(self, rhs: matrix3sym) -> matrix3sym {
        let [a0, a1, a2] = self.rows;
        let [b0, b1, b2] = rhs.rows;
        matrix3sym { rows: [a0 + b0, a1 + b1, a2 + b2] }
    }
}

impl ops::Sub<matrix3sym> for matrix3sym {
    type Output = matrix3sym;
    fn sub(self, rhs: matrix3sym) -> matrix3sym {
        let [a0, a1, a2] = self.rows;
        let [b0, b1, b2] = rhs.rows;
        matrix3sym { rows: [a0 - b0, a1 - b1, a2 - b2] }
    }
}

impl ops::Neg for matrix3sym {
    type Output = matrix3sym;
    fn neg(self) -> matrix3sym {
        let [r0, r1, r2] = self.rows;
        matrix3sym { rows: [-r0, -r1, -r2] }
    }
}

impl ops::Mul<E> for matrix3sym {
    type Output = matrix3sym;
    fn mul(self, rhs: E) -> matrix3sym {
        let [r0, r1, r2] = self.rows;
        matrix3sym { rows: [r0 * rhs.clone(), r1 * rhs.clone(), r2 * rhs] }
    }
}

impl ops::Mul<matrix3sym> for E {
    type Output = matrix3sym;
    fn mul(self, rhs: matrix3sym) -> matrix3sym {
        rhs * self
    }
}

impl ops::Mul<matrix3sym> for vect3sym {
    type Output = vect3sym;
    /// Row vector times matrix: `v * M = M^T * v`.
    fn mul(self, rhs: matrix3sym) -> vect3sym {
        rhs.transpose() * self
    }
}

impl ops::Mul<matrix3sym> for matrix3sym {
    type Output = matrix3sym;
    fn mul(self, rhs: matrix3sym) -> matrix3sym {
        let rhs_t = rhs.transpose();
        matrix3sym {
            rows: [
                vect3sym {
                    x: self.rows[0].clone() * rhs_t.rows[0].clone(),
                    y: self.rows[0].clone() * rhs_t.rows[1].clone(),
                    z: self.rows[0].clone() * rhs_t.rows[2].clone(),
                },
                vect3sym {
                    x: self.rows[1].clone() * rhs_t.rows[0].clone(),
                    y: self.rows[1].clone() * rhs_t.rows[1].clone(),
                    z: self.rows[1].clone() * rhs_t.rows[2].clone(),
                },
                vect3sym {
                    x: self.rows[2].clone() * rhs_t.rows[0].clone(),
                    y: self.rows[2].clone() * rhs_t.rows[1].clone(),
                    z: self.rows[2].clone() * rhs_t.rows[2].clone(),
                },
            ],
        }
    }
}

impl ops::Mul<vect3sym> for matrix3sym {
    type Output = vect3sym;
    fn mul(self, rhs: vect3sym) -> vect3sym {
        vect3sym {
            x: self.rows[0].x.clone() * rhs.x.clone() + self.rows[0].y.clone() * rhs.y.clone() + self.rows[0].z.clone() * rhs.z.clone(),
            y: self.rows[1].x.clone() * rhs.x.clone() + self.rows[1].y.clone() * rhs.y.clone() + self.rows[1].z.clone() * rhs.z.clone(),
            z: self.rows[2].x.clone() * rhs.x.clone() + self.rows[2].y.clone() * rhs.y.clone() + self.rows[2].z.clone() * rhs.z.clone(),
        }
    }
}

impl ops::Index<usize> for matrix3sym {
    type Output = vect3sym;
    fn index(&self, index: usize) -> &vect3sym {
        &self.rows[index]
    }
}

// ---------------------------------------------------------------------------
// matrix2sym
// ---------------------------------------------------------------------------

/// Symbolic 2x2 matrix, stored as two row vectors.
#[derive(Clone)]
pub struct matrix2sym {
    /// The two row vectors.
    pub rows: [vect2sym; 2],
}

impl matrix2sym {
    /// Create a symbolic 2x2 matrix. Row symbols are `{base}[0]`, `{base}[1]`,
    /// with each row having `.x`, `.y` components.
    pub fn new(base: &str) -> Self {
        matrix2sym {
            rows: [
                vect2sym::new(&format!("{}[0]", base)),
                vect2sym::new(&format!("{}[1]", base)),
            ],
        }
    }

    /// Build a 2D rotation matrix from a symbolic angle (radians, CCW).
    /// Mirrors `matrix2f::rotation`.
    pub fn rotation(angle: E) -> matrix2sym {
        let s = sin(angle.clone());
        let c = cos(angle);
        matrix2sym {
            rows: [
                vect2sym { x: c.clone(), y: -s.clone() },
                vect2sym { x: s,         y: c          },
            ],
        }
    }

    /// Return the transpose of this 2x2 matrix.
    pub fn transpose(&self) -> matrix2sym {
        matrix2sym {
            rows: [
                vect2sym { x: self.rows[0].x.clone(), y: self.rows[1].x.clone() },
                vect2sym { x: self.rows[0].y.clone(), y: self.rows[1].y.clone() },
            ],
        }
    }

    /// Identity matrix (constant entries).
    pub fn identity() -> matrix2sym {
        matrix2sym {
            rows: [
                vect2sym { x: crate::constant(1.0), y: crate::constant(0.0) },
                vect2sym { x: crate::constant(0.0), y: crate::constant(1.0) },
            ],
        }
    }

    /// Build a matrix from two row vectors.
    pub fn from_rows(r0: vect2sym, r1: vect2sym) -> matrix2sym {
        matrix2sym { rows: [r0, r1] }
    }

    /// Build a matrix from two column vectors.
    pub fn from_cols(c0: vect2sym, c1: vect2sym) -> matrix2sym {
        matrix2sym {
            rows: [
                vect2sym { x: c0.x, y: c1.x },
                vect2sym { x: c0.y, y: c1.y },
            ],
        }
    }

    /// Build a matrix from four scalar elements in row-major order.
    pub fn from_elements(m00: E, m01: E, m10: E, m11: E) -> matrix2sym {
        matrix2sym {
            rows: [
                vect2sym { x: m00, y: m01 },
                vect2sym { x: m10, y: m11 },
            ],
        }
    }

    /// Build a 2D rotation matrix from pre-computed sin and cos values.
    /// Mirrors `matrix2::rotation_from_sincos`.
    pub fn rotation_from_sincos(s: E, c: E) -> matrix2sym {
        matrix2sym {
            rows: [
                vect2sym { x: c.clone(), y: -s.clone() },
                vect2sym { x: s, y: c },
            ],
        }
    }

    /// Determinant.
    pub fn det(&self) -> E {
        self.rows[0].x.clone() * self.rows[1].y.clone()
            - self.rows[0].y.clone() * self.rows[1].x.clone()
    }

    /// Extract the rotation angle from a 2D rotation matrix. Mirrors
    /// `matrix2::get_rotation_angle`.
    pub fn get_rotation_angle(&self) -> E {
        crate::atan2(self.rows[1].x.clone(), self.rows[0].x.clone())
    }
}

impl ops::Add<matrix2sym> for matrix2sym {
    type Output = matrix2sym;
    fn add(self, rhs: matrix2sym) -> matrix2sym {
        let [a0, a1] = self.rows;
        let [b0, b1] = rhs.rows;
        matrix2sym { rows: [a0 + b0, a1 + b1] }
    }
}

impl ops::Sub<matrix2sym> for matrix2sym {
    type Output = matrix2sym;
    fn sub(self, rhs: matrix2sym) -> matrix2sym {
        let [a0, a1] = self.rows;
        let [b0, b1] = rhs.rows;
        matrix2sym { rows: [a0 - b0, a1 - b1] }
    }
}

impl ops::Neg for matrix2sym {
    type Output = matrix2sym;
    fn neg(self) -> matrix2sym {
        let [r0, r1] = self.rows;
        matrix2sym { rows: [-r0, -r1] }
    }
}

impl ops::Mul<E> for matrix2sym {
    type Output = matrix2sym;
    fn mul(self, rhs: E) -> matrix2sym {
        let [r0, r1] = self.rows;
        matrix2sym { rows: [r0 * rhs.clone(), r1 * rhs] }
    }
}

impl ops::Mul<matrix2sym> for E {
    type Output = matrix2sym;
    fn mul(self, rhs: matrix2sym) -> matrix2sym {
        rhs * self
    }
}

impl ops::Mul<matrix2sym> for vect2sym {
    type Output = vect2sym;
    /// Row vector times matrix: `v * M = M^T * v`.
    fn mul(self, rhs: matrix2sym) -> vect2sym {
        rhs.transpose() * self
    }
}

impl ops::Mul<matrix2sym> for matrix2sym {
    type Output = matrix2sym;
    fn mul(self, rhs: matrix2sym) -> matrix2sym {
        let rhs_t = rhs.transpose();
        matrix2sym {
            rows: [
                vect2sym {
                    x: self.rows[0].clone() * rhs_t.rows[0].clone(),
                    y: self.rows[0].clone() * rhs_t.rows[1].clone(),
                },
                vect2sym {
                    x: self.rows[1].clone() * rhs_t.rows[0].clone(),
                    y: self.rows[1].clone() * rhs_t.rows[1].clone(),
                },
            ],
        }
    }
}

impl ops::Mul<vect2sym> for matrix2sym {
    type Output = vect2sym;
    fn mul(self, rhs: vect2sym) -> vect2sym {
        vect2sym {
            x: self.rows[0].x.clone() * rhs.x.clone() + self.rows[0].y.clone() * rhs.y.clone(),
            y: self.rows[1].x.clone() * rhs.x.clone() + self.rows[1].y.clone() * rhs.y.clone(),
        }
    }
}

impl ops::Index<usize> for matrix2sym {
    type Output = vect2sym;
    fn index(&self, index: usize) -> &vect2sym {
        &self.rows[index]
    }
}

// ---------------------------------------------------------------------------
// quaternsym
// ---------------------------------------------------------------------------

/// Symbolic quaternion with scalar part `t` and vector part `v` (x, y, z).
#[derive(Clone)]
pub struct quaternsym {
    /// Scalar (real) component.
    pub t: E,
    /// Vector (imaginary) components.
    pub v: vect3sym,
}

impl quaternsym {
    /// Create a symbolic quaternion. Components are `{base}.t` (scalar) and
    /// `{base}.v.x`, `{base}.v.y`, `{base}.v.z` (vector).
    pub fn new(base: &str) -> Self {
        quaternsym {
            t: symbol(&format!("{}.t", base)),
            v: vect3sym::new(&format!("{}.v", base)),
        }
    }

    /// Identity quaternion (1 + 0i + 0j + 0k), representing no rotation.
    pub fn identity() -> quaternsym {
        quaternsym {
            t: crate::constant(1.0),
            v: vect3sym {
                x: crate::constant(0.0),
                y: crate::constant(0.0),
                z: crate::constant(0.0),
            },
        }
    }

    /// Unit quaternion from Euler angles (x=roll, y=pitch, z=yaw).
    /// Mirrors `quatern::from_euler_angles`.
    pub fn from_euler_angles(ea: &vect3sym) -> quaternsym {
        let half = crate::constant(0.5);
        let (shax, chax) = (sin(ea.x.clone() * half.clone()), cos(ea.x.clone() * half.clone()));
        let (shay, chay) = (sin(ea.y.clone() * half.clone()), cos(ea.y.clone() * half.clone()));
        let (shaz, chaz) = (sin(ea.z.clone() * half.clone()), cos(ea.z.clone() * half));
        quaternsym {
            t: chax.clone() * chay.clone() * chaz.clone()
                + shax.clone() * shay.clone() * shaz.clone(),
            v: vect3sym {
                x: shax.clone() * chay.clone() * chaz.clone()
                    - chax.clone() * shay.clone() * shaz.clone(),
                y: chax.clone() * shay.clone() * chaz.clone()
                    + shax.clone() * chay.clone() * shaz.clone(),
                z: chax * chay * shaz - shax * shay * chaz,
            },
        }
    }

    /// Unit quaternion from a rotation axis (must be unit length) and a
    /// symbolic angle. Mirrors `quatern::from_axis_angle`.
    pub fn from_axis_angle(axis: &vect3sym, angle: E) -> quaternsym {
        let half_angle = angle * crate::constant(0.5);
        quaternsym {
            t: cos(half_angle.clone()),
            v: axis.clone() * sin(half_angle),
        }
    }

    /// Dot product of two quaternions.
    pub fn dot(&self, q: &quaternsym) -> E {
        self.t.clone() * q.t.clone() + self.v.clone() * q.v.clone()
    }

    /// Norm (magnitude).
    pub fn norm(&self) -> E {
        crate::sqrt(self.dot(self))
    }

    /// Unit (normalized) quaternion.
    pub fn unit(self) -> quaternsym {
        let n = self.norm();
        quaternsym { t: self.t / n.clone(), v: self.v / n }
    }

    /// Conjugate (negated vector part).
    pub fn conj(&self) -> quaternsym {
        quaternsym { t: self.t.clone(), v: -self.v.clone() }
    }

    /// Rotate a 3D vector by this unit quaternion: `q * (0, v) * q'`.
    pub fn rotate(&self, v: &vect3sym) -> vect3sym {
        let pure = quaternsym { t: crate::constant(0.0), v: v.clone() };
        (self.clone() * pure * self.conj()).v
    }

    /// The equivalent 3x3 rotation matrix of this unit quaternion.
    /// Mirrors `quatern::rotation_matrix`.
    pub fn rotation_matrix(&self) -> matrix3sym {
        let one = || crate::constant(1.0);
        let two = || crate::constant(2.0);
        let (t, v) = (&self.t, &self.v);
        let x2 = v.x.clone() * v.x.clone();
        let y2 = v.y.clone() * v.y.clone();
        let z2 = v.z.clone() * v.z.clone();
        matrix3sym {
            rows: [
                vect3sym {
                    x: one() - two() * (y2.clone() + z2.clone()),
                    y: two() * (v.x.clone() * v.y.clone() - v.z.clone() * t.clone()),
                    z: two() * (v.x.clone() * v.z.clone() + v.y.clone() * t.clone()),
                },
                vect3sym {
                    x: two() * (v.x.clone() * v.y.clone() + v.z.clone() * t.clone()),
                    y: one() - two() * (x2.clone() + z2),
                    z: two() * (v.y.clone() * v.z.clone() - v.x.clone() * t.clone()),
                },
                vect3sym {
                    x: two() * (v.x.clone() * v.z.clone() - v.y.clone() * t.clone()),
                    y: two() * (v.y.clone() * v.z.clone() + v.x.clone() * t.clone()),
                    z: one() - two() * (x2 + y2),
                },
            ],
        }
    }

    /// Extract Euler angles (x=roll, y=pitch, z=yaw) from a unit quaternion.
    /// Continuous form of `quatern::get_euler_angles`: safe_asin clamps the
    /// pitch argument instead of branching at the gimbal boundary.
    pub fn get_euler_angles(&self) -> vect3sym {
        let one = || crate::constant(1.0);
        let two = || crate::constant(2.0);
        let (t, v) = (&self.t, &self.v);
        vect3sym {
            x: crate::atan2(
                two() * (t.clone() * v.x.clone() + v.y.clone() * v.z.clone()),
                one() - two() * (v.x.clone() * v.x.clone() + v.y.clone() * v.y.clone())),
            y: crate::safe_asin(two() * (t.clone() * v.y.clone() - v.z.clone() * v.x.clone())),
            z: crate::atan2(
                two() * (t.clone() * v.z.clone() + v.x.clone() * v.y.clone()),
                one() - two() * (v.y.clone() * v.y.clone() + v.z.clone() * v.z.clone())),
        }
    }
}

impl ops::Add<quaternsym> for quaternsym {
    type Output = quaternsym;
    fn add(self, rhs: quaternsym) -> quaternsym {
        quaternsym { t: self.t + rhs.t, v: self.v + rhs.v }
    }
}

impl ops::Sub<quaternsym> for quaternsym {
    type Output = quaternsym;
    fn sub(self, rhs: quaternsym) -> quaternsym {
        quaternsym { t: self.t - rhs.t, v: self.v - rhs.v }
    }
}

impl ops::Neg for quaternsym {
    type Output = quaternsym;
    fn neg(self) -> quaternsym {
        quaternsym { t: -self.t, v: -self.v }
    }
}

impl ops::Mul<E> for quaternsym {
    type Output = quaternsym;
    fn mul(self, rhs: E) -> quaternsym {
        quaternsym { t: self.t * rhs.clone(), v: self.v * rhs }
    }
}

impl ops::Mul<quaternsym> for E {
    type Output = quaternsym;
    fn mul(self, rhs: quaternsym) -> quaternsym {
        rhs * self
    }
}

impl ops::Mul<quaternsym> for quaternsym {
    type Output = quaternsym;
    /// Hamilton product. Mirrors the runtime `quatern` multiplication.
    fn mul(self, rhs: quaternsym) -> quaternsym {
        quaternsym {
            t: self.t.clone() * rhs.t.clone() - self.v.clone() * rhs.v.clone(),
            v: rhs.v.clone() * self.t
                + self.v.clone() * rhs.t
                + self.v % rhs.v,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vect3sym_cross_norm_unit_square_div() {
        use std::collections::HashMap;
        let a = vect3sym::new("a");
        let b = vect3sym::new("b");
        let vars = HashMap::from([
            ("a.x", 1.0), ("a.y", 2.0), ("a.z", 3.0),
            ("b.x", 4.0), ("b.y", 5.0), ("b.z", -6.0),
        ]);
        let ev = |e: &E| e.eval(&vars).unwrap();

        // cross: (1,2,3) x (4,5,-6) = (-27, 18, -3), and `%` is the same op
        let c = a.cross(&b);
        assert_eq!((ev(&c.x), ev(&c.y), ev(&c.z)), (-27.0, 18.0, -3.0));
        let r = a.clone() % b.clone();
        assert_eq!((ev(&r.x), ev(&r.y), ev(&r.z)), (-27.0, 18.0, -3.0));

        assert_eq!(ev(&a.square()), 14.0);
        assert!((ev(&a.norm()) - 14.0_f64.sqrt()).abs() < 1e-12);

        let u = a.clone().unit();
        let n = 14.0_f64.sqrt();
        assert!((ev(&u.x) - 1.0 / n).abs() < 1e-12);
        assert!((ev(&u.y) - 2.0 / n).abs() < 1e-12);
        assert!((ev(&u.z) - 3.0 / n).abs() < 1e-12);

        let d = a.clone() / crate::constant(2.0);
        assert_eq!((ev(&d.x), ev(&d.y), ev(&d.z)), (0.5, 1.0, 1.5));
    }

    #[test]
    fn matrix_constructors_det_and_row_vector_mul() {
        use std::collections::HashMap;
        let vars = HashMap::from([
            ("m[0].x", 2.0), ("m[0].y", 1.0), ("m[0].z", 3.0),
            ("m[1].x", 7.0), ("m[1].y", 5.0), ("m[1].z", 6.0),
            ("m[2].x", -5.0), ("m[2].y", 0.0), ("m[2].z", 1.0),
            ("v.x", 2.0), ("v.y", -1.0), ("v.z", 5.0),
        ]);
        let ev = |e: &E| e.eval(&vars).unwrap();

        // det matches the runtime matrix3 test fixture (det = 48).
        let m = matrix3sym::new("m");
        assert_eq!(ev(&m.det()), 48.0);

        // v * M = M^T v: first component is v dot col(0).
        let v = vect3sym::new("v");
        let vm = v.clone() * m.clone();
        assert_eq!(ev(&vm.x), 2.0 * 2.0 + (-1.0) * 7.0 + 5.0 * (-5.0));

        // identity leaves a vector unchanged; scalar mul scales det by k^3...
        let iv = matrix3sym::identity() * v.clone();
        assert_eq!((ev(&iv.x), ev(&iv.y), ev(&iv.z)), (2.0, -1.0, 5.0));

        // from_cols(transpose rows) == transpose; element (0,1) of
        // from_cols(r0, r1, r2) is r1.x.
        let fc = matrix3sym::from_cols(m.rows[0].clone(), m.rows[1].clone(), m.rows[2].clone());
        assert_eq!(ev(&fc.rows[0].y), 7.0);

        // 2x2: det, get_rotation_angle round trip, add/sub/neg/scalar mul.
        let a = crate::constant(0.7);
        let r = matrix2sym::rotation(a.clone());
        assert!((ev(&r.det()) - 1.0).abs() < 1e-12);
        assert!((ev(&r.get_rotation_angle()) - 0.7).abs() < 1e-12);
        let two = crate::constant(2.0);
        let s = (r.clone() * two) - r.clone() - r.clone(); // = 0 matrix
        assert_eq!(ev(&s.rows[0].x), 0.0);
        let n = -matrix2sym::identity();
        assert_eq!(ev(&n.rows[1].y), -1.0);
    }

    #[test]
    fn matrix3sym_rotation_from_axis_angle_matches_z_axis_rotation() {
        use std::collections::HashMap;
        // Rotation about the z axis must reduce to the 2D rotation block.
        let axis = vect3sym::new("ax");
        let r = matrix3sym::rotation_from_axis_angle(&axis, symbol("phi"));
        let vars = HashMap::from([
            ("ax.x", 0.0), ("ax.y", 0.0), ("ax.z", 1.0),
            ("phi", 0.7_f64),
        ]);
        let ev = |e: &E| e.eval(&vars).unwrap();
        assert!((ev(&r.rows[0].x) - 0.7_f64.cos()).abs() < 1e-12);
        assert!((ev(&r.rows[0].y) + 0.7_f64.sin()).abs() < 1e-12);
        assert!((ev(&r.rows[1].x) - 0.7_f64.sin()).abs() < 1e-12);
        assert!((ev(&r.rows[2].z) - 1.0).abs() < 1e-12);
        assert!((ev(&r.det()) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn quaternsym_algebra_matches_quaternion_identities() {
        use std::collections::HashMap;
        // Unit quaternion for axis z, angle 0.8: t = cos(0.4), v = (0,0,sin(0.4)).
        let vars = HashMap::from([
            ("q.t", 0.4_f64.cos()), ("q.v.x", 0.0), ("q.v.y", 0.0), ("q.v.z", 0.4_f64.sin()),
            ("p.x", 1.0), ("p.y", 2.0), ("p.z", 3.0),
        ]);
        let q = quaternsym::new("q");
        let p = vect3sym::new("p");
        let ev = |e: &E| e.eval(&vars).unwrap();

        // Rotating p about z by 0.8 rad.
        let (s, c) = 0.8_f64.sin_cos();
        let r = q.rotate(&p);
        assert!((ev(&r.x) - (c * 1.0 - s * 2.0)).abs() < 1e-12);
        assert!((ev(&r.y) - (s * 1.0 + c * 2.0)).abs() < 1e-12);
        assert!((ev(&r.z) - 3.0).abs() < 1e-12);

        // rotation_matrix agrees with rotate.
        let m = q.rotation_matrix();
        let mv = m * p.clone();
        assert!((ev(&mv.x) - ev(&r.x)).abs() < 1e-12);
        assert!((ev(&mv.y) - ev(&r.y)).abs() < 1e-12);

        // q * q' = identity; norm = 1; dot(q, q) = 1.
        let qq = q.clone() * q.conj();
        assert!((ev(&qq.t) - 1.0).abs() < 1e-12);
        assert!(ev(&qq.v.x).abs() < 1e-12);
        assert!((ev(&q.norm()) - 1.0).abs() < 1e-12);

        // get_euler_angles of a pure yaw rotation: (0, 0, 0.8).
        let ea = q.get_euler_angles();
        assert!(ev(&ea.x).abs() < 1e-12);
        assert!(ev(&ea.y).abs() < 1e-12);
        assert!((ev(&ea.z) - 0.8).abs() < 1e-12);

        // Constructors: from_axis_angle(z, 0.8) reproduces q's components;
        // from_euler_angles((0,0,0.8)) does too; identity rotates nothing.
        let axis = vect3sym::new("ax");
        let vars2 = HashMap::from([
            ("ax.x", 0.0), ("ax.y", 0.0), ("ax.z", 1.0), ("ang", 0.8_f64),
            ("e.x", 0.0), ("e.y", 0.0), ("e.z", 0.8),
        ]);
        let fa = quaternsym::from_axis_angle(&axis, symbol("ang"));
        assert!((fa.t.eval(&vars2).unwrap() - 0.4_f64.cos()).abs() < 1e-12);
        assert!((fa.v.z.eval(&vars2).unwrap() - 0.4_f64.sin()).abs() < 1e-12);
        let fe = quaternsym::from_euler_angles(&vect3sym::new("e"));
        assert!((fe.t.eval(&vars2).unwrap() - 0.4_f64.cos()).abs() < 1e-12);
        assert!((fe.v.z.eval(&vars2).unwrap() - 0.4_f64.sin()).abs() < 1e-12);
        let iv = quaternsym::identity().rotate(&p);
        assert!((ev(&iv.x) - 1.0).abs() < 1e-12);
        assert!((ev(&iv.y) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn matrix3sym_get_euler_angles_clamps_noisy_gimbal_boundary() {
        // Same regression as the runtime matrix3::get_euler_angles: the
        // emitted pitch expression must clamp an out-of-range (2,0)
        // entry instead of producing NaN.
        use std::collections::HashMap;
        let m = matrix3sym::new("m");
        let ea = m.get_euler_angles();
        let vars = HashMap::from([("m[2].x", -1.0000000001_f64)]);
        let v = ea.y.eval(&vars).unwrap();
        assert!(v.is_finite(), "pitch must be finite, got {} from {}", v, ea.y);
        assert!((v - std::f64::consts::FRAC_PI_2).abs() < 1e-6, "pitch={}", v);
    }

    #[test]
    fn matrix2sym_rotation_components() {
        // R(a) = [[cos a, -sin a], [sin a, cos a]]
        let r = matrix2sym::rotation(symbol("a"));
        assert_eq!(format!("{}", r.rows[0].x), "cos(a)");
        assert_eq!(format!("{}", r.rows[0].y), "-sin(a)");
        assert_eq!(format!("{}", r.rows[1].x), "sin(a)");
        assert_eq!(format!("{}", r.rows[1].y), "cos(a)");
    }

    #[test]
    fn matrix2sym_transpose_swaps_off_diagonal() {
        let r = matrix2sym::rotation(symbol("a"));
        let rt = r.transpose();
        // R^T(a) = [[cos a, sin a], [-sin a, cos a]] = R(-a)
        assert_eq!(format!("{}", rt.rows[0].x), "cos(a)");
        assert_eq!(format!("{}", rt.rows[0].y), "sin(a)");
        assert_eq!(format!("{}", rt.rows[1].x), "-sin(a)");
        assert_eq!(format!("{}", rt.rows[1].y), "cos(a)");
    }

    #[test]
    fn matrix2sym_mul_vect2sym_applies_rotation() {
        // R(a) * v = (cos(a)*v.x - sin(a)*v.y, sin(a)*v.x + cos(a)*v.y)
        let r = matrix2sym::rotation(symbol("a"));
        let v = vect2sym::new("v");
        let rv = r * v;
        assert_eq!(format!("{}", rv.x.simplify()),
            "v.x * cos(a) - v.y * sin(a)");
        assert_eq!(format!("{}", rv.y.simplify()),
            "v.x * sin(a) + v.y * cos(a)");
    }

    #[test]
    fn matrix2sym_mul_matrix2sym_yields_2d_composition() {
        // R(a) * R(b) -- the (0,0) entry is cos(a)*cos(b) - sin(a)*sin(b),
        // i.e. the expanded form of cos(a+b). We don't rely on a trig
        // identity collapser; just check the bilinear structure.
        let ra = matrix2sym::rotation(symbol("a"));
        let rb = matrix2sym::rotation(symbol("b"));
        let prod = ra * rb;
        assert_eq!(format!("{}", prod.rows[0].x.simplify()),
            "cos(a) * cos(b) - sin(a) * sin(b)");
        // (1, 0) = sin(a)*cos(b) + cos(a)*sin(b) = sin(a+b)
        assert_eq!(format!("{}", prod.rows[1].x.simplify()),
            "cos(b) * sin(a) + cos(a) * sin(b)");
    }
}
