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
}

#[cfg(test)]
mod tests {
    use super::*;

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
