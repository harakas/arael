//! A 2D angle parameter that caches the sin/cos of its angle. Set `angle`
//! before a solve, read it after; constraint bodies read `rotation_matrix`
//! (the 2x2 rotation, built from the cached sin/cos) and `angle` (the raw
//! scalar).
//!
//! ```ignore
//! struct Pose2 {
//!     rot: AngleParam,          // 1 param in the owner's span
//!     hb: SelfBlock<Pose2>,
//! }
//! // constraint body: b.rot.rotation_matrix.transpose() * (... a.rot.rotation_matrix * d ...)
//! //                  rad_diff(a.rot.angle + dth, b.rot.angle)
//! ```
//!
//! The 2D analog of [`SimpleEulerAngleParam`](crate::model::SimpleEulerAngleParam):
//! the angle is optimized directly (no gimbal lock in one dimension, so no
//! reference frame). The `sin`/`cos` of the angle are computed once per entity
//! in the assembly precompute; a constraint that rotates through the angle then
//! reads those two cached scalars instead of rebuilding `sin`/`cos` once per
//! observing edge. Both the rotation matrix `R(angle)` and its Jacobian
//! `dR/d(angle)` are built from the same two scalars -- no stored matrix, no
//! re-derived trig.

use crate::matrix::matrix2;
use crate::model::{Model, ModelSym, Param};
use crate::utils::Float;

/// A 2D rotation parameter: an angle plus its cached sin/cos. Constraint
/// bodies read `rotation_matrix` (computed from the cached scalars); Rust
/// callers can build it with [`rotation_matrix`](AngleParam::rotation_matrix).
/// See the module docs for the body-read semantics.
#[derive(Clone)]
pub struct AngleParam<T: Float = f64> {
    /// The optimizable angle in radians. Set it before a solve, read it
    /// after; constraint bodies read it directly as a scalar.
    pub angle: Param<T>,
    /// Cached `sin(angle)`; the rotation matrix and its Jacobian read it (pub
    /// so generated constraint code can). Refreshed each iteration; treat as
    /// read-only.
    pub sin: T,
    /// Cached `cos(angle)`; the rotation matrix and its Jacobian read it (pub
    /// so generated constraint code can). Refreshed each iteration; treat as
    /// read-only.
    pub cos: T,
}

/// Shorthand for the f32 instantiation; `AngleParam<f32>` works spelled out
/// too.
pub type AngleParamF = AngleParam<f32>;

impl<T: Float> AngleParam<T>
where
    Param<T>: Model,
{
    /// An angle parameter starting at `angle` radians.
    pub fn new(angle: T) -> AngleParam<T> {
        let mut p = AngleParam {
            angle: Param::new(angle),
            sin: T::zero(),
            cos: T::one(),
        };
        p.__precompute_symbolic();
        p
    }

    /// An angle frozen at `angle` (excluded from optimization).
    pub fn fixed(angle: T) -> AngleParam<T> {
        let mut p = AngleParam::new(angle);
        p.angle = Param::fixed(angle);
        p
    }

    /// The 2x2 rotation matrix at the current angle, built from the cached
    /// sin/cos. Constraint bodies read the field `rotation_matrix` (the same
    /// value); this is the Rust-caller accessor.
    pub fn rotation_matrix(&self) -> matrix2<T> {
        matrix2::rotation_from_sincos(self.sin, self.cos)
    }

    /// The hand-written twin of the generated symbolic precompute: `sin` and
    /// `cos` of the current angle. The rotation matrix and its Jacobian are
    /// both built from these two scalars, so nothing else is cached here.
    #[doc(hidden)]
    pub fn __precompute_symbolic(&mut self) {
        let (s, c) = self.angle.work().sin_cos();
        self.sin = s;
        self.cos = c;
    }
}

impl<T: Float> Default for AngleParam<T>
where
    Param<T>: Model,
{
    fn default() -> Self {
        AngleParam::new(T::zero())
    }
}

// The param folds into the owner's span exactly like a plain `Param<T>`;
// each state-changing path refreshes the cached sin/cos.
impl<T: Float> Model for AngleParam<T>
where
    Param<T>: Model,
{
    const PARAM_COUNT: u32 = 1;

    fn serialize_params<F: Float>(&mut self, data: &mut std::vec::Vec<F>) {
        Model::serialize_params(&mut self.angle, data);
    }
    fn deserialize_params<F: Float>(&mut self, data: &[F]) {
        Model::deserialize_params(&mut self.angle, data);
        Model::update_self(self);
    }
    fn update_params<F: Float>(&mut self, data: &[F]) {
        Model::update_params(&mut self.angle, data);
        self.__precompute_symbolic();
    }
    fn update_self(&mut self) {
        Model::update_self(&mut self.angle);
        self.__precompute_symbolic();
    }

    fn advance_params<F: Float>(&mut self, params: &mut [F]) {
        Model::advance_params(&mut self.angle, params);
    }

    fn serialize_size(&self) -> u32 {
        Model::serialize_size(&self.angle)
    }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        <Param<T> as Model>::param_symbols(&format!("{}.angle", base), out);
    }
}

// Bodies read an AngleParam through its `.rotation_matrix` and `.angle`
// (expanded by the macro's builtin layout), so the type-level Sym is the
// scalar angle's -- a single named symbol.
impl<T: Float> ModelSym for AngleParam<T> {
    type Sym = arael_sym::E;
    fn sym(base: &str) -> Self::Sym {
        arael_sym::symbol(base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_matches_rotation() {
        // The matrix built from the cached sin/cos equals matrix2::rotation.
        let mut p = AngleParam::<f64>::new(0.7);
        let mut data: std::vec::Vec<f64> = std::vec::Vec::new();
        Model::serialize_params(&mut p, &mut data); // assigns the index
        Model::update_self(&mut p);                   // work = value, precompute
        let r = matrix2::<f64>::rotation(0.7);
        let m = p.rotation_matrix();
        for k in 0..2 {
            assert!((m[k] - r[k]).norm() < 1e-15);
        }
    }
}
