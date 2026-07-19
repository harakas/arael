//! Unit-vector (S^2 direction) parameter -- a 2-DOF direction on the unit
//! sphere, the first in-tree `#[arael(component)]`-style compound parameter.
//!
//! The chart is a unit quaternion `ref_q` whose x-axis is the direction; the
//! solver sees a 2-DOF body-frame rotation delta about the frame's y and z
//! axes (rotating about the direction itself is the unobservable fiber and
//! is simply not a parameter). Constraint bodies read `.unit` as the
//! direction, expanded symbolically to the rotated first column of the
//! small-rotation matrix -- exact on the sphere for every trial delta, so a
//! step can never leave the manifold. `advance` folds the accepted delta
//! into the reference and re-centers (delta back to zero).
//!
//! Contract, matching the rotation params: `unit` is the initial direction
//! in and the optimized direction out, synced by `deserialize` after a
//! solve; the reference frame is solver-internal. Two equal directions may
//! hold different internal frames (the roll about the direction is hidden,
//! meaningless state).
//!
//! ```ignore
//! struct Landmark {
//!     dir: UnitVecParam,        // 2 params in the owner's span
//!     hb: SelfBlock<Landmark>,
//! }
//! // constraint body: [lm.dir.unit % measured] -- reads expand through the
//! // chart with exact derivatives.
//! ```

use crate::model::{Component, Model, Param, ParamType};
use crate::matrix::matrix3;
use crate::quatern::quatern;
use crate::vect::{vect2, vect3};

/// A unit direction with a 2-DOF tangent parameterization. See the module
/// docs for the chart and the body-read semantics.
pub struct UnitVecParam {
    /// The chart: unit quaternion, x-axis = direction. Solver-internal.
    ref_q: quatern<f64>,
    /// Cached `ref_q.rotation_matrix()`; the symbolic embed reads it as
    /// constants (pub so generated constraint code can). Refreshed whenever
    /// the reference moves; treat as read-only.
    pub rot: matrix3<f64>,
    /// The tangent delta (body-frame rotation about y and z). Zero-centred;
    /// fix it to freeze the direction.
    pub d: Param<vect2<f64>>,
    /// User-facing direction: set it (any nonzero length) before a solve,
    /// read it after `deserialize`.
    pub unit: vect3<f64>,
}

impl UnitVecParam {
    /// A direction parameter starting at `dir` (normalized here).
    pub fn new(dir: vect3<f64>) -> UnitVecParam {
        let mut p = UnitVecParam {
            ref_q: quatern::identity(),
            rot: matrix3::identity(),
            d: Param::new(vect2::new(0.0, 0.0)),
            unit: dir,
        };
        Component::start(&mut p);
        p
    }

    /// A direction parameter frozen at `dir` (excluded from optimization).
    pub fn fixed(dir: vect3<f64>) -> UnitVecParam {
        let mut p = UnitVecParam::new(dir);
        p.d = Param::fixed(vect2::new(0.0, 0.0));
        p
    }

    fn ex() -> vect3<f64> {
        vect3::new(1.0, 0.0, 0.0)
    }

    fn refresh(&mut self) {
        self.rot = self.ref_q.rotation_matrix();
    }
}

impl Default for UnitVecParam {
    fn default() -> Self {
        UnitVecParam::new(Self::ex())
    }
}

impl Component for UnitVecParam {
    fn start(&mut self) {
        self.unit = self.unit.unit();
        self.ref_q = quatern::from_two_vectors(Self::ex(), self.unit);
        self.refresh();
        self.d.value = vect2::new(0.0, 0.0);
    }

    fn update(&mut self) {
        let dq = quatern::from_rotation_vector_small(vect3::new(
            0.0,
            self.d.value.x,
            self.d.value.y,
        ));
        self.ref_q = (self.ref_q * dq).unit();
        self.refresh();
        self.d.value = vect2::new(0.0, 0.0);
    }

    fn finish(&mut self) {
        // The delta is normally zero here (advance re-centres on every
        // accepted step); fold it anyway so a hand-driven deserialize is
        // exact too.
        let dq = quatern::from_rotation_vector_small(vect3::new(
            0.0,
            self.d.value.x,
            self.d.value.y,
        ));
        self.unit = (self.ref_q * dq).rotate(Self::ex());
    }
}

// The hand-written twin of what a `#[arael(component)]` expansion emits:
// lifecycle hooks around the param recursion, and the advance shuttle
// (pull the accepted step, re-center, push the reset values back).
impl Model for UnitVecParam {
    const PARAM_COUNT: u32 = 2;

    fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
        Component::start(self);
        Model::serialize_params32(&mut self.d, data);
    }
    fn deserialize_params32(&mut self, data: &[f32]) {
        Model::deserialize_params32(&mut self.d, data);
        Component::finish(self);
    }
    fn update32(&mut self, data: &[f32]) {
        Model::update32(&mut self.d, data);
    }
    fn update_self(&mut self) {
        Model::update_self(&mut self.d);
    }

    fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
        Component::start(self);
        Model::serialize_params64(&mut self.d, data);
    }
    fn deserialize_params64(&mut self, data: &[f64]) {
        Model::deserialize_params64(&mut self.d, data);
        Component::finish(self);
    }
    fn update64(&mut self, data: &[f64]) {
        Model::update64(&mut self.d, data);
    }

    fn serialize_size(&self) -> u32 {
        Model::serialize_size(&self.d)
    }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        <Param<vect2<f64>> as Model>::param_symbols(&format!("{}.d", base), out);
    }

    fn advance_params32(&mut self, params: &mut [f32]) {
        Model::deserialize_params32(&mut self.d, params);
        Component::update(self);
        if self.d.index() != u32::MAX {
            let i = self.d.index() as usize;
            ParamType::write_to32(&self.d.value, &mut params[i..i + 2]);
        }
    }
    fn advance_params64(&mut self, params: &mut [f64]) {
        Model::deserialize_params64(&mut self.d, params);
        Component::update(self);
        if self.d.index() != u32::MAX {
            let i = self.d.index() as usize;
            ParamType::write_to64(&self.d.value, &mut params[i..i + 2]);
        }
    }
}

// The symbolic companion: bodies read a UnitVecParam through its `.unit`
// (expanded by the macro's builtin layout), so the type-level Sym is the
// direction's -- a vect3 of named symbols.
impl crate::model::ModelSym for UnitVecParam {
    type Sym = <vect3<f64> as crate::model::ModelSym>::Sym;
    fn sym(base: &str) -> Self::Sym {
        <vect3<f64> as crate::model::ModelSym>::sym(base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chart_and_embed_agree() {
        // The runtime retraction and the symbolic embed formula must be the
        // same map: q(1, (0,dx,dy)/2).rotate(ex) equals the closed form the
        // macro seeds (first column of the small-rotation matrix).
        let mut p = UnitVecParam::new(vect3::new(0.3, -0.8, 0.51));
        for (dx, dy) in [(0.0, 0.0), (0.2, -0.1), (-1.3, 0.7)] {
            let s2 = 1.0 + (dx * dx + dy * dy) * 0.25;
            let local = vect3::new(1.0 - (dx * dx + dy * dy) / (2.0 * s2), dy / s2, -dx / s2);
            let closed = p.rot * local;
            let dq = quatern::from_rotation_vector_small(vect3::new(0.0, dx, dy));
            let direct = (p.ref_q * dq).rotate(UnitVecParam::ex());
            assert!((closed - direct).norm() < 1e-14, "({}, {})", dx, dy);
            // Exact on the sphere for every delta, small or not.
            assert!((closed.norm() - 1.0).abs() < 1e-14);
        }
        // Re-centering folds and resets.
        p.d.value = vect2::new(0.2, -0.4);
        let before = (p.ref_q
            * quatern::from_rotation_vector_small(vect3::new(0.0, 0.2, -0.4)))
        .rotate(UnitVecParam::ex());
        Component::update(&mut p);
        assert!((p.rot * vect3::new(1.0, 0.0, 0.0) - before).norm() < 1e-14);
        assert_eq!(p.d.value.x, 0.0);
        assert_eq!(p.d.value.y, 0.0);
    }
}
