//! A unit-vector parameter: a direction on the unit sphere, optimized with
//! 2 degrees of freedom. Set `unit` before a solve, read it after -- it is
//! always a unit vector, and a step never leaves the sphere.
//!
//! ```ignore
//! struct Landmark {
//!     dir: UnitVecParam,        // 2 params in the owner's span
//!     hb: SelfBlock<Landmark>,
//! }
//! // constraint body: [lm.dir.unit % measured] -- reads expand with exact
//! // derivatives.
//! ```
//!
//! The two degrees of freedom are a body-frame rotation of a reference
//! direction about the two axes perpendicular to it; rotating about the
//! direction itself does nothing and is not a parameter. The reference is
//! solver-internal, so two equal directions may hold different internal
//! frames. On each accepted step the delta folds into the reference and
//! re-centers to zero. Constraint bodies read `.unit`, which carries the
//! embed and its exact Jacobian.
//!
//! Equivalent to a user-defined `#[arael(component)]` with the same fields
//! (see `examples/plane_slam_demo.rs` for that form); this is the
//! hand-written in-tree version.

use crate::model::{Component, Model, Param, ParamType};
use crate::matrix::matrix3;
use crate::quatern::quatern;
use crate::utils::Float;
use crate::vect::{vect2, vect3};

/// A unit direction with a 2-DOF tangent parameterization. See the module
/// docs for the chart and the body-read semantics.
#[derive(Clone)]
pub struct UnitVecParam<T: Float = f64>
where
    vect2<T>: ParamType,
{
    /// The chart: unit quaternion, x-axis = direction. Solver-internal.
    ref_q: quatern<T>,
    /// Cached `ref_q.rotation_matrix()`; the symbolic embed reads it as
    /// constants (pub so generated constraint code can). Refreshed whenever
    /// the reference moves; treat as read-only.
    pub rot: matrix3<T>,
    /// The tangent delta (body-frame rotation about y and z). Zero-centred;
    /// fix it to freeze the direction.
    pub d: Param<vect2<T>>,
    /// User-facing direction: set it (any nonzero length) before a solve,
    /// read it after `deserialize`. During a solve it holds the embed at the
    /// current delta, refreshed by the update paths.
    pub unit: vect3<T>,
    /// Jacobian cache `[d(unit)/d(d.x), d(unit)/d(d.y)]`, read by constraint
    /// Jacobians; refreshed with `unit` (skipped while `d` is fixed).
    pub unit_d: [vect3<T>; 2],
}

/// The f32 twin, by name: the macro classifies fields by the type path's
/// last segment, so f32 models spell the alias.
pub type UnitVecParamF = UnitVecParam<f32>;

fn c<T: Float>(x: f64) -> T {
    T::from(x).unwrap()
}

impl<T: Float> UnitVecParam<T>
where
    vect2<T>: ParamType,
    Param<vect2<T>>: Model,
{
    /// A direction parameter starting at `dir` (normalized here).
    pub fn new(dir: vect3<T>) -> UnitVecParam<T> {
        let mut p = UnitVecParam {
            ref_q: quatern::identity(),
            rot: matrix3::identity(),
            d: Param::new(vect2::new(T::zero(), T::zero())),
            unit: dir,
            unit_d: [vect3::new(T::zero(), T::zero(), T::zero()); 2],
        };
        Component::start(&mut p);
        p.__precompute_symbolic();
        p
    }

    /// A direction parameter frozen at `dir` (excluded from optimization).
    pub fn fixed(dir: vect3<T>) -> UnitVecParam<T> {
        let mut p = UnitVecParam::new(dir);
        p.d = Param::fixed(vect2::new(T::zero(), T::zero()));
        p
    }

    fn ex() -> vect3<T> {
        vect3::new(T::one(), T::zero(), T::zero())
    }

    fn refresh(&mut self) {
        self.rot = self.ref_q.rotation_matrix();
    }

    /// The hand-written twin of the generated symbolic precompute: the
    /// chart embed at the current delta into `unit`, and its Jacobian into
    /// `unit_d` (skipped while `d` is fixed -- nothing reads it then).
    #[doc(hidden)]
    pub fn __precompute_symbolic(&mut self) {
        let d = self.d.work();
        let u = d.x * d.x + d.y * d.y;
        let s2 = T::one() + u * c(0.25);
        let local = vect3::new(
            T::one() - u / (c::<T>(2.0) * s2),
            d.y / s2,
            -d.x / s2,
        );
        self.unit = self.rot * local;
        if self.d.index() != u32::MAX {
            let inv2 = T::one() / (s2 * s2);
            let half = c::<T>(0.5);
            let dlocal_dx = vect3::new(
                -d.x * inv2,
                -(d.x * d.y * half) * inv2,
                -(s2 - d.x * d.x * half) * inv2,
            );
            let dlocal_dy = vect3::new(
                -d.y * inv2,
                (s2 - d.y * d.y * half) * inv2,
                (d.x * d.y * half) * inv2,
            );
            self.unit_d = [self.rot * dlocal_dx, self.rot * dlocal_dy];
        }
    }
}

impl<T: Float> Default for UnitVecParam<T>
where
    vect2<T>: ParamType,
    Param<vect2<T>>: Model,
{
    fn default() -> Self {
        UnitVecParam::new(Self::ex())
    }
}

impl<T: Float> Component for UnitVecParam<T>
where
    vect2<T>: ParamType,
    Param<vect2<T>>: Model,
{
    fn start(&mut self) {
        self.unit = self.unit.unit();
        self.ref_q = quatern::from_two_vectors(Self::ex(), self.unit);
        self.refresh();
        self.d.value = vect2::new(T::zero(), T::zero());
    }

    fn update(&mut self) {
        let dq = quatern::from_rotation_vector_small(vect3::new(
            T::zero(),
            self.d.value.x,
            self.d.value.y,
        ));
        self.ref_q = (self.ref_q * dq).unit();
        self.refresh();
        self.d.value = vect2::new(T::zero(), T::zero());
    }

    fn finish(&mut self) {
        // The delta is normally zero here (advance re-centres on every
        // accepted step); fold it anyway so a hand-driven deserialize is
        // exact too.
        let dq = quatern::from_rotation_vector_small(vect3::new(
            T::zero(),
            self.d.value.x,
            self.d.value.y,
        ));
        self.unit = (self.ref_q * dq).rotate(Self::ex());
    }
}

// The hand-written twin of what a `#[arael(component)]` expansion emits:
// lifecycle hooks around the param recursion, and the advance shuttle
// (pull the accepted step, re-center, push the reset values back).
impl<T: Float> Model for UnitVecParam<T>
where
    vect2<T>: ParamType,
    Param<vect2<T>>: Model,
{
    const PARAM_COUNT: u32 = 2;

    fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
        Component::start(self);
        Model::serialize_params32(&mut self.d, data);
    }
    fn deserialize_params32(&mut self, data: &[f32]) {
        Model::deserialize_params32(&mut self.d, data);
        Component::finish(self);
        // Sync the working copy before precomputing: it still holds the
        // last trial's delta, and the embed must match what finish wrote.
        Model::update_self(self);
    }
    fn update32(&mut self, data: &[f32]) {
        Model::update32(&mut self.d, data);
        self.__precompute_symbolic();
    }
    fn update_self(&mut self) {
        Model::update_self(&mut self.d);
        self.__precompute_symbolic();
    }

    fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
        Component::start(self);
        Model::serialize_params64(&mut self.d, data);
    }
    fn deserialize_params64(&mut self, data: &[f64]) {
        Model::deserialize_params64(&mut self.d, data);
        Component::finish(self);
        Model::update_self(self);
    }
    fn update64(&mut self, data: &[f64]) {
        Model::update64(&mut self.d, data);
        self.__precompute_symbolic();
    }

    fn serialize_size(&self) -> u32 {
        Model::serialize_size(&self.d)
    }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        <Param<vect2<T>> as Model>::param_symbols(&format!("{}.d", base), out);
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
// direction's -- a vect3 of named symbols, scalar-independent.
impl<T: Float> crate::model::ModelSym for UnitVecParam<T>
where
    vect2<T>: ParamType,
    Param<vect2<T>>: Model,
{
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
        let mut p = UnitVecParam::<f64>::new(vect3::new(0.3, -0.8, 0.51));
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

    #[test]
    fn jacobian_cache_matches_finite_differences() {
        let mut p = UnitVecParam::<f64>::new(vect3::new(0.3, -0.8, 0.51));
        // A serialize pass assigns the delta its live index, so the guarded
        // cache fills.
        Model::serialize_params64(&mut p, &mut std::vec::Vec::new());
        // Drive the proper cycle: value -> update_self syncs the working
        // copy and refreshes unit + unit_d.
        let mut at = |p: &mut UnitVecParam<f64>, dx: f64, dy: f64| -> (vect3<f64>, [vect3<f64>; 2]) {
            p.d.value = vect2::new(dx, dy);
            Model::update_self(p);
            (p.unit, p.unit_d)
        };
        for (dx, dy) in [(0.0, 0.0), (0.15, -0.35)] {
            let (_, cache) = at(&mut p, dx, dy);
            let eps = 1e-7;
            for k in 0..2 {
                let (ox, oy) = if k == 0 { (eps, 0.0) } else { (0.0, eps) };
                let (up, _) = at(&mut p, dx + ox, dy + oy);
                let (um, _) = at(&mut p, dx - ox, dy - oy);
                let fd = (up - um) * (1.0 / (2.0 * eps));
                assert!((cache[k] - fd).norm() < 1e-6,
                    "unit_d[{}] at ({}, {}): {:?} vs fd {:?}", k, dx, dy, cache[k], fd);
            }
        }
    }
}
