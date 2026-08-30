//! Rigid transform parameter: translation and rotation optimized
//! together.
//!
//! Set `translation` and `rotation` before a solve, read them after.
//! Constraint bodies read `translation` and `rotation_matrix` (the cached
//! form of the same rotation).
//!
//! Name the field for the frames it maps between, and the type stays out
//! of the way -- a robot pose, a sensor mount and a map alignment are all
//! the same kind of thing:
//!
//! ```ignore
//! struct Frame {
//!     r2w: TransformParam,      // 6 params in the owner's span
//!     hb: SelfBlock<Frame>,
//! }
//! // constraint body: frame.r2w.rotation_matrix, frame.r2w.translation
//! ```
//!
//! The alternative is a `Param<vect3>` position beside a
//! [`QuaternionParam`](crate::model::QuaternionParam) -- six numbers
//! either way, but there the two are stepped independently. Here they move
//! as one: the solver steps in [twist](crate::se3) coordinates, so a step
//! that turns while it moves follows the arc a body traces rather than the
//! chord. The two agree for small steps and part company for large ones.
//!
//! That matters on long trajectories with large corrections. On a
//! 1000-pose plane-SLAM loop this converges in 8 iterations where the
//! independent form takes 28 and still stops short of the optimum; per
//! iteration the two cost the same. On problems whose corrections stay
//! small it changes nothing measurable -- two 3D pose graphs (sphere2500,
//! parking-garage) ran iteration for iteration identical -- so it is worth
//! reaching for when a solve is struggling on a long loop, not
//! everywhere.
//!
//! Freeze either half on its own with `optimize_translation` /
//! `optimize_rotation`.

use crate::matrix::matrix3;
use crate::model::{Component, Model, Param, ParamType};
use crate::quatern::quatern;
use crate::se3::{carry, se3};
use crate::utils::Float;
use crate::vect::vect3;

/// A pose with a coupled 6-DOF step. See the module docs.
#[derive(Clone)]
pub struct TransformParam<T: Float = f64>
where
    vect3<T>: ParamType,
{
    /// Translation: set it before a solve, read it after.
    pub translation: vect3<T>,
    /// Rotation: set it before a solve, read it after.
    pub rotation: quatern<T>,
    /// Whether the translation is optimized. Clear it to hold the
    /// translation while the rotation stays free, or the other way round.
    pub optimize_translation: bool,
    /// Whether the rotation is optimized.
    pub optimize_rotation: bool,

    /// The rotation as a matrix -- what constraint bodies read. Refreshed
    /// with the transform; treat as read-only.
    #[doc(hidden)]
    pub rotation_matrix: matrix3<T>,

    // --- solver-internal: the reference frame the step is measured from,
    // the step itself, and the Jacobian caches generated code reads.
    #[doc(hidden)]
    pub ref_translation: vect3<T>,
    #[doc(hidden)]
    pub ref_rotation: matrix3<T>,
    /// The step's two halves: `w` rotation, `d` translation. Solver
    /// internal -- freeze through the `optimize_*` flags, not these.
    #[doc(hidden)]
    pub w: Param<vect3<T>>,
    #[doc(hidden)]
    pub d: Param<vect3<T>>,
    /// `d(rotation_matrix)/dw`, `d(pos)/dd` and `d(pos)/dw`, refreshed with the pose;
    /// each is skipped while its half of the step is frozen.
    #[doc(hidden)]
    pub rotation_matrix_dw: [matrix3<T>; 3],
    #[doc(hidden)]
    pub translation_dd: [vect3<T>; 3],
    #[doc(hidden)]
    pub translation_dw: [vect3<T>; 3],

    ref_value: quatern<T>,
}

/// Shorthand for the f32 instantiation; `TransformParam<f32>` works
/// spelled out too.
pub type TransformParamF = TransformParam<f32>;

impl<T: Float> TransformParam<T>
where
    vect3<T>: ParamType,
    Param<vect3<T>>: Model,
{
    /// A pose starting at `pos` / `rotation`, fully optimized.
    pub fn new(translation: vect3<T>, rotation: quatern<T>) -> TransformParam<T> {
        let zero = vect3::new(T::zero(), T::zero(), T::zero());
        let mut p = TransformParam {
            translation,
            rotation,
            optimize_translation: true,
            optimize_rotation: true,
            rotation_matrix: rotation.rotation_matrix(),
            ref_translation: translation,
            ref_rotation: matrix3::identity(),
            w: Param::new(zero),
            d: Param::new(zero),
            rotation_matrix_dw: [matrix3::identity(); 3],
            translation_dd: [zero; 3],
            translation_dw: [zero; 3],
            ref_value: rotation,
        };
        Component::start(&mut p);
        p.__precompute_symbolic();
        p
    }

    /// A pose frozen where it starts (neither half optimized).
    pub fn fixed(translation: vect3<T>, rotation: quatern<T>) -> TransformParam<T> {
        let mut p = TransformParam::new(translation, rotation);
        p.optimize_translation = false;
        p.optimize_rotation = false;
        p
    }

    fn refresh(&mut self) {
        self.ref_rotation = self.ref_value.rotation_matrix();
    }

    /// The hand-written twin of the generated symbolic precompute: the
    /// pose at the current step, and its Jacobian (each half skipped while
    /// that half of the step is frozen -- nothing reads it then).
    #[doc(hidden)]
    pub fn __precompute_symbolic(&mut self) {
        let (w, d) = (self.w.work(), self.d.work());
        self.rotation_matrix = self.ref_rotation * matrix3::<T>::from_rotation_vector_small(w);
        self.translation = self.ref_translation + self.ref_rotation * carry(w, d);
        if self.w.index() != u32::MAX {
            // The composed rotation is linear in the retraction, so
            // d(R_ref R(w))/dw.k = R_ref dR(w)/dw.k -- the same identity
            // QuaternionParam uses.
            let dr = matrix3::<T>::from_rotation_vector_small_deriv(w);
            self.rotation_matrix_dw = [self.ref_rotation * dr[0],
                            self.ref_rotation * dr[1],
                            self.ref_rotation * dr[2]];
            // d(carry)/dw.k = (e_k x d)/2 + (e_k x (w x d) + w x (e_k x d))/6
            let wxd = w % d;
            for k in 0..3 {
                let e = basis::<T>(k);
                let t = (e % d) * cf::<T>(0.5)
                    + ((e % wxd) + (w % (e % d))) * cf::<T>(1.0 / 6.0);
                self.translation_dw[k] = self.ref_rotation * t;
            }
        }
        if self.d.index() != u32::MAX {
            // d(carry)/dd.k is the carry applied to the basis vector.
            for k in 0..3 {
                self.translation_dd[k] = self.ref_rotation * carry(w, basis::<T>(k));
            }
        }
    }
}

fn cf<T: Float>(x: f64) -> T {
    T::from(x).unwrap()
}

fn basis<T: Float>(k: usize) -> vect3<T> {
    match k {
        0 => vect3::new(T::one(), T::zero(), T::zero()),
        1 => vect3::new(T::zero(), T::one(), T::zero()),
        _ => vect3::new(T::zero(), T::zero(), T::one()),
    }
}

impl<T: Float> Default for TransformParam<T>
where
    vect3<T>: ParamType,
    Param<vect3<T>>: Model,
{
    fn default() -> Self {
        TransformParam::new(vect3::new(T::zero(), T::zero(), T::zero()), quatern::identity())
    }
}

impl<T: Float> Component for TransformParam<T>
where
    vect3<T>: ParamType,
    Param<vect3<T>>: Model,
{
    fn start(&mut self) {
        self.ref_value = self.rotation.unit();
        self.ref_translation = self.translation;
        self.refresh();
        let zero = vect3::new(T::zero(), T::zero(), T::zero());
        self.w.value = zero;
        self.d.value = zero;
        // The two flags are the single source of truth for what moves.
        self.d.optimize = self.optimize_translation;
        self.w.optimize = self.optimize_rotation;
    }

    fn update(&mut self) {
        // The accepted step folds into the reference. The step is in the
        // pose's own frame, so its translation is taken there too.
        let (t, q) = se3::new(self.d.value, self.w.value).translation_rotation();
        self.ref_translation = self.ref_translation + self.ref_rotation * t;
        self.ref_value = (self.ref_value * q).unit();
        self.refresh();
        let zero = vect3::new(T::zero(), T::zero(), T::zero());
        self.w.value = zero;
        self.d.value = zero;
    }

    fn finish(&mut self) {
        // The step is normally zero here (advance re-centres on every
        // accepted step); fold it anyway so a hand-driven deserialize is
        // exact too.
        let (t, q) = se3::new(self.d.value, self.w.value).translation_rotation();
        self.translation = self.ref_translation + self.ref_rotation * t;
        self.rotation = (self.ref_value * q).unit();
        self.rotation_matrix = self.rotation.rotation_matrix();
    }
}

// The hand-written twin of what a `#[arael(component)]` expansion emits:
// lifecycle hooks around the param recursion, and the advance shuttle
// (pull the accepted step, re-center, push the reset values back).
impl<T: Float> Model for TransformParam<T>
where
    vect3<T>: ParamType,
    Param<vect3<T>>: Model,
{
    const PARAM_COUNT: u32 = 6;

    fn serialize_params<F: Float>(&mut self, data: &mut std::vec::Vec<F>) {
        Component::start(self);
        Model::serialize_params(&mut self.w, data);
        Model::serialize_params(&mut self.d, data);
    }
    fn deserialize_params<F: Float>(&mut self, data: &[F]) {
        Model::deserialize_params(&mut self.w, data);
        Model::deserialize_params(&mut self.d, data);
        Component::finish(self);
        Model::update_self(self);
    }
    fn update_params<F: Float>(&mut self, data: &[F]) {
        Model::update_params(&mut self.w, data);
        Model::update_params(&mut self.d, data);
        self.__precompute_symbolic();
    }
    fn update_self(&mut self) {
        Model::update_self(&mut self.w);
        Model::update_self(&mut self.d);
        self.__precompute_symbolic();
    }

    fn serialize_size(&self) -> u32 {
        Model::serialize_size(&self.w) + Model::serialize_size(&self.d)
    }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        <Param<vect3<T>> as Model>::param_symbols(&format!("{}.w", base), out);
        <Param<vect3<T>> as Model>::param_symbols(&format!("{}.d", base), out);
    }

    fn advance_params<F: Float>(&mut self, params: &mut [F]) {
        Model::deserialize_params(&mut self.w, params);
        Model::deserialize_params(&mut self.d, params);
        Component::update(self);
        for p in [&self.w, &self.d] {
            if p.index() != u32::MAX {
                let i = p.index() as usize;
                ParamType::write_to(&p.value, &mut params[i..i + 3]);
            }
        }
    }
}

// The symbolic companion. Bodies read a TransformParam through `.rotation_matrix` and
// `.pos`, both expanded by the macro's builtin layout, so the type-level
// Sym is unused shape -- the position's.
impl<T: Float> crate::model::ModelSym for TransformParam<T>
where
    vect3<T>: ParamType,
{
    type Sym = <vect3<f64> as crate::model::ModelSym>::Sym;
    fn sym(base: &str) -> Self::Sym {
        <vect3<f64> as crate::model::ModelSym>::sym(base)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TransformParam<f64> {
        let q = (quatern::from_axis_angle(vect3::new(0.0, 0.0, 1.0), 0.7)
            * quatern::from_axis_angle(vect3::new(1.0, 0.0, 0.0), -0.3))
        .unit();
        let mut p = TransformParam::new(vect3::new(1.5, -0.4, 2.0), q);
        // A serialize pass gives the step live indices, so the guarded
        // Jacobian caches fill.
        Model::serialize_params(&mut p, &mut std::vec::Vec::<f64>::new());
        p
    }

    /// Every cached Jacobian against finite differences of the cached
    /// values, driven through the real update cycle.
    #[test]
    fn jacobian_caches_match_finite_differences() {
        let mut p = sample();
        let at = |p: &mut TransformParam<f64>, w: vect3<f64>, d: vect3<f64>| {
            p.w.value = w;
            p.d.value = d;
            Model::update_self(p);
        };
        for (w0, d0) in [
            (vect3::new(0.0, 0.0, 0.0), vect3::new(0.0, 0.0, 0.0)),
            (vect3::new(0.08, -0.05, 0.11), vect3::new(0.3, -0.2, 0.15)),
        ] {
            at(&mut p, w0, d0);
            let (rot_dw, tr_dw, tr_dd) =
                (p.rotation_matrix_dw, p.translation_dw, p.translation_dd);
            let eps = 1e-7;
            for k in 0..3 {
                let e = basis::<f64>(k) * eps;
                at(&mut p, w0 + e, d0);
                let (rp, tp) = (p.rotation_matrix, p.translation);
                at(&mut p, w0 - e, d0);
                let (rm, tm) = (p.rotation_matrix, p.translation);
                for r in 0..3 {
                    let fd = (rp[r] - rm[r]) * (1.0 / (2.0 * eps));
                    assert!((rot_dw[k][r] - fd).norm() < 1e-6,
                        "rotation_matrix_dw[{}] row {}", k, r);
                }
                let fd = (tp - tm) * (1.0 / (2.0 * eps));
                assert!((tr_dw[k] - fd).norm() < 1e-6, "translation_dw[{}]", k);

                at(&mut p, w0, d0 + e);
                let tp = p.translation;
                at(&mut p, w0, d0 - e);
                let tm = p.translation;
                let fd = (tp - tm) * (1.0 / (2.0 * eps));
                assert!((tr_dd[k] - fd).norm() < 1e-6, "translation_dd[{}]", k);
            }
        }
    }

    /// Folding an accepted step into the reference must land the transform
    /// exactly where the step said it would, or the solver would evaluate
    /// one transform and keep another.
    #[test]
    fn re_centering_preserves_the_transform() {
        let mut p = sample();
        p.w.value = vect3::new(0.05, -0.03, 0.09);
        p.d.value = vect3::new(0.4, 0.1, -0.25);
        Model::update_self(&mut p);
        let (want_t, want_r) = (p.translation, p.rotation_matrix);

        Component::update(&mut p);
        assert!(p.w.value.norm() < 1e-15 && p.d.value.norm() < 1e-15, "re-centred");
        Model::update_self(&mut p);
        assert!((p.translation - want_t).norm() < 1e-14, "translation moved on re-centre");
        for r in 0..3 {
            assert!((p.rotation_matrix[r] - want_r[r]).norm() < 1e-14,
                "rotation moved on re-centre");
        }
    }

    /// The user-facing rotation comes back as a quaternion, in sync with
    /// the matrix constraints read.
    #[test]
    fn rotation_reads_back_as_a_quaternion() {
        let mut p = sample();
        p.w.value = vect3::new(0.05, -0.03, 0.09);
        p.d.value = vect3::new(0.4, 0.1, -0.25);
        Model::update_self(&mut p);
        let want = p.rotation_matrix;
        Component::finish(&mut p);
        let from_q = p.rotation.rotation_matrix();
        for r in 0..3 {
            assert!((from_q[r] - want[r]).norm() < 1e-13,
                "rotation disagrees with rotation_matrix on row {}", r);
        }
    }

    /// Each half freezes on its own: the frozen one contributes no
    /// parameters and does not move.
    #[test]
    fn halves_freeze_independently() {
        let q = quatern::from_axis_angle(vect3::new(0.0, 1.0, 0.0), 0.4).unit();
        let t0 = vect3::new(3.0, -1.0, 0.5);

        let mut both = TransformParam::new(t0, q);
        let mut data = std::vec::Vec::<f64>::new();
        Model::serialize_params(&mut both, &mut data);
        assert_eq!(data.len(), 6, "both halves free");

        let mut rot_only = TransformParam::new(t0, q);
        rot_only.optimize_translation = false;
        let mut data = std::vec::Vec::<f64>::new();
        Model::serialize_params(&mut rot_only, &mut data);
        assert_eq!(data.len(), 3, "translation frozen");
        rot_only.w.value = vect3::new(0.1, 0.2, -0.15);
        Model::update_self(&mut rot_only);
        assert!((rot_only.translation - t0).norm() < 1e-14, "frozen translation moved");

        let mut trans_only = TransformParam::new(t0, q);
        trans_only.optimize_rotation = false;
        let mut data = std::vec::Vec::<f64>::new();
        Model::serialize_params(&mut trans_only, &mut data);
        assert_eq!(data.len(), 3, "rotation frozen");
        trans_only.d.value = vect3::new(0.3, -0.1, 0.2);
        Model::update_self(&mut trans_only);
        for r in 0..3 {
            assert!((trans_only.rotation_matrix[r] - q.rotation_matrix()[r]).norm() < 1e-14,
                "frozen rotation moved");
        }

        let mut fixed = TransformParam::fixed(t0, q);
        let mut data = std::vec::Vec::<f64>::new();
        Model::serialize_params(&mut fixed, &mut data);
        assert!(data.is_empty(), "a fixed transform serializes no params");
    }
}

// ---------------------------------------------------------------------------
// serde: the transform and its two flags, nothing else. The reference frame,
// the step, and every Jacobian cache are rebuilt on load exactly as `new`
// builds them -- they are derived from the transform, and a file carrying
// them would only be a way to disagree with it.
// ---------------------------------------------------------------------------

impl<T: Float> serde::Serialize for TransformParam<T>
where
    vect3<T>: ParamType,
    T: serde::Serialize,
{
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("TransformParam", 4)?;
        st.serialize_field("translation", &self.translation)?;
        st.serialize_field("rotation", &self.rotation)?;
        st.serialize_field("optimize_translation", &self.optimize_translation)?;
        st.serialize_field("optimize_rotation", &self.optimize_rotation)?;
        st.end()
    }
}

impl<'de, T: Float + serde::Deserialize<'de>> serde::Deserialize<'de> for TransformParam<T>
where
    vect3<T>: ParamType,
{
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, Visitor};
        struct V<U>(std::marker::PhantomData<U>);
        impl<'de2, U: Float + serde::Deserialize<'de2>> Visitor<'de2> for V<U>
        where
            vect3<U>: ParamType,
        {
            type Value = TransformParam<U>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("TransformParam")
            }
            fn visit_map<A: MapAccess<'de2>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let (mut t, mut r, mut ot, mut orr) = (None, None, None, None);
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "translation" => t = Some(map.next_value()?),
                        "rotation" => r = Some(map.next_value()?),
                        "optimize_translation" => ot = Some(map.next_value()?),
                        "optimize_rotation" => orr = Some(map.next_value()?),
                        _ => { let _ = map.next_value::<de::IgnoredAny>()?; }
                    }
                }
                let translation = t.unwrap_or_else(||
                    vect3::new(U::zero(), U::zero(), U::zero()));
                let rotation = r.unwrap_or_else(quatern::<U>::identity).unit();
                let mut p = TransformParam::new(translation, rotation);
                // Set before re-centring: the flags decide which half of the
                // step is live, and the caches follow from that.
                p.optimize_translation = ot.unwrap_or(true);
                p.optimize_rotation = orr.unwrap_or(true);
                Component::start(&mut p);
                p.__precompute_symbolic();
                Ok(p)
            }
        }
        d.deserialize_map(V(std::marker::PhantomData))
    }
}

// ---------------------------------------------------------------------------
// ScaledTransformParam: a similarity transform (Sim(3) state)
// ---------------------------------------------------------------------------

/// A similarity transform: [`TransformParam`]'s coupled pose step plus a
/// uniform scale, acting on a point as `s * (R * x) + t`. The Sim(3)
/// state of monocular loop closing.
///
/// Set `translation`, `rotation` and `scale` before a solve, read them
/// after. Constraint bodies read `rotation_matrix`, `translation` and
/// `scale_factor` (the cached scale). The scale is optimized as its
/// logarithm, so positivity is structural and a scale-difference
/// residual `log_s_a - log_s_b - log(s_ab)` is linear; freeze it alone
/// with `optimize_scale` (the fix-scale mode of stereo/RGB-D loop
/// closing).
#[derive(Clone)]
pub struct ScaledTransformParam<T: Float = f64>
where
    vect3<T>: ParamType,
{
    /// Translation: set it before a solve, read it after.
    pub translation: vect3<T>,
    /// Rotation: set it before a solve, read it after.
    pub rotation: quatern<T>,
    /// Scale: set it before a solve, read it after. Must be positive.
    pub scale: T,
    /// Whether the translation is optimized.
    pub optimize_translation: bool,
    /// Whether the rotation is optimized.
    pub optimize_rotation: bool,
    /// Whether the scale is optimized.
    pub optimize_scale: bool,

    /// The rotation as a matrix -- what constraint bodies read.
    #[doc(hidden)]
    pub rotation_matrix: matrix3<T>,
    /// The scale -- what constraint bodies read. Refreshed with the
    /// transform; treat as read-only.
    #[doc(hidden)]
    pub scale_factor: T,

    // --- solver-internal (see TransformParam)
    #[doc(hidden)]
    pub ref_translation: vect3<T>,
    #[doc(hidden)]
    pub ref_rotation: matrix3<T>,
    #[doc(hidden)]
    pub w: Param<vect3<T>>,
    #[doc(hidden)]
    pub d: Param<vect3<T>>,
    /// The log-scale parameter (absolute, not a delta).
    #[doc(hidden)]
    pub log_s: Param<T>,
    #[doc(hidden)]
    pub rotation_matrix_dw: [matrix3<T>; 3],
    #[doc(hidden)]
    pub translation_dd: [vect3<T>; 3],
    #[doc(hidden)]
    pub translation_dw: [vect3<T>; 3],

    ref_value: quatern<T>,
}

/// Shorthand for the f32 instantiation.
pub type ScaledTransformParamF = ScaledTransformParam<f32>;

impl<T: Float> ScaledTransformParam<T>
where
    vect3<T>: ParamType,
    Param<vect3<T>>: Model,
{
    /// A similarity starting at `translation` / `rotation` / `scale`,
    /// fully optimized.
    pub fn new(translation: vect3<T>, rotation: quatern<T>, scale: T) -> ScaledTransformParam<T> {
        let zero = vect3::new(T::zero(), T::zero(), T::zero());
        let mut p = ScaledTransformParam {
            translation,
            rotation,
            scale,
            optimize_translation: true,
            optimize_rotation: true,
            optimize_scale: true,
            rotation_matrix: rotation.rotation_matrix(),
            scale_factor: scale,
            ref_translation: translation,
            ref_rotation: matrix3::identity(),
            w: Param::new(zero),
            d: Param::new(zero),
            log_s: Param::new(scale.ln()),
            rotation_matrix_dw: [matrix3::identity(); 3],
            translation_dd: [zero; 3],
            translation_dw: [zero; 3],
            ref_value: rotation,
        };
        Component::start(&mut p);
        p.__precompute_symbolic();
        p
    }

    /// A similarity frozen where it starts (nothing optimized).
    pub fn fixed(translation: vect3<T>, rotation: quatern<T>, scale: T) -> ScaledTransformParam<T> {
        let mut p = ScaledTransformParam::new(translation, rotation, scale);
        p.optimize_translation = false;
        p.optimize_rotation = false;
        p.optimize_scale = false;
        Component::start(&mut p);
        p
    }

    fn refresh(&mut self) {
        self.ref_rotation = self.ref_value.rotation_matrix();
    }

    /// The hand-written twin of the generated symbolic precompute: the
    /// pose caches as in [`TransformParam`], plus the scale.
    #[doc(hidden)]
    pub fn __precompute_symbolic(&mut self) {
        let (w, d) = (self.w.work(), self.d.work());
        self.rotation_matrix = self.ref_rotation * matrix3::<T>::from_rotation_vector_small(w);
        self.translation = self.ref_translation + self.ref_rotation * carry(w, d);
        self.scale_factor = self.log_s.work().exp();
        if self.w.index() != u32::MAX {
            let dr = matrix3::<T>::from_rotation_vector_small_deriv(w);
            self.rotation_matrix_dw = [self.ref_rotation * dr[0],
                            self.ref_rotation * dr[1],
                            self.ref_rotation * dr[2]];
            let wxd = w % d;
            for k in 0..3 {
                let e = basis::<T>(k);
                let t = (e % d) * cf::<T>(0.5)
                    + ((e % wxd) + (w % (e % d))) * cf::<T>(1.0 / 6.0);
                self.translation_dw[k] = self.ref_rotation * t;
            }
        }
        if self.d.index() != u32::MAX {
            for k in 0..3 {
                self.translation_dd[k] = self.ref_rotation * carry(w, basis::<T>(k));
            }
        }
    }
}

impl<T: Float> Default for ScaledTransformParam<T>
where
    vect3<T>: ParamType,
    Param<vect3<T>>: Model,
{
    fn default() -> Self {
        ScaledTransformParam::new(
            vect3::new(T::zero(), T::zero(), T::zero()),
            quatern::identity(),
            T::one(),
        )
    }
}

impl<T: Float> Component for ScaledTransformParam<T>
where
    vect3<T>: ParamType,
    Param<vect3<T>>: Model,
{
    fn start(&mut self) {
        self.ref_value = self.rotation.unit();
        self.ref_translation = self.translation;
        self.refresh();
        let zero = vect3::new(T::zero(), T::zero(), T::zero());
        self.w.value = zero;
        self.d.value = zero;
        self.log_s.value = self.scale.ln();
        self.scale_factor = self.scale;
        self.d.optimize = self.optimize_translation;
        self.w.optimize = self.optimize_rotation;
        self.log_s.optimize = self.optimize_scale;
    }

    fn update(&mut self) {
        let (t, q) = se3::new(self.d.value, self.w.value).translation_rotation();
        self.ref_translation = self.ref_translation + self.ref_rotation * t;
        self.ref_value = (self.ref_value * q).unit();
        self.refresh();
        let zero = vect3::new(T::zero(), T::zero(), T::zero());
        self.w.value = zero;
        self.d.value = zero;
        self.scale_factor = self.log_s.value.exp();
    }

    fn finish(&mut self) {
        let (t, q) = se3::new(self.d.value, self.w.value).translation_rotation();
        self.translation = self.ref_translation + self.ref_rotation * t;
        self.rotation = (self.ref_value * q).unit();
        self.rotation_matrix = self.rotation.rotation_matrix();
        self.scale = self.log_s.value.exp();
        self.scale_factor = self.scale;
    }
}

impl<T: Float> Model for ScaledTransformParam<T>
where
    vect3<T>: ParamType,
    Param<vect3<T>>: Model,
{
    const PARAM_COUNT: u32 = 7;

    fn serialize_params<F: Float>(&mut self, data: &mut std::vec::Vec<F>) {
        Component::start(self);
        Model::serialize_params(&mut self.w, data);
        Model::serialize_params(&mut self.d, data);
        Model::serialize_params(&mut self.log_s, data);
    }
    fn deserialize_params<F: Float>(&mut self, data: &[F]) {
        Model::deserialize_params(&mut self.w, data);
        Model::deserialize_params(&mut self.d, data);
        Model::deserialize_params(&mut self.log_s, data);
        Component::finish(self);
        Model::update_self(self);
    }
    fn update_params<F: Float>(&mut self, data: &[F]) {
        Model::update_params(&mut self.w, data);
        Model::update_params(&mut self.d, data);
        Model::update_params(&mut self.log_s, data);
        self.__precompute_symbolic();
    }
    fn update_self(&mut self) {
        Model::update_self(&mut self.w);
        Model::update_self(&mut self.d);
        Model::update_self(&mut self.log_s);
        self.__precompute_symbolic();
    }

    fn serialize_size(&self) -> u32 {
        Model::serialize_size(&self.w)
            + Model::serialize_size(&self.d)
            + Model::serialize_size(&self.log_s)
    }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        <Param<vect3<T>> as Model>::param_symbols(&format!("{}.w", base), out);
        <Param<vect3<T>> as Model>::param_symbols(&format!("{}.d", base), out);
        <Param<T> as Model>::param_symbols(&format!("{}.log_s", base), out);
    }

    fn advance_params<F: Float>(&mut self, params: &mut [F]) {
        // w/d re-center on the chart; log_s is absolute and needs none.
        Model::deserialize_params(&mut self.w, params);
        Model::deserialize_params(&mut self.d, params);
        Model::deserialize_params(&mut self.log_s, params);
        Component::update(self);
        for p in [&self.w, &self.d] {
            if p.index() != u32::MAX {
                let i = p.index() as usize;
                ParamType::write_to(&p.value, &mut params[i..i + 3]);
            }
        }
    }
}

impl<T: Float> crate::model::ModelSym for ScaledTransformParam<T>
where
    vect3<T>: ParamType,
{
    type Sym = <vect3<f64> as crate::model::ModelSym>::Sym;
    fn sym(base: &str) -> Self::Sym {
        <vect3<f64> as crate::model::ModelSym>::sym(base)
    }
}
