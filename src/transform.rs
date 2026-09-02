//! Poses as parameters: a rigid transform the solver moves as one
//! 6-DOF quantity, and its scaled variant.
//!
//! A pose is a rotation and a translation that map points from one
//! frame to another. [`TransformParam`] holds one as a single
//! parameter: the solver steps it in local SE(3) [twist](crate::se3)
//! coordinates, so a step that turns while it moves follows the arc a
//! body traces rather than the chord. [`ScaledTransformParam`] adds a
//! uniform scale, the Sim(3) state of monocular loop closing.
//!
//! The alternative would be a separate `Param<vect3d>` position and a
//! [`QuaternionParam`](crate::model::QuaternionParam) rotation. Six
//! numbers either way, but stepped independently, and the two forms
//! part company on large corrections. On a 1000-pose plane-SLAM loop
//! the coupled step converges in 8 iterations where the independent form
//! takes 28 and still stops short of the optimum; where corrections stay
//! small (the sphere2500 and parking-garage pose graphs) they run
//! iteration for iteration identical.
//!
//! Name the field for the frames it maps between and the type stays out
//! of the way; a robot pose, a sensor mount and a map alignment are all
//! the same kind of thing:
//!
//! ```ignore
//! #[arael::model]
//! #[arael(constraint(hb, {
//!     let x_w = pose.r2w * pose.x_r;             // the point, in the world frame
//!     [x_w.x - pose.m_w.x, x_w.y - pose.m_w.y, x_w.z - pose.m_w.z]
//! }))]
//! struct Pose {
//!     r2w: TransformParam,      // transform from robot to world frame
//!     x_r: vect3d,              // a point in the robot frame
//!     m_w: vect3d,              // where it was seen, in the world frame
//!     hb: SelfBlock<Pose>,
//! }
//! ```
//!
//! Set `translation` and `rotation` before a solve, read them after;
//! freeze either half on its own with `optimize_translation` /
//! `optimize_rotation`.
//!
//! In a constraint body the field is a transform value with the algebra
//! of a transform. With `r2w` the robot pose (`R`, `t`) and `c2r` a
//! camera mounted on the robot:
//!
//! | body spelling | what it does | the same by hand |
//! |---|---|---|
//! | `x_w = pose.r2w * x_r` | a robot-frame point, seen in the world | `R * x_r + t` |
//! | `x_r = pose.r2w.inv() * x_w` | a world point, seen from the robot | `R.transpose() * (x_w - t)` |
//! | `c2w = pose.r2w * pose.c2r` | the camera's pose in the world | composition: `c2r` followed by `r2w` |
//! | `b2a = a.r2w.inv() * b.r2w` | b's pose, seen from a | composition: `b.r2w` followed by `a.r2w.inv()` |
//! | `pose.r2w.rotate(v_r)` | a robot-frame vector, in the world | `R * v_r`, no translation |
//!
//! `transform` / `inverse_transform` / `rotate` / `inverse_rotate` are
//! the named forms of the point and direction rows. The generated code
//! is the hand-written column, reading the same cached
//! `rotation_matrix` and `translation` a body can still read directly.
//!
//! Host code gets the same surface at runtime, with the same names:
//! `pose.r2w.transform(x_r)`, `&pose.r2w * x_r`, `pose.r2w.inv()`,
//! `let c2w = &pose.r2w * &pose.c2r;`, `a.r2w.inv() * &b.r2w`. Inverses
//! and compositions are the plain values [`transform3`] (rigid) and
//! [`scaled_transform3`] (with a scale), which carry the same methods;
//! a rigid result stays rigid and never pays for a scale it does not
//! have.

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
        // update_self, not the bare precompute: it copies every param's
        // value into its working copy first, so a symbolic field of an
        // absolute param (`scale_factor = exp(log_s)`) is right from
        // construction, not from the first solve.
        Model::update_self(&mut p);
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
        // update_self, not the bare precompute: it copies every param's
        // value into its working copy first, so a symbolic field of an
        // absolute param (`scale_factor = exp(log_s)`) is right from
        // construction, not from the first solve.
        Model::update_self(&mut p);
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

// ---------------------------------------------------------------------------
// Transforms as runtime values
// ---------------------------------------------------------------------------

/// A rigid transform as a plain value, acting on a point as `R x + t`.
/// The runtime twin of what a constraint body gets from a
/// [`TransformParam`] through `*` and `inv()`: the param converts into
/// it, its `inv()` and rigid compositions return it, and the same method
/// names apply to both. With the frame convention `tr2w`,
/// `tr2w.transform(x_r)` is `x_w` and `tr2w.inv() * x_w` is `x_r`.
/// No scale anywhere: a rigid action costs the matrix product and the
/// add, nothing else.
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct transform3<T: Float> {
    /// The rotation as a matrix.
    pub rotation_matrix: matrix3<T>,
    /// The translation.
    pub translation: vect3<T>,
}

/// Shorthand for the f64 instantiation.
#[allow(non_camel_case_types)]
pub type transform3d = transform3<f64>;
/// Shorthand for the f32 instantiation.
#[allow(non_camel_case_types)]
pub type transform3f = transform3<f32>;

/// A scaled transform as a plain value, acting on a point as
/// `s (R x) + t`: the twin of [`ScaledTransformParam`], and what any
/// composition involving a scale returns. Same methods as
/// [`transform3`]; the inverse action divides once per call.
#[derive(Clone, Copy, Debug)]
#[allow(non_camel_case_types)]
pub struct scaled_transform3<T: Float> {
    /// The rotation as a matrix.
    pub rotation_matrix: matrix3<T>,
    /// The translation.
    pub translation: vect3<T>,
    /// The uniform scale.
    pub scale: T,
}

/// Shorthand for the f64 instantiation.
#[allow(non_camel_case_types)]
pub type scaled_transform3d = scaled_transform3<f64>;
/// Shorthand for the f32 instantiation.
#[allow(non_camel_case_types)]
pub type scaled_transform3f = scaled_transform3<f32>;

impl<T: Float> transform3<T> {
    /// From a translation and a rotation.
    pub fn new(translation: vect3<T>, rotation: quatern<T>) -> Self {
        transform3 { rotation_matrix: rotation.rotation_matrix(), translation }
    }

    /// The identity.
    pub fn identity() -> Self {
        let z = T::zero();
        transform3 { rotation_matrix: matrix3::identity(), translation: vect3::new(z, z, z) }
    }

    /// The rotation as a quaternion.
    pub fn rotation(&self) -> quatern<T> {
        quatern::from_rotation_matrix(self.rotation_matrix)
    }

    /// The action on a point: `R x + t`.
    pub fn transform(&self, x: vect3<T>) -> vect3<T> {
        self.rotation_matrix * x + self.translation
    }

    /// The action of the inverse: `R^T (y - t)`.
    pub fn inverse_transform(&self, y: vect3<T>) -> vect3<T> {
        self.rotation_matrix.transpose() * (y - self.translation)
    }

    /// The rotation alone: `R v`.
    pub fn rotate(&self, v: vect3<T>) -> vect3<T> {
        self.rotation_matrix * v
    }

    /// The inverse rotation alone: `R^T v`.
    pub fn inverse_rotate(&self, v: vect3<T>) -> vect3<T> {
        self.rotation_matrix.transpose() * v
    }

    /// The inverse transform: `(R^T, -R^T t)`.
    pub fn inv(&self) -> Self {
        let rt = self.rotation_matrix.transpose();
        transform3 { rotation_matrix: rt, translation: -(rt * self.translation) }
    }

    /// Composition `self * rhs`, applying `rhs` first:
    /// `(R_a R_b, R_a t_b + t_a)`.
    pub fn compose(&self, rhs: &Self) -> Self {
        transform3 {
            rotation_matrix: self.rotation_matrix * rhs.rotation_matrix,
            translation: self.rotation_matrix * rhs.translation + self.translation,
        }
    }
}

impl<T: Float> scaled_transform3<T> {
    /// From a translation, a rotation and a scale.
    pub fn new(translation: vect3<T>, rotation: quatern<T>, scale: T) -> Self {
        scaled_transform3 { rotation_matrix: rotation.rotation_matrix(), translation, scale }
    }

    /// The identity.
    pub fn identity() -> Self {
        transform3::identity().into()
    }

    /// The rotation as a quaternion.
    pub fn rotation(&self) -> quatern<T> {
        quatern::from_rotation_matrix(self.rotation_matrix)
    }

    /// The action on a point: `s (R x) + t`.
    pub fn transform(&self, x: vect3<T>) -> vect3<T> {
        self.rotation_matrix * x * self.scale + self.translation
    }

    /// The action of the inverse: `R^T (y - t) / s`, one division.
    pub fn inverse_transform(&self, y: vect3<T>) -> vect3<T> {
        let k = T::one() / self.scale;
        self.rotation_matrix.transpose() * (y - self.translation) * k
    }

    /// The rotation alone: `R v`, never scaled.
    pub fn rotate(&self, v: vect3<T>) -> vect3<T> {
        self.rotation_matrix * v
    }

    /// The inverse rotation alone: `R^T v`, never scaled.
    pub fn inverse_rotate(&self, v: vect3<T>) -> vect3<T> {
        self.rotation_matrix.transpose() * v
    }

    /// The inverse transform: `(R^T, -R^T t / s, 1 / s)`, one division.
    pub fn inv(&self) -> Self {
        let rt = self.rotation_matrix.transpose();
        let k = T::one() / self.scale;
        scaled_transform3 {
            rotation_matrix: rt,
            translation: -(rt * self.translation) * k,
            scale: k,
        }
    }

    /// Composition `self * rhs`, applying `rhs` first:
    /// `(R_a R_b, s_a (R_a t_b) + t_a, s_a s_b)`.
    pub fn compose(&self, rhs: &Self) -> Self {
        scaled_transform3 {
            rotation_matrix: self.rotation_matrix * rhs.rotation_matrix,
            translation: self.rotation_matrix * rhs.translation * self.scale + self.translation,
            scale: self.scale * rhs.scale,
        }
    }
}

impl<T: Float> From<transform3<T>> for scaled_transform3<T> {
    fn from(t: transform3<T>) -> Self {
        scaled_transform3 { rotation_matrix: t.rotation_matrix, translation: t.translation, scale: T::one() }
    }
}

impl<T: Float> std::ops::Mul<vect3<T>> for transform3<T> {
    type Output = vect3<T>;
    fn mul(self, x: vect3<T>) -> vect3<T> {
        self.transform(x)
    }
}

impl<T: Float> std::ops::Mul<transform3<T>> for transform3<T> {
    type Output = transform3<T>;
    fn mul(self, rhs: transform3<T>) -> transform3<T> {
        self.compose(&rhs)
    }
}

impl<T: Float> std::ops::Mul<scaled_transform3<T>> for transform3<T> {
    type Output = scaled_transform3<T>;
    fn mul(self, rhs: scaled_transform3<T>) -> scaled_transform3<T> {
        scaled_transform3::from(self).compose(&rhs)
    }
}

impl<T: Float> std::ops::Mul<vect3<T>> for scaled_transform3<T> {
    type Output = vect3<T>;
    fn mul(self, x: vect3<T>) -> vect3<T> {
        self.transform(x)
    }
}

impl<T: Float> std::ops::Mul<scaled_transform3<T>> for scaled_transform3<T> {
    type Output = scaled_transform3<T>;
    fn mul(self, rhs: scaled_transform3<T>) -> scaled_transform3<T> {
        self.compose(&rhs)
    }
}

impl<T: Float> std::ops::Mul<transform3<T>> for scaled_transform3<T> {
    type Output = scaled_transform3<T>;
    fn mul(self, rhs: transform3<T>) -> scaled_transform3<T> {
        self.compose(&rhs.into())
    }
}

/// The transform surface on a pose param, reading the cached
/// `rotation_matrix` / `translation` (and scale) that constraint bodies
/// read, so a host and a body see the same pose at every point of a
/// solve. `*` takes the param by reference: `&pose.r2w * x`,
/// `&a.r2w * &b.r2w`, `a.r2w.inv() * &b.r2w`.
macro_rules! transform_param_value_api {
    ($ty:ident, $val:ident, $to:ident, |$p:ident| $build:expr) => {
        impl<T: Float> $ty<T>
        where
            vect3<T>: ParamType,
        {
            /// The pose as a plain value.
            pub fn $to(&self) -> $val<T> {
                let $p = self;
                $build
            }

            /// The action on a point.
            pub fn transform(&self, x: vect3<T>) -> vect3<T> {
                self.$to().transform(x)
            }

            /// The action of the inverse.
            pub fn inverse_transform(&self, y: vect3<T>) -> vect3<T> {
                self.$to().inverse_transform(y)
            }

            /// The rotation alone: `R v`, never scaled.
            pub fn rotate(&self, v: vect3<T>) -> vect3<T> {
                self.rotation_matrix * v
            }

            /// The inverse rotation alone: `R^T v`, never scaled.
            pub fn inverse_rotate(&self, v: vect3<T>) -> vect3<T> {
                self.rotation_matrix.transpose() * v
            }

            /// The inverse transform as a value.
            pub fn inv(&self) -> $val<T> {
                self.$to().inv()
            }
        }

        impl<T: Float> From<&$ty<T>> for $val<T>
        where
            vect3<T>: ParamType,
        {
            fn from(p: &$ty<T>) -> $val<T> {
                p.$to()
            }
        }

        impl<T: Float> std::ops::Mul<vect3<T>> for &$ty<T>
        where
            vect3<T>: ParamType,
        {
            type Output = vect3<T>;
            fn mul(self, x: vect3<T>) -> vect3<T> {
                self.transform(x)
            }
        }

        impl<T: Float> std::ops::Mul<&$ty<T>> for &$ty<T>
        where
            vect3<T>: ParamType,
        {
            type Output = $val<T>;
            fn mul(self, rhs: &$ty<T>) -> $val<T> {
                self.$to().compose(&rhs.$to())
            }
        }

        impl<T: Float> std::ops::Mul<$val<T>> for &$ty<T>
        where
            vect3<T>: ParamType,
        {
            type Output = $val<T>;
            fn mul(self, rhs: $val<T>) -> $val<T> {
                self.$to().compose(&rhs)
            }
        }

        impl<T: Float> std::ops::Mul<&$ty<T>> for $val<T>
        where
            vect3<T>: ParamType,
        {
            type Output = $val<T>;
            fn mul(self, rhs: &$ty<T>) -> $val<T> {
                self.compose(&rhs.$to())
            }
        }
    };
}

transform_param_value_api!(TransformParam, transform3, to_transform, |p| transform3 {
    rotation_matrix: p.rotation_matrix,
    translation: p.translation,
});
transform_param_value_api!(ScaledTransformParam, scaled_transform3, to_scaled_transform,
    |p| scaled_transform3 {
        rotation_matrix: p.rotation_matrix,
        translation: p.translation,
        scale: p.scale_factor,
    });

// Rigid and scaled mixed, from either side: the result carries the scale.

impl<T: Float> std::ops::Mul<&ScaledTransformParam<T>> for &TransformParam<T>
where
    vect3<T>: ParamType,
{
    type Output = scaled_transform3<T>;
    fn mul(self, rhs: &ScaledTransformParam<T>) -> scaled_transform3<T> {
        scaled_transform3::from(self.to_transform()).compose(&rhs.to_scaled_transform())
    }
}

impl<T: Float> std::ops::Mul<&TransformParam<T>> for &ScaledTransformParam<T>
where
    vect3<T>: ParamType,
{
    type Output = scaled_transform3<T>;
    fn mul(self, rhs: &TransformParam<T>) -> scaled_transform3<T> {
        self.to_scaled_transform().compose(&rhs.to_transform().into())
    }
}

impl<T: Float> std::ops::Mul<scaled_transform3<T>> for &TransformParam<T>
where
    vect3<T>: ParamType,
{
    type Output = scaled_transform3<T>;
    fn mul(self, rhs: scaled_transform3<T>) -> scaled_transform3<T> {
        scaled_transform3::from(self.to_transform()).compose(&rhs)
    }
}

impl<T: Float> std::ops::Mul<&TransformParam<T>> for scaled_transform3<T>
where
    vect3<T>: ParamType,
{
    type Output = scaled_transform3<T>;
    fn mul(self, rhs: &TransformParam<T>) -> scaled_transform3<T> {
        self.compose(&rhs.to_transform().into())
    }
}

impl<T: Float> std::ops::Mul<transform3<T>> for &ScaledTransformParam<T>
where
    vect3<T>: ParamType,
{
    type Output = scaled_transform3<T>;
    fn mul(self, rhs: transform3<T>) -> scaled_transform3<T> {
        self.to_scaled_transform().compose(&rhs.into())
    }
}

impl<T: Float> std::ops::Mul<&ScaledTransformParam<T>> for transform3<T>
where
    vect3<T>: ParamType,
{
    type Output = scaled_transform3<T>;
    fn mul(self, rhs: &ScaledTransformParam<T>) -> scaled_transform3<T> {
        scaled_transform3::from(self).compose(&rhs.to_scaled_transform())
    }
}

#[cfg(test)]
mod value_tests {
    use super::*;

    fn close(a: vect3<f64>, b: vect3<f64>) -> bool {
        (a - b).norm() < 1e-12
    }

    fn mclose(a: matrix3<f64>, b: matrix3<f64>) -> bool {
        [vect3::new(1.0, 0.0, 0.0), vect3::new(0.0, 1.0, 0.0), vect3::new(0.0, 0.0, 1.0)]
            .into_iter()
            .all(|e| close(a * e, b * e))
    }

    fn pose() -> (vect3<f64>, quatern<f64>) {
        (vect3::new(0.3, -0.2, 0.5),
         quatern::from_axis_angle(vect3::new(0.2, 0.5, 1.0).unit(), 0.7))
    }

    /// The value forms are the documented maps, on the params and on the
    /// plain values alike, and the inverse undoes them.
    #[test]
    fn actions_and_inverse() {
        let (t, q) = pose();
        let r = q.rotation_matrix();
        let x = vect3::new(1.0, 0.4, -0.3);
        let p = TransformParam::new(t, q);
        assert!(close(p.transform(x), r * x + t));
        assert!(close(&p * x, r * x + t));
        assert!(close(p.inverse_transform(x), r.transpose() * (x - t)));
        assert!(close(p.inv() * x, r.transpose() * (x - t)));
        assert!(close(p.rotate(x), r * x));
        assert!(close(p.inverse_rotate(x), r.transpose() * x));
        assert!(close(p.inv().transform(p.transform(x)), x));
        assert!(close(p.inv().inv() * x, p.transform(x)));
        assert!(mclose(p.to_transform().rotation().rotation_matrix(), r));
        assert!(close(transform3::identity() * x, x));

        let s = ScaledTransformParam::new(t, q, 1.3);
        // The body-facing scale is right from construction, before any solve.
        assert_eq!(s.scale_factor, 1.3);
        assert!(close(s.transform(x), r * x * 1.3 + t));
        assert!(close(&s * x, r * x * 1.3 + t));
        assert!(close(s.inverse_transform(x), r.transpose() * (x - t) / 1.3));
        assert!(close(s.inv() * x, r.transpose() * (x - t) / 1.3));
        assert!(close(s.rotate(x), r * x));
        assert!(close(s.inverse_rotate(x), r.transpose() * x));
        assert!(close(s.inv().transform(s.transform(x)), x));
        assert!((s.inv().scale - 1.0 / 1.3).abs() < 1e-15);
        assert!((s.inv().inv().scale - 1.3).abs() < 1e-14);
        assert!(close(scaled_transform3::identity() * x, x));
        let lifted: scaled_transform3<f64> = p.to_transform().into();
        assert!(close(lifted * x, p.transform(x)));
        assert_eq!(lifted.scale, 1.0);
    }

    /// Composition applies the right operand first; the relative pose
    /// `a.inv() * b` is the hand-written `R_a^T (t_b - t_a)` form; rigid
    /// stays rigid, and a scale on either side carries through.
    #[test]
    fn composition() {
        let (ta, qa) = pose();
        let tb = vect3::new(1.1, 0.4, 0.2);
        let qb = quatern::from_axis_angle(vect3::new(1.0, 0.1, -0.3).unit(), 0.4);
        let (ra, rb) = (qa.rotation_matrix(), qb.rotation_matrix());
        let a = TransformParam::new(ta, qa);
        let b = TransformParam::new(tb, qb);
        let x = vect3::new(-0.4, 0.9, 0.2);
        let ab: transform3<f64> = &a * &b;
        assert!(close(ab * x, ra * (rb * x + tb) + ta));
        assert!(close(ab.transform(x), a.transform(b.transform(x))));
        let rel: transform3<f64> = a.inv() * &b;
        assert!(close(rel.translation, ra.transpose() * (tb - ta)));
        assert!(mclose(rel.rotation_matrix, ra.transpose() * rb));
        assert!(close(rel * x, a.inverse_transform(b.transform(x))));
        assert!(close((&a * b.inv()) * x, a.transform(b.inverse_transform(x))));

        let sb = ScaledTransformParam::new(tb, qb, 2.0);
        let asb: scaled_transform3<f64> = &a * &sb;
        assert!(close(asb * x, a.transform(sb.transform(x))));
        assert!((asb.scale - 2.0).abs() < 1e-15);
        let sba: scaled_transform3<f64> = &sb * &a;
        assert!(close(sba * x, sb.transform(a.transform(x))));
        let srel: scaled_transform3<f64> = sb.inv() * &a;
        assert!(close(srel * x, sb.inverse_transform(a.transform(x))));
        assert!((srel.scale - 0.5).abs() < 1e-15);
        assert!(close((a.inv() * &sb) * x, a.inverse_transform(sb.transform(x))));
        assert!(close((&sb * a.inv()) * x, sb.transform(a.inverse_transform(x))));
        assert!(close((&a * sb.inv()) * x, a.transform(sb.inverse_transform(x))));
        let sc = ScaledTransformParam::new(ta, qa, 0.5);
        let ss: scaled_transform3<f64> = &sc * &sb;
        assert!(close(ss * x, sc.transform(sb.transform(x))));
        assert!((ss.scale - 1.0).abs() < 1e-15);
        let t: transform3<f64> = (&a).into();
        assert!(close(t * x, a.transform(x)));
        let u: scaled_transform3<f64> = (&sb).into();
        assert!(close(u * x, sb.transform(x)));
    }
}

