// ---------------------------------------------------------------------------
// ParamType -- types that can be optimization parameters
// ---------------------------------------------------------------------------

/// Types that can be optimization parameters (f32, f64, vect2, vect3).
///
/// Defines the size in scalar elements, human-readable suffixes for parameter
/// names, and conversion routines between the concrete type and f32/f64 slices.
pub trait ParamType: Copy + Default + 'static {
    const SIZE: usize;
    const SUFFIXES: &'static [&'static str];
    fn write_to<F: crate::utils::Float>(&self, dst: &mut [F]);
    fn read_from<F: crate::utils::Float>(src: &[F]) -> Self;
}

impl ParamType for f32 {
    const SIZE: usize = 1;
    const SUFFIXES: &'static [&'static str] = &[""];
    fn write_to<F: crate::utils::Float>(&self, dst: &mut [F]) { dst[0] = F::from(*self).unwrap(); }
    fn read_from<F: crate::utils::Float>(src: &[F]) -> Self { src[0].to_f32().unwrap() }
}

impl ParamType for f64 {
    const SIZE: usize = 1;
    const SUFFIXES: &'static [&'static str] = &[""];
    fn write_to<F: crate::utils::Float>(&self, dst: &mut [F]) { dst[0] = F::from(*self).unwrap(); }
    fn read_from<F: crate::utils::Float>(src: &[F]) -> Self { src[0].to_f64().unwrap() }
}

impl<T: crate::utils::Float> ParamType for crate::vect::vect2<T> {
    const SIZE: usize = 2;
    const SUFFIXES: &'static [&'static str] = &[".x", ".y"];
    fn write_to<F: crate::utils::Float>(&self, dst: &mut [F]) { dst[0] = F::from(self.x).unwrap(); dst[1] = F::from(self.y).unwrap(); }
    fn read_from<F: crate::utils::Float>(src: &[F]) -> Self { Self::new(T::from(src[0]).unwrap(), T::from(src[1]).unwrap()) }
}

impl<T: crate::utils::Float> ParamType for crate::vect::vect3<T> {
    const SIZE: usize = 3;
    const SUFFIXES: &'static [&'static str] = &[".x", ".y", ".z"];
    fn write_to<F: crate::utils::Float>(&self, dst: &mut [F]) { dst[0] = F::from(self.x).unwrap(); dst[1] = F::from(self.y).unwrap(); dst[2] = F::from(self.z).unwrap(); }
    fn read_from<F: crate::utils::Float>(src: &[F]) -> Self { Self::new(T::from(src[0]).unwrap(), T::from(src[1]).unwrap(), T::from(src[2]).unwrap()) }
}

// No const string arrays for arbitrary N: an empty SUFFIXES list makes
// `Param::param_symbols` fall back to indexed component names ("[i]").
impl<T: crate::utils::Float, const N: usize> ParamType for crate::vect::vect<T, N> {
    const SIZE: usize = N;
    const SUFFIXES: &'static [&'static str] = &[];
    fn write_to<F: crate::utils::Float>(&self, dst: &mut [F]) {
        for i in 0..N { dst[i] = F::from(self.e[i]).unwrap(); }
    }
    fn read_from<F: crate::utils::Float>(src: &[F]) -> Self {
        crate::vect::vect { e: std::array::from_fn(|i| T::from(src[i]).unwrap()) }
    }
}

/// One parameter slot of a model: a [`Param<T>`] or one of the euler-angle
/// parameter types. Gives the slot's width and where it sits in the flat
/// parameter vector, which is what Hessian block indices and parameter
/// spans are built from.
///
/// A slot is live or fixed as a whole (`Param::optimize`): a fixed slot
/// has no index and occupies nothing in the parameter vector, so the live
/// slots of a model stay contiguous there whatever is fixed between them.
///
/// A parameter type that holds slots rather than being one -- `AngleParam`,
/// `UnitVecParam`, `TransformParam` -- implements no `ParamSlot` of its
/// own. It must forward [`Model::fold_param_span`] to the slots it holds,
/// or the span of an entity holding it will not cover its parameters.
pub trait ParamSlot {
    /// Scalar components this slot occupies when live.
    const COUNT: usize;

    /// Index of the first component in the flat parameter vector, or
    /// `u32::MAX` when the slot is fixed.
    fn param_index(&self) -> u32;

    /// Write the per-component indices into `out` (`COUNT` long):
    /// consecutive from [`param_index`](Self::param_index), or `u32::MAX`
    /// throughout when fixed.
    fn write_param_indices(&self, out: &mut [u32]) {
        let base = self.param_index();
        if base == u32::MAX {
            out.fill(u32::MAX);
        } else {
            for (k, o) in out.iter_mut().enumerate() {
                *o = base + k as u32;
            }
        }
    }

    /// Fold this slot into a running `(smallest live index, live count)`.
    fn fold_param_span(&self, min: &mut u32, count: &mut u32) {
        let i = self.param_index();
        if i != u32::MAX {
            if i < *min {
                *min = i;
            }
            *count += Self::COUNT as u32;
        }
    }
}

// ---------------------------------------------------------------------------
// Param<T> -- wrapper for an optimizable parameter
// ---------------------------------------------------------------------------

/// Optimizable parameter wrapper.
///
/// Holds the persistent `value`, a `work` copy used during optimization
/// iterations, and an `index` into the flat parameter vector. When
/// `optimize` is false the parameter is fixed and excluded from the
/// parameter vector.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Param<T: ParamType> {
    #[serde(default = "default_true")]
    pub optimize: bool,
    pub value: T,
    #[serde(skip)]
    work: T,
    // A plain skip would restore 0 -- a valid parameter index; the
    // inactive sentinel must survive deserialization.
    #[serde(skip, default = "inactive_index")]
    index: u32,
}

fn default_true() -> bool { true }
fn inactive_index() -> u32 { u32::MAX }

/// An optimizable parameter at the type's zero value -- the state a
/// generated-interface `push()` hands out to fill.
impl<T: ParamType> Default for Param<T> {
    fn default() -> Self {
        Param::new(T::default())
    }
}

impl<T: ParamType> Param<T> {
    /// Create a new optimizable parameter with the given initial value.
    pub fn new(value: T) -> Self {
        Param { optimize: true, value, work: T::default(), index: u32::MAX }
    }

    /// Create a fixed (non-optimizable) parameter with the given value.
    pub fn fixed(value: T) -> Self {
        Param { optimize: false, value, work: T::default(), index: u32::MAX }
    }

    /// Return the current working-copy value (set during `update_params`).
    pub fn work(&self) -> T { self.work }
    /// Return a reference to the current working-copy value.
    pub fn work_ref(&self) -> &T { &self.work }
    /// Return a mutable reference to the current working-copy value.
    pub fn work_mut(&mut self) -> &mut T { &mut self.work }

    /// Return this parameter's index into the flat parameter vector, or `u32::MAX` if fixed.
    pub fn index(&self) -> u32 { self.index }

    /// Write this parameter's per-component indices into `out`, or `u32::MAX` if fixed.
    pub fn write_indices(&self, out: &mut [u32]) { ParamSlot::write_param_indices(self, out) }
}

impl<T: ParamType> ParamSlot for Param<T> {
    const COUNT: usize = T::SIZE;
    fn param_index(&self) -> u32 { self.index }
}

impl<T: ParamType + std::fmt::Debug> std::fmt::Debug for Param<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.optimize {
            write!(f, "Param({:?}, idx={})", self.value, self.index)
        } else {
            write!(f, "Param({:?}, fixed)", self.value)
        }
    }
}

// ---------------------------------------------------------------------------
// Model trait -- hierarchical serialize/deserialize/update protocol
// ---------------------------------------------------------------------------

/// Protocol for hierarchical parameter serialization, deserialization, and update.
///
/// You rarely need to implement this manually -- the `#[arael::model]` macro
/// generates it automatically for your structs. It is also implemented for
/// `Param<T>`, euler angle params, and collections (`Vec`, `Arena`, `Option`).
/// The trait drives the optimization loop:
///
/// - `serialize_params` -- flatten optimizable parameters into a vector
///   and assign indices.
/// - `deserialize_params` -- write optimized values back into `Param::value`.
/// - `update_params` -- copy a candidate parameter vector into working copies.
/// - `update_self` -- reset working copies to current `value` (and precompute
///   derived quantities like rotation matrices).
/// - `release_blocks` -- a no-op kept for callers: every Hessian entry
///   lives in the block store the solve keeps in its
///   [`Context`](crate::threads::Context), so a model holds no assembly
///   memory of its own to free.
///
/// The parameter-vector methods are generic over the solve precision `F`.
/// A model holds no Hessian entries: the block fields are markers, and the
/// values live in the block store the solve owns.
pub trait Model {
    fn serialize_params<F: crate::utils::Float>(&mut self, _data: &mut std::vec::Vec<F>) {}
    fn deserialize_params<F: crate::utils::Float>(&mut self, _data: &[F]) {}
    fn update_params<F: crate::utils::Float>(&mut self, _data: &[F]) {}
    fn update_self(&mut self) {}

    const PARAM_COUNT: u32 = 0;
    fn serialize_size(&self) -> u32 { 0 }
    fn param_symbols(_base: &str, _out: &mut std::vec::Vec<String>) {}


    /// Append this model's parameter blocks as `(offset, width)` spans of
    /// the flat parameter vector -- one span per entity, folded from the
    /// entity's own parameter slots (see
    /// [`fold_param_span`](Self::fold_param_span)). Valid only after
    /// `serialize` has assigned indices. Entities whose params are all
    /// fixed contribute nothing.
    fn collect_param_blocks(&self, _out: &mut std::vec::Vec<(u32, u32)>) {}

    /// Fold this model's own parameter slots into a running `(smallest
    /// live index, live count)` -- the span
    /// [`collect_param_blocks`](Self::collect_param_blocks) reports for
    /// one entity. Implemented by the parameter types and, through the
    /// macro, by `#[arael(component)]` structs, whose slots fold into
    /// the owning entity's span. Everything else contributes nothing:
    /// a collection is many entities, not one span, and a plain
    /// sub-model's parameters are not part of its owner's block.
    fn fold_param_span(&self, _min: &mut u32, _count: &mut u32) {}

    /// Release the assembly memory this model and its sub-models hold
    /// between solves. A block field declares where Hessian entries go and
    /// holds none of them -- they are in the block store the solve keeps
    /// in its [`Context`](crate::threads::Context) and go with it -- so
    /// the call frees nothing and is safe at any time. Default: no-op.
    fn release_blocks(&mut self) {}

    // Fold accepted-step euler angle deltas into their reference rotations
    // and zero the delta entries in the parameter vector. A no-op for
    // everything except EulerAngleParam, which re-centers after every
    // accepted LM step (the property that avoids gimbal lock). Recurses
    // through the model tree exactly like update/serialize, so params at
    // any nesting depth are advanced.
    fn advance_params<F: crate::utils::Float>(&mut self, _params: &mut [F]) {}

}

// ---------------------------------------------------------------------------
// Component -- compound-parameter lifecycle
// ---------------------------------------------------------------------------

/// Runtime lifecycle of a `#[arael(component)]` struct -- a compound
/// parameter whose `Param` fields fold into the OWNING struct's span.
/// The macro calls these around the solve; the symbolic meaning of the
/// component's user-facing fields is given separately by
/// `#[arael(symbolic = ...)]` field attributes (which the macro
/// differentiates at expansion time -- a trait method body cannot be).
///
/// Like any non-root model struct, a component may be generic over its
/// scalar: exactly one type parameter, bounded inline by `Float`
/// (`struct Dir<T: Float>`), with fields spelled generically
/// (`quatern<T>`, `Param<vect2<T>>`, bare `Param<T>`). One definition
/// then serves f64 and f32 models alike -- see
/// `examples/plane_slam_demo.rs`.
///
/// All methods default to no-ops: a stateless reparameterization
/// implements nothing.
pub trait Component {
    /// Seed the reference/chart from the user-facing value. Runs at
    /// serialize, before the component's params are read.
    fn start(&mut self) {}
    /// Re-center after an accepted step: the component's `Param` values
    /// hold the accepted step; fold them into the reference and reset
    /// them. Runs at the advance point; the macro writes the reset values
    /// back into the parameter vector afterwards.
    fn update(&mut self) {}
    /// Write the user-facing value back from the reference/params. Runs
    /// at deserialize.
    fn finish(&mut self) {}
}

// ---------------------------------------------------------------------------
// ExtendedModel -- user-defined constraint hook for root structs
// ---------------------------------------------------------------------------

/// Extension hooks for custom constraints on `#[arael(root, extended)]` structs.
///
/// Use this when you need constraints that can't be expressed via
/// `#[arael(constraint(...))]` at compile time -- for example, constraints
/// parsed from user input at runtime, or constraints that need access to
/// the full root struct.
///
/// A key use case is **runtime differentiation**: parse an equation string
/// with `arael_sym::parse`, symbolically differentiate with
/// `E::diff`, then evaluate numerically each solver iteration. This
/// powers the parametric expression dimensions in `arael-sketch` (where
/// the user types `d0 * 2 + 3` as a dimension value) and the
/// `runtime_fit_demo` example (which accepts an arbitrary curve-fitting
/// equation from the command line).
///
/// To use: mark the root struct with `#[arael(root, extended)]` and
/// implement this trait. The macro-generated `LmProblem` calls these
/// methods at the appropriate points in the optimization loop. Default
/// implementations are no-ops, so you only override what you need.
///
/// `extended` does not change the solver route by itself: a root that
/// only syncs derived state (`extended_update` / `extended_deserialize`
/// / `extended_cost`) keeps its static block pattern and the fast
/// structure-based sparse routes. A hook that pushes COO entries moves
/// the solve to compute-first pattern discovery, because those entries
/// exist only at runtime.
///
/// Custom gradient and Hessian contributions need nothing declared:
/// `extended_compute` is handed the solve's [`Coo`] list and writes the
/// gradient into the slice it is given.
///
/// # Execution order
///
/// Each solver iteration runs:
/// 1. `Model::update_params` -- copies params into working values
/// 2. **`extended_update`** -- set up derived state before calculations
/// 3. The block store is zeroed, its COO list included
/// 4. Macro-generated constraint loops -- fill the block store in the
///    [`Context`](crate::threads::Context)
/// 5. **`extended_compute`** -- push custom residuals into the [`Coo`]
///    list it is handed, writing grad entries directly into the
///    LM-provided global slice
/// 6. The store's own walk -- read every Hessian entry into the global
///    Hessian
///
/// For cost evaluation: `Model::update_params` -> `extended_update` ->
/// macro-generated cost loop -> **`extended_cost`**.
///
/// # Example
///
/// Robust curve fitting where the equation is parsed at runtime. The
/// residual and its derivatives are symbolic expressions evaluated
/// numerically each iteration (see `examples/runtime_fit_demo.rs`):
///
/// ```ignore
/// #[arael::model]
/// #[arael(root, extended)]
/// struct RegressionModel {
///     coeffs: refs::Vec<Coefficient>,         // optimizable parameters
///     residual_expr: Option<arael_sym::E>,    // parsed equation
///     derivs: Vec<(String, u32, arael_sym::E)>, // (name, param_index, d_residual/d_param)
///     data: Vec<(f64, f64)>,
///     param_names: Vec<String>,
/// }
///
/// // Setup: parse equation, differentiate symbolically (once)
/// let expr = arael_sym::parse("a * x + b").unwrap();
/// let residual = (expr - arael_sym::symbol("y")) / arael_sym::constant(sigma);
/// let dr_da = residual.diff("a");
/// let dr_db = residual.diff("b");
///
/// impl ExtendedModel<f64> for RegressionModel {
///     fn extended_compute(&mut self, params: &[f64], grad: &mut [f64],
///                         coo: &mut Coo<f64>) {
///         // Evaluate symbolically-differentiated expressions numerically
///         for &(x, y) in &self.data {
///             vars.insert("x", x);
///             vars.insert("y", y);
///             let r = self.residual_expr.eval(&vars).unwrap();
///             let dr: Vec<f64> = self.derivs.iter()
///                 .map(|(_, _, d)| d.eval(&vars).unwrap()).collect();
///             let indices: Vec<u32> = self.derivs.iter()
///                 .map(|(_, idx, _)| *idx).collect();
///             // add_residual writes 2*r*dr into `grad` AND pushes the
///             // full upper-triangle Hessian into the COO list
///             coo.add_residual(r, &indices, &dr, grad);
///         }
///     }
///
///     fn extended_cost(&self, params: &[f64]) -> f64 {
///         // Sum of squared residuals
///         self.data.iter().filter_map(|&(x, y)| {
///             vars.insert("x", x);
///             vars.insert("y", y);
///             let r = self.residual_expr.eval(&vars).ok()?;
///             Some(r * r)
///         }).sum()
///     }
/// }
/// ```
///
/// See `examples/runtime_fit_demo.rs` for the complete working example,
/// and `arael-sketch-solver` for a production use of this pattern with
/// parametric expression dimensions.
/// The trait is parameterized by the solve width: implement it at your
/// root's precision (`impl ExtendedModel<f64> for MyRoot`) with plainly
/// typed bodies. A root generated at one precision requires the matching
/// instantiation, so a width mismatch is a missing-impl compile error.
pub trait ExtendedModel<F: crate::utils::Float> {
    /// Called after `deserialize` writes optimized values back to `Param::value`.
    /// Use to sync derived persistent state (e.g. copy one param's value to another).
    fn extended_deserialize(&mut self) {}
    /// Called after `update_params`, before cost/constraint calculations.
    /// Use to compute derived state that constraints depend on.
    fn extended_update(&mut self, _params: &[F]) {}
    /// Additional cost contribution. Called after the
    /// macro-generated cost loop.
    fn extended_cost(&self, _params: &[F]) -> F { F::zero() }
    /// Compute custom constraint residuals. Called after the
    /// macro-generated constraints, on the calling thread. Writes
    /// gradient contributions directly into `grad` and Hessian entries
    /// into `coo`, the solve's own COO list.
    ///
    /// **Iteration-invariance contract:** the sparse solvers cache the
    /// Hessian sparsity pattern from the first iteration of a solve and
    /// replay it positionally on every later iteration. The number and
    /// order of the entries this hook pushes must therefore stay constant
    /// within one `lm_solve` call -- residual *values* may change freely,
    /// entry *structure* may not. Violations are detected and reported
    /// ("sparsity pattern changed between iterations"). Restructure
    /// between solves instead; `LmSolver::reset()` rebuilds the cached
    /// pattern.
    fn extended_compute(&mut self, _params: &[F], _grad: &mut [F], _coo: &mut Coo<F>) {}
    /// Append Jacobian rows for runtime constraints.
    /// `cid` is the constraint counter -- increment per constraint object.
    fn extended_jacobian(&mut self, _params: &[F], _rows: &mut std::vec::Vec<JacobianRow<F>>, _cid: &mut u32) {}
}


// ---------------------------------------------------------------------------
// Model impl for Param<T>
// ---------------------------------------------------------------------------

impl<T: ParamType> Model for Param<T> {
    fn fold_param_span(&self, min: &mut u32, count: &mut u32) {
        ParamSlot::fold_param_span(self, min, count)
    }

    fn serialize_params<F: crate::utils::Float>(&mut self, data: &mut std::vec::Vec<F>) {
        if self.optimize {
            self.index = data.len() as u32;
            let start = data.len();
            data.resize(start + T::SIZE, F::zero());
            self.value.write_to(&mut data[start..start + T::SIZE]);
        } else {
            self.index = u32::MAX;
        }
    }

    fn deserialize_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.value = T::read_from(&data[i..i + T::SIZE]);
        }
    }

    fn update_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.work = T::read_from(&data[i..i + T::SIZE]);
        } else {
            self.work = self.value;
        }
    }

    fn update_self(&mut self) {
        self.work = self.value;
    }

    const PARAM_COUNT: u32 = T::SIZE as u32;
    fn serialize_size(&self) -> u32 { if self.optimize { T::SIZE as u32 } else { 0 } }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        if T::SUFFIXES.len() == T::SIZE {
            for suffix in T::SUFFIXES {
                out.push(format!("{}{}", base, suffix));
            }
        } else {
            // Types without a const suffix table (vect<T, N>): indexed
            // component names.
            for i in 0..T::SIZE {
                out.push(format!("{}[{}]", base, i));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SimpleEulerAngleParam -- euler angles with precomputed rotation matrix
// ---------------------------------------------------------------------------

use crate::vect::vect3;
use crate::matrix::matrix3;

/// Euler angle parameter with a precomputed rotation matrix.
///
/// Stores roll/pitch/yaw (x/y/z) as a `vect3<T>`. On each update the
/// framework precomputes the full 3x3 rotation matrix and its derivative so
/// that constraint code can reference them without redundant trig calls.
///
/// Convention: x = roll, y = pitch, z = yaw. Axes: x = forward, y = left,
/// z = up. Rotation order: R = Rz(yaw) * Ry(pitch) * Rx(roll).
///
/// Suitable when angles stay far from gimbal lock (pitch near +-90 deg).
/// For near-gimbal-lock scenarios use [`EulerAngleParam`] instead.
#[derive(Clone, Copy)]
pub struct SimpleEulerAngleParam<T: crate::utils::Float> {
    pub optimize: bool,
    pub value: vect3<T>,
    work: vect3<T>,
    index: u32,
    #[doc(hidden)] pub rotation_matrix: matrix3<T>,
    /// Precomputed d(rotation_matrix)/d(value.{x,y,z}) -- the pose rotation
    /// Jacobian, so constraints that differentiate through the rotation read it
    /// once per pose instead of rebuilding it per observation.
    #[doc(hidden)] pub rotation_matrix_deriv: [matrix3<T>; 3],
}

impl<T: crate::utils::Float> Default for SimpleEulerAngleParam<T> {
    fn default() -> Self {
        Self {
            optimize: true,
            value: vect3::<T>::default(),
            work: vect3::<T>::default(),
            index: u32::MAX,
            rotation_matrix: matrix3::<T>::identity(),
            rotation_matrix_deriv: [matrix3::<T>::identity(),
                                    matrix3::<T>::identity(),
                                    matrix3::<T>::identity()],
        }
    }
}

impl<T: crate::utils::Float> SimpleEulerAngleParam<T> {
    /// Create a new optimizable euler angle parameter with the given initial angles.
    pub fn new(value: vect3<T>) -> Self {
        Self { value, ..Default::default() }
    }
    /// Create a fixed (non-optimizable) euler angle parameter.
    pub fn fixed(value: vect3<T>) -> Self {
        Self { optimize: false, value, ..Default::default() }
    }
    /// Return the current working-copy euler angles.
    pub fn work(&self) -> vect3<T> { self.work }
    /// Return this parameter's index into the flat parameter vector, or `u32::MAX` if fixed.
    pub fn index(&self) -> u32 { self.index }
    /// Write per-component indices into `out`, or `u32::MAX` if fixed.
    pub fn write_indices(&self, out: &mut [u32]) { ParamSlot::write_param_indices(self, out) }
    /// Precompute the rotation matrix and its derivative from current work value.
    #[doc(hidden)]
    pub fn __precompute(&mut self) {
        let (s, c) = self.work.sincos();
        self.rotation_matrix = matrix3::<T>::rotation_from_euler_angles_sincos(s, c);
        // Pose rotation Jacobian: dR/d(value.k), read once per pose by the
        // constraint Jacobian instead of rebuilt from sincos per observation.
        self.rotation_matrix_deriv = matrix3::<T>::rotation_from_euler_angles_sincos_deriv(s, c);
    }
}

impl<T: crate::utils::Float> ParamSlot for SimpleEulerAngleParam<T> {
    const COUNT: usize = 3;
    fn param_index(&self) -> u32 { self.index }
}

impl<T: crate::utils::Float> serde::Serialize for SimpleEulerAngleParam<T> where T: serde::Serialize {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("SimpleEulerAngleParam", 2)?;
        st.serialize_field("optimize", &self.optimize)?;
        st.serialize_field("value", &self.value)?;
        st.end()
    }
}

impl<'de, T: crate::utils::Float + serde::Deserialize<'de>> serde::Deserialize<'de> for SimpleEulerAngleParam<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, Visitor};
        struct V<U>(std::marker::PhantomData<U>);
        impl<'de2, U: crate::utils::Float + serde::Deserialize<'de2>> Visitor<'de2> for V<U> {
            type Value = SimpleEulerAngleParam<U>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("SimpleEulerAngleParam")
            }
            fn visit_map<A: MapAccess<'de2>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut opt = None; let mut val = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "optimize" => opt = Some(map.next_value()?),
                        "value" => val = Some(map.next_value()?),
                        _ => { let _ = map.next_value::<de::IgnoredAny>()?; }
                    }
                }
                Ok(SimpleEulerAngleParam {
                    optimize: opt.unwrap_or(true),
                    value: val.unwrap_or_default(),
                    ..Default::default()
                })
            }
        }
        d.deserialize_map(V::<T>(std::marker::PhantomData))
    }
}

impl<T: crate::utils::Float> Model for SimpleEulerAngleParam<T> where vect3<T>: ParamType {
    fn fold_param_span(&self, min: &mut u32, count: &mut u32) {
        ParamSlot::fold_param_span(self, min, count)
    }

    fn serialize_params<F: crate::utils::Float>(&mut self, data: &mut std::vec::Vec<F>) {
        if self.optimize {
            self.index = data.len() as u32;
            let start = data.len();
            data.resize(start + 3, F::zero());
            self.value.write_to(&mut data[start..start + 3]);
        } else { self.index = u32::MAX; }
    }
    fn deserialize_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.value = <vect3<T> as ParamType>::read_from(&data[i..i + 3]);
        }
    }
    fn update_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.work = <vect3<T> as ParamType>::read_from(&data[i..i + 3]);
        } else { self.work = self.value; }
    }
    fn update_self(&mut self) {
        self.work = self.value;
        self.__precompute();
    }

    const PARAM_COUNT: u32 = 3;
    fn serialize_size(&self) -> u32 { if self.optimize { 3 } else { 0 } }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        for suffix in <vect3<T> as ParamType>::SUFFIXES {
            out.push(format!("{}{}", base, suffix));
        }
    }
}

// ---------------------------------------------------------------------------
// EulerAngleParam -- gimbal-lock-free euler angles with reference rotation
// ---------------------------------------------------------------------------

/// Gimbal-lock-free euler angle parameter.
///
/// Instead of directly optimizing the three angles, this type maintains a
/// reference rotation matrix and optimizes a small delta rotation around it.
/// After each solver iteration, `advance()` folds the delta into the
/// reference rotation and resets the delta to zero, keeping the
/// linearization point near the identity where euler angles are well-behaved.
///
/// Convention: x = roll, y = pitch, z = yaw. Axes: x = forward, y = left,
/// z = up. Rotation order: R = Rz(yaw) * Ry(pitch) * Rx(roll).
///
/// The composed rotation matrix and derived euler angles / sincos values are
/// precomputed on each update for use in constraint expressions.
#[derive(Clone, Copy)]
pub struct EulerAngleParam<T: crate::utils::Float> {
    pub optimize: bool,
    pub value: vect3<T>,
    work: vect3<T>,
    index: u32,
    #[doc(hidden)] pub ref_rotation: matrix3<T>,
    #[doc(hidden)] pub delta: vect3<T>,
    #[doc(hidden)] pub rotation_matrix: matrix3<T>,
    /// Precomputed d(rotation_matrix)/d(delta.{x,y,z}) -- the pose rotation
    /// Jacobian, so constraints that differentiate through the rotation read it
    /// once per pose instead of rebuilding R_ref * dR(delta)/ddelta per
    /// observation.
    #[doc(hidden)] pub rotation_matrix_deriv: [matrix3<T>; 3],
}

impl<T: crate::utils::Float> Default for EulerAngleParam<T> {
    fn default() -> Self {
        Self {
            optimize: true,
            value: vect3::<T>::default(),
            work: vect3::<T>::default(),
            index: u32::MAX,
            ref_rotation: matrix3::<T>::identity(),
            delta: vect3::<T>::default(),
            rotation_matrix: matrix3::<T>::identity(),
            rotation_matrix_deriv: [matrix3::<T>::identity(),
                                    matrix3::<T>::identity(),
                                    matrix3::<T>::identity()],
        }
    }
}

impl<T: crate::utils::Float> EulerAngleParam<T> {
    /// Create a new optimizable euler angle parameter with the given initial angles.
    pub fn new(value: vect3<T>) -> Self {
        Self { value, ..Default::default() }
    }
    /// Create a fixed (non-optimizable) euler angle parameter.
    pub fn fixed(value: vect3<T>) -> Self {
        Self { optimize: false, value, ..Default::default() }
    }
    /// Return the current working-copy euler angles (derived from ref_rotation * delta).
    pub fn work(&self) -> vect3<T> { self.work }
    /// Return this parameter's index into the flat parameter vector, or `u32::MAX` if fixed.
    pub fn index(&self) -> u32 { self.index }
    /// Write per-component indices into `out`, or `u32::MAX` if fixed.
    pub fn write_indices(&self, out: &mut [u32]) { ParamSlot::write_param_indices(self, out) }
    /// Absorb current delta into reference rotation and reset delta.
    pub fn advance(&mut self) {
        self.ref_rotation = self.ref_rotation
            * matrix3::<T>::rotation_from_euler_angles(self.delta);
        self.delta = vect3::<T>::default();
    }
    /// Precompute composed rotation and work angles from current delta + ref_rotation.
    #[doc(hidden)]
    pub fn __precompute(&mut self) {
        let (s, c) = self.delta.sincos();
        let dea_rot = matrix3::<T>::rotation_from_euler_angles_sincos(s, c);
        self.rotation_matrix = self.ref_rotation * dea_rot;
        // Pose rotation Jacobian: the composed rotation is linear in the delta
        // rotation, so d(R_ref*R(d))/dd.k = R_ref * dR(d)/dd.k. Read once per
        // pose by the constraint Jacobian instead of composed per observation.
        let dea_rot_deriv = matrix3::<T>::rotation_from_euler_angles_sincos_deriv(s, c);
        self.rotation_matrix_deriv = [self.ref_rotation * dea_rot_deriv[0],
                                      self.ref_rotation * dea_rot_deriv[1],
                                      self.ref_rotation * dea_rot_deriv[2]];
        self.work = self.rotation_matrix.get_euler_angles();
    }
}

impl<T: crate::utils::Float> ParamSlot for EulerAngleParam<T> {
    const COUNT: usize = 3;
    fn param_index(&self) -> u32 { self.index }
}

impl<T: crate::utils::Float> serde::Serialize for EulerAngleParam<T> where T: serde::Serialize {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("EulerAngleParam", 2)?;
        st.serialize_field("optimize", &self.optimize)?;
        st.serialize_field("value", &self.value)?;
        st.end()
    }
}

impl<'de, T: crate::utils::Float + serde::Deserialize<'de>> serde::Deserialize<'de> for EulerAngleParam<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, Visitor};
        struct V<U>(std::marker::PhantomData<U>);
        impl<'de2, U: crate::utils::Float + serde::Deserialize<'de2>> Visitor<'de2> for V<U> {
            type Value = EulerAngleParam<U>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("EulerAngleParam")
            }
            fn visit_map<A: MapAccess<'de2>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut opt = None; let mut val = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "optimize" => opt = Some(map.next_value()?),
                        "value" => val = Some(map.next_value()?),
                        _ => { let _ = map.next_value::<de::IgnoredAny>()?; }
                    }
                }
                Ok(EulerAngleParam {
                    optimize: opt.unwrap_or(true),
                    value: val.unwrap_or_default(),
                    ..Default::default()
                })
            }
        }
        d.deserialize_map(V::<T>(std::marker::PhantomData))
    }
}

impl<T: crate::utils::Float> Model for EulerAngleParam<T> where vect3<T>: ParamType {
    fn fold_param_span(&self, min: &mut u32, count: &mut u32) {
        ParamSlot::fold_param_span(self, min, count)
    }

    fn serialize_params<F: crate::utils::Float>(&mut self, data: &mut std::vec::Vec<F>) {
        // Seed the reference from value for fixed params too -- constraints
        // evaluate a fixed rotation through ref_rotation as well.
        self.ref_rotation = matrix3::<T>::rotation_from_euler_angles(self.value);
        if self.optimize {
            self.index = data.len() as u32;
            data.push(F::zero()); data.push(F::zero()); data.push(F::zero());
        } else { self.index = u32::MAX; }
    }
    fn deserialize_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            let dea = <vect3<T> as ParamType>::read_from(&data[i..i + 3]);
            self.ref_rotation = self.ref_rotation
                * matrix3::<T>::rotation_from_euler_angles(dea);
            self.value = self.ref_rotation.get_euler_angles();
            self.delta = vect3::<T>::default();
        }
    }
    fn update_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.delta = <vect3<T> as ParamType>::read_from(&data[i..i + 3]);
        } else { self.delta = vect3::<T>::default(); }
    }
    fn update_self(&mut self) {
        self.ref_rotation = matrix3::<T>::rotation_from_euler_angles(self.value);
        self.delta = vect3::<T>::default();
        self.__precompute();
    }

    const PARAM_COUNT: u32 = 3;
    fn serialize_size(&self) -> u32 { if self.optimize { 3 } else { 0 } }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        for suffix in <vect3<T> as ParamType>::SUFFIXES {
            out.push(format!("{}{}", base, suffix));
        }
    }

    fn advance_params<F: crate::utils::Float>(&mut self, params: &mut [F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.advance();
            params[i] = F::zero(); params[i + 1] = F::zero(); params[i + 2] = F::zero();
        }
    }
}

// ---------------------------------------------------------------------------
// QuaternionParam -- gimbal-lock-free rotation with a quaternion reference
// ---------------------------------------------------------------------------

/// Gimbal-lock-free rotation parameter with a quaternion reference.
///
/// Like [`EulerAngleParam`], this optimizes a small three-angle delta rotation
/// around a reference orientation and re-centers after each solver iteration,
/// keeping the linearization point near the identity. The difference is that
/// the reference is kept as a unit quaternion (renormalized on every re-center)
/// rather than a rotation matrix, so it never drifts off SO(3).
///
/// Its delta is a rotation vector rather than euler angles, but it exposes the
/// same reference rotation matrix (`ref_rotation`) and composed rotation as
/// `EulerAngleParam`, so constraints consume it identically.
///
/// `value` is the initial orientation going in and the optimized orientation
/// coming out: the solver keeps its working state in an internal reference
/// quaternion and syncs `value` only when `deserialize_params` reads
/// the result back. As with the other rotation parameters, call
/// `deserialize(&result.x)` after a solve to get the result.
///
/// Convention: x = roll, y = pitch, z = yaw. Axes: x = forward, y = left,
/// z = up. Rotation order: R = Rz(yaw) * Ry(pitch) * Rx(roll).
#[derive(Clone, Copy)]
pub struct QuaternionParam<T: crate::utils::Float> {
    pub optimize: bool,
    /// Initial orientation in, optimized orientation out (synced by
    /// `deserialize_params`); a unit quaternion.
    pub value: crate::quatern::quatern<T>,
    /// Solver-internal reference orientation, kept as a unit quaternion;
    /// `advance` folds each accepted delta into it (renormalized).
    ref_value: crate::quatern::quatern<T>,
    work: vect3<T>,
    index: u32,
    #[doc(hidden)] pub ref_rotation: matrix3<T>,
    #[doc(hidden)] pub delta: vect3<T>,
    #[doc(hidden)] pub rotation_matrix: matrix3<T>,
    /// Precomputed d(rotation_matrix)/d(delta.{x,y,z}) -- the pose rotation
    /// Jacobian, so constraints that differentiate through the retraction read
    /// it once per pose instead of recomputing it per observation.
    #[doc(hidden)] pub rotation_matrix_deriv: [matrix3<T>; 3],
}

impl<T: crate::utils::Float> Default for QuaternionParam<T> {
    fn default() -> Self {
        Self {
            optimize: true,
            value: crate::quatern::quatern::<T>::identity(),
            ref_value: crate::quatern::quatern::<T>::identity(),
            work: vect3::<T>::default(),
            index: u32::MAX,
            ref_rotation: matrix3::<T>::identity(),
            delta: vect3::<T>::default(),
            rotation_matrix: matrix3::<T>::identity(),
            rotation_matrix_deriv: [matrix3::<T>::identity(),
                                    matrix3::<T>::identity(),
                                    matrix3::<T>::identity()],
        }
    }
}

impl<T: crate::utils::Float> QuaternionParam<T> {
    /// Create an optimizable rotation parameter from an initial quaternion.
    pub fn new(value: crate::quatern::quatern<T>) -> Self {
        Self { value: value.unit(), ..Default::default() }
    }
    /// Create a fixed (non-optimizable) rotation parameter.
    pub fn fixed(value: crate::quatern::quatern<T>) -> Self {
        Self { optimize: false, value: value.unit(), ..Default::default() }
    }
    /// Create from initial euler angles (roll, pitch, yaw).
    pub fn from_euler_angles(ea: vect3<T>) -> Self {
        Self::new(crate::quatern::quatern::<T>::from_euler_angles(ea))
    }
    /// Create from an initial axis-angle rotation (axis must be unit).
    pub fn from_axis_angle(axis: vect3<T>, angle: T) -> Self {
        Self::new(crate::quatern::quatern::<T>::from_axis_angle(axis, angle))
    }
    /// Create from an initial rotation matrix.
    pub fn from_rotation_matrix(m: matrix3<T>) -> Self {
        Self::new(crate::quatern::quatern::<T>::from_rotation_matrix(m))
    }
    /// Return the current working-copy euler angles (derived from ref_rotation * delta).
    pub fn work(&self) -> vect3<T> { self.work }
    /// Return this parameter's index into the flat parameter vector, or `u32::MAX` if fixed.
    pub fn index(&self) -> u32 { self.index }
    /// Write per-component indices into `out`, or `u32::MAX` if fixed.
    pub fn write_indices(&self, out: &mut [u32]) { ParamSlot::write_param_indices(self, out) }
    /// Absorb the current delta (a rotation vector) into the internal
    /// reference quaternion via the retraction (renormalized) and reset
    /// delta. `value` is left untouched -- it syncs on deserialize.
    pub fn advance(&mut self) {
        self.ref_value = (self.ref_value
            * crate::quatern::quatern::<T>::from_rotation_vector_small(self.delta)).unit();
        self.ref_rotation = self.ref_value.rotation_matrix();
        self.delta = vect3::<T>::default();
    }
    /// Precompute the composed rotation R_ref * R(delta) and derived euler
    /// angles from the current delta + ref_rotation, where R(delta) is the
    /// small-angle retraction of the rotation-vector delta.
    #[doc(hidden)]
    pub fn __precompute(&mut self) {
        let dea_rot = matrix3::<T>::from_rotation_vector_small(self.delta);
        self.rotation_matrix = self.ref_rotation * dea_rot;
        // Compose the retraction's Jacobian with the reference: the composed
        // rotation is linear in the retraction, so d(R_ref*R(d))/dd.k =
        // R_ref * dR(d)/dd.k. Read once per pose by the constraint Jacobian.
        let d = matrix3::<T>::from_rotation_vector_small_deriv(self.delta);
        self.rotation_matrix_deriv = [self.ref_rotation * d[0],
                                      self.ref_rotation * d[1],
                                      self.ref_rotation * d[2]];
        self.work = self.rotation_matrix.get_euler_angles();
    }
}

impl<T: crate::utils::Float> ParamSlot for QuaternionParam<T> {
    const COUNT: usize = 3;
    fn param_index(&self) -> u32 { self.index }
}

impl<T: crate::utils::Float> serde::Serialize for QuaternionParam<T> where T: serde::Serialize {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut st = s.serialize_struct("QuaternionParam", 2)?;
        st.serialize_field("optimize", &self.optimize)?;
        st.serialize_field("value", &self.value)?;
        st.end()
    }
}

impl<'de, T: crate::utils::Float + serde::Deserialize<'de>> serde::Deserialize<'de> for QuaternionParam<T> {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use serde::de::{self, MapAccess, Visitor};
        struct V<U>(std::marker::PhantomData<U>);
        impl<'de2, U: crate::utils::Float + serde::Deserialize<'de2>> Visitor<'de2> for V<U> {
            type Value = QuaternionParam<U>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("QuaternionParam")
            }
            fn visit_map<A: MapAccess<'de2>>(self, mut map: A) -> Result<Self::Value, A::Error> {
                let mut opt = None; let mut val = None;
                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "optimize" => opt = Some(map.next_value()?),
                        "value" => val = Some(map.next_value()?),
                        _ => { let _ = map.next_value::<de::IgnoredAny>()?; }
                    }
                }
                let value = val.unwrap_or_else(crate::quatern::quatern::<U>::identity);
                Ok(QuaternionParam {
                    optimize: opt.unwrap_or(true),
                    value: value.unit(),
                    ..Default::default()
                })
            }
        }
        d.deserialize_map(V::<T>(std::marker::PhantomData))
    }
}

impl<T: crate::utils::Float> Model for QuaternionParam<T> where vect3<T>: ParamType {
    fn fold_param_span(&self, min: &mut u32, count: &mut u32) {
        ParamSlot::fold_param_span(self, min, count)
    }

    fn serialize_params<F: crate::utils::Float>(&mut self, data: &mut std::vec::Vec<F>) {
        // Seed the internal reference from value for fixed params too --
        // constraints evaluate a fixed rotation through ref_rotation as well.
        self.ref_value = self.value.unit();
        self.ref_rotation = self.ref_value.rotation_matrix();
        if self.optimize {
            self.index = data.len() as u32;
            data.push(F::zero()); data.push(F::zero()); data.push(F::zero());
        } else { self.index = u32::MAX; }
    }
    fn deserialize_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            let dvec = <vect3<T> as ParamType>::read_from(&data[i..i + 3]);
            // Fold the handed-back rotation-vector delta with the same
            // retraction advance uses. The reference itself is not mutated,
            // so repeated deserialize calls are idempotent.
            self.value = (self.ref_value
                * crate::quatern::quatern::<T>::from_rotation_vector_small(dvec)).unit();
            self.ref_rotation = self.value.rotation_matrix();
            self.delta = vect3::<T>::default();
        }
    }
    fn update_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.delta = <vect3<T> as ParamType>::read_from(&data[i..i + 3]);
        } else { self.delta = vect3::<T>::default(); }
    }
    fn update_self(&mut self) {
        self.ref_value = self.value.unit();
        self.ref_rotation = self.ref_value.rotation_matrix();
        self.delta = vect3::<T>::default();
        self.__precompute();
    }

    const PARAM_COUNT: u32 = 3;
    fn serialize_size(&self) -> u32 { if self.optimize { 3 } else { 0 } }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        for suffix in <vect3<T> as ParamType>::SUFFIXES {
            out.push(format!("{}{}", base, suffix));
        }
    }

    fn advance_params<F: crate::utils::Float>(&mut self, params: &mut [F]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.advance();
            params[i] = F::zero(); params[i + 1] = F::zero(); params[i + 2] = F::zero();
        }
    }
}

// ---------------------------------------------------------------------------
// No-op Model impls for leaf types
// ---------------------------------------------------------------------------

// Every Model method defaults to a no-op, so a leaf type participates
// with an empty impl. Kept in step with impl_scalar_model_sym! below: a
// primitive field needs both a Model and a ModelSym to be usable without
// `#[arael(skip)]`.
macro_rules! impl_model_noop {
    ($($ty:ty),* $(,)?) => {
        $( impl Model for $ty {} )*
    };
}

impl_model_noop!(
    bool, char, String,
    i8, i16, i32, i64, i128, isize,
    u8, u16, u32, u64, u128, usize,
    f32, f64,
);

macro_rules! impl_model_noop_generic {
    ($($m:ident :: $t:ident),* $(,)?) => {
        $( impl<F: crate::utils::Float> Model for crate::$m::$t<F> {} )*
    };
}

impl_model_noop_generic!(
    vect::vect3, vect::vect2,
    matrix::matrix3, matrix::matrix2,
    quatern::quatern,
);

impl<F: crate::utils::Float, const N: usize> Model for crate::vect::vect<F, N> {}
impl<F: crate::utils::Float, const R: usize, const C: usize> Model
    for crate::matrix::matrix<F, R, C> {}

impl<T> Model for crate::refs::Ref<T> {}

// ---------------------------------------------------------------------------
// Collection Model impls — iterate and recurse
// ---------------------------------------------------------------------------

macro_rules! impl_model_collection {
    ($ty:ty, $iter_mut:ident) => {
        impl<T: Model> Model for $ty {
            fn serialize_params<F: crate::utils::Float>(&mut self, data: &mut std::vec::Vec<F>) {
                for item in self.$iter_mut() { item.serialize_params(data); }
            }
            fn deserialize_params<F: crate::utils::Float>(&mut self, data: &[F]) {
                for item in self.$iter_mut() { item.deserialize_params(data); }
            }
            fn update_params<F: crate::utils::Float>(&mut self, data: &[F]) {
                for item in self.$iter_mut() { item.update_params(data); }
            }
            fn update_self(&mut self) {
                for item in self.$iter_mut() { item.update_self(); }
            }
            fn advance_params<F: crate::utils::Float>(&mut self, params: &mut [F]) {
                for item in self.$iter_mut() { item.advance_params(params); }
            }
            fn collect_param_blocks(&self, out: &mut std::vec::Vec<(u32, u32)>) {
                for item in self.iter() { item.collect_param_blocks(out); }
            }
            fn release_blocks(&mut self) {
                for item in self.$iter_mut() { item.release_blocks(); }
            }
            fn serialize_size(&self) -> u32 {
                self.iter().map(|item| item.serialize_size()).sum()
            }
        }
    };
}

impl_model_collection!(std::vec::Vec<T>, iter_mut);
impl_model_collection!(crate::refs::Vec<T>, iter_mut);
impl_model_collection!(crate::refs::Deque<T>, iter_mut);

// Arena needs a manual impl because iter()/iter_mut() return impl Iterator
impl<T: Model> Model for crate::refs::Arena<T> {
    fn serialize_params<F: crate::utils::Float>(&mut self, data: &mut std::vec::Vec<F>) {
        for item in self.iter_mut() { item.serialize_params(data); }
    }
    fn deserialize_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        for item in self.iter_mut() { item.deserialize_params(data); }
    }
    fn update_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        for item in self.iter_mut() { item.update_params(data); }
    }
    fn update_self(&mut self) {
        for item in self.iter_mut() { item.update_self(); }
    }
    fn advance_params<F: crate::utils::Float>(&mut self, params: &mut [F]) {
        for item in self.iter_mut() { item.advance_params(params); }
    }
    fn collect_param_blocks(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        for item in self.iter() { item.collect_param_blocks(out); }
    }
    fn release_blocks(&mut self) {
        for item in self.iter_mut() { item.release_blocks(); }
    }
    fn serialize_size(&self) -> u32 {
        self.iter().map(|item| item.serialize_size()).sum()
    }
}

impl<T: Model> Model for Option<T> {
    fn serialize_params<F: crate::utils::Float>(&mut self, data: &mut std::vec::Vec<F>) {
        if let Some(inner) = self { inner.serialize_params(data); }
    }
    fn deserialize_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if let Some(inner) = self { inner.deserialize_params(data); }
    }
    fn update_params<F: crate::utils::Float>(&mut self, data: &[F]) {
        if let Some(inner) = self { inner.update_params(data); }
    }
    fn update_self(&mut self) {
        if let Some(inner) = self { inner.update_self(); }
    }
    fn advance_params<F: crate::utils::Float>(&mut self, params: &mut [F]) {
        if let Some(inner) = self { inner.advance_params(params); }
    }
    fn serialize_size(&self) -> u32 {
        if let Some(inner) = self { inner.serialize_size() } else { 0 }
    }
    fn collect_param_blocks(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        if let Some(inner) = self { inner.collect_param_blocks(out); }
    }
    fn release_blocks(&mut self) {
        if let Some(inner) = self { inner.release_blocks(); }
    }
}

// ---------------------------------------------------------------------------
// SelfBlock / CrossBlock -- per-constraint hessian block storage
// ---------------------------------------------------------------------------

/// Upper triangle index: element (i,j) with i<=j in an NxN symmetric matrix.
#[inline]
fn tri_idx(n: usize, i: usize, j: usize) -> usize {
    i * (2 * n - i - 1) / 2 + j
}

use arael_faer::{value_index, ValueIndex};

/// Where a block scatters into the assembled Hessian's value buffer.
///
/// A block with a static tile shape needs no per-scalar position map: the
/// whole tile follows from its origin and column stride, as
/// `origin + (col - col_start) * stride + (row - row_start)`, with the two
/// starts read off the block's own parameter indices. Storing this pair
/// beside the indices replaces a map that runs about one entry per Hessian
/// value.
///
/// A zero stride means there is no tile to walk, and the block scatters
/// through the per-scalar map instead: either the pattern is not
/// tile-expanded ([`MAPPED`](TilePosition::MAPPED)) or the block is entirely
/// fixed and scatters nothing at all ([`UNBOUND`](TilePosition::UNBOUND)).
#[derive(Clone, Copy, Debug)]
struct TilePosition {
    base: ValueIndex,
    stride: ValueIndex,
}

impl TilePosition {
    /// All parameters fixed: no tile, and the index walk emits nothing.
    const UNBOUND: Self = TilePosition { base: ValueIndex::MAX, stride: 0 };
    /// Pattern not tile-expanded: fall back to the per-scalar map.
    const MAPPED: Self = TilePosition { base: 0, stride: 0 };

    /// True if this block scatters through its tile rather than the map.
    #[inline]
    fn tiled(&self) -> bool { self.stride != 0 }

    /// Narrow a resolved position into the packed 32-bit slot.
    #[inline]
    fn bound(base: usize, stride: usize) -> Self {
        TilePosition { base: value_index(base), stride: value_index(stride) }
    }
}

/// Append a block's scatter target to the position stream: two words per
/// block, ahead of the per-entry positions a mapped block also pushes.
/// The block itself keeps nothing, so the stream is the whole binding and
/// a fresh set of blocks scatters through it without being rebound.
#[inline]
fn push_tile(out: &mut std::vec::Vec<ValueIndex>, pos: TilePosition) {
    out.push(pos.base);
    out.push(pos.stride);
}

/// Read the next block's scatter target from the position stream.
#[inline]
fn take_tile(positions: &[ValueIndex], cursor: &mut usize) -> TilePosition {
    assert!(*cursor + 2 <= positions.len(),
        "Hessian scatter ran past the bound pattern: the stores were never bound \
         to it (LmProblemInternals::bind_hessian_positions), their emission order \
         changed within the solve, or the pattern was bound at another store count");
    let pos = TilePosition { base: positions[*cursor], stride: positions[*cursor + 1] };
    *cursor += 2;
    pos
}

/// Smallest live index in `indices`, or `u32::MAX` if every slot is fixed.
///
/// Parameters serialize in declaration order and the index arrays are filled
/// in that same order, so live indices ascend with the slot and the first one
/// found is the smallest. `bind_tile` asserts this before relying on it.
#[inline]
fn tile_start(indices: &[u32]) -> u32 {
    for &i in indices {
        if i != u32::MAX {
            return i;
        }
    }
    u32::MAX
}

/// How a backend hands out scatter targets when a root binds its block
/// stores to an assembled pattern (`LmProblemInternals::bind_hessian_positions`).
pub enum HessianBinder<'a> {
    /// Tile-expanded pattern: every stored cell holds a full dense tile, so
    /// one lookup fixes a whole block and only the tile's origin and column
    /// stride need keeping.
    Tiled(&'a mut dyn FnMut(u32, u32) -> (usize, usize)),
    /// Pattern built from a COO pass: a cell's entries are not contiguous in
    /// the value buffer, so every scalar needs its own position and the
    /// blocks fall back to the per-scalar map.
    Scalar(&'a mut dyn FnMut(u32, u32) -> usize),
}

/// Bind one tile, checking the ascending-index invariant that lets
/// [`tile_start`] stop at the first live slot and lets the assembly derive a
/// local coordinate by subtraction.
#[inline]
fn bind_tile(
    bind: &mut dyn FnMut(u32, u32) -> (usize, usize),
    row: &[u32],
    col: &[u32],
) -> TilePosition {
    let (r, c) = (tile_start(row), tile_start(col));
    if r == u32::MAX || c == u32::MAX {
        return TilePosition::UNBOUND;
    }
    // Runs once per block per setup, so it is cheap next to the map it
    // replaces, and the failure it guards against is a silently wrong
    // Hessian rather than a crash.
    assert!(ascending(row) && ascending(col), "live parameter indices must ascend with slot order");
    let (base, stride) = bind(r.min(c), r.max(c));
    TilePosition::bound(base, stride)
}

/// True if the live entries of `indices` are strictly ascending.
fn ascending(indices: &[u32]) -> bool {
    let mut last = None;
    for &i in indices {
        if i == u32::MAX {
            continue;
        }
        if last.is_some_and(|l| i <= l) {
            return false;
        }
        last = Some(i);
    }
    true
}


/// Declares that one entity has a diagonal Hessian block.
///
/// The block itself is a marker. Its values -- the upper triangle of the
/// Gauss-Newton Hessian approximation (2·dr·dr^T) over that entity's own
/// parameters, and the entity's gradient -- live in a [`SelfBlockArray`]
/// in the block store, the generated `<Root>Blocks` struct the solve keeps
/// in its [`Context`](crate::threads::Context). The marker's
/// [`slot`](Self::slot) is its index into that array.
/// `N` equals `A::PARAM_COUNT`, `M` the triangle length `N*(N+1)/2`.
/// `T` is the float type (f32 or f64, default f64).
///
/// Declared as a field on the entity, one per entity that has parameters.
#[derive(Clone)]
pub struct SelfBlock<A, const N: usize, const M: usize, T: crate::utils::Float = f64> {
    /// This block's position in its container's array of the solve's
    /// block store, or `u32::MAX` before the store's build has wired it.
    /// An entity reached through a `Ref` has no ordinal at the write
    /// site, so the block carries its own.
    slot: u32,
    _marker: std::marker::PhantomData<(A, T)>,
}

impl<A, const N: usize, const M: usize, T: crate::utils::Float> Default for SelfBlock<A, N, M, T> {
    fn default() -> Self { Self::new() }
}

impl<A, const N: usize, const M: usize, T: crate::utils::Float> SelfBlock<A, N, M, T> {
    /// Compile-time guard that `M` matches `N` (the macro sets both; a
    /// hand-written mismatch fails here rather than corrupting silently).
    const CHECK_M: () = assert!(M == N * (N + 1) / 2, "SelfBlock: M must equal N*(N+1)/2");

    /// A new, unwired block.
    pub fn new() -> Self {
        let () = Self::CHECK_M;
        SelfBlock { slot: u32::MAX, _marker: std::marker::PhantomData }
    }

    /// This block's position in its container's store array; `u32::MAX`
    /// until the build wires it.
    #[inline]
    pub fn slot(&self) -> u32 { self.slot }

    /// Set this block's store position. The build calls it once per
    /// instance, in container order.
    #[inline]
    pub fn set_slot(&mut self, slot: u32) { self.slot = slot; }

    /// No-op: the block owns no storage to free. Kept so a generated
    /// `release_blocks` can call it uniformly.
    pub fn release(&mut self) {}
}

/// The heap-backed twin of [`SelfBlock`], kept as an alias of it.
///
/// Up to 0.8.3 a block field was the container: `SelfBlock` embedded its
/// Hessian triangle in the entity struct as a fixed array, and this twin
/// held the same behind a `Box` so it could be freed between solves and
/// skipped for a frozen sub-tree. From 0.9.0 a block field declares a
/// block and holds no values, so there is nothing to box and nothing to
/// choose between. Use [`SelfBlock`].
#[deprecated(since = "0.9.0", note = "a block declares, it does not store; use SelfBlock")]
pub type BoxedSelfBlock<A, const N: usize, const M: usize, T = f64> = SelfBlock<A, N, M, T>;

// The self-block arithmetic, shared by [`SelfBlock`] and the
// [`SelfBlockArray`] slabs: one block is its N parameter indices and the
// M entries of its upper triangle.

/// The entity's parameter span as `(offset, width)`; nothing when every
/// slot is fixed.
#[inline]
fn self_param_block<const N: usize>(indices: &[u32; N], out: &mut std::vec::Vec<(u32, u32)>) {
    let mut min = u32::MAX;
    let mut count = 0u32;
    for &i in indices {
        if i != u32::MAX {
            if i < min { min = i; }
            count += 1;
        }
    }
    if count > 0 {
        out.push((min, count));
    }
}

/// One representative coordinate of the block's cell: its anchor on the
/// diagonal of the entity partition.
#[inline]
fn self_cells<const N: usize>(indices: &[u32; N], out: &mut std::vec::Vec<(u32, u32)>) {
    let min = tile_start(indices);
    if min != u32::MAX {
        out.push((min, min));
    }
}

/// The block's scatter target, pushed into the stream; a scalar binder
/// follows it with one position per entry.
#[inline]
fn self_bind<const N: usize>(
    indices: &[u32; N], binder: &mut HessianBinder, out: &mut std::vec::Vec<ValueIndex>,
) {
    let resolve = match binder {
        HessianBinder::Tiled(bind) => {
            push_tile(out, bind_tile(*bind, indices, indices));
            return;
        }
        HessianBinder::Scalar(resolve) => resolve,
    };
    push_tile(out, TilePosition::MAPPED);
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            out.push(value_index(resolve(lo, hi)));
        }
    }
}

/// The dense symmetric scatter of one block.
#[inline]
fn self_dense<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M], out: &mut [F],
) {
    let n_total = (out.len() as f64).sqrt() as usize;
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        let gi = gi as usize;
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let gj = gj as usize;
            let val = F::from(hessian[tri_idx(N, i, j)]).unwrap();
            out[gi * n_total + gj] += val;
            if gi != gj {
                out[gj * n_total + gi] += val;
            }
        }
    }
}

/// The band scatter of one block (column-major, (kd+1)*n).
#[inline]
fn self_band<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M], band: &mut [F], kd: usize,
) -> Result<(), crate::simple_lm::BandOverflow> {
    let ldab = kd + 1;
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        let gi = gi as usize;
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let gj = gj as usize;
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            if hi - lo > kd {
                return Err(crate::simple_lm::BandOverflow { row: lo, col: hi, kd });
            }
            band[(kd + lo - hi) + hi * ldab] += F::from(hessian[tri_idx(N, i, j)]).unwrap();
        }
    }
    Ok(())
}

/// The COO scatter of one block, upper triangle only.
#[inline]
fn self_coo<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M], coo: &mut crate::simple_lm::CooMatrix<F>,
) {
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            coo.push(lo, hi, F::from(hessian[tri_idx(N, i, j)]).unwrap());
        }
    }
}

/// The direct CSC scatter of one block, by position lookup.
#[inline]
fn self_direct<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M], csc: &mut crate::simple_lm::CscMatrix<F>,
) {
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        for j in i..N {
            let gj = indices[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            let val = F::from(hessian[tri_idx(N, i, j)]).unwrap();
            if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                csc.vals[pos] += val;
            }
        }
    }
}

/// The indexed scatter of one block through the tile the stream holds for
/// it; a mapped block falls back to the per-entry positions behind it.
#[inline]
fn self_indexed<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M],
    vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize,
) {
    let pos = take_tile(positions, cursor);
    if !pos.tiled() {
        return self_mapped(indices, hessian, vals, positions, cursor);
    }
    let start = tile_start(indices) as usize;
    let (base, stride) = (pos.base as usize, pos.stride as usize);
    // Column offset of each live slot: invariant across the outer loop.
    let mut col = [0usize; N];
    for (c, &g) in std::iter::zip(&mut col, indices) {
        if g != u32::MAX {
            *c = (g as usize - start) * stride;
        }
    }
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        let row = base + (gi as usize - start);
        let tri = i * (2 * N - i - 1) / 2;
        for j in i..N {
            if indices[j] == u32::MAX { continue; }
            vals[row + col[j]] += F::from(hessian[tri + j]).unwrap();
        }
    }
}

/// The untiled case of [`self_indexed`]: one cached position per entry,
/// `cursor` advancing in lockstep with the block traversal. An all-fixed
/// block emits nothing and so consumes none.
#[inline]
fn self_mapped<const N: usize, const M: usize, T: crate::utils::Float, F: crate::utils::Float>(
    indices: &[u32; N], hessian: &[T; M],
    vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize,
) {
    for i in 0..N {
        let gi = indices[i];
        if gi == u32::MAX { continue; }
        for j in i..N {
            if indices[j] == u32::MAX { continue; }
            vals[positions[*cursor] as usize] += F::from(hessian[tri_idx(N, i, j)]).unwrap();
            *cursor += 1;
        }
    }
}

/// A slab of self blocks with the indices apart from the values: per
/// entity its container slot and its N parameter indices in one array,
/// the M entries of its upper triangle and its N gradient entries in one
/// flat array. A root's generated block store holds one per entity
/// container, for the entities its range writes (see [`crate::threads`]).
///
/// The build pushes the indices only and then calls
/// [`finish`](Self::finish), which sizes the value array in one zeroed
/// allocation and keeps it when the length has not changed. The sweep
/// zeroes the whole slab before it writes.
///
/// One entity's triangle and its gradient stash are adjacent in `data`,
/// at stride `M + N`: a residual writes both, so they share the region
/// the write pulls in rather than sitting in two arrays a thread has to
/// stream separately.
#[derive(Clone)]
pub struct SelfBlockArray<const N: usize, const M: usize, T: crate::utils::Float = f64> {
    entity: std::vec::Vec<u32>,
    indices: std::vec::Vec<[u32; N]>,
    /// Per entity: `M` triangle entries, then `N` gradient entries.
    data: std::vec::Vec<T>,
    /// Global slot to this slab's position, for a slab holding only some
    /// of the entities. Empty on a slab that holds them all, where the
    /// marker's slot addresses the data directly.
    map: std::vec::Vec<u32>,
}

impl<const N: usize, const M: usize, T: crate::utils::Float> Default for SelfBlockArray<N, M, T> {
    fn default() -> Self { Self::new() }
}

impl<const N: usize, const M: usize, T: crate::utils::Float> SelfBlockArray<N, M, T> {
    const CHECK_M: () = assert!(M == N * (N + 1) / 2, "SelfBlockArray: M must equal N*(N+1)/2");
    /// One entity's span of [`data`](Self::data): the triangle then the stash.
    const STRIDE: usize = M + N;

    /// An empty slab.
    pub fn new() -> Self {
        let () = Self::CHECK_M;
        SelfBlockArray {
            entity: std::vec::Vec::new(),
            indices: std::vec::Vec::new(),
            data: std::vec::Vec::new(),
            map: std::vec::Vec::new(),
        }
    }

    /// The number of entities.
    pub fn len(&self) -> usize { self.entity.len() }

    /// True if the slab holds no entity.
    pub fn is_empty(&self) -> bool { self.entity.is_empty() }

    /// Drop every entity's indices; the value storage stays for `finish`.
    ///
    /// The map is reset too, because the build reads it to ask whether it
    /// has already given a slot a place. A stale entry from the last solve
    /// would answer yes and the entity would never be pushed.
    pub fn clear(&mut self) {
        self.entity.clear();
        self.indices.clear();
        for m in &mut self.map { *m = u32::MAX; }
    }

    /// True if `slot` already has a place in this slab.
    #[inline]
    pub fn holds(&self, slot: usize) -> bool {
        self.map.get(slot).is_some_and(|&k| k != u32::MAX)
    }

    /// Room for `n` more entities' indices.
    pub fn reserve(&mut self, n: usize) {
        self.entity.reserve(n);
        self.indices.reserve(n);
    }

    /// Append the entity at container slot `entity` with its parameter
    /// indices, and return the slab slot it took.
    #[inline]
    pub fn push(&mut self, entity: u32, indices: &[u32; N]) -> u32 {
        let k = self.entity.len() as u32;
        self.entity.push(entity);
        self.indices.push(*indices);
        k
    }

    /// Room for `n` global slots in the map.
    ///
    /// The entries are not cleared, and do not need to be: a slot this
    /// slab never claims is never read, because a sweep walks the very
    /// instances the build walked and the build claims a place for every
    /// one of them. Only the length has to be right.
    pub fn map_resize(&mut self, n: usize) {
        if self.map.len() < n { self.map.resize(n, u32::MAX); }
    }

    /// Take global `slot` into this slab, returning where it landed.
    pub fn push_at(&mut self, slot: u32, entity: u32, indices: &[u32; N]) -> u32 {
        let k = self.push(entity, indices);
        if self.map.len() <= slot as usize {
            self.map.resize(slot as usize + 1, u32::MAX);
        }
        self.map[slot as usize] = k;
        k
    }

    /// Where global `slot` sits in this slab.
    ///
    /// A slot the build never claimed lands here as `u32::MAX` and the
    /// write would run off the end. That means the build's idea of which
    /// entities a store touches disagrees with what its sweep writes, so
    /// it says which slot rather than panicking on an index far away.
    #[inline(always)]
    pub fn slab_of(&self, slot: usize) -> usize {
        let k = self.map[slot];
        debug_assert!(k != u32::MAX,
            "slot {} is written by this store's sweep but the build gave it no place \
             in the slab", slot);
        k as usize
    }

    /// Size the value array to the entities pushed: a fresh zeroed
    /// allocation when the length changed, else the existing storage,
    /// whose stale values [`zero`](Self::zero) clears.
    pub fn finish(&mut self) {
        let n = self.entity.len() * Self::STRIDE;
        if self.data.len() != n {
            self.data = vec![T::zero(); n];
        }
    }

    /// Reset every triangle and every gradient stash.
    pub fn zero(&mut self) {
        self.data.fill(T::zero());
    }

    /// The container slot of entity `k`.
    pub fn entity(&self, k: usize) -> u32 { self.entity[k] }

    /// The parameter indices of entity `k` (`u32::MAX` for a fixed one).
    pub fn indices(&self, k: usize) -> &[u32; N] { &self.indices[k] }

    #[inline]
    fn block(&self, k: usize) -> &[T; M] {
        let base = k * Self::STRIDE;
        (&self.data[base..base + M]).try_into().unwrap()
    }

    /// Add one residual's contribution to entity `k`: `2 r dr` into its
    /// gradient stash and `2 dr dr^T` into its triangle, every slot,
    /// fixed or not. Branch-free: a fixed parameter's row is accumulated
    /// like any other and dropped by
    /// [`scatter_grad`](Self::scatter_grad) and the Hessian walks, which
    /// read the indices.
    #[inline]
    pub fn add_residual(&mut self, k: usize, r: T, dr: &[T; N]) {
        self.add_scaled(k, T::two(), r, dr);
    }

    /// [`add_residual`](Self::add_residual) against a slab holding only
    /// some of the entities: `slot` is the marker's, the map says where
    /// it sits here.
    #[inline]
    pub fn add_residual_mapped(&mut self, slot: usize, r: T, dr: &[T; N]) {
        let k = self.slab_of(slot);
        self.add_scaled(k, T::two(), r, dr);
    }

    /// [`add_residual`](Self::add_residual) scaled by the loss weight `w`.
    #[inline]
    pub fn add_residual_with_loss(&mut self, k: usize, w: T, r: T, dr: &[T; N]) {
        self.add_scaled(k, T::two() * w, r, dr);
    }

    /// [`add_residual_with_loss`](Self::add_residual_with_loss) through
    /// the map, as [`add_residual_mapped`](Self::add_residual_mapped).
    #[inline]
    pub fn add_residual_with_loss_mapped(&mut self, slot: usize, w: T, r: T, dr: &[T; N]) {
        let k = self.slab_of(slot);
        self.add_scaled(k, T::two() * w, r, dr);
    }

    // The two forms a sweep picks between, as one call it can make without
    // knowing which store it has. `MAPPED` is a constant of the sweep's
    // instantiation, so each copy keeps one arm and no test survives.

    /// [`add_residual`](Self::add_residual), through the map or not.
    #[inline(always)]
    pub fn add_residual_at<const MAPPED: bool>(&mut self, slot: usize, r: T, dr: &[T; N]) {
        if MAPPED { self.add_residual_mapped(slot, r, dr) } else { self.add_residual(slot, r, dr) }
    }

    /// [`add_residual_with_loss`](Self::add_residual_with_loss), likewise.
    #[inline(always)]
    pub fn add_residual_with_loss_at<const MAPPED: bool>(&mut self, slot: usize, w: T, r: T, dr: &[T; N]) {
        if MAPPED {
            self.add_residual_with_loss_mapped(slot, w, r, dr)
        } else {
            self.add_residual_with_loss(slot, w, r, dr)
        }
    }

    #[inline]
    fn add_scaled(&mut self, k: usize, scale: T, r: T, dr: &[T; N]) {
        let base = k * Self::STRIDE;
        let (h, g) = self.data[base..base + Self::STRIDE].split_at_mut(M);
        let h: &mut [T; M] = h.try_into().unwrap();
        let g: &mut [T; N] = g.try_into().unwrap();
        let sr = scale * r;
        for i in 0..N {
            g[i] += sr * dr[i];
            let tdi = scale * dr[i];
            for j in i..N {
                h[tri_idx(N, i, j)] += tdi * dr[j];
            }
        }
    }

    /// Add every entity's stash into the global gradient at its live
    /// indices; a fixed slot has none and is dropped here.
    pub fn scatter_grad<F: crate::utils::Float>(&self, grad: &mut [F]) {
        for k in 0..self.entity.len() {
            let idx = &self.indices[k];
            let base = k * Self::STRIDE + M;
            let g = &self.data[base..base + N];
            for i in 0..N {
                let gi = idx[i];
                if gi == u32::MAX { continue; }
                grad[gi as usize] += F::from(g[i]).unwrap();
            }
        }
    }

    /// Append entity `k`'s `(offset, width)` span of the flat parameter
    /// vector, or nothing when all of its params are fixed.
    pub fn collect_param_block(&self, k: usize, out: &mut std::vec::Vec<(u32, u32)>) {
        self_param_block(&self.indices[k], out);
    }

    /// Append one representative scalar coordinate of each entity's
    /// triangle, over every entity. Structure only, no values.
    pub fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        for idx in &self.indices {
            self_cells(idx, out);
        }
    }

    /// Record where each entity's triangle scatters in the assembled value
    /// buffer, over every entity. See
    /// [`Model::bind_hessian_positions`].
    pub fn bind_hessian_positions(&mut self, binder: &mut HessianBinder, out: &mut std::vec::Vec<ValueIndex>) {
        for idx in &self.indices {
            self_bind(idx, binder, out);
        }
    }

    /// Add every entity's triangle into the dense symmetric Hessian,
    /// both sides of the diagonal.
    pub fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        for k in 0..self.entity.len() {
            self_dense(&self.indices[k], self.block(k), hessian);
        }
    }

    /// Add every entity's triangle into an upper-band Hessian of
    /// half-bandwidth `kd` (LAPACK layout). Errors when an entry falls
    /// outside the band.
    pub fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
        -> Result<(), crate::simple_lm::BandOverflow>
    {
        for k in 0..self.entity.len() {
            self_band(&self.indices[k], self.block(k), band, kd)?;
        }
        Ok(())
    }

    /// Push every entity's triangle into a COO Hessian as `(row, col,
    /// value)` triplets.
    pub fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        for k in 0..self.entity.len() {
            self_coo(&self.indices[k], self.block(k), coo);
        }
    }

    /// Add every entity's triangle straight into an already-patterned CSC
    /// Hessian, looking each entry up by column.
    pub fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        for k in 0..self.entity.len() {
            self_direct(&self.indices[k], self.block(k), csc);
        }
    }

    /// Add every entity's triangle into the assembled value buffer through
    /// the positions [`bind_hessian_positions`](Self::bind_hessian_positions)
    /// recorded, reading them in the same order.
    pub fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        for k in 0..self.entity.len() {
            self_indexed(&self.indices[k], self.block(k), vals, positions, cursor);
        }
    }
}

/// Declares that two entities are coupled in the Hessian.
///
/// The block itself is a marker. Its values -- the rectangular A x B cross
/// pairs, and nothing else -- live in a [`CrossBlockArray`] in the block
/// store, the generated `<Root>Blocks` struct the solve keeps in its
/// [`Context`](crate::threads::Context). The marker's
/// [`slot`](Self::slot) is its index into that array. A's gradient and A-A
/// diagonal belong to A's own [`SelfBlock`]; same for B. So every
/// `∂r/∂p_i · ∂r/∂p_j` pair is written in exactly one place.
///
/// `NA = A::PARAM_COUNT`, `NB = B::PARAM_COUNT`, and the tile is NA×NB
/// row-major. `T` is the float type (f32 or f64, default f64).
///
/// Placement: normally a field on the constraint struct itself (one tile
/// per constraint instance). When many constraints couple the SAME two
/// entities, declare ONE CrossBlock on a struct that holds the constraint
/// collection and name it with the `constraint(parent.<field>, ...)` block
/// spec -- every instance then accumulates into that shared tile. With
/// Ref fields on the constraint (declared in `[Ref<A>, Ref<B>]` order)
/// all instances under one parent must reference the same (A, B) pair --
/// a mismatch panics at solve setup; with none, the parent's own ref
/// fields fill the slots and bodies read `parent.<ref>.<field>`. See
/// docs/MODEL.md, "Shared CrossBlock on a containing parent".
pub struct CrossBlock<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float = f64> {
    /// See [`SelfBlock::slot`].
    slot: u32,
    _marker: std::marker::PhantomData<(A, B, T)>,
}

impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> Clone for CrossBlock<A, B, NA, NB, P, T> {
    fn clone(&self) -> Self {
        CrossBlock { slot: self.slot, _marker: std::marker::PhantomData }
    }
}

impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> Default for CrossBlock<A, B, NA, NB, P, T> {
    fn default() -> Self { Self::new() }
}

impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> CrossBlock<A, B, NA, NB, P, T> {
    /// Compile-time guard that `P` matches `NA*NB` (set by the macro).
    const CHECK_P: () = assert!(P == NA * NB, "CrossBlock: P must equal NA*NB");

    /// A new, unwired tile.
    pub fn new() -> Self {
        let () = Self::CHECK_P;
        CrossBlock { slot: u32::MAX, _marker: std::marker::PhantomData }
    }

    /// Return the number of parameters belonging to model A.
    pub fn na(&self) -> usize { NA }
    /// Return the number of parameters belonging to model B.
    pub fn nb(&self) -> usize { NB }

    /// See [`SelfBlock::slot`].
    #[inline]
    pub fn slot(&self) -> u32 { self.slot }

    /// See [`SelfBlock::set_slot`].
    #[inline]
    pub fn set_slot(&mut self, slot: u32) { self.slot = slot; }

    /// No-op: the tile owns no storage to free.
    pub fn release(&mut self) {}
}

// The cross-block arithmetic, shared by [`CrossBlock`] and the
// [`CrossBlockArray`] lists: one block is its A and B indices, its tile
// position and its NA x NB row-major values.

/// `values += scale * dr_a * dr_b^T`, skipping the rows whose `dr_a` is zero.
#[inline]
fn cross_add<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float>(
    values: &mut [T; P], scale: T, dr_a: &[T; NA], dr_b: &[T; NB],
) {
    for i in 0..NA {
        let dai = dr_a[i];
        if dai == T::zero() { continue; }
        let row = i * NB;
        for j in 0..NB {
            values[row + j] += scale * dai * dr_b[j];
        }
    }
}

/// The dense symmetric scatter of one block.
#[inline]
fn cross_dense<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], hessian: &mut [F],
) {
    let n_total = (hessian.len() as f64).sqrt() as usize;
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let gi = gi as usize;
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let gj = gj as usize;
            let val = F::from(values[row + j]).unwrap();
            hessian[gi * n_total + gj] += val;
            hessian[gj * n_total + gi] += val;
        }
    }
}

/// The band scatter of one block (column-major, (kd+1)*n).
#[inline]
fn cross_band<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], band: &mut [F], kd: usize,
) -> Result<(), crate::simple_lm::BandOverflow> {
    let ldab = kd + 1;
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let gi = gi as usize;
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let gj = gj as usize;
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            if hi - lo > kd {
                return Err(crate::simple_lm::BandOverflow { row: lo, col: hi, kd });
            }
            let val = F::from(values[row + j]).unwrap();
            // Aliased slots (same entity in both refs): the triangle
            // stores each symmetric pair once, so the diagonal needs
            // both of the 2*dr_a*dr_b contributions explicitly.
            let val = if gi == gj { val + val } else { val };
            band[(kd + lo - hi) + hi * ldab] += val;
        }
    }
    Ok(())
}

/// One representative coordinate of a block's cell: the two entities are
/// contiguous spans, so every pair lands in one cell of the entity
/// partition (the diagonal cell when aliased).
#[inline]
fn cross_cells<const NA: usize, const NB: usize>(a: &[u32; NA], b: &[u32; NB], out: &mut std::vec::Vec<(u32, u32)>) {
    let min_live = |idx: &[u32]| {
        let mut min = u32::MAX;
        for &i in idx {
            if i != u32::MAX && i < min { min = i; }
        }
        min
    };
    let (ma, mb) = (min_live(a), min_live(b));
    if ma != u32::MAX && mb != u32::MAX {
        out.push((ma.min(mb), ma.max(mb)));
    }
}

/// The tile position of one block under `binder`; a scalar binder pushes
/// one position per entry and marks the block mapped.
#[inline]
fn cross_bind<const NA: usize, const NB: usize>(
    a: &[u32; NA], b: &[u32; NB], binder: &mut HessianBinder, out: &mut std::vec::Vec<ValueIndex>,
) {
    let resolve = match binder {
        HessianBinder::Tiled(bind) => {
            push_tile(out, bind_tile(*bind, a, b));
            return;
        }
        HessianBinder::Scalar(resolve) => resolve,
    };
    push_tile(out, TilePosition::MAPPED);
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            out.push(value_index(resolve(lo, hi)));
        }
    }
}

/// The COO scatter of one block, upper triangle only.
#[inline]
fn cross_coo<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], coo: &mut crate::simple_lm::CooMatrix<F>,
) {
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            let val = F::from(values[row + j]).unwrap();
            // Aliased diagonal: see cross_band.
            let val = if gi == gj { val + val } else { val };
            coo.push(lo, hi, val);
        }
    }
}

/// The direct CSC scatter of one block, by position lookup.
#[inline]
fn cross_direct<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], csc: &mut crate::simple_lm::CscMatrix<F>,
) {
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            let val = F::from(values[row + j]).unwrap();
            // Aliased diagonal: see cross_band.
            let val = if gi == gj { val + val } else { val };
            if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                csc.vals[pos] += val;
            }
        }
    }
}

/// The indexed scatter of one block through its bound tile; see the note
/// on [`SelfBlock::accumulate_hessian_sparse_indexed`].
#[inline]
fn cross_indexed<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P],
    vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize,
) {
    let pos = take_tile(positions, cursor);
    if !pos.tiled() {
        return cross_mapped(a, b, values, vals, positions, cursor);
    }
    let (sa, sb) = (tile_start(a), tile_start(b));
    let (base, stride) = (pos.base as usize, pos.stride as usize);
    if sa == sb {
        // Aliased: both slots index one entity, so the pairs land on a
        // diagonal tile and which side is the row flips per element.
        return cross_aliased(a, b, values, vals, base, stride, sa as usize);
    }
    // The tile holds the upper block triangle, so the lower-numbered
    // entity walks the rows and the other walks the columns.
    let (step_a, step_b) = if sa < sb { (1, stride) } else { (stride, 1) };
    let (sa, sb) = (sa as usize, sb as usize);
    // Offset of each live B slot: invariant across the outer loop.
    let mut off_b = [0usize; NB];
    for (o, &g) in std::iter::zip(&mut off_b, b) {
        if g != u32::MAX {
            *o = (g as usize - sb) * step_b;
        }
    }
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let pos = base + (gi as usize - sa) * step_a;
        let row = i * NB;
        for j in 0..NB {
            if b[j] == u32::MAX { continue; }
            vals[pos + off_b[j]] += F::from(values[row + j]).unwrap();
        }
    }
}

/// The aliased case of [`cross_indexed`]: one entity in both slots, every
/// pair on its diagonal tile.
#[inline]
fn cross_aliased<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], vals: &mut [F], base: usize, stride: usize, start: usize,
) {
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
            let val = F::from(values[row + j]).unwrap();
            // The triangle stores each symmetric pair once, so a pair that
            // lands on the diagonal needs both contributions.
            let val = if gi == gj { val + val } else { val };
            vals[base + (hi as usize - start) * stride + (lo as usize - start)] += val;
        }
    }
}

/// The untiled case of [`cross_indexed`]: one cached position per entry.
#[inline]
fn cross_mapped<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float, F: crate::utils::Float>(
    a: &[u32; NA], b: &[u32; NB], values: &[T; P], vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize,
) {
    for i in 0..NA {
        let gi = a[i];
        if gi == u32::MAX { continue; }
        let row = i * NB;
        for j in 0..NB {
            let gj = b[j];
            if gj == u32::MAX { continue; }
            let val = F::from(values[row + j]).unwrap();
            // Aliased diagonal: see cross_band.
            let val = if gi == gj { val + val } else { val };
            vals[positions[*cursor] as usize] += val;
            *cursor += 1;
        }
    }
}

/// A list of cross blocks with the indices apart from the values: per
/// block the A and B indices and the tile position in one array, the
/// `NA x NB` values in one flat array. A block store holds its cross
/// blocks this way. The build pushes the indices only and
/// then calls [`finish`](Self::finish), which sizes the value array in
/// one zeroed allocation (never written by the build; the sweep zeroes
/// each block before it writes it) and keeps it when the length has not
/// changed.
#[derive(Clone)]
pub struct CrossBlockArray<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float = f64> {
    a: std::vec::Vec<[u32; NA]>,
    b: std::vec::Vec<[u32; NB]>,
    values: std::vec::Vec<T>,
    /// The global slot this array's first block holds. A range of the walk
    /// is a contiguous run of the numbering, so an array covering one runs
    /// from here; zero on an array covering all of them.
    base: u32,
}

impl<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> Default for CrossBlockArray<NA, NB, P, T> {
    fn default() -> Self { Self::new() }
}

/// One block of a [`CrossBlockArray`] list, borrowed for the sweep's writes.
pub struct CrossBlockMut<'a, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> {
    values: &'a mut [T; P],
}

impl<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> CrossBlockMut<'_, NA, NB, P, T> {
    /// Reset the values to zero.
    #[inline]
    pub fn zero(&mut self) {
        self.values.fill(T::zero());
    }

    /// Add one residual's cross pairs `2 dr_a dr_b^T` into the tile. The
    /// residual itself belongs to the two entities' gradients, not here.
    #[inline]
    pub fn add_residual_cross(&mut self, _r: T, dr_a: &[T; NA], dr_b: &[T; NB]) {
        cross_add(self.values, T::two(), dr_a, dr_b);
    }

    /// [`add_residual_cross`](Self::add_residual_cross) with the pairs
    /// scaled by the loss weight `w`. `w = 1` is the plain form.
    #[inline]
    pub fn add_residual_cross_with_loss(&mut self, w: T, _r: T, dr_a: &[T; NA], dr_b: &[T; NB]) {
        cross_add(self.values, T::two() * w, dr_a, dr_b);
    }
}

impl<const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> CrossBlockArray<NA, NB, P, T> {
    const CHECK_P: () = assert!(P == NA * NB, "CrossBlockArray: P must equal NA*NB");

    /// An empty list.
    pub fn new() -> Self {
        let () = Self::CHECK_P;
        CrossBlockArray {
            a: std::vec::Vec::new(),
            b: std::vec::Vec::new(),
            values: std::vec::Vec::new(),
            base: 0,
        }
    }

    /// The number of blocks.
    pub fn len(&self) -> usize { self.a.len() }

    /// True if the list holds no block.
    pub fn is_empty(&self) -> bool { self.a.is_empty() }

    /// Drop every block's indices; the value storage stays for `finish`.
    pub fn clear(&mut self) {
        self.a.clear();
        self.b.clear();
    }

    /// Room for `n` more blocks' indices.
    pub fn reserve(&mut self, n: usize) {
        self.a.reserve(n);
        self.b.reserve(n);
    }

    /// Append a block with its parameter indices, unbound.
    #[inline]
    pub fn push(&mut self, a: &[u32; NA], b: &[u32; NB]) {
        self.a.push(*a);
        self.b.push(*b);
    }

    /// Size the value array to the blocks pushed: a fresh zeroed
    /// allocation when the length changed, else the existing storage,
    /// whose stale values the sweep overwrites.
    pub fn finish(&mut self) {
        let n = self.a.len() * P;
        if self.values.len() != n {
            self.values = vec![T::zero(); n];
        }
    }

    /// Reset every tile.
    pub fn zero(&mut self) {
        self.values.fill(T::zero());
    }

    /// Block `k` for writing.
    #[inline]
    pub fn block_mut(&mut self, k: usize) -> CrossBlockMut<'_, NA, NB, P, T> {
        let values: &mut [T; P] = (&mut self.values[k * P..(k + 1) * P]).try_into().unwrap();
        CrossBlockMut { values }
    }

    /// The global slot this array opens on.
    pub fn base(&self) -> u32 { self.base }

    /// Set the global slot this array opens on.
    pub fn set_base(&mut self, base: u32) { self.base = base; }

    /// [`block_mut`](Self::block_mut) by the marker's global slot, for an
    /// array covering one run of the walk rather than all of it.
    #[inline]
    pub fn block_mut_based(&mut self, slot: usize) -> CrossBlockMut<'_, NA, NB, P, T> {
        self.block_mut(slot - self.base as usize)
    }

    /// [`block_mut`](Self::block_mut), off the base or not. `MAPPED` is a
    /// constant of the sweep's instantiation, so each copy keeps one arm.
    #[inline(always)]
    pub fn block_mut_at<const MAPPED: bool>(&mut self, slot: usize) -> CrossBlockMut<'_, NA, NB, P, T> {
        if MAPPED { self.block_mut_based(slot) } else { self.block_mut(slot) }
    }

    /// The indices of block `k`.
    pub fn indices(&self, k: usize) -> (&[u32; NA], &[u32; NB]) {
        (&self.a[k], &self.b[k])
    }

    /// The values of block `k`, row-major `NA x NB`.
    pub fn values(&self, k: usize) -> &[T] {
        &self.values[k * P..(k + 1) * P]
    }

    fn block(&self, k: usize) -> &[T; P] {
        (&self.values[k * P..(k + 1) * P]).try_into().unwrap()
    }

    /// Append one representative scalar coordinate of each tile, over
    /// every block. Structure only, no values.
    pub fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        for k in 0..self.a.len() {
            cross_cells(&self.a[k], &self.b[k], out);
        }
    }

    /// Record where each tile scatters in the assembled value buffer, over
    /// every block. See
    /// [`Model::bind_hessian_positions`].
    pub fn bind_hessian_positions(&mut self, binder: &mut HessianBinder, out: &mut std::vec::Vec<ValueIndex>) {
        for k in 0..self.a.len() {
            cross_bind(&self.a[k], &self.b[k], binder, out);
        }
    }

    /// Add every tile into the dense symmetric Hessian, at `(A, B)` and
    /// its transpose.
    pub fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        for k in 0..self.a.len() {
            cross_dense(&self.a[k], &self.b[k], self.block(k), hessian);
        }
    }

    /// Add every tile into an upper-band Hessian of half-bandwidth `kd`
    /// (LAPACK layout). Errors when an entry falls outside the band.
    pub fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
        -> Result<(), crate::simple_lm::BandOverflow>
    {
        for k in 0..self.a.len() {
            cross_band(&self.a[k], &self.b[k], self.block(k), band, kd)?;
        }
        Ok(())
    }

    /// Push every tile into a COO Hessian as `(row, col, value)` triplets.
    pub fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        for k in 0..self.a.len() {
            cross_coo(&self.a[k], &self.b[k], self.block(k), coo);
        }
    }

    /// Add every tile straight into an already-patterned CSC Hessian,
    /// looking each entry up by column.
    pub fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        for k in 0..self.a.len() {
            cross_direct(&self.a[k], &self.b[k], self.block(k), csc);
        }
    }

    /// Add every tile into the assembled value buffer through the positions
    /// [`bind_hessian_positions`](Self::bind_hessian_positions) recorded,
    /// reading them in the same order.
    pub fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        for k in 0..self.a.len() {
            cross_indexed(&self.a[k], &self.b[k], self.block(k), vals, positions, cursor);
        }
    }
}


// ---------------------------------------------------------------------------

/// The heap-backed twin of [`CrossBlock`], kept as an alias of it. See
/// [`BoxedSelfBlock`] for what changed in 0.9.0. Use [`CrossBlock`].
#[deprecated(since = "0.9.0", note = "a block declares, it does not store; use CrossBlock")]
pub type BoxedCrossBlock<A, B, const NA: usize, const NB: usize, const P: usize, T = f64> =
    CrossBlock<A, B, NA, NB, P, T>;


/// Hessian entries in coordinate form: row, column and value, one entry at
/// a time.
///
/// [`SelfBlock`] and [`CrossBlock`] name their sides in the type, so the
/// solve hands them packed tiles of a size it knows in advance. Pairs that
/// cannot be named that way -- a constraint over more entities than a
/// pair, or residuals parsed at runtime -- go here instead, at the price
/// of a `Vec` push per entry. **Use a `CrossBlock` wherever the two sides
/// can be named.**
///
/// A `Coo` is not a model field: one belongs to the solve, which keeps one
/// per thread. A constraint reaches it with the `coo` keyword, and an
/// [`ExtendedModel`] hook is handed one, so nothing is declared.
///
/// It holds only the across-entity pairs (pairs whose two params belong to
/// different entity spans). The within-entity `H[A,A]` / `H[B,B]` / ...
/// diagonals and the gradient belong to each entity's `SelfBlock<Self>`,
/// so every `dr/dp_i * dr/dp_j` pair is written in exactly one place.
///
/// Two entry points:
/// - [`add_residual`](Coo::add_residual) for callers with a flat param
///   layout and no per-entity SelfBlocks: writes the gradient into the
///   provided global slice AND pushes the full upper-triangle Hessian
///   (including diagonal). One call, everything done.
/// - [`add_residual_cross`](Coo::add_residual_cross) for macro-emitted
///   N-ary constraints where each participating entity has its own
///   SelfBlock holding its grad+diagonal: stores ONLY across-entity pairs,
///   using the `entity_offsets` span list to skip within-entity pairs.
#[derive(Clone)]
pub struct Coo<T: crate::utils::Float = f64> {
    /// Hessian entries: upper-triangle (lo, hi, 2·dr_i·dr_j). Only cross-
    /// entity pairs are stored (within-entity pairs live in each entity's
    /// `SelfBlock`). Callers that manage their own flat param layout
    /// without per-entity blocks can pass each param as its own "entity"
    /// (entity_offsets = [0, 1, 2, ..., N]) to make every pair cross.
    pub hessian: std::vec::Vec<(u32, u32, T)>,
}

impl<T: crate::utils::Float> Default for Coo<T> {
    fn default() -> Self { Self::new() }
}

impl<T: crate::utils::Float> Coo<T> {
    /// An empty list.
    pub fn new() -> Self {
        Coo { hessian: std::vec::Vec::new() }
    }

    /// Reset to empty (called at start of each optimization step).
    pub fn zero(&mut self) {
        self.hessian.clear();
    }

    /// Entries pushed so far.
    pub fn len(&self) -> usize { self.hessian.len() }

    /// True while nothing has been pushed.
    pub fn is_empty(&self) -> bool { self.hessian.is_empty() }

    /// No-op: the entries are a heap `Vec` cleared each step, so there is
    /// nothing to release. Present for symmetry with the block markers.
    pub fn release(&mut self) {}

    /// One-shot entry for direct callers with a flat param layout and no
    /// per-entity SelfBlocks. Writes the gradient entries `2*r*dr[i]` into
    /// the provided global `grad` slice, and pushes the full upper-triangle
    /// Hessian `(i, j, 2*dr[i]*dr[j])` for every `i <= j` (including the
    /// diagonal) into `self.hessian`. `u32::MAX` entries in `indices` are
    /// skipped (fixed/non-optimizable params).
    pub fn add_residual(&mut self, r: T, indices: &[u32], dr: &[T], grad: &mut [T]) {
        let two = T::two();
        let n = indices.len();
        for i in 0..n {
            if indices[i] == u32::MAX { continue; }
            let gi = indices[i] as usize;
            grad[gi] += two * r * dr[i];
            for j in i..n {
                if indices[j] == u32::MAX { continue; }
                let (lo, hi) = if indices[i] <= indices[j] {
                    (indices[i], indices[j])
                } else {
                    (indices[j], indices[i])
                };
                self.hessian.push((lo, hi, two * dr[i] * dr[j]));
            }
        }
    }

    /// Robust-weighted variant of [`add_residual`](Self::add_residual): scales
    /// the gradient and Hessian pairs by the loss weight `w`. `w = 1` is
    /// bit-identical.
    pub fn add_residual_with_loss(&mut self, w: T, r: T, indices: &[u32], dr: &[T], grad: &mut [T]) {
        let two_w = T::two() * w;
        let wr = two_w * r;
        let n = indices.len();
        for i in 0..n {
            if indices[i] == u32::MAX { continue; }
            let gi = indices[i] as usize;
            grad[gi] += wr * dr[i];
            for j in i..n {
                if indices[j] == u32::MAX { continue; }
                let (lo, hi) = if indices[i] <= indices[j] {
                    (indices[i], indices[j])
                } else {
                    (indices[j], indices[i])
                };
                self.hessian.push((lo, hi, two_w * dr[i] * dr[j]));
            }
        }
    }

    /// Macro-emission entry for N-ary constraints where each participating
    /// entity has its own `SelfBlock<Self>` holding its grad + within-entity
    /// Hessian diagonal. Stores ONLY across-entity pairs -- within-entity
    /// pairs are skipped (they live in the entity SelfBlock).
    ///
    /// `entity_offsets` is the cumulative span boundary list
    /// (e.g. `[0, 6, 12, 18]` for three 6-param entities). Pairs `(i, j)`
    /// where `i` and `j` fall inside the same entity span are skipped.
    pub fn add_residual_cross(&mut self, _r: T, indices: &[u32], dr: &[T], entity_offsets: &[u32]) {
        let two = T::two();
        let n = indices.len();
        let span_of = |i: u32| -> u32 {
            let mut k = 0u32;
            for (idx, &off) in entity_offsets.iter().enumerate() {
                if off <= i { k = idx as u32; } else { break; }
            }
            k
        };
        for i in 0..n {
            if indices[i] == u32::MAX { continue; }
            let span_i = span_of(i as u32);
            for j in (i + 1)..n {
                if indices[j] == u32::MAX { continue; }
                let span_j = span_of(j as u32);
                if span_i == span_j { continue; }
                let (lo, hi) = if indices[i] <= indices[j] {
                    (indices[i], indices[j])
                } else {
                    (indices[j], indices[i])
                };
                let v = two * dr[i] * dr[j];
                // lo == hi here can only mean two slots of DIFFERENT spans
                // resolving to the same global parameter (aliased entities;
                // span_i == span_j above already excluded within-entity
                // diagonals). The symmetric pair collapses to one diagonal
                // cell, which needs both contributions.
                let v = if lo == hi { v + v } else { v };
                self.hessian.push((lo, hi, v));
            }
        }
    }

    /// Robust-weighted variant of
    /// [`add_residual_cross`](Self::add_residual_cross): scales the
    /// across-entity Hessian pairs by the loss weight `w`. `w = 1` is
    /// bit-identical. `_r` is ignored (present for uniform macro routing).
    pub fn add_residual_cross_with_loss(&mut self, w: T, _r: T, indices: &[u32], dr: &[T], entity_offsets: &[u32]) {
        let two_w = T::two() * w;
        let n = indices.len();
        let span_of = |i: u32| -> u32 {
            let mut k = 0u32;
            for (idx, &off) in entity_offsets.iter().enumerate() {
                if off <= i { k = idx as u32; } else { break; }
            }
            k
        };
        for i in 0..n {
            if indices[i] == u32::MAX { continue; }
            let span_i = span_of(i as u32);
            for j in (i + 1)..n {
                if indices[j] == u32::MAX { continue; }
                let span_j = span_of(j as u32);
                if span_i == span_j { continue; }
                let (lo, hi) = if indices[i] <= indices[j] {
                    (indices[i], indices[j])
                } else {
                    (indices[j], indices[i])
                };
                let v = two_w * dr[i] * dr[j];
                let v = if lo == hi { v + v } else { v };
                self.hessian.push((lo, hi, v));
            }
        }
    }

    /// Accumulate Hessian pairs into the full dense symmetric hessian.
    /// Generic over the target width; the conversion is an identity when it
    /// matches the block's storage.
    pub fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        let n_total = (hessian.len() as f64).sqrt() as usize;
        for &(i, j, v) in &self.hessian {
            let (i, j) = (i as usize, j as usize);
            let v = F::from(v).unwrap();
            hessian[i * n_total + j] += v;
            if i != j {
                hessian[j * n_total + i] += v;
            }
        }
    }

    /// Accumulate into upper-band format (LAPACK layout: A[r, c] at
    /// band[(kd + r - c) + c * ldab], matching SelfBlock/CrossBlock and
    /// the band solvers).
    pub fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
        -> Result<(), crate::simple_lm::BandOverflow>
    {
        let ldab = kd + 1;
        for &(row, col, v) in &self.hessian {
            let (r, c) = (row as usize, col as usize);
            if c < r || c - r > kd {
                return Err(crate::simple_lm::BandOverflow { row: r, col: c, kd });
            }
            band[(kd + r - c) + c * ldab] += F::from(v).unwrap();
        }
        Ok(())
    }

    /// Accumulate into COO sparse format. Upper triangle only.
    /// Emit one coordinate per stored entry (raw, as pushed by the
    /// extended model -- matching accumulate_hessian_sparse exactly).
    /// Requires the block populated (run a compute pass first).
    ///
    /// CONTRACT: a `Coo` must be refilled with the same entries
    /// in the same order every iteration of a solve. Count changes trip
    /// the indexed-fill assert; same-count cell or order changes
    /// produce a silently wrong Hessian. Rebuild the solver (reset the
    /// pattern) when the constraint structure changes.
    pub fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        for &(i, j, _) in &self.hessian {
            out.push((i, j));
        }
    }

    /// Push one scatter position per entry, in the emission order of
    /// [`accumulate_hessian_sparse`](Self::accumulate_hessian_sparse).
    ///
    /// The entries carry no static tile shape -- the pattern is only known
    /// after a compute -- so this block keeps the per-scalar map that
    /// [`SelfBlock`] and [`CrossBlock`] no longer need.
    pub fn bind_hessian_positions(
        &mut self,
        binder: &mut HessianBinder,
        out: &mut std::vec::Vec<ValueIndex>,
    ) {
        for &(i, j, _) in &self.hessian {
            out.push(value_index(match binder {
                HessianBinder::Tiled(bind) => bind(i, j).0,
                HessianBinder::Scalar(resolve) => resolve(i, j),
            }));
        }
    }

    /// Scatter this block into COO triplets.
    pub fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        for &(i, j, v) in &self.hessian {
            coo.push(i, j, F::from(v).unwrap());
        }
    }

    /// Accumulate directly into CSC vals via position lookup.
    pub fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        for &(row, col, v) in &self.hessian {
            if let Some(pos) = csc.find_pos(row as usize, col as usize) {
                csc.vals[pos] += F::from(v).unwrap();
            }
        }
    }

    /// Accumulate into CSC vals via precomputed position list.
    ///
    /// The position list is built once per solve from the first
    /// iteration's entry sequence; this block must push the same number
    /// of tuples every iteration (see `ExtendedModel` contract notes).
    pub fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        assert!(*cursor + self.hessian.len() <= positions.len(),
            "sparsity pattern changed between iterations: the COO list holds {} \
             entries but only {} slots remain in the cached pattern",
            self.hessian.len(), positions.len() - *cursor);
        for &(_, _, v) in &self.hessian {
            vals[positions[*cursor] as usize] += F::from(v).unwrap();
            *cursor += 1;
        }
    }
}

// ---------------------------------------------------------------------------
// Model impls for the block types
// ---------------------------------------------------------------------------
//
// A block is a node of the model tree like any other field: generated code
// walks it through the same uniform `Model::` recursion. The walk methods
// are generic over the target width and forward to the blocks' inherent
// methods, which convert stored values on accumulation -- an identity when
// the widths match, which is the only case generated roots emit.

// A self or cross block contributes nothing to the `Model` walks: it
// holds no values at all now, only its place in the solve's store, which
// the root walks itself. `release_blocks` stays a no-op so a generated
// `release` can call it uniformly.
impl<A, const N: usize, const M: usize, T: crate::utils::Float> Model for SelfBlock<A, N, M, T> {}
impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> Model
    for CrossBlock<A, B, NA, NB, P, T> {}

// ---------------------------------------------------------------------------
// Jacobian -- sparse Jacobian matrix for analysis (DOF, SVD, etc.)
// ---------------------------------------------------------------------------

/// Sparse Jacobian matrix.
///
/// Each row corresponds to one residual, with sparse partial derivatives
/// against the parameters involved. The primary consumer is SVD/rank
/// analysis for DOF detection.
///
/// Generated by `calc_jacobian()` when `#[arael(root, jacobian)]` is used.
pub struct Jacobian<T: crate::utils::Float = f64> {
    /// Number of parameters (columns).
    pub num_params: usize,
    /// Jacobian rows, one per residual. Ordered by constraint iteration order.
    pub rows: std::vec::Vec<JacobianRow<T>>,
}

/// One row of the Jacobian: a residual and its partial derivatives.
pub struct JacobianRow<T> {
    /// Constraint index -- matches the `#[arael(constraint_index)]` field
    /// on the source constraint struct. All residuals from the same
    /// constraint object share this value.
    pub constraint: u32,
    /// Human-readable label for this constraint attribute. Defaults to the
    /// constraint struct's type name; for structs with multiple constraint
    /// attributes, suffixed with `:N` where N is the attribute index. Can
    /// be overridden via `#[arael(constraint(hb, name = "custom", ...))]`.
    pub label: &'static str,
    /// Residual value.
    pub residual: T,
    /// Sparse partial derivatives: (global_param_index, dr/dp).
    /// Only active (optimizable) parameters included. Indices are unique:
    /// when a constraint touches the same parameter through several slots
    /// (aliased CrossBlock refs), the contributions are summed into one
    /// entry at construction (see `Jacobian::merge_duplicate_entries`).
    pub entries: std::vec::Vec<(u32, T)>,
}

impl<T: crate::utils::Float> Jacobian<T> {
    /// Number of residuals (rows).
    pub fn num_residuals(&self) -> usize { self.rows.len() }

    /// Sum entries that share a parameter index (a constraint reaching the
    /// same parameter through several slots, e.g. aliased CrossBlock refs:
    /// the total derivative is the sum of the per-slot partials). Called by
    /// the generated `calc_jacobian` so consumers can rely on unique
    /// indices per row.
    pub fn merge_duplicate_entries(&mut self) {
        for row in &mut self.rows {
            // Fast path: indices are already unique unless a constraint
            // reaches the same parameter through several slots (aliased
            // refs) -- a cheap scan keeps the common case free of sorting
            // and allocation. Rows are small (one entry per touched
            // parameter), so the quadratic scan is a handful of integer
            // compares.
            let n = row.entries.len();
            let has_dup = (1..n).any(|i| {
                let ji = row.entries[i].0;
                row.entries[..i].iter().any(|&(j, _)| j == ji)
            });
            if !has_dup { continue; }
            row.entries.sort_unstable_by_key(|&(j, _)| j);
            row.entries.dedup_by(|a, b| {
                if a.0 == b.0 {
                    b.1 = b.1 + a.1;
                    true
                } else {
                    false
                }
            });
        }
    }

    /// Residual vector.
    pub fn residuals(&self) -> std::vec::Vec<T> {
        self.rows.iter().map(|r| r.residual).collect()
    }

    /// Convert to dense row-major m x n matrix.
    pub fn to_dense(&self) -> std::vec::Vec<T> {
        let m = self.rows.len();
        let n = self.num_params;
        let mut data = vec![T::zero(); m * n];
        for (i, row) in self.rows.iter().enumerate() {
            for &(j, v) in &row.entries {
                data[i * n + j as usize] = v;
            }
        }
        data
    }

    /// Build an f64 dense row-major m x n matrix. Casts each entry from
    /// `T` via `num::NumCast`. Used by the SVD methods, which always
    /// operate in f64 regardless of `T` for rank-detection precision.
    fn to_dense_f64(&self) -> std::vec::Vec<f64> {
        let m = self.rows.len();
        let n = self.num_params;
        let mut data = vec![0.0f64; m * n];
        for (i, row) in self.rows.iter().enumerate() {
            for &(j, v) in &row.entries {
                data[i * n + j as usize] = <f64 as num::NumCast>::from(v).unwrap_or(0.0);
            }
        }
        data
    }

    /// Singular values of this Jacobian, sorted descending. Always
    /// computed in f64 regardless of `T`.
    ///
    /// Near-zero singular values count the degrees of freedom of the
    /// underlying constraint system. Backend choice mirrors the solver:
    /// nalgebra for small problems (n < 32), faer for larger.
    pub fn singular_values(&self) -> std::vec::Vec<f64> {
        let m = self.num_residuals();
        let n = self.num_params;
        if m == 0 || n == 0 { return std::vec::Vec::new(); }
        let dense = self.to_dense_f64();
        if n < 32 {
            let j = nalgebra::DMatrix::from_row_slice(m, n, &dense);
            j.singular_values().iter().cloned().collect()
        } else {
            let faer_j = faer::Mat::from_fn(m, n, |i, k| dense[i * n + k]);
            match faer_j.thin_svd() {
                Ok(svd) => {
                    let s = svd.S().column_vector();
                    (0..s.nrows()).map(|i| s[i]).collect()
                }
                Err(_) => std::vec::Vec::new(),
            }
        }
    }

    /// Full thin SVD: σ, U, V. Use this when you need the directions of
    /// rank deficiency; right singular vectors (columns of `V`) with
    /// σ ≈ 0 name the free-parameter directions in a DOF analysis.
    ///
    /// Thin dimensions: U is m×k, V is n×k, σ has k entries where
    /// k = min(m, n). Matrices stored row-major. Always in f64.
    pub fn svd(&self) -> SvdResult {
        let m = self.num_residuals();
        let n = self.num_params;
        let dense = self.to_dense_f64();
        svd_dense_f64(m, n, &dense)
    }

    /// L2 norm of each Jacobian column, in parameter-index order.
    /// Useful for column-preconditioning before SVD: scaling each
    /// column by `1 / col_norm` produces a matrix whose singular
    /// values reflect only row-space linear dependence, not the
    /// per-parameter scale differences that leak through from the
    /// residual formulation.
    pub fn column_l2_norms(&self) -> std::vec::Vec<f64> {
        let n = self.num_params;
        let mut sum_sq = vec![0.0f64; n];
        for row in &self.rows {
            for &(j, v) in &row.entries {
                let vf: f64 = <f64 as num::NumCast>::from(v).unwrap_or(0.0);
                sum_sq[j as usize] += vf * vf;
            }
        }
        sum_sq.into_iter().map(|s| s.sqrt()).collect()
    }

    /// Singular values of the column-normalised Jacobian (each column
    /// scaled by `1 / max(col_norm, 1e-15)`). Preserves the null-space
    /// (rank) of the Jacobian but flattens its spectrum: no scale-
    /// dependent conditioning leaks into rank detection.
    pub fn singular_values_column_normalised(&self) -> std::vec::Vec<f64> {
        let m = self.num_residuals();
        let n = self.num_params;
        if m == 0 || n == 0 { return std::vec::Vec::new(); }
        let col_norms = self.column_l2_norms();
        let mut dense = self.to_dense_f64();
        for r in 0..m {
            for c in 0..n {
                dense[r * n + c] /= col_norms[c].max(1e-15);
            }
        }
        if n < 32 {
            let j = nalgebra::DMatrix::from_row_slice(m, n, &dense);
            j.singular_values().iter().cloned().collect()
        } else {
            let faer_j = faer::Mat::from_fn(m, n, |i, k| dense[i * n + k]);
            match faer_j.thin_svd() {
                Ok(svd) => {
                    let s = svd.S().column_vector();
                    (0..s.nrows()).map(|i| s[i]).collect()
                }
                Err(_) => std::vec::Vec::new(),
            }
        }
    }

    /// Full SVD of the column-normalised Jacobian (see
    /// [`Self::singular_values_column_normalised`]). Also returns the
    /// column L2 norms used for normalisation so callers can back-
    /// transform right singular vectors from normalised parameter
    /// space to raw: `v_raw[i] = v[i] / col_norms[i]` (then renormalise
    /// to unit length if needed).
    pub fn svd_column_normalised(&self) -> (SvdResult, std::vec::Vec<f64>) {
        let m = self.num_residuals();
        let n = self.num_params;
        if m == 0 || n == 0 {
            return (SvdResult {
                singular_values: std::vec::Vec::new(),
                u: std::vec::Vec::new(),
                v: std::vec::Vec::new(),
                m, n,
            }, std::vec::Vec::new());
        }
        let col_norms = self.column_l2_norms();
        let mut dense = self.to_dense_f64();
        for r in 0..m {
            for c in 0..n {
                dense[r * n + c] /= col_norms[c].max(1e-15);
            }
        }
        (svd_dense_f64(m, n, &dense), col_norms)
    }
}

fn svd_dense_f64(m: usize, n: usize, dense: &[f64]) -> SvdResult {
    if m == 0 || n == 0 {
        return SvdResult {
            singular_values: std::vec::Vec::new(),
            u: std::vec::Vec::new(),
            v: std::vec::Vec::new(),
            m, n,
        };
    }
    let k = m.min(n);
    if n < 32 {
        let j = nalgebra::DMatrix::from_row_slice(m, n, dense);
        let svd = j.svd(true, true);
        let singular_values: std::vec::Vec<f64> = svd.singular_values.iter().cloned().collect();
        // INVARIANT: svd(true, true) above computes both U and V^t.
        let u_mat = svd.u.as_ref().expect("U requested");
        let vt_mat = svd.v_t.as_ref().expect("V^t requested");
        let mut u = vec![0.0f64; m * k];
        let mut v = vec![0.0f64; n * k];
        let kk = singular_values.len().min(k);
        for i in 0..m {
            for j in 0..kk {
                u[i * k + j] = u_mat[(i, j)];
            }
        }
        for i in 0..n {
            for j in 0..kk {
                v[i * k + j] = vt_mat[(j, i)];
            }
        }
        SvdResult { singular_values, u, v, m, n }
    } else {
        let faer_j = faer::Mat::from_fn(m, n, |i, k| dense[i * n + k]);
        match faer_j.thin_svd() {
            Ok(svd) => {
                let s = svd.S().column_vector();
                let singular_values: std::vec::Vec<f64> = (0..s.nrows()).map(|i| s[i]).collect();
                let u_mat = svd.U();
                let v_mat = svd.V();
                let mut u = vec![0.0f64; m * k];
                let mut v = vec![0.0f64; n * k];
                let kk = singular_values.len().min(k);
                for i in 0..m {
                    for j in 0..kk {
                        u[i * k + j] = u_mat[(i, j)];
                    }
                }
                for i in 0..n {
                    for j in 0..kk {
                        v[i * k + j] = v_mat[(i, j)];
                    }
                }
                SvdResult { singular_values, u, v, m, n }
            }
            Err(_) => SvdResult {
                singular_values: std::vec::Vec::new(),
                u: std::vec::Vec::new(),
                v: std::vec::Vec::new(),
                m, n,
            },
        }
    }
}

/// Result of an SVD decomposition of a [`Jacobian`]. Thin SVD: U is m×k,
/// V is n×k, σ has k = min(m, n) entries. Matrices are stored row-major.
/// Always in f64 regardless of the source Jacobian's element type.
pub struct SvdResult {
    /// Singular values, descending order.
    pub singular_values: std::vec::Vec<f64>,
    /// Left singular vectors, m×k row-major.
    pub u: std::vec::Vec<f64>,
    /// Right singular vectors, n×k row-major. Column `i` corresponds to
    /// `singular_values[i]`.
    pub v: std::vec::Vec<f64>,
    /// Number of residuals (rows of the original Jacobian).
    pub m: usize,
    /// Number of parameters (columns of the original Jacobian).
    pub n: usize,
}

// ---------------------------------------------------------------------------
// JacobianModel trait -- emitted by `#[arael(root, jacobian)]`
// ---------------------------------------------------------------------------

/// Instrumentation API emitted for root structs declared with
/// `#[arael(root, jacobian)]`.
///
/// Gives access to the sparse Jacobian matrix and a per-label cost table
/// for DOF analysis, constraint diagnostics, and sparsity inspection.
/// The methods mirror what the solver computes during `calc_cost` /
/// `calc_grad_hessian_*`, but retain constraint provenance
/// ([`JacobianRow::constraint`] and [`JacobianRow::label`]).
///
/// Intended for debugging and observability -- call sites on the hot
/// path should prefer the solver's `calc_cost` / `calc_grad_hessian_*`
/// methods, which are faster.
pub trait JacobianModel<T: crate::utils::Float> {
    /// Compute the sparse Jacobian at the given parameter vector. Each
    /// emitted row carries its source constraint ID
    /// ([`JacobianRow::constraint`]) and static label
    /// ([`JacobianRow::label`]). A robust `loss` scales rows and
    /// entries by `sqrt(rho'(s))`, so `J^T J` and `2 J^T r` match the
    /// assembled Gauss-Newton system.
    fn calc_jacobian(&mut self, params: &[T]) -> Jacobian<T>;

    /// Return the per-label cost total: which constraint group is
    /// contributing how much cost at a given parameter point.
    ///
    /// The macro-generated impl computes each block's ROBUSTIFIED cost
    /// (`rho(s)` under a `loss`), so the table sums to the solver's
    /// `calc_cost`. The row-derived default below does NOT apply
    /// losses; it serves hand-written impls without them.
    fn calc_cost_table(&mut self, params: &[T]) -> std::collections::HashMap<&'static str, T> {
        let j = self.calc_jacobian(params);
        let mut out = std::collections::HashMap::new();
        for row in &j.rows {
            let e = out.entry(row.label).or_insert(T::zero());
            *e += row.residual * row.residual;
        }
        out
    }
}

/// Build sparse Jacobian entries from index array and derivatives.
/// Filters out fixed parameters (index == u32::MAX).
pub fn jacobian_entries<T: crate::utils::Float>(indices: &[u32], derivatives: &[T]) -> std::vec::Vec<(u32, T)> {
    indices.iter().zip(derivatives.iter())
        .filter(|&(&idx, _)| idx != u32::MAX)
        .map(|(&idx, &d)| (idx, d))
        .collect()
}

// ---------------------------------------------------------------------------
// ModelSym -- symbolic companion type generation
// ---------------------------------------------------------------------------

/// Maps a concrete model type to its symbolic companion.
///
/// For each model struct `Foo`, the `#[arael::model]` macro generates a
/// `FooSym` struct whose fields are symbolic expressions (`arael_sym::E`),
/// and implements `ModelSym for Foo` with `type Sym = FooSym`. This is used
/// by the constraint code generator to build symbolic residual expressions
/// that can be differentiated at compile time.
pub trait ModelSym {
    type Sym;
    fn sym(base: &str) -> Self::Sym;
}

use arael_sym::E;

// Leaf fields carry no symbolic structure: their Sym is a single named symbol.
// A floating-point field reads as a symbolic constant inside a constraint body;
// integer, bool, char and string fields normally appear only in guards
// (evaluated at runtime), so every primitive works as a plain model field
// without `#[arael(skip)]`.
macro_rules! impl_scalar_model_sym {
    ($($t:ty),* $(,)?) => {
        $(
            impl ModelSym for $t {
                type Sym = E;
                fn sym(base: &str) -> E { arael_sym::symbol(base) }
            }
        )*
    };
}

impl_scalar_model_sym!(
    bool, char, String,
    i8, i16, i32, i64, i128, isize,
    u8, u16, u32, u64, u128, usize,
    f32, f64,
);

// One impl per math type family: the sym twin carries names and shapes
// only, so every precision shares it.
macro_rules! impl_math_model_sym {
    ($($m:ident :: $t:ident => $sym:ident),* $(,)?) => {
        $(
            impl<F: crate::utils::Float> ModelSym for crate::$m::$t<F> {
                type Sym = crate::$m::$sym;
                fn sym(base: &str) -> Self::Sym { crate::$m::$sym::new(base) }
            }
        )*
    };
}

impl_math_model_sym!(
    vect::vect3 => vect3sym,
    vect::vect2 => vect2sym,
    matrix::matrix3 => matrix3sym,
    matrix::matrix2 => matrix2sym,
    quatern::quatern => quaternsym,
);

// The N-dimensional types: the sym twin carries the dims as runtime
// values, taken from the const generics here.
impl<F: crate::utils::Float, const N: usize> ModelSym for crate::vect::vect<F, N> {
    type Sym = arael_sym::vectsym;
    fn sym(base: &str) -> Self::Sym { arael_sym::vectsym::new(base, N) }
}
impl<F: crate::utils::Float, const R: usize, const C: usize> ModelSym
    for crate::matrix::matrix<F, R, C>
{
    type Sym = arael_sym::matrixsym;
    fn sym(base: &str) -> Self::Sym { arael_sym::matrixsym::new(base, R, C) }
}

impl<T: ParamType + ModelSym> ModelSym for Param<T> {
    type Sym = T::Sym;
    fn sym(base: &str) -> Self::Sym { T::sym(base) }
}

impl<T: crate::utils::Float> ModelSym for SimpleEulerAngleParam<T>
    where vect3<T>: ModelSym
{
    type Sym = <vect3<T> as ModelSym>::Sym;
    fn sym(base: &str) -> Self::Sym { <vect3<T> as ModelSym>::sym(base) }
}

impl<T: crate::utils::Float> ModelSym for EulerAngleParam<T>
    where vect3<T>: ModelSym
{
    type Sym = <vect3<T> as ModelSym>::Sym;
    fn sym(base: &str) -> Self::Sym { <vect3<T> as ModelSym>::sym(base) }
}

impl<T: crate::utils::Float> ModelSym for QuaternionParam<T>
    where vect3<T>: ModelSym
{
    type Sym = <vect3<T> as ModelSym>::Sym;
    fn sym(base: &str) -> Self::Sym { <vect3<T> as ModelSym>::sym(base) }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vect::{vect3f, vect2f};

    #[test]
    fn test_param_f32_serialize_deserialize() {
        let mut a = Param::new(3.0f32);
        let mut b = Param::new(7.0f32);
        let mut c = Param::fixed(99.0f32);

        let mut data: std::vec::Vec<f32> = Vec::new();
        a.serialize_params(&mut data);
        b.serialize_params(&mut data);
        c.serialize_params(&mut data);

        assert_eq!(data, vec![3.0, 7.0]);
        assert_eq!(a.index, 0);
        assert_eq!(b.index, 1);
        assert_eq!(c.index, u32::MAX);

        // modify and deserialize
        data[0] = 10.0;
        data[1] = 20.0;
        a.deserialize_params(&data);
        b.deserialize_params(&data);
        c.deserialize_params(&data);
        assert_eq!(a.value, 10.0);
        assert_eq!(b.value, 20.0);
        assert_eq!(c.value, 99.0); // unchanged — fixed
    }

    #[test]
    fn test_param_vect3f_serialize() {
        let mut p = Param::new(vect3f::new(1.0, 2.0, 3.0));
        let mut data: std::vec::Vec<f32> = Vec::new();
        p.serialize_params(&mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert_eq!(p.index, 0);
    }

    #[test]
    fn test_param_update() {
        let mut p = Param::new(5.0f32);
        let mut data: std::vec::Vec<f32> = Vec::new();
        p.serialize_params(&mut data);
        data[0] = 42.0;

        p.update_params(&data);
        assert_eq!(p.work(), 42.0);
        assert_eq!(p.value, 5.0); // value unchanged

        p.update_self();
        assert_eq!(p.work(), 5.0); // work reset to value
    }

    #[test]
    fn test_param_fixed_update() {
        let mut p = Param::fixed(5.0f32);
        let mut data: std::vec::Vec<f32> = Vec::new();
        p.serialize_params(&mut data);
        assert!(data.is_empty()); // fixed param not serialized

        p.update_params(&data);
        assert_eq!(p.work(), 5.0); // gets value since not optimized
    }

    #[test]
    fn test_param_vect2f() {
        let mut p = Param::new(vect2f::new(1.0, 2.0));
        let mut data: std::vec::Vec<f32> = Vec::new();
        p.serialize_params(&mut data);
        assert_eq!(data, vec![1.0, 2.0]);

        data[0] = 10.0;
        data[1] = 20.0;
        p.update_params(&data);
        assert_eq!(p.work().x, 10.0);
        assert_eq!(p.work().y, 20.0);
    }

    #[test]
    fn test_param_f32_serialize64_roundtrip() {
        let mut a = Param::new(3.0f32);
        let mut b = Param::new(7.0f32);
        let mut c = Param::fixed(99.0f32);

        let mut data: std::vec::Vec<f64> = Vec::new();
        a.serialize_params(&mut data);
        b.serialize_params(&mut data);
        c.serialize_params(&mut data);

        assert_eq!(data, vec![3.0f64, 7.0]);
        assert_eq!(a.index, 0);
        assert_eq!(b.index, 1);
        assert_eq!(c.index, u32::MAX);

        // modify and deserialize through the f64 vector
        data[0] = 10.0;
        data[1] = 20.0;
        a.deserialize_params(&data);
        b.deserialize_params(&data);
        c.deserialize_params(&data);
        assert_eq!(a.value, 10.0f32);
        assert_eq!(b.value, 20.0f32);
        assert_eq!(c.value, 99.0f32);
    }

    #[test]
    fn test_param_vect3f_serialize64_roundtrip() {
        let mut p = Param::new(vect3f::new(1.0, 2.0, 3.0));
        let mut data: std::vec::Vec<f64> = Vec::new();
        p.serialize_params(&mut data);
        assert_eq!(data, vec![1.0f64, 2.0, 3.0]);

        data[0] = 10.5;
        data[1] = 20.5;
        data[2] = 30.5;
        p.update_params(&data);
        assert_eq!(p.work().x, 10.5f32);
        assert_eq!(p.work().y, 20.5f32);
        assert_eq!(p.work().z, 30.5f32);
    }

    #[test]
    fn test_param_fixed_update64() {
        let mut p = Param::fixed(5.0f32);
        let mut data: std::vec::Vec<f64> = Vec::new();
        p.serialize_params(&mut data);
        assert!(data.is_empty());

        p.update_params(&data);
        assert_eq!(p.work(), 5.0f32);
    }

    #[test]
    fn test_param_count() {
        assert_eq!(Param::<f32>::PARAM_COUNT, 1);
        assert_eq!(Param::<vect2f>::PARAM_COUNT, 2);
        assert_eq!(Param::<vect3f>::PARAM_COUNT, 3);
    }

    #[test]
    fn test_serialize_size() {
        let a = Param::new(1.0f32);
        let b = Param::fixed(2.0f32);
        let c = Param::new(vect3f::new(1.0, 2.0, 3.0));
        assert_eq!(a.serialize_size(), 1);
        assert_eq!(b.serialize_size(), 0);
        assert_eq!(c.serialize_size(), 3);
    }

    #[test]
    fn test_leaf_param_count_and_serialize_size() {
        // Leaf types have PARAM_COUNT 0 and serialize_size 0
        assert_eq!(f32::PARAM_COUNT, 0);
        assert_eq!(0.0f32.serialize_size(), 0);
        assert_eq!(vect3f::PARAM_COUNT, 0);
        assert_eq!(vect3f::new(1.0, 2.0, 3.0).serialize_size(), 0);
    }

    #[test]
    fn test_collection_serialize_size() {
        let mut v = vec![Param::new(1.0f32), Param::new(2.0f32), Param::fixed(3.0f32)];
        let mut data: std::vec::Vec<f32> = Vec::new();
        v.serialize_params(&mut data);
        // 2 optimized params
        assert_eq!(v.serialize_size(), 2);

        let none: Option<Param<f32>> = None;
        assert_eq!(none.serialize_size(), 0);
        let some = Some(Param::new(1.0f32));
        assert_eq!(some.serialize_size(), 1);
    }

    // -----------------------------------------------------------------
    // Block accumulation format equivalence: every block type must land
    // identical values through its dense and band accumulate paths. The
    // band format is the LAPACK upper-band convention used by the band
    // solvers: A[i, j] lives at band[(kd + i - j) + j * ldab] for
    // max(0, j - kd) <= i <= j, ldab = kd + 1.
    // -----------------------------------------------------------------

    /// Expand an upper-band matrix into a full dense symmetric n*n matrix.
    fn densify_band(band: &[f64], n: usize, kd: usize) -> Vec<f64> {
        let ldab = kd + 1;
        let mut full = vec![0.0; n * n];
        for j in 0..n {
            for i in j.saturating_sub(kd)..=j {
                let v = band[(kd + i - j) + j * ldab];
                full[i * n + j] = v;
                full[j * n + i] = v;
            }
        }
        full
    }

    // A slab holding only some entities reaches them through the map, and
    // must accumulate exactly what the whole slab does for those it holds.
    #[test]
    fn a_mapped_slab_matches_the_whole_one() {
        // Whole: four entities, the marker's slot addressing the data.
        let mut whole: SelfBlockArray<3, 6, f64> = SelfBlockArray::new();
        for e in 0..4u32 { whole.push(e, &[3 * e, 3 * e + 1, 3 * e + 2]); }
        whole.finish();

        // Partial: entities 1 and 3 only, in the order first met.
        let mut part: SelfBlockArray<3, 6, f64> = SelfBlockArray::new();
        part.map_resize(4);
        part.push_at(3, 3, &[9, 10, 11]);
        part.push_at(1, 1, &[3, 4, 5]);
        part.finish();
        assert_eq!(part.slab_of(3), 0);
        assert_eq!(part.slab_of(1), 1);

        let rows = [(1usize, 0.5, [1.0, -0.25, 0.75]), (3, -0.4, [0.2, 1.5, -0.6]),
                    (1, 0.9, [-1.0, 0.3, 0.1])];
        for (slot, r, dr) in rows {
            whole.add_residual(slot, r, &dr);
            part.add_residual_mapped(slot, r, &dr);
        }
        for (slot, r, dr) in rows {
            whole.add_residual_with_loss(slot, 0.25, r, &dr);
            part.add_residual_with_loss_mapped(slot, 0.25, r, &dr);
        }

        let n = 12;
        let (mut a, mut b) = (vec![0.0; n * n], vec![0.0; n * n]);
        whole.accumulate_hessian(&mut a);
        part.accumulate_hessian(&mut b);
        assert_eq!(a, b, "the mapped slab's Hessian must be the whole one's");
        let (mut ga, mut gb) = (vec![0.0; n], vec![0.0; n]);
        whole.scatter_grad(&mut ga);
        part.scatter_grad(&mut gb);
        assert_eq!(ga, gb, "and its gradient too");
    }

    #[test]
    fn map_resize_keeps_what_was_claimed() {
        let mut arr: SelfBlockArray<3, 6, f64> = SelfBlockArray::new();
        arr.map_resize(4);
        arr.push_at(2, 2, &[6, 7, 8]);
        // Sizing again must not forget a claim: the build sizes once per
        // solve and the entries outlive it.
        arr.map_resize(4);
        arr.map_resize(2);
        assert_eq!(arr.slab_of(2), 0);
    }

    // A cross array covering one run of the walk is addressed by the same
    // global slot the marker carries.
    #[test]
    fn a_based_cross_array_matches_the_whole_one() {
        let mut whole: CrossBlockArray<2, 2, 4, f64> = CrossBlockArray::new();
        for k in 0..4u32 { whole.push(&[2 * k, 2 * k + 1], &[8 + 2 * k, 9 + 2 * k]); }
        whole.finish();
        assert_eq!(whole.base(), 0);

        // The run [2, 4), which is what a thread owning that range holds.
        let mut run: CrossBlockArray<2, 2, 4, f64> = CrossBlockArray::new();
        for k in 2..4u32 { run.push(&[2 * k, 2 * k + 1], &[8 + 2 * k, 9 + 2 * k]); }
        run.finish();
        run.set_base(2);

        for slot in 2..4usize {
            let dr_a = [1.0 + slot as f64, -0.5];
            let dr_b = [0.25, 2.0 - slot as f64];
            whole.block_mut(slot).add_residual_cross(0.0, &dr_a, &dr_b);
            run.block_mut_based(slot).add_residual_cross(0.0, &dr_a, &dr_b);
        }
        assert_eq!(whole.values(2), run.values(0), "slot 2 is the run's first block");
        assert_eq!(whole.values(3), run.values(1));
    }

    // The property a cut has to satisfy: sweeping disjoint ranges into
    // separate stores and summing them gives what one store sweeping the
    // whole walk gives. This drives the arrays the way the generated sweep
    // does -- the entity's marker slot for a self write, the instance's for
    // a cross one -- so a base or a map that is wrong shows up here rather
    // than as a wrong Hessian in a threaded solve.
    //
    // Instance i couples two of six entities; entities 0, 2, 3 and 4 are
    // reached from both halves, which is the case where two slabs land on
    // the same Hessian tile.
    const PAIRS: [(usize, usize); 8] =
        [(0, 1), (1, 2), (2, 3), (3, 4), (3, 4), (4, 5), (5, 0), (0, 2)];

    fn residual(i: usize) -> (f64, [f64; 2], [f64; 2]) {
        (0.1 * (i as f64 + 1.0), [1.0 + i as f64, -0.5], [0.25, 2.0 - i as f64])
    }

    /// Sweep instances `[lo, hi)` into one store. `MAPPED` says whether the
    /// store holds a slice of the walk (a thread's) or all of it.
    fn sweep<const MAPPED: bool>(
        lo: usize, hi: usize,
        hess: &mut [f64], grad: &mut [f64],
    ) {
        let mut selfs: SelfBlockArray<2, 3, f64> = SelfBlockArray::new();
        let mut cross: CrossBlockArray<2, 2, 4, f64> = CrossBlockArray::new();
        if MAPPED {
            // The build claims a slab place the first time this range meets
            // an entity, and the cross array opens on the range's first slot.
            selfs.map_resize(6);
            let mut seen = [false; 6];
            for i in lo..hi {
                for e in [PAIRS[i].0, PAIRS[i].1] {
                    if !seen[e] {
                        seen[e] = true;
                        selfs.push_at(e as u32, e as u32, &[2 * e as u32, 2 * e as u32 + 1]);
                    }
                }
            }
            for i in lo..hi {
                let (a, b) = PAIRS[i];
                cross.push(&[2 * a as u32, 2 * a as u32 + 1], &[2 * b as u32, 2 * b as u32 + 1]);
            }
            cross.set_base(lo as u32);
        } else {
            for e in 0..6usize {
                selfs.push(e as u32, &[2 * e as u32, 2 * e as u32 + 1]);
            }
            for &(a, b) in PAIRS.iter() {
                cross.push(&[2 * a as u32, 2 * a as u32 + 1], &[2 * b as u32, 2 * b as u32 + 1]);
            }
        }
        selfs.finish();
        cross.finish();

        for i in lo..hi {
            let (a, b) = PAIRS[i];
            let (r, dr_a, dr_b) = residual(i);
            // The slot is the marker's either way; only the addressing differs.
            selfs.add_residual_at::<MAPPED>(a, r, &dr_a);
            selfs.add_residual_at::<MAPPED>(b, r, &dr_b);
            cross.block_mut_at::<MAPPED>(i).add_residual_cross(r, &dr_a, &dr_b);
        }
        selfs.accumulate_hessian(hess);
        cross.accumulate_hessian(hess);
        selfs.scatter_grad(grad);
    }

    #[test]
    fn disjoint_ranges_sum_to_the_whole_walk() {
        let n = 12;
        let (mut hw, mut gw) = (vec![0.0; n * n], vec![0.0; n]);
        sweep::<false>(0, 8, &mut hw, &mut gw);

        // Every way of cutting the walk in two must rebuild it.
        for cut in 1..8usize {
            let (mut h, mut g) = (vec![0.0; n * n], vec![0.0; n]);
            sweep::<true>(0, cut, &mut h, &mut g);
            sweep::<true>(cut, 8, &mut h, &mut g);
            for k in 0..n * n {
                assert!((h[k] - hw[k]).abs() < 1e-12,
                    "cut at {}: hessian[{}] {} vs whole {}", cut, k, h[k], hw[k]);
            }
            for k in 0..n {
                assert!((g[k] - gw[k]).abs() < 1e-12,
                    "cut at {}: grad[{}] {} vs whole {}", cut, k, g[k], gw[k]);
            }
        }
    }

    #[test]
    fn three_ranges_sum_to_the_whole_walk() {
        let n = 12;
        let (mut hw, mut gw) = (vec![0.0; n * n], vec![0.0; n]);
        sweep::<false>(0, 8, &mut hw, &mut gw);
        let (mut h, mut g) = (vec![0.0; n * n], vec![0.0; n]);
        for (lo, hi) in [(0, 3), (3, 3), (3, 6), (6, 8)] {   // one empty range
            sweep::<true>(lo, hi, &mut h, &mut g);
        }
        for k in 0..n * n {
            assert!((h[k] - hw[k]).abs() < 1e-12, "hessian[{}] {} vs {}", k, h[k], hw[k]);
        }
        for k in 0..n {
            assert!((g[k] - gw[k]).abs() < 1e-12, "grad[{}] {} vs {}", k, g[k], gw[k]);
        }
    }

    #[test]
    fn self_blocks_band_matches_dense() {
        let n = 4;
        let kd = 2;
        let mut arr: SelfBlockArray<3, 6, f64> = SelfBlockArray::new();
        arr.push(0, &[0, 1, 2]);
        arr.finish();
        arr.add_residual(0, 0.3, &[1.0, 0.5, -0.25]);
        arr.add_residual(0, -0.7, &[0.2, -1.5, 0.75]);

        let mut dense = vec![0.0; n * n];
        arr.accumulate_hessian(&mut dense);
        let mut band = vec![0.0; (kd + 1) * n];
        arr.accumulate_hessian_band(&mut band, kd).unwrap();
        assert_eq!(densify_band(&band, n, kd), dense);
    }

    #[test]
    fn cross_blocks_band_matches_dense() {
        let n = 5;
        let kd = 3;
        let mut arr: CrossBlockArray<2, 2, 4, f64> = CrossBlockArray::new();
        arr.push(&[0, 1], &[2, 3]);
        arr.finish();
        arr.block_mut(0).add_residual_cross(0.4, &[1.0, -0.5], &[0.25, 2.0]);
        arr.block_mut(0).add_residual_cross(-1.1, &[0.3, 0.7], &[-0.6, 0.1]);

        let mut dense = vec![0.0; n * n];
        arr.accumulate_hessian(&mut dense);
        let mut band = vec![0.0; (kd + 1) * n];
        arr.accumulate_hessian_band(&mut band, kd).unwrap();
        assert_eq!(densify_band(&band, n, kd), dense);
    }

    #[test]
    fn tripletblock_band_matches_dense() {
        let n = 4;
        let kd = 2;
        let mut blk: Coo<f64> = Coo::new();
        let mut grad = vec![0.0; n];
        blk.add_residual(0.3, &[0, 1, 2], &[1.0, 0.5, -0.25], &mut grad);
        blk.add_residual(-0.7, &[1, 3], &[2.0, -1.5], &mut grad);

        let mut dense = vec![0.0; n * n];
        blk.accumulate_hessian(&mut dense);
        let mut band = vec![0.0; (kd + 1) * n];
        blk.accumulate_hessian_band(&mut band, kd).unwrap();
        assert_eq!(densify_band(&band, n, kd), dense,
            "COO band accumulation must use the same upper-band \
             layout as SelfBlock/CrossBlock and the band solvers");
    }

    #[test]
    fn coo_band_rejects_an_entry_outside_the_band() {
        let mut blk: Coo<f64> = Coo::new();
        blk.hessian.push((0, 1, 1.0));
        blk.hessian.push((0, 3, 1.0));
        let n = 4;
        let kd = 1;
        let mut band = vec![0.0; (kd + 1) * n];
        let err = blk.accumulate_hessian_band(&mut band, kd).unwrap_err();
        assert_eq!((err.row, err.col, err.kd), (0, 3, 1));
    }

    #[test]
    fn coo_loss_weight_scales_gradient_and_pairs() {
        let n = 4;
        let (idx, dr) = ([0u32, 2, 3], [1.0_f64, -0.5, 0.25]);
        let (r, w) = (0.7_f64, 0.3_f64);
        let mut plain: Coo<f64> = Coo::new();
        let mut g_plain = vec![0.0; n];
        plain.add_residual(r, &idx, &dr, &mut g_plain);
        let mut lossy: Coo<f64> = Coo::new();
        let mut g_lossy = vec![0.0; n];
        lossy.add_residual_with_loss(w, r, &idx, &dr, &mut g_lossy);
        assert_eq!(plain.len(), lossy.len());
        for (a, b) in std::iter::zip(&plain.hessian, &lossy.hessian) {
            assert_eq!((a.0, a.1), (b.0, b.1), "same cells in the same order");
            assert!((w * a.2 - b.2).abs() < 1e-15, "{} vs {}", w * a.2, b.2);
        }
        for (a, b) in std::iter::zip(&g_plain, &g_lossy) {
            assert!((w * a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn tripletblock_aliased_spans_double_the_diagonal() {
        // Two spans resolving to the same global parameters (aliased
        // entities). The cross pair for a shared parameter collapses to
        // one diagonal tuple, which must carry BOTH 2*dr_i*dr_j
        // contributions: the full Hessian for the shared params is
        // 2 * (dr_a + dr_b) outer (dr_a + dr_b); the self blocks own the
        // dr_a*dr_a / dr_b*dr_b parts, this block the rest.
        let n = 2;
        let (da, db) = ([1.0_f64, 0.5], [-0.25_f64, 2.0]);
        let mut blk: Coo<f64> = Coo::new();
        // Slots: [a0, a1, b0, b1] with both spans on params [0, 1].
        blk.add_residual_cross(
            0.3,
            &[0, 1, 0, 1],
            &[da[0], da[1], db[0], db[1]],
            &[0, 2],
        );

        let mut dense = vec![0.0; n * n];
        blk.accumulate_hessian(&mut dense);

        // Expected cross-only contribution, uniform for every cell
        // including the diagonal: 2 * (da_i*db_j + db_i*da_j).
        for i in 0..n {
            for j in 0..n {
                let expected = 2.0 * (da[i] * db[j] + db[i] * da[j]);
                assert!((dense[i * n + j] - expected).abs() < 1e-14,
                    "H[{},{}] = {} expected {}", i, j, dense[i * n + j], expected);
            }
        }

        // All formats must agree on the aliased tuples.
        let kd = n - 1;
        let mut band = vec![0.0; (kd + 1) * n];
        blk.accumulate_hessian_band(&mut band, kd).unwrap();
        assert_eq!(densify_band(&band, n, kd), dense,
            "aliased triplet band differs from dense");
    }

    #[test]
    fn tripletblock_band_matches_self_block_array_band() {
        // The same residual pushed through both block types must produce
        // byte-identical gradient and band arrays.
        let n = 4;
        let kd = 3;
        let r = 0.9;
        let dr = [1.0, -2.0, 0.5];
        let idx = [0u32, 1, 3];

        let mut arr: SelfBlockArray<3, 6, f64> = SelfBlockArray::new();
        arr.push(0, &idx);
        arr.finish();
        arr.add_residual(0, r, &dr);
        let mut g_self = vec![0.0; n];
        arr.scatter_grad(&mut g_self);

        let mut g_triplet = vec![0.0; n];
        let mut tb: Coo<f64> = Coo::new();
        tb.add_residual(r, &idx, &dr, &mut g_triplet);

        assert_eq!(g_self, g_triplet);

        let mut band_self = vec![0.0; (kd + 1) * n];
        arr.accumulate_hessian_band(&mut band_self, kd).unwrap();
        let mut band_triplet = vec![0.0; (kd + 1) * n];
        tb.accumulate_hessian_band(&mut band_triplet, kd).unwrap();
        assert_eq!(band_self, band_triplet);
    }
}
