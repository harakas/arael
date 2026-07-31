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

    /// Return the current working-copy value (set during `update`).
    pub fn work(&self) -> T { self.work }
    /// Return a reference to the current working-copy value.
    pub fn work_ref(&self) -> &T { &self.work }
    /// Return a mutable reference to the current working-copy value.
    pub fn work_mut(&mut self) -> &mut T { &mut self.work }

    /// Return this parameter's index into the flat parameter vector, or `u32::MAX` if fixed.
    pub fn index(&self) -> u32 { self.index }

    /// Write this parameter's per-component indices into `out`, or `u32::MAX` if fixed.
    pub fn write_indices(&self, out: &mut [u32]) {
        if self.index == u32::MAX {
            for o in out.iter_mut() { *o = u32::MAX; }
        } else {
            for (k, o) in out.iter_mut().enumerate() {
                *o = self.index + k as u32;
            }
        }
    }
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
/// - `zero_blocks` / `accumulate_hessian*` -- zero and accumulate Hessian
///   blocks into dense, banded, COO, CSC, or indexed sparse formats.
///   Gradient is written directly into a global slice by each constraint,
///   not routed through these methods.
///
/// The parameter-vector and Hessian methods are generic over the solve
/// precision `F`; blocks store at their own precision and convert on
/// accumulation (an identity after monomorphization when the widths match).
pub trait Model {
    fn serialize_params<F: crate::utils::Float>(&mut self, _data: &mut std::vec::Vec<F>) {}
    fn deserialize_params<F: crate::utils::Float>(&mut self, _data: &[F]) {}
    fn update_params<F: crate::utils::Float>(&mut self, _data: &[F]) {}
    fn update_self(&mut self) {}

    const PARAM_COUNT: u32 = 0;
    fn serialize_size(&self) -> u32 { 0 }
    fn param_symbols(_base: &str, _out: &mut std::vec::Vec<String>) {}

    fn zero_blocks(&mut self) {}

    /// Append this model's parameter blocks as `(offset, width)` spans of
    /// the flat parameter vector -- one span per entity, read from each
    /// entity's `SelfBlock` indices. Valid only after `serialize` has
    /// assigned indices. Entities whose params are all fixed contribute
    /// nothing.
    fn collect_param_blocks(&self, _out: &mut std::vec::Vec<(u32, u32)>) {}

    /// Append one representative scalar coordinate per Hessian block cell
    /// this model's blocks touch (TripletBlocks: one per stored
    /// entry). Same traversal order as `accumulate_hessian_sparse`;
    /// valid after `serialize`. Structure-only: no numeric work.
    fn collect_hessian_cells(&self, _out: &mut std::vec::Vec<(u32, u32)>) {}
    /// Bind every block to its tile in the assembled value buffer, ready
    /// for `accumulate_hessian_sparse_indexed`. `bind` maps a scalar
    /// coordinate to its position and the column stride of its tile; blocks
    /// with a static tile shape keep the pair and derive the rest. Blocks
    /// with no static shape ([`TripletBlock`]) instead push one position per
    /// entry into `out`, in the emission order of
    /// `accumulate_hessian_sparse`. Builds the indexed map without a COO
    /// pass. Valid after `serialize`, and must be redone whenever parameter
    /// indices or the Hessian pattern change.
    fn bind_hessian_positions(
        &mut self,
        _binder: &mut HessianBinder,
        _out: &mut std::vec::Vec<ValueIndex>,
    ) {}

    /// Free every heap-backed (`Boxed*`) block's Hessian storage in this model
    /// and its sub-models, reclaiming the transient assembly memory between
    /// solves. Inline blocks are unaffected; the next solve re-allocates the
    /// boxed ones on demand. Default: no-op.
    fn release_blocks(&mut self) {}

    // Fold accepted-step euler angle deltas into their reference rotations
    // and zero the delta entries in the parameter vector. A no-op for
    // everything except EulerAngleParam, which re-centers after every
    // accepted LM step (the property that avoids gimbal lock). Recurses
    // through the model tree exactly like update/serialize, so params at
    // any nesting depth are advanced.
    fn advance_params<F: crate::utils::Float>(&mut self, _params: &mut [F]) {}

    // Hessian-only accumulation: blocks hold only Hessian entries after the
    // refactor; the gradient is written directly by constraint evaluation
    // into the LM-provided `grad` slice, so there is nothing for these
    // methods to do with grad.
    fn accumulate_hessian<F: crate::utils::Float>(&self, _hessian: &mut [F]) {}
    fn accumulate_hessian_band<F: crate::utils::Float>(&self, _band: &mut [F], _kd: usize) -> Result<(), crate::simple_lm::BandOverflow> { Ok(()) }
    fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, _coo: &mut crate::simple_lm::CooMatrix<F>) {}
    fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, _csc: &mut crate::simple_lm::CscMatrix<F>) {}
    fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, _vals: &mut [F], _positions: &[ValueIndex], _cursor: &mut usize) {}
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
/// `#[arael(constraint(...))]` at compile time — for example, constraints
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
/// To write custom gradient and Hessian contributions, add a
/// [`TripletBlock`] field to the root struct. The macro automatically
/// zeroes and accumulates it. In `extended_compute`, push residual
/// contributions into it via [`TripletBlock::add_residual`].
///
/// # Execution order
///
/// Each solver iteration runs:
/// 1. `Model::update` — copies params into working values
/// 2. **`extended_update`** — set up derived state before calculations
/// 3. `zero_blocks` — zeros all Hessian blocks (including TripletBlocks)
/// 4. Macro-generated constraint loops — fill SelfBlock/CrossBlock
/// 5. **`extended_compute`** — fill TripletBlocks with custom residuals
///    (writes grad entries directly into the LM-provided global slice)
/// 6. `accumulate_hessian*` — reads all Hessian blocks into global Hessian
///
/// For cost evaluation: `Model::update` → `extended_update` →
/// macro-generated cost loop → **`extended_cost`**.
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
///     hb: TripletBlock<f64>,                  // Gauss-Newton accumulator
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
/// impl ExtendedModel for RegressionModel {
///     fn extended_compute64(&mut self, params: &[f64], grad: &mut [f64]) {
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
///             // full upper-triangle Hessian into the TripletBlock
///             self.hb.add_residual(r, &indices, &dr, grad);
///         }
///     }
///
///     fn extended_cost64(&self, params: &[f64]) -> f64 {
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
pub trait ExtendedModel {
    /// Called after `deserialize64` writes optimized values back to `Param::value`.
    /// Use to sync derived persistent state (e.g. copy one param's value to another).
    fn extended_deserialize64(&mut self) {}
    /// Called after `deserialize32` writes optimized values back to `Param::value`.
    fn extended_deserialize32(&mut self) {}
    /// Called after `update_params` (f64), before cost/constraint calculations.
    /// Use to compute derived state that constraints depend on.
    fn extended_update64(&mut self, _params: &[f64]) {}
    /// Called after `update_params` (f32), before cost/constraint calculations.
    fn extended_update32(&mut self, _params: &[f32]) {}
    /// Additional cost contribution (f64). Called after the
    /// macro-generated cost loop.
    fn extended_cost64(&self, _params: &[f64]) -> f64 { 0.0 }
    /// Additional cost contribution (f32).
    fn extended_cost32(&self, _params: &[f32]) -> f32 { 0.0 }
    /// Compute custom constraint residuals (f64). Called after
    /// macro-generated constraints. Writes gradient contributions directly
    /// into `grad` and cross-entity Hessian pairs into a
    /// [`TripletBlock`] field.
    ///
    /// **Iteration-invariance contract:** the sparse solvers cache the
    /// Hessian sparsity pattern from the first iteration of a solve and
    /// replay it positionally on every later iteration. The number and
    /// order of Hessian entries this hook produces (TripletBlock tuples)
    /// must therefore stay constant within one `lm_solve` call --
    /// residual *values* may change freely, entry *structure* may not.
    /// Violations are detected and reported ("sparsity pattern changed
    /// between iterations"). Restructure between solves instead;
    /// `LmSolver::reset()` rebuilds the cached pattern.
    fn extended_compute64(&mut self, _params: &[f64], _grad: &mut [f64]) {}
    /// Compute custom constraint residuals (f32). See the
    /// iteration-invariance contract on [`ExtendedModel::extended_compute64`].
    fn extended_compute32(&mut self, _params: &[f32], _grad: &mut [f32]) {}
    /// Append Jacobian rows for runtime constraints (f64).
    /// `cid` is the constraint counter -- increment per constraint object.
    fn extended_jacobian64(&mut self, _params: &[f64], _rows: &mut std::vec::Vec<JacobianRow<f64>>, _cid: &mut u32) {}
    /// Append Jacobian rows for runtime constraints (f32).
    fn extended_jacobian32(&mut self, _params: &[f32], _rows: &mut std::vec::Vec<JacobianRow<f32>>, _cid: &mut u32) {}
}


// ---------------------------------------------------------------------------
// Model impl for Param<T>
// ---------------------------------------------------------------------------

impl<T: ParamType> Model for Param<T> {
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
        for suffix in T::SUFFIXES {
            out.push(format!("{}{}", base, suffix));
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
    pub fn write_indices(&self, out: &mut [u32]) {
        if self.index == u32::MAX {
            for o in out.iter_mut() { *o = u32::MAX; }
        } else {
            for (k, o) in out.iter_mut().enumerate() { *o = self.index + k as u32; }
        }
    }
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
    pub fn write_indices(&self, out: &mut [u32]) {
        if self.index == u32::MAX {
            for o in out.iter_mut() { *o = u32::MAX; }
        } else {
            for (k, o) in out.iter_mut().enumerate() { *o = self.index + k as u32; }
        }
    }
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
    pub fn write_indices(&self, out: &mut [u32]) {
        if self.index == u32::MAX {
            for o in out.iter_mut() { *o = u32::MAX; }
        } else {
            for (k, o) in out.iter_mut().enumerate() { *o = self.index + k as u32; }
        }
    }
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
            fn zero_blocks(&mut self) {
                for item in self.$iter_mut() { item.zero_blocks(); }
            }
            fn collect_param_blocks(&self, out: &mut std::vec::Vec<(u32, u32)>) {
                for item in self.iter() { item.collect_param_blocks(out); }
            }
            fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
                for item in self.iter() { item.collect_hessian_cells(out); }
            }
            fn bind_hessian_positions(&mut self, binder: &mut HessianBinder, out: &mut std::vec::Vec<ValueIndex>) {
                for item in self.$iter_mut() { item.bind_hessian_positions(binder, out); }
            }
            fn release_blocks(&mut self) {
                for item in self.$iter_mut() { item.release_blocks(); }
            }
            fn serialize_size(&self) -> u32 {
                self.iter().map(|item| item.serialize_size()).sum()
            }
            fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
                for item in self.iter() { item.accumulate_hessian(hessian); }
            }
            fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize) -> Result<(), crate::simple_lm::BandOverflow> {
                for item in self.iter() { item.accumulate_hessian_band(band, kd)?; }
                Ok(())
            }
            fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
                for item in self.iter() { item.accumulate_hessian_sparse(coo); }
            }
            fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
                for item in self.iter() { item.accumulate_hessian_sparse_direct(csc); }
            }
            fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
                for item in self.iter() { item.accumulate_hessian_sparse_indexed(vals, positions, cursor); }
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
    fn zero_blocks(&mut self) {
        for item in self.iter_mut() { item.zero_blocks(); }
    }
    fn collect_param_blocks(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        for item in self.iter() { item.collect_param_blocks(out); }
    }
    fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        for item in self.iter() { item.collect_hessian_cells(out); }
    }
    fn bind_hessian_positions(&mut self, binder: &mut HessianBinder, out: &mut std::vec::Vec<ValueIndex>) {
        for item in self.iter_mut() { item.bind_hessian_positions(binder, out); }
    }
    fn release_blocks(&mut self) {
        for item in self.iter_mut() { item.release_blocks(); }
    }
    fn serialize_size(&self) -> u32 {
        self.iter().map(|item| item.serialize_size()).sum()
    }
    fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        for item in self.iter() { item.accumulate_hessian(hessian); }
    }
    fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize) -> Result<(), crate::simple_lm::BandOverflow> {
        for item in self.iter() { item.accumulate_hessian_band(band, kd)?; }
        Ok(())
    }
    fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        for item in self.iter() { item.accumulate_hessian_sparse(coo); }
    }
    fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        for item in self.iter() { item.accumulate_hessian_sparse_direct(csc); }
    }
    fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        for item in self.iter() { item.accumulate_hessian_sparse_indexed(vals, positions, cursor); }
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
    fn zero_blocks(&mut self) {
        if let Some(inner) = self { inner.zero_blocks(); }
    }
    fn collect_param_blocks(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        if let Some(inner) = self { inner.collect_param_blocks(out); }
    }
    fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        if let Some(inner) = self { inner.collect_hessian_cells(out); }
    }
    fn bind_hessian_positions(&mut self, binder: &mut HessianBinder, out: &mut std::vec::Vec<ValueIndex>) {
        if let Some(inner) = self { inner.bind_hessian_positions(binder, out); }
    }
    fn release_blocks(&mut self) {
        if let Some(inner) = self { inner.release_blocks(); }
    }
    fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        if let Some(inner) = self { inner.accumulate_hessian(hessian); }
    }
    fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize) -> Result<(), crate::simple_lm::BandOverflow> {
        if let Some(inner) = self { inner.accumulate_hessian_band(band, kd)?; }
        Ok(())
    }
    fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        if let Some(inner) = self { inner.accumulate_hessian_sparse(coo); }
    }
    fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        if let Some(inner) = self { inner.accumulate_hessian_sparse_direct(csc); }
    }
    fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        if let Some(inner) = self { inner.accumulate_hessian_sparse_indexed(vals, positions, cursor); }
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

/// How a backend hands out scatter targets during
/// [`Model::bind_hessian_positions`].
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


/// Hessian block for a single model type.
///
/// Accumulates the upper triangle of the Gauss-Newton Hessian approximation
/// (2·dr·dr^T) from constraint residuals involving one model's parameters.
/// Gradient entries (2·r·dr) are written directly to the global gradient
/// vector by `add_residual` — no per-block grad buffer.
/// `N` equals `A::PARAM_COUNT`. `T` is the float type (f32 or f64, default f64).
///
/// Created by generated constraint code; users rarely construct these manually.
#[derive(Clone)]
pub struct SelfBlock<A, const N: usize, const M: usize, T: crate::utils::Float = f64> {
    indices: [u32; N],
    /// Value-buffer position of this block's tile origin and the tile's
    /// column stride, from [`bind_hessian_positions`](Self::bind_hessian_positions).
    /// `u32::MAX` base means unbound; see [`TilePosition`].
    pos: TilePosition,
    hessian: [T; M], // upper triangle, M = N(N+1)/2 (the macro sizes this)
    _marker: std::marker::PhantomData<(A, T)>,
}

impl<A, const N: usize, const M: usize, T: crate::utils::Float> Default for SelfBlock<A, N, M, T> {
    fn default() -> Self { Self::new() }
}

impl<A, const N: usize, const M: usize, T: crate::utils::Float> SelfBlock<A, N, M, T> {
    /// Compile-time guard that `M` matches `N` (the macro sets both; a
    /// hand-written mismatch fails here rather than corrupting silently).
    const CHECK_M: () = assert!(M == N * (N + 1) / 2, "SelfBlock: M must equal N*(N+1)/2");

    /// Create a new zeroed block.
    pub fn new() -> Self {
        let () = Self::CHECK_M;
        SelfBlock {
            indices: [u32::MAX; N],
            pos: TilePosition::UNBOUND,
            hessian: [T::zero(); M],
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the global parameter indices for this block.
    pub fn set_indices(&mut self, indices: &[u32; N]) {
        self.indices = *indices;
    }

    /// Append this entity's parameter span as `(offset, width)`: the
    /// smallest live index and the count of live (non-fixed) params.
    /// All-fixed entities append nothing.
    pub fn collect_param_block(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        let mut min = u32::MAX;
        let mut count = 0u32;
        for &i in &self.indices {
            if i != u32::MAX {
                if i < min { min = i; }
                count += 1;
            }
        }
        if count > 0 {
            out.push((min, count));
        }
    }

    /// Emit one representative scalar coordinate for this block's cell:
    /// (anchor, anchor) on the diagonal of the entity partition.
    /// All-fixed blocks emit nothing (they scatter nothing either).
    pub fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        let mut min = u32::MAX;
        for &i in &self.indices {
            if i != u32::MAX && i < min { min = i; }
        }
        if min != u32::MAX {
            out.push((min, min));
        }
    }

    /// Bind this block's diagonal tile for
    /// [`accumulate_hessian_sparse_indexed`](Self::accumulate_hessian_sparse_indexed).
    /// All of the block's parameters live in one entity span, so the whole
    /// upper triangle sits in a single tile and a tiled binder leaves `out`
    /// untouched. A scalar binder pushes one position per entry, in the
    /// emission order of
    /// [`accumulate_hessian_sparse`](Self::accumulate_hessian_sparse).
    pub fn bind_hessian_positions(
        &mut self,
        binder: &mut HessianBinder,
        out: &mut std::vec::Vec<ValueIndex>,
    ) {
        let resolve = match binder {
            HessianBinder::Tiled(bind) => {
                self.pos = bind_tile(*bind, &self.indices, &self.indices);
                return;
            }
            HessianBinder::Scalar(resolve) => resolve,
        };
        self.pos = TilePosition::MAPPED;
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                out.push(value_index(resolve(lo, hi)));
            }
        }
    }

    /// Reset Hessian to zero. Gradient lives in the global grad vector,
    /// which LM zeros before each compute pass.
    pub fn zero(&mut self) {
        self.hessian.fill(T::zero());
    }

    /// No-op: inline storage owns no heap to release. Present so a model's
    /// generated `release_blocks()` can call `.release()` on every block
    /// uniformly (see [`BoxedSelfBlock`], where it frees the Hessian).
    pub fn release(&mut self) {}

    /// Return true if any parameter in this block is being optimized.
    pub fn is_active(&self) -> bool {
        self.indices.iter().any(|&i| i != u32::MAX)
    }

    /// Add one residual's contribution. Writes `2·r·dr[i]` into the global
    /// `grad` at this block's indices, and accumulates `2·dr·dr^T` into the
    /// block's internal upper-triangular Hessian.
    pub fn add_residual(&mut self, r: T, dr: &[T; N], grad: &mut [T]) {
        let two = T::two();
        for i in 0..N {
            let gi = self.indices[i];
            // Fixed params (u32::MAX): no gradient slot, and no Hessian
            // pair involving them is ever read by the accumulate paths
            // (they skip inactive indices) -- skip the whole row.
            if gi == u32::MAX { continue; }
            grad[gi as usize] += two * r * dr[i];
            let tdi = two * dr[i];
            for j in i..N {
                if self.indices[j] == u32::MAX { continue; }
                self.hessian[tri_idx(N, i, j)] += tdi * dr[j];
            }
        }
    }

    /// Like [`add_residual`](Self::add_residual) but scales both the gradient
    /// and Hessian contribution by a robust weight `w` (the loss derivative
    /// `rho'(s)` at the block's squared norm). `w = 1` is bit-identical to
    /// `add_residual`.
    pub fn add_residual_with_loss(&mut self, w: T, r: T, dr: &[T; N], grad: &mut [T]) {
        let two_w = T::two() * w;
        let wr = two_w * r;
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            grad[gi as usize] += wr * dr[i];
            let tdi = two_w * dr[i];
            for j in i..N {
                if self.indices[j] == u32::MAX { continue; }
                self.hessian[tri_idx(N, i, j)] += tdi * dr[j];
            }
        }
    }

    /// Accumulate this block's Hessian into the full dense symmetric hessian.
    /// Generic over the target width; the conversion is an identity when it
    /// matches the block's storage.
    pub fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        let n_total = (hessian.len() as f64).sqrt() as usize;
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            let gi = gi as usize;
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let gj = gj as usize;
                let val = F::from(self.hessian[tri_idx(N, i, j)]).unwrap();
                hessian[gi * n_total + gj] += val;
                if gi != gj {
                    hessian[gj * n_total + gi] += val;
                }
            }
        }
    }

    /// Accumulate into upper-band format (column-major, (kd+1)*n).
    /// Returns Err if any element exceeds bandwidth kd.
    pub fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
        -> Result<(), crate::simple_lm::BandOverflow>
    {
        let ldab = kd + 1;
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            let gi = gi as usize;
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let gj = gj as usize;
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                if hi - lo > kd {
                    return Err(crate::simple_lm::BandOverflow { row: lo, col: hi, kd });
                }
                let val = F::from(self.hessian[tri_idx(N, i, j)]).unwrap();
                band[(kd + lo - hi) + hi * ldab] += val;
            }
        }
        Ok(())
    }

    /// Accumulate into COO (triplet) sparse format. Upper triangle only.
    pub fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = F::from(self.hessian[tri_idx(N, i, j)]).unwrap();
                coo.push(lo, hi, val);
            }
        }
    }

    /// Accumulate directly into CSC vals array using position lookup.
    pub fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = F::from(self.hessian[tri_idx(N, i, j)]).unwrap();
                if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                    csc.vals[pos] += val;
                }
            }
        }
    }

    /// Accumulate into the assembled value buffer through this block's bound
    /// tile, leaving `positions` and `cursor` untouched. Blocks bound against
    /// a pattern with no tile to walk fall back to the map.
    pub fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        if !self.pos.tiled() {
            return self.accumulate_mapped_indexed(vals, positions, cursor);
        }
        let start = tile_start(&self.indices);
        let (base, stride, start) =
            (self.pos.base as usize, self.pos.stride as usize, start as usize);
        // Column offset of each live slot: invariant across the outer loop.
        let mut col = [0usize; N];
        for (c, &g) in std::iter::zip(&mut col, &self.indices) {
            if g != u32::MAX {
                *c = (g as usize - start) * stride;
            }
        }
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            let row = base + (gi as usize - start);
            let tri = i * (2 * N - i - 1) / 2;
            for j in i..N {
                if self.indices[j] == u32::MAX { continue; }
                vals[row + col[j]] += F::from(self.hessian[tri + j]).unwrap();
            }
        }
    }

    /// The untiled case of
    /// [`accumulate_hessian_sparse_indexed`](Self::accumulate_hessian_sparse_indexed):
    /// one cached position per entry, `cursor` advancing in lockstep with the
    /// block traversal. An all-fixed block emits nothing and so consumes none.
    fn accumulate_mapped_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            for j in i..N {
                if self.indices[j] == u32::MAX { continue; }
                vals[positions[*cursor] as usize] += F::from(self.hessian[tri_idx(N, i, j)]).unwrap();
                *cursor += 1;
            }
        }
    }
}

/// Heap-backed twin of [`SelfBlock`]: the whole block (indices + Hessian)
/// lives behind a single `Box`, wrapped in `Option` so it can be released
/// between solves and re-allocated on demand. `None` = released (no
/// allocation at all); `Some` = one heap block. Every method delegates to the
/// inline [`SelfBlock`], so the two are bit-for-bit identical.
///
/// Opt into it (`hb: BoxedSelfBlock<Self>`) when the inline `[T; M]` would
/// fatten the entity struct too much (very large parameter counts), or to
/// reclaim all transient Hessian memory between solves via the root's
/// generated `release_blocks()`. Otherwise prefer [`SelfBlock`], which never
/// allocates. `N`/`M` are supplied by the macro, exactly as for `SelfBlock`.
pub struct BoxedSelfBlock<A, const N: usize, const M: usize, T: crate::utils::Float = f64> {
    inner: Option<std::boxed::Box<SelfBlock<A, N, M, T>>>,
}

impl<A, const N: usize, const M: usize, T: crate::utils::Float> Default for BoxedSelfBlock<A, N, M, T> {
    fn default() -> Self { Self::new() }
}

impl<A, const N: usize, const M: usize, T: crate::utils::Float> BoxedSelfBlock<A, N, M, T> {
    /// Create an empty (released) block. Storage is allocated by the first
    /// `set_indices` call that finds the block active.
    pub fn new() -> Self {
        BoxedSelfBlock { inner: None }
    }

    /// Materialise the inner block, allocating it if released.
    fn ensure(&mut self) -> &mut SelfBlock<A, N, M, T> {
        self.inner.get_or_insert_with(|| std::boxed::Box::new(SelfBlock::new()))
    }

    /// Whether this block currently holds a heap allocation. False when
    /// released, and false for an inactive block (its entity is not being
    /// optimized -- every index is the `u32::MAX` fixed sentinel), so a frozen
    /// sub-tree costs no Hessian memory. Mainly for diagnostics and tests.
    pub fn is_allocated(&self) -> bool {
        self.inner.is_some()
    }

    /// Set the global parameter indices. Allocates the block only if it is
    /// active (at least one index is being optimized); an all-fixed block is
    /// left released, so a frozen sub-tree never allocates its Hessian. The
    /// solver calls this once, before any `zero`/`add_residual`, so activity
    /// is settled before those run.
    pub fn set_indices(&mut self, indices: &[u32; N]) {
        if indices.iter().any(|&i| i != u32::MAX) {
            self.ensure().set_indices(indices);
        } else {
            self.inner = None;
        }
    }

    /// Reset the Hessian to zero. No-op when unallocated: `set_indices` has
    /// already materialised every active block, so there is nothing to zero
    /// for an inactive/released one.
    pub fn zero(&mut self) {
        if let Some(b) = &mut self.inner { b.zero(); }
    }

    /// Append this entity's parameter span; a released or all-fixed block
    /// appends nothing (see [`SelfBlock::collect_param_block`]).
    pub fn collect_param_block(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        if let Some(b) = &self.inner { b.collect_param_block(out); }
    }

    /// Free the whole block's heap allocation. The next `set_indices` on an
    /// active block re-allocates it.
    pub fn release(&mut self) {
        self.inner = None;
    }

    /// Return true if any parameter in this block is being optimized.
    pub fn is_active(&self) -> bool {
        self.inner.as_ref().is_some_and(|b| b.is_active())
    }

    /// Add one residual's contribution (see [`SelfBlock::add_residual`]).
    /// No-op when unallocated: an inactive block (all indices fixed) writes
    /// nothing to grad or Hessian anyway.
    pub fn add_residual(&mut self, r: T, dr: &[T; N], grad: &mut [T]) {
        if let Some(b) = &mut self.inner { b.add_residual(r, dr, grad); }
    }

    /// Robust-weighted residual add (see [`SelfBlock::add_residual_with_loss`]).
    /// No-op when unallocated.
    pub fn add_residual_with_loss(&mut self, w: T, r: T, dr: &[T; N], grad: &mut [T]) {
        if let Some(b) = &mut self.inner { b.add_residual_with_loss(w, r, dr, grad); }
    }

    /// Scatter this block into a dense n x n Hessian.
    pub fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        if let Some(b) = &self.inner { b.accumulate_hessian(hessian); }
    }

    /// Scatter this block into LAPACK upper-band storage with half-bandwidth `kd`.
    pub fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
        -> Result<(), crate::simple_lm::BandOverflow>
    {
        match &self.inner {
            Some(b) => b.accumulate_hessian_band(band, kd),
            None => Ok(()),
        }
    }

    /// Append this block's upper-triangle (row, col) cells to `out` -- the pattern pass.
    pub fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        if let Some(b) = &self.inner { b.collect_hessian_cells(out); }
    }

    /// Bind this block's tile in the assembled value buffer.
    pub fn bind_hessian_positions(
        &mut self,
        binder: &mut HessianBinder,
        out: &mut std::vec::Vec<ValueIndex>,
    ) {
        if let Some(b) = &mut self.inner { b.bind_hessian_positions(binder, out); }
    }

    /// Scatter this block into COO triplets.
    pub fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        if let Some(b) = &self.inner { b.accumulate_hessian_sparse(coo); }
    }

    /// Scatter this block into a prebuilt CSC structure by position lookup.
    pub fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        if let Some(b) = &self.inner { b.accumulate_hessian_sparse_direct(csc); }
    }

    /// Scatter this block into CSC values through cached positions.
    pub fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        if let Some(b) = &self.inner {
            b.accumulate_hessian_sparse_indexed(vals, positions, cursor);
        }
    }
}

/// Hessian block coupling two model types — stores ONLY the rectangular A×B
/// cross Hessian pairs. A's gradient and A-A diagonal live in A's own
/// [`SelfBlock`]; same for B. This matches the refactor where every
/// params-having Model owns a `SelfBlock<Self>`, and cross blocks carry only
/// the pieces that don't fit in a per-entity block.
///
/// `NA = A::PARAM_COUNT`, `NB = B::PARAM_COUNT`. Internal Hessian storage
/// is NA×NB row-major (one entry per cross pair). No grad, no A-A, no B-B.
/// `T` is the float type (f32 or f64, default f64).
#[derive(Clone)]
pub struct CrossBlock<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float = f64> {
    indices_a: [u32; NA],
    indices_b: [u32; NB],
    /// Origin and column stride of the A-B tile; see [`TilePosition`].
    pos: TilePosition,
    cross_hessian: [T; P],    // NA*NB row-major (the macro sizes this)
    _marker: std::marker::PhantomData<(A, B, T)>,
}

impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> Default for CrossBlock<A, B, NA, NB, P, T> {
    fn default() -> Self { Self::new() }
}

impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> CrossBlock<A, B, NA, NB, P, T> {
    /// Compile-time guard that `P` matches `NA*NB` (set by the macro).
    const CHECK_P: () = assert!(P == NA * NB, "CrossBlock: P must equal NA*NB");
    /// Create a new zeroed cross-block.
    pub fn new() -> Self {
        let () = Self::CHECK_P;
        CrossBlock {
            indices_a: [u32::MAX; NA],
            indices_b: [u32::MAX; NB],
            pos: TilePosition::UNBOUND,
            cross_hessian: [T::zero(); P],
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of parameters belonging to model A.
    pub fn na(&self) -> usize { NA }
    /// Return the number of parameters belonging to model B.
    pub fn nb(&self) -> usize { NB }

    /// Set the global parameter indices.
    ///
    /// The two slots may resolve to the same entity ("aliased", e.g. a
    /// distance between the two endpoints of a single line): the two
    /// symmetric accumulate writes then land on the same Hessian cell,
    /// summing to the 2 * dr_a * dr_b diagonal contribution the shared
    /// parameter requires. No special casing anywhere.
    pub fn set_indices(&mut self, a_indices: &[u32], b_indices: &[u32]) {
        debug_assert_eq!(a_indices.len(), NA);
        debug_assert_eq!(b_indices.len(), NB);
        self.indices_a.copy_from_slice(a_indices);
        self.indices_b.copy_from_slice(b_indices);
    }

    /// Reset cross hessian to zero.
    pub fn zero(&mut self) {
        self.cross_hessian.fill(T::zero());
    }

    /// No-op: inline storage owns no heap to release (see [`BoxedCrossBlock`]).
    pub fn release(&mut self) {}

    /// Return true if any parameter in this block is being optimized.
    pub fn is_active(&self) -> bool {
        self.indices_a.iter().any(|&i| i != u32::MAX)
            || self.indices_b.iter().any(|&i| i != u32::MAX)
    }

    /// Add one residual's cross contribution: accumulates `2 * dr_a[i] * dr_b[j]`
    /// into the A×B rectangular Hessian. Gradient and A-A / B-B pairs must be
    /// added separately via A's and B's SelfBlocks (they receive `r` + per-side
    /// `dr` directly).
    pub fn add_residual_cross(&mut self, _r: T, dr_a: &[T; NA], dr_b: &[T; NB]) {
        let two = T::two();
        for i in 0..NA {
            let dai = dr_a[i];
            if dai == T::zero() { continue; }
            let row = i * NB;
            for j in 0..NB {
                self.cross_hessian[row + j] += two * dai * dr_b[j];
            }
        }
    }

    /// Robust-weighted variant of [`add_residual_cross`](Self::add_residual_cross):
    /// scales the cross Hessian by the loss weight `w`. `w = 1` is bit-identical.
    /// `_r` is ignored (the gradient goes through the SelfBlocks); it is present
    /// so the macro can route every accumulation call uniformly.
    pub fn add_residual_cross_with_loss(&mut self, w: T, _r: T, dr_a: &[T; NA], dr_b: &[T; NB]) {
        let two_w = T::two() * w;
        for i in 0..NA {
            let dai = dr_a[i];
            if dai == T::zero() { continue; }
            let row = i * NB;
            for j in 0..NB {
                self.cross_hessian[row + j] += two_w * dai * dr_b[j];
            }
        }
    }

    /// Accumulate cross pairs into the full dense symmetric hessian.
    /// Generic over the target width; the conversion is an identity when it
    /// matches the block's storage.
    pub fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        let n_total = (hessian.len() as f64).sqrt() as usize;
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let gi = gi as usize;
            let row = i * NB;
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                let gj = gj as usize;
                let val = F::from(self.cross_hessian[row + j]).unwrap();
                hessian[gi * n_total + gj] += val;
                hessian[gj * n_total + gi] += val;
            }
        }
    }

    /// Accumulate into upper-band format (column-major, (kd+1)*n).
    pub fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
        -> Result<(), crate::simple_lm::BandOverflow>
    {
        let ldab = kd + 1;
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let gi = gi as usize;
            let row = i * NB;
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                let gj = gj as usize;
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                if hi - lo > kd {
                    return Err(crate::simple_lm::BandOverflow { row: lo, col: hi, kd });
                }
                let val = F::from(self.cross_hessian[row + j]).unwrap();
                // Aliased slots (same entity in both refs): the triangle
                // stores each symmetric pair once, so the diagonal needs
                // both of the 2*dr_a*dr_b contributions explicitly.
                let val = if gi == gj { val + val } else { val };
                band[(kd + lo - hi) + hi * ldab] += val;
            }
        }
        Ok(())
    }

    /// Emit one representative scalar coordinate for this block's cell.
    /// Both entities are contiguous spans, so every pair lands in one
    /// cell of the entity partition (the diagonal cell when aliased).
    pub fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        let min_live = |idx: &[u32]| {
            let mut min = u32::MAX;
            for &i in idx {
                if i != u32::MAX && i < min { min = i; }
            }
            min
        };
        let (ma, mb) = (min_live(&self.indices_a), min_live(&self.indices_b));
        if ma != u32::MAX && mb != u32::MAX {
            out.push((ma.min(mb), ma.max(mb)));
        }
    }

    /// Bind this block's tile for
    /// [`accumulate_hessian_sparse_indexed`](Self::accumulate_hessian_sparse_indexed).
    /// Both entities are contiguous spans, so every cross pair lands in the
    /// one tile (the diagonal tile when the two slots alias one entity).
    /// A scalar binder falls back to the map; see
    /// [`SelfBlock::bind_hessian_positions`].
    pub fn bind_hessian_positions(
        &mut self,
        binder: &mut HessianBinder,
        out: &mut std::vec::Vec<ValueIndex>,
    ) {
        let resolve = match binder {
            HessianBinder::Tiled(bind) => {
                self.pos = bind_tile(*bind, &self.indices_a, &self.indices_b);
                return;
            }
            HessianBinder::Scalar(resolve) => resolve,
        };
        self.pos = TilePosition::MAPPED;
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                out.push(value_index(resolve(lo, hi)));
            }
        }
    }

    /// Accumulate into COO sparse format. Upper triangle only.
    pub fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let row = i * NB;
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = F::from(self.cross_hessian[row + j]).unwrap();
                // Aliased diagonal: see accumulate_hessian_band.
                let val = if gi == gj { val + val } else { val };
                coo.push(lo, hi, val);
            }
        }
    }

    /// Accumulate directly into CSC vals via position lookup.
    pub fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let row = i * NB;
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = F::from(self.cross_hessian[row + j]).unwrap();
                // Aliased diagonal: see accumulate_hessian_band.
                let val = if gi == gj { val + val } else { val };
                if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                    csc.vals[pos] += val;
                }
            }
        }
    }

    /// Accumulate into the assembled value buffer through this block's bound
    /// tile; see the note on
    /// [`SelfBlock::accumulate_hessian_sparse_indexed`].
    pub fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        if !self.pos.tiled() {
            return self.accumulate_mapped_indexed(vals, positions, cursor);
        }
        let (sa, sb) = (tile_start(&self.indices_a), tile_start(&self.indices_b));
        let (base, stride) = (self.pos.base as usize, self.pos.stride as usize);
        if sa == sb {
            // Aliased: both slots index one entity, so the pairs land on a
            // diagonal tile and which side is the row flips per element.
            return self.accumulate_aliased_indexed(vals, base, stride, sa as usize);
        }
        // The tile holds the upper block triangle, so the lower-numbered
        // entity walks the rows and the other walks the columns.
        let (step_a, step_b) = if sa < sb { (1, stride) } else { (stride, 1) };
        let (sa, sb) = (sa as usize, sb as usize);
        // Offset of each live B slot: invariant across the outer loop.
        let mut off_b = [0usize; NB];
        for (o, &g) in std::iter::zip(&mut off_b, &self.indices_b) {
            if g != u32::MAX {
                *o = (g as usize - sb) * step_b;
            }
        }
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let pos = base + (gi as usize - sa) * step_a;
            let row = i * NB;
            for j in 0..NB {
                if self.indices_b[j] == u32::MAX { continue; }
                vals[pos + off_b[j]] += F::from(self.cross_hessian[row + j]).unwrap();
            }
        }
    }

    /// The aliased case of [`accumulate_hessian_sparse_indexed`](Self::accumulate_hessian_sparse_indexed):
    /// one entity in both slots, every pair on its diagonal tile.
    fn accumulate_aliased_indexed<F: crate::utils::Float>(&self, vals: &mut [F], base: usize, stride: usize, start: usize) {
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let row = i * NB;
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = F::from(self.cross_hessian[row + j]).unwrap();
                // The triangle stores each symmetric pair once, so a pair that
                // lands on the diagonal needs both contributions.
                let val = if gi == gj { val + val } else { val };
                vals[base + (hi as usize - start) * stride + (lo as usize - start)] += val;
            }
        }
    }

    /// The untiled case of
    /// [`accumulate_hessian_sparse_indexed`](Self::accumulate_hessian_sparse_indexed):
    /// one cached position per entry.
    fn accumulate_mapped_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let row = i * NB;
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                let val = F::from(self.cross_hessian[row + j]).unwrap();
                // Aliased diagonal: see accumulate_hessian_band.
                let val = if gi == gj { val + val } else { val };
                vals[positions[*cursor] as usize] += val;
                *cursor += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------

/// Heap-backed twin of [`CrossBlock`]: the whole block (both index arrays +
/// cross-Hessian) lives behind a single `Box`, wrapped in `Option` so it can
/// be released between solves and re-allocated on demand. `None` = released;
/// `Some` = one heap block. Every method delegates to the inline
/// [`CrossBlock`]. See [`BoxedSelfBlock`] for when to opt in.
pub struct BoxedCrossBlock<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float = f64> {
    inner: Option<std::boxed::Box<CrossBlock<A, B, NA, NB, P, T>>>,
}

impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> Default for BoxedCrossBlock<A, B, NA, NB, P, T> {
    fn default() -> Self { Self::new() }
}

impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> BoxedCrossBlock<A, B, NA, NB, P, T> {
    /// Create an empty (released) block. Storage is allocated by the first
    /// `set_indices` call that finds the block active.
    pub fn new() -> Self {
        BoxedCrossBlock { inner: None }
    }

    /// Materialise the inner block, allocating it if released.
    fn ensure(&mut self) -> &mut CrossBlock<A, B, NA, NB, P, T> {
        self.inner.get_or_insert_with(|| std::boxed::Box::new(CrossBlock::new()))
    }

    /// Return the number of parameters belonging to model A.
    pub fn na(&self) -> usize { NA }
    /// Return the number of parameters belonging to model B.
    pub fn nb(&self) -> usize { NB }

    /// Whether this block currently holds a heap allocation. False when
    /// released, and false for an inactive block (both endpoints are fixed --
    /// every index is the `u32::MAX` sentinel), so a link between two frozen
    /// entities costs no Hessian memory. Mainly for diagnostics and tests.
    pub fn is_allocated(&self) -> bool {
        self.inner.is_some()
    }

    /// Set the global parameter indices. Allocates the block only if it is
    /// active (at least one index, on either side, is being optimized); a link
    /// between two fully-fixed entities is left released. The solver calls this
    /// once, before any `zero`/`add_residual_cross`.
    pub fn set_indices(&mut self, a_indices: &[u32], b_indices: &[u32]) {
        let active = a_indices.iter().chain(b_indices.iter()).any(|&i| i != u32::MAX);
        if active {
            self.ensure().set_indices(a_indices, b_indices);
        } else {
            self.inner = None;
        }
    }

    /// Reset the cross-Hessian to zero. No-op when unallocated: `set_indices`
    /// has already materialised every active block.
    pub fn zero(&mut self) {
        if let Some(b) = &mut self.inner { b.zero(); }
    }

    /// Free the whole block's heap allocation. The next `set_indices` on an
    /// active block re-allocates it.
    pub fn release(&mut self) {
        self.inner = None;
    }

    /// Return true if any parameter in this block is being optimized.
    pub fn is_active(&self) -> bool {
        self.inner.as_ref().is_some_and(|b| b.is_active())
    }

    /// Add one residual's cross contribution (see [`CrossBlock::add_residual_cross`]).
    /// No-op when unallocated: an inactive block contributes nothing.
    pub fn add_residual_cross(&mut self, r: T, dr_a: &[T; NA], dr_b: &[T; NB]) {
        if let Some(b) = &mut self.inner { b.add_residual_cross(r, dr_a, dr_b); }
    }

    /// Robust-weighted cross add (see [`CrossBlock::add_residual_cross_with_loss`]).
    /// No-op when unallocated.
    pub fn add_residual_cross_with_loss(&mut self, w: T, r: T, dr_a: &[T; NA], dr_b: &[T; NB]) {
        if let Some(b) = &mut self.inner { b.add_residual_cross_with_loss(w, r, dr_a, dr_b); }
    }

    /// Scatter this block into a dense n x n Hessian.
    pub fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
        if let Some(b) = &self.inner { b.accumulate_hessian(hessian); }
    }

    /// Scatter this block into LAPACK upper-band storage with half-bandwidth `kd`.
    pub fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
        -> Result<(), crate::simple_lm::BandOverflow>
    {
        match &self.inner {
            Some(b) => b.accumulate_hessian_band(band, kd),
            None => Ok(()),
        }
    }

    /// Append this block's upper-triangle (row, col) cells to `out` -- the pattern pass.
    pub fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
        if let Some(b) = &self.inner { b.collect_hessian_cells(out); }
    }

    /// Bind this block's tile in the assembled value buffer.
    pub fn bind_hessian_positions(
        &mut self,
        binder: &mut HessianBinder,
        out: &mut std::vec::Vec<ValueIndex>,
    ) {
        if let Some(b) = &mut self.inner { b.bind_hessian_positions(binder, out); }
    }

    /// Scatter this block into COO triplets.
    pub fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
        if let Some(b) = &self.inner { b.accumulate_hessian_sparse(coo); }
    }

    /// Scatter this block into a prebuilt CSC structure by position lookup.
    pub fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
        if let Some(b) = &self.inner { b.accumulate_hessian_sparse_direct(csc); }
    }

    /// Scatter this block into CSC values through cached positions.
    pub fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
        if let Some(b) = &self.inner {
            b.accumulate_hessian_sparse_indexed(vals, positions, cursor);
        }
    }
}

// ---------------------------------------------------------------------------

/// Sparse cross-only Hessian accumulation block for N-ary constraints.
///
/// After the refactor: TripletBlock stores ONLY across-entity Hessian pairs
/// (pairs where the two params belong to different entity spans). The
/// within-entity `H[A,A]` / `H[B,B]` / ... diagonals live in each entity's
/// `SelfBlock<Self>`, and the gradient lives there too. This matches the
/// invariant that every `∂r/∂p_i · ∂r/∂p_j` pair is written to exactly one
/// block with no duplication.
///
/// **Prefer [`CrossBlock`] for 2-entity constraints.** CrossBlock uses packed
/// dense storage with compile-time-known NA×NB size, which is more
/// cache-friendly than COO. Use TripletBlock only when a constraint couples
/// 3+ entities.
///
/// Two entry points:
/// - [`add_residual`](TripletBlock::add_residual) for direct callers with a
///   flat param layout and no per-entity SelfBlocks: writes the gradient into
///   the provided global slice AND pushes the full upper-triangle Hessian
///   (including diagonal). One call, everything done.
/// - [`add_residual_cross`](TripletBlock::add_residual_cross) for macro-
///   emitted N-ary constraints where each participating entity has its own
///   SelfBlock holding its grad+diagonal: stores ONLY across-entity pairs,
///   using the `entity_offsets` span list to skip within-entity pairs.
#[derive(Clone)]
pub struct TripletBlock<T: crate::utils::Float = f64> {
    /// Hessian entries: upper-triangle (lo, hi, 2·dr_i·dr_j). Only cross-
    /// entity pairs are stored (within-entity pairs live in each entity's
    /// `SelfBlock`). Callers that manage their own flat param layout
    /// without per-entity blocks can pass each param as its own "entity"
    /// (entity_offsets = [0, 1, 2, ..., N]) to make every pair cross.
    pub hessian: std::vec::Vec<(u32, u32, T)>,
}

impl<T: crate::utils::Float> Default for TripletBlock<T> {
    fn default() -> Self { Self::new() }
}

impl<T: crate::utils::Float> TripletBlock<T> {
    /// Create an empty triplet block.
    pub fn new() -> Self {
        TripletBlock { hessian: std::vec::Vec::new() }
    }

    /// Reset to empty (called at start of each optimization step).
    pub fn zero(&mut self) {
        self.hessian.clear();
    }

    /// No-op: TripletBlock already stores its triplets in a heap Vec that is
    /// cleared each step, so there is nothing to release. Present so the
    /// generated `release_blocks()` can call `.release()` uniformly.
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
    /// Hessian diagonal. Stores ONLY across-entity pairs — within-entity
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
    /// CONTRACT: a TripletBlock must be refilled with the same entries
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
            "sparsity pattern changed between iterations: TripletBlock holds {} \
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

macro_rules! block_model_methods {
    () => {
        #[inline]
        fn zero_blocks(&mut self) { self.zero(); }
        #[inline]
        fn release_blocks(&mut self) { self.release(); }
        #[inline]
        fn accumulate_hessian<F: crate::utils::Float>(&self, hessian: &mut [F]) {
            self.accumulate_hessian(hessian);
        }
        #[inline]
        fn accumulate_hessian_band<F: crate::utils::Float>(&self, band: &mut [F], kd: usize)
            -> Result<(), crate::simple_lm::BandOverflow> {
            self.accumulate_hessian_band(band, kd)
        }
        #[inline]
        fn accumulate_hessian_sparse<F: crate::utils::Float>(&self, coo: &mut crate::simple_lm::CooMatrix<F>) {
            self.accumulate_hessian_sparse(coo);
        }
        #[inline]
        fn accumulate_hessian_sparse_direct<F: crate::utils::Float>(&self, csc: &mut crate::simple_lm::CscMatrix<F>) {
            self.accumulate_hessian_sparse_direct(csc);
        }
        #[inline]
        fn accumulate_hessian_sparse_indexed<F: crate::utils::Float>(&self, vals: &mut [F], positions: &[ValueIndex], cursor: &mut usize) {
            self.accumulate_hessian_sparse_indexed(vals, positions, cursor);
        }
        #[inline]
        fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
            self.collect_hessian_cells(out);
        }
        #[inline]
        fn bind_hessian_positions(
            &mut self,
            binder: &mut HessianBinder,
            out: &mut std::vec::Vec<ValueIndex>,
        ) {
            self.bind_hessian_positions(binder, out);
        }
    };
}

/// The self blocks also carry the entity's param-block span.
macro_rules! collect_param_block_method {
    () => {
        #[inline]
        fn collect_param_blocks(&self, out: &mut std::vec::Vec<(u32, u32)>) {
            self.collect_param_block(out);
        }
    };
}

impl<A, const N: usize, const M: usize, T: crate::utils::Float> Model for SelfBlock<A, N, M, T> {
    block_model_methods!();
    collect_param_block_method!();
}
impl<A, const N: usize, const M: usize, T: crate::utils::Float> Model for BoxedSelfBlock<A, N, M, T> {
    block_model_methods!();
    collect_param_block_method!();
}
impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> Model
    for CrossBlock<A, B, NA, NB, P, T> {
    block_model_methods!();
}
impl<A, B, const NA: usize, const NB: usize, const P: usize, T: crate::utils::Float> Model
    for BoxedCrossBlock<A, B, NA, NB, P, T> {
    block_model_methods!();
}
impl<T: crate::utils::Float> Model for TripletBlock<T> {
    block_model_methods!();
}

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

    #[test]
    fn selfblock_band_matches_dense() {
        let n = 4;
        let kd = 2;
        let mut blk: SelfBlock<Param<f64>, 3, 6, f64> = SelfBlock::new();
        blk.set_indices(&[0, 1, 2]);
        let mut grad = vec![0.0; n];
        blk.add_residual(0.3, &[1.0, 0.5, -0.25], &mut grad);
        blk.add_residual(-0.7, &[0.2, -1.5, 0.75], &mut grad);

        let mut dense = vec![0.0; n * n];
        blk.accumulate_hessian(&mut dense);
        let mut band = vec![0.0; (kd + 1) * n];
        blk.accumulate_hessian_band(&mut band, kd).unwrap();
        assert_eq!(densify_band(&band, n, kd), dense);
    }

    #[test]
    fn crossblock_band_matches_dense() {
        let n = 5;
        let kd = 3;
        let mut blk: CrossBlock<Param<f64>, Param<f64>, 2, 2, 4, f64> = CrossBlock::new();
        blk.set_indices(&[0, 1], &[2, 3]);
        blk.add_residual_cross(0.4, &[1.0, -0.5], &[0.25, 2.0]);
        blk.add_residual_cross(-1.1, &[0.3, 0.7], &[-0.6, 0.1]);

        let mut dense = vec![0.0; n * n];
        blk.accumulate_hessian(&mut dense);
        let mut band = vec![0.0; (kd + 1) * n];
        blk.accumulate_hessian_band(&mut band, kd).unwrap();
        assert_eq!(densify_band(&band, n, kd), dense);
    }

    #[test]
    fn tripletblock_band_matches_dense() {
        let n = 4;
        let kd = 2;
        let mut blk: TripletBlock<f64> = TripletBlock::new();
        let mut grad = vec![0.0; n];
        blk.add_residual(0.3, &[0, 1, 2], &[1.0, 0.5, -0.25], &mut grad);
        blk.add_residual(-0.7, &[1, 3], &[2.0, -1.5], &mut grad);

        let mut dense = vec![0.0; n * n];
        blk.accumulate_hessian(&mut dense);
        let mut band = vec![0.0; (kd + 1) * n];
        blk.accumulate_hessian_band(&mut band, kd).unwrap();
        assert_eq!(densify_band(&band, n, kd), dense,
            "TripletBlock band accumulation must use the same upper-band \
             layout as SelfBlock/CrossBlock and the band solvers");
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
        let mut blk: TripletBlock<f64> = TripletBlock::new();
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
    fn tripletblock_band_matches_selfblock_band() {
        // The same residual pushed through both block types must produce
        // byte-identical gradient and band arrays.
        let n = 4;
        let kd = 3;
        let r = 0.9;
        let dr = [1.0, -2.0, 0.5];
        let idx = [0u32, 1, 3];

        let mut g_self = vec![0.0; n];
        let mut sb: SelfBlock<Param<f64>, 3, 6, f64> = SelfBlock::new();
        sb.set_indices(&idx);
        sb.add_residual(r, &dr, &mut g_self);

        let mut g_triplet = vec![0.0; n];
        let mut tb: TripletBlock<f64> = TripletBlock::new();
        tb.add_residual(r, &idx, &dr, &mut g_triplet);

        assert_eq!(g_self, g_triplet);

        let mut band_self = vec![0.0; (kd + 1) * n];
        sb.accumulate_hessian_band(&mut band_self, kd).unwrap();
        let mut band_triplet = vec![0.0; (kd + 1) * n];
        tb.accumulate_hessian_band(&mut band_triplet, kd).unwrap();
        assert_eq!(band_self, band_triplet);
    }
}
