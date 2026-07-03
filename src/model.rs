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
    fn write_to32(&self, dst: &mut [f32]);
    fn read_from32(src: &[f32]) -> Self;
    fn write_to64(&self, dst: &mut [f64]);
    fn read_from64(src: &[f64]) -> Self;
}

impl ParamType for f32 {
    const SIZE: usize = 1;
    const SUFFIXES: &'static [&'static str] = &[""];
    fn write_to32(&self, dst: &mut [f32]) { dst[0] = *self; }
    fn read_from32(src: &[f32]) -> Self { src[0] }
    fn write_to64(&self, dst: &mut [f64]) { dst[0] = *self as f64; }
    fn read_from64(src: &[f64]) -> Self { src[0] as f32 }
}

impl ParamType for f64 {
    const SIZE: usize = 1;
    const SUFFIXES: &'static [&'static str] = &[""];
    fn write_to32(&self, dst: &mut [f32]) { dst[0] = *self as f32; }
    fn read_from32(src: &[f32]) -> Self { src[0] as f64 }
    fn write_to64(&self, dst: &mut [f64]) { dst[0] = *self; }
    fn read_from64(src: &[f64]) -> Self { src[0] }
}

impl ParamType for crate::vect::vect2<f64> {
    const SIZE: usize = 2;
    const SUFFIXES: &'static [&'static str] = &[".x", ".y"];
    fn write_to32(&self, dst: &mut [f32]) { dst[0] = self.x as f32; dst[1] = self.y as f32; }
    fn read_from32(src: &[f32]) -> Self { Self::new(src[0] as f64, src[1] as f64) }
    fn write_to64(&self, dst: &mut [f64]) { dst[0] = self.x; dst[1] = self.y; }
    fn read_from64(src: &[f64]) -> Self { Self::new(src[0], src[1]) }
}

impl ParamType for crate::vect::vect2<f32> {
    const SIZE: usize = 2;
    const SUFFIXES: &'static [&'static str] = &[".x", ".y"];
    fn write_to32(&self, dst: &mut [f32]) { dst[0] = self.x; dst[1] = self.y; }
    fn read_from32(src: &[f32]) -> Self { Self::new(src[0], src[1]) }
    fn write_to64(&self, dst: &mut [f64]) { dst[0] = self.x as f64; dst[1] = self.y as f64; }
    fn read_from64(src: &[f64]) -> Self { Self::new(src[0] as f32, src[1] as f32) }
}

impl ParamType for crate::vect::vect3<f64> {
    const SIZE: usize = 3;
    const SUFFIXES: &'static [&'static str] = &[".x", ".y", ".z"];
    fn write_to32(&self, dst: &mut [f32]) { dst[0] = self.x as f32; dst[1] = self.y as f32; dst[2] = self.z as f32; }
    fn read_from32(src: &[f32]) -> Self { Self::new(src[0] as f64, src[1] as f64, src[2] as f64) }
    fn write_to64(&self, dst: &mut [f64]) { dst[0] = self.x; dst[1] = self.y; dst[2] = self.z; }
    fn read_from64(src: &[f64]) -> Self { Self::new(src[0], src[1], src[2]) }
}

impl ParamType for crate::vect::vect3<f32> {
    const SIZE: usize = 3;
    const SUFFIXES: &'static [&'static str] = &[".x", ".y", ".z"];
    fn write_to32(&self, dst: &mut [f32]) { dst[0] = self.x; dst[1] = self.y; dst[2] = self.z; }
    fn read_from32(src: &[f32]) -> Self { Self::new(src[0], src[1], src[2]) }
    fn write_to64(&self, dst: &mut [f64]) { dst[0] = self.x as f64; dst[1] = self.y as f64; dst[2] = self.z as f64; }
    fn read_from64(src: &[f64]) -> Self { Self::new(src[0] as f32, src[1] as f32, src[2] as f32) }
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
#[derive(serde::Serialize, serde::Deserialize)]
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
/// - `serialize_params{32,64}` -- flatten optimizable parameters into a vector
///   and assign indices.
/// - `deserialize_params{32,64}` -- write optimized values back into `Param::value`.
/// - `update{32,64}` -- copy a candidate parameter vector into working copies.
/// - `update_self` -- reset working copies to current `value` (and precompute
///   derived quantities like rotation matrices).
/// - `zero_blocks` / `accumulate_hessian*` -- zero and accumulate Hessian
///   blocks into dense, banded, COO, CSC, or indexed sparse formats.
///   Gradient is written directly into a global slice by each constraint,
///   not routed through these methods.
pub trait Model {
    fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>);
    fn deserialize_params32(&mut self, data: &[f32]);
    fn update32(&mut self,data: &[f32]);
    fn update_self(&mut self);

    fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>);
    fn deserialize_params64(&mut self, data: &[f64]);
    fn update64(&mut self, data: &[f64]);

    const PARAM_COUNT: u32 = 0;
    fn serialize_size(&self) -> u32 { 0 }
    fn param_symbols(_base: &str, _out: &mut std::vec::Vec<String>) {}

    fn zero_blocks(&mut self) {}

    // Fold accepted-step euler angle deltas into their reference rotations
    // and zero the delta entries in the parameter vector. A no-op for
    // everything except EulerAngleParam, which re-centers after every
    // accepted LM step (the property that avoids gimbal lock). Recurses
    // through the model tree exactly like update/serialize, so params at
    // any nesting depth are advanced.
    fn advance_params32(&mut self, _params: &mut [f32]) {}
    fn advance_params64(&mut self, _params: &mut [f64]) {}

    // Hessian-only accumulation: blocks hold only Hessian entries after the
    // refactor; the gradient is written directly by constraint evaluation
    // into the LM-provided `grad` slice, so there is nothing for these
    // methods to do with grad.
    fn accumulate_hessian32(&self, _hessian: &mut [f32]) {}
    fn accumulate_hessian64(&self, _hessian: &mut [f64]) {}
    fn accumulate_hessian_band32(&self, _band: &mut [f32], _kd: usize) -> Result<(), crate::simple_lm::BandError> { Ok(()) }
    fn accumulate_hessian_band64(&self, _band: &mut [f64], _kd: usize) -> Result<(), crate::simple_lm::BandError> { Ok(()) }
    fn accumulate_hessian_sparse32(&self, _coo: &mut crate::simple_lm::CooMatrix<f32>) {}
    fn accumulate_hessian_sparse64(&self, _coo: &mut crate::simple_lm::CooMatrix<f64>) {}
    fn accumulate_hessian_sparse_direct32(&self, _csc: &mut crate::simple_lm::CscMatrix<f32>) {}
    fn accumulate_hessian_sparse_direct64(&self, _csc: &mut crate::simple_lm::CscMatrix<f64>) {}
    fn accumulate_hessian_sparse_indexed32(&self, _vals: &mut [f32], _positions: &[usize], _cursor: &mut usize) {}
    fn accumulate_hessian_sparse_indexed64(&self, _vals: &mut [f64], _positions: &[usize], _cursor: &mut usize) {}
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
    /// Called after `update64`, before cost/constraint calculations.
    /// Use to compute derived state that constraints depend on.
    fn extended_update64(&mut self, _params: &[f64]) {}
    /// Called after `update32`, before cost/constraint calculations.
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
    fn extended_compute64(&mut self, _params: &[f64], _grad: &mut [f64]) {}
    /// Compute custom constraint residuals (f32).
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
    fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
        if self.optimize {
            self.index = data.len() as u32;
            let start = data.len();
            data.resize(start + T::SIZE, 0.0);
            self.value.write_to32(&mut data[start..start + T::SIZE]);
        } else {
            self.index = u32::MAX;
        }
    }

    fn deserialize_params32(&mut self, data: &[f32]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.value = T::read_from32(&data[i..i + T::SIZE]);
        }
    }

    fn update32(&mut self,data: &[f32]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.work = T::read_from32(&data[i..i + T::SIZE]);
        } else {
            self.work = self.value;
        }
    }

    fn update_self(&mut self) {
        self.work = self.value;
    }

    fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
        if self.optimize {
            self.index = data.len() as u32;
            let start = data.len();
            data.resize(start + T::SIZE, 0.0);
            self.value.write_to64(&mut data[start..start + T::SIZE]);
        } else {
            self.index = u32::MAX;
        }
    }

    fn deserialize_params64(&mut self, data: &[f64]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.value = T::read_from64(&data[i..i + T::SIZE]);
        }
    }

    fn update64(&mut self, data: &[f64]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.work = T::read_from64(&data[i..i + T::SIZE]);
        } else {
            self.work = self.value;
        }
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
// SimpleEulerAngleParam -- euler angles with precomputed sincos + rotation matrix
// ---------------------------------------------------------------------------

use crate::vect::vect3;
use crate::matrix::matrix3;

/// Euler angle parameter with precomputed sin/cos and rotation matrix.
///
/// Stores roll/pitch/yaw (x/y/z) as a `vect3<T>`. On each update the
/// framework precomputes `sincos` and the full 3x3 rotation matrix so
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
    #[doc(hidden)] pub sincos: (vect3<T>, vect3<T>),
    #[doc(hidden)] pub rotation_matrix: matrix3<T>,
}

impl<T: crate::utils::Float> Default for SimpleEulerAngleParam<T> {
    fn default() -> Self {
        Self {
            optimize: true,
            value: vect3::<T>::default(),
            work: vect3::<T>::default(),
            index: u32::MAX,
            sincos: (vect3::<T>::default(), vect3::<T>::default()),
            rotation_matrix: matrix3::<T>::identity(),
        }
    }
}

impl<T: crate::utils::Float> SimpleEulerAngleParam<T> {
    /// Create a new optimizable euler angle parameter with the given initial angles.
    pub fn new(value: vect3<T>) -> Self {
        Self { optimize: true, value, work: vect3::<T>::default(), index: u32::MAX,
               sincos: (vect3::<T>::default(), vect3::<T>::default()),
               rotation_matrix: matrix3::<T>::identity() }
    }
    /// Create a fixed (non-optimizable) euler angle parameter.
    pub fn fixed(value: vect3<T>) -> Self {
        Self { optimize: false, value, work: vect3::<T>::default(), index: u32::MAX,
               sincos: (vect3::<T>::default(), vect3::<T>::default()),
               rotation_matrix: matrix3::<T>::identity() }
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
    /// Precompute sincos and rotation matrix from current work value.
    #[doc(hidden)]
    pub fn __precompute(&mut self) {
        self.sincos = self.work.sincos();
        self.rotation_matrix = matrix3::<T>::rotation_from_euler_angles_sincos(
            self.sincos.0, self.sincos.1);
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
    fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
        if self.optimize {
            self.index = data.len() as u32;
            let start = data.len();
            data.resize(start + 3, 0.0);
            self.value.write_to32(&mut data[start..start + 3]);
        } else { self.index = u32::MAX; }
    }
    fn deserialize_params32(&mut self, data: &[f32]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.value = <vect3<T> as ParamType>::read_from32(&data[i..i + 3]);
        }
    }
    fn update32(&mut self, data: &[f32]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.work = <vect3<T> as ParamType>::read_from32(&data[i..i + 3]);
        } else { self.work = self.value; }
    }
    fn update_self(&mut self) {
        self.work = self.value;
        self.__precompute();
    }

    fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
        if self.optimize {
            self.index = data.len() as u32;
            let start = data.len();
            data.resize(start + 3, 0.0);
            self.value.write_to64(&mut data[start..start + 3]);
        } else { self.index = u32::MAX; }
    }
    fn deserialize_params64(&mut self, data: &[f64]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.value = <vect3<T> as ParamType>::read_from64(&data[i..i + 3]);
        }
    }
    fn update64(&mut self, data: &[f64]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.work = <vect3<T> as ParamType>::read_from64(&data[i..i + 3]);
        } else { self.work = self.value; }
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
    #[doc(hidden)] pub sincos: (vect3<T>, vect3<T>),
    #[doc(hidden)] pub rotation_matrix: matrix3<T>,
    #[doc(hidden)] pub delta_sincos: (vect3<T>, vect3<T>),
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
            sincos: (vect3::<T>::default(), vect3::<T>::default()),
            rotation_matrix: matrix3::<T>::identity(),
            delta_sincos: (vect3::<T>::default(), vect3::<T>::default()),
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
    /// Precompute composed rotation, sincos, work from current delta + ref_rotation.
    #[doc(hidden)]
    pub fn __precompute(&mut self) {
        self.delta_sincos = self.delta.sincos();
        let dea_rot = matrix3::<T>::rotation_from_euler_angles_sincos(
            self.delta_sincos.0, self.delta_sincos.1);
        self.rotation_matrix = self.ref_rotation * dea_rot;
        self.work = self.rotation_matrix.get_euler_angles();
        self.sincos = self.work.sincos();
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
    fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
        if self.optimize {
            self.ref_rotation = matrix3::<T>::rotation_from_euler_angles(self.value);
            self.index = data.len() as u32;
            data.push(0.0); data.push(0.0); data.push(0.0);
        } else { self.index = u32::MAX; }
    }
    fn deserialize_params32(&mut self, data: &[f32]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            let dea = <vect3<T> as ParamType>::read_from32(&data[i..i + 3]);
            self.ref_rotation = self.ref_rotation
                * matrix3::<T>::rotation_from_euler_angles(dea);
            self.value = self.ref_rotation.get_euler_angles();
            self.delta = vect3::<T>::default();
        }
    }
    fn update32(&mut self, data: &[f32]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.delta = <vect3<T> as ParamType>::read_from32(&data[i..i + 3]);
        } else { self.delta = vect3::<T>::default(); }
    }
    fn update_self(&mut self) {
        self.delta = vect3::<T>::default();
        self.__precompute();
    }

    fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
        if self.optimize {
            self.ref_rotation = matrix3::<T>::rotation_from_euler_angles(self.value);
            self.index = data.len() as u32;
            data.push(0.0); data.push(0.0); data.push(0.0);
        } else { self.index = u32::MAX; }
    }
    fn deserialize_params64(&mut self, data: &[f64]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            let dea = <vect3<T> as ParamType>::read_from64(&data[i..i + 3]);
            self.ref_rotation = self.ref_rotation
                * matrix3::<T>::rotation_from_euler_angles(dea);
            self.value = self.ref_rotation.get_euler_angles();
            self.delta = vect3::<T>::default();
        }
    }
    fn update64(&mut self, data: &[f64]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.delta = <vect3<T> as ParamType>::read_from64(&data[i..i + 3]);
        } else { self.delta = vect3::<T>::default(); }
    }

    const PARAM_COUNT: u32 = 3;
    fn serialize_size(&self) -> u32 { if self.optimize { 3 } else { 0 } }
    fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
        for suffix in <vect3<T> as ParamType>::SUFFIXES {
            out.push(format!("{}{}", base, suffix));
        }
    }

    fn advance_params32(&mut self, params: &mut [f32]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.advance();
            params[i] = 0.0; params[i + 1] = 0.0; params[i + 2] = 0.0;
        }
    }
    fn advance_params64(&mut self, params: &mut [f64]) {
        if self.index != u32::MAX {
            let i = self.index as usize;
            self.advance();
            params[i] = 0.0; params[i + 1] = 0.0; params[i + 2] = 0.0;
        }
    }
}

// ---------------------------------------------------------------------------
// No-op Model impls for leaf types
// ---------------------------------------------------------------------------

macro_rules! impl_model_noop {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Model for $ty {
                fn serialize_params32(&mut self, _data: &mut std::vec::Vec<f32>) {}
                fn deserialize_params32(&mut self, _data: &[f32]) {}
                fn update32(&mut self,_data: &[f32]) {}
                fn update_self(&mut self) {}
                fn serialize_params64(&mut self, _data: &mut std::vec::Vec<f64>) {}
                fn deserialize_params64(&mut self, _data: &[f64]) {}
                fn update64(&mut self, _data: &[f64]) {}
            }
        )*
    };
}

impl_model_noop!(f32, f64, u32, i32, bool, usize, String);

macro_rules! impl_model_noop_generic {
    ($($ty:ty),* $(,)?) => {
        $(
            impl Model for $ty {
                fn serialize_params32(&mut self, _data: &mut std::vec::Vec<f32>) {}
                fn deserialize_params32(&mut self, _data: &[f32]) {}
                fn update32(&mut self,_data: &[f32]) {}
                fn update_self(&mut self) {}
                fn serialize_params64(&mut self, _data: &mut std::vec::Vec<f64>) {}
                fn deserialize_params64(&mut self, _data: &[f64]) {}
                fn update64(&mut self, _data: &[f64]) {}
            }
        )*
    };
}

impl_model_noop_generic!(
    crate::vect::vect3f, crate::vect::vect3d,
    crate::vect::vect2f, crate::vect::vect2d,
    crate::matrix::matrix3f, crate::matrix::matrix3d,
    crate::matrix::matrix2f, crate::matrix::matrix2d,
    crate::quatern::quaternf, crate::quatern::quaternd,
);

impl<T> Model for crate::refs::Ref<T> {
    fn serialize_params32(&mut self, _data: &mut std::vec::Vec<f32>) {}
    fn deserialize_params32(&mut self, _data: &[f32]) {}
    fn update32(&mut self,_data: &[f32]) {}
    fn update_self(&mut self) {}
    fn serialize_params64(&mut self, _data: &mut std::vec::Vec<f64>) {}
    fn deserialize_params64(&mut self, _data: &[f64]) {}
    fn update64(&mut self, _data: &[f64]) {}
}

// ---------------------------------------------------------------------------
// Collection Model impls — iterate and recurse
// ---------------------------------------------------------------------------

macro_rules! impl_model_collection {
    ($ty:ty, $iter_mut:ident) => {
        impl<T: Model> Model for $ty {
            fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
                for item in self.$iter_mut() { item.serialize_params32(data); }
            }
            fn deserialize_params32(&mut self, data: &[f32]) {
                for item in self.$iter_mut() { item.deserialize_params32(data); }
            }
            fn update32(&mut self,data: &[f32]) {
                for item in self.$iter_mut() { item.update32(data); }
            }
            fn update_self(&mut self) {
                for item in self.$iter_mut() { item.update_self(); }
            }
            fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
                for item in self.$iter_mut() { item.serialize_params64(data); }
            }
            fn deserialize_params64(&mut self, data: &[f64]) {
                for item in self.$iter_mut() { item.deserialize_params64(data); }
            }
            fn update64(&mut self, data: &[f64]) {
                for item in self.$iter_mut() { item.update64(data); }
            }
            fn advance_params32(&mut self, params: &mut [f32]) {
                for item in self.$iter_mut() { item.advance_params32(params); }
            }
            fn advance_params64(&mut self, params: &mut [f64]) {
                for item in self.$iter_mut() { item.advance_params64(params); }
            }
            fn zero_blocks(&mut self) {
                for item in self.$iter_mut() { item.zero_blocks(); }
            }
            fn serialize_size(&self) -> u32 {
                self.iter().map(|item| item.serialize_size()).sum()
            }
            fn accumulate_hessian32(&self, hessian: &mut [f32]) {
                for item in self.iter() { item.accumulate_hessian32(hessian); }
            }
            fn accumulate_hessian64(&self, hessian: &mut [f64]) {
                for item in self.iter() { item.accumulate_hessian64(hessian); }
            }
            fn accumulate_hessian_band32(&self, band: &mut [f32], kd: usize) -> Result<(), crate::simple_lm::BandError> {
                for item in self.iter() { item.accumulate_hessian_band32(band, kd)?; }
                Ok(())
            }
            fn accumulate_hessian_band64(&self, band: &mut [f64], kd: usize) -> Result<(), crate::simple_lm::BandError> {
                for item in self.iter() { item.accumulate_hessian_band64(band, kd)?; }
                Ok(())
            }
            fn accumulate_hessian_sparse32(&self, coo: &mut crate::simple_lm::CooMatrix<f32>) {
                for item in self.iter() { item.accumulate_hessian_sparse32(coo); }
            }
            fn accumulate_hessian_sparse64(&self, coo: &mut crate::simple_lm::CooMatrix<f64>) {
                for item in self.iter() { item.accumulate_hessian_sparse64(coo); }
            }
            fn accumulate_hessian_sparse_direct32(&self, csc: &mut crate::simple_lm::CscMatrix<f32>) {
                for item in self.iter() { item.accumulate_hessian_sparse_direct32(csc); }
            }
            fn accumulate_hessian_sparse_direct64(&self, csc: &mut crate::simple_lm::CscMatrix<f64>) {
                for item in self.iter() { item.accumulate_hessian_sparse_direct64(csc); }
            }
            fn accumulate_hessian_sparse_indexed32(&self, vals: &mut [f32], positions: &[usize], cursor: &mut usize) {
                for item in self.iter() { item.accumulate_hessian_sparse_indexed32(vals, positions, cursor); }
            }
            fn accumulate_hessian_sparse_indexed64(&self, vals: &mut [f64], positions: &[usize], cursor: &mut usize) {
                for item in self.iter() { item.accumulate_hessian_sparse_indexed64(vals, positions, cursor); }
            }
        }
    };
}

impl_model_collection!(std::vec::Vec<T>, iter_mut);
impl_model_collection!(crate::refs::Vec<T>, iter_mut);
impl_model_collection!(crate::refs::Deque<T>, iter_mut);

// Arena needs a manual impl because iter()/iter_mut() return impl Iterator
impl<T: Model> Model for crate::refs::Arena<T> {
    fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
        for item in self.iter_mut() { item.serialize_params32(data); }
    }
    fn deserialize_params32(&mut self, data: &[f32]) {
        for item in self.iter_mut() { item.deserialize_params32(data); }
    }
    fn update32(&mut self,data: &[f32]) {
        for item in self.iter_mut() { item.update32(data); }
    }
    fn update_self(&mut self) {
        for item in self.iter_mut() { item.update_self(); }
    }
    fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
        for item in self.iter_mut() { item.serialize_params64(data); }
    }
    fn deserialize_params64(&mut self, data: &[f64]) {
        for item in self.iter_mut() { item.deserialize_params64(data); }
    }
    fn update64(&mut self, data: &[f64]) {
        for item in self.iter_mut() { item.update64(data); }
    }
    fn advance_params32(&mut self, params: &mut [f32]) {
        for item in self.iter_mut() { item.advance_params32(params); }
    }
    fn advance_params64(&mut self, params: &mut [f64]) {
        for item in self.iter_mut() { item.advance_params64(params); }
    }
    fn zero_blocks(&mut self) {
        for item in self.iter_mut() { item.zero_blocks(); }
    }
    fn serialize_size(&self) -> u32 {
        self.iter().map(|item| item.serialize_size()).sum()
    }
    fn accumulate_hessian32(&self, hessian: &mut [f32]) {
        for item in self.iter() { item.accumulate_hessian32(hessian); }
    }
    fn accumulate_hessian64(&self, hessian: &mut [f64]) {
        for item in self.iter() { item.accumulate_hessian64(hessian); }
    }
    fn accumulate_hessian_band32(&self, band: &mut [f32], kd: usize) -> Result<(), crate::simple_lm::BandError> {
        for item in self.iter() { item.accumulate_hessian_band32(band, kd)?; }
        Ok(())
    }
    fn accumulate_hessian_band64(&self, band: &mut [f64], kd: usize) -> Result<(), crate::simple_lm::BandError> {
        for item in self.iter() { item.accumulate_hessian_band64(band, kd)?; }
        Ok(())
    }
    fn accumulate_hessian_sparse32(&self, coo: &mut crate::simple_lm::CooMatrix<f32>) {
        for item in self.iter() { item.accumulate_hessian_sparse32(coo); }
    }
    fn accumulate_hessian_sparse64(&self, coo: &mut crate::simple_lm::CooMatrix<f64>) {
        for item in self.iter() { item.accumulate_hessian_sparse64(coo); }
    }
    fn accumulate_hessian_sparse_direct32(&self, csc: &mut crate::simple_lm::CscMatrix<f32>) {
        for item in self.iter() { item.accumulate_hessian_sparse_direct32(csc); }
    }
    fn accumulate_hessian_sparse_direct64(&self, csc: &mut crate::simple_lm::CscMatrix<f64>) {
        for item in self.iter() { item.accumulate_hessian_sparse_direct64(csc); }
    }
    fn accumulate_hessian_sparse_indexed32(&self, vals: &mut [f32], positions: &[usize], cursor: &mut usize) {
        for item in self.iter() { item.accumulate_hessian_sparse_indexed32(vals, positions, cursor); }
    }
    fn accumulate_hessian_sparse_indexed64(&self, vals: &mut [f64], positions: &[usize], cursor: &mut usize) {
        for item in self.iter() { item.accumulate_hessian_sparse_indexed64(vals, positions, cursor); }
    }
}

impl<T: Model> Model for Option<T> {
    fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
        if let Some(inner) = self { inner.serialize_params32(data); }
    }
    fn deserialize_params32(&mut self, data: &[f32]) {
        if let Some(inner) = self { inner.deserialize_params32(data); }
    }
    fn update32(&mut self,data: &[f32]) {
        if let Some(inner) = self { inner.update32(data); }
    }
    fn update_self(&mut self) {
        if let Some(inner) = self { inner.update_self(); }
    }
    fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
        if let Some(inner) = self { inner.serialize_params64(data); }
    }
    fn deserialize_params64(&mut self, data: &[f64]) {
        if let Some(inner) = self { inner.deserialize_params64(data); }
    }
    fn update64(&mut self, data: &[f64]) {
        if let Some(inner) = self { inner.update64(data); }
    }
    fn advance_params32(&mut self, params: &mut [f32]) {
        if let Some(inner) = self { inner.advance_params32(params); }
    }
    fn advance_params64(&mut self, params: &mut [f64]) {
        if let Some(inner) = self { inner.advance_params64(params); }
    }
    fn serialize_size(&self) -> u32 {
        if let Some(inner) = self { inner.serialize_size() } else { 0 }
    }
    fn zero_blocks(&mut self) {
        if let Some(inner) = self { inner.zero_blocks(); }
    }
    fn accumulate_hessian32(&self, hessian: &mut [f32]) {
        if let Some(inner) = self { inner.accumulate_hessian32(hessian); }
    }
    fn accumulate_hessian64(&self, hessian: &mut [f64]) {
        if let Some(inner) = self { inner.accumulate_hessian64(hessian); }
    }
    fn accumulate_hessian_band32(&self, band: &mut [f32], kd: usize) -> Result<(), crate::simple_lm::BandError> {
        if let Some(inner) = self { inner.accumulate_hessian_band32(band, kd)?; }
        Ok(())
    }
    fn accumulate_hessian_band64(&self, band: &mut [f64], kd: usize) -> Result<(), crate::simple_lm::BandError> {
        if let Some(inner) = self { inner.accumulate_hessian_band64(band, kd)?; }
        Ok(())
    }
    fn accumulate_hessian_sparse32(&self, coo: &mut crate::simple_lm::CooMatrix<f32>) {
        if let Some(inner) = self { inner.accumulate_hessian_sparse32(coo); }
    }
    fn accumulate_hessian_sparse64(&self, coo: &mut crate::simple_lm::CooMatrix<f64>) {
        if let Some(inner) = self { inner.accumulate_hessian_sparse64(coo); }
    }
    fn accumulate_hessian_sparse_direct32(&self, csc: &mut crate::simple_lm::CscMatrix<f32>) {
        if let Some(inner) = self { inner.accumulate_hessian_sparse_direct32(csc); }
    }
    fn accumulate_hessian_sparse_direct64(&self, csc: &mut crate::simple_lm::CscMatrix<f64>) {
        if let Some(inner) = self { inner.accumulate_hessian_sparse_direct64(csc); }
    }
    fn accumulate_hessian_sparse_indexed32(&self, vals: &mut [f32], positions: &[usize], cursor: &mut usize) {
        if let Some(inner) = self { inner.accumulate_hessian_sparse_indexed32(vals, positions, cursor); }
    }
    fn accumulate_hessian_sparse_indexed64(&self, vals: &mut [f64], positions: &[usize], cursor: &mut usize) {
        if let Some(inner) = self { inner.accumulate_hessian_sparse_indexed64(vals, positions, cursor); }
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


/// Hessian block for a single model type.
///
/// Accumulates the upper triangle of the Gauss-Newton Hessian approximation
/// (2·dr·dr^T) from constraint residuals involving one model's parameters.
/// Gradient entries (2·r·dr) are written directly to the global gradient
/// vector by `add_residual` — no per-block grad buffer.
/// `N` equals `A::PARAM_COUNT`. `T` is the float type (f32 or f64, default f64).
///
/// Created by generated constraint code; users rarely construct these manually.
pub struct SelfBlock<A: Model, const N: usize, T: crate::utils::Float = f64> {
    indices: [u32; N],
    hessian: std::vec::Vec<T>,
    _marker: std::marker::PhantomData<(A, T)>,
}

impl<A: Model, const N: usize, T: crate::utils::Float> Default for SelfBlock<A, N, T> {
    fn default() -> Self { Self::new() }
}

impl<A: Model, const N: usize, T: crate::utils::Float> SelfBlock<A, N, T> {
    /// Create a new zeroed block.
    pub fn new() -> Self {
        SelfBlock {
            indices: [u32::MAX; N],
            hessian: vec![T::zero(); N * (N + 1) / 2],
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the global parameter indices for this block.
    pub fn set_indices(&mut self, indices: &[u32; N]) {
        self.indices = *indices;
    }

    /// Reset Hessian to zero. Gradient lives in the global grad vector,
    /// which LM zeros before each compute pass.
    pub fn zero(&mut self) {
        self.hessian.fill(T::zero());
    }

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
            if gi != u32::MAX {
                grad[gi as usize] += two * r * dr[i];
            }
            for j in i..N {
                self.hessian[tri_idx(N, i, j)] += two * dr[i] * dr[j];
            }
        }
    }

    /// Accumulate this block's Hessian into the full dense symmetric hessian.
    pub fn accumulate_hessian(&self, hessian: &mut [T]) {
        let n_total = (hessian.len() as f64).sqrt() as usize;
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            let gi = gi as usize;
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let gj = gj as usize;
                let val = self.hessian[tri_idx(N, i, j)];
                hessian[gi * n_total + gj] += val;
                if gi != gj {
                    hessian[gj * n_total + gi] += val;
                }
            }
        }
    }

    /// Accumulate into upper-band format (column-major, (kd+1)*n).
    /// Returns Err if any element exceeds bandwidth kd.
    pub fn accumulate_hessian_band(&self, band: &mut [T], kd: usize)
        -> Result<(), crate::simple_lm::BandError>
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
                    return Err(crate::simple_lm::BandError { row: lo, col: hi, kd });
                }
                let val = self.hessian[tri_idx(N, i, j)];
                band[(kd + lo - hi) + hi * ldab] += val;
            }
        }
        Ok(())
    }

    /// Accumulate into COO (triplet) sparse format. Upper triangle only.
    pub fn accumulate_hessian_sparse(&self, coo: &mut crate::simple_lm::CooMatrix<T>) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = self.hessian[tri_idx(N, i, j)];
                coo.push(lo, hi, val);
            }
        }
    }

    /// Accumulate directly into CSC vals array using position lookup.
    pub fn accumulate_hessian_sparse_direct(&self, csc: &mut crate::simple_lm::CscMatrix<T>) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = self.hessian[tri_idx(N, i, j)];
                if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                    csc.vals[pos] += val;
                }
            }
        }
    }

    /// Accumulate into CSC vals using precomputed position list.
    /// `cursor` advances through `positions` in lockstep with block traversal.
    pub fn accumulate_hessian_sparse_indexed(&self, vals: &mut [T], positions: &[usize], cursor: &mut usize) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let val = self.hessian[tri_idx(N, i, j)];
                vals[positions[*cursor]] += val;
                *cursor += 1;
            }
        }
    }
}

/// Hessian block coupling two model types — stores ONLY the rectangular A×B
/// cross Hessian pairs. A's gradient and A-A diagonal live in A's own
/// [`SelfBlock`]; same for B. This matches the refactor where every
/// params-having Model owns a SelfBlock<Self>, and cross blocks carry only
/// the pieces that don't fit in a per-entity block.
///
/// `NA = A::PARAM_COUNT`, `NB = B::PARAM_COUNT`. Internal Hessian storage
/// is NA×NB row-major (one entry per cross pair). No grad, no A-A, no B-B.
/// `T` is the float type (f32 or f64, default f64).
pub struct CrossBlock<A: Model, B: Model, const NA: usize, const NB: usize, T: crate::utils::Float = f64> {
    indices_a: [u32; NA],
    indices_b: [u32; NB],
    cross_hessian: std::vec::Vec<T>,    // NA*NB row-major
    _marker: std::marker::PhantomData<(A, B, T)>,
}

impl<A: Model, B: Model, const NA: usize, const NB: usize, T: crate::utils::Float> Default for CrossBlock<A, B, NA, NB, T> {
    fn default() -> Self { Self::new() }
}

impl<A: Model, B: Model, const NA: usize, const NB: usize, T: crate::utils::Float> CrossBlock<A, B, NA, NB, T> {
    /// Create a new zeroed cross-block.
    pub fn new() -> Self {
        CrossBlock {
            indices_a: [u32::MAX; NA],
            indices_b: [u32::MAX; NB],
            cross_hessian: vec![T::zero(); NA * NB],
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of parameters belonging to model A.
    pub fn na(&self) -> usize { NA }
    /// Return the number of parameters belonging to model B.
    pub fn nb(&self) -> usize { NB }

    /// Set the global parameter indices.
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

    /// Accumulate cross pairs into the full dense symmetric hessian.
    pub fn accumulate_hessian(&self, hessian: &mut [T]) {
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
                let val = self.cross_hessian[row + j];
                if gi == gj { continue; }
                hessian[gi * n_total + gj] += val;
                hessian[gj * n_total + gi] += val;
            }
        }
    }

    /// Accumulate into upper-band format (column-major, (kd+1)*n).
    pub fn accumulate_hessian_band(&self, band: &mut [T], kd: usize)
        -> Result<(), crate::simple_lm::BandError>
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
                if gi == gj { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                if hi - lo > kd {
                    return Err(crate::simple_lm::BandError { row: lo, col: hi, kd });
                }
                let val = self.cross_hessian[row + j];
                band[(kd + lo - hi) + hi * ldab] += val;
            }
        }
        Ok(())
    }

    /// Accumulate into COO sparse format. Upper triangle only.
    pub fn accumulate_hessian_sparse(&self, coo: &mut crate::simple_lm::CooMatrix<T>) {
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let row = i * NB;
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                if gi == gj { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = self.cross_hessian[row + j];
                coo.push(lo, hi, val);
            }
        }
    }

    /// Accumulate directly into CSC vals via position lookup.
    pub fn accumulate_hessian_sparse_direct(&self, csc: &mut crate::simple_lm::CscMatrix<T>) {
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let row = i * NB;
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                if gi == gj { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = self.cross_hessian[row + j];
                if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                    csc.vals[pos] += val;
                }
            }
        }
    }

    /// Accumulate into CSC vals via precomputed position list.
    pub fn accumulate_hessian_sparse_indexed(&self, vals: &mut [T], positions: &[usize], cursor: &mut usize) {
        for i in 0..NA {
            let gi = self.indices_a[i];
            if gi == u32::MAX { continue; }
            let row = i * NB;
            for j in 0..NB {
                let gj = self.indices_b[j];
                if gj == u32::MAX { continue; }
                if gi == gj { continue; }
                let val = self.cross_hessian[row + j];
                vals[positions[*cursor]] += val;
                *cursor += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TripletBlock -- sparse Hessian/gradient accumulation for N-ary constraints
// ---------------------------------------------------------------------------

/// Sparse cross-only Hessian accumulation block for N-ary constraints.
///
/// After the refactor: TripletBlock stores ONLY across-entity Hessian pairs
/// (pairs where the two params belong to different entity spans). The
/// within-entity H[A,A] / H[B,B] / ... diagonals live in each entity's
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
    pub fn new() -> Self {
        TripletBlock { hessian: std::vec::Vec::new() }
    }

    /// Reset to empty (called at start of each optimization step).
    pub fn zero(&mut self) {
        self.hessian.clear();
    }

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
                self.hessian.push((lo, hi, two * dr[i] * dr[j]));
            }
        }
    }

    /// Accumulate Hessian pairs into the full dense symmetric hessian.
    pub fn accumulate_hessian(&self, hessian: &mut [T]) {
        let n_total = (hessian.len() as f64).sqrt() as usize;
        for &(i, j, v) in &self.hessian {
            let (i, j) = (i as usize, j as usize);
            hessian[i * n_total + j] += v;
            if i != j {
                hessian[j * n_total + i] += v;
            }
        }
    }

    /// Accumulate into upper-band format (LAPACK layout: A[r, c] at
    /// band[(kd + r - c) + c * ldab], matching SelfBlock/CrossBlock and
    /// the band solvers).
    pub fn accumulate_hessian_band(&self, band: &mut [T], kd: usize)
        -> Result<(), crate::simple_lm::BandError>
    {
        let ldab = kd + 1;
        for &(row, col, v) in &self.hessian {
            let (r, c) = (row as usize, col as usize);
            if c < r || c - r > kd {
                return Err(crate::simple_lm::BandError { row: r, col: c, kd });
            }
            band[(kd + r - c) + c * ldab] += v;
        }
        Ok(())
    }

    /// Accumulate into COO sparse format. Upper triangle only.
    pub fn accumulate_hessian_sparse(&self, coo: &mut crate::simple_lm::CooMatrix<T>) {
        for &(i, j, v) in &self.hessian {
            coo.push(i, j, v);
        }
    }

    /// Accumulate directly into CSC vals via position lookup.
    pub fn accumulate_hessian_sparse_direct(&self, csc: &mut crate::simple_lm::CscMatrix<T>) {
        for &(row, col, v) in &self.hessian {
            if let Some(pos) = csc.find_pos(row as usize, col as usize) {
                csc.vals[pos] += v;
            }
        }
    }

    /// Accumulate into CSC vals via precomputed position list.
    pub fn accumulate_hessian_sparse_indexed(&self, vals: &mut [T], positions: &[usize], cursor: &mut usize) {
        for &(_, _, v) in &self.hessian {
            vals[positions[*cursor]] += v;
            *cursor += 1;
        }
    }
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
    /// Only active (optimizable) parameters included.
    pub entries: std::vec::Vec<(u32, T)>,
}

impl<T: crate::utils::Float> Jacobian<T> {
    /// Number of residuals (rows).
    pub fn num_residuals(&self) -> usize { self.rows.len() }

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
    /// ([`JacobianRow::label`]).
    fn calc_jacobian(&mut self, params: &[T]) -> Jacobian<T>;

    /// Return the per-label squared-residual total (`sum r^2` grouped
    /// by [`JacobianRow::label`]). Useful for seeing which constraint
    /// group is contributing how much cost at a given parameter point.
    ///
    /// Default implementation derives from `calc_jacobian`. Roots can
    /// override to skip derivative computation if they only need the
    /// cost split, but the default is usually fine.
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

impl ModelSym for bool {
    type Sym = E;
    fn sym(base: &str) -> E { arael_sym::symbol(base) }
}

impl ModelSym for u32 {
    type Sym = E;
    fn sym(base: &str) -> E { arael_sym::symbol(base) }
}

impl ModelSym for f32 {
    type Sym = E;
    fn sym(base: &str) -> E { arael_sym::symbol(base) }
}

impl ModelSym for f64 {
    type Sym = E;
    fn sym(base: &str) -> E { arael_sym::symbol(base) }
}

impl ModelSym for String {
    type Sym = E;
    fn sym(base: &str) -> E { arael_sym::symbol(base) }
}

impl ModelSym for crate::vect::vect3f {
    type Sym = crate::vect::vect3sym;
    fn sym(base: &str) -> Self::Sym { crate::vect::vect3sym::new(base) }
}

impl ModelSym for crate::vect::vect3d {
    type Sym = crate::vect::vect3sym;
    fn sym(base: &str) -> Self::Sym { crate::vect::vect3sym::new(base) }
}

impl ModelSym for crate::vect::vect2f {
    type Sym = crate::vect::vect2sym;
    fn sym(base: &str) -> Self::Sym { crate::vect::vect2sym::new(base) }
}

impl ModelSym for crate::vect::vect2d {
    type Sym = crate::vect::vect2sym;
    fn sym(base: &str) -> Self::Sym { crate::vect::vect2sym::new(base) }
}

impl ModelSym for crate::matrix::matrix3f {
    type Sym = crate::matrix::matrix3sym;
    fn sym(base: &str) -> Self::Sym { crate::matrix::matrix3sym::new(base) }
}

impl ModelSym for crate::matrix::matrix3d {
    type Sym = crate::matrix::matrix3sym;
    fn sym(base: &str) -> Self::Sym { crate::matrix::matrix3sym::new(base) }
}

impl ModelSym for crate::matrix::matrix2f {
    type Sym = crate::matrix::matrix2sym;
    fn sym(base: &str) -> Self::Sym { crate::matrix::matrix2sym::new(base) }
}

impl ModelSym for crate::matrix::matrix2d {
    type Sym = crate::matrix::matrix2sym;
    fn sym(base: &str) -> Self::Sym { crate::matrix::matrix2sym::new(base) }
}

impl ModelSym for crate::quatern::quaternf {
    type Sym = crate::quatern::quaternsym;
    fn sym(base: &str) -> Self::Sym { crate::quatern::quaternsym::new(base) }
}

impl ModelSym for crate::quatern::quaternd {
    type Sym = crate::quatern::quaternsym;
    fn sym(base: &str) -> Self::Sym { crate::quatern::quaternsym::new(base) }
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

        let mut data = Vec::new();
        a.serialize_params32(&mut data);
        b.serialize_params32(&mut data);
        c.serialize_params32(&mut data);

        assert_eq!(data, vec![3.0, 7.0]);
        assert_eq!(a.index, 0);
        assert_eq!(b.index, 1);
        assert_eq!(c.index, u32::MAX);

        // modify and deserialize
        data[0] = 10.0;
        data[1] = 20.0;
        a.deserialize_params32(&data);
        b.deserialize_params32(&data);
        c.deserialize_params32(&data);
        assert_eq!(a.value, 10.0);
        assert_eq!(b.value, 20.0);
        assert_eq!(c.value, 99.0); // unchanged — fixed
    }

    #[test]
    fn test_param_vect3f_serialize() {
        let mut p = Param::new(vect3f::new(1.0, 2.0, 3.0));
        let mut data = Vec::new();
        p.serialize_params32(&mut data);
        assert_eq!(data, vec![1.0, 2.0, 3.0]);
        assert_eq!(p.index, 0);
    }

    #[test]
    fn test_param_update() {
        let mut p = Param::new(5.0f32);
        let mut data = Vec::new();
        p.serialize_params32(&mut data);
        data[0] = 42.0;

        p.update32(&data);
        assert_eq!(p.work(), 42.0);
        assert_eq!(p.value, 5.0); // value unchanged

        p.update_self();
        assert_eq!(p.work(), 5.0); // work reset to value
    }

    #[test]
    fn test_param_fixed_update() {
        let mut p = Param::fixed(5.0f32);
        let mut data = Vec::new();
        p.serialize_params32(&mut data);
        assert!(data.is_empty()); // fixed param not serialized

        p.update32(&data);
        assert_eq!(p.work(), 5.0); // gets value since not optimized
    }

    #[test]
    fn test_param_vect2f() {
        let mut p = Param::new(vect2f::new(1.0, 2.0));
        let mut data = Vec::new();
        p.serialize_params32(&mut data);
        assert_eq!(data, vec![1.0, 2.0]);

        data[0] = 10.0;
        data[1] = 20.0;
        p.update32(&data);
        assert_eq!(p.work().x, 10.0);
        assert_eq!(p.work().y, 20.0);
    }

    #[test]
    fn test_param_f32_serialize64_roundtrip() {
        let mut a = Param::new(3.0f32);
        let mut b = Param::new(7.0f32);
        let mut c = Param::fixed(99.0f32);

        let mut data: std::vec::Vec<f64> = Vec::new();
        a.serialize_params64(&mut data);
        b.serialize_params64(&mut data);
        c.serialize_params64(&mut data);

        assert_eq!(data, vec![3.0f64, 7.0]);
        assert_eq!(a.index, 0);
        assert_eq!(b.index, 1);
        assert_eq!(c.index, u32::MAX);

        // modify and deserialize64
        data[0] = 10.0;
        data[1] = 20.0;
        a.deserialize_params64(&data);
        b.deserialize_params64(&data);
        c.deserialize_params64(&data);
        assert_eq!(a.value, 10.0f32);
        assert_eq!(b.value, 20.0f32);
        assert_eq!(c.value, 99.0f32);
    }

    #[test]
    fn test_param_vect3f_serialize64_roundtrip() {
        let mut p = Param::new(vect3f::new(1.0, 2.0, 3.0));
        let mut data: std::vec::Vec<f64> = Vec::new();
        p.serialize_params64(&mut data);
        assert_eq!(data, vec![1.0f64, 2.0, 3.0]);

        data[0] = 10.5;
        data[1] = 20.5;
        data[2] = 30.5;
        p.update64(&data);
        assert_eq!(p.work().x, 10.5f32);
        assert_eq!(p.work().y, 20.5f32);
        assert_eq!(p.work().z, 30.5f32);
    }

    #[test]
    fn test_param_fixed_update64() {
        let mut p = Param::fixed(5.0f32);
        let mut data: std::vec::Vec<f64> = Vec::new();
        p.serialize_params64(&mut data);
        assert!(data.is_empty());

        p.update64(&data);
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
        let mut data = Vec::new();
        v.serialize_params32(&mut data);
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
        let mut blk: SelfBlock<Param<f64>, 3, f64> = SelfBlock::new();
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
        let mut blk: CrossBlock<Param<f64>, Param<f64>, 2, 2, f64> = CrossBlock::new();
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
    fn tripletblock_band_matches_selfblock_band() {
        // The same residual pushed through both block types must produce
        // byte-identical gradient and band arrays.
        let n = 4;
        let kd = 3;
        let r = 0.9;
        let dr = [1.0, -2.0, 0.5];
        let idx = [0u32, 1, 3];

        let mut g_self = vec![0.0; n];
        let mut sb: SelfBlock<Param<f64>, 3, f64> = SelfBlock::new();
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
