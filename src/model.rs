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
    pub optimize: bool,
    pub value: T,
    #[serde(skip)]
    work: T,
    #[serde(skip)]
    index: u32,
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
/// - `serialize_params{32,64}` -- flatten optimizable parameters into a vector
///   and assign indices.
/// - `deserialize_params{32,64}` -- write optimized values back into `Param::value`.
/// - `update{32,64}` -- copy a candidate parameter vector into working copies.
/// - `update_self` -- reset working copies to current `value` (and precompute
///   derived quantities like rotation matrices).
/// - `zero_blocks` / `accumulate_blocks*` -- zero and accumulate Hessian blocks
///   into dense, banded, COO, CSC, or indexed sparse formats.
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
    fn accumulate_blocks32(&self, _grad: &mut [f32], _hessian: &mut [f32]) {}
    fn accumulate_blocks64(&self, _grad: &mut [f64], _hessian: &mut [f64]) {}
    fn accumulate_blocks_band32(&self, _grad: &mut [f32], _band: &mut [f32], _kd: usize) -> Result<(), crate::simple_lm::BandError> { Ok(()) }
    fn accumulate_blocks_band64(&self, _grad: &mut [f64], _band: &mut [f64], _kd: usize) -> Result<(), crate::simple_lm::BandError> { Ok(()) }
    fn accumulate_blocks_sparse32(&self, _grad: &mut [f32], _coo: &mut crate::simple_lm::CooMatrix<f32>) {}
    fn accumulate_blocks_sparse64(&self, _grad: &mut [f64], _coo: &mut crate::simple_lm::CooMatrix<f64>) {}
    fn accumulate_blocks_sparse_direct32(&self, _grad: &mut [f32], _csc: &mut crate::simple_lm::CscMatrix<f32>) {}
    fn accumulate_blocks_sparse_direct64(&self, _grad: &mut [f64], _csc: &mut crate::simple_lm::CscMatrix<f64>) {}
    fn accumulate_blocks_sparse_indexed32(&self, _grad: &mut [f32], _vals: &mut [f32], _positions: &[usize], _cursor: &mut usize) {}
    fn accumulate_blocks_sparse_indexed64(&self, _grad: &mut [f64], _vals: &mut [f64], _positions: &[usize], _cursor: &mut usize) {}
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

impl_model_noop!(f32, f64, u32, i32, bool, usize);

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
            fn zero_blocks(&mut self) {
                for item in self.$iter_mut() { item.zero_blocks(); }
            }
            fn serialize_size(&self) -> u32 {
                self.iter().map(|item| item.serialize_size()).sum()
            }
            fn accumulate_blocks32(&self, grad: &mut [f32], hessian: &mut [f32]) {
                for item in self.iter() { item.accumulate_blocks32(grad, hessian); }
            }
            fn accumulate_blocks64(&self, grad: &mut [f64], hessian: &mut [f64]) {
                for item in self.iter() { item.accumulate_blocks64(grad, hessian); }
            }
            fn accumulate_blocks_band32(&self, grad: &mut [f32], band: &mut [f32], kd: usize) -> Result<(), crate::simple_lm::BandError> {
                for item in self.iter() { item.accumulate_blocks_band32(grad, band, kd)?; }
                Ok(())
            }
            fn accumulate_blocks_band64(&self, grad: &mut [f64], band: &mut [f64], kd: usize) -> Result<(), crate::simple_lm::BandError> {
                for item in self.iter() { item.accumulate_blocks_band64(grad, band, kd)?; }
                Ok(())
            }
            fn accumulate_blocks_sparse32(&self, grad: &mut [f32], coo: &mut crate::simple_lm::CooMatrix<f32>) {
                for item in self.iter() { item.accumulate_blocks_sparse32(grad, coo); }
            }
            fn accumulate_blocks_sparse64(&self, grad: &mut [f64], coo: &mut crate::simple_lm::CooMatrix<f64>) {
                for item in self.iter() { item.accumulate_blocks_sparse64(grad, coo); }
            }
            fn accumulate_blocks_sparse_direct32(&self, grad: &mut [f32], csc: &mut crate::simple_lm::CscMatrix<f32>) {
                for item in self.iter() { item.accumulate_blocks_sparse_direct32(grad, csc); }
            }
            fn accumulate_blocks_sparse_direct64(&self, grad: &mut [f64], csc: &mut crate::simple_lm::CscMatrix<f64>) {
                for item in self.iter() { item.accumulate_blocks_sparse_direct64(grad, csc); }
            }
            fn accumulate_blocks_sparse_indexed32(&self, grad: &mut [f32], vals: &mut [f32], positions: &[usize], cursor: &mut usize) {
                for item in self.iter() { item.accumulate_blocks_sparse_indexed32(grad, vals, positions, cursor); }
            }
            fn accumulate_blocks_sparse_indexed64(&self, grad: &mut [f64], vals: &mut [f64], positions: &[usize], cursor: &mut usize) {
                for item in self.iter() { item.accumulate_blocks_sparse_indexed64(grad, vals, positions, cursor); }
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
    fn zero_blocks(&mut self) {
        for item in self.iter_mut() { item.zero_blocks(); }
    }
    fn serialize_size(&self) -> u32 {
        self.iter().map(|item| item.serialize_size()).sum()
    }
    fn accumulate_blocks32(&self, grad: &mut [f32], hessian: &mut [f32]) {
        for item in self.iter() { item.accumulate_blocks32(grad, hessian); }
    }
    fn accumulate_blocks64(&self, grad: &mut [f64], hessian: &mut [f64]) {
        for item in self.iter() { item.accumulate_blocks64(grad, hessian); }
    }
    fn accumulate_blocks_band32(&self, grad: &mut [f32], band: &mut [f32], kd: usize) -> Result<(), crate::simple_lm::BandError> {
        for item in self.iter() { item.accumulate_blocks_band32(grad, band, kd)?; }
        Ok(())
    }
    fn accumulate_blocks_band64(&self, grad: &mut [f64], band: &mut [f64], kd: usize) -> Result<(), crate::simple_lm::BandError> {
        for item in self.iter() { item.accumulate_blocks_band64(grad, band, kd)?; }
        Ok(())
    }
    fn accumulate_blocks_sparse32(&self, grad: &mut [f32], coo: &mut crate::simple_lm::CooMatrix<f32>) {
        for item in self.iter() { item.accumulate_blocks_sparse32(grad, coo); }
    }
    fn accumulate_blocks_sparse64(&self, grad: &mut [f64], coo: &mut crate::simple_lm::CooMatrix<f64>) {
        for item in self.iter() { item.accumulate_blocks_sparse64(grad, coo); }
    }
    fn accumulate_blocks_sparse_direct32(&self, grad: &mut [f32], csc: &mut crate::simple_lm::CscMatrix<f32>) {
        for item in self.iter() { item.accumulate_blocks_sparse_direct32(grad, csc); }
    }
    fn accumulate_blocks_sparse_direct64(&self, grad: &mut [f64], csc: &mut crate::simple_lm::CscMatrix<f64>) {
        for item in self.iter() { item.accumulate_blocks_sparse_direct64(grad, csc); }
    }
    fn accumulate_blocks_sparse_indexed32(&self, grad: &mut [f32], vals: &mut [f32], positions: &[usize], cursor: &mut usize) {
        for item in self.iter() { item.accumulate_blocks_sparse_indexed32(grad, vals, positions, cursor); }
    }
    fn accumulate_blocks_sparse_indexed64(&self, grad: &mut [f64], vals: &mut [f64], positions: &[usize], cursor: &mut usize) {
        for item in self.iter() { item.accumulate_blocks_sparse_indexed64(grad, vals, positions, cursor); }
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
    fn serialize_size(&self) -> u32 {
        if let Some(inner) = self { inner.serialize_size() } else { 0 }
    }
    fn zero_blocks(&mut self) {
        if let Some(inner) = self { inner.zero_blocks(); }
    }
    fn accumulate_blocks32(&self, grad: &mut [f32], hessian: &mut [f32]) {
        if let Some(inner) = self { inner.accumulate_blocks32(grad, hessian); }
    }
    fn accumulate_blocks64(&self, grad: &mut [f64], hessian: &mut [f64]) {
        if let Some(inner) = self { inner.accumulate_blocks64(grad, hessian); }
    }
    fn accumulate_blocks_band32(&self, grad: &mut [f32], band: &mut [f32], kd: usize) -> Result<(), crate::simple_lm::BandError> {
        if let Some(inner) = self { inner.accumulate_blocks_band32(grad, band, kd)?; }
        Ok(())
    }
    fn accumulate_blocks_band64(&self, grad: &mut [f64], band: &mut [f64], kd: usize) -> Result<(), crate::simple_lm::BandError> {
        if let Some(inner) = self { inner.accumulate_blocks_band64(grad, band, kd)?; }
        Ok(())
    }
    fn accumulate_blocks_sparse32(&self, grad: &mut [f32], coo: &mut crate::simple_lm::CooMatrix<f32>) {
        if let Some(inner) = self { inner.accumulate_blocks_sparse32(grad, coo); }
    }
    fn accumulate_blocks_sparse64(&self, grad: &mut [f64], coo: &mut crate::simple_lm::CooMatrix<f64>) {
        if let Some(inner) = self { inner.accumulate_blocks_sparse64(grad, coo); }
    }
    fn accumulate_blocks_sparse_direct32(&self, grad: &mut [f32], csc: &mut crate::simple_lm::CscMatrix<f32>) {
        if let Some(inner) = self { inner.accumulate_blocks_sparse_direct32(grad, csc); }
    }
    fn accumulate_blocks_sparse_direct64(&self, grad: &mut [f64], csc: &mut crate::simple_lm::CscMatrix<f64>) {
        if let Some(inner) = self { inner.accumulate_blocks_sparse_direct64(grad, csc); }
    }
    fn accumulate_blocks_sparse_indexed32(&self, grad: &mut [f32], vals: &mut [f32], positions: &[usize], cursor: &mut usize) {
        if let Some(inner) = self { inner.accumulate_blocks_sparse_indexed32(grad, vals, positions, cursor); }
    }
    fn accumulate_blocks_sparse_indexed64(&self, grad: &mut [f64], vals: &mut [f64], positions: &[usize], cursor: &mut usize) {
        if let Some(inner) = self { inner.accumulate_blocks_sparse_indexed64(grad, vals, positions, cursor); }
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
/// Accumulates gradient and the upper triangle of the Gauss-Newton Hessian
/// approximation from constraint residuals involving one model's parameters.
/// `N` equals `A::PARAM_COUNT`. `T` is the float type (f32 or f64, default f64).
///
/// Created by generated constraint code; users rarely construct these manually.
pub struct SelfBlock<A: Model, const N: usize, T: crate::utils::Float = f64> {
    indices: [u32; N],
    grad: [T; N],
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
            grad: std::array::from_fn(|_| T::zero()),
            hessian: vec![T::zero(); N * (N + 1) / 2],
            _marker: std::marker::PhantomData,
        }
    }

    /// Set the global parameter indices for this block.
    pub fn set_indices(&mut self, indices: &[u32; N]) {
        self.indices = *indices;
    }

    /// Reset gradient and hessian to zero.
    pub fn zero(&mut self) {
        self.grad = std::array::from_fn(|_| T::zero());
        self.hessian.fill(T::zero());
    }

    /// Return true if any parameter in this block is being optimized.
    pub fn is_active(&self) -> bool {
        self.indices.iter().any(|&i| i != u32::MAX)
    }

    /// Add one residual's contribution: accumulates 2*r*dr into gradient and 2*dr*dr^T into hessian.
    pub fn add_residual(&mut self, r: T, dr: &[T; N]) {
        let two = T::two();
        for i in 0..N {
            self.grad[i] += two * r * dr[i];
            for j in i..N {
                self.hessian[tri_idx(N, i, j)] += two * dr[i] * dr[j];
            }
        }
    }

    /// Accumulate this block into the full dense gradient and symmetric hessian.
    pub fn accumulate(&self, grad: &mut [T], hessian: &mut [T]) {
        let n_total = grad.len();
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            let gi = gi as usize;
            grad[gi] += self.grad[i];
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
    pub fn accumulate_band(&self, grad: &mut [T], band: &mut [T], kd: usize)
        -> Result<(), crate::simple_lm::BandError>
    {
        let ldab = kd + 1;
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            let gi = gi as usize;
            grad[gi] += self.grad[i];
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
    pub fn accumulate_sparse(&self, grad: &mut [T], coo: &mut crate::simple_lm::CooMatrix<T>) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            grad[gi as usize] += self.grad[i];
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
    pub fn accumulate_sparse_direct(&self, grad: &mut [T], csc: &mut crate::simple_lm::CscMatrix<T>) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            grad[gi as usize] += self.grad[i];
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = self.hessian[tri_idx(N, i, j)];
                if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                    csc.vals[pos] = csc.vals[pos] + val;
                }
            }
        }
    }

    /// Accumulate into CSC vals using precomputed position list.
    /// `cursor` advances through `positions` in lockstep with block traversal.
    pub fn accumulate_sparse_indexed(&self, grad: &mut [T], vals: &mut [T], positions: &[usize], cursor: &mut usize) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            grad[gi as usize] += self.grad[i];
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

/// Hessian block coupling two model types.
///
/// Accumulates gradient and the upper triangle of the Gauss-Newton Hessian
/// from constraint residuals that reference parameters from two different
/// models A and B. `N = A::PARAM_COUNT + B::PARAM_COUNT`. The first `na`
/// indices in the block belong to A, the rest to B.
/// `T` is the float type (f32 or f64, default f64).
pub struct CrossBlock<A: Model, B: Model, const N: usize, T: crate::utils::Float = f64> {
    indices: [u32; N],
    na: usize,
    grad: [T; N],
    hessian: std::vec::Vec<T>,
    _marker: std::marker::PhantomData<(A, B, T)>,
}

impl<A: Model, B: Model, const N: usize, T: crate::utils::Float> Default for CrossBlock<A, B, N, T> {
    fn default() -> Self { Self::new() }
}

impl<A: Model, B: Model, const N: usize, T: crate::utils::Float> CrossBlock<A, B, N, T> {
    /// Create a new zeroed cross-block.
    pub fn new() -> Self {
        let na = A::PARAM_COUNT as usize;
        CrossBlock {
            indices: [u32::MAX; N],
            na,
            grad: std::array::from_fn(|_| T::zero()),
            hessian: vec![T::zero(); N * (N + 1) / 2],
            _marker: std::marker::PhantomData,
        }
    }

    /// Return the number of parameters belonging to model A.
    pub fn na(&self) -> usize { self.na }
    /// Return the number of parameters belonging to model B.
    pub fn nb(&self) -> usize { N - self.na }

    /// Set the global parameter indices: A's indices first, then B's.
    pub fn set_indices(&mut self, a_indices: &[u32], b_indices: &[u32]) {
        debug_assert_eq!(a_indices.len(), self.na);
        debug_assert_eq!(b_indices.len(), self.nb());
        self.indices[..self.na].copy_from_slice(a_indices);
        self.indices[self.na..].copy_from_slice(b_indices);
    }

    /// Reset gradient and hessian to zero.
    pub fn zero(&mut self) {
        self.grad = std::array::from_fn(|_| T::zero());
        self.hessian.fill(T::zero());
    }

    /// Return true if any parameter in this block is being optimized.
    pub fn is_active(&self) -> bool {
        self.indices.iter().any(|&i| i != u32::MAX)
    }

    /// Add one residual's contribution: accumulates 2*r*dr into gradient and 2*dr*dr^T into hessian.
    pub fn add_residual(&mut self, r: T, dr: &[T; N]) {
        let two = T::two();
        for i in 0..N {
            self.grad[i] += two * r * dr[i];
            for j in i..N {
                self.hessian[tri_idx(N, i, j)] += two * dr[i] * dr[j];
            }
        }
    }

    /// Accumulate this block into the full dense gradient and symmetric hessian.
    pub fn accumulate(&self, grad: &mut [T], hessian: &mut [T]) {
        let n_total = grad.len();
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            let gi = gi as usize;
            grad[gi] += self.grad[i];
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
    pub fn accumulate_band(&self, grad: &mut [T], band: &mut [T], kd: usize)
        -> Result<(), crate::simple_lm::BandError>
    {
        let ldab = kd + 1;
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            let gi = gi as usize;
            grad[gi] += self.grad[i];
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
    pub fn accumulate_sparse(&self, grad: &mut [T], coo: &mut crate::simple_lm::CooMatrix<T>) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            grad[gi as usize] += self.grad[i];
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
    pub fn accumulate_sparse_direct(&self, grad: &mut [T], csc: &mut crate::simple_lm::CscMatrix<T>) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            grad[gi as usize] += self.grad[i];
            for j in i..N {
                let gj = self.indices[j];
                if gj == u32::MAX { continue; }
                let (lo, hi) = if gi <= gj { (gi, gj) } else { (gj, gi) };
                let val = self.hessian[tri_idx(N, i, j)];
                if let Some(pos) = csc.find_pos(lo as usize, hi as usize) {
                    csc.vals[pos] = csc.vals[pos] + val;
                }
            }
        }
    }

    /// Accumulate into CSC vals using precomputed position list.
    pub fn accumulate_sparse_indexed(&self, grad: &mut [T], vals: &mut [T], positions: &[usize], cursor: &mut usize) {
        for i in 0..N {
            let gi = self.indices[i];
            if gi == u32::MAX { continue; }
            grad[gi as usize] += self.grad[i];
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

impl ModelSym for f32 {
    type Sym = E;
    fn sym(base: &str) -> E { arael_sym::symbol(base) }
}

impl ModelSym for f64 {
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
}
