//! arael's GPU runtime and the kernels of its sparse solvers. The kernels
//! are written once in a C++ subset (`kernels/`) and compiled at run time
//! for the device at hand; a thin runtime per vendor runs them behind
//! [`Device`]: [`cuda::CudaDevice`] (feature `cuda`, NVRTC) and
//! [`hip::HipDevice`] (feature `hip`, hipRTC). Both load the vendor's
//! libraries at run time, so a build needs neither the CUDA toolkit nor
//! ROCm. `cpu::CpuDevice` (feature `cpu`) runs the same kernels on the
//! CPU, compiled by the host's C++ compiler, for tests without a GPU.
//! A runtime's `devices()` (`cuda::devices`, `hip::devices`,
//! `metal::devices`) lists the machine's devices ([`DeviceInfo`]) without
//! opening one.
//!
//! On a device: [`GpuSupernodal`] (the block supernodal Cholesky and its
//! level-set solve), [`GpuSchur`] (the Schur reduction, the refinement and
//! the back-substitution around a [`GpuFactor`]), [`GpuCg`] (conjugate
//! gradients on a reduction's reduced system, in a factor's place) and
//! [`LmVectors`] (the LM loop's vector passes). H, the gradient, the step and the LM vectors
//! are in the route's [`Values`] (f32 for an f32 root), the cost and the
//! reduction in its [`Wide`] scalar. The symbolic analysis stays on the
//! CPU in arael-faer. A sequence of launches records as a graph with
//! [`Device::capture`] and runs again with [`Device::replay`]. A device's
//! work can be traced ([`trace`]), and the CPU device can count every read
//! and write of its kernels (`cpu::CpuDevice::with_probe`).
//!
//! ```ignore
//! use arael_gpu::{cuda::CudaDevice, Arg, Device, Launch};
//! let dev = CudaDevice::new(0)?;
//! let k = dev.compile("KERNEL void k(GLOBAL float* x) { x[global_x()] *= 2.0f; }", "", "k")?;
//! let mut x = dev.alloc(4 * n)?;
//! dev.write(&mut x, 0, bytemuck::cast_slice(&host))?;
//! dev.launch(&k, Launch::linear(n as u32, 256), &[Arg::Block(&x, 0)])?;
//! dev.read(&x, 0, bytemuck::cast_slice_mut(&mut host))?;
//! ```

use arael_faer::schur::SchurReal;

pub mod cache;
#[cfg(feature = "cuda")]
pub mod cuda;
#[cfg(any(feature = "cuda", feature = "hip", feature = "cpu"))]
mod disk;
#[cfg(feature = "hip")]
pub mod hip;
#[cfg(feature = "hip")]
mod hip_sys;
#[cfg(feature = "cpu")]
pub mod cpu;
#[cfg(feature = "cpu")]
mod cpu_probe;
pub mod cg;
#[doc(hidden)]
pub mod fault;
pub mod lm;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub mod metal;
pub mod plan;
#[cfg(any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple")))]
mod pool;
pub mod schur;
pub mod supernodal;
pub mod trace;
/// Synthetic block systems for the tests and the examples (not API).
#[doc(hidden)]
pub mod test_util;

/// bytemuck, whose `Pod` bounds this crate's uploads take.
pub use bytemuck;
pub use cache::{KernelCache, kernel_cache, set_kernel_cache};
/// The settings of a conjugate gradient solve ([`GpuCg`]), arael-faer's.
pub use arael_faer::cg::CgParams;
pub use cg::{CgStatus, GpuCg, GpuCgParams, GpuCgStats};
#[cfg(any(feature = "cuda", feature = "hip", feature = "cpu"))]
pub use disk::take_cache_failure;
pub use lm::{LmVectors, Scalars};
pub use plan::GpuSupernodalParams;
pub use schur::{GpuSchur, GpuSchurError, GpuSchurParams, GraphKey, RefineStats};
pub use supernodal::{FactorizeError, GpuFactor, GpuSupernodal};

/// The device this crate's tests run on: CUDA when its runtime is built
/// (feature `cuda`), else HIP (`hip`), else Metal (`metal`), else the CPU
/// (`cpu`).
#[cfg(all(test, feature = "cuda"))]
pub(crate) type TestDevice = cuda::CudaDevice;
#[cfg(all(test, feature = "hip", not(feature = "cuda")))]
pub(crate) type TestDevice = hip::HipDevice;
#[cfg(all(test, feature = "metal", target_vendor = "apple", not(any(feature = "cuda", feature = "hip"))))]
pub(crate) type TestDevice = metal::MetalDevice;
#[cfg(all(test, feature = "cpu", not(any(feature = "cuda", feature = "hip", all(feature = "metal", target_vendor = "apple")))))]
pub(crate) type TestDevice = cpu::CpuDevice;
#[cfg(all(test, any(feature = "cuda", feature = "hip", feature = "cpu", all(feature = "metal", target_vendor = "apple"))))]
pub(crate) type TestBlock = <TestDevice as Device>::Block;

/// The header in front of every kernel source: what differs between targets.
pub const HEADER: &str = include_str!("../kernels/arael_gpu.h");

/// The wide scalar `W` (`kernels/wide.h`), in front of a source whose
/// kernels use it, with [`Wide::defines`] among the defines.
pub const WIDE: &str = include_str!("../kernels/wide.h");

/// The kernels a model's device assembly shares (`kernels/assembly.cpp`):
/// the gather, the rotation caches and the device functions the generated
/// constraint kernels call. A root's kernels are compiled after it.
pub const ASSEMBLY: &str = include_str!("../kernels/assembly.cpp");

/// A fixed parameter's global index in the index tables the kernels read:
/// its step is zero and it is not gathered from the parameter vector.
pub const FIXED: u32 = u32::MAX;

/// The bit of an assembly tile record's stride word that marks a tile its
/// kernel stores instead of adding to: the kernel is the tile's only
/// writer ([`ASSEMBLY`]'s `TILE_STORE`).
pub const TILE_STORE: u32 = 1 << 31;

/// The factor's scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuPrecision {
    F32,
    F64,
}

/// The wide scalar of a device route: what H, the gradient, the cost, the
/// Schur reduction and the LM loop's vectors are kept and summed in on the
/// device. Both forms are 8 bytes; the host reads and writes f64 and
/// converts at the boundary.
///
/// ```
/// use arael_gpu::Wide;
/// let w = Wide::Df32;
/// let bytes = w.encode(&[0.1, 1.0 / 3.0]);
/// let mut back = [0.0; 2];
/// w.decode(&bytes, &mut back);
/// assert!((back[0] - 0.1).abs() < 1e-15 && (back[1] - 1.0 / 3.0).abs() < 1e-15);
/// ```
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Wide {
    /// `double`.
    #[default]
    F64,
    /// df32: two f32 whose sum is the value, about 48 bits of significand
    /// and f32's exponent range; f32 arithmetic only, for devices without
    /// fast f64 (or without f64).
    Df32,
}

impl Wide {
    /// Its name: `f64` or `df32`.
    pub fn name(self) -> &'static str {
        match self {
            Wide::F64 => "f64",
            Wide::Df32 => "df32",
        }
    }

    /// The `#define`s a source including [`WIDE`] is compiled with.
    pub fn defines(self) -> &'static str {
        match self {
            Wide::F64 => "",
            Wide::Df32 => "#define ARAEL_WIDE_DF32 1\n",
        }
    }

    /// `v` as this scalar's 8 bytes, as a little-endian word: a df32 is hi
    /// in the low half, lo in the high one.
    pub fn bits(self, v: f64) -> u64 {
        match self {
            Wide::F64 => v.to_bits(),
            Wide::Df32 => {
                let hi = v as f32;
                let lo = if hi.is_finite() { (v - hi as f64) as f32 } else { 0.0 };
                hi.to_bits() as u64 | (lo.to_bits() as u64) << 32
            }
        }
    }

    /// The value of 8 bytes of this scalar, as [`bits`](Self::bits) lays them out.
    pub fn value(self, bits: u64) -> f64 {
        match self {
            Wide::F64 => f64::from_bits(bits),
            Wide::Df32 => f32::from_bits(bits as u32) as f64 + f32::from_bits((bits >> 32) as u32) as f64,
        }
    }

    /// `v` as an array of this scalar.
    pub fn encode(self, v: &[f64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(8 * v.len());
        self.encode_into(v, &mut out);
        out
    }

    /// [`encode`](Self::encode) appended to `out`.
    pub fn encode_into(self, v: &[f64], out: &mut Vec<u8>) {
        out.extend(v.iter().flat_map(|&x| self.bits(x).to_le_bytes()));
    }

    /// An array of this scalar into `out`, `bytes.len() / 8` values.
    pub fn decode(self, bytes: &[u8], out: &mut [f64]) {
        assert_eq!(bytes.len(), 8 * out.len());
        for (o, c) in out.iter_mut().zip(bytes.as_chunks::<8>().0) {
            *o = self.value(u64::from_le_bytes(*c));
        }
    }

    /// `v` as a kernel argument of type `W`.
    pub fn arg<'a, B>(self, v: f64) -> Arg<'a, B> {
        Arg::Bits(self.bits(v))
    }
}

/// The scalars of a device route: the values' (`V` in the kernels: H, the
/// gradient, the step, the LM loop's vectors) and the wide one (`W`,
/// [`Wide`]). An f32 root's values are f32, as the CPU's route keeps them;
/// an f64 root's are in the wide scalar. A [`Wide`] converts to the values
/// of an f64 root.
///
/// ```
/// use arael_gpu::{Values, Wide};
/// let v = Values::of(4, Wide::F64);
/// assert_eq!((v.is_f32(), v.wide(), v.bytes()), (true, Wide::F64, 4));
/// let mut back = [0.0; 2];
/// v.decode(&v.encode(&[0.5, 0.25]), &mut back);
/// assert_eq!(back, [0.5, 0.25]);
/// assert_eq!(Values::from(Wide::Df32).bytes(), 8);
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Values {
    f32: bool,
    wide: Wide,
}

impl From<Wide> for Values {
    fn from(wide: Wide) -> Self {
        Values { f32: false, wide }
    }
}

impl Values {
    /// f32 values on a route of wide scalar `wide`.
    pub fn f32(wide: Wide) -> Self {
        Values { f32: true, wide }
    }

    /// The values of a root of `scalar_bytes`-byte scalars (4 or 8) on a
    /// route of wide scalar `wide`.
    pub fn of(scalar_bytes: usize, wide: Wide) -> Self {
        Values { f32: scalar_bytes == 4, wide }
    }

    /// Whether the values are f32 (else in the wide scalar).
    pub fn is_f32(self) -> bool {
        self.f32
    }

    /// The route's wide scalar.
    pub fn wide(self) -> Wide {
        self.wide
    }

    /// Bytes a value takes: 4 or 8.
    pub fn bytes(self) -> usize {
        if self.f32 { 4 } else { 8 }
    }

    /// The stored scalar's name: `f32`, `f64` or `df32`.
    pub fn ty(self) -> &'static str {
        if self.f32 { "f32" } else { self.wide.name() }
    }

    /// The `#define`s a source including [`WIDE`] is compiled with: the wide
    /// scalar's and `VALUES_F32`.
    pub fn defines(self) -> String {
        format!("{}#define VALUES_F32 {}\n", self.wide.defines(), self.f32 as u32)
    }

    /// `v` as an array of values.
    pub fn encode(self, v: &[f64]) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.bytes() * v.len());
        self.encode_into(v, &mut out);
        out
    }

    /// [`encode`](Self::encode) appended to `out`.
    pub fn encode_into(self, v: &[f64], out: &mut Vec<u8>) {
        self.extend(v.iter().copied(), out);
    }

    /// The values of `v` appended to `out` in one pass, for an upload of
    /// many values: [`push`](Self::push) per value is several times slower.
    ///
    /// ```ignore
    /// values.extend(hvals.iter().map(|v| v.to_f64()), &mut bytes);
    /// ```
    pub fn extend(self, v: impl Iterator<Item = f64>, out: &mut Vec<u8>) {
        if self.f32 {
            out.extend(v.flat_map(|x| (x as f32).to_le_bytes()));
        } else {
            let wide = self.wide;
            out.extend(v.flat_map(|x| wide.bits(x).to_le_bytes()));
        }
    }

    /// `v` appended to `out` as one value.
    pub fn push(self, v: f64, out: &mut Vec<u8>) {
        if self.f32 {
            out.extend((v as f32).to_le_bytes());
        } else {
            out.extend(self.wide.bits(v).to_le_bytes());
        }
    }

    /// An array of values into `out`, `bytes.len()` over
    /// [`bytes`](Self::bytes) of them.
    pub fn decode(self, bytes: &[u8], out: &mut [f64]) {
        if self.f32 {
            assert_eq!(bytes.len(), 4 * out.len());
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *o = f32::from_le_bytes(*c) as f64;
            }
        } else {
            self.wide.decode(bytes, out);
        }
    }

    /// `v` as a kernel argument of type `V`.
    pub fn arg<'a, B>(self, v: f64) -> Arg<'a, B> {
        if self.f32 { Arg::F32(v as f32) } else { self.wide.arg(v) }
    }
}

/// A scalar the kernels compute in: `f32` or `f64`.
pub trait Scalar: bytemuck::Pod + Send + Sync + 'static {
    /// Its C++ type, `float` or `double`.
    const CPP: &'static str;
    /// `v` rounded to this scalar.
    fn from_f64(v: f64) -> Self;
    fn to_f64(self) -> f64;
}

impl Scalar for f32 {
    const CPP: &'static str = "float";
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        self as f64
    }
}

impl Scalar for f64 {
    const CPP: &'static str = "double";
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// Who makes a device: the kernel sizes a device takes by default are the
/// ones measured on its vendor's GPUs ([`GpuSupernodalParams`]'s `*_on`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Vendor {
    Nvidia,
    Amd,
    Apple,
    /// The CPU device.
    Cpu,
}

/// A device as its runtime lists it (`cuda::devices`, `hip::devices`,
/// `metal::devices`), without opening it.
///
/// ```ignore
/// for d in arael_gpu::cuda::devices()? {
///     println!("cuda:{} {}{}", d.index, d.name, if d.integrated { " (integrated)" } else { "" });
/// }
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DeviceInfo {
    /// The index the runtime's device type opens it by.
    pub index: usize,
    /// The device's name, as its driver reports it.
    pub name: String,
    /// Whether the device shares the host's memory: an integrated GPU.
    pub integrated: bool,
}

/// What a device's kernels are specialised on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arch {
    /// The device's name, as its driver reports it.
    pub name: String,
    pub vendor: Vendor,
    /// The compiler's target, e.g. `sm_89` or `gfx1030`.
    pub target: String,
    /// Threads that run in lockstep: a CUDA warp, a HIP wavefront, a Metal SIMD group.
    pub warp: u32,
    /// Shared memory one block may use, in bytes.
    pub shared_bytes: u32,
    /// Multiprocessors (CUDA SMs, HIP compute units): how many blocks run
    /// at once, a few each.
    pub multiprocessors: u32,
    /// Whether the device computes in f64.
    pub has_f64: bool,
    /// Whether its f32 arithmetic flushes subnormals to zero (Metal, in
    /// safe math too): a df32 below about 1e-31, or whose low part is
    /// subnormal, keeps no low part there.
    pub flushes_subnormals: bool,
    /// Whether the device shares the host's memory: an integrated GPU, or
    /// the CPU device.
    pub integrated: bool,
}

impl Arch {
    /// The `#define`s every kernel source is compiled with on this device.
    pub fn defines(&self) -> String {
        format!(
            "#define ARAEL_WARP {}\n#define ARAEL_SHARED_BYTES {}\n",
            self.warp, self.shared_bytes
        )
    }
}

/// Why a device operation failed.
#[derive(Clone, Debug)]
pub enum GpuError {
    /// No such device, or the vendor's runtime library is not installed.
    Unavailable(String),
    /// A call into the driver failed.
    Driver(String),
    /// A kernel source did not compile; `log` is the compiler's output.
    Compile { entry: String, log: String },
    /// The problem needs more of a resource than the device has, e.g. shared
    /// memory for a supernode's columns.
    Limit(String),
    /// An operation a graph capture cannot hold: anything but a launch while
    /// a capture is open, or a capture inside another.
    Capture(String),
    /// A kernel dump asked for (`ARAEL_KERNEL_DUMP`) could not be written.
    Dump(String),
}

impl core::fmt::Display for GpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpuError::Unavailable(why) => write!(f, "GPU unavailable: {why}"),
            GpuError::Driver(why) => write!(f, "GPU driver call failed: {why}"),
            GpuError::Compile { entry, log } => write!(f, "kernel `{entry}` did not compile:\n{log}"),
            GpuError::Limit(why) => write!(f, "beyond the GPU's limits: {why}"),
            GpuError::Capture(why) => write!(f, "not allowed in a graph capture: {why}"),
            GpuError::Dump(why) => write!(f, "kernel dump not written: {why}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// One argument of a kernel launch.
pub enum Arg<'a, B> {
    /// A pointer to a block's bytes from a byte offset on.
    Block(&'a B, usize),
    U32(u32),
    I32(i32),
    F32(f32),
    F64(f64),
    /// Eight bytes as they are, little-endian: a wide scalar's value ([`Wide::arg`]).
    Bits(u64),
    /// A struct taken by value: its bytes, at most [`TABLE_BYTES`]. The
    /// kernel declares it `TABLE_PARAM(T) name` (`arael_gpu.h`), with `T`
    /// of exactly these bytes, and reads it from the launch's parameters,
    /// not from device memory: for a table of launch constants.
    ///
    /// ```ignore
    /// // struct Cols { u64 w[3]; };  KERNEL void k(GLOBAL u8* b, TABLE_PARAM(Cols) c)
    /// let cols = [0u64, 256, 512];
    /// dev.launch(&k, Launch::linear(n, 256), &[Arg::Block(&b, 0), Arg::Bytes(bytemuck::cast_slice(&cols))])?;
    /// ```
    Bytes(&'a [u8]),
}

/// The most bytes an [`Arg::Bytes`] takes: the parameter space every
/// runtime gives a launch (CUDA's and HIP's 4 KB, Metal's `setBytes`).
pub const TABLE_BYTES: usize = 4096;

/// `bytes` copied into 8-byte aligned words, the last one zero-padded: an
/// [`Arg::Bytes`] as a launch keeps it, so that a kernel's struct of
/// `u64`s reads it in place.
pub(crate) fn table_words(bytes: &[u8]) -> Vec<u64> {
    assert!(bytes.len() <= TABLE_BYTES, "a table argument of {} bytes, at most {TABLE_BYTES}", bytes.len());
    let mut w = vec![0u64; bytes.len().div_ceil(8)];
    bytemuck::cast_slice_mut::<u64, u8>(&mut w)[..bytes.len()].copy_from_slice(bytes);
    w
}

impl<B> Clone for Arg<'_, B> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<B> Copy for Arg<'_, B> {}

/// The most arguments a launch takes.
pub const MAX_ARGS: usize = 16;

/// The shape of a launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    /// Blocks in x, y and z.
    pub grid: [u32; 3],
    /// Threads per block in x, y and z.
    pub block: [u32; 3],
    /// Dynamic shared memory per block, in bytes: up to [`Arch::shared_bytes`]
    /// less the kernel's static shared memory.
    pub shared_bytes: u32,
}

impl Launch {
    /// `n` threads in blocks of `block`, the last block partly idle.
    pub fn linear(n: u32, block: u32) -> Self {
        Launch { grid: [n.div_ceil(block).max(1), 1, 1], block: [block, 1, 1], shared_bytes: 0 }
    }

    /// Whether the grid holds no block (a dimension of 0): a launch of it
    /// queues nothing ([`Device::launch`]).
    ///
    /// ```
    /// use arael_gpu::Launch;
    /// assert!(Launch { grid: [0, 1, 1], block: [32, 1, 1], shared_bytes: 0 }.is_empty());
    /// assert!(!Launch::linear(0, 32).is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.grid.contains(&0)
    }
}

/// A named run of a device block's bytes, for the tools that inspect a
/// device's work ([`Device::describe`]: the trace, the CPU device's access
/// probe).
///
/// ```
/// use arael_gpu::Section;
/// let s = Section::new("x", 64, 400, 4, "f32").role("V").doc("the parameters");
/// assert_eq!((s.at, s.bytes / s.elem, s.ty.as_str()), (64, 100, "f32"));
/// ```
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Section {
    pub name: String,
    /// Byte offset in the block.
    pub at: usize,
    pub bytes: usize,
    /// Bytes of one element.
    pub elem: usize,
    /// The stored scalar: `u32`, `u64`, `f32`, `f64`, `df32`, `bytes`.
    pub ty: String,
    /// The kernels' typedef for it (`V`, `W`, `R`, `F`), or empty.
    pub role: String,
    /// A label grouping a block's sections (`structure`, `values`), or empty.
    pub group: String,
    pub doc: String,
    /// Elements per record, where the elements come in records (a tile's
    /// base, stride and first index: 3); 1 otherwise.
    pub record: usize,
}

impl Section {
    /// `bytes` bytes from byte `at` on, of `elem`-byte elements stored as `ty`.
    pub fn new(name: impl Into<String>, at: usize, bytes: usize, elem: usize, ty: &str) -> Self {
        Section { name: name.into(), at, bytes, elem, ty: ty.to_string(), role: String::new(), group: String::new(), doc: String::new(), record: 1 }
    }

    /// `count` u32 words from byte `at` on.
    pub fn u32s(name: impl Into<String>, at: usize, count: usize) -> Self {
        Self::new(name, at, 4 * count, 4, "u32")
    }

    /// `count` values of `values` (the kernels' `V`) from byte `at` on.
    pub fn values(name: impl Into<String>, at: usize, count: usize, values: Values) -> Self {
        Self::new(name, at, values.bytes() * count, values.bytes(), values.ty()).role("V")
    }

    /// `count` values of the wide scalar `wide` (the kernels' `W`) from byte `at` on.
    pub fn wide(name: impl Into<String>, at: usize, count: usize, wide: Wide) -> Self {
        Self::new(name, at, 8 * count, 8, wide.name()).role("W")
    }

    pub fn role(mut self, role: &str) -> Self {
        self.role = role.to_string();
        self
    }

    pub fn group(mut self, group: &str) -> Self {
        self.group = group.to_string();
        self
    }

    pub fn doc(mut self, doc: impl Into<String>) -> Self {
        self.doc = doc.into();
        self
    }

    /// The elements come in records of `elems` each.
    ///
    /// ```
    /// # use arael_gpu::Section;
    /// // 49 tiles of (base, stride, first index)
    /// let tiles = Section::u32s("E0_TILE", 0, 3 * 49).record(3);
    /// assert_eq!((tiles.bytes, tiles.elem, tiles.record), (588, 4, 3));
    /// ```
    pub fn record(mut self, elems: usize) -> Self {
        self.record = elems;
        self
    }
}

/// The sections of a plan's words ([`Device::describe`]): a header of
/// `header` words, then the arrays its `slots` (slot, name, doc) hold the
/// word offsets of, each running to the next array's offset or the plan's
/// end. Arrays at one offset (all but one empty) are one section named
/// after them all.
pub(crate) fn plan_sections(words: &[u32], header: usize, slots: &[(usize, &str, &str)]) -> Vec<Section> {
    let mut arrays: Vec<(usize, &str, &str)> = slots.iter().map(|&(slot, name, doc)| (words[slot] as usize, name, doc)).collect();
    arrays.sort_by_key(|a| a.0);
    let mut out = vec![Section::u32s("header", 0, header).doc("the word offsets of the plan's arrays, a slot each")];
    let mut i = 0;
    while i < arrays.len() {
        let at = arrays[i].0;
        let mut j = i;
        while j + 1 < arrays.len() && arrays[j + 1].0 == at {
            j += 1;
        }
        let end = arrays.get(j + 1).map_or(words.len(), |a| a.0);
        let name: Vec<&str> = arrays[i..=j].iter().map(|a| a.1).collect();
        if end > at && at >= header {
            out.push(Section::u32s(name.join("|"), 4 * at, end - at).doc(arrays[j].2));
        }
        i = j + 1;
    }
    out
}

/// What one launch read or wrote of one section, as the CPU device's probe
/// counts it ([`Access`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Touch {
    /// Accesses, and the bytes they moved.
    pub n: u64,
    pub bytes: u64,
    /// Distinct elements touched.
    pub unique: u64,
    /// The first and the last byte touched, relative to the section.
    pub lo: u64,
    pub hi: u64,
    /// Device threads that touched it.
    pub threads: u64,
    /// Accesses right after the same thread's previous access to the
    /// section (`seq`), or at its address again (`same`).
    pub seq: u64,
    pub same: u64,
    /// Threads whose first access is right after the previous thread's
    /// first (`adjacent`), or at its address (`shared`).
    pub adjacent: u64,
    pub shared: u64,
    /// The access widths seen, a bit per width: 1, 2, 4, 8, 16 bytes.
    pub sizes: u8,
}

/// One launch's accesses to one section of one of its block arguments,
/// measured ([`Device::take_accesses`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Access {
    /// The launch's first argument that points into the block.
    pub arg: usize,
    /// The section's index in the block's [`Device::describe`] list;
    /// `None` for bytes outside every section, or in a block not described.
    pub section: Option<usize>,
    pub read: Touch,
    pub write: Touch,
}

/// The stream a launch is queued on ([`Device::launch_on`]): the device's
/// own, or the side stream, whose work overlaps it between a
/// [`Device::fork`] and the [`Device::join`] after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lane {
    Main,
    Side,
}

/// The side stream's forks open on a device ([`Device::fork`]), for the
/// checks every runtime makes on them.
#[derive(Debug, Default)]
pub(crate) struct Forks(core::sync::atomic::AtomicUsize);

impl Forks {
    /// Forks open.
    pub(crate) fn open(&self) -> usize {
        self.0.load(core::sync::atomic::Ordering::Acquire)
    }

    pub(crate) fn fork(&self) {
        self.0.fetch_add(1, core::sync::atomic::Ordering::AcqRel);
    }

    /// Panics without an open fork.
    pub(crate) fn join(&self) {
        if self.0.fetch_sub(1, core::sync::atomic::Ordering::AcqRel) == 0 {
            self.0.fetch_add(1, core::sync::atomic::Ordering::AcqRel);
            panic!("a join without a fork");
        }
    }

    /// Panics for a side launch without an open fork.
    pub(crate) fn check(&self, lane: Lane) {
        assert!(lane == Lane::Main || self.open() > 0, "a side launch outside a fork and its join");
    }

    /// [`GpuError::Capture`] when the forks open differ from `at_begin`, the
    /// count when the capture began.
    pub(crate) fn check_capture(&self, at_begin: usize) -> Result<(), GpuError> {
        if self.open() != at_begin {
            return Err(GpuError::Capture("the capture ended with the side stream forked and not joined".into()));
        }
        Ok(())
    }
}

/// One GPU, a stream of its own, and the kernels compiled for it.
///
/// Every operation is ordered on the stream. [`Device::read`] and
/// [`Device::sync`] are the only ones that wait for the device. A device is
/// shared between threads (solvers hold it in an `Arc`), and what it makes
/// moves between them.
///
/// Launches may also go to a side stream ([`Lane::Side`]) between a
/// [`fork`](Device::fork) and a [`join`](Device::join), to run alongside
/// the main stream's; a device with one stream runs them in order on it.
pub trait Device: Send + Sync {
    /// Device memory: a run of bytes.
    type Block: Send + Sync;
    /// A compiled entry point.
    type Kernel: Send + Sync;
    /// A recorded sequence of launches ([`Device::capture`]).
    type Graph: Send;
    /// Pinned (page-locked) host memory, the source of an asynchronous
    /// [`Device::write_from_host`].
    type Host: Send + Sync;

    /// The runtime's name: `cuda`, `hip`.
    const RUNTIME: &'static str;

    /// What this device's kernels are specialised on.
    fn arch(&self) -> &Arch;

    /// `bytes` of device memory, zeroed. A block that is dropped goes back
    /// to the device and is handed out again: an allocation takes the
    /// smallest free block that holds it, and a fresh block has room for a
    /// quarter more, so a solve of a slightly changed structure allocates
    /// nothing new.
    fn alloc(&self, bytes: usize) -> Result<Self::Block, GpuError>;

    /// Blocks the device has allocated from the driver over its life.
    fn allocations(&self) -> usize;

    /// Bytes of device memory the device holds, in blocks in use or free.
    fn reserved_bytes(&self) -> usize;

    /// Pinned host blocks the device has allocated from the driver over its
    /// life; a dropped one is handed out again, as device blocks are.
    fn host_allocations(&self) -> usize;

    /// Copies `src` into `dst` from byte `offset` on. `src` may be reused when this returns.
    fn write(&self, dst: &mut Self::Block, offset: usize, src: &[u8]) -> Result<(), GpuError>;

    /// `bytes` of pinned (page-locked, cached) host memory. Its contents
    /// start undefined.
    fn alloc_host(&self, bytes: usize) -> Result<Self::Host, GpuError>;

    /// The bytes of `host`, to write, once every copy queued from it has
    /// completed.
    fn host_mut<'a>(&self, host: &'a mut Self::Host) -> Result<&'a mut [u8], GpuError>;

    /// Queues a copy of `host[range]` into `dst` from byte `offset` on and
    /// returns without waiting; [`host_mut`](Self::host_mut) waits for it
    /// before `host` is written again.
    ///
    /// ```ignore
    /// let mut staging = dev.alloc_host(n)?;
    /// dev.host_mut(&mut staging)?[..n].copy_from_slice(&bytes);
    /// dev.write_from_host(&mut block, 0, &staging, 0..n)?;
    /// ```
    fn write_from_host(&self, dst: &mut Self::Block, offset: usize, host: &Self::Host, range: core::ops::Range<usize>) -> Result<(), GpuError>;

    /// Copies `dst.len()` bytes of `src` from byte `offset` on, after the work queued before it.
    fn read(&self, src: &Self::Block, offset: usize, dst: &mut [u8]) -> Result<(), GpuError>;

    /// [`read`](Self::read) of several ranges, `(src, offset, dst)` each, waiting once.
    ///
    /// ```ignore
    /// dev.read_many(&mut [(&a, 0, &mut status[..]), (&b, at, &mut scalars[..])])?;
    /// ```
    fn read_many(&self, reads: &mut [(&Self::Block, usize, &mut [u8])]) -> Result<(), GpuError>;

    /// Entry point `entry` of `source`, compiled with `defines` in front of [`HEADER`].
    /// A source compiles once per device; later calls find it in memory.
    fn compile(&self, source: &str, defines: &str, entry: &str) -> Result<Self::Kernel, GpuError> {
        Ok(self.compile_many(source, defines, &[entry])?.remove(0))
    }

    /// The entry points `entries` of `source`, as [`compile`](Self::compile)
    /// gives each, the source looked up once.
    fn compile_many(&self, source: &str, defines: &str, entries: &[&str]) -> Result<Vec<Self::Kernel>, GpuError>;

    /// The translation unit [`compile_many`](Self::compile_many) compiles
    /// for `source` and `defines`: the architecture's defines, `defines`,
    /// [`HEADER`], `source`.
    fn unit(&self, source: &str, defines: &str) -> String {
        format!("{}{defines}\n{HEADER}\n{source}", self.arch().defines())
    }

    /// Queues `kernel` with `args` (at most [`MAX_ARGS`]). A launch over an
    /// empty grid ([`Launch::is_empty`]) queues nothing: it is not run,
    /// recorded into a capture, timed or traced.
    fn launch(
        &self,
        kernel: &Self::Kernel,
        launch: Launch,
        args: &[Arg<'_, Self::Block>],
    ) -> Result<(), GpuError>;

    /// Queues `kernel` on `lane`: [`launch`](Self::launch) on the main
    /// stream, on the side stream between a [`fork`](Self::fork) and the
    /// [`join`](Self::join) after it (a side launch outside those panics).
    /// The CPU device has one stream and runs the side's launches on it, in
    /// order.
    ///
    /// ```ignore
    /// dev.launch(&update_next, launch, &args)?;
    /// dev.fork()?;
    /// dev.launch(&update_rest, big, &args)?;
    /// dev.launch_on(Lane::Side, &factor_next, small, &args)?; // runs alongside update_rest
    /// dev.join()?;
    /// ```
    fn launch_on(&self, lane: Lane, kernel: &Self::Kernel, launch: Launch, args: &[Arg<'_, Self::Block>]) -> Result<(), GpuError>;

    /// Opens the side stream: its launches from now on start after the main
    /// stream's work queued so far.
    fn fork(&self) -> Result<(), GpuError>;

    /// Closes the side stream (a join without a fork panics): the main
    /// stream's work queued from now on starts after the side's. Reads and
    /// writes follow the main stream only, so a side launch's results are
    /// read after a join; a capture whose forks it did not join fails
    /// ([`GpuError::Capture`]).
    fn join(&self) -> Result<(), GpuError>;

    /// Waits until everything queued has run, on both streams.
    fn sync(&self) -> Result<(), GpuError>;

    /// The launches `record` queues, recorded as a graph instead of run.
    ///
    /// Only launches may be queued while the capture is open: every other
    /// operation of the device fails with [`GpuError::Capture`]. The graph
    /// holds the launches' arguments as they were, block addresses included,
    /// so the blocks it uses outlive it, and its inputs change by writing
    /// into those blocks between replays.
    ///
    /// ```ignore
    /// let graph = dev.capture(|| dev.launch(&k, Launch::linear(n, 256), &[Arg::Block(&x, 0)]))?;
    /// dev.write(&mut x, 0, bytes)?;
    /// dev.replay(&graph)?;
    /// ```
    fn capture(&self, record: impl FnOnce() -> Result<(), GpuError>) -> Result<Self::Graph, GpuError>;

    /// Queues `graph`'s launches, in their recorded order.
    fn replay(&self, graph: &Self::Graph) -> Result<(), GpuError>;

    /// Names `block` and its sections for the tools that inspect a device's
    /// work: [`trace::Traced`](crate::trace::Traced) records them, the CPU
    /// device's probe counts accesses by them. A device that inspects
    /// nothing ignores the call and does not run `sections`.
    ///
    /// ```ignore
    /// dev.describe(&block, "lm", &|| vec![Section::u32s("pos", 0, n), Section::values("x", at_x, n, values)]);
    /// ```
    fn describe(&self, _block: &Self::Block, _name: &str, _sections: &dyn Fn() -> Vec<Section>) {}

    /// Opens a named scope around the operations up to the matching
    /// [`scope_end`](Self::scope_end), for a trace; nothing otherwise.
    fn scope_begin(&self, _name: &str) {}

    /// Closes the scope the last [`scope_begin`](Self::scope_begin) opened.
    fn scope_end(&self) {}

    /// The accesses of each launch run since the last call, in their order,
    /// when the device measures them (the CPU device with its probe on);
    /// empty otherwise.
    fn take_accesses(&self) -> Vec<Vec<Access>> {
        Vec::new()
    }

    /// Starts (`true`) or stops timing each launch on the device, for
    /// [`take_times`](Self::take_times). A timed launch runs alone: the
    /// host waits for it, so launches overlap neither one another nor the
    /// host. A capture made while timing keeps its launches, and its
    /// replays issue them one by one, timed. For measuring, not for speed;
    /// a device that does not time ignores the call.
    ///
    /// ```ignore
    /// dev.time_launches(true);
    /// dev.launch(&k, Launch::linear(n, 256), &[Arg::Block(&x, 0)])?;
    /// let seconds = dev.take_times()[0];
    /// ```
    fn time_launches(&self, _on: bool) {}

    /// The device time, in seconds, of each launch since the last call, in
    /// launch order, a replayed graph's launches among them; empty unless
    /// [`time_launches`](Self::time_launches) is on.
    fn take_times(&self) -> Vec<f64> {
        Vec::new()
    }

    /// Marks this point of the device's main queue as stamp `slot` (below
    /// [`STAMPS`]), for [`span`](Self::span): one queued marker, no wait,
    /// the launches around it undisturbed. A device that keeps no stamps
    /// ignores it.
    ///
    /// ```ignore
    /// dev.stamp(0)?;
    /// dev.launch(&k, Launch::linear(n, 256), &[Arg::Block(&x, 0)])?;
    /// dev.stamp(1)?;
    /// dev.read(&x, 0, &mut out)?;      // waits: the device has passed both
    /// let seconds = dev.span(0, 1)?;   // Some(..) on a device with stamps
    /// ```
    ///
    /// # Panics
    ///
    /// When `slot` is not below [`STAMPS`].
    fn stamp(&self, slot: usize) -> Result<(), GpuError> {
        assert!(slot < STAMPS, "stamp {slot} of {STAMPS}");
        Ok(())
    }

    /// The device time in seconds from stamp `from` to stamp `to`, once the
    /// device has passed both (a read waits for it). `None` on a device
    /// that keeps no stamps, and when either was never made.
    fn span(&self, _from: usize, _to: usize) -> Result<Option<f64>, GpuError> {
        Ok(None)
    }
}

/// The stamps a device keeps ([`Device::stamp`]).
pub const STAMPS: usize = 16;

/// A scope of a device's operations ([`Device::scope_begin`]), closed when
/// dropped.
///
/// ```ignore
/// let _scope = Scope::new(&*dev, "attempt");
/// ```
pub struct Scope<'a, D: Device + ?Sized>(&'a D);

impl<'a, D: Device + ?Sized> Scope<'a, D> {
    pub fn new(dev: &'a D, name: &str) -> Self {
        dev.scope_begin(name);
        Scope(dev)
    }
}

impl<D: Device + ?Sized> Drop for Scope<'_, D> {
    fn drop(&mut self) {
        self.0.scope_end();
    }
}

/// [`GpuError::Unavailable`] when `uses_f64` and the device has no f64
/// (Metal): `what` names what needs it.
pub(crate) fn check_f64(arch: &Arch, uses_f64: bool, what: &str) -> Result<(), GpuError> {
    if uses_f64 && !arch.has_f64 {
        return Err(GpuError::Unavailable(format!("{} has no f64, which {what} needs", arch.name)));
    }
    Ok(())
}

/// `#define NAME VALUEu` for each pair, for a kernel source's constants.
pub(crate) fn define_u32s(s: &mut String, consts: &[(&str, usize)]) {
    use core::fmt::Write as _;
    for (name, v) in consts {
        assert!(*v <= u32::MAX as usize, "{name} = {v} is no u32");
        writeln!(s, "#define {name} {v}u").unwrap();
    }
}

/// Whether `M` is `f64` (otherwise it is `f32`, the only other [`SchurReal`]).
pub(crate) fn is_f64<M: SchurReal + 'static>() -> bool {
    if core::any::TypeId::of::<M>() == core::any::TypeId::of::<f64>() {
        return true;
    }
    assert_eq!(core::any::TypeId::of::<M>(), core::any::TypeId::of::<f32>(), "SchurReal is f32 or f64");
    false
}

/// The bytes of `v`.
pub(crate) fn scalar_bytes<M: SchurReal>(v: &[M]) -> &[u8] {
    // SAFETY: SchurReal is sealed to f32 and f64, plain values without padding.
    unsafe { core::slice::from_raw_parts(v.as_ptr() as *const u8, core::mem::size_of_val(v)) }
}

/// Byte offsets of a block's sections, each on a 256-byte boundary.
pub(crate) struct Layout {
    at: usize,
}

impl Layout {
    pub fn new() -> Self {
        Layout { at: 0 }
    }

    /// The offset of a section of `bytes` (at least one), after the last.
    pub fn take(&mut self, bytes: usize) -> usize {
        let start = self.at;
        self.at = (self.at + bytes.max(1)).next_multiple_of(256);
        start
    }

    /// The block's size.
    pub fn bytes(&self) -> usize {
        self.at
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The assembly source defines the index words the host writes as the
    /// host's constants.
    #[test]
    fn the_assembly_defines_the_hosts_index_words() {
        for (name, v) in [("FIXED", FIXED), ("TILE_STORE", TILE_STORE)] {
            assert!(ASSEMBLY.contains(&format!("#define {name} {v:#010x}u\n")), "{name} {v:#010x}");
        }
    }

    /// [`Values::extend`] writes the bytes [`Values::push`] writes a value
    /// at a time, in each scalar, and they decode back to the values the
    /// scalar holds.
    #[test]
    fn values_extend_as_push_does() {
        let v = [0.1, 1.0 / 3.0, -2.5e7, 0.0, f64::INFINITY, 1e-30];
        for values in [Values::from(Wide::F64), Values::from(Wide::Df32), Values::f32(Wide::Df32)] {
            let mut one = Vec::new();
            v.iter().for_each(|&x| values.push(x, &mut one));
            let mut all = vec![7u8];
            values.extend(v.iter().copied(), &mut all);
            assert_eq!(all[1..], one[..], "{values:?}");
            assert_eq!(values.encode(&v), one, "{values:?}");
            let mut back = [0.0; 6];
            values.decode(&one, &mut back);
            for (b, x) in back.iter().zip(v) {
                let held = if values.is_f32() { x as f32 as f64 } else { values.wide().value(values.wide().bits(x)) };
                assert_eq!(*b, held, "{values:?}");
            }
        }
    }
}
