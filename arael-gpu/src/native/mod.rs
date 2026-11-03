//! arael's own GPU runtime: kernels written once in a C++ subset that
//! compiles for CUDA, HIP and Metal, compiled at run time for the device at
//! hand, and a thin runtime per vendor behind [`Device`]. It replaces the
//! CubeCL route stage by stage (docs/dev/NATIVE_GPU_PLAN.md).
//!
//! On a device: [`NativeSupernodal`] (the factorization and its solve),
//! [`NativeSchur`] (the Schur reduction, the refinement and the
//! back-substitution around a factor) and [`NativeLmVectors`] (the LM loop's
//! vector passes). A sequence of launches records as a graph with
//! [`Device::capture`] and runs again with [`Device::replay`].
//!
//! ```ignore
//! use arael_gpu::native::{cuda::CudaDevice, Arg, Device, Launch};
//! let dev = CudaDevice::new(0)?;
//! let k = dev.compile("KERNEL void k(GLOBAL float* x) { x[global_x()] *= 2.0f; }", "", "k")?;
//! let mut x = dev.alloc(4 * n)?;
//! dev.write(&mut x, 0, bytemuck::cast_slice(&host))?;
//! dev.launch(&k, Launch::linear(n as u32, 256), &[Arg::Block(&x, 0)])?;
//! dev.read(&x, 0, bytemuck::cast_slice_mut(&mut host))?;
//! ```

use arael_faer::schur::SchurReal;

#[cfg(feature = "native-cuda")]
pub mod cuda;
#[cfg(feature = "native-cuda")]
mod disk;
pub mod lm;
pub mod schur;
pub mod supernodal;

#[cfg(feature = "native-cuda")]
pub use disk::take_cache_failure;
pub use lm::NativeLmVectors;
pub use schur::{NativeSchur, NativeSchurError};
pub use supernodal::{FactorizeError, NativeFactor, NativeSupernodal};

/// The header in front of every kernel source: what differs between targets.
pub const HEADER: &str = include_str!("../../kernels/arael_gpu.h");

/// The wide scalar `W` (`kernels/wide.h`), in front of a source whose
/// kernels use it, with [`Wide::defines`] among the defines.
pub const WIDE: &str = include_str!("../../kernels/wide.h");

/// The wide scalar of a device route: what H, the gradient, the cost, the
/// Schur reduction and the LM loop's vectors are kept and summed in on the
/// device. Both forms are 8 bytes; the host reads and writes f64 and
/// converts at the boundary.
///
/// ```
/// use arael_gpu::native::Wide;
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

/// The kernels a model's device assembly shares (`kernels/assembly.cpp`):
/// the gather, the rotation caches and the device functions the generated
/// constraint kernels call. A root's kernels are compiled after it.
pub const ASSEMBLY: &str = include_str!("../../kernels/assembly.cpp");

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

/// What a device's kernels are specialised on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Arch {
    /// The device's name, as its driver reports it.
    pub name: String,
    /// The compiler's target, e.g. `sm_89`.
    pub target: String,
    /// Threads that run in lockstep: a CUDA warp, a HIP wavefront, a Metal SIMD group.
    pub warp: u32,
    /// Shared memory one block may use, in bytes.
    pub shared_bytes: u32,
    /// Whether the device computes in f64.
    pub has_f64: bool,
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
pub enum NativeError {
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
}

impl core::fmt::Display for NativeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NativeError::Unavailable(why) => write!(f, "GPU unavailable: {why}"),
            NativeError::Driver(why) => write!(f, "GPU driver call failed: {why}"),
            NativeError::Compile { entry, log } => write!(f, "kernel `{entry}` did not compile:\n{log}"),
            NativeError::Limit(why) => write!(f, "beyond the GPU's limits: {why}"),
            NativeError::Capture(why) => write!(f, "not allowed in a graph capture: {why}"),
        }
    }
}

impl std::error::Error for NativeError {}

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
}

/// One GPU, a stream of its own, and the kernels compiled for it.
///
/// Every operation is ordered on the stream. [`Device::read`] and
/// [`Device::sync`] are the only ones that wait for the device. A device is
/// shared between threads (solvers hold it in an `Arc`), and what it makes
/// moves between them.
pub trait Device: Send + Sync {
    /// Device memory: a run of bytes.
    type Block: Send + Sync;
    /// A compiled entry point.
    type Kernel: Send + Sync;
    /// A recorded sequence of launches ([`Device::capture`]).
    type Graph: Send;

    /// What this device's kernels are specialised on.
    fn arch(&self) -> &Arch;

    /// `bytes` of device memory, zeroed.
    fn alloc(&self, bytes: usize) -> Result<Self::Block, NativeError>;

    /// Copies `src` into `dst` from byte `offset` on. `src` may be reused when this returns.
    fn write(&self, dst: &mut Self::Block, offset: usize, src: &[u8]) -> Result<(), NativeError>;

    /// Copies `dst.len()` bytes of `src` from byte `offset` on, after the work queued before it.
    fn read(&self, src: &Self::Block, offset: usize, dst: &mut [u8]) -> Result<(), NativeError>;

    /// [`read`](Self::read) of several ranges, `(src, offset, dst)` each, waiting once.
    ///
    /// ```ignore
    /// dev.read_many(&mut [(&a, 0, &mut status[..]), (&b, at, &mut scalars[..])])?;
    /// ```
    fn read_many(&self, reads: &mut [(&Self::Block, usize, &mut [u8])]) -> Result<(), NativeError>;

    /// Entry point `entry` of `source`, compiled with `defines` in front of [`HEADER`].
    /// A source compiles once per device; later calls find it in memory.
    fn compile(&self, source: &str, defines: &str, entry: &str) -> Result<Self::Kernel, NativeError>;

    /// Queues `kernel` with `args` (at most [`MAX_ARGS`]).
    fn launch(
        &self,
        kernel: &Self::Kernel,
        launch: Launch,
        args: &[Arg<'_, Self::Block>],
    ) -> Result<(), NativeError>;

    /// Waits until everything queued has run.
    fn sync(&self) -> Result<(), NativeError>;

    /// The launches `record` queues, recorded as a graph instead of run.
    ///
    /// Only launches may be queued while the capture is open: every other
    /// operation of the device fails with [`NativeError::Capture`]. The graph
    /// holds the launches' arguments as they were, block addresses included,
    /// so the blocks it uses outlive it, and its inputs change by writing
    /// into those blocks between replays.
    ///
    /// ```ignore
    /// let graph = dev.capture(|| dev.launch(&k, Launch::linear(n, 256), &[Arg::Block(&x, 0)]))?;
    /// dev.write(&mut x, 0, bytes)?;
    /// dev.replay(&graph)?;
    /// ```
    fn capture(&self, record: impl FnOnce() -> Result<(), NativeError>) -> Result<Self::Graph, NativeError>;

    /// Queues `graph`'s launches, in their recorded order.
    fn replay(&self, graph: &Self::Graph) -> Result<(), NativeError>;
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
