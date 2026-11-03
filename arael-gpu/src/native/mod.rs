//! arael's own GPU runtime: kernels written once in a C++ subset that
//! compiles for CUDA, HIP and Metal, compiled at run time for the device at
//! hand, and a thin runtime per vendor behind [`Device`]. It replaces the
//! CubeCL route stage by stage (docs/dev/NATIVE_GPU_PLAN.md).
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

#[cfg(feature = "native-cuda")]
pub mod cuda;
pub mod supernodal;

pub use supernodal::{FactorizeError, NativeSupernodal};

/// The header in front of every kernel source: what differs between targets.
pub const HEADER: &str = include_str!("../../kernels/arael_gpu.h");

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
}

impl core::fmt::Display for NativeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NativeError::Unavailable(why) => write!(f, "GPU unavailable: {why}"),
            NativeError::Driver(why) => write!(f, "GPU driver call failed: {why}"),
            NativeError::Compile { entry, log } => write!(f, "kernel `{entry}` did not compile:\n{log}"),
            NativeError::Limit(why) => write!(f, "beyond the GPU's limits: {why}"),
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
/// [`Device::sync`] are the only ones that wait for the device.
pub trait Device {
    /// Device memory: a run of bytes.
    type Block;
    /// A compiled entry point.
    type Kernel;

    /// What this device's kernels are specialised on.
    fn arch(&self) -> &Arch;

    /// `bytes` of device memory, zeroed.
    fn alloc(&self, bytes: usize) -> Result<Self::Block, NativeError>;

    /// Copies `src` into `dst` from byte `offset` on. `src` may be reused when this returns.
    fn write(&self, dst: &mut Self::Block, offset: usize, src: &[u8]) -> Result<(), NativeError>;

    /// Copies `dst.len()` bytes of `src` from byte `offset` on, after the work queued before it.
    fn read(&self, src: &Self::Block, offset: usize, dst: &mut [u8]) -> Result<(), NativeError>;

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
}
