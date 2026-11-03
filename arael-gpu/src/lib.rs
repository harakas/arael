//! GPU kernels for arael's sparse solvers, written once in CubeCL and run on
//! CUDA or HIP: the block supernodal Cholesky in arael's panel layout and
//! its level-set solve ([`GpuSupernodal`]).
//!
//! The device does the numeric work; the symbolic analysis stays on the CPU
//! in arael-faer. A [`GpuSupernodal`] is built from a
//! [`SupernodalSymbolic`](arael_faer::supernodal::SupernodalSymbolic) once
//! per sparsity structure, then factorizes any matrix of that structure and
//! solves against the factor, in `f64` or in `f32` (for refinement against
//! the f64 matrix, which arael's solver drives).
//!
//! [`GpuSchur`] forms the reduced system of a Schur-complement route on the
//! device from the Hessian's values, and runs the solve, the refinement and
//! the back-substitution there against a [`GpuFactor`].
//!
//! Both run on a [`GpuContext`]: a stream of its own and an arena their
//! buffers are leased from and returned to, so solving problem after problem
//! through one context stops allocating once the largest has been seen.
//!
//! Features `cuda` and `hip` select the runtimes; without one the kernels
//! compile but no context can be made.
//!
//! ```ignore
//! use arael_gpu::{GpuBackend, GpuContext, GpuFactor, GpuPrecision};
//! let ctx = GpuContext::new(GpuBackend::Cuda, 0)?;
//! let mut g = GpuFactor::new(&ctx, GpuPrecision::F32, &sn, a.symbolic(), &Default::default());
//! g.factorize(a.vals())?;
//! g.solve(&mut rhs);
//! ```

pub mod assembly;
pub mod cache;
pub mod context;
pub mod host;
pub mod kernels;
pub mod lm;
pub mod native;
pub mod plan;
pub mod schur;
pub mod supernodal;
/// Synthetic block systems for the tests and the examples (not API).
#[doc(hidden)]
pub mod test_util;

use cubecl::prelude::*;

pub use cache::{KernelCache, kernel_cache, set_kernel_cache};
pub use context::{GpuContext, Lease};
/// CubeCL, for kernels written against this crate (arael's generated device code).
pub use cubecl;
/// A captured launch sequence ([`capture`]), replayed with `unsafe { graph.replay() }`.
pub use cubecl_runtime::client::Graph;
/// bytemuck, whose `Pod` bounds this crate's uploads take.
pub use bytemuck;
pub use plan::GpuSupernodalParams;
pub use schur::{GpuSchur, GpuSchurError, GraphKey, RefineStats};
pub use supernodal::GpuSupernodal;

use arael_faer::schur::SchurReal;
use arael_faer::supernodal::{SupernodalError, SupernodalSymbolic};

/// The factor scalars: `f32` and `f64`.
pub trait Real:
    Float
    + CubeElement
    + cubecl::num_traits::NumCast
    + cubecl::num_traits::ToPrimitive
    + CubePrimitive<Scalar: AtomicNumeric + PlaneNumeric>
{
}
impl Real for f32 {}
impl Real for f64 {}

/// Which runtime a [`client`] is made on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuBackend {
    /// NVIDIA, through the CUDA driver (feature `cuda`).
    Cuda,
    /// AMD, through HIP (feature `hip`).
    Hip,
}

/// Why no client could be made.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuError {
    /// The backend was not compiled in.
    Unavailable { backend: GpuBackend, reason: &'static str },
}

impl core::fmt::Display for GpuError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GpuError::Unavailable { backend, reason } => write!(f, "{backend:?} backend unavailable: {reason}"),
        }
    }
}

impl std::error::Error for GpuError {}

/// A client on device `index` of `backend`, on the calling thread's stream
/// ([`GpuContext::new`] makes one on a stream of its own).
pub fn client(backend: GpuBackend, index: usize) -> Result<Client, GpuError> {
    cache::configure();
    match backend {
        GpuBackend::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use cubecl_runtime::runtime::Runtime;
                Ok(cubecl::cuda::CudaRuntime::client(&cubecl::cuda::CudaDevice::new(index)))
            }
            #[cfg(not(feature = "cuda"))]
            {
                let _ = index;
                Err(GpuError::Unavailable { backend, reason: "arael-gpu built without the `cuda` feature" })
            }
        }
        GpuBackend::Hip => {
            #[cfg(feature = "hip")]
            {
                use cubecl_runtime::runtime::Runtime;
                Ok(cubecl::hip::HipRuntime::client(&cubecl::hip::device::AmdDevice::new(index)))
            }
            #[cfg(not(feature = "hip"))]
            {
                let _ = index;
                Err(GpuError::Unavailable { backend, reason: "arael-gpu built without the `hip` feature" })
            }
        }
    }
}

/// Graph captures against blocking waits, process-wide. CubeCL opens a
/// capture in the driver's global mode, in which a wait issued on any
/// thread of the process fails while a capture is open, and a read waits
/// on the reading thread, not on CubeCL's device thread. Every wait in this
/// crate holds the lock shared (the native runtime's driver calls too), a
/// capture holds it exclusively.
pub(crate) static CAPTURE: std::sync::RwLock<()> = std::sync::RwLock::new(());

/// Wait for everything submitted to the client.
pub fn sync(client: &Client) {
    let _shared = CAPTURE.read().unwrap_or_else(|e| e.into_inner());
    cubecl::future::block_on(client.sync()).expect("device sync");
}

/// [`sync`], reporting the device's error instead of panicking.
pub(crate) fn try_sync(client: &Client) -> Result<(), cubecl_runtime::server::ServerError> {
    let _shared = CAPTURE.read().unwrap_or_else(|e| e.into_inner());
    cubecl::future::block_on(client.sync())
}

/// Read `handles` back in one wait (see [`sync`] on why through here).
pub fn read(client: &Client, handles: Vec<cubecl::server::Handle>) -> Result<Vec<cubecl::bytes::Bytes>, cubecl_runtime::server::ServerError> {
    let _shared = CAPTURE.read().unwrap_or_else(|e| e.into_inner());
    client.read(handles)
}

/// Read `handle` back, waiting for it (see [`sync`] on why through here).
pub fn read_one(client: &Client, handle: cubecl::server::Handle) -> Result<cubecl::bytes::Bytes, cubecl_runtime::server::ServerError> {
    let _shared = CAPTURE.read().unwrap_or_else(|e| e.into_inner());
    client.read_one(handle)
}

/// Record the launches `record` issues on `client`'s stream as a graph,
/// without running them: on the device's thread with nothing else
/// scheduled against it, and with no wait of this crate in flight or
/// starting (see [`CAPTURE`]). `record` issues kernel launches only, on
/// buffers that outlive the graph; a run of the same launches before it
/// (after `client.graph_prepare()`) warms the pools the graph records.
pub fn capture<'a>(client: &'a Client, record: impl FnOnce() + Send + 'a) -> Graph {
    let _exclusive = CAPTURE.write().unwrap_or_else(|e| e.into_inner());
    client
        .exclusive(move || {
            client.start_capture().expect("start a graph capture");
            record();
            client.stop_capture().expect("capture the graph")
        })
        .expect("run the graph capture exclusively")
}

/// The factor's scalar.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GpuPrecision {
    F32,
    F64,
}

/// A [`GpuSupernodal`] of either scalar behind one type, for a caller whose
/// own scalar is generic.
pub enum GpuFactor {
    F32(GpuSupernodal<f32>),
    F64(GpuSupernodal<f64>),
}

impl GpuFactor {
    /// As [`GpuSupernodal::new`], in `precision`.
    pub fn new(
        ctx: &GpuContext,
        precision: GpuPrecision,
        sn: &SupernodalSymbolic,
        asym: &arael_faer::bsc::SymbolicSparseBlockColMat<plan::SparseIndex>,
        params: &GpuSupernodalParams,
    ) -> Self {
        match precision {
            GpuPrecision::F32 => GpuFactor::F32(GpuSupernodal::new(ctx, sn, asym, params)),
            GpuPrecision::F64 => GpuFactor::F64(GpuSupernodal::new(ctx, sn, asym, params)),
        }
    }

    /// As [`GpuSupernodal::factorize`].
    pub fn factorize<M: SchurReal + 'static>(&mut self, vals: &[M]) -> Result<(), SupernodalError> {
        match self {
            GpuFactor::F32(g) => g.factorize(vals),
            GpuFactor::F64(g) => g.factorize(vals),
        }
    }

    /// As [`GpuSupernodal::solve`].
    pub fn solve<M: SchurReal + 'static>(&mut self, rhs: &mut [M]) {
        match self {
            GpuFactor::F32(g) => g.solve(rhs),
            GpuFactor::F64(g) => g.solve(rhs),
        }
    }

    /// The factor's scalar.
    pub fn precision(&self) -> GpuPrecision {
        match self {
            GpuFactor::F32(_) => GpuPrecision::F32,
            GpuFactor::F64(_) => GpuPrecision::F64,
        }
    }

    /// The context the factor runs on; a [`GpuSchur`] feeding it shares it,
    /// so the two stay on one stream.
    pub fn context(&self) -> &GpuContext {
        match self {
            GpuFactor::F32(g) => g.context(),
            GpuFactor::F64(g) => g.context(),
        }
    }

    /// As [`GpuSupernodal::id`].
    pub fn id(&self) -> u64 {
        match self {
            GpuFactor::F32(g) => g.id(),
            GpuFactor::F64(g) => g.id(),
        }
    }
}
