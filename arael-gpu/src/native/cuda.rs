//! [`Device`] on CUDA: the driver API and NVRTC through cudarc, both loaded
//! at run time.

use super::{Arch, Arg, Device, HEADER, Launch, MAX_ARGS, NativeError};
use cudarc::driver::sys::{CUdevice_attribute, CUfunction_attribute};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, result};
use cudarc::nvrtc::{CompileError, CompileOptions, compile_ptx_with_opts};
use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex, RwLockReadGuard};

/// One CUDA device and a stream of its own.
pub struct CudaDevice {
    // The driver objects are released in `drop`, under the capture guard.
    ctx: ManuallyDrop<Arc<CudaContext>>,
    stream: ManuallyDrop<Arc<CudaStream>>,
    arch: Arch,
    /// Compiled modules by the hash of their full source.
    modules: ManuallyDrop<Mutex<HashMap<u64, Arc<CudaModule>>>>,
}

impl core::fmt::Debug for CudaDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CudaDevice").field("arch", &self.arch).finish_non_exhaustive()
    }
}

impl Drop for CudaDevice {
    fn drop(&mut self) {
        let _shared = capture_guard();
        // SAFETY: each is dropped once, here, and not used after.
        unsafe {
            ManuallyDrop::drop(&mut self.modules);
            ManuallyDrop::drop(&mut self.stream);
            ManuallyDrop::drop(&mut self.ctx);
        }
    }
}

/// Device memory on a [`CudaDevice`].
pub struct CudaBlock {
    // Freed in `drop`, under the capture guard.
    bytes: ManuallyDrop<CudaSlice<u8>>,
}

impl CudaBlock {
    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Whether it holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.bytes.len() == 0
    }
}

impl Drop for CudaBlock {
    fn drop(&mut self) {
        let _shared = capture_guard();
        // SAFETY: dropped once, here, and not used after.
        unsafe { ManuallyDrop::drop(&mut self.bytes) };
    }
}

/// A compiled entry point on a [`CudaDevice`].
#[derive(Debug)]
pub struct CudaKernel {
    // Holds its module, unloaded in `drop` under the capture guard when this
    // is the last holder.
    function: ManuallyDrop<CudaFunction>,
}

impl Drop for CudaKernel {
    fn drop(&mut self) {
        let _shared = capture_guard();
        // SAFETY: dropped once, here, and not used after.
        unsafe { ManuallyDrop::drop(&mut self.function) };
    }
}

/// A shared hold of the crate's graph-capture lock ([`crate::CAPTURE`]),
/// taken around every driver call CUDA refuses while another thread of the
/// process captures a graph in global mode, as CubeCL's captures do (the
/// refused call also invalidates that capture): allocations, frees, copies,
/// waits, module loads. Launches into the device's own stream are allowed
/// and do not take it. Goes with CubeCL (NATIVE_GPU_PLAN.md, stage N4).
fn capture_guard() -> RwLockReadGuard<'static, ()> {
    crate::CAPTURE.read().unwrap_or_else(|e| e.into_inner())
}

// `launch` passes scalars in the low bytes of 8-byte slots.
const _: () = assert!(cfg!(target_endian = "little"), "the CUDA runtime assumes a little-endian host");

fn driver(e: impl core::fmt::Display) -> NativeError {
    NativeError::Driver(e.to_string())
}

impl CudaDevice {
    /// Device `index`. [`NativeError::Unavailable`] when there is no such
    /// device or the CUDA driver is not installed.
    pub fn new(index: usize) -> Result<Self, NativeError> {
        let _shared = capture_guard();
        // cudarc loads the driver library on first use and panics if it is missing.
        let ctx = std::panic::catch_unwind(|| CudaContext::new(index))
            .map_err(|_| NativeError::Unavailable("the CUDA driver library is not installed".into()))?
            .map_err(|e| NativeError::Unavailable(format!("CUDA device {index}: {e}")))?;
        // Streams are ordered by our own events (GPU.md 7.6), not by events
        // cudarc records on every buffer use.
        unsafe { ctx.disable_event_tracking() };
        let stream = ctx.new_stream().map_err(driver)?;
        let (major, minor) = ctx.compute_capability().map_err(driver)?;
        let attr = |a| ctx.attribute(a).map_err(driver);
        let arch = Arch {
            name: ctx.name().map_err(driver)?,
            target: format!("sm_{major}{minor}"),
            warp: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_WARP_SIZE)? as u32,
            shared_bytes: attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)?
                as u32,
            has_f64: true,
        };
        Ok(CudaDevice {
            ctx: ManuallyDrop::new(ctx),
            stream: ManuallyDrop::new(stream),
            arch,
            modules: ManuallyDrop::new(Mutex::new(HashMap::new())),
        })
    }

    fn module(&self, entry: &str, source: String) -> Result<Arc<CudaModule>, NativeError> {
        let mut hasher = DefaultHasher::new();
        source.hash(&mut hasher);
        let key = hasher.finish();
        if let Some(module) = self.modules.lock().unwrap().get(&key) {
            return Ok(module.clone());
        }
        // NVRTC's defaults, stated: contraction on, correctly rounded division
        // and square root, subnormals kept (the plan's "default math").
        let opts = CompileOptions {
            fmad: Some(true),
            prec_div: Some(true),
            prec_sqrt: Some(true),
            ftz: Some(false),
            options: vec![format!("--gpu-architecture={}", self.arch.target), "--std=c++17".into()],
            name: Some(format!("{entry}.cu")),
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(&source, opts).map_err(|e| NativeError::Compile {
            entry: entry.to_string(),
            log: match e {
                CompileError::CompileError { log, .. } => log.to_string_lossy().into_owned(),
                other => format!("{other:?}"),
            },
        })?;
        let module = {
            let _shared = capture_guard();
            self.ctx.load_module(ptx).map_err(driver)?
        };
        self.modules.lock().unwrap().insert(key, module.clone());
        Ok(module)
    }
}

impl Device for CudaDevice {
    type Block = CudaBlock;
    type Kernel = CudaKernel;

    fn arch(&self) -> &Arch {
        &self.arch
    }

    fn alloc(&self, bytes: usize) -> Result<CudaBlock, NativeError> {
        let _shared = capture_guard();
        Ok(CudaBlock { bytes: ManuallyDrop::new(self.stream.alloc_zeros::<u8>(bytes).map_err(driver)?) })
    }

    fn write(&self, dst: &mut CudaBlock, offset: usize, src: &[u8]) -> Result<(), NativeError> {
        let mut view = dst.bytes.slice_mut(offset..offset + src.len());
        let _shared = capture_guard();
        self.stream.memcpy_htod(src, &mut view).map_err(driver)
    }

    fn read(&self, src: &CudaBlock, offset: usize, dst: &mut [u8]) -> Result<(), NativeError> {
        let view = src.bytes.slice(offset..offset + dst.len());
        let _shared = capture_guard();
        self.stream.memcpy_dtoh(&view, dst).map_err(driver)?;
        self.stream.synchronize().map_err(driver)
    }

    fn compile(&self, source: &str, defines: &str, entry: &str) -> Result<CudaKernel, NativeError> {
        let full = format!("{}{defines}\n{HEADER}\n{source}", self.arch.defines());
        let module = self.module(entry, full)?;
        let _shared = capture_guard();
        let function = module.load_function(entry).map_err(driver)?;
        // Launches may take all the shared memory the device allows a block,
        // not only the 48 KB a kernel gets without asking.
        let fixed = function.shared_size_bytes().map_err(driver)?;
        function
            .set_attribute(
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                self.arch.shared_bytes as i32 - fixed,
            )
            .map_err(driver)?;
        Ok(CudaKernel { function: ManuallyDrop::new(function) })
    }

    fn launch(&self, kernel: &CudaKernel, launch: Launch, args: &[Arg<'_, CudaBlock>]) -> Result<(), NativeError> {
        assert!(args.len() <= MAX_ARGS, "a launch takes at most {MAX_ARGS} arguments");
        // Each argument's value in a slot of its own, the driver reading as
        // many bytes from the slot as the kernel's parameter has (the low
        // bytes on a little-endian host).
        let mut values = [0u64; MAX_ARGS];
        for (slot, arg) in values.iter_mut().zip(args) {
            *slot = match *arg {
                Arg::Block(b, offset) => {
                    assert!(offset <= b.bytes.len(), "argument offset {offset} past the block's {} bytes", b.bytes.len());
                    b.bytes.device_ptr(&self.stream).0 + offset as u64
                }
                Arg::U32(v) => v as u64,
                Arg::I32(v) => v as u32 as u64,
                Arg::F32(v) => v.to_bits() as u64,
                Arg::F64(v) => v.to_bits(),
            };
        }
        let mut params = [core::ptr::null_mut::<core::ffi::c_void>(); MAX_ARGS];
        for (p, v) in params.iter_mut().zip(values.iter_mut()) {
            *p = v as *mut u64 as *mut core::ffi::c_void;
        }
        let [gx, gy, gz] = launch.grid;
        let [bx, by, bz] = launch.block;
        self.ctx.bind_to_thread().map_err(driver)?;
        // Safety: the kernel's parameters are the caller's to match; the
        // argument slots live until the call returns, and the launch only
        // queues.
        unsafe {
            result::launch_kernel(
                kernel.function.cu_function(),
                (gx, gy, gz),
                (bx, by, bz),
                launch.shared_bytes,
                self.stream.cu_stream(),
                &mut params[..args.len()],
            )
        }
        .map_err(driver)
    }

    fn sync(&self) -> Result<(), NativeError> {
        let _shared = capture_guard();
        self.stream.synchronize().map_err(driver)
    }
}
