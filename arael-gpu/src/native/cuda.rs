//! [`Device`] on CUDA: the driver API and NVRTC through cudarc, both loaded
//! at run time.

use super::disk::DiskCache;
use super::{Arch, Arg, Device, HEADER, Launch, MAX_ARGS, NativeError};
use cudarc::driver::sys::{self, CUdevice_attribute, CUfunction_attribute};
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, result};
use cudarc::nvrtc::{self, Ptx};
use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLockReadGuard};

/// One CUDA device and a stream of its own.
pub struct CudaDevice {
    // The driver objects are released in `drop`, under the capture guard.
    ctx: ManuallyDrop<Arc<CudaContext>>,
    stream: ManuallyDrop<Arc<CudaStream>>,
    arch: Arch,
    /// Compiled modules by their full source.
    modules: ManuallyDrop<Mutex<HashMap<String, Arc<CudaModule>>>>,
    /// A graph capture is open on the stream.
    capturing: AtomicBool,
    /// Blocks allocated over the device's life.
    allocations: AtomicUsize,
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

impl CudaKernel {
    /// Local memory per thread, in bytes: per-thread arrays the compiler
    /// could not keep in registers, and spills.
    pub fn local_bytes(&self) -> Result<usize, NativeError> {
        Ok(self.function.local_size_bytes().map_err(driver)? as usize)
    }

    /// Registers per thread.
    pub fn registers(&self) -> Result<usize, NativeError> {
        Ok(self.function.num_regs().map_err(driver)? as usize)
    }
}

/// A captured graph on a [`CudaDevice`], instantiated for replay.
pub struct CudaGraph {
    exec: sys::CUgraphExec,
    // Keeps the context the graph belongs to alive.
    ctx: Arc<CudaContext>,
}

// SAFETY: the executable graph is a driver handle, usable from any thread
// with its context bound; `replay` takes `&CudaGraph` but launches are
// ordered by the driver behind the graph's earlier launches. Not `Sync`:
// the driver does not synchronize calls on one graph from several threads.
unsafe impl Send for CudaGraph {}

impl Drop for CudaGraph {
    fn drop(&mut self) {
        let _shared = capture_guard();
        if self.ctx.bind_to_thread().is_ok() {
            // SAFETY: the handle came from instantiate and is destroyed once.
            // An error here has nobody to go to: the graph is gone either way.
            let _ = unsafe { result::graph::exec_destroy(self.exec) };
        }
    }
}

/// A capture open on a stream, ended on drop if [`OpenCapture::end`] was
/// not reached (a failed or panicking recording), so the stream is usable
/// again.
struct OpenCapture<'a> {
    dev: &'a CudaDevice,
    open: bool,
}

impl<'a> OpenCapture<'a> {
    fn begin(dev: &'a CudaDevice) -> Result<Self, NativeError> {
        if dev.capturing.swap(true, Ordering::AcqRel) {
            return Err(NativeError::Capture("a capture is already open on this device".into()));
        }
        let mut open = OpenCapture { dev, open: false };
        let _shared = capture_guard();
        dev.ctx.bind_to_thread().map_err(driver)?;
        // Thread-local: the capture forbids unsafe calls on this thread only,
        // so other threads (and contexts) keep working while it is open.
        // SAFETY: the stream is the device's own and alive.
        unsafe { result::stream::begin_capture(dev.stream.cu_stream(), sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL) }
            .map_err(driver)?;
        open.open = true;
        Ok(open)
    }

    /// Ends the capture; its graph, or null when nothing was recorded.
    fn end(mut self) -> Result<RawGraph, NativeError> {
        self.open = false;
        let _shared = capture_guard();
        self.dev.ctx.bind_to_thread().map_err(driver)?;
        // SAFETY: the capture was begun on this stream by `begin`.
        let graph = unsafe { result::stream::end_capture(self.dev.stream.cu_stream()) }.map_err(driver)?;
        Ok(RawGraph(graph))
    }
}

impl Drop for OpenCapture<'_> {
    fn drop(&mut self) {
        if self.open {
            let _shared = capture_guard();
            if self.dev.ctx.bind_to_thread().is_ok() {
                // SAFETY: as in `end`; the partial graph is discarded.
                if let Ok(graph) = unsafe { result::stream::end_capture(self.dev.stream.cu_stream()) } {
                    drop(RawGraph(graph));
                }
            }
        }
        self.dev.capturing.store(false, Ordering::Release);
    }
}

/// A captured, not instantiated graph, destroyed on drop.
struct RawGraph(sys::CUgraph);

impl Drop for RawGraph {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the handle came from end_capture and is destroyed once.
            let _ = unsafe { result::graph::destroy(self.0) };
        }
    }
}

/// A shared hold of the crate's graph-capture lock ([`crate::CAPTURE`]),
/// taken around every driver call CUDA refuses while another thread of the
/// process captures a graph in global mode, as CubeCL's captures do (the
/// refused call also invalidates that capture): allocations, frees, copies,
/// waits, module loads, and around our own captures' opening, closing and
/// instantiation and the replays. Launches into the device's own stream are
/// allowed and do not take it. Never taken twice on one thread (a waiting
/// writer would deadlock the second): a recording runs without it. Goes
/// with CubeCL (NATIVE_GPU_PLAN.md, stage N4).
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
            capturing: AtomicBool::new(false),
            allocations: AtomicUsize::new(0),
        })
    }

    /// Blocks allocated over the device's life: constant once a solver's
    /// buffers exist, which is what the preallocation rule checks.
    pub fn allocations(&self) -> usize {
        self.allocations.load(Ordering::Relaxed)
    }

    /// [`NativeError::Capture`] while a capture is open: `op` would be
    /// recorded into it or would break it.
    fn not_capturing(&self, op: &str) -> Result<(), NativeError> {
        if self.capturing.load(Ordering::Acquire) {
            return Err(NativeError::Capture(format!("{op} while a capture is open on the device")));
        }
        Ok(())
    }

    /// The module of `source`: in memory when this device loaded it before,
    /// else from the disk cache when an earlier process compiled it, else
    /// compiled (`entry` names it in the compiler's messages). A cached
    /// entry the driver does not load is compiled again and written over.
    fn module(&self, entry: &str, source: String) -> Result<Arc<CudaModule>, NativeError> {
        self.module_in(entry, source, DiskCache::open("cuda").as_ref())
    }

    /// [`module`](Self::module) with the disk cache `disk`.
    fn module_in(&self, entry: &str, source: String, disk: Option<&DiskCache>) -> Result<Arc<CudaModule>, NativeError> {
        if let Some(module) = self.modules.lock().unwrap().get(&source) {
            return Ok(module.clone());
        }
        let opts = self.compile_options();
        let key = self.cache_key(&opts, &source)?;
        let cached = disk.and_then(|d| d.load(key.as_bytes())).and_then(|cubin| self.load(cubin).ok());
        let module = match cached {
            Some(module) => module,
            None => {
                let cubin = compile_cubin(entry, &source, &opts)?;
                if let Some(d) = disk {
                    d.store(key.as_bytes(), &cubin);
                }
                self.load(cubin)?
            }
        };
        self.modules.lock().unwrap().insert(source, module.clone());
        Ok(module)
    }

    /// What a cubin of `source` compiled with `opts` depends on.
    fn cache_key(&self, opts: &[String], source: &str) -> Result<String, NativeError> {
        Ok(format!("{}\nnvrtc {}\n{source}", opts.join(" "), nvrtc_version()?))
    }

    /// A cubin loaded as a module.
    fn load(&self, cubin: Vec<u8>) -> Result<Arc<CudaModule>, NativeError> {
        let _shared = capture_guard();
        self.ctx.load_module(Ptx::from_binary(cubin)).map_err(driver)
    }

    /// NVRTC's options: the device's architecture, so the output is its
    /// machine code (a cubin, which loads without the driver's JIT), and
    /// NVRTC's defaults stated: contraction on, correctly rounded division
    /// and square root, subnormals kept (the plan's "default math").
    fn compile_options(&self) -> Vec<String> {
        [&*format!("--gpu-architecture={}", self.arch.target), "--std=c++17", "--fmad=true", "--prec-div=true", "--prec-sqrt=true", "--ftz=false"]
            .map(String::from)
            .to_vec()
    }
}

/// NVRTC's version, `major.minor`.
fn nvrtc_version() -> Result<String, NativeError> {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: two out-parameters.
    unsafe { nvrtc::sys::nvrtcVersion(&mut major, &mut minor) }.result().map_err(|e| NativeError::Unavailable(format!("NVRTC: {e:?}")))?;
    Ok(format!("{major}.{minor}"))
}

/// NVRTC compilations in this process.
static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Kernel sources NVRTC has compiled in this process: those neither a
/// device's memory nor the disk cache held.
pub fn compiles() -> usize {
    COMPILES.load(Ordering::Relaxed)
}

/// `source` compiled with `opts` by NVRTC to a cubin.
fn compile_cubin(entry: &str, source: &str, opts: &[String]) -> Result<Vec<u8>, NativeError> {
    use nvrtc::{result, sys};
    COMPILES.fetch_add(1, Ordering::Relaxed);
    let failed = |what: &str, e: nvrtc::result::NvrtcError| NativeError::Compile { entry: entry.to_string(), log: format!("{what}: {e:?}") };
    let src = std::ffi::CString::new(source).map_err(|_| NativeError::Compile { entry: entry.to_string(), log: "a NUL byte in the source".into() })?;
    let name = std::ffi::CString::new(format!("{entry}.cu")).expect("an entry name without NUL");
    let prog = result::create_program(&src, Some(&name)).map_err(|e| failed("create the program", e))?;
    /// Destroys the program on every way out.
    struct Program(sys::nvrtcProgram);
    impl Drop for Program {
        fn drop(&mut self) {
            // SAFETY: created above, destroyed once.
            let _ = unsafe { result::destroy_program(self.0) };
        }
    }
    let prog = Program(prog);
    // SAFETY: a live program; `src` and `name` outlive it.
    if let Err(e) = unsafe { result::compile_program(prog.0, opts) } {
        // SAFETY: as above.
        let log = unsafe { result::get_program_log(prog.0) }.map_err(|e| failed("read the compile log", e))?;
        // SAFETY: NVRTC's log is NUL-terminated.
        let log = unsafe { std::ffi::CStr::from_ptr(log.as_ptr()) }.to_string_lossy().into_owned();
        return Err(NativeError::Compile { entry: entry.to_string(), log: format!("{e:?}\n{log}") });
    }
    let mut size = 0usize;
    // SAFETY: a compiled program and an out-parameter.
    unsafe { sys::nvrtcGetCUBINSize(prog.0, &mut size) }.result().map_err(|e| failed("size the cubin", e))?;
    let mut cubin = vec![0u8; size];
    // SAFETY: `cubin` holds the size NVRTC asked for.
    unsafe { sys::nvrtcGetCUBIN(prog.0, cubin.as_mut_ptr().cast()) }.result().map_err(|e| failed("read the cubin", e))?;
    Ok(cubin)
}

impl Device for CudaDevice {
    type Block = CudaBlock;
    type Kernel = CudaKernel;
    type Graph = CudaGraph;

    fn arch(&self) -> &Arch {
        &self.arch
    }

    fn alloc(&self, bytes: usize) -> Result<CudaBlock, NativeError> {
        self.not_capturing("an allocation")?;
        let _shared = capture_guard();
        let bytes = ManuallyDrop::new(self.stream.alloc_zeros::<u8>(bytes).map_err(driver)?);
        self.allocations.fetch_add(1, Ordering::Relaxed);
        Ok(CudaBlock { bytes })
    }

    fn write(&self, dst: &mut CudaBlock, offset: usize, src: &[u8]) -> Result<(), NativeError> {
        self.not_capturing("a write")?;
        let mut view = dst.bytes.slice_mut(offset..offset + src.len());
        let _shared = capture_guard();
        self.stream.memcpy_htod(src, &mut view).map_err(driver)
    }

    fn read(&self, src: &CudaBlock, offset: usize, dst: &mut [u8]) -> Result<(), NativeError> {
        self.not_capturing("a read")?;
        let view = src.bytes.slice(offset..offset + dst.len());
        let _shared = capture_guard();
        self.stream.memcpy_dtoh(&view, dst).map_err(driver)?;
        self.stream.synchronize().map_err(driver)
    }

    fn read_many(&self, reads: &mut [(&CudaBlock, usize, &mut [u8])]) -> Result<(), NativeError> {
        self.not_capturing("a read")?;
        let _shared = capture_guard();
        for (src, offset, dst) in reads.iter_mut() {
            let view = src.bytes.slice(*offset..*offset + dst.len());
            self.stream.memcpy_dtoh(&view, &mut **dst).map_err(driver)?;
        }
        self.stream.synchronize().map_err(driver)
    }

    fn compile(&self, source: &str, defines: &str, entry: &str) -> Result<CudaKernel, NativeError> {
        self.not_capturing("a compile")?;
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
                Arg::Bits(v) => v,
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
        self.not_capturing("a wait")?;
        let _shared = capture_guard();
        self.stream.synchronize().map_err(driver)
    }

    /// Stream capture in thread-local mode, instantiated and uploaded. The
    /// capture lock is held shared around the driver calls that open, close
    /// and instantiate the capture, as around every other non-launch call,
    /// but not while `record` runs: a thread-local capture forbids nothing on
    /// other threads, and `record` only launches.
    fn capture(&self, record: impl FnOnce() -> Result<(), NativeError>) -> Result<CudaGraph, NativeError> {
        let open = OpenCapture::begin(self)?;
        let recorded = record();
        let graph = open.end();
        // the recording's error first: it is why the capture broke, if it did
        recorded?;
        let graph = graph?;
        if graph.0.is_null() {
            return Err(NativeError::Capture("the capture recorded nothing".into()));
        }
        // (A `CudaGraph` drops outside the guards below: its drop takes one.)
        let exec = {
            let _shared = capture_guard();
            self.ctx.bind_to_thread().map_err(driver)?;
            // SAFETY: a valid graph from end_capture; the executable graph is
            // independent of it.
            unsafe { result::graph::instantiate(graph.0, sys::CUgraphInstantiate_flags(0)) }.map_err(driver)?
        };
        drop(graph);
        let graph = CudaGraph { exec, ctx: (*self.ctx).clone() };
        {
            let _shared = capture_guard();
            // the first replay then pays no setup
            // SAFETY: the executable graph and the stream are alive.
            unsafe { result::graph::upload(graph.exec, self.stream.cu_stream()) }.map_err(driver)?;
        }
        Ok(graph)
    }

    fn replay(&self, graph: &CudaGraph) -> Result<(), NativeError> {
        self.not_capturing("a replay")?;
        assert!(Arc::ptr_eq(&graph.ctx, &self.ctx), "a graph replays on the context it was captured on");
        let _shared = capture_guard();
        self.ctx.bind_to_thread().map_err(driver)?;
        // SAFETY: the executable graph is alive; its blocks are the caller's
        // to keep alive (the capture's contract).
        unsafe { result::graph::launch(graph.exec, self.stream.cu_stream()) }.map_err(driver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A cached entry whose binary the driver refuses (not a cubin at all,
    /// its length and checksum right) is compiled again and written over.
    #[test]
    fn a_cached_binary_the_driver_refuses_is_compiled_again() {
        let dir = std::env::temp_dir().join(format!("arael-cuda-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = DiskCache::at(dir.clone());
        let dev = CudaDevice::new(0).unwrap();
        let source = format!("{}\n{HEADER}\nKERNEL void k(GLOBAL float* x) {{ x[0] = 1.0f; }}", dev.arch.defines());
        let key = dev.cache_key(&dev.compile_options(), &source).unwrap();
        disk.store(key.as_bytes(), b"not a cubin");
        assert_eq!(disk.load(key.as_bytes()).as_deref(), Some(&b"not a cubin"[..]));
        let module = dev.module_in("k", source.clone(), Some(&disk)).unwrap();
        module.load_function("k").unwrap();
        let cubin = disk.load(key.as_bytes()).expect("the entry written over");
        assert_ne!(cubin, b"not a cubin");
        // which a device without it in memory loads
        CudaDevice::new(0).unwrap().load(cubin).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
