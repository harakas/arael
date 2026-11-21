//! [`Device`] on HIP: the HIP runtime and hipRTC through our own bindings
//! (`hip_sys`), both loaded at run time.

use crate::disk::DiskCache;
use crate::hip_sys::{rt, rtc};
use crate::pool::{Pool, capacity};
use crate::{Arch, Arg, Device, GpuError, Lane, Launch, MAX_ARGS};
use std::collections::HashMap;
use std::ffi::{CStr, CString, c_void};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

/// The HIP runtime and hipRTC, loaded once per process.
struct Libs {
    rt: rt::HipRuntime,
    rtc: rtc::HipRtc,
}

/// The first of `names` the loader finds.
fn load<L>(names: &[&str], open: impl Fn(&str) -> Result<L, libloading::Error>) -> Result<L, String> {
    let mut why = Vec::new();
    for name in names {
        match open(name) {
            Ok(lib) => return Ok(lib),
            Err(e) => why.push(format!("{name}: {e}")),
        }
    }
    Err(why.join("; "))
}

fn libs() -> Result<&'static Libs, GpuError> {
    static LIBS: OnceLock<Result<Libs, String>> = OnceLock::new();
    LIBS.get_or_init(|| {
        // SAFETY: loading the vendor's libraries runs their initializers, as
        // linking them would; every symbol the bindings name is resolved here.
        let rt = load(&["libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6"], |n| unsafe { rt::HipRuntime::new(n) })?;
        let rtc = load(&["libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6"], |n| unsafe { rtc::HipRtc::new(n) })?;
        Ok(Libs { rt, rtc })
    })
    .as_ref()
    .map_err(|e| GpuError::Unavailable(format!("the HIP runtime or hipRTC is not installed: {e}")))
}

/// `e` as a result, `what` naming the call.
fn check(l: &Libs, e: rt::hipError_t, what: &str) -> Result<(), GpuError> {
    if e == rt::hipError_t_hipSuccess {
        return Ok(());
    }
    // SAFETY: the runtime returns a static string for any error code.
    let name = unsafe { CStr::from_ptr(l.rt.hipGetErrorName(e)) }.to_string_lossy();
    Err(GpuError::Driver(format!("{what}: {name}")))
}

fn check_rtc(l: &Libs, e: rtc::hiprtcResult, entry: &str, what: &str) -> Result<(), GpuError> {
    if e == rtc::hiprtcResult_HIPRTC_SUCCESS {
        return Ok(());
    }
    // SAFETY: hipRTC returns a static string for any result code.
    let name = unsafe { CStr::from_ptr(l.rtc.hiprtcGetErrorString(e)) }.to_string_lossy();
    Err(GpuError::Compile { entry: entry.to_string(), log: format!("{what}: {name}") })
}

/// `hipStreamNonBlocking`: the stream does not wait for the legacy default stream.
const STREAM_NON_BLOCKING: u32 = 1;

/// One HIP device, a stream of its own and a side stream
/// ([`Device::launch_on`]). HIP's current device is per thread, so every
/// call sets it first.
pub struct HipDevice {
    l: &'static Libs,
    index: i32,
    stream: rt::hipStream_t,
    /// At the device's highest priority, so its blocks start as soon as the
    /// main stream's leave room.
    side: rt::hipStream_t,
    /// Recorded on the main stream and waited for by the side at a fork.
    fork_event: rt::hipEvent_t,
    /// Recorded on the side stream and waited for by the main at a join.
    join_event: rt::hipEvent_t,
    /// The side stream's forks open.
    forks: crate::Forks,
    arch: Arch,
    /// Compiled modules by their full source.
    modules: Mutex<HashMap<String, Arc<HipModule>>>,
    /// A graph capture is open on the stream.
    capturing: AtomicBool,
    /// Launches are timed ([`Device::time_launches`]).
    timing: AtomicBool,
    timer: Mutex<Timer>,
    /// The stamps' events ([`Device::stamp`]), each made at its first use;
    /// null until then.
    stamps: Mutex<[rt::hipEvent_t; crate::STAMPS]>,
    /// The device memory the blocks come from and go back to.
    pool: Arc<Pool<HipMem>>,
    /// The pinned host memory the host blocks come from and go back to.
    host_pool: Arc<Pool<PinnedMem>>,
}

// SAFETY: the stream is a runtime handle usable from any thread (each call
// sets the device first); operations on it are ordered by the runtime.
unsafe impl Send for HipDevice {}
unsafe impl Sync for HipDevice {}

impl core::fmt::Debug for HipDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HipDevice").field("arch", &self.arch).finish_non_exhaustive()
    }
}

impl Drop for HipDevice {
    fn drop(&mut self) {
        // SAFETY: the streams and events were made in `new` and are destroyed
        // once; an error here has nobody to go to.
        unsafe {
            let _ = self.l.rt.hipSetDevice(self.index);
            let _ = self.l.rt.hipStreamSynchronize(self.stream);
            let _ = self.l.rt.hipStreamSynchronize(self.side);
            let _ = self.l.rt.hipEventDestroy(self.fork_event);
            let _ = self.l.rt.hipEventDestroy(self.join_event);
            let timer = self.timer.get_mut().unwrap_or_else(|e| e.into_inner());
            let _ = self.l.rt.hipEventDestroy(timer.start);
            let _ = self.l.rt.hipEventDestroy(timer.end);
            for &stamp in self.stamps.get_mut().unwrap_or_else(|e| e.into_inner()).iter().filter(|s| !s.is_null()) {
                let _ = self.l.rt.hipEventDestroy(stamp);
            }
            let _ = self.l.rt.hipStreamDestroy(self.side);
            let _ = self.l.rt.hipStreamDestroy(self.stream);
        }
    }
}

/// Device memory from hipMalloc, freed on drop.
struct HipMem {
    ptr: *mut c_void,
    l: &'static Libs,
    index: i32,
}

// SAFETY: a device address, valid on any thread of the process.
unsafe impl Send for HipMem {}
unsafe impl Sync for HipMem {}

impl Drop for HipMem {
    fn drop(&mut self) {
        // SAFETY: allocated by hipMalloc in `alloc`, freed once.
        unsafe {
            let _ = self.l.rt.hipSetDevice(self.index);
            let _ = self.l.rt.hipFree(self.ptr);
        }
    }
}

/// Device memory on a [`HipDevice`], from its pool and back to it on drop.
pub struct HipBlock {
    /// Some until the drop; its own size, `cap`, may exceed `len`.
    mem: Option<HipMem>,
    cap: usize,
    len: usize,
    pool: Arc<Pool<HipMem>>,
}

impl HipBlock {
    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn ptr(&self) -> *mut c_void {
        self.mem.as_ref().expect("a block's memory until its drop").ptr
    }
}

impl Drop for HipBlock {
    fn drop(&mut self) {
        if let Some(mem) = self.mem.take() {
            self.pool.give(self.cap, mem);
        }
    }
}

/// A pinned host allocation and the event recorded after each copy from
/// it, freed when its pool goes.
struct PinnedMem {
    ptr: *mut u8,
    event: rt::hipEvent_t,
    l: &'static Libs,
    index: i32,
}

// SAFETY: page-locked host memory and an event handle, valid on any thread
// of the process; writes to the memory go through `host_mut`, which waits
// for the copies from it.
unsafe impl Send for PinnedMem {}
unsafe impl Sync for PinnedMem {}

impl Drop for PinnedMem {
    fn drop(&mut self) {
        // SAFETY: allocated by hipHostMalloc and hipEventCreateWithFlags in
        // `alloc_host`, freed once, after the copies from it completed.
        unsafe {
            let _ = self.l.rt.hipSetDevice(self.index);
            let _ = self.l.rt.hipEventSynchronize(self.event);
            let _ = self.l.rt.hipEventDestroy(self.event);
            let _ = self.l.rt.hipHostFree(self.ptr.cast());
        }
    }
}

/// Pinned host memory on a [`HipDevice`] ([`Device::alloc_host`]), from its
/// pool and back to it on drop.
pub struct HipHost {
    /// Some until the drop; its own size, `cap`, may exceed `len`.
    mem: Option<PinnedMem>,
    cap: usize,
    len: usize,
    pool: Arc<Pool<PinnedMem>>,
}

impl HipHost {
    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn mem(&self) -> &PinnedMem {
        self.mem.as_ref().expect("a host block's memory until its drop")
    }
}

impl Drop for HipHost {
    fn drop(&mut self) {
        // copies from it may still run: the next taker waits for them
        if let Some(mem) = self.mem.take() {
            self.pool.give(self.cap, mem);
        }
    }
}

/// A loaded code object, unloaded when its last kernel goes.
struct HipModule {
    module: rt::hipModule_t,
    l: &'static Libs,
    index: i32,
}

// SAFETY: a runtime handle, usable from any thread.
unsafe impl Send for HipModule {}
unsafe impl Sync for HipModule {}

impl Drop for HipModule {
    fn drop(&mut self) {
        // SAFETY: loaded by hipModuleLoadData, unloaded once.
        unsafe {
            let _ = self.l.rt.hipSetDevice(self.index);
            let _ = self.l.rt.hipModuleUnload(self.module);
        }
    }
}

/// A compiled entry point on a [`HipDevice`].
pub struct HipKernel {
    function: rt::hipFunction_t,
    module: Arc<HipModule>,
}

// SAFETY: a runtime handle, usable from any thread; its module lives as long.
unsafe impl Send for HipKernel {}
unsafe impl Sync for HipKernel {}

impl core::fmt::Debug for HipKernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HipKernel").finish_non_exhaustive()
    }
}

impl HipKernel {
    fn attribute(&self, attr: rt::hipFunction_attribute, what: &str) -> Result<usize, GpuError> {
        let mut v = 0;
        let l = self.module.l;
        // SAFETY: a live function and an out-parameter.
        check(l, unsafe { l.rt.hipFuncGetAttribute(&mut v, attr, self.function) }, what)?;
        Ok(v as usize)
    }

    /// Scratch (private) memory per thread, in bytes: per-thread arrays the
    /// compiler could not keep in registers, and spills.
    pub fn local_bytes(&self) -> Result<usize, GpuError> {
        self.attribute(rt::hipFunction_attribute_HIP_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, "hipFuncGetAttribute(local size)")
    }

    /// Registers per thread.
    pub fn registers(&self) -> Result<usize, GpuError> {
        self.attribute(rt::hipFunction_attribute_HIP_FUNC_ATTRIBUTE_NUM_REGS, "hipFuncGetAttribute(registers)")
    }
}

/// A launch as the runtime takes it: what a capture made while timing
/// keeps, to issue again at each replay.
struct TimedLaunch {
    function: rt::hipFunction_t,
    // Keeps the function's module loaded.
    _module: Arc<HipModule>,
    lane: Lane,
    launch: Launch,
    values: [u64; MAX_ARGS],
    tables: Tables,
}

/// A launch's [`Arg::Bytes`] arguments: (argument index, its bytes in
/// aligned words). Empty, and not allocated, for a launch without one.
type Tables = Vec<(usize, Vec<u64>)>;

/// The timing of launches ([`Device::time_launches`]).
struct Timer {
    /// Recorded on a launch's stream before and after it.
    start: rt::hipEvent_t,
    end: rt::hipEvent_t,
    /// Seconds per launch since the last [`Device::take_times`].
    times: Vec<f64>,
    /// The launches of the capture open while timing.
    capture: Option<Vec<TimedLaunch>>,
}

/// A captured graph on a [`HipDevice`], instantiated for replay; or, when
/// captured while timing, its launches (`exec` null).
pub struct HipGraph {
    exec: rt::hipGraphExec_t,
    timed: Vec<TimedLaunch>,
    l: &'static Libs,
    index: i32,
}

// SAFETY: the executable graph is a runtime handle, usable from any thread;
// not `Sync`, as for CUDA.
unsafe impl Send for HipGraph {}

impl Drop for HipGraph {
    fn drop(&mut self) {
        if self.exec.is_null() {
            return;
        }
        // SAFETY: made by hipGraphInstantiate, destroyed once.
        unsafe {
            let _ = self.l.rt.hipSetDevice(self.index);
            let _ = self.l.rt.hipGraphExecDestroy(self.exec);
        }
    }
}

/// A capture open on a stream, ended on drop if [`OpenCapture::end`] was
/// not reached, so the stream is usable again.
struct OpenCapture<'a> {
    dev: &'a HipDevice,
    open: bool,
}

impl<'a> OpenCapture<'a> {
    fn begin(dev: &'a HipDevice) -> Result<Self, GpuError> {
        if dev.capturing.swap(true, Ordering::AcqRel) {
            return Err(GpuError::Capture("a capture is already open on this device".into()));
        }
        let mut open = OpenCapture { dev, open: false };
        dev.bind()?;
        // Thread-local: the capture forbids unsafe calls on this thread only.
        // SAFETY: the stream is the device's own and alive.
        let e = unsafe { dev.l.rt.hipStreamBeginCapture(dev.stream, rt::hipStreamCaptureMode_hipStreamCaptureModeThreadLocal) };
        check(dev.l, e, "hipStreamBeginCapture")?;
        open.open = true;
        Ok(open)
    }

    /// Ends the capture; its graph, or null when nothing was recorded.
    fn end(mut self) -> Result<RawGraph, GpuError> {
        self.open = false;
        self.dev.bind()?;
        let mut graph = core::ptr::null_mut();
        // SAFETY: the capture was begun on this stream by `begin`.
        check(self.dev.l, unsafe { self.dev.l.rt.hipStreamEndCapture(self.dev.stream, &mut graph) }, "hipStreamEndCapture")?;
        Ok(RawGraph(graph, self.dev.l))
    }
}

impl Drop for OpenCapture<'_> {
    fn drop(&mut self) {
        if self.open {
            if self.dev.bind().is_ok() {
                let mut graph = core::ptr::null_mut();
                // SAFETY: as in `end`; the partial graph is discarded.
                if unsafe { self.dev.l.rt.hipStreamEndCapture(self.dev.stream, &mut graph) } == rt::hipError_t_hipSuccess {
                    drop(RawGraph(graph, self.dev.l));
                }
            }
        }
        self.dev.capturing.store(false, Ordering::Release);
    }
}

/// A captured, not instantiated graph, destroyed on drop.
struct RawGraph(rt::hipGraph_t, &'static Libs);

impl Drop for RawGraph {
    fn drop(&mut self) {
        if !self.0.is_null() {
            // SAFETY: the handle came from hipStreamEndCapture and is destroyed once.
            let _ = unsafe { self.1.rt.hipGraphDestroy(self.0) };
        }
    }
}

// `launch` passes scalars in the low bytes of 8-byte slots.
const _: () = assert!(cfg!(target_endian = "little"), "the HIP runtime assumes a little-endian host");

/// hipRTC compilations in this process.
static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Kernel sources hipRTC has compiled in this process: those neither a
/// device's memory nor the disk cache held.
pub fn compiles() -> usize {
    COMPILES.load(Ordering::Relaxed)
}

/// A C string of `s`, which holds no NUL.
fn c_string(s: &str, entry: &str) -> Result<CString, GpuError> {
    CString::new(s).map_err(|_| GpuError::Compile { entry: entry.to_string(), log: "a NUL byte in the source".into() })
}

/// `source` compiled with `opts` by hipRTC to a code object.
fn compile_code(l: &Libs, entry: &str, source: &str, opts: &[String]) -> Result<Vec<u8>, GpuError> {
    COMPILES.fetch_add(1, Ordering::Relaxed);
    let src = c_string(source, entry)?;
    let name = c_string(&format!("{entry}.hip"), entry)?;
    let mut prog = core::ptr::null_mut();
    // SAFETY: valid C strings; no headers.
    let e = unsafe { l.rtc.hiprtcCreateProgram(&mut prog, src.as_ptr(), name.as_ptr(), 0, core::ptr::null(), core::ptr::null()) };
    check_rtc(l, e, entry, "create the program")?;
    /// Destroys the program on every way out.
    struct Program<'a>(rtc::hiprtcProgram, &'a Libs);
    impl Drop for Program<'_> {
        fn drop(&mut self) {
            // SAFETY: created above, destroyed once.
            let _ = unsafe { self.1.rtc.hiprtcDestroyProgram(&mut self.0) };
        }
    }
    let prog = Program(prog, l);
    let copts: Vec<CString> = opts.iter().map(|o| c_string(o, entry)).collect::<Result<_, _>>()?;
    let ptrs: Vec<*const core::ffi::c_char> = copts.iter().map(|o| o.as_ptr()).collect();
    // SAFETY: a live program; the options outlive the call.
    let compiled = unsafe { l.rtc.hiprtcCompileProgram(prog.0, ptrs.len() as i32, ptrs.as_ptr()) };
    if compiled != rtc::hiprtcResult_HIPRTC_SUCCESS {
        let mut size = 0usize;
        // SAFETY: a live program and out-parameters sized by hipRTC.
        let log = unsafe {
            let _ = l.rtc.hiprtcGetProgramLogSize(prog.0, &mut size);
            let mut log = vec![0u8; size.max(1)];
            let _ = l.rtc.hiprtcGetProgramLog(prog.0, log.as_mut_ptr().cast());
            String::from_utf8_lossy(&log[..size.saturating_sub(1)]).into_owned()
        };
        // SAFETY: as in check_rtc.
        let what = unsafe { CStr::from_ptr(l.rtc.hiprtcGetErrorString(compiled)) }.to_string_lossy();
        return Err(GpuError::Compile { entry: entry.to_string(), log: format!("{what}\n{log}") });
    }
    let mut size = 0usize;
    // SAFETY: a compiled program and an out-parameter.
    check_rtc(l, unsafe { l.rtc.hiprtcGetCodeSize(prog.0, &mut size) }, entry, "size the code object")?;
    let mut code = vec![0u8; size];
    // SAFETY: `code` holds the size hipRTC asked for.
    check_rtc(l, unsafe { l.rtc.hiprtcGetCode(prog.0, code.as_mut_ptr().cast()) }, entry, "read the code object")?;
    Ok(code)
}

/// hipRTC's version, `major.minor`.
fn hiprtc_version(l: &Libs) -> Result<String, GpuError> {
    let (mut major, mut minor) = (0, 0);
    // SAFETY: two out-parameters.
    check_rtc(l, unsafe { l.rtc.hiprtcVersion(&mut major, &mut minor) }, "", "hiprtcVersion")?;
    Ok(format!("{major}.{minor}"))
}

/// A fixed-size C string field as a Rust string.
fn field(chars: &[core::ffi::c_char]) -> String {
    let bytes: Vec<u8> = chars.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

/// The HIP devices of this machine, by index, without opening one.
/// [`GpuError::Unavailable`] when the HIP runtime or hipRTC is not
/// installed, or the runtime does not start.
///
/// ```ignore
/// let discrete = arael_gpu::hip::devices()?.into_iter().find(|d| !d.integrated);
/// ```
pub fn devices() -> Result<Vec<crate::DeviceInfo>, GpuError> {
    let l = libs()?;
    let unavailable = |e: GpuError| match e {
        GpuError::Driver(why) => GpuError::Unavailable(format!("the HIP runtime does not start ({why})")),
        e => e,
    };
    // SAFETY: plain runtime calls with out-parameters.
    unsafe {
        check(l, l.rt.hipInit(0), "hipInit").map_err(unavailable)?;
        let mut count = 0;
        check(l, l.rt.hipGetDeviceCount(&mut count), "hipGetDeviceCount").map_err(unavailable)?;
        (0..count)
            .map(|i| {
                let mut prop: rt::hipDeviceProp_tR0600 = core::mem::zeroed();
                check(l, l.rt.hipGetDevicePropertiesR0600(&mut prop, i), "hipGetDeviceProperties")?;
                Ok(crate::DeviceInfo { index: i as usize, name: field(&prop.name), integrated: prop.integrated != 0 })
            })
            .collect()
    }
}

impl HipDevice {
    /// Device `index`. [`GpuError::Unavailable`] when there is no such
    /// device or the HIP runtime or hipRTC is not installed.
    pub fn new(index: usize) -> Result<Self, GpuError> {
        let l = libs()?;
        let unavailable = |e: GpuError| GpuError::Unavailable(format!("HIP device {index}: {e}"));
        // SAFETY: plain runtime calls with out-parameters.
        unsafe {
            check(l, l.rt.hipInit(0), "hipInit").map_err(unavailable)?;
            let mut count = 0;
            check(l, l.rt.hipGetDeviceCount(&mut count), "hipGetDeviceCount").map_err(unavailable)?;
            if index >= count as usize {
                return Err(GpuError::Unavailable(format!("HIP device {index}: the system has {count}")));
            }
            let index = index as i32;
            check(l, l.rt.hipSetDevice(index), "hipSetDevice")?;
            let mut prop: rt::hipDeviceProp_tR0600 = core::mem::zeroed();
            check(l, l.rt.hipGetDevicePropertiesR0600(&mut prop, index), "hipGetDeviceProperties")?;
            let mut stream = core::ptr::null_mut();
            check(l, l.rt.hipStreamCreateWithFlags(&mut stream, STREAM_NON_BLOCKING), "hipStreamCreateWithFlags")?;
            let (mut least, mut highest) = (0, 0);
            check(l, l.rt.hipDeviceGetStreamPriorityRange(&mut least, &mut highest), "hipDeviceGetStreamPriorityRange")?;
            let mut side = core::ptr::null_mut();
            check(l, l.rt.hipStreamCreateWithPriority(&mut side, STREAM_NON_BLOCKING, highest), "hipStreamCreateWithPriority")?;
            let (mut fork_event, mut join_event) = (core::ptr::null_mut(), core::ptr::null_mut());
            check(l, l.rt.hipEventCreateWithFlags(&mut fork_event, rt::hipEventDisableTiming), "hipEventCreateWithFlags")?;
            check(l, l.rt.hipEventCreateWithFlags(&mut join_event, rt::hipEventDisableTiming), "hipEventCreateWithFlags")?;
            // the timer's events keep their timing (no flag)
            let (mut start, mut end) = (core::ptr::null_mut(), core::ptr::null_mut());
            check(l, l.rt.hipEventCreateWithFlags(&mut start, 0), "hipEventCreateWithFlags")?;
            check(l, l.rt.hipEventCreateWithFlags(&mut end, 0), "hipEventCreateWithFlags")?;
            // the code object's target: the architecture without its feature flags
            let gcn = field(&prop.gcnArchName);
            let target = gcn.split(':').next().unwrap_or(&gcn).to_string();
            let arch = Arch {
                name: field(&prop.name),
                vendor: crate::Vendor::Amd,
                target,
                warp: prop.warpSize as u32,
                shared_bytes: prop.sharedMemPerBlock as u32,
                multiprocessors: prop.multiProcessorCount as u32,
                has_f64: true,
                flushes_subnormals: false,
                integrated: prop.integrated != 0,
            };
            Ok(HipDevice {
                l,
                index,
                stream,
                side,
                fork_event,
                join_event,
                forks: crate::Forks::default(),
                arch,
                modules: Mutex::new(HashMap::new()),
                capturing: AtomicBool::new(false),
                timing: AtomicBool::new(false),
                timer: Mutex::new(Timer { start, end, times: Vec::new(), capture: None }),
                stamps: Mutex::new([core::ptr::null_mut(); crate::STAMPS]),
                pool: Arc::new(Pool::new()),
                host_pool: Arc::new(Pool::new()),
            })
        }
    }

    /// This thread's current device made this one.
    fn bind(&self) -> Result<(), GpuError> {
        // SAFETY: a plain runtime call.
        check(self.l, unsafe { self.l.rt.hipSetDevice(self.index) }, "hipSetDevice")
    }

    fn timer(&self) -> std::sync::MutexGuard<'_, Timer> {
        self.timer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Queues `function` on `stream` with the argument slots `values`.
    fn issue(&self, function: rt::hipFunction_t, stream: rt::hipStream_t, launch: Launch, values: &mut [u64; MAX_ARGS], tables: &mut Tables) -> Result<(), GpuError> {
        let mut params = [core::ptr::null_mut::<c_void>(); MAX_ARGS];
        for (p, v) in params.iter_mut().zip(values.iter_mut()) {
            *p = v as *mut u64 as *mut c_void;
        }
        // a table's parameter points at its words: the runtime copies as many
        // bytes as the kernel's parameter has
        for (i, words) in tables.iter_mut() {
            params[*i] = words.as_mut_ptr() as *mut c_void;
        }
        let [gx, gy, gz] = launch.grid;
        let [bx, by, bz] = launch.block;
        self.bind()?;
        // SAFETY: the kernel's parameters are the caller's to match; the
        // argument slots live until the call returns, and the launch only queues.
        let e = unsafe { self.l.rt.hipModuleLaunchKernel(function, gx, gy, gz, bx, by, bz, launch.shared_bytes, stream, params.as_mut_ptr(), core::ptr::null_mut()) };
        check(self.l, e, "hipModuleLaunchKernel")
    }

    /// `launch` alone on its stream, between two events whose distance is
    /// its time on the device.
    fn run_timed(&self, timer: &mut Timer, launch: &TimedLaunch) -> Result<(), GpuError> {
        let stream = match launch.lane {
            Lane::Main => self.stream,
            Lane::Side => self.side,
        };
        self.bind()?;
        // SAFETY: the device's stream and the timer's events.
        check(self.l, unsafe { self.l.rt.hipEventRecord(timer.start, stream) }, "hipEventRecord")?;
        let mut values = launch.values;
        let mut tables = launch.tables.clone();
        self.issue(launch.function, stream, launch.launch, &mut values, &mut tables)?;
        let mut ms = 0f32;
        // SAFETY: as above; the elapsed time is asked once the end event is reached.
        unsafe {
            check(self.l, self.l.rt.hipEventRecord(timer.end, stream), "hipEventRecord")?;
            check(self.l, self.l.rt.hipEventSynchronize(timer.end), "hipEventSynchronize")?;
            check(self.l, self.l.rt.hipEventElapsedTime(&mut ms, timer.start, timer.end), "hipEventElapsedTime")?;
        }
        timer.times.push(ms as f64 * 1e-3);
        Ok(())
    }

    /// A capture while timing: `record`'s launches kept, not run; the
    /// graph's replays issue them one by one.
    fn capture_timed(&self, record: impl FnOnce() -> Result<(), GpuError>) -> Result<HipGraph, GpuError> {
        /// The capture closed again, a failed or panicking recording included.
        struct Open<'a>(&'a HipDevice);
        impl Drop for Open<'_> {
            fn drop(&mut self) {
                self.0.timer().capture = None;
                self.0.capturing.store(false, Ordering::Release);
            }
        }
        if self.capturing.swap(true, Ordering::AcqRel) {
            return Err(GpuError::Capture("a capture is already open on this device".into()));
        }
        let open = Open(self);
        let forks = self.forks.open();
        self.timer().capture = Some(Vec::new());
        let recorded = record();
        let timed = self.timer().capture.take().unwrap_or_default();
        drop(open);
        let joined = self.forks.check_capture(forks);
        recorded?;
        joined?;
        if timed.is_empty() {
            return Err(GpuError::Capture("the capture recorded nothing".into()));
        }
        Ok(HipGraph { exec: core::ptr::null_mut(), timed, l: self.l, index: self.index })
    }

    /// [`GpuError::Capture`] while a capture is open.
    fn not_capturing(&self, op: &str) -> Result<(), GpuError> {
        if self.capturing.load(Ordering::Acquire) {
            return Err(GpuError::Capture(format!("{op} while a capture is open on the device")));
        }
        Ok(())
    }

    /// hipRTC's options: the device's architecture, and hipRTC's math
    /// defaults (contraction on, IEEE otherwise).
    fn compile_options(&self) -> Vec<String> {
        vec![format!("--offload-arch={}", self.arch.target), "-std=c++17".into(), "-O3".into()]
    }

    /// The module of `source`: in memory when this device loaded it before,
    /// else from the disk cache when an earlier process compiled it, else
    /// compiled. A cached entry the runtime does not load is compiled again
    /// and written over.
    fn module(&self, entry: &str, source: String) -> Result<Arc<HipModule>, GpuError> {
        self.module_in(entry, source, DiskCache::open("hip").as_ref())
    }

    /// [`module`](Self::module) with the disk cache `disk`.
    fn module_in(&self, entry: &str, source: String, disk: Option<&DiskCache>) -> Result<Arc<HipModule>, GpuError> {
        if let Some(module) = self.modules.lock().unwrap().get(&source) {
            return Ok(module.clone());
        }
        let opts = self.compile_options();
        let key = self.cache_key(&opts, &source)?;
        let cached = disk.and_then(|d| d.load(key.as_bytes())).and_then(|code| self.load(&code).ok());
        let module = match cached {
            Some(module) => module,
            None => {
                let code = compile_code(self.l, entry, &source, &opts)?;
                if let Some(d) = disk {
                    d.store(key.as_bytes(), &code);
                }
                self.load(&code)?
            }
        };
        self.modules.lock().unwrap().insert(source, module.clone());
        Ok(module)
    }

    /// What a code object of `source` compiled with `opts` depends on.
    fn cache_key(&self, opts: &[String], source: &str) -> Result<String, GpuError> {
        Ok(format!("{}\nhiprtc {}\n{source}", opts.join(" "), hiprtc_version(self.l)?))
    }

    /// A code object loaded as a module. The runtime takes the object
    /// without its length, so anything not starting as an ELF file is
    /// refused here rather than read past its end.
    fn load(&self, code: &[u8]) -> Result<Arc<HipModule>, GpuError> {
        if !code.starts_with(b"\x7fELF") {
            return Err(GpuError::Driver("hipModuleLoadData: not an ELF code object".into()));
        }
        self.bind()?;
        let mut module = core::ptr::null_mut();
        // SAFETY: `code` is a whole code object, read by the call.
        check(self.l, unsafe { self.l.rt.hipModuleLoadData(&mut module, code.as_ptr().cast()) }, "hipModuleLoadData")?;
        Ok(Arc::new(HipModule { module, l: self.l, index: self.index }))
    }
}

impl Device for HipDevice {
    type Block = HipBlock;
    type Kernel = HipKernel;
    type Graph = HipGraph;
    type Host = HipHost;
    const RUNTIME: &'static str = "hip";

    fn arch(&self) -> &Arch {
        &self.arch
    }

    fn allocations(&self) -> usize {
        self.pool.allocations()
    }

    fn reserved_bytes(&self) -> usize {
        self.pool.reserved_bytes()
    }

    fn host_allocations(&self) -> usize {
        self.host_pool.allocations()
    }

    fn alloc(&self, bytes: usize) -> Result<HipBlock, GpuError> {
        crate::fault::check("alloc")?;
        self.not_capturing("an allocation")?;
        self.bind()?;
        let (cap, mem) = match self.pool.take(bytes) {
            Some(held) => held,
            None => {
                let cap = capacity(bytes);
                let mut ptr = core::ptr::null_mut();
                // SAFETY: an out-parameter.
                check(self.l, unsafe { self.l.rt.hipMalloc(&mut ptr, cap) }, "hipMalloc")?;
                self.pool.made(cap);
                (cap, HipMem { ptr, l: self.l, index: self.index })
            }
        };
        // back to the pool if the zeroing fails
        let block = HipBlock { mem: Some(mem), cap, len: bytes, pool: self.pool.clone() };
        // SAFETY: the bytes asked for, zeroed in order on the stream.
        check(self.l, unsafe { self.l.rt.hipMemsetAsync(block.ptr(), 0, bytes, self.stream) }, "hipMemsetAsync")?;
        Ok(block)
    }

    fn write(&self, dst: &mut HipBlock, offset: usize, src: &[u8]) -> Result<(), GpuError> {
        crate::fault::check("write")?;
        self.not_capturing("a write")?;
        assert!(offset + src.len() <= dst.len, "a write past the block's {} bytes", dst.len);
        self.bind()?;
        // SAFETY: within the block; HIP copies pageable host memory before
        // the call returns, so `src` may be reused after it.
        let e = unsafe { self.l.rt.hipMemcpyHtoDAsync(dst.ptr().byte_add(offset), src.as_ptr().cast(), src.len(), self.stream) };
        check(self.l, e, "hipMemcpyHtoDAsync")
    }

    fn alloc_host(&self, bytes: usize) -> Result<HipHost, GpuError> {
        crate::fault::check("alloc_host")?;
        let (cap, mem) = match self.host_pool.take(bytes) {
            Some(taken) => taken,
            None => {
                let cap = capacity(bytes);
                self.host_pool.made(cap);
                self.bind()?;
                let mut event = core::ptr::null_mut();
                // SAFETY: an out-parameter.
                check(self.l, unsafe { self.l.rt.hipEventCreateWithFlags(&mut event, rt::hipEventDisableTiming) }, "hipEventCreateWithFlags")?;
                let mut ptr = core::ptr::null_mut();
                // SAFETY: an out-parameter; contents left undefined.
                let e = unsafe { self.l.rt.hipHostMalloc(&mut ptr, cap, 0) };
                if let Err(err) = check(self.l, e, "hipHostMalloc") {
                    // SAFETY: the event created above, destroyed once.
                    unsafe {
                        let _ = self.l.rt.hipEventDestroy(event);
                    }
                    return Err(err);
                }
                (cap, PinnedMem { ptr: ptr.cast(), event, l: self.l, index: self.index })
            }
        };
        Ok(HipHost { mem: Some(mem), cap, len: bytes, pool: self.host_pool.clone() })
    }

    fn host_mut<'a>(&self, host: &'a mut HipHost) -> Result<&'a mut [u8], GpuError> {
        crate::fault::check("host_mut")?;
        self.bind()?;
        let mem = host.mem();
        // SAFETY: the host block's event.
        check(self.l, unsafe { self.l.rt.hipEventSynchronize(mem.event) }, "hipEventSynchronize")?;
        // SAFETY: the allocation holds `len` bytes, no copy from it pending.
        Ok(unsafe { core::slice::from_raw_parts_mut(mem.ptr, host.len) })
    }

    fn write_from_host(&self, dst: &mut HipBlock, offset: usize, host: &HipHost, range: core::ops::Range<usize>) -> Result<(), GpuError> {
        crate::fault::check("write_from_host")?;
        self.not_capturing("a write")?;
        assert!(range.start <= range.end && range.end <= host.len, "a copy past the host block's {} bytes", host.len);
        assert!(offset + range.len() <= dst.len, "a write past the block's {} bytes", dst.len);
        if range.is_empty() {
            return Ok(());
        }
        self.bind()?;
        let mem = host.mem();
        // SAFETY: within the block and the host memory; the host memory is
        // not written before the event recorded below has passed.
        let e = unsafe { self.l.rt.hipMemcpyHtoDAsync(dst.ptr().byte_add(offset), mem.ptr.add(range.start).cast(), range.len(), self.stream) };
        check(self.l, e, "hipMemcpyHtoDAsync")?;
        // SAFETY: the host block's event and the device's stream.
        check(self.l, unsafe { self.l.rt.hipEventRecord(mem.event, self.stream) }, "hipEventRecord")
    }

    fn read(&self, src: &HipBlock, offset: usize, dst: &mut [u8]) -> Result<(), GpuError> {
        self.read_many(&mut [(src, offset, dst)])
    }

    fn read_many(&self, reads: &mut [(&HipBlock, usize, &mut [u8])]) -> Result<(), GpuError> {
        crate::fault::check("read")?;
        self.not_capturing("a read")?;
        self.bind()?;
        for (src, offset, dst) in reads.iter_mut() {
            assert!(*offset + dst.len() <= src.len, "a read past the block's {} bytes", src.len);
            // SAFETY: within the block; `dst` is written before the wait returns.
            let e = unsafe { self.l.rt.hipMemcpyDtoHAsync(dst.as_mut_ptr().cast(), src.ptr().byte_add(*offset), dst.len(), self.stream) };
            check(self.l, e, "hipMemcpyDtoHAsync")?;
        }
        // SAFETY: the device's stream.
        check(self.l, unsafe { self.l.rt.hipStreamSynchronize(self.stream) }, "hipStreamSynchronize")
    }

    fn compile_many(&self, source: &str, defines: &str, entries: &[&str]) -> Result<Vec<HipKernel>, GpuError> {
        crate::fault::check("compile")?;
        self.not_capturing("a compile")?;
        let full = self.unit(source, defines);
        let module = self.module(entries.first().copied().unwrap_or("module"), full)?;
        self.bind()?;
        entries
            .iter()
            .map(|&entry| {
                let name = c_string(entry, entry)?;
                let mut function = core::ptr::null_mut();
                // SAFETY: a loaded module and a C string.
                check(self.l, unsafe { self.l.rt.hipModuleGetFunction(&mut function, module.module, name.as_ptr()) }, &format!("hipModuleGetFunction({entry})"))?;
                Ok(HipKernel { function, module: module.clone() })
            })
            .collect()
    }

    fn launch(&self, kernel: &HipKernel, launch: Launch, args: &[Arg<'_, HipBlock>]) -> Result<(), GpuError> {
        self.launch_on(Lane::Main, kernel, launch, args)
    }

    fn launch_on(&self, lane: Lane, kernel: &HipKernel, launch: Launch, args: &[Arg<'_, HipBlock>]) -> Result<(), GpuError> {
        if launch.is_empty() {
            return Ok(());
        }
        crate::fault::check("launch")?;
        self.forks.check(lane);
        let stream = match lane {
            Lane::Main => self.stream,
            Lane::Side => self.side,
        };
        assert!(args.len() <= MAX_ARGS, "a launch takes at most {MAX_ARGS} arguments");
        let mut values = [0u64; MAX_ARGS];
        let mut tables = Tables::new();
        for (i, (slot, arg)) in values.iter_mut().zip(args).enumerate() {
            *slot = match *arg {
                Arg::Block(b, offset) => {
                    assert!(offset <= b.len, "argument offset {offset} past the block's {} bytes", b.len);
                    b.ptr() as u64 + offset as u64
                }
                Arg::U32(v) => v as u64,
                Arg::I32(v) => v as u32 as u64,
                Arg::F32(v) => v.to_bits() as u64,
                Arg::F64(v) => v.to_bits(),
                Arg::Bits(v) => v,
                Arg::Bytes(bytes) => {
                    tables.push((i, crate::table_words(bytes)));
                    0
                }
            };
        }
        if self.timing.load(Ordering::Acquire) {
            let timed = TimedLaunch { function: kernel.function, _module: kernel.module.clone(), lane, launch, values, tables };
            let mut timer = self.timer();
            return match timer.capture.as_mut() {
                // kept for the graph's replays
                Some(launches) => {
                    launches.push(timed);
                    Ok(())
                }
                None => self.run_timed(&mut timer, &timed),
            };
        }
        self.issue(kernel.function, stream, launch, &mut values, &mut tables)
    }

    fn time_launches(&self, on: bool) {
        self.timing.store(on, Ordering::Release);
    }

    fn take_times(&self) -> Vec<f64> {
        core::mem::take(&mut self.timer().times)
    }

    /// An event recorded on the main stream, with its timing kept (no flag).
    fn stamp(&self, slot: usize) -> Result<(), GpuError> {
        self.bind()?;
        let mut stamps = self.stamps.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: the device's stream; the event is made here once and
        // destroyed with the device.
        unsafe {
            if stamps[slot].is_null() {
                check(self.l, self.l.rt.hipEventCreateWithFlags(&mut stamps[slot], 0), "hipEventCreateWithFlags")?;
            }
            check(self.l, self.l.rt.hipEventRecord(stamps[slot], self.stream), "hipEventRecord")
        }
    }

    fn span(&self, from: usize, to: usize) -> Result<Option<f64>, GpuError> {
        let stamps = self.stamps.lock().unwrap_or_else(|e| e.into_inner());
        if stamps[from].is_null() || stamps[to].is_null() {
            return Ok(None);
        }
        self.bind()?;
        let mut ms = 0f32;
        // SAFETY: two events of this device, both reached: the caller waited.
        check(self.l, unsafe { self.l.rt.hipEventElapsedTime(&mut ms, stamps[from], stamps[to]) }, "hipEventElapsedTime")?;
        Ok(Some(ms as f64 * 1e-3))
    }

    /// The side stream waits for an event recorded on the main one; in a
    /// capture, the wait brings the side stream into it.
    fn fork(&self) -> Result<(), GpuError> {
        crate::fault::check("fork")?;
        self.bind()?;
        // SAFETY: the device's streams and event.
        unsafe {
            check(self.l, self.l.rt.hipEventRecord(self.fork_event, self.stream), "hipEventRecord")?;
            check(self.l, self.l.rt.hipStreamWaitEvent(self.side, self.fork_event, 0), "hipStreamWaitEvent")?;
        }
        self.forks.fork();
        Ok(())
    }

    /// The main stream waits for an event recorded on the side one; in a
    /// capture, the wait joins the side stream back.
    fn join(&self) -> Result<(), GpuError> {
        crate::fault::check("join")?;
        self.forks.join();
        self.bind()?;
        // SAFETY: the device's streams and event.
        unsafe {
            check(self.l, self.l.rt.hipEventRecord(self.join_event, self.side), "hipEventRecord")?;
            check(self.l, self.l.rt.hipStreamWaitEvent(self.stream, self.join_event, 0), "hipStreamWaitEvent")
        }
    }

    fn sync(&self) -> Result<(), GpuError> {
        crate::fault::check("sync")?;
        self.not_capturing("a wait")?;
        self.bind()?;
        // SAFETY: the device's streams.
        unsafe {
            check(self.l, self.l.rt.hipStreamSynchronize(self.stream), "hipStreamSynchronize")?;
            check(self.l, self.l.rt.hipStreamSynchronize(self.side), "hipStreamSynchronize")
        }
    }

    /// Stream capture in thread-local mode, instantiated and uploaded, as on CUDA.
    fn capture(&self, record: impl FnOnce() -> Result<(), GpuError>) -> Result<HipGraph, GpuError> {
        crate::fault::check("capture")?;
        if self.timing.load(Ordering::Acquire) {
            return self.capture_timed(record);
        }
        let forks = self.forks.open();
        let open = OpenCapture::begin(self)?;
        let recorded = record();
        // ours first: HIP's own error for a fork not joined says less
        let joined = self.forks.check_capture(forks);
        let graph = open.end();
        recorded?;
        joined?;
        let graph = graph?;
        if graph.0.is_null() {
            return Err(GpuError::Capture("the capture recorded nothing".into()));
        }
        self.bind()?;
        let mut exec = core::ptr::null_mut();
        // SAFETY: a valid graph from hipStreamEndCapture; the executable
        // graph is independent of it.
        let e = unsafe { self.l.rt.hipGraphInstantiate(&mut exec, graph.0, core::ptr::null_mut(), core::ptr::null_mut(), 0) };
        check(self.l, e, "hipGraphInstantiate")?;
        drop(graph);
        let graph = HipGraph { exec, timed: Vec::new(), l: self.l, index: self.index };
        // the first replay then pays no setup
        // SAFETY: the executable graph and the stream are alive.
        check(self.l, unsafe { self.l.rt.hipGraphUpload(graph.exec, self.stream) }, "hipGraphUpload")?;
        Ok(graph)
    }

    fn replay(&self, graph: &HipGraph) -> Result<(), GpuError> {
        crate::fault::check("replay")?;
        self.not_capturing("a replay")?;
        assert_eq!(graph.index, self.index, "a graph replays on the device it was captured on");
        if graph.exec.is_null() {
            let mut timer = self.timer();
            return graph.timed.iter().try_for_each(|launch| self.run_timed(&mut timer, launch));
        }
        self.bind()?;
        // SAFETY: the executable graph is alive; its blocks are the caller's to keep alive.
        check(self.l, unsafe { self.l.rt.hipGraphLaunch(graph.exec, self.stream) }, "hipGraphLaunch")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HEADER;

    /// A cached entry the runtime refuses (not a code object at all, its
    /// length and checksum right) is compiled again and written over.
    #[test]
    fn a_cached_binary_the_runtime_refuses_is_compiled_again() {
        let dir = std::env::temp_dir().join(format!("arael-hip-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = DiskCache::at(dir.clone());
        let dev = HipDevice::new(0).unwrap();
        let source = format!("{}\n{HEADER}\nKERNEL void k(GLOBAL float* x) {{ x[0] = 1.0f; }}", dev.arch.defines());
        let key = dev.cache_key(&dev.compile_options(), &source).unwrap();
        disk.store(key.as_bytes(), b"not a code object");
        assert_eq!(disk.load(key.as_bytes()).as_deref(), Some(&b"not a code object"[..]));
        assert!(matches!(dev.load(b"not a code object"), Err(GpuError::Driver(_))));
        dev.module_in("k", source.clone(), Some(&disk)).unwrap();
        let code = disk.load(key.as_bytes()).expect("the entry written over");
        assert!(code.starts_with(b"\x7fELF"));
        // which a device without it in memory loads
        HipDevice::new(0).unwrap().load(&code).unwrap();
        std::fs::remove_dir_all(dir).unwrap();
    }
}
