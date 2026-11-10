//! [`Device`] on Metal: the GPU of an Apple silicon Mac, through
//! Metal.framework (objc2-metal). Metal ships with macOS, so nothing loads
//! at run time and a build needs no Xcode.
//!
//! Kernels compile at run time from the same source as on CUDA and HIP,
//! with arael_gpu.h's Metal branch in front and each entry's parameters
//! rewritten into Metal's buffer bindings ([`signature`]). Metal has no
//! f64: a factor is f32 and the wide scalar df32.
//!
//! The stream is one command queue. Launches are encoded into an open
//! command buffer, committed when the host waits and every
//! [`COMMIT_EVERY`] launches. Memory is shared between the host and the
//! GPU: a write when nothing is queued is a copy into the buffer, otherwise
//! a copy queued behind the work before it, as on a CUDA stream; a read
//! waits, then copies out of the buffer.
//!
//! [`MetalDevice::time_kernels`] times each launch on the GPU, summed per
//! kernel.
//!
//! ```ignore
//! let dev = arael_gpu::metal::MetalDevice::new(0)?;
//! let k = dev.compile("KERNEL void k(GLOBAL float* x) { x[global_x()] = 1.0f; }", "", "k")?;
//! ```

mod signature;

use core::ffi::{c_char, c_void};
use core::ptr::NonNull;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::ThreadId;

use objc2::msg_send;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue, MTLCompileOptions,
    MTLComputeCommandEncoder, MTLComputePipelineState, MTLCopyAllDevices, MTLDevice, MTLGPUFamily, MTLLibrary, MTLMathFloatingPointFunctions,
    MTLMathMode, MTLResourceOptions, MTLSize,
};

use crate::pool::{Pool, capacity};
use crate::{Arch, Arg, Device, GpuError, Launch, MAX_ARGS};
use signature::{Entry, Param};

/// Launches per command buffer before it is committed without a wait, so
/// the GPU starts while the host still encodes.
pub const COMMIT_EVERY: usize = 128;

/// A Metal buffer in shared storage, the host's and the GPU's at once.
struct Buf(Retained<ProtocolObject<dyn MTLBuffer>>);

// SAFETY: Metal's buffers are thread-safe objects; their contents are read
// and written in the order the device's operations give.
unsafe impl Send for Buf {}
unsafe impl Sync for Buf {}

impl Buf {
    fn ptr(&self) -> *mut u8 {
        self.0.contents().as_ptr().cast()
    }
}

/// A command buffer and its serial number on the device's queue.
struct CmdBuf {
    cb: Retained<ProtocolObject<dyn MTLCommandBuffer>>,
    serial: u64,
}

/// The device's queue: the command buffer being encoded, its compute
/// encoder, and the command buffers committed and not yet seen complete.
struct Stream {
    open: Option<CmdBuf>,
    encoder: Option<Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>>,
    /// Launches in the open command buffer.
    launches: usize,
    /// Committed, oldest first.
    flight: VecDeque<CmdBuf>,
    /// The serial of the next command buffer.
    next: u64,
    /// The first error of a command buffer seen complete without a wait,
    /// for the next wait to return.
    failed: Option<String>,
    /// Whether launches are timed ([`MetalDevice::time_kernels`]).
    timing: bool,
    /// Each timed kernel's launches and GPU seconds, until taken.
    times: HashMap<String, (usize, f64)>,
}

/// A kernel's GPU time over its launches while [`MetalDevice::time_kernels`]
/// was on.
#[derive(Clone, Debug, PartialEq)]
pub struct KernelTime {
    pub name: String,
    pub launches: usize,
    pub seconds: f64,
}

// SAFETY: command buffers and encoders are used only under the device's
// lock on the stream, one thread at a time.
unsafe impl Send for Stream {}

/// One Metal GPU and a command queue of its own.
pub struct MetalDevice {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    queue: Retained<ProtocolObject<dyn MTLCommandQueue>>,
    arch: Arch,
    stream: Mutex<Stream>,
    /// Compiled sources by their full text.
    modules: Mutex<HashMap<String, Arc<Module>>>,
    /// The open capture: its thread and what it recorded.
    capture: Mutex<Option<(ThreadId, Vec<Recorded>)>>,
    pool: Arc<Pool<Buf>>,
    host_pool: Arc<Pool<HostMem>>,
}

impl core::fmt::Debug for MetalDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalDevice").field("arch", &self.arch).finish_non_exhaustive()
    }
}

/// Device memory on a [`MetalDevice`], from its pool and back to it on drop.
pub struct MetalBlock {
    /// Some until the drop; its own size may exceed `len`.
    mem: Option<Buf>,
    len: usize,
    pool: Arc<Pool<Buf>>,
}

impl MetalBlock {
    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn buf(&self) -> &Buf {
        self.mem.as_ref().expect("a block's memory until its drop")
    }
}

impl Drop for MetalBlock {
    fn drop(&mut self) {
        if let Some(mem) = self.mem.take() {
            self.pool.give(mem.0.length(), mem);
        }
    }
}

/// A host block's memory and the serial of the last command buffer that
/// copies from it.
struct HostMem {
    buf: Buf,
    last_use: AtomicU64,
}

/// Host memory on a [`MetalDevice`] ([`Device::alloc_host`]), shared with
/// the GPU, from its pool and back to it on drop.
pub struct MetalHost {
    mem: Option<HostMem>,
    cap: usize,
    len: usize,
    pool: Arc<Pool<HostMem>>,
}

impl MetalHost {
    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn mem(&self) -> &HostMem {
        self.mem.as_ref().expect("a host block's memory until its drop")
    }
}

impl Drop for MetalHost {
    fn drop(&mut self) {
        // copies from it may still run: the next taker waits for them
        if let Some(mem) = self.mem.take() {
            self.pool.give(self.cap, mem);
        }
    }
}

/// A compiled source.
struct Module {
    /// Its entries, as the rewrite read them.
    entries: Vec<Entry>,
    library: Retained<ProtocolObject<dyn MTLLibrary>>,
}

/// A compiled entry point on a [`MetalDevice`].
#[derive(Clone)]
pub struct MetalKernel {
    inner: Arc<KernelInner>,
}

struct KernelInner {
    pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    entry: Entry,
}

impl core::fmt::Debug for MetalKernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("MetalKernel").field("name", &self.inner.entry.name).finish_non_exhaustive()
    }
}

impl MetalKernel {
    /// The most threads a block of this kernel may have: at most 1024, fewer
    /// when its registers do not allow them.
    pub fn max_threads(&self) -> usize {
        self.inner.pipeline.maxTotalThreadsPerThreadgroup()
    }

    /// Shared memory its source declares, in bytes.
    pub fn static_shared_bytes(&self) -> usize {
        self.inner.pipeline.staticThreadgroupMemoryLength()
    }
}

/// A launch argument as a graph keeps it.
enum Bound {
    Buffer(Retained<ProtocolObject<dyn MTLBuffer>>, usize),
    Bytes([u8; 8], usize),
}

/// A launch as a graph keeps it: the kernel, its shape and its arguments
/// (the blocks' buffers retained; the blocks outlive the graph).
struct Recorded {
    kernel: MetalKernel,
    launch: Launch,
    args: Vec<Bound>,
}

// SAFETY: a recorded buffer's handle is only bound by a replay, under the
// device's lock; Metal's buffers are thread-safe objects.
unsafe impl Send for Recorded {}

/// The launches a capture recorded, encoded again by a replay.
pub struct MetalGraph {
    launches: Vec<Recorded>,
}

/// Metal compilations in this process.
static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Kernel sources Metal's compiler has compiled in this process: those no
/// device held in memory. (Metal keeps its own cache of compiled sources;
/// arael's disk cache is not used.)
pub fn compiles() -> usize {
    COMPILES.load(Ordering::Relaxed)
}

impl Stream {
    /// The open command buffer, made when there is none.
    fn open(&mut self, queue: &ProtocolObject<dyn MTLCommandQueue>) -> Result<&CmdBuf, GpuError> {
        if self.open.is_none() {
            let cb = queue.commandBuffer().ok_or_else(|| GpuError::Driver("no command buffer from the queue".into()))?;
            self.open = Some(CmdBuf { cb, serial: self.next });
            self.next += 1;
        }
        Ok(self.open.as_ref().unwrap())
    }

    /// The compute encoder of the open command buffer, made when there is none.
    fn encoder(&mut self, queue: &ProtocolObject<dyn MTLCommandQueue>) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>, GpuError> {
        if self.encoder.is_none() {
            let enc = self.open(queue)?.cb.computeCommandEncoder().ok_or_else(|| GpuError::Driver("no compute encoder".into()))?;
            self.encoder = Some(enc);
        }
        Ok(self.encoder.as_deref().unwrap())
    }

    /// `encode` on a blit encoder of the open command buffer, after the
    /// launches before it.
    fn blit(&mut self, queue: &ProtocolObject<dyn MTLCommandQueue>, encode: impl FnOnce(&ProtocolObject<dyn MTLBlitCommandEncoder>)) -> Result<(), GpuError> {
        if let Some(enc) = self.encoder.take() {
            enc.endEncoding();
        }
        let blit = self.open(queue)?.cb.blitCommandEncoder().ok_or_else(|| GpuError::Driver("no blit encoder".into()))?;
        encode(&blit);
        blit.endEncoding();
        Ok(())
    }

    /// Commits the open command buffer, if any, without waiting.
    fn commit(&mut self) {
        if let Some(enc) = self.encoder.take() {
            enc.endEncoding();
        }
        if let Some(open) = self.open.take() {
            open.cb.commit();
            self.flight.push_back(open);
        }
        self.launches = 0;
    }

    /// Drops the committed command buffers that have completed, keeping
    /// the first error for the next wait.
    fn reap(&mut self) {
        while let Some(front) = self.flight.front() {
            match front.cb.status() {
                MTLCommandBufferStatus::Completed => {}
                MTLCommandBufferStatus::Error => {
                    let why = status_error(&front.cb);
                    self.failed.get_or_insert(why);
                }
                _ => return,
            }
            self.flight.pop_front();
        }
    }

    /// Nothing queued, encoded or running.
    fn idle(&mut self) -> bool {
        self.reap();
        self.open.is_none() && self.flight.is_empty()
    }

    /// Waits until the command buffers up to `serial` have completed (all
    /// with `u64::MAX`); the first error of one, or of one seen before.
    fn wait_until(&mut self, serial: u64) -> Result<(), GpuError> {
        if self.open.as_ref().is_some_and(|o| o.serial <= serial) {
            self.commit();
        }
        while let Some(front) = self.flight.front() {
            if front.serial > serial {
                break;
            }
            front.cb.waitUntilCompleted();
            if front.cb.status() == MTLCommandBufferStatus::Error {
                let why = status_error(&front.cb);
                self.failed.get_or_insert(why);
            }
            self.flight.pop_front();
        }
        match self.failed.take() {
            Some(why) => Err(GpuError::Driver(format!("a command buffer failed: {why}"))),
            None => Ok(()),
        }
    }

    /// Commits the open command buffer, waits for it, and adds its GPU time
    /// to kernel `name`'s.
    fn time_open(&mut self, name: &str) -> Result<(), GpuError> {
        let Some(serial) = self.open.as_ref().map(|o| o.serial) else {
            return Ok(());
        };
        self.commit();
        let cb = self.flight.back().expect("the command buffer just committed").cb.clone();
        self.wait_until(serial)?;
        // SAFETY: both take no arguments and return a CFTimeInterval (an
        // f64, seconds); the command buffer has completed.
        let (start, end): (f64, f64) = unsafe { (msg_send![&*cb, GPUStartTime], msg_send![&*cb, GPUEndTime]) };
        match self.times.get_mut(name) {
            Some(t) => *t = (t.0 + 1, t.1 + (end - start)),
            None => {
                self.times.insert(name.to_string(), (1, end - start));
            }
        }
        Ok(())
    }
}

/// The error of a failed command buffer, as text.
fn status_error(cb: &ProtocolObject<dyn MTLCommandBuffer>) -> String {
    cb.error().map_or_else(|| "no error given".to_string(), |e| e.localizedDescription().to_string())
}

/// The GPU cores of the first Apple GPU, from the IO registry (Metal does
/// not report them).
fn gpu_cores() -> Result<u32, GpuError> {
    #[link(name = "IOKit", kind = "framework")]
    unsafe extern "C" {
        fn IOServiceMatching(name: *const c_char) -> *mut c_void;
        fn IOServiceGetMatchingService(main_port: u32, matching: *mut c_void) -> u32;
        fn IORegistryEntryCreateCFProperty(entry: u32, key: *const c_void, allocator: *const c_void, options: u32) -> *const c_void;
        fn IOObjectRelease(object: u32) -> i32;
    }
    #[link(name = "CoreFoundation", kind = "framework")]
    unsafe extern "C" {
        fn CFStringCreateWithCString(allocator: *const c_void, s: *const c_char, encoding: u32) -> *const c_void;
        fn CFGetTypeID(cf: *const c_void) -> usize;
        fn CFNumberGetTypeID() -> usize;
        fn CFNumberGetValue(number: *const c_void, the_type: isize, value: *mut c_void) -> bool;
        fn CFRelease(cf: *const c_void);
    }
    const UTF8: u32 = 0x0800_0100;
    const SINT32: isize = 3;
    let missing = || GpuError::Unavailable("the GPU's core count is not in the IO registry (AGXAccelerator, gpu-core-count)".into());
    // SAFETY: IOKit and CoreFoundation calls on objects made here, each
    // released once; IOServiceGetMatchingService consumes the dictionary.
    unsafe {
        let service = IOServiceGetMatchingService(0, IOServiceMatching(c"AGXAccelerator".as_ptr()));
        if service == 0 {
            return Err(missing());
        }
        let key = CFStringCreateWithCString(core::ptr::null(), c"gpu-core-count".as_ptr(), UTF8);
        let value = IORegistryEntryCreateCFProperty(service, key, core::ptr::null(), 0);
        CFRelease(key);
        IOObjectRelease(service);
        if value.is_null() {
            return Err(missing());
        }
        let mut cores = 0i32;
        let ok = CFGetTypeID(value) == CFNumberGetTypeID() && CFNumberGetValue(value, SINT32, (&mut cores as *mut i32).cast());
        CFRelease(value);
        if !ok || cores <= 0 {
            return Err(missing());
        }
        Ok(cores as u32)
    }
}

impl MetalDevice {
    /// Metal device `index` (0 is the Mac's GPU). [`GpuError::Unavailable`]
    /// when there is no such device or it is not an Apple GPU of family 7
    /// or later (M1 on).
    pub fn new(index: usize) -> Result<Self, GpuError> {
        autoreleasepool(|_| {
            let devices = MTLCopyAllDevices();
            let device = devices.iter().nth(index).ok_or_else(|| GpuError::Unavailable(format!("no Metal device {index} ({} found)", devices.len())))?;
            let name = device.name().to_string();
            if !device.supportsFamily(MTLGPUFamily::Apple7) {
                return Err(GpuError::Unavailable(format!("Metal device {index} ({name}) is not an Apple GPU of family 7 or later")));
            }
            let family = [(MTLGPUFamily::Apple10, 10), (MTLGPUFamily::Apple9, 9), (MTLGPUFamily::Apple8, 8)]
                .into_iter()
                .find(|(f, _)| device.supportsFamily(*f))
                .map_or(7, |(_, n)| n);
            let queue = device.newCommandQueue().ok_or_else(|| GpuError::Driver("no command queue".into()))?;
            let arch = Arch {
                name,
                target: format!("apple{family}"),
                // checked against each pipeline's execution width
                warp: 32,
                shared_bytes: device.maxThreadgroupMemoryLength() as u32,
                multiprocessors: gpu_cores()?,
                has_f64: false,
                // measured: in safe math too (docs/dev/METAL_PLAN.md)
                flushes_subnormals: true,
                integrated: device.hasUnifiedMemory(),
            };
            Ok(MetalDevice {
                device,
                queue,
                arch,
                stream: Mutex::new(Stream { open: None, encoder: None, launches: 0, flight: VecDeque::new(), next: 0, failed: None, timing: false, times: HashMap::new() }),
                modules: Mutex::new(HashMap::new()),
                capture: Mutex::new(None),
                pool: Arc::new(Pool::new()),
                host_pool: Arc::new(Pool::new()),
            })
        })
    }

    fn stream(&self) -> MutexGuard<'_, Stream> {
        self.stream.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Starts (`true`) or stops (`false`) timing each launch on the GPU,
    /// for [`take_kernel_times`](Self::take_kernel_times). A timed launch
    /// runs in a command buffer of its own, which the host waits for, so
    /// launches no longer overlap one another or the host's encoding.
    ///
    /// ```ignore
    /// dev.time_kernels(true);
    /// factor.factorize_device(vals, fail)?;
    /// dev.time_kernels(false);
    /// for t in dev.take_kernel_times() {
    ///     println!("{} {} {:.3} ms", t.name, t.launches, t.seconds * 1e3);
    /// }
    /// ```
    pub fn time_kernels(&self, on: bool) {
        self.stream().timing = on;
    }

    /// The GPU time of each kernel launched while timing was on, the
    /// largest first; the sums start again from zero.
    pub fn take_kernel_times(&self) -> Vec<KernelTime> {
        let times = core::mem::take(&mut self.stream().times);
        let mut v: Vec<KernelTime> = times.into_iter().map(|(name, (launches, seconds))| KernelTime { name, launches, seconds }).collect();
        v.sort_by(|a, b| b.seconds.total_cmp(&a.seconds).then_with(|| a.name.cmp(&b.name)));
        v
    }

    /// Whether the calling thread has a capture open.
    fn capturing(&self) -> bool {
        let me = std::thread::current().id();
        self.capture.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|(t, _)| *t == me)
    }

    fn not_capturing(&self, op: &str) -> Result<(), GpuError> {
        if self.capturing() {
            return Err(GpuError::Capture(format!("{op} while a capture is open on the device")));
        }
        Ok(())
    }

    /// A shared buffer of `bytes` (zeroed by Metal).
    fn new_buffer(&self, bytes: usize) -> Result<Buf, GpuError> {
        self.device
            .newBufferWithLength_options(bytes.max(1), MTLResourceOptions::StorageModeShared)
            .map(Buf)
            .ok_or_else(|| GpuError::Limit(format!("a Metal buffer of {bytes} bytes")))
    }

    /// The module of `unit`: in memory when this device compiled it before,
    /// else compiled (`entry` names it in the compiler's messages).
    fn module(&self, entry: &str, unit: String) -> Result<Arc<Module>, GpuError> {
        if let Some(m) = self.modules.lock().unwrap_or_else(|e| e.into_inner()).get(&unit) {
            return Ok(m.clone());
        }
        let (msl, entries) = signature::rewrite(&unit).map_err(|log| GpuError::Compile { entry: entry.to_string(), log })?;
        let options = MTLCompileOptions::new();
        if objc2::available!(macos = 15.0) {
            options.setMathMode(MTLMathMode::Safe);
            options.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Precise);
        } else {
            #[allow(deprecated)]
            options.setFastMathEnabled(false);
        }
        COMPILES.fetch_add(1, Ordering::Relaxed);
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(&msl), Some(&options))
            .map_err(|e| GpuError::Compile { entry: entry.to_string(), log: e.localizedDescription().to_string() })?;
        let module = Arc::new(Module { entries, library });
        self.modules.lock().unwrap_or_else(|e| e.into_inner()).insert(unit, module.clone());
        Ok(module)
    }

    /// Encodes one launch of `kernel` on the open command buffer.
    fn encode(&self, s: &mut Stream, kernel: &MetalKernel, launch: &Launch, args: &[Bound]) -> Result<(), GpuError> {
        if s.timing {
            // a command buffer of its own: the copies queued before stay out of its time
            s.commit();
        }
        let enc = s.encoder(&self.queue)?;
        enc.setComputePipelineState(&kernel.inner.pipeline);
        for (i, a) in args.iter().enumerate() {
            match a {
                // SAFETY: a live buffer and an offset within it.
                Bound::Buffer(b, offset) => unsafe { enc.setBuffer_offset_atIndex(Some(b), *offset, i) },
                // SAFETY: `len` bytes of the value, copied by Metal now.
                Bound::Bytes(v, len) => unsafe { enc.setBytes_length_atIndex(NonNull::from(v).cast(), *len, i) },
            }
        }
        if kernel.inner.entry.dynamic_shared {
            // SAFETY: a length within the device's limit (checked at launch), a multiple of 16.
            unsafe { enc.setThreadgroupMemoryLength_atIndex((launch.shared_bytes as usize).next_multiple_of(16).max(16), 0) };
        }
        let [gx, gy, gz] = launch.grid;
        let [bx, by, bz] = launch.block;
        let size = |x: u32, y: u32, z: u32| MTLSize { width: x as usize, height: y as usize, depth: z as usize };
        enc.dispatchThreadgroups_threadsPerThreadgroup(size(gx, gy, gz), size(bx, by, bz));
        s.launches += 1;
        if s.timing {
            s.time_open(&kernel.inner.entry.name)?;
        } else if s.launches >= COMMIT_EVERY {
            s.commit();
        }
        Ok(())
    }
}

impl Device for MetalDevice {
    type Block = MetalBlock;
    type Kernel = MetalKernel;
    type Graph = MetalGraph;
    type Host = MetalHost;
    const RUNTIME: &'static str = "metal";

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

    fn alloc(&self, bytes: usize) -> Result<MetalBlock, GpuError> {
        crate::fault::check("alloc")?;
        self.not_capturing("an allocation")?;
        autoreleasepool(|_| {
            let (mem, fresh) = match self.pool.take(bytes) {
                Some((_, mem)) => (mem, false),
                None => {
                    let cap = capacity(bytes);
                    let mem = self.new_buffer(cap)?;
                    self.pool.made(cap);
                    (mem, true)
                }
            };
            let block = MetalBlock { mem: Some(mem), len: bytes, pool: self.pool.clone() };
            // a fresh buffer is zeroed; a reused one may still be read by queued work
            if !fresh && bytes > 0 {
                let mut s = self.stream();
                if s.idle() {
                    // SAFETY: `bytes` within the buffer, nothing queued uses it.
                    unsafe { core::ptr::write_bytes(block.buf().ptr(), 0, bytes) };
                } else {
                    let buf = &block.buf().0;
                    s.blit(&self.queue, |b| b.fillBuffer_range_value(buf, NSRange::new(0, bytes), 0))?;
                }
            }
            Ok(block)
        })
    }

    fn write(&self, dst: &mut MetalBlock, offset: usize, src: &[u8]) -> Result<(), GpuError> {
        crate::fault::check("write")?;
        self.not_capturing("a write")?;
        assert!(offset + src.len() <= dst.len, "a write past the block's {} bytes", dst.len);
        if src.is_empty() {
            return Ok(());
        }
        autoreleasepool(|_| {
            let mut s = self.stream();
            if s.idle() {
                // SAFETY: within the block; nothing queued uses it.
                unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), dst.buf().ptr().add(offset), src.len()) };
                return Ok(());
            }
            // SAFETY: Metal copies `src` into the new buffer now.
            let staging = unsafe {
                self.device.newBufferWithBytes_length_options(NonNull::from(src).cast(), src.len(), MTLResourceOptions::StorageModeShared)
            }
            .ok_or_else(|| GpuError::Limit(format!("a Metal staging buffer of {} bytes", src.len())))?;
            let to = &dst.buf().0;
            // SAFETY: both ranges within their buffers.
            s.blit(&self.queue, |b| unsafe { b.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(&staging, 0, to, offset, src.len()) })
        })
    }

    fn alloc_host(&self, bytes: usize) -> Result<MetalHost, GpuError> {
        crate::fault::check("alloc_host")?;
        autoreleasepool(|_| {
            let (cap, mem) = match self.host_pool.take(bytes) {
                Some(taken) => taken,
                None => {
                    let cap = capacity(bytes);
                    let buf = self.new_buffer(cap)?;
                    self.host_pool.made(cap);
                    (cap, HostMem { buf, last_use: AtomicU64::new(0) })
                }
            };
            Ok(MetalHost { mem: Some(mem), cap, len: bytes, pool: self.host_pool.clone() })
        })
    }

    fn host_mut<'a>(&self, host: &'a mut MetalHost) -> Result<&'a mut [u8], GpuError> {
        crate::fault::check("host_mut")?;
        let mem = host.mem();
        let last = mem.last_use.load(Ordering::Acquire);
        if last > 0 {
            // serials start at 0; last_use holds the serial plus one
            autoreleasepool(|_| self.stream().wait_until(last - 1))?;
        }
        // SAFETY: the buffer holds `len` bytes, no copy from it pending.
        Ok(unsafe { core::slice::from_raw_parts_mut(mem.buf.ptr(), host.len) })
    }

    fn write_from_host(&self, dst: &mut MetalBlock, offset: usize, host: &MetalHost, range: core::ops::Range<usize>) -> Result<(), GpuError> {
        crate::fault::check("write_from_host")?;
        self.not_capturing("a write")?;
        assert!(range.start <= range.end && range.end <= host.len, "a copy past the host block's {} bytes", host.len);
        assert!(offset + range.len() <= dst.len, "a write past the block's {} bytes", dst.len);
        if range.is_empty() {
            return Ok(());
        }
        autoreleasepool(|_| {
            let mut s = self.stream();
            let mem = host.mem();
            let (from, to) = (&mem.buf.0, &dst.buf().0);
            // SAFETY: both ranges within their buffers; the host block is not
            // written before this command buffer completes (host_mut waits).
            s.blit(&self.queue, |b| unsafe { b.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(from, range.start, to, offset, range.len()) })?;
            let serial = s.open.as_ref().expect("the blit's command buffer").serial;
            mem.last_use.store(serial + 1, Ordering::Release);
            Ok(())
        })
    }

    fn read(&self, src: &MetalBlock, offset: usize, dst: &mut [u8]) -> Result<(), GpuError> {
        self.read_many(&mut [(src, offset, dst)])
    }

    fn read_many(&self, reads: &mut [(&MetalBlock, usize, &mut [u8])]) -> Result<(), GpuError> {
        crate::fault::check("read")?;
        self.not_capturing("a read")?;
        autoreleasepool(|_| self.stream().wait_until(u64::MAX))?;
        for (src, offset, dst) in reads.iter_mut() {
            assert!(*offset + dst.len() <= src.len, "a read past the block's {} bytes", src.len);
            // SAFETY: within the block; the work before has completed.
            unsafe { core::ptr::copy_nonoverlapping(src.buf().ptr().add(*offset), dst.as_mut_ptr(), dst.len()) };
        }
        Ok(())
    }

    fn compile_many(&self, source: &str, defines: &str, entries: &[&str]) -> Result<Vec<MetalKernel>, GpuError> {
        crate::fault::check("compile")?;
        self.not_capturing("a compile")?;
        autoreleasepool(|_| {
            let first = entries.first().copied().unwrap_or("module");
            let module = self.module(first, self.unit(source, defines))?;
            entries
                .iter()
                .map(|&name| {
                    let entry = module
                        .entries
                        .iter()
                        .find(|e| e.name == name)
                        .cloned()
                        .ok_or_else(|| GpuError::Compile { entry: name.to_string(), log: "no such kernel entry in the source".into() })?;
                    let function = module
                        .library
                        .newFunctionWithName(&NSString::from_str(name))
                        .ok_or_else(|| GpuError::Compile { entry: name.to_string(), log: "not in the compiled library".into() })?;
                    let pipeline = self
                        .device
                        .newComputePipelineStateWithFunction_error(&function)
                        .map_err(|e| GpuError::Compile { entry: name.to_string(), log: e.localizedDescription().to_string() })?;
                    if pipeline.threadExecutionWidth() != self.arch.warp as usize {
                        return Err(GpuError::Unavailable(format!(
                            "kernel `{name}` runs {} lanes in lockstep on {}, the kernels need {}",
                            pipeline.threadExecutionWidth(),
                            self.arch.name,
                            self.arch.warp
                        )));
                    }
                    Ok(MetalKernel { inner: Arc::new(KernelInner { pipeline, entry }) })
                })
                .collect()
        })
    }

    fn launch(&self, kernel: &MetalKernel, launch: Launch, args: &[Arg<'_, MetalBlock>]) -> Result<(), GpuError> {
        crate::fault::check("launch")?;
        assert!(args.len() <= MAX_ARGS, "a launch takes at most {MAX_ARGS} arguments");
        let entry = &kernel.inner.entry;
        assert_eq!(args.len(), entry.params.len(), "kernel `{}` takes {} arguments", entry.name, entry.params.len());
        let threads = (launch.block[0] * launch.block[1] * launch.block[2]) as usize;
        if threads > kernel.max_threads() {
            return Err(GpuError::Limit(format!("kernel `{}` runs at most {} threads per block, not {threads}", entry.name, kernel.max_threads())));
        }
        let shared = kernel.static_shared_bytes() + if entry.dynamic_shared { launch.shared_bytes as usize } else { 0 };
        if shared > self.arch.shared_bytes as usize {
            return Err(GpuError::Limit(format!("kernel `{}` takes {shared} bytes of shared memory, the device {}", entry.name, self.arch.shared_bytes)));
        }
        let mut bound = Vec::with_capacity(args.len());
        for (i, (arg, param)) in args.iter().zip(&entry.params).enumerate() {
            let b = match (*arg, param) {
                (Arg::Block(b, offset), Param::Buffer) => {
                    assert!(offset <= b.len, "argument offset {offset} past the block's {} bytes", b.len);
                    Bound::Buffer(b.buf().0.clone(), offset)
                }
                (Arg::U32(v), Param::Scalar(_)) => Bound::Bytes((v as u64).to_le_bytes(), 4),
                (Arg::I32(v), Param::Scalar(_)) => Bound::Bytes((v as u32 as u64).to_le_bytes(), 4),
                (Arg::F32(v), Param::Scalar(_)) => Bound::Bytes((v.to_bits() as u64).to_le_bytes(), 4),
                (Arg::Bits(v), Param::Scalar(_)) => Bound::Bytes(v.to_le_bytes(), 8),
                (Arg::F64(_), _) => panic!("kernel `{}`: argument {i} is an f64, which Metal does not have", entry.name),
                (_, Param::Buffer) => panic!("kernel `{}`: argument {i} is a value, the parameter a buffer", entry.name),
                (Arg::Block(..), Param::Scalar(t)) => panic!("kernel `{}`: argument {i} is a buffer, the parameter a `{t}`", entry.name),
            };
            bound.push(b);
        }
        {
            let me = std::thread::current().id();
            let mut capture = self.capture.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((t, recorded)) = capture.as_mut()
                && *t == me
            {
                recorded.push(Recorded { kernel: kernel.clone(), launch, args: bound });
                return Ok(());
            }
        }
        autoreleasepool(|_| self.encode(&mut self.stream(), kernel, &launch, &bound))
    }

    fn sync(&self) -> Result<(), GpuError> {
        crate::fault::check("sync")?;
        self.not_capturing("a wait")?;
        autoreleasepool(|_| self.stream().wait_until(u64::MAX))
    }

    fn capture(&self, record: impl FnOnce() -> Result<(), GpuError>) -> Result<MetalGraph, GpuError> {
        crate::fault::check("capture")?;
        /// Closes the capture on every way out, a panicking recording's too.
        struct Open<'a>(&'a Mutex<Option<(ThreadId, Vec<Recorded>)>>);
        impl Open<'_> {
            fn take(&self) -> Vec<Recorded> {
                self.0.lock().unwrap_or_else(|e| e.into_inner()).take().map(|(_, r)| r).unwrap_or_default()
            }
        }
        impl Drop for Open<'_> {
            fn drop(&mut self) {
                self.take();
            }
        }
        {
            let mut capture = self.capture.lock().unwrap_or_else(|e| e.into_inner());
            if capture.is_some() {
                return Err(GpuError::Capture("a capture is already open on this device".into()));
            }
            *capture = Some((std::thread::current().id(), Vec::new()));
        }
        let open = Open(&self.capture);
        let recorded = record();
        let launches = open.take();
        // the recording's error first; an empty recording replays as nothing
        recorded?;
        Ok(MetalGraph { launches })
    }

    fn replay(&self, graph: &MetalGraph) -> Result<(), GpuError> {
        crate::fault::check("replay")?;
        self.not_capturing("a replay")?;
        autoreleasepool(|_| {
            let mut s = self.stream();
            for r in &graph.launches {
                self.encode(&mut s, &r.kernel, &r.launch, &r.args)?;
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::HEADER;

    /// The device's description: an Apple GPU of family 7 or later, 32
    /// lanes, no f64, its cores counted.
    #[test]
    fn the_device_describes_itself() {
        let dev = MetalDevice::new(0).unwrap();
        let a = dev.arch();
        assert!(a.target.starts_with("apple"), "{a:?}");
        assert_eq!(a.warp, 32);
        assert!(!a.has_f64);
        assert!(a.multiprocessors >= 7, "{a:?}");
        assert!(a.shared_bytes >= 32 * 1024, "{a:?}");
        assert!(matches!(MetalDevice::new(99), Err(GpuError::Unavailable(_))));
    }

    /// Writes queued behind launches land after them, and a reused block is
    /// zeroed behind the launches that still read it.
    #[test]
    fn writes_and_zeroing_are_ordered_behind_launches() {
        let dev = MetalDevice::new(0).unwrap();
        let src = "KERNEL void add(GLOBAL float* x, GLOBAL const float* y, u32 n) { u32 i = global_x(); if (i < n) x[i] += y[i]; }";
        let k = dev.compile(src, "", "add").unwrap();
        let n = 1000u32;
        let x = dev.alloc(4 * n as usize).unwrap();
        let mut y = dev.alloc(4 * n as usize).unwrap();
        dev.write(&mut y, 0, bytemuck::cast_slice(&vec![1.0f32; n as usize])).unwrap();
        dev.launch(&k, Launch::linear(n, 256), &[Arg::Block(&x, 0), Arg::Block(&y, 0), Arg::U32(n)]).unwrap();
        // queued: lands between the two launches
        dev.write(&mut y, 0, bytemuck::cast_slice(&vec![2.0f32; n as usize])).unwrap();
        dev.launch(&k, Launch::linear(n, 256), &[Arg::Block(&x, 0), Arg::Block(&y, 0), Arg::U32(n)]).unwrap();
        let mut out = vec![0f32; n as usize];
        dev.read(&x, 0, bytemuck::cast_slice_mut(&mut out)).unwrap();
        assert!(out.iter().all(|&v| v == 3.0), "{:?}", &out[..4]);
        // y back to the pool while a launch reading it is queued, then taken and zeroed
        dev.launch(&k, Launch::linear(n, 256), &[Arg::Block(&x, 0), Arg::Block(&y, 0), Arg::U32(n)]).unwrap();
        drop(y);
        let z = dev.alloc(4 * n as usize).unwrap();
        dev.launch(&k, Launch::linear(n, 256), &[Arg::Block(&x, 0), Arg::Block(&z, 0), Arg::U32(n)]).unwrap();
        dev.read(&x, 0, bytemuck::cast_slice_mut(&mut out)).unwrap();
        assert!(out.iter().all(|&v| v == 5.0), "{:?}", &out[..4]);
        assert_eq!(dev.allocations(), 2);
    }

    /// Timed launches are summed per kernel, the largest first, and still
    /// compute; untimed ones are not counted; taking the sums resets them.
    #[test]
    fn launches_are_timed_per_kernel_while_timing_is_on() {
        let dev = MetalDevice::new(0).unwrap();
        let src = "KERNEL void add(GLOBAL float* x, u32 n) { u32 i = global_x(); if (i < n) x[i] += 1.0f; }\n\
                   KERNEL void slow(GLOBAL float* x, u32 n) { u32 i = global_x(); if (i < n) for (u32 k = 0; k < 4096u; ++k) x[i] = x[i] * 0.5f + 1.0f; }";
        let add = dev.compile(src, "", "add").unwrap();
        let slow = dev.compile(src, "", "slow").unwrap();
        let n = 1 << 20;
        let x = dev.alloc(4 * n as usize).unwrap();
        let args = [Arg::Block(&x, 0), Arg::U32(n)];
        dev.launch(&add, Launch::linear(n, 256), &args).unwrap();
        dev.time_kernels(true);
        for _ in 0..3 {
            dev.launch(&add, Launch::linear(n, 256), &args).unwrap();
        }
        dev.launch(&slow, Launch::linear(n, 256), &args).unwrap();
        dev.time_kernels(false);
        dev.launch(&add, Launch::linear(n, 256), &args).unwrap();
        let mut out = vec![0f32; n as usize];
        dev.read(&x, 0, bytemuck::cast_slice_mut(&mut out)).unwrap();
        assert!(out.iter().all(|&v| v == 3.0), "{:?}", &out[..4]);
        let times = dev.take_kernel_times();
        assert_eq!(times.iter().map(|t| (t.name.as_str(), t.launches)).collect::<Vec<_>>(), [("slow", 1), ("add", 3)], "{times:?}");
        assert!(times.iter().all(|t| t.seconds > 0.0), "{times:?}");
        assert!(dev.take_kernel_times().is_empty());
    }

    /// Every kernel source compiles on Metal: the factor's, the reduction's,
    /// the LM vectors' and the assembly's, in f32 with df32 as the wide
    /// scalar.
    #[test]
    fn every_kernel_source_compiles() {
        use crate::test_util::ba_like;
        use crate::{ASSEMBLY, GpuSchur, GpuSupernodal, LmVectors, WIDE, Wide};
        use arael_faer::schur::schur_symbolic;
        use arael_faer::supernodal::{SupernodalParams, SupernodalSymbolic, nd_block_order};
        let dev = Arc::new(MetalDevice::new(0).unwrap());
        let (h, _, _, elim) = ba_like(30, 200, 5);
        let sym = schur_symbolic(h.symbolic(), &elim).unwrap();
        let order = nd_block_order(&sym.s);
        let sn = SupernodalSymbolic::new(&sym.s, Some(&order), &SupernodalParams::default()).unwrap();
        GpuSupernodal::<f32, _>::new(&dev, &sn, &sym.s, &Default::default(), Wide::Df32).unwrap();
        GpuSchur::new(&dev, &sym, h.symbolic(), Wide::Df32).unwrap();
        LmVectors::new(&dev, &[0u32, 4, 8], Wide::Df32).unwrap();
        let defines = format!("{}#define SCALAR float\n", Wide::Df32.defines());
        dev.compile(&format!("{WIDE}\n{ASSEMBLY}"), &defines, "k_gather_params").unwrap();
    }

    /// The unit compiled is the rewritten one; an entry the rewrite cannot
    /// read is a compile error naming it.
    #[test]
    fn an_unreadable_entry_is_a_compile_error() {
        let dev = MetalDevice::new(0).unwrap();
        match dev.compile("KERNEL void k(float* x) { x[0] = 1.0f; }", "", "k") {
            Err(GpuError::Compile { log, .. }) => assert!(log.contains("pointer without GLOBAL"), "{log}"),
            other => panic!("{other:?}"),
        }
        let unit = dev.unit("KERNEL void k(GLOBAL float* x) { x[0] = 1.0f; }", "");
        assert!(unit.contains(HEADER));
    }
}
