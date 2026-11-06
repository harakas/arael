//! The CPU device: arael's GPU kernels compiled by the host's C++ compiler
//! and run on the CPU, for tests and CI on machines without a GPU.
//!
//! A kernel source is compiled as for a GPU, with `kernels/cpu.h` and
//! `arael_gpu.h`'s CPU branch in front, into a shared library loaded at run
//! time; every entry point gets a launcher that reads its arguments from
//! the launch's slots as the GPU drivers do. The launches run in the
//! runtime (`kernels/cpu_runtime.cpp`), a library loaded once per process:
//! a launch runs its blocks one after another on the calling thread, each
//! block's threads as fibers that switch at barriers. Device memory is host
//! memory; reads and writes are copies, launches finish before they
//! return, and a graph is the list of the launches it recorded.
//!
//! The compiler is `ARAEL_CPU_CXX`, else `CXX`, else `c++` (GCC or Clang,
//! C++17, POSIX `ucontext`, an ELF platform). Compiled libraries are kept
//! in the kernel cache (`ARAEL_KERNEL_CACHE`) under `cpu`.
//!
//! ```ignore
//! let dev = arael_gpu::cpu::CpuDevice::new(0)?;
//! let k = dev.compile("KERNEL void k(GLOBAL float* x) { x[global_x()] = 1.0f; }", "", "k")?;
//! ```

use core::ffi::{c_char, c_void};
use core::ptr::NonNull;
use std::alloc::Layout;
use std::collections::HashMap;
use std::ffi::CStr;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::ThreadId;

use crate::disk::DiskCache;
use crate::pool::{Pool, capacity};
use crate::{Arch, Arg, Device, GpuError, HEADER, Launch, MAX_ARGS};

/// The kernels' side of the CPU target, in front of every kernel source.
const CPU_H: &str = include_str!("../kernels/cpu.h");

/// The runtime's source, compiled with [`CPU_H`] in front.
const RUNTIME_SOURCE: &str = include_str!("../kernels/cpu_runtime.cpp");

/// The compiler's options: a shared library exporting what it marks alone
/// (a library shares nothing with another), C++17, contraction on with FMA
/// (the GPU compilers' default; GCC fuses from -O2 on), no fast math, and
/// no `_FORTIFY_SOURCE`, whose longjmp check refuses the fibers' switches.
const OPTIONS: &[&str] = &[
    "-x",
    "c++",
    "-std=c++17",
    "-O2",
    "-fPIC",
    "-shared",
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
    "-ffp-contract=fast",
    #[cfg(target_arch = "x86_64")]
    "-mfma",
    "-fno-fast-math",
    "-U_FORTIFY_SOURCE",
    "-D_FORTIFY_SOURCE=0",
    "-w",
];

/// [`GpuError::Unavailable`] on a CPU without FMA, which the kernels are
/// compiled for.
fn has_fma() -> Result<(), GpuError> {
    #[cfg(target_arch = "x86_64")]
    if !std::arch::is_x86_feature_detected!("fma") {
        return Err(GpuError::Unavailable("the CPU device needs FMA, which this CPU lacks".into()));
    }
    Ok(())
}

/// The host compiler: `ARAEL_CPU_CXX`, else `CXX`, else `c++`.
fn compiler() -> String {
    std::env::var("ARAEL_CPU_CXX").or_else(|_| std::env::var("CXX")).unwrap_or_else(|_| "c++".to_string())
}

/// The first line of the compiler's `--version`, for the cache key.
fn compiler_version() -> Result<String, GpuError> {
    static VERSION: OnceLock<Result<String, String>> = OnceLock::new();
    VERSION
        .get_or_init(|| {
            let cxx = compiler();
            let out = Command::new(&cxx).arg("--version").output().map_err(|e| format!("the CPU device's compiler `{cxx}`: {e}"))?;
            Ok(String::from_utf8_lossy(&out.stdout).lines().next().unwrap_or_default().to_string())
        })
        .clone()
        .map_err(GpuError::Unavailable)
}

/// Compilations in this process.
static COMPILES: AtomicUsize = AtomicUsize::new(0);

/// Kernel sources the host compiler has compiled in this process: those
/// neither a device's memory nor the disk cache held.
pub fn compiles() -> usize {
    COMPILES.load(Ordering::Relaxed)
}

/// A fresh path in the temporary directory with `extension`.
fn temporary(extension: &str) -> PathBuf {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("arael-cpu-{}-{n}.{extension}", std::process::id()))
}

/// Host memory standing in for device memory, 64-byte aligned, reached only
/// through its pointer (kernels write it behind shared references).
pub(crate) struct Mem {
    ptr: NonNull<u8>,
    cap: usize,
}

// SAFETY: a plain allocation owned by this value; access is ordered by the
// device's operations, as device memory is.
unsafe impl Send for Mem {}
unsafe impl Sync for Mem {}

impl Mem {
    fn new(cap: usize) -> Result<Mem, GpuError> {
        let layout = Layout::from_size_align(cap.max(1), 64).map_err(|e| GpuError::Limit(e.to_string()))?;
        // SAFETY: a layout of non-zero size.
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).ok_or_else(|| GpuError::Limit(format!("{cap} bytes of host memory")))?;
        Ok(Mem { ptr, cap })
    }
}

impl Drop for Mem {
    fn drop(&mut self) {
        // SAFETY: allocated in `new` with this layout.
        unsafe { std::alloc::dealloc(self.ptr.as_ptr(), Layout::from_size_align_unchecked(self.cap.max(1), 64)) }
    }
}

/// A block of the CPU device's memory, back to the device's pool when dropped.
pub struct CpuBlock {
    mem: Option<Mem>,
    len: usize,
    pool: Arc<Pool<Mem>>,
}

impl CpuBlock {
    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn ptr(&self) -> *mut u8 {
        self.mem.as_ref().expect("a live block").ptr.as_ptr()
    }
}

impl Drop for CpuBlock {
    fn drop(&mut self) {
        if let Some(mem) = self.mem.take() {
            self.pool.give(mem.cap, mem);
        }
    }
}

/// Host memory a copy to the device reads (pinned on a GPU, plain here).
pub struct CpuHost {
    mem: Option<Mem>,
    len: usize,
    pool: Arc<Pool<Mem>>,
}

impl CpuHost {
    /// Its size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether it holds no bytes.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for CpuHost {
    fn drop(&mut self) {
        if let Some(mem) = self.mem.take() {
            self.pool.give(mem.cap, mem);
        }
    }
}

/// A compiled source: the shared library its launchers are in.
struct Module {
    lib: libloading::Library,
}

/// The runtime's table (`arael_cpu::Runtime`), for the kernel libraries.
struct Runtime(*const c_void);

// SAFETY: a pointer to a constant table in a library never unloaded.
unsafe impl Send for Runtime {}
unsafe impl Sync for Runtime {}

/// The runtime: compiled (or found in the disk cache) and loaded at the
/// first call in the process, kept loaded until the process ends.
fn runtime() -> Result<&'static Runtime, GpuError> {
    static RUNTIME: OnceLock<Result<Runtime, GpuError>> = OnceLock::new();
    RUNTIME
        .get_or_init(|| {
            let source = format!("#define ARAEL_CPU_RUNTIME 1\n{CPU_H}\n{RUNTIME_SOURCE}");
            let (lib, _) = library(DiskCache::open("cpu-runtime").as_ref(), "arael_cpu_runtime", &source)?;
            // SAFETY: the runtime's entry, of this type.
            let table = unsafe { lib.get::<unsafe extern "C" fn() -> *const c_void>(b"arael_cpu_runtime\0") }
                .map_err(|e| GpuError::Driver(format!("the CPU runtime: {e}")))?;
            // SAFETY: returns a pointer to its constant table.
            let table = unsafe { table() };
            // the fibers wait in its code between launches
            std::mem::forget(lib);
            Ok(Runtime(table))
        })
        .as_ref()
        .map_err(Clone::clone)
}

/// What a library of `source` depends on.
fn cache_key(source: &str) -> Result<String, GpuError> {
    Ok(format!("{} {}\n{}\n{source}", compiler(), OPTIONS.join(" "), compiler_version()?))
}

/// The library of `source`, and whether it was compiled: from `disk`, else
/// compiled (`entry` names it in the compiler's messages) and stored there.
/// A cached entry the loader refuses is compiled again and written over.
fn library(disk: Option<&DiskCache>, entry: &str, source: &str) -> Result<(libloading::Library, bool), GpuError> {
    let key = cache_key(source)?;
    if let Some(lib) = disk.and_then(|d| d.load(key.as_bytes())).and_then(|lib| load(&lib).ok()) {
        return Ok((lib, false));
    }
    let lib = compile(entry, source)?;
    if let Some(d) = disk {
        d.store(key.as_bytes(), &lib);
    }
    Ok((load(&lib)?, true))
}

/// An entry point's launcher: the argument slots, the launch's grid, block
/// and dynamic shared bytes; 0 on success, else 1 with the error's text.
type Launcher = unsafe extern "C" fn(*mut *mut c_void, *const u32, *mut *const c_char) -> i32;

/// A compiled entry point.
#[derive(Clone)]
pub struct CpuKernel {
    /// Keeps the library its launcher is in loaded.
    _module: Arc<Module>,
    launcher: Launcher,
    name: Arc<str>,
}

impl core::fmt::Debug for CpuKernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuKernel").field("name", &self.name).finish_non_exhaustive()
    }
}

/// A launch as a graph keeps it: the kernel, its shape and its argument
/// slots (block addresses included, the blocks outliving the graph).
#[derive(Clone)]
struct Recorded {
    kernel: CpuKernel,
    launch: Launch,
    slots: [u64; MAX_ARGS],
}

/// The launches a capture recorded, run again by a replay.
pub struct CpuGraph {
    launches: Vec<Recorded>,
}

impl core::fmt::Debug for CpuDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CpuDevice").field("arch", &self.arch).finish_non_exhaustive()
    }
}

/// The CPU as a [`Device`].
pub struct CpuDevice {
    arch: Arch,
    /// Compiled modules by their full source.
    modules: Mutex<HashMap<String, Arc<Module>>>,
    /// The open capture: its thread and what it recorded.
    capture: Mutex<Option<(ThreadId, Vec<Recorded>)>>,
    /// Held by every copy and launch: they run one at a time, in the order
    /// of their calls, as on a GPU's stream.
    stream: Mutex<()>,
    pool: Arc<Pool<Mem>>,
    host_pool: Arc<Pool<Mem>>,
}

impl CpuDevice {
    /// The CPU device; `index` is 0 (there is one).
    pub fn new(index: usize) -> Result<Self, GpuError> {
        if index != 0 {
            return Err(GpuError::Unavailable(format!("the CPU device has index 0 only, not {index}")));
        }
        // a device needs FMA, the compiler and the runtime: no device without them
        has_fma()?;
        runtime()?;
        Ok(CpuDevice {
            arch: Arch { name: "CPU".into(), target: "host".into(), warp: 32, shared_bytes: 64 * 1024, multiprocessors: 1, has_f64: true },
            modules: Mutex::new(HashMap::new()),
            capture: Mutex::new(None),
            stream: Mutex::new(()),
            pool: Arc::new(Pool::new()),
            host_pool: Arc::new(Pool::new()),
        })
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

    /// The stream, held through a copy or a launch.
    fn stream(&self) -> std::sync::MutexGuard<'_, ()> {
        self.stream.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// The module of `source`: in memory when this device loaded it before,
    /// else from the disk cache, else compiled; given the runtime's table.
    fn module(&self, entry: &str, source: String) -> Result<Arc<Module>, GpuError> {
        if let Some(module) = self.modules.lock().unwrap_or_else(|e| e.into_inner()).get(&source) {
            return Ok(module.clone());
        }
        let (lib, compiled) = library(DiskCache::open(Self::RUNTIME).as_ref(), entry, &source)?;
        if compiled {
            COMPILES.fetch_add(1, Ordering::Relaxed);
        }
        // SAFETY: cpu.h's init, of this type.
        let init = unsafe { lib.get::<unsafe extern "C" fn(*const c_void)>(b"arael_cpu_init\0") }
            .map_err(|e| GpuError::Driver(format!("the CPU device's library of `{entry}`: {e}")))?;
        // SAFETY: the runtime's table, which outlives the library.
        unsafe { init(runtime()?.0) };
        let module = Arc::new(Module { lib });
        self.modules.lock().unwrap_or_else(|e| e.into_inner()).insert(source, module.clone());
        Ok(module)
    }
}

/// `source` compiled into a shared library's bytes.
fn compile(entry: &str, source: &str) -> Result<Vec<u8>, GpuError> {
    let (src, out) = (temporary("cpp"), temporary("so"));
    let io = |e: std::io::Error| GpuError::Driver(format!("the CPU device's compile of `{entry}`: {e}"));
    std::fs::write(&src, source).map_err(io)?;
    let cxx = compiler();
    let run = Command::new(&cxx).args(OPTIONS).arg(&src).arg("-o").arg(&out).output();
    let _ = std::fs::remove_file(&src);
    let run = run.map_err(|e| GpuError::Unavailable(format!("the CPU device's compiler `{cxx}`: {e}")))?;
    if !run.status.success() {
        let _ = std::fs::remove_file(&out);
        return Err(GpuError::Compile { entry: entry.to_string(), log: String::from_utf8_lossy(&run.stderr).into_owned() });
    }
    let lib = std::fs::read(&out).map_err(io);
    let _ = std::fs::remove_file(&out);
    lib
}

/// A shared library's bytes loaded. Its file is removed once loaded: the
/// mapping keeps the file's inode, so a later library cannot take it and
/// pass for this one.
fn load(lib: &[u8]) -> Result<libloading::Library, GpuError> {
    let path = temporary("so");
    std::fs::write(&path, lib).map_err(|e| GpuError::Driver(format!("the CPU device's library: {e}")))?;
    // SAFETY: a library this device compiled from a kernel source; its
    // initializers are the C++ library's and the kernels' statics.
    let loaded = unsafe { libloading::Library::new(&path) };
    let _ = std::fs::remove_file(&path);
    loaded.map_err(|e| GpuError::Driver(format!("the CPU device's library: {e}")))
}

/// Run one launch: its argument slots handed to the launcher.
fn run(kernel: &CpuKernel, launch: &Launch, slots: &[u64; MAX_ARGS]) -> Result<(), GpuError> {
    let mut slots = *slots;
    let mut params = [core::ptr::null_mut::<c_void>(); MAX_ARGS];
    for (p, v) in params.iter_mut().zip(slots.iter_mut()) {
        *p = v as *mut u64 as *mut c_void;
    }
    let [gx, gy, gz] = launch.grid;
    let [bx, by, bz] = launch.block;
    let shape = [gx, gy, gz, bx, by, bz, launch.shared_bytes];
    let mut error: *const c_char = core::ptr::null();
    // SAFETY: the launcher of this kernel's entry point; the parameters are
    // the caller's to match, the slots live through the call.
    let rc = unsafe { (kernel.launcher)(params.as_mut_ptr(), shape.as_ptr(), &mut error) };
    if rc != 0 {
        // SAFETY: the launcher sets a static NUL-terminated text with an error.
        let why = if error.is_null() { "failed".into() } else { unsafe { CStr::from_ptr(error) }.to_string_lossy() };
        return Err(GpuError::Driver(format!("the kernel `{}` on the CPU: {why}", kernel.name)));
    }
    Ok(())
}

impl Device for CpuDevice {
    type Block = CpuBlock;
    type Kernel = CpuKernel;
    type Graph = CpuGraph;
    type Host = CpuHost;
    const RUNTIME: &'static str = "cpu";

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

    fn alloc(&self, bytes: usize) -> Result<CpuBlock, GpuError> {
        crate::fault::check("alloc")?;
        self.not_capturing("an allocation")?;
        let mem = match self.pool.take(bytes) {
            Some((_, mem)) => mem,
            None => {
                let cap = capacity(bytes);
                let mem = Mem::new(cap)?;
                self.pool.made(cap);
                mem
            }
        };
        // SAFETY: the allocation holds at least `bytes`.
        unsafe { core::ptr::write_bytes(mem.ptr.as_ptr(), 0, bytes) };
        Ok(CpuBlock { mem: Some(mem), len: bytes, pool: self.pool.clone() })
    }

    fn write(&self, dst: &mut CpuBlock, offset: usize, src: &[u8]) -> Result<(), GpuError> {
        crate::fault::check("write")?;
        self.not_capturing("a write")?;
        let _stream = self.stream();
        assert!(offset + src.len() <= dst.len, "a write past the block's {} bytes", dst.len);
        // SAFETY: within the block (checked above), which `src` cannot alias.
        unsafe { core::ptr::copy_nonoverlapping(src.as_ptr(), dst.ptr().add(offset), src.len()) };
        Ok(())
    }

    fn alloc_host(&self, bytes: usize) -> Result<CpuHost, GpuError> {
        crate::fault::check("alloc_host")?;
        let mem = match self.host_pool.take(bytes) {
            Some((_, mem)) => mem,
            None => {
                let cap = capacity(bytes);
                let mem = Mem::new(cap)?;
                self.host_pool.made(cap);
                mem
            }
        };
        Ok(CpuHost { mem: Some(mem), len: bytes, pool: self.host_pool.clone() })
    }

    fn host_mut<'a>(&self, host: &'a mut CpuHost) -> Result<&'a mut [u8], GpuError> {
        crate::fault::check("host_mut")?;
        let mem = host.mem.as_ref().expect("a live host block");
        // SAFETY: the allocation holds `len` bytes; its contents may be
        // uninitialized bytes the caller writes before reading.
        Ok(unsafe { core::slice::from_raw_parts_mut(mem.ptr.as_ptr(), host.len) })
    }

    fn write_from_host(&self, dst: &mut CpuBlock, offset: usize, host: &CpuHost, range: core::ops::Range<usize>) -> Result<(), GpuError> {
        crate::fault::check("write_from_host")?;
        self.not_capturing("a write")?;
        let _stream = self.stream();
        assert!(range.end <= host.len, "a copy past the host block's {} bytes", host.len);
        assert!(offset + range.len() <= dst.len, "a write past the block's {} bytes", dst.len);
        let src = host.mem.as_ref().expect("a live host block").ptr.as_ptr();
        // SAFETY: both ranges within their allocations (checked above), which differ.
        unsafe { core::ptr::copy_nonoverlapping(src.add(range.start), dst.ptr().add(offset), range.len()) };
        Ok(())
    }

    fn read(&self, src: &CpuBlock, offset: usize, dst: &mut [u8]) -> Result<(), GpuError> {
        crate::fault::check("read")?;
        self.not_capturing("a read")?;
        let _stream = self.stream();
        assert!(offset + dst.len() <= src.len, "a read past the block's {} bytes", src.len);
        // SAFETY: within the block (checked above), which `dst` cannot alias.
        unsafe { core::ptr::copy_nonoverlapping(src.ptr().add(offset), dst.as_mut_ptr(), dst.len()) };
        Ok(())
    }

    fn read_many(&self, reads: &mut [(&CpuBlock, usize, &mut [u8])]) -> Result<(), GpuError> {
        crate::fault::check("read")?;
        self.not_capturing("a read")?;
        let _stream = self.stream();
        for (src, offset, dst) in reads.iter_mut() {
            assert!(*offset + dst.len() <= src.len, "a read past the block's {} bytes", src.len);
            // SAFETY: as in `read`.
            unsafe { core::ptr::copy_nonoverlapping(src.ptr().add(*offset), dst.as_mut_ptr(), dst.len()) };
        }
        Ok(())
    }

    fn compile_many(&self, source: &str, defines: &str, entries: &[&str]) -> Result<Vec<CpuKernel>, GpuError> {
        crate::fault::check("compile")?;
        self.not_capturing("a compile")?;
        let mut full = self.unit(source, defines);
        for e in entries {
            full.push_str(&format!(
                "ARAEL_CPU_EXPORT int arael_cpu_launch_{e}(void** p, const unsigned* l, const char** error) {{ return arael_cpu::launch({e}, p, l, error); }}\n"
            ));
        }
        let module = self.module(entries.first().copied().unwrap_or("module"), full)?;
        entries
            .iter()
            .map(|&e| {
                let symbol = format!("arael_cpu_launch_{e}\0");
                // SAFETY: the launcher generated above for `e`, of type `Launcher`.
                let launcher = unsafe { module.lib.get::<Launcher>(symbol.as_bytes()) }
                    .map_err(|err| GpuError::Driver(format!("the CPU device's launcher of `{e}`: {err}")))?;
                Ok(CpuKernel { _module: module.clone(), launcher: *launcher, name: e.into() })
            })
            .collect()
    }

    /// The CPU's unit: the target's define, the architecture's defines,
    /// cpu.h (ahead of `defines`, which its system headers must not see),
    /// `defines`, [`HEADER`], `source`.
    fn unit(&self, source: &str, defines: &str) -> String {
        format!("#define ARAEL_CPU_TARGET 1\n{}{CPU_H}\n{defines}\n{HEADER}\n{source}\n", self.arch.defines())
    }

    fn launch(&self, kernel: &CpuKernel, launch: Launch, args: &[Arg<'_, CpuBlock>]) -> Result<(), GpuError> {
        crate::fault::check("launch")?;
        assert!(args.len() <= MAX_ARGS, "a launch takes at most {MAX_ARGS} arguments");
        let mut slots = [0u64; MAX_ARGS];
        for (slot, arg) in slots.iter_mut().zip(args) {
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
            };
        }
        {
            let me = std::thread::current().id();
            let mut capture = self.capture.lock().unwrap_or_else(|e| e.into_inner());
            if let Some((t, recorded)) = capture.as_mut()
                && *t == me
            {
                recorded.push(Recorded { kernel: kernel.clone(), launch, slots });
                return Ok(());
            }
        }
        let _stream = self.stream();
        run(kernel, &launch, &slots)
    }

    fn sync(&self) -> Result<(), GpuError> {
        crate::fault::check("sync")?;
        self.not_capturing("a sync")
    }

    fn capture(&self, record: impl FnOnce() -> Result<(), GpuError>) -> Result<CpuGraph, GpuError> {
        crate::fault::check("capture")?;
        /// Closes the capture on every way out, a panicking recording's too.
        struct Open<'a>(&'a Mutex<Option<(ThreadId, Vec<Recorded>)>>);
        impl Open<'_> {
            fn close(&self) -> Vec<Recorded> {
                self.0.lock().unwrap_or_else(|e| e.into_inner()).take().map(|(_, l)| l).unwrap_or_default()
            }
        }
        impl Drop for Open<'_> {
            fn drop(&mut self) {
                self.close();
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
        let launches = open.close();
        // the recording's error first: it is why the capture broke, if it did
        recorded?;
        Ok(CpuGraph { launches })
    }

    fn replay(&self, graph: &CpuGraph) -> Result<(), GpuError> {
        crate::fault::check("replay")?;
        self.not_capturing("a replay")?;
        let _stream = self.stream();
        for l in &graph.launches {
            run(&l.kernel, &l.launch, &l.slots)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A block's sum through shared memory and barriers; threads past `n`
    /// add nothing.
    const BLOCK_SUM: &str = r#"
KERNEL void block_sum(GLOBAL const float* x, GLOBAL float* out, u32 n) {
    SHARED float s[256];
    u32 t = thread_x(), i = global_x();
    s[t] = i < n ? x[i] : 0.0f;
    BARRIER();
    for (u32 o = block_dim_x() / 2; o > 0; o >>= 1) {
        if (t < o) s[t] += s[t + o];
        BARRIER();
    }
    if (t == 0) out[block_x()] = s[0];
}
"#;

    /// `block_sum` of 1..=n in blocks of 256 on `dev`: the blocks' sums.
    fn block_sums(dev: &CpuDevice, k: &CpuKernel, n: u32) -> Vec<f32> {
        block_sums_in(dev, k, n, 256)
    }

    /// `block_sum` of 1..=n in blocks of `block` (a power of two up to 256).
    fn block_sums_in(dev: &CpuDevice, k: &CpuKernel, n: u32, block: u32) -> Vec<f32> {
        let x: Vec<f32> = (1..=n).map(|v| v as f32).collect();
        let blocks = n.div_ceil(block);
        let mut xb = dev.alloc(4 * n as usize).unwrap();
        dev.write(&mut xb, 0, bytemuck::cast_slice(&x)).unwrap();
        let out = dev.alloc(4 * blocks as usize).unwrap();
        dev.launch(k, Launch::linear(n, block), &[Arg::Block(&xb, 0), Arg::Block(&out, 0), Arg::U32(n)]).unwrap();
        let mut sums = vec![0f32; blocks as usize];
        dev.read(&out, 0, bytemuck::cast_slice_mut(&mut sums)).unwrap();
        sums
    }

    fn expected_block_sums(n: u32) -> Vec<f32> {
        expected_block_sums_in(n, 256)
    }

    fn expected_block_sums_in(n: u32, block: u32) -> Vec<f32> {
        (0..n.div_ceil(block)).map(|b| (b * block + 1..=((b + 1) * block).min(n)).map(|v| v as f32).sum()).collect()
    }

    #[test]
    fn a_block_shares_memory_and_meets_at_barriers() {
        let dev = CpuDevice::new(0).unwrap();
        let k = dev.compile(BLOCK_SUM, "", "block_sum").unwrap();
        assert_eq!(block_sums(&dev, &k, 1000), expected_block_sums(1000));
    }

    /// A thread that has returned no longer takes part in a barrier, as on
    /// the GPU: the others pass it.
    #[test]
    fn returned_threads_leave_the_barriers() {
        const EARLY: &str = r#"
KERNEL void early(GLOBAL u32* out) {
    SHARED u32 count[1];
    u32 t = thread_x();
    if (t == 0) count[0] = 0;
    BARRIER();
    if (t % 3 == 0) return;
    atomic_add(&count[0], 1u);
    BARRIER();
    if (t == 1) out[block_x()] = count[0];
}
"#;
        let dev = CpuDevice::new(0).unwrap();
        let k = dev.compile(EARLY, "", "early").unwrap();
        let out = dev.alloc(4 * 3).unwrap();
        dev.launch(&k, Launch { grid: [3, 1, 1], block: [64, 1, 1], shared_bytes: 0 }, &[Arg::Block(&out, 0)]).unwrap();
        let mut counts = [0u32; 3];
        dev.read(&out, 0, bytemuck::cast_slice_mut(&mut counts)).unwrap();
        assert_eq!(counts, [42; 3], "64 threads less the 22 multiples of 3");
    }

    /// A shuffle and a warp sum exchange the lanes' values within each warp.
    #[test]
    fn lanes_exchange_values_within_their_warp() {
        const LANES: &str = r#"
KERNEL void lanes(GLOBAL float* out) {
    u32 i = global_x();
    float v = (float)i;
    out[3 * i] = shfl_xor(v, 1u);
    out[3 * i + 1] = shfl(v, 5u);
    out[3 * i + 2] = warp_sum(v);
}
"#;
        let dev = CpuDevice::new(0).unwrap();
        let k = dev.compile(LANES, "", "lanes").unwrap();
        let out = dev.alloc(4 * 3 * 128).unwrap();
        dev.launch(&k, Launch::linear(128, 64), &[Arg::Block(&out, 0)]).unwrap();
        let mut v = vec![0f32; 3 * 128];
        dev.read(&out, 0, bytemuck::cast_slice_mut(&mut v)).unwrap();
        for i in 0..128usize {
            let warp = i / 32 * 32;
            assert_eq!(v[3 * i], (i ^ 1) as f32, "shfl_xor in {i}");
            assert_eq!(v[3 * i + 1], (warp + 5) as f32, "shfl in {i}");
            assert_eq!(v[3 * i + 2], (warp..warp + 32).sum::<usize>() as f32, "warp_sum in {i}");
        }
    }

    /// The launch's dynamic shared memory is the block's own scratch.
    #[test]
    fn dynamic_shared_memory_holds_the_launch_bytes() {
        const REVERSE: &str = r#"
KERNEL void reverse(GLOBAL float* x) {
    DYNAMIC_SHARED(float, s);
    u32 t = thread_x(), n = block_dim_x();
    s[t] = x[global_x()];
    BARRIER();
    x[global_x()] = s[n - 1 - t];
}
"#;
        let dev = CpuDevice::new(0).unwrap();
        let k = dev.compile(REVERSE, "", "reverse").unwrap();
        let host: Vec<f32> = (0..200).map(|v| v as f32).collect();
        let mut x = dev.alloc(4 * 200).unwrap();
        dev.write(&mut x, 0, bytemuck::cast_slice(&host)).unwrap();
        dev.launch(&k, Launch { grid: [2, 1, 1], block: [100, 1, 1], shared_bytes: 400 }, &[Arg::Block(&x, 0)]).unwrap();
        let mut out = vec![0f32; 200];
        dev.read(&x, 0, bytemuck::cast_slice_mut(&mut out)).unwrap();
        let want: Vec<f32> = (0..200).map(|i| (i / 100 * 100 + 99 - i % 100) as f32).collect();
        assert_eq!(out, want);
    }

    /// A barrier that a live thread never reaches ends the launch with an
    /// error, and the next launch on the thread starts its fibers afresh.
    #[test]
    fn a_barrier_never_met_is_an_error_and_the_device_goes_on() {
        const STUCK: &str = "KERNEL void stuck(GLOBAL u32* out) { if (thread_x() == 0) WARP_BARRIER(); else BARRIER(); out[thread_x()] = 1; }";
        let dev = CpuDevice::new(0).unwrap();
        let k = dev.compile(STUCK, "", "stuck").unwrap();
        let out = dev.alloc(4 * 32).unwrap();
        let err = dev.launch(&k, Launch::linear(32, 32), &[Arg::Block(&out, 0)]).unwrap_err();
        assert!(matches!(&err, GpuError::Driver(why) if why.contains("stuck") && why.contains("barrier")), "{err}");
        let sum = dev.compile(BLOCK_SUM, "", "block_sum").unwrap();
        assert_eq!(block_sums(&dev, &sum, 600), expected_block_sums(600));
    }

    /// A kernel library unloads (its device and kernels gone) while the
    /// thread's fibers live on: they wait in the runtime, not in it, so the
    /// next library's launches on the thread run. Each library launches
    /// larger blocks than the one before, starting fibers of its own.
    #[test]
    fn fibers_outlive_the_libraries_they_ran() {
        for (i, block) in [32, 64, 128, 256, 256].into_iter().enumerate() {
            let dev = CpuDevice::new(0).unwrap();
            let k = dev.compile(BLOCK_SUM, &format!("#define LIBRARY {i}\n"), "block_sum").unwrap();
            assert_eq!(block_sums_in(&dev, &k, 700, block), expected_block_sums_in(700, block), "library {i}");
        }
    }

    /// Two kernel libraries share nothing: the same inline function's
    /// shared array is each library's own.
    #[test]
    fn libraries_keep_their_statics_apart() {
        const SCRATCH: &str = r#"
DEVICE float* scratch() { SHARED float s[SIZE]; return s; }
KERNEL void put(GLOBAL float* out) {
    float* s = scratch();
    if (thread_x() == 0) { out[0] = s[0]; s[0] = 7.0f; }
}
"#;
        let dev = CpuDevice::new(0).unwrap();
        let out = dev.alloc(4).unwrap();
        for size in [4, 8] {
            let k = dev.compile(SCRATCH, &format!("#define SIZE {size}\n"), "put").unwrap();
            dev.launch(&k, Launch::linear(1, 32), &[Arg::Block(&out, 0)]).unwrap();
            let mut seen = [0f32];
            dev.read(&out, 0, bytemuck::cast_slice_mut(&mut seen)).unwrap();
            assert_eq!(seen, [0.0], "SIZE {size} saw another library's write");
        }
    }

    /// Threads launching on one device take turns; each gets its own sums.
    #[test]
    fn launches_from_several_threads_run_one_at_a_time() {
        let dev = CpuDevice::new(0).unwrap();
        let k = dev.compile(BLOCK_SUM, "", "block_sum").unwrap();
        std::thread::scope(|s| {
            for t in 0..4u32 {
                let (dev, k) = (&dev, &k);
                s.spawn(move || {
                    for round in 0..10 {
                        let n = 300 + 97 * t + round;
                        assert_eq!(block_sums(dev, k, n), expected_block_sums(n));
                    }
                });
            }
        });
    }

    /// A cached entry the loader refuses (not a library at all, its length
    /// and checksum right) is compiled again and written over.
    #[test]
    fn a_cached_library_the_loader_refuses_is_compiled_again() {
        let dir = std::env::temp_dir().join(format!("arael-cpu-cache-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let disk = DiskCache::at(dir.clone());
        let source = "ARAEL_CPU_EXPORT int arael_cpu_answer() { return 42; }";
        let source = format!("#define ARAEL_CPU_RUNTIME 1\n{CPU_H}\n{source}");
        let key = cache_key(&source).unwrap();
        disk.store(key.as_bytes(), b"not a library");
        let (lib, compiled) = library(Some(&disk), "answer", &source).unwrap();
        assert!(compiled);
        // SAFETY: the function above, of this type.
        assert_eq!(unsafe { lib.get::<unsafe extern "C" fn() -> i32>(b"arael_cpu_answer\0").unwrap()() }, 42);
        assert_ne!(disk.load(key.as_bytes()).expect("the entry written over"), b"not a library");
        // which the next load finds
        let (_, compiled) = library(Some(&disk), "answer", &source).unwrap();
        assert!(!compiled);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
