//! [`Traced`]: a [`Device`] that does what another does and writes down
//! what it was asked: every allocation, copy, kernel launch with its
//! arguments, graph capture and replay, as one JSON object per line of the
//! trace file. `tools/devmap` turns a trace into a browsable map of a
//! solve: which kernels run in what order on which sections of device
//! memory.
//!
//! The trace file is the process's: `ARAEL_GPU_TRACE=<path>` in the
//! environment, or [`start`]. A device is traced by wrapping it
//! ([`traced`]); nothing is recorded, and nothing costs, for a device that
//! is not wrapped.
//!
//! ```ignore
//! arael_gpu::trace::start("solve.trace")?;
//! let dev = arael_gpu::trace::traced(&Arc::new(CudaDevice::new(0)?));
//! let block = dev.alloc(1024)?;      // {"e":"alloc","block":1,"bytes":1024}
//! dev.launch(&k, launch, &[Arg::Block(&block, 0)])?;
//! ```
//!
//! The events, by their `e`:
//!
//! | `e` | fields |
//! |---|---|
//! | `device` | `dev`, `runtime`, `name`, `target`, `warp`, `integrated`, `has_f64`, `timed`: its launches are timed ([`time`]) |
//! | `unit` | `unit`, `entries`, `defines`, `source`: a kernel source compiled |
//! | `alloc` | `block`, `bytes` |
//! | `free` | `block` |
//! | `describe` | `block`, `name`, `sections`: `name`, `at`, `bytes`, `elem`, `ty`, `role`, `group`, `doc` each, `record` (elements per record) where above 1, and for index sections `content`: what kind of sequence their words are, and `sample`: their first and last words (`head`, `tail`) |
//! | `write` | `block`, `at`, `bytes`, `via` (`pageable` or `pinned`): host to device |
//! | `read` | `block`, `at`, `bytes`, `group`: device to host, the host waits; reads of one wait share `group` |
//! | `launch` | `k`, `unit`, `grid`, `threads`, `shared`, `lane`, `args`, `acc` when measured, `us` (its microseconds on the device) when timed |
//! | `capture` | `graph`, `events`: the launches and scopes recorded |
//! | `replay` | `graph`, `acc` (per launch) when measured, `us` (per launch) when timed |
//! | `scope`, `end` | `name`: a named scope opens, the last one closes |
//! | `fork`, `join`, `sync` | |
//!
//! A launch's `args` are `{"b": block, "at": byte offset}` or a scalar
//! (`{"u": 7}`, `{"i": -1}`, `{"f": 0.5}`, `{"d": 0.5}`, `{"bits": "0x.."}`).
//! An `acc` entry is one section's measured accesses ([`Access`]):
//! `{"b": block, "s": section index or -1, "r": {...}, "w": {...}}` with
//! [`Touch`]'s fields.

use std::any::Any;
use std::collections::HashMap;
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread::ThreadId;

use crate::{Access, Arch, Arg, Device, GpuError, Lane, Launch, MAX_ARGS, Section, Touch};

/// The trace file, when a trace is on.
static SINK: Mutex<Option<std::io::BufWriter<std::fs::File>>> = Mutex::new(None);

/// Whether the environment's `ARAEL_GPU_TRACE` was looked at.
static FROM_ENV: OnceLock<()> = OnceLock::new();

/// Whether the devices wrapped from now on time their launches ([`time`]).
static TIMED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Whether the devices traced from now on time each launch
/// ([`Device::time_launches`]): a launch event then has its `us`, a replay
/// its launches'. Also on with `ARAEL_GPU_TRACE_TIME` set (to anything but
/// 0) in the environment. Timed launches run one at a time, so a timed
/// trace is for the launches' times, not for the solve's.
pub fn time(on: bool) {
    TIMED.store(on, Ordering::Release);
}

/// Starts the process's trace in the file `path` (truncated), ending one
/// that was on.
pub fn start(path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
    FROM_ENV.get_or_init(|| ());
    let file = std::fs::File::create(path)?;
    *SINK.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::io::BufWriter::new(file));
    Ok(())
}

/// Ends the process's trace, its file complete.
pub fn stop() {
    FROM_ENV.get_or_init(|| ());
    if let Some(mut w) = SINK.lock().unwrap_or_else(|e| e.into_inner()).take() {
        let _ = w.flush();
    }
}

/// Whether a trace is on: started by [`start`], or by `ARAEL_GPU_TRACE`
/// (a path) in the environment, opened at the first call.
///
/// # Panics
///
/// When `ARAEL_GPU_TRACE` names a file that cannot be created.
pub fn active() -> bool {
    FROM_ENV.get_or_init(|| {
        if let Some(path) = std::env::var_os("ARAEL_GPU_TRACE").filter(|p| !p.is_empty()) {
            let file = std::fs::File::create(&path)
                .unwrap_or_else(|e| panic!("ARAEL_GPU_TRACE={}: the trace file cannot be created: {e}", path.to_string_lossy()));
            *SINK.lock().unwrap_or_else(|e| e.into_inner()) = Some(std::io::BufWriter::new(file));
        }
        if std::env::var_os("ARAEL_GPU_TRACE_TIME").is_some_and(|v| !v.is_empty() && v != "0") {
            time(true);
        }
    });
    SINK.lock().unwrap_or_else(|e| e.into_inner()).is_some()
}

/// One event's line into the trace; `flush`: the file is brought up to date.
fn emit(line: &str, flush: bool) {
    if let Some(w) = SINK.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        let _ = writeln!(w, "{line}");
        if flush {
            let _ = w.flush();
        }
    }
}

/// `s` as a JSON string.
fn quote(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// The traced twin of `dev`: the same one for the same device every time,
/// kept (with the device) for the process's life.
pub fn traced<D: Device + 'static>(dev: &Arc<D>) -> Arc<Traced<D>> {
    static TWINS: Mutex<Vec<(usize, Box<dyn Any + Send + Sync>)>> = Mutex::new(Vec::new());
    let key = Arc::as_ptr(dev) as usize;
    let mut twins = TWINS.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(t) = twins.iter().find(|(k, _)| *k == key).and_then(|(_, t)| t.downcast_ref::<Arc<Traced<D>>>()) {
        return t.clone();
    }
    let t = Arc::new(Traced::new(dev.clone()));
    twins.push((key, Box::new(t.clone())));
    t
}

/// Numbers blocks, units, graphs and read groups across the process's devices.
static NEXT: AtomicU64 = AtomicU64::new(1);

fn next() -> u64 {
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// A [`Device`] that forwards to `D` and writes every operation into the
/// process's trace (module docs).
pub struct Traced<D: Device> {
    inner: Arc<D>,
    id: u64,
    /// Compiled units by their source and defines.
    units: Mutex<HashMap<(String, String), u64>>,
    /// The open capture: its thread and the events recorded.
    capture: Mutex<Option<(ThreadId, Vec<String>)>>,
}

/// A traced device's block.
pub struct TracedBlock<B> {
    inner: B,
    id: u64,
}

impl<B> Drop for TracedBlock<B> {
    fn drop(&mut self) {
        emit(&format!("{{\"e\":\"free\",\"block\":{}}}", self.id), false);
    }
}

/// A traced device's kernel: the entry point's name with it.
pub struct TracedKernel<K> {
    inner: K,
    name: Arc<str>,
    unit: u64,
}

/// A traced device's graph.
pub struct TracedGraph<G> {
    inner: G,
    id: u64,
}

impl<D: Device> Traced<D> {
    /// `inner` traced; [`traced`] gives one twin per device.
    pub fn new(inner: Arc<D>) -> Self {
        let id = next();
        let a = inner.arch();
        let timed = TIMED.load(Ordering::Acquire);
        if timed {
            inner.time_launches(true);
        }
        emit(
            &format!(
                "{{\"e\":\"device\",\"dev\":{id},\"runtime\":{},\"name\":{},\"target\":{},\"warp\":{},\"integrated\":{},\"has_f64\":{},\"timed\":{timed}}}",
                quote(D::RUNTIME),
                quote(&a.name),
                quote(&a.target),
                a.warp,
                a.integrated,
                a.has_f64
            ),
            true,
        );
        Traced { inner, id, units: Mutex::new(HashMap::new()), capture: Mutex::new(None) }
    }

    /// The device underneath.
    pub fn inner(&self) -> &Arc<D> {
        &self.inner
    }

    /// An event of this device: into the open capture of the calling
    /// thread, else into the trace.
    fn event(&self, line: String, flush: bool) {
        let me = std::thread::current().id();
        let mut capture = self.capture.lock().unwrap_or_else(|e| e.into_inner());
        match capture.as_mut() {
            Some((t, events)) if *t == me => events.push(line),
            _ => emit(&line, flush),
        }
    }

    fn capturing(&self) -> bool {
        let me = std::thread::current().id();
        self.capture.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|(t, _)| *t == me)
    }
}

/// A sequence of at most this many pieces of `base + i * stride`, eight
/// words a piece at least, is described by its pieces.
const PIECES: usize = 4;
/// The widest record looked for in an index section that declares none.
const RECORD_MAX: usize = 16;
/// The records a section needs before a record width counts as found from
/// some of its fields; a width whose every field is regular takes three.
const RECORDS_MIN: usize = 16;

/// What kind of sequence index words are.
#[derive(Debug, PartialEq)]
enum Seq {
    Constant(u64),
    /// `base + i * stride`.
    Arithmetic(u64, i64),
    /// A few pieces of `base + i * stride`: (first index, base, stride) each.
    Pieces(Vec<(usize, u64, i64)>),
    /// Anything else. `runs`: the maximal stretches of one difference
    /// between neighbours. `gcd`: the greatest common divisor of the words.
    /// `hash`: of the words divided by `gcd`, so that a sequence and its
    /// multiples (offsets that are 9 times an index) share it.
    Irregular { monotone: bool, runs: usize, distinct: usize, gcd: u64, hash: u64 },
}

fn diff(a: u64, b: u64) -> i64 {
    b.wrapping_sub(a) as i64
}

/// Whether the words are `base + i * stride`, a constant included.
fn regular(mut w: impl Iterator<Item = u64>) -> bool {
    let (Some(a), Some(b)) = (w.next(), w.next()) else { return true };
    let (d, mut prev) = (diff(a, b), b);
    w.all(|v| {
        let fits = diff(prev, v) == d;
        prev = v;
        fits
    })
}

/// `w` as at most [`PIECES`] pieces of `base + i * stride`, or None where
/// it takes more, or fewer than eight words a piece (any eight words are
/// four pairs).
fn pieces(w: &[u64]) -> Option<Vec<(usize, u64, i64)>> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < w.len() {
        if out.len() == PIECES {
            return None;
        }
        let d = if i + 1 < w.len() { diff(w[i], w[i + 1]) } else { 0 };
        let mut last = i + 1;
        while last + 1 < w.len() && diff(w[last], w[last + 1]) == d {
            last += 1;
        }
        out.push((i, w[i], d));
        i = last + 1;
    }
    (w.len() >= 8 * out.len()).then_some(out)
}

fn gcd(a: u64, b: u64) -> u64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

impl Seq {
    fn of(w: &[u64]) -> Seq {
        if w.iter().all(|&v| v == w[0]) {
            return Seq::Constant(w[0]);
        }
        if regular(w.iter().copied()) {
            return Seq::Arithmetic(w[0], diff(w[0], w[1]));
        }
        if let Some(pieces) = pieces(w) {
            return Seq::Pieces(pieces);
        }
        let runs = 1 + w.windows(3).filter(|t| diff(t[0], t[1]) != diff(t[1], t[2])).count();
        let monotone = w.windows(2).all(|p| p[0] <= p[1]);
        let mut sorted = w.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        // not all zero here: the divisor is at least 1
        let g = w.iter().fold(0, |g, &v| gcd(g, v));
        // FNV-1a
        let hash = w.iter().fold(0xcbf2_9ce4_8422_2325u64, |h, &v| (v / g).to_le_bytes().iter().fold(h, |h, &b| (h ^ b as u64).wrapping_mul(0x0100_0000_01b3)));
        Seq::Irregular { monotone, runs, distinct: sorted.len(), gcd: g, hash }
    }

    /// As JSON, with the word count `n` after the kind where given (a
    /// section's own sequence; a record's field has none).
    fn json(&self, n: Option<usize>) -> String {
        let n = n.map_or(String::new(), |n| format!(",\"n\":{n}"));
        match self {
            Seq::Constant(v) => format!("{{\"kind\":\"constant\"{n},\"value\":{v}}}"),
            Seq::Arithmetic(base, stride) => format!("{{\"kind\":\"arithmetic\"{n},\"base\":{base},\"stride\":{stride}}}"),
            Seq::Pieces(pieces) => {
                let list: Vec<String> = pieces.iter().map(|(at, base, stride)| format!("[{at},{base},{stride}]")).collect();
                format!("{{\"kind\":\"pieces\"{n},\"pieces\":[{}]}}", list.join(","))
            }
            Seq::Irregular { monotone, runs, distinct, gcd, hash } => format!(
                "{{\"kind\":\"{}\"{n},\"runs\":{runs},\"distinct\":{distinct},\"gcd\":{gcd},\"hash\":\"{hash:016x}\"}}",
                if *monotone { "monotone" } else { "other" }
            ),
        }
    }
}

/// What kind of sequence the index words `w` are, as JSON. `record` is the
/// section's declared record width (1: none).
///
/// | `kind` | fields | meaning |
/// |---|---|---|
/// | `constant` | `value` | every word is `value` |
/// | `arithmetic` | `base`, `stride` | word i is `base + i * stride` |
/// | `pieces` | `pieces`: `[first index, base, stride]` each | at most four such stretches |
/// | `records` | `period`, `declared`, `fields`: a kind each | records of `period` words, each field (words j, j + period, ...) classified by itself |
/// | `monotone`, `other` | `runs`, `distinct`, `gcd`, `hash` | non-decreasing, or neither; see [`Seq::Irregular`] |
///
/// A section that declares no record width is `records` where a width up
/// to 16 makes every field constant or arithmetic (three records at
/// least), else where the width with the largest share of such fields has
/// one at least (16 records at least).
///
/// ```text
/// {"kind":"arithmetic","n":7776,"base":0,"stride":27}
/// {"kind":"records","n":147,"period":3,"declared":true,"fields":[
///   {"kind":"arithmetic","base":0,"stride":81},{"kind":"constant","value":9},{"kind":"arithmetic","base":0,"stride":9}]}
/// ```
fn content(w: &[u64], record: usize) -> String {
    let n = w.len();
    let field = |p: usize, j: usize| w[j..].iter().step_by(p).copied();
    let records = |p: usize, declared: bool| {
        let fields: Vec<String> = (0..p).map(|j| Seq::of(&field(p, j).collect::<Vec<u64>>()).json(None)).collect();
        format!("{{\"kind\":\"records\",\"n\":{n},\"period\":{p},\"declared\":{declared},\"fields\":[{}]}}", fields.join(","))
    };
    if regular(w.iter().copied()) {
        return Seq::of(w).json(Some(n));
    }
    if record > 1 && n.is_multiple_of(record) {
        return records(record, true);
    }
    if let Some(pieces) = pieces(w) {
        return Seq::Pieces(pieces).json(Some(n));
    }
    // (regular fields, width) of the best width so far
    let mut best = (0, 1);
    for p in 2..=RECORD_MAX {
        // three records at least: two fit any strides
        if n < 3 * p || !n.is_multiple_of(p) {
            continue;
        }
        let fits = (0..p).filter(|&j| regular(field(p, j))).count();
        if fits == p {
            return records(p, false);
        }
        if n >= RECORDS_MIN * p && fits * best.1 > best.0 * p {
            best = (fits, p);
        }
    }
    if best.0 > 0 { records(best.1, false) } else { Seq::of(w).json(Some(n)) }
}

/// How many of an index section's first words a trace keeps.
const SAMPLE_HEAD: usize = 64;
/// How many of its last words, when it has more than both together.
const SAMPLE_TAIL: usize = 16;

/// The first and the last of the index words `w`, as JSON: `head` holds all
/// of them when they are at most [`SAMPLE_HEAD`] + [`SAMPLE_TAIL`], else
/// the first [`SAMPLE_HEAD`], and `tail` the last [`SAMPLE_TAIL`].
///
/// ```text
/// {"head":[0,9,18,27]}
/// {"head":[0,9,18, ...],"tail":[ ..., 8982,8991]}
/// ```
fn sample(w: &[u64]) -> String {
    let list = |w: &[u64]| w.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(",");
    if w.len() <= SAMPLE_HEAD + SAMPLE_TAIL {
        format!("{{\"head\":[{}]}}", list(w))
    } else {
        format!("{{\"head\":[{}],\"tail\":[{}]}}", list(&w[..SAMPLE_HEAD]), list(&w[w.len() - SAMPLE_TAIL..]))
    }
}

/// One measured touch as JSON.
fn touch(t: &Touch) -> String {
    format!(
        "{{\"n\":{},\"bytes\":{},\"unique\":{},\"lo\":{},\"hi\":{},\"threads\":{},\"seq\":{},\"same\":{},\"adjacent\":{},\"shared\":{},\"sizes\":{}}}",
        t.n, t.bytes, t.unique, t.lo, t.hi, t.threads, t.seq, t.same, t.adjacent, t.shared, t.sizes
    )
}

/// A launch's measured accesses as JSON, its block arguments' ids in `blocks`.
fn accesses(acc: &[Access], blocks: &[u64]) -> String {
    let items: Vec<String> = acc
        .iter()
        .map(|a| {
            let s = a.section.map_or(-1, |s| s as i64);
            let (r, w) = (if a.read.n > 0 { touch(&a.read) } else { "null".into() }, if a.write.n > 0 { touch(&a.write) } else { "null".into() });
            format!("{{\"b\":{},\"s\":{s},\"r\":{r},\"w\":{w}}}", blocks.get(a.arg).copied().unwrap_or(0))
        })
        .collect();
    format!("[{}]", items.join(","))
}

impl<D: Device> Device for Traced<D> {
    type Block = TracedBlock<D::Block>;
    type Kernel = TracedKernel<D::Kernel>;
    type Graph = TracedGraph<D::Graph>;
    type Host = D::Host;

    const RUNTIME: &'static str = D::RUNTIME;

    fn arch(&self) -> &Arch {
        self.inner.arch()
    }

    fn alloc(&self, bytes: usize) -> Result<Self::Block, GpuError> {
        let inner = self.inner.alloc(bytes)?;
        let id = next();
        emit(&format!("{{\"e\":\"alloc\",\"dev\":{},\"block\":{id},\"bytes\":{bytes}}}", self.id), false);
        Ok(TracedBlock { inner, id })
    }

    fn allocations(&self) -> usize {
        self.inner.allocations()
    }

    fn reserved_bytes(&self) -> usize {
        self.inner.reserved_bytes()
    }

    fn host_allocations(&self) -> usize {
        self.inner.host_allocations()
    }

    fn write(&self, dst: &mut Self::Block, offset: usize, src: &[u8]) -> Result<(), GpuError> {
        self.inner.write(&mut dst.inner, offset, src)?;
        emit(&format!("{{\"e\":\"write\",\"block\":{},\"at\":{offset},\"bytes\":{},\"via\":\"pageable\"}}", dst.id, src.len()), false);
        Ok(())
    }

    fn alloc_host(&self, bytes: usize) -> Result<Self::Host, GpuError> {
        self.inner.alloc_host(bytes)
    }

    fn host_mut<'a>(&self, host: &'a mut Self::Host) -> Result<&'a mut [u8], GpuError> {
        self.inner.host_mut(host)
    }

    fn write_from_host(&self, dst: &mut Self::Block, offset: usize, host: &Self::Host, range: core::ops::Range<usize>) -> Result<(), GpuError> {
        let bytes = range.len();
        self.inner.write_from_host(&mut dst.inner, offset, host, range)?;
        emit(&format!("{{\"e\":\"write\",\"block\":{},\"at\":{offset},\"bytes\":{bytes},\"via\":\"pinned\"}}", dst.id), false);
        Ok(())
    }

    fn read(&self, src: &Self::Block, offset: usize, dst: &mut [u8]) -> Result<(), GpuError> {
        self.inner.read(&src.inner, offset, dst)?;
        emit(&format!("{{\"e\":\"read\",\"block\":{},\"at\":{offset},\"bytes\":{},\"group\":{}}}", src.id, dst.len(), next()), true);
        Ok(())
    }

    fn read_many(&self, reads: &mut [(&Self::Block, usize, &mut [u8])]) -> Result<(), GpuError> {
        let group = next();
        let lines: Vec<String> =
            reads.iter().map(|(b, at, dst)| format!("{{\"e\":\"read\",\"block\":{},\"at\":{at},\"bytes\":{},\"group\":{group}}}", b.id, dst.len())).collect();
        let mut inner: Vec<(&D::Block, usize, &mut [u8])> = reads.iter_mut().map(|(b, at, dst)| (&b.inner, *at, &mut **dst)).collect();
        self.inner.read_many(&mut inner)?;
        for (i, l) in lines.iter().enumerate() {
            emit(l, i + 1 == lines.len());
        }
        Ok(())
    }

    fn compile_many(&self, source: &str, defines: &str, entries: &[&str]) -> Result<Vec<Self::Kernel>, GpuError> {
        let kernels = self.inner.compile_many(source, defines, entries)?;
        let unit = {
            let mut units = self.units.lock().unwrap_or_else(|e| e.into_inner());
            *units.entry((source.to_string(), defines.to_string())).or_insert_with(next)
        };
        let names: Vec<String> = entries.iter().map(|e| quote(e)).collect();
        emit(
            &format!(
                "{{\"e\":\"unit\",\"dev\":{},\"unit\":{unit},\"entries\":[{}],\"defines\":{},\"source\":{}}}",
                self.id,
                names.join(","),
                quote(&format!("{}{defines}", self.inner.arch().defines())),
                quote(source)
            ),
            false,
        );
        Ok(kernels.into_iter().zip(entries).map(|(inner, &e)| TracedKernel { inner, name: e.into(), unit }).collect())
    }

    fn unit(&self, source: &str, defines: &str) -> String {
        self.inner.unit(source, defines)
    }

    fn launch(&self, kernel: &Self::Kernel, launch: Launch, args: &[Arg<'_, Self::Block>]) -> Result<(), GpuError> {
        self.launch_on(Lane::Main, kernel, launch, args)
    }

    fn launch_on(&self, lane: Lane, kernel: &Self::Kernel, launch: Launch, args: &[Arg<'_, Self::Block>]) -> Result<(), GpuError> {
        assert!(args.len() <= MAX_ARGS, "a launch takes at most {MAX_ARGS} arguments");
        let mut inner = [Arg::U32(0); MAX_ARGS];
        let mut blocks = [0u64; MAX_ARGS];
        let mut shown = Vec::with_capacity(args.len());
        for (i, a) in args.iter().enumerate() {
            inner[i] = match *a {
                Arg::Block(b, at) => {
                    blocks[i] = b.id;
                    shown.push(format!("{{\"b\":{},\"at\":{at}}}", b.id));
                    Arg::Block(&b.inner, at)
                }
                Arg::U32(v) => {
                    shown.push(format!("{{\"u\":{v}}}"));
                    Arg::U32(v)
                }
                Arg::I32(v) => {
                    shown.push(format!("{{\"i\":{v}}}"));
                    Arg::I32(v)
                }
                Arg::F32(v) => {
                    shown.push(if v.is_finite() { format!("{{\"f\":{v}}}") } else { format!("{{\"f\":\"{v}\"}}") });
                    Arg::F32(v)
                }
                Arg::F64(v) => {
                    shown.push(if v.is_finite() { format!("{{\"d\":{v}}}") } else { format!("{{\"d\":\"{v}\"}}") });
                    Arg::F64(v)
                }
                Arg::Bits(v) => {
                    shown.push(format!("{{\"bits\":\"{v:#018x}\"}}"));
                    Arg::Bits(v)
                }
                Arg::Bytes(bytes) => {
                    shown.push(format!("{{\"table\":\"{} bytes\"}}", bytes.len()));
                    Arg::Bytes(bytes)
                }
            };
        }
        match lane {
            Lane::Main => self.inner.launch(&kernel.inner, launch, &inner[..args.len()])?,
            Lane::Side => self.inner.launch_on(lane, &kernel.inner, launch, &inner[..args.len()])?,
        }
        let mut line = format!(
            "{{\"e\":\"launch\",\"k\":{},\"unit\":{},\"grid\":[{},{},{}],\"threads\":[{},{},{}],\"shared\":{},\"lane\":\"{}\",\"args\":[{}]",
            quote(&kernel.name),
            kernel.unit,
            launch.grid[0],
            launch.grid[1],
            launch.grid[2],
            launch.block[0],
            launch.block[1],
            launch.block[2],
            launch.shared_bytes,
            if lane == Lane::Main { "main" } else { "side" },
            shown.join(",")
        );
        if !self.capturing() {
            if let Some(acc) = self.inner.take_accesses().pop() {
                line.push_str(&format!(",\"acc\":{}", accesses(&acc, &blocks)));
            }
            // a timed device's time of the launch, in microseconds
            if let Some(seconds) = self.inner.take_times().pop() {
                line.push_str(&format!(",\"us\":{:.2}", seconds * 1e6));
            }
        }
        line.push('}');
        self.event(line, false);
        Ok(())
    }

    fn fork(&self) -> Result<(), GpuError> {
        self.inner.fork()?;
        self.event("{\"e\":\"fork\"}".into(), false);
        Ok(())
    }

    fn join(&self) -> Result<(), GpuError> {
        self.inner.join()?;
        self.event("{\"e\":\"join\"}".into(), false);
        Ok(())
    }

    fn sync(&self) -> Result<(), GpuError> {
        self.inner.sync()?;
        emit("{\"e\":\"sync\"}", true);
        Ok(())
    }

    fn capture(&self, record: impl FnOnce() -> Result<(), GpuError>) -> Result<Self::Graph, GpuError> {
        {
            let mut capture = self.capture.lock().unwrap_or_else(|e| e.into_inner());
            if capture.is_some() {
                return Err(GpuError::Capture("a capture is already open on this device".into()));
            }
            *capture = Some((std::thread::current().id(), Vec::new()));
        }
        let graph = self.inner.capture(record);
        let events = self.capture.lock().unwrap_or_else(|e| e.into_inner()).take().map(|(_, e)| e).unwrap_or_default();
        let inner = graph?;
        let id = next();
        emit(&format!("{{\"e\":\"capture\",\"graph\":{id},\"events\":[{}]}}", events.join(",")), false);
        Ok(TracedGraph { inner, id })
    }

    fn replay(&self, graph: &Self::Graph) -> Result<(), GpuError> {
        self.inner.replay(&graph.inner)?;
        // the measured launches of the replay carry their arguments' indices only:
        // the capture's events name the blocks
        let acc = self.inner.take_accesses();
        // a timed device's times of the graph's launches, in microseconds
        let times = self.inner.take_times();
        let us = if times.is_empty() { String::new() } else { format!(",\"us\":[{}]", times.iter().map(|s| format!("{:.2}", s * 1e6)).collect::<Vec<_>>().join(",")) };
        if acc.is_empty() {
            emit(&format!("{{\"e\":\"replay\",\"graph\":{}{us}}}", graph.id), false);
        } else {
            let per: Vec<String> = acc
                .iter()
                .map(|launch| {
                    let items: Vec<String> = launch
                        .iter()
                        .map(|a| {
                            let s = a.section.map_or(-1, |s| s as i64);
                            let (r, w) = (if a.read.n > 0 { touch(&a.read) } else { "null".into() }, if a.write.n > 0 { touch(&a.write) } else { "null".into() });
                            format!("{{\"arg\":{},\"s\":{s},\"r\":{r},\"w\":{w}}}", a.arg)
                        })
                        .collect();
                    format!("[{}]", items.join(","))
                })
                .collect();
            emit(&format!("{{\"e\":\"replay\",\"graph\":{},\"acc\":[{}]{us}}}", graph.id, per.join(",")), false);
        }
        Ok(())
    }

    fn describe(&self, block: &Self::Block, name: &str, sections: &dyn Fn() -> Vec<Section>) {
        let list = sections();
        let items: Vec<String> = list
            .iter()
            .map(|s| {
                // an index section's words as they are on the device now: what
                // kind of sequence they are, and the first and last of them
                let content = match (s.ty.as_str(), s.elem) {
                    ("u32" | "u64", 4 | 8) if s.bytes >= s.elem => {
                        let mut bytes = vec![0u8; s.bytes];
                        match self.inner.read(&block.inner, s.at, &mut bytes) {
                            Ok(()) => {
                                let words: Vec<u64> = if s.elem == 4 {
                                    bytes.as_chunks::<4>().0.iter().map(|c| u32::from_le_bytes(*c) as u64).collect()
                                } else {
                                    bytes.as_chunks::<8>().0.iter().map(|c| u64::from_le_bytes(*c)).collect()
                                };
                                format!(",\"content\":{},\"sample\":{}", content(&words, s.record), sample(&words))
                            }
                            Err(_) => String::new(),
                        }
                    }
                    _ => String::new(),
                };
                let record = if s.record > 1 { format!(",\"record\":{}", s.record) } else { String::new() };
                format!(
                    "{{\"name\":{},\"at\":{},\"bytes\":{},\"elem\":{},\"ty\":{},\"role\":{},\"group\":{},\"doc\":{}{record}{content}}}",
                    quote(&s.name),
                    s.at,
                    s.bytes,
                    s.elem,
                    quote(&s.ty),
                    quote(&s.role),
                    quote(&s.group),
                    quote(&s.doc)
                )
            })
            .collect();
        emit(&format!("{{\"e\":\"describe\",\"block\":{},\"name\":{},\"sections\":[{}]}}", block.id, quote(name), items.join(",")), false);
        self.inner.describe(&block.inner, name, &|| list.clone());
    }

    fn scope_begin(&self, name: &str) {
        self.event(format!("{{\"e\":\"scope\",\"name\":{}}}", quote(name)), false);
    }

    fn scope_end(&self) {
        let flush = !self.capturing();
        self.event("{\"e\":\"end\"}".into(), flush);
    }

    fn take_accesses(&self) -> Vec<Vec<Access>> {
        Vec::new()
    }

    /// The device underneath is timed; its times go into the trace's
    /// launch and replay events, so [`Device::take_times`] has none here.
    fn time_launches(&self, on: bool) {
        self.inner.time_launches(on);
    }

    fn stamp(&self, slot: usize) -> Result<(), GpuError> {
        self.inner.stamp(slot)
    }

    fn span(&self, from: usize, to: usize) -> Result<Option<f64>, GpuError> {
        self.inner.span(from, to)
    }
}

#[cfg(test)]
mod tests {
    use super::{content, sample};

    /// Index words are told apart by the sequence they are: one value, a
    /// base and a stride, a few pieces of that, records whose fields each
    /// are classified, a non-decreasing list, anything else.
    #[test]
    fn index_words_are_classified_by_their_sequence() {
        assert_eq!(content(&[3; 10], 1), r#"{"kind":"constant","n":10,"value":3}"#);
        let arithmetic: Vec<u64> = (0..100).map(|i| 12 + 9 * i).collect();
        assert_eq!(content(&arithmetic, 1), r#"{"kind":"arithmetic","n":100,"base":12,"stride":9}"#);
        let falling: Vec<u64> = (0..5).map(|i| 40 - 4 * i).collect();
        assert_eq!(content(&falling, 1), r#"{"kind":"arithmetic","n":5,"base":40,"stride":-4}"#);
        // block widths: 138 blocks of 9, then 15 of 3
        let widths: Vec<u64> = (0..153).map(|i| if i < 138 { 9 } else { 3 }).collect();
        assert_eq!(content(&widths, 1), r#"{"kind":"pieces","n":153,"pieces":[[0,9,0],[138,3,0]]}"#);
        // two records say nothing about their fields, and eight words are four pairs: a table
        let table: Vec<u64> = vec![256, 2048, 2816, 96256, 1496064, 1497856, 1498880, 1592320];
        assert!(content(&table, 1).starts_with(r#"{"kind":"monotone","n":8,"runs":"#), "{}", content(&table, 1));
        // row pointers of rows of 3 or 4 entries
        let mut ptr = vec![0u64];
        for i in 0..20 {
            ptr.push(ptr[i] + 3 + (i as u64 % 2));
        }
        assert!(content(&ptr, 1).starts_with(r#"{"kind":"monotone","n":21,"runs":"#), "{}", content(&ptr, 1));
        let scattered: Vec<u64> = (0..100).map(|i| (i * 37 + 11) % 100).collect();
        let c = content(&scattered, 1);
        assert!(c.starts_with(r#"{"kind":"other","n":100,"runs":"#) && c.contains(r#""distinct":100,"gcd":1,"hash":""#), "{c}");
    }

    /// Records are classified a field at a time: by the declared width, or
    /// by the width that makes the most fields regular.
    #[test]
    fn a_record_s_fields_are_classified_each() {
        // (base, stride, first): every field regular, found without a declaration from three records on
        let tiles: Vec<u64> = (0..50).flat_map(|i| [45 * i, 9, 3 * i]).collect();
        let regular = r#"{"kind":"records","n":150,"period":3,"declared":false,"fields":[{"kind":"arithmetic","base":0,"stride":45},{"kind":"constant","value":9},{"kind":"arithmetic","base":0,"stride":3}]}"#;
        assert_eq!(content(&tiles, 1), regular);
        assert_eq!(content(&tiles, 3), regular.replace("\"declared\":false", "\"declared\":true"));
        // the base irregular (tiles of points seen by 2 to 5 cameras), the stride and the first index regular
        let (mut base, mut x) = (30618u64, 12345u64);
        let points: Vec<u64> = (0..40u64)
            .flat_map(|i| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                base += 27 * (2 + (x >> 33) % 4) + 9;
                [base, 3, 3348 + 3 * i]
            })
            .collect();
        let c = content(&points, 3);
        assert!(c.starts_with(r#"{"kind":"records","n":120,"period":3,"declared":true,"fields":[{"kind":"monotone","runs":"#), "{c}");
        assert!(c.ends_with(r#"{"kind":"constant","value":3},{"kind":"arithmetic","base":3348,"stride":3}]}"#), "{c}");
        // undeclared: the width with the largest share of regular fields, from 16 records on
        assert_eq!(content(&points, 1), c.replace("\"declared\":true", "\"declared\":false"));
        assert!(content(&points[..3 * 15], 1).starts_with(r#"{"kind":"other","n":45,"#), "15 records are too few: {}", content(&points[..45], 1));
        // a declared width holds whatever the fields are
        let c = content(&[5, 1, 9, 2, 7, 3], 2);
        assert!(c.starts_with(r#"{"kind":"records","n":6,"period":2,"declared":true,"fields":[{"kind":"other","#), "{c}");
    }

    /// A sequence and its multiples share a hash, and differ in their
    /// greatest common divisor: an offset array that is 9 times an index
    /// array is found by it.
    #[test]
    fn a_sequence_and_its_multiples_share_a_hash() {
        let hash = |c: &str| c.split("\"hash\":\"").nth(1).unwrap().split('"').next().unwrap().to_string();
        let slots: Vec<u64> = (0..200u64).map(|i| (i * i + 3 * i) % 47 + 1).collect();
        let offsets: Vec<u64> = slots.iter().map(|s| 9 * s).collect();
        let other: Vec<u64> = (0..200u64).map(|i| (i * i + 5 * i) % 47 + 1).collect();
        let (a, b, c) = (content(&slots, 1), content(&offsets, 1), content(&other, 1));
        assert!(a.contains("\"gcd\":1,") && b.contains("\"gcd\":9,"), "{a}\n{b}");
        assert_eq!(hash(&a), hash(&b));
        assert_ne!(hash(&a), hash(&c));
    }

    /// A trace keeps the words of a short index section whole, and of a
    /// long one the first 64 and the last 16.
    #[test]
    fn an_index_section_s_first_and_last_words_are_kept() {
        assert_eq!(sample(&[256, 2048, 2816]), r#"{"head":[256,2048,2816]}"#);
        let short: Vec<u64> = (0..80).map(|i| 9 * i).collect();
        let kept = sample(&short);
        assert!(kept.starts_with(r#"{"head":[0,9,18,"#) && kept.ends_with(",702,711]}") && !kept.contains("tail"), "80 words fit: {kept}");
        let long: Vec<u64> = (0..1000).map(|i| 9 * i).collect();
        let kept = sample(&long);
        let (head, tail) = kept.split_once(r#"],"tail":["#).unwrap_or_else(|| panic!("a head and a tail: {kept}"));
        assert!(head.starts_with(r#"{"head":[0,9,18,"#) && head.ends_with(",558,567"), "the first 64: {head}");
        assert_eq!(head.matches(',').count() + 1, 64);
        assert!(tail.starts_with("8856,8865,") && tail.ends_with(",8982,8991]}"), "the last 16: {tail}");
        assert_eq!(tail.matches(',').count() + 1, 16);
    }
}
