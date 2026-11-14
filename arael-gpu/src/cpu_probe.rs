//! The CPU device's access probe: every load and store a kernel makes in
//! device memory, counted per launch and per section of the blocks it was
//! given ([`crate::Access`]).
//!
//! With the probe on, the CPU device compiles its kernel libraries with
//! clang's load and store tracing (`-fsanitize-coverage=trace-loads,
//! trace-stores`): the compiler calls a hook with the address of each
//! access of the optimized code, cpu.h forwards it to [`hook`], and the
//! launch running on the calling thread counts it. Vectorization is off in
//! those libraries, so an access is one element. The counts are those of
//! the host compiler's code: a GPU compiler may keep a value in a register
//! where this one reloads it, or the reverse.

use std::cell::Cell;
use std::collections::HashMap;
use std::ffi::c_void;

use crate::{Access, Section, Touch};

/// clang's options for a probed library, on top of the device's own: the
/// tracing hooks (which need an edge-coverage mode to be emitted at all),
/// and no vectorization.
pub(crate) const OPTIONS: &[&str] =
    &["-fsanitize-coverage=func,trace-pc-guard,trace-loads,trace-stores", "-fno-vectorize", "-fno-slp-vectorize"];

/// The compiler of the probed libraries: `ARAEL_CPU_PROBE_CXX`, else `clang++`.
pub(crate) fn compiler() -> String {
    std::env::var("ARAEL_CPU_PROBE_CXX").unwrap_or_else(|_| "clang++".to_string())
}

/// One section's counters over the running launch, for reads or writes.
#[derive(Clone, Default)]
struct Count {
    touch: Touch,
    /// The last access: its address, width and thread.
    prev: usize,
    prev_size: usize,
    prev_tid: u64,
    /// The highest thread seen so far (plus one), and its first address.
    threads_seen: u64,
    first_prev: usize,
    any: bool,
}

/// A described section of a block, with the launch each of its elements
/// was last read and written in.
struct Region {
    /// Its index in the block's description.
    index: usize,
    at: usize,
    end: usize,
    elem: usize,
    read_in: Vec<u32>,
    written_in: Vec<u32>,
}

impl Region {
    fn new(index: usize, at: usize, end: usize, elem: usize) -> Region {
        Region { index, at, end, elem: elem.max(1), read_in: Vec::new(), written_in: Vec::new() }
    }
}

/// A block's sections, sorted by offset, and one more region for the bytes
/// outside them.
struct Regions {
    sections: Vec<Region>,
    rest: Region,
}

/// A block argument of the running launch.
struct Block {
    base: usize,
    len: usize,
    arg: usize,
    /// Counters per region (the last one the rest's): reads, writes.
    counts: Vec<[Count; 2]>,
}

/// The probe of one CPU device.
#[derive(Default)]
pub(crate) struct Probe {
    /// The described blocks by their address.
    regions: HashMap<usize, Regions>,
    /// The running launch: its number and its block arguments.
    launch: u32,
    blocks: Vec<Block>,
    /// The Run of the calling thread (cpu.h), from the runtime's table.
    current: Option<unsafe extern "C" fn() -> *const u32>,
    /// The launches run and not yet taken.
    done: Vec<Vec<Access>>,
}

thread_local! {
    /// The probe of the launch running on this thread.
    static RUNNING: Cell<*mut Probe> = const { Cell::new(core::ptr::null_mut()) };
}

impl Probe {
    /// A probe that asks `current` (the runtime's) for the running thread.
    pub(crate) fn new(current: unsafe extern "C" fn() -> *const u32) -> Probe {
        Probe { current: Some(current), ..Default::default() }
    }

    /// The sections of the block at `base`, of `len` bytes.
    pub(crate) fn describe(&mut self, base: usize, len: usize, sections: &[Section]) {
        let mut regions: Vec<Region> =
            sections.iter().enumerate().filter(|(_, s)| s.bytes > 0).map(|(i, s)| Region::new(i, s.at, s.at + s.bytes, s.elem)).collect();
        regions.sort_by_key(|r| r.at);
        self.regions.insert(base, Regions { sections: regions, rest: Region::new(usize::MAX, 0, len, 4) });
    }

    /// The block at `base` is another's now: its description goes.
    pub(crate) fn forget(&mut self, base: usize) {
        self.regions.remove(&base);
    }

    /// `run` as one launch whose block arguments are `blocks` (argument
    /// index, address, bytes), its accesses counted.
    pub(crate) fn launch<R>(&mut self, blocks: &[(usize, usize, usize)], run: impl FnOnce() -> R) -> R {
        self.launch = self.launch.wrapping_add(1).max(1);
        self.blocks.clear();
        for &(arg, base, len) in blocks {
            if self.blocks.iter().any(|b| b.base == base) {
                continue;
            }
            // a block not described is one region of 4-byte elements
            let regions = self.regions.entry(base).or_insert_with(|| Regions { sections: Vec::new(), rest: Region::new(usize::MAX, 0, len, 4) });
            self.blocks.push(Block { base, len, arg, counts: vec![Default::default(); regions.sections.len() + 1] });
        }
        RUNNING.with(|r| r.set(self as *mut Probe));
        let out = run();
        RUNNING.with(|r| r.set(core::ptr::null_mut()));
        let mut accesses = Vec::new();
        for b in &self.blocks {
            let sections = &self.regions[&b.base].sections;
            for (i, c) in b.counts.iter().enumerate() {
                if c[0].any || c[1].any {
                    accesses.push(Access { arg: b.arg, section: sections.get(i).map(|r| r.index), read: c[0].touch, write: c[1].touch });
                }
            }
        }
        self.done.push(accesses);
        out
    }

    /// The launches run since the last call.
    pub(crate) fn take(&mut self) -> Vec<Vec<Access>> {
        std::mem::take(&mut self.done)
    }

    /// One access of `size` bytes at `addr`.
    #[inline]
    fn touch(&mut self, addr: usize, size: usize, write: bool) {
        let Some(b) = self.blocks.iter_mut().find(|b| addr >= b.base && addr - b.base < b.len) else { return };
        let off = addr - b.base;
        let r = self.regions.get_mut(&b.base).expect("a launch's blocks have their regions");
        // the section holding the offset: the last at or before it
        let i = r.sections.partition_point(|s| s.at <= off);
        let (index, region) = match i.checked_sub(1).filter(|&i| off < r.sections[i].end) {
            Some(i) => (i, &mut r.sections[i]),
            None => (r.sections.len(), &mut r.rest),
        };
        // SAFETY: a launch is running, so the runtime's Run is the thread's.
        let run = unsafe { (self.current.expect("a probe with its runtime"))() };
        // SAFETY: cpu.h's Run: grid[3], block[3], bid[3], threads, warp, cur.
        let (grid, bid, threads, cur) = unsafe { ([*run, *run.add(1)], [*run.add(6), *run.add(7), *run.add(8)], *run.add(9), *run.add(11)) };
        let block = (bid[2] as u64 * grid[1] as u64 + bid[1] as u64) * grid[0] as u64 + bid[0] as u64;
        let tid = block * threads as u64 + cur as u64;
        let c = &mut b.counts[index][write as usize];
        let t = &mut c.touch;
        let rel = (off - region.at) as u64;
        if !c.any {
            (t.lo, t.hi) = (rel, rel + size as u64 - 1);
        }
        t.lo = t.lo.min(rel);
        t.hi = t.hi.max(rel + size as u64 - 1);
        t.n += 1;
        t.bytes += size as u64;
        t.sizes |= (size as u8).min(16);
        // a new element of this launch
        let seen = if write { &mut region.written_in } else { &mut region.read_in };
        if seen.is_empty() {
            seen.resize((region.end - region.at).div_ceil(region.elem), 0);
        }
        let e = (off - region.at) / region.elem;
        if seen[e] != self.launch {
            seen[e] = self.launch;
            t.unique += 1;
        }
        if c.any && c.prev_tid == tid {
            if addr == c.prev + c.prev_size {
                t.seq += 1;
            } else if addr == c.prev {
                t.same += 1;
            }
        }
        // threads start in order: one past the highest seen is a new one
        if tid >= c.threads_seen {
            if c.any && tid == c.threads_seen {
                if addr == c.first_prev + size {
                    t.adjacent += 1;
                } else if addr == c.first_prev {
                    t.shared += 1;
                }
            }
            t.threads += 1;
            c.threads_seen = tid + 1;
            c.first_prev = addr;
        }
        (c.prev, c.prev_size, c.prev_tid, c.any) = (addr, size, tid, true);
    }
}

/// cpu.h's hook: an access of `info & 0xff` bytes at `addr`, a store with
/// bit 8 of `info`.
pub(crate) unsafe extern "C" fn hook(addr: *const c_void, info: u32) {
    let probe = RUNNING.with(|r| r.get());
    if !probe.is_null() {
        // SAFETY: set by `launch` on this thread for the launch's duration,
        // which holds the probe exclusively.
        unsafe { (*probe).touch(addr as usize, (info & 0xff) as usize, info & 0x100 != 0) };
    }
}
