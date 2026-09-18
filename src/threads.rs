//! Per-thread block stores for the threaded cost and assembly sweeps,
//! which a `#[arael(root, par)]` root gets under the `rayon` feature.
//!
//! The macro generates one store type per root: the same block arrays
//! the sequential path writes, one set per thread. A [`Cut`] divides
//! each top-level walk into one contiguous slot range per store, so a
//! store holds only the blocks its own range writes. The model is
//! read-only during a sweep; each thread writes only its own store; a
//! serial gather after the join adds the gradients and scatters the
//! blocks into the Hessian. A solve's [`Context`] owns the stores,
//! type-erased, so the root carries no field for them; the generated
//! code takes them back by their type. Without the feature, or without
//! `par`, the cut stays empty and the one store is the whole model.

use std::any::Any;
use std::time::{Duration, Instant};

/// A container a sweep can walk part of: the five shapes a constraint
/// or entity collection takes, each able to yield a slot range.
pub trait Leaves {
    type Item;
    /// Number of index slots (an arena counts its free slots too).
    fn count(&self) -> usize;
    /// Every live item with its slot index, in container order. The cut
    /// is built over this: a weight per unit, and the slot it falls on.
    fn each(&self, f: impl FnMut(u32, &Self::Item));
    /// The live items of the slot range `[lo, hi)`, in container order.
    ///
    /// Slots, not positions, so a range means the same thing whether or
    /// not the container has holes, and `hi` clamps to [`count`](Self::count).
    /// This is how a sweep walks part of a container: `0 .. count()` is the
    /// whole of it and has to cost what the plain iterator costs.
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &Self::Item>;
}

impl<T> Leaves for std::vec::Vec<T> {
    type Item = T;
    fn count(&self) -> usize { self.len() }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (i, x) in self.iter().enumerate() { f(i as u32, x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        let hi = (hi as usize).min(self.len());
        let lo = (lo as usize).min(hi);
        self[lo..hi].iter()
    }
}

impl<T> Leaves for crate::refs::Vec<T> {
    type Item = T;
    fn count(&self) -> usize { self.len() }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (i, x) in self.iter().enumerate() { f(i as u32, x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        self.range(lo, hi)
    }
}

impl<T> Leaves for crate::refs::Deque<T> {
    type Item = T;
    fn count(&self) -> usize { self.len() }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (i, x) in self.iter().enumerate() { f(i as u32, x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        self.range(lo, hi)
    }
}

impl<T> Leaves for crate::refs::Arena<T> {
    type Item = T;
    fn count(&self) -> usize { self.slot_count() }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (r, x) in self.iter_refs() { f(r.index(), x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        self.iter_range(lo, hi)
    }
}

impl<T> Leaves for Option<T> {
    type Item = T;
    fn count(&self) -> usize { 1 }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        if let Some(x) = self { f(0, x); }
    }
    #[inline(always)]
    fn range_iter(&self, lo: u32, hi: u32) -> impl Iterator<Item = &T> {
        // One slot, taken or not.
        let live = lo == 0 && hi > 0;
        self.as_ref().filter(|_| live).into_iter()
    }
}

/// Timing of one threaded phase over a solve, summed over its calls:
/// the region from dispatch to join (or the sequential run of the
/// stores), and per call the shortest, longest and total time one store
/// took. The gap between the longest task and the region is the
/// dispatch, the wake-up and the join; the gap between the shortest and
/// the longest is the imbalance.
#[derive(Clone, Debug, Default)]
pub struct PhaseTiming {
    /// The calls that were dispatched.
    pub par: FormTiming,
    /// The calls that ran the stores on the calling thread.
    pub seq: FormTiming,
}

/// One form's share of a phase, summed over its calls.
#[derive(Clone, Debug, Default)]
pub struct FormTiming {
    pub calls: usize,
    pub region: Duration,
    pub task_sum: Duration,
    pub task_max: Duration,
    pub task_min: Duration,
}

impl PhaseTiming {
    pub fn record(&mut self, region: Duration, dispatched: bool, tasks: impl Iterator<Item = Duration>) {
        let f = if dispatched { &mut self.par } else { &mut self.seq };
        f.calls += 1;
        f.region += region;
        let (mut sum, mut max, mut min) = (Duration::ZERO, Duration::ZERO, Duration::MAX);
        for t in tasks {
            sum += t;
            max = max.max(t);
            min = min.min(t);
        }
        f.task_sum += sum;
        f.task_max += max;
        if min != Duration::MAX { f.task_min += min; }
    }

    pub fn calls(&self) -> usize { self.par.calls + self.seq.calls }

    fn report(&self, threads: usize) -> String {
        format!("par {}; seq {}", self.par.report(threads), self.seq.report(threads))
    }
}

impl FormTiming {
    fn report(&self, threads: usize) -> String {
        if self.calls == 0 { return "x0".to_string(); }
        let per = |d: Duration| d.as_secs_f64() * 1e3 / self.calls as f64;
        format!("x{}: region {:.3} ms, tasks max {:.3} min {:.3} mean {:.3}",
            self.calls, per(self.region), per(self.task_max),
            per(self.task_min), per(self.task_sum) / threads.max(1) as f64)
    }
}

/// A clock the timing turns on: `start` reads the time only then, so a
/// solve without timing makes no clock calls at all.
#[derive(Clone, Copy, Debug)]
pub struct Clock {
    on: bool,
}

impl Clock {
    /// A clock that reads the time only when `on`.
    #[inline]
    pub fn new(on: bool) -> Self { Clock { on } }

    #[inline]
    pub fn start(self) -> Option<Instant> {
        if self.on { Some(Instant::now()) } else { None }
    }

    #[inline]
    pub fn stop(self, started: Option<Instant>) -> Duration {
        started.map_or(Duration::ZERO, |t| t.elapsed())
    }
}

/// Where a solve's sweeps spent their time, recorded only when the
/// context's timing is on ([`Context::set_timing`]); the call counts are
/// kept either way. Per assembly: the parameter update, the sweep
/// region, the gradient gather and the Hessian scatter; per cost
/// evaluation: the update and the sweep region.
#[derive(Clone, Debug, Default)]
pub struct ParTiming {
    /// Whether the clocks run.
    pub on: bool,
    pub assembly_update: Duration,
    pub assembly: PhaseTiming,
    pub gather_grad: Duration,
    pub scatter: Duration,
    pub cost_update: Duration,
    pub cost: PhaseTiming,
}

impl ParTiming {
    /// One line, per-call means in milliseconds.
    pub fn report(&self, threads: usize) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let per = |d: Duration, n: usize| ms(d) / n.max(1) as f64;
        format!("assembly [{}] update {:.3}, grad gather {:.3}, scatter {:.3}; cost [{}] update {:.3}",
            self.assembly.report(threads),
            per(self.assembly_update, self.assembly.calls()),
            per(self.gather_grad, self.assembly.calls()),
            per(self.scatter, self.assembly.calls()),
            self.cost.report(threads),
            per(self.cost_update, self.cost.calls()))
    }
}

/// What a solve reuses between its calls and between solves: the thread
/// count, the root's block stores and the cut that divides the walks
/// among them. The solve entries make one per solve; a caller holds one
/// across solves ([`LmSession`](crate::simple_lm::LmSession),
/// [`lm_solve_with_context`](crate::simple_lm::lm_solve_with_context)) so
/// the stores' allocations carry over. The stores are held type-erased
/// and taken back by their type by the root's generated code; a context
/// used with another root starts that root's stores fresh.
pub struct Context {
    threads: usize,
    timing: bool,
    /// The model's extended hook pushes COO entries, so the Hessian
    /// pattern is only knowable after a compute. Asked once per solve by
    /// running the hook (see
    /// `LmProblemInternals::extended_hook_writes_coo`), because nothing
    /// static can answer it.
    runtime_coo: bool,
    /// Every store of the solve, held as one `Vec<S>` type-erased. The
    /// list keeps whatever it has allocated between solves; `blocks_len`
    /// says how much of it this solve uses.
    blocks: Option<Box<dyn AnyStore>>,
    blocks_len: usize,
    cut: Cut,
    sweeps: ParTiming,
}

/// What one phase's sweeps did over a solve: the form they ran in, and
/// how many calls there were.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseChoice {
    /// True when the phase was dispatched over the stores.
    pub threaded: bool,
    /// Calls of the phase in this solve.
    pub calls: usize,
}

/// The thread counts a solve was given, before anything is measured.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ThreadCounts {
    /// Threads for the cost and assembly sweeps.
    pub sweeps_asked: usize,
    /// Threads for the linear solve.
    pub linear: usize,
}

/// What a solve's sweeps did, for the result's report. Only a solve that
/// ran a generated sweep has one.
#[derive(Clone, Debug, Default)]
pub struct SweepReport {
    /// Stores the sweeps were given: 1 means the whole model in one, run
    /// on the calling thread.
    pub threads: usize,
    pub assembly: PhaseChoice,
    pub cost: PhaseChoice,
    /// Where the sweeps' time went, when the solve gathered timing
    /// ([`LmConfig::gather_timing`](crate::simple_lm::LmConfig::gather_timing)
    /// or [`Context::set_timing`]); all zero otherwise, and `on` says
    /// which.
    pub timing: ParTiming,
}

impl Default for Context {
    fn default() -> Self { Self::new() }
}

impl Clone for Context {
    fn clone(&self) -> Self {
        Context {
            threads: self.threads,
            timing: self.timing,
            runtime_coo: self.runtime_coo,
            blocks: self.blocks.as_ref().map(|b| b.store_clone()),
            blocks_len: self.blocks_len,
            cut: self.cut.clone(),
            sweeps: self.sweeps.clone(),
        }
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").field("threads", &self.threads).finish()
    }
}

impl Context {
    /// One thread, no stores, timing off.
    pub fn new() -> Self {
        Context {
            threads: 1, timing: false, runtime_coo: false,
            blocks: None, blocks_len: 0, cut: Cut::new(),
            sweeps: ParTiming::default(),
        }
    }

    /// Record that this solve's extended hook pushes COO entries, so its
    /// Hessian pattern is only knowable after a compute. Set once per
    /// solve, before the first assembly.
    pub fn set_runtime_coo(&mut self, on: bool) { self.runtime_coo = on; }

    /// Whether the hook pushes COO entries (see
    /// [`set_runtime_coo`](Self::set_runtime_coo)).
    pub fn runtime_coo(&self) -> bool { self.runtime_coo }

    /// Time the phases of the solves through this context. Off by
    /// default: then the sweeps never read a clock.
    pub fn set_timing(&mut self, on: bool) {
        self.timing = on;
        // The sweeps' clocks read this: without it they count calls and
        // report every duration as zero.
        self.sweeps.on = on;
    }

    /// Whether the solves through this context are timed.
    pub fn timing(&self) -> bool { self.timing }

    /// The thread count of the next solve, resolved like
    /// [`LmConfig::num_threads`](crate::simple_lm::LmConfig::num_threads)
    /// (0 is the pool's size). The solve entries set it from the config,
    /// which starts a solve: the timing counts what this solve does, not
    /// what the last one through the context did.
    pub fn set_threads(&mut self, n: usize) {
        self.threads = pool_size(n).max(1);
        self.sweeps = ParTiming { on: self.timing, ..ParTiming::default() };
    }

    /// The resolved thread count; 1 when the sweeps run on the calling
    /// thread.
    pub fn threads(&self) -> usize { self.threads.max(1) }

    /// What the sweeps of the last solve through this context did.
    /// `None` until a generated sweep has run through it: a model whose
    /// evaluation is hand-written never fills one in.
    pub fn sweeps(&self) -> Option<SweepReport> {
        if self.blocks.is_none() { return None; }
        let phase = |p: &PhaseTiming| PhaseChoice {
            threaded: p.par.calls > 0,
            calls: p.calls(),
        };
        Some(SweepReport {
            threads: self.blocks_len.max(1),
            assembly: phase(&self.sweeps.assembly),
            cost: phase(&self.sweeps.cost),
            timing: self.sweeps.clone(),
        })
    }

    /// This root's Hessian block stores, if the context holds them: one
    /// per thread, plus the tail's. A solve that threads nothing has
    /// exactly one, and every walk over the list reads the same whatever
    /// its length.
    pub fn blocks_list<S: BlockStore>(&self) -> Option<&[S]> {
        self.blocks.as_ref()
            .and_then(|b| b.store_any().downcast_ref::<std::vec::Vec<S>>())
            .map(|v| &v[..self.blocks_len.min(v.len())])
    }

    /// The stores of this solve, `n` of them, empty when the context held
    /// none or another root's. The generated build fills them.
    ///
    /// The list is never shortened, only re-lengthened: a solve with fewer
    /// stores than the last one keeps the spare allocations and uses the
    /// front of the list.
    pub fn blocks_list_mut<S: BlockStore>(&mut self, n: usize) -> &mut [S] {
        let n = n.max(1);
        let holds = self.blocks.as_ref()
            .is_some_and(|b| b.store_any().is::<std::vec::Vec<S>>());
        if !holds {
            self.blocks = Some(Box::new(std::vec::Vec::<S>::new()));
        }
        let v = self.blocks.as_mut().unwrap().store_any_mut()
            .downcast_mut::<std::vec::Vec<S>>().unwrap();
        if v.len() < n { v.resize_with(n, S::default); }
        self.blocks_len = n;
        &mut v[..n]
    }

    /// This root's first store, if the context holds it.
    pub fn blocks<S: BlockStore>(&self) -> Option<&S> {
        self.blocks_list::<S>().and_then(|v| v.first())
    }

    /// This root's only store, for a solve that keeps one.
    pub fn blocks_mut<S: BlockStore>(&mut self) -> &mut S {
        &mut self.blocks_list_mut::<S>(1)[0]
    }

    /// The stores this solve is already using and the cut, without
    /// changing how many. An assembly runs the list the solve was begun
    /// with; asking for a count here would narrow it under the cut.
    pub fn blocks_and_cut_active_mut<S: BlockStore>(&mut self) -> (&mut [S], &Cut) {
        let n = self.blocks_len.max(1);
        self.blocks_and_cut_mut::<S>(n)
    }

    /// What a sweep needs from the context at once: its stores, the cut
    /// that says which units each covers, and somewhere to record where
    /// the time went. Three disjoint fields, so one call rather than
    /// three borrows. Both the assembly and the cost evaluation take
    /// their region's parts this way.
    pub fn sweep_parts_mut<S: BlockStore>(&mut self)
        -> (&mut [S], &Cut, &mut ParTiming)
    {
        let n = self.blocks_len.max(1);
        let holds = self.blocks.as_ref()
            .is_some_and(|b| b.store_any().is::<std::vec::Vec<S>>());
        if !holds {
            self.blocks = Some(Box::new(std::vec::Vec::<S>::new()));
        }
        let v = self.blocks.as_mut().unwrap().store_any_mut()
            .downcast_mut::<std::vec::Vec<S>>().unwrap();
        if v.len() < n { v.resize_with(n, S::default); }
        self.blocks_len = n;
        (&mut v[..n], &self.cut, &mut self.sweeps)
    }

    /// Where this solve's sweeps spent their time.
    pub fn sweep_timing(&self) -> &ParTiming { &self.sweeps }

    /// The timing, to record into.
    pub fn sweep_timing_mut(&mut self) -> &mut ParTiming { &mut self.sweeps }

    /// The stores this solve is already using, without changing how many.
    /// A walk that only reads must not narrow the list it found.
    pub fn blocks_active_mut<S: BlockStore>(&mut self) -> &mut [S] {
        let n = self.blocks_len.max(1);
        self.blocks_list_mut::<S>(n)
    }

    /// How this solve's walks are divided among its stores.
    pub fn cut(&self) -> &Cut { &self.cut }

    /// The cut, to build.
    pub fn cut_mut(&mut self) -> &mut Cut { &mut self.cut }

    /// The stores and the cut together. The sweeps need both, and taking
    /// them one at a time would borrow the context twice.
    pub fn blocks_and_cut_mut<S: BlockStore>(&mut self, n: usize) -> (&mut [S], &Cut) {
        let n = n.max(1);
        let holds = self.blocks.as_ref()
            .is_some_and(|b| b.store_any().is::<std::vec::Vec<S>>());
        if !holds {
            self.blocks = Some(Box::new(std::vec::Vec::<S>::new()));
        }
        let v = self.blocks.as_mut().unwrap().store_any_mut()
            .downcast_mut::<std::vec::Vec<S>>().unwrap();
        if v.len() < n { v.resize_with(n, S::default); }
        self.blocks_len = n;
        (&mut v[..n], &self.cut)
    }
}

/// A root's generated block store, held type-erased so the context does
/// not name it. `Send + Sync` so a root may keep a context in a field
/// and stay `Sync` for the dispatch.
trait AnyStore: Any + Send + Sync {
    fn store_any(&self) -> &dyn Any;
    fn store_any_mut(&mut self) -> &mut dyn Any;
    fn store_clone(&self) -> Box<dyn AnyStore>;
}

/// What a generated block store is: the macro implements it on the type
/// it emits per root.
pub trait BlockStore: Clone + Default + Send + Sync + 'static {}

impl<S: BlockStore> AnyStore for S {
    fn store_any(&self) -> &dyn Any { self }
    fn store_any_mut(&mut self) -> &mut dyn Any { self }
    fn store_clone(&self) -> Box<dyn AnyStore> { Box::new(self.clone()) }
}

// The context's payload is the whole list, not one store. Spelled out
// rather than blanket: a blanket over `Any + Send + Sync + Clone` also
// covers `&Box<dyn AnyStore>`, and method calls stop derefing to the
// trait object.
impl<S: BlockStore> AnyStore for std::vec::Vec<S> {
    fn store_any(&self) -> &dyn Any { self }
    fn store_any_mut(&mut self) -> &mut dyn Any { self }
    fn store_clone(&self) -> Box<dyn AnyStore> { Box::new(self.clone()) }
}

/// The thread count a solve's `num_threads` setting means: 0 is the
/// rayon pool's size. Without the `rayon` feature every count is 1.
pub fn pool_size(num_threads: usize) -> usize {
    #[cfg(feature = "rayon")]
    {
        match num_threads {
            0 => rayon::current_num_threads(),
            n => n,
        }
    }
    #[cfg(not(feature = "rayon"))]
    {
        let _ = num_threads;
        1
    }
}

/// Run `f` over every store, told which one it is so it can take its own
/// row of the cut: one task each on the rayon pool when `par`, else in
/// order on the calling thread. The two forms do the same arithmetic in
/// the same order per store -- the region is what changes, not the work.
pub fn run_indexed<M: Send>(par: bool, stores: &mut [M], f: impl Fn(usize, &mut M) + Sync) {
    #[cfg(feature = "rayon")]
    if par {
        rayon::in_place_scope(|s| {
            for (i, m) in stores.iter_mut().enumerate() {
                let f = &f;
                s.spawn(move |_| f(i, m));
            }
        });
        return;
    }
    let _ = par;
    for (i, m) in stores.iter_mut().enumerate() {
        f(i, m);
    }
}

// ---------------------------------------------------------------------------
// The partition
// ---------------------------------------------------------------------------

/// The start of store `t`'s range when `n` slots are divided into `p`
/// contiguous ranges of about equal length; `t == p` gives `n`.
#[inline]
pub fn cut(n: usize, p: usize, t: usize) -> usize {
    (n * t).div_ceil(p)
}

/// How a solve's walks are divided among its stores.
///
/// One row per store, one entry per walk: the slot range that store
/// covers of that walk. An empty cut is the whole model in one store,
/// which is what a sequential solve uses and what every solve uses until
/// a cut is built for it -- every row then reads as the empty table
/// [`walk_range`] treats as "all of it".
#[derive(Clone, Debug, Default)]
pub struct Cut {
    /// `stores * walks` entries, one row per store.
    ranges: std::vec::Vec<(u32, u32)>,
    walks: usize,
    stores: usize,
}

impl Cut {
    /// A cut dividing nothing: the whole model, one store.
    pub fn new() -> Self { Cut { ranges: std::vec::Vec::new(), walks: 0, stores: 0 } }

    /// True while no cut has been built.
    pub fn is_empty(&self) -> bool { self.stores == 0 }

    /// The stores this cut divides the walks among.
    pub fn stores(&self) -> usize { self.stores }

    /// The walks it has a range for.
    pub fn walks(&self) -> usize { self.walks }

    /// Forget the cut: back to the whole model in one store. This is the
    /// single answer to "is a store a slice or all of it" -- an empty cut
    /// means whole, everywhere that asks.
    pub fn clear(&mut self) {
        self.ranges.clear();
        self.stores = 0;
        self.walks = 0;
    }

    /// Start a cut of `walks` walks over `stores` stores, every range
    /// covering nothing until it is set. Keeps its allocation.
    pub fn reset(&mut self, stores: usize, walks: usize) {
        self.ranges.clear();
        self.ranges.resize(stores * walks, (0, 0));
        self.stores = stores;
        self.walks = walks;
    }

    /// Give store `s` the slot range `[lo, hi)` of walk `w`.
    pub fn set(&mut self, s: usize, w: usize, lo: u32, hi: u32) {
        self.ranges[s * self.walks + w] = (lo, hi);
    }

    /// The ranges store `s` covers, one per walk. Empty while no cut has
    /// been built, which every walk reads as the whole of its container.
    #[inline]
    pub fn store(&self, s: usize) -> &[(u32, u32)] {
        if self.is_empty() { return &[]; }
        &self.ranges[s * self.walks..(s + 1) * self.walks]
    }
}

/// The slot range walk `k` of a sweep should cover.
///
/// A sweep is handed one range per top-level walk, in the order the walks
/// are emitted. An empty table means the whole model, which is what the
/// sequential path passes and what every walk gets until a cut exists to
/// fill the table in: [`Leaves::range_iter`] clamps `u32::MAX` to the
/// container's length, so the full range is the plain walk.
#[inline(always)]
pub fn walk_range(ranges: &[(u32, u32)], k: usize) -> (u32, u32) {
    if ranges.is_empty() { (0, u32::MAX) } else { ranges[k] }
}

#[cfg(test)]
mod tests {
    use super::*;

    // The property every later stage rests on: a set of contiguous slot
    // ranges covering the container yields the full walk, in order, once
    // each -- holes and container kind notwithstanding.
    fn ranges_partition_the_walk<C: Leaves>(c: &C, whole: &[i32])
    where C::Item: PartialEq<i32> + std::fmt::Debug + Copy + Into<i32> {
        let n = c.count() as u32;
        let got: std::vec::Vec<i32> = c.range_iter(0, n).map(|x| (*x).into()).collect();
        assert_eq!(got, whole, "0..count is the whole walk");
        for cuts in [vec![], vec![n / 2], vec![n / 3, 2 * n / 3], (0..=n).collect()] {
            let mut bounds = vec![0u32];
            bounds.extend(cuts);
            bounds.push(n);
            let mut seen: std::vec::Vec<i32> = std::vec::Vec::new();
            for w in bounds.windows(2) {
                seen.extend(c.range_iter(w[0], w[1]).map(|x| (*x).into()));
            }
            assert_eq!(seen, whole, "ranges {:?} must rebuild the walk", bounds);
        }
    }

    #[test]
    fn an_unbuilt_cut_is_the_whole_model() {
        let c = Cut::new();
        assert!(c.is_empty());
        // Every store reads the empty table, which every walk takes as all
        // of its container.
        assert!(c.store(0).is_empty());
        assert_eq!(walk_range(c.store(0), 0), (0, u32::MAX));
    }

    #[test]
    fn a_cut_hands_each_store_its_own_row() {
        let mut c = Cut::new();
        c.reset(3, 2);
        assert!(!c.is_empty());
        assert_eq!((c.stores(), c.walks()), (3, 2));
        // Two walks of 100 slots, cut three ways.
        for (s, (lo, hi)) in [(0usize, (0u32, 34u32)), (1, (34, 67)), (2, (67, 100))] {
            c.set(s, 0, lo, hi);
            c.set(s, 1, lo, hi);
        }
        assert_eq!(c.store(0), &[(0, 34), (0, 34)]);
        assert_eq!(c.store(2), &[(67, 100), (67, 100)]);
        assert_eq!(walk_range(c.store(1), 1), (34, 67));
        // The rows together cover each walk once, which is what keeps a
        // contribution from being counted twice or dropped.
        let mut covered: std::vec::Vec<(u32, u32)> =
            (0..3).map(|s| walk_range(c.store(s), 0)).collect();
        covered.sort();
        assert_eq!(covered, vec![(0, 34), (34, 67), (67, 100)]);
    }

    #[derive(Clone, Default, Debug, PartialEq)]
    struct TestStore { tag: u32 }
    impl BlockStore for TestStore {}

    #[test]
    fn the_store_list_keeps_its_stores_and_uses_the_front() {
        let mut ctx = Context::new();
        for (i, s) in ctx.blocks_list_mut::<TestStore>(4).iter_mut().enumerate() {
            s.tag = i as u32 + 1;
        }
        assert_eq!(ctx.blocks_list::<TestStore>().unwrap().len(), 4);

        // A solve wanting fewer stores uses the front of the list.
        assert_eq!(ctx.blocks_list_mut::<TestStore>(1).len(), 1);
        assert_eq!(ctx.blocks_list::<TestStore>().unwrap().len(), 1);
        assert_eq!(ctx.blocks::<TestStore>().unwrap().tag, 1);

        // The spares were kept, not dropped and rebuilt: growing back finds
        // the same stores, which is what keeps their allocations.
        let back: std::vec::Vec<u32> =
            ctx.blocks_list_mut::<TestStore>(4).iter().map(|s| s.tag).collect();
        assert_eq!(back, vec![1, 2, 3, 4]);
    }

    #[test]
    fn blocks_mut_is_the_first_store_and_narrows_to_it() {
        let mut ctx = Context::new();
        ctx.blocks_list_mut::<TestStore>(3)[0].tag = 7;
        assert_eq!(ctx.blocks_mut::<TestStore>().tag, 7);
        // It is the single-store accessor, so it says the solve uses one.
        assert_eq!(ctx.blocks_list::<TestStore>().unwrap().len(), 1);
    }

    #[test]
    fn another_roots_store_starts_over() {
        #[derive(Clone, Default)]
        struct OtherStore;
        impl BlockStore for OtherStore {}

        let mut ctx = Context::new();
        ctx.blocks_list_mut::<TestStore>(2)[1].tag = 5;
        assert!(ctx.blocks_list::<OtherStore>().is_none(), "a different root holds none");
        assert_eq!(ctx.blocks_list_mut::<OtherStore>(2).len(), 2);
        // Taking it for one root drops the other's, as a context serves one.
        assert!(ctx.blocks_list::<TestStore>().is_none());
    }

    #[test]
    fn range_iter_partitions_every_container() {
        let v: std::vec::Vec<i32> = vec![1, 2, 3, 4, 5];
        ranges_partition_the_walk(&v, &[1, 2, 3, 4, 5]);

        let rv = crate::refs::Vec::from_vec(vec![1, 2, 3, 4, 5]);
        ranges_partition_the_walk(&rv, &[1, 2, 3, 4, 5]);

        let mut dq: crate::refs::Deque<i32> = crate::refs::Deque::new();
        for x in [2, 3, 4, 5] { dq.push_back(x); }
        dq.push_front(1);
        ranges_partition_the_walk(&dq, &[1, 2, 3, 4, 5]);

        // An arena with holes: the slot count spans them, the walk does not.
        let mut ar: crate::refs::Arena<i32> = crate::refs::Arena::with_block_size(2);
        let r: std::vec::Vec<_> = (1..=7).map(|x| ar.push(x)).collect();
        ar.remove(r[1]);
        ar.remove(r[5]);
        ranges_partition_the_walk(&ar, &[1, 3, 4, 5, 7]);

        let some: Option<i32> = Some(9);
        ranges_partition_the_walk(&some, &[9]);
        let none: Option<i32> = None;
        ranges_partition_the_walk(&none, &[]);
    }

    #[test]
    fn cut_covers_the_leaves_in_order() {
        for &(n, p) in &[(0usize, 1usize), (1, 4), (3, 4), (4, 4), (10, 3), (204_472, 4)] {
            assert_eq!(cut(n, p, 0), 0);
            assert_eq!(cut(n, p, p), n);
            let lens: std::vec::Vec<usize> = (0..p).map(|t| cut(n, p, t + 1) - cut(n, p, t)).collect();
            let (lo, hi) = (lens.iter().min().unwrap(), lens.iter().max().unwrap());
            assert!(hi - lo <= 1, "n {} p {}: {:?}", n, p, lens);
        }
    }

    #[test]
    fn a_clock_off_reads_nothing() {
        let off = Clock { on: false };
        assert!(off.start().is_none());
        assert_eq!(off.stop(None), Duration::ZERO);
        let on = Clock { on: true };
        let t = on.start();
        assert!(t.is_some());
        let _ = on.stop(t);
        let mut ctx = Context::new();
        assert!(!ctx.timing());
        ctx.set_timing(true);
        assert!(ctx.clone().timing());
    }
}
