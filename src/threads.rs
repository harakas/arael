//! Per-thread mirrors for the threaded cost and assembly sweeps, which
//! every root gets under the `rayon` feature.
//!
//! The macro generates one mirror type per root: a struct holding, for
//! every constraint container, the list of leaves (constraint instances)
//! assigned to one thread with their cross blocks, and for every entity
//! container a slab of [`Partial`](crate::model::Partial) blocks, one per
//! entity the thread touches. The model is read-only during a sweep; each
//! thread writes only its own mirror; a serial gather scatters the mirrors
//! into the gradient and the Hessian. A solve's [`Context`] owns the
//! mirrors, type-erased, so the root carries no field for them; the
//! generated code takes them back by their type. Nothing here is used
//! without the feature, or by a root with a form the mirrors do not
//! cover: such a root keeps the sequential sweeps at every thread count.

use std::any::Any;
use std::time::{Duration, Instant};

/// A container the mirrors address by index: constraint instances and
/// entities are read from the model through `at`, never through a
/// reference held across a sweep.
pub trait Leaves {
    type Item;
    /// Number of index slots (an arena counts its free slots too).
    fn count(&self) -> usize;
    /// The item at slot `i`. Panics on a free arena slot.
    fn at(&self, i: u32) -> &Self::Item;
    /// Every live item with its slot index, in container order.
    fn each(&self, f: impl FnMut(u32, &Self::Item));
}

impl<T> Leaves for std::vec::Vec<T> {
    type Item = T;
    fn count(&self) -> usize { self.len() }
    #[inline(always)]
    fn at(&self, i: u32) -> &T { &self[i as usize] }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (i, x) in self.iter().enumerate() { f(i as u32, x); }
    }
}

impl<T> Leaves for crate::refs::Vec<T> {
    type Item = T;
    fn count(&self) -> usize { self.len() }
    #[inline(always)]
    fn at(&self, i: u32) -> &T { &self[i as usize] }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (i, x) in self.iter().enumerate() { f(i as u32, x); }
    }
}

impl<T> Leaves for crate::refs::Deque<T> {
    type Item = T;
    fn count(&self) -> usize { self.len() }
    #[inline(always)]
    fn at(&self, i: u32) -> &T { &self[i as usize] }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (i, x) in self.iter().enumerate() { f(i as u32, x); }
    }
}

impl<T> Leaves for crate::refs::Arena<T> {
    type Item = T;
    fn count(&self) -> usize { self.slot_count() }
    #[inline(always)]
    fn at(&self, i: u32) -> &T {
        self.at_slot(i as usize).expect("mirror leaf names a freed arena slot")
    }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        for (r, x) in self.iter_refs() { f(r.index(), x); }
    }
}

impl<T> Leaves for Option<T> {
    type Item = T;
    fn count(&self) -> usize { 1 }
    #[inline(always)]
    fn at(&self, _i: u32) -> &T { self.as_ref().expect("mirror leaf names an empty Option") }
    fn each(&self, mut f: impl FnMut(u32, &T)) {
        if let Some(x) = self { f(0, x); }
    }
}

/// Timing of one threaded phase over a solve, summed over its calls:
/// the region from dispatch to join (or the sequential run of the
/// mirrors), and per call the shortest, longest and total task time of
/// the mirrors. The gap between the longest task and the region is the
/// dispatch, the wake-up and the join; the gap between the shortest and
/// the longest is the imbalance.
#[derive(Clone, Debug, Default)]
pub struct PhaseTiming {
    /// The calls that were dispatched.
    pub par: FormTiming,
    /// The calls that ran the mirrors on the calling thread.
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
    #[inline]
    pub fn start(self) -> Option<Instant> {
        if self.on { Some(Instant::now()) } else { None }
    }

    #[inline]
    pub fn stop(self, started: Option<Instant>) -> Duration {
        started.map_or(Duration::ZERO, |t| t.elapsed())
    }
}

/// Where a solve's time goes on the mirror path, reset at every
/// `set_threads` and recorded only when the context's timing is on
/// ([`Context::set_timing`]); the counts are kept either way. Per
/// assembly: the parameter update, the sweep region, the gradient
/// gather and the Hessian scatter; per cost evaluation: the update and
/// the sweep region.
#[derive(Clone, Debug, Default)]
pub struct ParTiming {
    /// Whether the clocks run.
    pub on: bool,
    pub builds: usize,
    pub build: Duration,
    /// The build's phases: the fill of the mirrors from the leaves, and
    /// the block arrays' `finish`.
    pub build_fill: Duration,
    pub build_finish: Duration,
    /// Over every mirror, after the build: leaves, cross blocks, and
    /// partials (an entity touched from several mirrors counts once
    /// per mirror).
    pub leaves: usize,
    pub blocks: usize,
    pub partials: usize,
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
        format!("build {:.3} ms x{} (fill {:.3}, finish {:.3}; leaves {}, blocks {}, partials {}); assembly [{}] update {:.3}, grad gather {:.3}, scatter {:.3}; cost [{}] update {:.3}",
            per(self.build, self.builds), self.builds,
            per(self.build_fill, self.builds), per(self.build_finish, self.builds),
            self.leaves, self.blocks, self.partials,
            self.assembly.report(threads),
            per(self.assembly_update, self.assembly.calls()),
            per(self.gather_grad, self.assembly.calls()),
            per(self.scatter, self.assembly.calls()),
            self.cost.report(threads),
            per(self.cost_update, self.cost.calls()))
    }
}

/// The root's mirrors: one per thread, plus the state that decides
/// whether a phase runs over them dispatched or on the calling thread.
pub struct Mirrors<M> {
    mirrors: std::vec::Vec<M>,
    threads: usize,
    /// The assembly's form for the current solve.
    pub assembly: Trial,
    /// The cost evaluation's form for the current solve.
    pub cost: Trial,
    /// Partition scratch, kept between solves for its allocations.
    pub build: Builder,
    /// Where the current solve's time went.
    pub timing: ParTiming,
}

impl<M> Default for Mirrors<M> {
    fn default() -> Self { Self::new() }
}

impl<M: Clone> Clone for Mirrors<M> {
    fn clone(&self) -> Self {
        Mirrors {
            mirrors: self.mirrors.clone(),
            threads: self.threads,
            assembly: self.assembly.clone(),
            cost: self.cost.clone(),
            build: self.build.clone(),
            timing: self.timing.clone(),
        }
    }
}

impl<M> std::fmt::Debug for Mirrors<M> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mirrors").field("threads", &self.threads).finish()
    }
}

impl<M> Mirrors<M> {
    /// No mirrors: the sequential sweeps run.
    pub fn new() -> Self {
        Mirrors {
            mirrors: std::vec::Vec::new(),
            threads: 0,
            assembly: Trial::default(),
            cost: Trial::default(),
            build: Builder::default(),
            timing: ParTiming::default(),
        }
    }

    /// Record one assembly sweep region and every mirror's task time.
    pub fn record_assembly(&mut self, region: Duration, dispatched: bool, task: impl Fn(&M) -> Duration) {
        self.timing.assembly.record(region, dispatched, self.mirrors.iter().map(task));
    }

    /// Record one cost sweep region and every mirror's task time.
    pub fn record_cost(&mut self, region: Duration, dispatched: bool, task: impl Fn(&M) -> Duration) {
        self.timing.cost.record(region, dispatched, self.mirrors.iter().map(task));
    }

    /// Turn the timing on or off for the solves to come.
    pub fn enable_timing(&mut self, on: bool) {
        self.timing.on = on;
    }

    /// The clock the sweeps time with: reads nothing while the timing
    /// is off.
    pub fn clock(&self) -> Clock {
        Clock { on: self.timing.on }
    }

    /// True when the mirrors are in use: every phase runs over them.
    pub fn is_on(&self) -> bool { self.threads > 1 }

    /// The thread count the mirrors were sized for; 0 or 1 when off.
    pub fn threads(&self) -> usize { self.threads }

    /// Size the mirrors for `n` threads (1 or 0 turns them off) and reset
    /// the trials for a new solve. Returns true when the count changed,
    /// so the caller can rebuild the mirrors or rewire the model's
    /// blocks.
    pub fn set_threads(&mut self, n: usize) -> bool
    where
        M: Default,
    {
        let n = if n > 1 { n } else { 0 };
        let changed = n != self.threads;
        self.threads = n;
        self.mirrors.resize_with(n, M::default);
        self.assembly.reset();
        self.cost.reset();
        self.timing = ParTiming::default();
        changed
    }

    pub fn as_slice(&self) -> &[M] { &self.mirrors }
    pub fn as_mut_slice(&mut self) -> &mut [M] { &mut self.mirrors }

    /// The mirrors and the partition scratch, for the build.
    pub fn split(&mut self) -> (&mut [M], &mut Builder) {
        (&mut self.mirrors, &mut self.build)
    }
}

/// What a solve reuses between its calls and between solves: the thread
/// count, and the root's mirrors. The solve entries make one
/// per solve; a caller holds one across solves
/// ([`LmSession`](crate::simple_lm::LmSession),
/// [`lm_solve_with_context`](crate::simple_lm::lm_solve_with_context)) so
/// the mirrors' allocations carry over. The mirrors are stored
/// type-erased and taken back by their type by the root's generated
/// code; a context used with another root starts that root's mirrors
/// fresh.
pub struct Context {
    threads: usize,
    timing: bool,
    mirrors: Option<Box<dyn AnyMirrors>>,
}

/// What one phase's sweeps did over a solve: the form they run in, and
/// the two times the trial measured before it chose (sequential first,
/// dispatched second), if it got to measure both.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PhaseChoice {
    /// True when the phase runs dispatched over the mirrors.
    pub threaded: bool,
    /// The trial's two totals, sequential and dispatched, per call.
    pub measured: Option<(Duration, Duration)>,
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

/// What a solve's sweeps did, for the result's report. Only a model with
/// a threaded sweep path has one.
#[derive(Clone, Debug, Default)]
pub struct SweepReport {
    /// Mirrors the sweeps were given: 0 or 1 means they ran on the
    /// calling thread.
    pub threads: usize,
    pub assembly: PhaseChoice,
    pub cost: PhaseChoice,
    /// Where the sweeps' time went, when the solve gathered timing
    /// ([`LmConfig::gather_timing`](crate::simple_lm::LmConfig::gather_timing)
    /// or [`Context::set_timing`]); all zero otherwise, and `on` says
    /// which.
    pub timing: ParTiming,
}

/// The type-erased mirror store: `Any` for the downcast, and cloneable
/// so the context is. `Send + Sync` so a root may keep a context in a
/// field and stay `Sync` for the dispatch.
trait AnyMirrors: Any + Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
    fn clone_box(&self) -> Box<dyn AnyMirrors>;
    fn sweeps(&self) -> SweepReport;
}

impl<M: Clone + Send + Sync + 'static> AnyMirrors for Mirrors<M> {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn clone_box(&self) -> Box<dyn AnyMirrors> { Box::new(self.clone()) }
    fn sweeps(&self) -> SweepReport {
        let phase = |t: &Trial, p: &PhaseTiming| PhaseChoice {
            threaded: self.is_on() && t.par(),
            measured: t.measured(),
            calls: p.calls(),
        };
        SweepReport {
            threads: self.threads,
            assembly: phase(&self.assembly, &self.timing.assembly),
            cost: phase(&self.cost, &self.timing.cost),
            timing: self.timing.clone(),
        }
    }
}

impl Default for Context {
    fn default() -> Self { Self::new() }
}

impl Clone for Context {
    fn clone(&self) -> Self {
        Context {
            threads: self.threads,
            timing: self.timing,
            mirrors: self.mirrors.as_ref().map(|m| m.clone_box()),
        }
    }
}

impl std::fmt::Debug for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Context").field("threads", &self.threads).finish()
    }
}

impl Context {
    /// One thread, no mirrors, timing off.
    pub fn new() -> Self {
        Context { threads: 1, timing: false, mirrors: None }
    }

    /// Time the phases of the solves through this context (the mirrors'
    /// `timing`). Off by default: then the sweeps never read a clock.
    pub fn set_timing(&mut self, on: bool) {
        self.timing = on;
    }

    /// Whether the solves through this context are timed.
    pub fn timing(&self) -> bool { self.timing }

    /// The thread count of the next solve, resolved like
    /// [`LmConfig::num_threads`](crate::simple_lm::LmConfig::num_threads)
    /// (0 is the pool's size). The solve entries set it from the config.
    pub fn set_threads(&mut self, n: usize) {
        self.threads = pool_size(n).max(1);
    }

    /// The resolved thread count; 1 when the sweeps run on the calling
    /// thread.
    pub fn threads(&self) -> usize { self.threads.max(1) }

    /// What the sweeps of the last solve through this context did.
    /// `None` when the model has no threaded sweep path: the macro
    /// generates one only for a root whose every form the mirrors
    /// cover, and only with the `rayon` feature.
    pub fn sweeps(&self) -> Option<SweepReport> {
        self.mirrors.as_ref().map(|m| m.sweeps())
    }

    /// The mirrors of root type `M`, if this context holds them.
    pub fn mirrors<M: 'static>(&self) -> Option<&Mirrors<M>> {
        self.mirrors.as_ref().and_then(|m| m.as_any().downcast_ref::<Mirrors<M>>())
    }

    /// The mirrors of root type `M`, created off when the context holds
    /// none or another root's.
    pub fn mirrors_mut<M: Clone + Send + Sync + 'static>(&mut self) -> &mut Mirrors<M> {
        let holds = self.mirrors.as_ref().is_some_and(|m| m.as_any().is::<Mirrors<M>>());
        if !holds {
            self.mirrors = Some(Box::new(Mirrors::<M>::new()));
        }
        self.mirrors.as_mut().unwrap().as_any_mut().downcast_mut::<Mirrors<M>>().unwrap()
    }
}

/// Which form a phase takes during one solve. The first calls alternate
/// between the two forms, [`SAMPLES`](Self::SAMPLES) of each, timed: on
/// the calling thread first, then dispatched. The form with the smaller
/// total runs from then on. Two samples per form, interleaved, so that
/// a dispatch that happens to find the workers awake (right after
/// another dispatched phase) does not decide alone: the typical call of
/// a phase comes after the linear solve, with the workers asleep, and
/// that is the time to compare. Both forms walk the mirrors, so the
/// choice changes the time and nothing else.
#[derive(Clone, Debug, Default)]
pub struct Trial {
    calls: u32,
    seq: Duration,
    par: Duration,
    par_wins: bool,
}

impl Trial {
    /// Timed calls per form before the decision.
    pub const SAMPLES: u32 = 2;

    /// Start a solve: the next calls are the trial.
    pub fn reset(&mut self) {
        self.calls = 0;
        self.seq = Duration::ZERO;
        self.par = Duration::ZERO;
    }

    /// Whether the next call is dispatched.
    pub fn par(&self) -> bool {
        if self.calls < 2 * Self::SAMPLES {
            self.calls % 2 == 1
        } else {
            self.par_wins
        }
    }

    /// The clock for a trial call; `None` once the form is decided.
    pub fn start(&self) -> Option<Instant> {
        (self.calls < 2 * Self::SAMPLES).then(Instant::now)
    }

    /// Record a trial call's time.
    pub fn finish(&mut self, started: Option<Instant>) {
        let Some(t) = started else { return };
        let dt = t.elapsed();
        if self.calls % 2 == 0 {
            self.seq += dt;
        } else {
            self.par += dt;
        }
        self.calls += 1;
        if self.calls == 2 * Self::SAMPLES {
            self.par_wins = self.par < self.seq;
        }
    }

    /// The summed times of the two forms, once the trial is complete.
    pub fn measured(&self) -> Option<(Duration, Duration)> {
        (self.calls >= 2 * Self::SAMPLES).then_some((self.seq, self.par))
    }
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

/// Run `f` over every mirror: one task per mirror on the rayon pool when
/// `par`, else in order on the calling thread. The two forms do the same
/// arithmetic in the same order per mirror.
pub fn run<M: Send>(par: bool, mirrors: &mut [M], f: impl Fn(&mut M) + Sync) {
    #[cfg(feature = "rayon")]
    if par {
        rayon::in_place_scope(|s| {
            for m in mirrors.iter_mut() {
                s.spawn(|_| f(m));
            }
        });
        return;
    }
    let _ = par;
    for m in mirrors.iter_mut() {
        f(m);
    }
}

// ---------------------------------------------------------------------------
// The partition
// ---------------------------------------------------------------------------

/// Hands out partial slots while the mirrors are filled. The generated
/// build drives it: the entity counts, then per constraint container its
/// leaves in container order, cut by [`cut`] into one contiguous range
/// per thread, each leaf pushed to its thread's mirror with a slot per
/// role. An entity touched from several ranges has a partial in each of
/// their mirrors, so leaves grouped by the entities they share (a
/// point's observations next to each other) duplicate the least.
#[derive(Clone, Debug, Default)]
pub struct Builder {
    threads: usize,
    entities: std::vec::Vec<EntityScratch>,
}

#[derive(Clone, Debug, Default)]
struct EntityScratch {
    len: usize,
    /// `t + 1` when `slot` holds the entity's slot in mirror `t`.
    stamp: std::vec::Vec<u32>,
    slot: std::vec::Vec<u32>,
}

/// The start of thread `t`'s range when `n` leaves are cut into `p`
/// contiguous ranges of about equal length; `t == p` gives `n`.
#[inline]
pub fn cut(n: usize, p: usize, t: usize) -> usize {
    (n * t).div_ceil(p)
}

impl Builder {
    /// Start a build for `threads` mirrors over `entities` entity
    /// containers.
    pub fn reset(&mut self, threads: usize, entities: usize) {
        self.threads = threads;
        self.entities.resize_with(entities, EntityScratch::default);
        for e in &mut self.entities {
            e.len = 0;
            e.stamp.clear();
            e.slot.clear();
        }
    }

    /// Set the slot count of entity container `e`.
    pub fn set_entity_len(&mut self, e: usize, len: usize) {
        let s = &mut self.entities[e];
        s.len = len;
        s.stamp.clear();
        s.stamp.resize(len, 0);
        s.slot.clear();
        s.slot.resize(len, 0);
    }

    /// The slot count of entity container `e`.
    pub fn entity_len(&self, e: usize) -> usize { self.entities[e].len }

    /// The slot of `entity` in mirror `t`, if it has one.
    #[inline]
    pub fn slot(&self, e: usize, t: usize, entity: u32) -> Option<u32> {
        let s = &self.entities[e];
        (s.stamp[entity as usize] == t as u32 + 1).then(|| s.slot[entity as usize])
    }

    /// Record `entity`'s slot in mirror `t`.
    #[inline]
    pub fn claim(&mut self, e: usize, t: usize, entity: u32, slot: u32) {
        let s = &mut self.entities[e];
        s.stamp[entity as usize] = t as u32 + 1;
        s.slot[entity as usize] = slot;
    }

    /// The thread count of this build.
    pub fn threads(&self) -> usize { self.threads }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    #[test]
    fn slots_are_per_thread() {
        let mut b = Builder::default();
        b.reset(3, 1);
        b.set_entity_len(0, 5);
        assert_eq!(b.entity_len(0), 5);
        assert_eq!(b.slot(0, 1, 4), None);
        b.claim(0, 1, 4, 7);
        assert_eq!(b.slot(0, 1, 4), Some(7));
        assert_eq!(b.slot(0, 0, 4), None);
        assert_eq!(b.slot(0, 2, 4), None);
    }

    #[test]
    fn trial_alternates_then_keeps_the_faster() {
        let mut t = Trial::default();
        for k in 0..2 * Trial::SAMPLES {
            assert_eq!(t.par(), k % 2 == 1, "call {} alternates", k);
            let s = t.start();
            assert!(s.is_some());
            // The sequential form is made the slow one.
            if k % 2 == 0 { std::thread::sleep(Duration::from_millis(2)); }
            t.finish(s);
        }
        assert!(t.measured().is_some());
        assert!(t.par(), "the quicker dispatched form is kept");
        assert!(t.start().is_none());
        t.par();
        t.reset();
        assert!(!t.par());
        assert!(t.measured().is_none());
    }

    #[test]
    fn run_visits_every_mirror() {
        let mut m = vec![0u32; 5];
        run(false, &mut m, |x| *x += 1);
        run(true, &mut m, |x| *x += 1);
        assert!(m.iter().all(|&x| x == 2));
    }
}
