//! The solve context and what it reports: the thread count of a solve,
//! the block stores it keeps between solves, and where the threaded cost
//! and assembly sweeps spent their time.
//!
//! A [`Context`] is what a caller holds across solves
//! ([`LmSession`](crate::simple_lm::LmSession),
//! [`lm_solve_with_context`](crate::simple_lm::lm_solve_with_context)):
//! it owns one block store per thread, type-erased, so the root carries
//! no field for them, and the [`SweepReport`] a solve's result carries
//! comes from it. The stores themselves, the cut that divides the walks
//! among them and the accessors the generated code uses are the inner
//! API in [`crate::store`]. Without the `threads` feature, or with
//! `seq`, a solve has one store and it is the whole model.

use std::time::Duration;
use crate::store::{AnyStore, Cut};
pub use crate::store::StoreFootprint;

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
    /// Add one sweep: its region and its tasks' times. Called by the
    /// generated sweeps.
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

/// Where a solve's sweeps spent their time, recorded only when the
/// context's timing is on ([`Context::set_timing`]); the call counts are
/// kept either way. Per assembly: the parameter update, the store
/// zeroing, the sweep region, the gradient gather, the zeroing of the
/// Hessian's value buffer and the Hessian scatter; per cost evaluation:
/// the update and the sweep region; once per pattern, the binding of
/// the stores to it.
#[derive(Clone, Debug, Default)]
pub struct ParTiming {
    /// Whether the clocks run.
    pub on: bool,
    /// Binding the stores' blocks to an assembled pattern: the position
    /// stream's build, with its sort and its cut into chunks.
    pub bind: Duration,
    pub assembly_update: Duration,
    /// Zeroing a store's tiles and gradient stashes, which each sweep
    /// task does to its own store first: the longest store's, a part of
    /// the sweep region and of that task's time.
    pub assembly_zero: Duration,
    pub assembly: PhaseTiming,
    pub gather_grad: Duration,
    /// Zeroing the assembled Hessian's value buffer before the scatter,
    /// on the routes that scatter into one.
    pub assembly_zero_vals: Duration,
    pub scatter: Duration,
    pub cost_update: Duration,
    pub cost: PhaseTiming,
}

impl ParTiming {
    /// One line: the bind in full, the rest as per-call means, in
    /// milliseconds.
    pub fn report(&self, threads: usize) -> String {
        let ms = |d: Duration| d.as_secs_f64() * 1e3;
        let per = |d: Duration, n: usize| ms(d) / n.max(1) as f64;
        format!("bind {:.3}; assembly [{}] update {:.3}, zero {:.3}, grad gather {:.3}, zero vals {:.3}, scatter {:.3}; cost [{}] update {:.3}",
            ms(self.bind),
            self.assembly.report(threads),
            per(self.assembly_update, self.assembly.calls()),
            per(self.assembly_zero, self.assembly.calls()),
            per(self.gather_grad, self.assembly.calls()),
            per(self.assembly_zero_vals, self.assembly.calls()),
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
    pub(crate) threads: usize,
    pub(crate) timing: bool,
    /// The model's extended hook pushes COO entries, so the Hessian
    /// pattern is only knowable after a compute. Asked once per solve by
    /// running the hook (see
    /// `LmProblemInternals::extended_hook_writes_coo`), because nothing
    /// static can answer it.
    pub(crate) runtime_coo: bool,
    /// Every store of the solve, held as one `Vec<S>` type-erased. The
    /// list keeps whatever it has allocated between solves; `blocks_len`
    /// says how much of it this solve uses.
    pub(crate) blocks: Option<Box<dyn AnyStore>>,
    pub(crate) blocks_len: usize,
    /// The shape the stores were built for: the model's count per
    /// block array, as the generated `__shape` reads it. A direct call
    /// with a model of another shape trips on it.
    pub(crate) shape: Vec<u64>,
    /// What one whole store of the model holds, as the build records it.
    pub(crate) whole: StoreFootprint,
    pub(crate) cut: Cut,
    pub(crate) sweeps: ParTiming,
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
    /// What each store holds, in store order.
    pub held: Vec<StoreFootprint>,
    /// What one whole store of the model would hold; summed `held`
    /// against it is the duplication of the split.
    pub whole: StoreFootprint,
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
            shape: self.shape.clone(),
            whole: self.whole,
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
            blocks: None, blocks_len: 0, shape: Vec::new(), whole: StoreFootprint::default(),
            cut: Cut::new(), sweeps: ParTiming::default(),
        }
    }

    /// Time the phases of the solves through this context. Off by
    /// default: then the sweeps never read a clock. A solve with
    /// [`LmConfig::gather_timing`](crate::simple_lm::LmConfig::gather_timing)
    /// turns it on, and it stays on for the solves that follow.
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
    /// (0 is the pool's size). Every solve entry sets it from the config,
    /// so a call here matters only when the sweeps are driven without a
    /// solve. Setting it also resets the sweep timing: a solve's report
    /// counts what that solve did, not what the last one did.
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
            held: self.footprints(),
            whole: self.whole,
        })
    }
}

/// The thread count a new [`LmConfig`](crate::simple_lm::LmConfig) starts
/// from: `ARAEL_NUM_THREADS` when the environment sets it, else 1. On the
/// scale of `num_threads` (1 sequential, `n` threads, 0 every core), read
/// once per process. A count set in code wins over it, as with
/// `RAYON_NUM_THREADS` and `OMP_NUM_THREADS`. A value that is not a count
/// is an error, not 1.
pub fn default_num_threads() -> usize {
    static DEFAULT: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *DEFAULT.get_or_init(|| match std::env::var_os("ARAEL_NUM_THREADS") {
        None => 1,
        Some(v) => {
            let text = v.to_str()
                .unwrap_or_else(|| panic!("ARAEL_NUM_THREADS is not valid text: {:?}", v));
            num_threads_from(text).unwrap_or_else(|why| panic!("ARAEL_NUM_THREADS: {}", why))
        }
    })
}

/// `ARAEL_NUM_THREADS`'s text as a count: a non-negative integer, blanks
/// around it ignored; empty is unset.
fn num_threads_from(text: &str) -> Result<usize, String> {
    let t = text.trim();
    if t.is_empty() {
        return Ok(1);
    }
    t.parse::<usize>().map_err(|_| {
        format!("expected a thread count (1 sequential, n threads, 0 every core), got {:?}", text)
    })
}

/// The thread count a solve's `num_threads` setting means: 0 is the
/// rayon pool's size. Without the `threads` feature every count is 1.
pub(crate) fn pool_size(num_threads: usize) -> usize {
    #[cfg(feature = "threads")]
    {
        match num_threads {
            0 => rayon::current_num_threads(),
            n => n,
        }
    }
    #[cfg(not(feature = "threads"))]
    {
        let _ = num_threads;
        1
    }
}

#[cfg(test)]
mod num_threads_tests {
    use super::num_threads_from;

    #[test]
    fn a_count_is_read_and_empty_is_unset() {
        assert_eq!(num_threads_from("4"), Ok(4));
        assert_eq!(num_threads_from("0"), Ok(0));
        assert_eq!(num_threads_from(" 8 "), Ok(8));
        assert_eq!(num_threads_from(""), Ok(1));
        assert_eq!(num_threads_from("  "), Ok(1));
    }

    #[test]
    fn anything_else_is_an_error() {
        for bad in ["four", "-1", "2.5", "1e3", "0x4"] {
            let why = num_threads_from(bad).expect_err(bad);
            assert!(why.contains(bad), "{}", why);
        }
    }
}
