//! The worker threads of arael's own threaded stages: the cost and
//! assembly sweeps, and the linear solve's stages that do not run on
//! faer's kernels.
//!
//! Inner API. Public so that arael and its generated sweeps can reach
//! it, and not stable.
//!
//! One worker per task beyond the first, each parked on its own
//! condition variable with room for one job, never spinning; the
//! calling thread runs the first task and blocks until every worker
//! has finished. The workers are spawned on first use, kept until
//! [`shutdown`] joins them, and grown when a larger task count asks.
//! One dispatch runs at a time: a second caller waits for the first to
//! finish.
//!
//! A panic in any task, a worker's or the caller's own, resumes on the
//! calling thread once every worker has reported, so the tasks are
//! never left running behind a returned call.
//!
//! The task borrows the caller's data, and the workers outlive the
//! call, so the pointer handed to them has its lifetime erased.
//! That is the contract of `std::thread::scope`, over threads that
//! persist: it holds because [`run`] does not return before every
//! worker has reported, whatever the tasks did. The erasure and the
//! dereference are the module's only unsafe code.

use std::any::Any;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::thread::JoinHandle;

type Task = dyn Fn(usize) + Sync;
type Panic = Box<dyn Any + Send + 'static>;

/// One job: the task and the index it runs for. The task lives on the
/// caller's stack; the caller does not return from [`run`] until every
/// job has finished, which keeps the pointer valid while a worker holds
/// it.
struct Job {
    task: *const Task,
    index: usize,
}
// Read only while the caller blocks in `run`, and the task is `Sync`.
unsafe impl Send for Job {}

/// What a worker finds in its slot when woken.
enum Order {
    Run(Job),
    Stop,
}

struct Worker {
    slot: Mutex<Option<Order>>,
    wake: Condvar,
}

/// A worker and the thread it runs on.
struct Hand {
    worker: Arc<Worker>,
    thread: JoinHandle<()>,
}

/// What the workers report back: how many have finished, and the first
/// panic among them.
struct Done {
    state: Mutex<(usize, Option<Panic>)>,
    all: Condvar,
}

struct Pool {
    dispatch: Mutex<()>,
    hands: Mutex<Vec<Hand>>,
    done: Arc<Done>,
}

const WORKER_NAME: &str = "arael-pool-";

/// A lock that survives a poisoned mutex: the state behind every mutex
/// here is consistent whether or not its last holder panicked.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn wait<'a, T>(c: &Condvar, g: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
    c.wait(g).unwrap_or_else(|e| e.into_inner())
}

fn pool() -> &'static Pool {
    static POOL: OnceLock<Pool> = OnceLock::new();
    POOL.get_or_init(|| Pool {
        dispatch: Mutex::new(()),
        hands: Mutex::new(Vec::new()),
        done: Arc::new(Done { state: Mutex::new((0, None)), all: Condvar::new() }),
    })
}

/// True on a worker thread, where a dispatch or a shutdown would wait
/// on itself.
fn on_a_worker() -> bool {
    std::thread::current().name().is_some_and(|n| n.starts_with(WORKER_NAME))
}

impl Pool {
    /// The first `n` workers, spawning what is missing.
    fn ensure(&self, n: usize) -> Vec<Arc<Worker>> {
        let mut hands = lock(&self.hands);
        while hands.len() < n {
            let worker = Arc::new(Worker { slot: Mutex::new(None), wake: Condvar::new() });
            let (w, done) = (Arc::clone(&worker), Arc::clone(&self.done));
            let thread = std::thread::Builder::new()
                .name(format!("{}{}", WORKER_NAME, hands.len() + 1))
                .spawn(move || worker_loop(w, done))
                .expect("spawn a pool worker");
            hands.push(Hand { worker, thread });
        }
        hands[..n].iter().map(|h| Arc::clone(&h.worker)).collect()
    }
}

fn worker_loop(w: Arc<Worker>, done: Arc<Done>) {
    loop {
        let order = {
            let mut slot = lock(&w.slot);
            while slot.is_none() {
                slot = wait(&w.wake, slot);
            }
            slot.take().unwrap()
        };
        let job = match order {
            Order::Run(job) => job,
            Order::Stop => return,
        };
        // SAFETY: the caller blocks in `run` until this worker has
        // reported below, so the task it points at is alive.
        let outcome = catch_unwind(AssertUnwindSafe(|| unsafe { (*job.task)(job.index) }));
        let mut state = lock(&done.state);
        state.0 += 1;
        if let Err(p) = outcome
            && state.1.is_none()
        {
            state.1 = Some(p);
        }
        done.all.notify_one();
    }
}

/// Run `task(i)` for every `i` below `n`: index 0 on the calling thread,
/// the others one per worker, and return when all have finished. The
/// task must be safe to call from several threads at once for distinct
/// indices, which is what its `Sync` bound and the distinct indices
/// give. A panic in any of them resumes here after the join. Not to be
/// called from inside a task.
pub fn run(n: usize, task: &(dyn Fn(usize) + Sync)) {
    if n <= 1 {
        if n == 1 {
            task(0);
        }
        return;
    }
    assert!(!on_a_worker(), "pool::run from inside a pool task would wait on itself");
    let pool = pool();
    let _one_at_a_time = lock(&pool.dispatch);
    let workers = pool.ensure(n - 1);
    // The task outlives this call: the wait below does not return before
    // every worker has finished with it. That is the whole contract.
    let ptr: *const Task =
        unsafe { std::mem::transmute::<&(dyn Fn(usize) + Sync), &'static Task>(task) };
    *lock(&pool.done.state) = (0, None);
    for (k, w) in workers.iter().enumerate() {
        let mut slot = lock(&w.slot);
        *slot = Some(Order::Run(Job { task: ptr, index: k + 1 }));
        w.wake.notify_one();
    }
    let mine = catch_unwind(AssertUnwindSafe(|| task(0))).err();
    let theirs = {
        let mut state = lock(&pool.done.state);
        while state.0 < n - 1 {
            state = wait(&pool.done.all, state);
        }
        state.1.take()
    };
    if let Some(p) = mine.or(theirs) {
        resume_unwind(p);
    }
}

/// Run `f(i, &mut items[i])` for every item, each on a thread of its
/// own, and return when all have finished.
pub fn run_over<M: Send>(items: &mut [M], f: impl Fn(usize, &mut M) + Sync) {
    // Each item waits in a slot of its own, and the task for its index
    // takes it out: no item is reached from two threads.
    let slots: Vec<Mutex<Option<&mut M>>> = items.iter_mut().map(|m| Mutex::new(Some(m))).collect();
    run(slots.len(), &|i| {
        let m = lock(&slots[i]).take().expect("an item is taken once");
        f(i, m);
    });
}

/// Stop every worker and join its thread. Waits for a dispatch in
/// progress to finish first. The next [`run`] that needs workers spawns
/// them again. Not to be called from inside a task.
pub fn shutdown() {
    assert!(!on_a_worker(), "pool::shutdown from inside a pool task would wait on itself");
    let pool = pool();
    let _one_at_a_time = lock(&pool.dispatch);
    let mut hands = lock(&pool.hands);
    for h in hands.iter() {
        *lock(&h.worker.slot) = Some(Order::Stop);
        h.worker.wake.notify_one();
    }
    for h in hands.drain(..) {
        // A worker leaves its loop only through Stop; a task's panic is
        // caught inside it, so the join has nothing to report.
        let _ = h.thread.join();
    }
}

/// How many worker threads are alive.
pub fn workers() -> usize {
    lock(&pool().hands).len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    /// The pool is one per process, so the tests that count its workers
    /// or stop it take turns.
    fn turn() -> MutexGuard<'static, ()> {
        static TURN: Mutex<()> = Mutex::new(());
        lock(&TURN)
    }

    #[test]
    fn every_index_runs_once() {
        let _t = turn();
        for n in 0..10 {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            run(n, &|i| { hits[i].fetch_add(1, Ordering::SeqCst); });
            for (i, h) in hits.iter().enumerate() {
                assert_eq!(h.load(Ordering::SeqCst), 1, "index {} of {}", i, n);
            }
        }
    }

    #[test]
    fn the_first_index_runs_on_the_calling_thread() {
        let _t = turn();
        let caller = std::thread::current().id();
        let seen = Mutex::new(Vec::new());
        run(4, &|i| { seen.lock().unwrap().push((i, std::thread::current().id())); });
        let seen = seen.lock().unwrap();
        for (i, id) in seen.iter() {
            assert_eq!(*id == caller, *i == 0, "index {}", i);
        }
    }

    #[test]
    fn items_are_handed_out_one_per_thread() {
        let _t = turn();
        let mut items: Vec<usize> = vec![0; 6];
        run_over(&mut items, |i, m| { *m = i * 10; });
        assert_eq!(items, [0, 10, 20, 30, 40, 50]);
    }

    #[test]
    fn a_panicking_task_resumes_on_the_caller_after_the_join() {
        let _t = turn();
        let finished = AtomicUsize::new(0);
        let r = catch_unwind(AssertUnwindSafe(|| {
            run(4, &|i| {
                if i == 2 { panic!("task {} failed", i); }
                std::thread::sleep(Duration::from_millis(5));
                finished.fetch_add(1, Ordering::SeqCst);
            });
        }));
        let p = r.expect_err("the panic reaches the caller");
        assert_eq!(p.downcast_ref::<String>().map(String::as_str), Some("task 2 failed"));
        assert_eq!(finished.load(Ordering::SeqCst), 3, "the other tasks ran to the end");
        // The pool is whole afterwards.
        let hits = AtomicUsize::new(0);
        run(4, &|_| { hits.fetch_add(1, Ordering::SeqCst); });
        assert_eq!(hits.load(Ordering::SeqCst), 4);
    }

    #[test]
    fn a_panic_on_the_calling_thread_waits_for_the_workers() {
        let _t = turn();
        let finished = AtomicUsize::new(0);
        let r = catch_unwind(AssertUnwindSafe(|| {
            run(4, &|i| {
                if i == 0 { panic!("the caller's own task failed"); }
                std::thread::sleep(Duration::from_millis(5));
                finished.fetch_add(1, Ordering::SeqCst);
            });
        }));
        assert!(r.is_err());
        assert_eq!(finished.load(Ordering::SeqCst), 3, "the workers finished before the panic resumed");
    }

    #[test]
    fn the_pool_grows_with_the_call_and_keeps_its_workers() {
        let _t = turn();
        shutdown();
        for (n, alive) in [(2usize, 1usize), (8, 7), (3, 7), (6, 7), (1, 7)] {
            let hits = AtomicUsize::new(0);
            run(n, &|_| { hits.fetch_add(1, Ordering::SeqCst); });
            assert_eq!(hits.load(Ordering::SeqCst), n);
            assert_eq!(workers(), alive, "after a run of {}", n);
        }
    }

    #[test]
    fn shutdown_joins_every_worker_and_the_next_run_spawns_again() {
        let _t = turn();
        shutdown();
        run(4, &|_| {});
        assert_eq!(workers(), 3);
        shutdown();
        assert_eq!(workers(), 0);
        shutdown();
        assert_eq!(workers(), 0, "a second shutdown finds nothing to do");
        let hits = AtomicUsize::new(0);
        run(4, &|_| { hits.fetch_add(1, Ordering::SeqCst); });
        assert_eq!(hits.load(Ordering::SeqCst), 4);
        assert_eq!(workers(), 3);
    }

    #[test]
    fn shutdown_waits_for_a_dispatch_in_progress() {
        let _t = turn();
        let finished = Arc::new(AtomicUsize::new(0));
        let started = Arc::new(AtomicUsize::new(0));
        let (f, s) = (Arc::clone(&finished), Arc::clone(&started));
        let caller = std::thread::spawn(move || {
            run(4, &|_| {
                s.fetch_add(1, Ordering::SeqCst);
                std::thread::sleep(Duration::from_millis(20));
                f.fetch_add(1, Ordering::SeqCst);
            });
        });
        // The dispatch holds its lock once a task runs, so the shutdown
        // below queues behind it.
        while started.load(Ordering::SeqCst) == 0 {
            std::thread::sleep(Duration::from_millis(1));
        }
        shutdown();
        assert_eq!(finished.load(Ordering::SeqCst), 4, "every task ran to the end before the join");
        caller.join().unwrap();
    }

    #[test]
    fn concurrent_callers_take_turns() {
        let _t = turn();
        let handles: Vec<_> = (0..3).map(|t| std::thread::spawn(move || {
            for _ in 0..20 {
                let mut items = vec![0usize; 5];
                run_over(&mut items, |i, m| { *m = t * 100 + i; });
                assert_eq!(items, (0..5).map(|i| t * 100 + i).collect::<Vec<_>>());
            }
        })).collect();
        for h in handles { h.join().unwrap(); }
    }
}
