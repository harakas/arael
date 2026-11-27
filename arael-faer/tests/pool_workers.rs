//! The worker pool's worker counts. The pool is one per process and the
//! unit tests of several modules run on it, so the tests that count its
//! workers have this process to themselves.

use arael_faer::pool::{run, shutdown, workers};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, MutexGuard};

/// The tests here count the same pool's workers, so they take turns.
fn turn() -> MutexGuard<'static, ()> {
    static TURN: Mutex<()> = Mutex::new(());
    TURN.lock().unwrap_or_else(|e| e.into_inner())
}

/// A run of `n` tasks leaves `n - 1` workers, and a smaller run after it
/// stops none.
#[test]
fn the_pool_grows_with_the_call_and_keeps_its_workers() {
    let _t = turn();
    shutdown();
    for (n, alive) in [(2usize, 1usize), (8, 7), (3, 7), (6, 7), (1, 7)] {
        let hits = AtomicUsize::new(0);
        run(n, &|_| {
            hits.fetch_add(1, Ordering::SeqCst);
        });
        assert_eq!(hits.load(Ordering::SeqCst), n);
        assert_eq!(workers(), alive, "after a run of {n}");
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
    run(4, &|_| {
        hits.fetch_add(1, Ordering::SeqCst);
    });
    assert_eq!(hits.load(Ordering::SeqCst), 4);
    assert_eq!(workers(), 3);
}
