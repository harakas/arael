// The probe driver every Rust runner goes through.
//
// A runner supplies one thing: a solve capped at N iterations, from a fresh
// problem, reporting what it did. This drives the probes, applies the rules a
// probe's number has to satisfy, and hands back a table row.
//
// The rules are not decoration. Each is a bug these benchmarks had:
//   - the first solve in a process pays cold allocator and cache costs the rest
//     do not, so an un-warmed probe charges them to whichever probe runs first;
//   - a complete iteration is t(2 iters) - t(1 iter), and differencing two noisy
//     single samples produced a NEGATIVE iteration time;
//   - a "first iteration" that rejected a step is mostly wasted factorizations,
//     so its time -- and everything derived from it -- is not reported at all;
//   - a probe has to reset the problem before it can re-solve it, and timing
//     that reset charges the solver for it. It is the runner, not this driver,
//     that starts the clock ([`timed`]), at the solve and not before -- the
//     boundary every C++ runner draws, each starting its timer once its problem
//     is built. On a slow core the reset is not small: an arael model clone is
//     ~8 ms on a Pi Zero.
//
// The C++ runners get the same treatment from benchmarks/cpp/bench.h.

use crate::probe::{first_iter_ms, two_iter_ms, PROBE_SUBROUNDS};
use crate::table::Row;

/// What one capped solve did.
pub struct Outcome<S> {
    /// How long the SOLVE took -- not the problem reset that had to precede it.
    /// Measure it with [`timed`].
    pub ms: f64,
    /// Accepted, cost-decreasing steps.
    pub accepted: usize,
    /// Attempts: accepted steps plus damping retries, each of which costs a
    /// factorization. Equal to `accepted` for a solver with no retry loop, and
    /// for one whose retries cannot be counted (GTSAM).
    pub attempts: usize,
    pub solution: S,
}

/// Time one solve, in milliseconds. Wrap the solve call and nothing else: not
/// the model clone or graph rebuild that reset the problem, not the extraction
/// of the solution afterwards.
pub fn timed<T>(solve: impl FnOnce() -> T) -> (f64, T) {
    let t0 = std::time::Instant::now();
    let out = solve();
    (t0.elapsed().as_secs_f64() * 1e3, out)
}

/// Fastest of [`PROBE_SUBROUNDS`] runs of the same capped solve.
fn best_of<S>(solve: &mut impl FnMut(usize) -> Outcome<S>, max_iters: usize) -> Outcome<S> {
    let mut best: Option<Outcome<S>> = None;
    for _ in 0..PROBE_SUBROUNDS {
        let o = solve(max_iters);
        if best.as_ref().is_none_or(|b| o.ms < b.ms) {
            best = Some(o);
        }
    }
    best.expect("PROBE_SUBROUNDS must be at least 1")
}

/// Drive the probes and the full solve.
///
/// `solve` must start from the problem's INITIAL state every time. A probe that
/// left the model advanced gave the solve that followed a warm start it never
/// asked for, and the iteration count changed under us.
pub fn run<S>(full_iters: usize, mut solve: impl FnMut(usize) -> Outcome<S>) -> Row<S> {
    let _ = solve(1); // warmup, discarded

    let first = best_of(&mut solve, 1);
    let first_ms = first_iter_ms(first.ms, first.attempts, first.accepted);

    let two = best_of(&mut solve, 2);
    let two_ms = two_iter_ms(two.ms, first_ms, two.accepted);

    let full = solve(full_iters);

    Row::new(full.ms, first_ms, full.attempts, full.solution)
        .accepted(full.accepted)
        .full_ms(two_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The problem reset a runner does before each probe must not be charged to
    /// the solver. A runner that resets slowly (an arael model clone, a factrs
    /// graph rebuild) would otherwise report a first iteration inflated by it.
    #[test]
    fn the_problem_reset_is_not_timed() {
        let reset = std::time::Duration::from_millis(30);
        let row = run(3, |iters| {
            std::thread::sleep(reset); // the reset: rebuild/clone the problem
            let (ms, _) = timed(|| std::thread::sleep(std::time::Duration::from_millis(1)));
            Outcome { ms, accepted: iters, attempts: iters, solution: () }
        });
        assert!(row.first_iter_ms < 15.0,
            "the {:?} reset leaked into the first iteration: {} ms",
            reset, row.first_iter_ms);
        assert!(row.solve_ms < 15.0,
            "the {:?} reset leaked into the solve: {} ms", reset, row.solve_ms);
    }
}
