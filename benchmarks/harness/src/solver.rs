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
//     so its time -- and everything derived from it -- is not reported at all.
//
// The C++ runners get the same treatment from benchmarks/cpp/bench.h.

use crate::probe::{first_iter_ms, two_iter_ms, PROBE_SUBROUNDS};
use crate::table::Row;

/// What one capped solve did.
pub struct Outcome<S> {
    /// Accepted, cost-decreasing steps.
    pub accepted: usize,
    /// Attempts: accepted steps plus damping retries, each of which costs a
    /// factorization. Equal to `accepted` for a solver with no retry loop, and
    /// for one whose retries cannot be counted (GTSAM).
    pub attempts: usize,
    pub solution: S,
}

/// Fastest of [`PROBE_SUBROUNDS`] runs of the same capped solve.
fn best_of<S>(solve: &mut impl FnMut(usize) -> Outcome<S>, max_iters: usize) -> (f64, Outcome<S>) {
    let mut best = f64::INFINITY;
    let mut out = None;
    for _ in 0..PROBE_SUBROUNDS {
        let t0 = std::time::Instant::now();
        let o = solve(max_iters);
        best = best.min(t0.elapsed().as_secs_f64() * 1e3);
        out = Some(o);
    }
    (best, out.expect("PROBE_SUBROUNDS must be at least 1"))
}

/// Drive the probes and the full solve.
///
/// `solve` must start from the problem's INITIAL state every time. A probe that
/// left the model advanced gave the solve that followed a warm start it never
/// asked for, and the iteration count changed under us.
pub fn run<S>(full_iters: usize, mut solve: impl FnMut(usize) -> Outcome<S>) -> Row<S> {
    let _ = solve(1); // warmup, discarded

    let (first_ms, first) = best_of(&mut solve, 1);
    let first_ms = first_iter_ms(first_ms, first.attempts, first.accepted);

    let (two_ms, two) = best_of(&mut solve, 2);
    let two_ms = two_iter_ms(two_ms, first_ms, two.accepted);

    let t0 = std::time::Instant::now();
    let full = solve(full_iters);
    let solve_ms = t0.elapsed().as_secs_f64() * 1e3;

    Row::new(solve_ms, first_ms, full.attempts, full.solution)
        .accepted(full.accepted)
        .full_ms(two_ms)
}
