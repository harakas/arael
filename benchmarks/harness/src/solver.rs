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

/// BENCH_MAX_ITERS: the iterations the reported solve runs, in place of
/// the runner's own count, as for the C++ runners (`benchmarks/cpp/bench.h`).
/// A capped run is a diagnostic, not a table. A value that is not a count of
/// at least 1 is an error.
pub fn max_iters() -> Option<usize> {
    max_iters_from(std::env::var("BENCH_MAX_ITERS").ok().as_deref())
}

/// The iterations the reported solve runs: `wanted`, or [`max_iters`].
pub fn full_iters(wanted: usize) -> usize {
    max_iters().unwrap_or(wanted)
}

fn max_iters_from(v: Option<&str>) -> Option<usize> {
    let v = v?;
    match v.parse() {
        Ok(n) if n >= 1 => Some(n),
        _ => panic!("BENCH_MAX_ITERS={}: expected an iteration count of at least 1", v),
    }
}

/// BENCH_SOLVE_ONLY=1: a row runs the reported solve alone, without the
/// warmup and the probes, as do the C++ runners. For one solve traced or
/// probed (with BENCH_MAX_ITERS=1, one attempt); its first-iteration time is
/// the solve's own when it was one clean accepted step, and it has no
/// full-iter. `0` or unset: the probes too; any other value is an error.
pub fn solve_only() -> bool {
    solve_only_from(std::env::var("BENCH_SOLVE_ONLY").ok().as_deref())
}

fn solve_only_from(v: Option<&str>) -> bool {
    match v {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => panic!("BENCH_SOLVE_ONLY={other}: expected 0 or 1"),
    }
}

/// Drive the probes and the full solve, the full solve held to
/// [`full_iters`] and alone under [`solve_only`].
///
/// `solve` must start from the problem's INITIAL state every time. A probe that
/// left the model advanced gave the solve that followed a warm start it never
/// asked for, and the iteration count changed under us.
pub fn run<S>(full_iters: usize, solve: impl FnMut(usize) -> Outcome<S>) -> Row<S> {
    drive(self::full_iters(full_iters), solve_only(), solve)
}

fn drive<S>(full_iters: usize, solve_only: bool, mut solve: impl FnMut(usize) -> Outcome<S>) -> Row<S> {
    if solve_only {
        let o = solve(full_iters);
        return Row::new(o.ms, first_iter_ms(o.ms, o.attempts, o.accepted), o.attempts, o.solution).accepted(o.accepted);
    }
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
        let row = drive(3, false, |iters| {
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

    /// The probes run at 1 and 2 iterations whatever the cap; the full
    /// solve takes the cap.
    #[test]
    fn the_cap_holds_the_full_solve() {
        let mut caps = Vec::new();
        drive(7, false, |iters| {
            caps.push(iters);
            Outcome { ms: 1.0, accepted: iters, attempts: iters, solution: () }
        });
        assert_eq!(caps, [1, 1, 1, 2, 2, 7]);
    }

    /// Solve-only runs the full solve once, held to the cap, and reports
    /// it: its first-iteration time only when it was one clean accepted
    /// step, and no t(2).
    #[test]
    fn solve_only_is_one_capped_solve() {
        let mut caps = Vec::new();
        let row = drive(1, true, |iters| {
            caps.push(iters);
            Outcome { ms: 2.0, accepted: 1, attempts: 1, solution: () }
        });
        assert_eq!(caps, [1]);
        assert_eq!((row.solve_ms, row.first_iter_ms, row.iterations, row.accepted, row.full_ms), (2.0, 2.0, 1, Some(1), None));
        // a retried first step has no first-iteration time
        let row = drive(3, true, |iters| Outcome { ms: 5.0, accepted: 2, attempts: iters, solution: () });
        assert_eq!((row.solve_ms, row.iterations, row.accepted), (5.0, 3, Some(2)));
        assert!(row.first_iter_ms.is_nan());
    }

    /// BENCH_MAX_ITERS is a count of at least 1; anything else is an
    /// error.
    #[test]
    fn bench_max_iters_is_a_count() {
        assert_eq!(max_iters_from(None), None);
        assert_eq!(max_iters_from(Some("3")), Some(3));
        for bad in ["0", "x", ""] {
            assert!(std::panic::catch_unwind(|| max_iters_from(Some(bad))).is_err(), "{bad:?} accepted");
        }
    }

    /// BENCH_SOLVE_ONLY is 1 or 0 (or unset); anything else is an error.
    #[test]
    fn bench_solve_only_is_0_or_1() {
        assert!(!solve_only_from(None) && !solve_only_from(Some("0")) && solve_only_from(Some("1")));
        for bad in ["yes", "true", ""] {
            assert!(std::panic::catch_unwind(|| solve_only_from(Some(bad))).is_err(), "{bad:?} accepted");
        }
    }
}
