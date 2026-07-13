// Sub-rounds for the one- and two-iteration probes.
//
// A complete iteration is read off as t(2 iters) - t(1 iter). Differencing two
// noisy measurements amplifies the noise, so each side is the fastest of
// several runs of the same probe rather than a single sample, and the minima
// are taken before the subtraction.
pub const PROBE_SUBROUNDS: usize = 2;

/// A first-iteration time means something only if that iteration WAS one
/// iteration: a single attempt, accepted. An iteration that rejected a step is
/// mostly wasted factorizations, and every number derived from it -- full-iter
/// above all, which is t(2) - t(1) -- inherits that. Returns NaN otherwise, and
/// the harness prints NaN as "-".
///
/// The subprocess runners report the same two counts over the JSON protocol
/// (`first_attempts`, `first_accepted`); this is the in-process half of it, so
/// arael and factrs are held to the rule the external systems are held to.
pub fn first_iter_ms(ms: f64, attempts: usize, accepted: usize) -> f64 {
    if attempts == 1 && accepted == 1 { ms } else { f64::NAN }
}

/// t(2 iterations), or `None` when it cannot be differenced against a clean
/// t(1): either the first iteration was not clean, or the second step was a
/// damping retry rather than an accepted step.
pub fn two_iter_ms(ms: f64, first_iter_ms: f64, two_accepted: usize) -> Option<f64> {
    (first_iter_ms.is_finite() && two_accepted >= 2).then_some(ms)
}
