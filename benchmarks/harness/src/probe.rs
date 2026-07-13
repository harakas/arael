// Timing probes, and the rules for when a probe's number means anything.

/// Sub-rounds per probe: the reported time is the fastest of these.
///
/// One complete iteration is read off as t(2 iterations) - t(1 iteration).
/// Differencing two noisy measurements amplifies the noise, so each side is a
/// minimum before the subtraction rather than a single sample.
pub const PROBE_SUBROUNDS: usize = 2;

/// A first-iteration time means something only if that iteration WAS one
/// iteration: a single attempt, accepted. An iteration that rejected a step is
/// mostly wasted factorizations -- g2o at the wrong damping burned six trials in
/// its first iteration on sphere2500 -- and every number derived from it,
/// full-iter above all, inherits that. Returns NaN otherwise; the table prints
/// NaN as "-".
pub fn first_iter_ms(ms: f64, attempts: usize, accepted: usize) -> f64 {
    if attempts == 1 && accepted == 1 { ms } else { f64::NAN }
}

/// t(2 iterations), or `None` when it cannot be differenced against a clean
/// t(1): either the first iteration was not clean, or the second step was a
/// damping retry rather than an accepted step.
pub fn two_iter_ms(ms: f64, first_iter_ms: f64, two_accepted: usize) -> Option<f64> {
    (first_iter_ms.is_finite() && two_accepted >= 2).then_some(ms)
}

/// A measured millisecond value, or "-" when the harness could not measure it
/// cleanly.
pub fn fmt1(v: f64) -> String {
    if v.is_finite() { format!("{:.1}", v) } else { "-".to_string() }
}
