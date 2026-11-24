// The environment variables the benchmarks share, one name per setting.
//
// A setting that means the same in several benchmarks is read here, so it
// has one name and one set of values; benchmarks/README.md lists them. What
// one benchmark alone has (its dataset, its scene size) stays in that
// benchmark under its own prefix.

/// Names that were replaced: `(old, new)`. A run with an old name set stops
/// ([`check`]), so a setting typed under its old name is never ignored.
pub const RENAMED: &[(&str, &str)] = &[
    ("THREADS", "BENCH_THREADS"),
    ("SCHUR_THREADS", "BENCH_THREADS"),
    ("SLAM_VERBOSE", "VERBOSE"),
    ("SLAM_TIMING", "TIMING"),
    ("SLAM_DRIVER", "DRIVER"),
    ("PGO_DRIVER", "DRIVER"),
    ("SLAM_LAMBDA0", "ARAEL_LAMBDA0"),
    ("BAL_LAMBDA0", "ARAEL_LAMBDA0"),
    ("POSES", "SLAM_POSES"),
    ("BAL_SYSTEMS", "BENCH_SYSTEMS"),
    ("SLAM_SYSTEMS", "BENCH_SYSTEMS"),
    ("PGO_SYSTEMS", "BENCH_SYSTEMS"),
    ("PLANE_SYSTEMS", "BENCH_SYSTEMS"),
    ("LOC_SYSTEMS", "BENCH_SYSTEMS"),
    ("BAL_NO_MEM", "BENCH_NO_MEM"),
    ("SLAM_NO_MEM", "BENCH_NO_MEM"),
    ("PLANE_NO_MEM", "BENCH_NO_MEM"),
    ("LOC_NO_MEM", "BENCH_NO_MEM"),
    ("BAL_MEM_ITERS", "BENCH_MEM_ITERS"),
    ("SLAM_MEM_ITERS", "BENCH_MEM_ITERS"),
    ("PLANE_MEM_ITERS", "BENCH_MEM_ITERS"),
    ("LOC_MEM_ITERS", "BENCH_MEM_ITERS"),
    ("BAL_COV", "BENCH_COV"),
    ("SLAM_COV", "BENCH_COV"),
    ("LOC_COV", "BENCH_COV"),
    ("BAL_ORDERING", "ARAEL_ORDERING"),
    ("SLAM_ORDERING", "ARAEL_ORDERING"),
    ("PGO_ORDERING", "ARAEL_ORDERING"),
    ("BAL_SCHUR_POLICY", "ARAEL_SCHUR"),
    ("SLAM_SCHUR", "ARAEL_SCHUR"),
    ("PGO_SCHUR", "ARAEL_SCHUR"),
    ("PLANE_SCHUR", "ARAEL_SCHUR"),
    ("SLAM_ENVELOPE", "ARAEL_ENVELOPE"),
    ("PLANE_ENVELOPE", "ARAEL_ENVELOPE"),
    ("BAL_CG_TOL", "ARAEL_CG_TOL"),
    ("SLAM_CG_TOL", "ARAEL_CG_TOL"),
    ("BAL_CG_MAXITER", "ARAEL_CG_MAXITER"),
    ("SLAM_CG_MAXITER", "ARAEL_CG_MAXITER"),
    ("BAL_CG_RESTART", "ARAEL_CG_RESTART"),
    ("SLAM_CG_RESTART", "ARAEL_CG_RESTART"),
    ("BAL_COV_ORDERING", "ARAEL_COV_ORDERING"),
    ("SLAM_COV_ORDERING", "ARAEL_COV_ORDERING"),
];

/// Names that were removed, with nothing in their place. A run with one
/// set stops ([`check`]).
pub const REMOVED: &[&str] = &[
    // slam: the memory pass and the covariance benchmark on the
    // well_conditioned preset
    "SLAM_WC",
];

/// Stops the process when a replaced name ([`RENAMED`]) is set, saying
/// which name to use, a removed one ([`REMOVED`]), or one of `not_read`:
/// the shared names this binary does not read. Every benchmark binary
/// calls it first.
///
/// ```ignore
/// fn main() {
///     // pgo has no covariance benchmark and no conjugate gradient rows
///     bench_harness::env::check(&["BENCH_COV", "ARAEL_CG_TOL", "ARAEL_CG_MAXITER", "ARAEL_CG_RESTART"]);
///     bench_harness::pin::enforce_cores();
/// }
/// ```
pub fn check(not_read: &[&str]) {
    if let Some(message) = refusal(|name| std::env::var_os(name).is_some(), not_read) {
        eprintln!("{message}");
        std::process::exit(2);
    }
}

/// What [`check`] prints: one line per replaced or unread name that
/// `is_set`, or `None` when there is none.
fn refusal(is_set: impl Fn(&str) -> bool, not_read: &[&str]) -> Option<String> {
    let renamed = RENAMED
        .iter()
        .filter(|(old, _)| is_set(old))
        .map(|(old, new)| format!("{old} is set: the benchmarks read {new} instead (see benchmarks/README.md)"));
    let removed = REMOVED
        .iter()
        .filter(|name| is_set(name))
        .map(|name| format!("{name} is set: it was removed and nothing reads it"));
    let unread = not_read
        .iter()
        .filter(|name| is_set(name))
        .map(|name| format!("{name} is set: this benchmark does not read it (see benchmarks/README.md)"));
    let lines: Vec<String> = renamed.chain(removed).chain(unread).collect();
    (!lines.is_empty()).then(|| lines.join("\n"))
}

/// The value of `name`, parsed. Unset is `None`; a value that does not
/// parse stops the run, naming what was `expected`.
pub fn parsed<T: std::str::FromStr>(name: &str, expected: &str) -> Option<T> {
    parsed_from(name, std::env::var(name).ok().as_deref(), expected)
}

fn parsed_from<T: std::str::FromStr>(name: &str, value: Option<&str>, expected: &str) -> Option<T> {
    let value = value?;
    Some(value.trim().parse().unwrap_or_else(|_| panic!("{name}={value}: expected {expected}")))
}

/// An on/off variable: `1` is on, `0` or unset is off. Any other value
/// stops the run.
pub fn flag(name: &str) -> bool {
    flag_from(name, std::env::var(name).ok().as_deref())
}

fn flag_from(name: &str, value: Option<&str>) -> bool {
    match value.map(str::trim) {
        None | Some("0") => false,
        Some("1") => true,
        Some(other) => panic!("{name}={other}: expected 0 or 1"),
    }
}

/// ROUNDS: how many interleaved rounds the benchmark runs, `default`
/// when unset. The reported time is the minimum over them.
pub fn rounds(default: usize) -> usize {
    rounds_from(std::env::var("ROUNDS").ok().as_deref(), default)
}

fn rounds_from(value: Option<&str>, default: usize) -> usize {
    match parsed_from::<usize>("ROUNDS", value, "a round count of at least 1") {
        Some(0) => panic!("ROUNDS=0: expected a round count of at least 1"),
        Some(n) => n,
        None => default,
    }
}

/// VERBOSE=1: arael logs each iteration of its solves, and the external
/// runners' messages are printed ([`crate::external::run`]).
pub fn verbose() -> bool {
    flag("VERBOSE")
}

/// TIMING=1: each arael solve prints its phase times
/// ([`crate::arael::print_timing`]), and the external runners print theirs.
pub fn timing() -> bool {
    flag("TIMING")
}

/// REPORT=1: the table prints the solve report of each arael row under it.
pub fn report() -> bool {
    flag("REPORT")
}

/// BENCH_COV=1: the benchmark times covariance recovery instead of the
/// solves.
pub fn cov() -> bool {
    flag("BENCH_COV")
}

/// Whether the peak memory of the in-process rows is measured: on unless
/// BENCH_NO_MEM=1.
pub fn memory_pass() -> bool {
    !flag("BENCH_NO_MEM")
}

/// BENCH_MEM_ITERS: the iterations of the solve the memory pass runs, 3
/// when unset. The peak is reached in the first factorization, so a short
/// solve measures it.
pub fn mem_iters() -> usize {
    parsed("BENCH_MEM_ITERS", "an iteration count").unwrap_or(3)
}

/// BENCH_SYSTEMS: the row filter, or `None` when every row runs.
pub fn systems() -> Option<String> {
    std::env::var("BENCH_SYSTEMS").ok()
}

/// Whether the row `label` runs under BENCH_SYSTEMS ([`wants_in`]).
pub fn wants(label: &str) -> bool {
    wants_in(systems().as_deref(), label)
}

/// Whether `label` passes `filter`: a comma-separated list of patterns, of
/// which one must match. A pattern matches a label that contains it; a
/// pattern ending in `$` matches the whole label only. No filter passes
/// every label.
///
/// ```ignore
/// assert!(wants_in(Some("ceres, arael LM f64$"), "arael LM f64"));
/// assert!(!wants_in(Some("arael LM f64$"), "arael LM f64 schur"));
/// ```
pub fn wants_in(filter: Option<&str>, label: &str) -> bool {
    let Some(filter) = filter else { return true };
    filter.split(',').map(str::trim).filter(|p| !p.is_empty()).any(|pattern| match pattern.strip_suffix('$') {
        Some(whole) => label == whole,
        None => label.contains(pattern),
    })
}

/// The header's value for the row filter.
pub fn systems_value() -> String {
    format!("{} [BENCH_SYSTEMS]", systems().as_deref().unwrap_or("all"))
}

/// Prints that a filtered run cannot be validated across systems, when
/// BENCH_SYSTEMS is set.
pub fn note_filtered() {
    if let Some(filter) = systems() {
        eprintln!("BENCH_SYSTEMS={filter} -- partial run, the cross-system validation covers only what ran");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A replaced name that is set is reported with its new name; names
    /// in use are not.
    #[test]
    fn a_replaced_name_is_refused_with_its_new_name() {
        assert_eq!(refusal(|_| false, &[]), None);
        assert_eq!(refusal(|name| name == "BENCH_THREADS" || name == "ROUNDS", &[]), None);
        let one = refusal(|name| name == "PGO_SYSTEMS", &[]).unwrap();
        assert_eq!(one, "PGO_SYSTEMS is set: the benchmarks read BENCH_SYSTEMS instead (see benchmarks/README.md)");
        let two = refusal(|name| name == "THREADS" || name == "SLAM_TIMING", &[]).unwrap();
        assert!(two.contains("THREADS is set: the benchmarks read BENCH_THREADS") && two.contains("SLAM_TIMING is set: the benchmarks read TIMING"), "{two}");
    }

    /// A shared name the benchmark does not read is refused when set, and
    /// only then.
    #[test]
    fn a_shared_name_the_benchmark_does_not_read_is_refused() {
        let not_read = ["BENCH_COV", "ARAEL_ENVELOPE"];
        assert_eq!(refusal(|name| name == "BENCH_SYSTEMS", &not_read), None);
        let one = refusal(|name| name == "BENCH_COV", &not_read).unwrap();
        assert_eq!(one, "BENCH_COV is set: this benchmark does not read it (see benchmarks/README.md)");
        let both = refusal(|name| name == "BENCH_COV" || name == "BAL_COV", &not_read).unwrap();
        assert_eq!(both.lines().count(), 2, "{both}");
    }

    /// A removed name is refused in every benchmark, whatever it reads.
    #[test]
    fn a_removed_name_is_refused() {
        assert_eq!(refusal(|name| name == "SLAM_WC", &[]).unwrap(), "SLAM_WC is set: it was removed and nothing reads it");
        for name in REMOVED {
            assert!(!RENAMED.iter().any(|(old, new)| old == name || new == name), "{name} is both removed and renamed");
        }
    }

    /// ROUNDS: unset is the benchmark's default, a count of at least 1 is
    /// taken, 0 and words are errors.
    #[test]
    fn rounds_is_a_count_of_at_least_one() {
        assert_eq!(rounds_from(None, 5), 5);
        assert_eq!(rounds_from(Some("1"), 5), 1);
        assert!(std::panic::catch_unwind(|| rounds_from(Some("0"), 5)).is_err());
        assert!(std::panic::catch_unwind(|| rounds_from(Some("many"), 5)).is_err());
    }

    /// No name is both a replacement and replaced: a refusal never names
    /// a name that is refused in turn.
    #[test]
    fn no_name_is_both_replaced_and_in_use() {
        for (old, new) in RENAMED {
            assert!(!RENAMED.iter().any(|(o, _)| o == new), "{new} replaces {old} and is itself replaced");
        }
    }

    /// A flag is 1, 0 or unset; `TIMING=0` is off, and a word is an error.
    #[test]
    fn a_flag_is_0_or_1() {
        assert!(!flag_from("TIMING", None) && !flag_from("TIMING", Some("0")) && flag_from("TIMING", Some("1")));
        assert!(flag_from("TIMING", Some(" 1 ")));
        for bad in ["yes", "true", "", "2"] {
            assert!(std::panic::catch_unwind(|| flag_from("TIMING", Some(bad))).is_err(), "{bad:?} accepted");
        }
    }

    /// A value that does not parse is an error, not the default.
    #[test]
    fn a_value_that_does_not_parse_is_an_error() {
        assert_eq!(parsed_from::<usize>("ROUNDS", None, "a count"), None);
        assert_eq!(parsed_from::<usize>("ROUNDS", Some("5"), "a count"), Some(5));
        assert!(std::panic::catch_unwind(|| parsed_from::<usize>("ROUNDS", Some("five"), "a count")).is_err());
    }

    /// The row filter: substrings, `$` for a whole label, empty patterns
    /// ignored.
    #[test]
    fn the_row_filter_matches_substrings_and_whole_labels() {
        assert!(wants_in(None, "anything"));
        assert!(wants_in(Some("arael"), "arael LM f64 schur"));
        assert!(wants_in(Some("ceres, schur"), "arael LM f64 schur"));
        assert!(!wants_in(Some("ceres,g2o"), "arael LM f64 schur"));
        assert!(wants_in(Some("arael LM f32 schur$"), "arael LM f32 schur"));
        assert!(!wants_in(Some("arael LM f32 schur$"), "arael LM f32 schur-cg"));
        assert!(!wants_in(Some(","), "arael LM f64"));
    }
}
