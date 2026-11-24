// The configuration header a benchmark prints at startup: one line per
// setting, with the environment variable that changes it in brackets.
//
// The values are read from the objects the run uses, not from the
// environment again, so the header says what ran.

pub struct Header {
    title: String,
    lines: Vec<(String, String)>,
}

impl Header {
    pub fn new(title: &str) -> Self {
        Header { title: title.to_string(), lines: Vec::new() }
    }

    pub fn line(mut self, key: &str, value: impl std::fmt::Display) -> Self {
        self.lines.push((key.to_string(), value.to_string()));
        self
    }

    /// The `rounds` line ([`rounds_value`]).
    pub fn rounds(self, rounds: usize) -> Self {
        self.line("rounds", rounds_value(rounds))
    }

    /// The `systems` line: the row filter ([`crate::env::systems_value`]).
    pub fn systems(self) -> Self {
        self.line("systems", crate::env::systems_value())
    }

    /// The `arael gpu` and `arael f32 factor` lines
    /// ([`crate::arael::gpu_value`], [`crate::arael::f32_factor_value`]).
    pub fn gpu(self) -> Self {
        self.line("arael gpu", crate::arael::gpu_value()).line("arael f32 factor", crate::arael::f32_factor_value())
    }

    /// The `solver verbose` line: VERBOSE and TIMING.
    pub fn verbose(self) -> Self {
        let on_off = |on: bool| if on { "on" } else { "off" };
        self.line(
            "solver verbose",
            format!("{} [VERBOSE], per-solve timing {} [TIMING]", on_off(crate::env::verbose()), on_off(crate::env::timing())),
        )
    }

    /// The `memory pass` line: BENCH_NO_MEM.
    pub fn memory(self) -> Self {
        self.line("memory pass", format!("{} [BENCH_NO_MEM]", if crate::env::memory_pass() { "on" } else { "off" }))
    }

    /// The `pinned to core` line ([`core_value`]).
    pub fn core(self) -> Self {
        let core = std::env::var("BENCH_CORE").unwrap_or_else(|_| "?".to_string());
        self.line("pinned to core", core_value(&core, crate::pin::threads(), crate::pin::pinned()))
    }

    pub fn print(&self) {
        println!("=== {} configuration ===", self.title);
        let w = self.lines.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        for (key, value) in &self.lines {
            println!("{:<w$} : {}", key, value, w = w);
        }
    }
}

/// The pin line's value: the cores the run is held to (`core`, as the
/// kernel names them), or that it is not pinned, and the thread count
/// every thread pool is capped at.
fn core_value(core: &str, threads: usize, pinned: bool) -> String {
    if pinned {
        format!("{core} (every thread pool capped at {threads}) [BENCH_THREADS, BENCH_PIN]")
    } else {
        format!("no, BENCH_PIN=0 (cpus allowed {core}; every thread pool capped at {threads}) [BENCH_THREADS]")
    }
}

/// The rounds line's value: how many interleaved rounds, how many
/// sub-rounds each probe runs (or none: the solve alone), and the full
/// solve's cap when set.
///
/// ```ignore
/// println!("rounds : {}", bench_harness::header::rounds_value(5));
/// ```
pub fn rounds_value(rounds: usize) -> String {
    let probes = if crate::solver::solve_only() {
        "the full solve alone, no warmup or probes [BENCH_SOLVE_ONLY]".to_string()
    } else {
        format!("{} probe sub-rounds per round", crate::probe::PROBE_SUBROUNDS)
    };
    let cap = match crate::solver::max_iters() {
        Some(n) => format!(", the full solve held to {} iteration{}: a diagnostic [BENCH_MAX_ITERS]", n, if n == 1 { "" } else { "s" }),
        None => String::new(),
    };
    format!("{} [ROUNDS], {}{}", rounds, probes, cap)
}

/// "run" or "skipped", for an optional system.
pub fn on_off(skipped: bool) -> &'static str {
    if skipped { "skipped" } else { "run" }
}

#[cfg(test)]
mod tests {
    use super::core_value;

    /// The pin line names the thread count the run resolved to, pinned or
    /// not.
    #[test]
    fn the_pin_line_reports_the_resolved_thread_count() {
        assert_eq!(core_value("12-15", 4, true), "12-15 (every thread pool capped at 4) [BENCH_THREADS, BENCH_PIN]");
        let free = core_value("0-15", 4, false);
        assert!(free.starts_with("no, BENCH_PIN=0") && free.contains("capped at 4"), "{free}");
    }
}
