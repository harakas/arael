// The configuration header every benchmark prints at startup.
//
// A pasted result has to carry the settings that produced it. Values come from
// the objects the run actually uses, not from re-reading the environment and
// re-applying defaults in the printing code -- that is how a header drifts from
// the run it describes. Each line names the env var that changes it, so the
// header doubles as the knob list.

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

    /// The one line every benchmark has ([`rounds_value`]).
    pub fn rounds(self, rounds: usize) -> Self {
        self.line("rounds", rounds_value(rounds))
    }

    /// The GPU the arael rows are given and their single-precision factor
    /// ([`crate::arael::gpu_value`], [`crate::arael::f32_factor_value`]).
    pub fn gpu(self) -> Self {
        self.line("arael gpu", crate::arael::gpu_value()).line("arael f32 factor", crate::arael::f32_factor_value())
    }

    /// And the pin, which every timing depends on.
    pub fn core(self) -> Self {
        let n = crate::pin::threads();
        let core = std::env::var("BENCH_CORE").unwrap_or_else(|_| "?".to_string());
        if crate::pin::pinned() {
            self.line("pinned to core", format!("{} (every thread pool capped at {}) [BENCH_THREADS, BENCH_PIN]", core, n))
        } else {
            self.line("pinned to core", format!("no, BENCH_PIN=0 (cpus allowed {}; every thread pool capped at {}) [BENCH_THREADS]", core, n))
        }
    }

    pub fn print(&self) {
        println!("=== {} configuration ===", self.title);
        let w = self.lines.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        for (key, value) in &self.lines {
            println!("{:<w$} : {}", key, value, w = w);
        }
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
