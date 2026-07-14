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

    /// The one line every benchmark has: how many interleaved rounds, and how
    /// many sub-rounds each probe runs.
    pub fn rounds(self, rounds: usize) -> Self {
        self.line("rounds", format!("{} [ROUNDS], {} probe sub-rounds per round",
            rounds, crate::probe::PROBE_SUBROUNDS))
    }

    /// And the pin, which every timing depends on.
    pub fn core(self) -> Self {
        let n = crate::pin::threads();
        self.line("pinned to core", format!("{} (every thread pool capped at {}) [BENCH_THREADS]",
            std::env::var("BENCH_CORE").unwrap_or_else(|_| "?".to_string()), n))
    }

    pub fn print(&self) {
        println!("=== {} configuration ===", self.title);
        let w = self.lines.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        for (key, value) in &self.lines {
            println!("{:<w$} : {}", key, value, w = w);
        }
    }
}

/// "run" or "skipped", for an optional system.
pub fn on_off(skipped: bool) -> &'static str {
    if skipped { "skipped" } else { "run" }
}
