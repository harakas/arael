// The protocol an external runner speaks: one JSON line on stdout.
//
// Parsed as JSON, not by searching the line for a key -- a substring search for
// "accepted" also matches inside "second_accepted", which is printed first, and
// that silently reported the probe's step count as the solve's for every
// subprocess system on one of pgo's two code paths.

use std::process::Command;

pub struct Protocol {
    pub solve_ms: f64,
    pub first_iter_ms: f64,
    /// Attempts: accepted steps plus damping retries.
    pub iterations: usize,
    pub accepted: Option<usize>,
    /// t(2 iterations), already gated on the second step having been accepted
    /// and the first iteration having been clean.
    pub full_ms: Option<f64>,
    /// Anything else the runner reported (a cost cross-check, say).
    pub json: serde_json::Value,
}

/// Run an external benchmark process and read its protocol line.
///
/// Asserts the child inherited the core pin: a subprocess that escaped it is
/// not measuring what the rest of the table measures.
pub fn run(mut cmd: Command) -> Protocol {
    let out = cmd.output().unwrap_or_else(|e| panic!("failed to run {:?}: {}", cmd, e));
    assert!(out.status.success(), "{:?} failed: {}", cmd,
        String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let line = text.lines().rev().find(|l| l.contains("solve_ms"))
        .unwrap_or_else(|| panic!("no protocol line from {:?}", cmd));
    let json: serde_json::Value = serde_json::from_str(line)
        .unwrap_or_else(|e| panic!("bad protocol line from {:?}: {} -- {}", cmd, e, line));
    let get = |key: &str| -> Option<f64> { json.get(key)?.as_f64() };

    let core = std::env::var("BENCH_CORE").unwrap();
    assert!(json["cpus_allowed"].as_str() == Some(core.as_str()),
        "{:?} not pinned to CPU {}: {}", cmd, core, line);

    // A first-iteration time is only meaningful if that iteration WAS one clean
    // iteration: a single attempt, accepted. Everything derived from it -- the
    // complete iteration above all, which is t(2) - t(1) -- inherits the lie
    // otherwise. Runners that report the counts are held to them.
    let first_clean = match (get("first_attempts"), get("first_accepted")) {
        (Some(a), Some(ok)) => a as usize == 1 && ok as usize == 1,
        _ => true, // runner does not report it; nothing to check against
    };
    let full_ms = match (get("second_run_ms"), get("second_accepted")) {
        (Some(ms), Some(acc)) if acc as usize >= 2 && first_clean => Some(ms),
        _ => None,
    };
    let first_iter_ms = if first_clean {
        get("first_iter_ms").expect("first_iter_ms")
    } else {
        f64::NAN
    };

    Protocol {
        solve_ms: get("solve_ms").expect("solve_ms"),
        first_iter_ms,
        iterations: get("iterations").expect("iterations") as usize,
        accepted: get("accepted").map(|v| v as usize),
        full_ms,
        json,
    }
}
