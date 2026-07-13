// Peak memory.
//
// VmHWM is a high-water mark for the whole PROCESS, so solvers that share one
// contaminate each other's peak: the second solver inherits the first's, and an
// allocator that retains freed pages hides the difference entirely. Each system
// is therefore measured in a process of its own, running the solve alone.

/// Peak resident set of this process, in MB.
pub fn peak_rss_mb() -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").expect("read /proc/self/status");
    let kb: f64 = status
        .lines()
        .find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.split_whitespace().next())
        .and_then(|v| v.parse().ok())
        .expect("no VmHWM");
    kb / 1024.0
}

/// Run this executable again in memory-measurement mode and read back the peak
/// it printed. `env` names the variable the child checks to know which system to
/// run alone.
pub fn measure(env: &str, which: &str, extra: &[(&str, &str)]) -> Option<f64> {
    let exe = std::env::current_exe().ok()?;
    let mut cmd = std::process::Command::new(exe);
    cmd.env(env, which);
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout)
        .ok()?
        .lines()
        .rev()
        .find_map(|l| l.strip_prefix("PEAK_MB "))
        .and_then(|v| v.trim().parse().ok())
}

/// Printed by the child in memory-measurement mode; read by [`measure`].
pub fn report_peak() {
    println!("PEAK_MB {:.1}", peak_rss_mb());
}
