// Covariance-scaling benchmark glue shared by bal, slam and loc: parse the
// machine-readable COV lines a C++ runner prints, and print the aligned scaling
// table. Each benchmark supplies its own entities (cameras/points, poses/
// landmarks) and its arael measurements; only the parsing and layout are shared.

use std::collections::HashMap;
use std::process::Command;

/// A spread of `n` indices over `[base, base+count)`: evenly sampled, or all of
/// them when `n >= count`. The C++ runners use the same formula (`bench::spread`)
/// so every method queries the same entities.
pub fn spread(base: usize, count: usize, n: usize) -> Vec<usize> {
    if n >= count {
        (0..count).map(|i| base + i).collect()
    } else {
        (0..n).map(|i| base + i * count / n).collect()
    }
}

/// The 1, 2, 8, 32 query counts that fit in `count`, plus `count` itself ("all")
/// when `with_all`. The scaling columns.
pub fn query_counts(count: usize, with_all: bool) -> Vec<usize> {
    let mut ns: Vec<usize> = [1usize, 2, 8, 32].into_iter().filter(|&n| n <= count).collect();
    if with_all && !ns.contains(&count) {
        ns.push(count);
    }
    ns
}

/// Per-cell wall-time cap in seconds (`COV_CELL_CAP_S`, default 120). A cell
/// projected or measured to exceed it is left un-run and marked `*` -- nothing
/// waits longer than this.
pub fn cell_cap_s() -> f64 {
    std::env::var("COV_CELL_CAP_S").ok().and_then(|v| v.parse().ok()).unwrap_or(120.0)
}

/// Whether a cell at count `n` is projected to exceed the cap, from the previous
/// completed cell `(prev_n, prev_ms)`. Per-query cost is ~linear in the count, so
/// this scales the last measurement -- conservative for methods (like a sparse QR)
/// whose real cost grows slower than linearly, which is the safe direction.
pub fn projected_too_long(prev: Option<(usize, f64)>, n: usize, cap_s: f64) -> bool {
    matches!(prev, Some((pn, pms)) if pn > 0 && pms * (n as f64 / pn as f64) > cap_s * 1e3)
}

/// Run `measure(n)` for each query count, skipping any cell projected past the
/// cap and flagging any that overran it. `measure` returns `(median_ms, reps)`.
/// A skipped or overrunning cell is returned with `ms = INFINITY` (rendered `*`).
pub fn scale_counts(
    counts: Vec<usize>,
    cap_s: f64,
    mut measure: impl FnMut(usize) -> (f64, usize),
) -> Vec<(usize, f64, usize)> {
    let mut out = Vec::new();
    let mut prev: Option<(usize, f64)> = None;
    for n in counts {
        if projected_too_long(prev, n, cap_s) {
            out.push((n, f64::INFINITY, 0));
            continue;
        }
        let (ms, reps) = measure(n);
        prev = Some((n, ms));
        out.push((n, if ms > cap_s * 1e3 { f64::INFINITY } else { ms }, reps));
    }
    out
}

/// A C++ cov runner's parsed output: `(entity, N) -> (median_ms, reps)` from its
/// `COV <entity> <N> <ms> <reps>` lines, plus the validation std-dev line it
/// wrote to stderr.
pub struct CovCpp {
    pub cells: HashMap<(String, usize), (f64, usize)>,
    pub stddev: Option<String>,
}

impl CovCpp {
    /// The `(median_ms, reps)` cell for an entity at query count `n`, if present.
    pub fn cell(&self, entity: &str, n: usize) -> Option<&(f64, usize)> {
        self.cells.get(&(entity.to_string(), n))
    }
}

/// Run a C++ cov runner and parse its output. `args` is the full argument list
/// (each runner names its own cov subcommand and inputs).
pub fn run_cov_cpp(mut cmd: Command, args: &[&str]) -> CovCpp {
    let out = cmd.args(args).output().expect("failed to run cov runner");
    let mut cells = HashMap::new();
    for line in String::from_utf8_lossy(&out.stdout).lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        if let ["COV", entity, n, ms, reps] = f[..] {
            if let (Ok(n), Ok(reps)) = (n.parse::<usize>(), reps.parse::<usize>()) {
                // "toolong" == skipped or over the cap (rendered `*`).
                let ms = if ms == "toolong" { f64::INFINITY } else { ms.parse().unwrap_or(f64::NAN) };
                cells.insert((entity.to_string(), n), (ms, reps));
            }
        }
    }
    let stderr = String::from_utf8_lossy(&out.stderr);
    let stddev = stderr.lines().find(|l| l.contains("std dev")).map(|s| s.trim().to_string());
    CovCpp { cells, stddev }
}

/// A `median_ms (reps)` cell: `*` past the time cap, `fail` when the method could
/// not compute it, `-` when it does not cover that count.
pub fn fmt_cell(cell: Option<&(f64, usize)>) -> String {
    match cell {
        Some(&(ms, _)) if ms.is_infinite() => "*".to_string(),
        Some(&(ms, reps)) if ms.is_finite() => format!("{ms:.1} ({reps})"),
        Some(_) => "fail".to_string(),
        None => "-".to_string(),
    }
}

/// An aligned scaling table: a header per query count, then one row per method.
pub fn print_table(headers: &[String], rows: &[(&str, Vec<String>)]) {
    print!("    {:<18}", "");
    for h in headers {
        print!("{h:>15}");
    }
    println!();
    for (label, cells) in rows {
        print!("    {label:<18}");
        for c in cells {
            print!("{c:>15}");
        }
        println!();
    }
}
