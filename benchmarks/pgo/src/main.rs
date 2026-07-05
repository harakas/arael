// Pose-graph benchmark (2D SE2 and 3D SE3 datasets): arael vs
// tiny-solver vs GTSAM vs Ceres vs g2o vs factrs vs SymForce.
//
// Methodology (see README.md):
// - identical weighted cost for every system, validated by evaluating one
//   reference cost function on every system's final poses (hard asserts);
// - total time = min over N interleaved rounds; iterations from the run
//   itself; first-iteration time = a fresh optimize capped at 1 iteration;
// - all systems stop on the same criterion class (abs/rel improvement
//   below 1e-5 -- the tiny-solver and GTSAM defaults).

mod arael_runner;
mod arael_runner3;
mod factrs_runner;
mod factrs_runner3;
mod g2o;
mod g2o3;
mod tiny_runner;
mod tiny_runner3;

use g2o::PoseIn;

struct Cell {
    solve_ms: f64,
    first_iter_ms: f64,
    iterations: usize,
    // Accepted steps, for systems that report it (arael). Displayed as
    // "accepted(total)"; total includes damping retries.
    accepted: Option<usize>,
    poses: Vec<PoseIn>,
    cost: f64,
}

// GTSAM runs through its Python wheel: create a venv anywhere, `pip
// install gtsam`, and point GTSAM_PYTHON at its python3 (default:
// ./gtsam-venv/bin/python3 next to this crate). Without it the GTSAM
// rows are skipped with a warning.
fn gtsam_available() -> Option<String> {
    let venv = std::env::var("GTSAM_PYTHON")
        .unwrap_or_else(|_| "gtsam-venv/bin/python3".to_string());
    std::path::Path::new(&venv).exists().then_some(venv)
}

// Run an external benchmark process speaking the shared protocol: JSON
// line {solve_ms, first_iter_ms, iterations, accepted?, cpus_allowed}
// on stdout, "x y theta" lines in poses_out. Asserts the subprocess
// inherited the single-core pin.
fn run_external(mut cmd: std::process::Command, poses_out: &str, n_poses: usize) -> (f64, f64, usize, Option<usize>, Vec<PoseIn>) {
    let out = cmd.output().unwrap_or_else(|e| panic!("failed to run {:?}: {}", cmd, e));
    assert!(out.status.success(), "{:?} failed: {}", cmd, String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let line = text.lines().rev().find(|l| l.contains("solve_ms"))
        .unwrap_or_else(|| panic!("no protocol line from {:?}", cmd));
    let get = |key: &str| -> Option<f64> {
        let i = line.find(key)?;
        let rest = &line[i + key.len() + 2..];
        rest.trim_start_matches(':').trim()
            .split(|c: char| c == ',' || c == '}')
            .next().unwrap().trim().parse().ok()
    };
    let solve_ms = get("solve_ms").expect("solve_ms");
    let first_iter_ms = get("first_iter_ms").expect("first_iter_ms");
    let iterations = get("\"iterations\"").expect("iterations") as usize;
    let accepted = get("accepted").map(|v| v as usize);
    // Active check: the subprocess must have inherited the pin.
    let core = std::env::var("PGO_BENCH_CORE").unwrap();
    assert!(line.contains(&format!("\"cpus_allowed\": \"{}\"", core)),
        "{:?} not pinned to CPU {}: {}", cmd, core, line);
    let poses: Vec<PoseIn> = std::fs::read_to_string(poses_out).unwrap()
        .lines()
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            PoseIn { x: v[0], y: v[1], th: v[2] }
        })
        .collect();
    assert_eq!(poses.len(), n_poses);
    (solve_ms, first_iter_ms, iterations, accepted, poses)
}

fn run_gtsam(python: &str, ds_path: &str, kind: &str, weighted: bool, n_poses: usize) -> (f64, f64, usize, Option<usize>, Vec<PoseIn>) {
    let poses_out = format!("/tmp/gtsam_poses_{}.txt", kind);
    let weights = if weighted { "info" } else { "unit" };
    let mut cmd = std::process::Command::new(python);
    cmd.args(["gtsam_bench.py", ds_path, kind, &poses_out, weights]);
    run_external(cmd, &poses_out, n_poses)
}

fn run_ceres(ds_path: &str, weighted: bool, n_poses: usize) -> (f64, f64, usize, Option<usize>, Vec<PoseIn>) {
    let mut cmd = std::process::Command::new("cpp/build/ceres_bench");
    cmd.args([ds_path, "/tmp/ceres_poses.txt", if weighted { "info" } else { "unit" }]);
    run_external(cmd, "/tmp/ceres_poses.txt", n_poses)
}

// factrs's f32 mode can fail outright (its Cholesky panics on a
// non-positive pivot on city10000 -- single precision loses positive
// definiteness at 30000 parameters and its lambda floor of 1e-10 cannot
// regularize it). A crashed run is a reportable outcome, not a harness
// failure.
fn run_factrs32(ds_path: &str, kind: &str, weighted: bool, n_poses: usize) -> Option<(f64, f64, usize, Option<usize>, Vec<PoseIn>)> {
    let poses_out = format!("/tmp/factrs32_poses_{}.txt", kind);
    let mut probe = std::process::Command::new("factrs32/target/release/factrs32-bench");
    probe.args([ds_path, kind, &poses_out, if weighted { "info" } else { "unit" }]);
    let out = probe.output().ok()?;
    if !out.status.success() {
        return None;
    }
    let mut cmd = std::process::Command::new("factrs32/target/release/factrs32-bench");
    cmd.args([ds_path, kind, &poses_out, if weighted { "info" } else { "unit" }]);
    Some(run_external(cmd, &poses_out, n_poses))
}

fn run_symforce(ds_path: &str, prec: &str, weighted: bool, n_poses: usize) -> (f64, f64, usize, Option<usize>, Vec<PoseIn>) {
    let poses_out = format!("/tmp/symforce_poses_{}.txt", prec);
    let mut cmd = std::process::Command::new("cpp/build/symforce_bench");
    cmd.args([ds_path, prec, &poses_out, if weighted { "info" } else { "unit" }]);
    run_external(cmd, &poses_out, n_poses)
}

fn run_g2o(ds_path: &str, kind: &str, weighted: bool, n_poses: usize) -> (f64, f64, usize, Option<usize>, Vec<PoseIn>) {
    let poses_out = format!("/tmp/g2o_poses_{}.txt", kind);
    let mut cmd = std::process::Command::new("cpp/build/g2o_bench");
    cmd.args([ds_path, kind, &poses_out, if weighted { "info" } else { "unit" }]);
    run_external(cmd, &poses_out, n_poses)
}

// Single-core enforcement, applied before any thread pool can spawn:
// every known threading knob is forced to 1 (OpenMP, BLAS flavors, TBB,
// rayon) and the process is pinned to CPU 0 -- child processes (the GTSAM
// runner) inherit both the environment and the affinity mask, and the
// GTSAM runner reports its allowed-CPU list back so the pin is verified,
// not assumed.
fn enforce_single_core() {
    for var in [
        "RAYON_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TBB_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ] {
        std::env::set_var(var, "1");
    }
    // Pin to the LAST core: core 0 preferentially receives timer ticks
    // and IRQs, so a benchmark pinned there shares its core with kernel
    // housekeeping.
    let last_core = std::thread::available_parallelism().map(|n| n.get() - 1).unwrap_or(0);
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(last_core, &mut set);
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        assert_eq!(rc, 0, "sched_setaffinity failed");
    }
    std::env::set_var("PGO_BENCH_CORE", last_core.to_string());
}

// Run an external 3D benchmark process: same JSON protocol, poses file
// carries "x y z qx qy qz qw" lines. `cost_key` names the JSON field
// where the runner reports the canonical cost of the initial estimate
// as computed by ITS OWN code; `check` asserts it equals the reference
// cost function's value.
fn run_external3_checked(mut cmd: std::process::Command, args: &[&str], poses_out: &str,
                         n_poses: usize, cost_key: Option<&str>, check: &dyn Fn(f64))
    -> (f64, f64, usize, Option<usize>, Vec<g2o3::Pose3In>) {
    cmd.args(args);
    let out = cmd.output().unwrap_or_else(|e| panic!("failed to run {:?}: {}", cmd, e));
    assert!(out.status.success(), "{:?} failed: {}", cmd, String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let line = text.lines().rev().find(|l| l.contains("solve_ms"))
        .unwrap_or_else(|| panic!("no protocol line from {:?}", cmd));
    let get = |key: &str| -> Option<f64> {
        let i = line.find(key)?;
        let rest = &line[i + key.len() + 2..];
        rest.trim_start_matches(':').trim()
            .split(|c: char| c == ',' || c == '}')
            .next().unwrap().trim().parse().ok()
    };
    let solve_ms = get("solve_ms").expect("solve_ms");
    let first_iter_ms = get("first_iter_ms").expect("first_iter_ms");
    let iterations = get("\"iterations\"").expect("iterations") as usize;
    let accepted = get("\"accepted\"").map(|v| v as usize);
    if let Some(key) = cost_key {
        check(get(key).unwrap_or_else(|| panic!("missing {} from {:?}", key, cmd)));
    }
    let core = std::env::var("PGO_BENCH_CORE").unwrap();
    assert!(line.contains(&format!("\"cpus_allowed\": \"{}\"", core)),
        "{:?} not pinned to CPU {}: {}", cmd, core, line);
    let poses: Vec<g2o3::Pose3In> = std::fs::read_to_string(poses_out).unwrap()
        .lines()
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            g2o3::Pose3In {
                t: arael::vect::vect3d::new(v[0], v[1], v[2]),
                q: [v[3], v[4], v[5], v[6]],
            }
        })
        .collect();
    assert_eq!(poses.len(), n_poses);
    (solve_ms, first_iter_ms, iterations, accepted, poses)
}

// 3D (SE3) datasets: arael f64/f32, tiny-solver, factrs, Ceres and g2o;
// the GTSAM and SymForce 3D runners come in a later pass. Same
// methodology: one loader, one reference cost, one validation,
// min-of-N interleaved timing. The g2o and Ceres runners additionally
// report the canonical cost of the initial estimate computed by THEIR
// code (g2o's chi2 with the conjugated information, Ceres's 2x
// initial_cost); the harness asserts both equal the reference cost
// function's value -- a bit-level cross-implementation check of the
// cost every system minimizes.
fn run_dataset3(name: &str, path: &str, rounds: usize, f32_floor_note: Option<&str>) {
    struct Cell3 {
        solve_ms: f64,
        first_iter_ms: f64,
        iterations: usize,
        accepted: Option<usize>,
        poses: Vec<g2o3::Pose3In>,
        cost: f64,
    }
    let ds = g2o3::load3(path);
    println!("\n=== {} : {} poses, {} edges, {} parameters ===",
        name, ds.poses.len(), ds.edges.len(), ds.poses.len() * 6);
    println!("initial reference cost: {:.3}", g2o3::reference_cost3(&ds, &ds.poses));

    let cpp3_available = std::path::Path::new("cpp/build/ceres_bench3").exists()
        && std::path::Path::new("cpp/build/g2o_bench3").exists();
    if !cpp3_available {
        eprintln!("WARNING: cpp/build 3D runners missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping ceres/g2o rows");
    }
    let initial_cost = g2o3::reference_cost3(&ds, &ds.poses);

    let mut cells: Vec<(String, Cell3)> = Vec::new();
    let record = |label: &str, solve_ms: f64, first_iter_ms: f64,
                  iterations: usize, accepted: Option<usize>,
                  poses: Vec<g2o3::Pose3In>, cells: &mut Vec<(String, Cell3)>,
                  ds: &g2o3::Dataset3| {
        let cost = g2o3::reference_cost3(ds, &poses);
        if let Some((_, prev)) = cells.iter_mut().find(|(l, _)| l == label) {
            prev.solve_ms = prev.solve_ms.min(solve_ms);
            prev.first_iter_ms = prev.first_iter_ms.min(first_iter_ms);
        } else {
            cells.push((label.to_string(),
                Cell3 { solve_ms, first_iter_ms, iterations, accepted, poses, cost }));
        }
    };

    for round in 0..rounds {
        let a64 = arael_runner3::run_f64(&ds);
        record("arael LM f64", a64.solve_ms, a64.first_iter_ms, a64.iterations,
            Some(a64.accepted), a64.poses, &mut cells, &ds);
        let a32 = arael_runner3::run_f32(&ds);
        record("arael LM f32", a32.solve_ms, a32.first_iter_ms, a32.iterations,
            Some(a32.accepted), a32.poses, &mut cells, &ds);
        let tgn = tiny_runner3::run_gn(&ds);
        record("tiny-solver GN", tgn.solve_ms, tgn.first_iter_ms, tgn.iterations,
            None, tgn.poses, &mut cells, &ds);
        let tlm = tiny_runner3::run_lm(&ds);
        record("tiny-solver LM", tlm.solve_ms, tlm.first_iter_ms, tlm.iterations,
            None, tlm.poses, &mut cells, &ds);
        let fgn = factrs_runner3::run_gn(&ds);
        record("factrs GN", fgn.solve_ms, fgn.first_iter_ms, fgn.iterations,
            None, fgn.poses, &mut cells, &ds);
        let flm = factrs_runner3::run_lm(&ds);
        record("factrs LM", flm.solve_ms, flm.first_iter_ms, flm.iterations,
            None, flm.poses, &mut cells, &ds);
        let symforce3_available = std::path::Path::new("cpp/build/symforce_bench3").exists();
        if cpp3_available {
            let check = |line_cost: f64| {
                assert!(((line_cost - initial_cost) / initial_cost).abs() < 1e-9,
                    "external initial cost {} disagrees with reference {}",
                    line_cost, initial_cost);
            };
            let (ms, fi, it, acc, poses) = run_external3_checked(
                std::process::Command::new("cpp/build/ceres_bench3"),
                &[path, "/tmp/ceres3_poses.txt"], "/tmp/ceres3_poses.txt",
                ds.poses.len(), Some("initial_cost"), &check);
            record("ceres LM", ms, fi, it, acc, poses, &mut cells, &ds);
            for kind in ["lm", "gn"] {
                let poses_out = format!("/tmp/g2o3_poses_{}.txt", kind);
                let (ms, fi, it, acc, poses) = run_external3_checked(
                    std::process::Command::new("cpp/build/g2o_bench3"),
                    &[path, kind, &poses_out], &poses_out,
                    ds.poses.len(), Some("initial_chi2"), &check);
                record(&format!("g2o {}", kind.to_uppercase()), ms, fi, it, acc, poses, &mut cells, &ds);
            }
            if symforce3_available {
                for prec in ["f64", "f32"] {
                    let poses_out = format!("/tmp/symforce3_poses_{}.txt", prec);
                    let (ms, fi, it, acc, poses) = run_external3_checked(
                        std::process::Command::new("cpp/build/symforce_bench3"),
                        &[path, prec, &poses_out], &poses_out,
                        ds.poses.len(), Some("initial_cost"), &check);
                    let label = if prec == "f32" { "symforce LM f32" } else { "symforce LM" };
                    record(label, ms, fi, it, acc, poses, &mut cells, &ds);
                }
            }
        }
        // GTSAM batch rows: its native BetweenFactorPose3 minimizes the
        // full SE3 log-map objective -- the documented deviation from the
        // canonical residual (second-order small at noise-level
        // residuals), so no initial-cost cross-check applies; the
        // validation gates judge its solutions like everyone else's.
        if let Some(py) = gtsam_available() {
            for kind in ["lm", "gn"] {
                let poses_out = format!("/tmp/gtsam3_poses_{}.txt", kind);
                let (ms, fi, it, acc, poses) = run_external3_checked(
                    std::process::Command::new(&py),
                    &["gtsam_bench.py", path, kind, &poses_out], &poses_out,
                    ds.poses.len(), None, &|_| {});
                record(&format!("gtsam {}", kind.to_uppercase()), ms, fi, it, acc, poses, &mut cells, &ds);
            }
        }
        eprintln!("  round {}/{} done", round + 1, rounds);
    }

    // Same validation gates as 2D: within 1% of the best cost AND within
    // 5 cm rigid-aligned RMSE of the best solution; arael rows must
    // converge and at least one external system must agree.
    let best_idx = (0..cells.len())
        .min_by(|&i, &j| cells[i].1.cost.partial_cmp(&cells[j].1.cost).unwrap())
        .unwrap();
    let best = cells[best_idx].1.cost;
    let best_poses = cells[best_idx].1.poses.clone();
    let converged = |c: &Cell3| {
        (c.cost - best) / best < 1e-2
            && g2o3::aligned_rmse3(&best_poses, &c.poses) < 0.05
    };

    println!("\n{:<18} {:>10} {:>9} {:>10} {:>12} {:>14}",
        "system", "total ms", "iters", "ms/iter", "1st-iter ms", "final cost");
    for (label, c) in &cells {
        let iters = match c.accepted {
            Some(a) => format!("{}({})", a, c.iterations),
            None => format!("{}", c.iterations),
        };
        println!("{:<18} {:>10.1} {:>9} {:>10.2} {:>12.1} {:>14.4}{}",
            label, c.solve_ms, iters,
            c.solve_ms / c.iterations.max(1) as f64,
            c.first_iter_ms, c.cost,
            if converged(c) {
                String::new()
            } else {
                format!("  <- did not reach the common optimum (aligned RMSE {:.4} m)",
                    g2o3::aligned_rmse3(&best_poses, &c.poses))
            });
    }

    let mut notes: Vec<String> = Vec::new();
    for (label, c) in &cells {
        if label == "arael LM f64" {
            assert!(converged(c), "{} failed to converge: {} vs best {} (aligned RMSE {:.4})",
                label, c.cost, best, g2o3::aligned_rmse3(&best_poses, &c.poses));
        }
        if label == "arael LM f32" && !converged(c) {
            match f32_floor_note {
                Some(n) => notes.push(format!("arael LM f32: {}", n)),
                None => panic!("{} failed to converge: {} vs best {} (aligned RMSE {:.4})",
                    label, c.cost, best, g2o3::aligned_rmse3(&best_poses, &c.poses)),
            }
        }
    }
    let external_agree = cells.iter()
        .filter(|(l, c)| !l.starts_with("arael") && converged(c))
        .count();
    assert!(external_agree >= 1,
        "no external system confirms the best cost {} -- cannot validate", best);
    for note in &notes {
        println!("{:<18} {}", "", note);
    }
    let conv = cells.iter().filter(|(_, c)| converged(c)).count();
    println!("validation: {}/{} systems at the common optimum ({:.4}: cost within 1%, \
              aligned RMSE to best < 5 cm), anchored by {} external system(s)",
        conv, cells.len(), best, external_agree);
}

fn main() {
    enforce_single_core();
    tiny_runner::install_iter_counter();
    let bench_dir = std::env::current_dir().unwrap();
    // (name, file, use the file's information matrices?). The unweighted
    // M3500 row is the configuration tiny-solver's shipped benchmark runs.
    let datasets = [
        ("M3500 unweighted", bench_dir.join("datasets/input_M3500_g2o.g2o"), false),
        ("M3500", bench_dir.join("datasets/input_M3500_g2o.g2o"), true),
        ("city10000", bench_dir.join("datasets/city10000.g2o"), true),
    ];
    let rounds: usize = std::env::var("ROUNDS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(5);
    // PGO_ONLY=<substring> runs only the datasets whose name contains it.
    let only = std::env::var("PGO_ONLY").ok();
    let selected = |name: &str| only.as_deref().map_or(true, |f| name.contains(f));
    let python = gtsam_available();
    if python.is_none() {
        eprintln!("WARNING: GTSAM python not found (set GTSAM_PYTHON); skipping GTSAM rows");
    }
    let cpp_available = std::path::Path::new("cpp/build/ceres_bench").exists()
        && std::path::Path::new("cpp/build/g2o_bench").exists();
    if !cpp_available {
        eprintln!("WARNING: cpp/build runners missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping ceres/g2o rows");
    }
    let factrs32_available = std::path::Path::new("factrs32/target/release/factrs32-bench").exists();
    if !factrs32_available {
        eprintln!("WARNING: factrs32 runner missing (cargo build -r in factrs32/); skipping factrs f32 rows");
    }
    let symforce_available = std::path::Path::new("cpp/build/symforce_bench").exists();
    if !symforce_available {
        eprintln!("WARNING: symforce runner missing (build cpp with -DSYMFORCE_DIR=<built symforce checkout>); skipping symforce rows");
    }

    for (name, path, weighted) in &datasets {
        if !selected(name) {
            continue;
        }
        let path = path.to_str().unwrap();
        let ds = g2o::load(path, !weighted);
        println!("\n=== {} : {} poses, {} edges, {} parameters ===",
            name, ds.poses.len(), ds.edges.len(), ds.poses.len() * 3);
        let initial_cost = g2o::reference_cost(&ds, &ds.poses);
        println!("initial reference cost: {:.3}", initial_cost);

        // One measured cell per system; times are min-of-N interleaved.
        let mut cells: Vec<(String, Cell)> = Vec::new();
        let mut failed_notes: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let record = |label: &str, out: (f64, f64, usize, Option<usize>, Vec<PoseIn>), cells: &mut Vec<(String, Cell)>| {
            let (solve_ms, first_iter_ms, iterations, accepted, poses) = out;
            let cost = g2o::reference_cost(&ds, &poses);
            if let Some((_, prev)) = cells.iter_mut().find(|(l, _)| l == label) {
                prev.solve_ms = prev.solve_ms.min(solve_ms);
                prev.first_iter_ms = prev.first_iter_ms.min(first_iter_ms);
            } else {
                cells.push((label.to_string(),
                    Cell { solve_ms, first_iter_ms, iterations, accepted, poses, cost }));
            }
        };

        for round in 0..rounds {
            let a64 = arael_runner::run_f64(&ds);
            record("arael LM f64", (a64.solve_ms, a64.first_iter_ms, a64.iterations, Some(a64.accepted), a64.poses), &mut cells);
            let a32 = arael_runner::run_f32(&ds);
            record("arael LM f32", (a32.solve_ms, a32.first_iter_ms, a32.iterations, Some(a32.accepted), a32.poses), &mut cells);
            let tgn = tiny_runner::run_gn(&ds);
            record("tiny-solver GN", (tgn.solve_ms, tgn.first_iter_ms, tgn.iterations, None, tgn.poses), &mut cells);
            let tlm = tiny_runner::run_lm(&ds);
            record("tiny-solver LM", (tlm.solve_ms, tlm.first_iter_ms, tlm.iterations, None, tlm.poses), &mut cells);
            let fgn = factrs_runner::run_gn(&ds);
            record("factrs GN", (fgn.solve_ms, fgn.first_iter_ms, fgn.iterations, None, fgn.poses), &mut cells);
            let flm = factrs_runner::run_lm(&ds);
            record("factrs LM", (flm.solve_ms, flm.first_iter_ms, flm.iterations, None, flm.poses), &mut cells);
            if factrs32_available {
                match run_factrs32(path, "gn", *weighted, ds.poses.len()) {
                    Some(out) => record("factrs GN f32", out, &mut cells),
                    None => {
                        failed_notes.insert(
                            "factrs GN f32: solver crashed (f32 Cholesky non-positive pivot)".to_string());
                    }
                }
            }
            if let Some(py) = &python {
                record("gtsam LM", run_gtsam(py, path, "lm", *weighted, ds.poses.len()), &mut cells);
                record("gtsam GN", run_gtsam(py, path, "gn", *weighted, ds.poses.len()), &mut cells);
            }
            if symforce_available {
                record("symforce LM", run_symforce(path, "f64", *weighted, ds.poses.len()), &mut cells);
                record("symforce LM f32", run_symforce(path, "f32", *weighted, ds.poses.len()), &mut cells);
            }
            if cpp_available {
                record("ceres LM", run_ceres(path, *weighted, ds.poses.len()), &mut cells);
                record("g2o LM", run_g2o(path, "lm", *weighted, ds.poses.len()), &mut cells);
                record("g2o GN", run_g2o(path, "gn", *weighted, ds.poses.len()), &mut cells);
            }
            eprintln!("  round {}/{} done", round + 1, rounds);
        }

        // Incremental reference row: GTSAM driven the way its own
        // city10000 example works (one ISAM2 update per pose). A
        // different algorithm answering the online-estimation question;
        // listed for context, timed once, "iters" = incremental updates.
        if let Some(py) = &python {
            record("gtsam ISAM2 (incr)", run_gtsam(py, path, "isam2", *weighted, ds.poses.len()), &mut cells);
        }

        // Convergence is judged against the best solution by BOTH cost
        // (within 1%) and geometry (rigid-aligned RMSE under 5 cm to the
        // best solution) -- the cost surface has near-flat directions
        // where a plateau 0.9% above the optimum can sit meters away
        // geometrically (observed with g2o LM on the weighted M3500).
        // Failures to reach the common optimum are real, reportable
        // solver behavior (see README), not benchmark errors -- but
        // arael rows must always converge, and at least one external
        // system must agree so correctness is anchored by an
        // independent implementation.
        let best_idx = (0..cells.len())
            .min_by(|&i, &j| cells[i].1.cost.partial_cmp(&cells[j].1.cost).unwrap())
            .unwrap();
        let best = cells[best_idx].1.cost;
        let best_poses = cells[best_idx].1.poses.clone();
        let converged = |c: &Cell| {
            (c.cost - best) / best < 1e-2
                && g2o::aligned_rmse(&best_poses, &c.poses) < 0.05
        };

        // iters column: "accepted(total)" where the system reports both;
        // total includes damping retries. Other systems report only their
        // outer iteration count.
        println!("\n{:<18} {:>10} {:>9} {:>10} {:>12} {:>14}",
            "system", "total ms", "iters", "ms/iter", "1st-iter ms", "final cost");
        for (label, c) in &cells {
            let iters = match c.accepted {
                Some(a) => format!("{}({})", a, c.iterations),
                None => format!("{}", c.iterations),
            };
            println!("{:<18} {:>10.1} {:>9} {:>10.2} {:>12.1} {:>14.4}{}",
                label, c.solve_ms, iters,
                c.solve_ms / c.iterations.max(1) as f64,
                c.first_iter_ms, c.cost,
                if converged(c) { "" } else { "  <- did not reach the common optimum" });
        }

        for (label, c) in &cells {
            if label.starts_with("arael") {
                assert!(converged(c), "{} failed to converge: {} vs best {}", label, c.cost, best);
            }
        }
        let external_agree = cells.iter()
            .filter(|(l, c)| !l.starts_with("arael") && converged(c))
            .count();
        assert!(external_agree >= 1,
            "no external system confirms the best cost {} -- cannot validate", best);
        for note in &failed_notes {
            println!("{:<18} {}", "", note);
        }
        let conv = cells.iter().filter(|(_, c)| converged(c)).count();
        println!("validation: {}/{} systems at the common optimum ({:.4}: cost within 1%, \
                  aligned RMSE to best < 5 cm), anchored by {} external system(s)",
            conv, cells.len(), best, external_agree);
    }

    // 3D (SE3) datasets, weighted with the files' full 6x6 information.
    // The garage's f32 note is a measured limitation, not a guess: with
    // the tolerance stop disabled the f32 solve plateaus at the same
    // cost after 22 accepted steps, 0.2 m from the f64 optimum -- the
    // per-step cost decrease is below single-precision evaluation noise
    // (optimum cost 1.27 spread over 6275 edges).
    let datasets3 = [
        ("sphere2500", bench_dir.join("datasets/sphere2500.g2o"), None),
        ("parking-garage", bench_dir.join("datasets/parking-garage.g2o"),
            Some("stops 0.2-0.3 m short along the near-flat directions; the cost-decrease \
                  signal is below f32 evaluation noise on this dataset (verified by probe; \
                  SymForce's f32 lands on the same floor)")),
    ];
    for (name, path, f32_note) in &datasets3 {
        if !selected(name) {
            continue;
        }
        run_dataset3(name, path.to_str().unwrap(), rounds, *f32_note);
    }
}
