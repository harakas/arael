// 2D pose-graph benchmark: arael vs tiny-solver vs GTSAM.
//
// Methodology (see README.md):
// - identical weighted cost for every system, validated by evaluating one
//   reference cost function on every system's final poses (hard asserts);
// - total time = min over N interleaved rounds; iterations from the run
//   itself; first-iteration time = a fresh optimize capped at 1 iteration;
// - all systems stop on the same criterion class (abs/rel improvement
//   below 1e-5 -- the tiny-solver and GTSAM defaults).

mod arael_runner;
mod g2o;
mod tiny_runner;

use g2o::PoseIn;

struct Cell {
    solve_ms: f64,
    first_iter_ms: f64,
    iterations: usize,
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

fn run_gtsam(python: &str, ds_path: &str, kind: &str, weighted: bool, n_poses: usize) -> (f64, f64, usize, Vec<PoseIn>) {
    let poses_out = format!("/tmp/gtsam_poses_{}.txt", kind);
    let weights = if weighted { "info" } else { "unit" };
    let out = std::process::Command::new(python)
        .args(["gtsam_bench.py", ds_path, kind, &poses_out, weights])
        .output()
        .expect("failed to run gtsam_bench.py");
    assert!(out.status.success(), "gtsam bench failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let line = text.lines().last().unwrap();
    let get = |key: &str| -> f64 {
        let i = line.find(key).unwrap_or_else(|| panic!("missing {} in {}", key, line));
        let rest = &line[i + key.len() + 2..];
        rest.trim_start_matches(':').trim()
            .split(|c: char| c == ',' || c == '}')
            .next().unwrap().trim().parse().unwrap()
    };
    let solve_ms = get("solve_ms");
    let first_iter_ms = get("first_iter_ms");
    let iterations = get("iterations") as usize;
    // Active check: the GTSAM subprocess must have inherited the pin.
    let core = std::env::var("PGO_BENCH_CORE").unwrap();
    assert!(line.contains(&format!("\"cpus_allowed\": \"{}\"", core)),
        "gtsam subprocess not pinned to CPU {}: {}", core, line);
    let poses: Vec<PoseIn> = std::fs::read_to_string(&poses_out).unwrap()
        .lines()
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            PoseIn { x: v[0], y: v[1], th: v[2] }
        })
        .collect();
    assert_eq!(poses.len(), n_poses);
    (solve_ms, first_iter_ms, iterations, poses)
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
    let python = gtsam_available();
    if python.is_none() {
        eprintln!("WARNING: GTSAM python not found (set GTSAM_PYTHON); skipping GTSAM rows");
    }

    for (name, path, weighted) in &datasets {
        let path = path.to_str().unwrap();
        let ds = g2o::load(path, !weighted);
        println!("\n=== {} : {} poses, {} edges, {} parameters ===",
            name, ds.poses.len(), ds.edges.len(), ds.poses.len() * 3);
        let initial_cost = g2o::reference_cost(&ds, &ds.poses);
        println!("initial reference cost: {:.3}", initial_cost);

        // One measured cell per system; times are min-of-N interleaved.
        let mut cells: Vec<(String, Cell)> = Vec::new();
        let record = |label: &str, out: (f64, f64, usize, Vec<PoseIn>), cells: &mut Vec<(String, Cell)>| {
            let (solve_ms, first_iter_ms, iterations, poses) = out;
            let cost = g2o::reference_cost(&ds, &poses);
            if let Some((_, prev)) = cells.iter_mut().find(|(l, _)| l == label) {
                prev.solve_ms = prev.solve_ms.min(solve_ms);
                prev.first_iter_ms = prev.first_iter_ms.min(first_iter_ms);
            } else {
                cells.push((label.to_string(),
                    Cell { solve_ms, first_iter_ms, iterations, poses, cost }));
            }
        };

        for round in 0..rounds {
            let a64 = arael_runner::run_f64(&ds);
            record("arael LM f64", (a64.solve_ms, a64.first_iter_ms, a64.iterations, a64.poses), &mut cells);
            let a32 = arael_runner::run_f32(&ds);
            record("arael LM f32", (a32.solve_ms, a32.first_iter_ms, a32.iterations, a32.poses), &mut cells);
            let tgn = tiny_runner::run_gn(&ds);
            record("tiny-solver GN", (tgn.solve_ms, tgn.first_iter_ms, tgn.iterations, tgn.poses), &mut cells);
            let tlm = tiny_runner::run_lm(&ds);
            record("tiny-solver LM", (tlm.solve_ms, tlm.first_iter_ms, tlm.iterations, tlm.poses), &mut cells);
            if let Some(py) = &python {
                record("gtsam LM", run_gtsam(py, path, "lm", *weighted, ds.poses.len()), &mut cells);
                record("gtsam GN", run_gtsam(py, path, "gn", *weighted, ds.poses.len()), &mut cells);
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

        // Convergence is judged from the measured cost itself: a row is
        // "converged" when it is within 1% of the best cost any system
        // reached. Failures to reach the common optimum are real,
        // reportable solver behavior (see README), not benchmark errors --
        // but arael rows must always converge, and at least one external
        // system must agree with the best cost so correctness is anchored
        // by an independent implementation.
        let best = cells.iter().map(|(_, c)| c.cost).fold(f64::MAX, f64::min);
        let converged = |c: &Cell| (c.cost - best) / best < 1e-2;

        println!("\n{:<18} {:>10} {:>6} {:>10} {:>12} {:>14}",
            "system", "total ms", "iters", "ms/iter", "1st-iter ms", "final cost");
        for (label, c) in &cells {
            println!("{:<18} {:>10.1} {:>6} {:>10.2} {:>12.1} {:>14.4}{}",
                label, c.solve_ms, c.iterations,
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
        let conv: Vec<&(String, Cell)> = cells.iter().filter(|(_, c)| converged(c)).collect();
        for i in 1..conv.len() {
            let rmse = g2o::aligned_rmse(&conv[0].1.poses, &conv[i].1.poses);
            assert!(rmse < 0.05,
                "{} vs {}: aligned RMSE {:.4} m too large",
                conv[0].0, conv[i].0, rmse);
        }
        println!("validation: {}/{} systems at the common optimum ({:.4}), \
                  agreement anchored by {} external system(s), pairwise aligned RMSE < 5 cm",
            conv.len(), cells.len(), best, external_agree);
    }
}
