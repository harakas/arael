// BAL bundle-adjustment benchmark: arael vs Ceres, same methodology as
// benchmarks/pgo (see its README):
// - one loader, one reference cost; Ceres's self-reported initial cost
//   is ASSERTED against the reference cost function on every run;
// - validation: cost within 1% of the best AND similarity-aligned
//   relative camera-center RMSE under 5e-3 (bundle adjustment has a
//   7-DOF gauge, BAL units are arbitrary, and landmark positions have
//   near-flat directions -- see bal::camera_centers);
// - total time = min over N interleaved rounds; first-iteration time =
//   a fresh optimize capped at 1 iteration; single core, verified;
// - peak MB = VmHWM, arael rows measured in a fresh subprocess per
//   solver (BAL_NO_MEM=1 skips), external rows self-reported;
// - full-it ms = the cost of one FULL accepted iteration, undiluted by
//   rejected damping attempts (which skip the re-linearization and so
//   deflate ms/iter = total/attempts). arael: steady-state per-phase
//   means summed (first calls excluded -- one-time structure costs);
//   ceres: per-phase totals over call counts from its Summary; g2o:
//   mean whole-iteration time over single-lambda-trial iterations
//   after the first.

mod arael_runner;
mod bal;

use bal::CameraIn;
use arael::vect::vect3d;

struct Cell {
    solve_ms: f64,
    first_iter_ms: f64,
    iterations: usize,
    accepted: Option<usize>,
    peak_mb: f64,
    full_iter_ms: f64,
    cameras: Vec<CameraIn>,
    points: Vec<vect3d>,
    cost: f64,
}

fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status").unwrap_or_default()
        .lines().find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or(0)
}

// Peak resident memory of one arael solver, measured in a FRESH process
// (BAL_MEMSOLVER selects which; the process runs only that solver capped
// at a few iterations and prints its VmHWM). Isolating each solver in
// its own process gives a clean peak -- no allocator retention from a
// previous solver, and the same measurement basis as the external
// subprocesses, which report their own VmHWM.
//
// A cholmod-gpl build links CHOLMOD's BLAS/LAPACK stack, which inflates
// every row's VmHWM by shared-library baseline that a faer-only
// deployment would not carry. BAL_MEM_EXE=<path to a default build>
// sources the non-gpl rows' memory from that clean binary instead; the
// cholmod-gpl row always self-measures (the stack is part of its cost).
fn measure_peak_mb(which: &str, ds_path: &str) -> f64 {
    let exe = match std::env::var("BAL_MEM_EXE") {
        Ok(alt) if which != "arael_gpl" => std::path::PathBuf::from(alt),
        _ => std::env::current_exe().unwrap(),
    };
    let out = std::process::Command::new(exe)
        .env("BAL_MEMSOLVER", which)
        .env("BAL_MEM_DATASET", ds_path)
        .output().unwrap();
    String::from_utf8_lossy(&out.stdout).lines()
        .find_map(|l| l.strip_prefix("PEAK_RSS_KB:"))
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0).unwrap_or(0.0)
}

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

struct ExtOut {
    solve_ms: f64,
    first_iter_ms: f64,
    iterations: usize,
    accepted: Option<usize>,
    peak_mb: f64,
    full_iter_ms: f64,
    cameras: Vec<CameraIn>,
    points: Vec<vect3d>,
}

// Run an external subprocess: JSON protocol line on stdout (the
// initial cost under `cost_key` cross-checked against the reference),
// 9-value camera lines followed by 3-value point lines in params_out.
fn run_external(mut cmd: std::process::Command, params_out: &str, cost_key: &str,
                n_cams: usize, n_points: usize, expected_initial: f64) -> ExtOut {
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
    // 0 (shown as "-") with binaries predating these protocol fields.
    let peak_mb = get("peak_rss_kb").map_or(0.0, |kb| kb / 1024.0);
    let full_iter_ms = get("full_iter_ms").filter(|v| *v > 0.0).unwrap_or(0.0);
    let initial = get(cost_key).unwrap_or_else(|| panic!("missing {} from {:?}", cost_key, cmd));
    assert!(((initial - expected_initial) / expected_initial).abs() < 1e-9,
        "{:?} initial cost {} disagrees with reference {}", cmd, initial, expected_initial);
    let core = std::env::var("PGO_BENCH_CORE").unwrap();
    assert!(line.contains(&format!("\"cpus_allowed\": \"{}\"", core)),
        "{:?} not pinned to CPU {}: {}", cmd, core, line);

    let text = std::fs::read_to_string(params_out).unwrap();
    let mut lines = text.lines();
    let mut cameras = Vec::with_capacity(n_cams);
    for _ in 0..n_cams {
        let v: Vec<f64> = lines.next().unwrap()
            .split_whitespace().map(|t| t.parse().unwrap()).collect();
        cameras.push(CameraIn {
            rodrigues: vect3d::new(v[0], v[1], v[2]),
            t: vect3d::new(v[3], v[4], v[5]),
            f: v[6],
            k1: v[7],
            k2: v[8],
        });
    }
    let mut points = Vec::with_capacity(n_points);
    for _ in 0..n_points {
        let v: Vec<f64> = lines.next().unwrap()
            .split_whitespace().map(|t| t.parse().unwrap()).collect();
        points.push(vect3d::new(v[0], v[1], v[2]));
    }
    ExtOut { solve_ms, first_iter_ms, iterations, accepted, peak_mb, full_iter_ms,
             cameras, points }
}

fn run_ceres(ds_path: &str, linsolver: &str, n_cams: usize, n_points: usize,
             expected_initial: f64) -> ExtOut {
    let params_out = format!("/tmp/ceres_bal_{}.txt", linsolver);
    let mut cmd = std::process::Command::new("cpp/build/ceres_bal");
    cmd.args([ds_path, &params_out, linsolver]);
    run_external(cmd, &params_out, "initial_cost", n_cams, n_points, expected_initial)
}

// g2o's proper BA configuration (its own bal_example): marginalized
// point vertices -> Schur elimination, CHOLMOD on the reduced camera
// system. Unit information makes its chi2 the reference cost exactly.
fn run_g2o(ds_path: &str, n_cams: usize, n_points: usize, expected_initial: f64) -> ExtOut {
    let params_out = "/tmp/g2o_bal.txt";
    let mut cmd = std::process::Command::new("cpp/build/g2o_bal");
    cmd.args([ds_path, params_out]);
    run_external(cmd, params_out, "initial_chi2", n_cams, n_points, expected_initial)
}

fn main() {
    enforce_single_core();

    // Memory-measurement mode: run one solver, print peak RSS, exit.
    if let Ok(which) = std::env::var("BAL_MEMSOLVER") {
        let ds_path = std::env::var("BAL_MEM_DATASET").expect("BAL_MEM_DATASET");
        let iters: usize = std::env::var("BAL_MEM_ITERS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(3);
        let ds = bal::load(&ds_path);
        match which.as_str() {
            "arael_f64" => { std::hint::black_box(arael_runner::run_f64_capped(&ds, iters)); }
            #[cfg(feature = "cholmod-gpl")]
            "arael_gpl" => { std::hint::black_box(arael_runner::run_f64_supernodal_capped(&ds, iters)); }
            "arael_f32" => { std::hint::black_box(arael_runner::run_f32_capped(&ds, iters)); }
            "arael_f64_schur" => { std::hint::black_box(arael_runner::run_f64_schur_capped(&ds, iters)); }
            "arael_f32_schur" => { std::hint::black_box(arael_runner::run_f32_schur_capped(&ds, iters)); }
            other => { eprintln!("unknown BAL_MEMSOLVER {}", other); }
        }
        println!("PEAK_RSS_KB: {}", peak_rss_kb());
        return;
    }

    let bench_dir = std::env::current_dir().unwrap();
    // Only Ladybug-49 is vendored; fetch_datasets.sh downloads the rest.
    // Missing files are skipped with a note. Ceres's dense_schur row is
    // only sensible while the camera count keeps the dense Schur
    // complement small (its own guidance: a few hundred cameras).
    // Ladybug-1723 is exploratory (BAL_ONLY=1723): with the shared 1e-5
    // tolerances neither system actually converges there -- they stop at
    // plateaus 1.9% apart (arael's the lower one), so the mutual
    // validation gate cannot pass; making the big problem a fair race
    // needs a tighter shared termination criterion first.
    // The fourth column is arael's initial lambda for the dataset --
    // per-problem, like every damping knob in these benchmarks (see the
    // README's damping section; ARAEL_LAMBDA0 overrides): 1e-3 on
    // Ladybug-138 (1e-4 plateaus 1.4% high there), 1e-4 elsewhere.
    let datasets = [
        ("Ladybug-49", "datasets/problem-49-7776-pre.txt", true, "1e-5"),
        ("Ladybug-138", "datasets/problem-138-19878-pre.txt", true, "1e-3"),
        ("Ladybug-372", "datasets/problem-372-47423-pre.txt", true, "1e-4"),
    ];
    let datasets_exploratory = [
        ("Ladybug-1723", "datasets/problem-1723-156502-pre.txt", false, "1e-4"),
    ];
    let only = std::env::var("BAL_ONLY").ok();
    let datasets: Vec<_> = datasets.iter()
        .chain(datasets_exploratory.iter().filter(|(n, _, _, _)| {
            only.as_deref().map_or(false, |f| n.contains(f))
        }))
        .collect();
    let user_lambda0 = std::env::var("ARAEL_LAMBDA0").ok();
    let rounds: usize = std::env::var("ROUNDS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(5);
    let ceres_available = std::path::Path::new("cpp/build/ceres_bal").exists();
    if !ceres_available {
        eprintln!("WARNING: cpp/build/ceres_bal missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping ceres rows");
    }
    let g2o_available = std::path::Path::new("cpp/build/g2o_bal").exists();
    if !g2o_available {
        eprintln!("WARNING: cpp/build/g2o_bal missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping g2o rows");
    }

    for (name, rel_path, dense_schur_ok, lambda0) in &datasets {
        if user_lambda0.is_none() {
            std::env::set_var("ARAEL_LAMBDA0", lambda0);
        }
        let path_buf = bench_dir.join(rel_path);
        let path = path_buf.to_str().unwrap();
        if !path_buf.exists() {
            eprintln!("NOTE: {} missing ({}); run ./fetch_datasets.sh", name, rel_path);
            continue;
        }
        if let Some(f) = only.as_deref() {
            if !name.contains(f) {
                continue;
            }
        }
        let ds = bal::load(path);
        println!("\n=== {} : {} cameras, {} points, {} observations, {} parameters ===",
            name, ds.cameras.len(), ds.points.len(), ds.observations.len(),
            ds.cameras.len() * 9 + ds.points.len() * 3);
        let initial_cost = bal::reference_cost(&ds, &ds.cameras, &ds.points);
        println!("initial reference cost: {:.3}", initial_cost);

        let mut cells: Vec<(String, Cell)> = Vec::new();
        let record = |label: &str, solve_ms: f64, first_iter_ms: f64, iterations: usize,
                      accepted: Option<usize>, peak_mb: f64, full_iter_ms: f64,
                      cameras: Vec<CameraIn>, points: Vec<vect3d>,
                      cells: &mut Vec<(String, Cell)>, ds: &bal::Dataset| {
            let min_pos = |prev: f64, new: f64| {
                if new <= 0.0 { prev } else if prev > 0.0 { prev.min(new) } else { new }
            };
            let cost = bal::reference_cost(ds, &cameras, &points);
            if let Some((_, prev)) = cells.iter_mut().find(|(l, _)| l == label) {
                prev.solve_ms = prev.solve_ms.min(solve_ms);
                prev.first_iter_ms = prev.first_iter_ms.min(first_iter_ms);
                prev.peak_mb = min_pos(prev.peak_mb, peak_mb);
                prev.full_iter_ms = min_pos(prev.full_iter_ms, full_iter_ms);
            } else {
                cells.push((label.to_string(),
                    Cell { solve_ms, first_iter_ms, iterations, accepted, peak_mb,
                           full_iter_ms, cameras, points, cost }));
            }
        };

        for round in 0..rounds {
            let a64 = arael_runner::run_f64(&ds);
            record("arael LM f64 sparse", a64.solve_ms, a64.first_iter_ms, a64.iterations,
                Some(a64.accepted), 0.0, a64.full_iter_ms, a64.cameras, a64.points,
                &mut cells, &ds);
            #[cfg(feature = "cholmod-gpl")]
            {
                let ag = arael_runner::run_f64_supernodal(&ds);
                record("arael LM f64 cholmod-gpl", ag.solve_ms, ag.first_iter_ms, ag.iterations,
                    Some(ag.accepted), 0.0, ag.full_iter_ms, ag.cameras, ag.points,
                    &mut cells, &ds);
            }
            let a32 = arael_runner::run_f32(&ds);
            record("arael LM f32 sparse", a32.solve_ms, a32.first_iter_ms, a32.iterations,
                Some(a32.accepted), 0.0, a32.full_iter_ms, a32.cameras, a32.points,
                &mut cells, &ds);
            let q64 = arael_runner::run_f64_schur(&ds);
            record("arael LM f64 schur", q64.solve_ms, q64.first_iter_ms, q64.iterations,
                Some(q64.accepted), 0.0, q64.full_iter_ms, q64.cameras, q64.points,
                &mut cells, &ds);
            let q32 = arael_runner::run_f32_schur(&ds);
            record("arael LM f32 schur", q32.solve_ms, q32.first_iter_ms, q32.iterations,
                Some(q32.accepted), 0.0, q32.full_iter_ms, q32.cameras, q32.points,
                &mut cells, &ds);
            if ceres_available {
                let solvers: &[&str] = if *dense_schur_ok {
                    &["dense_schur", "sparse_schur", "iterative_schur"]
                } else {
                    &["sparse_schur", "iterative_schur"]
                };
                for linsolver in solvers {
                    let e = run_ceres(path, linsolver, ds.cameras.len(), ds.points.len(), initial_cost);
                    record(&format!("ceres {}", linsolver), e.solve_ms, e.first_iter_ms,
                        e.iterations, e.accepted, e.peak_mb, e.full_iter_ms,
                        e.cameras, e.points, &mut cells, &ds);
                }
            }
            if g2o_available {
                let e = run_g2o(path, ds.cameras.len(), ds.points.len(), initial_cost);
                record("g2o LM (schur)", e.solve_ms, e.first_iter_ms, e.iterations,
                    e.accepted, e.peak_mb, e.full_iter_ms, e.cameras, e.points,
                    &mut cells, &ds);
            }
            eprintln!("  round {}/{} done", round + 1, rounds);
        }

        // Peak memory for the arael rows (fresh subprocess per solver; the
        // external rows self-report). BAL_NO_MEM=1 skips the re-solves.
        let measure_mem = std::env::var("BAL_NO_MEM").map_or(true, |v| v != "1");
        if measure_mem {
            let mem_key = |label: &str| -> Option<&'static str> {
                match label {
                    "arael LM f64 sparse" => Some("arael_f64"),
                    "arael LM f64 cholmod-gpl" => Some("arael_gpl"),
                    "arael LM f32 sparse" => Some("arael_f32"),
                    "arael LM f64 schur" => Some("arael_f64_schur"),
                    "arael LM f32 schur" => Some("arael_f32_schur"),
                    _ => None,
                }
            };
            for (label, c) in cells.iter_mut() {
                if let Some(key) = mem_key(label) {
                    c.peak_mb = measure_peak_mb(key, path);
                }
            }
        }

        let best_idx = (0..cells.len())
            .min_by(|&i, &j| cells[i].1.cost.partial_cmp(&cells[j].1.cost).unwrap())
            .unwrap();
        let best = cells[best_idx].1.cost;
        let best_centers = bal::camera_centers(&cells[best_idx].1.cameras);
        // Geometric gate at 5e-3 of scene extent: converged solutions
        // scatter up to ~1.3e-3 around the deepest stop while agreeing
        // in cost to 0.13% (the BA valley is that flat), whereas
        // measured non-converged plateaus sit at 5.8e-3 and beyond --
        // and fail the 1% cost gate anyway.
        let converged = |c: &Cell| {
            (c.cost - best) / best < 1e-2
                && bal::aligned_relative_rmse(&bal::camera_centers(&c.cameras), &best_centers) < 5e-3
        };

        println!("\n{:<26} {:>10} {:>9} {:>10} {:>11} {:>12} {:>8} {:>16}",
            "system", "total ms", "iters", "ms/iter", "full-it ms", "1st-iter ms", "peak MB", "final cost");
        for (label, c) in &cells {
            let iters = match c.accepted {
                Some(a) => format!("{}({})", a, c.iterations),
                None => format!("{}", c.iterations),
            };
            let mem = if c.peak_mb > 0.0 { format!("{:.1}", c.peak_mb) } else { "-".to_string() };
            let full_it = if c.full_iter_ms > 0.0 { format!("{:.2}", c.full_iter_ms) } else { "-".to_string() };
            println!("{:<26} {:>10.1} {:>9} {:>10.2} {:>11} {:>12.1} {:>8} {:>16.4}{}",
                label, c.solve_ms, iters,
                c.solve_ms / c.iterations.max(1) as f64,
                full_it, c.first_iter_ms, mem, c.cost,
                if converged(c) {
                    String::new()
                } else {
                    format!("  <- did not reach the common optimum (aligned rel RMSE {:.2e})",
                        bal::aligned_relative_rmse(&bal::camera_centers(&c.cameras), &best_centers))
                });
        }

        for (label, c) in &cells {
            if label == "arael LM f64 sparse" {
                assert!(converged(c), "{} failed to converge: {} vs best {}", label, c.cost, best);
            }
        }
        let external_agree = cells.iter()
            .filter(|(l, c)| !l.starts_with("arael") && converged(c))
            .count();
        assert!(external_agree >= 1,
            "no external system confirms the best cost {} -- cannot validate", best);
        let conv = cells.iter().filter(|(_, c)| converged(c)).count();
        println!("validation: {}/{} systems at the common optimum ({:.4}: cost within 1%, \
                  similarity-aligned relative camera-center RMSE < 5e-3), anchored by {} external system(s)",
            conv, cells.len(), best, external_agree);
    }
}
