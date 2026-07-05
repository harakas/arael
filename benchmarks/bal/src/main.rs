// BAL bundle-adjustment benchmark: arael vs Ceres, same methodology as
// benchmarks/pgo (see its README):
// - one loader, one reference cost; Ceres's self-reported initial cost
//   is ASSERTED against the reference cost function on every run;
// - validation: cost within 1% of the best AND similarity-aligned
//   relative camera-center RMSE under 5e-3 (bundle adjustment has a
//   7-DOF gauge, BAL units are arbitrary, and landmark positions have
//   near-flat directions -- see bal::camera_centers);
// - total time = min over N interleaved rounds; first-iteration time =
//   a fresh optimize capped at 1 iteration; single core, verified.

mod arael_runner;
mod bal;

use bal::CameraIn;
use arael::vect::vect3d;

struct Cell {
    solve_ms: f64,
    first_iter_ms: f64,
    iterations: usize,
    accepted: Option<usize>,
    cameras: Vec<CameraIn>,
    points: Vec<vect3d>,
    cost: f64,
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

// Run an external subprocess: JSON protocol line on stdout (the
// initial cost under `cost_key` cross-checked against the reference),
// 9-value camera lines followed by 3-value point lines in params_out.
fn run_external(mut cmd: std::process::Command, params_out: &str, cost_key: &str,
                n_cams: usize, n_points: usize, expected_initial: f64)
    -> (f64, f64, usize, Option<usize>, Vec<CameraIn>, Vec<vect3d>) {
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
    (solve_ms, first_iter_ms, iterations, accepted, cameras, points)
}

fn run_ceres(ds_path: &str, linsolver: &str, n_cams: usize, n_points: usize,
             expected_initial: f64)
    -> (f64, f64, usize, Option<usize>, Vec<CameraIn>, Vec<vect3d>) {
    let params_out = format!("/tmp/ceres_bal_{}.txt", linsolver);
    let mut cmd = std::process::Command::new("cpp/build/ceres_bal");
    cmd.args([ds_path, &params_out, linsolver]);
    run_external(cmd, &params_out, "initial_cost", n_cams, n_points, expected_initial)
}

// g2o's proper BA configuration (its own bal_example): marginalized
// point vertices -> Schur elimination, CHOLMOD on the reduced camera
// system. Unit information makes its chi2 the reference cost exactly.
fn run_g2o(ds_path: &str, n_cams: usize, n_points: usize, expected_initial: f64)
    -> (f64, f64, usize, Option<usize>, Vec<CameraIn>, Vec<vect3d>) {
    let params_out = "/tmp/g2o_bal.txt";
    let mut cmd = std::process::Command::new("cpp/build/g2o_bal");
    cmd.args([ds_path, params_out]);
    run_external(cmd, params_out, "initial_chi2", n_cams, n_points, expected_initial)
}

fn main() {
    enforce_single_core();
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
        ("Ladybug-49", "datasets/problem-49-7776-pre.txt", true, "1e-4"),
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
                      accepted: Option<usize>, cameras: Vec<CameraIn>, points: Vec<vect3d>,
                      cells: &mut Vec<(String, Cell)>, ds: &bal::Dataset| {
            let cost = bal::reference_cost(ds, &cameras, &points);
            if let Some((_, prev)) = cells.iter_mut().find(|(l, _)| l == label) {
                prev.solve_ms = prev.solve_ms.min(solve_ms);
                prev.first_iter_ms = prev.first_iter_ms.min(first_iter_ms);
            } else {
                cells.push((label.to_string(),
                    Cell { solve_ms, first_iter_ms, iterations, accepted, cameras, points, cost }));
            }
        };

        for round in 0..rounds {
            let a64 = arael_runner::run_f64(&ds);
            record("arael LM f64", a64.solve_ms, a64.first_iter_ms, a64.iterations,
                Some(a64.accepted), a64.cameras, a64.points, &mut cells, &ds);
            let a32 = arael_runner::run_f32(&ds);
            record("arael LM f32", a32.solve_ms, a32.first_iter_ms, a32.iterations,
                Some(a32.accepted), a32.cameras, a32.points, &mut cells, &ds);
            if ceres_available {
                let solvers: &[&str] = if *dense_schur_ok {
                    &["dense_schur", "sparse_schur"]
                } else {
                    &["sparse_schur"]
                };
                for linsolver in solvers {
                    let (ms, fi, it, acc, cams, pts) =
                        run_ceres(path, linsolver, ds.cameras.len(), ds.points.len(), initial_cost);
                    record(&format!("ceres {}", linsolver), ms, fi, it, acc, cams, pts,
                        &mut cells, &ds);
                }
            }
            if g2o_available {
                let (ms, fi, it, acc, cams, pts) =
                    run_g2o(path, ds.cameras.len(), ds.points.len(), initial_cost);
                record("g2o LM (schur)", ms, fi, it, acc, cams, pts, &mut cells, &ds);
            }
            eprintln!("  round {}/{} done", round + 1, rounds);
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

        println!("\n{:<20} {:>10} {:>9} {:>10} {:>12} {:>16}",
            "system", "total ms", "iters", "ms/iter", "1st-iter ms", "final cost");
        for (label, c) in &cells {
            let iters = match c.accepted {
                Some(a) => format!("{}({})", a, c.iterations),
                None => format!("{}", c.iterations),
            };
            println!("{:<20} {:>10.1} {:>9} {:>10.2} {:>12.1} {:>16.4}{}",
                label, c.solve_ms, iters,
                c.solve_ms / c.iterations.max(1) as f64,
                c.first_iter_ms, c.cost,
                if converged(c) {
                    String::new()
                } else {
                    format!("  <- did not reach the common optimum (aligned rel RMSE {:.2e})",
                        bal::aligned_relative_rmse(&bal::camera_centers(&c.cameras), &best_centers))
                });
        }

        for (label, c) in &cells {
            if label == "arael LM f64" {
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
