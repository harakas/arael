// Localization benchmark: a trajectory estimated against a fixed landmark
// map (bearings + odometry + drift/tilt priors), arael vs the field. Same
// methodology as benchmarks/slam and benchmarks/pgo: one scene generator,
// one reference cost, cross-checked initial cost, single-core pinning,
// min-of-N interleaved timing. LOC_POSES=N sets the pose count (landmarks
// scale as 4N).

mod arael_runner;
mod factrs_runner;
mod scene;
mod tiny_runner;

use scene::{Scene, SceneConfig, Solution};

// Every system's initial cost is cross-checked against the one reference
// cost function, all at this shared tolerance. 1e-5 accommodates arael's
// fast_atan bearing residuals (max error < 1e-6 rad, measured ~4e-7
// relative on the cost); a real model mismatch shows up orders above it
// for any system.
const INITIAL_COST_RTOL: f64 = 1e-5;

fn enforce_single_core() {
    for var in ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "TBB_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"] {
        std::env::set_var(var, "1");
    }
    let core = std::thread::available_parallelism().map(|n| n.get() - 1).unwrap_or(0);
    std::env::set_var("LOC_CORE", core.to_string());
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        libc::CPU_SET(core, &mut set);
        let rc = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        assert_eq!(rc, 0, "sched_setaffinity failed");
    }
}

fn config() -> SceneConfig {
    let mut cfg = SceneConfig::default();
    if let Ok(n) = std::env::var("LOC_POSES") {
        if let Ok(n) = n.parse() {
            cfg.num_poses = n;
            cfg.num_landmarks = 4 * n;
        }
    }
    cfg
}

fn peak_rss_kb() -> u64 {
    std::fs::read_to_string("/proc/self/status").unwrap_or_default()
        .lines().find_map(|l| l.strip_prefix("VmHWM:"))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or(0)
}

// Peak resident memory of one solver, measured in a FRESH process
// (LOC_MEMSOLVER selects which; the process runs only that solver and
// prints its VmHWM). Isolating each solver gives a clean peak.
fn measure_peak_mb(which: &str, poses: usize) -> f64 {
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .env("LOC_MEMSOLVER", which)
        .env("LOC_POSES", poses.to_string())
        .env("LOC_MEM_ITERS", "3")
        .output().unwrap();
    String::from_utf8_lossy(&out.stdout).lines()
        .find_map(|l| l.strip_prefix("PEAK_RSS_KB:"))
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0).unwrap_or(0.0)
}

fn pose_rmse(a: &Solution, b: &Solution) -> f64 {
    let n = a.poses.len() as f64;
    let s: f64 = a.poses.iter().zip(&b.poses)
        .map(|((pa, _), (pb, _))| (*pa - *pb).square()).sum();
    (s / n).sqrt()
}

fn initial_solution(scene: &Scene) -> Solution {
    Solution {
        poses: scene.poses.iter()
            .map(|p| (arael::vect::vect3d::from(p.init_pos), arael::vect::vect3d::from(p.init_ea)))
            .collect(),
    }
}

fn main() {
    enforce_single_core();
    let cfg = config();
    let scene = scene::generate(&cfg);

    // Memory-measurement mode: run one solver, print peak RSS, exit.
    if let Ok(which) = std::env::var("LOC_MEMSOLVER") {
        let iters: usize = std::env::var("LOC_MEM_ITERS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(200);
        match which.as_str() {
            "arael_f64" => { std::hint::black_box(arael_runner::run_capped(&scene, iters)); }
            "arael_f32" => { std::hint::black_box(arael_runner::run_f32_capped(&scene, iters)); }
            "tiny" => {
                std::env::set_var("TINY_MAXITER", iters.to_string());
                tiny_runner::install_iter_counter();
                std::hint::black_box(tiny_runner::run_lm(&scene).solution);
            }
            "factrs" => { std::hint::black_box(factrs_runner::run(&scene).solution); }
            other => { eprintln!("unknown LOC_MEMSOLVER {}", other); }
        }
        println!("PEAK_RSS_KB: {}", peak_rss_kb());
        return;
    }

    // Phase-timing mode: interleaved f64/f32 arael solves with per-phase
    // instrumentation (LmConfig::gather_timing), steady-state means with
    // each phase's first call excluded. For chasing where a precision
    // does or does not speed up.
    if std::env::var("LOC_TIMING").map_or(false, |v| v == "1") {
        let rounds: usize = std::env::var("ROUNDS").ok()
            .and_then(|v| v.parse().ok()).unwrap_or(100);
        let mut t64: Vec<arael::simple_lm::LmTiming> = Vec::new();
        let mut t32: Vec<arael::simple_lm::LmTiming> = Vec::new();
        for _ in 0..rounds {
            t64.push(arael_runner::run_timed_once(&scene));
            t32.push(arael_runner::run_timed_once_f32(&scene));
        }
        let steady = |ts: &[arael::simple_lm::LmTiming]| -> [(f64, usize, f64); 4] {
            // (steady mean us, steady samples, mean first us) per phase
            let mut out = [(0.0, 0usize, 0.0); 4];
            let phases = |t: &arael::simple_lm::LmTiming| [
                (t.assembly, t.first_assembly, t.assembly_count),
                (t.linear_solve, t.first_linear_solve, t.linear_solve_count),
                (t.cost_eval, t.first_cost_eval, t.cost_eval_count),
                (t.advance, t.first_advance, t.advance_count),
            ];
            for k in 0..4 {
                let (mut sum_us, mut n, mut first_us) = (0.0, 0usize, 0.0);
                for t in ts {
                    let (total, first, count) = phases(t)[k];
                    if count >= 1 { first_us += first.as_secs_f64() * 1e6; }
                    if count >= 2 {
                        sum_us += (total - first).as_secs_f64() * 1e6;
                        n += count - 1;
                    }
                }
                out[k] = (if n > 0 { sum_us / n as f64 } else { 0.0 }, n,
                          first_us / ts.len() as f64);
            }
            out
        };
        let s64 = steady(&t64);
        let s32 = steady(&t32);
        println!("phase timing, {} poses, {} interleaved rounds per precision",
            scene.poses.len(), rounds);
        println!("config: {}", arael_runner::config_report(scene.poses.len()));
        println!("(steady-state mean per call, us; first call of each phase excluded and shown apart)");
        println!("{:<14} {:>12} {:>12} {:>9} {:>14} {:>14}",
            "phase", "f64 us", "f32 us", "f32/f64", "f64 1st-call", "f32 1st-call");
        let names = ["assembly", "linear solve", "cost eval", "advance"];
        for k in 0..4 {
            println!("{:<14} {:>12.2} {:>12.2} {:>9.3} {:>14.2} {:>14.2}",
                names[k], s64[k].0, s32[k].0,
                if s64[k].0 > 0.0 { s32[k].0 / s64[k].0 } else { 0.0 },
                s64[k].2, s32[k].2);
        }
        let full = |s: &[(f64, usize, f64); 4]| s.iter().map(|p| p.0).sum::<f64>();
        println!("{:<14} {:>12.2} {:>12.2} {:>9.3}", "full iter",
            full(&s64), full(&s32),
            if full(&s64) > 0.0 { full(&s32) / full(&s64) } else { 0.0 });
        println!("steady samples per phase: f64 {:?}, f32 {:?}",
            s64.iter().map(|p| p.1).collect::<Vec<_>>(),
            s32.iter().map(|p| p.1).collect::<Vec<_>>());
        return;
    }

    let init_sol = initial_solution(&scene);
    let initial_cost = scene::reference_cost(&scene, &init_sol);
    println!("scene: {} poses, {} landmarks (fixed), {} frines, {} odometry pairs, {} parameters",
        scene.poses.len(), scene.landmarks.len(), scene.frines.len(), scene.odo.len(),
        scene.poses.len() * 6);
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    println!("config: ROUNDS={}; {}", rounds, arael_runner::config_report(scene.poses.len()));
    println!("initial reference cost: {:.4}", initial_cost);

    // Cross-check: the arael model must compute the same cost at the
    // initial estimate as the reference function.
    let arael_init = arael_runner::initial_cost(&scene);
    let rel = ((arael_init - initial_cost) / initial_cost).abs();
    assert!(rel < INITIAL_COST_RTOL, "arael initial cost {} vs reference {} (rel {:.2e})",
        arael_init, initial_cost, rel);
    println!("arael initial cost matches reference to {:.2e}", rel);

    let tiny_init = tiny_runner::initial_cost(&scene);
    let rel = ((tiny_init - initial_cost) / initial_cost).abs();
    assert!(rel < INITIAL_COST_RTOL, "tiny initial cost {} vs reference {} (rel {:.2e})",
        tiny_init, initial_cost, rel);
    println!("tiny  initial cost matches reference to {:.2e}", rel);

    let factrs_init = factrs_runner::initial_cost(&scene);
    let rel = ((factrs_init - initial_cost) / initial_cost).abs();
    assert!(rel < INITIAL_COST_RTOL, "factrs initial cost {} vs reference {} (rel {:.2e})",
        factrs_init, initial_cost, rel);
    println!("factrs initial cost matches reference to {:.2e}", rel);

    tiny_runner::install_iter_counter();

    // (label, solve_ms, first_iter_ms, iterations, accepted?, full_iter_ms, solution)
    type Cell = (String, f64, f64, usize, Option<usize>, f64, Solution);
    let mut cells: Vec<Cell> = Vec::new();
    let record = |label: &str, sm: f64, fm: f64, it: usize, acc: Option<usize>,
                  fi: f64, sol: Solution, cells: &mut Vec<Cell>| {
        if let Some(c) = cells.iter_mut().find(|c| c.0 == label) {
            c.1 = c.1.min(sm);
            c.2 = c.2.min(fm);
            if fi > 0.0 { c.5 = if c.5 > 0.0 { c.5.min(fi) } else { fi }; }
        } else {
            cells.push((label.to_string(), sm, fm, it, acc, fi, sol));
        }
    };

    // C++ runners execute as subprocesses over an exported copy of the scene.
    let scene_path = "/tmp/loc_scene.txt";
    scene::write_scene(&scene, scene_path);
    let ceres_ok = std::path::Path::new("cpp/build/ceres_loc").exists();
    if !ceres_ok {
        eprintln!("WARNING: cpp/build/ceres_loc missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping Ceres");
    }
    let ceres_solvers: Vec<String> = std::env::var("CERES_SOLVERS")
        .unwrap_or_else(|_| "sparse_normal_cholesky,sparse_schur,iterative_schur".into())
        .split(',').map(|s| s.to_string()).collect();
    let mut subproc_peaks: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let symforce_ok = std::path::Path::new("cpp/build/symforce_loc").exists();
    if !symforce_ok {
        eprintln!("WARNING: cpp/build/symforce_loc missing (build with -DSYMFORCE_DIR=...); skipping SymForce");
    }
    let g2o_ok = std::path::Path::new("cpp/build/g2o_loc").exists();
    if !g2o_ok {
        eprintln!("WARNING: cpp/build/g2o_loc missing (needs g2o + cholmod); skipping g2o");
    }
    let gtsam_ok = std::path::Path::new("cpp/build/gtsam_loc").exists();
    if !gtsam_ok {
        eprintln!("WARNING: cpp/build/gtsam_loc missing (needs libgtsam-dev); skipping GTSAM");
    }

    let skip_tiny = std::env::var("LOC_SKIP_TINY").map_or(false, |v| v == "1");
    for _ in 0..rounds {
        let a = arael_runner::run(&scene);
        record("arael LM f64", a.solve_ms, a.first_iter_ms, a.iterations, Some(a.accepted), a.full_iter_ms, a.solution, &mut cells);
        let a32 = arael_runner::run_f32(&scene);
        record("arael LM f32", a32.solve_ms, a32.first_iter_ms, a32.iterations, Some(a32.accepted), a32.full_iter_ms, a32.solution, &mut cells);
        if !skip_tiny {
            let t = tiny_runner::run_lm(&scene);
            record("tiny-solver LM", t.solve_ms, t.first_iter_ms, t.iterations, None, 0.0, t.solution, &mut cells);
        }
        let fa = factrs_runner::run(&scene);
        record("factrs LM", fa.solve_ms, fa.first_iter_ms, fa.iterations, None, 0.0, fa.solution, &mut cells);
        if ceres_ok {
            for solver in &ceres_solvers {
                let c = run_ceres(scene_path, solver, scene.poses.len());
                let rel = ((c.initial_cost - initial_cost) / initial_cost).abs();
                assert!(rel < INITIAL_COST_RTOL, "ceres initial cost {} vs reference {} (rel {:.2e})",
                    c.initial_cost, initial_cost, rel);
                let core = last_core();
                assert!(c.cpus == core.to_string(), "ceres not pinned to core {}: {}", core, c.cpus);
                let label = format!("ceres {}", solver);
                subproc_peaks.insert(label.clone(), c.peak_mb);
                record(&label, c.solve_ms, c.first_iter_ms, c.iterations, Some(c.accepted),
                    0.0, c.solution, &mut cells);
            }
        }
        if symforce_ok {
            for (precision, label) in [("f64", "symforce LM f64"), ("f32", "symforce LM f32")] {
                let sf = run_symforce(scene_path, precision, scene.poses.len());
                let rel = ((sf.initial_cost - initial_cost) / initial_cost).abs();
                assert!(rel < INITIAL_COST_RTOL, "symforce {} initial cost {} vs reference {} (rel {:.2e})",
                    precision, sf.initial_cost, initial_cost, rel);
                let core = last_core();
                assert!(sf.cpus == core.to_string(), "symforce not pinned to core {}: {}", core, sf.cpus);
                subproc_peaks.insert(label.to_string(), sf.peak_mb);
                record(label, sf.solve_ms, sf.first_iter_ms, sf.iterations, Some(sf.accepted),
                    0.0, sf.solution, &mut cells);
            }
        }
        if g2o_ok {
            let g = run_g2o(scene_path, "lm", scene.poses.len());
            let rel = ((g.initial_cost - initial_cost) / initial_cost).abs();
            assert!(rel < INITIAL_COST_RTOL, "g2o initial cost {} vs reference {} (rel {:.2e})",
                g.initial_cost, initial_cost, rel);
            let core = last_core();
            assert!(g.cpus == core.to_string(), "g2o not pinned to core {}: {}", core, g.cpus);
            subproc_peaks.insert("g2o LM".to_string(), g.peak_mb);
            record("g2o LM", g.solve_ms, g.first_iter_ms, g.iterations, Some(g.accepted),
                0.0, g.solution, &mut cells);
        }
        if gtsam_ok {
            let gt = run_gtsam(scene_path, scene.poses.len());
            let rel = ((gt.initial_cost - initial_cost) / initial_cost).abs();
            assert!(rel < INITIAL_COST_RTOL, "gtsam initial cost {} vs reference {} (rel {:.2e})",
                gt.initial_cost, initial_cost, rel);
            let core = last_core();
            assert!(gt.cpus == core.to_string(), "gtsam not pinned to core {}: {}", core, gt.cpus);
            subproc_peaks.insert("gtsam LM".to_string(), gt.peak_mb);
            record("gtsam LM", gt.solve_ms, gt.first_iter_ms, gt.iterations, Some(gt.accepted),
                0.0, gt.solution, &mut cells);
        }
    }

    // Validation: best cost, each row within 1% cost AND < 5 cm pose RMSE
    // to the best (gauge is fixed by the landmark map + priors, so absolute
    // positions are comparable -- no alignment needed).
    let costs: Vec<f64> = cells.iter().map(|c| scene::reference_cost(&scene, &c.6)).collect();
    let best = costs.iter().cloned().fold(f64::MAX, f64::min);
    let best_i = costs.iter().position(|&c| c == best).unwrap();
    let best_sol = &cells[best_i].6;

    let measure_mem = std::env::var("LOC_NO_MEM").map_or(true, |v| v != "1");
    let mem_key = |label: &str| -> Option<&'static str> {
        match label {
            "arael LM f64" => Some("arael_f64"),
            "arael LM f32" => Some("arael_f32"),
            "tiny-solver LM" => Some("tiny"),
            "factrs LM" => Some("factrs"),
            _ => None,
        }
    };
    let mems: Vec<f64> = cells.iter().map(|c| {
        if let Some(&m) = subproc_peaks.get(&c.0) { m }
        else if measure_mem { mem_key(&c.0).map_or(0.0, |k| measure_peak_mb(k, scene.poses.len())) }
        else { 0.0 }
    }).collect();

    println!("\n{:<30} {:>10} {:>9} {:>10} {:>11} {:>12} {:>10} {:>16}",
        "system", "total ms", "iters", "ms/iter", "full-it ms", "1st-iter ms", "peak MB", "final cost");
    for (i, c) in cells.iter().enumerate() {
        let iters = match c.4 { Some(a) => format!("{}({})", a, c.3), None => format!("{}", c.3) };
        let rmse = pose_rmse(&c.6, best_sol);
        let ok = (costs[i] - best) / best < 1e-2 && rmse < 0.05;
        let mem = if mems[i] > 0.0 { format!("{:.1}", mems[i]) } else { "-".to_string() };
        // arael rows carry a per-phase measured value (c.5). External
        // systems derive theirs from the min-of-rounds totals: every steady
        // iteration of a rejection-free run is a full one, so
        // (total - 1st-iter) / (attempts - 1) isolates it. Computed from
        // the MERGED minima, not per round -- a min over per-round
        // differences would cherry-pick rounds with an inflated first-iter
        // measurement. "-" with rejected attempts or no attempt accounting.
        let ext_fi = match c.4 {
            Some(acc) if c.5 == 0.0 && acc == c.3 && c.3 >= 2 =>
                (c.1 - c.2) / (c.3 - 1) as f64,
            _ => 0.0,
        };
        let fi = if c.5 > 0.0 { c.5 } else { ext_fi };
        let full_it = if fi > 0.0 { format!("{:.2}", fi) } else { "-".to_string() };
        println!("{:<30} {:>10.1} {:>9} {:>10.2} {:>11} {:>12.1} {:>10} {:>16.4}{}",
            c.0, c.1, iters, c.1 / c.3.max(1) as f64, full_it, c.2, mem, costs[i],
            if ok { String::new() } else { format!("  <- off optimum (RMSE {:.3} m)", rmse) });
    }
    let conv = (0..cells.len()).filter(|&i| (costs[i]-best)/best < 1e-2
        && pose_rmse(&cells[i].6, best_sol) < 0.05).count();
    println!("validation: {}/{} at the common optimum ({:.4}: cost within 1%, pose RMSE < 5 cm)",
        conv, cells.len(), best);
    let _ = &init_sol;
}

fn last_core() -> usize {
    std::env::var("LOC_CORE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

struct CeresOut {
    solve_ms: f64, first_iter_ms: f64, iterations: usize, accepted: usize,
    initial_cost: f64, peak_mb: f64, cpus: String, solution: Solution,
}

fn run_ceres(scene_path: &str, linsolver: &str, n_poses: usize) -> CeresOut {
    let sol_out = "/tmp/loc_ceres_sol.txt";
    let out = std::process::Command::new("cpp/build/ceres_loc")
        .args([scene_path, sol_out, linsolver])
        .output().expect("failed to run ceres_loc");
    assert!(out.status.success(), "ceres_loc failed: {}", String::from_utf8_lossy(&out.stderr));
    parse_subproc_out(&out.stdout, sol_out, n_poses)
}

fn run_symforce(scene_path: &str, precision: &str, n_poses: usize) -> CeresOut {
    let sol_out = "/tmp/loc_symforce_sol.txt";
    let out = std::process::Command::new("cpp/build/symforce_loc")
        .args([scene_path, precision, sol_out])
        .output().expect("failed to run symforce_loc");
    assert!(out.status.success(), "symforce_loc failed: {}", String::from_utf8_lossy(&out.stderr));
    parse_subproc_out(&out.stdout, sol_out, n_poses)
}

fn run_g2o(scene_path: &str, mode: &str, n_poses: usize) -> CeresOut {
    let sol_out = "/tmp/loc_g2o_sol.txt";
    let out = std::process::Command::new("cpp/build/g2o_loc")
        .args([scene_path, mode, sol_out])
        .output().expect("failed to run g2o_loc");
    assert!(out.status.success(), "g2o_loc failed: {}", String::from_utf8_lossy(&out.stderr));
    parse_subproc_out(&out.stdout, sol_out, n_poses)
}

fn run_gtsam(scene_path: &str, n_poses: usize) -> CeresOut {
    let sol_out = "/tmp/loc_gtsam_sol.txt";
    let out = std::process::Command::new("cpp/build/gtsam_loc")
        .args([scene_path, "lm", sol_out])
        .output().expect("failed to run gtsam_loc");
    assert!(out.status.success(), "gtsam_loc failed: {}", String::from_utf8_lossy(&out.stderr));
    parse_subproc_out(&out.stdout, sol_out, n_poses)
}

// Parse the shared {solve_ms, ..., peak_rss_kb, cpus_allowed} JSON line the
// C++ runners print on stdout (trailing profiling output is ignored -- the
// line is located by "solve_ms").
fn parse_subproc_out(stdout: &[u8], sol_out: &str, n_poses: usize) -> CeresOut {
    let text = String::from_utf8_lossy(stdout);
    let line = text.lines().rev().find(|l| l.contains("solve_ms")).expect("no protocol line");
    let get_opt = |key: &str| -> Option<f64> {
        let i = line.find(key)?;
        let rest = &line[i + key.len() + 2..];
        rest.trim_start_matches(':').trim()
            .split(|c: char| c == ',' || c == '}').next().unwrap().trim().parse().ok()
    };
    let get = |key: &str| -> f64 {
        get_opt(key).unwrap_or_else(|| panic!("missing {}", key))
    };
    let cpus = {
        let i = line.find("cpus_allowed").unwrap();
        let rest = &line[i..];
        rest.split('"').nth(2).unwrap_or("?").to_string()
    };
    CeresOut {
        solve_ms: get("solve_ms"),
        first_iter_ms: get("first_iter_ms"),
        iterations: get("\"iterations\"") as usize,
        accepted: get("accepted") as usize,
        initial_cost: get("initial_cost"),
        peak_mb: get("peak_rss_kb") / 1024.0,
        cpus,
        solution: scene::read_solution(sol_out, n_poses),
    }
}
