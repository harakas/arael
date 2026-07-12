// Heterogeneous visual-inertial SLAM benchmark: arael vs tiny-solver.
// Same methodology as benchmarks/pgo and benchmarks/bal: one scene
// generator, one reference cost function, cross-checked initial cost,
// verified single core, min-of-N interleaved timing.

mod arael_runner;
mod factrs_runner;
mod scene;
mod tiny_runner;

use scene::{Scene, SceneConfig, Solution};

fn enforce_single_core() {
    for var in ["RAYON_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS", "TBB_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS"] {
        std::env::set_var(var, "1");
    }
    // Capture the core BEFORE pinning: afterwards available_parallelism
    // reports 1 (single core), so it can't be recomputed.
    let core = std::thread::available_parallelism().map(|n| n.get() - 1).unwrap_or(0);
    std::env::set_var("SLAM_CORE", core.to_string());
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
    if let Ok(n) = std::env::var("SLAM_POSES") {
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
// (SLAM_MEMSOLVER selects which; the process runs only that solver and
// prints its VmHWM). Isolating each solver in its own process gives a
// clean peak -- no allocator retention from a previous solver, and the
// same measurement basis as the Ceres subprocess.
//
// A feature build (eigen/cholmod/cholmod-gpl) links C libraries that
// inflate every row's VmHWM by a few MB of shared-library baseline a
// pure-Rust deployment would not carry. SLAM_MEM_EXE=<path to a default
// build> sources the rows' memory from that clean binary instead; the
// default build self-measures cleanly (the output records which).
fn measure_peak_mb(which: &str, poses: usize) -> f64 {
    let exe = match std::env::var("SLAM_MEM_EXE") {
        Ok(alt) => std::path::PathBuf::from(alt),
        _ => std::env::current_exe().unwrap(),
    };
    let out = std::process::Command::new(exe)
        .env("SLAM_MEMSOLVER", which)
        .env("SLAM_POSES", poses.to_string())
        .env("SLAM_MEM_ITERS", "3") // peak fill-in is reached in the first factorization
        .output().unwrap();
    String::from_utf8_lossy(&out.stdout).lines()
        .find_map(|l| l.strip_prefix("PEAK_RSS_KB:"))
        .and_then(|v| v.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0).unwrap_or(0.0)
}

fn main() {
    enforce_single_core();
    let cfg = config();
    let scene = scene::generate(&cfg);

    // Memory-measurement mode: run one solver, print peak RSS, exit.
    if let Ok(which) = std::env::var("SLAM_MEMSOLVER") {
        let iters: usize = std::env::var("SLAM_MEM_ITERS").ok()
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
            other => { eprintln!("unknown SLAM_MEMSOLVER {}", other); }
        }
        println!("PEAK_RSS_KB: {}", peak_rss_kb());
        return;
    }
    let init_sol = initial_solution(&scene);
    let initial_cost = scene::reference_cost(&scene, &init_sol);
    println!("scene: {} poses, {} landmarks, {} frines, {} odometry pairs, {} parameters",
        scene.poses.len(), scene.landmarks_init.len(), scene.frines.len(), scene.odo.len(),
        scene.poses.len() * 6 + scene.landmarks_init.len() * 3);
    println!("initial reference cost: {:.4}", initial_cost);

    // Cross-check: the arael model must compute the same cost at the
    // initial estimate as the reference function.
    let arael_init = arael_runner::initial_cost(&scene);
    let rel = ((arael_init - initial_cost) / initial_cost).abs();
    assert!(rel < 1e-9, "arael initial cost {} vs reference {} (rel {:.2e})",
        arael_init, initial_cost, rel);
    println!("arael initial cost matches reference to {:.2e}", rel);

    let tiny_init = tiny_runner::initial_cost(&scene);
    let rel = ((tiny_init - initial_cost) / initial_cost).abs();
    assert!(rel < 1e-9, "tiny initial cost {} vs reference {} (rel {:.2e})",
        tiny_init, initial_cost, rel);
    println!("tiny  initial cost matches reference to {:.2e}", rel);

    let factrs_init = factrs_runner::initial_cost(&scene);
    let rel = ((factrs_init - initial_cost) / initial_cost).abs();
    assert!(rel < 1e-9, "factrs initial cost {} vs reference {} (rel {:.2e})",
        factrs_init, initial_cost, rel);
    println!("factrs initial cost matches reference to {:.2e}", rel);

    // Hessian sparsity bitmap (eyeball the fill), env-gated.
    if let Ok(v) = std::env::var("SLAM_HESSIAN_BITMAP") {
        let out = if v == "1" || v.is_empty() { "hessian.png".to_string() } else { v };
        arael_runner::write_hessian_bitmap(&scene, &out);
    }

    tiny_runner::install_iter_counter();
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);

    // (label, solve_ms, first_iter_ms, iterations, accepted?, solution)
    let mut cells: Vec<(String, f64, f64, usize, Option<usize>, Solution)> = Vec::new();
    let record = |label: &str, sm: f64, fm: f64, it: usize, acc: Option<usize>,
                  sol: Solution, cells: &mut Vec<(String, f64, f64, usize, Option<usize>, Solution)>| {
        if let Some(c) = cells.iter_mut().find(|c| c.0 == label) {
            c.1 = c.1.min(sm);
            c.2 = c.2.min(fm);
        } else {
            cells.push((label.to_string(), sm, fm, it, acc, sol));
        }
    };
    // Ceres runs as a subprocess over an exported copy of the scene.
    let scene_path = "/tmp/slam_scene.txt";
    scene::write_scene(&scene, scene_path);
    let ceres_ok = std::path::Path::new("cpp/build/ceres_slam").exists();
    if !ceres_ok {
        eprintln!("WARNING: cpp/build/ceres_slam missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping Ceres");
    }
    let ceres_solvers: Vec<String> = std::env::var("CERES_SOLVERS")
        .unwrap_or_else(|_| "sparse_normal_cholesky,sparse_schur,iterative_schur".into())
        .split(',').map(|s| s.to_string()).collect();
    // Peaks reported by subprocess solvers (Ceres, SymForce, g2o), keyed by
    // row label; they measure their own VmHWM, so no re-solve is needed.
    let mut subproc_peaks: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    let symforce_ok = std::path::Path::new("cpp/build/symforce_slam").exists();
    if !symforce_ok {
        eprintln!("WARNING: cpp/build/symforce_slam missing (build with -DSYMFORCE_DIR=...); skipping SymForce");
    }
    let g2o_ok = std::path::Path::new("cpp/build/g2o_slam").exists();
    if !g2o_ok {
        eprintln!("WARNING: cpp/build/g2o_slam missing (needs g2o + cholmod); skipping g2o");
    }
    let gtsam_ok = std::path::Path::new("cpp/build/gtsam_slam").exists();
    if !gtsam_ok {
        eprintln!("WARNING: cpp/build/gtsam_slam missing (needs libgtsam-dev); skipping GTSAM");
    }

    let skip_tiny = std::env::var("SLAM_SKIP_TINY").map_or(false, |v| v == "1");
    // SLAM_SYSTEMS=<comma-separated substrings> runs only the matching rows
    // (e.g. SLAM_SYSTEMS=arael). Unset runs everything. A filtered run
    // validates only against whatever ran -- for iterating, not publishing.
    let systems_filter = std::env::var("SLAM_SYSTEMS").ok();
    let want = |label: &str| -> bool {
        systems_filter.as_deref().is_none_or(|f| {
            f.split(',').any(|pat| label.contains(pat.trim()))
        })
    };
    if systems_filter.is_some() {
        eprintln!("SLAM_SYSTEMS={} -- partial run, cross-system validation is not meaningful",
            systems_filter.as_deref().unwrap_or(""));
    }
    for _ in 0..rounds {
        if want("arael LM f64") {
        let a = arael_runner::run(&scene);
        record("arael LM f64", a.solve_ms, a.first_iter_ms, a.iterations, Some(a.accepted), a.solution, &mut cells);
        }
        if want("arael LM f32") {
        let a32 = arael_runner::run_f32(&scene);
        record("arael LM f32", a32.solve_ms, a32.first_iter_ms, a32.iterations, Some(a32.accepted), a32.solution, &mut cells);
        }
        if !skip_tiny && want("tiny-solver LM") {
            let t = tiny_runner::run_lm(&scene);
            record("tiny-solver LM", t.solve_ms, t.first_iter_ms, t.iterations, None, t.solution, &mut cells);
        }
        if want("factrs LM") {
        let fa = factrs_runner::run(&scene);
        record("factrs LM", fa.solve_ms, fa.first_iter_ms, fa.iterations, None, fa.solution, &mut cells);
        }
        if ceres_ok {
            for solver in &ceres_solvers {
                if !want(&format!("ceres {}", solver)) {
                    continue;
                }
                let c = run_ceres(scene_path, solver, scene.poses.len(), scene.landmarks_init.len());
                let rel = ((c.initial_cost - initial_cost) / initial_cost).abs();
                assert!(rel < 1e-9, "ceres initial cost {} vs reference {} (rel {:.2e})",
                    c.initial_cost, initial_cost, rel);
                let core = last_core();
                assert!(c.cpus == core.to_string(), "ceres not pinned to core {}: {}", core, c.cpus);
                let label = format!("ceres {}", solver);
                subproc_peaks.insert(label.clone(), c.peak_mb);
                record(&label, c.solve_ms, c.first_iter_ms, c.iterations, Some(c.accepted),
                    c.solution, &mut cells);
            }
        }
        if symforce_ok {
            for (precision, label) in [("f64", "symforce LM f64"), ("f32", "symforce LM f32")] {
                if !want(label) {
                    continue;
                }
                let sf = run_symforce(scene_path, precision, scene.poses.len(), scene.landmarks_init.len());
                let rel = ((sf.initial_cost - initial_cost) / initial_cost).abs();
                assert!(rel < 1e-9, "symforce {} initial cost {} vs reference {} (rel {:.2e})",
                    precision, sf.initial_cost, initial_cost, rel);
                let core = last_core();
                assert!(sf.cpus == core.to_string(), "symforce not pinned to core {}: {}", core, sf.cpus);
                subproc_peaks.insert(label.to_string(), sf.peak_mb);
                record(label, sf.solve_ms, sf.first_iter_ms, sf.iterations, Some(sf.accepted),
                    sf.solution, &mut cells);
            }
        }
        if g2o_ok && want("g2o LM") {
            let g = run_g2o(scene_path, "lm", scene.poses.len(), scene.landmarks_init.len());
            let rel = ((g.initial_cost - initial_cost) / initial_cost).abs();
            assert!(rel < 1e-9, "g2o initial cost {} vs reference {} (rel {:.2e})",
                g.initial_cost, initial_cost, rel);
            let core = last_core();
            assert!(g.cpus == core.to_string(), "g2o not pinned to core {}: {}", core, g.cpus);
            subproc_peaks.insert("g2o LM".to_string(), g.peak_mb);
            record("g2o LM", g.solve_ms, g.first_iter_ms, g.iterations, Some(g.accepted),
                g.solution, &mut cells);
        }
        if gtsam_ok && want("gtsam LM") {
            let gt = run_gtsam(scene_path, scene.poses.len(), scene.landmarks_init.len());
            let rel = ((gt.initial_cost - initial_cost) / initial_cost).abs();
            assert!(rel < 1e-9, "gtsam initial cost {} vs reference {} (rel {:.2e})",
                gt.initial_cost, initial_cost, rel);
            let core = last_core();
            assert!(gt.cpus == core.to_string(), "gtsam not pinned to core {}: {}", core, gt.cpus);
            subproc_peaks.insert("gtsam LM".to_string(), gt.peak_mb);
            record("gtsam LM", gt.solve_ms, gt.first_iter_ms, gt.iterations, Some(gt.accepted),
                gt.solution, &mut cells);
        }
    }

    // Validation: best cost, then each row within 1% cost AND < 5 cm
    // aligned-free translation RMSE to the best (gauge is fixed by GPS +
    // priors, so no alignment needed -- absolute positions are comparable).
    let costs: Vec<f64> = cells.iter().map(|c| scene::reference_cost(&scene, &c.5)).collect();
    let best = costs.iter().cloned().fold(f64::MAX, f64::min);
    let best_i = costs.iter().position(|&c| c == best).unwrap();
    let best_sol = &cells[best_i].5;

    // Peak memory per solver (fresh subprocess each). Env SLAM_NO_MEM
    // skips it (it re-solves once per solver).
    let measure_mem = std::env::var("SLAM_NO_MEM").map_or(true, |v| v != "1");
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
    // Provenance of the Rust rows' peak MB, so every log records which
    // binary the numbers came from (feature builds link extra C
    // libraries into self-measured RSS; see README Methodology).
    if measure_mem {
        let featured = cfg!(any(feature = "eigen", feature = "cholmod", feature = "cholmod-gpl"));
        match std::env::var("SLAM_MEM_EXE") {
            Ok(exe) => println!("peak MB: Rust rows measured via SLAM_MEM_EXE ({})", exe),
            Err(_) if featured => println!(
                "peak MB: self-measured inside a FEATURE build (linked C libraries inflate RSS) --                  set SLAM_MEM_EXE to a default-build binary"),
            Err(_) => println!("peak MB: self-measured (default build)"),
        }
    }

    println!("\n{:<30} {:>10} {:>9} {:>10} {:>12} {:>10} {:>16}",
        "system", "total ms", "iters", "ms/iter", "1st-iter ms", "peak MB", "final cost");
    for (i, c) in cells.iter().enumerate() {
        let iters = match c.4 { Some(a) => format!("{}({})", a, c.3), None => format!("{}", c.3) };
        let rmse = pose_rmse(&c.5, best_sol);
        let ok = (costs[i] - best) / best < 1e-2 && rmse < 0.05;
        let mem = if mems[i] > 0.0 { format!("{:.1}", mems[i]) } else { "-".to_string() };
        println!("{:<30} {:>10.1} {:>9} {:>10.2} {:>12.1} {:>10} {:>16.4}{}",
            c.0, c.1, iters, c.1 / c.3.max(1) as f64, c.2, mem, costs[i],
            if ok { String::new() } else { format!("  <- off optimum (RMSE {:.3} m)", rmse) });
    }
    let conv = (0..cells.len()).filter(|&i| (costs[i]-best)/best < 1e-2
        && pose_rmse(&cells[i].5, best_sol) < 0.05).count();
    println!("validation: {}/{} at the common optimum ({:.4}: cost within 1%, pose RMSE < 5 cm)",
        conv, cells.len(), best);
    let _ = &init_sol;
}

fn last_core() -> usize {
    std::env::var("SLAM_CORE").ok().and_then(|v| v.parse().ok()).unwrap_or(0)
}

struct CeresOut {
    solve_ms: f64, first_iter_ms: f64, iterations: usize, accepted: usize,
    initial_cost: f64, peak_mb: f64, cpus: String, solution: Solution,
}

fn run_ceres(scene_path: &str, linsolver: &str, n_poses: usize, n_landmarks: usize) -> CeresOut {
    let sol_out = "/tmp/slam_ceres_sol.txt";
    let out = std::process::Command::new("cpp/build/ceres_slam")
        .args([scene_path, sol_out, linsolver])
        .output().expect("failed to run ceres_slam");
    assert!(out.status.success(), "ceres_slam failed: {}", String::from_utf8_lossy(&out.stderr));
    parse_subproc_out(&out.stdout, sol_out, n_poses, n_landmarks)
}

// SymForce runs as a subprocess (like Ceres) over the exported scene; its
// generated C++ factors emit the same JSON protocol, precision selected
// by the second arg ("f64" or "f32"). It reports its own peak RSS.
fn run_symforce(scene_path: &str, precision: &str, n_poses: usize, n_landmarks: usize) -> CeresOut {
    let sol_out = "/tmp/slam_symforce_sol.txt";
    let out = std::process::Command::new("cpp/build/symforce_slam")
        .args([scene_path, precision, sol_out])
        .output().expect("failed to run symforce_slam");
    assert!(out.status.success(), "symforce_slam failed: {}", String::from_utf8_lossy(&out.stderr));
    parse_subproc_out(&out.stdout, sol_out, n_poses, n_landmarks)
}

// g2o runs as a subprocess (like Ceres) over the exported scene; custom
// edges with analytic Jacobians, same JSON protocol. mode is "lm" or "gn".
fn run_g2o(scene_path: &str, mode: &str, n_poses: usize, n_landmarks: usize) -> CeresOut {
    let sol_out = "/tmp/slam_g2o_sol.txt";
    let out = std::process::Command::new("cpp/build/g2o_slam")
        .args([scene_path, mode, sol_out])
        .output().expect("failed to run g2o_slam");
    assert!(out.status.success(), "g2o_slam failed: {}", String::from_utf8_lossy(&out.stderr));
    parse_subproc_out(&out.stdout, sol_out, n_poses, n_landmarks)
}

// GTSAM runs as a subprocess (like Ceres); custom NoiseModelFactorN
// factors with analytic Jacobians, same JSON protocol.
fn run_gtsam(scene_path: &str, n_poses: usize, n_landmarks: usize) -> CeresOut {
    let sol_out = "/tmp/slam_gtsam_sol.txt";
    let out = std::process::Command::new("cpp/build/gtsam_slam")
        .args([scene_path, "lm", sol_out])
        .output().expect("failed to run gtsam_slam");
    assert!(out.status.success(), "gtsam_slam failed: {}", String::from_utf8_lossy(&out.stderr));
    parse_subproc_out(&out.stdout, sol_out, n_poses, n_landmarks)
}

// Parse the shared {solve_ms, ..., peak_rss_kb, cpus_allowed} JSON line
// that the Ceres and SymForce runners both print on stdout (any trailing
// profiling output is ignored -- the line is located by "solve_ms").
fn parse_subproc_out(stdout: &[u8], sol_out: &str, n_poses: usize, n_landmarks: usize) -> CeresOut {
    let text = String::from_utf8_lossy(stdout);
    let line = text.lines().rev().find(|l| l.contains("solve_ms")).expect("no protocol line");
    let get = |key: &str| -> f64 {
        let i = line.find(key).unwrap_or_else(|| panic!("missing {}", key));
        let rest = &line[i + key.len() + 2..];
        rest.trim_start_matches(':').trim()
            .split(|c: char| c == ',' || c == '}').next().unwrap().trim().parse().unwrap()
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
        solution: scene::read_solution(sol_out, n_poses, n_landmarks),
    }
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
        landmarks: scene.landmarks_init.iter().map(|l| arael::vect::vect3d::from(*l)).collect(),
    }
}
