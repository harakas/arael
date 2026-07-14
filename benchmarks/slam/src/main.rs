// Heterogeneous visual-inertial SLAM benchmark: arael vs tiny-solver.
// Same methodology as benchmarks/pgo and benchmarks/bal: one scene
// generator, one reference cost function, cross-checked initial cost,
// verified single core, min-of-N interleaved timing.

mod arael_runner;
mod factrs_runner;
mod scene;
mod tiny_runner;

use scene::{Scene, SceneConfig, Solution};


/// The settings this run used, printed before anything else: a pasted result has
/// to carry them. Values come from the objects the run actually uses, so the
/// header cannot drift from what ran.
fn print_header(scene: &Scene, cfg: &SceneConfig, rounds: usize, skip_tiny: bool,
                systems_filter: &Option<String>, ceres_solvers: &[String]) {
    use bench_harness::header::{on_off, Header};
    let arael_cfg = bench_harness::arael::config::<arael_runner::Path>(scene, 0);
    Header::new("slam-bench")
        .rounds(rounds)
        .line("scene", format!("{} poses, {} landmarks, seed {} [SLAM_POSES]",
            cfg.num_poses, cfg.num_landmarks, cfg.seed))
        .line("systems", format!("{} [SLAM_SYSTEMS]",
            systems_filter.as_deref().unwrap_or("all")))
        .line("optional systems", format!("tiny-solver {} [RUN_TINY]", on_off(skip_tiny)))
        .line("ceres solvers", format!("{} [CERES_SOLVERS]", ceres_solvers.join(", ")))
        .line("arael lambda0", format!("{:e} (f64), {:e} (f32) [ARAEL_LAMBDA0]",
            bench_harness::arael::lambda0::<arael_runner::Path>(scene),
            bench_harness::arael::lambda0::<arael_runner::PathF>(scene)))
        .line("arael damping", format!("{} [DRIVER: default|nielsen]",
            if bench_harness::arael::nielsen::<arael_runner::Path>() { "Nielsen gain-ratio driver" }
            else { "fixed ladder (default driver)" }))
        .line("arael backend", format!("{} [SLAM_ARAEL_SOLVER: schur|faer|cholmod]",
            std::env::var("SLAM_ARAEL_SOLVER").unwrap_or_else(|_| "schur".to_string())))
        .line("arael termination", format!("abs {:e}, rel {:e}, patience {}, min_iters {}",
            arael_cfg.abs_precision, arael_cfg.rel_precision,
            arael_cfg.patience, arael_cfg.min_iters))
        .line("solver verbose", format!("{} [VERBOSE], per-solve timing {} [TIMING]",
            if arael_cfg.verbose { "on" } else { "off" },
            if std::env::var("TIMING").is_ok() { "on" } else { "off" }))
        .line("memory pass", format!("{} [SLAM_NO_MEM]",
            if std::env::var("SLAM_NO_MEM").is_err() { "on" } else { "off" }))
        .core()
        .print();
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

// Peak memory for the in-process rows.
//
// VmHWM is a high-water mark for the whole PROCESS, so arael, factrs and
// tiny-solver sharing one would each report the largest peak anything before
// them reached. Each is therefore run alone, in a process of its own, doing
// nothing but the solve. The subprocess systems report their own.
fn mem_pass() -> bool {
    let Ok(which) = std::env::var("SLAM_MEM") else { return false };
    let scene = scene::generate(&config());
    // The peak fill-in is reached in the first factorization, so a capped solve
    // measures the same high-water mark as the full one, faster.
    let iters: usize = std::env::var("SLAM_MEM_ITERS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(3);
    match which.as_str() {
        "arael LM f64" => { std::hint::black_box(arael_runner::run_capped(&scene, iters)); }
        "arael LM f32" => { std::hint::black_box(arael_runner::run_f32_capped(&scene, iters)); }
        "tiny-solver LM" => {
            std::env::set_var("TINY_MAXITER", iters.to_string());
            tiny_runner::install_iter_counter();
            std::hint::black_box(tiny_runner::run_lm(&scene).solution);
        }
        "factrs LM" => { std::hint::black_box(factrs_runner::run(&scene).solution); }
        other => panic!("unknown system for the memory pass: {}", other),
    }
    bench_harness::mem::report_peak();
    true
}

fn main() {
    bench_harness::pin::enforce_single_core();
    if mem_pass() {
        return;
    }
    let cfg = config();
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    // Ceres calls it SPARSE_NORMAL_CHOLESKY; the table says what it is.
    fn ceres_label(solver: &str) -> String {
        let short = solver.strip_prefix("sparse_normal_").map_or(solver, |_| "sparse_cholesky");
        format!("ceres {}", short)
    }
    let ceres_solvers: Vec<String> = std::env::var("CERES_SOLVERS")
        .unwrap_or_else(|_| "sparse_normal_cholesky,sparse_schur,iterative_schur".into())
        .split(',').map(|s| s.to_string()).collect();
    // tiny-solver is off by default (RUN_TINY=1 brings it back): it is an order
    // of magnitude slower than the field and only compresses the scale the other
    // systems are read on. The harness runs and validates it exactly as it does
    // the others.
    let skip_tiny = std::env::var("RUN_TINY").is_err();
    // SLAM_SYSTEMS=<comma-separated substrings> runs only the matching rows
    // (e.g. SLAM_SYSTEMS=arael). Unset runs everything. A filtered run
    // validates only against whatever ran -- for iterating, not publishing.
    let systems_filter = std::env::var("SLAM_SYSTEMS").ok();

    let scene = scene::generate(&cfg);
    print_header(&scene, &cfg, rounds, skip_tiny, &systems_filter, &ceres_solvers);

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

    let geo = Geo(&scene);
    let mut t = bench_harness::table::Table::new(&geo);

    // Ceres runs as a subprocess over an exported copy of the scene.
    let scene_path = "/tmp/slam_scene.txt";
    scene::write_scene(&scene, scene_path);
    let ceres_ok = std::path::Path::new("cpp/build/ceres_slam").exists();
    if !ceres_ok {
        eprintln!("WARNING: cpp/build/ceres_slam missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping Ceres");
    }
    // Peaks reported by subprocess solvers (Ceres, SymForce, g2o), keyed by
    // row label; they measure their own VmHWM, so no re-solve is needed.
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
        t.record("arael LM f64", a);
        }
        if want("arael LM f32") {
        let a32 = arael_runner::run_f32(&scene);
        t.record("arael LM f32", a32);
        }
        if !skip_tiny && want("tiny-solver LM") {
            t.record("tiny-solver LM", tiny_runner::run_lm(&scene));
        }
        if want("factrs LM") {
        t.record("factrs LM", factrs_runner::run(&scene));
        }
        if ceres_ok {
            for solver in &ceres_solvers {
                if !want(&ceres_label(solver)) {
                    continue;
                }
                let c = run_ceres(scene_path, solver, scene.poses.len(), scene.landmarks_init.len());
                let rel = ((c.initial_cost - initial_cost) / initial_cost).abs();
                assert!(rel < 1e-9, "ceres initial cost {} vs reference {} (rel {:.2e})",
                    c.initial_cost, initial_cost, rel);
                t.record(&ceres_label(solver), c.row);
            }
        }
        if symforce_ok {
            for (precision, label) in [("f64", "symforce LM f64"), ("f32", "symforce LM f32")] {
                if !want(label) {
                    continue;
                }
                let sf = run_symforce(scene_path, precision, scene.poses.len(), scene.landmarks_init.len());
                let rel = ((sf.initial_cost - initial_cost) / initial_cost).abs();
                assert!(rel < 1e-9, "symforce initial cost {} vs reference {} (rel {:.2e})",
                    sf.initial_cost, initial_cost, rel);
                t.record(label, sf.row);
            }
        }
        if g2o_ok && want("g2o LM") {
            let g = run_g2o(scene_path, "lm", scene.poses.len(), scene.landmarks_init.len());
            let rel = ((g.initial_cost - initial_cost) / initial_cost).abs();
            assert!(rel < 1e-9, "g2o initial cost {} vs reference {} (rel {:.2e})",
                g.initial_cost, initial_cost, rel);
            t.record("g2o LM", g.row);
        }
        if gtsam_ok && want("gtsam LM") {
            let gt = run_gtsam(scene_path, scene.poses.len(), scene.landmarks_init.len());
            let rel = ((gt.initial_cost - initial_cost) / initial_cost).abs();
            assert!(rel < 1e-9, "gtsam initial cost {} vs reference {} (rel {:.2e})",
                gt.initial_cost, initial_cost, rel);
            t.record("gtsam LM", gt.row);
        }
    }

    if std::env::var("SLAM_NO_MEM").is_err() {
        let poses = scene.poses.len().to_string();
        for label in ["arael LM f64", "arael LM f32", "tiny-solver LM", "factrs LM"] {
            if !want(label) || (skip_tiny && label == "tiny-solver LM") {
                continue;
            }
            if let Some(mb) = bench_harness::mem::measure(
                "SLAM_MEM", label, &[("SLAM_POSES", poses.as_str())]) {
                t.set_peak_mb(label, mb);
            }
        }
    }
    t.print();
    let _ = &init_sol;
}

/// One external runner: the harness parses its protocol line and asserts the
/// core pin; the scene's own cost cross-check stays here, because it is what
/// proves the system minimizes the same objective.
struct Ext {
    row: bench_harness::table::Row<Solution>,
    initial_cost: f64,
}

fn run_ext(mut cmd: std::process::Command, args: &[&str], sol_out: &str,
           n_poses: usize, n_landmarks: usize) -> Ext {
    cmd.args(args);
    let p = bench_harness::external::run(cmd);
    let solution = scene::read_solution(sol_out, n_poses, n_landmarks);
    let mut row = bench_harness::table::Row::new(
        p.solve_ms, p.first_iter_ms, p.iterations, solution);
    row.accepted = p.accepted;
    row.full_ms = p.full_ms;
    row.peak_mb = p.json.get("peak_mb").and_then(|v| v.as_f64());
    Ext {
        row,
        initial_cost: p.json.get("initial_cost").and_then(|v| v.as_f64())
            .expect("runner reported no initial_cost"),
    }
}

fn run_ceres(scene_path: &str, linsolver: &str, n_poses: usize, n_landmarks: usize) -> Ext {
    let sol_out = "/tmp/slam_ceres_sol.txt";
    run_ext(std::process::Command::new("cpp/build/ceres_slam"),
        &[scene_path, sol_out, linsolver], sol_out, n_poses, n_landmarks)
}

fn run_symforce(scene_path: &str, precision: &str, n_poses: usize, n_landmarks: usize) -> Ext {
    let sol_out = "/tmp/slam_symforce_sol.txt";
    run_ext(std::process::Command::new("cpp/build/symforce_slam"),
        &[scene_path, precision, sol_out], sol_out, n_poses, n_landmarks)
}

fn run_g2o(scene_path: &str, mode: &str, n_poses: usize, n_landmarks: usize) -> Ext {
    let sol_out = "/tmp/slam_g2o_sol.txt";
    run_ext(std::process::Command::new("cpp/build/g2o_slam"),
        &[scene_path, mode, sol_out], sol_out, n_poses, n_landmarks)
}

fn run_gtsam(scene_path: &str, n_poses: usize, n_landmarks: usize) -> Ext {
    let sol_out = "/tmp/slam_gtsam_sol.txt";
    run_ext(std::process::Command::new("cpp/build/gtsam_slam"),
        &[scene_path, "lm", sol_out], sol_out, n_poses, n_landmarks)
}


// The geometry the shared table is generic over.
struct Geo<'a>(&'a Scene);
impl bench_harness::table::Geometry for Geo<'_> {
    type Solution = Solution;
    fn cost(&self, sol: &Solution) -> f64 { scene::reference_cost(self.0, sol) }
    fn distance(a: &Solution, b: &Solution) -> f64 { pose_rmse(a, b) }
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
