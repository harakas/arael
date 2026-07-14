// Localization benchmark: a trajectory estimated against a fixed landmark map
// (bearings + odometry + drift/tilt priors), arael vs the field. One scene
// generator, one reference cost, cross-checked initial cost, and the shared
// benchmark harness for everything else -- the probes, the timing rules, the
// table, the core pin. LOC_POSES=N sets the pose count (landmarks scale as 4N).

mod arael_runner;
mod factrs_runner;
mod scene;
mod tiny_runner;

use scene::{Scene, SceneConfig, Solution};

// Every system's initial cost is cross-checked against the one reference cost
// function, all at this shared tolerance. 1e-5 accommodates arael's fast_atan
// bearing residuals (max error < 1e-6 rad, measured ~4e-7 relative on the
// cost); a real model mismatch shows up orders above it for any system.
const INITIAL_COST_RTOL: f64 = 1e-5;

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

// Ceres calls it SPARSE_NORMAL_CHOLESKY; the table says what it is.
fn ceres_label(solver: &str) -> String {
    let short = solver.strip_prefix("sparse_normal_").map_or(solver, |_| "sparse_cholesky");
    format!("ceres {}", short)
}

/// The settings this run used, printed before anything else: a pasted result has
/// to carry them. Values come from the objects the run actually uses, so the
/// header cannot drift from what ran.
fn print_header(scene: &Scene, cfg: &SceneConfig, rounds: usize, skip_tiny: bool,
                systems_filter: &Option<String>, ceres_solvers: &[String]) {
    use bench_harness::header::{on_off, Header};
    let arael_cfg = bench_harness::arael::config::<arael_runner::Path>(scene, 0);
    Header::new("loc-bench")
        .rounds(rounds)
        .line("scene", format!("{} poses, {} landmarks (fixed), seed {} [LOC_POSES]",
            cfg.num_poses, cfg.num_landmarks, cfg.seed))
        .line("systems", format!("{} [LOC_SYSTEMS]",
            systems_filter.as_deref().unwrap_or("all")))
        .line("optional systems", format!("tiny-solver {} [RUN_TINY]", on_off(skip_tiny)))
        .line("ceres solvers", format!("{} [CERES_SOLVERS]", ceres_solvers.join(", ")))
        .line("arael lambda0", format!("{:e} (f64), {:e} (f32) [ARAEL_LAMBDA0]",
            bench_harness::arael::lambda0::<arael_runner::Path>(scene),
            bench_harness::arael::lambda0::<arael_runner::PathF>(scene)))
        .line("arael damping", format!("{} [DRIVER: default|nielsen]",
            if bench_harness::arael::nielsen::<arael_runner::Path>() { "Nielsen gain-ratio driver" }
            else { "fixed ladder (default driver)" }))
        .line("arael backend", format!("{} [LOC_ARAEL_SOLVER: band|faer]",
            arael_runner::backend()))
        .line("arael termination", format!("abs {:e}, rel {:e}, patience {}, min_iters {}",
            arael_cfg.abs_precision, arael_cfg.rel_precision,
            arael_cfg.patience, arael_cfg.min_iters))
        .line("solver verbose", format!("{} [VERBOSE], per-solve timing {} [TIMING]",
            if arael_cfg.verbose { "on" } else { "off" },
            if std::env::var("TIMING").is_ok() { "on" } else { "off" }))
        .line("memory pass", format!("{} [LOC_NO_MEM]",
            if std::env::var("LOC_NO_MEM").is_err() { "on" } else { "off" }))
        .core()
        .print();
}

// Peak memory for the in-process rows.
//
// VmHWM is a high-water mark for the whole PROCESS, so arael, factrs and
// tiny-solver sharing one would each report the largest peak anything before
// them reached. Each is therefore run alone, in a process of its own, doing
// nothing but the solve. The subprocess systems report their own.
fn mem_pass() -> bool {
    let Ok(which) = std::env::var("LOC_MEM") else { return false };
    let scene = scene::generate(&config());
    // The peak is reached in the first factorization, so a capped solve
    // measures the same high-water mark as the full one, faster.
    let iters: usize = std::env::var("LOC_MEM_ITERS").ok()
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

/// Interleaved f64/f32 solves with arael's per-phase instrumentation, reported
/// as steady-state means with each phase's first call (which carries the
/// one-time structure costs) excluded and shown apart. For chasing where a
/// precision does or does not speed up; LOC_PHASES=1.
fn phase_timing(scene: &Scene, rounds: usize) {
    let mut t64: Vec<arael::simple_lm::LmTiming> = Vec::new();
    let mut t32: Vec<arael::simple_lm::LmTiming> = Vec::new();
    for _ in 0..rounds {
        t64.push(arael_runner::run_timed_once(scene));
        t32.push(arael_runner::run_timed_once_f32(scene));
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
    println!("\nphase timing, {} poses, {} interleaved rounds per precision",
        scene.poses.len(), rounds);
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
}

fn main() {
    bench_harness::pin::enforce_cores();
    if mem_pass() {
        return;
    }

    let cfg = config();
    let scene = scene::generate(&cfg);
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let ceres_solvers: Vec<String> = std::env::var("CERES_SOLVERS")
        .unwrap_or_else(|_| "sparse_normal_cholesky,sparse_schur,iterative_schur".into())
        .split(',').map(|s| s.to_string()).collect();
    // tiny-solver is off by default (RUN_TINY=1 brings it back): it is an order
    // of magnitude slower than the field and only compresses the scale the other
    // systems are read on. The harness runs and validates it exactly as it does
    // the others.
    let skip_tiny = std::env::var("RUN_TINY").is_err();
    // LOC_SYSTEMS=<comma-separated substrings> runs only the matching rows (e.g.
    // LOC_SYSTEMS=arael). Unset runs everything. A filtered run validates only
    // against whatever ran -- for iterating, not publishing.
    let systems_filter = std::env::var("LOC_SYSTEMS").ok();
    print_header(&scene, &cfg, rounds, skip_tiny, &systems_filter, &ceres_solvers);

    let init_sol = initial_solution(&scene);
    let initial_cost = scene::reference_cost(&scene, &init_sol);
    println!("scene: {} poses, {} landmarks (fixed), {} frines, {} odometry pairs, {} parameters",
        scene.poses.len(), scene.landmarks.len(), scene.frines.len(), scene.odo.len(),
        scene.poses.len() * 6);
    println!("initial reference cost: {:.4}", initial_cost);

    // Cross-check: each in-process system must compute the same cost at the
    // initial estimate as the reference function -- which is what proves they
    // all minimize the same objective. The subprocess systems are checked
    // against the same number as they report it.
    for (name, cost) in [
        ("arael ", arael_runner::initial_cost(&scene)),
        ("tiny  ", tiny_runner::initial_cost(&scene)),
        ("factrs", factrs_runner::initial_cost(&scene)),
    ] {
        let rel = ((cost - initial_cost) / initial_cost).abs();
        assert!(rel < INITIAL_COST_RTOL, "{} initial cost {} vs reference {} (rel {:.2e})",
            name, cost, initial_cost, rel);
        println!("{} initial cost matches reference to {:.2e}", name, rel);
    }

    if std::env::var("LOC_PHASES").is_ok() {
        phase_timing(&scene, rounds);
        return;
    }

    tiny_runner::install_iter_counter();

    let geo = Geo(&scene);
    let mut t = bench_harness::table::Table::new(&geo);

    // The C++ runners execute as subprocesses over an exported copy of the scene.
    let scene_path = "/tmp/loc_scene.txt";
    scene::write_scene(&scene, scene_path);
    let ceres_ok = std::path::Path::new("cpp/build/ceres_loc").exists();
    if !ceres_ok {
        eprintln!("WARNING: cpp/build/ceres_loc missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping Ceres");
    }
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

    let want = |label: &str| -> bool {
        systems_filter.as_deref().is_none_or(|f| {
            f.split(',').any(|pat| label.contains(pat.trim()))
        })
    };
    if systems_filter.is_some() {
        eprintln!("LOC_SYSTEMS={} -- partial run, cross-system validation is not meaningful",
            systems_filter.as_deref().unwrap_or(""));
    }
    // Interleaved rounds: the table keeps the minimum of each cell, so a system
    // cannot be charged for a disturbance that hit only its part of the run.
    let n_poses = scene.poses.len();
    for _ in 0..rounds {
        if want("arael LM f64") {
            t.record("arael LM f64", arael_runner::run(&scene));
        }
        if want("arael LM f32") {
            t.record("arael LM f32", arael_runner::run_f32(&scene));
        }
        if !skip_tiny && want("tiny-solver LM") {
            t.record("tiny-solver LM", tiny_runner::run_lm(&scene));
        }
        if want("factrs LM") {
            t.record("factrs LM", factrs_runner::run(&scene));
        }
        if ceres_ok {
            for solver in &ceres_solvers {
                let label = ceres_label(solver);
                if !want(&label) {
                    continue;
                }
                let c = run_ceres(scene_path, solver, n_poses);
                check_initial(&label, c.initial_cost, initial_cost);
                t.record(&label, c.row);
            }
        }
        if symforce_ok {
            for (precision, label) in [("f64", "symforce LM f64"), ("f32", "symforce LM f32")] {
                if !want(label) {
                    continue;
                }
                let sf = run_symforce(scene_path, precision, n_poses);
                check_initial(label, sf.initial_cost, initial_cost);
                t.record(label, sf.row);
            }
        }
        if g2o_ok && want("g2o LM") {
            let g = run_g2o(scene_path, "lm", n_poses);
            check_initial("g2o LM", g.initial_cost, initial_cost);
            t.record("g2o LM", g.row);
        }
        if gtsam_ok && want("gtsam LM") {
            let gt = run_gtsam(scene_path, n_poses);
            check_initial("gtsam LM", gt.initial_cost, initial_cost);
            t.record("gtsam LM", gt.row);
        }
    }

    if std::env::var("LOC_NO_MEM").is_err() {
        let poses = n_poses.to_string();
        for label in ["arael LM f64", "arael LM f32", "tiny-solver LM", "factrs LM"] {
            if !want(label) || (skip_tiny && label == "tiny-solver LM") {
                continue;
            }
            if let Some(mb) = bench_harness::mem::measure(
                "LOC_MEM", label, &[("LOC_POSES", poses.as_str())]) {
                t.set_peak_mb(label, mb);
            }
        }
    }
    t.print();
}

/// One external system's cost at the initial estimate, as it computes it, must
/// be the reference cost -- the proof it minimizes the same objective.
fn check_initial(label: &str, reported: f64, reference: f64) {
    let rel = ((reported - reference) / reference).abs();
    assert!(rel < INITIAL_COST_RTOL, "{} initial cost {} vs reference {} (rel {:.2e})",
        label, reported, reference, rel);
}

/// One external runner: the harness parses its protocol line and asserts the
/// core pin; the scene's own cost cross-check stays here, because it is what
/// proves the system minimizes the same objective.
struct Ext {
    row: bench_harness::table::Row<Solution>,
    initial_cost: f64,
}

fn run_ext(mut cmd: std::process::Command, args: &[&str], sol_out: &str, n_poses: usize) -> Ext {
    cmd.args(args);
    let p = bench_harness::external::run(cmd);
    let solution = scene::read_solution(sol_out, n_poses);
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

fn run_ceres(scene_path: &str, linsolver: &str, n_poses: usize) -> Ext {
    let sol_out = "/tmp/loc_ceres_sol.txt";
    run_ext(std::process::Command::new("cpp/build/ceres_loc"),
        &[scene_path, sol_out, linsolver], sol_out, n_poses)
}

fn run_symforce(scene_path: &str, precision: &str, n_poses: usize) -> Ext {
    let sol_out = "/tmp/loc_symforce_sol.txt";
    run_ext(std::process::Command::new("cpp/build/symforce_loc"),
        &[scene_path, precision, sol_out], sol_out, n_poses)
}

fn run_g2o(scene_path: &str, mode: &str, n_poses: usize) -> Ext {
    let sol_out = "/tmp/loc_g2o_sol.txt";
    run_ext(std::process::Command::new("cpp/build/g2o_loc"),
        &[scene_path, mode, sol_out], sol_out, n_poses)
}

fn run_gtsam(scene_path: &str, n_poses: usize) -> Ext {
    let sol_out = "/tmp/loc_gtsam_sol.txt";
    run_ext(std::process::Command::new("cpp/build/gtsam_loc"),
        &[scene_path, "lm", sol_out], sol_out, n_poses)
}

// The geometry the shared table is generic over.
struct Geo<'a>(&'a Scene);
impl bench_harness::table::Geometry for Geo<'_> {
    type Solution = Solution;
    fn cost(&self, sol: &Solution) -> f64 { scene::reference_cost(self.0, sol) }
    fn distance(a: &Solution, b: &Solution) -> f64 { pose_rmse(a, b) }
}

// The gauge is fixed by the landmark map and the priors, so absolute positions
// are comparable across systems -- no alignment needed before the RMSE.
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
