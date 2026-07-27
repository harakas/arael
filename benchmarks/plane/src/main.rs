// Plane SLAM benchmark: arael (whose S^2 plane-normal parameterization is a
// user-defined #[arael(component)]) vs g2o, Ceres and factrs. Same
// methodology as the sibling benchmarks: one scene generator, one reference
// cost function, cross-checked initial cost, verified single core, min-of-N
// interleaved timing.
//
// The application is g2o's plane_slam example (libg2o-doc): SE3 poses on a
// loop, plane landmarks (normal + distance) observed relative to each pose,
// odometry between consecutive poses.

mod arael_runner;
mod factrs_runner;
mod scene;

use scene::{RawScene, Scene, Solution};

fn print_header(raw: &RawScene, rounds: usize, systems_filter: &Option<String>) {
    use bench_harness::header::Header;
    let arael_cfg = bench_harness::arael::config::<arael_runner::World>(raw, 0);
    Header::new("plane-bench")
        .rounds(rounds)
        .line("scene", format!("{} poses, {} planes, seed {} [PLANE_POSES, PLANE_SHARED]",
            raw.poses.len(), raw.planes.len(), scene::SEED))
        .line("systems", format!("{} [PLANE_SYSTEMS]",
            systems_filter.as_deref().unwrap_or("all")))
        .line("arael lambda0", format!("{:e} (f64), {:e} (f32) [ARAEL_LAMBDA0]",
            bench_harness::arael::lambda0::<arael_runner::World>(raw),
            bench_harness::arael::lambda0::<arael_runner::WorldF>(raw)))
        .line("arael damping", format!("{} [DRIVER: fixed|nielsen]",
            if bench_harness::arael::nielsen::<arael_runner::World>() { "Nielsen gain-ratio driver" }
            else { "fixed ladder (default driver)" }))
        .line("termination", format!("{:e} [PLANE_TOL], f32 rows {:e} [PLANE_TOL_F32]",
            arael_runner::tolerance(), arael_runner::tolerance_f32()))
        .line("arael termination", format!("abs {:e}, rel {:e}, patience {}, min_iters {}",
            arael_cfg.abs_precision, arael_cfg.rel_precision,
            arael_cfg.patience, arael_cfg.min_iters))
        .line("solver verbose", format!("{} [VERBOSE], per-solve timing {} [TIMING]",
            if arael_cfg.verbose { "on" } else { "off" },
            if std::env::var("TIMING").is_ok() { "on" } else { "off" }))
        .line("memory pass", format!("{} [PLANE_NO_MEM]",
            if std::env::var("PLANE_NO_MEM").is_err() { "on" } else { "off" }))
        .core()
        .print();
}

// Peak memory for the in-process rows, each in a process of its own (VmHWM
// is a process-wide high-water mark).
fn mem_pass() -> bool {
    let Ok(which) = std::env::var("PLANE_MEM") else { return false };
    let sc = scene::make_scene();
    // The peak fill-in is reached in the first factorization, so a capped
    // solve measures the same high-water mark as the full one, faster.
    let iters: usize = std::env::var("PLANE_MEM_ITERS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(3);
    match which.as_str() {
        "arael LM f64" => { std::hint::black_box(arael_runner::run_capped(&sc.raw, iters)); }
        "arael LM f32" => { std::hint::black_box(arael_runner::run_f32_capped(&sc.raw, iters)); }
        "factrs LM" => { std::hint::black_box(factrs_runner::run(&sc.raw).solution); }
        other => panic!("unknown system for the memory pass: {}", other),
    }
    bench_harness::mem::report_peak();
    true
}

// The geometry the shared table is generic over.
struct Geo<'a>(&'a RawScene);
impl bench_harness::table::Geometry for Geo<'_> {
    type Solution = Solution;
    fn cost(&self, sol: &Solution) -> f64 { scene::reference_cost(self.0, sol) }
    fn distance(a: &Solution, b: &Solution) -> f64 { pose_rmse(a, b) }
}

fn pose_rmse(a: &Solution, b: &Solution) -> f64 {
    let n = a.poses.len() as f64;
    let s: f64 = a.poses.iter().zip(&b.poses)
        .map(|(pa, pb)| { let d = pa.t - pb.t; d * d })
        .sum();
    (s / n).sqrt()
}

/// One external runner: the harness parses its protocol line and asserts the
/// core pin; the initial-cost cross-check stays here, because it is what
/// proves the system minimizes the same objective.
fn run_ext(cmd: &str, args: &[&str], sol_out: &str, n_poses: usize,
           initial_cost: f64) -> bench_harness::table::Row<Solution> {
    let mut c = std::process::Command::new(cmd);
    c.args(args);
    let p = bench_harness::external::run(c);
    let reported = p.json.get("initial_cost").and_then(|v| v.as_f64())
        .expect("runner reported no initial_cost");
    let rel = ((reported - initial_cost) / initial_cost).abs();
    assert!(rel < 1e-9, "{} initial cost {} vs reference {} (rel {:.2e})",
        cmd, reported, initial_cost, rel);
    let mut row = bench_harness::table::Row::new(
        p.solve_ms, p.first_iter_ms, p.iterations,
        scene::read_solution(sol_out, n_poses));
    row.accepted = p.accepted;
    row.full_ms = p.full_ms;
    row.peak_mb = p.json.get("peak_mb").and_then(|v| v.as_f64());
    row
}

fn main() {
    bench_harness::pin::enforce_cores();
    if mem_pass() {
        return;
    }
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    // PLANE_SYSTEMS=<comma-separated substrings> runs only the matching rows.
    // A filtered run validates only against whatever ran -- for iterating,
    // not publishing.
    let systems_filter = std::env::var("PLANE_SYSTEMS").ok();

    let sc: Scene = scene::make_scene();
    let raw = &sc.raw;
    print_header(raw, rounds, &systems_filter);

    let init_sol = Solution { poses: raw.poses.clone(), planes: raw.planes.clone() };
    let initial_cost = scene::reference_cost(raw, &init_sol);
    println!("scene: {} poses, {} planes, {} odometry pairs, {} observations, {} parameters",
        raw.poses.len(), raw.planes.len(), raw.odos.len(), raw.obs.len(),
        raw.poses.len() * 7 + raw.planes.len() * 3);
    println!("initial reference cost: {:.4}", initial_cost);

    // Cross-checks: every in-process system must compute the same cost at the
    // initial estimate as the reference function (the external runners report
    // theirs over the protocol, checked in run_ext).
    let arael_init = arael_runner::initial_cost(raw);
    let rel = ((arael_init - initial_cost) / initial_cost).abs();
    assert!(rel < 1e-9, "arael initial cost {} vs reference {} (rel {:.2e})",
        arael_init, initial_cost, rel);
    println!("arael  initial cost matches reference to {:.2e}", rel);

    let factrs_init = factrs_runner::initial_cost(raw);
    let rel = ((factrs_init - initial_cost) / initial_cost).abs();
    assert!(rel < 1e-9, "factrs initial cost {} vs reference {} (rel {:.2e})",
        factrs_init, initial_cost, rel);
    println!("factrs initial cost matches reference to {:.2e}", rel);

    let geo = Geo(raw);
    let mut t = bench_harness::table::Table::new(&geo);

    // g2o and Ceres run as subprocesses over an exported copy of the scene.
    let scene_path = "/tmp/plane_scene.txt";
    scene::write_scene_file(scene_path, &sc);
    let g2o_ok = std::path::Path::new("cpp/build/g2o_plane").exists();
    if !g2o_ok {
        eprintln!("WARNING: cpp/build/g2o_plane missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping g2o");
    }
    let ceres_ok = std::path::Path::new("cpp/build/ceres_plane").exists();
    if !ceres_ok {
        eprintln!("WARNING: cpp/build/ceres_plane missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping Ceres");
    }
    let gtsam_ok = std::path::Path::new("cpp/build/gtsam_plane").exists();
    if !gtsam_ok {
        eprintln!("WARNING: cpp/build/gtsam_plane missing (needs libgtsam-dev); skipping GTSAM");
    }
    let symforce_ok = std::path::Path::new("cpp/build/symforce_plane").exists();
    if !symforce_ok {
        eprintln!("WARNING: cpp/build/symforce_plane missing (build with -DSYMFORCE_DIR=...); skipping SymForce");
    }

    let want = |label: &str| -> bool {
        systems_filter.as_deref().is_none_or(|f| {
            f.split(',').any(|pat| label.contains(pat.trim()))
        })
    };
    if systems_filter.is_some() {
        eprintln!("PLANE_SYSTEMS={} -- partial run, cross-system validation is not meaningful",
            systems_filter.as_deref().unwrap_or(""));
    }
    for _ in 0..rounds {
        if want("arael LM f64") {
            t.record_result("arael LM f64", arael_runner::run(raw));
        }
        if want("arael LM f32") {
            t.record_result("arael LM f32", arael_runner::run_f32(raw));
        }
        if want("factrs LM") {
            t.record("factrs LM", factrs_runner::run(raw));
        }
        if g2o_ok && want("g2o LM") {
            t.record("g2o LM", run_ext("cpp/build/g2o_plane",
                &[scene_path, "/tmp/plane_g2o_sol.txt"],
                "/tmp/plane_g2o_sol.txt", raw.poses.len(), initial_cost));
        }
        if ceres_ok && want("ceres LM") {
            t.record("ceres LM", run_ext("cpp/build/ceres_plane",
                &[scene_path, "/tmp/plane_ceres_sol.txt"],
                "/tmp/plane_ceres_sol.txt", raw.poses.len(), initial_cost));
        }
        if gtsam_ok && want("gtsam LM") {
            t.record("gtsam LM", run_ext("cpp/build/gtsam_plane",
                &[scene_path, "/tmp/plane_gtsam_sol.txt"],
                "/tmp/plane_gtsam_sol.txt", raw.poses.len(), initial_cost));
        }
        if symforce_ok {
            for (precision, label) in [("f64", "symforce LM f64"), ("f32", "symforce LM f32")] {
                if !want(label) {
                    continue;
                }
                let sol = format!("/tmp/plane_symforce_sol_{}.txt", precision);
                t.record(label, run_ext("cpp/build/symforce_plane",
                    &[scene_path, precision, &sol],
                    &sol, raw.poses.len(), initial_cost));
            }
        }
    }

    if std::env::var("PLANE_NO_MEM").is_err() {
        let poses = raw.poses.len().to_string();
        let shared = std::env::var("PLANE_SHARED").unwrap_or_default();
        for label in ["arael LM f64", "arael LM f32", "factrs LM"] {
            if !want(label) {
                continue;
            }
            let mut extra = vec![("PLANE_POSES", poses.as_str())];
            if !shared.is_empty() {
                extra.push(("PLANE_SHARED", shared.as_str()));
            }
            if let Some(mb) = bench_harness::mem::measure("PLANE_MEM", label, &extra) {
                t.set_peak_mb(label, mb);
            }
        }
    }
    t.print();
}
