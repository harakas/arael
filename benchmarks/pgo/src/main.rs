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
fn run_external(cmd: std::process::Command, poses_out: &str, n_poses: usize)
    -> bench_harness::table::Row<Vec<PoseIn>> {
    let p = bench_harness::external::run(cmd);
    let poses: Vec<PoseIn> = std::fs::read_to_string(poses_out).unwrap()
        .lines()
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            PoseIn { x: v[0], y: v[1], th: v[2] }
        })
        .collect();
    assert_eq!(poses.len(), n_poses);
    let mut row = bench_harness::table::Row::new(
        p.solve_ms, p.first_iter_ms, p.iterations, poses);
    row.accepted = p.accepted;
    row.full_ms = p.full_ms;
    row.peak_mb = p.json.get("peak_mb").and_then(|v| v.as_f64());
    row
}

fn run_gtsam(python: &str, ds_path: &str, kind: &str, weighted: bool, n_poses: usize) -> bench_harness::table::Row<Vec<PoseIn>> {
    let poses_out = format!("/tmp/gtsam_poses_{}.txt", kind);
    let weights = if weighted { "info" } else { "unit" };
    let mut cmd = std::process::Command::new(python);
    cmd.args(["gtsam_bench.py", ds_path, kind, &poses_out, weights]);
    run_external(cmd, &poses_out, n_poses)
}

fn run_ceres(ds_path: &str, weighted: bool, n_poses: usize) -> bench_harness::table::Row<Vec<PoseIn>> {
    let mut cmd = std::process::Command::new("cpp/build/ceres_bench");
    cmd.args([ds_path, "/tmp/ceres_poses.txt", if weighted { "info" } else { "unit" }]);
    run_external(cmd, "/tmp/ceres_poses.txt", n_poses)
}

// factrs's f32 mode can fail outright (its Cholesky panics on a
// non-positive pivot on city10000 -- single precision loses positive
// definiteness at 30000 parameters and its lambda floor of 1e-10 cannot
// regularize it). A crashed run is a reportable outcome, not a harness
// failure.
fn run_factrs32(ds_path: &str, kind: &str, weighted: bool, n_poses: usize) -> Option<bench_harness::table::Row<Vec<PoseIn>>> {
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

// Same for the SE3 datasets. The graph is the parent crate's SE3 runner
// compiled against factrs's f32 dtype (see factrs32/src/main3.rs).
fn run_factrs32_3d(ds_path: &str, kind: &str, n_poses: usize)
    -> Option<bench_harness::table::Row<Vec<g2o3::Pose3In>>> {
    let poses_out = format!("/tmp/factrs32_3d_poses_{}.txt", kind);
    let bin = "factrs32/target/release/factrs32-bench3";
    let mut probe = std::process::Command::new(bin);
    probe.args([ds_path, kind, &poses_out]);
    if !probe.output().ok()?.status.success() {
        return None;
    }
    Some(run_external3_checked(
        std::process::Command::new(bin),
        &[ds_path, kind, &poses_out], &poses_out, n_poses, None, &|_| {}))
}

// g2o's initial damping, per dataset. sphere2500 needs far more than the rest:
// at 1e-12 its first Levenberg iteration burns six trials (five rejected, each
// a full factorization) before it finds a step, and the solve keeps retrying all
// the way down. At 1e-6 the first iteration is a single clean attempt and the
// retries stop. On the other datasets 1e-6 is much too strong -- it triples
// parking-garage's iteration count -- so they keep 1e-12.
fn g2o_lambda(dataset: &str) -> &'static str {
    if dataset.contains("sphere") { "1e-6" } else { "1e-12" }
}


fn run_g2o(ds_path: &str, kind: &str, weighted: bool, n_poses: usize)
    -> bench_harness::table::Row<Vec<PoseIn>> {
    let poses_out = format!("/tmp/g2o_poses_{}.txt", kind);
    let mut cmd = std::process::Command::new("cpp/build/g2o_bench");
    cmd.args([ds_path, kind, &poses_out, if weighted { "info" } else { "unit" }]);
    // g2o's own initial chi2 equals the reference cost bit-for-bit (the edges
    // carry the conjugated information); the 2D table checks it in the runner.
    cmd.env("G2O_LAMBDA_INIT", g2o_lambda(ds_path));
    run_external(cmd, &poses_out, n_poses)
}

fn run_symforce(ds_path: &str, prec: &str, weighted: bool, n_poses: usize) -> bench_harness::table::Row<Vec<PoseIn>> {
    let poses_out = format!("/tmp/symforce_poses_{}.txt", prec);
    let mut cmd = std::process::Command::new("cpp/build/symforce_bench");
    cmd.args([ds_path, prec, &poses_out, if weighted { "info" } else { "unit" }]);
    run_external(cmd, &poses_out, n_poses)
}


// Run an external 3D benchmark process: same JSON protocol, poses file
// carries "x y z qx qy qz qw" lines. `cost_key` names the JSON field
// where the runner reports the canonical cost of the initial estimate
// as computed by ITS OWN code; `check` asserts it equals the reference
// cost function's value.
fn run_external3_checked(mut cmd: std::process::Command, args: &[&str], poses_out: &str,
                         n_poses: usize, cost_key: Option<&str>, check: &dyn Fn(f64))
    -> bench_harness::table::Row<Vec<g2o3::Pose3In>> {
    cmd.args(args);
    let p = bench_harness::external::run(cmd);
    // The system's OWN code reports the cost of the initial estimate; the
    // harness asserts it equals the reference cost. That is what proves every
    // system is minimizing the same objective, not merely converging to it.
    if let Some(key) = cost_key {
        check(p.json.get(key).and_then(|v| v.as_f64())
            .unwrap_or_else(|| panic!("missing {} from the runner", key)));
    }
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
    let mut row = bench_harness::table::Row::new(
        p.solve_ms, p.first_iter_ms, p.iterations, poses);
    row.accepted = p.accepted;
    row.full_ms = p.full_ms;
    row.peak_mb = p.json.get("peak_mb").and_then(|v| v.as_f64());
    row
}

// The two geometries the table is generic over: how to score a solution, and
// how to compare two of them.
struct Geo2<'a>(&'a g2o::Dataset);
impl bench_harness::table::Geometry for Geo2<'_> {
    type Solution = Vec<g2o::PoseIn>;
    fn cost(&self, poses: &Vec<g2o::PoseIn>) -> f64 { g2o::reference_cost(self.0, poses) }
    fn distance(a: &Vec<g2o::PoseIn>, b: &Vec<g2o::PoseIn>) -> f64 { g2o::aligned_rmse(a, b) }
}

struct Geo3<'a>(&'a g2o3::Dataset3);
impl bench_harness::table::Geometry for Geo3<'_> {
    type Solution = Vec<g2o3::Pose3In>;
    fn cost(&self, poses: &Vec<g2o3::Pose3In>) -> f64 { g2o3::reference_cost3(self.0, poses) }
    fn distance(a: &Vec<g2o3::Pose3In>, b: &Vec<g2o3::Pose3In>) -> f64 { g2o3::aligned_rmse3(a, b) }
}

fn run_dataset3(name: &str, path: &str, rounds: usize, f32_floor_note: Option<&str>) {
    let ds = g2o3::load3(path);
    println!("\n=== {} : {} poses, {} edges, {} parameters ===",
        name, ds.poses.len(), ds.edges.len(), ds.poses.len() * 6);
    let initial_cost = g2o3::reference_cost3(&ds, &ds.poses);
    println!("initial reference cost: {:.3}", initial_cost);

    let cpp3_available = std::path::Path::new("cpp/build/ceres_bench3").exists()
        && std::path::Path::new("cpp/build/g2o_bench3").exists();
    if !cpp3_available {
        eprintln!("WARNING: cpp/build 3D runners missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping those rows");
    }
    let factrs32_3d_available =
        std::path::Path::new("factrs32/target/release/factrs32-bench3").exists();
    if !factrs32_3d_available {
        eprintln!("WARNING: factrs32 SE3 runner missing (cargo build -r in factrs32/); \
                   skipping factrs f32 rows");
    }

    let geo = Geo3(&ds);
    let mut t = bench_harness::table::Table::with_f32_floor(&geo, f32_floor_note);

    for round in 0..rounds {
        let a64 = arael_runner3::run_f64(&ds);
        t.record("arael LM f64", a64);
        let a32 = arael_runner3::run_f32(&ds);
        t.record("arael LM f32", a32);
        if !skip_tiny() {
            let tgn = tiny_runner3::run_gn(&ds);
            t.record("tiny-solver GN", tgn);
            let tlm = tiny_runner3::run_lm(&ds);
            t.record("tiny-solver LM", tlm);
        }
        let fgn = factrs_runner3::run_gn(&ds);
        t.record("factrs GN", fgn);
        let flm = factrs_runner3::run_lm(&ds);
        t.record("factrs LM", flm);
        if factrs32_3d_available {
            for (kind, label) in [("gn", "factrs GN f32"), ("lm", "factrs LM f32")] {
                match run_factrs32_3d(path, kind, ds.poses.len()) {
                    Some(row) => t.record(label, row),
                    None => {
                        t.notes.insert(format!(
                            "{}: solver crashed (f32 Cholesky non-positive pivot)", label));
                    }
                }
            }
        }
        let symforce3_available = std::path::Path::new("cpp/build/symforce_bench3").exists();
        if cpp3_available {
            let check = |line_cost: f64| {
                assert!(((line_cost - initial_cost) / initial_cost).abs() < 1e-9,
                    "external initial cost {} disagrees with reference {}",
                    line_cost, initial_cost);
            };
            let row = run_external3_checked(
                std::process::Command::new("cpp/build/ceres_bench3"),
                &[path, "/tmp/ceres3_poses.txt"], "/tmp/ceres3_poses.txt",
                ds.poses.len(), Some("initial_cost"), &check);
            t.record("ceres LM", row);
            for kind in ["lm", "gn"] {
                let poses_out = format!("/tmp/g2o3_poses_{}.txt", kind);
                let mut g2o_cmd = std::process::Command::new("cpp/build/g2o_bench3");
                g2o_cmd.env("G2O_LAMBDA_INIT", g2o_lambda(name));
                let row = run_external3_checked(
                    g2o_cmd,
                    &[path, kind, &poses_out], &poses_out,
                    ds.poses.len(), Some("initial_cost"), &check);
                t.record(&format!("g2o {}", kind.to_uppercase()), row);
            }
            if symforce3_available {
                for prec in ["f64", "f32"] {
                    let poses_out = format!("/tmp/symforce3_poses_{}.txt", prec);
                    let row = run_external3_checked(
                        std::process::Command::new("cpp/build/symforce_bench3"),
                        &[path, prec, &poses_out], &poses_out,
                        ds.poses.len(), Some("initial_cost"), &check);
                    let label = if prec == "f32" { "symforce LM f32" } else { "symforce LM" };
                    t.record(label, row);
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
                let row = run_external3_checked(
                    std::process::Command::new(&py),
                    &["gtsam_bench.py", path, kind, &poses_out], &poses_out,
                    ds.poses.len(), None, &|_| {});
                t.record(&format!("gtsam {}", kind.to_uppercase()), row);
            }
        }
        eprintln!("  round {}/{} done", round + 1, rounds);
    }

    for (label, mb) in measure_in_process_memory(path, true, true,
        &["arael LM f64", "arael LM f32", "factrs GN", "factrs LM"]) {
        t.set_peak_mb(&label, mb);
    }
    t.print();
}

fn skip_tiny() -> bool {
    std::env::var("RUN_TINY").is_err()
}

/// GTSAM's ISAM2 is off by default (RUN_ISAM=1 brings it back). It answers the
/// incremental question -- one update per pose -- rather than the batch one this
/// benchmark asks, so its row is not comparable to the others, and it is slow.
fn skip_isam() -> bool {
    std::env::var("RUN_ISAM").is_err()
}

// The run's configuration, printed at startup so a pasted result carries the
// settings that produced it. Values are read back from the actual config
// objects and env accessors the runners use, so the header cannot drift from
// what was run; the env var that changes each one is named in brackets.
fn print_header(rounds: usize, only: &Option<String>) {
    let c64 = bench_harness::arael::config::<arael_runner::Graph>(&g2o::Dataset::default(), 0);
    let nielsen = bench_harness::arael::nielsen();
    let ordering = arael_runner::ordering();

    println!("=== pgo-bench configuration ===");
    println!("rounds            : {} [ROUNDS], {} probe sub-rounds per round",
        rounds, bench_harness::probe::PROBE_SUBROUNDS);
    println!("datasets          : {} [PGO_ONLY]",
        only.as_deref().unwrap_or("all"));
    let on_off = |skipped: bool| if skipped { "skipped" } else { "run" };
    println!("optional systems  : tiny-solver {} [RUN_TINY], isam {} [RUN_ISAM]",
        on_off(skip_tiny()), on_off(skip_isam()));
    println!("arael lambda0     : {:e} (2D), {:e} (3D) [ARAEL_LAMBDA0]",
        arael_runner::LAMBDA0_2D,
        arael_runner3::LAMBDA0_3D);
    println!("arael damping     : {} [PGO_DRIVER: default|nielsen]",
        if nielsen { "Nielsen gain-ratio driver" } else { "fixed ladder (default driver)" });
    // Auto is not a third ordering -- on a pose graph there is nothing to
    // marginalize, so there is no reduced system and it factorizes under AMD.
    let resolves_to = match ordering {
        arael::simple_lm::FaerOrdering::Auto => " -- AMD here (no marginalizable blocks)",
        _ => "",
    };
    println!("arael ordering    : {:?}{} [PGO_ORDERING: auto|nd]", ordering, resolves_to);
    println!("arael termination : abs {:e}, rel {:e}, patience {}, min_iters {}",
        c64.abs_precision, c64.rel_precision, c64.patience, c64.min_iters);
    println!("solver verbose    : {} [VERBOSE], per-solve timing {} [TIMING]",
        if c64.verbose { "on" } else { "off" },
        if std::env::var("TIMING").is_ok() { "on" } else { "off" });
    println!("pinned to core    : {} (all thread pools forced to 1)",
        std::env::var("BENCH_CORE").unwrap_or_else(|_| "?".to_string()));
}

// Peak memory for the in-process rows.
//
// VmHWM is a high-water mark for the whole PROCESS, so arael, factrs and
// tiny-solver sharing one would each report the largest peak anything before
// them reached. Each is therefore run alone, in a process of its own, doing
// nothing but the solve.
fn mem_pass() -> bool {
    let Ok(which) = std::env::var("PGO_MEM") else { return false };
    let path = std::env::var("PGO_MEM_DATASET").expect("PGO_MEM_DATASET");
    let weighted = std::env::var("PGO_MEM_WEIGHTED").is_ok();
    let three_d = std::env::var("PGO_MEM_3D").is_ok();
    if three_d {
        let ds = g2o3::load3(&path);
        match which.as_str() {
            "arael LM f64" => { arael_runner3::run_f64(&ds); }
            "arael LM f32" => { arael_runner3::run_f32(&ds); }
            "factrs GN" => { factrs_runner3::run_gn(&ds); }
            "factrs LM" => { factrs_runner3::run_lm(&ds); }
            other => panic!("unknown system for the memory pass: {}", other),
        }
    } else {
        let ds = g2o::load(&path, !weighted);
        match which.as_str() {
            "arael LM f64" => { arael_runner::run_f64(&ds); }
            "arael LM f32" => { arael_runner::run_f32(&ds); }
            "factrs GN" => { factrs_runner::run_gn(&ds); }
            "factrs LM" => { factrs_runner::run_lm(&ds); }
            other => panic!("unknown system for the memory pass: {}", other),
        }
    }
    bench_harness::mem::report_peak();
    true
}

/// The in-process systems, measured one process each.
fn measure_in_process_memory(path: &str, weighted: bool, three_d: bool, labels: &[&str])
    -> Vec<(String, f64)> {
    let mut out = Vec::new();
    for label in labels {
        let mut extra: Vec<(&str, &str)> = vec![("PGO_MEM_DATASET", path)];
        if weighted { extra.push(("PGO_MEM_WEIGHTED", "1")); }
        if three_d { extra.push(("PGO_MEM_3D", "1")); }
        if let Some(mb) = bench_harness::mem::measure("PGO_MEM", label, &extra) {
            out.push((label.to_string(), mb));
        }
    }
    out
}

fn main() {
    bench_harness::pin::enforce_single_core();
    if mem_pass() {
        return;
    }
    tiny_runner::install_iter_counter();
    let bench_dir = std::env::current_dir().unwrap();
    // (name, file, use the file's information matrices?). Every system is fed
    // the files' information matrices; the unit-weight path stays wired up
    // (pass false) but no dataset uses it.
    let datasets = [
        ("M3500", bench_dir.join("datasets/input_M3500_g2o.g2o"), true),
        ("city10000", bench_dir.join("datasets/city10000.g2o"), true),
    ];
    let rounds: usize = std::env::var("ROUNDS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(5);
    // PGO_ONLY=<substring> runs only the datasets whose name contains it.
    let only = std::env::var("PGO_ONLY").ok();
    let selected = |name: &str| only.as_deref().map_or(true, |f| name.contains(f));
    print_header(rounds, &only);
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

        let geo = Geo2(&ds);
        let mut t = bench_harness::table::Table::new(&geo);

        for round in 0..rounds {
            let a64 = arael_runner::run_f64(&ds);
            t.record("arael LM f64", a64);
            let a32 = arael_runner::run_f32(&ds);
            t.record("arael LM f32", a32);
            if !skip_tiny() {
                let tgn = tiny_runner::run_gn(&ds);
                t.record("tiny-solver GN", tgn);
                let tlm = tiny_runner::run_lm(&ds);
                t.record("tiny-solver LM", tlm);
            }
            let fgn = factrs_runner::run_gn(&ds);
            t.record("factrs GN", fgn);
            let flm = factrs_runner::run_lm(&ds);
            t.record("factrs LM", flm);
            if factrs32_available {
                for (kind, label) in [("gn", "factrs GN f32"), ("lm", "factrs LM f32")] {
                    match run_factrs32(path, kind, *weighted, ds.poses.len()) {
                        Some(out) => t.record(label, out),
                        None => {
                            t.notes.insert(format!(
                                "{}: solver crashed (f32 Cholesky non-positive pivot)", label));
                        }
                    }
                }
            }
            if let Some(py) = &python {
                t.record("gtsam LM", run_gtsam(py, path, "lm", *weighted, ds.poses.len()));
                t.record("gtsam GN", run_gtsam(py, path, "gn", *weighted, ds.poses.len()));
            }
            if symforce_available {
                t.record("symforce LM", run_symforce(path, "f64", *weighted, ds.poses.len()));
                t.record("symforce LM f32", run_symforce(path, "f32", *weighted, ds.poses.len()));
            }
            if cpp_available {
                t.record("ceres LM", run_ceres(path, *weighted, ds.poses.len()));
                t.record("g2o LM", run_g2o(path, "lm", *weighted, ds.poses.len()));
                t.record("g2o GN", run_g2o(path, "gn", *weighted, ds.poses.len()));
            }
            eprintln!("  round {}/{} done", round + 1, rounds);
        }

        // Incremental reference row: GTSAM driven the way its own
        // city10000 example works (one ISAM2 update per pose). A
        // different algorithm answering the online-estimation question;
        // listed for context, timed once, "iters" = incremental updates.
        if let Some(py) = &python {
            if !skip_isam() {
                t.record("gtsam ISAM2 (incr)", run_gtsam(py, path, "isam2", *weighted, ds.poses.len()));
            }
        }

        // Convergence is judged against the best solution by BOTH cost
        // (within 1%) and geometry (rigid-aligned RMSE under 5 cm to the
        // best solution) -- the cost surface has near-flat directions
        // where a plateau 0.9% above the optimum can sit meters away
        // geometrically (observed with g2o LM on the weighted M3500).
        for (label, mb) in measure_in_process_memory(
            path, *weighted, false,
            &["arael LM f64", "arael LM f32", "factrs GN", "factrs LM"]) {
            t.set_peak_mb(&label, mb);
        }
        t.print();
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
