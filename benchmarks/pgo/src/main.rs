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

mod arael_pipeline;
mod table;
mod arael_runner;
mod arael_runner3;
mod factrs_counting;
mod probe;
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
fn run_external(mut cmd: std::process::Command, poses_out: &str, n_poses: usize) -> (f64, f64, usize, Option<usize>, Option<f64>, Vec<PoseIn>) {
    let out = cmd.output().unwrap_or_else(|e| panic!("failed to run {:?}: {}", cmd, e));
    assert!(out.status.success(), "{:?} failed: {}", cmd, String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let line = text.lines().rev().find(|l| l.contains("solve_ms"))
        .unwrap_or_else(|| panic!("no protocol line from {:?}", cmd));
    let json: serde_json::Value = serde_json::from_str(line)
        .unwrap_or_else(|e| panic!("bad protocol line from {:?}: {} -- {}", cmd, e, line));
    let get = |key: &str| -> Option<f64> { json.get(key)?.as_f64() };
    let solve_ms = get("solve_ms").expect("solve_ms");
    let first_iter_ms = get("first_iter_ms").expect("first_iter_ms");
    let iterations = get("iterations").expect("iterations") as usize;
    let accepted = get("accepted").map(|v| v as usize);
    // A complete iteration, measured: t(2 iterations) - t(1 iteration), the
    // setup cancelling out. Only meaningful if that second step was ACCEPTED --
    // a rejected one only redoes the linear solve and is not an iteration.
    // t(2 iterations), raw. One iteration is this minus first_iter_ms, but the
    // minima over rounds have to be taken BEFORE the subtraction: differencing
    // two noisy cold runs can come out negative.
    // A first-iteration time is only meaningful if that iteration WAS one clean
    // iteration: a single attempt, accepted. A runner that rejected a step in
    // it (g2o at the wrong damping burned six trials there) reports a number
    // that is mostly wasted factorizations, and every quantity derived from it
    // -- full-iter above all, which is t(2) - t(1) -- inherits the lie. Runners
    // that report the counts get held to them; the number is dropped otherwise.
    let first_iter_clean = match (get("first_attempts"), get("first_accepted")) {
        (Some(a), Some(ok)) => a as usize == 1 && ok as usize == 1,
        _ => true, // runner does not report it; nothing to check against
    };
    let full_ms = match (get("second_run_ms"), get("second_accepted")) {
        (Some(ms), Some(acc)) if acc as usize >= 2 && first_iter_clean => Some(ms),
        _ => None,
    };
    // NaN travels through the min-of-rounds plumbing and prints as "-".
    let first_iter_ms = if first_iter_clean { first_iter_ms } else { f64::NAN };
    // Active check: the subprocess must have inherited the pin.
    let core = std::env::var("PGO_BENCH_CORE").unwrap();
    assert!(json["cpus_allowed"].as_str() == Some(core.as_str()),
        "{:?} not pinned to CPU {}: {}", cmd, core, line);
    let poses: Vec<PoseIn> = std::fs::read_to_string(poses_out).unwrap()
        .lines()
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|t| t.parse().unwrap()).collect();
            PoseIn { x: v[0], y: v[1], th: v[2] }
        })
        .collect();
    assert_eq!(poses.len(), n_poses);
    (solve_ms, first_iter_ms, iterations, accepted, full_ms, poses)
}

fn run_gtsam(python: &str, ds_path: &str, kind: &str, weighted: bool, n_poses: usize) -> (f64, f64, usize, Option<usize>, Option<f64>, Vec<PoseIn>) {
    let poses_out = format!("/tmp/gtsam_poses_{}.txt", kind);
    let weights = if weighted { "info" } else { "unit" };
    let mut cmd = std::process::Command::new(python);
    cmd.args(["gtsam_bench.py", ds_path, kind, &poses_out, weights]);
    run_external(cmd, &poses_out, n_poses)
}

fn run_ceres(ds_path: &str, weighted: bool, n_poses: usize) -> (f64, f64, usize, Option<usize>, Option<f64>, Vec<PoseIn>) {
    let mut cmd = std::process::Command::new("cpp/build/ceres_bench");
    cmd.args([ds_path, "/tmp/ceres_poses.txt", if weighted { "info" } else { "unit" }]);
    run_external(cmd, "/tmp/ceres_poses.txt", n_poses)
}

// factrs's f32 mode can fail outright (its Cholesky panics on a
// non-positive pivot on city10000 -- single precision loses positive
// definiteness at 30000 parameters and its lambda floor of 1e-10 cannot
// regularize it). A crashed run is a reportable outcome, not a harness
// failure.
fn run_factrs32(ds_path: &str, kind: &str, weighted: bool, n_poses: usize) -> Option<(f64, f64, usize, Option<usize>, Option<f64>, Vec<PoseIn>)> {
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
    -> Option<(f64, f64, usize, Option<usize>, Option<f64>, Vec<g2o3::Pose3In>)> {
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

// A measured millisecond value, or "-" when the harness could not measure it
// cleanly (see first_iter_clean).
fn fmt1(v: f64) -> String {
    if v.is_finite() { format!("{:.1}", v) } else { "-".to_string() }
}

fn run_symforce(ds_path: &str, prec: &str, weighted: bool, n_poses: usize) -> (f64, f64, usize, Option<usize>, Option<f64>, Vec<PoseIn>) {
    let poses_out = format!("/tmp/symforce_poses_{}.txt", prec);
    let mut cmd = std::process::Command::new("cpp/build/symforce_bench");
    cmd.args([ds_path, prec, &poses_out, if weighted { "info" } else { "unit" }]);
    run_external(cmd, &poses_out, n_poses)
}

fn run_g2o(ds_path: &str, kind: &str, weighted: bool, n_poses: usize) -> (f64, f64, usize, Option<usize>, Option<f64>, Vec<PoseIn>) {
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
    -> (f64, f64, usize, Option<usize>, Option<f64>, Vec<g2o3::Pose3In>) {
    cmd.args(args);
    let out = cmd.output().unwrap_or_else(|e| panic!("failed to run {:?}: {}", cmd, e));
    assert!(out.status.success(), "{:?} failed: {}", cmd, String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let line = text.lines().rev().find(|l| l.contains("solve_ms"))
        .unwrap_or_else(|| panic!("no protocol line from {:?}", cmd));
    let json: serde_json::Value = serde_json::from_str(line)
        .unwrap_or_else(|e| panic!("bad protocol line from {:?}: {} -- {}", cmd, e, line));
    let get = |key: &str| -> Option<f64> { json.get(key)?.as_f64() };
    let solve_ms = get("solve_ms").expect("solve_ms");
    let first_iter_ms = get("first_iter_ms").expect("first_iter_ms");
    let iterations = get("iterations").expect("iterations") as usize;
    let accepted = get("accepted").map(|v| v as usize);
    // A first-iteration time is only meaningful if that iteration WAS one clean
    // iteration: a single attempt, accepted. A runner that rejected a step in
    // it (g2o at the wrong damping burned six trials there) reports a number
    // that is mostly wasted factorizations, and every quantity derived from it
    // -- full-iter above all, which is t(2) - t(1) -- inherits the lie. Runners
    // that report the counts get held to them; the number is dropped otherwise.
    let first_iter_clean = match (get("first_attempts"), get("first_accepted")) {
        (Some(a), Some(ok)) => a as usize == 1 && ok as usize == 1,
        _ => true, // runner does not report it; nothing to check against
    };
    let full_ms = match (get("second_run_ms"), get("second_accepted")) {
        (Some(ms), Some(acc)) if acc as usize >= 2 && first_iter_clean => Some(ms),
        _ => None,
    };
    // NaN travels through the min-of-rounds plumbing and prints as "-".
    let first_iter_ms = if first_iter_clean { first_iter_ms } else { f64::NAN };
    if let Some(key) = cost_key {
        check(get(key).unwrap_or_else(|| panic!("missing {} from {:?}", key, cmd)));
    }
    let core = std::env::var("PGO_BENCH_CORE").unwrap();
    assert!(json["cpus_allowed"].as_str() == Some(core.as_str()),
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
    (solve_ms, first_iter_ms, iterations, accepted, full_ms, poses)
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
// The two geometries the table is generic over: how to score a solution, and
// how to compare two of them.
struct Geo2<'a>(&'a g2o::Dataset);
impl table::Geometry for Geo2<'_> {
    type Pose = g2o::PoseIn;
    fn cost(&self, poses: &[g2o::PoseIn]) -> f64 { g2o::reference_cost(self.0, poses) }
    fn aligned_rmse(a: &[g2o::PoseIn], b: &[g2o::PoseIn]) -> f64 { g2o::aligned_rmse(a, b) }
}

struct Geo3<'a>(&'a g2o3::Dataset3);
impl table::Geometry for Geo3<'_> {
    type Pose = g2o3::Pose3In;
    fn cost(&self, poses: &[g2o3::Pose3In]) -> f64 { g2o3::reference_cost3(self.0, poses) }
    fn aligned_rmse(a: &[g2o3::Pose3In], b: &[g2o3::Pose3In]) -> f64 { g2o3::aligned_rmse3(a, b) }
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
    let mut t = table::Table::with_f32_floor(&geo, f32_floor_note);

    for round in 0..rounds {
        let a64 = arael_runner3::run_f64(&ds);
        t.record("arael LM f64", (a64.solve_ms, a64.first_iter_ms, a64.iterations, Some(a64.accepted), a64.two_iter_ms, a64.poses));
        let a32 = arael_runner3::run_f32(&ds);
        t.record("arael LM f32", (a32.solve_ms, a32.first_iter_ms, a32.iterations, Some(a32.accepted), a32.two_iter_ms, a32.poses));
        if !skip_tiny() {
            let tgn = tiny_runner3::run_gn(&ds);
            t.record("tiny-solver GN", (tgn.solve_ms, tgn.first_iter_ms, tgn.iterations, None, None, tgn.poses));
            let tlm = tiny_runner3::run_lm(&ds);
            t.record("tiny-solver LM", (tlm.solve_ms, tlm.first_iter_ms, tlm.iterations, None, None, tlm.poses));
        }
        let fgn = factrs_runner3::run_gn(&ds);
        t.record("factrs GN", (fgn.solve_ms, fgn.first_iter_ms, fgn.iterations, Some(fgn.accepted), fgn.two_iter_ms, fgn.poses));
        let flm = factrs_runner3::run_lm(&ds);
        t.record("factrs LM", (flm.solve_ms, flm.first_iter_ms, flm.iterations, Some(flm.accepted), flm.two_iter_ms, flm.poses));
        if factrs32_3d_available {
            for (kind, label) in [("gn", "factrs GN f32"), ("lm", "factrs LM f32")] {
                match run_factrs32_3d(path, kind, ds.poses.len()) {
                    Some((ms, fi, it, acc, full, poses)) =>
                        t.record(label, (ms, fi, it, acc, full, poses)),
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
            let (ms, fi, it, acc, full, poses) = run_external3_checked(
                std::process::Command::new("cpp/build/ceres_bench3"),
                &[path, "/tmp/ceres3_poses.txt"], "/tmp/ceres3_poses.txt",
                ds.poses.len(), Some("initial_cost"), &check);
            t.record("ceres LM", (ms, fi, it, acc, full, poses));
            for kind in ["lm", "gn"] {
                let poses_out = format!("/tmp/g2o3_poses_{}.txt", kind);
                let mut g2o_cmd = std::process::Command::new("cpp/build/g2o_bench3");
                g2o_cmd.env("G2O_LAMBDA_INIT", g2o_lambda(name));
                let (ms, fi, it, acc, full, poses) = run_external3_checked(
                    g2o_cmd,
                    &[path, kind, &poses_out], &poses_out,
                    ds.poses.len(), Some("initial_chi2"), &check);
                t.record(&format!("g2o {}", kind.to_uppercase()), (ms, fi, it, acc, full, poses));
            }
            if symforce3_available {
                for prec in ["f64", "f32"] {
                    let poses_out = format!("/tmp/symforce3_poses_{}.txt", prec);
                    let (ms, fi, it, acc, full, poses) = run_external3_checked(
                        std::process::Command::new("cpp/build/symforce_bench3"),
                        &[path, prec, &poses_out], &poses_out,
                        ds.poses.len(), Some("initial_cost"), &check);
                    let label = if prec == "f32" { "symforce LM f32" } else { "symforce LM" };
                    t.record(label, (ms, fi, it, acc, full, poses));
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
                let (ms, fi, it, acc, full, poses) = run_external3_checked(
                    std::process::Command::new(&py),
                    &["gtsam_bench.py", path, kind, &poses_out], &poses_out,
                    ds.poses.len(), None, &|_| {});
                t.record(&format!("gtsam {}", kind.to_uppercase()), (ms, fi, it, acc, full, poses));
            }
        }
        eprintln!("  round {}/{} done", round + 1, rounds);
    }

    t.print();
}

fn skip_tiny() -> bool {
    std::env::var("RUN_TINY").is_err()
}

/// SKIP_ISAM=1 drops the GTSAM ISAM2 row: it is slow, and it answers the
/// incremental question rather than the batch one this benchmark asks.
fn skip_isam() -> bool {
    std::env::var("SKIP_ISAM").is_ok()
}

// The run's configuration, printed at startup so a pasted result carries the
// settings that produced it. Values are read back from the actual config
// objects and env accessors the runners use, so the header cannot drift from
// what was run; the env var that changes each one is named in brackets.
fn print_header(rounds: usize, only: &Option<String>) {
    let c64 = arael_pipeline::config::<arael_runner::Graph>(0);
    let nielsen = arael_pipeline::nielsen();
    let ordering = arael_runner::ordering();

    println!("=== pgo-bench configuration ===");
    println!("rounds            : {} [ROUNDS], {} probe sub-rounds per round",
        rounds, probe::PROBE_SUBROUNDS);
    println!("datasets          : {} [PGO_ONLY]",
        only.as_deref().unwrap_or("all"));
    let on_off = |skipped: bool| if skipped { "skipped" } else { "run" };
    println!("optional systems  : tiny-solver {} [RUN_TINY], isam {} [SKIP_ISAM]",
        on_off(skip_tiny()), on_off(skip_isam()));
    println!("arael lambda0     : {:e} (2D), {:e} (3D) [ARAEL_LAMBDA0]",
        arael_pipeline::lambda0::<arael_runner::Graph>(),
        arael_pipeline::lambda0::<arael_runner3::Graph3>());
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
        std::env::var("PGO_BENCH_CORE").unwrap_or_else(|_| "?".to_string()));
}

fn main() {
    enforce_single_core();
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
        let mut t = table::Table::new(&geo);

        for round in 0..rounds {
            let a64 = arael_runner::run_f64(&ds);
            t.record("arael LM f64", (a64.solve_ms, a64.first_iter_ms, a64.iterations, Some(a64.accepted), a64.two_iter_ms, a64.poses));
            let a32 = arael_runner::run_f32(&ds);
            t.record("arael LM f32", (a32.solve_ms, a32.first_iter_ms, a32.iterations, Some(a32.accepted), a32.two_iter_ms, a32.poses));
            if !skip_tiny() {
                let tgn = tiny_runner::run_gn(&ds);
                t.record("tiny-solver GN", (tgn.solve_ms, tgn.first_iter_ms, tgn.iterations, None, None, tgn.poses));
                let tlm = tiny_runner::run_lm(&ds);
                t.record("tiny-solver LM", (tlm.solve_ms, tlm.first_iter_ms, tlm.iterations, None, None, tlm.poses));
            }
            let fgn = factrs_runner::run_gn(&ds);
            t.record("factrs GN", (fgn.solve_ms, fgn.first_iter_ms, fgn.iterations,
                Some(fgn.accepted), fgn.two_iter_ms, fgn.poses));
            let flm = factrs_runner::run_lm(&ds);
            t.record("factrs LM", (flm.solve_ms, flm.first_iter_ms, flm.iterations,
                Some(flm.accepted), flm.two_iter_ms, flm.poses));
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
