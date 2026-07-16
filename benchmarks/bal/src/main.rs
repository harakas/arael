// BAL bundle-adjustment benchmark: arael vs Ceres and g2o, on the shared
// benchmark harness (benchmarks/harness) -- the same probes, timing rules, table
// and core pin as pgo, slam and loc. What is local to this benchmark is its
// problem: the BAL loader, the Snavely reprojection cost every system is scored
// by, and the 7-DOF gauge that makes its validation gate a relative one.

mod arael_runner;
mod bal;

use std::rc::Rc;

use arael::vect::vect3d;
use arael_runner::{Problem, Route, Solution};
use bal::Dataset;

/// The datasets, and the initial damping each one wants.
///
/// Damping is per-problem in every one of these benchmarks (see the README):
/// 1e-3 on Ladybug-138, where 1e-4 plateaus 1.4% high, and 1e-4 on the larger
/// ones. Ladybug-49 takes 5e-5: at 1e-5 its very first step overshoots the cost
/// from 1.7e6 to 2.5e16 and is rejected, which costs a factorization, denies the
/// first iteration any meaning, and is slower overall than damping that lands the
/// step. ARAEL_LAMBDA0 overrides.
///
/// `dense_schur` is Ceres's own guidance: only while the camera count keeps the
/// dense Schur complement small (a few hundred cameras).
struct Bench {
    name: &'static str,
    path: &'static str,
    dense_schur: bool,
    lambda0: f64,
    /// Ceres's initial trust-region radius: the same knob as arael's lambda under
    /// another name, and per-dataset for the same reason. Ceres ships 1e4, which
    /// is right on Ladybug-49. On Ladybug-138 it gets the SECOND step of every
    /// Ceres row rejected, so no iteration can be measured there -- and 1e2 is
    /// faster besides, and lets iterative_schur converge at all (at 1e4 it stops
    /// 60% above the optimum). CERES_RADIUS0 overrides.
    ceres_radius0: f64,
}

const CERES_SHIPPED_RADIUS0: f64 = 1e4;

const DATASETS: [Bench; 3] = [
    Bench { name: "Ladybug-49", path: "datasets/problem-49-7776-pre.txt",
            dense_schur: true, lambda0: 5e-5, ceres_radius0: CERES_SHIPPED_RADIUS0 },
    Bench { name: "Ladybug-138", path: "datasets/problem-138-19878-pre.txt",
            dense_schur: true, lambda0: 1e-3, ceres_radius0: 1e2 },
    Bench { name: "Ladybug-372", path: "datasets/problem-372-47423-pre.txt",
            dense_schur: true, lambda0: 1e-4, ceres_radius0: CERES_SHIPPED_RADIUS0 },
];

/// The radius for one Ceres ROW. Ceres's three linear solvers are given the
/// damping each of them wants, the same courtesy every other system gets:
/// iterative_schur takes inexact (CG) steps, so the trust region interacts with
/// it differently, and on Ladybug-372 the shipped radius leaves its second step
/// rejected while 1e2 both measures cleanly and runs 30% faster. The exact rows
/// keep the shipped radius there, where it is already clean and reaches a lower
/// cost.
fn ceres_radius0(b: &Bench, linsolver: &str) -> f64 {
    match (b.name, linsolver) {
        ("Ladybug-372", "iterative_schur") => 1e2,
        _ => b.ceres_radius0,
    }
}

/// Exploratory (BAL_ONLY=1723). It is not a result yet: with the shared 1e-5
/// tolerances no system converges here, so the mutual validation gate cannot
/// pass, and arael's f32 assembly overflows to a NaN Hessian diagonal at 485k
/// parameters whatever the damping. What the damping below DOES buy is a
/// measurable iteration -- the first two steps land, so full-iter exists.
///
/// Both values come from BAL_LAMBDAS (see `lambda_sweep`). 485k parameters need
/// far heavier damping than the small problems: arael's second step is rejected
/// at anything below 5e-2 (1e-1 keeps a margin), and Ceres's second step is
/// rejected at its shipped 1e4 and at 1e2, landing at 1e1 for both of its rows.
/// g2o keeps its own auto-lambda heuristic, which already lands both steps.
const EXPLORATORY: [Bench; 1] = [
    Bench { name: "Ladybug-1723", path: "datasets/problem-1723-156502-pre.txt",
            dense_schur: false, lambda0: 1e-1, ceres_radius0: 1e1 },
];

/// The arael rows: the same model, factorized four ways.
fn routes() -> Vec<(Route, bool)> {
    #[allow(unused_mut)] // only the cholmod-gpl build pushes
    let mut r = vec![
        (Route::Sparse, false),
        (Route::Sparse, true),
        (Route::Schur, false),
        (Route::Schur, true),
    ];
    #[cfg(feature = "cholmod-gpl")]
    r.push((Route::CholmodGpl, false));
    r
}

fn label_of(route: Route, f32_row: bool) -> String {
    route.label(if f32_row { "f32" } else { "f64" })
}

/// The settings this run used, printed before anything else: a pasted result has
/// to carry them. Values come from the objects the run actually uses.
fn print_header(rounds: usize, only: &Option<String>, systems: &Option<String>,
                probe: &Problem) {
    use bench_harness::header::Header;
    let cfg = bench_harness::arael::config::<arael_runner::Scene>(probe, 0);
    Header::new("bal-bench")
        .rounds(rounds)
        .line("datasets", format!("{} [BAL_ONLY]", only.as_deref().unwrap_or("all")))
        .line("systems", format!("{} [BAL_SYSTEMS]", systems.as_deref().unwrap_or("all")))
        .line("arael lambda0", match std::env::var("ARAEL_LAMBDA0") {
            Ok(v) => format!("{} -- ARAEL_LAMBDA0 overrides every dataset", v),
            Err(_) => format!("per dataset: {} [ARAEL_LAMBDA0]",
                DATASETS.iter().chain(EXPLORATORY.iter())
                    .map(|b| format!("{} {:e}", b.name, b.lambda0))
                    .collect::<Vec<_>>().join(", ")),
        })
        .line("ceres radius0", match std::env::var("CERES_RADIUS0") {
            Ok(v) => format!("{} -- CERES_RADIUS0 overrides every row", v),
            Err(_) => format!("per dataset: {} (Ladybug-372 iterative_schur 1e2) [CERES_RADIUS0]",
                DATASETS.iter().chain(EXPLORATORY.iter())
                    .map(|b| format!("{} {:e}", b.name, b.ceres_radius0))
                    .collect::<Vec<_>>().join(", ")),
        })
        .line("arael damping", format!("{} [DRIVER: nielsen|fixed]",
            if bench_harness::arael::nielsen::<arael_runner::Scene>() {
                "Nielsen gain-ratio driver (the default here)"
            } else {
                "fixed ladder"
            }))
        .line("arael schur ordering", format!("{} [BAL_ORDERING: nd|amd]",
            if std::env::var("BAL_ORDERING").as_deref() == Ok("amd") { "AMD" }
            else { "nested dissection" }))
        .line("arael termination", format!("abs {:e}, rel {:e}, patience {}, min_iters {}",
            cfg.abs_precision, cfg.rel_precision, cfg.patience, cfg.min_iters))
        .line("solver verbose", format!("{} [VERBOSE], per-solve timing {} [TIMING]",
            if cfg.verbose { "on" } else { "off" },
            if std::env::var("TIMING").is_ok() { "on" } else { "off" }))
        .line("memory pass", format!("{} [BAL_NO_MEM]",
            if std::env::var("BAL_NO_MEM").is_err() { "on" } else { "off" }))
        .core()
        .print();
}

// Peak memory for the arael rows.
//
// VmHWM is a high-water mark for the whole PROCESS, so the arael rows sharing one
// would each report the largest peak anything before them reached. Each is
// therefore run alone, in a process of its own, doing nothing but the solve. The
// subprocess systems (Ceres, g2o) report their own.
fn mem_pass() -> bool {
    let Ok(which) = std::env::var("BAL_MEM") else { return false };
    let path = std::env::var("BAL_MEM_DATASET").expect("BAL_MEM_DATASET");
    let lambda0: f64 = std::env::var("BAL_MEM_LAMBDA0").expect("BAL_MEM_LAMBDA0")
        .parse().expect("BAL_MEM_LAMBDA0");
    let iters: usize = std::env::var("BAL_MEM_ITERS").ok()
        .and_then(|v| v.parse().ok()).unwrap_or(3);
    let ds = Rc::new(bal::load(&path));
    for (route, f32_row) in routes() {
        if label_of(route, f32_row) != which {
            continue;
        }
        let p = Problem { ds, lambda0, route };
        if f32_row {
            std::hint::black_box(arael_runner::run_f32_capped(&p, iters));
        } else {
            std::hint::black_box(arael_runner::run_f64_capped(&p, iters));
        }
        bench_harness::mem::report_peak();
        return true;
    }
    panic!("unknown system for the memory pass: {}", which);
}

/// Ceres's linear solvers for one dataset. dense_schur only while the camera
/// count keeps the dense Schur complement small (Ceres's own guidance).
fn ceres_linsolvers(b: &Bench) -> &'static [&'static str] {
    if b.dense_schur {
        &["dense_schur", "sparse_schur", "iterative_schur"]
    } else {
        &["sparse_schur", "iterative_schur"]
    }
}

fn bench_dir() -> std::path::PathBuf {
    std::env::current_dir().unwrap()
}

/// BAL_LAMBDAS=<comma-separated values>: a damping sweep, and nothing else.
///
/// For each selected row and each value it runs ONE solve capped at 1 iteration
/// and one capped at 2 -- no warmup, no sub-rounds, no full solve, no memory pass
/// -- and reports whether those two iterations were clean. That is all the
/// benchmark's per-iteration number needs (full-iter is t(2) - t(1), reported only
/// when the first iteration was a single accepted step), and on the big problem it
/// is the difference between seconds and hours.
///
///   BAL_ONLY=1723 BAL_SYSTEMS="f64 schur" BAL_LAMBDAS=1e-4,1e-2,1 cargo run -r
///
/// The values are arael's initial lambda. For the external rows the same sweep
/// runs over their own damping knob -- Ceres's trust-region radius, g2o's initial
/// lambda -- with BENCH_MAX_ITERS holding their solve to the two iterations too.
fn lambda_sweep(b: &Bench, ds: &Rc<Dataset>, lambdas: &[f64],
                want: impl Fn(&str) -> bool) {
    println!("\n{:<24} {:>9} {:>7} {:>5} {:>8} {:>5} {:>10} {:>15}",
        "system", "damping", "t(1)", "acc/att", "t(2)", "acc", "full-iter", "cost after 2");
    for (route, f32_row) in routes() {
        let label = label_of(route, f32_row);
        if !want(&label) {
            continue;
        }
        for &lambda0 in lambdas {
            let p = Problem { ds: Rc::clone(ds), lambda0, route };
            let one = if f32_row { arael_runner::probe_f32(&p, 1) }
                      else { arael_runner::probe_f64(&p, 1) };
            let two = if f32_row { arael_runner::probe_f32(&p, 2) }
                      else { arael_runner::probe_f64(&p, 2) };
            report_sweep(&label, lambda0, (one.ms, one.accepted, one.attempts),
                (two.ms, two.accepted),
                bal::reference_cost(ds, &two.solution.cameras, &two.solution.points));
        }
    }
    // The external rows, on their own damping knob (Ceres's trust-region radius,
    // g2o's initial lambda), held to the same two iterations by BENCH_QUICK.
    let path = bench_dir().join(b.path);
    let path = path.to_str().unwrap();
    if std::path::Path::new("cpp/build/ceres_bal").exists() {
        for linsolver in ceres_linsolvers(b) {
            let label = format!("ceres {}", linsolver);
            if !want(&label) {
                continue;
            }
            for &radius0 in lambdas {
                let params_out = format!("/tmp/ceres_bal_{}.txt", linsolver);
                let mut cmd = std::process::Command::new("cpp/build/ceres_bal");
                cmd.env("BENCH_QUICK", "1").env("CERES_RADIUS0", format!("{:e}", radius0));
                sweep_external(&label, radius0, cmd,
                    &[path, &params_out, linsolver], &params_out, ds);
            }
        }
    }
    if std::path::Path::new("cpp/build/g2o_bal").exists() && want("g2o LM (schur)") {
        for &lambda in lambdas {
            let params_out = "/tmp/g2o_bal.txt";
            let mut cmd = std::process::Command::new("cpp/build/g2o_bal");
            cmd.env("BENCH_QUICK", "1").env("G2O_LAMBDA_INIT", format!("{:e}", lambda));
            sweep_external("g2o LM (schur)", lambda, cmd,
                &[path, params_out], params_out, ds);
        }
    }
}

/// The external runner's own two probes, read off its protocol line.
fn sweep_external(label: &str, lambda: f64, mut cmd: std::process::Command,
                  args: &[&str], params_out: &str, ds: &Dataset) {
    cmd.args(args);
    let p = bench_harness::external::run(cmd);
    let get = |k: &str| p.json.get(k).and_then(|v| v.as_f64()).unwrap_or(0.0);
    let sol = read_solution(params_out, ds.cameras.len(), ds.points.len());
    report_sweep(label, lambda,
        (p.first_iter_ms, get("first_accepted") as usize, get("first_attempts") as usize),
        (get("second_run_ms"), get("second_accepted") as usize),
        bal::reference_cost(ds, &sol.cameras, &sol.points));
}

/// One row of the sweep. "clean" is the verdict that matters: it is what lets
/// full-iter exist at all.
fn report_sweep(label: &str, lambda: f64, one: (f64, usize, usize),
                two: (f64, usize), cost: f64) {
    let (ms1, acc1, att1) = one;
    let (ms2, acc2) = two;
    let clean = att1 == 1 && acc1 == 1 && acc2 >= 2;
    let full = if clean && ms2 > ms1 { format!("{:.0}", ms2 - ms1) } else { "-".into() };
    println!("{:<24} {:>9.0e} {:>7.0} {:>5} {:>8.0} {:>5} {:>10} {:>15.1}{}",
        label, lambda,
        ms1, format!("{}/{}", acc1, att1),
        ms2, format!("{}/-", acc2),
        full, cost,
        if clean { "" } else { "   <- first two iterations not clean" });
}

fn main() {
    bench_harness::pin::enforce_cores();
    if mem_pass() {
        return;
    }

    let bench_dir = std::env::current_dir().unwrap();
    let only = std::env::var("BAL_ONLY").ok();
    // BAL_SYSTEMS=<comma-separated substrings> runs only the matching rows (e.g.
    // BAL_SYSTEMS=schur). Unset runs everything. A filtered run validates only
    // against whatever ran -- for iterating, not publishing.
    let systems = std::env::var("BAL_SYSTEMS").ok();
    let want = |label: &str| -> bool {
        systems.as_deref().is_none_or(|f| f.split(',').any(|pat| label.contains(pat.trim())))
    };
    if systems.is_some() {
        eprintln!("BAL_SYSTEMS={} -- partial run, cross-system validation is not meaningful",
            systems.as_deref().unwrap_or(""));
    }
    let rounds: usize = std::env::var("ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
    let sweep: Option<Vec<f64>> = std::env::var("BAL_LAMBDAS").ok().map(|v| {
        v.split(',')
            .map(|t| t.trim().parse().unwrap_or_else(|_| panic!("bad BAL_LAMBDAS value: {}", t)))
            .collect()
    });

    let selected: Vec<&Bench> = DATASETS.iter()
        .chain(EXPLORATORY.iter().filter(|b| {
            only.as_deref().is_some_and(|f| b.name.contains(f))
        }))
        .filter(|b| only.as_deref().is_none_or(|f| b.name.contains(f)))
        .collect();

    // The header needs a Problem to read the config off; any one will do.
    let probe = Problem {
        ds: Rc::new(Dataset { cameras: Vec::new(), points: Vec::new(), observations: Vec::new() }),
        lambda0: selected.first().map_or(1e-4, |b| b.lambda0),
        route: Route::Sparse,
    };
    print_header(rounds, &only, &systems, &probe);

    if std::env::var("BAL_COV").is_ok() {
        cov_benchmark(&selected, &bench_dir);
        return;
    }

    let ceres_ok = std::path::Path::new("cpp/build/ceres_bal").exists();
    if !ceres_ok {
        eprintln!("WARNING: cpp/build/ceres_bal missing (cmake -B cpp/build cpp && cmake --build cpp/build); skipping Ceres");
    }
    let g2o_ok = std::path::Path::new("cpp/build/g2o_bal").exists();
    if !g2o_ok {
        eprintln!("WARNING: cpp/build/g2o_bal missing (needs g2o + cholmod); skipping g2o");
    }

    for b in &selected {
        let path_buf = bench_dir.join(b.path);
        if !path_buf.exists() {
            eprintln!("NOTE: {} missing ({}); run ./fetch_datasets.sh", b.name, b.path);
            continue;
        }
        let path = path_buf.to_str().unwrap();
        let ds = Rc::new(bal::load(path));
        println!("\n=== {} : {} cameras, {} points, {} observations, {} parameters ===",
            b.name, ds.cameras.len(), ds.points.len(), ds.observations.len(),
            ds.cameras.len() * 9 + ds.points.len() * 3);
        let initial_cost = bal::reference_cost(&ds, &ds.cameras, &ds.points);
        println!("initial reference cost: {:.3}", initial_cost);

        // BAL_LAMBDAS: the damping sweep instead of the benchmark.
        if let Some(lambdas) = &sweep {
            lambda_sweep(b, &ds, lambdas, &want);
            continue;
        }

        let geo = Geo(&ds);
        let mut t = bench_harness::table::Table::new(&geo);

        for _ in 0..rounds {
            for (route, f32_row) in routes() {
                let label = label_of(route, f32_row);
                if !want(&label) {
                    continue;
                }
                let p = Problem { ds: Rc::clone(&ds), lambda0: b.lambda0, route };
                let row = if f32_row {
                    arael_runner::run_f32(&p)
                } else {
                    arael_runner::run_f64(&p)
                };
                t.record(&label, row);
            }
            if ceres_ok {
                for linsolver in ceres_linsolvers(b) {
                    let label = format!("ceres {}", linsolver);
                    if !want(&label) {
                        continue;
                    }
                    let e = run_ceres(path, linsolver, ceres_radius0(b, linsolver), &ds);
                    check_initial(&label, e.initial_cost, initial_cost);
                    t.record(&label, e.row);
                }
            }
            if g2o_ok && want("g2o LM (schur)") {
                let e = run_g2o(path, &ds);
                check_initial("g2o LM (schur)", e.initial_cost, initial_cost);
                t.record("g2o LM (schur)", e.row);
            }
        }

        if std::env::var("BAL_NO_MEM").is_err() {
            let lambda0 = format!("{:e}", b.lambda0);
            for (route, f32_row) in routes() {
                let label = label_of(route, f32_row);
                if !want(&label) {
                    continue;
                }
                if let Some(mb) = bench_harness::mem::measure("BAL_MEM", &label, &[
                    ("BAL_MEM_DATASET", path),
                    ("BAL_MEM_LAMBDA0", lambda0.as_str()),
                ]) {
                    t.set_peak_mb(&label, mb);
                }
            }
        }
        t.print();
    }
}

/// An external system's cost at the initial estimate, as it computes it, must be
/// the reference cost -- the proof it minimizes the same objective.
fn check_initial(label: &str, reported: f64, reference: f64) {
    let rel = ((reported - reference) / reference).abs();
    assert!(rel < 1e-9, "{} initial cost {} vs reference {} (rel {:.2e})",
        label, reported, reference, rel);
}

struct Ext {
    row: bench_harness::table::Row<Solution>,
    initial_cost: f64,
}

/// One external runner: the harness parses its protocol line and asserts the core
/// pin; the cost cross-check stays here, because it is what proves the system
/// minimizes the same objective. Ceres reports a cost and g2o a chi2 -- with unit
/// information they are the same number, and both report it as `initial_cost`.
fn run_ext(mut cmd: std::process::Command, args: &[&str], params_out: &str,
           ds: &Dataset) -> Ext {
    cmd.args(args);
    let p = bench_harness::external::run(cmd);
    let solution = read_solution(params_out, ds.cameras.len(), ds.points.len());
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

fn run_ceres(path: &str, linsolver: &str, radius0: f64, ds: &Dataset) -> Ext {
    let params_out = format!("/tmp/ceres_bal_{}.txt", linsolver);
    let mut cmd = std::process::Command::new("cpp/build/ceres_bal");
    // The dataset's radius, unless the caller set one -- their override wins.
    if std::env::var("CERES_RADIUS0").is_err() {
        cmd.env("CERES_RADIUS0", format!("{:e}", radius0));
    }
    run_ext(cmd, &[path, &params_out, linsolver], &params_out, ds)
}

/// g2o's own BA configuration (its bal_example): marginalized point vertices ->
/// Schur elimination, CHOLMOD on the reduced camera system.
fn run_g2o(path: &str, ds: &Dataset) -> Ext {
    let params_out = "/tmp/g2o_bal.txt";
    run_ext(std::process::Command::new("cpp/build/g2o_bal"),
        &[path, params_out], params_out, ds)
}

// ---- Covariance-scaling benchmark (BAL_COV=1) ------------------------------

use bench_harness::cov::{fmt_cell, print_table, run_cov_cpp};

fn cov_benchmark(selected: &[&Bench], bench_dir: &std::path::Path) {
    let ceres = std::path::Path::new("cpp/build/ceres_bal").exists();
    let g2o = std::path::Path::new("cpp/build/g2o_bal").exists();
    let budget = std::env::var("COV_BUDGET_S").unwrap_or_else(|_| "5".into());
    let cap_s = bench_harness::cov::cell_cap_s();
    println!("\ncovariance scaling: 6-DOF camera pose + 3-DOF point marginals.");
    println!("gauge = cameras 0,1 fixed; intrinsics fixed (known calibration).");
    println!("arael and Ceres build cold (assemble + factor + query); g2o reuses its solve factor (warm).");
    println!("cells: median ms (reps); budget {budget}s [COV_BUDGET_S]; - not covered; * over the {cap_s:.0}s cap [COV_CELL_CAP_S].");
    if !ceres {
        eprintln!("WARNING: cpp/build/ceres_bal missing; skipping Ceres");
    }
    if !g2o {
        eprintln!("WARNING: cpp/build/g2o_bal missing; skipping g2o");
    }

    for b in selected {
        let path_buf = bench_dir.join(b.path);
        if !path_buf.exists() {
            eprintln!("NOTE: {} missing ({}); run ./fetch_datasets.sh", b.name, b.path);
            continue;
        }
        let path = path_buf.to_str().unwrap();
        let ds = Rc::new(bal::load(path));
        let p = Problem { ds: ds.clone(), lambda0: b.lambda0, route: Route::Schur };
        println!("\n=== {} : {} cameras, {} points ===", b.name, ds.cameras.len(), ds.points.len());

        let ar = arael_runner::cov_bench(&p);
        let cc = ceres.then(|| run_cov_cpp(std::process::Command::new("cpp/build/ceres_bal"), &[path, "cov"]));
        let gc = g2o.then(|| run_cov_cpp(std::process::Command::new("cpp/build/g2o_bal"), &[path, "cov"]));

        // Validation: camera[2] pose std dev, each system independently.
        let s = ar.sd_cam2;
        println!("  std dev camera[2]: arael  t=({:.4},{:.4},{:.4}) rot=({:.5},{:.5},{:.5})",
            s[0], s[1], s[2], s[3], s[4], s[5]);
        if let Some(l) = cc.as_ref().and_then(|c| c.stddev.as_ref()) {
            println!("                     {l}");
        }
        if let Some(l) = gc.as_ref().and_then(|c| c.stddev.as_ref()) {
            println!("                     {l}");
        }

        let ceres_cell = |ent: &str, n: usize| cc.as_ref().and_then(|c| c.cell(ent, n));
        let g2o_cell = |ent: &str, n: usize| gc.as_ref().and_then(|c| c.cell(ent, n));
        let all_marg = format!("{:.1} ({})", ar.allmarg_ms, ar.allmarg_reps);

        // Camera pose table. Columns are arael's query counts (1,2,8,32,all).
        let cam_ns: Vec<usize> = ar.perquery_cam.iter().map(|&(n, ..)| n).collect();
        let last = *cam_ns.last().unwrap();
        let cam_headers: Vec<String> =
            cam_ns.iter().map(|&n| if n == last { "all".into() } else { n.to_string() }).collect();
        println!("  camera pose (6-DOF):");
        print_table(&cam_headers, &[
            ("arael PerQuery", ar.perquery_cam.iter().map(|&(_, ms, r)| fmt_cell(Some(&(ms, r)))).collect()),
            ("arael AllMarginals", cam_ns.iter().map(|&n| if n == last { all_marg.clone() } else { "-".into() }).collect()),
            ("Ceres SPARSE_QR", cam_ns.iter().map(|&n| fmt_cell(ceres_cell("cam", n))).collect()),
            ("g2o Marginals", cam_ns.iter().map(|&n| fmt_cell(g2o_cell("cam", n))).collect()),
        ]);

        // Point table. Columns are arael's point query counts (1,2,8,32,all).
        let pt_ns: Vec<usize> = ar.perquery_point.iter().map(|&(n, ..)| n).collect();
        let pt_last = *pt_ns.last().unwrap();
        let pt_headers: Vec<String> =
            pt_ns.iter().map(|&n| if n == pt_last { "all".into() } else { n.to_string() }).collect();
        println!("  point (3-DOF):");
        print_table(&pt_headers, &[
            ("arael PerQuery", ar.perquery_point.iter().map(|&(_, ms, r)| fmt_cell(Some(&(ms, r)))).collect()),
            ("arael AllMarginals", pt_ns.iter().map(|&n| if n == pt_last { all_marg.clone() } else { "-".into() }).collect()),
            ("Ceres SPARSE_QR", pt_ns.iter().map(|&n| fmt_cell(ceres_cell("point", n))).collect()),
        ]);
        println!("  g2o marginalizes points -> camera poses only. AllMarginals covers all cameras AND points at once.");
    }
}

/// 9-value camera lines, then 3-value point lines.
fn read_solution(path: &str, n_cams: usize, n_points: usize) -> Solution {
    let text = std::fs::read_to_string(path).unwrap();
    let mut lines = text.lines();
    let mut cameras = Vec::with_capacity(n_cams);
    for _ in 0..n_cams {
        let v: Vec<f64> = lines.next().unwrap()
            .split_whitespace().map(|t| t.parse().unwrap()).collect();
        cameras.push(bal::CameraIn {
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
    Solution { cameras, points }
}

// The geometry the shared table is generic over.
struct Geo<'a>(&'a Dataset);

impl bench_harness::table::Geometry for Geo<'_> {
    type Solution = Solution;

    // Bundle adjustment has a 7-DOF gauge and BAL's units are arbitrary, so the
    // distance between two solutions is a SIMILARITY-ALIGNED, RELATIVE
    // camera-centre RMSE, not a distance in metres.
    //
    // Converged solutions scatter up to ~1.3e-3 around the deepest stop while
    // agreeing in cost to 0.13% (the BA valley is that flat); measured
    // non-converged plateaus sit at 5.8e-3 and beyond, and fail the 1% cost gate
    // anyway.
    const DISTANCE_GATE: f64 = 5e-3;
    const DISTANCE_GATE_NAME: &'static str =
        "similarity-aligned relative camera-centre RMSE < 5e-3";

    fn cost(&self, s: &Solution) -> f64 {
        bal::reference_cost(self.0, &s.cameras, &s.points)
    }

    fn distance(a: &Solution, b: &Solution) -> f64 {
        bal::aligned_relative_rmse(&bal::camera_centers(&a.cameras),
                                   &bal::camera_centers(&b.cameras))
    }
}
