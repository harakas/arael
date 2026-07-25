// End-to-end parity: build the fixture problem through the GENERATED
// C++ interface (compiled and linked against the capi staticlib), and
// through the model crate directly in Rust; the two paths run the same
// deterministic solver, so every printed value must round-trip
// EXACTLY. Skipped with a note when no C++ compiler is available.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem, LmStatus, SolveFailureKind};
use cxx_fit::{Fit, N, Obs};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fill(fit: &mut Fit) {
    for i in 0..6 {
        let x = i as f64;
        fit.obs.push(Obs { x, y: 2.0 * x + 1.0 + if i % 2 == 0 { 0.05 } else { -0.05 } });
    }
    let t = [1.5, -0.3, 0.7];
    let w = [1.0, 2.0, 0.5];
    for i in 0..3 {
        fit.items.push(N {
            v: Param::default(),
            t: t[i],
            w: w[i],
            hb: SelfBlock::new(),
        });
    }
}

/// The shim's status mapping (kept in lockstep with emit_ffi).
fn code(s: &LmStatus) -> f64 {
    (match s {
        LmStatus::Converged => 0,
        LmStatus::CostThreshold => 1,
        LmStatus::MaxIterations => 2,
        LmStatus::GradientTolerance => 3,
        LmStatus::ParameterTolerance => 4,
        LmStatus::PredictedReduction => 5,
        LmStatus::LambdaCeiling => 6,
        LmStatus::DriverTerminated => 7,
        LmStatus::ObserverTerminated => 8,
        LmStatus::TimeLimit => 9,
        LmStatus::RetryBudgetExhausted => 10,
        LmStatus::Aborted => 11,
    }) as f64
}

fn find_compiler() -> Option<&'static str> {
    for cc in ["c++", "g++", "clang++"] {
        if Command::new(cc).arg("--version")
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status().map(|s| s.success()).unwrap_or(false)
        {
            return Some(cc);
        }
    }
    None
}

#[test]
fn cxx_interface_matches_rust_exactly() {
    let Some(cc) = find_compiler() else {
        eprintln!("cxx parity: no C++ compiler found, skipping");
        return;
    };
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    // The staticlib the C++ program links.
    let status = Command::new("cargo")
        .args(["build", "-p", "cxx-fit-capi", "--release"])
        .current_dir(&ws)
        .status().expect("cargo spawn");
    assert!(status.success(), "capi build failed");

    let bin = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cxx_parity");
    let status = Command::new(cc)
        .arg("-std=c++17").arg("-O2")
        .arg("-I").arg(ws.join("model/cxx/include"))
        .arg(ws.join("runner/tests/parity_main.cpp"))
        .arg(ws.join("target/release/libcxx_fit_capi.a"))
        .arg("-lpthread").arg("-ldl").arg("-lm")
        .arg("-o").arg(&bin)
        .status().expect("compiler spawn");
    assert!(status.success(), "C++ compile/link failed");

    let out = Command::new(&bin).output().expect("run");
    assert!(out.status.success(), "C++ run failed: {}",
        String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let mut got: HashMap<String, f64> = HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if let (Some(n), Some(v)) = (it.next(), it.next()) {
            got.insert(n.to_string(), v.parse().unwrap());
        }
    }
    let g = |n: &str| *got.get(n).unwrap_or_else(|| panic!("C++ output missing `{n}`"));

    // The same problem in Rust; every C++ value must match it exactly
    // (deterministic same code path; %.17e round-trips f64 exactly).
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let mut fit = Fit::default();
    fill(&mut fit);
    assert!(fit.validate().is_clean());
    assert_eq!(g("clean"), 1.0);
    assert_eq!(g("n_obs"), 6.0);
    assert_eq!(g("n_items"), 3.0);
    assert_eq!(g("obs3_y"), fit.obs[3].y);
    assert_eq!(g("item1_t"), fit.items[1].t);

    let r = fit.solve_dense(&cfg).unwrap();
    assert!(r.status.is_success(), "{:?}", r.status);
    assert_eq!(g("dense_status"), code(&r.status));
    assert_eq!(g("dense_start"), r.start_cost);
    assert_eq!(g("dense_end"), r.end_cost);
    assert_eq!(g("dense_iters"), r.iterations as f64);
    assert_eq!(g("dense_m"), fit.m.value);
    assert_eq!(g("dense_c"), fit.c.value);
    for i in 0..3 {
        assert_eq!(g(&format!("dense_v{i}")), fit.items[i].v.value, "v{i}");
    }
    // And the solution is the right one (least squares of the +-0.05
    // pattern keeps m at 2, c near 1).
    assert!((fit.m.value - 2.0).abs() < 0.05, "m {}", fit.m.value);
    assert!((fit.c.value - 1.0).abs() < 0.1, "c {}", fit.c.value);

    let mut fit2 = Fit::default();
    fill(&mut fit2);
    let r2 = fit2.solve_sparse(&cfg).unwrap();
    assert_eq!(g("sparse_status"), code(&r2.status));
    assert_eq!(g("sparse_end"), r2.end_cost);
    assert_eq!(g("sparse_m"), fit2.m.value);
    assert_eq!(g("sparse_c"), fit2.c.value);

    // The degenerate model (unconstrained root params, nonzero cost)
    // fails with DegenerateDiagonal in Rust and status -1 + text in C++.
    let mut bad = Fit::default();
    bad.items.push(N { v: Param::default(), t: 1.0, w: 1.0, hb: SelfBlock::new() });
    let e = bad.solve_dense(&cfg).expect_err("degenerate must fail");
    assert!(matches!(e.kind, SolveFailureKind::DegenerateDiagonal { .. }),
        "{:?}", e.kind);
    assert_eq!(g("bad_status"), -1.0);
    assert_eq!(g("bad_has_error"), 1.0);
}

#[test]
fn generated_tree_is_current() {
    // `cargo arael check` as a test: the committed generated files must
    // match what the tool generates from the model today.
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let status = Command::new("cargo")
        .args(["run", "-q", "-p", "cargo-arael", "--manifest-path"])
        .arg(ws.join("../Cargo.toml"))
        .args(["--", "check", "--manifest-dir"])
        .arg(ws.join("model"))
        .status().expect("cargo spawn");
    assert!(status.success(), "generated tree is stale -- rerun `cargo arael export`");
}
