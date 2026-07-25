// Multi-root export parity: both roots of the cxx-mr fixture solved
// from ONE C++ translation unit (nested namespaces, one capi
// staticlib), compared exactly against the same solves in Rust.

use arael::model::{Param, SelfBlock};
use arael::simple_lm::{LmConfig, LmProblem, LmStatus};
use cxx_mr::{Cell, Decay, Line, Ob};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

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
fn both_roots_from_one_translation_unit() {
    let Some(cc) = find_compiler() else {
        eprintln!("cxx multiroot: no C++ compiler found, skipping");
        return;
    };
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");

    let status = Command::new("cargo")
        .args(["build", "-p", "cxx-mr-capi", "--release"])
        .current_dir(&ws)
        .status().expect("cargo spawn");
    assert!(status.success(), "capi build failed");

    let bin = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cxx_multiroot");
    let status = Command::new(cc)
        .arg("-std=c++17").arg("-O2").arg("-ffp-contract=off")
        .arg("-I").arg(ws.join("mr/cxx/include"))
        .arg(ws.join("runner/tests/multiroot_main.cpp"))
        .arg(ws.join("target/release/libcxx_mr_capi.a"))
        .arg("-lpthread").arg("-ldl").arg("-lm")
        .arg("-o").arg(&bin)
        .status().expect("compiler spawn");
    assert!(status.success(), "C++ compile/link failed");

    let out = Command::new(&bin).output().expect("run");
    assert!(out.status.success(), "C++ run failed: {}",
        String::from_utf8_lossy(&out.stderr));
    let mut got: HashMap<String, f64> = HashMap::new();
    for line in String::from_utf8(out.stdout).unwrap().lines() {
        let mut it = line.split_whitespace();
        if let (Some(n), Some(v)) = (it.next(), it.next()) {
            got.insert(n.to_string(), v.parse().unwrap());
        }
    }
    verify_mr(&got);
}

/// The Rust mirror both language twins are compared against.
fn verify_mr(got: &HashMap<String, f64>) {
    let g = |n: &str| *got.get(n).unwrap_or_else(|| panic!("output missing `{n}`"));

    // Root A (f64) mirrored.
    let mut line = Line::default();
    for i in 1..=4 {
        let x = i as f64;
        line.obs.push(Ob { x, y: 3.0 * x + if i % 2 == 0 { 0.25 } else { -0.25 } });
    }
    let lc = LmConfig::<f64> { max_iters: 30, ..Default::default() };
    let lr = line.solve_dense(&lc).unwrap();
    assert!(lr.status.is_success(), "{:?}", lr.status);
    assert_eq!(g("line_status"), code(&lr.status));
    assert_eq!(g("line_end"), lr.end_cost);
    assert_eq!(g("line_k"), line.k.value);
    assert!((line.k.value - 3.0).abs() < 0.1);

    // Root B (f32) mirrored.
    let mut decay = Decay::default();
    for (i, t) in [0.5f32, -1.5, 2.0].into_iter().enumerate() {
        decay.cells.push(Cell {
            v: Param::default(),
            t,
            w: 1.0 + i as f32,
            hb: SelfBlock::new(),
        });
    }
    let dc = LmConfig::<f32> { max_iters: 30, ..Default::default() };
    let dr = decay.solve_dense(&dc).unwrap();
    assert!(dr.status.is_success(), "{:?}", dr.status);
    assert_eq!(g("decay_status"), code(&dr.status));
    assert_eq!(g("decay_end"), dr.end_cost as f64);
    for i in 0..3 {
        let r = decay.cells.ref_at(i);
        assert_eq!(g(&format!("cell{i}")), decay.cells[r].v.value as f64);
    }
}

#[test]
fn both_roots_from_one_python_interpreter() {
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    if !Command::new("python3").arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false)
    {
        eprintln!("python multiroot: no python3 found, skipping");
        return;
    }
    let status = Command::new("cargo")
        .args(["build", "-p", "cxx-mr-capi", "--release"])
        .current_dir(&ws)
        .status().expect("cargo spawn");
    assert!(status.success(), "capi build failed");

    let out = Command::new("python3")
        .arg(ws.join("runner/tests/multiroot_main.py"))
        .env("ARAEL_CAPI", ws.join("target/release/libcxx_mr_capi.so"))
        .output().expect("python spawn");
    assert!(out.status.success(), "python run failed: {}",
        String::from_utf8_lossy(&out.stderr));
    let mut got: HashMap<String, f64> = HashMap::new();
    for line in String::from_utf8(out.stdout).unwrap().lines() {
        let mut it = line.split_whitespace();
        if let (Some(n), Some(v)) = (it.next(), it.next()) {
            got.insert(n.to_string(), v.parse().unwrap());
        }
    }
    verify_mr(&got);
}

#[test]
fn mr_generated_tree_is_current() {
    // `cargo arael check` on the multi-root fixture.
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let status = Command::new("cargo")
        .args(["run", "-q", "-p", "cargo-arael", "--manifest-path"])
        .arg(ws.join("../Cargo.toml"))
        .args(["--", "check", "--manifest-dir"])
        .arg(ws.join("mr"))
        .status().expect("cargo spawn");
    assert!(status.success(), "mr generated tree is stale -- rerun `cargo arael export`");
}
