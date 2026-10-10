// The generated wasm-bindgen crates: the fixture's and the multi-root
// fixture's compile for wasm32-unknown-unknown (a type check, needs the
// target and no CLI), and the fixture's runs under node and solves the
// fixture problem to the same result as Rust, exactly (needs the
// wasm-bindgen CLI at the version the crate pins, and node). Each part
// is skipped with a note when its tools are missing.
#[path = "parity_verify.rs"]
mod parity_verify;

use arael::simple_lm::{LmConfig, LmProblem, LmStatus, RootProblem};
use cxx_fit::Fit;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

fn ws() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

fn wasm_target_installed() -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).lines().any(|l| l.trim() == "wasm32-unknown-unknown"))
        .unwrap_or(false)
}

/// The version the generated manifest pins, from the manifest itself.
fn pinned_cli_version(wasm_dir: &Path) -> Option<String> {
    let text = std::fs::read_to_string(wasm_dir.join("Cargo.toml")).ok()?;
    text.lines()
        .find(|l| l.starts_with("wasm-bindgen = "))
        .and_then(|l| l.split('"').nth(1))
        .map(|v| v.trim_start_matches('=').to_string())
}

fn cli_version() -> Option<String> {
    let out = Command::new("wasm-bindgen").arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    text.split_whitespace().nth(1).map(str::to_string)
}

fn node_present() -> bool {
    Command::new("node").arg("--version")
        .stdout(std::process::Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false)
}

fn check(dir: &Path) {
    let status = Command::new("cargo")
        .args(["check", "--target", "wasm32-unknown-unknown"])
        .current_dir(dir)
        .status()
        .expect("cargo spawn");
    assert!(status.success(), "{} does not compile for wasm32", dir.display());
}

#[test]
fn generated_wasm_crates_compile_for_wasm32() {
    if !wasm_target_installed() {
        eprintln!("wasm: the wasm32-unknown-unknown target is not installed, skipping");
        return;
    }
    check(&ws().join("model/wasm"));
    check(&ws().join("mr/wasm"));
}

#[test]
fn wasm_interface_matches_rust_exactly() {
    if !wasm_target_installed() {
        eprintln!("wasm parity: the wasm32-unknown-unknown target is not installed, skipping");
        return;
    }
    let wasm_dir = ws().join("model/wasm");
    let pinned = pinned_cli_version(&wasm_dir).expect("the wasm manifest pins wasm-bindgen");
    match cli_version() {
        Some(v) if v == pinned => {}
        Some(v) => {
            eprintln!("wasm parity: wasm-bindgen CLI {v} installed, the crate pins {pinned}, skipping");
            return;
        }
        None => {
            eprintln!("wasm parity: no wasm-bindgen CLI, skipping");
            return;
        }
    }
    if !node_present() {
        eprintln!("wasm parity: no node, skipping");
        return;
    }
    let status = Command::new("cargo")
        .args(["build", "--release", "--target", "wasm32-unknown-unknown"])
        .current_dir(&wasm_dir)
        .status().expect("cargo spawn");
    assert!(status.success(), "wasm build failed");
    let pkg = Path::new(env!("CARGO_TARGET_TMPDIR")).join("wasm_parity_pkg");
    let status = Command::new("wasm-bindgen")
        .args(["--target", "nodejs", "--out-dir"]).arg(&pkg)
        .arg(wasm_dir.join("target/wasm32-unknown-unknown/release/cxx_fit_wasm.wasm"))
        .status().expect("wasm-bindgen spawn");
    assert!(status.success(), "wasm-bindgen failed");
    let out = Command::new("node")
        .arg(ws().join("runner/tests/wasm_parity.mjs"))
        .arg(&pkg)
        .output().expect("node spawn");
    assert!(out.status.success(), "node run failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let mut got: HashMap<String, f64> = HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        if let (Some(n), Some(v)) = (it.next(), it.next()) {
            got.insert(n.to_string(), v.parse().expect("a number"));
        }
    }
    let g = |n: &str| *got.get(n).unwrap_or_else(|| panic!("output missing `{n}`"));

    let mut fit = Fit::default();
    parity_verify::fill(&mut fit);
    assert_eq!(g("clean"), 1.0);
    parity_verify::version_matches(&g);
    assert_eq!(g("n_obs"), 6.0);
    assert_eq!(g("n_items"), 3.0);
    assert_eq!(g("obs3_y"), fit.obs[3].y);
    assert_eq!(g("vn_h11"), fit.vns[0].h.rows[1].e[1]);
    assert_eq!(g("vn_v2"), fit.vns[0].v.value.e[2]);
    let mut params = Vec::new();
    fit.serialize(&mut params);
    assert_eq!(g("initial_cost"), fit.calc_cost(&params));
    let cfg = LmConfig { max_iters: 50, ..Default::default() };
    let r = fit.solve_dense(&cfg).unwrap();
    assert_eq!(g("status"), parity_verify::code(&r.status));
    assert_eq!(g("status_named"), (r.status == LmStatus::Converged) as u8 as f64);
    assert_eq!(g("status_text_len"), r.status.as_str().len() as f64);
    assert_eq!(g("th_sweeps_asked"), r.threads.sweeps_asked as f64);
    assert_eq!(g("th_linear"), r.threads.linear as f64);
    assert_eq!(g("th_fell_back"), r.threads.fell_back() as u8 as f64);
    assert_eq!(g("th_has_sweeps"), r.threads.sweeps.is_some() as u8 as f64);
    assert_eq!(g("enum_schur_force"), 1.0);
    assert_eq!(g("iterations"), r.iterations as f64);
    assert_eq!(g("start_cost"), r.start_cost);
    assert_eq!(g("end_cost"), r.end_cost);
    assert_eq!(g("m"), fit.m.value);
    assert_eq!(g("c"), fit.c.value);
    for i in 0..3 {
        assert_eq!(g(&format!("item{i}_v")), fit.items[i].v.value);
    }
    for i in 0..4 {
        assert_eq!(g(&format!("vn_v{i}_after")), fit.vns[0].v.value.e[i]);
    }
    // The session solve and the covariance over its context.
    {
        use arael::covariance::{CovMode, Covariance};
        use arael::simple_lm::{LmSession, SparseFaer};
        let mut sess = LmSession::new(SparseFaer::new());
        let rs = sess.solve(&mut fit, &cfg).unwrap();
        assert_eq!(g("sess_end"), rs.end_cost);
        let cov = sess.assemble_covariance(&mut fit, CovMode::AllMarginals).unwrap();
        assert_eq!(g("sess_cov_item0"), cov.marginal_cov(&fit.items[0]).unwrap()[(0, 0)]);
    }
    // The covariance asked for in the config, carried by the result.
    {
        use arael::covariance::{CovError, CovMode};
        let cfgc = LmConfig { covariance: Some(CovMode::AllMarginals), ..cfg.clone() };
        let rc = fit.solve_dense(&cfgc).unwrap();
        assert_eq!(g("cfg_cov_item0"), rc.covariance.as_ref().unwrap().marginal_cov(&fit.items[0]).unwrap()[(0, 0)]);
        assert_eq!(g("cfg_cov_absent"), 1.0);
        assert_eq!(r.covariance.err(), Some(CovError::NotRequested));
    }
}
