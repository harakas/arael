// End-to-end parity: build the fixture problem through the GENERATED
// C++ interface (compiled and linked against the capi staticlib), and
// through the model crate directly in Rust; the two paths run the same
// deterministic solver, so every printed value must round-trip
// EXACTLY. Skipped with a note when no C++ compiler is available.

#[path = "parity_verify.rs"]
mod parity_verify;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

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
        .arg("-std=c++17").arg("-O2").arg("-ffp-contract=off")
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
    parity_verify::verify(&got);
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
