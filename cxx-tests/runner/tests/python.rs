// End-to-end parity for the GENERATED Python interface: the same
// fixture problem as parity_main.cpp, built through ctypes over the
// capi cdylib, verified against the same Rust mirror. Skipped with a
// note when python3 is absent.

#[path = "parity_verify.rs"]
mod parity_verify;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

#[test]
fn python_interface_matches_rust_exactly() {
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    if Command::new("python3").arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false) == false
    {
        eprintln!("python parity: no python3 found, skipping");
        return;
    }

    // The cdylib the Python side loads.
    let status = Command::new("cargo")
        .args(["build", "-p", "cxx-fit-capi", "--release"])
        .current_dir(&ws)
        .status().expect("cargo spawn");
    assert!(status.success(), "capi build failed");

    let out = Command::new("python3")
        .arg(ws.join("runner/tests/parity_main.py"))
        .env("ARAEL_CAPI", ws.join("target/release/libcxx_fit_capi.so"))
        .output().expect("python spawn");
    assert!(out.status.success(), "python run failed: {}",
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

/// The vendored column transport on its own: every input shape the
/// vectorized calls accept and every refusal, no cdylib involved.
#[test]
fn column_transport_accepts_and_refuses_the_documented_inputs() {
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    if Command::new("python3").arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false) == false
    {
        eprintln!("columns: no python3 found, skipping");
        return;
    }
    let out = Command::new("python3")
        .arg(ws.join("runner/tests/columns_main.py"))
        .output().expect("python spawn");
    assert!(out.status.success(), "columns run failed: {}{}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));
}
