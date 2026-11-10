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

/// The generated Python interface on Metal (macOS): the capi built with
/// `gpu-metal`, the fixture's f64 model refused without mixed precision
/// and with explicit f64 wide values, solved with it to the CPU's
/// optimum, the default wide values the device's (df32).
#[cfg(target_vendor = "apple")]
#[test]
fn python_interface_solves_on_metal() {
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    if Command::new("python3").arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false) == false
    {
        eprintln!("python on metal: no python3 found, skipping");
        return;
    }
    // its own target directory: the parity tests load the capi built
    // without a GPU runtime
    let target = ws.join("target/metal");
    let status = Command::new("cargo")
        .args(["build", "-p", "cxx-fit-capi", "--release", "--features", "gpu-metal", "--target-dir"])
        .arg(&target)
        .current_dir(&ws)
        .status().expect("cargo spawn");
    assert!(status.success(), "capi build failed");
    let out = Command::new("python3")
        .arg(ws.join("runner/tests/metal_main.py"))
        .env("ARAEL_CAPI", target.join("release/libcxx_fit_capi.dylib"))
        .output().expect("python spawn");
    assert!(out.status.success(), "python run failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    let got: HashMap<String, f64> = text
        .lines()
        .filter_map(|l| l.split_once(' ').map(|(n, v)| (n.to_string(), v.trim().parse().unwrap())))
        .collect();
    let g = |n: &str| *got.get(n).unwrap_or_else(|| panic!("no `{n}` in\n{text}"));
    // the C interface's kind of SolverUnavailable
    let unavailable = 8.0;
    assert_eq!(g("default_wide"), 2.0, "GpuWide::Default");
    assert_eq!(g("f64_unmixed_kind"), unavailable, "an f64 model without mixed precision");
    assert_eq!(g("f64_wide_kind"), unavailable, "explicit f64 wide values");
    assert_eq!(g("metal_kind"), -1.0, "{text}");
    assert_eq!((g("metal_gpu_factor"), g("metal_f32_factor")), (1.0, 1.0), "the f32 factor on the device");
    let (cpu, metal) = (g("cpu_end"), g("metal_end"));
    assert!((metal - cpu).abs() <= 1e-9 * cpu.abs().max(1e-12), "Metal's refined solve ends at {metal}, the CPU's at {cpu}");
}

/// The vendored math types pass by value through C both ways, the small
/// ones too (a struct of up to four floats travels in registers on arm64,
/// where ctypes passes a struct holding an array wrongly).
#[test]
fn vendored_math_types_cross_c_by_value() {
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let have = |p: &str| Command::new(p).arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false);
    if !have("python3") || !have("cc") {
        eprintln!("by value: no python3 or cc found, skipping");
        return;
    }
    let lib = Path::new(env!("CARGO_TARGET_TMPDIR")).join("libbyvalue.so");
    let status = Command::new("cc")
        .args(["-O1", "-shared", "-fPIC", "-o"])
        .arg(&lib)
        .arg(ws.join("runner/tests/byvalue.c"))
        .status().expect("cc spawn");
    assert!(status.success(), "byvalue.c build failed");
    let out = Command::new("python3")
        .arg(ws.join("runner/tests/byvalue_main.py"))
        .arg(&lib)
        .output().expect("python spawn");
    assert!(out.status.success(), "python run failed: {}", String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    assert_eq!(text.lines().count(), 7, "{text}");
    let bad: Vec<&str> = text.lines().filter(|l| !l.ends_with(" ok")).collect();
    assert!(bad.is_empty(), "{}", bad.join("\n"));
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
