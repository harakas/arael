// Eigen interop: the vendored arael/eigen.hpp against the system Eigen,
// through the value types and through the generated interface (a
// Frame's anchor and pose). Skipped with a note when no C++ compiler or
// no Eigen is found; the parity test keeps proving the math headers
// compile without Eigen.
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

/// Compiler flags that put Eigen on the include path: pkg-config
/// first, then the usual install directories.
fn find_eigen() -> Option<Vec<String>> {
    if let Ok(out) = Command::new("pkg-config").args(["--cflags", "eigen3"]).output()
        && out.status.success()
    {
        let flags: Vec<String> = String::from_utf8_lossy(&out.stdout)
            .split_whitespace().map(String::from).collect();
        if !flags.is_empty() {
            return Some(flags);
        }
    }
    for dir in ["/usr/include/eigen3", "/usr/local/include/eigen3", "/opt/homebrew/include/eigen3"] {
        if Path::new(dir).join("Eigen/Core").exists() {
            return Some(vec![format!("-I{dir}")]);
        }
    }
    None
}

#[test]
fn eigen_interop_round_trips() {
    let Some(cc) = find_compiler() else {
        eprintln!("eigen interop: no C++ compiler found, skipping");
        return;
    };
    let Some(eigen) = find_eigen() else {
        eprintln!("eigen interop: Eigen not found, skipping");
        return;
    };
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let status = Command::new("cargo")
        .args(["build", "-p", "cxx-fit-capi", "--release"])
        .current_dir(&ws)
        .status().expect("cargo spawn");
    assert!(status.success(), "capi build failed");
    let bin = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cxx_eigen");
    let status = Command::new(cc)
        .arg("-std=c++17").arg("-O2").arg("-Wall").arg("-Wextra")
        .arg("-I").arg(ws.join("model/cxx/include"))
        .args(&eigen)
        .arg(ws.join("runner/tests/eigen_main.cpp"))
        .arg(ws.join("target/release/libcxx_fit_capi.a"))
        .arg("-lpthread").arg("-ldl").arg("-lm")
        .arg("-o").arg(&bin)
        .status().expect("compiler spawn");
    assert!(status.success(), "C++ compile/link failed");
    let out = Command::new(&bin).output().expect("run");
    assert!(out.status.success(), "C++ run failed: {}",
        String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.trim_end().ends_with("ok"), "unexpected output: {text}");
}
