// Parity test for the arael C++ math headers
// (cargo-arael/headers/arael/): computes golden values with arael's
// Rust types, compiles and runs tests/cxx_math/main.cpp against the
// headers, and compares every printed value. The euler convention and
// every ported formula must match; skipped (with a note) when no C++
// compiler is available.

#[path = "cxx_math/golden.rs"]
mod golden_mod;
use golden_mod::golden;

fn find_compiler() -> Option<&'static str> {
    for cc in ["c++", "g++", "clang++"] {
        if std::process::Command::new(cc).arg("--version")
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
fn cxx_math_headers_match_rust() {
    let Some(cc) = find_compiler() else {
        eprintln!("cxx_math: no C++ compiler found, skipping");
        return;
    };
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let headers = manifest.join("cargo-arael/headers");
    let src = manifest.join("tests/cxx_math/main.cpp");
    let bin = std::path::Path::new(env!("CARGO_TARGET_TMPDIR")).join("cxx_math_parity");

    let status = std::process::Command::new(cc)
        .arg("-std=c++17").arg("-O2").arg("-ffp-contract=off")
        .arg("-I").arg(&headers)
        .arg(&src).arg("-o").arg(&bin)
        .status().expect("compiler spawn");
    assert!(status.success(), "C++ compile failed");

    let output = std::process::Command::new(&bin).output().expect("run");
    assert!(output.status.success(), "C++ run failed");
    let text = String::from_utf8(output.stdout).unwrap();

    let mut got: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(name), Some(val)) = (it.next(), it.next()) else { continue };
        got.insert(name.to_string(), val.parse::<f64>().unwrap());
    }

    let expected = golden();
    assert_eq!(got.len(), expected.len(),
        "value count mismatch: C++ printed {}, Rust computed {}", got.len(), expected.len());
    let mut worst = 0.0f64;
    for (name, want) in &expected {
        let have = *got.get(name).unwrap_or_else(|| panic!("C++ output missing `{}`", name));
        // f64 paths agree to a few ulps (same formulas, same libm);
        // the f32 round trip is compared at f32 accuracy.
        let tol = if name.starts_with("f32") { 1e-6 } else { 1e-13 };
        let err = (have - want).abs() / (1.0 + want.abs());
        worst = worst.max(if name.starts_with("f32") { 0.0 } else { err });
        assert!(err <= tol, "`{}`: C++ {} vs Rust {} (rel err {})", name, have, want, err);
    }
    eprintln!("cxx_math: {} values matched, worst f64 rel err {:.3e}", expected.len(), worst);
}
