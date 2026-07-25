// Parity test for the arael Python math library
// (cargo-arael/python/arael/): the same golden values as cxx_math,
// computed by tests/py_math/main.py, compared per value. Python
// computes in doubles and rounds through the storage type, so the
// f32-prefixed entries get a slightly looser tolerance than the C++
// twin (which computes in f32 proper). Skipped (with a note) when no
// python3 is available.

#[path = "cxx_math/golden.rs"]
mod golden_mod;
use golden_mod::golden;

#[test]
fn py_math_library_matches_rust() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let script = manifest.join("tests/py_math/main.py");
    let output = match std::process::Command::new("python3").arg(&script).output() {
        Ok(o) => o,
        Err(_) => {
            eprintln!("py_math: no python3 found, skipping");
            return;
        }
    };
    assert!(output.status.success(), "python run failed: {}",
        String::from_utf8_lossy(&output.stderr));
    let text = String::from_utf8(output.stdout).unwrap();

    let mut got: std::collections::HashMap<String, f64> = std::collections::HashMap::new();
    for line in text.lines() {
        let mut it = line.split_whitespace();
        let (Some(name), Some(val)) = (it.next(), it.next()) else { continue };
        got.insert(name.to_string(), val.parse::<f64>().unwrap());
    }

    let expected = golden();
    assert_eq!(got.len(), expected.len(),
        "value count mismatch: python printed {}, Rust computed {}", got.len(), expected.len());
    let mut worst = 0.0f64;
    for (name, want) in &expected {
        let have = *got.get(name).unwrap_or_else(|| panic!("python output missing `{}`", name));
        let tol = if name.starts_with("f32") { 1e-5 } else { 1e-13 };
        let err = (have - want).abs() / (1.0 + want.abs());
        worst = worst.max(if name.starts_with("f32") { 0.0 } else { err });
        assert!(err <= tol, "`{}`: python {} vs Rust {} (rel err {})", name, have, want, err);
    }
    eprintln!("py_math: {} values matched, worst f64 rel err {:.3e}", expected.len(), worst);
}
