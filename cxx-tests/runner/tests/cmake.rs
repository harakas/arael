// The generated CMake glue, driven exactly as a consumer would:
// configure the checked-in consumer project against the generated
// cxx/ tree, build (which runs cargo for the capi staticlib), run the
// smoke binary. Skipped with a note when cmake or a C++ compiler is
// missing.

use std::path::{Path, PathBuf};
use std::process::Command;

fn have(tool: &str) -> bool {
    Command::new(tool).arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status().map(|s| s.success()).unwrap_or(false)
}

#[test]
fn cmake_consumer_builds_and_runs() {
    if !have("cmake") || !have("c++") {
        eprintln!("cmake consumer: cmake or c++ missing, skipping");
        return;
    }
    let ws: PathBuf = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let build = Path::new(env!("CARGO_TARGET_TMPDIR")).join("cmake_build");
    let _ = std::fs::remove_dir_all(&build);

    let status = Command::new("cmake")
        .arg("-S").arg(ws.join("runner/tests/cmake_consumer"))
        .arg("-B").arg(&build)
        .arg(format!("-DMODEL_CXX_DIR={}", ws.join("model/cxx").display()))
        .status().expect("cmake spawn");
    assert!(status.success(), "cmake configure failed");

    let status = Command::new("cmake")
        .arg("--build").arg(&build)
        .status().expect("cmake build spawn");
    assert!(status.success(), "cmake build failed");

    let out = Command::new(build.join("smoke")).output().expect("smoke run");
    assert!(out.status.success(), "smoke failed: {}",
        String::from_utf8_lossy(&out.stderr));
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(text.contains("m ") && text.contains("end_cost "), "{text}");
}
