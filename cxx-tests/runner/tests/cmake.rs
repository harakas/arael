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

    // The same interface linked into a shared library.
    let out = Command::new(build.join("smoke_shared")).output().expect("smoke_shared run");
    assert!(out.status.success(), "smoke_shared failed: {}{}",
        String::from_utf8_lossy(&out.stdout), String::from_utf8_lossy(&out.stderr));

    // The staticlib's symbols stay inside the shared library: exported, the
    // asm routines faer uses on x86-64 (private-gemm-x86) call one another
    // through lazily bound PLT slots, which they do not survive.
    #[cfg(target_os = "linux")]
    {
        if !have("nm") {
            eprintln!("cmake consumer: nm missing, export check skipped");
            return;
        }
        let out = Command::new("nm").args(["-D", "--defined-only"])
            .arg(build.join("libconsumer_shared.so"))
            .output().expect("nm spawn");
        assert!(out.status.success(), "nm failed");
        let names: Vec<String> = String::from_utf8_lossy(&out.stdout).lines()
            .filter_map(|l| l.split_whitespace().last().map(str::to_string))
            .collect();
        assert!(names.iter().any(|n| n == "consumer_fit_slope"), "{names:?}");
        let leaked: Vec<&String> = names.iter()
            .filter(|n| n.starts_with("fit_") || is_rust_symbol(n))
            .collect();
        assert!(leaked.is_empty(),
            "{} staticlib symbols exported, e.g. {:?}", leaked.len(), &leaked[..leaked.len().min(5)]);
    }
}

// A Rust symbol: v0 mangling, or legacy mangling ending in the hash
// `17h<16 hex digits>E`.
#[cfg(target_os = "linux")]
fn is_rust_symbol(name: &str) -> bool {
    if name.starts_with("_R") {
        return true;
    }
    let b = name.as_bytes();
    name.starts_with("_ZN") && b.len() > 20 && name.ends_with('E')
        && &name[b.len() - 20..b.len() - 17] == "17h"
        && name[b.len() - 17..b.len() - 1].bytes().all(|c| c.is_ascii_hexdigit())
}
