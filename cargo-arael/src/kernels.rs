//! `cargo arael kernels`: the device kernels of every `#[arael(root)]` a
//! build expands, one file per root. The build runs with
//! `ARAEL_KERNELS_DIR` set, and the macro writes `<dir>/<crate>/<Root>.cu`
//! for each root it expands: the root's kernels in C++, commented with the
//! model's names and source lines, or why the root has no device form.

use std::path::{Path, PathBuf};

/// The environment variable the macro writes the kernels under.
pub const KERNELS_DIR_VAR: &str = "ARAEL_KERNELS_DIR";

/// The cargo arguments a run without any builds with: every target, so
/// the roots of tests, examples and benches are expanded too.
pub const DEFAULT_CARGO_ARGS: &[&str] = &["--all-targets"];

/// From `cargo metadata --no-deps` output: the target directory, and the
/// source files of the targets to rebuild, those of the package whose
/// manifest is `manifest` (every package's when none is, as for a virtual
/// workspace manifest).
pub fn targets(metadata: &str, manifest: &Path) -> Result<(PathBuf, Vec<PathBuf>), String> {
    let v: serde_json::Value = serde_json::from_str(metadata).map_err(|e| format!("cargo metadata: {e}"))?;
    let target_dir = v["target_directory"].as_str().ok_or("cargo metadata: no target_directory")?;
    let packages = v["packages"].as_array().ok_or("cargo metadata: no packages")?;
    let mine: Vec<&serde_json::Value> = packages.iter().filter(|p| p["manifest_path"].as_str().map(Path::new) == Some(manifest)).collect();
    let chosen = if mine.is_empty() { packages.iter().collect() } else { mine };
    let mut sources: Vec<PathBuf> = chosen
        .iter()
        .flat_map(|p| p["targets"].as_array().into_iter().flatten())
        .filter_map(|t| t["src_path"].as_str().map(PathBuf::from))
        .collect();
    sources.sort();
    sources.dedup();
    Ok((PathBuf::from(target_dir), sources))
}

/// One line about a kernels file `text`: its constraints' count, or why
/// its root has no device form.
pub fn summary(text: &str) -> String {
    if let Some(why) = text.lines().next().and_then(|l| l.split_once(" has no device form: ")).map(|(_, why)| why) {
        return format!("no device form: {why}");
    }
    let n = text.lines().filter(|l| l.starts_with("// C") && l.contains(": a constraint of ")).count();
    format!("{n} constraint{}", if n == 1 { "" } else { "s" })
}

/// Every kernels file under `out`, `(<crate>/<Root>.cu, summary)`, sorted.
pub fn collect(out: &Path) -> Result<Vec<(String, String)>, String> {
    let mut files = Vec::new();
    let read_dir = |d: &Path| std::fs::read_dir(d).map_err(|e| format!("{}: {e}", d.display()));
    for krate in read_dir(out)?.flatten() {
        if !krate.path().is_dir() {
            continue;
        }
        for f in read_dir(&krate.path())?.flatten() {
            let path = f.path();
            if path.extension().is_some_and(|x| x == "cu") {
                let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
                let rel = format!("{}/{}", krate.file_name().to_string_lossy(), f.file_name().to_string_lossy());
                files.push((rel, summary(&text)));
            }
        }
    }
    files.sort();
    Ok(files)
}

/// Build the crate in `dir` with `cargo build <cargo_args>` (default
/// [`DEFAULT_CARGO_ARGS`]), its targets' sources touched so every root
/// expands again, and the kernels written under `out` (default
/// `<target dir>/arael-kernels`, emptied first); prints the files.
pub fn run_kernels(dir: &Path, out: Option<&Path>, cargo_args: &[String]) -> Result<(), String> {
    let dir = std::fs::canonicalize(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    let manifest = dir.join("Cargo.toml");
    let meta = std::process::Command::new("cargo")
        .args(["metadata", "--no-deps", "--format-version", "1", "--manifest-path"])
        .arg(&manifest)
        .output()
        .map_err(|e| format!("cargo metadata: {e}"))?;
    if !meta.status.success() {
        return Err(format!("cargo metadata failed:\n{}", String::from_utf8_lossy(&meta.stderr)));
    }
    let (target_dir, sources) = targets(&String::from_utf8_lossy(&meta.stdout), &manifest)?;
    // The variable is not in cargo's fingerprint: a touched target
    // compiles again, and its macros expand again.
    for s in &sources {
        std::fs::OpenOptions::new()
            .append(true)
            .open(s)
            .and_then(|f| f.set_modified(std::time::SystemTime::now()))
            .map_err(|e| format!("touch {}: {e}", s.display()))?;
    }
    // absolute: the macro writes from rustc's working directory
    let out = match out {
        Some(o) if o.is_absolute() => o.to_path_buf(),
        Some(o) => std::env::current_dir().map_err(|e| e.to_string())?.join(o),
        None => target_dir.join("arael-kernels"),
    };
    match std::fs::remove_dir_all(&out) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(format!("{}: {e}", out.display())),
        _ => {}
    }
    let args: Vec<String> = if cargo_args.is_empty() { DEFAULT_CARGO_ARGS.iter().map(|s| s.to_string()).collect() } else { cargo_args.to_vec() };
    let status = std::process::Command::new("cargo")
        .arg("build")
        .args(&args)
        .current_dir(&dir)
        .env(KERNELS_DIR_VAR, &out)
        .status()
        .map_err(|e| format!("cargo build: {e}"))?;
    if !status.success() {
        return Err("cargo build failed".to_string());
    }
    let files = if out.exists() { collect(&out)? } else { Vec::new() };
    if files.is_empty() {
        return Err(format!("the build expanded no #[arael(root)]: nothing in {}", out.display()));
    }
    println!("kernels in {}:", out.display());
    for (rel, s) in files {
        println!("  {rel}: {s}");
    }
    Ok(())
}
