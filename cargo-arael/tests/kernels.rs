// `cargo arael kernels` without a build: the targets it touches, read off
// `cargo metadata`, and the listing of the files the macro wrote.

use cargo_arael::kernels::{collect, summary, targets};
use std::path::{Path, PathBuf};

const METADATA: &str = r#"{
  "target_directory": "/w/target",
  "packages": [
    { "manifest_path": "/w/model/Cargo.toml",
      "targets": [ { "src_path": "/w/model/src/lib.rs" }, { "src_path": "/w/model/tests/fit.rs" } ] },
    { "manifest_path": "/w/other/Cargo.toml",
      "targets": [ { "src_path": "/w/other/src/main.rs" } ] }
  ]
}"#;

#[test]
fn the_package_of_the_manifest_has_its_targets_touched() {
    let (target, sources) = targets(METADATA, Path::new("/w/model/Cargo.toml")).unwrap();
    assert_eq!(target, PathBuf::from("/w/target"));
    assert_eq!(sources, vec![PathBuf::from("/w/model/src/lib.rs"), PathBuf::from("/w/model/tests/fit.rs")]);
}

#[test]
fn a_virtual_manifest_has_every_package_touched() {
    let (_, sources) = targets(METADATA, Path::new("/w/Cargo.toml")).unwrap();
    assert_eq!(sources.len(), 3);
}

#[test]
fn a_kernels_file_is_summed_up_by_its_constraints_or_its_reason() {
    let kernels = "// Device assembly of the root `S`\n// C0: a constraint of `A` (src/lib.rs:3), 2 rows\n// C1: a constraint of `B`, 1 row\n";
    assert_eq!(summary(kernels), "2 constraints");
    assert_eq!(summary("// `S` (src/lib.rs:9) has no device form: a `coo` pair has no device form yet\n"), "no device form: a `coo` pair has no device form yet");
}

#[test]
fn the_files_are_listed_per_crate_sorted() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("kernels-collect");
    let _ = std::fs::remove_dir_all(&dir);
    for (rel, text) in [("b/Two.cu", "// C0: a constraint of `T`, 1 row\n"), ("a/One.cu", "// `One` has no device form: why\n"), ("a/notes.txt", "")] {
        let path = dir.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }
    assert_eq!(
        collect(&dir).unwrap(),
        vec![("a/One.cu".to_string(), "no device form: why".to_string()), ("b/Two.cu".to_string(), "1 constraint".to_string())]
    );
}
