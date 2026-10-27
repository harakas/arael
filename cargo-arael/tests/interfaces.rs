// Choosing the interface trees an export writes: the names, the
// manifest key, and the files each choice generates for the golden
// fixture's model.

use cargo_arael::export::{generate, generate_interfaces, manifest_interfaces, Interfaces};
use cargo_arael::ir::Model;
use std::path::PathBuf;

fn model() -> Model {
    Model::parse(include_str!("golden/fit.json")).unwrap()
}

fn files(interfaces: Interfaces) -> Vec<(PathBuf, String)> {
    let m = model();
    generate_interfaces(&[&m], "fit-model", "{ version = \"0.9\" }", None, interfaces).unwrap()
}

/// The top-level directories a file list writes into, in order of first
/// appearance.
fn trees(files: &[(PathBuf, String)]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for (path, _) in files {
        let top = path.components().next().unwrap().as_os_str().to_string_lossy().to_string();
        if !out.contains(&top) {
            out.push(top);
        }
    }
    out
}

#[test]
fn names_parse_and_cxx_and_python_bring_capi() {
    let s = |names: &str| Interfaces::parse(names.split(',')).unwrap().names();
    assert_eq!(s("cxx"), ["capi", "cxx"]);
    assert_eq!(s("python"), ["capi", "python"]);
    assert_eq!(s("wasm"), ["wasm"]);
    assert_eq!(s("capi"), ["capi"]);
    assert_eq!(s(" wasm , cxx ,"), ["capi", "cxx", "wasm"]);
    assert_eq!(s("capi,cxx,python,wasm"), Interfaces::ALL.names());
}

#[test]
fn unknown_or_no_names_are_errors() {
    let err = Interfaces::parse("cxx,c++".split(',')).unwrap_err();
    assert!(err.contains("unknown interface `c++`") && err.contains("capi, cxx, python, wasm"), "{err}");
    let err = Interfaces::parse(",".split(',')).unwrap_err();
    assert!(err.contains("no interface named"), "{err}");
}

#[test]
fn the_manifest_key_is_read_under_package_metadata_arael() {
    let one_line = "[package]\nname = \"m\"\n\n[package.metadata.arael]\nnamespace = \"ns\"\ninterfaces = [\"cxx\"] # only C++\n";
    assert_eq!(manifest_interfaces(one_line).unwrap().unwrap().names(), ["capi", "cxx"]);

    let lines = "[package.metadata.arael]\ninterfaces = [\n  \"python\", # the ctypes package\n  'wasm',\n]\n[dependencies]\n";
    assert_eq!(manifest_interfaces(lines).unwrap().unwrap().names(), ["capi", "python", "wasm"]);

    let none = "[package]\nname = \"m\"\n[package.metadata.arael]\nnamespace = \"ns\"\n";
    assert_eq!(manifest_interfaces(none).unwrap(), None);

    // The same key in another table is not the export's.
    let elsewhere = "[package.metadata.other]\ninterfaces = [\"nope\"]\n";
    assert_eq!(manifest_interfaces(elsewhere).unwrap(), None);
}

#[test]
fn a_malformed_manifest_key_is_an_error() {
    let err = manifest_interfaces("[package.metadata.arael]\ninterfaces = \"cxx\"\n").unwrap_err();
    assert!(err.contains("is a list of names"), "{err}");
    let err = manifest_interfaces("[package.metadata.arael]\ninterfaces = [\"cxx\",\n").unwrap_err();
    assert!(err.contains("not closed"), "{err}");
    let err = manifest_interfaces("[package.metadata.arael]\ninterfaces = [\"cpp\"]\n").unwrap_err();
    assert!(err.contains("unknown interface `cpp`"), "{err}");
}

#[test]
fn each_choice_generates_its_trees_with_the_full_export_contents() {
    let all = files(Interfaces::ALL);
    let m = model();
    let generated = generate(&[&m], "fit-model", "{ version = \"0.9\" }", None).unwrap();
    assert_eq!(all, generated, "every tree is what generate writes");
    assert_eq!(trees(&all), ["capi", "cxx", "python", "wasm"]);

    for (names, expect) in [
        ("cxx", vec!["capi", "cxx"]),
        ("python", vec!["capi", "python"]),
        ("wasm", vec!["wasm"]),
        ("capi", vec!["capi"]),
    ] {
        let chosen = files(Interfaces::parse(names.split(',')).unwrap());
        assert_eq!(trees(&chosen), expect, "{names}");
        // The same files, in the same order and with the same content,
        // as the full export's files of those trees.
        let from_all: Vec<_> = all.iter()
            .filter(|(p, _)| expect.iter().any(|t| p.starts_with(t)))
            .cloned().collect();
        assert_eq!(chosen, from_all, "{names}");
    }
}
