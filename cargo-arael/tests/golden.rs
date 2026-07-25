// Golden-file tests: a fixed sidecar JSON in, byte-exact emitter
// output out. The golden files are the cxx-tests fixture's committed
// artifacts, so this pins the generator without needing a model build.
// Regenerate after intentional emitter changes:
//   cp cxx-tests/model/capi/src/lib.rs cargo-arael/tests/golden/fit_ffi.rs
//   cp cxx-tests/model/cxx/include/fit.hpp cargo-arael/tests/golden/fit.hpp
// (after `cargo arael export` in cxx-tests/model).

use cargo_arael::{emit_ffi, emit_hpp, ir::Model};

fn model() -> Model {
    Model::parse(include_str!("golden/fit.json")).unwrap()
}

#[test]
fn ffi_shim_matches_golden() {
    let got = emit_ffi::emit(&model(), "cxx_fit").unwrap();
    let want = include_str!("golden/fit_ffi.rs");
    assert!(got == want,
        "ffi emitter drifted from golden (see file header for regeneration)");
}

#[test]
fn hpp_matches_golden() {
    let got = emit_hpp::emit(&model(), "cxx_fit").unwrap();
    let want = include_str!("golden/fit.hpp");
    assert!(got == want,
        "hpp emitter drifted from golden (see file header for regeneration)");
}

#[test]
fn builtin_component_types_do_not_block_class_order() {
    // Builtin components (TransformParam, UnitVecParam, ...) appear in
    // the sidecar's type table but are inlined as methods, never
    // emitted as classes; the children-first ordering must not wait
    // for them.
    let mut v: serde_json::Value =
        serde_json::from_str(include_str!("golden/fit.json")).unwrap();
    v["types"]["TransformParamF"] = serde_json::json!({
        "role": "component", "param_count": 6, "builtin": true, "fields": []
    });
    v["types"]["Pose"]["fields"].as_array_mut().unwrap().insert(0,
        serde_json::json!({"name": "r2w", "kind": "component", "of": "TransformParamF"}));
    let m = Model::parse(&v.to_string()).unwrap();
    emit_hpp::emit(&m, "cxx_fit").expect("builtin component must not deadlock the topo order");
}

#[test]
fn unsupported_kinds_error_loudly() {
    let mut m = model();
    let t = m.types.get_mut("N").unwrap();
    t.fields[0].kind = "data".to_string();
    t.fields[0].of = Some("SomethingWeird".to_string());
    let e = emit_ffi::emit(&m, "cxx_fit").unwrap_err();
    assert!(e.contains("not supported"), "{e}");
}
