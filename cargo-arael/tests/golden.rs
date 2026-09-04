// Golden-file tests: a fixed sidecar JSON in, byte-exact emitter
// output out. The golden files are the cxx-tests fixture's committed
// artifacts, so this pins the generator without needing a model build.
// Regenerate after intentional emitter changes:
//   cp cxx-tests/model/capi/src/lib.rs cargo-arael/tests/golden/fit_ffi.rs
//   cp cxx-tests/model/python/cxx_fit/fit.py cargo-arael/tests/golden/fit.py
//   cp cxx-tests/model/python/cxx_fit/_fit_ffi.py cargo-arael/tests/golden/fit_ffi.py
//   cp cxx-tests/model/cxx/include/fit.hpp cargo-arael/tests/golden/fit.hpp
// (after `cargo arael export` in cxx-tests/model), and when the fixture
// model itself changed, its sidecar too:
//   cp cxx-tests/model/target/arael-sidecar/Fit.json cargo-arael/tests/golden/fit.json

use cargo_arael::{emit_ffi, emit_hpp, emit_py, ir::Model};

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
fn py_matches_golden() {
    let (ffi, api) = emit_py::emit(&model(), "cxx_fit_capi").unwrap();
    assert!(ffi == include_str!("golden/fit_ffi.py"),
        "python ffi emitter drifted from golden (see file header for regeneration)");
    assert!(api == include_str!("golden/fit.py"),
        "python api emitter drifted from golden (see file header for regeneration)");
}

#[test]
fn generic_entities_are_instantiated_at_the_root_precision() {
    // The cxx-mr fixture's f32 root over a generic `Cell<T>`: the
    // sidecar spells the entity concretely and marks it generic; the
    // shim imports it as an alias at the root's precision, and the
    // header reads it at that precision. Regenerate the fixture after
    // intentional changes:
    //   cp cxx-tests/mr/target/arael-sidecar/Decay.json cargo-arael/tests/golden/mr_decay.json
    let m = Model::parse(include_str!("golden/mr_decay.json")).unwrap();
    assert!(m.types["Cell"].generic);
    assert!(!m.types["Decay"].generic);
    assert_eq!(m.types["Cell"].fields[0].of.as_deref(), Some("f32"));

    let shim = emit_ffi::emit(&m, "cxx_mr").unwrap();
    assert!(shim.contains("type Cell = cxx_mr::Cell<f32>;"), "{shim}");
    assert!(!shim.contains("use cxx_mr::{Cell"), "{shim}");
    assert!(shim.contains("use cxx_mr::{Decay};"), "{shim}");

    let hpp = emit_hpp::emit(&m, "cxx_mr").unwrap();
    assert!(hpp.contains("float v() const"), "{hpp}");
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

/// The golden sidecar with one extra entity type of `n` f64 data fields
/// (`f0`..) in a `std::vec::Vec` on the root.
fn with_wide_type(n: usize, extra: &[(&str, &str)]) -> Model {
    let mut v: serde_json::Value =
        serde_json::from_str(include_str!("golden/fit.json")).unwrap();
    let mut fields: Vec<serde_json::Value> = (0..n)
        .map(|i| serde_json::json!({"name": format!("f{i}"), "kind": "data", "of": "f64"}))
        .collect();
    for (name, of) in extra {
        fields.push(serde_json::json!({"name": name, "kind": "data", "of": of}));
    }
    v["types"]["Wide"] = serde_json::json!({
        "role": "entity", "param_count": 0, "fields": fields
    });
    v["types"]["Fit"]["fields"].as_array_mut().unwrap().push(serde_json::json!({
        "name": "wides", "kind": "collection", "of": "Wide",
        "container": "vec", "spelled": "std::vec::Vec<Wide>"
    }));
    Model::parse(&v.to_string()).unwrap()
}

#[test]
fn more_than_64_leaves_take_a_second_mask_word() {
    // 70 leaves: leaf 64 sits in mask word 1, bit 0, at slot 2 + 64.
    let m = with_wide_type(70, &[]);
    let ffi = emit_ffi::emit(&m, "cxx_fit").unwrap();
    assert!(ffi.contains("assign_slots_wide(e: &mut Wide, s: *const u64)"));
    assert!(ffi.contains("if *s.add(1) & (1u64 << 0) != 0 {\n        e.f64 = f64::from_bits(*s.add(66));"),
        "leaf 64 must read mask word 1 and slot 66");
    assert!(ffi.contains("assign_slots_wide(&mut e, slots.add(i * 72))"),
        "record = 2 mask words + 70 slots");
    let (_, api) = emit_py::emit(&m, "cxx_fit_capi").unwrap();
    assert!(api.contains(&format!("_wide_rec = struct.Struct(\"=QQ{}\")", "d".repeat(70))));
    assert!(api.contains("(m >> 0) & 0xFFFFFFFFFFFFFFFF, (m >> 64) & 0xFFFFFFFFFFFFFFFF"));
    assert!(api.contains("else: m |= 1 << 69\n"));
}

#[test]
fn a_field_named_index_keeps_its_name() {
    // The wrapper's `.index` accessor steps aside; `.ref` stays.
    let m = with_wide_type(1, &[("index", "u32")]);
    let (_, api) = emit_py::emit(&m, "cxx_fit_capi").unwrap();
    let cls = api.split("class Wide:").nth(1).unwrap()
        .split("\nclass ").next().unwrap();
    assert!(cls.contains("    def ref(self):"));
    assert!(!cls.contains("    def index(self):\n        \"\"\"The index this wrapper"));
    assert!(cls.contains("# field `index` takes the name the index accessor would have"));
    assert!(cls.contains("    def index(self, v):"), "the field's own setter is still there");
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
