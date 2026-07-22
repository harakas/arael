//! Generic `#[arael::model]` structs: exactly one `Float`-bounded type
//! parameter, never on a root, and one instantiation per entity name in
//! a root. Each violation must be a clear macro error, not silent
//! miscompilation.
//!
//! Regenerate snapshots with:
//!     TRYBUILD=overwrite cargo test --test generic_model_errors

#[test]
fn invalid_generic_models_are_rejected() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/generic_model_errors/generic_root.rs");
    t.compile_fail("tests/generic_model_errors/two_type_params.rs");
    t.compile_fail("tests/generic_model_errors/mixed_precision_root.rs");
}
