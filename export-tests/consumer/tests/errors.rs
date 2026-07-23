//! Reaching for a type the model crate did not export must fail with the
//! tombstone reason, not a generic missing-layout error.
//!
//! Regenerate snapshots with:
//!     TRYBUILD=overwrite cargo test --test errors

#[test]
fn unexported_types_are_tombstoned() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/errors/hidden_field.rs");
}
