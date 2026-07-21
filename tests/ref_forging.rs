//! A `Ref` is handed out by the collection that stores the element --
//! `push`, `alloc`, `ref_at`, or iteration -- and cannot be built from a
//! bare index. Forging one would let a number from anywhere address an
//! element, which is the aliasing this API exists to prevent.
//!
//! Regenerate snapshots with:
//!     TRYBUILD=overwrite cargo test --test ref_forging

#[test]
fn a_ref_cannot_be_forged_from_an_index() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/ref_forging/forge_from_index.rs");
}
