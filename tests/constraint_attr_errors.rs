//! Compile-fail fixtures for `#[arael(constraint(...))]` attribute
//! argument shapes the parser rejects. The positional form is
//! restricted to one block, or the specific 2-item
//! `(<local_self_block>, root.<triplet>)` case; everything else N >= 2
//! must use the bracketed form.
//!
//! Regenerate snapshots with:
//!     TRYBUILD=overwrite cargo test --test constraint_attr_errors

#[test]
fn constraint_attr_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/constraint_attr_errors/positional_three_way.rs");
    t.compile_fail("tests/constraint_attr_errors/positional_two_non_root.rs");
    t.compile_fail("tests/constraint_attr_errors/positional_root_triplet.rs");
}
