//! Compile-fail fixtures for `#[arael(constraint(...))]` attribute
//! argument shapes the parser rejects. The positional form is
//! restricted to one block; every N >= 2 list (including the 2-item
//! `[<local_self_block>, root.<triplet>]` shape) must be bracketed.
//!
//! Regenerate snapshots with:
//!     TRYBUILD=overwrite cargo test --test constraint_attr_errors

#[test]
fn constraint_attr_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/constraint_attr_errors/positional_three_way.rs");
    t.compile_fail("tests/constraint_attr_errors/positional_two_non_root.rs");
    t.compile_fail("tests/constraint_attr_errors/positional_root_triplet.rs");
    t.compile_fail("tests/constraint_attr_errors/typo_gaurd.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_bad_value.rs");
    t.compile_fail("tests/constraint_attr_errors/cross_bad_arity.rs");
    t.compile_fail("tests/constraint_attr_errors/root_typo_keyword.rs");
    t.compile_fail("tests/constraint_attr_errors/root_fit_combined.rs");
}
