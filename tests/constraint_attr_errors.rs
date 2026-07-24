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
    t.compile_fail("tests/constraint_attr_errors/marginalize_bad_field.rs");
    t.compile_fail("tests/constraint_attr_errors/root_fit_combined.rs");
    t.compile_fail("tests/constraint_attr_errors/root_before_entity.rs");
    t.compile_fail("tests/constraint_attr_errors/same_name_structs.rs");
    t.compile_fail("tests/constraint_attr_errors/root_selfblock_missing_field.rs");
    t.compile_fail("tests/constraint_attr_errors/root_selfblock_triplet_primary.rs");
    t.compile_fail("tests/constraint_attr_errors/root_selfblock_entity_params.rs");
    t.compile_fail("tests/constraint_attr_errors/duplicate_containment.rs");
    t.compile_fail("tests/constraint_attr_errors/optional_duplicate_containment.rs");
    t.compile_fail("tests/constraint_attr_errors/cross_on_single_instance.rs");
    t.compile_fail("tests/constraint_attr_errors/nested_duplicate_containment.rs");
    t.compile_fail("tests/constraint_attr_errors/cyclic_containment_ref.rs");
}
