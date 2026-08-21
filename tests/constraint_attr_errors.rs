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
    t.compile_fail("tests/constraint_attr_errors/parent_selfblock_entity_params.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_selfblock_root_parent.rs");
    t.compile_fail("tests/constraint_attr_errors/alias_container.rs");
    t.compile_fail("tests/constraint_attr_errors/alias_container_late.rs");
    t.compile_fail("tests/constraint_attr_errors/block_precision_f32_under_f64.rs");
    t.compile_fail("tests/constraint_attr_errors/block_precision_f64_under_f32.rs");
    t.compile_fail("tests/constraint_attr_errors/block_precision_nested.rs");
    t.compile_fail("tests/constraint_attr_errors/block_precision_generic_inst.rs");
    t.compile_fail("tests/constraint_attr_errors/block_precision_mixed_fields.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_triplet_bad_field.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_triplet_root_parent.rs");
    t.compile_fail("tests/constraint_attr_errors/root_triplet_bad_field.rs");
    t.compile_fail("tests/constraint_attr_errors/option_after_root.rs");
    t.compile_fail("tests/constraint_attr_errors/nested_after_root.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_bad_field.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_triplet_primary.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_entity_params.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_ref_mismatch.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_extra_blocks.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_root_parent.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_unclaimed.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_mixed_attrs.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_refs_ambiguous.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_over_bad.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_over_aliased.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_parent_params.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_bare_ref_read.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_cross_unknown_parent_field.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_value_param_read.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_value_two_levels.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_value_root_held.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_value_ambiguous.rs");
    t.compile_fail("tests/constraint_attr_errors/parent_value_guard_root_held.rs");
}
