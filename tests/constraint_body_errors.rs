//! Compile-fail fixtures for type errors inside constraint bodies --
//! the dispatch layer must reject wrong operand types with a clear
//! message instead of panicking or silently misrouting.
//!
//! Regenerate snapshots with:
//!     TRYBUILD=overwrite cargo test --test constraint_body_errors

#[test]
fn constraint_body_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/constraint_body_errors/rem_non_vec3.rs");
    t.compile_fail("tests/constraint_body_errors/cross_wrong_arg.rs");
    t.compile_fail("tests/constraint_body_errors/field_typo.rs");
    t.compile_fail("tests/constraint_body_errors/unknown_binding.rs");
    t.compile_fail("tests/constraint_body_errors/macro_stmt.rs");
    t.compile_fail("tests/constraint_body_errors/stray_expr_stmt.rs");
    t.compile_fail("tests/constraint_body_errors/vectn_index_range.rs");
    t.compile_fail("tests/constraint_body_errors/vectn_dim_mismatch.rs");
    t.compile_fail("tests/constraint_body_errors/vectn_nonliteral_index.rs");
    t.compile_fail("tests/constraint_body_errors/vectn_nonliteral_dim.rs");
    t.compile_fail("tests/constraint_body_errors/transform_neg.rs");
    t.compile_fail("tests/constraint_body_errors/transform_right_mul.rs");
    t.compile_fail("tests/constraint_body_errors/transform_residual.rs");
    t.compile_fail("tests/constraint_body_errors/transform_scale_on_rigid.rs");
    t.compile_fail("tests/constraint_body_errors/match_vector_arm.rs");
    t.compile_fail("tests/constraint_body_errors/match_vector_scrutinee.rs");
    t.compile_fail("tests/constraint_body_errors/match_out_of_order.rs");
    t.compile_fail("tests/constraint_body_errors/match_sparse.rs");
    t.compile_fail("tests/constraint_body_errors/match_guard.rs");
    t.compile_fail("tests/constraint_body_errors/match_wild_not_last.rs");
    t.compile_fail("tests/constraint_body_errors/match_non_literal.rs");
}
