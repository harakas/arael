//! Compile-fail fixtures for `#[arael::function]`.
//!
//! Each `.rs` file under `tests/user_function_errors/` triggers one
//! class of malformed user-function declaration. A matching `.stderr`
//! snapshot verifies the message and span.
//!
//! Regenerate snapshots after message changes with:
//!     TRYBUILD=overwrite cargo test --test user_function_errors
//!
//! Error messages come from our own macro (`arael-macros/src/function.rs`
//! and `arael-macros/src/constraint.rs`) so snapshot drift across rustc
//! versions is minimal -- the body of each snapshot is the message we
//! emit, with just rustc's surrounding " --> file:line:col" scaffolding.

#[test]
fn user_function_compile_errors() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/user_function_errors/bad_signature.rs");
    t.compile_fail("tests/user_function_errors/mismatched_derivs_count.rs");
    t.compile_fail("tests/user_function_errors/value_only_on_parameter.rs");
    t.compile_fail("tests/user_function_errors/root_type_mismatch.rs");
    t.compile_fail("tests/user_function_errors/root_missing_in_call.rs");
    t.compile_fail("tests/user_function_errors/root_to_rootless.rs");
    t.compile_fail("tests/user_function_errors/root_in_symbolic_field.rs");
    t.compile_fail("tests/user_function_errors/tuple_without_flag.rs");
    t.compile_fail("tests/user_function_errors/flag_without_tuple.rs");
    t.compile_fail("tests/user_function_errors/derivs_on_numeric.rs");
    t.compile_fail("tests/user_function_errors/partials_count_off.rs");
    t.compile_fail("tests/user_function_errors/same_name_form_b.rs");
    t.compile_fail("tests/user_function_errors/unknown_function_in_body.rs");
    t.compile_fail("tests/user_function_errors/arity_mismatch_at_call.rs");
    t.compile_fail("tests/user_function_errors/bad_deriv_string.rs");
    t.compile_fail("tests/user_function_errors/match_out_of_order.rs");
    t.compile_fail("tests/user_function_errors/match_guard.rs");
    t.compile_fail("tests/user_function_errors/match_block_arm.rs");
    t.compile_fail("tests/user_function_errors/typed_wrong_kind.rs");
    t.compile_fail("tests/user_function_errors/typed_tuple_arithmetic.rs");
    t.compile_fail("tests/user_function_errors/typed_recursion.rs");
}
