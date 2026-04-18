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
    t.compile_fail("tests/user_function_errors/missing_derivs_form_b.rs");
    t.compile_fail("tests/user_function_errors/same_name_form_b.rs");
    t.compile_fail("tests/user_function_errors/unknown_function_in_body.rs");
    t.compile_fail("tests/user_function_errors/arity_mismatch_at_call.rs");
}
