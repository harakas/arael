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
}
