//! Cross-referenced `#[arael::function]` siblings must be callable
//! from ordinary Rust, not just from inside `#[arael::model]`
//! constraint bodies. The runtime registry populated via `inventory`
//! is what makes that work: Form A bodies and Form B derivative
//! closures seed their `FunctionBag` from
//! `arael::user_fn::with_registry_bag`, so `f`'s deriv referencing
//! `g` (declared later in the same file, or in another module)
//! resolves at call time.
//!
//! These tests would panic with "unknown function: g" before the
//! registry was introduced -- that was the "documented limitation"
//! removed along with this plumbing.

use arael_sym::{E, symbol};

// Form B pair with mutually-recursive derivatives. `f`'s deriv
// references `g`, declared AFTER `f` in the source.

#[arael::function(f_mut, derivs = [g_mut(x)])]
fn f_mut_eval(x: f64) -> f64 { x.sin() }

#[arael::function(g_mut, derivs = [-f_mut(x)])]
fn g_mut_eval(x: f64) -> f64 { x.cos() }

#[test]
fn extern_cross_ref_runtime() {
    let x = symbol("y");
    let e = f_mut(x.clone());
    // f_mut returns the Func node at y; differentiating by y should
    // give g_mut(y) (chain rule: d f_mut(y) / dy = g_mut(y) * 1).
    let de = e.diff("y");
    let s = format!("{de}");
    assert!(s.contains("g_mut(y)"),
        "expected f_mut.diff to reference g_mut(y), got {s:?}");

    // And g_mut's deriv chains into -f_mut.
    let dg = g_mut(x).diff("y");
    let sg = format!("{dg}");
    assert!(sg.contains("f_mut(y)"),
        "expected g_mut.diff to reference f_mut(y), got {sg:?}");

    // Both should evaluate to the numerical sin'/cos' at y = 0.5.
    let mut env = std::collections::HashMap::new();
    env.insert("y", 0.5);
    // f_mut(y).diff(y) = g_mut(y) = cos(y) ~ 0.8776
    let v = de.eval(&env).expect("f_mut derivative must evaluate");
    assert!((v - 0.5_f64.cos()).abs() < 1e-12,
        "f_mut'(0.5) = {v}, expected {}", 0.5_f64.cos());
    // g_mut(y).diff(y) = -f_mut(y) = -sin(y) ~ -0.4794
    let v = dg.eval(&env).expect("g_mut derivative must evaluate");
    assert!((v - (-0.5_f64.sin())).abs() < 1e-12,
        "g_mut'(0.5) = {v}, expected {}", -0.5_f64.sin());
}

// Form A that references a Form B user fn declared afterwards.

#[arael::function]
fn applied_later(z: E) -> E { later_fn(z) * 2.0 }

#[arael::function(later_fn, derivs = [1.0])]
fn later_fn_eval(x: f64) -> f64 { x + 1.0 }

#[test]
fn symbolic_forward_ref_runtime() {
    let y = symbol("y");
    let e = applied_later(y);
    // Body is `later_fn(y) * 2`. Differentiating by y via chain rule:
    // d/dy later_fn(y) * 2 = 1 * 2 = 2.
    let de = e.diff("y");
    let mut env = std::collections::HashMap::new();
    env.insert("y", 3.0);
    let v = de.eval(&env).expect("derivative must evaluate");
    assert!((v - 2.0).abs() < 1e-12,
        "applied_later'(3.0) = {v}, expected 2.0");
    // Forward evaluation: applied_later(3) = (3 + 1) * 2 = 8.
    let v = e.eval(&env).expect("forward must evaluate");
    assert!((v - 8.0).abs() < 1e-12,
        "applied_later(3.0) = {v}, expected 8.0");
}

// Form A with an explicit deriv that references another user fn.

#[arael::function(derivs = [later_fn(a)])]
fn with_xref_deriv(a: E) -> E { a * a }

#[test]
fn symbolic_explicit_deriv_xref() {
    let y = symbol("y");
    let de = with_xref_deriv(y).diff("y");
    let mut env = std::collections::HashMap::new();
    env.insert("y", 4.0);
    // Explicit deriv said `later_fn(a)`; at y = 4, later_fn(4) = 5.
    let v = de.eval(&env).expect("explicit deriv must evaluate");
    assert!((v - 5.0).abs() < 1e-12,
        "with_xref_deriv'(4.0) = {v}, expected 5.0");
}

// Form A: the previously-latent bug. square(symbol("y")) must return
// `y * y`, not `x * x`. Before the registry + explicit substitution
// the body's free symbol "x" leaked into the result verbatim.

#[arael::function]
fn square2(x: E) -> E { x * x }

#[test]
fn symbolic_arg_substituted() {
    let y = symbol("y");
    let e = square2(y);
    let s = format!("{e}");
    assert!(s.contains("y") && !s.contains('x'),
        "square2(y) must substitute y for x, got {s:?}");
    let mut env = std::collections::HashMap::new();
    env.insert("y", 3.0);
    let v = e.eval(&env).expect("forward must evaluate");
    assert!((v - 9.0).abs() < 1e-12, "square2(3) = {v}, expected 9");
}
