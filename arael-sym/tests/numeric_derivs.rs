//! Extern functions whose Rust eval returns the value and the partials
//! together, and the derivative that does not exist.

use arael_sym::*;
use std::collections::HashMap;

// f(x, y) = x * y^2, with the partials y^2 and 2 x y.
fn lookup(args: &[f64]) -> (f64, Vec<f64>) {
    let (x, y) = (args[0], args[1]);
    (x * y * y, vec![y * y, 2.0 * x * y])
}

fn f() -> impl Fn(Vec<E>) -> E {
    extern_func_numeric_derivs("f", 2, "f_eval", false, Some(lookup))
}

fn fxy() -> E {
    f()(vec![symbol("x"), symbol("y")])
}

#[test]
fn value_and_partials_read_the_same_call() {
    let v = fxy();
    assert_eq!(format!("{}", v), "f(x, y).0");
    assert_eq!(v.to_rust("f64"), "f_eval(x, y, true).0");
    assert_eq!(format!("{}", v.diff("x")), "f(x, y).1[0]");
    assert_eq!(format!("{}", v.diff("y")), "f(x, y).1[1]");
    assert_eq!(v.diff("x").to_rust("f64"), "f_eval(x, y, true).1[0]");
    assert_eq!(partial_of(&v, 1).unwrap(), v.diff("y"));
    assert!(partial_of(&v, 2).is_none());
    assert!(partial_of(&symbol("x"), 0).is_none());
}

#[test]
fn whether_the_partials_are_read_is_the_last_argument() {
    let v = fxy();
    assert_eq!(with_partials_read(&v, false).to_rust("f64"), "f_eval(x, y, false).0");
    assert_eq!(with_partials_read(&v.diff("x"), false).to_rust("f64"), "f_eval(x, y, false).1[0]");
    assert_eq!(with_partials_read(&symbol("x"), false), symbol("x"));
}

#[test]
fn eval_reads_the_result() {
    let v = fxy();
    let vars = HashMap::from([("x", 3.0), ("y", 2.0)]);
    assert_eq!(v.eval(&vars).unwrap(), 12.0);
    assert_eq!(v.diff("x").eval(&vars).unwrap(), 4.0);
    assert_eq!(v.diff("y").eval(&vars).unwrap(), 12.0);
    let silent = extern_func_numeric_derivs("g", 1, "g_eval", false, None)(vec![symbol("x")]);
    assert!(silent.eval(&vars).unwrap_err().contains("no eval fn"));
}

#[test]
fn the_chain_rule_runs_through_the_partials() {
    // f(2x, y): d/dx = 2 f_1.
    let v = f()(vec![symbol("x") * 2.0, symbol("y")]);
    let vars = HashMap::from([("x", 3.0), ("y", 2.0)]);
    assert_eq!(v.eval(&vars).unwrap(), 24.0);
    assert_eq!(v.diff("x").eval(&vars).unwrap(), 8.0);
    assert_eq!(v.diff("y").eval(&vars).unwrap(), 24.0);
}

#[test]
fn a_context_argument_is_written_as_it_is_and_never_differentiated_by() {
    let v = extern_func_numeric_derivs("f", 2, "f_eval", true, None)(vec![symbol("scene"), symbol("x")]);
    assert_eq!(v.to_rust("f64"), "f_eval(scene, x, true).0");
    assert_eq!(v.diff("x").to_rust("f64"), "f_eval(scene, x, true).1[0]");
    assert!(v.diff("scene").is_zero());
    assert!(partial_of(&v, 1).is_none());
}

#[test]
fn cse_shares_one_call_between_value_and_partials() {
    let v = fxy();
    let (lets, outs) = cse(&[v.clone(), v.diff("x"), v.diff("y")]);
    assert_eq!(lets.len(), 1, "{lets:?}");
    assert_eq!(lets[0].1.to_rust("f64"), "f_eval(x, y, true)");
    let name = &lets[0].0;
    assert_eq!(outs[0].to_rust("f64"), format!("{name}.0"));
    assert_eq!(outs[1].to_rust("f64"), format!("{name}.1[0]"));
    assert_eq!(outs[2].to_rust("f64"), format!("{name}.1[1]"));
}

#[test]
fn a_partial_has_no_derivative() {
    let dd = fxy().diff("x").diff("x");
    let text = format!("{}", dd);
    assert!(text.contains("no derivative of f"), "{text}");
    let err = dd.eval(&HashMap::from([("x", 1.0), ("y", 1.0)])).unwrap_err();
    assert!(err.contains("no derivative of f"), "{err}");
    assert!(dd.to_rust("f64").contains("compile_error!"));
}

#[test]
fn a_function_declared_without_a_derivative() {
    fn g_eval(args: &[f64]) -> f64 { args[0].sqrt() }
    let g = extern_func1("g", "g_eval",
        |_| [no_derivative("g", "declared without a derivative")], g_eval);
    let (x, y) = (symbol("x"), symbol("y"));
    let e = g(x.clone()) + y.clone();
    // Over a parameter the marker comes through; the other variable
    // never touches g's derivative.
    assert!(format!("{}", e.diff("x")).contains("no derivative of g"));
    assert_eq!(format!("{}", e.diff("y")), "1");
    assert_eq!(e.diff("y").to_rust("f64"), "1.0_f64");
}

#[test]
fn the_bag_calls_such_a_function_through_its_value() {
    let mut bag = FunctionBag::new();
    bag.add_with_kind("f", vec!["__p0".to_string(), "__p1".to_string()],
        FuncKind::ExternNumericDerivs {
            call_path: "f_eval".to_string(), partials_read: true, context_arg: false,
            eval_fn: Some(lookup),
        });
    let e = parse_with_functions("f(x, y) + 1", &bag).unwrap();
    assert_eq!(format!("{}", e), "f(x, y).0 + 1");
    assert_eq!(e.diff("x").to_rust("f64"), "f_eval(x, y, true).1[0]");
}
