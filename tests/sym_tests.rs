use std::collections::HashMap;
use arael::sym::*;
use arael::sym;

// ============================================================
// Helpers
// ============================================================

fn approx_eq(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-8
}

/// Verify symbolic derivative numerically via finite differences
fn check_diff_numerically(expr: &E, var: &str, vars: &HashMap<&str, f64>, eps: f64) {
    let deriv = expr.diff(var);
    let symbolic_val = deriv.eval(vars).unwrap();

    let mut vars_plus = vars.clone();
    let mut vars_minus = vars.clone();
    let v = *vars.get(var).unwrap();
    vars_plus.insert(var, v + eps);
    vars_minus.insert(var, v - eps);
    let numerical_val = (expr.eval(&vars_plus).unwrap() - expr.eval(&vars_minus).unwrap()) / (2.0 * eps);

    assert!(
        approx_eq(symbolic_val, numerical_val),
        "diff check failed for d/d{var}({expr}): symbolic={symbolic_val}, numerical={numerical_val}"
    );
}

// ============================================================
// Construction & structural equality
// ============================================================

#[test]
fn test_symbol_equality() {
    let x1 = symbol("x");
    let x2 = symbol("x");
    let y = symbol("y");
    assert_eq!(x1, x2);
    assert_ne!(x1, y);
}

#[test]
fn test_const_equality() {
    assert_eq!(constant(3.0), constant(3.0));
    assert_ne!(constant(3.0), constant(4.0));
}

#[test]
fn test_expr_tree_equality() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let e1 = x + y;
        let e2 = symbol("x") + symbol("y");
        assert_eq!(e1, e2);
    }
}

// ============================================================
// Display
// ============================================================

#[test]
fn test_display_basic() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        assert_eq!(format!("{}", x + y), "x + y");
        assert_eq!(format!("{}", x * y), "x * y");
        assert_eq!(format!("{}", x - y), "x - y");
        assert_eq!(format!("{}", x / y), "x / y");
    }
}

#[test]
fn test_display_precedence() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let z = symbol("z");
        // mul has higher precedence than add, so no parens needed
        // auto-simplify sorts: higher degree first, so y*z before x
        let e = x + y * z;
        assert_eq!(format!("{}", &e), "y * z + x");
    }
}

#[test]
fn test_display_neg() {
    let x = symbol("x");
    assert_eq!(format!("{}", -x), "-x");
}

#[test]
fn test_display_functions() {
    sym! {
        let x = symbol("x");
        assert_eq!(format!("{}", sin(x)), "sin(x)");
        assert_eq!(format!("{}", cos(x)), "cos(x)");
        assert_eq!(format!("{}", exp(x)), "exp(x)");
        assert_eq!(format!("{}", ln(x)), "ln(x)");
        assert_eq!(format!("{}", sqrt(x)), "sqrt(x)");
    }
}

#[test]
fn test_display_const_integers() {
    assert_eq!(format!("{}", constant(3.0)), "3");
    assert_eq!(format!("{}", constant(-1.0)), "-1");
    assert_eq!(format!("{}", constant(0.5)), "0.5");
}

// ============================================================
// Differentiation — basic
// ============================================================

#[test]
fn test_diff_symbol() {
    let x = symbol("x");
    let y = symbol("y");
    assert_eq!(x.diff("x"), constant(1.0));
    assert_eq!(y.diff("x"), constant(0.0));
}

#[test]
fn test_diff_const() {
    assert_eq!(constant(5.0).diff("x"), constant(0.0));
}

#[test]
fn test_diff_add() {
    let x = symbol("x");
    let y = symbol("y");
    let sum = x + y;
    // d/dx(x+y) = 1 + 0
    let d = sum.diff("x");
    let vars = HashMap::from([("x", 2.0), ("y", 3.0)]);
    assert!(approx_eq(d.eval(&vars).unwrap(), 1.0));
}

#[test]
fn test_diff_mul_product_rule() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let prod = x * y;
        // d/dx(x*y) = 1*y + x*0 = y
        let d = prod.diff("x");
        let vars = HashMap::from([("x", 2.0), ("y", 3.0)]);
        assert!(approx_eq(d.eval(&vars).unwrap(), 3.0)); // = y
    }
}

#[test]
fn test_diff_div_quotient_rule() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let e = x / y;
        // d/dx(x/y) = 1/y
        let _d = e.diff("x");
        let vars = HashMap::from([("x", 2.0), ("y", 3.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-6);
    }
}

#[test]
fn test_diff_pow_x_squared() {
    sym! {
        let x = symbol("x");
        let e = pow(x, constant(2.0));
        // d/dx(x^2) = 2x (via general formula)
        let d = e.diff("x");
        let vars = HashMap::from([("x", 3.0)]);
        assert!(approx_eq(d.eval(&vars).unwrap(), 6.0));
    }
}

#[test]
fn test_diff_pow_x_cubed() {
    sym! {
        let x = symbol("x");
        let e = pow(x, constant(3.0));
        let d = e.diff("x");
        let vars = HashMap::from([("x", 2.0)]);
        // d/dx(x^3) = 3x^2 = 12
        assert!(approx_eq(d.eval(&vars).unwrap(), 12.0));
    }
}

// ============================================================
// Differentiation — trig
// ============================================================

#[test]
fn test_diff_sin() {
    sym! {
        let x = symbol("x");
        let e = sin(x);
        let vars = HashMap::from([("x", 1.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_cos() {
    sym! {
        let x = symbol("x");
        let e = cos(x);
        let vars = HashMap::from([("x", 1.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_tan() {
    sym! {
        let x = symbol("x");
        let e = tan(x);
        let vars = HashMap::from([("x", 0.5)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_asin() {
    sym! {
        let x = symbol("x");
        let e = asin(x);
        let vars = HashMap::from([("x", 0.5)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_acos() {
    sym! {
        let x = symbol("x");
        let e = acos(x);
        let vars = HashMap::from([("x", 0.5)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_atan() {
    sym! {
        let x = symbol("x");
        let e = atan(x);
        let vars = HashMap::from([("x", 1.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_atan2() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let e = atan2(y, x);
        let vars = HashMap::from([("x", 2.0), ("y", 1.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
        check_diff_numerically(&e, "y", &vars, 1e-7);
    }
}

// ============================================================
// Differentiation — hyperbolic
// ============================================================

#[test]
fn test_diff_sinh() {
    sym! {
        let x = symbol("x");
        let e = sinh(x);
        let vars = HashMap::from([("x", 1.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_cosh() {
    sym! {
        let x = symbol("x");
        let e = cosh(x);
        let vars = HashMap::from([("x", 1.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_tanh() {
    sym! {
        let x = symbol("x");
        let e = tanh(x);
        let vars = HashMap::from([("x", 0.5)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

// ============================================================
// Differentiation — exp / log
// ============================================================

#[test]
fn test_diff_exp() {
    sym! {
        let x = symbol("x");
        let e = exp(x);
        let vars = HashMap::from([("x", 1.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_ln() {
    sym! {
        let x = symbol("x");
        let e = ln(x);
        let vars = HashMap::from([("x", 2.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_log2() {
    sym! {
        let x = symbol("x");
        let e = log2(x);
        let vars = HashMap::from([("x", 3.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_log10() {
    sym! {
        let x = symbol("x");
        let e = log10(x);
        let vars = HashMap::from([("x", 3.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_sqrt() {
    sym! {
        let x = symbol("x");
        let e = sqrt(x);
        let vars = HashMap::from([("x", 4.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

// ============================================================
// Differentiation — chain rule
// ============================================================

#[test]
fn test_diff_chain_sin_x_squared() {
    sym! {
        let x = symbol("x");
        let e = sin(pow(x, constant(2.0)));
        let vars = HashMap::from([("x", 1.5)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_chain_exp_sin() {
    sym! {
        let x = symbol("x");
        let e = exp(sin(x));
        let vars = HashMap::from([("x", 1.0)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

#[test]
fn test_diff_chain_nested() {
    sym! {
        let x = symbol("x");
        // ln(cos(x^2))
        let e = ln(cos(pow(x, constant(2.0))));
        let vars = HashMap::from([("x", 0.5)]);
        check_diff_numerically(&e, "x", &vars, 1e-7);
    }
}

// ============================================================
// Eval
// ============================================================

#[test]
fn test_eval_add() {
    let x = symbol("x");
    let y = symbol("y");
    let e = x + y;
    let vars = HashMap::from([("x", 1.0), ("y", 2.0)]);
    assert!(approx_eq(e.eval(&vars).unwrap(), 3.0));
}

#[test]
fn test_eval_sin_pi_half() {
    let x = symbol("x");
    let e = sin(x);
    let vars = HashMap::from([("x", std::f64::consts::FRAC_PI_2)]);
    assert!(approx_eq(e.eval(&vars).unwrap(), 1.0));
}

#[test]
fn test_eval_complex() {
    sym! {
        let x = symbol("x");
        // x^2 + 3*x + 1
        let e = pow(x, constant(2.0)) + constant(3.0) * x + constant(1.0);
        let vars = HashMap::from([("x", 2.0)]);
        // 4 + 6 + 1 = 11
        assert!(approx_eq(e.eval(&vars).unwrap(), 11.0));
    }
}

// ============================================================
// Substitution
// ============================================================

#[test]
fn test_subs_basic() {
    let x = symbol("x");
    let y = symbol("y");
    let e = x + y;
    let result = e.subs("x", &constant(3.0));
    let vars = HashMap::from([("y", 2.0)]);
    assert!(approx_eq(result.eval(&vars).unwrap(), 5.0));
}

#[test]
fn test_subs_expr_for_var() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let e = x * x;
        // Substitute x -> y + 1
        let result = e.subs("x", &(y + constant(1.0)));
        let vars = HashMap::from([("y", 2.0)]);
        // (y+1)^2 = 9
        assert!(approx_eq(result.eval(&vars).unwrap(), 9.0));
    }
}

// ============================================================
// Free variables
// ============================================================

#[test]
fn test_free_vars() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let z = symbol("z");
        let e = x * y + sin(z);
        let fv = e.free_vars();
        assert_eq!(fv.len(), 3);
        assert!(fv.contains("x"));
        assert!(fv.contains("y"));
        assert!(fv.contains("z"));
    }
}

#[test]
fn test_free_vars_const() {
    let e = constant(5.0);
    assert!(e.free_vars().is_empty());
}

// ============================================================
// Gradient / Jacobian
// ============================================================

#[test]
fn test_gradient() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        // f = x^2 + y^2
        let f = pow(x, constant(2.0)) + pow(y, constant(2.0));
        let grad = f.diff_all(&["x", "y"]);
        let vars = HashMap::from([("x", 3.0), ("y", 4.0)]);
        // df/dx = 2x = 6, df/dy = 2y = 8
        assert!(approx_eq(grad[0].eval(&vars).unwrap(), 6.0));
        assert!(approx_eq(grad[1].eval(&vars).unwrap(), 8.0));
    }
}

#[test]
fn test_jacobian() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let f1 = x * y;      // f1 = x*y
        let f2 = x + y;      // f2 = x+y
        let j = jacobian(&[f1, f2], &["x", "y"]);
        let vars = HashMap::from([("x", 2.0), ("y", 3.0)]);
        let jeval = j.eval(&vars).unwrap();
        // J = [[y, x], [1, 1]] = [[3, 2], [1, 1]]
        assert!(approx_eq(jeval[0][0], 3.0)); // df1/dx = y
        assert!(approx_eq(jeval[0][1], 2.0)); // df1/dy = x
        assert!(approx_eq(jeval[1][0], 1.0)); // df2/dx = 1
        assert!(approx_eq(jeval[1][1], 1.0)); // df2/dy = 1
    }
}

// ============================================================
// Simplification
// ============================================================

#[test]
fn test_simplify_add_zero() {
    sym! {
        let x = symbol("x");
        assert_eq!((x + 0.0).simplify(), x);
        assert_eq!((0.0 + x).simplify(), x);
    }
}

#[test]
fn test_simplify_mul_one() {
    sym! {
        let x = symbol("x");
        assert_eq!((x * 1.0).simplify(), x);
        assert_eq!((1.0 * x).simplify(), x);
    }
}

#[test]
fn test_simplify_mul_zero() {
    sym! {
        let x = symbol("x");
        assert_eq!((x * 0.0).simplify(), constant(0.0));
        assert_eq!((0.0 * x).simplify(), constant(0.0));
    }
}

#[test]
fn test_simplify_constant_fold() {
    let e = constant(2.0) + constant(3.0);
    assert_eq!(e.simplify(), constant(5.0));
}

#[test]
fn test_simplify_double_neg() {
    sym! {
        let x = symbol("x");
        let e = -(-x);
        assert_eq!(e.simplify(), x);
    }
}

#[test]
fn test_simplify_sub_self() {
    sym! {
        let x = symbol("x");
        assert_eq!((x - x).simplify(), constant(0.0));
    }
}

#[test]
fn test_simplify_div_self() {
    sym! {
        let x = symbol("x");
        assert_eq!((x / x).simplify(), constant(1.0));
    }
}

#[test]
fn test_simplify_pow_zero() {
    let x = symbol("x");
    assert_eq!(pow(x, constant(0.0)).simplify(), constant(1.0));
}

#[test]
fn test_simplify_pow_one() {
    sym! {
        let x = symbol("x");
        assert_eq!(pow(x, constant(1.0)).simplify(), x);
    }
}

#[test]
fn test_simplify_ln_exp() {
    sym! {
        let x = symbol("x");
        assert_eq!(ln(exp(x)).simplify(), x);
    }
}

#[test]
fn test_simplify_exp_ln() {
    sym! {
        let x = symbol("x");
        assert_eq!(exp(ln(x)).simplify(), x);
    }
}

#[test]
fn test_simplify_nested() {
    sym! {
        let x = symbol("x");
        // (x + 0) * 1 should simplify to x
        let e = (x + 0.0) * 1.0;
        assert_eq!(e.simplify(), x);
    }
}

// ============================================================
// Expand
// ============================================================

#[test]
fn test_expand_basic() {
    sym! {
        let x = symbol("x");
        let a = symbol("a");
        let b = symbol("b");
        // x * (a + b) → x*a + x*b
        let e = x * (a + b);
        let expanded = e.expand();
        // Verify by evaluation
        let vars = HashMap::from([("x", 2.0), ("a", 3.0), ("b", 4.0)]);
        assert!(approx_eq(e.eval(&vars).unwrap(), expanded.eval(&vars).unwrap()));
        // Check structure: should be Add
        assert_eq!(format!("{}", &expanded), "a * x + b * x");
    }
}

#[test]
fn test_expand_double() {
    sym! {
        let a = symbol("a");
        let b = symbol("b");
        let c = symbol("c");
        let d = symbol("d");
        let e = (a + b) * (c + d);
        let expanded = e.expand();
        let vars = HashMap::from([("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]);
        assert!(approx_eq(e.eval(&vars).unwrap(), expanded.eval(&vars).unwrap()));
    }
}

#[test]
fn test_expand_pow() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        // (x + y)^2
        let e = pow(x + y, constant(2.0));
        let expanded = e.expand();
        let vars = HashMap::from([("x", 2.0), ("y", 3.0)]);
        assert!(approx_eq(e.eval(&vars).unwrap(), expanded.eval(&vars).unwrap()));
    }
}

// ============================================================
// Collect
// ============================================================

#[test]
fn test_collect_basic() {
    sym! {
        let x = symbol("x");
        let a = symbol("a");
        let b = symbol("b");
        // a*x + b*x → (a + b)*x
        let e = a * x + b * x;
        let collected = e.collect(&x);
        let vars = HashMap::from([("x", 2.0), ("a", 3.0), ("b", 4.0)]);
        assert!(approx_eq(e.eval(&vars).unwrap(), collected.eval(&vars).unwrap()));
    }
}

// ============================================================
// LaTeX output
// ============================================================

#[test]
fn test_latex_frac() {
    let x = symbol("x");
    let y = symbol("y");
    let e = x / y;
    assert_eq!(e.to_latex(), "\\frac{x}{y}");
}

#[test]
fn test_latex_pow() {
    let x = symbol("x");
    let e = pow(x, constant(2.0));
    assert_eq!(e.to_latex(), "x^{2}");
}

#[test]
fn test_latex_sqrt() {
    let x = symbol("x");
    assert_eq!(sqrt(x).to_latex(), "\\sqrt{x}");
}

#[test]
fn test_latex_sin() {
    let x = symbol("x");
    assert_eq!(sin(x).to_latex(), "\\sin\\left(x\\right)");
}

#[test]
fn test_latex_abs() {
    let x = symbol("x");
    assert_eq!(abs(x).to_latex(), "\\left|x\\right|");
}

#[test]
fn test_latex_exp() {
    let x = symbol("x");
    assert_eq!(exp(x).to_latex(), "e^{x}");
}

// ============================================================
// Rust code output
// ============================================================

#[test]
fn test_rust_f64() {
    let x = symbol("x");
    let e = sin(x);
    assert_eq!(e.to_rust("f64"), "x.sin()");
}

#[test]
fn test_rust_f32() {
    let x = symbol("x");
    let e = sin(x);
    assert_eq!(e.to_rust("f32"), "x.sin()");
}

#[test]
fn test_rust_const_f64() {
    let e = constant(2.0);
    assert_eq!(e.to_rust("f64"), "2.0_f64");
}

#[test]
fn test_rust_const_f32() {
    let e = constant(2.0);
    assert_eq!(e.to_rust("f32"), "2.0_f32");
}

#[test]
fn test_rust_pow() {
    let x = symbol("x");
    let e = pow(x, constant(2.0));
    assert_eq!(e.to_rust("f64"), "x.powf(2.0_f64)");
}

#[test]
fn test_rust_add() {
    let x = symbol("x");
    let y = symbol("y");
    assert_eq!((x + y).to_rust("f64"), "x + y");
}

#[test]
fn test_mul_commutative_equality() {
    // a*b and b*a should be structurally equal after simplification
    let x = symbol("x");
    let y = symbol("y");
    let ab = sin(x.clone()) * cos(y.clone());
    let ba = cos(y) * sin(x);
    assert_eq!(ab, ba, "multiplication should be commutative: {} != {}", ab, ba);
}

#[test]
fn test_mul_commutative_three_factors() {
    let a = symbol("a");
    let b = symbol("b");
    let c = symbol("c");
    let abc = a.clone() * b.clone() * c.clone();
    let cab = c * a * b;
    assert_eq!(abc, cab, "3-way product should be equal regardless of order");
}

// ============================================================
// Linear algebra — SymVec
// ============================================================

#[test]
fn test_symvec_add() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let v1 = SymVec::new(vec![x, y]);
        let v2 = SymVec::new(vec![y, x]);
        let sum = v1 + v2;
        let vars = HashMap::from([("x", 1.0), ("y", 2.0)]);
        let result = sum.eval(&vars).unwrap();
        assert!(approx_eq(result[0], 3.0));
        assert!(approx_eq(result[1], 3.0));
    }
}

#[test]
fn test_symvec_scalar_mul() {
    sym! {
        let x = symbol("x");
        let v = SymVec::new(vec![x, constant(2.0)]);
        let scaled = v * constant(3.0);
        let vars = HashMap::from([("x", 4.0)]);
        let result = scaled.eval(&vars).unwrap();
        assert!(approx_eq(result[0], 12.0));
        assert!(approx_eq(result[1], 6.0));
    }
}

#[test]
fn test_symvec_dot() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let v1 = SymVec::new(vec![x, y]);
        let v2 = SymVec::new(vec![y, x]);
        let dot = v1.dot(&v2);
        let vars = HashMap::from([("x", 2.0), ("y", 3.0)]);
        // x*y + y*x = 2*x*y = 12
        assert!(approx_eq(dot.eval(&vars).unwrap(), 12.0));
    }
}

#[test]
fn test_symvec_diff() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let v = SymVec::new(vec![pow(x, constant(2.0)), x * y]);
        let dv = v.diff("x");
        let vars = HashMap::from([("x", 3.0), ("y", 4.0)]);
        let result = dv.eval(&vars).unwrap();
        assert!(approx_eq(result[0], 6.0)); // 2x = 6
        assert!(approx_eq(result[1], 4.0)); // y = 4
    }
}

// ============================================================
// Linear algebra — SymMat
// ============================================================

#[test]
fn test_symmat_identity() {
    let id = SymMat::identity(2);
    let vars = HashMap::new();
    let result = id.eval(&vars).unwrap();
    assert!(approx_eq(result[0][0], 1.0));
    assert!(approx_eq(result[0][1], 0.0));
    assert!(approx_eq(result[1][0], 0.0));
    assert!(approx_eq(result[1][1], 1.0));
}

#[test]
fn test_symmat_zeros() {
    let z = SymMat::zeros(2, 3);
    let vars = HashMap::new();
    let result = z.eval(&vars).unwrap();
    for row in &result {
        for val in row {
            assert!(approx_eq(*val, 0.0));
        }
    }
}

#[test]
fn test_symmat_add() {
    sym! {
        let x = symbol("x");
        let m1 = SymMat::new(1, 2, vec![x, constant(1.0)]);
        let m2 = SymMat::new(1, 2, vec![constant(2.0), x]);
        let sum = m1 + m2;
        let vars = HashMap::from([("x", 3.0)]);
        let result = sum.eval(&vars).unwrap();
        assert!(approx_eq(result[0][0], 5.0));
        assert!(approx_eq(result[0][1], 4.0));
    }
}

#[test]
fn test_symmat_mul() {
    sym! {
        let a = symbol("a");
        let b = symbol("b");
        let c = symbol("c");
        let d = symbol("d");
        // 2x2 * 2x2
        let m1 = SymMat::new(2, 2, vec![a, b, c, d]);
        let m2 = SymMat::identity(2);
        let result = (m1.clone() * m2).simplify();
        let vars = HashMap::from([("a", 1.0), ("b", 2.0), ("c", 3.0), ("d", 4.0)]);
        let ev_orig = m1.eval(&vars).unwrap();
        let ev_result = result.eval(&vars).unwrap();
        assert!(approx_eq(ev_orig[0][0], ev_result[0][0]));
        assert!(approx_eq(ev_orig[0][1], ev_result[0][1]));
        assert!(approx_eq(ev_orig[1][0], ev_result[1][0]));
        assert!(approx_eq(ev_orig[1][1], ev_result[1][1]));
    }
}

#[test]
fn test_symmat_vec_mul() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let m = SymMat::new(2, 2, vec![constant(1.0), constant(2.0), constant(3.0), constant(4.0)]);
        let v = SymVec::new(vec![x, y]);
        let result = m * v;
        let vars = HashMap::from([("x", 1.0), ("y", 2.0)]);
        let ev = result.eval(&vars).unwrap();
        assert!(approx_eq(ev[0], 5.0));  // 1*1 + 2*2
        assert!(approx_eq(ev[1], 11.0)); // 3*1 + 4*2
    }
}

#[test]
fn test_symmat_transpose() {
    let m = SymMat::new(2, 3, vec![
        constant(1.0), constant(2.0), constant(3.0),
        constant(4.0), constant(5.0), constant(6.0),
    ]);
    let mt = m.transpose();
    assert_eq!(mt.rows, 3);
    assert_eq!(mt.cols, 2);
    let vars = HashMap::new();
    let ev = mt.eval(&vars).unwrap();
    assert!(approx_eq(ev[0][0], 1.0));
    assert!(approx_eq(ev[0][1], 4.0));
    assert!(approx_eq(ev[1][0], 2.0));
    assert!(approx_eq(ev[1][1], 5.0));
    assert!(approx_eq(ev[2][0], 3.0));
    assert!(approx_eq(ev[2][1], 6.0));
}

#[test]
fn test_symmat_diff() {
    sym! {
        let x = symbol("x");
        let m = SymMat::new(1, 2, vec![pow(x, constant(2.0)), x]);
        let dm = m.diff("x");
        let vars = HashMap::from([("x", 3.0)]);
        let ev = dm.eval(&vars).unwrap();
        assert!(approx_eq(ev[0][0], 6.0)); // 2x
        assert!(approx_eq(ev[0][1], 1.0)); // 1
    }
}

#[test]
fn test_jacobian_as_symmat() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let f1 = x * y;
        let f2 = x + y;
        let j = jacobian(&[f1, f2], &["x", "y"]);
        assert_eq!(j.rows, 2);
        assert_eq!(j.cols, 2);
    }
}

// ============================================================
// SymVec / SymMat formatters
// ============================================================

#[test]
fn test_symvec_display() {
    let v = SymVec::new(vec![symbol("x"), symbol("y")]);
    assert_eq!(format!("{v}"), "[x, y]");
}

#[test]
fn test_symmat_display() {
    let m = SymMat::new(2, 2, vec![
        symbol("a"), symbol("b"),
        symbol("c"), symbol("d"),
    ]);
    assert_eq!(format!("{m}"), "[a, b; c, d]");
}

#[test]
fn test_symvec_latex() {
    let v = SymVec::new(vec![symbol("x"), symbol("y")]);
    assert_eq!(v.to_latex(), "\\begin{pmatrix} x \\\\ y \\end{pmatrix}");
}

#[test]
fn test_symmat_latex() {
    let m = SymMat::new(1, 2, vec![symbol("x"), symbol("y")]);
    assert_eq!(m.to_latex(), "\\begin{pmatrix} x & y \\end{pmatrix}");
}

// ============================================================
// New tests — polish round
// ============================================================

#[test]
fn test_display_neg_neg() {
    sym! {
        let x = symbol("x");
        // auto-simplify: -(-x) → x
        let e = -(-x);
        assert_eq!(format!("{e}"), "x");
    }
}

#[test]
fn test_c_alias() {
    assert_eq!(c(5.0), constant(5.0));
}

#[test]
fn test_free_vars_sorted() {
    sym! {
        let z = symbol("z");
        let a = symbol("a");
        let m = symbol("m");
        let e = z + a + m;
        let fv: Vec<String> = e.free_vars().into_iter().collect();
        assert_eq!(fv, vec!["a", "m", "z"]);
    }
}

#[test]
fn test_simplify_x_times_x() {
    sym! {
        let x = symbol("x");
        assert_eq!((x * x).simplify(), pow(x, constant(2.0)));
    }
}

#[test]
fn test_simplify_x_plus_x() {
    sym! {
        let x = symbol("x");
        assert_eq!((x + x).simplify(), constant(2.0) * x);
    }
}

#[test]
fn test_simplify_like_terms() {
    sym! {
        let x = symbol("x");
        let e = constant(3.0) * x + constant(2.0) * x;
        assert_eq!(e.simplify(), constant(5.0) * x);
    }
}

#[test]
fn test_higher_order_derivative() {
    sym! {
        let x = symbol("x");
        let e = pow(x, constant(4.0));
        let d2 = e.diff("x").simplify().diff("x").simplify();
        assert_eq!(d2, constant(12.0) * pow(x, constant(2.0)));
    }
}

// ============================================================
// Fraction simplification (SymPy-comparison tests)
// ============================================================

#[test]
fn test_div_chain_flattens() {
    sym! {
        let a = symbol("a");
        let b = symbol("b");
        let c_sym = symbol("c");
        let e = (a / b / c_sym).simplify();
        assert_eq!(format!("{e}"), "a / (b * c)");
    }
}

#[test]
fn test_cross_fraction_cancel() {
    sym! {
        let a = symbol("a");
        let b = symbol("b");
        let e = (a * pow(b, c(2.0)) / b).simplify();
        assert_eq!(e, a * b);
    }
}

#[test]
fn test_power_cancel_in_fraction() {
    sym! {
        let g = symbol("gamma");
        let e = (pow(g, c(3.0)) / pow(g, c(2.0))).simplify();
        assert_eq!(e, g);
    }
}

#[test]
fn test_reciprocal_div_flatten() {
    // a / (b / c) → a * c / b
    sym! {
        let a = symbol("a");
        let b = symbol("b");
        let c_sym = symbol("c");
        let e = (a / (b / c_sym)).simplify();
        assert_eq!(format!("{e}"), "a * c / b");
    }
}

#[test]
fn test_diff_with_symbol_arg() {
    sym! {
        let x = symbol("x");
        let e = pow(x, c(3.0));
        assert_eq!(e.diff("x").simplify(), e.diff(x).simplify());
    }
}

#[test]
fn test_sympy_error_suppression_numerical() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let gamma = symbol("gamma");
        let a = symbol("a");
        let b = symbol("b");
        let sigma = symbol("sigma");
        let r = (y - (a * x + b)) / sigma;
        let rs = gamma * atan(r / gamma);
        let s_expr = pow(rs, c(2.0));

        // Numerical check: derivative is correct
        let vars = HashMap::from([
            ("x", 1.0), ("y", 5.0), ("gamma", 2.0),
            ("a", 0.5), ("b", 1.0), ("sigma", 3.0),
        ]);
        check_diff_numerically(&s_expr, "x", &vars, 1e-6);

        // Structural: verify it simplifies to a single top-level fraction
        let ds = s_expr.diff(x).simplify();
        // Should be Div(num, den) at top level — not Div(Div(...))
        match ds.as_ref() {
            arael::sym::Expr::Div(_, den) => {
                // Denominator should not itself be a Div
                assert!(!matches!(den.as_ref(), arael::sym::Expr::Div(..)),
                    "derivative has nested division: {ds}");
            }
            _ => {} // Not a Div at all is also fine
        }
    }
}

// ============================================================
// Parser tests
// ============================================================

#[test]
fn test_parse_basic_arithmetic() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        assert_eq!(parse("x + y").unwrap(), x + y);
        assert_eq!(parse("x - y").unwrap(), x - y);
        assert_eq!(parse("x * y").unwrap(), x * y);
        assert_eq!(parse("x / y").unwrap(), x / y);
        assert_eq!(parse("2 * x").unwrap(), c(2.0) * x);
        assert_eq!(parse("3.14").unwrap(), constant(3.14));
    }
}

#[test]
fn test_parse_precedence() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let z = symbol("z");
        // * binds tighter than +
        assert_eq!(parse("x + y * z").unwrap(), x + y * z);
        // parentheses override
        assert_eq!(parse("(x + y) * z").unwrap(), (x + y) * z);
    }
}

#[test]
fn test_parse_power() {
    sym! {
        let x = symbol("x");
        assert_eq!(parse("x^2").unwrap(), pow(x, c(2.0)));
        // right-associative
        assert_eq!(parse("x^2^3").unwrap(), pow(x, pow(c(2.0), c(3.0))));
    }
}

#[test]
fn test_parse_unary_minus() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        assert_eq!(parse("-x").unwrap(), -x);
        assert_eq!(parse("x + -y").unwrap(), x + (-y));
        assert_eq!(parse("-3").unwrap(), constant(-3.0));
    }
}

#[test]
fn test_parse_functions() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        assert_eq!(parse("sin(x)").unwrap(), sin(x));
        assert_eq!(parse("cos(x)").unwrap(), cos(x));
        assert_eq!(parse("tan(x)").unwrap(), tan(x));
        assert_eq!(parse("exp(x)").unwrap(), exp(x));
        assert_eq!(parse("ln(x)").unwrap(), ln(x));
        assert_eq!(parse("sqrt(x)").unwrap(), sqrt(x));
        assert_eq!(parse("abs(x)").unwrap(), abs(x));
        assert_eq!(parse("asin(x)").unwrap(), asin(x));
        assert_eq!(parse("acos(x)").unwrap(), acos(x));
        assert_eq!(parse("atan(x)").unwrap(), atan(x));
        assert_eq!(parse("sinh(x)").unwrap(), sinh(x));
        assert_eq!(parse("cosh(x)").unwrap(), cosh(x));
        assert_eq!(parse("tanh(x)").unwrap(), tanh(x));
        assert_eq!(parse("log2(x)").unwrap(), log2(x));
        assert_eq!(parse("log10(x)").unwrap(), log10(x));
        assert_eq!(parse("atan2(y, x)").unwrap(), atan2(y, x));
        assert_eq!(parse("pow(x, 2)").unwrap(), pow(x, c(2.0)));
    }
}

#[test]
fn test_parse_named_constants() {
    assert_eq!(parse("pi").unwrap(), constant(std::f64::consts::PI));
    assert_eq!(parse("e").unwrap(), constant(std::f64::consts::E));
}

#[test]
fn test_parse_nested() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        assert_eq!(
            parse("sqrt(atan2(y, x) + pi)").unwrap(),
            sqrt(atan2(y, x) + constant(std::f64::consts::PI))
        );
        assert_eq!(
            parse("exp(sin(x))").unwrap(),
            exp(sin(x))
        );
    }
}

#[test]
fn test_parse_errors() {
    assert!(parse("").is_err());
    assert!(parse("sin(x, y)").is_err()); // sin takes 1 arg
    assert!(parse("foo(x)").is_err());     // unknown function
    assert!(parse("x +").is_err());        // unexpected eof
    assert!(parse("x )").is_err());        // unexpected token
}

#[test]
fn test_parse_fromstr() {
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let e: E = "x + y".parse().unwrap();
        assert_eq!(e, x + y);
    }
}

#[test]
fn test_parse_complex_expression() {
    // Parse and evaluate should give same result as building programmatically
    sym! {
        let x = symbol("x");
        let parsed = parse("x^2 + 3*x + 1").unwrap();
        let built = pow(x, c(2.0)) + c(3.0) * x + c(1.0);
        let vars = HashMap::from([("x", 2.0)]);
        assert!(approx_eq(parsed.eval(&vars).unwrap(), built.eval(&vars).unwrap()));
    }
}

// ============================================================
// CSE tests
// ============================================================

#[test]
fn test_cse_no_common() {
    let x = symbol("x");
    let y = symbol("y");
    let e1 = x + c(1.0);
    let e2 = y + c(2.0);
    let (intermediates, results) = cse(&[e1, e2]);
    assert!(intermediates.is_empty());
    assert_eq!(results.len(), 2);
}

#[test]
fn test_cse_shared_subexpr() {
    // (x + y) appears in both expressions
    let x = symbol("x");
    let y = symbol("y");
    let (e1, e2) = arael::sym! {
        let sum = x + y;
        let e1 = sum * c(2.0);
        let e2 = sum * c(3.0);
        (e1, e2)
    };
    let (intermediates, results) = cse(&[e1, e2]);
    assert!(!intermediates.is_empty(), "should extract x+y");
    assert_eq!(results.len(), 2);
    // Evaluate: build vars including intermediate values
    let mut vars: HashMap<&str, f64> = HashMap::from([("x", 3.0), ("y", 4.0)]);
    for (name, expr) in &intermediates {
        let val = expr.eval(&vars).unwrap();
        vars.insert(name.as_str(), val);
    }
    assert!(approx_eq(results[0].eval(&vars).unwrap(), 14.0)); // (3+4)*2
    assert!(approx_eq(results[1].eval(&vars).unwrap(), 21.0)); // (3+4)*3
}

/// Helper: evaluate CSE result by building vars map with intermediate values
fn eval_cse(intermediates: &[(String, E)], expr: &E, base_vars: &HashMap<&str, f64>) -> f64 {
    let mut vars = base_vars.clone();
    for (name, e) in intermediates {
        let val = e.eval(&vars).unwrap();
        vars.insert(name.as_str(), val);
    }
    expr.eval(&vars).unwrap()
}

#[test]
fn test_cse_constants_not_extracted() {
    let x = symbol("x");
    let e1 = x.clone() + x.clone();
    let e2 = x.clone() * x.clone();
    let (intermediates, _results) = cse(&[e1, e2]);
    for (_, expr) in &intermediates {
        assert!(!matches!(*expr.as_ref(), Expr::Sym(_)),
            "should not extract bare symbols");
    }
}

#[test]
fn test_cse_trig() {
    let x = symbol("x");
    let sx = sin(x);
    let e1 = sx.clone() * sx.clone();
    let e2 = sx + c(1.0);
    let (intermediates, results) = cse(&[e1, e2]);
    assert!(!intermediates.is_empty(), "should extract sin(x)");
    let vars = HashMap::from([("x", 1.0)]);
    let sin1 = 1.0_f64.sin();
    assert!(approx_eq(eval_cse(&intermediates, &results[0], &vars), sin1 * sin1));
    assert!(approx_eq(eval_cse(&intermediates, &results[1], &vars), sin1 + 1.0));
}

#[test]
fn test_cse_nested() {
    let x = symbol("x");
    let y = symbol("y");
    let z = symbol("z");
    let sum = x + y;
    let prod = sum.clone() * z;
    let e1 = prod.clone() + sum;
    let e2 = prod * c(2.0);
    let (intermediates, results) = cse(&[e1, e2]);
    assert!(!intermediates.is_empty(), "should extract at least one subexpr");
    let vars = HashMap::from([("x", 2.0), ("y", 3.0), ("z", 4.0)]);
    assert!(approx_eq(eval_cse(&intermediates, &results[0], &vars), 25.0));
    assert!(approx_eq(eval_cse(&intermediates, &results[1], &vars), 40.0));
}

#[test]
fn test_cse_multiple_expressions() {
    let a = symbol("a");
    let b = symbol("b");
    let ab = a * b;
    let e1 = ab.clone() + c(1.0);
    let e2 = ab.clone() + c(2.0);
    let e3 = ab + c(3.0);
    let (intermediates, results) = cse(&[e1, e2, e3]);
    assert!(!intermediates.is_empty());
    let vars = HashMap::from([("a", 5.0), ("b", 7.0)]);
    assert!(approx_eq(eval_cse(&intermediates, &results[0], &vars), 36.0));
    assert!(approx_eq(eval_cse(&intermediates, &results[1], &vars), 37.0));
    assert!(approx_eq(eval_cse(&intermediates, &results[2], &vars), 38.0));
}

#[test]
fn test_cse_preserves_values() {
    let x = symbol("x");
    let y = symbol("y");
    let xy = x * y;
    let e1 = sin(xy.clone()) + cos(xy.clone());
    let e2 = sin(xy.clone()) * cos(xy);
    let vars = HashMap::from([("x", 1.5), ("y", 2.3)]);
    let orig1 = e1.eval(&vars).unwrap();
    let orig2 = e2.eval(&vars).unwrap();
    let (intermediates, results) = cse(&[e1, e2]);
    assert!(approx_eq(eval_cse(&intermediates, &results[0], &vars), orig1),
        "CSE changed result 0");
    assert!(approx_eq(eval_cse(&intermediates, &results[1], &vars), orig2),
        "CSE changed result 1");
}

#[test]
fn test_cse_empty() {
    let (intermediates, results) = cse(&[]);
    assert!(intermediates.is_empty());
    assert!(results.is_empty());
}

#[test]
fn test_cse_single_expression() {
    let x = symbol("x");
    let sx = sin(x);
    let e = sx.clone() * sx.clone() + sx;
    let (intermediates, results) = cse(&[e]);
    assert!(!intermediates.is_empty(), "should extract sin(x) from single expr");
    let vars = HashMap::from([("x", 0.7)]);
    let sin07 = 0.7_f64.sin();
    assert!(approx_eq(eval_cse(&intermediates, &results[0], &vars), sin07 * sin07 + sin07));
}

#[test]
fn test_cse_derivative_sharing() {
    // After differentiation, common subexpressions across residual + derivatives
    let x = symbol("x");
    let y = symbol("y");
    let r = sin(x.clone() * y.clone()) + cos(x.clone() * y.clone());
    let dr_x = r.diff("x");
    let dr_y = r.diff("y");
    let vars = HashMap::from([("x", 1.2), ("y", 0.8)]);
    let orig_r = r.eval(&vars).unwrap();
    let orig_dx = dr_x.eval(&vars).unwrap();
    let orig_dy = dr_y.eval(&vars).unwrap();
    let (intermediates, results) = cse(&[r, dr_x, dr_y]);
    // x*y, sin(x*y), cos(x*y) should all be extracted
    assert!(intermediates.len() >= 3, "should extract at least 3 subexprs, got {}", intermediates.len());
    assert!(approx_eq(eval_cse(&intermediates, &results[0], &vars), orig_r));
    assert!(approx_eq(eval_cse(&intermediates, &results[1], &vars), orig_dx));
    assert!(approx_eq(eval_cse(&intermediates, &results[2], &vars), orig_dy));
}

#[test]
fn test_cse_chain_rule_sharing() {
    // Derivatives via chain rule create shared inner derivatives
    let x = symbol("x");
    let inner = x.clone() * x.clone() + c(1.0); // x^2 + 1
    let f = sin(inner.clone());
    let g = cos(inner);
    let df = f.diff("x");
    let dg = g.diff("x");
    let vars = HashMap::from([("x", 1.5)]);
    let orig_df = df.eval(&vars).unwrap();
    let orig_dg = dg.eval(&vars).unwrap();
    let (intermediates, results) = cse(&[df.clone(), dg.clone()]);
    assert!(!intermediates.is_empty(), "should extract chain rule subexprs");
    assert!(approx_eq(eval_cse(&intermediates, &results[0], &vars), orig_df));
    assert!(approx_eq(eval_cse(&intermediates, &results[1], &vars), orig_dg));
}

#[test]
fn test_cse_rotation_matrix_like() {
    // Simulate rotation matrix pattern: sin/cos used in multiple products
    let a = symbol("a");
    let sa = sin(a.clone());
    let ca = cos(a);
    let e1 = sa.clone() * sa.clone() + ca.clone() * ca.clone(); // should simplify to 1, but CSE doesn't simplify
    let e2 = sa.clone() * ca.clone() * c(2.0);
    let e3 = ca.clone() * ca.clone() - sa.clone() * sa.clone();
    let vars = HashMap::from([("a", 0.7)]);
    let orig = [e1.eval(&vars).unwrap(), e2.eval(&vars).unwrap(), e3.eval(&vars).unwrap()];
    let (intermediates, results) = cse(&[e1, e2, e3]);
    assert!(intermediates.len() >= 2, "should extract sin(a) and cos(a)");
    for (i, r) in results.iter().enumerate() {
        assert!(approx_eq(eval_cse(&intermediates, r, &vars), orig[i]),
            "result {} differs", i);
    }
}

#[test]
fn test_cse_many_expressions() {
    // 10 expressions sharing substructure
    let x = symbol("x");
    let y = symbol("y");
    let common = sin(x.clone()) * cos(y.clone());
    let exprs: Vec<E> = (0..10).map(|i| {
        common.clone() + c(i as f64)
    }).collect();
    let vars = HashMap::from([("x", 1.0), ("y", 2.0)]);
    let origs: Vec<f64> = exprs.iter().map(|e| e.eval(&vars).unwrap()).collect();
    let (intermediates, results) = cse(&exprs);
    assert!(!intermediates.is_empty());
    for (i, r) in results.iter().enumerate() {
        assert!(approx_eq(eval_cse(&intermediates, r, &vars), origs[i]),
            "result {} differs", i);
    }
}

#[test]
fn test_cse_negation() {
    // Negated subexpressions should be extracted if repeated
    let x = symbol("x");
    let y = symbol("y");
    let sum = x + y;
    let neg_sum = -sum.clone();
    let e1 = neg_sum.clone() * c(2.0);
    let e2 = neg_sum * c(3.0);
    let vars = HashMap::from([("x", 3.0), ("y", 4.0)]);
    let (intermediates, results) = cse(&[e1, e2]);
    assert!(!intermediates.is_empty());
    assert!(approx_eq(eval_cse(&intermediates, &results[0], &vars), -14.0));
    assert!(approx_eq(eval_cse(&intermediates, &results[1], &vars), -21.0));
}

#[test]
fn test_cse_division_reciprocal() {
    // 1/x used multiple times
    let x = symbol("x");
    let y = symbol("y");
    let e1 = y.clone() / x.clone();
    let e2 = c(2.0) / x;
    let vars = HashMap::from([("x", 4.0), ("y", 8.0)]);
    let (intermediates, results) = cse(&[e1, e2]);
    // May or may not extract 1/x depending on how division is represented
    assert!(approx_eq(eval_cse(&intermediates, &results[0], &vars), 2.0));
    assert!(approx_eq(eval_cse(&intermediates, &results[1], &vars), 0.5));
}

#[test]
fn test_cse_to_rust_output() {
    // Verify CSE + to_rust produces compilable code (no excess parens)
    let x = symbol("x");
    let y = symbol("y");
    let sum = x + y;
    let e1 = sum.clone() * c(2.0);
    let e2 = sum * c(3.0);
    let (intermediates, results) = cse(&[e1, e2]);
    for (name, expr) in &intermediates {
        let code = expr.to_rust("f64");
        assert!(!code.starts_with('(') || code.contains("(x + y)"),
            "intermediate {} has unnecessary outer parens: {}", name, code);
    }
    for r in &results {
        let code = r.to_rust("f64");
        // Should be clean: __x0 * 2.0_f64 (no outer parens)
        assert!(!code.is_empty(), "empty result code");
    }
}

#[test]
fn test_cse_large_expression_count() {
    // Stress test with many expressions
    let x = symbol("x");
    let base = sin(x.clone()) + cos(x);
    let exprs: Vec<E> = (0..50).map(|i| base.clone() * c(i as f64)).collect();
    let (intermediates, results) = cse(&exprs);
    assert!(!intermediates.is_empty());
    assert_eq!(results.len(), 50);
    let vars = HashMap::from([("x", 0.5)]);
    let base_val = 0.5_f64.sin() + 0.5_f64.cos();
    for (i, r) in results.iter().enumerate() {
        assert!(approx_eq(eval_cse(&intermediates, r, &vars), base_val * i as f64),
            "result {} differs", i);
    }
}

#[test]
fn test_mixed_ops_with_i64() {
    // Bare integer literals (no `.0` suffix) must compose with E in
    // both directions for every basic operator. Without the i64
    // impls this test fails to compile. Exact output strings depend
    // on the simplifier's canonical ordering -- we verify numeric
    // equivalence at x = 4 instead.
    use arael_sym::*;
    use std::collections::HashMap;
    let x = symbol("x");
    let vars = HashMap::from([("x", 4.0)]);

    let cases: &[(E, f64)] = &[
        (x.clone() + 2, 4.0 + 2.0),
        (2 + x.clone(), 2.0 + 4.0),
        (x.clone() - 2, 4.0 - 2.0),
        (5 - x.clone(), 5.0 - 4.0),
        (x.clone() * 3, 4.0 * 3.0),
        (3 * x.clone(), 3.0 * 4.0),
        (x.clone() / 4, 4.0 / 4.0),
        (8 / x.clone(), 8.0 / 4.0),
    ];
    for (i, (expr, expected)) in cases.iter().enumerate() {
        let v = expr.eval(&vars).unwrap();
        assert!((v - expected).abs() < 1e-12,
            "case {}: got {} expected {} (expr: {})", i, v, expected, expr);
    }
}

#[test]
fn test_eval_unbound_symbol() {
    use std::collections::HashMap;
    let expr = arael_sym::symbol("x") + arael_sym::symbol("y");
    let mut vars = HashMap::new();
    vars.insert("x", 1.0);
    // "y" is missing -- should return Err, not panic
    let result = expr.eval(&vars);
    assert!(result.is_err());
    assert!(result.unwrap_err().contains("unbound symbol: y"));

    // With both vars bound, should succeed
    vars.insert("y", 2.0);
    assert!((expr.eval(&vars).unwrap() - 3.0).abs() < 1e-10);
}

// ============================================================
// Power-of-product expansion: (a*b)^n -> a^n * b^n is only valid
// for integer exponents. For fractional n and negative factors the
// two sides differ ((xy)^0.5 = 1 but x^0.5 * y^0.5 = NaN at
// x = y = -1), so simplify must leave those unexpanded.
// ============================================================

#[test]
fn test_pow_product_fractional_exponent_not_distributed() {
    use std::collections::HashMap;
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let e = pow(x * y, constant(0.5)) * symbol("z");
        let s = e.simplify();
        // Value must be preserved where the original is defined:
        // (-1 * -1)^0.5 * 3 = 3. Distribution would produce NaN.
        let vars = HashMap::from([("x", -1.0), ("y", -1.0), ("z", 3.0)]);
        let v = s.eval(&vars).unwrap();
        assert!((v - 3.0).abs() < 1e-12, "got {} from {}", v, s);
    }
}

#[test]
fn test_pow_negated_base_fractional_exponent_no_nan_coefficient() {
    use std::collections::HashMap;
    sym! {
        let x = symbol("x");
        // (-2 * x)^0.5: expansion would bake the coefficient
        // (-2)^0.5 = NaN into the tree.
        // The outer multiplication forces the flatten_mul path where the
        // expansion lives.
        let e = pow(constant(-2.0) * x, constant(0.5)) * symbol("y");
        let s = e.simplify();
        let vars = HashMap::from([("x", -1.0), ("y", 3.0)]);
        let v = s.eval(&vars).unwrap();
        // ((-2)*(-1))^0.5 * 3 = 3*sqrt(2)
        assert!((v - 3.0 * 2.0_f64.sqrt()).abs() < 1e-12, "got {} from {}", v, s);
    }
}

#[test]
fn test_pow_product_integer_exponents_still_expand_and_agree() {
    use std::collections::HashMap;
    sym! {
        let x = symbol("x");
        let y = symbol("y");
        let vars = HashMap::from([("x", 3.0), ("y", 5.0)]);
        // Positive, negative, and zero integer exponents: expansion is
        // valid and values must agree with the unsimplified expression.
        for n in [2.0, 3.0, -1.0, -2.0, 0.0] {
            let e = pow(x * y, constant(n)) * constant(7.0);
            let s = e.simplify();
            let expect = (15.0_f64).powf(n) * 7.0;
            let v = s.eval(&vars).unwrap();
            assert!((v - expect).abs() < 1e-9 * expect.abs().max(1.0),
                "n={}: got {} expected {} (simplified: {})", n, v, expect, s);
        }
        // The square actually expands structurally (the point of the
        // rule). Expansion happens where the Pow participates in a
        // product, so wrap it in one:
        let sq = (pow(x * y, constant(2.0)) * symbol("z")).simplify();
        let shown = format!("{}", sq);
        assert!(!shown.contains("(x * y)") && shown.contains("x^2"),
            "expected expanded form, got {}", shown);
        // Negated base with integer exponent: sign handled via coefficient.
        let neg = pow(-x, constant(3.0)) * symbol("y");
        let vars2 = HashMap::from([("x", 2.0), ("y", 4.0)]);
        let v2 = neg.simplify().eval(&vars2).unwrap();
        assert!((v2 - (-32.0)).abs() < 1e-12, "got {}", v2);
    }
}

#[test]
fn cse_output_is_deterministic_across_runs() {
    // CSE candidate selection and divisor extraction iterate hash maps;
    // per-instance random state used to break ties randomly, so two
    // structurally identical inputs could emit differently-named/ordered
    // temporaries (nondeterministic generated code, B23). Build the same
    // expression set from scratch many times: every run must produce the
    // identical intermediate list and rewritten expressions. The
    // expression is chosen to have MANY equal-savings candidates so a
    // random tie-break would be caught quickly.
    let build = || -> Vec<E> {
        let mk = |n: &str| symbol(n);
        let mut out = Vec::new();
        for pair in [("a", "b"), ("c", "d"), ("e2", "f"), ("g", "h")] {
            let (x, y) = (mk(pair.0), mk(pair.1));
            let u = sin(x.clone() * y.clone()) + cos(x.clone() * y.clone());
            let v = sin(x.clone() * y.clone()) - cos(x.clone() * y.clone());
            let w = (x.clone() + y.clone()) / (x.clone() * y.clone());
            out.push(u.clone() * v.clone() + w.clone());
            out.push(u * v - w);
        }
        out
    };

    let render = |exprs: &[E]| -> String {
        let (intermediates, results) = cse(exprs);
        let mut s = String::new();
        for (name, e) in &intermediates {
            s.push_str(&format!("{} = {}\n", name, e));
        }
        for r in &results {
            s.push_str(&format!("{}\n", r));
        }
        s
    };

    let first = render(&build());
    for _ in 0..20 {
        assert_eq!(render(&build()), first, "CSE output changed between runs");
    }
}
