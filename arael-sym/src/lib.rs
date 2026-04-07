//! Symbolic math library for expression trees, automatic differentiation,
//! simplification, and code generation.
//!
//! `arael-sym` provides a lightweight computer algebra system built around a
//! reference-counted expression tree ([`E`]).  Expressions are constructed from
//! symbols and constants, combined with standard arithmetic operators (which
//! auto-simplify), and then differentiated, evaluated, pretty-printed, or
//! compiled to Rust source code.
//!
//! This crate is the symbolic engine behind the
//! [`arael`](https://docs.rs/arael) optimization framework, where it powers
//! compile-time constraint differentiation and code generation. It can also
//! be used independently for any symbolic math task.
//!
//! # Scope and limitations
//!
//! `arael-sym` is focused on what's needed for nonlinear optimization:
//! scalar expressions, differentiation, and code generation. Compared to
//! a full CAS like Python's SymPy, it does **not** support:
//!
//! - Symbolic integration
//! - Equation solving (solve for x)
//! - Symbolic matrix algebra (symbolic determinant, inverse, eigenvalues)
//! - Polynomial factoring, GCD, partial fractions
//! - Limits, series expansion, Taylor series
//! - Assumptions / domain reasoning (positive, real, integer)
//! - Pattern matching / rewrite rules
//! - Pretty-printing of intermediate simplification steps
//!
//! # Examples
//!
//! The [`sym!`] macro auto-inserts `.clone()` on reused variables, so you
//! can write natural math without ownership boilerplate.
//!
//! ## Basics
//!
//! ```
//! use arael_sym::*;
//! let result = sym! {
//!     let x = symbol("x");
//!     let f = x * x + 3.0 * x + 1.0;
//!     format!("{}", f)
//! };
//! assert_eq!(result, "x^2 + 3 * x + 1");
//! ```
//!
//! ## Differentiation
//!
//! ```
//! use arael_sym::*;
//! let result = sym! {
//!     let x = symbol("x");
//!     let f = sin(x) * x;
//!     // Product rule + chain rule applied automatically:
//!     format!("{}", f.diff("x"))
//! };
//! assert_eq!(result, "x * cos(x) + sin(x)");
//! ```
//!
//! ## Evaluation
//!
//! ```
//! use arael_sym::*;
//! let val = sym! {
//!     let x = symbol("x");
//!     let f = x * x + 1.0;
//!     let vars = std::collections::HashMap::from([("x", 3.0)]);
//!     f.eval(&vars).unwrap()
//! };
//! assert_eq!(val, 10.0);
//! ```
//!
//! ## Code generation
//!
//! ```
//! use arael_sym::*;
//! let (code1, code2) = sym! {
//!     let f = sin(symbol("x")) + 1.0;
//!     let g = atan2(symbol("y"), symbol("x"));
//!     (f.to_rust("f64"), g.to_rust("f32"))
//! };
//! assert_eq!(code1, "x.sin() + 1.0_f64");
//! assert_eq!(code2, "y.atan2(x)");
//! ```
//!
//! ## Common Subexpression Elimination (CSE)
//!
//! ```
//! use arael_sym::*;
//! sym! {
//!     let x = symbol("x");
//!     let shared = sin(x) * cos(x);
//!     let e1 = shared + 1.0;
//!     let e2 = shared * 2.0;
//!     let (intermediates, simplified) = cse(&[e1, e2]);
//!     for (name, val) in &intermediates {
//!         println!("let {} = {};", name, val);
//!     }
//!     for s in &simplified {
//!         println!("{}", s);
//!     }
//! };
//! // Output:
//! //   let __x0 = cos(x) * sin(x);
//! //   __x0 + 1
//! //   2 * __x0
//! ```
//!
//! ## Vectors and Matrices
//!
//! ```
//! use arael_sym::*;
//! let dot = sym! {
//!     let v = SymVec::new(vec![symbol("x"), symbol("y"), symbol("z")]);
//!     let w = SymVec::new(vec![c(1.0), c(2.0), c(3.0)]);
//!     format!("{}", v.dot(&w))
//! };
//! assert_eq!(dot, "x + 2 * y + 3 * z");
//! ```
//!
//! ## Jacobian
//!
//! ```
//! use arael_sym::*;
//! let (j00, j01, j10, j11) = sym! {
//!     let x = symbol("x");
//!     let y = symbol("y");
//!     let f = vec![x * y, sin(x) + y];
//!     let j = jacobian(&f, &["x", "y"]);
//!     // j is 2x2: [[df0/dx, df0/dy], [df1/dx, df1/dy]]
//!     (format!("{}", j.get(0, 0)),
//!      format!("{}", j.get(0, 1)),
//!      format!("{}", j.get(1, 0)),
//!      format!("{}", j.get(1, 1)))
//! };
//! assert_eq!(j00, "y");      // d(x*y)/dx
//! assert_eq!(j01, "x");      // d(x*y)/dy
//! assert_eq!(j10, "cos(x)"); // d(sin(x)+y)/dx
//! assert_eq!(j11, "1");      // d(sin(x)+y)/dy
//! ```
//!
//!
//! ## Parsing
//!
//! ```
//! use arael_sym::*;
//! let f = parse("sin(x)^2 + cos(x)^2").unwrap();
//! assert_eq!(format!("{}", f), "sin(x)^2 + cos(x)^2");
//!
//! let vars = std::collections::HashMap::from([("x", 1.0)]);
//! assert!((f.eval(&vars).unwrap() - 1.0).abs() < 1e-10);
//! ```

#![allow(clippy::should_implement_trait)]

mod diff;
mod eval;
mod fmt;
mod simplify;
mod linalg;
mod parse;
pub mod geo;
pub mod cse;

use std::hash::{Hash, Hasher};
use std::rc::Rc;

/// Symbolic expression wrapper.
///
/// Reference-counted (cheap to clone).  All arithmetic operations auto-simplify.
/// Dereferences to [`Expr`] so all methods on `Expr` (e.g. [`Expr::diff`],
/// [`Expr::eval`], [`Expr::simplify`]) are available directly on `E`.
#[derive(Clone, PartialEq)]
pub struct E(Rc<Expr>);

impl Eq for E {}

impl E {
    fn new(expr: Expr) -> E {
        E(Rc::new(expr))
    }

    /// Collect all symbol names referenced in this expression.
    pub fn symbols(&self) -> std::collections::HashSet<String> {
        let mut out = std::collections::HashSet::new();
        self.collect_symbols(&mut out);
        out
    }

    fn collect_symbols(&self, out: &mut std::collections::HashSet<String>) {
        match &*self.0 {
            Expr::Sym(s) => { out.insert(s.clone()); }
            Expr::Const(_) => {}
            Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
            | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
            | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
            | Expr::Exp(a) | Expr::Ln(a) | Expr::Log2(a) | Expr::Log10(a)
            | Expr::Sqrt(a) | Expr::Abs(a) => { a.collect_symbols(out); }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
            | Expr::Div(a, b) | Expr::Pow(a, b) | Expr::Atan2(a, b) => {
                a.collect_symbols(out);
                b.collect_symbols(out);
            }
            Expr::Func { args, .. } => {
                for arg in args { arg.collect_symbols(out); }
            }
        }
    }

    /// Substitute symbols in this expression. Each pair `(from, to)` replaces
    /// occurrences of `from` with `to`. Returns a new expression.
    pub fn substitute(&self, subs: &[(E, E)]) -> E {
        for (from, to) in subs {
            if self == from { return to.clone(); }
        }
        match &*self.0 {
            Expr::Sym(_) | Expr::Const(_) => self.clone(),
            Expr::Neg(a) => -a.substitute(subs),
            Expr::Add(a, b) => a.substitute(subs) + b.substitute(subs),
            Expr::Sub(a, b) => a.substitute(subs) - b.substitute(subs),
            Expr::Mul(a, b) => a.substitute(subs) * b.substitute(subs),
            Expr::Div(a, b) => a.substitute(subs) / b.substitute(subs),
            Expr::Pow(a, b) => pow(a.substitute(subs), b.substitute(subs)),
            Expr::Sin(a) => sin(a.substitute(subs)),
            Expr::Cos(a) => cos(a.substitute(subs)),
            Expr::Tan(a) => tan(a.substitute(subs)),
            Expr::Asin(a) => asin(a.substitute(subs)),
            Expr::Acos(a) => acos(a.substitute(subs)),
            Expr::Atan(a) => atan(a.substitute(subs)),
            Expr::Atan2(a, b) => atan2(a.substitute(subs), b.substitute(subs)),
            Expr::Sinh(a) => sinh(a.substitute(subs)),
            Expr::Cosh(a) => cosh(a.substitute(subs)),
            Expr::Tanh(a) => tanh(a.substitute(subs)),
            Expr::Exp(a) => exp(a.substitute(subs)),
            Expr::Ln(a) => ln(a.substitute(subs)),
            Expr::Log2(a) => log2(a.substitute(subs)),
            Expr::Log10(a) => ln(a.substitute(subs)) / ln(constant(10.0)),
            Expr::Sqrt(a) => sqrt(a.substitute(subs)),
            Expr::Abs(a) => abs(a.substitute(subs)),
            Expr::Func { name, params, body, args } => {
                let new_args = args.iter().map(|a| a.substitute(subs)).collect();
                E::new(Expr::Func { name: name.clone(), params: params.clone(), body: body.clone(), args: new_args })
            }
        }
    }
}

impl std::ops::Deref for E {
    type Target = Expr;
    fn deref(&self) -> &Expr {
        &self.0
    }
}

impl AsRef<Expr> for E {
    fn as_ref(&self) -> &Expr {
        &self.0
    }
}

/// Expression AST node.
///
/// Normally constructed via [`symbol`], [`constant`], and the free-standing
/// math functions (e.g. [`sin`], [`cos`], [`pow`]) rather than directly.
#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    /// Named symbolic variable.
    Sym(String),
    /// Numeric constant.
    Const(f64),
    /// Unary negation.
    Neg(E),
    /// Addition.
    Add(E, E),
    /// Subtraction.
    Sub(E, E),
    /// Multiplication.
    Mul(E, E),
    /// Division.
    Div(E, E),
    /// Exponentiation (base^exponent).
    Pow(E, E),
    /// Sine.
    Sin(E),
    /// Cosine.
    Cos(E),
    /// Tangent.
    Tan(E),
    /// Arcsine.
    Asin(E),
    /// Arccosine.
    Acos(E),
    /// Arctangent.
    Atan(E),
    /// Two-argument arctangent (atan2(y, x)).
    Atan2(E, E),
    /// Hyperbolic sine.
    Sinh(E),
    /// Hyperbolic cosine.
    Cosh(E),
    /// Hyperbolic tangent.
    Tanh(E),
    /// Exponential (e^x).
    Exp(E),
    /// Natural logarithm.
    Ln(E),
    /// Base-2 logarithm.
    Log2(E),
    /// Base-10 logarithm.
    Log10(E),
    /// Square root.
    Sqrt(E),
    /// Absolute value.
    Abs(E),
    /// User-defined function application.
    Func {
        /// Function name (for display).
        name: String,
        /// Formal parameter names.
        params: Vec<String>,
        /// Body expression in terms of params.
        body: E,
        /// Actual argument expressions.
        args: Vec<E>,
    },
}

impl Eq for Expr {}

impl Hash for Expr {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            Expr::Sym(s) => s.hash(state),
            Expr::Const(v) => v.to_bits().hash(state),
            Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
            | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
            | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
            | Expr::Exp(a) | Expr::Ln(a) | Expr::Log2(a) | Expr::Log10(a)
            | Expr::Sqrt(a) | Expr::Abs(a) => a.hash(state),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
            | Expr::Div(a, b) | Expr::Pow(a, b) | Expr::Atan2(a, b) => {
                a.hash(state);
                b.hash(state);
            }
            Expr::Func { name, params, body, args } => {
                name.hash(state);
                params.hash(state);
                body.hash(state);
                args.hash(state);
            }
        }
    }
}

impl Hash for E {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

// --- Constructors ---

/// Create a named symbolic variable.
pub fn symbol(name: &str) -> E {
    E::new(Expr::Sym(name.to_string()))
}

/// Create a numeric constant.
pub fn constant(val: f64) -> E {
    E::new(Expr::Const(val))
}

/// Short alias for [`constant`]. Common in math notation.
pub fn c(val: f64) -> E { constant(val) }

/// Symbolic sine function.
pub fn sin(e: E) -> E { E::new(Expr::Sin(e)) }
/// Symbolic cosine function.
pub fn cos(e: E) -> E { E::new(Expr::Cos(e)) }
/// Symbolic tangent function.
pub fn tan(e: E) -> E { E::new(Expr::Tan(e)) }
/// Symbolic arcsine function.
pub fn asin(e: E) -> E { E::new(Expr::Asin(e)) }
/// Symbolic arccosine function.
pub fn acos(e: E) -> E { E::new(Expr::Acos(e)) }
/// Symbolic arctangent function.
pub fn atan(e: E) -> E { E::new(Expr::Atan(e)) }
/// Symbolic two-argument arctangent: atan2(y, x).
pub fn atan2(y: E, x: E) -> E { E::new(Expr::Atan2(y, x)) }
/// Symbolic hyperbolic sine function.
pub fn sinh(e: E) -> E { E::new(Expr::Sinh(e)) }
/// Symbolic hyperbolic cosine function.
pub fn cosh(e: E) -> E { E::new(Expr::Cosh(e)) }
/// Symbolic hyperbolic tangent function.
pub fn tanh(e: E) -> E { E::new(Expr::Tanh(e)) }
/// Symbolic exponential function (e^x).
pub fn exp(e: E) -> E { E::new(Expr::Exp(e)) }
/// Symbolic natural logarithm.
pub fn ln(e: E) -> E { E::new(Expr::Ln(e)) }
/// Symbolic base-2 logarithm.
pub fn log2(e: E) -> E { E::new(Expr::Log2(e)) }
/// Symbolic base-10 logarithm.
pub fn log10(e: E) -> E { E::new(Expr::Log10(e)) }
/// Symbolic square root.
pub fn sqrt(e: E) -> E { E::new(Expr::Sqrt(e)) }
/// Symbolic absolute value.
pub fn abs(e: E) -> E { E::new(Expr::Abs(e)) }
/// Symbolic power function. Auto-simplifies (e.g. x^0 = 1, x^1 = x).
pub fn pow(base: E, exponent: E) -> E { E::new(Expr::Pow(base, exponent)).simplify() }

// --- Operator overloads for E (auto-simplify like SymPy) ---

impl std::ops::Add for E {
    type Output = E;
    fn add(self, rhs: E) -> E {
        E::new(Expr::Add(self, rhs)).simplify()
    }
}

impl std::ops::Sub for E {
    type Output = E;
    fn sub(self, rhs: E) -> E {
        E::new(Expr::Sub(self, rhs)).simplify()
    }
}

impl std::ops::Mul for E {
    type Output = E;
    fn mul(self, rhs: E) -> E {
        E::new(Expr::Mul(self, rhs)).simplify()
    }
}

impl std::ops::Div for E {
    type Output = E;
    fn div(self, rhs: E) -> E {
        E::new(Expr::Div(self, rhs)).simplify()
    }
}

impl std::ops::Neg for E {
    type Output = E;
    fn neg(self) -> E {
        E::new(Expr::Neg(self)).simplify()
    }
}

// --- Mixed ops: E with f64 (auto-simplify) ---

impl std::ops::Add<f64> for E {
    type Output = E;
    fn add(self, rhs: f64) -> E { E::new(Expr::Add(self, constant(rhs))).simplify() }
}

impl std::ops::Add<E> for f64 {
    type Output = E;
    fn add(self, rhs: E) -> E { E::new(Expr::Add(constant(self), rhs)).simplify() }
}

impl std::ops::Sub<f64> for E {
    type Output = E;
    fn sub(self, rhs: f64) -> E { E::new(Expr::Sub(self, constant(rhs))).simplify() }
}

impl std::ops::Sub<E> for f64 {
    type Output = E;
    fn sub(self, rhs: E) -> E { E::new(Expr::Sub(constant(self), rhs)).simplify() }
}

impl std::ops::Mul<f64> for E {
    type Output = E;
    fn mul(self, rhs: f64) -> E { E::new(Expr::Mul(self, constant(rhs))).simplify() }
}

impl std::ops::Mul<E> for f64 {
    type Output = E;
    fn mul(self, rhs: E) -> E { E::new(Expr::Mul(constant(self), rhs)).simplify() }
}

impl std::ops::Div<f64> for E {
    type Output = E;
    fn div(self, rhs: f64) -> E { E::new(Expr::Div(self, constant(rhs))).simplify() }
}

impl std::ops::Div<E> for f64 {
    type Output = E;
    fn div(self, rhs: E) -> E { E::new(Expr::Div(constant(self), rhs)).simplify() }
}

// --- Custom function support ---

/// Expand a Func node by substituting params -> args in the body.
pub(crate) fn expand_func(params: &[String], body: &E, args: &[E]) -> E {
    let mut expanded = body.clone();
    for (p, a) in params.iter().zip(args.iter()) {
        expanded = expanded.subs(p, a);
    }
    expanded
}

/// Create a unary custom function. Returns a closure usable as `f(expr)`.
pub fn custom_func1(name: &str, param: &str, body: E) -> impl Fn(E) -> E + Clone {
    let name = name.to_string();
    let param = param.to_string();
    move |arg: E| {
        E::new(Expr::Func {
            name: name.clone(),
            params: vec![param.clone()],
            body: body.clone(),
            args: vec![arg],
        })
    }
}

/// Create a binary custom function. Returns a closure usable as `f(a, b)`.
pub fn custom_func2(name: &str, params: [&str; 2], body: E) -> impl Fn(E, E) -> E + Clone {
    let name = name.to_string();
    let params = [params[0].to_string(), params[1].to_string()];
    move |a: E, b: E| {
        E::new(Expr::Func {
            name: name.clone(),
            params: vec![params[0].clone(), params[1].clone()],
            body: body.clone(),
            args: vec![a, b],
        })
    }
}

/// Create an n-ary custom function. Returns a closure usable as `f(vec![...])`.
pub fn custom_func(name: &str, params: &[&str], body: E) -> impl Fn(Vec<E>) -> E + Clone {
    let name = name.to_string();
    let params: Vec<String> = params.iter().map(|s| s.to_string()).collect();
    move |args: Vec<E>| {
        assert_eq!(args.len(), params.len(),
            "custom function '{}' expects {} args, got {}", name, params.len(), args.len());
        E::new(Expr::Func {
            name: name.clone(),
            params: params.clone(),
            body: body.clone(),
            args,
        })
    }
}

// Re-export linalg types
pub use linalg::{SymVec, SymMat, jacobian};
pub use diff::DiffVar;
pub use parse::{parse, ParseError};
pub use geo::{vect2sym, vect3sym, matrix2sym, matrix3sym, quaternsym};
pub use cse::cse;
pub use arael_sym_macros::sym;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn custom_func_identity_display() {
        sym! {
            let t = symbol("t");
            let identity = custom_func1("identity", "t", t);
            let x = symbol("x");
            assert_eq!(format!("{}", identity(x)), "identity(x)");
        }
    }

    #[test]
    fn custom_func_identity_diff() {
        sym! {
            let t = symbol("t");
            let identity = custom_func1("identity", "t", t);
            let x = symbol("x");
            let f = identity(x);
            assert_eq!(format!("{}", f.diff("x")), "1");
        }
    }

    #[test]
    fn custom_func_identity_chain_rule() {
        sym! {
            let t = symbol("t");
            let identity = custom_func1("identity", "t", t);
            let x = symbol("x");
            let f = identity(x * x);
            assert_eq!(format!("{}", f.diff("x")), "2 * x");
        }
    }

    #[test]
    fn custom_func_identity_eval() {
        sym! {
            let t = symbol("t");
            let identity = custom_func1("identity", "t", t);
            let x = symbol("x");
            let f = identity(x);
            let vars = HashMap::from([("x", 5.0)]);
            assert_eq!(f.eval(&vars).unwrap(), 5.0);
        }
    }

    #[test]
    fn custom_func_square() {
        sym! {
            let t = symbol("t");
            let square = custom_func1("square", "t", t * t);
            let x = symbol("x");
            let f = square(x + 1.0);
            assert_eq!(format!("{}", f), "square(x + 1)");
            assert_eq!(format!("{}", f.diff("x")), "2 * (x + 1)");
        }
    }

    #[test]
    fn custom_func_square_eval() {
        sym! {
            let t = symbol("t");
            let square = custom_func1("square", "t", t * t);
            let x = symbol("x");
            let f = square(x);
            let vars = HashMap::from([("x", 4.0)]);
            assert_eq!(f.eval(&vars).unwrap(), 16.0);
        }
    }

    #[test]
    fn custom_func_binary() {
        sym! {
            let a = symbol("a");
            let b = symbol("b");
            let f = custom_func2("prod", ["a", "b"], a * b);
            let x = symbol("x");
            let y = symbol("y");
            let result = f(x, y);
            assert_eq!(format!("{}", result), "prod(x, y)");
            assert_eq!(format!("{}", result.diff("x")), "y");
            assert_eq!(format!("{}", result.diff("y")), "x");
        }
    }

    #[test]
    fn custom_func_nested() {
        sym! {
            let t = symbol("t");
            let identity = custom_func1("identity", "t", t);
            let square = custom_func1("square", "t", t * t);
            let x = symbol("x");
            let f = identity(square(x));
            assert_eq!(format!("{}", f), "identity(square(x))");
            assert_eq!(format!("{}", f.diff("x")), "2 * x");
        }
    }

    #[test]
    fn custom_func_my_sin() {
        sym! {
            let t = symbol("t");
            let my_sin = custom_func1("my_sin", "t", sin(t));
            let x = symbol("x");
            let f = my_sin(x);
            assert_eq!(format!("{}", f), "my_sin(x)");
            assert_eq!(format!("{}", f.diff("x")), "cos(x)");
        }
    }

    #[test]
    fn custom_func_my_sin_chain_rule() {
        sym! {
            let t = symbol("t");
            let my_sin = custom_func1("my_sin", "t", sin(t));
            let x = symbol("x");
            let f = my_sin(x * x);
            assert_eq!(format!("{}", f.diff("x")), "2 * x * cos(x^2)");
        }
    }

    #[test]
    fn custom_func_to_rust() {
        sym! {
            let t = symbol("t");
            let identity = custom_func1("identity", "t", t);
            let x = symbol("x");
            let f = identity(x);
            assert_eq!(f.to_rust("f64"), "x");
        }
    }

    #[test]
    fn custom_func_latex() {
        sym! {
            let t = symbol("t");
            let identity = custom_func1("identity", "t", t);
            let x = symbol("x");
            let f = identity(x);
            assert_eq!(f.to_latex(), "\\operatorname{identity}\\left(x\\right)");
        }
    }

    #[test]
    fn custom_func_free_vars() {
        sym! {
            let t = symbol("t");
            let identity = custom_func1("identity", "t", t);
            let x = symbol("x");
            let f = identity(x + symbol("y"));
            let vars = f.free_vars();
            assert!(vars.contains("x"));
            assert!(vars.contains("y"));
            assert!(!vars.contains("t"));
        }
    }

    #[test]
    fn custom_func_subs() {
        sym! {
            let t = symbol("t");
            let identity = custom_func1("identity", "t", t);
            let x = symbol("x");
            let f = identity(x);
            let g = f.subs("x", &constant(3.0));
            assert_eq!(format!("{}", g), "identity(3)");
        }
    }

    #[test]
    fn custom_func_simplify_constants() {
        sym! {
            let t = symbol("t");
            let square = custom_func1("square", "t", t * t);
            let f = square(constant(3.0));
            let s = f.simplify();
            assert_eq!(format!("{}", s), "9");
        }
    }

    #[test]
    fn custom_func_nary() {
        sym! {
            let a = symbol("a");
            let b = symbol("b");
            let c_sym = symbol("c");
            let f = custom_func("triple_sum", &["a", "b", "c"], a + b + c_sym);
            let x = symbol("x");
            let y = symbol("y");
            let z = symbol("z");
            let result = f(vec![x, y, z]);
            assert_eq!(format!("{}", result), "triple_sum(x, y, z)");
            assert_eq!(format!("{}", result.diff("x")), "1");
        }
    }

    #[test]
    fn custom_func_expand() {
        sym! {
            let t = symbol("t");
            let square = custom_func1("square", "t", t * t);
            let x = symbol("x");
            let f = square(x + 1.0);
            let expanded = f.expand();
            assert_eq!(format!("{}", expanded), "x^2 + 2 * x + 1");
        }
    }
}

