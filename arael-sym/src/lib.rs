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
//!     f.eval(&vars)
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
//! assert!((f.eval(&vars) - 1.0).abs() < 1e-10);
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

// Re-export linalg types
pub use linalg::{SymVec, SymMat, jacobian};
pub use diff::DiffVar;
pub use parse::{parse, ParseError};
pub use geo::{vect2sym, vect3sym, matrix2sym, matrix3sym, quaternsym};
pub use cse::cse;
pub use arael_sym_macros::sym;

