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
//!
//! ## Named constants
//!
//! Named constants survive simplification (unlike numeric `Const` which may
//! be folded away). Built-in: [`pi`], [`epsilon`], [`euler`]. Custom
//! constants via [`named_const`]. The [`sym!`] macro accepts `pi` and
//! `epsilon` as bare identifiers.
//!
//! ```
//! use arael_sym::*;
//! sym! {
//!     let x = symbol("x");
//!     let f = x * x + epsilon;           // bare identifier, no parens needed
//!     assert_eq!(format!("{}", f), "x^2 + epsilon");
//!     assert_eq!(format!("{}", sin(pi).simplify()), "0");
//!     assert_eq!(format!("{}", cos(pi).simplify()), "-1");
//!     assert_eq!(format!("{}", ln(euler()).simplify()), "1");
//! };
//! ```
//!
//! ## Identity and evaluation order
//!
//! The simplifier flattens and reorders additive terms, which can cause
//! floating-point cancellation in generated code. For example,
//! `1 - x^2 + epsilon^2` might be reordered to `-x^2 + epsilon^2 + 1`,
//! and at `x=1` the tiny `epsilon^2` gets absorbed into `-1 + 1` before
//! it can contribute.
//!
//! The [`identity`] function acts as a barrier: `identity(expr)` evaluates
//! to `expr` and differentiates as `1`, but the simplifier cannot reorder
//! terms across it. Codegen wraps the body in parentheses to preserve
//! evaluation order in the generated Rust code.
//!
//! ```
//! use arael_sym::*;
//! sym! {
//!     let x = symbol("x");
//!     // Without identity: terms may reorder, epsilon^2 lost at x=1
//!     // With identity: (1 - x^2) evaluates first, then epsilon^2 is added
//!     let safe = identity(c(1.0) - x * x) + epsilon * epsilon;
//!     let code = safe.to_rust("f64");
//!     // Body is wrapped in parens in generated code
//!     assert!(code.contains("(-x.powf(2.0_f64) + 1.0_f64)"));
//! };
//! ```
//!
//! This pattern is used internally by [`safe_asin`] and [`safe_acos`] to
//! keep `epsilon^2` from being lost to floating-point cancellation in the
//! derivative `1/sqrt(1 - x^2 + epsilon^2)`.
//!
//! ## Custom functions
//!
//! Define reusable symbolic functions with automatic differentiation.
//! The factory functions return closures that can be called like regular
//! functions.
//!
//! ```
//! use arael_sym::*;
//! sym! {
//!     let t = symbol("t");
//!     let square = simple_func1("square", |t| t * t);
//!     let x = symbol("x");
//!     let f = square(x + 1.0);
//!     assert_eq!(format!("{}", f), "square(x + 1)");
//!     assert_eq!(format!("{}", f.diff("x")), "2 * (x + 1)");
//!     // Codegen inlines the expanded body:
//!     assert_eq!(f.to_rust("f64"), "(x + 1.0_f64).powf(2.0_f64)");
//! };
//! ```
//!
//! ## Extern functions
//!
//! When a function's runtime behavior differs from its derivative (e.g.
//! angle normalization), use extern functions. They generate a function
//! call in codegen while differentiating through a separate symbolic body.
//!
//! ```
//! use arael_sym::*;
//! fn my_angle_diff(args: &[f64]) -> f64 {
//!     let d = args[0] - args[1];
//!     d - (2.0 * std::f64::consts::PI)
//!       * (d / (2.0 * std::f64::consts::PI) + 0.5).floor()
//! }
//! sym! {
//!     // codegen emits my_mod::angle_diff(a, b)
//!     // differentiation uses gradient of (a - b)
//!     // eval uses my_angle_diff
//!     let angle_diff = extern_func2("angle_diff", "my_mod::angle_diff",
//!         grad2(|a, b| a - b), my_angle_diff);
//!     let x = symbol("x");
//!     let y = symbol("y");
//!     let f = angle_diff(x * x, y);
//!     assert_eq!(format!("{}", f.diff("x")), "2 * x");
//!     assert_eq!(f.to_rust("f64"), "my_mod::angle_diff(x.powf(2.0_f64), y)");
//!     // eval uses the native eval_fn:
//!     let vars = std::collections::HashMap::from([("x", 0.0), ("y", 6.283185307179586)]);
//!     assert!(f.eval(&vars).unwrap().abs() < 1e-10); // 0 - 2pi wraps to 0
//! };
//! ```
//!
//! Built-in [`rad_diff`] and [`rad_sum`] are extern functions with
//! rollover-safe angle normalization to \[-pi, pi\].
//!
//! ## Heaviside and clamp
//!
//! Pragmatic functions for optimization near numerical boundaries.
//! `heaviside` has derivative 0 everywhere (not Dirac delta).
//! `clamp` has pass-through derivative (as if clamping were not there).
//!
//! ```
//! use arael_sym::*;
//! sym! {
//!     // clamp prevents NaN from asin outside [-1, 1]
//!     // Note: derivative still diverges at +/-1. One can prevent it
//!     // by providing custom derivatives with simple_func1_derivs as
//!     // is done in the built-in safe_asin().
//!     let my_asin = simple_func1("my_asin",
//!         |t| asin(clamp(t, c(-1.0), c(1.0))));
//!     let x = symbol("x");
//!     let f = my_asin(x);
//!     let vars = std::collections::HashMap::from([("x", 1.5)]);
//!     // Clamped to asin(1.0) = pi/2, no NaN
//!     let val = f.eval(&vars).unwrap();
//!     assert!((val - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
//! };
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
            Expr::Const(_) | Expr::NamedConst { .. } => {}
            Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
            | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
            | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
            | Expr::Exp(a) | Expr::Ln(a) | Expr::Log2(a) | Expr::Log10(a)
            | Expr::Sqrt(a) | Expr::Abs(a)
            | Expr::Heaviside(a) => { a.collect_symbols(out); }
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
            | Expr::Div(a, b) | Expr::Pow(a, b) | Expr::Atan2(a, b) => {
                a.collect_symbols(out);
                b.collect_symbols(out);
            }
            Expr::Clamp(a, b, c) => {
                a.collect_symbols(out);
                b.collect_symbols(out);
                c.collect_symbols(out);
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
            Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => self.clone(),
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
            Expr::Heaviside(a) => heaviside(a.substitute(subs)),
            Expr::Clamp(a, lo, hi) => clamp(a.substitute(subs), lo.substitute(subs), hi.substitute(subs)),
            Expr::Func { name, params, kind, args } => {
                let new_args = args.iter().map(|a| a.substitute(subs)).collect();
                E::new(Expr::Func { name: name.clone(), params: params.clone(), kind: kind.clone(), args: new_args })
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
    /// Heaviside step function: 0 if x < 0, 1 if x >= 0. Derivative is 0.
    Heaviside(E),
    /// Clamp value to [lo, hi]. Derivative passes through (= d(val)/dvar).
    Clamp(E, E, E),
    /// Named constant (pi, epsilon, e, or user-defined).
    /// Survives simplification (unlike Const which may be folded away).
    NamedConst {
        name: String,
        value: f64,
        rust_f32: String,
        rust_f64: String,
        latex: String,
    },
    /// User-defined function application.
    Func {
        /// Function name (for display).
        name: String,
        /// Formal parameter names.
        params: Vec<String>,
        /// Function behavior (differentiation, codegen, eval).
        kind: FuncKind,
        /// Actual argument expressions.
        args: Vec<E>,
    },
}

/// Describes what kind of function behavior to use for differentiation,
/// evaluation, and code generation.
#[derive(Debug, Clone, PartialEq)]
#[allow(unpredictable_function_pointer_comparisons)]
pub enum FuncKind {
    /// Body auto-differentiated. Body used for eval and codegen (inlined).
    Symbolic { body: E },
    /// Explicit per-argument derivatives. Body used for eval and codegen (inlined).
    SymbolicDerivs { body: E, derivs: Vec<E> },
    /// Explicit per-argument derivatives. Codegen emits `call_path(args...)`.
    /// `eval_fn` used for eval (required).
    Extern { derivs: Vec<E>, eval_fn: fn(&[f64]) -> f64, call_path: String },
}

impl FuncKind {
    /// Body for auto-differentiation (Symbolic only).
    pub fn auto_diff_body(&self) -> Option<&E> {
        match self {
            FuncKind::Symbolic { body } => Some(body),
            _ => None,
        }
    }

    /// Explicit per-argument derivatives (SymbolicDerivs and Extern).
    pub fn derivs(&self) -> Option<&[E]> {
        match self {
            FuncKind::SymbolicDerivs { derivs, .. } | FuncKind::Extern { derivs, .. } => Some(derivs),
            FuncKind::Symbolic { .. } => None,
        }
    }

    /// Body for symbolic eval and codegen inlining (Symbolic variants).
    pub fn body(&self) -> Option<&E> {
        match self {
            FuncKind::Symbolic { body } | FuncKind::SymbolicDerivs { body, .. } => Some(body),
            FuncKind::Extern { .. } => None,
        }
    }

    /// Native eval function (Extern only).
    pub fn eval_fn(&self) -> Option<fn(&[f64]) -> f64> {
        match self {
            FuncKind::Extern { eval_fn, .. } => Some(*eval_fn),
            _ => None,
        }
    }
}

impl Hash for FuncKind {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            FuncKind::Symbolic { body } => body.hash(state),
            FuncKind::SymbolicDerivs { body, derivs } => {
                body.hash(state);
                derivs.hash(state);
            }
            FuncKind::Extern { derivs, eval_fn, call_path } => {
                derivs.hash(state);
                (*eval_fn as usize).hash(state);
                call_path.hash(state);
            }
        }
    }
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
            | Expr::Sqrt(a) | Expr::Abs(a)
            | Expr::Heaviside(a) => a.hash(state),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
            | Expr::Div(a, b) | Expr::Pow(a, b) | Expr::Atan2(a, b) => {
                a.hash(state);
                b.hash(state);
            }
            Expr::Clamp(a, b, c) => {
                a.hash(state);
                b.hash(state);
                c.hash(state);
            }
            Expr::NamedConst { name, value, .. } => {
                name.hash(state);
                value.to_bits().hash(state);
            }
            Expr::Func { name, params, kind, args } => {
                name.hash(state);
                params.hash(state);
                kind.hash(state);
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

/// Create a named constant with explicit display, eval, codegen, and LaTeX representations.
pub fn named_const(name: &str, value: f64, rust_f32: &str, rust_f64: &str, latex: &str) -> E {
    E::new(Expr::NamedConst {
        name: name.to_string(), value,
        rust_f32: rust_f32.to_string(), rust_f64: rust_f64.to_string(),
        latex: latex.to_string(),
    })
}

/// $\pi = 3.14159\ldots$
pub fn pi() -> E {
    named_const("pi", std::f64::consts::PI,
        "std::f32::consts::PI", "std::f64::consts::PI", "\\pi")
}

/// Machine epsilon $\epsilon$ (`f64::EPSILON` $\approx 2.22 \times 10^{-16}$).
pub fn epsilon() -> E {
    named_const("epsilon", f64::EPSILON,
        "f32::EPSILON", "f64::EPSILON", "\\epsilon")
}

/// Euler's number $e = 2.71828\ldots$
pub fn euler() -> E {
    named_const("e", std::f64::consts::E,
        "std::f32::consts::E", "std::f64::consts::E", "e")
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
/// Symbolic Heaviside step function: 0 if x < 0, 1 if x >= 0.
pub fn heaviside(e: E) -> E { E::new(Expr::Heaviside(e)) }
/// Symbolic clamp: clamp value to [lo, hi]. Derivative passes through.
pub fn clamp(val: E, lo: E, hi: E) -> E { E::new(Expr::Clamp(val, lo, hi)) }
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
/// Codegen inlines the expanded body.
///
/// # Example
/// ```
/// use arael_sym::*;
/// sym! {
///     let square = simple_func1("square", |t| t * t);
///     let x = symbol("x");
///     assert_eq!(format!("{}", square(x + 1.0)), "square(x + 1)");
///     assert_eq!(format!("{}", square(x).diff("x")), "2 * x");
/// };
/// ```
pub fn simple_func1(name: &str, body: impl Fn(E) -> E) -> impl Fn(E) -> E + Clone {
    let name = name.to_string();
    let body = body(symbol("__p0"));
    move |arg: E| {
        E::new(Expr::Func {
            name: name.clone(),
            params: vec!["__p0".to_string()],
            kind: FuncKind::Symbolic { body: body.clone() },
            args: vec![arg],
        })
    }
}

/// Create a binary custom function. Returns a closure usable as `f(a, b)`.
/// Codegen inlines the expanded body.
pub fn simple_func2(name: &str, body: impl Fn(E, E) -> E) -> impl Fn(E, E) -> E + Clone {
    let name = name.to_string();
    let body = body(symbol("__p0"), symbol("__p1"));
    move |a: E, b: E| {
        E::new(Expr::Func {
            name: name.clone(),
            params: vec!["__p0".to_string(), "__p1".to_string()],
            kind: FuncKind::Symbolic { body: body.clone() },
            args: vec![a, b],
        })
    }
}

/// Create an n-ary custom function. Returns a closure usable as `f(vec![...])`.
/// Codegen inlines the expanded body.
pub fn simple_func(name: &str, arity: usize, body: impl Fn(Vec<E>) -> E) -> impl Fn(Vec<E>) -> E + Clone {
    let name = name.to_string();
    let params: Vec<String> = (0..arity).map(|i| format!("__p{}", i)).collect();
    let syms: Vec<E> = params.iter().map(|p| symbol(p)).collect();
    let body = body(syms);
    move |args: Vec<E>| {
        assert_eq!(args.len(), arity,
            "custom function '{}' expects {} args, got {}", name, arity, args.len());
        E::new(Expr::Func {
            name: name.clone(),
            params: params.clone(),
            kind: FuncKind::Symbolic { body: body.clone() },
            args,
        })
    }
}

/// Create a unary function with explicit derivatives. Body used for eval
/// and codegen (inlined).
pub fn simple_func1_derivs(
    name: &str, body: impl Fn(E) -> E, derivs: impl Fn(E) -> [E; 1],
) -> impl Fn(E) -> E + Clone {
    let name = name.to_string();
    let p0 = symbol("__p0");
    let body = body(p0.clone());
    let d = derivs(p0);
    move |a: E| {
        E::new(Expr::Func {
            name: name.clone(),
            params: vec!["__p0".to_string()],
            kind: FuncKind::SymbolicDerivs { body: body.clone(), derivs: vec![d[0].clone()] },
            args: vec![a],
        })
    }
}

/// Create a binary function with explicit derivatives. Body used for eval
/// and codegen (inlined).
///
/// # Example
/// ```
/// use arael_sym::*;
/// sym! {
///     // Or use the built-in safe_atan2():
///     let a = symbol("a");
///     let f = safe_atan2(sin(a), cos(a));
///     assert_eq!(format!("{}", f), "safe_atan2(sin(a), cos(a))");
/// };
/// ```
pub fn simple_func2_derivs(
    name: &str, body: impl Fn(E, E) -> E, derivs: impl Fn(E, E) -> [E; 2],
) -> impl Fn(E, E) -> E + Clone {
    let name = name.to_string();
    let p0 = symbol("__p0");
    let p1 = symbol("__p1");
    let body = body(p0.clone(), p1.clone());
    let d = derivs(p0, p1);
    move |a: E, b: E| {
        E::new(Expr::Func {
            name: name.clone(),
            params: vec!["__p0".to_string(), "__p1".to_string()],
            kind: FuncKind::SymbolicDerivs { body: body.clone(), derivs: vec![d[0].clone(), d[1].clone()] },
            args: vec![a, b],
        })
    }
}

/// Create an n-ary function with explicit derivatives. Body used for eval
/// and codegen (inlined).
pub fn simple_func_derivs(
    name: &str, arity: usize, body: impl Fn(Vec<E>) -> E, derivs: impl Fn(Vec<E>) -> Vec<E>,
) -> impl Fn(Vec<E>) -> E + Clone {
    let name = name.to_string();
    let params: Vec<String> = (0..arity).map(|i| format!("__p{}", i)).collect();
    let syms: Vec<E> = params.iter().map(|p| symbol(p)).collect();
    let body = body(syms.clone());
    let d = derivs(syms);
    assert_eq!(d.len(), arity, "derivs must return {} elements", arity);
    move |args: Vec<E>| {
        assert_eq!(args.len(), arity,
            "function '{}' expects {} args, got {}", name, arity, args.len());
        E::new(Expr::Func {
            name: name.clone(),
            params: params.clone(),
            kind: FuncKind::SymbolicDerivs { body: body.clone(), derivs: d.clone() },
            args,
        })
    }
}

/// Create a unary extern function: codegen emits `call_path(arg)`,
/// explicit derivatives for differentiation, `eval_fn` for eval.
pub fn extern_func1(
    name: &str, call_path: &str,
    derivs: impl Fn(E) -> [E; 1],
    eval_fn: fn(&[f64]) -> f64,
) -> impl Fn(E) -> E + Clone {
    let name = name.to_string();
    let call_path = call_path.to_string();
    let d = derivs(symbol("__p0"));
    move |a: E| {
        E::new(Expr::Func {
            name: name.clone(),
            params: vec!["__p0".to_string()],
            kind: FuncKind::Extern {
                derivs: vec![d[0].clone()],
                eval_fn,
                call_path: call_path.clone(),
            },
            args: vec![a],
        })
    }
}

/// Create a binary extern function: codegen emits `call_path(a, b)`,
/// explicit derivatives for differentiation, `eval_fn` for eval.
///
/// Use [`grad2`] to auto-compute derivatives from a body expression.
///
/// # Example
/// ```
/// use arael_sym::*;
/// sym! {
///     let f = extern_func2("rad_diff", "arael::utils::rad_diff",
///         grad2(|a, b| a - b),
///         |args: &[f64]| args[0] - args[1]);
///     let x = symbol("x");
///     let y = symbol("y");
///     assert_eq!(format!("{}", f(x, y).diff("x")), "1");
///     assert_eq!(f(x, y).to_rust("f64"), "arael::utils::rad_diff(x, y)");
/// };
/// ```
pub fn extern_func2(
    name: &str, call_path: &str,
    derivs: impl Fn(E, E) -> [E; 2],
    eval_fn: fn(&[f64]) -> f64,
) -> impl Fn(E, E) -> E + Clone {
    let name = name.to_string();
    let call_path = call_path.to_string();
    let d = derivs(symbol("__p0"), symbol("__p1"));
    move |a: E, b: E| {
        E::new(Expr::Func {
            name: name.clone(),
            params: vec!["__p0".to_string(), "__p1".to_string()],
            kind: FuncKind::Extern {
                derivs: vec![d[0].clone(), d[1].clone()],
                eval_fn,
                call_path: call_path.clone(),
            },
            args: vec![a, b],
        })
    }
}

/// Create an n-ary extern function: codegen emits `call_path(args...)`,
/// explicit derivatives for differentiation, `eval_fn` for eval.
pub fn extern_func(
    name: &str, arity: usize, call_path: &str,
    derivs: impl Fn(Vec<E>) -> Vec<E>,
    eval_fn: fn(&[f64]) -> f64,
) -> impl Fn(Vec<E>) -> E + Clone {
    let name = name.to_string();
    let call_path = call_path.to_string();
    let params: Vec<String> = (0..arity).map(|i| format!("__p{}", i)).collect();
    let syms: Vec<E> = params.iter().map(|p| symbol(p)).collect();
    let d = derivs(syms);
    assert_eq!(d.len(), arity, "derivs must return {} elements", arity);
    move |args: Vec<E>| {
        assert_eq!(args.len(), arity,
            "extern function '{}' expects {} args, got {}", name, arity, args.len());
        E::new(Expr::Func {
            name: name.clone(),
            params: params.clone(),
            kind: FuncKind::Extern {
                derivs: d.clone(),
                eval_fn,
                call_path: call_path.clone(),
            },
            args,
        })
    }
}

/// Compute the gradient of a unary function symbolically.
/// Returns a closure suitable for `simple_func1_derivs` or `extern_func1`.
pub fn grad1(body: impl Fn(E) -> E) -> impl Fn(E) -> [E; 1] + Clone {
    let p = symbol("__g0");
    let d = body(p).diff("__g0");
    move |a: E| { [d.subs("__g0", &a)] }
}

/// Compute the gradient of a binary function symbolically.
/// Returns a closure suitable for `simple_func2_derivs` or `extern_func2`.
pub fn grad2(body: impl Fn(E, E) -> E) -> impl Fn(E, E) -> [E; 2] + Clone {
    let p0 = symbol("__g0");
    let p1 = symbol("__g1");
    let expr = body(p0, p1);
    let d0 = expr.diff("__g0");
    let d1 = expr.diff("__g1");
    move |a: E, b: E| {
        [d0.subs("__g0", &a).subs("__g1", &b),
         d1.subs("__g0", &a).subs("__g1", &b)]
    }
}

/// Normalize radians to [-pi, pi].
fn rad2rad(v: f64) -> f64 {
    use std::f64::consts::PI;
    if v < -PI || v > PI {
        v - (2.0 * PI) * (v / (2.0 * PI) + 0.5).floor()
    } else {
        v
    }
}

/// Rollover-safe radian difference: $(a - b)$ normalized to $[-\pi, \pi]$.
///
/// Differentiation treats it as $a - b$: $\frac{\partial}{\partial a} = 1$, $\frac{\partial}{\partial b} = -1$.
pub fn rad_diff(a: E, b: E) -> E {
    extern_func2("rad_diff", "arael::utils::rad_diff",
        grad2(|a, b| a - b),
        |args: &[f64]| rad2rad(args[0] - args[1]))(a, b)
}

/// Rollover-safe radian sum: $(a + b)$ normalized to $[-\pi, \pi]$.
///
/// Differentiation treats it as $a + b$: $\frac{\partial}{\partial a} = 1$, $\frac{\partial}{\partial b} = 1$.
pub fn rad_sum(a: E, b: E) -> E {
    extern_func2("rad_sum", "arael::utils::rad_sum",
        grad2(|a, b| a + b),
        |args: &[f64]| rad2rad(args[0] + args[1]))(a, b)
}

/// Identity function: $\text{identity}(x) = x$, $\frac{d}{dx} = 1$.
///
/// The simplifier does not look inside Func nodes, so `identity(a - b)`
/// prevents term reordering across the boundary. Codegen wraps the inlined
/// body in parentheses to preserve evaluation order.
///
/// Use this to guard expressions against floating-point cancellation.
/// For example, $\text{identity}(1 - x^2) + \epsilon^2$ ensures
/// the subtraction evaluates first, then $\epsilon^2$ is added to the result.
pub fn identity(x: E) -> E {
    simple_func1("identity", |t| t)(x)
}

/// Safe atan2 with non-diverging derivatives.
///
/// $$\text{atan2\\_safe}(y, x) = \text{atan2}(y, x)$$
///
/// $$\frac{\partial}{\partial y} = \frac{x}{x^2 + y^2 + \epsilon^2}, \quad
///   \frac{\partial}{\partial x} = \frac{-y}{x^2 + y^2 + \epsilon^2}$$
///
/// The $\epsilon^2$ term prevents division by zero at $(0, 0)$.
pub fn safe_atan2(y: E, x: E) -> E {
    simple_func2_derivs("safe_atan2",
        |y, x| atan2(y, x),
        |y, x| {
            let eps2 = epsilon() * epsilon();
            let d = x.clone()*x.clone() + y.clone()*y.clone() + eps2;
            [x / d.clone(), -y / d]
        })(y, x)
}

/// Safe asin with clamped domain and non-diverging derivative.
///
/// $$\text{asin\\_safe}(x) = \arcsin(\text{clamp}(x, -1, 1))$$
///
/// $$\frac{d}{dx} = \frac{1}{\sqrt{\text{identity}(1 - x^2) + \epsilon^2}}$$
///
/// The [`identity`] guard prevents the simplifier from reordering
/// $1 - x^2$ and $\epsilon^2$, avoiding floating-point cancellation.
pub fn safe_asin(x: E) -> E {
    simple_func1_derivs("safe_asin",
        |x| asin(clamp(x, c(-1.0), c(1.0))),
        |x| [c(1.0) / sqrt(identity(c(1.0) - x.clone()*x) + epsilon()*epsilon())]
    )(x)
}

/// Safe acos with clamped domain and non-diverging derivative.
///
/// $$\text{acos\\_safe}(x) = \arccos(\text{clamp}(x, -1, 1))$$
///
/// $$\frac{d}{dx} = \frac{-1}{\sqrt{\text{identity}(1 - x^2) + \epsilon^2}}$$
pub fn safe_acos(x: E) -> E {
    simple_func1_derivs("safe_acos",
        |x| acos(clamp(x, c(-1.0), c(1.0))),
        |x| [-c(1.0) / sqrt(identity(c(1.0) - x.clone()*x) + epsilon()*epsilon())]
    )(x)
}

/// Safe square root: clamps negative inputs to zero, non-diverging derivative.
///
/// $$\text{safe\_sqrt}(x) = \sqrt{\max(x, 0)}$$
///
/// $$\frac{d}{dx} = \frac{1}{2\sqrt{x + \epsilon^2}}$$
///
/// Negative inputs evaluate as zero. The runtime function asserts if the input
/// is more than noise-level negative. The $\epsilon^2$ term prevents the
/// derivative from diverging at $x = 0$.
pub fn safe_sqrt(x: E) -> E {
    extern_func1("safe_sqrt", "arael::utils::safe_sqrt",
        |x| [c(0.5) / sqrt(identity(x) + epsilon()*epsilon())],
        |args| {
            let v = args[0];
            if v <= 0.0 { 0.0 } else { v.sqrt() }
        }
    )(x)
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
    fn simple_func_identity_display() {
        sym! {
            let identity = simple_func1("identity", |t| t);
            let x = symbol("x");
            assert_eq!(format!("{}", identity(x)), "identity(x)");
        }
    }

    #[test]
    fn simple_func_identity_diff() {
        sym! {
            let identity = simple_func1("identity", |t| t);
            let x = symbol("x");
            let f = identity(x);
            assert_eq!(format!("{}", f.diff("x")), "1");
        }
    }

    #[test]
    fn simple_func_identity_chain_rule() {
        sym! {
            let identity = simple_func1("identity", |t| t);
            let x = symbol("x");
            let f = identity(x * x);
            assert_eq!(format!("{}", f.diff("x")), "2 * x");
        }
    }

    #[test]
    fn simple_func_identity_eval() {
        sym! {
            let identity = simple_func1("identity", |t| t);
            let x = symbol("x");
            let f = identity(x);
            let vars = HashMap::from([("x", 5.0)]);
            assert_eq!(f.eval(&vars).unwrap(), 5.0);
        }
    }

    #[test]
    fn simple_func_square() {
        sym! {
            let square = simple_func1("square", |t| t * t);
            let x = symbol("x");
            let f = square(x + 1.0);
            assert_eq!(format!("{}", f), "square(x + 1)");
            assert_eq!(format!("{}", f.diff("x")), "2 * (x + 1)");
        }
    }

    #[test]
    fn simple_func_square_eval() {
        sym! {
            let square = simple_func1("square", |t| t * t);
            let x = symbol("x");
            let f = square(x);
            let vars = HashMap::from([("x", 4.0)]);
            assert_eq!(f.eval(&vars).unwrap(), 16.0);
        }
    }

    #[test]
    fn simple_func_binary() {
        sym! {
            let f = simple_func2("prod", |a, b| a * b);
            let x = symbol("x");
            let y = symbol("y");
            let result = f(x, y);
            assert_eq!(format!("{}", result), "prod(x, y)");
            assert_eq!(format!("{}", result.diff("x")), "y");
            assert_eq!(format!("{}", result.diff("y")), "x");
        }
    }

    #[test]
    fn simple_func_nested() {
        sym! {
            let identity = simple_func1("identity", |t| t);
            let square = simple_func1("square", |t| t * t);
            let x = symbol("x");
            let f = identity(square(x));
            assert_eq!(format!("{}", f), "identity(square(x))");
            assert_eq!(format!("{}", f.diff("x")), "2 * x");
        }
    }

    #[test]
    fn simple_func_my_sin() {
        sym! {
            let my_sin = simple_func1("my_sin", |t| sin(t));
            let x = symbol("x");
            let f = my_sin(x);
            assert_eq!(format!("{}", f), "my_sin(x)");
            assert_eq!(format!("{}", f.diff("x")), "cos(x)");
        }
    }

    #[test]
    fn simple_func_my_sin_chain_rule() {
        sym! {
            let my_sin = simple_func1("my_sin", |t| sin(t));
            let x = symbol("x");
            let f = my_sin(x * x);
            assert_eq!(format!("{}", f.diff("x")), "2 * x * cos(x^2)");
        }
    }

    #[test]
    fn simple_func_to_rust() {
        sym! {
            let identity = simple_func1("identity", |t| t);
            let x = symbol("x");
            let f = identity(x);
            assert_eq!(f.to_rust("f64"), "x");
        }
    }

    #[test]
    fn simple_func_latex() {
        sym! {
            let identity = simple_func1("identity", |t| t);
            let x = symbol("x");
            let f = identity(x);
            assert_eq!(f.to_latex(), "\\operatorname{identity}\\left(x\\right)");
        }
    }

    #[test]
    fn simple_func_free_vars() {
        sym! {
            let identity = simple_func1("identity", |t| t);
            let x = symbol("x");
            let f = identity(x + symbol("y"));
            let vars = f.free_vars();
            assert!(vars.contains("x"));
            assert!(vars.contains("y"));
            assert!(!vars.contains("t"));
        }
    }

    #[test]
    fn simple_func_subs() {
        sym! {
            let identity = simple_func1("identity", |t| t);
            let x = symbol("x");
            let f = identity(x);
            let g = f.subs("x", &constant(3.0));
            assert_eq!(format!("{}", g), "identity(3)");
        }
    }

    #[test]
    fn simple_func_simplify_constants() {
        sym! {
            let square = simple_func1("square", |t| t * t);
            let f = square(constant(3.0));
            let s = f.simplify();
            assert_eq!(format!("{}", s), "9");
        }
    }

    #[test]
    fn simple_func_nary() {
        sym! {
            let f = simple_func("triple_sum", 3, |v| v[0].clone() + v[1].clone() + v[2].clone());
            let x = symbol("x");
            let y = symbol("y");
            let z = symbol("z");
            let result = f(vec![x, y, z]);
            assert_eq!(format!("{}", result), "triple_sum(x, y, z)");
            assert_eq!(format!("{}", result.diff("x")), "1");
        }
    }

    #[test]
    fn simple_func_expand() {
        sym! {
            let square = simple_func1("square", |t| t * t);
            let x = symbol("x");
            let f = square(x + 1.0);
            let expanded = f.expand();
            assert_eq!(format!("{}", expanded), "x^2 + 2 * x + 1");
        }
    }

    // --- Simple func derivs tests ---

    #[test]
    fn simple_func_derivs_codegen() {
        // codegen should inline the body, not the derivs
        sym! {
            let f = simple_func1_derivs("inv", |t| 1.0 / t, |t| [-1.0 / (t * t)]);
            let x = symbol("x");
            assert_eq!(f(x).to_rust("f64"), "1.0_f64 / x");
        }
    }

    // --- Safe function tests ---

    #[test]
    fn safe_atan2_diff() {
        sym! {
            let a = symbol("a");
            let b = symbol("b");
            let f = safe_atan2(a, b);
            let da = f.diff("a");
            let vars = HashMap::from([("a", 1.0), ("b", 1.0)]);
            let v = da.eval(&vars).unwrap();
            assert!((v - 0.5).abs() < 1e-10, "d/da at (1,1) = {}, expected 0.5", v);
        }
    }

    #[test]
    fn safe_atan2_eval() {
        sym! {
            let a = symbol("a");
            let b = symbol("b");
            let f = safe_atan2(a, b);
            let vars = HashMap::from([("a", 1.0), ("b", 1.0)]);
            let v = f.eval(&vars).unwrap();
            assert!((v - std::f64::consts::FRAC_PI_4).abs() < 1e-10);
        }
    }

    #[test]
    fn safe_atan2_chain_rule() {
        sym! {
            let t = symbol("t");
            let f = safe_atan2(sin(t), cos(t));
            let df = f.diff("t");
            let vars = HashMap::from([("t", 0.5)]);
            let v = df.eval(&vars).unwrap();
            assert!((v - 1.0).abs() < 1e-8, "df/dt at t=0.5 = {}, expected 1", v);
        }
    }

    #[test]
    fn safe_atan2_at_zero() {
        sym! {
            let a = symbol("a");
            let b = symbol("b");
            let da = safe_atan2(a, b).diff("a");
            let vars = HashMap::from([("a", 0.0), ("b", 0.0)]);
            let v = da.eval(&vars).unwrap();
            assert!(v.is_finite(), "derivative at (0,0) should be finite, got {}", v);
        }
    }

    #[test]
    fn safe_asin_eval() {
        sym! {
            let x = symbol("x");
            let f = safe_asin(x);
            // Normal value
            let vars = HashMap::from([("x", 0.5)]);
            assert!((f.eval(&vars).unwrap() - 0.5_f64.asin()).abs() < 1e-10);
            // Clamped: safe_asin(1.5) = asin(1.0) = pi/2
            let vars = HashMap::from([("x", 1.5)]);
            assert!((f.eval(&vars).unwrap() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        }
    }

    #[test]
    fn safe_asin_deriv_finite() {
        sym! {
            let x = symbol("x");
            let da = safe_asin(x).diff("x");
            // At x=1.0, vanilla asin derivative diverges; safe version stays finite
            let vars = HashMap::from([("x", 1.0)]);
            let v = da.eval(&vars).unwrap();
            assert!(v.is_finite(), "safe_asin derivative at 1.0 should be finite, got {}", v);
        }
    }

    #[test]
    fn safe_acos_eval() {
        sym! {
            let x = symbol("x");
            let f = safe_acos(x);
            let vars = HashMap::from([("x", 0.5)]);
            assert!((f.eval(&vars).unwrap() - 0.5_f64.acos()).abs() < 1e-10);
            // Clamped: safe_acos(-1.5) = acos(-1.0) = pi
            let vars = HashMap::from([("x", -1.5)]);
            assert!((f.eval(&vars).unwrap() - std::f64::consts::PI).abs() < 1e-10);
        }
    }

    #[test]
    fn identity_codegen_parens() {
        sym! {
            let x = symbol("x");
            let f = identity(c(1.0) - x * x) + epsilon * epsilon;
            let code = f.to_rust("f64");
            // identity forces parens around its body
            assert!(code.contains("(-x.powf(2.0_f64) + 1.0_f64)"),
                "expected parens around identity body, got: {}", code);
        }
    }

    #[test]
    fn identity_diff() {
        sym! {
            let x = symbol("x");
            let f = identity(x * x);
            assert_eq!(format!("{}", f.diff("x")), "2 * x");
        }
    }

    #[test]
    fn safe_acos_deriv_finite() {
        sym! {
            let x = symbol("x");
            let da = safe_acos(x).diff("x");
            let vars = HashMap::from([("x", 1.0)]);
            let v = da.eval(&vars).unwrap();
            assert!(v.is_finite(), "safe_acos derivative at 1.0 should be finite, got {}", v);
        }
    }

    #[test]
    fn safe_sqrt_eval() {
        sym! {
            let x = symbol("x");
            let f = safe_sqrt(x);
            let vars = HashMap::from([("x", 4.0)]);
            assert!((f.eval(&vars).unwrap() - 2.0).abs() < 1e-10);
            // Negative input: safe_sqrt(-1e-10) = 0 (clamped)
            let vars = HashMap::from([("x", -1e-10)]);
            assert!(f.eval(&vars).unwrap().abs() < 1e-10);
            // Zero: safe_sqrt(0) = 0
            let vars = HashMap::from([("x", 0.0)]);
            assert!(f.eval(&vars).unwrap().abs() < 1e-10);
        }
    }

    #[test]
    fn safe_sqrt_deriv_at_zero() {
        sym! {
            let x = symbol("x");
            let df = safe_sqrt(x).diff("x");
            // At x=0, vanilla sqrt derivative diverges; safe version stays finite
            let vars = HashMap::from([("x", 0.0)]);
            let v = df.eval(&vars).unwrap();
            assert!(v.is_finite(), "safe_sqrt derivative at 0 should be finite, got {}", v);
        }
    }

    // --- Grad helper tests ---

    #[test]
    fn grad2_basic() {
        sym! {
            let g = grad2(|a, b| a * b);
            let x = symbol("x");
            let y = symbol("y");
            let [da, db] = g(x, y);
            assert_eq!(format!("{}", da), "y");
            assert_eq!(format!("{}", db), "x");
        }
    }

    #[test]
    fn grad1_basic() {
        sym! {
            let g = grad1(|t| t * t);
            let x = symbol("x");
            let [dt] = g(x);
            assert_eq!(format!("{}", dt), "2 * x");
        }
    }

    // --- Extern function tests ---

    #[test]
    fn extern_func_display() {
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_diff(x, y);
            assert_eq!(format!("{}", f), "rad_diff(x, y)");
        }
    }

    #[test]
    fn extern_func_diff() {
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_diff(x, y);
            assert_eq!(format!("{}", f.diff("x")), "1");
            assert_eq!(format!("{}", f.diff("y")), "-1");
        }
    }

    #[test]
    fn extern_func_chain_rule() {
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_diff(x * x, y);
            assert_eq!(format!("{}", f.diff("x")), "2 * x");
        }
    }

    #[test]
    fn extern_func_eval() {
        // For small angles, rad_diff(a,b) = a - b (no wrapping needed)
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_diff(x, y);
            let vars = HashMap::from([("x", 0.3), ("y", 0.1)]);
            let v = f.eval(&vars).unwrap();
            assert!((v - 0.2).abs() < 1e-10);
        }
    }

    #[test]
    fn extern_func_eval_wrapping() {
        // rad_diff(0, 2*pi) should be 0 (wrapping)
        sym! {
            let x = symbol("x");
            let f = rad_diff(constant(0.0), x);
            let vars = HashMap::from([("x", 2.0 * std::f64::consts::PI)]);
            let v = f.eval(&vars).unwrap();
            assert!(v.abs() < 1e-10, "rad_diff(0, 2*pi) = {}, expected 0", v);
        }
    }

    #[test]
    fn extern_func_to_rust() {
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_diff(x, y);
            let code = f.to_rust("f64");
            assert_eq!(code, "arael::utils::rad_diff(x, y)");
        }
    }

    #[test]
    fn extern_func_latex() {
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_diff(x, y);
            assert_eq!(f.to_latex(), "\\operatorname{rad\\_diff}\\left(x, y\\right)");
        }
    }

    #[test]
    fn extern_func_subs() {
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_diff(x, y);
            let g = f.subs("x", &constant(1.0));
            assert_eq!(format!("{}", g), "rad_diff(1, y)");
        }
    }

    #[test]
    fn extern_func_no_const_fold() {
        // Extern functions should not be constant-folded in simplify
        sym! {
            let f = rad_diff(constant(1.0), constant(2.0));
            let s = f.simplify();
            assert_eq!(format!("{}", s), "rad_diff(1, 2)");
        }
    }

    #[test]
    fn extern_func_no_expand() {
        // Extern functions should stay opaque on expand
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_diff(x + 1.0, y);
            let expanded = f.expand();
            assert_eq!(format!("{}", expanded), "rad_diff(x + 1, y)");
        }
    }

    #[test]
    fn extern_func_free_vars() {
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_diff(x, y);
            let vars = f.free_vars();
            assert!(vars.contains("x"));
            assert!(vars.contains("y"));
            assert!(!vars.contains("__a"));
            assert!(!vars.contains("__b"));
        }
    }

    #[test]
    fn rad_sum_diff() {
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_sum(x, y);
            assert_eq!(format!("{}", f.diff("x")), "1");
            assert_eq!(format!("{}", f.diff("y")), "1");
        }
    }

    #[test]
    fn rad_sum_to_rust() {
        sym! {
            let x = symbol("x");
            let y = symbol("y");
            let f = rad_sum(x, y);
            assert_eq!(f.to_rust("f64"), "arael::utils::rad_sum(x, y)");
        }
    }

    #[test]
    fn extern_func_def() {
        sym! {
            fn my_eval(args: &[f64]) -> f64 { args[0] - args[1] }
            let my_diff = extern_func2("my_diff", "my_mod::diff",
                grad2(|a, b| a - b), my_eval);
            let x = symbol("x");
            let y = symbol("y");
            let f = my_diff(x, y);
            assert_eq!(format!("{}", f), "my_diff(x, y)");
            assert_eq!(format!("{}", f.diff("x")), "1");
            assert_eq!(format!("{}", f.diff("y")), "-1");
            assert_eq!(f.to_rust("f64"), "my_mod::diff(x, y)");
        }
    }

    // --- Heaviside tests ---

    #[test]
    fn heaviside_eval() {
        let vars = HashMap::from([("x", 0.0)]);
        sym! {
            let x = symbol("x");
            let h = heaviside(x);
            assert_eq!(h.eval(&HashMap::from([("x", -1.0)])).unwrap(), 0.0);
            assert_eq!(h.eval(&vars).unwrap(), 1.0);
            assert_eq!(h.eval(&HashMap::from([("x", 3.0)])).unwrap(), 1.0);
        }
    }

    #[test]
    fn heaviside_diff() {
        sym! {
            let x = symbol("x");
            assert_eq!(format!("{}", heaviside(x).diff("x")), "0");
            assert_eq!(format!("{}", heaviside(x * x - 1.0).diff("x")), "0");
        }
    }

    #[test]
    fn heaviside_display() {
        sym! {
            let x = symbol("x");
            assert_eq!(format!("{}", heaviside(x)), "H(x)");
        }
    }

    #[test]
    fn heaviside_composition_diff() {
        sym! {
            let x = symbol("x");
            // d/dx [H(1-x) * x^2] = 2x (H' = 0, product rule kills that term)
            let f = heaviside(1.0 - x) * x * x;
            assert_eq!(format!("{}", f.diff("x")), "2 * x * H(-x + 1)");
        }
    }

    // --- Clamp tests ---

    #[test]
    fn clamp_eval() {
        sym! {
            let x = symbol("x");
            let f = clamp(x, c(0.0), c(1.0));
            assert_eq!(f.eval(&HashMap::from([("x", 0.5)])).unwrap(), 0.5);
            assert_eq!(f.eval(&HashMap::from([("x", -2.0)])).unwrap(), 0.0);
            assert_eq!(f.eval(&HashMap::from([("x", 5.0)])).unwrap(), 1.0);
        }
    }

    #[test]
    fn clamp_diff_passthrough() {
        sym! {
            let x = symbol("x");
            // d/dx clamp(x, 0, 1) = 1 (pass-through)
            assert_eq!(format!("{}", clamp(x, c(0.0), c(1.0)).diff("x")), "1");
            // d/dx clamp(x^2, 0, 1) = 2x (chain rule on first arg)
            assert_eq!(format!("{}", clamp(x * x, c(0.0), c(1.0)).diff("x")), "2 * x");
        }
    }

    #[test]
    fn clamp_display() {
        sym! {
            let x = symbol("x");
            assert_eq!(format!("{}", clamp(x, c(0.0), c(1.0))), "clamp(x, 0, 1)");
        }
    }

    #[test]
    fn clamp_simplify_constants() {
        sym! {
            let f = clamp(c(5.0), c(0.0), c(1.0));
            assert_eq!(format!("{}", f.simplify()), "1");
            let g = clamp(c(-3.0), c(0.0), c(1.0));
            assert_eq!(format!("{}", g.simplify()), "0");
            let h = clamp(c(0.5), c(0.0), c(1.0));
            assert_eq!(format!("{}", h.simplify()), "0.5");
        }
    }

    // --- clamp-based safe_asin tests (simple_func1 version) ---

    #[test]
    fn clamp_asin_eval() {
        sym! {
            let my_asin = simple_func1("my_asin", |t| asin(clamp(t, c(-1.0), c(1.0))));
            let x = symbol("x");

            // Normal value
            let f = my_asin(x);
            let val = f.eval(&HashMap::from([("x", 0.5)])).unwrap();
            assert!((val - 0.5_f64.asin()).abs() < 1e-10);

            // Out of range: no NaN
            let val_hi = f.eval(&HashMap::from([("x", 1.5)])).unwrap();
            assert!((val_hi - std::f64::consts::FRAC_PI_2).abs() < 1e-10);

            let val_lo = f.eval(&HashMap::from([("x", -1.5)])).unwrap();
            assert!((val_lo + std::f64::consts::FRAC_PI_2).abs() < 1e-10);
        }
    }

    #[test]
    fn clamp_asin_diff() {
        sym! {
            let my_asin = simple_func1("my_asin", |t| asin(clamp(t, c(-1.0), c(1.0))));
            let x = symbol("x");
            let f = my_asin(x);
            // Derivative: 1/sqrt(1 - clamp(x,-1,1)^2) * 1 (clamp pass-through)
            let df = f.diff("x");
            // Numerically verify at x=0.5
            let vars = HashMap::from([("x", 0.5)]);
            let dval = df.eval(&vars).unwrap();
            let expected = 1.0 / (1.0 - 0.25_f64).sqrt(); // 1/sqrt(0.75)
            assert!((dval - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn heaviside_to_rust() {
        sym! {
            let x = symbol("x");
            assert_eq!(heaviside(x).to_rust("f64"), "x.heaviside()");
        }
    }

    #[test]
    fn clamp_to_rust() {
        sym! {
            let x = symbol("x");
            assert_eq!(clamp(x, c(0.0), c(1.0)).to_rust("f64"), "x.clamp(0.0_f64, 1.0_f64)");
        }
    }

    #[test]
    fn parse_heaviside() {
        let f = parse("H(x)").unwrap();
        assert_eq!(format!("{}", f), "H(x)");
        assert_eq!(format!("{}", f.diff("x")), "0");
    }

    #[test]
    fn parse_clamp() {
        let f = parse("clamp(x, 0, 1)").unwrap();
        assert_eq!(format!("{}", f), "clamp(x, 0, 1)");
        assert_eq!(format!("{}", f.diff("x")), "1");
    }

    // --- Named constant tests ---

    #[test]
    fn named_const_pi_display() {
        assert_eq!(format!("{}", pi()), "pi");
    }

    #[test]
    fn named_const_pi_eval() {
        let vars = HashMap::new();
        assert_eq!(pi().eval(&vars).unwrap(), std::f64::consts::PI);
    }

    #[test]
    fn named_const_pi_diff() {
        assert_eq!(format!("{}", pi().diff("x")), "0");
    }

    #[test]
    fn named_const_pi_codegen() {
        assert_eq!(pi().to_rust("f64"), "std::f64::consts::PI");
        assert_eq!(pi().to_rust("f32"), "std::f32::consts::PI");
    }

    #[test]
    fn named_const_pi_latex() {
        assert_eq!(pi().to_latex(), "\\pi");
    }

    #[test]
    fn named_const_epsilon_display() {
        assert_eq!(format!("{}", epsilon()), "epsilon");
    }

    #[test]
    fn named_const_epsilon_eval() {
        let vars = HashMap::new();
        assert_eq!(epsilon().eval(&vars).unwrap(), f64::EPSILON);
    }

    #[test]
    fn named_const_epsilon_codegen() {
        assert_eq!(epsilon().to_rust("f64"), "f64::EPSILON");
        assert_eq!(epsilon().to_rust("f32"), "f32::EPSILON");
    }

    #[test]
    fn named_const_euler_display() {
        assert_eq!(format!("{}", euler()), "e");
    }

    #[test]
    fn named_const_euler_eval() {
        let vars = HashMap::new();
        assert_eq!(euler().eval(&vars).unwrap(), std::f64::consts::E);
    }

    #[test]
    fn named_const_euler_codegen() {
        assert_eq!(euler().to_rust("f64"), "std::f64::consts::E");
    }

    #[test]
    fn named_const_epsilon_survives_simplification() {
        sym! {
            let x = symbol("x");
            let f = (x + epsilon()).simplify();
            assert_eq!(format!("{}", f), "x + epsilon");
        }
    }

    #[test]
    fn named_const_not_free_var() {
        sym! {
            let x = symbol("x");
            let f = x + pi();
            let vars = f.free_vars();
            assert!(vars.contains("x"));
            assert!(!vars.contains("pi"));
        }
    }

    #[test]
    fn named_const_custom() {
        let tau = named_const("tau", std::f64::consts::TAU,
            "std::f32::consts::TAU", "std::f64::consts::TAU", "\\tau");
        assert_eq!(format!("{}", tau), "tau");
        let vars = HashMap::new();
        assert_eq!(tau.eval(&vars).unwrap(), std::f64::consts::TAU);
        assert_eq!(tau.to_rust("f64"), "std::f64::consts::TAU");
        assert_eq!(tau.to_latex(), "\\tau");
    }

    // --- Algebraic simplification of named constants ---

    #[test]
    fn named_const_pi_add_pi() {
        sym! {
            let f = (pi() + pi()).simplify();
            assert_eq!(format!("{}", f), "2 * pi");
        }
    }

    #[test]
    fn named_const_pi_sub_pi() {
        sym! {
            let f = (pi() - pi()).simplify();
            assert_eq!(format!("{}", f), "0");
        }
    }

    #[test]
    fn named_const_pi_mul_pi() {
        sym! {
            let f = (pi() * pi()).simplify();
            assert_eq!(format!("{}", f), "pi^2");
        }
    }

    #[test]
    fn named_const_epsilon_add() {
        sym! {
            let x = symbol("x");
            let f = (x + epsilon() + epsilon()).simplify();
            assert_eq!(format!("{}", f), "x + 2 * epsilon");
        }
    }

    // --- Trig-pi simplification ---

    #[test]
    fn trig_sin_pi() {
        sym! { assert_eq!(format!("{}", sin(pi()).simplify()), "0"); }
    }

    #[test]
    fn trig_cos_pi() {
        sym! { assert_eq!(format!("{}", cos(pi()).simplify()), "-1"); }
    }

    #[test]
    fn trig_sin_pi_half() {
        sym! { assert_eq!(format!("{}", sin(pi() / 2.0).simplify()), "1"); }
    }

    #[test]
    fn trig_cos_pi_half() {
        sym! { assert_eq!(format!("{}", cos(pi() / 2.0).simplify()), "0"); }
    }

    #[test]
    fn trig_sin_pi_quarter() {
        sym! {
            let f = sin(pi() / 4.0).simplify();
            let vars = HashMap::new();
            let v = f.eval(&vars).unwrap();
            assert!((v - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
        }
    }

    #[test]
    fn trig_cos_pi_third() {
        sym! {
            let f = cos(pi() / 3.0).simplify();
            assert_eq!(format!("{}", f), "0.5");
        }
    }

    #[test]
    fn trig_sin_2pi() {
        sym! { assert_eq!(format!("{}", sin(2.0 * pi()).simplify()), "0"); }
    }

    #[test]
    fn trig_cos_2pi() {
        sym! { assert_eq!(format!("{}", cos(2.0 * pi()).simplify()), "1"); }
    }

    #[test]
    fn trig_tan_pi() {
        sym! { assert_eq!(format!("{}", tan(pi()).simplify()), "0"); }
    }

    #[test]
    fn trig_sin_pi_sixth() {
        sym! { assert_eq!(format!("{}", sin(pi() / 6.0).simplify()), "0.5"); }
    }

    // --- Log/exp-e simplification ---

    #[test]
    fn ln_e() {
        sym! { assert_eq!(format!("{}", ln(euler()).simplify()), "1"); }
    }

    // --- sym! macro bare identifier tests ---

    #[test]
    fn sym_macro_bare_pi() {
        sym! {
            let x = symbol("x");
            let f = 2.0 * pi * x;
            assert_eq!(format!("{}", f), "2 * x * pi");
        }
    }

    #[test]
    fn sym_macro_bare_epsilon() {
        sym! {
            let x = symbol("x");
            let f = x * x + epsilon;
            assert_eq!(format!("{}", f), "x^2 + epsilon");
        }
    }

    #[test]
    fn sym_macro_pi_call_still_works() {
        // pi() with parens should also work (not double-rewritten)
        sym! {
            let f = pi();
            assert_eq!(format!("{}", f), "pi");
        }
    }

    #[test]
    fn ln_e_pow_x() {
        sym! {
            let x = symbol("x");
            let f = ln(pow(euler(), x)).simplify();
            assert_eq!(format!("{}", f), "x");
        }
    }
}

