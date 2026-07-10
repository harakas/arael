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
//! See [`docs/SYM.md`](https://github.com/harakas/arael/blob/master/docs/SYM.md)
//! for the full reference with worked examples for every feature,
//! and [`examples/sym_demo.rs`](https://github.com/harakas/arael/blob/master/examples/sym_demo.rs)
//! for a runnable walkthrough (`cargo run --example sym_demo`).
//! The tour below hits the high points.
//!
//! ## Basics
//!
//! The [`symbols!`] macro expands each bare identifier to
//! `symbol("<name>")` and returns a tuple -- you write the name once
//! instead of twice per variable. The [`sym!`] macro auto-inserts
//! `.clone()` on every reused variable so the body reads as natural
//! math without ownership boilerplate.
//!
//! Every expression has type [`E`], defined as
//! `struct E(Rc<Expr>)`. Cloning is cheap (a reference-count bump) --
//! the `.clone()` calls `sym!` inserts don't duplicate the
//! expression tree.
//!
//! ```
//! use arael_sym::*;
//! let result = sym! {
//!     let (x, y) = symbols!(x, y);
//!     let f = x * y - 1.0 + pow(x, 2.0);
//!     format!("{}", f)
//! };
//! assert_eq!(result, "x * y + x^2 - 1");
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
//!     format!("{}", f.diff(x))
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
//!     let (x, y) = symbols!(x, y);
//!     let f = sin(x) + 1.0;
//!     let g = atan2(y, x);
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
//!     let (x, y, z) = symbols!(x, y, z);
//!     let v = SymVec::new([x, y, z]);
//!     let w = SymVec::new([1.0, 2.0, 3.0]);
//!     format!("{}", v.dot(&w))
//! };
//! assert_eq!(dot, "x + 2 * y + 3 * z");
//! ```
//!
//! ## Geometric primitives
//!
//! Fixed-shape companions to the runtime `vect{2,3}f` / `matrix{2,3}f` types
//! used inside `#[arael::model]` constraint bodies. They live in
//! [`geo`] and are re-exported at the crate root:
//! [`vect2sym`], [`vect3sym`], [`matrix2sym`], [`matrix3sym`], [`quaternsym`].
//!
//! ```
//! use arael_sym::*;
//! let r = matrix2sym::rotation(symbol("a"));      // 2D rotation
//! let v = vect2sym::new("v");
//! let rv = r * v;
//! assert_eq!(format!("{}", rv.x.simplify()), "v.x * cos(a) - v.y * sin(a)");
//! ```
//!
//! See `docs/SYM.md` for the full surface (transpose, mat*vec, mat*mat,
//! `vect3sym::rotation_matrix()`, `matrix3sym::get_euler_angles()`, etc.).
//!
//! ## Jacobian
//!
//! ```
//! use arael_sym::*;
//! let (j00, j01, j10, j11) = sym! {
//!     let (x, y) = symbols!(x, y);
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
//!     let safe = identity(1.0 - x * x) + epsilon * epsilon;
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
//!     let x = symbol("x");
//!     let square = simple_func1("square", |t| t * t);
//!     let f = square(x + 1.0);
//!     assert_eq!(format!("{}", f), "square(x + 1)");
//!     assert_eq!(format!("{}", f.diff(x)), "2 * (x + 1)");
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
//!     let (x, y) = symbols!(x, y);
//!     let f = angle_diff(x * x, y);
//!     assert_eq!(format!("{}", f.diff(x)), "2 * x");
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
//!         |t| asin(clamp(t, -1.0, 1.0)));
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

    /// Return true if this expression is the literal constant zero.
    /// After `simplify`, a structurally-zero expression (e.g. the
    /// derivative of a residual with respect to a parameter it does not
    /// touch) is exactly `Const(0.0)`; codegen uses this to elide dead
    /// derivative emission and block accumulation calls.
    pub fn is_zero(&self) -> bool {
        matches!(&*self.0, Expr::Const(v) if *v == 0.0)
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
            Expr::Clamp(a, b, c) | Expr::Branch(a, b, c) => {
                a.collect_symbols(out);
                b.collect_symbols(out);
                c.collect_symbols(out);
            }
            Expr::Func { params, kind, args, .. } => {
                // Captured symbols: a function body may reference symbols
                // beyond its params (eval resolves them from the outer
                // vars map), so they are free symbols of this expression.
                // Params themselves are bound.
                if let Some(body) = kind.body() {
                    let mut inner = std::collections::HashSet::new();
                    body.collect_symbols(&mut inner);
                    for n in inner {
                        if !params.contains(&n) { out.insert(n); }
                    }
                }
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
            Expr::Log10(a) => log10(a.substitute(subs)),
            Expr::Sqrt(a) => sqrt(a.substitute(subs)),
            Expr::Abs(a) => abs(a.substitute(subs)),
            Expr::Heaviside(a) => heaviside(a.substitute(subs)),
            Expr::Clamp(a, lo, hi) => clamp(a.substitute(subs), lo.substitute(subs), hi.substitute(subs)),
            Expr::Branch(q, a, b) => branch(q.substitute(subs), a.substitute(subs), b.substitute(subs)),
            Expr::Func { name, params, kind, args } => {
                let new_args = args.iter().map(|a| a.substitute(subs)).collect();
                E::new(Expr::Func { name: name.clone(), params: params.clone(), kind: kind.clone(), args: new_args })
            }
        }
    }

    /// Replace every occurrence of the named function with the expression
    /// built by `f` from its (already-transformed) arguments, recursing
    /// through the whole tree. Matches built-in nodes by their
    /// conventional name (`"sin"`, `"atan2"`, `"pow"`, ... -- the names in
    /// [`FUNCTIONS`]) and [`Expr::Func`] nodes by their `name`. Operators
    /// (`+`, `*`, unary `-`, ...) are not addressable. `f` receives the
    /// node's arguments in order; backs e.g. the `#[arael(root,
    /// fast_atan)]` keyword, which maps `atan`/`atan2` onto [`fast_atan`]
    /// / [`fast_atan2`].
    pub fn replace_function(&self, name: &str, f: &dyn Fn(&[E]) -> E) -> E {
        let rec = |a: &E| a.replace_function(name, f);
        macro_rules! un {
            ($nm:literal, $ctor:ident, $a:ident) => {{
                let a = rec($a);
                if name == $nm { f(&[a]) } else { $ctor(a) }
            }};
        }
        match &*self.0 {
            Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => self.clone(),
            Expr::Neg(a) => -rec(a),
            Expr::Add(a, b) => rec(a) + rec(b),
            Expr::Sub(a, b) => rec(a) - rec(b),
            Expr::Mul(a, b) => rec(a) * rec(b),
            Expr::Div(a, b) => rec(a) / rec(b),
            Expr::Pow(a, b) => {
                let (a, b) = (rec(a), rec(b));
                if name == "pow" { f(&[a, b]) } else { pow(a, b) }
            }
            Expr::Sin(a) => un!("sin", sin, a),
            Expr::Cos(a) => un!("cos", cos, a),
            Expr::Tan(a) => un!("tan", tan, a),
            Expr::Asin(a) => un!("asin", asin, a),
            Expr::Acos(a) => un!("acos", acos, a),
            Expr::Atan(a) => un!("atan", atan, a),
            Expr::Atan2(y, x) => {
                let (y, x) = (rec(y), rec(x));
                if name == "atan2" { f(&[y, x]) } else { atan2(y, x) }
            }
            Expr::Sinh(a) => un!("sinh", sinh, a),
            Expr::Cosh(a) => un!("cosh", cosh, a),
            Expr::Tanh(a) => un!("tanh", tanh, a),
            Expr::Exp(a) => un!("exp", exp, a),
            Expr::Ln(a) => un!("ln", ln, a),
            Expr::Log2(a) => un!("log2", log2, a),
            Expr::Log10(a) => un!("log10", log10, a),
            Expr::Sqrt(a) => un!("sqrt", sqrt, a),
            Expr::Abs(a) => un!("abs", abs, a),
            Expr::Heaviside(a) => un!("heaviside", heaviside, a),
            Expr::Clamp(a, lo, hi) => {
                let (a, lo, hi) = (rec(a), rec(lo), rec(hi));
                if name == "clamp" { f(&[a, lo, hi]) } else { clamp(a, lo, hi) }
            }
            Expr::Branch(q, a, b) => {
                let (q, a, b) = (rec(q), rec(a), rec(b));
                if name == "branch" { f(&[q, a, b]) } else { branch(q, a, b) }
            }
            Expr::Func { name: fname, params, kind, args } => {
                let new_args: Vec<E> = args.iter().map(rec).collect();
                if fname == name {
                    f(&new_args)
                } else {
                    E::new(Expr::Func { name: fname.clone(), params: params.clone(),
                                        kind: kind.clone(), args: new_args })
                }
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
    /// Branch select: `branch(q, a, b) = q >= 0 ? a : b`. Only the taken
    /// side is evaluated. The derivative selects the taken side's
    /// derivative -- the switch is piecewise-constant, so `q` contributes
    /// nothing (like Heaviside).
    Branch(E, E, E),
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
            Expr::Clamp(a, b, c) | Expr::Branch(a, b, c) => {
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

/// Types that can name a symbolic variable for operations that key
/// into the expression tree by name -- `diff`, `subs`, `collect`.
/// Implemented for `&str`, `String`, `&String`, and [`E`] (when it
/// wraps a `Sym` node), so you can write `expr.diff("x")` or
/// `expr.diff(&my_symbol)` and reach the same variable. The blanket
/// `var_expr` default builds a fresh `Sym` node from the name;
/// implementations on [`E`] override it to reuse the caller's handle
/// and avoid an allocation.
pub trait AsVarName {
    /// Return the variable name as a string slice.
    fn var_name(&self) -> &str;

    /// Return an `E` representing this variable. Default: build a
    /// fresh `Sym` node from `var_name()`.
    fn var_expr(&self) -> E {
        symbol(self.var_name())
    }
}

impl AsVarName for &str {
    fn var_name(&self) -> &str { self }
}

impl AsVarName for &&str {
    fn var_name(&self) -> &str { self }
}

impl AsVarName for str {
    fn var_name(&self) -> &str { self }
}

impl AsVarName for String {
    fn var_name(&self) -> &str { self.as_str() }
}

impl AsVarName for &String {
    fn var_name(&self) -> &str { self.as_str() }
}

impl AsVarName for &E {
    fn var_name(&self) -> &str { (*self).var_name() }
    fn var_expr(&self) -> E { (*self).clone() }
}

impl AsVarName for E {
    fn var_name(&self) -> &str {
        match self.as_ref() {
            Expr::Sym(name) => name.as_str(),
            _ => panic!("AsVarName::var_name: expected a symbol, got `{self}`"),
        }
    }
    fn var_expr(&self) -> E { self.clone() }
}

/// Create several symbolic variables at once and return them as a
/// tuple. Each identifier becomes a fresh [`E`] whose name is that
/// identifier stringified, sparing the caller from writing the name
/// twice per variable.
///
/// ```
/// use arael_sym::*;
/// let (x, y, z) = symbols!(x, y, z);
/// assert_eq!(format!("{}", x * y + z), "x * y + z");
/// ```
///
/// A trailing comma in the expansion makes the single-identifier
/// form a 1-tuple (`(E,)`); for a single symbol [`symbol`] is
/// usually the clearer spelling.
#[macro_export]
macro_rules! symbols {
    ($($name:ident),+ $(,)?) => {
        ( $( $crate::symbol(stringify!($name)) ),+ , )
    };
}

/// Create a numeric constant.
pub fn constant(val: f64) -> E {
    E::new(Expr::Const(val))
}

impl From<f64> for E {
    fn from(v: f64) -> E { constant(v) }
}

impl From<i64> for E {
    fn from(v: i64) -> E { constant(v as f64) }
}

impl From<i32> for E {
    fn from(v: i32) -> E { constant(v as f64) }
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

/// Integer value of `v` when it is exactly integer-valued AND small
/// enough for the f64 -> i64 conversion to be lossless.
///
/// The magnitude bound is load-bearing, not a nicety: every f64 at or
/// above 2^53 has no fractional bits, so `v == v.floor()` is true for
/// ALL huge floats (1e300 included) -- and casting those to i64
/// saturates, silently producing a wrong integer. Below 2^53
/// (about 9.007e15) every integer is exactly representable and the
/// round-trip is exact; 1e15 is a round decimal bound comfortably
/// inside that. Shared by integer-exponent detection (simplify) and
/// literal formatting (fmt).
pub(crate) fn as_exact_int(v: f64) -> Option<i64> {
    if v == v.floor() && v.abs() < 1e15 {
        Some(v as i64)
    } else {
        None
    }
}

/// Machine epsilon $\epsilon$ (`f64::EPSILON` $\approx 2.22 \times 10^{-16}$).
pub fn epsilon() -> E {
    named_const("epsilon", f64::EPSILON,
        "f32::EPSILON", "f64::EPSILON", "\\epsilon")
}

/// Machine epsilon anchored to the TYPE of `anchor` (its value is ignored):
/// codegen emits `arael::utils::epsilon_for(<anchor>)`, so the precision is
/// inferred from a nearby concrete value -- f32 code gets `f32::EPSILON`, f64
/// code `f64::EPSILON`. Symbolic eval returns `f64::EPSILON`; the derivative
/// w.r.t. the anchor is 0 (it is a constant).
///
/// Prefer this over the nullary [`epsilon`] inside a machine-precision guard
/// emitted into type-inferred (unsuffixed) code: [`epsilon`] folds to a single
/// literal there, fixing the wrong precision for f32; `epsilon_for` does not.
pub fn epsilon_for(anchor: E) -> E {
    extern_func1("epsilon_for", "arael::utils::epsilon_for",
        |_| [c(0.0)],
        |_args: &[f64]| f64::EPSILON)(anchor)
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
/// Accepts `impl Into<E>` on all three args so bare numeric bounds
/// compose naturally: `clamp(x, -1.0, 1.0)`.
pub fn clamp(val: impl Into<E>, lo: impl Into<E>, hi: impl Into<E>) -> E {
    E::new(Expr::Clamp(val.into(), lo.into(), hi.into()))
}
/// Symbolic branch select: `branch(q, a, b) = q >= 0 ? a : b`. Only the taken
/// side is evaluated (in both interpretation and generated code), so `a` / `b`
/// may be undefined off their side. The derivative selects the taken side's
/// derivative -- the switch `q` contributes nothing, like Heaviside. Accepts
/// `impl Into<E>` so bare numeric arms compose: `branch(x, 1.0, -1.0)`.
pub fn branch(q: impl Into<E>, a: impl Into<E>, b: impl Into<E>) -> E {
    E::new(Expr::Branch(q.into(), a.into(), b.into()))
}
/// Symbolic power function. Auto-simplifies (e.g. x^0 = 1, x^1 = x).
/// Accepts `impl Into<E>` for both args so bare numeric literals
/// compose naturally: `pow(x, 2.0)`, `pow(x, 3)`.
pub fn pow(base: impl Into<E>, exponent: impl Into<E>) -> E {
    E::new(Expr::Pow(base.into(), exponent.into())).simplify()
}

// ---------------------------------------------------------------------------
// Name-based function lookup
//
// Users that parse an expression tree (for example arael-macros turning a
// constraint body or a fit expression into an arael_sym::E) need to map
// function-name tokens like "sin", "atan2", "clamp" to the actual arael-sym
// function. Keeping the authoritative list here (next to the functions
// themselves) means external dispatchers don't have to duplicate it and
// new functions land everywhere for free.
// ---------------------------------------------------------------------------

/// A scalar function exported by arael-sym, discovered by name. Tagged by
/// arity so callers can validate the argument count without a second table.
#[derive(Clone, Copy)]
pub enum FunctionRef {
    Unary(fn(E) -> E),
    Binary(fn(E, E) -> E),
    Ternary(fn(E, E, E) -> E),
}

/// The authoritative table of scalar functions arael-sym exposes by name.
/// Adding a new `pub fn foo` above should add an entry here as well; every
/// string-based dispatcher (the parser, the macro's constraint/fit
/// dispatchers, user-facing autocompleters) reads from this one table.
pub const FUNCTIONS: &[(&str, FunctionRef)] = &[
    // Unary trig
    ("sin", FunctionRef::Unary(sin)),
    ("cos", FunctionRef::Unary(cos)),
    ("tan", FunctionRef::Unary(tan)),
    ("asin", FunctionRef::Unary(asin)),
    ("acos", FunctionRef::Unary(acos)),
    ("atan", FunctionRef::Unary(atan)),
    ("sinh", FunctionRef::Unary(sinh)),
    ("cosh", FunctionRef::Unary(cosh)),
    ("tanh", FunctionRef::Unary(tanh)),
    // Unary exp / log / pow-ish
    ("exp", FunctionRef::Unary(exp)),
    ("ln", FunctionRef::Unary(ln)),
    ("log2", FunctionRef::Unary(log2)),
    ("log10", FunctionRef::Unary(log10)),
    ("sqrt", FunctionRef::Unary(sqrt)),
    ("abs", FunctionRef::Unary(abs)),
    ("heaviside", FunctionRef::Unary(heaviside)),
    // Unary "safe" variants
    ("identity", FunctionRef::Unary(identity)),
    ("cached", FunctionRef::Unary(cached)),
    ("safe_sqrt", FunctionRef::Unary(safe_sqrt)),
    ("safe_asin", FunctionRef::Unary(safe_asin)),
    ("safe_acos", FunctionRef::Unary(safe_acos)),
    // Machine epsilon anchored to the argument's type (value ignored).
    ("epsilon_for", FunctionRef::Unary(epsilon_for)),
    ("fast_atan", FunctionRef::Unary(fast_atan)),
    // Binary
    ("atan2", FunctionRef::Binary(atan2)),
    ("pow", FunctionRef::Binary(pow)),
    ("safe_atan2", FunctionRef::Binary(safe_atan2)),
    ("fast_atan2", FunctionRef::Binary(fast_atan2)),
    ("rad_diff", FunctionRef::Binary(rad_diff)),
    ("rad_sum", FunctionRef::Binary(rad_sum)),
    // Ternary
    ("clamp", FunctionRef::Ternary(clamp)),
    ("branch", FunctionRef::Ternary(branch)),
];

/// Look up a scalar function by its conventional name. Returns `None` for
/// unrecognized names -- callers typically emit a user-facing error in that
/// case.
pub fn function_by_name(name: &str) -> Option<FunctionRef> {
    FUNCTIONS.iter().find(|(n, _)| *n == name).map(|(_, f)| *f)
}

/// Iterate over the names of every scalar function arael-sym exposes.
/// Useful for autocomplete and "what functions are available?" queries.
pub fn function_names() -> impl Iterator<Item = &'static str> {
    FUNCTIONS.iter().map(|(n, _)| *n)
}

// ---------------------------------------------------------------------------
// FunctionBag -- extensible registry of user-defined functions
// ---------------------------------------------------------------------------

/// An extensible registry of user-defined symbolic functions, used by
/// [`parse::parse_with_functions`] to make runtime-constructed
/// functions recognisable by the string parser.
///
/// Built-in functions (`sin`, `cos`, `clamp`, etc.) are *not* stored in
/// the bag -- the parser falls back to [`function_by_name`] for any
/// name the bag doesn't carry, so built-ins are always available
/// regardless of what's in the bag. An empty bag means "built-ins
/// only", which is what [`parse::parse`] uses.
///
/// Names registered in the bag shadow built-ins with the same name.
///
/// ## Registering a function
///
/// Pick the entry point that fits how you have the function in hand:
///
/// - [`add1`](Self::add1) / [`add2`](Self::add2) -- register a closure
///   of arity 1 / 2, typically one produced by [`simple_func1`] /
///   [`simple_func2`] / [`extern_func1`] / [`extern_func2`]. The bag
///   invokes it once with placeholder symbols to extract name, params,
///   and kind.
/// - [`addN`](Self::addN) -- register an n-ary closure over `Vec<E>`.
///   Pairs with [`simple_func`] / [`simple_func_derivs`] /
///   [`extern_func`]. No upper arity bound.
/// - [`add`](Self::add) -- register an already-formed `Expr::Func`
///   value (for example, the output of
///   [`simple_func1`]`("sq", |t| t*t)(symbol("x"))`).
/// - [`add_symbolic`](Self::add_symbolic) -- when the body is an
///   already-built `E` (e.g. from [`parse::parse`]) and you don't want
///   to wrap it in a closure. Body is auto-differentiated.
/// - [`add_with_kind`](Self::add_with_kind) -- escape hatch: name,
///   parameter list, and a hand-built [`FuncKind`] directly.
///
/// ## Variable / parameter shadowing
///
/// Parameters declared when the function is registered always shadow
/// variables of the same name in the caller's eval context. For
/// example, after:
///
/// ```ignore
/// let mut bag = FunctionBag::new();
/// bag.add_symbolic("sq", vec!["x".into()], parse("x*x").unwrap());
/// let e = parse_with_functions("sq(3)", &bag).unwrap();
/// let vars = [("x", 5.0)].into_iter().collect();
/// let r = e.eval(&vars).unwrap(); // 9.0, not 25.0
/// ```
///
/// the outer `x = 5.0` is shadowed inside the function body by the
/// formal parameter `x = 3.0` for the duration of the call.
///
/// ## See also
///
/// [`examples/calc_demo.rs`](https://github.com/harakas/arael/blob/master/examples/calc_demo.rs)
/// is a bc-style REPL calculator built on top of `FunctionBag` +
/// [`parse::parse_with_functions`]: variables, runtime function
/// definitions (`name(args) = expr`), `vars` / `funcs` listings, and
/// readline-style history.
#[derive(Clone)]
pub struct FunctionBag {
    // Name -> (params, kind). Args are filled in at call time to build
    // a fresh Expr::Func per invocation. Mirrors Expr::Func directly.
    table: std::collections::HashMap<String, BagFunction>,
}

#[derive(Clone)]
struct BagFunction {
    params: std::vec::Vec<String>,
    kind: FuncKind,
}

impl Default for FunctionBag {
    fn default() -> Self { Self::new() }
}

fn extract_func_template(e: E, source: &str) -> Result<(String, std::vec::Vec<String>, FuncKind), String> {
    match (*e.0).clone() {
        Expr::Func { name, params, kind, .. } => Ok((name, params, kind)),
        _ => Err(format!("{source}: expected Expr::Func, got a different expression")),
    }
}

impl FunctionBag {
    /// Empty bag. Built-in functions remain available via the parser's
    /// fallback lookup; only user-added functions go here.
    pub fn new() -> Self {
        Self { table: std::collections::HashMap::new() }
    }

    /// Register a pre-built `Expr::Func` value. Use when you already
    /// have an `E` (for example by calling one of the
    /// [`simple_func1`] / [`simple_func2`] / [`simple_func`] /
    /// [`extern_func1`] / [`extern_func2`] / [`extern_func`]
    /// constructors on placeholder args).
    ///
    /// For registering closures directly, use [`add1`](Self::add1) /
    /// [`add2`](Self::add2) / [`addN`](Self::addN).
    ///
    /// Returns `Err` if `e` is not an `Expr::Func`.
    pub fn add(&mut self, e: E) -> Result<(), String> {
        let (name, params, kind) = extract_func_template(e, "FunctionBag::add")?;
        self.table.insert(name, BagFunction { params, kind });
        Ok(())
    }

    /// Register a unary closure. The bag invokes it once with a
    /// placeholder symbol to extract `(name, params, kind)`.
    ///
    /// ```ignore
    /// bag.add1(simple_func1("sq", |t| t.clone() * t)).unwrap();
    /// ```
    pub fn add1<F>(&mut self, f: F) -> Result<(), String>
    where F: FnOnce(E) -> E
    {
        let e = f(symbol("__a0"));
        let (name, params, kind) = extract_func_template(e, "FunctionBag::add1")?;
        self.table.insert(name, BagFunction { params, kind });
        Ok(())
    }

    /// Register a binary closure.
    ///
    /// ```ignore
    /// bag.add2(simple_func2("hypot",
    ///     |a, b| sqrt(a.clone()*a + b.clone()*b))).unwrap();
    /// ```
    pub fn add2<F>(&mut self, f: F) -> Result<(), String>
    where F: FnOnce(E, E) -> E
    {
        let e = f(symbol("__a0"), symbol("__a1"));
        let (name, params, kind) = extract_func_template(e, "FunctionBag::add2")?;
        self.table.insert(name, BagFunction { params, kind });
        Ok(())
    }

    /// Register an n-ary closure. Pairs with [`simple_func`] /
    /// [`simple_func_derivs`] / [`extern_func`] for arities >= 3 and
    /// for functions whose arity is known only at runtime. The
    /// closure takes `Vec<E>` to match the shape those constructors
    /// return (`impl Fn(Vec<E>) -> E`).
    ///
    /// ```ignore
    /// bag.addN(4, simple_func("blend", 4, |args: Vec<E>|
    ///     args[0].clone() + args[1].clone() + args[2].clone() + args[3].clone()
    /// )).unwrap();
    /// ```
    #[allow(non_snake_case)]
    pub fn addN<F>(&mut self, arity: usize, f: F) -> Result<(), String>
    where F: FnOnce(std::vec::Vec<E>) -> E
    {
        let placeholders: std::vec::Vec<E> =
            (0..arity).map(|i| symbol(&format!("__a{i}"))).collect();
        let e = f(placeholders);
        let (name, params, kind) = extract_func_template(e, "FunctionBag::addN")?;
        self.table.insert(name, BagFunction { params, kind });
        Ok(())
    }

    /// Convenience: register a symbolic function from an explicit
    /// `name`, parameter list, and body `E` whose free symbols match
    /// the params. Use this when you have the body as an already-built
    /// expression (e.g. from [`parse`]) rather than as a closure.
    pub fn add_symbolic(&mut self, name: impl Into<String>, params: std::vec::Vec<String>, body: E) {
        self.table.insert(
            name.into(),
            BagFunction { params, kind: FuncKind::Symbolic { body } },
        );
    }

    /// Direct form: register a function from name + parameters + kind.
    /// Most callers should prefer [`add`](Self::add) (closures / E)
    /// or [`add_symbolic`](Self::add_symbolic) (parsed body) -- this
    /// is the escape hatch for building an unusual `FuncKind` by hand.
    pub fn add_with_kind(
        &mut self,
        name: impl Into<String>,
        params: std::vec::Vec<String>,
        kind: FuncKind,
    ) {
        self.table.insert(name.into(), BagFunction { params, kind });
    }

    /// Remove a function by name. Returns `true` if it was present.
    /// Does not affect built-ins.
    pub fn remove(&mut self, name: &str) -> bool {
        self.table.remove(name).is_some()
    }

    /// Is this name registered in the bag? Does *not* consider
    /// built-ins.
    pub fn contains(&self, name: &str) -> bool {
        self.table.contains_key(name)
    }

    /// Collect all names registered in the bag. Order is unspecified.
    pub fn names(&self) -> std::vec::Vec<String> {
        self.table.keys().cloned().collect()
    }

    /// Iterate over `(name, arity)` pairs for every function in the
    /// bag. Same data as [`names`](Self::names) with arity attached.
    pub fn entries(&self) -> impl Iterator<Item = (&str, usize)> {
        self.table.iter().map(|(k, v)| (k.as_str(), v.params.len()))
    }

    /// Look up a function's parameter names and kind. Returns `None`
    /// if `name` isn't in the bag. Useful for pretty-printing or
    /// re-creating an `Expr::Func` outside the parser.
    pub fn get_info(&self, name: &str) -> Option<(&[String], &FuncKind)> {
        let f = self.table.get(name)?;
        Some((&f.params, &f.kind))
    }

    /// Build an `Expr::Func` by looking up `name` in this bag and
    /// pairing it with `args`. Returns `None` if `name` is not
    /// registered; returns `Some(Err(..))` if the arity disagrees.
    /// `None` means the name isn't in the bag -- callers that want
    /// built-ins as a fallback should route through
    /// [`parse::parse_with_functions`] or
    /// [`function_by_name`](crate::function_by_name).
    pub fn call(&self, name: &str, args: &[E]) -> Option<Result<E, String>> {
        let f = self.table.get(name)?;
        if args.len() != f.params.len() {
            return Some(Err(format!(
                "{} expects {} argument(s), got {}",
                name, f.params.len(), args.len()
            )));
        }
        let func = E::new(Expr::Func {
            name: name.to_string(),
            params: f.params.clone(),
            kind: f.kind.clone(),
            args: args.to_vec(),
        });
        Some(Ok(func))
    }
}

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

// --- Mixed ops: E with i64 (auto-simplify).
//
// The pure-E and E-with-f64 impls above already cover the common
// cases. These i64 impls let bare integer literals (`2 * x`,
// `x + 1`) work without an explicit `.0` suffix: Rust's type
// inference picks i64 when no concrete type is pinned, and
// integer literals with type annotations (`2i64 * x`) also flow
// through here. We convert to f64 at construction time to keep
// the expression tree representation uniform.

impl std::ops::Add<i64> for E {
    type Output = E;
    fn add(self, rhs: i64) -> E { E::new(Expr::Add(self, constant(rhs as f64))).simplify() }
}

impl std::ops::Add<E> for i64 {
    type Output = E;
    fn add(self, rhs: E) -> E { E::new(Expr::Add(constant(self as f64), rhs)).simplify() }
}

impl std::ops::Sub<i64> for E {
    type Output = E;
    fn sub(self, rhs: i64) -> E { E::new(Expr::Sub(self, constant(rhs as f64))).simplify() }
}

impl std::ops::Sub<E> for i64 {
    type Output = E;
    fn sub(self, rhs: E) -> E { E::new(Expr::Sub(constant(self as f64), rhs)).simplify() }
}

impl std::ops::Mul<i64> for E {
    type Output = E;
    fn mul(self, rhs: i64) -> E { E::new(Expr::Mul(self, constant(rhs as f64))).simplify() }
}

impl std::ops::Mul<E> for i64 {
    type Output = E;
    fn mul(self, rhs: E) -> E { E::new(Expr::Mul(constant(self as f64), rhs)).simplify() }
}

impl std::ops::Div<i64> for E {
    type Output = E;
    fn div(self, rhs: i64) -> E { E::new(Expr::Div(self, constant(rhs as f64))).simplify() }
}

impl std::ops::Div<E> for i64 {
    type Output = E;
    fn div(self, rhs: E) -> E { E::new(Expr::Div(constant(self as f64), rhs)).simplify() }
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
///     assert_eq!(format!("{}", square(x).diff(x)), "2 * x");
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
///     let (x, y) = symbols!(x, y);
///     assert_eq!(format!("{}", f(x, y).diff(x)), "1");
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
    if !(-PI..=PI).contains(&v) {
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

/// Caching / substitution barrier: `cached(x) = x` in both value and codegen
/// (the simplifier does not look inside, so the wrapped expression stays intact
/// as a unit, and codegen inlines it in parentheses). Unlike [`identity`] it is
/// STICKY under differentiation -- `diff` re-wraps it, `d(cached(g))/dx =
/// cached(dg/dx)` -- so the barrier survives into the derivative. That lets a
/// subexpression AND its derivative each be matched and substituted (e.g. a
/// composed rotation and its Jacobian, both replaced by per-entity precomputed
/// reads). Nested `cached(..cached()..)` is fine: substitute the outer wholesale.
pub fn cached(x: E) -> E {
    simple_func1("cached", |t| t)(x)
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
        atan2,
        |y, x| {
            // epsilon anchored to x so its precision follows the argument type
            // in type-inferred codegen (see `epsilon_for`).
            let e = epsilon_for(x.clone());
            let d = x.clone()*x.clone() + y.clone()*y.clone() + e.clone()*e;
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
        // Clamp x to [-1, 1] in the derivative too, so eval at |x| > 1
        // gives a finite value (1 / epsilon) instead of NaN. The body's
        // clamp keeps `asin` inside its domain; this one keeps the
        // derivative's `1 - x^2` non-negative.
        |x| {
            let xc = clamp(x, c(-1.0), c(1.0));
            // Anchor the epsilon guard to xc so codegen infers its precision
            // from the (concrete field-typed) argument -- f32::EPSILON in f32
            // code, f64::EPSILON in f64. The nullary epsilon() would fold to a
            // single-precision literal in type-inferred constraint code.
            let e = epsilon_for(xc.clone());
            [c(1.0) / sqrt(identity(c(1.0) - xc.clone()*xc) + e.clone()*e)]
        }
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
        // Same fix as `safe_asin`: clamp x in the derivative so
        // `1 - x^2` stays non-negative for any input.
        |x| {
            let xc = clamp(x, c(-1.0), c(1.0));
            let e = epsilon_for(xc.clone());
            [-c(1.0) / sqrt(identity(c(1.0) - xc.clone()*xc) + e.clone()*e)]
        }
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
        // Guard the derivative's `x` against negative inputs so
        // `sqrt(x + eps^2)` stays defined. `heaviside(x)` folds
        // negative x to zero; at eval time that gives `0.5 / eps`,
        // finite and large, instead of NaN.
        |x| {
            // epsilon anchored to x so its precision follows the argument type
            // in type-inferred codegen (see `epsilon_for`).
            let e = epsilon_for(x.clone());
            [c(0.5) / sqrt(identity(x.clone() * heaviside(x)) + e.clone()*e)]
        },
        |args| {
            let v = args[0];
            if v <= 0.0 { 0.0 } else { v.sqrt() }
        }
    )(x)
}

/// Fast approximate atan (max error < 1e-6 radians). Codegen emits
/// `arael::utils::fast_atan(x)`; eval runs the same polynomial (the f64
/// port of that implementation, kept in lockstep); the derivative is the
/// exact `1 / (1 + x^2)` -- within the approximation error of the
/// polynomial's own slope.
pub fn fast_atan(x: E) -> E {
    extern_func1("fast_atan", "arael::utils::fast_atan",
        |x| [c(1.0) / (c(1.0) + x.clone() * x)],
        |args| fast_atan_eval(args[0]))(x)
}

/// Fast approximate atan2 (max error < 1e-6 radians; does not handle
/// atan2(+-inf, +-inf)). Codegen emits `arael::utils::fast_atan2(y, x)`;
/// eval mirrors that implementation; the derivatives are the exact
/// rational forms.
pub fn fast_atan2(y: E, x: E) -> E {
    extern_func2("fast_atan2", "arael::utils::fast_atan2",
        |y, x| {
            let d = x.clone() * x.clone() + y.clone() * y.clone();
            [x / d.clone(), -y / d]
        },
        |args| fast_atan2_eval(args[0], args[1]))(y, x)
}

/// f64 port of `arael::utils::Float::fast_atan` (same folds, same
/// polynomial, same constants) for symbolic eval of [`fast_atan`].
fn fast_atan_eval(x: f64) -> f64 {
    const SIXTH_PI: f64 = 5.235_987_755_982_988e-1;
    const TAN_SIXTH_PI: f64 = 5.773_502_691_896_257e-1;
    const TAN_TWELFTH_PI: f64 = 2.679_491_924_311_227e-1;
    const C1: f64 = 1.6867629106;
    const C2: f64 = 0.4378497304;
    const C3: f64 = 1.6867633134;
    let negative = x < 0.0;
    let mut x = x.abs();
    let inverted = x > 1.0;
    if inverted { x = x.recip(); }
    let mut y = if x > TAN_TWELFTH_PI {
        let x = (x - TAN_SIXTH_PI) / (1.0 + TAN_SIXTH_PI * x);
        let x2 = x * x;
        x * (C1 + x2 * C2) / (C3 + x2) + SIXTH_PI
    } else {
        let x2 = x * x;
        x * (C1 + x2 * C2) / (C3 + x2)
    };
    if inverted { y = std::f64::consts::FRAC_PI_2 - y; }
    if negative { -y } else { y }
}

/// f64 port of `arael::utils::fast_atan2` for symbolic eval of
/// [`fast_atan2`].
fn fast_atan2_eval(y: f64, x: f64) -> f64 {
    if x > 0.0 {
        fast_atan_eval(y / x)
    } else if x == 0.0 {
        if y == 0.0 { 0.0 } else { std::f64::consts::FRAC_PI_2.copysign(y) }
    } else if y >= 0.0 {
        fast_atan_eval(y / x) + std::f64::consts::PI
    } else {
        fast_atan_eval(y / x) - std::f64::consts::PI
    }
}

// Re-export linalg types
pub use linalg::{SymVec, SymMat, jacobian};
pub use parse::{parse, parse_with_functions, ParseError};
pub use geo::{vect2sym, vect3sym, matrix2sym, matrix3sym, quaternsym};
pub use cse::cse;
pub use arael_sym_macros::sym;

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // --- assorted wrong-output fixes (verified red before each fix) ---

    #[test]
    fn substitute_preserves_log10() {
        // substitute() rewrote Log10(a) into ln(a)/ln(10) -- a copy-paste
        // structural rewrite that changed the node type and emitted code.
        let f = log10(symbol("x"));
        let g = f.substitute(&[(symbol("q"), symbol("r"))]);
        assert_eq!(format!("{}", g), "log10(x)");
    }

    #[test]
    fn replace_function_rewrites_builtins_and_funcs_by_name() {
        // Built-in node by conventional name, recursing into arguments.
        let f = sin(atan2(symbol("a"), cos(symbol("b")))) + atan(symbol("a"));
        let g = f.replace_function("atan2", &|args| safe_atan2(args[0].clone(), args[1].clone()));
        assert_eq!(format!("{}", g), "sin(safe_atan2(a, cos(b))) + atan(a)");
        // Untouched name: expression reconstructs unchanged.
        let h = f.replace_function("sinh", &|args| cosh(args[0].clone()));
        assert_eq!(format!("{}", f), format!("{}", h));
        // Func nodes match by their name.
        let k = safe_sqrt(symbol("x")).replace_function("safe_sqrt", &|args| sqrt(args[0].clone()));
        assert_eq!(format!("{}", k), "sqrt(x)");
    }

    #[test]
    fn fast_atan_matches_exact_within_tolerance() {
        // The registered fast_atan/fast_atan2 eval within the documented
        // 1e-6 radian bound, across octants and both atan2 half-planes.
        for i in 0..200 {
            let x = -10.0 + i as f64 * 0.1003;
            let vars: HashMap<&str, f64> = [("x", x)].into();
            let fa = fast_atan(symbol("x")).eval(&vars).unwrap();
            assert!((fa - x.atan()).abs() < 1e-6, "fast_atan({}) = {} vs {}", x, fa, x.atan());
        }
        for (y, x) in [(0.3, 1.7), (2.1, -0.4), (-1.3, -2.2), (-0.7, 0.9), (1.0, 0.0), (-1.0, 0.0)] {
            let vars: HashMap<&str, f64> = [("y", y), ("x", x)].into();
            let fa = fast_atan2(symbol("y"), symbol("x")).eval(&vars).unwrap();
            assert!((fa - y.atan2(x)).abs() < 1e-6,
                "fast_atan2({}, {}) = {} vs {}", y, x, fa, y.atan2(x));
        }
    }

    #[test]
    fn fast_atan_codegen_and_derivative() {
        let f = fast_atan2(symbol("y"), symbol("x"));
        assert_eq!(f.to_rust(""), "arael::utils::fast_atan2(y, x)");
        // Exact rational derivative, no atan anywhere in it.
        let d = f.diff("y");
        assert!(!format!("{}", d).contains("atan"), "d/dy = {}", d);
        let vars: HashMap<&str, f64> = [("y", 0.5), ("x", 2.0)].into();
        let expected = 2.0 / (2.0 * 2.0 + 0.5 * 0.5);
        assert!((d.eval(&vars).unwrap() - expected).abs() < 1e-12);
    }

    #[test]
    fn nested_pow_display_reparses_to_same_value() {
        // Pow(Pow(x,2),3) printed as x^2^3, which re-parses
        // right-associatively as x^(2^3) = x^8, not (x^2)^3 = x^6.
        let f = pow(pow(symbol("x"), constant(2.0)), constant(3.0));
        let printed = format!("{}", f);
        let reparsed = crate::parse::parse(&printed).unwrap();
        let mut vars = HashMap::new();
        vars.insert("x", 2.0);
        assert_eq!(reparsed.eval(&vars).unwrap(), f.eval(&vars).unwrap(),
            "display `{}` changes value on reparse", printed);
    }

    #[test]
    fn nonfinite_constants_emit_valid_rust() {
        // Constants folded to inf/NaN emitted `inf_f64` / `NaN_f64` --
        // not valid Rust tokens.
        let f = symbol("x") + constant(f64::INFINITY);
        let code = f.to_rust("f64");
        assert!(!code.contains("inf_"), "bad literal in `{}`", code);
        assert!(code.contains("INFINITY"), "expected INFINITY in `{}`", code);
        let g = symbol("x") + constant(f64::NAN);
        let code = g.to_rust("f32");
        assert!(!code.contains("NaN_"), "bad literal in `{}`", code);
        // Type-inferred context must still emit something type-correct.
        let code = f.to_rust("");
        assert!(!code.contains("inf"), "bad literal in `{}`", code);
    }

    #[test]
    fn division_by_zero_constant_is_not_simplified() {
        // simplify's fraction flatten computed coeff = ca/cb unguarded,
        // baking an inf coefficient into the tree for x / 0.
        let f = (symbol("x") * constant(2.0)) / (symbol("y") * constant(0.0));
        let g = f.simplify();
        let mut vars = HashMap::new();
        vars.insert("x", 1.0);
        vars.insert("y", 1.0);
        // Division by zero happens at eval time (IEEE inf), and the tree
        // must not contain a folded non-finite coefficient.
        assert!(g.eval(&vars).unwrap().is_infinite());
        assert!(!format!("{}", g).contains("inf"), "folded inf into `{}`", g);
    }

    #[test]
    fn zero_pow_symbolic_exponent_stays_symbolic() {
        // 0^b -> 0 fired for symbolic b: wrong for b == 0 (0^0 = 1)
        // and for b < 0 (0^b = inf).
        let f = pow(constant(0.0), symbol("b"));
        let mut vars = HashMap::new();
        vars.insert("b", 0.0);
        assert_eq!(f.simplify().eval(&vars).unwrap(), 1.0, "0^0 must be 1");
        vars.insert("b", -1.0);
        assert!(f.simplify().eval(&vars).unwrap().is_infinite(), "0^-1 must be inf");
    }

    #[test]
    fn heaviside_nan_matches_runtime_semantics() {
        // eval/simplify used `v < 0` (NaN -> 1); the runtime
        // utils::heaviside uses `v >= 0` (NaN -> 0). Interpreted and
        // compiled constraint code disagreed on NaN input.
        let f = heaviside(symbol("x"));
        let mut vars = HashMap::new();
        vars.insert("x", f64::NAN);
        assert_eq!(f.eval(&vars).unwrap(), 0.0, "eval heaviside(NaN)");
        let g = heaviside(constant(f64::NAN)).simplify();
        assert_eq!(format!("{}", g), "0", "simplify heaviside(NaN)");
    }

    #[test]
    fn parse_pi_e_are_named_constants() {
        // parse folded pi/e to numeric literals, so printed output lost
        // the name and codegen emitted decimal literals -- while sym! and
        // the documented behavior keep them as named constants.
        let f = crate::parse::parse("pi * x").unwrap();
        assert!(format!("{}", f).contains("pi"), "pi lost in `{}`", f);
        // The named constant folds exactly: sin(pi) is 0, not sin(3.14...)
        // = 1.2e-16 as it was with a numeric literal.
        let g = crate::parse::parse("sin(pi)").unwrap();
        assert_eq!(g.simplify().eval(&HashMap::new()).unwrap(), 0.0);
        let h = crate::parse::parse("e * x").unwrap();
        assert!(format!("{}", h).contains('e'), "e lost in `{}`", h);
    }

    #[test]
    fn sym_macro_does_not_clone_method_receivers() {
        // The auto-clone visitor wrapped tracked method-call receivers,
        // so `parts.push(..)` mutated a temporary clone -- silent loss.
        sym! {
            let x = symbol("x");
            let mut parts: std::vec::Vec<E> = std::vec::Vec::new();
            parts.push(x * 2.0);
            parts.push(x + 1.0);
            assert_eq!(parts.len(), 2, "push mutated a clone, not the vec");
        }
    }

    #[test]
    fn sym_macro_tracks_typed_let_bindings() {
        // `let f: E = ...` (a typed pattern) was not tracked, so reusing
        // f moved it twice and failed to compile inside sym!.
        sym! {
            let x = symbol("x");
            let f: E = x * x;
            let g = f + f;
            let mut vars = HashMap::new();
            vars.insert("x", 3.0);
            assert_eq!(g.eval(&vars).unwrap(), 18.0);
        }
    }

    #[test]
    fn func_free_vars_include_captured_symbols() {
        // A function body may capture symbols beyond its params; eval
        // resolves them from the outer vars map, but free_vars/subs did
        // not walk the body -- callers could not know what to supply,
        // and subs silently skipped the capture.
        sym! {
            let w = symbol("w");
            let scaled = simple_func1("scaled", |t| t * w);
            let x = symbol("x");
            let f = scaled(x);
            let fv = f.free_vars();
            assert!(fv.contains("w"), "captured symbol missing from free_vars: {:?}", fv);
            assert!(fv.contains("x"));
            // subs must reach into the body for captured symbols
            let g = f.subs("w", &constant(3.0));
            let mut vars = HashMap::new();
            vars.insert("x", 2.0);
            assert_eq!(g.eval(&vars).unwrap(), 6.0);
        }
    }

    #[test]
    fn tiny_coefficients_survive_simplification() {
        // build_sum used to prune "zero" terms with `c.abs() > f64::EPSILON`,
        // silently deleting legitimate tiny terms (small residual weights,
        // regularization constants) from generated code. Like-term
        // coefficients are summed exactly, so true cancellation already
        // yields 0.0 -- the prune must compare against exact zero.
        let f = constant(1e-18) * symbol("x") + symbol("y");
        let mut vars = HashMap::new();
        vars.insert("x", 1.0);
        vars.insert("y", 0.0);
        assert_eq!(f.eval(&vars).unwrap(), 1e-18, "1e-18*x term was dropped: {}", f);

        // Exact cancellation still prunes to a clean tree.
        let g = symbol("x") - symbol("x") + symbol("y");
        assert_eq!(format!("{}", g), "y");
    }

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
    fn cached_acts_as_identity() {
        sym! {
            let x = symbol("x");
            // Value/codegen is an identity, but STICKY under differentiation:
            // d(cached(g))/dx = cached(dg/dx), so the barrier survives.
            let vars = HashMap::from([("x", 5.0)]);
            assert_eq!(cached(x.clone()).eval(&vars).unwrap(), 5.0);
            assert_eq!(format!("{}", cached(x.clone()).diff("x")), "cached(1)");
            // Barrier like identity: the wrapped subtraction stays intact and
            // codegen inlines it in parentheses (evaluation order preserved).
            assert_eq!(cached(x.clone() - c(1.0)).to_rust(""), "(x - 1.0)");
            // Registered builtin, so it parses and is usable in constraints.
            assert!(crate::parse("cached(x)").is_ok());
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
    fn epsilon_for_codegen_eval_diff() {
        sym! {
            let x = symbol("x");
            let e = epsilon_for(x.clone());
            // Anchored call in every codegen context, so the precision follows
            // the anchor -- unlike nullary epsilon(), which folds to a single-
            // precision literal in a type-inferred context.
            assert_eq!(e.to_rust(""), "arael::utils::epsilon_for(x)");
            assert_eq!(e.to_rust("f32"), "arael::utils::epsilon_for(x)");
            // Symbolic eval is f64::EPSILON regardless of the anchor value.
            let vars = HashMap::from([("x", 3.0)]);
            assert_eq!(e.eval(&vars).unwrap(), f64::EPSILON);
            // Constant w.r.t. the anchor: derivative is 0.
            assert_eq!(format!("{}", epsilon_for(x).diff("x")), "0");
        }
    }

    #[test]
    fn safe_asin_guard_uses_anchored_epsilon_not_folded_literal() {
        sym! {
            let x = symbol("x");
            // The derivative guard must reach type-inferred codegen as the
            // anchored epsilon_for call (so f32 code gets f32::EPSILON), never
            // the folded f64 epsilon literal.
            let code = safe_asin(x).diff("x").to_rust("");
            assert!(code.contains("arael::utils::epsilon_for("),
                "guard should use epsilon_for, got: {}", code);
            assert!(!code.contains("4.930380657631324e-32"),
                "guard must not fold to the f64 epsilon literal, got: {}", code);
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

    /// Regression: the derivative of `safe_asin` / `safe_acos` /
    /// `safe_sqrt` must stay finite even for inputs well outside the
    /// safe domain. Previously each derivative formula used the raw
    /// `x` unclamped, so for `|x| > 1` (asin/acos) or `x < 0` (sqrt)
    /// the inner `sqrt(1 - x^2 + eps^2)` / `sqrt(x + eps^2)` evaluated
    /// a negative operand and produced NaN.
    #[test]
    fn safe_derivs_finite_outside_domain() {
        sym! {
            let x = symbol("x");
            let d_asin = safe_asin(x).diff("x");
            let d_acos = safe_acos(x).diff("x");
            let d_sqrt = safe_sqrt(x).diff("x");
            for v in [-5.0_f64, -1.5, 1.5, 5.0] {
                let vars = HashMap::from([("x", v)]);
                let a = d_asin.eval(&vars).unwrap();
                let c = d_acos.eval(&vars).unwrap();
                assert!(a.is_finite(), "safe_asin'({}) should be finite, got {}", v, a);
                assert!(c.is_finite(), "safe_acos'({}) should be finite, got {}", v, c);
            }
            for v in [-5.0_f64, -1.0, -1e-12, 0.0] {
                let vars = HashMap::from([("x", v)]);
                let s = d_sqrt.eval(&vars).unwrap();
                assert!(s.is_finite(), "safe_sqrt'({}) should be finite, got {}", v, s);
            }
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

    #[test]
    fn branch_eval() {
        sym! {
            let x = symbol("x");
            let f = branch(x, c(10.0), c(-10.0));
            assert_eq!(f.eval(&HashMap::from([("x", 0.5)])).unwrap(), 10.0);
            assert_eq!(f.eval(&HashMap::from([("x", 0.0)])).unwrap(), 10.0);   // q == 0 selects a
            assert_eq!(f.eval(&HashMap::from([("x", -0.1)])).unwrap(), -10.0);
        }
    }

    #[test]
    fn branch_eval_only_taken_side() {
        sym! {
            let x = symbol("x");
            // The untaken side (ln(-1) = NaN) must not be evaluated.
            let f = branch(x, c(1.0), ln(c(-1.0)));
            assert_eq!(f.eval(&HashMap::from([("x", 1.0)])).unwrap(), 1.0);
        }
    }

    #[test]
    fn branch_diff_selects_side() {
        sym! {
            let x = symbol("x");
            let q = symbol("q");
            // d/dx branch(q, x^2, 5) = branch(q, 2x, 0): selects the taken side's slope.
            let d = branch(q, x * x, c(5.0)).diff("x");
            assert_eq!(d.eval(&HashMap::from([("q", 1.0), ("x", 2.0)])).unwrap(), 4.0);
            assert_eq!(d.eval(&HashMap::from([("q", -1.0), ("x", 2.0)])).unwrap(), 0.0);
        }
    }

    #[test]
    fn branch_display() {
        sym! {
            let x = symbol("x");
            assert_eq!(format!("{}", branch(x, c(1.0), c(-1.0))), "branch(x, 1, -1)");
        }
    }

    #[test]
    fn branch_simplify_constant_condition() {
        sym! {
            let a = symbol("a");
            let b = symbol("b");
            assert_eq!(format!("{}", branch(c(2.0), a.clone(), b.clone()).simplify()), "a");
            assert_eq!(format!("{}", branch(c(0.0), a.clone(), b.clone()).simplify()), "a");
            assert_eq!(format!("{}", branch(c(-1.0), a, b).simplify()), "b");
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

