//! Code for GPU kernels written with CubeCL.
//!
//! [CubeCL](https://github.com/tracel-ai/cubecl) is a Rust crate for GPU
//! kernels: a function marked `#[cube]` is compiled, at its first launch,
//! for CUDA, HIP or another of CubeCL's runtimes. Its body may use only the
//! part of Rust that CubeCL understands. [`E::to_cubecl`] prints an
//! expression as Rust that such a body accepts, as [`E::to_rust`] prints it
//! for ordinary Rust; [`Intermediate::to_cubecl`] prints one statement of a
//! CSE result, and [`cubecl_body`] runs the CSE and prints in one call.
//! arael's device assembly generates its kernels' bodies this way.
//!
//! ```
//! use arael_sym::{sin, symbols};
//! let (x, y) = symbols!(x, y);
//! let e = sin(x) * y + 2.0;
//! assert_eq!(e.to_cubecl("f64").unwrap(), "((y * f64::sin(x)) + 2.0f64)");
//! ```
//!
//! How the printed code differs from [`E::to_rust`]'s:
//!
//! | | `to_rust` | `to_cubecl` |
//! |---|---|---|
//! | scalar | any, or inferred (`""`) | `"f32"` or `"f64"`, every literal suffixed and rounded to it |
//! | functions | methods, `x.sin()` | associated functions, `f64::sin(x)` |
//! | whole powers | `x.powi(3)` | products up to 8, `(x * x * x)`; others `powf` |
//! | `branch` printed directly | `if`: the taken side only | `select`: both sides evaluated |
//! | `branch` after CSE | `let (a, ..) = if q >= 0.0 { .. } else { .. };` | names declared, then an `if` assigning them: the taken side only |
//! | `clamp`, `heaviside` | `if` | `select` |
//! | `with_wide` | `arael::utils::Wide` and `narrow` | computed in `f64`, cast back |
//!
//! A `branch` printed on its own is CubeCL's `select`, which evaluates both
//! sides and returns the taken one: a NaN or an infinity on the other side
//! does not reach the result, but its work is done.
//!
//! ```
//! use arael_sym::{branch, sqrt, symbols};
//! let (x,) = symbols!(x);
//! // sqrt(x) is computed for a negative x too (NaN), and discarded
//! let e = branch(x.clone(), sqrt(x), 0.0);
//! assert_eq!(e.to_cubecl("f32").unwrap(), "select(x >= 0.0f32, f32::sqrt(x), 0.0f32)");
//! ```
//!
//! After CSE ([`cubecl_body`], or [`cse_scoped_wide`](crate::cse_scoped_wide)
//! and [`Intermediate::to_cubecl`]) every branch is an `if` that runs only the
//! taken side, the names it binds declared before it:
//!
//! ```
//! use arael_sym::{branch, cubecl_body, sqrt, symbols};
//! let (x,) = symbols!(x);
//! let body = cubecl_body(&[branch(x.clone(), sqrt(x), 0.0)], "f32").unwrap();
//! assert_eq!(body.stmts, "let mut __m0 = 0.0f32;\nif x >= 0.0f32 {\n__m0 = f32::sqrt(x);\n} else {\n__m0 = 0.0f32;\n}\n");
//! assert_eq!(body.outputs, ["__m0"]);
//! ```
//!
//! Outputs under one condition share one `if`, and work only the taken side
//! needs is computed inside it: a guarded residual and its derivative.
//!
//! ```
//! use arael_sym::{branch, cubecl_body, sqrt, symbols};
//! let (q, x, y) = symbols!(q, x, y);
//! let r = branch(q, sqrt(x) * y, 0.0);
//! let body = cubecl_body(&[r.clone(), r.diff("x")], "f64").unwrap();
//! assert_eq!(body.stmts, "\
//! let mut __m0 = 0.0f64;
//! let mut __m1 = 0.0f64;
//! if q >= 0.0f64 {
//! let __x2 = f64::sqrt(x);
//! __m0 = (y * __x2);
//! __m1 = ((0.5f64 * y) / __x2);
//! } else {
//! __m0 = 0.0f64;
//! __m1 = 0.0f64;
//! }
//! ");
//! assert_eq!(body.outputs, ["__m0", "__m1"]);
//! ```
//!
//! What has no CubeCL form is an error naming it: a `select` on an index,
//! an external function (`#[arael::function]`, `fast_atan` and the other
//! numeric helpers of `arael::utils`), and a wide scalar other than `f64`.

use crate::cse::{Arm, Intermediate, Switch};
use crate::{E, Expr, FuncKind};

/// Why an expression has no CubeCL form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CubeclError(pub String);

impl std::fmt::Display for CubeclError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CubeclError {}

/// What the printer prints against: the scalar of the code being printed
/// (inside a `with_wide` block, f64), the kernel's own scalar, and the
/// symbol map.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    ft: &'a str,
    base: &'a str,
    map: &'a dyn Fn(&str) -> Option<String>,
}

impl<'a> Ctx<'a> {
    fn new(scalar: &'a str, map: &'a dyn Fn(&str) -> Option<String>) -> Result<Self, CubeclError> {
        if scalar != "f32" && scalar != "f64" {
            return Err(CubeclError(format!("a CubeCL scalar is `f32` or `f64`, not `{scalar}`")));
        }
        Ok(Ctx { ft: scalar, base: scalar, map })
    }

    fn wide(&self) -> Ctx<'a> {
        Ctx { ft: "f64", ..*self }
    }
}

type Res = Result<(), CubeclError>;

/// A CSE temporary's name (`__x3`, `__w1`, `__m2`): printed as it is,
/// never mapped.
fn is_temp(s: &str) -> bool {
    let rest = s.strip_prefix("__x").or_else(|| s.strip_prefix("__w")).or_else(|| s.strip_prefix("__m"));
    rest.is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
}

fn verbatim(s: &str) -> Option<String> {
    Some(s.to_string())
}

impl E {
    /// The expression as Rust for the body of a CubeCL `#[cube]` function,
    /// in `scalar` (`"f32"` or `"f64"`), every symbol by its own name; see
    /// the [module](crate::cubecl) for how it differs from
    /// [`to_rust`](Self::to_rust).
    ///
    /// ```
    /// use arael_sym::{pow, symbols};
    /// let (x, y) = symbols!(x, y);
    /// assert_eq!(pow(x - y, 2.0).to_cubecl("f32").unwrap(), "((x - y) * (x - y))");
    /// ```
    pub fn to_cubecl(&self, scalar: &str) -> Result<String, CubeclError> {
        self.to_cubecl_with(scalar, &verbatim)
    }

    /// [`to_cubecl`](Self::to_cubecl) with every symbol `s` printed as
    /// `map(s)`, the kernel value it reads (`None` is an error); CSE
    /// temporaries (`__x3`, ...) are printed as they are.
    ///
    /// ```
    /// use arael_sym::symbols;
    /// let (x, y) = symbols!(x, y);
    /// let map = |s: &str| Some(format!("v[{}]", if s == "x" { 0 } else { 1 }));
    /// assert_eq!((x * y).to_cubecl_with("f64", &map).unwrap(), "(v[0] * v[1])");
    /// ```
    pub fn to_cubecl_with(&self, scalar: &str, map: &dyn Fn(&str) -> Option<String>) -> Result<String, CubeclError> {
        let ctx = Ctx::new(scalar, map)?;
        let mut buf = String::new();
        write_expr(&mut buf, self, &ctx)?;
        Ok(buf)
    }
}

impl Intermediate {
    /// The statement as Rust for the body of a CubeCL `#[cube]` function,
    /// in `scalar`, every symbol by its own name: one line per binding,
    /// every name it binds in scope after it, as with
    /// [`to_rust`](Self::to_rust).
    pub fn to_cubecl(&self, scalar: &str) -> Result<String, CubeclError> {
        self.to_cubecl_with(scalar, &verbatim)
    }

    /// [`to_cubecl`](Self::to_cubecl) with symbols mapped, as
    /// [`E::to_cubecl_with`].
    pub fn to_cubecl_with(&self, scalar: &str, map: &dyn Fn(&str) -> Option<String>) -> Result<String, CubeclError> {
        let ctx = Ctx::new(scalar, map)?;
        let mut buf = String::new();
        write_stmt(&mut buf, self, &ctx)?;
        Ok(buf)
    }
}

/// [`cubecl_body`]'s result: the statements, then one expression per output,
/// reading the names the statements bind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CubeclBody {
    pub stmts: String,
    pub outputs: Vec<String>,
}

/// `exprs` through [`cse_scoped_wide`](crate::cse_scoped_wide) (wide scalar
/// f64) and printed as CubeCL in `scalar`, every symbol by its own name: the
/// shared work once, then each output.
///
/// ```
/// use arael_sym::{cubecl_body, sqrt, symbols};
/// let (x, y) = symbols!(x, y);
/// // a range residual and its two partials share the square root and its
/// // reciprocal
/// let r = sqrt(x.clone() * x.clone() + y.clone() * y.clone()) - 1.0;
/// let body = cubecl_body(&[r.clone(), r.diff("x"), r.diff("y")], "f32").unwrap();
/// assert_eq!(body.stmts, "let __x0 = f32::sqrt(((x * x) + (y * y)));\nlet __x1 = (1.0f32 / __x0);\n");
/// assert_eq!(body.outputs, ["(__x0 - 1.0f32)", "(x * __x1)", "(y * __x1)"]);
/// ```
pub fn cubecl_body(exprs: &[E], scalar: &str) -> Result<CubeclBody, CubeclError> {
    cubecl_body_with(exprs, scalar, &verbatim)
}

/// [`cubecl_body`] with symbols mapped, as [`E::to_cubecl_with`].
pub fn cubecl_body_with(exprs: &[E], scalar: &str, map: &dyn Fn(&str) -> Option<String>) -> Result<CubeclBody, CubeclError> {
    let ctx = Ctx::new(scalar, map)?;
    let (inters, outs) = crate::cse_scoped_wide(exprs, "f64");
    let mut stmts = String::new();
    for it in &inters {
        write_stmt(&mut stmts, it, &ctx)?;
    }
    let outputs = outs
        .iter()
        .map(|o| {
            let mut buf = String::new();
            write_expr(&mut buf, o, &ctx)?;
            Ok(buf)
        })
        .collect::<Result<Vec<String>, CubeclError>>()?;
    Ok(CubeclBody { stmts, outputs })
}

/// A literal of `v` in `ft`.
fn literal(buf: &mut String, v: f64, ft: &str) {
    if v.is_nan() {
        buf.push_str(&format!("(0.0{ft} / 0.0{ft})"));
    } else if v.is_infinite() {
        buf.push_str(&format!("({}1.0{ft} / 0.0{ft})", if v < 0.0 { "-" } else { "" }));
    } else {
        let digits = if ft == "f32" { format!("{:?}", v as f32) } else { format!("{v:?}") };
        if v < 0.0 || digits.starts_with('-') {
            buf.push_str(&format!("({digits}{ft})"));
        } else {
            buf.push_str(&format!("{digits}{ft}"));
        }
    }
}

fn call(buf: &mut String, ft: &str, f: &str, args: &[&E], ctx: &Ctx) -> Res {
    buf.push_str(ft);
    buf.push_str("::");
    buf.push_str(f);
    buf.push('(');
    for (i, a) in args.iter().enumerate() {
        if i > 0 {
            buf.push_str(", ");
        }
        write_expr(buf, a, ctx)?;
    }
    buf.push(')');
    Ok(())
}

fn binary(buf: &mut String, a: &E, op: &str, b: &E, ctx: &Ctx) -> Res {
    buf.push('(');
    write_expr(buf, a, ctx)?;
    buf.push_str(op);
    write_expr(buf, b, ctx)?;
    buf.push(')');
    Ok(())
}

/// `select(q >= 0, a, b)`: both sides are evaluated, and a NaN `q` takes
/// `b`, as the Rust form's `if` does.
fn select_ge0(buf: &mut String, q: &E, a: &dyn Fn(&mut String) -> Res, b: &dyn Fn(&mut String) -> Res, ctx: &Ctx) -> Res {
    buf.push_str("select(");
    write_expr(buf, q, ctx)?;
    buf.push_str(&format!(" >= 0.0{}, ", ctx.ft));
    a(buf)?;
    buf.push_str(", ");
    b(buf)?;
    buf.push(')');
    Ok(())
}

fn write_expr(buf: &mut String, e: &E, ctx: &Ctx) -> Res {
    let ft = ctx.ft;
    match e.as_ref() {
        Expr::Sym(name) => {
            if is_temp(name) {
                buf.push_str(name);
            } else {
                match (ctx.map)(name) {
                    Some(s) => buf.push_str(&s),
                    None => return Err(CubeclError(format!("the kernel has no value for `{name}`"))),
                }
            }
        }
        Expr::Const(v) => literal(buf, *v, ft),
        Expr::NamedConst { value, .. } => literal(buf, *value, ft),
        Expr::Neg(a) => {
            buf.push_str("(-");
            write_expr(buf, a, ctx)?;
            buf.push(')');
        }
        Expr::Add(a, b) => binary(buf, a, " + ", b, ctx)?,
        Expr::Sub(a, b) => binary(buf, a, " - ", b, ctx)?,
        Expr::Mul(a, b) => binary(buf, a, " * ", b, ctx)?,
        Expr::Div(a, b) => binary(buf, a, " / ", b, ctx)?,
        Expr::Pow(b, ex) => write_pow(buf, b, ex, ctx)?,
        Expr::Sin(a) => call(buf, ft, "sin", &[a], ctx)?,
        Expr::Cos(a) => call(buf, ft, "cos", &[a], ctx)?,
        Expr::Tan(a) => call(buf, ft, "tan", &[a], ctx)?,
        Expr::Asin(a) => call(buf, ft, "asin", &[a], ctx)?,
        Expr::Acos(a) => call(buf, ft, "acos", &[a], ctx)?,
        Expr::Atan(a) => call(buf, ft, "atan", &[a], ctx)?,
        Expr::Atan2(y, x) => call(buf, ft, "atan2", &[y, x], ctx)?,
        Expr::Sinh(a) => call(buf, ft, "sinh", &[a], ctx)?,
        Expr::Cosh(a) => call(buf, ft, "cosh", &[a], ctx)?,
        Expr::Tanh(a) => call(buf, ft, "tanh", &[a], ctx)?,
        Expr::Exp(a) => call(buf, ft, "exp", &[a], ctx)?,
        Expr::Ln(a) => call(buf, ft, "ln", &[a], ctx)?,
        Expr::Log2(a) | Expr::Log10(a) => {
            // ln(a) / ln(base): CubeCL has the natural logarithm only
            let inv_ln = if matches!(e.as_ref(), Expr::Log2(_)) { std::f64::consts::LOG2_E } else { std::f64::consts::LOG10_E };
            buf.push('(');
            call(buf, ft, "ln", &[a], ctx)?;
            buf.push_str(" * ");
            literal(buf, inv_ln, ft);
            buf.push(')');
        }
        Expr::Sqrt(a) => call(buf, ft, "sqrt", &[a], ctx)?,
        Expr::Abs(a) => call(buf, ft, "abs", &[a], ctx)?,
        Expr::Heaviside(a) => {
            let one = |b: &mut String| {
                literal(b, 1.0, ft);
                Ok(())
            };
            let zero = |b: &mut String| {
                literal(b, 0.0, ft);
                Ok(())
            };
            select_ge0(buf, a, &one, &zero, ctx)?
        }
        Expr::Clamp(v, lo, hi) => {
            // v < lo ? lo : (v > hi ? hi : v), as arael's `clamp`
            buf.push_str("select(");
            binary(buf, v, " < ", lo, ctx)?;
            buf.push_str(", ");
            write_expr(buf, lo, ctx)?;
            buf.push_str(", select(");
            binary(buf, v, " > ", hi, ctx)?;
            buf.push_str(", ");
            write_expr(buf, hi, ctx)?;
            buf.push_str(", ");
            write_expr(buf, v, ctx)?;
            buf.push_str("))");
        }
        Expr::Branch(q, a, b) => select_ge0(buf, q, &|s| write_expr(s, a, ctx), &|s| write_expr(s, b, ctx), ctx)?,
        Expr::Select { .. } => return Err(CubeclError("a `select` on an index has no CubeCL form yet".into())),
        Expr::Func { name, params, kind, args } => {
            if name == "with_wide" && args.len() == 1 {
                // a scope printed inline: the inside in f64 over widened
                // leaves, narrowed back to the surrounding scalar
                let inner = crate::widen_leaves(&crate::strip_with_wide(&args[0]));
                narrow(buf, &inner, &ctx.wide(), ft)?;
            } else if name == "widen" && args.len() == 1 {
                // a leaf of the body, in the kernel's scalar, read into a
                // block of the wide scalar
                let leaf = Ctx { ft: ctx.base, ..*ctx };
                if ctx.base == "f64" {
                    write_expr(buf, &args[0], &leaf)?;
                } else {
                    buf.push_str("f64::cast_from(");
                    write_expr(buf, &args[0], &leaf)?;
                    buf.push(')');
                }
            } else if name == "wide" && args.len() == 1 {
                write_expr(buf, &args[0], ctx)?;
            } else if let Some(body) = kind.body() {
                write_expr(buf, &crate::expand_func(params, body, args), ctx)?;
            } else {
                let what = match kind {
                    FuncKind::Extern { call_path, .. } | FuncKind::ExternNumericDerivs { call_path, .. } => call_path.clone(),
                    _ => name.clone(),
                };
                return Err(CubeclError(format!("`{what}` has no CubeCL form")));
            }
        }
    }
    Ok(())
}

/// `v` printed in the wide scalar, then cast to `to` when that differs.
fn narrow(buf: &mut String, v: &E, wide: &Ctx, to: &str) -> Res {
    if to == wide.ft {
        return write_expr(buf, v, wide);
    }
    buf.push_str(to);
    buf.push_str("::cast_from(");
    write_expr(buf, v, wide)?;
    buf.push(')');
    Ok(())
}

fn write_pow(buf: &mut String, b: &E, ex: &E, ctx: &Ctx) -> Res {
    let ft = ctx.ft;
    if let Expr::Const(n) = ex.as_ref() {
        let n = *n;
        if n == 0.5 {
            return call(buf, ft, "sqrt", &[b], ctx);
        }
        if n == -0.5 {
            buf.push('(');
            literal(buf, 1.0, ft);
            buf.push_str(" / ");
            call(buf, ft, "sqrt", &[b], ctx)?;
            buf.push(')');
            return Ok(());
        }
        if n.fract() == 0.0 && n.abs() <= 8.0 {
            // a product, exact for a negative base as Rust's powi is
            let k = n.abs() as usize;
            if k == 0 {
                literal(buf, 1.0, ft);
                return Ok(());
            }
            buf.push('(');
            if n < 0.0 {
                literal(buf, 1.0, ft);
                buf.push_str(" / (");
            }
            for i in 0..k {
                if i > 0 {
                    buf.push_str(" * ");
                }
                write_expr(buf, b, ctx)?;
            }
            if n < 0.0 {
                buf.push(')');
            }
            buf.push(')');
            return Ok(());
        }
    }
    call(buf, ft, "powf", &[b, ex], ctx)
}

fn names_zeroed(buf: &mut String, names: &[String], ft: &str) {
    for n in names {
        buf.push_str(&format!("let mut {n} = 0.0{ft};\n"));
    }
}

/// An arm's statements, then its values assigned to the match's names.
fn write_arm(buf: &mut String, names: &[String], arm: &Arm, ctx: &Ctx) -> Res {
    for it in &arm.inters {
        write_stmt(buf, it, ctx)?;
    }
    for (n, v) in names.iter().zip(&arm.values) {
        buf.push_str(n);
        buf.push_str(" = ");
        write_expr(buf, v, ctx)?;
        buf.push_str(";\n");
    }
    Ok(())
}

fn write_stmt(buf: &mut String, it: &Intermediate, ctx: &Ctx) -> Res {
    match it {
        Intermediate::Let { name, expr } => {
            buf.push_str("let ");
            buf.push_str(name);
            buf.push_str(" = ");
            write_expr(buf, expr, ctx)?;
            buf.push_str(";\n");
        }
        Intermediate::Match { switch: Switch::Sign(q), names, arms, default } => {
            // the names declared before, assigned in either side: only the
            // taken side's statements run here, and a NaN `q` takes the
            // else side, as in the Rust form
            names_zeroed(buf, names, ctx.ft);
            buf.push_str("if ");
            write_expr(buf, q, ctx)?;
            buf.push_str(&format!(" >= 0.0{} {{\n", ctx.ft));
            write_arm(buf, names, &arms[0], ctx)?;
            buf.push_str("} else {\n");
            write_arm(buf, names, default.as_ref().expect("a fused branch has an else side"), ctx)?;
            buf.push_str("}\n");
        }
        Intermediate::Match { switch: Switch::Index(_), .. } => {
            return Err(CubeclError("a `select` on an index has no CubeCL form yet".into()));
        }
        Intermediate::Wide { names, wide_ft, inters, values, .. } => {
            if wide_ft != "f64" {
                return Err(CubeclError(format!("a wide scalar of `{wide_ft}` has no CubeCL form (only f64)")));
            }
            // the block's names are unique across the result, so its
            // statements need no scope of their own
            let wide = ctx.wide();
            for it in inters {
                write_stmt(buf, it, &wide)?;
            }
            for (n, v) in names.iter().zip(values) {
                buf.push_str("let ");
                buf.push_str(n);
                buf.push_str(" = ");
                narrow(buf, v, &wide, ctx.base)?;
                buf.push_str(";\n");
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{branch, cos, cse_scoped_wide, pow, sin, symbol, with_wide};

    fn map(s: &str) -> Option<String> {
        Some(format!("v_{}", s.replace('.', "_")))
    }

    #[test]
    fn literals_carry_the_scalar_and_round_to_it() {
        let x = symbol("x");
        assert_eq!((x.clone() * 0.1).to_cubecl("f64").unwrap(), "(0.1f64 * x)");
        assert_eq!((x.clone() * 0.1).to_cubecl("f32").unwrap(), "(0.1f32 * x)");
        assert_eq!((x.clone() + 1e-12).to_cubecl("f64").unwrap(), "(x + 1e-12f64)");
        assert_eq!((x * -2.5).to_cubecl("f64").unwrap(), "((-2.5f64) * x)");
    }

    #[test]
    fn integer_powers_are_products_and_halves_square_roots() {
        let x = symbol("x");
        assert_eq!(pow(x.clone(), 3.0).to_cubecl("f64").unwrap(), "(x * x * x)");
        assert_eq!(pow(x.clone(), -2.0).to_cubecl("f64").unwrap(), "(1.0f64 / (x * x))");
        assert_eq!(pow(x.clone(), 0.5).to_cubecl("f64").unwrap(), "f64::sqrt(x)");
        assert_eq!(pow(x, 1.7).to_cubecl("f64").unwrap(), "f64::powf(x, 1.7f64)");
    }

    /// An inline branch evaluates both sides (`select`); a branch the CSE
    /// fuses with others on its condition runs only the taken side's
    /// statements, in an `if` that assigns the names declared before it.
    #[test]
    fn a_branch_is_a_select_and_a_fused_branch_an_if() {
        let (q, a) = (symbol("q"), symbol("a"));
        assert_eq!(branch(q.clone(), sin(a.clone()), 0.0).to_cubecl("f64").unwrap(), "select(q >= 0.0f64, f64::sin(a), 0.0f64)");
        let outs = [branch(q.clone(), sin(a.clone()) * cos(a.clone()), 0.0), branch(q, sin(a.clone()) + cos(a), 1.0)];
        let body = cubecl_body(&outs, "f64").unwrap();
        assert!(body.stmts.contains("if q >= 0.0f64 {") && body.stmts.contains("} else {"), "{}", body.stmts);
        assert!(body.stmts.contains("let mut __m"), "{}", body.stmts);
    }

    #[test]
    fn a_wide_block_widens_its_leaves_and_narrows_its_values() {
        let (x, y) = (symbol("x"), symbol("y"));
        let outs = [with_wide(x.clone() * y.clone() - 1.0) * 3.0];
        let (inters, values) = cse_scoped_wide(&outs, "f64");
        let code: String = inters.iter().map(|i| i.to_cubecl("f32").unwrap()).collect();
        assert!(code.contains("f64::cast_from(x)") && code.contains("f64::cast_from(y)"), "{code}");
        assert!(code.contains("f32::cast_from(") && code.contains("1.0f64"), "{code}");
        assert!(values[0].to_cubecl("f32").unwrap().contains("3.0f32"));
        // in an f64 kernel the block is the identity
        let code64: String = inters.iter().map(|i| i.to_cubecl("f64").unwrap()).collect();
        assert!(!code64.contains("cast_from"), "{code64}");
    }

    #[test]
    fn the_map_renames_symbols_but_not_cse_temporaries() {
        let (x, y) = (symbol("x"), symbol("y"));
        let r = sin(x.clone() * y.clone()) + cos(x.clone() * y.clone());
        let body = cubecl_body_with(&[r.clone(), r.diff("x")], "f64", &map).unwrap();
        assert!(body.stmts.contains("let __x0 = (v_x * v_y);"), "{}", body.stmts);
        assert!(body.outputs.iter().all(|o| !o.contains("v___x")), "{:?}", body.outputs);
    }

    #[test]
    fn what_has_no_cubecl_form_is_named() {
        let x = symbol("x");
        let none = |_: &str| None;
        assert_eq!(x.to_cubecl_with("f64", &none).unwrap_err().0, "the kernel has no value for `x`");
        assert!(crate::fast_atan(x.clone()).to_cubecl("f64").unwrap_err().0.contains("fast_atan"));
        assert!(x.to_cubecl("f16").is_err());
    }
}
