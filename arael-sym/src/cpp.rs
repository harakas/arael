//! Code for GPU kernels written in C++.
//!
//! arael's own GPU runtime compiles kernels written in a C++ subset that
//! CUDA, HIP and Metal share (NVRTC, hipRTC and Metal's compiler at run
//! time). [`E::to_cpp`] prints an expression as C++ for such a kernel, as
//! [`E::to_rust`] prints it for Rust; [`Intermediate::to_cpp`] prints one
//! statement of a CSE result, and [`cpp_body`] runs the CSE and prints in
//! one call. arael's device assembly generates its kernels' bodies this way.
//!
//! ```
//! use arael_sym::{sin, symbols};
//! let (x, y) = symbols!(x, y);
//! let e = sin(x) * y + 2.0;
//! assert_eq!(e.to_cpp("f64").unwrap(), "((y * sin(x)) + 2.0)");
//! assert_eq!(e.to_cpp("f32").unwrap(), "((y * sin(x)) + 2.0f)");
//! ```
//!
//! | | `to_rust` | `to_cpp` |
//! |---|---|---|
//! | scalar | any, or inferred (`""`) | `"f32"` (`float`) or `"f64"` (`double`), every literal rounded to it |
//! | functions | methods, `x.sin()` | overloaded calls, `sin(x)`, `fabs(x)`, `log(x)` |
//! | whole powers | `x.powi(3)` | products up to 8, `(x * x * x)`; others `pow` |
//! | `branch` printed directly | `if` | `?:`: the taken side only |
//! | `branch` after CSE | `let (a, ..) = if q >= 0.0 { .. } else { .. };` | names declared, then an `if` assigning them |
//! | `with_wide` | `arael::utils::Wide` and `narrow` | computed in `double`, or a type the caller names ([`E::to_cpp_in`]), cast back |
//!
//! After CSE ([`cpp_body`]) every branch is an `if` that runs only the taken
//! side, the names it binds declared before it:
//!
//! ```
//! use arael_sym::{branch, cpp_body, sqrt, symbols};
//! let (x,) = symbols!(x);
//! let body = cpp_body(&[branch(x.clone(), sqrt(x), 0.0)], "f32").unwrap();
//! assert_eq!(body.stmts, "float __m0 = 0.0f;\nif (x >= 0.0f) {\n__m0 = sqrt(x);\n} else {\n__m0 = 0.0f;\n}\n");
//! assert_eq!(body.outputs, ["__m0"]);
//! ```
//!
//! A wide block can be computed in a type the kernel defines instead of
//! `double` (`to_cpp_in`, `cpp_body_in`): arael's kernels name their wide
//! scalar `W`, `double` or a two-float type. A literal of such a type prints
//! as `ARAEL_WIDE(d, hi, lo)`, a macro the kernel defines: the value as a
//! double literal, then its two-f32 split (`hi` the nearest f32, `lo` the
//! f32 nearest the rest).
//!
//! ```
//! use arael_sym::{symbols, with_wide};
//! let (x,) = symbols!(x);
//! let e = with_wide(x * 0.1 - 1.0);
//! assert_eq!(e.to_cpp_in("f32", "W", &|s| Some(s.to_string())).unwrap(),
//!     "((float)((ARAEL_WIDE(0.1, 0.1f, (-1.4901161e-9f)) * ((W)x)) - ARAEL_WIDE(1.0, 1.0f, 0.0f)))");
//! ```
//!
//! arael's own external functions print as calls of their helpers in
//! arael's kernel library ([`EXTERN_CPP`]):
//!
//! ```
//! use arael_sym::{rad_diff, safe_sqrt, symbols};
//! let (a, b) = symbols!(a, b);
//! assert_eq!(rad_diff(a.clone(), b).to_cpp("f64").unwrap(), "arael_rad_diff(a, b)");
//! assert_eq!(safe_sqrt(a).to_cpp("f32").unwrap(), "arael_safe_sqrt(a)");
//! ```
//!
//! What has no C++ form yet is an error naming it: a `select` on an index,
//! and the other external functions (`#[arael::function]`, `fast_atan` and
//! `fast_atan2` of `arael::utils`).

use crate::cse::{Arm, Intermediate, Switch};
use crate::{E, Expr, FuncKind};

/// Why an expression has no C++ form.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CppError(pub String);

impl std::fmt::Display for CppError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CppError {}

/// What the printer prints against: the C++ type of the code being printed
/// (inside a `with_wide` block, the wide type), the kernel's own type, the
/// wide type, and the symbol map.
#[derive(Clone, Copy)]
struct Ctx<'a> {
    ft: &'a str,
    base: &'a str,
    wide: &'a str,
    map: &'a dyn Fn(&str) -> Option<String>,
}

impl<'a> Ctx<'a> {
    fn new(scalar: &str, wide: &'a str, map: &'a dyn Fn(&str) -> Option<String>) -> Result<Self, CppError> {
        let ft = match scalar {
            "f32" => "float",
            "f64" => "double",
            _ => return Err(CppError(format!("a kernel scalar is `f32` or `f64`, not `{scalar}`"))),
        };
        Ok(Ctx { ft, base: ft, wide, map })
    }

    fn wide(&self) -> Ctx<'a> {
        Ctx { ft: self.wide, ..*self }
    }
}

type Res = Result<(), CppError>;

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
    /// The expression as C++ for a kernel of arael's GPU runtime, in
    /// `scalar` (`"f32"` or `"f64"`), every symbol by its own name; see the
    /// [module](crate::cpp) for how it differs from [`to_rust`](Self::to_rust).
    ///
    /// ```
    /// use arael_sym::{pow, symbols};
    /// let (x,) = symbols!(x);
    /// assert_eq!(pow(x, 3.0).to_cpp("f32").unwrap(), "(x * x * x)");
    /// ```
    pub fn to_cpp(&self, scalar: &str) -> Result<String, CppError> {
        self.to_cpp_with(scalar, &verbatim)
    }

    /// As [`to_cpp`](Self::to_cpp), every symbol printed as `map(name)`:
    /// the value the kernel reads it from. A symbol the map has no value
    /// for is an error.
    ///
    /// ```
    /// use arael_sym::symbols;
    /// let (x,) = symbols!(x);
    /// let read = |s: &str| Some(format!("v[{}]", if s == "x" { 0 } else { 1 }));
    /// assert_eq!((x * 2.0).to_cpp_with("f64", &read).unwrap(), "(2.0 * v[0])");
    /// ```
    pub fn to_cpp_with(&self, scalar: &str, map: &dyn Fn(&str) -> Option<String>) -> Result<String, CppError> {
        self.to_cpp_in(scalar, "double", map)
    }

    /// As [`to_cpp_with`](Self::to_cpp_with), wide blocks computed in the
    /// C++ type `wide` (see the [module](crate::cpp) for its literals).
    pub fn to_cpp_in(&self, scalar: &str, wide: &str, map: &dyn Fn(&str) -> Option<String>) -> Result<String, CppError> {
        let ctx = Ctx::new(scalar, wide, map)?;
        let mut buf = String::new();
        write_expr(&mut buf, self, &ctx)?;
        Ok(buf)
    }
}

impl Intermediate {
    /// One statement of a CSE result ([`cse_scoped_wide`](crate::cse_scoped_wide))
    /// as C++, in `scalar`, symbols by their own names.
    pub fn to_cpp(&self, scalar: &str) -> Result<String, CppError> {
        self.to_cpp_with(scalar, &verbatim)
    }

    /// As [`to_cpp`](Self::to_cpp), symbols through `map`.
    pub fn to_cpp_with(&self, scalar: &str, map: &dyn Fn(&str) -> Option<String>) -> Result<String, CppError> {
        self.to_cpp_in(scalar, "double", map)
    }

    /// As [`to_cpp_with`](Self::to_cpp_with), wide blocks in the C++ type `wide`.
    pub fn to_cpp_in(&self, scalar: &str, wide: &str, map: &dyn Fn(&str) -> Option<String>) -> Result<String, CppError> {
        let ctx = Ctx::new(scalar, wide, map)?;
        let mut buf = String::new();
        write_stmt(&mut buf, self, &ctx)?;
        Ok(buf)
    }
}

/// A kernel body: the statements of a CSE, then one expression per output.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CppBody {
    /// The statements, one per line.
    pub stmts: String,
    /// One expression per output, valid after `stmts`.
    pub outputs: Vec<String>,
}

/// `exprs` after [`cse_scoped_wide`](crate::cse_scoped_wide), printed as C++
/// in `scalar`, symbols by their own names.
///
/// ```
/// use arael_sym::{cpp_body, sin, symbols};
/// let (x,) = symbols!(x);
/// let body = cpp_body(&[sin(x.clone()) * 2.0, sin(x)], "f64").unwrap();
/// assert_eq!(body.stmts, "const double __x0 = sin(x);\n");
/// assert_eq!(body.outputs, ["(2.0 * __x0)", "__x0"]);
/// ```
pub fn cpp_body(exprs: &[E], scalar: &str) -> Result<CppBody, CppError> {
    cpp_body_with(exprs, scalar, &verbatim)
}

/// As [`cpp_body`], symbols through `map`.
pub fn cpp_body_with(exprs: &[E], scalar: &str, map: &dyn Fn(&str) -> Option<String>) -> Result<CppBody, CppError> {
    cpp_body_in(exprs, scalar, "double", map)
}

/// As [`cpp_body_with`], wide blocks in the C++ type `wide`.
pub fn cpp_body_in(exprs: &[E], scalar: &str, wide: &str, map: &dyn Fn(&str) -> Option<String>) -> Result<CppBody, CppError> {
    let ctx = Ctx::new(scalar, wide, map)?;
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
        .collect::<Result<Vec<String>, CppError>>()?;
    Ok(CppBody { stmts, outputs })
}

/// `v` as a literal of `ft`, rounded to it; a negative one in parentheses.
/// Of a type other than `float` and `double`, `ARAEL_WIDE(d, hi, lo)`.
fn literal(buf: &mut String, v: f64, ft: &str) {
    if ft != "float" && ft != "double" {
        let hi = v as f32;
        let lo = if hi.is_finite() { (v - hi as f64) as f32 } else { 0.0 };
        buf.push_str("ARAEL_WIDE(");
        literal(buf, v, "double");
        buf.push_str(", ");
        literal(buf, hi as f64, "float");
        buf.push_str(", ");
        literal(buf, lo as f64, "float");
        buf.push(')');
        return;
    }
    let suffix = if ft == "float" { "f" } else { "" };
    if v.is_nan() {
        buf.push_str(&format!("(0.0{suffix} / 0.0{suffix})"));
    } else if v.is_infinite() {
        buf.push_str(&format!("({}1.0{suffix} / 0.0{suffix})", if v < 0.0 { "-" } else { "" }));
    } else {
        let digits = if ft == "float" { format!("{:?}", v as f32) } else { format!("{v:?}") };
        if digits.starts_with('-') {
            buf.push_str(&format!("({digits}{suffix})"));
        } else {
            buf.push_str(&format!("{digits}{suffix}"));
        }
    }
}

fn call(buf: &mut String, f: &str, args: &[&E], ctx: &Ctx) -> Res {
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

/// `((q >= 0) ? a : b)`: only the taken side is evaluated.
fn ternary_ge0(buf: &mut String, q: &E, a: &dyn Fn(&mut String) -> Res, b: &dyn Fn(&mut String) -> Res, ctx: &Ctx) -> Res {
    buf.push_str("((");
    write_expr(buf, q, ctx)?;
    buf.push_str(" >= ");
    literal(buf, 0.0, ctx.ft);
    buf.push_str(") ? ");
    a(buf)?;
    buf.push_str(" : ");
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
                    None => return Err(CppError(format!("the kernel has no value for `{name}`"))),
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
        Expr::Sin(a) => call(buf, "sin", &[a], ctx)?,
        Expr::Cos(a) => call(buf, "cos", &[a], ctx)?,
        Expr::Tan(a) => call(buf, "tan", &[a], ctx)?,
        Expr::Asin(a) => call(buf, "asin", &[a], ctx)?,
        Expr::Acos(a) => call(buf, "acos", &[a], ctx)?,
        Expr::Atan(a) => call(buf, "atan", &[a], ctx)?,
        Expr::Atan2(y, x) => call(buf, "atan2", &[y, x], ctx)?,
        Expr::Sinh(a) => call(buf, "sinh", &[a], ctx)?,
        Expr::Cosh(a) => call(buf, "cosh", &[a], ctx)?,
        Expr::Tanh(a) => call(buf, "tanh", &[a], ctx)?,
        Expr::Exp(a) => call(buf, "exp", &[a], ctx)?,
        Expr::Ln(a) => call(buf, "log", &[a], ctx)?,
        Expr::Log2(a) => call(buf, "log2", &[a], ctx)?,
        Expr::Log10(a) => call(buf, "log10", &[a], ctx)?,
        Expr::Sqrt(a) => call(buf, "sqrt", &[a], ctx)?,
        Expr::Abs(a) => call(buf, "fabs", &[a], ctx)?,
        Expr::Heaviside(a) => {
            let one = |b: &mut String| {
                literal(b, 1.0, ft);
                Ok(())
            };
            let zero = |b: &mut String| {
                literal(b, 0.0, ft);
                Ok(())
            };
            ternary_ge0(buf, a, &one, &zero, ctx)?
        }
        Expr::Clamp(v, lo, hi) => {
            // v < lo ? lo : (v > hi ? hi : v), as arael's `clamp`
            buf.push('(');
            binary(buf, v, " < ", lo, ctx)?;
            buf.push_str(" ? ");
            write_expr(buf, lo, ctx)?;
            buf.push_str(" : (");
            binary(buf, v, " > ", hi, ctx)?;
            buf.push_str(" ? ");
            write_expr(buf, hi, ctx)?;
            buf.push_str(" : ");
            write_expr(buf, v, ctx)?;
            buf.push_str("))");
        }
        Expr::Branch(q, a, b) => ternary_ge0(buf, q, &|s| write_expr(s, a, ctx), &|s| write_expr(s, b, ctx), ctx)?,
        Expr::Select { .. } => return Err(CppError("a `select` on an index has no C++ form yet".into())),
        Expr::Func { name, params, kind, args } => {
            if name == "with_wide" && args.len() == 1 {
                // a scope printed inline: the inside in double over widened
                // leaves, narrowed back to the surrounding scalar
                let inner = crate::widen_leaves(&crate::strip_with_wide(&args[0]));
                narrow(buf, &inner, &ctx.wide(), ft)?;
            } else if name == "widen" && args.len() == 1 {
                // a leaf of the body, in the kernel's scalar, read into a
                // block of the wide scalar
                let leaf = Ctx { ft: ctx.base, ..*ctx };
                if ctx.base == ctx.wide {
                    write_expr(buf, &args[0], &leaf)?;
                } else {
                    buf.push_str("((");
                    buf.push_str(ctx.wide);
                    buf.push(')');
                    write_expr(buf, &args[0], &leaf)?;
                    buf.push(')');
                }
            } else if name == "wide" && args.len() == 1 {
                write_expr(buf, &args[0], ctx)?;
            } else if let Some(body) = kind.body() {
                write_expr(buf, &crate::expand_func(params, body, args), ctx)?;
            } else if let Some(f) = extern_cpp(kind) {
                let args: Vec<&E> = args.iter().collect();
                call(buf, f, &args, ctx)?;
            } else {
                let what = match kind {
                    FuncKind::Extern { call_path, .. } | FuncKind::ExternNumericDerivs { call_path, .. } => call_path.clone(),
                    _ => name.clone(),
                };
                return Err(CppError(format!("`{what}` has no C++ form")));
            }
        }
    }
    Ok(())
}

/// The C++ helpers of arael's own external functions (`arael::utils`), by
/// their Rust path: device functions of arael's kernel library, overloaded
/// for `float` and `double`, computing what the Rust ones do.
pub const EXTERN_CPP: &[(&str, &str)] = &[
    ("arael::utils::rad_diff", "arael_rad_diff"),
    ("arael::utils::rad_sum", "arael_rad_sum"),
    ("arael::utils::safe_sqrt", "arael_safe_sqrt"),
    ("arael::utils::epsilon_for", "arael_epsilon_for"),
    ("arael::utils::exp_m1", "expm1"),
    ("arael::utils::ln_1p", "log1p"),
];

/// The C++ function an external function's call prints as.
fn extern_cpp(kind: &FuncKind) -> Option<&'static str> {
    let FuncKind::Extern { call_path, .. } = kind else { return None };
    EXTERN_CPP.iter().find(|(path, _)| path == call_path).map(|&(_, f)| f)
}

/// `v`, computed in `wide`, as `to`.
fn narrow(buf: &mut String, v: &E, wide: &Ctx, to: &str) -> Res {
    if to == wide.ft {
        return write_expr(buf, v, wide);
    }
    buf.push_str("((");
    buf.push_str(to);
    buf.push(')');
    write_expr(buf, v, wide)?;
    buf.push(')');
    Ok(())
}

fn write_pow(buf: &mut String, b: &E, ex: &E, ctx: &Ctx) -> Res {
    let ft = ctx.ft;
    if let Expr::Const(n) = ex.as_ref() {
        let n = *n;
        if n == 0.5 {
            return call(buf, "sqrt", &[b], ctx);
        }
        if n == -0.5 {
            buf.push('(');
            literal(buf, 1.0, ft);
            buf.push_str(" / ");
            call(buf, "sqrt", &[b], ctx)?;
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
    call(buf, "pow", &[b, ex], ctx)
}

fn names_zeroed(buf: &mut String, names: &[String], ft: &str) {
    for n in names {
        buf.push_str(ft);
        buf.push(' ');
        buf.push_str(n);
        buf.push_str(" = ");
        literal(buf, 0.0, ft);
        buf.push_str(";\n");
    }
}

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
            buf.push_str("const ");
            buf.push_str(ctx.ft);
            buf.push(' ');
            buf.push_str(name);
            buf.push_str(" = ");
            write_expr(buf, expr, ctx)?;
            buf.push_str(";\n");
        }
        Intermediate::Match { switch: Switch::Sign(q), names, arms, default } => {
            // the names declared before, assigned in either side: only the
            // taken side's statements run, and a NaN `q` takes the else side,
            // as in the Rust form
            names_zeroed(buf, names, ctx.ft);
            buf.push_str("if (");
            write_expr(buf, q, ctx)?;
            buf.push_str(" >= ");
            literal(buf, 0.0, ctx.ft);
            buf.push_str(") {\n");
            write_arm(buf, names, &arms[0], ctx)?;
            buf.push_str("} else {\n");
            write_arm(buf, names, default.as_ref().expect("a fused branch has an else side"), ctx)?;
            buf.push_str("}\n");
        }
        Intermediate::Match { switch: Switch::Index(_), .. } => {
            return Err(CppError("a `select` on an index has no C++ form yet".into()));
        }
        Intermediate::Wide { names, inters, values, .. } => {
            // computed in the kernel's wide type whatever the host's (f64 or
            // df32): the block's names are unique across the result, so its
            // statements need no scope of their own
            let wide = ctx.wide();
            for it in inters {
                write_stmt(buf, it, &wide)?;
            }
            for (n, v) in names.iter().zip(values) {
                buf.push_str("const ");
                buf.push_str(ctx.base);
                buf.push(' ');
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
    use crate::{branch, clamp, cos, cse_scoped_wide, heaviside, pow, sin, symbol, with_wide};

    fn map(s: &str) -> Option<String> {
        Some(format!("v_{}", s.replace('.', "_")))
    }

    #[test]
    fn literals_carry_the_scalar_and_round_to_it() {
        let x = symbol("x");
        assert_eq!((x.clone() * 0.1).to_cpp("f64").unwrap(), "(0.1 * x)");
        assert_eq!((x.clone() * 0.1).to_cpp("f32").unwrap(), "(0.1f * x)");
        assert_eq!((x.clone() + 1e-12).to_cpp("f64").unwrap(), "(x + 1e-12)");
        assert_eq!((x.clone() + 1e-12).to_cpp("f32").unwrap(), "(x + 1e-12f)");
        assert_eq!((x * -2.5).to_cpp("f64").unwrap(), "((-2.5) * x)");
    }

    #[test]
    fn integer_powers_are_products_and_halves_square_roots() {
        let x = symbol("x");
        assert_eq!(pow(x.clone(), 3.0).to_cpp("f64").unwrap(), "(x * x * x)");
        assert_eq!(pow(x.clone(), -2.0).to_cpp("f64").unwrap(), "(1.0 / (x * x))");
        assert_eq!(pow(x.clone(), 0.5).to_cpp("f64").unwrap(), "sqrt(x)");
        assert_eq!(pow(x, 1.7).to_cpp("f32").unwrap(), "pow(x, 1.7f)");
    }

    #[test]
    fn functions_are_the_overloaded_calls() {
        let x = symbol("x");
        assert_eq!(crate::abs(x.clone()).to_cpp("f32").unwrap(), "fabs(x)");
        assert_eq!(crate::ln(x.clone()).to_cpp("f64").unwrap(), "log(x)");
        assert_eq!(crate::log2(x.clone()).to_cpp("f64").unwrap(), "log2(x)");
        assert_eq!(crate::atan2(x.clone(), symbol("y")).to_cpp("f64").unwrap(), "atan2(x, y)");
    }

    #[test]
    fn a_branch_is_a_ternary_and_a_fused_branch_an_if() {
        let (q, a) = (symbol("q"), symbol("a"));
        assert_eq!(branch(q.clone(), sin(a.clone()), 0.0).to_cpp("f64").unwrap(), "((q >= 0.0) ? sin(a) : 0.0)");
        assert_eq!(heaviside(q.clone()).to_cpp("f32").unwrap(), "((q >= 0.0f) ? 1.0f : 0.0f)");
        assert_eq!(clamp(a.clone(), 0.0, 1.0).to_cpp("f64").unwrap(), "((a < 0.0) ? 0.0 : ((a > 1.0) ? 1.0 : a))");
        let outs = [branch(q.clone(), sin(a.clone()) * cos(a.clone()), 0.0), branch(q, sin(a.clone()) + cos(a), 1.0)];
        let body = cpp_body(&outs, "f64").unwrap();
        assert!(body.stmts.contains("if (q >= 0.0) {") && body.stmts.contains("} else {"), "{}", body.stmts);
        assert!(body.stmts.contains("double __m"), "{}", body.stmts);
    }

    #[test]
    fn a_wide_block_widens_its_leaves_and_narrows_its_values() {
        let (x, y) = (symbol("x"), symbol("y"));
        let outs = [with_wide(x.clone() * y.clone() - 1.0) * 3.0];
        let (inters, values) = cse_scoped_wide(&outs, "f64");
        let code: String = inters.iter().map(|i| i.to_cpp("f32").unwrap()).collect();
        assert!(code.contains("((double)x)") && code.contains("((double)y)"), "{code}");
        assert!(code.contains("((float)") && code.contains("1.0"), "{code}");
        assert!(values[0].to_cpp("f32").unwrap().contains("3.0f"));
        // in a double kernel the block is the identity
        let code64: String = inters.iter().map(|i| i.to_cpp("f64").unwrap()).collect();
        assert!(!code64.contains("(double)") && !code64.contains("(float)"), "{code64}");
    }

    /// In a named wide type: leaves cast to it, literals as ARAEL_WIDE with
    /// their split, the values cast back to the kernel's scalar; a host wide
    /// scalar other than f64 prints the same.
    #[test]
    fn a_wide_block_in_a_named_type() {
        let (x, y) = (symbol("x"), symbol("y"));
        let outs = [with_wide(x.clone() * y.clone() - 0.1) * 3.0];
        for host_wide in ["f64", "df32"] {
            let (inters, _) = cse_scoped_wide(&outs, host_wide);
            let code: String = inters.iter().map(|i| i.to_cpp_in("f32", "W", &verbatim).unwrap()).collect();
            assert!(code.contains("((W)x)") && code.contains("((W)y)"), "{code}");
            assert!(code.contains("ARAEL_WIDE(0.1, 0.1f, (-1.4901161e-9f))"), "{code}");
            assert!(code.contains("((float)"), "{code}");
            assert!(!code.contains("double"), "{code}");
        }
        // an f64 kernel's leaves are cast too: W is not double there
        let (inters, _) = cse_scoped_wide(&outs, "f64");
        let code: String = inters.iter().map(|i| i.to_cpp_in("f64", "W", &verbatim).unwrap()).collect();
        assert!(code.contains("((W)x)") && code.contains("((double)"), "{code}");
    }

    #[test]
    fn the_map_renames_symbols_but_not_cse_temporaries() {
        let (x, y) = (symbol("x"), symbol("y"));
        let r = sin(x.clone() * y.clone()) + cos(x.clone() * y.clone());
        let body = cpp_body_with(&[r.clone(), r.diff("x")], "f64", &map).unwrap();
        assert!(body.stmts.contains("const double __x0 = (v_x * v_y);"), "{}", body.stmts);
        assert!(body.outputs.iter().all(|o| !o.contains("v___x")), "{:?}", body.outputs);
    }

    /// The example of docs/SYM.md ("Code for GPU kernels (C++)").
    #[test]
    fn the_range_example_of_the_docs() {
        let (x, y) = (symbol("x"), symbol("y"));
        let r = crate::sqrt(x.clone() * x.clone() + y.clone() * y.clone()) - 1.0;
        let body = cpp_body(&[r.clone(), r.diff("x"), r.diff("y")], "f32").unwrap();
        assert_eq!(body.stmts, "const float __x0 = sqrt(((x * x) + (y * y)));\nconst float __x1 = (1.0f / __x0);\n");
        assert_eq!(body.outputs, ["(__x0 - 1.0f)", "(x * __x1)", "(y * __x1)"]);
    }

    #[test]
    fn what_has_no_cpp_form_is_named() {
        let x = symbol("x");
        let none = |_: &str| None;
        assert_eq!(x.to_cpp_with("f64", &none).unwrap_err().0, "the kernel has no value for `x`");
        assert!(crate::fast_atan(x.clone()).to_cpp("f64").unwrap_err().0.contains("fast_atan"));
        assert!(x.to_cpp("f16").is_err());
    }

    /// arael's own external functions print as calls of their kernel
    /// helpers, and so do their derivatives.
    #[test]
    fn arael_externs_print_as_their_helpers() {
        let (a, b) = (symbol("a"), symbol("b"));
        assert_eq!(crate::rad_diff(a.clone(), b.clone()).to_cpp("f64").unwrap(), "arael_rad_diff(a, b)");
        assert_eq!(crate::rad_sum(a.clone(), b.clone()).to_cpp("f32").unwrap(), "arael_rad_sum(a, b)");
        assert_eq!(crate::safe_sqrt(a.clone()).to_cpp("f32").unwrap(), "arael_safe_sqrt(a)");
        assert_eq!(crate::exp_m1(a.clone()).to_cpp("f64").unwrap(), "expm1(a)");
        assert_eq!(crate::ln_1p(a.clone()).to_cpp("f64").unwrap(), "log1p(a)");
        assert_eq!(crate::epsilon_for(a.clone()).to_cpp("f32").unwrap(), "arael_epsilon_for(a)");
        let d = crate::safe_sqrt(a.clone()).diff("a").to_cpp("f64").unwrap();
        assert!(d.contains("arael_epsilon_for(a)"), "{d}");
        assert_eq!(crate::rad_diff(a.clone(), b).diff("b").to_cpp("f64").unwrap(), "(-1.0)");
    }
}
