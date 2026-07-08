use std::fmt;
use super::{Expr, E};

// Operator precedence for minimal parenthesization
fn precedence(e: &Expr) -> u8 {
    match e {
        Expr::Add(..) | Expr::Sub(..) => 1,
        Expr::Mul(..) | Expr::Div(..) => 2,
        Expr::Neg(..) => 3,
        Expr::Pow(..) => 4,
        _ => 10, // atoms and functions
    }
}

fn fmt_child(f: &mut fmt::Formatter<'_>, child: &Expr, parent_prec: u8, right_assoc: bool) -> fmt::Result {
    let child_prec = precedence(child);
    let needs_parens = if right_assoc {
        child_prec < parent_prec || (child_prec == parent_prec && parent_prec <= 2)
    } else {
        child_prec < parent_prec
    };
    if needs_parens {
        write!(f, "(")?;
        fmt::Display::fmt(child, f)?;
        write!(f, ")")
    } else {
        fmt::Display::fmt(child, f)
    }
}

fn fmt_unary(f: &mut fmt::Formatter<'_>, name: &str, arg: &Expr) -> fmt::Result {
    write!(f, "{name}(")?;
    fmt::Display::fmt(arg, f)?;
    write!(f, ")")
}

fn fmt_binary_fn(f: &mut fmt::Formatter<'_>, name: &str, a: &Expr, b: &Expr) -> fmt::Result {
    write!(f, "{name}(")?;
    fmt::Display::fmt(a, f)?;
    write!(f, ", ")?;
    fmt::Display::fmt(b, f)?;
    write!(f, ")")
}

/// Human-readable constant: integer-valued floats print without a
/// decimal point (Display and LaTeX).
fn fmt_const(f: &mut fmt::Formatter<'_>, v: f64) -> fmt::Result {
    match crate::as_exact_int(v) {
        Some(i) => write!(f, "{i}"),
        None => write!(f, "{v}"),
    }
}

/// Rust-source constant: always a FLOAT literal (integer-valued floats
/// get a trailing `.0` so they cannot parse as integer literals), with
/// the float-type suffix when one is requested. An empty `ft` emits an
/// unsuffixed literal for type-inferred contexts (macro-generated code).
fn rust_float_literal(v: f64, ft: &str) -> String {
    if !v.is_finite() {
        // `{v}` prints inf/NaN, which are not Rust tokens. Suffixed
        // contexts get the typed constant path; the type-inferred
        // context gets a literal expression that adopts the surrounding
        // precision (a `f64::` path would pin it).
        if ft.is_empty() {
            return if v.is_nan() { "(0.0 / 0.0)".to_string() }
                else if v > 0.0 { "(1.0 / 0.0)".to_string() }
                else { "(-1.0 / 0.0)".to_string() };
        }
        let path = if v.is_nan() { "NAN" } else if v > 0.0 { "INFINITY" } else { "NEG_INFINITY" };
        return format!("{ft}::{path}");
    }
    let base = match crate::as_exact_int(v) {
        Some(i) => format!("{i}.0"),
        None => format!("{v}"),
    };
    if ft.is_empty() { base } else { format!("{base}_{ft}") }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Sym(name) => write!(f, "{name}"),
            Expr::Const(v) => fmt_const(f, *v),
            Expr::NamedConst { name, .. } => write!(f, "{name}"),
            Expr::Neg(a) => {
                write!(f, "-")?;
                let needs_parens = matches!(a.as_ref(), Expr::Add(..) | Expr::Sub(..) | Expr::Neg(_));
                if needs_parens {
                    write!(f, "(")?;
                    fmt::Display::fmt(a.as_ref(), f)?;
                    write!(f, ")")
                } else {
                    fmt::Display::fmt(a.as_ref(), f)
                }
            }
            Expr::Add(a, b) => {
                let p = precedence(self);
                fmt_child(f, a, p, false)?;
                write!(f, " + ")?;
                fmt_child(f, b, p, false)
            }
            Expr::Sub(a, b) => {
                let p = precedence(self);
                fmt_child(f, a, p, false)?;
                write!(f, " - ")?;
                fmt_child(f, b, p, true)
            }
            Expr::Mul(a, b) => {
                let p = precedence(self);
                fmt_child(f, a, p, false)?;
                write!(f, " * ")?;
                fmt_child(f, b, p, false)
            }
            Expr::Div(a, b) => {
                let p = precedence(self);
                fmt_child(f, a, p, false)?;
                write!(f, " / ")?;
                fmt_child(f, b, p, true)
            }
            Expr::Pow(a, b) => {
                // A Pow base must also be parenthesized: `x^2^3` re-parses
                // right-associatively as x^(2^3), not the (x^2)^3 it prints.
                let base_needs = precedence(a) < precedence(self)
                    || matches!(&**a, Expr::Pow(..));
                if base_needs {
                    write!(f, "(")?;
                    fmt::Display::fmt(a.as_ref(), f)?;
                    write!(f, ")")?;
                } else {
                    fmt::Display::fmt(a.as_ref(), f)?;
                }
                write!(f, "^")?;
                let exp_needs = precedence(b) < 10;
                if exp_needs {
                    write!(f, "(")?;
                    fmt::Display::fmt(b.as_ref(), f)?;
                    write!(f, ")")
                } else {
                    fmt::Display::fmt(b.as_ref(), f)
                }
            }
            Expr::Sin(a) => fmt_unary(f, "sin", a),
            Expr::Cos(a) => fmt_unary(f, "cos", a),
            Expr::Tan(a) => fmt_unary(f, "tan", a),
            Expr::Asin(a) => fmt_unary(f, "asin", a),
            Expr::Acos(a) => fmt_unary(f, "acos", a),
            Expr::Atan(a) => fmt_unary(f, "atan", a),
            Expr::Atan2(y, x) => fmt_binary_fn(f, "atan2", y, x),
            Expr::Sinh(a) => fmt_unary(f, "sinh", a),
            Expr::Cosh(a) => fmt_unary(f, "cosh", a),
            Expr::Tanh(a) => fmt_unary(f, "tanh", a),
            Expr::Exp(a) => fmt_unary(f, "exp", a),
            Expr::Ln(a) => fmt_unary(f, "ln", a),
            Expr::Log2(a) => fmt_unary(f, "log2", a),
            Expr::Log10(a) => fmt_unary(f, "log10", a),
            Expr::Sqrt(a) => fmt_unary(f, "sqrt", a),
            Expr::Abs(a) => fmt_unary(f, "abs", a),
            Expr::Heaviside(a) => fmt_unary(f, "H", a),
            Expr::Clamp(val, lo, hi) => {
                write!(f, "clamp(")?;
                fmt::Display::fmt(val.as_ref(), f)?;
                write!(f, ", ")?;
                fmt::Display::fmt(lo.as_ref(), f)?;
                write!(f, ", ")?;
                fmt::Display::fmt(hi.as_ref(), f)?;
                write!(f, ")")
            }
            Expr::Branch(q, a, b) => {
                write!(f, "branch(")?;
                fmt::Display::fmt(q.as_ref(), f)?;
                write!(f, ", ")?;
                fmt::Display::fmt(a.as_ref(), f)?;
                write!(f, ", ")?;
                fmt::Display::fmt(b.as_ref(), f)?;
                write!(f, ")")
            }
            Expr::Func { name, args, .. } => {
                write!(f, "{name}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    fmt::Display::fmt(arg.as_ref(), f)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl fmt::Display for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self.as_ref(), f)
    }
}

impl fmt::Debug for E {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Use Display for Debug too — more readable
        fmt::Display::fmt(self.as_ref(), f)
    }
}

// --- LaTeX output ---

impl Expr {
    /// Format the expression as LaTeX math notation.
    ///
    /// Produces a string suitable for embedding in LaTeX documents, using
    /// `\frac`, `\sqrt`, `\sin`, etc.
    pub fn to_latex(&self) -> String {
        let mut buf = String::new();
        self.write_latex(&mut buf);
        buf
    }

    fn write_latex(&self, buf: &mut String) {
        match self {
            Expr::Sym(name) => buf.push_str(name),
            Expr::Const(v) => {
                match crate::as_exact_int(*v) {
                    Some(i) => buf.push_str(&format!("{i}")),
                    None => buf.push_str(&format!("{v}")),
                }
            }
            Expr::NamedConst { latex, .. } => buf.push_str(latex),
            Expr::Neg(a) => {
                buf.push('-');
                let needs_parens = matches!(a.as_ref(), Expr::Add(..) | Expr::Sub(..));
                if needs_parens {
                    buf.push_str("\\left(");
                    a.write_latex(buf);
                    buf.push_str("\\right)");
                } else {
                    a.write_latex(buf);
                }
            }
            Expr::Add(a, b) => {
                a.write_latex(buf);
                buf.push_str(" + ");
                b.write_latex(buf);
            }
            Expr::Sub(a, b) => {
                a.write_latex(buf);
                buf.push_str(" - ");
                let needs_parens = matches!(b.as_ref(), Expr::Add(..) | Expr::Sub(..));
                if needs_parens {
                    buf.push_str("\\left(");
                    b.write_latex(buf);
                    buf.push_str("\\right)");
                } else {
                    b.write_latex(buf);
                }
            }
            Expr::Mul(a, b) => {
                let a_needs = matches!(a.as_ref(), Expr::Add(..) | Expr::Sub(..));
                if a_needs {
                    buf.push_str("\\left(");
                    a.write_latex(buf);
                    buf.push_str("\\right)");
                } else {
                    a.write_latex(buf);
                }
                buf.push_str(" \\cdot ");
                let b_needs = matches!(b.as_ref(), Expr::Add(..) | Expr::Sub(..));
                if b_needs {
                    buf.push_str("\\left(");
                    b.write_latex(buf);
                    buf.push_str("\\right)");
                } else {
                    b.write_latex(buf);
                }
            }
            Expr::Div(a, b) => {
                buf.push_str("\\frac{");
                a.write_latex(buf);
                buf.push_str("}{");
                b.write_latex(buf);
                buf.push('}');
            }
            Expr::Pow(a, b) => {
                let needs_parens = matches!(
                    a.as_ref(),
                    Expr::Add(..) | Expr::Sub(..) | Expr::Mul(..) | Expr::Div(..) | Expr::Neg(..)
                );
                if needs_parens {
                    buf.push_str("\\left(");
                    a.write_latex(buf);
                    buf.push_str("\\right)");
                } else {
                    a.write_latex(buf);
                }
                buf.push_str("^{");
                b.write_latex(buf);
                buf.push('}');
            }
            Expr::Sin(a) => Self::write_latex_fn(buf, "\\sin", a),
            Expr::Cos(a) => Self::write_latex_fn(buf, "\\cos", a),
            Expr::Tan(a) => Self::write_latex_fn(buf, "\\tan", a),
            Expr::Asin(a) => Self::write_latex_fn(buf, "\\arcsin", a),
            Expr::Acos(a) => Self::write_latex_fn(buf, "\\arccos", a),
            Expr::Atan(a) => Self::write_latex_fn(buf, "\\arctan", a),
            Expr::Atan2(y, x) => {
                buf.push_str("\\operatorname{atan2}\\left(");
                y.write_latex(buf);
                buf.push_str(", ");
                x.write_latex(buf);
                buf.push_str("\\right)");
            }
            Expr::Sinh(a) => Self::write_latex_fn(buf, "\\sinh", a),
            Expr::Cosh(a) => Self::write_latex_fn(buf, "\\cosh", a),
            Expr::Tanh(a) => Self::write_latex_fn(buf, "\\tanh", a),
            Expr::Exp(a) => {
                buf.push_str("e^{");
                a.write_latex(buf);
                buf.push('}');
            }
            Expr::Ln(a) => Self::write_latex_fn(buf, "\\ln", a),
            Expr::Log2(a) => Self::write_latex_fn(buf, "\\log_2", a),
            Expr::Log10(a) => Self::write_latex_fn(buf, "\\log_{10}", a),
            Expr::Sqrt(a) => {
                buf.push_str("\\sqrt{");
                a.write_latex(buf);
                buf.push('}');
            }
            Expr::Abs(a) => {
                buf.push_str("\\left|");
                a.write_latex(buf);
                buf.push_str("\\right|");
            }
            Expr::Heaviside(a) => Self::write_latex_fn(buf, "H", a),
            Expr::Clamp(val, lo, hi) => {
                buf.push_str("\\operatorname{clamp}\\left(");
                val.write_latex(buf);
                buf.push_str(", ");
                lo.write_latex(buf);
                buf.push_str(", ");
                hi.write_latex(buf);
                buf.push_str("\\right)");
            }
            Expr::Branch(q, a, b) => {
                buf.push_str("\\operatorname{branch}\\left(");
                q.write_latex(buf);
                buf.push_str(", ");
                a.write_latex(buf);
                buf.push_str(", ");
                b.write_latex(buf);
                buf.push_str("\\right)");
            }
            Expr::Func { name, args, .. } => {
                let escaped = name.replace('_', "\\_");
                buf.push_str(&format!("\\operatorname{{{escaped}}}\\left("));
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 { buf.push_str(", "); }
                    arg.write_latex(buf);
                }
                buf.push_str("\\right)");
            }
        }
    }

    fn write_latex_fn(buf: &mut String, name: &str, arg: &Expr) {
        buf.push_str(name);
        buf.push_str("\\left(");
        arg.write_latex(buf);
        buf.push_str("\\right)");
    }

    // --- Rust code output ---

    /// Generate Rust source code for this expression.
    ///
    /// The `float_type` parameter (e.g. `"f64"`) controls numeric literal
    /// suffixes. Pass an empty string to omit type suffixes.
    pub fn to_rust(&self, float_type: &str) -> String {
        let mut buf = String::new();
        self.write_rust(&mut buf, float_type, 0);
        buf
    }

    // Precedence levels (matching Rust):
    // 0 = top level
    // 5 = Add, Sub
    // 6 = Mul, Div
    // 7 = Unary Neg
    // 8 = Atoms, method calls (never need parens)
    fn prec(&self) -> u8 {
        match self {
            Expr::Add(_, _) | Expr::Sub(_, _) => 5,
            Expr::Mul(_, _) | Expr::Div(_, _) => 6,
            Expr::Neg(_) => 7,
            _ => 8,
        }
    }

    fn write_rust(&self, buf: &mut String, ft: &str, parent_prec: u8) {
        // Type-inferred context: fold symbol-free subtrees to a single
        // literal. NamedConsts (pi, epsilon, ...) deliberately survive
        // simplification, so trees like epsilon^2 reach codegen intact --
        // but an unsuffixed literal cannot take method calls
        // (`2.2e-16.powf(2.0)` is an ambiguous numeric type), so the
        // subtree must become one literal that infers its precision from
        // the surrounding expression.
        if ft.is_empty()
            && !matches!(self, Expr::Const(_) | Expr::Sym(_))
            && let e = crate::E::new(self.clone())
            && e.symbols().is_empty()
            && let Ok(v) = e.eval(&std::collections::HashMap::new())
            && v.is_finite() {
                let lit = rust_float_literal(v, "");
                if v < 0.0 && parent_prec > 5 {
                    buf.push('(');
                    buf.push_str(&lit);
                    buf.push(')');
                } else {
                    buf.push_str(&lit);
                }
                return;
        }

        let my_prec = self.prec();
        // Need parens when our precedence is lower than parent's,
        // or for Sub/Div right-hand side at same precedence (non-associative)
        let need_parens = my_prec < parent_prec;
        if need_parens { buf.push('('); }

        match self {
            Expr::Sym(name) => buf.push_str(name),
            Expr::Const(v) => buf.push_str(&rust_float_literal(*v, ft)),
            Expr::NamedConst { rust_f32, rust_f64, value, .. } => {
                if ft == "f32" {
                    buf.push_str(rust_f32);
                } else if ft.is_empty() {
                    // Type-inferred context (macro-generated constraint
                    // code emits unsuffixed literals throughout): emit the
                    // numeric value so it adopts the surrounding
                    // expression's precision. Exact for pi/e; for epsilon
                    // an f32 context gets the f64 magnitude, which keeps
                    // the safe_* derivative guards finite.
                    buf.push_str(&rust_float_literal(*value, ""));
                } else {
                    buf.push_str(rust_f64);
                }
            }
            Expr::Neg(a) => {
                buf.push('-');
                a.write_rust(buf, ft, 7);
            }
            Expr::Add(a, b) => {
                a.write_rust(buf, ft, 5);
                buf.push_str(" + ");
                b.write_rust(buf, ft, 6); // right side: need parens for sub at same level
            }
            Expr::Sub(a, b) => {
                a.write_rust(buf, ft, 5);
                buf.push_str(" - ");
                b.write_rust(buf, ft, 6); // right side of sub: parens for add/sub
            }
            Expr::Mul(a, b) => {
                a.write_rust(buf, ft, 6);
                buf.push_str(" * ");
                b.write_rust(buf, ft, 7); // right side: need parens for div at same level
            }
            Expr::Div(a, b) => {
                a.write_rust(buf, ft, 6);
                buf.push_str(" / ");
                b.write_rust(buf, ft, 7); // right side of div: parens for mul/div
            }
            Expr::Pow(a, b) => {
                a.write_rust(buf, ft, 8);
                buf.push_str(".powf(");
                b.write_rust(buf, ft, 0);
                buf.push(')');
            }
            Expr::Sin(a) => Self::write_rust_method(buf, ft, a, "sin"),
            Expr::Cos(a) => Self::write_rust_method(buf, ft, a, "cos"),
            Expr::Tan(a) => Self::write_rust_method(buf, ft, a, "tan"),
            Expr::Asin(a) => Self::write_rust_method(buf, ft, a, "asin"),
            Expr::Acos(a) => Self::write_rust_method(buf, ft, a, "acos"),
            Expr::Atan(a) => Self::write_rust_method(buf, ft, a, "atan"),
            Expr::Atan2(y, x) => {
                y.write_rust(buf, ft, 8);
                buf.push_str(".atan2(");
                x.write_rust(buf, ft, 0);
                buf.push(')');
            }
            Expr::Sinh(a) => Self::write_rust_method(buf, ft, a, "sinh"),
            Expr::Cosh(a) => Self::write_rust_method(buf, ft, a, "cosh"),
            Expr::Tanh(a) => Self::write_rust_method(buf, ft, a, "tanh"),
            Expr::Exp(a) => Self::write_rust_method(buf, ft, a, "exp"),
            Expr::Ln(a) => Self::write_rust_method(buf, ft, a, "ln"),
            Expr::Log2(a) => Self::write_rust_method(buf, ft, a, "log2"),
            Expr::Log10(a) => Self::write_rust_method(buf, ft, a, "log10"),
            Expr::Sqrt(a) => Self::write_rust_method(buf, ft, a, "sqrt"),
            Expr::Abs(a) => Self::write_rust_method(buf, ft, a, "abs"),
            Expr::Heaviside(a) => Self::write_rust_method(buf, ft, a, "heaviside"),
            Expr::Clamp(val, lo, hi) => {
                val.write_rust(buf, ft, 8);
                buf.push_str(".clamp(");
                lo.write_rust(buf, ft, 0);
                buf.push_str(", ");
                hi.write_rust(buf, ft, 0);
                buf.push(')');
            }
            Expr::Branch(q, a, b) => {
                // q >= 0 ? a : b, lazy (only the taken side runs). The `0.0`
                // infers to the expression's float type from the comparison.
                buf.push_str("(if ");
                q.write_rust(buf, ft, 0);
                buf.push_str(" >= 0.0 { ");
                a.write_rust(buf, ft, 0);
                buf.push_str(" } else { ");
                b.write_rust(buf, ft, 0);
                buf.push_str(" })");
            }
            Expr::Func { name, params, kind, args } => {
                if let Some(body) = kind.body() {
                    // Symbolic: inline the expanded body.
                    // Identity functions force parentheses to preserve eval order.
                    let prec = if name == "identity" { 8 } else { parent_prec };
                    crate::expand_func(params, body, args).write_rust(buf, ft, prec);
                } else if let crate::FuncKind::Extern { call_path, .. } = kind {
                    // Extern: emit function call
                    buf.push_str(call_path);
                    buf.push('(');
                    for (i, arg) in args.iter().enumerate() {
                        if i > 0 { buf.push_str(", "); }
                        arg.write_rust(buf, ft, 0);
                    }
                    buf.push(')');
                }
                return; // already wrote, skip trailing paren logic
            }
        }

        if need_parens { buf.push(')'); }
    }

    fn write_rust_method(buf: &mut String, ft: &str, arg: &Expr, method: &str) {
        arg.write_rust(buf, ft, 8);
        buf.push('.');
        buf.push_str(method);
        buf.push_str("()");
    }
}
