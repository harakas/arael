use super::{AsVarName, Expr, E, constant, sin, cos, cosh, sinh, tanh, exp, ln, sqrt, abs, branch};

// The derivative is built raw and canonicalized by a single simplify() at the
// end of `diff`, so these constructors only need to drop the trivial 0/1
// identities (which also keeps zero-derivative terms from bloating the tree);
// ordering and flattening are left to that final pass.

fn mul_fast(a: E, b: E) -> E {
    if let Expr::Const(v) = a.as_ref() {
        if *v == 0.0 { return constant(0.0); }
        if *v == 1.0 { return b; }
    }
    if let Expr::Const(v) = b.as_ref() {
        if *v == 0.0 { return constant(0.0); }
        if *v == 1.0 { return a; }
    }
    E::new(Expr::Mul(a, b))
}

fn add_fast(a: E, b: E) -> E {
    if let Expr::Const(v) = a.as_ref() { if *v == 0.0 { return b; } }
    if let Expr::Const(v) = b.as_ref() { if *v == 0.0 { return a; } }
    E::new(Expr::Add(a, b))
}

fn sub_fast(a: E, b: E) -> E {
    if let Expr::Const(v) = b.as_ref() { if *v == 0.0 { return a; } }
    E::new(Expr::Sub(a, b))
}

fn div_fast(a: E, b: E) -> E {
    if let Expr::Const(v) = a.as_ref() { if *v == 0.0 { return constant(0.0); } }
    if let Expr::Const(v) = b.as_ref() { if *v == 1.0 { return a; } }
    E::new(Expr::Div(a, b))
}

fn pow_fast(a: E, n: f64) -> E {
    E::new(Expr::Pow(a, constant(n)))
}

fn neg_fast(a: E) -> E {
    if let Expr::Const(v) = a.as_ref() { return constant(-v); }
    E::new(Expr::Neg(a))
}

impl E {
    /// Symbolically differentiate this expression with respect to a variable.
    ///
    /// Applies the chain rule, product rule, and quotient rule automatically.
    /// The result is simplified.
    pub fn diff(&self, var: impl AsVarName) -> E {
        self.diff_var(var.var_name()).simplify()
    }

    fn diff_var(&self, var: &str) -> E {
        let zero = || constant(0.0);
        let one = || constant(1.0);
        let two = || constant(2.0);

        match &*self.0 {
            Expr::Sym(name) => {
                if name == var { one() } else { zero() }
            }
            Expr::Const(_) | Expr::NamedConst { .. } => zero(),
            Expr::Neg(a) => {
                // -a'
                neg_fast(a.diff_var(var))
            }
            Expr::Add(a, b) => {
                // a' + b'
                add_fast(a.diff_var(var), b.diff_var(var))
            }
            Expr::Sub(a, b) => {
                // a' - b'
                sub_fast(a.diff_var(var), b.diff_var(var))
            }
            Expr::Mul(a, b) => {
                // product rule: a'*b + a*b'
                let da = a.diff_var(var);
                let db = b.diff_var(var);
                add_fast(mul_fast(da, b.clone()), mul_fast(a.clone(), db))
            }
            Expr::Div(a, b) => {
                // quotient rule: (a'*b - a*b') / b^2
                let da = a.diff_var(var);
                let db = b.diff_var(var);
                div_fast(sub_fast(mul_fast(da, b.clone()), mul_fast(a.clone(), db)), pow_fast(b.clone(), 2.0))
            }
            Expr::Pow(a, b) => {
                let da = a.diff_var(var);
                let db = b.diff_var(var);
                if matches!(b.as_ref(), Expr::Const(_)) {
                    // power rule: b * a^(b-1) * a'
                    mul_fast(mul_fast(b.clone(), E::new(Expr::Pow(a.clone(), sub_fast(b.clone(), constant(1.0))))), da)
                } else if matches!(a.as_ref(), Expr::Const(_)) {
                    // constant base: a^b * ln(a) * b'
                    mul_fast(mul_fast(E::new(Expr::Pow(a.clone(), b.clone())), ln(a.clone())), db)
                } else {
                    // general: a^b * (b' * ln(a) + b * a' / a)
                    let base = E::new(Expr::Pow(a.clone(), b.clone()));
                    mul_fast(base, add_fast(mul_fast(db, ln(a.clone())), div_fast(mul_fast(b.clone(), da), a.clone())))
                }
            }
            Expr::Sin(a) => {
                // cos(a) * a'
                mul_fast(cos(a.clone()), a.diff_var(var))
            }
            Expr::Cos(a) => {
                // -sin(a) * a'
                neg_fast(mul_fast(sin(a.clone()), a.diff_var(var)))
            }
            Expr::Tan(a) => {
                // a' / cos(a)^2
                div_fast(a.diff_var(var), pow_fast(cos(a.clone()), 2.0))
            }
            Expr::Asin(a) => {
                // a' / sqrt(1 - a^2)
                div_fast(a.diff_var(var), sqrt(sub_fast(one(), pow_fast(a.clone(), 2.0))))
            }
            Expr::Acos(a) => {
                // -a' / sqrt(1 - a^2)
                neg_fast(div_fast(a.diff_var(var), sqrt(sub_fast(one(), pow_fast(a.clone(), 2.0)))))
            }
            Expr::Atan(a) => {
                // a' / (1 + a^2)
                div_fast(a.diff_var(var), add_fast(one(), pow_fast(a.clone(), 2.0)))
            }
            Expr::Atan2(y, x) => {
                // (x*y' - y*x') / (x^2 + y^2)
                let dy = y.diff_var(var);
                let dx = x.diff_var(var);
                div_fast(sub_fast(mul_fast(x.clone(), dy), mul_fast(y.clone(), dx)), add_fast(pow_fast(x.clone(), 2.0), pow_fast(y.clone(), 2.0)))
            }
            Expr::Sinh(a) => {
                // cosh(a) * a'
                mul_fast(cosh(a.clone()), a.diff_var(var))
            }
            Expr::Cosh(a) => {
                // sinh(a) * a'
                mul_fast(sinh(a.clone()), a.diff_var(var))
            }
            Expr::Tanh(a) => {
                // a' * (1 - tanh(a)^2)
                mul_fast(a.diff_var(var), sub_fast(one(), pow_fast(tanh(a.clone()), 2.0)))
            }
            Expr::Exp(a) => {
                // exp(a) * a'
                mul_fast(exp(a.clone()), a.diff_var(var))
            }
            Expr::Ln(a) => {
                // a' / a
                div_fast(a.diff_var(var), a.clone())
            }
            Expr::Log2(a) => {
                // a' / (a * ln(2))
                div_fast(a.diff_var(var), mul_fast(a.clone(), ln(constant(2.0))))
            }
            Expr::Log10(a) => {
                // a' / (a * ln(10))
                div_fast(a.diff_var(var), mul_fast(a.clone(), ln(constant(10.0))))
            }
            Expr::Sqrt(a) => {
                // a' / (2 * sqrt(a))
                div_fast(a.diff_var(var), mul_fast(two(), sqrt(a.clone())))
            }
            Expr::Abs(a) => {
                // a * a' / |a|
                div_fast(mul_fast(a.clone(), a.diff_var(var)), abs(a.clone()))
            }
            Expr::Heaviside(_) => {
                zero()
            }
            Expr::Clamp(val, _, _) => {
                val.diff_var(var)
            }
            Expr::Branch(q, a, b) => {
                branch(q.clone(), a.diff_var(var), b.diff_var(var))
            }
            Expr::Select { index, arms, default } => {
                // The index is piecewise constant: the derivative is the
                // taken arm's, selected by the same index.
                crate::select(
                    index.clone(),
                    arms.iter().map(|a| a.diff_var(var)).collect(),
                    default.as_ref().map(|d| d.diff_var(var)))
            }
            Expr::Func { name, params, kind, args } => {
                // cached() is a STICKY barrier: d(cached(g))/dx = cached(dg/dx).
                if name == "cached" && args.len() == 1 {
                    crate::cached(args[0].diff_var(var))
                } else if let Some(body) = kind.auto_diff_body() {
                    super::expand_func(params, body, args).diff_var(var)
                } else {
                    // Explicit derivs: df/dvar = sum_i(df/dp_i * dp_i/dvar)
                    let derivs = kind.derivs().unwrap();
                    let mut acc = zero();
                    for (d, a) in derivs.iter().zip(args.iter()) {
                        let da = a.diff_var(var);
                        if !matches!(da.as_ref(), Expr::Const(v) if *v == 0.0) {
                            acc = add_fast(acc, mul_fast(super::expand_func(params, d, args), da));
                        }
                    }
                    acc
                }
            }
        }
    }
}

impl Expr {
    /// Differentiate a bare `Expr`. Wraps it in an [`E`] and defers to
    /// [`E::diff`], so callers holding an `Expr` rather than an `E` keep working.
    pub fn diff(&self, var: impl AsVarName) -> E {
        E::new(self.clone()).diff(var)
    }
}
