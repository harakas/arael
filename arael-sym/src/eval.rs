use std::collections::{BTreeSet, HashMap};
use super::{Expr, E};

impl Expr {
    /// Evaluate the expression numerically given variable bindings.
    ///
    /// Returns `Err` if any symbol in the expression is not bound in `vars`.
    pub fn eval(&self, vars: &HashMap<&str, f64>) -> Result<f64, String> {
        match self {
            Expr::Sym(name) => {
                vars.get(name.as_str()).copied()
                    .ok_or_else(|| format!("unbound symbol: {name}"))
            }
            Expr::Const(v) => Ok(*v),
            Expr::NamedConst { value, .. } => Ok(*value),
            Expr::Neg(a) => Ok(-a.eval(vars)?),
            Expr::Add(a, b) => Ok(a.eval(vars)? + b.eval(vars)?),
            Expr::Sub(a, b) => Ok(a.eval(vars)? - b.eval(vars)?),
            Expr::Mul(a, b) => Ok(a.eval(vars)? * b.eval(vars)?),
            Expr::Div(a, b) => Ok(a.eval(vars)? / b.eval(vars)?),
            Expr::Pow(a, b) => Ok(a.eval(vars)?.powf(b.eval(vars)?)),
            Expr::Sin(a) => Ok(a.eval(vars)?.sin()),
            Expr::Cos(a) => Ok(a.eval(vars)?.cos()),
            Expr::Tan(a) => Ok(a.eval(vars)?.tan()),
            Expr::Asin(a) => Ok(a.eval(vars)?.asin()),
            Expr::Acos(a) => Ok(a.eval(vars)?.acos()),
            Expr::Atan(a) => Ok(a.eval(vars)?.atan()),
            Expr::Atan2(y, x) => Ok(y.eval(vars)?.atan2(x.eval(vars)?)),
            Expr::Sinh(a) => Ok(a.eval(vars)?.sinh()),
            Expr::Cosh(a) => Ok(a.eval(vars)?.cosh()),
            Expr::Tanh(a) => Ok(a.eval(vars)?.tanh()),
            Expr::Exp(a) => Ok(a.eval(vars)?.exp()),
            Expr::Ln(a) => Ok(a.eval(vars)?.ln()),
            Expr::Log2(a) => Ok(a.eval(vars)?.log2()),
            Expr::Log10(a) => Ok(a.eval(vars)?.log10()),
            Expr::Sqrt(a) => Ok(a.eval(vars)?.sqrt()),
            Expr::Abs(a) => Ok(a.eval(vars)?.abs()),
            Expr::Heaviside(a) => {
                let v = a.eval(vars)?;
                // Same branch sense as the runtime utils::heaviside so
                // interpreted and compiled code agree -- including on NaN
                // (>= is false for NaN, so heaviside(NaN) = 0).
                Ok(if v >= 0.0 { 1.0 } else { 0.0 })
            }
            Expr::Clamp(val, lo, hi) => {
                let v = val.eval(vars)?;
                let l = lo.eval(vars)?;
                let h = hi.eval(vars)?;
                Ok(v.clamp(l, h))
            }
            Expr::Branch(q, a, b) => {
                // q >= 0 ? a : b, with the same >= 0 / NaN sense as Heaviside
                // (NaN is not >= 0, so it selects b). Only the taken side is
                // evaluated, matching the generated if/else.
                if q.eval(vars)? >= 0.0 { a.eval(vars) } else { b.eval(vars) }
            }
            Expr::Select { index, arms, default } => {
                // Only the taken arm is evaluated, matching the generated match.
                match select_arm(index.eval(vars)?, arms.len(), default.is_some())? {
                    Some(i) => arms[i].eval(vars),
                    None => default.as_ref().unwrap().eval(vars),
                }
            }
            Expr::Func { params, kind, args, .. } => {
                if let Some(f) = kind.eval_fn() {
                    let vals: Result<Vec<f64>, _> = args.iter().map(|a| a.eval(vars)).collect();
                    Ok(f(&vals?))
                } else {
                    // INVARIANT: eval_fn() is None only for the Symbolic variants,
                    // which always carry a body (Extern is the only eval_fn kind).
                    let body = kind.body().expect("FuncKind must have body or eval_fn");
                    super::expand_func(params, body, args).eval(vars)
                }
            }
        }
    }

    /// Substitute all occurrences of the named variable with `replacement`.
    ///
    /// `var` can be any [`AsVarName`](crate::AsVarName) -- a `&str`, a `String`, or an
    /// [`E`] handle wrapping a `Sym` node. Returns a new expression
    /// with the substitution applied throughout.
    pub fn subs(&self, var: impl crate::AsVarName, replacement: &E) -> E {
        self.subs_by_name(var.var_name(), replacement)
    }

    fn subs_by_name(&self, var: &str, replacement: &E) -> E {
        match self {
            Expr::Sym(name) if name == var => replacement.clone(),
            Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => E::new(self.clone()),
            Expr::Neg(a) => -a.subs_by_name(var, replacement),
            Expr::Add(a, b) => a.subs_by_name(var, replacement) + b.subs_by_name(var, replacement),
            Expr::Sub(a, b) => a.subs_by_name(var, replacement) - b.subs_by_name(var, replacement),
            Expr::Mul(a, b) => a.subs_by_name(var, replacement) * b.subs_by_name(var, replacement),
            Expr::Div(a, b) => a.subs_by_name(var, replacement) / b.subs_by_name(var, replacement),
            Expr::Pow(a, b) => E::new(Expr::Pow(a.subs_by_name(var, replacement), b.subs_by_name(var, replacement))),
            Expr::Sin(a) => E::new(Expr::Sin(a.subs_by_name(var, replacement))),
            Expr::Cos(a) => E::new(Expr::Cos(a.subs_by_name(var, replacement))),
            Expr::Tan(a) => E::new(Expr::Tan(a.subs_by_name(var, replacement))),
            Expr::Asin(a) => E::new(Expr::Asin(a.subs_by_name(var, replacement))),
            Expr::Acos(a) => E::new(Expr::Acos(a.subs_by_name(var, replacement))),
            Expr::Atan(a) => E::new(Expr::Atan(a.subs_by_name(var, replacement))),
            Expr::Atan2(y, x) => E::new(Expr::Atan2(y.subs_by_name(var, replacement), x.subs_by_name(var, replacement))),
            Expr::Sinh(a) => E::new(Expr::Sinh(a.subs_by_name(var, replacement))),
            Expr::Cosh(a) => E::new(Expr::Cosh(a.subs_by_name(var, replacement))),
            Expr::Tanh(a) => E::new(Expr::Tanh(a.subs_by_name(var, replacement))),
            Expr::Exp(a) => E::new(Expr::Exp(a.subs_by_name(var, replacement))),
            Expr::Ln(a) => E::new(Expr::Ln(a.subs_by_name(var, replacement))),
            Expr::Log2(a) => E::new(Expr::Log2(a.subs_by_name(var, replacement))),
            Expr::Log10(a) => E::new(Expr::Log10(a.subs_by_name(var, replacement))),
            Expr::Sqrt(a) => E::new(Expr::Sqrt(a.subs_by_name(var, replacement))),
            Expr::Abs(a) => E::new(Expr::Abs(a.subs_by_name(var, replacement))),
            Expr::Heaviside(a) => E::new(Expr::Heaviside(a.subs_by_name(var, replacement))),
            Expr::Clamp(a, lo, hi) => E::new(Expr::Clamp(a.subs_by_name(var, replacement), lo.subs_by_name(var, replacement), hi.subs_by_name(var, replacement))),
            Expr::Branch(q, a, b) => E::new(Expr::Branch(q.subs_by_name(var, replacement), a.subs_by_name(var, replacement), b.subs_by_name(var, replacement))),
            Expr::Select { index, arms, default } => E::new(Expr::Select {
                index: index.subs_by_name(var, replacement),
                arms: arms.iter().map(|a| a.subs_by_name(var, replacement)).collect(),
                default: default.as_ref().map(|d| d.subs_by_name(var, replacement)),
            }),
            Expr::Func { name, params, kind, args } => {
                let new_args = args.iter().map(|a| a.subs_by_name(var, replacement)).collect();
                // Captured symbols: substitute inside the body/derivs too,
                // unless the name is shadowed by a param (bound inside).
                let new_kind = if params.iter().any(|p| p == var) {
                    kind.clone()
                } else {
                    match kind {
                        crate::FuncKind::Symbolic { body } => crate::FuncKind::Symbolic {
                            body: body.subs_by_name(var, replacement),
                        },
                        crate::FuncKind::SymbolicDerivs { body, derivs } => crate::FuncKind::SymbolicDerivs {
                            body: body.subs_by_name(var, replacement),
                            derivs: derivs.iter().map(|d| d.subs_by_name(var, replacement)).collect(),
                        },
                        crate::FuncKind::Extern { .. } => kind.clone(),
                    }
                };
                E::new(Expr::Func { name: name.clone(), params: params.clone(), kind: new_kind, args: new_args })
            }
        }
    }

    /// Collect all free (unbound) variable names in the expression.
    ///
    /// Returns a sorted set of variable name strings.
    pub fn free_vars(&self) -> BTreeSet<String> {
        let mut set = BTreeSet::new();
        self.collect_vars(&mut set);
        set
    }

    fn collect_vars(&self, set: &mut BTreeSet<String>) {
        match self {
            Expr::Sym(name) => { set.insert(name.clone()); }
            Expr::Const(_) | Expr::NamedConst { .. } => {}
            Expr::Neg(a) => a.collect_vars(set),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
            | Expr::Div(a, b) | Expr::Pow(a, b) | Expr::Atan2(a, b) => {
                a.collect_vars(set);
                b.collect_vars(set);
            }
            Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
            | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
            | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
            | Expr::Exp(a) | Expr::Ln(a) | Expr::Log2(a) | Expr::Log10(a)
            | Expr::Sqrt(a) | Expr::Abs(a)
            | Expr::Heaviside(a) => {
                a.collect_vars(set);
            }
            Expr::Clamp(a, lo, hi) | Expr::Branch(a, lo, hi) => {
                a.collect_vars(set);
                lo.collect_vars(set);
                hi.collect_vars(set);
            }
            Expr::Select { index, arms, default } => {
                index.collect_vars(set);
                for a in arms { a.collect_vars(set); }
                if let Some(d) = default { d.collect_vars(set); }
            }
            Expr::Func { name: _, params, kind, args } => {
                for arg in args { arg.collect_vars(set); }
                // Captured symbols: body may reference symbols beyond its
                // params; they are free variables of this expression.
                if let Some(body) = kind.body() {
                    let mut inner = BTreeSet::new();
                    body.collect_vars(&mut inner);
                    for n in inner {
                        if !params.contains(&n) { set.insert(n); }
                    }
                }
            }
        }
    }

    /// Differentiate with respect to multiple variables, returning a vector.
    ///
    /// Equivalent to calling [`Expr::diff`] for each variable in order.
    pub fn diff_all(&self, vars: &[&str]) -> Vec<E> {
        vars.iter().map(|v| self.diff(v)).collect()
    }
}

/// Resolve a select index the way the generated `match` does: the value
/// must be an exact integer (a NaN or fractional index is an error, as the
/// runtime `SelectIndex` conversion panics on it); `0..n` names an arm, any
/// other value takes the default (`None`) or is an error when there is none.
pub(crate) fn select_arm(v: f64, n: usize, has_default: bool) -> Result<Option<usize>, String> {
    if !v.is_finite() || v.fract() != 0.0 {
        return Err(format!("select index {v} is not an integer"));
    }
    if v >= 0.0 && v < n as f64 {
        return Ok(Some(v as usize));
    }
    if has_default {
        Ok(None)
    } else {
        Err(format!("select index {v} out of range 0..{n}"))
    }
}
