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
            Expr::Func { params, body, args, .. } => {
                super::expand_func(params, body, args).eval(vars)
            }
        }
    }

    /// Substitute all occurrences of the named variable with `replacement`.
    ///
    /// Returns a new expression with the substitution applied throughout.
    pub fn subs(&self, var: &str, replacement: &E) -> E {
        match self {
            Expr::Sym(name) if name == var => replacement.clone(),
            Expr::Sym(_) | Expr::Const(_) => E::new(self.clone()),
            Expr::Neg(a) => -a.subs(var, replacement),
            Expr::Add(a, b) => a.subs(var, replacement) + b.subs(var, replacement),
            Expr::Sub(a, b) => a.subs(var, replacement) - b.subs(var, replacement),
            Expr::Mul(a, b) => a.subs(var, replacement) * b.subs(var, replacement),
            Expr::Div(a, b) => a.subs(var, replacement) / b.subs(var, replacement),
            Expr::Pow(a, b) => E::new(Expr::Pow(a.subs(var, replacement), b.subs(var, replacement))),
            Expr::Sin(a) => E::new(Expr::Sin(a.subs(var, replacement))),
            Expr::Cos(a) => E::new(Expr::Cos(a.subs(var, replacement))),
            Expr::Tan(a) => E::new(Expr::Tan(a.subs(var, replacement))),
            Expr::Asin(a) => E::new(Expr::Asin(a.subs(var, replacement))),
            Expr::Acos(a) => E::new(Expr::Acos(a.subs(var, replacement))),
            Expr::Atan(a) => E::new(Expr::Atan(a.subs(var, replacement))),
            Expr::Atan2(y, x) => E::new(Expr::Atan2(y.subs(var, replacement), x.subs(var, replacement))),
            Expr::Sinh(a) => E::new(Expr::Sinh(a.subs(var, replacement))),
            Expr::Cosh(a) => E::new(Expr::Cosh(a.subs(var, replacement))),
            Expr::Tanh(a) => E::new(Expr::Tanh(a.subs(var, replacement))),
            Expr::Exp(a) => E::new(Expr::Exp(a.subs(var, replacement))),
            Expr::Ln(a) => E::new(Expr::Ln(a.subs(var, replacement))),
            Expr::Log2(a) => E::new(Expr::Log2(a.subs(var, replacement))),
            Expr::Log10(a) => E::new(Expr::Log10(a.subs(var, replacement))),
            Expr::Sqrt(a) => E::new(Expr::Sqrt(a.subs(var, replacement))),
            Expr::Abs(a) => E::new(Expr::Abs(a.subs(var, replacement))),
            Expr::Func { name, params, body, args } => {
                let new_args = args.iter().map(|a| a.subs(var, replacement)).collect();
                E::new(Expr::Func { name: name.clone(), params: params.clone(), body: body.clone(), args: new_args })
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
            Expr::Const(_) => {}
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
            | Expr::Sqrt(a) | Expr::Abs(a) => {
                a.collect_vars(set);
            }
            Expr::Func { args, .. } => {
                for arg in args { arg.collect_vars(set); }
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
