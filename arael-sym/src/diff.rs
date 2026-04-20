use super::{AsVarName, Expr, E, constant, sin, cos, cosh, sinh, tanh, exp, ln, sqrt, abs, pow};

impl Expr {
    /// Symbolically differentiate this expression with respect to a variable.
    ///
    /// Applies the chain rule, product rule, and quotient rule automatically.
    /// The result is simplified.
    pub fn diff(&self, var: impl AsVarName) -> E {
        let var = var.var_name();
        let zero = || constant(0.0);
        let one = || constant(1.0);
        let two = || constant(2.0);

        match self {
            Expr::Sym(name) => {
                if name == var { one() } else { zero() }
            }
            Expr::Const(_) | Expr::NamedConst { .. } => zero(),
            Expr::Neg(a) => {
                -a.diff(var)
            }
            Expr::Add(a, b) => {
                a.diff(var) + b.diff(var)
            }
            Expr::Sub(a, b) => {
                a.diff(var) - b.diff(var)
            }
            Expr::Mul(a, b) => {
                // product rule: a'*b + a*b'
                let da = a.diff(var);
                let db = b.diff(var);
                da * b.clone() + a.clone() * db
            }
            Expr::Div(a, b) => {
                // quotient rule: (a'*b - a*b') / b^2
                let da = a.diff(var);
                let db = b.diff(var);
                (da * b.clone() - a.clone() * db) / pow(b.clone(), two())
            }
            Expr::Pow(a, b) => {
                let da = a.diff(var);
                let db = b.diff(var);
                if matches!(b.as_ref(), Expr::Const(_)) {
                    // Power rule: n * a^(n-1) * a'
                    b.clone() * pow(a.clone(), b.clone() - constant(1.0)) * da
                } else if matches!(a.as_ref(), Expr::Const(_)) {
                    // Constant base: a^b * ln(a) * b'
                    pow(a.clone(), b.clone()) * ln(a.clone()) * db
                } else {
                    // General: a^b * (b' * ln(a) + b * a' / a)
                    let base = pow(a.clone(), b.clone());
                    base * (db * ln(a.clone()) + b.clone() * da / a.clone())
                }
            }
            Expr::Sin(a) => {
                cos(a.clone()) * a.diff(var)
            }
            Expr::Cos(a) => {
                -(sin(a.clone()) * a.diff(var))
            }
            Expr::Tan(a) => {
                // a' / cos(a)^2
                a.diff(var) / pow(cos(a.clone()), two())
            }
            Expr::Asin(a) => {
                // a' / sqrt(1 - a^2)
                a.diff(var) / sqrt(one() - pow(a.clone(), two()))
            }
            Expr::Acos(a) => {
                // -a' / sqrt(1 - a^2)
                -(a.diff(var) / sqrt(one() - pow(a.clone(), two())))
            }
            Expr::Atan(a) => {
                // a' / (1 + a^2)
                a.diff(var) / (one() + pow(a.clone(), two()))
            }
            Expr::Atan2(y, x) => {
                // (x*dy - y*dx) / (x^2 + y^2)
                let dy = y.diff(var);
                let dx = x.diff(var);
                (x.clone() * dy - y.clone() * dx) / (pow(x.clone(), two()) + pow(y.clone(), two()))
            }
            Expr::Sinh(a) => {
                cosh(a.clone()) * a.diff(var)
            }
            Expr::Cosh(a) => {
                sinh(a.clone()) * a.diff(var)
            }
            Expr::Tanh(a) => {
                // a' * (1 - tanh(a)^2)
                a.diff(var) * (one() - pow(tanh(a.clone()), two()))
            }
            Expr::Exp(a) => {
                exp(a.clone()) * a.diff(var)
            }
            Expr::Ln(a) => {
                // a' / a
                a.diff(var) / a.clone()
            }
            Expr::Log2(a) => {
                // a' / (a * ln(2))
                a.diff(var) / (a.clone() * ln(constant(2.0)))
            }
            Expr::Log10(a) => {
                // a' / (a * ln(10))
                a.diff(var) / (a.clone() * ln(constant(10.0)))
            }
            Expr::Sqrt(a) => {
                // a' / (2 * sqrt(a))
                a.diff(var) / (two() * sqrt(a.clone()))
            }
            Expr::Abs(a) => {
                // a * a' / |a|  (i.e., sign(a) * a')
                a.clone() * a.diff(var) / abs(a.clone())
            }
            Expr::Heaviside(_) => {
                // Derivative is 0 everywhere (pragmatic, not Dirac delta)
                zero()
            }
            Expr::Clamp(val, _, _) => {
                // Pass-through: derivative ignores the clamping
                val.diff(var)
            }
            Expr::Func { params, kind, args, .. } => {
                if let Some(body) = kind.auto_diff_body() {
                    // Auto-diff: expand body, differentiate
                    super::expand_func(params, body, args).diff(var)
                } else {
                    // Explicit derivs: chain rule df/dvar = sum_i(df/dp_i * dp_i/dvar)
                    let derivs = kind.derivs().unwrap();
                    let mut result = zero();
                    for (d, a) in derivs.iter().zip(args.iter()) {
                        let da = a.diff(var);
                        if !matches!(da.as_ref(), Expr::Const(v) if *v == 0.0) {
                            result = result + super::expand_func(params, d, args) * da;
                        }
                    }
                    result
                }
            }
        }
    }
}
