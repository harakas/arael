// Model validation report types. The checks live on
// [`LmProblem`](crate::simple_lm::LmProblem): `validate()` runs every
// check and returns a `Diagnostic`; `check_gradients()` /
// `numeric_gradient()` are the standalone gradient pieces.

/// The report of a validation pass: every issue found, in one sweep.
/// Empty means the model passed every check that ran.
#[derive(Debug, Default)]
pub struct Diagnostic {
    /// Everything found, in discovery order.
    pub issues: Vec<Issue>,
}

impl Diagnostic {
    /// No issues found.
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.issues.is_empty() {
            return write!(f, "model is clean");
        }
        for (i, issue) in self.issues.iter().enumerate() {
            if i > 0 {
                writeln!(f)?;
            }
            write!(f, "{}", issue)?;
        }
        Ok(())
    }
}

/// One finding of a validation pass. Parameter indices refer to the
/// flat serialized vector -- the same indices solve failures report.
#[derive(Debug, Clone, PartialEq)]
pub enum Issue {
    /// A parameter serialized to NaN or infinity.
    NonFiniteParam {
        /// Scalar index in the flat parameter vector.
        param: usize,
        /// The offending value.
        value: f64,
    },
    /// No constraint curvature reaches this parameter: its Gauss-Newton
    /// Hessian diagonal is structurally absent or numerically zero at
    /// the current state. A solve would fail with
    /// `DegenerateDiagonal { fault: Zero }`.
    UnconstrainedParam {
        /// Scalar index in the flat parameter vector.
        param: usize,
    },
    /// The Hessian diagonal is negative or NaN at this parameter -- the
    /// assembly is poisoned (`J^T J`'s diagonal is a sum of squares, so
    /// neither can happen in healthy arithmetic).
    BadDiagonal {
        /// Scalar index in the flat parameter vector.
        param: usize,
        /// The offending diagonal value.
        value: f64,
    },
    /// A `Ref` no longer resolves in its collection -- the slot was
    /// removed or replaced since the ref was issued. Accessing it
    /// during a solve panics.
    StaleRef {
        /// Where the ref lives, e.g. `edges[3].a`.
        path: String,
    },
    /// The assembled gradient disagrees with central finite differences
    /// of `calc_cost` -- a wrong hand-declared derivative
    /// (`#[arael::function]` `derivs`, a `deriv =` cache) or a broken
    /// assembly path.
    GradientMismatch {
        /// Scalar index in the flat parameter vector.
        param: usize,
        /// The assembled (analytic) gradient component.
        analytic: f64,
        /// The finite-difference estimate.
        numeric: f64,
    },
}

impl std::fmt::Display for Issue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Issue::NonFiniteParam { param, value } => {
                write!(f, "param {}: non-finite value {}", param, value)
            }
            Issue::UnconstrainedParam { param } => {
                write!(f, "param {}: no constraint reaches it (zero Hessian diagonal)", param)
            }
            Issue::BadDiagonal { param, value } => {
                write!(f, "param {}: poisoned Hessian diagonal {}", param, value)
            }
            Issue::StaleRef { path } => {
                write!(f, "{}: stale Ref (its slot was removed or replaced)", path)
            }
            Issue::GradientMismatch { param, analytic, numeric } => {
                write!(f, "param {}: gradient mismatch, assembled {} vs finite-difference {}",
                    param, analytic, numeric)
            }
        }
    }
}
