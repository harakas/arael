//! The error type of evaluation and of the function bag.

use std::fmt;
use crate::parse::ParseError;

/// What evaluating an expression, registering a function in a
/// [`FunctionBag`](crate::FunctionBag) or calling one can fail with.
///
/// Parsing has its own [`ParseError`], which carries the position in the
/// text; it is wrapped here as [`Parse`](Self::Parse) (with a `From`
/// impl) so one `?` covers a parse followed by an eval.
#[derive(Debug, Clone, PartialEq)]
pub enum SymError {
    /// `eval` met a symbol that `vars` does not bind.
    UnboundSymbol(String),
    /// A select index that is not a finite integer.
    SelectIndexNotInteger(f64),
    /// A select index outside `0..arms`, with no default arm to take.
    SelectIndexOutOfRange { value: f64, arms: usize },
    /// An extern function that runs only in generated code was evaluated.
    NoEval(String),
    /// A derivative that does not exist was evaluated; see
    /// [`no_derivative`](crate::no_derivative).
    NoDerivative { of: String, why: String },
    /// A call with the wrong number of arguments.
    Arity { function: String, expected: usize, got: usize },
    /// A call whose arguments have the wrong shape, or an eval fn that
    /// returned fewer partial derivatives than were asked for.
    BadCall { function: String, message: String },
    /// A `FunctionBag::add*` method was given something other than a
    /// function call; `source` names the method.
    NotAFunction { source: String },
    /// The text did not parse.
    Parse(ParseError),
}

impl SymError {
    pub(crate) fn bad_call(function: impl Into<String>, message: impl Into<String>) -> Self {
        SymError::BadCall { function: function.into(), message: message.into() }
    }
}

impl fmt::Display for SymError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SymError::UnboundSymbol(name) => write!(f, "unbound symbol: {name}"),
            SymError::SelectIndexNotInteger(v) => write!(f, "select index {v} is not an integer"),
            SymError::SelectIndexOutOfRange { value, arms } =>
                write!(f, "select index {value} out of range 0..{arms}"),
            SymError::NoEval(name) =>
                write!(f, "{name} has no eval fn: it evaluates only where generated code calls it"),
            SymError::NoDerivative { of, why } => write!(f, "no derivative of {of}: {why}"),
            SymError::Arity { function, expected, got } =>
                write!(f, "{function} expects {expected} argument(s), got {got}"),
            SymError::BadCall { function, message } => write!(f, "{function} {message}"),
            SymError::NotAFunction { source } =>
                write!(f, "{source}: expected Expr::Func, got a different expression"),
            SymError::Parse(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for SymError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SymError::Parse(e) => Some(e),
            _ => None,
        }
    }
}

impl From<ParseError> for SymError {
    fn from(e: ParseError) -> Self {
        SymError::Parse(e)
    }
}
