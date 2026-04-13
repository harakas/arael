use super::*;
use std::fmt;

/// Error type for expression parsing.
///
/// Contains the byte position of the error and a human-readable message.
#[derive(Debug, Clone)]
pub struct ParseError {
    /// Byte offset in the input where the error occurred.
    pub pos: usize,
    /// Human-readable error description.
    pub msg: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parse error at position {}: {}", self.pos, self.msg)
    }
}

impl std::error::Error for ParseError {}

// --- Tokens ---

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Number(f64),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    LParen,
    RParen,
    Comma,
    Eof,
}

struct Lexer {
    chars: Vec<char>,
    pos: usize,
}

impl Lexer {
    fn new(input: &str) -> Self {
        Lexer { chars: input.chars().collect(), pos: 0 }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.chars.len() && self.chars[self.pos].is_ascii_whitespace() {
            self.pos += 1;
        }
    }

    fn next_token(&mut self) -> Result<(Token, usize), ParseError> {
        self.skip_whitespace();
        let start = self.pos;

        if self.pos >= self.chars.len() {
            return Ok((Token::Eof, start));
        }

        let ch = self.chars[self.pos];
        self.pos += 1;

        match ch {
            '+' => Ok((Token::Plus, start)),
            '-' => Ok((Token::Minus, start)),
            '*' => Ok((Token::Star, start)),
            '/' => Ok((Token::Slash, start)),
            '^' => Ok((Token::Caret, start)),
            '(' => Ok((Token::LParen, start)),
            ')' => Ok((Token::RParen, start)),
            ',' => Ok((Token::Comma, start)),
            c if c.is_ascii_digit() || c == '.' => {
                let mut s = String::new();
                s.push(c);
                while self.pos < self.chars.len()
                    && (self.chars[self.pos].is_ascii_digit() || self.chars[self.pos] == '.')
                {
                    s.push(self.chars[self.pos]);
                    self.pos += 1;
                }
                let val: f64 = s.parse().map_err(|_| ParseError {
                    pos: start,
                    msg: format!("invalid number: {s}"),
                })?;
                Ok((Token::Number(val), start))
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let mut s = String::new();
                s.push(c);
                while self.pos < self.chars.len()
                    && (self.chars[self.pos].is_ascii_alphanumeric() || self.chars[self.pos] == '_' || self.chars[self.pos] == '.')
                {
                    s.push(self.chars[self.pos]);
                    self.pos += 1;
                }
                Ok((Token::Ident(s), start))
            }
            _ => Err(ParseError {
                pos: start,
                msg: format!("unexpected character: '{ch}'"),
            }),
        }
    }
}

// --- Parser ---

struct Parser<'a> {
    tokens: Vec<(Token, usize)>,
    pos: usize,
    bag: Option<&'a FunctionBag>,
}

impl<'a> Parser<'a> {
    fn from_str(input: &str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        loop {
            let tok = lexer.next_token()?;
            let is_eof = tok.0 == Token::Eof;
            tokens.push(tok);
            if is_eof { break; }
        }
        Ok(Parser { tokens, pos: 0, bag: None })
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos].0
    }

    fn peek_pos(&self) -> usize {
        self.tokens[self.pos].1
    }

    fn advance(&mut self) -> &Token {
        let tok = &self.tokens[self.pos].0;
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
        tok
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ParseError> {
        if self.peek() == expected {
            self.advance();
            Ok(())
        } else {
            Err(ParseError {
                pos: self.peek_pos(),
                msg: format!("expected {expected:?}, got {:?}", self.peek()),
            })
        }
    }

    // expr = term (('+' | '-') term)*
    fn parse_expr(&mut self) -> Result<E, ParseError> {
        let mut left = self.parse_term()?;
        loop {
            match self.peek() {
                Token::Plus => { self.advance(); let right = self.parse_term()?; left = left + right; }
                Token::Minus => { self.advance(); let right = self.parse_term()?; left = left - right; }
                _ => break,
            }
        }
        Ok(left)
    }

    // term = unary (('*' | '/') unary)*
    fn parse_term(&mut self) -> Result<E, ParseError> {
        let mut left = self.parse_unary()?;
        loop {
            match self.peek() {
                Token::Star => { self.advance(); let right = self.parse_unary()?; left = left * right; }
                Token::Slash => { self.advance(); let right = self.parse_unary()?; left = left / right; }
                _ => break,
            }
        }
        Ok(left)
    }

    // unary = '-' unary | power
    fn parse_unary(&mut self) -> Result<E, ParseError> {
        if *self.peek() == Token::Minus {
            self.advance();
            let expr = self.parse_unary()?;
            Ok(-expr)
        } else {
            self.parse_power()
        }
    }

    // power = atom ('^' power)?   (right-associative)
    fn parse_power(&mut self) -> Result<E, ParseError> {
        let base = self.parse_atom()?;
        if *self.peek() == Token::Caret {
            self.advance();
            let exp = self.parse_unary()?;
            Ok(pow(base, exp))
        } else {
            Ok(base)
        }
    }

    // atom = NUMBER | IDENT | IDENT '(' args ')' | '(' expr ')'
    fn parse_atom(&mut self) -> Result<E, ParseError> {
        match self.peek().clone() {
            Token::Number(v) => {
                self.advance();
                Ok(constant(v))
            }
            Token::Ident(name) => {
                self.advance();
                if *self.peek() == Token::LParen {
                    // Function call
                    self.advance(); // consume '('
                    let mut args = Vec::new();
                    if *self.peek() != Token::RParen {
                        args.push(self.parse_expr()?);
                        while *self.peek() == Token::Comma {
                            self.advance();
                            args.push(self.parse_expr()?);
                        }
                    }
                    self.expect(&Token::RParen)?;
                    build_function_call(&name, args, self.bag)
                } else {
                    // Named constant or symbol
                    match name.as_str() {
                        "pi" => Ok(constant(std::f64::consts::PI)),
                        "e" => Ok(constant(std::f64::consts::E)),
                        _ => Ok(symbol(&name)),
                    }
                }
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            Token::Eof => Err(ParseError {
                pos: self.peek_pos(),
                msg: "unexpected end of input".to_string(),
            }),
            _ => Err(ParseError {
                pos: self.peek_pos(),
                msg: format!("unexpected token: {:?}", self.peek()),
            }),
        }
    }
}

fn build_function_call(name: &str, args: Vec<E>, bag: Option<&FunctionBag>) -> Result<E, ParseError> {
    // "H" is a parser-only alias for heaviside; normalize before lookup.
    let lookup_name = if name == "H" { "heaviside" } else { name };
    // User-defined functions in the bag take priority over built-ins.
    if let Some(bag) = bag
        && let Some(result) = bag.call(lookup_name, &args)
    {
        return result.map_err(|msg| ParseError { pos: 0, msg });
    }
    let fnref = crate::function_by_name(lookup_name).ok_or_else(|| ParseError {
        pos: 0,
        msg: format!("unknown function: {name}"),
    })?;
    match fnref {
        crate::FunctionRef::Unary(f) => {
            if args.len() != 1 {
                return Err(ParseError {
                    pos: 0,
                    msg: format!("{name} expects 1 argument, got {}", args.len()),
                });
            }
            Ok(f(args.into_iter().next().unwrap()))
        }
        crate::FunctionRef::Binary(f) => {
            if args.len() != 2 {
                return Err(ParseError {
                    pos: 0,
                    msg: format!("{name} expects 2 arguments, got {}", args.len()),
                });
            }
            let mut it = args.into_iter();
            Ok(f(it.next().unwrap(), it.next().unwrap()))
        }
        crate::FunctionRef::Ternary(f) => {
            if args.len() != 3 {
                return Err(ParseError {
                    pos: 0,
                    msg: format!("{name} expects 3 arguments, got {}", args.len()),
                });
            }
            let mut it = args.into_iter();
            Ok(f(it.next().unwrap(), it.next().unwrap(), it.next().unwrap()))
        }
    }
}

/// Parse a string into a symbolic expression, using only built-in
/// functions.
///
/// Supports standard infix notation with `+`, `-`, `*`, `/`, `^` (power),
/// parentheses, and function calls (`sin`, `cos`, `tan`, `asin`, `acos`,
/// `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log2`, `log10`,
/// `sqrt`, `abs`, `heaviside` (alias `H`), `clamp`, `pow`, `rad_diff`,
/// `rad_sum`, `safe_atan2`, `safe_sqrt`, `safe_asin`, `safe_acos`,
/// `identity`). See the full list in [`crate::FUNCTIONS`].
///
/// The identifiers `pi` and `e` are recognized as named constants.
/// All other identifiers become symbolic variables.
///
/// Use [`parse_with_functions`] to also recognise user-defined
/// functions (e.g. from [`FunctionBag::add_symbolic`]) during parsing.
///
/// # Errors
///
/// Returns a [`ParseError`] on invalid syntax, unknown functions, or
/// wrong argument counts.
pub fn parse(input: &str) -> Result<E, ParseError> {
    parse_with_functions(input, &FunctionBag::new())
}

/// Parse a string into a symbolic expression, consulting a
/// [`FunctionBag`] before the built-in function table.
///
/// Names in `bag` take priority over built-ins with the same name
/// (shadowing). If a function name is not in `bag`, the parser falls
/// back to [`crate::function_by_name`], so built-ins remain available
/// regardless of what's in the bag. Passing `&FunctionBag::new()`
/// (empty) is equivalent to calling [`parse`].
///
/// # Example
///
/// ```
/// use arael_sym::{parse_with_functions, FunctionBag, parse, constant, symbol};
/// use std::collections::HashMap;
///
/// let mut bag = FunctionBag::new();
/// // Register a user-defined function whose body uses `t` as a symbol
/// // to stand for the formal parameter.
/// bag.add_symbolic("sq", vec!["t".to_string()], parse("t*t").unwrap());
///
/// let e = parse_with_functions("sq(3)", &bag).unwrap();
/// let vars: HashMap<&str, f64> = HashMap::new();
/// assert_eq!(e.eval(&vars).unwrap(), 9.0);
/// ```
///
/// # Errors
///
/// As [`parse`], plus:
/// - Arity mismatch between a call-site and the bag's stored parameter
///   list.
///
/// # See also
///
/// [`examples/calc_demo.rs`](https://github.com/harakas/arael/blob/master/examples/calc_demo.rs)
/// is an end-to-end REPL example that uses this function for both
/// expression evaluation and runtime function definitions.
pub fn parse_with_functions(input: &str, bag: &FunctionBag) -> Result<E, ParseError> {
    let mut parser = Parser::from_str(input)?;
    parser.bag = Some(bag);
    let expr = parser.parse_expr()?;
    if *parser.peek() != Token::Eof {
        return Err(ParseError {
            pos: parser.peek_pos(),
            msg: format!("unexpected token after expression: {:?}", parser.peek()),
        });
    }
    Ok(expr)
}

impl std::str::FromStr for E {
    type Err = ParseError;
    fn from_str(s: &str) -> Result<E, ParseError> {
        parse(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{constant, simple_func1, symbol};
    use std::collections::HashMap;

    fn noenv() -> HashMap<&'static str, f64> {
        HashMap::new()
    }

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} !~= {b} (tol {tol})");
    }

    // --- Backward compat: plain `parse` keeps its existing surface. ---

    #[test]
    fn parse_arithmetic() {
        let e = parse("1 + 2 * 3").unwrap();
        approx(e.eval(&noenv()).unwrap(), 7.0, 1e-12);
    }

    #[test]
    fn parse_builtin_unary() {
        let e = parse("sin(0) + cos(0)").unwrap();
        approx(e.eval(&noenv()).unwrap(), 1.0, 1e-12);
    }

    #[test]
    fn parse_builtin_binary_atan2() {
        let e = parse("atan2(1, 1)").unwrap();
        approx(e.eval(&noenv()).unwrap(), std::f64::consts::FRAC_PI_4, 1e-12);
    }

    #[test]
    fn parse_builtin_sqrt_square_roundtrip() {
        let e = parse("sqrt(2) * sqrt(2)").unwrap();
        approx(e.eval(&noenv()).unwrap(), 2.0, 1e-10);
    }

    #[test]
    fn parse_builtin_ternary_clamp() {
        let e = parse("clamp(5, 0, 1)").unwrap();
        approx(e.eval(&noenv()).unwrap(), 1.0, 1e-12);
    }

    #[test]
    fn parse_heaviside_h_alias() {
        let e = parse("heaviside(0.5) + H(0.5)").unwrap();
        approx(e.eval(&noenv()).unwrap(), 2.0, 1e-12);
    }

    #[test]
    fn parse_rejects_unknown_function() {
        let err = parse("nope(x)").unwrap_err();
        assert!(err.msg.contains("unknown function"), "{err}");
    }

    #[test]
    fn parse_rejects_wrong_arity() {
        let err = parse("sin(1, 2)").unwrap_err();
        assert!(err.msg.contains("1 argument"), "{err}");
    }

    // --- parse_with_functions behaviour. ---

    #[test]
    fn parse_with_functions_empty_bag_falls_through_to_builtins() {
        let bag = FunctionBag::new();
        let e = parse_with_functions("sin(0) + 1", &bag).unwrap();
        approx(e.eval(&noenv()).unwrap(), 1.0, 1e-12);
    }

    #[test]
    fn parse_with_functions_user_symbolic_call() {
        let mut bag = FunctionBag::new();
        bag.add_symbolic("sq", vec!["t".into()], parse("t*t").unwrap());
        let e = parse_with_functions("sq(2.0)", &bag).unwrap();
        approx(e.eval(&noenv()).unwrap(), 4.0, 1e-12);
    }

    #[test]
    fn parse_with_functions_unknown_in_empty_bag_fails() {
        let bag = FunctionBag::new();
        let err = parse_with_functions("sq(1)", &bag).unwrap_err();
        assert!(err.msg.contains("unknown function"), "{err}");
    }

    #[test]
    fn parse_with_functions_shadows_builtin() {
        let mut bag = FunctionBag::new();
        // `sin` in the bag is a constant-7 "function" regardless of input.
        bag.add_symbolic("sin", vec!["x".into()], constant(7.0));
        let e = parse_with_functions("sin(0.5)", &bag).unwrap();
        approx(e.eval(&noenv()).unwrap(), 7.0, 1e-12);
    }

    #[test]
    fn parse_with_functions_h_alias_still_works() {
        let bag = FunctionBag::new();
        let e = parse_with_functions("H(0.5)", &bag).unwrap();
        approx(e.eval(&noenv()).unwrap(), 1.0, 1e-12);
    }

    #[test]
    fn bag_add_func_round_trip() {
        // Build an Expr::Func E via simple_func1 and register via add_func.
        let sq_e = simple_func1("sq", |t| t.clone() * t)(symbol("t"));
        let mut bag = FunctionBag::new();
        bag.add_func(sq_e).unwrap();
        let e = parse_with_functions("sq(3)", &bag).unwrap();
        approx(e.eval(&noenv()).unwrap(), 9.0, 1e-12);
    }

    #[test]
    fn bag_add_func_rejects_non_func() {
        let mut bag = FunctionBag::new();
        let err = bag.add_func(constant(1.0)).unwrap_err();
        assert!(err.contains("expected Expr::Func"), "{err}");
    }

    #[test]
    fn parse_with_functions_rejects_wrong_arity() {
        let mut bag = FunctionBag::new();
        bag.add_symbolic("sq", vec!["t".into()], parse("t*t").unwrap());
        let err = parse_with_functions("sq(1, 2)", &bag).unwrap_err();
        assert!(err.msg.contains("1 argument"), "{err}");
    }

    #[test]
    fn parameter_shadowing() {
        // x = 5 in the outer vars map; sq(x) = x*x registered. Calling
        // sq(3) must yield 9, not 25: the formal parameter `x` shadows
        // the outer `x = 5` for the duration of the function body.
        let mut bag = FunctionBag::new();
        bag.add_symbolic("sq", vec!["x".into()], parse("x*x").unwrap());
        let e = parse_with_functions("sq(3)", &bag).unwrap();
        let vars: HashMap<&str, f64> = [("x", 5.0)].into_iter().collect();
        approx(e.eval(&vars).unwrap(), 9.0, 1e-12);
    }

    #[test]
    fn chained_user_functions_compose() {
        let mut bag = FunctionBag::new();
        bag.add_symbolic("sq", vec!["t".into()], parse("t*t").unwrap());
        // mag's body is parsed with the CURRENT bag -- which already
        // contains `sq` -- so `sqrt(sq(a) + sq(b))` resolves fully.
        let mag_body = parse_with_functions("sqrt(sq(a) + sq(b))", &bag).unwrap();
        bag.add_symbolic("mag", vec!["a".into(), "b".into()], mag_body);
        let e = parse_with_functions("mag(3, 4)", &bag).unwrap();
        approx(e.eval(&noenv()).unwrap(), 5.0, 1e-10);
    }

    #[test]
    fn bag_remove_and_contains() {
        let mut bag = FunctionBag::new();
        bag.add_symbolic("sq", vec!["t".into()], parse("t*t").unwrap());
        assert!(bag.contains("sq"));
        assert!(bag.remove("sq"));
        assert!(!bag.contains("sq"));
        assert!(!bag.remove("sq"));
    }

    #[test]
    fn bag_names_and_entries() {
        let mut bag = FunctionBag::new();
        bag.add_symbolic("sq", vec!["t".into()], parse("t*t").unwrap());
        bag.add_symbolic("mag", vec!["a".into(), "b".into()], parse("a+b").unwrap());
        let mut names = bag.names();
        names.sort();
        assert_eq!(names, vec!["mag".to_string(), "sq".to_string()]);
        let mut entries: Vec<(String, usize)> =
            bag.entries().map(|(n, a)| (n.to_string(), a)).collect();
        entries.sort();
        assert_eq!(entries, vec![("mag".to_string(), 2), ("sq".to_string(), 1)]);
    }
}
