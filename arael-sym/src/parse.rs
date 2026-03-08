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
                    && (self.chars[self.pos].is_ascii_alphanumeric() || self.chars[self.pos] == '_')
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

struct Parser {
    tokens: Vec<(Token, usize)>,
    pos: usize,
}

impl Parser {
    fn from_str(input: &str) -> Result<Self, ParseError> {
        let mut lexer = Lexer::new(input);
        let mut tokens = Vec::new();
        loop {
            let tok = lexer.next_token()?;
            let is_eof = tok.0 == Token::Eof;
            tokens.push(tok);
            if is_eof { break; }
        }
        Ok(Parser { tokens, pos: 0 })
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
                    build_function_call(&name, args)
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

fn build_function_call(name: &str, args: Vec<E>) -> Result<E, ParseError> {
    match name {
        // Unary functions
        "sin" => expect_unary(name, args, sin),
        "cos" => expect_unary(name, args, cos),
        "tan" => expect_unary(name, args, tan),
        "asin" => expect_unary(name, args, asin),
        "acos" => expect_unary(name, args, acos),
        "atan" => expect_unary(name, args, atan),
        "sinh" => expect_unary(name, args, sinh),
        "cosh" => expect_unary(name, args, cosh),
        "tanh" => expect_unary(name, args, tanh),
        "exp" => expect_unary(name, args, exp),
        "ln" => expect_unary(name, args, ln),
        "log2" => expect_unary(name, args, log2),
        "log10" => expect_unary(name, args, log10),
        "sqrt" => expect_unary(name, args, sqrt),
        "abs" => expect_unary(name, args, abs),
        // Binary functions
        "atan2" => expect_binary(name, args, atan2),
        "pow" => expect_binary(name, args, pow),
        _ => Err(ParseError {
            pos: 0,
            msg: format!("unknown function: {name}"),
        }),
    }
}

fn expect_unary(name: &str, args: Vec<E>, f: fn(E) -> E) -> Result<E, ParseError> {
    if args.len() != 1 {
        Err(ParseError {
            pos: 0,
            msg: format!("{name} expects 1 argument, got {}", args.len()),
        })
    } else {
        Ok(f(args.into_iter().next().unwrap()))
    }
}

fn expect_binary(name: &str, args: Vec<E>, f: fn(E, E) -> E) -> Result<E, ParseError> {
    if args.len() != 2 {
        Err(ParseError {
            pos: 0,
            msg: format!("{name} expects 2 arguments, got {}", args.len()),
        })
    } else {
        let mut it = args.into_iter();
        Ok(f(it.next().unwrap(), it.next().unwrap()))
    }
}

/// Parse a string into a symbolic expression.
///
/// Supports standard infix notation with `+`, `-`, `*`, `/`, `^` (power),
/// parentheses, and function calls (`sin`, `cos`, `tan`, `asin`, `acos`,
/// `atan`, `atan2`, `sinh`, `cosh`, `tanh`, `exp`, `ln`, `log2`, `log10`,
/// `sqrt`, `abs`, `pow`).
///
/// The identifiers `pi` and `e` are recognized as named constants.
/// All other identifiers become symbolic variables.
///
/// # Errors
///
/// Returns a [`ParseError`] on invalid syntax, unknown functions, or
/// wrong argument counts.
pub fn parse(input: &str) -> Result<E, ParseError> {
    let mut parser = Parser::from_str(input)?;
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
