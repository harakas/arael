//! bc-style REPL calculator for arael-sym.
//!
//! Demonstrates `parse_with_functions` + `FunctionBag`: the user can
//! define variables and multi-argument functions at runtime, and they
//! become recognisable in subsequent expressions parsed by the REPL.
//!
//! Input grammar:
//!
//! - `expr`                       evaluate an expression to f64
//! - `name = expr`                bind a variable
//! - `name(arg1, arg2, ...) = expr`  define a function
//! - `vars`                       list variables
//! - `funcs`                      list functions
//! - `undef name`                 remove a variable or function
//! - `quit` / `exit` / Ctrl-D     leave the REPL
//!
//! Up/down arrows recall prior input, Ctrl-R does reverse-incremental
//! search (both via the `rustyline` crate). History persists across
//! sessions in `${cache_dir}/arael_calc_history`.
//!
//! # Example session
//!
//! ```text
//! > 2 + 3
//! 5
//! > pi / 4
//! 0.7853981633974483
//! > x = 1.5
//! > sq(t) = t*t
//! > sq(x)
//! 2.25
//! > mag(a, b) = sqrt(sq(a) + sq(b))
//! > mag(3, 4)
//! 5
//! > vars
//!   x = 1.5
//! > funcs
//!   mag(2 args), sq(1 arg)
//! > quit
//! ```

use arael_sym::{parse_with_functions, FuncKind, FunctionBag};
use rustyline::error::ReadlineError;
use rustyline::DefaultEditor;
use std::collections::HashMap;

const HELP_TEXT: &str = "\
Input grammar:
  expr                         evaluate (e.g. 2 + sin(0.5))
  name = expr                  assign a variable
  name(arg1, arg2) = expr      define a function
  vars                         list variables
  funcs                        list user-defined functions
  undef name                   remove variable or function
  help                         show this message
  quit / exit / Ctrl-D         leave the REPL

Built-in functions (always available): sin, cos, tan, asin, acos, atan,
atan2, sinh, cosh, tanh, exp, ln, log2, log10, sqrt, abs, heaviside (H),
clamp, pow, rad_diff, rad_sum, safe_atan2, safe_sqrt, safe_asin,
safe_acos, identity.
Named constants: pi, e.";

fn history_path() -> Option<std::path::PathBuf> {
    let mut p = dirs::cache_dir()?;
    p.push("arael_calc_history");
    Some(p)
}

fn main() {
    let mut bag = FunctionBag::new();
    let mut vars: HashMap<String, f64> = HashMap::new();

    let mut rl = match DefaultEditor::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("failed to initialise line editor: {e}");
            std::process::exit(1);
        }
    };

    let hist = history_path();
    if let Some(ref h) = hist {
        let _ = rl.load_history(h);
    }

    println!("arael-sym calculator. Type 'help' for usage, 'quit' to exit.");

    loop {
        let line = match rl.readline("> ") {
            Ok(s) => s,
            Err(ReadlineError::Eof) | Err(ReadlineError::Interrupted) => break,
            Err(e) => {
                eprintln!("input error: {e}");
                break;
            }
        };
        let line_trim = line.trim();
        if line_trim.is_empty() {
            continue;
        }
        let _ = rl.add_history_entry(line_trim);

        if let Err(msg) = handle_line(line_trim, &mut bag, &mut vars) {
            eprintln!("error: {msg}");
        }
    }

    if let Some(ref h) = hist {
        let _ = rl.save_history(h);
    }
}

fn handle_line(
    line: &str,
    bag: &mut FunctionBag,
    vars: &mut HashMap<String, f64>,
) -> Result<(), String> {
    // Simple commands first.
    match line {
        "quit" | "exit" => std::process::exit(0),
        "help" => {
            println!("{HELP_TEXT}");
            return Ok(());
        }
        "vars" => {
            if vars.is_empty() {
                println!("  (no variables)");
            } else {
                let mut names: Vec<&String> = vars.keys().collect();
                names.sort();
                for n in names {
                    println!("  {} = {}", n, vars[n]);
                }
            }
            return Ok(());
        }
        "funcs" => {
            let mut names = bag.names();
            if names.is_empty() {
                println!("  (no user functions; built-ins always available)");
            } else {
                names.sort();
                for name in &names {
                    let (params, kind) = match bag.get_info(name) {
                        Some(t) => t,
                        None => continue,
                    };
                    let body = match kind {
                        FuncKind::Symbolic { body } => body.to_string(),
                        FuncKind::SymbolicDerivs { body, .. } => body.to_string(),
                        FuncKind::Extern { call_path, .. } => format!("<extern: {call_path}>"),
                    };
                    println!("  {}({}) = {}", name, params.join(", "), body);
                }
            }
            return Ok(());
        }
        _ => {}
    }

    if let Some(name) = line.strip_prefix("undef ") {
        return handle_undef(name.trim(), bag, vars);
    }

    // Is this an assignment?  Split on the FIRST unparenthesised '=' so
    // something like `f(x) = x + 1` is recognised as a definition, not
    // an assignment of `f(x)` (a call) to `x + 1`.
    if let Some(eq_pos) = find_top_level_eq(line) {
        let (lhs, rhs) = line.split_at(eq_pos);
        let rhs = rhs[1..].trim();
        return handle_assignment(lhs.trim(), rhs, bag, vars);
    }

    // Otherwise: evaluate an expression.
    let e = parse_with_functions(line, bag).map_err(|err| err.msg)?;
    let var_refs: HashMap<&str, f64> = vars.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let val = e.eval(&var_refs)?;
    println!("{val}");
    Ok(())
}

fn handle_undef(
    name: &str,
    bag: &mut FunctionBag,
    vars: &mut HashMap<String, f64>,
) -> Result<(), String> {
    if vars.remove(name).is_some() {
        return Ok(());
    }
    if bag.remove(name) {
        return Ok(());
    }
    Err(format!("'{name}' is not defined"))
}

/// Return the byte offset of the first top-level '=' (not inside
/// parentheses), or None.
fn find_top_level_eq(line: &str) -> Option<usize> {
    let mut depth = 0usize;
    for (i, c) in line.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth = depth.saturating_sub(1),
            '=' if depth == 0 => return Some(i),
            _ => {}
        }
    }
    None
}

fn handle_assignment(
    lhs: &str,
    rhs: &str,
    bag: &mut FunctionBag,
    vars: &mut HashMap<String, f64>,
) -> Result<(), String> {
    if let Some((name, params)) = split_function_signature(lhs) {
        // Function definition: parse RHS with the current bag so the
        // new body can call any previously-defined user function.
        let body = parse_with_functions(rhs, bag).map_err(|err| err.msg)?;
        bag.add_symbolic(name, params, body);
        return Ok(());
    }

    // Variable assignment: LHS must be a bare identifier; RHS is
    // evaluated to an f64 using the current var map.
    if !is_bare_ident(lhs) {
        return Err(format!("invalid assignment target: '{lhs}'"));
    }
    let e = parse_with_functions(rhs, bag).map_err(|err| err.msg)?;
    let var_refs: HashMap<&str, f64> = vars.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let val = e.eval(&var_refs)?;
    vars.insert(lhs.to_string(), val);
    Ok(())
}

/// If `lhs` looks like `name(arg1, arg2, ...)`, return `(name, params)`.
/// Otherwise None.
fn split_function_signature(lhs: &str) -> Option<(String, Vec<String>)> {
    let open = lhs.find('(')?;
    if !lhs.ends_with(')') {
        return None;
    }
    let name = lhs[..open].trim();
    if !is_bare_ident(name) {
        return None;
    }
    let inner = &lhs[open + 1..lhs.len() - 1];
    let params: Vec<String> = if inner.trim().is_empty() {
        Vec::new()
    } else {
        inner.split(',').map(|s| s.trim().to_string()).collect()
    };
    if !params.iter().all(|p| is_bare_ident(p)) {
        return None;
    }
    Some((name.to_string(), params))
}

fn is_bare_ident(s: &str) -> bool {
    let mut it = s.chars();
    match it.next() {
        Some(c) if c.is_alphabetic() || c == '_' => {}
        _ => return false,
    }
    it.all(|c| c.is_alphanumeric() || c == '_')
}

