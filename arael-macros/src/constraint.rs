//! Constraint attribute: typed symbolic expression interpreter and code generator.
//!
//! Interprets constraint body expressions at compile time using arael-sym types,
//! differentiates symbolically, and generates compiled evaluate code.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Expr, Stmt, Pat};
use std::collections::HashMap;

use arael_sym::{self, E, vect2sym, vect3sym, matrix2sym, matrix3sym, quaternsym};

use crate::{registry_lookup, SymFieldType, extract_wrapper_inner};

// ---------------------------------------------------------------------------
// Typed symbolic value
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum SymVal {
    Scalar(E),
    Vec2(vect2sym),
    Vec3(vect3sym),
    Mat2(arael_sym::geo::matrix2sym),
    Mat3(matrix3sym),
    Quat(quaternsym),
    /// Universal euler angles: composed ea for .x/.y/.z, composed rotation for .rotation_matrix()
    UniversalEulerAngles {
        ea: vect3sym,       // get_euler_angles(R_ref * rotation(ea_delta))
        rot: matrix3sym,    // R_ref * rotation(ea_delta)
    },
}

impl SymVal {
    fn type_name(&self) -> &'static str {
        match self {
            SymVal::Scalar(_) => "scalar",
            SymVal::Vec2(_) => "vec2",
            SymVal::Vec3(_) => "vec3",
            SymVal::Mat2(_) => "mat2",
            SymVal::Mat3(_) => "mat3",
            SymVal::Quat(_) => "quat",
            SymVal::UniversalEulerAngles { .. } => "universal_euler_angles",
        }
    }
}

// ---------------------------------------------------------------------------
// Constraint context
// ---------------------------------------------------------------------------

struct ConstraintCtx {
    // variable name -> SymVal
    bindings: HashMap<String, SymVal>,
}

impl ConstraintCtx {
    fn new() -> Self {
        ConstraintCtx { bindings: HashMap::new() }
    }

    /// Create a SymVal for a struct field, given the field's sym type and a base name.
    fn make_sym_val(base: &str, sft: &SymFieldType) -> SymVal {
        match sft {
            SymFieldType::Scalar => SymVal::Scalar(arael_sym::symbol(base)),
            SymFieldType::Vec2 => SymVal::Vec2(vect2sym::new(base)),
            SymFieldType::Vec3 => SymVal::Vec3(vect3sym::new(base)),
            SymFieldType::Mat2 => SymVal::Mat2(matrix2sym::new(base)),
            SymFieldType::Mat3 => SymVal::Mat3(matrix3sym::new(base)),
            SymFieldType::Quat => SymVal::Quat(quaternsym::new(base)),
            SymFieldType::Struct(_) | SymFieldType::OptionalStruct(_) | SymFieldType::Skip => {
                // Struct fields are resolved lazily via field access
                SymVal::Scalar(arael_sym::symbol(base))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Expression interpreter
// ---------------------------------------------------------------------------

fn eval_expr(expr: &Expr, ctx: &mut ConstraintCtx) -> Result<SymVal, syn::Error> {
    match expr {
        // Variable reference
        Expr::Path(ep) if ep.qself.is_none() => {
            if let Some(ident) = ep.path.get_ident() {
                let name = ident.to_string();
                // Bindings shadow named constants, matching Rust `let`
                // semantics: `let e = ...` must refer to the binding, not
                // silently to Euler's number.
                if let Some(val) = ctx.bindings.get(&name) {
                    return Ok(val.clone());
                }
                // Named constants
                match name.as_str() {
                    "pi" => return Ok(SymVal::Scalar(arael_sym::pi())),
                    "epsilon" => return Ok(SymVal::Scalar(arael_sym::epsilon())),
                    "e" => return Ok(SymVal::Scalar(arael_sym::euler())),
                    _ => {}
                }
                return Err(syn::Error::new_spanned(ident,
                    format!("unknown variable '{}' in constraint", name)));
            }
            Err(syn::Error::new_spanned(expr, "unsupported path in constraint"))
        }

        // Field access: expr.field
        Expr::Field(ef) => {
            let field_name = match &ef.member {
                syn::Member::Named(n) => n.to_string(),
                syn::Member::Unnamed(i) => i.index.to_string(),
            };

            // Try to resolve as a dotted path first (e.g., "pose.ea" as a binding key)
            let dotted = build_dotted_path(expr);
            if let Some(ref path) = dotted
                && let Some(val) = ctx.bindings.get(path) {
                    return Ok(val.clone());
                }

            // Try evaluating the base for component access on known types
            if let Ok(base) = eval_expr(&ef.base, ctx) {
                match (&base, field_name.as_str()) {
                    // Vec3 component access
                    (SymVal::Vec3(v), "x") => return Ok(SymVal::Scalar(v.x.clone())),
                    (SymVal::Vec3(v), "y") => return Ok(SymVal::Scalar(v.y.clone())),
                    (SymVal::Vec3(v), "z") => return Ok(SymVal::Scalar(v.z.clone())),
                    // UniversalEulerAngles component access → composed ea
                    (SymVal::UniversalEulerAngles { ea, .. }, "x") => return Ok(SymVal::Scalar(ea.x.clone())),
                    (SymVal::UniversalEulerAngles { ea, .. }, "y") => return Ok(SymVal::Scalar(ea.y.clone())),
                    (SymVal::UniversalEulerAngles { ea, .. }, "z") => return Ok(SymVal::Scalar(ea.z.clone())),
                    // Vec2 component access
                    (SymVal::Vec2(v), "x") => return Ok(SymVal::Scalar(v.x.clone())),
                    (SymVal::Vec2(v), "y") => return Ok(SymVal::Scalar(v.y.clone())),
                    // Quaternion parts
                    (SymVal::Quat(q), "t") => return Ok(SymVal::Scalar(q.t.clone())),
                    (SymVal::Quat(q), "v") => return Ok(SymVal::Vec3(q.v.clone())),
                    _ => {}
                }
            }

            // Fallback: create a scalar symbol for the dotted path
            if let Some(ref path) = dotted {
                return Ok(SymVal::Scalar(arael_sym::symbol(path)));
            }

            Err(syn::Error::new_spanned(expr,
                format!("cannot resolve field access .{}", field_name)))
        }

        // Method call: expr.method(args)
        Expr::MethodCall(mc) => {
            let receiver = eval_expr(&mc.receiver, ctx)?;
            let method = mc.method.to_string();
            match (&receiver, method.as_str()) {
                (SymVal::Vec3(v), "rotation_matrix") => {
                    Ok(SymVal::Mat3(v.rotation_matrix()))
                }
                (SymVal::UniversalEulerAngles { rot, .. }, "rotation_matrix") => {
                    Ok(SymVal::Mat3(rot.clone()))
                }
                (SymVal::Mat3(m), "transpose") => {
                    Ok(SymVal::Mat3(m.transpose()))
                }
                (SymVal::Mat3(m), "get_euler_angles") => {
                    Ok(SymVal::Vec3(m.get_euler_angles()))
                }
                (SymVal::Mat2(m), "transpose") => {
                    Ok(SymVal::Mat2(m.transpose()))
                }
                (SymVal::Mat2(m), "det") => Ok(SymVal::Scalar(m.det())),
                (SymVal::Mat3(m), "det") => Ok(SymVal::Scalar(m.det())),
                (SymVal::Mat2(m), "get_rotation_angle") => {
                    Ok(SymVal::Scalar(m.get_rotation_angle()))
                }
                (SymVal::Vec2(v), "norm") => Ok(SymVal::Scalar(v.norm())),
                (SymVal::Vec2(v), "square") => Ok(SymVal::Scalar(v.square())),
                (SymVal::Vec2(v), "unit") => Ok(SymVal::Vec2(v.clone().unit())),
                (SymVal::Vec2(v), "across") => Ok(SymVal::Vec2(v.clone().across())),
                (SymVal::Vec2(v), "cross") => {
                    if mc.args.len() != 1 {
                        return Err(syn::Error::new_spanned(&mc.method, ".cross() requires 1 argument"));
                    }
                    let arg = eval_expr(&mc.args[0], ctx)?;
                    match arg {
                        SymVal::Vec2(rhs) => Ok(SymVal::Scalar(v.cross(&rhs))),
                        _ => Err(syn::Error::new_spanned(&mc.args[0], ".cross() argument must be Vec2")),
                    }
                }
                (SymVal::Quat(q), "norm") => Ok(SymVal::Scalar(q.norm())),
                (SymVal::Quat(q), "unit") => Ok(SymVal::Quat(q.clone().unit())),
                (SymVal::Quat(q), "conj") => Ok(SymVal::Quat(q.conj())),
                (SymVal::Quat(q), "rotation_matrix") => Ok(SymVal::Mat3(q.rotation_matrix())),
                (SymVal::Quat(q), "get_euler_angles") => Ok(SymVal::Vec3(q.get_euler_angles())),
                (SymVal::Quat(q), "dot") => {
                    if mc.args.len() != 1 {
                        return Err(syn::Error::new_spanned(&mc.method, ".dot() requires 1 argument"));
                    }
                    match eval_expr(&mc.args[0], ctx)? {
                        SymVal::Quat(rhs) => Ok(SymVal::Scalar(q.dot(&rhs))),
                        other => Err(syn::Error::new_spanned(&mc.args[0],
                            format!(".dot() on a quaternion requires a quaternion argument, got {}", other.type_name()))),
                    }
                }
                (SymVal::Quat(q), "rotate") => {
                    if mc.args.len() != 1 {
                        return Err(syn::Error::new_spanned(&mc.method, ".rotate() requires 1 argument"));
                    }
                    match eval_expr(&mc.args[0], ctx)? {
                        SymVal::Vec3(v) => Ok(SymVal::Vec3(q.rotate(&v))),
                        other => Err(syn::Error::new_spanned(&mc.args[0],
                            format!(".rotate() requires a Vec3 argument, got {}", other.type_name()))),
                    }
                }
                (SymVal::Vec3(v), "norm") => Ok(SymVal::Scalar(v.norm())),
                (SymVal::Vec3(v), "square") => Ok(SymVal::Scalar(v.square())),
                (SymVal::Vec3(v), "unit") => Ok(SymVal::Vec3(v.clone().unit())),
                (SymVal::Vec3(v), "cross") => {
                    if mc.args.len() != 1 {
                        return Err(syn::Error::new_spanned(&mc.method, ".cross() requires 1 argument"));
                    }
                    let arg = eval_expr(&mc.args[0], ctx)?;
                    match arg {
                        SymVal::Vec3(rhs) => Ok(SymVal::Vec3(v.cross(&rhs))),
                        _ => Err(syn::Error::new_spanned(&mc.args[0], ".cross() argument must be Vec3")),
                    }
                }
                _ => Err(syn::Error::new_spanned(&mc.method,
                    format!("unsupported method .{}() on {}", method, receiver.type_name()))),
            }
        }

        // Binary operations
        Expr::Binary(eb) => {
            let left = eval_expr(&eb.left, ctx)?;
            let right = eval_expr(&eb.right, ctx)?;
            match eb.op {
                syn::BinOp::Add(_) => sym_add(left, right, expr),
                syn::BinOp::Sub(_) => sym_sub(left, right, expr),
                syn::BinOp::Mul(_) => sym_mul(left, right, expr),
                syn::BinOp::Div(_) => sym_div(left, right, expr),
                syn::BinOp::Rem(_) => sym_rem(left, right, expr),
                _ => Err(syn::Error::new_spanned(expr, "unsupported operator in constraint")),
            }
        }

        // Unary negation
        Expr::Unary(eu) => {
            let inner = eval_expr(&eu.expr, ctx)?;
            match eu.op {
                syn::UnOp::Neg(_) => match inner {
                    SymVal::Scalar(e) => Ok(SymVal::Scalar(-e)),
                    SymVal::Vec2(v) => Ok(SymVal::Vec2(-v)),
                    SymVal::Vec3(v) => Ok(SymVal::Vec3(-v)),
                    SymVal::UniversalEulerAngles { ea, .. } => Ok(SymVal::Vec3(-ea)),
                    SymVal::Mat2(m) => Ok(SymVal::Mat2(-m)),
                    SymVal::Mat3(m) => Ok(SymVal::Mat3(-m)),
                    SymVal::Quat(q) => Ok(SymVal::Quat(-q)),
                },
                _ => Err(syn::Error::new_spanned(expr, "unsupported unary operator")),
            }
        }

        // Function calls: atan2, atan, sin, cos, etc.
        Expr::Call(ec) => {
            if let Expr::Path(func_path) = ec.func.as_ref() {
                let args: Vec<SymVal> = ec.args.iter()
                    .map(|a| eval_expr(a, ctx))
                    .collect::<Result<_, _>>()?;
                // Single-segment path: scalar fn registry (sin/cos/atan2/user fns)
                if let Some(func_name) = func_path.path.get_ident() {
                    return eval_function(&func_name.to_string(), args, expr);
                }
                // Multi-segment path: static constructors on symbolic types,
                // e.g. matrix2sym::rotation(angle). Match by the last two
                // segments so all of `matrix2sym::rotation`,
                // `arael::matrix::matrix2sym::rotation`, etc. resolve the same.
                let segs: Vec<String> = func_path.path.segments.iter()
                    .map(|s| s.ident.to_string()).collect();
                if segs.len() >= 2 {
                    let ty = &segs[segs.len() - 2];
                    let func = &segs[segs.len() - 1];
                    return eval_static_constructor(ty, func, args, expr);
                }
            }
            Err(syn::Error::new_spanned(expr, "unsupported function call in constraint"))
        }

        // Parenthesized
        Expr::Paren(ep) => eval_expr(&ep.expr, ctx),

        // Array literal [err1, err2] — we don't handle this here, it's the final result
        Expr::Array(_) => Err(syn::Error::new_spanned(expr,
            "array literal should be the final expression, not nested")),

        // Literals
        Expr::Lit(el) => match &el.lit {
            syn::Lit::Float(lf) => {
                let val: f64 = lf.base10_parse()?;
                Ok(SymVal::Scalar(arael_sym::constant(val)))
            }
            syn::Lit::Int(li) => {
                let val: i64 = li.base10_parse()?;
                Ok(SymVal::Scalar(arael_sym::constant(val as f64)))
            }
            _ => Err(syn::Error::new_spanned(expr, "unsupported literal in constraint")),
        },

        // Index access: expr[i]. Vec2/Vec3 arms make chained m[i][j]
        // element access work (m[i] yields the row vector).
        Expr::Index(idx) => {
            let base = eval_expr(&idx.expr, ctx)?;
            let i = literal_index(&idx.index)?;
            match &base {
                SymVal::Mat3(m) => match m.rows.get(i) {
                    Some(row) => Ok(SymVal::Vec3(row.clone())),
                    None => Err(syn::Error::new_spanned(&idx.index, "matrix3 index out of range")),
                },
                SymVal::Mat2(m) => match m.rows.get(i) {
                    Some(row) => Ok(SymVal::Vec2(row.clone())),
                    None => Err(syn::Error::new_spanned(&idx.index, "matrix2 index out of range")),
                },
                SymVal::Vec3(v) => match i {
                    0 => Ok(SymVal::Scalar(v.x.clone())),
                    1 => Ok(SymVal::Scalar(v.y.clone())),
                    2 => Ok(SymVal::Scalar(v.z.clone())),
                    _ => Err(syn::Error::new_spanned(&idx.index, "vec3 index out of range")),
                },
                SymVal::Vec2(v) => match i {
                    0 => Ok(SymVal::Scalar(v.x.clone())),
                    1 => Ok(SymVal::Scalar(v.y.clone())),
                    _ => Err(syn::Error::new_spanned(&idx.index, "vec2 index out of range")),
                },
                _ => Err(syn::Error::new_spanned(expr,
                    format!("cannot index into {}", base.type_name()))),
            }
        }

        _ => Err(syn::Error::new_spanned(expr, "unsupported expression in constraint")),
    }
}

/// Build a dotted path string from a field access chain: a.b.c -> "a.b.c"
fn build_dotted_path(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Path(ep) if ep.qself.is_none() => {
            ep.path.get_ident().map(|i| i.to_string())
        }
        Expr::Field(ef) => {
            let base = build_dotted_path(&ef.base)?;
            let field = match &ef.member {
                syn::Member::Named(n) => n.to_string(),
                syn::Member::Unnamed(i) => i.index.to_string(),
            };
            Some(format!("{}.{}", base, field))
        }
        _ => None,
    }
}

/// Build a `FunctionBag` pre-populated with every user-registered
/// `#[arael::function]` so calls inside a user function's body /
/// derivatives can resolve each other. Mutually-recursive references
/// (e.g. `f`'s deriv is `g(x)`, `g`'s deriv is `-f(x)`) are handled by
/// a two-pass build: the first pass registers every user fn with
/// placeholder derivs / bodies so all names resolve at parse time;
/// the second pass re-parses under the full bag and replaces the
/// stubs.
fn build_user_function_bag() -> arael_sym::FunctionBag {
    fn dummy_eval(_args: &[f64]) -> f64 { 0.0 }
    let all = crate::registry_all_functions();
    let mut bag = arael_sym::FunctionBag::new();

    // Pass 1: stubs so cross-references resolve when parsing.
    for uf in &all {
        match uf {
            crate::UserFunction::Symbolic { sym_name, param_names, .. } => {
                // Stub body: symbol(sym_name) -- irrelevant for dispatch as
                // the second pass replaces it. Parser dispatches function
                // calls by name regardless of body shape.
                bag.add_symbolic(
                    sym_name.clone(),
                    param_names.clone(),
                    arael_sym::symbol(sym_name),
                );
            }
            crate::UserFunction::Extern { sym_name, eval_path, param_names, arity, .. } => {
                let zero_derivs: Vec<arael_sym::E> =
                    (0..*arity).map(|_| arael_sym::constant(0.0)).collect();
                bag.add_with_kind(
                    sym_name.clone(),
                    param_names.clone(),
                    arael_sym::FuncKind::Extern {
                        derivs: zero_derivs,
                        eval_fn: dummy_eval,
                        call_path: eval_path.clone(),
                    },
                );
            }
        }
    }

    // Pass 2: re-parse each fn's body / derivs under the full bag and
    // replace the stubs. Errors surface at the first constraint-body
    // use-site (where the user's own span is available for the error
    // message); here we silently skip failing entries.
    for uf in &all {
        match uf {
            crate::UserFunction::Symbolic { sym_name, param_names, body, .. } => {
                let mut sub = arael_sym::FunctionBag::new();
                // Inherit all entries from the full bag by re-registering
                // them as stubs (same as pass 1). Then parse body against
                // sub, then re-insert into main bag.
                //
                // Simpler: parse against the main bag directly. Parameter
                // names are registered as placeholder 0-ary symbolic fns
                // in the sub-bag.
                for (p, ph) in param_names.iter().zip(param_names.iter().map(|p| arael_sym::symbol(p))) {
                    sub.add_symbolic(p.clone(), Vec::<String>::new(), ph);
                }
                // Copy every user-fn entry to sub:
                for other in &all {
                    match other {
                        crate::UserFunction::Symbolic {
                            sym_name: on, param_names: opn, ..
                        } => {
                            sub.add_symbolic(on.clone(), opn.clone(), arael_sym::symbol(on));
                        }
                        crate::UserFunction::Extern {
                            sym_name: on, eval_path: oep, param_names: opn, arity: oa, ..
                        } => {
                            let zero: Vec<arael_sym::E> =
                                (0..*oa).map(|_| arael_sym::constant(0.0)).collect();
                            sub.add_with_kind(on.clone(), opn.clone(),
                                arael_sym::FuncKind::Extern {
                                    derivs: zero, eval_fn: dummy_eval,
                                    call_path: oep.clone(),
                                });
                        }
                    }
                }
                let body_e = match arael_sym::parse_with_functions(body, &sub) {
                    Ok(e) => e,
                    Err(_) => continue,
                };
                bag.add_symbolic(sym_name.clone(), param_names.clone(), body_e);
            }
            crate::UserFunction::Extern {
                sym_name, eval_path, param_names, arity, deriv_strings, ..
            } => {
                let mut sub = arael_sym::FunctionBag::new();
                for (p, ph) in param_names.iter().zip(param_names.iter().map(|p| arael_sym::symbol(p))) {
                    sub.add_symbolic(p.clone(), Vec::<String>::new(), ph);
                }
                for other in &all {
                    match other {
                        crate::UserFunction::Symbolic {
                            sym_name: on, param_names: opn, ..
                        } => {
                            sub.add_symbolic(on.clone(), opn.clone(), arael_sym::symbol(on));
                        }
                        crate::UserFunction::Extern {
                            sym_name: on, eval_path: oep, param_names: opn, arity: oa, ..
                        } => {
                            let zero: Vec<arael_sym::E> =
                                (0..*oa).map(|_| arael_sym::constant(0.0)).collect();
                            sub.add_with_kind(on.clone(), opn.clone(),
                                arael_sym::FuncKind::Extern {
                                    derivs: zero, eval_fn: dummy_eval,
                                    call_path: oep.clone(),
                                });
                        }
                    }
                }
                let mut derivs: Vec<arael_sym::E> = Vec::with_capacity(*arity);
                let mut ok = true;
                for s in deriv_strings {
                    match arael_sym::parse_with_functions(s, &sub) {
                        Ok(e) => derivs.push(e),
                        Err(_) => { ok = false; break; }
                    }
                }
                if !ok { continue; }
                bag.add_with_kind(
                    sym_name.clone(),
                    param_names.clone(),
                    arael_sym::FuncKind::Extern {
                        derivs,
                        eval_fn: dummy_eval,
                        call_path: eval_path.clone(),
                    },
                );
            }
        }
    }
    bag
}

fn eval_function(name: &str, args: Vec<SymVal>, span: &Expr) -> Result<SymVal, syn::Error> {
    // Delegate scalar functions to arael-sym's name-based lookup. Arity and
    // scalar-ness are validated here since SymVal is a macro-local type.
    if let Some(fnref) = arael_sym::function_by_name(name) {
        return match fnref {
            arael_sym::FunctionRef::Unary(f) => {
                if args.len() != 1 {
                    return Err(syn::Error::new_spanned(span, format!("{} expects 1 arg", name)));
                }
                match &args[0] {
                    SymVal::Scalar(e) => Ok(SymVal::Scalar(f(e.clone()))),
                    _ => Err(syn::Error::new_spanned(span, format!("{} expects a scalar argument", name))),
                }
            }
            arael_sym::FunctionRef::Binary(f) => {
                if args.len() != 2 {
                    return Err(syn::Error::new_spanned(span, format!("{} expects 2 args", name)));
                }
                match (&args[0], &args[1]) {
                    (SymVal::Scalar(a), SymVal::Scalar(b)) => Ok(SymVal::Scalar(f(a.clone(), b.clone()))),
                    _ => Err(syn::Error::new_spanned(span, format!("{} expects scalar arguments", name))),
                }
            }
            arael_sym::FunctionRef::Ternary(f) => {
                if args.len() != 3 {
                    return Err(syn::Error::new_spanned(span, format!("{} expects 3 args", name)));
                }
                match (&args[0], &args[1], &args[2]) {
                    (SymVal::Scalar(a), SymVal::Scalar(b), SymVal::Scalar(c)) =>
                        Ok(SymVal::Scalar(f(a.clone(), b.clone(), c.clone()))),
                    _ => Err(syn::Error::new_spanned(span, format!("{} expects scalar arguments", name))),
                }
            }
        };
    }
    // User-defined `#[arael::function]` fallback: consult the macro
    // registry for a matching user function. If found, dispatch through
    // arael-sym's own parser with a bag binding args to param names and
    // registering every other user function. This keeps a single surface
    // language and handles cross-referencing user functions (e.g.
    // elliptic_k's derivative mentions elliptic_e).
    if let Some(uf) = crate::registry_lookup_function(name) {
        let expected_arity = uf.param_names().len();
        if args.len() != expected_arity {
            return Err(syn::Error::new_spanned(span, format!(
                "{} expects {} arg(s), got {}", name, expected_arity, args.len())));
        }
        let arg_es: Vec<arael_sym::E> = args.iter().map(|a| match a {
            SymVal::Scalar(e) => Ok(e.clone()),
            _ => Err(syn::Error::new_spanned(span,
                format!("{} expects scalar argument(s)", name))),
        }).collect::<Result<_, _>>()?;

        // Build bag for resolving OTHER user-fn calls inside the body.
        // Parameter names themselves are parsed as symbols and then
        // substituted with the actual arg expressions -- that keeps the
        // generated code pointing at the caller's binding (e.g. `m.x.work()`
        // rather than a naked `x`).
        let bag = build_user_function_bag();
        let subs: Vec<(arael_sym::E, arael_sym::E)> = uf.param_names().iter().zip(arg_es.iter())
            .map(|(pname, e)| (arael_sym::symbol(pname), e.clone()))
            .collect();

        match &uf {
            crate::UserFunction::Symbolic { body, .. } => {
                let parsed = arael_sym::parse_with_functions(body, &bag)
                    .map_err(|err| syn::Error::new_spanned(span,
                        format!("arael::function `{}` body parse: {}", name, err)))?;
                return Ok(SymVal::Scalar(parsed.substitute(&subs)));
            }
            crate::UserFunction::Extern { eval_path, deriv_strings, .. } => {
                // Parse each deriv against the full user-function bag plus
                // the user's own param names (so `g(x)` resolves if `g` is
                // another user fn and `x` is our param). The resulting E
                // uses the user's chosen param names as symbols; rewrite
                // those to __p0 / __p1 / ... so arael-sym's chain-rule
                // substitution at diff time works as expected.
                fn __dummy_eval(_args: &[f64]) -> f64 { 0.0 }
                let user_param_syms: Vec<arael_sym::E> = uf.param_names().iter()
                    .map(|p| arael_sym::symbol(p)).collect();
                let placeholder_param_syms: Vec<arael_sym::E> = (0..expected_arity)
                    .map(|i| arael_sym::symbol(&format!("__p{}", i))).collect();
                let rewrite_subs: Vec<(arael_sym::E, arael_sym::E)> = user_param_syms.iter()
                    .zip(placeholder_param_syms.iter())
                    .map(|(u, p)| (u.clone(), p.clone())).collect();

                let mut combined = build_user_function_bag();
                for (pname, e) in uf.param_names().iter().zip(user_param_syms.iter()) {
                    combined.add_symbolic(pname.clone(), std::vec::Vec::<String>::new(), e.clone());
                }
                let mut derivs_e: Vec<arael_sym::E> = Vec::with_capacity(deriv_strings.len());
                for s in deriv_strings {
                    let d = arael_sym::parse_with_functions(s, &combined)
                        .map_err(|err| syn::Error::new_spanned(span,
                            format!("arael::function `{}` deriv parse: {}", name, err)))?;
                    // Rewrite user param names → placeholder names so
                    // arael-sym's chain rule substitutes them with the
                    // actual arg expressions at diff time.
                    derivs_e.push(d.substitute(&rewrite_subs));
                }
                let node = arael_sym::extern_func(
                    name,
                    expected_arity,
                    eval_path,
                    {
                        let derivs_e = derivs_e.clone();
                        move |_: std::vec::Vec<arael_sym::E>| derivs_e.clone()
                    },
                    __dummy_eval,
                )(arg_es.clone());
                return Ok(SymVal::Scalar(node));
            }
        }
    }

    // Vector-typed functions stay local: they operate on the macro's
    // SymVal::Vec2 type, which arael-sym doesn't know about.
    match name {
        "dot" | "cross2" => {
            if args.len() != 2 { return Err(syn::Error::new_spanned(span, format!("{} expects 2 args", name))); }
            let mut it = args.into_iter();
            let a = it.next().unwrap();
            let b = it.next().unwrap();
            if name == "dot" {
                sym_mul(a, b, span)
            } else {
                match (a, b) {
                    (SymVal::Vec2(va), SymVal::Vec2(vb)) => Ok(SymVal::Scalar(va.cross(&vb))),
                    _ => Err(syn::Error::new_spanned(span, "cross2 expects Vec2 arguments")),
                }
            }
        }
        _ => Err(syn::Error::new_spanned(span, format!("unknown function '{}' in constraint", name))),
    }
}

/// Parse an index expression that must be a literal integer.
fn literal_index(index_expr: &Expr) -> Result<usize, syn::Error> {
    if let Expr::Lit(lit) = index_expr
        && let syn::Lit::Int(li) = &lit.lit {
            return li.base10_parse();
        }
    Err(syn::Error::new_spanned(index_expr, "index must be a literal integer"))
}

/// Extract Vec3 from SymVal, coercing UniversalEulerAngles to its composed ea.
fn as_vec3(v: SymVal) -> Option<vect3sym> {
    match v {
        SymVal::Vec3(v) => Some(v),
        SymVal::UniversalEulerAngles { ea, .. } => Some(ea),
        _ => None,
    }
}

fn sym_add(left: SymVal, right: SymVal, span: &Expr) -> Result<SymVal, syn::Error> {
    match (&left, &right) {
        (SymVal::Scalar(_), SymVal::Scalar(_)) => {
            if let (SymVal::Scalar(a), SymVal::Scalar(b)) = (left, right) { Ok(SymVal::Scalar(a + b)) } else { unreachable!() }
        }
        (SymVal::Vec2(_), SymVal::Vec2(_)) => {
            if let (SymVal::Vec2(a), SymVal::Vec2(b)) = (left, right) { Ok(SymVal::Vec2(a + b)) } else { unreachable!() }
        }
        (SymVal::Mat2(_), SymVal::Mat2(_)) => {
            if let (SymVal::Mat2(a), SymVal::Mat2(b)) = (left, right) { Ok(SymVal::Mat2(a + b)) } else { unreachable!() }
        }
        (SymVal::Mat3(_), SymVal::Mat3(_)) => {
            if let (SymVal::Mat3(a), SymVal::Mat3(b)) = (left, right) { Ok(SymVal::Mat3(a + b)) } else { unreachable!() }
        }
        (SymVal::Quat(_), SymVal::Quat(_)) => {
            if let (SymVal::Quat(a), SymVal::Quat(b)) = (left, right) { Ok(SymVal::Quat(a + b)) } else { unreachable!() }
        }
        _ => {
            if let (Some(a), Some(b)) = (as_vec3(left), as_vec3(right)) {
                Ok(SymVal::Vec3(a + b))
            } else {
                Err(syn::Error::new_spanned(span, "type mismatch in addition"))
            }
        }
    }
}

fn sym_sub(left: SymVal, right: SymVal, span: &Expr) -> Result<SymVal, syn::Error> {
    match (&left, &right) {
        (SymVal::Scalar(_), SymVal::Scalar(_)) => {
            if let (SymVal::Scalar(a), SymVal::Scalar(b)) = (left, right) { Ok(SymVal::Scalar(a - b)) } else { unreachable!() }
        }
        (SymVal::Vec2(_), SymVal::Vec2(_)) => {
            if let (SymVal::Vec2(a), SymVal::Vec2(b)) = (left, right) { Ok(SymVal::Vec2(a - b)) } else { unreachable!() }
        }
        (SymVal::Mat2(_), SymVal::Mat2(_)) => {
            if let (SymVal::Mat2(a), SymVal::Mat2(b)) = (left, right) { Ok(SymVal::Mat2(a - b)) } else { unreachable!() }
        }
        (SymVal::Mat3(_), SymVal::Mat3(_)) => {
            if let (SymVal::Mat3(a), SymVal::Mat3(b)) = (left, right) { Ok(SymVal::Mat3(a - b)) } else { unreachable!() }
        }
        (SymVal::Quat(_), SymVal::Quat(_)) => {
            if let (SymVal::Quat(a), SymVal::Quat(b)) = (left, right) { Ok(SymVal::Quat(a - b)) } else { unreachable!() }
        }
        _ => {
            if let (Some(a), Some(b)) = (as_vec3(left), as_vec3(right)) {
                Ok(SymVal::Vec3(a - b))
            } else {
                Err(syn::Error::new_spanned(span, "type mismatch in subtraction"))
            }
        }
    }
}

fn sym_mul(left: SymVal, right: SymVal, span: &Expr) -> Result<SymVal, syn::Error> {
    // Euler angle params coerce to their composed angle vector, exactly as
    // in sym_add/sym_sub.
    let left = match left {
        SymVal::UniversalEulerAngles { ea, .. } => SymVal::Vec3(ea),
        other => other,
    };
    let right = match right {
        SymVal::UniversalEulerAngles { ea, .. } => SymVal::Vec3(ea),
        other => other,
    };
    match (left, right) {
        (SymVal::Scalar(a), SymVal::Scalar(b)) => Ok(SymVal::Scalar(a * b)),
        (SymVal::Scalar(a), SymVal::Vec2(b)) => Ok(SymVal::Vec2(arael_sym::geo::vect2sym { x: a.clone() * b.x, y: a * b.y })),
        (SymVal::Vec2(a), SymVal::Scalar(b)) => Ok(SymVal::Vec2(arael_sym::geo::vect2sym { x: a.x * b.clone(), y: a.y * b })),
        (SymVal::Scalar(a), SymVal::Vec3(b)) => Ok(SymVal::Vec3(a * b)),
        (SymVal::Vec3(a), SymVal::Scalar(b)) => Ok(SymVal::Vec3(a * b)),
        (SymVal::Vec2(a), SymVal::Vec2(b)) => Ok(SymVal::Scalar(a * b)), // dot product
        (SymVal::Vec3(a), SymVal::Vec3(b)) => Ok(SymVal::Scalar(a * b)), // dot product
        (SymVal::Mat2(a), SymVal::Vec2(b)) => Ok(SymVal::Vec2(a * b)),
        (SymVal::Mat2(a), SymVal::Mat2(b)) => Ok(SymVal::Mat2(a * b)),
        (SymVal::Mat3(a), SymVal::Vec3(b)) => Ok(SymVal::Vec3(a * b)),
        (SymVal::Mat3(a), SymVal::Mat3(b)) => Ok(SymVal::Mat3(a * b)),
        (SymVal::Vec2(a), SymVal::Mat2(b)) => Ok(SymVal::Vec2(a * b)), // v * M = M^T v
        (SymVal::Vec3(a), SymVal::Mat3(b)) => Ok(SymVal::Vec3(a * b)), // v * M = M^T v
        (SymVal::Scalar(a), SymVal::Mat2(b)) => Ok(SymVal::Mat2(a * b)),
        (SymVal::Mat2(a), SymVal::Scalar(b)) => Ok(SymVal::Mat2(a * b)),
        (SymVal::Scalar(a), SymVal::Mat3(b)) => Ok(SymVal::Mat3(a * b)),
        (SymVal::Mat3(a), SymVal::Scalar(b)) => Ok(SymVal::Mat3(a * b)),
        (SymVal::Quat(a), SymVal::Quat(b)) => Ok(SymVal::Quat(a * b)), // Hamilton product
        (SymVal::Scalar(a), SymVal::Quat(b)) => Ok(SymVal::Quat(a * b)),
        (SymVal::Quat(a), SymVal::Scalar(b)) => Ok(SymVal::Quat(a * b)),
        _ => Err(syn::Error::new_spanned(span, "type mismatch in multiplication")),
    }
}

/// Static constructor dispatch for `matrix2sym::rotation(angle)` etc.
/// Match-by-last-two-segments lets users write either `matrix2sym::rotation(x)`
/// (with `use arael::matrix::matrix2sym;`) or the fully qualified
/// `arael::matrix::matrix2sym::rotation(x)`; both end with the same pair.
fn eval_static_constructor(ty: &str, func: &str, args: Vec<SymVal>, span: &Expr)
    -> Result<SymVal, syn::Error>
{
    // Small extractors so each constructor arm reads as its signature.
    let scalar = |i: usize| -> Result<E, syn::Error> {
        match &args[i] {
            SymVal::Scalar(e) => Ok(e.clone()),
            other => Err(syn::Error::new_spanned(span,
                format!("{}::{} argument {} must be scalar, got {}",
                    ty, func, i + 1, other.type_name()))),
        }
    };
    let vec2 = |i: usize| -> Result<vect2sym, syn::Error> {
        match &args[i] {
            SymVal::Vec2(v) => Ok(v.clone()),
            other => Err(syn::Error::new_spanned(span,
                format!("{}::{} argument {} must be Vec2, got {}",
                    ty, func, i + 1, other.type_name()))),
        }
    };
    let vec3 = |i: usize| -> Result<vect3sym, syn::Error> {
        match &args[i] {
            SymVal::Vec3(v) => Ok(v.clone()),
            other => Err(syn::Error::new_spanned(span,
                format!("{}::{} argument {} must be Vec3, got {}",
                    ty, func, i + 1, other.type_name()))),
        }
    };
    let arity = |n: usize| -> Result<(), syn::Error> {
        if args.len() == n { Ok(()) } else {
            Err(syn::Error::new_spanned(span,
                format!("{}::{} expects {} argument(s), got {}",
                    ty, func, n, args.len())))
        }
    };

    match (ty, func) {
        ("matrix2sym", "rotation") => {
            arity(1)?;
            Ok(SymVal::Mat2(matrix2sym::rotation(scalar(0)?)))
        }
        ("matrix2sym", "rotation_from_sincos") => {
            arity(2)?;
            Ok(SymVal::Mat2(matrix2sym::rotation_from_sincos(scalar(0)?, scalar(1)?)))
        }
        ("matrix2sym", "identity") => {
            arity(0)?;
            Ok(SymVal::Mat2(matrix2sym::identity()))
        }
        ("matrix2sym", "from_rows") => {
            arity(2)?;
            Ok(SymVal::Mat2(matrix2sym::from_rows(vec2(0)?, vec2(1)?)))
        }
        ("matrix2sym", "from_cols") => {
            arity(2)?;
            Ok(SymVal::Mat2(matrix2sym::from_cols(vec2(0)?, vec2(1)?)))
        }
        ("matrix2sym", "from_elements") => {
            arity(4)?;
            Ok(SymVal::Mat2(matrix2sym::from_elements(
                scalar(0)?, scalar(1)?, scalar(2)?, scalar(3)?)))
        }
        ("matrix3sym", "identity") => {
            arity(0)?;
            Ok(SymVal::Mat3(matrix3sym::identity()))
        }
        ("matrix3sym", "from_rows") => {
            arity(3)?;
            Ok(SymVal::Mat3(matrix3sym::from_rows(vec3(0)?, vec3(1)?, vec3(2)?)))
        }
        ("matrix3sym", "from_cols") => {
            arity(3)?;
            Ok(SymVal::Mat3(matrix3sym::from_cols(vec3(0)?, vec3(1)?, vec3(2)?)))
        }
        ("matrix3sym", "from_elements") => {
            arity(9)?;
            Ok(SymVal::Mat3(matrix3sym::from_elements(
                scalar(0)?, scalar(1)?, scalar(2)?,
                scalar(3)?, scalar(4)?, scalar(5)?,
                scalar(6)?, scalar(7)?, scalar(8)?)))
        }
        ("matrix3sym", "rotation_from_euler_angles") => {
            arity(1)?;
            Ok(SymVal::Mat3(matrix3sym::rotation_from_euler_angles(&vec3(0)?)))
        }
        ("matrix3sym", "rotation_from_axis_angle") => {
            arity(2)?;
            Ok(SymVal::Mat3(matrix3sym::rotation_from_axis_angle(&vec3(0)?, scalar(1)?)))
        }
        ("quaternsym", "identity") => {
            arity(0)?;
            Ok(SymVal::Quat(quaternsym::identity()))
        }
        ("quaternsym", "from_euler_angles") => {
            arity(1)?;
            Ok(SymVal::Quat(quaternsym::from_euler_angles(&vec3(0)?)))
        }
        ("quaternsym", "from_axis_angle") => {
            arity(2)?;
            Ok(SymVal::Quat(quaternsym::from_axis_angle(&vec3(0)?, scalar(1)?)))
        }
        _ => Err(syn::Error::new_spanned(span,
            format!("unsupported constructor `{}::{}` in constraint", ty, func))),
    }
}

fn sym_div(left: SymVal, right: SymVal, span: &Expr) -> Result<SymVal, syn::Error> {
    let left = match left {
        SymVal::UniversalEulerAngles { ea, .. } => SymVal::Vec3(ea),
        other => other,
    };
    match (left, right) {
        (SymVal::Scalar(a), SymVal::Scalar(b)) => Ok(SymVal::Scalar(a / b)),
        (SymVal::Vec2(a), SymVal::Scalar(b)) => Ok(SymVal::Vec2(a / b)),
        (SymVal::Vec3(a), SymVal::Scalar(b)) => Ok(SymVal::Vec3(a / b)),
        _ => Err(syn::Error::new_spanned(span, "unsupported division types")),
    }
}

fn sym_rem(left: SymVal, right: SymVal, span: &Expr) -> Result<SymVal, syn::Error> {
    match (left, right) {
        // `%` is the cross product operator, mirroring the runtime vect3.
        (SymVal::Vec3(a), SymVal::Vec3(b)) => Ok(SymVal::Vec3(a % b)),
        _ => Err(syn::Error::new_spanned(span,
            "`%` (cross product) requires Vec3 operands")),
    }
}

// ---------------------------------------------------------------------------
// Constraint attribute parsing and code generation
// ---------------------------------------------------------------------------

pub struct ConstraintVar {
    pub name: String,
    pub type_name: Option<String>,
}

pub struct ConstraintAttr {
    /// Declared block-field names on the constraint struct. Single-ident
    /// form `constraint(hb, ...)` parses to a 1-element Vec; bracketed form
    /// `constraint([hb_ab, hb_ac, hb_bc], ...)` parses to N elements. Each
    /// element may be a dotted path (e.g. `pose.hb_pose`) for remote-block
    /// references.
    pub block_fields: Vec<String>,
    pub parent_name: Option<String>,  // e.g. "lm" for parent=lm
    pub guard: Option<String>,        // runtime guard expression, e.g. "self.info.gps.is_some()"
    pub name: Option<String>,         // optional label for Jacobian rows, e.g. name = "sweep"
    pub vars: Vec<ConstraintVar>,     // explicit variables (legacy, may be empty)
    pub body_stmts: Vec<Stmt>,
}

impl ConstraintAttr {
    /// The first (and for single-block constraints, only) block-field name.
    /// Shorthand for `&self.block_fields[0]`; callers that currently handle
    /// only one block continue to use this. Multi-block routing uses
    /// `block_fields` directly.
    pub fn primary_block_field(&self) -> &str { &self.block_fields[0] }
}

/// Parse all `#[arael(constraint(...))]` attributes (supports multiple per struct).
#[allow(dead_code)]
pub fn parse_constraint_attrs(attrs: &[syn::Attribute]) -> syn::Result<Vec<ConstraintAttr>> {
    let mut results = Vec::new();
    for attr in attrs {
        if !attr.path().is_ident("arael") { continue; }
        let content: TokenStream2 = attr.parse_args()?;
        let tokens: Vec<proc_macro2::TokenTree> = content.into_iter().collect();
        if tokens.is_empty() { continue; }

        if let proc_macro2::TokenTree::Ident(ref ident) = tokens[0] {
            if *ident != "constraint" { continue; }

            if tokens.len() < 2 {
                return Err(syn::Error::new_spanned(ident, "expected constraint(...)"));
            }
            if let proc_macro2::TokenTree::Group(ref group) = tokens[1] {
                if group.delimiter() != proc_macro2::Delimiter::Parenthesis {
                    return Err(syn::Error::new_spanned(ident, "expected parentheses after constraint"));
                }
                let inner: Vec<proc_macro2::TokenTree> = group.stream().into_iter().collect();
                if let Some(c) = parse_constraint_inner_impl(&inner, ident)? {
                    results.push(c);
                }
            }
        }
    }
    Ok(results)
}

fn parse_constraint_inner_impl(
    tokens: &[proc_macro2::TokenTree],
    err_span: &proc_macro2::Ident,
) -> syn::Result<Option<ConstraintAttr>> {
    // Syntax: constraint(hb, [parent=name,] [var: Type, ...], { body })
    // Multi-block form: constraint([hb_ab, hb_ac, hb_bc], [parent=...,] { body })
    let mut pos = 0;
    let mut block_fields: Vec<String> = Vec::new();
    let mut parent_name: Option<String> = None;
    let mut guard: Option<String> = None;
    let mut name_label: Option<String> = None;
    let mut vars: Vec<ConstraintVar> = Vec::new();
    // Track the first-token span of each positional block field so we
    // can point errors back at the offending item. Only populated for
    // items that come in via the positional loop below; entries from
    // the bracketed branch skip this vec because the bracketed form is
    // unrestricted (any N >= 1 is fine).
    let mut positional_spans: Vec<proc_macro2::Span> = Vec::new();
    let mut was_bracketed = false;

    // Bracketed list form: first token is [ ... ]. Walk its stream as a
    // comma-separated list of (dotted) idents to populate block_fields.
    if let Some(proc_macro2::TokenTree::Group(g)) = tokens.first()
        && g.delimiter() == proc_macro2::Delimiter::Bracket {
            let inner: Vec<proc_macro2::TokenTree> = g.stream().into_iter().collect();
            let mut ipos = 0;
            while ipos < inner.len() {
                match &inner[ipos] {
                    proc_macro2::TokenTree::Ident(id) => {
                        let mut full_name = id.to_string();
                        ipos += 1;
                        // Collect dotted segments: ident.ident.ident
                        while let Some(proc_macro2::TokenTree::Punct(p)) = inner.get(ipos) {
                            if p.as_char() == '.' {
                                ipos += 1;
                                if let Some(proc_macro2::TokenTree::Ident(next_id)) = inner.get(ipos) {
                                    full_name = format!("{}.{}", full_name, next_id);
                                    ipos += 1;
                                } else { break; }
                            } else { break; }
                        }
                        block_fields.push(full_name);
                    }
                    proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => { ipos += 1; }
                    tt => return Err(syn::Error::new_spanned(err_span,
                        format!("expected ident or ',' in constraint block list, got `{}`", tt))),
                }
            }
            if block_fields.is_empty() {
                return Err(syn::Error::new_spanned(err_span,
                    "constraint block list must name at least one field"));
            }
            pos += 1;
            was_bracketed = true;
            // Skip optional trailing comma between list and next arg
            if let Some(proc_macro2::TokenTree::Punct(p)) = tokens.get(pos)
                && p.as_char() == ',' { pos += 1; }
        }

    loop {
        match tokens.get(pos) {
            Some(proc_macro2::TokenTree::Ident(id)) => {
                let name = id.to_string();
                let ident_span = id.span();
                pos += 1;
                // Check for = (parent=lm) or : (var: Type)
                match tokens.get(pos) {
                    Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '=' => {
                        pos += 1;
                        if name == "parent" {
                            if let Some(proc_macro2::TokenTree::Ident(val)) = tokens.get(pos) {
                                parent_name = Some(val.to_string());
                                pos += 1;
                            } else {
                                return Err(syn::Error::new(ident_span,
                                    "parent = expects an entity field name"));
                            }
                        } else if name == "guard" {
                            // Collect all tokens until next comma or brace group
                            let mut guard_tokens = Vec::new();
                            while pos < tokens.len() {
                                match &tokens[pos] {
                                    proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => break,
                                    proc_macro2::TokenTree::Group(g) if g.delimiter() == proc_macro2::Delimiter::Brace => break,
                                    t => { guard_tokens.push(t.clone()); pos += 1; }
                                }
                            }
                            let guard_ts: proc_macro2::TokenStream = guard_tokens.into_iter().collect();
                            guard = Some(guard_ts.to_string());
                        } else if name == "name" {
                            // Expect a string literal
                            if let Some(proc_macro2::TokenTree::Literal(lit)) = tokens.get(pos) {
                                let s = lit.to_string();
                                // Strip surrounding quotes
                                if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                                    name_label = Some(s[1..s.len()-1].to_string());
                                } else {
                                    return Err(syn::Error::new_spanned(err_span,
                                        "name = \"...\" expects a string literal"));
                                }
                                pos += 1;
                            } else {
                                return Err(syn::Error::new_spanned(err_span,
                                    "name = \"...\" expects a string literal"));
                            }
                        } else {
                            // A silently swallowed key here is dangerous:
                            // `gaurd = ...` would compile as an unguarded,
                            // always-active constraint.
                            return Err(syn::Error::new(ident_span,
                                format!("unknown constraint attribute key `{}`, expected `parent`, `guard`, or `name`", name)));
                        }
                    }
                    Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == ':' => {
                        pos += 1;
                        let type_name = match tokens.get(pos) {
                            Some(proc_macro2::TokenTree::Ident(type_id)) => {
                                pos += 1;
                                Some(type_id.to_string())
                            }
                            _ => return Err(syn::Error::new_spanned(err_span, "expected type after :")),
                        };
                        if block_fields.is_empty() {
                            block_fields.push(name);
                            positional_spans.push(ident_span);
                        } else {
                            vars.push(ConstraintVar { name, type_name });
                        }
                    }
                    _ => {
                        // Check for dotted path: ident.ident (e.g., pose.hb_pose)
                        let mut full_name = name.clone();
                        while let Some(proc_macro2::TokenTree::Punct(p)) = tokens.get(pos) {
                            if p.as_char() == '.' {
                                pos += 1;
                                if let Some(proc_macro2::TokenTree::Ident(next_id)) = tokens.get(pos) {
                                    full_name = format!("{}.{}", full_name, next_id);
                                    pos += 1;
                                } else {
                                    break;
                                }
                            } else {
                                break;
                            }
                        }
                        // Any bare ident / dotted path at a positional
                        // slot (not `name=val`, not `name: Type`) is a
                        // block-field reference. The positional form
                        // is restricted: see the post-loop check for
                        // the allowed shapes.
                        block_fields.push(full_name);
                        positional_spans.push(ident_span);
                    }
                }
            }
            Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == ',' => {
                pos += 1;
            }
            Some(proc_macro2::TokenTree::Group(g))
                if g.delimiter() == proc_macro2::Delimiter::Brace => {
                break;
            }
            _ => {
                return Err(syn::Error::new_spanned(err_span,
                    "expected: constraint(hb, [parent=name,] { body })"));
            }
        }
    }

    if block_fields.is_empty() {
        return Err(syn::Error::new_spanned(err_span,
            "constraint needs at least the block field name"));
    }

    // Positional block lists are restricted to N = 1. Any N >= 2 list
    // (including `(<local>, root.<triplet>)`) must use the bracketed
    // form `constraint([a, b, ...], { body })` so the attribute has one
    // unambiguous shape for multi-block constraints.
    if !was_bracketed && positional_spans.len() >= 2 {
        let span = positional_spans[1];
        return Err(syn::Error::new(span,
            "constraint(...) accepts a single positional block; wrap \
             N >= 2 block lists in brackets: \
             `constraint([a, b, ...], { body })`"));
    }

    // Parse the body block
    let body_group = match tokens.get(pos) {
        Some(proc_macro2::TokenTree::Group(g)) => g,
        _ => return Err(syn::Error::new_spanned(err_span, "expected { body }")),
    };
    let block_tokens = proc_macro2::TokenStream::from(
        proc_macro2::TokenTree::Group(body_group.clone())
    );
    let block: syn::Block = syn::parse2(block_tokens)?;

    Ok(Some(ConstraintAttr {
        block_fields,
        parent_name,
        guard,
        name: name_label,
        vars,
        body_stmts: block.stmts,
    }))
}

/// Generate the debug `constraints()` function that returns symbolic expressions.
#[allow(dead_code)]
pub fn generate_constraint_impl(
    struct_name: &syn::Ident,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    constraint: &ConstraintAttr,
) -> syn::Result<TokenStream2> {
    // Build variable setup: let pose = Pose::sym("pose"); etc.
    let struct_layout = registry_lookup(&struct_name.to_string());
    let ref_paths = struct_layout.as_ref().map(|l| &l.ref_paths[..]).unwrap_or(&[]);

    let block_field = fields.iter().find(|f| {
        f.ident.as_ref().map(|i| i.to_string()) == Some(constraint.primary_block_field().to_string())
    }).ok_or_else(|| {
        syn::Error::new_spanned(struct_name,
            format!("constraint block field '{}' not found", constraint.primary_block_field()))
    })?;
    let (a_type, _b_type) = extract_block_type_args(&block_field.ty)?;
    let parent_name = constraint.parent_name.clone()
        .unwrap_or_else(|| a_type.to_lowercase());

    // Collect variable setup statements
    let mut var_setup: Vec<TokenStream2> = Vec::new();

    // Ref fields
    for (field_name, _) in ref_paths {
        if let Some(field) = fields.iter().find(|f| f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone()))
            && let Some((_, inner_ident)) = extract_wrapper_inner(&field.ty, "Ref") {
                let var_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
                let type_ident = syn::Ident::new(&inner_ident.to_string(), inner_ident.span());
                let name_str = field_name.as_str();
                var_setup.push(quote! { let #var_ident = #type_ident::sym(#name_str); });
            }
    }
    // Parent
    let parent_ident = syn::Ident::new(&parent_name, proc_macro2::Span::call_site());
    let a_type_ident = syn::Ident::new(&a_type, proc_macro2::Span::call_site());
    let parent_name_str = parent_name.as_str();
    var_setup.push(quote! { let #parent_ident = #a_type_ident::sym(#parent_name_str); });
    // Root
    var_setup.push(quote! { let path = Path::sym("path"); });

    // The constraint body: all stmts except the last, then wrap the last (array) in .to_vec()
    let body_stmts = &constraint.body_stmts;
    let (init_stmts, final_expr) = if body_stmts.len() > 1 {
        (&body_stmts[..body_stmts.len()-1], &body_stmts[body_stmts.len()-1])
    } else {
        (&body_stmts[..0], &body_stmts[0])
    };

    Ok(quote! {
        impl #struct_name {
            /// Auto-generated: returns symbolic constraint expressions for debug/inspection.
            pub fn constraints() -> std::vec::Vec<arael::sym::E> {
                use arael::model::ModelSym;
                use arael::sym::{atan2, atan, sin, cos};
                arael::sym! {
                    #(#var_setup)*
                    #(#init_stmts)*
                    #final_expr .to_vec()
                }
            }
        }
    })
}

/// Build symbolic substitution pairs for euler_angles precomputation.
/// Given a variable base (e.g. "pose") and field name (e.g. "ea"),
/// returns pairs (from_expr, to_expr) that replace sin/cos with precomputed values
/// and rotation matrix entries with precomputed matrix fields.
fn build_euler_substitutions(var_base: &str, field_name: &str) -> Vec<(arael_sym::E, arael_sym::E)> {
    use arael_sym::{symbol, sin, cos};

    let mut subs = Vec::new();

    // Rotation matrix substitutions FIRST (larger patterns, applied before sin/cos)
    let ea_sym = arael_sym::vect3sym::new(&format!("{}.{}.work()", var_base, field_name));
    let rot = ea_sym.rotation_matrix();

    for row in 0..3 {
        for (col_name, col_expr) in [("x", &rot.rows[row].x), ("y", &rot.rows[row].y), ("z", &rot.rows[row].z)] {
            let to_sym = symbol(&format!("{}.{}.rotation_matrix[{}].{}", var_base, field_name, row, col_name));
            subs.push((col_expr.clone(), to_sym));
        }
    }

    // sin/cos substitutions SECOND (catch remaining occurrences not part of rotation matrix)
    for comp in &["x", "y", "z"] {
        let param_sym = symbol(&format!("{}.{}.work().{}", var_base, field_name, comp));
        let sin_from = sin(param_sym.clone());
        let cos_from = cos(param_sym);
        let sin_to = symbol(&format!("{}.{}.sincos.0.{}", var_base, field_name, comp));
        let cos_to = symbol(&format!("{}.{}.sincos.1.{}", var_base, field_name, comp));
        subs.push((sin_from, sin_to));
        subs.push((cos_from, cos_to));
    }

    subs
}

/// Build symbolic substitutions for universal_euler_angles precomputation.
/// Substitutes composed rotation (R_ref * rotation(ea_delta)) with precomputed matrix,
/// and sin/cos of ea_delta with precomputed sincos.
fn build_universal_euler_substitutions(var_base: &str, field_name: &str) -> Vec<(arael_sym::E, arael_sym::E)> {
    use arael_sym::{symbol, sin, cos};

    let mut subs = Vec::new();

    // Build composed rotation: R_ref * rotation(ea_delta)
    let r_ref_sym = matrix3sym::new(&format!("{}.{}.ref_rotation", var_base, field_name));
    let dea_sym = vect3sym::new(&format!("{}.{}.delta", var_base, field_name));
    let composed = r_ref_sym * dea_sym.rotation_matrix();

    // Substitute composed rotation entries with precomputed rotation_matrix
    for row in 0..3 {
        for (col_name, col_expr) in [("x", &composed.rows[row].x), ("y", &composed.rows[row].y), ("z", &composed.rows[row].z)] {
            let to_sym = symbol(&format!("{}.{}.rotation_matrix[{}].{}", var_base, field_name, row, col_name));
            subs.push((col_expr.clone(), to_sym));
        }
    }

    // sin/cos of ea.delta → ea.delta_sincos
    for comp in &["x", "y", "z"] {
        let param_sym = symbol(&format!("{}.{}.delta.{}", var_base, field_name, comp));
        let sin_from = sin(param_sym.clone());
        let cos_from = cos(param_sym);
        let sin_to = symbol(&format!("{}.{}.delta_sincos.0.{}", var_base, field_name, comp));
        let cos_to = symbol(&format!("{}.{}.delta_sincos.1.{}", var_base, field_name, comp));
        subs.push((sin_from, sin_to));
        subs.push((cos_from, cos_to));
    }

    subs
}

/// Apply substitutions to a list of expressions. Returns the modified expressions.
fn apply_substitutions(exprs: &mut Vec<arael_sym::E>, subs: &[(arael_sym::E, arael_sym::E)]) {
    for (from, to) in subs {
        for e in exprs.iter_mut() {
            *e = arael_sym::cse::replace_pub(e, from, to);
        }
    }
}

/// Recursively register sym bindings for a variable and all its nested struct fields.
/// `key_prefix` is used for binding lookup (e.g. "pose.info.gps")
/// `sym_prefix` is used for generated code (e.g. "pose.info.gps.as_ref().unwrap()")
fn register_bindings_recursive(ctx: &mut ConstraintCtx, key_prefix: &str, sym_prefix: &str, type_name: &str) {
    if let Some(layout) = registry_lookup(type_name) {
        for (field_name, sft) in &layout.fields {
            if matches!(sft, SymFieldType::Skip) { continue; }
            let is_param = layout.param_fields.contains(field_name);
            let sym_base = if is_param {
                format!("{}.{}.work()", sym_prefix, field_name)
            } else {
                format!("{}.{}", sym_prefix, field_name)
            };
            let binding_key = format!("{}.{}", key_prefix, field_name);
            match sft {
                SymFieldType::Struct(inner_type) => {
                    let nested_key = format!("{}.{}", key_prefix, field_name);
                    let nested_sym = format!("{}.{}", sym_prefix, field_name);
                    register_bindings_recursive(ctx, &nested_key, &nested_sym, inner_type);
                }
                SymFieldType::OptionalStruct(inner_type) => {
                    let nested_key = format!("{}.{}", key_prefix, field_name);
                    let nested_sym = format!("{}.{}.as_ref().unwrap()", sym_prefix, field_name);
                    register_bindings_recursive(ctx, &nested_key, &nested_sym, inner_type);
                }
                _ => {
                    let is_universal_ea = is_param
                        && layout.universal_euler_angle_fields.contains(field_name);
                    if is_universal_ea {
                        // Build composed rotation and euler angles symbolically
                        let r_ref_sym = matrix3sym::new(
                            &format!("{}.{}.ref_rotation", sym_prefix, field_name));
                        let dea_sym = vect3sym::new(
                            &format!("{}.{}.delta", sym_prefix, field_name));
                        let composed_rot = r_ref_sym * dea_sym.rotation_matrix();
                        let composed_ea = composed_rot.get_euler_angles();
                        ctx.bindings.insert(binding_key,
                            SymVal::UniversalEulerAngles {
                                ea: composed_ea,
                                rot: composed_rot,
                            });
                    } else {
                        ctx.bindings.insert(binding_key,
                            ConstraintCtx::make_sym_val(&sym_base, sft));
                    }
                    // For Param fields, also register .value as a constant
                    if is_param {
                        let value_key = format!("{}.{}_value", key_prefix, field_name);
                        let value_base = format!("{}.{}.value", sym_prefix, field_name);
                        ctx.bindings.insert(value_key, ConstraintCtx::make_sym_val(&value_base, sft));
                    }
                }
            }
        }
    }
}

fn parse_sym_code(code: &str) -> syn::Result<Expr> {
    syn::parse_str(code).map_err(|e| {
        syn::Error::new(proc_macro2::Span::call_site(),
            format!("failed to parse generated code: {}\ncode: {}", e, &code[..code.len().min(200)]))
    })
}

fn extract_block_type_args(ty: &syn::Type) -> syn::Result<(String, Option<String>)> {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            // TripletBlock has no entity type args — all entities come from Ref fields
            if seg.ident == "TripletBlock" {
                return Ok(("__triplet__".to_string(), None));
            }
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                let type_args: Vec<&syn::Type> = args.args.iter()
                    .filter_map(|a| if let syn::GenericArgument::Type(t) = a { Some(t) } else { None })
                    .collect();
                if seg.ident == "SelfBlock" && !type_args.is_empty() {
                    let a = type_ident_name(type_args[0])?;
                    return Ok((a, None));
                }
                if seg.ident == "CrossBlock" && type_args.len() >= 2 {
                    let a = type_ident_name(type_args[0])?;
                    let b = type_ident_name(type_args[1])?;
                    return Ok((a, Some(b)));
                }
            }
        }
    Err(syn::Error::new_spanned(ty, "expected SelfBlock<A>, CrossBlock<A, B>, or TripletBlock"))
}

/// Per-pair routing info for a multi-CrossBlock constraint. Built by
/// `build_multi_cross_routing` from the declared block-field list and the
/// struct's ref-field layout.
#[derive(Clone)]
#[allow(dead_code)]  // a_idx/b_idx kept for diagnostics; emission reads starts/counts
pub struct MultiCrossRouting {
    pub block_ident: syn::Ident,
    /// Index of A-side ref in triplet_entities.
    pub a_idx: usize,
    /// Index of B-side ref in triplet_entities.
    pub b_idx: usize,
    /// Starting offset in __all_idx of entity A's params.
    pub a_start: usize,
    pub a_count: usize,
    pub b_start: usize,
    pub b_count: usize,
}

/// Build the multi-cross routing table for a constraint struct that
/// declares multiple block fields (all CrossBlocks). For each declared
/// CrossBlock field, resolves which (ordered) ref pair it serves, using
/// `#[arael(cross = (refA, refB))]` when present and type-based
/// auto-resolution otherwise. Every unordered entity pair must be covered
/// by exactly one CrossBlock; uncovered pairs and ambiguous auto-resolution
/// produce compile-time errors.
pub fn build_multi_cross_routing(
    fields: &syn::FieldsNamed,
    block_fields: &[String],
    triplet_entities: &[(syn::Ident, syn::Ident, usize, usize)],
    struct_ident: &syn::Ident,
) -> syn::Result<Vec<MultiCrossRouting>> {
    let mut out: Vec<MultiCrossRouting> = Vec::new();
    // Normalized unordered pairs already claimed (prevents two CrossBlocks
    // on the same Hessian pair).
    let mut claimed: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
    let candidates_desc = || triplet_entities.iter()
        .map(|(v, _, _, _)| v.to_string()).collect::<Vec<_>>().join(", ");

    for block_name in block_fields {
        // Dotted-path entries (e.g. `pose.hb_pose`) are remote-block
        // references resolved by the remote-block emission path, not
        // local CrossBlock fields on this struct. Skip them here --
        // they don't participate in per-pair routing.
        if block_name.contains('.') { continue; }
        let field = fields.named.iter().find(|f|
            f.ident.as_ref().map(|i| i.to_string()) == Some(block_name.clone())
        ).ok_or_else(|| syn::Error::new_spanned(struct_ident,
            format!("constraint declares block field `{}` but no such field found on `{}`",
                    block_name, struct_ident)))?;

        let (a_type, b_type_opt) = extract_block_type_args(&field.ty)?;
        if a_type == "__triplet__" {
            return Err(syn::Error::new_spanned(struct_ident,
                format!("field `{}` is a TripletBlock -- multi-block constraints currently require CrossBlock fields only", block_name)));
        }
        let b_type = b_type_opt.ok_or_else(|| syn::Error::new_spanned(struct_ident,
            format!("field `{}` must be CrossBlock<A, B>, not SelfBlock (SelfBlock<Self> lives on the entity struct, not here)", block_name)))?;

        // Parse #[arael(cross = (refA, refB))] on this field, if present.
        let cross_attr = crate::parse_arael_attr(&field.attrs)?;
        let cross_refs: Option<(String, String)> = match cross_attr {
            Some(crate::AraelAttr::Cross(refs)) if refs.len() == 2 =>
                Some((refs[0].clone(), refs[1].clone())),
            _ => None,
        };

        // Resolve entity indices (A-side, B-side).
        let (a_idx, b_idx) = if let Some((ra, rb)) = cross_refs {
            let a_idx = triplet_entities.iter().position(|(v, _, _, _)| v == &ra)
                .ok_or_else(|| syn::Error::new_spanned(struct_ident,
                    format!("on field `{}`: cross = ({}, {}) references unknown ref field `{}` (candidates: {})",
                        block_name, ra, rb, ra, candidates_desc())))?;
            let b_idx = triplet_entities.iter().position(|(v, _, _, _)| v == &rb)
                .ok_or_else(|| syn::Error::new_spanned(struct_ident,
                    format!("on field `{}`: cross = ({}, {}) references unknown ref field `{}` (candidates: {})",
                        block_name, ra, rb, rb, candidates_desc())))?;
            if a_idx == b_idx {
                return Err(syn::Error::new_spanned(struct_ident,
                    format!("on field `{}`: cross = ({}, {}) names the same ref field twice", block_name, ra, rb)));
            }
            // Types must match the CrossBlock's A and B exactly (order-sensitive).
            let a_ty = triplet_entities[a_idx].1.to_string();
            let b_ty = triplet_entities[b_idx].1.to_string();
            if a_ty != a_type {
                return Err(syn::Error::new_spanned(struct_ident,
                    format!("on field `{}: CrossBlock<{}, {}>`: cross = ({}, {}) -- `{}` is Ref<{}>, expected Ref<{}>",
                        block_name, a_type, b_type, ra, rb, ra, a_ty, a_type)));
            }
            if b_ty != b_type {
                return Err(syn::Error::new_spanned(struct_ident,
                    format!("on field `{}: CrossBlock<{}, {}>`: cross = ({}, {}) -- `{}` is Ref<{}>, expected Ref<{}>",
                        block_name, a_type, b_type, ra, rb, rb, b_ty, b_type)));
            }
            (a_idx, b_idx)
        } else {
            // Type-based auto-resolution. When A == B (same type on both
            // sides), (i, j) and (j, i) label the same Hessian pair, so we
            // canonicalize on i < j to avoid spurious ambiguity. When
            // A != B, search all ordered pairs.
            let mut pairs: Vec<(usize, usize)> = Vec::new();
            let same_type = a_type == b_type;
            for ai in 0..triplet_entities.len() {
                for bi in 0..triplet_entities.len() {
                    if ai == bi { continue; }
                    if same_type && ai > bi { continue; }
                    if triplet_entities[ai].1 == a_type
                        && triplet_entities[bi].1 == b_type
                    {
                        pairs.push((ai, bi));
                    }
                }
            }
            match pairs.len() {
                0 => return Err(syn::Error::new_spanned(struct_ident,
                    format!("on field `{}: CrossBlock<{}, {}>`: no Ref<{}> + Ref<{}> pair found on the struct",
                        block_name, a_type, b_type, a_type, b_type))),
                1 => pairs[0],
                _ => {
                    let listed = pairs.iter().map(|(a,b)|
                        format!("({}, {})", triplet_entities[*a].0, triplet_entities[*b].0)
                    ).collect::<Vec<_>>().join(", ");
                    return Err(syn::Error::new_spanned(struct_ident,
                        format!("on field `{}: CrossBlock<{}, {}>` is ambiguous -- matches multiple ref pairs: {}. Add `#[arael(cross = (refA, refB))]` to disambiguate",
                            block_name, a_type, b_type, listed)));
                }
            }
        };

        // Unordered-pair uniqueness: (a, b) and (b, a) are the same Hessian
        // pair and must not be claimed twice.
        let norm = if a_idx < b_idx { (a_idx, b_idx) } else { (b_idx, a_idx) };
        if !claimed.insert(norm) {
            return Err(syn::Error::new_spanned(struct_ident,
                format!("on field `{}`: ref pair ({}, {}) is already claimed by another CrossBlock on this constraint",
                    block_name, triplet_entities[a_idx].0, triplet_entities[b_idx].0)));
        }

        let (_, _, a_start, a_count) = &triplet_entities[a_idx];
        let (_, _, b_start, b_count) = &triplet_entities[b_idx];
        out.push(MultiCrossRouting {
            block_ident: syn::Ident::new(block_name, proc_macro2::Span::call_site()),
            a_idx, b_idx,
            a_start: *a_start, a_count: *a_count,
            b_start: *b_start, b_count: *b_count,
        });
    }

    // Every unordered entity pair must be covered. Dropping any pair
    // would violate the "never drop cross-Hessian" invariant.
    for i in 0..triplet_entities.len() {
        for j in (i + 1)..triplet_entities.len() {
            if !claimed.contains(&(i, j)) {
                return Err(syn::Error::new_spanned(struct_ident,
                    format!("residual on `{}` references entity pair ({}, {}) but no declared CrossBlock covers it -- add `CrossBlock<{}, {}>` with `#[arael(cross = ({}, {}))]` to the struct and list it in `#[arael(constraint([...], ...))]`",
                        struct_ident,
                        triplet_entities[i].0, triplet_entities[j].0,
                        triplet_entities[i].1, triplet_entities[j].1,
                        triplet_entities[i].0, triplet_entities[j].0)));
            }
        }
    }

    Ok(out)
}

fn type_ident_name(ty: &syn::Type) -> syn::Result<String> {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            return Ok(seg.ident.to_string());
        }
    Err(syn::Error::new_spanned(ty, "expected a simple type name"))
}

fn add_param_symbols(base: &str, sft: &SymFieldType, out: &mut Vec<String>) {
    match sft {
        SymFieldType::Scalar => out.push(base.to_string()),
        SymFieldType::Vec2 => {
            out.push(format!("{}.x", base));
            out.push(format!("{}.y", base));
        }
        SymFieldType::Vec3 => {
            out.push(format!("{}.x", base));
            out.push(format!("{}.y", base));
            out.push(format!("{}.z", base));
        }
        _ => {}
    }
}

/// Generate `calc_cost` and `calc_grad_hessian` methods on the root struct.
/// `precision` is "f32" or "f64".
pub fn generate_root_methods(
    root_name: &syn::Ident,
    root_fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    precision: &str,
    custom: bool,
    jacobian: bool,
) -> syn::Result<TokenStream2> {
    let stashed = crate::registry_constraints();
    let root_var_name = root_name.to_string().to_lowercase();
    let root_var_ident = syn::Ident::new(&root_var_name, proc_macro2::Span::call_site());
    let cast_type: syn::Type = syn::parse_str(precision)
        .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
            format!("invalid precision type '{}': {}", precision, e)))?;

    // Root SelfBlock index setup: when the root struct has its own
    // Params + a SelfBlock<Self>, its set_indices must be called at
    // __set_block_indices time so any constraint that touches root
    // params (including nested multi-cross constraints referencing
    // the root) finds valid global indices on the root's self-block.
    // The existing per-constraint set_block_indices_loops only fires
    // when a constraint is attached to the root itself; this prelude
    // runs unconditionally whenever the root has Params and the
    // mandatory SelfBlock<Self> field.
    let root_self_block_prelude: TokenStream2 = {
        let root_layout = registry_lookup(&root_name.to_string());
        let root_hb_field = root_layout.as_ref().and_then(|l| l.self_block_field.clone());
        let root_param_fields = root_layout.as_ref().map(|l| l.param_fields.clone()).unwrap_or_default();
        if let Some(hb) = root_hb_field.as_ref().filter(|_| !root_param_fields.is_empty()) {
            let hb_ident = syn::Ident::new(hb, proc_macro2::Span::call_site());
            let layout = root_layout.as_ref().unwrap();
            let mut count: usize = 0;
            let mut idx_stmts: Vec<TokenStream2> = Vec::new();
            for pf in &root_param_fields {
                let pf_ident = syn::Ident::new(pf, proc_macro2::Span::call_site());
                let size = layout.fields.iter()
                    .find(|(n, _)| n == pf)
                    .map(|(_, sft)| match sft {
                        SymFieldType::Scalar => 1usize,
                        SymFieldType::Vec2 => 2,
                        SymFieldType::Vec3 => 3,
                        _ => 0,
                    }).unwrap_or(0);
                let offset = count;
                let end = offset + size;
                idx_stmts.push(quote! {
                    self.#pf_ident.write_indices(&mut __root_self_idx[#offset..#end]);
                });
                count += size;
            }
            if count == 0 {
                quote! {}
            } else {
                quote! {
                    let mut __root_self_idx = [0u32; #count];
                    #(#idx_stmts)*
                    self.#hb_ident.set_indices(&__root_self_idx);
                }
            }
        } else {
            quote! {}
        }
    };

    let constraint_impls: Vec<TokenStream2> = Vec::new();
    let mut cost_loops: Vec<TokenStream2> = Vec::new();
    let mut grad_hessian_loops: Vec<TokenStream2> = Vec::new();
    let mut jacobian_loops: Vec<TokenStream2> = Vec::new();
    let mut set_block_indices_loops: Vec<TokenStream2> = Vec::new();

    // Grouping for root-level cross-constraints on the same collection.
    // Merges multiple #[arael(constraint(...))] attributes into one loop per collection.
    struct CrossCollectionGroup {
        rc_ident: syn::Ident,
        a_param_count: usize,
        b_param_count: usize,
        block_ident: syn::Ident,
        constraint_index_field: Option<syn::Ident>,
        // Shared across all attributes on this struct (recomputed from the first attribute)
        a_idx_stmts: Vec<TokenStream2>,
        b_idx_stmts: Vec<TokenStream2>,
        resolve_stmts: Vec<TokenStream2>,
        root_var_ident: syn::Ident,
        // Per-attribute entries (with guards baked in, matching SelfBlock pattern)
        cost_entries: Vec<TokenStream2>,
        gh_entries: Vec<TokenStream2>,
        jac_entries: Vec<TokenStream2>,
    }
    let mut cross_groups: std::collections::HashMap<String, CrossCollectionGroup> = std::collections::HashMap::new();

    // Per-CrossBlock info for a multi-cross constraint (one entry per
    // declared CrossBlock field). The entity-span setup (__all_idx via
    // triplet_idx_stmts) is shared across all CrossBlocks on the same
    // struct; each entry just knows which slice of __all_idx to pass to
    // its own set_indices call and which dr sub-slices to write.
    #[derive(Clone)]
    struct MultiCrossBlockInfo {
        block_ident: syn::Ident,
        /// Starting offset in __all_idx of entity A's params.
        a_start: usize,
        /// Number of scalar params for entity A.
        a_count: usize,
        /// Starting offset in __all_idx of entity B's params.
        b_start: usize,
        /// Number of scalar params for entity B.
        b_count: usize,
    }

    // Grouping for TripletBlock constraints on the same collection.
    //
    // Also used for multi-cross constraints (N-entity constraint declared
    // with multiple CrossBlock fields instead of a single TripletBlock):
    // when `multi_cross_blocks` is non-empty, the final
    // TripletBlock.add_residual_cross call in the gh loop is replaced by
    // per-pair CrossBlock.add_residual_cross calls (emitted into
    // gh_entries), and the set_block_indices loop emits one set_indices
    // per declared CrossBlock over slices of __all_idx.
    struct TripletCollectionGroup {
        rc_ident: syn::Ident,
        triplet_param_count: usize,
        block_ident: syn::Ident,
        constraint_index_field: Option<syn::Ident>,
        triplet_idx_stmts: Vec<TokenStream2>,
        entity_offsets: Vec<u32>,           // cumulative entity span boundaries
        resolve_stmts: Vec<TokenStream2>,
        root_var_ident: syn::Ident,
        cost_entries: Vec<TokenStream2>,
        gh_entries: Vec<TokenStream2>,
        jac_entries: Vec<TokenStream2>,
        /// Non-empty only for multi-cross constraints. When populated,
        /// set_block_indices emits per-CrossBlock set_indices calls
        /// instead of the trivial TripletBlock no-op.
        multi_cross_blocks: Vec<MultiCrossBlockInfo>,
        /// Per-entity SelfBlock.set_indices calls. Needed when the
        /// participating entity has no other constraint of its own that
        /// would set its SelfBlock indices (which would leave indices at
        /// the u32::MAX sentinel, silently skipping every add_residual).
        /// Skipped for the root entity — its indices are set by the
        /// unconditional root_self_block_prelude at method entry.
        entity_self_indices: Vec<TokenStream2>,
    }
    let mut triplet_groups: std::collections::HashMap<String, TripletCollectionGroup> = std::collections::HashMap::new();

    // Grouping for constraints that iterate the same collection.
    // Merges SelfBlock + nested CrossBlock into a single loop per collection.
    struct CollectionGroup {
        coll_ident: syn::Ident,
        self_var: syn::Ident,
        a_type_ident: syn::Ident,
        // SelfBlock: index setup + constraint entries
        self_block: Option<SelfBlockInfo>,
        // Cost/GH/Jacobian entries that go directly in the outer loop (SelfBlock constraints)
        cost_entries: Vec<TokenStream2>,
        gh_entries: Vec<TokenStream2>,
        jac_entries: Vec<TokenStream2>,
        // Nested CrossBlock: inner loops over frines
        nested_cost_loops: Vec<TokenStream2>,
        nested_gh_loops: Vec<TokenStream2>,
        nested_jac_loops: Vec<TokenStream2>,
    }
    struct SelfBlockInfo {
        a_param_count: usize,
        a_idx_stmts: Vec<TokenStream2>,
        block_ident: syn::Ident,
    }
    let mut collection_groups: std::collections::HashMap<String, CollectionGroup> = std::collections::HashMap::new();

    // Grouping for SelfBlock constraints that live on a single-instance entity
    // (the root itself, or a direct-composed sub-model field). Keyed by the
    // access path ("self" for RootSelf, "self.<field>" for DirectField).
    // Multiple #[arael(constraint(...))] attributes on the same entity merge
    // into one emitted block per path.
    struct SingleInstanceGroup {
        accessor_read: TokenStream2,
        accessor_write: TokenStream2,
        self_var: syn::Ident,
        root_var_ident: syn::Ident,
        a_type_ident: syn::Ident,
        a_param_count: usize,
        a_idx_stmts: Vec<TokenStream2>,
        block_ident: syn::Ident,
        constraint_index_field: Option<syn::Ident>,
        cost_entries: Vec<TokenStream2>,
        gh_entries: Vec<TokenStream2>,
        jac_entries: Vec<TokenStream2>,
    }
    let mut single_instance_groups: std::collections::HashMap<String, SingleInstanceGroup> = std::collections::HashMap::new();

    let mut _generated_constraints_fn: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Collect all types reachable from this root (for multi-root support)
    let reachable = {
        let mut set = std::collections::HashSet::new();
        let mut queue = Vec::new();
        // Seed with the root itself — lets RootSelf constraints and direct-composed
        // sub-models resolve through the BFS (root's layout Struct fields expand).
        queue.push(root_name.to_string());
        // Seed with types directly in root fields (inner of Vec<T>, Deque<T>, etc.)
        let root_fields_parsed: syn::FieldsNamed = syn::parse2(quote! { { #root_fields } })?;
        for field in &root_fields_parsed.named {
            if let syn::Type::Path(tp) = &field.ty
                && let Some(seg) = tp.path.segments.last() {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                        && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
                            && let Ok(name) = type_ident_name(inner) {
                                queue.push(name);
                            }
                }
        }
        // BFS through type registry
        while let Some(type_name) = queue.pop() {
            if !set.insert(type_name.clone()) { continue; }
            if let Some(layout) = registry_lookup(&type_name) {
                for (_, sft) in &layout.fields {
                    if let SymFieldType::Struct(s) = sft {
                        queue.push(s.clone());
                    }
                }
            }
        }
        set
    };

    // Count constraint attributes per struct (for default label naming).
    let mut attr_count_per_struct: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
    for sc in &stashed {
        if !reachable.contains(&sc.struct_name) { continue; }
        *attr_count_per_struct.entry(sc.struct_name.clone()).or_insert(0) += 1;
    }
    let mut attr_idx_per_struct: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    for sc in &stashed {
        // Skip constraints for types not reachable from this root
        if !reachable.contains(&sc.struct_name) { continue; }
        // Determine this attribute's index within the struct (for default label).
        let this_idx = {
            let slot = attr_idx_per_struct.entry(sc.struct_name.clone()).or_insert(0);
            let idx = *slot;
            *slot += 1;
            idx
        };
        let total_attrs = *attr_count_per_struct.get(&sc.struct_name).unwrap_or(&1);
        // Re-parse constraint
        let attr_ts: proc_macro2::TokenStream = sc.attr_tokens.parse()
            .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                format!("failed to re-parse constraint for {}: {}", sc.struct_name, e)))?;
        let attr_tokens: Vec<proc_macro2::TokenTree> = attr_ts.into_iter().collect();
        let err_ident = syn::Ident::new(&sc.struct_name, proc_macro2::Span::call_site());

        let constraint = match &attr_tokens[0] {
            proc_macro2::TokenTree::Ident(id) if *id == "constraint" => {
                if let Some(proc_macro2::TokenTree::Group(g)) = attr_tokens.get(1) {
                    // The stash string round trip loses the original
                    // spans, so errors would point at the root attribute.
                    // Prefix the message with the constraint's recorded
                    // source location instead.
                    parse_constraint_inner_impl(
                        &g.stream().into_iter().collect::<Vec<_>>(), &err_ident)
                        .map_err(|e| syn::Error::new(e.span(),
                            format!("{}:{}: {}", sc.attr_file, sc.attr_line, e)))?
                } else { None }
            }
            _ => None,
        };
        let constraint = match constraint { Some(c) => c, None => continue };

        // Compute the JacobianRow label for this constraint attribute:
        // - If user provided `name = "..."`, use it verbatim
        // - Otherwise use the struct name, suffixed with ":<idx>" if multi-attribute
        let label_str: String = if let Some(ref n) = constraint.name {
            n.clone()
        } else if total_attrs <= 1 {
            sc.struct_name.clone()
        } else {
            format!("{}:{}", sc.struct_name, this_idx)
        };
        let label_literal = syn::LitStr::new(&label_str, proc_macro2::Span::call_site());

        // Re-parse fields
        let fields_ts: proc_macro2::TokenStream = sc.fields_tokens.parse()
            .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                format!("failed to re-parse fields for {}: {}", sc.struct_name, e)))?;
        let fields: syn::FieldsNamed = syn::parse2(quote! { { #fields_ts } })?;
        let struct_ident = syn::Ident::new(&sc.struct_name, proc_macro2::Span::call_site());

        // Generate the debug constraints() function (only once per struct,
        // skip for constraints that use _value bindings since Sym can't represent them)
        let _has_value_refs = constraint.body_stmts.iter().any(|s| {
            format!("{}", quote! { #s }).contains("_value")
        });
        // Skip debug constraints() function for now (it has hardcoded root type)
        // TODO: pass root_name to generate_constraint_impl

        // Now generate the traversal code for root methods
        // Check if block_field is a dotted path (remote block, e.g. pose.hb_pose)
        let is_remote_block = constraint.primary_block_field().contains('.');

        let (a_type, b_type, remote_block_info) = if is_remote_block {
            // Remote block: e.g. "pose.hb_pose" means the block lives on a Ref<Pose>'s field
            let parts: Vec<&str> = constraint.primary_block_field().split('.').collect();
            let ref_field_name = parts[0];
            let target_block_field = parts[1];

            // Find the Ref<T> field on the constraint struct to get target type
            let ref_field = fields.named.iter().find(|f|
                f.ident.as_ref().map(|i| i.to_string()) == Some(ref_field_name.to_string())
            ).ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(),
                format!("remote block field '{}' not found on {}", ref_field_name, sc.struct_name)))?;
            let (_, ref_type_ident) = extract_wrapper_inner(&ref_field.ty, "Ref")
                .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(),
                    format!("field '{}' is not a Ref<T>", ref_field_name)))?;
            let target_type = ref_type_ident.to_string();

            // Find the parent struct that contains this constraint struct
            let parent_type = {
                let guard = crate::SYM_REGISTRY.lock().unwrap();
                guard.as_ref().and_then(|reg| {
                    reg.layouts.iter().find(|(_, layout)| {
                        layout.fields.iter().any(|(_, sft)| {
                            if let SymFieldType::Struct(s) = sft { *s == sc.struct_name } else { false }
                        })
                    }).map(|(name, _)| name.clone())
                })
            }.ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(),
                format!("cannot find parent struct containing {}", sc.struct_name)))?;

            (parent_type, None, Some((ref_field_name.to_string(), target_block_field.to_string(), target_type)))
        } else {
            // Local block field on this struct
            let block_field_obj = fields.named.iter().find(|f|
                f.ident.as_ref().map(|i| i.to_string()) == Some(constraint.primary_block_field().to_string())
            );
            if block_field_obj.is_none() { continue; }
            let (a, b) = extract_block_type_args(&block_field_obj.unwrap().ty)?;
            (a, b, None)
        };

        let parent_name = constraint.parent_name.clone()
            .unwrap_or_else(|| a_type.to_lowercase());
        let a_type_ident = syn::Ident::new(&a_type, proc_macro2::Span::call_site());

        let is_self_block = b_type.is_none() && !is_remote_block && a_type != "__triplet__";
        let is_triplet = a_type == "__triplet__";

        // Multi-cross: constraint declares multiple block fields. Valid
        // only when the non-remote block fields are all CrossBlock (mixing
        // TripletBlock in a multi-block list is out of scope). Overrides
        // the primary-block-driven flags: in multi-cross mode, cross pairs
        // are routed per-block, not to a single CrossBlock or
        // TripletBlock. The entity-span setup (__all_idx +
        // triplet_entities) is shared with is_triplet. Multi-cross may
        // coexist with is_remote_block when the *primary* block is a
        // dotted-path remote reference (e.g. `pose.hb_pose`) and the
        // additional block fields are local CrossBlocks.
        let is_multi_cross = constraint.block_fields.len() > 1
            && !is_self_block && !is_triplet;

        // Does any declared local CrossBlock field reference the root
        // type? If so, root joins the constraint's entity list as an
        // implicit participant (no Ref<T> field needed; root accessed
        // via `&*__self_ref` / `&mut *self`).
        let root_type_str = root_name.to_string();
        let has_root_entity = is_multi_cross && constraint.block_fields.iter().any(|bf| {
            if bf.contains('.') { return false; }
            let Some(field) = fields.named.iter().find(|f|
                f.ident.as_ref().map(|i| i.to_string()) == Some(bf.clone())) else { return false; };
            let Ok((a, b_opt)) = extract_block_type_args(&field.ty) else { return false; };
            a == root_type_str || b_opt.as_deref() == Some(root_type_str.as_str())
        });

        // Self-primary + root-owned TripletBlock shape:
        //   #[arael(constraint(<local_self_block>, root.<triplet>, {...}))]
        // Primary block is the entity's own SelfBlock<Self>, secondary
        // is `root.<field>` naming a TripletBlock<T> on root. Body
        // touches both self params and root params; diagonal writes
        // land on each entity's SelfBlock<Self>, cross pairs go to the
        // root's TripletBlock (COO). Self is treated like an implicit
        // entity — parallel to `has_root_entity` for CrossBlock-backed
        // multi-cross, but with COO storage and no per-pair routing.
        let root_triplet_field: Option<syn::Ident> = if is_self_block {
            constraint.block_fields.iter()
                .filter_map(|bf| bf.strip_prefix("root.").map(|s| s.to_string()))
                .find_map(|rest| {
                    root_fields.iter().find(|f|
                        f.ident.as_ref().map(|i| i.to_string()) == Some(rest.clone()))
                        .and_then(|f| {
                            if let syn::Type::Path(tp) = &f.ty
                                && let Some(seg) = tp.path.segments.last()
                                && seg.ident == "TripletBlock" {
                                    return Some(syn::Ident::new(&rest, proc_macro2::Span::call_site()));
                                }
                            None
                        })
                })
        } else { None };
        let is_root_triplet_self = root_triplet_field.is_some();

        // For SelfBlock: the struct itself is in a root collection
        // For CrossBlock: find parent collection + frines field
        let self_var_name = if is_self_block {
            a_type.to_lowercase()
        } else {
            parent_name.clone()
        };

        // Resolve where the constrained entity lives on the root — a Vec/Deque/Arena
        // collection, a plain struct-typed field (direct composition), or the root
        // itself. Non-SelfBlock constraints still require a Collection (see below).
        // When the primary block is a dotted-path remote reference, the
        // iteration structure follows the remote path (iterate parent
        // collection -> frines) regardless of whether extra local
        // CrossBlock fields are declared for multi-cross routing.
        // Otherwise is_triplet / is_multi_cross / is_self_block all
        // iterate on the constraint struct itself (or its parent for
        // nested cross), so the struct name is the right coll_type.
        let coll_type = if is_remote_block { &a_type }
            else if is_triplet || is_multi_cross || is_self_block { &sc.struct_name }
            else { &a_type };
        let entity_location = match resolve_entity_location(root_fields, &root_name.to_string(), coll_type) {
            Some(loc) => loc,
            None => continue,
        };
        if !is_self_block && !matches!(entity_location, EntityLocation::Collection { .. }) {
            // TripletBlock / CrossBlock constraints without a root collection for
            // their A-type / constraint struct fall through — direct-composed
            // sub-models with cross-block constraints are not yet supported.
            continue;
        }
        // `coll_ident(_str)` is only consumed by the Collection SelfBlock path and the
        // nested CrossBlock path (both of which require a Collection). For DirectField
        // / RootSelf we divert below and these placeholders are never read.
        let (coll_ident_str, coll_ident) = match &entity_location {
            EntityLocation::Collection { field, .. } => {
                (field.clone(), syn::Ident::new(field, proc_macro2::Span::call_site()))
            }
            EntityLocation::DirectField { field } => {
                (String::new(), syn::Ident::new(field, proc_macro2::Span::call_site()))
            }
            EntityLocation::RootSelf => {
                (String::new(), root_name.clone())
            }
        };

        // CrossBlock/remote: find frines field and build ref resolution
        let mut frines_ident = None;
        let mut resolve_stmts = Vec::new();
        let mut parent_ident = None;
        let mut is_root_level_cross = false;  // constraint struct lives directly on root

        if is_triplet || is_multi_cross || (!is_self_block || is_remote_block) {
            // First try: constraint struct nested under A-type (e.g. PointFrine under PointLandmark)
            let parent_layout = registry_lookup(&a_type);
            let frines_field = parent_layout.as_ref().and_then(|l| {
                l.fields.iter().find(|(_, sft)| {
                    if let SymFieldType::Struct(s) = sft { s == &sc.struct_name } else { false }
                }).map(|(name, _)| name.clone())
            });

            if let Some(ff) = frines_field {
                // Nested case (e.g. PointFrine under PointLandmark)
                frines_ident = Some(syn::Ident::new(&ff, proc_macro2::Span::call_site()));
                parent_ident = Some(syn::Ident::new(&parent_name, proc_macro2::Span::call_site()));
            } else {
                // Root-level case (e.g. PosePair, CoincidentPP directly on root)
                // The constraint struct has its own root collection
                let root_coll = find_root_collection(root_fields, &sc.struct_name);
                if root_coll.is_none() { continue; }
                let (rc_name, _) = root_coll.unwrap();
                frines_ident = Some(syn::Ident::new(&rc_name, proc_macro2::Span::call_site()));
                is_root_level_cross = true;
            }

            let struct_layout = registry_lookup(&sc.struct_name);
            let ref_paths = struct_layout.as_ref().map(|l| l.ref_paths.clone()).unwrap_or_default();
            for (field_name, resolve_path) in &ref_paths {
                let field_ident_inner = syn::Ident::new(field_name, proc_macro2::Span::call_site());
                let adjusted_path = resolve_path.replace("root.", "self.");
                let resolve_expr: syn::Expr = syn::parse_str(
                    &format!("{}[__frine.{}]", adjusted_path, field_name)
                ).map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                    format!("failed to parse resolve path: {}", e)))?;
                resolve_stmts.push(quote! { let #field_ident_inner = &#resolve_expr; });
            }
        }

        // Re-process constraint body to get residual-only code and full code
        // We need the symbolic expressions again
        let root_name_str = root_name.to_string();
        let (residual_exprs, param_symbols) = interpret_constraint_body(
            &struct_ident, &fields.named, &constraint, &root_name_str)?;
        check_residual_coverage(sc, &struct_ident, &residual_exprs, &param_symbols)?;

        // Apply euler_angles substitutions from all referenced types
        let mut all_subs: Vec<(arael_sym::E, arael_sym::E)> = Vec::new();
        // Build var_infos to know which variables reference which types
        let struct_layout_for_subs = registry_lookup(&sc.struct_name);
        let ref_paths_for_subs = struct_layout_for_subs.as_ref()
            .map(|l| l.ref_paths.clone()).unwrap_or_default();
        // Check each variable's type for euler_angle_fields
        for (field_name, _) in &ref_paths_for_subs {
            if let Some(field) = fields.named.iter().find(|f|
                f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone()))
                && let Some((_, inner_ident)) = extract_wrapper_inner(&field.ty, "Ref") {
                    let type_name = inner_ident.to_string();
                    if let Some(layout) = registry_lookup(&type_name) {
                        for ea in &layout.euler_angle_fields {
                            all_subs.extend(build_euler_substitutions(field_name, ea));
                        }
                        for ea in &layout.universal_euler_angle_fields {
                            all_subs.extend(build_universal_euler_substitutions(field_name, ea));
                        }
                    }
                }
        }
        // Check the self/parent type for euler_angles
        if let Some(a_layout) = registry_lookup(&a_type) {
            for ea in &a_layout.euler_angle_fields {
                all_subs.extend(build_euler_substitutions(&self_var_name, ea));
            }
            for ea in &a_layout.universal_euler_angle_fields {
                all_subs.extend(build_universal_euler_substitutions(&self_var_name, ea));
            }
        }
        // For SelfBlock, also check the struct itself (it IS the A type)
        if is_self_block
            && let Some(self_layout) = registry_lookup(&sc.struct_name) {
                for ea in &self_layout.euler_angle_fields {
                    all_subs.extend(build_euler_substitutions(&self_var_name, ea));
                }
            }

        let block_ident = if is_remote_block {
            // For remote blocks, the actual block field name is the last segment
            let parts: Vec<&str> = constraint.primary_block_field().split('.').collect();
            syn::Ident::new(parts.last().unwrap(), proc_macro2::Span::call_site())
        } else {
            syn::Ident::new(constraint.primary_block_field(), proc_macro2::Span::call_site())
        };
        let param_strs: Vec<&str> = param_symbols.iter().map(|s| s.as_str()).collect();
        let n_params = param_symbols.len();
        let n_residuals = residual_exprs.len();

        // TripletBlock per-entity span info (needed by gh_stmts emission below).
        // Built alongside triplet_idx_stmts later in this same iteration —
        // collect it here so the emission has access. Each entry:
        // (ref-field ident bound in scope, entity type ident, dr-slice start,
        //  entity param count).
        let mut triplet_entities: Vec<(syn::Ident, syn::Ident, usize, usize)> = Vec::new();
        let multi_cross_routing: Vec<MultiCrossRouting>;
        if is_root_triplet_self {
            // Entities are [self, root] in that order. Self is accessed
            // via `__item` (iter_mut item on the struct's collection);
            // root is bound as `<root_lc>` from `let <root_lc> = &*__self_ref;`
            // in the surrounding emission loop.
            let self_layout = registry_lookup(&sc.struct_name);
            if let Some(layout) = self_layout {
                let mut self_count = 0usize;
                for pf in &layout.param_fields {
                    let size = layout.fields.iter()
                        .find(|(n, _)| n == pf)
                        .map(|(_, sft)| match sft {
                            SymFieldType::Scalar => 1usize,
                            SymFieldType::Vec2 => 2,
                            SymFieldType::Vec3 => 3,
                            _ => 0,
                        }).unwrap_or(0);
                    self_count += size;
                }
                if self_count > 0 {
                    triplet_entities.push((
                        syn::Ident::new("__item", proc_macro2::Span::call_site()),
                        syn::Ident::new(&sc.struct_name, proc_macro2::Span::call_site()),
                        0, self_count,
                    ));
                }
                if let Some(root_layout) = registry_lookup(&root_type_str) {
                    let mut root_count = 0usize;
                    for pf in &root_layout.param_fields {
                        let size = root_layout.fields.iter()
                            .find(|(n, _)| n == pf)
                            .map(|(_, sft)| match sft {
                                SymFieldType::Scalar => 1usize,
                                SymFieldType::Vec2 => 2,
                                SymFieldType::Vec3 => 3,
                                _ => 0,
                            }).unwrap_or(0);
                        root_count += size;
                    }
                    if root_count > 0 {
                        triplet_entities.push((
                            syn::Ident::new(&root_type_str.to_lowercase(), proc_macro2::Span::call_site()),
                            root_name.clone(),
                            self_count, root_count,
                        ));
                    }
                }
            }
        } else if is_triplet || is_multi_cross {
            let struct_layout = registry_lookup(&sc.struct_name);
            let ref_paths = struct_layout.as_ref().map(|l| l.ref_paths.clone()).unwrap_or_default();
            let mut used = std::collections::HashSet::new();
            let mut offset = 0usize;
            for (field_name, _) in &ref_paths {
                if !used.insert(field_name.clone()) { continue; }
                if let Some(field) = fields.named.iter().find(|f|
                    f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone()))
                    && let Some((_, inner_ident)) = extract_wrapper_inner(&field.ty, "Ref") {
                        let type_name = inner_ident.to_string();
                        if let Some(layout) = registry_lookup(&type_name) {
                            let var_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
                            let type_ident = syn::Ident::new(&type_name, proc_macro2::Span::call_site());
                            let entity_start = offset;
                            for pf in &layout.param_fields {
                                let size = layout.fields.iter()
                                    .find(|(n, _)| n == pf)
                                    .map(|(_, sft)| match sft {
                                        SymFieldType::Scalar => 1usize,
                                        SymFieldType::Vec2 => 2,
                                        SymFieldType::Vec3 => 3,
                                        _ => 0,
                                    }).unwrap_or(0);
                                offset += size;
                            }
                            let entity_count = offset - entity_start;
                            if entity_count > 0 {
                                triplet_entities.push((var_ident, type_ident, entity_start, entity_count));
                            }
                        }
                    }
            }
            // Append root as an implicit entity when any declared
            // CrossBlock references the root type. The var_ident is the
            // root's lowercased name (already bound in emitted scope as
            // `let <root_lc> = &*__self_ref;`).
            if has_root_entity
                && let Some(root_layout) = registry_lookup(&root_type_str)
            {
                let entity_start = offset;
                for pf in &root_layout.param_fields {
                    let size = root_layout.fields.iter()
                        .find(|(n, _)| n == pf)
                        .map(|(_, sft)| match sft {
                            SymFieldType::Scalar => 1usize,
                            SymFieldType::Vec2 => 2,
                            SymFieldType::Vec3 => 3,
                            _ => 0,
                        }).unwrap_or(0);
                    offset += size;
                }
                let entity_count = offset - entity_start;
                if entity_count > 0 {
                    let var_ident = syn::Ident::new(
                        &root_type_str.to_lowercase(), proc_macro2::Span::call_site());
                    let type_ident = root_name.clone();
                    triplet_entities.push((var_ident, type_ident, entity_start, entity_count));
                }
            }
        }

        // Multi-cross routing: one entry per declared CrossBlock on the
        // constraint struct. Every unordered ref pair in triplet_entities
        // must be claimed; ambiguous type matches without
        // `#[arael(cross = (refA, refB))]` are rejected. Empty Vec for
        // non-multi-cross constraints.
        multi_cross_routing = if is_multi_cross {
            build_multi_cross_routing(
                &fields, &constraint.block_fields, &triplet_entities, &struct_ident)?
        } else {
            Vec::new()
        };

        // A- and B-entity param counts (scalar width). Hoisted up here so
        // both the gh_stmts cross-emission and the index-building code can
        // use them. `param_symbols` is ordered A-first then B for cross
        // blocks; the first a_param_count derivatives correspond to A's
        // params, the next b_param_count to B's.
        let a_param_count = registry_lookup(&a_type).map(|l| l.param_fields.iter().map(|pf| {
            l.fields.iter().find(|(n, _)| n == pf).map(|(_, sft)| match sft {
                SymFieldType::Scalar => 1usize, SymFieldType::Vec2 => 2, SymFieldType::Vec3 => 3, _ => 0,
            }).unwrap_or(0)
        }).sum::<usize>()).unwrap_or(0);
        let b_param_count = b_type.as_ref().and_then(|b| registry_lookup(b)).map(|l| l.param_fields.iter().map(|pf| {
            l.fields.iter().find(|(n, _)| n == pf).map(|(_, sft)| match sft {
                SymFieldType::Scalar => 1usize, SymFieldType::Vec2 => 2, SymFieldType::Vec3 => 3, _ => 0,
            }).unwrap_or(0)
        }).sum::<usize>()).unwrap_or(0);

        // Resolve the A- and B-var idents for CrossBlock's 3-call emission.
        let a_var_ident_for_block: Option<syn::Ident> = if is_self_block {
            Some(syn::Ident::new("__item", proc_macro2::Span::call_site()))
        } else if is_root_level_cross {
            let struct_layout = registry_lookup(&sc.struct_name);
            let a_ref_field = struct_layout.as_ref().and_then(|l| {
                l.ref_paths.iter().find(|(field_name, _)| {
                    fields.named.iter().any(|f| {
                        f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone())
                            && extract_wrapper_inner(&f.ty, "Ref")
                                .map(|(_, id)| *id == a_type)
                                .unwrap_or(false)
                    })
                }).map(|(name, _)| name.clone())
            }).unwrap_or_else(|| a_type.to_lowercase());
            Some(syn::Ident::new(&a_ref_field, proc_macro2::Span::call_site()))
        } else {
            Some(syn::Ident::new("__item", proc_macro2::Span::call_site()))
        };
        let b_var_ident_for_block: Option<syn::Ident> = if let Some(ref b_type_name) = b_type {
            let struct_layout_b = registry_lookup(&sc.struct_name);
            let ref_paths_b = struct_layout_b.as_ref().map(|l| l.ref_paths.clone()).unwrap_or_default();
            let mut skip_first_match = is_root_level_cross && a_type == *b_type_name;
            ref_paths_b.iter().find(|(field_name, _)| {
                let matches = fields.named.iter().any(|f| {
                    f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone())
                        && extract_wrapper_inner(&f.ty, "Ref")
                            .map(|(_, id)| id.to_string() == *b_type_name)
                            .unwrap_or(false)
                });
                if matches && skip_first_match {
                    skip_first_match = false;
                    return false;
                }
                matches
            }).map(|(name, _)| syn::Ident::new(name, proc_macro2::Span::call_site()))
        } else {
            None
        };

        // --- Cost-only code: differentiate FIRST, then apply substitutions, then CSE ---
        // Apply substitutions to residuals (cost-only, no derivatives)
        let mut cost_exprs = residual_exprs.clone();
        apply_substitutions(&mut cost_exprs, &all_subs);
        let (cost_intermediates, cost_simplified) = arael_sym::cse(&cost_exprs);
        let mut cost_stmts = Vec::new();
        for (name, expr) in &cost_intermediates {
            let name_ident = syn::Ident::new(name, proc_macro2::Span::call_site());
            let code: Expr = parse_sym_code(&expr.to_rust(""))?;
            cost_stmts.push(quote! { let #name_ident= #code; });
        }
        for (ri, r) in cost_simplified.iter().enumerate() {
            let r_ident = syn::Ident::new(&format!("__r_{}", ri), proc_macro2::Span::call_site());
            let r_expr: Expr = parse_sym_code(&r.to_rust(""))?;
            cost_stmts.push(quote! {
                let #r_ident= #r_expr;
                __cost += (#r_ident as #cast_type) * (#r_ident as #cast_type);
            });
        }

        // --- Grad+hessian code with CSE ---
        // Collect all expressions: residuals + all derivatives (from originals, before substitution)
        let mut all_gh_exprs: Vec<arael_sym::E> = Vec::new();
        for r in &residual_exprs {
            all_gh_exprs.push(r.clone());
            for p in &param_strs {
                all_gh_exprs.push(r.diff(*p));
            }
        }
        // Apply substitutions AFTER differentiation, BEFORE CSE
        apply_substitutions(&mut all_gh_exprs, &all_subs);
        let (gh_intermediates, gh_simplified) = arael_sym::cse(&all_gh_exprs);

        let mut gh_stmts = Vec::new();

        // Cross-block prelude: cache mutable refs to A's and B's SelfBlocks
        // via unsafe raw-pointer cast. One-time, reused across all residuals.
        // Mirrors the remote-block pattern (`__target_block`) which coexists
        // fine with the body's immutable reads. The per-call unsafe{} form
        // tripped borrow-checker in multi-residual bodies.
        let is_cross_block = !is_self_block && !is_triplet && !is_remote_block && !is_multi_cross;
        if is_cross_block {
            let b_type_name = b_type.as_ref().expect("cross block requires B");
            let b_type_ident = syn::Ident::new(b_type_name, proc_macro2::Span::call_site());
            let a_hb = registry_lookup(&a_type)
                .and_then(|l| l.self_block_field.clone())
                .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                    format!("type `{}` must declare a `SelfBlock<Self>` field (cross-block participants need a self-block)", a_type)))?;
            let b_hb = registry_lookup(b_type_name)
                .and_then(|l| l.self_block_field.clone())
                .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                    format!("type `{}` must declare a `SelfBlock<Self>` field (cross-block participants need a self-block)", b_type_name)))?;
            let a_hb_ident = syn::Ident::new(&a_hb, proc_macro2::Span::call_site());
            let b_hb_ident = syn::Ident::new(&b_hb, proc_macro2::Span::call_site());
            let a_var_id = a_var_ident_for_block.as_ref()
                .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                    "cross-block constraint missing A-var binding"))?;
            let b_var_id = b_var_ident_for_block.as_ref()
                .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                    "cross-block constraint missing B-var binding"))?;
            // `&raw mut` → *mut, bypassing borrow checker's &mut tracking.
            // For nested cross (a_var = __item, already &mut Parent) taking the
            // whole __item via raw pointer conflicts with `__item.frines.iter_mut()`
            // in the surrounding loop. The narrow-field form is accepted
            // because hb_drift/hb_pose are disjoint from frines.
            // For root-level cross where a_var is bound immutably (e.g. `prev =
            // &self.poses[...]`), we must cast through *const→*mut to re-
            // acquire mutability; the macro chose this binding.
            let a_raw_expr: TokenStream2 = if is_root_level_cross {
                quote! { &raw mut (*(#a_var_id as *const #a_type_ident as *mut #a_type_ident)).#a_hb_ident }
            } else {
                // nested cross: a_var is already &mut (outer __item)
                quote! { &raw mut #a_var_id.#a_hb_ident }
            };
            let b_raw_expr: TokenStream2 = if is_root_level_cross {
                quote! { &raw mut (*(#b_var_id as *const #b_type_ident as *mut #b_type_ident)).#b_hb_ident }
            } else {
                // For nested cross B is the ref-resolved &mut Pose via resolve_stmts
                quote! { &raw mut (*(#b_var_id as *const #b_type_ident as *mut #b_type_ident)).#b_hb_ident }
            };
            gh_stmts.push(quote! {
                let __a_self_block_ptr: *mut _ = unsafe { #a_raw_expr };
                let __b_self_block_ptr: *mut _ = unsafe { #b_raw_expr };
            });
        }

        for (name, expr) in &gh_intermediates {
            let name_ident = syn::Ident::new(name, proc_macro2::Span::call_site());
            let code: Expr = parse_sym_code(&expr.to_rust(""))?;
            gh_stmts.push(quote! { let #name_ident= #code; });
        }

        // Pre-residual setup for is_root_triplet_self: build __all_idx
        // (concatenation of entity param indices, self-first) and
        // __entity_offsets once per __item iteration, so per-residual
        // TripletBlock.add_residual_cross calls can pass them directly.
        if is_root_triplet_self {
            let self_layout = registry_lookup(&sc.struct_name)
                .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                    format!("type `{}` not in registry", sc.struct_name)))?;
            let root_layout = registry_lookup(&root_type_str)
                .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                    format!("root type `{}` not in registry", root_type_str)))?;
            let param_size = |layout: &crate::SymLayout, pf: &str| -> usize {
                layout.fields.iter().find(|(n, _)| n == pf)
                    .map(|(_, sft)| match sft {
                        SymFieldType::Scalar => 1usize,
                        SymFieldType::Vec2 => 2,
                        SymFieldType::Vec3 => 3,
                        _ => 0,
                    }).unwrap_or(0)
            };
            let mut self_count = 0usize;
            let mut self_idx_stmts: Vec<TokenStream2> = Vec::new();
            for pf in &self_layout.param_fields {
                let pf_ident = syn::Ident::new(pf, proc_macro2::Span::call_site());
                let size = param_size(&self_layout, pf);
                let offset = self_count;
                let end = offset + size;
                self_idx_stmts.push(quote! {
                    __item.#pf_ident.write_indices(&mut __all_idx[#offset..#end]);
                });
                self_count += size;
            }
            let mut root_count = 0usize;
            let mut root_idx_stmts: Vec<TokenStream2> = Vec::new();
            for pf in &root_layout.param_fields {
                let pf_ident = syn::Ident::new(pf, proc_macro2::Span::call_site());
                let size = param_size(&root_layout, pf);
                let offset = self_count + root_count;
                let end = offset + size;
                root_idx_stmts.push(quote! {
                    (*__self_ref).#pf_ident.write_indices(&mut __all_idx[#offset..#end]);
                });
                root_count += size;
            }
            let total = self_count + root_count;
            let sc_u32 = self_count as u32;
            let total_u32 = total as u32;
            gh_stmts.push(quote! {
                let mut __all_idx = [0u32; #total];
                #(#self_idx_stmts)*
                #(#root_idx_stmts)*
                let __entity_offsets: [u32; 3] = [0u32, #sc_u32, #total_u32];
            });
        }

        let mut idx = 0;
        for ri in 0..n_residuals {
            let r_ident = syn::Ident::new(&format!("__r_{}", ri), proc_macro2::Span::call_site());
            let r_expr: Expr = parse_sym_code(&gh_simplified[idx].to_rust(""))?;
            // Accumulate the cost alongside the derivatives: the residual
            // value is already in hand, so the fused calc_cost_grad_hessian_*
            // entry points get the cost for free (saves a separate cost-only
            // model evaluation in the LM loop).
            gh_stmts.push(quote! {
                let #r_ident= #r_expr;
                __cost += (#r_ident as #cast_type) * (#r_ident as #cast_type);
            });
            idx += 1;

            let mut dr_idents = Vec::new();
            for pi in 0..n_params {
                let dr_ident = syn::Ident::new(&format!("__dr_{}_{}", ri, pi), proc_macro2::Span::call_site());
                let dr_expr: Expr = parse_sym_code(&gh_simplified[idx].to_rust(""))?;
                gh_stmts.push(quote! { let #dr_ident= #dr_expr; });
                dr_idents.push(dr_ident);
                idx += 1;
            }
            let dr_f64: Vec<TokenStream2> = dr_idents.iter().map(|d| quote! { #d as #cast_type }).collect();
            if is_triplet {
                // TripletBlock: per-entity SelfBlock writes grad + within-entity
                // diagonals; triplet block gets only cross-entity pairs.
                let mut triplet_calls: Vec<TokenStream2> = Vec::new();
                for (var_id, type_id, start, count) in &triplet_entities {
                    let hb = registry_lookup(&type_id.to_string())
                        .and_then(|l| l.self_block_field.clone())
                        .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                            format!("type `{}` must declare a `SelfBlock<Self>` field (required as triplet participant)", type_id)))?;
                    let hb_ident = syn::Ident::new(&hb, proc_macro2::Span::call_site());
                    let entity_dr: Vec<TokenStream2> = dr_f64.iter().skip(*start).take(*count).cloned().collect();
                    // Ref-resolved var is &T immutable via resolve_stmts; use raw-ptr cast.
                    triplet_calls.push(quote! {
                        unsafe {
                            (*(#var_id as *const #type_id as *mut #type_id)).#hb_ident
                                .add_residual(#r_ident as #cast_type, &[#(#entity_dr),*], grad);
                        }
                    });
                }
                gh_stmts.push(quote! {
                    #(#triplet_calls)*
                    __frine.#block_ident.add_residual_cross(
                        #r_ident as #cast_type,
                        &__all_idx,
                        &[#(#dr_f64),*],
                        &__entity_offsets,
                    );
                });
            } else if is_multi_cross {
                // Multi-cross: per-entity SelfBlock writes (same as triplet)
                // + one CrossBlock.add_residual_cross per declared CrossBlock
                // field. Every unordered pair of entities is covered by
                // exactly one CrossBlock (verified by routing build).
                //
                // Three per-entity emission modes:
                //   - Root entity (type matches root_name): write via
                //     (&mut *self).<root_hb>.add_residual — no Ref cast.
                //   - Remote-block primary (is_remote_block && type matches
                //     the remote target type): skip the per-entity call;
                //     the __target_block.add_residual below (inside is_remote
                //     branch) handles pose's SelfBlock.
                //   - Regular Ref entity: unsafe *const→*mut cast pattern.
                let root_ident_str = root_name.to_string();
                let remote_target_type: Option<String> = if is_remote_block {
                    remote_block_info.as_ref().map(|(_, _, t)| t.clone())
                } else { None };
                let mut self_block_calls: Vec<TokenStream2> = Vec::new();
                let mut remote_self_block_call: Option<TokenStream2> = None;
                for (var_id, type_id, start, count) in &triplet_entities {
                    let entity_dr: Vec<TokenStream2> = dr_f64.iter().skip(*start).take(*count).cloned().collect();
                    let type_id_str = type_id.to_string();
                    if type_id_str == root_ident_str {
                        // Root: access via `__self_ref` with *const→*mut cast to
                        // the root struct type. Look up root's SelfBlock field
                        // name from the registry.
                        let hb = registry_lookup(&type_id_str)
                            .and_then(|l| l.self_block_field.clone())
                            .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                                format!("root type `{}` must declare a `SelfBlock<Self>` field (required as implicit multi-cross participant)", type_id)))?;
                        let hb_ident = syn::Ident::new(&hb, proc_macro2::Span::call_site());
                        self_block_calls.push(quote! {
                            unsafe {
                                (*(__self_ref as *const #type_id as *mut #type_id)).#hb_ident
                                    .add_residual(#r_ident as #cast_type, &[#(#entity_dr),*], grad);
                            }
                        });
                        continue;
                    }
                    if remote_target_type.as_deref() == Some(type_id_str.as_str()) {
                        // Remote primary: defer to __target_block call.
                        remote_self_block_call = Some(quote! {
                            __target_block.add_residual(#r_ident as #cast_type, &[#(#entity_dr),*], grad);
                        });
                        continue;
                    }
                    let hb = registry_lookup(&type_id_str)
                        .and_then(|l| l.self_block_field.clone())
                        .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                            format!("type `{}` must declare a `SelfBlock<Self>` field (required as multi-cross participant)", type_id)))?;
                    let hb_ident = syn::Ident::new(&hb, proc_macro2::Span::call_site());
                    self_block_calls.push(quote! {
                        unsafe {
                            (*(#var_id as *const #type_id as *mut #type_id)).#hb_ident
                                .add_residual(#r_ident as #cast_type, &[#(#entity_dr),*], grad);
                        }
                    });
                }
                let mut cross_block_calls: Vec<TokenStream2> = Vec::new();
                for route in &multi_cross_routing {
                    let block = &route.block_ident;
                    let dr_a: Vec<TokenStream2> = dr_f64.iter()
                        .skip(route.a_start).take(route.a_count).cloned().collect();
                    let dr_b: Vec<TokenStream2> = dr_f64.iter()
                        .skip(route.b_start).take(route.b_count).cloned().collect();
                    cross_block_calls.push(quote! {
                        __frine.#block.add_residual_cross(
                            #r_ident as #cast_type,
                            &[#(#dr_a),*],
                            &[#(#dr_b),*],
                        );
                    });
                }
                gh_stmts.push(quote! {
                    #(#self_block_calls)*
                    #remote_self_block_call
                    #(#cross_block_calls)*
                });
            } else if is_remote_block {
                gh_stmts.push(quote! {
                    __target_block.add_residual(#r_ident as #cast_type, &[#(#dr_f64),*], grad);
                });
            } else if is_self_block {
                if is_root_triplet_self {
                    // Self-primary + root-owned TripletBlock. dr_f64 is
                    // [dr_self..., dr_root...]. Three writes preserve
                    // every J^T J pair:
                    //   1. __item.<hb_self>.add_residual  -- (self, self)
                    //      diagonal + grad.
                    //   2. (*__self_ref).<hb_root>.add_residual -- (root,
                    //      root) diagonal + grad.
                    //   3. (*__self_ref).<hbt>.add_residual_cross -- the
                    //      (self, root) across-entity block, COO storage.
                    let (_, _, _, self_count) = triplet_entities[0];
                    let (_, _, root_start, root_count) = triplet_entities[1];
                    let dr_self: Vec<TokenStream2> =
                        dr_f64.iter().take(self_count).cloned().collect();
                    let dr_root: Vec<TokenStream2> =
                        dr_f64.iter().skip(root_start).take(root_count).cloned().collect();
                    let root_hb = registry_lookup(&root_type_str)
                        .and_then(|l| l.self_block_field.clone())
                        .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                            format!("root type `{}` must declare a `SelfBlock<Self>` field (required as root-triplet participant)", root_type_str)))?;
                    let root_hb_ident = syn::Ident::new(&root_hb, proc_macro2::Span::call_site());
                    let triplet_ident = root_triplet_field.as_ref().unwrap();
                    gh_stmts.push(quote! {
                        __item.#block_ident.add_residual(#r_ident as #cast_type, &[#(#dr_self),*], grad);
                        unsafe {
                            (*(__self_ref as *const #root_name as *mut #root_name)).#root_hb_ident
                                .add_residual(#r_ident as #cast_type, &[#(#dr_root),*], grad);
                        }
                        unsafe {
                            (*(__self_ref as *const #root_name as *mut #root_name)).#triplet_ident
                                .add_residual_cross(
                                    #r_ident as #cast_type,
                                    &__all_idx,
                                    &[#(#dr_f64),*],
                                    &__entity_offsets,
                                );
                        }
                    });
                } else {
                    gh_stmts.push(quote! {
                        __item.#block_ident.add_residual(#r_ident as #cast_type, &[#(#dr_f64),*], grad);
                    });
                }
            } else {
                // CrossBlock: split dr into dr_a (first a_param_count) + dr_b
                // (next b_param_count). Three calls: A's SelfBlock gets
                // grad[A] + H[A,A] diagonal; B's SelfBlock same for B; the
                // cross block holds only the A-B rectangular cross Hessian.
                // __a_self_block_ptr / __b_self_block_ptr cached at top of gh_stmts.
                let dr_a: Vec<TokenStream2> = dr_f64.iter().take(a_param_count).cloned().collect();
                let dr_b: Vec<TokenStream2> = dr_f64.iter().skip(a_param_count).take(b_param_count).cloned().collect();
                gh_stmts.push(quote! {
                    unsafe { (*__a_self_block_ptr).add_residual(#r_ident as #cast_type, &[#(#dr_a),*], grad); }
                    unsafe { (*__b_self_block_ptr).add_residual(#r_ident as #cast_type, &[#(#dr_b),*], grad); }
                    __frine.#block_ident.add_residual_cross(#r_ident as #cast_type, &[#(#dr_a),*], &[#(#dr_b),*]);
                });
            }
        }

        // --- Jacobian code: same intermediates + residuals + derivatives, push rows ---
        let mut jac_stmts = Vec::new();
        if jacobian {
            // Reuse the same CSE'd expressions
            for (name, expr) in &gh_intermediates {
                let name_ident = syn::Ident::new(name, proc_macro2::Span::call_site());
                let code: Expr = parse_sym_code(&expr.to_rust(""))?;
                jac_stmts.push(quote! { let #name_ident= #code; });
            }
            let mut jidx = 0;
            for ri in 0..n_residuals {
                let r_ident = syn::Ident::new(&format!("__r_{}", ri), proc_macro2::Span::call_site());
                let r_expr: Expr = parse_sym_code(&gh_simplified[jidx].to_rust(""))?;
                jac_stmts.push(quote! { let #r_ident= #r_expr; });
                jidx += 1;

                let mut dr_idents = Vec::new();
                for pi in 0..n_params {
                    let dr_ident = syn::Ident::new(&format!("__dr_{}_{}", ri, pi), proc_macro2::Span::call_site());
                    let dr_expr: Expr = parse_sym_code(&gh_simplified[jidx].to_rust(""))?;
                    jac_stmts.push(quote! { let #dr_ident= #dr_expr; });
                    dr_idents.push(dr_ident);
                    jidx += 1;
                }
                let dr_f64: Vec<TokenStream2> = dr_idents.iter().map(|d| quote! { #d as #cast_type }).collect();
                jac_stmts.push(quote! {
                    __jac_rows.push(arael::model::JacobianRow {
                        constraint: __jac_cid,
                        label: #label_literal,
                        residual: #r_ident as #cast_type,
                        entries: arael::model::jacobian_entries(&__jac_idx, &[#(#dr_f64),*]),
                    });
                });
            }
        }

        // Build index setup code — separate A (parent) and B (ref) indices
        let mut a_idx_stmts = Vec::new();
        let mut b_idx_stmts = Vec::new();
        if let Some(a_layout) = registry_lookup(&a_type) {
            let mut offset = 0usize;
            for pf in &a_layout.param_fields {
                let pf_ident = syn::Ident::new(pf, proc_macro2::Span::call_site());
                let size = a_layout.fields.iter()
                    .find(|(n, _)| n == pf)
                    .map(|(_, sft)| match sft {
                        SymFieldType::Scalar => 1usize,
                        SymFieldType::Vec2 => 2,
                        SymFieldType::Vec3 => 3,
                        _ => 0,
                    }).unwrap_or(0);
                let end = offset + size;
                let a_item = if is_self_block {
                    syn::Ident::new("__item", proc_macro2::Span::call_site())
                } else if is_root_level_cross {
                    // For root-level cross, A-type ref is the first Ref<A> field on the constraint struct
                    let struct_layout = registry_lookup(&sc.struct_name);
                    let a_ref_field = struct_layout.as_ref().and_then(|l| {
                        l.ref_paths.iter().find(|(field_name, _)| {
                            fields.named.iter().any(|f| {
                                f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone())
                                    && extract_wrapper_inner(&f.ty, "Ref")
                                        .map(|(_, id)| *id == a_type)
                                        .unwrap_or(false)
                            })
                        }).map(|(name, _)| name.clone())
                    }).unwrap_or_else(|| a_type.to_lowercase());
                    syn::Ident::new(&a_ref_field, proc_macro2::Span::call_site())
                } else {
                    syn::Ident::new("__item", proc_macro2::Span::call_site())
                };
                a_idx_stmts.push(quote! {
                    #a_item.#pf_ident.write_indices(&mut __a_idx[#offset..#end]);
                });
                offset = end;
            }
        }
        if let Some(ref b_type_name) = b_type
            && let Some(b_layout) = registry_lookup(b_type_name) {
                // Find ref field matching B type
                let struct_layout_b = registry_lookup(&sc.struct_name);
                let ref_paths_b = struct_layout_b.as_ref().map(|l| l.ref_paths.clone()).unwrap_or_default();
                // For root-level cross where A==B, the first ref is used for A,
                // so skip it to find B's ref (the second one of the same type)
                let mut skip_first_match = is_root_level_cross && a_type == *b_type_name;
                let b_ref_field = ref_paths_b.iter().find(|(field_name, _)| {
                    let matches = fields.named.iter().any(|f| {
                        f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone())
                            && extract_wrapper_inner(&f.ty, "Ref")
                                .map(|(_, id)| id.to_string() == *b_type_name)
                                .unwrap_or(false)
                    });
                    if matches && skip_first_match {
                        skip_first_match = false;
                        return false; // skip first match (used for A)
                    }
                    matches
                });
                if let Some((b_field_name, _)) = b_ref_field {
                    let b_var_ident = syn::Ident::new(b_field_name, proc_macro2::Span::call_site());
                    let mut offset = 0usize;
                    for pf in &b_layout.param_fields {
                        let pf_ident = syn::Ident::new(pf, proc_macro2::Span::call_site());
                        let size = b_layout.fields.iter()
                            .find(|(n, _)| n == pf)
                            .map(|(_, sft)| match sft {
                                SymFieldType::Scalar => 1usize,
                                SymFieldType::Vec2 => 2,
                                SymFieldType::Vec3 => 3,
                                _ => 0,
                            }).unwrap_or(0);
                        let end = offset + size;
                        b_idx_stmts.push(quote! {
                            #b_var_ident.#pf_ident.write_indices(&mut __b_idx[#offset..#end]);
                        });
                        offset = end;
                    }
                }
            }

        let a_param_count = registry_lookup(&a_type).map(|l| l.param_fields.iter().map(|pf| {
            l.fields.iter().find(|(n, _)| n == pf).map(|(_, sft)| match sft {
                SymFieldType::Scalar => 1usize, SymFieldType::Vec2 => 2, SymFieldType::Vec3 => 3, _ => 0,
            }).unwrap_or(0)
        }).sum::<usize>()).unwrap_or(0);
        let b_param_count = b_type.as_ref().and_then(|b| registry_lookup(b)).map(|l| l.param_fields.iter().map(|pf| {
            l.fields.iter().find(|(n, _)| n == pf).map(|(_, sft)| match sft {
                SymFieldType::Scalar => 1usize, SymFieldType::Vec2 => 2, SymFieldType::Vec3 => 3, _ => 0,
            }).unwrap_or(0)
        }).sum::<usize>()).unwrap_or(0);

        // TripletBlock: build flat index array from all ref fields.
        // Entity span layout is computed above for gh_stmts; here we emit the
        // write_indices() calls per-param-field.
        let mut triplet_idx_stmts: Vec<TokenStream2> = Vec::new();
        let mut triplet_param_count = 0usize;
        if is_triplet || is_multi_cross {
            let struct_layout = registry_lookup(&sc.struct_name);
            let ref_paths = struct_layout.as_ref().map(|l| l.ref_paths.clone()).unwrap_or_default();
            let mut used = std::collections::HashSet::new();
            for (field_name, _) in &ref_paths {
                if !used.insert(field_name.clone()) { continue; }
                if let Some(field) = fields.named.iter().find(|f|
                    f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone()))
                    && let Some((_, inner_ident)) = extract_wrapper_inner(&field.ty, "Ref") {
                        let type_name = inner_ident.to_string();
                        if let Some(layout) = registry_lookup(&type_name) {
                            let var_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
                            for pf in &layout.param_fields {
                                let pf_ident = syn::Ident::new(pf, proc_macro2::Span::call_site());
                                let size = layout.fields.iter()
                                    .find(|(n, _)| n == pf)
                                    .map(|(_, sft)| match sft {
                                        SymFieldType::Scalar => 1usize,
                                        SymFieldType::Vec2 => 2,
                                        SymFieldType::Vec3 => 3,
                                        _ => 0,
                                    }).unwrap_or(0);
                                let offset = triplet_param_count;
                                let end = offset + size;
                                triplet_idx_stmts.push(quote! {
                                    #var_ident.#pf_ident.write_indices(&mut __all_idx[#offset..#end]);
                                });
                                triplet_param_count += size;
                            }
                        }
                    }
            }
            // Append root's write_indices calls when root is an implicit
            // entity. Accessed via (*__self_ref).<param> since root is the
            // enclosing `Self`.
            if has_root_entity
                && let Some(root_layout) = registry_lookup(&root_type_str)
            {
                for pf in &root_layout.param_fields {
                    let pf_ident = syn::Ident::new(pf, proc_macro2::Span::call_site());
                    let size = root_layout.fields.iter()
                        .find(|(n, _)| n == pf)
                        .map(|(_, sft)| match sft {
                            SymFieldType::Scalar => 1usize,
                            SymFieldType::Vec2 => 2,
                            SymFieldType::Vec3 => 3,
                            _ => 0,
                        }).unwrap_or(0);
                    let offset = triplet_param_count;
                    let end = offset + size;
                    triplet_idx_stmts.push(quote! {
                        (*__self_ref).#pf_ident.write_indices(&mut __all_idx[#offset..#end]);
                    });
                    triplet_param_count += size;
                }
            }
        }
        // Cumulative entity offsets derived from triplet_entities (for the
        // add_residual span boundaries).
        let triplet_entity_offsets: Vec<u32> = {
            let mut v: Vec<u32> = vec![0];
            for (_, _, _start, count) in &triplet_entities {
                let next = v.last().unwrap() + *count as u32;
                v.push(next);
            }
            v
        };

        // Parse guard expression — replace "self" with the loop variable
        let guard_expr: Option<syn::Expr> = constraint.guard.as_ref()
            .map(|g| {
                let adjusted = if is_self_block {
                    g.replacen("self.", &format!("{}.", self_var_name), 10)
                } else {
                    // CrossBlock/TripletBlock: "self." refers to the constraint struct (__frine)
                    g.replacen("self.", "__frine.", 10)
                };
                syn::parse_str(&adjusted)
            })
            .transpose()
            .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                format!("failed to parse guard expression: {}", e)))?;

        if is_remote_block {
            // Remote block: iterate parent collection -> frines,
            // but write to a block on the referenced struct (e.g. pose.hb_pose)
            let frines_ident = frines_ident.unwrap();
            let parent_ident = parent_ident.unwrap();
            let (ref_field_name, _, target_type) = remote_block_info.as_ref().unwrap();
            let ref_field_ident = syn::Ident::new(ref_field_name, proc_macro2::Span::call_site());
            let target_type_ident = syn::Ident::new(target_type, proc_macro2::Span::call_site());

            // Find the root collection that contains the target type (for resolving the ref)
            let target_coll = find_root_collection(root_fields, target_type);
            let target_coll_ident = target_coll.map(|(name, _)|
                syn::Ident::new(&name, proc_macro2::Span::call_site()));

            // Build index setup for target type's params
            let target_layout = registry_lookup(target_type);
            let target_param_count = target_layout.as_ref().map(|l| l.param_fields.iter().map(|pf| {
                l.fields.iter().find(|(n, _)| n == pf).map(|(_, sft)| match sft {
                    SymFieldType::Scalar => 1usize, SymFieldType::Vec2 => 2, SymFieldType::Vec3 => 3, _ => 0,
                }).unwrap_or(0)
            }).sum::<usize>()).unwrap_or(0);

            let mut target_idx_stmts = Vec::new();
            if let Some(ref tl) = target_layout {
                let mut offset = 0usize;
                for pf in &tl.param_fields {
                    let pf_ident = syn::Ident::new(pf, proc_macro2::Span::call_site());
                    let size = tl.fields.iter().find(|(n, _)| n == pf).map(|(_, sft)| match sft {
                        SymFieldType::Scalar => 1usize, SymFieldType::Vec2 => 2, SymFieldType::Vec3 => 3, _ => 0,
                    }).unwrap_or(0);
                    let end = offset + size;
                    target_idx_stmts.push(quote! {
                        __target_ref.#pf_ident.write_indices(&mut __a_idx[#offset..#end]);
                    });
                    offset = end;
                }
            }

            let marker = source_marker(sc);

            // Cost loop: iterate parent -> frines, resolve refs, evaluate
            cost_loops.push(quote! {
                {
                    #marker
                    for __lm in self.#coll_ident.iter() {
                        for __frine in &__lm.#frines_ident {
                            #(#resolve_stmts)*
                            let #parent_ident = __lm;
                            let #root_var_ident = &*__self_ref;
                            #(#cost_stmts)*
                        }
                    }
                }
            });

            // Grad+hessian loop: same traversal but get mutable access
            // to target block. When is_multi_cross also holds (primary
            // remote block + extra local CrossBlocks), switch to
            // iter_mut so __frine.<local_cross>.add_residual_cross and
            // set_indices see a &mut Frine.
            let _target_coll_id = target_coll_ident.unwrap();
            let marker_gh = marker.clone();
            let entity_self_indices: Vec<TokenStream2> = {
                // Per-entity SelfBlock set_indices + __all_idx setup,
                // needed when any Ref entity (other than the remote
                // target) participates in a local CrossBlock and so
                // needs its hb.indices set. Only populated when
                // is_multi_cross.
                if is_multi_cross {
                    let root_ident_str_local = root_name.to_string();
                    let mut v: Vec<TokenStream2> = Vec::new();
                    for (var_id, type_id, start, count) in &triplet_entities {
                        if type_id.to_string() == root_ident_str_local { continue; }
                        if *count == 0 { continue; }
                        // Remote target's SelfBlock is set via __target_block below; skip.
                        if type_id.to_string() == *target_type { continue; }
                        let hb = registry_lookup(&type_id.to_string())
                            .and_then(|l| l.self_block_field.clone())
                            .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                                format!("type `{}` must declare a `SelfBlock<Self>` field (required as multi-cross/remote participant for set_indices)", type_id)))?;
                        let hb_ident = syn::Ident::new(&hb, proc_macro2::Span::call_site());
                        let end = start + count;
                        let cnt = *count;
                        v.push(quote! {
                            unsafe {
                                (*(#var_id as *const #type_id as *mut #type_id)).#hb_ident.set_indices(
                                    <&[u32; #cnt]>::try_from(&__all_idx[#start..#end]).unwrap()
                                );
                            }
                        });
                    }
                    v
                } else { Vec::new() }
            };
            let tp_remote = triplet_param_count;
            let triplet_idx_stmts_remote = triplet_idx_stmts.clone();
            if is_multi_cross {
                grad_hessian_loops.push(quote! {
                    {
                        #marker_gh
                        for __lm in self.#coll_ident.iter_mut() {
                            let #parent_ident = unsafe { &*(__lm as *const #a_type_ident) };
                            for __frine in __lm.#frines_ident.iter_mut() {
                                #(#resolve_stmts)*
                                let #root_var_ident = &*__self_ref;
                                let __target_block = unsafe {
                                    &mut (*(#ref_field_ident
                                        as *const #target_type_ident as *mut #target_type_ident)).#block_ident
                                };
                                { #(#gh_stmts)* }
                            }
                        }
                    }
                });
            } else {
                grad_hessian_loops.push(quote! {
                    {
                        #marker_gh
                        for __lm in self.#coll_ident.iter() {
                            let #parent_ident = __lm;
                            for __frine in &__lm.#frines_ident {
                                #(#resolve_stmts)*
                                let #root_var_ident = &*__self_ref;
                                let __target_block = unsafe {
                                    &mut (*(#ref_field_ident
                                        as *const #target_type_ident as *mut #target_type_ident)).#block_ident
                                };
                                { #(#gh_stmts)* }
                            }
                        }
                    }
                });
            }

            if is_multi_cross {
                // Multi-cross remote: emit per-CrossBlock set_indices on
                // each frine alongside the target (remote) set_indices.
                let mcb_calls: Vec<TokenStream2> = multi_cross_routing.iter().map(|r| {
                    let block = &r.block_ident;
                    let a_start = r.a_start; let a_end = r.a_start + r.a_count;
                    let b_start = r.b_start; let b_end = r.b_start + r.b_count;
                    quote! {
                        __frine.#block.set_indices(
                            &__all_idx[#a_start..#a_end],
                            &__all_idx[#b_start..#b_end],
                        );
                    }
                }).collect();
                set_block_indices_loops.push(quote! {
                    for __lm in self.#coll_ident.iter_mut() {
                        let #parent_ident = unsafe { &*(__lm as *const #a_type_ident) };
                        for __frine in __lm.#frines_ident.iter_mut() {
                            #(#resolve_stmts)*
                            let __target_ref = #ref_field_ident;
                            let __target_block = unsafe {
                                &mut (*(#ref_field_ident
                                    as *const #target_type_ident as *mut #target_type_ident)).#block_ident
                            };
                            let mut __a_idx = [0u32; #target_param_count];
                            #(#target_idx_stmts)*
                            __target_block.set_indices(&__a_idx);
                            let mut __all_idx = [0u32; #tp_remote];
                            #(#triplet_idx_stmts_remote)*
                            #(#entity_self_indices)*
                            #(#mcb_calls)*
                        }
                    }
                });
            } else {
                set_block_indices_loops.push(quote! {
                    for __lm in self.#coll_ident.iter() {
                        for __frine in &__lm.#frines_ident {
                            #(#resolve_stmts)*
                            let __target_ref = #ref_field_ident;
                            let __target_block = unsafe {
                                &mut (*(#ref_field_ident
                                    as *const #target_type_ident as *mut #target_type_ident)).#block_ident
                            };
                            let mut __a_idx = [0u32; #target_param_count];
                            #(#target_idx_stmts)*
                            __target_block.set_indices(&__a_idx);
                        }
                    }
                });
            }
        } else if is_self_block {
            let self_var = syn::Ident::new(&self_var_name, proc_macro2::Span::call_site());
            let marker = source_marker(sc);

            let cost_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #marker #(#cost_stmts)* } }
            } else {
                quote! { { #marker #(#cost_stmts)* } }
            };

            let gh_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #marker #(#gh_stmts)* } }
            } else {
                quote! { { #marker #(#gh_stmts)* } }
            };

            let jac_entry = if !jac_stmts.is_empty() {
                let entry = if let Some(ref guard) = guard_expr {
                    quote! { if #guard { #marker #(#jac_stmts)* } }
                } else {
                    quote! { { #marker #(#jac_stmts)* } }
                };
                Some(entry)
            } else { None };

            match &entity_location {
                EntityLocation::Collection { .. } => {
                    // SelfBlock on a Vec/Deque/Arena: group by collection name for merged-loop emission.
                    let group_key = coll_ident_str.clone();
                    let group = collection_groups.entry(group_key).or_insert_with(|| CollectionGroup {
                        coll_ident: coll_ident.clone(),
                        self_var: self_var.clone(),
                        a_type_ident: a_type_ident.clone(),
                        self_block: None,
                        cost_entries: Vec::new(),
                        gh_entries: Vec::new(),
                        jac_entries: Vec::new(),
                        nested_cost_loops: Vec::new(),
                        nested_gh_loops: Vec::new(),
                        nested_jac_loops: Vec::new(),
                    });
                    if group.self_block.is_none() {
                        group.self_block = Some(SelfBlockInfo {
                            a_param_count,
                            a_idx_stmts: a_idx_stmts.clone(),
                            block_ident: block_ident.clone(),
                        });
                    }
                    group.cost_entries.push(cost_entry);
                    group.gh_entries.push(gh_entry);
                    if let Some(je) = jac_entry { group.jac_entries.push(je); }
                }
                EntityLocation::RootSelf | EntityLocation::DirectField { .. } => {
                    // SelfBlock on the root itself or on a direct-composed sub-model:
                    // emit a single evaluation (no loop). Group by access path so
                    // multiple constraints on the same entity merge into one block.
                    let (group_key, accessor_read, accessor_write) = match &entity_location {
                        EntityLocation::RootSelf => (
                            "self".to_string(),
                            quote! { &*__self_ref },
                            quote! { &mut *self },
                        ),
                        EntityLocation::DirectField { field } => {
                            let fi = syn::Ident::new(field, proc_macro2::Span::call_site());
                            (
                                format!("self.{}", field),
                                quote! { &self.#fi },
                                quote! { &mut self.#fi },
                            )
                        }
                        _ => unreachable!(),
                    };
                    let ci_field = registry_lookup(&sc.struct_name)
                        .and_then(|l| l.constraint_index_field.as_ref().map(|f| {
                            syn::Ident::new(f, proc_macro2::Span::call_site())
                        }));
                    let group = single_instance_groups.entry(group_key).or_insert_with(|| SingleInstanceGroup {
                        accessor_read,
                        accessor_write,
                        self_var: self_var.clone(),
                        root_var_ident: root_var_ident.clone(),
                        a_type_ident: a_type_ident.clone(),
                        a_param_count,
                        a_idx_stmts: a_idx_stmts.clone(),
                        block_ident: block_ident.clone(),
                        constraint_index_field: ci_field,
                        cost_entries: Vec::new(),
                        gh_entries: Vec::new(),
                        jac_entries: Vec::new(),
                    });
                    group.cost_entries.push(cost_entry);
                    group.gh_entries.push(gh_entry);
                    if let Some(je) = jac_entry { group.jac_entries.push(je); }
                }
            }
        } else if is_triplet || is_multi_cross {
            // TripletBlock or multi-CrossBlock: N-ary constraint, flat
            // iteration on root collection. Both share the outer loop +
            // __all_idx setup; emission differs only in the gh_stmts
            // contents (single TripletBlock call vs. per-pair CrossBlock
            // calls). set_block_indices is populated with per-CrossBlock
            // set_indices when multi_cross_blocks is non-empty.
            let rc_ident = frines_ident.unwrap();
            let group_key = rc_ident.to_string();
            let marker = source_marker(sc);

            let cost_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #marker #(#cost_stmts)* } }
            } else {
                quote! { { #marker #(#cost_stmts)* } }
            };
            let gh_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #marker #(#gh_stmts)* } }
            } else {
                quote! { { #marker #(#gh_stmts)* } }
            };
            let jac_entry = if !jac_stmts.is_empty() {
                if let Some(ref guard) = guard_expr {
                    Some(quote! { if #guard { #marker #(#jac_stmts)* } })
                } else {
                    Some(quote! { { #marker #(#jac_stmts)* } })
                }
            } else { None };

            // Convert multi_cross_routing entries into MultiCrossBlockInfo
            // for the group. Empty for single-TripletBlock constraints.
            let mcb: Vec<MultiCrossBlockInfo> = multi_cross_routing.iter().map(|r| {
                MultiCrossBlockInfo {
                    block_ident: r.block_ident.clone(),
                    a_start: r.a_start, a_count: r.a_count,
                    b_start: r.b_start, b_count: r.b_count,
                }
            }).collect();

            // Build per-entity SelfBlock.set_indices calls for this group.
            // Skipped entirely for the root entity (handled globally by
            // root_self_block_prelude). Each call uses a slice of
            // __all_idx and converts it to a fixed-size array via
            // TryFrom so SelfBlock's `&[u32; N]` signature is satisfied.
            let root_ident_str = root_name.to_string();
            let mut entity_set_indices: Vec<TokenStream2> = Vec::new();
            for (var_id, type_id, start, count) in &triplet_entities {
                if type_id.to_string() == root_ident_str { continue; }
                if *count == 0 { continue; }
                let hb = registry_lookup(&type_id.to_string())
                    .and_then(|l| l.self_block_field.clone())
                    .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                        format!("type `{}` must declare a `SelfBlock<Self>` field (required as multi-cross/triplet participant for set_indices)", type_id)))?;
                let hb_ident = syn::Ident::new(&hb, proc_macro2::Span::call_site());
                let end = start + count;
                let cnt = *count;
                entity_set_indices.push(quote! {
                    unsafe {
                        (*(#var_id as *const #type_id as *mut #type_id)).#hb_ident.set_indices(
                            <&[u32; #cnt]>::try_from(&__all_idx[#start..#end]).unwrap()
                        );
                    }
                });
            }

            let group = triplet_groups.entry(group_key).or_insert_with(|| {
                let ci_field = crate::registry_lookup(&sc.struct_name)
                    .and_then(|l| l.constraint_index_field.as_ref().map(|f| {
                        syn::Ident::new(f, proc_macro2::Span::call_site())
                    }));
                TripletCollectionGroup {
                    rc_ident: rc_ident.clone(),
                    triplet_param_count,
                    block_ident: block_ident.clone(),
                    constraint_index_field: ci_field,
                    triplet_idx_stmts: triplet_idx_stmts.clone(),
                    entity_offsets: triplet_entity_offsets.clone(),
                    resolve_stmts: resolve_stmts.clone(),
                    root_var_ident: root_var_ident.clone(),
                    cost_entries: Vec::new(),
                    gh_entries: Vec::new(),
                    jac_entries: Vec::new(),
                    multi_cross_blocks: mcb.clone(),
                    entity_self_indices: entity_set_indices.clone(),
                }
            });
            // If multi_cross_blocks was empty at group creation (first
            // attribute was a TripletBlock) but a subsequent attribute is
            // multi-cross -- error. For now we reject mixed.
            if group.multi_cross_blocks.is_empty() != mcb.is_empty() {
                return Err(syn::Error::new_spanned(&struct_ident,
                    format!("on `{}`: cannot mix TripletBlock and multi-CrossBlock constraint attributes on the same struct", struct_ident)));
            }
            group.cost_entries.push(cost_entry);
            group.gh_entries.push(gh_entry);
            if let Some(je) = jac_entry { group.jac_entries.push(je); }
        } else if is_root_level_cross {
            // Root-level CrossBlock: constraint struct is directly on root (e.g. PosePair, CoincidentPP)
            // Flat iteration, no nesting. Multiple #[arael(constraint(...))] attributes on the
            // same struct are merged into a single loop per collection via cross_groups.
            let rc_ident = frines_ident.unwrap();
            let group_key = rc_ident.to_string();
            let marker = source_marker(sc);

            let cost_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #marker #(#cost_stmts)* } }
            } else {
                quote! { { #marker #(#cost_stmts)* } }
            };
            let gh_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #marker #(#gh_stmts)* } }
            } else {
                quote! { { #marker #(#gh_stmts)* } }
            };
            let jac_entry = if !jac_stmts.is_empty() {
                if let Some(ref guard) = guard_expr {
                    Some(quote! { if #guard { #marker #(#jac_stmts)* } })
                } else {
                    Some(quote! { { #marker #(#jac_stmts)* } })
                }
            } else { None };

            let group = cross_groups.entry(group_key).or_insert_with(|| {
                let ci_field = crate::registry_lookup(&sc.struct_name)
                    .and_then(|l| l.constraint_index_field.as_ref().map(|f| {
                        syn::Ident::new(f, proc_macro2::Span::call_site())
                    }));
                CrossCollectionGroup {
                    rc_ident: rc_ident.clone(),
                    a_param_count,
                    b_param_count,
                    block_ident: block_ident.clone(),
                    constraint_index_field: ci_field,
                    a_idx_stmts: a_idx_stmts.clone(),
                    b_idx_stmts: b_idx_stmts.clone(),
                    resolve_stmts: resolve_stmts.clone(),
                    root_var_ident: root_var_ident.clone(),
                    cost_entries: Vec::new(),
                    gh_entries: Vec::new(),
                    jac_entries: Vec::new(),
                }
            });
            group.cost_entries.push(cost_entry);
            group.gh_entries.push(gh_entry);
            if let Some(je) = jac_entry { group.jac_entries.push(je); }
        } else {
            // Nested CrossBlock: add inner loops to the collection group
            let frines_ident = frines_ident.unwrap();
            let parent_ident = parent_ident.unwrap();
            let group_key = coll_ident_str.clone();

            let self_var = syn::Ident::new(&a_type.to_lowercase(), proc_macro2::Span::call_site());
            let marker = source_marker(sc);

            let nested_cost = quote! {
                {
                    #marker
                    let #parent_ident = __item;
                    for __frine in &__item.#frines_ident {
                        #(#resolve_stmts)*
                        let #root_var_ident = &*__self_ref;
                        #(#cost_stmts)*
                    }
                }
            };

            let nested_gh = quote! {
                {
                    #marker
                    let #parent_ident = unsafe { &*(__item as *const #a_type_ident) };
                    for __frine in __item.#frines_ident.iter_mut() {
                        #(#resolve_stmts)*
                        let #root_var_ident = &*__self_ref;
                        { #(#gh_stmts)* }
                    }
                }
            };

            let nested_jac = if !jac_stmts.is_empty() {
                let resolve_stmts_j = resolve_stmts.clone();
                let b_idx_stmts_j = b_idx_stmts.clone();
                let marker_j = marker.clone();
                Some(quote! {
                    {
                        #marker_j
                        let #parent_ident = __item;
                        for __frine in &__item.#frines_ident {
                            #(#resolve_stmts_j)*
                            let #root_var_ident = &*__self_ref;
                            let __jac_idx: std::vec::Vec<u32> = {
                                let mut __b_idx = [0u32; #b_param_count];
                                #(#b_idx_stmts_j)*
                                let mut __v = std::vec::Vec::with_capacity(#a_param_count + #b_param_count);
                                __v.extend_from_slice(&__jac_a_idx);
                                __v.extend_from_slice(&__b_idx);
                                __v
                            };
                            #(#jac_stmts)*
                            __jac_cid += 1;
                        }
                    }
                })
            } else { None };

            let group = collection_groups.entry(group_key).or_insert_with(|| CollectionGroup {
                coll_ident: coll_ident.clone(),
                self_var: self_var.clone(),
                a_type_ident: a_type_ident.clone(),
                self_block: None,
                cost_entries: Vec::new(),
                gh_entries: Vec::new(),
                jac_entries: Vec::new(),
                nested_cost_loops: Vec::new(),
                nested_gh_loops: Vec::new(),
                nested_jac_loops: Vec::new(),
            });
            group.nested_cost_loops.push(nested_cost);
            group.nested_gh_loops.push(nested_gh);
            if let Some(nj) = nested_jac { group.nested_jac_loops.push(nj); }

            {
                let ci_set_nested = crate::registry_lookup(&sc.struct_name)
                    .and_then(|l| l.constraint_index_field.as_ref().map(|f| {
                        let fi = syn::Ident::new(f, proc_macro2::Span::call_site());
                        quote! { __frine.#fi = __cid; }
                    }));
                set_block_indices_loops.push(quote! {
                    for __item in self.#coll_ident.iter_mut() {
                        let mut __a_idx = [0u32; #a_param_count];
                        #(#a_idx_stmts)*
                        for __frine in __item.#frines_ident.iter_mut() {
                            #(#resolve_stmts)*
                            let mut __b_idx = [0u32; #b_param_count];
                            #(#b_idx_stmts)*
                            __frine.#block_ident.set_indices(&__a_idx, &__b_idx);
                            #ci_set_nested
                            __cid += 1;
                        }
                    }
                });
            }
        }
    }

    // Emit merged loops for collection groups FIRST, then append existing
    // non-merged loops. This ensures SelfBlock entities get lower constraint
    // IDs than cross-block/triplet constraints.
    let mut merged_cost: Vec<TokenStream2> = Vec::new();
    let mut merged_gh: Vec<TokenStream2> = Vec::new();
    let mut merged_jac: Vec<TokenStream2> = Vec::new();
    let mut merged_sbi: Vec<TokenStream2> = Vec::new();
    for group in collection_groups.values() {
        let coll = &group.coll_ident;
        let self_var = &group.self_var;
        let a_type = &group.a_type_ident;
        let cost_entries = &group.cost_entries;
        let gh_entries = &group.gh_entries;
        let jac_entries = &group.jac_entries;
        let nested_cost = &group.nested_cost_loops;
        let nested_gh = &group.nested_gh_loops;
        let nested_jac = &group.nested_jac_loops;

        // Merged cost loop: SelfBlock entries + nested CrossBlock inner loops
        merged_cost.push(quote! {
            for __item in self.#coll.iter() {
                let #self_var = __item;
                let #root_var_ident = &*__self_ref;
                #(#cost_entries)*
                #(#nested_cost)*
            }
        });

        // Merged grad+hessian loop
        merged_gh.push(quote! {
            for __item in self.#coll.iter_mut() {
                let #self_var = unsafe { &*(__item as *const #a_type) };
                let #root_var_ident = &*__self_ref;
                #(#gh_entries)*
                #(#nested_gh)*
            }
        });

        // Merged Jacobian loop (only if Jacobian entries/nested exist)
        if !jac_entries.is_empty() || !nested_jac.is_empty() {
            let a_count = group.self_block.as_ref().map(|sb| sb.a_param_count).unwrap_or(0);
            let a_idx_stmts_j: Vec<_> = group.self_block.as_ref()
                .map(|sb| sb.a_idx_stmts.clone()).unwrap_or_default();
            merged_jac.push(quote! {
                for __item in self.#coll.iter() {
                    let #self_var = __item;
                    let #root_var_ident = &*__self_ref;
                    let __jac_idx: std::vec::Vec<u32> = {
                        let mut __a_idx = [0u32; #a_count];
                        #(#a_idx_stmts_j)*
                        __a_idx.to_vec()
                    };
                    let __jac_a_idx = __jac_idx.clone();
                    #(#jac_entries)*
                    #(#nested_jac)*
                    __jac_cid += 1;
                }
            });
        }

        // set_block_indices loop (only if there's a SelfBlock)
        if let Some(ref sb) = group.self_block {
            let a_count = sb.a_param_count;
            let a_idx = &sb.a_idx_stmts;
            let block = &sb.block_ident;
            let a_type_name = a_type.to_string();
            let ci_set = crate::registry_lookup(&a_type_name)
                .and_then(|l| l.constraint_index_field.as_ref().map(|f| {
                    let fi = syn::Ident::new(f, proc_macro2::Span::call_site());
                    quote! { __item.#fi = __cid; }
                }));
            merged_sbi.push(quote! {
                for __item in self.#coll.iter_mut() {
                    let mut __a_idx = [0u32; #a_count];
                    #(#a_idx)*
                    __item.#block.set_indices(&__a_idx);
                    #ci_set
                    __cid += 1;
                }
            });
        }
    }

    // Auto-wire SelfBlock indices for "passive" entities: structs that have
    // Param + SelfBlock<Self> but no self-constraint referencing that block,
    // yet participate as A or B in some cross-block (e.g. landmarks in a
    // BA-style problem where bearings are owned by a peer struct). Without
    // this loop the macro emits add_residual calls into a SelfBlock whose
    // parameter indices are still u32::MAX, so its contributions silently
    // get dropped by accumulate_hessian -- the Hessian diagonal stays zero
    // and Cholesky blows up.
    //
    // A self-constraint would have created a collection_group with self_block
    // populated; the loop above already emits set_indices for those. Here we
    // walk root's collections one more time and emit set_indices for any
    // collection whose inner type has Param + SelfBlock but no group entry.
    // No __cid bump -- this isn't a constraint, just index wiring.
    {
        let mut wired: std::collections::HashSet<String> = std::collections::HashSet::new();
        for (key, group) in &collection_groups {
            if group.self_block.is_some() { wired.insert(key.clone()); }
        }
        let root_fields_passive: syn::FieldsNamed = syn::parse2(quote! { { #root_fields } })?;
        for field in &root_fields_passive.named {
            let field_ident = match field.ident.as_ref() {
                Some(i) => i.clone(),
                None => continue,
            };
            let field_name = field_ident.to_string();
            if wired.contains(&field_name) { continue; }
            // Pull T from Vec<T> / Deque<T> / refs::Vec<T> / refs::Deque<T>.
            // Filter on the OUTER segment name so SelfBlock<Self> / CrossBlock<...>
            // / Param<...> / Option<...> at the root don't get misread as
            // collections of their first type argument.
            let inner_name: Option<String> = if let syn::Type::Path(tp) = &field.ty
                && let Some(seg) = tp.path.segments.last()
                && matches!(seg.ident.to_string().as_str(), "Vec" | "Deque")
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
            { type_ident_name(inner).ok() } else { None };
            let type_name = match inner_name { Some(s) => s, None => continue };
            if !reachable.contains(&type_name) { continue; }
            let layout = match registry_lookup(&type_name) { Some(l) => l, None => continue };
            if layout.param_fields.is_empty() { continue; }
            let hb_field = match layout.self_block_field.clone() {
                Some(s) => s, None => continue,
            };
            let hb_ident = syn::Ident::new(&hb_field, proc_macro2::Span::call_site());
            let mut a_idx_stmts: Vec<TokenStream2> = Vec::new();
            let mut offset = 0usize;
            for pf in &layout.param_fields {
                let pf_ident = syn::Ident::new(pf, proc_macro2::Span::call_site());
                let size = layout.fields.iter()
                    .find(|(n, _)| n == pf)
                    .map(|(_, sft)| match sft {
                        SymFieldType::Scalar => 1usize,
                        SymFieldType::Vec2 => 2,
                        SymFieldType::Vec3 => 3,
                        _ => 0,
                    }).unwrap_or(0);
                if size == 0 { continue; }
                let end = offset + size;
                a_idx_stmts.push(quote! {
                    __item.#pf_ident.write_indices(&mut __a_idx[#offset..#end]);
                });
                offset = end;
            }
            if offset == 0 { continue; }
            let a_count = offset;
            merged_sbi.push(quote! {
                for __item in self.#field_ident.iter_mut() {
                    let mut __a_idx = [0u32; #a_count];
                    #(#a_idx_stmts)*
                    __item.#hb_ident.set_indices(&__a_idx);
                }
            });
        }
    }

    // Emit single-instance entity groups (RootSelf + DirectField). No loops —
    // each block evaluates its constraint(s) exactly once, with __item bound to
    // the entity (or a reborrow of self for RootSelf). Still advances __cid /
    // __jac_cid by one per constraint, matching the count=1 case of the
    // collection path. Emitted after the collection groups so entity IDs
    // remain deterministic, and before cross/triplet loops so self-entities
    // continue to get lower IDs than cross constraints.
    for group in single_instance_groups.values() {
        let accessor_read = &group.accessor_read;
        let accessor_write = &group.accessor_write;
        let self_var = &group.self_var;
        let root_var = &group.root_var_ident;
        let a_type = &group.a_type_ident;
        let a_count = group.a_param_count;
        let a_idx_stmts = &group.a_idx_stmts;
        let block_ident = &group.block_ident;
        let cost_entries = &group.cost_entries;
        let gh_entries = &group.gh_entries;
        let jac_entries = &group.jac_entries;
        let ci_set = group.constraint_index_field.as_ref().map(|fi| {
            quote! { __item.#fi = __cid; }
        });

        merged_cost.push(quote! {
            {
                let __item = #accessor_read;
                let #self_var = __item;
                let #root_var = &*__self_ref;
                #(#cost_entries)*
            }
        });

        merged_gh.push(quote! {
            {
                let __item = #accessor_write;
                let #self_var = unsafe { &*(__item as *const #a_type) };
                let #root_var = &*__self_ref;
                #(#gh_entries)*
            }
        });

        if !jac_entries.is_empty() {
            merged_jac.push(quote! {
                {
                    let __item = #accessor_read;
                    let #self_var = __item;
                    let #root_var = &*__self_ref;
                    let __jac_idx: std::vec::Vec<u32> = {
                        let mut __a_idx = [0u32; #a_count];
                        #(#a_idx_stmts)*
                        __a_idx.to_vec()
                    };
                    let __jac_a_idx = __jac_idx.clone();
                    let _ = &__jac_a_idx;
                    #(#jac_entries)*
                    __jac_cid += 1;
                }
            });
        }

        merged_sbi.push(quote! {
            {
                let __item = #accessor_write;
                let mut __a_idx = [0u32; #a_count];
                #(#a_idx_stmts)*
                __item.#block_ident.set_indices(&__a_idx);
                #ci_set
                __cid += 1;
            }
        });
    }

    // Emit merged cross-constraint loops (one per collection, all attributes inside)
    for group in cross_groups.values() {
        let rc_ident = &group.rc_ident;
        let a_param_count = group.a_param_count;
        let b_param_count = group.b_param_count;
        let block_ident = &group.block_ident;
        let a_idx_stmts = &group.a_idx_stmts;
        let b_idx_stmts = &group.b_idx_stmts;
        let resolve_stmts = &group.resolve_stmts;
        let root_var = &group.root_var_ident;
        let cost_entries = &group.cost_entries;
        let gh_entries = &group.gh_entries;
        let jac_entries = &group.jac_entries;
        let ci_set = group.constraint_index_field.as_ref().map(|fi| {
            quote! { __frine.#fi = __cid; }
        });

        cost_loops.push(quote! {
            for __frine in self.#rc_ident.iter() {
                #(#resolve_stmts)*
                let #root_var = &*__self_ref;
                #(#cost_entries)*
            }
        });

        grad_hessian_loops.push(quote! {
            for __frine in self.#rc_ident.iter_mut() {
                #(#resolve_stmts)*
                let #root_var = &*__self_ref;
                #(#gh_entries)*
            }
        });

        if !jac_entries.is_empty() {
            jacobian_loops.push(quote! {
                for __frine in self.#rc_ident.iter() {
                    #(#resolve_stmts)*
                    let #root_var = &*__self_ref;
                    let __jac_idx: std::vec::Vec<u32> = {
                        let mut __a_idx = [0u32; #a_param_count];
                        #(#a_idx_stmts)*
                        let mut __b_idx = [0u32; #b_param_count];
                        #(#b_idx_stmts)*
                        let mut __v = std::vec::Vec::with_capacity(#a_param_count + #b_param_count);
                        __v.extend_from_slice(&__a_idx);
                        __v.extend_from_slice(&__b_idx);
                        __v
                    };
                    #(#jac_entries)*
                    __jac_cid += 1;
                }
            });
        }

        set_block_indices_loops.push(quote! {
            for __frine in self.#rc_ident.iter_mut() {
                #(#resolve_stmts)*
                let mut __a_idx = [0u32; #a_param_count];
                #(#a_idx_stmts)*
                let mut __b_idx = [0u32; #b_param_count];
                #(#b_idx_stmts)*
                __frine.#block_ident.set_indices(&__a_idx, &__b_idx);
                #ci_set
                __cid += 1;
            }
        });
    }

    // Emit merged TripletBlock loops (one per collection, with set_block_indices)
    for group in triplet_groups.values() {
        let rc_ident = &group.rc_ident;
        let tp = group.triplet_param_count;
        let block_ident = &group.block_ident;
        let triplet_idx_stmts = &group.triplet_idx_stmts;
        let resolve_stmts = &group.resolve_stmts;
        let root_var = &group.root_var_ident;
        let cost_entries = &group.cost_entries;
        let gh_entries = &group.gh_entries;
        let jac_entries = &group.jac_entries;
        let ci_set = group.constraint_index_field.as_ref().map(|fi| {
            quote! { __frine.#fi = __cid; }
        });

        cost_loops.push(quote! {
            for __frine in self.#rc_ident.iter() {
                #(#resolve_stmts)*
                let #root_var = &*__self_ref;
                #(#cost_entries)*
            }
        });

        let entity_offsets = &group.entity_offsets;
        let entity_offsets_len = entity_offsets.len();
        grad_hessian_loops.push(quote! {
            for __frine in self.#rc_ident.iter_mut() {
                #(#resolve_stmts)*
                let #root_var = &*__self_ref;
                let mut __all_idx = [0u32; #tp];
                #(#triplet_idx_stmts)*
                let __entity_offsets: [u32; #entity_offsets_len] = [#(#entity_offsets),*];
                #(#gh_entries)*
            }
        });

        if !jac_entries.is_empty() {
            jacobian_loops.push(quote! {
                for __frine in self.#rc_ident.iter() {
                    #(#resolve_stmts)*
                    let #root_var = &*__self_ref;
                    let __jac_idx: std::vec::Vec<u32> = {
                        let mut __all_idx = [0u32; #tp];
                        #(#triplet_idx_stmts)*
                        __all_idx.to_vec()
                    };
                    #(#jac_entries)*
                    __jac_cid += 1;
                }
            });
        }

        // Multi-cross: emit a set_indices call per declared CrossBlock
        // field plus per-entity SelfBlock set_indices so the entities'
        // hb.indices leave their u32::MAX sentinel (otherwise every
        // add_residual on them would silently skip). Each CrossBlock's
        // a/b slices are cut from __all_idx using the entity-span
        // (start, count) pairs recorded in routing. Single-TripletBlock
        // groups also need per-entity set_indices for the same reason.
        let entity_self_indices = &group.entity_self_indices;
        if !group.multi_cross_blocks.is_empty() {
            let mcb_calls: Vec<TokenStream2> = group.multi_cross_blocks.iter().map(|mcb| {
                let block = &mcb.block_ident;
                let a_start = mcb.a_start; let a_end = mcb.a_start + mcb.a_count;
                let b_start = mcb.b_start; let b_end = mcb.b_start + mcb.b_count;
                quote! {
                    __frine.#block.set_indices(
                        &__all_idx[#a_start..#a_end],
                        &__all_idx[#b_start..#b_end],
                    );
                }
            }).collect();
            set_block_indices_loops.push(quote! {
                for __frine in self.#rc_ident.iter_mut() {
                    #(#resolve_stmts)*
                    let mut __all_idx = [0u32; #tp];
                    #(#triplet_idx_stmts)*
                    #(#entity_self_indices)*
                    #(#mcb_calls)*
                    #ci_set
                    __cid += 1;
                }
            });
        } else {
            // TripletBlock: set per-entity SelfBlock indices (needed so
            // the per-entity add_residual writes don't silently skip),
            // plus __cid assignment.
            set_block_indices_loops.push(quote! {
                for __frine in self.#rc_ident.iter_mut() {
                    #(#resolve_stmts)*
                    let mut __all_idx = [0u32; #tp];
                    #(#triplet_idx_stmts)*
                    #(#entity_self_indices)*
                    #ci_set
                    __cid += 1;
                }
            });
            let _ = (block_ident, triplet_idx_stmts, resolve_stmts); // silence unused warnings
        }
    }

    // Prepend merged SelfBlock loops before cross/triplet loops
    // so entities get lower constraint IDs than cross-block constraints.
    let mut ordered_cost = merged_cost; ordered_cost.append(&mut cost_loops);
    let cost_loops = ordered_cost;
    let mut ordered_gh = merged_gh; ordered_gh.append(&mut grad_hessian_loops);
    let grad_hessian_loops = ordered_gh;
    let mut ordered_jac = merged_jac; ordered_jac.append(&mut jacobian_loops);
    let jacobian_loops = ordered_jac;
    let mut ordered_sbi = merged_sbi; ordered_sbi.append(&mut set_block_indices_loops);
    let set_block_indices_loops = ordered_sbi;

    // Generate methods on root -- precision-aware
    let prec_type: syn::Type = syn::parse_str(precision)
        .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
            format!("invalid precision type '{}': {}", precision, e)))?;
    let update_method = syn::Ident::new(
        &format!("update{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_method = syn::Ident::new(
        &format!("accumulate_hessian{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_band_method = syn::Ident::new(
        &format!("accumulate_hessian_band{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_sparse_method = syn::Ident::new(
        &format!("accumulate_hessian_sparse{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_sparse_direct_method = syn::Ident::new(
        &format!("accumulate_hessian_sparse_direct{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_sparse_indexed_method = syn::Ident::new(
        &format!("accumulate_hessian_sparse_indexed{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());

    // advance(): fold accepted-step euler angle deltas. Recurses through
    // the whole model tree via Model::advance_params, so EA params at any
    // location (collections, root-level fields, direct-composed structs,
    // nested sub-models) are re-centered.
    let advance_call = if precision == "f32" {
        quote! { arael::model::Model::advance_params32(self, params); }
    } else {
        quote! { arael::model::Model::advance_params64(self, params); }
    };

    // `extended_compute_call` now passes `grad` so the extended hook can
    // write gradient entries directly into the LM-provided slice.
    let (extended_update_call, extended_cost_call, extended_compute_call) = if precision == "f64" {
        (quote! { arael::model::ExtendedModel::extended_update64(self, params); },
         quote! { __cost += arael::model::ExtendedModel::extended_cost64(self, params); },
         quote! { arael::model::ExtendedModel::extended_compute64(self, params, grad); })
    } else {
        (quote! { arael::model::ExtendedModel::extended_update32(self, params); },
         quote! { __cost += arael::model::ExtendedModel::extended_cost32(self, params); },
         quote! { arael::model::ExtendedModel::extended_compute32(self, params, grad); })
    };

    let extended_jacobian_call = if custom {
        if precision == "f64" {
            quote! { arael::model::ExtendedModel::extended_jacobian64(self, params, &mut __jac_rows, &mut __jac_cid); }
        } else {
            quote! { arael::model::ExtendedModel::extended_jacobian32(self, params, &mut __jac_rows, &mut __jac_cid); }
        }
    } else {
        quote! {}
    };

    let mut tokens = quote! {
        #(#constraint_impls)*

        impl #root_name {
            pub fn serialize64(&mut self, data: &mut std::vec::Vec<f64>) {
                arael::model::Model::serialize_params64(self, data);
                self.__set_block_indices();
            }
            pub fn deserialize64(&mut self, data: &[f64]) {
                arael::model::Model::deserialize_params64(self, data);
                arael::model::ExtendedModel::extended_deserialize64(self);
            }
            pub fn serialize32(&mut self, data: &mut std::vec::Vec<f32>) {
                arael::model::Model::serialize_params32(self, data);
                self.__set_block_indices();
            }
            pub fn deserialize32(&mut self, data: &[f32]) {
                arael::model::Model::deserialize_params32(self, data);
                arael::model::ExtendedModel::extended_deserialize32(self);
            }

            fn __set_block_indices(&mut self) {
                let mut __cid: u32 = 0;
                let _ = &__cid; // suppress unused warning when no constraint_index fields
                let __self_ref = unsafe { &*(self as *const Self) };
                let _ = __self_ref; // consumed only when root params participate
                #root_self_block_prelude
                #(#set_block_indices_loops)*
            }

            /// Returns the cost (sum of squared residuals, excluding
            /// extended-model residuals) as a byproduct of the sweep.
            fn __compute_blocks(&mut self, params: &[#prec_type], grad: &mut [#prec_type]) -> #prec_type {
                arael::model::Model::#update_method(self, params);
                #extended_update_call
                let __self_ref = unsafe { &*(self as *const Self) };
                self.zero_blocks();
                let mut __cost = 0.0 as #prec_type;
                #(#grad_hessian_loops)*
                #extended_compute_call
                __cost
            }
        }
    };

    // Generate JacobianModel impl if requested
    if jacobian {
        let ext_update = if custom { extended_update_call.clone() } else { quote! {} };
        tokens.extend(quote! {
            impl arael::model::JacobianModel<#prec_type> for #root_name {
                fn calc_jacobian(&mut self, params: &[#prec_type]) -> arael::model::Jacobian<#prec_type> {
                    arael::model::Model::#update_method(self, params);
                    #ext_update
                    let __self_ref = unsafe { &*(self as *const Self) };
                    let mut __jac_rows: std::vec::Vec<arael::model::JacobianRow<#prec_type>> = std::vec::Vec::new();
                    let mut __jac_cid: u32 = 0;
                    #(#jacobian_loops)*
                    #extended_jacobian_call
                    let mut __jac = arael::model::Jacobian { num_params: params.len(), rows: __jac_rows };
                    // Shared-parameter slots (aliased CrossBlock refs) emit one
                    // entry per slot; merge so rows carry unique indices.
                    __jac.merge_duplicate_entries();
                    __jac
                }
            }
        });
    }

    // Summary doc-comment listing every constraint's source location. Shows
    // up above the generated `impl LmProblem` block in `cargo expand`, and
    // appears in rustdoc output for the root type too.
    let summary_docs: Vec<TokenStream2> = {
        let mut lines: Vec<TokenStream2> = Vec::new();
        lines.push({
            let s = syn::LitStr::new(
                "Auto-generated by `#[arael::model]` / `#[arael(root)]`. Constraint sources:",
                proc_macro2::Span::call_site());
            quote! { #[doc = #s] }
        });
        for sc in &stashed {
            if !reachable.contains(&sc.struct_name) { continue; }
            let line = format!(
                "- `{}[{}]` @ {}:{}",
                sc.struct_name, sc.label_hint, sc.attr_file, sc.attr_line
            );
            let lit = syn::LitStr::new(&line, proc_macro2::Span::call_site());
            lines.push(quote! { #[doc = #lit] });
        }
        lines
    };

    tokens.extend(quote! {

        #(#summary_docs)*
        impl arael::simple_lm::LmProblem<#prec_type> for #root_name {
            fn calc_cost(&mut self, params: &[#prec_type]) -> #prec_type {
                arael::model::Model::#update_method(self, params);
                #extended_update_call
                let __self_ref = unsafe { &*(self as *const Self) };
                let mut __cost = 0.0 as #prec_type;
                #(#cost_loops)*
                #extended_cost_call
                __cost
            }

            fn calc_grad_hessian_dense(&mut self, params: &[#prec_type], grad: &mut [#prec_type], hessian: &mut [#prec_type]) -> #prec_type {
                grad.iter_mut().for_each(|g| *g = 0.0);
                let mut __cost = self.__compute_blocks(params, grad);
                #extended_cost_call
                hessian.iter_mut().for_each(|h| *h = 0.0);
                self.#accumulate_method(hessian);
                __cost
            }

            fn calc_grad_hessian_band(&mut self, params: &[#prec_type], grad: &mut [#prec_type], band: &mut [#prec_type], kd: usize) -> Result<#prec_type, arael::simple_lm::BandError> {
                grad.iter_mut().for_each(|g| *g = 0.0);
                let mut __cost = self.__compute_blocks(params, grad);
                #extended_cost_call
                band.iter_mut().for_each(|b| *b = 0.0);
                self.#accumulate_band_method(band, kd)?;
                Ok(__cost)
            }

            fn calc_grad_hessian_sparse(&mut self, params: &[#prec_type], grad: &mut [#prec_type], coo: &mut arael::simple_lm::CooMatrix<#prec_type>) -> #prec_type {
                grad.iter_mut().for_each(|g| *g = 0.0);
                let mut __cost = self.__compute_blocks(params, grad);
                #extended_cost_call
                coo.clear();
                self.#accumulate_sparse_method(coo);
                __cost
            }

            fn calc_grad_hessian_sparse_direct(&mut self, params: &[#prec_type], grad: &mut [#prec_type], csc: &mut arael::simple_lm::CscMatrix<#prec_type>) -> #prec_type {
                grad.iter_mut().for_each(|g| *g = 0.0);
                let mut __cost = self.__compute_blocks(params, grad);
                #extended_cost_call
                csc.vals.iter_mut().for_each(|v| *v = 0.0 as #prec_type);
                self.#accumulate_sparse_direct_method(csc);
                __cost
            }

            fn calc_grad_hessian_sparse_indexed(&mut self, params: &[#prec_type], grad: &mut [#prec_type], vals: &mut [#prec_type], positions: &[usize]) -> #prec_type {
                grad.iter_mut().for_each(|g| *g = 0.0);
                let mut __cost = self.__compute_blocks(params, grad);
                #extended_cost_call
                vals.iter_mut().for_each(|v| *v = 0.0 as #prec_type);
                let mut cursor = 0usize;
                self.#accumulate_sparse_indexed_method(vals, positions, &mut cursor);
                __cost
            }

            fn advance(&mut self, params: &mut [#prec_type]) {
                #advance_call
            }
        }
    });

    // Generate default ExtendedModel impl unless `extended` flag is set
    if !custom {
        tokens.extend(quote! {
            impl arael::model::ExtendedModel for #root_name {}
        });
    }

    Ok(tokens)
}

/// Interpret constraint body and return (residual expressions, param symbols).
fn interpret_constraint_body(
    struct_name: &syn::Ident,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    constraint: &ConstraintAttr,
    root_type_name: &str,
) -> syn::Result<(Vec<E>, Vec<String>)> {
    let is_remote = constraint.primary_block_field().contains('.');
    let (a_type, b_type) = if is_remote {
        // Remote block: e.g. "pose.hb_pose" — target type from Ref field
        let parts: Vec<&str> = constraint.primary_block_field().split('.').collect();
        let ref_field_name = parts[0];
        let ref_field = fields.iter().find(|f|
            f.ident.as_ref().map(|i| i.to_string()) == Some(ref_field_name.to_string())
        ).ok_or_else(|| syn::Error::new_spanned(struct_name,
            format!("remote block ref field '{}' not found", ref_field_name)))?;
        let (_, inner) = extract_wrapper_inner(&ref_field.ty, "Ref")
            .ok_or_else(|| syn::Error::new_spanned(struct_name,
                format!("field '{}' is not Ref<T>", ref_field_name)))?;
        // Find the parent struct (who contains this constraint struct)
        let parent_type = {
            let guard = crate::SYM_REGISTRY.lock().unwrap();
            guard.as_ref().and_then(|reg| {
                reg.layouts.iter().find(|(_, layout)| {
                    layout.fields.iter().any(|(_, sft)| {
                        if let SymFieldType::Struct(s) = sft { *s == struct_name.to_string() } else { false }
                    })
                }).map(|(name, _)| name.clone())
            })
        }.unwrap_or_else(|| inner.to_string());
        (parent_type, Some(inner.to_string()))
    } else {
        let block_field = fields.iter().find(|f| {
            f.ident.as_ref().map(|i| i.to_string()) == Some(constraint.primary_block_field().to_string())
        }).ok_or_else(|| {
            syn::Error::new_spanned(struct_name, format!("block field '{}' not found", constraint.primary_block_field()))
        })?;
        extract_block_type_args(&block_field.ty)?
    };
    let struct_layout = registry_lookup(&struct_name.to_string());
    let ref_paths = struct_layout.as_ref().map(|l| &l.ref_paths[..]).unwrap_or(&[]);
    let parent_name = constraint.parent_name.clone().unwrap_or_else(|| a_type.to_lowercase());

    // Multi-cross vs single-cross discriminator — needed before building
    // var_infos so we can skip the parent_name entry in multi-cross
    // (where the primary A is already covered by a Ref field and adding a
    // parent_name alias would pollute param_symbols with duplicate params
    // under a second var name).
    let is_multi_cross_early = constraint.block_fields.len() > 1;

    // Build var_infos
    let mut var_infos: Vec<(String, String)> = Vec::new();
    if !constraint.vars.is_empty() {
        for var in &constraint.vars {
            if let Some(ref tn) = var.type_name { var_infos.push((var.name.clone(), tn.clone())); }
        }
    } else {
        for (field_name, _) in ref_paths {
            if let Some(field) = fields.iter().find(|f| f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone()))
                && let Some((_, inner_ident)) = extract_wrapper_inner(&field.ty, "Ref") {
                    var_infos.push((field_name.clone(), inner_ident.to_string()));
                }
        }
        // Skip the parent_name alias only for pure multi-cross (not
        // remote-block, no root-triplet). In the pure case, parent_name
        // = a_type.lower() is always a duplicate of a ref field on the
        // struct. For remote-block + multi-cross, parent_name names the
        // parent *type* (e.g. `lm` -> PointLandmark) which is distinct
        // from the refs. For self-primary + `root.<triplet>`, there is
        // no Ref<Self> field and parent_name is the only path that
        // resolves `<self_lc>.*` in the body.
        let has_root_triplet_block = constraint.block_fields.iter()
            .any(|bf| bf.starts_with("root."));
        let is_pure_multi_cross =
            is_multi_cross_early && !is_remote && !has_root_triplet_block;
        if a_type != "__triplet__" && !is_pure_multi_cross {
            var_infos.push((parent_name.clone(), a_type.clone()));
        }
        let root_var = root_type_name.to_lowercase();
        var_infos.push((root_var, root_type_name.to_string()));
    }

    // Setup context — recursively register all fields including nested structs
    let mut ctx = ConstraintCtx::new();
    for (var_name, type_name) in &var_infos {
        register_bindings_recursive(&mut ctx, var_name, var_name, type_name);
    }

    // Register the constraint struct's own non-Ref fields
    // For CrossBlock: accessible via lowercase struct name, code uses __frine
    // For SelfBlock: the struct IS the variable (already registered above via var_infos)
    if b_type.is_some() || a_type == "__triplet__" {
        // Use a simple name derived from the struct name
        // Derive self-reference name from struct: PosePair -> "posepair"
        let self_var = struct_name.to_string().to_lowercase();
        register_bindings_recursive(&mut ctx, &self_var, "__frine", &struct_name.to_string());
    }

    // Collect param symbols
    let mut param_symbols: Vec<String> = Vec::new();
    let is_triplet = a_type == "__triplet__";
    // Multi-cross: multiple block fields. Treat like triplet for
    // param-symbol collection (gather params from ALL ref fields, not
    // just the primary block's A/B). Routing in the emission path
    // ensures every cross pair is covered by a declared CrossBlock.
    // Multi-cross may coexist with a remote primary block (e.g. the
    // first block is `pose.hb_pose`, the rest are local CrossBlocks).
    let is_multi_cross = constraint.block_fields.len() > 1;

    // Root-as-entity: if any declared local CrossBlock references the
    // root type, OR any block is `root.<triplet>`, include root's
    // Params in the symbol set too.
    let has_root_entity = is_multi_cross && constraint.block_fields.iter().any(|bf| {
        if bf.starts_with("root.") { return true; }
        if bf.contains('.') { return false; }
        let Some(field) = fields.iter().find(|f|
            f.ident.as_ref().map(|i| i.to_string()) == Some(bf.clone())) else { return false; };
        let Ok((a, b_opt)) = extract_block_type_args(&field.ty) else { return false; };
        a == root_type_name || b_opt.as_deref() == Some(root_type_name)
    });

    if is_triplet || is_multi_cross {
        // TripletBlock / multi-cross: collect params from ALL ref fields
        // (no A/B distinction). Root's Params are included when a
        // declared CrossBlock<X, Root> opts the root into the constraint
        // (has_root_entity) — the root is bound as `<root_lc>` in the
        // emission scope.
        let mut used_vars = std::collections::HashSet::new();
        for (var_name, type_name) in &var_infos {
            if type_name == root_type_name && !has_root_entity { continue; } // skip root unless opted in
            if !used_vars.insert(var_name.clone()) { continue; }
            if let Some(layout) = registry_lookup(type_name) {
                for pf in &layout.param_fields {
                    let sym_base = if layout.universal_euler_angle_fields.contains(pf) {
                        format!("{}.{}.delta", var_name, pf)
                    } else {
                        format!("{}.{}.work()", var_name, pf)
                    };
                    add_param_symbols(&sym_base,
                        layout.fields.iter().find(|(n, _)| n == pf).map(|(_, t)| t).unwrap(),
                        &mut param_symbols);
                }
            }
        }
    } else {
        // SelfBlock/CrossBlock: collect params from A and optionally B
        let a_var_name = {
            var_infos.iter().find(|(_, tn)| *tn == a_type)
                .map(|(vn, _)| vn.clone()).unwrap_or(parent_name.clone())
        };

        if let Some(a_layout) = registry_lookup(&a_type) {
            for pf in &a_layout.param_fields {
                let sym_base = if a_layout.universal_euler_angle_fields.contains(pf) {
                    format!("{}.{}.delta", a_var_name, pf)
                } else {
                    format!("{}.{}.work()", a_var_name, pf)
                };
                add_param_symbols(&sym_base,
                    a_layout.fields.iter().find(|(n, _)| n == pf).map(|(_, t)| t).unwrap(),
                    &mut param_symbols);
            }
        }
        if let Some(ref b_type_name) = b_type
            && let Some(b_layout) = registry_lookup(b_type_name) {
                let b_var = var_infos.iter().find(|(vn, tn)| {
                    tn == b_type_name && *vn != a_var_name
                }).or_else(|| var_infos.iter().find(|(_, tn)| tn == b_type_name))
                    .map(|(vn, _)| vn.clone()).unwrap_or_else(|| b_type_name.to_lowercase());
                for pf in &b_layout.param_fields {
                    let sym_base = if b_layout.universal_euler_angle_fields.contains(pf) {
                        format!("{}.{}.delta", b_var, pf)
                    } else {
                        format!("{}.{}.work()", b_var, pf)
                    };
                    add_param_symbols(&sym_base,
                        b_layout.fields.iter().find(|(n, _)| n == pf).map(|(_, t)| t).unwrap(),
                        &mut param_symbols);
                }
            }
    }

    // Interpret body
    let mut residuals: Vec<E> = Vec::new();
    for stmt in &constraint.body_stmts {
        match stmt {
            Stmt::Local(local) => {
                let name = match &local.pat {
                    Pat::Ident(pi) => pi.ident.to_string(),
                    _ => return Err(syn::Error::new_spanned(&local.pat, "simple let binding required")),
                };
                let init = local.init.as_ref().ok_or_else(|| syn::Error::new_spanned(local, "initializer required"))?;
                let val = eval_expr(&init.expr, &mut ctx)?;
                ctx.bindings.insert(name, val);
            }
            Stmt::Expr(expr, _) => {
                if let Expr::Array(arr) = expr {
                    for elem in &arr.elems {
                        match eval_expr(elem, &mut ctx)? {
                            SymVal::Scalar(e) => residuals.push(e),
                            _ => return Err(syn::Error::new_spanned(elem, "residual must be scalar")),
                        }
                    }
                } else {
                    match eval_expr(expr, &mut ctx)? {
                        SymVal::Scalar(e) => residuals.push(e),
                        _ => return Err(syn::Error::new_spanned(expr, "residual must be scalar")),
                    }
                }
            }
            _ => {}
        }
    }

    if !params_from_registry_check(&param_symbols) {
        let mut all_vars = std::collections::BTreeSet::new();
        for r in &residuals { all_vars.extend(r.free_vars()); }
        for var in &all_vars {
            if var.contains(".work()") { param_symbols.push(var.clone()); }
        }
    }

    Ok((residuals, param_symbols))
}

/// Zero-cost source-origin marker: emits a doc-attribute on a nested dummy
/// const so `cargo expand` renders a `///` line above each constraint block.
/// `#[doc = "..."]` is the token form of a `///` comment, and pretty-printers
/// typically render it back into the `///` syntax. Attaching it to
/// `const _: () = ();` keeps the attribute legal inside function bodies (bare
/// `#[doc]` on a statement isn't allowed on stable).
fn source_marker(sc: &crate::StashedConstraint) -> TokenStream2 {
    let text = format!(
        " arael: {}[{}] @ {}:{}",
        sc.struct_name, sc.label_hint, sc.attr_file, sc.attr_line
    );
    let lit = syn::LitStr::new(&text, proc_macro2::Span::call_site());
    quote! {
        #[doc = #lit]
        const _: () = ();
    }
}

/// `.work()` coverage check: every `.work()` symbol reached by the residuals
/// must be in `param_symbols`. Otherwise the generated `add_residual` call
/// emits a gradient vector that silently drops the derivative w.r.t. the
/// missing parameter — the optimizer compiles but fails to move it.
///
/// Common trigger: referencing a root-struct Param from a sub-entity's
/// constraint (root-level params need their own block machinery; not yet
/// supported). Error message leads with `file:line:` so most terminals /
/// IDEs auto-link the diagnostic text to the offending constraint
/// attribute, even though the syn::Error span (at the root's
/// `#[arael::model]`) can't be fixed — spans don't cross proc-macro
/// invocations.
pub(crate) fn check_residual_coverage(
    sc: &crate::StashedConstraint,
    struct_name: &syn::Ident,
    residuals: &[E],
    param_symbols: &[String],
) -> syn::Result<()> {
    let param_set: std::collections::HashSet<&str> =
        param_symbols.iter().map(|s| s.as_str()).collect();
    let mut missing: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    for r in residuals {
        for v in r.free_vars() {
            if v.contains(".work()") && !param_set.contains(v.as_str()) {
                missing.insert(v);
            }
        }
    }
    if missing.is_empty() { return Ok(()); }
    let list: Vec<String> = missing.into_iter().collect();

    // Tailor the hint to the shape of the mismatch. Three cases:
    //  (1) Every missing param's top-level binding is a `Ref<T>` field on the
    //      constraint struct itself — so the user HAS enough refs declared,
    //      just too many of them for a CrossBlock. Switching the block to
    //      `TripletBlock<T>` covers all ref-referenced entities at once.
    //  (2) Some missing param references something not in the struct's refs
    //      (typically a root-level ident like `path.foo`) — root-level params
    //      don't have a block slot yet.
    //  (3) Mix of the two.
    let ref_field_names: std::collections::HashSet<String> = registry_lookup(&struct_name.to_string())
        .map(|l| l.ref_paths.iter().map(|(n, _)| n.clone()).collect())
        .unwrap_or_default();
    let head_of = |s: &str| -> String {
        s.split_once('.').map(|(h, _)| h.to_string()).unwrap_or_else(|| s.to_string())
    };
    let all_via_struct_refs = !ref_field_names.is_empty()
        && list.iter().all(|m| ref_field_names.contains(&head_of(m)));
    let hint = if all_via_struct_refs {
        " — all missing params resolve through this struct's own `Ref<T>` fields, but there are more than CrossBlock<A, B> can cover. Switch the block to `TripletBlock<T>` to include every ref-referenced entity.".to_string()
    } else {
        " (SelfBlock<A> covers A; CrossBlock<A, B> covers A and B; TripletBlock<T> covers every Ref<T> field on the constraint struct; root-level params have no block slot yet)".to_string()
    };

    let msg = format!(
        "{}:{}: constraint on `struct {}` references param(s) outside its hessian block: [{}]{}",
        sc.attr_file, sc.attr_line, struct_name, list.join(", "), hint
    );
    Err(syn::Error::new_spanned(struct_name, msg))
}

fn params_from_registry_check(params: &[String]) -> bool {
    !params.is_empty()
}

fn find_root_collection(
    root_fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    type_name: &str,
) -> Option<(String, String)> {
    // Find a field in root that is a collection (Vec/Deque) of the given type
    for field in root_fields {
        let field_name = field.ident.as_ref()?.to_string();
        if let syn::Type::Path(tp) = &field.ty
            && let Some(seg) = tp.path.segments.last() {
                let container = seg.ident.to_string();
                if (container == "Vec" || container == "Deque" || container == "Arena")
                    && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                        && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
                            && let Ok(inner_name) = type_ident_name(inner)
                                && inner_name == type_name {
                                    return Some((field_name, container));
                                }
            }
    }
    None
}

/// Where on the root struct an entity of a given type lives.
#[derive(Clone)]
enum EntityLocation {
    /// Vec<T> / Deque<T> / Arena<T> field on root. Multi-instance, iterated.
    Collection { field: String },
    /// Plain struct-typed field on root (e.g. `sub: Sub`). Single instance.
    DirectField { field: String },
    /// The constraint's entity type is the root struct itself. Single instance, accessor is `self`.
    RootSelf,
}

fn resolve_entity_location(
    root_fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    root_name: &str,
    type_name: &str,
) -> Option<EntityLocation> {
    if type_name == root_name {
        return Some(EntityLocation::RootSelf);
    }
    if let Some((field, _container)) = find_root_collection(root_fields, type_name) {
        return Some(EntityLocation::Collection { field });
    }
    // Look for a plain struct-typed field whose type name matches.
    for field in root_fields {
        let field_name = field.ident.as_ref()?.to_string();
        if let syn::Type::Path(tp) = &field.ty
            && let Some(seg) = tp.path.segments.last()
                && matches!(seg.arguments, syn::PathArguments::None)
                && seg.ident == type_name {
                    return Some(EntityLocation::DirectField { field: field_name });
                }
    }
    None
}

#[allow(dead_code)]
fn find_var_for_type_annotated(vars: &[ConstraintVar], type_name: &str) -> syn::Result<String> {
    // First check explicit type annotations
    for v in vars {
        if let Some(ref tn) = v.type_name
            && tn == type_name {
                return Ok(v.name.clone());
            }
    }
    // Fall back to name matching
    let var_names: Vec<String> = vars.iter().map(|v| v.name.clone()).collect();
    find_var_for_type(&var_names, type_name)
}

#[allow(dead_code)]
fn find_var_for_type(var_names: &[String], type_name: &str) -> syn::Result<String> {
    // Exact lowercase match: Pose -> pose
    let lower = type_name.to_lowercase();
    for v in var_names {
        if v.to_lowercase() == lower {
            return Ok(v.clone());
        }
    }
    // Check if the type is in the registry and matches a variable name
    for v in var_names {
        if let Some(layout) = find_layout_for_var(v) {
            // This variable resolved to a layout — check if that layout's origin
            // name matches the type we're looking for. This is fragile but works
            // for cases like variable "lm" matching to registered type "PointLandmark"
            // if the user happened to register under that name.
            let _ = layout; // just checking existence
        }
    }
    // Fallback: try matching variable name to type name by checking if the registry
    // entry for the type was stored. Then find a variable that has a matching layout.
    if registry_lookup(type_name).is_some() {
        // The type is registered. Find which variable has a layout that matches.
        for v in var_names {
            if let Some(var_layout) = find_layout_for_var(v)
                && let Some(type_layout) = registry_lookup(type_name)
                    && var_layout.param_fields == type_layout.param_fields {
                        return Ok(v.clone());
                    }
        }
    }
    // Last resort: if there's only one unmatched variable, use it
    // For now, just return an error with helpful message
    Err(syn::Error::new(proc_macro2::Span::call_site(),
        format!("cannot determine which constraint variable corresponds to type '{}'. \
                 Name the variable '{}' to match automatically.", type_name, lower)))
}

#[allow(dead_code)]
fn find_layout_for_var(var_name: &str) -> Option<crate::SymLayout> {
    let guard = crate::SYM_REGISTRY.lock().unwrap();
    let reg = guard.as_ref()?;

    // Direct match
    if let Some(l) = reg.layouts.get(var_name) { return Some(l.clone()); }

    // Capitalize first letter
    let capitalized = {
        let mut c = var_name.chars();
        match c.next() {
            None => String::new(),
            Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        }
    };
    if let Some(l) = reg.layouts.get(&capitalized) { return Some(l.clone()); }

    // Case-insensitive match
    let lower = var_name.to_lowercase();
    for (k, v) in &reg.layouts {
        if k.to_lowercase() == lower {
            return Some(v.clone());
        }
    }

    None
}

/// Generate code for all stashed constraints (called when #[arael(root)] is processed).
#[allow(dead_code)]
pub fn generate_all_stashed_constraints() -> syn::Result<TokenStream2> {
    let stashed = crate::registry_constraints();
    let mut all_impls = Vec::new();

    for sc in &stashed {
        // Re-parse the constraint attribute tokens
        let attr_ts: proc_macro2::TokenStream = sc.attr_tokens.parse()
            .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                format!("failed to re-parse constraint tokens for {}: {}", sc.struct_name, e)))?;
        let attr_tokens: Vec<proc_macro2::TokenTree> = attr_ts.into_iter().collect();

        // Parse as constraint attribute
        let err_ident = syn::Ident::new(&sc.struct_name, proc_macro2::Span::call_site());
        let constraint = match &attr_tokens[0] {
            proc_macro2::TokenTree::Ident(id) if *id == "constraint" => {
                if let Some(proc_macro2::TokenTree::Group(g)) = attr_tokens.get(1) {
                    let inner: Vec<proc_macro2::TokenTree> = g.stream().into_iter().collect();
                    parse_constraint_inner_impl(&inner, &err_ident)?
                } else {
                    None
                }
            }
            _ => None,
        };

        let constraint = match constraint {
            Some(c) => c,
            None => continue,
        };

        // Re-parse the fields
        let fields_ts: proc_macro2::TokenStream = sc.fields_tokens.parse()
            .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                format!("failed to re-parse fields for {}: {}", sc.struct_name, e)))?;
        let fields: syn::FieldsNamed = syn::parse2(quote::quote! { { #fields_ts } })?;

        let struct_ident = syn::Ident::new(&sc.struct_name, proc_macro2::Span::call_site());
        let impl_code = generate_constraint_impl(&struct_ident, &fields.named, &constraint)?;
        all_impls.push(impl_code);
    }

    Ok(quote::quote! { #(#all_impls)* })
}

