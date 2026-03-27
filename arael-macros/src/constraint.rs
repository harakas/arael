//! Constraint attribute: typed symbolic expression interpreter and code generator.
//!
//! Interprets constraint body expressions at compile time using arael-sym types,
//! differentiates symbolically, and generates compiled evaluate code.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{Expr, Stmt, Pat};
use std::collections::HashMap;

use arael_sym::{self, E, vect2sym, vect3sym, matrix3sym};

use crate::{registry_lookup, SymFieldType, extract_wrapper_inner};

// ---------------------------------------------------------------------------
// Typed symbolic value
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub enum SymVal {
    Scalar(E),
    Vec2(vect2sym),
    Vec3(vect3sym),
    Mat3(matrix3sym),
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
            SymVal::Mat3(_) => "mat3",
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
            SymFieldType::Mat3 => SymVal::Mat3(matrix3sym::new(base)),
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
                if let Some(val) = ctx.bindings.get(&name) {
                    return Ok(val.clone());
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
            if let Some(ref path) = dotted {
                if let Some(val) = ctx.bindings.get(path) {
                    return Ok(val.clone());
                }
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
                _ => Err(syn::Error::new_spanned(expr, "unsupported operator in constraint")),
            }
        }

        // Unary negation
        Expr::Unary(eu) => {
            let inner = eval_expr(&eu.expr, ctx)?;
            match eu.op {
                syn::UnOp::Neg(_) => match inner {
                    SymVal::Scalar(e) => Ok(SymVal::Scalar(-e)),
                    SymVal::Vec3(v) => Ok(SymVal::Vec3(-v)),
                    _ => Err(syn::Error::new_spanned(expr, "cannot negate this type")),
                },
                _ => Err(syn::Error::new_spanned(expr, "unsupported unary operator")),
            }
        }

        // Function calls: atan2, atan, sin, cos, etc.
        Expr::Call(ec) => {
            if let Expr::Path(func_path) = ec.func.as_ref() {
                if let Some(func_name) = func_path.path.get_ident() {
                    let args: Vec<SymVal> = ec.args.iter()
                        .map(|a| eval_expr(a, ctx))
                        .collect::<Result<_, _>>()?;
                    return eval_function(&func_name.to_string(), args, expr);
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

        // Index access: expr[i]
        Expr::Index(idx) => {
            let base = eval_expr(&idx.expr, ctx)?;
            let index_expr = &idx.index;
            match &base {
                SymVal::Mat3(m) => {
                    if let Expr::Lit(lit) = index_expr.as_ref() {
                        if let syn::Lit::Int(li) = &lit.lit {
                            let i: usize = li.base10_parse()?;
                            return Ok(SymVal::Vec3(m.rows[i].clone()));
                        }
                    }
                    Err(syn::Error::new_spanned(index_expr, "matrix index must be a literal integer"))
                }
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

fn eval_function(name: &str, args: Vec<SymVal>, span: &Expr) -> Result<SymVal, syn::Error> {
    match name {
        "atan2" => {
            if args.len() != 2 { return Err(syn::Error::new_spanned(span, "atan2 expects 2 args")); }
            match (&args[0], &args[1]) {
                (SymVal::Scalar(y), SymVal::Scalar(x)) =>
                    Ok(SymVal::Scalar(arael_sym::atan2(y.clone(), x.clone()))),
                _ => Err(syn::Error::new_spanned(span, "atan2 expects scalar arguments")),
            }
        }
        "atan" => {
            if args.len() != 1 { return Err(syn::Error::new_spanned(span, "atan expects 1 arg")); }
            match &args[0] {
                SymVal::Scalar(e) => Ok(SymVal::Scalar(arael_sym::atan(e.clone()))),
                _ => Err(syn::Error::new_spanned(span, "atan expects a scalar argument")),
            }
        }
        "sin" | "cos" | "tan" | "exp" | "ln" | "sqrt" | "abs" => {
            if args.len() != 1 { return Err(syn::Error::new_spanned(span, format!("{} expects 1 arg", name))); }
            match &args[0] {
                SymVal::Scalar(e) => {
                    let f = match name {
                        "sin" => arael_sym::sin,
                        "cos" => arael_sym::cos,
                        "tan" => arael_sym::tan,
                        "exp" => arael_sym::exp,
                        "ln" => arael_sym::ln,
                        "sqrt" => arael_sym::sqrt,
                        "abs" => arael_sym::abs,
                        _ => unreachable!(),
                    };
                    Ok(SymVal::Scalar(f(e.clone())))
                }
                _ => Err(syn::Error::new_spanned(span, format!("{} expects a scalar argument", name))),
            }
        }
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
    match (left, right) {
        (SymVal::Scalar(a), SymVal::Scalar(b)) => Ok(SymVal::Scalar(a * b)),
        (SymVal::Scalar(a), SymVal::Vec2(b)) => Ok(SymVal::Vec2(arael_sym::geo::vect2sym { x: a.clone() * b.x, y: a * b.y })),
        (SymVal::Vec2(a), SymVal::Scalar(b)) => Ok(SymVal::Vec2(arael_sym::geo::vect2sym { x: a.x * b.clone(), y: a.y * b })),
        (SymVal::Scalar(a), SymVal::Vec3(b)) => Ok(SymVal::Vec3(a * b)),
        (SymVal::Vec3(a), SymVal::Scalar(b)) => Ok(SymVal::Vec3(a * b)),
        (SymVal::Vec2(a), SymVal::Vec2(b)) => Ok(SymVal::Scalar(a * b)), // dot product
        (SymVal::Vec3(a), SymVal::Vec3(b)) => Ok(SymVal::Scalar(a * b)), // dot product
        (SymVal::Mat3(a), SymVal::Vec3(b)) => Ok(SymVal::Vec3(a * b)),
        (SymVal::Mat3(a), SymVal::Mat3(b)) => Ok(SymVal::Mat3(a * b)),
        _ => Err(syn::Error::new_spanned(span, "type mismatch in multiplication")),
    }
}

fn sym_div(left: SymVal, right: SymVal, span: &Expr) -> Result<SymVal, syn::Error> {
    match (left, right) {
        (SymVal::Scalar(a), SymVal::Scalar(b)) => Ok(SymVal::Scalar(a / b)),
        (SymVal::Vec2(a), SymVal::Scalar(b)) => Ok(SymVal::Vec2(a / b)),
        _ => Err(syn::Error::new_spanned(span, "unsupported division types")),
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
    pub block_field: String,
    pub parent_name: Option<String>,  // e.g. "lm" for parent=lm
    pub guard: Option<String>,        // runtime guard expression, e.g. "self.info.gps.is_some()"
    pub vars: Vec<ConstraintVar>,     // explicit variables (legacy, may be empty)
    pub body_stmts: Vec<Stmt>,
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
            if ident.to_string() != "constraint" { continue; }

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
    let mut pos = 0;
    let mut block_field: Option<String> = None;
    let mut parent_name: Option<String> = None;
    let mut guard: Option<String> = None;
    let mut vars: Vec<ConstraintVar> = Vec::new();

    loop {
        match tokens.get(pos) {
            Some(proc_macro2::TokenTree::Ident(id)) => {
                let name = id.to_string();
                pos += 1;
                // Check for = (parent=lm) or : (var: Type)
                match tokens.get(pos) {
                    Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '=' => {
                        pos += 1;
                        if name == "parent" {
                            if let Some(proc_macro2::TokenTree::Ident(val)) = tokens.get(pos) {
                                parent_name = Some(val.to_string());
                                pos += 1;
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
                        if block_field.is_none() {
                            block_field = Some(name);
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
                        if block_field.is_none() {
                            block_field = Some(full_name);
                        } else {
                            vars.push(ConstraintVar { name: full_name, type_name: None });
                        }
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

    let block_field = block_field.ok_or_else(|| {
        syn::Error::new_spanned(err_span, "constraint needs at least the block field name")
    })?;

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
        block_field,
        parent_name,
        guard,
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
        f.ident.as_ref().map(|i| i.to_string()) == Some(constraint.block_field.clone())
    }).ok_or_else(|| {
        syn::Error::new_spanned(struct_name,
            format!("constraint block field '{}' not found", constraint.block_field))
    })?;
    let (a_type, _b_type) = extract_block_type_args(&block_field.ty)?;
    let parent_name = constraint.parent_name.clone()
        .unwrap_or_else(|| a_type.to_lowercase());

    // Collect variable setup statements
    let mut var_setup: Vec<TokenStream2> = Vec::new();

    // Ref fields
    for (field_name, _) in ref_paths {
        if let Some(field) = fields.iter().find(|f| f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone())) {
            if let Some((_, inner_ident)) = extract_wrapper_inner(&field.ty, "Ref") {
                let var_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
                let type_ident = syn::Ident::new(&inner_ident.to_string(), inner_ident.span());
                let name_str = field_name.as_str();
                var_setup.push(quote! { let #var_ident = #type_ident::sym(#name_str); });
            }
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
    if let syn::Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
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
    }
    Err(syn::Error::new_spanned(ty, "expected SelfBlock<A>, CrossBlock<A, B>, or TripletBlock"))
}

fn type_ident_name(ty: &syn::Type) -> syn::Result<String> {
    if let syn::Type::Path(tp) = ty {
        if let Some(seg) = tp.path.segments.last() {
            return Ok(seg.ident.to_string());
        }
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
) -> syn::Result<TokenStream2> {
    let stashed = crate::registry_take_constraints();
    let root_var_name = root_name.to_string().to_lowercase();
    let root_var_ident = syn::Ident::new(&root_var_name, proc_macro2::Span::call_site());
    let cast_type: syn::Type = syn::parse_str(precision)
        .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
            format!("invalid precision type '{}': {}", precision, e)))?;
    let constraint_impls: Vec<TokenStream2> = Vec::new();
    let mut cost_loops: Vec<TokenStream2> = Vec::new();
    let mut grad_hessian_loops: Vec<TokenStream2> = Vec::new();
    let mut set_block_indices_loops: Vec<TokenStream2> = Vec::new();

    // Grouping for constraints that iterate the same collection.
    // Merges SelfBlock + nested CrossBlock into a single loop per collection.
    struct CollectionGroup {
        coll_ident: syn::Ident,
        self_var: syn::Ident,
        a_type_ident: syn::Ident,
        // SelfBlock: index setup + constraint entries
        self_block: Option<SelfBlockInfo>,
        // Cost/GH entries that go directly in the outer loop (SelfBlock constraints)
        cost_entries: Vec<TokenStream2>,
        gh_entries: Vec<TokenStream2>,
        // Nested CrossBlock: inner loops over frines
        nested_cost_loops: Vec<TokenStream2>,
        nested_gh_loops: Vec<TokenStream2>,
    }
    struct SelfBlockInfo {
        a_param_count: usize,
        a_idx_stmts: Vec<TokenStream2>,
        block_ident: syn::Ident,
    }
    let mut collection_groups: std::collections::HashMap<String, CollectionGroup> = std::collections::HashMap::new();

    let mut _generated_constraints_fn: std::collections::HashSet<String> = std::collections::HashSet::new();

    // Collect all types reachable from this root (for multi-root support)
    let reachable = {
        let mut set = std::collections::HashSet::new();
        let mut queue = Vec::new();
        // Seed with types directly in root fields
        let root_fields_parsed: syn::FieldsNamed = syn::parse2(quote! { { #root_fields } })?;
        for field in &root_fields_parsed.named {
            if let syn::Type::Path(tp) = &field.ty {
                if let Some(seg) = tp.path.segments.last() {
                    // Extract inner type from Vec<T>, Deque<T>, etc.
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                            if let Ok(name) = type_ident_name(inner) {
                                queue.push(name);
                            }
                        }
                    }
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

    for sc in &stashed {
        // Skip constraints for types not reachable from this root
        if !reachable.contains(&sc.struct_name) { continue; }
        // Re-parse constraint
        let attr_ts: proc_macro2::TokenStream = sc.attr_tokens.parse()
            .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                format!("failed to re-parse constraint for {}: {}", sc.struct_name, e)))?;
        let attr_tokens: Vec<proc_macro2::TokenTree> = attr_ts.into_iter().collect();
        let err_ident = syn::Ident::new(&sc.struct_name, proc_macro2::Span::call_site());

        let constraint = match &attr_tokens[0] {
            proc_macro2::TokenTree::Ident(id) if id.to_string() == "constraint" => {
                if let Some(proc_macro2::TokenTree::Group(g)) = attr_tokens.get(1) {
                    parse_constraint_inner_impl(
                        &g.stream().into_iter().collect::<Vec<_>>(), &err_ident)?
                } else { None }
            }
            _ => None,
        };
        let constraint = match constraint { Some(c) => c, None => continue };

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
        let is_remote_block = constraint.block_field.contains('.');

        let (a_type, b_type, remote_block_info) = if is_remote_block {
            // Remote block: e.g. "pose.hb_pose" means the block lives on a Ref<Pose>'s field
            let parts: Vec<&str> = constraint.block_field.split('.').collect();
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
                f.ident.as_ref().map(|i| i.to_string()) == Some(constraint.block_field.clone())
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

        // For SelfBlock: the struct itself is in a root collection
        // For CrossBlock: find parent collection + frines field
        let self_var_name = if is_self_block {
            a_type.to_lowercase()
        } else {
            parent_name.clone()
        };

        // Find root collection containing the constrained type
        let coll_type = if is_triplet || is_self_block { &sc.struct_name } else { &a_type };
        let root_collection = find_root_collection(root_fields, coll_type);
        if root_collection.is_none() { continue; }
        let (coll_ident_str, _) = root_collection.unwrap();
        let coll_ident = syn::Ident::new(&coll_ident_str, proc_macro2::Span::call_site());

        // CrossBlock/remote: find frines field and build ref resolution
        let mut frines_ident = None;
        let mut resolve_stmts = Vec::new();
        let mut parent_ident = None;
        let mut is_root_level_cross = false;  // constraint struct lives directly on root

        if is_triplet || (!is_self_block || is_remote_block) {
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

        // Apply euler_angles substitutions from all referenced types
        let mut all_subs: Vec<(arael_sym::E, arael_sym::E)> = Vec::new();
        // Build var_infos to know which variables reference which types
        let struct_layout_for_subs = registry_lookup(&sc.struct_name);
        let ref_paths_for_subs = struct_layout_for_subs.as_ref()
            .map(|l| l.ref_paths.clone()).unwrap_or_default();
        // Check each variable's type for euler_angle_fields
        for (field_name, _) in &ref_paths_for_subs {
            if let Some(field) = fields.named.iter().find(|f|
                f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone())) {
                if let Some((_, inner_ident)) = extract_wrapper_inner(&field.ty, "Ref") {
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
        if is_self_block {
            if let Some(self_layout) = registry_lookup(&sc.struct_name) {
                for ea in &self_layout.euler_angle_fields {
                    all_subs.extend(build_euler_substitutions(&self_var_name, ea));
                }
            }
        }

        let block_ident = if is_remote_block {
            // For remote blocks, the actual block field name is the last segment
            let parts: Vec<&str> = constraint.block_field.split('.').collect();
            syn::Ident::new(parts.last().unwrap(), proc_macro2::Span::call_site())
        } else {
            syn::Ident::new(&constraint.block_field, proc_macro2::Span::call_site())
        };
        let param_strs: Vec<&str> = param_symbols.iter().map(|s| s.as_str()).collect();
        let n_params = param_symbols.len();
        let n_residuals = residual_exprs.len();

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
        for (name, expr) in &gh_intermediates {
            let name_ident = syn::Ident::new(name, proc_macro2::Span::call_site());
            let code: Expr = parse_sym_code(&expr.to_rust(""))?;
            gh_stmts.push(quote! { let #name_ident= #code; });
        }
        let mut idx = 0;
        for ri in 0..n_residuals {
            let r_ident = syn::Ident::new(&format!("__r_{}", ri), proc_macro2::Span::call_site());
            let r_expr: Expr = parse_sym_code(&gh_simplified[idx].to_rust(""))?;
            gh_stmts.push(quote! { let #r_ident= #r_expr; });
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
                // TripletBlock: pass indices + derivatives as slices
                gh_stmts.push(quote! {
                    __frine.#block_ident.add_residual(#r_ident as #cast_type, &__all_idx, &[#(#dr_f64),*]);
                });
            } else if is_remote_block {
                gh_stmts.push(quote! {
                    __target_block.add_residual(#r_ident as #cast_type, &[#(#dr_f64),*]);
                });
            } else {
                let block_owner = if is_self_block {
                    syn::Ident::new("__item", proc_macro2::Span::call_site())
                } else {
                    syn::Ident::new("__frine", proc_macro2::Span::call_site())
                };
                gh_stmts.push(quote! {
                    #block_owner.#block_ident.add_residual(#r_ident as #cast_type, &[#(#dr_f64),*]);
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
                                        .map(|(_, id)| id.to_string() == a_type)
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
        if let Some(ref b_type_name) = b_type {
            if let Some(b_layout) = registry_lookup(b_type_name) {
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

        // TripletBlock: build flat index array from all ref fields
        let mut triplet_idx_stmts: Vec<TokenStream2> = Vec::new();
        let mut triplet_param_count = 0usize;
        if is_triplet {
            let struct_layout = registry_lookup(&sc.struct_name);
            let ref_paths = struct_layout.as_ref().map(|l| l.ref_paths.clone()).unwrap_or_default();
            let mut used = std::collections::HashSet::new();
            for (field_name, _) in &ref_paths {
                if !used.insert(field_name.clone()) { continue; }
                if let Some(field) = fields.named.iter().find(|f|
                    f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone())) {
                    if let Some((_, inner_ident)) = extract_wrapper_inner(&field.ty, "Ref") {
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
            }
        }

        // Parse guard expression — replace "self" with the loop variable
        let guard_expr: Option<syn::Expr> = constraint.guard.as_ref()
            .map(|g| {
                let adjusted = if is_self_block {
                    g.replacen("self.", &format!("{}.", self_var_name), 10)
                } else {
                    g.clone()
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

            // Cost loop: iterate parent -> frines, resolve refs, evaluate
            cost_loops.push(quote! {
                for __lm in self.#coll_ident.iter() {
                    for __frine in &__lm.#frines_ident {
                        #(#resolve_stmts)*
                        let #parent_ident = __lm;
                        let #root_var_ident = &*__self_ref;
                        #(#cost_stmts)*
                    }
                }
            });

            // Grad+hessian loop: same traversal but get mutable access to target block
            let _target_coll_id = target_coll_ident.unwrap();
            grad_hessian_loops.push(quote! {
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
            });

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
        } else if is_self_block {
            // SelfBlock: group by collection for merged loop generation
            let self_var = syn::Ident::new(&self_var_name, proc_macro2::Span::call_site());
            let group_key = coll_ident_str.clone();

            let cost_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #(#cost_stmts)* } }
            } else {
                quote! { { #(#cost_stmts)* } }
            };

            let gh_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #(#gh_stmts)* } }
            } else {
                quote! { { #(#gh_stmts)* } }
            };

            let group = collection_groups.entry(group_key).or_insert_with(|| CollectionGroup {
                coll_ident: coll_ident.clone(),
                self_var: self_var.clone(),
                a_type_ident: a_type_ident.clone(),
                self_block: None,
                cost_entries: Vec::new(),
                gh_entries: Vec::new(),
                nested_cost_loops: Vec::new(),
                nested_gh_loops: Vec::new(),
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
        } else if is_triplet {
            // TripletBlock: N-ary constraint, flat iteration on root collection
            let rc_ident = frines_ident.unwrap();
            let tp = triplet_param_count;

            cost_loops.push(quote! {
                for __frine in self.#rc_ident.iter() {
                    #(#resolve_stmts)*
                    let #root_var_ident = &*__self_ref;
                    #(#cost_stmts)*
                }
            });

            grad_hessian_loops.push(quote! {
                for __frine in self.#rc_ident.iter_mut() {
                    #(#resolve_stmts)*
                    let #root_var_ident = &*__self_ref;
                    let mut __all_idx = [0u32; #tp];
                    #(#triplet_idx_stmts)*
                    { #(#gh_stmts)* }
                }
            });
            // No set_block_indices needed for TripletBlock
        } else if is_root_level_cross {
            // Root-level CrossBlock: constraint struct is directly on root (e.g. PosePair, CoincidentPP)
            // Flat iteration, no nesting. frines_ident = root collection name of constraint struct.
            let rc_ident = frines_ident.unwrap();

            cost_loops.push(quote! {
                for __frine in self.#rc_ident.iter() {
                    #(#resolve_stmts)*
                    let #root_var_ident = &*__self_ref;
                    #(#cost_stmts)*
                }
            });

            grad_hessian_loops.push(quote! {
                for __frine in self.#rc_ident.iter_mut() {
                    #(#resolve_stmts)*
                    let #root_var_ident = &*__self_ref;
                    { #(#gh_stmts)* }
                }
            });

            set_block_indices_loops.push(quote! {
                for __frine in self.#rc_ident.iter_mut() {
                    #(#resolve_stmts)*
                    let mut __a_idx = [0u32; #a_param_count];
                    #(#a_idx_stmts)*
                    let mut __b_idx = [0u32; #b_param_count];
                    #(#b_idx_stmts)*
                    __frine.#block_ident.set_indices(&__a_idx, &__b_idx);
                }
            });
        } else {
            // Nested CrossBlock: add inner loops to the collection group
            let frines_ident = frines_ident.unwrap();
            let parent_ident = parent_ident.unwrap();
            let group_key = coll_ident_str.clone();

            let self_var = syn::Ident::new(&a_type.to_lowercase(), proc_macro2::Span::call_site());

            let nested_cost = quote! {
                {
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
                    let #parent_ident = unsafe { &*(__item as *const #a_type_ident) };
                    for __frine in __item.#frines_ident.iter_mut() {
                        #(#resolve_stmts)*
                        let #root_var_ident = &*__self_ref;
                        { #(#gh_stmts)* }
                    }
                }
            };

            let group = collection_groups.entry(group_key).or_insert_with(|| CollectionGroup {
                coll_ident: coll_ident.clone(),
                self_var: self_var.clone(),
                a_type_ident: a_type_ident.clone(),
                self_block: None,
                cost_entries: Vec::new(),
                gh_entries: Vec::new(),
                nested_cost_loops: Vec::new(),
                nested_gh_loops: Vec::new(),
            });
            group.nested_cost_loops.push(nested_cost);
            group.nested_gh_loops.push(nested_gh);

            set_block_indices_loops.push(quote! {
                for __item in self.#coll_ident.iter_mut() {
                    let mut __a_idx = [0u32; #a_param_count];
                    #(#a_idx_stmts)*
                    for __frine in __item.#frines_ident.iter_mut() {
                        #(#resolve_stmts)*
                        let mut __b_idx = [0u32; #b_param_count];
                        #(#b_idx_stmts)*
                        __frine.#block_ident.set_indices(&__a_idx, &__b_idx);
                    }
                }
            });
        }
    }

    // Emit merged loops for collection groups
    for (_key, group) in &collection_groups {
        let coll = &group.coll_ident;
        let self_var = &group.self_var;
        let a_type = &group.a_type_ident;
        let cost_entries = &group.cost_entries;
        let gh_entries = &group.gh_entries;
        let nested_cost = &group.nested_cost_loops;
        let nested_gh = &group.nested_gh_loops;

        // Merged cost loop: SelfBlock entries + nested CrossBlock inner loops
        cost_loops.push(quote! {
            for __item in self.#coll.iter() {
                let #self_var = __item;
                let #root_var_ident = &*__self_ref;
                #(#cost_entries)*
                #(#nested_cost)*
            }
        });

        // Merged grad+hessian loop
        grad_hessian_loops.push(quote! {
            for __item in self.#coll.iter_mut() {
                let #self_var = unsafe { &*(__item as *const #a_type) };
                let #root_var_ident = &*__self_ref;
                #(#gh_entries)*
                #(#nested_gh)*
            }
        });

        // set_block_indices loop (only if there's a SelfBlock)
        if let Some(ref sb) = group.self_block {
            let a_count = sb.a_param_count;
            let a_idx = &sb.a_idx_stmts;
            let block = &sb.block_ident;
            set_block_indices_loops.push(quote! {
                for __item in self.#coll.iter_mut() {
                    let mut __a_idx = [0u32; #a_count];
                    #(#a_idx)*
                    __item.#block.set_indices(&__a_idx);
                }
            });
        }
    }

    // Generate methods on root -- precision-aware
    let prec_type: syn::Type = syn::parse_str(precision)
        .map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
            format!("invalid precision type '{}': {}", precision, e)))?;
    let update_method = syn::Ident::new(
        &format!("update{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_method = syn::Ident::new(
        &format!("accumulate_blocks{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_band_method = syn::Ident::new(
        &format!("accumulate_blocks_band{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_sparse_method = syn::Ident::new(
        &format!("accumulate_blocks_sparse{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_sparse_direct_method = syn::Ident::new(
        &format!("accumulate_blocks_sparse_direct{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());
    let accumulate_sparse_indexed_method = syn::Ident::new(
        &format!("accumulate_blocks_sparse_indexed{}", if precision == "f32" { "32" } else { "64" }),
        proc_macro2::Span::call_site());

    // Build advance() body: absorb universal_euler_angles deltas
    let advance_stmts = {
        let mut stmts: Vec<TokenStream2> = Vec::new();
        let root_fields_for_advance: syn::FieldsNamed = syn::parse2(quote! { { #root_fields } })?;
        for field in &root_fields_for_advance.named {
            let field_ident = field.ident.as_ref().unwrap().clone();
            if let Some((_, inner_ident)) = crate::extract_wrapper_inner(&field.ty, "Vec")
                .or_else(|| crate::extract_wrapper_inner(&field.ty, "Deque"))
                .or_else(|| crate::extract_wrapper_inner(&field.ty, "Arena"))
            {
                let inner_name = inner_ident.to_string();
                if let Some(layout) = crate::registry_lookup(&inner_name) {
                    for ea_field in &layout.universal_euler_angle_fields {
                        let ea_ident = syn::Ident::new(ea_field, proc_macro2::Span::call_site());
                        stmts.push(quote! {
                            for __item in self.#field_ident.iter_mut() {
                                let __idx = __item.#ea_ident.index() as usize;
                                __item.#ea_ident.advance();
                                params[__idx] = 0.0 as #cast_type;
                                params[__idx + 1] = 0.0 as #cast_type;
                                params[__idx + 2] = 0.0 as #cast_type;
                            }
                        });
                    }
                }
            }
        }
        stmts
    };

    let (extended_cost_call, extended_compute_call) = if precision == "f64" {
        (quote! { __cost += arael::model::ExtendedModel::extended_cost64(self, params); },
         quote! { arael::model::ExtendedModel::extended_compute64(self, params); })
    } else {
        (quote! { __cost += arael::model::ExtendedModel::extended_cost32(self, params); },
         quote! { arael::model::ExtendedModel::extended_compute32(self, params); })
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
            }
            pub fn serialize32(&mut self, data: &mut std::vec::Vec<f32>) {
                arael::model::Model::serialize_params32(self, data);
                self.__set_block_indices();
            }
            pub fn deserialize32(&mut self, data: &[f32]) {
                arael::model::Model::deserialize_params32(self, data);
            }

            fn __set_block_indices(&mut self) {
                #(#set_block_indices_loops)*
            }

            fn __compute_blocks(&mut self, params: &[#prec_type]) {
                arael::model::Model::#update_method(self, params);
                let __self_ref = unsafe { &*(self as *const Self) };
                self.zero_blocks();
                #(#grad_hessian_loops)*
                #extended_compute_call
            }
        }

        impl arael::simple_lm::LmProblem<#prec_type> for #root_name {
            fn calc_cost(&mut self, params: &[#prec_type]) -> #prec_type {
                arael::model::Model::#update_method(self, params);
                let __self_ref = unsafe { &*(self as *const Self) };
                let mut __cost = 0.0 as #prec_type;
                #(#cost_loops)*
                #extended_cost_call
                __cost
            }

            fn calc_grad_hessian_dense(&mut self, params: &[#prec_type], grad: &mut [#prec_type], hessian: &mut [#prec_type]) {
                self.__compute_blocks(params);
                grad.iter_mut().for_each(|g| *g = 0.0);
                hessian.iter_mut().for_each(|h| *h = 0.0);
                self.#accumulate_method(grad, hessian);
            }

            fn calc_grad_hessian_band(&mut self, params: &[#prec_type], grad: &mut [#prec_type], band: &mut [#prec_type], kd: usize) -> Result<(), arael::simple_lm::BandError> {
                self.__compute_blocks(params);
                grad.iter_mut().for_each(|g| *g = 0.0);
                band.iter_mut().for_each(|b| *b = 0.0);
                self.#accumulate_band_method(grad, band, kd)
            }

            fn calc_grad_hessian_sparse(&mut self, params: &[#prec_type], grad: &mut [#prec_type], coo: &mut arael::simple_lm::CooMatrix<#prec_type>) {
                self.__compute_blocks(params);
                grad.iter_mut().for_each(|g| *g = 0.0);
                coo.clear();
                self.#accumulate_sparse_method(grad, coo);
            }

            fn calc_grad_hessian_sparse_direct(&mut self, params: &[#prec_type], grad: &mut [#prec_type], csc: &mut arael::simple_lm::CscMatrix<#prec_type>) {
                self.__compute_blocks(params);
                grad.iter_mut().for_each(|g| *g = 0.0);
                csc.vals.iter_mut().for_each(|v| *v = 0.0 as #prec_type);
                self.#accumulate_sparse_direct_method(grad, csc);
            }

            fn calc_grad_hessian_sparse_indexed(&mut self, params: &[#prec_type], grad: &mut [#prec_type], vals: &mut [#prec_type], positions: &[usize]) {
                self.__compute_blocks(params);
                grad.iter_mut().for_each(|g| *g = 0.0);
                vals.iter_mut().for_each(|v| *v = 0.0 as #prec_type);
                let mut cursor = 0usize;
                self.#accumulate_sparse_indexed_method(grad, vals, positions, &mut cursor);
            }

            fn advance(&mut self, params: &mut [#prec_type]) {
                #(#advance_stmts)*
            }
        }
    };

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
    let is_remote = constraint.block_field.contains('.');
    let (a_type, b_type) = if is_remote {
        // Remote block: e.g. "pose.hb_pose" — target type from Ref field
        let parts: Vec<&str> = constraint.block_field.split('.').collect();
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
            f.ident.as_ref().map(|i| i.to_string()) == Some(constraint.block_field.clone())
        }).ok_or_else(|| {
            syn::Error::new_spanned(struct_name, format!("block field '{}' not found", constraint.block_field))
        })?;
        extract_block_type_args(&block_field.ty)?
    };
    let struct_layout = registry_lookup(&struct_name.to_string());
    let ref_paths = struct_layout.as_ref().map(|l| &l.ref_paths[..]).unwrap_or(&[]);
    let parent_name = constraint.parent_name.clone().unwrap_or_else(|| a_type.to_lowercase());

    // Build var_infos
    let mut var_infos: Vec<(String, String)> = Vec::new();
    if !constraint.vars.is_empty() {
        for var in &constraint.vars {
            if let Some(ref tn) = var.type_name { var_infos.push((var.name.clone(), tn.clone())); }
        }
    } else {
        for (field_name, _) in ref_paths {
            if let Some(field) = fields.iter().find(|f| f.ident.as_ref().map(|i| i.to_string()) == Some(field_name.clone())) {
                if let Some((_, inner_ident)) = extract_wrapper_inner(&field.ty, "Ref") {
                    var_infos.push((field_name.clone(), inner_ident.to_string()));
                }
            }
        }
        if a_type != "__triplet__" {
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

    if is_triplet {
        // TripletBlock: collect params from ALL ref fields (no A/B distinction)
        let mut used_vars = std::collections::HashSet::new();
        for (var_name, type_name) in &var_infos {
            if type_name == root_type_name { continue; } // skip root
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
        if let Some(ref b_type_name) = b_type {
            if let Some(b_layout) = registry_lookup(b_type_name) {
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
        if let syn::Type::Path(tp) = &field.ty {
            if let Some(seg) = tp.path.segments.last() {
                let container = seg.ident.to_string();
                if container == "Vec" || container == "Deque" || container == "Arena" {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                        if let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                            if let Ok(inner_name) = type_ident_name(inner) {
                                if inner_name == type_name {
                                    return Some((field_name, container));
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    None
}

#[allow(dead_code)]
fn find_var_for_type_annotated(vars: &[ConstraintVar], type_name: &str) -> syn::Result<String> {
    // First check explicit type annotations
    for v in vars {
        if let Some(ref tn) = v.type_name {
            if tn == type_name {
                return Ok(v.name.clone());
            }
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
    if let Some(_) = registry_lookup(type_name) {
        // The type is registered. Find which variable has a layout that matches.
        for v in var_names {
            if let Some(var_layout) = find_layout_for_var(v) {
                if let Some(type_layout) = registry_lookup(type_name) {
                    if var_layout.param_fields == type_layout.param_fields {
                        return Ok(v.clone());
                    }
                }
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
    let stashed = crate::registry_take_constraints();
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
            proc_macro2::TokenTree::Ident(id) if id.to_string() == "constraint" => {
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

