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

/// Generated body of `__precompute_symbolic`: numeric evaluation of every
/// `#[arael(symbolic = ...)]` field and every declared `#[arael(deriv = ...)]`
/// cache, shared through one CSE pass. Called wherever the model refreshes
/// its computed fields, so generated constraint code can read the fields
/// (and their deriv caches) instead of re-deriving the expressions per
/// observation.
pub(crate) fn generate_symbolic_precompute(
    type_name: &str,
    fields: &[(String, SymFieldType)],
    param_fields: &[String],
    symbolic_fields: &[(String, String)],
    deriv_fields: &[(String, String, String)],
    scalar_generic: Option<&str>,
) -> syn::Result<TokenStream2> {
    if symbolic_fields.is_empty() {
        return Ok(TokenStream2::new());
    }
    let sp = proc_macro2::Span::call_site();
    let mut scratch = ConstraintCtx::new();
    for (fname, sft) in fields {
        if matches!(sft,
            SymFieldType::Skip
            | SymFieldType::Struct(_)
            | SymFieldType::OptionalStruct(_)) { continue; }
        let is_param = param_fields.iter().any(|p| p == fname);
        let base = if is_param {
            format!("self.{}.work()", fname)
        } else {
            format!("self.{}", fname)
        };
        scratch.bindings.insert(fname.clone(), ConstraintCtx::make_sym_val(&base, sft));
        if is_param {
            scratch.bindings.insert(format!("{}_value", fname),
                ConstraintCtx::make_sym_val(&format!("self.{}.value", fname), sft));
        }
    }
    // (assignment target, component expression, Some(by-param) for
    // deriv-cache entries), values in declaration order then deriv-cache
    // entries. Everything stays fully inline (no reads of the just-written
    // fields), so the CSE below owns all the sharing and the emission order
    // cannot go stale.
    let mut assigns: Vec<(String, E, Option<String>)> = Vec::new();
    let mut sym_vals: Vec<(String, SymVal)> = Vec::new();
    for (fname, expr_str) in symbolic_fields {
        let parsed: Expr = syn::parse_str(expr_str).map_err(|e| {
            syn::Error::new(sp, format!(
                "symbolic = expression on `{}.{}` does not parse: {}",
                type_name, fname, e))
        })?;
        let val = eval_expr(&parsed, &mut scratch).map_err(|e| {
            syn::Error::new(e.span(), format!(
                "symbolic = expression on `{}.{}`: {}", type_name, fname, e))
        })?;
        let comps = symval_components(&val).ok_or_else(|| syn::Error::new(sp,
            format!("symbolic field `{}.{}`: only scalar, vec2 and vec3 \
                     shapes can be precomputed", type_name, fname)))?;
        for (suffix, e) in &comps {
            assigns.push((format!("self.{}{}", fname, suffix), e.clone(), None));
        }
        sym_vals.push((fname.clone(), val.clone()));
        scratch.bindings.insert(fname.clone(), val);
    }
    for (dfield, of, by) in deriv_fields {
        let Some((_, of_val)) = sym_vals.iter().find(|(n, _)| n == of) else {
            return Err(syn::Error::new(sp, format!(
                "`{}.{}`: deriv `of = {}` must name a `symbolic =` field",
                type_name, dfield, of)));
        };
        let by_sft = fields.iter()
            .find(|(n, _)| n == by)
            .filter(|_| param_fields.iter().any(|p| p == by))
            .map(|(_, t)| t);
        let with = |names: &[&str]| -> Vec<String> {
            names.iter().map(|c| format!("self.{}.work().{}", by, c)).collect()
        };
        let dvars: Vec<String> = match by_sft {
            Some(SymFieldType::Scalar) => vec![format!("self.{}.work()", by)],
            Some(SymFieldType::Vec2) => with(&["x", "y"]),
            Some(SymFieldType::Vec3) => with(&["x", "y", "z"]),
            _ => return Err(syn::Error::new(sp, format!(
                "`{}.{}`: deriv `by = {}` must name a scalar, vec2 or vec3 \
                 param field", type_name, dfield, by))),
        };
        let comps = symval_components(of_val).expect("of is a precomputed shape");
        for (k, dvar) in dvars.iter().enumerate() {
            for (suffix, e) in &comps {
                assigns.push((format!("self.{}[{}]{}", dfield, k, suffix),
                    e.diff(dvar.as_str()), Some(by.clone())));
            }
        }
    }
    let exprs: Vec<E> = assigns.iter().map(|(_, e, _)| e.clone()).collect();
    let (inters, outs) = arael_sym::cse(&exprs);
    // In a generic `T: Float` struct an unsuffixed literal cannot infer
    // its type; emit every literal through a local conversion closure.
    // The closure's explicit return type pins each `__c(lit)` to the
    // scalar parameter -- a generic helper fn would leave `__c(x) * t`
    // with an unresolved `?F: Mul<T>` obligation instead of `?F = T`.
    let generic = scalar_generic.is_some();
    let emit = |e: &E| -> syn::Result<Expr> {
        parse_sym_code(&if generic { e.to_rust_generic() } else { e.to_rust("") })
    };
    let mut stmts: Vec<TokenStream2> = Vec::new();
    if let Some(sg) = scalar_generic {
        let sg = syn::Ident::new(sg, sp);
        stmts.push(quote! {
            #[allow(unused_variables)]
            let __c = |v: f64| -> #sg { #sg::from(v).unwrap() };
        });
    }
    for (iname, e) in &inters {
        let id = syn::Ident::new(iname, sp);
        let code = emit(e)?;
        stmts.push(quote! { let #id = #code; });
    }
    // Values unconditionally; deriv-cache stores grouped per `by` param and
    // guarded on it being optimized -- a fixed param's Jacobian entries are
    // never read, so filling its cache would be pure waste.
    let mut guarded: std::collections::BTreeMap<String, Vec<TokenStream2>> =
        std::collections::BTreeMap::new();
    for ((lhs, _, by), out) in assigns.iter().zip(&outs) {
        let lhs_expr: Expr = syn::parse_str(lhs).map_err(|e| syn::Error::new(sp,
            format!("internal: bad precompute target `{}`: {}", lhs, e)))?;
        let code = emit(out)?;
        let assign = quote! { #lhs_expr = #code; };
        match by {
            None => stmts.push(assign),
            Some(b) => guarded.entry(b.clone()).or_default().push(assign),
        }
    }
    for (by, group) in guarded {
        let by_id = syn::Ident::new(&by, sp);
        stmts.push(quote! {
            if self.#by_id.index() != u32::MAX {
                #(#group)*
            }
        });
    }
    Ok(quote! {
        /// Refresh `symbolic =` field values and declared `deriv =` caches
        /// from the current parameters. Generated; called by the update and
        /// deserialize paths.
        #[doc(hidden)]
        pub fn __precompute_symbolic(&mut self) {
            #(#stmts)*
        }
    })
}

/// The scalar components of a symbolic value, each with the suffix its
/// field-read symbol carries ("" for a scalar) -- the same suffixes the
/// runtime type indexes by, so a component doubles as an assignment
/// target (`self.rot[0].x`). None for shapes the symbolic-field cache
/// does not support.
fn symval_components(v: &SymVal) -> Option<Vec<(&'static str, E)>> {
    Some(match v {
        SymVal::Scalar(e) => vec![("", e.clone())],
        SymVal::Vec2(v2) => vec![(".x", v2.x.clone()), (".y", v2.y.clone())],
        SymVal::Vec3(v3) => vec![(".x", v3.x.clone()), (".y", v3.y.clone()), (".z", v3.z.clone())],
        SymVal::Mat2(m) => vec![
            ("[0].x", m.rows[0].x.clone()), ("[0].y", m.rows[0].y.clone()),
            ("[1].x", m.rows[1].x.clone()), ("[1].y", m.rows[1].y.clone()),
        ],
        SymVal::Mat3(m) => vec![
            ("[0].x", m.rows[0].x.clone()), ("[0].y", m.rows[0].y.clone()), ("[0].z", m.rows[0].z.clone()),
            ("[1].x", m.rows[1].x.clone()), ("[1].y", m.rows[1].y.clone()), ("[1].z", m.rows[1].z.clone()),
            ("[2].x", m.rows[2].x.clone()), ("[2].y", m.rows[2].y.clone()), ("[2].z", m.rows[2].z.clone()),
        ],
        SymVal::Quat(q) => vec![
            (".t", q.t.clone()),
            (".v.x", q.v.x.clone()), (".v.y", q.v.y.clone()), (".v.z", q.v.z.clone()),
        ],
        SymVal::UniversalEulerAngles { .. } => return None,
    })
}

/// Rebuild a symbolic value of `shape`'s kind from replacement components.
fn symval_from_components(shape: &SymVal, mut comps: Vec<E>) -> SymVal {
    let c = |i: usize| comps[i].clone();
    match shape {
        SymVal::Scalar(_) => SymVal::Scalar(comps.remove(0)),
        SymVal::Vec2(_) => SymVal::Vec2(vect2sym::from_components(c(0), c(1))),
        SymVal::Vec3(_) => SymVal::Vec3(vect3sym::from_components(c(0), c(1), c(2))),
        SymVal::Mat2(_) => SymVal::Mat2(matrix2sym::from_elements(c(0), c(1), c(2), c(3))),
        SymVal::Mat3(_) => SymVal::Mat3(matrix3sym::from_elements(
            c(0), c(1), c(2), c(3), c(4), c(5), c(6), c(7), c(8))),
        SymVal::Quat(_) => SymVal::Quat(quaternsym {
            t: c(0),
            v: vect3sym::from_components(c(1), c(2), c(3)),
        }),
        SymVal::UniversalEulerAngles { .. } => unreachable!("guarded by symval_components"),
    }
}

struct ConstraintCtx {
    // variable name -> SymVal
    bindings: HashMap<String, SymVal>,
    // Entity variables (heads of pre-registered dotted paths) and their
    // registered type names. Drives field-path validation: a dotted path
    // starting at a known entity that is neither pre-registered nor a
    // skip/opaque field is a typo, reported with a suggestion instead of
    // being spliced into generated code as a free symbol.
    entity_vars: HashMap<String, String>,
    // User `let` binding names. Lets shadow pre-registered entity paths
    // (Rust semantics), so field lookup consults these before the dotted
    // binding table.
    lets: std::collections::HashSet<String>,
    // Substitutions collected while registering bindings: cached() units of
    // symbolic component fields (values and declared deriv caches), resolved
    // after differentiation to precomputed field reads -- the same pattern
    // as the rotation-matrix substitutions.
    subs: Vec<(arael_sym::E, arael_sym::E)>,
}

impl ConstraintCtx {
    fn new() -> Self {
        ConstraintCtx {
            bindings: HashMap::new(),
            entity_vars: HashMap::new(),
            lets: std::collections::HashSet::new(),
            subs: Vec::new(),
        }
    }

    /// The bindings a constraint body may start a path from, for error
    /// messages: entity variables with their types (the self alias is
    /// the lowercased struct name), then user `let`s.
    fn available_bindings_hint(&self) -> String {
        let mut vars: Vec<String> = self.entity_vars.iter()
            .map(|(name, ty)| format!("{} ({})", name, ty))
            .collect();
        vars.sort();
        let mut lets: Vec<&str> = self.lets.iter().map(|s| s.as_str()).collect();
        lets.sort();
        vars.extend(lets.into_iter().map(|l| format!("{} (let)", l)));
        if vars.is_empty() {
            "none".to_string()
        } else {
            vars.join(", ")
        }
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
        // Block with `let` intermediates and one final expression -- the
        // constraint-body grammar, usable anywhere an expression is (a
        // `symbolic =` chain, most usefully). The bindings are scoped to
        // the block: shadowed values are restored and new names removed on
        // exit, so intermediates cannot leak into sibling expressions.
        Expr::Block(blk) => {
            let mut saved: Vec<(String, Option<SymVal>, bool)> = Vec::new();
            let mut result: Option<SymVal> = None;
            let n = blk.block.stmts.len();
            for (si, stmt) in blk.block.stmts.iter().enumerate() {
                match stmt {
                    Stmt::Local(local) => {
                        let name = match &local.pat {
                            Pat::Ident(pi) => pi.ident.to_string(),
                            _ => return Err(syn::Error::new_spanned(&local.pat,
                                "simple let binding required")),
                        };
                        let init = local.init.as_ref().ok_or_else(||
                            syn::Error::new_spanned(local, "initializer required"))?;
                        let val = eval_expr(&init.expr, ctx)?;
                        let prev = ctx.bindings.insert(name.clone(), val);
                        let newly = ctx.lets.insert(name.clone());
                        saved.push((name, prev, newly));
                    }
                    Stmt::Expr(e, semi) => {
                        if si + 1 != n || semi.is_some() {
                            return Err(syn::Error::new_spanned(e,
                                "a block expression holds `let` bindings and ONE final \
                                 expression, no trailing semicolon"));
                        }
                        result = Some(eval_expr(e, ctx)?);
                    }
                    other => return Err(syn::Error::new_spanned(other,
                        "only `let` bindings may precede the final expression")),
                }
            }
            for (name, prev, newly) in saved.into_iter().rev() {
                match prev {
                    Some(v) => { ctx.bindings.insert(name.clone(), v); }
                    None => { ctx.bindings.remove(&name); }
                }
                if newly {
                    ctx.lets.remove(&name);
                }
            }
            result.ok_or_else(|| syn::Error::new_spanned(expr,
                "empty block: a final expression is required"))
        }
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
                    format!("unknown variable '{}' in constraint; available bindings: {}",
                        name, ctx.available_bindings_hint())));
            }
            Err(syn::Error::new_spanned(expr, "unsupported path in constraint"))
        }

        // Field access: expr.field
        Expr::Field(ef) => {
            let field_name = match &ef.member {
                syn::Member::Named(n) => n.to_string(),
                syn::Member::Unnamed(i) => i.index.to_string(),
            };

            // Try to resolve as a dotted path first (e.g., "pose.ea" as a
            // binding key) -- unless the path head is a user `let`, which
            // shadows pre-registered entity paths (Rust semantics; the
            // dotted lookup used to win, silently ignoring the let).
            let dotted = build_dotted_path(expr);
            let head_is_let = dotted.as_ref()
                .and_then(|p| p.split('.').next())
                .is_some_and(|h| ctx.lets.contains(h));
            if !head_is_let
                && let Some(ref path) = dotted
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

            // Fallback: a scalar symbol spliced verbatim into generated
            // code. Legitimate for skip fields and fields of types not
            // registered with #[arael::model]; a typo would otherwise
            // surface as a rustc E0609 at the root struct with no hint of
            // the constraint, or silently compile against an unintended
            // field. Validate against the registered layouts first.
            if let Some(ref path) = dotted {
                if head_is_let {
                    return Err(syn::Error::new_spanned(expr,
                        format!("cannot access `{}`: the path root is a local `let` binding \
                                 and the field is not a known component", path)));
                }
                let head = path.split('.').next().unwrap();
                if let Some(type_name) = ctx.entity_vars.get(head).cloned() {
                    validate_entity_path(&type_name, head, &path[head.len() + 1..], expr)?;
                } else if !ctx.entity_vars.is_empty() {
                    // In a constraint body every path must start at a
                    // binding: an unknown head is a typo (usually the
                    // self alias, which is the LOWERCASED STRUCT NAME).
                    // It used to be spliced verbatim into generated code
                    // and surface as a bare rustc "cannot find value"
                    // pointing at the whole macro. (Symbolic-precompute
                    // contexts have no entity vars and keep the
                    // passthrough for component field reads.)
                    return Err(syn::Error::new_spanned(expr,
                        format!("unknown binding '{}' in `{}`; available bindings: {}",
                            head, path, ctx.available_bindings_hint())));
                }
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
                    // cached() so the SimpleEulerAngleParam precompute substitution
                    // matches the entries (CSE still dedupes the shared sin/cos
                    // inside; a raw non-param vect3 just gets a harmless barrier).
                    Ok(SymVal::Mat3(cache_rotation_entries(&v.rotation_matrix())))
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
                (SymVal::Mat3(m), "get_rotation_vector_small") => {
                    Ok(SymVal::Vec3(m.get_rotation_vector_small()))
                }
                (SymVal::Mat3(m), "col") | (SymVal::Mat3(m), "row") => {
                    let name = mc.method.to_string();
                    if mc.args.len() != 1 {
                        return Err(syn::Error::new_spanned(&mc.method,
                            format!(".{}() requires 1 argument", name)));
                    }
                    let idx = match &mc.args[0] {
                        syn::Expr::Lit(l) => match &l.lit {
                            syn::Lit::Int(i) => i.base10_parse::<usize>().ok(),
                            _ => None,
                        },
                        _ => None,
                    };
                    match idx {
                        Some(i) if i < 3 => Ok(SymVal::Vec3(
                            if name == "col" { m.col(i) } else { m.row(i) })),
                        _ => Err(syn::Error::new_spanned(&mc.args[0],
                            format!(".{}() index must be an integer literal 0, 1 or 2", name))),
                    }
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
fn build_user_function_bag() -> syn::Result<arael_sym::FunctionBag> {
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
                // A body that does not parse must be reported HERE, with
                // the function named -- skipping it used to surface as a
                // misleading unknown-function error at some caller.
                let body_e = arael_sym::parse_with_functions(body, &sub)
                    .map_err(|err| syn::Error::new(proc_macro2::Span::call_site(),
                        format!("arael::function `{}`: body does not parse: {}",
                            sym_name, err)))?;
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
                // Same rule for deriv expressions: a malformed one is an
                // error naming the function and the string, not a silent
                // drop from the bag.
                let mut derivs: Vec<arael_sym::E> = Vec::with_capacity(*arity);
                for (i, s) in deriv_strings.iter().enumerate() {
                    let e = arael_sym::parse_with_functions(s, &sub)
                        .map_err(|err| syn::Error::new(proc_macro2::Span::call_site(),
                            format!("arael::function `{}`: derivs[{}] `{}` does not parse: {}",
                                sym_name, i, s, err)))?;
                    derivs.push(e);
                }
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
    Ok(bag)
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
        let bag = build_user_function_bag()?;
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

                let mut combined = build_user_function_bag()?;
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
        // The rotation of the normalized first-order quaternion
        // (1, v/2) -- rational (no trig), exact for every v, and the same
        // retraction `QuaternionParam` applies internally. A component
        // that owns its own rotation delta builds its chart with this.
        ("matrix3sym", "from_rotation_vector_small") => {
            arity(1)?;
            Ok(SymVal::Mat3(matrix3sym::from_rotation_vector_small(&vec3(0)?)))
        }
        ("vect2sym", "from_components") => {
            arity(2)?;
            Ok(SymVal::Vec2(vect2sym::from_components(scalar(0)?, scalar(1)?)))
        }
        ("vect3sym", "from_components") => {
            arity(3)?;
            Ok(SymVal::Vec3(vect3sym::from_components(scalar(0)?, scalar(1)?, scalar(2)?)))
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
    pub loss: Option<String>,         // robust loss closure, e.g. "|s| loss_huber(s, self.k)"
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
    let mut loss: Option<String> = None;
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
                            // Collect tokens until the comma before the body.
                            // A top-level brace group only terminates the
                            // guard when it is the FINAL token (the body with
                            // a missing comma) -- guards may legitimately
                            // contain block expressions.
                            let mut guard_tokens = Vec::new();
                            while pos < tokens.len() {
                                match &tokens[pos] {
                                    proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => break,
                                    proc_macro2::TokenTree::Group(g)
                                        if g.delimiter() == proc_macro2::Delimiter::Brace
                                            && pos + 1 == tokens.len() => break,
                                    t => { guard_tokens.push(t.clone()); pos += 1; }
                                }
                            }
                            let guard_ts: proc_macro2::TokenStream = guard_tokens.into_iter().collect();
                            guard = Some(guard_ts.to_string());
                        } else if name == "loss" {
                            // A robust-loss closure `|s| <expr>`. Collect its
                            // tokens up to the comma before the body, with the
                            // same final-brace guard as `guard` (a trailing
                            // `{ body }` with no comma terminates it).
                            let mut loss_tokens = Vec::new();
                            while pos < tokens.len() {
                                match &tokens[pos] {
                                    proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => break,
                                    proc_macro2::TokenTree::Group(g)
                                        if g.delimiter() == proc_macro2::Delimiter::Brace
                                            && pos + 1 == tokens.len() => break,
                                    t => { loss_tokens.push(t.clone()); pos += 1; }
                                }
                            }
                            let loss_ts: proc_macro2::TokenStream = loss_tokens.into_iter().collect();
                            loss = Some(loss_ts.to_string());
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
                                format!("unknown constraint attribute key `{}`, expected `parent`, `guard`, `loss`, or `name`", name)));
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
        loss,
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
/// returns pairs (from_expr, to_expr) that replace rotation matrix entries
/// and their derivatives with precomputed matrix fields. Any sin/cos of the
/// angles left outside those patterns stays inline (evaluated via `work()`).
fn build_euler_substitutions(var_base: &str, field_name: &str) -> Vec<(arael_sym::E, arael_sym::E)> {
    let mut subs = Vec::new();

    let ea_sym = arael_sym::vect3sym::new(&format!("{}.{}.work()", var_base, field_name));
    let rot = ea_sym.rotation_matrix();
    push_rotation_matrix_subs(&mut subs, &rot, var_base, field_name);
    let dvar_prefix = format!("{}.{}.work()", var_base, field_name);
    push_rotation_deriv_subs(&mut subs, &rot, &dvar_prefix, var_base, field_name);

    subs
}

/// Build symbolic substitutions for universal_euler_angles precomputation.
/// Substitutes composed rotation (R_ref * rotation(ea_delta)) entries and
/// their derivatives with precomputed matrix fields. The delta is
/// solver-internal, so no sin/cos of it can appear outside those patterns.
fn build_universal_euler_substitutions(var_base: &str, field_name: &str) -> Vec<(arael_sym::E, arael_sym::E)> {
    let mut subs = Vec::new();

    // Build composed rotation: R_ref * rotation(ea_delta)
    let r_ref_sym = matrix3sym::new(&format!("{}.{}.ref_rotation", var_base, field_name));
    let dea_sym = vect3sym::new(&format!("{}.{}.delta", var_base, field_name));
    let composed = r_ref_sym * dea_sym.rotation_matrix();

    push_rotation_matrix_subs(&mut subs, &composed, var_base, field_name);
    let dvar_prefix = format!("{}.{}.delta", var_base, field_name);
    push_rotation_deriv_subs(&mut subs, &composed, &dvar_prefix, var_base, field_name);

    subs
}

/// Wrap each entry of a symbolic rotation matrix in `cached()`. The composed
/// R_ref * retraction(delta) entries otherwise get mixed into the surrounding
/// residual math and reshaped by simplification, so `replace_pub` can no longer
/// match them against the precompute substitution. `cached()` is an identity
/// barrier: the entry stays a stable, matchable subtree, and the substitution
/// (which also wraps its `from` in `cached()`) hits it reliably. See
/// [`build_universal_rotvec_substitutions`].
fn cache_rotation_entries(m: &matrix3sym) -> matrix3sym {
    let c = |e: &arael_sym::E| arael_sym::cached(e.clone());
    matrix3sym::from_elements(
        c(&m.rows[0].x), c(&m.rows[0].y), c(&m.rows[0].z),
        c(&m.rows[1].x), c(&m.rows[1].y), c(&m.rows[1].z),
        c(&m.rows[2].x), c(&m.rows[2].y), c(&m.rows[2].z),
    )
}

/// Push `cached(entry) -> <field>.rotation_matrix[row].col` substitutions for
/// every entry of a composed rotation matrix. Shared by the three rotation-param
/// substitution builders; the `from` is wrapped in `cached()` to match the
/// cached entries the binding / `.rotation_matrix()` emits (see
/// [`cache_rotation_entries`]).
fn push_rotation_matrix_subs(
    subs: &mut Vec<(arael_sym::E, arael_sym::E)>,
    composed: &matrix3sym,
    var_base: &str,
    field_name: &str,
) {
    use arael_sym::{symbol, cached};
    for (row, r) in composed.rows.iter().enumerate() {
        for (col_name, col_expr) in [("x", &r.x), ("y", &r.y), ("z", &r.z)] {
            let to_sym = symbol(&format!("{}.{}.rotation_matrix[{}].{}", var_base, field_name, row, col_name));
            subs.push((cached(col_expr.clone()), to_sym));
        }
    }
}

/// Push `cached(d(entry)/d(param.k)) -> <field>.rotation_matrix_deriv[k][row].col`
/// substitutions for a rotation matrix's Jacobian. cached() is a STICKY barrier
/// under differentiation (d(cached(g))/dx = cached(dg/dx)), so the constraint
/// Jacobian carries these cached derivative units; they resolve to the per-pose
/// precomputed `rotation_matrix_deriv` field -- computed once per pose in
/// `__precompute` instead of re-derived at every observation. Shared by all
/// three rotation-param builders; `dvar_prefix` is the base of the parameter to
/// differentiate by (`<field>.delta` for the delta-based params, `<field>.work()`
/// for the direct euler param), each with `.x/.y/.z` appended.
fn push_rotation_deriv_subs(
    subs: &mut Vec<(arael_sym::E, arael_sym::E)>,
    composed: &matrix3sym,
    dvar_prefix: &str,
    var_base: &str,
    field_name: &str,
) {
    use arael_sym::{symbol, cached};
    for (k, dcomp) in ["x", "y", "z"].iter().enumerate() {
        let dvar = format!("{}.{}", dvar_prefix, dcomp);
        for (row, r) in composed.rows.iter().enumerate() {
            for (col_name, col_expr) in [("x", &r.x), ("y", &r.y), ("z", &r.z)] {
                let deriv = col_expr.diff(dvar.as_str());
                let to_sym = symbol(&format!("{}.{}.rotation_matrix_deriv[{}][{}].{}",
                    var_base, field_name, k, row, col_name));
                subs.push((cached(deriv), to_sym));
            }
        }
    }
}

/// Substitutions for a QuaternionParam (rotation-vector delta) field: replace
/// the composed rotation R_ref * retraction(delta) entries with reads of the
/// precomputed `<field>.rotation_matrix`. The retraction is rational (no
/// transcendentals), so there is no sincos analog to precompute. Each `from`
/// is wrapped in `cached()` to match the cached entries the binding emits (see
/// [`cache_rotation_entries`]).
fn build_universal_rotvec_substitutions(var_base: &str, field_name: &str) -> Vec<(arael_sym::E, arael_sym::E)> {
    let mut subs = Vec::new();
    let r_ref_sym = matrix3sym::new(&format!("{}.{}.ref_rotation", var_base, field_name));
    let dea_sym = vect3sym::new(&format!("{}.{}.delta", var_base, field_name));
    let composed = r_ref_sym * matrix3sym::from_rotation_vector_small(&dea_sym);
    push_rotation_matrix_subs(&mut subs, &composed, var_base, field_name);
    let dvar_prefix = format!("{}.{}.delta", var_base, field_name);
    push_rotation_deriv_subs(&mut subs, &composed, &dvar_prefix, var_base, field_name);
    subs
}

/// Apply substitutions to a list of expressions. Returns the modified expressions.
fn apply_substitutions(exprs: &mut Vec<arael_sym::E>, subs: &[(arael_sym::E, arael_sym::E)]) {
    // Every target is a cached()/symbol node matched by exact structural
    // equality; replace_many resolves them all in one memoized pass, applying
    // subs in list order (first mapping wins on a duplicate target).
    for e in exprs.iter_mut() {
        *e = arael_sym::cse::replace_many(e, subs);
    }
}

/// `#[arael(root, fast_atan)]`: route every atan/atan2 in the generated
/// code through `arael_sym::fast_atan` / `fast_atan2` (max error < 1e-6
/// radians). Applied after differentiation, so derivatives stay the
/// exact rational forms; only the emitted call targets change.
fn replace_atan_fast(exprs: &mut Vec<arael_sym::E>) {
    for e in exprs.iter_mut() {
        *e = e
            .replace_function("atan", &|args| arael_sym::fast_atan(args[0].clone()))
            .replace_function("atan2", &|args| arael_sym::fast_atan2(args[0].clone(), args[1].clone()));
    }
}

/// Recursively register sym bindings for a variable and all its nested struct fields.
/// `key_prefix` is used for binding lookup (e.g. "pose.info.gps")
/// `sym_prefix` is used for generated code (e.g. "pose.info.gps.as_ref().unwrap()")
/// One optimizable run of a type's flat layout, in serialize (field
/// declaration) order: a direct Param field, or -- through a
/// `#[arael(component)]` struct field -- a nested one. `path` is dotted
/// relative to the type ("w", "dir.d").
#[derive(Clone)]
struct ParamSlot {
    path: String,
    sft: SymFieldType,
    /// EulerAngleParam / QuaternionParam: symbol is `.delta`, not `.work()`.
    universal_delta: bool,
}

fn param_slot_size(sft: &SymFieldType) -> usize {
    match sft {
        SymFieldType::Scalar => 1,
        SymFieldType::Vec2 => 2,
        SymFieldType::Vec3 => 3,
        _ => 0,
    }
}

fn collect_param_slots(type_name: &str, prefix: &str, out: &mut Vec<ParamSlot>) {
    let Some(layout) = registry_lookup(type_name) else { return };
    for (fname, sft) in &layout.fields {
        let path = if prefix.is_empty() { fname.clone() } else { format!("{}.{}", prefix, fname) };
        if layout.param_fields.contains(fname) {
            if param_slot_size(sft) > 0 {
                out.push(ParamSlot {
                    path,
                    sft: sft.clone(),
                    universal_delta: layout.universal_euler_angle_fields.contains(fname)
                        || layout.universal_rotvec_fields.contains(fname),
                });
            }
        } else if let SymFieldType::Struct(inner) = sft
            && registry_lookup(inner).map(|l| l.component).unwrap_or(false)
        {
            collect_param_slots(inner, &path, out);
        }
    }
}

fn param_slots(type_name: &str) -> Vec<ParamSlot> {
    let mut v = Vec::new();
    collect_param_slots(type_name, "", &mut v);
    v
}

fn param_total(type_name: &str) -> usize {
    param_slots(type_name).iter().map(|s| param_slot_size(&s.sft)).sum()
}

/// `base.dir.d` field-access tokens for a dotted slot path.
fn slot_access(base: TokenStream2, path: &str) -> TokenStream2 {
    let mut t = base;
    for seg in path.split('.') {
        let id = syn::Ident::new(seg, proc_macro2::Span::call_site());
        t = quote! { #t.#id };
    }
    t
}

fn register_bindings_recursive(
    ctx: &mut ConstraintCtx,
    key_prefix: &str,
    sym_prefix: &str,
    type_name: &str,
) -> syn::Result<()> {
    let mut stack = Vec::new();
    register_bindings_guarded(ctx, key_prefix, sym_prefix, type_name, &mut stack)
}

/// The recursive body of [`register_bindings_recursive`], with a
/// type-name path stack. Binding registration follows Ref fields on
/// purpose (a body reads `tie.a.v` through the ref), so a containment +
/// ref cycle (A holds B, B refs A) would recurse forever without the
/// guard -- it used to overflow rustc's stack (SIGSEGV, no diagnostic).
/// A type already on the CURRENT path stops: bindings deeper than one
/// cycle turn are unreachable in any finite body, while the same type
/// on parallel branches (a diamond) still registers on each.
fn register_bindings_guarded(
    ctx: &mut ConstraintCtx,
    key_prefix: &str,
    sym_prefix: &str,
    type_name: &str,
    stack: &mut Vec<String>,
) -> syn::Result<()> {
    if stack.iter().any(|s| s == type_name) {
        return Ok(());
    }
    stack.push(type_name.to_string());
    let result = register_bindings_body(ctx, key_prefix, sym_prefix, type_name, stack);
    stack.pop();
    result
}

fn register_bindings_body(
    ctx: &mut ConstraintCtx,
    key_prefix: &str,
    sym_prefix: &str,
    type_name: &str,
    stack: &mut Vec<String>,
) -> syn::Result<()> {
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
                    register_bindings_guarded(ctx, &nested_key, &nested_sym, inner_type, stack)?;
                }
                SymFieldType::OptionalStruct(inner_type) => {
                    // CONTRACT (documented in MODEL.md, "Guards and
                    // optional data"): a body reading through an Option
                    // sub-struct must be guarded so it never evaluates
                    // when the field is None -- the read is this unwrap.
                    // A guard suppresses the body on every path (cost,
                    // grad/Hessian, jacobian), so the unwrap is safe
                    // under it; an unguarded read panics on the first
                    // None at solve time.
                    let nested_key = format!("{}.{}", key_prefix, field_name);
                    let nested_sym = format!("{}.{}.as_ref().unwrap()", sym_prefix, field_name);
                    register_bindings_guarded(ctx, &nested_key, &nested_sym, inner_type, stack)?;
                }
                _ => {
                    let is_universal_ea = is_param
                        && layout.universal_euler_angle_fields.contains(field_name);
                    let is_universal_rotvec = is_param
                        && layout.universal_rotvec_fields.contains(field_name);
                    if is_universal_ea || is_universal_rotvec {
                        // Build composed rotation R_ref * R(delta) symbolically.
                        // EulerAngleParam maps the delta through the euler
                        // rotation; QuaternionParam through the so(3) exp map.
                        let r_ref_sym = matrix3sym::new(
                            &format!("{}.{}.ref_rotation", sym_prefix, field_name));
                        let dea_sym = vect3sym::new(
                            &format!("{}.{}.delta", sym_prefix, field_name));
                        let delta_rot = if is_universal_rotvec {
                            // Sqrt-free rotation matrix of the retraction normalize(1, delta/2).
                            matrix3sym::from_rotation_vector_small(&dea_sym)
                        } else {
                            dea_sym.rotation_matrix()
                        };
                        let composed_rot = r_ref_sym * delta_rot;
                        // The composed entries get reshaped by simplification and
                        // mixed into the residual; wrap each in cached() (a
                        // barrier) so the precompute substitution matches them
                        // reliably -- both the euler-angle and rotvec deltas (see
                        // cache_rotation_entries).
                        let composed_rot = cache_rotation_entries(&composed_rot);
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

        // Second pass: `#[arael(symbolic = <expr>)]` fields. Each expression
        // is evaluated over the struct's OWN fields (bare names; params read
        // as param symbols, data as constants, `<param>_value` as constants),
        // and the result REPLACES the field's plain-constant binding -- body
        // reads of the field then carry the expression's derivatives.
        // Declaration order: a later symbolic field sees the earlier ones.
        if !layout.symbolic_fields.is_empty() {
            let mut scratch = ConstraintCtx::new();
            for (field_name, sft) in &layout.fields {
                if matches!(sft,
                    SymFieldType::Skip
                    | SymFieldType::Struct(_)
                    | SymFieldType::OptionalStruct(_)) { continue; }
                let is_param = layout.param_fields.contains(field_name);
                let sym_base = if is_param {
                    format!("{}.{}.work()", sym_prefix, field_name)
                } else {
                    format!("{}.{}", sym_prefix, field_name)
                };
                scratch.bindings.insert(field_name.clone(),
                    ConstraintCtx::make_sym_val(&sym_base, sft));
                if is_param {
                    scratch.bindings.insert(format!("{}_value", field_name),
                        ConstraintCtx::make_sym_val(
                            &format!("{}.{}.value", sym_prefix, field_name), sft));
                }
            }
            for (fname, expr_str) in &layout.symbolic_fields {
                let parsed: Expr = syn::parse_str(expr_str).map_err(|e| {
                    syn::Error::new(proc_macro2::Span::call_site(),
                        format!("symbolic = expression on `{}.{}` does not parse: {}",
                            type_name, fname, e))
                })?;
                let val = eval_expr(&parsed, &mut scratch).map_err(|e| {
                    syn::Error::new(e.span(),
                        format!("symbolic = expression on `{}.{}`: {}",
                            type_name, fname, e))
                })?;
                // Wrap each component in cached() -- a sticky substitution
                // barrier -- so bodies and sibling expressions carry
                // matchable units. After differentiation the value units
                // resolve to reads of the precomputed field, and (for
                // declared `deriv =` caches) the derivative units to reads
                // of the cache array -- the rotation-matrix pattern.
                let val = if let Some(comps) = symval_components(&val) {
                    let mut wrapped = std::vec::Vec::new();
                    for (suffix, e) in &comps {
                        let ce = arael_sym::cached(e.clone());
                        ctx.subs.push((ce.clone(), arael_sym::symbol(
                            &format!("{}.{}{}", sym_prefix, fname, suffix))));
                        wrapped.push(ce);
                    }
                    for (dfield, of, by) in &layout.deriv_fields {
                        if of != fname { continue; }
                        let by_sft = layout.fields.iter()
                            .find(|(n, _)| n == by).map(|(_, t)| t);
                        let comps_of = |names: &[&str]| -> std::vec::Vec<String> {
                            names.iter().map(|c|
                                format!("{}.{}.work().{}", sym_prefix, by, c)).collect()
                        };
                        let dvars: std::vec::Vec<String> = match by_sft {
                            Some(SymFieldType::Scalar) =>
                                vec![format!("{}.{}.work()", sym_prefix, by)],
                            Some(SymFieldType::Vec2) => comps_of(&["x", "y"]),
                            Some(SymFieldType::Vec3) => comps_of(&["x", "y", "z"]),
                            _ => return Err(syn::Error::new(
                                proc_macro2::Span::call_site(),
                                format!("`{}.{}`: deriv `by = {}` must name a \
                                         scalar, vec2 or vec3 param field",
                                    type_name, dfield, by))),
                        };
                        for (k, dvar) in dvars.iter().enumerate() {
                            for (suffix, e) in &comps {
                                ctx.subs.push((
                                    arael_sym::cached(e.diff(dvar.as_str())),
                                    arael_sym::symbol(&format!("{}.{}[{}]{}",
                                        sym_prefix, dfield, k, suffix))));
                            }
                        }
                    }
                    symval_from_components(&val, wrapped)
                } else {
                    val
                };
                scratch.bindings.insert(fname.clone(), val.clone());
                ctx.bindings.insert(format!("{}.{}", key_prefix, fname), val);
            }
        }
    }
    Ok(())
}

/// Walk a dotted field path through the registered layouts. Paths through
/// `#[arael(skip)]` fields or types without a registered layout are opaque
/// and allowed (they emit verbatim field access); a segment that names no
/// field on a registered type is a typo -- error with the closest match.
fn validate_entity_path(type_name: &str, var_head: &str, rest: &str, span: &Expr) -> syn::Result<()> {
    let mut cur_type = type_name.to_string();
    let mut walked = var_head.to_string();
    let mut segs = rest.split('.').peekable();
    while let Some(seg) = segs.next() {
        let Some(layout) = registry_lookup(&cur_type) else {
            return Ok(()); // unregistered type: opaque, trust the user
        };
        // Param `.value` aliases register as `<field>_value` bindings; if
        // one reaches this fallback it is either a typo or a suffix on a
        // param that resolves fine -- treat the bare field name.
        let field = layout.fields.iter().find(|(n, _)| n == seg);
        let Some((_, _)) = field else {
            let mut pool: Vec<String> = layout.fields.iter().map(|(n, _)| n.clone()).collect();
            for pf in &layout.param_fields {
                pool.push(format!("{}_value", pf));
            }
            let suggestion = pool.iter()
                .map(|c| (edit_distance(seg, c), c.as_str()))
                .filter(|(d, _)| *d <= 2)
                .min_by_key(|(d, c)| (*d, c.to_string()))
                .map(|(_, c)| format!(" (did you mean `{}`?)", c))
                .unwrap_or_default();
            return Err(syn::Error::new_spanned(span,
                format!("no field `{}` on `{}` (in `{}.{}`){}",
                    seg, cur_type, walked, seg, suggestion)));
        };
        walked = format!("{}.{}", walked, seg);
        match &field.unwrap().1 {
            SymFieldType::Struct(inner) | SymFieldType::OptionalStruct(inner) => {
                cur_type = inner.clone();
            }
            SymFieldType::Skip => return Ok(()), // opaque from here on
            _ => {
                // Leaf field with segments left: pre-registration and the
                // component arms already cover every valid access, so
                // whatever remains is not a component of this field.
                if let Some(extra) = segs.next() {
                    return Err(syn::Error::new_spanned(span,
                        format!("`{}` has no component `{}` (in `{}`)", walked, extra, walked)));
                }
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Replace every `self` path in a guard expression with the given
/// identifier, on the AST (no string surgery).
/// Rename every occurrence of ident `from` to `to` in a token stream,
/// recursing into groups. Used to rewrite generated constraint code from
/// body-binding names (entity var, root var) to direct loop-item / `self`
/// access: reads become disjoint field projections through the loop's
/// `&mut` item or `self`, so the emission needs no whole-struct alias
/// binding (the old `&*(self as *const Self)` laundering was undefined
/// behavior under Stacked/Tree Borrows).
/// Renames only identifiers in VARIABLE position. An identifier directly
/// after `.` (a field or method) or after `:` (a path segment) names
/// something inside another value and has nothing to do with the binding
/// being renamed -- a model field named like the root type's lowercase
/// (`m2` under root `M2`) or like the constraint struct's own lowercase
/// otherwise turned into `self` / `__item` mid-path.
fn rename_ident(ts: TokenStream2, from: &str, to: &str) -> TokenStream2 {
    let mut out = TokenStream2::new();
    let mut after_selector = false;
    for tt in ts {
        let selector = matches!(&tt,
            proc_macro2::TokenTree::Punct(p) if p.as_char() == '.' || p.as_char() == ':');
        let renamable = !after_selector;
        after_selector = selector;
        out.extend(std::iter::once(match tt {
            proc_macro2::TokenTree::Ident(id) if renamable && id == from => {
                proc_macro2::TokenTree::Ident(proc_macro2::Ident::new(to, id.span()))
            }
            proc_macro2::TokenTree::Group(g) => {
                // A group opens a fresh expression: its first token is never
                // in selector position.
                let mut ng = proc_macro2::Group::new(g.delimiter(), rename_ident(g.stream(), from, to));
                ng.set_span(g.span());
                proc_macro2::TokenTree::Group(ng)
            }
            other => other,
        }));
    }
    out
}

fn rewrite_guard_self(e: &mut syn::Expr, replacement: &str) {
    use syn::visit_mut::VisitMut;
    struct R<'a>(&'a str);
    impl<'a> VisitMut for R<'a> {
        fn visit_expr_mut(&mut self, node: &mut syn::Expr) {
            if let syn::Expr::Path(p) = node
                && p.qself.is_none()
                && p.path.is_ident("self") {
                    let ident = syn::Ident::new(self.0, proc_macro2::Span::call_site());
                    *node = syn::parse_quote!(#ident);
                    return;
                }
            syn::visit_mut::visit_expr_mut(self, node);
        }
    }
    R(replacement).visit_expr_mut(e);
}

/// Rewrite every `self` path to `replacement` across a constraint BODY, so
/// `self.x` names the constraint's own entity (its lowercased struct name),
/// exactly as `self` does in a guard. Applied once before the body is
/// interpreted; downstream resolution then treats it like the ordinary
/// `<struct_lower>.x` form users already write.
fn rewrite_body_self(stmts: &mut [syn::Stmt], replacement: &str) {
    use syn::visit_mut::VisitMut;
    struct R<'a>(&'a str);
    impl<'a> VisitMut for R<'a> {
        fn visit_expr_mut(&mut self, node: &mut syn::Expr) {
            if let syn::Expr::Path(p) = node
                && p.qself.is_none()
                && p.path.is_ident("self") {
                    let ident = syn::Ident::new(self.0, proc_macro2::Span::call_site());
                    *node = syn::parse_quote!(#ident);
                    return;
                }
            syn::visit_mut::visit_expr_mut(self, node);
        }
    }
    let mut r = R(replacement);
    for stmt in stmts.iter_mut() {
        r.visit_stmt_mut(stmt);
    }
}

/// Levenshtein distance, for typo suggestions.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for i in 1..=a.len() {
        let mut cur = vec![i; b.len() + 1];
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        prev = cur;
    }
    prev[b.len()]
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
                if (seg.ident == "SelfBlock" || seg.ident == "BoxedSelfBlock") && !type_args.is_empty() {
                    let a = type_ident_name(type_args[0])?;
                    return Ok((a, None));
                }
                if (seg.ident == "CrossBlock" || seg.ident == "BoxedCrossBlock") && type_args.len() >= 2 {
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

/// Expression of type `Option<&Collection>` resolving a ref field's
/// target collection, relative to `self` (the root) and the current
/// entity `__e`. `root.`-rooted paths descend plain fields; a path
/// whose head is another ref field of the same entity chains through
/// that ref with `get()`. Returns `None` for shapes the walker does not
/// handle (`parent.`-scoped refs in nested sub-models).
fn ref_target_expr(
    elem_ref_paths: &[(String, String)],
    path: &str,
) -> Option<TokenStream2> {
    let (head, rest) = path.split_once('.')?;
    let segs: Vec<syn::Ident> = rest
        .split('.')
        .map(|s| syn::Ident::new(s, proc_macro2::Span::call_site()))
        .collect();
    if head == "root" {
        return Some(quote! { Some(&self.#(#segs).*) });
    }
    let head_path = &elem_ref_paths.iter().find(|(f, _)| f == head)?.1;
    let head_expr = ref_target_expr(elem_ref_paths, head_path)?;
    let head_ident = syn::Ident::new(head, proc_macro2::Span::call_site());
    Some(quote! {
        #head_expr.and_then(|__c| __c.get(__e.#head_ident)).map(|__p| &__p.#(#segs).*)
    })
}

/// Stale-ref checks for one entity bound as `__e`: one `if` per ref
/// field whose target the walker can resolve. `path_fmt` / `path_args`
/// format the reported location (e.g. `"edges[{}].{}"` with the loop
/// index).
fn ref_checks_for_entity(
    ref_paths: &[(String, String)],
    path_fmt: &str,
    path_args: &TokenStream2,
) -> Vec<TokenStream2> {
    let mut checks = Vec::new();
    for (ref_field, path) in ref_paths {
        let Some(target) = ref_target_expr(ref_paths, path) else { continue };
        let ref_ident = syn::Ident::new(ref_field, proc_macro2::Span::call_site());
        let ref_name = ref_field.clone();
        checks.push(quote! {
            if #target.map_or(true, |__c| __c.get(__e.#ref_ident).is_none()) {
                __issues.push(arael::validate::Issue::StaleRef {
                    path: format!(#path_fmt, #path_args #ref_name),
                });
            }
        });
    }
    checks
}

/// The `RootProblem::collect_ref_issues` override: walk the root's own
/// ref fields, its direct struct fields, and every root-level
/// collection whose element type carries `#[arael(ref = ...)]` fields,
/// reporting each ref that no longer resolves. `parent.`-scoped refs
/// inside nested sub-models are not walked. Emits nothing when there is
/// nothing to check (the trait default applies).
fn generate_ref_issue_walker(root_name: &str) -> TokenStream2 {
    let Some(root_layout) = registry_lookup(root_name) else {
        return quote! {};
    };
    let mut body: Vec<TokenStream2> = Vec::new();
    // The root's own ref fields.
    {
        let checks = ref_checks_for_entity(&root_layout.ref_paths, "{}", &quote! {});
        if !checks.is_empty() {
            body.push(quote! {
                {
                    let __e = &*self;
                    #(#checks)*
                }
            });
        }
    }
    for (field, sft) in &root_layout.fields {
        let SymFieldType::Struct(elem) = sft else { continue };
        // A ref field of the root itself is checked above, not descended.
        if root_layout.ref_paths.iter().any(|(f, _)| f == field) {
            continue;
        }
        let Some(elem_layout) = registry_lookup(elem) else { continue };
        if elem_layout.ref_paths.is_empty() {
            continue;
        }
        let field_ident = syn::Ident::new(field, proc_macro2::Span::call_site());
        if root_layout.collection_fields.contains(field) {
            let fmt = format!("{}[{{}}].{{}}", field);
            let checks = ref_checks_for_entity(&elem_layout.ref_paths, &fmt, &quote! { __i, });
            if !checks.is_empty() {
                body.push(quote! {
                    for (__i, __e) in self.#field_ident.iter().enumerate() {
                        #(#checks)*
                    }
                });
            }
        } else {
            let fmt = format!("{}.{{}}", field);
            let checks = ref_checks_for_entity(&elem_layout.ref_paths, &fmt, &quote! {});
            if !checks.is_empty() {
                body.push(quote! {
                    {
                        let __e = &self.#field_ident;
                        #(#checks)*
                    }
                });
            }
        }
    }
    if body.is_empty() {
        return quote! {};
    }
    quote! {
        fn collect_ref_issues(&self, __issues: &mut std::vec::Vec<arael::validate::Issue>) {
            #(#body)*
        }
    }
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
    fast_atan: bool,
    marginalize_hint_fn: &Option<TokenStream2>,
    marginalize_candidates_fn: &Option<TokenStream2>,
    has_triplet_block: bool,
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
        let _ = &root_param_fields;
        if let Some(hb) = root_hb_field.as_ref().filter(|_| param_total(&root_name.to_string()) > 0) {
            let hb_ident = syn::Ident::new(hb, proc_macro2::Span::call_site());
            let layout = root_layout.as_ref().unwrap();
            let mut count: usize = 0;
            let mut idx_stmts: Vec<TokenStream2> = Vec::new();
            let _ = layout;
            for slot in param_slots(&root_name.to_string()) {
                let size = param_slot_size(&slot.sft);
                let offset = count;
                let end = offset + size;
                let access = slot_access(quote! { self }, &slot.path);
                idx_stmts.push(quote! {
                    #access.write_indices(&mut __root_self_idx[#offset..#end]);
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
        // Outer hops from root to the constraint collection `rc_ident`. Empty
        // for a constraint struct directly on the root (PosePair on root);
        // [paths] when the constraint lives in root.paths[k].<rc_ident>.
        prefix: Vec<AccessSegment>,
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
    // All four emission group maps are BTreeMaps: their iteration order
    // drives code emission (loop order, __cid assignment, floating-point
    // accumulation order). HashMap's per-instance random state made every
    // rustc invocation emit differently -- non-reproducible builds and
    // last-ulp numeric drift between recompiles.
    let mut cross_groups: std::collections::BTreeMap<String, CrossCollectionGroup> = std::collections::BTreeMap::new();

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
        entity_index_copies: Vec<TokenStream2>,
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
    let mut triplet_groups: std::collections::BTreeMap<String, TripletCollectionGroup> = std::collections::BTreeMap::new();

    // Grouping for constraints that iterate the same collection.
    // Merges SelfBlock + nested CrossBlock into a single loop per collection.
    struct CollectionGroup {
        coll_ident: syn::Ident,
        // Outer hops from root down to `coll_ident`'s container. Empty for a
        // collection directly on the root; e.g. [paths] when the entity lives
        // in root.paths[k].<coll_ident>. The emitter wraps the per-entity loop
        // in one loop per prefix segment.
        prefix: Vec<AccessSegment>,
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
    let mut collection_groups: std::collections::BTreeMap<String, CollectionGroup> = std::collections::BTreeMap::new();

    // Grouping for SelfBlock constraints that live on a single-instance entity
    // (the root itself, or a direct-composed sub-model field). Keyed by the
    // access path ("self" for RootSelf, "self.<field>" for DirectField).
    // Multiple #[arael(constraint(...))] attributes on the same entity merge
    // into one emitted block per path.
    struct SingleInstanceGroup {
        accessor_read: TokenStream2,
        accessor_write: TokenStream2,
        /// An Option<Entity> location: the accessors yield Option refs and
        /// every emitted block is wrapped in `if let Some(__item) = ...`.
        optional: bool,
        self_var: syn::Ident,
        root_var_ident: syn::Ident,
        a_param_count: usize,
        a_idx_stmts: Vec<TokenStream2>,
        block_ident: syn::Ident,
        constraint_index_field: Option<syn::Ident>,
        cost_entries: Vec<TokenStream2>,
        gh_entries: Vec<TokenStream2>,
        jac_entries: Vec<TokenStream2>,
    }
    let mut single_instance_groups: std::collections::BTreeMap<String, SingleInstanceGroup> = std::collections::BTreeMap::new();

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
                    // OptionalStruct too: a type reachable only through an
                    // Option<T> field used to be invisible here, silently
                    // dropping its constraints.
                    if let SymFieldType::Struct(s) | SymFieldType::OptionalStruct(s) = sft {
                        queue.push(s.clone());
                    }
                }
            }
        }
        set
    };

    // Ordering guard: every collection element type in the root's fields
    // must have a registered layout by the time the root expands. Macro
    // expansion is file-order top-down, so a #[arael::model] struct
    // defined AFTER the root (or in another crate: the registry is
    // per-rustc-process) is invisible -- its constraints would be
    // silently dropped. Collection elements must implement Model, so a
    // missing layout here is always an error, never an external type.
    {
        let root_fields_ordered: syn::FieldsNamed = syn::parse2(quote! { { #root_fields } })?;
        // The registry and the emitted access paths key entities by BARE
        // type name, so one root cannot hold two instantiations of the
        // same generic entity (`Vec<Pose<f32>>` next to `Vec<Pose<f64>>`)
        // -- nor two spellings of one instantiation. name -> full spelling.
        let mut spellings: std::collections::HashMap<String, String> =
            std::collections::HashMap::new();
        let mut check_spelling = |field: &syn::Field, name: &str, ty: &syn::Type|
            -> syn::Result<()> {
            let spelled = quote! { #ty }.to_string()
                .replace(" < ", "<").replace(" > ", ">").replace(" >", ">");
            match spellings.get(name) {
                Some(prev) if *prev != spelled => Err(syn::Error::new_spanned(field,
                    format!("root holds `{}` both as `{}` and `{}` -- entities are \
                             resolved by bare type name, so a root must spell every \
                             use of `{}` identically (one instantiation per root)",
                            name, prev, spelled, name))),
                _ => {
                    spellings.insert(name.to_string(), spelled);
                    Ok(())
                }
            }
        };
        for field in &root_fields_ordered.named {
            if field_is_skipped(field) { continue; }
            if let syn::Type::Path(tp) = &field.ty
                && let Some(seg) = tp.path.segments.last() {
                    if matches!(seg.ident.to_string().as_str(), "Vec" | "Deque" | "Arena")
                        && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                        && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
                        && let Ok(name) = type_ident_name(inner) {
                            if registry_lookup(&name).is_none() {
                                if let Some(reason) = crate::registry_excluded_reason(&name) {
                                    return Err(syn::Error::new_spanned(field,
                                        format!("collection element type `{}` was not exported by its \
                                                 defining crate: {}", name, reason)));
                                }
                                return Err(syn::Error::new_spanned(field,
                                    format!("collection element type `{}` has no registered #[arael::model] \
                                             layout: define it (or import its crate's arael_import! bundle) \
                                             BEFORE the root struct -- macro expansion is top-down file \
                                             order within one crate", name)));
                            }
                            check_spelling(field, &name, inner)?;
                    } else {
                        // Direct struct-typed field of a registered entity.
                        let name = seg.ident.to_string();
                        if registry_lookup(&name).is_some() {
                            check_spelling(field, &name, &field.ty)?;
                        }
                    }
                }
        }
    }

    // Block precision must match the root's solve precision -- storage /
    // Param precision is free (the walks cast at the boundary). Checked
    // across every reachable type so the mismatch names the struct and
    // block field instead of surfacing as an E0308 inside generated code.
    // A "generic" block precision resolves per instantiation: the holder's
    // recorded element spelling (`Vec<G<f32>>`) carries it.
    {
        let mut sorted: Vec<&String> = reachable.iter().collect();
        sorted.sort();
        for tn in sorted {
            let Some(layout) = registry_lookup(tn) else { continue };
            if let Some((bfield, bp)) = &layout.block_precision
                && bp != "generic" && bp != precision {
                    return Err(syn::Error::new(root_name.span(), format!(
                        "`{}` declares block `{}` at {}, but root `{}` solves at {} -- \
                         block precision must match the root; storage/Param precision \
                         may differ (casts happen at the boundary)",
                        tn, bfield, bp, root_name, precision)));
                }
            for (hfield, elem, fl) in &layout.inst_precisions {
                if fl != precision && reachable.contains(elem)
                    && registry_lookup(elem).and_then(|l| l.block_precision)
                        .is_some_and(|(_, p)| p == "generic") {
                    return Err(syn::Error::new(root_name.span(), format!(
                        "`{}.{}` instantiates generic model `{}` at {}, but root `{}` \
                         solves at {} -- the instantiation sets the block precision, \
                         which must match the root",
                        tn, hfield, elem, fl, root_name, precision)));
                }
            }
        }
    }

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
        let mut constraint = match constraint { Some(c) => c, None => continue };
        // In a constraint body `self` names the constraint itself, matching the
        // guard. Rewrite `self.x` -> `<struct_lower>.x` (the constraint's own
        // entity) so the existing name-based resolution handles it uniformly.
        rewrite_body_self(&mut constraint.body_stmts, &sc.struct_name.to_lowercase());

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
        //
        // `root.<field>` as the PRIMARY block: the constraint writes the
        // ROOT's own SelfBlock<Self> -- the "shared parameter set, many
        // observations" shape, where the entity supplies only data and every
        // param lives on the root. Validated here, before the dotted-path
        // check treats `root` as a Ref field name.
        let root_self_primary: Option<syn::Ident> = if let Some(rest) =
            constraint.primary_block_field().strip_prefix("root.")
        {
            let rest = rest.to_string();
            let root_field = root_fields.iter().find(|f|
                f.ident.as_ref().map(|i| i.to_string()) == Some(rest.clone()))
                .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(),
                    format!("{}:{}: constraint names `root.{}` but root `{}` has no field `{}`",
                        sc.attr_file, sc.attr_line, rest, root_name, rest)))?;
            let seg_name = if let syn::Type::Path(tp) = &root_field.ty {
                tp.path.segments.last().map(|s| s.ident.to_string()).unwrap_or_default()
            } else { String::new() };
            match seg_name.as_str() {
                "SelfBlock" => {
                    let (a, b) = extract_block_type_args(&root_field.ty)?;
                    if a != root_name.to_string() || b.is_some() {
                        return Err(syn::Error::new(proc_macro2::Span::call_site(),
                            format!("{}:{}: `root.{}` must be the root's `SelfBlock<Self>`, \
                                     found `SelfBlock<{}>`",
                                sc.attr_file, sc.attr_line, rest, a)));
                    }
                }
                "TripletBlock" => {
                    return Err(syn::Error::new(proc_macro2::Span::call_site(),
                        format!("{}:{}: `root.{}` is a TripletBlock -- a root-owned \
                                 TripletBlock is a secondary block: use \
                                 `constraint([<local_self_block>, root.{}], ...)`",
                            sc.attr_file, sc.attr_line, rest, rest)));
                }
                _ => {
                    return Err(syn::Error::new(proc_macro2::Span::call_site(),
                        format!("{}:{}: a `root.<field>` primary block must name the \
                                 root's `SelfBlock<Self>` field; `root.{}` is not one",
                            sc.attr_file, sc.attr_line, rest)));
                }
            }
            // The constraint may touch ONLY root params: the entity's own
            // params would form (entity, root) cross pairs this block cannot
            // hold, and dropping them is never acceptable.
            let entity_has_params = registry_lookup(&sc.struct_name)
                .map(|l| !l.param_fields.is_empty()).unwrap_or(false);
            if entity_has_params {
                return Err(syn::Error::new(proc_macro2::Span::call_site(),
                    format!("{}:{}: `{}` has its own Param fields, so a `root.{}` \
                             constraint would drop the (entity, root) cross pairs -- \
                             declare a `SelfBlock<{}>` on `{}` and route through a \
                             root-owned TripletBlock: \
                             `constraint([<self_block>, root.<triplet>], ...)`",
                        sc.attr_file, sc.attr_line, sc.struct_name, rest,
                        sc.struct_name, sc.struct_name)));
            }
            Some(syn::Ident::new(&rest, proc_macro2::Span::call_site()))
        } else { None };

        // `parent.<selfblock>` primary: the data-only entity is contained
        // in a parameter-bearing entity (any depth below the root) and
        // writes into THAT entity's SelfBlock -- the non-root analog of
        // `root.<selfblock>`. (field ident, parent type name).
        let parent_self_primary: Option<(syn::Ident, String)> = if let Some(rest) =
            constraint.primary_block_field().strip_prefix("parent.")
        {
            let parent_type = find_containing_parent(&root_name.to_string(), &sc.struct_name)
                .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(),
                    format!("{}:{}: `parent.{}`: no registered struct contains `{}`",
                        sc.attr_file, sc.attr_line, rest, sc.struct_name)))?;
            if parent_type == root_name.to_string() {
                return Err(syn::Error::new(proc_macro2::Span::call_site(),
                    format!("{}:{}: `{}`'s containing parent is the root -- use \
                             `constraint(root.{}, ...)`",
                        sc.attr_file, sc.attr_line, sc.struct_name, rest)));
            }
            let playout = registry_lookup(&parent_type)
                .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(),
                    format!("{}:{}: parent type `{}` not in registry",
                        sc.attr_file, sc.attr_line, parent_type)))?;
            if playout.self_block_field.as_deref() != Some(rest) {
                return Err(syn::Error::new(proc_macro2::Span::call_site(),
                    format!("{}:{}: `parent.{}` does not name the `SelfBlock<Self>` field                              of the containing parent `{}` (which is {})",
                        sc.attr_file, sc.attr_line, rest, parent_type,
                        playout.self_block_field.as_deref()
                            .map(|f| format!("`{}`", f))
                            .unwrap_or_else(|| "missing".to_string()))));
            }
            // The constraint may touch ONLY parent params: the entity's own
            // params would form (entity, parent) cross pairs this block
            // cannot hold, and dropping them is never acceptable.
            let entity_has_params = registry_lookup(&sc.struct_name)
                .map(|l| !l.param_fields.is_empty()).unwrap_or(false);
            if entity_has_params {
                return Err(syn::Error::new(proc_macro2::Span::call_site(),
                    format!("{}:{}: `{}` has its own Param fields, so a `parent.{}`                              constraint would drop the (entity, parent) cross pairs --                              declare a `SelfBlock<{}>` on `{}` and couple through a                              CrossBlock/TripletBlock instead",
                        sc.attr_file, sc.attr_line, sc.struct_name, rest,
                        sc.struct_name, sc.struct_name)));
            }
            Some((syn::Ident::new(rest, proc_macro2::Span::call_site()), parent_type))
        } else { None };

        // Check if block_field is a dotted path (remote block, e.g. pose.hb_pose)
        let is_remote_block = constraint.primary_block_field().contains('.')
            && root_self_primary.is_none() && parent_self_primary.is_none();

        let (a_type, b_type, remote_block_info) = if root_self_primary.is_some()
            || parent_self_primary.is_some() {
            // Iterate the entity's containment; every param is the root's
            // (or the containing parent's), so there is no local block and
            // no remote Ref target.
            (sc.struct_name.clone(), None, None)
        } else if is_remote_block {
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

            // Find the parent struct that contains this constraint struct.
            // The CURRENT root wins when it holds the collection directly
            // (root-level observations): the global scan is alphabetical
            // and another root holding the same collection would hijack
            // the resolution.
            let parent_type = find_containing_parent(&root_name.to_string(), &sc.struct_name)
                .ok_or_else(|| syn::Error::new(proc_macro2::Span::call_site(),
                    format!("cannot find parent struct containing {}", sc.struct_name)))?;

            (parent_type, None, Some((ref_field_name.to_string(), target_block_field.to_string(), target_type)))
        } else {
            // Local block field on this struct
            let block_field_obj = fields.named.iter().find(|f|
                f.ident.as_ref().map(|i| i.to_string()) == Some(constraint.primary_block_field().to_string())
            );
            let Some(_) = block_field_obj else {
                return Err(syn::Error::new(proc_macro2::Span::call_site(),
                    format!("{}:{}: constraint names block field `{}` but `{}` has no such \
                             field -- the constraint would be silently dropped",
                        sc.attr_file, sc.attr_line,
                        constraint.primary_block_field(), sc.struct_name)));
            };
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
        // Duplicate containment guard, over EVERY containment path (one-hop
        // and nested alike). SelfBlock sweeps handle a type held in several
        // collections -- one sweep per path, below. Everything else drives
        // its iteration from a SINGLE resolved location and would silently
        // skip the rest: a self-block type mixed with a single-instance
        // holding, a cross/triplet CONSTRAINT struct under two paths, or a
        // frines-style constraint under a duplicated parent. Reject those
        // loudly.
        let containing_paths = containment_paths(root_fields, &root_name.to_string(), coll_type);
        let reject: Option<(&str, Vec<Vec<AccessSegment>>)> = if is_self_block {
            let all_collections = containing_paths.iter()
                .all(|p| p.last().is_some_and(|s| s.collection));
            (containing_paths.len() > 1 && !all_collections)
                .then(|| (coll_type.as_str(), containing_paths.clone()))
        } else {
            let struct_paths = containment_paths(root_fields, &root_name.to_string(), &sc.struct_name);
            if struct_paths.len() > 1 {
                Some((sc.struct_name.as_str(), struct_paths))
            } else if struct_paths.is_empty() && containing_paths.len() > 1 {
                // Constraint struct not on the root: iteration goes through
                // its parent (the A-type), which must be unique.
                Some((coll_type.as_str(), containing_paths.clone()))
            } else {
                None
            }
        };
        // Aliasing guard: a constraint entity contained under the SAME
        // root collection one of its refs targets (A holds B, B refs
        // root.aa) would make the sweep iterate `self.aa` mutably while
        // writing ref-target blocks into `self.aa` -- unsound, and the
        // generated code fails with a bare E0502 pointing at the macro.
        // (This containment+ref cycle used to overflow rustc during
        // binding registration before the cycle guard there.) Reject it
        // with the shape named. `parent.`-scoped refs are fine: they
        // resolve into sibling fields of the iterated entity.
        if !is_self_block
            && let Some(layout) = registry_lookup(&sc.struct_name) {
                let struct_paths = containment_paths(root_fields, &root_name.to_string(), &sc.struct_name);
                for (rf, rpath) in &layout.ref_paths {
                    let Some(rest) = rpath.strip_prefix("root.") else { continue };
                    let target_root_field = rest.split('.').next().unwrap_or(rest);
                    for path in &struct_paths {
                        if path.first().is_some_and(|s| s.field == target_root_field) {
                            return Err(syn::Error::new(proc_macro2::Span::call_site(),
                                format!("{}:{}: `{}` is contained under `{}` (via {}) and its \
                                         ref field `{}` targets that same collection -- the \
                                         sweep would iterate `{}` mutably while writing into \
                                         it. Hold `{}` outside the referenced collection \
                                         (e.g. a root-level collection)",
                                    sc.attr_file, sc.attr_line, sc.struct_name,
                                    target_root_field, path_display(path), rf,
                                    target_root_field, sc.struct_name)));
                        }
                    }
                }
            }
        if let Some((dup_type, paths)) = reject {
            let names: Vec<String> = paths.iter().map(|p| path_display(p)).collect();
            return Err(syn::Error::new(proc_macro2::Span::call_site(),
                format!("`{}` is contained in multiple locations ({}); its \
                    constraints would only be evaluated for `{}`. Multiple \
                    containment locations are supported only for \
                    SelfBlock-constrained entities in collections -- wrap \
                    the others in distinct types",
                    dup_type, names.join(", "), names[0])));
        }
        if !is_self_block
            && !(is_remote_block
                && matches!(entity_location, EntityLocation::RootSelf))
            // An Option-held triplet/multi-cross constraint struct is fine:
            // Option iterates as a zero-or-one collection. (For a plain
            // cross, entity_location is the TARGET type's location, so
            // Optional there means an Option-held parent entity -- an
            // unsupported iteration shape, rejected below.)
            && !(matches!(entity_location, EntityLocation::OptionalField { .. })
                && (is_triplet || is_multi_cross))
            && !matches!(entity_location,
                EntityLocation::Collection { .. } | EntityLocation::Nested { .. }) {
            // TripletBlock / CrossBlock constraints drive their iteration
            // from the constraint struct's containing collection, so a
            // single-instance location (direct field, Option field, the
            // root itself without a remote block) has no loop to emit.
            // Rejected loudly -- a silent skip here would drop the
            // constraint. (A remote-block constraint whose parent IS the
            // root passes above: its sweep is a single loop over that
            // collection; a Nested A-type is fine, cross emission wraps
            // its own prefix loops.)
            return Err(syn::Error::new(proc_macro2::Span::call_site(),
                format!("{}:{}: cross/triplet constraint on `{}` needs the \
                         constraint struct to live in a collection \
                         (Vec/Deque/Arena) on the root; it is a single \
                         instance here, so its sweep has no loop to iterate \
                         -- put it in a collection (a collection of one is \
                         fine)",
                    sc.attr_file, sc.attr_line, sc.struct_name)));
        }
        // `coll_ident(_str)` is only consumed by the Collection SelfBlock path and the
        // nested CrossBlock path (both of which require a Collection). For DirectField
        // / RootSelf we divert below and these placeholders are never read.
        // (SelfBlock sweeps take their per-path idents/prefixes from
        // `containing_paths`, not from here.)
        let (coll_ident_str, coll_ident) = match &entity_location {
            EntityLocation::Collection { field, .. } => {
                (field.clone(), syn::Ident::new(field, proc_macro2::Span::call_site()))
            }
            EntityLocation::Nested { segments } => {
                // Last segment holds the entity; earlier segments are the loops
                // wrapped around it.
                let last = segments.last().expect("Nested location has >= 1 segment");
                (last.field.clone(), syn::Ident::new(&last.field, proc_macro2::Span::call_site()))
            }
            EntityLocation::DirectField { field }
            | EntityLocation::OptionalField { field } => {
                (String::new(), syn::Ident::new(field, proc_macro2::Span::call_site()))
            }
            EntityLocation::RootSelf => {
                (String::new(), root_name.clone())
            }
        };
        // CrossBlock/remote: find frines field and build ref resolution
        let mut frines_ident = None;
        let mut resolve_stmts = Vec::new();
        // Same bindings with #[allow(unused_variables)], re-emitted after
        // each residual row's block writes: the writes take temporary
        // `&mut` into the resolved collections, ending the row's shared
        // borrows, so the next row re-establishes them (cheap address
        // math; measured at parity with the old aliased-pointer code).
        let mut resolve_reread_stmts = Vec::new();
        let mut entity_index_copies: Vec<TokenStream2> = Vec::new();
        // field name -> "self."-rooted index path (e.g. "self.poses"),
        // used to build the WRITE access path for an entity's blocks:
        // `self.poses[__frine.pose].hb_pose` -- a temporary exclusive
        // borrow, taken per call, so aliased entities are sound.
        let mut ref_index_paths: std::collections::HashMap<String, String> = std::collections::HashMap::new();
        let mut parent_ident = None;
        let mut is_root_level_cross = false;  // constraint struct lives on root (possibly nested)
        // Outer hops to the constraint collection when it lives in a sub-model
        // collection (empty when the constraint struct is directly on root).
        let mut cross_prefix: Vec<AccessSegment> = Vec::new();

        if is_triplet || is_multi_cross || (!is_self_block || is_remote_block) {
            // First try: constraint struct nested under A-type (e.g. PointFrine
            // under PointLandmark). An `Option<Frine>` field works the same --
            // Option iterates as a zero-or-one collection, so the emitted
            // loops need no special casing.
            let parent_layout = registry_lookup(&a_type);
            let frines_field = parent_layout.as_ref().and_then(|l| {
                l.fields.iter().find(|(_, sft)| {
                    matches!(sft,
                        SymFieldType::Struct(s) | SymFieldType::OptionalStruct(s)
                            if s == &sc.struct_name)
                }).map(|(name, _)| name.clone())
            });

            if let Some(ff) = frines_field {
                // Nested case (e.g. PointFrine under PointLandmark)
                frines_ident = Some(syn::Ident::new(&ff, proc_macro2::Span::call_site()));
                parent_ident = Some(syn::Ident::new(&parent_name, proc_macro2::Span::call_site()));
            } else {
                // Constraint struct in a root collection (PosePair directly on
                // root) or nested in a sub-model collection (PosePair in
                // root.paths[k].pose_pairs). resolve_entity_location finds both.
                let (rc_name, prefix) = match find_root_collection(root_fields, &sc.struct_name) {
                    Some(name) => (name, Vec::new()),
                    None => match resolve_entity_location(root_fields, &root_name.to_string(), &sc.struct_name) {
                        Some(EntityLocation::Nested { segments }) => {
                            let last = segments.last().expect("Nested has >= 1 segment");
                            (last.field.clone(), segments[..segments.len() - 1].to_vec())
                        }
                        // An Option<Frine> iterates as a zero-or-one
                        // collection (Option's iter/iter_mut), so the
                        // emitted loops handle it unchanged.
                        Some(EntityLocation::OptionalField { field }) => {
                            (field.clone(), Vec::new())
                        }
                        // A plain direct field has no iteration at all:
                        // skipping would silently drop the constraint.
                        // (A `None` resolution is different -- the type is
                        // not contained in this root at all, e.g. another
                        // root's constraint struct reachable through a Ref
                        // edge -- and stays skipped.)
                        Some(EntityLocation::DirectField { field }) => {
                            return Err(syn::Error::new(proc_macro2::Span::call_site(),
                                format!("{}:{}: cross/triplet constraint struct `{}` is held \
                                         as a plain single-instance field (root field `{}`); \
                                         its sweep needs something to iterate -- hold it in \
                                         a Vec/Deque/Arena or an Option",
                                    sc.attr_file, sc.attr_line, sc.struct_name, field)));
                        }
                        _ => continue,
                    },
                };
                frines_ident = Some(syn::Ident::new(&rc_name, proc_macro2::Span::call_site()));
                cross_prefix = prefix;
                is_root_level_cross = true;
            }

            let struct_layout = registry_lookup(&sc.struct_name);
            let ref_paths = struct_layout.as_ref().map(|l| l.ref_paths.clone()).unwrap_or_default();
            let mut seen_ref_fields: std::collections::HashSet<String> = std::collections::HashSet::new();
            for (field_name, resolve_path) in &ref_paths {
                let field_ident_inner = syn::Ident::new(field_name, proc_macro2::Span::call_site());
                // `root.<coll>` -> `self.<coll>`. `parent.<coll>` -> the
                // containing sub-model instance (`__seg{n-1}`) for a nested
                // constraint; only reached when cross_prefix is non-empty.
                let adjusted_path = resolve_path.replace("root.", "self.");
                let adjusted_path = if cross_prefix.is_empty() {
                    adjusted_path
                } else {
                    adjusted_path.replace("parent.",
                        &format!("__seg{}.", cross_prefix.len() - 1))
                };
                let resolve_expr: syn::Expr = syn::parse_str(
                    &format!("{}[__frine.{}]", adjusted_path, field_name)
                ).map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                    format!("failed to parse resolve path: {}", e)))?;
                resolve_stmts.push(quote! { let #field_ident_inner = &#resolve_expr; });
                // Copy the Ref index to a local ONCE per constraint
                // instance: rereads and block writes then index through the
                // local, so the optimizer never has to re-load the Ref
                // through the collection borrow after a write (measurable
                // reload cost otherwise).
                if seen_ref_fields.insert(field_name.clone()) {
                    let ei_ident = syn::Ident::new(&format!("__ei_{}", field_name), proc_macro2::Span::call_site());
                    entity_index_copies.push(quote! {
                        #[allow(unused_variables)]
                        let #ei_ident = __frine.#field_ident_inner;
                    });
                }
                let reread_expr: syn::Expr = syn::parse_str(
                    &format!("{}[__ei_{}]", adjusted_path, field_name)
                ).map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                    format!("failed to parse resolve path: {}", e)))?;
                resolve_reread_stmts.push(quote! {
                    #[allow(unused_variables)]
                    let #field_ident_inner = &#reread_expr;
                });
                ref_index_paths.insert(field_name.clone(), adjusted_path);
            }
        }

        // Build a `self.`-rooted mutable access expression for an entity's
        // field: `self.poses[__frine.pose]`. Resolve paths may chain through
        // other refs ("pose.info.features"); substitute recursively until
        // the path is `self.`-rooted.
        let entity_access_expr = |field_name: &str| -> syn::Result<syn::Expr> {
            let mut path = match ref_index_paths.get(field_name) {
                Some(p) => format!("{}[__ei_{}]", p, field_name),
                None => return Err(syn::Error::new(proc_macro2::Span::call_site(),
                    format!("no resolve path for entity ref `{}`", field_name))),
            };
            for _ in 0..8 {
                // `self.` (root) and `__seg{i}.` (a nested sub-model loop
                // variable, in scope in the emitted loop) are both already
                // rooted at a valid mutable place.
                if path.starts_with("self.") || path.starts_with("__seg") { break; }
                let head = path.split('.').next().unwrap_or("").to_string();
                let sub = ref_index_paths.get(&head)
                    .map(|p| format!("{}[__ei_{}]", p, head));
                match sub {
                    Some(s) => path = format!("{}{}", s, &path[head.len()..]),
                    None => return Err(syn::Error::new(proc_macro2::Span::call_site(),
                        format!("cannot root entity access path `{}` at self", path))),
                }
            }
            syn::parse_str(&path).map_err(|e| syn::Error::new(proc_macro2::Span::call_site(),
                format!("failed to parse entity access path `{}`: {}", path, e)))
        };

        // Re-process constraint body to get residual-only code and full code
        // We need the symbolic expressions again
        let root_name_str = root_name.to_string();
        let (residual_exprs, param_symbols, loss_expr, component_subs) = interpret_constraint_body(
            &struct_ident, &fields.named, &constraint, &root_name_str)
            .map_err(|e| syn::Error::new(e.span(),
                format!("{}:{}: {}", sc.attr_file, sc.attr_line, e)))?;
        check_residual_coverage(sc, &struct_ident, &residual_exprs, &param_symbols)?;

        // Apply euler_angles substitutions from all referenced types
        let mut all_subs: Vec<(arael_sym::E, arael_sym::E)> = Vec::new();
        // Symbolic component-field units (values + declared deriv caches),
        // collected while the bindings were registered.
        all_subs.extend(component_subs);
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
                        for rv in &layout.universal_rotvec_fields {
                            all_subs.extend(build_universal_rotvec_substitutions(field_name, rv));
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
            for rv in &a_layout.universal_rotvec_fields {
                all_subs.extend(build_universal_rotvec_substitutions(&self_var_name, rv));
            }
        }
        // For SelfBlock, also check the struct itself (it IS the A type)
        if is_self_block
            && let Some(self_layout) = registry_lookup(&sc.struct_name) {
                for ea in &self_layout.euler_angle_fields {
                    all_subs.extend(build_euler_substitutions(&self_var_name, ea));
                }
            }

        let block_ident = if let Some(root_hb) = &root_self_primary {
            // The root's SelfBlock field. Only ever emitted through
            // `self.<field>` (the entity has no block of its own).
            root_hb.clone()
        } else if let Some((parent_hb, _)) = &parent_self_primary {
            // The parent's SelfBlock field, emitted through the prefix
            // binding (the entity has no block of its own).
            parent_hb.clone()
        } else if is_remote_block {
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
        // A `parent.<selfblock>` constraint entity must live in a
        // collection (or Option) INSIDE its parent, so the sweep has the
        // parent instance in scope as a prefix binding.
        let parent_prefix: Option<Vec<AccessSegment>> = if parent_self_primary.is_some() {
            match &entity_location {
                EntityLocation::Nested { segments }
                    if segments.last().is_some_and(|l| l.collection || l.optional)
                        && segments.len() >= 2 =>
                {
                    Some(segments[..segments.len() - 1].to_vec())
                }
                _ => {
                    return Err(syn::Error::new(proc_macro2::Span::call_site(),
                        format!("{}:{}: a `parent.<selfblock>` constraint must live in a                                  collection (Vec/Deque/Arena) or an Option INSIDE its                                  parent entity",
                            sc.attr_file, sc.attr_line)));
                }
            }
        } else { None };
        // The joined co-entity of a self-primary form: the root
        // (`root.<selfblock>` / `[hb, root.<triplet>]`, accessed as
        // `self`) or the containing parent (`parent.<selfblock>`,
        // accessed as the innermost prefix binding).
        let (joined_accessor, joined_type, joined_var): (TokenStream2, String, String) =
            if let Some((_, ptype)) = &parent_self_primary {
                let pvar = constraint.parent_name.clone()
                    .unwrap_or_else(|| ptype.to_lowercase());
                (nested_container(parent_prefix.as_deref().unwrap()), ptype.clone(), pvar)
            } else {
                (quote! { self }, root_type_str.clone(), root_type_str.to_lowercase())
            };
        // The co-entity joins the entity list for every self-primary
        // form; either span may be absent (a param-less entity pushes no
        // self entry, and root./parent.<selfblock> never has one).
        let is_root_joined = is_root_triplet_self || root_self_primary.is_some()
            || parent_self_primary.is_some();
        if is_root_joined {
            // Entities are [self, joined] in that order. Self is accessed
            // via `__item` (iter_mut item on the struct's collection);
            // the joined co-entity through its accessor (`self` for the
            // root, the prefix binding for a parent).
            // param_total walks #[arael(component)] fields, so component
            // params count into the spans like direct ones.
            let self_count = param_total(&sc.struct_name);
            if self_count > 0 {
                triplet_entities.push((
                    syn::Ident::new("__item", proc_macro2::Span::call_site()),
                    syn::Ident::new(&sc.struct_name, proc_macro2::Span::call_site()),
                    0, self_count,
                ));
            }
            let joined_count = param_total(&joined_type);
            if joined_count > 0 {
                triplet_entities.push((
                    syn::Ident::new(&joined_var, proc_macro2::Span::call_site()),
                    syn::Ident::new(&joined_type, proc_macro2::Span::call_site()),
                    self_count, joined_count,
                ));
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
                        if registry_lookup(&type_name).is_some() {
                            let var_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
                            let type_ident = syn::Ident::new(&type_name, proc_macro2::Span::call_site());
                            let entity_start = offset;
                            offset += param_total(&type_name);
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
            if has_root_entity && registry_lookup(&root_type_str).is_some() {
                let entity_start = offset;
                offset += param_total(&root_type_str);
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
        let a_param_count = param_total(&a_type);
        let b_param_count = b_type.as_ref().map(|b| param_total(b)).unwrap_or(0);

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

        // --- Robust loss setup ---
        // With a loss the block accumulates its squared residual norm into
        // __block_cost, then contributes rho(s) to __cost and scales every
        // Hessian/gradient write by the weight __w = rho'(s). Without a loss,
        // every token below is empty and the emission is byte-identical.
        let loss_present = loss_expr.is_some();
        let (m_add, m_cross): (TokenStream2, TokenStream2) = if loss_present {
            (quote! { add_residual_with_loss }, quote! { add_residual_cross_with_loss })
        } else {
            (quote! { add_residual }, quote! { add_residual_cross })
        };
        // Build the finalize statements (compute rho, and for the gh path the
        // weight) from the loss expression, sharing the residual pipeline:
        // differentiate for the weight, then substitute + fast_atan + CSE.
        // The block accumulator lives at RESIDUAL precision (inferred from
        // the rows), not the block's: loss expressions mix it with model
        // fields, and a model can store f32 fields while solving f64
        // (blocks f64). Casts to #cast_type happen at the boundaries
        // (__w, __cost) instead.
        let block_cost_decl: TokenStream2 = if loss_present {
            quote! { let mut __block_cost = 0.0; }
        } else { quote! {} };
        let emit_loss = |want_weight: bool| -> syn::Result<TokenStream2> {
            let Some(loss_e) = &loss_expr else { return Ok(quote! {}); };
            let mut exprs = vec![loss_e.clone()];
            if want_weight { exprs.push(loss_e.diff(LOSS_ARG_SYM)); }
            apply_substitutions(&mut exprs, &all_subs);
            if fast_atan { replace_atan_fast(&mut exprs); }
            let (ints, simplified) = arael_sym::cse(&exprs);
            let mut stmts = Vec::new();
            for (name, expr) in &ints {
                let ni = syn::Ident::new(name, proc_macro2::Span::call_site());
                let code: Expr = parse_sym_code(&expr.to_rust(""))?;
                stmts.push(quote! { let #ni = #code; });
            }
            let rho_code: Expr = parse_sym_code(&simplified[0].to_rust(""))?;
            let weight_stmt = if want_weight {
                let w_code: Expr = parse_sym_code(&simplified[1].to_rust(""))?;
                quote! { let __w = (#w_code) as #cast_type; }
            } else { quote! {} };
            Ok(quote! {
                #(#stmts)*
                #weight_stmt
                __cost += (#rho_code) as #cast_type;
            })
        };
        let loss_cost_finalize = emit_loss(false)?;
        let loss_gh_finalize = emit_loss(true)?;
        // Per-row cost accumulator: into __block_cost under a loss, else __cost.
        let cost_acc: TokenStream2 = if loss_present { quote! { __block_cost } } else { quote! { __cost } };
        // Row-squared term for the accumulator: __cost is #cast_type, so
        // rows cast; __block_cost stays at the rows' own precision.
        let row_sq = |r_ident: &syn::Ident| -> TokenStream2 {
            if loss_present {
                quote! { #r_ident * #r_ident }
            } else {
                quote! { (#r_ident as #cast_type) * (#r_ident as #cast_type) }
            }
        };

        // --- Cost-only code: differentiate FIRST, then apply substitutions, then CSE ---
        // Apply substitutions to residuals (cost-only, no derivatives)
        let mut cost_exprs = residual_exprs.clone();
        apply_substitutions(&mut cost_exprs, &all_subs);
        if fast_atan { replace_atan_fast(&mut cost_exprs); }
        let (cost_intermediates, cost_simplified) = arael_sym::cse(&cost_exprs);
        let mut cost_stmts = Vec::new();
        cost_stmts.push(block_cost_decl.clone());
        for (name, expr) in &cost_intermediates {
            let name_ident = syn::Ident::new(name, proc_macro2::Span::call_site());
            let code: Expr = parse_sym_code(&expr.to_rust(""))?;
            cost_stmts.push(quote! { let #name_ident= #code; });
        }
        for (ri, r) in cost_simplified.iter().enumerate() {
            let r_ident = syn::Ident::new(&format!("__r_{}", ri), proc_macro2::Span::call_site());
            let r_expr: Expr = parse_sym_code(&r.to_rust(""))?;
            let sq = row_sq(&r_ident);
            cost_stmts.push(quote! {
                let #r_ident= #r_expr;
                #cost_acc += #sq;
            });
        }
        cost_stmts.push(loss_cost_finalize);

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
        if fast_atan { replace_atan_fast(&mut all_gh_exprs); }
        let (gh_intermediates, gh_simplified) = arael_sym::cse(&all_gh_exprs);

        let mut gh_stmts = Vec::new();
        gh_stmts.push(block_cost_decl.clone());

        // Cross-block write targets: mutable access paths taken fresh at
        // every add_residual call (a temporary exclusive borrow, ending at
        // the statement). Aliased A/B slots (both refs resolving to one
        // entity) are sound automatically: the two writes are sequential
        // borrows of the same place, never simultaneous.
        let is_cross_block = !is_self_block && !is_triplet && !is_remote_block && !is_multi_cross;
        let (a_write_target, b_write_target): (Option<TokenStream2>, Option<TokenStream2>) = if is_cross_block {
            let b_type_name = b_type.as_ref().expect("cross block requires B");
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
            let a_target: TokenStream2 = if is_root_level_cross {
                // A is a Ref entity: index the owning collection directly.
                let access = entity_access_expr(&a_var_id.to_string())?;
                quote! { #access.#a_hb_ident }
            } else {
                // Nested cross: A is the parent (`__item`, the outer loop's
                // &mut item); its block is a field disjoint from the frines
                // collection the inner loop iterates.
                quote! { __item.#a_hb_ident }
            };
            let b_access = entity_access_expr(&b_var_id.to_string())?;
            let b_target: TokenStream2 = quote! { #b_access.#b_hb_ident };
            (Some(a_target), Some(b_target))
        } else { (None, None) };

        // Remote (dotted-path) block write target: the block lives on a
        // Ref-resolved entity (e.g. `pose.hb_pose`); write through its
        // owning collection slot, fresh per call.
        let remote_target_write: Option<TokenStream2> = if is_remote_block {
            let (ref_field_name, _, _) = remote_block_info.as_ref().unwrap();
            let access = entity_access_expr(ref_field_name)?;
            Some(quote! { #access.#block_ident })
        } else { None };

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
            // param_slots walks #[arael(component)] fields, so component
            // params are wired into the span like direct ones.
            let _ = (&self_layout, &root_layout);
            let mut self_count = 0usize;
            let mut self_idx_stmts: Vec<TokenStream2> = Vec::new();
            for slot in param_slots(&sc.struct_name) {
                let size = param_slot_size(&slot.sft);
                if size == 0 { continue; }
                let offset = self_count;
                let end = offset + size;
                let access = slot_access(quote! { __item }, &slot.path);
                self_idx_stmts.push(quote! {
                    #access.write_indices(&mut __all_idx[#offset..#end]);
                });
                self_count += size;
            }
            let mut root_count = 0usize;
            let mut root_idx_stmts: Vec<TokenStream2> = Vec::new();
            for slot in param_slots(&root_type_str) {
                let size = param_slot_size(&slot.sft);
                if size == 0 { continue; }
                let offset = self_count + root_count;
                let end = offset + size;
                let access = slot_access(quote! { self }, &slot.path);
                root_idx_stmts.push(quote! {
                    #access.write_indices(&mut __all_idx[#offset..#end]);
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
        // Remote-block constraints write only through the target path (plus
        // per-frine cross blocks in multi-cross mode): emit ALL rows'
        // computes first and defer the writes to the end, in original
        // order. Reads are then never interrupted, so no reread bindings
        // are needed (the old code cached one &mut across rows; per-row
        // re-resolution measurably regressed the band bench). Measured
        // remote-only: extending deferral to the other families regressed
        // the sparse bench, so they keep the interleaved compute/write
        // shape with per-row rereads.
        // A loss forces deferral: every write is scaled by __w = rho'(s),
        // which is only known once __block_cost has summed all rows.
        let defer_writes = is_remote_block || loss_present;
        let mut deferred_writes: Vec<TokenStream2> = Vec::new();
        for ri in 0..n_residuals {
            // Residual rows interleave reads (residual + derivative
            // evaluation through shared borrows) and writes (temporary
            // exclusive borrows into the resolved collections). The writes
            // end the previous row's shared borrows, so every row after the
            // first re-establishes the ref bindings.
            if ri > 0 && !resolve_reread_stmts.is_empty() && !defer_writes {
                gh_stmts.push(quote! { #(#resolve_reread_stmts)* });
            }
            let r_ident = syn::Ident::new(&format!("__r_{}", ri), proc_macro2::Span::call_site());
            let r_expr: Expr = parse_sym_code(&gh_simplified[idx].to_rust(""))?;
            // Accumulate the cost alongside the derivatives: the residual
            // value is already in hand, so the fused calc_cost_grad_hessian_*
            // entry points get the cost for free (saves a separate cost-only
            // model evaluation in the LM loop). Under a loss this sums into
            // __block_cost = |r|^2 instead, and rho(s) is added to __cost once.
            let sq = row_sq(&r_ident);
            gh_stmts.push(quote! {
                let #r_ident= #r_expr;
                #cost_acc += #sq;
            });
            idx += 1;
            // The leading argument every accumulation call passes: the residual
            // cast to the block type, prefixed by the weight when a loss is on.
            let wr: TokenStream2 = if loss_present {
                quote! { __w, #r_ident as #cast_type }
            } else {
                quote! { #r_ident as #cast_type }
            };

            // Structurally-zero derivatives (residual does not touch the
            // parameter) are known post-simplify: skip their declarations,
            // inline 0.0 literals where a full-width slice is still needed,
            // and elide whole block calls when an entity's span is all
            // zeros -- the accumulated contribution would be exactly 0.
            let mut dr_zero: Vec<bool> = Vec::with_capacity(n_params);
            let mut dr_f64: Vec<TokenStream2> = Vec::with_capacity(n_params);
            for pi in 0..n_params {
                let zero = gh_simplified[idx].is_zero();
                dr_zero.push(zero);
                if zero {
                    dr_f64.push(quote! { 0.0 as #cast_type });
                } else {
                    let dr_ident = syn::Ident::new(&format!("__dr_{}_{}", ri, pi), proc_macro2::Span::call_site());
                    let dr_expr: Expr = parse_sym_code(&gh_simplified[idx].to_rust(""))?;
                    gh_stmts.push(quote! { let #dr_ident= #dr_expr; });
                    dr_f64.push(quote! { #dr_ident as #cast_type });
                }
                idx += 1;
            }
            let span_zero = |start: usize, count: usize| -> bool {
                count == 0 || dr_zero[start..start + count].iter().all(|&z| z)
            };
            let all_zero = span_zero(0, n_params);
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
                    if span_zero(*start, *count) { continue; }
                    let entity_dr: Vec<TokenStream2> = dr_f64.iter().skip(*start).take(*count).cloned().collect();
                    // Write through a fresh temporary exclusive borrow of the
                    // entity's owning collection slot.
                    let access = entity_access_expr(&var_id.to_string())?;
                    let _ = type_id;
                    triplet_calls.push(quote! {
                        #access.#hb_ident
                            .#m_add(#wr, &[#(#entity_dr),*], grad);
                    });
                }
                // Cross pairs need two live spans: with <= 1 nonzero span
                // every cross product is structurally zero.
                let nonzero_spans = triplet_entities.iter()
                    .filter(|(_, _, start, count)| !span_zero(*start, *count))
                    .count();
                let cross_call = if nonzero_spans <= 1 { quote! {} } else { quote! {
                    __frine.#block_ident.#m_cross(
                        #wr,
                        &__all_idx,
                        &[#(#dr_f64),*],
                        &__entity_offsets,
                    );
                }};
                let writes = quote! {
                    #(#triplet_calls)*
                    #cross_call
                };
                if defer_writes { deferred_writes.push(writes); } else { gh_stmts.push(writes); }
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
                    if span_zero(*start, *count) { continue; }
                    let entity_dr: Vec<TokenStream2> = dr_f64.iter().skip(*start).take(*count).cloned().collect();
                    let type_id_str = type_id.to_string();
                    if type_id_str == root_ident_str {
                        // Root: its block is a root field, disjoint from the
                        // iterated collection -- write directly through self.
                        let hb = registry_lookup(&type_id_str)
                            .and_then(|l| l.self_block_field.clone())
                            .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                                format!("root type `{}` must declare a `SelfBlock<Self>` field (required as implicit multi-cross participant)", type_id)))?;
                        let hb_ident = syn::Ident::new(&hb, proc_macro2::Span::call_site());
                        self_block_calls.push(quote! {
                            self.#hb_ident
                                .#m_add(#wr, &[#(#entity_dr),*], grad);
                        });
                        continue;
                    }
                    if remote_target_type.as_deref() == Some(type_id_str.as_str()) {
                        // Remote primary: write through the remote target path.
                        let rtw = remote_target_write.as_ref().unwrap();
                        remote_self_block_call = Some(quote! {
                            #rtw.#m_add(#wr, &[#(#entity_dr),*], grad);
                        });
                        continue;
                    }
                    let hb = registry_lookup(&type_id_str)
                        .and_then(|l| l.self_block_field.clone())
                        .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                            format!("type `{}` must declare a `SelfBlock<Self>` field (required as multi-cross participant)", type_id)))?;
                    let hb_ident = syn::Ident::new(&hb, proc_macro2::Span::call_site());
                    let access = entity_access_expr(&var_id.to_string())?;
                    self_block_calls.push(quote! {
                        #access.#hb_ident
                            .#m_add(#wr, &[#(#entity_dr),*], grad);
                    });
                }
                let mut cross_block_calls: Vec<TokenStream2> = Vec::new();
                for route in &multi_cross_routing {
                    if span_zero(route.a_start, route.a_count)
                        || span_zero(route.b_start, route.b_count) { continue; }
                    let block = &route.block_ident;
                    let dr_a: Vec<TokenStream2> = dr_f64.iter()
                        .skip(route.a_start).take(route.a_count).cloned().collect();
                    let dr_b: Vec<TokenStream2> = dr_f64.iter()
                        .skip(route.b_start).take(route.b_count).cloned().collect();
                    cross_block_calls.push(quote! {
                        __frine.#block.#m_cross(
                            #wr,
                            &[#(#dr_a),*],
                            &[#(#dr_b),*],
                        );
                    });
                }
                let writes = quote! {
                    #(#self_block_calls)*
                    #remote_self_block_call
                    #(#cross_block_calls)*
                };
                if defer_writes { deferred_writes.push(writes); } else { gh_stmts.push(writes); }
            } else if is_remote_block {
                if !all_zero {
                    let rtw = remote_target_write.as_ref().unwrap();
                    deferred_writes.push(quote! {
                        #rtw.#m_add(#wr, &[#(#dr_f64),*], grad);
                    });
                }
            } else if is_self_block {
                if is_root_joined {
                    // Root-coupled self-primary: [hb, root.hbt] or a
                    // `root.<selfblock>` primary. dr_f64 is
                    // [dr_self..., dr_root...], and either span may be
                    // absent (a param-less entity has no self entry;
                    // `root.<selfblock>` never has one). Up to three
                    // writes preserve every J^T J pair:
                    //   1. __item.<hb_self>.add_residual  -- (self, self)
                    //      diagonal + grad.
                    //   2. self.<hb_root>.add_residual -- (root, root)
                    //      diagonal + grad (root fields are disjoint from
                    //      the iterated collection).
                    //   3. self.<hbt>.add_residual_cross -- the (self,
                    //      root) across-entity block, COO storage; only
                    //      when a triplet is declared and both spans live.
                    let self_entry = triplet_entities.iter()
                        .find(|(v, _, _, _)| *v == "__item").cloned();
                    let root_entry = triplet_entities.iter()
                        .find(|(v, _, _, _)| *v != "__item").cloned();
                    let joined_hb = registry_lookup(&joined_type)
                        .and_then(|l| l.self_block_field.clone())
                        .ok_or_else(|| syn::Error::new_spanned(&struct_ident,
                            format!("`{}` must declare a `SelfBlock<Self>` field (required as coupled participant)", joined_type)))?;
                    if let Some(named) = &root_self_primary
                        && named.to_string() != joined_hb {
                        return Err(syn::Error::new_spanned(&struct_ident,
                            format!("`root.{}` does not name the root's `SelfBlock<Self>` \
                                     field (which is `{}`)", named, joined_hb)));
                    }
                    let joined_hb_ident = syn::Ident::new(&joined_hb, proc_macro2::Span::call_site());
                    let self_call = match &self_entry {
                        Some((_, _, s_start, s_count)) if !span_zero(*s_start, *s_count) => {
                            let dr_self: Vec<TokenStream2> = dr_f64.iter()
                                .skip(*s_start).take(*s_count).cloned().collect();
                            quote! { __item.#block_ident.#m_add(#wr, &[#(#dr_self),*], grad); }
                        }
                        _ => quote! {},
                    };
                    let root_call = match &root_entry {
                        Some((_, _, r_start, r_count)) if !span_zero(*r_start, *r_count) => {
                            let dr_root: Vec<TokenStream2> = dr_f64.iter()
                                .skip(*r_start).take(*r_count).cloned().collect();
                            quote! { #joined_accessor.#joined_hb_ident.#m_add(#wr, &[#(#dr_root),*], grad); }
                        }
                        _ => quote! {},
                    };
                    // The (self, root) cross pairs need a declared triplet
                    // and both spans live.
                    let cross_call = match (&root_triplet_field, &self_entry, &root_entry) {
                        (Some(triplet_ident), Some((_, _, ss, sc_)), Some((_, _, rs, rc)))
                            if !span_zero(*ss, *sc_) && !span_zero(*rs, *rc) => quote! {
                                self.#triplet_ident
                                    .#m_cross(
                                        #wr,
                                        &__all_idx,
                                        &[#(#dr_f64),*],
                                        &__entity_offsets,
                                    );
                            },
                        _ => quote! {},
                    };
                    let writes = quote! {
                        #self_call
                        #root_call
                        #cross_call
                    };
                    if defer_writes { deferred_writes.push(writes); } else { gh_stmts.push(writes); }
                } else if !all_zero {
                    let writes = quote! {
                        __item.#block_ident.#m_add(#wr, &[#(#dr_f64),*], grad);
                    };
                    if defer_writes { deferred_writes.push(writes); } else { gh_stmts.push(writes); }
                }
            } else {
                // CrossBlock: split dr into dr_a (first a_param_count) + dr_b
                // (next b_param_count). Three calls: A's SelfBlock gets
                // grad[A] + H[A,A] diagonal; B's SelfBlock same for B; the
                // cross block holds only the A-B rectangular cross Hessian.
                // Each write takes a fresh temporary exclusive borrow.
                let dr_a: Vec<TokenStream2> = dr_f64.iter().take(a_param_count).cloned().collect();
                let dr_b: Vec<TokenStream2> = dr_f64.iter().skip(a_param_count).take(b_param_count).cloned().collect();
                let a_zero = span_zero(0, a_param_count);
                let b_zero = span_zero(a_param_count, b_param_count);
                let a_target = a_write_target.as_ref().unwrap();
                let b_target = b_write_target.as_ref().unwrap();
                let a_call = if a_zero { quote! {} } else { quote! {
                    #a_target.#m_add(#wr, &[#(#dr_a),*], grad);
                }};
                let b_call = if b_zero { quote! {} } else { quote! {
                    #b_target.#m_add(#wr, &[#(#dr_b),*], grad);
                }};
                let cross_call = if a_zero || b_zero { quote! {} } else { quote! {
                    __frine.#block_ident.#m_cross(#wr, &[#(#dr_a),*], &[#(#dr_b),*]);
                }};
                let writes = quote! {
                    #a_call
                    #b_call
                    #cross_call
                };
                if defer_writes { deferred_writes.push(writes); } else { gh_stmts.push(writes); }
            }
        }

        // Finalize the loss before the deferred writes: __cost += rho(s) and
        // let __w = rho'(s), which every deferred write below scales by.
        gh_stmts.push(loss_gh_finalize);
        if !deferred_writes.is_empty() {
            gh_stmts.push(quote! { #(#deferred_writes)* });
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

        // Build index setup code — separate A (parent) and B (ref) indices.
        // The slot walk folds `#[arael(component)]` params in serialize order.
        let mut a_idx_stmts = Vec::new();
        let mut b_idx_stmts = Vec::new();
        {
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
            let mut offset = 0usize;
            for slot in param_slots(&a_type) {
                let size = param_slot_size(&slot.sft);
                let end = offset + size;
                let access = slot_access(quote! { #a_item }, &slot.path);
                a_idx_stmts.push(quote! {
                    #access.write_indices(&mut __a_idx[#offset..#end]);
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
                    let _ = &b_layout;
                    let mut offset = 0usize;
                    for slot in param_slots(b_type_name) {
                        let size = param_slot_size(&slot.sft);
                        let end = offset + size;
                        let access = slot_access(quote! { #b_var_ident }, &slot.path);
                        b_idx_stmts.push(quote! {
                            #access.write_indices(&mut __b_idx[#offset..#end]);
                        });
                        offset = end;
                    }
                }
            }

        let a_param_count = param_total(&a_type);
        let b_param_count = b_type.as_ref().map(|b| param_total(b)).unwrap_or(0);

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
                        if registry_lookup(&type_name).is_some() {
                            let var_ident = syn::Ident::new(field_name, proc_macro2::Span::call_site());
                            for slot in param_slots(&type_name) {
                                let size = param_slot_size(&slot.sft);
                                if size == 0 { continue; }
                                let offset = triplet_param_count;
                                let end = offset + size;
                                let access = slot_access(quote! { #var_ident }, &slot.path);
                                triplet_idx_stmts.push(quote! {
                                    #access.write_indices(&mut __all_idx[#offset..#end]);
                                });
                                triplet_param_count += size;
                            }
                        }
                    }
            }
            // Append root's write_indices calls when root is an implicit
            // entity. Accessed via `self.<param>` since root is the
            // enclosing `Self`.
            if has_root_entity && registry_lookup(&root_type_str).is_some() {
                for slot in param_slots(&root_type_str) {
                    let size = param_slot_size(&slot.sft);
                    if size == 0 { continue; }
                    let offset = triplet_param_count;
                    let end = offset + size;
                    let access = slot_access(quote! { self }, &slot.path);
                    triplet_idx_stmts.push(quote! {
                        #access.write_indices(&mut __all_idx[#offset..#end]);
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

        // Parse the guard and rewrite `self` on the AST -- string surgery
        // (`replacen("self.", ..)`) capped out at 10 occurrences, corrupted
        // identifiers merely containing "self.", and could not handle
        // block expressions.
        let guard_expr: Option<syn::Expr> = constraint.guard.as_ref()
            .map(|g| -> syn::Result<syn::Expr> {
                let mut e: syn::Expr = syn::parse_str(g).map_err(|err|
                    syn::Error::new(proc_macro2::Span::call_site(),
                        format!("failed to parse guard expression `{}`: {}", g, err)))?;
                let replacement = if is_self_block {
                    self_var_name.clone()
                } else {
                    // CrossBlock/TripletBlock: `self` is the constraint struct (__frine)
                    "__frine".to_string()
                };
                rewrite_guard_self(&mut e, &replacement);
                Ok(e)
            })
            .transpose()?;

        if is_remote_block {
            // Remote block: iterate parent collection -> frines,
            // but write to a block on the referenced struct (e.g. pose.hb_pose)
            let frines_ident = frines_ident.unwrap();
            let parent_ident = parent_ident.unwrap();
            // Parent IS the root: the constraint structs live in a
            // root-level collection, so the sweep is one loop directly
            // over it and the parent binding is the root itself.
            let parent_is_root = matches!(entity_location, EntityLocation::RootSelf);
            let parent_rename_to = if parent_is_root { "self" } else { "__lm" };
            let (ref_field_name, _, target_type) = remote_block_info.as_ref().unwrap();
            let ref_field_ident = syn::Ident::new(ref_field_name, proc_macro2::Span::call_site());
            let _target_type_ident = syn::Ident::new(target_type, proc_macro2::Span::call_site());

            // Find the root collection that contains the target type (for resolving the ref)
            let target_coll = find_root_collection(root_fields, target_type);
            let target_coll_ident = target_coll.map(|name|
                syn::Ident::new(&name, proc_macro2::Span::call_site()));

            // Index setup for the target type's params. `param_slots` walks
            // into `#[arael(component)]` fields, so a component's params sit
            // in the target's span exactly as they do for a self/cross block.
            let target_param_count = param_total(target_type);

            let mut target_idx_stmts = Vec::new();
            let mut offset = 0usize;
            for slot in param_slots(target_type) {
                let size = param_slot_size(&slot.sft);
                let end = offset + size;
                let access = slot_access(quote! { __target_ref }, &slot.path);
                target_idx_stmts.push(quote! {
                    #access.write_indices(&mut __a_idx[#offset..#end]);
                });
                offset = end;
            }

            let marker = source_marker(sc);

            // A guard (`self` already rewritten to `__frine`, the loop item)
            // wraps the residual statements so a filtered instance contributes
            // nothing to cost, gradient or Hessian.
            let remote_guarded_cost = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #(#cost_stmts)* } }
            } else {
                quote! { #(#cost_stmts)* }
            };
            let remote_guarded_gh = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #(#gh_stmts)* } }
            } else {
                quote! { { #(#gh_stmts)* } }
            };

            // Cost loop: iterate parent -> frines, resolve refs, evaluate
            if parent_is_root {
                cost_loops.push(quote! {
                    {
                        #marker
                        for __frine in self.#frines_ident.iter() {
                            #(#resolve_stmts)*
                            #[allow(unused_variables)]
                            let #parent_ident = &*__self_ref;
                            let #root_var_ident = &*__self_ref;
                            #remote_guarded_cost
                        }
                    }
                });
            } else {
                cost_loops.push(quote! {
                    {
                        #marker
                        for __lm in self.#coll_ident.iter() {
                            for __frine in __lm.#frines_ident.iter() {
                                #(#resolve_stmts)*
                                let #parent_ident = __lm;
                                let #root_var_ident = &*__self_ref;
                                #remote_guarded_cost
                            }
                        }
                    }
                });
            }

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
            // Remote-target writes go through the owning collection slot
            // (built into gh_stmts as `self.<coll>[__frine.<ref>].<block>`),
            // a temporary exclusive borrow per call -- disjoint from the
            // parent collection this loop iterates. Parent reads are
            // `__lm.*` field projections; root reads go through `self`.
            let gh_body = quote! {
                #(#entity_index_copies)*
                #(#resolve_reread_stmts)*
                #remote_guarded_gh
            };
            let gh_loop = match (parent_is_root, is_multi_cross) {
                (true, true) => quote! {
                    { #marker_gh for __frine in self.#frines_ident.iter_mut() { #gh_body } }
                },
                (true, false) => quote! {
                    { #marker_gh for __frine in self.#frines_ident.iter() { #gh_body } }
                },
                (false, true) => quote! {
                    {
                        #marker_gh
                        for __lm in self.#coll_ident.iter_mut() {
                            for __frine in __lm.#frines_ident.iter_mut() { #gh_body }
                        }
                    }
                },
                (false, false) => quote! {
                    {
                        #marker_gh
                        for __lm in self.#coll_ident.iter() {
                            for __frine in __lm.#frines_ident.iter() { #gh_body }
                        }
                    }
                },
            };
            let gh_loop = rename_ident(
                rename_ident(gh_loop, &parent_name, parent_rename_to), &root_var_name, "self");
            grad_hessian_loops.push(gh_loop);

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
                let rtw = remote_target_write.as_ref().unwrap();
                let sbi_body = quote! {
                    #(#entity_index_copies)*
                    #(#resolve_stmts)*
                    let __target_ref = #ref_field_ident;
                    let mut __a_idx = [0u32; #target_param_count];
                    #(#target_idx_stmts)*
                    let mut __all_idx = [0u32; #tp_remote];
                    #(#triplet_idx_stmts_remote)*
                    #rtw.set_indices(&__a_idx);
                    #(#entity_self_indices)*
                    #(#mcb_calls)*
                };
                let sbi_loop = if parent_is_root {
                    quote! { for __frine in self.#frines_ident.iter_mut() { #sbi_body } }
                } else {
                    quote! {
                        for __lm in self.#coll_ident.iter_mut() {
                            for __frine in __lm.#frines_ident.iter_mut() { #sbi_body }
                        }
                    }
                };
                let sbi_loop = rename_ident(
                    rename_ident(sbi_loop, &parent_name, parent_rename_to), &root_var_name, "self");
                set_block_indices_loops.push(sbi_loop);
            } else {
                let rtw = remote_target_write.as_ref().unwrap();
                let sbi_body = quote! {
                    #(#entity_index_copies)*
                    #(#resolve_stmts)*
                    let __target_ref = #ref_field_ident;
                    let mut __a_idx = [0u32; #target_param_count];
                    #(#target_idx_stmts)*
                    #rtw.set_indices(&__a_idx);
                };
                let sbi_loop = if parent_is_root {
                    quote! { for __frine in self.#frines_ident.iter() { #sbi_body } }
                } else {
                    quote! {
                        for __lm in self.#coll_ident.iter() {
                            for __frine in __lm.#frines_ident.iter() { #sbi_body }
                        }
                    }
                };
                let sbi_loop = rename_ident(
                    rename_ident(sbi_loop, &parent_name, parent_rename_to), &root_var_name, "self");
                set_block_indices_loops.push(sbi_loop);
            }
        } else if is_self_block {
            let self_var = syn::Ident::new(&self_var_name, proc_macro2::Span::call_site());
            let marker = source_marker(sc);

            let cost_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #marker #(#cost_stmts)* } }
            } else {
                quote! { { #marker #(#cost_stmts)* } }
            };
            // parent.<selfblock>: parent reads in the cost body resolve
            // through the innermost prefix binding (the cost loop binds
            // the self alias and root, but not the parent).
            let cost_entry = match parent_prefix.as_ref() {
                Some(p) => rename_ident(cost_entry, &joined_var,
                    &format!("__seg{}", p.len() - 1)),
                None => cost_entry,
            };

            // Grad+hessian entries access the entity through `__item` (the
            // loop's `&mut` item) directly: reads are field projections,
            // writes are temporary exclusive borrows of the block field --
            // no whole-struct alias binding needed. Root-field reads go
            // through `self` (disjoint from the iterated collection).
            let gh_entry = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #marker #(#gh_stmts)* } }
            } else {
                quote! { { #marker #(#gh_stmts)* } }
            };
            let gh_entry = {
                let renamed = rename_ident(gh_entry, &self_var_name, "__item");
                if root_var_name != self_var_name {
                    rename_ident(renamed, &root_var_name, "self")
                } else { renamed }
            };
            // parent.<selfblock>: the parent binding in body/write tokens
            // becomes the innermost prefix binding of the nested sweep.
            let parent_seg_rename: Option<(String, String)> = parent_prefix.as_ref()
                .map(|p| (joined_var.clone(), format!("__seg{}", p.len() - 1)));
            let gh_entry = match &parent_seg_rename {
                Some((from, to)) => rename_ident(gh_entry, from, to),
                None => gh_entry,
            };

            let jac_entry = if !jac_stmts.is_empty() {
                let entry = if let Some(ref guard) = guard_expr {
                    quote! { if #guard { #marker #(#jac_stmts)* } }
                } else {
                    quote! { { #marker #(#jac_stmts)* } }
                };
                let entry = match parent_prefix.as_ref() {
                    Some(p) => rename_ident(entry, &joined_var,
                        &format!("__seg{}", p.len() - 1)),
                    None => entry,
                };
                Some(entry)
            } else { None };

            match &entity_location {
                EntityLocation::Collection { .. } | EntityLocation::Nested { .. } => {
                    // SelfBlock on a Vec/Deque/Arena (possibly nested below
                    // the root): group by full access path for merged-loop
                    // emission. A type held in SEVERAL collections -- root
                    // level or nested, in any mix -- gets one sweep per
                    // containment path: same entries, distinct groups (the
                    // entries reference `__item` and `self` only, never the
                    // collection path). The guard above has already rejected
                    // any non-collection path in the set.
                    let locations: Vec<(syn::Ident, Vec<AccessSegment>, String)> =
                        containing_paths.iter().map(|segments| {
                            let last = segments.last().expect("path has >= 1 segment");
                            (syn::Ident::new(&last.field, proc_macro2::Span::call_site()),
                             segments[..segments.len() - 1].to_vec(),
                             path_display(segments))
                        }).collect();
                    for (loc_ident, loc_prefix, loc_key) in locations {
                        let group = collection_groups.entry(loc_key).or_insert_with(|| CollectionGroup {
                            coll_ident: loc_ident.clone(),
                            prefix: loc_prefix.clone(),
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
                        // A root./parent.<selfblock> constraint has no entity
                        // block: registering one would emit set_indices on a
                        // field the entity does not have. (The root's own
                        // SelfBlock is wired by root_self_block_prelude; a
                        // parent's by the passive nested wiring.)
                        if group.self_block.is_none() && root_self_primary.is_none()
                            && parent_self_primary.is_none() {
                            group.self_block = Some(SelfBlockInfo {
                                a_param_count,
                                a_idx_stmts: a_idx_stmts.clone(),
                                block_ident: block_ident.clone(),
                            });
                        }
                        group.cost_entries.push(cost_entry.clone());
                        group.gh_entries.push(gh_entry.clone());
                        if let Some(ref je) = jac_entry { group.jac_entries.push(je.clone()); }
                    }
                }
                EntityLocation::RootSelf
                | EntityLocation::DirectField { .. }
                | EntityLocation::OptionalField { .. } => {
                    if root_self_primary.is_some() {
                        // The single-instance emission writes through the
                        // ENTITY's block field, which this form does not have.
                        return Err(syn::Error::new(proc_macro2::Span::call_site(),
                            format!("{}:{}: a `root.<selfblock>` constraint must live on \
                                     an entity in a collection (Vec/Deque/Arena); for a \
                                     constraint on the root itself use \
                                     `constraint({}, ...)` directly",
                                sc.attr_file, sc.attr_line,
                                constraint.primary_block_field().strip_prefix("root.").unwrap())));
                    }
                    // (parent.<selfblock> single-instance shapes were already
                    // rejected at the parent_prefix validation above.)
                    // SelfBlock on the root itself, on a direct-composed
                    // sub-model, or on an Option<Entity>: emit a single
                    // evaluation (no loop; if-let-wrapped for Option). Group
                    // by access path so multiple constraints on the same
                    // entity merge into one block.
                    let (group_key, accessor_read, accessor_write, optional) = match &entity_location {
                        EntityLocation::RootSelf => (
                            "self".to_string(),
                            quote! { &*__self_ref },
                            quote! { &mut *self },
                            false,
                        ),
                        EntityLocation::DirectField { field } => {
                            let fi = syn::Ident::new(field, proc_macro2::Span::call_site());
                            (
                                format!("self.{}", field),
                                quote! { &self.#fi },
                                quote! { &mut self.#fi },
                                false,
                            )
                        }
                        EntityLocation::OptionalField { field } => {
                            let fi = syn::Ident::new(field, proc_macro2::Span::call_site());
                            (
                                format!("self.{}", field),
                                quote! { __self_ref.#fi.as_ref() },
                                quote! { self.#fi.as_mut() },
                                true,
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
                        optional,
                        self_var: self_var.clone(),
                        root_var_ident: root_var_ident.clone(),
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
            // Entry re-establishes the ref bindings at its top: a preceding
            // entry's block writes ended the loop-level shared borrows.
            // Rereads go BEFORE the guard -- guards may reference resolved
            // entity vars (e.g. `guard = arc.is_ellipse`).
            let gh_entry = if let Some(ref guard) = guard_expr {
                quote! { { #marker #(#entity_index_copies)* #(#resolve_reread_stmts)* if #guard { #(#gh_stmts)* } } }
            } else {
                quote! { { #marker #(#entity_index_copies)* #(#resolve_reread_stmts)* #(#gh_stmts)* } }
            };
            let gh_entry = rename_ident(gh_entry, &root_var_name, "self");
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
                let access = entity_access_expr(&var_id.to_string())?;
                let _ = type_id;
                entity_set_indices.push(quote! {
                    #access.#hb_ident.set_indices(
                        <&[u32; #cnt]>::try_from(&__all_idx[#start..#end]).unwrap()
                    );
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
                    entity_index_copies: entity_index_copies.clone(),
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
            // Entry-top rereads before the guard: see the triplet entry above.
            let gh_entry = if let Some(ref guard) = guard_expr {
                quote! { { #marker #(#entity_index_copies)* #(#resolve_reread_stmts)* if #guard { #(#gh_stmts)* } } }
            } else {
                quote! { { #marker #(#entity_index_copies)* #(#resolve_reread_stmts)* #(#gh_stmts)* } }
            };
            let gh_entry = rename_ident(gh_entry, &root_var_name, "self");
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
                    prefix: cross_prefix.clone(),
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

            // Honor an optional guard: `self` was already rewritten to
            // `__frine` in guard_expr, matching the loop variable here.
            let nested_cost_body = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #(#cost_stmts)* } }
            } else {
                quote! { #(#cost_stmts)* }
            };
            let nested_cost = quote! {
                {
                    #marker
                    let #parent_ident = __item;
                    for __frine in &__item.#frines_ident {
                        #(#resolve_stmts)*
                        let #root_var_ident = &*__self_ref;
                        #nested_cost_body
                    }
                }
            };

            // Parent reads become `__item.*` field projections (disjoint
            // from the iterated frines field); root reads go through
            // `self`; entity writes are temporary borrows built into
            // gh_stmts. No alias bindings remain.
            let nested_gh_body = if let Some(ref guard) = guard_expr {
                quote! { if #guard { #(#gh_stmts)* } }
            } else {
                quote! { { #(#gh_stmts)* } }
            };
            let nested_gh = quote! {
                {
                    #marker
                    for __frine in __item.#frines_ident.iter_mut() {
                        #(#entity_index_copies)*
                        #(#resolve_reread_stmts)*
                        #nested_gh_body
                    }
                }
            };
            let nested_gh = {
                let renamed = rename_ident(nested_gh, &parent_name, "__item");
                rename_ident(renamed, &root_var_name, "self")
            };

            let nested_jac = if !jac_stmts.is_empty() {
                let resolve_stmts_j = resolve_stmts.clone();
                let b_idx_stmts_j = b_idx_stmts.clone();
                let marker_j = marker.clone();
                // Emit rows only for constraint instances the guard admits.
                let jac_body = quote! {
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
                };
                let jac_body = if let Some(ref guard) = guard_expr {
                    quote! { if #guard { #jac_body } }
                } else {
                    quote! { #jac_body }
                };
                Some(quote! {
                    {
                        #marker_j
                        let #parent_ident = __item;
                        for __frine in &__item.#frines_ident {
                            #(#resolve_stmts_j)*
                            let #root_var_ident = &*__self_ref;
                            #jac_body
                        }
                    }
                })
            } else { None };

            let group = collection_groups.entry(group_key).or_insert_with(|| CollectionGroup {
                coll_ident: coll_ident.clone(),
                prefix: Vec::new(),
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
        let prefix = &group.prefix;
        let ctn = nested_container(prefix);     // `self` (one-hop) or `__seg{n-1}` (nested)
        let self_var = &group.self_var;
        let a_type = &group.a_type_ident;
        let _ = a_type;
        let cost_entries = &group.cost_entries;
        let gh_entries = &group.gh_entries;
        let jac_entries = &group.jac_entries;
        let nested_cost = &group.nested_cost_loops;
        let nested_gh = &group.nested_gh_loops;
        let nested_jac = &group.nested_jac_loops;

        // Merged cost loop: SelfBlock entries + nested CrossBlock inner loops
        merged_cost.push(wrap_in_prefix(prefix, false, quote! {
            for __item in #ctn.#coll.iter() {
                let #self_var = __item;
                let #root_var_ident = &*__self_ref;
                #(#cost_entries)*
                #(#nested_cost)*
            }
        }));

        // Merged grad+hessian loop. Entries access the entity as `__item`
        // and the root as `self` directly (renamed at entry creation) --
        // no alias bindings.
        merged_gh.push(wrap_in_prefix(prefix, true, quote! {
            for __item in #ctn.#coll.iter_mut() {
                #(#gh_entries)*
                #(#nested_gh)*
            }
        }));

        // Merged Jacobian loop (only if Jacobian entries/nested exist)
        if !jac_entries.is_empty() || !nested_jac.is_empty() {
            let a_count = group.self_block.as_ref().map(|sb| sb.a_param_count).unwrap_or(0);
            let a_idx_stmts_j: Vec<_> = group.self_block.as_ref()
                .map(|sb| sb.a_idx_stmts.clone()).unwrap_or_default();
            merged_jac.push(wrap_in_prefix(prefix, false, quote! {
                for __item in #ctn.#coll.iter() {
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
            }));
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
            merged_sbi.push(wrap_in_prefix(prefix, true, quote! {
                for __item in #ctn.#coll.iter_mut() {
                    let mut __a_idx = [0u32; #a_count];
                    #(#a_idx)*
                    __item.#block.set_indices(&__a_idx);
                    #ci_set
                    __cid += 1;
                }
            }));
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
                && matches!(seg.ident.to_string().as_str(), "Vec" | "Deque" | "Arena")
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
            { type_ident_name(inner).ok() } else { None };
            let type_name = match inner_name { Some(s) => s, None => continue };
            if !reachable.contains(&type_name) { continue; }
            let layout = match registry_lookup(&type_name) { Some(l) => l, None => continue };
            // No `param_fields.is_empty()` pre-filter here: it counts only
            // DIRECT Param fields, so an entity whose params live entirely
            // inside an #[arael(component)] was skipped and its SelfBlock
            // never got indices -- every gradient and Hessian contribution
            // to it silently dropped. `param_slots` below walks components,
            // and the `offset == 0` guard covers the param-less case.
            let hb_field = match layout.self_block_field.clone() {
                Some(s) => s, None => continue,
            };
            let hb_ident = syn::Ident::new(&hb_field, proc_macro2::Span::call_site());
            let mut a_idx_stmts: Vec<TokenStream2> = Vec::new();
            let mut offset = 0usize;
            for slot in param_slots(&type_name) {
                let size = param_slot_size(&slot.sft);
                if size == 0 { continue; }
                let end = offset + size;
                let access = slot_access(quote! { __item }, &slot.path);
                a_idx_stmts.push(quote! {
                    #access.write_indices(&mut __a_idx[#offset..#end]);
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

        // Same pass for passive DIRECT-COMPOSED entities: a bare
        // struct-typed root field holding Params + SelfBlock<Self> with no
        // self-constraint kept u32::MAX indices (the exact silent-drop
        // this wiring exists to prevent, in the DirectField location).
        // set_indices is idempotent, so re-wiring an already-wired block
        // is harmless.
        for field in &root_fields_passive.named {
            let field_ident = match field.ident.as_ref() {
                Some(i) => i.clone(),
                None => continue,
            };
            // Generic args are ignored: `pose: Pose<f32>` resolves the
            // same layout as `pose: Pose` (shapes are precision-free).
            let type_name = if let syn::Type::Path(tp) = &field.ty
                && let Some(seg) = tp.path.segments.last()
            { seg.ident.to_string() } else { continue };
            if !reachable.contains(&type_name) { continue; }
            let layout = match registry_lookup(&type_name) { Some(l) => l, None => continue };
            // No `param_fields.is_empty()` pre-filter here: it counts only
            // DIRECT Param fields, so an entity whose params live entirely
            // inside an #[arael(component)] was skipped and its SelfBlock
            // never got indices -- every gradient and Hessian contribution
            // to it silently dropped. `param_slots` below walks components,
            // and the `offset == 0` guard covers the param-less case.
            let hb_field = match layout.self_block_field.clone() {
                Some(s) => s, None => continue,
            };
            let hb_ident = syn::Ident::new(&hb_field, proc_macro2::Span::call_site());
            let mut idx_stmts: Vec<TokenStream2> = Vec::new();
            let mut offset = 0usize;
            for slot in param_slots(&type_name) {
                let size = param_slot_size(&slot.sft);
                if size == 0 { continue; }
                let end = offset + size;
                let access = slot_access(quote! { self.#field_ident }, &slot.path);
                idx_stmts.push(quote! {
                    #access.write_indices(&mut __d_idx[#offset..#end]);
                });
                offset = end;
            }
            if offset == 0 { continue; }
            let d_count = offset;
            merged_sbi.push(quote! {
                {
                    let mut __d_idx = [0u32; #d_count];
                    #(#idx_stmts)*
                    self.#field_ident.#hb_ident.set_indices(&__d_idx);
                }
            });
        }

        // Passive NESTED entities: a block-bearing entity two or more hops
        // below the root (e.g. root.paths[k].poses) with no self-constraint.
        // The two passes above reach only the root's own collections / direct
        // fields; walk the reachable set for anything whose location is Nested
        // and wire its SelfBlock through the same prefix loops. Sorted
        // iteration keeps emission deterministic (B11); the `wired` skip avoids
        // double-wiring an entity a self-constraint already handled.
        let mut nested_types: Vec<&String> = reachable.iter().collect();
        nested_types.sort();
        for type_name in nested_types {
            let segments = match resolve_entity_location(root_fields, &root_name.to_string(), type_name) {
                Some(EntityLocation::Nested { segments }) => segments,
                _ => continue, // root-self / direct / one-hop handled above
            };
            let joined: String = segments.iter().map(|s| s.field.clone())
                .collect::<Vec<_>>().join(".");
            if wired.contains(&joined) { continue; }
            let layout = match registry_lookup(type_name) { Some(l) => l, None => continue };
            // No `param_fields.is_empty()` pre-filter here: it counts only
            // DIRECT Param fields, so an entity whose params live entirely
            // inside an #[arael(component)] was skipped and its SelfBlock
            // never got indices -- every gradient and Hessian contribution
            // to it silently dropped. `param_slots` below walks components,
            // and the `offset == 0` guard covers the param-less case.
            let hb_field = match layout.self_block_field.clone() { Some(s) => s, None => continue };
            let hb_ident = syn::Ident::new(&hb_field, proc_macro2::Span::call_site());
            let mut a_idx_stmts: Vec<TokenStream2> = Vec::new();
            let mut offset = 0usize;
            for slot in param_slots(type_name) {
                let size = param_slot_size(&slot.sft);
                if size == 0 { continue; }
                let end = offset + size;
                let access = slot_access(quote! { __item }, &slot.path);
                a_idx_stmts.push(quote! {
                    #access.write_indices(&mut __a_idx[#offset..#end]);
                });
                offset = end;
            }
            if offset == 0 { continue; }
            let a_count = offset;
            let prefix = &segments[..segments.len() - 1];
            let coll_ident = syn::Ident::new(&segments.last().unwrap().field,
                proc_macro2::Span::call_site());
            let ctn = nested_container(prefix);
            merged_sbi.push(wrap_in_prefix(prefix, true, quote! {
                for __item in #ctn.#coll_ident.iter_mut() {
                    let mut __a_idx = [0u32; #a_count];
                    #(#a_idx_stmts)*
                    __item.#hb_ident.set_indices(&__a_idx);
                }
            }));
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
        let a_count = group.a_param_count;
        let a_idx_stmts = &group.a_idx_stmts;
        let block_ident = &group.block_ident;
        let cost_entries = &group.cost_entries;
        let gh_entries = &group.gh_entries;
        let jac_entries = &group.jac_entries;
        let ci_set = group.constraint_index_field.as_ref().map(|fi| {
            quote! { __item.#fi = __cid; }
        });

        // An Option<Entity> location evaluates only when Some -- a None
        // contributes nothing anywhere, like an empty collection (__cid /
        // __jac_cid advance only for the live instance, consistently
        // across the cost / gh / jac / set-indices passes).
        let wrap = |accessor: &TokenStream2, body: TokenStream2| -> TokenStream2 {
            if group.optional {
                quote! { if let Some(__item) = #accessor { #body } }
            } else {
                quote! { { let __item = #accessor; #body } }
            }
        };

        merged_cost.push(wrap(accessor_read, quote! {
            let #self_var = __item;
            let #root_var = &*__self_ref;
            #(#cost_entries)*
        }));

        merged_gh.push(wrap(accessor_write, quote! {
            #(#gh_entries)*
        }));

        if !jac_entries.is_empty() {
            merged_jac.push(wrap(accessor_read, quote! {
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
            }));
        }

        merged_sbi.push(wrap(accessor_write, quote! {
            let mut __a_idx = [0u32; #a_count];
            #(#a_idx_stmts)*
            __item.#block_ident.set_indices(&__a_idx);
            #ci_set
            __cid += 1;
        }));
    }

    // Emit merged cross-constraint loops (one per collection, all attributes inside)
    for group in cross_groups.values() {
        let rc_ident = &group.rc_ident;
        let prefix = &group.prefix;
        let ctn = nested_container(prefix);   // `self` (root-level) or `__seg{n-1}` (nested)
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

        cost_loops.push(wrap_in_prefix(prefix, false, quote! {
            for __frine in #ctn.#rc_ident.iter() {
                #(#resolve_stmts)*
                let #root_var = &*__self_ref;
                #(#cost_entries)*
            }
        }));

        // Entries carry their own ref rereads at the top; root reads go
        // through `self` (renamed at entry creation).
        grad_hessian_loops.push(wrap_in_prefix(prefix, true, quote! {
            for __frine in #ctn.#rc_ident.iter_mut() {
                #(#gh_entries)*
            }
        }));

        if !jac_entries.is_empty() {
            jacobian_loops.push(wrap_in_prefix(prefix, false, quote! {
                for __frine in #ctn.#rc_ident.iter() {
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
            }));
        }

        set_block_indices_loops.push(wrap_in_prefix(prefix, true, quote! {
            for __frine in #ctn.#rc_ident.iter_mut() {
                #(#resolve_stmts)*
                let mut __a_idx = [0u32; #a_param_count];
                #(#a_idx_stmts)*
                let mut __b_idx = [0u32; #b_param_count];
                #(#b_idx_stmts)*
                __frine.#block_ident.set_indices(&__a_idx, &__b_idx);
                #ci_set
                __cid += 1;
            }
        }));
    }

    // Emit merged TripletBlock loops (one per collection, with set_block_indices)
    for group in triplet_groups.values() {
        let rc_ident = &group.rc_ident;
        let tp = group.triplet_param_count;
        let block_ident = &group.block_ident;
        let triplet_idx_stmts = &group.triplet_idx_stmts;
        let resolve_stmts = &group.resolve_stmts;
        let entity_index_copies = &group.entity_index_copies;
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
        // Loop-level resolves feed the __all_idx build; entries re-establish
        // their own bindings (a preceding entry's writes end these borrows).
        grad_hessian_loops.push(quote! {
            for __frine in self.#rc_ident.iter_mut() {
                #(#resolve_stmts)*
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
                    #(#entity_index_copies)*
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
                    #(#entity_index_copies)*
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

    // Precision-dependent pieces for the generated RootProblem impl (the
    // solve_with / solve_dense / solve_sparse entry points are LmProblem
    // default methods gated on it -- nothing solver-side is generated).
    let (solve_serialize, solve_deserialize) = if precision == "f32" {
        (quote! { serialize32 }, quote! { deserialize32 })
    } else {
        (quote! { serialize64 }, quote! { deserialize64 })
    };
    let requires_compute = custom || has_triplet_block;
    let (cells_method, positions_method) = if precision == "f32" {
        (quote! { collect_hessian_cells32 }, quote! { accumulate_hessian_positions32 })
    } else {
        (quote! { collect_hessian_cells64 }, quote! { accumulate_hessian_positions64 })
    };

    let ref_issue_walker = generate_ref_issue_walker(&root_name.to_string());

    let mut tokens = quote! {
        #(#constraint_impls)*

        impl arael::simple_lm::RootProblem<#prec_type> for #root_name {
            fn serialize(&mut self, data: &mut std::vec::Vec<#prec_type>) {
                self.#solve_serialize(data)
            }
            fn deserialize(&mut self, data: &[#prec_type]) {
                self.#solve_deserialize(data)
            }
            fn param_block_spans(&self) -> std::vec::Vec<(u32, u32)> {
                let mut __out = std::vec::Vec::new();
                arael::model::Model::collect_param_blocks(self, &mut __out);
                __out
            }
            #marginalize_hint_fn
            #ref_issue_walker
        }

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
                #root_self_block_prelude
                #(#set_block_indices_loops)*
            }

            /// Returns the cost (sum of squared residuals, excluding
            /// extended-model residuals) as a byproduct of the sweep.
            fn __compute_blocks(&mut self, params: &[#prec_type], grad: &mut [#prec_type]) -> #prec_type {
                // Generated expressions may call Float trait methods
                // (e.g. heaviside from safe-function derivatives).
                use arael::utils::Float as _;
                arael::model::Model::#update_method(self, params);
                #extended_update_call
                arael::model::Model::zero_blocks(self);
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
                    // Generated expressions may call Float trait methods
                    // (e.g. heaviside from safe-function derivatives).
                    use arael::utils::Float as _;
                    arael::model::Model::#update_method(self, params);
                    #ext_update
                    // Read-only traversal: a plain shared reborrow suffices.
                    let __self_ref = &*self;
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
            fn hessian_pattern_requires_compute(&self) -> bool { #requires_compute }
            #marginalize_hint_fn
            #marginalize_candidates_fn
            fn collect_hessian_cells(&self, out: &mut std::vec::Vec<(u32, u32)>) {
                arael::model::Model::#cells_method(self, out)
            }
            fn accumulate_hessian_positions(&self, resolve: &mut dyn FnMut(u32, u32) -> usize, out: &mut std::vec::Vec<usize>) {
                arael::model::Model::#positions_method(self, resolve, out)
            }
            fn collect_param_block_spans(&self, out: &mut std::vec::Vec<(u32, u32)>) {
                arael::model::Model::collect_param_blocks(self, out)
            }
            fn calc_cost(&mut self, params: &[#prec_type]) -> #prec_type {
                // Generated expressions may call Float trait methods
                // (e.g. heaviside from safe-function derivatives).
                use arael::utils::Float as _;
                arael::model::Model::#update_method(self, params);
                #extended_update_call
                // Read-only traversal: a plain shared reborrow suffices.
                let __self_ref = &*self;
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
                arael::model::Model::#accumulate_method(self, hessian);
                __cost
            }

            fn calc_grad_hessian_band(&mut self, params: &[#prec_type], grad: &mut [#prec_type], band: &mut [#prec_type], kd: usize) -> Result<#prec_type, arael::simple_lm::BandOverflow> {
                grad.iter_mut().for_each(|g| *g = 0.0);
                let mut __cost = self.__compute_blocks(params, grad);
                #extended_cost_call
                band.iter_mut().for_each(|b| *b = 0.0);
                arael::model::Model::#accumulate_band_method(self, band, kd)?;
                Ok(__cost)
            }

            fn calc_grad_hessian_sparse(&mut self, params: &[#prec_type], grad: &mut [#prec_type], coo: &mut arael::simple_lm::CooMatrix<#prec_type>) -> #prec_type {
                grad.iter_mut().for_each(|g| *g = 0.0);
                let mut __cost = self.__compute_blocks(params, grad);
                #extended_cost_call
                coo.clear();
                arael::model::Model::#accumulate_sparse_method(self, coo);
                __cost
            }

            fn calc_grad_hessian_sparse_direct(&mut self, params: &[#prec_type], grad: &mut [#prec_type], csc: &mut arael::simple_lm::CscMatrix<#prec_type>) -> #prec_type {
                grad.iter_mut().for_each(|g| *g = 0.0);
                let mut __cost = self.__compute_blocks(params, grad);
                #extended_cost_call
                csc.vals.iter_mut().for_each(|v| *v = 0.0 as #prec_type);
                arael::model::Model::#accumulate_sparse_direct_method(self, csc);
                __cost
            }

            fn calc_grad_hessian_sparse_indexed(&mut self, params: &[#prec_type], grad: &mut [#prec_type], vals: &mut [#prec_type], positions: &[usize]) -> #prec_type {
                grad.iter_mut().for_each(|g| *g = 0.0);
                let mut __cost = self.__compute_blocks(params, grad);
                #extended_cost_call
                vals.iter_mut().for_each(|v| *v = 0.0 as #prec_type);
                let mut cursor = 0usize;
                arael::model::Model::#accumulate_sparse_indexed_method(self, vals, positions, &mut cursor);
                // The cached position map is replayed by cursor and assumes
                // an identical entry sequence every iteration. A shorter
                // sequence (a TripletBlock or extended constraint emitting
                // fewer entries than when the pattern was built) would
                // scatter every subsequent block into wrong slots -- a
                // silently wrong Hessian.
                assert!(cursor == positions.len(),
                    "sparsity pattern changed between iterations: {} Hessian entries \
                     accumulated but the cached pattern has {} (TripletBlock / extended \
                     constraint entry counts must stay constant within one solve)",
                    cursor, positions.len());
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

/// The synthetic symbol the loss closure's argument binds to. The loss
/// codegen accumulates the block's squared residual norm into a local of this
/// name, and the loss/weight expressions read it back by rendering the symbol
/// verbatim.
pub const LOSS_ARG_SYM: &str = "__block_cost";

/// Interpret constraint body and return (residual expressions, param symbols,
/// optional robust-loss expression rho(s) in terms of [`LOSS_ARG_SYM`]).
fn interpret_constraint_body(
    struct_name: &syn::Ident,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    constraint: &ConstraintAttr,
    root_type_name: &str,
) -> syn::Result<(Vec<E>, Vec<String>, Option<E>, Vec<(E, E)>)> {
    // `root.<selfblock>` primary: the entity supplies only data, every param
    // is the root's. Deep validation lives in the traversal side (which sees
    // the real field types); here it only shapes the symbol environment.
    let is_root_self_primary = constraint.primary_block_field().starts_with("root.");
    // `parent.<selfblock>` primary: like the root form, but the params
    // belong to the entity CONTAINING this constraint struct.
    let is_parent_self_primary = constraint.primary_block_field().starts_with("parent.");
    let is_remote = constraint.primary_block_field().contains('.')
        && !is_root_self_primary && !is_parent_self_primary;
    let (a_type, b_type) = if is_root_self_primary || is_parent_self_primary {
        (struct_name.to_string(), None)
    } else if is_remote {
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
        // through the same shared scan the traversal side uses. The
        // fallback to the target type covers per-struct expansion phases
        // where the parent is not registered yet.
        let parent_type = find_containing_parent(root_type_name, &struct_name.to_string())
            .unwrap_or_else(|| inner.to_string());
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
        if is_parent_self_primary {
            // Self alias binds the constraint struct; the `parent =`
            // alias (default: the parent type, lowercased) binds the
            // CONTAINING entity, whose params this block holds.
            var_infos.push((struct_name.to_string().to_lowercase(), struct_name.to_string()));
            let parent_type = find_containing_parent(root_type_name, &struct_name.to_string())
                .unwrap_or_else(|| root_type_name.to_string());
            let pvar = constraint.parent_name.clone()
                .unwrap_or_else(|| parent_type.to_lowercase());
            var_infos.push((pvar, parent_type));
        } else if a_type != "__triplet__" && !is_pure_multi_cross {
            var_infos.push((parent_name.clone(), a_type.clone()));
        }
        let root_var = root_type_name.to_lowercase();
        var_infos.push((root_var, root_type_name.to_string()));
    }

    // Setup context — recursively register all fields including nested structs
    let mut ctx = ConstraintCtx::new();
    for (var_name, type_name) in &var_infos {
        // Two variables with the same name and different types is a real
        // collision (e.g. a Ref field named `path` vs the auto-registered
        // root variable for a root type `Path`): whichever registered
        // last would silently win.
        if let Some(prev) = ctx.entity_vars.get(var_name)
            && prev != type_name {
                return Err(syn::Error::new_spanned(struct_name,
                    format!("variable name `{}` is ambiguous in this constraint: it refers \
                             to both `{}` and `{}` (a Ref field colliding with an \
                             auto-registered variable) -- rename the field",
                        var_name, prev, type_name)));
            }
        ctx.entity_vars.insert(var_name.clone(), type_name.clone());
        register_bindings_recursive(&mut ctx, var_name, var_name, type_name)?;
    }

    // `root` aliases the root variable in constraint bodies, matching the
    // `root.<field>` block spec: `root.a` and `<root_lc>.a` are the same
    // param. The alias renders through the lowercased-type access path, so
    // everything downstream (param symbols, emission bindings) is untouched.
    // A Ref field genuinely named `root` keeps its own meaning.
    if !ctx.entity_vars.contains_key("root")
        && var_infos.iter().any(|(_, tn)| tn == root_type_name)
    {
        let root_var = root_type_name.to_lowercase();
        ctx.entity_vars.insert("root".to_string(), root_type_name.to_string());
        register_bindings_recursive(&mut ctx, "root", &root_var, root_type_name)?;
    }

    // Register the constraint struct's own non-Ref fields
    // For CrossBlock: accessible via lowercase struct name, code uses __frine
    // For SelfBlock: the struct IS the variable (already registered above via var_infos)
    if b_type.is_some() || a_type == "__triplet__" {
        // Use a simple name derived from the struct name
        // Derive self-reference name from struct: PosePair -> "posepair"
        let self_var = struct_name.to_string().to_lowercase();
        ctx.entity_vars.entry(self_var.clone()).or_insert_with(|| struct_name.to_string());
        register_bindings_recursive(&mut ctx, &self_var, "__frine", &struct_name.to_string())?;
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
            // param_slots walks #[arael(component)] fields, so component
            // params appear in the symbol span like direct ones.
            for slot in param_slots(type_name) {
                let sym_base = if slot.universal_delta {
                    format!("{}.{}.delta", var_name, slot.path)
                } else {
                    format!("{}.{}.work()", var_name, slot.path)
                };
                add_param_symbols(&sym_base, &slot.sft, &mut param_symbols);
            }
        }
    } else {
        // SelfBlock/CrossBlock: collect params from A and optionally B
        let a_var_name = {
            var_infos.iter().find(|(_, tn)| *tn == a_type)
                .map(|(vn, _)| vn.clone()).unwrap_or(parent_name.clone())
        };

        for slot in param_slots(&a_type) {
            let sym_base = if slot.universal_delta {
                format!("{}.{}.delta", a_var_name, slot.path)
            } else {
                format!("{}.{}.work()", a_var_name, slot.path)
            };
            add_param_symbols(&sym_base, &slot.sft, &mut param_symbols);
        }
        if let Some(ref b_type_name) = b_type {
            let b_var = var_infos.iter().find(|(vn, tn)| {
                tn == b_type_name && *vn != a_var_name
            }).or_else(|| var_infos.iter().find(|(_, tn)| tn == b_type_name))
                .map(|(vn, _)| vn.clone()).unwrap_or_else(|| b_type_name.to_lowercase());
            for slot in param_slots(b_type_name) {
                let sym_base = if slot.universal_delta {
                    format!("{}.{}.delta", b_var, slot.path)
                } else {
                    format!("{}.{}.work()", b_var, slot.path)
                };
                add_param_symbols(&sym_base, &slot.sft, &mut param_symbols);
            }
        }

        // `root.<selfblock>` primary: the residuals read the root's params,
        // bound as the lowercased root type (the same name the root-triplet
        // form uses). The entity contributes no params of its own.
        if is_root_self_primary {
            let root_var = root_type_name.to_lowercase();
            for slot in param_slots(root_type_name) {
                let sym_base = if slot.universal_delta {
                    format!("{}.{}.delta", root_var, slot.path)
                } else {
                    format!("{}.{}.work()", root_var, slot.path)
                };
                add_param_symbols(&sym_base, &slot.sft, &mut param_symbols);
            }
        }

        // `parent.<selfblock>` primary: every param is the containing
        // entity's, read through the parent binding.
        if is_parent_self_primary {
            let parent_type = find_containing_parent(root_type_name, &struct_name.to_string())
                .unwrap_or_else(|| root_type_name.to_string());
            let pvar = constraint.parent_name.clone()
                .unwrap_or_else(|| parent_type.to_lowercase());
            for slot in param_slots(&parent_type) {
                let sym_base = if slot.universal_delta {
                    format!("{}.{}.delta", pvar, slot.path)
                } else {
                    format!("{}.{}.work()", pvar, slot.path)
                };
                add_param_symbols(&sym_base, &slot.sft, &mut param_symbols);
            }
        }
    }

    // Interpret body. Only `let` bindings and one final residual
    // expression are meaningful here; anything else used to be silently
    // dropped (macros, items) or silently treated as extra residuals
    // (stray semicolon-terminated expressions).
    let mut residuals: Vec<E> = Vec::new();
    let n_stmts = constraint.body_stmts.len();
    for (si, stmt) in constraint.body_stmts.iter().enumerate() {
        match stmt {
            Stmt::Local(local) => {
                let name = match &local.pat {
                    Pat::Ident(pi) => pi.ident.to_string(),
                    _ => return Err(syn::Error::new_spanned(&local.pat, "simple let binding required")),
                };
                let init = local.init.as_ref().ok_or_else(|| syn::Error::new_spanned(local, "initializer required"))?;
                let val = eval_expr(&init.expr, &mut ctx)?;
                ctx.lets.insert(name.clone());
                ctx.bindings.insert(name, val);
            }
            Stmt::Expr(expr, semi) => {
                if si + 1 != n_stmts {
                    return Err(syn::Error::new_spanned(expr,
                        "only `let` bindings may precede the final residual expression \
                         (this expression statement would otherwise be treated as extra \
                         residuals or dropped)"));
                }
                if semi.is_some() {
                    return Err(syn::Error::new_spanned(expr,
                        "the residual expression must not end with a semicolon"));
                }
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
            Stmt::Macro(m) => {
                return Err(syn::Error::new_spanned(m,
                    "macro statements are not supported in constraint bodies \
                     (they were silently dropped before)"));
            }
            Stmt::Item(item) => {
                return Err(syn::Error::new_spanned(item,
                    "item declarations are not supported in constraint bodies"));
            }
        }
    }

    if !params_from_registry_check(&param_symbols) {
        let mut all_vars = std::collections::BTreeSet::new();
        for r in &residuals { all_vars.extend(r.free_vars()); }
        for var in &all_vars {
            if var.contains(".work()") { param_symbols.push(var.clone()); }
        }
    }

    // Optional robust loss: a closure `|s| <expr>` over the block's squared
    // residual norm. Evaluate its body against the same ctx as the residuals
    // (so field reads like `self.k` / `parent.gamma` resolve identically),
    // with the argument bound to the synthetic LOSS_ARG_SYM symbol.
    let loss_expr = if let Some(loss_src) = &constraint.loss {
        let mut closure: syn::ExprClosure = syn::parse_str(loss_src)
            .map_err(|e| syn::Error::new_spanned(struct_name,
                format!("constraint `loss` must be a closure `|s| <expr>`: {}", e)))?;
        if closure.inputs.len() != 1 {
            return Err(syn::Error::new_spanned(struct_name,
                "constraint `loss` closure takes exactly one argument (the squared residual norm)"));
        }
        let arg = match &closure.inputs[0] {
            syn::Pat::Ident(pi) => pi.ident.to_string(),
            syn::Pat::Type(pt) => match &*pt.pat {
                syn::Pat::Ident(pi) => pi.ident.to_string(),
                _ => return Err(syn::Error::new_spanned(struct_name,
                    "constraint `loss` argument must be a plain identifier")),
            },
            _ => return Err(syn::Error::new_spanned(struct_name,
                "constraint `loss` argument must be a plain identifier")),
        };
        // Mirror the body's `self` rewrite so field reads resolve the same way.
        rewrite_guard_self(&mut closure.body, &struct_name.to_string().to_lowercase());
        ctx.bindings.insert(arg.clone(), SymVal::Scalar(arael_sym::symbol(LOSS_ARG_SYM)));
        ctx.lets.insert(arg);
        match eval_expr(&closure.body, &mut ctx)? {
            SymVal::Scalar(e) => Some(e),
            other => return Err(syn::Error::new_spanned(struct_name,
                format!("constraint `loss` must evaluate to a scalar, got {}", other.type_name()))),
        }
    } else {
        None
    };

    Ok((residuals, param_symbols, loss_expr, ctx.subs))
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

/// Root fields containing `type_name` -- collections (Vec/Deque/Arena) of it
/// (`true`) or direct struct-typed fields (`false`) -- in declaration order.
/// Multiple matches are supported only for SelfBlock-constrained entities in
/// collections; the guard in `generate_root_methods` rejects the rest.
/// How a root field holds an entity type.
#[derive(Clone, Copy, PartialEq)]
enum ContainKind {
    /// Vec / Deque / Arena -- multi-instance, iterated.
    Collection,
    /// Plain struct-typed field -- single instance.
    Direct,
    /// Option<T> -- zero or one instance; iterates as a zero-or-one
    /// collection (frines) or is if-let wrapped (single-instance sweeps).
    Optional,
}

/// `#[arael(skip)]` on the field: excluded from the model entirely --
/// not serialized, not updated, and (via [`root_containments`]) not a
/// containment location, so no constraint sweep ever runs over it.
fn field_is_skipped(field: &syn::Field) -> bool {
    field.attrs.iter().any(|a| {
        a.path().is_ident("arael")
            && a.parse_args::<proc_macro2::TokenStream>().map_or(false, |ts| {
                ts.into_iter().next().is_some_and(|t| t.to_string() == "skip")
            })
    })
}

/// THE one-hop containment walk: every root field holding `type_name`,
/// in declaration order, with how it holds it. The single syntax-side
/// authority on which root fields count as containment -- location
/// resolution, collection lookup, and the duplicate-containment guard
/// are all views of this list, so a containment rule (a new container
/// kind, an attribute exclusion) is added exactly once.
/// `#[arael(skip)]` fields are not containment: serialize/update ignore
/// them, so a sweep over one would evaluate never-updated params.
fn root_containments(
    root_fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    type_name: &str,
) -> Vec<(String, ContainKind)> {
    let mut out = Vec::new();
    for field in root_fields {
        let Some(ident) = field.ident.as_ref() else { continue };
        if field_is_skipped(field) { continue; }
        if let syn::Type::Path(tp) = &field.ty
            && let Some(seg) = tp.path.segments.last() {
                let container = seg.ident.to_string();
                let wrapped_matches = || {
                    if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                        && let Some(syn::GenericArgument::Type(inner)) = args.args.first()
                            && let Ok(inner_name) = type_ident_name(inner) {
                                inner_name == type_name
                            } else { false }
                };
                match container.as_str() {
                    "Vec" | "Deque" | "Arena" => {
                        if wrapped_matches() {
                            out.push((ident.to_string(), ContainKind::Collection));
                        }
                    }
                    "Option" => {
                        if wrapped_matches() {
                            out.push((ident.to_string(), ContainKind::Optional));
                        }
                    }
                    _ => {
                        if seg.ident == type_name {
                            out.push((ident.to_string(), ContainKind::Direct));
                        }
                    }
                }
            }
    }
    out
}

/// First root collection (Vec/Deque/Arena) holding `type_name`.
fn find_root_collection(
    root_fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    type_name: &str,
) -> Option<String> {
    root_containments(root_fields, type_name).into_iter()
        .find(|(_, k)| *k == ContainKind::Collection)
        .map(|(f, _)| f)
}

/// The struct whose layout CONTAINS `struct_name` (a Struct or
/// OptionalStruct field; a collection records its element type as
/// Struct). The current root wins when it holds the struct directly --
/// the global registry scan is alphabetical, and another root holding
/// the same collection would hijack the resolution. Shared by the two
/// remote-block resolution phases so they cannot drift.
fn find_containing_parent(root_type_name: &str, struct_name: &str) -> Option<String> {
    let holds = |sft: &SymFieldType| matches!(sft,
        SymFieldType::Struct(s) | SymFieldType::OptionalStruct(s)
            if s == struct_name);
    let current_root_holds = registry_lookup(root_type_name)
        .map(|l| l.fields.iter().any(|(_, sft)| holds(sft)))
        .unwrap_or(false);
    if current_root_holds {
        return Some(root_type_name.to_string());
    }
    let guard = crate::SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    guard.as_ref().and_then(|reg| {
        reg.layouts.iter().find(|(_, layout)| {
            layout.fields.iter().any(|(_, sft)| holds(sft))
        }).map(|(name, _)| name.clone())
    })
}

/// Where on the root struct an entity of a given type lives.
#[derive(Clone)]
enum EntityLocation {
    /// Vec<T> / Deque<T> / Arena<T> field on root. Multi-instance, iterated.
    Collection { field: String },
    /// Plain struct-typed field on root (e.g. `sub: Sub`). Single instance.
    DirectField { field: String },
    /// Option<T> field on root. Zero or one instance; sweeps are wrapped
    /// in `if let Some(..)` so a None contributes nothing.
    OptionalField { field: String },
    /// The constraint's entity type is the root struct itself. Single instance, accessor is `self`.
    RootSelf,
    /// Entity reachable two or more hops below the root through a chain of
    /// collection / direct-struct fields (e.g. `root.paths[k].poses`). The
    /// last segment is the collection holding the entity; earlier segments are
    /// the loops the emitter wraps around it.
    Nested { segments: Vec<AccessSegment> },
}

fn resolve_entity_location(
    root_fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    root_name: &str,
    type_name: &str,
) -> Option<EntityLocation> {
    if type_name == root_name {
        return Some(EntityLocation::RootSelf);
    }
    // One-hop containment from the shared walk. A collection wins over a
    // single-instance holding whatever the declaration order (the
    // duplicate guard rejects every mixed multi-holding shape anyway;
    // this keeps the resolution stable for the legal ones).
    let one_hop = root_containments(root_fields, type_name);
    if let Some((field, _)) = one_hop.iter().find(|(_, k)| *k == ContainKind::Collection) {
        return Some(EntityLocation::Collection { field: field.clone() });
    }
    if let Some((field, kind)) = one_hop.into_iter().next() {
        return Some(match kind {
            ContainKind::Direct => EntityLocation::DirectField { field },
            ContainKind::Optional => EntityLocation::OptionalField { field },
            ContainKind::Collection => unreachable!(),
        });
    }
    // Not a direct child of the root: walk the registry for a deeper
    // containment path (e.g. root -> Vec<Path> -> Deque<Pose>).
    if let Some(segments) = resolve_nested_path(root_name, type_name) {
        return Some(EntityLocation::Nested { segments });
    }
    None
}

/// One hop in a root -> entity access path through the model tree.
#[derive(Clone, Debug, PartialEq)]
struct AccessSegment {
    /// Field name on the containing struct.
    field: String,
    /// true = Vec/Deque/Arena (the emitter iterates it); false = a plain
    /// struct-typed field (single instance).
    collection: bool,
    /// true = Option<T>: iterated like a zero-or-one collection, so a
    /// None along the path contributes nothing.
    optional: bool,
}

/// EVERY containment path from the root to `target`, shallowest first:
/// the one-hop locations from the shared syntax walk
/// ([`root_containments`], so `#[arael(skip)]` fields are excluded),
/// then every distinct registry path of two or more hops -- a
/// duplicated intermediate collection yields one path per holding
/// field. The duplicate-containment guard and the per-path SelfBlock
/// sweep emission both consume this list, so a path one of them sees,
/// the other sees too.
fn containment_paths(
    root_fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    root_name: &str,
    target: &str,
) -> Vec<Vec<AccessSegment>> {
    let mut out: Vec<Vec<AccessSegment>> = root_containments(root_fields, target)
        .into_iter()
        .map(|(field, kind)| vec![AccessSegment {
            field,
            collection: kind == ContainKind::Collection,
            optional: kind == ContainKind::Optional,
        }])
        .collect();
    fn walk(
        cur: &str,
        target: &str,
        path: &mut Vec<AccessSegment>,
        stack: &mut Vec<String>,
        out: &mut Vec<Vec<AccessSegment>>,
    ) {
        let Some(layout) = registry_lookup(cur) else { return };
        if stack.iter().any(|s| s == cur) { return; } // cycle guard
        stack.push(cur.to_string());
        for (fname, sft) in &layout.fields {
            // Ref<T> fields point at an entity, they do not CONTAIN it.
            if layout.ref_paths.iter().any(|(rf, _)| rf == fname) { continue; }
            let (child, collection, optional) = match sft {
                SymFieldType::Struct(c) =>
                    (c.clone(), layout.collection_fields.iter().any(|f| f == fname), false),
                SymFieldType::OptionalStruct(c) => (c.clone(), false, true),
                _ => continue,
            };
            path.push(AccessSegment { field: fname.clone(), collection, optional });
            // Depth-1 hits already came from the syntax walk above.
            if child == target && path.len() >= 2 {
                out.push(path.clone());
            }
            walk(&child, target, path, stack, out);
            path.pop();
        }
        stack.pop();
    }
    let mut path = Vec::new();
    let mut stack = Vec::new();
    walk(root_name, target, &mut path, &mut stack, &mut out);
    out
}

/// Dotted display form of a containment path (`ga.subs`).
fn path_display(segments: &[AccessSegment]) -> String {
    segments.iter().map(|s| s.field.as_str()).collect::<Vec<_>>().join(".")
}

/// Walk the type registry from `root_name` to find where a struct of
/// `target` type lives, returning the chain of field hops (root -> ... ->
/// the field holding `target`; the last segment's element type is `target`).
/// Follows only CONTAINMENT edges -- collection fields and direct struct
/// fields -- and skips `Ref<T>` fields (recorded in `ref_paths`, not
/// containment) and block/param fields. Returns the first path found in
/// deterministic field order. `None` if `target` is not reachable by
/// containment (or is the root itself / a one-hop child; those are handled by
/// `resolve_entity_location`'s direct arms).
fn resolve_nested_path(root_name: &str, target: &str) -> Option<Vec<AccessSegment>> {
    fn walk(cur: &str, target: &str, path: &mut Vec<AccessSegment>, stack: &mut Vec<String>)
        -> Option<Vec<AccessSegment>>
    {
        let layout = registry_lookup(cur)?;
        if stack.iter().any(|s| s == cur) { return None; } // cycle guard
        stack.push(cur.to_string());
        for (fname, sft) in &layout.fields {
            // Ref<T> fields point at an entity, they do not CONTAIN it.
            if layout.ref_paths.iter().any(|(rf, _)| rf == fname) { continue; }
            let (child, collection, optional) = match sft {
                SymFieldType::Struct(c) =>
                    (c.clone(), layout.collection_fields.iter().any(|f| f == fname), false),
                SymFieldType::OptionalStruct(c) => (c.clone(), false, true),
                _ => continue, // Scalar/Vec2/.../Skip: not a struct edge
            };
            path.push(AccessSegment { field: fname.clone(), collection, optional });
            if child == target {
                return Some(path.clone());
            }
            if let Some(found) = walk(&child, target, path, stack) {
                return Some(found);
            }
            path.pop();
        }
        stack.pop();
        None
    }
    let mut path = Vec::new();
    let mut stack = Vec::new();
    walk(root_name, target, &mut path, &mut stack)
}

/// The accessor the innermost entity loop uses for its own collection: `self`
/// when the prefix is empty (one-hop), otherwise the last prefix loop variable
/// `__seg{n-1}` (e.g. the current `Path` when iterating `root.paths`).
fn nested_container(prefix: &[AccessSegment]) -> TokenStream2 {
    if prefix.is_empty() {
        quote! { self }
    } else {
        let last = syn::Ident::new(&format!("__seg{}", prefix.len() - 1),
            proc_macro2::Span::call_site());
        quote! { #last }
    }
}

/// Wrap `inner` in one `for` loop per `prefix` segment (the outer hops root ->
/// ... -> the container of the innermost collection). `inner` is expected to
/// reference the accessor returned by `nested_container(prefix)`. `mutable`
/// picks iter_mut vs iter (the whole borrow chain must agree). A collection
/// segment emits a loop binding `__seg{i}`; a direct struct segment emits a
/// `let` binding. With an empty prefix this returns `inner` unchanged, so
/// one-hop emission is byte-identical to before.
fn wrap_in_prefix(prefix: &[AccessSegment], mutable: bool, inner: TokenStream2) -> TokenStream2 {
    let mut code = inner;
    for (i, seg) in prefix.iter().enumerate().rev() {
        let field = syn::Ident::new(&seg.field, proc_macro2::Span::call_site());
        let bind = syn::Ident::new(&format!("__seg{}", i), proc_macro2::Span::call_site());
        let parent: TokenStream2 = if i == 0 {
            quote! { self }
        } else {
            let pv = syn::Ident::new(&format!("__seg{}", i - 1), proc_macro2::Span::call_site());
            quote! { #pv }
        };
        code = if seg.collection || seg.optional {
            // An Option segment iterates as a zero-or-one collection: a
            // None along the path contributes nothing, and the binding
            // inside the loop is the CONTAINED struct, so later hops and
            // the inner sweep are container-agnostic.
            if mutable { quote! { for #bind in #parent.#field.iter_mut() { #code } } }
            else       { quote! { for #bind in #parent.#field.iter()     { #code } } }
        } else if mutable {
            quote! { { let #bind = &mut #parent.#field; #code } }
        } else {
            quote! { { let #bind = &#parent.#field; #code } }
        };
    }
    code
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
    let guard = crate::SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
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


#[cfg(test)]
mod nested_path_tests {
    use super::*;
    use crate::{SymLayout, registry_store};

    // Build a minimal SymLayout for resolver tests. Only `fields`,
    // `collection_fields` and `ref_paths` matter to resolve_nested_path.
    fn layout(fields: &[(&str, SymFieldType)], collections: &[&str]) -> SymLayout {
        SymLayout {
            fields: fields.iter().map(|(n, t)| (n.to_string(), t.clone())).collect(),
            collection_fields: collections.iter().map(|s| s.to_string()).collect(),
            param_fields: Vec::new(),
            ref_paths: Vec::new(),
            euler_angle_fields: Vec::new(),
            universal_euler_angle_fields: Vec::new(),
            universal_rotvec_fields: Vec::new(),
            symbolic_fields: Vec::new(),
            deriv_fields: Vec::new(),
            component: false,
            constraint_index_field: None,
            self_block_field: None,
        }
    }

    fn coll(field: &str) -> AccessSegment {
        AccessSegment { field: field.to_string(), collection: true, optional: false }
    }

    #[test]
    fn resolves_entity_two_collection_hops_below_root() {
        // Map { paths: Vec<Path>, landmarks: Vec<Landmark> }
        // Path { poses: Deque<Pose>, pose_pairs: Vec<PosePair> }
        // Distinct "Nrt" names so the process-global registry never collides.
        registry_store("NrtMap", layout(
            &[("paths", SymFieldType::Struct("NrtPath".into())),
              ("landmarks", SymFieldType::Struct("NrtLandmark".into()))],
            &["paths", "landmarks"])).unwrap();
        registry_store("NrtPath", layout(
            &[("poses", SymFieldType::Struct("NrtPose".into())),
              ("pose_pairs", SymFieldType::Struct("NrtPosePair".into()))],
            &["poses", "pose_pairs"])).unwrap();
        registry_store("NrtPose", layout(&[], &[])).unwrap();
        registry_store("NrtLandmark", layout(&[], &[])).unwrap();

        // Two-hop nested entity.
        assert_eq!(resolve_nested_path("NrtMap", "NrtPose"),
                   Some(vec![coll("paths"), coll("poses")]));
        // One-hop child is still found (real use routes it through the direct arms).
        assert_eq!(resolve_nested_path("NrtMap", "NrtLandmark"),
                   Some(vec![coll("landmarks")]));
        // Unreachable.
        assert_eq!(resolve_nested_path("NrtMap", "Nonexistent"), None);
    }

    #[test]
    fn ref_fields_are_not_containment() {
        // A constraint struct with a Ref<Target> must NOT make Target reachable
        // by containment (that would be a spurious path through a reference).
        registry_store("NrtRefRoot", layout(
            &[("cons", SymFieldType::Struct("NrtCons".into()))], &["cons"])).unwrap();
        let mut cons = layout(&[("target", SymFieldType::Struct("NrtTarget".into()))], &[]);
        cons.ref_paths.push(("target".to_string(), "root.somewhere".to_string()));
        registry_store("NrtCons", cons).unwrap();
        registry_store("NrtTarget", layout(&[], &[])).unwrap();

        assert_eq!(resolve_nested_path("NrtRefRoot", "NrtTarget"), None);
    }
}
