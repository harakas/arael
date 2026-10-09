//! `#[arael::function]` attribute macro.
//!
//! Registers a user-defined function in the process-wide macro registry
//! so `#[arael::model]` constraint bodies can call it like a built-in.
//! Two forms, distinguished by the attributed fn's signature:
//!
//! - **Form A**: `fn name(x: E, ...) -> E { body }` -- purely symbolic.
//!   Body captured as an arael-sym source string and stashed. The fn
//!   itself is rewritten to delegate to `arael_sym::parse_with_functions`
//!   so ordinary Rust callers (runtime-fit, `ExtendedModel`) still work.
//!
//! - **Form B**: `#[arael::function(sym_name, derivs = [...])] fn eval(x: f32, ...) -> f32 { ... }`
//!   -- opaque numerical eval + explicit symbolic derivatives. The eval
//!   fn is kept as-written; a sibling `pub fn sym_name(x: E, ...) -> E`
//!   is emitted, wrapping `arael_sym::extern_func`.
//!
//! - **Form C**: `fn name(p: vect3sym, k: E, ...) -> (E, E) { let ...; expr }`
//!   -- typed. Parameters and the result are the body language's values
//!   (`E`, `vect2sym`, `vect3sym`, `matrix2sym`, `matrix3sym`,
//!   `quaternsym`, a tuple or an `[E; N]` array of them), the body a
//!   block with `let` bindings. The block is stashed as source and
//!   evaluated by the constraint-body interpreter at every call site, so
//!   a call inlines exactly what the body would have written. The fn is
//!   emitted as ordinary Rust over the arael-sym types (reads of locals
//!   cloned, `match` lowered to `select`) so it stays callable from user
//!   code.
//!
//! Forms A and B insert no `.clone()`: body / deriv tokens are
//! stringified verbatim and handed to `arael_sym::parse_with_functions`
//! at use time, which has no Rust move semantics.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::Parse;

use crate::{registry_store_function, UserFunction};

/// Entry point. Dispatch on signature, then form-specific handling.
pub fn function_attribute(attr: TokenStream2, item: TokenStream2) -> syn::Result<TokenStream2> {
    let input: syn::ItemFn = syn::parse2(item)
        .map_err(|e| syn::Error::new(e.span(),
            format!("#[arael::function] must be attached to a fn: {}", e)))?;
    match classify_signature(&input.sig, &input.block)? {
        SignatureKind::FormA => form_a(attr, input),
        SignatureKind::FormB(shape) => form_b(attr, input, shape),
        SignatureKind::FormC { param_kinds, ret_kinds, ret_tuple } =>
            form_c(attr, input, param_kinds, ret_kinds, ret_tuple),
    }
}

enum SignatureKind {
    /// `fn name(x: E, ...) -> E`
    FormA,
    /// `fn name([root: &R,] x: f32, ...[, derivs: bool]) -> f32 | (f32, [f32; N])`
    /// or the same over `f64`
    FormB(FormBShape),
    /// `fn name(p: vect3sym, x: E, ...) -> <kind> | (<kind>, ...) | [E; N]`
    FormC { param_kinds: Vec<String>, ret_kinds: Vec<String>, ret_tuple: bool },
}

/// The value kinds a Form C signature may name.
pub(crate) const TYPED_KINDS: [&str; 6] =
    ["E", "vect2sym", "vect3sym", "matrix2sym", "matrix3sym", "quaternsym"];

fn classify_signature(sig: &syn::Signature, block: &syn::Block) -> syn::Result<SignatureKind> {
    let ret_ty = match &sig.output {
        syn::ReturnType::Type(_, ty) => ty.as_ref(),
        syn::ReturnType::Default => return Err(syn::Error::new_spanned(sig,
            "#[arael::function] requires an explicit return type: `E`, `f32`, or `f64`")),
    };
    let ret_name = type_last_ident(ret_ty).map(|s| s.to_string());
    let param_tys: Vec<String> = sig.inputs.iter().filter_map(|arg| {
        if let syn::FnArg::Typed(pt) = arg {
            type_last_ident(&pt.ty).map(|s| s.to_string())
        } else { None }
    }).collect();

    if sig.inputs.is_empty() {
        return Err(syn::Error::new_spanned(sig,
            "#[arael::function] requires at least one parameter"));
    }

    let single_expression = block.stmts.len() == 1
        && matches!(block.stmts[0], syn::Stmt::Expr(_, None));
    if ret_name.as_deref() == Some("E") && param_tys.len() == sig.inputs.len()
        && param_tys.iter().all(|t| t == "E") && single_expression
    {
        return Ok(SignatureKind::FormA);
    }
    if let Some(shape) = form_b_shape(sig, ret_ty)? {
        return Ok(SignatureKind::FormB(shape));
    }
    if let Some((ret_kinds, ret_tuple)) = typed_return(ret_ty)
        && param_tys.len() == sig.inputs.len()
        && param_tys.iter().all(|t| TYPED_KINDS.contains(&t.as_str()))
    {
        return Ok(SignatureKind::FormC { param_kinds: param_tys, ret_kinds, ret_tuple });
    }
    Err(syn::Error::new_spanned(sig,
        "#[arael::function] requires one of:\n  \
         - Form A: `fn name(x: E, ...) -> E` with a single-expression body\n  \
         - Form B: `fn name([root: &Root,] x: f32, ...[, derivs: bool]) -> f32 | (f32, [f32; N])` \
         (or `f64`; the scalars uniform, the root a shared reference)\n  \
         - Form C: params and result among `E`, `vect2sym`, `vect3sym`, `matrix2sym`, \
         `matrix3sym`, `quaternsym`, a tuple of them or `[E; N]`; the body may hold `let` bindings"))
}

/// The kinds a Form C result names: one for a plain type, one per
/// element for a tuple, N copies of `E` for `[E; N]`.
fn typed_return(ty: &syn::Type) -> Option<(Vec<String>, bool)> {
    let kind = |t: &syn::Type| -> Option<String> {
        let n = type_last_ident(t)?.to_string();
        TYPED_KINDS.contains(&n.as_str()).then_some(n)
    };
    match ty {
        syn::Type::Tuple(tt) if !tt.elems.is_empty() => {
            let kinds: Option<Vec<String>> = tt.elems.iter().map(kind).collect();
            kinds.map(|k| (k, true))
        }
        syn::Type::Array(ta) => {
            let elem = kind(&ta.elem)?;
            if elem != "E" { return None; }
            let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(n), .. }) = &ta.len else {
                return None;
            };
            let n: usize = n.base10_parse().ok()?;
            (n > 0).then(|| (vec![elem; n], true))
        }
        other => kind(other).map(|k| (vec![k], false)),
    }
}

fn type_last_ident(ty: &syn::Type) -> Option<&syn::Ident> {
    match ty {
        syn::Type::Path(tp) => tp.path.segments.last().map(|seg| &seg.ident),
        _ => None,
    }
}

fn param_idents(sig: &syn::Signature) -> syn::Result<Vec<syn::Ident>> {
    let mut out = Vec::new();
    for arg in &sig.inputs {
        match arg {
            syn::FnArg::Typed(pt) => match pt.pat.as_ref() {
                syn::Pat::Ident(pi) => out.push(pi.ident.clone()),
                other => return Err(syn::Error::new_spanned(other,
                    "#[arael::function] fn params must be plain identifiers")),
            },
            syn::FnArg::Receiver(r) => return Err(syn::Error::new_spanned(r,
                "#[arael::function] cannot be attached to methods")),
        }
    }
    Ok(out)
}

/// Extract a single-expression body (supports `{ expr }`, rejects statements).
fn single_expression_body(block: &syn::Block) -> syn::Result<&syn::Expr> {
    if block.stmts.len() != 1 {
        return Err(syn::Error::new_spanned(block,
            "#[arael::function] Form A body must be a single expression (no `let` bindings or statements)"));
    }
    match &block.stmts[0] {
        syn::Stmt::Expr(expr, None) => Ok(expr),
        other => Err(syn::Error::new_spanned(other,
            "#[arael::function] Form A body must be a trailing expression (no trailing `;`)")),
    }
}

/// Rewrite every `match` in a Form A body or deriv expression into the
/// parser's `select` / `select_or` call, so the stored arael-sym source
/// needs no `match` of its own. Arm rules are the constraint body's
/// (`match_arm_patterns`); an arm must be a single expression.
fn desugar_match(expr: &mut syn::Expr) -> syn::Result<()> {
    struct Desugar(Option<syn::Error>);
    impl syn::visit_mut::VisitMut for Desugar {
        fn visit_expr_mut(&mut self, e: &mut syn::Expr) {
            // Children first, so nested matches are already calls.
            syn::visit_mut::visit_expr_mut(self, e);
            let syn::Expr::Match(m) = e else { return };
            match match_to_select(m) {
                Ok(call) => *e = call,
                Err(err) => if self.0.is_none() { self.0 = Some(err) },
            }
        }
    }
    let mut v = Desugar(None);
    syn::visit_mut::VisitMut::visit_expr_mut(&mut v, expr);
    v.0.map_or(Ok(()), Err)
}

fn match_to_select(m: &syn::ExprMatch) -> syn::Result<syn::Expr> {
    let (numbered, default) = crate::constraint::match_arm_patterns(&m.arms)?;
    let mut args: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]> =
        syn::punctuated::Punctuated::new();
    args.push((*m.expr).clone());
    for i in 0..numbered.len() {
        args.push(arm_expr(&m.arms[i])?);
    }
    if let Some(d) = default {
        args.push(arm_expr(&m.arms[d])?);
    }
    let func = syn::Ident::new(
        if default.is_some() { "select_or" } else { "select" }, m.match_token.span);
    Ok(syn::parse_quote!(#func(#args)))
}

/// An arm body as one expression; a block is accepted when it holds
/// exactly one trailing expression.
fn arm_expr(arm: &syn::Arm) -> syn::Result<syn::Expr> {
    const MSG: &str = "a match arm in a #[arael::function] body must be a single expression";
    match &*arm.body {
        syn::Expr::Block(b) if b.block.stmts.len() == 1 => match &b.block.stmts[0] {
            syn::Stmt::Expr(e, None) => Ok(e.clone()),
            other => Err(syn::Error::new_spanned(other, MSG)),
        },
        syn::Expr::Block(b) => Err(syn::Error::new_spanned(b, MSG)),
        other => Ok(other.clone()),
    }
}

struct FormAAttrs {
    deriv_strings: Option<Vec<String>>,
}

struct FormBAttrs {
    sym_name: syn::Ident,
    /// `None`: a function declared without a derivative.
    deriv_strings: Option<Vec<String>>,
}

impl syn::parse::Parse for FormAAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let mut deriv_strings: Option<Vec<String>> = None;
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let _eq: syn::Token![=] = input.parse()?;
            if key == "derivs" {
                deriv_strings = Some(parse_deriv_array(input)?);
            } else {
                return Err(syn::Error::new(key.span(),
                    format!("unknown arael::function arg `{}` (expected `derivs`)", key)));
            }
            if input.is_empty() { break; }
            let _: syn::Token![,] = input.parse()?;
        }
        Ok(FormAAttrs { deriv_strings })
    }
}

impl syn::parse::Parse for FormBAttrs {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        if input.is_empty() {
            return Err(syn::Error::new(input.span(),
                "#[arael::function] on a numerical fn requires the symbolic name as first positional arg: `#[arael::function(<sym_name>, derivs = [...])]`"));
        }
        let sym_name: syn::Ident = input.parse()?;
        if !input.is_empty() {
            let _: syn::Token![,] = input.parse()?;
        }
        let mut deriv_strings: Option<Vec<String>> = None;
        while !input.is_empty() {
            let key: syn::Ident = input.parse()?;
            let _eq: syn::Token![=] = input.parse()?;
            if key == "derivs" {
                deriv_strings = Some(parse_deriv_array(input)?);
            } else {
                return Err(syn::Error::new(key.span(),
                    format!("unknown arael::function arg `{}` (expected `derivs`)", key)));
            }
            if input.is_empty() { break; }
            let _: syn::Token![,] = input.parse()?;
        }
        Ok(FormBAttrs { sym_name, deriv_strings })
    }
}

fn parse_deriv_array(input: syn::parse::ParseStream) -> syn::Result<Vec<String>> {
    let content;
    syn::bracketed!(content in input);
    let items: syn::punctuated::Punctuated<syn::Expr, syn::Token![,]> =
        content.parse_terminated(syn::Expr::parse, syn::Token![,])?;
    let mut out = Vec::with_capacity(items.len());
    for mut e in items {
        desugar_match(&mut e)?;
        out.push(quote!(#e).to_string());
    }
    Ok(out)
}

fn source_location(span: proc_macro2::Span) -> (String, u32) {
    (span.file(), span.start().line as u32)
}

// ---- Form A: fn name(x: E, ...) -> E { body } -----------------------------

fn form_a(attr: TokenStream2, input: syn::ItemFn) -> syn::Result<TokenStream2> {
    let attrs: FormAAttrs = syn::parse2(attr)?;
    let sym_name = input.sig.ident.to_string();
    let param_i = param_idents(&input.sig)?;
    let param_strs: Vec<String> = param_i.iter().map(|i| i.to_string()).collect();
    let arity = param_i.len();

    if let Some(derivs) = &attrs.deriv_strings {
        if derivs.len() != arity {
            return Err(syn::Error::new_spanned(&input.sig,
                format!("#[arael::function(derivs = [...])] on `{}`: expected {} deriv expression{}, got {}",
                    sym_name, arity, if arity == 1 { "" } else { "s" }, derivs.len())));
        }
    }

    let mut body_expr = single_expression_body(&input.block)?.clone();
    desugar_match(&mut body_expr)?;
    let body_string = quote!(#body_expr).to_string();

    let (attr_file, attr_line) = source_location(input.sig.ident.span());
    registry_store_function(&sym_name, UserFunction::Symbolic {
        sym_name: sym_name.clone(),
        param_names: param_strs.clone(),
        body: body_string.clone(),
        deriv_strings: attrs.deriv_strings.clone(),
        attr_file,
        attr_line,
    }).map_err(|msg| syn::Error::new(input.sig.ident.span(), msg))?;

    // Emit a fn whose body delegates to arael-sym's parser. Keeps
    // ordinary-Rust callability (ExtendedModel / FunctionBag users)
    // and consults the runtime user-fn registry so calls to other
    // `#[arael::function]`s in the body resolve at call time.
    let vis = &input.vis;
    let sig = &input.sig;
    let body_lit = syn::LitStr::new(&body_string, proc_macro2::Span::call_site());
    let param_lits: Vec<syn::LitStr> = param_strs.iter()
        .map(|s| syn::LitStr::new(s, proc_macro2::Span::call_site()))
        .collect();
    let sym_name_lit = syn::LitStr::new(&sym_name, proc_macro2::Span::call_site());
    let deriv_srcs_tokens: TokenStream2 = match &attrs.deriv_strings {
        None => quote! { ::core::option::Option::None },
        Some(ds) => {
            let lits: Vec<syn::LitStr> = ds.iter()
                .map(|s| syn::LitStr::new(s, proc_macro2::Span::call_site()))
                .collect();
            quote! { ::core::option::Option::Some(&[#(#lits),*]) }
        }
    };

    // When explicit derivs are given, the sibling must return a Func
    // node so differentiation honors the user's derivs instead of
    // auto-diffing the body. Without derivs we inline the body as a
    // plain E tree -- identical behavior to writing the body by hand.
    let runtime_body: TokenStream2 = if attrs.deriv_strings.is_some() {
        quote! {
            let __args: ::std::vec::Vec<::arael::sym::E> = ::std::vec![#(#param_i),*];
            ::arael::user_fn::with_registry_bag(|__bag| {
                __bag.call(#sym_name_lit, &__args)
                    .expect("#[arael::function] registry entry missing (inventory not populated?)")
                    .expect("#[arael::function] arity mismatch")
            })
        }
    } else {
        quote! {
            let __parsed = ::arael::user_fn::with_registry_bag(|__bag| {
                ::arael::sym::parse_with_functions(#body_lit, __bag)
            }).expect(concat!("#[arael::function ", stringify!(#sig), "] body parse failed"));
            let __subs: ::std::vec::Vec<(::arael::sym::E, ::arael::sym::E)> = ::std::vec![
                #( (::arael::sym::symbol(#param_lits), #param_i), )*
            ];
            __parsed.substitute(&__subs)
        }
    };

    Ok(quote! {
        ::arael::inventory::submit! {
            ::arael::user_fn::UserFnEntry {
                sym_name: #sym_name_lit,
                param_names: &[#(#param_lits),*],
                kind: ::arael::user_fn::UserFnKind::Symbolic {
                    body_src: #body_lit,
                    deriv_srcs: #deriv_srcs_tokens,
                },
            }
        }

        #vis #sig { #runtime_body }
    })
}

// ---- Form C: fn name(p: vect3sym, x: E, ...) -> kind | (kinds) | [E; N] ---

fn form_c(
    attr: TokenStream2,
    input: syn::ItemFn,
    param_kinds: Vec<String>,
    ret_kinds: Vec<String>,
    ret_tuple: bool,
) -> syn::Result<TokenStream2> {
    if !attr.is_empty() {
        return Err(syn::Error::new_spanned(attr,
            "#[arael::function] on a typed fn takes no arguments: its derivatives come from \
             the body, and `derivs` is a Form A / Form B option"));
    }
    let sym_name = input.sig.ident.to_string();
    let param_i = param_idents(&input.sig)?;
    let param_strs: Vec<String> = param_i.iter().map(|i| i.to_string()).collect();

    // The block as written, `match` included: the constraint-body
    // interpreter evaluates it at every call site.
    let block = &input.block;
    let body_string = quote!(#block).to_string();
    let (attr_file, attr_line) = source_location(input.sig.ident.span());
    registry_store_function(&sym_name, UserFunction::Typed {
        sym_name: sym_name.clone(),
        param_names: param_strs.clone(),
        param_kinds,
        ret_kinds,
        ret_tuple,
        body: body_string,
        attr_file,
        attr_line,
    }).map_err(|msg| syn::Error::new(input.sig.ident.span(), msg))?;

    // The Rust twin: the same block over the arael-sym types. Reads of
    // parameters and `let` locals are cloned, since the sym types are
    // owned values with by-value operators; builtins and the sym types'
    // constructors get their full paths; `match` lowers to `select`.
    let mut twin = (*input.block).clone();
    let mut locals: Vec<String> = param_strs.clone();
    for stmt in &twin.stmts {
        if let syn::Stmt::Local(local) = stmt {
            collect_pattern_idents(&local.pat, &mut locals);
        }
    }
    struct Twin { locals: Vec<String>, error: Option<syn::Error> }
    impl syn::visit_mut::VisitMut for Twin {
        fn visit_expr_mut(&mut self, e: &mut syn::Expr) {
            // Children first: nested matches are calls, operands cloned.
            match e {
                // The callee of a call is a function name, never a local:
                // a builtin of the body language resolves in arael-sym, a
                // constructor on a sym type too, anything else (another
                // user function) where the twin is declared.
                syn::Expr::Call(call) => {
                    for a in call.args.iter_mut() { self.visit_expr_mut(a); }
                    if let syn::Expr::Path(p) = call.func.as_mut()
                        && p.qself.is_none()
                    {
                        let segs: Vec<String> =
                            p.path.segments.iter().map(|s| s.ident.to_string()).collect();
                        let qualified = match segs.as_slice() {
                            [f] => arael_sym::function_by_name(f).is_some(),
                            [ty, _] => TYPED_KINDS.contains(&ty.as_str()),
                            _ => false,
                        };
                        if qualified && p.path.leading_colon.is_none() {
                            let path = &p.path;
                            *call.func = syn::parse_quote!(::arael::sym::#path);
                        }
                    }
                }
                _ => syn::visit_mut::visit_expr_mut(self, e),
            }
            match e {
                syn::Expr::Path(p) if p.qself.is_none() => {
                    if let Some(id) = p.path.get_ident()
                        && self.locals.iter().any(|l| id == l)
                    {
                        *e = syn::parse_quote!(::core::clone::Clone::clone(&#id));
                    }
                }
                syn::Expr::Match(m) => match match_to_rust_select(m) {
                    Ok(call) => *e = call,
                    Err(err) => if self.error.is_none() { self.error = Some(err) },
                },
                _ => {}
            }
        }
    }
    let mut v = Twin { locals, error: None };
    syn::visit_mut::VisitMut::visit_block_mut(&mut v, &mut twin);
    if let Some(err) = v.error { return Err(err); }

    // The twin's uses are inside constraint bodies, where rustc sees
    // none of them.
    let vis = &input.vis;
    let sig = &input.sig;
    let attrs = &input.attrs;
    Ok(quote! {
        #(#attrs)*
        #[allow(dead_code)]
        #vis #sig #twin
    })
}

/// Every identifier a `let` pattern binds: a name, or the names of a
/// tuple of names (`_` binds nothing).
fn collect_pattern_idents(pat: &syn::Pat, out: &mut Vec<String>) {
    match pat {
        syn::Pat::Ident(pi) => out.push(pi.ident.to_string()),
        syn::Pat::Tuple(pt) => for p in &pt.elems { collect_pattern_idents(p, out); },
        syn::Pat::Type(pt) => collect_pattern_idents(&pt.pat, out),
        syn::Pat::Paren(pp) => collect_pattern_idents(&pp.pat, out),
        _ => {}
    }
}

/// A `match` in a Form C twin as arael-sym's `select(index, arms,
/// default)` call, valid Rust over `E`.
fn match_to_rust_select(m: &syn::ExprMatch) -> syn::Result<syn::Expr> {
    let (numbered, default) = crate::constraint::match_arm_patterns(&m.arms)?;
    let scrutinee = &*m.expr;
    let arms: Vec<syn::Expr> = (0..numbered.len())
        .map(|i| arm_expr(&m.arms[i]))
        .collect::<syn::Result<_>>()?;
    let default: TokenStream2 = match default {
        Some(d) => {
            let e = arm_expr(&m.arms[d])?;
            quote!(::core::option::Option::Some(::arael::sym::E::from(#e)))
        }
        None => quote!(::core::option::Option::None),
    };
    Ok(syn::parse_quote!(::arael::sym::select(
        #scrutinee,
        ::std::vec![#(::arael::sym::E::from(#arms)),*],
        #default)))
}

// ---- Form B: fn name_eval([root: &R,] x: f32/f64, ...[, derivs: bool]) -> same | (same, [same; N])

/// What a Form B signature says: the scalar type; the root parameter
/// (its type's last segment, and whether that is a generic parameter of
/// the fn); whether a trailing `derivs: bool` is there; and the partial
/// count of a `(S, [S; N])` return.
struct FormBShape {
    scalar_ty: String,
    root: Option<(String, bool)>,
    flag: bool,
    numeric: Option<usize>,
}

/// `Some` when the signature is Form B, `None` when it is not; an error
/// when it is Form B with the flag and the tuple return not matching.
fn form_b_shape(sig: &syn::Signature, ret_ty: &syn::Type) -> syn::Result<Option<FormBShape>> {
    let inputs: Vec<&syn::PatType> = sig.inputs.iter().filter_map(|a| match a {
        syn::FnArg::Typed(pt) => Some(pt),
        syn::FnArg::Receiver(_) => None,
    }).collect();
    if inputs.len() != sig.inputs.len() {
        return Ok(None);
    }
    let scalar_name = |t: &syn::Type| -> Option<String> {
        type_last_ident(t).map(|i| i.to_string()).filter(|n| n == "f32" || n == "f64")
    };
    // The return: `S`, or `(S, [S; N])`.
    let (scalar_ty, numeric) = match ret_ty {
        syn::Type::Tuple(tt) if tt.elems.len() == 2 => {
            let Some(s) = scalar_name(&tt.elems[0]) else { return Ok(None) };
            let syn::Type::Array(arr) = &tt.elems[1] else { return Ok(None) };
            if scalar_name(&arr.elem).as_deref() != Some(s.as_str()) {
                return Ok(None);
            }
            let syn::Expr::Lit(syn::ExprLit { lit: syn::Lit::Int(n), .. }) = &arr.len else {
                return Ok(None);
            };
            let n: usize = n.base10_parse().map_err(|e| syn::Error::new(n.span(), e))?;
            (s, Some(n))
        }
        other => match scalar_name(other) {
            Some(s) => (s, None),
            None => return Ok(None),
        },
    };
    // The root: a shared reference first.
    let mut first = 0;
    let root = match inputs.first().map(|pt| pt.ty.as_ref()) {
        Some(syn::Type::Reference(r)) if r.mutability.is_none() => {
            let Some(name) = type_last_ident(&r.elem) else { return Ok(None) };
            let generic = sig.generics.type_params().any(|tp| tp.ident == *name);
            first = 1;
            Some((name.to_string(), generic))
        }
        _ => None,
    };
    // The flag: a `bool` last.
    let flag = inputs.len() > first
        && type_last_ident(&inputs[inputs.len() - 1].ty).is_some_and(|i| i == "bool");
    let last = inputs.len() - usize::from(flag);
    let scalars = &inputs[first..last];
    if !scalars.iter().all(|pt| scalar_name(&pt.ty).as_deref() == Some(scalar_ty.as_str())) {
        return Ok(None);
    }
    if scalars.is_empty() && root.is_none() {
        return Ok(None);
    }
    if flag != numeric.is_some() {
        return Err(syn::Error::new_spanned(sig, if flag {
            "a trailing `derivs: bool` goes with the tuple return `(S, [S; N])`: the value and its partials"
        } else {
            "a tuple return `(S, [S; N])` needs a trailing `derivs: bool` parameter, true when the partials are read"
        }));
    }
    Ok(Some(FormBShape { scalar_ty, root, flag, numeric }))
}

fn form_b(attr: TokenStream2, input: syn::ItemFn, shape: FormBShape) -> syn::Result<TokenStream2> {
    let FormBShape { scalar_ty, root, flag, numeric } = shape;
    let attrs: FormBAttrs = syn::parse2(attr)?;
    let eval_ident = input.sig.ident.clone();
    let eval_name = eval_ident.to_string();
    let sym_name = attrs.sym_name.to_string();
    if sym_name == eval_name {
        return Err(syn::Error::new(attrs.sym_name.span(),
            "symbolic sibling name must differ from the eval fn name (convention: <name>_eval for the eval fn, <name> for the sibling)"));
    }

    let all_params = param_idents(&input.sig)?;
    let takes_root = root.is_some();
    let scalar_params: Vec<syn::Ident> =
        all_params[usize::from(takes_root)..all_params.len() - usize::from(flag)].to_vec();
    let n_scalar = scalar_params.len();
    if let Some(n) = numeric && n != n_scalar {
        return Err(syn::Error::new_spanned(&input.sig.output, format!(
            "`[{scalar_ty}; {n}]` holds one partial per scalar parameter, and `{eval_name}` has {n_scalar}")));
    }
    if numeric.is_some() && attrs.deriv_strings.is_some() {
        return Err(syn::Error::new(attrs.sym_name.span(),
            "a fn that returns its partials takes no `derivs = [...]`"));
    }
    if let Some(d) = &attrs.deriv_strings && d.len() != n_scalar {
        return Err(syn::Error::new_spanned(&input.sig,
            format!("#[arael::function(derivs = [...])] on `{}`: expected {} deriv expression{}, got {}",
                sym_name, n_scalar, if n_scalar == 1 { "" } else { "s" }, d.len())));
    }

    // The symbolic parameters: the root first when there is one, then
    // the scalars. The flag belongs to the generated code.
    let param_i: Vec<syn::Ident> = all_params[..all_params.len() - usize::from(flag)].to_vec();
    let param_strs: Vec<String> = param_i.iter().map(|i| i.to_string()).collect();
    let arity = param_i.len();

    let (attr_file, attr_line) = source_location(attrs.sym_name.span());
    registry_store_function(&sym_name, UserFunction::Extern {
        sym_name: sym_name.clone(),
        eval_path: eval_name.clone(),
        param_names: param_strs.clone(),
        arity,
        scalar_ty: scalar_ty.clone(),
        deriv_strings: attrs.deriv_strings.clone(),
        root: root.clone(),
        numeric: numeric.is_some(),
        attr_file,
        attr_line,
    }).map_err(|msg| syn::Error::new(attrs.sym_name.span(), msg))?;

    let sym_ident = syn::Ident::new(&sym_name, proc_macro2::Span::call_site());
    let adapter_ident = syn::Ident::new(
        &format!("__arael_fn_{sym_name}_eval_adapter"),
        proc_macro2::Span::call_site(),
    );
    let vis = &input.vis;
    let param_lits: Vec<syn::LitStr> = param_strs.iter()
        .map(|s| syn::LitStr::new(s, proc_macro2::Span::call_site()))
        .collect();
    let sym_name_lit = syn::LitStr::new(&sym_name, proc_macro2::Span::call_site());
    let eval_path_lit = syn::LitStr::new(&eval_name, proc_macro2::Span::call_site());
    let arity_lit = syn::LitInt::new(&arity.to_string(), proc_macro2::Span::call_site());
    let arity_idx: Vec<syn::LitInt> = (0..arity)
        .map(|i| syn::LitInt::new(&i.to_string(), proc_macro2::Span::call_site()))
        .collect();
    let scalar_ident = syn::Ident::new(&scalar_ty, proc_macro2::Span::call_site());
    let root_lit = syn::LitBool::new(takes_root, proc_macro2::Span::call_site());

    // The scalar args from `&[f64]` to the user's f32/f64 eval fn. The
    // adapter never sees a root: there is none at runtime.
    let eval_call_args: Vec<TokenStream2> = (0..n_scalar).map(|i| {
        let idx = syn::LitInt::new(&i.to_string(), proc_macro2::Span::call_site());
        if scalar_ty == "f32" {
            quote! { (args[#idx] as f32) }
        } else {
            quote! { args[#idx] }
        }
    }).collect();
    let ret_cast: TokenStream2 = if scalar_ty == "f32" { quote! { as f64 } } else { quote! {} };
    let root_panic = syn::LitStr::new(
        &format!("arael::function `{sym_name}` reads the model root: it evaluates only inside a constraint body"),
        proc_macro2::Span::call_site());

    let eval_fn = &input;

    // The numeric shape: the eval fn returns the value and the partials.
    if numeric.is_some() {
        let adapter = if takes_root { quote! {} } else { quote! {
            #[doc(hidden)]
            fn #adapter_ident(args: &[f64]) -> (f64, ::std::vec::Vec<f64>) {
                let _scalar_check: #scalar_ident = 0.0;
                let _ = args;
                let (__v, __d) = #eval_ident( #(#eval_call_args,)* true );
                (__v as f64, __d.iter().map(|__x| *__x as f64).collect())
            }
        } };
        let eval_fn_tokens = if takes_root { quote! { None } } else { quote! { Some(#adapter_ident) } };
        return Ok(quote! {
            #eval_fn

            #adapter

            ::arael::inventory::submit! {
                ::arael::user_fn::UserFnEntry {
                    sym_name: #sym_name_lit,
                    param_names: &[#(#param_lits),*],
                    kind: ::arael::user_fn::UserFnKind::ExternNumericDerivs {
                        eval_fn: #eval_fn_tokens,
                        call_path: #eval_path_lit,
                        root: #root_lit,
                    },
                }
            }

            #[allow(non_snake_case)]
            #vis fn #sym_ident( #( #param_i : ::arael::sym::E ),* ) -> ::arael::sym::E {
                let __f = ::arael::sym::extern_func_numeric_derivs(
                    #sym_name_lit, #arity_lit, #eval_path_lit, #root_lit, #eval_fn_tokens);
                __f(::std::vec![ #( #param_i ),* ])
            }
        });
    }

    // The scalar shapes: the eval fn's value, with the attribute's
    // derivatives or none.
    let adapter_body = if takes_root {
        quote! { panic!(#root_panic) }
    } else {
        quote! { #eval_ident( #(#eval_call_args),* ) #ret_cast }
    };
    let (deriv_srcs_tokens, derivs_body) = match &attrs.deriv_strings {
        None => {
            let why = "declared without a derivative; add derivs = [...] or return partials";
            let entries: Vec<TokenStream2> = (0..arity).map(|i| {
                if takes_root && i == 0 {
                    quote! { ::arael::sym::constant(0.0) }
                } else {
                    quote! { ::arael::sym::no_derivative(#sym_name_lit, #why) }
                }
            }).collect();
            (quote! { None }, quote! {
                let _ = &__syms;
                ::std::vec![ #(#entries),* ]
            })
        }
        Some(ds) => {
            let deriv_lits: Vec<syn::LitStr> = ds.iter()
                .map(|s| syn::LitStr::new(s, proc_macro2::Span::call_site()))
                .collect();
            let root_entry = if takes_root { quote! { ::arael::sym::constant(0.0), } } else { quote! {} };
            let root_alias = if takes_root {
                quote! { (::arael::sym::symbol("root"), __syms[0].clone()), }
            } else {
                quote! {}
            };
            (quote! { Some(&[#(#deriv_lits),*]) }, quote! {
                let __bag = ::arael::user_fn::registry_bag();
                // Parse each deriv under the registry bag. Bare idents
                // naming the user's params (e.g. `x`) become free
                // `symbol("x")` nodes; rewrite them to the placeholder
                // `__p_N` syms extern_func hands us. The root answers to
                // its own name and to `root`.
                let __user_to_ph: ::std::vec::Vec<(::arael::sym::E, ::arael::sym::E)> = ::std::vec![
                    #root_alias
                    #( (::arael::sym::symbol(#param_lits), __syms[#arity_idx].clone()), )*
                ];
                ::std::vec![
                    #root_entry
                    #( ::arael::sym::parse_with_functions(#deriv_lits, &__bag)
                        .expect("#[arael::function] deriv parse failed")
                        .substitute(&__user_to_ph), )*
                ]
            })
        }
    };

    Ok(quote! {
        #eval_fn

        #[doc(hidden)]
        fn #adapter_ident(args: &[f64]) -> f64 {
            let _scalar_check: #scalar_ident = 0.0;  // static proof scalar_ty matches
            let _ = args;
            #adapter_body
        }

        ::arael::inventory::submit! {
            ::arael::user_fn::UserFnEntry {
                sym_name: #sym_name_lit,
                param_names: &[#(#param_lits),*],
                kind: ::arael::user_fn::UserFnKind::Extern {
                    deriv_srcs: #deriv_srcs_tokens,
                    eval_fn: #adapter_ident,
                    call_path: #eval_path_lit,
                    root: #root_lit,
                },
            }
        }

        #[allow(non_snake_case)]
        #vis fn #sym_ident( #( #param_i : ::arael::sym::E ),* ) -> ::arael::sym::E {
            let __f = ::arael::sym::extern_func(
                #sym_name_lit, #arity_lit, #eval_path_lit,
                move |__syms: ::std::vec::Vec<::arael::sym::E>| -> ::std::vec::Vec<::arael::sym::E> {
                    #derivs_body
                },
                #adapter_ident,
            );
            __f(::std::vec![ #( #param_i ),* ])
        }
    })
}
