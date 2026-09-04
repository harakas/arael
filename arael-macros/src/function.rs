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
//! No `.clone()` insertion: body / deriv tokens are stringified verbatim
//! and handed to `arael_sym::parse_with_functions` at use time, which has
//! no Rust move semantics.

use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::parse::Parse;

use crate::{registry_store_function, UserFunction};

/// Entry point. Dispatch on signature, then form-specific handling.
pub fn function_attribute(attr: TokenStream2, item: TokenStream2) -> syn::Result<TokenStream2> {
    let input: syn::ItemFn = syn::parse2(item)
        .map_err(|e| syn::Error::new(e.span(),
            format!("#[arael::function] must be attached to a fn: {}", e)))?;
    match classify_signature(&input.sig)? {
        SignatureKind::FormA => form_a(attr, input),
        SignatureKind::FormB { scalar_ty } => form_b(attr, input, scalar_ty),
    }
}

enum SignatureKind {
    /// `fn name(x: E, ...) -> E`
    FormA,
    /// `fn name(x: f32, ...) -> f32` or `f64`
    FormB { scalar_ty: String },
}

fn classify_signature(sig: &syn::Signature) -> syn::Result<SignatureKind> {
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

    if param_tys.is_empty() {
        return Err(syn::Error::new_spanned(sig,
            "#[arael::function] requires at least one parameter"));
    }

    if ret_name.as_deref() == Some("E") && param_tys.iter().all(|t| t == "E") {
        return Ok(SignatureKind::FormA);
    }
    if let Some(r) = ret_name.as_deref() {
        if (r == "f32" || r == "f64") && param_tys.iter().all(|t| t == r) {
            return Ok(SignatureKind::FormB { scalar_ty: r.to_string() });
        }
    }
    Err(syn::Error::new_spanned(sig,
        "#[arael::function] requires one of:\n  \
         - Form A: `fn name(x: E, ...) -> E`\n  \
         - Form B: `fn name(x: f32, ...) -> f32` (or `f64`, all params and return type uniform)"))
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
    deriv_strings: Vec<String>,
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
        let deriv_strings = deriv_strings.ok_or_else(|| syn::Error::new(sym_name.span(),
            "#[arael::function] on a numerical eval fn requires `derivs = [...]`"))?;
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
    (format!("{:?}", span), span.start().line as u32)
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
    });

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

// ---- Form B: fn name_eval(x: f32/f64, ...) -> same ------------------------

fn form_b(attr: TokenStream2, input: syn::ItemFn, scalar_ty: String) -> syn::Result<TokenStream2> {
    let attrs: FormBAttrs = syn::parse2(attr)?;
    let eval_ident = input.sig.ident.clone();
    let eval_name = eval_ident.to_string();
    let sym_name = attrs.sym_name.to_string();
    if sym_name == eval_name {
        return Err(syn::Error::new(attrs.sym_name.span(),
            "symbolic sibling name must differ from the eval fn name (convention: <name>_eval for the eval fn, <name> for the sibling)"));
    }

    let param_i = param_idents(&input.sig)?;
    let param_strs: Vec<String> = param_i.iter().map(|i| i.to_string()).collect();
    let arity = param_i.len();

    if attrs.deriv_strings.len() != arity {
        return Err(syn::Error::new_spanned(&input.sig,
            format!("#[arael::function(derivs = [...])] on `{}`: expected {} deriv expression{}, got {}",
                sym_name, arity, if arity == 1 { "" } else { "s" }, attrs.deriv_strings.len())));
    }

    let (attr_file, attr_line) = source_location(attrs.sym_name.span());
    registry_store_function(&sym_name, UserFunction::Extern {
        sym_name: sym_name.clone(),
        eval_path: eval_name.clone(),
        param_names: param_strs.clone(),
        arity,
        scalar_ty: scalar_ty.clone(),
        deriv_strings: attrs.deriv_strings.clone(),
        attr_file,
        attr_line,
    });

    let sym_ident = syn::Ident::new(&sym_name, proc_macro2::Span::call_site());
    let adapter_ident = syn::Ident::new(
        &format!("__arael_fn_{sym_name}_eval_adapter"),
        proc_macro2::Span::call_site(),
    );
    let vis = &input.vis;
    let deriv_lits: Vec<syn::LitStr> = attrs.deriv_strings.iter()
        .map(|s| syn::LitStr::new(s, proc_macro2::Span::call_site()))
        .collect();
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

    // Build args from `&[f64]` to the user's f32/f64 eval fn.
    let eval_call_args: Vec<TokenStream2> = (0..arity).map(|i| {
        let idx = syn::LitInt::new(&i.to_string(), proc_macro2::Span::call_site());
        if scalar_ty == "f32" {
            quote! { (args[#idx] as f32) }
        } else {
            quote! { args[#idx] }
        }
    }).collect();
    let ret_cast: TokenStream2 = if scalar_ty == "f32" { quote! { as f64 } } else { quote! {} };

    let eval_fn = &input;

    Ok(quote! {
        #eval_fn

        #[doc(hidden)]
        fn #adapter_ident(args: &[f64]) -> f64 {
            let _scalar_check: #scalar_ident = 0.0;  // static proof scalar_ty matches
            let _ = args;  // silence unused when arity=0 (impossible, but safe)
            #eval_ident( #(#eval_call_args),* ) #ret_cast
        }

        ::arael::inventory::submit! {
            ::arael::user_fn::UserFnEntry {
                sym_name: #sym_name_lit,
                param_names: &[#(#param_lits),*],
                kind: ::arael::user_fn::UserFnKind::Extern {
                    deriv_srcs: &[#(#deriv_lits),*],
                    eval_fn: #adapter_ident,
                    call_path: #eval_path_lit,
                },
            }
        }

        #[allow(non_snake_case)]
        #vis fn #sym_ident( #( #param_i : ::arael::sym::E ),* ) -> ::arael::sym::E {
            let __f = ::arael::sym::extern_func(
                #sym_name_lit, #arity_lit, #eval_path_lit,
                move |__syms: ::std::vec::Vec<::arael::sym::E>| -> ::std::vec::Vec<::arael::sym::E> {
                    let __bag = ::arael::user_fn::registry_bag();
                    // Parse each deriv under the registry bag. Bare
                    // idents naming the user's params (e.g. `x`) become
                    // free `symbol("x")` nodes; rewrite them to the
                    // placeholder `__p_N` syms extern_func hands us.
                    let __user_to_ph: ::std::vec::Vec<(::arael::sym::E, ::arael::sym::E)> = ::std::vec![
                        #( (::arael::sym::symbol(#param_lits), __syms[#arity_idx].clone()), )*
                    ];
                    ::std::vec![
                        #( ::arael::sym::parse_with_functions(#deriv_lits, &__bag)
                            .expect("#[arael::function] deriv parse failed")
                            .substitute(&__user_to_ph), )*
                    ]
                },
                #adapter_ident,
            );
            __f(::std::vec![ #( #param_i ),* ])
        }
    })
}
