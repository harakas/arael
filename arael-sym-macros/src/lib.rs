//! Proc macros for the [`arael-sym`](https://docs.rs/arael-sym) symbolic math crate.
//!
//! This crate exists solely to provide the [`sym!`] macro, which is
//! re-exported by `arael-sym`. You should depend on `arael-sym` rather
//! than this crate directly.
//!
//! The macro auto-inserts `.clone()` on reused `let`-bound variables,
//! eliminating ownership boilerplate when building symbolic expressions.

use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse_macro_input,
    visit_mut::VisitMut,
    Block, Expr, ExprBlock, ExprForLoop, ExprPath, Local, Pat, Stmt,
};

/// Auto-clone macro for symbolic math expressions.
///
/// Walks a block of code and inserts `.clone()` on every reuse of a
/// `let`-bound variable. Since `E` wraps `Rc<Expr>`, cloning is cheap
/// (just a reference count bump), but writing `.clone()` everywhere
/// obscures the math. This macro lets you write natural expressions:
///
/// ```ignore
/// arael_sym::sym! {
///     let x = symbol("x");
///     let f = sin(x) * x + 1.0;  // x is auto-cloned
///     println!("{}", f.diff("x"));
/// }
/// ```
#[proc_macro]
pub fn sym(input: TokenStream) -> TokenStream {
    let block: Block = parse_macro_input!(input with parse_block_body);
    let mut visitor = SymVisitor::new();
    let mut block = block;
    visitor.visit_block_mut(&mut block);
    let stmts = &block.stmts;
    let output = quote! { { #(#stmts)* } };
    output.into()
}

/// Parse the macro input as the *body* of a block (list of statements),
/// wrapping it in a Block so syn can handle it.
fn parse_block_body(input: syn::parse::ParseStream) -> syn::Result<Block> {
    let stmts = Block::parse_within(input)?;
    Ok(Block {
        brace_token: syn::token::Brace::default(),
        stmts,
    })
}

// ---------------------------------------------------------------------------
// Visitor
// ---------------------------------------------------------------------------

struct SymVisitor {
    /// Stack of scopes. Each scope is a set of variable names tracked for cloning.
    scopes: Vec<Vec<String>>,
}

impl SymVisitor {
    fn new() -> Self {
        SymVisitor {
            scopes: vec![vec![]],
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(vec![]);
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn track(&mut self, name: &str) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.push(name.to_string());
        }
    }

    fn is_tracked(&self, name: &str) -> bool {
        self.scopes.iter().any(|scope| scope.contains(&String::from(name)))
    }

    /// Extract variable names from a pattern (handles ident, tuple, struct destructuring, etc.)
    fn collect_pat_idents(&self, pat: &Pat) -> Vec<String> {
        let mut names = vec![];
        collect_pat_idents_into(pat, &mut names);
        names
    }

    /// Named constants that get rewritten from bare identifiers to function calls.
    /// Named constants rewritten from bare identifiers to function calls.
    /// Note: "e" is NOT included because it's too common as a variable name.
    /// Use euler() explicitly for Euler's number in sym! blocks.
    const NAMED_CONSTS: &'static [(&'static str, &'static str)] = &[
        ("pi", "pi"), ("epsilon", "epsilon"),
    ];

    /// Check if an expression is a path to a named constant (to avoid double-rewriting in calls).
    fn is_named_const_path(expr: &Expr) -> bool {
        if let Expr::Path(ExprPath { path, qself: None, attrs }) = expr
            && attrs.is_empty()
                && let Some(ident) = path.get_ident() {
                    let name = ident.to_string();
                    return Self::NAMED_CONSTS.iter().any(|&(n, f)| name == n || name == f);
                }
        false
    }

    /// Clone-wrap an expression if it's a tracked variable path,
    /// or rewrite named constants (pi, epsilon, e) to function calls.
    fn maybe_clone_expr(&self, expr: &mut Expr) {
        if let Expr::Path(ExprPath { path, qself: None, attrs }) = expr
            && attrs.is_empty()
                && let Some(ident) = path.get_ident() {
                    let name = ident.to_string();
                    // Rewrite named constants: pi -> pi(), e -> euler(), etc.
                    for &(const_name, func_name) in Self::NAMED_CONSTS {
                        if name == const_name {
                            let span = ident.span();
                            let func_ident = proc_macro2::Ident::new(func_name, span);
                            let call_expr: Expr = syn::parse_quote_spanned! { span =>
                                #func_ident()
                            };
                            *expr = call_expr;
                            return;
                        }
                    }
                    if self.is_tracked(&name) {
                        let span = ident.span();
                        let clone_expr: Expr = syn::parse_quote_spanned! { span =>
                            #ident.clone()
                        };
                        *expr = clone_expr;
                    }
                }
    }

    /// Process statements in order: for each `let`, visit the init expr first
    /// (to clone already-tracked vars), then register the new binding.
    fn process_stmts(&mut self, stmts: &mut Vec<Stmt>) {
        let mut i = 0;
        while i < stmts.len() {
            match &mut stmts[i] {
                Stmt::Local(local) => {
                    // Visit the init expression with current tracking
                    if let Some(init) = &mut local.init {
                        self.visit_expr_mut(&mut init.expr);
                        if let Some((_, diverge)) = &mut init.diverge {
                            self.visit_expr_mut(diverge);
                        }
                    }
                    // Now track the new bindings
                    let names = self.collect_pat_idents(&local.pat);
                    for name in &names {
                        if name != "_" {
                            self.track(name);
                        }
                    }
                }
                Stmt::Expr(expr, _semi) => {
                    self.visit_expr_mut(expr);
                }
                Stmt::Item(_item) => {
                    // Don't descend into items (fn, struct, etc.)
                }
                Stmt::Macro(stmt_macro) => {
                    if stmt_macro.mac.path.is_ident("symbols") {
                        // See the matching guard in visit_expr_mut.
                    } else {
                        self.process_macro_tokens(&mut stmt_macro.mac.tokens);
                    }
                }
            }
            i += 1;
        }
    }

    /// Process tokens inside a macro invocation, cloning tracked variables.
    fn process_macro_tokens(&mut self, tokens: &mut TokenStream2) {
        let new_tokens = self.rewrite_token_stream(tokens.clone());
        *tokens = new_tokens;
    }

    /// Rewrite a token stream, replacing tracked idents with ident.clone()
    /// but being smart about format strings and punctuation.
    fn rewrite_token_stream(&self, tokens: TokenStream2) -> TokenStream2 {
        use proc_macro2::{Delimiter, TokenTree};

        let token_vec: Vec<TokenTree> = tokens.into_iter().collect();
        let mut result = TokenStream2::new();
        let len = token_vec.len();
        let mut i = 0;

        while i < len {
            match &token_vec[i] {
                TokenTree::Ident(ident) => {
                    let name = ident.to_string();
                    let next_is_bang = i + 1 < len && matches!(&token_vec[i + 1], TokenTree::Punct(p) if p.as_char() == '!');
                    let next_is_colon = i + 1 < len && matches!(&token_vec[i + 1], TokenTree::Punct(p) if p.as_char() == ':');
                    let is_keyword = matches!(name.as_str(),
                        "let" | "mut" | "ref" | "if" | "else" | "while" | "for" | "in" |
                        "loop" | "match" | "return" | "break" | "continue" | "fn" |
                        "struct" | "enum" | "impl" | "trait" | "type" | "use" | "mod" |
                        "pub" | "self" | "super" | "crate" | "as" | "const" | "static" |
                        "extern" | "move" | "async" | "await" | "dyn" | "true" | "false" |
                        "where" | "unsafe"
                    );

                    // Rewrite named constants in macro tokens (skip if already called)
                    let next_is_paren = i + 1 < len && matches!(&token_vec[i + 1], TokenTree::Group(g) if g.delimiter() == Delimiter::Parenthesis);
                    let mut is_named_const = false;
                    if !is_keyword && !next_is_bang && !next_is_colon && !next_is_paren {
                        for &(const_name, func_name) in Self::NAMED_CONSTS {
                            if name == const_name {
                                let span = ident.span();
                                let func_ident = proc_macro2::Ident::new(func_name, span);
                                result.extend(std::iter::once(TokenTree::Ident(func_ident)));
                                result.extend(std::iter::once(TokenTree::Group(proc_macro2::Group::new(
                                    Delimiter::Parenthesis,
                                    TokenStream2::new(),
                                ))));
                                is_named_const = true;
                                break;
                            }
                        }
                    }
                    if is_named_const {
                        // already handled
                    } else if !is_keyword && !next_is_bang && !next_is_colon && self.is_tracked(&name) {
                        let span = ident.span();
                        let clone_ident = proc_macro2::Ident::new("clone", span);
                        result.extend(std::iter::once(TokenTree::Ident(ident.clone())));
                        result.extend(std::iter::once(TokenTree::Punct(proc_macro2::Punct::new('.', proc_macro2::Spacing::Alone))));
                        result.extend(std::iter::once(TokenTree::Ident(clone_ident)));
                        result.extend(std::iter::once(TokenTree::Group(proc_macro2::Group::new(
                            Delimiter::Parenthesis,
                            TokenStream2::new(),
                        ))));
                    } else {
                        result.extend(std::iter::once(TokenTree::Ident(ident.clone())));
                    }
                }
                TokenTree::Group(group) => {
                    let new_inner = self.rewrite_token_stream(group.stream());
                    let mut new_group = proc_macro2::Group::new(group.delimiter(), new_inner);
                    new_group.set_span(group.span());
                    result.extend(std::iter::once(TokenTree::Group(new_group)));
                }
                other => {
                    result.extend(std::iter::once(other.clone()));
                }
            }
            i += 1;
        }

        result
    }

    /// Visit an expression but don't clone direct variable references at the top level.
    /// Used for `&x` -- we don't want `&x.clone()`.
    fn visit_expr_no_clone(&mut self, expr: &mut Expr) {
        match expr {
            Expr::Path(_) => {
                // Don't clone -- this is behind a &
            }
            _ => {
                self.visit_expr_mut(expr);
            }
        }
    }
}

impl VisitMut for SymVisitor {
    fn visit_block_mut(&mut self, block: &mut Block) {
        self.push_scope();
        self.process_stmts(&mut block.stmts);
        self.pop_scope();
    }

    fn visit_expr_mut(&mut self, expr: &mut Expr) {
        match expr {
            Expr::Path(_) => { self.maybe_clone_expr(expr); }
            Expr::Block(ExprBlock { block, .. }) => { self.visit_block_mut(block); }
            Expr::ForLoop(ExprForLoop { pat, expr: iter_expr, body, .. }) => {
                self.visit_expr_mut(iter_expr);
                self.push_scope();
                for name in self.collect_pat_idents(pat) {
                    if name != "_" { self.track(&name); }
                }
                self.process_stmts(&mut body.stmts);
                self.pop_scope();
            }
            Expr::Closure(closure) => {
                self.push_scope();
                for input in &closure.inputs {
                    for name in self.collect_pat_idents(input) {
                        if name != "_" { self.track(&name); }
                    }
                }
                self.visit_expr_mut(&mut closure.body);
                self.pop_scope();
            }
            Expr::If(expr_if) => {
                self.visit_expr_mut(&mut expr_if.cond);
                self.visit_block_mut(&mut expr_if.then_branch);
                if let Some((_, else_branch)) = &mut expr_if.else_branch {
                    self.visit_expr_mut(else_branch);
                }
            }
            Expr::While(expr_while) => {
                self.visit_expr_mut(&mut expr_while.cond);
                self.visit_block_mut(&mut expr_while.body);
            }
            Expr::Loop(expr_loop) => { self.visit_block_mut(&mut expr_loop.body); }
            Expr::Match(expr_match) => {
                self.visit_expr_mut(&mut expr_match.expr);
                for arm in &mut expr_match.arms {
                    self.push_scope();
                    for name in self.collect_pat_idents(&arm.pat) {
                        if name != "_" { self.track(&name); }
                    }
                    if let Some((_, guard_expr)) = &mut arm.guard {
                        self.visit_expr_mut(guard_expr);
                    }
                    self.visit_expr_mut(&mut arm.body);
                    self.pop_scope();
                }
            }
            Expr::MethodCall(mc) => {
                self.visit_expr_mut(&mut mc.receiver);
                for arg in &mut mc.args { self.visit_expr_mut(arg); }
            }
            Expr::Call(call) => {
                // Don't rewrite named constants in function position (they're already being called)
                if !SymVisitor::is_named_const_path(&call.func) {
                    self.visit_expr_mut(&mut call.func);
                }
                for arg in &mut call.args { self.visit_expr_mut(arg); }
            }
            Expr::Binary(bin) => {
                self.visit_expr_mut(&mut bin.left);
                self.visit_expr_mut(&mut bin.right);
            }
            Expr::Unary(un) => { self.visit_expr_mut(&mut un.expr); }
            Expr::Paren(paren) => { self.visit_expr_mut(&mut paren.expr); }
            Expr::Field(field) => { self.visit_expr_mut(&mut field.base); }
            Expr::Index(idx) => {
                self.visit_expr_mut(&mut idx.expr);
                self.visit_expr_mut(&mut idx.index);
            }
            Expr::Reference(reference) => { self.visit_expr_no_clone(&mut reference.expr); }
            Expr::Tuple(tuple) => { for elem in &mut tuple.elems { self.visit_expr_mut(elem); } }
            Expr::Array(arr) => { for elem in &mut arr.elems { self.visit_expr_mut(elem); } }
            Expr::Return(ret) => { if let Some(expr) = &mut ret.expr { self.visit_expr_mut(expr); } }
            Expr::Assign(assign) => { self.visit_expr_mut(&mut assign.right); }
            Expr::Range(range) => {
                if let Some(start) = &mut range.start { self.visit_expr_mut(start); }
                if let Some(end) = &mut range.end { self.visit_expr_mut(end); }
            }
            Expr::Struct(expr_struct) => {
                for field in &mut expr_struct.fields { self.visit_expr_mut(&mut field.expr); }
                if let Some(rest) = &mut expr_struct.rest { self.visit_expr_mut(rest); }
            }
            Expr::Cast(cast) => { self.visit_expr_mut(&mut cast.expr); }
            Expr::Let(expr_let) => { self.visit_expr_mut(&mut expr_let.expr); }
            Expr::Macro(expr_macro) => {
                // `symbols!(a, b, c)` takes bare identifiers as names,
                // not variable uses -- don't rewrite them into
                // `a.clone()`, which would break the macro's ident
                // pattern. Same rule applies to any future
                // ident-argument helpers we add.
                if expr_macro.mac.path.is_ident("symbols") { return; }
                self.process_macro_tokens(&mut expr_macro.mac.tokens);
            }
            Expr::Repeat(repeat) => {
                self.visit_expr_mut(&mut repeat.expr);
                self.visit_expr_mut(&mut repeat.len);
            }
            Expr::Unsafe(expr_unsafe) => { self.visit_block_mut(&mut expr_unsafe.block); }
            Expr::Try(expr_try) => { self.visit_expr_mut(&mut expr_try.expr); }
            Expr::Yield(expr_yield) => { if let Some(expr) = &mut expr_yield.expr { self.visit_expr_mut(expr); } }
            _ => { syn::visit_mut::visit_expr_mut(self, expr); }
        }
    }

    fn visit_local_mut(&mut self, _local: &mut Local) {
        // Handled by process_stmts
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_pat_idents_into(pat: &Pat, names: &mut Vec<String>) {
    match pat {
        Pat::Ident(pi) => { names.push(pi.ident.to_string()); }
        Pat::Tuple(pt) => { for elem in &pt.elems { collect_pat_idents_into(elem, names); } }
        Pat::TupleStruct(pts) => { for elem in &pts.elems { collect_pat_idents_into(elem, names); } }
        Pat::Struct(ps) => { for field in &ps.fields { collect_pat_idents_into(&field.pat, names); } }
        Pat::Slice(ps) => { for elem in &ps.elems { collect_pat_idents_into(elem, names); } }
        Pat::Reference(pr) => { collect_pat_idents_into(&pr.pat, names); }
        Pat::Or(por) => { for case in &por.cases { collect_pat_idents_into(case, names); } }
        Pat::Paren(pp) => { collect_pat_idents_into(&pp.pat, names); }
        Pat::Wild(_) | Pat::Lit(_) | Pat::Rest(_) | Pat::Const(_) => {}
        _ => {}
    }
}
