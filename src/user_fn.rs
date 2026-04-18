//! Runtime registry for user-defined functions declared via
//! `#[arael::function]`. Populated at program start via [`inventory`],
//! consulted by the emitted sibling fns so calls like `sigmoid(x)` or
//! `my_safe_asin(x)` from ordinary Rust can resolve every other
//! registered user function (including forward references and mutual
//! recursion).
//!
//! The compile-time `#[arael::model]` interpreter has its own registry
//! inside `arael-macros`; this module mirrors that at runtime for
//! callers outside the constraint-body path (e.g. `ExtendedModel`
//! bodies that build `E` trees from user fns).

use std::cell::OnceCell;

use arael_sym::{E, FuncKind, FunctionBag, constant, parse_with_functions, symbol};

/// One entry per `#[arael::function]` declaration, submitted into the
/// inventory at program start. The `&'static` fields come from literals
/// the macro expands into.
pub struct UserFnEntry {
    pub sym_name: &'static str,
    pub param_names: &'static [&'static str],
    pub kind: UserFnKind,
}

pub enum UserFnKind {
    /// Form A: `fn name(x: E, ...) -> E { body }`. `body_src` is the
    /// arael-sym source string captured at attribute expansion.
    Symbolic {
        body_src: &'static str,
        /// Explicit derivative strings, one per parameter. `None` means
        /// auto-diff the body at use time.
        deriv_srcs: Option<&'static [&'static str]>,
    },
    /// Form B: `fn name_eval(x: f32 | f64, ...) -> <same>`. `call_path`
    /// is the eval fn's name (resolved in the user's crate); `eval_fn`
    /// is a pointer to a shim that adapts the scalar signature to
    /// `fn(&[f64]) -> f64`.
    Extern {
        deriv_srcs: &'static [&'static str],
        eval_fn: fn(&[f64]) -> f64,
        call_path: &'static str,
    },
}

inventory::collect!(UserFnEntry);

thread_local! {
    static REGISTRY_BAG: OnceCell<FunctionBag> = const { OnceCell::new() };
}

/// Run `f` with a reference to the thread-local `FunctionBag` populated
/// from every `#[arael::function]` declaration in the current binary.
/// Built once per thread on first access (arael-sym's `E` is `Rc`, so
/// a single process-wide cache would require `Arc`).
pub fn with_registry_bag<R>(f: impl FnOnce(&FunctionBag) -> R) -> R {
    REGISTRY_BAG.with(|slot| f(slot.get_or_init(build_registry_bag)))
}

/// Clone of the thread-local registry bag. Callers that need to add
/// their own bindings (e.g. parameter -> arg mappings) should clone and
/// extend this bag rather than mutating the shared one.
pub fn registry_bag() -> FunctionBag {
    with_registry_bag(|b| b.clone())
}

/// Two-pass build: pass 1 registers every entry with a placeholder so
/// cross-references resolve at parse time; pass 2 parses each body /
/// deriv under the fully-populated bag and replaces the placeholder.
/// Mirrors `arael-macros::constraint::build_user_function_bag` for the
/// compile-time path.
fn build_registry_bag() -> FunctionBag {
    fn zero_derivs(arity: usize) -> Vec<E> {
        (0..arity).map(|_| constant(0.0)).collect()
    }

    let entries: Vec<&'static UserFnEntry> =
        inventory::iter::<UserFnEntry>.into_iter().collect();
    let mut bag = FunctionBag::new();

    // Pass 1: stubs. The stub body / derivs are irrelevant for dispatch;
    // the parser only uses them to form a Func node with the matching
    // name + param count. Pass 2 replaces the entries with the real
    // parsed bodies.
    for e in &entries {
        let params: Vec<String> = e.param_names.iter().map(|s| s.to_string()).collect();
        match &e.kind {
            UserFnKind::Symbolic { .. } => {
                bag.add_symbolic(e.sym_name.to_string(), params, symbol(e.sym_name));
            }
            UserFnKind::Extern { eval_fn, call_path, deriv_srcs } => {
                bag.add_with_kind(
                    e.sym_name.to_string(),
                    params,
                    FuncKind::Extern {
                        derivs: zero_derivs(deriv_srcs.len()),
                        eval_fn: *eval_fn,
                        call_path: call_path.to_string(),
                    },
                );
            }
        }
    }

    // Pass 2: real parse under the stubbed bag.
    for e in &entries {
        match &e.kind {
            UserFnKind::Symbolic { body_src, deriv_srcs } => {
                let body = parse_with_functions(body_src, &bag).unwrap_or_else(|err| {
                    panic!(
                        "#[arael::function `{}`] body parse failed at runtime: {}",
                        e.sym_name, err
                    )
                });
                let params: Vec<String> =
                    e.param_names.iter().map(|s| s.to_string()).collect();
                let kind = match deriv_srcs {
                    None => FuncKind::Symbolic { body },
                    Some(ds) => {
                        let derivs: Vec<E> = ds
                            .iter()
                            .map(|s| {
                                parse_with_functions(s, &bag).unwrap_or_else(|err| {
                                    panic!(
                                        "#[arael::function `{}`] deriv parse failed at runtime: {}",
                                        e.sym_name, err
                                    )
                                })
                            })
                            .collect();
                        FuncKind::SymbolicDerivs { body, derivs }
                    }
                };
                bag.add_with_kind(e.sym_name.to_string(), params, kind);
            }
            UserFnKind::Extern { deriv_srcs, eval_fn, call_path } => {
                // Parse each deriv against the stubbed bag; free symbols
                // for parameter names become placeholders that the Extern
                // funckind substitutes with actual args at chain-rule
                // time. The arael-sym Extern convention is to use
                // `__p0`, `__p1`, ... as placeholders, so rewrite the
                // user's own parameter names to match.
                let user_to_placeholder: Vec<(E, E)> = e
                    .param_names
                    .iter()
                    .enumerate()
                    .map(|(i, p)| (symbol(p), symbol(&format!("__p{i}"))))
                    .collect();
                let derivs: Vec<E> = deriv_srcs
                    .iter()
                    .map(|s| {
                        parse_with_functions(s, &bag)
                            .unwrap_or_else(|err| {
                                panic!(
                                    "#[arael::function `{}`] deriv parse failed at runtime: {}",
                                    e.sym_name, err
                                )
                            })
                            .substitute(&user_to_placeholder)
                    })
                    .collect();
                let placeholder_params: Vec<String> =
                    (0..e.param_names.len()).map(|i| format!("__p{i}")).collect();
                bag.add_with_kind(
                    e.sym_name.to_string(),
                    placeholder_params,
                    FuncKind::Extern {
                        derivs,
                        eval_fn: *eval_fn,
                        call_path: call_path.to_string(),
                    },
                );
            }
        }
    }
    bag
}
