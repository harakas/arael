//! Procedural macros for the arael optimization framework.
//!
//! This crate provides the `#[arael::model]` attribute macro and the
//! `#[derive(Model)]` derive macro.
//!
//! ## `#[arael::model]`
//!
//! Applied to a struct, this attribute macro:
//!
//! - Generates the `Model` trait implementation (serialize, deserialize, update,
//!   Hessian-block accumulation) by inspecting `Param<T>`, `SimpleEulerAngleParam`,
//!   and `EulerAngleParam` fields.
//! - Rewrites shorthand block types: `SelfBlock<A>` becomes
//!   `SelfBlock<A, {A_PARAM_COUNT}>`, and `CrossBlock<A, B>` becomes
//!   `CrossBlock<A, B, {A_PARAM_COUNT}, {B_PARAM_COUNT}>` (two const
//!   generics — NA and NB are stored separately so the cross Hessian is
//!   a rectangular NA*NB block).
//!   `TripletBlock` is passed through as-is (COO sparse, for 3+ entity constraints).
//! - Requires every params-having struct to declare exactly one
//!   `SelfBlock<Self>` field (the canonical home for its gradient +
//!   within-entity Hessian diagonal). Exemptions: `#[arael(fit(...))]`
//!   (auto-skipped — fit generates its own LmProblem) and
//!   `#[arael(skip_self_block)]` (explicit opt-out for bag-of-params
//!   structs whose params are written only by a parent's ExtendedModel).
//! - Detects `SimpleEulerAngleParam` and `EulerAngleParam` fields by type
//!   name and generates appropriate precompute calls for rotation matrices.
//! - Generates the symbolic companion struct (`FooSym`) and `ModelSym` impl.
//!
//! ## `#[arael(root)]`
//!
//! When placed on the root model struct, triggers code generation for all
//! stashed `#[arael(constraint(...))]` attributes: symbolic differentiation,
//! CSE, and emission of `LmProblem` trait methods.

mod constraint;
mod function;

use std::collections::{HashSet, HashMap};
use std::sync::Mutex;
use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    parse_macro_input,
    Expr, Pat, Stmt,
};

// ---------------------------------------------------------------------------
// Sym field registry — shared state between #[arael::model] invocations
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
enum SymFieldType {
    Scalar,
    Vec2,
    Vec3,
    Mat2,
    Mat3,
    Struct(String),        // reference to another registered struct
    OptionalStruct(String), // Option<T> wrapping a struct
    Skip,
}

#[derive(Clone, Debug)]
struct SymLayout {
    fields: Vec<(String, SymFieldType)>,
    param_fields: Vec<String>,       // field names that are Param<T>
    ref_paths: Vec<(String, String)>, // (field_name, resolution_path) for #[arael(ref = ...)]
    euler_angle_fields: Vec<String>,  // field names detected as SimpleEulerAngleParam
    universal_euler_angle_fields: Vec<String>, // field names detected as EulerAngleParam
    /// Symbolic substitutions: (from_expr, to_expr) applied before CSE.
    /// Stored as string pairs for thread-safe registry storage.
    #[allow(dead_code)]
    substitutions: Vec<(String, String)>, // (from_sym_str, to_sym_str)
    /// Field name of `#[arael(constraint_index)]` u32 field, if present.
    constraint_index_field: Option<String>,
    /// Field name of the struct's `SelfBlock<Self>` — detected automatically
    /// during `#[arael::model]` expansion. Required for every params-having
    /// Model after the CrossBlock/TripletBlock refactor: the self-block is
    /// the single home for that entity's gradient + A-A Hessian diagonal,
    /// so cross constraints need to know the field name to write to it.
    self_block_field: Option<String>,
}

/// Stashed constraint: struct name + raw attribute tokens + source-location
/// metadata, waiting for root to generate code.
///
/// Spans themselves are not stashed: `proc_macro2::Span` is backed by an Rc
/// into the proc-macro bridge's handle table that is invalidated once the
/// originating macro invocation returns (tried; rustc panics). But primitive
/// location DATA (file path, line number) survives fine — we extract it at
/// stash time via proc-macro2's `span-locations` feature and carry it as
/// plain strings/u32s. Used both to prefix error messages with
/// `file:line:` and to emit `arael: <label> @ file:line` markers into
/// generated code so `cargo expand` is easy to navigate.
#[derive(Clone)]
struct StashedConstraint {
    struct_name: String,
    attr_file: String,           // source file of the `#[arael(constraint(...))]` attribute
    attr_line: u32,
    label_hint: String,          // `name = "..."` if given, else struct name
    attr_tokens: String,         // serialized constraint attribute content
    fields_tokens: String,       // serialized struct fields
}

/// User-defined function registered via `#[arael::function]`. Carried in
/// the process-wide macro registry so the constraint-body interpreter can
/// dispatch calls like `elliptic_k(k)` to arael-sym's parser with an
/// appropriate `FunctionBag`.
#[derive(Clone)]
// Dead-code analysis ignores usage via the derived Clone impl, so the
// fields below appear unread even though they are read when an
// UserFunction is cloned out of the registry. Silence the lint.
#[allow(dead_code)]
pub(crate) enum UserFunction {
    /// Form A: attribute sits on `fn <sym_name>(x: E, ...) -> E`. Body is
    /// an arael-sym expression, captured as a source string. Derivatives
    /// are optional -- when present, they override auto-diff from the body.
    Symbolic {
        sym_name: String,
        param_names: Vec<String>,
        body: String,            // arael-sym source, `TokenStream::to_string()`
        deriv_strings: Option<Vec<String>>, // None -> auto-diff from body
        attr_file: String,
        attr_line: u32,
    },
    /// Form B: attribute sits on `fn <eval_name>(x: f32 | f64, ...) -> <same>`.
    /// First positional attr arg names the symbolic sibling. Eval body is
    /// opaque. Derivatives required.
    Extern {
        sym_name: String,          // e.g. "elliptic_k"
        eval_path: String,         // e.g. "elliptic_k_eval" (resolved at use site)
        param_names: Vec<String>,  // eval fn's scalar param names
        arity: usize,
        scalar_ty: String,         // "f32" or "f64"
        deriv_strings: Vec<String>, // one per param
        attr_file: String,
        attr_line: u32,
    },
}

impl UserFunction {
    #[allow(dead_code)]
    pub(crate) fn sym_name(&self) -> &str {
        match self {
            UserFunction::Symbolic { sym_name, .. } |
            UserFunction::Extern   { sym_name, .. } => sym_name,
        }
    }
    #[allow(dead_code)]
    pub(crate) fn param_names(&self) -> &[String] {
        match self {
            UserFunction::Symbolic { param_names, .. } |
            UserFunction::Extern   { param_names, .. } => param_names,
        }
    }
}

struct Registry {
    layouts: HashMap<String, SymLayout>,
    constraints: Vec<StashedConstraint>,
    functions: HashMap<String, UserFunction>,
}

static SYM_REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn registry_init() -> Registry {
    Registry {
        layouts: HashMap::new(),
        constraints: Vec::new(),
        functions: HashMap::new(),
    }
}

fn registry_store(name: &str, layout: SymLayout) {
    let mut guard = SYM_REGISTRY.lock().unwrap();
    let reg = guard.get_or_insert_with(registry_init);
    reg.layouts.insert(name.to_string(), layout);
}

fn registry_lookup(name: &str) -> Option<SymLayout> {
    let guard = SYM_REGISTRY.lock().unwrap();
    guard.as_ref().and_then(|reg| reg.layouts.get(name).cloned())
}

fn registry_stash_constraint(c: StashedConstraint) {
    let mut guard = SYM_REGISTRY.lock().unwrap();
    let reg = guard.get_or_insert_with(registry_init);
    reg.constraints.push(c);
}

fn registry_take_constraints() -> Vec<StashedConstraint> {
    let mut guard = SYM_REGISTRY.lock().unwrap();
    guard.as_mut().map(|reg| std::mem::take(&mut reg.constraints)).unwrap_or_default()
}

#[allow(dead_code)]
pub(crate) fn registry_store_function(name: &str, f: UserFunction) {
    let mut guard = SYM_REGISTRY.lock().unwrap();
    let reg = guard.get_or_insert_with(registry_init);
    reg.functions.insert(name.to_string(), f);
}

#[allow(dead_code)]
pub(crate) fn registry_lookup_function(name: &str) -> Option<UserFunction> {
    let guard = SYM_REGISTRY.lock().unwrap();
    guard.as_ref().and_then(|reg| reg.functions.get(name).cloned())
}

/// Snapshot every registered user function. Used by the constraint-body
/// interpreter to build a full `FunctionBag` for `parse_with_functions`.
#[allow(dead_code)]
pub(crate) fn registry_all_functions() -> Vec<UserFunction> {
    let guard = SYM_REGISTRY.lock().unwrap();
    guard.as_ref().map(|reg| reg.functions.values().cloned().collect()).unwrap_or_default()
}

/// Scan the `constraint(...)` token list for `name = "<str>"`. Returns the
/// string literal value if found. Used to produce a readable marker label
/// (`name` is how users disambiguate multiple constraints on one struct).
fn extract_constraint_label(tokens: &[proc_macro2::TokenTree]) -> Option<String> {
    // Expect: `constraint`, Group(parens containing the inner tokens).
    let group = match tokens.get(1)? {
        proc_macro2::TokenTree::Group(g)
            if g.delimiter() == proc_macro2::Delimiter::Parenthesis => g,
        _ => return None,
    };
    let inner: Vec<proc_macro2::TokenTree> = group.stream().into_iter().collect();
    // Walk looking for the sequence `name = "..."`.
    let mut i = 0;
    while i + 2 < inner.len() {
        if let (
            proc_macro2::TokenTree::Ident(id),
            proc_macro2::TokenTree::Punct(p),
            proc_macro2::TokenTree::Literal(lit),
        ) = (&inner[i], &inner[i + 1], &inner[i + 2])
            && *id == "name" && p.as_char() == '='
        {
            let s = lit.to_string();
            // Strip surrounding quotes if present.
            if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
                return Some(s[1..s.len() - 1].to_string());
            }
            return Some(s);
        }
        i += 1;
    }
    None
}


// ===========================================================================
// #[derive(Model)] — generate Model trait impl for structs
// ===========================================================================

/// Attribute macro for arael model structs.
///
/// Generates the `Model` trait implementation, rewrites `SelfBlock<A>` to
/// `SelfBlock<A, {A_PARAM_COUNT}>` and `CrossBlock<A, B>` to
/// `CrossBlock<A, B, {A_PARAM_COUNT}, {B_PARAM_COUNT}>` (two separate
/// const generics so the cross Hessian is a rectangular NA*NB block;
/// TripletBlock is recognized but not rewritten), emits a
/// `const StructName_PARAM_COUNT: usize = ...;`, and produces the symbolic
/// companion struct with `ModelSym` impl.
///
/// # Struct-level attributes
///
/// ## `#[arael(constraint(block, { ... }))]`
///
/// Define a constraint expression on this struct. The body is symbolically
/// differentiated at compile time. Returns a residual array.
///
/// ```ignore
/// #[arael::model]
/// #[arael(constraint(hb_pose, {
///     [(pose.ea.x - pose.info.tilt_roll) * path.tilt_isigma,
///      (pose.ea.y - pose.info.tilt_pitch) * path.tilt_isigma]
/// }))]
/// struct Pose { /* ... */ }
/// ```
///
/// Options:
/// - `guard = expr` -- conditional evaluation; the constraint contributes
///   only when `expr` evaluates to `true` at runtime.
/// - `parent = name` -- name the parent variable in nested constraints.
/// - `name = "label"` -- override the `JacobianRow::label` for rows emitted
///   by this attribute. Defaults to the struct's type name; for structs
///   with multiple `constraint` attributes, the default is
///   `<StructName>:N` where N is the attribute's 0-based index. Useful
///   when a single struct has several residual groups (drift, target,
///   min-soft, etc.) and you want each group identifiable in
///   DOF / sparsity analyses.
///
/// ```ignore
/// #[arael::model]
/// #[arael(constraint(hb, name = "drift", {
///     let d = point.pos - point.pos_value;
///     [d.x * sketch.drift_isigma, d.y * sketch.drift_isigma]
/// }))]
/// #[arael(constraint(hb, guard = self.has_fix_x, name = "fix_x", {
///     [(point.pos.x - point.fix_x) * sketch.constraint_isigma]
/// }))]
/// struct Point { /* ... */ }
/// ```
///
/// ## `#[arael(fit(data, |e| expr))]`
///
/// Auto-generate a complete `LmProblem` implementation for simple curve
/// fitting. Iterates over `data`, evaluates the residual expression for each
/// entry, and generates `calc_cost()`, `calc_grad_hessian_*()`, and
/// `fit_with()` methods.
///
/// ```ignore
/// #[arael::model]
/// #[arael(fit(data, |e| {
///     let plain_r = (a * e.x + b - e.y) / sigma;
///     gamma * atan(plain_r / gamma)
/// }))]
/// struct LinearModel {
///     a: Param<f32>,
///     b: Param<f32>,
///     data: Vec<DataEntry>,
///     sigma: f32,
///     gamma: f32,
/// }
/// ```
///
/// ## `#[arael(root)]` / `#[arael(root, f32)]`
///
/// Mark this struct as the optimization root. Triggers code generation for
/// all stashed `constraint` attributes in the model hierarchy. Generates
/// the `LmProblem` trait implementation with methods: `calc_cost()`,
/// `calc_grad_hessian_dense()`, `calc_grad_hessian_band()`,
/// `calc_grad_hessian_sparse()`, `calc_grad_hessian_sparse_direct()`,
/// `calc_grad_hessian_sparse_indexed()`, and `advance()`.
/// Also generates `serialize64()` / `deserialize64()` convenience methods
/// and `__set_block_indices()` / `__compute_blocks()` internals.
///
/// Use `f32` for single-precision optimization.
///
/// Optional keywords (comma-separated after `root`):
/// - `extended` -- the user implements `ExtendedModel` for runtime
///   constraints (no default impl generated)
/// - `jacobian` -- generates an impl of
///   [`arael::model::JacobianModel<T>`](../arael/model/trait.JacobianModel.html)
///   providing `calc_jacobian(&mut self, params) -> Jacobian<T>` and the
///   default-method `calc_cost_table(&mut self, params) -> HashMap<&'static str, T>`
///   (squared-residual sum grouped by attribute label). Intended for
///   debug/instrumentation (DOF analysis, constraint diagnostics,
///   per-label cost breakdown). Uses the same symbolically differentiated
///   expressions as the Hessian path.
///
/// ## `#[arael(skip_self_block)]`
///
/// Opt-out from the "every params-having struct must declare a
/// `SelfBlock<Self>`" requirement. Use for bag-of-params structs whose
/// params are never touched by their own constraints or by cross/triplet
/// constraints -- e.g. a coefficient wrapper whose gradient is written
/// exclusively by a parent's `ExtendedModel`, or a struct that only
/// exercises `serialize_params32`/`update32` without ever going through
/// the LM solver.
///
/// ```ignore
/// #[arael::model]
/// #[arael(skip_self_block)]
/// struct Coefficient {
///     value: Param<f64>,   // written by RegressionModel::extended_compute64
/// }
/// ```
///
/// Safety net: if a `skip_self_block` struct is later pulled into a
/// `CrossBlock<A, B>` or `TripletBlock` constraint, the cross-block
/// emitter still errors ("type `X` must declare a `SelfBlock<Self>`
/// field"). The opt-out cannot silently break cross/triplet usage.
///
/// ## `#[arael(constraint_index)]`
///
/// Field attribute on a `u32` field in a constraint struct. The macro
/// auto-assigns a sequential global constraint ID during
/// `__set_block_indices()`. The field is automatically skipped from
/// `Model` serialization and `ModelSym` companion generation.
/// `Jacobian` rows carry the same ID for tracing back to constraint objects.
///
/// ```ignore
/// #[arael::model]
/// #[arael(constraint(hb, { ... }))]
/// struct MyConstraint {
///     #[arael(ref = root.points)]
///     a: Ref<Point>,
///     #[arael(constraint_index)]
///     ci: u32,
///     hb: CrossBlock<Point, Point>,
/// }
/// ```
///
/// ```ignore
/// #[arael::model]
/// #[arael(root)]
/// struct Path {
///     poses: refs::Deque<Pose>,
///     landmarks: refs::Vec<PointLandmark>,
///     /* ... */
/// }
/// ```
///
/// The generated `calc_cost` traverses all constraints, evaluating
/// CSE-optimized compiled code. Example fragment (from `cargo expand`):
///
/// ```ignore
/// fn calc_cost(&mut self, params: &[f64]) -> f64 {
///     arael::model::Model::update64(self, params);
///     let mut __cost = 0.0 as f64;
///     for __item in self.poses.iter() {
///         // Tilt constraint -- precomputed rotation matrix used directly:
///         let __r_0 = (pose.ea.work().x - pose.info.tilt_roll)
///             * path.tilt_isigma;
///         let __r_1 = (pose.ea.work().y - pose.info.tilt_pitch)
///             * path.tilt_isigma;
///         __cost += __r_0 * __r_0 + __r_1 * __r_1;
///         // ... odometry, GPS, feature constraints with CSE intermediates
///     }
///     __cost
/// }
/// ```
///
/// View the full generated code with:
///
/// ```ignore
/// cargo expand --example slam_demo
/// ```
#[proc_macro_attribute]
pub fn model(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let mut input = parse_macro_input!(item as syn::DeriveInput);
    match model_attribute(&mut input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// Register a user-defined function for use in `#[arael::model]`
/// constraint bodies. Two forms:
///
/// - **Form A** (purely symbolic): `fn name(x: E, ...) -> E { body_expr }`
///   -- no positional argument; `body_expr` is the arael-sym expression
///   the call materialises into. Optional `derivs = [expr, ...]` to
///   override auto-diff.
/// - **Form B** (opaque eval + derivs): `#[arael::function(name, derivs = [expr, ...])]`
///   on `fn name_eval(x: f32, ...) -> f32` (or `f64`) -- positional
///   `name` is the symbolic sibling the macro emits. `derivs` required.
///
/// See the arael crate-level docs and
/// [examples/runtime_fit_demo.rs](https://github.com/harakas/arael) for
/// usage context.
#[proc_macro_attribute]
pub fn function(attr: TokenStream, item: TokenStream) -> TokenStream {
    match function::function_attribute(attr.into(), item.into()) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn model_attribute(input: &mut syn::DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;

    // Enums are treated as NOP leaves: zero params, trivial Model + ModelSym
    // impls. Lets callers put #[arael::model] on data-less enums used as
    // metadata fields (e.g. style, direction, mode) so those fields don't
    // need #[arael(skip)] at every use site.
    if matches!(input.data, syn::Data::Enum(_)) {
        return Ok(emit_trivial_model_for_enum(input));
    }

    // Compute PARAM_COUNT from Param<T> fields
    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(named) => &named.named,
            _ => return Err(syn::Error::new_spanned(input, "arael::model requires named fields")),
        },
        _ => return Err(syn::Error::new_spanned(input, "arael::model requires a struct or enum")),
    };

    let mut param_count: u32 = 0;
    let mut sym_fields: Vec<(String, SymFieldType)> = Vec::new();
    let mut param_field_names_for_reg: Vec<String> = Vec::new();
    let mut ref_paths_for_reg: Vec<(String, String)> = Vec::new();
    let mut euler_angle_fields_reg: Vec<String> = Vec::new();
    let mut universal_euler_angle_fields_reg: Vec<String> = Vec::new();
    let mut constraint_index_field_reg: Option<String> = None;
    // Detect SelfBlock<Self> field — this struct's canonical grad+diag home.
    let mut self_block_field_reg: Option<String> = None;
    for field in fields {
        let field_name = field.ident.as_ref().unwrap().to_string();
        if is_self_block_for(&field.ty, &name.to_string()) {
            if self_block_field_reg.is_some() {
                return Err(syn::Error::new_spanned(field,
                    format!("`{}` already has a SelfBlock<Self> field; at most one is allowed", name)));
            }
            self_block_field_reg = Some(field_name);
        }
    }
    for field in fields {
        let field_name = field.ident.as_ref().unwrap().to_string();
        // Check for #[arael(ref = ...)] or #[arael(constraint_index)] on this field
        if let Ok(Some(attr)) = parse_arael_attr(&field.attrs) {
            match attr {
                AraelAttr::RefResolve(path) => ref_paths_for_reg.push((field_name.clone(), path)),
                AraelAttr::ConstraintIndex => {
                    constraint_index_field_reg = Some(field_name.clone());
                    sym_fields.push((field_name, SymFieldType::Skip));
                    continue;
                }
                AraelAttr::Skip => {
                    sym_fields.push((field_name, SymFieldType::Skip));
                    continue;
                }
                _ => {}
            }
        }
        // Detect euler angle param types by type name (in addition to attribute)
        if let Some(ea_kind) = is_euler_angle_param_type(&field.ty) {
            match ea_kind {
                "simple" => euler_angle_fields_reg.push(field_name.clone()),
                "universal" => universal_euler_angle_fields_reg.push(field_name.clone()),
                _ => {}
            }
        }
        if is_param_type(&field.ty) {
            param_count += param_type_size(&field.ty);
            param_field_names_for_reg.push(field_name.clone());
            let sft = match param_type_size(&field.ty) {
                1 => SymFieldType::Scalar,
                2 => SymFieldType::Vec2,
                3 => SymFieldType::Vec3,
                _ => SymFieldType::Scalar,
            };
            sym_fields.push((field_name, sft));
        } else if is_hessian_block_type(&field.ty) || is_option_hessian_block(&field.ty) {
            sym_fields.push((field_name, SymFieldType::Skip));
        } else if is_sym_skip_type(&field.ty) {
            // For Vec<T>/Deque<T>, record the inner type for constraint traversal
            let inner_type = extract_wrapper_inner(&field.ty, "Vec")
                .or_else(|| extract_wrapper_inner(&field.ty, "Deque"))
                .or_else(|| extract_wrapper_inner(&field.ty, "Arena"));
            if let Some((_, inner_ident)) = inner_type {
                sym_fields.push((field_name, SymFieldType::Struct(inner_ident.to_string())));
            } else {
                sym_fields.push((field_name, SymFieldType::Skip));
            }
        } else {
            let sft = classify_field_sym_type(&field.ty);
            sym_fields.push((field_name, sft));
        }
    }
    // Every params-having Model must declare exactly one `SelfBlock<Self>`
    // field. It is the canonical home for this entity's gradient +
    // within-entity Hessian diagonal — both macro-emitted constraints and
    // cross/triplet constraints writing to one of this type's params route
    // through it. A params-having struct with no SelfBlock would silently
    // drop those contributions.
    //
    // Exemptions:
    // - `#[arael(fit(...))]` structs generate their own LmProblem impl and
    //   do not route through Model blocks at all.
    // - `#[arael(skip_self_block)]` is an explicit opt-out for bag-of-params
    //   structs whose params are only written by ExtendedModel or a parent
    //   via direct path (no self-constraints, no cross/triplet usage).
    let has_fit = parse_fit_attr(&input.attrs)?.is_some();
    let has_skip_self_block = has_struct_attr_ident(&input.attrs, "skip_self_block");
    if param_count > 0 && self_block_field_reg.is_none() && !has_fit && !has_skip_self_block {
        return Err(syn::Error::new_spanned(name,
            format!("`{}` has {} parameter{} but no `SelfBlock<Self>` field — \
                     add e.g. `hb: arael::model::SelfBlock<Self>` so its grad \
                     and Hessian diagonal have a home, or annotate the struct \
                     with `#[arael(skip_self_block)]` if its params are \
                     written exclusively by a parent's ExtendedModel",
                    name, param_count, if param_count == 1 { "" } else { "s" })));
    }

    // Build symbolic substitutions for euler_angles fields
    let mut substitutions_reg: Vec<(String, String)> = Vec::new();
    for ea_field in &euler_angle_fields_reg {
        // These substitutions will be applied with a variable base prefix later.
        // Store them as template patterns with the field name as placeholder.
        // sin(FIELD.work().x) -> FIELD_sincos.0.x, etc.
        for (comp, _idx) in [("x", 0), ("y", 1), ("z", 2)] {
            let sin_from = format!("SIN({}.work().{})", ea_field, comp);
            let sin_to = format!("{}_sincos.0.{}", ea_field, comp);
            let cos_from = format!("COS({}.work().{})", ea_field, comp);
            let cos_to = format!("{}_sincos.1.{}", ea_field, comp);
            substitutions_reg.push((sin_from, sin_to));
            substitutions_reg.push((cos_from, cos_to));
        }
    }

    // Register injected fields in sym layout
    for ea_field in &euler_angle_fields_reg {
        sym_fields.push((format!("{}_sincos", ea_field), SymFieldType::Skip));
        sym_fields.push((format!("{}_rotation_matrix", ea_field), SymFieldType::Mat3));
    }
    for ea_field in &universal_euler_angle_fields_reg {
        sym_fields.push((format!("{}_ref_rotation", ea_field), SymFieldType::Skip));
        sym_fields.push((format!("{}_delta", ea_field), SymFieldType::Skip));
        sym_fields.push((format!("{}_sincos", ea_field), SymFieldType::Skip));
        sym_fields.push((format!("{}_rotation_matrix", ea_field), SymFieldType::Mat3));
        sym_fields.push((format!("{}_delta_sincos", ea_field), SymFieldType::Skip));
    }

    registry_store(&name.to_string(), SymLayout {
        fields: sym_fields,
        param_fields: param_field_names_for_reg,
        ref_paths: ref_paths_for_reg,
        euler_angle_fields: euler_angle_fields_reg.clone(),
        universal_euler_angle_fields: universal_euler_angle_fields_reg.clone(),
        substitutions: substitutions_reg,
        constraint_index_field: constraint_index_field_reg,
        self_block_field: self_block_field_reg,
    });

    // No field injection needed — SimpleEulerAngleParam/EulerAngleParam contain their own state.

    // Re-read fields (no injection, but keep the pattern for compatibility)
    let _fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(named) => &named.named,
            _ => unreachable!(),
        },
        _ => unreachable!(),
    };

    // Rewrite SelfBlock<A> and CrossBlock<A, B> field types
    if let syn::Data::Struct(ref mut data) = input.data
        && let syn::Fields::Named(ref mut named) = data.fields {
            for field in named.named.iter_mut() {
                rewrite_block_type(&mut field.ty);
            }
        }

    // Emit const
    let const_name = syn::Ident::new(&format!("{}_PARAM_COUNT", name), name.span());
    let param_count_lit = param_count as usize;

    // Generate Model impl and friends using the (now-rewritten) struct
    let model_impl = impl_model(input)?;

    // Strip #[arael(...)] attributes from the emitted struct so they don't
    // get re-interpreted as attribute macro invocations
    input.attrs.retain(|attr| !attr.path().is_ident("arael"));
    if let syn::Data::Struct(ref mut data) = input.data
        && let syn::Fields::Named(ref mut named) = data.fields {
            for field in named.named.iter_mut() {
                field.attrs.retain(|attr| !attr.path().is_ident("arael"));
            }
        }

    Ok(quote! {
        #input
        #[allow(non_upper_case_globals)]
        const #const_name: usize = #param_count_lit;
        #model_impl
    })
}

/// Emit a trivial Model + ModelSym impl for a data-less enum. All Model
/// methods are no-ops, PARAM_COUNT is 0, and the ModelSym companion is an
/// empty struct. The enum itself is emitted unchanged (attributes stripped).
fn emit_trivial_model_for_enum(input: &mut syn::DeriveInput) -> TokenStream2 {
    let name = &input.ident;
    let sym_name = syn::Ident::new(&format!("{}Sym", name), name.span());
    let const_name = syn::Ident::new(&format!("{}_PARAM_COUNT", name), name.span());
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Register a zero-field sym layout so constraint macros that look up
    // this type find an entry (important if the enum is ever used as a
    // nested struct field type in constraint bodies).
    registry_store(&name.to_string(), SymLayout {
        fields: Vec::new(),
        param_fields: Vec::new(),
        ref_paths: Vec::new(),
        euler_angle_fields: Vec::new(),
        universal_euler_angle_fields: Vec::new(),
        substitutions: Vec::new(),
        constraint_index_field: None,
        self_block_field: None,
    });

    // Strip any #[arael(...)] attributes from the emitted item.
    input.attrs.retain(|attr| !attr.path().is_ident("arael"));

    quote! {
        #input
        #[allow(non_upper_case_globals)]
        const #const_name: usize = 0;

        #[derive(Clone)]
        pub struct #sym_name;

        impl arael::model::ModelSym for #name {
            type Sym = #sym_name;
            fn sym(_base: &str) -> #sym_name { #sym_name }
        }

        impl #impl_generics arael::model::Model for #name #ty_generics #where_clause {
            fn serialize_params32(&mut self, _data: &mut std::vec::Vec<f32>) {}
            fn deserialize_params32(&mut self, _data: &[f32]) {}
            fn update32(&mut self, _data: &[f32]) {}
            fn update_self(&mut self) {}
            fn serialize_params64(&mut self, _data: &mut std::vec::Vec<f64>) {}
            fn deserialize_params64(&mut self, _data: &[f64]) {}
            fn update64(&mut self, _data: &[f64]) {}
            const PARAM_COUNT: u32 = 0;
            fn serialize_size(&self) -> u32 { 0 }
            fn param_symbols(_base: &str, _out: &mut std::vec::Vec<String>) {}
            fn zero_blocks(&mut self) {}
            fn accumulate_hessian32(&self, _hessian: &mut [f32]) {}
            fn accumulate_hessian64(&self, _hessian: &mut [f64]) {}
            fn accumulate_hessian_band32(&self, _band: &mut [f32], _kd: usize) -> Result<(), arael::simple_lm::BandError> { Ok(()) }
            fn accumulate_hessian_band64(&self, _band: &mut [f64], _kd: usize) -> Result<(), arael::simple_lm::BandError> { Ok(()) }
            fn accumulate_hessian_sparse32(&self, _coo: &mut arael::simple_lm::CooMatrix<f32>) {}
            fn accumulate_hessian_sparse64(&self, _coo: &mut arael::simple_lm::CooMatrix<f64>) {}
            fn accumulate_hessian_sparse_direct32(&self, _csc: &mut arael::simple_lm::CscMatrix<f32>) {}
            fn accumulate_hessian_sparse_direct64(&self, _csc: &mut arael::simple_lm::CscMatrix<f64>) {}
            fn accumulate_hessian_sparse_indexed32(&self, _vals: &mut [f32], _positions: &[usize], _cursor: &mut usize) {}
            fn accumulate_hessian_sparse_indexed64(&self, _vals: &mut [f64], _positions: &[usize], _cursor: &mut usize) {}
        }
    }
}

/// Classify a non-Param field's sym type from its type path.
fn classify_field_sym_type(ty: &syn::Type) -> SymFieldType {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            return match name.as_str() {
                "f32" | "f64" | "bool" | "u32" | "i32" | "usize" => SymFieldType::Scalar,
                "vect2f" | "vect2d" => SymFieldType::Vec2,
                "vect3f" | "vect3d" => SymFieldType::Vec3,
                "matrix3f" | "matrix3d" => SymFieldType::Mat3,
                "matrix2f" | "matrix2d" => SymFieldType::Mat2,
                _ => {
                    // Check if it's a Ref<T> — extract inner type name
                    if let Some((_, inner_ident)) = extract_wrapper_inner(ty, "Ref") {
                        return SymFieldType::Struct(inner_ident.to_string());
                    }
                    if let Some((_, inner_ident)) = extract_wrapper_inner(ty, "Option") {
                        return SymFieldType::OptionalStruct(inner_ident.to_string());
                    }
                    // Assume it's a struct
                    SymFieldType::Struct(name)
                }
            };
        }
    SymFieldType::Skip
}

/// Extract the SIZE of a Param<T> field's inner type.
fn param_type_size(ty: &syn::Type) -> u32 {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            if name == "Param"
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                    && let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return inner_type_size(inner);
                    }
            if name == "SimpleEulerAngleParam" || name == "EulerAngleParam" {
                return 3; // always vect3 = 3 params
            }
        }
    0
}

/// True iff any `#[arael(...)]` struct-level attribute carries the given
/// bare identifier (e.g. `#[arael(skip_self_block)]`, `#[arael(root,
/// extended)]`). Accepts the ident anywhere in the top-level token list.
fn has_struct_attr_ident(attrs: &[syn::Attribute], ident: &str) -> bool {
    for attr in attrs {
        if !attr.path().is_ident("arael") { continue; }
        let Ok(content) = attr.parse_args::<TokenStream2>() else { continue; };
        for tok in content {
            if let proc_macro2::TokenTree::Ident(id) = tok
                && id == ident {
                    return true;
                }
        }
    }
    false
}

/// Return the ParamType::SIZE for known types.
fn inner_type_size(ty: &syn::Type) -> u32 {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            return match seg.ident.to_string().as_str() {
                "f32" | "f64" => 1,
                "vect2f" | "vect2d" => 2,
                "vect3f" | "vect3d" => 3,
                _ => 0,
            };
        }
    0
}

/// Detect whether `ty` is `SelfBlock<SelfName, ...>` (possibly wrapped in
/// Option<>) — used by `#[arael::model]` to find the struct's canonical
/// self-block field for grad + A-A Hessian accumulation.
fn is_self_block_for(ty: &syn::Type, self_name: &str) -> bool {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            if seg.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                    && let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                        return is_self_block_for(inner_ty, self_name);
                    }
                return false;
            }
            if seg.ident == "SelfBlock"
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                && let Some(syn::GenericArgument::Type(syn::Type::Path(inner_tp))) = args.args.first()
                && let Some(inner_seg) = inner_tp.path.segments.last()
            {
                return inner_seg.ident == self_name;
            }
        }
    false
}

/// Rewrite SelfBlock<A> to SelfBlock<A, {A_PARAM_COUNT}> and
/// SelfBlock<A, f32> to SelfBlock<A, {A_PARAM_COUNT}, f32> and
/// CrossBlock<A, B> to CrossBlock<A, B, {A_PARAM_COUNT}, {B_PARAM_COUNT}> and
/// CrossBlock<A, B, f32> to CrossBlock<A, B, {A_PARAM_COUNT}, {B_PARAM_COUNT}, f32>.
fn rewrite_block_type(ty: &mut syn::Type) {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last_mut() {
            let type_name = seg.ident.to_string();
            if type_name == "SelfBlock" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    let type_args: Vec<&syn::Type> = args.args.iter()
                        .filter_map(|a| if let syn::GenericArgument::Type(t) = a { Some(t) } else { None })
                        .collect();
                    if type_args.len() == 1 || type_args.len() == 2 {
                        // First arg is the model type A
                        if let syn::Type::Path(a_path) = type_args[0]
                            && let Some(a_seg) = a_path.path.segments.last() {
                                let const_name = syn::Ident::new(
                                    &format!("{}_PARAM_COUNT", a_seg.ident),
                                    a_seg.ident.span(),
                                );
                                let a_path = type_args[0];
                                if type_args.len() == 2 {
                                    // SelfBlock<A, f32> -> SelfBlock<A, {N}, f32>
                                    let float_ty = type_args[1];
                                    let new_ty: syn::Type = syn::parse_quote! {
                                        SelfBlock<#a_path, { #const_name }, #float_ty>
                                    };
                                    *ty = new_ty;
                                } else {
                                    // SelfBlock<A> -> SelfBlock<A, {N}>
                                    let new_ty: syn::Type = syn::parse_quote! {
                                        SelfBlock<#a_path, { #const_name }>
                                    };
                                    *ty = new_ty;
                                }
                            }
                    }
                }
            } else if type_name == "CrossBlock"
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    let type_args: Vec<&syn::Type> = args.args.iter()
                        .filter_map(|a| if let syn::GenericArgument::Type(t) = a { Some(t) } else { None })
                        .collect();
                    if (type_args.len() == 2 || type_args.len() == 3)
                        && let (syn::Type::Path(a_path), syn::Type::Path(b_path)) =
                            (type_args[0], type_args[1])
                            && let (Some(a_seg), Some(b_seg)) = (a_path.path.segments.last(), b_path.path.segments.last()) {
                                let a_const = syn::Ident::new(
                                    &format!("{}_PARAM_COUNT", a_seg.ident),
                                    a_seg.ident.span(),
                                );
                                let b_const = syn::Ident::new(
                                    &format!("{}_PARAM_COUNT", b_seg.ident),
                                    b_seg.ident.span(),
                                );
                                let a_ty = type_args[0];
                                let b_ty = type_args[1];
                                if type_args.len() == 3 {
                                    // CrossBlock<A, B, f32> -> CrossBlock<A, B, {NA}, {NB}, f32>
                                    let float_ty = type_args[2];
                                    let new_ty: syn::Type = syn::parse_quote! {
                                        CrossBlock<#a_ty, #b_ty, { #a_const }, { #b_const }, #float_ty>
                                    };
                                    *ty = new_ty;
                                } else {
                                    // CrossBlock<A, B> -> CrossBlock<A, B, {NA}, {NB}>
                                    let new_ty: syn::Type = syn::parse_quote! {
                                        CrossBlock<#a_ty, #b_ty, { #a_const }, { #b_const }>
                                    };
                                    *ty = new_ty;
                                }
                            }
                }
        }
}

#[proc_macro_derive(Model, attributes(arael))]
pub fn derive_model(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as syn::DeriveInput);
    match impl_model(&input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn impl_model(input: &syn::DeriveInput) -> syn::Result<TokenStream2> {
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(named) => &named.named,
            _ => return Err(syn::Error::new_spanned(input, "Model derive requires named fields")),
        },
        _ => return Err(syn::Error::new_spanned(input, "Model derive requires a struct")),
    };

    // Pass 1: identify Param<T> fields
    let mut param_field_names: HashSet<String> = HashSet::new();
    for field in fields {
        if is_param_type(&field.ty)
            && let Some(ident) = &field.ident {
                param_field_names.insert(ident.to_string());
            }
    }

    // Detect euler angle param types and generate precompute calls
    let mut euler_compute_stmts: Vec<TokenStream2> = Vec::new();
    for field in fields {
        if is_euler_angle_param_type(&field.ty).is_some() {
            let ea_ident = field.ident.as_ref().unwrap();
            euler_compute_stmts.push(quote! {
                self.#ea_ident.__precompute();
            });
        }
    }

    // Pass 2: classify fields and generate method bodies
    let mut serialize_stmts: Vec<TokenStream2> = Vec::new();
    let mut deserialize_stmts: Vec<TokenStream2> = Vec::new();
    let mut update_phase1: Vec<TokenStream2> = Vec::new();
    let mut update_self_phase1: Vec<TokenStream2> = Vec::new();
    let mut compute_stmts: Vec<TokenStream2> = Vec::new();
    let mut serialize64_stmts: Vec<TokenStream2> = Vec::new();
    let mut deserialize64_stmts: Vec<TokenStream2> = Vec::new();
    let mut update64_phase1: Vec<TokenStream2> = Vec::new();
    let mut zero_blocks_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian32_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian64_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian_band32_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian_band64_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian_sparse32_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian_sparse64_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian_sparse_direct32_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian_sparse_direct64_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian_sparse_indexed32_stmts: Vec<TokenStream2> = Vec::new();
    let mut accumulate_hessian_sparse_indexed64_stmts: Vec<TokenStream2> = Vec::new();
    let mut param_count_terms: Vec<TokenStream2> = Vec::new();
    let mut serialize_size_stmts: Vec<TokenStream2> = Vec::new();
    let mut param_symbols_stmts: Vec<TokenStream2> = Vec::new();

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let attr = parse_arael_attr(&field.attrs)?;

        match attr {
            // Cross is a constraint-struct-only attribute for CrossBlock
            // fields; treated like the default path here (no param, no
            // compute, just a block field).
            Some(AraelAttr::Skip) | Some(AraelAttr::ConstraintIndex) => continue,
            Some(AraelAttr::Compute(expr_tokens)) => {
                let substituted = substitute_param_idents(expr_tokens, &param_field_names);
                compute_stmts.push(quote! { self.#ident = #substituted; });
            }
            Some(AraelAttr::RefResolve(_)) | Some(AraelAttr::Cross(_)) | None => {
                // HessianBlock fields: skip serialize, handle in zero/accumulate
                if is_hessian_block_type(&field.ty) {
                    zero_blocks_stmts.push(quote! { self.#ident.zero(); });
                    let acc_dense = quote! { self.#ident.accumulate_hessian(hessian); };
                    let acc_band = quote! { self.#ident.accumulate_hessian_band(band, kd)?; };
                    let acc_sparse = quote! { self.#ident.accumulate_hessian_sparse(coo); };
                    let acc_sparse_direct = quote! { self.#ident.accumulate_hessian_sparse_direct(csc); };
                    let acc_sparse_indexed = quote! { self.#ident.accumulate_hessian_sparse_indexed(vals, positions, cursor); };
                    if block_is_f32(&field.ty) {
                        accumulate_hessian32_stmts.push(acc_dense);
                        accumulate_hessian_band32_stmts.push(acc_band);
                        accumulate_hessian_sparse32_stmts.push(acc_sparse);
                        accumulate_hessian_sparse_direct32_stmts.push(acc_sparse_direct);
                        accumulate_hessian_sparse_indexed32_stmts.push(acc_sparse_indexed);
                    } else {
                        accumulate_hessian64_stmts.push(acc_dense);
                        accumulate_hessian_band64_stmts.push(acc_band);
                        accumulate_hessian_sparse64_stmts.push(acc_sparse);
                        accumulate_hessian_sparse_direct64_stmts.push(acc_sparse_direct);
                        accumulate_hessian_sparse_indexed64_stmts.push(acc_sparse_indexed);
                    }
                    continue;
                }
                // Option<HessianBlock> fields: skip serialize, handle in zero/accumulate
                if is_option_hessian_block(&field.ty) {
                    zero_blocks_stmts.push(quote! {
                        if let Some(ref mut __hb) = self.#ident { __hb.zero(); }
                    });
                    let acc_dense = quote! {
                        if let Some(ref __hb) = self.#ident { __hb.accumulate_hessian(hessian); }
                    };
                    let acc_band = quote! {
                        if let Some(ref __hb) = self.#ident { __hb.accumulate_hessian_band(band, kd)?; }
                    };
                    let acc_sparse = quote! {
                        if let Some(ref __hb) = self.#ident { __hb.accumulate_hessian_sparse(coo); }
                    };
                    let acc_sparse_direct = quote! {
                        if let Some(ref __hb) = self.#ident { __hb.accumulate_hessian_sparse_direct(csc); }
                    };
                    let acc_sparse_indexed = quote! {
                        if let Some(ref __hb) = self.#ident { __hb.accumulate_hessian_sparse_indexed(vals, positions, cursor); }
                    };
                    if block_is_f32(&field.ty) {
                        accumulate_hessian32_stmts.push(acc_dense);
                        accumulate_hessian_band32_stmts.push(acc_band);
                        accumulate_hessian_sparse32_stmts.push(acc_sparse);
                        accumulate_hessian_sparse_direct32_stmts.push(acc_sparse_direct);
                        accumulate_hessian_sparse_indexed32_stmts.push(acc_sparse_indexed);
                    } else {
                        accumulate_hessian64_stmts.push(acc_dense);
                        accumulate_hessian_band64_stmts.push(acc_band);
                        accumulate_hessian_sparse64_stmts.push(acc_sparse);
                        accumulate_hessian_sparse_direct64_stmts.push(acc_sparse_direct);
                        accumulate_hessian_sparse_indexed64_stmts.push(acc_sparse_indexed);
                    }
                    continue;
                }

                let ty = &field.ty;

                if is_param_type(ty) {
                    param_count_terms.push(quote! {
                        <#ty as arael::model::Model>::PARAM_COUNT
                    });
                    let field_name = ident.to_string();
                    param_symbols_stmts.push(quote! {
                        <#ty as arael::model::Model>::param_symbols(
                            &format!("{}.{}", base, #field_name), out
                        );
                    });
                }

                // All param types (Param, SimpleEulerAngleParam, EulerAngleParam)
                // use their own Model trait impl for serialize/deserialize/update.
                serialize_stmts.push(quote! {
                    arael::model::Model::serialize_params32(&mut self.#ident, data);
                });
                deserialize_stmts.push(quote! {
                    arael::model::Model::deserialize_params32(&mut self.#ident, data);
                });
                update_phase1.push(quote! {
                    arael::model::Model::update32(&mut self.#ident, data);
                });
                update_self_phase1.push(quote! {
                    arael::model::Model::update_self(&mut self.#ident);
                });
                serialize64_stmts.push(quote! {
                    arael::model::Model::serialize_params64(&mut self.#ident, data);
                });
                deserialize64_stmts.push(quote! {
                    arael::model::Model::deserialize_params64(&mut self.#ident, data);
                });
                update64_phase1.push(quote! {
                    arael::model::Model::update64(&mut self.#ident, data);
                });
                serialize_size_stmts.push(quote! {
                    arael::model::Model::serialize_size(&self.#ident)
                });
                // Also recurse into sub-models for zero/accumulate
                zero_blocks_stmts.push(quote! {
                    arael::model::Model::zero_blocks(&mut self.#ident);
                });
                accumulate_hessian32_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian32(&self.#ident, hessian);
                });
                accumulate_hessian64_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian64(&self.#ident, hessian);
                });
                accumulate_hessian_band32_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian_band32(&self.#ident, band, kd)?;
                });
                accumulate_hessian_band64_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian_band64(&self.#ident, band, kd)?;
                });
                accumulate_hessian_sparse32_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian_sparse32(&self.#ident, coo);
                });
                accumulate_hessian_sparse64_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian_sparse64(&self.#ident, coo);
                });
                accumulate_hessian_sparse_direct32_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian_sparse_direct32(&self.#ident, csc);
                });
                accumulate_hessian_sparse_direct64_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian_sparse_direct64(&self.#ident, csc);
                });
                accumulate_hessian_sparse_indexed32_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian_sparse_indexed32(&self.#ident, vals, positions, cursor);
                });
                accumulate_hessian_sparse_indexed64_stmts.push(quote! {
                    arael::model::Model::accumulate_hessian_sparse_indexed64(&self.#ident, vals, positions, cursor);
                });
            }
        }
    }

    let model_impl = quote! {
        impl #impl_generics arael::model::Model for #name #ty_generics #where_clause {
            fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
                #(#serialize_stmts)*
            }
            fn deserialize_params32(&mut self, data: &[f32]) {
                #(#deserialize_stmts)*
            }
            fn update32(&mut self, data: &[f32]) {
                #(#update_phase1)*
                #(#compute_stmts)*
                #(#euler_compute_stmts)*
            }
            fn update_self(&mut self) {
                #(#update_self_phase1)*
                #(#compute_stmts)*
                #(#euler_compute_stmts)*
            }
            fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
                #(#serialize64_stmts)*
            }
            fn deserialize_params64(&mut self, data: &[f64]) {
                #(#deserialize64_stmts)*
            }
            fn update64(&mut self, data: &[f64]) {
                #(#update64_phase1)*
                #(#compute_stmts)*
                #(#euler_compute_stmts)*
            }
            const PARAM_COUNT: u32 = 0 #(+ #param_count_terms)*;
            fn serialize_size(&self) -> u32 {
                0 #(+ #serialize_size_stmts)*
            }
            fn param_symbols(base: &str, out: &mut std::vec::Vec<String>) {
                #(#param_symbols_stmts)*
            }
            fn zero_blocks(&mut self) {
                #(#zero_blocks_stmts)*
            }
            fn accumulate_hessian32(&self, hessian: &mut [f32]) {
                #(#accumulate_hessian32_stmts)*
            }
            fn accumulate_hessian64(&self, hessian: &mut [f64]) {
                #(#accumulate_hessian64_stmts)*
            }
            fn accumulate_hessian_band32(&self, band: &mut [f32], kd: usize) -> Result<(), arael::simple_lm::BandError> {
                #(#accumulate_hessian_band32_stmts)*
                Ok(())
            }
            fn accumulate_hessian_band64(&self, band: &mut [f64], kd: usize) -> Result<(), arael::simple_lm::BandError> {
                #(#accumulate_hessian_band64_stmts)*
                Ok(())
            }
            fn accumulate_hessian_sparse32(&self, coo: &mut arael::simple_lm::CooMatrix<f32>) {
                #(#accumulate_hessian_sparse32_stmts)*
            }
            fn accumulate_hessian_sparse64(&self, coo: &mut arael::simple_lm::CooMatrix<f64>) {
                #(#accumulate_hessian_sparse64_stmts)*
            }
            fn accumulate_hessian_sparse_direct32(&self, csc: &mut arael::simple_lm::CscMatrix<f32>) {
                #(#accumulate_hessian_sparse_direct32_stmts)*
            }
            fn accumulate_hessian_sparse_direct64(&self, csc: &mut arael::simple_lm::CscMatrix<f64>) {
                #(#accumulate_hessian_sparse_direct64_stmts)*
            }
            fn accumulate_hessian_sparse_indexed32(&self, vals: &mut [f32], positions: &[usize], cursor: &mut usize) {
                #(#accumulate_hessian_sparse_indexed32_stmts)*
            }
            fn accumulate_hessian_sparse_indexed64(&self, vals: &mut [f64], positions: &[usize], cursor: &mut usize) {
                #(#accumulate_hessian_sparse_indexed64_stmts)*
            }
        }
    };

    // Generate *Sym companion struct and ModelSym impl
    let sym_impl = generate_sym_impl(name, fields)?;

    // Check for #[arael(fit(...))] on the struct
    let fit_impl = match parse_fit_attr(&input.attrs)? {
        Some(fit) => generate_fit_impl(name, fields, &fit)?,
        None => quote! {},
    };

    // Check for #[arael(constraint(...))] — stash ALL constraints for later generation.
    // Capture plain (file, line) data from span while we're still inside the
    // originating invocation — Span itself doesn't survive the bridge but
    // primitives do.
    {
        let fields_ts = quote! { #fields };
        for attr in &input.attrs {
            if !attr.path().is_ident("arael") { continue; }
            let content: TokenStream2 = attr.parse_args().unwrap_or_default();
            let tvec: Vec<proc_macro2::TokenTree> = content.into_iter().collect();
            if tvec.is_empty() { continue; }
            if let proc_macro2::TokenTree::Ident(ref id) = tvec[0]
                && *id == "constraint" {
                    use syn::spanned::Spanned as _;
                    let attr_span = attr.span();
                    let label_hint = extract_constraint_label(&tvec)
                        .unwrap_or_else(|| name.to_string());
                    let tokens: TokenStream2 = tvec.into_iter().collect();
                    registry_stash_constraint(StashedConstraint {
                        struct_name: name.to_string(),
                        attr_file: attr_span.file(),
                        attr_line: attr_span.start().line as u32,
                        label_hint,
                        attr_tokens: tokens.to_string(),
                        fields_tokens: fields_ts.to_string(),
                    });
                }
        }
    }

    // Check for #[arael(root)] or #[arael(root, f32)] — trigger generation of all stashed constraints
    let root_info = input.attrs.iter().find_map(|attr| {
        if !attr.path().is_ident("arael") { return None; }
        let content: TokenStream2 = attr.parse_args().ok()?;
        let tvec: Vec<proc_macro2::TokenTree> = content.into_iter().collect();
        if let Some(proc_macro2::TokenTree::Ident(id)) = tvec.first() {
            if *id != "root" { return None; }
            // Parse optional keywords after comma: f32/f64, extended, jacobian
            let mut precision = "f64".to_string();
            let mut custom = false;
            let mut jacobian = false;
            let mut pos = 1;
            while pos < tvec.len() {
                if let proc_macro2::TokenTree::Punct(p) = &tvec[pos]
                    && p.as_char() == ',' {
                        pos += 1;
                        if let Some(proc_macro2::TokenTree::Ident(kw)) = tvec.get(pos) {
                            let kw_str = kw.to_string();
                            if kw_str == "f32" || kw_str == "f64" {
                                precision = kw_str;
                            } else if kw_str == "extended" {
                                custom = true;
                            } else if kw_str == "jacobian" {
                                jacobian = true;
                            }
                        }
                    }
                pos += 1;
            }
            return Some((precision, custom, jacobian));
        }
        None
    });
    let root_precision = root_info.as_ref().map(|(p, _, _)| p.clone());
    let root_custom = root_info.as_ref().map(|(_, c, _)| *c).unwrap_or(false);
    let root_jacobian = root_info.as_ref().map(|(_, _, j)| *j).unwrap_or(false);

    let constraint_impls = if let Some(ref precision) = root_precision {
        constraint::generate_root_methods(name, fields, precision, root_custom, root_jacobian)?
    } else {
        quote! {}
    };

    Ok(quote! {
        #model_impl
        #sym_impl
        #fit_impl
        #constraint_impls
    })
}

/// Check if a type is `Param<...>` by looking at the last path segment.
fn is_param_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            return name == "Param" || name == "SimpleEulerAngleParam" || name == "EulerAngleParam";
        }
    false
}

fn is_euler_angle_param_type(ty: &syn::Type) -> Option<&'static str> {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            if name == "SimpleEulerAngleParam" { return Some("simple"); }
            if name == "EulerAngleParam" { return Some("universal"); }
        }
    None
}

/// Check if a block type (SelfBlock or CrossBlock) has explicit f32 as its last type arg.
/// SelfBlock<A, f32> or CrossBlock<A, B, f32> -> true. Default (no float arg) -> false.
/// Also handles Option<SelfBlock<..., f32>>.
fn block_is_f32(ty: &syn::Type) -> bool {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            // Unwrap Option<...> if needed
            if seg.ident == "Option" {
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                    && let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                        return block_is_f32(inner_ty);
                    }
                return false;
            }
            // SelfBlock or CrossBlock: check if last type arg is f32
            if let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                && let Some(syn::GenericArgument::Type(syn::Type::Path(last))) = args.args.last()
                    && let Some(last_seg) = last.path.segments.last() {
                        return last_seg.ident == "f32";
                    }
        }
    false
}

/// Check if a type is `SelfBlock<...>` or `CrossBlock<...>`.
fn is_hessian_block_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            return matches!(name.as_str(), "SelfBlock" | "CrossBlock" | "TripletBlock");
        }
    false
}

/// Check if a type is `Option<SelfBlock<...>>` or `Option<CrossBlock<...>>`.
fn is_option_hessian_block(ty: &syn::Type) -> bool {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
            && seg.ident == "Option"
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                    && let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first() {
                        return is_hessian_block_type(inner_ty);
                    }
    false
}

/// Check if a type should be skipped in Sym generation (Vec, Deque, SelfBlock, CrossBlock).
fn is_sym_skip_type(ty: &syn::Type) -> bool {
    if is_hessian_block_type(ty) || is_option_hessian_block(ty) {
        return true;
    }
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            return matches!(name.as_str(), "Vec" | "Deque" | "Arena");
        }
    false
}

/// Extract the inner type T from a generic wrapper like Ref<T> or Option<T>.
/// Returns the inner type and the last ident of T's path (e.g. "Pose").
fn extract_wrapper_inner<'a>(ty: &'a syn::Type, wrapper: &str) -> Option<(&'a syn::Type, &'a syn::Ident)> {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last()
            && seg.ident == wrapper
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                    && let Some(syn::GenericArgument::Type(inner_ty)) = args.args.first()
                        && let syn::Type::Path(inner_tp) = inner_ty
                            && let Some(inner_seg) = inner_tp.path.segments.last() {
                                return Some((inner_ty, &inner_seg.ident));
                            }
    None
}

/// Convert a type name to a lowercase plural collection name: Pose -> poses
fn to_collection_name(ident: &syn::Ident) -> String {
    let s = ident.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && c.is_uppercase() {
            result.push('_');
        }
        result.push(c.to_ascii_lowercase());
    }
    result.push('s');
    result
}

/// Generate the `*Sym` companion struct and `ModelSym` impl.
fn generate_sym_impl(
    name: &syn::Ident,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) -> syn::Result<TokenStream2> {
    let sym_name = syn::Ident::new(&format!("{}Sym", name), name.span());

    let mut sym_fields: Vec<TokenStream2> = Vec::new();
    let mut sym_inits: Vec<TokenStream2> = Vec::new();

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let ty = &field.ty;

        // Skip fields with #[arael(skip)], #[arael(compute = ...)], or #[arael(constraint_index)]
        match parse_arael_attr(&field.attrs)? {
            Some(AraelAttr::Skip) | Some(AraelAttr::Compute(_)) | Some(AraelAttr::ConstraintIndex) => continue,
            _ => {}
        }

        // Skip collection and hessian block types
        if is_sym_skip_type(ty) {
            continue;
        }

        let field_name = ident.to_string();

        // Ref<T>: resolve via collection lookup, e.g. "poses[base.pose]"
        if let Some((inner_ty, inner_ident)) = extract_wrapper_inner(ty, "Ref") {
            let collection = to_collection_name(inner_ident);
            sym_fields.push(quote! {
                pub #ident: <#inner_ty as arael::model::ModelSym>::Sym,
            });
            sym_inits.push(quote! {
                #ident: <#inner_ty as arael::model::ModelSym>::sym(
                    &format!("{}[{}.{}]", #collection, base, #field_name)
                ),
            });
            continue;
        }

        // Option<T>: unwrap via as_ref().unwrap(), e.g. "base.gps.as_ref().unwrap()"
        if let Some((inner_ty, _)) = extract_wrapper_inner(ty, "Option") {
            sym_fields.push(quote! {
                pub #ident: <#inner_ty as arael::model::ModelSym>::Sym,
            });
            sym_inits.push(quote! {
                #ident: <#inner_ty as arael::model::ModelSym>::sym(
                    &format!("{}.{}.as_ref().unwrap()", base, #field_name)
                ),
            });
            continue;
        }

        // Regular fields
        sym_fields.push(quote! {
            pub #ident: <#ty as arael::model::ModelSym>::Sym,
        });

        sym_inits.push(quote! {
            #ident: <#ty as arael::model::ModelSym>::sym(&format!("{}.{}", base, #field_name)),
        });
    }

    Ok(quote! {
        #[derive(Clone)]
        pub struct #sym_name {
            #(#sym_fields)*
        }

        impl arael::model::ModelSym for #name {
            type Sym = #sym_name;
            fn sym(base: &str) -> #sym_name {
                #sym_name {
                    #(#sym_inits)*
                }
            }
        }
    })
}

pub(crate) enum AraelAttr {
    Skip,
    Compute(TokenStream2),
    RefResolve(String),  // resolution path, e.g. "root.poses"
    ConstraintIndex,     // marks a u32 field as constraint index
    /// Ref-pair binding for a CrossBlock field on a constraint struct:
    /// `#[arael(cross = (refA, refB))]`. The two idents must name `Ref<X>`
    /// / `Ref<Y>` fields on the same struct whose types match the
    /// CrossBlock's `<A, B>` type parameters. Used by the routing table to
    /// disambiguate multiple CrossBlocks with the same type signature.
    #[allow(dead_code)]  // read by routing-table construction in constraint.rs
    Cross(Vec<String>),
}

/// Parse `#[arael(skip)]`, `#[arael(compute = <expr>)]`,
/// `#[arael(ref = <path>)]`, `#[arael(constraint_index)]`, or
/// `#[arael(cross = (refA, refB))]` from field attributes.
pub(crate) fn parse_arael_attr(attrs: &[syn::Attribute]) -> syn::Result<Option<AraelAttr>> {
    for attr in attrs {
        if attr.path().is_ident("arael") {
            let content: TokenStream2 = attr.parse_args()?;
            let tokens: Vec<proc_macro2::TokenTree> = content.into_iter().collect();

            if tokens.is_empty() {
                continue;
            }

            if let proc_macro2::TokenTree::Ident(ref ident) = tokens[0] {
                let kw = ident.to_string();
                if kw == "skip" {
                    return Ok(Some(AraelAttr::Skip));
                }
                if kw == "constraint_index" {
                    return Ok(Some(AraelAttr::ConstraintIndex));
                }
                // #[arael(ref = root.poses)]
                if kw == "ref" {
                    if tokens.len() >= 3
                        && let proc_macro2::TokenTree::Punct(ref p) = tokens[1]
                            && p.as_char() == '=' {
                                let path_tokens: TokenStream2 =
                                    tokens[2..].iter().cloned().collect();
                                return Ok(Some(AraelAttr::RefResolve(path_tokens.to_string())));
                            }
                    return Err(syn::Error::new_spanned(
                        &tokens[0],
                        "expected `ref = <path>`",
                    ));
                }
                if kw == "compute" {
                    if tokens.len() >= 3
                        && let proc_macro2::TokenTree::Punct(ref p) = tokens[1]
                            && p.as_char() == '=' {
                                let expr_tokens: TokenStream2 =
                                    tokens[2..].iter().cloned().collect();
                                return Ok(Some(AraelAttr::Compute(expr_tokens)));
                            }
                    return Err(syn::Error::new_spanned(
                        &tokens[0],
                        "expected `compute = <expression>`",
                    ));
                }
                // #[arael(cross = (refA, refB))]
                if kw == "cross" {
                    if tokens.len() >= 3
                        && let proc_macro2::TokenTree::Punct(ref p) = tokens[1]
                        && p.as_char() == '='
                        && let proc_macro2::TokenTree::Group(ref g) = tokens[2]
                        && g.delimiter() == proc_macro2::Delimiter::Parenthesis {
                            let mut refs: Vec<String> = Vec::new();
                            for tt in g.stream() {
                                match tt {
                                    proc_macro2::TokenTree::Ident(id) => refs.push(id.to_string()),
                                    proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => {}
                                    other => return Err(syn::Error::new_spanned(&tokens[0],
                                        format!("expected ref field name or ',' in cross = (...), got `{}`", other))),
                                }
                            }
                            if refs.len() != 2 {
                                return Err(syn::Error::new_spanned(&tokens[0],
                                    format!("cross = (...) expects exactly two ref field names, got {}", refs.len())));
                            }
                            return Ok(Some(AraelAttr::Cross(refs)));
                        }
                    return Err(syn::Error::new_spanned(
                        &tokens[0],
                        "expected `cross = (refA, refB)`",
                    ));
                }
            }

            return Err(syn::Error::new_spanned(
                attr,
                "unknown arael attribute, expected `skip`, `compute = <expr>`, `ref = <path>`, `constraint_index`, or `cross = (refA, refB)`",
            ));
        }
    }
    Ok(None)
}

/// In a compute expression, replace bare identifiers matching Param field names
/// with `self.<name>.work()`. Identifiers that are part of a `::` path are not
/// replaced (e.g. `matrix3f::rotation_from_euler_angles` stays as-is).
fn substitute_param_idents(
    tokens: TokenStream2,
    param_names: &HashSet<String>,
) -> TokenStream2 {
    use proc_macro2::{TokenTree, Group};

    let token_vec: Vec<TokenTree> = tokens.into_iter().collect();
    let mut result = TokenStream2::new();
    let len = token_vec.len();

    for i in 0..len {
        match &token_vec[i] {
            TokenTree::Ident(ident) => {
                let name = ident.to_string();
                if param_names.contains(&name) {
                    let prev_is_colon = i >= 1
                        && matches!(&token_vec[i - 1], TokenTree::Punct(p) if p.as_char() == ':');
                    let next_is_colon = i + 1 < len
                        && matches!(&token_vec[i + 1], TokenTree::Punct(p) if p.as_char() == ':');

                    if !prev_is_colon && !next_is_colon {
                        let span = ident.span();
                        let self_id = proc_macro2::Ident::new("self", span);
                        let field_id = proc_macro2::Ident::new(&name, span);
                        let work_id = proc_macro2::Ident::new("work", span);
                        result.extend(quote! { #self_id.#field_id.#work_id() });
                        continue;
                    }
                }
                result.extend(std::iter::once(TokenTree::Ident(ident.clone())));
            }
            TokenTree::Group(group) => {
                let inner = substitute_param_idents(group.stream(), param_names);
                let mut new_group = Group::new(group.delimiter(), inner);
                new_group.set_span(group.span());
                result.extend(std::iter::once(TokenTree::Group(new_group)));
            }
            other => {
                result.extend(std::iter::once(other.clone()));
            }
        }
    }

    result
}

// ===========================================================================
// #[arael(fit(...))] — auto-generate cost, gradient, hessian, and fit methods
// ===========================================================================

struct FitAttr {
    data_field: proc_macro2::Ident,
    loop_var: proc_macro2::Ident,
    body_stmts: Vec<Stmt>,
}

/// Parse `#[arael(fit(data, |e| { ... }))]` from struct-level attributes.
fn parse_fit_attr(attrs: &[syn::Attribute]) -> syn::Result<Option<FitAttr>> {
    for attr in attrs {
        if !attr.path().is_ident("arael") {
            continue;
        }
        let content: TokenStream2 = attr.parse_args()?;
        let tokens: Vec<proc_macro2::TokenTree> = content.into_iter().collect();
        if tokens.is_empty() {
            continue;
        }

        if let proc_macro2::TokenTree::Ident(ref ident) = tokens[0] {
            if *ident != "fit" {
                continue;
            }

            if tokens.len() < 2 {
                return Err(syn::Error::new_spanned(
                    ident,
                    "expected fit(data_field, |var| { ... })",
                ));
            }

            if let proc_macro2::TokenTree::Group(ref group) = tokens[1] {
                if group.delimiter() != proc_macro2::Delimiter::Parenthesis {
                    return Err(syn::Error::new_spanned(
                        ident,
                        "expected parentheses after fit",
                    ));
                }
                let inner: Vec<proc_macro2::TokenTree> =
                    group.stream().into_iter().collect();
                return parse_fit_inner(&inner, ident);
            }

            return Err(syn::Error::new_spanned(
                ident,
                "expected fit(...)",
            ));
        }
    }
    Ok(None)
}

fn parse_fit_inner(
    tokens: &[proc_macro2::TokenTree],
    err_span: &proc_macro2::Ident,
) -> syn::Result<Option<FitAttr>> {
    // Expected tokens: data_field , | loop_var | { body }
    //              or: data_field , | loop_var | expr
    let mut pos = 0;

    let data_field = match tokens.get(pos) {
        Some(proc_macro2::TokenTree::Ident(id)) => id.clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                err_span,
                "expected data field name as first argument to fit()",
            ))
        }
    };
    pos += 1;

    // comma
    match tokens.get(pos) {
        Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == ',' => {}
        _ => {
            return Err(syn::Error::new_spanned(
                err_span,
                "expected comma after data field name",
            ))
        }
    }
    pos += 1;

    // | loop_var |
    match tokens.get(pos) {
        Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '|' => {}
        _ => {
            return Err(syn::Error::new_spanned(
                err_span,
                "expected |variable| closure syntax",
            ))
        }
    }
    pos += 1;

    let loop_var = match tokens.get(pos) {
        Some(proc_macro2::TokenTree::Ident(id)) => id.clone(),
        _ => {
            return Err(syn::Error::new_spanned(
                err_span,
                "expected loop variable name",
            ))
        }
    };
    pos += 1;

    match tokens.get(pos) {
        Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '|' => {}
        _ => {
            return Err(syn::Error::new_spanned(
                err_span,
                "expected closing | after loop variable",
            ))
        }
    }
    pos += 1;

    // Either { block } or remaining expression tokens
    let body_stmts = match tokens.get(pos) {
        Some(proc_macro2::TokenTree::Group(g))
            if g.delimiter() == proc_macro2::Delimiter::Brace =>
        {
            let block_tokens =
                proc_macro2::TokenStream::from(proc_macro2::TokenTree::Group(g.clone()));
            let block: syn::Block = syn::parse2(block_tokens)?;
            block.stmts
        }
        _ => {
            // Remaining tokens form a direct expression
            let remaining: TokenStream2 = tokens[pos..].iter().cloned().collect();
            let expr: Expr = syn::parse2(remaining)?;
            vec![Stmt::Expr(expr, None)]
        }
    };

    Ok(Some(FitAttr {
        data_field,
        loop_var,
        body_stmts,
    }))
}

// ---------------------------------------------------------------------------
// syn::Expr → arael_sym::E conversion
// ---------------------------------------------------------------------------

struct SymContext {
    loop_var: String,
    param_names: Vec<String>,
    constant_names: HashSet<String>,
    let_bindings: HashMap<String, arael_sym::E>,
    data_fields: HashSet<String>,
    used_constants: HashSet<String>,
}

fn syn_expr_to_sym(expr: &Expr, ctx: &mut SymContext) -> syn::Result<arael_sym::E> {
    match expr {
        Expr::Path(ep) if ep.qself.is_none() => {
            if let Some(ident) = ep.path.get_ident() {
                let name = ident.to_string();
                // Check let bindings first (inline expansion)
                if let Some(e) = ctx.let_bindings.get(&name) {
                    return Ok(e.clone());
                }
                // Check params
                if ctx.param_names.contains(&name) {
                    return Ok(arael_sym::symbol(&name));
                }
                // Check constants
                if ctx.constant_names.contains(&name) {
                    ctx.used_constants.insert(name.clone());
                    return Ok(arael_sym::symbol(&name));
                }
                return Err(syn::Error::new_spanned(
                    ident,
                    format!("unknown variable '{name}' in fit expression (not a Param field, plain field, or let binding)"),
                ));
            }
            Err(syn::Error::new_spanned(
                expr,
                "qualified paths not supported in fit expression",
            ))
        }

        Expr::Field(ef) => {
            // e.x where e is the loop variable
            if let Expr::Path(base_path) = ef.base.as_ref()
                && let Some(base_ident) = base_path.path.get_ident()
                    && *base_ident == ctx.loop_var
                        && let syn::Member::Named(field_name) = &ef.member {
                            let sym_name =
                                format!("{}_{}", ctx.loop_var, field_name);
                            ctx.data_fields.insert(field_name.to_string());
                            return Ok(arael_sym::symbol(&sym_name));
                        }
            Err(syn::Error::new_spanned(
                expr,
                "only loop_variable.field access is supported in fit expressions",
            ))
        }

        Expr::Binary(eb) => {
            let left = syn_expr_to_sym(&eb.left, ctx)?;
            let right = syn_expr_to_sym(&eb.right, ctx)?;
            match eb.op {
                syn::BinOp::Add(_) => Ok(left + right),
                syn::BinOp::Sub(_) => Ok(left - right),
                syn::BinOp::Mul(_) => Ok(left * right),
                syn::BinOp::Div(_) => Ok(left / right),
                _ => Err(syn::Error::new_spanned(
                    eb.op,
                    "only +, -, *, / operators are supported in fit expressions",
                )),
            }
        }

        Expr::Unary(eu) => {
            let inner = syn_expr_to_sym(&eu.expr, ctx)?;
            match eu.op {
                syn::UnOp::Neg(_) => Ok(-inner),
                _ => Err(syn::Error::new_spanned(
                    expr,
                    "only unary negation is supported in fit expressions",
                )),
            }
        }

        Expr::Lit(el) => match &el.lit {
            syn::Lit::Float(lf) => {
                let val: f64 = lf.base10_parse()?;
                Ok(arael_sym::constant(val))
            }
            syn::Lit::Int(li) => {
                let val: i64 = li.base10_parse()?;
                Ok(arael_sym::constant(val as f64))
            }
            _ => Err(syn::Error::new_spanned(
                expr,
                "only numeric literals are supported in fit expressions",
            )),
        },

        Expr::Call(ec) => {
            if let Expr::Path(func_path) = ec.func.as_ref()
                && let Some(func_name) = func_path.path.get_ident() {
                    let args: Vec<arael_sym::E> = ec
                        .args
                        .iter()
                        .map(|a| syn_expr_to_sym(a, ctx))
                        .collect::<Result<_, _>>()?;

                    let fname = func_name.to_string();
                    return match arael_sym::function_by_name(&fname) {
                        Some(arael_sym::FunctionRef::Unary(f)) => expect_sym_unary(func_name, args, f),
                        Some(arael_sym::FunctionRef::Binary(f)) => expect_sym_binary(func_name, args, f),
                        Some(arael_sym::FunctionRef::Ternary(f)) => expect_sym_ternary(func_name, args, f),
                        None => Err(syn::Error::new_spanned(
                            func_name,
                            format!("unknown function '{fname}' in fit expression"),
                        )),
                    };
                }
            Err(syn::Error::new_spanned(
                expr,
                "unsupported function call in fit expression",
            ))
        }

        Expr::Paren(ep) => syn_expr_to_sym(&ep.expr, ctx),

        Expr::Group(eg) => syn_expr_to_sym(&eg.expr, ctx),

        _ => Err(syn::Error::new_spanned(
            expr,
            "unsupported expression type in fit expression",
        )),
    }
}

fn expect_sym_unary(
    name: &syn::Ident,
    args: Vec<arael_sym::E>,
    f: fn(arael_sym::E) -> arael_sym::E,
) -> syn::Result<arael_sym::E> {
    if args.len() != 1 {
        return Err(syn::Error::new_spanned(
            name,
            format!("{} expects 1 argument, got {}", name, args.len()),
        ));
    }
    Ok(f(args.into_iter().next().unwrap()))
}

fn expect_sym_binary(
    name: &syn::Ident,
    args: Vec<arael_sym::E>,
    f: fn(arael_sym::E, arael_sym::E) -> arael_sym::E,
) -> syn::Result<arael_sym::E> {
    if args.len() != 2 {
        return Err(syn::Error::new_spanned(
            name,
            format!("{} expects 2 arguments, got {}", name, args.len()),
        ));
    }
    let mut it = args.into_iter();
    Ok(f(it.next().unwrap(), it.next().unwrap()))
}

fn expect_sym_ternary(
    name: &syn::Ident,
    args: Vec<arael_sym::E>,
    f: fn(arael_sym::E, arael_sym::E, arael_sym::E) -> arael_sym::E,
) -> syn::Result<arael_sym::E> {
    if args.len() != 3 {
        return Err(syn::Error::new_spanned(
            name,
            format!("{} expects 3 arguments, got {}", name, args.len()),
        ));
    }
    let mut it = args.into_iter();
    Ok(f(it.next().unwrap(), it.next().unwrap(), it.next().unwrap()))
}

// ---------------------------------------------------------------------------
// Code generation: calc_cost, calc_grad_hessian, fit, fit_with
// ---------------------------------------------------------------------------

fn generate_fit_impl(
    name: &syn::Ident,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
    fit: &FitAttr,
) -> syn::Result<TokenStream2> {
    // 1. Classify fields
    let mut param_names: Vec<String> = Vec::new();
    let mut constant_names: HashSet<String> = HashSet::new();
    let data_field_name = fit.data_field.to_string();

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let field_name = ident.to_string();
        let attr = parse_arael_attr(&field.attrs)?;
        if matches!(attr, Some(AraelAttr::Skip) | Some(AraelAttr::Compute(_))) {
            continue;
        }
        if is_param_type(&field.ty) {
            param_names.push(field_name);
        } else if field_name != data_field_name {
            constant_names.insert(field_name);
        }
    }

    // 2. Process body: convert let bindings + final expression to sym
    let mut ctx = SymContext {
        loop_var: fit.loop_var.to_string(),
        param_names: param_names.clone(),
        constant_names,
        let_bindings: HashMap::new(),
        data_fields: HashSet::new(),
        used_constants: HashSet::new(),
    };

    let mut residual: Option<arael_sym::E> = None;

    for stmt in &fit.body_stmts {
        match stmt {
            Stmt::Local(local) => {
                let binding_name = match &local.pat {
                    Pat::Ident(pi) => pi.ident.to_string(),
                    _ => {
                        return Err(syn::Error::new_spanned(
                            &local.pat,
                            "only simple `let name = expr;` bindings are supported in fit expressions",
                        ))
                    }
                };
                let init = local.init.as_ref().ok_or_else(|| {
                    syn::Error::new_spanned(local, "let binding must have an initializer")
                })?;
                let sym_expr = syn_expr_to_sym(&init.expr, &mut ctx)?;
                ctx.let_bindings.insert(binding_name, sym_expr);
            }
            Stmt::Expr(expr, _) => {
                residual = Some(syn_expr_to_sym(expr, &mut ctx)?);
            }
            _ => {
                return Err(syn::Error::new_spanned(
                    fit.body_stmts.first(),
                    "only let bindings and expressions are supported in fit body",
                ))
            }
        }
    }

    let residual = residual.ok_or_else(|| {
        syn::Error::new_spanned(
            &fit.data_field,
            "fit body must end with a residual expression",
        )
    })?;

    // 3. Differentiate w.r.t. each param
    let n = param_names.len();
    let derivatives: Vec<arael_sym::E> = param_names
        .iter()
        .map(|p| residual.diff(p.as_str()))
        .collect();

    // 4. Generate Rust code strings and parse to syn::Expr
    let r_code = residual.to_rust("f32");
    let r_expr: Expr = syn::parse_str(&r_code).map_err(|e| {
        syn::Error::new_spanned(
            &fit.data_field,
            format!("failed to parse generated residual code: {e}\ngenerated: {r_code}"),
        )
    })?;

    let dr_exprs: Vec<Expr> = derivatives
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let code = d.to_rust("f32");
            syn::parse_str(&code).map_err(|e| {
                syn::Error::new_spanned(
                    &fit.data_field,
                    format!(
                        "failed to parse generated derivative code for param '{}': {e}\ngenerated: {code}",
                        param_names[i]
                    ),
                )
            })
        })
        .collect::<Result<_, _>>()?;

    // 5. Build code fragments

    // Param unpacking: let a = params[0]; let b = params[1]; ...
    let param_unpack: Vec<TokenStream2> = param_names
        .iter()
        .enumerate()
        .map(|(idx, pname)| {
            let id = proc_macro2::Ident::new(pname, proc_macro2::Span::call_site());
            quote! { let #id = params[#idx]; }
        })
        .collect();

    // Constant binding: let sigma = self.sigma; ...
    let mut sorted_constants: Vec<&String> = ctx.used_constants.iter().collect();
    sorted_constants.sort();
    let constant_bind: Vec<TokenStream2> = sorted_constants
        .iter()
        .map(|cname| {
            let id = proc_macro2::Ident::new(cname, proc_macro2::Span::call_site());
            quote! { let #id = self.#id; }
        })
        .collect();

    // Data field binding: let e_x = e.x; ...
    let mut sorted_data_fields: Vec<&String> = ctx.data_fields.iter().collect();
    sorted_data_fields.sort();
    let loop_var_id = &fit.loop_var;
    let data_bind: Vec<TokenStream2> = sorted_data_fields
        .iter()
        .map(|fname| {
            let sym_id = proc_macro2::Ident::new(
                &format!("{}_{}", ctx.loop_var, fname),
                proc_macro2::Span::call_site(),
            );
            let field_id = proc_macro2::Ident::new(fname, proc_macro2::Span::call_site());
            quote! { let #sym_id = #loop_var_id.#field_id; }
        })
        .collect();

    // Derivative bindings: let __dr_0 = <expr>; ...
    let dr_idents: Vec<proc_macro2::Ident> = (0..n)
        .map(|i| proc_macro2::Ident::new(&format!("__dr_{i}"), proc_macro2::Span::call_site()))
        .collect();
    let dr_bindings: Vec<TokenStream2> = (0..n)
        .map(|i| {
            let dr_id = &dr_idents[i];
            let dr_expr = &dr_exprs[i];
            quote! { let #dr_id: f32 = #dr_expr; }
        })
        .collect();

    // Gradient accumulation
    let grad_accum: Vec<TokenStream2> = (0..n)
        .map(|i| {
            let dr_id = &dr_idents[i];
            quote! { grad[#i] += 2.0_f32 * __r * #dr_id; }
        })
        .collect();

    // Hessian accumulation (upper triangle only)
    let hessian_accum: Vec<TokenStream2> = (0..n)
        .flat_map(|i| {
            let dr_idents = &dr_idents;
            (i..n).map(move |j| {
                let idx = i * n + j;
                let dr_i = &dr_idents[i];
                let dr_j = &dr_idents[j];
                quote! { hessian[#idx] += 2.0_f32 * #dr_i * #dr_j; }
            })
        })
        .collect();

    // Symmetric fill (lower triangle)
    let hessian_symmetry: Vec<TokenStream2> = (0..n)
        .flat_map(|i| {
            (i + 1..n).map(move |j| {
                let ij = i * n + j;
                let ji = j * n + i;
                quote! { hessian[#ji] = hessian[#ij]; }
            })
        })
        .collect();

    let data_field_id = &fit.data_field;

    Ok(quote! {
        impl arael::simple_lm::LmProblem<f32> for #name {
            fn calc_cost(&mut self, params: &[f32]) -> f32 {
                #(#param_unpack)*
                #(#constant_bind)*
                let mut __cost = 0.0_f32;
                for #loop_var_id in &self.#data_field_id {
                    #(#data_bind)*
                    let __r: f32 = #r_expr;
                    __cost += __r * __r;
                }
                __cost
            }

            fn calc_grad_hessian_dense(
                &mut self,
                params: &[f32],
                grad: &mut [f32],
                hessian: &mut [f32],
            ) -> f32 {
                #(#param_unpack)*
                #(#constant_bind)*
                grad.iter_mut().for_each(|g| *g = 0.0);
                hessian.iter_mut().for_each(|h| *h = 0.0);
                let mut __cost = 0.0_f32;
                for #loop_var_id in &self.#data_field_id {
                    #(#data_bind)*
                    let __r: f32 = #r_expr;
                    __cost += __r * __r;
                    #(#dr_bindings)*
                    #(#grad_accum)*
                    #(#hessian_accum)*
                }
                #(#hessian_symmetry)*
                __cost
            }

            fn calc_grad_hessian_band(
                &mut self,
                _params: &[f32],
                _grad: &mut [f32],
                _band: &mut [f32],
                _kd: usize,
            ) -> Result<f32, arael::simple_lm::BandError> {
                unimplemented!("fit models do not support band assembly")
            }

            fn calc_grad_hessian_sparse(
                &mut self,
                _params: &[f32],
                _grad: &mut [f32],
                _coo: &mut arael::simple_lm::CooMatrix<f32>,
            ) -> f32 {
                unimplemented!("fit models do not support sparse assembly")
            }

            fn calc_grad_hessian_sparse_direct(
                &mut self,
                _params: &[f32],
                _grad: &mut [f32],
                _csc: &mut arael::simple_lm::CscMatrix<f32>,
            ) -> f32 {
                unimplemented!("fit models do not support sparse direct assembly")
            }

            fn calc_grad_hessian_sparse_indexed(
                &mut self,
                _params: &[f32],
                _grad: &mut [f32],
                _vals: &mut [f32],
                _positions: &[usize],
            ) -> f32 {
                unimplemented!("fit models do not support sparse indexed assembly")
            }
        }

        impl #name {
            pub fn fit(&mut self) -> arael::simple_lm::LmResult<f32> {
                self.fit_with(&arael::simple_lm::LmConfig::default())
            }

            pub fn fit_with(
                &mut self,
                config: &arael::simple_lm::LmConfig<f32>,
            ) -> arael::simple_lm::LmResult<f32> {
                let mut __params = std::vec::Vec::new();
                arael::model::Model::serialize_params32(self, &mut __params);
                let __result = arael::simple_lm::solve_f32(
                    &__params,
                    self,
                    config,
                );
                arael::model::Model::deserialize_params32(self, &__result.x);
                __result
            }
        }
    })
}
