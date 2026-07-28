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
mod sidecar;

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
    Quat,
    Struct(String),        // reference to another registered struct
    OptionalStruct(String), // Option<T> wrapping a struct
    Skip,
}

#[derive(Clone, Debug, Default)]
struct SymLayout {
    fields: Vec<(String, SymFieldType)>,
    /// Field names that are Vec/Deque/Arena collections of a struct. A
    /// collection field's element type is still recorded in `fields` as
    /// `SymFieldType::Struct(elem)` (indistinguishable there from a direct
    /// struct field or a `Ref<T>`); this set is what lets the entity-location
    /// resolver tell "iterate this collection" from "single struct field" when
    /// walking the model tree to any depth.
    collection_fields: Vec<String>,
    param_fields: Vec<String>,       // field names that are Param<T>
    ref_paths: Vec<(String, String)>, // (field_name, resolution_path) for #[arael(ref = ...)]
    euler_angle_fields: Vec<String>,  // field names detected as SimpleEulerAngleParam
    universal_euler_angle_fields: Vec<String>, // field names detected as EulerAngleParam
    universal_rotvec_fields: Vec<String>, // field names detected as QuaternionParam (rotation-vector delta)
    /// `#[arael(symbolic = <expr>)]` fields: (field name, expression source).
    /// Body reads of the field expand to the expression, evaluated over the
    /// struct's OWN fields (params as param symbols) -- a derivative-carrying
    /// computed field. Declaration order; later entries may read earlier ones.
    symbolic_fields: Vec<(String, String)>,
    /// `#[arael(deriv = <of>, by = <param>)]` fields: (field name, of, by).
    /// Declared Jacobian caches, filled by the symbolic precompute and read
    /// by constraint Jacobians in place of the re-derived expression.
    deriv_fields: Vec<(String, String, String)>,
    /// Atom-cached symbolic fields: `(field, by_param, [(atom_expr, cache_field)])`.
    /// The computed `field` is not stored; instead named sub-expressions of it
    /// (its "atoms", e.g. `sin(angle)`, `cos(angle)`) are precomputed into
    /// scalar `cache_field`s. Both the field's value reads AND its derivative
    /// reads (by `by_param`) redirect to those scalar caches, so the field
    /// needs no storage and no re-derived trig -- e.g. the 2x2 rotation matrix
    /// and its Jacobian both read the two stored `sin`/`cos` scalars.
    atom_cached_fields: Vec<(String, String, Vec<(String, String)>)>,
    /// `#[arael(component)]`: this struct is a compound parameter whose
    /// Params fold into the owning struct's span.
    component: bool,
    /// Field name of `#[arael(constraint_index)]` u32 field, if present.
    constraint_index_field: Option<String>,
    /// Field name of the struct's `SelfBlock<Self>` — detected automatically
    /// during `#[arael::model]` expansion. Required for every params-having
    /// Model after the CrossBlock/TripletBlock refactor: the self-block is
    /// the single home for that entity's gradient + A-A Hessian diagonal,
    /// so cross constraints need to know the field name to write to it.
    self_block_field: Option<String>,
    /// Fields whose type is an UNRECOGNIZED generic wrapper naming another
    /// type: (field, wrapper, held type name). Containers are dispatched by
    /// literal last-segment name, so an aliased container (`use refs::Vec
    /// as RVec` + `pts: RVec<P>`) reads as an opaque field and `P` silently
    /// stops being containment there. Recorded at classification; the
    /// moment `held` turns out to be a registered model type (either
    /// definition order) expansion fails loudly instead.
    suspect_wrappers: Vec<(String, String, String)>,
    /// Solve precision of this struct's block fields: `None` = no blocks,
    /// otherwise (block field name, "f32" | "f64" | "generic"). "generic"
    /// = the block scalar is the struct's own type parameter, resolved per
    /// instantiation. Block precision must match the root's solve
    /// precision; storage/Param precision is free (the walks cast at the
    /// boundary). The root check reads this to reject mismatches with an
    /// error naming the field instead of an E0308 in generated code.
    block_precision: Option<(String, String)>,
    /// Fields whose element type is spelled with an explicit float first
    /// argument (`nodes: Vec<G<f32>>`, `g: G<f32>`): (field, element type,
    /// "f32" | "f64"). For a generic model element this is its block
    /// precision at this holding -- the layout alone loses the spelling,
    /// so the root check resolves "generic" through these records.
    inst_precisions: Vec<(String, String, String)>,
    /// Names of `TripletBlock<T>` fields on this struct. Lets a child's
    /// `[hb, parent.<field>]` block spec validate the named field against
    /// the containing parent (block fields are `Skip` in `fields`, so the
    /// name is otherwise unrecoverable from the layout).
    triplet_block_fields: Vec<String>,
    /// `#[arael(root)]` struct. A root's expansion consumes the constraint
    /// stash; the registration-time ordering guard errors on any model
    /// type that registers later yet is reachable from this root -- its
    /// constraints were silently dropped.
    is_root: bool,
    /// Every declared field with its type spelling as written (normalized
    /// whitespace), in declaration order. `fields` collapses data types
    /// (`f64` / `bool` / `String` all read as Scalar or Struct), so this
    /// is what the JSON sidecar emits for interface generators that need
    /// real types. Injected fields (euler rotation caches) are absent.
    spelled_types: Vec<(String, String)>,
}

/// Total optimizable scalars of a registered type, following
/// `#[arael(component)]` struct fields recursively.
fn registry_param_total(type_name: &str) -> u32 {
    let Some(l) = registry_lookup(type_name) else { return 0 };
    let mut n = 0u32;
    for (f, sft) in &l.fields {
        if l.param_fields.contains(f) {
            n += match sft {
                SymFieldType::Scalar => 1,
                SymFieldType::Vec2 => 2,
                SymFieldType::Vec3 => 3,
                _ => 0,
            };
        } else if let SymFieldType::Struct(inner) = sft
            && registry_lookup(inner).map(|x| x.component).unwrap_or(false)
        {
            n += registry_param_total(inner);
        }
    }
    n
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

/// One `pub` model struct's contribution to the crate's export bundle
/// (`arael::export_models!()` -> the crate's `arael_import!` macro).
#[derive(Clone)]
struct ExportEntry {
    name: String,
    /// Original struct tokens, before block rewriting and attr stripping.
    tokens: String,
    /// Param count computed at definition time; the importer recomputes
    /// from the tokens and cross-checks (catches macro version skew).
    param_count: u32,
    /// Why the struct is NOT importable (a non-pub field), if so. Carried
    /// as a tombstone so the importer's failure names the reason.
    excluded: Option<String>,
}

struct Registry {
    // BTreeMap: parent-struct scans iterate layouts to find the struct
    // containing a given child type; HashMap order made that pick (and
    // with it generated code) differ between rustc invocations when
    // several structs embed the same type. BTreeMap makes it the
    // alphabetically first -- deterministic (B11).
    layouts: std::collections::BTreeMap<String, SymLayout>,
    constraints: Vec<StashedConstraint>,
    functions: HashMap<String, UserFunction>,
    /// Every `CrossBlock<A, B>` entity pair seen anywhere in the model, in
    /// declaration order. This is the coupling graph the Schur detector
    /// reasons over: a Hessian tile can join an A to a B only if such a
    /// block exists (the macro must emit a block for every J^T J pair), so
    /// a set of types with no CrossBlock among them is provably
    /// uncoupled -- exactly what marginalization requires.
    cross_pairs: std::collections::BTreeSet<(String, String)>,
    /// Export bundle accumulated in definition order (which IS dependency
    /// order -- the crate could not compile otherwise). Drained into the
    /// crate's `arael_import!` macro by `export_models!()`.
    exports: Vec<ExportEntry>,
    /// Names already imported via `__register_model!` this session --
    /// a diamond import (the same bundle reached twice) re-validates the
    /// layout but must not stash constraints or emit consts again.
    imported: std::collections::HashSet<String>,
    /// Tombstones: types that exist in an imported crate but were not
    /// exported, with the reason. Lookup failures report these precisely.
    excluded: HashMap<String, String>,
}

static SYM_REGISTRY: Mutex<Option<Registry>> = Mutex::new(None);

fn registry_init() -> Registry {
    Registry {
        layouts: std::collections::BTreeMap::new(),
        constraints: Vec::new(),
        functions: HashMap::new(),
        cross_pairs: std::collections::BTreeSet::new(),
        exports: Vec::new(),
        imported: std::collections::HashSet::new(),
        excluded: HashMap::new(),
    }
}

/// Append to the export bundle; a name already present (a transitively
/// re-imported struct next to a direct import) keeps its first entry.
fn registry_stash_export(entry: ExportEntry) {
    let mut guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let reg = guard.get_or_insert_with(registry_init);
    if !reg.exports.iter().any(|e| e.name == entry.name) {
        reg.exports.push(entry);
    }
}

fn registry_exports() -> Vec<ExportEntry> {
    let guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    guard.as_ref().map(|r| r.exports.clone()).unwrap_or_default()
}

/// Mark a type as imported; false if it already was (diamond import).
fn registry_mark_imported(name: &str) -> bool {
    let mut guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let reg = guard.get_or_insert_with(registry_init);
    reg.imported.insert(name.to_string())
}

fn registry_store_tombstone(name: &str, reason: &str) {
    let mut guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let reg = guard.get_or_insert_with(registry_init);
    reg.excluded.entry(name.to_string()).or_insert_with(|| reason.to_string());
}

pub(crate) fn registry_excluded_reason(name: &str) -> Option<String> {
    let guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    guard.as_ref().and_then(|r| r.excluded.get(name).cloned())
}

/// Record a `CrossBlock<A, B>` coupling (unordered; stored sorted).
fn registry_record_cross(a: &str, b: &str) {
    let mut guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let reg = guard.get_or_insert_with(registry_init);
    let pair = if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    };
    reg.cross_pairs.insert(pair);
}

/// Snapshot of every recorded CrossBlock coupling.
fn registry_cross_pairs() -> Vec<(String, String)> {
    let guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    guard.as_ref().map(|r| r.cross_pairs.iter().cloned().collect()).unwrap_or_default()
}

/// Returns an error if a DIFFERENT layout is already registered under
/// this name: the registry is keyed by bare struct name, so two
/// #[arael::model] structs with the same name (different modules) would
/// silently last-write-win and corrupt each other's generated code.
/// Re-registering an identical layout (e.g. cfg-duplicated expansion)
/// stays allowed.
fn registry_store(name: &str, layout: SymLayout) -> Result<(), String> {
    let mut guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let reg = guard.get_or_insert_with(registry_init);
    if let Some(prev) = reg.layouts.get(name)
        && format!("{:?}", prev) != format!("{:?}", layout) {
            return Err(format!(
                "a different #[arael::model] struct named `{}` is already registered \
                 (the registry is keyed by bare struct name; rename one of them)", name));
        }
    // A suspect wrapper recorded earlier (an unrecognized container whose
    // held type was not yet registered) naming THIS type is now proven to
    // hold a model type -- that containment is silently dropped, so fail
    // loudly. This catches the holder-expanded-first definition order; the
    // holder's own expansion catches the other. The layout is still
    // inserted so downstream expansions do not cascade secondary errors.
    let suspect_hit = if layout.fields.is_empty() { None } else {
        reg.layouts.iter()
            .flat_map(|(holder, l)| l.suspect_wrappers.iter()
                .map(move |s| (holder.as_str(), s)))
            .chain(layout.suspect_wrappers.iter().map(|s| (name, s)))
            .find(|(_, (_, _, held))| held == name)
            .map(|(holder, (field, wrapper, _))| format!(
                "`{}` is a model type, but field `{}` of `{}` holds it inside \
                 unrecognized container `{}` -- containers are recognized by \
                 literal name (Vec, Deque, Arena, Option, Ref), aliases are \
                 invisible to the macro; spell the container literally, or mark \
                 that field `#[arael(skip)]` if it is deliberately outside the \
                 model", name, field, holder, wrapper))
    };
    // Ordering guard, registration side: a type registering AFTER a root
    // that can already reach it means that root's expansion consumed the
    // constraint stash without this type -- its constraints were silently
    // dropped (params still serialize through the runtime recursion, so
    // the model would solve quietly wrong). BFS each expanded root's
    // layout through Struct/OptionalStruct links; layouts are skip-aware
    // by construction. Fieldless layouts (enums) carry nothing droppable;
    // an identical re-registration passed this check the first time.
    let ordering_hit = if layout.fields.is_empty() || reg.layouts.contains_key(name) {
        None
    } else {
        let links = |l: &SymLayout| -> Vec<String> {
            l.fields.iter().filter_map(|(_, sft)| match sft {
                SymFieldType::Struct(s) | SymFieldType::OptionalStruct(s) => Some(s.clone()),
                _ => None,
            }).collect()
        };
        reg.layouts.iter()
            .filter(|(rname, rl)| rl.is_root && rname.as_str() != name)
            .find(|(_, rl)| {
                let mut seen = std::collections::HashSet::new();
                let mut queue = links(rl);
                while let Some(t) = queue.pop() {
                    if t == name { return true; }
                    if !seen.insert(t.clone()) { continue; }
                    if let Some(l) = reg.layouts.get(&t) {
                        queue.extend(links(l));
                    }
                }
                false
            })
            .map(|(rname, _)| format!(
                "`{}` is reachable from root `{}` but defined after it -- the root's \
                 expansion has already run, so `{}`'s constraints were dropped; define \
                 it (or import its crate's bundle) BEFORE the root",
                name, rname, name))
    };
    reg.layouts.insert(name.to_string(), layout);
    match suspect_hit.or(ordering_hit) {
        Some(msg) => Err(msg),
        None => Ok(()),
    }
}

fn registry_lookup(name: &str) -> Option<SymLayout> {
    let guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    guard.as_ref().and_then(|reg| reg.layouts.get(name).cloned())
        .or_else(|| builtin_component_layout(name))
}

/// Layouts of the in-tree components (defined in arael's own source, so
/// they are never expanded in a downstream crate's macro session). The
/// runtime halves live in the arael crate; these give the macro the same
/// knowledge a `#[arael(component)]` expansion would have registered.
fn builtin_component_layout(name: &str) -> Option<SymLayout> {
    match name {
        // S^2 direction: reference quaternion frame (x-axis = direction),
        // 2-DOF body-frame delta about the frame's y/z axes. The embed is
        // the first column of the small-rotation matrix of the normalized
        // quaternion (1, (0, d.x, d.y)/2), rotated by the cached reference
        // rotation -- exact on the sphere for every delta.
        // A rigid transform: reference frame plus a coupled 6-DOF step.
        // The translation half of the step is carried through the rotation
        // happening alongside it (see crate::se3).
        "TransformParam" | "TransformParamF" => Some(SymLayout {
            fields: vec![
                ("ref_rotation".to_string(), SymFieldType::Mat3),
                ("ref_translation".to_string(), SymFieldType::Vec3),
                ("w".to_string(), SymFieldType::Vec3),
                ("d".to_string(), SymFieldType::Vec3),
                ("rotation_matrix".to_string(), SymFieldType::Mat3),
                ("translation".to_string(), SymFieldType::Vec3),
                ("rotation_matrix_dw".to_string(), SymFieldType::Skip),
                ("translation_dd".to_string(), SymFieldType::Skip),
                ("translation_dw".to_string(), SymFieldType::Skip),
            ],
            collection_fields: Vec::new(),
            param_fields: vec!["w".to_string(), "d".to_string()],
            ref_paths: Vec::new(),
            euler_angle_fields: Vec::new(),
            universal_euler_angle_fields: Vec::new(),
            universal_rotvec_fields: Vec::new(),
            symbolic_fields: vec![
                ("rotation_matrix".to_string(),
                 "ref_rotation * matrix3sym::from_rotation_vector_small(w)".to_string()),
                ("translation".to_string(),
                 "{ let carried = d + (w % d) * 0.5 \
                    + (w % (w % d)) * 0.16666666666666666; \
                    ref_translation + ref_rotation * carried }".to_string()),
            ],
            deriv_fields: vec![
                ("rotation_matrix_dw".to_string(), "rotation_matrix".to_string(), "w".to_string()),
                ("translation_dd".to_string(), "translation".to_string(), "d".to_string()),
                ("translation_dw".to_string(), "translation".to_string(), "w".to_string()),
            ],
            component: true,
            ..Default::default()
        }),
        // 2D angle: optimized directly (no reference frame -- there is no
        // gimbal lock in one dimension), with the rotation matrix and its
        // derivative cached so a body that rotates through the angle reads
        // constants instead of rebuilding sin/cos per observation.
        "AngleParam" | "AngleParamF" => Some(SymLayout {
            fields: vec![
                ("angle".to_string(), SymFieldType::Scalar),
                ("sin".to_string(), SymFieldType::Scalar),
                ("cos".to_string(), SymFieldType::Scalar),
            ],
            collection_fields: Vec::new(),
            param_fields: vec!["angle".to_string()],
            ref_paths: Vec::new(),
            euler_angle_fields: Vec::new(),
            universal_euler_angle_fields: Vec::new(),
            universal_rotvec_fields: Vec::new(),
            symbolic_fields: vec![
                ("rotation_matrix".to_string(),
                 "matrix2sym::rotation(angle)".to_string()),
            ],
            deriv_fields: Vec::new(),
            // The rotation matrix is computed, not stored: its sin/cos atoms
            // are precomputed into the `sin`/`cos` scalars, and both its value
            // and its Jacobian (by `angle`) read those two floats -- no stored
            // matrix, no re-derived trig.
            atom_cached_fields: vec![
                ("rotation_matrix".to_string(), "angle".to_string(), vec![
                    ("sin(angle)".to_string(), "sin".to_string()),
                    ("cos(angle)".to_string(), "cos".to_string()),
                ]),
            ],
            component: true,
            ..Default::default()
        }),
        "UnitVecParam" | "UnitVecParamF" => Some(SymLayout {
            fields: vec![
                ("rot".to_string(), SymFieldType::Mat3),
                ("d".to_string(), SymFieldType::Vec2),
                ("unit".to_string(), SymFieldType::Vec3),
                ("unit_d".to_string(), SymFieldType::Skip),
            ],
            collection_fields: Vec::new(),
            param_fields: vec!["d".to_string()],
            ref_paths: Vec::new(),
            euler_angle_fields: Vec::new(),
            universal_euler_angle_fields: Vec::new(),
            universal_rotvec_fields: Vec::new(),
            symbolic_fields: vec![
                ("unit".to_string(),
                 "{ let s2 = 1.0 + (d.x * d.x + d.y * d.y) * 0.25; \
                    let local = vect3sym::from_components(\
                        1.0 - (d.x * d.x + d.y * d.y) / (2.0 * s2), \
                        d.y / s2, 0.0 - d.x / s2); \
                    rot * local }".to_string()),
            ],
            deriv_fields: vec![
                ("unit_d".to_string(), "unit".to_string(), "d".to_string()),
            ],
            component: true,
            ..Default::default()
        }),
        _ => None,
    }
}

fn registry_stash_constraint(c: StashedConstraint) {
    let mut guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let reg = guard.get_or_insert_with(registry_init);
    reg.constraints.push(c);
}

// Clone, never take: a crate can define several #[arael(root)] models,
// and each root selects its own constraints via the reachability filter.
// A take here would hand the first root everything and later roots
// nothing, silently generating no-op solvers for them.
fn registry_constraints() -> Vec<StashedConstraint> {
    let guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    guard.as_ref().map(|reg| reg.constraints.clone()).unwrap_or_default()
}

#[allow(dead_code)]
pub(crate) fn registry_store_function(name: &str, f: UserFunction) {
    let mut guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    let reg = guard.get_or_insert_with(registry_init);
    reg.functions.insert(name.to_string(), f);
}

#[allow(dead_code)]
pub(crate) fn registry_lookup_function(name: &str) -> Option<UserFunction> {
    let guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
    guard.as_ref().and_then(|reg| reg.functions.get(name).cloned())
}

/// Snapshot every registered user function. Used by the constraint-body
/// interpreter to build a full `FunctionBag` for `parse_with_functions`.
#[allow(dead_code)]
pub(crate) fn registry_all_functions() -> Vec<UserFunction> {
    let guard = SYM_REGISTRY.lock().unwrap_or_else(|p| p.into_inner());
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
/// ## `#[arael(fit(data, |e| expr))]` / `#[arael(fit64(data, |e| expr))]`
///
/// Auto-generate a complete `LmProblem` implementation for simple curve
/// fitting. Iterates over `data`, evaluates the residual expression for each
/// entry, and generates `calc_cost()` / `calc_grad_hessian_*()` (dense only)
/// plus the `FitProblem` impl, which unlocks FitProblem's default `fit()` /
/// `fit_with()` entry points (`use arael::simple_lm::FitProblem` or the
/// prelude to call them). `fit(...)` is f32 throughout; `fit64(...)` is the
/// f64 variant.
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
/// A trailing `loss = |s| rho(s)` applies a block M-estimator to each point's
/// squared residual `s = r^2`: the cost becomes `sum rho(s)` and each point's
/// gradient and Gauss-Newton Hessian are scaled by the weight `rho'(s)`. Use a
/// built-in `loss_huber` / `loss_cauchy` / `loss_tukey`, or any arael-sym
/// expression in `s`:
///
/// ```ignore
/// #[arael(fit(data, |e| a * e.x + b - e.y, loss = |s| loss_cauchy(s, k)))]
/// struct RobustLine { a: Param<f32>, b: Param<f32>, data: Vec<Pt>, k: f32 }
/// ```
///
/// ## `#[arael(root)]` / `#[arael(root, f32)]`
///
/// Mark this struct as the optimization root. Triggers code generation for
/// all stashed `constraint` attributes in the model hierarchy. Generates
/// the `LmProblem` trait implementation (`calc_cost()`, the
/// `calc_grad_hessian_*` assembly family, `advance()`) and the
/// `RootProblem` impl (`serialize` / `deserialize` -- the parameter round
/// trip), which unlocks LmProblem's default solve entry points
/// `solve_with` / `solve_dense` / `solve_sparse` on the type
/// (`use arael::simple_lm::LmProblem` to call them).
/// Also generates suffixed `serialize64()` / `deserialize64()` (and
/// `32`) convenience methods and `__set_block_indices()` /
/// `__compute_blocks()` internals.
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
/// - `fast_atan` -- the generated code calls `arael::utils::fast_atan` /
///   `fast_atan2` (max error < 1e-6 radians) instead of the libm atan /
///   atan2 for every occurrence in residuals, gradients, Hessians, and
///   Jacobians. Derivatives are unaffected (they are the exact rational
///   forms either way).
/// - `marginalize(field, ...)` -- marks landmark-style fields (small
///   parameter blocks coupled to other parameters but never to each
///   other) for the sparse solver to eliminate first. Generates
///   `RootProblem::marginalize_hint()` with the fields' parameter
///   ranges; `solve_sparse` feeds it to
///   `SparseFaer::with_marginalize`, which orders those parameters
///   first in the factorization (replacing AMD).
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

/// Emit this crate's `arael_import!` macro, carrying every `pub`
/// `#[arael::model]` struct and enum defined ABOVE this invocation
/// (macro expansion is top-down: place it after all model definitions,
/// e.g. at the bottom of lib.rs). An importing crate then registers all
/// of them in one line:
///
/// ```ignore
/// use model_crate::{Pose, Frine};
/// model_crate::arael_import!();   // before defining models over them
/// ```
///
/// A `pub` struct with a non-pub field is carried as a tombstone: it
/// cannot be used from another crate (generated code reads fields
/// directly), and the importer's error says which field.
#[proc_macro]
pub fn export_models(input: TokenStream) -> TokenStream {
    if !input.is_empty() {
        return syn::Error::new(proc_macro2::Span::call_site(),
            "export_models! takes no arguments").to_compile_error().into();
    }
    let entries = registry_exports();
    let mut body = TokenStream2::new();
    let mut names: Vec<String> = Vec::new();
    for e in &entries {
        let name_ident = syn::Ident::new(&e.name, proc_macro2::Span::call_site());
        if let Some(reason) = &e.excluded {
            body.extend(quote! {
                ::arael::__register_model! { @excluded #name_ident #reason ; }
            });
        } else {
            let toks: TokenStream2 = match e.tokens.parse() {
                Ok(t) => t,
                Err(err) => return syn::Error::new(proc_macro2::Span::call_site(),
                    format!("internal: stashed tokens for `{}` do not re-parse: {}",
                        e.name, err)).to_compile_error().into(),
            };
            let count = e.param_count as usize;
            body.extend(quote! {
                ::arael::__register_model! { @expect #name_ident #count ; #toks }
            });
            names.push(e.name.clone());
        }
    }
    let doc = format!(
        "Registers this crate's arael model layouts ({}) in the importing \
         crate's macro session. Invoke before defining models over them.",
        names.join(", "));
    quote! {
        #[doc = #doc]
        #[macro_export]
        macro_rules! arael_import {
            () => { #body };
        }
    }.into()
}

/// Cross-crate registration: re-runs the registration half of
/// `#[arael::model]` on a struct's original tokens inside the IMPORTING
/// crate's macro session, emitting only the `<Name>_PARAM_COUNT` const
/// (which block rewrites in the importing crate resolve). Invoked by the
/// `arael_import!` macros that `export_models!()` generates.
#[doc(hidden)]
#[proc_macro]
pub fn __register_model(input: TokenStream) -> TokenStream {
    match register_model_import(input.into()) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn register_model_import(ts: TokenStream2) -> syn::Result<TokenStream2> {
    let sp = proc_macro2::Span::call_site();
    let mut iter = ts.into_iter().peekable();

    // Optional header: `@expect Name COUNT ;` or `@excluded Name "reason" ;`
    let mut expect: Option<(String, usize)> = None;
    if matches!(iter.peek(), Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '@') {
        iter.next();
        let kind = match iter.next() {
            Some(proc_macro2::TokenTree::Ident(id)) => id.to_string(),
            other => return Err(syn::Error::new(sp,
                format!("expected `expect` or `excluded` after `@`, got {:?}", other))),
        };
        let name = match iter.next() {
            Some(proc_macro2::TokenTree::Ident(id)) => id.to_string(),
            other => return Err(syn::Error::new(sp,
                format!("expected a type name, got {:?}", other))),
        };
        let payload = iter.next();
        match iter.next() {
            Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == ';' => {}
            other => return Err(syn::Error::new(sp,
                format!("expected `;` after the @{} header, got {:?}", kind, other))),
        }
        match kind.as_str() {
            "excluded" => {
                let reason = match payload {
                    Some(proc_macro2::TokenTree::Literal(l)) =>
                        syn::parse_str::<syn::LitStr>(&l.to_string())
                            .map(|s| s.value())
                            .unwrap_or_else(|_| l.to_string()),
                    other => return Err(syn::Error::new(sp,
                        format!("expected a reason string, got {:?}", other))),
                };
                registry_store_tombstone(&name, &reason);
                registry_stash_export(ExportEntry {
                    name, tokens: String::new(), param_count: 0,
                    excluded: Some(reason),
                });
                return Ok(TokenStream2::new());
            }
            "expect" => {
                let count = match payload {
                    Some(proc_macro2::TokenTree::Literal(l)) =>
                        l.to_string().trim_end_matches("usize").parse::<usize>()
                            .map_err(|_| syn::Error::new(sp,
                                format!("bad param count literal `{}`", l)))?,
                    other => return Err(syn::Error::new(sp,
                        format!("expected a param count, got {:?}", other))),
                };
                expect = Some((name, count));
            }
            other => return Err(syn::Error::new(sp,
                format!("unknown @{} header", other))),
        }
    }

    let rest: TokenStream2 = iter.collect();
    let original_tokens = rest.to_string();
    let mut input: syn::DeriveInput = syn::parse2(rest)?;
    let name = input.ident.clone();

    if has_struct_attr_ident(&input.attrs, "root") {
        return Err(syn::Error::new_spanned(&input.ident,
            format!("`{}` is a #[arael(root)] and cannot be imported -- its \
                     generated solver is already ordinary pub API", name)));
    }
    if parse_fit_attr(&input.attrs)?.is_some() {
        return Err(syn::Error::new_spanned(&input.ident,
            format!("`{}` is a #[arael(fit(...))] struct and cannot be imported", name)));
    }

    // Diamond import: re-registration validates the layout is identical
    // (registry_store errors otherwise) but must not stash constraints or
    // emit the const again.
    let newly = registry_mark_imported(&name.to_string());

    if matches!(input.data, syn::Data::Enum(_)) {
        register_enum_layout(&name)?;
        return Ok(TokenStream2::new());
    }

    let param_count = register_model_layout(&input)?;
    if let Some((exp_name, exp_count)) = &expect {
        if *exp_name != name.to_string() || *exp_count != param_count as usize {
            return Err(syn::Error::new_spanned(&input.ident,
                format!("imported `{}` computes {} params but its defining crate \
                         recorded {} -- the two crates were built with \
                         incompatible arael-macros versions",
                        name, param_count, exp_count)));
        }
    }
    if !newly {
        return Ok(TokenStream2::new());
    }

    // Defense in depth: the bundle only carries structs that passed the
    // field check at export, but hand-written invocations reach here too.
    if let syn::Data::Struct(data) = &input.data
        && let syn::Fields::Named(named) = &data.fields {
            for f in &named.named {
                let skipped = matches!(parse_arael_attr(&f.attrs),
                    Ok(Some(AraelAttr::Skip)));
                if !skipped && !matches!(f.vis, syn::Visibility::Public(_)) {
                    return Err(syn::Error::new_spanned(&input.ident,
                        format!("cannot import `{}`: field `{}` is not pub -- \
                                 generated code in this crate reads it directly",
                                name, f.ident.as_ref().unwrap())));
                }
            }
        }

    // Rewrite block types on our copy: records CrossBlock coupling pairs
    // in this session and makes the stashed constraint fields identical
    // to what the defining crate's expansion stashed.
    if let syn::Data::Struct(ref mut data) = input.data
        && let syn::Fields::Named(ref mut named) = data.fields {
            for field in named.named.iter_mut() {
                rewrite_block_type(&mut field.ty);
            }
        }
    if let syn::Data::Struct(data) = &input.data
        && let syn::Fields::Named(named) = &data.fields {
            stash_constraints(&name, &input.attrs, &named.named);
        }

    // Transitive export: a model crate that imports this bundle and calls
    // export_models!() carries these structs along for ITS importers.
    registry_stash_export(ExportEntry {
        name: name.to_string(),
        tokens: original_tokens,
        param_count,
        excluded: None,
    });

    let const_name = syn::Ident::new(&format!("{}_PARAM_COUNT", name), name.span());
    let count_lit = param_count as usize;
    Ok(quote! {
        #[allow(non_upper_case_globals)]
        #[doc(hidden)]
        const #const_name: usize = #count_lit;
    })
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
    // Original tokens, before block rewriting and attr stripping: what the
    // export bundle carries for cross-crate re-registration.
    let original_tokens = quote!(#input).to_string();
    let name = &input.ident;

    // Enums are treated as NOP leaves: zero params, trivial Model + ModelSym
    // impls. Lets callers put #[arael::model] on data-less enums used as
    // metadata fields (e.g. style, direction, mode) so those fields don't
    // need #[arael(skip)] at every use site.
    if matches!(input.data, syn::Data::Enum(_)) {
        return emit_trivial_model_for_enum(input);
    }

    let param_count = register_model_layout(input)?;

    // Every `pub` model struct joins the crate's export bundle
    // (`arael::export_models!()` -> the crate's `arael_import!` macro).
    // Roots and fit structs are not importable: their generated solvers
    // are already ordinary pub API. A struct with a non-pub field rides
    // along as a tombstone -- generated code in the importing crate reads
    // fields directly, so it cannot be used there, and the tombstone lets
    // the importer's failure say why.
    if matches!(input.vis, syn::Visibility::Public(_))
        && !has_struct_attr_ident(&input.attrs, "root")
        && parse_fit_attr(&input.attrs)?.is_none()
    {
        let excluded = match &input.data {
            syn::Data::Struct(data) => match &data.fields {
                syn::Fields::Named(named) => named.named.iter().find_map(|f| {
                    let skipped = matches!(parse_arael_attr(&f.attrs),
                        Ok(Some(AraelAttr::Skip)));
                    if skipped || matches!(f.vis, syn::Visibility::Public(_)) {
                        None
                    } else {
                        Some(format!("field `{}` is not pub",
                            f.ident.as_ref().unwrap()))
                    }
                }),
                _ => None,
            },
            _ => None,
        };
        registry_stash_export(ExportEntry {
            name: name.to_string(),
            tokens: original_tokens,
            param_count,
            excluded,
        });
    }

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


/// Stash every `#[arael(constraint(...))]` on the struct for later
/// generation at a root. Capture plain (file, line) data from span while
/// we're still inside the originating invocation — Span itself doesn't
/// survive the bridge but primitives do. Shared by the in-crate
/// expansion and `__register_model!`; both call it with the
/// block-REWRITTEN fields, so stashed content is identical either way.
fn stash_constraints(
    name: &syn::Ident,
    attrs: &[syn::Attribute],
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) {
    let fields_ts = quote! { #fields };
    for attr in attrs {
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

/// The registration half of `#[arael::model]`: validate the struct,
/// compute its layout, and store it in the session registry. Shared by
/// the in-crate expansion and `__register_model!` (cross-crate import),
/// which registers without emitting. Returns the struct's param count.
fn register_model_layout(input: &syn::DeriveInput) -> syn::Result<u32> {
    let name = &input.ident;

    // Compute PARAM_COUNT from Param<T> fields
    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(named) => &named.named,
            _ => return Err(syn::Error::new_spanned(input, "arael::model requires named fields")),
        },
        _ => return Err(syn::Error::new_spanned(input, "arael::model requires a struct or enum")),
    };

    // A `#[arael::model]` struct may be generic over exactly one scalar
    // type parameter bounded by `Float`; the layout registry stores shapes
    // only, so one registration covers every instantiation. The root must
    // be concrete -- it is where all generated solver code becomes real,
    // and its f32/f64 marker drives that. `fit(...)` structs have their
    // own codegen path and stay concrete too.
    let is_component = has_struct_attr_ident(&input.attrs, "component");
    let mut scalar_generic: Option<String> = None;
    if !input.generics.params.is_empty() {
        if has_struct_attr_ident(&input.attrs, "root") {
            return Err(syn::Error::new_spanned(&input.generics,
                format!("`{}`: a #[arael(root)] struct must be concrete -- \
                         instantiate generic entities in a concrete root \
                         (e.g. `poses: refs::Vec<Pose<f32>>`)", name)));
        }
        if has_struct_attr_ident(&input.attrs, "fit") {
            return Err(syn::Error::new_spanned(&input.generics,
                format!("`{}`: a #[arael(fit(...))] struct must be concrete", name)));
        }
        if input.generics.lifetimes().next().is_some()
            || input.generics.const_params().next().is_some()
            || input.generics.type_params().count() != 1 {
            return Err(syn::Error::new_spanned(&input.generics,
                format!("generic model `{}`: exactly one type parameter is \
                         supported, bounded by `Float` (e.g. `struct {}<T: Float>`)",
                        name, name)));
        }
        let tp = input.generics.type_params().next().unwrap();
        let has_float_bound = tp.bounds.iter().any(|b| matches!(b,
            syn::TypeParamBound::Trait(t)
                if t.path.segments.last().map(|s| s.ident == "Float").unwrap_or(false)));
        if !has_float_bound {
            return Err(syn::Error::new_spanned(tp,
                format!("generic model `{}`: the type parameter `{}` must carry \
                         an inline `Float` bound (`{}: arael::utils::Float`)",
                        name, tp.ident, tp.ident)));
        }
        scalar_generic = Some(tp.ident.to_string());
    }

    let mut param_count: u32 = 0;
    let mut sym_fields: Vec<(String, SymFieldType)> = Vec::new();
    let mut collection_fields_reg: Vec<String> = Vec::new();
    let mut param_field_names_for_reg: Vec<String> = Vec::new();
    let mut ref_paths_for_reg: Vec<(String, String)> = Vec::new();
    let mut euler_angle_fields_reg: Vec<String> = Vec::new();
    let mut universal_euler_angle_fields_reg: Vec<String> = Vec::new();
    let mut universal_rotvec_fields_reg: Vec<String> = Vec::new();
    let mut symbolic_fields_reg: Vec<(String, String)> = Vec::new();
    // (deriv field name, of = symbolic field, by = param field)
    let mut deriv_fields_reg: Vec<(String, String, String)> = Vec::new();
    // (field, wrapper, held) -- unrecognized generic wrappers naming a
    // not-yet-registered type; fires if the type registers later.
    let mut suspect_wrappers_reg: Vec<(String, String, String)> = Vec::new();
    let mut block_precision_reg: Option<(String, String)> = None;
    let mut inst_precisions_reg: Vec<(String, String, String)> = Vec::new();
    let mut triplet_block_fields_reg: Vec<String> = Vec::new();
    let mut spelled_types_reg: Vec<(String, String)> = Vec::new();
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
        spelled_types_reg.push((field_name.clone(), type_spelling(&field.ty)));
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
                AraelAttr::Symbolic(expr_tokens) => {
                    // The field stays an ordinary data field (classified
                    // below); only its body-read meaning changes.
                    symbolic_fields_reg.push((field_name.clone(), expr_tokens.to_string()));
                }
                AraelAttr::Deriv { of, by } => {
                    // A declared Jacobian cache: plain storage to the
                    // runtime (skip), meaningful only to the symbolic
                    // precompute and the constraint substitutions.
                    deriv_fields_reg.push((field_name.clone(), of, by));
                    sym_fields.push((field_name, SymFieldType::Skip));
                    continue;
                }
                _ => {}
            }
        }
        // Containers are recognized by literal type name: an aliased
        // container (`use refs::Vec as RVec`) reads as an opaque field and
        // its contents silently stop being containment. Reject any
        // unrecognized wrapper holding a registered model type;
        // `#[arael(skip)]` documents a deliberate out-of-model holding.
        {
            let mut suspects = Vec::new();
            collect_wrapper_suspects(&field.ty, scalar_generic.as_deref(), &mut suspects);
            for (wrapper, held) in suspects {
                if registry_lookup(&held).map_or(false, |l| !l.fields.is_empty()) {
                    return Err(syn::Error::new_spanned(field, format!(
                        "field `{}`: unrecognized container `{}` holds model type `{}` -- \
                         containers are recognized by literal name (Vec, Deque, Arena, \
                         Option, Ref), aliases are invisible to the macro; spell the \
                         container literally, or mark the field `#[arael(skip)]` if it \
                         is deliberately outside the model",
                        field_name, wrapper, held)));
                }
                suspect_wrappers_reg.push((field_name.clone(), wrapper, held));
            }
        }
        // All of a struct's blocks solve at one precision; record it and
        // reject a mix outright.
        if let Some(p) = block_field_precision(&field.ty, scalar_generic.as_deref()) {
            match &block_precision_reg {
                Some((first_field, first_p)) if *first_p != p => {
                    return Err(syn::Error::new_spanned(field, format!(
                        "`{}` mixes block precisions: `{}` is {}, `{}` is {} -- \
                         all block fields of a struct solve at one precision",
                        name, first_field, first_p, field_name, p)));
                }
                Some(_) => {}
                None => block_precision_reg = Some((field_name.clone(), p)),
            }
        }
        if let Some((elem, fl)) = inst_precision_of(&field.ty) {
            inst_precisions_reg.push((field_name.clone(), elem, fl));
        }
        {
            let bare = if let Some((inner, _)) = extract_wrapper_inner(&field.ty, "Option") {
                inner
            } else { &field.ty };
            if let syn::Type::Path(tp) = bare
                && let Some(seg) = tp.path.segments.last()
                && seg.ident == "TripletBlock" {
                    triplet_block_fields_reg.push(field_name.clone());
                }
        }
        // Detect euler angle param types by type name (in addition to attribute)
        if let Some(ea_kind) = is_euler_angle_param_type(&field.ty) {
            match ea_kind {
                "simple" => euler_angle_fields_reg.push(field_name.clone()),
                "universal" => universal_euler_angle_fields_reg.push(field_name.clone()),
                "universal_rotvec" => universal_rotvec_fields_reg.push(field_name.clone()),
                _ => {}
            }
        }
        // A field whose type was TOMBSTONED by an imported bundle (a pub
        // struct with non-pub fields) can never work here -- generated
        // code reads its fields directly. Fail with the recorded reason
        // instead of silently misclassifying it as plain data.
        {
            let mut inner_names: Vec<String> = Vec::new();
            if let syn::Type::Path(tp) = &field.ty
                && let Some(seg) = tp.path.segments.last() {
                    inner_names.push(seg.ident.to_string());
                }
            for wrapper in ["Ref", "Vec", "Deque", "Arena", "Option"] {
                if let Some((_, id)) = extract_wrapper_inner(&field.ty, wrapper) {
                    inner_names.push(id.to_string());
                }
            }
            for n in inner_names {
                if registry_lookup(&n).is_none()
                    && let Some(reason) = registry_excluded_reason(&n) {
                        return Err(syn::Error::new_spanned(field,
                            format!("`{}` was not exported by its defining \
                                     crate: {}", n, reason)));
                    }
            }
        }
        // Component-typed fields fold their params into this struct's count
        // (the component is registered before its owner, top-down rule).
        // Generic args are ignored: the registry is keyed by the bare type
        // name and a component's layout is the same at every precision, so
        // `UnitVec`, `UnitVec<f64>` and `UnitVec<f32>` all resolve.
        if !is_param_type(&field.ty)
            && let syn::Type::Path(tp) = &field.ty
            && let Some(seg) = tp.path.segments.last()
            && registry_lookup(&seg.ident.to_string()).map(|l| l.component).unwrap_or(false)
        {
            param_count += registry_param_total(&seg.ident.to_string());
        }
        if is_param_type(&field.ty) {
            param_count += param_type_size(&field.ty, scalar_generic.as_deref());
            param_field_names_for_reg.push(field_name.clone());
            let sft = match param_type_size(&field.ty, scalar_generic.as_deref()) {
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
                collection_fields_reg.push(field_name.clone());
                sym_fields.push((field_name, SymFieldType::Struct(inner_ident.to_string())));
            } else {
                sym_fields.push((field_name, SymFieldType::Skip));
            }
        } else {
            let sft = classify_field_sym_type(&field.ty, scalar_generic.as_deref());
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
    // `#[arael(component)]`: a compound parameter. Its Params fold into the
    // OWNING struct's span, so it has no SelfBlock of its own, carries no
    // constraints, and holds no collections. Runtime lifecycle comes from
    // the `arael::model::Component` trait the user implements.
    if is_component {
        if self_block_field_reg.is_some() {
            return Err(syn::Error::new_spanned(name,
                format!("`{}` is a component -- its params fold into the owning \
                         struct's block, so it must not declare a SelfBlock<Self>", name)));
        }
        if !collection_fields_reg.is_empty() {
            return Err(syn::Error::new_spanned(name,
                format!("`{}` is a component and may not hold collections", name)));
        }
        for attr in &input.attrs {
            if attr.path().is_ident("arael")
                && let Ok(content) = attr.parse_args::<TokenStream2>()
                && content.clone().into_iter().next().map(|t| t.to_string() == "constraint").unwrap_or(false) {
                return Err(syn::Error::new_spanned(name,
                    format!("`{}` is a component and may not carry constraints -- \
                             residuals belong to the entities that own it", name)));
            }
        }
    }
    if param_count > 0 && self_block_field_reg.is_none() && !has_fit && !has_skip_self_block
        && !is_component {
        return Err(syn::Error::new_spanned(name,
            format!("`{}` has {} parameter{} but no `SelfBlock<Self>` field — \
                     add e.g. `hb: arael::model::SelfBlock<Self>` so its grad \
                     and Hessian diagonal have a home, or annotate the struct \
                     with `#[arael(skip_self_block)]` if its params are \
                     written exclusively by a parent's ExtendedModel",
                    name, param_count, if param_count == 1 { "" } else { "s" })));
    }

    // Register injected fields in sym layout
    for ea_field in &euler_angle_fields_reg {
        sym_fields.push((format!("{}_rotation_matrix", ea_field), SymFieldType::Mat3));
    }
    for ea_field in &universal_euler_angle_fields_reg {
        sym_fields.push((format!("{}_ref_rotation", ea_field), SymFieldType::Skip));
        sym_fields.push((format!("{}_delta", ea_field), SymFieldType::Skip));
        sym_fields.push((format!("{}_rotation_matrix", ea_field), SymFieldType::Mat3));
    }

    registry_store(&name.to_string(), SymLayout {
        fields: sym_fields,
        collection_fields: collection_fields_reg,
        param_fields: param_field_names_for_reg,
        ref_paths: ref_paths_for_reg,
        euler_angle_fields: euler_angle_fields_reg.clone(),
        universal_euler_angle_fields: universal_euler_angle_fields_reg.clone(),
        universal_rotvec_fields: universal_rotvec_fields_reg.clone(),
        symbolic_fields: symbolic_fields_reg,
        deriv_fields: deriv_fields_reg,
        atom_cached_fields: Vec::new(),
        component: is_component,
        constraint_index_field: constraint_index_field_reg,
        self_block_field: self_block_field_reg,
        suspect_wrappers: suspect_wrappers_reg,
        block_precision: block_precision_reg,
        inst_precisions: inst_precisions_reg,
        triplet_block_fields: triplet_block_fields_reg,
        is_root: has_struct_attr_ident(&input.attrs, "root"),
        spelled_types: spelled_types_reg,
    }).map_err(|msg| syn::Error::new_spanned(name, msg))?;

    Ok(param_count)
}

/// Emit a trivial Model + ModelSym impl for a data-less enum. All Model
/// methods are no-ops, PARAM_COUNT is 0, and the ModelSym companion is an
/// empty struct. The enum itself is emitted unchanged (attributes stripped).
/// Register a zero-field sym layout so constraint macros that look up
/// this type find an entry (important if the enum is ever used as a
/// nested struct field type in constraint bodies).
fn register_enum_layout(name: &syn::Ident) -> syn::Result<()> {
    registry_store(&name.to_string(), SymLayout::default())
        .map_err(|msg| syn::Error::new_spanned(name, msg))
}

fn emit_trivial_model_for_enum(input: &mut syn::DeriveInput) -> syn::Result<TokenStream2> {
    let original_tokens = quote!(#input).to_string();
    let name = &input.ident;
    let sym_name = syn::Ident::new(&format!("{}Sym", name), name.span());
    let const_name = syn::Ident::new(&format!("{}_PARAM_COUNT", name), name.span());
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    register_enum_layout(name)?;

    // Pub enums join the export bundle like structs: an imported entity
    // may carry a field of this enum type.
    if matches!(input.vis, syn::Visibility::Public(_)) {
        registry_stash_export(ExportEntry {
            name: name.to_string(),
            tokens: original_tokens,
            param_count: 0,
            excluded: None,
        });
    }

    // Strip any #[arael(...)] attributes from the emitted item.
    input.attrs.retain(|attr| !attr.path().is_ident("arael"));

    Ok(quote! {
        #input
        #[allow(non_upper_case_globals)]
        const #const_name: usize = 0;

        #[derive(Clone)]
        pub struct #sym_name;

        impl arael::model::ModelSym for #name {
            type Sym = #sym_name;
            fn sym(_base: &str) -> #sym_name { #sym_name }
        }

        impl #impl_generics arael::model::Model for #name #ty_generics #where_clause {}
    })
}

/// Classify a non-Param field's sym type from its type path. The layout
/// stores shapes, not precisions, so the generic spellings (`vect2<T>`,
/// bare `T` when `T` is the struct's scalar type parameter, passed as
/// `scalar_generic`) classify the same as the suffixed aliases.
fn classify_field_sym_type(ty: &syn::Type, scalar_generic: Option<&str>) -> SymFieldType {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            if scalar_generic == Some(name.as_str()) {
                return SymFieldType::Scalar;
            }
            return match name.as_str() {
                "f32" | "f64" | "bool" | "u32" | "i32" | "usize" => SymFieldType::Scalar,
                "vect2f" | "vect2d" | "vect2" => SymFieldType::Vec2,
                "vect3f" | "vect3d" | "vect3" => SymFieldType::Vec3,
                "matrix3f" | "matrix3d" | "matrix3" => SymFieldType::Mat3,
                "matrix2f" | "matrix2d" | "matrix2" => SymFieldType::Mat2,
                "quaternf" | "quaternd" | "quatern" => SymFieldType::Quat,
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
fn param_type_size(ty: &syn::Type, scalar_generic: Option<&str>) -> u32 {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            if name == "Param"
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                    && let Some(syn::GenericArgument::Type(inner)) = args.args.first() {
                        return inner_type_size(inner, scalar_generic);
                    }
            if name == "SimpleEulerAngleParam" || name == "EulerAngleParam"
                || name == "QuaternionParam" {
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

/// Return the ParamType::SIZE for known types. `scalar_generic` names the
/// owning struct's scalar type parameter, so `Param<T>` sizes like
/// `Param<f64>` and `Param<vect2<T>>` like `Param<vect2d>`.
fn inner_type_size(ty: &syn::Type, scalar_generic: Option<&str>) -> u32 {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            if scalar_generic == Some(name.as_str()) {
                return 1;
            }
            return match name.as_str() {
                "f32" | "f64" => 1,
                "vect2f" | "vect2d" | "vect2" => 2,
                "vect3f" | "vect3d" | "vect3" => 3,
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
            if (seg.ident == "SelfBlock" || seg.ident == "BoxedSelfBlock")
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments
                && let Some(syn::GenericArgument::Type(syn::Type::Path(inner_tp))) = args.args.first()
                && let Some(inner_seg) = inner_tp.path.segments.last()
            {
                return inner_seg.ident == self_name;
            }
        }
    false
}

/// Rewrite SelfBlock<A> to SelfBlock<A, {A_PARAM_COUNT}, {N(N+1)/2}> and
/// CrossBlock<A, B> to CrossBlock<A, B, {A_PARAM_COUNT}, {B_PARAM_COUNT}, {NA*NB}>
/// (both with an optional trailing f32 float arg). The BoxedSelfBlock /
/// BoxedCrossBlock variants take the identical const dims -- they wrap the
/// inline block behind a Box -- so they are rewritten the same way, preserving
/// the original type name.
fn rewrite_block_type(ty: &mut syn::Type) {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last_mut() {
            let type_name = seg.ident.to_string();
            let block_ident = seg.ident.clone();
            if type_name == "SelfBlock" || type_name == "BoxedSelfBlock" {
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
                                    // <A, f32> -> <A, {N}, {N(N+1)/2}, f32>
                                    let float_ty = type_args[1];
                                    let new_ty: syn::Type = syn::parse_quote! {
                                        #block_ident<#a_path, { #const_name }, { #const_name * (#const_name + 1) / 2 }, #float_ty>
                                    };
                                    *ty = new_ty;
                                } else {
                                    // <A> -> <A, {N}, {N(N+1)/2}>
                                    let new_ty: syn::Type = syn::parse_quote! {
                                        #block_ident<#a_path, { #const_name }, { #const_name * (#const_name + 1) / 2 }>
                                    };
                                    *ty = new_ty;
                                }
                            }
                    }
                }
            } else if (type_name == "CrossBlock" || type_name == "BoxedCrossBlock")
                && let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    let type_args: Vec<&syn::Type> = args.args.iter()
                        .filter_map(|a| if let syn::GenericArgument::Type(t) = a { Some(t) } else { None })
                        .collect();
                    if (type_args.len() == 2 || type_args.len() == 3)
                        && let (syn::Type::Path(a_path), syn::Type::Path(b_path)) =
                            (type_args[0], type_args[1])
                            && let (Some(a_seg), Some(b_seg)) = (a_path.path.segments.last(), b_path.path.segments.last()) {
                                registry_record_cross(
                                    &a_seg.ident.to_string(),
                                    &b_seg.ident.to_string(),
                                );
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
                                    // <A, B, f32> -> <A, B, {NA}, {NB}, {NA*NB}, f32>
                                    let float_ty = type_args[2];
                                    let new_ty: syn::Type = syn::parse_quote! {
                                        #block_ident<#a_ty, #b_ty, { #a_const }, { #b_const }, { #a_const * #b_const }, #float_ty>
                                    };
                                    *ty = new_ty;
                                } else {
                                    // <A, B> -> <A, B, {NA}, {NB}, {NA*NB}>
                                    let new_ty: syn::Type = syn::parse_quote! {
                                        #block_ident<#a_ty, #b_ty, { #a_const }, { #b_const }, { #a_const * #b_const }>
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

    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(named) => &named.named,
            _ => return Err(syn::Error::new_spanned(input, "Model derive requires named fields")),
        },
        _ => return Err(syn::Error::new_spanned(input, "Model derive requires a struct")),
    };

    // A generic struct's generated impls bound every walked field type on
    // Model. Block Model impls exist per precision (that is the routing),
    // so `SelfBlock<A<T>, ..., T>: Model` holds only at each concrete
    // instantiation -- and the requirement must PROPAGATE: a collection of
    // nested constraint structs (`Vec<Frine<T>>`) carries its element's
    // block bounds in its own signature, so an owner naming the collection
    // resolves them at its instantiation too.
    let mut generics = input.generics.clone();
    if let Some(tp) = input.generics.type_params().next() {
        let scalar = tp.ident.to_string();
        let mentions_scalar = |ty: &syn::Type| {
            quote!(#ty).into_iter().any(|t| token_stream_has_ident(t, &scalar))
        };
        for field in fields {
            match parse_arael_attr(&field.attrs)? {
                Some(AraelAttr::Skip) | Some(AraelAttr::ConstraintIndex)
                | Some(AraelAttr::Deriv { .. }) | Some(AraelAttr::Compute(_)) => continue,
                _ => {}
            }
            // Bare scalar fields are excluded from the walks entirely.
            if let syn::Type::Path(tp2) = &field.ty
                && tp2.qself.is_none()
                && tp2.path.segments.len() == 1
                && tp2.path.segments[0].ident == scalar.as_str()
                && matches!(tp2.path.segments[0].arguments, syn::PathArguments::None)
                && !is_param_type(&field.ty)
            {
                continue;
            }
            let ty = &field.ty;
            if mentions_scalar(ty) {
                generics.make_where_clause().predicates.push(
                    syn::parse_quote! { #ty: arael::model::Model });
            }
        }
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    // Component role (registered by model_attribute before this runs):
    // wraps this struct's Model impl in the Component lifecycle calls.
    let is_component_struct = registry_lookup(&name.to_string())
        .map(|l| l.component).unwrap_or(false);
    // The per-Param slice writeback the component's advance needs after
    // Component::update reset the values.
    let mut comp_writeback32: Vec<TokenStream2> = Vec::new();
    let mut comp_writeback64: Vec<TokenStream2> = Vec::new();

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
    let mut release_blocks_stmts: Vec<TokenStream2> = Vec::new();
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
    let mut advance32_stmts: Vec<TokenStream2> = Vec::new();
    let mut advance64_stmts: Vec<TokenStream2> = Vec::new();
    let mut param_count_terms: Vec<TokenStream2> = Vec::new();
    let mut serialize_size_stmts: Vec<TokenStream2> = Vec::new();
    let mut param_symbols_stmts: Vec<TokenStream2> = Vec::new();
    // Param-bearing fields in serialize order, with their size expression --
    // consumed by the marginalize root keyword to compute param ranges
    // with exactly the serialize walk's field selection.
    // (field, size expr, element type name). The type name is what the
    // Schur detector's coupling graph is built over.
    let mut size_walk: Vec<(syn::Ident, TokenStream2, Option<String>)> = Vec::new();
    // Per-struct recursion for Model::collect_param_blocks (entity spans
    // read from SelfBlock indices).
    let mut collect_param_blocks_stmts: Vec<TokenStream2> = Vec::new();
    // Per-struct recursion for the structure-only Hessian walks (cells
    // and scatter positions), split by block precision to mirror the
    // accumulate stmt lists exactly -- emission order is the invariant.
    let mut has_triplet_block = false;
    let mut collect_cells32_stmts: Vec<TokenStream2> = Vec::new();
    let mut collect_cells64_stmts: Vec<TokenStream2> = Vec::new();
    let mut positions32_stmts: Vec<TokenStream2> = Vec::new();
    let mut positions64_stmts: Vec<TokenStream2> = Vec::new();

    // The struct's scalar type parameter, when generic: a bare `T` data
    // field is excluded from the Model walks below -- for concrete scalars
    // every Model method is a no-op, and emitting the calls would demand
    // `T: Model` from a plain data field.
    let scalar_generic_impl: Option<String> =
        input.generics.type_params().next().map(|tp| tp.ident.to_string());

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let attr = parse_arael_attr(&field.attrs)?;

        if !is_param_type(&field.ty)
            && let syn::Type::Path(tp) = &field.ty
            && tp.qself.is_none()
            && tp.path.segments.len() == 1
            && let Some(seg) = tp.path.segments.first()
            && matches!(seg.arguments, syn::PathArguments::None)
            && scalar_generic_impl.as_deref() == Some(seg.ident.to_string().as_str())
        {
            continue;
        }

        match attr {
            // Cross is a constraint-struct-only attribute for CrossBlock
            // fields; treated like the default path here (no param, no
            // compute, just a block field).
            // Deriv caches are plain storage to the runtime walks, like Skip.
            Some(AraelAttr::Skip) | Some(AraelAttr::ConstraintIndex)
            | Some(AraelAttr::Deriv { .. }) => continue,
            Some(AraelAttr::Compute(expr_tokens)) => {
                let substituted = substitute_param_idents(expr_tokens, &param_field_names);
                compute_stmts.push(quote! { self.#ident = #substituted; });
            }
            // Symbolic fields stay ordinary data fields at runtime; only
            // their constraint-body reads differ.
            Some(AraelAttr::RefResolve(_)) | Some(AraelAttr::Cross(_))
            | Some(AraelAttr::Symbolic(_)) | None => {
                // Block fields walk through the same uniform Model
                // recursion as every other field: each block's Model impl
                // participates in exactly its own precision's walks (the
                // other family is the trait's no-op default), so the
                // precision sort happens at monomorphization -- which is
                // what lets a generic struct's `SelfBlock<A, T>` sort
                // itself per instantiation.
                if let syn::Type::Path(tp) = &field.ty
                    && let Some(seg) = tp.path.segments.last()
                        && seg.ident == "TripletBlock" {
                            has_triplet_block = true;
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
                    if is_component_struct && is_euler_angle_param_type(ty).is_none() {
                        comp_writeback32.push(quote! {
                            if self.#ident.index() != u32::MAX {
                                let __i = self.#ident.index() as usize;
                                let __n = <#ty as arael::model::Model>::PARAM_COUNT as usize;
                                arael::model::ParamType::write_to32(&self.#ident.value, &mut params[__i..__i + __n]);
                            }
                        });
                        comp_writeback64.push(quote! {
                            if self.#ident.index() != u32::MAX {
                                let __i = self.#ident.index() as usize;
                                let __n = <#ty as arael::model::Model>::PARAM_COUNT as usize;
                                arael::model::ParamType::write_to64(&self.#ident.value, &mut params[__i..__i + __n]);
                            }
                        });
                    }
                } else if let syn::Type::Path(tp) = ty
                    && let Some(seg) = tp.path.segments.last()
                    && registry_lookup(&seg.ident.to_string()).map(|l| l.component).unwrap_or(false)
                {
                    // A component-typed field: its params fold into this
                    // struct's span (serialize recursion below carries them;
                    // the count and symbol walk must too).
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
                advance32_stmts.push(quote! {
                    arael::model::Model::advance_params32(&mut self.#ident, params);
                });
                advance64_stmts.push(quote! {
                    arael::model::Model::advance_params64(&mut self.#ident, params);
                });
                serialize_size_stmts.push(quote! {
                    arael::model::Model::serialize_size(&self.#ident)
                });
                let elem_name = extract_wrapper_inner(&field.ty, "Vec")
                    .or_else(|| extract_wrapper_inner(&field.ty, "Deque"))
                    .or_else(|| extract_wrapper_inner(&field.ty, "Arena"))
                    .map(|(_, id)| id.to_string())
                    .or_else(|| if let syn::Type::Path(tp) = &field.ty {
                        tp.path.segments.last().map(|s| s.ident.to_string())
                    } else {
                        None
                    });
                size_walk.push((ident.clone(), quote! {
                    arael::model::Model::serialize_size(&self.#ident) as usize
                }, elem_name));
                // Also recurse into sub-models for zero/accumulate
                zero_blocks_stmts.push(quote! {
                    arael::model::Model::zero_blocks(&mut self.#ident);
                });
                collect_param_blocks_stmts.push(quote! {
                    arael::model::Model::collect_param_blocks(&self.#ident, out);
                });
                collect_cells32_stmts.push(quote! {
                    arael::model::Model::collect_hessian_cells32(&self.#ident, out);
                });
                collect_cells64_stmts.push(quote! {
                    arael::model::Model::collect_hessian_cells64(&self.#ident, out);
                });
                positions32_stmts.push(quote! {
                    arael::model::Model::bind_hessian_positions32(&mut self.#ident, binder, out);
                });
                positions64_stmts.push(quote! {
                    arael::model::Model::bind_hessian_positions64(&mut self.#ident, binder, out);
                });
                release_blocks_stmts.push(quote! {
                    arael::model::Model::release_blocks(&mut self.#ident);
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

    // Component lifecycle wrapping: seed the chart before the params are
    // read, write the user-facing value back after deserialize, and at the
    // advance point pull the accepted step in, re-center, and push the reset
    // values back out to the slice.
    let comp_start = if is_component_struct {
        quote! { arael::model::Component::start(self); }
    } else { quote! {} };
    let comp_finish = if is_component_struct {
        quote! { arael::model::Component::finish(self); }
    } else { quote! {} };
    let comp_advance32 = if is_component_struct {
        quote! {
            arael::model::Model::deserialize_params32(self, params);
            arael::model::Component::update(self);
            #(#comp_writeback32)*
        }
    } else { quote! {} };
    let comp_advance64 = if is_component_struct {
        quote! {
            arael::model::Model::deserialize_params64(self, params);
            arael::model::Component::update(self);
            #(#comp_writeback64)*
        }
    } else { quote! {} };

    // `symbolic =` fields (and their declared `deriv =` caches) get a
    // generated numeric refresh, called wherever computed fields refresh.
    // The layout was registered by the attribute pass before this runs; a
    // bare-derive struct without one simply has no symbolic fields.
    let precompute_impl = match crate::registry_lookup(&name.to_string()) {
        Some(l) if !l.symbolic_fields.is_empty() =>
            crate::constraint::generate_symbolic_precompute(
                &name.to_string(), &l.fields, &l.param_fields,
                &l.symbolic_fields, &l.deriv_fields,
                input.generics.type_params().next()
                    .map(|tp| tp.ident.to_string()).as_deref())?,
        _ => TokenStream2::new(),
    };
    let precompute_call: TokenStream2 = if precompute_impl.is_empty() {
        quote! {}
    } else {
        quote! { self.__precompute_symbolic(); }
    };
    // The deserialize paths run after a solve, when the params' WORKING
    // copies still hold the last trial state: sync them (update_self does,
    // and then precomputes) or the refreshed fields would be evaluated at
    // a stale delta, disagreeing with what Component::finish wrote.
    let precompute_deser: TokenStream2 = if precompute_impl.is_empty() {
        quote! {}
    } else {
        quote! { arael::model::Model::update_self(self); }
    };
    let precompute_block: TokenStream2 = if precompute_impl.is_empty() {
        quote! {}
    } else {
        quote! {
            impl #impl_generics #name #ty_generics #where_clause {
                #precompute_impl
            }
        }
    };

    let model_impl = quote! {
        impl #impl_generics arael::model::Model for #name #ty_generics #where_clause {
            fn serialize_params32(&mut self, data: &mut std::vec::Vec<f32>) {
                #comp_start
                #(#serialize_stmts)*
            }
            fn deserialize_params32(&mut self, data: &[f32]) {
                #(#deserialize_stmts)*
                #comp_finish
                #precompute_deser
            }
            fn update32(&mut self, data: &[f32]) {
                #(#update_phase1)*
                #(#compute_stmts)*
                #(#euler_compute_stmts)*
                #precompute_call
            }
            fn update_self(&mut self) {
                #(#update_self_phase1)*
                #(#compute_stmts)*
                #(#euler_compute_stmts)*
                #precompute_call
            }
            fn serialize_params64(&mut self, data: &mut std::vec::Vec<f64>) {
                #comp_start
                #(#serialize64_stmts)*
            }
            fn deserialize_params64(&mut self, data: &[f64]) {
                #(#deserialize64_stmts)*
                #comp_finish
                #precompute_deser
            }
            fn update64(&mut self, data: &[f64]) {
                #(#update64_phase1)*
                #(#compute_stmts)*
                #(#euler_compute_stmts)*
                #precompute_call
            }
            fn advance_params32(&mut self, params: &mut [f32]) {
                #(#advance32_stmts)*
                #comp_advance32
            }
            fn advance_params64(&mut self, params: &mut [f64]) {
                #(#advance64_stmts)*
                #comp_advance64
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
            fn collect_param_blocks(&self, out: &mut std::vec::Vec<(u32, u32)>) {
                let _ = &out;
                #(#collect_param_blocks_stmts)*
            }
            fn collect_hessian_cells64(&self, out: &mut std::vec::Vec<(u32, u32)>) {
                let _ = &out;
                #(#collect_cells64_stmts)*
            }
            fn collect_hessian_cells32(&self, out: &mut std::vec::Vec<(u32, u32)>) {
                let _ = &out;
                #(#collect_cells32_stmts)*
            }
            fn bind_hessian_positions64(&mut self, binder: &mut arael::model::HessianBinder, out: &mut std::vec::Vec<arael::ValueIndex>) {
                let _ = (&binder, &out);
                #(#positions64_stmts)*
            }
            fn bind_hessian_positions32(&mut self, binder: &mut arael::model::HessianBinder, out: &mut std::vec::Vec<arael::ValueIndex>) {
                let _ = (&binder, &out);
                #(#positions32_stmts)*
            }
            fn release_blocks(&mut self) {
                #(#release_blocks_stmts)*
            }
            fn accumulate_hessian32(&self, hessian: &mut [f32]) {
                #(#accumulate_hessian32_stmts)*
            }
            fn accumulate_hessian64(&self, hessian: &mut [f64]) {
                #(#accumulate_hessian64_stmts)*
            }
            fn accumulate_hessian_band32(&self, band: &mut [f32], kd: usize) -> Result<(), arael::simple_lm::BandOverflow> {
                #(#accumulate_hessian_band32_stmts)*
                Ok(())
            }
            fn accumulate_hessian_band64(&self, band: &mut [f64], kd: usize) -> Result<(), arael::simple_lm::BandOverflow> {
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
            fn accumulate_hessian_sparse_indexed32(&self, vals: &mut [f32], positions: &[arael::ValueIndex], cursor: &mut usize) {
                #(#accumulate_hessian_sparse_indexed32_stmts)*
            }
            fn accumulate_hessian_sparse_indexed64(&self, vals: &mut [f64], positions: &[arael::ValueIndex], cursor: &mut usize) {
                #(#accumulate_hessian_sparse_indexed64_stmts)*
            }
        }
    };

    // Generate *Sym companion struct and ModelSym impl
    let sym_impl = generate_sym_impl(name, &input.generics, fields)?;

    // Check for #[arael(fit(...))] on the struct
    let fit_impl = match parse_fit_attr(&input.attrs)? {
        Some(fit) => generate_fit_impl(name, fields, &fit)?,
        None => quote! {},
    };

    // Check for #[arael(constraint(...))] — stash ALL constraints for later generation.
    stash_constraints(name, &input.attrs, fields);

    // Check for #[arael(root)] or #[arael(root, f32)] — trigger generation of all stashed constraints
    let root_info = input.attrs.iter().find_map(|attr| {
        if !attr.path().is_ident("arael") { return None; }
        let content: TokenStream2 = attr.parse_args().ok()?;
        let tvec: Vec<proc_macro2::TokenTree> = content.into_iter().collect();
        if let Some(proc_macro2::TokenTree::Ident(id)) = tvec.first() {
            if *id != "root" { return None; }
            // Parse optional keywords after comma: f32/f64, extended,
            // jacobian, fast_atan, marginalize(fields). Unknown
            // keywords are hard errors: a silently ignored typo
            // (`jacobain`) or a combined `fit(...)` would otherwise no-op.
            let mut precision = "f64".to_string();
            let mut custom = false;
            let mut jacobian = false;
            let mut fast_atan = false;
            let mut marginalize: Vec<syn::Ident> = Vec::new();
            let mut pos = 1;
            while pos < tvec.len() {
                match &tvec[pos] {
                    proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => {}
                    other => return Some(Err(syn::Error::new(other.span(),
                        "expected `,` between root keywords"))),
                }
                pos += 1;
                match tvec.get(pos) {
                    Some(proc_macro2::TokenTree::Ident(kw)) => {
                        let kw_str = kw.to_string();
                        if kw_str == "f32" || kw_str == "f64" {
                            precision = kw_str.clone();
                        } else if kw_str == "extended" {
                            custom = true;
                        } else if kw_str == "jacobian" {
                            jacobian = true;
                        } else if kw_str == "fast_atan" {
                            fast_atan = true;
                        } else if kw_str == "marginalize" {
                            // Takes a parenthesized field list:
                            // marginalize(landmarks) or (a, b).
                            pos += 1;
                            let Some(proc_macro2::TokenTree::Group(g)) = tvec.get(pos) else {
                                return Some(Err(syn::Error::new(kw.span(),
                                    "marginalize requires a field list: marginalize(field, ...)")));
                            };
                            for t in g.stream() {
                                match t {
                                    proc_macro2::TokenTree::Ident(f) => marginalize.push(f),
                                    proc_macro2::TokenTree::Punct(p) if p.as_char() == ',' => {}
                                    other => return Some(Err(syn::Error::new(other.span(),
                                        "marginalize expects a comma-separated list of field names"))),
                                }
                            }
                            if marginalize.is_empty() {
                                return Some(Err(syn::Error::new(g.span(),
                                    "marginalize requires at least one field name")));
                            }
                            pos += 1;
                            continue;
                        } else if kw_str == "fit" || kw_str == "fit64" {
                            return Some(Err(syn::Error::new(kw.span(),
                                "fit(...) cannot be combined with root; use a separate #[arael(fit(...))] attribute")));
                        } else {
                            return Some(Err(syn::Error::new(kw.span(),
                                format!("unknown root keyword `{}`, expected `f32`, `f64`, `extended`, `jacobian`, `fast_atan`, or `marginalize(...)`", kw_str))));
                        }
                        pos += 1;
                        // Skip a group following a keyword (e.g. a stray
                        // parenthesized argument) -- nothing else takes one.
                        if let Some(proc_macro2::TokenTree::Group(g)) = tvec.get(pos) {
                            return Some(Err(syn::Error::new(g.span(),
                                format!("root keyword `{}` takes no arguments", kw_str))));
                        }
                    }
                    Some(other) => return Some(Err(syn::Error::new(other.span(),
                        "expected a keyword after `,` in root attribute"))),
                    None => {} // trailing comma
                }
            }
            return Some(Ok((precision, custom, jacobian, fast_atan, marginalize)));
        }
        None
    });
    let root_info = match root_info {
        Some(r) => Some(r?),
        None => None,
    };
    let root_precision = root_info.as_ref().map(|(p, _, _, _, _)| p.clone());
    let root_custom = root_info.as_ref().map(|(_, c, _, _, _)| *c).unwrap_or(false);
    let root_jacobian = root_info.as_ref().map(|(_, _, j, _, _)| *j).unwrap_or(false);
    let root_fast_atan = root_info.as_ref().map(|(_, _, _, f, _)| *f).unwrap_or(false);
    let root_eliminate = root_info.as_ref().map(|(_, _, _, _, e)| e.clone()).unwrap_or_default();

    // Schur auto-detection: which parameter blocks may be marginalized.
    //
    // The coupling graph has one node per entity type appearing in a
    // parameter-bearing root field, and an edge for every CrossBlock<A, B>
    // anywhere in the model (a SELF-LOOP when A == B, e.g. odometry's
    // CrossBlock<Pose, Pose>). A set of types is eliminable exactly when it
    // is an INDEPENDENT SET: no edge among its members. That is sound
    // because the macro emits a block for every J^T J pair, so no
    // CrossBlock<A, B> means no Hessian tile can ever join an A to a B --
    // which is precisely the block-diagonal Hee that marginalization needs.
    //
    // A model can have several legal sets (bundle adjustment: cameras and
    // points couple only to each other, so BOTH {cameras} and {points}
    // qualify), and which one pays depends on instance counts the macro
    // cannot see. So every maximal independent set is emitted as a
    // candidate and the solver picks at runtime. Several landmark TYPES in
    // one set is the normal case, not a special one -- points and lines with
    // different block sizes are simply two nodes with no edge between them.
    //
    // TripletBlock roots emit nothing: their Hessian pattern is only known
    // after a compute pass, so no static claim about coupling is possible
    // (the Schur backend refuses those models anyway).
    let marginalize_candidates_fn = if root_precision.is_some() && !has_triplet_block {
        let cross = registry_cross_pairs();
        // Graph nodes are ENTITY types only: those owning a SelfBlock, i.e.
        // a diagonal Hessian block. Constraint structs (an Odo holding a
        // CrossBlock<Pose, Pose>) carry no parameters of their own and no
        // diagonal block, so they are not variables and cannot be
        // marginalized -- they are the EDGES of this graph, not nodes.
        let is_entity = |t: &str| {
            registry_lookup(t).is_some_and(|l| l.self_block_field.is_some())
        };
        let mut types: Vec<String> = Vec::new();
        for (_, _, elem) in &size_walk {
            if let Some(e) = elem
                && is_entity(e)
                && !types.contains(e) {
                    types.push(e.clone());
                }
        }
        let coupled = |a: &str, b: &str| {
            cross.iter().any(|(x, y)| {
                (x == a && y == b) || (x == b && y == a)
            })
        };
        // Independent sets, brute-forced: the graph has one node per entity
        // type, so it is tiny. Bail out rather than explode on a pathological
        // model.
        let mut candidates: Vec<Vec<String>> = Vec::new();
        if !types.is_empty() && types.len() <= 16 {
            let mut sets: Vec<Vec<String>> = Vec::new();
            for mask in 1u32..(1u32 << types.len()) {
                let members: Vec<String> = (0..types.len())
                    .filter(|i| mask & (1 << i) != 0)
                    .map(|i| types[i].clone())
                    .collect();
                // independent: no member self-couples, no pair couples
                let ok = members.iter().all(|a| !coupled(a, a))
                    && members.iter().enumerate().all(|(i, a)| {
                        members[i + 1..].iter().all(|b| !coupled(a, b))
                    });
                if ok {
                    sets.push(members);
                }
            }
            // keep only the maximal ones (no other set is a strict superset)
            for set in &sets {
                let maximal = !sets.iter().any(|other| {
                    other.len() > set.len() && set.iter().all(|m| other.contains(m))
                });
                if maximal {
                    candidates.push(set.clone());
                }
            }
        }
        // Every candidate must leave something behind: eliminating the whole
        // model is not a reduction, it is the same factorization by another
        // name.
        candidates.retain(|set| types.iter().any(|t| !set.contains(t)));

        if candidates.is_empty() {
            None
        } else {
            let per_candidate: Vec<TokenStream2> = candidates.iter().map(|set| {
                let last = size_walk.iter().rposition(|(_, _, e)|
                    e.as_ref().is_some_and(|e| set.contains(e)));
                let Some(last) = last else {
                    return quote! {};
                };
                let stmts: Vec<TokenStream2> = size_walk[..=last].iter().map(|(_, size, elem)| {
                    if elem.as_ref().is_some_and(|e| set.contains(e)) {
                        quote! {
                            let __start = __off;
                            __off += #size;
                            __ranges.push(__start..__off);
                        }
                    } else {
                        quote! { __off += #size; }
                    }
                }).collect();
                quote! {
                    {
                        let mut __ranges = std::vec::Vec::new();
                        let mut __off = 0usize;
                        #(#stmts)*
                        __out.push(__ranges);
                    }
                }
            }).collect();
            Some(quote! {
                fn marginalize_candidates(&self) -> std::vec::Vec<std::vec::Vec<std::ops::Range<usize>>> {
                    let mut __out = std::vec::Vec::new();
                    #(#per_candidate)*
                    __out
                }
            })
        }
    } else {
        None
    };

    // marginalize(fields): generate the RootProblem::marginalize_hint
    // override. Ranges come from the same field walk serialize uses, so
    // fixed params and nested models are counted identically.
    let marginalize_hint_fn = if root_eliminate.is_empty() {
        None
    } else {
        for id in &root_eliminate {
            if !size_walk.iter().any(|(f, _, _)| f == id) {
                return Err(syn::Error::new(id.span(), format!(
                    "marginalize: `{}` is not a parameter-bearing field of this struct", id)));
            }
        }
        // Walk fields in serialize order up to the last marked one.
        let last = size_walk.iter().rposition(|(f, _, _)|
            root_eliminate.iter().any(|id| id == f)).unwrap();
        let stmts: Vec<TokenStream2> = size_walk[..=last].iter().map(|(f, size, _)| {
            if root_eliminate.iter().any(|id| id == f) {
                quote! {
                    let __start = __off;
                    __off += #size;
                    __out.push(__start..__off);
                }
            } else {
                quote! { __off += #size; }
            }
        }).collect();
        Some(quote! {
            fn marginalize_hint(&self) -> std::vec::Vec<std::ops::Range<usize>> {
                let mut __out = std::vec::Vec::new();
                let mut __off = 0usize;
                #(#stmts)*
                __out
            }
        })
    };

    let constraint_impls = if let Some(ref precision) = root_precision {
        constraint::generate_root_methods(name, fields, precision, root_custom, root_jacobian,
            root_fast_atan, &marginalize_hint_fn, &marginalize_candidates_fn, has_triplet_block)?
    } else {
        quote! {}
    };

    Ok(quote! {
        #model_impl
        #precompute_block
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
            return name == "Param" || name == "SimpleEulerAngleParam" || name == "EulerAngleParam"
                || name == "QuaternionParam";
        }
    false
}

fn is_euler_angle_param_type(ty: &syn::Type) -> Option<&'static str> {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            if name == "SimpleEulerAngleParam" { return Some("simple"); }
            if name == "EulerAngleParam" { return Some("universal"); }
            // QuaternionParam: same ref_rotation * R(delta) composition, but the
            // delta is a rotation vector mapped through the exp map, not euler.
            if name == "QuaternionParam" { return Some("universal_rotvec"); }
        }
    None
}

/// Check if a type is `SelfBlock<...>` or `CrossBlock<...>`.
fn is_hessian_block_type(ty: &syn::Type) -> bool {
    if let syn::Type::Path(tp) = ty
        && let Some(seg) = tp.path.segments.last() {
            let name = seg.ident.to_string();
            return matches!(name.as_str(),
                "SelfBlock" | "CrossBlock" | "TripletBlock" | "BoxedSelfBlock" | "BoxedCrossBlock");
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

/// A type's spelling as written, with token-stream whitespace removed
/// around `::`, `<`, `>` and after commas: `refs :: Vec < Pose >` reads
/// back as `refs::Vec<Pose>`. Recorded per field for the JSON sidecar.
fn type_spelling(ty: &syn::Type) -> String {
    quote! { #ty }.to_string()
        .replace(" :: ", "::")
        .replace(" < ", "<")
        .replace(" > ", ">")
        .replace(" >", ">")
        .replace(" ,", ",")
}

/// Names that can never be a registered model type -- filtered out of
/// suspect-wrapper recording so `Vec<Pose<f32>>` seen before `Pose`'s own
/// expansion records no noise.
fn is_never_model_name(name: &str) -> bool {
    matches!(name,
        "f32" | "f64" | "bool" | "u8" | "u16" | "u32" | "u64" | "usize"
        | "i8" | "i16" | "i32" | "i64" | "isize" | "char" | "String" | "str"
        | "vect2f" | "vect2d" | "vect2" | "vect3f" | "vect3d" | "vect3"
        | "matrix2f" | "matrix2d" | "matrix2" | "matrix3f" | "matrix3d" | "matrix3"
        | "quaternf" | "quaternd" | "quatern")
}

/// Last-segment idents of every type mentioned in `ty`'s subtree, minus
/// scalar/math names and the owning struct's scalar generic.
fn collect_type_idents(ty: &syn::Type, scalar_generic: Option<&str>, out: &mut Vec<String>) {
    match ty {
        syn::Type::Path(tp) => {
            if let Some(seg) = tp.path.segments.last() {
                let name = seg.ident.to_string();
                if Some(name.as_str()) != scalar_generic && !is_never_model_name(&name) {
                    out.push(name);
                }
                if let syn::PathArguments::AngleBracketed(args) = &seg.arguments {
                    for a in &args.args {
                        if let syn::GenericArgument::Type(t) = a {
                            collect_type_idents(t, scalar_generic, out);
                        }
                    }
                }
            }
        }
        syn::Type::Reference(r) => collect_type_idents(&r.elem, scalar_generic, out),
        syn::Type::Tuple(t) => {
            for e in &t.elems { collect_type_idents(e, scalar_generic, out); }
        }
        _ => {}
    }
}

/// (wrapper, held) pairs where `held` sits inside a generic wrapper the
/// macro does not recognize. Containers are dispatched by literal
/// last-segment name, so `use refs::Vec as RVec` + `pts: RVec<P>` reads
/// as an opaque data field: `P` silently stops being containment there.
/// Recognized containers are recursed into (`Vec<RVec<P>>` still flags);
/// block/param wrappers stop -- their model-type argument is legitimate.
fn collect_wrapper_suspects(
    ty: &syn::Type,
    scalar_generic: Option<&str>,
    out: &mut Vec<(String, String)>,
) {
    let syn::Type::Path(tp) = ty else { return };
    let Some(seg) = tp.path.segments.last() else { return };
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else { return };
    let type_args: Vec<&syn::Type> = args.args.iter().filter_map(|a| match a {
        syn::GenericArgument::Type(t) => Some(t),
        _ => None,
    }).collect();
    if type_args.is_empty() { return; }
    let name = seg.ident.to_string();
    match name.as_str() {
        "Vec" | "Deque" | "Arena" | "Option" | "Ref" => {
            for t in type_args { collect_wrapper_suspects(t, scalar_generic, out); }
        }
        "SelfBlock" | "CrossBlock" | "TripletBlock" | "BoxedSelfBlock" | "BoxedCrossBlock"
        | "Param" | "SimpleEulerAngleParam" | "EulerAngleParam" | "QuaternionParam"
        | "PhantomData" => {}
        _ => {
            // A registered model type's own generics are its scalar
            // parameter (`Pose<f32>`), not containment.
            if Some(name.as_str()) == scalar_generic || registry_lookup(&name).is_some() {
                return;
            }
            let mut held = Vec::new();
            for t in type_args { collect_type_idents(t, scalar_generic, &mut held); }
            for h in held { out.push((name.clone(), h)); }
        }
    }
}

/// Solve precision of a block field's user-spelled type (pre-rewrite):
/// `SelfBlock<A[, S]>` / `CrossBlock<A, B[, S]>` / `TripletBlock[<S>]`,
/// Boxed and Option-wrapped variants included. None if not a block field
/// or the scalar spelling is unrecognized. A missing scalar is the f64
/// default; the struct's own type parameter reads as "generic".
fn block_field_precision(ty: &syn::Type, scalar_generic: Option<&str>) -> Option<String> {
    let ty = if let Some((inner, _)) = extract_wrapper_inner(ty, "Option") { inner } else { ty };
    let syn::Type::Path(tp) = ty else { return None };
    let seg = tp.path.segments.last()?;
    let scalar_pos = match seg.ident.to_string().as_str() {
        "TripletBlock" => 0,
        "SelfBlock" | "BoxedSelfBlock" => 1,
        "CrossBlock" | "BoxedCrossBlock" => 2,
        _ => return None,
    };
    let type_args: Vec<&syn::Type> = match &seg.arguments {
        syn::PathArguments::AngleBracketed(args) => args.args.iter()
            .filter_map(|a| if let syn::GenericArgument::Type(t) = a { Some(t) } else { None })
            .collect(),
        _ => Vec::new(),
    };
    let Some(syn::Type::Path(stp)) = type_args.get(scalar_pos) else {
        return Some("f64".to_string());
    };
    let scalar = stp.path.segments.last()?.ident.to_string();
    match scalar.as_str() {
        "f32" | "f64" => Some(scalar),
        s if Some(s) == scalar_generic => Some("generic".to_string()),
        _ => None,
    }
}

/// Element instantiation with an explicit float first argument:
/// `Vec<G<f32>>` / `Option<G<f32>>` / a bare `g: G<f32>` field. Returns
/// (element type name, "f32" | "f64"). Blocks, params, and math types are
/// not elements.
fn inst_precision_of(ty: &syn::Type) -> Option<(String, String)> {
    let elem = ["Vec", "Deque", "Arena", "Option"].iter()
        .find_map(|w| extract_wrapper_inner(ty, w).map(|(t, _)| t))
        .unwrap_or(ty);
    let syn::Type::Path(tp) = elem else { return None };
    let seg = tp.path.segments.last()?;
    let name = seg.ident.to_string();
    if is_never_model_name(&name) || is_param_type(elem) || is_hessian_block_type(elem) {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &seg.arguments else { return None };
    let first = args.args.iter().find_map(|a|
        if let syn::GenericArgument::Type(t) = a { Some(t) } else { None })?;
    let syn::Type::Path(ftp) = first else { return None };
    let fname = ftp.path.segments.last()?.ident.to_string();
    match fname.as_str() {
        "f32" | "f64" => Some((name, fname)),
        _ => None,
    }
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
    generics: &syn::Generics,
    fields: &syn::punctuated::Punctuated<syn::Field, syn::token::Comma>,
) -> syn::Result<TokenStream2> {
    let sym_name = syn::Ident::new(&format!("{}Sym", name), name.span());

    // Sym twins are precision-independent, so a generic struct shares ONE
    // non-generic companion: project field types to the f64 instantiation
    // when naming their associated sym types.
    let scalar_generic = generics.type_params().next().map(|tp| tp.ident.to_string());
    let subst = |ty: &syn::Type| -> syn::Type {
        match &scalar_generic {
            Some(g) => subst_ident_f64(ty, g),
            None => ty.clone(),
        }
    };

    let mut sym_fields: Vec<TokenStream2> = Vec::new();
    let mut sym_inits: Vec<TokenStream2> = Vec::new();

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let ty = &field.ty;

        // Skip fields with #[arael(skip)], #[arael(compute = ...)], or #[arael(constraint_index)]
        match parse_arael_attr(&field.attrs)? {
            Some(AraelAttr::Skip) | Some(AraelAttr::Compute(_)) | Some(AraelAttr::ConstraintIndex)
            | Some(AraelAttr::Deriv { .. }) => continue,
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
            let inner_ty = subst(inner_ty);
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
            let inner_ty = subst(inner_ty);
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
        let ty = subst(ty);
        sym_fields.push(quote! {
            pub #ident: <#ty as arael::model::ModelSym>::Sym,
        });

        sym_inits.push(quote! {
            #ident: <#ty as arael::model::ModelSym>::sym(&format!("{}.{}", base, #field_name)),
        });
    }

    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    Ok(quote! {
        #[derive(Clone)]
        pub struct #sym_name {
            #(#sym_fields)*
        }

        impl #impl_generics arael::model::ModelSym for #name #ty_generics #where_clause {
            type Sym = #sym_name;
            fn sym(base: &str) -> #sym_name {
                #sym_name {
                    #(#sym_inits)*
                }
            }
        }
    })
}

/// Whether a token tree contains the bare ident `name` at any depth.
fn token_stream_has_ident(tt: proc_macro2::TokenTree, name: &str) -> bool {
    match tt {
        proc_macro2::TokenTree::Ident(id) => id == name,
        proc_macro2::TokenTree::Group(g) =>
            g.stream().into_iter().any(|t| token_stream_has_ident(t, name)),
        _ => false,
    }
}

/// Replace every occurrence of the bare ident `from` in a type's token
/// stream with `f64`. Projects a generic struct's field types onto the
/// f64 instantiation for its sym companion: sym twins are
/// precision-independent, so any concrete instantiation names the same
/// associated types.
fn subst_ident_f64(ty: &syn::Type, from: &str) -> syn::Type {
    fn walk(ts: TokenStream2, from: &str) -> TokenStream2 {
        ts.into_iter().map(|tt| match tt {
            proc_macro2::TokenTree::Ident(id) if id == from =>
                proc_macro2::TokenTree::Ident(proc_macro2::Ident::new("f64", id.span())),
            proc_macro2::TokenTree::Group(g) => {
                let inner = walk(g.stream(), from);
                let mut ng = proc_macro2::Group::new(g.delimiter(), inner);
                ng.set_span(g.span());
                proc_macro2::TokenTree::Group(ng)
            }
            other => other,
        }).collect()
    }
    let tokens = walk(quote! { #ty }, from);
    syn::parse2(tokens).expect("type with scalar generic substituted must reparse")
}

pub(crate) enum AraelAttr {
    Skip,
    Compute(TokenStream2),
    /// `#[arael(symbolic = <expr>)]`: constraint-body reads of this data
    /// field expand to the expression (a derivative-carrying computed field).
    Symbolic(TokenStream2),
    /// `#[arael(deriv = <symbolic field>, by = <param field>)]`: this field
    /// is the declared Jacobian cache `d(of)/d(by)` -- an array with one
    /// entry per component of `by`, each shaped like `of`. Filled by the
    /// generated symbolic precompute; constraint Jacobians read it instead
    /// of re-deriving the expression per observation.
    Deriv { of: String, by: String },
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
                if kw == "symbolic" {
                    if tokens.len() >= 3
                        && let proc_macro2::TokenTree::Punct(ref p) = tokens[1]
                            && p.as_char() == '=' {
                                let expr_tokens: TokenStream2 =
                                    tokens[2..].iter().cloned().collect();
                                // Validate now so a malformed expression errors
                                // at the field, not deep in body resolution.
                                syn::parse2::<syn::Expr>(expr_tokens.clone())
                                    .map_err(|e| syn::Error::new_spanned(&tokens[0],
                                        format!("symbolic = expression does not parse: {}", e)))?;
                                return Ok(Some(AraelAttr::Symbolic(expr_tokens)));
                            }
                    return Err(syn::Error::new_spanned(
                        &tokens[0],
                        "expected `symbolic = <expression>`",
                    ));
                }
                // #[arael(deriv = <field>, by = <param field>)]
                if kw == "deriv" {
                    let parts: Vec<String> = tokens.iter().skip(1)
                        .filter_map(|t| match t {
                            proc_macro2::TokenTree::Ident(id) => Some(id.to_string()),
                            _ => None,
                        })
                        .collect();
                    if tokens.len() >= 3
                        && matches!(&tokens[1],
                            proc_macro2::TokenTree::Punct(p) if p.as_char() == '=')
                        && parts.len() == 3
                        && parts[1] == "by"
                    {
                        return Ok(Some(AraelAttr::Deriv {
                            of: parts[0].clone(),
                            by: parts[2].clone(),
                        }));
                    }
                    return Err(syn::Error::new_spanned(
                        &tokens[0],
                        "expected `deriv = <symbolic field>, by = <param field>`",
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
                "unknown arael attribute, expected `skip`, `compute = <expr>`, `symbolic = <expr>`, `ref = <path>`, `constraint_index`, or `cross = (refA, refB)`",
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
    /// "f32" for `fit(...)`, "f64" for `fit64(...)`.
    precision: &'static str,
    /// Optional robust-loss closure `|s| <expr>` over the squared residual, e.g.
    /// "|s| loss_cauchy(s, gamma)". Stored as source, parsed at codegen.
    loss: Option<String>,
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
            let precision = if *ident == "fit" {
                "f32"
            } else if *ident == "fit64" {
                "f64"
            } else {
                continue;
            };

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
                return parse_fit_inner(&inner, ident, precision);
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
    precision: &'static str,
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

    // Split off an optional trailing `, loss = |s| <expr>`. A residual body has
    // no top-level comma (call-argument commas sit inside group tokens), so the
    // first top-level comma followed by `loss` is the separator.
    let mut loss: Option<String> = None;
    let mut body_end = tokens.len();
    for i in pos..tokens.len() {
        if let proc_macro2::TokenTree::Punct(p) = &tokens[i] {
            if p.as_char() == ','
                && matches!(tokens.get(i + 1), Some(proc_macro2::TokenTree::Ident(id)) if id == "loss")
            {
                body_end = i;
                match tokens.get(i + 2) {
                    Some(proc_macro2::TokenTree::Punct(p)) if p.as_char() == '=' => {}
                    _ => return Err(syn::Error::new_spanned(err_span, "expected `loss = |s| <expr>`")),
                }
                let loss_ts: TokenStream2 = tokens[i + 3..].iter().cloned().collect();
                if loss_ts.is_empty() {
                    return Err(syn::Error::new_spanned(err_span, "expected a closure after `loss =`"));
                }
                loss = Some(loss_ts.to_string());
                break;
            }
        }
    }

    let body_tokens = &tokens[pos..body_end];
    // Either { block } or a direct expression.
    let body_stmts = match body_tokens.first() {
        Some(proc_macro2::TokenTree::Group(g))
            if body_tokens.len() == 1 && g.delimiter() == proc_macro2::Delimiter::Brace =>
        {
            let block_tokens =
                proc_macro2::TokenStream::from(proc_macro2::TokenTree::Group(g.clone()));
            let block: syn::Block = syn::parse2(block_tokens)?;
            block.stmts
        }
        _ => {
            let remaining: TokenStream2 = body_tokens.iter().cloned().collect();
            let expr: Expr = syn::parse2(remaining)?;
            vec![Stmt::Expr(expr, None)]
        }
    };

    Ok(Some(FitAttr {
        data_field,
        loop_var,
        body_stmts,
        precision,
        loss,
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
    // Precision-dependent pieces: `fit(...)` is f32, `fit64(...)` is f64.
    let prec = fit.precision;
    let prec_type: syn::Type = syn::parse_str(prec)
        .map_err(|e| syn::Error::new_spanned(name,
            format!("invalid fit precision '{}': {}", prec, e)))?;
    let (fit_serialize, fit_deserialize) = if prec == "f32" {
        (quote! { serialize_params32 }, quote! { deserialize_params32 })
    } else {
        (quote! { serialize_params64 }, quote! { deserialize_params64 })
    };
    // 1. Classify fields
    let mut param_names: Vec<String> = Vec::new();
    let mut constant_names: HashSet<String> = HashSet::new();
    let data_field_name = fit.data_field.to_string();

    for field in fields {
        let ident = field.ident.as_ref().unwrap();
        let field_name = ident.to_string();
        let attr = parse_arael_attr(&field.attrs)?;
        if matches!(attr, Some(AraelAttr::Skip) | Some(AraelAttr::Compute(_)) | Some(AraelAttr::Deriv { .. })) {
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

    // Optional robust loss `|s| rho(s)` over the squared residual s = r^2. The
    // body is evaluated against the same ctx as the residual (constants like
    // `gamma` resolve identically), with the argument bound to the synthetic
    // LOSS_ARG_SYM symbol. rho(s) contributes to the cost; the weight rho'(s)
    // scales that point's gradient and Gauss-Newton Hessian.
    let loss_rho: Option<arael_sym::E> = if let Some(loss_src) = &fit.loss {
        let closure: syn::ExprClosure = syn::parse_str(loss_src).map_err(|e| {
            syn::Error::new_spanned(&fit.data_field,
                format!("fit `loss` must be a closure `|s| <expr>`: {e}"))
        })?;
        if closure.inputs.len() != 1 {
            return Err(syn::Error::new_spanned(&fit.data_field,
                "fit `loss` closure takes exactly one argument (the squared residual)"));
        }
        let arg = match &closure.inputs[0] {
            Pat::Ident(pi) => pi.ident.to_string(),
            Pat::Type(pt) => match &*pt.pat {
                Pat::Ident(pi) => pi.ident.to_string(),
                _ => return Err(syn::Error::new_spanned(&fit.data_field,
                    "fit `loss` argument must be a plain identifier")),
            },
            _ => return Err(syn::Error::new_spanned(&fit.data_field,
                "fit `loss` argument must be a plain identifier")),
        };
        ctx.let_bindings.insert(arg, arael_sym::symbol(constraint::LOSS_ARG_SYM));
        Some(syn_expr_to_sym(&closure.body, &mut ctx)?)
    } else {
        None
    };

    // 3. Differentiate w.r.t. each param
    let n = param_names.len();
    let derivatives: Vec<arael_sym::E> = param_names
        .iter()
        .map(|p| residual.diff(p.as_str()))
        .collect();

    // 4. Generate Rust code strings and parse to syn::Expr
    let r_code = residual.to_rust(prec);
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
            let code = d.to_rust(prec);
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
            let prec_type = &prec_type;
            quote! { let #dr_id: #prec_type = #dr_expr; }
        })
        .collect();

    // Robust-loss fragments. `s = __block_cost = r^2`; the loss contributes
    // rho(s) to the cost and its weight rho'(s) scales every gradient/Hessian
    // write. Without a loss every piece is empty and the emission is
    // byte-identical to the plain least-squares form.
    let loss_arg_id = proc_macro2::Ident::new(constraint::LOSS_ARG_SYM, proc_macro2::Span::call_site());
    let emit_loss = |want_weight: bool| -> syn::Result<(TokenStream2, Expr)> {
        let rho = loss_rho.as_ref().unwrap();
        let mut exprs = vec![rho.clone()];
        if want_weight {
            exprs.push(rho.diff(constraint::LOSS_ARG_SYM));
        }
        let (ints, simplified) = arael_sym::cse(&exprs);
        let mut stmts = vec![quote! { let #loss_arg_id: #prec_type = __r * __r; }];
        for (nm, e) in &ints {
            let id = proc_macro2::Ident::new(nm, proc_macro2::Span::call_site());
            let code: Expr = syn::parse_str(&e.to_rust(prec))?;
            stmts.push(quote! { let #id = #code; });
        }
        if want_weight {
            let w_code: Expr = syn::parse_str(&simplified[1].to_rust(prec))?;
            stmts.push(quote! { let __w: #prec_type = #w_code; });
        }
        let rho_code: Expr = syn::parse_str(&simplified[0].to_rust(prec))?;
        Ok((quote! { #(#stmts)* }, rho_code))
    };
    let (cost_loss_setup, cost_add, gh_loss_setup, gh_cost_add, weight_mul):
        (TokenStream2, TokenStream2, TokenStream2, TokenStream2, TokenStream2) = if loss_rho.is_some() {
        let (cost_setup, cost_rho) = emit_loss(false)?;
        let (gh_setup, gh_rho) = emit_loss(true)?;
        (
            cost_setup,
            quote! { __cost += (#cost_rho) as #prec_type; },
            gh_setup,
            quote! { __cost += (#gh_rho) as #prec_type; },
            quote! { __w * },
        )
    } else {
        (
            quote! {}, quote! { __cost += __r * __r; },
            quote! {}, quote! { __cost += __r * __r; },
            quote! {},
        )
    };

    // Gradient accumulation
    let grad_accum: Vec<TokenStream2> = (0..n)
        .map(|i| {
            let dr_id = &dr_idents[i];
            let prec_type = &prec_type;
            let weight_mul = &weight_mul;
            quote! { grad[#i] += (2.0 as #prec_type) * #weight_mul __r * #dr_id; }
        })
        .collect();

    // Hessian accumulation (upper triangle only)
    let hessian_accum: Vec<TokenStream2> = (0..n)
        .flat_map(|i| {
            let dr_idents = &dr_idents;
            let prec_type = &prec_type;
            let weight_mul = &weight_mul;
            (i..n).map(move |j| {
                let idx = i * n + j;
                let dr_i = &dr_idents[i];
                let dr_j = &dr_idents[j];
                quote! { hessian[#idx] += (2.0 as #prec_type) * #weight_mul #dr_i * #dr_j; }
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
        impl arael::simple_lm::LmProblem<#prec_type> for #name {
            fn calc_cost(&mut self, params: &[#prec_type]) -> #prec_type {
                #(#param_unpack)*
                #(#constant_bind)*
                let mut __cost = (0.0 as #prec_type);
                for #loop_var_id in &self.#data_field_id {
                    #(#data_bind)*
                    let __r: #prec_type = #r_expr;
                    #cost_loss_setup
                    #cost_add
                }
                __cost
            }

            fn calc_grad_hessian_dense(
                &mut self,
                params: &[#prec_type],
                grad: &mut [#prec_type],
                hessian: &mut [#prec_type],
            ) -> #prec_type {
                #(#param_unpack)*
                #(#constant_bind)*
                grad.iter_mut().for_each(|g| *g = 0.0);
                hessian.iter_mut().for_each(|h| *h = 0.0);
                let mut __cost = (0.0 as #prec_type);
                for #loop_var_id in &self.#data_field_id {
                    #(#data_bind)*
                    let __r: #prec_type = #r_expr;
                    #gh_loss_setup
                    #gh_cost_add
                    #(#dr_bindings)*
                    #(#grad_accum)*
                    #(#hessian_accum)*
                }
                #(#hessian_symmetry)*
                __cost
            }

            fn calc_grad_hessian_band(
                &mut self,
                _params: &[#prec_type],
                _grad: &mut [#prec_type],
                _band: &mut [#prec_type],
                _kd: usize,
            ) -> Result<#prec_type, arael::simple_lm::BandOverflow> {
                unimplemented!("fit models do not support band assembly")
            }

            fn calc_grad_hessian_sparse(
                &mut self,
                _params: &[#prec_type],
                _grad: &mut [#prec_type],
                _coo: &mut arael::simple_lm::CooMatrix<#prec_type>,
            ) -> #prec_type {
                unimplemented!("fit models do not support sparse assembly")
            }

            fn calc_grad_hessian_sparse_direct(
                &mut self,
                _params: &[#prec_type],
                _grad: &mut [#prec_type],
                _csc: &mut arael::simple_lm::CscMatrix<#prec_type>,
            ) -> #prec_type {
                unimplemented!("fit models do not support sparse direct assembly")
            }

            fn calc_grad_hessian_sparse_indexed(
                &mut self,
                _params: &[#prec_type],
                _grad: &mut [#prec_type],
                _vals: &mut [#prec_type],
                _positions: &[arael::ValueIndex],
            ) -> #prec_type {
                unimplemented!("fit models do not support sparse indexed assembly")
            }
        }

        impl arael::simple_lm::FitProblem<#prec_type> for #name {
            fn serialize(&mut self, data: &mut std::vec::Vec<#prec_type>) {
                arael::model::Model::#fit_serialize(self, data)
            }
            fn deserialize(&mut self, data: &[#prec_type]) {
                arael::model::Model::#fit_deserialize(self, data)
            }
        }
    })
}
