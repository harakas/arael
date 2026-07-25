//! C++ header emitter: thin inline wrapper classes over the shim's C
//! ABI. Entity views hold raw pointers (stable for refs::Vec-backed
//! collections; std::vec::Vec-backed ones invalidate on push -- the
//! view's comment says which).

use crate::ir::{Field, Model, snake};

fn scalar_cpp(of: &str) -> Option<&'static str> {
    Some(match of {
        "f64" => "double",
        "f32" => "float",
        "bool" => "bool",
        "u32" => "uint32_t",
        "i32" => "int32_t",
        _ => return None,
    })
}

fn float_cpp(precision: &str) -> &'static str {
    if precision == "f32" { "float" } else { "double" }
}

/// (ffi decls, class methods) for one accessor-bearing field.
fn field_surface(
    fn_prefix: &str,
    ptr_ty: &str,
    ptr_expr: &str,
    f: &Field,
) -> Option<(String, String)> {
    let name = &f.name;
    let of = f.of.as_deref().unwrap_or("");
    match f.kind.as_str() {
        "data" | "param" => {
            let c = scalar_cpp(of)?;
            let decls = format!(
                "{c} {fn_prefix}_{name}(const {ptr_ty}*);\nvoid {fn_prefix}_set_{name}({ptr_ty}*, {c});\n");
            let methods = format!(
                "    {c} {name}() const {{ return ffi::{fn_prefix}_{name}({ptr_expr}); }}\n\
                 \x20   void set_{name}({c} v) {{ ffi::{fn_prefix}_set_{name}({ptr_expr}, v); }}\n");
            Some((decls, methods))
        }
        _ => None,
    }
}

pub fn emit(model: &Model) -> Result<String, String> {
    let root = &model.root;
    let root_sn = snake(root);
    let fp = float_cpp(&model.precision);
    let mut ffi_decls = String::new();
    let mut classes = String::new();

    // Opaque C types: the handle plus every entity.
    let mut opaque: Vec<String> = vec![root.clone()];
    let mut entities: Vec<(&String, &crate::ir::Type)> = Vec::new();
    for (tn, t) in &model.types {
        if t.role == "entity" && !t.builtin {
            opaque.push(tn.clone());
            entities.push((tn, t));
        }
    }

    for (tn, t) in &entities {
        let sn = snake(tn);
        let mut methods = String::new();
        let mut opaque_note = String::new();
        for f in &t.fields {
            if let Some((d, m)) = field_surface(&sn, tn, "p_", f) {
                ffi_decls.push_str(&d);
                methods.push_str(&m);
            } else if f.kind == "opaque" {
                opaque_note.push_str(&format!(
                    "    // field `{}`: {} -- opaque, no accessor generated\n",
                    f.name, f.of.as_deref().unwrap_or("?")));
            }
        }
        classes.push_str(&format!(
"/// A `{tn}` in its collection. Thin pointer wrapper -- validity
/// follows the collection's storage (see the view).
class {tn}Ref {{
public:
    explicit {tn}Ref(ffi::{tn}* p) : p_(p) {{}}
{methods}{opaque_note}private:
    ffi::{tn}* p_;
}};

"));
    }

    // Root collections -> views; root scalar fields -> World methods.
    let root_ty = model.types.get(root).ok_or("root type missing")?;
    let mut world_methods = String::new();
    let mut world_opaque_note = String::new();
    let mut views = String::new();
    for f in &root_ty.fields {
        match f.kind.as_str() {
            "collection" => {
                let elem = f.of.as_deref().ok_or("collection without element")?;
                let field = &f.name;
                let prefix = format!("{root_sn}_{field}");
                let stable = f.spelled.as_deref().unwrap_or("").starts_with("refs::");
                let stability = if stable {
                    "Element pointers are STABLE across pushes (chunked storage)."
                } else {
                    "std::vec::Vec storage: pushes may MOVE elements -- re-fetch\n/// element refs after a push."
                };
                ffi_decls.push_str(&format!(
                    "uint32_t {prefix}_len(const {root}*);\n\
                     {elem}* {prefix}_push({root}*);\n\
                     {elem}* {prefix}_at({root}*, uint32_t);\n"));
                let view = format!("{}View", camel(field));
                views.push_str(&format!(
"/// `{root}.{field}`. {stability}
class {view} {{
public:
    explicit {view}(ffi::{root}* h) : h_(h) {{}}
    uint32_t size() const {{ return ffi::{prefix}_len(h_); }}
    {elem}Ref push() {{ return {elem}Ref(ffi::{prefix}_push(h_)); }}
    {elem}Ref operator[](uint32_t i) {{ return {elem}Ref(ffi::{prefix}_at(h_, i)); }}
private:
    ffi::{root}* h_;
}};

"));
                world_methods.push_str(&format!(
                    "    {view} {field}() {{ return {view}(h_); }}\n"));
            }
            _ => {
                if let Some((d, m)) = field_surface(&root_sn, root, "h_", f) {
                    ffi_decls.push_str(&d);
                    world_methods.push_str(&m);
                } else if f.kind == "opaque" {
                    world_opaque_note.push_str(&format!(
                        "    // field `{}`: {} -- opaque, no accessor generated\n",
                        f.name, f.of.as_deref().unwrap_or("?")));
                }
            }
        }
    }

    let opaque_decls: String = opaque.iter()
        .map(|t| format!("struct {t};\n")).collect();

    ffi_decls.push_str(&format!(
        "{root}* {root_sn}_new(void);\n\
         void {root_sn}_free({root}*);\n\
         const char* {root_sn}_last_error(const {root}*);\n\
         const char* {root_sn}_validate({root}*);\n\
         int32_t {root_sn}_solve_dense({root}*, const LmConfig*, LmResult*);\n\
         int32_t {root_sn}_solve_sparse({root}*, const LmConfig*, LmResult*);\n"));

    Ok(format!(
"// GENERATED by cargo-arael from the `{root}` model sidecar. Do not
// edit; regenerate with `cargo arael export`.
#pragma once

#include <cstdint>
#include <cmath>

namespace arael {{

/// Why a solve stopped. Non-negative codes come from the solver;
/// SolverFailed carries text via last_error(), Panicked likewise.
enum class LmStatus : int32_t {{
    Converged = 0,
    CostThreshold = 1,
    MaxIterations = 2,
    GradientTolerance = 3,
    ParameterTolerance = 4,
    PredictedReduction = 5,
    LambdaCeiling = 6,
    DriverTerminated = 7,
    ObserverTerminated = 8,
    TimeLimit = 9,
    RetryBudgetExhausted = 10,
    Aborted = 11,
    SolverFailed = -1,
    Panicked = -2,
}};

/// Sentinel-based: fields left at UINT32_MAX / NaN keep the preset's
/// value (preset 0 = solver defaults, 1 = conservative).
struct LmConfig {{
    uint32_t preset = 0;
    uint32_t max_iters = UINT32_MAX;
    uint32_t min_iters = UINT32_MAX;
    uint32_t patience = UINT32_MAX;
    {fp} abs_precision = NAN;
    {fp} rel_precision = NAN;
    {fp} initial_lambda = NAN;
    {fp} cost_threshold = NAN;

    static LmConfig defaults() {{ return LmConfig{{}}; }}
    static LmConfig conservative() {{ LmConfig c; c.preset = 1; return c; }}
}};

struct LmResult {{
    {fp} start_cost;
    {fp} end_cost;
    uint32_t iterations;
    uint32_t accepted_iterations;
    LmStatus status;
    {fp} lambda;

    /// True for the healthy terminations (the solver returned a state
    /// it stands behind).
    bool ok() const {{ return static_cast<int32_t>(status) >= 0
        && status != LmStatus::Aborted; }}
}};

namespace ffi {{
{opaque_decls}
extern \"C\" {{
{ffi_decls}}}
}} // namespace ffi

{classes}{views}/// The `{root}` model. Owns the Rust-side object; move-only.
class {root} {{
public:
    {root}() : h_(ffi::{root_sn}_new()) {{}}
    ~{root}() {{ if (h_) ffi::{root_sn}_free(h_); }}
    {root}(const {root}&) = delete;
    {root}& operator=(const {root}&) = delete;
    {root}({root}&& o) noexcept : h_(o.h_) {{ o.h_ = nullptr; }}
    {root}& operator=({root}&& o) noexcept {{
        if (this != &o) {{
            if (h_) ffi::{root_sn}_free(h_);
            h_ = o.h_;
            o.h_ = nullptr;
        }}
        return *this;
    }}

{world_methods}{world_opaque_note}
    LmResult solve_dense(const LmConfig& cfg = LmConfig{{}}) {{
        LmResult r;
        ffi::{root_sn}_solve_dense(h_, &cfg, &r);
        return r;
    }}
    LmResult solve_sparse(const LmConfig& cfg = LmConfig{{}}) {{
        LmResult r;
        ffi::{root_sn}_solve_sparse(h_, &cfg, &r);
        return r;
    }}
    /// Empty string when the model is clean, the Diagnostic text
    /// otherwise. The returned pointer is valid until the next call on
    /// this model.
    const char* validate() {{ return ffi::{root_sn}_validate(h_); }}
    const char* last_error() const {{ return ffi::{root_sn}_last_error(h_); }}

private:
    ffi::{root}* h_;
}};

}} // namespace arael
"))
}

/// CamelCase of a snake field name for the view class: pose_ties -> PoseTies.
fn camel(field: &str) -> String {
    let mut out = String::new();
    let mut up = true;
    for c in field.chars() {
        if c == '_' {
            up = true;
        } else if up {
            out.push(c.to_ascii_uppercase());
            up = false;
        } else {
            out.push(c);
        }
    }
    out
}
