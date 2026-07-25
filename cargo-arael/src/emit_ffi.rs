//! Rust FFI shim emitter: `#[no_mangle] extern "C"` functions over the
//! model, compiled in the generated capi crate (edition 2021, so no
//! inner unsafe blocks are needed). The surface is pure C ABI -- it
//! serves both the C++ header and the Python ctypes module.
//!
//! Field coverage: scalar and math-typed data and params, euler /
//! quaternion rotation params (typed `.value` access -- the documented
//! set-before / read-after fields), built-in components (TransformParam
//! translation+rotation, UnitVecParam unit), user components and nested
//! sub-models (pointer accessors + their own field surface), refs
//! (packed u32 transport), Option entities, and vec / deque / arena
//! collections at any depth.

use crate::ir::{Field, Model, Type, snake};

fn scalar_c(of: &str) -> Option<&'static str> {
    Some(match of {
        "f64" => "f64",
        "f32" => "f32",
        "bool" => "bool",
        "u32" => "u32",
        "i32" => "i32",
        _ => return None,
    })
}

/// Math type -> its `#[repr(C)]` mirror in the shim.
fn math_mirror(of: &str) -> Option<&'static str> {
    Some(match of {
        "vect2f" => "CVec2F32",
        "vect2d" => "CVec2F64",
        "vect3f" => "CVec3F32",
        "vect3d" => "CVec3F64",
        "matrix2f" => "CMat2F32",
        "matrix2d" => "CMat2F64",
        "matrix3f" => "CMat3F32",
        "matrix3d" => "CMat3F64",
        "quaternf" => "CQuatF32",
        "quaternd" => "CQuatF64",
        _ => return None,
    })
}

/// The repr(C) mirror types and their conversions, emitted once.
const MIRRORS: &str = r#"
macro_rules! c_vec2 {
    ($name:ident, $t:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name { pub x: $t, pub y: $t }
        impl From<arael::vect::vect2<$t>> for $name {
            fn from(v: arael::vect::vect2<$t>) -> Self { Self { x: v.x, y: v.y } }
        }
        impl From<$name> for arael::vect::vect2<$t> {
            fn from(v: $name) -> Self { arael::vect::vect2::new(v.x, v.y) }
        }
    };
}
macro_rules! c_vec3 {
    ($name:ident, $t:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name { pub x: $t, pub y: $t, pub z: $t }
        impl From<arael::vect::vect3<$t>> for $name {
            fn from(v: arael::vect::vect3<$t>) -> Self { Self { x: v.x, y: v.y, z: v.z } }
        }
        impl From<$name> for arael::vect::vect3<$t> {
            fn from(v: $name) -> Self { arael::vect::vect3::new(v.x, v.y, v.z) }
        }
    };
}
macro_rules! c_mat2 {
    ($name:ident, $t:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name { pub m: [[$t; 2]; 2] }
        impl From<arael::matrix::matrix2<$t>> for $name {
            fn from(v: arael::matrix::matrix2<$t>) -> Self {
                Self { m: [[v.rows[0].x, v.rows[0].y], [v.rows[1].x, v.rows[1].y]] }
            }
        }
        impl From<$name> for arael::matrix::matrix2<$t> {
            fn from(v: $name) -> Self {
                arael::matrix::matrix2::from_elements(v.m[0][0], v.m[0][1], v.m[1][0], v.m[1][1])
            }
        }
    };
}
macro_rules! c_mat3 {
    ($name:ident, $t:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name { pub m: [[$t; 3]; 3] }
        impl From<arael::matrix::matrix3<$t>> for $name {
            fn from(v: arael::matrix::matrix3<$t>) -> Self {
                Self { m: [
                    [v.rows[0].x, v.rows[0].y, v.rows[0].z],
                    [v.rows[1].x, v.rows[1].y, v.rows[1].z],
                    [v.rows[2].x, v.rows[2].y, v.rows[2].z],
                ] }
            }
        }
        impl From<$name> for arael::matrix::matrix3<$t> {
            fn from(v: $name) -> Self { arael::matrix::matrix3::from_array(v.m) }
        }
    };
}
macro_rules! c_quat {
    ($name:ident, $t:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy)]
        pub struct $name { pub t: $t, pub x: $t, pub y: $t, pub z: $t }
        impl From<arael::quatern::quatern<$t>> for $name {
            fn from(q: arael::quatern::quatern<$t>) -> Self {
                Self { t: q.t, x: q.v.x, y: q.v.y, z: q.v.z }
            }
        }
        impl From<$name> for arael::quatern::quatern<$t> {
            fn from(q: $name) -> Self {
                arael::quatern::quatern::new(q.t, arael::vect::vect3::new(q.x, q.y, q.z))
            }
        }
    };
}
c_vec2!(CVec2F32, f32);
c_vec2!(CVec2F64, f64);
c_vec3!(CVec3F32, f32);
c_vec3!(CVec3F64, f64);
c_mat2!(CMat2F32, f32);
c_mat2!(CMat2F64, f64);
c_mat3!(CMat3F32, f32);
c_mat3!(CMat3F64, f64);
c_quat!(CQuatF32, f32);
c_quat!(CQuatF64, f64);
"#;

enum Flavor {
    Refs,
    Std,
}

fn vec_flavor(f: &Field) -> Result<Flavor, String> {
    let sp = f.spelled.as_deref().unwrap_or("");
    if sp.starts_with("refs::Vec<") || sp.contains("::refs::Vec<") {
        Ok(Flavor::Refs)
    } else if sp.starts_with("std::vec::Vec<") {
        Ok(Flavor::Std)
    } else {
        Err(format!(
            "collection `{}` spelled `{}`: spell it `refs::Vec<..>` or `std::vec::Vec<..>` \
             so the generator knows the container flavor",
            f.name, sp))
    }
}

fn unsupported(type_name: &str, f: &Field) -> String {
    format!("`{}.{}` kind `{}` (of {:?}) is not supported by cargo-arael yet",
        type_name, f.name, f.kind, f.of)
}

/// Simple get/set pair.
fn rw(out: &mut String, prefix: &str, name: &str, ptr_ty: &str, access: &str,
      c_ty: &str, get_expr: &str, set_stmt: &str) {
    out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {prefix}_{name}(p: *const {ptr_ty}) -> {c_ty} {{
    {get_expr}
}}
#[no_mangle]
pub unsafe extern \"C\" fn {prefix}_set_{name}(p: *mut {ptr_ty}, v: {c_ty}) {{
    {set_stmt}
}}
"));
    let _ = access;
}

/// One collection's function family; `owner` is the pointer type the
/// functions take, `access` the expression reaching the model struct.
fn collection_fns(
    out: &mut String,
    fn_prefix: &str,
    owner: &str,
    access: &str,
    type_name: &str,
    f: &Field,
) -> Result<(), String> {
    let field = &f.name;
    let elem = f.of.as_deref().ok_or("collection without element")?;
    let container = f.container.as_deref().unwrap_or("vec");
    out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_len(p: *const {owner}) -> u32 {{
    {access}.{field}.len() as u32
}}
"));
    let ref_get = format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_get(p: *mut {owner}, r: u32) -> *mut {elem} {{
    let m = &mut {access}.{field};
    &mut m[arael::refs::Ref::from_raw(r)] as *mut {elem}
}}
");
    let refs_extras = |out: &mut String| {
        out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_ref_at(p: *const {owner}, i: u32) -> u32 {{
    {access}.{field}.ref_at(i as usize).to_raw()
}}
"));
        out.push_str(&ref_get);
    };
    match container {
        "vec" => {
            let push_body = match vec_flavor(f)? {
                Flavor::Refs => format!(
"    let m = &mut {access}.{field};
    let r = m.push(Default::default());
    &mut m[r] as *mut {elem}"),
                Flavor::Std => format!(
"    let m = &mut {access}.{field};
    m.push(Default::default());
    m.last_mut().unwrap() as *mut {elem}"),
            };
            out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_push(p: *mut {owner}) -> *mut {elem} {{
{push_body}
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_at(p: *mut {owner}, i: u32) -> *mut {elem} {{
    let m = &mut {access}.{field};
    &mut m[i as usize] as *mut {elem}
}}
"));
            if matches!(vec_flavor(f)?, Flavor::Refs) {
                refs_extras(out);
            }
        }
        "deque" => {
            for method in ["push_back", "push_front"] {
                out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_{method}(p: *mut {owner}) -> *mut {elem} {{
    let m = &mut {access}.{field};
    let r = m.{method}(Default::default());
    &mut m[r] as *mut {elem}
}}
"));
            }
            out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_at(p: *mut {owner}, i: u32) -> *mut {elem} {{
    let m = &mut {access}.{field};
    &mut m[i as usize] as *mut {elem}
}}
"));
            refs_extras(out);
        }
        "arena" => {
            out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_push(p: *mut {owner}) -> u32 {{
    let m = &mut {access}.{field};
    m.push(Default::default()).to_raw()
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_remove(p: *mut {owner}, r: u32) -> bool {{
    let m = &mut {access}.{field};
    m.remove(arael::refs::Ref::from_raw(r)).is_some()
}}
"));
            // No ref_at: an arena has holes, so index-based refs are
            // meaningless -- refs come from push().
            out.push_str(&ref_get);
        }
        other => return Err(format!("`{type_name}.{field}`: unknown container `{other}`")),
    }
    Ok(())
}

/// All accessor functions for one field, over `access` (an expression
/// reaching the OWNING struct from `p: *{const,mut} {ptr_ty}`).
fn field_accessors(
    out: &mut String,
    model: &Model,
    fn_prefix: &str,
    ptr_ty: &str,
    access: &str,
    type_name: &str,
    f: &Field,
) -> Result<(), String> {
    let name = &f.name;
    let of = f.of.as_deref().unwrap_or("");
    match f.kind.as_str() {
        "data" => {
            if let Some(c) = scalar_c(of) {
                rw(out, fn_prefix, name, ptr_ty, access, c,
                    &format!("{access}.{name}"),
                    &format!("{access}.{name} = v;"));
            } else if let Some(m) = math_mirror(of) {
                rw(out, fn_prefix, name, ptr_ty, access, m,
                    &format!("{access}.{name}.into()"),
                    &format!("{access}.{name} = v.into();"));
            } else {
                return Err(unsupported(type_name, f));
            }
        }
        "param" => {
            if let Some(c) = scalar_c(of) {
                rw(out, fn_prefix, name, ptr_ty, access, c,
                    &format!("{access}.{name}.value"),
                    &format!("{access}.{name}.value = v;"));
            } else if let Some(m) = math_mirror(of) {
                rw(out, fn_prefix, name, ptr_ty, access, m,
                    &format!("{access}.{name}.value.into()"),
                    &format!("{access}.{name}.value = v.into();"));
            } else {
                return Err(unsupported(type_name, f));
            }
            rw(out, &format!("{fn_prefix}_{name}"), "optimize", ptr_ty, access, "bool",
                &format!("{access}.{name}.optimize"),
                &format!("{access}.{name}.optimize = v;"));
        }
        "euler_param" => {
            let scalar = f.scalar.as_deref().unwrap_or("f64");
            let variant = f.variant.as_deref().unwrap_or("simple");
            let m = match (variant, scalar) {
                ("rotvec", "f32") => "CQuatF32",
                ("rotvec", _) => "CQuatF64",
                (_, "f32") => "CVec3F32",
                (_, _) => "CVec3F64",
            };
            // `.value` is the documented set-before / read-after field
            // on every rotation param.
            rw(out, fn_prefix, name, ptr_ty, access, m,
                &format!("{access}.{name}.value.into()"),
                &format!("{access}.{name}.value = v.into();"));
            rw(out, &format!("{fn_prefix}_{name}"), "optimize", ptr_ty, access, "bool",
                &format!("{access}.{name}.optimize"),
                &format!("{access}.{name}.optimize = v;"));
        }
        "component" => match of {
            "TransformParam" | "TransformParamF" => {
                let (v3, q) = if of == "TransformParamF" {
                    ("CVec3F32", "CQuatF32")
                } else {
                    ("CVec3F64", "CQuatF64")
                };
                rw(out, &format!("{fn_prefix}_{name}"), "translation", ptr_ty, access, v3,
                    &format!("{access}.{name}.translation.into()"),
                    &format!("{access}.{name}.translation = v.into();"));
                rw(out, &format!("{fn_prefix}_{name}"), "rotation", ptr_ty, access, q,
                    &format!("{access}.{name}.rotation.into()"),
                    &format!("{access}.{name}.rotation = v.into();"));
                for flag in ["optimize_translation", "optimize_rotation"] {
                    rw(out, &format!("{fn_prefix}_{name}"), flag, ptr_ty, access, "bool",
                        &format!("{access}.{name}.{flag}"),
                        &format!("{access}.{name}.{flag} = v;"));
                }
            }
            "UnitVecParam" | "UnitVecParamF" => {
                let v3 = if of == "UnitVecParamF" { "CVec3F32" } else { "CVec3F64" };
                rw(out, &format!("{fn_prefix}_{name}"), "unit", ptr_ty, access, v3,
                    &format!("{access}.{name}.unit.into()"),
                    &format!("{access}.{name}.unit = v.into();"));
            }
            _ => {
                // A user component: expose a pointer, its own surface is
                // emitted like an entity's.
                if !model.types.contains_key(of) {
                    return Err(unsupported(type_name, f));
                }
                out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_{name}_ptr(p: *mut {ptr_ty}) -> *mut {of} {{
    let a = &mut {access}.{name};
    a as *mut {of}
}}
"));
            }
        },
        "struct" => {
            if !model.types.contains_key(of) {
                return Err(unsupported(type_name, f));
            }
            out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_{name}_ptr(p: *mut {ptr_ty}) -> *mut {of} {{
    let a = &mut {access}.{name};
    a as *mut {of}
}}
"));
        }
        "optional" => {
            if !model.types.contains_key(of) {
                return Err(unsupported(type_name, f));
            }
            out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_has_{name}(p: *const {ptr_ty}) -> bool {{
    {access}.{name}.is_some()
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_make_{name}(p: *mut {ptr_ty}) -> *mut {of} {{
    let a = &mut {access}.{name};
    *a = Some(Default::default());
    a.as_mut().unwrap() as *mut {of}
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_clear_{name}(p: *mut {ptr_ty}) {{
    {access}.{name} = None;
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_{name}(p: *mut {ptr_ty}) -> *mut {of} {{
    match &mut {access}.{name} {{
        Some(e) => e as *mut {of},
        None => std::ptr::null_mut(),
    }}
}}
"));
        }
        "ref" => {
            rw(out, fn_prefix, name, ptr_ty, access, "u32",
                &format!("{access}.{name}.to_raw()"),
                &format!("{access}.{name} = arael::refs::Ref::from_raw(v);"));
        }
        "collection" => {
            collection_fns(out, &format!("{fn_prefix}_{name}"), ptr_ty, access, type_name, f)?;
        }
        "self_block" | "cross_block" | "triplet_block" | "skip" | "opaque" => {}
        _ => return Err(unsupported(type_name, f)),
    }
    Ok(())
}

/// Non-root, non-builtin types whose field surface gets emitted
/// (entities, user components, nested sub-models).
pub fn surfaced_types(model: &Model) -> Vec<(&String, &Type)> {
    model.types.iter()
        .filter(|(tn, t)| *tn != &model.root && !t.builtin
            && matches!(t.role.as_str(), "entity" | "component"))
        .collect()
}

pub fn emit(model: &Model, model_crate: &str) -> Result<String, String> {
    let root = &model.root;
    let root_sn = snake(root);
    let fp = &model.precision;
    let handle = format!("{root}Handle");

    let mut used: Vec<&str> = vec![root.as_str()];
    for (tn, _) in surfaced_types(model) {
        used.push(tn.as_str());
    }
    used.sort();
    used.dedup();

    let mut out = String::new();
    out.push_str(&format!(
"// GENERATED by cargo-arael from the `{root}` model sidecar. Do not edit;
// regenerate with `cargo arael export` (check drift with `cargo arael check`).
#![allow(clippy::missing_safety_doc)]

use std::ffi::CString;
use std::os::raw::c_char;
use std::panic::{{AssertUnwindSafe, catch_unwind}};
use arael::simple_lm::{{LmConfig, LmProblem, LmStatus}};
use {model_crate}::{{{}}};

/// The opaque handle the C ABI hands out: the model plus the error /
/// diagnostic text buffer `last_error` points into.
pub struct {handle} {{
    model: {root},
    text: CString,
}}
{MIRRORS}
fn status_code(s: &LmStatus) -> i32 {{
    match s {{
        LmStatus::Converged => 0,
        LmStatus::CostThreshold => 1,
        LmStatus::MaxIterations => 2,
        LmStatus::GradientTolerance => 3,
        LmStatus::ParameterTolerance => 4,
        LmStatus::PredictedReduction => 5,
        LmStatus::LambdaCeiling => 6,
        LmStatus::DriverTerminated => 7,
        LmStatus::ObserverTerminated => 8,
        LmStatus::TimeLimit => 9,
        LmStatus::RetryBudgetExhausted => 10,
        LmStatus::Aborted => 11,
    }}
}}

fn set_text(h: &mut {handle}, msg: &str) {{
    h.text = CString::new(msg.replace('\\0', \" \")).unwrap();
}}

fn panic_text(p: Box<dyn std::any::Any + Send>) -> String {{
    if let Some(s) = p.downcast_ref::<&str>() {{
        (*s).to_string()
    }} else if let Some(s) = p.downcast_ref::<String>() {{
        s.clone()
    }} else {{
        \"panic\".to_string()
    }}
}}

/// Sentinel-based config: UINT32_MAX / NaN fields keep the preset's
/// value (0 = defaults, 1 = conservative, 2 = well_conditioned).
#[repr(C)]
pub struct CLmConfig {{
    pub preset: u32,
    pub max_iters: u32,
    pub min_iters: u32,
    pub patience: u32,
    pub abs_precision: {fp},
    pub rel_precision: {fp},
    pub initial_lambda: {fp},
    pub cost_threshold: {fp},
}}

impl CLmConfig {{
    fn to_config(&self) -> LmConfig<{fp}> {{
        let mut c: LmConfig<{fp}> = match self.preset {{
            1 => LmConfig::conservative(),
            2 => LmConfig::well_conditioned(),
            _ => LmConfig::default(),
        }};
        if self.max_iters != u32::MAX {{ c.max_iters = self.max_iters as usize; }}
        if self.min_iters != u32::MAX {{ c.min_iters = self.min_iters as usize; }}
        if self.patience != u32::MAX {{ c.patience = self.patience as usize; }}
        if !self.abs_precision.is_nan() {{ c.abs_precision = self.abs_precision; }}
        if !self.rel_precision.is_nan() {{ c.rel_precision = self.rel_precision; }}
        if !self.initial_lambda.is_nan() {{ c.initial_lambda = self.initial_lambda; }}
        if !self.cost_threshold.is_nan() {{ c.cost_threshold = self.cost_threshold; }}
        c
    }}
}}

#[repr(C)]
pub struct CLmResult {{
    pub start_cost: {fp},
    pub end_cost: {fp},
    pub iterations: u32,
    pub accepted_iterations: u32,
    pub status: i32,
    pub lambda: {fp},
}}

#[no_mangle]
pub extern \"C\" fn {root_sn}_new() -> *mut {handle} {{
    Box::into_raw(Box::new({handle} {{
        model: Default::default(),
        text: CString::default(),
    }}))
}}

#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_free(h: *mut {handle}) {{
    if !h.is_null() {{
        drop(Box::from_raw(h));
    }}
}}

#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_last_error(h: *const {handle}) -> *const c_char {{
    (*h).text.as_ptr()
}}

/// Empty string when the model is clean, the Diagnostic text otherwise.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_validate(h: *mut {handle}) -> *const c_char {{
    let hh = &mut *h;
    let text = match catch_unwind(AssertUnwindSafe(|| {{
        let d = hh.model.validate();
        if d.is_clean() {{ String::new() }} else {{ d.to_string() }}
    }})) {{
        Ok(t) => t,
        Err(p) => panic_text(p),
    }};
    set_text(hh, &text);
    hh.text.as_ptr()
}}
",
        used.join(", ")));

    for method in ["solve_dense", "solve_sparse"] {
        out.push_str(&format!(
"
/// Returns the status code (>= 0: LmStatus; -1: solve failure; -2:
/// panic). Failure text via {root_sn}_last_error.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_{method}(
    h: *mut {handle},
    cfg: *const CLmConfig,
    out: *mut CLmResult,
) -> i32 {{
    let hh = &mut *h;
    let c = (*cfg).to_config();
    *out = CLmResult {{
        start_cost: 0.0, end_cost: 0.0, iterations: 0,
        accepted_iterations: 0, status: -1, lambda: 0.0,
    }};
    match catch_unwind(AssertUnwindSafe(|| hh.model.{method}(&c))) {{
        Ok(Ok(r)) => {{
            let code = status_code(&r.status);
            *out = CLmResult {{
                start_cost: r.start_cost,
                end_cost: r.end_cost,
                iterations: r.iterations as u32,
                accepted_iterations: r.accepted_iterations as u32,
                status: code,
                lambda: r.final_lambda,
            }};
            set_text(hh, \"\");
            code
        }}
        Ok(Err(f)) => {{
            set_text(hh, &format!(\"solve failure: {{:?}}\", f.kind));
            (*out).status = -1;
            -1
        }}
        Err(p) => {{
            let msg = panic_text(p);
            set_text(hh, &msg);
            (*out).status = -2;
            -2
        }}
    }}
}}
"));
    }

    // Root fields through the handle.
    let root_ty = model.types.get(root).ok_or("root type missing from sidecar")?;
    for f in &root_ty.fields {
        field_accessors(&mut out, model, &root_sn, &handle, "(*p).model", root, f)?;
    }

    // Every surfaced type's fields over raw pointers.
    for (tn, t) in surfaced_types(model) {
        let sn = snake(tn);
        for f in &t.fields {
            field_accessors(&mut out, model, &sn, tn, "(*p)", tn, f)?;
        }
    }

    Ok(out)
}
