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
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_reserve(p: *mut {owner}, additional: u32) {{
    {access}.{field}.reserve(additional as usize);
}}
"));
    let ref_get = format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_get(p: *mut {owner}, r: u32) -> *mut {elem} {{
    let m = &mut {access}.{field};
    &mut m[arael::refs::Ref::from_raw(r)] as *mut {elem}
}}
/// True while `r` addresses a live element of this collection.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_contains(p: *const {owner}, r: u32) -> bool {{
    {access}.{field}.contains_ref(arael::refs::Ref::from_raw(r))
}}
/// Like get, but null for a stale or foreign ref instead of a panic.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_try_get(p: *mut {owner}, r: u32) -> *mut {elem} {{
    let m = &mut {access}.{field};
    match m.get_mut(arael::refs::Ref::from_raw(r)) {{
        Some(e) => e as *mut {elem},
        None => std::ptr::null_mut(),
    }}
}}
");
    let refs_extras = |out: &mut String, first_name: &str, last_name: &str| {
        out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_ref_at(p: *const {owner}, i: u32) -> u32 {{
    {access}.{field}.ref_at(i as usize).to_raw()
}}
/// Ref of the first/last element, or u32::MAX when empty.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_{first_name}(p: *const {owner}) -> u32 {{
    match {access}.{field}.{first_name}() {{
        Some(r) => r.to_raw(),
        None => u32::MAX,
    }}
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_{last_name}(p: *const {owner}) -> u32 {{
    match {access}.{field}.{last_name}() {{
        Some(r) => r.to_raw(),
        None => u32::MAX,
    }}
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
/// Drops the last element; false when already empty.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_pop(p: *mut {owner}) -> bool {{
    {access}.{field}.pop().is_some()
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_clear(p: *mut {owner}) {{
    {access}.{field}.clear();
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_truncate(p: *mut {owner}, len: u32) {{
    {access}.{field}.truncate(len as usize);
}}
"));
            if matches!(vec_flavor(f)?, Flavor::Refs) {
                refs_extras(out, "first_ref", "last_ref");
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
/// Drops the back element; false when already empty.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_pop_back(p: *mut {owner}) -> bool {{
    {access}.{field}.pop_back().is_some()
}}
/// Drops the front element; false when already empty.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_pop_front(p: *mut {owner}) -> bool {{
    {access}.{field}.pop_front().is_some()
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_clear(p: *mut {owner}) {{
    {access}.{field}.clear();
}}
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_truncate(p: *mut {owner}, len: u32) {{
    {access}.{field}.truncate(len as usize);
}}
"));
            refs_extras(out, "front_ref", "back_ref");
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
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_clear(p: *mut {owner}) {{
    {access}.{field}.clear();
}}
"));
            // No ref_at: an arena has holes, so index-based refs are
            // meaningless -- refs come from push() or the cursor.
            out.push_str(&ref_get);
            out.push_str(&format!(
"/// First live element's packed ref, or u32::MAX when empty.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_first(p: *const {owner}) -> u32 {{
    match {access}.{field}.first_ref() {{
        Some(r) => r.to_raw(),
        None => u32::MAX,
    }}
}}
/// The next live element after `r`'s slot, or u32::MAX past the end.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_next(p: *const {owner}, r: u32) -> u32 {{
    match {access}.{field}.next_ref(arael::refs::Ref::from_raw(r)) {{
        Some(n) => n.to_raw(),
        None => u32::MAX,
    }}
}}
/// Last live element's packed ref, or u32::MAX when empty.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_last(p: *const {owner}) -> u32 {{
    match {access}.{field}.last_ref() {{
        Some(r) => r.to_raw(),
        None => u32::MAX,
    }}
}}
/// The previous live element before `r`'s slot, or u32::MAX past the front.
#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_prev(p: *const {owner}, r: u32) -> u32 {{
    match {access}.{field}.prev_ref(arael::refs::Ref::from_raw(r)) {{
        Some(n) => n.to_raw(),
        None => u32::MAX,
    }}
}}
"));
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
                // The chart tangent basis (read-only): d unit / d chart,
                // one column per chart parameter.
                for i in 0..2 {
                    out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_{name}_unit_d{i}(p: *const {ptr_ty}) -> {v3} {{
    {access}.{name}.unit_d[{i}].into()
}}
"));
                }
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
use arael::covariance::{{CovAssembly, CovMode, Covariance}};
use arael::simple_lm::{{LmConfig, LmProblem, LmStatus}};
use {model_crate}::{{{}}};

/// The opaque handle the C ABI hands out: the model, the error /
/// diagnostic text buffer `last_error` points into, and the covariance
/// assembly once requested.
pub struct {handle} {{
    model: {root},
    text: CString,
    cov: Option<CovAssembly>,
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

/// C mirror of `arael::option<T>` ({{has, value}}; layouts must match).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct COptF {{
    pub has: bool,
    pub v: {fp},
}}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct COptSeconds {{
    pub has: bool,
    pub v: f64,
}}

fn copt(o: Option<{fp}>) -> COptF {{
    match o {{
        Some(v) => COptF {{ has: true, v }},
        None => COptF {{ has: false, v: 0.0 }},
    }}
}}

fn opt_of(c: COptF) -> Option<{fp}> {{
    c.has.then_some(c.v)
}}

/// The solver config as REAL values: constructed by
/// {root_sn}_lm_config (which copies them out of the actual Rust
/// LmConfig preset), edited freely, passed back whole. The preset tag
/// stays the base for the Rust fields this struct does not expose
/// (lambda driver, observer, gather_timing). Field order is C ABI.
#[repr(C)]
pub struct CLmConfig {{
    pub preset: u32,
    pub max_iters: u32,
    pub min_iters: u32,
    pub patience: u32,
    pub num_threads: u32,
    pub verbose: bool,
    pub abs_precision: {fp},
    pub rel_precision: {fp},
    pub initial_lambda: {fp},
    pub cost_threshold: {fp},
    pub lambda_floor: {fp},
    pub gradient_tolerance: COptF,
    pub parameter_tolerance: COptF,
    pub predicted_reduction_tolerance: COptF,
    pub min_diagonal: COptF,
    pub time_limit_seconds: COptSeconds,
}}

fn preset_config(preset: u32) -> LmConfig<{fp}> {{
    match preset {{
        1 => LmConfig::conservative(),
        2 => LmConfig::well_conditioned(),
        _ => LmConfig::default(),
    }}
}}

/// Fill `out` with the preset's actual Rust values (0 = defaults,
/// 1 = conservative, 2 = well_conditioned).
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_lm_config(preset: u32, out: *mut CLmConfig) {{
    let c = preset_config(preset);
    *out = CLmConfig {{
        preset,
        max_iters: c.max_iters as u32,
        min_iters: c.min_iters as u32,
        patience: c.patience as u32,
        num_threads: c.num_threads as u32,
        verbose: c.verbose,
        abs_precision: c.abs_precision,
        rel_precision: c.rel_precision,
        initial_lambda: c.initial_lambda,
        cost_threshold: c.cost_threshold,
        lambda_floor: c.lambda_floor,
        gradient_tolerance: copt(c.gradient_tolerance),
        parameter_tolerance: copt(c.parameter_tolerance),
        predicted_reduction_tolerance: copt(c.predicted_reduction_tolerance),
        min_diagonal: copt(c.min_diagonal),
        time_limit_seconds: match c.time_limit {{
            Some(d) => COptSeconds {{ has: true, v: d.as_secs_f64() }},
            None => COptSeconds {{ has: false, v: 0.0 }},
        }},
    }};
}}

impl CLmConfig {{
    fn to_config(&self) -> LmConfig<{fp}> {{
        let mut c = preset_config(self.preset);
        c.max_iters = self.max_iters as usize;
        c.min_iters = self.min_iters as usize;
        c.patience = self.patience as usize;
        c.num_threads = self.num_threads as usize;
        c.verbose = self.verbose;
        c.abs_precision = self.abs_precision;
        c.rel_precision = self.rel_precision;
        c.initial_lambda = self.initial_lambda;
        c.cost_threshold = self.cost_threshold;
        c.lambda_floor = self.lambda_floor;
        c.gradient_tolerance = opt_of(self.gradient_tolerance);
        c.parameter_tolerance = opt_of(self.parameter_tolerance);
        c.predicted_reduction_tolerance = opt_of(self.predicted_reduction_tolerance);
        c.min_diagonal = opt_of(self.min_diagonal);
        c.time_limit = self.time_limit_seconds.has.then(|| {{
            std::time::Duration::from_secs_f64(self.time_limit_seconds.v)
        }});
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
    pub final_lambda: {fp},
}}

#[no_mangle]
pub extern \"C\" fn {root_sn}_new() -> *mut {handle} {{
    Box::into_raw(Box::new({handle} {{
        model: Default::default(),
        text: CString::default(),
        cov: None,
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
        accepted_iterations: 0, status: -1, final_lambda: 0.0,
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
                final_lambda: r.final_lambda,
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

    // Cost evaluation at the current parameter values.
    let (ser, cast) = if fp == "f32" {
        ("serialize32", " as f64")
    } else {
        ("serialize64", "")
    };
    out.push_str(&format!(
"
/// Total cost at the current parameter values (evaluated at the
/// model's precision, returned as f64).
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_cost(h: *mut {handle}) -> f64 {{
    let m = &mut (*h).model;
    let mut params = Vec::new();
    m.{ser}(&mut params);
    m.calc_cost(&params){cast}
}}
"));

    // Covariance: assemble on the handle, then per-entity marginals.
    out.push_str(&format!(
"
/// mode: 0 = PerQuery, 1 = AllMarginals, 2 = TriDiagonal. Returns 0,
/// -1 (error, text via {root_sn}_last_error), -2 (panic).
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_assemble_covariance(h: *mut {handle}, mode: u32) -> i32 {{
    let hh = &mut *h;
    let m = match mode {{
        0 => CovMode::PerQuery,
        2 => CovMode::TriDiagonal,
        _ => CovMode::AllMarginals,
    }};
    hh.cov = None;
    match catch_unwind(AssertUnwindSafe(|| hh.model.assemble_covariance(m))) {{
        Ok(Ok(c)) => {{
            hh.cov = Some(c);
            set_text(hh, \"\");
            0
        }}
        Ok(Err(e)) => {{
            set_text(hh, &format!(\"{{}}\", e));
            -1
        }}
        Err(p) => {{
            let msg = panic_text(p);
            set_text(hh, &msg);
            -2
        }}
    }}
}}
"));
    for (tn, t) in surfaced_types(model) {
        if t.role != "entity" || t.param_count == 0 {
            continue;
        }
        let sn = format!("{root_sn}_{}", snake(tn));
        out.push_str(&format!(
"
/// Row-major dim x dim marginal covariance (f64) of one `{tn}`; returns
/// dim, or -1 (error) / -2 (panic) / -3 (no assembly or buffer too
/// small), text via {root_sn}_last_error.
#[no_mangle]
pub unsafe extern \"C\" fn {sn}_marginal_cov(
    h: *mut {handle},
    p: *const {tn},
    out: *mut f64,
    cap: u32,
) -> i32 {{
    let hh = &mut *h;
    let Some(cov) = hh.cov.as_ref() else {{
        set_text(hh, \"marginal_cov: assemble_covariance was not called\");
        return -3;
    }};
    match catch_unwind(AssertUnwindSafe(|| cov.marginal_cov(&*p))) {{
        Ok(Ok(m)) => {{
            let dim = m.nrows();
            if (dim * dim) as u32 > cap {{
                set_text(hh, \"marginal_cov: buffer too small\");
                return -3;
            }}
            for r in 0..dim {{
                for c in 0..dim {{
                    *out.add(r * dim + c) = m[(r, c)];
                }}
            }}
            dim as i32
        }}
        Ok(Err(e)) => {{
            set_text(hh, &format!(\"{{}}\", e));
            -1
        }}
        Err(p2) => {{
            let msg = panic_text(p2);
            set_text(hh, &msg);
            -2
        }}
    }}
}}
"));
    }

    // Cross-covariance for every ordered pair of parameterized entities.
    let cov_entities: Vec<(&String, &crate::ir::Type)> = surfaced_types(model)
        .into_iter()
        .filter(|(_, t)| t.role == "entity" && t.param_count > 0)
        .collect();
    for (an, _) in &cov_entities {
        for (bn, _) in &cov_entities {
            let a_sn = snake(an);
            let b_sn = snake(bn);
            out.push_str(&format!(
"
/// Row-major pa x pb cross-covariance (f64) between a `{an}` and a
/// `{bn}`; returns the row count, or -1 (error) / -2 (panic) / -3 (no
/// assembly or buffer too small), text via {root_sn}_last_error.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_{a_sn}_{b_sn}_cross_cov(
    h: *mut {handle},
    a: *const {an},
    b: *const {bn},
    out: *mut f64,
    cap: u32,
) -> i32 {{
    let hh = &mut *h;
    let Some(cov) = hh.cov.as_ref() else {{
        set_text(hh, \"cross_cov: assemble_covariance was not called\");
        return -3;
    }};
    match catch_unwind(AssertUnwindSafe(|| cov.cross_cov(&*a, &*b))) {{
        Ok(Ok(m)) => {{
            let (rows, cols) = (m.nrows(), m.ncols());
            if (rows * cols) as u32 > cap {{
                set_text(hh, \"cross_cov: buffer too small\");
                return -3;
            }}
            for r in 0..rows {{
                for c in 0..cols {{
                    *out.add(r * cols + c) = m[(r, c)];
                }}
            }}
            rows as i32
        }}
        Ok(Err(e)) => {{
            set_text(hh, &format!(\"{{}}\", e));
            -1
        }}
        Err(p2) => {{
            let msg = panic_text(p2);
            set_text(hh, &msg);
            -2
        }}
    }}
}}
"));
        }
    }

    // Root fields through the handle.
    let root_ty = model.types.get(root).ok_or("root type missing from sidecar")?;
    for f in &root_ty.fields {
        field_accessors(&mut out, model, &root_sn, &handle, "(*p).model", root, f)?;
    }

    // Every surfaced type's fields over raw pointers. Symbols carry
    // the root prefix so several generated models link into one binary
    // without collisions.
    for (tn, t) in surfaced_types(model) {
        let sn = format!("{root_sn}_{}", snake(tn));
        for f in &t.fields {
            field_accessors(&mut out, model, &sn, tn, "(*p)", tn, f)?;
        }
    }

    Ok(out)
}
