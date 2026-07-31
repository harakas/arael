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
            "AngleParam" | "AngleParamF" => {
                let (sc, m) = if of == "AngleParamF" {
                    ("f32", "CMat2F32")
                } else {
                    ("f64", "CMat2F64")
                };
                // `angle` is the set-before / read-after scalar, plus its flag.
                rw(out, &format!("{fn_prefix}_{name}"), "angle", ptr_ty, access, sc,
                    &format!("{access}.{name}.angle.value"),
                    &format!("{access}.{name}.angle.value = v;"));
                rw(out, &format!("{fn_prefix}_{name}_angle"), "optimize", ptr_ty, access, "bool",
                    &format!("{access}.{name}.angle.optimize"),
                    &format!("{access}.{name}.angle.optimize = v;"));
                // Rotation matrix at the current angle (read-only, computed).
                out.push_str(&format!(
"#[no_mangle]
pub unsafe extern \"C\" fn {fn_prefix}_{name}_rotation_matrix(p: *const {ptr_ty}) -> {m} {{
    {access}.{name}.rotation_matrix().into()
}}
"));
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

/// The whole single-root shim file: header + body.
pub fn emit(model: &Model, model_crate: &str) -> Result<String, String> {
    let root = &model.root;
    Ok(format!(
"// GENERATED by cargo-arael from the `{root}` model sidecar. Do not edit;
// regenerate with `cargo arael export` (check drift with `cargo arael check`).
#![allow(clippy::missing_safety_doc)]

{}", emit_body(model, model_crate)?))
}

/// One root's self-contained shim (use statements through accessors,
/// no file header) -- a multi-root capi wraps one per root in a
/// module; symbols carry the root prefix, so they coexist in one lib.
pub fn emit_body(model: &Model, model_crate: &str) -> Result<String, String> {
    let root = &model.root;
    let root_sn = snake(root);
    let fp = &model.precision;
    let handle = format!("{root}Handle");
    // The cost-table surface exists only for `#[arael(root, jacobian)]`
    // roots (the sidecar's `jacobian` flag).
    let ct_field = if model.jacobian {
        "\n    cost_table: Vec<(CString, f64)>,"
    } else {
        ""
    };
    let ct_init = if model.jacobian {
        "\n        cost_table: Vec::new(),"
    } else {
        ""
    };

    let mut used: Vec<&str> = vec![root.as_str()];
    for (tn, _) in surfaced_types(model) {
        used.push(tn.as_str());
    }
    used.sort();
    used.dedup();

    let mut out = String::new();
    out.push_str(&format!(
"use std::ffi::CString;
use std::os::raw::c_char;
use std::panic::{{AssertUnwindSafe, catch_unwind}};
use arael::covariance::{{CovAssembly, CovMode, Covariance}};
use arael::simple_lm::{{
    LmConfig, LmProblem, LmSession, LmStatus, SparseFaer, SparseFaerOptions,
}};
use {model_crate}::{{{}}};

/// The opaque handle the C ABI hands out: the model, the error /
/// diagnostic text buffer `last_error` points into, and the
/// covariance assembly once requested.
pub struct {handle} {{
    model: {root},
    text: CString,
    cov: Option<CovAssembly>,{ct_field}
}}

/// The Rust side of a solve result, boxed behind `CLmResult::detail`:
/// the full `LmResult` (report text, backend plan, per-step timing)
/// plus the buffer `{root_sn}_result_report` renders into. Owned by
/// the caller; released with `{root_sn}_result_free`.
pub struct ResultDetail {{
    result: arael::simple_lm::LmResult<{fp}>,
    buf: CString,
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

/// One damped attempt, as the observer callback sees it. Field order
/// is C ABI. `params` points at the CURRENT parameter vector for this
/// attempt; valid only during the callback.
#[repr(C)]
pub struct CLmIter {{
    pub iter: u32,
    pub inner: u32,
    pub accepted: bool,
    pub factorization_failed: bool,
    pub cost: {fp},
    pub new_cost: {fp},
    pub lambda: {fp},
    pub accepted_total: u32,
    pub params: *const {fp},
    pub params_len: u32,
}}

/// The C observer: called once per damped attempt; return false to
/// stop the solve (status ObserverTerminated).
type CObserverFn = extern \"C\" fn(*mut core::ffi::c_void, *const CLmIter) -> bool;

#[derive(Clone)]
struct CObserver {{
    f: CObserverFn,
    user: *mut core::ffi::c_void,
}}

impl arael::simple_lm::LmObserver<{fp}> for CObserver {{
    fn on_iteration(
        &mut self,
        it: &arael::simple_lm::LmIter<'_, {fp}>,
    ) -> std::ops::ControlFlow<()> {{
        let c = CLmIter {{
            iter: it.iter as u32,
            inner: it.inner as u32,
            accepted: it.accepted,
            factorization_failed: it.factorization_failed,
            cost: it.cost,
            new_cost: it.new_cost,
            lambda: it.lambda,
            accepted_total: it.accepted_total as u32,
            params: it.params.as_ptr(),
            params_len: it.params.len() as u32,
        }};
        if (self.f)(self.user, &c) {{
            std::ops::ControlFlow::Continue(())
        }} else {{
            std::ops::ControlFlow::Break(())
        }}
    }}
}}

/// The solver config as REAL values: constructed by
/// {root_sn}_lm_config (which copies them out of the actual Rust
/// LmConfig preset), edited freely, passed back whole. The preset tag
/// stays the base for the Rust fields this struct does not expose
/// (lambda driver). Field order is C ABI.
#[repr(C)]
pub struct CLmConfig {{
    pub preset: u32,
    pub max_iters: u32,
    pub min_iters: u32,
    pub patience: u32,
    pub num_threads: u32,
    pub verbose: bool,
    pub gather_timing: bool,
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
    pub observer: Option<CObserverFn>,
    pub observer_user: *mut core::ffi::c_void,
}}

/// The sparse backend's options as plain data: constructed by
/// {root_sn}_sparse_options (which copies the actual Rust defaults),
/// edited freely, passed to {root_sn}_solve_sparse (null there means
/// these defaults). Field order is C ABI. Not exposed: the
/// marginalize range list (the model's own marginalize hint covers
/// it) and the iterative Schur routes.
#[repr(C)]
pub struct CSparseOptions {{
    /// SchurPolicy: 0 Auto, 1 Force, 2 Never.
    pub schur: u32,
    /// FaerOrdering: 0 Auto, 1 Amd, 2 MarginalizeFirst, 3 Natural,
    /// 4 NestedDissection.
    pub ordering: u32,
    /// EnvelopeMode: 0 Auto, 1 Always, 2 Never.
    pub envelope: u32,
    /// Envelope panel width; 0 picks it automatically.
    pub envelope_panel_width: u32,
    pub supernodal: bool,
    pub narrow_band: bool,
    /// SchurPolicy::Auto tuning: the reduction must beat the whole
    /// system by this flop factor to be taken...
    pub flop_margin: f64,
    /// ...and below this cheap ratio it is taken without the exact
    /// pricing.
    pub obvious_flop_ratio: f64,
    /// Conjugate-gradient tolerance for the iterative routes.
    pub cg_tol: f64,
    /// SchurSolve: 0 Factorize, 1 Iterative, 2 IterativeImplicit.
    /// The iterative routes solve the reduced system by preconditioned
    /// conjugate gradients; pair them with schur = Force (without a
    /// reduction the solve fails rather than taking another route).
    pub schur_solve: u32,
    /// CG iteration cap; 0 = unlimited.
    pub cg_max_iters: u32,
    /// CG restart interval; 0 = never.
    pub cg_restart_every: u32,
}}

impl CSparseOptions {{
    /// An out-of-range enum tag panics; the callers catch it, so it
    /// surfaces as PanicError / AraelError with the tag in the
    /// message. Unreachable through the typed wrappers.
    fn to_options(&self) -> SparseFaerOptions {{
        use arael::simple_lm::{{EnvelopeMode, FaerOrdering, SchurPolicy}};
        let policy = match self.schur {{
            0 => SchurPolicy::Auto {{
                flop_margin: self.flop_margin,
                obvious_flop_ratio: self.obvious_flop_ratio,
            }},
            1 => SchurPolicy::Force,
            2 => SchurPolicy::Never,
            t => panic!(\"unknown schur policy tag {{t}}\"),
        }};
        let ordering = match self.ordering {{
            0 => FaerOrdering::Auto,
            1 => FaerOrdering::Amd,
            2 => FaerOrdering::MarginalizeFirst,
            3 => FaerOrdering::Natural,
            4 => FaerOrdering::NestedDissection,
            t => panic!(\"unknown ordering tag {{t}}\"),
        }};
        let envelope = match self.envelope {{
            0 => EnvelopeMode::Auto,
            1 => EnvelopeMode::Always,
            2 => EnvelopeMode::Never,
            t => panic!(\"unknown envelope mode tag {{t}}\"),
        }};
        let width = self.envelope_panel_width;
        let opts = SparseFaerOptions::auto()
            .with_policy(policy)
            .with_ordering(ordering)
            .with_envelope_schur(envelope)
            .with_envelope_panel_width((width > 0).then_some(width as usize))
            .with_supernodal(self.supernodal)
            .with_narrow_band(self.narrow_band);
        let cg = arael::simple_lm::CgOptions {{
            tol: self.cg_tol,
            max_iters: self.cg_max_iters as usize,
            restart_every: self.cg_restart_every as usize,
        }};
        match self.schur_solve {{
            0 => opts,
            1 => opts.with_iterative_schur(cg),
            2 => opts.with_implicit_schur(cg),
            t => panic!(\"unknown schur solve tag {{t}}\"),
        }}
    }}
}}

/// Fill `out` with the sparse backend's actual Rust defaults.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_sparse_options(out: *mut CSparseOptions) {{
    use arael::simple_lm::{{EnvelopeMode, FaerOrdering, SchurPolicy}};
    let d = SparseFaerOptions::default();
    let (flop_margin, obvious_flop_ratio) = match d.policy {{
        SchurPolicy::Auto {{ flop_margin, obvious_flop_ratio }} => {{
            (flop_margin, obvious_flop_ratio)
        }}
        _ => (0.0, 0.0),
    }};
    *out = CSparseOptions {{
        schur: match d.policy {{
            SchurPolicy::Auto {{ .. }} => 0,
            SchurPolicy::Force => 1,
            SchurPolicy::Never => 2,
        }},
        ordering: match d.ordering {{
            FaerOrdering::Auto => 0,
            FaerOrdering::Amd => 1,
            FaerOrdering::MarginalizeFirst => 2,
            FaerOrdering::Natural => 3,
            FaerOrdering::NestedDissection => 4,
        }},
        envelope: match d.envelope {{
            EnvelopeMode::Auto => 0,
            EnvelopeMode::Always => 1,
            EnvelopeMode::Never => 2,
        }},
        envelope_panel_width: d.envelope_panel_width.unwrap_or(0) as u32,
        supernodal: d.supernodal,
        narrow_band: d.narrow_band,
        flop_margin,
        obvious_flop_ratio,
        cg_tol: arael::simple_lm::CgOptions::default().tol,
        schur_solve: 0,
        cg_max_iters: 0,
        cg_restart_every: 0,
    }};
}}

fn preset_config(preset: u32) -> LmConfig<{fp}> {{
    match preset {{
        1 => LmConfig::conservative(),
        2 => LmConfig::well_conditioned(),
        3 => LmConfig::ill_conditioned(),
        _ => LmConfig::default(),
    }}
}}

/// Fill `out` with the preset's actual Rust values (0 = defaults,
/// 1 = conservative, 2 = well_conditioned, 3 = ill_conditioned).
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
        gather_timing: c.gather_timing,
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
        observer: None,
        observer_user: std::ptr::null_mut(),
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
        c.gather_timing = self.gather_timing;
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
        if let Some(f) = self.observer {{
            c = c.with_observer(CObserver {{ f, user: self.observer_user }});
        }}
        c
    }}
}}

/// LmTiming mirror: per-phase wall-clock seconds plus call counts.
/// The Rust per-step records (LmTiming::steps) stay Rust-side.
#[repr(C)]
#[derive(Default, Clone, Copy)]
pub struct CLmTiming {{
    pub total: f64,
    pub assembly: f64,
    pub first_assembly: f64,
    pub analysis: f64,
    pub linear_solve: f64,
    pub first_linear_solve: f64,
    pub cost_eval: f64,
    pub first_cost_eval: f64,
    pub advance: f64,
    pub first_advance: f64,
    pub assembly_count: u32,
    pub analysis_count: u32,
    pub linear_solve_count: u32,
    pub cost_eval_count: u32,
    pub advance_count: u32,
}}

#[repr(C)]
pub struct CLmResult {{
    pub start_cost: {fp},
    pub end_cost: {fp},
    pub iterations: u32,
    pub accepted_iterations: u32,
    pub status: i32,
    pub final_lambda: {fp},
    /// Valid iff has_timing (config.gather_timing was set).
    pub timing: CLmTiming,
    pub has_timing: bool,
    /// The full Rust result (render it with {root_sn}_result_report,
    /// read its plan with {root_sn}_result_plan). Owned by the caller:
    /// release with {root_sn}_result_free. On a failed solve it holds
    /// the partial result when the solver produced one, else null.
    pub detail: *mut ResultDetail,
}}

/// {{has, v}} mirrors of `arael::option<T>` for the plan's optional
/// statistics (layouts must match the C++ side).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct COptDouble {{
    pub has: bool,
    pub v: f64,
}}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct COptU32 {{
    pub has: bool,
    pub v: u32,
}}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct COptI32 {{
    pub has: bool,
    pub v: i32,
}}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct CRouteFlops {{
    pub reduced: f64,
    pub full: f64,
}}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct COptRouteFlops {{
    pub has: bool,
    pub v: CRouteFlops,
}}

/// Mirror of arael's SchurPlan: what the sparse backend decided.
/// Field order is C ABI and follows the Rust struct.
#[repr(C)]
pub struct CSchurPlan {{
    pub reduced: bool,
    pub eliminated_blocks: u32,
    pub eliminated_params: u32,
    pub kept_params: u32,
    pub fill_ratio: COptDouble,
    pub route_flops: COptRouteFlops,
    pub cg_iterations: COptU32,
    pub flop_ratio: COptDouble,
    /// ReducedOrdering: 0 NaturalBanded, 1 NaturalDense, 2 Amd, 3 Nd.
    pub ordering: COptI32,
    pub kept_bandwidth: u32,
    pub envelope: bool,
}}

unsafe fn fill_plan(out: *mut CSchurPlan, p: &arael::simple_lm::SchurPlan) {{
    use arael::simple_lm::ReducedOrdering;
    let od = |o: Option<f64>| match o {{
        Some(v) => COptDouble {{ has: true, v }},
        None => COptDouble {{ has: false, v: 0.0 }},
    }};
    *out = CSchurPlan {{
        reduced: p.reduced,
        eliminated_blocks: p.eliminated_blocks as u32,
        eliminated_params: p.eliminated_params as u32,
        kept_params: p.kept_params as u32,
        fill_ratio: od(p.fill_ratio),
        route_flops: match p.route_flops {{
            Some((reduced, full)) => COptRouteFlops {{
                has: true,
                v: CRouteFlops {{ reduced, full }},
            }},
            None => COptRouteFlops {{
                has: false,
                v: CRouteFlops {{ reduced: 0.0, full: 0.0 }},
            }},
        }},
        cg_iterations: match p.cg_iterations {{
            Some(n) => COptU32 {{ has: true, v: n as u32 }},
            None => COptU32 {{ has: false, v: 0 }},
        }},
        flop_ratio: od(p.flop_ratio),
        ordering: match p.ordering {{
            Some(o) => COptI32 {{
                has: true,
                v: match o {{
                    ReducedOrdering::NaturalBanded => 0,
                    ReducedOrdering::NaturalDense => 1,
                    ReducedOrdering::Amd => 2,
                    ReducedOrdering::Nd => 3,
                }},
            }},
            None => COptI32 {{ has: false, v: 0 }},
        }},
        kept_bandwidth: p.kept_bandwidth as u32,
        envelope: p.envelope,
    }};
}}

unsafe fn zero_result(out: *mut CLmResult) {{
    *out = CLmResult {{
        start_cost: 0.0, end_cost: 0.0, iterations: 0,
        accepted_iterations: 0, status: -1, final_lambda: 0.0,
        timing: CLmTiming::default(), has_timing: false,
        detail: std::ptr::null_mut(),
    }};
}}

unsafe fn fill_result(out: *mut CLmResult, r: &arael::simple_lm::LmResult<{fp}>) -> i32 {{
    let code = status_code(&r.status);
    let (timing, has_timing) = match &r.timing {{
        Some(t) => (CLmTiming {{
            total: t.total.as_secs_f64(),
            assembly: t.assembly.as_secs_f64(),
            first_assembly: t.first_assembly.as_secs_f64(),
            analysis: t.analysis.as_secs_f64(),
            linear_solve: t.linear_solve.as_secs_f64(),
            first_linear_solve: t.first_linear_solve.as_secs_f64(),
            cost_eval: t.cost_eval.as_secs_f64(),
            first_cost_eval: t.first_cost_eval.as_secs_f64(),
            advance: t.advance.as_secs_f64(),
            first_advance: t.first_advance.as_secs_f64(),
            assembly_count: t.assembly_count as u32,
            analysis_count: t.analysis_count as u32,
            linear_solve_count: t.linear_solve_count as u32,
            cost_eval_count: t.cost_eval_count as u32,
            advance_count: t.advance_count as u32,
        }}, true),
        None => (CLmTiming::default(), false),
    }};
    *out = CLmResult {{
        start_cost: r.start_cost,
        end_cost: r.end_cost,
        iterations: r.iterations as u32,
        accepted_iterations: r.accepted_iterations as u32,
        status: code,
        final_lambda: r.final_lambda,
        timing,
        has_timing,
        detail: std::ptr::null_mut(),
    }};
    code
}}

fn boxed(r: arael::simple_lm::LmResult<{fp}>) -> *mut ResultDetail {{
    Box::into_raw(Box::new(ResultDetail {{ result: r, buf: CString::default() }}))
}}

#[no_mangle]
pub extern \"C\" fn {root_sn}_new() -> *mut {handle} {{
    Box::into_raw(Box::new({handle} {{
        model: Default::default(),
        text: CString::default(),
        cov: None,{ct_init}
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

/// Text report of the result behind `d` (status, cost, iterations,
/// damping, plus the timing breakdown and the backend's plan when it
/// has them). `pretty` adds colour and box-drawing glyphs. The
/// pointer is valid until the next report call on the same result or
/// {root_sn}_result_free.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_result_report(d: *mut ResultDetail, pretty: bool) -> *const c_char {{
    let dd = &mut *d;
    let text = if pretty {{ dd.result.pretty_report() }} else {{ dd.result.report() }};
    dd.buf = CString::new(text.replace('\\0', \" \")).unwrap_or_default();
    dd.buf.as_ptr()
}}

/// The sparse backend's plan for the result behind `d`. Returns false
/// with `out` untouched when the solve carried none (dense and band
/// solves).
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_result_plan(d: *const ResultDetail, out: *mut CSchurPlan) -> bool {{
    match &(*d).result.solver {{
        Some(arael::simple_lm::SolverReport::Schur(p)) => {{
            fill_plan(out, p);
            true
        }}
        _ => false,
    }}
}}

/// Release the Rust result behind a CLmResult. Null is fine.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_result_free(d: *mut ResultDetail) {{
    if !d.is_null() {{
        drop(Box::from_raw(d));
    }}
}}

/// Drop arael log messages above `level` (0 Off, 1 Error, 2 Warn,
/// 3 Info; Info -- everything -- is the default). Process-wide: all
/// models and roots share it.
#[no_mangle]
pub extern \"C\" fn {root_sn}_set_log_level(level: u32) {{
    arael::log::set_level(match level {{
        0 => arael::log::Level::Off,
        1 => arael::log::Level::Error,
        2 => arael::log::Level::Warn,
        _ => arael::log::Level::Info,
    }});
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

    out.push_str(&format!(
"
/// Returns the status code (>= 0: LmStatus; -1: solve failure; -2:
/// panic). Failure text via {root_sn}_last_error. On success (and on
/// a failure that got past its first assembly, where `out` carries
/// the partial result) `out.detail` owns the full Rust result --
/// release it with {root_sn}_result_free.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_solve_dense(
    h: *mut {handle},
    cfg: *const CLmConfig,
    out: *mut CLmResult,
) -> i32 {{
    let hh = &mut *h;
    let c = (*cfg).to_config();
    zero_result(out);
    match catch_unwind(AssertUnwindSafe(|| hh.model.solve_dense(&c))) {{
        Ok(Ok(r)) => {{
            let code = fill_result(out, &r);
            (*out).detail = boxed(r);
            set_text(hh, \"\");
            code
        }}
        Ok(Err(f)) => {{
            set_text(hh, &f.to_string());
            if let Some(p) = f.partial {{
                fill_result(out, &p);
                (*out).status = -1;
                (*out).detail = boxed(*p);
            }}
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

/// As {root_sn}_solve_dense, with the sparse backend. `opts` selects
/// its route (ordering, Schur policy, envelope); null means the
/// defaults ({root_sn}_sparse_options shows them).
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_solve_sparse(
    h: *mut {handle},
    cfg: *const CLmConfig,
    opts: *const CSparseOptions,
    out: *mut CLmResult,
) -> i32 {{
    let hh = &mut *h;
    let c = (*cfg).to_config();
    zero_result(out);
    match catch_unwind(AssertUnwindSafe(|| {{
        if opts.is_null() {{
            hh.model.solve_sparse(&c)
        }} else {{
            let mut s = SparseFaer::<{fp}>::from_options(&(*opts).to_options());
            hh.model.solve_with(&mut s, &c)
        }}
    }})) {{
        Ok(Ok(r)) => {{
            let code = fill_result(out, &r);
            (*out).detail = boxed(r);
            set_text(hh, \"\");
            code
        }}
        Ok(Err(f)) => {{
            set_text(hh, &f.to_string());
            if let Some(p) = f.partial {{
                fill_result(out, &p);
                (*out).status = -1;
                (*out).detail = boxed(*p);
            }}
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

/// A warm-reuse session over the sparse backend (Rust's LmSession):
/// keeps the analysis -- pattern, ordering, symbolic factorization,
/// Schur plan -- across solves, so only the first pays for it. Warm
/// solves are bit-identical to cold ones. A parameter-count change
/// re-analyzes by itself; {root_sn}_session_invalidate covers a
/// structural change at the same count (solving warm through one is
/// undefined).
pub struct {root}Session {{
    // Err carries a construction panic (a bad options tag), reported
    // by the first solve.
    session: Result<LmSession<{fp}, SparseFaer<{fp}>>, String>,
}}

/// New session; `opts` as in {root_sn}_solve_sparse (null =
/// defaults). Never fails: a bad options tag is reported by the
/// first solve (status -2 with the tag in the text).
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_session_new(
    opts: *const CSparseOptions,
) -> *mut {root}Session {{
    let session = catch_unwind(AssertUnwindSafe(|| {{
        if opts.is_null() {{
            LmSession::new(SparseFaer::<{fp}>::new())
        }} else {{
            LmSession::new(SparseFaer::<{fp}>::from_options(&(*opts).to_options()))
        }}
    }})).map_err(panic_text);
    Box::into_raw(Box::new({root}Session {{ session }}))
}}

#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_session_free(s: *mut {root}Session) {{
    if !s.is_null() {{
        drop(Box::from_raw(s));
    }}
}}

/// Drop the learned structure; the next solve runs cold.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_session_invalidate(s: *mut {root}Session) {{
    if let Ok(sess) = &mut (*s).session {{
        sess.invalidate();
    }}
}}

/// As {root_sn}_solve_sparse, through the session's cached analysis.
/// Error text lands on the model handle ({root_sn}_last_error).
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_session_solve(
    s: *mut {root}Session,
    h: *mut {handle},
    cfg: *const CLmConfig,
    out: *mut CLmResult,
) -> i32 {{
    let ss = &mut *s;
    let hh = &mut *h;
    let c = (*cfg).to_config();
    zero_result(out);
    let session = match &mut ss.session {{
        Ok(sess) => sess,
        Err(msg) => {{
            let msg = msg.clone();
            set_text(hh, &msg);
            (*out).status = -2;
            return -2;
        }}
    }};
    match catch_unwind(AssertUnwindSafe(|| session.solve(&mut hh.model, &c))) {{
        Ok(Ok(r)) => {{
            let code = fill_result(out, &r);
            (*out).detail = boxed(r);
            set_text(hh, \"\");
            code
        }}
        Ok(Err(f)) => {{
            set_text(hh, &f.to_string());
            if let Some(p) = f.partial {{
                fill_result(out, &p);
                (*out).status = -1;
                (*out).detail = boxed(*p);
            }}
            -1
        }}
        Err(p) => {{
            // A panic mid-solve may leave the session's cached
            // analysis half-built; drop it so the next solve is cold.
            session.invalidate();
            let msg = panic_text(p);
            set_text(hh, &msg);
            (*out).status = -2;
            -2
        }}
    }}
}}
"));

    // Band solve: the free-function entry point (no RootProblem
    // method), so the wrapper serializes and deserializes itself.
    let (b_ser, b_deser, b_fn) = if fp == "f32" {
        ("serialize32", "deserialize32", "solve_band_f32")
    } else {
        ("serialize64", "deserialize64", "solve_band")
    };
    out.push_str(&format!(
"
/// Band Cholesky solve; `kd` is the Hessian half-bandwidth in scalar
/// parameters. Returns the status code (>= 0: LmStatus; -1: solve
/// failure; -2: panic). Failure text via {root_sn}_last_error;
/// `out.detail` as in the other solves.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_solve_band(
    h: *mut {handle},
    kd: u32,
    cfg: *const CLmConfig,
    out: *mut CLmResult,
) -> i32 {{
    let hh = &mut *h;
    let c = (*cfg).to_config();
    zero_result(out);
    match catch_unwind(AssertUnwindSafe(|| {{
        let mut x0 = Vec::new();
        hh.model.{b_ser}(&mut x0);
        arael::simple_lm::{b_fn}(&x0, kd as usize, &mut hh.model, &c).map(|r| {{
            hh.model.{b_deser}(&r.x);
            r
        }})
    }})) {{
        Ok(Ok(r)) => {{
            let code = fill_result(out, &r);
            (*out).detail = boxed(r);
            set_text(hh, \"\");
            code
        }}
        Ok(Err(f)) => {{
            set_text(hh, &f.to_string());
            if let Some(p) = f.partial {{
                fill_result(out, &p);
                (*out).status = -1;
                (*out).detail = boxed(*p);
            }}
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

    // Cost evaluation at the current parameter values.
    let (ser, cast) = if fp == "f32" {
        ("serialize32", " as f64")
    } else {
        ("serialize64", "")
    };
    out.push_str(&format!(
"
/// Total cost at the current parameter values (evaluated at the
/// model's precision, returned as f64). NaN on a caught panic, with
/// the text via {root_sn}_last_error (a healthy cost can also be NaN
/// -- non-finite parameters -- so check last_error to distinguish).
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_cost(h: *mut {handle}) -> f64 {{
    let hh = &mut *h;
    match catch_unwind(AssertUnwindSafe(|| {{
        let mut params = Vec::new();
        hh.model.{ser}(&mut params);
        hh.model.calc_cost(&params){cast}
    }})) {{
        Ok(c) => {{
            set_text(hh, \"\");
            c
        }}
        Err(p) => {{
            let msg = panic_text(p);
            set_text(hh, &msg);
            f64::NAN
        }}
    }}
}}
"));

    // Per-constraint cost breakdown, only for `#[arael(root, jacobian)]`
    // roots (calc_cost_table lives on the JacobianModel trait the
    // attribute generates).
    if model.jacobian {
        out.push_str(&format!(
"
/// Per-constraint cost breakdown at the current parameters: each
/// block's robustified cost (a `loss` applied) grouped by constraint
/// label (`name = \"...\"` on the constraint attribute, else the
/// struct name); the table sums to {root_sn}_cost. Sorts the
/// table by label, stores it on the handle, and returns the row count
/// (-2: panic, text via {root_sn}_last_error). Read the rows with
/// {root_sn}_cost_table_name / _value.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_cost_table(h: *mut {handle}) -> i32 {{
    let hh = &mut *h;
    match catch_unwind(AssertUnwindSafe(|| {{
        let mut params = Vec::new();
        hh.model.{ser}(&mut params);
        arael::model::JacobianModel::calc_cost_table(&mut hh.model, &params)
    }})) {{
        Ok(t) => {{
            let mut rows: Vec<(&str, f64)> =
                t.into_iter().map(|(k, v)| (k, v{cast})).collect();
            rows.sort_by(|a, b| a.0.cmp(b.0));
            hh.cost_table = rows.into_iter()
                .map(|(k, v)| (CString::new(k).unwrap_or_default(), v))
                .collect();
            hh.cost_table.len() as i32
        }}
        Err(p) => {{
            let msg = panic_text(p);
            set_text(hh, &msg);
            -2
        }}
    }}
}}

/// Label of cost-table row `i`; valid until the next
/// {root_sn}_cost_table call.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_cost_table_name(h: *const {handle}, i: u32) -> *const c_char {{
    let hh = &*h;
    hh.cost_table[i as usize].0.as_ptr()
}}

/// Value of cost-table row `i`.
#[no_mangle]
pub unsafe extern \"C\" fn {root_sn}_cost_table_value(h: *const {handle}, i: u32) -> f64 {{
    let hh = &*h;
    hh.cost_table[i as usize].1
}}
"));
    }

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
/// Row-major dim x dim conditional covariance (f64) of one `{tn}`
/// (all other parameters held fixed); returns dim, or -1 (error) /
/// -2 (panic) / -3 (no assembly or buffer too small), text via
/// {root_sn}_last_error.
#[no_mangle]
pub unsafe extern \"C\" fn {sn}_conditional_cov(
    h: *mut {handle},
    p: *const {tn},
    out: *mut f64,
    cap: u32,
) -> i32 {{
    let hh = &mut *h;
    let Some(cov) = hh.cov.as_ref() else {{
        set_text(hh, \"conditional_cov: assemble_covariance was not called\");
        return -3;
    }};
    match catch_unwind(AssertUnwindSafe(|| cov.conditional_cov(&*p))) {{
        Ok(Ok(m)) => {{
            let dim = m.nrows();
            if (dim * dim) as u32 > cap {{
                set_text(hh, \"conditional_cov: buffer too small\");
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
/// Per-parameter standard deviations (sqrt of the marginal diagonal)
/// of one `{tn}`; returns the count, or -1 (error) / -2 (panic) / -3
/// (no assembly or buffer too small), text via {root_sn}_last_error.
/// Works on every CovMode, including TriDiagonal.
#[no_mangle]
pub unsafe extern \"C\" fn {sn}_std_dev(
    h: *mut {handle},
    p: *const {tn},
    out: *mut f64,
    cap: u32,
) -> i32 {{
    let hh = &mut *h;
    let Some(cov) = hh.cov.as_ref() else {{
        set_text(hh, \"std_dev: assemble_covariance was not called\");
        return -3;
    }};
    match catch_unwind(AssertUnwindSafe(|| cov.std_dev(&*p))) {{
        Ok(Ok(sd)) => {{
            if sd.len() as u32 > cap {{
                set_text(hh, \"std_dev: buffer too small\");
                return -3;
            }}
            for (i, v) in sd.iter().enumerate() {{
                *out.add(i) = *v;
            }}
            sd.len() as i32
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
