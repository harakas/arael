//! The leaf walk shared by the shim and Python emitters: every settable
//! scalar an element wrapper exposes, in field order. The shim's
//! `push_n` slot record, its per-field column functions and the Python
//! `push(**fields)` format string all derive from this one list, so
//! the two sides cannot drift.
//!
//! A slot record is `mask words` then one 8-byte slot per scalar leaf:
//! a scalar, ref or bool field is one leaf, a math value one leaf per
//! component, a `Param` its value plus an `optimize` flag. Leaf `i`
//! sets bit `i % 64` of mask word `i / 64`; the shim assigns only the
//! masked leaves onto a `Default::default()` element.

use crate::ir::{Model, Type};

#[derive(Debug, Clone, PartialEq)]
pub enum LeafTy {
    F64,
    F32,
    U32,
    I32,
    Bool,
    Ref,
    /// `n` scalars of `scalar` (`f64` / `f32`), crossing through the
    /// shim's repr(C) `mirror` type.
    Math { scalar: String, n: usize, mirror: String },
}

#[derive(Debug, Clone)]
pub struct Leaf {
    /// Python keyword / property name (`pose_translation`, `pos_optimize`).
    pub name: String,
    /// Rust path from the element (`pose.translation`, `pos.optimize`).
    pub access: String,
    pub ty: LeafTy,
}

impl LeafTy {
    /// Slots the leaf occupies in a record; components of a column element.
    pub fn slots(&self) -> usize {
        match self {
            LeafTy::Math { n, .. } => *n,
            _ => 1,
        }
    }

    /// `struct` code(s) for the leaf's slots.
    pub fn pack_code(&self) -> String {
        match self {
            LeafTy::F64 | LeafTy::F32 => "d".to_string(),
            LeafTy::U32 | LeafTy::Bool | LeafTy::Ref => "Q".to_string(),
            LeafTy::I32 => "q".to_string(),
            LeafTy::Math { n, .. } => "d".repeat(*n),
        }
    }

    /// The C scalar type of one column element.
    pub fn column_c(&self) -> &str {
        match self {
            LeafTy::F64 => "f64",
            LeafTy::F32 => "f32",
            LeafTy::U32 | LeafTy::Ref => "u32",
            LeafTy::I32 => "i32",
            LeafTy::Bool => "u8",
            LeafTy::Math { scalar, .. } => scalar,
        }
    }

    /// The buffer-protocol format code the Python side checks a column
    /// against.
    pub fn column_code(&self) -> &str {
        match self {
            LeafTy::F64 => "d",
            LeafTy::F32 => "f",
            LeafTy::U32 | LeafTy::Ref => "I",
            LeafTy::I32 => "i",
            LeafTy::Bool => "B",
            LeafTy::Math { scalar, .. } => if scalar == "f32" { "f" } else { "d" },
        }
    }
}

/// Mask words a record of `n_leaves` needs (at least one).
pub fn mask_words(n_leaves: usize) -> usize {
    n_leaves.div_ceil(64).max(1)
}

/// Total slots in one record: mask words plus every leaf's slots.
pub fn record_slots(leaves: &[Leaf]) -> usize {
    mask_words(leaves.len()) + leaves.iter().map(|l| l.ty.slots()).sum::<usize>()
}

fn scalar_ty(of: &str) -> Option<LeafTy> {
    Some(match of {
        "f64" => LeafTy::F64,
        "f32" => LeafTy::F32,
        "u32" => LeafTy::U32,
        "i32" => LeafTy::I32,
        "bool" => LeafTy::Bool,
        _ => return None,
    })
}

fn math_ty(of: &str) -> Option<LeafTy> {
    let (scalar, n, mirror) = match of {
        "vect2f" => ("f32", 2, "CVec2F32".to_string()),
        "vect2d" => ("f64", 2, "CVec2F64".to_string()),
        "vect3f" => ("f32", 3, "CVec3F32".to_string()),
        "vect3d" => ("f64", 3, "CVec3F64".to_string()),
        "matrix2f" => ("f32", 4, "CMat2F32".to_string()),
        "matrix2d" => ("f64", 4, "CMat2F64".to_string()),
        "matrix3f" => ("f32", 9, "CMat3F32".to_string()),
        "matrix3d" => ("f64", 9, "CMat3F64".to_string()),
        "quaternf" => ("f32", 4, "CQuatF32".to_string()),
        "quaternd" => ("f64", 4, "CQuatF64".to_string()),
        _ => {
            let inst = crate::ir::ndim_math(of)?;
            let n = inst.1.iter().product();
            let mirror = crate::emit_ffi::ndim_mirror_name(&inst);
            return Some(LeafTy::Math { scalar: inst.0, n, mirror });
        }
    };
    Some(LeafTy::Math { scalar: scalar.to_string(), n, mirror })
}

fn value_ty(of: &str) -> Option<LeafTy> {
    scalar_ty(of).or_else(|| math_ty(of))
}

fn leaf(name: &str, access: &str, ty: LeafTy) -> Leaf {
    Leaf { name: name.to_string(), access: access.to_string(), ty }
}

/// The settable leaves of `t`, in field order. Fields with no scalar
/// surface (nested structs and user components, options, collections,
/// blocks, opaque data) contribute nothing; they keep their per-element
/// accessors.
pub fn leaves(model: &Model, t: &Type) -> Vec<Leaf> {
    let mut out = Vec::new();
    for f in &t.fields {
        let name = f.name.as_str();
        let of = f.of.as_deref().unwrap_or("");
        match f.kind.as_str() {
            "data" => {
                if let Some(ty) = value_ty(of) {
                    out.push(leaf(name, name, ty));
                }
            }
            "param" => {
                if let Some(ty) = value_ty(of) {
                    out.push(leaf(name, &format!("{name}.value"), ty));
                    out.push(leaf(&format!("{name}_optimize"),
                                  &format!("{name}.optimize"), LeafTy::Bool));
                }
            }
            "euler_param" => {
                let scalar = f.scalar.as_deref().unwrap_or("f64");
                let variant = f.variant.as_deref().unwrap_or("simple");
                let ty = match (variant, scalar) {
                    ("rotvec", "f32") => math_ty("quaternf"),
                    ("rotvec", _) => math_ty("quaternd"),
                    (_, "f32") => math_ty("vect3f"),
                    (_, _) => math_ty("vect3d"),
                }.expect("builtin math type");
                out.push(leaf(name, &format!("{name}.value"), ty));
                out.push(leaf(&format!("{name}_optimize"),
                              &format!("{name}.optimize"), LeafTy::Bool));
            }
            "component" => match of {
                "TransformParam" | "TransformParamF"
                | "ScaledTransformParam" | "ScaledTransformParamF" => {
                    let f32 = of.ends_with('F');
                    let scaled = of.starts_with("Scaled");
                    let (v3, q) = if f32 { ("vect3f", "quaternf") } else { ("vect3d", "quaternd") };
                    out.push(leaf(&format!("{name}_translation"),
                                  &format!("{name}.translation"),
                                  math_ty(v3).expect("builtin")));
                    out.push(leaf(&format!("{name}_rotation"),
                                  &format!("{name}.rotation"),
                                  math_ty(q).expect("builtin")));
                    if scaled {
                        out.push(leaf(&format!("{name}_scale"), &format!("{name}.scale"),
                                      if f32 { LeafTy::F32 } else { LeafTy::F64 }));
                    }
                    let flags: &[&str] = if scaled {
                        &["optimize_translation", "optimize_rotation", "optimize_scale"]
                    } else {
                        &["optimize_translation", "optimize_rotation"]
                    };
                    for flag in flags {
                        out.push(leaf(&format!("{name}_{flag}"),
                                      &format!("{name}.{flag}"), LeafTy::Bool));
                    }
                }
                "UnitVecParam" | "UnitVecParamF" => {
                    let v3 = if of == "UnitVecParamF" { "vect3f" } else { "vect3d" };
                    out.push(leaf(&format!("{name}_unit"), &format!("{name}.unit"),
                                  math_ty(v3).expect("builtin")));
                }
                "AngleParam" | "AngleParamF" => {
                    let sc = if of == "AngleParamF" { LeafTy::F32 } else { LeafTy::F64 };
                    out.push(leaf(&format!("{name}_angle"),
                                  &format!("{name}.angle.value"), sc));
                    out.push(leaf(&format!("{name}_angle_optimize"),
                                  &format!("{name}.angle.optimize"), LeafTy::Bool));
                }
                _ => {}
            },
            "ref" => out.push(leaf(name, name, LeafTy::Ref)),
            _ => {}
        }
    }
    let _ = model;
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::Field;
    use std::collections::BTreeMap;

    fn field(name: &str, kind: &str, of: Option<&str>) -> Field {
        Field {
            name: name.into(),
            kind: kind.into(),
            of: of.map(Into::into),
            ..Default::default()
        }
    }

    fn model_with(fields: Vec<Field>) -> (Model, Type) {
        let t = Type { role: "entity".into(), param_count: 0, self_block: None,
                       builtin: false, fields };
        let m = Model { schema: 1, root: "R".into(), precision: "f64".into(),
                        jacobian: false, types: BTreeMap::new(), constraints: vec![] };
        (m, t)
    }

    fn names(lv: &[Leaf]) -> Vec<(String, String, usize)> {
        lv.iter().map(|l| (l.name.clone(), l.access.clone(), l.ty.slots())).collect()
    }

    #[test]
    fn walk_expands_every_leaf_kind_in_field_order() {
        let mut ea = field("ea", "euler_param", None);
        ea.variant = Some("simple".into());
        let mut rv = field("rv", "euler_param", None);
        rv.variant = Some("rotvec".into());
        rv.scalar = Some("f32".into());
        let (m, t) = model_with(vec![
            field("x", "data", Some("f64")),
            field("n", "data", Some("i32")),
            field("pos", "param", Some("vect3d")),
            ea,
            rv,
            field("pose", "component", Some("TransformParamF")),
            field("st", "component", Some("ScaledTransformParam")),
            field("dir", "component", Some("UnitVecParam")),
            field("heading", "component", Some("AngleParam")),
            field("gain", "component", Some("Gain")),      // user component: no leaves
            field("info", "struct", Some("Info")),          // nested: no leaves
            field("gps", "optional", Some("GpsObs")),       // option: no leaves
            field("items", "collection", Some("N")),        // collection: no leaves
            field("hb", "self_block", None),
            field("a", "ref", Some("Pose")),
            field("h", "data", Some("matrix<f32, 2, 4>")),
        ]);
        let lv = leaves(&m, &t);
        assert_eq!(names(&lv), vec![
            ("x".into(), "x".into(), 1),
            ("n".into(), "n".into(), 1),
            ("pos".into(), "pos.value".into(), 3),
            ("pos_optimize".into(), "pos.optimize".into(), 1),
            ("ea".into(), "ea.value".into(), 3),
            ("ea_optimize".into(), "ea.optimize".into(), 1),
            ("rv".into(), "rv.value".into(), 4),
            ("rv_optimize".into(), "rv.optimize".into(), 1),
            ("pose_translation".into(), "pose.translation".into(), 3),
            ("pose_rotation".into(), "pose.rotation".into(), 4),
            ("pose_optimize_translation".into(), "pose.optimize_translation".into(), 1),
            ("pose_optimize_rotation".into(), "pose.optimize_rotation".into(), 1),
            ("st_translation".into(), "st.translation".into(), 3),
            ("st_rotation".into(), "st.rotation".into(), 4),
            ("st_scale".into(), "st.scale".into(), 1),
            ("st_optimize_translation".into(), "st.optimize_translation".into(), 1),
            ("st_optimize_rotation".into(), "st.optimize_rotation".into(), 1),
            ("st_optimize_scale".into(), "st.optimize_scale".into(), 1),
            ("dir_unit".into(), "dir.unit".into(), 3),
            ("heading_angle".into(), "heading.angle.value".into(), 1),
            ("heading_angle_optimize".into(), "heading.angle.optimize".into(), 1),
            ("a".into(), "a".into(), 1),
            ("h".into(), "h".into(), 8),
        ]);
        assert_eq!(lv[1].ty, LeafTy::I32);
        assert_eq!(lv[3].ty, LeafTy::Bool);
        assert!(matches!(&lv[6].ty, LeafTy::Math { scalar, n: 4, mirror }
            if scalar == "f32" && mirror == "CQuatF32"));
        assert!(matches!(&lv[8].ty, LeafTy::Math { scalar, n: 3, mirror }
            if scalar == "f32" && mirror == "CVec3F32"));
        assert!(matches!(&lv[13].ty, LeafTy::Math { scalar, n: 4, mirror }
            if scalar == "f64" && mirror == "CQuatF64"));
        assert_eq!(lv[14].ty, LeafTy::F64);
        assert!(matches!(&lv[22].ty, LeafTy::Math { scalar, n: 8, mirror }
            if scalar == "f32" && mirror == "CMatF32x2x4"));
        assert_eq!(lv[21].ty, LeafTy::Ref);
        // 1 mask word + 49 slots.
        assert_eq!(record_slots(&lv), 50);
    }

    #[test]
    fn walk_of_a_leafless_type_is_empty_but_still_one_mask_word() {
        let (m, t) = model_with(vec![
            field("gain", "component", Some("Gain")),
            field("hb", "self_block", None),
        ]);
        let lv = leaves(&m, &t);
        assert!(lv.is_empty());
        assert_eq!(record_slots(&lv), 1);
    }

    #[test]
    fn mask_words_round_up() {
        assert_eq!(mask_words(0), 1);
        assert_eq!(mask_words(64), 1);
        assert_eq!(mask_words(65), 2);
    }

    #[test]
    fn codes_follow_the_leaf_type() {
        assert_eq!(LeafTy::I32.pack_code(), "q");
        assert_eq!(LeafTy::Bool.column_c(), "u8");
        let m = math_ty("quaternd").unwrap();
        assert_eq!(m.slots(), 4);
        assert_eq!(m.pack_code(), "dddd");
        assert_eq!(m.column_code(), "d");
        let n = math_ty("matrix<f32, 2, 4>").unwrap();
        assert_eq!(n.slots(), 8);
        assert_eq!(n.column_c(), "f32");
    }
}
