//! JSON model sidecar: one file per root describing the model layout
//! for interface generators (schema in docs/SIDECAR.md). Emitted at
//! root expansion -- the registry and the reachable set are complete
//! there -- when `ARAEL_SIDECAR_DIR` is set. `cargo arael export`
//! drives this; manual use: `ARAEL_SIDECAR_DIR=out cargo build`.

use crate::{SymFieldType, SymLayout, registry_lookup, registry_param_total};

pub(crate) fn emit(
    dir: &str,
    root: &str,
    precision: &str,
    jacobian: bool,
    reachable_sorted: &[String],
) -> Result<(), String> {
    let json = build_json(root, precision, jacobian, reachable_sorted);
    let path = std::path::Path::new(dir);
    std::fs::create_dir_all(path).map_err(|e| format!("create {}: {}", dir, e))?;
    let file = path.join(format!("{}.json", root));
    std::fs::write(&file, json).map_err(|e| format!("write {}: {}", file.display(), e))?;
    Ok(())
}

/// JSON string escape (quotes, backslashes, control characters).
fn esc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

fn q(s: &str) -> String {
    format!("\"{}\"", esc(s))
}

/// Scalar / math type names emitted as `data` fields; everything else
/// non-model becomes `opaque` (present, no accessor).
fn is_data_name(n: &str) -> bool {
    matches!(n,
        "f32" | "f64" | "bool"
        | "u8" | "u16" | "u32" | "u64" | "usize"
        | "i8" | "i16" | "i32" | "i64" | "isize"
        | "vect2" | "vect2f" | "vect2d" | "vect3" | "vect3f" | "vect3d"
        | "matrix2" | "matrix2f" | "matrix2d" | "matrix3" | "matrix3f" | "matrix3d"
        | "quatern" | "quaternf" | "quaternd")
}

/// Last generic argument of a spelling if it is a float name, else the
/// default: `SelfBlock<Pose, f32>` -> "f32", `TripletBlock` -> "f64".
fn block_scalar(spelling: &str) -> &str {
    let inner = spelling.find('<')
        .map(|i| &spelling[i + 1..spelling.len().saturating_sub(1)])
        .unwrap_or("");
    match inner.rsplit(',').next().map(str::trim) {
        Some("f32") => "f32",
        Some("f64") => "f64",
        _ => "f64",
    }
}

/// Inner of a single-generic spelling: `Param<vect3f>` -> `vect3f`.
fn generic_inner(spelling: &str) -> &str {
    match (spelling.find('<'), spelling.ends_with('>')) {
        (Some(i), true) => &spelling[i + 1..spelling.len() - 1],
        _ => spelling,
    }
}

fn sft_size(sft: &SymFieldType) -> u32 {
    match sft {
        SymFieldType::Scalar => 1,
        SymFieldType::Vec2 => 2,
        SymFieldType::Vec3 => 3,
        _ => 0,
    }
}

/// One field's JSON object body (the `{...}` content, no braces).
fn field_kind(layout: &SymLayout, fname: &str, spelling: &str) -> String {
    let sft = layout.fields.iter().find(|(n, _)| n == fname).map(|(_, s)| s);
    let bare = spelling.strip_prefix("Option<")
        .and_then(|s| s.strip_suffix('>'))
        .unwrap_or(spelling);
    let mut parts: Vec<String> = vec![format!("\"name\": {}", q(fname))];
    let head = bare.split('<').next().unwrap_or(bare);
    let head_last = head.rsplit("::").next().unwrap_or(head);

    if matches!(head_last, "SelfBlock" | "BoxedSelfBlock") {
        parts.push("\"kind\": \"self_block\"".into());
        parts.push(format!("\"scalar\": {}", q(block_scalar(bare))));
    } else if matches!(head_last, "CrossBlock" | "BoxedCrossBlock") {
        parts.push("\"kind\": \"cross_block\"".into());
        let inner = generic_inner(bare);
        let args: Vec<&str> = inner.split(',').map(str::trim).collect();
        if args.len() >= 2 {
            parts.push(format!("\"a\": {}", q(args[0])));
            parts.push(format!("\"b\": {}", q(args[1])));
        }
        parts.push(format!("\"scalar\": {}", q(block_scalar(bare))));
    } else if head_last == "TripletBlock" {
        parts.push("\"kind\": \"triplet_block\"".into());
        parts.push(format!("\"scalar\": {}", q(block_scalar(bare))));
    } else if layout.param_fields.iter().any(|f| f == fname) {
        let (variant, is_euler) = if layout.euler_angle_fields.iter().any(|f| f == fname) {
            ("simple", true)
        } else if layout.universal_euler_angle_fields.iter().any(|f| f == fname) {
            ("universal", true)
        } else if layout.universal_rotvec_fields.iter().any(|f| f == fname) {
            ("rotvec", true)
        } else {
            ("", false)
        };
        if is_euler {
            parts.push("\"kind\": \"euler_param\"".into());
            parts.push(format!("\"variant\": {}", q(variant)));
            parts.push(format!("\"scalar\": {}", q(generic_inner(spelling))));
            parts.push("\"params\": 3".into());
        } else {
            parts.push("\"kind\": \"param\"".into());
            parts.push(format!("\"of\": {}", q(generic_inner(spelling))));
            parts.push(format!("\"params\": {}", sft.map(sft_size).unwrap_or(0)));
        }
    } else if let Some((_, target)) = layout.ref_paths.iter().find(|(n, _)| n == fname) {
        parts.push("\"kind\": \"ref\"".into());
        parts.push(format!("\"of\": {}", q(generic_inner(bare))));
        parts.push(format!("\"target\": {}", q(target)));
    } else if layout.collection_fields.iter().any(|f| f == fname) {
        let container = if head_last == "Deque" { "deque" }
            else if head_last == "Arena" { "arena" }
            else { "vec" };
        let elem = match sft {
            Some(SymFieldType::Struct(e)) => e.clone(),
            _ => generic_inner(bare).to_string(),
        };
        parts.push("\"kind\": \"collection\"".into());
        parts.push(format!("\"container\": {}", q(container)));
        parts.push(format!("\"of\": {}", q(&elem)));
        parts.push(format!("\"spelled\": {}", q(spelling)));
    } else {
        match sft {
            Some(SymFieldType::OptionalStruct(inner)) => {
                if registry_lookup(inner).is_some_and(|l| !l.fields.is_empty()) {
                    parts.push("\"kind\": \"optional\"".into());
                    parts.push(format!("\"of\": {}", q(inner)));
                } else {
                    parts.push("\"kind\": \"opaque\"".into());
                    parts.push(format!("\"of\": {}", q(spelling)));
                }
            }
            Some(SymFieldType::Struct(inner)) => {
                match registry_lookup(inner) {
                    Some(l) if l.component => {
                        parts.push("\"kind\": \"component\"".into());
                        parts.push(format!("\"of\": {}", q(inner)));
                    }
                    Some(l) if !l.fields.is_empty() => {
                        parts.push("\"kind\": \"struct\"".into());
                        parts.push(format!("\"of\": {}", q(inner)));
                    }
                    _ if is_data_name(inner) => {
                        parts.push("\"kind\": \"data\"".into());
                        parts.push(format!("\"of\": {}", q(spelling)));
                    }
                    _ => {
                        parts.push("\"kind\": \"opaque\"".into());
                        parts.push(format!("\"of\": {}", q(spelling)));
                    }
                }
            }
            Some(SymFieldType::Scalar) | Some(SymFieldType::Vec2) | Some(SymFieldType::Vec3)
            | Some(SymFieldType::Mat2) | Some(SymFieldType::Mat3) | Some(SymFieldType::Quat) => {
                parts.push("\"kind\": \"data\"".into());
                parts.push(format!("\"of\": {}", q(spelling)));
            }
            _ => {
                parts.push("\"kind\": \"skip\"".into());
                parts.push(format!("\"of\": {}", q(spelling)));
            }
        }
    }
    if layout.symbolic_fields.iter().any(|(n, _)| n == fname) {
        parts.push("\"symbolic\": true".into());
    }
    parts.join(", ")
}

pub(crate) fn build_json(
    root: &str,
    precision: &str,
    jacobian: bool,
    reachable_sorted: &[String],
) -> String {
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str("  \"schema\": 1,\n");
    out.push_str(&format!("  \"root\": {},\n", q(root)));
    out.push_str(&format!("  \"precision\": {},\n", q(precision)));
    out.push_str(&format!("  \"jacobian\": {},\n", jacobian));
    out.push_str("  \"types\": {\n");

    let mut type_entries: Vec<String> = Vec::new();
    for tn in reachable_sorted {
        let Some(layout) = registry_lookup(tn) else { continue };
        if layout.fields.is_empty() && layout.spelled_types.is_empty() { continue; }
        let role = if tn == root { "root" }
            else if layout.component { "component" }
            else { "entity" };
        let mut e = String::new();
        e.push_str(&format!("    {}: {{\n", q(tn)));
        e.push_str(&format!("      \"role\": {},\n", q(role)));
        e.push_str(&format!("      \"param_count\": {},\n", registry_param_total(tn)));
        if let Some(sb) = &layout.self_block_field {
            e.push_str(&format!("      \"self_block\": {},\n", q(sb)));
        }
        if layout.spelled_types.is_empty() {
            // Built-in component (TransformParam etc.): registered from
            // arael's own source, no field spellings. The generator
            // special-cases these by name.
            e.push_str("      \"builtin\": true,\n");
            e.push_str("      \"fields\": []\n");
        } else {
            e.push_str("      \"fields\": [\n");
            let rows: Vec<String> = layout.spelled_types.iter()
                .map(|(fname, sp)| format!("        {{{}}}", field_kind(&layout, fname, sp)))
                .collect();
            e.push_str(&rows.join(",\n"));
            e.push_str("\n      ]\n");
        }
        e.push_str("    }");
        type_entries.push(e);
    }
    out.push_str(&type_entries.join(",\n"));
    out.push_str("\n  },\n");

    out.push_str("  \"constraints\": [\n");
    let rows: Vec<String> = crate::registry_constraints().iter()
        .filter(|sc| reachable_sorted.iter().any(|t| *t == sc.struct_name))
        .map(|sc| format!(
            "    {{\"on\": {}, \"label\": {}, \"file\": {}, \"line\": {}}}",
            q(&sc.struct_name), q(&sc.label_hint), q(&sc.attr_file), sc.attr_line))
        .collect();
    out.push_str(&rows.join(",\n"));
    out.push_str("\n  ]\n");
    out.push_str("}\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry_store;

    fn store(name: &str, l: SymLayout) {
        registry_store(name, l).unwrap();
    }

    #[test]
    fn sidecar_json_for_a_hand_built_model() {
        // Distinct "Scj" prefix: the registry is process-global across
        // unit tests.
        store("ScjPose", SymLayout {
            fields: vec![
                ("pos".into(), SymFieldType::Vec3),
                ("stamp".into(), SymFieldType::Scalar),
                ("note".into(), SymFieldType::Struct("String".into())),
                ("hb".into(), SymFieldType::Skip),
            ],
            param_fields: vec!["pos".into()],
            self_block_field: Some("hb".into()),
            spelled_types: vec![
                ("pos".into(), "Param<vect3f>".into()),
                ("stamp".into(), "f64".into()),
                ("note".into(), "String".into()),
                ("hb".into(), "SelfBlock<ScjPose, f32>".into()),
            ],
            ..Default::default()
        });
        store("ScjWorld", SymLayout {
            fields: vec![("poses".into(), SymFieldType::Struct("ScjPose".into()))],
            collection_fields: vec!["poses".into()],
            is_root: true,
            spelled_types: vec![("poses".into(), "refs::Vec<ScjPose>".into())],
            ..Default::default()
        });

        let json = build_json("ScjWorld", "f32", false,
            &["ScjPose".to_string(), "ScjWorld".to_string()]);

        assert!(json.contains("\"schema\": 1"), "{json}");
        assert!(json.contains("\"root\": \"ScjWorld\""), "{json}");
        assert!(json.contains("\"precision\": \"f32\""), "{json}");
        assert!(json.contains("\"jacobian\": false"), "{json}");
        assert!(json.contains(
            "{\"name\": \"pos\", \"kind\": \"param\", \"of\": \"vect3f\", \"params\": 3}"),
            "{json}");
        assert!(json.contains(
            "{\"name\": \"stamp\", \"kind\": \"data\", \"of\": \"f64\"}"), "{json}");
        assert!(json.contains(
            "{\"name\": \"note\", \"kind\": \"opaque\", \"of\": \"String\"}"), "{json}");
        assert!(json.contains(
            "{\"name\": \"hb\", \"kind\": \"self_block\", \"scalar\": \"f32\"}"), "{json}");
        assert!(json.contains(
            "{\"name\": \"poses\", \"kind\": \"collection\", \"container\": \"vec\", \
             \"of\": \"ScjPose\", \"spelled\": \"refs::Vec<ScjPose>\"}"), "{json}");
        assert!(json.contains("\"self_block\": \"hb\""), "{json}");
        assert!(json.contains("\"role\": \"root\""), "{json}");
        // Valid-enough JSON: braces balance.
        assert_eq!(json.matches('{').count(), json.matches('}').count(), "{json}");
    }

    #[test]
    fn sidecar_kinds_for_refs_euler_option_and_components() {
        store("ScjOff", SymLayout {
            fields: vec![
                ("ref_c".into(), SymFieldType::Scalar),
                ("d".into(), SymFieldType::Scalar),
                ("c".into(), SymFieldType::Scalar),
            ],
            param_fields: vec!["d".into()],
            component: true,
            symbolic_fields: vec![("c".into(), "ref_c + d".into())],
            spelled_types: vec![
                ("ref_c".into(), "f64".into()),
                ("d".into(), "Param<f64>".into()),
                ("c".into(), "f64".into()),
            ],
            ..Default::default()
        });
        store("ScjGps", SymLayout {
            fields: vec![("m".into(), SymFieldType::Vec3)],
            spelled_types: vec![("m".into(), "vect3d".into())],
            ..Default::default()
        });
        store("ScjNode", SymLayout {
            fields: vec![
                ("ea".into(), SymFieldType::Vec3),
                ("off".into(), SymFieldType::Struct("ScjOff".into())),
                ("gps".into(), SymFieldType::OptionalStruct("ScjGps".into())),
                ("hb".into(), SymFieldType::Skip),
            ],
            param_fields: vec!["ea".into()],
            euler_angle_fields: vec!["ea".into()],
            self_block_field: Some("hb".into()),
            spelled_types: vec![
                ("ea".into(), "SimpleEulerAngleParam<f64>".into()),
                ("off".into(), "ScjOff".into()),
                ("gps".into(), "Option<ScjGps>".into()),
                ("hb".into(), "SelfBlock<ScjNode>".into()),
            ],
            ..Default::default()
        });
        store("ScjTie", SymLayout {
            fields: vec![
                ("a".into(), SymFieldType::Struct("ScjNode".into())),
                ("d".into(), SymFieldType::Scalar),
                ("hb".into(), SymFieldType::Skip),
            ],
            ref_paths: vec![("a".into(), "root.nodes".into())],
            spelled_types: vec![
                ("a".into(), "Ref<ScjNode>".into()),
                ("d".into(), "f64".into()),
                ("hb".into(), "CrossBlock<ScjNode, ScjNode>".into()),
            ],
            ..Default::default()
        });

        let json = build_json("ScjNode", "f64", true,
            &["ScjGps".into(), "ScjNode".into(), "ScjOff".into(), "ScjTie".into()]);

        assert!(json.contains("\"jacobian\": true"), "{json}");

        assert!(json.contains(
            "{\"name\": \"ea\", \"kind\": \"euler_param\", \"variant\": \"simple\", \
             \"scalar\": \"f64\", \"params\": 3}"), "{json}");
        assert!(json.contains(
            "{\"name\": \"off\", \"kind\": \"component\", \"of\": \"ScjOff\"}"), "{json}");
        assert!(json.contains(
            "{\"name\": \"gps\", \"kind\": \"optional\", \"of\": \"ScjGps\"}"), "{json}");
        assert!(json.contains(
            "{\"name\": \"a\", \"kind\": \"ref\", \"of\": \"ScjNode\", \
             \"target\": \"root.nodes\"}"), "{json}");
        assert!(json.contains(
            "{\"name\": \"hb\", \"kind\": \"cross_block\", \"a\": \"ScjNode\", \
             \"b\": \"ScjNode\", \"scalar\": \"f64\"}"), "{json}");
        assert!(json.contains(
            "{\"name\": \"c\", \"kind\": \"data\", \"of\": \"f64\", \"symbolic\": true}"),
            "{json}");
        assert!(json.contains("\"role\": \"component\""), "{json}");
    }
}
