//! The model IR: a typed mirror of the JSON sidecar (docs/SIDECAR.md,
//! schema 1). Every emitter consumes this, so the backends cannot
//! drift from each other.

// The IR mirrors the whole schema; fields the current emitters do not
// read yet are still part of the interface.
#![allow(dead_code)]

use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize, Debug)]
pub struct Model {
    pub schema: u32,
    pub root: String,
    pub precision: String,
    /// The root was declared `#[arael(root, jacobian)]`; gates the
    /// per-constraint cost-table surface.
    #[serde(default)]
    pub jacobian: bool,
    pub types: BTreeMap<String, Type>,
    #[serde(default)]
    pub constraints: Vec<Constraint>,
}

#[derive(Deserialize, Debug)]
pub struct Type {
    pub role: String,
    pub param_count: u32,
    #[serde(default)]
    pub self_block: Option<String>,
    #[serde(default)]
    pub builtin: bool,
    pub fields: Vec<Field>,
}

#[derive(Deserialize, Debug)]
pub struct Field {
    pub name: String,
    pub kind: String,
    #[serde(default)]
    pub of: Option<String>,
    #[serde(default)]
    pub params: Option<u32>,
    #[serde(default)]
    pub container: Option<String>,
    #[serde(default)]
    pub spelled: Option<String>,
    #[serde(default)]
    pub target: Option<String>,
    #[serde(default)]
    pub variant: Option<String>,
    #[serde(default)]
    pub scalar: Option<String>,
    #[serde(default)]
    pub a: Option<String>,
    #[serde(default)]
    pub b: Option<String>,
    #[serde(default)]
    pub symbolic: bool,
}

#[derive(Deserialize, Debug)]
pub struct Constraint {
    pub on: String,
    pub label: String,
    pub file: String,
    pub line: u32,
}

impl Model {
    pub fn parse(json: &str) -> Result<Model, String> {
        let m: Model = serde_json::from_str(json).map_err(|e| format!("sidecar parse: {e}"))?;
        if m.schema != 1 {
            return Err(format!("sidecar schema {} not supported (want 1)", m.schema));
        }
        Ok(m)
    }
}

/// snake_case of a CamelCase type name: PoseTie -> pose_tie.
pub fn snake(name: &str) -> String {
    let mut out = String::new();
    for (i, c) in name.chars().enumerate() {
        if c.is_uppercase() {
            if i > 0 {
                out.push('_');
            }
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}
