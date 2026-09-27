//! wasm-bindgen crate emitter: JavaScript classes over a root model,
//! shaped like the Python package. The generated crate wraps the model
//! crate directly, not the C ABI: one `Rc<RefCell<Root>>` per root,
//! container accessors and entity handles that hold a clone of it and
//! re-resolve their element on every access, single values as plain
//! objects, bulk access as typed arrays, and the solver surface
//! (config, options, session, result, covariance) as classes.

use crate::ir::{snake, Field, Model, Type};
use crate::leaves::{leaves, Leaf, LeafTy};

/// The wasm-bindgen crate version the generated manifest pins; the CLI
/// must match it exactly.
pub const WASM_BINDGEN_VERSION: &str = "0.2.129";

/// camelCase of a snake_case name: `rot_angle` -> `rotAngle`.
pub fn camel(s: &str) -> String {
    let mut out = String::new();
    let mut up = false;
    for c in s.chars() {
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

/// PascalCase of a snake_case name: `rot_angle` -> `RotAngle`.
fn pascal(s: &str) -> String {
    let c = camel(s);
    let mut it = c.chars();
    match it.next() {
        Some(f) => f.to_ascii_uppercase().to_string() + it.as_str(),
        None => String::new(),
    }
}

/// The Rust method name behind a JavaScript name: `rotAngle` ->
/// `rot_angle`, so a getter and its setter pair up.
fn snake_of_camel(js: &str) -> String {
    let mut out = String::new();
    for c in js.chars() {
        if c.is_ascii_uppercase() {
            out.push('_');
            out.push(c.to_ascii_lowercase());
        } else {
            out.push(c);
        }
    }
    out
}

/// How one value crosses the boundary: the Rust type JavaScript sees,
/// an expression turning a model value `{x}` into it, and one turning
/// the received `v` back into the model value (fallible for objects).
struct Conv {
    js_ty: &'static str,
    to: String,
    from: String,
}

fn conv(of: &str) -> Option<Conv> {
    let c = |js_ty, to: &str, from: &str| Some(Conv {
        js_ty, to: to.to_string(), from: from.to_string(),
    });
    match of {
        "f64" => c("f64", "{x}", "v"),
        "f32" => c("f64", "{x} as f64", "v as f32"),
        "bool" => c("bool", "{x}", "v"),
        "u32" => c("u32", "{x}", "v"),
        "i32" => c("i32", "{x}", "v"),
        "vect2f" | "vect2d" => c("JsValue", "vect2_to_js({x})", "vect2_from_js(&v)?"),
        "vect3f" | "vect3d" => c("JsValue", "vect3_to_js({x})", "vect3_from_js(&v)?"),
        "matrix2f" | "matrix2d" => c("JsValue", "mat2_to_js({x})", "mat2_from_js(&v)?"),
        "matrix3f" | "matrix3d" => c("JsValue", "mat3_to_js({x})", "mat3_from_js(&v)?"),
        "quaternf" | "quaternd" => c("JsValue", "quat_to_js({x})", "quat_from_js(&v)?"),
        _ => {
            let (_, dims) = crate::ir::ndim_math(of)?;
            if dims.len() == 1 {
                c("JsValue", "vecn_to_js({x})", "vecn_from_js(&v)?")
            } else {
                c("JsValue", "matn_to_js({x})", "matn_from_js(&v)?")
            }
        }
    }
}

/// The scalar a leaf's flat form carries, and the typed array it fills.
fn leaf_array(ty: &LeafTy) -> (&'static str, &'static str) {
    match ty {
        LeafTy::F64 | LeafTy::F32 | LeafTy::Math { .. } => ("f64", "js_sys::Float64Array"),
        LeafTy::U32 | LeafTy::Ref => ("u32", "js_sys::Uint32Array"),
        LeafTy::I32 => ("i32", "js_sys::Int32Array"),
        LeafTy::Bool => ("u8", "js_sys::Uint8Array"),
    }
}

/// Non-root, non-builtin types whose surface gets a handle class.
fn surfaced(model: &Model) -> Vec<(&String, &Type)> {
    let mut v: Vec<(&String, &Type)> = model.types.iter()
        .filter(|(tn, t)| **tn != model.root && !t.builtin)
        .collect();
    v.sort_by(|a, b| a.0.cmp(b.0));
    v
}

/// One root's generation state.
struct Ctx<'a> {
    model: &'a Model,
    /// Prefix on every class name, the root's name in a multi-root crate.
    prefix: String,
    fp: &'a str,
}

impl Ctx<'_> {
    /// The model type behind a type name, spelled at the root's precision.
    fn mty(&self, tn: &str) -> String {
        match self.model.types.get(tn) {
            Some(t) if t.generic => format!("m::{tn}<{}>", self.fp),
            _ => format!("m::{tn}"),
        }
    }
    /// The wrapper class of a type.
    fn cls(&self, tn: &str) -> String {
        format!("{}{tn}", self.prefix)
    }
    fn root(&self) -> String {
        self.mty(&self.model.root)
    }
}

/// The wrapper the accessors are generated on: the model type it
/// reaches, and whether it is the root (reached directly) or a handle
/// (reached through its resolvers).
struct Owner {
    ty: String,
    is_root: bool,
}

/// The `with` (mutable) and `view` (shared) methods of a wrapper.
fn with_impl(o: &Owner) -> String {
    if o.is_root {
        format!(
"    fn with<R>(&self, f: impl FnOnce(&mut {ty}) -> R) -> Result<R, JsValue> {{
        let mut g = self.root.borrow_mut();
        Ok(f(&mut *g))
    }}
", ty = o.ty)
    } else {
        format!(
"    fn with<R>(&self, f: impl FnOnce(&mut {ty}) -> R) -> Result<R, JsValue> {{
        let mut g = self.root.borrow_mut();
        let e = (self.at)(&mut *g).ok_or_else(stale)?;
        Ok(f(e))
    }}
", ty = o.ty)
    }
}

/// A child handle's two resolvers, built from this wrapper's own:
/// `at` over `o: &mut Owner` yielding `Option<&mut Child>`, `see` the
/// same over a shared reference.
fn child(o: &Owner, at: &str, see: &str) -> String {
    if o.is_root {
        format!("at: mk_at(move |o| {{ {at} }}), see: mk_see(move |o| {{ {see} }})")
    } else {
        format!(
            "at: {{ let p = self.at.clone(); mk_at(move |g| p(g).and_then(|o| {{ {at} }})) }}, \
             see: {{ let p = self.see.clone(); mk_see(move |g| p(g).and_then(|o| {{ {see} }})) }}")
    }
}

/// A getter/setter pair over `access` (an expression over `e`).
fn rw(out: &mut String, js: &str, cv: &Conv, access: &str) {
    let to = cv.to.replace("{x}", access);
    let get = snake_of_camel(js);
    out.push_str(&format!(
"    #[wasm_bindgen(getter, js_name = \"{js}\")]
    pub fn {get}(&self) -> Result<{ty}, JsValue> {{
        self.with(|e| {to})
    }}
    #[wasm_bindgen(setter, js_name = \"{js}\")]
    pub fn set_{get}(&self, v: {ty}) -> Result<(), JsValue> {{
        let v = {from};
        self.with(|e| {{ {access} = v; }})
    }}
", ty = cv.js_ty, from = cv.from));
}

/// A read-only getter.
fn ro(out: &mut String, js: &str, cv: &Conv, access: &str) {
    let to = cv.to.replace("{x}", access);
    out.push_str(&format!(
"    #[wasm_bindgen(getter, js_name = \"{js}\")]
    pub fn {get}(&self) -> Result<{ty}, JsValue> {{
        self.with(|e| {to})
    }}
", get = snake_of_camel(js), ty = cv.js_ty));
}

fn unsupported(type_name: &str, f: &Field) -> String {
    format!("`{type_name}.{}`: field kind `{}` of `{}` has no JavaScript form",
        f.name, f.kind, f.of.as_deref().unwrap_or("?"))
}

/// A method handing out the handle of a directly held sub-model or
/// user component.
fn nested(out: &mut String, cx: &Ctx, o: &Owner, name: &str, js: &str, of: &str) {
    let cls = cx.cls(of);
    let ch = child(o, &format!("Some(&mut o.{name})"), &format!("Some(&o.{name})"));
    out.push_str(&format!(
"    #[wasm_bindgen(js_name = \"{js}\")]
    pub fn {name}(&self) -> {cls} {{
        {cls} {{ root: self.root.clone(), {ch} }}
    }}
"));
}

/// The accessors of one field on one wrapper.
fn field_accessors(
    out: &mut String,
    cx: &Ctx,
    o: &Owner,
    type_name: &str,
    f: &Field,
) -> Result<(), String> {
    let name = &f.name;
    let js = camel(name);
    let of = f.of.as_deref().unwrap_or("");
    match f.kind.as_str() {
        "data" => {
            let cv = conv(of).ok_or_else(|| unsupported(type_name, f))?;
            rw(out, &js, &cv, &format!("e.{name}"));
        }
        "param" => {
            let cv = conv(of).ok_or_else(|| unsupported(type_name, f))?;
            rw(out, &js, &cv, &format!("e.{name}.value"));
            rw(out, &format!("{js}Optimize"), &conv("bool").unwrap(), &format!("e.{name}.optimize"));
        }
        "euler_param" => {
            let scalar = f.scalar.as_deref().unwrap_or("f64");
            let variant = f.variant.as_deref().unwrap_or("simple");
            let of = match (variant, scalar) {
                ("rotvec", "f32") => "quaternf",
                ("rotvec", _) => "quaternd",
                (_, "f32") => "vect3f",
                (_, _) => "vect3d",
            };
            rw(out, &js, &conv(of).unwrap(), &format!("e.{name}.value"));
            rw(out, &format!("{js}Optimize"), &conv("bool").unwrap(), &format!("e.{name}.optimize"));
        }
        "component" => match of {
            "TransformParam" | "TransformParamF"
            | "ScaledTransformParam" | "ScaledTransformParamF" => {
                let f32 = of.ends_with('F');
                let scaled = of.starts_with("Scaled");
                let (v3, q, sc) = if f32 { ("vect3f", "quaternf", "f32") } else { ("vect3d", "quaternd", "f64") };
                rw(out, &format!("{js}Translation"), &conv(v3).unwrap(), &format!("e.{name}.translation"));
                rw(out, &format!("{js}Rotation"), &conv(q).unwrap(), &format!("e.{name}.rotation"));
                if scaled {
                    rw(out, &format!("{js}Scale"), &conv(sc).unwrap(), &format!("e.{name}.scale"));
                }
                let flags: &[&str] = if scaled {
                    &["optimize_translation", "optimize_rotation", "optimize_scale"]
                } else {
                    &["optimize_translation", "optimize_rotation"]
                };
                for flag in flags {
                    rw(out, &format!("{js}{}", pascal(flag)), &conv("bool").unwrap(),
                       &format!("e.{name}.{flag}"));
                }
            }
            "UnitVecParam" | "UnitVecParamF" => {
                let v3 = if of == "UnitVecParamF" { "vect3f" } else { "vect3d" };
                rw(out, &format!("{js}Unit"), &conv(v3).unwrap(), &format!("e.{name}.unit"));
                for i in 0..2 {
                    ro(out, &format!("{js}UnitD{i}"), &conv(v3).unwrap(), &format!("e.{name}.unit_d[{i}]"));
                }
            }
            "AngleParam" | "AngleParamF" => {
                let (sc, m2) = if of == "AngleParamF" { ("f32", "matrix2f") } else { ("f64", "matrix2d") };
                rw(out, &format!("{js}Angle"), &conv(sc).unwrap(), &format!("e.{name}.angle.value"));
                rw(out, &format!("{js}AngleOptimize"), &conv("bool").unwrap(), &format!("e.{name}.angle.optimize"));
                let to = conv(m2).unwrap().to.replace("{x}", &format!("e.{name}.rotation_matrix()"));
                out.push_str(&format!(
"    /// The rotation matrix at the current angle.
    #[wasm_bindgen(js_name = \"{js}RotationMatrix\")]
    pub fn {name}_rotation_matrix(&self) -> Result<JsValue, JsValue> {{
        self.with(|e| {to})
    }}
"));
            }
            _ => {
                if !cx.model.types.contains_key(of) {
                    return Err(unsupported(type_name, f));
                }
                nested(out, cx, o, name, &js, of);
            }
        },
        "struct" => {
            if !cx.model.types.contains_key(of) {
                return Err(unsupported(type_name, f));
            }
            nested(out, cx, o, name, &js, of);
        }
        "optional" => {
            if !cx.model.types.contains_key(of) {
                return Err(unsupported(type_name, f));
            }
            let cls = cx.cls(of);
            let ch = child(o, &format!("o.{name}.as_mut()"), &format!("o.{name}.as_ref()"));
            let big = pascal(name);
            out.push_str(&format!(
"    #[wasm_bindgen(js_name = \"has{big}\")]
    pub fn has_{name}(&self) -> Result<bool, JsValue> {{
        self.with(|e| e.{name}.is_some())
    }}
    /// Put a default `{of}` in place, replacing any, and hand it back.
    #[wasm_bindgen(js_name = \"make{big}\")]
    pub fn make_{name}(&self) -> Result<{cls}, JsValue> {{
        self.with(|e| {{ e.{name} = Some(Default::default()); }})?;
        Ok({cls} {{ root: self.root.clone(), {ch} }})
    }}
    #[wasm_bindgen(js_name = \"clear{big}\")]
    pub fn clear_{name}(&self) -> Result<(), JsValue> {{
        self.with(|e| {{ e.{name} = None; }})
    }}
    /// The `{of}` in place, or undefined.
    #[wasm_bindgen(js_name = \"{js}\")]
    pub fn {name}(&self) -> Result<Option<{cls}>, JsValue> {{
        let has = self.with(|e| e.{name}.is_some())?;
        Ok(has.then(|| {cls} {{ root: self.root.clone(), {ch} }}))
    }}
"));
        }
        "ref" => {
            out.push_str(&format!(
"    /// The ref as a number; 4294967295 is none.
    #[wasm_bindgen(getter, js_name = \"{js}\")]
    pub fn {name}(&self) -> Result<u32, JsValue> {{
        self.with(|e| e.{name}.to_raw())
    }}
    #[wasm_bindgen(setter, js_name = \"{js}\")]
    pub fn set_{name}(&self, v: u32) -> Result<(), JsValue> {{
        self.with(|e| {{ e.{name} = arael::refs::Ref::from_raw(v); }})
    }}
"));
        }
        "collection" => {
            let acc = format!("{}{}", cx.cls(type_name), pascal(name));
            let ch = child(o, "Some(o)", "Some(o)");
            out.push_str(&format!(
"    /// The `{name}` collection.
    #[wasm_bindgen(js_name = \"{js}\")]
    pub fn {name}(&self) -> {acc} {{
        {acc} {{ root: self.root.clone(), {ch} }}
    }}
"));
        }
        "self_block" | "cross_block" | "triplet_block" | "skip" | "opaque" => {}
        _ => return Err(unsupported(type_name, f)),
    }
    Ok(())
}

/// The container accessor class of one collection field.
fn collection_class(
    out: &mut String,
    cx: &Ctx,
    owner_tn: &str,
    f: &Field,
) -> Result<(), String> {
    let name = &f.name;
    let elem = f.of.as_deref().ok_or("collection without element")?;
    let container = f.container.as_deref().unwrap_or("vec");
    let spelled = f.spelled.as_deref().unwrap_or("");
    let refs_vec = container == "vec"
        && (spelled.starts_with("refs::Vec<") || spelled.contains("::refs::Vec<"));
    if container == "vec" && !refs_vec && !spelled.starts_with("std::vec::Vec<") {
        return Err(format!(
            "collection `{name}` spelled `{spelled}`: spell it `refs::Vec<..>` or \
             `std::vec::Vec<..>` so the generator knows the container flavor"));
    }
    let acc = format!("{}{}", cx.cls(owner_tn), pascal(name));
    let ecls = cx.cls(elem);
    let oty = cx.mty(owner_tn);
    let root = cx.root();
    // Element resolvers from the accessor's own (which reach the owner).
    let by_ref = format!(
        "at: {{ let p = self.at.clone(); mk_at(move |g| p(g).and_then(|o| o.{name}.get_mut(arael::refs::Ref::from_raw(r)))) }}, \
         see: {{ let p = self.see.clone(); mk_see(move |g| p(g).and_then(|o| o.{name}.get(arael::refs::Ref::from_raw(r)))) }}");
    let by_index = format!(
        "at: {{ let p = self.at.clone(); mk_at(move |g| p(g).and_then(|o| o.{name}.get_mut(i))) }}, \
         see: {{ let p = self.see.clone(); mk_see(move |g| p(g).and_then(|o| o.{name}.get(i))) }}");
    let handle = |ch: &str| format!("{ecls} {{ root: self.root.clone(), {ch} }}");
    out.push_str(&format!(
"
/// The `{owner_tn}.{name}` collection, `{spelled}`.
#[wasm_bindgen]
pub struct {acc} {{
    root: Rc<RefCell<{root}>>,
    at: At<{oty}>,
    see: See<{oty}>,
}}

#[wasm_bindgen]
impl {acc} {{
    fn with<R>(&self, f: impl FnOnce(&mut {oty}) -> R) -> Result<R, JsValue> {{
        let mut g = self.root.borrow_mut();
        let o = (self.at)(&mut *g).ok_or_else(stale)?;
        Ok(f(o))
    }}
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> Result<u32, JsValue> {{
        self.with(|o| o.{name}.len() as u32)
    }}
    pub fn reserve(&self, additional: u32) -> Result<(), JsValue> {{
        self.with(|o| o.{name}.reserve(additional as usize))
    }}
    pub fn clear(&self) -> Result<(), JsValue> {{
        self.with(|o| o.{name}.clear())
    }}
"));
    let ref_get = format!(
"    /// The element a ref addresses; throws on a stale or foreign ref.
    pub fn get(&self, r: u32) -> Result<{ecls}, JsValue> {{
        if !self.contains(r)? {{ return Err(js_err(\"ref does not address a live element\")); }}
        Ok({h})
    }}
    /// The element a ref addresses, or undefined.
    #[wasm_bindgen(js_name = \"tryGet\")]
    pub fn try_get(&self, r: u32) -> Result<Option<{ecls}>, JsValue> {{
        Ok(self.contains(r)?.then(|| {h}))
    }}
    /// True while `r` addresses a live element of this collection.
    pub fn contains(&self, r: u32) -> Result<bool, JsValue> {{
        self.with(|o| o.{name}.contains_ref(arael::refs::Ref::from_raw(r)))
    }}
", h = handle(&by_ref));
    let ref_at = format!(
"    /// The ref of the element at `i` as a number; 4294967295 past the end.
    #[wasm_bindgen(js_name = \"refAt\")]
    pub fn ref_at(&self, i: u32) -> Result<u32, JsValue> {{
        self.with(|o| {{
            if (i as usize) < o.{name}.len() {{ o.{name}.ref_at(i as usize).to_raw() }} else {{ u32::MAX }}
        }})
    }}
    /// The refs of the elements from `start`, `n` of them.
    #[wasm_bindgen(js_name = \"getRefsN\")]
    pub fn get_refs_n(&self, start: u32, n: u32) -> Result<js_sys::Uint32Array, JsValue> {{
        let v = self.with(|o| {{
            let (s, n) = (start as usize, n as usize);
            if s + n > o.{name}.len() {{ return Err(js_err(\"range past the end\")); }}
            Ok((s..s + n).map(|i| o.{name}.ref_at(i).to_raw()).collect::<Vec<u32>>())
        }})??;
        Ok(js_sys::Uint32Array::from(&v[..]))
    }}
");
    let ends = |first: &str, last: &str| format!(
"    /// The {first} element's ref, or 4294967295 when empty.
    #[wasm_bindgen(js_name = \"{first_js}\")]
    pub fn {first}(&self) -> Result<u32, JsValue> {{
        self.with(|o| o.{name}.{first}().map_or(u32::MAX, |r| r.to_raw()))
    }}
    /// The {last} element's ref, or 4294967295 when empty.
    #[wasm_bindgen(js_name = \"{last_js}\")]
    pub fn {last}(&self) -> Result<u32, JsValue> {{
        self.with(|o| o.{name}.{last}().map_or(u32::MAX, |r| r.to_raw()))
    }}
", first_js = camel(first), last_js = camel(last));
    let push_n = format!(
"    /// Append `n` default elements; returns the index of the first.
    #[wasm_bindgen(js_name = \"pushN\")]
    pub fn push_n(&self, n: u32) -> Result<u32, JsValue> {{
        self.with(|o| {{
            let first = o.{name}.len();
            o.{name}.reserve(n as usize);
            for _ in 0..n {{ o.{name}.push(Default::default()); }}
            first as u32
        }})
    }}
");
    match container {
        "vec" if refs_vec => {
            out.push_str(&format!(
"    /// Append a default `{elem}` and hand it back.
    pub fn push(&self) -> Result<{ecls}, JsValue> {{
        let r = self.with(|o| o.{name}.push(Default::default()).to_raw())?;
        Ok({h})
    }}
{push_n}    /// The element at index `i`; throws past the end.
    pub fn at(&self, i: u32) -> Result<{ecls}, JsValue> {{
        let r = self.with(|o| {{
            if (i as usize) < o.{name}.len() {{ Some(o.{name}.ref_at(i as usize).to_raw()) }} else {{ None }}
        }})?.ok_or_else(|| js_err(\"index past the end\"))?;
        Ok({h})
    }}
    /// Drops the last element; false when already empty.
    pub fn pop(&self) -> Result<bool, JsValue> {{
        self.with(|o| o.{name}.pop().is_some())
    }}
    pub fn truncate(&self, len: u32) -> Result<(), JsValue> {{
        self.with(|o| o.{name}.truncate(len as usize))
    }}
", h = handle(&by_ref)));
            out.push_str(&ref_at);
            out.push_str(&ends("first_ref", "last_ref"));
            out.push_str(&ref_get);
        }
        "vec" => {
            out.push_str(&format!(
"    /// Append a default `{elem}` and hand it back.
    pub fn push(&self) -> Result<{ecls}, JsValue> {{
        let i = self.with(|o| {{ o.{name}.push(Default::default()); o.{name}.len() - 1 }})?;
        Ok({h})
    }}
{push_n}    /// The element at index `i`; throws past the end.
    pub fn at(&self, i: u32) -> Result<{ecls}, JsValue> {{
        let i = i as usize;
        if !self.with(|o| i < o.{name}.len())? {{ return Err(js_err(\"index past the end\")); }}
        Ok({h})
    }}
    /// Drops the last element; false when already empty.
    pub fn pop(&self) -> Result<bool, JsValue> {{
        self.with(|o| o.{name}.pop().is_some())
    }}
    pub fn truncate(&self, len: u32) -> Result<(), JsValue> {{
        self.with(|o| o.{name}.truncate(len as usize))
    }}
", h = handle(&by_index)));
        }
        "deque" => {
            for (method, js, end) in [("push_back", "pushBack", "back"), ("push_front", "pushFront", "front")] {
                out.push_str(&format!(
"    /// A default `{elem}` at the {end}, handed back.
    #[wasm_bindgen(js_name = \"{js}\")]
    pub fn {method}(&self) -> Result<{ecls}, JsValue> {{
        let r = self.with(|o| o.{name}.{method}(Default::default()).to_raw())?;
        Ok({h})
    }}
", h = handle(&by_ref)));
            }
            out.push_str(&format!(
"    /// The element at index `i`; throws past the end.
    pub fn at(&self, i: u32) -> Result<{ecls}, JsValue> {{
        let r = self.with(|o| {{
            if (i as usize) < o.{name}.len() {{ Some(o.{name}.ref_at(i as usize).to_raw()) }} else {{ None }}
        }})?.ok_or_else(|| js_err(\"index past the end\"))?;
        Ok({h})
    }}
    /// Drops the back element; false when already empty.
    #[wasm_bindgen(js_name = \"popBack\")]
    pub fn pop_back(&self) -> Result<bool, JsValue> {{
        self.with(|o| o.{name}.pop_back().is_some())
    }}
    /// Drops the front element; false when already empty.
    #[wasm_bindgen(js_name = \"popFront\")]
    pub fn pop_front(&self) -> Result<bool, JsValue> {{
        self.with(|o| o.{name}.pop_front().is_some())
    }}
    pub fn truncate(&self, len: u32) -> Result<(), JsValue> {{
        self.with(|o| o.{name}.truncate(len as usize))
    }}
", h = handle(&by_ref)));
            out.push_str(&ref_at);
            out.push_str(&ends("front_ref", "back_ref"));
            out.push_str(&ref_get);
        }
        "arena" => {
            out.push_str(&format!(
"    /// Add a default `{elem}`; returns its ref.
    pub fn push(&self) -> Result<u32, JsValue> {{
        self.with(|o| o.{name}.push(Default::default()).to_raw())
    }}
    /// Remove the element a ref addresses; false when it was not live.
    pub fn remove(&self, r: u32) -> Result<bool, JsValue> {{
        self.with(|o| o.{name}.remove(arael::refs::Ref::from_raw(r)).is_some())
    }}
    /// The first live element's ref, or 4294967295 when empty.
    pub fn first(&self) -> Result<u32, JsValue> {{
        self.with(|o| o.{name}.first_ref().map_or(u32::MAX, |r| r.to_raw()))
    }}
    /// The next live element after `r`, or 4294967295 past the end.
    pub fn next(&self, r: u32) -> Result<u32, JsValue> {{
        self.with(|o| o.{name}.next_ref(arael::refs::Ref::from_raw(r)).map_or(u32::MAX, |r| r.to_raw()))
    }}
    /// The last live element's ref, or 4294967295 when empty.
    pub fn last(&self) -> Result<u32, JsValue> {{
        self.with(|o| o.{name}.last_ref().map_or(u32::MAX, |r| r.to_raw()))
    }}
    /// The previous live element before `r`, or 4294967295 past the front.
    pub fn prev(&self, r: u32) -> Result<u32, JsValue> {{
        self.with(|o| o.{name}.prev_ref(arael::refs::Ref::from_raw(r)).map_or(u32::MAX, |r| r.to_raw()))
    }}
"));
            out.push_str(&ref_get);
        }
        other => return Err(format!("`{owner_tn}.{name}`: unknown container `{other}`")),
    }
    // Bulk access by index, for the containers an index reaches.
    if container != "arena" {
        let et = cx.model.types.get(elem).ok_or_else(|| format!("unknown element type `{elem}`"))?;
        for leaf in leaves(cx.model, et) {
            bulk(out, name, &leaf);
        }
    }
    out.push_str("}\n");
    Ok(())
}

/// The bulk getter and setter of one leaf over an indexed container.
fn bulk(out: &mut String, coll: &str, leaf: &Leaf) {
    let (scalar, arr) = leaf_array(&leaf.ty);
    let big = pascal(&leaf.name);
    let access = &leaf.access;
    let (read, write) = match &leaf.ty {
        LeafTy::F64 | LeafTy::U32 | LeafTy::I32 => (
            format!("out.push(e.{access});"),
            format!("e.{access} = c[0];")),
        LeafTy::F32 => (
            format!("out.push(e.{access} as f64);"),
            format!("e.{access} = c[0] as f32;")),
        LeafTy::Bool => (
            format!("out.push(e.{access} as u8);"),
            format!("e.{access} = c[0] != 0;")),
        LeafTy::Ref => (
            format!("out.push(e.{access}.to_raw());"),
            format!("e.{access} = arael::refs::Ref::from_raw(c[0]);")),
        LeafTy::Math { .. } => (
            format!("e.{access}.flat(&mut out);"),
            format!("e.{access} = Flat::unflat(c);")),
    };
    let slots = leaf.ty.slots();
    out.push_str(&format!(
"    /// `{leaf}` of the elements from `start`, `n` of them, {slots} value(s) each.
    #[wasm_bindgen(js_name = \"get{big}N\")]
    pub fn get_{sn}_n(&self, start: u32, n: u32) -> Result<{arr}, JsValue> {{
        let v = self.with(|o| {{
            let (s, n) = (start as usize, n as usize);
            if s + n > o.{coll}.len() {{ return Err(js_err(\"range past the end\")); }}
            let mut out: Vec<{scalar}> = Vec::with_capacity(n * {slots});
            for i in s..s + n {{ let e = &o.{coll}[i]; {read} }}
            Ok(out)
        }})??;
        Ok({arr}::from(&v[..]))
    }}
    /// Set `{leaf}` of the elements from `start`, {slots} value(s) each.
    #[wasm_bindgen(js_name = \"set{big}N\")]
    pub fn set_{sn}_n(&self, start: u32, values: &[{scalar}]) -> Result<(), JsValue> {{
        self.with(|o| {{
            let s = start as usize;
            let n = values.len() / {slots};
            if s + n > o.{coll}.len() {{ return Err(js_err(\"range past the end\")); }}
            for (k, c) in values.chunks_exact({slots}).enumerate() {{ let e = &mut o.{coll}[s + k]; {write} }}
            Ok(())
        }})?
    }}
", leaf = leaf.name, sn = leaf.name));
}

/// The handle class of one entity, component or sub-model type.
fn type_class(out: &mut String, cx: &Ctx, tn: &str, t: &Type) -> Result<(), String> {
    let cls = cx.cls(tn);
    let ty = cx.mty(tn);
    let root = cx.root();
    let o = Owner { ty: ty.clone(), is_root: false };
    out.push_str(&format!(
"
/// A `{tn}` in its owner's storage, addressed by key rather than by
/// pointer: the element is re-resolved on every access, so growing the
/// collection cannot leave this handle dangling.
#[wasm_bindgen]
pub struct {cls} {{
    root: Rc<RefCell<{root}>>,
    at: At<{ty}>,
    see: See<{ty}>,
}}

#[wasm_bindgen]
impl {cls} {{
{with}", with = with_impl(&o)));
    for f in &t.fields {
        field_accessors(out, cx, &o, tn, f)?;
    }
    out.push_str("}\n");
    for f in &t.fields {
        if f.kind == "collection" {
            collection_class(out, cx, tn, f)?;
        }
    }
    Ok(())
}

/// The solver surface of one root: config, options, result, session,
/// covariance.
fn solver_classes(out: &mut String, cx: &Ctx) -> Result<(), String> {
    let fp = cx.fp;
    let p = &cx.prefix;
    let root_cls = cx.cls(&cx.model.root);
    let root_ty = cx.root();
    let cfg = format!("{p}LmConfig");
    let opts = format!("{p}SparseOptions");
    let res = format!("{p}LmResult");
    let sess = format!("{p}LmSession");
    let cov = format!("{p}Covariance");
    out.push_str(&format!(
"
/// The solver configuration: a preset's values, edited field by field.
#[wasm_bindgen]
pub struct {cfg} {{
    preset: u32,
    #[wasm_bindgen(js_name = \"maxIters\")]
    pub max_iters: u32,
    #[wasm_bindgen(js_name = \"minIters\")]
    pub min_iters: u32,
    pub patience: u32,
    #[wasm_bindgen(js_name = \"numThreads\")]
    pub num_threads: u32,
    pub verbose: bool,
    #[wasm_bindgen(js_name = \"gatherTiming\")]
    pub gather_timing: bool,
    #[wasm_bindgen(js_name = \"absPrecision\")]
    pub abs_precision: f64,
    #[wasm_bindgen(js_name = \"relPrecision\")]
    pub rel_precision: f64,
    #[wasm_bindgen(js_name = \"initialLambda\")]
    pub initial_lambda: f64,
    #[wasm_bindgen(js_name = \"costThreshold\")]
    pub cost_threshold: f64,
    #[wasm_bindgen(js_name = \"lambdaFloor\")]
    pub lambda_floor: f64,
    gradient_tolerance: Option<f64>,
    parameter_tolerance: Option<f64>,
    predicted_reduction_tolerance: Option<f64>,
    min_diagonal: Option<f64>,
    time_limit_seconds: Option<f64>,
    assembly_threads: Option<u32>,
}}

#[wasm_bindgen]
impl {cfg} {{
    fn from_preset(preset: u32) -> {cfg} {{
        let c = preset_config(preset);
        {cfg} {{
            preset,
            max_iters: c.max_iters as u32,
            min_iters: c.min_iters as u32,
            patience: c.patience as u32,
            num_threads: c.num_threads as u32,
            verbose: c.verbose,
            gather_timing: c.gather_timing,
            abs_precision: c.abs_precision as f64,
            rel_precision: c.rel_precision as f64,
            initial_lambda: c.initial_lambda as f64,
            cost_threshold: c.cost_threshold as f64,
            lambda_floor: c.lambda_floor as f64,
            gradient_tolerance: c.gradient_tolerance.map(|v| v as f64),
            parameter_tolerance: c.parameter_tolerance.map(|v| v as f64),
            predicted_reduction_tolerance: c.predicted_reduction_tolerance.map(|v| v as f64),
            min_diagonal: c.min_diagonal.map(|v| v as f64),
            time_limit_seconds: c.time_limit.map(|d| d.as_secs_f64()),
            assembly_threads: c.assembly_threads.map(|n| n as u32),
        }}
    }}
    fn to_config(&self) -> arael::simple_lm::LmConfig<{fp}> {{
        let mut c = preset_config(self.preset);
        c.max_iters = self.max_iters as usize;
        c.min_iters = self.min_iters as usize;
        c.patience = self.patience as usize;
        c.num_threads = self.num_threads as usize;
        c.assembly_threads = self.assembly_threads.map(|n| n as usize);
        c.verbose = self.verbose;
        c.gather_timing = self.gather_timing;
        c.abs_precision = self.abs_precision as {fp};
        c.rel_precision = self.rel_precision as {fp};
        c.initial_lambda = self.initial_lambda as {fp};
        c.cost_threshold = self.cost_threshold as {fp};
        c.lambda_floor = self.lambda_floor as {fp};
        c.gradient_tolerance = self.gradient_tolerance.map(|v| v as {fp});
        c.parameter_tolerance = self.parameter_tolerance.map(|v| v as {fp});
        c.predicted_reduction_tolerance = self.predicted_reduction_tolerance.map(|v| v as {fp});
        c.min_diagonal = self.min_diagonal.map(|v| v as {fp});
        c.time_limit = self.time_limit_seconds.map(std::time::Duration::from_secs_f64);
        c
    }}
    /// The defaults.
    #[wasm_bindgen(constructor)]
    pub fn new() -> {cfg} {{ Self::from_preset(0) }}
    pub fn defaults() -> {cfg} {{ Self::from_preset(0) }}
    pub fn conservative() -> {cfg} {{ Self::from_preset(1) }}
    #[wasm_bindgen(js_name = \"wellConditioned\")]
    pub fn well_conditioned() -> {cfg} {{ Self::from_preset(2) }}
    #[wasm_bindgen(js_name = \"illConditioned\")]
    pub fn ill_conditioned() -> {cfg} {{ Self::from_preset(3) }}
    #[wasm_bindgen(getter, js_name = \"gradientTolerance\")]
    pub fn gradient_tolerance(&self) -> Option<f64> {{ self.gradient_tolerance }}
    #[wasm_bindgen(setter, js_name = \"gradientTolerance\")]
    pub fn set_gradient_tolerance(&mut self, v: Option<f64>) {{ self.gradient_tolerance = v; }}
    #[wasm_bindgen(getter, js_name = \"parameterTolerance\")]
    pub fn parameter_tolerance(&self) -> Option<f64> {{ self.parameter_tolerance }}
    #[wasm_bindgen(setter, js_name = \"parameterTolerance\")]
    pub fn set_parameter_tolerance(&mut self, v: Option<f64>) {{ self.parameter_tolerance = v; }}
    #[wasm_bindgen(getter, js_name = \"predictedReductionTolerance\")]
    pub fn predicted_reduction_tolerance(&self) -> Option<f64> {{ self.predicted_reduction_tolerance }}
    #[wasm_bindgen(setter, js_name = \"predictedReductionTolerance\")]
    pub fn set_predicted_reduction_tolerance(&mut self, v: Option<f64>) {{ self.predicted_reduction_tolerance = v; }}
    #[wasm_bindgen(getter, js_name = \"minDiagonal\")]
    pub fn min_diagonal(&self) -> Option<f64> {{ self.min_diagonal }}
    #[wasm_bindgen(setter, js_name = \"minDiagonal\")]
    pub fn set_min_diagonal(&mut self, v: Option<f64>) {{ self.min_diagonal = v; }}
    #[wasm_bindgen(getter, js_name = \"timeLimitSeconds\")]
    pub fn time_limit_seconds(&self) -> Option<f64> {{ self.time_limit_seconds }}
    #[wasm_bindgen(setter, js_name = \"timeLimitSeconds\")]
    pub fn set_time_limit_seconds(&mut self, v: Option<f64>) {{ self.time_limit_seconds = v; }}
    #[wasm_bindgen(getter, js_name = \"assemblyThreads\")]
    pub fn assembly_threads(&self) -> Option<u32> {{ self.assembly_threads }}
    #[wasm_bindgen(setter, js_name = \"assemblyThreads\")]
    pub fn set_assembly_threads(&mut self, v: Option<u32>) {{ self.assembly_threads = v; }}
}}

/// The sparse backend's options as plain data, starting from the Rust
/// defaults. The enum fields carry the tags of the C ABI: schur 0 Auto,
/// 1 Force, 2 Never; ordering 0 Auto, 1 Amd, 2 MarginalizeFirst,
/// 3 Natural, 4 NestedDissection; envelope 0 Auto, 1 Always, 2 Never;
/// schurSolve 0 Factorize, 1 Iterative, 2 IterativeImplicit;
/// blockSupernodal 0 Auto, 1 Always, 2 Never.
#[wasm_bindgen]
pub struct {opts} {{
    pub schur: u32,
    pub ordering: u32,
    pub envelope: u32,
    #[wasm_bindgen(js_name = \"envelopePanelWidth\")]
    pub envelope_panel_width: u32,
    pub supernodal: bool,
    #[wasm_bindgen(js_name = \"narrowBand\")]
    pub narrow_band: bool,
    #[wasm_bindgen(js_name = \"flopMargin\")]
    pub flop_margin: f64,
    #[wasm_bindgen(js_name = \"obviousFlopRatio\")]
    pub obvious_flop_ratio: f64,
    #[wasm_bindgen(js_name = \"cgTol\")]
    pub cg_tol: f64,
    #[wasm_bindgen(js_name = \"schurSolve\")]
    pub schur_solve: u32,
    #[wasm_bindgen(js_name = \"cgMaxIters\")]
    pub cg_max_iters: u32,
    #[wasm_bindgen(js_name = \"cgRestartEvery\")]
    pub cg_restart_every: u32,
    #[wasm_bindgen(js_name = \"blockSupernodal\")]
    pub block_supernodal: u32,
    #[wasm_bindgen(js_name = \"blockSupernodalBatch\")]
    pub block_supernodal_batch: f64,
    #[wasm_bindgen(js_name = \"blockSupernodalMemoryLean\")]
    pub block_supernodal_memory_lean: bool,
}}

#[wasm_bindgen]
impl {opts} {{
    /// The Rust defaults.
    #[wasm_bindgen(constructor)]
    pub fn new() -> {opts} {{
        use arael::simple_lm::{{BlockSupernodalMode, EnvelopeMode, FaerOrdering, SchurPolicy}};
        let d = SparseFaerOptions::default();
        let (flop_margin, obvious_flop_ratio) = match d.policy {{
            SchurPolicy::Auto {{ flop_margin, obvious_flop_ratio }} => (flop_margin, obvious_flop_ratio),
            _ => (0.0, 0.0),
        }};
        {opts} {{
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
            block_supernodal: match d.block_supernodal {{
                BlockSupernodalMode::Auto => 0,
                BlockSupernodalMode::Always => 1,
                BlockSupernodalMode::Never => 2,
            }},
            block_supernodal_batch: d.block_supernodal_batch.unwrap_or(0.0),
            block_supernodal_memory_lean: d.block_supernodal_memory_lean,
        }}
    }}
    fn to_options(&self) -> Result<SparseFaerOptions, JsValue> {{
        use arael::simple_lm::{{BlockSupernodalMode, EnvelopeMode, FaerOrdering, SchurPolicy}};
        let policy = match self.schur {{
            0 => SchurPolicy::Auto {{ flop_margin: self.flop_margin, obvious_flop_ratio: self.obvious_flop_ratio }},
            1 => SchurPolicy::Force,
            2 => SchurPolicy::Never,
            t => return Err(js_err(&format!(\"unknown schur policy tag {{t}}\"))),
        }};
        let ordering = match self.ordering {{
            0 => FaerOrdering::Auto,
            1 => FaerOrdering::Amd,
            2 => FaerOrdering::MarginalizeFirst,
            3 => FaerOrdering::Natural,
            4 => FaerOrdering::NestedDissection,
            t => return Err(js_err(&format!(\"unknown ordering tag {{t}}\"))),
        }};
        let envelope = match self.envelope {{
            0 => EnvelopeMode::Auto,
            1 => EnvelopeMode::Always,
            2 => EnvelopeMode::Never,
            t => return Err(js_err(&format!(\"unknown envelope mode tag {{t}}\"))),
        }};
        let block_supernodal = match self.block_supernodal {{
            0 => BlockSupernodalMode::Auto,
            1 => BlockSupernodalMode::Always,
            2 => BlockSupernodalMode::Never,
            t => return Err(js_err(&format!(\"unknown block supernodal mode tag {{t}}\"))),
        }};
        let batch = self.block_supernodal_batch;
        let width = self.envelope_panel_width;
        let o = SparseFaerOptions::auto()
            .with_policy(policy)
            .with_ordering(ordering)
            .with_envelope_schur(envelope)
            .with_envelope_panel_width((width > 0).then_some(width as usize))
            .with_supernodal(self.supernodal)
            .with_narrow_band(self.narrow_band)
            .with_block_supernodal(block_supernodal)
            .with_block_supernodal_batching((batch > 0.0).then_some(batch))
            .with_block_supernodal_memory_lean(self.block_supernodal_memory_lean);
        let cg = arael::simple_lm::CgOptions {{
            tol: self.cg_tol,
            max_iters: self.cg_max_iters as usize,
            restart_every: self.cg_restart_every as usize,
        }};
        Ok(match self.schur_solve {{
            0 => o,
            1 => o.with_iterative_schur(cg),
            2 => o.with_implicit_schur(cg),
            t => return Err(js_err(&format!(\"unknown schur solve tag {{t}}\"))),
        }})
    }}
}}

/// A completed solve. `status` is the C ABI code (0 Converged, 1
/// CostThreshold, 2 MaxIterations, 3 GradientTolerance, 4
/// ParameterTolerance, 5 PredictedReduction, 6 LambdaCeiling, 7
/// DriverTerminated, 8 ObserverTerminated, 9 TimeLimit, 10
/// RetryBudgetExhausted, 11 Aborted); `statusName` spells it.
#[wasm_bindgen]
pub struct {res} {{
    r: arael::simple_lm::LmResult<{fp}>,
}}

#[wasm_bindgen]
impl {res} {{
    #[wasm_bindgen(getter, js_name = \"startCost\")]
    pub fn start_cost(&self) -> f64 {{ self.r.start_cost as f64 }}
    #[wasm_bindgen(getter, js_name = \"endCost\")]
    pub fn end_cost(&self) -> f64 {{ self.r.end_cost as f64 }}
    #[wasm_bindgen(getter)]
    pub fn iterations(&self) -> u32 {{ self.r.iterations as u32 }}
    #[wasm_bindgen(getter, js_name = \"acceptedIterations\")]
    pub fn accepted_iterations(&self) -> u32 {{ self.r.accepted_iterations as u32 }}
    #[wasm_bindgen(getter)]
    pub fn status(&self) -> i32 {{ status_code(&self.r.status) }}
    #[wasm_bindgen(getter, js_name = \"statusName\")]
    pub fn status_name(&self) -> String {{ format!(\"{{:?}}\", self.r.status) }}
    /// Did the solve reach a minimum, as opposed to running out of
    /// something?
    #[wasm_bindgen(getter, js_name = \"isSuccess\")]
    pub fn is_success(&self) -> bool {{ self.r.status.is_success() }}
    #[wasm_bindgen(getter, js_name = \"finalLambda\")]
    pub fn final_lambda(&self) -> f64 {{ self.r.final_lambda as f64 }}
    /// The report text, plain.
    pub fn report(&self) -> String {{ self.r.report() }}
    /// The report text with colour and glyphs.
    #[wasm_bindgen(js_name = \"prettyReport\")]
    pub fn pretty_report(&self) -> String {{ self.r.pretty_report() }}
    /// Per-phase seconds and call counts, or undefined when the solve
    /// ran without `gatherTiming`.
    pub fn timing(&self) -> JsValue {{
        match &self.r.timing {{
            Some(t) => js_obj(&[
                (\"total\", t.total.as_secs_f64()),
                (\"assembly\", t.assembly.as_secs_f64()),
                (\"firstAssembly\", t.first_assembly.as_secs_f64()),
                (\"analysis\", t.analysis.as_secs_f64()),
                (\"linearSolve\", t.linear_solve.as_secs_f64()),
                (\"firstLinearSolve\", t.first_linear_solve.as_secs_f64()),
                (\"costEval\", t.cost_eval.as_secs_f64()),
                (\"firstCostEval\", t.first_cost_eval.as_secs_f64()),
                (\"advance\", t.advance.as_secs_f64()),
                (\"firstAdvance\", t.first_advance.as_secs_f64()),
                (\"assemblyCount\", t.assembly_count as f64),
                (\"analysisCount\", t.analysis_count as f64),
                (\"linearSolveCount\", t.linear_solve_count as f64),
                (\"costEvalCount\", t.cost_eval_count as f64),
                (\"advanceCount\", t.advance_count as f64),
            ]),
            None => JsValue::UNDEFINED,
        }}
    }}
    /// The sparse backend's plan, or undefined when the solve carried
    /// none. Absent statistics are undefined; `ordering` is 0
    /// NaturalBanded, 1 NaturalDense, 2 Amd, 3 Nd.
    pub fn plan(&self) -> JsValue {{
        use arael::simple_lm::ReducedOrdering;
        let p = match &self.r.solver {{
            Some(arael::simple_lm::SolverReport::Schur(p)) => p,
            _ => return JsValue::UNDEFINED,
        }};
        let o = js_sys::Object::new();
        js_set(&o, \"reduced\", &JsValue::from_bool(p.reduced));
        js_set(&o, \"eliminatedBlocks\", &JsValue::from_f64(p.eliminated_blocks as f64));
        js_set(&o, \"eliminatedParams\", &JsValue::from_f64(p.eliminated_params as f64));
        js_set(&o, \"keptParams\", &JsValue::from_f64(p.kept_params as f64));
        js_set(&o, \"fillRatio\", &js_opt(p.fill_ratio));
        js_set(&o, \"routeFlops\", &match p.route_flops {{
            Some((reduced, full)) => js_obj(&[(\"reduced\", reduced), (\"full\", full)]),
            None => JsValue::UNDEFINED,
        }});
        js_set(&o, \"cgIterations\", &js_opt(p.cg_iterations.map(|n| n as f64)));
        js_set(&o, \"flopRatio\", &js_opt(p.flop_ratio));
        js_set(&o, \"ordering\", &js_opt(p.ordering.map(|o| match o {{
            ReducedOrdering::NaturalBanded => 0.0,
            ReducedOrdering::NaturalDense => 1.0,
            ReducedOrdering::Amd => 2.0,
            ReducedOrdering::Nd => 3.0,
        }})));
        js_set(&o, \"keptBandwidth\", &JsValue::from_f64(p.kept_bandwidth as f64));
        js_set(&o, \"envelope\", &JsValue::from_bool(p.envelope));
        js_set(&o, \"blockSupernodal\", &JsValue::from_bool(p.block_supernodal));
        o.into()
    }}
    /// The per-attempt timeline, one object per attempted step; empty
    /// without `gatherTiming`.
    pub fn steps(&self) -> JsValue {{
        let arr = js_sys::Array::new();
        if let Some(t) = &self.r.timing {{
            for s in &t.steps {{
                let o = js_sys::Object::new();
                js_set(&o, \"iter\", &JsValue::from_f64(s.iter as f64));
                js_set(&o, \"inner\", &JsValue::from_f64(s.inner as f64));
                js_set(&o, \"accepted\", &JsValue::from_bool(s.accepted));
                js_set(&o, \"factorizationFailed\", &JsValue::from_bool(s.factorization_failed));
                js_set(&o, \"lambda\", &JsValue::from_f64(s.lambda));
                js_set(&o, \"cost\", &JsValue::from_f64(s.cost));
                js_set(&o, \"newCost\", &JsValue::from_f64(s.new_cost));
                js_set(&o, \"stepNorm\", &JsValue::from_f64(s.step_norm));
                js_set(&o, \"gradMax\", &JsValue::from_f64(s.grad_max));
                js_set(&o, \"time\", &JsValue::from_f64(s.time.as_secs_f64()));
                js_set(&o, \"assembly\", &JsValue::from_f64(s.assembly.as_secs_f64()));
                js_set(&o, \"analysis\", &JsValue::from_f64(s.analysis.as_secs_f64()));
                js_set(&o, \"linearSolve\", &JsValue::from_f64(s.linear_solve.as_secs_f64()));
                js_set(&o, \"costEval\", &JsValue::from_f64(s.cost_eval.as_secs_f64()));
                js_set(&o, \"advance\", &JsValue::from_f64(s.advance.as_secs_f64()));
                arr.push(&o);
            }}
        }}
        arr.into()
    }}
}}

/// A warm re-solve: keeps the backend and what it learns about one
/// problem's structure across solves. Call `invalidate` after any
/// structural change to the model.
#[wasm_bindgen]
pub struct {sess} {{
    s: arael::simple_lm::LmSession<{fp}, SparseFaer<{fp}>>,
}}

#[wasm_bindgen]
impl {sess} {{
    /// A session over the sparse backend, with its defaults or `opts`.
    #[wasm_bindgen(constructor)]
    pub fn new(opts: Option<{opts}>) -> Result<{sess}, JsValue> {{
        let backend = match opts {{
            None => SparseFaer::<{fp}>::new(),
            Some(o) => SparseFaer::<{fp}>::from_options(&o.to_options()?),
        }};
        Ok({sess} {{ s: arael::simple_lm::LmSession::new(backend) }})
    }}
    /// As `solveSparse`, through the session's cached analysis.
    pub fn solve(&mut self, model: &{root_cls}, cfg: &{cfg}) -> Result<{res}, JsValue> {{
        let c = cfg.to_config();
        let mut g = model.root.borrow_mut();
        self.s.solve(&mut *g, &c).map(|r| {res} {{ r }}).map_err(failure)
    }}
    /// Drop the learned structure; the next solve runs cold.
    pub fn invalidate(&mut self) {{ self.s.invalidate(); }}
}}

/// The parameter covariance at the solution, `Sigma = 2 H^-1`, queried
/// per entity.
#[wasm_bindgen]
pub struct {cov} {{
    cov: CovAssembly,
    root: Rc<RefCell<{root_ty}>>,
}}

#[wasm_bindgen]
impl {cov} {{
    /// What the assembly decided: `ordering` 1 Amd, 2 NestedDissection,
    /// 3 Natural; `candidateFlops` when Auto priced them; the symbolic
    /// analyses built; whether the block route ran.
    pub fn plan(&self) -> JsValue {{
        let p = self.cov.plan();
        let o = js_sys::Object::new();
        js_set(&o, \"ordering\", &JsValue::from_f64(match p.ordering {{
            CovOrdering::Auto => 0.0,
            CovOrdering::Amd => 1.0,
            CovOrdering::NestedDissection => 2.0,
            CovOrdering::Natural => 3.0,
        }}));
        js_set(&o, \"candidateFlops\", &match p.candidate_flops {{
            Some((amd, nd)) => js_obj(&[(\"amd\", amd), (\"nd\", nd)]),
            None => JsValue::UNDEFINED,
        }});
        js_set(&o, \"symbolicsBuilt\", &JsValue::from_f64(p.symbolics_built as f64));
        js_set(&o, \"blockRoute\", &JsValue::from_bool(self.cov.took_block_route()));
        o.into()
    }}
"));
    // Per entity type with parameters: marginal, conditional, std_dev;
    // per pair: cross. Read through the handles' shared resolvers.
    let mut ents: Vec<(&String, &Type)> = cx.model.types.iter()
        .filter(|(tn, t)| t.role == "entity" && t.param_count > 0 && **tn != cx.model.root)
        .collect();
    ents.sort_by(|a, b| a.0.cmp(b.0));
    for (tn, _) in &ents {
        let ecls = cx.cls(tn);
        out.push_str(&format!(
"    /// Row-major dim x dim marginal covariance of one `{tn}`.
    #[wasm_bindgen(js_name = \"marginal{tn}\")]
    pub fn marginal_{sn}(&self, e: &{ecls}) -> Result<js_sys::Float64Array, JsValue> {{
        let g = self.root.borrow();
        let x = (e.see)(&*g).ok_or_else(stale)?;
        let m = self.cov.marginal_cov(x).map_err(|e| js_err(&e.to_string()))?;
        Ok(dense_to_js(m.nrows(), m.ncols(), |i, j| m[(i, j)]))
    }}
    /// Row-major dim x dim conditional covariance of one `{tn}`.
    #[wasm_bindgen(js_name = \"conditional{tn}\")]
    pub fn conditional_{sn}(&self, e: &{ecls}) -> Result<js_sys::Float64Array, JsValue> {{
        let g = self.root.borrow();
        let x = (e.see)(&*g).ok_or_else(stale)?;
        let m = self.cov.conditional_cov(x).map_err(|e| js_err(&e.to_string()))?;
        Ok(dense_to_js(m.nrows(), m.ncols(), |i, j| m[(i, j)]))
    }}
    /// Per-parameter standard deviations of one `{tn}`.
    #[wasm_bindgen(js_name = \"stdDev{tn}\")]
    pub fn std_dev_{sn}(&self, e: &{ecls}) -> Result<js_sys::Float64Array, JsValue> {{
        let g = self.root.borrow();
        let x = (e.see)(&*g).ok_or_else(stale)?;
        let sd = self.cov.std_dev(x).map_err(|e| js_err(&e.to_string()))?;
        Ok(js_sys::Float64Array::from(&sd[..]))
    }}
", sn = snake(tn)));
    }
    for (an, _) in &ents {
        for (bn, _) in &ents {
            let (acls, bcls) = (cx.cls(an), cx.cls(bn));
            out.push_str(&format!(
"    /// Row-major cross covariance between one `{an}` and one `{bn}`.
    #[wasm_bindgen(js_name = \"cross{an}{bn}\")]
    pub fn cross_{asn}_{bsn}(&self, a: &{acls}, b: &{bcls}) -> Result<js_sys::Float64Array, JsValue> {{
        let g = self.root.borrow();
        let xa = (a.see)(&*g).ok_or_else(stale)?;
        let xb = (b.see)(&*g).ok_or_else(stale)?;
        let m = self.cov.cross_cov(xa, xb).map_err(|e| js_err(&e.to_string()))?;
        Ok(dense_to_js(m.nrows(), m.ncols(), |i, j| m[(i, j)]))
    }}
", asn = snake(an), bsn = snake(bn)));
        }
    }
    out.push_str("}\n");
    Ok(())
}

/// The root class: its own fields, the solve entries and the
/// covariance assembly.
fn root_class(out: &mut String, cx: &Ctx) -> Result<(), String> {
    let model = cx.model;
    let root = &model.root;
    let cls = cx.cls(root);
    let ty = cx.root();
    let fp = cx.fp;
    let p = &cx.prefix;
    let band_fn = if fp == "f32" { "solve_band_f32" } else { "solve_band" };
    let t = model.types.get(root).ok_or("root type missing from the sidecar")?;
    let o = Owner { ty: ty.clone(), is_root: true };
    out.push_str(&format!(
"
/// The `{root}` model and its solver entries.
#[wasm_bindgen]
pub struct {cls} {{
    root: Rc<RefCell<{ty}>>,
}}

#[wasm_bindgen]
impl {cls} {{
    #[wasm_bindgen(constructor)]
    pub fn new() -> {cls} {{
        {cls} {{ root: Rc::new(RefCell::new(Default::default())) }}
    }}
{with}    /// The cost at the current parameters.
    pub fn cost(&self) -> f64 {{
        let mut g = self.root.borrow_mut();
        let mut params = Vec::new();
        g.serialize(&mut params);
        g.calc_cost(&params) as f64
    }}
    /// The model's diagnostics as text; empty when clean.
    pub fn validate(&self) -> String {{
        let mut g = self.root.borrow_mut();
        let d = g.validate();
        if d.is_clean() {{ String::new() }} else {{ d.to_string() }}
    }}
    #[wasm_bindgen(js_name = \"solveDense\")]
    pub fn solve_dense(&self, cfg: &{p}LmConfig) -> Result<{p}LmResult, JsValue> {{
        let c = cfg.to_config();
        let mut g = self.root.borrow_mut();
        g.solve_dense(&c).map(|r| {p}LmResult {{ r }}).map_err(failure)
    }}
    /// The sparse solve, with the backend's defaults or `opts`.
    #[wasm_bindgen(js_name = \"solveSparse\")]
    pub fn solve_sparse(&self, cfg: &{p}LmConfig, opts: Option<{p}SparseOptions>) -> Result<{p}LmResult, JsValue> {{
        let c = cfg.to_config();
        let mut g = self.root.borrow_mut();
        let r = match opts {{
            None => g.solve_sparse(&c),
            Some(o) => {{
                let mut s = SparseFaer::<{fp}>::from_options(&o.to_options()?);
                g.solve_with(&mut s, &c)
            }}
        }};
        r.map(|r| {p}LmResult {{ r }}).map_err(failure)
    }}
    /// The band solve, `kd` the half-bandwidth in scalar parameters.
    #[wasm_bindgen(js_name = \"solveBand\")]
    pub fn solve_band(&self, kd: u32, cfg: &{p}LmConfig) -> Result<{p}LmResult, JsValue> {{
        let c = cfg.to_config();
        let mut g = self.root.borrow_mut();
        let mut x0 = Vec::new();
        g.serialize(&mut x0);
        arael::simple_lm::{band_fn}(&x0, kd as usize, &mut *g, &c)
            .map(|r| {{ g.deserialize(&r.x); {p}LmResult {{ r }} }})
            .map_err(failure)
    }}
    /// The covariance at the current parameters; `mode` 0 PerQuery,
    /// 1 AllMarginals (the default), 2 TriDiagonal.
    #[wasm_bindgen(js_name = \"assembleCovariance\")]
    pub fn assemble_covariance(&self, mode: Option<u32>) -> Result<{p}Covariance, JsValue> {{
        self.assemble_covariance_with(mode.unwrap_or(1), 0, 0)
    }}
    /// `assembleCovariance` with the assembly spelled out: `ordering` 0
    /// Auto, 1 Amd, 2 NestedDissection, 3 Natural; `blockSupernodal` 0
    /// Auto, 1 Always, 2 Never.
    #[wasm_bindgen(js_name = \"assembleCovarianceWith\")]
    pub fn assemble_covariance_with(&self, mode: u32, ordering: u32, block_supernodal: u32) -> Result<{p}Covariance, JsValue> {{
        let m = match mode {{
            0 => CovMode::PerQuery,
            2 => CovMode::TriDiagonal,
            _ => CovMode::AllMarginals,
        }};
        let opts = CovOptions {{
            ordering: match ordering {{
                1 => CovOrdering::Amd,
                2 => CovOrdering::NestedDissection,
                3 => CovOrdering::Natural,
                _ => CovOrdering::Auto,
            }},
            block_supernodal: match block_supernodal {{
                1 => arael::simple_lm::BlockSupernodalMode::Always,
                2 => arael::simple_lm::BlockSupernodalMode::Never,
                _ => arael::simple_lm::BlockSupernodalMode::Auto,
            }},
        }};
        let mut g = self.root.borrow_mut();
        let cov = g.assemble_covariance_with(m, &opts).map_err(|e| js_err(&format!(\"{{}}\", e)))?;
        Ok({p}Covariance {{ cov, root: self.root.clone() }})
    }}
", with = with_impl(&o)));
    for f in &t.fields {
        field_accessors(out, cx, &o, root, f)?;
    }
    out.push_str("}\n");
    for f in &t.fields {
        if f.kind == "collection" {
            collection_class(out, cx, root, f)?;
        }
    }
    Ok(())
}

/// The helpers every root's classes share, once per crate.
const PRELUDE: &str = r#"
#[wasm_bindgen(start)]
pub fn __arael_start() {
    console_error_panic_hook::set_once();
}

fn js_err(msg: &str) -> JsValue {
    js_sys::Error::new(msg).into()
}

fn stale() -> JsValue {
    js_err("stale handle: the element it addressed is gone")
}

fn js_set(o: &js_sys::Object, key: &str, v: &JsValue) {
    let _ = js_sys::Reflect::set(o, &JsValue::from_str(key), v);
}

fn js_obj(pairs: &[(&str, f64)]) -> JsValue {
    let o = js_sys::Object::new();
    for (k, v) in pairs {
        js_set(&o, k, &JsValue::from_f64(*v));
    }
    o.into()
}

fn js_opt(v: Option<f64>) -> JsValue {
    match v {
        Some(x) => JsValue::from_f64(x),
        None => JsValue::UNDEFINED,
    }
}

fn js_num(v: &JsValue, key: &str) -> Result<f64, JsValue> {
    js_sys::Reflect::get(v, &JsValue::from_str(key))?
        .as_f64()
        .ok_or_else(|| js_err(&format!("`{key}` is not a number")))
}

fn js_at(v: &JsValue, i: usize) -> Result<JsValue, JsValue> {
    js_sys::Reflect::get_u32(v, i as u32)
}

fn js_num_at(v: &JsValue, i: usize) -> Result<f64, JsValue> {
    js_at(v, i)?.as_f64().ok_or_else(|| js_err(&format!("element {i} is not a number")))
}

/// A dense matrix as a row-major typed array.
fn dense_to_js(rows: usize, cols: usize, at: impl Fn(usize, usize) -> f64) -> js_sys::Float64Array {
    let mut v = Vec::with_capacity(rows * cols);
    for i in 0..rows {
        for j in 0..cols {
            v.push(at(i, j));
        }
    }
    js_sys::Float64Array::from(&v[..])
}

/// A scalar that crosses as a JavaScript number.
trait Js: arael::utils::Float {
    fn to(self) -> f64;
    fn of(v: f64) -> Self;
}
impl Js for f64 {
    fn to(self) -> f64 { self }
    fn of(v: f64) -> Self { v }
}
impl Js for f32 {
    fn to(self) -> f64 { self as f64 }
    fn of(v: f64) -> Self { v as f32 }
}

fn vect2_to_js<T: Js>(v: arael::vect::vect2<T>) -> JsValue {
    js_obj(&[("x", v.x.to()), ("y", v.y.to())])
}
fn vect2_from_js<T: Js>(v: &JsValue) -> Result<arael::vect::vect2<T>, JsValue> {
    Ok(arael::vect::vect2::new(T::of(js_num(v, "x")?), T::of(js_num(v, "y")?)))
}
fn vect3_to_js<T: Js>(v: arael::vect::vect3<T>) -> JsValue {
    js_obj(&[("x", v.x.to()), ("y", v.y.to()), ("z", v.z.to())])
}
fn vect3_from_js<T: Js>(v: &JsValue) -> Result<arael::vect::vect3<T>, JsValue> {
    Ok(arael::vect::vect3::new(T::of(js_num(v, "x")?), T::of(js_num(v, "y")?), T::of(js_num(v, "z")?)))
}
/// A quaternion crosses as `{t, x, y, z}`, the scalar part first.
fn quat_to_js<T: Js>(q: arael::quatern::quatern<T>) -> JsValue {
    js_obj(&[("t", q.t.to()), ("x", q.v.x.to()), ("y", q.v.y.to()), ("z", q.v.z.to())])
}
fn quat_from_js<T: Js>(v: &JsValue) -> Result<arael::quatern::quatern<T>, JsValue> {
    Ok(arael::quatern::quatern::new(
        T::of(js_num(v, "t")?),
        arael::vect::vect3::new(T::of(js_num(v, "x")?), T::of(js_num(v, "y")?), T::of(js_num(v, "z")?)),
    ))
}
fn rows_to_js(rows: &[Vec<f64>]) -> JsValue {
    let outer = js_sys::Array::new();
    for r in rows {
        let inner = js_sys::Array::new();
        for x in r {
            inner.push(&JsValue::from_f64(*x));
        }
        outer.push(&inner);
    }
    outer.into()
}
fn rows_from_js(v: &JsValue, r: usize, c: usize) -> Result<Vec<Vec<f64>>, JsValue> {
    let mut out = Vec::with_capacity(r);
    for i in 0..r {
        let row = js_at(v, i)?;
        let mut vals = Vec::with_capacity(c);
        for j in 0..c {
            vals.push(js_num_at(&row, j)?);
        }
        out.push(vals);
    }
    Ok(out)
}
/// A matrix crosses as an array of row arrays.
fn mat2_to_js<T: Js>(m: arael::matrix::matrix2<T>) -> JsValue {
    rows_to_js(&[vec![m.rows[0].x.to(), m.rows[0].y.to()], vec![m.rows[1].x.to(), m.rows[1].y.to()]])
}
fn mat2_from_js<T: Js>(v: &JsValue) -> Result<arael::matrix::matrix2<T>, JsValue> {
    let r = rows_from_js(v, 2, 2)?;
    Ok(arael::matrix::matrix2::from_elements(T::of(r[0][0]), T::of(r[0][1]), T::of(r[1][0]), T::of(r[1][1])))
}
fn mat3_to_js<T: Js>(m: arael::matrix::matrix3<T>) -> JsValue {
    rows_to_js(&[
        vec![m.rows[0].x.to(), m.rows[0].y.to(), m.rows[0].z.to()],
        vec![m.rows[1].x.to(), m.rows[1].y.to(), m.rows[1].z.to()],
        vec![m.rows[2].x.to(), m.rows[2].y.to(), m.rows[2].z.to()],
    ])
}
fn mat3_from_js<T: Js>(v: &JsValue) -> Result<arael::matrix::matrix3<T>, JsValue> {
    let r = rows_from_js(v, 3, 3)?;
    Ok(arael::matrix::matrix3::from_array(std::array::from_fn(|i| std::array::from_fn(|j| T::of(r[i][j])))))
}
fn vecn_to_js<T: Js, const N: usize>(v: arael::vect::vect<T, N>) -> JsValue {
    let arr = js_sys::Array::new();
    for x in v.e {
        arr.push(&JsValue::from_f64(x.to()));
    }
    arr.into()
}
fn vecn_from_js<T: Js, const N: usize>(v: &JsValue) -> Result<arael::vect::vect<T, N>, JsValue> {
    let mut e = [T::of(0.0); N];
    for (i, x) in e.iter_mut().enumerate() {
        *x = T::of(js_num_at(v, i)?);
    }
    Ok(arael::vect::vect { e })
}
fn matn_to_js<T: Js, const R: usize, const C: usize>(m: arael::matrix::matrix<T, R, C>) -> JsValue {
    let rows: Vec<Vec<f64>> = m.rows.iter().map(|r| r.e.iter().map(|x| x.to()).collect()).collect();
    rows_to_js(&rows)
}
fn matn_from_js<T: Js, const R: usize, const C: usize>(v: &JsValue) -> Result<arael::matrix::matrix<T, R, C>, JsValue> {
    let r = rows_from_js(v, R, C)?;
    Ok(arael::matrix::matrix {
        rows: std::array::from_fn(|i| arael::vect::vect { e: std::array::from_fn(|j| T::of(r[i][j])) }),
    })
}

/// A value's flat form, for the bulk accessors.
trait Flat {
    fn flat(&self, out: &mut Vec<f64>);
    fn unflat(c: &[f64]) -> Self;
}
impl<T: Js> Flat for arael::vect::vect2<T> {
    fn flat(&self, out: &mut Vec<f64>) { out.extend([self.x.to(), self.y.to()]); }
    fn unflat(c: &[f64]) -> Self { arael::vect::vect2::new(T::of(c[0]), T::of(c[1])) }
}
impl<T: Js> Flat for arael::vect::vect3<T> {
    fn flat(&self, out: &mut Vec<f64>) { out.extend([self.x.to(), self.y.to(), self.z.to()]); }
    fn unflat(c: &[f64]) -> Self { arael::vect::vect3::new(T::of(c[0]), T::of(c[1]), T::of(c[2])) }
}
impl<T: Js> Flat for arael::quatern::quatern<T> {
    fn flat(&self, out: &mut Vec<f64>) { out.extend([self.t.to(), self.v.x.to(), self.v.y.to(), self.v.z.to()]); }
    fn unflat(c: &[f64]) -> Self {
        arael::quatern::quatern::new(T::of(c[0]), arael::vect::vect3::new(T::of(c[1]), T::of(c[2]), T::of(c[3])))
    }
}
impl<T: Js> Flat for arael::matrix::matrix2<T> {
    fn flat(&self, out: &mut Vec<f64>) {
        out.extend([self.rows[0].x.to(), self.rows[0].y.to(), self.rows[1].x.to(), self.rows[1].y.to()]);
    }
    fn unflat(c: &[f64]) -> Self {
        arael::matrix::matrix2::from_elements(T::of(c[0]), T::of(c[1]), T::of(c[2]), T::of(c[3]))
    }
}
impl<T: Js> Flat for arael::matrix::matrix3<T> {
    fn flat(&self, out: &mut Vec<f64>) {
        for r in &self.rows {
            out.extend([r.x.to(), r.y.to(), r.z.to()]);
        }
    }
    fn unflat(c: &[f64]) -> Self {
        arael::matrix::matrix3::from_array(std::array::from_fn(|i| std::array::from_fn(|j| T::of(c[i * 3 + j]))))
    }
}
impl<T: Js, const N: usize> Flat for arael::vect::vect<T, N> {
    fn flat(&self, out: &mut Vec<f64>) { out.extend(self.e.iter().map(|x| x.to())); }
    fn unflat(c: &[f64]) -> Self { arael::vect::vect { e: std::array::from_fn(|i| T::of(c[i])) } }
}
impl<T: Js, const R: usize, const C: usize> Flat for arael::matrix::matrix<T, R, C> {
    fn flat(&self, out: &mut Vec<f64>) {
        for r in &self.rows {
            out.extend(r.e.iter().map(|x| x.to()));
        }
    }
    fn unflat(c: &[f64]) -> Self {
        arael::matrix::matrix {
            rows: std::array::from_fn(|i| arael::vect::vect { e: std::array::from_fn(|j| T::of(c[i * C + j])) }),
        }
    }
}

fn status_code(s: &LmStatus) -> i32 {
    match s {
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
    }
}
"#;

/// The generated crate's `src/lib.rs` over one or more roots of one
/// model crate.
pub fn emit(models: &[&Model], model_crate: &str) -> Result<String, String> {
    let multi = models.len() > 1;
    let roots: Vec<String> = models.iter().map(|m| format!("`{}`", m.root)).collect();
    let mut out = format!(
"// GENERATED by cargo-arael from the {} model sidecar{}. Do not edit;
// regenerate with `cargo arael export` (check drift with `cargo arael check`).
#![allow(clippy::all)]
#![allow(dead_code, unused_imports, unused_variables)]

use std::cell::RefCell;
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use arael::covariance::{{CovAssembly, CovMode, CovOptions, CovOrdering, Covariance as _}};
use arael::simple_lm::{{LmProblem, LmStatus, RootProblem, SparseFaer, SparseFaerOptions}};
use {model_crate} as m;
", roots.join(" / "), if multi { "s" } else { "" });
    out.push_str(PRELUDE);
    for model in models {
        let cx = Ctx {
            model,
            prefix: if multi { model.root.clone() } else { String::new() },
            fp: &model.precision,
        };
        emit_root(&mut out, &cx)?;
    }
    Ok(out)
}

fn emit_root(out: &mut String, cx: &Ctx) -> Result<(), String> {
    let root = cx.root();
    let sn = snake(&cx.model.root);
    out.push_str(&format!(
"
// ---------------------------------------------------------------- {}
pub mod {sn} {{
use super::*;

/// A handle's way to its element: through the model, mutably.
type At<T> = Rc<dyn for<'a> Fn(&'a mut {root}) -> Option<&'a mut T>>;
/// The same, shared.
type See<T> = Rc<dyn for<'a> Fn(&'a {root}) -> Option<&'a T>>;

fn mk_at<T>(f: impl for<'a> Fn(&'a mut {root}) -> Option<&'a mut T> + 'static) -> At<T> {{
    Rc::new(f)
}}
fn mk_see<T>(f: impl for<'a> Fn(&'a {root}) -> Option<&'a T> + 'static) -> See<T> {{
    Rc::new(f)
}}

fn preset_config(preset: u32) -> arael::simple_lm::LmConfig<{fp}> {{
    match preset {{
        1 => arael::simple_lm::LmConfig::conservative(),
        2 => arael::simple_lm::LmConfig::well_conditioned(),
        3 => arael::simple_lm::LmConfig::ill_conditioned(),
        _ => arael::simple_lm::LmConfig::default(),
    }}
}}

fn failure(f: arael::simple_lm::SolveFailure<{fp}>) -> JsValue {{
    js_err(&f.to_string())
}}
", cx.model.root, fp = cx.fp));
    root_class(out, cx)?;
    for (tn, t) in surfaced(cx.model) {
        type_class(out, cx, tn, t)?;
    }
    solver_classes(out, cx)?;
    out.push_str(&format!("\n}} // mod {sn}\n"));
    Ok(())
}

/// The generated crate's manifest, written once. `arael_dep` is the
/// model's arael dependency as seen from `wasm/`, one level below the
/// model crate like `capi/`.
pub fn manifest(crate_name: &str, arael_dep: &str) -> String {
    format!(
"# {marker}: wasm-bindgen crate for the `{crate_name}` model. Written once -- edit freely; delete the file to regenerate.
#
# Build:
#   cargo build --release --target wasm32-unknown-unknown
#   wasm-bindgen --target web --weak-refs --out-dir pkg \\
#       target/wasm32-unknown-unknown/release/{ident}_wasm.wasm
# The wasm-bindgen CLI must be the version pinned below:
#   cargo install wasm-bindgen-cli --version {wb}
[package]
name = \"{crate_name}-wasm\"
version = \"0.1.0\"
edition = \"2021\"

# Its own workspace: built for wasm32 on its own, apart from the model
# workspace's native builds.
[workspace]

[lib]
crate-type = [\"cdylib\", \"rlib\"]

[dependencies]
{crate_name} = {{ path = \"..\" }}
arael = {arael_dep}
wasm-bindgen = \"={wb}\"
js-sys = \"0.3\"
console_error_panic_hook = \"0.1\"

[profile.release]
opt-level = 3
codegen-units = 1

# The arael macro crates optimized in every profile, so the model's
# code generation does not dominate the build.
[profile.dev.package.arael-macros]
opt-level = 3
[profile.dev.package.arael-sym]
opt-level = 3
[profile.release.package.arael-macros]
opt-level = 3
[profile.release.package.arael-sym]
opt-level = 3
",
        marker = crate::export::MARKER,
        ident = crate_name.replace('-', "_"),
        wb = WASM_BINDGEN_VERSION)
}
