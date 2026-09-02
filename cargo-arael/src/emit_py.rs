//! Python module emitter: pure-ctypes bindings over the same C ABI
//! the C++ header wraps, as two files per root -- `_{root}_ffi.py`
//! (the signature table) and `{root}.py` (the API classes). Mirrors
//! the C++ classes one-to-one, idiomatic where Python is: fields are
//! properties, collections speak len/[]/iteration, absent options are
//! None, and failures raise AraelError.

use crate::emit_ffi::surfaced_types;
use crate::ir::{Field, Model, Type, snake};
use crate::leaves::{Leaf, LeafTy, leaves, mask_words, record_slots};

fn ct_scalar(of: &str) -> Option<&'static str> {
    Some(match of {
        "f64" => "ctypes.c_double",
        "f32" => "ctypes.c_float",
        "bool" => "ctypes.c_bool",
        "u32" => "ctypes.c_uint32",
        "i32" => "ctypes.c_int32",
        _ => return None,
    })
}

fn ct_math(of: &str) -> Option<String> {
    Some(match of {
        "vect2f" => "_m.vect2f".to_string(),
        "vect2d" => "_m.vect2d".to_string(),
        "vect3f" => "_m.vect3f".to_string(),
        "vect3d" => "_m.vect3d".to_string(),
        "matrix2f" => "_m.matrix2f".to_string(),
        "matrix2d" => "_m.matrix2d".to_string(),
        "matrix3f" => "_m.matrix3f".to_string(),
        "matrix3d" => "_m.matrix3d".to_string(),
        "quaternf" => "_m.quaternf".to_string(),
        "quaternd" => "_m.quaternd".to_string(),
        _ => {
            // N-dimensional instantiations resolve through the cached
            // ctypes class factories in math.py.
            let (scalar, dims) = crate::ir::ndim_math(of)?;
            let sfx = if scalar == "f32" { "f" } else { "d" };
            return Some(match dims.len() {
                1 => format!("_m.vectn{}({})", sfx, dims[0]),
                _ => format!("_m.matrixn{}({}, {})", sfx, dims[0], dims[1]),
            });
        }
    })
}

/// (ctypes type, needs sequence coercion) of a get/set field.
fn value_ct(f: &Field) -> Option<(String, bool)> {
    let of = f.of.as_deref().unwrap_or("");
    match f.kind.as_str() {
        "data" | "param" => ct_scalar(of).map(|t| (t.to_string(), false))
            .or_else(|| ct_math(of).map(|t| (t, true))),
        "euler_param" => {
            let s = f.scalar.as_deref().unwrap_or("f64");
            let t = match (f.variant.as_deref().unwrap_or("simple"), s) {
                ("rotvec", "f32") => "_m.quaternf",
                ("rotvec", _) => "_m.quaternd",
                (_, "f32") => "_m.vect3f",
                (_, _) => "_m.vect3d",
            };
            Some((t.to_string(), true))
        }
        _ => None,
    }
}

struct Py {
    sigs: String,
    body: String,
}

fn sig(py: &mut Py, name: &str, args: &[&str], res: &str) {
    py.sigs.push_str(&format!("    (\"{name}\", [{}], {res}),\n", args.join(", ")));
}

/// A property over a get/set pair, with optional sequence coercion.
/// `sym` is the C symbol tail (`{sym}` / `set_{sym}` suffixed onto
/// `{prefix}_`), `pyname` the property name.
fn prop(py: &mut Py, cls: &mut String, prefix: &str, sym: &str, pyname: &str,
        ct: &str, coerce: bool) {
    sig(py, &format!("{prefix}_{sym}"), &["ctypes.c_void_p"], ct);
    sig(py, &format!("{prefix}_set_{sym}"), &["ctypes.c_void_p", ct], "None");
    let make = if coerce {
        format!("v if isinstance(v, {ct}) else {ct}(v)")
    } else {
        "v".to_string()
    };
    cls.push_str(&format!(
"    @property
    def {pyname}(self):
        return _f.{prefix}_{sym}(self._p)

    @{pyname}.setter
    def {pyname}(self, v):
        _f.{prefix}_set_{sym}(self._p, {make})

"));
}

fn optimize_prop(py: &mut Py, cls: &mut String, prefix: &str, name: &str) {
    sig(py, &format!("{prefix}_{name}_optimize"), &["ctypes.c_void_p"],
        "ctypes.c_bool");
    sig(py, &format!("{prefix}_{name}_set_optimize"),
        &["ctypes.c_void_p", "ctypes.c_bool"], "None");
    cls.push_str(&format!(
"    @property
    def {name}_optimize(self):
        return _f.{prefix}_{name}_optimize(self._p)

    @{name}_optimize.setter
    def {name}_optimize(self, v):
        _f.{prefix}_{name}_set_optimize(self._p, bool(v))

"));
}

/// Joins `items` with ", ", breaking lines past `width` columns; the
/// first line starts at column `col`, continuation lines at `cont`.
fn wrap_items(items: &[String], col: usize, width: usize, cont: &str) -> String {
    let mut out = String::new();
    let mut col = col;
    for (k, it) in items.iter().enumerate() {
        if k == 0 {
            out.push_str(it);
            col += it.len();
        } else if col + 2 + it.len() > width {
            out.push_str(",\n");
            out.push_str(cont);
            out.push_str(it);
            col = cont.len() + it.len();
        } else {
            out.push_str(", ");
            out.push_str(it);
            col += 2 + it.len();
        }
    }
    out
}

/// The `push(**fields)` pieces of one element type: the keyword
/// parameter list (`a=None, b=None`), the prelude turning absent
/// keywords into defaults and present ones into mask bits, and the
/// `pack_into` arguments (mask words first).
fn push_parts(lv: &[Leaf], col: usize, cont: &str) -> (String, String, String) {
    let params: Vec<String> = lv.iter().map(|l| format!("{}=None", l.name)).collect();
    let mut prelude = String::new();
    let mut args: Vec<String> = Vec::new();
    let words = mask_words(lv.len());
    if words == 1 {
        args.push("m".to_string());
    } else {
        for w in 0..words {
            args.push(format!("(m >> {}) & 0xFFFFFFFFFFFFFFFF", w * 64));
        }
    }
    for (i, l) in lv.iter().enumerate() {
        let n = &l.name;
        // The coercions stay inline: a helper call per keyword would
        // cost as much as the crossing itself.
        match &l.ty {
            LeafTy::Math { n: k, .. } => prelude.push_str(&format!(
"        if {n} is None: {n} = _Z{k}
        else:
            m |= 1 << {i}; {n} = tuple({n})
            if len({n}) != {k}: {n} = _cols.flat({n}, {k})
")),
            _ => {
                let (dflt, coerce) = match &l.ty {
                    LeafTy::F64 | LeafTy::F32 => ("0.0", String::new()),
                    LeafTy::Bool => ("0", format!("; {n} = 1 if {n} else 0")),
                    LeafTy::Ref => ("0", format!("; {n} = getattr({n}, \"raw\", {n})")),
                    _ => ("0", String::new()),
                };
                prelude.push_str(&format!(
"        if {n} is None: {n} = {dflt}
        else: m |= 1 << {i}{coerce}
"));
            }
        }
        args.push(match l.ty {
            LeafTy::Math { .. } => format!("*{n}"),
            _ => n.clone(),
        });
    }
    (wrap_items(&params, col, 76, cont), prelude, wrap_items(&args, 12, 76, "            "))
}

/// The per-leaf column methods of an index-addressable collection view.
fn column_methods(py: &mut Py, cls: &mut String, prefix: &str, lv: &[Leaf]) {
    for l in lv {
        let leaf = &l.name;
        let code = l.ty.column_code();
        let k = l.ty.slots();
        sig(py, &format!("{prefix}_set_{leaf}_n"),
            &["ctypes.c_void_p", "ctypes.c_uint32", "ctypes.c_void_p",
              "ctypes.c_uint32", "ctypes.c_int64"], "ctypes.c_bool");
        sig(py, &format!("{prefix}_get_{leaf}_n"),
            &["ctypes.c_void_p", "ctypes.c_uint32", "ctypes.c_void_p",
              "ctypes.c_uint32", "ctypes.c_int64"], "ctypes.c_bool");
        let shape = if k == 1 { "an (n,) array".to_string() } else { format!("an (n, {k}) array") };
        cls.push_str(&format!(
"    def _set_{leaf}(self, start, n, v):
        ptr, stride, _keep = _cols.column_in(v, \"{code}\", {k}, n, \"{leaf}\")
        if not _f.{prefix}_set_{leaf}_n(self._p, start, ptr, n, stride):
            raise IndexError(\"{leaf}: %d + %d exceeds the collection\" % (start, n))

    def set_{leaf}(self, v):
        \"\"\"Sets `{leaf}` on every element in one call: one value for
        all of them, or a sequence with one per element (a numpy array
        of the matching dtype is read in place).\"\"\"
        self._set_{leaf}(0, len(self), v)

    def get_{leaf}(self):
        \"\"\"`{leaf}` of every element in one call, as {shape}
        (numpy when importable, else a flat ctypes array).\"\"\"
        n = len(self)
        buf, ptr, stride = _cols.column_out(\"{code}\", {k}, n)
        _f.{prefix}_get_{leaf}_n(self._p, 0, ptr, n, stride)
        return _cols.column_finish(buf, \"{code}\", {k}, n)

"));
    }
}

/// `push_many` over `push_fn` (the shim's n-element push) for an
/// index-addressable collection view.
fn push_many_method(cls: &mut String, prefix: &str, push_fn: &str, lv: &[Leaf]) {
    let (params, _, _) = push_parts(lv, 30, "                  ");
    let pairs: Vec<String> = lv.iter().map(|l| format!("(\"{0}\", {0})", l.name)).collect();
    let mut sets = String::new();
    for l in lv {
        sets.push_str(&format!(
"        if {0} is not None:
            self._set_{0}(i0, n, {0})
", l.name));
    }
    let params = if lv.is_empty() { String::new() } else { format!(", *, {params}") };
    cls.push_str(&format!(
"    def push_many(self, n=None{params}):
        \"\"\"Appends `n` elements in one call. Each keyword is one value
        for all of them or a sequence with one per element (a numpy
        array of the matching dtype is read in place); `n` may be
        omitted when some keyword is a sequence. Returns the index of
        the first new element.\"\"\"
        n = _cols.count(n, ({}))
        i0 = len(self)
        _f.{prefix}_{push_fn}(self._p, None, n)
{sets}        return i0

", wrap_items(&pairs, 27, 76, "                        ")));
}

/// One collection field's view class + the owner property line.
fn collection_py(
    py: &mut Py,
    model: &Model,
    owner: &str,
    sym_prefix: &str,
    owner_cls: &mut String,
    f: &Field,
) -> Result<(), String> {
    let field = &f.name;
    let elem = f.of.as_deref().ok_or("collection without element")?;
    let prefix = format!("{sym_prefix}_{field}");
    let container = f.container.as_deref().unwrap_or("vec");
    let refs_flavor = f.spelled.as_deref().unwrap_or("").contains("refs::");
    let kind = match container { "deque" => "Deque", "arena" => "Arena", _ => "Vec" };
    let view = format!("{owner}{}{kind}", crate::emit_hpp::camel(field));
    let elem_ty = model.types.get(elem)
        .ok_or_else(|| format!("collection `{field}`: element type `{elem}` not in sidecar"))?;
    let lv = leaves(model, elem_ty);
    let elem_sn = snake(elem);
    let slots = format!("_{elem_sn}_slots");
    let rec = format!("_{elem_sn}_rec");

    sig(py, &format!("{prefix}_len"), &["ctypes.c_void_p"], "ctypes.c_uint32");
    sig(py, &format!("{prefix}_reserve"),
        &["ctypes.c_void_p", "ctypes.c_uint32"], "None");

    let mut cls = format!(
"class {view}:
    \"\"\"View of `{field}` ({} of {elem}); element wrappers re-resolve
    their pointer by key on every access, so growing the collection
    cannot leave them dangling. Mutating while iterating is undefined.

    Construction and bulk edits cross the FFI once per call: push(**fields)
    fills the new element's fields in the same call as the push;
    push_many(**arrays) appends many; set_<field>(values) / get_<field>()
    move one column of the whole collection.\"\"\"

    __slots__ = (\"_p\",)

    def __init__(self, p):
        self._p = p

    def __len__(self):
        return _f.{prefix}_len(self._p)

    def reserve(self, additional):
        _f.{prefix}_reserve(self._p, additional)

", container);

    // The keyed push: one crossing carrying every given field, returning
    // the wrapper built from the key the shim hands back.
    let push_method = |cls: &mut String, name: &str, shim: &str, doc: &str| {
        // A type with no scalar leaves has nothing to name: no bare `*`.
        let head = if lv.is_empty() {
            format!("    def {name}(self")
        } else {
            format!("    def {name}(self, *, ")
        };
        let (params, prelude, args) = push_parts(&lv, head.len(), "                 ");
        let ret = if container == "arena" {
            format!("        return {elem}Ref(_f.{prefix}_{shim}(self._p, {slots}, 1))\n")
        } else if refs_flavor {
            format!(
"        r = {elem}Ref(_f.{prefix}_{shim}(self._p, {slots}, 1))
        return {elem}(lambda k=r.raw: _f.{prefix}_get(self._p, k), r)
")
        } else {
            format!(
"        i = _f.{prefix}_{shim}(self._p, {slots}, 1)
        return {elem}(lambda i=i: _f.{prefix}_at(self._p, i), i)
")
        };
        cls.push_str(&format!(
"{head}{params}):
        \"\"\"{doc}\"\"\"
        m = 0
{prelude}        {rec}.pack_into({slots}, 0,
            {args})
{ret}
"));
    };

    match container {
        "vec" | "deque" => {
            sig(py, &format!("{prefix}_at"),
                &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_void_p");
            sig(py, &format!("{prefix}_clear"), &["ctypes.c_void_p"], "None");
            sig(py, &format!("{prefix}_truncate"),
                &["ctypes.c_void_p", "ctypes.c_uint32"], "None");
            let getitem_ref = if refs_flavor {
                format!(
"        if isinstance(i, {elem}Ref):
            return {elem}(lambda r=i.raw: _f.{prefix}_get(self._p, r), i)
")
            } else {
                String::new()
            };
            cls.push_str(&format!(
"    def __getitem__(self, i):
{getitem_ref}        n = len(self)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(i)
        return {elem}(lambda i=i: _f.{prefix}_at(self._p, i), i)

    def __iter__(self):
        for i in range(len(self)):
            yield {elem}(lambda i=i: _f.{prefix}_at(self._p, i), i)

    def clear(self):
        _f.{prefix}_clear(self._p)

    def truncate(self, n):
        _f.{prefix}_truncate(self._p, n)

"));
            let slot_args = &["ctypes.c_void_p", "ctypes.POINTER(ctypes.c_uint64)",
                              "ctypes.c_uint32"];
            if container == "vec" {
                sig(py, &format!("{prefix}_push"), &["ctypes.c_void_p"],
                    "ctypes.c_void_p");
                sig(py, &format!("{prefix}_pop"), &["ctypes.c_void_p"],
                    "ctypes.c_bool");
                sig(py, &format!("{prefix}_push_n"), slot_args, "ctypes.c_uint32");
                push_method(&mut cls, "push", "push_n",
                    "Appends one element and returns it; each keyword sets that
        field in the same call, an omitted one keeps the Rust default.");
                cls.push_str(&format!(
"    def pop(self):
        \"\"\"Drops the last element; False when already empty.\"\"\"
        return _f.{prefix}_pop(self._p)

"));
                push_many_method(&mut cls, &prefix, "push_n", &lv);
            } else {
                for m in ["push_back", "push_front"] {
                    sig(py, &format!("{prefix}_{m}"), &["ctypes.c_void_p"],
                        "ctypes.c_void_p");
                    sig(py, &format!("{prefix}_{m}_n"), slot_args, "ctypes.c_uint32");
                }
                push_method(&mut cls, "push_back", "push_back_n",
                    "Appends one element at the back and returns it; each keyword
        sets that field in the same call, an omitted one keeps the Rust
        default.");
                push_method(&mut cls, "push_front", "push_front_n",
                    "Inserts one element at the front and returns it; each keyword
        sets that field in the same call, an omitted one keeps the Rust
        default.");
                for m in ["pop_back", "pop_front"] {
                    sig(py, &format!("{prefix}_{m}"), &["ctypes.c_void_p"],
                        "ctypes.c_bool");
                    cls.push_str(&format!(
"    def {m}(self):
        \"\"\"Drops one end; False when already empty.\"\"\"
        return _f.{prefix}_{m}(self._p)

"));
                }
                push_many_method(&mut cls, &prefix, "push_back_n", &lv);
            }
            column_methods(py, &mut cls, &prefix, &lv);
        }
        "arena" => {
            sig(py, &format!("{prefix}_push"), &["ctypes.c_void_p"],
                "ctypes.c_uint32");
            sig(py, &format!("{prefix}_push_n"),
                &["ctypes.c_void_p", "ctypes.POINTER(ctypes.c_uint64)",
                  "ctypes.c_uint32"], "ctypes.c_uint32");
            sig(py, &format!("{prefix}_remove"),
                &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_bool");
            sig(py, &format!("{prefix}_clear"), &["ctypes.c_void_p"], "None");
            for m in ["first", "next", "last", "prev"] {
                let args: &[&str] = if m == "first" || m == "last" {
                    &["ctypes.c_void_p"]
                } else {
                    &["ctypes.c_void_p", "ctypes.c_uint32"]
                };
                sig(py, &format!("{prefix}_{m}"), args, "ctypes.c_uint32");
            }
            push_method(&mut cls, "push", "push_n",
                "New element's ref (get()/[] take it back); each keyword sets
        that field in the same call, an omitted one keeps the Rust
        default.");
            cls.push_str(&format!(
"    def remove(self, r):
        return _f.{prefix}_remove(self._p, _raw(r))

    def clear(self):
        _f.{prefix}_clear(self._p)

    def __getitem__(self, r):
        r = r if isinstance(r, {elem}Ref) else {elem}Ref(int(r))
        return {elem}(lambda k=r.raw: _f.{prefix}_get(self._p, k), r)

    def __iter__(self):
        \"\"\"Live slots in order; yields element wrappers (refs() for
        the refs).\"\"\"
        r = _f.{prefix}_first(self._p)
        while r != 0xFFFFFFFF:
            yield {elem}(lambda k=r: _f.{prefix}_get(self._p, k), {elem}Ref(r))
            r = _f.{prefix}_next(self._p, r)

    def refs(self):
        r = _f.{prefix}_first(self._p)
        while r != 0xFFFFFFFF:
            yield {elem}Ref(r)
            r = _f.{prefix}_next(self._p, r)

"));
        }
        other => return Err(format!("unknown container `{other}`")),
    }

    if refs_flavor {
        sig(py, &format!("{prefix}_get"),
            &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_void_p");
        sig(py, &format!("{prefix}_contains"),
            &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_bool");
        sig(py, &format!("{prefix}_try_get"),
            &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_void_p");
        cls.push_str(&format!(
"    def get(self, r):
        r = r if isinstance(r, {elem}Ref) else {elem}Ref(int(r))
        return {elem}(lambda k=r.raw: _f.{prefix}_get(self._p, k), r)

    def __contains__(self, r):
        return _f.{prefix}_contains(self._p, _raw(r))

    def try_get(self, r):
        \"\"\"The element, or None for a stale or foreign ref.\"\"\"
        r = r if isinstance(r, {elem}Ref) else {elem}Ref(int(r))
        p = _f.{prefix}_try_get(self._p, r.raw)
        return {elem}(lambda k=r.raw: _f.{prefix}_get(self._p, k), r) if p else None

"));
        if container != "arena" {
            sig(py, &format!("{prefix}_ref_at"),
                &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_uint32");
            let (first, last) = if container == "deque" {
                ("front_ref", "back_ref")
            } else {
                ("first_ref", "last_ref")
            };
            sig(py, &format!("{prefix}_{first}"), &["ctypes.c_void_p"],
                "ctypes.c_uint32");
            sig(py, &format!("{prefix}_{last}"), &["ctypes.c_void_p"],
                "ctypes.c_uint32");
            cls.push_str(&format!(
"    def ref_at(self, i):
        return {elem}Ref(_f.{prefix}_ref_at(self._p, i))

    def {first}(self):
        \"\"\"Ref of the first element; null when empty.\"\"\"
        return {elem}Ref(_f.{prefix}_{first}(self._p))

    def {last}(self):
        \"\"\"Ref of the last element; null when empty.\"\"\"
        return {elem}Ref(_f.{prefix}_{last}(self._p))

"));
            sig(py, &format!("{prefix}_get_refs_n"),
                &["ctypes.c_void_p", "ctypes.c_uint32", "ctypes.c_void_p",
                  "ctypes.c_uint32"], "ctypes.c_bool");
            cls.push_str(&format!(
"    def get_refs(self):
        \"\"\"The ref of every element in one call, as a uint32 array of
        raw handles in index order (numpy when importable, else a ctypes
        array) -- what the ref keywords of push_many take.\"\"\"
        n = len(self)
        buf, ptr, _stride = _cols.column_out(\"I\", 1, n)
        _f.{prefix}_get_refs_n(self._p, 0, ptr, n)
        return _cols.column_finish(buf, \"I\", 1, n)

"));
        }
    }

    py.body.push_str(&cls);
    py.body.push('\n');
    owner_cls.push_str(&format!(
"    @property
    def {field}(self):
        return {view}(self._p)

"));
    Ok(())
}

/// One field's ffi signatures + owner-class lines.
fn field_py(
    py: &mut Py,
    model: &Model,
    owner: &str,
    prefix: &str,
    owner_cls: &mut String,
    f: &Field,
) -> Result<(), String> {
    let name = &f.name;
    let of = f.of.as_deref().unwrap_or("");
    match f.kind.as_str() {
        "data" | "param" | "euler_param" => {
            let Some((ct, coerce)) = value_ct(f) else {
                return Err(format!("`{owner}.{name}`: unsupported {} of {of}", f.kind));
            };
            prop(py, owner_cls, prefix, name, name, &ct, coerce);
            if f.kind != "data" {
                optimize_prop(py, owner_cls, prefix, name);
            }
        }
        "component" => match of {
            "TransformParam" | "TransformParamF"
            | "ScaledTransformParam" | "ScaledTransformParamF" => {
                let f32 = of.ends_with('F');
                let scaled = of.starts_with("Scaled");
                let (v3, q, sc, val) = if f32 {
                    ("_m.vect3f", "_m.quaternf", "ctypes.c_float", "f")
                } else {
                    ("_m.vect3d", "_m.quaternd", "ctypes.c_double", "d")
                };
                let p2 = format!("{prefix}_{name}");
                prop(py, owner_cls, &p2, "translation",
                     &format!("{name}_translation"), v3, true);
                prop(py, owner_cls, &p2, "rotation",
                     &format!("{name}_rotation"), q, true);
                if scaled {
                    prop(py, owner_cls, &p2, "scale", &format!("{name}_scale"), sc, false);
                }
                let flags: &[&str] = if scaled {
                    &["optimize_translation", "optimize_rotation", "optimize_scale"]
                } else {
                    &["optimize_translation", "optimize_rotation"]
                };
                for flag in flags {
                    prop(py, owner_cls, &p2, flag,
                         &format!("{name}_{flag}"), "ctypes.c_bool", false);
                }
                let (view, value) = if scaled {
                    ("ScaledTransformParamView", format!("_tf.scaled_transform3{val}"))
                } else {
                    ("TransformParamView", format!("_tf.transform3{val}"))
                };
                owner_cls.push_str(&format!(
"    @property
    def {name}(self):
        \"\"\"`{name}` as a live transform: `.translation`, `.rotation` and
        the optimize flags read and write through, and it acts like a
        transform (`{name} * x`, `{name}.inv() * y`, `a.{name}.inv() * b.{name}`).\"\"\"
        return _tf.{view}(self, \"{name}\", {value})

"));
            }
            "UnitVecParam" | "UnitVecParamF" => {
                let v3 = if of == "UnitVecParamF" { "_m.vect3f" } else { "_m.vect3d" };
                let p2 = format!("{prefix}_{name}");
                prop(py, owner_cls, &p2, "unit", &format!("{name}_unit"), v3,
                     true);
                for i in 0..2 {
                    sig(py, &format!("{p2}_unit_d{i}"), &["ctypes.c_void_p"], v3);
                    owner_cls.push_str(&format!(
"    @property
    def {name}_unit_d{i}(self):
        \"\"\"Chart tangent basis (read-only).\"\"\"
        return _f.{p2}_unit_d{i}(self._p)

"));
                }
            }
            "AngleParam" | "AngleParamF" => {
                let (sc, m) = if of == "AngleParamF" {
                    ("ctypes.c_float", "_m.matrix2f")
                } else {
                    ("ctypes.c_double", "_m.matrix2d")
                };
                let p2 = format!("{prefix}_{name}");
                // angle value (read/write) + optimize flag (read/write)
                prop(py, owner_cls, &p2, "angle", &format!("{name}_angle"), sc, false);
                sig(py, &format!("{p2}_angle_optimize"), &["ctypes.c_void_p"], "ctypes.c_bool");
                sig(py, &format!("{p2}_angle_set_optimize"),
                    &["ctypes.c_void_p", "ctypes.c_bool"], "None");
                owner_cls.push_str(&format!(
"    @property
    def {name}_angle_optimize(self):
        return _f.{p2}_angle_optimize(self._p)

    @{name}_angle_optimize.setter
    def {name}_angle_optimize(self, v):
        _f.{p2}_angle_set_optimize(self._p, bool(v))

"));
                // rotation matrix at the current angle (read-only, computed)
                sig(py, &format!("{p2}_rotation_matrix"), &["ctypes.c_void_p"], m);
                owner_cls.push_str(&format!(
"    @property
    def {name}_rotation_matrix(self):
        \"\"\"Rotation matrix at the current angle (read-only).\"\"\"
        return _f.{p2}_rotation_matrix(self._p)

"));
            }
            _ => {
                if !model.types.contains_key(of) {
                    return Err(format!("`{owner}.{name}`: unknown component {of}"));
                }
                sig(py, &format!("{prefix}_{name}_ptr"), &["ctypes.c_void_p"],
                    "ctypes.c_void_p");
                owner_cls.push_str(&format!(
"    @property
    def {name}(self):
        return {of}(lambda: _f.{prefix}_{name}_ptr(self._p))

"));
            }
        },
        "struct" => {
            sig(py, &format!("{prefix}_{name}_ptr"), &["ctypes.c_void_p"],
                "ctypes.c_void_p");
            owner_cls.push_str(&format!(
"    @property
    def {name}(self):
        return {of}(lambda: _f.{prefix}_{name}_ptr(self._p))

"));
        }
        "optional" => {
            sig(py, &format!("{prefix}_has_{name}"), &["ctypes.c_void_p"],
                "ctypes.c_bool");
            sig(py, &format!("{prefix}_make_{name}"), &["ctypes.c_void_p"],
                "ctypes.c_void_p");
            sig(py, &format!("{prefix}_clear_{name}"), &["ctypes.c_void_p"],
                "None");
            sig(py, &format!("{prefix}_{name}"), &["ctypes.c_void_p"],
                "ctypes.c_void_p");
            sig(py, &format!("{prefix}_make_{name}_slots"),
                &["ctypes.c_void_p", "ctypes.POINTER(ctypes.c_uint64)"],
                "ctypes.c_void_p");
            // make_<name>(**fields): the option's entity from one slot
            // record, like push(**fields).
            let of_ty = model.types.get(of)
                .ok_or_else(|| format!("`{owner}.{name}`: unknown option type {of}"))?;
            let lv = leaves(model, of_ty);
            let of_sn = snake(of);
            let head = if lv.is_empty() {
                format!("    def make_{name}(self")
            } else {
                format!("    def make_{name}(self, *, ")
            };
            let (params, prelude, args) = push_parts(&lv, head.len(), "                 ");
            owner_cls.push_str(&format!(
"    @property
    def {name}(self):
        \"\"\"The `{of}`, or None while absent (make_{name}() creates).\"\"\"
        if not _f.{prefix}_{name}(self._p):
            return None
        return {of}(lambda: _f.{prefix}_{name}(self._p))

{head}{params}):
        \"\"\"Creates the `{of}` (replacing one already there) and returns
        it; each keyword sets that field in the same call, an omitted one
        keeps the Rust default.\"\"\"
        m = 0
{prelude}        _{of_sn}_rec.pack_into(_{of_sn}_slots, 0,
            {args})
        _f.{prefix}_make_{name}_slots(self._p, _{of_sn}_slots)
        return {of}(lambda: _f.{prefix}_{name}(self._p))

    def clear_{name}(self):
        _f.{prefix}_clear_{name}(self._p)

"));
        }
        "ref" => {
            sig(py, &format!("{prefix}_{name}"), &["ctypes.c_void_p"],
                "ctypes.c_uint32");
            sig(py, &format!("{prefix}_set_{name}"),
                &["ctypes.c_void_p", "ctypes.c_uint32"], "None");
            owner_cls.push_str(&format!(
"    @property
    def {name}(self):
        return {of}Ref(_f.{prefix}_{name}(self._p))

    @{name}.setter
    def {name}(self, r):
        _f.{prefix}_set_{name}(self._p, _raw(r))

"));
        }
        "collection" => collection_py(py, model, owner, prefix, owner_cls, f)?,
        "opaque" => {
            owner_cls.push_str(&format!(
                "    # field `{name}`: {of} -- opaque, no accessor generated\n\n"));
        }
        "skip" | "self_block" | "cross_block" | "triplet_block" => {}
        other => return Err(format!("`{owner}.{name}`: kind `{other}`?")),
    }
    Ok(())
}

/// The (ffi module, api module) pair for one root; `lib_ident` is the
/// cdylib's crate ident (`{crate}_capi`).
pub fn emit(model: &Model, lib_ident: &str) -> Result<(String, String), String> {
    let root = &model.root;
    let root_sn = snake(root);
    let fp = if model.precision == "f32" { "ctypes.c_float" } else { "ctypes.c_double" };
    let surfaced = surfaced_types(model);
    let mut py = Py { sigs: String::new(), body: String::new() };

    // Refs first (null default, valid, equality).
    for (tn, _) in &surfaced {
        py.body.push_str(&format!(
"class {tn}Ref:
    \"\"\"Typed handle into the collection that issued it -- the Python
    spelling of Rust's `Ref<{tn}>`. Default-constructed it is the null
    sentinel.\"\"\"

    __slots__ = (\"raw\",)

    def __init__(self, raw=0xFFFFFFFF):
        self.raw = raw

    @property
    def valid(self):
        return self.raw != 0xFFFFFFFF

    def __eq__(self, o):
        return isinstance(o, {tn}Ref) and o.raw == self.raw

    def __hash__(self):
        return hash(({tn}Ref, self.raw))

    def __repr__(self):
        return \"{tn}Ref(%s)\" % (self.raw if self.valid else \"null\")


"));
    }

    // Zero tuples standing in for absent math keywords in push(**fields),
    // one per component count in use.
    let mut zs = std::collections::BTreeSet::new();
    for (_, t) in &surfaced {
        for l in leaves(model, t) {
            if let LeafTy::Math { n, .. } = l.ty {
                zs.insert(n);
            }
        }
    }
    for n in &zs {
        py.body.push_str(&format!("_Z{n} = (0.0,) * {n}\n"));
    }
    if !zs.is_empty() {
        py.body.push_str("\n\n");
    }

    // Entity/component classes, children-first (same order rule as C++).
    let emitted: Vec<&str> = surfaced.iter().map(|(tn, _)| tn.as_str()).collect();
    let mut remaining: Vec<(&String, &Type)> = surfaced.clone();
    let mut done: Vec<&str> = Vec::new();
    while !remaining.is_empty() {
        let Some(pos) = remaining.iter().position(|(_, t)| {
            crate::emit_hpp::deps(t).iter()
                .all(|d| done.contains(d) || !emitted.contains(d))
        }) else {
            return Err(format!("containment cycle among: {}",
                remaining.iter().map(|(tn, _)| tn.as_str())
                    .collect::<Vec<_>>().join(", ")));
        };
        let (tn, t) = remaining.remove(pos);
        let prefix = format!("{root_sn}_{}", snake(tn));
        // The slot record push(**fields) packs for this type: mask
        // word(s) then one slot per leaf, and the scratch it packs into.
        let lv = leaves(model, t);
        let codes: String = "Q".repeat(mask_words(lv.len()))
            + &lv.iter().map(|l| l.ty.pack_code()).collect::<String>();
        py.body.push_str(&format!(
"_{0}_rec = struct.Struct(\"={codes}\")
_{0}_slots = (ctypes.c_uint64 * {1})()


", snake(tn), record_slots(&lv)));
        // `.index` steps aside for a field of that name; `ref` cannot be
        // a field (a Rust keyword).
        let index_prop = if t.fields.iter().any(|f| f.name == "index") {
            "    # field `index` takes the name the index accessor would have\n\n".to_string()
        } else {
            format!(
"    @property
    def index(self):
        \"\"\"The index this wrapper was looked up by (TypeError when it
        was a {tn}Ref).\"\"\"
        k = self._key
        if isinstance(k, int):
            return k
        raise TypeError(\"{tn} addressed by ref, not by index\")

")
        };
        let mut cls = format!(
"class {tn}:
    \"\"\"A `{tn}` in its owner's storage, addressed by key rather than by
    pointer: the pointer is re-resolved on every access, so growing the
    collection cannot leave this wrapper dangling.\"\"\"

    __slots__ = (\"_at\", \"_key\")
    param_count = {}

    def __init__(self, at, key=None):
        # Zero-argument callable returning a currently-valid pointer, and
        # the key it resolves by (a {tn}Ref or an index; None for a
        # nested element).
        self._at = at
        self._key = key

    @property
    def _p(self):
        return self._at()

    @property
    def ref(self):
        \"\"\"The {tn}Ref this wrapper was looked up by (TypeError when it
        was an index).\"\"\"
        k = self._key
        if isinstance(k, {tn}Ref):
            return k
        raise TypeError(\"{tn} addressed by index, not by ref\")

{index_prop}", t.param_count);
        for f in &t.fields {
            field_py(&mut py, model, tn, &prefix, &mut cls, f)?;
        }
        py.body.push_str(&cls);
        py.body.push('\n');
        done.push(tn.as_str());
    }

    // Covariance view.
    let mut cov_methods = String::new();
    sig(&mut py, &format!("{root_sn}_assemble_covariance"),
        &["ctypes.c_void_p", "ctypes.c_uint32",
          "ctypes.POINTER(ctypes.c_void_p)"], "ctypes.c_int32");
    sig(&mut py, &format!("{root_sn}_assemble_covariance_with"),
        &["ctypes.c_void_p", "ctypes.c_uint32", "ctypes.c_uint32",
          "ctypes.c_uint32", "ctypes.POINTER(ctypes.c_void_p)"],
        "ctypes.c_int32");
    sig(&mut py, &format!("{root_sn}_cov_error"), &["ctypes.c_void_p"],
        "ctypes.c_char_p");
    sig(&mut py, &format!("{root_sn}_cov_free"), &["ctypes.c_void_p"],
        "None");
    sig(&mut py, &format!("{root_sn}_cov_plan"),
        &["ctypes.c_void_p", "ctypes.POINTER(_solver.CovPlan)"], "None");
    let cov_entities: Vec<(&String, &Type)> = surfaced.iter()
        .filter(|(_, t)| t.role == "entity" && t.param_count > 0)
        .map(|(tn, t)| (*tn, *t))
        .collect();
    let mut marg_arms = String::new();
    let mut cond_arms = String::new();
    let mut sd_arms = String::new();
    for (tn, t) in &cov_entities {
        let sn = format!("{root_sn}_{}", snake(tn));
        for q in ["marginal_cov", "conditional_cov", "std_dev"] {
            sig(&mut py, &format!("{sn}_{q}"),
                &["ctypes.c_void_p", "ctypes.c_void_p",
                  "ctypes.POINTER(ctypes.c_double)", "ctypes.c_uint32"],
                "ctypes.c_int32");
        }
        let n = t.param_count;
        let shape = if n <= 3 {
            format!("_shape_{n}")
        } else {
            format!("lambda b: _shape_n(b, {n})")
        };
        marg_arms.push_str(&format!(
"        if isinstance(e, {tn}):
            return ({shape})(_cov_query(self._h, _f.{sn}_marginal_cov, e._p, {sq}))
", sq = n * n));
        cond_arms.push_str(&format!(
"        if isinstance(e, {tn}):
            return ({shape})(_cov_query(self._h, _f.{sn}_conditional_cov, e._p, {sq}))
", sq = n * n));
        sd_arms.push_str(&format!(
"        if isinstance(e, {tn}):
            return list(_cov_query(self._h, _f.{sn}_std_dev, e._p, {n}))
"));
    }
    cov_methods.push_str(&format!(
"    def marginal(self, e):
        \"\"\"The entity's marginal covariance block.\"\"\"
{marg_arms}        raise TypeError(\"no marginal for %r\" % (e,))

    def conditional(self, e):
        \"\"\"Conditional covariance (all other parameters fixed).\"\"\"
{cond_arms}        raise TypeError(\"no conditional for %r\" % (e,))

    def std_dev(self, e):
        \"\"\"Per-parameter standard deviations (every CovMode, incl.
        TriDiagonal).\"\"\"
{sd_arms}        raise TypeError(\"no std_dev for %r\" % (e,))
"));
    for (an, ta) in &cov_entities {
        for (bn, tb) in &cov_entities {
            let a_sn = snake(an);
            let b_sn = snake(bn);
            sig(&mut py, &format!("{root_sn}_{a_sn}_{b_sn}_cross_cov"),
                &["ctypes.c_void_p", "ctypes.c_void_p", "ctypes.c_void_p",
                  "ctypes.POINTER(ctypes.c_double)", "ctypes.c_uint32"],
                "ctypes.c_int32");
            cov_methods.push_str(&format!(
"    def _cross_{a_sn}_{b_sn}(self, a, b):
        buf = (ctypes.c_double * {sz})()
        rows = _f.{root_sn}_{a_sn}_{b_sn}_cross_cov(self._h, a._p, b._p, buf, {sz})
        if rows < 0:
            raise AraelError(
                rows, (_f.{root_sn}_cov_error(self._h) or b\"\").decode())
        return tuple(tuple(buf[r * {cols} + c] for c in range({cols}))
                     for r in range(rows))
", sz = ta.param_count * tb.param_count, cols = tb.param_count));
        }
    }
    let mut cross_dispatch = String::new();
    for (an, _) in &cov_entities {
        for (bn, _) in &cov_entities {
            cross_dispatch.push_str(&format!(
"        if isinstance(a, {an}) and isinstance(b, {bn}):
            return self._cross_{}_{}(a, b)
", snake(an), snake(bn)));
        }
    }

    py.body.push_str(&format!(
"def _cov_query(c, fn, p, cap):
    buf = (ctypes.c_double * cap)()
    n = fn(c, p, buf, cap)
    if n < 0:
        raise AraelError(
            n, (_f.{root_sn}_cov_error(c) or b\"\").decode())
    return buf


def _shape_1(buf):
    return float(buf[0])


def _shape_2(buf):
    return _m.matrix2d.from_elements(buf[0], buf[1], buf[2], buf[3])


def _shape_3(buf):
    return _m.matrix3d.from_elements(*[buf[i] for i in range(9)])


def _shape_n(buf, n):
    return tuple(tuple(buf[r * n + c] for c in range(n)) for r in range(n))


class Covariance:
    \"\"\"An assembled covariance, OWNED: released on garbage
    collection (free() to force it), independent of later assemblies.
    Per-entity queries (marginal/conditional by param count: 1 ->
    float, 2/3 -> matrix2d/3d, larger -> row-major tuples). Entity
    arguments must come from the live model.\"\"\"

    __slots__ = (\"_h\",)

    def __init__(self, h):
        self._h = h

    def plan(self):
        \"\"\"What the assembly decided -- the ordering it kept, what the
        candidates priced at, and how many symbolics it built.\"\"\"
        p = CovPlan()
        _f.{root_sn}_cov_plan(self._h, ctypes.byref(p))
        return p

    def free(self):
        if self._h:
            _f.{root_sn}_cov_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

{cov_methods}
    def cross(self, a, b):
        \"\"\"Row-major cross-covariance block between two entities.\"\"\"
{cross_dispatch}        raise TypeError(\"no cross-covariance for %r x %r\" % (a, b))


"));

    // Root surface.
    sig(&mut py, &format!("{root_sn}_new"), &[], "ctypes.c_void_p");
    sig(&mut py, &format!("{root_sn}_free"), &["ctypes.c_void_p"], "None");
    sig(&mut py, &format!("{root_sn}_last_error"), &["ctypes.c_void_p"],
        "ctypes.c_char_p");
    sig(&mut py, &format!("{root_sn}_last_failure"),
        &["ctypes.c_void_p", "ctypes.POINTER(_solver.SolveFailure)"],
        "ctypes.c_bool");
    sig(&mut py, &format!("{root_sn}_validate"), &["ctypes.c_void_p"],
        "ctypes.c_char_p");
    sig(&mut py, &format!("{root_sn}_set_log_level"), &["ctypes.c_uint32"],
        "None");
    sig(&mut py, &format!("{root_sn}_result_report"),
        &["ctypes.c_void_p", "ctypes.c_bool"], "ctypes.c_char_p");
    sig(&mut py, &format!("{root_sn}_result_plan"),
        &["ctypes.c_void_p", "ctypes.POINTER(_solver.SchurPlan)"],
        "ctypes.c_bool");
    sig(&mut py, &format!("{root_sn}_result_steps"),
        &["ctypes.c_void_p", "ctypes.POINTER(_solver.LmStep)",
          "ctypes.c_uint64"],
        "ctypes.c_uint64");
    sig(&mut py, &format!("{root_sn}_result_free"), &["ctypes.c_void_p"],
        "None");
    sig(&mut py, &format!("{root_sn}_cost"), &["ctypes.c_void_p"],
        "ctypes.c_double");
    // The cost-table surface exists only for `#[arael(root, jacobian)]`
    // roots (the sidecar's `jacobian` flag).
    if model.jacobian {
        sig(&mut py, &format!("{root_sn}_cost_table"), &["ctypes.c_void_p"],
            "ctypes.c_int32");
        sig(&mut py, &format!("{root_sn}_cost_table_name"),
            &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_char_p");
        sig(&mut py, &format!("{root_sn}_cost_table_value"),
            &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_double");
        sig(&mut py, &format!("{root_sn}_calc_jacobian"),
            &["ctypes.c_void_p", "ctypes.POINTER(ctypes.c_void_p)"],
            "ctypes.c_int32");
        sig(&mut py, &format!("{root_sn}_jac_error"), &["ctypes.c_void_p"],
            "ctypes.c_char_p");
        sig(&mut py, &format!("{root_sn}_jac_free"), &["ctypes.c_void_p"],
            "None");
        sig(&mut py, &format!("{root_sn}_jac_num_residuals"),
            &["ctypes.c_void_p"], "ctypes.c_uint64");
        sig(&mut py, &format!("{root_sn}_jac_num_params"),
            &["ctypes.c_void_p"], "ctypes.c_uint64");
        sig(&mut py, &format!("{root_sn}_jac_singular_values"),
            &["ctypes.c_void_p", "ctypes.c_bool",
              "ctypes.POINTER(ctypes.c_double)", "ctypes.c_uint64"],
            "ctypes.c_int64");
        sig(&mut py, &format!("{root_sn}_jac_column_l2_norms"),
            &["ctypes.c_void_p", "ctypes.POINTER(ctypes.c_double)",
              "ctypes.c_uint64"],
            "ctypes.c_int64");
        py.body.push_str(&format!(
"class Jacobian:
    \"\"\"A computed Jacobian, OWNED: released on garbage collection
    (free() to force it). A snapshot of the parameters at the call;
    later solves or edits do not touch it.\"\"\"

    __slots__ = (\"_h\",)

    def __init__(self, h):
        self._h = h

    def free(self):
        if self._h:
            _f.{root_sn}_jac_free(self._h)
            self._h = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    @property
    def num_residuals(self):
        \"\"\"Number of residual rows.\"\"\"
        return _f.{root_sn}_jac_num_residuals(self._h)

    @property
    def num_params(self):
        \"\"\"Number of parameter columns.\"\"\"
        return _f.{root_sn}_jac_num_params(self._h)

    def singular_values(self, column_normalised=False):
        \"\"\"Singular values, descending, always f64 (near-zero
        values count the free DOF). column_normalised scales each
        column to unit L2 norm first, so the spectrum reflects
        row-space rank alone, not per-parameter scale.\"\"\"
        return self._vals(
            lambda buf, cap: _f.{root_sn}_jac_singular_values(
                self._h, column_normalised, buf, cap))

    def column_l2_norms(self):
        \"\"\"L2 norm of each Jacobian column, in parameter-index
        order.\"\"\"
        return self._vals(
            lambda buf, cap: _f.{root_sn}_jac_column_l2_norms(
                self._h, buf, cap))

    def _vals(self, fn):
        def ck(n):
            if n < 0:
                raise AraelError(
                    n, (_f.{root_sn}_jac_error(self._h) or b\"\").decode())
            return n
        n = ck(fn(None, 0))
        if n == 0:
            return []
        buf = (ctypes.c_double * n)()
        ck(fn(buf, n))
        return list(buf)


"));
    }
    let ct_method = if model.jacobian {
        format!(
"    def cost_table(self):
        \"\"\"Per-constraint cost breakdown at the current parameters:
        {{label: that group's robustified cost}} (a `loss` applied;
        the label is `name` on the constraint attribute, else the
        struct name). The table sums to cost(). Raises on a
        panic.\"\"\"
        n = _f.{root_sn}_cost_table(self._p)
        if n < 0:
            raise AraelError(n, _err(self._p))
        return {{
            _f.{root_sn}_cost_table_name(self._p, i).decode():
                _f.{root_sn}_cost_table_value(self._p, i)
            for i in range(n)
        }}

    def calc_jacobian(self):
        \"\"\"The sparse Jacobian at the current parameters (a
        snapshot). Raises on a panic.\"\"\"
        j = ctypes.c_void_p()
        code = _f.{root_sn}_calc_jacobian(self._p, ctypes.byref(j))
        if code != 0:
            raise AraelError(code, _err(self._p))
        return Jacobian(j)

")
    } else {
        String::new()
    };
    sig(&mut py, &format!("{root_sn}_lm_config"),
        &["ctypes.c_uint32", "ctypes.POINTER(LmConfigRaw)"], "None");
    sig(&mut py, &format!("{root_sn}_sparse_options"),
        &["ctypes.POINTER(_solver.SparseOptions)"], "None");
    sig(&mut py, &format!("{root_sn}_solve_dense"),
        &["ctypes.c_void_p", "ctypes.POINTER(LmConfigRaw)",
          "ctypes.POINTER(LmResultRaw)"], "ctypes.c_int32");
    sig(&mut py, &format!("{root_sn}_solve_sparse"),
        &["ctypes.c_void_p", "ctypes.POINTER(LmConfigRaw)",
          "ctypes.POINTER(_solver.SparseOptions)",
          "ctypes.POINTER(LmResultRaw)"], "ctypes.c_int32");
    sig(&mut py, &format!("{root_sn}_session_new"),
        &["ctypes.POINTER(_solver.SparseOptions)"], "ctypes.c_void_p");
    sig(&mut py, &format!("{root_sn}_session_free"),
        &["ctypes.c_void_p"], "None");
    sig(&mut py, &format!("{root_sn}_session_invalidate"),
        &["ctypes.c_void_p"], "None");
    sig(&mut py, &format!("{root_sn}_session_solve"),
        &["ctypes.c_void_p", "ctypes.c_void_p", "ctypes.POINTER(LmConfigRaw)",
          "ctypes.POINTER(LmResultRaw)"], "ctypes.c_int32");
    sig(&mut py, &format!("{root_sn}_solve_band"),
        &["ctypes.c_void_p", "ctypes.c_uint32", "ctypes.POINTER(LmConfigRaw)",
          "ctypes.POINTER(LmResultRaw)"], "ctypes.c_int32");

    let root_ty = model.types.get(root).ok_or("root type missing")?;
    let mut root_cls = format!(
"class {root}:
    \"\"\"The model. Owns the underlying Rust instance; free() (or GC)
    releases it. One model, one thread.\"\"\"

    def __init__(self):
        load()
        self._p = _f.{root_sn}_new()

    def free(self):
        if self._p:
            _f.{root_sn}_free(self._p)
            self._p = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def _solved(self, code, res):
        if code < 0:
            failure = None
            if code == -1:
                fl = SolveFailure()
                if _f.{root_sn}_last_failure(self._p, ctypes.byref(fl)):
                    failure = fl
            raise AraelError(code, _err(self._p),
                             res if res._detail else None, failure)
        return res

    def solve_dense(self, cfg=None):
        cfg = cfg if cfg is not None else LmConfig()
        r = LmResult()
        return self._solved(
            _f.{root_sn}_solve_dense(self._p, ctypes.byref(cfg),
                                     ctypes.byref(r)), r)

    def solve_sparse(self, cfg=None, opts=None):
        \"\"\"Sparse solve; `opts` (a SparseOptions) selects the
        backend's route, None means the defaults.\"\"\"
        cfg = cfg if cfg is not None else LmConfig()
        r = LmResult()
        return self._solved(
            _f.{root_sn}_solve_sparse(
                self._p, ctypes.byref(cfg),
                ctypes.byref(opts) if opts is not None else None,
                ctypes.byref(r)), r)

    def solve_band(self, kd, cfg=None):
        \"\"\"Band Cholesky solve; kd is the half-bandwidth in scalar
        parameters.\"\"\"
        cfg = cfg if cfg is not None else LmConfig()
        r = LmResult()
        return self._solved(
            _f.{root_sn}_solve_band(self._p, kd, ctypes.byref(cfg),
                                    ctypes.byref(r)), r)

    def assemble_covariance(self, mode=CovMode.ALL_MARGINALS,
                            ordering=CovOrdering.AUTO,
                            block_supernodal=BlockSupernodalMode.AUTO):
        \"\"\"Prepare the covariance at the current (solved) parameters.

        `ordering` and `block_supernodal` decide what producing it costs,
        never what it is. ALL_MARGINALS ignores `block_supernodal` and stays
        on the scalar factor.\"\"\"
        c = ctypes.c_void_p()
        code = _f.{root_sn}_assemble_covariance_with(
            self._p, int(mode), int(ordering), int(block_supernodal),
            ctypes.byref(c))
        if code != 0:
            raise AraelError(code, _err(self._p))
        return Covariance(c)

    def cost(self):
        \"\"\"Total cost at the current parameter values (no solve).\"\"\"
        return _f.{root_sn}_cost(self._p)

{ct_method}    def validate(self):
        \"\"\"Empty string when the model is clean, the diagnostic text
        otherwise.\"\"\"
        return _f.{root_sn}_validate(self._p).decode()

    def last_error(self):
        return _err(self._p)

");
    for f in &root_ty.fields {
        field_py(&mut py, model, root, &root_sn, &mut root_cls, f)?;
    }
    py.body.push_str(&root_cls);

    let ffi_mod = format!(
"# GENERATED by cargo-arael from the `{root}` model sidecar. Do not
# edit; regenerate with `cargo arael export`.
\"\"\"ctypes signature table for the `{root}` C ABI; load() in the API
module binds it to the cdylib.\"\"\"

import ctypes

from .arael import math as _m
from .arael import solver as _solver

_T = _solver.lm_types({fp})
LmConfigRaw = _T[\"LmConfig\"]
LmResultRaw = _T[\"LmResult\"]
LmIter = _T[\"LmIter\"]
SparseOptionsRaw = _solver.SparseOptions

SIGS = [
{sigs}]


def bind(lib):
    g = globals()
    for name, argtypes, restype in SIGS:
        fn = getattr(lib, name)
        fn.argtypes = argtypes
        fn.restype = restype
        g[name] = fn
", sigs = py.sigs);

    let api_mod = format!(
"# GENERATED by cargo-arael from the `{root}` model sidecar. Do not
# edit; regenerate with `cargo arael export`.
\"\"\"Python interface for the `{root}` model: build the problem, solve,
read the results back. Pure ctypes over the capi cdylib; see
docs/CXX.md for the shared surface semantics (this module mirrors the
C++ classes one-to-one).\"\"\"

import ctypes
import os
import struct

from . import _{root_sn}_ffi as _f
from .arael import columns as _cols
from .arael import math as _m
from .arael import transform as _tf
from .arael.solver import (AraelError, BlockSupernodalMode, CovMode,
                           CovOrdering, CovPlan, DiagonalFault, EnvelopeMode,
                           FaerOrdering, LmPreset, LmStatus, LmStep,
                           LmTiming, LogLevel, ReducedOrdering, SchurPlan,
                           SchurPolicy, SchurSolve, SolveFailure,
                           SolveFailureKind)

LmIter = _f.LmIter

_lib = None


def load(path=None):
    \"\"\"Load and bind the capi cdylib. Called implicitly on first
    model construction; call explicitly to pick a specific library.
    Resolution: explicit path, $ARAEL_CAPI, then the conventional
    cargo build locations next to the package.\"\"\"
    global _lib
    if _lib is not None:
        return
    pkg = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    names = [\"lib{{}}.so\", \"lib{{}}.dylib\", \"{{}}.dll\"]
    tried = []
    candidates = [path, os.environ.get(\"ARAEL_CAPI\")]
    for rel in (\"../capi/target\", \"../target\", \"../../target\"):
        for profile in (\"release\", \"debug\"):
            for n in names:
                candidates.append(os.path.join(
                    pkg, rel, profile, n.format(\"{lib_ident}\")))
    for c in candidates:
        if not c:
            continue
        if not os.path.exists(c):
            tried.append(c)
            continue
        _lib = ctypes.CDLL(c)
        _f.bind(_lib)
        return
    raise AraelError(-1, \"no capi cdylib found; build it with \"
        \"`cargo build --release -p <crate>-capi` or set ARAEL_CAPI. \"
        \"Tried: \" + \", \".join(tried))


def set_log_level(level):
    \"\"\"Drop arael log messages above `level` (a LogLevel; INFO --
    everything -- is the default). Process-wide: all models and roots
    share it.\"\"\"
    load()
    _f.{root_sn}_set_log_level(int(level))


def _raw(r):
    return r.raw if hasattr(r, \"raw\") else int(r)


def _err(h):
    return (_f.{root_sn}_last_error(h) or b\"\").decode()


class LmConfig(_f.LmConfigRaw):
    \"\"\"The solver configuration, holding the chosen preset's actual
    Rust values (fetched through the FFI at construction).\"\"\"

    def __init__(self, preset=LmPreset.DEFAULTS):
        load()
        _f.{root_sn}_lm_config(int(preset), ctypes.byref(self))

    @classmethod
    def defaults(cls):
        return cls(LmPreset.DEFAULTS)

    @classmethod
    def conservative(cls):
        return cls(LmPreset.CONSERVATIVE)

    @classmethod
    def well_conditioned(cls):
        return cls(LmPreset.WELL_CONDITIONED)

    @classmethod
    def ill_conditioned(cls):
        return cls(LmPreset.ILL_CONDITIONED)


class SparseOptions(_f.SparseOptionsRaw):
    \"\"\"The sparse backend's options, holding the actual Rust
    defaults (fetched through the FFI at construction). Edit fields,
    pass to solve_sparse.\"\"\"

    def __init__(self):
        load()
        _f.{root_sn}_sparse_options(ctypes.byref(self))


class LmSession:
    \"\"\"Warm reuse over repeated sparse solves: keeps the analysis
    (pattern, ordering, symbolic factorization, Schur plan) across
    solves, so only the first pays for it. Warm solves are
    bit-identical to cold ones. A parameter-count change re-analyzes
    by itself; call invalidate() after a structural change at the
    same count (solving warm through one is undefined).\"\"\"

    def __init__(self, opts=None):
        load()
        self._s = _f.{root_sn}_session_new(
            ctypes.byref(opts) if opts is not None else None)

    def free(self):
        if self._s:
            _f.{root_sn}_session_free(self._s)
            self._s = None

    def __del__(self):
        try:
            self.free()
        except Exception:
            pass

    def solve(self, model, cfg=None):
        \"\"\"Solve through the session; contract as solve_sparse.\"\"\"
        cfg = cfg if cfg is not None else LmConfig()
        r = LmResult()
        return model._solved(
            _f.{root_sn}_session_solve(self._s, model._p, ctypes.byref(cfg),
                                       ctypes.byref(r)), r)

    def invalidate(self):
        \"\"\"Drop the learned structure; the next solve runs cold.\"\"\"
        _f.{root_sn}_session_invalidate(self._s)


class LmResult(_f.LmResultRaw):
    \"\"\"A completed solve (see arael.solver for the fields); owns the
    full Rust-side result until garbage collected.\"\"\"

    def report(self):
        \"\"\"Text report: status, cost, iterations, damping, plus the
        timing breakdown and the backend's plan when gathered.\"\"\"
        if not self._detail:
            return \"\"
        return _f.{root_sn}_result_report(self._detail, False).decode()

    def pretty_report(self):
        \"\"\"report() with colour and box-drawing glyphs.\"\"\"
        if not self._detail:
            return \"\"
        return _f.{root_sn}_result_report(self._detail, True).decode()

    @property
    def plan(self):
        \"\"\"The sparse backend's SchurPlan, or None when the solve
        carried none (dense and band solves).\"\"\"
        p = SchurPlan()
        if self._detail and _f.{root_sn}_result_plan(self._detail,
                                                     ctypes.byref(p)):
            return p
        return None

    @property
    def steps(self):
        \"\"\"The per-attempt timeline (list of arael.solver.LmStep);
        empty unless the solve ran with gather_timing.\"\"\"
        if not self._detail:
            return []
        n = _f.{root_sn}_result_steps(self._detail, None, 0)
        if not n:
            return []
        buf = (LmStep * n)()
        _f.{root_sn}_result_steps(self._detail, buf, n)
        return list(buf)

    def __del__(self):
        d, self._detail = self._detail, None
        if d:
            try:
                _f.{root_sn}_result_free(d)
            except Exception:
                pass


{body}", body = py.body);

    Ok((ffi_mod, api_mod))
}
