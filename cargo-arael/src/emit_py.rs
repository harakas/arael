//! Python module emitter: pure-ctypes bindings over the same C ABI
//! the C++ header wraps, as two files per root -- `_{root}_ffi.py`
//! (the signature table) and `{root}.py` (the API classes). Mirrors
//! the C++ classes one-to-one, idiomatic where Python is: fields are
//! properties, collections speak len/[]/iteration, absent options are
//! None, and failures raise AraelError.

use crate::emit_ffi::surfaced_types;
use crate::ir::{Field, Model, Type, snake};

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

fn ct_math(of: &str) -> Option<&'static str> {
    Some(match of {
        "vect2f" => "_m.vect2f",
        "vect2d" => "_m.vect2d",
        "vect3f" => "_m.vect3f",
        "vect3d" => "_m.vect3d",
        "matrix2f" => "_m.matrix2f",
        "matrix2d" => "_m.matrix2d",
        "matrix3f" => "_m.matrix3f",
        "matrix3d" => "_m.matrix3d",
        "quaternf" => "_m.quaternf",
        "quaternd" => "_m.quaternd",
        _ => return None,
    })
}

/// (ctypes type, needs sequence coercion) of a get/set field.
fn value_ct(f: &Field) -> Option<(&'static str, bool)> {
    let of = f.of.as_deref().unwrap_or("");
    match f.kind.as_str() {
        "data" | "param" => ct_scalar(of).map(|t| (t, false))
            .or_else(|| ct_math(of).map(|t| (t, true))),
        "euler_param" => {
            let s = f.scalar.as_deref().unwrap_or("f64");
            Some(match (f.variant.as_deref().unwrap_or("simple"), s) {
                ("rotvec", "f32") => ("_m.quaternf", true),
                ("rotvec", _) => ("_m.quaternd", true),
                (_, "f32") => ("_m.vect3f", true),
                (_, _) => ("_m.vect3d", true),
            })
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

/// One collection field's view class + the owner property line.
fn collection_py(
    py: &mut Py,
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

    sig(py, &format!("{prefix}_len"), &["ctypes.c_void_p"], "ctypes.c_uint32");
    sig(py, &format!("{prefix}_reserve"),
        &["ctypes.c_void_p", "ctypes.c_uint32"], "None");

    let mut cls = format!(
"class {view}:
    \"\"\"View of `{field}` ({} of {elem}); element wrappers re-resolve
    their pointer by key on every access, so growing the collection
    cannot leave them dangling. Mutating while iterating is undefined.\"\"\"

    __slots__ = (\"_p\",)

    def __init__(self, p):
        self._p = p

    def __len__(self):
        return _f.{prefix}_len(self._p)

    def reserve(self, additional):
        _f.{prefix}_reserve(self._p, additional)

", container);

    match container {
        "vec" | "deque" => {
            sig(py, &format!("{prefix}_at"),
                &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_void_p");
            sig(py, &format!("{prefix}_clear"), &["ctypes.c_void_p"], "None");
            sig(py, &format!("{prefix}_truncate"),
                &["ctypes.c_void_p", "ctypes.c_uint32"], "None");
            // Key the freshly pushed element: a Ref where the collection
            // has generations (so a later removal invalidates it loudly),
            // otherwise its index.
            let new_key = if refs_flavor {
                "self.ref_at(len(self) - 1)"
            } else {
                "len(self) - 1"
            };
            let front_key = if refs_flavor { "self.ref_at(0)" } else { "0" };
            let getitem_ref = if refs_flavor {
                format!(
"        if isinstance(i, {elem}Ref):
            return {elem}(lambda r=i.raw: _f.{prefix}_get(self._p, r))
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
        return {elem}(lambda i=i: _f.{prefix}_at(self._p, i))

    def __iter__(self):
        for i in range(len(self)):
            yield {elem}(lambda i=i: _f.{prefix}_at(self._p, i))

    def clear(self):
        _f.{prefix}_clear(self._p)

    def truncate(self, n):
        _f.{prefix}_truncate(self._p, n)

"));
            if container == "vec" {
                sig(py, &format!("{prefix}_push"), &["ctypes.c_void_p"],
                    "ctypes.c_void_p");
                sig(py, &format!("{prefix}_pop"), &["ctypes.c_void_p"],
                    "ctypes.c_bool");
                cls.push_str(&format!(
"    def push(self):
        _f.{prefix}_push(self._p)
        return self[{new_key}]

    def pop(self):
        \"\"\"Drops the last element; False when already empty.\"\"\"
        return _f.{prefix}_pop(self._p)

"));
            } else {
                for (m, new_key) in [("push_back", new_key), ("push_front", front_key)] {
                    sig(py, &format!("{prefix}_{m}"), &["ctypes.c_void_p"],
                        "ctypes.c_void_p");
                    cls.push_str(&format!(
"    def {m}(self):
        _f.{prefix}_{m}(self._p)
        return self[{new_key}]

"));
                }
                for m in ["pop_back", "pop_front"] {
                    sig(py, &format!("{prefix}_{m}"), &["ctypes.c_void_p"],
                        "ctypes.c_bool");
                    cls.push_str(&format!(
"    def {m}(self):
        \"\"\"Drops one end; False when already empty.\"\"\"
        return _f.{prefix}_{m}(self._p)

"));
                }
            }
        }
        "arena" => {
            sig(py, &format!("{prefix}_push"), &["ctypes.c_void_p"],
                "ctypes.c_uint32");
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
            cls.push_str(&format!(
"    def push(self):
        \"\"\"New element's ref (get()/[] take it back).\"\"\"
        return {elem}Ref(_f.{prefix}_push(self._p))

    def remove(self, r):
        return _f.{prefix}_remove(self._p, _raw(r))

    def clear(self):
        _f.{prefix}_clear(self._p)

    def __getitem__(self, r):
        return {elem}(lambda k=_raw(r): _f.{prefix}_get(self._p, k))

    def __iter__(self):
        \"\"\"Live slots in order; yields element wrappers (refs() for
        the refs).\"\"\"
        r = _f.{prefix}_first(self._p)
        while r != 0xFFFFFFFF:
            yield {elem}(lambda k=r: _f.{prefix}_get(self._p, k))
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
        return {elem}(lambda k=_raw(r): _f.{prefix}_get(self._p, k))

    def __contains__(self, r):
        return _f.{prefix}_contains(self._p, _raw(r))

    def try_get(self, r):
        \"\"\"The element, or None for a stale or foreign ref.\"\"\"
        p = _f.{prefix}_try_get(self._p, _raw(r))
        return {elem}(lambda k=_raw(r): _f.{prefix}_get(self._p, k)) if p else None

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
            prop(py, owner_cls, prefix, name, name, ct, coerce);
            if f.kind != "data" {
                optimize_prop(py, owner_cls, prefix, name);
            }
        }
        "component" => match of {
            "TransformParam" | "TransformParamF" => {
                let (v3, q) = if of == "TransformParamF" {
                    ("_m.vect3f", "_m.quaternf")
                } else {
                    ("_m.vect3d", "_m.quaternd")
                };
                let p2 = format!("{prefix}_{name}");
                prop(py, owner_cls, &p2, "translation",
                     &format!("{name}_translation"), v3, true);
                prop(py, owner_cls, &p2, "rotation",
                     &format!("{name}_rotation"), q, true);
                for flag in ["optimize_translation", "optimize_rotation"] {
                    prop(py, owner_cls, &p2, flag,
                         &format!("{name}_{flag}"), "ctypes.c_bool", false);
                }
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
            owner_cls.push_str(&format!(
"    @property
    def {name}(self):
        \"\"\"The `{of}`, or None while absent (make_{name}() creates).\"\"\"
        if not _f.{prefix}_{name}(self._p):
            return None
        return {of}(lambda: _f.{prefix}_{name}(self._p))

    def make_{name}(self):
        _f.{prefix}_make_{name}(self._p)
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
        "collection" => collection_py(py, owner, prefix, owner_cls, f)?,
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
        let mut cls = format!(
"class {tn}:
    \"\"\"A `{tn}` in its owner's storage, addressed by key rather than by
    pointer: the pointer is re-resolved on every access, so growing the
    collection cannot leave this wrapper dangling.\"\"\"

    __slots__ = (\"_at\",)
    param_count = {}

    def __init__(self, at):
        # Zero-argument callable returning a currently-valid pointer.
        self._at = at

    @property
    def _p(self):
        return self._at()

", t.param_count);
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
        &["ctypes.c_void_p", "ctypes.c_uint32"], "ctypes.c_int32");
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
            raise AraelError(rows, _err(self._h))
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
"def _cov_query(h, fn, p, cap):
    buf = (ctypes.c_double * cap)()
    n = fn(h, p, buf, cap)
    if n < 0:
        raise AraelError(n, _err(h))
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
    \"\"\"Covariance prepared at the solution; per-entity queries
    (marginal/conditional by param count: 1 -> float, 2/3 ->
    matrix2d/3d, larger -> row-major tuples). Valid until the model is
    dropped or reassembled.\"\"\"

    __slots__ = (\"_h\",)

    def __init__(self, h):
        self._h = h

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
    sig(&mut py, &format!("{root_sn}_validate"), &["ctypes.c_void_p"],
        "ctypes.c_char_p");
    sig(&mut py, &format!("{root_sn}_result_report"),
        &["ctypes.c_void_p", "ctypes.c_bool"], "ctypes.c_char_p");
    sig(&mut py, &format!("{root_sn}_result_plan"),
        &["ctypes.c_void_p", "ctypes.POINTER(_solver.SchurPlan)"],
        "ctypes.c_bool");
    sig(&mut py, &format!("{root_sn}_result_free"), &["ctypes.c_void_p"],
        "None");
    sig(&mut py, &format!("{root_sn}_cost"), &["ctypes.c_void_p"],
        "ctypes.c_double");
    sig(&mut py, &format!("{root_sn}_lm_config"),
        &["ctypes.c_uint32", "ctypes.POINTER(LmConfigRaw)"], "None");
    for m in ["solve_dense", "solve_sparse"] {
        sig(&mut py, &format!("{root_sn}_{m}"),
            &["ctypes.c_void_p", "ctypes.POINTER(LmConfigRaw)",
              "ctypes.POINTER(LmResultRaw)"], "ctypes.c_int32");
    }
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
            raise AraelError(code, _err(self._p),
                             res if res._detail else None)
        return res

    def solve_dense(self, cfg=None):
        cfg = cfg if cfg is not None else LmConfig()
        r = LmResult()
        return self._solved(
            _f.{root_sn}_solve_dense(self._p, ctypes.byref(cfg),
                                     ctypes.byref(r)), r)

    def solve_sparse(self, cfg=None):
        cfg = cfg if cfg is not None else LmConfig()
        r = LmResult()
        return self._solved(
            _f.{root_sn}_solve_sparse(self._p, ctypes.byref(cfg),
                                      ctypes.byref(r)), r)

    def solve_band(self, kd, cfg=None):
        \"\"\"Band Cholesky solve; kd is the half-bandwidth in scalar
        parameters.\"\"\"
        cfg = cfg if cfg is not None else LmConfig()
        r = LmResult()
        return self._solved(
            _f.{root_sn}_solve_band(self._p, kd, ctypes.byref(cfg),
                                    ctypes.byref(r)), r)

    def assemble_covariance(self, mode=CovMode.ALL_MARGINALS):
        code = _f.{root_sn}_assemble_covariance(self._p, int(mode))
        if code != 0:
            raise AraelError(code, _err(self._p))
        return Covariance(self._p)

    def cost(self):
        \"\"\"Total cost at the current parameter values (no solve).\"\"\"
        return _f.{root_sn}_cost(self._p)

    def validate(self):
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

from . import _{root_sn}_ffi as _f
from .arael import math as _m
from .arael.solver import (AraelError, CovMode, LmPreset, LmStatus,
                           LmTiming, ReducedOrdering, SchurPlan)

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
