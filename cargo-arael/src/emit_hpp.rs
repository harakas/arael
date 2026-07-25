//! C++ header emitter: thin inline wrapper classes over the shim's C
//! ABI, using the arael math headers for value types. Entity classes
//! hold raw pointers (stable for refs-flavored containers;
//! std::vec::Vec-backed ones invalidate on push -- each view's comment
//! says which). Classes are emitted children-first (containment is
//! cycle-free), so nested views can construct element wrappers inline.

use crate::emit_ffi::surfaced_types;
use crate::ir::{Field, Model, Type, snake};

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

/// Math type -> the arael C++ math type (layout-matched to the shim's
/// repr(C) mirrors).
fn math_cpp(of: &str) -> Option<&'static str> {
    Some(match of {
        "vect2f" => "vect2f",
        "vect2d" => "vect2d",
        "vect3f" => "vect3f",
        "vect3d" => "vect3d",
        "matrix2f" => "matrix2f",
        "matrix2d" => "matrix2d",
        "matrix3f" => "matrix3f",
        "matrix3d" => "matrix3d",
        "quaternf" => "quaternf",
        "quaternd" => "quaternd",
        _ => return None,
    })
}

fn float_cpp(precision: &str) -> &'static str {
    if precision == "f32" { "float" } else { "double" }
}

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

/// Value type of a get/set field as (C++ type, ffi decl type) --
/// identical strings here since the extern decls use the arael math
/// types directly.
fn value_type(f: &Field) -> Option<&'static str> {
    let of = f.of.as_deref().unwrap_or("");
    match f.kind.as_str() {
        "data" | "param" => scalar_cpp(of).or_else(|| math_cpp(of)),
        "euler_param" => {
            let s = f.scalar.as_deref().unwrap_or("f64");
            Some(match (f.variant.as_deref().unwrap_or("simple"), s) {
                ("rotvec", "f32") => "quaternf",
                ("rotvec", _) => "quaternd",
                (_, "f32") => "vect3f",
                (_, _) => "vect3d",
            })
        }
        _ => None,
    }
}

struct Cpp {
    ffi: String,
    body: String,
}

/// One collection field's view class + the owner method line.
fn collection_cpp(
    cpp: &mut Cpp,
    owner: &str,
    sym_prefix: &str,
    owner_methods: &mut String,
    f: &Field,
) -> Result<(), String> {
    let field = &f.name;
    let elem = f.of.as_deref().ok_or("collection without element")?;
    let prefix = format!("{sym_prefix}_{field}");
    let container = f.container.as_deref().unwrap_or("vec");
    let refs_flavor = f.spelled.as_deref().unwrap_or("").contains("refs::");
    // The view class is named by the container's nature -- nothing else
    // about the wrapper says how it stores or behaves.
    let kind = match container { "deque" => "Deque", "arena" => "Arena", _ => "Vec" };
    let view = format!("{owner}{}{kind}", camel(field));

    cpp.ffi.push_str(&format!(
        "uint32_t {prefix}_len(const {owner}*);\n\
         void {prefix}_reserve({owner}*, uint32_t);\n"));
    let mut methods = format!(
        "    uint32_t size() const {{ return ffi::{prefix}_len(h_); }}\n\
         \x20   bool empty() const {{ return size() == 0; }}\n\
         \x20   void reserve(uint32_t additional) {{ ffi::{prefix}_reserve(h_, additional); }}\n");
    match container {
        "vec" => {
            cpp.ffi.push_str(&format!(
                "{elem}* {prefix}_push({owner}*);\n{elem}* {prefix}_at({owner}*, uint32_t);\n"));
            cpp.ffi.push_str(&format!(
                "bool {prefix}_pop({owner}*);\nvoid {prefix}_clear({owner}*);\n\
                 void {prefix}_truncate({owner}*, uint32_t);\n"));
            methods.push_str(&format!(
                "    {elem} push() {{ return {elem}(ffi::{prefix}_push(h_)); }}\n\
                 \x20   {elem} operator[](uint32_t i) {{ return {elem}(ffi::{prefix}_at(h_, i)); }}\n\
                 \x20   /// Front/back of a non-empty vec (empty = UB, like STL).\n\
                 \x20   {elem} front() {{ return (*this)[0]; }}\n\
                 \x20   {elem} back() {{ return (*this)[size() - 1]; }}\n\
                 \x20   /// Drops the last element; false when already empty.\n\
                 \x20   bool pop() {{ return ffi::{prefix}_pop(h_); }}\n\
                 \x20   void clear() {{ ffi::{prefix}_clear(h_); }}\n\
                 \x20   void truncate(uint32_t n) {{ ffi::{prefix}_truncate(h_, n); }}\n"));
        }
        "deque" => {
            cpp.ffi.push_str(&format!(
                "{elem}* {prefix}_push_back({owner}*);\n\
                 {elem}* {prefix}_push_front({owner}*);\n\
                 {elem}* {prefix}_at({owner}*, uint32_t);\n"));
            cpp.ffi.push_str(&format!(
                "bool {prefix}_pop_back({owner}*);\nbool {prefix}_pop_front({owner}*);\n\
                 void {prefix}_clear({owner}*);\nvoid {prefix}_truncate({owner}*, uint32_t);\n"));
            methods.push_str(&format!(
                "    {elem} push_back() {{ return {elem}(ffi::{prefix}_push_back(h_)); }}\n\
                 \x20   {elem} push_front() {{ return {elem}(ffi::{prefix}_push_front(h_)); }}\n\
                 \x20   {elem} operator[](uint32_t i) {{ return {elem}(ffi::{prefix}_at(h_, i)); }}\n\
                 \x20   /// Front/back of a non-empty deque (empty = UB, like STL).\n\
                 \x20   {elem} front() {{ return (*this)[0]; }}\n\
                 \x20   {elem} back() {{ return (*this)[size() - 1]; }}\n\
                 \x20   /// Drop one end; false when already empty.\n\
                 \x20   bool pop_back() {{ return ffi::{prefix}_pop_back(h_); }}\n\
                 \x20   bool pop_front() {{ return ffi::{prefix}_pop_front(h_); }}\n\
                 \x20   void clear() {{ ffi::{prefix}_clear(h_); }}\n\
                 \x20   void truncate(uint32_t n) {{ ffi::{prefix}_truncate(h_, n); }}\n"));
        }
        "arena" => {
            cpp.ffi.push_str(&format!(
                "uint32_t {prefix}_push({owner}*);\nbool {prefix}_remove({owner}*, uint32_t);\n"));
            cpp.ffi.push_str(&format!("void {prefix}_clear({owner}*);\n"));
            methods.push_str(&format!(
                "    {elem}Ref push() {{ return {elem}Ref{{ffi::{prefix}_push(h_)}}; }}\n\
                 \x20   bool remove({elem}Ref r) {{ return ffi::{prefix}_remove(h_, r.raw); }}\n\
                 \x20   void clear() {{ ffi::{prefix}_clear(h_); }}\n"));
        }
        other => return Err(format!("unknown container `{other}`")),
    }
    if refs_flavor {
        // An arena has holes, so index-based refs are meaningless there:
        // refs come from push().
        if container != "arena" {
            cpp.ffi.push_str(&format!(
                "uint32_t {prefix}_ref_at(const {owner}*, uint32_t);\n"));
            methods.push_str(&format!(
                "    {elem}Ref ref_at(uint32_t i) const {{ return {elem}Ref{{ffi::{prefix}_ref_at(h_, i)}}; }}\n"));
        }
        cpp.ffi.push_str(&format!(
            "{elem}* {prefix}_get({owner}*, uint32_t);\n\
             bool {prefix}_contains(const {owner}*, uint32_t);\n\
             {elem}* {prefix}_try_get({owner}*, uint32_t);\n"));
        methods.push_str(&format!(
            "    {elem} get({elem}Ref r) {{ return {elem}(ffi::{prefix}_get(h_, r.raw)); }}\n\
             \x20   /// True while r addresses a live element here.\n\
             \x20   bool contains({elem}Ref r) const {{ return ffi::{prefix}_contains(h_, r.raw); }}\n\
             \x20   /// Like get, but empty for a stale or foreign ref.\n\
             \x20   option<{elem}> try_get({elem}Ref r) {{\n\
             \x20       ffi::{elem}* p = ffi::{prefix}_try_get(h_, r.raw);\n\
             \x20       return p ? option<{elem}>({elem}(p)) : option<{elem}>();\n\
             \x20   }}\n"));
    }
    if container == "arena" {
        cpp.ffi.push_str(&format!(
            "uint32_t {prefix}_first(const {owner}*);\n\
             uint32_t {prefix}_next(const {owner}*, uint32_t);\n\
             uint32_t {prefix}_last(const {owner}*);\n\
             uint32_t {prefix}_prev(const {owner}*, uint32_t);\n"));
        methods.push_str(&format!(
"    /// Bidirectional iterator over the live slots. Standard C++
    /// contract: modifying the container while iterating is undefined
    /// behavior. Dereference yields a value wrapper ({elem}), like
    /// vector<bool> -- reference is a value type.
    class iterator {{
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = {elem};
        using difference_type = std::ptrdiff_t;
        using reference = {elem};
        struct arrow {{ {elem} v; {elem}* operator->() {{ return &v; }} }};
        using pointer = arrow;

        iterator() : h_(nullptr), r_(UINT32_MAX) {{}}
        iterator(ffi::{owner}* h, uint32_t r) : h_(h), r_(r) {{}}
        {elem} operator*() const {{ return {elem}(ffi::{prefix}_get(h_, r_)); }}
        arrow operator->() const {{ return arrow{{**this}}; }}
        {elem}Ref ref() const {{ return {elem}Ref{{r_}}; }}
        iterator& operator++() {{ r_ = ffi::{prefix}_next(h_, r_); return *this; }}
        iterator& operator--() {{
            r_ = r_ == UINT32_MAX ? ffi::{prefix}_last(h_)
                                  : ffi::{prefix}_prev(h_, r_);
            return *this;
        }}
        iterator operator++(int) {{ iterator t = *this; ++*this; return t; }}
        iterator operator--(int) {{ iterator t = *this; --*this; return t; }}
        bool operator==(const iterator& o) const {{ return r_ == o.r_; }}
        bool operator!=(const iterator& o) const {{ return r_ != o.r_; }}
    private:
        ffi::{owner}* h_;
        uint32_t r_;
    }};
    iterator begin() {{ return iterator(h_, ffi::{prefix}_first(h_)); }}
    iterator end() {{ return iterator(h_, UINT32_MAX); }}\n"));
    } else {
        methods.push_str(&format!(
"    /// Bidirectional iterator. Standard C++ contract: modifying the
    /// container while iterating is undefined behavior. Dereference
    /// yields a value wrapper ({elem}), like vector<bool> --
    /// reference is a value type.
    class iterator {{
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = {elem};
        using difference_type = std::ptrdiff_t;
        using reference = {elem};
        struct arrow {{ {elem} v; {elem}* operator->() {{ return &v; }} }};
        using pointer = arrow;

        iterator() : h_(nullptr), i_(0) {{}}
        iterator(ffi::{owner}* h, uint32_t i) : h_(h), i_(i) {{}}
        {elem} operator*() const {{ return {elem}(ffi::{prefix}_at(h_, i_)); }}
        arrow operator->() const {{ return arrow{{**this}}; }}
        iterator& operator++() {{ ++i_; return *this; }}
        iterator& operator--() {{ --i_; return *this; }}
        iterator operator++(int) {{ iterator t = *this; ++*this; return t; }}
        iterator operator--(int) {{ iterator t = *this; --*this; return t; }}
        bool operator==(const iterator& o) const {{ return i_ == o.i_; }}
        bool operator!=(const iterator& o) const {{ return i_ != o.i_; }}
    private:
        ffi::{owner}* h_;
        uint32_t i_;
    }};
    iterator begin() {{ return iterator(h_, 0); }}
    iterator end() {{ return iterator(h_, size()); }}\n"));
    }

    // Reverse iteration, shared by every container kind: hand-rolled
    // instead of std::reverse_iterator, whose C++17 operator-> takes
    // the address of a temporary for proxy (value-reference)
    // iterators.
    methods.push_str(&format!(
"    class reverse_iterator {{
    public:
        using iterator_category = std::bidirectional_iterator_tag;
        using value_type = {elem};
        using difference_type = std::ptrdiff_t;
        using reference = {elem};
        using pointer = iterator::arrow;

        reverse_iterator() {{}}
        explicit reverse_iterator(iterator base) : base_(base) {{}}
        iterator base() const {{ return base_; }}
        {elem} operator*() const {{ iterator t = base_; --t; return *t; }}
        pointer operator->() const {{ return pointer{{**this}}; }}
        reverse_iterator& operator++() {{ --base_; return *this; }}
        reverse_iterator& operator--() {{ ++base_; return *this; }}
        reverse_iterator operator++(int) {{ reverse_iterator t = *this; ++*this; return t; }}
        reverse_iterator operator--(int) {{ reverse_iterator t = *this; --*this; return t; }}
        bool operator==(const reverse_iterator& o) const {{ return base_ == o.base_; }}
        bool operator!=(const reverse_iterator& o) const {{ return base_ != o.base_; }}
    private:
        iterator base_;
    }};
    reverse_iterator rbegin() {{ return reverse_iterator(end()); }}
    reverse_iterator rend() {{ return reverse_iterator(begin()); }}\n"));

    let stability = if refs_flavor {
        "Element pointers are STABLE across pushes (chunked storage)."
    } else {
        "std::vec::Vec storage: pushes may MOVE elements -- re-fetch element refs after a push."
    };
    cpp.body.push_str(&format!(
"/// `{owner}.{field}`. {stability}
class {view} {{
public:
    explicit {view}(ffi::{owner}* h) : h_(h) {{}}
{methods}private:
    ffi::{owner}* h_;
}};

"));
    owner_methods.push_str(&format!(
        "    {view} {field}() {{ return {view}(h_); }}\n"));
    Ok(())
}

/// Owner-class methods (+ ffi decls, + any view classes) for one field.
/// `owner` names the opaque ffi pointer type the accessors take.
fn field_cpp(
    cpp: &mut Cpp,
    _model: &Model,
    owner: &str,
    prefix: &str,
    owner_methods: &mut String,
    f: &Field,
) -> Result<(), String> {
    let name = &f.name;
    let of = f.of.as_deref().unwrap_or("");
    match f.kind.as_str() {
        "data" | "param" | "euler_param" => {
            if let Some(t) = value_type(f) {
                cpp.ffi.push_str(&format!(
                    "{t} {prefix}_{name}(const {owner}*);\nvoid {prefix}_set_{name}({owner}*, {t});\n"));
                owner_methods.push_str(&format!(
                    "    {t} {name}() const {{ return ffi::{prefix}_{name}(h_); }}\n\
                     \x20   void set_{name}({t} v) {{ ffi::{prefix}_set_{name}(h_, v); }}\n"));
            } else {
                return Err(format!("`{owner}.{name}`: unsupported {} of {of}", f.kind));
            }
            if f.kind != "data" {
                cpp.ffi.push_str(&format!(
                    "bool {prefix}_{name}_optimize(const {owner}*);\n\
                     void {prefix}_{name}_set_optimize({owner}*, bool);\n"));
                owner_methods.push_str(&format!(
                    "    bool {name}_optimize() const {{ return ffi::{prefix}_{name}_optimize(h_); }}\n\
                     \x20   void set_{name}_optimize(bool v) {{ ffi::{prefix}_{name}_set_optimize(h_, v); }}\n"));
            }
        }
        "component" => match of {
            "TransformParam" | "TransformParamF" => {
                let (v3, q) = if of == "TransformParamF" {
                    ("vect3f", "quaternf")
                } else {
                    ("vect3d", "quaternd")
                };
                for (part, t) in [("translation", v3), ("rotation", q)] {
                    cpp.ffi.push_str(&format!(
                        "{t} {prefix}_{name}_{part}(const {owner}*);\n\
                         void {prefix}_{name}_set_{part}({owner}*, {t});\n"));
                    owner_methods.push_str(&format!(
                        "    {t} {name}_{part}() const {{ return ffi::{prefix}_{name}_{part}(h_); }}\n\
                         \x20   void set_{name}_{part}({t} v) {{ ffi::{prefix}_{name}_set_{part}(h_, v); }}\n"));
                }
                for flag in ["optimize_translation", "optimize_rotation"] {
                    cpp.ffi.push_str(&format!(
                        "bool {prefix}_{name}_{flag}(const {owner}*);\n\
                         void {prefix}_{name}_set_{flag}({owner}*, bool);\n"));
                    owner_methods.push_str(&format!(
                        "    bool {name}_{flag}() const {{ return ffi::{prefix}_{name}_{flag}(h_); }}\n\
                         \x20   void set_{name}_{flag}(bool v) {{ ffi::{prefix}_{name}_set_{flag}(h_, v); }}\n"));
                }
            }
            "UnitVecParam" | "UnitVecParamF" => {
                let v3 = if of == "UnitVecParamF" { "vect3f" } else { "vect3d" };
                cpp.ffi.push_str(&format!(
                    "{v3} {prefix}_{name}_unit(const {owner}*);\n\
                     void {prefix}_{name}_set_unit({owner}*, {v3});\n"));
                owner_methods.push_str(&format!(
                    "    {v3} {name}_unit() const {{ return ffi::{prefix}_{name}_unit(h_); }}\n\
                     \x20   void set_{name}_unit({v3} v) {{ ffi::{prefix}_{name}_set_unit(h_, v); }}\n"));
            }
            _ => {
                cpp.ffi.push_str(&format!("{of}* {prefix}_{name}_ptr({owner}*);\n"));
                owner_methods.push_str(&format!(
                    "    {of} {name}() {{ return {of}(ffi::{prefix}_{name}_ptr(h_)); }}\n"));
            }
        },
        "struct" => {
            cpp.ffi.push_str(&format!("{of}* {prefix}_{name}_ptr({owner}*);\n"));
            owner_methods.push_str(&format!(
                "    {of} {name}() {{ return {of}(ffi::{prefix}_{name}_ptr(h_)); }}\n"));
        }
        "optional" => {
            cpp.ffi.push_str(&format!(
                "bool {prefix}_has_{name}(const {owner}*);\n\
                 {of}* {prefix}_make_{name}({owner}*);\n\
                 void {prefix}_clear_{name}({owner}*);\n\
                 {of}* {prefix}_{name}({owner}*);\n"));
            owner_methods.push_str(&format!(
                "    bool has_{name}() const {{ return ffi::{prefix}_has_{name}(h_); }}\n\
                 \x20   {of} make_{name}() {{ return {of}(ffi::{prefix}_make_{name}(h_)); }}\n\
                 \x20   void clear_{name}() {{ ffi::{prefix}_clear_{name}(h_); }}\n\
                 \x20   option<{of}> {name}() {{\n\
                 \x20       ffi::{of}* p = ffi::{prefix}_{name}(h_);\n\
                 \x20       return p ? option<{of}>({of}(p)) : option<{of}>();\n\
                 \x20   }}\n"));
        }
        "ref" => {
            cpp.ffi.push_str(&format!(
                "uint32_t {prefix}_{name}(const {owner}*);\n\
                 void {prefix}_set_{name}({owner}*, uint32_t);\n"));
            owner_methods.push_str(&format!(
                "    {of}Ref {name}() const {{ return {of}Ref{{ffi::{prefix}_{name}(h_)}}; }}\n\
                 \x20   void set_{name}({of}Ref r) {{ ffi::{prefix}_set_{name}(h_, r.raw); }}\n"));
        }
        "collection" => collection_cpp(cpp, owner, prefix, owner_methods, f)?,
        "self_block" | "cross_block" | "triplet_block" | "skip" => {}
        "opaque" => {
            owner_methods.push_str(&format!(
                "    // field `{name}`: {of} -- opaque, no accessor generated\n"));
        }
        other => return Err(format!("`{owner}.{name}`: unsupported kind `{other}`")),
    }
    Ok(())
}

/// Types this type's class depends on having been emitted first.
fn deps(t: &Type) -> Vec<&str> {
    t.fields.iter()
        .filter(|f| matches!(f.kind.as_str(),
            "struct" | "optional" | "collection" | "component"))
        .filter_map(|f| f.of.as_deref())
        .collect()
}

pub fn emit(model: &Model, ns: &str) -> Result<String, String> {
    let root = &model.root;
    let root_sn = snake(root);
    let fp = float_cpp(&model.precision);
    let mut cpp = Cpp { ffi: String::new(), body: String::new() };

    let surfaced = surfaced_types(model);
    let mut opaque_decls: String = format!("struct {root};\n");
    let mut ref_decls = String::new();
    for (tn, _) in &surfaced {
        opaque_decls.push_str(&format!("struct {tn};\n"));
        ref_decls.push_str(&format!(
            "/// Typed handle into the collection that issued it -- the C++\n/// spelling of Rust's `Ref<{tn}>`. Default-constructed it is the\n/// null sentinel (same as Rust `Ref::default()`).\nstruct {tn}Ref {{ uint32_t raw = UINT32_MAX; }};\n"));
    }

    // Children-first class order (containment is cycle-free).
    let mut remaining: Vec<(&String, &Type)> = surfaced.clone();
    let mut done: Vec<&str> = Vec::new();
    while !remaining.is_empty() {
        let Some(pos) = remaining.iter().position(|(_, t)| {
            deps(t).iter().all(|d| done.contains(d) || !model.types.contains_key(*d))
        }) else {
            return Err(format!("containment cycle among: {}",
                remaining.iter().map(|(tn, _)| tn.as_str()).collect::<Vec<_>>().join(", ")));
        };
        let (tn, t) = remaining.remove(pos);
        let mut methods = String::new();
        let prefix = format!("{root_sn}_{}", snake(tn));
        for f in &t.fields {
            field_cpp(&mut cpp, model, tn, &prefix, &mut methods, f)?;
        }
        cpp.body.push_str(&format!(
"/// A `{tn}` in its owner's storage; a thin pointer wrapper (validity
/// follows the storage -- see the owning container).
class {tn} {{
public:
    {tn}() : h_(nullptr) {{}}
    explicit {tn}(ffi::{tn}* p) : h_(p) {{}}
    /// False when default-constructed (e.g. inside an empty option).
    bool valid() const {{ return h_ != nullptr; }}
    /// The underlying C pointer -- the relaxed escape hatch.
    ffi::{tn}* raw() const {{ return h_; }}
{methods}private:
    ffi::{tn}* h_;
}};

"));
        done.push(tn.as_str());
    }

    // Covariance view: typed per-entity marginals (2x2 / 3x3 blocks as
    // matrix2d / matrix3d, 1x1 as double, larger via a raw buffer).
    {
        let mut methods = String::new();
        cpp.ffi.push_str(&format!(
            "int32_t {root_sn}_assemble_covariance({root}*, uint32_t);\n"));
        for (tn, t) in &surfaced {
            if t.role != "entity" || t.param_count == 0 {
                continue;
            }
            let sn = format!("{root_sn}_{}", snake(tn));
            cpp.ffi.push_str(&format!(
                "int32_t {sn}_marginal_cov({root}*, const {tn}*, double*, uint32_t);\n"));
            match t.param_count {
                1 => methods.push_str(&format!(
"    result<double, CovError> marginal(const {tn}& e) {{
        double b[1];
        if (ffi::{sn}_marginal_cov(h_, e.raw(), b, 1) < 0) return fail<double>();
        return result<double, CovError>::ok(b[0]);
    }}\n")),
                2 => methods.push_str(&format!(
"    result<matrix2d, CovError> marginal(const {tn}& e) {{
        double b[4];
        if (ffi::{sn}_marginal_cov(h_, e.raw(), b, 4) < 0) return fail<matrix2d>();
        return result<matrix2d, CovError>::ok(
            matrix2d::from_elements(b[0], b[1], b[2], b[3]));
    }}\n")),
                3 => methods.push_str(&format!(
"    result<matrix3d, CovError> marginal(const {tn}& e) {{
        double b[9];
        if (ffi::{sn}_marginal_cov(h_, e.raw(), b, 9) < 0) return fail<matrix3d>();
        return result<matrix3d, CovError>::ok(matrix3d::from_elements(
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8]));
    }}\n")),
                _ => methods.push_str(&format!(
"    /// Row-major dim x dim into out; returns dim or a negative code.
    int32_t marginal(const {tn}& e, double* out, uint32_t cap) {{
        return ffi::{sn}_marginal_cov(h_, e.raw(), out, cap);
    }}\n")),
            }
        }
        cpp.body.push_str(&format!(
"/// Covariance prepared at the solution (root.assemble_covariance);
/// queries answer per-entity marginal blocks. Valid until the model
/// is dropped or reassembled.
class Covariance {{
public:
    Covariance() : h_(nullptr) {{}}
    explicit Covariance(ffi::{root}* h) : h_(h) {{}}
{methods}private:
    template<class T> result<T, CovError> fail() {{
        return result<T, CovError>::err({{ffi::{root_sn}_last_error(h_)}});
    }}
    ffi::{root}* h_;
}};

"));
    }

    // Root class: views + field methods + solve surface.
    let root_ty = model.types.get(root).ok_or("root type missing")?;
    let mut world_methods = String::new();
    for f in &root_ty.fields {
        field_cpp(&mut cpp, model, root, &root_sn, &mut world_methods, f)?;
    }
    cpp.ffi.push_str(&format!(
        "void {root_sn}_lm_config(uint32_t, LmConfig*);\n\
         {root}* {root_sn}_new(void);\n\
         void {root_sn}_free({root}*);\n\
         const char* {root_sn}_last_error(const {root}*);\n\
         const char* {root_sn}_validate({root}*);\n\
         int32_t {root_sn}_solve_dense({root}*, const LmConfig*, LmResult*);\n\
         int32_t {root_sn}_solve_sparse({root}*, const LmConfig*, LmResult*);\n"));

    let ffi_decls = &cpp.ffi;
    let body = &cpp.body;
    Ok(format!(
"// GENERATED by cargo-arael from the `{root}` model sidecar. Do not
// edit; regenerate with `cargo arael export`.
#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <iterator>
#include \"arael/math.hpp\"
#include \"arael/result.hpp\"
#include \"arael/solver.hpp\"

namespace {ns} {{

/// The arael value vocabulary this interface uses, re-exported so a
/// single `using namespace {ns};` brings the model AND the math.
using arael::vect2;
using arael::vect3;
using arael::matrix2;
using arael::matrix3;
using arael::quatern;
using arael::vect2f;
using arael::vect2d;
using arael::vect3f;
using arael::vect3d;
using arael::matrix2f;
using arael::matrix2d;
using arael::matrix3f;
using arael::matrix3d;
using arael::quaternf;
using arael::quaternd;
using arael::option;
using arael::result;
using arael::LmStatus;
using arael::LmPreset;
using arael::LmConfigT;
using arael::LmResultT;
using arael::SolveResultT;
using arael::SolveError;
using arael::CovMode;
using arael::CovError;

/// Instantiations of the shared solver surface (arael/solver.hpp) at
/// this model's precision, plus the config constructor that fetches
/// the preset's actual Rust values through this root's FFI.
using LmResult = LmResultT<{fp}>;
using SolveResult = SolveResultT<{fp}>;

struct LmConfig : LmConfigT<{fp}> {{
    LmConfig(LmPreset p = LmPreset::Defaults);
    static LmConfig defaults() {{ return LmConfig(LmPreset::Defaults); }}
    static LmConfig conservative() {{ return LmConfig(LmPreset::Conservative); }}
    static LmConfig well_conditioned() {{ return LmConfig(LmPreset::WellConditioned); }}
}};

{ref_decls}
namespace ffi {{
{opaque_decls}
extern \"C\" {{
{ffi_decls}}}
}} // namespace ffi

inline LmConfig::LmConfig(LmPreset p) {{
    ffi::{root_sn}_lm_config(uint32_t(p), this);
}}

{body}/// The `{root}` model. Owns the Rust-side object; move-only.
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

{world_methods}
    /// Ok(LmResult) for every healthy termination, Err(SolveError) for
    /// a solve failure (-1) or a caught panic (-2) -- the same split
    /// Rust's SolveResult makes.
    SolveResult solve_dense(const LmConfig& cfg = LmConfig{{}}) {{
        LmResult r;
        int32_t code = ffi::{root_sn}_solve_dense(h_, &cfg, &r);
        if (code >= 0) return SolveResult::ok(r);
        return SolveResult::err({{static_cast<LmStatus>(code), last_error()}});
    }}
    SolveResult solve_sparse(const LmConfig& cfg = LmConfig{{}}) {{
        LmResult r;
        int32_t code = ffi::{root_sn}_solve_sparse(h_, &cfg, &r);
        if (code >= 0) return SolveResult::ok(r);
        return SolveResult::err({{static_cast<LmStatus>(code), last_error()}});
    }}
    /// Prepare the covariance at the current (solved) parameters; query
    /// per-entity marginals on the returned view.
    result<Covariance, CovError> assemble_covariance(CovMode mode = CovMode::AllMarginals) {{
        if (ffi::{root_sn}_assemble_covariance(h_, uint32_t(mode)) != 0)
            return result<Covariance, CovError>::err({{last_error()}});
        return result<Covariance, CovError>::ok(Covariance(h_));
    }}
    /// Empty string when the model is clean, the Diagnostic text
    /// otherwise. The returned pointer is valid until the next call on
    /// this model.
    const char* validate() {{ return ffi::{root_sn}_validate(h_); }}
    const char* last_error() const {{ return ffi::{root_sn}_last_error(h_); }}

private:
    ffi::{root}* h_;
}};

}} // namespace {ns}
"))
}
