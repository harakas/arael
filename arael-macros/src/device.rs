//! The device head of `#[arael(root)]`: a constraint's body as two C++
//! kernels (the assembly's and the cost's) over its entities' parameter
//! values, rotation caches and data leaves, and the walk that packs a
//! model's instances for them, as an implementation of
//! `arael::__gpu::DeviceRoot`. All of it is wrapped in
//! `arael::__gpu_items!`, which drops it when arael is built without its
//! `gpu` feature.
//!
//! Covered for now: root-level cross constraints (two entities with self
//! blocks reached through refs, the holder's CrossBlock), root-level
//! constraints of one entity reached through a ref (written into its self
//! block), and the constraints of an entity of a root-level collection on
//! itself (with a guard over its data fields, applied when packing),
//! without a loss or an index `select`, over Euler-angle and
//! `QuaternionParam` rotations, atom caches (an `AngleParam`'s sin and cos,
//! computed in the kernel from the angle) and `f32` or `f64` roots. A
//! leaf that follows the parameters and that the device does not compute
//! (a `UnitVecParam`'s unit vector) leaves the root without a device
//! route, as does any other constraint.

use std::collections::BTreeSet;

use arael_sym::cse::Intermediate;
use arael_sym::E;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};

/// One entity of a device constraint.
pub(crate) struct DevSide {
    pub ty: String,
    /// The entity's binding in the walk (`cam`).
    pub var: String,
    /// Its SelfBlock field.
    pub hb: String,
    /// Its parameter symbols in slot order (`cam.t.work().x`, ...).
    pub params: Vec<String>,
}

/// One device constraint, as the root's emission needs it.
pub(crate) struct DevConstraint {
    /// The group whose walk packs it.
    pub group_key: String,
    pub a: DevSide,
    /// The second entity of a constraint of two.
    pub b: Option<DevSide>,
    /// Of a constraint of two, the CrossBlock's holder type and field.
    pub holder: String,
    pub block: syn::Ident,
    /// The data leaves, as the Rust expressions the walk reads.
    pub data: Vec<String>,
    /// The rotation fields read.
    pub rots: Vec<DevRot>,
    /// The two kernels in C++, named by `cid`: they read every column
    /// through the layout table by the constraint's own names (`C<cid>_SA`,
    /// `C<cid>_EVA`, `C<cid>_ROT<r>`, ...), which the root aliases to its
    /// numbering ([`device_root`]).
    pub cpp: String,
    pub cid: usize,
    /// Residual rows.
    pub rows: usize,
    /// The constraint's guard over data alone ([`guard_on_data`]), applied
    /// when packing: an instance it drops is not packed.
    pub filter: Option<TokenStream2>,
}

/// A rotation field a device constraint reads.
#[derive(Clone)]
pub(crate) struct DevRot {
    /// Its constraint side.
    pub side: usize,
    /// Its field path in the entity.
    pub path: String,
    /// The offset of its step among the entity's parameters.
    pub step: usize,
    /// A `QuaternionParam` (else an Euler-angle parameter).
    pub quaternion: bool,
}

/// A CSE temporary's name: `__x3`, `__w1`, `__m2`.
fn is_temp(s: &str) -> bool {
    let rest = s.strip_prefix("__x").or_else(|| s.strip_prefix("__w")).or_else(|| s.strip_prefix("__m"));
    rest.is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
}

fn syms_of(e: &E, out: &mut BTreeSet<String>) {
    for s in e.symbols() {
        if !is_temp(&s) {
            out.insert(s);
        }
    }
}

fn syms_of_inters(inters: &[Intermediate], out: &mut BTreeSet<String>) {
    for it in inters {
        match it {
            Intermediate::Let { expr, .. } => syms_of(expr, out),
            Intermediate::Match { switch, arms, default, .. } => {
                match switch {
                    arael_sym::cse::Switch::Index(e) | arael_sym::cse::Switch::Sign(e) => syms_of(e, out),
                }
                for arm in arms.iter().chain(default.iter()) {
                    syms_of_inters(&arm.inters, out);
                    for v in &arm.values {
                        syms_of(v, out);
                    }
                }
            }
            Intermediate::Wide { inters, values, .. } => {
                syms_of_inters(inters, out);
                for v in values {
                    syms_of(v, out);
                }
            }
        }
    }
}

/// A rotation cache read: `<var>.<path>.rotation_matrix[r].c` (`kind` 0),
/// `..._wide[r].c` (1) or `..._deriv[k][r].c` (2), as (var, path, kind,
/// offset within the entity's 9 or 27 values).
fn rotation_leaf(s: &str) -> Option<(String, String, u8, usize)> {
    let at = s.find(".rotation_matrix")?;
    let (head, tail) = (&s[..at], &s[at + ".rotation_matrix".len()..]);
    let (var, path) = head.split_once('.')?;
    let (kind, rest) = if let Some(r) = tail.strip_prefix("_wide") {
        (1u8, r)
    } else if let Some(r) = tail.strip_prefix("_deriv") {
        (2u8, r)
    } else {
        (0u8, tail)
    };
    fn idx(r: &str) -> Option<(usize, &str)> {
        let r = r.strip_prefix('[')?;
        let close = r.find(']')?;
        Some((r[..close].trim().parse().ok()?, &r[close + 1..]))
    }
    let (first, rest) = idx(rest)?;
    let (k, row, rest) = if kind == 2 {
        let (row, rest) = idx(rest)?;
        (first, row, rest)
    } else {
        (0, first, rest)
    };
    let col = match rest {
        ".x" => 0,
        ".y" => 1,
        ".z" => 2,
        _ => return None,
    };
    if row > 2 || k > 2 {
        return None;
    }
    Some((var.to_string(), path.to_string(), kind, k * 9 + row * 3 + col))
}

/// True if `field` of the registered type `ty` is a parameter or a
/// component of parameters.
fn parameter_field(ty: &str, field: &str) -> bool {
    crate::registry_lookup(ty).is_some_and(|l| {
        l.param_fields.iter().any(|f| f == field)
            || l.fields.iter().any(|(f, t)| {
                f == field && matches!(t, crate::SymFieldType::Struct(inner) if crate::registry_lookup(inner).is_some_and(|c| c.component))
            })
    })
}

/// Why the device cannot pack `s` as data: it reads a parameter field of
/// one of the entities, or a component of parameters, other than through
/// the leaves the device computes (the parameters, the rotations, the atom
/// caches), so it would hold the value the solve started from.
fn parameter_state(s: &str, a: &DevSide, b: Option<&DevSide>) -> Option<String> {
    let (var, rest) = s.split_once('.')?;
    let side = [Some(a), b].into_iter().flatten().find(|sd| sd.var == var)?;
    let field = rest.split(['.', '[']).next()?;
    parameter_field(&side.ty, field)
        .then(|| format!("the device reads `{s}`, which follows the parameters of `{var}.{field}`, without computing it"))
}

/// Why the device cannot apply `guard` when it packs the instances: the
/// guard may read the entity bound as `var` (of type `ty`) only through
/// its data fields, so that it holds through the solve, and nothing else
/// by name.
pub(crate) fn guard_on_data(guard: &syn::Expr, var: &str, ty: &str) -> Result<(), String> {
    use proc_macro2::TokenTree;
    fn flat(ts: TokenStream2, out: &mut Vec<TokenTree>) {
        for t in ts {
            match t {
                TokenTree::Group(g) => flat(g.stream(), out),
                t => out.push(t),
            }
        }
    }
    let mut toks = Vec::new();
    flat(quote! { #guard }, &mut toks);
    let layout = crate::registry_lookup(ty).ok_or_else(|| format!("`{ty}` has no layout"))?;
    for (k, t) in toks.iter().enumerate() {
        let TokenTree::Ident(id) = t else { continue };
        let after_dot = k > 0 && matches!(&toks[k - 1], TokenTree::Punct(p) if p.as_char() == '.');
        let name = id.to_string();
        if after_dot || name == "true" || name == "false" {
            continue;
        }
        if name != var {
            return Err(format!("a guard reading `{name}`"));
        }
        let field = match (toks.get(k + 1), toks.get(k + 2)) {
            (Some(TokenTree::Punct(p)), Some(TokenTree::Ident(f))) if p.as_char() == '.' => f.to_string(),
            _ => return Err(format!("a guard reading `{var}` whole")),
        };
        if parameter_field(ty, &field) || !layout.fields.iter().any(|(f, _)| *f == field) {
            return Err(format!("a guard reading `{var}.{field}`, not a data field"));
        }
    }
    Ok(())
}

/// What a leaf symbol reads in the kernel.
enum Leaf {
    Param(usize, usize),
    Rot(usize, u8, usize),
    Data(usize),
    /// An atom cache (`a.rot.sin`), computed from its atom over the
    /// parameters (`sin(a.rot.angle.work())`).
    Atom(E),
}

/// The kernels of a root-level constraint of one entity or of two (`b`):
/// `gh` and `guard` from the assembly's CSE (rows then partials, A's then
/// B's), `cost` from the cost's. `ft` is the root's scalar. `atoms` are the
/// body's atom caches and their atoms (`a.rot.sin`, `sin(a.rot.angle.work())`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_constraint(
    cid: usize,
    group_key: String,
    ft: &str,
    a: DevSide,
    b: Option<DevSide>,
    holder: String,
    block: syn::Ident,
    rows: usize,
    gh: (&[Intermediate], &[E]),
    guard: Option<(&[Intermediate], &E)>,
    cost: (&[Intermediate], &[E]),
    atoms: &[(String, E)],
) -> Result<DevConstraint, String> {
    if ft != "f32" && ft != "f64" {
        return Err(format!("a root of `{ft}` has no device assembly"));
    }
    let (na, nb) = (a.params.len(), b.as_ref().map_or(0, |b| b.params.len()));
    // the leaves each kernel reads
    let mut gh_syms = BTreeSet::new();
    syms_of_inters(gh.0, &mut gh_syms);
    gh.1.iter().for_each(|e| syms_of(e, &mut gh_syms));
    if let Some((pre, q)) = guard {
        syms_of_inters(pre, &mut gh_syms);
        syms_of(q, &mut gh_syms);
    }
    let mut cost_syms = BTreeSet::new();
    syms_of_inters(cost.0, &mut cost_syms);
    cost.1.iter().for_each(|e| syms_of(e, &mut cost_syms));
    // a kernel computing an atom reads its parameters
    let atom = |s: &str| atoms.iter().find(|(c, _)| c == s).map(|(_, e)| e);
    for syms in [&mut gh_syms, &mut cost_syms] {
        let used: Vec<&E> = syms.iter().filter_map(|s| atom(s)).collect();
        let mut more = BTreeSet::new();
        used.into_iter().for_each(|e| syms_of(e, &mut more));
        syms.append(&mut more);
    }
    let all: BTreeSet<&String> = gh_syms.iter().chain(&cost_syms).collect();
    // classified, the data leaves and the rotations numbered in order
    let mut data: Vec<String> = Vec::new();
    let mut rots: Vec<DevRot> = Vec::new();
    let mut leaf: std::collections::BTreeMap<String, Leaf> = Default::default();
    for s in all {
        let l = if let Some(e) = atom(s) {
            if let Some(d) = e.symbols().into_iter().find(|d| !a.params.contains(d) && !b.as_ref().is_some_and(|b| b.params.contains(d))) {
                return Err(format!("the atom cache `{s}` reads `{d}`, not a parameter of the constraint's entities"));
            }
            Leaf::Atom(e.clone())
        } else if let Some(p) = a.params.iter().position(|p| p == s) {
            Leaf::Param(0, p)
        } else if let Some(p) = b.as_ref().and_then(|b| b.params.iter().position(|p| p == s)) {
            Leaf::Param(1, p)
        } else if let Some((var, path, kind, off)) = rotation_leaf(s) {
            let side = if var == a.var {
                0
            } else if b.as_ref().is_some_and(|b| var == b.var) {
                1
            } else {
                return Err(format!("the rotation `{s}` is not of the constraint's entities"));
            };
            let r = match rots.iter().position(|r| r.side == side && r.path == path) {
                Some(r) => r,
                None => {
                    // the device composes an Euler-angle or a rotation-vector
                    // step with its reference
                    let sd = if side == 0 { &a } else { b.as_ref().expect("side 1 is b's") };
                    let layout = crate::registry_lookup(&sd.ty);
                    let euler = layout.as_ref().is_some_and(|l| {
                        l.universal_euler_angle_fields.contains(&path) || l.mixed_euler_angle_fields.iter().any(|(f, _)| *f == path)
                    });
                    let quaternion = layout.as_ref().is_some_and(|l| l.universal_rotvec_fields.contains(&path));
                    if !euler && !quaternion {
                        return Err(format!("the rotation `{}.{}` is not an EulerAngleParam or a QuaternionParam", sd.var, path));
                    }
                    // the rotation's step: its entity's `<var>.<path>.delta.x` slot
                    let step = format!("{}.{}.delta.x", sd.var, path);
                    let step = sd.params.iter().position(|p| *p == step)
                        .ok_or_else(|| format!("the rotation `{}.{}` has no step among its entity's parameters", sd.var, path))?;
                    rots.push(DevRot { side, path, step, quaternion });
                    rots.len() - 1
                }
            };
            Leaf::Rot(r, kind, off)
        } else {
            if let Some(why) = parameter_state(s, &a, b.as_ref()) {
                return Err(why);
            }
            data.push(s.clone());
            Leaf::Data(data.len() - 1)
        };
        leaf.insert(s.clone(), l);
    }
    let names: std::collections::BTreeMap<String, String> = leaf.keys().enumerate().map(|(i, s)| (s.clone(), format!("__l{i}"))).collect();
    let cpp = cpp_kernels(cid, ft, (na, nb), rows, &leaf, &names, &rots, (&gh_syms, &cost_syms), gh, guard, cost)?;
    Ok(DevConstraint { group_key, a, b, holder, block, data, rots, cpp, cid, rows, filter: None })
}

/// The kernels of [`device_constraint`] in C++
/// (`arael-gpu/kernels/assembly.cpp` is compiled in front of them): every
/// column is read through the layout table `L` at the start of the model's
/// block `B`, by the constraint's own column names.
#[allow(clippy::too_many_arguments)]
fn cpp_kernels(
    cid: usize,
    ft: &str,
    (na, nb): (usize, usize),
    rows: usize,
    leaf: &std::collections::BTreeMap<String, Leaf>,
    names: &std::collections::BTreeMap<String, String>,
    rots: &[DevRot],
    (gh_syms, cost_syms): (&BTreeSet<String>, &BTreeSet<String>),
    gh: (&[Intermediate], &[E]),
    guard: Option<(&[Intermediate], &E)>,
    cost: (&[Intermediate], &[E]),
) -> Result<String, String> {
    if na == 0 {
        return Err("a device constraint's entities have parameters".into());
    }
    let pair = nb > 0;
    let map = |s: &str| -> Option<String> { names.get(s).cloned() };
    // a leaf's type and the read that gives it
    let read = |l: &Leaf| -> Result<(&'static str, String), String> {
        Ok(match *l {
            Leaf::Param(0, p) => ("F", format!("COL(B, F, C{cid}_EVA)[sa * {na}u + {p}u]")),
            Leaf::Param(_, p) => ("F", format!("COL(B, F, C{cid}_EVB)[sb * {nb}u + {p}u]")),
            Leaf::Rot(r, kind, off) => {
                let s = if rots[r].side == 0 { "sa" } else { "sb" };
                match kind {
                    0 => ("F", format!("COL(B, F, C{cid}_ROT{r})[{s} * 9u + {off}u]")),
                    1 => ("W", format!("COL(B, W, C{cid}_ROTW{r})[{s} * 9u + {off}u]")),
                    _ => ("F", format!("COL(B, F, C{cid}_DROT{r})[{s} * 27u + {off}u]")),
                }
            }
            Leaf::Data(j) => ("F", format!("COL(B, F, C{cid}_D{j})[o]")),
            Leaf::Atom(ref e) => ("F", e.to_cpp_in(ft, "W", &map).map_err(|e| e.0)?),
        })
    };
    // the atoms after the parameters they read
    let lets = |syms: &BTreeSet<String>| -> Result<String, String> {
        let (atoms, plain): (Vec<&String>, Vec<&String>) = syms.iter().partition(|s| matches!(leaf[*s], Leaf::Atom(_)));
        plain.into_iter().chain(atoms)
            .map(|s| {
                let (t, e) = read(&leaf[s])?;
                Ok(format!("const {t} {} = {e};\n", names[s]))
            })
            .collect()
    };
    // wide blocks in W, the kernel's wide scalar (wide.h)
    let stmts = |inters: &[Intermediate]| -> Result<String, String> {
        inters.iter().map(|i| i.to_cpp_in(ft, "W", &map).map_err(|e| e.0)).collect()
    };
    let expr = |e: &E| -> Result<String, String> { e.to_cpp_in(ft, "W", &map).map_err(|e| e.0) };
    let prologue = |b: &str| {
        let sb = if pair { format!("const u32 sb = COL({b}, u32, C{cid}_SB)[o];\n") } else { String::new() };
        format!(
            "GLOBAL const u64* L = (GLOBAL const u64*)B;\nconst u32 o = global_x();\nW c = 0.0f;\nif (o < n) {{\n\
             const u32 sa = COL({b}, u32, C{cid}_SA)[o];\n{sb}"
        )
    };
    // the linearization: per instance its record, R rows of the residual
    // and the two entities' derivatives, zero where the guard drops it
    let n_partials = na + nb;
    let stride = 1 + n_partials;
    let recw = rows * stride;
    let mut body = stmts(gh.0)?;
    for ri in 0..rows {
        let r = expr(&gh.1[ri * stride])?;
        body.push_str(&format!("rec[{}] = {r};\n", ri * stride));
        for p in 0..n_partials {
            let d = expr(&gh.1[ri * stride + 1 + p])?;
            body.push_str(&format!("rec[{}] = {d};\n", ri * stride + 1 + p));
        }
    }
    let gh_code = match guard {
        Some((pre, q)) => format!("{}if ({} >= 0.0f) {{\n{body}}}\n", stmts(pre)?, expr(q)?),
        None => body,
    };
    let costs: String = (0..rows).map(|ri| format!("c += (W)rec[{0}] * (W)rec[{0}];\n", ri * stride)).collect();
    let gh_kernel = format!(
        "KERNEL_THREADS(256) void __dk{cid}_gh(GLOBAL u8* RESTRICT B, u32 n) {{\n{}{}\
         F rec[{recw}];\n#pragma unroll\nfor (u32 k = 0; k < {recw}u; k++) rec[k] = (F)0.0f;\n{gh_code}{costs}\
         GLOBAL F* out = COL_MUT(B, F, C{cid}_REC) + o * {recw}u;\n#pragma unroll\nfor (u32 k = 0; k < {recw}u; k++) out[k] = rec[k];\n\
         }}\ncost_partial(COL_MUT(B, W, C{cid}_PART), c);\n}}\n",
        prologue("B"),
        lets(gh_syms)?
    );
    // the sums of the records: each entity side, the cross tiles, the cost
    let mut reduce = format!(
        "KERNEL_THREADS(256) void __dk{cid}_ra(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT h, GLOBAL W* RESTRICT g, u32 slots) {{\n\
         GLOBAL const u64* L = (GLOBAL const u64*)B;\n\
         reduce_entity<{na}u, {rows}u, {stride}u, 1u>(COL(B, F, C{cid}_REC), COL(B, u32, C{cid}_AOFF), COL(B, u32, C{cid}_AINST), COL(B, u32, C{cid}_IDXA), COL(B, u32, C{cid}_TILEA), h, g, slots);\n}}\n\
         KERNEL_THREADS(256) void __dk{cid}_csum(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT cost, u32 parts) {{\n\
         GLOBAL const u64* L = (GLOBAL const u64*)B;\ncost_sum(COL(B, W, C{cid}_PART), parts, cost);\n}}\n"
    );
    if pair {
        reduce.push_str(&format!(
            "KERNEL_THREADS(256) void __dk{cid}_rb(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT h, GLOBAL W* RESTRICT g, u32 slots) {{\n\
             GLOBAL const u64* L = (GLOBAL const u64*)B;\n\
             reduce_entity<{nb}u, {rows}u, {stride}u, {}u>(COL(B, F, C{cid}_REC), COL(B, u32, C{cid}_BOFF), COL(B, u32, C{cid}_BINST), COL(B, u32, C{cid}_IDXB), COL(B, u32, C{cid}_TILEB), h, g, slots);\n}}\n\
             KERNEL_THREADS(256) void __dk{cid}_rx(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT h, u32 slots) {{\n\
             GLOBAL const u64* L = (GLOBAL const u64*)B;\n\
             reduce_cross<{na}u, {nb}u, {rows}u, {stride}u>(COL(B, F, C{cid}_REC), COL(B, u32, C{cid}_XOFF), COL(B, u32, C{cid}_XINST), COL(B, u32, C{cid}_SA), COL(B, u32, C{cid}_SB), \
             COL(B, u32, C{cid}_IDXA), COL(B, u32, C{cid}_TILEA), COL(B, u32, C{cid}_IDXB), COL(B, u32, C{cid}_TILEB), COL(B, u32, C{cid}_TILEX), h, slots);\n}}\n",
            1 + na
        ));
    }
    // the cost: the rows alone
    let mut cbody = stmts(cost.0)?;
    for (ri, r) in cost.1.iter().enumerate() {
        let r = expr(r)?;
        cbody.push_str(&format!("const F r{ri} = {r};\nc += (W)r{ri} * (W)r{ri};\n"));
    }
    let cost_kernel = format!(
        "KERNEL_THREADS(256) void __dk{cid}_cost(GLOBAL u8* RESTRICT B, u32 n) {{\n{}{}{cbody}}}\ncost_partial(COL_MUT(B, W, C{cid}_PART), c);\n}}\n",
        prologue("B"),
        lets(cost_syms)?
    );
    Ok(format!("{gh_kernel}{reduce}{cost_kernel}"))
}

/// The walk of one cross group, as the root's emission hands it over.
pub(crate) struct DevGroup {
    pub rc_ident: syn::Ident,
    pub resolve_stmts: Vec<TokenStream2>,
    pub root_var_ident: syn::Ident,
}

/// The device items of a root whose every constraint has a device form.
pub(crate) fn device_root(
    root_name: &syn::Ident,
    ft: &str,
    store_ty: &syn::Ident,
    constraints: &[DevConstraint],
    groups: &std::collections::BTreeMap<String, DevGroup>,
    array_ident: &dyn Fn(&str, &str) -> syn::Ident,
) -> TokenStream2 {
    let scalar: syn::Type = syn::parse_str(ft).unwrap();
    // the device's numbering of the entity arrays, rotation fields and
    // cross arrays the constraints read
    let mut entities: Vec<(String, String, usize)> = Vec::new();
    let mut rotations: Vec<(usize, DevRot)> = Vec::new();
    let mut crosses: Vec<(String, String)> = Vec::new();
    let entity_of = |entities: &mut Vec<(String, String, usize)>, s: &DevSide| -> usize {
        match entities.iter().position(|(t, hb, _)| *t == s.ty && *hb == s.hb) {
            Some(e) => e,
            None => {
                entities.push((s.ty.clone(), s.hb.clone(), s.params.len()));
                entities.len() - 1
            }
        }
    };
    struct Ids {
        ea: usize,
        /// Of a constraint of two: the second side's entity array and the cross array.
        pair: Option<(usize, usize)>,
        rots: Vec<usize>,
    }
    let mut ids: Vec<Ids> = Vec::new();
    for c in constraints {
        let ea = entity_of(&mut entities, &c.a);
        let pair = c.b.as_ref().map(|b| {
            let eb = entity_of(&mut entities, b);
            let x = match crosses.iter().position(|(h, bl)| *h == c.holder && c.block == bl) {
                Some(x) => x,
                None => {
                    crosses.push((c.holder.clone(), c.block.to_string()));
                    crosses.len() - 1
                }
            };
            (eb, x)
        });
        let rots = c
            .rots
            .iter()
            .map(|rot| {
                let e = if rot.side == 0 { ea } else { pair.expect("a rotation of side 1 is b's").0 };
                match rotations.iter().position(|(re, r)| *re == e && r.path == rot.path) {
                    Some(r) => r,
                    None => {
                        rotations.push((e, rot.clone()));
                        rotations.len() - 1
                    }
                }
            })
            .collect();
        ids.push(Ids { ea, pair, rots });
    }
    let rot_shape: Vec<TokenStream2> = rotations
        .iter()
        .map(|(e, r)| {
            let step = r.step;
            let kind = if r.quaternion { quote! { Quaternion } } else { quote! { Euler } };
            quote! { ::arael::__gpu::DeviceRotation { entity: #e, step: #step, kind: ::arael::__gpu::RotationKind::#kind } }
        })
        .collect();
    let entity_n: Vec<usize> = entities.iter().map(|e| e.2).collect();
    let n_cross = crosses.len();
    let cshape: Vec<TokenStream2> = constraints.iter().zip(&ids).map(|(c, id)| {
        let (nd, rows, ea) = (c.data.len(), c.rows, id.ea);
        match id.pair {
            Some((eb, x)) => quote! { ::arael::__gpu::DeviceConstraint { words: 3usize, data: #nd, rows: #rows, a: #ea, pair: Some((#eb, #x)) } },
            None => quote! { ::arael::__gpu::DeviceConstraint { words: 1usize, data: #nd, rows: #rows, a: #ea, pair: None } },
        }
    }).collect();
    // the C++ kernels, each constraint's column names aliased to the
    // root's numbering, which the host's layout table follows
    let mut cpp = String::new();
    for (c, id) in constraints.iter().zip(&ids) {
        let k = c.cid;
        let mut alias = |local: &str, root: String| cpp.push_str(&format!("#define ARAEL_COL_C{k}_{local} ARAEL_COL_{root}\n"));
        alias("EVA", format!("E{}_EV", id.ea));
        alias("IDXA", format!("E{}_IDX", id.ea));
        alias("TILEA", format!("E{}_TILE", id.ea));
        if let Some((eb, x)) = id.pair {
            alias("EVB", format!("E{eb}_EV"));
            alias("IDXB", format!("E{eb}_IDX"));
            alias("TILEB", format!("E{eb}_TILE"));
            alias("TILEX", format!("X{x}_TILE"));
        }
        for (lr, &r) in id.rots.iter().enumerate() {
            alias(&format!("ROT{lr}"), format!("R{r}_ROT"));
            alias(&format!("ROTW{lr}"), format!("R{r}_ROTW"));
            alias(&format!("DROT{lr}"), format!("R{r}_DROT"));
        }
        cpp.push_str(&c.cpp);
    }
    // tables: one store built for the model as it is, every slot in it
    let table_entity: Vec<TokenStream2> = entities.iter().map(|(t, hb, _)| {
        let arr = array_ident(t, hb);
        quote! { { let (__i, __t) = __store.#arr.device_table(resolve); __tables.idx.push(__i); __tables.tile.push(__t); } }
    }).collect();
    let table_cross: Vec<TokenStream2> = crosses.iter().map(|(h, b)| {
        let arr = array_ident(h, b);
        quote! { __tables.cross.push(__store.#arr.device_table(resolve)); }
    }).collect();
    // the pack walk: per group the cost sweep's loop, per constraint its
    // slots, data, the entities' fixed values and reference rotations
    let mut walks: Vec<TokenStream2> = Vec::new();
    for (key, g) in groups {
        let entries: Vec<TokenStream2> = constraints.iter().zip(&ids).filter(|(c, _)| c.group_key == *key).map(|(c, id)| {
            let cid = c.cid;
            let (av, ahb) = (format_ident!("{}", c.a.var), format_ident!("{}", c.a.hb));
            let data: Vec<syn::Expr> = c.data.iter().map(|d| syn::parse_str(d).expect("a data leaf is an expression")).collect();
            let fixed = |side: &DevSide, e: usize, s: TokenStream2| -> TokenStream2 {
                let n = side.params.len();
                let vals: Vec<syn::Expr> = side.params.iter().map(|p| syn::parse_str(p).expect("a parameter symbol is an expression")).collect();
                let ks: Vec<usize> = (0..n).collect();
                quote! { { let __f = &mut __pack.fixed[#e]; #( __f[#s * #n + #ks] = ::arael::__gpu::leaf_f64(#vals) as #scalar; )* } }
            };
            let fa = fixed(&c.a, id.ea, quote! { __sa });
            // the instance's slots, and the second side's fixed values
            let (slots, fb) = match (&c.b, id.pair) {
                (Some(b), Some((eb, _))) => {
                    let (bv, bhb, block) = (format_ident!("{}", b.var), format_ident!("{}", b.hb), &c.block);
                    let fb = fixed(b, eb, quote! { __sb });
                    (quote! {
                        let __sb = #bv.#bhb.slot() as usize;
                        __pack.inst[#cid].extend_from_slice(&[__sa as u32, __sb as u32, __frine.#block.slot()]);
                    }, fb)
                }
                _ => (quote! { __pack.inst[#cid].push(__sa as u32); }, quote! {}),
            };
            let rots: Vec<TokenStream2> = c.rots.iter().zip(&id.rots).map(|(rot, &r)| {
                let (var, s) = if rot.side == 0 {
                    (format_ident!("{}", c.a.var), quote! { __sa })
                } else {
                    (format_ident!("{}", c.b.as_ref().expect("a rotation of side 1 is b's").var), quote! { __sb })
                };
                let field: syn::Expr = syn::parse_str(&format!("{}.{}", var, rot.path)).expect("a rotation field path");
                if rot.quaternion {
                    return quote! {
                        __pack.refr[#r][#s * 4..#s * 4 + 4].copy_from_slice(&#field.__reference_quaternion());
                        __pack.packed[#r][#s] = true;
                    };
                }
                quote! {
                    {
                        __pack.packed[#r][#s] = true;
                        let __m = &#field.ref_rotation;
                        let __r = &mut __pack.refr[#r];
                        for __i in 0..3usize {
                            __r[#s * 9 + __i * 3] = ::arael::__gpu::leaf_f64(__m[__i].x);
                            __r[#s * 9 + __i * 3 + 1] = ::arael::__gpu::leaf_f64(__m[__i].y);
                            __r[#s * 9 + __i * 3 + 2] = ::arael::__gpu::leaf_f64(__m[__i].z);
                        }
                    }
                }
            }).collect();
            let cond = c.filter.clone().unwrap_or_else(|| quote! { true });
            quote! {
                if #cond {
                    let __sa = #av.#ahb.slot() as usize;
                    #slots
                    __pack.data[#cid].extend_from_slice(&[#( (#data) as #scalar ),*]);
                    #fa
                    #fb
                    #(#rots)*
                }
            }
        }).collect();
        if entries.is_empty() {
            continue;
        }
        let (rc, resolve, root_var) = (&g.rc_ident, &g.resolve_stmts, &g.root_var_ident);
        walks.push(quote! {
            for __frine in self.#rc.iter() {
                #(#resolve)*
                #[allow(unused_variables)]
                let #root_var = &*self;
                #(#entries)*
            }
        });
    }
    let n_constraints = constraints.len();
    quote! {
        ::arael::__gpu_items! {
            impl ::arael::__gpu::DeviceRoot<#scalar> for #root_name {
                fn device_shape(&self) -> ::arael::__gpu::DeviceShape {
                    ::arael::__gpu::DeviceShape {
                        entities: vec![#(#entity_n),*],
                        rotations: vec![#(#rot_shape),*],
                        crosses: #n_cross,
                        constraints: vec![#(#cshape),*],
                    }
                }

                fn device_tables(&self, resolve: &mut dyn FnMut(u32, u32) -> (usize, usize, usize)) -> ::arael::__gpu::DeviceTables {
                    let mut __store = #store_ty::default();
                    self.__build_blocks(&mut __store);
                    let mut __tables = ::arael::__gpu::DeviceTables::default();
                    #(#table_entity)*
                    #(#table_cross)*
                    __tables
                }

                #[allow(unused_variables, clippy::all)]
                fn device_pack(&self, __pack: &mut ::arael::__gpu::DevicePack<#scalar>) {
                    use arael::utils::{Float as _, SelectIndex as _};
                    use arael::store::Leaves as _;
                    let _ = #n_constraints;
                    #(#walks)*
                }

                fn device_cpp(&self) -> &'static str {
                    #cpp
                }

                fn device_restore(&mut self, refs: &mut dyn FnMut(u32) -> Option<arael::model::Reference>) {
                    arael::model::Model::set_references(self, refs);
                }
            }
        }
    }
}
