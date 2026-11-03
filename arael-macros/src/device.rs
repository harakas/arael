//! The device head of `#[arael(root)]`: a constraint's body as two C++
//! kernels (the assembly's and the cost's) over its entities' parameter
//! values, rotation caches and data leaves, and the walk that packs a
//! model's instances for them, as an implementation of
//! `arael::__gpu::DeviceRoot`. All of it is wrapped in
//! `arael::__gpu_items!`, which drops it when arael is built without its
//! `gpu` feature.
//!
//! Covered for now: root-level cross constraints (two entities with self
//! blocks reached through refs, the holder's CrossBlock), without a guard,
//! a loss or an index `select`, over Euler-angle rotations and `f32` or
//! `f64` roots. A root with any other constraint gets no device route.

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
    /// The cross group whose walk packs it.
    pub group_key: String,
    pub a: DevSide,
    pub b: DevSide,
    /// The CrossBlock's holder type and field.
    pub holder: String,
    pub block: syn::Ident,
    /// The data leaves, as the Rust expressions the walk reads.
    pub data: Vec<String>,
    /// The rotation fields read: (side, field path, the offset of the
    /// rotation's step among that entity's parameters).
    pub rots: Vec<(usize, String, usize)>,
    /// The two kernels in C++, named by `cid`: they read every column
    /// through the layout table by the constraint's own names (`C<cid>_SA`,
    /// `C<cid>_EVA`, `C<cid>_ROT<r>`, ...), which the root aliases to its
    /// numbering ([`device_root`]).
    pub cpp: String,
    pub cid: usize,
    /// Residual rows.
    pub rows: usize,
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

/// What a leaf symbol reads in the kernel.
enum Leaf {
    Param(usize, usize),
    Rot(usize, u8, usize),
    Data(usize),
}

/// The kernels of a root-level cross constraint: `gh` and `guard` from the
/// assembly's CSE (rows then partials, A's then B's), `cost` from the
/// cost's. `ft` is the root's scalar.
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_constraint(
    cid: usize,
    group_key: String,
    ft: &str,
    a: DevSide,
    b: DevSide,
    holder: String,
    block: syn::Ident,
    rows: usize,
    gh: (&[Intermediate], &[E]),
    guard: Option<(&[Intermediate], &E)>,
    cost: (&[Intermediate], &[E]),
) -> Result<DevConstraint, String> {
    if ft != "f32" && ft != "f64" {
        return Err(format!("a root of `{ft}` has no device assembly"));
    }
    let (na, nb) = (a.params.len(), b.params.len());
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
    let all: BTreeSet<&String> = gh_syms.iter().chain(&cost_syms).collect();
    // classified, the data leaves and the rotations numbered in order
    let mut data: Vec<String> = Vec::new();
    let mut rots: Vec<(usize, String, usize)> = Vec::new();
    let mut leaf: std::collections::BTreeMap<String, Leaf> = Default::default();
    for s in all {
        let l = if let Some(p) = a.params.iter().position(|p| p == s) {
            Leaf::Param(0, p)
        } else if let Some(p) = b.params.iter().position(|p| p == s) {
            Leaf::Param(1, p)
        } else if let Some((var, path, kind, off)) = rotation_leaf(s) {
            let side = if var == a.var {
                0
            } else if var == b.var {
                1
            } else {
                return Err(format!("the rotation `{s}` is not of the constraint's entities"));
            };
            let r = match rots.iter().position(|(sd, p, _)| *sd == side && *p == path) {
                Some(r) => r,
                None => {
                    // the device composes an Euler-angle step with its reference
                    let sd = if side == 0 { &a } else { &b };
                    let euler = crate::registry_lookup(&sd.ty).is_some_and(|l| {
                        l.universal_euler_angle_fields.contains(&path) || l.mixed_euler_angle_fields.iter().any(|(f, _)| *f == path)
                    });
                    if !euler {
                        return Err(format!("the rotation `{}.{}` is not an EulerAngleParam", sd.var, path));
                    }
                    // the rotation's step: its entity's `<var>.<path>.delta.x` slot
                    let step = format!("{}.{}.delta.x", sd.var, path);
                    let off = sd.params.iter().position(|p| *p == step)
                        .ok_or_else(|| format!("the rotation `{}.{}` has no step among its entity's parameters", sd.var, path))?;
                    rots.push((side, path, off));
                    rots.len() - 1
                }
            };
            Leaf::Rot(r, kind, off)
        } else {
            data.push(s.clone());
            Leaf::Data(data.len() - 1)
        };
        leaf.insert(s.clone(), l);
    }
    let names: std::collections::BTreeMap<String, String> = leaf.keys().enumerate().map(|(i, s)| (s.clone(), format!("__l{i}"))).collect();
    let cpp = cpp_kernels(cid, ft, (na, nb), rows, &leaf, &names, &rots, (&gh_syms, &cost_syms), gh, guard, cost)?;
    Ok(DevConstraint { group_key, a, b, holder, block, data, rots, cpp, cid, rows })
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
    rots: &[(usize, String, usize)],
    (gh_syms, cost_syms): (&BTreeSet<String>, &BTreeSet<String>),
    gh: (&[Intermediate], &[E]),
    guard: Option<(&[Intermediate], &E)>,
    cost: (&[Intermediate], &[E]),
) -> Result<String, String> {
    if na == 0 || nb == 0 {
        return Err("a device constraint's entities have parameters".into());
    }
    // a leaf's type and the read that gives it
    let read = |l: &Leaf| -> (&'static str, String) {
        match *l {
            Leaf::Param(0, p) => ("F", format!("COL(B, F, C{cid}_EVA)[sa * {na}u + {p}u]")),
            Leaf::Param(_, p) => ("F", format!("COL(B, F, C{cid}_EVB)[sb * {nb}u + {p}u]")),
            Leaf::Rot(r, kind, off) => {
                let s = if rots[r].0 == 0 { "sa" } else { "sb" };
                match kind {
                    0 => ("F", format!("COL(B, F, C{cid}_ROT{r})[{s} * 9u + {off}u]")),
                    1 => ("W", format!("COL(B, W, C{cid}_ROTW{r})[{s} * 9u + {off}u]")),
                    _ => ("F", format!("COL(B, F, C{cid}_DROT{r})[{s} * 27u + {off}u]")),
                }
            }
            Leaf::Data(j) => ("F", format!("COL(B, F, C{cid}_D{j})[o]")),
        }
    };
    let lets = |syms: &BTreeSet<String>| -> String {
        syms.iter()
            .map(|s| {
                let (t, e) = read(&leaf[s]);
                format!("const {t} {} = {e};\n", names[s])
            })
            .collect()
    };
    let map = |s: &str| -> Option<String> { names.get(s).cloned() };
    // wide blocks in W, the kernel's wide scalar (wide.h)
    let stmts = |inters: &[Intermediate]| -> Result<String, String> {
        inters.iter().map(|i| i.to_cpp_in(ft, "W", &map).map_err(|e| e.0)).collect()
    };
    let expr = |e: &E| -> Result<String, String> { e.to_cpp_in(ft, "W", &map).map_err(|e| e.0) };
    let prologue = |b: &str| {
        format!(
            "GLOBAL const u64* L = (GLOBAL const u64*)B;\nconst u32 o = global_x();\nW c = 0.0f;\nif (o < n) {{\n\
             const u32 sa = COL({b}, u32, C{cid}_SA)[o];\nconst u32 sb = COL({b}, u32, C{cid}_SB)[o];\n"
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
        lets(gh_syms)
    );
    // the sums of the records: each entity side, the cross tiles, the cost
    let reduce = format!(
        "KERNEL_THREADS(256) void __dk{cid}_ra(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT h, GLOBAL W* RESTRICT g, u32 slots) {{\n\
         GLOBAL const u64* L = (GLOBAL const u64*)B;\n\
         reduce_entity<{na}u, {rows}u, {stride}u, 1u>(COL(B, F, C{cid}_REC), COL(B, u32, C{cid}_AOFF), COL(B, u32, C{cid}_AINST), COL(B, u32, C{cid}_IDXA), COL(B, u32, C{cid}_TILEA), h, g, slots);\n}}\n\
         KERNEL_THREADS(256) void __dk{cid}_rb(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT h, GLOBAL W* RESTRICT g, u32 slots) {{\n\
         GLOBAL const u64* L = (GLOBAL const u64*)B;\n\
         reduce_entity<{nb}u, {rows}u, {stride}u, {}u>(COL(B, F, C{cid}_REC), COL(B, u32, C{cid}_BOFF), COL(B, u32, C{cid}_BINST), COL(B, u32, C{cid}_IDXB), COL(B, u32, C{cid}_TILEB), h, g, slots);\n}}\n\
         KERNEL_THREADS(256) void __dk{cid}_rx(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT h, u32 slots) {{\n\
         GLOBAL const u64* L = (GLOBAL const u64*)B;\n\
         reduce_cross<{na}u, {nb}u, {rows}u, {stride}u>(COL(B, F, C{cid}_REC), COL(B, u32, C{cid}_XOFF), COL(B, u32, C{cid}_XINST), COL(B, u32, C{cid}_SA), COL(B, u32, C{cid}_SB), \
         COL(B, u32, C{cid}_IDXA), COL(B, u32, C{cid}_TILEA), COL(B, u32, C{cid}_IDXB), COL(B, u32, C{cid}_TILEB), COL(B, u32, C{cid}_TILEX), h, slots);\n}}\n\
         KERNEL_THREADS(256) void __dk{cid}_csum(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT cost, u32 parts) {{\n\
         GLOBAL const u64* L = (GLOBAL const u64*)B;\ncost_sum(COL(B, W, C{cid}_PART), parts, cost);\n}}\n",
        1 + na
    );
    // the cost: the rows alone
    let mut cbody = stmts(cost.0)?;
    for (ri, r) in cost.1.iter().enumerate() {
        let r = expr(r)?;
        cbody.push_str(&format!("const F r{ri} = {r};\nc += (W)r{ri} * (W)r{ri};\n"));
    }
    let cost_kernel = format!(
        "KERNEL_THREADS(256) void __dk{cid}_cost(GLOBAL u8* RESTRICT B, u32 n) {{\n{}{}{cbody}}}\ncost_partial(COL_MUT(B, W, C{cid}_PART), c);\n}}\n",
        prologue("B"),
        lets(cost_syms)
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
    let mut rotations: Vec<(usize, String, usize)> = Vec::new();
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
        eb: usize,
        x: usize,
        rots: Vec<usize>,
    }
    let mut ids: Vec<Ids> = Vec::new();
    for c in constraints {
        let ea = entity_of(&mut entities, &c.a);
        let eb = entity_of(&mut entities, &c.b);
        let x = match crosses.iter().position(|(h, b)| *h == c.holder && c.block == b) {
            Some(x) => x,
            None => {
                crosses.push((c.holder.clone(), c.block.to_string()));
                crosses.len() - 1
            }
        };
        let rots = c
            .rots
            .iter()
            .map(|(side, path, off)| {
                let e = if *side == 0 { ea } else { eb };
                match rotations.iter().position(|(re, rp, _)| *re == e && rp == path) {
                    Some(r) => r,
                    None => {
                        rotations.push((e, path.clone(), *off));
                        rotations.len() - 1
                    }
                }
            })
            .collect();
        ids.push(Ids { ea, eb, x, rots });
    }
    let rot_shape: Vec<TokenStream2> = rotations.iter().map(|(e, _, off)| quote! { (#e, #off) }).collect();
    let entity_n: Vec<usize> = entities.iter().map(|e| e.2).collect();
    let n_cross = crosses.len();
    let cshape: Vec<TokenStream2> = constraints.iter().zip(&ids).map(|(c, id)| {
        let (nd, rows, ea, eb, x) = (c.data.len(), c.rows, id.ea, id.eb, id.x);
        quote! { ::arael::__gpu::DeviceConstraint { words: 3usize, data: #nd, rows: #rows, a: #ea, b: #eb, cross: #x } }
    }).collect();
    // the C++ kernels, each constraint's column names aliased to the
    // root's numbering, which the host's layout table follows
    let mut cpp = String::new();
    for (c, id) in constraints.iter().zip(&ids) {
        let k = c.cid;
        let mut alias = |local: &str, root: String| cpp.push_str(&format!("#define ARAEL_COL_C{k}_{local} ARAEL_COL_{root}\n"));
        alias("EVA", format!("E{}_EV", id.ea));
        alias("EVB", format!("E{}_EV", id.eb));
        alias("IDXA", format!("E{}_IDX", id.ea));
        alias("IDXB", format!("E{}_IDX", id.eb));
        alias("TILEA", format!("E{}_TILE", id.ea));
        alias("TILEB", format!("E{}_TILE", id.eb));
        alias("TILEX", format!("X{}_TILE", id.x));
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
            let (av, ahb, bv, bhb, block) = (format_ident!("{}", c.a.var), format_ident!("{}", c.a.hb), format_ident!("{}", c.b.var), format_ident!("{}", c.b.hb), &c.block);
            let data: Vec<syn::Expr> = c.data.iter().map(|d| syn::parse_str(d).expect("a data leaf is an expression")).collect();
            let fixed = |side: &DevSide, e: usize, s: TokenStream2| -> TokenStream2 {
                let n = side.params.len();
                let vals: Vec<syn::Expr> = side.params.iter().map(|p| syn::parse_str(p).expect("a parameter symbol is an expression")).collect();
                let ks: Vec<usize> = (0..n).collect();
                quote! { { let __f = &mut __pack.fixed[#e]; #( __f[#s * #n + #ks] = (#vals) as #scalar; )* } }
            };
            let (ea, eb) = (id.ea, id.eb);
            let fa = fixed(&c.a, ea, quote! { __sa });
            let fb = fixed(&c.b, eb, quote! { __sb });
            let rots: Vec<TokenStream2> = c.rots.iter().zip(&id.rots).map(|((side, path, _), &r)| {
                let (var, s) = if *side == 0 { (format_ident!("{}", c.a.var), quote! { __sa }) } else { (format_ident!("{}", c.b.var), quote! { __sb }) };
                let field: syn::Expr = syn::parse_str(&format!("{}.{}", var, path)).expect("a rotation field path");
                quote! {
                    {
                        let __m = &#field.ref_rotation;
                        let __r = &mut __pack.refr[#r];
                        for __i in 0..3usize {
                            __r[#s * 9 + __i * 3] = __m[__i].x as f64;
                            __r[#s * 9 + __i * 3 + 1] = __m[__i].y as f64;
                            __r[#s * 9 + __i * 3 + 2] = __m[__i].z as f64;
                        }
                    }
                }
            }).collect();
            quote! {
                {
                    let __sa = #av.#ahb.slot() as usize;
                    let __sb = #bv.#bhb.slot() as usize;
                    __pack.inst[#cid].extend_from_slice(&[__sa as u32, __sb as u32, __frine.#block.slot()]);
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

                fn device_restore(&mut self, refs: &mut dyn FnMut(u32) -> Option<[f64; 9]>) {
                    arael::model::Model::set_references(self, refs);
                }
            }
        }
    }
}
