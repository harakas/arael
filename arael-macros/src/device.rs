//! The device head of `#[arael(root)]`: a constraint's body as two C++
//! kernels (the assembly's and the cost's) over its entities' parameter
//! values, the caches of their referenced fields and its data leaves, each
//! leaf read from the table it belongs to (the instance's columns, its
//! entity's data record, the root's record); the records' Rust and C++
//! mirrors; and the walks that write a model's structure (once per
//! structure: the entities' indices, the guards, the kept instances' slots)
//! and its values (per solve) for them, as an implementation of
//! `arael::__gpu::DeviceRoot`. All of it is wrapped in
//! `arael::__gpu_items!`, which drops it when arael is built without its
//! `gpu` feature.
//!
//! A constraint is read off the block writes its CPU assembly makes (the
//! constraint emission's plan): a side per entity with parameters written
//! into a self block (an entity through a ref, the instance itself, its
//! parent, an ancestor, the root), a pair per cross block between two of
//! them (the instance's own or a parent's), whatever the constraint's form.
//! Its instances are walked along the instance type's containment paths,
//! the walk binding the names the plan's block places, the body's data
//! leaves and the guard read. Guards are evaluated once per structure (a
//! guard does not change during a solve); a robust loss scales the record
//! by `sqrt(rho'(s))`, as the CPU's Jacobian rows are. Over referenced fields
//! (`SimpleEulerAngleParam` angles, computed and never re-centred; Euler-angle
//! and `QuaternionParam` rotations, `TransformParam` poses, `UnitVecParam`
//! directions, whose references the device keeps and moves), atom caches
//! (an `AngleParam`'s sin and cos, computed in the kernel from the angle)
//! and `f32` or `f64` roots. A constraint has no device form when its rows
//! write `coo` pairs, when an index `select` or `match` picks its rows or
//! its loss, or when it reads a leaf that follows the parameters and that
//! the device does not compute (a `ScaledTransformParam`'s scale); its
//! root then assembles on the CPU.

use std::collections::BTreeSet;

use arael_sym::cse::Intermediate;
use arael_sym::E;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};

/// One entity of a device constraint.
pub(crate) struct DevSide {
    pub ty: String,
    /// The head of its symbols in the body (`cam` of `cam.t.work().x`).
    pub var: String,
    /// Its SelfBlock field.
    pub hb: String,
    /// Its parameter symbols in slot order (`cam.t.work().x`, ...).
    pub params: Vec<String>,
    /// Its self block in the instance walk (`self.cams[__ei_cam].hb`,
    /// `__item.hb`), whose marker's slot is its row.
    pub slot: TokenStream2,
}

/// A cross tile of a device constraint: between its sides `i` and `j`
/// (`i < j`), the CrossBlock field `block` of the type `holder`, reached in
/// the walk as `access` (`__frine.hb`).
pub(crate) struct DevPair {
    pub i: usize,
    pub j: usize,
    pub holder: String,
    pub block: String,
    pub access: TokenStream2,
}

/// One device constraint, as the root's emission needs it.
pub(crate) struct DevConstraint {
    /// The group whose walk packs it.
    pub group_key: String,
    /// Its entities, in the order of a row's partials.
    pub sides: Vec<DevSide>,
    /// The cross tiles between pairs of its sides.
    pub pairs: Vec<DevPair>,
    /// The instance's type.
    pub holder: String,
    /// The instance data leaves, as the Rust expressions the walk reads.
    pub inst: Vec<String>,
    /// Per side, the entity data leaves, as paths in the entity
    /// (`prior.x`).
    pub ent: Vec<Vec<String>>,
    /// The root data leaves, as paths in the root.
    pub root: Vec<String>,
    /// The referenced fields read.
    pub refs: Vec<DevRef>,
    /// The kernels in C++, named by `cid`: they read every column through
    /// the layout table by the constraint's own names (`C<cid>_S0`,
    /// `C<cid>_EV0`, `C<cid>_VAL<r>`, ...), which the root aliases to its
    /// numbering ([`device_root`]).
    pub cpp: String,
    pub cid: usize,
    /// Residual rows.
    pub rows: usize,
    /// The constraint's guard, evaluated once per structure by the
    /// structure walk: an instance it drops gets no row.
    pub filter: Option<TokenStream2>,
}

/// A referenced parameter field a device constraint reads: a parameter
/// whose step is measured from a reference that moves to it after every
/// accepted step.
#[derive(Clone)]
pub(crate) struct DevRef {
    /// Its constraint side.
    pub side: usize,
    /// Its field in the entity.
    pub path: String,
    /// The offset of its step among the entity's parameters.
    pub step: usize,
    pub kind: RefKind,
}

/// The referenced parameter types the device keeps the reference of
/// (`arael::__gpu::ReferenceKind`).
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RefKind {
    Euler,
    Quaternion,
    Transform,
    UnitVec,
    SimpleEuler,
}

/// Which column of a referenced field's values a leaf reads.
#[derive(Clone, Copy)]
enum RefCol {
    /// The values in F.
    Val,
    /// The values in W.
    ValW,
    /// The derivatives by the step, in F.
    Der,
}

impl RefKind {
    /// The kind of the field `field` of the registered type `ty`.
    fn of(ty: &str, field: &str) -> Option<RefKind> {
        let l = crate::registry_lookup(ty)?;
        if l.universal_euler_angle_fields.iter().any(|f| f == field) || l.mixed_euler_angle_fields.iter().any(|(f, _)| f == field) {
            return Some(RefKind::Euler);
        }
        if l.universal_rotvec_fields.iter().any(|f| f == field) {
            return Some(RefKind::Quaternion);
        }
        if l.euler_angle_fields.iter().any(|f| f == field) {
            return Some(RefKind::SimpleEuler);
        }
        match l.fields.iter().find(|(f, _)| f == field)? {
            (_, crate::SymFieldType::Struct(t)) => match t.as_str() {
                "TransformParam" | "TransformParamF" => Some(RefKind::Transform),
                "UnitVecParam" | "UnitVecParamF" => Some(RefKind::UnitVec),
                _ => None,
            },
            _ => None,
        }
    }

    /// The values and the derivatives one entity holds (`ReferenceKind`'s
    /// `values` and `derivatives`).
    fn widths(self) -> (usize, usize) {
        match self {
            RefKind::Euler | RefKind::Quaternion | RefKind::SimpleEuler => (9, 27),
            RefKind::Transform => (12, 45),
            RefKind::UnitVec => (3, 6),
        }
    }

    /// The parameters of the step after its first, as `<var>.<field>`
    /// suffixes: (the first's, the others' with their offsets).
    fn step(self) -> (&'static str, &'static [(&'static str, usize)]) {
        match self {
            RefKind::Euler | RefKind::Quaternion => ("delta.x", &[("delta.y", 1), ("delta.z", 2)]),
            RefKind::Transform => ("w.work().x", &[("w.work().y", 1), ("w.work().z", 2), ("d.work().x", 3), ("d.work().y", 4), ("d.work().z", 5)]),
            RefKind::UnitVec => ("d.work().x", &[("d.work().y", 1)]),
            RefKind::SimpleEuler => ("work().x", &[("work().y", 1), ("work().z", 2)]),
        }
    }

    /// The column and the offset of the cache `tail` reads (what follows
    /// `<var>.<field>.`), per entity.
    fn leaf(self, tail: &str) -> Option<(RefCol, usize)> {
        fn idx(r: &str) -> Option<(usize, &str)> {
            let r = r.strip_prefix('[')?;
            let close = r.find(']')?;
            let i: usize = r[..close].trim().parse().ok()?;
            (i < 3).then_some((i, &r[close + 1..]))
        }
        fn comp(r: &str) -> Option<usize> {
            match r {
                ".x" => Some(0),
                ".y" => Some(1),
                ".z" => Some(2),
                _ => None,
            }
        }
        // `<name>[r].c`, `<name>[k][r].c`, `<name>.c`, `<name>[k].c`
        let mat = |r: &str| idx(r).and_then(|(row, r)| Some(row * 3 + comp(r)?));
        let dmat = |r: &str| idx(r).and_then(|(k, r)| Some(k * 9 + mat(r)?));
        let dvec = |r: &str| idx(r).and_then(|(k, r)| Some(k * 3 + comp(r)?));
        let (name, rest) = tail.split_at(tail.find(['[', '.']).unwrap_or(tail.len()));
        match (self, name) {
            (RefKind::Euler | RefKind::Quaternion | RefKind::Transform | RefKind::SimpleEuler, "rotation_matrix") => Some((RefCol::Val, mat(rest)?)),
            (RefKind::Euler, "rotation_matrix_wide") => Some((RefCol::ValW, mat(rest)?)),
            (RefKind::Euler | RefKind::Quaternion | RefKind::SimpleEuler, "rotation_matrix_deriv") => Some((RefCol::Der, dmat(rest)?)),
            (RefKind::Transform, "translation") => Some((RefCol::Val, 9 + comp(rest)?)),
            (RefKind::Transform, "rotation_matrix_dw") => Some((RefCol::Der, dmat(rest)?)),
            (RefKind::Transform, "translation_dd") => Some((RefCol::Der, 27 + dvec(rest)?)),
            (RefKind::Transform, "translation_dw") => Some((RefCol::Der, 36 + dvec(rest)?)),
            (RefKind::UnitVec, "unit") => Some((RefCol::Val, comp(rest)?)),
            (RefKind::UnitVec, "unit_d") => Some((RefCol::Der, dvec(rest)?)),
            _ => None,
        }
    }

    /// The values walk's read of the reference of `field`, an `[f64; N]`;
    /// `None` for a kind without a reference.
    fn read(self, field: &TokenStream2) -> Option<TokenStream2> {
        Some(match self {
            RefKind::SimpleEuler => return None,
            RefKind::Euler => quote! {
                {
                    let __m = &#field.ref_rotation;
                    let __f = ::arael::__gpu::leaf_f64;
                    [__f(__m[0].x), __f(__m[0].y), __f(__m[0].z), __f(__m[1].x), __f(__m[1].y), __f(__m[1].z), __f(__m[2].x), __f(__m[2].y), __f(__m[2].z)]
                }
            },
            RefKind::Quaternion | RefKind::UnitVec => quote! { #field.__reference_quaternion() },
            RefKind::Transform => quote! { #field.__reference() },
        })
    }

    fn ident(self) -> TokenStream2 {
        match self {
            RefKind::Euler => quote! { Euler },
            RefKind::Quaternion => quote! { Quaternion },
            RefKind::Transform => quote! { Transform },
            RefKind::UnitVec => quote! { UnitVec },
            RefKind::SimpleEuler => quote! { SimpleEuler },
        }
    }
}

/// A CSE temporary's name: `__x3`, `__w1`, `__m2`.
/// A robust loss of a device constraint, over the squared norm of its
/// instance's rows ([`LOSS_ARG_SYM`]): `rho(s)` for the cost kernel, and
/// `rho(s)` and the weight `rho'(s)` with shared statements for the
/// assembly kernel.
pub(crate) struct DevLoss {
    pub cost: (Vec<Intermediate>, E),
    pub gh: (Vec<Intermediate>, E, E),
}

use crate::constraint::LOSS_ARG_SYM;

fn is_temp(s: &str) -> bool {
    let rest = s.strip_prefix("__x").or_else(|| s.strip_prefix("__w")).or_else(|| s.strip_prefix("__m"));
    rest.is_some_and(|r| !r.is_empty() && r.bytes().all(|b| b.is_ascii_digit()))
}

/// The leaves `e` reads: its symbols but the CSE's temporaries and the
/// loss's argument, which the kernel computes.
fn syms_of(e: &E, out: &mut BTreeSet<String>) {
    for s in e.symbols() {
        if !is_temp(&s) && s != LOSS_ARG_SYM {
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

/// The referenced field a leaf reads a cache of: (side, field, kind,
/// column, offset). `None` for a leaf of no referenced field.
fn reference_leaf(s: &str, sides: &[DevSide]) -> Option<(usize, String, RefKind, RefCol, usize)> {
    let (var, rest) = s.split_once('.')?;
    let (side, sd) = sides.iter().enumerate().find(|(_, sd)| sd.var == var)?;
    let (field, tail) = rest.split_once('.')?;
    let kind = RefKind::of(&sd.ty, field)?;
    let (col, off) = kind.leaf(tail)?;
    Some((side, field.to_string(), kind, col, off))
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
/// the leaves the device computes (the parameters, the referenced fields' caches, the atom
/// caches), so it would hold the value the solve started from.
fn parameter_state(s: &str, sides: &[DevSide]) -> Option<String> {
    let (var, rest) = s.split_once('.')?;
    let side = sides.iter().find(|sd| sd.var == var)?;
    let field = rest.split(['.', '[']).next()?;
    parameter_field(&side.ty, field)
        .then(|| format!("the device reads `{s}`, which follows the parameters of `{var}.{field}`, without computing it"))
}

/// A record field's name for the model path `path`: `prior_rot_t[0].x` is
/// `prior_rot_t_0_x`. The same in the Rust and the C++ mirrors.
pub(crate) fn field_name(path: &str) -> String {
    let mut out = String::with_capacity(path.len());
    for ch in path.chars() {
        if ch.is_ascii_alphanumeric() || ch == '_' {
            out.push(ch);
        } else if !out.ends_with('_') {
            out.push('_');
        }
    }
    out.trim_end_matches('_').to_string()
}

/// What a leaf symbol reads in the kernel.
enum Leaf {
    Param(usize, usize),
    /// A cache of referenced field `r`: its column and offset per entity.
    Ref(usize, RefCol, usize),
    /// Instance data column `j`.
    Inst(usize),
    /// A field of side `s`'s entity data record.
    Ent(usize, String),
    /// A field of the root's data record.
    Root(String),
    /// An atom cache (`a.rot.sin`), computed from its atom over the
    /// parameters (`sin(a.rot.angle.work())`).
    Atom(E),
}

/// The kernels of a constraint over the entities `sides`, with a cross tile
/// per pair of `pairs`, an instance of `holder`: `gh` and `guard` from the
/// assembly's CSE (per row the residual, then the partials side by side),
/// `cost` from the cost's. `ft` is the root's scalar. `atoms` are the
/// body's atom caches and their atoms (`a.rot.sin`, `sin(a.rot.angle.work())`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn device_constraint(
    cid: usize,
    group_key: String,
    ft: &str,
    sides: Vec<DevSide>,
    pairs: Vec<DevPair>,
    holder: String,
    rows: usize,
    gh: (&[Intermediate], &[E]),
    guard: Option<(&[Intermediate], &E)>,
    cost: (&[Intermediate], &[E]),
    loss: Option<&DevLoss>,
    atoms: &[(String, E)],
    root_var: &str,
) -> Result<DevConstraint, String> {
    if ft != "f32" && ft != "f64" {
        return Err(format!("a root of `{ft}` has no device assembly"));
    }
    if let Some(sd) = sides.iter().find(|sd| sd.params.is_empty()) {
        return Err(format!("the entity `{}` has no parameters", sd.var));
    }
    if let Some(p) = pairs.iter().find(|p| p.i >= p.j || p.j >= sides.len()) {
        return Err(format!("the cross tile `{}` joins sides {} and {} of {}", p.block, p.i, p.j, sides.len()));
    }
    let ns: Vec<usize> = sides.iter().map(|sd| sd.params.len()).collect();
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
    if let Some(l) = loss {
        syms_of_inters(&l.gh.0, &mut gh_syms);
        [&l.gh.1, &l.gh.2].into_iter().for_each(|e| syms_of(e, &mut gh_syms));
        syms_of_inters(&l.cost.0, &mut cost_syms);
        syms_of(&l.cost.1, &mut cost_syms);
    }
    // a kernel computing an atom reads its parameters
    let atom = |s: &str| atoms.iter().find(|(c, _)| c == s).map(|(_, e)| e);
    for syms in [&mut gh_syms, &mut cost_syms] {
        let used: Vec<&E> = syms.iter().filter_map(|s| atom(s)).collect();
        let mut more = BTreeSet::new();
        used.into_iter().for_each(|e| syms_of(e, &mut more));
        syms.append(&mut more);
    }
    let all: BTreeSet<&String> = gh_syms.iter().chain(&cost_syms).collect();
    // classified, the data leaves and the referenced fields numbered in order
    let mut inst: Vec<String> = Vec::new();
    let mut ent: Vec<Vec<String>> = sides.iter().map(|_| Vec::new()).collect();
    let mut root: Vec<String> = Vec::new();
    let mut refs: Vec<DevRef> = Vec::new();
    let mut leaf: std::collections::BTreeMap<String, Leaf> = Default::default();
    let param = |s: &str| sides.iter().enumerate().find_map(|(i, sd)| sd.params.iter().position(|p| p == s).map(|p| (i, p)));
    for s in all {
        let l = if let Some(e) = atom(s) {
            if let Some(d) = e.symbols().into_iter().find(|d| param(d).is_none()) {
                return Err(format!("the atom cache `{s}` reads `{d}`, not a parameter of the constraint's entities"));
            }
            Leaf::Atom(e.clone())
        } else if let Some((side, p)) = param(s) {
            Leaf::Param(side, p)
        } else if let Some((side, path, kind, col, off)) = reference_leaf(s, &sides) {
            let r = match refs.iter().position(|r| r.side == side && r.path == path) {
                Some(r) => r,
                None => {
                    // the step: its entity's parameters at `<var>.<field>.<first>`
                    // and after
                    let sd = &sides[side];
                    let (first, rest) = kind.step();
                    let name = |suffix: &str| format!("{}.{}.{}", sd.var, path, suffix);
                    let step = sd.params.iter().position(|p| *p == name(first))
                        .ok_or_else(|| format!("`{}.{}` has no step among its entity's parameters", sd.var, path))?;
                    if let Some((p, _)) = rest.iter().find(|(p, k)| sd.params.get(step + k) != Some(&name(p))) {
                        return Err(format!("`{}.{}`'s step is not in order among its entity's parameters at `{p}`", sd.var, path));
                    }
                    refs.push(DevRef { side, path, step, kind });
                    refs.len() - 1
                }
            };
            Leaf::Ref(r, col, off)
        } else {
            if let Some(why) = parameter_state(s, &sides) {
                return Err(why);
            }
            let (var, path) = s.split_once('.').unwrap_or((s.as_str(), ""));
            let side = sides.iter().position(|sd| sd.var == var);
            match side {
                Some(side) if !path.is_empty() => {
                    ent[side].push(path.to_string());
                    Leaf::Ent(side, field_name(path))
                }
                _ if var == root_var && !path.is_empty() => {
                    root.push(path.to_string());
                    Leaf::Root(field_name(path))
                }
                _ => {
                    inst.push(s.clone());
                    Leaf::Inst(inst.len() - 1)
                }
            }
        };
        leaf.insert(s.clone(), l);
    }
    let names: std::collections::BTreeMap<String, String> = leaf.keys().enumerate().map(|(i, s)| (s.clone(), format!("__l{i}"))).collect();
    let pair_sides: Vec<(usize, usize)> = pairs.iter().map(|p| (p.i, p.j)).collect();
    let cpp = cpp_kernels(cid, ft, (&ns, &pair_sides), rows, &leaf, &names, &refs, (&gh_syms, &cost_syms), gh, guard, cost, loss)?;
    Ok(DevConstraint { group_key, sides, pairs, holder, inst, ent, root, refs, cpp, cid, rows, filter: None })
}

/// The kernels of [`device_constraint`] in C++
/// (`arael-gpu/kernels/assembly.cpp` is compiled in front of them): every
/// column is read through the layout table `L` at the start of the model's
/// block `B`, by the constraint's own column names. `ns` are the sides'
/// parameters, `pairs` the sides of each cross tile.
#[allow(clippy::too_many_arguments)]
fn cpp_kernels(
    cid: usize,
    ft: &str,
    (ns, pairs): (&[usize], &[(usize, usize)]),
    rows: usize,
    leaf: &std::collections::BTreeMap<String, Leaf>,
    names: &std::collections::BTreeMap<String, String>,
    refs: &[DevRef],
    (gh_syms, cost_syms): (&BTreeSet<String>, &BTreeSet<String>),
    gh: (&[Intermediate], &[E]),
    guard: Option<(&[Intermediate], &E)>,
    cost: (&[Intermediate], &[E]),
    loss: Option<&DevLoss>,
) -> Result<String, String> {
    // where each side's partials start in a row
    let offs: Vec<usize> = ns.iter().scan(1, |at, &n| {
        let o = *at;
        *at += n;
        Some(o)
    }).collect();
    // the loss's argument is the kernel's local of that name
    let map = |s: &str| -> Option<String> {
        if s == LOSS_ARG_SYM {
            Some(s.to_string())
        } else {
            names.get(s).cloned()
        }
    };
    // a leaf's type and the read that gives it
    let read = |l: &Leaf| -> Result<(&'static str, String), String> {
        Ok(match *l {
            Leaf::Param(i, p) => ("F", format!("COL(B, F, C{cid}_EV{i})[s{i} * {}u + {p}u]", ns[i])),
            Leaf::Ref(r, col, off) => {
                let i = refs[r].side;
                let (nv, nd) = refs[r].kind.widths();
                match col {
                    RefCol::Val => ("F", format!("COL(B, F, C{cid}_VAL{r})[s{i} * {nv}u + {off}u]")),
                    RefCol::ValW => ("W", format!("COL(B, W, C{cid}_VALW{r})[s{i} * {nv}u + {off}u]")),
                    RefCol::Der => ("F", format!("COL(B, F, C{cid}_DER{r})[s{i} * {nd}u + {off}u]")),
                }
            }
            Leaf::Inst(j) => ("F", format!("COL(B, F, C{cid}_D{j})[o]")),
            Leaf::Ent(i, ref f) => ("F", format!("COL(B, C{cid}_Rec{i}, C{cid}_DATA{i})[s{i}].{f}")),
            Leaf::Root(ref f) => ("F", format!("COL(B, RootData, ROOT)[0].{f}")),
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
        let slots: String = (0..ns.len()).map(|i| format!("const u32 s{i} = COL({b}, u32, C{cid}_S{i})[o];\n")).collect();
        format!("GLOBAL const u64* L = (GLOBAL const u64*)B;\nconst u32 o = global_x();\nW c = 0.0f;\nif (o < n) {{\n{slots}")
    };
    // the linearization: per instance its record, R rows of the residual
    // and the sides' derivatives, zero where the guard drops it
    let n_partials: usize = ns.iter().sum();
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
    // Under a loss the cost is rho(s) of the rows' squared norm s, and the
    // record is scaled by
    // sqrt(rho'(s)) (zero where rho' is not positive, as the CPU's
    // Jacobian rows are): the reducers' sums of its products are then the
    // CPU's rho'(s)-weighted gradient and Hessian. In a block of its own,
    // so the loss's temporaries do not meet the body's. Behind the guard,
    // as the CPU's assembly skips the loss where the rows are off.
    let costs = match loss {
        Some(l) => {
            body.push_str(&format!("F {LOSS_ARG_SYM} = (F)0.0f;\n"));
            for ri in 0..rows {
                body.push_str(&format!("{LOSS_ARG_SYM} += rec[{0}] * rec[{0}];\n", ri * stride));
            }
            body.push_str(&format!(
                "{{\n{}c += (W)({});\nconst F __lw = {};\nconst F __sw = __lw > (F)0.0f ? sqrt(__lw) : (F)0.0f;\n\
                 #pragma unroll\nfor (u32 k = 0; k < {recw}u; k++) rec[k] *= __sw;\n}}\n",
                stmts(&l.gh.0)?,
                expr(&l.gh.1)?,
                expr(&l.gh.2)?
            ));
            String::new()
        }
        None => (0..rows).map(|ri| format!("c += (W)rec[{0}] * (W)rec[{0}];\n", ri * stride)).collect(),
    };
    let gh_code = match guard {
        Some((pre, q)) => format!("{}if ({} >= 0.0f) {{\n{body}}}\n", stmts(pre)?, expr(q)?),
        None => body,
    };
    let gh_kernel = format!(
        "KERNEL_THREADS(256) void __dk{cid}_gh(GLOBAL u8* RESTRICT B, u32 n) {{\n{}{}\
         F rec[{recw}];\n#pragma unroll\nfor (u32 k = 0; k < {recw}u; k++) rec[k] = (F)0.0f;\n{gh_code}{costs}\
         GLOBAL F* out = COL_MUT(B, F, C{cid}_REC) + o * {recw}u;\n#pragma unroll\nfor (u32 k = 0; k < {recw}u; k++) out[k] = rec[k];\n\
         }}\ncost_partial(COL_MUT(B, W, C{cid}_PART), c);\n}}\n",
        prologue("B"),
        lets(gh_syms)?
    );
    // the sums of the records: the cost, each side's entities, each pair's
    // cross tiles
    let mut reduce = format!(
        "KERNEL_THREADS(256) void __dk{cid}_csum(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT cost, u32 parts) {{\n\
         GLOBAL const u64* L = (GLOBAL const u64*)B;\ncost_sum(COL(B, W, C{cid}_PART), parts, cost);\n}}\n"
    );
    for (i, (&n, &off)) in ns.iter().zip(&offs).enumerate() {
        reduce.push_str(&format!(
            "KERNEL_THREADS(256) void __dk{cid}_r{i}(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT h, GLOBAL W* RESTRICT g, u32 slots) {{\n\
             GLOBAL const u64* L = (GLOBAL const u64*)B;\n\
             reduce_entity<{n}u, {rows}u, {stride}u, {off}u>(COL(B, F, C{cid}_REC), COL(B, u32, C{cid}_OFF{i}), COL(B, u32, C{cid}_INST{i}), \
             COL(B, u32, C{cid}_IDX{i}), COL(B, u32, C{cid}_TILE{i}), h, g, slots);\n}}\n"
        ));
    }
    for (p, &(i, j)) in pairs.iter().enumerate() {
        let (ni, nj, oi, oj) = (ns[i], ns[j], offs[i], offs[j]);
        reduce.push_str(&format!(
            "KERNEL_THREADS(256) void __dk{cid}_x{p}(GLOBAL const u8* RESTRICT B, GLOBAL W* RESTRICT h, u32 slots) {{\n\
             GLOBAL const u64* L = (GLOBAL const u64*)B;\n\
             reduce_cross<{ni}u, {nj}u, {rows}u, {stride}u, {oi}u, {oj}u>(COL(B, F, C{cid}_REC), COL(B, u32, C{cid}_XOFF{p}), COL(B, u32, C{cid}_XINST{p}), \
             COL(B, u32, C{cid}_S{i}), COL(B, u32, C{cid}_S{j}), COL(B, u32, C{cid}_IDX{i}), COL(B, u32, C{cid}_TILE{i}), \
             COL(B, u32, C{cid}_IDX{j}), COL(B, u32, C{cid}_TILE{j}), COL(B, u32, C{cid}_TILEX{p}), h, slots);\n}}\n"
        ));
    }
    // the cost: the rows alone, or rho of their squared norm
    let mut cbody = stmts(cost.0)?;
    if loss.is_some() {
        cbody.push_str(&format!("F {LOSS_ARG_SYM} = (F)0.0f;\n"));
    }
    for (ri, r) in cost.1.iter().enumerate() {
        let r = expr(r)?;
        cbody.push_str(&match loss {
            Some(_) => format!("const F r{ri} = {r};\n{LOSS_ARG_SYM} += r{ri} * r{ri};\n"),
            None => format!("const F r{ri} = {r};\nc += (W)r{ri} * (W)r{ri};\n"),
        });
    }
    if let Some(l) = loss {
        cbody.push_str(&format!("{{\n{}c += (W)({});\n}}\n", stmts(&l.cost.0)?, expr(&l.cost.1)?));
    }
    let cost_kernel = format!(
        "KERNEL_THREADS(256) void __dk{cid}_cost(GLOBAL u8* RESTRICT B, u32 n) {{\n{}{}{cbody}}}\ncost_partial(COL_MUT(B, W, C{cid}_PART), c);\n}}\n",
        prologue("B"),
        lets(cost_syms)?
    );
    Ok(format!("{gh_kernel}{reduce}{cost_kernel}"))
}

/// The walk over a constraint's instances, as the root's emission hands it
/// over: loops around a body that reads the instance as `__frine`, its
/// entities, its parents and the root by the names the body and the plan's
/// block places use.
pub(crate) struct DevGroup {
    pub walk: Box<dyn Fn(&TokenStream2) -> TokenStream2>,
}

/// A data record of the image (an entity table's or the root's): its fields
/// in order, in F, padded to 16 bytes, as a Rust struct and a C++ struct of
/// the same layout.
struct Record {
    /// (field name, model path)
    fields: Vec<(String, String)>,
    pad: usize,
}

impl Record {
    fn new(paths: &[String], ft: &str) -> Result<Record, String> {
        let mut fields: Vec<(String, String)> = Vec::new();
        for p in paths {
            let f = field_name(p);
            match fields.iter().find(|(n, _)| *n == f) {
                Some((_, q)) if q != p => return Err(format!("the data fields `{q}` and `{p}` take one record field name `{f}`")),
                Some(_) => {}
                None => fields.push((f, p.clone())),
            }
        }
        let per16 = if ft == "f64" { 2 } else { 4 };
        let pad = (per16 - fields.len() % per16) % per16;
        Ok(Record { fields, pad })
    }

    fn bytes(&self, ft: &str) -> usize {
        if self.fields.is_empty() {
            return 0;
        }
        (self.fields.len() + self.pad) * if ft == "f64" { 8 } else { 4 }
    }

    /// The C++ struct, `name`.
    fn cpp(&self, name: &str) -> String {
        if self.fields.is_empty() {
            return String::new();
        }
        let mut s = format!("struct alignas(16) {name} {{");
        for (f, p) in &self.fields {
            s.push_str(&format!(" F {f}; /* {p} */"));
        }
        if self.pad > 0 {
            s.push_str(&format!(" F _pad[{}];", self.pad));
        }
        s.push_str(" };\n");
        s
    }

    /// The Rust struct, `name`, with its `Pod` implementation.
    fn rust(&self, name: &syn::Ident, scalar: &syn::Type) -> TokenStream2 {
        if self.fields.is_empty() {
            return quote! {};
        }
        let fs: Vec<syn::Ident> = self.fields.iter().map(|(f, _)| format_ident!("{}", f)).collect();
        let pad = self.pad;
        let pad_field = (pad > 0).then(|| quote! { _pad: [#scalar; #pad], });
        quote! {
            #[repr(C)]
            #[derive(Clone, Copy)]
            #[allow(non_camel_case_types)]
            #[doc(hidden)]
            pub struct #name { #( #fs: #scalar, )* #pad_field }
            // SAFETY: `repr(C)` scalars of one type, padded by hand to a
            // multiple of 16 bytes: no padding bytes, any bit pattern valid.
            unsafe impl ::arael::__gpu::bytemuck::Zeroable for #name {}
            unsafe impl ::arael::__gpu::bytemuck::Pod for #name {}
        }
    }

    /// The record's value with each field read from `acc.<path>`.
    fn value(&self, name: &syn::Ident, acc: &TokenStream2, scalar: &syn::Type) -> TokenStream2 {
        let fs: Vec<syn::Ident> = self.fields.iter().map(|(f, _)| format_ident!("{}", f)).collect();
        let reads: Vec<syn::Expr> = self
            .fields
            .iter()
            .map(|(_, p)| syn::parse_str(&format!("__acc.{p}")).expect("a data path is an expression"))
            .collect();
        let pad = self.pad;
        let pad_field = (pad > 0).then(|| quote! { _pad: [0.0 as #scalar; #pad], });
        quote! {
            {
                let __acc = #acc;
                #name { #( #fs: (#reads) as #scalar, )* #pad_field }
            }
        }
    }
}

/// The device items of a root whose every constraint has a device form.
/// `entity_walk(type, body)` walks every entity of `type` in slot order,
/// `body` reading it as `__ent`; `idx_stmts(type, acc, idx)` fills `idx`
/// with the parameters' global indices of the entity `acc`.
pub(crate) fn device_root(
    root_name: &syn::Ident,
    ft: &str,
    constraints: &[DevConstraint],
    groups: &std::collections::BTreeMap<String, DevGroup>,
    entity_walk: &dyn Fn(&str, TokenStream2) -> TokenStream2,
    idx_stmts: &dyn Fn(&str, &TokenStream2, &syn::Ident) -> Vec<TokenStream2>,
) -> TokenStream2 {
    let scalar: syn::Type = syn::parse_str(ft).unwrap();
    // the device's numbering of the entity tables, referenced fields and
    // cross arrays the constraints read; per entity table its type, self
    // block field, parameters (paths in the entity) and data paths
    struct Entity {
        ty: String,
        hb: String,
        params: Vec<String>,
        data: Vec<String>,
    }
    let mut entities: Vec<Entity> = Vec::new();
    let mut references: Vec<(usize, DevRef)> = Vec::new();
    let mut crosses: Vec<(String, String)> = Vec::new();
    let mut root_data: Vec<String> = Vec::new();
    let entity_of = |entities: &mut Vec<Entity>, s: &DevSide, data: &[String]| -> usize {
        let e = match entities.iter().position(|en| en.ty == s.ty && en.hb == s.hb) {
            Some(e) => e,
            None => {
                let prefix = format!("{}.", s.var);
                let params = s.params.iter().map(|p| p.strip_prefix(&prefix).expect("a parameter of the side's entity").to_string()).collect();
                entities.push(Entity { ty: s.ty.clone(), hb: s.hb.clone(), params, data: Vec::new() });
                entities.len() - 1
            }
        };
        for d in data {
            if !entities[e].data.contains(d) {
                entities[e].data.push(d.clone());
            }
        }
        e
    };
    struct Ids {
        /// Per side its entity table.
        sides: Vec<usize>,
        /// Per pair its sides and its cross array.
        pairs: Vec<(usize, usize, usize)>,
        refs: Vec<usize>,
    }
    let mut ids: Vec<Ids> = Vec::new();
    for c in constraints {
        let sides: Vec<usize> = c.sides.iter().zip(&c.ent).map(|(sd, ent)| entity_of(&mut entities, sd, ent)).collect();
        let pairs = c
            .pairs
            .iter()
            .map(|p| {
                let x = match crosses.iter().position(|(h, bl)| *h == p.holder && *bl == p.block) {
                    Some(x) => x,
                    None => {
                        crosses.push((p.holder.clone(), p.block.clone()));
                        crosses.len() - 1
                    }
                };
                (p.i, p.j, x)
            })
            .collect();
        let refs = c
            .refs
            .iter()
            .map(|rf| {
                let e = sides[rf.side];
                match references.iter().position(|(re, r)| *re == e && r.path == rf.path) {
                    Some(r) => r,
                    None => {
                        references.push((e, rf.clone()));
                        references.len() - 1
                    }
                }
            })
            .collect();
        for d in &c.root {
            if !root_data.contains(d) {
                root_data.push(d.clone());
            }
        }
        ids.push(Ids { sides, pairs, refs });
    }
    // the records and their mirrors
    let records: Result<Vec<Record>, String> = entities.iter().map(|en| Record::new(&en.data, ft)).collect();
    let (records, root_record) = match (records, Record::new(&root_data, ft)) {
        (Ok(r), Ok(rr)) => (r, rr),
        (Err(e), _) | (_, Err(e)) => {
            let msg = format!("{root_name}: {e}");
            return quote! { ::core::compile_error!(#msg); };
        }
    };
    let rec_ident = |e: usize| format_ident!("__{}DevE{}", root_name, e);
    let root_ident = format_ident!("__{}DevRoot", root_name);
    let rust_records: Vec<TokenStream2> = records.iter().enumerate().map(|(e, r)| r.rust(&rec_ident(e), &scalar)).collect();
    let rust_root = root_record.rust(&root_ident, &scalar);
    let entity_bytes: Vec<usize> = records.iter().map(|r| r.bytes(ft)).collect();
    let root_bytes = root_record.bytes(ft);

    let ref_shape: Vec<TokenStream2> = references
        .iter()
        .map(|(e, r)| {
            let (step, kind) = (r.step, r.kind.ident());
            quote! { ::arael::__gpu::DeviceReference { entity: #e, step: #step, kind: ::arael::__gpu::ReferenceKind::#kind } }
        })
        .collect();
    let entity_n: Vec<usize> = entities.iter().map(|e| e.params.len()).collect();
    let n_cross = crosses.len();
    let cshape: Vec<TokenStream2> = constraints.iter().zip(&ids).map(|(c, id)| {
        let (nd, rows, sides) = (c.inst.len(), c.rows, &id.sides);
        let pairs = id.pairs.iter().map(|(i, j, x)| quote! { (#i, #j, #x) });
        quote! { ::arael::__gpu::DeviceConstraint { data: #nd, rows: #rows, sides: vec![#(#sides),*], pairs: vec![#(#pairs),*] } }
    }).collect();
    // the C++: the records, then the kernels, each constraint's column and
    // record names aliased to the root's numbering, which the host's layout
    // table follows
    let mut cpp = String::new();
    for (e, r) in records.iter().enumerate() {
        cpp.push_str(&r.cpp(&format!("E{e}Data")));
    }
    cpp.push_str(&root_record.cpp("RootData"));
    for (c, id) in constraints.iter().zip(&ids) {
        let k = c.cid;
        let mut alias = |local: &str, root: String| cpp.push_str(&format!("#define ARAEL_COL_C{k}_{local} ARAEL_COL_{root}\n"));
        for (i, e) in id.sides.iter().enumerate() {
            for col in ["EV", "IDX", "TILE", "DATA"] {
                alias(&format!("{col}{i}"), format!("E{e}_{col}"));
            }
        }
        for (p, (_, _, x)) in id.pairs.iter().enumerate() {
            alias(&format!("TILEX{p}"), format!("X{x}_TILE"));
        }
        for (lr, &r) in id.refs.iter().enumerate() {
            alias(&format!("VAL{lr}"), format!("R{r}_VAL"));
            alias(&format!("VALW{lr}"), format!("R{r}_VALW"));
            alias(&format!("DER{lr}"), format!("R{r}_DER"));
        }
        for (i, e) in id.sides.iter().enumerate() {
            if !c.ent[i].is_empty() {
                cpp.push_str(&format!("typedef E{e}Data C{k}_Rec{i};\n"));
            }
        }
        cpp.push_str(&c.cpp);
    }

    // the structure walk: per entity table its indices in slot order, per
    // group the cost sweep's loop, per constraint its guard and its kept
    // instances' slots
    let idx_ident = format_ident!("__idx");
    let structure_entities: Vec<TokenStream2> = entities.iter().enumerate().map(|(e, en)| {
        let (hb, n, ty) = (format_ident!("{}", en.hb), en.params.len(), &en.ty);
        let fill = idx_stmts(&en.ty, &quote! { __ent }, &idx_ident);
        let msg = format!("{ty}: the device numbers the entities in container order, as the block markers are");
        let body = quote! {
            assert_eq!(__ent.#hb.slot() as usize, __n, #msg);
            let mut __idx = [u32::MAX; #n];
            #(#fill)*
            __s.idx[#e].extend_from_slice(&__idx);
            __n += 1;
        };
        let walk = entity_walk(&en.ty, body);
        quote! { { let mut __n = 0usize; #walk } }
    }).collect();
    // the values walk: per entity table its fixed values, references and
    // data record; the root's record; per group the cost sweep's loop, per
    // kept instance its data
    let values_entities: Vec<TokenStream2> = entities.iter().enumerate().map(|(e, en)| {
        let reads: Vec<syn::Expr> = en.params.iter().map(|p| syn::parse_str(&format!("__ent.{p}")).expect("a parameter path is an expression")).collect();
        let fixed = quote! { __w.fixed(#e, __n, [#( ::arael::__gpu::leaf_f64(#reads) as #scalar ),*]); };
        let refs: Vec<TokenStream2> = references.iter().enumerate().filter(|(_, (re, _))| *re == e).filter_map(|(r, (_, rf))| {
            let field: syn::Expr = syn::parse_str(&format!("__ent.{}", rf.path)).expect("a referenced field path");
            let read = rf.kind.read(&quote! { #field })?;
            Some(quote! { __w.reference(#r, __n, #read); })
        }).collect();
        let data = (!records[e].fields.is_empty()).then(|| {
            let value = records[e].value(&rec_ident(e), &quote! { __ent }, &scalar);
            quote! { __w.entity_data(#e, __n, #value); }
        });
        let walk = entity_walk(&en.ty, quote! {
            #fixed
            #(#refs)*
            #data
            __n += 1;
        });
        quote! { { let mut __n = 0usize; #walk } }
    }).collect();
    let values_root = (!root_record.fields.is_empty()).then(|| {
        let value = root_record.value(&root_ident, &quote! { self }, &scalar);
        quote! { __w.root_data(#value); }
    });
    let mut structure_walks: Vec<TokenStream2> = Vec::new();
    let mut values_walks: Vec<TokenStream2> = Vec::new();
    for (key, g) in groups {
        let mine: Vec<(&DevConstraint, &Ids)> = constraints.iter().zip(&ids).filter(|(c, _)| c.group_key == *key).collect();
        if mine.is_empty() {
            continue;
        }
        // per constraint its keep flags and its kept instances' words: the
        // sides' slots, then the pairs' cross slots
        let structure: Vec<TokenStream2> = mine.iter().map(|(c, _)| {
            let cid = c.cid;
            let words = c
                .sides
                .iter()
                .map(|sd| {
                    let slot = &sd.slot;
                    quote! { #slot.slot() }
                })
                .chain(c.pairs.iter().map(|p| {
                    let access = &p.access;
                    quote! { #access.slot() }
                }));
            let cond = c.filter.clone().unwrap_or_else(|| quote! { true });
            quote! {
                {
                    let __k: bool = #cond;
                    __s.keep[#cid].push(__k);
                    if __k {
                        __s.rows[#cid].extend_from_slice(&[#(#words),*]);
                    }
                }
            }
        }).collect();
        let values: Vec<TokenStream2> = mine.iter().map(|(c, _)| {
            let cid = c.cid;
            let reads: Vec<syn::Expr> = c.inst.iter().map(|d| syn::parse_str(d).expect("a data leaf is an expression")).collect();
            let write = (!reads.is_empty()).then(|| quote! { __w.instance(#cid, __row[#cid], [#( (#reads) as #scalar ),*]); });
            let msg = format!("{}: the model has more instances than its structure: it changed without LmSession::invalidate()", c.holder);
            quote! {
                {
                    let __k = *__keep[#cid].get(__i[#cid]).expect(#msg);
                    __i[#cid] += 1;
                    if __k {
                        #write
                        __row[#cid] += 1;
                    }
                }
            }
        }).collect();
        structure_walks.push((g.walk)(&quote! { #(#structure)* }));
        values_walks.push((g.walk)(&quote! { #(#values)* }));
    }
    let n_constraints = constraints.len();
    let fewer: Vec<String> = constraints.iter().map(|c| format!("{}: the model has fewer instances than its structure: it changed without LmSession::invalidate()", c.holder)).collect();
    let cids: Vec<usize> = constraints.iter().map(|c| c.cid).collect();
    quote! {
        ::arael::__gpu_items! {
            #(#rust_records)*
            #rust_root

            impl ::arael::__gpu::DeviceRoot<#scalar> for #root_name {
                fn device_shape(&self) -> ::arael::__gpu::DeviceShape {
                    ::arael::__gpu::DeviceShape {
                        entities: vec![#(#entity_n),*],
                        references: vec![#(#ref_shape),*],
                        crosses: #n_cross,
                        constraints: vec![#(#cshape),*],
                        entity_data: vec![#(#entity_bytes),*],
                        root_data: #root_bytes,
                    }
                }

                #[allow(unused_variables, clippy::all)]
                fn device_structure(&self, __s: &mut ::arael::__gpu::DeviceStructure) {
                    use arael::utils::{Float as _, SelectIndex as _};
                    use arael::store::Leaves as _;
                    #(#structure_entities)*
                    #(#structure_walks)*
                }

                #[allow(unused_variables, unused_mut, clippy::all)]
                fn device_values(&self, __keep: &[::std::vec::Vec<bool>], __w: &mut ::arael::__gpu::DeviceWriter<'_, #scalar>) {
                    use arael::utils::{Float as _, SelectIndex as _};
                    use arael::store::Leaves as _;
                    #(#values_entities)*
                    #values_root
                    let mut __i = [0usize; #n_constraints];
                    let mut __row = [0usize; #n_constraints];
                    #(#values_walks)*
                    #( assert_eq!(__i[#cids], __keep[#cids].len(), #fewer); )*
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
