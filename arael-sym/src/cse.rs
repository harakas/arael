//! Common Subexpression Elimination (CSE).
//!
//! Iteratively extracts repeated subexpressions, working bottom-up (deepest
//! first) so that inner replacements create new matching opportunities at
//! higher levels.
//!
//! Select arms are scopes: a subexpression used only inside arms is never
//! hoisted above the match, and every select on one index across the
//! batch is emitted as a single `match` returning one value per select,
//! with its own CSE run inside each arm.

use std::collections::{BTreeSet, HashMap};
use crate::{E, Expr, symbol};

/// What a fused [`Intermediate::Match`] switches on.
#[derive(Debug, Clone, PartialEq)]
pub enum Switch {
    /// A `select` index: `match (index).select_index() { k => arms[k], _ => default }`.
    Index(E),
    /// A `branch` condition: `if q >= 0 { arms[0] } else { default }`.
    Sign(E),
}

impl Switch {
    /// The expression switched on.
    pub fn expr(&self) -> &E {
        match self {
            Switch::Index(e) | Switch::Sign(e) => e,
        }
    }

    fn expr_mut(&mut self) -> &mut E {
        match self {
            Switch::Index(e) | Switch::Sign(e) => e,
        }
    }
}

/// One statement of a CSE result, in emission order.
#[derive(Debug, Clone, PartialEq)]
pub enum Intermediate {
    /// `let name = expr;`
    Let { name: String, expr: E },
    /// One switch shared by every select or branch on the same key in
    /// the scope: `let (names, ..) = match index { k => arms[k], _ =>
    /// default };` or `let (names, ..) = if q >= 0 { arms[0] } else {
    /// default };`. Each arm carries its own intermediates, so work used
    /// by one arm only is computed inside that arm. A select without a
    /// default panics on any index outside `0..arms.len()`; a branch
    /// always has its else side as the default.
    Match { switch: Switch, names: Vec<String>, arms: Vec<Arm>, default: Option<Arm> },
}

/// One arm of an [`Intermediate::Match`]: local intermediates, then one
/// value per name of the match.
#[derive(Debug, Clone, PartialEq)]
pub struct Arm {
    pub inters: Vec<Intermediate>,
    pub values: Vec<E>,
}

impl Intermediate {
    /// The `(name, expr)` of a plain `let`; `None` for a match.
    pub fn as_let(&self) -> Option<(&str, &E)> {
        match self {
            Intermediate::Let { name, expr } => Some((name, expr)),
            Intermediate::Match { .. } => None,
        }
    }

    /// The names this statement defines.
    pub fn names(&self) -> Vec<&str> {
        match self {
            Intermediate::Let { name, .. } => vec![name],
            Intermediate::Match { names, .. } => names.iter().map(String::as_str).collect(),
        }
    }

    /// Free variables read by this statement, nested arms included.
    fn free_vars(&self) -> BTreeSet<String> {
        match self {
            Intermediate::Let { expr, .. } => expr.free_vars(),
            Intermediate::Match { switch, arms, default, .. } => {
                let mut set = switch.expr().free_vars();
                for arm in arms.iter().chain(default.iter()) {
                    for i in &arm.inters { set.extend(i.free_vars()); }
                    for v in &arm.values { set.extend(v.free_vars()); }
                }
                set
            }
        }
    }
}

/// Cost of evaluating an expression (number of operations).
fn expr_cost(e: &E) -> usize {
    match e.as_ref() {
        Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => 0,
        Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
        | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
        | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
        | Expr::Exp(a) | Expr::Ln(a) | Expr::Log2(a) | Expr::Log10(a)
        | Expr::Sqrt(a) | Expr::Abs(a)
        | Expr::Heaviside(a) => 1 + expr_cost(a),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
        | Expr::Div(a, b) | Expr::Pow(a, b) | Expr::Atan2(a, b) => {
            1 + expr_cost(a) + expr_cost(b)
        }
        Expr::Clamp(a, b, c) | Expr::Branch(a, b, c) => 1 + expr_cost(a) + expr_cost(b) + expr_cost(c),
        // One arm runs: the index plus the most expensive arm.
        Expr::Select { index, arms, default } => {
            1 + expr_cost(index)
                + arms.iter().chain(default.iter()).map(expr_cost).max().unwrap_or(0)
        }
        Expr::Func { args, .. } => {
            1 + args.iter().map(expr_cost).sum::<usize>()
        }
    }
}

/// Depth of an expression tree.
fn expr_depth(e: &E) -> usize {
    match e.as_ref() {
        Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => 0,
        Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
        | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
        | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
        | Expr::Exp(a) | Expr::Ln(a) | Expr::Log2(a) | Expr::Log10(a)
        | Expr::Sqrt(a) | Expr::Abs(a)
        | Expr::Heaviside(a) => 1 + expr_depth(a),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
        | Expr::Div(a, b) | Expr::Pow(a, b) | Expr::Atan2(a, b) => {
            1 + expr_depth(a).max(expr_depth(b))
        }
        Expr::Clamp(a, b, c) | Expr::Branch(a, b, c) => 1 + expr_depth(a).max(expr_depth(b)).max(expr_depth(c)),
        Expr::Select { index, arms, default } => {
            1 + expr_depth(index)
                .max(arms.iter().chain(default.iter()).map(expr_depth).max().unwrap_or(0))
        }
        Expr::Func { args, .. } => {
            1 + args.iter().map(expr_depth).max().unwrap_or(0)
        }
    }
}

/// Every direct child of a node. `Func` children are its arguments (the
/// body is inlined at emission); a `Select`'s are its index, arms and
/// default.
fn children(e: &E) -> Vec<&E> {
    match e.as_ref() {
        Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => vec![],
        Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
        | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
        | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
        | Expr::Exp(a) | Expr::Ln(a) | Expr::Log2(a) | Expr::Log10(a)
        | Expr::Sqrt(a) | Expr::Abs(a)
        | Expr::Heaviside(a) => vec![a],
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
        | Expr::Div(a, b) | Expr::Pow(a, b) | Expr::Atan2(a, b) => vec![a, b],
        Expr::Clamp(a, b, c) | Expr::Branch(a, b, c) => vec![a, b, c],
        Expr::Select { index, arms, default } => {
            let mut v = vec![index];
            v.extend(arms.iter());
            v.extend(default.iter());
            v
        }
        Expr::Func { args, .. } => args.iter().collect(),
    }
}

/// Rebuild a node with every direct child mapped through `f`.
fn map_children(e: &E, f: &mut dyn FnMut(&E) -> E) -> E {
    E::new(match e.as_ref() {
        Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => return e.clone(),
        Expr::Neg(a) => Expr::Neg(f(a)),
        Expr::Sin(a) => Expr::Sin(f(a)),
        Expr::Cos(a) => Expr::Cos(f(a)),
        Expr::Tan(a) => Expr::Tan(f(a)),
        Expr::Asin(a) => Expr::Asin(f(a)),
        Expr::Acos(a) => Expr::Acos(f(a)),
        Expr::Atan(a) => Expr::Atan(f(a)),
        Expr::Sinh(a) => Expr::Sinh(f(a)),
        Expr::Cosh(a) => Expr::Cosh(f(a)),
        Expr::Tanh(a) => Expr::Tanh(f(a)),
        Expr::Exp(a) => Expr::Exp(f(a)),
        Expr::Ln(a) => Expr::Ln(f(a)),
        Expr::Log2(a) => Expr::Log2(f(a)),
        Expr::Log10(a) => Expr::Log10(f(a)),
        Expr::Sqrt(a) => Expr::Sqrt(f(a)),
        Expr::Abs(a) => Expr::Abs(f(a)),
        Expr::Heaviside(a) => Expr::Heaviside(f(a)),
        Expr::Add(a, b) => Expr::Add(f(a), f(b)),
        Expr::Sub(a, b) => Expr::Sub(f(a), f(b)),
        Expr::Mul(a, b) => Expr::Mul(f(a), f(b)),
        Expr::Div(a, b) => Expr::Div(f(a), f(b)),
        Expr::Pow(a, b) => Expr::Pow(f(a), f(b)),
        Expr::Atan2(a, b) => Expr::Atan2(f(a), f(b)),
        Expr::Clamp(a, b, c) => Expr::Clamp(f(a), f(b), f(c)),
        Expr::Branch(a, b, c) => Expr::Branch(f(a), f(b), f(c)),
        Expr::Select { index, arms, default } => Expr::Select {
            index: f(index),
            arms: arms.iter().map(|a| f(a)).collect(),
            default: default.as_ref().map(|d| f(d)),
        },
        Expr::Func { name, params, kind, args } => Expr::Func {
            name: name.clone(), params: params.clone(), kind: kind.clone(),
            args: args.iter().map(|a| f(a)).collect(),
        },
    })
}

/// Walk expression tree, count occurrences of each subexpression. When
/// `scoped`, nodes under a select arm count in `inside`, the rest in
/// `outside`: an expression seen only inside arms is never hoisted above
/// the match. Unscoped, everything counts as outside.
fn count_subexprs(e: &E, in_arm: bool, scoped: bool, outside: &mut HashMap<E, usize>, inside: &mut HashMap<E, usize>) {
    let counts: &mut HashMap<E, usize> = if in_arm { &mut *inside } else { &mut *outside };
    *counts.entry(e.clone()).or_insert(0) += 1;
    match e.as_ref() {
        Expr::Select { index, arms, default } => {
            count_subexprs(index, in_arm, scoped, outside, inside);
            for a in arms.iter().chain(default.iter()) {
                count_subexprs(a, in_arm || scoped, scoped, outside, inside);
            }
        }
        Expr::Branch(q, a, b) => {
            count_subexprs(q, in_arm, scoped, outside, inside);
            count_subexprs(a, in_arm || scoped, scoped, outside, inside);
            count_subexprs(b, in_arm || scoped, scoped, outside, inside);
        }
        _ => for c in children(e) { count_subexprs(c, in_arm, scoped, outside, inside); },
    }
}

/// Replace all occurrences of a sub-expression with another in the given
/// expression.
///
/// Performs a structural walk of the expression tree, replacing every node
/// that is equal to `target` with `replacement`. For product targets, also
/// detects when the target's factors are a subset of a larger product.
pub fn replace_pub(e: &E, target: &E, replacement: &E) -> E {
    replace(e, target, replacement)
}

fn replace(e: &E, target: &E, replacement: &E) -> E {
    if e == target {
        return replacement.clone();
    }
    match e.as_ref() {
        Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => e.clone(),
        Expr::Neg(a) => E::new(Expr::Neg(replace(a, target, replacement))),
        Expr::Add(a, b) => E::new(Expr::Add(replace(a, target, replacement), replace(b, target, replacement))),
        Expr::Sub(a, b) => E::new(Expr::Sub(replace(a, target, replacement), replace(b, target, replacement))),
        Expr::Mul(_, _) => {
            // Factor-aware replacement: if target is a product and its factors
            // are a subset of this product's factors, replace them.
            if matches!(target.as_ref(), Expr::Mul(_, _)) {
                let (e_coeff, e_factors) = flatten_mul_factors(e);
                let (t_coeff, t_factors) = flatten_mul_factors(target);
                if t_coeff == 1.0 && t_factors.len() <= e_factors.len() {
                    // Check if all target factors appear in e's factors
                    let mut remaining = e_factors.clone();
                    let mut all_found = true;
                    for tf in &t_factors {
                        if let Some(pos) = remaining.iter().position(|f| f == tf) {
                            remaining.remove(pos);
                        } else {
                            all_found = false;
                            break;
                        }
                    }
                    if all_found {
                        // Replace target factors with replacement, keep remaining
                        remaining.push(replacement.clone());
                        // Recurse on remaining factors in case of nested matches
                        let result = build_mul_from_factors(e_coeff, remaining);
                        return replace(&result, target, replacement);
                    }
                }
            }
            // Default: recurse into children
            let (a, b) = match e.as_ref() {
                Expr::Mul(a, b) => (a, b),
                _ => unreachable!(),
            };
            E::new(Expr::Mul(replace(a, target, replacement), replace(b, target, replacement)))
        }
        Expr::Div(a, b) => E::new(Expr::Div(replace(a, target, replacement), replace(b, target, replacement))),
        Expr::Pow(a, b) => E::new(Expr::Pow(replace(a, target, replacement), replace(b, target, replacement))),
        Expr::Atan2(a, b) => E::new(Expr::Atan2(replace(a, target, replacement), replace(b, target, replacement))),
        Expr::Sin(a) => E::new(Expr::Sin(replace(a, target, replacement))),
        Expr::Cos(a) => E::new(Expr::Cos(replace(a, target, replacement))),
        Expr::Tan(a) => E::new(Expr::Tan(replace(a, target, replacement))),
        Expr::Asin(a) => E::new(Expr::Asin(replace(a, target, replacement))),
        Expr::Acos(a) => E::new(Expr::Acos(replace(a, target, replacement))),
        Expr::Atan(a) => E::new(Expr::Atan(replace(a, target, replacement))),
        Expr::Sinh(a) => E::new(Expr::Sinh(replace(a, target, replacement))),
        Expr::Cosh(a) => E::new(Expr::Cosh(replace(a, target, replacement))),
        Expr::Tanh(a) => E::new(Expr::Tanh(replace(a, target, replacement))),
        Expr::Exp(a) => E::new(Expr::Exp(replace(a, target, replacement))),
        Expr::Ln(a) => E::new(Expr::Ln(replace(a, target, replacement))),
        Expr::Log2(a) => E::new(Expr::Log2(replace(a, target, replacement))),
        Expr::Log10(a) => E::new(Expr::Log10(replace(a, target, replacement))),
        Expr::Sqrt(a) => E::new(Expr::Sqrt(replace(a, target, replacement))),
        Expr::Abs(a) => E::new(Expr::Abs(replace(a, target, replacement))),
        Expr::Heaviside(a) => E::new(Expr::Heaviside(replace(a, target, replacement))),
        Expr::Clamp(a, b, c) => E::new(Expr::Clamp(replace(a, target, replacement), replace(b, target, replacement), replace(c, target, replacement))),
        Expr::Branch(a, b, c) => E::new(Expr::Branch(replace(a, target, replacement), replace(b, target, replacement), replace(c, target, replacement))),
        Expr::Select { .. } => map_children(e, &mut |c| replace(c, target, replacement)),
        Expr::Func { name, params, kind, args } => {
            let new_args = args.iter().map(|a| replace(a, target, replacement)).collect();
            E::new(Expr::Func { name: name.clone(), params: params.clone(), kind: kind.clone(), args: new_args })
        }
    }
}

/// Like [`replace`] for a select target, but never descends into select
/// arms: a select nested in another select's arm belongs to that arm's
/// scope and is fused there.
fn replace_outside_arms(e: &E, target: &E, replacement: &E) -> E {
    if e == target {
        return replacement.clone();
    }
    match e.as_ref() {
        Expr::Select { index, arms, default } => E::new(Expr::Select {
            index: replace_outside_arms(index, target, replacement),
            arms: arms.clone(),
            default: default.clone(),
        }),
        Expr::Branch(q, a, b) => E::new(Expr::Branch(
            replace_outside_arms(q, target, replacement), a.clone(), b.clone())),
        _ => map_children(e, &mut |c| replace_outside_arms(c, target, replacement)),
    }
}

/// Every distinct select or branch node at this scope level, in
/// first-appearance order. Switches inside another switch's arms are
/// that arm's business.
fn collect_switches(e: &E, out: &mut Vec<E>) {
    let key = match e.as_ref() {
        Expr::Select { index, .. } => index,
        Expr::Branch(q, _, _) => q,
        _ => {
            for c in children(e) { collect_switches(c, out); }
            return;
        }
    };
    if !out.contains(e) {
        out.push(e.clone());
    }
    collect_switches(key, out);
}

/// Apply many `target -> replacement` substitutions to `e` in one memoized
/// traversal. A node structurally equal to a target is replaced wholesale;
/// other nodes are rebuilt around their substituted children. A per-node
/// pointer memo makes each shared subtree cost once, so this does the work of N
/// separate [`replace_pub`] passes in a single walk.
///
/// `subs` is an ordered list: on a duplicate target the first pair wins, as if
/// the substitutions were applied in sequence. Targets match by exact
/// structural equality -- a `Mul` target needing factor-subset matching must go
/// through [`replace_pub`].
pub fn replace_many(e: &E, subs: &[(E, E)]) -> E {
    if subs.is_empty() {
        return e.clone();
    }
    let mut map: HashMap<&E, &E> = HashMap::with_capacity(subs.len());
    for (from, to) in subs {
        // or_insert, not insert: a repeated target keeps its FIRST replacement,
        // so the result matches applying `subs` in order. Rotation identities
        // make some targets structurally equal (e.g. dR[i][j]/d.. equals a plain
        // R[k][l]), and both replacement fields carry the same runtime value.
        map.entry(from).or_insert(to);
    }
    let mut memo: HashMap<*const Expr, E> = HashMap::new();
    replace_many_inner(e, &map, &mut memo)
}

fn replace_many_inner(e: &E, map: &HashMap<&E, &E>, memo: &mut HashMap<*const Expr, E>) -> E {
    let ptr = e.as_ref() as *const Expr;
    if let Some(r) = memo.get(&ptr) {
        return r.clone();
    }
    let rec = replace_many_inner;
    // A whole-node match takes precedence over descending into it.
    let result = if let Some(to) = map.get(e) {
        (*to).clone()
    } else {
        match e.as_ref() {
            Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => e.clone(),
            Expr::Neg(a) => E::new(Expr::Neg(rec(a, map, memo))),
            Expr::Sin(a) => E::new(Expr::Sin(rec(a, map, memo))),
            Expr::Cos(a) => E::new(Expr::Cos(rec(a, map, memo))),
            Expr::Tan(a) => E::new(Expr::Tan(rec(a, map, memo))),
            Expr::Asin(a) => E::new(Expr::Asin(rec(a, map, memo))),
            Expr::Acos(a) => E::new(Expr::Acos(rec(a, map, memo))),
            Expr::Atan(a) => E::new(Expr::Atan(rec(a, map, memo))),
            Expr::Sinh(a) => E::new(Expr::Sinh(rec(a, map, memo))),
            Expr::Cosh(a) => E::new(Expr::Cosh(rec(a, map, memo))),
            Expr::Tanh(a) => E::new(Expr::Tanh(rec(a, map, memo))),
            Expr::Exp(a) => E::new(Expr::Exp(rec(a, map, memo))),
            Expr::Ln(a) => E::new(Expr::Ln(rec(a, map, memo))),
            Expr::Log2(a) => E::new(Expr::Log2(rec(a, map, memo))),
            Expr::Log10(a) => E::new(Expr::Log10(rec(a, map, memo))),
            Expr::Sqrt(a) => E::new(Expr::Sqrt(rec(a, map, memo))),
            Expr::Abs(a) => E::new(Expr::Abs(rec(a, map, memo))),
            Expr::Heaviside(a) => E::new(Expr::Heaviside(rec(a, map, memo))),
            Expr::Add(a, b) => E::new(Expr::Add(rec(a, map, memo), rec(b, map, memo))),
            Expr::Sub(a, b) => E::new(Expr::Sub(rec(a, map, memo), rec(b, map, memo))),
            Expr::Mul(a, b) => E::new(Expr::Mul(rec(a, map, memo), rec(b, map, memo))),
            Expr::Div(a, b) => E::new(Expr::Div(rec(a, map, memo), rec(b, map, memo))),
            Expr::Pow(a, b) => E::new(Expr::Pow(rec(a, map, memo), rec(b, map, memo))),
            Expr::Atan2(a, b) => E::new(Expr::Atan2(rec(a, map, memo), rec(b, map, memo))),
            Expr::Clamp(a, b, c) => E::new(Expr::Clamp(rec(a, map, memo), rec(b, map, memo), rec(c, map, memo))),
            Expr::Branch(a, b, c) => E::new(Expr::Branch(rec(a, map, memo), rec(b, map, memo), rec(c, map, memo))),
            Expr::Select { .. } => map_children(e, &mut |c| rec(c, map, memo)),
            Expr::Func { name, params, kind, args } => {
                let new_args = args.iter().map(|a| rec(a, map, memo)).collect();
                E::new(Expr::Func { name: name.clone(), params: params.clone(), kind: kind.clone(), args: new_args })
            }
        }
    };
    memo.insert(ptr, result.clone());
    result
}

/// Flatten a Mul tree into coefficient + list of non-constant factors.
fn flatten_mul_factors(e: &E) -> (f64, Vec<E>) {
    match e.as_ref() {
        Expr::Mul(a, b) => {
            let (ca, mut fa) = flatten_mul_factors(a);
            let (cb, fb) = flatten_mul_factors(b);
            fa.extend(fb);
            (ca * cb, fa)
        }
        Expr::Neg(a) => {
            let (c, f) = flatten_mul_factors(a);
            (-c, f)
        }
        Expr::Const(v) => (*v, vec![]),
        _ => (1.0, vec![e.clone()]),
    }
}

/// Build a Mul expression from coefficient and factors.
fn build_mul_from_factors(coeff: f64, factors: Vec<E>) -> E {
    if factors.is_empty() {
        return E::new(Expr::Const(coeff));
    }
    let mut iter = factors.into_iter();
    let mut result = iter.next().unwrap();
    for f in iter {
        result = E::new(Expr::Mul(result, f));
    }
    if coeff == 1.0 {
        result
    } else if coeff == -1.0 {
        E::new(Expr::Neg(result))
    } else {
        E::new(Expr::Mul(E::new(Expr::Const(coeff)), result))
    }
}

/// Common Subexpression Elimination.
///
/// Iteratively extracts repeated subexpressions, deepest first. Each
/// iteration: count all subexprs, pick the best candidate (deepest with
/// count >= 2), replace it everywhere, repeat until no more candidates.
/// Returns the named intermediates, in dependency order, and the
/// rewritten batch.
///
/// This is the flat form: every intermediate is a `let`, and a select's
/// arms count like the rest of the batch (as `branch` sides do), so work
/// shared by arms is hoisted above the select and the select itself is
/// rendered inline by `to_rust`. [`cse_scoped`] keeps each arm's work
/// inside the arm instead.
pub fn cse(exprs: &[E]) -> (Vec<(String, E)>, Vec<E>) {
    let mut counter = 0usize;
    let (inters, results) = cse_scope(exprs, &mut counter, false);
    let lets = inters.into_iter().map(|it| match it {
        Intermediate::Let { name, expr } => (name, expr),
        Intermediate::Match { .. } => unreachable!("flat cse never fuses"),
    }).collect();
    (lets, results)
}

/// [`cse`] with select arms as scopes.
///
/// A subexpression used only inside select arms is never hoisted above
/// the match; one used outside as well is hoisted once and reused inside.
/// Every select on one index across the batch fuses into a single
/// [`Intermediate::Match`] yielding one value per select, and CSE runs
/// again inside each arm over that arm's values, so nested selects fuse
/// inside their arm the same way. Names are unique across the whole
/// result, nested scopes included. Returns the statements to emit, in
/// dependency order, and the rewritten batch. For a batch without selects
/// the result is [`cse`]'s, as `Let` statements.
pub fn cse_scoped(exprs: &[E]) -> (Vec<Intermediate>, Vec<E>) {
    let mut counter = 0usize;
    cse_scope(exprs, &mut counter, true)
}

fn cse_scope(exprs: &[E], counter: &mut usize, scoped: bool) -> (Vec<Intermediate>, Vec<E>) {
    if exprs.is_empty() {
        return (vec![], vec![]);
    }

    let mut results = exprs.to_vec();
    let mut lets: Vec<(String, E)> = Vec::new();

    loop {
        // Count subexpressions across results AND intermediate definitions
        let mut outside: HashMap<E, usize> = HashMap::new();
        let mut inside: HashMap<E, usize> = HashMap::new();
        for r in &results {
            count_subexprs(r, false, scoped, &mut outside, &mut inside);
        }
        for (_, expr) in &lets {
            count_subexprs(expr, false, scoped, &mut outside, &mut inside);
        }

        // Find the best candidate: used >= 2 times, cost >= 1, and at
        // least one symbol. Only expressions seen outside a select arm
        // are candidates; uses inside arms add to the savings (the value
        // exists above the match, so reusing it there is free) but never
        // make a candidate on their own. Symbol-free (constant)
        // subexpressions are never extracted: they cost nothing at
        // runtime (the compiler folds them), and hoisting one into its
        // own `let` strips the type context that unsuffixed literals in
        // generated code rely on (e.g. `let __x = 2.2e-16.powf(2.0);` is
        // an ambiguous numeric type, while the same expression inline
        // infers from its surroundings).
        // Rank by savings = (uses - 1) * cost -- how many ops we save.
        // The display string as the final tie-break makes the choice a
        // total order: HashMap iteration order is randomized per
        // instance, and max_by_key keeps the LAST maximum it sees, so
        // without it two identical builds pick different candidates and
        // emit differently-named/ordered temporaries (nondeterministic
        // generated code).
        let best = outside.into_iter()
            .map(|(e, o)| {
                let uses = o + inside.get(&e).copied().unwrap_or(0);
                (e, uses)
            })
            .filter(|(e, uses)| *uses >= 2 && expr_cost(e) >= 1 && !e.symbols().is_empty())
            .max_by_key(|(e, uses)| {
                let cost = expr_cost(e);
                let savings = (*uses - 1) * cost;
                // Primary: most savings
                // Secondary: prefer deeper (to enable further extraction)
                // Tertiary: display string, for determinism
                (savings, expr_depth(e), format!("{}", e))
            });

        let (subexpr, _uses) = match best {
            Some(b) => b,
            None => break,
        };

        let var_name = format!("__x{}", *counter);
        *counter += 1;
        let var_sym = symbol(&var_name);

        // Replace in all results
        for r in results.iter_mut() {
            *r = replace(r, &subexpr, &var_sym);
        }

        // Replace in existing intermediates' definitions too
        for (_, expr) in lets.iter_mut() {
            *expr = replace(expr, &subexpr, &var_sym);
        }

        lets.push((var_name, subexpr));
    }

    // Post-pass: extract common divisors as reciprocals.
    // If `/ x` appears 2+ times, extract `__xN = 1.0 / x` and replace
    // `a / x` with `a * __xN`. Same scope rule as above: a divisor seen
    // only inside arms is left to the arm's own pass.
    let mut outside: HashMap<E, usize> = HashMap::new();
    let mut inside: HashMap<E, usize> = HashMap::new();
    for r in &results {
        count_divisors(r, false, scoped, &mut outside, &mut inside);
    }
    for (_, expr) in &lets {
        count_divisors(expr, false, scoped, &mut outside, &mut inside);
    }
    // Sort for determinism: HashMap iteration order would name and
    // order the reciprocal temporaries randomly across builds.
    let mut divisors: Vec<(E, usize)> = outside.into_iter()
        .map(|(e, o)| {
            let uses = o + inside.get(&e).copied().unwrap_or(0);
            (e, uses)
        })
        .collect();
    divisors.sort_by_key(|(e, _)| format!("{}", e));
    for (divisor, uses) in divisors {
        if uses >= 2 {
            let var_name = format!("__x{}", *counter);
            *counter += 1;
            let var_sym = symbol(&var_name);
            let recip = E::new(Expr::Div(E::new(Expr::Const(1.0)), divisor.clone()));
            for r in results.iter_mut() {
                *r = replace_divisor(r, &divisor, &var_sym);
            }
            for (_, expr) in lets.iter_mut() {
                *expr = replace_divisor(expr, &divisor, &var_sym);
            }
            lets.push((var_name, recip));
        }
    }

    let mut intermediates: Vec<Intermediate> = lets.into_iter()
        .map(|(name, expr)| Intermediate::Let { name, expr })
        .collect();

    // The flat form leaves selects in place for inline emission.
    if !scoped {
        return (topo_sort_intermediates(intermediates), results);
    }

    // Fuse: every select at this scope level becomes part of one match
    // per (index, arm count, default presence), in first-appearance
    // order. Each member select is replaced by the name its match binds.
    let mut switches: Vec<E> = Vec::new();
    for r in &results {
        collect_switches(r, &mut switches);
    }
    for it in &intermediates {
        if let Intermediate::Let { expr, .. } = it {
            collect_switches(expr, &mut switches);
        }
    }
    // (switch, arm count, default present, member nodes)
    let mut groups: Vec<(Switch, usize, bool, Vec<E>)> = Vec::new();
    for s in switches {
        let (switch, n_arms, has_default) = match s.as_ref() {
            Expr::Select { index, arms, default } =>
                (Switch::Index(index.clone()), arms.len(), default.is_some()),
            Expr::Branch(q, _, _) => (Switch::Sign(q.clone()), 1, true),
            _ => unreachable!(),
        };
        match groups.iter_mut()
            .find(|(k, n, d, _)| *k == switch && *n == n_arms && *d == has_default) {
            Some(g) => g.3.push(s.clone()),
            None => groups.push((switch, n_arms, has_default, vec![s.clone()])),
        }
    }
    for (switch, n_arms, has_default, members) in groups {
        let names: Vec<String> = members.iter().map(|_| {
            let n = format!("__m{}", *counter);
            *counter += 1;
            n
        }).collect();
        for (m, name) in members.iter().zip(&names) {
            let sym = symbol(name);
            for r in results.iter_mut() {
                *r = replace_outside_arms(r, m, &sym);
            }
            for it in intermediates.iter_mut() {
                match it {
                    Intermediate::Let { expr, .. } => *expr = replace_outside_arms(expr, m, &sym),
                    // An earlier switch's key may hold this node; its
                    // arms are a scope of their own and are done already.
                    Intermediate::Match { switch: sw, .. } => {
                        let key = sw.expr_mut();
                        *key = replace_outside_arms(key, m, &sym);
                    }
                }
            }
        }
        let parts = |m: &E| -> (Vec<E>, Option<E>) {
            match m.as_ref() {
                Expr::Select { arms, default, .. } => (arms.clone(), default.clone()),
                Expr::Branch(_, a, b) => (vec![a.clone()], Some(b.clone())),
                _ => unreachable!(),
            }
        };
        let arms: Vec<Arm> = (0..n_arms).map(|k| {
            let exprs: Vec<E> = members.iter().map(|m| parts(m).0[k].clone()).collect();
            let (inters, values) = cse_scope(&exprs, counter, true);
            Arm { inters, values }
        }).collect();
        let default = has_default.then(|| {
            let exprs: Vec<E> = members.iter().map(|m| parts(m).1.unwrap()).collect();
            let (inters, values) = cse_scope(&exprs, counter, true);
            Arm { inters, values }
        });
        intermediates.push(Intermediate::Match { switch, names, arms, default });
    }

    // Topological sort: ensure each intermediate is defined before it's used.
    let intermediates = topo_sort_intermediates(intermediates);
    (intermediates, results)
}

/// Count how many times each expression appears as a divisor (right side
/// of Div), split by whether the division sits inside a select arm when
/// `scoped` (see [`count_subexprs`]).
fn count_divisors(e: &E, in_arm: bool, scoped: bool, outside: &mut HashMap<E, usize>, inside: &mut HashMap<E, usize>) {
    match e.as_ref() {
        Expr::Div(a, b) => {
            let counts: &mut HashMap<E, usize> = if in_arm { &mut *inside } else { &mut *outside };
            *counts.entry(b.clone()).or_insert(0) += 1;
            count_divisors(a, in_arm, scoped, outside, inside);
            count_divisors(b, in_arm, scoped, outside, inside);
        }
        Expr::Select { index, arms, default } => {
            count_divisors(index, in_arm, scoped, outside, inside);
            for a in arms.iter().chain(default.iter()) {
                count_divisors(a, in_arm || scoped, scoped, outside, inside);
            }
        }
        Expr::Branch(q, a, b) => {
            count_divisors(q, in_arm, scoped, outside, inside);
            count_divisors(a, in_arm || scoped, scoped, outside, inside);
            count_divisors(b, in_arm || scoped, scoped, outside, inside);
        }
        _ => for c in children(e) { count_divisors(c, in_arm, scoped, outside, inside); },
    }
}

/// Replace `a / divisor` with `a * replacement` in expression `e`.
fn replace_divisor(e: &E, divisor: &E, replacement: &E) -> E {
    match e.as_ref() {
        Expr::Div(a, b) if b == divisor => {
            let a2 = replace_divisor(a, divisor, replacement);
            E::new(Expr::Mul(a2, replacement.clone()))
        }
        Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => e.clone(),
        Expr::Neg(a) => E::new(Expr::Neg(replace_divisor(a, divisor, replacement))),
        Expr::Add(a, b) => E::new(Expr::Add(replace_divisor(a, divisor, replacement), replace_divisor(b, divisor, replacement))),
        Expr::Sub(a, b) => E::new(Expr::Sub(replace_divisor(a, divisor, replacement), replace_divisor(b, divisor, replacement))),
        Expr::Mul(a, b) => E::new(Expr::Mul(replace_divisor(a, divisor, replacement), replace_divisor(b, divisor, replacement))),
        Expr::Div(a, b) => E::new(Expr::Div(replace_divisor(a, divisor, replacement), replace_divisor(b, divisor, replacement))),
        Expr::Pow(a, b) => E::new(Expr::Pow(replace_divisor(a, divisor, replacement), replace_divisor(b, divisor, replacement))),
        Expr::Atan2(a, b) => E::new(Expr::Atan2(replace_divisor(a, divisor, replacement), replace_divisor(b, divisor, replacement))),
        Expr::Sin(a) => E::new(Expr::Sin(replace_divisor(a, divisor, replacement))),
        Expr::Cos(a) => E::new(Expr::Cos(replace_divisor(a, divisor, replacement))),
        Expr::Tan(a) => E::new(Expr::Tan(replace_divisor(a, divisor, replacement))),
        Expr::Asin(a) => E::new(Expr::Asin(replace_divisor(a, divisor, replacement))),
        Expr::Acos(a) => E::new(Expr::Acos(replace_divisor(a, divisor, replacement))),
        Expr::Atan(a) => E::new(Expr::Atan(replace_divisor(a, divisor, replacement))),
        Expr::Sinh(a) => E::new(Expr::Sinh(replace_divisor(a, divisor, replacement))),
        Expr::Cosh(a) => E::new(Expr::Cosh(replace_divisor(a, divisor, replacement))),
        Expr::Tanh(a) => E::new(Expr::Tanh(replace_divisor(a, divisor, replacement))),
        Expr::Exp(a) => E::new(Expr::Exp(replace_divisor(a, divisor, replacement))),
        Expr::Ln(a) => E::new(Expr::Ln(replace_divisor(a, divisor, replacement))),
        Expr::Log2(a) => E::new(Expr::Log2(replace_divisor(a, divisor, replacement))),
        Expr::Log10(a) => E::new(Expr::Log10(replace_divisor(a, divisor, replacement))),
        Expr::Sqrt(a) => E::new(Expr::Sqrt(replace_divisor(a, divisor, replacement))),
        Expr::Abs(a) => E::new(Expr::Abs(replace_divisor(a, divisor, replacement))),
        Expr::Heaviside(a) => E::new(Expr::Heaviside(replace_divisor(a, divisor, replacement))),
        Expr::Clamp(a, b, c) => E::new(Expr::Clamp(replace_divisor(a, divisor, replacement), replace_divisor(b, divisor, replacement), replace_divisor(c, divisor, replacement))),
        Expr::Branch(a, b, c) => E::new(Expr::Branch(replace_divisor(a, divisor, replacement), replace_divisor(b, divisor, replacement), replace_divisor(c, divisor, replacement))),
        Expr::Select { .. } => map_children(e, &mut |c| replace_divisor(c, divisor, replacement)),
        Expr::Func { name, params, kind, args } => {
            let new_args = args.iter().map(|a| replace_divisor(a, divisor, replacement)).collect();
            E::new(Expr::Func { name: name.clone(), params: params.clone(), kind: kind.clone(), args: new_args })
        }
    }
}

/// Topological sort of intermediates so dependencies come first.
fn topo_sort_intermediates(intermediates: Vec<Intermediate>) -> Vec<Intermediate> {
    use std::collections::HashSet;

    let names: HashSet<String> = intermediates.iter()
        .flat_map(|it| it.names().into_iter().map(str::to_string)).collect();

    // Build dependency graph: for each intermediate, which other intermediates
    // does it reference? Sorted: HashSet iteration order would randomize the
    // dependents lists and with them the emitted definition order.
    let deps: Vec<Vec<String>> = intermediates.iter().map(|it| {
        let vars = it.free_vars();
        let mut d: Vec<String> = vars.into_iter().filter(|v| names.contains(v)).collect();
        d.sort();
        d
    }).collect();

    // Kahn's algorithm
    let n = intermediates.len();
    let name_to_idx: HashMap<String, usize> = intermediates.iter().enumerate()
        .flat_map(|(i, it)| it.names().into_iter().map(move |n| (n.to_string(), i))).collect();

    let mut in_degree = vec![0usize; n];
    let mut dependents: Vec<Vec<usize>> = vec![vec![]; n];
    for (i, dep_set) in deps.iter().enumerate() {
        for dep_name in dep_set {
            if let Some(&j) = name_to_idx.get(dep_name) {
                in_degree[i] += 1;
                dependents[j].push(i);
            }
        }
    }

    let mut queue: Vec<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut sorted = Vec::with_capacity(n);

    while let Some(idx) = queue.pop() {
        sorted.push(idx);
        for &dep in &dependents[idx] {
            in_degree[dep] -= 1;
            if in_degree[dep] == 0 {
                queue.push(dep);
            }
        }
    }

    // A cycle would make the queue dry up early and the tail of the
    // definitions silently vanish from the generated code.
    assert_eq!(sorted.len(), n,
        "CSE topological sort dropped definitions: dependency cycle among intermediates");

    sorted.into_iter().map(|i| intermediates[i].clone()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{symbol, cached};

    #[test]
    fn replace_many_agrees_with_sequential() {
        let (x, y, z) = (symbol("x"), symbol("y"), symbol("z"));
        let (a, b) = (symbol("a"), symbol("b"));
        // A shared cached() subtree plus a bare symbol, both substitution targets.
        let cxy = cached(x.clone() * y.clone());
        let expr = cxy.clone() * z.clone() + cxy.clone() / z.clone();

        let subs = [(cxy.clone(), a.clone()), (z.clone(), b.clone())];
        let got = replace_many(&expr, &subs);

        // Sequential reference: the two replace_pub passes the old code did.
        let seq = replace_pub(&replace_pub(&expr, &cxy, &a), &z, &b);
        assert_eq!(got, seq);
        assert!(!format!("{}", got).contains("cached"));
    }

    #[test]
    fn replace_many_whole_node_match_does_not_descend() {
        // cached(x) matches as a whole; the inner x is not separately rewritten.
        let x = symbol("x");
        let cx = cached(x.clone());
        let subs = [(cx.clone(), symbol("field")), (x.clone(), symbol("wrong"))];
        assert_eq!(replace_many(&cx, &subs), symbol("field"));
    }

    #[test]
    fn replace_many_first_duplicate_target_wins() {
        // A target repeated with different replacements keeps the first, matching
        // sequential application in list order. Rotation identities make some
        // matrix entries and derivative entries structurally equal, so this case
        // is real.
        let x = symbol("x");
        let subs = [(x.clone(), symbol("first")), (x.clone(), symbol("second"))];
        assert_eq!(replace_many(&x, &subs), symbol("first"));
    }

    #[test]
    fn replace_many_empty_is_identity() {
        let e = symbol("x") + symbol("y");
        assert_eq!(replace_many(&e, &[]), e);
    }

    // --- select arms as scopes, selects fused per index ---

    use crate::{constant, select, sqrt};

    /// (key expression, names, arms, default) of a fused switch.
    fn the_match(it: &Intermediate) -> (&E, &[String], &[Arm], Option<&Arm>) {
        match it {
            Intermediate::Match { switch, names, arms, default } =>
                (switch.expr(), names, arms, default.as_ref()),
            Intermediate::Let { .. } => panic!("expected a match, got {it}"),
        }
    }

    fn the_switch(it: &Intermediate) -> &Switch {
        match it {
            Intermediate::Match { switch, .. } => switch,
            Intermediate::Let { .. } => panic!("expected a match, got {it}"),
        }
    }

    #[test]
    fn select_arm_only_work_stays_in_its_arm() {
        // rho and rho' both select on k; sqrt(x*y) is shared by their
        // arm 1 only. It must be computed inside that arm, once, and
        // nothing may be hoisted above the match.
        let (k, x, y) = (symbol("k"), symbol("x"), symbol("y"));
        let root = sqrt(x.clone() * y.clone());
        let rho = select(k.clone(), vec![x.clone(), constant(2.0) * root.clone() - y.clone()], None);
        let w = select(k.clone(), vec![constant(1.0), y.clone() / root.clone()], None);
        let (inters, results) = cse_scoped(&[rho, w]);
        assert_eq!(inters.len(), 1, "{inters:?}");
        let (index, names, arms, default) = the_match(&inters[0]);
        assert_eq!(*index, k);
        assert_eq!(names, ["__m0", "__m1"]);
        assert!(default.is_none());
        assert_eq!(results, vec![symbol("__m0"), symbol("__m1")]);
        assert!(arms[0].inters.is_empty());
        assert_eq!(arms[0].values, vec![x.clone(), constant(1.0)]);
        assert_eq!(arms[1].inters.len(), 1);
        assert_eq!(arms[1].inters[0].as_let().unwrap().1, &root);
        let code = inters[0].to_rust("");
        assert!(code.starts_with("let (__m0, __m1) = { let __sel = k.select_index(); match __sel { 0 => (x, 1.0), 1 => { let __x2 = (x * y).sqrt(); "), "{code}");
        assert!(code.ends_with("_ => panic!(\"select index {} out of range 0..2\", __sel) } };"), "{code}");
    }

    #[test]
    fn select_work_also_used_outside_is_hoisted_once() {
        // sqrt(x) is used above the match and inside an arm: hoist it
        // once and reuse it inside.
        let (k, x) = (symbol("k"), symbol("x"));
        let e = sqrt(x.clone()) + select(k.clone(), vec![sqrt(x.clone()), constant(1.0)], None);
        let (inters, results) = cse_scoped(&[e]);
        assert_eq!(inters.len(), 2, "{inters:?}");
        assert_eq!(inters[0].as_let().unwrap(), ("__x0", &sqrt(x.clone())));
        let (_, names, arms, _) = the_match(&inters[1]);
        assert_eq!(names, ["__m1"]);
        assert_eq!(arms[0].values, vec![symbol("__x0")]);
        assert_eq!(format!("{}", results[0]), "__x0 + __m1");
    }

    #[test]
    fn selects_on_different_indices_are_separate_matches() {
        let (k, j, x, y) = (symbol("k"), symbol("j"), symbol("x"), symbol("y"));
        let e = select(k.clone(), vec![x.clone(), y.clone()], None)
            + select(j.clone(), vec![x.clone(), y.clone()], None);
        let (inters, results) = cse_scoped(&[e]);
        assert_eq!(inters.len(), 2);
        let mut indices = vec![the_match(&inters[0]).0.clone(), the_match(&inters[1]).0.clone()];
        indices.sort_by_key(|i| format!("{i}"));
        assert_eq!(indices, vec![j, k]);
        assert_eq!(format!("{}", results[0]), "__m0 + __m1");
    }

    #[test]
    fn nested_select_fuses_inside_its_arm() {
        let (k, j, a, b, c) = (symbol("k"), symbol("j"), symbol("a"), symbol("b"), symbol("c"));
        let inner = select(j.clone(), vec![a.clone() * b.clone(), a.clone() * b.clone() + constant(1.0)], None);
        let e = select(k.clone(), vec![inner, c.clone()], None);
        let (inters, results) = cse_scoped(&[e]);
        assert_eq!(inters.len(), 1);
        let (index, names, arms, _) = the_match(&inters[0]);
        assert_eq!(*index, k);
        assert_eq!(names, ["__m0"]);
        assert_eq!(results, vec![symbol("__m0")]);
        // Arm 0 holds the inner match; arm 1 is the bare value.
        assert_eq!(arms[0].inters.len(), 1);
        let (inner_index, inner_names, inner_arms, _) = the_match(&arms[0].inters[0]);
        assert_eq!(*inner_index, j);
        assert_eq!(inner_names, ["__m1"]);
        assert_eq!(inner_arms[0].values, vec![a.clone() * b.clone()]);
        assert_eq!(arms[0].values, vec![symbol("__m1")]);
        assert!(arms[1].inters.is_empty());
        assert_eq!(arms[1].values, vec![c]);
    }

    #[test]
    fn select_default_arm_is_rendered_not_panic() {
        let (k, x, y) = (symbol("k"), symbol("x"), symbol("y"));
        let (inters, _) = cse_scoped(&[select(k, vec![x], Some(y))]);
        assert_eq!(inters[0].to_rust(""),
            "let __m0 = { let __sel = k.select_index(); match __sel { 0 => x, _ => y } };");
        assert_eq!(format!("{}", inters[0]), "let __m0 = match k { 0 => x, _ => y };");
    }

    #[test]
    fn fused_match_renders_a_tuple() {
        let (k, x, y) = (symbol("k"), symbol("x"), symbol("y"));
        let (inters, results) = cse_scoped(&[
            select(k.clone(), vec![x.clone(), y.clone()], None),
            select(k, vec![y, x], None),
        ]);
        assert_eq!(results, vec![symbol("__m0"), symbol("__m1")]);
        assert_eq!(inters[0].to_rust(""),
            "let (__m0, __m1) = { let __sel = k.select_index(); match __sel { \
             0 => (x, y), 1 => (y, x), _ => panic!(\"select index {} out of range 0..2\", __sel) } };");
        assert_eq!(format!("{}", inters[0]),
            "let (__m0, __m1) = match k { 0 => (x, y), 1 => (y, x), _ => panic };");
    }

    #[test]
    fn select_hoisted_when_repeated_whole() {
        // The same select in two outputs is one hoisted value, and its
        // match defines that value.
        let (k, x, y) = (symbol("k"), symbol("x"), symbol("y"));
        let s = select(k.clone(), vec![x.clone(), y.clone()], None);
        let (inters, results) = cse_scoped(&[s.clone() * x.clone(), s * y.clone()]);
        // Match first (defines __m1), then the let it feeds.
        assert_eq!(*the_match(&inters[0]).0, k);
        assert_eq!(inters[1].as_let().unwrap(), ("__x0", &symbol("__m1")));
        assert_eq!(format!("{}", results[0]), "x * __x0");
        assert_eq!(format!("{}", results[1]), "y * __x0");
    }

    #[test]
    fn loss_kernels_and_weight_fuse_into_one_match() {
        // A robust loss selecting per kind and its weight rho'(s): one
        // match; inside the Huber arm its two branches fuse on k2 - s
        // into one if, with the square root on the else side only; the
        // Cauchy reciprocal inside the Cauchy arm only. This is the
        // SYM.md example; keep the two in step.
        let (k, s, k2) = (symbol("k"), symbol("s"), symbol("k2"));
        let rho = select(k, vec![
            s.clone(),
            crate::loss_huber(s.clone(), k2.clone()),
            crate::loss_cauchy(s.clone(), k2),
        ], None);
        let w = rho.diff("s");
        let (inters, simplified) = cse_scoped(&[rho, w]);
        assert_eq!(inters.len(), 1);
        assert_eq!(format!("{} {}", simplified[0], simplified[1]), "__m0 __m1");
        let code = inters[0].to_rust("f64");
        assert_eq!(code,
            "let (__m0, __m1) = { let __sel = k.select_index(); match __sel { \
             0 => (s, 1.0_f64), \
             1 => { let __x2 = k2 - s; \
             let (__m3, __m4) = if __x2 >= 0.0 { (s, 1.0_f64) } \
             else { let __x5 = (k2 * s).sqrt(); (-k2 + 2.0_f64 * __x5, k2 / __x5) }; \
             (__m3, __m4) }, \
             2 => { let __x6 = s / k2 + 1.0_f64; (k2 * __x6.ln(), 1.0_f64 / __x6) }, \
             _ => panic!(\"select index {} out of range 0..3\", __sel) } };");
    }

    #[test]
    fn flat_cse_keeps_its_shape_and_hoists_across_arms() {
        // The flat form: plain (name, expr) lets, arms transparent, the
        // select left in place for inline emission.
        let (k, x, y) = (symbol("k"), symbol("x"), symbol("y"));
        let root = sqrt(x.clone() * y.clone());
        let rho = select(k.clone(), vec![x.clone(), root.clone()], None);
        let w = select(k, vec![constant(1.0), root.clone() + constant(1.0)], None);
        let (lets, results) = cse(&[rho, w]);
        assert_eq!(lets, vec![("__x0".to_string(), root)]);
        assert!(matches!(results[0].as_ref(), Expr::Select { .. }));
        assert!(results[0].to_rust("").starts_with("{ let __sel = k.select_index(); match __sel { 0 => x, 1 => __x0, "));
    }

    #[test]
    fn branch_sides_are_scopes_and_fuse_on_the_condition() {
        // Huber alone: rho and rho' branch on k2 - s. The condition is
        // shared above (hoisted once), the two branches fuse into one
        // if, and the square root sits on the else side only.
        let (s, k2) = (symbol("s"), symbol("k2"));
        let rho = crate::loss_huber(s.clone(), k2.clone());
        let w = rho.diff("s");
        let (inters, results) = cse_scoped(&[rho, w]);
        assert_eq!(inters.len(), 2, "{inters:?}");
        assert_eq!(inters[0].to_rust(""), "let __x0 = k2 - s;");
        assert!(matches!(the_switch(&inters[1]), Switch::Sign(_)));
        assert_eq!(inters[1].to_rust(""),
            "let (__m1, __m2) = if __x0 >= 0.0 { (s, 1.0) } \
             else { let __x3 = (k2 * s).sqrt(); (-k2 + 2.0 * __x3, k2 / __x3) };");
        assert_eq!(results, vec![symbol("__m1"), symbol("__m2")]);
    }

    #[test]
    fn piecewise_emits_nested_ifs() {
        let (x, a0, a1, a2, b0, b1) = (symbol("x"), symbol("a0"), symbol("a1"),
            symbol("a2"), symbol("b0"), symbol("b1"));
        let e = crate::piecewise(x, vec![a0, a1, a2], vec![b0, b1]);
        let (inters, results) = cse_scoped(&[e]);
        assert_eq!(results, vec![symbol("__m0")]);
        assert_eq!(inters.len(), 1);
        assert_eq!(inters[0].to_rust(""),
            "let __m0 = if b0 - x >= 0.0 { a0 } \
             else { let __m1 = if b1 - x >= 0.0 { a1 } else { a2 }; __m1 };");
    }

    #[test]
    fn select_and_branch_nest_either_way() {
        let (q, k, x, y, z) = (symbol("q"), symbol("k"), symbol("x"), symbol("y"), symbol("z"));
        let e = crate::branch(q.clone(), select(k.clone(), vec![x.clone(), y.clone()], None), z.clone());
        let (inters, _) = cse_scoped(&[e]);
        assert!(matches!(the_switch(&inters[0]), Switch::Sign(_)));
        let then_side = &the_match(&inters[0]).2[0];
        assert!(matches!(the_switch(&then_side.inters[0]), Switch::Index(_)));

        let e = select(k, vec![crate::branch(q, x, y), z], None);
        let (inters, _) = cse_scoped(&[e]);
        assert!(matches!(the_switch(&inters[0]), Switch::Index(_)));
        let arm0 = &the_match(&inters[0]).2[0];
        assert!(matches!(the_switch(&arm0.inters[0]), Switch::Sign(_)));
    }

    #[test]
    fn flat_cse_hoists_across_branch_sides_as_before() {
        let (s, k2) = (symbol("s"), symbol("k2"));
        let rho = crate::loss_huber(s.clone(), k2);
        let w = rho.diff("s");
        let (lets, results) = cse(&[rho, w]);
        let hoisted: Vec<String> = lets.iter().map(|(_, e)| format!("{e}")).collect();
        assert!(hoisted.iter().any(|e| e.starts_with("sqrt(")), "{hoisted:?}");
        assert!(results.iter().all(|r| matches!(r.as_ref(), Expr::Branch(..))));
    }

    #[test]
    fn divisor_inside_arm_only_is_not_hoisted() {
        // 1/z twice inside one arm is that arm's reciprocal; nothing above.
        let (k, x, y, z) = (symbol("k"), symbol("x"), symbol("y"), symbol("z"));
        let e = select(k, vec![x.clone() / z.clone() + y.clone() / z.clone(), x.clone()], None);
        let (inters, _) = cse_scoped(&[e]);
        assert_eq!(inters.len(), 1);
        let arms = the_match(&inters[0]).2;
        assert_eq!(arms[0].inters.len(), 1);
        assert_eq!(arms[0].inters[0].as_let().unwrap().1, &(constant(1.0) / z));
    }
}
