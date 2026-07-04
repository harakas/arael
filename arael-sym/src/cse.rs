//! Common Subexpression Elimination (CSE).
//!
//! Iteratively extracts repeated subexpressions, working bottom-up (deepest
//! first) so that inner replacements create new matching opportunities at
//! higher levels.

use std::collections::HashMap;
use crate::{E, Expr, symbol};

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
        Expr::Clamp(a, b, c) => 1 + expr_cost(a) + expr_cost(b) + expr_cost(c),
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
        Expr::Clamp(a, b, c) => 1 + expr_depth(a).max(expr_depth(b)).max(expr_depth(c)),
        Expr::Func { args, .. } => {
            1 + args.iter().map(expr_depth).max().unwrap_or(0)
        }
    }
}

/// Walk expression tree, count occurrences of each subexpression.
fn count_subexprs(e: &E, counts: &mut HashMap<E, usize>) {
    *counts.entry(e.clone()).or_insert(0) += 1;
    match e.as_ref() {
        Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => {}
        Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
        | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
        | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
        | Expr::Exp(a) | Expr::Ln(a) | Expr::Log2(a) | Expr::Log10(a)
        | Expr::Sqrt(a) | Expr::Abs(a)
        | Expr::Heaviside(a) => count_subexprs(a, counts),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
        | Expr::Div(a, b) | Expr::Pow(a, b) | Expr::Atan2(a, b) => {
            count_subexprs(a, counts);
            count_subexprs(b, counts);
        }
        Expr::Clamp(a, b, c) => {
            count_subexprs(a, counts);
            count_subexprs(b, counts);
            count_subexprs(c, counts);
        }
        Expr::Func { args, .. } => {
            for arg in args { count_subexprs(arg, counts); }
        }
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
        Expr::Func { name, params, kind, args } => {
            let new_args = args.iter().map(|a| replace(a, target, replacement)).collect();
            E::new(Expr::Func { name: name.clone(), params: params.clone(), kind: kind.clone(), args: new_args })
        }
    }
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
/// Iteratively extracts repeated subexpressions, deepest first.
/// Each iteration: count all subexprs, pick the best candidate (deepest with
/// count >= 2), replace it everywhere, repeat until no more candidates.
pub fn cse(exprs: &[E]) -> (Vec<(String, E)>, Vec<E>) {
    if exprs.is_empty() {
        return (vec![], vec![]);
    }

    let mut results = exprs.to_vec();
    let mut intermediates: Vec<(String, E)> = Vec::new();
    let mut var_counter = 0usize;

    loop {
        // Count subexpressions across results AND intermediate definitions
        let mut counts: HashMap<E, usize> = HashMap::new();
        for r in &results {
            count_subexprs(r, &mut counts);
        }
        for (_, expr) in &intermediates {
            count_subexprs(expr, &mut counts);
        }

        // Find the best candidate: count >= 2, cost >= 1, and at least
        // one symbol. Symbol-free (constant) subexpressions are never
        // extracted: they cost nothing at runtime (the compiler folds
        // them), and hoisting one into its own `let` strips the type
        // context that unsuffixed literals in generated code rely on
        // (e.g. `let __x = 2.2e-16.powf(2.0);` is an ambiguous numeric
        // type, while the same expression inline infers from its
        // surroundings).
        // Rank by savings = (count - 1) * cost — how many ops we save.
        // The display string as the final tie-break makes the choice a
        // total order: HashMap iteration order is randomized per
        // instance, and max_by_key keeps the LAST maximum it sees, so
        // without it two identical builds pick different candidates and
        // emit differently-named/ordered temporaries (nondeterministic
        // generated code).
        let best = counts.into_iter()
            .filter(|(e, count)| *count >= 2 && expr_cost(e) >= 1 && !e.symbols().is_empty())
            .max_by_key(|(e, count)| {
                let cost = expr_cost(e);
                let savings = (*count - 1) * cost;
                // Primary: most savings
                // Secondary: prefer deeper (to enable further extraction)
                // Tertiary: display string, for determinism
                (savings, expr_depth(e), format!("{}", e))
            });

        let (subexpr, _count) = match best {
            Some(b) => b,
            None => break,
        };

        let var_name = format!("__x{}", var_counter);
        var_counter += 1;
        let var_sym = symbol(&var_name);

        // Replace in all results
        for r in results.iter_mut() {
            *r = replace(r, &subexpr, &var_sym);
        }

        // Replace in existing intermediates' definitions too
        for (_, expr) in intermediates.iter_mut() {
            *expr = replace(expr, &subexpr, &var_sym);
        }

        intermediates.push((var_name, subexpr));
    }

    // Post-pass: extract common divisors as reciprocals.
    // If `/ x` appears 2+ times, extract `__xN = 1.0 / x` and replace
    // `a / x` with `a * __xN`.
    let mut divisor_counts: HashMap<E, usize> = HashMap::new();
    for r in &results {
        count_divisors(r, &mut divisor_counts);
    }
    for (_, expr) in &intermediates {
        count_divisors(expr, &mut divisor_counts);
    }
    // Sort for determinism: HashMap iteration order would name and
    // order the reciprocal temporaries randomly across builds.
    let mut divisors: Vec<(E, usize)> = divisor_counts.into_iter().collect();
    divisors.sort_by_key(|(e, _)| format!("{}", e));
    for (divisor, count) in divisors {
        if count >= 2 {
            let var_name = format!("__x{}", var_counter);
            var_counter += 1;
            let var_sym = symbol(&var_name);
            let recip = E::new(Expr::Div(E::new(Expr::Const(1.0)), divisor.clone()));
            for r in results.iter_mut() {
                *r = replace_divisor(r, &divisor, &var_sym);
            }
            for (_, expr) in intermediates.iter_mut() {
                *expr = replace_divisor(expr, &divisor, &var_sym);
            }
            intermediates.push((var_name, recip));
        }
    }

    // Topological sort: ensure each intermediate is defined before it's used.
    let intermediates = topo_sort_intermediates(intermediates);
    (intermediates, results)
}

/// Count how many times each expression appears as a divisor (right side of Div).
fn count_divisors(e: &E, counts: &mut HashMap<E, usize>) {
    match e.as_ref() {
        Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => {}
        Expr::Div(a, b) => {
            *counts.entry(b.clone()).or_insert(0) += 1;
            count_divisors(a, counts);
            count_divisors(b, counts);
        }
        Expr::Neg(a) | Expr::Sin(a) | Expr::Cos(a) | Expr::Tan(a)
        | Expr::Asin(a) | Expr::Acos(a) | Expr::Atan(a)
        | Expr::Sinh(a) | Expr::Cosh(a) | Expr::Tanh(a)
        | Expr::Exp(a) | Expr::Ln(a) | Expr::Log2(a) | Expr::Log10(a)
        | Expr::Sqrt(a) | Expr::Abs(a)
        | Expr::Heaviside(a) => count_divisors(a, counts),
        Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b)
        | Expr::Pow(a, b) | Expr::Atan2(a, b) => {
            count_divisors(a, counts);
            count_divisors(b, counts);
        }
        Expr::Clamp(a, b, c) => {
            count_divisors(a, counts);
            count_divisors(b, counts);
            count_divisors(c, counts);
        }
        Expr::Func { args, .. } => {
            for arg in args { count_divisors(arg, counts); }
        }
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
        Expr::Func { name, params, kind, args } => {
            let new_args = args.iter().map(|a| replace_divisor(a, divisor, replacement)).collect();
            E::new(Expr::Func { name: name.clone(), params: params.clone(), kind: kind.clone(), args: new_args })
        }
    }
}

/// Topological sort of intermediates so dependencies come first.
fn topo_sort_intermediates(intermediates: Vec<(String, E)>) -> Vec<(String, E)> {
    use std::collections::HashSet;

    let names: HashSet<String> = intermediates.iter().map(|(n, _)| n.clone()).collect();

    // Build dependency graph: for each intermediate, which other intermediates
    // does it reference? Sorted: HashSet iteration order would randomize the
    // dependents lists and with them the emitted definition order.
    let deps: Vec<Vec<String>> = intermediates.iter().map(|(_, expr)| {
        let vars = expr.free_vars();
        let mut d: Vec<String> = vars.into_iter().filter(|v| names.contains(v)).collect();
        d.sort();
        d
    }).collect();

    // Kahn's algorithm
    let n = intermediates.len();
    let name_to_idx: HashMap<String, usize> = intermediates.iter().enumerate()
        .map(|(i, (n, _))| (n.clone(), i)).collect();

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
