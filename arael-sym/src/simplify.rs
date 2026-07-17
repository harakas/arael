use std::cmp::Ordering;
use super::{Expr, E, constant};

fn is_const(e: &Expr, v: f64) -> bool {
    matches!(e, Expr::Const(c) if *c == v)
}

fn is_const_int(e: &Expr) -> Option<i64> {
    if let Expr::Const(v) = e {
        return crate::as_exact_int(*v);
    }
    None
}

// ---------------------------------------------------------------------------
// Canonical ordering for expressions
// ---------------------------------------------------------------------------

fn type_priority(e: &Expr) -> u8 {
    match e {
        Expr::Const(_) => 100, // constants sort last in Add, first extracted in Mul
        Expr::Sym(_) => 0,
        Expr::Pow(base, _) => {
            // Pow of symbol sorts alongside symbols for degree ordering
            if matches!(base.as_ref(), Expr::Sym(_)) { 0 } else { 2 }
        }
        Expr::Mul(_, _) => 1,
        Expr::Neg(_) => 3,
        Expr::Add(_, _) | Expr::Sub(_, _) => 4,
        _ => 5, // functions
    }
}

/// Extract the "leading symbol name" for ordering purposes.
fn leading_sym(e: &Expr) -> Option<&str> {
    match e {
        Expr::Sym(s) => Some(s),
        Expr::Pow(base, _) => leading_sym(base),
        Expr::Mul(a, b) => leading_sym(a).or_else(|| leading_sym(b)),
        Expr::Neg(a) => leading_sym(a),
        _ => None,
    }
}

/// Estimate "degree" for ordering within Add (higher degree first).
fn degree(e: &Expr) -> i64 {
    match e {
        Expr::Sym(_) => 1,
        Expr::Pow(_, exp) => {
            if let Expr::Const(v) = exp.as_ref() {
                *v as i64
            } else {
                2 // treat non-const exponent as degree 2
            }
        }
        Expr::Mul(a, b) => degree(a) + degree(b),
        Expr::Neg(a) => degree(a),
        Expr::Const(_) => 0,
        _ => 1,
    }
}

/// Canonical comparison for sorting factors in Mul.
fn mul_factor_cmp(a: &E, b: &E) -> Ordering {
    let sa = leading_sym(a);
    let sb = leading_sym(b);
    match (sa, sb) {
        (Some(sa), Some(sb)) => {
            let cmp = sa.cmp(sb);
            if cmp != Ordering::Equal { return cmp; }
            // Same leading symbol: compare by degree (lower first in Mul context)
            degree(a).cmp(&degree(b))
        }
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => {
            let cmp = type_priority(a).cmp(&type_priority(b));
            if cmp != Ordering::Equal { return cmp; }
            // Tiebreaker: compare by string representation for deterministic ordering
            format!("{}", a).cmp(&format!("{}", b))
        }
    }
}

/// Canonical comparison for sorting terms in Add.
/// Higher degree first, then alphabetical, constants last.
fn add_term_cmp(a: &E, b: &E) -> Ordering {
    let pa = type_priority(a);
    let pb = type_priority(b);
    // Constants always last
    if pa == 100 && pb != 100 { return Ordering::Greater; }
    if pa != 100 && pb == 100 { return Ordering::Less; }
    if pa == 100 && pb == 100 {
        // Both constants: compare by value
        if let (Expr::Const(va), Expr::Const(vb)) = (a.as_ref(), b.as_ref()) {
            return vb.partial_cmp(va).unwrap_or(Ordering::Equal);
        }
        return Ordering::Equal;
    }

    // Higher degree first
    let da = degree(a);
    let db = degree(b);
    if da != db { return db.cmp(&da); }

    // Same degree: alphabetical by leading symbol
    let sa = leading_sym(a);
    let sb = leading_sym(b);
    match (sa, sb) {
        (Some(sa), Some(sb)) => sa.cmp(sb),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

// ---------------------------------------------------------------------------
// Mul normalization helpers
// ---------------------------------------------------------------------------

/// Extract (base, const_exponent) from a factor. Nested constant powers
/// collapse via `(x^a)^b = x^(a*b)`, so `(cc^2)^2` reports base `cc`,
/// exponent `4` and cancels against a `cc^4` elsewhere. Folding is limited
/// to integer exponents: for a fractional exponent over a possibly-negative
/// base the two forms differ (`(x^2)^0.5 = |x| != x`).
fn base_and_exp(e: &E) -> (E, f64) {
    if let Expr::Pow(base, exp) = e.as_ref()
        && let Expr::Const(n) = exp.as_ref() {
            if n.fract() == 0.0 {
                let (inner_base, inner_exp) = base_and_exp(base);
                if inner_exp.fract() == 0.0 {
                    return (inner_base, inner_exp * *n);
                }
            }
            return (base.clone(), *n);
        }
    (e.clone(), 1.0)
}

/// Flatten nested Mul into a list of factors, extracting a numeric coefficient.
/// Also expands (a*b)^n → a^n * b^n for constant exponents.
fn flatten_mul(e: &E) -> (f64, Vec<E>) {
    match e.as_ref() {
        Expr::Mul(a, b) => {
            let (ca, mut fa) = flatten_mul(a);
            let (cb, fb) = flatten_mul(b);
            fa.extend(fb);
            (ca * cb, fa)
        }
        Expr::Neg(inner) => {
            let (c, f) = flatten_mul(inner);
            (-c, f)
        }
        Expr::Const(v) => (*v, vec![]),
        // (a * b * ...)^n → a^n * b^n * ... (expand compound-base powers).
        // Only valid for INTEGER exponents: for fractional n and negative
        // factors the sides differ ((xy)^0.5 = 1 but x^0.5 * y^0.5 = NaN
        // at x = y = -1), and a negative extracted coefficient would bake
        // a NaN into the tree via powf.
        Expr::Pow(base, exp) if matches!(base.as_ref(), Expr::Mul(..) | Expr::Neg(..)) => {
            if is_const_int(exp.as_ref()).is_some() {
                let n = match exp.as_ref() { Expr::Const(v) => *v, _ => unreachable!() };
                let (c_base, factors) = flatten_mul(base);
                let coeff = c_base.powf(n);
                let powered: Vec<E> = factors
                    .into_iter()
                    .map(|f| E::new(Expr::Pow(f, exp.clone())))
                    .collect();
                (coeff, powered)
            } else {
                (1.0, vec![e.clone()])
            }
        }
        _ => (1.0, vec![e.clone()]),
    }
}

/// Combine factors with the same base by summing exponents.
/// Returns a list of (base, total_exponent) pairs.
fn combine_powers(factors: Vec<E>) -> Vec<(E, f64)> {
    let mut groups: Vec<(E, f64)> = Vec::new();
    for f in factors {
        let (base, exp) = base_and_exp(&f);
        if let Some(entry) = groups.iter_mut().find(|(b, _)| *b == base) {
            entry.1 += exp;
        } else {
            groups.push((base, exp));
        }
    }
    groups
}

/// Build a simplified Mul from coefficient and factor list.
fn build_product(coeff: f64, mut factors: Vec<E>) -> E {
    if coeff == 0.0 { return constant(0.0); }

    // Sort factors canonically
    factors.sort_by(mul_factor_cmp);

    // Build factor chain
    let factors_expr = if factors.is_empty() {
        return constant(coeff);
    } else {
        let mut iter = factors.into_iter();
        let first = iter.next().unwrap();
        iter.fold(first, |acc, f| E::new(Expr::Mul(acc, f)))
    };

    if coeff == 1.0 {
        factors_expr
    } else if coeff == -1.0 {
        E::new(Expr::Neg(factors_expr))
    } else {
        E::new(Expr::Mul(constant(coeff), factors_expr))
    }
}

/// Full Mul simplification: flatten, fold constants, combine powers, sort, rebuild.
fn simplify_product(a: E, b: E) -> E {
    // Flatten both sides
    let (ca, fa) = flatten_mul(&a);
    let (cb, fb) = flatten_mul(&b);

    let coeff = ca * cb;
    let mut all_factors = fa;
    all_factors.extend(fb);

    if coeff == 0.0 { return constant(0.0); }
    if all_factors.is_empty() { return constant(coeff); }

    // If any factor is a Div, combine everything into a single fraction
    // e.g. Mul(gamma^2, Div(-a, gamma*sigma)) → Div(-a*gamma, sigma)
    let has_div = all_factors.iter().any(|f| matches!(f.as_ref(), Expr::Div(..)));
    if has_div {
        let mut num_factors = Vec::new();
        let mut den_factors = Vec::new();
        let mut num_coeff = coeff;
        for f in all_factors {
            let (fc, nf, df) = flatten_fraction(&f);
            num_coeff *= fc;
            num_factors.extend(nf);
            den_factors.extend(df);
        }
        let num_groups = combine_powers(num_factors);
        let den_groups = combine_powers(den_factors);
        let (final_coeff, final_num, final_den) = cancel_common(num_coeff, num_groups, den_groups);
        let num_expr = build_product_from_groups(final_coeff, final_num);
        let den_expr = build_product_from_groups(1.0, final_den);
        if is_const(&den_expr, 1.0) {
            return num_expr;
        }
        return E::new(Expr::Div(num_expr, den_expr));
    }

    // Combine like bases
    let groups = combine_powers(all_factors);

    // Rebuild factors from groups
    let mut factors: Vec<E> = Vec::new();
    for (base, exp) in groups {
        if exp == 0.0 {
            // x^0 = 1, skip
        } else if exp == 1.0 {
            factors.push(base);
        } else {
            factors.push(E::new(Expr::Pow(base, constant(exp))));
        }
    }

    // Sort factors into canonical order so a*b == b*a structurally
    factors.sort_by(mul_factor_cmp);

    build_product(coeff, factors)
}

// ---------------------------------------------------------------------------
// Add normalization helpers
// ---------------------------------------------------------------------------

/// Flatten Add/Sub/Neg into a list of (coefficient, base) pairs.
/// The base is the expression without the numeric coefficient.
fn flatten_additive(e: &E) -> Vec<(f64, E)> {
    match e.as_ref() {
        Expr::Add(a, b) => {
            let mut terms = flatten_additive(a);
            terms.extend(flatten_additive(b));
            terms
        }
        Expr::Sub(a, b) => {
            let mut terms = flatten_additive(a);
            let neg_terms: Vec<(f64, E)> = flatten_additive(b)
                .into_iter()
                .map(|(c, base)| (-c, base))
                .collect();
            terms.extend(neg_terms);
            terms
        }
        Expr::Neg(inner) => {
            flatten_additive(inner)
                .into_iter()
                .map(|(c, base)| (-c, base))
                .collect()
        }
        _ => {
            let (coeff, base) = extract_coeff(e);
            vec![(coeff, base)]
        }
    }
}

/// Extract numeric coefficient and base from a term.
fn extract_coeff(e: &E) -> (f64, E) {
    match e.as_ref() {
        Expr::Const(v) => (*v, constant(1.0)),
        Expr::Mul(a, b) => {
            if let Expr::Const(v) = a.as_ref() {
                // Const * rest — check if rest is also Mul(Const, ...)
                let (inner_c, inner_b) = extract_coeff(b);
                return (v * inner_c, inner_b);
            }
            if let Expr::Const(v) = b.as_ref() {
                let (inner_c, inner_b) = extract_coeff(a);
                return (v * inner_c, inner_b);
            }
            (1.0, e.clone())
        }
        Expr::Neg(inner) => {
            let (c, base) = extract_coeff(inner);
            (-c, base)
        }
        _ => (1.0, e.clone()),
    }
}

/// Group terms by base, summing numeric coefficients.
fn combine_like_terms(terms: Vec<(f64, E)>) -> Vec<(f64, E)> {
    let mut groups: Vec<(f64, E)> = Vec::new();
    for (coeff, base) in terms {
        if let Some(entry) = groups.iter_mut().find(|(_, b)| *b == base) {
            entry.0 += coeff;
        } else {
            groups.push((coeff, base));
        }
    }
    groups
}

/// Build an Add/Sub chain from (coefficient, base) pairs.
fn build_sum(mut terms: Vec<(f64, E)>) -> E {
    // Remove zero-coefficient terms. Exact comparison on purpose:
    // like-term coefficients are summed as plain f64, so true
    // cancellation (x - x) yields exactly 0.0. An epsilon tolerance
    // here silently deleted legitimate tiny terms (small residual
    // weights, regularization constants) from the generated code.
    terms.retain(|(c, _)| *c != 0.0);

    if terms.is_empty() {
        return constant(0.0);
    }

    // Sort terms: sort bases for consistent output
    terms.sort_by(|(_, a), (_, b)| add_term_cmp(a, b));

    let make_term = |coeff: f64, base: E| -> E {
        if is_const(&base, 1.0) {
            constant(coeff)
        } else if coeff == 1.0 {
            base
        } else if coeff == -1.0 {
            E::new(Expr::Neg(base))
        } else {
            E::new(Expr::Mul(constant(coeff), base))
        }
    };

    let mut iter = terms.into_iter();
    let (first_c, first_b) = iter.next().unwrap();
    let mut result = make_term(first_c, first_b);

    for (coeff, base) in iter {
        if coeff > 0.0 {
            result = E::new(Expr::Add(result, make_term(coeff, base)));
        } else {
            result = E::new(Expr::Sub(result, make_term(-coeff, base)));
        }
    }

    result
}

/// Full Add simplification: flatten, combine like terms, sort, rebuild.
fn simplify_sum(a: E, b: E, negate_b: bool) -> E {
    let mut terms = flatten_additive(&a);
    let b_terms = flatten_additive(&b);
    if negate_b {
        terms.extend(b_terms.into_iter().map(|(c, base)| (-c, base)));
    } else {
        terms.extend(b_terms);
    }

    let combined = combine_like_terms(terms);
    build_sum(combined)
}

// ---------------------------------------------------------------------------
// Div / fraction helpers
// ---------------------------------------------------------------------------

/// Flatten an expression into (coeff, numerator_factors, denominator_factors).
/// Handles Div chains: (a/b)/c → num=[a_factors], den=[b_factors, c_factors]
/// Handles reciprocals: a/(b/c) → num=[a_factors, c_factors], den=[b_factors]
fn flatten_fraction(e: &E) -> (f64, Vec<E>, Vec<E>) {
    match e.as_ref() {
        Expr::Div(a, b) => {
            let (ca, na, da) = flatten_fraction(a);
            let (cb, nb, db) = flatten_fraction(b);
            // (na/da) / (nb/db) = (na*db) / (da*nb)
            let mut num = na;
            num.extend(db);
            let mut den = da;
            den.extend(nb);
            (ca / cb, num, den)
        }
        _ => {
            let (c, factors) = flatten_mul(e);
            (c, factors, vec![])
        }
    }
}

/// Cancel common bases between numerator and denominator power groups.
/// gamma^3 in num + gamma^2 in den → gamma^1 in num.
/// sigma^1 in num + sigma^2 in den → sigma^1 in den.
fn cancel_common(
    coeff: f64,
    mut num: Vec<(E, f64)>,
    den: Vec<(E, f64)>,
) -> (f64, Vec<(E, f64)>, Vec<(E, f64)>) {
    let mut final_den = Vec::new();
    for (base, den_exp) in den {
        if let Some(entry) = num.iter_mut().find(|(b, _)| *b == base) {
            entry.1 -= den_exp;
        } else {
            final_den.push((base, den_exp));
        }
    }
    // Move negative-exponent entries from num to den
    let mut moved = Vec::new();
    for (i, (_base, exp)) in num.iter().enumerate() {
        if *exp < 0.0 {
            moved.push(i);
        }
    }
    for i in moved.into_iter().rev() {
        let (base, exp) = num.remove(i);
        final_den.push((base, -exp));
    }
    num.retain(|(_, exp)| *exp != 0.0);
    (coeff, num, final_den)
}

/// Build a product from (base, exponent) groups.
fn build_product_from_groups(coeff: f64, groups: Vec<(E, f64)>) -> E {
    let factors: Vec<E> = groups
        .into_iter()
        .map(|(base, exp)| {
            if exp == 1.0 {
                base
            } else {
                E::new(Expr::Pow(base, constant(exp)))
            }
        })
        .collect();
    build_product(coeff, factors)
}

fn simplify_div(a: E, b: E) -> E {
    // Quick constant cases
    if let (Expr::Const(va), Expr::Const(vb)) = (a.as_ref(), b.as_ref())
        && *vb != 0.0 {
            return constant(va / vb);
        }
    if is_const(&a, 0.0) { return constant(0.0); }
    if is_const(&b, 1.0) { return a; }
    if a == b { return constant(1.0); }

    // Flatten both sides, collecting all num/den factors across Div chains
    let (ca, na, da) = flatten_fraction(&a);
    let (cb, nb, db) = flatten_fraction(&b);
    // A zero denominator coefficient must not be folded (ca/cb would bake
    // inf/NaN into the tree); leave the division for IEEE eval semantics.
    if cb == 0.0 {
        return E::new(Expr::Div(a, b));
    }
    // a/b = (na*db) / (da*nb), coeff = ca/cb
    let mut num_factors = na;
    num_factors.extend(db);
    let mut den_factors = da;
    den_factors.extend(nb);
    let coeff = ca / cb;

    if coeff == 0.0 { return constant(0.0); }

    // Combine powers within each side
    let num_groups = combine_powers(num_factors);
    let den_groups = combine_powers(den_factors);

    // Cancel common bases between num and den
    let (coeff, final_num, final_den) = cancel_common(coeff, num_groups, den_groups);

    // Rebuild
    let num_expr = build_product_from_groups(coeff, final_num);
    let den_expr = build_product_from_groups(1.0, final_den);

    if is_const(&den_expr, 1.0) {
        num_expr
    } else {
        E::new(Expr::Div(num_expr, den_expr))
    }
}

// ---------------------------------------------------------------------------
// Main simplify
// ---------------------------------------------------------------------------

// A binary node is "already normal" -- its simplify_{product,sum,div} would
// return it unchanged -- when both sides are atomic and cannot combine: no
// nested product/quotient/sum/constant, distinct leading symbols (so no
// like-base/term merge), and (for products/sums) canonical order. Conservative:
// returns false whenever a combine might apply, so the real simplifier still
// runs. Lets the simplify arms keep the original Rc instead of rebuilding.
fn distinct_leading(a: &E, b: &E) -> bool {
    matches!((leading_sym(a.as_ref()), leading_sym(b.as_ref())), (Some(x), Some(y)) if x != y)
}
/// An atomic factor: nothing `flatten_mul`/`simplify_product` would decompose.
/// A compound-base power (`(a*b)^n`, `(-a)^n`) expands, so it is NOT atomic.
fn plain_factor(e: &E) -> bool {
    match e.as_ref() {
        Expr::Mul(..) | Expr::Div(..) | Expr::Const(_) | Expr::Neg(..) => false,
        Expr::Pow(base, _) => !matches!(base.as_ref(), Expr::Mul(..) | Expr::Neg(..)),
        _ => true,
    }
}
fn mul_is_normal(a: &E, b: &E) -> bool {
    plain_factor(a) && plain_factor(b) && distinct_leading(a, b) && mul_factor_cmp(a, b) != Ordering::Greater
}
fn add_is_normal(a: &E, b: &E) -> bool {
    // Exclude Neg: simplify_sum rewrites x + (-y) to x - y, so an Add with a
    // negated operand is not yet normal.
    let plain = |e: &E| !matches!(e.as_ref(), Expr::Add(..) | Expr::Sub(..) | Expr::Const(_) | Expr::Neg(..));
    plain(a) && plain(b) && distinct_leading(a, b) && add_term_cmp(a, b) != Ordering::Greater
}
fn div_is_normal(a: &E, b: &E) -> bool {
    plain_factor(a) && plain_factor(b) && distinct_leading(a, b)
}

impl E {
    /// Apply algebraic simplification rules.
    ///
    /// Performs constant folding, identity elimination (0+x=x, 1*x=x),
    /// like-term collection, power combination, fraction cancellation, and
    /// canonical ordering. Iterates until a fixed point is reached.
    ///
    /// Identity-preserving: a subexpression that simplifies to itself keeps its
    /// `Rc`, so structural sharing survives the pass and unchanged nodes are not
    /// reallocated.
    pub fn simplify(&self) -> E {
        // Fixpoint iteration. simplify_once is identity-preserving and pure: if a
        // pass returns the same Rc it was given, that expression is already a
        // fixpoint, so the next pass would change nothing -- stop without the
        // extra traversal.
        let mut result = self.clone();
        for _ in 0..11 {
            let next = result.simplify_once();
            if std::rc::Rc::ptr_eq(&next.0, &result.0) || next == result {
                return next;
            }
            result = next;
        }
        result
    }

    fn simplify_once(&self) -> E {
        self.0.simplify_once_inner(self)
    }
}

impl Expr {
    /// Bare-`Expr` entry to [`E::simplify`], for callers holding an `Expr`.
    pub fn simplify(&self) -> E {
        E::new(self.clone()).simplify()
    }

    fn simplify_once_inner(&self, orig: &E) -> E {
        /// Check if expression is the named constant "pi".
        fn is_pi(e: &E) -> bool {
            matches!(e.as_ref(), Expr::NamedConst { name, .. } if name == "pi")
        }

        /// Check if expression is the named constant "e" (Euler's number).
        fn is_euler(e: &E) -> bool {
            matches!(e.as_ref(), Expr::NamedConst { name, .. } if name == "e")
        }

        /// Extract coefficient k if expression is of the form k*pi.
        /// Returns Some(k) for: pi->1, 2*pi->2, pi/2->0.5, -pi->-1, etc.
        fn pi_coeff(e: &E) -> Option<f64> {
            if is_pi(e) { return Some(1.0); }
            match e.as_ref() {
                Expr::Neg(inner) => pi_coeff(inner).map(|c| -c),
                Expr::Mul(a, b) => {
                    if let Expr::Const(c) = a.as_ref() && is_pi(b) { return Some(*c); }
                    if let Expr::Const(c) = b.as_ref() && is_pi(a) { return Some(*c); }
                    None
                }
                Expr::Div(a, b) => {
                    if let Expr::Const(d) = b.as_ref() { return pi_coeff(a).map(|c| c / d); }
                    None
                }
                _ => None,
            }
        }

        /// Try to simplify sin(k*pi) for special values of k.
        /// Uses twelfths: k*12 mod 24 to cover pi/6, pi/4, pi/3, pi/2, etc.
        fn sin_pi(k: f64) -> Option<E> {
            let twelfths = k * 12.0;
            if (twelfths - twelfths.round()).abs() > 1e-9 { return None; }
            let idx = ((twelfths.round() as i64) % 24 + 24) % 24;
            // sin at 0, pi/12, pi/6, pi/4, pi/3, 5pi/12, pi/2, ...
            match idx {
                0 | 12 => Some(constant(0.0)),                          // sin(0), sin(pi)
                6 | 18 => Some(if idx == 6 { constant(1.0) } else { constant(-1.0) }), // sin(pi/2), sin(3pi/2)
                2 | 10 => Some(constant(0.5)),                          // sin(pi/6), sin(5pi/6)
                14 | 22 => Some(constant(-0.5)),                        // sin(7pi/6), sin(11pi/6)
                3 | 9 => Some(crate::sqrt(constant(2.0)) / 2.0),              // sin(pi/4), sin(3pi/4)
                15 | 21 => Some(-crate::sqrt(constant(2.0)) / 2.0),           // sin(5pi/4), sin(7pi/4)
                4 | 8 => Some(crate::sqrt(constant(3.0)) / 2.0),              // sin(pi/3), sin(2pi/3)
                16 | 20 => Some(-crate::sqrt(constant(3.0)) / 2.0),           // sin(4pi/3), sin(5pi/3)
                _ => None,
            }
        }

        /// Try to simplify cos(k*pi) for special values of k.
        fn cos_pi(k: f64) -> Option<E> {
            // cos(k*pi) = sin((k + 0.5)*pi)
            sin_pi(k + 0.5)
        }


        match self {
            Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => orig.clone(),

            Expr::Neg(a) => {
                let a2 = a.simplify_once();
                if let Expr::Neg(inner) = a2.as_ref() {
                    return inner.clone();
                }
                if let Expr::Const(v) = a2.as_ref() {
                    return constant(-v);
                }
                if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Neg(a2)) }
            }

            Expr::Add(a, b) => {
                let a2 = a.simplify_once();
                let b2 = b.simplify_once();
                if add_is_normal(&a2, &b2) {
                    if std::rc::Rc::ptr_eq(&a2.0, &a.0) && std::rc::Rc::ptr_eq(&b2.0, &b.0) { return orig.clone(); }
                    return E::new(Expr::Add(a2, b2));
                }
                simplify_sum(a2, b2, false)
            }

            Expr::Sub(a, b) => {
                let a = a.simplify_once();
                let b = b.simplify_once();
                simplify_sum(a, b, true)
            }

            Expr::Mul(a, b) => {
                let a2 = a.simplify_once();
                let b2 = b.simplify_once();
                if mul_is_normal(&a2, &b2) {
                    if std::rc::Rc::ptr_eq(&a2.0, &a.0) && std::rc::Rc::ptr_eq(&b2.0, &b.0) { return orig.clone(); }
                    return E::new(Expr::Mul(a2, b2));
                }
                simplify_product(a2, b2)
            }

            Expr::Div(a, b) => {
                let a2 = a.simplify_once();
                let b2 = b.simplify_once();
                if div_is_normal(&a2, &b2) {
                    if std::rc::Rc::ptr_eq(&a2.0, &a.0) && std::rc::Rc::ptr_eq(&b2.0, &b.0) { return orig.clone(); }
                    return E::new(Expr::Div(a2, b2));
                }
                simplify_div(a2, b2)
            }

            Expr::Pow(a, b) => {
                let a2 = a.simplify_once();
                let b2 = b.simplify_once();
                if let (Expr::Const(va), Expr::Const(vb)) = (a2.as_ref(), b2.as_ref()) {
                    return constant(va.powf(*vb));
                }
                if is_const(&b2, 0.0) { return constant(1.0); }
                if is_const(&b2, 1.0) { return a2; }
                // No 0^b -> 0 rule for symbolic b: wrong for b == 0
                // (0^0 = 1) and b < 0 (0^b = inf). Constant b is already
                // folded by the Const/Const branch above.
                if is_const(&a2, 1.0) { return constant(1.0); }
                // (x^m)^n = x^(m*n), integer exponents only (sign-safe -- see
                // base_and_exp). Collapses nested powers a later pass or the
                // Const/Const branch above then folds further.
                if let (Expr::Const(n), Expr::Pow(inner, ie)) = (b2.as_ref(), a2.as_ref())
                    && let Expr::Const(m) = ie.as_ref()
                    && n.fract() == 0.0 && m.fract() == 0.0 {
                        return E::new(Expr::Pow(inner.clone(), constant(m * n)));
                }
                if std::rc::Rc::ptr_eq(&a2.0, &a.0) && std::rc::Rc::ptr_eq(&b2.0, &b.0) { orig.clone() } else { E::new(Expr::Pow(a2, b2)) }
            }

            // Inverse function pairs
            Expr::Ln(a) => {
                let a2 = a.simplify_once();
                if let Expr::Exp(inner) = a2.as_ref() { return inner.clone(); }
                if let Expr::Const(v) = a2.as_ref() { return constant(v.ln()); }
                if is_euler(&a2) { return constant(1.0); }
                // ln(e^n) -> n
                if let Expr::Pow(base, exp) = a2.as_ref()
                    && is_euler(base) { return exp.clone(); }
                if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Ln(a2)) }
            }
            Expr::Exp(a) => {
                let a2 = a.simplify_once();
                if let Expr::Ln(inner) = a2.as_ref() { return inner.clone(); }
                if let Expr::Const(v) = a2.as_ref() { return constant(v.exp()); }
                if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Exp(a2)) }
            }

            // Trig functions: constant-fold + pi rules
            Expr::Sin(a) => {
                let a2 = a.simplify_once();
                if let Expr::Const(v) = a2.as_ref() { return constant(v.sin()); }
                if let Some(k) = pi_coeff(&a2) && let Some(v) = sin_pi(k) { return v; }
                if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Sin(a2)) }
            }
            Expr::Cos(a) => {
                let a2 = a.simplify_once();
                if let Expr::Const(v) = a2.as_ref() { return constant(v.cos()); }
                if let Some(k) = pi_coeff(&a2) && let Some(v) = cos_pi(k) { return v; }
                if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Cos(a2)) }
            }
            Expr::Tan(a) => {
                let a2 = a.simplify_once();
                if let Expr::Const(v) = a2.as_ref() { return constant(v.tan()); }
                // tan(n*pi) = 0 for integer n
                if let Some(k) = pi_coeff(&a2)
                    && (k - k.round()).abs() < 1e-9 { return constant(0.0); }
                if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Tan(a2)) }
            }
            Expr::Asin(a) => { let a2 = a.simplify_once(); if let Expr::Const(v) = a2.as_ref() { return constant(v.asin()); } if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Asin(a2)) } }
            Expr::Acos(a) => { let a2 = a.simplify_once(); if let Expr::Const(v) = a2.as_ref() { return constant(v.acos()); } if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Acos(a2)) } }
            Expr::Atan(a) => { let a2 = a.simplify_once(); if let Expr::Const(v) = a2.as_ref() { return constant(v.atan()); } if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Atan(a2)) } }
            Expr::Sinh(a) => { let a2 = a.simplify_once(); if let Expr::Const(v) = a2.as_ref() { return constant(v.sinh()); } if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Sinh(a2)) } }
            Expr::Cosh(a) => { let a2 = a.simplify_once(); if let Expr::Const(v) = a2.as_ref() { return constant(v.cosh()); } if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Cosh(a2)) } }
            Expr::Tanh(a) => { let a2 = a.simplify_once(); if let Expr::Const(v) = a2.as_ref() { return constant(v.tanh()); } if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Tanh(a2)) } }
            Expr::Log2(a) => { let a2 = a.simplify_once(); if let Expr::Const(v) = a2.as_ref() { return constant(v.log2()); } if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Log2(a2)) } }
            Expr::Log10(a) => { let a2 = a.simplify_once(); if let Expr::Const(v) = a2.as_ref() { return constant(v.log10()); } if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Log10(a2)) } }
            Expr::Sqrt(a) => {
                let a2 = a.simplify_once();
                if let Expr::Const(v) = a2.as_ref() { return constant(v.sqrt()); }
                if let Expr::Pow(base, exp) = a2.as_ref()
                    && is_const(exp, 2.0) {
                        return E::new(Expr::Abs(base.clone()));
                    }
                if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Sqrt(a2)) }
            }
            Expr::Abs(a) => { let a2 = a.simplify_once(); if let Expr::Const(v) = a2.as_ref() { return constant(v.abs()); } if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Abs(a2)) } }
            Expr::Heaviside(a) => {
                let a2 = a.simplify_once();
                if let Expr::Const(v) = a2.as_ref() {
                    // Same branch sense as runtime utils::heaviside
                    // (heaviside(NaN) = 0).
                    return constant(if *v >= 0.0 { 1.0 } else { 0.0 });
                }
                if std::rc::Rc::ptr_eq(&a2.0, &a.0) { orig.clone() } else { E::new(Expr::Heaviside(a2)) }
            }
            Expr::Clamp(val, lo, hi) => {
                let val2 = val.simplify_once();
                let lo2 = lo.simplify_once();
                let hi2 = hi.simplify_once();
                if let (Expr::Const(v), Expr::Const(l), Expr::Const(h)) = (val2.as_ref(), lo2.as_ref(), hi2.as_ref()) {
                    return constant(v.clamp(*l, *h));
                }
                if std::rc::Rc::ptr_eq(&val2.0, &val.0) && std::rc::Rc::ptr_eq(&lo2.0, &lo.0) && std::rc::Rc::ptr_eq(&hi2.0, &hi.0) {
                    orig.clone()
                } else {
                    E::new(Expr::Clamp(val2, lo2, hi2))
                }
            }
            Expr::Branch(q, a, b) => {
                let q2 = q.simplify_once();
                let a2 = a.simplify_once();
                let b2 = b.simplify_once();
                // Constant condition collapses to the taken side.
                if let Expr::Const(qv) = q2.as_ref() {
                    return if *qv >= 0.0 { a2 } else { b2 };
                }
                if std::rc::Rc::ptr_eq(&q2.0, &q.0) && std::rc::Rc::ptr_eq(&a2.0, &a.0) && std::rc::Rc::ptr_eq(&b2.0, &b.0) {
                    orig.clone()
                } else {
                    E::new(Expr::Branch(q2, a2, b2))
                }
            }
            Expr::Atan2(y, x) => {
                let y2 = y.simplify_once();
                let x2 = x.simplify_once();
                if let (Expr::Const(vy), Expr::Const(vx)) = (y2.as_ref(), x2.as_ref()) {
                    return constant(vy.atan2(*vx));
                }
                if std::rc::Rc::ptr_eq(&y2.0, &y.0) && std::rc::Rc::ptr_eq(&x2.0, &x.0) { orig.clone() } else { E::new(Expr::Atan2(y2, x2)) }
            }
            Expr::Func { name, params, kind, args } => {
                let new_args: Vec<E> = args.iter().map(|a| a.simplify_once()).collect();
                // Constant-fold functions with a symbolic body; Extern stays opaque
                if let Some(body) = kind.body()
                    && new_args.iter().all(|a| matches!(a.as_ref(), Expr::Const(_))) {
                        let expanded = crate::expand_func(params, body, &new_args);
                        return expanded.simplify_once();
                    }
                // No arg changed: keep the original node so its Rc (and any
                // sharing of it, e.g. cached() rotation entries) survives.
                if new_args.iter().zip(args).all(|(n, o)| std::rc::Rc::ptr_eq(&n.0, &o.0)) {
                    return orig.clone();
                }
                E::new(Expr::Func {
                    name: name.clone(), params: params.clone(),
                    kind: kind.clone(), args: new_args,
                })
            }
        }
    }

    /// Expand products and integer powers over sums.
    ///
    /// Distributes multiplication: `(a + b) * c` becomes `a*c + b*c`.
    /// Integer powers up to 8 are expanded: `(a + b)^3` becomes the full
    /// multinomial expansion. The result is simplified afterwards.
    pub fn expand(&self) -> E {
        self.expand_inner().simplify()
    }

    fn expand_inner(&self) -> E {
        match self {
            Expr::Sym(_) | Expr::Const(_) | Expr::NamedConst { .. } => E::new(self.clone()),
            Expr::Neg(a) => E::new(Expr::Neg(a.expand_inner())),
            Expr::Add(a, b) => E::new(Expr::Add(a.expand_inner(), b.expand_inner())),
            Expr::Sub(a, b) => E::new(Expr::Sub(a.expand_inner(), b.expand_inner())),
            Expr::Mul(a, b) => {
                let a = a.expand_inner();
                let b = b.expand_inner();
                if let Expr::Add(b1, b2) = b.as_ref() {
                    let left = E::new(Expr::Mul(a.clone(), b1.clone()));
                    let right = E::new(Expr::Mul(a, b2.clone()));
                    return E::new(Expr::Add(left.expand_inner(), right.expand_inner()));
                }
                if let Expr::Sub(b1, b2) = b.as_ref() {
                    let left = E::new(Expr::Mul(a.clone(), b1.clone()));
                    let right = E::new(Expr::Mul(a, b2.clone()));
                    return E::new(Expr::Sub(left.expand_inner(), right.expand_inner()));
                }
                if let Expr::Add(a1, a2) = a.as_ref() {
                    let left = E::new(Expr::Mul(a1.clone(), b.clone()));
                    let right = E::new(Expr::Mul(a2.clone(), b));
                    return E::new(Expr::Add(left.expand_inner(), right.expand_inner()));
                }
                if let Expr::Sub(a1, a2) = a.as_ref() {
                    let left = E::new(Expr::Mul(a1.clone(), b.clone()));
                    let right = E::new(Expr::Mul(a2.clone(), b));
                    return E::new(Expr::Sub(left.expand_inner(), right.expand_inner()));
                }
                E::new(Expr::Mul(a, b))
            }
            Expr::Div(a, b) => E::new(Expr::Div(a.expand_inner(), b.expand_inner())),
            Expr::Pow(base, exp) => {
                let base = base.expand_inner();
                let exp = exp.expand_inner();
                if let Some(n) = is_const_int(&exp)
                    && (2..=8).contains(&n) {
                        let mut result = base.clone();
                        for _ in 1..n {
                            result = E::new(Expr::Mul(result, base.clone()));
                        }
                        return result.expand_inner();
                    }
                E::new(Expr::Pow(base, exp))
            }
            Expr::Sin(a) => E::new(Expr::Sin(a.expand_inner())),
            Expr::Cos(a) => E::new(Expr::Cos(a.expand_inner())),
            Expr::Tan(a) => E::new(Expr::Tan(a.expand_inner())),
            Expr::Asin(a) => E::new(Expr::Asin(a.expand_inner())),
            Expr::Acos(a) => E::new(Expr::Acos(a.expand_inner())),
            Expr::Atan(a) => E::new(Expr::Atan(a.expand_inner())),
            Expr::Atan2(y, x) => E::new(Expr::Atan2(y.expand_inner(), x.expand_inner())),
            Expr::Sinh(a) => E::new(Expr::Sinh(a.expand_inner())),
            Expr::Cosh(a) => E::new(Expr::Cosh(a.expand_inner())),
            Expr::Tanh(a) => E::new(Expr::Tanh(a.expand_inner())),
            Expr::Exp(a) => E::new(Expr::Exp(a.expand_inner())),
            Expr::Ln(a) => E::new(Expr::Ln(a.expand_inner())),
            Expr::Log2(a) => E::new(Expr::Log2(a.expand_inner())),
            Expr::Log10(a) => E::new(Expr::Log10(a.expand_inner())),
            Expr::Sqrt(a) => E::new(Expr::Sqrt(a.expand_inner())),
            Expr::Abs(a) => E::new(Expr::Abs(a.expand_inner())),
            Expr::Heaviside(a) => E::new(Expr::Heaviside(a.expand_inner())),
            Expr::Clamp(val, lo, hi) => E::new(Expr::Clamp(val.expand_inner(), lo.expand_inner(), hi.expand_inner())),
            Expr::Branch(q, a, b) => E::new(Expr::Branch(q.expand_inner(), a.expand_inner(), b.expand_inner())),
            Expr::Func { name, params, kind, args } => {
                let expanded_args: Vec<E> = args.iter().map(|a| a.expand_inner()).collect();
                if let Some(body) = kind.body() {
                    crate::expand_func(params, body, &expanded_args).expand_inner()
                } else {
                    E::new(Expr::Func {
                        name: name.clone(), params: params.clone(),
                        kind: kind.clone(), args: expanded_args,
                    })
                }
            }
        }
    }

    /// Collect like terms containing `var` by structural match.
    ///
    /// `var` can be any [`AsVarName`](crate::AsVarName) -- a `&str`, a `String`, or an
    /// [`E`] handle wrapping a `Sym` node. Groups additive terms
    /// that share `var` as a factor, summing their coefficients.
    /// For example, `a*x + b*x + c` becomes `(a + b)*x + c`.
    pub fn collect(&self, var: impl crate::AsVarName) -> E {
        let var = var.var_expr();
        let terms = flatten_add_simple(&E::new(self.clone()));
        let mut with_var: Vec<E> = Vec::new();
        let mut without_var: Vec<E> = Vec::new();

        for term in &terms {
            if let Some(coeff) = extract_factor(term, &var) {
                with_var.push(coeff);
            } else {
                without_var.push(term.clone());
            }
        }

        let mut result: Option<E> = None;

        if !with_var.is_empty() {
            let coeff_sum = sum_terms(with_var);
            let collected = coeff_sum * var;
            result = Some(collected);
        }

        for t in without_var {
            result = Some(match result {
                Some(acc) => acc + t,
                None => t,
            });
        }

        result.unwrap_or_else(|| constant(0.0))
    }
}

fn flatten_add_simple(e: &E) -> Vec<E> {
    match e.as_ref() {
        Expr::Add(a, b) => {
            let mut terms = flatten_add_simple(a);
            terms.extend(flatten_add_simple(b));
            terms
        }
        _ => vec![e.clone()],
    }
}

fn extract_factor(term: &E, var: &E) -> Option<E> {
    if term == var {
        return Some(constant(1.0));
    }
    if let Expr::Mul(a, b) = term.as_ref() {
        if b == var { return Some(a.clone()); }
        if a == var { return Some(b.clone()); }
    }
    None
}

fn sum_terms(terms: Vec<E>) -> E {
    let mut iter = terms.into_iter();
    let first = iter.next().unwrap();
    iter.fold(first, |acc, t| acc + t)
}
