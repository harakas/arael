//! Expression-based constraints for parametric dimensions.
//!
//! An ExpressionConstraint holds a symbolic expression (arael_sym::E)
//! and its pre-computed symbolic derivatives. At each solver iteration,
//! it evaluates the expression and derivatives numerically, accumulating
//! into a TripletBlock.

use std::collections::HashMap;
use arael_sym::E;
use arael::model::TripletBlock;  // used in compute() parameter
use crate::symbol_bag::SymbolBag;
use crate::{RangeBound, RangeValue};

/// A constraint defined by a symbolic expression at runtime.
/// The residual is `expr * constraint_isigma`.
pub struct ExpressionConstraint {
    pub expr: E,
    pub param_derivs: Vec<(String, E)>,
    pub indices: Vec<u32>,
    pub description: String,
    /// Set by extended_jacobian to match the Jacobian row's constraint field.
    pub cid: u32,
    /// Static label for JacobianRow (always "dimension" for dimension-derived exprs).
    pub label: &'static str,
}

impl ExpressionConstraint {
    /// Create an unresolved expression constraint (symbols not yet mapped
    /// to parameter indices). Call `resolve()` with a SymbolBag before solving.
    pub fn new_unresolved(expr: E, description: String) -> Self {
        ExpressionConstraint {
            expr,
            param_derivs: Vec::new(),
            indices: Vec::new(),
            description,
            cid: 0,
            label: "dimension",
        }
    }

    /// Resolve symbols: expand derived properties, compute derivatives,
    /// and map symbol names to parameter indices.
    pub fn resolve(&mut self, bag: &SymbolBag) {
        let expanded = expand_derived(&self.expr, bag);
        let all_symbols = expanded.symbols();
        let mut param_derivs = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for sym in &all_symbols {
            if bag.param_indices.contains_key(sym.as_str())
                && seen.insert(sym.clone()) {
                    let deriv = expanded.diff(sym.clone());
                    param_derivs.push((sym.clone(), deriv));
                }
        }
        self.expr = expanded;
        self.param_derivs = param_derivs;
        self.indices = self.param_derivs.iter()
            .map(|(name, _)| bag.param_indices.get(name).copied().unwrap_or(u32::MAX))
            .collect();
    }

    /// Compute the residual and derivatives, write grad directly into the
    /// global `grad` vector and push cross pairs into the shared
    /// TripletBlock. Each param symbol is treated as its own "entity"
    /// (per-param span boundaries), so every pair is cross → full upper
    /// triangle stored in the TripletBlock.
    pub fn compute(&self, vars: &HashMap<&str, f64>, constraint_isigma: f64,
                   hb: &mut TripletBlock<f64>, grad: &mut [f64]) -> Result<(), String> {
        let r = self.expr.eval(vars)? * constraint_isigma;
        let dr: Vec<f64> = self.param_derivs.iter()
            .map(|(_, deriv)| Ok::<_, String>(deriv.eval(vars)? * constraint_isigma))
            .collect::<Result<_, _>>()?;
        hb.add_residual(r, &self.indices, &dr, grad);
        Ok(())
    }

    /// Compute the squared residual (cost contribution).
    pub fn cost(&self, vars: &HashMap<&str, f64>, constraint_isigma: f64) -> Result<f64, String> {
        let r = self.expr.eval(vars)? * constraint_isigma;
        Ok(r * r)
    }

    /// Compute residual and sparse derivative entries for a Jacobian row.
    pub fn jacobian_row(&self, vars: &HashMap<&str, f64>, constraint_isigma: f64)
        -> Result<(f64, Vec<(u32, f64)>), String>
    {
        let r = self.expr.eval(vars)? * constraint_isigma;
        let dr: Result<Vec<f64>, String> = self.param_derivs.iter()
            .map(|(_, deriv)| Ok(deriv.eval(vars)? * constraint_isigma))
            .collect();
        Ok((r, arael::model::jacobian_entries(&self.indices, &dr?)))
    }
}

/// Expand derived symbols in an expression to their base parameter form.
/// E.g. if expr contains symbol "L0.length", replace it with
/// sqrt((L0.p2.x - L0.p1.x)^2 + (L0.p2.y - L0.p1.y)^2).
pub fn expand_derived(expr: &E, bag: &SymbolBag) -> E {
    let mut result = expr.clone();
    // Iterate until no more derived/dim symbols remain (max 16 to prevent infinite loops)
    for _ in 0..16 {
        let symbols = result.symbols();
        let mut substitutions: Vec<(E, E)> = Vec::new();
        for sym in &symbols {
            if let Some(expansion) = bag.derived.get(sym.as_str()) {
                substitutions.push((arael_sym::symbol(sym), expansion.clone()));
            }
            if let Some(&val) = bag.dim_values.get(sym.as_str()) {
                substitutions.push((arael_sym::symbol(sym), arael_sym::constant(val)));
            }
        }
        if substitutions.is_empty() { break; }
        result = result.substitute(&substitutions);
    }
    result
}

/// Result of a token-level symbol rewrite over one expression string.
pub struct ExprRewrite {
    pub text: String,
    /// (from, to) pairs actually substituted, in source order.
    pub edits: Vec<(String, String)>,
    /// Tokens that reference the target entity but had no mapping
    /// (e.g. `L0.length` after a trim removed a piece). The caller
    /// marks the owning dimension/parameter broken.
    pub unresolved: Vec<String>,
}

/// Rewrite entity-name symbols in an expression string.
///
/// Scans maximal dotted identifier tokens (`L0`, `L0.length`,
/// `L0.p1.x`) and replaces each token found in `map`, leaving all
/// other text -- spacing, numbers, operators -- untouched. A token
/// equal to `target_name` or starting with `target_name + "."` that
/// has no mapping is reported in `unresolved` and left as-is.
pub fn rewrite_expr_symbols(
    src: &str,
    map: &std::collections::HashMap<String, String>,
    target_name: &str,
) -> ExprRewrite {
    let bytes = src.as_bytes();
    let mut out = String::with_capacity(src.len());
    let mut edits = Vec::new();
    let mut unresolved = Vec::new();
    let is_ident_start = |b: u8| b.is_ascii_alphabetic() || b == b'_';
    let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let prefix = format!("{}.", target_name);
    let mut i = 0;
    while i < bytes.len() {
        if is_ident_start(bytes[i]) {
            let start = i;
            i += 1;
            while i < bytes.len() && is_ident(bytes[i]) { i += 1; }
            // Extend across `.ident` segments to the longest dotted chain.
            while i + 1 < bytes.len() && bytes[i] == b'.' && is_ident_start(bytes[i + 1]) {
                i += 2;
                while i < bytes.len() && is_ident(bytes[i]) { i += 1; }
            }
            let token = &src[start..i];
            if let Some(repl) = map.get(token) {
                edits.push((token.to_string(), repl.clone()));
                out.push_str(repl);
            } else {
                if token == target_name || token.starts_with(&prefix) {
                    unresolved.push(token.to_string());
                }
                out.push_str(token);
            }
        } else {
            // Copy the run up to the next identifier start as a slice:
            // UTF-8 safe (identifier starts are ASCII, so the slice
            // boundaries are char boundaries).
            let start = i;
            i += 1;
            while i < bytes.len() && !is_ident_start(bytes[i]) { i += 1; }
            out.push_str(&src[start..i]);
        }
    }
    ExprRewrite { text: out, edits, unresolved }
}

impl crate::Sketch {
    /// Rewrite every stored expression string -- dimension expressions,
    /// live range bounds, user parameters -- through
    /// [`rewrite_expr_symbols`]. Owners of an expression that still
    /// references `target_name` unresolvably are marked broken.
    /// Returns human-readable report lines, one per edit or break.
    pub fn rewrite_expression_symbols(
        &mut self,
        map: &std::collections::HashMap<String, String>,
        target_name: &str,
    ) -> Vec<String> {
        let mut report = Vec::new();
        let rewrite_one = |owner: &str, src: &mut String, broken: &mut bool,
                               report: &mut Vec<String>| {
            let r = rewrite_expr_symbols(src, map, target_name);
            if !r.edits.is_empty() {
                report.push(format!("{} \"{}\" -> \"{}\"", owner, src, r.text));
                *src = r.text;
            }
            if !r.unresolved.is_empty() {
                *broken = true;
                report.push(format!("{} marked broken (references {})",
                    owner, r.unresolved.join(", ")));
            }
        };
        for dim in &mut self.dimensions {
            let name = dim.name.clone();
            if let Some(expr) = dim.expr_str.as_mut() {
                rewrite_one(&name, expr, &mut dim.broken, &mut report);
            }
            if let Some(range) = dim.range.as_mut() {
                let mut slots: Vec<&mut RangeValue> = match range {
                    RangeBound::Min(v) | RangeBound::Max(v) => vec![v],
                    RangeBound::Between(lo, hi) => vec![lo, hi],
                };
                for v in slots.iter_mut() {
                    if let RangeValue::Live(src) = v {
                        rewrite_one(&name, src, &mut dim.broken, &mut report);
                    }
                }
            }
        }
        for p in &mut self.user_params {
            let name = p.name.clone();
            rewrite_one(&name, &mut p.expr_str, &mut p.broken, &mut report);
        }
        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arael::simple_lm::RootProblem; // Sketch::serialize
    use arael::vect::vect2d;
    use crate::Sketch;

    #[test]
    fn test_expression_constraint_constant_dim() {
        // Set up: two lines with a length dimension d0=10
        let mut sketch = Sketch::new();
        let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
        let _l1 = sketch.add_line(vect2d::new(5.0, 0.0), vect2d::new(8.0, 0.0));
        sketch.lines[l0].constraints.has_length = true;
        sketch.lines[l0].constraints.length = 10.0;
        sketch.dimensions.push(crate::Dimension {
            did: 0,
            kind: crate::DimensionKind::LineLength(l0),
            value: 10.0,
            offset: vect2d::new(0.0, 1.0),
            text_along: 0.0,
            name: "d0".into(),
            expr_str: None, broken: false, derived: false,
            range: None,
        });

        let mut params = Vec::new();
        sketch.serialize(&mut params);
        let bag = SymbolBag::build(&sketch);

        // Create expression constraint: L1.length - d0 = 0
        let expr = arael_sym::symbol("L1.length") - arael_sym::symbol("d0");
        let mut ec = ExpressionConstraint::new_unresolved(expr, "L1.length = d0".into());
        ec.resolve(&bag);

        // Check that it references L1's parameters
        assert!(!ec.indices.is_empty(), "should reference L1 params");
        assert!(ec.param_derivs.iter().any(|(name, _): &(String, _)| name.starts_with("L1.")),
            "should have L1 derivatives");

        // Evaluate: L1 length is 3, d0 is 10, so residual should be 3-10 = -7
        let vars = bag.eval_vars(&params);
        let r = ec.expr.eval(&vars).unwrap();
        assert!((r - (-7.0)).abs() < 0.01, "residual should be -7, got {}", r);
    }

    #[test]
    fn test_expression_constraint_derived_property() {
        let mut sketch = Sketch::new();
        sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 4.0));

        let mut params = Vec::new();
        sketch.serialize(&mut params);
        let bag = SymbolBag::build(&sketch);

        // L0.length should be 5
        let length_expr = bag.resolve("L0.length").unwrap();
        let vars = bag.eval_vars(&params);
        let length = length_expr.eval(&vars).unwrap();
        assert!((length - 5.0).abs() < 0.01);

        // Create constraint: L0.length - 7 = 0 (want length to be 7)
        let expr = arael_sym::symbol("L0.length") - arael_sym::constant(7.0);
        let mut ec = ExpressionConstraint::new_unresolved(expr, "L0.length = 7".into());
        ec.resolve(&bag);

        // Should have derivatives w.r.t. L0.p1.x, L0.p1.y, L0.p2.x, L0.p2.y
        assert_eq!(ec.param_derivs.len(), 4,
            "should have 4 derivatives, got {}: {:?}",
            ec.param_derivs.len(),
            ec.param_derivs.iter().map(|(n, _): &(String, _)| n.as_str()).collect::<Vec<_>>());
    }
}

#[cfg(test)]
mod rewrite_tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
    }

    #[test]
    fn test_rewrite_endpoints_and_sum() {
        let m = map(&[
            ("L0.p1.x", "L5.p1.x"),
            ("L0.p2.x", "L6.p2.x"),
            ("L0.length", "(L5.length + L6.length)"),
        ]);
        let r = rewrite_expr_symbols("L0.length * 2 + L0.p1.x", &m, "L0");
        assert_eq!(r.text, "(L5.length + L6.length) * 2 + L0.p1.x".replace("L0.p1.x", "L5.p1.x"));
        assert_eq!(r.edits.len(), 2);
        assert!(r.unresolved.is_empty());
    }

    #[test]
    fn test_rewrite_no_partial_name_match() {
        // L1 must not match inside L10; L0.p1 must not eat L0.p1.x.
        let m = map(&[("L1.length", "L9.length")]);
        let r = rewrite_expr_symbols("L10.length + L1.length", &m, "L1");
        assert_eq!(r.text, "L10.length + L9.length");
        assert_eq!(r.edits.len(), 1);
        assert!(r.unresolved.is_empty());
    }

    #[test]
    fn test_rewrite_unresolved_reports() {
        let m = map(&[("A0.radius", "A5.radius")]);
        let r = rewrite_expr_symbols("A0.sweep + A0.radius", &m, "A0");
        assert_eq!(r.text, "A0.sweep + A5.radius");
        assert_eq!(r.unresolved, vec!["A0.sweep".to_string()]);
    }

    #[test]
    fn test_rewrite_preserves_numbers_and_spacing() {
        let m = map(&[("d0", "d9")]);
        let r = rewrite_expr_symbols("  2.5*d0 +sqrt( d0 )", &m, "L0");
        assert_eq!(r.text, "  2.5*d9 +sqrt( d9 )");
    }

    #[test]
    fn test_sketch_rewrite_marks_broken() {
        use arael::vect::vect2d;
        let mut s = crate::Sketch::new();
        let l0 = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
        let _l1 = s.add_line(vect2d::new(0.0, 1.0), vect2d::new(3.0, 1.0));
        // A user param referencing L0.length with no mapping for it.
        s.user_params.push(crate::UserParam {
            name: "w".into(), expr_str: "L0.length * 2".into(),
            value: 6.0, broken: false,
        });
        // An expression dimension on L1 referencing L0.p1.x with a mapping.
        s.dimensions.push(crate::Dimension {
            did: 0, kind: crate::DimensionKind::LineLength(_l1),
            value: 3.0, offset: vect2d::new(0.0, 1.0), text_along: 0.0,
            name: "d0".into(), expr_str: Some("L0.p1.x + 5".into()),
            broken: false, derived: false, range: None,
        });
        let _ = l0;
        let m = map(&[("L0.p1.x", "L5.p1.x")]);
        let report = s.rewrite_expression_symbols(&m, "L0");
        assert_eq!(s.dimensions[0].expr_str.as_deref(), Some("L5.p1.x + 5"));
        assert!(!s.dimensions[0].broken);
        assert!(s.user_params[0].broken, "param referencing unmapped L0.length must break");
        assert_eq!(s.user_params[0].expr_str, "L0.length * 2");
        assert_eq!(report.len(), 2, "one edit line, one broken line: {:?}", report);
    }
}
