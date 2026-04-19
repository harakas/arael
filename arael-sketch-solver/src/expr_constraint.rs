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

/// A constraint defined by a symbolic expression at runtime.
/// The residual is `expr * constraint_isigma`.
pub struct ExpressionConstraint {
    pub expr: E,
    pub param_derivs: Vec<(String, E)>,
    pub indices: Vec<u32>,
    pub description: String,
    /// Set by extended_jacobian64 to match the Jacobian row's constraint field.
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

#[cfg(test)]
mod tests {
    use super::*;
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
            kind: crate::DimensionKind::LineLength(l0),
            value: 10.0,
            offset: vect2d::new(0.0, 1.0),
            text_along: 0.0,
            name: "d0".into(),
            expr_str: None, broken: false, derived: false,
            range: None,
        });

        let mut params = Vec::new();
        sketch.serialize64(&mut params);
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
        sketch.serialize64(&mut params);
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
