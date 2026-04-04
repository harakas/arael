//! Symbol bag: runtime mapping from user-friendly names to parameter
//! indices and symbolic expressions. Used for parametric equations in
//! dimensions.

use std::collections::HashMap;
use arael_sym::E;
use crate::Sketch;

/// Runtime symbol table mapping user-visible names to parameter info.
pub struct SymbolBag {
    /// Direct parameters: name -> global parameter index.
    /// e.g. "L0.p1.x" -> 0, "L0.p1.y" -> 1, "L0.p2.x" -> 2, ...
    pub param_indices: HashMap<String, u32>,
    /// Dimension target values: name -> constant value.
    /// e.g. "d0" -> 5.0
    pub dim_values: HashMap<String, f64>,
    /// Derived properties: name -> symbolic expression in terms of
    /// direct parameter symbols. e.g. "L0.length" -> sqrt(...)
    pub derived: HashMap<String, E>,
}

impl SymbolBag {
    /// Build a symbol bag from the current sketch state.
    /// Must be called after `serialize64()` so parameter indices are assigned.
    pub fn build(sketch: &Sketch) -> Self {
        let mut param_indices = HashMap::new();
        let mut dim_values = HashMap::new();
        let mut derived = HashMap::new();

        // Points: P{n}.pos.x, P{n}.pos.y
        for r in sketch.points.refs() {
            let p = &sketch.points[r];
            let name = &p.name;
            if p.pos.optimize {
                let idx = p.pos.index();
                param_indices.insert(format!("{}.pos.x", name), idx);
                param_indices.insert(format!("{}.pos.y", name), idx + 1);
                param_indices.insert(format!("{}.x", name), idx);
                param_indices.insert(format!("{}.y", name), idx + 1);
            } else {
                dim_values.insert(format!("{}.pos.x", name), p.pos.value.x);
                dim_values.insert(format!("{}.pos.y", name), p.pos.value.y);
                dim_values.insert(format!("{}.x", name), p.pos.value.x);
                dim_values.insert(format!("{}.y", name), p.pos.value.y);
            }
        }

        // Lines: L{n}.p1.x, L{n}.p1.y, L{n}.p2.x, L{n}.p2.y
        for r in sketch.lines.refs() {
            let l = &sketch.lines[r];
            let name = &l.name;
            if l.p1.optimize {
                let idx = l.p1.index();
                param_indices.insert(format!("{}.p1.x", name), idx);
                param_indices.insert(format!("{}.p1.y", name), idx + 1);
            } else {
                dim_values.insert(format!("{}.p1.x", name), l.p1.value.x);
                dim_values.insert(format!("{}.p1.y", name), l.p1.value.y);
            }
            if l.p2.optimize {
                let idx = l.p2.index();
                param_indices.insert(format!("{}.p2.x", name), idx);
                param_indices.insert(format!("{}.p2.y", name), idx + 1);
            } else {
                dim_values.insert(format!("{}.p2.x", name), l.p2.value.x);
                dim_values.insert(format!("{}.p2.y", name), l.p2.value.y);
            }
            // Derived: L{n}.length, L{n}.angle
            let p1x = arael_sym::symbol(&format!("{}.p1.x", name));
            let p1y = arael_sym::symbol(&format!("{}.p1.y", name));
            let p2x = arael_sym::symbol(&format!("{}.p2.x", name));
            let p2y = arael_sym::symbol(&format!("{}.p2.y", name));
            let dx = p2x.clone() - p1x.clone();
            let dy = p2y.clone() - p1y.clone();
            derived.insert(format!("{}.length", name),
                arael_sym::sqrt(dx.clone() * dx.clone() + dy.clone() * dy.clone()));
            derived.insert(format!("{}.angle", name),
                arael_sym::atan2(dy, dx));
        }

        // Arcs: A{n}.center.x, A{n}.center.y, A{n}.radius, etc.
        for r in sketch.arcs.refs() {
            let a = &sketch.arcs[r];
            let name = &a.name;
            if a.center.optimize {
                let idx = a.center.index();
                param_indices.insert(format!("{}.center.x", name), idx);
                param_indices.insert(format!("{}.center.y", name), idx + 1);
            } else {
                dim_values.insert(format!("{}.center.x", name), a.center.value.x);
                dim_values.insert(format!("{}.center.y", name), a.center.value.y);
            }
            if a.radius.optimize {
                param_indices.insert(format!("{}.radius", name), a.radius.index());
            } else {
                dim_values.insert(format!("{}.radius", name), a.radius.value);
            }
            if a.start_angle.optimize {
                param_indices.insert(format!("{}.start_angle", name), a.start_angle.index());
            } else {
                dim_values.insert(format!("{}.start_angle", name), a.start_angle.value);
            }
            if a.end_angle.optimize {
                param_indices.insert(format!("{}.end_angle", name), a.end_angle.index());
            } else {
                dim_values.insert(format!("{}.end_angle", name), a.end_angle.value);
            }
            // Derived
            let r_sym = arael_sym::symbol(&format!("{}.radius", name));
            derived.insert(format!("{}.diameter", name), r_sym.clone() * arael_sym::constant(2.0));
            let sa_sym = arael_sym::symbol(&format!("{}.start_angle", name));
            let ea_sym = arael_sym::symbol(&format!("{}.end_angle", name));
            derived.insert(format!("{}.sweep", name),
                (ea_sym.clone() - sa_sym.clone()) * arael_sym::constant(180.0 / std::f64::consts::PI));
            // Arc endpoint positions: start.x/y, end.x/y
            let cx_sym = arael_sym::symbol(&format!("{}.center.x", name));
            let cy_sym = arael_sym::symbol(&format!("{}.center.y", name));
            derived.insert(format!("{}.start.x", name),
                cx_sym.clone() + r_sym.clone() * arael_sym::cos(sa_sym.clone()));
            derived.insert(format!("{}.start.y", name),
                cy_sym.clone() + r_sym.clone() * arael_sym::sin(sa_sym));
            derived.insert(format!("{}.end.x", name),
                cx_sym + r_sym.clone() * arael_sym::cos(ea_sym.clone()));
            derived.insert(format!("{}.end.y", name),
                cy_sym + r_sym * arael_sym::sin(ea_sym));
        }

        // Dimensions: d{n} -> target value or live expression
        for dim in &sketch.dimensions {
            if dim.broken {
                // Broken expression: expose as constant with frozen value
                dim_values.insert(dim.name.clone(), dim.value);
            } else if let Some(ref expr_str) = dim.expr_str {
                // Expression dimension: resolve to live symbolic expression
                if let Ok(parsed) = arael_sym::parse(expr_str) {
                    derived.insert(dim.name.clone(), parsed);
                } else {
                    dim_values.insert(dim.name.clone(), dim.value);
                }
            } else {
                dim_values.insert(dim.name.clone(), dim.value);
            }
        }

        // User parameters: name -> constant or live expression
        for p in &sketch.user_params {
            if p.broken {
                dim_values.insert(p.name.clone(), p.value);
            } else if let Ok(parsed) = arael_sym::parse(&p.expr_str) {
                // If it's a plain numeric literal, store as constant
                if p.expr_str.trim().parse::<f64>().is_ok() {
                    dim_values.insert(p.name.clone(), p.value);
                } else {
                    derived.insert(p.name.clone(), parsed);
                }
            } else {
                dim_values.insert(p.name.clone(), p.value);
            }
        }

        SymbolBag { param_indices, dim_values, derived }
    }

    /// Resolve a symbol name to a symbolic expression.
    /// Returns an `E` that is either:
    /// - A direct parameter symbol (for param_indices entries)
    /// - A constant (for dim_values entries)
    /// - A derived expression (for derived entries)
    /// - None if not found
    pub fn resolve(&self, name: &str) -> Option<E> {
        if self.param_indices.contains_key(name) {
            return Some(arael_sym::symbol(name));
        }
        if let Some(&val) = self.dim_values.get(name) {
            return Some(arael_sym::constant(val));
        }
        if let Some(expr) = self.derived.get(name) {
            return Some(expr.clone());
        }
        None
    }

    /// Get current parameter values as a HashMap for E::eval().
    /// `params` is the flat parameter vector from the solver.
    pub fn eval_vars<'a>(&'a self, params: &[f64]) -> HashMap<&'a str, f64> {
        let mut vars = HashMap::new();
        for (name, &idx) in &self.param_indices {
            if idx != u32::MAX {
                vars.insert(name.as_str(), params[idx as usize]);
            }
        }
        // Add dimension values as constants
        for (name, &val) in &self.dim_values {
            vars.insert(name.as_str(), val);
        }
        vars
    }

    /// Get the list of direct parameter symbols referenced by an expression.
    /// Returns (symbol_name, parameter_index) pairs.
    pub fn referenced_params(&self, expr: &E) -> Vec<(String, u32)> {
        let symbols = expr.symbols();
        let mut result = Vec::new();
        for sym in symbols {
            if let Some(&idx) = self.param_indices.get(&sym) {
                result.push((sym, idx));
            }
            // Dimension refs and derived don't have param indices
            // (they're constants or expanded expressions)
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arael::vect::vect2d;

    #[test]
    fn test_symbol_bag_basic() {
        let mut sketch = Sketch::new();
        let _l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 4.0));
        let _l1 = sketch.add_line(vect2d::new(1.0, 1.0), vect2d::new(5.0, 2.0));

        // Must serialize to assign indices
        let mut params = Vec::new();
        sketch.serialize64(&mut params);

        let bag = SymbolBag::build(&sketch);

        // Check direct params exist
        assert!(bag.param_indices.contains_key("L0.p1.x"));
        assert!(bag.param_indices.contains_key("L0.p1.y"));
        assert!(bag.param_indices.contains_key("L0.p2.x"));
        assert!(bag.param_indices.contains_key("L1.p1.x"));

        // Check derived properties
        assert!(bag.derived.contains_key("L0.length"));
        assert!(bag.derived.contains_key("L0.angle"));

        // Check param index ordering
        let idx_p1x = bag.param_indices["L0.p1.x"];
        let idx_p1y = bag.param_indices["L0.p1.y"];
        assert_eq!(idx_p1y, idx_p1x + 1);

        // Evaluate L0.length using current params
        let vars = bag.eval_vars(&params);
        let length_expr = bag.derived.get("L0.length").unwrap();
        let length = length_expr.eval(&vars).unwrap();
        assert!((length - 5.0).abs() < 0.01, "L0 length should be 5, got {}", length);
    }

    #[test]
    fn test_symbol_bag_resolve() {
        let mut sketch = Sketch::new();
        sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 4.0));

        let mut params = Vec::new();
        sketch.serialize64(&mut params);
        let bag = SymbolBag::build(&sketch);

        // Direct param resolves to symbol
        let resolved = bag.resolve("L0.p1.x");
        assert!(resolved.is_some());

        // Derived resolves to expression
        let resolved = bag.resolve("L0.length");
        assert!(resolved.is_some());

        // Unknown returns None
        assert!(bag.resolve("L99.p1.x").is_none());
    }

    #[test]
    fn test_symbol_bag_arc_endpoints() {
        let mut sketch = Sketch::new();
        // Arc centered at (1,2), radius 3, from 0 to PI/2
        let sa = 0.0f64;
        let ea = std::f64::consts::FRAC_PI_2;
        sketch.add_arc(vect2d::new(1.0, 2.0), 3.0, sa, ea, false);

        let mut params = Vec::new();
        sketch.serialize64(&mut params);
        let bag = SymbolBag::build(&sketch);

        // Check derived endpoint symbols exist
        assert!(bag.derived.contains_key("A0.start.x"), "missing A0.start.x");
        assert!(bag.derived.contains_key("A0.start.y"), "missing A0.start.y");
        assert!(bag.derived.contains_key("A0.end.x"), "missing A0.end.x");
        assert!(bag.derived.contains_key("A0.end.y"), "missing A0.end.y");

        // Evaluate and check values
        let vars = bag.eval_vars(&params);
        let a = &sketch.arcs[sketch.arcs.refs().next().unwrap()];
        let expected_sx = a.center.value.x + a.radius.value * a.start_angle.value.cos();
        let expected_sy = a.center.value.y + a.radius.value * a.start_angle.value.sin();
        let expected_ex = a.center.value.x + a.radius.value * a.end_angle.value.cos();
        let expected_ey = a.center.value.y + a.radius.value * a.end_angle.value.sin();

        let sx = bag.derived["A0.start.x"].eval(&vars).unwrap();
        let sy = bag.derived["A0.start.y"].eval(&vars).unwrap();
        let ex = bag.derived["A0.end.x"].eval(&vars).unwrap();
        let ey = bag.derived["A0.end.y"].eval(&vars).unwrap();

        assert!((sx - expected_sx).abs() < 0.01, "start.x: expected {}, got {}", expected_sx, sx);
        assert!((sy - expected_sy).abs() < 0.01, "start.y: expected {}, got {}", expected_sy, sy);
        assert!((ex - expected_ex).abs() < 0.01, "end.x: expected {}, got {}", expected_ex, ex);
        assert!((ey - expected_ey).abs() < 0.01, "end.y: expected {}, got {}", expected_ey, ey);
    }
}
