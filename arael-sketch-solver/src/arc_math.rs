// Rotated-ellipse parametrisation: the single source for the arc/ellipse
// point and tangent formulas. Numeric callers use the `Arc` methods or the
// raw fns; constraint DSL bodies and expression builders use the
// `#[arael::function]` symbolic forms below.
//
// This file is `include!`d into lib.rs before constraints.rs so the
// `#[arael::function]` registrations expand before the constraint bodies
// that call them.

/// Point on a rotated ellipse at parametric angle `t`.
pub fn ellipse_point(center: vect2d, rx: f64, ry: f64, rot: f64, t: f64) -> vect2d {
    let ct = t.cos();
    let st = t.sin();
    let cr = rot.cos();
    let sr = rot.sin();
    vect2d::new(
        center.x + rx * ct * cr - ry * st * sr,
        center.y + rx * ct * sr + ry * st * cr,
    )
}

/// Tangent direction of a rotated ellipse at parametric angle `t`
/// (derivative of [`ellipse_point`] with respect to `t`).
pub fn ellipse_tangent(rx: f64, ry: f64, rot: f64, t: f64) -> vect2d {
    let ct = t.cos();
    let st = t.sin();
    let cr = rot.cos();
    let sr = rot.sin();
    vect2d::new(
        -rx * st * cr - ry * ct * sr,
        -rx * st * sr + ry * ct * cr,
    )
}

impl Arc {
    /// Point on the arc/ellipse at parametric angle `t`.
    pub fn point_at(&self, t: f64) -> vect2d {
        ellipse_point(self.center.value, self.radius.value, self.radius_b.value,
                      self.rotation.value, t)
    }

    /// Tangent direction at parametric angle `t`.
    pub fn tangent_at(&self, t: f64) -> vect2d {
        ellipse_tangent(self.radius.value, self.radius_b.value, self.rotation.value, t)
    }

    /// Position of the arc's start endpoint.
    pub fn start_pos(&self) -> vect2d {
        self.point_at(self.start_angle.value)
    }

    /// Position of the arc's end endpoint.
    pub fn end_pos(&self) -> vect2d {
        self.point_at(self.end_angle.value)
    }
}

/// x of the ellipse point -- symbolic twin of [`ellipse_point`].
#[arael::function]
pub fn arc_point_x(cx: arael_sym::E, r: arael_sym::E, rb: arael_sym::E,
                   rot: arael_sym::E, t: arael_sym::E) -> arael_sym::E {
    cx + r * cos(t) * cos(rot) - rb * sin(t) * sin(rot)
}

/// y of the ellipse point -- symbolic twin of [`ellipse_point`].
#[arael::function]
pub fn arc_point_y(cy: arael_sym::E, r: arael_sym::E, rb: arael_sym::E,
                   rot: arael_sym::E, t: arael_sym::E) -> arael_sym::E {
    cy + r * cos(t) * sin(rot) + rb * sin(t) * cos(rot)
}

/// x of the ellipse tangent -- symbolic twin of [`ellipse_tangent`].
#[arael::function]
pub fn arc_tangent_x(r: arael_sym::E, rb: arael_sym::E,
                     rot: arael_sym::E, t: arael_sym::E) -> arael_sym::E {
    -r * sin(t) * cos(rot) - rb * cos(t) * sin(rot)
}

/// y of the ellipse tangent -- symbolic twin of [`ellipse_tangent`].
#[arael::function]
pub fn arc_tangent_y(r: arael_sym::E, rb: arael_sym::E,
                     rot: arael_sym::E, t: arael_sym::E) -> arael_sym::E {
    -r * sin(t) * sin(rot) + rb * cos(t) * cos(rot)
}

/// Symbolic (x, y) of an arc/ellipse point at the given angle expression,
/// built from the arc's canonical parameter symbol names.
pub fn arc_endpoint_symbols(arc_name: &str, angle: arael_sym::E) -> (arael_sym::E, arael_sym::E) {
    let sym = |field: &str| arael_sym::symbol(&format!("{}.{}", arc_name, field));
    let (r, rb, rot) = (sym("radius"), sym("radius_b"), sym("rotation"));
    (arc_point_x(sym("center.x"), r.clone(), rb.clone(), rot.clone(), angle.clone()),
     arc_point_y(sym("center.y"), r, rb, rot, angle))
}

#[cfg(test)]
mod arc_math_tests {
    use super::*;

    fn test_sketch() -> Sketch {
        let mut sketch = Sketch::new();
        sketch.add_ellipse(vect2d::new(1.5, -2.0), 3.0, 1.2, 0.7, false);
        sketch
    }

    #[test]
    fn test_point_at_matches_raw() {
        let s = test_sketch();
        let a = &s.arcs[s.arcs.refs().next().unwrap()];
        for i in 0..12 {
            let t = i as f64 * 0.55;
            let p = a.point_at(t);
            let q = ellipse_point(a.center.value, a.radius.value, a.radius_b.value,
                                  a.rotation.value, t);
            assert!((p.x - q.x).abs() < 1e-12 && (p.y - q.y).abs() < 1e-12);
        }
    }

    #[test]
    fn test_tangent_is_point_derivative() {
        let s = test_sketch();
        let a = &s.arcs[s.arcs.refs().next().unwrap()];
        let h = 1e-6;
        for i in 0..12 {
            let t = i as f64 * 0.55;
            let tv = a.tangent_at(t);
            let pp = a.point_at(t + h);
            let pm = a.point_at(t - h);
            let nx = (pp.x - pm.x) / (2.0 * h);
            let ny = (pp.y - pm.y) / (2.0 * h);
            assert!((tv.x - nx).abs() < 1e-6, "tangent x at t={}: {} vs {}", t, tv.x, nx);
            assert!((tv.y - ny).abs() < 1e-6, "tangent y at t={}: {} vs {}", t, tv.y, ny);
        }
    }

    #[test]
    fn test_symbolic_matches_numeric() {
        let (cx, cy, r, rb, rot, t) = (1.5, -2.0, 3.0, 1.2, 0.7, 2.3);
        let sym = |n: &str| arael_sym::symbol(n);
        let ex = arc_point_x(sym("cx"), sym("r"), sym("rb"), sym("rot"), sym("t"));
        let ey = arc_point_y(sym("cy"), sym("r"), sym("rb"), sym("rot"), sym("t"));
        let tx = arc_tangent_x(sym("r"), sym("rb"), sym("rot"), sym("t"));
        let ty = arc_tangent_y(sym("r"), sym("rb"), sym("rot"), sym("t"));
        let mut vars = std::collections::HashMap::new();
        vars.insert("cx", cx);
        vars.insert("cy", cy);
        vars.insert("r", r);
        vars.insert("rb", rb);
        vars.insert("rot", rot);
        vars.insert("t", t);
        let p = ellipse_point(vect2d::new(cx, cy), r, rb, rot, t);
        let tv = ellipse_tangent(r, rb, rot, t);
        assert!((ex.eval(&vars).unwrap() - p.x).abs() < 1e-12);
        assert!((ey.eval(&vars).unwrap() - p.y).abs() < 1e-12);
        assert!((tx.eval(&vars).unwrap() - tv.x).abs() < 1e-12);
        assert!((ty.eval(&vars).unwrap() - tv.y).abs() < 1e-12);
    }
}
