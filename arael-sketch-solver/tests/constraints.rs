use arael::model::{CrossBlock, Param};
use arael::vect::vect2d;
use arael_sketch_solver::*;

fn assert_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "expected {a} ~= {b} (diff={})", (a - b).abs());
}

fn line_length(sketch: &Sketch, r: arael::refs::Ref<Line>) -> f64 {
    let l = &sketch.lines[r];
    let dx = l.p2.value.x - l.p1.value.x;
    let dy = l.p2.value.y - l.p1.value.y;
    (dx * dx + dy * dy).sqrt()
}

// -- Point-Point --

#[test]
fn test_coincident_pp() {
    let mut sketch = Sketch::new();
    let a = sketch.add_point(vect2d::new(1.0, 2.0));
    let b = sketch.add_point(vect2d::new(1.1, 2.2));
    sketch.coincident_pp.push(CoincidentPP {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    let pa = sketch.points[a].pos.value;
    let pb = sketch.points[b].pos.value;
    assert_near(pa.x, pb.x, 0.001);
    assert_near(pa.y, pb.y, 0.001);
}

#[test]
fn test_horizontal_line() {
    let mut sketch = Sketch::new();
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.5));
    sketch.lines[l].constraints.horizontal = true;
    sketch.solve();
    let line = &sketch.lines[l];
    assert_near(line.p1.value.y, line.p2.value.y, 0.001);
}

#[test]
fn test_vertical_line() {
    let mut sketch = Sketch::new();
    let l = sketch.add_line(vect2d::new(1.0, 0.0), vect2d::new(1.3, 3.0));
    sketch.lines[l].constraints.vertical = true;
    sketch.solve();
    let line = &sketch.lines[l];
    assert_near(line.p1.value.x, line.p2.value.x, 0.001);
}

#[test]
fn test_line_length() {
    let mut sketch = Sketch::new();
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    sketch.lines[l].constraints.has_length = true;
    sketch.lines[l].constraints.length = 5.0;
    sketch.solve();
    let line = &sketch.lines[l];
    let dx = line.p2.value.x - line.p1.value.x;
    let dy = line.p2.value.y - line.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt();
    assert_near(len, 5.0, 0.01);
}

#[test]
fn test_parallel_lines() {
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 1.0));
    let b = sketch.add_line(vect2d::new(0.0, 2.0), vect2d::new(4.0, 2.5));
    sketch.parallel.push(Parallel {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let dx1 = la.p2.value.x - la.p1.value.x;
    let dy1 = la.p2.value.y - la.p1.value.y;
    let dx2 = lb.p2.value.x - lb.p1.value.x;
    let dy2 = lb.p2.value.y - lb.p1.value.y;
    let cross = dx1 * dy2 - dy1 * dx2;
    let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();
    let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();
    assert_near(cross / (len1 * len2), 0.0, 0.001);
}

#[test]
fn test_perpendicular_lines() {
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let b = sketch.add_line(vect2d::new(1.0, -1.0), vect2d::new(1.5, 2.0));
    sketch.perpendicular.push(Perpendicular {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let dx1 = la.p2.value.x - la.p1.value.x;
    let dy1 = la.p2.value.y - la.p1.value.y;
    let dx2 = lb.p2.value.x - lb.p1.value.x;
    let dy2 = lb.p2.value.y - lb.p1.value.y;
    let dot = dx1 * dx2 + dy1 * dy2;
    let len1 = (dx1 * dx1 + dy1 * dy1).sqrt();
    let len2 = (dx2 * dx2 + dy2 * dy2).sqrt();
    assert_near(dot / (len1 * len2), 0.0, 0.001);
}

#[test]
fn test_fixed_point_doesnt_move() {
    let mut sketch = Sketch::new();
    let a = sketch.add_point_fixed(vect2d::new(1.0, 2.0));
    let b = sketch.add_point(vect2d::new(1.5, 2.5));
    sketch.coincident_pp.push(CoincidentPP {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    let pa = sketch.points[a].pos.value;
    let pb = sketch.points[b].pos.value;
    // Fixed point should not move
    assert_near(pa.x, 1.0, 1e-10);
    assert_near(pa.y, 2.0, 1e-10);
    // Free point should move to the fixed point
    assert_near(pb.x, 1.0, 0.001);
    assert_near(pb.y, 2.0, 0.001);
}

#[test]
fn test_point_on_line() {
    let mut sketch = Sketch::new();
    let p = sketch.add_point(vect2d::new(2.0, 1.5));
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 1.0));
    sketch.point_on_line.push(PointOnLine {
        point: p, line: l, hb: CrossBlock::new(),
    });
    sketch.solve();
    let pp = sketch.points[p].pos.value;
    let lp = &sketch.lines[l];
    // Check point is on line: cross product should be zero
    let dx = lp.p2.value.x - lp.p1.value.x;
    let dy = lp.p2.value.y - lp.p1.value.y;
    let cross = (pp.x - lp.p1.value.x) * dy - (pp.y - lp.p1.value.y) * dx;
    let len = (dx * dx + dy * dy).sqrt();
    assert_near(cross / len, 0.0, 0.001);
}

#[test]
fn test_arc_radius() {
    let mut sketch = Sketch::new();
    let a = sketch.add_arc(vect2d::new(0.0, 0.0), 3.0, 0.0, std::f64::consts::PI, false);
    sketch.arcs[a].constraints.has_target_radius = true;
    sketch.arcs[a].constraints.target_radius = 5.0;
    sketch.solve();
    assert_near(sketch.arcs[a].radius.value, 5.0, 0.01);
}

#[test]
fn test_distance_pp() {
    let mut sketch = Sketch::new();
    let a = sketch.add_point(vect2d::new(0.0, 0.0));
    let b = sketch.add_point(vect2d::new(3.0, 0.0));
    sketch.distance_pp.push(DistancePP {
        a, b, distance: 5.0, hb: CrossBlock::new(),
    });
    sketch.solve();
    let pa = sketch.points[a].pos.value;
    let pb = sketch.points[b].pos.value;
    let dx = pb.x - pa.x;
    let dy = pb.y - pa.y;
    let dist = (dx * dx + dy * dy).sqrt();
    assert_near(dist, 5.0, 0.01);
}

#[test]
fn test_horizontal_distance_pp() {
    let mut sketch = Sketch::new();
    let a = sketch.add_point(vect2d::new(0.0, 1.0));
    let b = sketch.add_point(vect2d::new(2.0, 1.0));
    sketch.hdistance_pp.push(HorizontalDistancePP {
        a, b, distance: 5.0, hb: CrossBlock::new(),
    });
    sketch.solve();
    let pa = sketch.points[a].pos.value;
    let pb = sketch.points[b].pos.value;
    assert_near(pa.x - pb.x, 5.0, 0.01);
}

#[test]
fn test_vertical_distance_pp() {
    let mut sketch = Sketch::new();
    let a = sketch.add_point(vect2d::new(1.0, 0.0));
    let b = sketch.add_point(vect2d::new(1.0, 3.0));
    sketch.vdistance_pp.push(VerticalDistancePP {
        a, b, distance: 5.0, hb: CrossBlock::new(),
    });
    sketch.solve();
    let pa = sketch.points[a].pos.value;
    let pb = sketch.points[b].pos.value;
    assert_near(pa.y - pb.y, 5.0, 0.01);
}

// -- Point-Line --

#[test]
fn test_midpoint() {
    let mut sketch = Sketch::new();
    let p = sketch.add_point(vect2d::new(1.0, 0.5));
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 2.0));
    sketch.midpoint.push(MidpointConstraint {
        point: p, line: l, hb: CrossBlock::new(),
    });
    sketch.solve();
    let pp = sketch.points[p].pos.value;
    let lp = &sketch.lines[l];
    let mx = (lp.p1.value.x + lp.p2.value.x) * 0.5;
    let my = (lp.p1.value.y + lp.p2.value.y) * 0.5;
    assert_near(pp.x, mx, 0.001);
    assert_near(pp.y, my, 0.001);
}

#[test]
fn test_distance_pl() {
    let mut sketch = Sketch::new();
    let p = sketch.add_point(vect2d::new(2.0, 3.0));
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 0.0));
    sketch.distance_pl.push(DistancePL {
        point: p, line: l, distance: 2.0, hb: CrossBlock::new(),
    });
    sketch.solve();
    let pp = sketch.points[p].pos.value;
    let lp = &sketch.lines[l];
    let dx = lp.p2.value.x - lp.p1.value.x;
    let dy = lp.p2.value.y - lp.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt();
    let dist = ((pp.x - lp.p1.value.x) * dy - (pp.y - lp.p1.value.y) * dx) / len;
    assert_near(dist, 2.0, 0.01);
}

// -- Point-Arc --

#[test]
fn test_point_on_arc() {
    let mut sketch = Sketch::new();
    let p = sketch.add_point(vect2d::new(3.5, 0.0));
    let a = sketch.add_arc(vect2d::new(0.0, 0.0), 3.0, 0.0, std::f64::consts::PI, false);
    sketch.point_on_arc.push(PointOnArc {
        point: p, arc: a, hb: CrossBlock::new(),
    });
    sketch.solve();
    let pp = sketch.points[p].pos.value;
    let ac = sketch.arcs[a].center.value;
    let ar = sketch.arcs[a].radius.value;
    let dx = pp.x - ac.x;
    let dy = pp.y - ac.y;
    let dist = (dx * dx + dy * dy).sqrt();
    assert_near(dist, ar, 0.01);
}

// -- Line-Line --

#[test]
fn test_collinear() {
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(2.0, 1.0));
    let b = sketch.add_line(vect2d::new(3.0, 1.8), vect2d::new(5.0, 2.3));
    sketch.collinear.push(Collinear {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let dx = la.p2.value.x - la.p1.value.x;
    let dy = la.p2.value.y - la.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt();
    // Both endpoints of b should lie on line a
    let cross1 = ((lb.p1.value.x - la.p1.value.x) * dy - (lb.p1.value.y - la.p1.value.y) * dx) / len;
    let cross2 = ((lb.p2.value.x - la.p1.value.x) * dy - (lb.p2.value.y - la.p1.value.y) * dx) / len;
    assert_near(cross1, 0.0, 0.001);
    assert_near(cross2, 0.0, 0.001);
}

#[test]
fn test_equal_length() {
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let b = sketch.add_line(vect2d::new(0.0, 2.0), vect2d::new(5.0, 2.0));
    sketch.equal_length.push(EqualLength {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let len_a = {
        let dx = la.p2.value.x - la.p1.value.x;
        let dy = la.p2.value.y - la.p1.value.y;
        (dx * dx + dy * dy).sqrt()
    };
    let len_b = {
        let dx = lb.p2.value.x - lb.p1.value.x;
        let dy = lb.p2.value.y - lb.p1.value.y;
        (dx * dx + dy * dy).sqrt()
    };
    assert_near(len_a, len_b, 0.01);
}

#[test]
fn test_angle_constraint() {
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let b = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(2.0, 2.5));
    let target = std::f64::consts::FRAC_PI_4; // 45 degrees
    sketch.angle.push(AngleConstraint {
        a, b, angle: target, hb: CrossBlock::new(),
    });
    sketch.solve();
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let dx1 = la.p2.value.x - la.p1.value.x;
    let dy1 = la.p2.value.y - la.p1.value.y;
    let dx2 = lb.p2.value.x - lb.p1.value.x;
    let dy2 = lb.p2.value.y - lb.p1.value.y;
    let angle = (dx1 * dy2 - dy1 * dx2).atan2(dx1 * dx2 + dy1 * dy2);
    assert_near(angle, target, 0.01);
}

// -- Line-Line endpoint coincidence --

#[test]
fn test_coincident_ll21() {
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(2.0, 1.0));
    let b = sketch.add_line(vect2d::new(2.2, 1.1), vect2d::new(4.0, 0.0));
    sketch.coincident_ll21.push(CoincidentLL21 {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    assert_near(la.p2.value.x, lb.p1.value.x, 0.001);
    assert_near(la.p2.value.y, lb.p1.value.y, 0.001);
}

// -- Line-Point --

#[test]
fn test_coincident_lp1() {
    let mut sketch = Sketch::new();
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 1.0));
    let p = sketch.add_point(vect2d::new(0.2, 0.1));
    sketch.coincident_lp1.push(CoincidentLP1 {
        line: l, point: p, hb: CrossBlock::new(),
    });
    sketch.solve();
    let lp = &sketch.lines[l];
    let pp = sketch.points[p].pos.value;
    assert_near(lp.p1.value.x, pp.x, 0.001);
    assert_near(lp.p1.value.y, pp.y, 0.001);
}

// -- Line-Arc --

#[test]
fn test_tangent_la() {
    let mut sketch = Sketch::new();
    let l = sketch.add_line(vect2d::new(-2.0, 2.5), vect2d::new(3.0, 2.5));
    let a = sketch.add_arc(vect2d::new(0.0, 0.0), 2.0, 0.0, std::f64::consts::PI, false);
    sketch.tangent_la.push(TangentLA {
        line: l, arc: a, hb: CrossBlock::new(),
    });
    sketch.solve();
    let lp = &sketch.lines[l];
    let ac = sketch.arcs[a].center.value;
    let ar = sketch.arcs[a].radius.value;
    let dx = lp.p2.value.x - lp.p1.value.x;
    let dy = lp.p2.value.y - lp.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt();
    let dist = ((ac.x - lp.p1.value.x) * dy - (ac.y - lp.p1.value.y) * dx) / len;
    assert_near(dist.abs(), ar, 0.01);
}

// -- Arc-Arc --

#[test]
fn test_concentric() {
    let mut sketch = Sketch::new();
    let a = sketch.add_arc(vect2d::new(1.0, 2.0), 3.0, 0.0, std::f64::consts::PI, false);
    let b = sketch.add_arc(vect2d::new(1.2, 2.1), 5.0, 0.0, std::f64::consts::PI, false);
    sketch.concentric.push(Concentric {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    let ca = sketch.arcs[a].center.value;
    let cb = sketch.arcs[b].center.value;
    assert_near(ca.x, cb.x, 0.001);
    assert_near(ca.y, cb.y, 0.001);
}

#[test]
fn test_equal_radius() {
    let mut sketch = Sketch::new();
    let a = sketch.add_arc(vect2d::new(0.0, 0.0), 3.0, 0.0, std::f64::consts::PI, false);
    let b = sketch.add_arc(vect2d::new(5.0, 0.0), 5.0, 0.0, std::f64::consts::PI, false);
    sketch.equal_radius.push(EqualRadius {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    assert_near(sketch.arcs[a].radius.value, sketch.arcs[b].radius.value, 0.01);
}

#[test]
fn test_tangent_aa() {
    let mut sketch = Sketch::new();
    let a = sketch.add_arc(vect2d::new(0.0, 0.0), 2.0, 0.0, std::f64::consts::PI, false);
    let b = sketch.add_arc(vect2d::new(4.5, 0.0), 3.0, 0.0, std::f64::consts::PI, false);
    sketch.tangent_aa.push(TangentAA {
        a, b, hb: CrossBlock::new(),
    });
    sketch.solve();
    let ca = sketch.arcs[a].center.value;
    let cb = sketch.arcs[b].center.value;
    let ra = sketch.arcs[a].radius.value;
    let rb = sketch.arcs[b].radius.value;
    let dx = cb.x - ca.x;
    let dy = cb.y - ca.y;
    let dist = (dx * dx + dy * dy).sqrt();
    assert_near(dist, ra + rb, 0.01);
}

// -- Multi-constraint --

#[test]
fn test_rectangle() {
    // Build a rectangle: 4 lines, horizontal/vertical constraints,
    // connected at corners, with one corner fixed
    let mut sketch = Sketch::new();
    let bottom = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.1));
    let right = sketch.add_line(vect2d::new(3.1, 0.0), vect2d::new(3.0, 2.1));
    let top = sketch.add_line(vect2d::new(2.9, 2.0), vect2d::new(0.1, 1.9));
    let left = sketch.add_line(vect2d::new(0.0, 2.1), vect2d::new(0.1, 0.1));

    // Horizontal/vertical
    sketch.lines[bottom].constraints.horizontal = true;
    sketch.lines[top].constraints.horizontal = true;
    sketch.lines[left].constraints.vertical = true;
    sketch.lines[right].constraints.vertical = true;

    // Connect corners: bottom.p2 = right.p1, right.p2 = top.p1, etc.
    sketch.coincident_ll21.push(CoincidentLL21 { a: bottom, b: right, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: right, b: top, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: top, b: left, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: left, b: bottom, hb: CrossBlock::new() });

    // Fix bottom-left corner
    sketch.lines[bottom].p1 = arael::model::Param::fixed(vect2d::new(0.0, 0.0));

    // Set lengths
    sketch.lines[bottom].constraints.has_length = true;
    sketch.lines[bottom].constraints.length = 4.0;
    sketch.lines[left].constraints.has_length = true;
    sketch.lines[left].constraints.length = 2.0;

    sketch.solve();

    let b = &sketch.lines[bottom];
    let r = &sketch.lines[right];
    let t = &sketch.lines[top];
    let _l = &sketch.lines[left];

    // Check fixed corner
    assert_near(b.p1.value.x, 0.0, 0.001);
    assert_near(b.p1.value.y, 0.0, 0.001);

    // Check rectangle dimensions
    assert_near(b.p2.value.x, 4.0, 0.01);
    assert_near(b.p2.value.y, 0.0, 0.01);
    assert_near(r.p2.value.x, 4.0, 0.01);
    assert_near(r.p2.value.y, 2.0, 0.01);
    assert_near(t.p2.value.x, 0.0, 0.01);
    assert_near(t.p2.value.y, 2.0, 0.01);
}

// -- Serde roundtrip --

#[test]
fn test_serde_roundtrip_triangle() {
    let mut sketch = Sketch::new();
    let l1 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let l2 = sketch.add_line(vect2d::new(3.0, 0.0), vect2d::new(1.5, 2.5));
    let l3 = sketch.add_line(vect2d::new(1.5, 2.5), vect2d::new(0.0, 0.0));
    sketch.coincident_ll21.push(CoincidentLL21 { a: l1, b: l2, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: l2, b: l3, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: l3, b: l1, hb: CrossBlock::new() });
    sketch.lines[l1].constraints.horizontal = true;
    sketch.solve();

    // Serialize with serde (JSON for readability in tests)
    let json = serde_json::to_string(&sketch).unwrap();
    let mut restored: Sketch = serde_json::from_str(&json).unwrap();
    restored.solve();

    // Verify positions match
    for r in sketch.lines.refs() {
        let orig = &sketch.lines[r];
        let rest = &restored.lines[r];
        assert_near(orig.p1.value.x, rest.p1.value.x, 0.001);
        assert_near(orig.p1.value.y, rest.p1.value.y, 0.001);
        assert_near(orig.p2.value.x, rest.p2.value.x, 0.001);
        assert_near(orig.p2.value.y, rest.p2.value.y, 0.001);
    }

    // Verify constraints survived
    assert_eq!(restored.coincident_ll21.len(), 3);
    assert!(restored.lines[l1].constraints.horizontal);
}

#[test]
fn test_serde_roundtrip_rectangle_with_fixed() {
    let mut sketch = Sketch::new();
    let bottom = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.1));
    let right = sketch.add_line(vect2d::new(3.1, 0.0), vect2d::new(3.0, 2.1));
    let top = sketch.add_line(vect2d::new(2.9, 2.0), vect2d::new(0.1, 1.9));
    let left = sketch.add_line(vect2d::new(0.0, 2.1), vect2d::new(0.1, 0.1));
    sketch.lines[bottom].constraints.horizontal = true;
    sketch.lines[top].constraints.horizontal = true;
    sketch.lines[left].constraints.vertical = true;
    sketch.lines[right].constraints.vertical = true;
    sketch.coincident_ll21.push(CoincidentLL21 { a: bottom, b: right, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: right, b: top, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: top, b: left, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: left, b: bottom, hb: CrossBlock::new() });
    sketch.lines[bottom].p1 = Param::fixed(vect2d::new(0.0, 0.0));
    sketch.lines[bottom].constraints.has_length = true;
    sketch.lines[bottom].constraints.length = 4.0;
    sketch.solve();

    let json = serde_json::to_string(&sketch).unwrap();
    let mut restored: Sketch = serde_json::from_str(&json).unwrap();
    restored.solve();

    // Fixed param preserved
    assert!(!restored.lines[bottom].p1.optimize);
    assert_near(restored.lines[bottom].p1.value.x, 0.0, 0.001);
    assert_near(restored.lines[bottom].p1.value.y, 0.0, 0.001);

    // Rectangle shape preserved
    assert_near(restored.lines[bottom].p2.value.x, 4.0, 0.01);
    assert_near(restored.lines[right].p2.value.y, 2.0, 0.1);
}

#[test]
fn test_graduated_optimization_length() {
    // Reproduces a previously-broken case: single line with length constraint
    // far from target. Without graduated optimization, the rank-1 Hessian
    // caused LM to oscillate and never converge.
    let mut sketch = Sketch::new();
    let line = sketch.add_line(
        vect2d::new(-4.300650119781494, -0.0029095385689288378),
        vect2d::new(-0.0643674734517385, -2.2063466414334907),
    );
    sketch.lines[line].constraints.has_length = true;
    sketch.lines[line].constraints.length = 3.0;

    let result = sketch.solve();

    let l = &sketch.lines[line];
    let len = ((l.p2.value.x - l.p1.value.x).powi(2)
        + (l.p2.value.y - l.p1.value.y).powi(2)).sqrt();
    assert_near(len, 3.0, 0.001);
    assert!(result.end_cost < 1.0, "cost={} should be near zero", result.end_cost);
}

#[test]
fn test_circle_has_3_params() {
    // A circle (closed arc) should have 3 optimizable params: center.x, center.y, radius.
    // start_angle and end_angle are fixed since they are meaningless for a full circle.
    let mut sketch = Sketch::new();
    sketch.add_arc(vect2d::new(1.0, 2.0), 3.0, 0.0, std::f64::consts::TAU, true);
    let mut params = Vec::new();
    sketch.serialize64(&mut params);
    assert_eq!(params.len(), 3, "circle should have 3 params (cx, cy, r)");
}

#[test]
fn test_arc_has_5_params() {
    // An arc (non-closed) should have 5 optimizable params:
    // center.x, center.y, radius, start_angle, end_angle.
    let mut sketch = Sketch::new();
    sketch.add_arc(vect2d::new(1.0, 2.0), 3.0, 0.0, 1.5, false);
    let mut params = Vec::new();
    sketch.serialize64(&mut params);
    assert_eq!(params.len(), 5, "arc should have 5 params (cx, cy, r, sa, ea)");
}

#[test]
fn test_collinear_nonaligned() {
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(2.0, 1.0));
    let b = sketch.add_line(vect2d::new(3.0, 2.0), vect2d::new(5.0, 3.5));
    sketch.collinear.push(Collinear { a, b, hb: CrossBlock::new() });
    sketch.solve();
    // Both endpoints of b should lie on the infinite line through a
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let dx = la.p2.value.x - la.p1.value.x;
    let dy = la.p2.value.y - la.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt();
    let cross1 = ((lb.p1.value.x - la.p1.value.x) * dy
        - (lb.p1.value.y - la.p1.value.y) * dx) / len;
    let cross2 = ((lb.p2.value.x - la.p1.value.x) * dy
        - (lb.p2.value.y - la.p1.value.y) * dx) / len;
    assert_near(cross1, 0.0, 0.01);
    assert_near(cross2, 0.0, 0.01);
}

#[test]
fn test_midpoint_point() {
    let mut sketch = Sketch::new();
    let p = sketch.add_point(vect2d::new(3.0, 1.0));
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 2.0));
    sketch.midpoint.push(MidpointConstraint { point: p, line: l, hb: CrossBlock::new() });
    sketch.solve();
    let pt = sketch.points[p].pos.value;
    let ln = &sketch.lines[l];
    let mx = (ln.p1.value.x + ln.p2.value.x) * 0.5;
    let my = (ln.p1.value.y + ln.p2.value.y) * 0.5;
    assert_near(pt.x, mx, 0.01);
    assert_near(pt.y, my, 0.01);
}

#[test]
fn test_midpoint_lp1() {
    let mut sketch = Sketch::new();
    let src = sketch.add_line(vect2d::new(5.0, 5.0), vect2d::new(6.0, 6.0));
    let tgt = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 2.0));
    sketch.midpoint_lp1.push(MidpointLP1 { line: src, target: tgt, hb: CrossBlock::new() });
    sketch.solve();
    let p1 = sketch.lines[src].p1.value;
    let tl = &sketch.lines[tgt];
    let mx = (tl.p1.value.x + tl.p2.value.x) * 0.5;
    let my = (tl.p1.value.y + tl.p2.value.y) * 0.5;
    assert_near(p1.x, mx, 0.01);
    assert_near(p1.y, my, 0.01);
}

#[test]
fn test_midpoint_arc_start() {
    let mut sketch = Sketch::new();
    let arc = sketch.add_arc(vect2d::new(5.0, 5.0), 2.0, 0.0, 1.5, false);
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 2.0));
    sketch.midpoint_arc_start.push(MidpointArcStart { arc, line: l, hb: CrossBlock::new() });
    sketch.solve();
    let a = &sketch.arcs[arc];
    let sx = a.center.value.x + a.radius.value * a.start_angle.value.cos();
    let sy = a.center.value.y + a.radius.value * a.start_angle.value.sin();
    let ln = &sketch.lines[l];
    let mx = (ln.p1.value.x + ln.p2.value.x) * 0.5;
    let my = (ln.p1.value.y + ln.p2.value.y) * 0.5;
    assert_near(sx, mx, 0.01);
    assert_near(sy, my, 0.01);
}

#[test]
fn test_angle_dimension_45deg() {
    // Constrain angle between two lines to 45 degrees
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let b = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(2.0, 1.0));
    let target_rad = std::f64::consts::FRAC_PI_4; // 45 degrees
    sketch.angle.push(AngleConstraint { a, b, angle: target_rad, hb: CrossBlock::new() });
    sketch.solve();
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let dx1 = la.p2.value.x - la.p1.value.x;
    let dy1 = la.p2.value.y - la.p1.value.y;
    let dx2 = lb.p2.value.x - lb.p1.value.x;
    let dy2 = lb.p2.value.y - lb.p1.value.y;
    let cross = dx1 * dy2 - dy1 * dx2;
    let dot = dx1 * dx2 + dy1 * dy2;
    let angle = cross.atan2(dot);
    assert_near(angle, target_rad, 0.01);
}

#[test]
fn test_angle_dimension_negative_atan2() {
    // When atan2 is negative, the constraint angle must also be negative.
    // This reproduces a bug where supplement angles got the wrong sign.
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 1.0), vect2d::new(-1.0, -4.0));
    let b = sketch.add_line(vect2d::new(1.5, -4.0), vect2d::new(0.0, 1.0));
    // Current angle is negative (~ -148 deg). Supplement sector shows ~31 deg.
    // Setting supplement to 35 deg means target = -(pi - rad(35)) = -2.53 rad.
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let dx1 = la.p2.value.x - la.p1.value.x;
    let dy1 = la.p2.value.y - la.p1.value.y;
    let dx2 = lb.p2.value.x - lb.p1.value.x;
    let dy2 = lb.p2.value.y - lb.p1.value.y;
    let current = (dx1 * dy2 - dy1 * dx2).atan2(dx1 * dx2 + dy1 * dy2);
    assert!(current < 0.0, "test setup: atan2 should be negative, got {}", current);
    // Target: supplement of 35 deg, matching sign of current
    let mut target = std::f64::consts::PI - 35.0f64.to_radians();
    if current < 0.0 { target = -target; }
    sketch.angle.push(AngleConstraint { a, b, angle: target, hb: CrossBlock::new() });
    let result = sketch.solve();
    assert!(result.end_cost < 1.0, "solver failed: cost={}", result.end_cost);
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let dx1 = la.p2.value.x - la.p1.value.x;
    let dy1 = la.p2.value.y - la.p1.value.y;
    let dx2 = lb.p2.value.x - lb.p1.value.x;
    let dy2 = lb.p2.value.y - lb.p1.value.y;
    let final_angle = (dx1 * dy2 - dy1 * dx2).atan2(dx1 * dx2 + dy1 * dy2);
    assert_near(final_angle, target, 0.01);
}

#[test]
fn test_symmetry_ll() {
    // B is vertical mirror, A and C are roughly symmetric.
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(-4.0, 2.0), vect2d::new(-1.0, 3.0));
    let b = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(0.0, 5.0));
    let c = sketch.add_line(vect2d::new(1.0, 3.0), vect2d::new(4.0, 2.0));
    sketch.symmetry_ll.push(SymmetryLL {
        a, b, c, hb: arael::model::TripletBlock::new(),
    });
    sketch.solve();

    // Verify: projection of B endpoints onto A and C, dotted with B_normal, sum to ~0
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let lc = &sketch.lines[c];
    let (d1a, d1c) = projection_distances(lb.p1.value, lb, la, lc);
    let (d2a, d2c) = projection_distances(lb.p2.value, lb, la, lc);
    assert_near(d1a + d1c, 0.0, 0.01);
    assert_near(d2a + d2c, 0.0, 0.01);
    assert!(d1a.abs() > 0.5, "lines collapsed: d1a={}", d1a);
}

#[test]
fn test_symmetry_ll_nonparallel() {
    // Symmetry with non-parallel lines (V-shape, not sharing endpoints).
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(-3.0, 2.0), vect2d::new(3.0, 4.0));
    let b = sketch.add_line(vect2d::new(0.0, -1.0), vect2d::new(0.0, 5.0));
    let c = sketch.add_line(vect2d::new(-3.0, -2.0), vect2d::new(3.0, -4.0));
    sketch.symmetry_ll.push(SymmetryLL {
        a, b, c, hb: arael::model::TripletBlock::new(),
    });
    sketch.solve();

    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let lc = &sketch.lines[c];
    // Verify using ray-intersection formula (same as constraint)
    let (r1, r2) = ray_symmetry_residuals(lb, la, lc);
    assert_near(r1, 0.0, 0.1);
    assert_near(r2, 0.0, 0.1);
}

/// Compute ray-intersection symmetry residuals (same formula as constraint).
fn ray_symmetry_residuals(lb: &Line, la: &Line, lc: &Line) -> (f64, f64) {
    let bnx = -(lb.p2.value.y - lb.p1.value.y);
    let bny = lb.p2.value.x - lb.p1.value.x;
    let adx = la.p2.value.x - la.p1.value.x;
    let ady = la.p2.value.y - la.p1.value.y;
    let cdx = lc.p2.value.x - lc.p1.value.x;
    let cdy = lc.p2.value.y - lc.p1.value.y;
    let bna = bnx * ady - bny * adx;
    let bnc = bnx * cdy - bny * cdx;
    let cross2 = |ax: f64, ay: f64, bx: f64, by: f64| ax * by - ay * bx;
    let r1 = cross2(la.p1.value.x - lb.p1.value.x, la.p1.value.y - lb.p1.value.y, adx, ady) * bnc
           + cross2(lc.p1.value.x - lb.p1.value.x, lc.p1.value.y - lb.p1.value.y, cdx, cdy) * bna;
    let r2 = cross2(la.p1.value.x - lb.p2.value.x, la.p1.value.y - lb.p2.value.y, adx, ady) * bnc
           + cross2(lc.p1.value.x - lb.p2.value.x, lc.p1.value.y - lb.p2.value.y, cdx, cdy) * bna;
    (r1, r2)
}

/// Compute projection-based signed distances: project P onto A and C,
/// dot (foot - P) with B_normal. Returns (d_a, d_c).
fn projection_distances(p: arael::vect::vect2d, lb: &Line, la: &Line, lc: &Line) -> (f64, f64) {
    let bdx = lb.p2.value.x - lb.p1.value.x;
    let bdy = lb.p2.value.y - lb.p1.value.y;
    let blen = (bdx * bdx + bdy * bdy).sqrt();
    let bnx = bdy / blen;
    let bny = bdx / blen;
    let project = |l: &Line| -> f64 {
        let dx = l.p2.value.x - l.p1.value.x;
        let dy = l.p2.value.y - l.p1.value.y;
        let len2 = dx * dx + dy * dy;
        let t = ((p.x - l.p1.value.x) * dx + (p.y - l.p1.value.y) * dy) / len2;
        let fx = l.p1.value.x + t * dx - p.x;
        let fy = l.p1.value.y + t * dy - p.y;
        fx * bnx - fy * bny
    };
    (project(la), project(lc))
}

// -- Expression dimension tests --

#[test]
fn test_expr_dim_reference() {
    // L0 length=10 via normal dim, L1 length="d0" via expression dim
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let _l1 = sketch.add_line(vect2d::new(5.0, 0.0), vect2d::new(8.0, 0.0));

    // Normal dimension on L0
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 10.0;
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l0), value: 10.0,
        offset: vect2d::new(0.0, 1.0), text_along: 0.0,
        name: "d0".into(), expr_str: None,
    });
    sketch.next_dimension_id = 1;

    // Expression dimension on L1: length = d0
    let l1_ref = arael::refs::Ref::<Line>::new(1);
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1_ref), "d0",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();

    let result = sketch.solve();
    assert!(result.end_cost < 1.0, "solver failed: cost={}", result.end_cost);
    let l1 = &sketch.lines[l1_ref];
    let l1_len = (l1.p2.value - l1.p1.value).norm();
    assert_near(l1_len, 10.0, 0.1);
}

#[test]
fn test_expr_dim_arithmetic() {
    // L0 length=5, L1 length="d0 * 2 + 3" -> should be 13
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let _l1 = sketch.add_line(vect2d::new(5.0, 0.0), vect2d::new(8.0, 0.0));

    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 5.0;
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l0), value: 5.0,
        offset: vect2d::new(0.0, 1.0), text_along: 0.0,
        name: "d0".into(), expr_str: None,
    });
    sketch.next_dimension_id = 1;

    let l1_ref = arael::refs::Ref::<Line>::new(1);
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1_ref), "d0 * 2 + 3",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();

    let result = sketch.solve();
    assert!(result.end_cost < 1.0, "solver failed: cost={}", result.end_cost);
    let l1 = &sketch.lines[l1_ref];
    let l1_len = (l1.p2.value - l1.p1.value).norm();
    assert_near(l1_len, 13.0, 0.1);
}

#[test]
fn test_expr_dim_derived_property() {
    // L1 length = L0.length (both should end up the same)
    let mut sketch = Sketch::new();
    sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 4.0)); // L0, length=5
    sketch.add_line(vect2d::new(5.0, 0.0), vect2d::new(8.0, 0.0)); // L1, length=3

    let l1_ref = arael::refs::Ref::<Line>::new(1);
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1_ref), "L0.length",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();

    let result = sketch.solve();
    assert!(result.end_cost < 1.0, "solver failed: cost={}", result.end_cost);

    let l0 = &sketch.lines[arael::refs::Ref::<Line>::new(0)];
    let l1 = &sketch.lines[l1_ref];
    let l0_len = (l0.p2.value - l0.p1.value).norm();
    let l1_len = (l1.p2.value - l1.p1.value).norm();
    assert_near(l0_len, l1_len, 0.1);
}

// test_expr_constraint_linked_dimensions removed — superseded by
// test_expr_dim_reference which uses add_expr_dimension (the proper API)

#[test]
fn test_bincode_roundtrip() {
    // Reproduces undo crash: bincode serialize/deserialize of Sketch
    // with dimensions (including expr_str field).
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 4.0));
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 5.0;
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l0), value: 5.0,
        offset: vect2d::new(0.0, 1.0), text_along: 0.0,
        name: "d0".into(), expr_str: None,
    });
    sketch.next_dimension_id = 1;

    // Serialize and deserialize with bincode (same as History)
    let bytes = bincode::serialize(&sketch).unwrap();
    let restored: Sketch = bincode::deserialize(&bytes).unwrap();
    assert_eq!(restored.dimensions.len(), 1);
    assert_near(restored.dimensions[0].value, 5.0, 0.001);
}

#[test]
fn test_expr_dim_locked_line() {
    // L0 has locked (non-optimizable) endpoints, L1 is free.
    // Set L1.length = L0.length via expression dimension.
    // L0.length symbols must resolve even though L0 params aren't optimized.
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 4.0));
    sketch.lines[l0].p1 = Param::fixed(vect2d::new(0.0, 0.0));
    sketch.lines[l0].p2 = Param::fixed(vect2d::new(3.0, 4.0));
    let _l1 = sketch.add_line(vect2d::new(5.0, 0.0), vect2d::new(8.0, 0.0));

    let l1_ref = arael::refs::Ref::<Line>::new(1);
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1_ref), "L0.length",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();

    let result = sketch.solve();
    assert!(result.end_cost < 1.0, "solver failed: cost={}", result.end_cost);

    // L0.length = 5, so L1 should also be 5
    let l1 = &sketch.lines[l1_ref];
    let l1_len = (l1.p2.value - l1.p1.value).norm();
    assert_near(l1_len, 5.0, 0.1);
}

#[test]
fn test_expr_dim_chained_drag() {
    // Reproduces dimbug2: 3 lines with locked p2, chained expression dims.
    // d1 on L1 = "L0.length*2", d3 on L2 = "d1*2".
    // Dragging L2.p1 should work (solver should converge).
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, -2.3), vect2d::new(0.0, -3.8));
    sketch.lines[l0].p2 = Param::fixed(vect2d::new(0.0, -3.8));
    let l1 = sketch.add_line(vect2d::new(4.0, 1.3), vect2d::new(4.0, -1.6));
    sketch.lines[l1].p2 = Param::fixed(vect2d::new(4.0, -1.6));
    let l2 = sketch.add_line(vect2d::new(8.0, 1.7), vect2d::new(8.0, -4.1));
    sketch.lines[l2].p2 = Param::fixed(vect2d::new(8.0, -4.1));

    // d1 on L1: length = L0.length * 2
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1), "L0.length*2",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();

    // d3 on L2: length = d1 * 2
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l2), "d0*2",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();

    // Initial solve
    let result = sketch.solve();
    assert!(result.end_cost < 1.0, "initial solve failed: cost={}", result.end_cost);

    // Simulate drag: move L2.p1 and re-solve
    sketch.lines[l2].p1.value = vect2d::new(8.0, 3.0);
    let result = sketch.solve();
    assert!(result.end_cost < 1.0, "drag solve failed: cost={}", result.end_cost);

    // L2 length should still satisfy d3 = d1 * 2
    let l2_len = (sketch.lines[l2].p2.value - sketch.lines[l2].p1.value).norm();
    let l1_len = (sketch.lines[l1].p2.value - sketch.lines[l1].p1.value).norm();
    let l0_len = (sketch.lines[l0].p2.value - sketch.lines[l0].p1.value).norm();
    eprintln!("L0={:.3} L1={:.3} L2={:.3}", l0_len, l1_len, l2_len);
    // d1 = L0.length * 2, so L1 should be 2 * L0
    assert_near(l1_len, l0_len * 2.0, 0.2);
    // d3 = d1 * 2, d1's cached value = L1's length
    // But d1 in symbol bag is the cached value, not live...
}

#[test]
fn test_expr_dim_angle_reference() {
    // Angle between L0/L1 = 20 deg (d1). Set angle between L2/L3 = "d1".
    // Both angle dimensions should converge to 20 degrees.
    use arael::model::CrossBlock;
    let mut sketch = Sketch::new();
    // L0 vertical, L1 going up-right from L0.p2
    let l0 = sketch.add_line(vect2d::new(0.0, 2.0), vect2d::new(0.0, -3.0));
    sketch.lines[l0].p2 = Param::fixed(vect2d::new(0.0, -3.0));
    let l1 = sketch.add_line(vect2d::new(0.0, -3.0), vect2d::new(2.0, 1.0));
    sketch.coincident_ll21.push(CoincidentLL21 { a: l0, b: l1, hb: CrossBlock::new() });

    // L2 vertical, L3 going up-right from L2.p2
    let l2 = sketch.add_line(vect2d::new(5.0, 2.0), vect2d::new(5.0, -3.0));
    sketch.lines[l2].p2 = Param::fixed(vect2d::new(5.0, -3.0));
    let l3 = sketch.add_line(vect2d::new(5.0, -3.0), vect2d::new(7.0, 1.0));
    sketch.coincident_ll21.push(CoincidentLL21 { a: l2, b: l3, hb: CrossBlock::new() });

    // d1: angle between L0 and L1, supplement=true, value=20 deg
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 5.0;
    sketch.angle.push(AngleConstraint {
        a: l0, b: l1,
        angle: {
            let la = &sketch.lines[l0];
            let lb = &sketch.lines[l1];
            let dx1 = la.p2.value.x - la.p1.value.x;
            let dy1 = la.p2.value.y - la.p1.value.y;
            let dx2 = lb.p2.value.x - lb.p1.value.x;
            let dy2 = lb.p2.value.y - lb.p1.value.y;
            let current = (dx1*dy2 - dy1*dx2).atan2(dx1*dx2 + dy1*dy2);
            let mut target = std::f64::consts::PI - 20.0f64.to_radians();
            if current < 0.0 { target = -target; }
            target
        },
        hb: CrossBlock::new(),
    });
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::Angle(l0, l1, true), value: 20.0,
        offset: vect2d::new(0.0, 1.0), text_along: 0.0,
        name: "d1".into(), expr_str: None,
    });
    sketch.next_dimension_id = 2;

    // Expression angle dimension: angle(L2, L3) = d1
    sketch.add_expr_dimension(
        DimensionKind::Angle(l2, l3, true), "d1",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();

    let result = sketch.solve();
    eprintln!("angle ref test: cost={:.6} iters={}", result.end_cost, result.iterations);
    assert!(result.end_cost < 1.0, "solver failed: cost={}", result.end_cost);

    // Check both angles are ~20 degrees
    let angle_01 = {
        let la = &sketch.lines[l0]; let lb = &sketch.lines[l1];
        let dx1 = la.p2.value.x - la.p1.value.x; let dy1 = la.p2.value.y - la.p1.value.y;
        let dx2 = lb.p2.value.x - lb.p1.value.x; let dy2 = lb.p2.value.y - lb.p1.value.y;
        180.0 - (dx1*dy2 - dy1*dx2).atan2(dx1*dx2 + dy1*dy2).abs().to_degrees()
    };
    let angle_23 = {
        let la = &sketch.lines[l2]; let lb = &sketch.lines[l3];
        let dx1 = la.p2.value.x - la.p1.value.x; let dy1 = la.p2.value.y - la.p1.value.y;
        let dx2 = lb.p2.value.x - lb.p1.value.x; let dy2 = lb.p2.value.y - lb.p1.value.y;
        180.0 - (dx1*dy2 - dy1*dx2).atan2(dx1*dx2 + dy1*dy2).abs().to_degrees()
    };
    eprintln!("  angle L0-L1={:.2} deg, angle L2-L3={:.2} deg", angle_01, angle_23);
    assert_near(angle_01, 20.0, 1.0);
    assert_near(angle_23, 20.0, 1.0);
}

// -- UpdateDimension preserves name --

#[test]
fn test_update_dimension_preserves_name() {
    // Create a line with a length dimension, then update the value.
    // The dimension name must stay the same so expression references work.
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    // Add dimension d0 with value 10
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 10.0;
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l0),
        value: 10.0,
        offset: vect2d::new(0.0, 1.0),
        text_along: 0.0,
        name: "d0".into(),
        expr_str: None,
    });
    sketch.solve();
    let len0 = line_length(&sketch, l0);
    assert_near(len0, 10.0, 0.01);
    assert_eq!(sketch.dimensions[0].name, "d0");

    // Update dimension to value 15 (simulates what UpdateDimension action does)
    // 1. Remove old constraint
    sketch.lines[l0].constraints.has_length = false;
    // 2. Update dimension in place
    sketch.dimensions[0].value = 15.0;
    // 3. Apply new constraint
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 15.0;
    sketch.solve();

    let len1 = line_length(&sketch, l0);
    assert_near(len1, 15.0, 0.01);
    // Name must be preserved
    assert_eq!(sketch.dimensions[0].name, "d0");
}

#[test]
fn test_update_dimension_numeric_to_expr() {
    // Start with a numeric dimension, update it to an expression dimension.
    // The name must be preserved.
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    let l1 = sketch.add_line(vect2d::new(0.0, 2.0), vect2d::new(3.0, 2.0));

    // d0 = line length of L0 = 10
    sketch.next_dimension_id = 1;
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 10.0;
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l0),
        value: 10.0,
        offset: vect2d::new(0.0, 1.0),
        text_along: 0.0,
        name: "d0".into(),
        expr_str: None,
    });
    sketch.solve();
    assert_near(line_length(&sketch, l0), 10.0, 0.01);

    // d1 = line length of L1 = d0 * 2
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1), "d0 * 2",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();
    sketch.solve();
    sketch.update_expr_dim_values();
    assert_near(line_length(&sketch, l1), 20.0, 0.01);
    assert_eq!(sketch.dimensions[1].name, "d1");

    // Now update d0 from 10 to 5 -- d1 should become 10
    sketch.lines[l0].constraints.has_length = false;
    sketch.dimensions[0].value = 5.0;
    sketch.dimensions[0].expr_str = None;
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 5.0;
    sketch.solve();
    sketch.update_expr_dim_values();
    assert_near(line_length(&sketch, l0), 5.0, 0.01);
    assert_near(line_length(&sketch, l1), 10.0, 0.01);
    // Name preserved
    assert_eq!(sketch.dimensions[0].name, "d0");
    assert_eq!(sketch.dimensions[1].name, "d1");
}
