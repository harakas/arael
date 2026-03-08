use arael::model::{CrossBlock, Param};
use arael::vect::vect2d;
use arael_sketch::*;

fn assert_near(a: f64, b: f64, tol: f64) {
    assert!((a - b).abs() < tol, "expected {a} ~= {b} (diff={})", (a - b).abs());
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
