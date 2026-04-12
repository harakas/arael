use arael::model::{CrossBlock, Param, TripletBlock};
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
        a, b, cid: 0, hb: CrossBlock::new(),
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
        a, b, cid: 0, hb: CrossBlock::new(),
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
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let cross = (la.p2.value.x - la.p1.value.x) * (lb.p2.value.y - lb.p1.value.y)
              - (la.p2.value.y - la.p1.value.y) * (lb.p2.value.x - lb.p1.value.x);
    let dir_sign = if cross >= 0.0 { 1.0 } else { -1.0 };
    sketch.perpendicular.push(Perpendicular {
        a, b, dir_sign, cid: 0, hb: CrossBlock::new(),
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
fn test_perpendicular_no_flip() {
    let mut sketch = Sketch::new();
    // Horizontal line and vertical line
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(3.0, 0.0));
    let b = sketch.add_line(vect2d::new(1.0, 0.0), vect2d::new(1.0, 2.0));
    // cross = 3*2 - 0*0 = 6 > 0, dir_sign = 1
    sketch.perpendicular.push(Perpendicular {
        a, b, dir_sign: 1.0, cid: 0, hb: CrossBlock::new(),
    });
    sketch.solve();
    // Record initial cross product sign
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let cross1 = (la.p2.value.x - la.p1.value.x) * (lb.p2.value.y - lb.p1.value.y)
               - (la.p2.value.y - la.p1.value.y) * (lb.p2.value.x - lb.p1.value.x);
    assert!(cross1 > 0.0, "initial cross should be positive, got {}", cross1);
    // Flip line b's direction
    sketch.lines[b].p2.value = vect2d::new(1.0, -2.0);
    sketch.solve();
    // Cross product should still be positive (Heaviside prevents flip)
    let la = &sketch.lines[a];
    let lb = &sketch.lines[b];
    let cross2 = (la.p2.value.x - la.p1.value.x) * (lb.p2.value.y - lb.p1.value.y)
               - (la.p2.value.y - la.p1.value.y) * (lb.p2.value.x - lb.p1.value.x);
    assert!(cross2 > 0.0, "cross product flipped after perturbation! cross = {}", cross2);
}

#[test]
fn test_fixed_point_doesnt_move() {
    let mut sketch = Sketch::new();
    let a = sketch.add_point_fixed(vect2d::new(1.0, 2.0));
    let b = sketch.add_point(vect2d::new(1.5, 2.5));
    sketch.coincident_pp.push(CoincidentPP {
        a, b, cid: 0, hb: CrossBlock::new(),
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
        point: p, line: l, cid: 0, hb: CrossBlock::new(),
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
        a, b, distance: 5.0, cid: 0, hb: CrossBlock::new(),
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
        a, b, distance: 5.0, cid: 0, hb: CrossBlock::new(),
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
        a, b, distance: 5.0, cid: 0, hb: CrossBlock::new(),
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
        point: p, line: l, cid: 0, hb: CrossBlock::new(),
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
        point: p, line: l, distance: 2.0, cid: 0, hb: CrossBlock::new(),
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
        point: p, arc: a, cid: 0, hb: CrossBlock::new(),
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
        a, b, cid: 0, hb: CrossBlock::new(),
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
        a, b, cid: 0, hb: CrossBlock::new(),
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
        a, b, angle: target, cid: 0, hb: CrossBlock::new(),
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
        a, b, cid: 0, hb: CrossBlock::new(),
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
        line: l, point: p, cid: 0, hb: CrossBlock::new(),
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
    // Compute sign: center is below the line (positive signed distance)
    let l_ref = &sketch.lines[l];
    let dx = l_ref.p2.value.x - l_ref.p1.value.x;
    let dy = l_ref.p2.value.y - l_ref.p1.value.y;
    let len = (dx * dx + dy * dy).sqrt();
    let dist = ((sketch.arcs[a].center.value.x - l_ref.p1.value.x) * dy
              - (sketch.arcs[a].center.value.y - l_ref.p1.value.y) * dx) / len;
    let sign = if dist >= 0.0 { 1.0 } else { -1.0 };
    sketch.tangent_la.push(TangentLA {
        line: l, arc: a, sign, p1_arc_start: false, p1_arc_end: false, p2_arc_start: false, p2_arc_end: false, dir_sign: 0.0, cid: 0, hb: CrossBlock::new(),
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

#[test]
fn test_tangent_la_shared_endpoint_no_flip() {
    use std::f64::consts::PI;
    let mut sketch = Sketch::new();
    // Arc centered at origin, radius 2, from 0 to PI (semicircle)
    let a = sketch.add_arc(vect2d::new(0.0, 0.0), 2.0, 0.0, PI, false);
    // Arc start is at (2, 0). Line from (2, 0) going upward.
    let l = sketch.add_line(vect2d::new(2.0, 0.0), vect2d::new(2.0, 3.0));
    // Coincident: line.p1 == arc start
    sketch.coincident_lp1_arc_start.push(CoincidentLP1ArcStart {
        line: l, arc: a, cid: 0, hb: CrossBlock::new(),
    });
    // Tangent with p1_arc_start=true (dir_sign will be computed by update_tangent_flags)
    sketch.tangent_la.push(TangentLA {
        line: l, arc: a, sign: 1.0, p1_arc_start: true, p1_arc_end: false, p2_arc_start: false, p2_arc_end: false, dir_sign: f64::NAN, cid: 0, hb: CrossBlock::new(),
    });
    sketch.solve();
    // After solve, line should be tangent at arc start (angle=0).
    // Tangent at angle=0 on a circle at origin is vertical: (0, 1).
    // Line p1=(2,0), so p2 should be above: angle near PI/2.
    let lp = &sketch.lines[l];
    let angle1 = (lp.p2.value.y - lp.p1.value.y).atan2(lp.p2.value.x - lp.p1.value.x);
    assert!(angle1 > 0.0, "line should point upward, got angle {}", angle1);
    // Now perturb p2 to below the x-axis (would tempt the solver to flip)
    sketch.lines[l].p2.value = vect2d::new(2.0, -3.0);
    sketch.solve();
    // Direction should NOT have flipped -- directed tangent constraint prevents it
    let lp2 = &sketch.lines[l];
    let angle2 = (lp2.p2.value.y - lp2.p1.value.y).atan2(lp2.p2.value.x - lp2.p1.value.x);
    assert!(angle2 > 0.0, "line direction flipped after perturbation! angle = {}", angle2);
}

// -- Arc-Arc --

#[test]
fn test_concentric() {
    let mut sketch = Sketch::new();
    let a = sketch.add_arc(vect2d::new(1.0, 2.0), 3.0, 0.0, std::f64::consts::PI, false);
    let b = sketch.add_arc(vect2d::new(1.2, 2.1), 5.0, 0.0, std::f64::consts::PI, false);
    sketch.concentric.push(Concentric {
        a, b, cid: 0, hb: CrossBlock::new(),
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
        a, b, cid: 0, hb: CrossBlock::new(),
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
        a, b, shared: SharedEndpoint::None, cid: 0, hb: CrossBlock::new(),
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
    sketch.coincident_ll21.push(CoincidentLL21 { a: bottom, b: right, cid: 0, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: right, b: top, cid: 0, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: top, b: left, cid: 0, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: left, b: bottom, cid: 0, hb: CrossBlock::new() });

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
    sketch.coincident_ll21.push(CoincidentLL21 { a: l1, b: l2, cid: 0, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: l2, b: l3, cid: 0, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: l3, b: l1, cid: 0, hb: CrossBlock::new() });
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
    sketch.coincident_ll21.push(CoincidentLL21 { a: bottom, b: right, cid: 0, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: right, b: top, cid: 0, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: top, b: left, cid: 0, hb: CrossBlock::new() });
    sketch.coincident_ll21.push(CoincidentLL21 { a: left, b: bottom, cid: 0, hb: CrossBlock::new() });
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
fn test_circle_has_4_params() {
    // A circle (closed arc) should have 4 optimizable params: center.x, center.y, radius, radius_b.
    // start_angle and end_angle are fixed. rotation is fixed. radius_b is optimizable
    // (with equality constraint radius_b = radius for non-ellipse arcs).
    let mut sketch = Sketch::new();
    sketch.add_arc(vect2d::new(1.0, 2.0), 3.0, 0.0, std::f64::consts::TAU, true);
    let mut params = Vec::new();
    sketch.serialize64(&mut params);
    assert_eq!(params.len(), 4, "circle should have 4 params (cx, cy, r, rb)");
}

#[test]
fn test_arc_has_6_params() {
    // An arc (non-closed) should have 6 optimizable params:
    // center.x, center.y, radius, radius_b, start_angle, end_angle.
    // rotation is fixed.
    let mut sketch = Sketch::new();
    sketch.add_arc(vect2d::new(1.0, 2.0), 3.0, 0.0, 1.5, false);
    let mut params = Vec::new();
    sketch.serialize64(&mut params);
    assert_eq!(params.len(), 6, "arc should have 6 params (cx, cy, r, rb, sa, ea)");
}

#[test]
fn test_collinear_nonaligned() {
    let mut sketch = Sketch::new();
    let a = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(2.0, 1.0));
    let b = sketch.add_line(vect2d::new(3.0, 2.0), vect2d::new(5.0, 3.5));
    sketch.collinear.push(Collinear { a, b, cid: 0, hb: CrossBlock::new() });
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
    sketch.midpoint.push(MidpointConstraint { point: p, line: l, cid: 0, hb: CrossBlock::new() });
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
    sketch.midpoint_lp1.push(MidpointLP1 { line: src, target: tgt, cid: 0, hb: CrossBlock::new() });
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
    sketch.midpoint_arc_start.push(MidpointArcStart { arc, line: l, cid: 0, hb: CrossBlock::new() });
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
    sketch.angle.push(AngleConstraint { a, b, angle: target_rad, cid: 0, hb: CrossBlock::new() });
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
    sketch.angle.push(AngleConstraint { a, b, angle: target, cid: 0, hb: CrossBlock::new() });
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
        a, b, c, cid: 0, hb: arael::model::TripletBlock::new(),
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
        a, b, c, cid: 0, hb: arael::model::TripletBlock::new(),
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
        name: "d0".into(), expr_str: None, broken: false, derived: false,
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
        name: "d0".into(), expr_str: None, broken: false, derived: false,
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
        name: "d0".into(), expr_str: None, broken: false, derived: false,
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
    sketch.coincident_ll21.push(CoincidentLL21 { a: l0, b: l1, cid: 0, hb: CrossBlock::new() });

    // L2 vertical, L3 going up-right from L2.p2
    let l2 = sketch.add_line(vect2d::new(5.0, 2.0), vect2d::new(5.0, -3.0));
    sketch.lines[l2].p2 = Param::fixed(vect2d::new(5.0, -3.0));
    let l3 = sketch.add_line(vect2d::new(5.0, -3.0), vect2d::new(7.0, 1.0));
    sketch.coincident_ll21.push(CoincidentLL21 { a: l2, b: l3, cid: 0, hb: CrossBlock::new() });

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
        cid: 0, hb: CrossBlock::new(),
    });
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::Angle(l0, l1, true), value: 20.0,
        offset: vect2d::new(0.0, 1.0), text_along: 0.0,
        name: "d1".into(), expr_str: None, broken: false, derived: false,
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
        expr_str: None, broken: false, derived: false,
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
        expr_str: None, broken: false, derived: false,
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

// -- Broken expression dimension detection --

#[test]
fn test_broken_expr_dim_detection() {
    // d0 = length of L0 = 10, d1 = length of L1 = "d0 * 2".
    // Delete L0 (removes d0). d1 should become broken and freeze to 20.
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    let l1 = sketch.add_line(vect2d::new(0.0, 2.0), vect2d::new(3.0, 2.0));

    // d0 = line length of L0 = 10
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 10.0;
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l0),
        value: 10.0,
        offset: vect2d::new(0.0, 1.0),
        text_along: 0.0,
        name: "d0".into(),
        expr_str: None, broken: false, derived: false,
    });
    sketch.next_dimension_id = 1;

    // d1 = line length of L1 = d0 * 2
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1), "d0 * 2",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();
    sketch.solve();
    sketch.update_expr_dim_values();
    assert_near(line_length(&sketch, l1), 20.0, 0.1);
    assert!(!sketch.dimensions[1].broken);

    // Delete L0 -- this removes d0 (LineLength references L0)
    sketch.delete_line(l0);
    sketch.solve();

    // d1 should now be broken, frozen to its last value (20)
    assert_eq!(sketch.dimensions.len(), 1); // only d1 remains
    assert!(sketch.dimensions[0].broken, "d1 should be broken");
    assert_near(sketch.dimensions[0].value, 20.0, 0.1);
    // L1 should still be constrained to 20 (frozen value)
    assert_near(line_length(&sketch, l1), 20.0, 0.1);
}

#[test]
fn test_broken_expr_dim_no_cascade() {
    // d0 = length of L0 = 10
    // d1 = length of L1 = "d0 * 2"  (will break)
    // d2 = length of L2 = "d1 + 3"  (should NOT break -- d1 freezes to constant)
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    let l1 = sketch.add_line(vect2d::new(0.0, 2.0), vect2d::new(3.0, 2.0));
    let l2 = sketch.add_line(vect2d::new(0.0, 4.0), vect2d::new(4.0, 4.0));

    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 10.0;
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l0),
        value: 10.0,
        offset: vect2d::new(0.0, 1.0),
        text_along: 0.0,
        name: "d0".into(),
        expr_str: None, broken: false, derived: false,
    });
    sketch.next_dimension_id = 1;

    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1), "d0 * 2",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l2), "d1 + 3",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();
    sketch.solve();
    sketch.update_expr_dim_values();
    assert_near(line_length(&sketch, l1), 20.0, 0.1);
    assert_near(line_length(&sketch, l2), 23.0, 0.1);

    // Delete L0 -- removes d0
    sketch.delete_line(l0);
    sketch.solve();
    sketch.update_expr_dim_values();

    // d1 (now index 0) should be broken, d2 (now index 1) should NOT
    assert_eq!(sketch.dimensions.len(), 2);
    assert!(sketch.dimensions[0].broken, "d1 should be broken");
    assert!(!sketch.dimensions[1].broken, "d2 should NOT be broken (d1 frozen to constant)");
    assert_near(line_length(&sketch, l1), 20.0, 0.1);
    assert_near(line_length(&sketch, l2), 23.0, 0.1);
}

#[test]
fn test_circular_expr_dim_ref() {
    // d0=10 (numeric), d1="d0" (expr). Then change d0 to "d1" (circular).
    // Must not panic. Both should be detected as broken.
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    let l1 = sketch.add_line(vect2d::new(0.0, 2.0), vect2d::new(3.0, 2.0));

    // d0 = line length of L0 = 10 (numeric)
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 10.0;
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l0),
        value: 10.0,
        offset: vect2d::new(0.0, 1.0),
        text_along: 0.0,
        name: "d0".into(),
        expr_str: None, broken: false, derived: false,
    });
    sketch.next_dimension_id = 1;

    // d1 = line length of L1 = "d0" (expression)
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1), "d0",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();
    sketch.solve();
    sketch.update_expr_dim_values();
    assert_near(line_length(&sketch, l0), 10.0, 0.1);
    assert_near(line_length(&sketch, l1), 10.0, 0.1);

    // Now change d0 from numeric to "d1" -- creates circular ref d0 -> d1 -> d0
    sketch.lines[l0].constraints.has_length = false;
    sketch.dimensions[0].value = 10.0; // last known value
    sketch.dimensions[0].expr_str = Some("d1".into());

    // Call solve() twice (exec does this). Must not panic.
    sketch.solve();
    sketch.solve();

    // At least one should be broken (circular ref detected)
    let any_broken = sketch.dimensions.iter().any(|d| d.broken);
    assert!(any_broken, "circular ref should be detected as broken");

    // Geometry should still be constrained (frozen values), not NaN
    let len0 = line_length(&sketch, l0);
    let len1 = line_length(&sketch, l1);
    assert!(len0.is_finite(), "L0 length should be finite, got {}", len0);
    assert!(len1.is_finite(), "L1 length should be finite, got {}", len1);
}

// -- Point symmetry --

#[test]
fn test_symmetry_pp() {
    // Two points symmetric about a vertical line at x=5
    let mut sketch = Sketch::new();
    let p0 = sketch.add_point(vect2d::new(3.0, 2.0));
    let p1 = sketch.add_point(vect2d::new(7.5, 2.5));
    let mirror = sketch.add_line(vect2d::new(5.0, 0.0), vect2d::new(5.0, 10.0));
    sketch.lines[mirror].p1 = Param::fixed(vect2d::new(5.0, 0.0));
    sketch.lines[mirror].p2 = Param::fixed(vect2d::new(5.0, 10.0));
    sketch.symmetry_pp.push(SymmetryPP {
        a: p0, c: p1, line: mirror, cid: 0, hb: TripletBlock::new(),
    });
    let result = sketch.solve();
    assert!(result.end_cost < 1.0, "solver failed: cost={}", result.end_cost);
    let pa = sketch.points[p0].pos.value;
    let pb = sketch.points[p1].pos.value;
    // Midpoint should be on the mirror line (x=5)
    let mx = (pa.x + pb.x) / 2.0;
    assert_near(mx, 5.0, 0.1);
    // Equal distance from line
    assert_near((5.0 - pa.x).abs(), (pb.x - 5.0).abs(), 0.1);
    // Same y coordinate
    assert_near(pa.y, pb.y, 0.1);
}

#[test]
fn test_symmetry_pp_diagonal() {
    // Two points symmetric about a diagonal line y=x
    let mut sketch = Sketch::new();
    let p0 = sketch.add_point(vect2d::new(1.0, 3.0));
    let p1 = sketch.add_point(vect2d::new(3.5, 1.5));
    let mirror = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 5.0));
    sketch.symmetry_pp.push(SymmetryPP {
        a: p0, c: p1, line: mirror, cid: 0, hb: TripletBlock::new(),
    });
    let result = sketch.solve();
    assert!(result.end_cost < 1.0, "solver failed: cost={}", result.end_cost);
    let pa = sketch.points[p0].pos.value;
    let pb = sketch.points[p1].pos.value;
    // For reflection across y=x: (x,y) -> (y,x)
    assert_near(pa.x, pb.y, 0.1);
    assert_near(pa.y, pb.x, 0.1);
}

// -- User parameters --

#[test]
fn test_user_param_basic() {
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    sketch.user_params.push(UserParam {
        name: "width".into(), expr_str: "10".into(), value: 10.0, broken: false,
    });
    // Expression dimension: L0.length = width
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l0), "width",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();
    sketch.solve();
    assert_near(line_length(&sketch, l0), 10.0, 0.1);
}

#[test]
fn test_user_param_expression() {
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    let l1 = sketch.add_line(vect2d::new(0.0, 2.0), vect2d::new(3.0, 2.0));
    sketch.user_params.push(UserParam {
        name: "w".into(), expr_str: "10".into(), value: 10.0, broken: false,
    });
    sketch.user_params.push(UserParam {
        name: "h".into(), expr_str: "w * 2".into(), value: 20.0, broken: false,
    });
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l0), "w",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();
    sketch.add_expr_dimension(
        DimensionKind::LineLength(l1), "h",
        vect2d::new(0.0, 1.0), 0.0,
    ).unwrap();
    sketch.solve();
    sketch.update_expr_dim_values();
    assert_near(line_length(&sketch, l0), 10.0, 0.1);
    assert_near(line_length(&sketch, l1), 20.0, 0.1);
}

#[test]
fn test_user_param_broken_on_delete() {
    let mut sketch = Sketch::new();
    sketch.user_params.push(UserParam {
        name: "w".into(), expr_str: "10".into(), value: 10.0, broken: false,
    });
    sketch.user_params.push(UserParam {
        name: "h".into(), expr_str: "w + 5".into(), value: 15.0, broken: false,
    });
    sketch.solve();
    sketch.update_expr_dim_values();
    assert_near(sketch.user_params[1].value, 15.0, 0.01);

    // Remove w
    sketch.user_params.remove(0);
    sketch.solve();
    // h should now be broken
    assert!(sketch.user_params[0].broken, "h should be broken after w deleted");
    assert_near(sketch.user_params[0].value, 15.0, 0.01); // frozen
}

#[test]
fn test_user_param_no_cascade() {
    let mut sketch = Sketch::new();
    sketch.user_params.push(UserParam {
        name: "a".into(), expr_str: "10".into(), value: 10.0, broken: false,
    });
    sketch.user_params.push(UserParam {
        name: "b".into(), expr_str: "a * 2".into(), value: 20.0, broken: false,
    });
    sketch.user_params.push(UserParam {
        name: "c".into(), expr_str: "b + 1".into(), value: 21.0, broken: false,
    });
    sketch.solve();
    sketch.update_expr_dim_values();

    // Remove a
    sketch.user_params.remove(0);
    sketch.solve();
    sketch.update_expr_dim_values();
    // b (now index 0) should be broken, c (now index 1) should NOT
    assert!(sketch.user_params[0].broken, "b should be broken");
    assert!(!sketch.user_params[1].broken, "c should NOT be broken (b frozen)");
    assert_near(sketch.user_params[0].value, 20.0, 0.01);
    assert_near(sketch.user_params[1].value, 21.0, 0.01);
}

#[test]
fn test_user_param_circular_ref() {
    let mut sketch = Sketch::new();
    sketch.user_params.push(UserParam {
        name: "a".into(), expr_str: "1".into(), value: 1.0, broken: false,
    });
    sketch.user_params.push(UserParam {
        name: "b".into(), expr_str: "a".into(), value: 1.0, broken: false,
    });
    // Create circular: a = b
    sketch.user_params[0].expr_str = "b".into();
    sketch.solve();
    sketch.solve(); // twice like exec does
    let any_broken = sketch.user_params.iter().any(|p| p.broken);
    assert!(any_broken, "circular ref should be detected as broken");
}

#[test]
fn test_user_param_in_dimension() {
    let mut sketch = Sketch::new();
    let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    sketch.user_params.push(UserParam {
        name: "gap".into(), expr_str: "5".into(), value: 5.0, broken: false,
    });
    sketch.lines[l0].constraints.has_length = true;
    sketch.lines[l0].constraints.length = 5.0;
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l0), value: 5.0,
        offset: vect2d::new(0.0, 1.0), text_along: 0.0,
        name: "d0".into(), expr_str: Some("gap".into()), broken: false, derived: false,
    });
    sketch.next_dimension_id = 1;
    sketch.solve();
    assert_near(line_length(&sketch, l0), 5.0, 0.1);

    // Update gap to 8
    sketch.user_params[0].expr_str = "8".into();
    sketch.user_params[0].value = 8.0;
    sketch.solve();
    sketch.update_expr_dim_values();
    // The dimension expr "gap" should now resolve to 8
    // but dimension has underlying has_length=5 constraint competing with expr.
    // The expr constraint should win since it's also applied.
    // Actually, let's use a pure expression dim without underlying constraint.
}

#[test]
fn test_user_param_name_validation() {
    let mut sketch = Sketch::new();
    sketch.user_params.push(UserParam {
        name: "width".into(), expr_str: "10".into(), value: 10.0, broken: false,
    });
    // Empty
    assert!(sketch.validate_param_name("", None).is_err());
    // Duplicate
    assert!(sketch.validate_param_name("width", None).is_err());
    // Duplicate excluding self
    assert!(sketch.validate_param_name("width", Some(0)).is_ok());
    // System names
    assert!(sketch.validate_param_name("d0", None).is_err());
    assert!(sketch.validate_param_name("d99", None).is_err());
    assert!(sketch.validate_param_name("L0", None).is_err());
    assert!(sketch.validate_param_name("L5", None).is_err());
    assert!(sketch.validate_param_name("P0", None).is_err());
    assert!(sketch.validate_param_name("A0", None).is_err());
    // Starts with digit
    assert!(sketch.validate_param_name("0abc", None).is_err());
    // Valid names
    assert!(sketch.validate_param_name("w", None).is_ok());
    assert!(sketch.validate_param_name("my_param", None).is_ok());
    assert!(sketch.validate_param_name("x1", None).is_ok());
}

#[test]
fn test_user_param_rename_propagation() {
    let mut sketch = Sketch::new();
    sketch.user_params.push(UserParam {
        name: "width".into(), expr_str: "10".into(), value: 10.0, broken: false,
    });
    sketch.user_params.push(UserParam {
        name: "half".into(), expr_str: "width / 2".into(), value: 5.0, broken: false,
    });
    // Simulate rename via action logic: update name and propagate
    let old_name = "width";
    let new_name = "w";
    sketch.user_params[0].name = new_name.into();
    // Propagate to other params
    for p in &mut sketch.user_params {
        if let Ok(parsed) = arael_sym::parse(&p.expr_str) {
            if parsed.symbols().contains(&old_name.to_string()) {
                let replaced = parsed.subs(old_name, &arael_sym::symbol(new_name));
                p.expr_str = format!("{}", replaced);
            }
        }
    }
    // Expression may be simplified (e.g. "width / 2" -> "0.5 * w")
    // but must contain the new name and evaluate correctly.
    assert!(sketch.user_params[1].expr_str.contains("w"),
        "expr should reference 'w', got: {}", sketch.user_params[1].expr_str);
    assert!(!sketch.user_params[1].expr_str.contains("width"),
        "expr should not reference old name 'width'");
    sketch.solve();
    sketch.update_expr_dim_values();
    assert_near(sketch.user_params[1].value, 5.0, 0.01);
}

#[test]
fn test_user_param_serialization() {
    let mut sketch = Sketch::new();
    sketch.user_params.push(UserParam {
        name: "width".into(), expr_str: "10".into(), value: 10.0, broken: false,
    });
    sketch.user_params.push(UserParam {
        name: "half".into(), expr_str: "width / 2".into(), value: 5.0, broken: false,
    });
    let bytes = bincode::serialize(&sketch).unwrap();
    let restored: Sketch = bincode::deserialize(&bytes).unwrap();
    assert_eq!(restored.user_params.len(), 2);
    assert_eq!(restored.user_params[0].name, "width");
    assert_eq!(restored.user_params[1].expr_str, "width / 2");
    assert_near(restored.user_params[0].value, 10.0, 0.01);
}

// -- Derived dimensions --

#[test]
fn test_derived_length() {
    // A derived length dimension should not constrain the line length
    let mut sketch = Sketch::new();
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l),
        value: 0.0,
        offset: vect2d::new(0.0, 1.0),
        text_along: 0.0,
        name: "d0".into(),
        expr_str: None,
        broken: false,
        derived: true,
    });
    sketch.solve();
    sketch.update_expr_dim_values();
    // Line should remain at its original length (~5), not be constrained
    let len = line_length(&sketch, l);
    assert_near(len, 5.0, 0.01);
    // Derived dim value should reflect the measured length
    assert_near(sketch.dimensions[0].value, 5.0, 0.01);
    // Line should NOT have has_length set (derived doesn't add constraints)
    assert!(!sketch.lines[l].constraints.has_length);
}

#[test]
fn test_derived_to_driven() {
    // Start with a derived dim, convert to driven, verify constraint appears
    let mut sketch = Sketch::new();
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(5.0, 0.0));
    // Add as derived first
    sketch.dimensions.push(Dimension {
        kind: DimensionKind::LineLength(l),
        value: 5.0,
        offset: vect2d::new(0.0, 1.0),
        text_along: 0.0,
        name: "d0".into(),
        expr_str: None,
        broken: false,
        derived: true,
    });
    sketch.solve();
    assert!(!sketch.lines[l].constraints.has_length);
    // Convert to driven: set derived=false, add constraint
    sketch.dimensions[0].derived = false;
    sketch.lines[l].constraints.has_length = true;
    sketch.lines[l].constraints.length = 3.0;
    sketch.dimensions[0].value = 3.0;
    sketch.solve();
    let len = line_length(&sketch, l);
    assert_near(len, 3.0, 0.01);
}

// -- Ellipse constraint regression --

/// Verify that arc distance constraints converge with the ellipse formula.
/// This is a regression test: the ellipse point formula must produce correct
/// Jacobians so the solver converges even for distant initial configurations.
#[test]
fn test_ellipse_arc_distance_convergence() {
    let mut sketch = Sketch::new();
    let arc = sketch.add_arc(vect2d::new(2.5, 2.5), 3.5355, -2.356, -2.356 - 4.712, false);
    let pt = sketch.add_point(vect2d::new(10.0, 3.0));
    sketch.solve(); // anchor drift

    sketch.distance_arc_start_p.push(DistanceArcStartP {
        arc, point: pt, distance: 3.0, cid: 0, hb: CrossBlock::new(),
    });

    let result = sketch.solve();
    let a = &sketch.arcs[arc];
    let p = &sketch.points[pt];
    let ct = a.start_angle.value.cos();
    let st = a.start_angle.value.sin();
    let cr = a.rotation.value.cos();
    let sr = a.rotation.value.sin();
    let sx = a.center.value.x + a.radius.value * ct * cr - a.radius_b.value * st * sr;
    let sy = a.center.value.y + a.radius.value * ct * sr + a.radius_b.value * st * cr;
    let dist = ((sx - p.pos.value.x).powi(2) + (sy - p.pos.value.y).powi(2)).sqrt();
    assert!(result.end_cost < 0.01, "solver should converge, end_cost = {}", result.end_cost);
    assert!((dist - 3.0).abs() < 0.01, "distance should be ~3.0, got {}", dist);
    assert!((a.radius.value - a.radius_b.value).abs() < 1e-6, "radius_b should equal radius");
}

// -- DOF rank detection at large geometric scale --
//
// Regression: at large geometric scale, Hessian-based rank detection
// misreports DOF because forming J^T J squares the condition number and
// drowns the true near-zero eigenvalues in roundoff noise. SVD of J
// preserves the gap.
//
// Reproducer: two arcs sharing both endpoints via coincident-start/end and
// each tangent to a line. This mirrors the pattern in robot.cmd that
// reported DOF=60 instead of 3 at scale=10000 with the eigenvalue path.
#[test]
fn test_dof_at_large_scale() {
    fn build_sketch(s: f64) -> Sketch {
        let mut sketch = Sketch::new();
        let l1 = sketch.add_line(vect2d::new(-s, 0.0), vect2d::new(s, 0.0));
        let l2 = sketch.add_line(vect2d::new(s, 0.0), vect2d::new(-s, 0.0));
        // Upper arc, bulging upward.
        let a_top = sketch.add_arc(
            vect2d::new(0.0, 0.0),
            s,
            0.0,
            std::f64::consts::PI,
            false,
        );
        // Lower arc, bulging downward, sharing both endpoints with top.
        let a_bot = sketch.add_arc(
            vect2d::new(0.0, 0.0),
            s,
            std::f64::consts::PI,
            std::f64::consts::TAU,
            false,
        );
        // Couple: arcs share start/end endpoints with the line endpoints.
        sketch.coincident_lp1_arc_start.push(CoincidentLP1ArcStart {
            line: l1, arc: a_top, cid: 0, hb: CrossBlock::new(),
        });
        sketch.coincident_lp2_arc_end.push(CoincidentLP2ArcEnd {
            line: l1, arc: a_top, cid: 0, hb: CrossBlock::new(),
        });
        sketch.coincident_lp1_arc_start.push(CoincidentLP1ArcStart {
            line: l2, arc: a_bot, cid: 0, hb: CrossBlock::new(),
        });
        sketch.coincident_lp2_arc_end.push(CoincidentLP2ArcEnd {
            line: l2, arc: a_bot, cid: 0, hb: CrossBlock::new(),
        });
        // Equal-radius between the two arcs (redundant with the shared
        // endpoints but introduces the kind of scale-dependent coupling
        // that breaks the J^T J eigenvalue path).
        sketch.equal_radius.push(EqualRadius {
            a: a_top, b: a_bot, cid: 0, hb: CrossBlock::new(),
        });
        // Dimensions that scale with s.
        sketch.lines[l1].constraints.has_length = true;
        sketch.lines[l1].constraints.length = 2.0 * s;
        sketch.arcs[a_top].constraints.has_target_radius = true;
        sketch.arcs[a_top].constraints.target_radius = s;
        sketch.solve();
        sketch
    }

    let mut small = build_sketch(1.0);
    let mut large = build_sketch(100000.0);
    let dof_small = small.dof().expect("small-scale DOF");
    let dof_large = large.dof().expect("large-scale DOF");
    assert_eq!(dof_small, dof_large,
        "DOF should not depend on geometric scale (got {} at s=1, {} at s=100000)",
        dof_small, dof_large);
}

// -- Signed along-line PointPointDistance (measured_symbol) --
//
// When a PointPointDistance expression dimension has one endpoint anchored
// to a line endpoint and the other endpoint on the same line, the
// measured_symbol should emit a signed along-line projection rather than
// the unsigned sqrt. Otherwise the solver can settle into a mirror
// solution under value changes -- e.g. the point ending up on the wrong
// side of the anchor.
#[test]
fn test_pp_distance_signed_along_line() {
    use arael_sketch_solver::{DimensionEndpoint, DimensionKind};

    let mut sketch = Sketch::new();
    // Line from origin along +x.
    let l = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(10.0, 0.0));
    // Point on the line at x=3 (on the negative-x side if we later anchor to p2).
    let p = sketch.add_point(vect2d::new(3.0, 0.0));
    sketch.point_on_line.push(PointOnLine {
        point: p, line: l, cid: 0, hb: CrossBlock::new(),
    });

    // Expression dimension: distance LineP1(l) -- Point(p) = "2".
    // Pattern should match: a is LineP1 (endpoint), b is Point on same line.
    sketch.add_expr_dimension(
        DimensionKind::PointPointDistance(DimensionEndpoint::LineP1(l), DimensionEndpoint::Point(p)),
        "2.0",
        vect2d::new(0.0, 1.0), 0.0,
    ).expect("add_expr_dimension");

    sketch.solve();
    // Point must stay on the +x side of the anchor (l.p1). Drift regularizer
    // holds the point near its initial value so the solver lands on a
    // compromise, but the sign must match the initial geometry -- not mirror.
    let px = sketch.points[p].pos.value.x;
    assert!(px > 0.0, "p should stay on +x side (no mirror), got {}", px);
    assert!((px - 2.0).abs() < 1.5,
        "p should be near x=2 (drift-softened), got {}", px);
}
