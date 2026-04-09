// Coordinate transforms, snapping, hit testing helpers for the sketch editor.

use arael::vect::vect2d;
use arael_sketch_solver::*;

// Distance from a sketch-space point to a line segment
pub fn point_to_segment_dist(p: vect2d, a: vect2d, b: vect2d) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-12 {
        return ((p.x - a.x).powi(2) + (p.y - a.y).powi(2)).sqrt();
    }
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
    let t = t.clamp(0.0, 1.0);
    let proj_x = a.x + t * dx;
    let proj_y = a.y + t * dy;
    ((p.x - proj_x).powi(2) + (p.y - proj_y).powi(2)).sqrt()
}

// Compute circumscribed circle arc from 3 points (start, end, mid on arc).
// Returns (center, radius, start_angle, end_angle, ccw) or None if collinear.
// `ccw` is true if the arc from p1 to p2 passing through p3 goes counter-clockwise.
pub fn circumscribed_arc(p1: vect2d, p2: vect2d, p3: vect2d) -> Option<(vect2d, f64, f64, f64, bool)> {
    let ax = p1.x; let ay = p1.y;
    let bx = p2.x; let by = p2.y;
    let cx = p3.x; let cy = p3.y;
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-12 { return None; } // collinear
    let aa = ax * ax + ay * ay;
    let bb = bx * bx + by * by;
    let cc = cx * cx + cy * cy;
    let ux = (aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / d;
    let uy = (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / d;
    let center = vect2d::new(ux, uy);
    let radius = ((ax - ux).powi(2) + (ay - uy).powi(2)).sqrt();

    // Angles from center to start (p1) and end (p2)
    let sa = (ay - uy).atan2(ax - ux);
    let ea = (by - uy).atan2(bx - ux);

    // Check if mid point (p3) is on the arc going sa->ea counterclockwise.
    let ma = (cy - uy).atan2(cx - ux);

    // Normalize angle difference to [0, 2*PI)
    let norm = |a: f64| -> f64 { let r = a % std::f64::consts::TAU; if r < 0.0 { r + std::f64::consts::TAU } else { r } };
    let span_ccw = norm(ea - sa);
    let mid_ccw = norm(ma - sa);

    // Keep start/end as-is (matching user's p1/p2), record direction
    let ccw = mid_ccw < span_ccw;
    Some((center, radius, sa, ea, ccw))
}

// Distance from point to arc/ellipse curve. Returns (distance, nearest point on curve).
pub fn point_to_arc_dist(p: vect2d, a: &Arc) -> (f64, vect2d) {
    if a.is_ellipse {
        return point_to_ellipse_dist(p, a);
    }
    let dx = p.x - a.center.value.x;
    let dy = p.y - a.center.value.y;
    let dist_to_center = (dx * dx + dy * dy).sqrt();
    let angle = dy.atan2(dx);
    let r = a.radius.value;

    if a.closed {
        // Full circle: nearest point is projection onto circle
        if dist_to_center < 1e-12 {
            return (r, vect2d::new(a.center.value.x + r, a.center.value.y));
        }
        let proj = vect2d::new(
            a.center.value.x + r * dx / dist_to_center,
            a.center.value.y + r * dy / dist_to_center,
        );
        ((dist_to_center - r).abs(), proj)
    } else {
        // Partial arc: check if angle falls within arc range
        let sa = a.start_angle.value;
        let ea = a.end_angle.value;
        let norm = |v: f64| -> f64 { let rv = v % std::f64::consts::TAU; if rv < 0.0 { rv + std::f64::consts::TAU } else { rv } };
        let (span, a_norm) = if a.ccw {
            (norm(ea - sa), norm(angle - sa))
        } else {
            (norm(sa - ea), norm(sa - angle))
        };

        if a_norm <= span {
            // Angle is within arc range
            if dist_to_center < 1e-12 {
                let proj = vect2d::new(
                    a.center.value.x + r * angle.cos(),
                    a.center.value.y + r * angle.sin(),
                );
                return (r, proj);
            }
            let proj = vect2d::new(
                a.center.value.x + r * dx / dist_to_center,
                a.center.value.y + r * dy / dist_to_center,
            );
            ((dist_to_center - r).abs(), proj)
        } else {
            // Outside arc range: nearest is one of the endpoints
            let sp = arc_start_pos(a);
            let ep = arc_end_pos(a);
            let ds = ((p.x - sp.x).powi(2) + (p.y - sp.y).powi(2)).sqrt();
            let de = ((p.x - ep.x).powi(2) + (p.y - ep.y).powi(2)).sqrt();
            if ds < de { (ds, sp) } else { (de, ep) }
        }
    }
}

/// Nearest point on an ellipse via tessellation + Newton refinement.
fn point_to_ellipse_dist(p: vect2d, a: &Arc) -> (f64, vect2d) {
    let sa = a.start_angle.value;
    let ea = a.end_angle.value;
    let span = if a.closed { std::f64::consts::TAU } else { ea - sa };
    let n = 64;
    let mut best_t = sa;
    let mut best_d = f64::MAX;
    for i in 0..=n {
        let t = sa + span * (i as f64 / n as f64);
        let q = arc_point_at(a, t);
        let d = (p.x - q.x).powi(2) + (p.y - q.y).powi(2);
        if d < best_d { best_d = d; best_t = t; }
    }
    // Newton refinement (minimize squared distance)
    let dt = 1e-6;
    for _ in 0..8 {
        let q = arc_point_at(a, best_t);
        let qp = arc_point_at(a, best_t + dt);
        let qm = arc_point_at(a, best_t - dt);
        let f = (p.x - q.x) * (qp.x - qm.x) / (2.0 * dt) + (p.y - q.y) * (qp.y - qm.y) / (2.0 * dt);
        let df = -((qp.x - qm.x).powi(2) + (qp.y - qm.y).powi(2)) / (4.0 * dt * dt)
            + (p.x - q.x) * (qp.x - 2.0 * q.x + qm.x) / (dt * dt)
            + (p.y - q.y) * (qp.y - 2.0 * q.y + qm.y) / (dt * dt);
        if df.abs() < 1e-20 { break; }
        best_t -= f / df;
    }
    let nearest = arc_point_at(a, best_t);
    let dist = ((p.x - nearest.x).powi(2) + (p.y - nearest.y).powi(2)).sqrt();
    (dist, nearest)
}

/// Compute a point on the arc/ellipse at parametric angle t.
pub fn arc_point_at(a: &Arc, t: f64) -> vect2d {
    let ct = t.cos();
    let st = t.sin();
    let cr = a.rotation.value.cos();
    let sr = a.rotation.value.sin();
    vect2d::new(
        a.center.value.x + a.radius.value * ct * cr - a.radius_b.value * st * sr,
        a.center.value.y + a.radius.value * ct * sr + a.radius_b.value * st * cr,
    )
}

pub fn arc_start_pos(a: &Arc) -> vect2d {
    arc_point_at(a, a.start_angle.value)
}

pub fn arc_end_pos(a: &Arc) -> vect2d {
    arc_point_at(a, a.end_angle.value)
}

pub fn project_onto_segment(p: vect2d, a: vect2d, b: vect2d) -> vect2d {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-12 { return a; }
    let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0);
    vect2d::new(a.x + t * dx, a.y + t * dy)
}

/// Intersection of two infinite lines (p1-p2 and p3-p4).
/// Returns midpoint of closest approach if nearly parallel.
pub fn line_line_intersection(p1: vect2d, p2: vect2d, p3: vect2d, p4: vect2d) -> vect2d {
    let d1x = p2.x - p1.x;
    let d1y = p2.y - p1.y;
    let d2x = p4.x - p3.x;
    let d2y = p4.y - p3.y;
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-12 {
        // Nearly parallel -- return midpoint
        return vect2d::new((p1.x + p3.x) / 2.0, (p1.y + p3.y) / 2.0);
    }
    let t = ((p3.x - p1.x) * d2y - (p3.y - p1.y) * d2x) / denom;
    vect2d::new(p1.x + t * d1x, p1.y + t * d1y)
}
