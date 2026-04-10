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

/// SVG endpoint-to-center parameterization for elliptic arcs.
/// Given start/end points, radii, rotation, and arc selection flags,
/// returns (center, start_angle, end_angle, rx, ry) or None if degenerate.
///
/// Implements the algorithm from SVG spec F.6.5-F.6.6:
/// https://www.w3.org/TR/SVG2/implnote.html#ArcConversionEndpointToCenter
pub fn svg_arc_to_center(
    p1: vect2d, p2: vect2d, mut rx: f64, mut ry: f64,
    rotation: f64, large_arc: bool, sweep: bool,
) -> Option<(vect2d, f64, f64, f64, f64)> {

    // F.6.2: if endpoints are identical, skip
    if (p1.x - p2.x).abs() < 1e-12 && (p1.y - p2.y).abs() < 1e-12 { return None; }

    rx = rx.abs();
    ry = ry.abs();
    if rx < 1e-12 || ry < 1e-12 { return None; }

    let cos_r = rotation.cos();
    let sin_r = rotation.sin();

    // F.6.5.1: compute (x1', y1')
    let dx = (p1.x - p2.x) / 2.0;
    let dy = (p1.y - p2.y) / 2.0;
    let x1p = cos_r * dx + sin_r * dy;
    let y1p = -sin_r * dx + cos_r * dy;

    // F.6.6.2: ensure radii are large enough
    let lambda = (x1p * x1p) / (rx * rx) + (y1p * y1p) / (ry * ry);
    if lambda > 1.0 {
        let s = lambda.sqrt();
        rx *= s;
        ry *= s;
    }

    // F.6.5.2: compute (cx', cy')
    let rx2 = rx * rx;
    let ry2 = ry * ry;
    let x1p2 = x1p * x1p;
    let y1p2 = y1p * y1p;
    let num = (rx2 * ry2 - rx2 * y1p2 - ry2 * x1p2).max(0.0);
    let den = rx2 * y1p2 + ry2 * x1p2;
    let sq = if den.abs() < 1e-30 { 0.0 } else { (num / den).sqrt() };
    let sign = if large_arc == sweep { -1.0 } else { 1.0 };
    let cxp = sign * sq * rx * y1p / ry;
    let cyp = sign * sq * -ry * x1p / rx;

    // F.6.5.3: compute center (cx, cy)
    let mx = (p1.x + p2.x) / 2.0;
    let my = (p1.y + p2.y) / 2.0;
    let cx = cos_r * cxp - sin_r * cyp + mx;
    let cy = sin_r * cxp + cos_r * cyp + my;

    // F.6.5.5-6: compute start angle and sweep
    let ux = (x1p - cxp) / rx;
    let uy = (y1p - cyp) / ry;
    let vx = (-x1p - cxp) / rx;
    let vy = (-y1p - cyp) / ry;

    let start_angle = uy.atan2(ux);
    let mut sweep_angle = {
        let dot = ux * vx + uy * vy;
        let len = (ux * ux + uy * uy).sqrt() * (vx * vx + vy * vy).sqrt();
        let cos_a = (dot / len).clamp(-1.0, 1.0);
        let cross = ux * vy - uy * vx;
        let a = cos_a.acos();
        if cross < 0.0 { -a } else { a }
    };

    // Adjust sweep for sweep flag
    if !sweep && sweep_angle > 0.0 {
        sweep_angle -= std::f64::consts::TAU;
    } else if sweep && sweep_angle < 0.0 {
        sweep_angle += std::f64::consts::TAU;
    }

    let end_angle = start_angle + sweep_angle;
    Some((vect2d::new(cx, cy), start_angle, end_angle, rx, ry))
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
    let t_min = sa.min(sa + span);
    let t_max = sa.max(sa + span);
    let n = 64;
    let mut best_t = sa;
    let mut best_d = f64::MAX;
    for i in 0..=n {
        let t = sa + span * (i as f64 / n as f64);
        let q = arc_point_at(a, t);
        let d = (p.x - q.x).powi(2) + (p.y - q.y).powi(2);
        if d < best_d { best_d = d; best_t = t; }
    }
    // Newton refinement (minimize squared distance), clamped to arc range
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
        best_t = (best_t - f / df).clamp(t_min, t_max);
    }
    // Also check endpoints explicitly
    let sp = arc_start_pos(a);
    let ep = arc_end_pos(a);
    let nearest = arc_point_at(a, best_t);
    let dist = ((p.x - nearest.x).powi(2) + (p.y - nearest.y).powi(2)).sqrt();
    let ds = ((p.x - sp.x).powi(2) + (p.y - sp.y).powi(2)).sqrt();
    let de = ((p.x - ep.x).powi(2) + (p.y - ep.y).powi(2)).sqrt();
    if ds < dist && ds < de { (ds, sp) }
    else if de < dist { (de, ep) }
    else { (dist, nearest) }
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
