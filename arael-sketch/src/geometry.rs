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
// Returns (center, radius, start_angle, end_angle, swapped) or None if collinear.
// `swapped` is true if start/end angles were swapped (arc goes the other way).
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
    // If not, swap to go the other way.
    let ma = (cy - uy).atan2(cx - ux);

    // Normalize angle difference to [0, 2*PI)
    let norm = |a: f64| -> f64 { let r = a % std::f64::consts::TAU; if r < 0.0 { r + std::f64::consts::TAU } else { r } };
    let span_ccw = norm(ea - sa);
    let mid_ccw = norm(ma - sa);

    if mid_ccw < span_ccw {
        // Mid is on the CCW arc from sa to ea
        Some((center, radius, sa, ea, false))
    } else {
        // Mid is on the other side; swap start/end
        Some((center, radius, ea, sa, true))
    }
}

// Distance from point to arc curve. Returns (distance, nearest point on arc).
pub fn point_to_arc_dist(p: vect2d, a: &Arc) -> (f64, vect2d) {
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
        let span = norm(ea - sa);
        let a_norm = norm(angle - sa);

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

pub fn arc_start_pos(a: &Arc) -> vect2d {
    vect2d::new(
        a.center.value.x + a.radius.value * a.start_angle.value.cos(),
        a.center.value.y + a.radius.value * a.start_angle.value.sin(),
    )
}

pub fn arc_end_pos(a: &Arc) -> vect2d {
    vect2d::new(
        a.center.value.x + a.radius.value * a.end_angle.value.cos(),
        a.center.value.y + a.radius.value * a.end_angle.value.sin(),
    )
}

pub fn project_onto_segment(p: vect2d, a: vect2d, b: vect2d) -> vect2d {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-12 { return a; }
    let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0);
    vect2d::new(a.x + t * dx, a.y + t * dy)
}
