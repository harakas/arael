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
/// <https://www.w3.org/TR/SVG2/implnote.html#ArcConversionEndpointToCenter>
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

/// Compute the tangent direction of an arc/ellipse at parametric angle t.
/// This is the derivative of arc_point_at with respect to t.
pub fn arc_tangent_at(a: &Arc, t: f64) -> vect2d {
    a.tangent_at(t)
}

/// Parametric angle of the arc/ellipse point in the direction of `p`
/// as seen from the center (in the ellipse frame). For a point on the
/// curve this is its parameter.
pub fn arc_param_at_point(a: &Arc, p: vect2d) -> f64 {
    let (s, c) = a.rotation.value.sin_cos();
    let dx = p.x - a.center.value.x;
    let dy = p.y - a.center.value.y;
    let u = dx * c + dy * s;
    let v = -dx * s + dy * c;
    (v / a.radius_b.value.max(1e-12)).atan2(u / a.radius.value.max(1e-12))
}

/// The circle through `s` and `e` whose tangent at `s` is `t`
/// (direction, either sign): center and radius. None when `e` lies on
/// that tangent line (no such circle) or the points coincide.
pub fn circle_tangent_through(s: vect2d, e: vect2d, t: vect2d) -> Option<(vect2d, f64)> {
    let tl = (t.x * t.x + t.y * t.y).sqrt();
    if tl < 1e-12 { return None; }
    let n = vect2d::new(-t.y / tl, t.x / tl);
    let dx = e.x - s.x;
    let dy = e.y - s.y;
    let dn = dx * n.x + dy * n.y;
    if dn.abs() < 1e-12 { return None; }
    // |C - s| = |C - e| with C = s + k n:  k = |e - s|^2 / (2 (e - s) . n)
    let k = (dx * dx + dy * dy) / (2.0 * dn);
    Some((vect2d::new(s.x + k * n.x, s.y + k * n.y), k.abs()))
}

/// Compute a point on the arc/ellipse at parametric angle t.
pub fn arc_point_at(a: &Arc, t: f64) -> vect2d {
    a.point_at(t)
}

pub fn arc_start_pos(a: &Arc) -> vect2d {
    a.start_pos()
}

pub fn arc_end_pos(a: &Arc) -> vect2d {
    a.end_pos()
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

// ---------------------------------------------------------------------------
// Curve-curve intersections (trim / split support)
// ---------------------------------------------------------------------------

/// One crossing between two curves. `t_a` / `t_b` are the parameter of
/// the hit on each curve: `t` in [0,1] for a line segment, the
/// parametric angle for an arc/ellipse.
#[derive(Clone, Copy, Debug)]
pub struct CurveHit {
    pub t_a: f64,
    pub t_b: f64,
    pub pos: vect2d,
}

fn norm_tau(v: f64) -> f64 {
    let r = v % std::f64::consts::TAU;
    if r < 0.0 { r + std::f64::consts::TAU } else { r }
}

const SPAN_EPS: f64 = 1e-9;

/// Bring an absolute parametric angle into the arc's own span
/// coordinate. Returns the in-span parameter (monotonic from
/// start_angle to end_angle in the arc's direction), or None when the
/// angle falls outside a partial arc's span. Closed arcs accept every
/// angle, mapped into [start, start + TAU).
pub fn arc_param_in_span(arc: &Arc, theta: f64) -> Option<f64> {
    let sa = arc.start_angle.value;
    let ea = arc.end_angle.value;
    if arc.closed {
        return Some(sa + norm_tau(theta - sa));
    }
    if arc.ccw {
        let span = ea - sa;
        let off = norm_tau(theta - sa);
        if off <= span + SPAN_EPS { Some(sa + off) } else { None }
    } else {
        let span = sa - ea;
        let off = norm_tau(sa - theta);
        if off <= span + SPAN_EPS { Some(sa - off) } else { None }
    }
}

/// Parametric angle of a position on (or near) the arc's curve:
/// inverse of `arc_point_at` up to radial distance. Uses the arc's
/// local frame, so it is exact for ellipses too.
fn arc_theta_of_pos(arc: &Arc, p: vect2d) -> f64 {
    let rot = arc.rotation.value;
    let (cos_r, sin_r) = (rot.cos(), rot.sin());
    let dx = p.x - arc.center.value.x;
    let dy = p.y - arc.center.value.y;
    let xl = cos_r * dx + sin_r * dy;
    let yl = -sin_r * dx + cos_r * dy;
    let (ra, rb) = arc_radii(arc);
    (yl / rb).atan2(xl / ra)
}

/// Effective (semi-major, semi-minor) radii; circular arcs use
/// `radius` for both (radius_b is only softly tied to it).
fn arc_radii(arc: &Arc) -> (f64, f64) {
    if arc.is_ellipse {
        (arc.radius.value, arc.radius_b.value)
    } else {
        (arc.radius.value, arc.radius.value)
    }
}

/// Segment-segment intersection. Returns the single transversal
/// crossing with parameters on both segments, or None for parallel /
/// collinear / out-of-range configurations.
pub fn intersect_segments(a1: vect2d, a2: vect2d, b1: vect2d, b2: vect2d) -> Option<CurveHit> {
    let d1x = a2.x - a1.x;
    let d1y = a2.y - a1.y;
    let d2x = b2.x - b1.x;
    let d2y = b2.y - b1.y;
    let denom = d1x * d2y - d1y * d2x;
    if denom.abs() < 1e-12 { return None; }
    let t_a = ((b1.x - a1.x) * d2y - (b1.y - a1.y) * d2x) / denom;
    let t_b = ((b1.x - a1.x) * d1y - (b1.y - a1.y) * d1x) / denom;
    let eps = 1e-9;
    if !(-eps..=1.0 + eps).contains(&t_a) || !(-eps..=1.0 + eps).contains(&t_b) {
        return None;
    }
    Some(CurveHit {
        t_a: t_a.clamp(0.0, 1.0),
        t_b: t_b.clamp(0.0, 1.0),
        pos: vect2d::new(a1.x + t_a * d1x, a1.y + t_a * d1y),
    })
}

/// Segment-arc intersection, exact for circles and ellipses alike:
/// the arc's local frame scales the ellipse to a circle, where the
/// segment-circle quadratic applies. `t_a` is the segment parameter,
/// `t_b` the arc's in-span parametric angle. Hits outside the segment
/// or the arc's span are filtered out.
pub fn intersect_segment_arc(p1: vect2d, p2: vect2d, arc: &Arc) -> Vec<CurveHit> {
    let (ra, rb) = arc_radii(arc);
    if ra < 1e-12 || rb < 1e-12 { return Vec::new(); }
    let rot = arc.rotation.value;
    let (cos_r, sin_r) = (rot.cos(), rot.sin());
    let c = arc.center.value;
    // Local frame, scaled so the curve is a unit circle.
    let to_unit = |p: vect2d| {
        let dx = p.x - c.x;
        let dy = p.y - c.y;
        vect2d::new((cos_r * dx + sin_r * dy) / ra, (-sin_r * dx + cos_r * dy) / rb)
    };
    let a = to_unit(p1);
    let b = to_unit(p2);
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let qa = dx * dx + dy * dy;
    if qa < 1e-24 { return Vec::new(); }
    let qb = 2.0 * (a.x * dx + a.y * dy);
    let qc = a.x * a.x + a.y * a.y - 1.0;
    let disc = qb * qb - 4.0 * qa * qc;
    if disc < 0.0 { return Vec::new(); }
    let sq = disc.sqrt();
    let mut out = Vec::new();
    let mut roots = vec![(-qb - sq) / (2.0 * qa)];
    if sq > 1e-12 { roots.push((-qb + sq) / (2.0 * qa)); }
    let eps = 1e-9;
    for t in roots {
        if !(-eps..=1.0 + eps).contains(&t) { continue; }
        let t = t.clamp(0.0, 1.0);
        let u = vect2d::new(a.x + t * dx, a.y + t * dy);
        let theta = u.y.atan2(u.x);
        if let Some(tb) = arc_param_in_span(arc, theta) {
            out.push(CurveHit {
                t_a: t,
                t_b: tb,
                pos: vect2d::new(
                    p1.x + t * (p2.x - p1.x),
                    p1.y + t * (p2.y - p1.y),
                ),
            });
        }
    }
    out
}

/// Arc-arc intersection. Circle-circle goes through the closed-form
/// two-circle construction; any pair involving an ellipse falls back
/// to sampling arc `a`'s span for sign changes of `b`'s implicit
/// equation, refined by bisection (transversal crossings only).
/// `t_a` / `t_b` are in-span parametric angles.
pub fn intersect_arcs(a: &Arc, b: &Arc) -> Vec<CurveHit> {
    if !a.is_ellipse && !b.is_ellipse {
        return intersect_circle_circle(a, b);
    }
    // Implicit "inside/outside" function of b, sign-stable.
    let (rb_a, rb_b) = arc_radii(b);
    if rb_a < 1e-12 || rb_b < 1e-12 { return Vec::new(); }
    let rot = b.rotation.value;
    let (cos_r, sin_r) = (rot.cos(), rot.sin());
    let cb = b.center.value;
    let g = |p: vect2d| {
        let dx = p.x - cb.x;
        let dy = p.y - cb.y;
        let xl = (cos_r * dx + sin_r * dy) / rb_a;
        let yl = (-sin_r * dx + cos_r * dy) / rb_b;
        xl * xl + yl * yl - 1.0
    };
    let sa = a.start_angle.value;
    let ea = if a.closed { sa + std::f64::consts::TAU } else { a.end_angle.value };
    let n = 256;
    let step = (ea - sa) / n as f64;
    if step.abs() < 1e-15 { return Vec::new(); }
    let mut out: Vec<CurveHit> = Vec::new();
    let mut prev_t = sa;
    let mut prev_g = g(a.point_at(prev_t));
    for i in 1..=n {
        let t = sa + step * i as f64;
        let gv = g(a.point_at(t));
        if prev_g == 0.0 || prev_g * gv < 0.0 {
            // Bisect [prev_t, t] to the crossing.
            let (mut lo, mut hi) = (prev_t, t);
            let mut g_lo = prev_g;
            for _ in 0..60 {
                let mid = 0.5 * (lo + hi);
                let gm = g(a.point_at(mid));
                if g_lo * gm <= 0.0 { hi = mid; } else { lo = mid; g_lo = gm; }
            }
            let t_hit = 0.5 * (lo + hi);
            let pos = a.point_at(t_hit);
            let theta_b = arc_theta_of_pos(b, pos);
            if let (Some(ta), Some(tb)) = (arc_param_in_span(a, t_hit), arc_param_in_span(b, theta_b)) {
                // Dedup near-identical roots from adjacent intervals.
                if !out.iter().any(|h| (h.t_a - ta).abs() < 1e-7) {
                    out.push(CurveHit { t_a: ta, t_b: tb, pos });
                }
            }
        }
        prev_t = t;
        prev_g = gv;
    }
    out
}

/// Closed-form circle-circle intersection with span filtering on both
/// arcs. Tangency (single touch point) is included.
fn intersect_circle_circle(a: &Arc, b: &Arc) -> Vec<CurveHit> {
    let c1 = a.center.value;
    let c2 = b.center.value;
    let r1 = a.radius.value;
    let r2 = b.radius.value;
    let dx = c2.x - c1.x;
    let dy = c2.y - c1.y;
    let d = (dx * dx + dy * dy).sqrt();
    if d < 1e-12 { return Vec::new(); } // concentric: none or infinite
    let tol = 1e-9 * (r1 + r2 + d);
    if d > r1 + r2 + tol || d < (r1 - r2).abs() - tol { return Vec::new(); }
    let along = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
    let h2 = r1 * r1 - along * along;
    let h = if h2 > 0.0 { h2.sqrt() } else { 0.0 };
    let bx = c1.x + along * dx / d;
    let by = c1.y + along * dy / d;
    let candidates: Vec<vect2d> = if h < tol {
        vec![vect2d::new(bx, by)]
    } else {
        vec![
            vect2d::new(bx - h * dy / d, by + h * dx / d),
            vect2d::new(bx + h * dy / d, by - h * dx / d),
        ]
    };
    let mut out = Vec::new();
    for pos in candidates {
        let ta = (pos.y - c1.y).atan2(pos.x - c1.x);
        let tb = (pos.y - c2.y).atan2(pos.x - c2.x);
        if let (Some(ta), Some(tb)) = (arc_param_in_span(a, ta), arc_param_in_span(b, tb)) {
            out.push(CurveHit { t_a: ta, t_b: tb, pos });
        }
    }
    out
}

/// In-span parametric angle of the point on the arc nearest to `p`.
/// For partial arcs a nearest point outside the span clamps to the
/// nearer endpoint's angle.
pub fn nearest_arc_param(arc: &Arc, p: vect2d) -> f64 {
    if !arc.is_ellipse {
        let theta = (p.y - arc.center.value.y).atan2(p.x - arc.center.value.x);
        if let Some(t) = arc_param_in_span(arc, theta) { return t; }
        // Outside span: nearer endpoint.
        let sp = arc.start_pos();
        let ep = arc.end_pos();
        let ds = (p.x - sp.x).powi(2) + (p.y - sp.y).powi(2);
        let de = (p.x - ep.x).powi(2) + (p.y - ep.y).powi(2);
        return if ds <= de { arc.start_angle.value } else { arc.end_angle.value };
    }
    // Ellipse: sample the span, refine by local search.
    let sa = arc.start_angle.value;
    let ea = if arc.closed { sa + std::f64::consts::TAU } else { arc.end_angle.value };
    let n = 128;
    let mut best_t = sa;
    let mut best_d = f64::MAX;
    for i in 0..=n {
        let t = sa + (ea - sa) * (i as f64 / n as f64);
        let q = arc.point_at(t);
        let d = (p.x - q.x).powi(2) + (p.y - q.y).powi(2);
        if d < best_d { best_d = d; best_t = t; }
    }
    // Golden-section-style shrink around the best sample.
    let mut lo = best_t - (ea - sa).abs() / n as f64;
    let mut hi = best_t + (ea - sa).abs() / n as f64;
    for _ in 0..40 {
        let m1 = lo + (hi - lo) / 3.0;
        let m2 = hi - (hi - lo) / 3.0;
        let d1 = { let q = arc.point_at(m1); (p.x - q.x).powi(2) + (p.y - q.y).powi(2) };
        let d2 = { let q = arc.point_at(m2); (p.x - q.x).powi(2) + (p.y - q.y).powi(2) };
        if d1 < d2 { hi = m2; } else { lo = m1; }
    }
    let t = 0.5 * (lo + hi);
    let (t_min, t_max) = (sa.min(ea), sa.max(ea));
    t.clamp(t_min, t_max)
}

#[cfg(test)]
mod intersect_tests {
    use super::*;
    use arael_sketch_solver::Sketch;
    use std::f64::consts::{PI, TAU, FRAC_PI_2};

    fn near(a: f64, b: f64, tol: f64) -> bool { (a - b).abs() < tol }
    fn near_v(a: vect2d, b: vect2d, tol: f64) -> bool {
        near(a.x, b.x, tol) && near(a.y, b.y, tol)
    }

    #[test]
    fn test_segments_cross() {
        let h = intersect_segments(
            vect2d::new(0.0, 0.0), vect2d::new(4.0, 0.0),
            vect2d::new(1.0, -1.0), vect2d::new(1.0, 3.0),
        ).unwrap();
        assert!(near(h.t_a, 0.25, 1e-12));
        assert!(near(h.t_b, 0.25, 1e-12));
        assert!(near_v(h.pos, vect2d::new(1.0, 0.0), 1e-12));
    }

    #[test]
    fn test_segments_miss_and_parallel() {
        // Crossing point beyond segment b.
        assert!(intersect_segments(
            vect2d::new(0.0, 0.0), vect2d::new(4.0, 0.0),
            vect2d::new(1.0, 1.0), vect2d::new(1.0, 3.0),
        ).is_none());
        // Parallel.
        assert!(intersect_segments(
            vect2d::new(0.0, 0.0), vect2d::new(4.0, 0.0),
            vect2d::new(0.0, 1.0), vect2d::new(4.0, 1.0),
        ).is_none());
        // Collinear overlap: no transversal crossing.
        assert!(intersect_segments(
            vect2d::new(0.0, 0.0), vect2d::new(4.0, 0.0),
            vect2d::new(1.0, 0.0), vect2d::new(5.0, 0.0),
        ).is_none());
    }

    #[test]
    fn test_segment_circle_two_hits() {
        let mut s = Sketch::new();
        let a = s.add_arc(vect2d::new(0.0, 0.0), 1.0, 0.0, TAU, true);
        let arc = &s.arcs[a];
        let hits = intersect_segment_arc(vect2d::new(-2.0, 0.0), vect2d::new(2.0, 0.0), arc);
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|h| near_v(h.pos, vect2d::new(-1.0, 0.0), 1e-9)));
        assert!(hits.iter().any(|h| near_v(h.pos, vect2d::new(1.0, 0.0), 1e-9)));
        // Params round-trip through point_at.
        for h in &hits {
            assert!(near_v(arc.point_at(h.t_b), h.pos, 1e-9));
        }
    }

    #[test]
    fn test_segment_circle_tangent_single_hit() {
        let mut s = Sketch::new();
        let a = s.add_arc(vect2d::new(0.0, 0.0), 1.0, 0.0, TAU, true);
        let hits = intersect_segment_arc(
            vect2d::new(-2.0, 1.0), vect2d::new(2.0, 1.0), &s.arcs[a]);
        assert_eq!(hits.len(), 1);
        assert!(near_v(hits[0].pos, vect2d::new(0.0, 1.0), 1e-6));
    }

    #[test]
    fn test_segment_partial_arc_span_filter() {
        let mut s = Sketch::new();
        // Upper half circle, ccw from 0 to PI.
        let a = s.add_arc(vect2d::new(0.0, 0.0), 1.0, 0.0, PI, false);
        let hits = intersect_segment_arc(
            vect2d::new(-2.0, 0.5), vect2d::new(2.0, 0.5), &s.arcs[a]);
        assert_eq!(hits.len(), 2, "chord at y=0.5 crosses the upper half twice");
        // Chord at y=-0.5 misses the upper half entirely.
        let hits = intersect_segment_arc(
            vect2d::new(-2.0, -0.5), vect2d::new(2.0, -0.5), &s.arcs[a]);
        assert!(hits.is_empty());
    }

    #[test]
    fn test_segment_cw_arc_span_filter() {
        let mut s = Sketch::new();
        // Lower half circle drawn clockwise from 0 to -PI.
        let a = s.add_arc_with_dir(vect2d::new(0.0, 0.0), 1.0, 0.0, PI, false, false);
        let arc = &s.arcs[a];
        assert!(arc.end_angle.value < arc.start_angle.value, "cw convention");
        let hits = intersect_segment_arc(
            vect2d::new(-2.0, -0.5), vect2d::new(2.0, -0.5), arc);
        assert_eq!(hits.len(), 2);
        for h in &hits {
            assert!(near_v(arc.point_at(h.t_b), h.pos, 1e-9));
        }
    }

    #[test]
    fn test_segment_rotated_ellipse() {
        let mut s = Sketch::new();
        let a = s.add_ellipse(vect2d::new(1.0, 2.0), 3.0, 1.0, FRAC_PI_2, true);
        let arc = &s.arcs[a];
        // Major axis is vertical after the 90 deg rotation: the curve
        // spans y in [-1, 5] at x = 1. A horizontal chord at y = 2
        // crosses at x = 1 +- 1 (the semi-minor).
        let hits = intersect_segment_arc(
            vect2d::new(-5.0, 2.0), vect2d::new(5.0, 2.0), arc);
        assert_eq!(hits.len(), 2);
        assert!(hits.iter().any(|h| near_v(h.pos, vect2d::new(0.0, 2.0), 1e-9)));
        assert!(hits.iter().any(|h| near_v(h.pos, vect2d::new(2.0, 2.0), 1e-9)));
        for h in &hits {
            assert!(near_v(arc.point_at(h.t_b), h.pos, 1e-9));
        }
    }

    #[test]
    fn test_circle_circle_two_and_none() {
        let mut s = Sketch::new();
        let a = s.add_arc(vect2d::new(0.0, 0.0), 1.0, 0.0, TAU, true);
        let b = s.add_arc(vect2d::new(1.0, 0.0), 1.0, 0.0, TAU, true);
        let hits = intersect_arcs(&s.arcs[a], &s.arcs[b]);
        assert_eq!(hits.len(), 2);
        for h in &hits {
            assert!(near(h.pos.x, 0.5, 1e-9));
            assert!(near(h.pos.y.abs(), (0.75f64).sqrt(), 1e-9));
            assert!(near_v(s.arcs[a].point_at(h.t_a), h.pos, 1e-9));
            assert!(near_v(s.arcs[b].point_at(h.t_b), h.pos, 1e-9));
        }
        let c = s.add_arc(vect2d::new(5.0, 0.0), 1.0, 0.0, TAU, true);
        assert!(intersect_arcs(&s.arcs[a], &s.arcs[c]).is_empty());
    }

    #[test]
    fn test_circle_circle_tangent() {
        let mut s = Sketch::new();
        let a = s.add_arc(vect2d::new(0.0, 0.0), 1.0, 0.0, TAU, true);
        let b = s.add_arc(vect2d::new(2.0, 0.0), 1.0, 0.0, TAU, true);
        let hits = intersect_arcs(&s.arcs[a], &s.arcs[b]);
        assert_eq!(hits.len(), 1);
        assert!(near_v(hits[0].pos, vect2d::new(1.0, 0.0), 1e-9));
    }

    #[test]
    fn test_ellipse_circle_sampling() {
        let mut s = Sketch::new();
        let e = s.add_ellipse(vect2d::new(0.0, 0.0), 3.0, 1.0, 0.0, true);
        let c = s.add_arc(vect2d::new(3.0, 0.0), 1.0, 0.0, TAU, true);
        let hits = intersect_arcs(&s.arcs[e], &s.arcs[c]);
        assert_eq!(hits.len(), 2, "circle at the ellipse vertex crosses twice");
        for h in &hits {
            assert!(near_v(s.arcs[e].point_at(h.t_a), h.pos, 1e-6));
            assert!(near_v(s.arcs[c].point_at(h.t_b), h.pos, 1e-6));
        }
    }

    #[test]
    fn test_nearest_arc_param() {
        let mut s = Sketch::new();
        let a = s.add_arc(vect2d::new(0.0, 0.0), 2.0, 0.0, PI, false);
        // Point near the top of the arc.
        let t = nearest_arc_param(&s.arcs[a], vect2d::new(0.1, 3.0));
        let q = s.arcs[a].point_at(t);
        assert!(near_v(q, vect2d::new(0.0, 2.0), 0.15));
        // Point below the arc's span clamps to the nearer endpoint.
        let t = nearest_arc_param(&s.arcs[a], vect2d::new(2.0, -1.0));
        assert!(near(t, 0.0, 1e-9), "clamps to start angle, got {}", t);
        // Ellipse path.
        let e = s.add_ellipse(vect2d::new(0.0, 0.0), 3.0, 1.0, 0.0, true);
        let t = nearest_arc_param(&s.arcs[e], vect2d::new(4.0, 0.0));
        let q = s.arcs[e].point_at(t);
        assert!(near_v(q, vect2d::new(3.0, 0.0), 1e-3));
    }
}
