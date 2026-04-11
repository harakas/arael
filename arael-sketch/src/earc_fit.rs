//! Elliptic arc fitting from endpoint+tangent+bulge constraints.
//!
//! Uses the conic pencil through two points with prescribed tangent directions.
//! The family of conics tangent to lines L1 at p1 and L2 at p2 is:
//!   L1*L2 + lambda * M^2 = 0
//! where M is the chord line through p1, p2. Lambda parameterizes the family.
//! Given a bulge value, we solve for lambda analytically.

use arael::vect::vect2d;

/// Fit an elliptic arc through two points with given tangent directions and bulge.
///
/// `bulge` is dimensionless: perpendicular distance from chord to arc midpoint,
/// divided by half the chord length. b=0 is straight, b=1 is semicircle-like.
///
/// Returns (center, rx, ry, rotation, start_angle, end_angle, ccw) or None if degenerate.
pub fn fit_earc_tangent(
    p1: vect2d, t1: vect2d, p2: vect2d, t2: vect2d, bulge: f64,
) -> Option<(vect2d, f64, f64, f64, f64, f64, bool)> {
    let chord_dx = p2.x - p1.x;
    let chord_dy = p2.y - p1.y;
    let chord_len = (chord_dx * chord_dx + chord_dy * chord_dy).sqrt();
    if chord_len < 1e-12 { return None; }
    let half_chord = chord_len / 2.0;
    let chord_mid = vect2d::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);

    // Tangent lines in ax + by + c = 0 form (line through p_i with direction t_i)
    // Normal to tangent direction: (t.y, -t.x) gives (a, b), then c = -(a*px + b*py)
    let l1 = (t1.y, -t1.x, -(t1.y * p1.x - t1.x * p1.y));
    let l2 = (t2.y, -t2.x, -(t2.y * p2.x - t2.x * p2.y));

    // Chord line through p1, p2: ax + by + c = 0
    let m = (chord_dy, -chord_dx, -(chord_dy * p1.x - chord_dx * p1.y));
    // Normalize m so that m(point) gives signed distance * chord_len
    let m_len = (m.0 * m.0 + m.1 * m.1).sqrt();
    if m_len < 1e-12 { return None; }
    let m = (m.0 / m_len, m.1 / m_len, m.2 / m_len);

    // Bulge direction: perpendicular to chord, determine sign from tangent config
    let chord_ux = chord_dx / chord_len;
    let chord_uy = chord_dy / chord_len;
    // The bulge direction is perpendicular to chord. We need to figure out which side.
    // Use the tangent-normal intersection to determine the arc side.
    let n1 = vect2d::new(-t1.y, t1.x);
    let n2 = vect2d::new(-t2.y, t2.x);
    let denom = n1.x * n2.y - n1.y * n2.x;
    let bulge_sign = if denom.abs() > 1e-12 {
        let s = (chord_dx * n2.y - chord_dy * n2.x) / denom;
        let center_guess = vect2d::new(p1.x + s * n1.x, p1.y + s * n1.y);
        // Which side of the chord is the center?
        let center_side = m.0 * center_guess.x + m.1 * center_guess.y + m.2;
        if center_side >= 0.0 { -1.0 } else { 1.0 } // arc bulges opposite to center
    } else {
        1.0
    };

    let perp_x = -chord_uy;
    let perp_y = chord_ux;

    let l1v = [l1.0, l1.1, l1.2];
    let l2v = [l2.0, l2.1, l2.2];
    let mv = [m.0, m.1, m.2];

    // Compute target midpoint directly from bulge (raw, dimensionless)
    let target_mid = vect2d::new(
        chord_mid.x + perp_x * bulge_sign * bulge * half_chord,
        chord_mid.y + perp_y * bulge_sign * bulge * half_chord,
    );

    // Find lambda for this target
    let l1_val = l1.0 * target_mid.x + l1.1 * target_mid.y + l1.2;
    let l2_val = l2.0 * target_mid.x + l2.1 * target_mid.y + l2.2;
    let m_val = m.0 * target_mid.x + m.1 * target_mid.y + m.2;

    if m_val.abs() < 1e-12 { return None; }

    let lambda = -(l1_val * l2_val) / (m_val * m_val);

    // Build conic matrix
    let mut q = [[0.0f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            q[i][j] = (l1v[i] * l2v[j] + l2v[i] * l1v[j]) / 2.0 + lambda * mv[i] * mv[j];
        }
    }

    // Extract ellipse parameters from conic matrix
    // The 2x2 sub-matrix [[A, B/2], [B/2, C]] gives orientation and axes
    let a = q[0][0];
    let b = q[0][1]; // = q[1][0], already B/2
    let c = q[1][1];
    let d = q[0][2]; // D/2
    let e = q[1][2]; // E/2
    let f = q[2][2];

    // Check it's an ellipse: det of upper-left 2x2 > 0
    let det2 = a * c - b * b;
    if det2 <= 1e-12 {
        return None; // not an ellipse (parabola or hyperbola)
    }

    // Center: solve [A, B/2; B/2, C] * [cx; cy] = -[D/2; E/2]
    let cx = (b * e - c * d) / det2;
    let cy = (b * d - a * e) / det2;
    let center = vect2d::new(cx, cy);

    // Eigenvalues of [[A, B], [B, C]] give 1/rx^2 and 1/ry^2 (up to scale)
    let trace = a + c;
    let disc = ((a - c) * (a - c) + 4.0 * b * b).sqrt();
    let ev1 = (trace + disc) / 2.0;
    let ev2 = (trace - disc) / 2.0;

    if ev1.abs() < 1e-12 || ev2.abs() < 1e-12 { return None; }

    // Scale factor from the constant term
    let f_prime = f + d * cx + e * cy;  // = Q evaluated shifted to center
    // For canonical form: ev1 * x'^2 + ev2 * y'^2 + f' = 0
    // => x'^2 / (-f'/ev1) + y'^2 / (-f'/ev2) = 1
    let rx_sq = -f_prime / ev1;
    let ry_sq = -f_prime / ev2;

    if rx_sq <= 0.0 || ry_sq <= 0.0 { return None; }

    let rx = rx_sq.sqrt();
    let ry = ry_sq.sqrt();

    // Rotation: eigenvector of [[A, B], [B, C]] for ev1
    let rot = if b.abs() > 1e-12 {
        (ev1 - a).atan2(b)
    } else if a <= c {
        0.0
    } else {
        std::f64::consts::FRAC_PI_2
    };

    // Parametric angles: find t such that ellipse_point(t) = p1, p2
    let cr = rot.cos();
    let sr = rot.sin();
    // Transform p1, p2 to ellipse-local coordinates (unrotated, centered)
    let lx1 = (p1.x - cx) * cr + (p1.y - cy) * sr;
    let ly1 = -(p1.x - cx) * sr + (p1.y - cy) * cr;
    let lx2 = (p2.x - cx) * cr + (p2.y - cy) * sr;
    let ly2 = -(p2.x - cx) * sr + (p2.y - cy) * cr;
    // (lx, ly) = (rx*cos(t), ry*sin(t)) => t = atan2(ly/ry, lx/rx)
    let sa = (ly1 / ry).atan2(lx1 / rx);
    let ea = (ly2 / ry).atan2(lx2 / rx);

    // Determine CCW from tangent direction at start
    let arc_tan_x = -rx * sa.sin() * cr - ry * sa.cos() * sr;
    let arc_tan_y = -rx * sa.sin() * sr + ry * sa.cos() * cr;
    let dot = arc_tan_x * t1.x + arc_tan_y * t1.y;
    let ccw = dot > 0.0;

    Some((center, rx, ry, rot, sa, ea, ccw))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arc_point(center: vect2d, rx: f64, ry: f64, rot: f64, t: f64) -> vect2d {
        let cr = rot.cos(); let sr = rot.sin();
        let ct = t.cos(); let st = t.sin();
        vect2d::new(
            center.x + rx * ct * cr - ry * st * sr,
            center.y + rx * ct * sr + ry * st * cr,
        )
    }

    fn arc_tangent(rx: f64, ry: f64, rot: f64, t: f64) -> vect2d {
        let cr = rot.cos(); let sr = rot.sin();
        let ct = t.cos(); let st = t.sin();
        vect2d::new(
            -rx * st * cr - ry * ct * sr,
            -rx * st * sr + ry * ct * cr,
        )
    }

    #[test]
    fn test_fit_semicircle_bulge1() {
        let result = fit_earc_tangent(
            vect2d::new(0.0, 0.0), vect2d::new(0.0, 1.0),
            vect2d::new(10.0, 0.0), vect2d::new(0.0, -1.0),
            1.0,
        );
        let (center, rx, ry, rot, sa, ea, _ccw) = result.expect("should converge");
        eprintln!("b=1: center=({:.4},{:.4}) rx={:.4} ry={:.4} rot={:.4}", center.x, center.y, rx, ry, rot);
        let sp = arc_point(center, rx, ry, rot, sa);
        let ep = arc_point(center, rx, ry, rot, ea);
        assert!((sp.x - 0.0).abs() < 0.01, "start x: {:.4}", sp.x);
        assert!((sp.y - 0.0).abs() < 0.01, "start y: {:.4}", sp.y);
        assert!((ep.x - 10.0).abs() < 0.01, "end x: {:.4}", ep.x);
        assert!((ep.y - 0.0).abs() < 0.01, "end y: {:.4}", ep.y);
        assert!((rx - ry).abs() < 0.01, "should be circular: rx={:.4} ry={:.4}", rx, ry);
    }

    #[test]
    fn test_fit_small_bulge() {
        let result = fit_earc_tangent(
            vect2d::new(0.0, 0.0), vect2d::new(1.0, 0.0),
            vect2d::new(5.0, 5.0), vect2d::new(0.0, 1.0),
            0.3,
        );
        let (center, rx, ry, rot, sa, ea, _ccw) = result.expect("should converge");
        eprintln!("b=0.3: center=({:.4},{:.4}) rx={:.4} ry={:.4} rot={:.4}", center.x, center.y, rx, ry, rot);
        let sp = arc_point(center, rx, ry, rot, sa);
        let ep = arc_point(center, rx, ry, rot, ea);
        assert!((sp.x - 0.0).abs() < 0.01, "start x: {:.4}", sp.x);
        assert!((sp.y - 0.0).abs() < 0.01, "start y: {:.4}", sp.y);
        assert!((ep.x - 5.0).abs() < 0.01, "end x: {:.4}", ep.x);
        assert!((ep.y - 5.0).abs() < 0.01, "end y: {:.4}", ep.y);
    }

    #[test]
    fn test_fit_tangent_directions() {
        let result = fit_earc_tangent(
            vect2d::new(0.0, 0.0), vect2d::new(1.0, 0.0),
            vect2d::new(5.0, 5.0), vect2d::new(0.0, 1.0),
            0.5,
        );
        let (_center, rx, ry, rot, sa, ea, _ccw) = result.expect("should converge");
        let ts = arc_tangent(rx, ry, rot, sa);
        let ts_len = (ts.x * ts.x + ts.y * ts.y).sqrt();
        let cross_s = (ts.x * 0.0 - ts.y * 1.0).abs() / ts_len;
        assert!(cross_s < 0.01, "start tangent should be horizontal, cross={:.4}", cross_s);
        let te = arc_tangent(rx, ry, rot, ea);
        let te_len = (te.x * te.x + te.y * te.y).sqrt();
        let cross_e = (te.x * 1.0 - te.y * 0.0).abs() / te_len;
        assert!(cross_e < 0.01, "end tangent should be vertical, cross={:.4}", cross_e);
    }

    #[test]
    fn test_fit_exact_endpoints() {
        // The key test: endpoints must be exact (analytic, not approximate)
        let result = fit_earc_tangent(
            vect2d::new(1.0, 0.0), vect2d::new(1.0, 0.0),
            vect2d::new(3.0, 1.0), vect2d::new(0.0, 1.0),
            0.2,
        );
        let (center, rx, ry, rot, sa, ea, _ccw) = result.expect("should converge");
        let sp = arc_point(center, rx, ry, rot, sa);
        let ep = arc_point(center, rx, ry, rot, ea);
        assert!((sp.x - 1.0).abs() < 0.001, "start x: {:.6}", sp.x);
        assert!((sp.y - 0.0).abs() < 0.001, "start y: {:.6}", sp.y);
        assert!((ep.x - 3.0).abs() < 0.001, "end x: {:.6}", ep.x);
        assert!((ep.y - 1.0).abs() < 0.001, "end y: {:.6}", ep.y);
    }

    #[test]
    fn test_fit_degenerate_zero_chord() {
        let result = fit_earc_tangent(
            vect2d::new(0.0, 0.0), vect2d::new(1.0, 0.0),
            vect2d::new(0.0, 0.0), vect2d::new(0.0, 1.0),
            0.5,
        );
        assert!(result.is_none(), "zero chord should be degenerate");
    }
}
