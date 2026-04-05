// Canvas rendering, grid, dimension drawing, styled lines, constraint markers.

use eframe::egui;
use arael::refs::Ref;
use arael::utils::rad2rad;
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::tools::*;
use crate::geometry::*;
use crate::EditorApp;

/// Compute (start_angle, span) for an arc respecting ccw flag.
/// CCW arcs have positive span, CW arcs have negative span.
fn arc_span(a: &Arc) -> (f64, f64) {
    let sa = a.start_angle.value;
    let ea = a.end_angle.value;
    let norm = |v: f64| -> f64 { let r = v % std::f64::consts::TAU; if r < 0.0 { r + std::f64::consts::TAU } else { r } };
    if a.ccw {
        (sa, norm(ea - sa))
    } else {
        (sa, -norm(sa - ea))
    }
}

impl EditorApp {
    // Draw rotated text using egui's native TextShape.angle support.
    // `center` is where text should be centered, `dir_x/dir_y` is the unit direction along the text.
    // Returns the bounding segment (text_start, text_end) in screen coords.
    pub fn draw_rotated_text(&self, painter: &egui::Painter, center: egui::Pos2,
                          dir_x: f32, dir_y: f32, text: &str,
                          font: egui::FontId, color: egui::Color32) -> (egui::Pos2, egui::Pos2) {
        // Ensure text reads left-to-right: flip direction if pointing left
        let (dx, dy) = if dir_x < 0.0 { (-dir_x, -dir_y) } else { (dir_x, dir_y) };
        let angle = dy.atan2(dx); // rotation angle in radians

        // Layout the text to get its size
        let galley = painter.layout_no_wrap(text.to_string(), font, color);
        let text_width = galley.rect.width();
        let text_height = galley.rect.height();

        // Position: pivot is top-left of the galley, rotated around that point.
        // We want the text centered at `center`, offset perpendicular to read above the line.
        // Compute the top-left position before rotation such that after rotation the text is centered.
        let half_w = text_width / 2.0;
        let half_h = text_height / 2.0;
        // Center of unrotated text at (pos.x + half_w, pos.y + half_h).
        // After rotating by `angle` around pos, center moves to:
        //   pos + rotate(half_w, half_h, angle)
        // We want that to equal `center - normal * offset` (slightly above the line)
        let nx = -dy;
        let ny = dx;
        let target_x = center.x - nx * (half_h + 2.0);
        let target_y = center.y - ny * (half_h + 2.0);
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let rotated_cx = half_w * cos_a - half_h * sin_a;
        let rotated_cy = half_w * sin_a + half_h * cos_a;
        let pos = egui::Pos2::new(target_x - rotated_cx, target_y - rotated_cy);

        let shape = egui::epaint::TextShape::new(pos, galley, color)
            .with_angle(angle);
        painter.add(shape);

        // Return text extent segment for hit testing
        let ts = egui::Pos2::new(center.x - dx * half_w, center.y - dy * half_w);
        let te = egui::Pos2::new(center.x + dx * half_w, center.y + dy * half_w);
        (ts, te)
    }

    // Draw a dimension annotation. Returns (text_start, text_end) screen segment for hit testing.
    pub fn draw_dimension(&self, painter: &egui::Painter, kind: &DimensionKind, value: f64,
                       offset: vect2d, text_along: f64, color: egui::Color32, is_radius: bool,
                       is_expr: bool, is_derived: bool) -> (egui::Pos2, egui::Pos2) {
        if is_radius {
            if let DimensionKind::ArcRadius(r) = kind {
                let a = &self.sketch.arcs[*r];
                let angle = offset.x;
                let edge = vect2d::new(
                    a.center.value.x + a.radius.value * angle.cos(),
                    a.center.value.y + a.radius.value * angle.sin(),
                );
                let arrow_len = a.radius.value * 0.6;
                let inner = vect2d::new(
                    edge.x - arrow_len * angle.cos(),
                    edge.y - arrow_len * angle.sin(),
                );
                let se = self.to_screen(edge);
                let si = self.to_screen(inner);
                let stroke = egui::Stroke::new(1.0, color);
                painter.line_segment([se, si], stroke);
                // Arrowhead at edge
                let adx = si.x - se.x;
                let ady = si.y - se.y;
                let alen = (adx * adx + ady * ady).sqrt().max(1.0);
                let ax = adx / alen;
                let ay = ady / alen;
                let asz = 6.0;
                painter.line_segment([se, egui::Pos2::new(se.x + ax * asz + ay * asz * 0.4, se.y + ay * asz - ax * asz * 0.4)], stroke);
                painter.line_segment([se, egui::Pos2::new(se.x + ax * asz - ay * asz * 0.4, se.y + ay * asz + ax * asz * 0.4)], stroke);
                // Text along arrow
                let mid = egui::Pos2::new((se.x + si.x) / 2.0, (se.y + si.y) / 2.0);
                let text = if is_derived { format!("(R{:.2})", value) }
                    else if is_expr { format!("fx: R{:.2}", value) }
                    else { format!("R{:.2}", value) };
                return self.draw_rotated_text(painter, mid, ax, ay, &text,
                    egui::FontId::proportional(12.0), color);
            }
        }

        // Angle dimension: draw arc between two lines
        if let DimensionKind::Angle(a_ref, b_ref, supplement) = kind {
            return self.draw_angle_dimension(painter, *a_ref, *b_ref, *supplement,
                value, offset, text_along, color, is_expr, is_derived);
        }

        // Sweep dimension: arc annotation from start_angle to end_angle
        if let DimensionKind::ArcSweep(r) = kind {
            return self.draw_sweep_dimension(painter, *r, value, offset, text_along,
                color, is_expr, is_derived);
        }

        let (p1_sketch, p2_sketch) = self.dim_endpoints(kind);
        let dx = p2_sketch.x - p1_sketch.x;
        let dy = p2_sketch.y - p1_sketch.y;
        let len = (dx * dx + dy * dy).sqrt().max(1e-12);
        let nx = -dy / len;
        let ny = dx / len;
        let off = offset.y;

        let q1 = vect2d::new(p1_sketch.x + nx * off, p1_sketch.y + ny * off);
        let q2 = vect2d::new(p2_sketch.x + nx * off, p2_sketch.y + ny * off);

        let sq1 = self.to_screen(q1);
        let sq2 = self.to_screen(q2);
        let sp1 = self.to_screen(p1_sketch);
        let sp2 = self.to_screen(p2_sketch);

        let stroke = egui::Stroke::new(1.0, color);

        // Extension lines
        // For point-line distance, the line-side extension goes from the nearest
        // line endpoint to the dimension arrow position (not from the foot projection)
        if let DimensionKind::PointLineDistance(_, line_ref) = kind {
            let l = &self.sketch.lines[*line_ref];
            // Find which endpoint is closer to the foot (p2_sketch)
            let d1 = ((l.p1.value.x - p2_sketch.x).powi(2) + (l.p1.value.y - p2_sketch.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - p2_sketch.x).powi(2) + (l.p2.value.y - p2_sketch.y).powi(2)).sqrt();
            let nearest = if d1 < d2 { self.to_screen(l.p1.value) } else { self.to_screen(l.p2.value) };
            painter.line_segment([sp1, sq1], egui::Stroke::new(0.5, color));
            painter.line_segment([nearest, sq2], egui::Stroke::new(0.5, color));
        } else {
            painter.line_segment([sp1, sq1], egui::Stroke::new(0.5, color));
            painter.line_segment([sp2, sq2], egui::Stroke::new(0.5, color));
        }

        // Arrowheads and dimension line
        let adx = sq2.x - sq1.x;
        let ady = sq2.y - sq1.y;
        let alen = (adx * adx + ady * ady).sqrt().max(1.0);
        let ax = adx / alen;
        let ay = ady / alen;
        let asz = 6.0;

        // Text position along the line
        let text = if is_derived { format!("({:.2})", value) }
            else if is_expr { format!("fx: {:.2}", value) }
            else { format!("{:.2}", value) };
        let char_width = 12.0 * 0.6;
        let text_half_w = text.len() as f32 * char_width / 2.0;
        let text_center = egui::Pos2::new(
            (sq1.x + sq2.x) / 2.0 + ax * (text_along as f32) * alen,
            (sq1.y + sq2.y) / 2.0 + ay * (text_along as f32) * alen,
        );

        // Dimension line: extend beyond endpoints if text is outside
        let line_start;
        let line_end;
        let text_left = text_center.x - ax * text_half_w;
        let text_left_y = text_center.y - ay * text_half_w;
        let text_right = text_center.x + ax * text_half_w;
        let text_right_y = text_center.y + ay * text_half_w;

        // Project text edges onto the line to find extension
        let proj_left = (text_left - sq1.x) * ax + (text_left_y - sq1.y) * ay;
        let proj_right = (text_right - sq1.x) * ax + (text_right_y - sq1.y) * ay;
        let margin = 4.0;

        let min_proj = proj_left.min(proj_right) - margin;
        let max_proj = proj_right.max(proj_left) + margin;

        line_start = if min_proj < 0.0 {
            egui::Pos2::new(sq1.x + ax * min_proj, sq1.y + ay * min_proj)
        } else { sq1 };
        line_end = if max_proj > alen {
            egui::Pos2::new(sq1.x + ax * max_proj, sq1.y + ay * max_proj)
        } else { sq2 };

        // Draw dimension line (possibly extended)
        painter.line_segment([line_start, line_end], stroke);

        // Arrowheads at original endpoints (sq1, sq2)
        painter.line_segment([sq1, egui::Pos2::new(sq1.x + ax * asz + ay * asz * 0.4, sq1.y + ay * asz - ax * asz * 0.4)], stroke);
        painter.line_segment([sq1, egui::Pos2::new(sq1.x + ax * asz - ay * asz * 0.4, sq1.y + ay * asz + ax * asz * 0.4)], stroke);
        painter.line_segment([sq2, egui::Pos2::new(sq2.x - ax * asz + ay * asz * 0.4, sq2.y - ay * asz - ax * asz * 0.4)], stroke);
        painter.line_segment([sq2, egui::Pos2::new(sq2.x - ax * asz - ay * asz * 0.4, sq2.y - ay * asz + ax * asz * 0.4)], stroke);

        // Value text rotated along the dimension line
        self.draw_rotated_text(painter, text_center, ax, ay, &text,
            egui::FontId::proportional(12.0), color)
    }

    /// Compute the angle sector start and sweep for an angle dimension.
    /// offset.x stores the sector midpoint angle chosen during placement.
    /// Finds the actual sector around offset.x using the 4 half-line boundaries.
    /// Compute sector start and sweep for an angle dimension.
    /// Uses the supplement flag to select the correct pair of opposing sectors,
    /// then picks the one closest to offset.x. This is stable under line rotation.
    pub fn angle_dim_sector(&self, a_ref: Ref<Line>, b_ref: Ref<Line>, supplement: bool,
                        offset: vect2d) -> (vect2d, f64, f64) {
        let la = &self.sketch.lines[a_ref];
        let lb = &self.sketch.lines[b_ref];
        let ix = crate::geometry::line_line_intersection(
            la.p1.value, la.p2.value, lb.p1.value, lb.p2.value);

        let da = vect2d::new(la.p2.value.x - la.p1.value.x, la.p2.value.y - la.p1.value.y);
        let db = vect2d::new(lb.p2.value.x - lb.p1.value.x, lb.p2.value.y - lb.p1.value.y);
        let ang_a = da.y.atan2(da.x);
        let ang_b = db.y.atan2(db.x);
        let pi = std::f64::consts::PI;
        // Compute sweep for the supplement pair
        let ang_a_eff = if supplement { ang_a + pi } else { ang_a };
        let sweep = rad2rad(ang_b - ang_a_eff);

        // Two opposing sectors: start at ang_a_eff and ang_a_eff+pi
        let start1 = rad2rad(ang_a_eff);
        let start2 = rad2rad(ang_a_eff + pi);
        let mid1 = rad2rad(start1 + sweep * 0.5);
        let mid2 = rad2rad(start2 + sweep * 0.5);

        // Pick the sector whose midpoint is closest to offset.x
        let d1 = rad2rad(offset.x - mid1).abs();
        let d2 = rad2rad(offset.x - mid2).abs();
        let start = if d1 <= d2 { start1 } else { start2 };

        (ix, start, sweep)
    }

    /// Determine which of the 4 angle sectors the mouse is in.
    /// Returns (sector_midpoint_angle, supplement_flag).
    /// The 4 half-lines (ang_a, ang_a+pi, ang_b, ang_b+pi) divide the plane
    /// into 4 sectors. We sort them, find which sector the mouse is in,
    /// and determine the supplement flag from the bounding half-lines.
    pub fn angle_dim_sector_from_mouse(&self, a_ref: Ref<Line>, b_ref: Ref<Line>,
                                        mouse_angle: f64) -> (f64, bool) {
        let la = &self.sketch.lines[a_ref];
        let lb = &self.sketch.lines[b_ref];
        let da = vect2d::new(la.p2.value.x - la.p1.value.x, la.p2.value.y - la.p1.value.y);
        let db = vect2d::new(lb.p2.value.x - lb.p1.value.x, lb.p2.value.y - lb.p1.value.y);
        let ang_a = da.y.atan2(da.x);
        let ang_b = db.y.atan2(db.x);
        let pi = std::f64::consts::PI;
        // 4 half-line directions, tagged: (angle, is_line_a)
        let mut halves = [
            (rad2rad(ang_a), true),
            (rad2rad(ang_a + pi), true),
            (rad2rad(ang_b), false),
            (rad2rad(ang_b + pi), false),
        ];
        halves.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Find which sector the mouse is in
        let m = rad2rad(mouse_angle);
        let mut sector_idx = 3; // default: between halves[3] and halves[0] (wrapping)
        for i in 0..4 {
            let next = (i + 1) % 4;
            let mut span = halves[next].0 - halves[i].0;
            if span <= 0.0 { span += 2.0 * pi; }
            let mut delta = m - halves[i].0;
            if delta < 0.0 { delta += 2.0 * pi; }
            if delta < span {
                sector_idx = i;
                break;
            }
        }

        // Sector is between halves[sector_idx] and halves[(sector_idx+1)%4]
        let i0 = sector_idx;
        let i1 = (sector_idx + 1) % 4;
        let a0 = halves[i0].0;
        let a1 = halves[i1].0;
        let mut span = a1 - a0;
        if span <= 0.0 { span += 2.0 * pi; }
        let mid = rad2rad(a0 + span * 0.5);

        // Determine supplement: compute the direct angle between lines
        let cross = da.x * db.y - da.y * db.x;
        let dot = da.x * db.x + da.y * db.y;
        let direct_angle = cross.atan2(dot).abs();
        // If this sector's span is closer to pi-direct_angle, it's the supplement
        let supplement = (span - (pi - direct_angle)).abs() < (span - direct_angle).abs();

        (mid, supplement)
    }

    /// Pick between 2 opposing sectors for a given supplement value.
    /// Used when dragging an existing dimension (no sector type change allowed).
    pub fn angle_dim_opposing_sector(&self, a_ref: Ref<Line>, b_ref: Ref<Line>,
                                      supplement: bool, mouse_angle: f64) -> f64 {
        let la = &self.sketch.lines[a_ref];
        let lb = &self.sketch.lines[b_ref];
        let da = vect2d::new(la.p2.value.x - la.p1.value.x, la.p2.value.y - la.p1.value.y);
        let db = vect2d::new(lb.p2.value.x - lb.p1.value.x, lb.p2.value.y - lb.p1.value.y);
        let ang_a = da.y.atan2(da.x);
        let ang_b = db.y.atan2(db.x);
        let pi = std::f64::consts::PI;
        let normalize = |mut a: f64| -> f64 {
            while a > pi { a -= 2.0 * pi; }
            while a <= -pi { a += 2.0 * pi; }
            a
        };

        let ang_a_eff = if supplement { ang_a + pi } else { ang_a };
        let sweep = normalize(ang_b - ang_a_eff);

        // Two opposing sectors: start at ang_a_eff and ang_a_eff+pi
        let mid1 = normalize(ang_a_eff + sweep * 0.5);
        let mid2 = normalize(ang_a_eff + pi + sweep * 0.5);
        let d1 = normalize(mouse_angle - mid1).abs();
        let d2 = normalize(mouse_angle - mid2).abs();
        if d1 <= d2 { mid1 } else { mid2 }
    }

    /// Draw an angle dimension arc with arrowheads, extension lines, and text.
    /// text_along: 0 = arc midpoint, +/- shifts along arc. Values outside
    /// [-0.5, 0.5] extend beyond the sector with a thin extension arc.
    fn draw_angle_dimension(&self, painter: &egui::Painter, a_ref: Ref<Line>,
                            b_ref: Ref<Line>, supplement: bool, value: f64,
                            offset: vect2d, text_along: f64,
                            color: egui::Color32, is_expr: bool, is_derived: bool) -> (egui::Pos2, egui::Pos2) {
        let (ix, start_angle, sweep) = self.angle_dim_sector(a_ref, b_ref, supplement, offset);
        let radius = offset.y.max(0.3);
        let stroke = egui::Stroke::new(1.0, color);
        let ext_stroke = egui::Stroke::new(0.5, color);
        let six = self.to_screen(ix);

        // Draw thin extension lines from intersection to arc endpoints
        let arc_start_pt = vect2d::new(ix.x + radius * start_angle.cos(), ix.y + radius * start_angle.sin());
        let arc_end_pt = vect2d::new(ix.x + radius * (start_angle + sweep).cos(),
                                     ix.y + radius * (start_angle + sweep).sin());
        painter.line_segment([six, self.to_screen(arc_start_pt)], ext_stroke);
        painter.line_segment([six, self.to_screen(arc_end_pt)], ext_stroke);

        // Draw main arc as polyline
        let draw_arc = |a_start: f64, a_sweep: f64, s: egui::Stroke| {
            let n = ((a_sweep.abs() * 20.0).ceil() as usize).max(8);
            let pts: Vec<egui::Pos2> = (0..=n).map(|i| {
                let t = i as f64 / n as f64;
                let ang = a_start + a_sweep * t;
                self.to_screen(vect2d::new(ix.x + radius * ang.cos(), ix.y + radius * ang.sin()))
            }).collect();
            for w in pts.windows(2) { painter.line_segment([w[0], w[1]], s); }
            pts
        };
        let points = draw_arc(start_angle, sweep, stroke);

        // Arrowheads at both ends of main arc
        let asz = 6.0;
        let draw_arrow = |tip: egui::Pos2, prev: egui::Pos2| {
            let adx = prev.x - tip.x;
            let ady = prev.y - tip.y;
            let alen = (adx * adx + ady * ady).sqrt().max(1.0);
            let (ax, ay) = (adx / alen, ady / alen);
            painter.line_segment([tip, egui::Pos2::new(tip.x + ax * asz + ay * asz * 0.4,
                tip.y + ay * asz - ax * asz * 0.4)], stroke);
            painter.line_segment([tip, egui::Pos2::new(tip.x + ax * asz - ay * asz * 0.4,
                tip.y + ay * asz + ax * asz * 0.4)], stroke);
        };
        if points.len() >= 2 {
            draw_arrow(points[0], points[1]);
            let n = points.len();
            draw_arrow(points[n - 1], points[n - 2]);
        }

        // Text position along arc: text_along=0 is center, +-0.5 at edges
        let text_angle = start_angle + sweep * (0.5 + text_along);

        // If text is outside sector, draw extension arc past the text
        // Extra angle to extend under the text (half text width in screen px / arc radius in screen px)
        let screen_radius = (self.to_screen(vect2d::new(ix.x + radius, ix.y)).x - six.x).abs().max(1.0);
        let text_half_angle = 20.0 / screen_radius; // ~20px half-width in angle
        let extra = (text_half_angle as f64) * sweep.signum();
        if text_along < -0.5 {
            let ext_sweep = sweep * (text_along + 0.5) - extra;
            draw_arc(start_angle, ext_sweep, ext_stroke);
        } else if text_along > 0.5 {
            let ext_sweep = sweep * (text_along - 0.5) + extra;
            draw_arc(start_angle + sweep, ext_sweep, ext_stroke);
        }

        // Draw rotated text tangent to arc at text position
        let text_pt = vect2d::new(ix.x + radius * text_angle.cos(), ix.y + radius * text_angle.sin());
        let screen_pt = self.to_screen(text_pt);
        // Tangent direction in screen space (Y is flipped vs math convention)
        let sign = if sweep >= 0.0 { 1.0f32 } else { -1.0f32 };
        let tx = -(text_angle.sin() as f32) * sign;
        let ty = -(text_angle.cos() as f32) * sign;
        let text = if is_derived { format!("({:.1}\u{00b0})", value) }
            else if is_expr { format!("fx: {:.1}\u{00b0}", value) }
            else { format!("{:.1}\u{00b0}", value) };
        self.draw_rotated_text(painter, screen_pt, tx, ty, &text,
            egui::FontId::proportional(12.0), color)
    }

    fn draw_sweep_dimension(&self, painter: &egui::Painter, arc_ref: Ref<Arc>,
                             value: f64, offset: vect2d, text_along: f64,
                             color: egui::Color32, is_expr: bool, is_derived: bool) -> (egui::Pos2, egui::Pos2) {
        let a = &self.sketch.arcs[arc_ref];
        let cx = a.center.value.x;
        let cy = a.center.value.y;
        let (start_angle, sweep) = arc_span(a);
        // Annotation arc radius: arc radius + offset.y (positive = outside, negative = inside)
        let radius = (a.radius.value + offset.y).max(0.1);
        let stroke = egui::Stroke::new(1.0, color);
        let ext_stroke = egui::Stroke::new(0.5, color);
        let center = vect2d::new(cx, cy);
        let sc = self.to_screen(center);

        // Extension lines from arc endpoints to annotation arc
        let arc_start = vect2d::new(cx + a.radius.value * start_angle.cos(),
                                    cy + a.radius.value * start_angle.sin());
        let ann_start = vect2d::new(cx + radius * start_angle.cos(),
                                    cy + radius * start_angle.sin());
        let end_angle = start_angle + sweep;
        let arc_end = vect2d::new(cx + a.radius.value * end_angle.cos(),
                                  cy + a.radius.value * end_angle.sin());
        let ann_end = vect2d::new(cx + radius * end_angle.cos(),
                                  cy + radius * end_angle.sin());
        painter.line_segment([self.to_screen(arc_start), self.to_screen(ann_start)], ext_stroke);
        painter.line_segment([self.to_screen(arc_end), self.to_screen(ann_end)], ext_stroke);

        // Main arc polyline
        let draw_arc = |a_start: f64, a_sweep: f64, s: egui::Stroke| -> Vec<egui::Pos2> {
            let n = ((a_sweep.abs() * 20.0).ceil() as usize).max(8);
            let pts: Vec<egui::Pos2> = (0..=n).map(|i| {
                let t = i as f64 / n as f64;
                let ang = a_start + a_sweep * t;
                self.to_screen(vect2d::new(cx + radius * ang.cos(), cy + radius * ang.sin()))
            }).collect();
            for w in pts.windows(2) { painter.line_segment([w[0], w[1]], s); }
            pts
        };
        let points = draw_arc(start_angle, sweep, stroke);

        // Arrowheads
        let asz = 6.0;
        let draw_arrow = |tip: egui::Pos2, prev: egui::Pos2| {
            let adx = prev.x - tip.x;
            let ady = prev.y - tip.y;
            let alen = (adx * adx + ady * ady).sqrt().max(1.0);
            let (ax, ay) = (adx / alen, ady / alen);
            painter.line_segment([tip, egui::Pos2::new(tip.x + ax * asz + ay * asz * 0.4,
                tip.y + ay * asz - ax * asz * 0.4)], stroke);
            painter.line_segment([tip, egui::Pos2::new(tip.x + ax * asz - ay * asz * 0.4,
                tip.y + ay * asz + ax * asz * 0.4)], stroke);
        };
        if points.len() >= 2 {
            draw_arrow(points[0], points[1]);
            let n = points.len();
            draw_arrow(points[n - 1], points[n - 2]);
        }

        // Text along arc
        let text_angle = start_angle + sweep * (0.5 + text_along);
        let _ = sc; // center screen pos available if needed

        // Extension arcs for out-of-range text
        let screen_radius = (self.to_screen(vect2d::new(cx + radius, cy)).x - sc.x).abs().max(1.0);
        let text_half_angle = 20.0 / screen_radius;
        let extra = (text_half_angle as f64) * sweep.signum();
        if text_along < -0.5 {
            let ext_sweep = sweep * (text_along + 0.5) - extra;
            draw_arc(start_angle, ext_sweep, ext_stroke);
        } else if text_along > 0.5 {
            let ext_sweep = sweep * (text_along - 0.5) + extra;
            draw_arc(start_angle + sweep, ext_sweep, ext_stroke);
        }

        let text_pt = vect2d::new(cx + radius * text_angle.cos(), cy + radius * text_angle.sin());
        let screen_pt = self.to_screen(text_pt);
        let sign = if sweep >= 0.0 { 1.0f32 } else { -1.0f32 };
        let tx = -(text_angle.sin() as f32) * sign;
        let ty = -(text_angle.cos() as f32) * sign;
        let text = if is_derived { format!("({:.1}\u{00b0})", value) }
            else if is_expr { format!("fx: {:.1}\u{00b0}", value) }
            else { format!("{:.1}\u{00b0}", value) };
        self.draw_rotated_text(painter, screen_pt, tx, ty, &text,
            egui::FontId::proportional(12.0), color)
    }

    // Compute the screen-space text segment for a dimension (for hit testing without drawing)
    pub fn dim_text_segment(&self, dim: &Dimension) -> (egui::Pos2, egui::Pos2) {
        let is_radius = matches!(dim.kind, DimensionKind::ArcRadius(_));
        let text = if is_radius { format!("R{:.2}", dim.value) } else { format!("{:.2}", dim.value) };
        let char_width = 12.0 * 0.6;
        let total_width = text.len() as f32 * char_width;

        if is_radius {
            if let DimensionKind::ArcRadius(r) = dim.kind {
                let a = &self.sketch.arcs[r];
                let angle = dim.offset.x;
                let edge = vect2d::new(
                    a.center.value.x + a.radius.value * angle.cos(),
                    a.center.value.y + a.radius.value * angle.sin(),
                );
                let arrow_len = a.radius.value * 0.6;
                let inner = vect2d::new(
                    edge.x - arrow_len * angle.cos(),
                    edge.y - arrow_len * angle.sin(),
                );
                let se = self.to_screen(edge);
                let si = self.to_screen(inner);
                let adx = si.x - se.x;
                let ady = si.y - se.y;
                let alen = (adx * adx + ady * ady).sqrt().max(1.0);
                let dx = if adx / alen < 0.0 { -adx / alen } else { adx / alen };
                let dy = if adx / alen < 0.0 { -ady / alen } else { ady / alen };
                let text_offset = 8.0;
                let tnx = -dy;
                let tny = dx;
                let mid = egui::Pos2::new(
                    (se.x + si.x) / 2.0 - tnx * text_offset,
                    (se.y + si.y) / 2.0 - tny * text_offset,
                );
                return (
                    egui::Pos2::new(mid.x - dx * total_width / 2.0, mid.y - dy * total_width / 2.0),
                    egui::Pos2::new(mid.x + dx * total_width / 2.0, mid.y + dy * total_width / 2.0),
                );
            }
        }

        // Sweep dimension: text along arc
        if let DimensionKind::ArcSweep(r) = dim.kind {
            let a = &self.sketch.arcs[r];
            let cx = a.center.value.x;
            let cy = a.center.value.y;
            let (start_angle, sweep) = arc_span(a);
            let radius = (a.radius.value + dim.offset.y).max(0.1);
            let text_angle = start_angle + sweep * (0.5 + dim.text_along);
            let text_pt = vect2d::new(cx + radius * text_angle.cos(), cy + radius * text_angle.sin());
            let screen_pt = self.to_screen(text_pt);
            let sign = if sweep >= 0.0 { 1.0f32 } else { -1.0f32 };
            let tx = -(text_angle.sin() as f32) * sign;
            let ty = -(text_angle.cos() as f32) * sign;
            let (dx, dy) = if tx < 0.0 { (-tx, -ty) } else { (tx, ty) };
            let nx = -dy;
            let ny = dx;
            let half_h = 6.0;
            let mid = egui::Pos2::new(screen_pt.x - nx * (half_h + 2.0), screen_pt.y - ny * (half_h + 2.0));
            return (
                egui::Pos2::new(mid.x - dx * total_width / 2.0, mid.y - dy * total_width / 2.0),
                egui::Pos2::new(mid.x + dx * total_width / 2.0, mid.y + dy * total_width / 2.0),
            );
        }

        // Angle dimension: text along arc -- match draw_rotated_text positioning
        if let DimensionKind::Angle(a_ref, b_ref, supplement) = dim.kind {
            let (ix, start, sweep) = self.angle_dim_sector(a_ref, b_ref, supplement, dim.offset);
            let radius = dim.offset.y.max(0.3);
            let text_angle = start + sweep * (0.5 + dim.text_along);
            let text_pt = vect2d::new(ix.x + radius * text_angle.cos(), ix.y + radius * text_angle.sin());
            let screen_pt = self.to_screen(text_pt);
            // Same tangent as draw_angle_dimension
            let sign = if sweep >= 0.0 { 1.0f32 } else { -1.0f32 };
            let tx = -(text_angle.sin() as f32) * sign;
            let ty = -(text_angle.cos() as f32) * sign;
            // draw_rotated_text ensures left-to-right
            let (dx, dy) = if tx < 0.0 { (-tx, -ty) } else { (tx, ty) };
            let nx = -dy;
            let ny = dx;
            let half_h = 6.0;
            let mid = egui::Pos2::new(screen_pt.x - nx * (half_h + 2.0), screen_pt.y - ny * (half_h + 2.0));
            return (
                egui::Pos2::new(mid.x - dx * total_width / 2.0, mid.y - dy * total_width / 2.0),
                egui::Pos2::new(mid.x + dx * total_width / 2.0, mid.y + dy * total_width / 2.0),
            );
        }

        let (p1, p2) = self.dim_endpoints(&dim.kind);
        let ddx = p2.x - p1.x;
        let ddy = p2.y - p1.y;
        let len = (ddx * ddx + ddy * ddy).sqrt().max(1e-12);
        let nx = -ddy / len;
        let ny = ddx / len;
        let off = dim.offset.y;
        let q1 = vect2d::new(p1.x + nx * off, p1.y + ny * off);
        let q2 = vect2d::new(p2.x + nx * off, p2.y + ny * off);
        let sq1 = self.to_screen(q1);
        let sq2 = self.to_screen(q2);
        let adx = sq2.x - sq1.x;
        let ady = sq2.y - sq1.y;
        let alen = (adx * adx + ady * ady).sqrt().max(1.0);
        let dx = if adx / alen < 0.0 { -adx / alen } else { adx / alen };
        let dy = if adx / alen < 0.0 { -ady / alen } else { ady / alen };
        // Apply same perpendicular offset as draw_rotated_text (text is above the line)
        let text_offset = 8.0;
        let tnx = -dy;
        let tny = dx;
        // Apply text_along offset
        let along_offset = dim.text_along as f32 * alen;
        let mid = egui::Pos2::new(
            (sq1.x + sq2.x) / 2.0 + (adx / alen) * along_offset - tnx * text_offset,
            (sq1.y + sq2.y) / 2.0 + (ady / alen) * along_offset - tny * text_offset,
        );
        (
            egui::Pos2::new(mid.x - dx * total_width / 2.0, mid.y - dy * total_width / 2.0),
            egui::Pos2::new(mid.x + dx * total_width / 2.0, mid.y + dy * total_width / 2.0),
        )
    }

    // Distance from a screen point to a screen-space line segment
    pub fn screen_point_to_segment_dist(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len2 = dx * dx + dy * dy;
        if len2 < 1.0 {
            return ((p.x - a.x).powi(2) + (p.y - a.y).powi(2)).sqrt();
        }
        let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0);
        let proj_x = a.x + t * dx;
        let proj_y = a.y + t * dy;
        ((p.x - proj_x).powi(2) + (p.y - proj_y).powi(2)).sqrt()
    }

    // Compute the marker position for a line, offset perpendicular by `offset_px` screen pixels.
    // `along` shifts along the line direction (for stacking multiple markers).
    pub fn line_marker_pos(&self, line_ref: Ref<Line>, offset_px: f32, along: f32) -> egui::Pos2 {
        let l = &self.sketch.lines[line_ref];
        let p1 = self.to_screen(l.p1.value);
        let p2 = self.to_screen(l.p2.value);
        let mx = (p1.x + p2.x) / 2.0;
        let my = (p1.y + p2.y) / 2.0;
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let len = (dx * dx + dy * dy).sqrt().max(1.0);
        // Normal (perpendicular), always point "up" (negative y in screen)
        let nx = -dy / len;
        let ny = dx / len;
        let sign = if ny > 0.0 { -1.0 } else { 1.0 };
        let ux = dx / len;
        let uy = dy / len;
        egui::Pos2::new(
            mx + nx * offset_px * sign + ux * along,
            my + ny * offset_px * sign + uy * along,
        )
    }

    // Compute marker position for an arc (at the midpoint of the arc curve).
    // Position a constraint marker inside the arc curve, spread along it by index.
    pub fn arc_marker_pos(&self, arc_ref: Ref<Arc>, idx: i32) -> egui::Pos2 {
        let a = &self.sketch.arcs[arc_ref];
        let (sa, span) = if a.closed { (0.0, std::f64::consts::TAU) } else { arc_span(a) };
        let mid_angle = sa + span / 2.0;
        // Spread markers along the arc near the midpoint
        let angle_offset = idx as f64 * 12.0 / (a.radius.value * self.scale as f64).max(1.0);
        let angle = mid_angle + angle_offset;
        // Place inside the curve (negative offset from radius)
        let r = a.radius.value - 10.0 / self.scale as f64;
        let pos = vect2d::new(
            a.center.value.x + r * angle.cos(),
            a.center.value.y + r * angle.sin(),
        );
        self.to_screen(pos)
    }

    // Build constraint markers for the current frame
    pub fn build_constraint_markers(&mut self) {
        self.constraint_markers.clear();

        // Track how many markers each line/arc already has (for stacking)
        let mut line_marker_count: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();
        let mut arc_marker_count: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();

        let add_line_marker = |this: &EditorApp, markers: &mut Vec<ConstraintMarker>,
                                    line: Ref<Line>, symbol: ConstraintSymbol, id: ConstraintId,
                                    counts: &mut std::collections::HashMap<u32, i32>| {
            let idx = *counts.get(&line.index()).unwrap_or(&0);
            *counts.entry(line.index()).or_insert(0) += 1;
            let along = (idx as f32 - 0.5) * 14.0; // spread along the line
            let pos = this.line_marker_pos(line, 10.0, along);
            markers.push(ConstraintMarker { pos, symbol, id });
        };

        let add_arc_marker = |this: &EditorApp, markers: &mut Vec<ConstraintMarker>,
                                    arc: Ref<Arc>, symbol: ConstraintSymbol, id: ConstraintId,
                                    counts: &mut std::collections::HashMap<u32, i32>| {
            let idx = *counts.get(&arc.index()).unwrap_or(&0);
            *counts.entry(arc.index()).or_insert(0) += 1;
            let pos = this.arc_marker_pos(arc, idx);
            markers.push(ConstraintMarker { pos, symbol, id });
        };

        let add_point_marker = |this: &EditorApp, markers: &mut Vec<ConstraintMarker>,
                                    point: Ref<Point>, symbol: ConstraintSymbol, id: ConstraintId| {
            let p = this.sketch.points[point].pos.value;
            let pos = this.to_screen(p);
            let pos = egui::pos2(pos.x, pos.y - 12.0); // offset above the point
            markers.push(ConstraintMarker { pos, symbol, id });
        };

        // Collect markers into a temporary vec, then assign
        let mut markers = Vec::new();

        // Self-constraints on lines
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            if l.constraints.horizontal {
                add_line_marker(self, &mut markers, r, ConstraintSymbol::H, ConstraintId::Horizontal(r), &mut line_marker_count);
            }
            if l.constraints.vertical {
                add_line_marker(self, &mut markers, r, ConstraintSymbol::V, ConstraintId::Vertical(r), &mut line_marker_count);
            }
        }

        // Shared constraints
        for (i, c) in self.sketch.parallel.iter().enumerate() {
            let id = ConstraintId::Parallel(i);
            add_line_marker(self, &mut markers, c.a, ConstraintSymbol::Parallel, id, &mut line_marker_count);
            add_line_marker(self, &mut markers, c.b, ConstraintSymbol::Parallel, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.perpendicular.iter().enumerate() {
            let id = ConstraintId::Perpendicular(i);
            add_line_marker(self, &mut markers, c.a, ConstraintSymbol::Perpendicular, id, &mut line_marker_count);
            add_line_marker(self, &mut markers, c.b, ConstraintSymbol::Perpendicular, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.equal_length.iter().enumerate() {
            let id = ConstraintId::EqualLength(i);
            add_line_marker(self, &mut markers, c.a, ConstraintSymbol::Equal, id, &mut line_marker_count);
            add_line_marker(self, &mut markers, c.b, ConstraintSymbol::Equal, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.equal_radius.iter().enumerate() {
            let id = ConstraintId::EqualRadius(i);
            add_arc_marker(self, &mut markers, c.a, ConstraintSymbol::Equal, id, &mut arc_marker_count);
            add_arc_marker(self, &mut markers, c.b, ConstraintSymbol::Equal, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.collinear.iter().enumerate() {
            let id = ConstraintId::Collinear(i);
            add_line_marker(self, &mut markers, c.a, ConstraintSymbol::Collinear, id, &mut line_marker_count);
            add_line_marker(self, &mut markers, c.b, ConstraintSymbol::Collinear, id, &mut line_marker_count);
        }
        // Midpoint constraints -- place marker on the target line
        for (i, c) in self.sketch.midpoint.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::Point, i);
            add_line_marker(self, &mut markers, c.line, ConstraintSymbol::Midpoint, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.midpoint_lp1.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::LP1, i);
            add_line_marker(self, &mut markers, c.target, ConstraintSymbol::Midpoint, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.midpoint_lp2.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::LP2, i);
            add_line_marker(self, &mut markers, c.target, ConstraintSymbol::Midpoint, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.midpoint_arc_start.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::ArcStart, i);
            add_line_marker(self, &mut markers, c.line, ConstraintSymbol::Midpoint, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.midpoint_arc_end.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::ArcEnd, i);
            add_line_marker(self, &mut markers, c.line, ConstraintSymbol::Midpoint, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.midpoint_arc_point.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::ArcPoint, i);
            add_arc_marker(self, &mut markers, c.arc, ConstraintSymbol::Midpoint, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.midpoint_lp1_arc.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::LP1Arc, i);
            add_arc_marker(self, &mut markers, c.arc, ConstraintSymbol::Midpoint, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.midpoint_lp2_arc.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::LP2Arc, i);
            add_arc_marker(self, &mut markers, c.arc, ConstraintSymbol::Midpoint, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.midpoint_arc_start_arc.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::ArcStartArc, i);
            add_arc_marker(self, &mut markers, c.b, ConstraintSymbol::Midpoint, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.midpoint_arc_end_arc.iter().enumerate() {
            let id = ConstraintId::Midpoint(MidpointKind::ArcEndArc, i);
            add_arc_marker(self, &mut markers, c.b, ConstraintSymbol::Midpoint, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.symmetry_ll.iter().enumerate() {
            let id = ConstraintId::Symmetry(i);
            add_line_marker(self, &mut markers, c.a, ConstraintSymbol::Symmetry, id, &mut line_marker_count);
            add_line_marker(self, &mut markers, c.b, ConstraintSymbol::Symmetry, id, &mut line_marker_count);
            add_line_marker(self, &mut markers, c.c, ConstraintSymbol::Symmetry, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.symmetry_pp.iter().enumerate() {
            let id = ConstraintId::SymmetryPP(i);
            add_line_marker(self, &mut markers, c.line, ConstraintSymbol::Symmetry, id, &mut line_marker_count);
            add_point_marker(self, &mut markers, c.a, ConstraintSymbol::Symmetry, id);
            add_point_marker(self, &mut markers, c.c, ConstraintSymbol::Symmetry, id);
        }
        for (i, c) in self.sketch.symmetry_aa.iter().enumerate() {
            let id = ConstraintId::SymmetryAA(i);
            add_line_marker(self, &mut markers, c.line, ConstraintSymbol::Symmetry, id, &mut line_marker_count);
            add_arc_marker(self, &mut markers, c.a, ConstraintSymbol::Symmetry, id, &mut arc_marker_count);
            add_arc_marker(self, &mut markers, c.c, ConstraintSymbol::Symmetry, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.tangent_la.iter().enumerate() {
            let id = ConstraintId::TangentLA(i);
            add_line_marker(self, &mut markers, c.line, ConstraintSymbol::Tangent, id, &mut line_marker_count);
            add_arc_marker(self, &mut markers, c.arc, ConstraintSymbol::Tangent, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.tangent_aa.iter().enumerate() {
            let id = ConstraintId::TangentAA(i);
            add_arc_marker(self, &mut markers, c.a, ConstraintSymbol::Tangent, id, &mut arc_marker_count);
            add_arc_marker(self, &mut markers, c.b, ConstraintSymbol::Tangent, id, &mut arc_marker_count);
        }

        // Coincident display setup
        let sel = &self.selection;
        let pt_sel = |r: Ref<Point>| sel.contains(&Selection::Point(r));
        let lp1_sel = |r: Ref<Line>| sel.contains(&Selection::LineP1(r));
        let lp2_sel = |r: Ref<Line>| sel.contains(&Selection::LineP2(r));
        let ac_sel = |r: Ref<Arc>| sel.contains(&Selection::ArcCenter(r));
        let as_sel = |r: Ref<Arc>| sel.contains(&Selection::ArcStart(r));
        let ae_sel = |r: Ref<Arc>| sel.contains(&Selection::ArcEnd(r));

        // Helper point bridges: show as single markers
        let mut helper_point_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut coinc_count: std::collections::HashMap<u64, i32> = std::collections::HashMap::new();
        let pos_key = |p: egui::Pos2| -> u64 { ((p.x * 100.0) as u64) << 32 | ((p.y * 100.0) as u64) };
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if !p.helper { continue; }
            helper_point_ids.insert(r.index());

            let bridge_id = ConstraintId::HelperBridge(r);
            let bridge_selected = sel.contains(&Selection::Constraint(bridge_id));
            let mut visible = bridge_selected;
            if !visible {
                for c in &self.sketch.coincident_lp1 { if c.point == r { visible |= lp1_sel(c.line); } }
                for c in &self.sketch.coincident_lp2 { if c.point == r { visible |= lp2_sel(c.line); } }
                for c in &self.sketch.coincident_arc_center { if c.point == r { visible |= ac_sel(c.arc); } }
                for c in &self.sketch.coincident_arc_start { if c.point == r { visible |= as_sel(c.arc); } }
                for c in &self.sketch.coincident_arc_end { if c.point == r { visible |= ae_sel(c.arc); } }
                for c in &self.sketch.coincident_pp { if c.a == r { visible |= pt_sel(c.b); } if c.b == r { visible |= pt_sel(c.a); } }
            }
            if visible {
                let pos = self.to_screen(p.pos.value);
                let key = pos_key(pos);
                let idx = *coinc_count.get(&key).unwrap_or(&0);
                *coinc_count.entry(key).or_insert(0) += 1;
                let offset = egui::Vec2::new(8.0 + idx as f32 * 12.0, -8.0);
                markers.push(ConstraintMarker { pos: pos + offset, symbol: ConstraintSymbol::Coincident, id: bridge_id });
            }
        }

        // Coincident constraints - collect, skip those involving helper points
        // Phase 1: collect all coincident markers with their base position and visibility flag
        struct CoincidentEntry {
            base_pos: egui::Pos2,
            id: ConstraintId,
            vertex_selected: bool,
        }
        let mut coinc_entries: Vec<CoincidentEntry> = Vec::new();

        let mut add_coinc_entry = |_markers: &mut Vec<ConstraintMarker>, pos: egui::Pos2, id: ConstraintId, visible: bool| {
            coinc_entries.push(CoincidentEntry { base_pos: pos, id, vertex_selected: visible });
        };

        // Skip constraints that reference helper points (those are shown as HelperBridge markers)
        let skip_if_helper_pp = |c: &CoincidentPP| -> bool {
            helper_point_ids.contains(&c.a.index()) || helper_point_ids.contains(&c.b.index())
        };
        let skip_if_helper_pt = |pt: Ref<Point>| -> bool {
            helper_point_ids.contains(&pt.index())
        };

        macro_rules! coinc {
            ($markers:expr, $coll:expr, $kind:expr, $pos_expr:expr, $vis_expr:expr) => {
                for (i, c) in $coll.iter().enumerate() {
                    let id = ConstraintId::Coincident($kind, i);
                    let pos = $pos_expr(c);
                    let vis = $vis_expr(c);
                    add_coinc_entry(&mut $markers, pos, id, vis);
                }
            };
            ($markers:expr, $coll:expr, $kind:expr, $pos_expr:expr, $vis_expr:expr, skip_helper: $skip:expr) => {
                for (i, c) in $coll.iter().enumerate() {
                    if $skip(c) { continue; }
                    let id = ConstraintId::Coincident($kind, i);
                    let pos = $pos_expr(c);
                    let vis = $vis_expr(c);
                    add_coinc_entry(&mut $markers, pos, id, vis);
                }
            };
        }

        // Point-Point
        coinc!(markers, self.sketch.coincident_pp, CoincidentKind::PP,
            |c: &CoincidentPP| self.to_screen(self.sketch.points[c.a].pos.value),
            |c: &CoincidentPP| pt_sel(c.a) || pt_sel(c.b),
            skip_helper: |c: &CoincidentPP| skip_if_helper_pp(c));
        // Line-Point
        coinc!(markers, self.sketch.coincident_lp1, CoincidentKind::LP1,
            |c: &CoincidentLP1| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &CoincidentLP1| lp1_sel(c.line) || pt_sel(c.point),
            skip_helper: |c: &CoincidentLP1| skip_if_helper_pt(c.point));
        coinc!(markers, self.sketch.coincident_lp2, CoincidentKind::LP2,
            |c: &CoincidentLP2| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &CoincidentLP2| lp2_sel(c.line) || pt_sel(c.point),
            skip_helper: |c: &CoincidentLP2| skip_if_helper_pt(c.point));
        // Line-Line
        coinc!(markers, self.sketch.coincident_ll11, CoincidentKind::LL11,
            |c: &CoincidentLL11| self.to_screen(self.sketch.lines[c.a].p1.value),
            |c: &CoincidentLL11| lp1_sel(c.a) || lp1_sel(c.b));
        coinc!(markers, self.sketch.coincident_ll12, CoincidentKind::LL12,
            |c: &CoincidentLL12| self.to_screen(self.sketch.lines[c.a].p1.value),
            |c: &CoincidentLL12| lp1_sel(c.a) || lp2_sel(c.b));
        coinc!(markers, self.sketch.coincident_ll21, CoincidentKind::LL21,
            |c: &CoincidentLL21| self.to_screen(self.sketch.lines[c.a].p2.value),
            |c: &CoincidentLL21| lp2_sel(c.a) || lp1_sel(c.b));
        coinc!(markers, self.sketch.coincident_ll22, CoincidentKind::LL22,
            |c: &CoincidentLL22| self.to_screen(self.sketch.lines[c.a].p2.value),
            |c: &CoincidentLL22| lp2_sel(c.a) || lp2_sel(c.b));
        // Point on line/arc
        coinc!(markers, self.sketch.point_on_line, CoincidentKind::PointOnLine,
            |c: &PointOnLine| self.to_screen(self.sketch.points[c.point].pos.value),
            |c: &PointOnLine| pt_sel(c.point),
            skip_helper: |c: &PointOnLine| skip_if_helper_pt(c.point));
        coinc!(markers, self.sketch.point_on_arc, CoincidentKind::PointOnArc,
            |c: &PointOnArc| self.to_screen(self.sketch.points[c.point].pos.value),
            |c: &PointOnArc| pt_sel(c.point),
            skip_helper: |c: &PointOnArc| skip_if_helper_pt(c.point));
        // Line endpoint on line
        coinc!(markers, self.sketch.line_p1_on_line, CoincidentKind::LP1OnLine,
            |c: &LineP1OnLine| self.to_screen(self.sketch.lines[c.a].p1.value),
            |c: &LineP1OnLine| lp1_sel(c.a));
        coinc!(markers, self.sketch.line_p1_on_arc, CoincidentKind::LP1OnArc,
            |c: &LineP1OnArc| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &LineP1OnArc| lp1_sel(c.line));
        coinc!(markers, self.sketch.line_p2_on_arc, CoincidentKind::LP2OnArc,
            |c: &LineP2OnArc| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &LineP2OnArc| lp2_sel(c.line));
        coinc!(markers, self.sketch.line_p2_on_line, CoincidentKind::LP2OnLine,
            |c: &LineP2OnLine| self.to_screen(self.sketch.lines[c.a].p2.value),
            |c: &LineP2OnLine| lp2_sel(c.a));
        // Point-Arc
        coinc!(markers, self.sketch.coincident_arc_center, CoincidentKind::ArcCenter,
            |c: &CoincidentArcCenter| self.to_screen(self.sketch.arcs[c.arc].center.value),
            |c: &CoincidentArcCenter| pt_sel(c.point) || ac_sel(c.arc),
            skip_helper: |c: &CoincidentArcCenter| skip_if_helper_pt(c.point));
        coinc!(markers, self.sketch.coincident_arc_start, CoincidentKind::ArcStart,
            |c: &CoincidentArcStart| self.to_screen(arc_start_pos(&self.sketch.arcs[c.arc])),
            |c: &CoincidentArcStart| pt_sel(c.point) || as_sel(c.arc),
            skip_helper: |c: &CoincidentArcStart| skip_if_helper_pt(c.point));
        coinc!(markers, self.sketch.coincident_arc_end, CoincidentKind::ArcEnd,
            |c: &CoincidentArcEnd| self.to_screen(arc_end_pos(&self.sketch.arcs[c.arc])),
            |c: &CoincidentArcEnd| pt_sel(c.point) || ae_sel(c.arc),
            skip_helper: |c: &CoincidentArcEnd| skip_if_helper_pt(c.point));
        // Line-Arc
        coinc!(markers, self.sketch.coincident_lp1_arc_center, CoincidentKind::LP1ArcCenter,
            |c: &CoincidentLP1ArcCenter| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &CoincidentLP1ArcCenter| lp1_sel(c.line) || ac_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp2_arc_center, CoincidentKind::LP2ArcCenter,
            |c: &CoincidentLP2ArcCenter| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &CoincidentLP2ArcCenter| lp2_sel(c.line) || ac_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp1_arc_start, CoincidentKind::LP1ArcStart,
            |c: &CoincidentLP1ArcStart| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &CoincidentLP1ArcStart| lp1_sel(c.line) || as_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp2_arc_start, CoincidentKind::LP2ArcStart,
            |c: &CoincidentLP2ArcStart| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &CoincidentLP2ArcStart| lp2_sel(c.line) || as_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp1_arc_end, CoincidentKind::LP1ArcEnd,
            |c: &CoincidentLP1ArcEnd| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &CoincidentLP1ArcEnd| lp1_sel(c.line) || ae_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp2_arc_end, CoincidentKind::LP2ArcEnd,
            |c: &CoincidentLP2ArcEnd| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &CoincidentLP2ArcEnd| lp2_sel(c.line) || ae_sel(c.arc));
        // Arc-Arc
        coinc!(markers, self.sketch.concentric, CoincidentKind::ArcCenterStart, // reuse for concentric
            |c: &Concentric| self.to_screen(self.sketch.arcs[c.a].center.value),
            |c: &Concentric| ac_sel(c.a) || ac_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_center_start, CoincidentKind::ArcCenterStart,
            |c: &CoincidentArcCenterStart| self.to_screen(self.sketch.arcs[c.a].center.value),
            |c: &CoincidentArcCenterStart| ac_sel(c.a) || as_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_center_end, CoincidentKind::ArcCenterEnd,
            |c: &CoincidentArcCenterEnd| self.to_screen(self.sketch.arcs[c.a].center.value),
            |c: &CoincidentArcCenterEnd| ac_sel(c.a) || ae_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_start_center, CoincidentKind::ArcStartCenter,
            |c: &CoincidentArcStartCenter| self.to_screen(arc_start_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcStartCenter| as_sel(c.a) || ac_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_end_center, CoincidentKind::ArcEndCenter,
            |c: &CoincidentArcEndCenter| self.to_screen(arc_end_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcEndCenter| ae_sel(c.a) || ac_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_start_start, CoincidentKind::ArcStartStart,
            |c: &CoincidentArcStartStart| self.to_screen(arc_start_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcStartStart| as_sel(c.a) || as_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_start_end, CoincidentKind::ArcStartEnd,
            |c: &CoincidentArcStartEnd| self.to_screen(arc_start_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcStartEnd| as_sel(c.a) || ae_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_end_start, CoincidentKind::ArcEndStart,
            |c: &CoincidentArcEndStart| self.to_screen(arc_end_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcEndStart| ae_sel(c.a) || as_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_end_end, CoincidentKind::ArcEndEnd,
            |c: &CoincidentArcEndEnd| self.to_screen(arc_end_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcEndEnd| ae_sel(c.a) || ae_sel(c.b));

        // Phase 2: determine which positions should show markers.
        // A position is visible if any entry there has vertex_selected=true OR any entry there is selected as a constraint.
        let mut pos_visible: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let pos_key = |p: egui::Pos2| -> u64 { ((p.x * 100.0) as u64) << 32 | ((p.y * 100.0) as u64) };
        for e in &coinc_entries {
            let key = pos_key(e.base_pos);
            if e.vertex_selected || sel.contains(&Selection::Constraint(e.id)) {
                pos_visible.insert(key);
            }
        }

        // Phase 3: add visible coincident markers with stacking
        let mut coinc_count: std::collections::HashMap<u64, i32> = std::collections::HashMap::new();
        for e in &coinc_entries {
            let key = pos_key(e.base_pos);
            if !pos_visible.contains(&key) { continue; }
            let idx = *coinc_count.get(&key).unwrap_or(&0);
            *coinc_count.entry(key).or_insert(0) += 1;
            let offset = egui::Vec2::new(8.0 + idx as f32 * 12.0, -8.0);
            markers.push(ConstraintMarker {
                pos: e.base_pos + offset,
                symbol: ConstraintSymbol::Coincident,
                id: e.id,
            });
        }

        self.constraint_markers = markers;
    }

    // Draw the canvas
    pub fn draw_canvas(&self, painter: &egui::Painter, rect: egui::Rect, mouse_screen: egui::Pos2) {
        let c = &self.colors;
        let empty_set = std::collections::HashSet::new();
        let (pt_locked, l_p1_locked, l_p2_locked, arc_c_locked) = if self.show_constraints {
            self.compute_locked_sets()
        } else {
            (empty_set.clone(), empty_set.clone(), empty_set.clone(), empty_set.clone())
        };

        // Compute which line/arc endpoints are coincident-connected (to hide blue dots)
        let mut connected_lp1: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut connected_lp2: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut connected_arc_s: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut connected_arc_e: std::collections::HashSet<u32> = std::collections::HashSet::new();
        // LL coincidences
        for c in &self.sketch.coincident_ll11 { connected_lp1.insert(c.a.index()); connected_lp1.insert(c.b.index()); }
        for c in &self.sketch.coincident_ll12 { connected_lp1.insert(c.a.index()); connected_lp2.insert(c.b.index()); }
        for c in &self.sketch.coincident_ll21 { connected_lp2.insert(c.a.index()); connected_lp1.insert(c.b.index()); }
        for c in &self.sketch.coincident_ll22 { connected_lp2.insert(c.a.index()); connected_lp2.insert(c.b.index()); }
        // Line-Arc
        for c in &self.sketch.coincident_lp1_arc_center { connected_lp1.insert(c.line.index()); }
        for c in &self.sketch.coincident_lp2_arc_center { connected_lp2.insert(c.line.index()); }
        for c in &self.sketch.coincident_lp1_arc_start { connected_lp1.insert(c.line.index()); connected_arc_s.insert(c.arc.index()); }
        for c in &self.sketch.coincident_lp2_arc_start { connected_lp2.insert(c.line.index()); connected_arc_s.insert(c.arc.index()); }
        for c in &self.sketch.coincident_lp1_arc_end { connected_lp1.insert(c.line.index()); connected_arc_e.insert(c.arc.index()); }
        for c in &self.sketch.coincident_lp2_arc_end { connected_lp2.insert(c.line.index()); connected_arc_e.insert(c.arc.index()); }
        // Arc-Arc
        for c in &self.sketch.coincident_arc_start_start { connected_arc_s.insert(c.a.index()); connected_arc_s.insert(c.b.index()); }
        for c in &self.sketch.coincident_arc_start_end { connected_arc_s.insert(c.a.index()); connected_arc_e.insert(c.b.index()); }
        for c in &self.sketch.coincident_arc_end_start { connected_arc_e.insert(c.a.index()); connected_arc_s.insert(c.b.index()); }
        for c in &self.sketch.coincident_arc_end_end { connected_arc_e.insert(c.a.index()); connected_arc_e.insert(c.b.index()); }

        // Compute constraint-highlighted entities
        let mut highlight_lines: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut highlight_arcs: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut highlight_points: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut highlight_line_p1: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut highlight_line_p2: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut highlight_arc_start: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut highlight_arc_end: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut highlight_arc_center: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for sel in &self.selection {
            if let Selection::Constraint(id) = sel {
                let ce = self.constraint_entities(*id);
                for l in ce.lines { highlight_lines.insert(l.index()); }
                for a in ce.arcs { highlight_arcs.insert(a.index()); }
                for p in ce.points { highlight_points.insert(p.index()); }
                for l in ce.line_p1s { highlight_line_p1.insert(l.index()); }
                for l in ce.line_p2s { highlight_line_p2.insert(l.index()); }
                for a in ce.arc_starts { highlight_arc_start.insert(a.index()); }
                for a in ce.arc_ends { highlight_arc_end.insert(a.index()); }
                for a in ce.arc_centers { highlight_arc_center.insert(a.index()); }
            }
        }
        let highlight_color = egui::Color32::from_rgb(255, 120, 180); // pink

        // Background
        painter.rect_filled(rect, 0.0, c.background);

        // Grid
        self.draw_grid(painter, rect);

        // Lines
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            let p1 = self.to_screen(l.p1.value);
            let p2 = self.to_screen(l.p2.value);

            let selected = self.selection.contains(&Selection::Line(r));
            let color = if selected { c.line_selected }
                else if highlight_lines.contains(&r.index()) { highlight_color }
                else { c.line };
            let width = if l.style == LineStyle::Solid { 2.0 } else { 1.0 };
            draw_styled_polyline(painter, &[p1, p2], egui::Stroke::new(width, color), l.style);

            // Endpoints -- highlight individually if selected
            let p1_selected = self.is_endpoint_selected(r, true);
            let p2_selected = self.is_endpoint_selected(r, false);

            let p1_highlighted = highlight_line_p1.contains(&r.index());
            let p2_highlighted = highlight_line_p2.contains(&r.index());

            let ep1_color = if p1_selected { c.endpoint_selected }
                else if p1_highlighted { highlight_color }
                else if selected { c.endpoint_line_selected }
                else if l_p1_locked.contains(&r.index()) { c.point_locked }
                else { c.endpoint };
            let ep2_color = if p2_selected { c.endpoint_selected }
                else if p2_highlighted { highlight_color }
                else if selected { c.endpoint_line_selected }
                else if l_p2_locked.contains(&r.index()) { c.point_locked }
                else { c.endpoint };

            let ep1_radius = if p1_selected || p1_highlighted { 6.0 } else { 4.0 };
            let ep2_radius = if p2_selected || p2_highlighted { 6.0 } else { 4.0 };

            // Hide endpoint dot if coincident-connected, unless selected, locked, or highlighted
            let near_p1 = (mouse_screen.x - p1.x).powi(2) + (mouse_screen.y - p1.y).powi(2) < 225.0; // 15px
            let near_p2 = (mouse_screen.x - p2.x).powi(2) + (mouse_screen.y - p2.y).powi(2) < 225.0;
            let show_p1 = p1_selected || p1_highlighted || selected
                || l_p1_locked.contains(&r.index())
                || !connected_lp1.contains(&r.index())
                || near_p1;
            let show_p2 = p2_selected || p2_highlighted || selected
                || l_p2_locked.contains(&r.index())
                || !connected_lp2.contains(&r.index())
                || near_p2;
            if show_p1 { painter.circle_filled(p1, ep1_radius, ep1_color); }
            if show_p2 { painter.circle_filled(p2, ep2_radius, ep2_color); }

        }

        // Points (skip helper points)
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            if self.drag_point == Some(r) { continue; }
            let sp = self.to_screen(p.pos.value);
            let selected = self.selection.contains(&Selection::Point(r));
            let color = if selected { c.point_selected }
                else if highlight_points.contains(&r.index()) { highlight_color }
                else if pt_locked.contains(&r.index()) { c.point_locked }
                else { c.point };
            painter.circle_filled(sp, 5.0, color);
        }

        // Arcs
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let center = self.to_screen(a.center.value);
            let radius_px = a.radius.value as f32 * self.scale;
            let arc_selected = self.selection.contains(&Selection::Arc(r));
            let arc_color = if arc_selected { c.line_selected }
                else if highlight_arcs.contains(&r.index()) { highlight_color }
                else { c.arc };
            let arc_width = if a.style == LineStyle::Solid { 2.0 } else { 1.0 };
            let stroke = egui::Stroke::new(arc_width, arc_color);

            // Tessellate arc/circle
            let (sa, span) = if a.closed {
                (0.0, std::f64::consts::TAU)
            } else {
                arc_span(a)
            };
            let n_segs = ((span.abs() * radius_px as f64 / 4.0).ceil() as usize).clamp(8, 256);
            let points: Vec<egui::Pos2> = (0..=n_segs).map(|i| {
                let t = sa + span * (i as f64 / n_segs as f64);
                self.to_screen(vect2d::new(
                    a.center.value.x + a.radius.value * t.cos(),
                    a.center.value.y + a.radius.value * t.sin(),
                ))
            }).collect();
            draw_styled_polyline(painter, &points, stroke, a.style);

            if !a.closed {
                let start_sel = self.selection.contains(&Selection::ArcStart(r));
                let end_sel = self.selection.contains(&Selection::ArcEnd(r));
                let start_hl = highlight_arc_start.contains(&r.index());
                let end_hl = highlight_arc_end.contains(&r.index());
                let start_color = if start_sel { c.endpoint_selected }
                    else if start_hl { highlight_color }
                    else { c.endpoint };
                let end_color = if end_sel { c.endpoint_selected }
                    else if end_hl { highlight_color }
                    else { c.endpoint };
                let sp = points[0];
                let ep = *points.last().unwrap();
                let near_start = (mouse_screen.x - sp.x).powi(2) + (mouse_screen.y - sp.y).powi(2) < 225.0;
                let near_end = (mouse_screen.x - ep.x).powi(2) + (mouse_screen.y - ep.y).powi(2) < 225.0;
                let show_start = start_sel || start_hl || arc_selected || !connected_arc_s.contains(&r.index()) || near_start;
                let show_end = end_sel || end_hl || arc_selected || !connected_arc_e.contains(&r.index()) || near_end;
                if show_start { painter.circle_filled(points[0], if start_sel || start_hl { 6.0 } else { 4.0 }, start_color); }
                if show_end { painter.circle_filled(*points.last().unwrap(), if end_sel || end_hl { 6.0 } else { 4.0 }, end_color); }
            }
            // Center point
            let center_sel = self.selection.contains(&Selection::ArcCenter(r));
            let center_hl = highlight_arc_center.contains(&r.index());
            let center_locked = arc_c_locked.contains(&r.index());
            let center_color = if center_sel { c.endpoint_selected }
                else if center_hl { highlight_color }
                else if center_locked { c.point_locked }
                else { c.endpoint };
            painter.circle_filled(center, if center_sel { 5.0 } else { 3.0 }, center_color);
        }

        // Origin crosshair
        let origin = self.to_screen(vect2d::new(0.0, 0.0));
        let sz = 10.0;
        painter.line_segment(
            [egui::Pos2::new(origin.x - sz, origin.y), egui::Pos2::new(origin.x + sz, origin.y)],
            egui::Stroke::new(1.0, c.origin));
        painter.line_segment(
            [egui::Pos2::new(origin.x, origin.y - sz), egui::Pos2::new(origin.x, origin.y + sz)],
            egui::Stroke::new(1.0, c.origin));

        // Constraint markers (drawn with painter lines)
        for marker in &self.constraint_markers {
            let selected = self.selection.contains(&Selection::Constraint(marker.id));
            let color = if selected {
                egui::Color32::from_rgb(220, 40, 40)
            } else {
                c.constraint_marker
            };
            let w = if selected { 2.0 } else { 1.5 };
            let s = if selected { 7.0 } else { 5.0 }; // half-size
            let p = marker.pos;
            let stroke = egui::Stroke::new(w, color);
            match marker.symbol {
                ConstraintSymbol::H => {
                    // H shape: two verticals close together + horizontal crossbar
                    let g = s * 0.45;
                    painter.line_segment([egui::Pos2::new(p.x - g, p.y - s), egui::Pos2::new(p.x - g, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x + g, p.y - s), egui::Pos2::new(p.x + g, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x - g, p.y), egui::Pos2::new(p.x + g, p.y)], stroke);
                }
                ConstraintSymbol::V => {
                    // V shape: two diagonals meeting at bottom
                    painter.line_segment([egui::Pos2::new(p.x - s, p.y - s), egui::Pos2::new(p.x, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x + s, p.y - s), egui::Pos2::new(p.x, p.y + s)], stroke);
                }
                ConstraintSymbol::Parallel => {
                    // Two vertical parallel lines
                    let g = s * 0.35;
                    painter.line_segment([egui::Pos2::new(p.x - g, p.y - s), egui::Pos2::new(p.x - g, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x + g, p.y - s), egui::Pos2::new(p.x + g, p.y + s)], stroke);
                }
                ConstraintSymbol::Perpendicular => {
                    // T shape: horizontal line on bottom, vertical up from center
                    painter.line_segment([egui::Pos2::new(p.x - s, p.y + s), egui::Pos2::new(p.x + s, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x, p.y + s), egui::Pos2::new(p.x, p.y - s)], stroke);
                }
                ConstraintSymbol::Equal => {
                    // Two horizontal parallel lines
                    let g = s * 0.3;
                    painter.line_segment([egui::Pos2::new(p.x - s, p.y - g), egui::Pos2::new(p.x + s, p.y - g)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x - s, p.y + g), egui::Pos2::new(p.x + s, p.y + g)], stroke);
                }
                ConstraintSymbol::Tangent => {
                    // Small circle with a diagonal line tangent at top-right
                    let r = s * 0.45;
                    let cx = p.x - s * 0.15;
                    let cy = p.y + s * 0.15;
                    painter.circle_stroke(egui::Pos2::new(cx, cy), r, stroke);
                    // Touch point at 45 deg, nudged outward by stroke width
                    let k = std::f32::consts::FRAC_1_SQRT_2;
                    let ro = r + w;
                    let tx = cx + ro * k;
                    let ty = cy - ro * k;
                    // Tangent direction is perpendicular to radius.
                    // Radius direction at 45 deg: (k, -k). Perpendicular: (k, k).
                    let half = s * 0.9;
                    painter.line_segment([
                        egui::Pos2::new(tx - k * half, ty - k * half),
                        egui::Pos2::new(tx + k * half, ty + k * half),
                    ], stroke);
                }
                ConstraintSymbol::Collinear => {
                    // Diagonal line with gap in the middle
                    painter.line_segment([
                        egui::Pos2::new(p.x - s * 0.7, p.y + s * 0.7),
                        egui::Pos2::new(p.x - s * 0.1, p.y + s * 0.1),
                    ], stroke);
                    painter.line_segment([
                        egui::Pos2::new(p.x + s * 0.1, p.y - s * 0.1),
                        egui::Pos2::new(p.x + s * 0.7, p.y - s * 0.7),
                    ], stroke);
                }
                ConstraintSymbol::Midpoint => {
                    // Triangle pointing up
                    let h = s * 1.56;
                    let half_w = s * 1.04;
                    let top = egui::Pos2::new(p.x, p.y - h * 0.5);
                    let bl = egui::Pos2::new(p.x - half_w, p.y + h * 0.5);
                    let br = egui::Pos2::new(p.x + half_w, p.y + h * 0.5);
                    painter.line_segment([top, bl], stroke);
                    painter.line_segment([bl, br], stroke);
                    painter.line_segment([br, top], stroke);
                }
                ConstraintSymbol::Symmetry => {
                    // Three parallel vertical lines, middle one dashed
                    let h = s * 1.2;
                    let gap = s * 0.75;
                    let thin = egui::Stroke::new(w * 0.6, color);
                    // Outer lines (solid)
                    for dx in [-gap, gap] {
                        painter.line_segment([
                            egui::Pos2::new(p.x + dx as f32, p.y - h),
                            egui::Pos2::new(p.x + dx as f32, p.y + h),
                        ], thin);
                    }
                    // Middle line (dashed)
                    let dash = h * 0.4;
                    let mut y = p.y - h;
                    while y < p.y + h {
                        let y_end = (y + dash).min(p.y + h);
                        painter.line_segment([
                            egui::Pos2::new(p.x, y),
                            egui::Pos2::new(p.x, y_end),
                        ], thin);
                        y += dash * 2.0;
                    }
                }
                ConstraintSymbol::Coincident => {
                    // Corner with dot: small filled square + lines going right and up
                    let d = s * 0.25;
                    painter.rect_filled(
                        egui::Rect::from_center_size(
                            egui::Pos2::new(p.x - s * 0.3, p.y + s * 0.3),
                            egui::Vec2::splat(d * 2.0),
                        ), 0.0, color);
                    // Line going right
                    painter.line_segment([
                        egui::Pos2::new(p.x - s * 0.3, p.y + s * 0.3),
                        egui::Pos2::new(p.x + s * 0.7, p.y + s * 0.3),
                    ], stroke);
                    // Line going up
                    painter.line_segment([
                        egui::Pos2::new(p.x - s * 0.3, p.y + s * 0.3),
                        egui::Pos2::new(p.x - s * 0.3, p.y - s * 0.7),
                    ], stroke);
                }
            }
        }

        // Dimension annotations
        let dim_color = egui::Color32::from_rgb(200, 100, 50);
        let dim_sel_color = egui::Color32::from_rgb(220, 40, 40);
        let dim_broken_color = egui::Color32::from_rgb(255, 30, 30);
        for (i, dim) in self.sketch.dimensions.iter().enumerate() {
            let selected = self.selection.contains(&Selection::Dimension(i));
            let color = if dim.broken { dim_broken_color }
                        else if selected { dim_sel_color }
                        else { dim_color };
            let is_radius = matches!(dim.kind, DimensionKind::ArcRadius(_));
            let is_expr = dim.expr_str.is_some();
            self.draw_dimension(painter, &dim.kind, dim.value, dim.offset, dim.text_along, color, is_radius, is_expr, dim.derived);
        }

        // Redraw selected and locked points/endpoints on top so they're not obscured
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let selected = self.selection.contains(&Selection::Point(r));
            let locked = pt_locked.contains(&r.index());
            if selected || locked {
                let sp = self.to_screen(p.pos.value);
                let color = if selected { c.point_selected } else { c.point_locked };
                painter.circle_filled(sp, if selected { 6.0 } else { 5.0 }, color);
            }
        }
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            let p1s = self.is_endpoint_selected(r, true);
            let p2s = self.is_endpoint_selected(r, false);
            let p1l = l_p1_locked.contains(&r.index());
            let p2l = l_p2_locked.contains(&r.index());
            if p1s || p1l {
                let p1 = self.to_screen(l.p1.value);
                // Selected on top of locked: draw locked first, then selected ring
                if p1l { painter.circle_filled(p1, 5.0, c.point_locked); }
                if p1s { painter.circle_filled(p1, 6.0, c.endpoint_selected); }
                // If both, draw a green dot inside the orange
                if p1s && p1l { painter.circle_filled(p1, 3.0, c.point_locked); }
            }
            if p2s || p2l {
                let p2 = self.to_screen(l.p2.value);
                if p2l { painter.circle_filled(p2, 5.0, c.point_locked); }
                if p2s { painter.circle_filled(p2, 6.0, c.endpoint_selected); }
                if p2s && p2l { painter.circle_filled(p2, 3.0, c.point_locked); }
            }
        }
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let cs = self.selection.contains(&Selection::ArcCenter(r));
            let cl = arc_c_locked.contains(&r.index());
            if cs || cl {
                let center = self.to_screen(a.center.value);
                if cl { painter.circle_filled(center, 4.0, c.point_locked); }
                if cs { painter.circle_filled(center, 5.0, c.endpoint_selected); }
                if cs && cl { painter.circle_filled(center, 2.5, c.point_locked); }
            }
            if !a.closed {
                if self.selection.contains(&Selection::ArcStart(r)) {
                    let sp = self.to_screen(arc_start_pos(a));
                    painter.circle_filled(sp, 6.0, c.endpoint_selected);
                }
                if self.selection.contains(&Selection::ArcEnd(r)) {
                    let ep = self.to_screen(arc_end_pos(a));
                    painter.circle_filled(ep, 6.0, c.endpoint_selected);
                }
            }
        }
    }

    pub fn draw_grid(&self, painter: &egui::Painter, rect: egui::Rect) {
        let grid_color = self.colors.grid;

        // Determine grid spacing based on zoom
        let mut spacing = 1.0_f32;
        while spacing * self.scale < 30.0 { spacing *= 5.0; }
        while spacing * self.scale > 150.0 { spacing /= 5.0; }

        let tl = self.to_sketch(rect.left_top());
        let br = self.to_sketch(rect.right_bottom());

        let x_start = (tl.x.min(br.x) as f32 / spacing).floor() * spacing;
        let x_end = (tl.x.max(br.x) as f32 / spacing).ceil() * spacing;
        let y_start = (tl.y.min(br.y) as f32 / spacing).floor() * spacing;
        let y_end = (tl.y.max(br.y) as f32 / spacing).ceil() * spacing;

        let mut x = x_start;
        while x <= x_end {
            let sx = self.to_screen(vect2d::new(x as f64, 0.0)).x;
            painter.line_segment(
                [egui::Pos2::new(sx, rect.top()), egui::Pos2::new(sx, rect.bottom())],
                egui::Stroke::new(0.5, grid_color));
            x += spacing;
        }
        let mut y = y_start;
        while y <= y_end {
            let sy = self.to_screen(vect2d::new(0.0, y as f64)).y;
            painter.line_segment(
                [egui::Pos2::new(rect.left(), sy), egui::Pos2::new(rect.right(), sy)],
                egui::Stroke::new(0.5, grid_color));
            y += spacing;
        }
    }
}

// Draw a polyline with the given style (solid, dashed, dash-dot).
// `points` is a slice of screen-space positions.
pub fn draw_styled_polyline(painter: &egui::Painter, points: &[egui::Pos2], stroke: egui::Stroke, style: LineStyle) {
    match style {
        LineStyle::Solid => {
            for w in points.windows(2) {
                painter.line_segment([w[0], w[1]], stroke);
            }
        }
        LineStyle::Dashed => {
            draw_pattern_polyline(painter, points, stroke, &[10.0, 6.0]);
        }
        LineStyle::DashDot => {
            draw_pattern_polyline(painter, points, stroke, &[10.0, 4.0, 2.0, 4.0]);
        }
    }
}

// Draw a polyline with a repeating dash pattern (lengths in pixels).
// Even indices are drawn, odd indices are gaps.
pub fn draw_pattern_polyline(painter: &egui::Painter, points: &[egui::Pos2], stroke: egui::Stroke, pattern: &[f32]) {
    if points.len() < 2 || pattern.is_empty() { return; }
    let mut pat_idx = 0;
    let mut pat_remaining = pattern[0];
    let mut drawing = true; // even indices = draw, odd = gap
    let mut seg_start = points[0];

    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len < 1e-6 { continue; }
        let ux = dx / seg_len;
        let uy = dy / seg_len;
        let mut consumed = 0.0f32;

        // Walk along this segment
        while consumed < seg_len - 0.01 {
            let remaining_seg = seg_len - consumed;
            if pat_remaining <= remaining_seg {
                // Pattern element ends within this segment
                let end_x = a.x + ux * (consumed + pat_remaining);
                let end_y = a.y + uy * (consumed + pat_remaining);
                let end = egui::Pos2::new(end_x, end_y);
                if drawing {
                    painter.line_segment([seg_start, end], stroke);
                }
                consumed += pat_remaining;
                seg_start = end;
                // Advance pattern
                drawing = !drawing;
                pat_idx = (pat_idx + 1) % pattern.len();
                pat_remaining = pattern[pat_idx];
            } else {
                // Segment ends before pattern element
                pat_remaining -= remaining_seg;
                if drawing {
                    painter.line_segment([seg_start, b], stroke);
                }
                seg_start = b;
                consumed = seg_len;
            }
        }
    }
}
