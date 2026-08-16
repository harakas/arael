// Canvas overlays drawn on top of the sketch: tool previews, snap and
// constraint hints, box-select marquee, status bar, and the DOF/cost
// corner readout.

use eframe::egui;
use arael::vect::vect2d;
use arael::refs::Ref;
use arael_sketch_solver::*;
use crate::tools::*;
use arael_sketch_backend::geometry::*;
use crate::app_update::hv_snap_from;
use crate::EditorApp;

/// Draw the Collinear glyph at hint ("hover / emphasized") size.
/// Delegates to the shared glyph so live hints look the same as
/// committed markers.
pub(crate) fn draw_collinear_marker(painter: &egui::Painter, pos: egui::Pos2, color: egui::Color32) {
    crate::drawing::collinear_glyph(painter, pos, 7.0, egui::Stroke::new(2.0, color));
}

/// Draw an H or V constraint symbol at hint size. Delegates to the
/// shared glyph so live hints look the same as committed markers.
pub(crate) fn draw_hv_marker(painter: &egui::Painter, pos: egui::Pos2, horizontal: bool, color: egui::Color32) {
    crate::drawing::hv_glyph(painter, pos, horizontal, 7.0, egui::Stroke::new(2.0, color));
}

// Small right-angle corner marker placed at the corner where the drawn
// line meets the host line. Two short segments form the open "L".
//
// The host-arm direction is chosen to lie along the portion of the host
// line that actually exists: if the corner sits at host.P1, the arm
// points toward P2; if at host.P2, toward P1. For a mid-body corner both
// sides are valid, so we fall back to the quadrant that also sits along
// the drawn-line direction -- keeping the marker visually "hugging" the
// drawn line's side.
pub(crate) fn draw_perp_corner_marker(
    painter: &egui::Painter,
    corner_screen: egui::Pos2,
    host_p1: vect2d,
    host_p2: vect2d,
    corner_sketch: vect2d,
    drawn_free: vect2d,
    _scale: f32,
    color: egui::Color32,
) {
    let dir_screen = |from: vect2d, to: vect2d| -> Option<(f32, f32)> {
        let dx = (to.x - from.x) as f32;
        // Screen y is flipped vs sketch y.
        let dy = -(to.y - from.y) as f32;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-6 { None } else { Some((dx / len, dy / len)) }
    };
    let Some((dx, dy)) = dir_screen(corner_sketch, drawn_free) else { return; };

    // Host-arm direction: prefer "into the existing host line".
    let d1_sq = (corner_sketch.x - host_p1.x).powi(2) + (corner_sketch.y - host_p1.y).powi(2);
    let d2_sq = (corner_sketch.x - host_p2.x).powi(2) + (corner_sketch.y - host_p2.y).powi(2);
    let host_dir_target = if d1_sq < d2_sq * 0.01 {
        host_p2   // corner ~ host.P1; point toward P2
    } else if d2_sq < d1_sq * 0.01 {
        host_p1   // corner ~ host.P2; point toward P1
    } else {
        // Mid-body: neither endpoint clearly closer. Use the host-parallel
        // direction whose dot with the drawn direction is positive (marker
        // hugs the drawn line).
        let hdx = (host_p2.x - host_p1.x) as f32;
        let hdy = -(host_p2.y - host_p1.y) as f32;
        if hdx * dx + hdy * dy >= 0.0 { host_p2 } else { host_p1 }
    };
    let Some((hx, hy)) = dir_screen(corner_sketch, host_dir_target) else { return; };

    let s = 10.0_f32;
    let a = egui::Pos2::new(corner_screen.x + hx * s, corner_screen.y + hy * s);
    let b = egui::Pos2::new(corner_screen.x + dx * s, corner_screen.y + dy * s);
    let c = egui::Pos2::new(corner_screen.x + (hx + dx) * s, corner_screen.y + (hy + dy) * s);
    let stroke = egui::Stroke::new(1.5, color);
    painter.line_segment([a, c], stroke);
    painter.line_segment([b, c], stroke);
}

impl EditorApp {
    /// Draw everything that sits on top of the canvas for the current
    /// frame: tool previews, snap/constraint hints, the box-select
    /// marquee, hint text, and the DOF / cost / version corner.
    pub(crate) fn draw_canvas_overlays(&mut self, ui: &mut egui::Ui, painter: &egui::Painter,
                                       rect: egui::Rect, mouse_screen: egui::Pos2,
                                       mouse_sketch: vect2d, hit_threshold: f64) {
        // Split/Trim hover preview: the span a click would isolate
        // (split, green) or remove (trim, red). A whole-entity trim
        // shows the full curve in the trim colour.
        if matches!(self.tool, Tool::Split | Tool::Trim) {
            self.draw_split_trim_preview(painter, mouse_sketch, hit_threshold);
        }
        // Scale center crosshair.
        if self.tool == Tool::Scale
            && let Some(c) = self.scale_center
        {
            let p = self.to_screen(c);
            let stroke = egui::Stroke::new(2.0, self.colors.constraint_marker_selected);
            let s = 9.0;
            painter.line_segment([egui::Pos2::new(p.x - s, p.y), egui::Pos2::new(p.x + s, p.y)], stroke);
            painter.line_segment([egui::Pos2::new(p.x, p.y - s), egui::Pos2::new(p.x, p.y + s)], stroke);
            painter.circle_stroke(p, s * 0.6, egui::Stroke::new(1.0, self.colors.constraint_marker_selected));
        }
        // Dimension preview while placing (not when editing an existing dimension)
        if (self.dim_placing || (self.dim_editing && self.dim_edit_did.is_none())) && self.dim_kind.is_some() {
            let kind = self.dim_kind.unwrap();
            let measured = self.measure_dimension(&kind);
            let is_radius = matches!(kind, DimensionKind::ArcRadius(_) | DimensionKind::ArcRadiusB(_));
            let preview_color = self.colors.dimension_preview;
            self.draw_dimension(&painter, &kind, measured, self.dim_offset, self.dim_text_along, preview_color, is_radius, false, false, false);
        }

        // Draw overlays ON TOP of canvas: preview line and cursor crosshair
        // Midpoint-snap visual: a filled "hover-point" dot with the
        // triangle-up glyph in the selected-constraint color, offset
        // perpendicular to the host entity so it sits on the OPPOSITE
        // side from where normal constraint markers live (keeps them
        // from stacking). Sized "hover / emphasized" (s=7, w=2) so
        // it's clearly a live UI hint vs. a permanent marker.
        //
        // `pt` is the actual snap point (constraint anchor); the
        // triangle draws at `pt + offset` on the opposite side from
        // where normal constraint markers live.
        let draw_midpoint_marker = |pt: egui::Pos2, offset: egui::Vec2| {
            let p = pt + offset;
            painter.circle_filled(pt, 4.0, self.colors.endpoint);
            let color = self.colors.constraint_marker_selected;
            let stroke = egui::Stroke::new(2.0, color);
            let s = 7.0_f32;
            let h = s * 1.56;
            let half_w = s * 1.04;
            let top = egui::Pos2::new(p.x, p.y - h * 0.5);
            let bl = egui::Pos2::new(p.x - half_w, p.y + h * 0.5);
            let br = egui::Pos2::new(p.x + half_w, p.y + h * 0.5);
            painter.line_segment([top, bl], stroke);
            painter.line_segment([bl, br], stroke);
            painter.line_segment([br, top], stroke);
        };

        // Compute the screen-space offset vector for the midpoint hint,
        // pointing to the OPPOSITE side from where line_marker_pos /
        // arc_marker_pos put normal constraint markers. Returns None if
        // the target isn't a midpoint variant.
        // Snap hint primitives. The X marks WHERE the endpoint will
        // land. The box signals "this is a specific point" -- drawn
        // alongside any marker whose snap target is a discrete point
        // (standalone point, line/arc endpoint, arc center, or a
        // line/arc midpoint). Body snaps skip the box.
        const SNAP_S: f32 = 9.75;
        let draw_snap_x = |pt: egui::Pos2| {
            let stroke = egui::Stroke::new(1.0, self.colors.constraint_marker_selected);
            painter.line_segment(
                [egui::Pos2::new(pt.x - SNAP_S, pt.y - SNAP_S),
                 egui::Pos2::new(pt.x + SNAP_S, pt.y + SNAP_S)], stroke);
            painter.line_segment(
                [egui::Pos2::new(pt.x - SNAP_S, pt.y + SNAP_S),
                 egui::Pos2::new(pt.x + SNAP_S, pt.y - SNAP_S)], stroke);
        };
        // Plain horizontal/vertical "+" cursor cross. Drawn during
        // tool mode at the live cursor location to make the active
        // placement point visible. Replaced by the snap-specific
        // marker (X / box / triangle / corner) whenever a snap fires.
        let draw_cursor_cross = |pt: egui::Pos2| {
            let stroke = egui::Stroke::new(1.0, self.colors.constraint_marker_selected);
            painter.line_segment(
                [egui::Pos2::new(pt.x - SNAP_S, pt.y),
                 egui::Pos2::new(pt.x + SNAP_S, pt.y)], stroke);
            painter.line_segment(
                [egui::Pos2::new(pt.x, pt.y - SNAP_S),
                 egui::Pos2::new(pt.x, pt.y + SNAP_S)], stroke);
        };
        let draw_snap_box = |pt: egui::Pos2| {
            let stroke = egui::Stroke::new(1.0, self.colors.constraint_marker_selected);
            let tl = egui::Pos2::new(pt.x - SNAP_S, pt.y - SNAP_S);
            let tr = egui::Pos2::new(pt.x + SNAP_S, pt.y - SNAP_S);
            let bl = egui::Pos2::new(pt.x - SNAP_S, pt.y + SNAP_S);
            let br = egui::Pos2::new(pt.x + SNAP_S, pt.y + SNAP_S);
            painter.line_segment([tl, tr], stroke);
            painter.line_segment([tr, br], stroke);
            painter.line_segment([br, bl], stroke);
            painter.line_segment([bl, tl], stroke);
        };

        let midpoint_hint_offset = |t: &SnapTarget| -> Option<egui::Vec2> {
            const OFFSET_PX: f32 = 12.0;
            match t {
                SnapTarget::LineMidpoint(r) => {
                    let l = &self.sketch.lines[*r];
                    let p1 = self.to_screen(l.p1.value);
                    let p2 = self.to_screen(l.p2.value);
                    let dx = p2.x - p1.x;
                    let dy = p2.y - p1.y;
                    let len = (dx * dx + dy * dy).sqrt().max(1.0);
                    let nx = -dy / len;
                    let ny = dx / len;
                    // Normal markers use sign that forces "up" (ny < 0);
                    // we want the opposite -> force "down" (ny > 0).
                    let sign = if ny > 0.0 { 1.0 } else { -1.0 };
                    Some(egui::vec2(nx * OFFSET_PX * sign, ny * OFFSET_PX * sign))
                }
                SnapTarget::ArcMidpoint(r) => {
                    let a = &self.sketch.arcs[*r];
                    let mid_t = (a.start_angle.value + a.end_angle.value) * 0.5;
                    let mp = arc_point_at(a, mid_t);
                    let center = self.to_screen(a.center.value);
                    let mps = self.to_screen(mp);
                    // Radial outward in screen space (normal markers sit
                    // INSIDE the curve; hint goes OUTSIDE).
                    let dx = mps.x - center.x;
                    let dy = mps.y - center.y;
                    let len = (dx * dx + dy * dy).sqrt().max(1.0);
                    Some(egui::vec2(dx / len * OFFSET_PX, dy / len * OFFSET_PX))
                }
                _ => None,
            }
        };

        // Dispatch a snap hint by target kind: midpoint variants get
        // the triangle marker (existing), body snaps get a thin X,
        // point-like snaps get an X framed by a square.
        let draw_snap_hint = |screen_pt: egui::Pos2, t: &SnapTarget| {
            match t {
                SnapTarget::LineMidpoint(_) | SnapTarget::ArcMidpoint(_) => {
                    if let Some(off) = midpoint_hint_offset(t) {
                        draw_midpoint_marker(screen_pt, off);
                    }
                    draw_snap_box(screen_pt);
                }
                SnapTarget::Line(_) | SnapTarget::ArcBody(_) => {
                    draw_snap_x(screen_pt);
                }
                SnapTarget::Point(_) | SnapTarget::LineP1(_) | SnapTarget::LineP2(_)
                | SnapTarget::ArcCenter(_) | SnapTarget::ArcStart(_) | SnapTarget::ArcEnd(_) => {
                    draw_snap_box(screen_pt);
                }
            }
        };

        if let Some(ref state) = self.line_draw {
            let p1 = self.to_screen(state.start.pos);
            // Honor end-point snap in the preview so the user sees
            // exactly where the endpoint will land. Suppress any snap
            // whose target position coincides with the segment's start
            // -- a zero-length line is rejected anyway, and snapping
            // the end back to the start just paints a misleading box.
            let end_snap = self.find_snap_target(mouse_sketch, hit_threshold)
                .filter(|(p, _)| {
                    let dx = p.x - state.start.pos.x;
                    let dy = p.y - state.start.pos.y;
                    dx * dx + dy * dy > 1e-12
                });
            // Perp snap is allowed alongside an end-line-body snap:
            // both project to lines, so combining them yields the
            // unique intersection of the perp axis and the snapped
            // line, and BOTH constraints fire at commit. For other
            // end snaps (point / endpoint / midpoint / arc) the
            // position is already fixed, so perp is suppressed.
            let perp_eligible = match end_snap {
                None => true,
                Some((_, SnapTarget::Line(_))) => true,
                _ => false,
            };
            // Any line passing through state.start.pos is a candidate --
            // the chained previous segment, a line whose body the
            // start was placed on, a coincident endpoint elsewhere.
            // Pick the host that gives the closest-to-exact 90.
            let perp = if perp_eligible {
                self.find_best_perp_host_at(state.start.pos, mouse_sketch, crate::PERP_SNAP_PX, None)
            } else { None };
            // When both end-line-body snap AND start-perp fire, the
            // endpoint sits at their intersection.
            let combined: Option<vect2d> = match (&end_snap, &perp) {
                (Some((_, SnapTarget::Line(other))), Some((host, _))) => {
                    let hl = &self.sketch.lines[*host];
                    let hdx = hl.p2.value.x - hl.p1.value.x;
                    let hdy = hl.p2.value.y - hl.p1.value.y;
                    // Skip if perp axis is parallel to the snapped
                    // line (no unique intersection -- the helper
                    // returns a midpoint fallback that misleads).
                    let ol = &self.sketch.lines[*other];
                    let odx = ol.p2.value.x - ol.p1.value.x;
                    let ody = ol.p2.value.y - ol.p1.value.y;
                    let cross = (-hdy) * ody - hdx * odx;
                    if cross.abs() < 1e-9 {
                        None
                    } else {
                        let perp_p2 = vect2d::new(state.start.pos.x - hdy, state.start.pos.y + hdx);
                        Some(line_line_intersection(state.start.pos, perp_p2, ol.p1.value, ol.p2.value))
                    }
                }
                _ => None,
            };
            // End-side perpendicular: when start-perp didn't decide
            // the position and the end snaps to a line whose angle
            // to the drawn direction is near 90, snap end to the
            // foot of the perpendicular dropped from start onto the
            // target line. Skip self-pairing (host == target).
            let end_perp: Option<(Ref<Line>, vect2d)> = if combined.is_none() {
                match end_snap {
                    Some((p_on_target, SnapTarget::Line(target))) => {
                        let host_ref = state.start.snap.and_then(EditorApp::perp_host_from_snap);
                        if host_ref == Some(target) {
                            None
                        } else {
                            let tl = &self.sketch.lines[target];
                            self.try_perp_end_snap(state.start.pos, tl.p1.value, tl.p2.value, p_on_target, crate::PERP_SNAP_PX)
                                .map(|foot| (target, foot))
                        }
                    }
                    _ => None,
                }
            } else { None };
            // Auto-collinear preview: pull end onto a host's
            // infinite line when the cursor is aligned with a
            // line passing through start.
            let collinear_preview = if !self.snap_disabled
                && combined.is_none()
                && end_perp.is_none()
                && end_snap.is_none()
                && perp.is_none()
            {
                self.find_best_collinear_host_at(state.start.pos, mouse_sketch, crate::PERP_SNAP_PX, None)
            } else { None };
            // Auto-H/V preview: when no stronger hint decides the
            // endpoint, pull it onto the nearest axis through the
            // start so the user sees the axis-aligned line they
            // will commit.
            let hv_preview = if !self.snap_disabled
                && combined.is_none()
                && end_perp.is_none()
                && end_snap.is_none()
                && perp.is_none()
                && collinear_preview.is_none()
            {
                hv_snap_from(state.start.pos, mouse_sketch, self.scale, crate::PERP_SNAP_PX)
            } else { None };
            let end_pt = match (combined, &end_perp, &end_snap, &perp, &collinear_preview, &hv_preview) {
                (Some(p), _, _, _, _, _) => self.to_screen(p),
                (_, Some((_, p)), _, _, _, _) => self.to_screen(*p),
                (_, _, Some((p, _)), _, _, _) => self.to_screen(*p),
                (_, _, None, Some((_, p)), _, _) => self.to_screen(*p),
                (_, _, _, _, Some((_, p)), _) => self.to_screen(*p),
                (_, _, _, _, _, Some((_, p))) => self.to_screen(*p),
                _ => mouse_screen,
            };
            painter.line_segment([p1, end_pt],
                egui::Stroke::new(1.5, self.colors.preview_line));
            painter.circle_filled(p1, 4.0, self.colors.endpoint);
            if let Some((host, p)) = collinear_preview {
                // Marker on the drawn line (side-offset from its
                // own midpoint), and on the host at its committed
                // marker position so both legs of the pair show.
                let end_screen = self.to_screen(p);
                let mx = (p1.x + end_screen.x) * 0.5;
                let my = (p1.y + end_screen.y) * 0.5;
                let dx = end_screen.x - p1.x;
                let dy = end_screen.y - p1.y;
                let len = (dx * dx + dy * dy).sqrt().max(1.0);
                let nx = -dy / len;
                let ny = dx / len;
                let sign = if ny > 0.0 { -1.0 } else { 1.0 };
                let off = 10.0f32;
                let pos = egui::Pos2::new(mx + nx * off * sign, my + ny * off * sign);
                draw_collinear_marker(&painter, pos,
                    self.colors.constraint_marker_selected);
                let host_pos = self.line_marker_pos(host, 10.0, 0.0);
                draw_collinear_marker(&painter, host_pos,
                    self.colors.constraint_marker_selected);
            }
            if let Some((horizontal, p)) = hv_preview {
                // Side-offset placement matching committed H/V
                // markers (see line_marker_pos).
                let end_screen = self.to_screen(p);
                let mx = (p1.x + end_screen.x) * 0.5;
                let my = (p1.y + end_screen.y) * 0.5;
                let dx = end_screen.x - p1.x;
                let dy = end_screen.y - p1.y;
                let len = (dx * dx + dy * dy).sqrt().max(1.0);
                let nx = -dy / len;
                let ny = dx / len;
                let sign = if ny > 0.0 { -1.0 } else { 1.0 };
                let off = 10.0f32;
                let pos = egui::Pos2::new(mx + nx * off * sign, my + ny * off * sign);
                draw_hv_marker(&painter, pos, horizontal,
                    self.colors.constraint_marker_selected);
            }

            // No start-snap marker during 2nd-click placement: the
            // user just placed it, and a marker that persists for
            // the entire move-to-end-point phase is noise. The perp
            // corner marker below still draws because it conveys
            // an actively-firing constraint at the moving endpoint.
            if let Some((_, t)) = &end_snap { draw_snap_hint(end_pt, t); }
            if end_snap.is_none() && perp.is_none() && end_perp.is_none() { draw_cursor_cross(end_pt); }
            if let Some((host, snapped)) = perp {
                let hl = &self.sketch.lines[host];
                let drawn_free = combined.unwrap_or(snapped);
                draw_perp_corner_marker(
                    &painter, p1,
                    hl.p1.value, hl.p2.value,
                    state.start.pos, drawn_free,
                    self.scale, self.colors.constraint_marker_selected,
                );
            }
            if let Some((target, foot)) = end_perp {
                let tl = &self.sketch.lines[target];
                draw_perp_corner_marker(
                    &painter, self.to_screen(foot),
                    tl.p1.value, tl.p2.value,
                    foot, state.start.pos,
                    self.scale, self.colors.constraint_marker_selected,
                );
            }
        } else if matches!(self.tool,
            Tool::DrawLine | Tool::DrawPoint
            | Tool::DrawCircle | Tool::DrawEllipse | Tool::DrawArc | Tool::DrawRect)
            && self.circle_draw.is_none() && self.arc_draw.is_none()
            && self.rect_draw.is_none() && self.ellipse_draw.is_none()
        {
            // Pre-first-click hint: snap marker if a target is in
            // range, otherwise the plain "+" cursor cross so the
            // user can still see the live placement point.
            match self.find_snap_target(mouse_sketch, hit_threshold) {
                Some((pos, t)) => draw_snap_hint(self.to_screen(pos), &t),
                None => draw_cursor_cross(mouse_screen),
            }
        }

        // Live snap preview during endpoint drag. update_drag has
        // already overridden drag_pt.pos to the snapped location;
        // this paints the visual marker so the user sees why.
        if let Some((pos, ref t)) = self.drag_snap_preview {
            draw_snap_hint(self.to_screen(pos), t);
        }
        // Auto-collinear hint during line-endpoint drag: render the
        // Collinear glyph on both the dragged line and the host so
        // the user sees which pair the release will tie together.
        if let Some((line, host)) = self.drag_collinear_hint {
            let p1 = self.line_marker_pos(line, 10.0, 0.0);
            let p2 = self.line_marker_pos(host, 10.0, 0.0);
            draw_collinear_marker(&painter, p1,
                self.colors.constraint_marker_selected);
            draw_collinear_marker(&painter, p2,
                self.colors.constraint_marker_selected);
        }
        // Auto-H/V hint during line-endpoint drag: render the H or
        // V glyph offset to the side of the line, matching the
        // placement of a committed H/V marker.
        if let Some((line, horizontal)) = self.drag_hv_hint {
            let pos = self.line_marker_pos(line, 10.0, 0.0);
            draw_hv_marker(&painter, pos, horizontal,
                self.colors.constraint_marker_selected);
        }
        // Auto-perpendicular hint during drag: corner marker at the
        // opposite (anchored) endpoint of the dragged line.
        if let Some((host, anchor)) = self.drag_perp_snap {
            // The dragged endpoint current position is already pulled
            // to the perpendicular projection by update_drag.
            let drag_pos = match self.grab {
                Some(GrabTarget::LineP1(r)) => self.sketch.lines[r].p1.value,
                Some(GrabTarget::LineP2(r)) => self.sketch.lines[r].p2.value,
                _ => anchor,
            };
            let hl = &self.sketch.lines[host];
            draw_perp_corner_marker(
                &painter, self.to_screen(anchor),
                hl.p1.value, hl.p2.value,
                anchor, drag_pos,
                self.scale, self.colors.constraint_marker_selected,
            );
        }

        // Circle preview: radius from the session (mouse, rim snap or
        // typed -- the tool handler refreshed it this frame).
        if let Some(ref state) = self.circle_draw {
            let c = state.center.pos;
            let center = self.to_screen(c);
            painter.circle_stroke(center, state.r as f32 * self.scale,
                egui::Stroke::new(1.5, self.colors.preview_line));
            painter.circle_filled(center, 4.0, self.colors.endpoint);
            if let Some(ref t) = state.center.snap { draw_snap_hint(center, t); }
            let edge_pt = self.to_screen(arael::vect::vect2d::new(
                c.x + state.dir.x * state.r, c.y + state.dir.y * state.r));
            match &state.live_snap {
                Some((p, t)) => draw_snap_hint(self.to_screen(*p), t),
                None => draw_cursor_cross(edge_pt),
            }
        }

        // Ellipse preview: center dot, axis segments, outline, and
        // the live length as a dimension readout. Direction and
        // untyped lengths were refreshed from the mouse by the tool
        // handler this frame.
        if let Some(ref state) = self.ellipse_draw {
            let c = state.center.pos;
            let center_pt = self.to_screen(c);
            painter.circle_filled(center_pt, 4.0, self.colors.endpoint);
            let dir = state.dir;
            let (rx, ry) = if state.axis_fixed {
                (state.rx, state.ry.max(1e-9))
            } else {
                // 2:1 proportion until the axis is fixed.
                (state.rx, state.rx * 0.5)
            };
            if rx > 1e-9 {
                let end = arael::vect::vect2d::new(c.x + dir.x * rx, c.y + dir.y * rx);
                let end_pt = self.to_screen(end);
                let stroke = egui::Stroke::new(1.5, self.colors.preview_line);
                painter.line_segment([center_pt, end_pt], stroke);
                if state.axis_fixed {
                    // Minor segment on the mouse's side of the axis.
                    let minor = arael::vect::vect2d::new(
                        c.x - dir.y * ry * state.ry_sign,
                        c.y + dir.x * ry * state.ry_sign,
                    );
                    painter.line_segment([center_pt, self.to_screen(minor)], stroke);
                    // Major length stays visible as a dim readout
                    // while the minor is being picked.
                    let mid = self.to_screen(arael::vect::vect2d::new(
                        c.x + dir.x * rx * 0.5,
                        c.y + dir.y * rx * 0.5,
                    ));
                    painter.text(
                        egui::Pos2::new(mid.x + 10.0, mid.y + 12.0),
                        egui::Align2::LEFT_TOP,
                        format!("{:.4}", rx),
                        egui::FontId::proportional(12.0),
                        self.colors.dimension_preview,
                    );
                }
                let n = 64;
                let pts: Vec<egui::Pos2> = (0..=n).map(|i| {
                    let t = i as f64 / n as f64 * std::f64::consts::TAU;
                    let (s, co) = t.sin_cos();
                    self.to_screen(arael::vect::vect2d::new(
                        c.x + dir.x * rx * co - dir.y * ry * s,
                        c.y + dir.y * rx * co + dir.x * ry * s,
                    ))
                }).collect();
                painter.add(egui::Shape::line(pts, stroke));
                // H/V marker beside the axis, line-preview convention.
                if !state.axis_fixed && let Some(horizontal) = state.hv {
                    let mx = (center_pt.x + end_pt.x) * 0.5;
                    let my = (center_pt.y + end_pt.y) * 0.5;
                    let dx = end_pt.x - center_pt.x;
                    let dy = end_pt.y - center_pt.y;
                    let len = (dx * dx + dy * dy).sqrt().max(1.0);
                    let nx = -dy / len;
                    let ny = dx / len;
                    let sign = if ny > 0.0 { -1.0 } else { 1.0 };
                    let off = 10.0f32;
                    let pos = egui::Pos2::new(mx + nx * off * sign, my + ny * off * sign);
                    draw_hv_marker(&painter, pos, horizontal,
                        self.colors.constraint_marker_selected);
                }
                // Aimed rim point: snap marker when a target took it
                // over, else the plain cursor cross on the axis end.
                match &state.live_snap {
                    Some((p, t)) => draw_snap_hint(self.to_screen(*p), t),
                    None => if !state.axis_fixed { draw_cursor_cross(end_pt); },
                }
            }
            if let Some(ref t) = state.center.snap { draw_snap_hint(center_pt, t); }
        }

        // Rect preview: sides and quadrant from the session (mouse,
        // corner snap or typed -- the tool handler refreshed them
        // this frame).
        if let Some(ref state) = self.rect_draw {
            let c = state.corner.pos;
            let c1 = self.to_screen(c);
            painter.circle_filled(c1, 4.0, self.colors.endpoint);
            if let Some(ref t) = state.corner.snap { draw_snap_hint(c1, t); }
            let opposite = arael::vect::vect2d::new(c.x + state.sx * state.w, c.y + state.sy * state.h);
            let c3 = self.to_screen(opposite);
            let c2 = egui::Pos2::new(c3.x, c1.y);
            let c4 = egui::Pos2::new(c1.x, c3.y);
            let stroke = egui::Stroke::new(1.5, self.colors.preview_line);
            painter.line_segment([c1, c2], stroke);
            painter.line_segment([c2, c3], stroke);
            painter.line_segment([c3, c4], stroke);
            painter.line_segment([c4, c1], stroke);
            match &state.live_snap {
                Some((p, t)) => draw_snap_hint(self.to_screen(*p), t),
                None => draw_cursor_cross(c3),
            }
        }

        // Arc preview
        if let Some(ref state) = self.arc_draw {
            let start_screen = self.to_screen(state.start.pos);
            painter.circle_filled(start_screen, 4.0, self.colors.endpoint);
            if let Some(ref t) = state.start.snap { draw_snap_hint(start_screen, t); }
            if let Some(PlacedPoint { pos: end, snap: ref snap_end }) = state.end {
                let end_screen = self.to_screen(end);
                painter.circle_filled(end_screen, 4.0, self.colors.endpoint);
                if let Some(t) = snap_end { draw_snap_hint(end_screen, t); }
                // Live snap for the mid-point click.
                let mid_snap = self.find_snap_target(mouse_sketch, hit_threshold)
                    .filter(|(p, _)| {
                        let d_s = (p.x - state.start.pos.x).powi(2) + (p.y - state.start.pos.y).powi(2);
                        let d_e = (p.x - end.x).powi(2) + (p.y - end.y).powi(2);
                        d_s > 1e-12 && d_e > 1e-12
                    });
                let mid_sketch = mid_snap.map_or(mouse_sketch, |(p, _)| p);
                // Preview arc through start, end, and mid (or mouse).
                if let Some((c, r, sa, ea, ccw)) = circumscribed_arc(state.start.pos, end, mid_sketch) {
                    let norm = |v: f64| -> f64 { let rv = v % std::f64::consts::TAU; if rv < 0.0 { rv + std::f64::consts::TAU } else { rv } };
                    let span = if ccw { norm(ea - sa) } else { -norm(sa - ea) };
                    let n_segs = 64usize;
                    let points: Vec<egui::Pos2> = (0..=n_segs).map(|i| {
                        let t = sa + span * (i as f64 / n_segs as f64);
                        self.to_screen(vect2d::new(c.x + r * t.cos(), c.y + r * t.sin()))
                    }).collect();
                    for w in points.windows(2) {
                        painter.line_segment([w[0], w[1]],
                            egui::Stroke::new(1.5, self.colors.preview_line));
                    }
                }
                if let Some((_, t)) = &mid_snap { draw_snap_hint(self.to_screen(mid_sketch), t); }
                if mid_snap.is_none() { draw_cursor_cross(self.to_screen(mid_sketch)); }
            } else {
                // Only start placed; draw line to mouse as hint.
                let end_snap = self.find_snap_target(mouse_sketch, hit_threshold)
                    .filter(|(p, _)| {
                        let dx = p.x - state.start.pos.x;
                        let dy = p.y - state.start.pos.y;
                        dx * dx + dy * dy > 1e-12
                    });
                let end_sketch = end_snap.map_or(mouse_sketch, |(p, _)| p);
                let end_pt = self.to_screen(end_sketch);
                painter.line_segment([start_screen, end_pt],
                    egui::Stroke::new(1.0, self.colors.preview_line));
                if let Some((_, t)) = &end_snap { draw_snap_hint(end_pt, t); }
                if end_snap.is_none() { draw_cursor_cross(end_pt); }
            }
        }

        // Box-select marquee: dashed rectangle between the press
        // origin and the current mouse. Drawn in selection colour
        // so it reads as a live selection overlay.
        if let Some(start) = self.box_select_start {
            let end = mouse_screen;
            let min = egui::Pos2::new(start.x.min(end.x), start.y.min(end.y));
            let max = egui::Pos2::new(start.x.max(end.x), start.y.max(end.y));
            let col = self.colors.constraint_marker_selected;
            let stroke = egui::Stroke::new(1.0, col);
            painter.line_segment([min, egui::Pos2::new(max.x, min.y)], stroke);
            painter.line_segment([egui::Pos2::new(max.x, min.y), max], stroke);
            painter.line_segment([max, egui::Pos2::new(min.x, max.y)], stroke);
            painter.line_segment([egui::Pos2::new(min.x, max.y), min], stroke);
            // Faint fill to make the zone obvious.
            let fill = egui::Color32::from_rgba_unmultiplied(col.r(), col.g(), col.b(), 32);
            painter.rect_filled(
                egui::Rect::from_min_max(min, max),
                0.0,
                fill,
            );
        }

        if self.snap_disabled {
            painter.text(
                egui::Pos2::new(rect.right() - 10.0, rect.top() + 10.0),
                egui::Align2::RIGHT_TOP,
                "SNAP OFF (Q / Cmd / Ctrl)",
                egui::FontId::proportional(14.0),
                self.colors.constraint_marker_selected,
            );
        }

        // Command cursor crosshair (full canvas lines)
        if let Some(pos) = self.command_cursor {
            let sp = self.to_screen(pos);
            let stroke = egui::Stroke::new(0.5, self.colors.command_cursor);
            painter.line_segment(
                [egui::Pos2::new(sp.x, rect.top()), egui::Pos2::new(sp.x, rect.bottom())], stroke);
            painter.line_segment(
                [egui::Pos2::new(rect.left(), sp.y), egui::Pos2::new(rect.right(), sp.y)], stroke);
        }

        // Status bar at bottom (hidden after first modification)
        if self.show_hints {
        let status = match self.tool {
            Tool::Select => "Select: click to select/deselect, drag entity to move, drag empty space for box-select (Shift to extend).",
            Tool::DrawPoint => "Point: click to place.",
            Tool::DrawLine => if self.line_draw.is_some() {
                "Line: click to place end point (chains next line). Escape to finish."
            } else {
                "Line: click to place start point. Snaps to nearby points/endpoints."
            },
            Tool::DrawCircle => if self.circle_draw.is_some() {
                "Circle: click to set the radius, or type it and press Enter. Escape cancels."
            } else {
                "Circle: click to place center. O switches to Ellipse."
            },
            Tool::DrawEllipse => if self.ellipse_draw.as_ref().is_some_and(|s| s.axis_fixed) {
                "Ellipse: click the minor extent, or type its length and press Enter. Escape cancels."
            } else if self.ellipse_draw.is_some() {
                "Ellipse: click the end of the major axis. Type the length, Tab to the angle; both typed, Enter fixes the axis. Snaps to H/V (hold Q to disable)."
            } else {
                "Ellipse: click to place center. O switches to Circle."
            },
            Tool::DrawArc => if let Some(ref s) = self.arc_draw {
                if s.end.is_some() {
                    "Arc: click a point on the arc."
                } else {
                    "Arc: click to place end point."
                }
            } else {
                "Arc: click to place start point."
            },
            Tool::DrawRect => if self.rect_draw.is_some() {
                "Rect: click the opposite corner, or type the width, Tab, the height and press Enter. Escape cancels."
            } else {
                "Rect: click to place first corner."
            },
            Tool::Fillet => if self.fillet_pending.is_some() {
                "Fillet: type radius and press Enter. Escape to cancel."
            } else {
                "Fillet: click a connecting endpoint, or select two lines. F switches to Chamfer."
            },
            Tool::Chamfer => if self.fillet_pending.is_some() {
                "Chamfer: type corner-to-end distance and press Enter. Escape to cancel."
            } else {
                "Chamfer: click a connecting endpoint, or select two lines. F switches to Fillet."
            },
            Tool::Split => "Split: click a line/arc to break it at the crossings around the click. B switches to Trim.",
            Tool::Trim => "Trim: click the span to remove (cut at the crossings around it). B switches to Split.",
            Tool::Scale => if self.scale_pending.is_some() {
                "Scale: type the factor and press Enter. Clicks adjust the set; double-click moves the center. Escape to cancel."
            } else if self.scale_center.is_some() {
                "Scale: click or drag a box over entities to include; the factor input opens once the set is non-empty."
            } else {
                "Scale: click or drag a box over entities to include, double-click a point to set the center."
            },
            Tool::ConstraintMode(_) => "Constraint: click entities to apply. Escape to cancel.",
            Tool::Dimension => if self.dim_editing {
                "Dimension: type value and press Enter. Escape to cancel."
            } else {
                "Dimension: click a line/arc, or two points. Escape to cancel."
            },
        };
        painter.text(
            egui::Pos2::new(rect.left() + 10.0, rect.bottom() - 20.0),
            egui::Align2::LEFT_CENTER,
            status,
            egui::FontId::proportional(12.0),
            self.colors.status_text,
        );
        } // show_hints

        // DOF + cost + version at bottom-right. During a drag the
        // cell's cache is invalid (the apparatus changed the
        // structure); show the pre-drag DOF from the drag-start rank
        // analysis instead of "..." -- the number the user cares
        // about doesn't change mid-gesture.
        let dof = self.sketch.cached_dof()
            .or_else(|| self.drag_rank.as_ref().map(|r| r.nullity));
        let dof_str = match dof {
            Some(0) => "DOF: 0 (fully constrained)".to_string(),
            Some(d) => format!("DOF: {}", d),
            None => "DOF: ...".to_string(),
        };
        let version_str = format!("arael v{}", env!("CARGO_PKG_VERSION"));
        let info = format!("{}  |  cost: {:.6}  |  ", dof_str, self.last_cost);
        let version_galley = painter.layout_no_wrap(
            version_str.clone(), egui::FontId::proportional(11.0), self.colors.status_text);
        let version_w = version_galley.size().x;
        painter.text(
            egui::Pos2::new(rect.right() - 10.0 - version_w, rect.bottom() - 20.0),
            egui::Align2::RIGHT_CENTER,
            info,
            egui::FontId::proportional(11.0),
            self.colors.status_text,
        );
        let version_rect = egui::Rect::from_min_size(
            egui::Pos2::new(rect.right() - 10.0 - version_w, rect.bottom() - 28.0),
            egui::Vec2::new(version_w, 16.0),
        );
        ui.put(version_rect, egui::Hyperlink::from_label_and_url(
            egui::RichText::new(version_str).size(11.0),
            "https://github.com/harakas/arael",
        ).open_in_new_tab(true));
    }

    /// Draw the split/trim hover span. The span endpoints come from
    /// the same bracketing the click will use, so the preview and the
    /// action agree by construction.
    fn draw_split_trim_preview(&self, painter: &egui::Painter, mouse_sketch: vect2d, hit_threshold: f64) {
        use arael_sketch_backend::split::SplitTarget;
        let Some((target, span)) = self.split_trim_preview(mouse_sketch, hit_threshold) else {
            return;
        };
        let trim = self.tool == Tool::Trim;
        // Split with nothing to cut at: no preview (the click errors).
        if span.is_none() && !trim {
            return;
        }
        let color = if trim { self.colors.trim_preview } else { self.colors.split_preview };
        let stroke = egui::Stroke::new(3.5, color);
        match target {
            SplitTarget::Line(r) => {
                let l = &self.sketch.lines[r];
                let (a, b) = match span {
                    Some((t0, t1)) => (
                        vect2d::new(
                            l.p1.value.x + t0 * (l.p2.value.x - l.p1.value.x),
                            l.p1.value.y + t0 * (l.p2.value.y - l.p1.value.y),
                        ),
                        vect2d::new(
                            l.p1.value.x + t1 * (l.p2.value.x - l.p1.value.x),
                            l.p1.value.y + t1 * (l.p2.value.y - l.p1.value.y),
                        ),
                    ),
                    None => (l.p1.value, l.p2.value),
                };
                painter.line_segment([self.to_screen(a), self.to_screen(b)], stroke);
            }
            SplitTarget::Arc(r) => {
                let arc = &self.sketch.arcs[r];
                let (t0, t1) = match span {
                    Some(s) => s,
                    None => {
                        let sa = arc.start_angle.value;
                        let ea = if arc.closed { sa + std::f64::consts::TAU } else { arc.end_angle.value };
                        (sa, ea)
                    }
                };
                let n = 48;
                let pts: Vec<egui::Pos2> = (0..=n)
                    .map(|i| {
                        let t = t0 + (t1 - t0) * (i as f64 / n as f64);
                        self.to_screen(arc.point_at(t))
                    })
                    .collect();
                for w in pts.windows(2) {
                    painter.line_segment([w[0], w[1]], stroke);
                }
            }
        }
    }
}
