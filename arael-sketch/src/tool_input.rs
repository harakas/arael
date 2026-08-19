// Per-tool pointer/keyboard input handlers, one method per tool arm
// of the canvas input match in update().

use eframe::egui;
use arael::utils::rad2rad;
use arael::vect::vect2d;
use arael::refs::Ref;
use crate::app_update::hv_snap_from;
use arael_sketch_solver::*;
use crate::tools::*;
use arael_sketch_backend::actions::Action;
use arael_sketch_backend::geometry::*;
use crate::EditorApp;

/// Dimension label placement computed from the mouse position.
pub(crate) struct DimPlacement {
    pub offset: vect2d,
    /// None when the dimension kind has no text_along notion
    /// (radius and concentric-distance labels).
    pub text_along: Option<f64>,
    /// Kind change from dynamic axis / sector picking; only produced
    /// by unlocked (placement preview) mode.
    pub new_kind: Option<DimensionKind>,
}

impl EditorApp {
    /// Compute a dimension's offset / text_along from the mouse.
    /// `locked` keeps the dimension's identity fixed (dragging an
    /// existing dim): the radius axis and the angle sector family
    /// stay as stored. Unlocked placement (preview before the
    /// confirming click) may switch ArcRadius <-> ArcRadiusB and the
    /// angle supplement, reported via `new_kind`, and clamps the
    /// angle-family text position into its sector.
    pub(crate) fn dim_placement_from_mouse(&self, kind: &DimensionKind, mouse: vect2d,
                                           locked: bool) -> Option<DimPlacement> {
        let place = |offset: vect2d, text_along: Option<f64>| {
            Some(DimPlacement { offset, text_along, new_kind: None })
        };
        match *kind {
            DimensionKind::ArcRadius(r) | DimensionKind::ArcRadiusB(r) => {
                if !self.sketch.arcs.contains_ref(r) { return None; }
                let a = &self.sketch.arcs[r];
                if a.is_ellipse {
                    let dx = mouse.x - a.center.value.x;
                    let dy = mouse.y - a.center.value.y;
                    let rot = a.rotation.value;
                    let mut is_b = matches!(kind, DimensionKind::ArcRadiusB(_));
                    if !locked {
                        // Project the mouse onto the axes: whichever is
                        // nearer picks radius vs radius_b.
                        let major = (dx * rot.cos() + dy * rot.sin()).abs();
                        let minor = (-dx * rot.sin() + dy * rot.cos()).abs();
                        is_b = minor > major;
                    }
                    // Projection onto the chosen axis picks the side.
                    let axis_angle = if is_b { rot + std::f64::consts::FRAC_PI_2 } else { rot };
                    let proj = dx * axis_angle.cos() + dy * axis_angle.sin();
                    let new_kind = (!locked).then(|| if is_b { DimensionKind::ArcRadiusB(r) }
                                                     else { DimensionKind::ArcRadius(r) });
                    Some(DimPlacement {
                        offset: vect2d::new(if proj >= 0.0 { 1.0 } else { -1.0 }, 0.0),
                        text_along: None,
                        new_kind,
                    })
                } else {
                    // Circle: free angle relative to start_angle.
                    let abs_angle = (mouse.y - a.center.value.y)
                        .atan2(mouse.x - a.center.value.x);
                    place(vect2d::new(abs_angle - a.start_angle.value, 0.0), None)
                }
            }
            DimensionKind::ArcSweep(r) => {
                let a = &self.sketch.arcs[r];
                let cx = a.center.value.x;
                let cy = a.center.value.y;
                // Parametric angle of the mouse in the ellipse's own
                // frame; the radial offset is measured against the
                // base curve point at that angle. Exact for circles,
                // and follows the annotation-curve family
                // draw_sweep_dimension renders for ellipses.
                let rot = a.rotation.value;
                let dx = mouse.x - cx;
                let dy = mouse.y - cy;
                let lx = dx * rot.cos() + dy * rot.sin();
                let ly = -dx * rot.sin() + dy * rot.cos();
                let mouse_angle = (ly / a.radius_b.value.max(1e-9))
                    .atan2(lx / a.radius.value.max(1e-9));
                let base = a.point_at(mouse_angle);
                let base_dist = ((base.x - cx).powi(2) + (base.y - cy).powi(2)).sqrt();
                let dist = (dx * dx + dy * dy).sqrt();
                let offset_y = dist - base_dist;
                let sa = a.start_angle.value;
                let sweep = a.end_angle.value - sa;
                let delta = rad2rad(mouse_angle - sa);
                let along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                place(vect2d::new(0.0, offset_y), Some(along))
            }
            DimensionKind::Angle(a, b, sup) => {
                let la = &self.sketch.lines[a];
                let lb = &self.sketch.lines[b];
                let ix = line_line_intersection(
                    la.p1.value, la.p2.value, lb.p1.value, lb.p2.value);
                let dist = ((mouse.x - ix.x).powi(2) + (mouse.y - ix.y).powi(2)).sqrt();
                let mouse_angle = (mouse.y - ix.y).atan2(mouse.x - ix.x);
                let (sector_mid, sup) = if locked {
                    // Drag keeps the stored sector family: lock to the
                    // 2 opposing sectors with the same supplement.
                    (self.angle_dim_opposing_sector(a, b, sup, mouse_angle), sup)
                } else {
                    // Placement roams all 4 sectors and picks the
                    // supplement from the mouse.
                    self.angle_dim_sector_from_mouse(a, b, mouse_angle)
                };
                let offset = vect2d::new(sector_mid, dist.max(0.3));
                let (_ix, start, sweep) = self.angle_dim_sector(a, b, sup, offset);
                let delta = rad2rad(mouse_angle - start);
                let mut along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                if !locked { along = along.clamp(-0.5, 0.5); }
                Some(DimPlacement {
                    offset,
                    text_along: Some(along),
                    new_kind: (!locked).then_some(DimensionKind::Angle(a, b, sup)),
                })
            }
            DimensionKind::LineAngle(r) => {
                let l = &self.sketch.lines[r];
                let p1 = l.p1.value;
                let line_angle = (l.p2.value.y - l.p1.value.y).atan2(l.p2.value.x - l.p1.value.x);
                let dist = ((mouse.x - p1.x).powi(2) + (mouse.y - p1.y).powi(2)).sqrt();
                let mouse_angle = (mouse.y - p1.y).atan2(mouse.x - p1.x);
                let sweep = line_angle;
                let delta = rad2rad(mouse_angle);
                let mut along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                if !locked { along = along.clamp(-0.5, 0.5); }
                place(vect2d::new(0.0, dist.max(0.3)), Some(along))
            }
            DimensionKind::ArcRotation(r) => {
                // Same sector placement as LineAngle, anchored at the
                // ellipse center and sweeping to the current rotation
                // angle of the major axis.
                let a = &self.sketch.arcs[r];
                let center = a.center.value;
                let rotation = a.rotation.value;
                let dist = ((mouse.x - center.x).powi(2) + (mouse.y - center.y).powi(2)).sqrt();
                let mouse_angle = (mouse.y - center.y).atan2(mouse.x - center.x);
                let sweep = rotation;
                let delta = rad2rad(mouse_angle);
                let mut along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                if !locked { along = along.clamp(-0.5, 0.5); }
                place(vect2d::new(0.0, dist.max(0.3)), Some(along))
            }
            DimensionKind::ConcentricDistance(a_ref, _b_ref) => {
                // 2D text anchor in world coords relative to the
                // shared center. The leader rotates to follow the
                // text; text renders parallel to the leader; an
                // extension line connects the outer arrow tip to the
                // text when the anchor sits past the outer radius.
                let center = self.sketch.arcs[a_ref].center.value;
                place(vect2d::new(mouse.x - center.x, mouse.y - center.y), None)
            }
            DimensionKind::HDistance(..) | DimensionKind::VDistance(..) => {
                let horizontal = matches!(kind, DimensionKind::HDistance(..));
                let (p1, p2) = self.dim_endpoints(kind);
                let mid = vect2d::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
                let offset_val = if horizontal {
                    mouse.y - mid.y
                } else {
                    mouse.x - mid.x
                };
                // text_along: project mouse onto the dimension line direction
                let (q1, q2) = if horizontal {
                    let y = mid.y + offset_val;
                    (vect2d::new(p1.x, y), vect2d::new(p2.x, y))
                } else {
                    let x = mid.x + offset_val;
                    (vect2d::new(x, p1.y), vect2d::new(x, p2.y))
                };
                let ddx = q2.x - q1.x;
                let ddy = q2.y - q1.y;
                let dlen = (ddx * ddx + ddy * ddy).sqrt().max(1e-12);
                let qmx = (q1.x + q2.x) / 2.0;
                let qmy = (q1.y + q2.y) / 2.0;
                let along = ((mouse.x - qmx) * ddx + (mouse.y - qmy) * ddy) / (dlen * dlen);
                place(vect2d::new(0.0, offset_val), Some(along))
            }
            _ => {
                // Decompose mouse into perpendicular offset and
                // along-line position.
                let (p1, p2) = self.dim_endpoints(kind);
                let ddx = p2.x - p1.x;
                let ddy = p2.y - p1.y;
                let len = (ddx * ddx + ddy * ddy).sqrt().max(1e-12);
                let ux = ddx / len;
                let uy = ddy / len;
                let nx = -ddy / len;
                let ny = ddx / len;
                let mx = (p1.x + p2.x) / 2.0;
                let my = (p1.y + p2.y) / 2.0;
                let rel_x = mouse.x - mx;
                let rel_y = mouse.y - my;
                let perp = rel_x * nx + rel_y * ny;
                let along = (rel_x * ux + rel_y * uy) / len;
                place(vect2d::new(0.0, perp), Some(along))
            }
        }
    }

    /// Canvas input for `Tool::Select`.
    pub(crate) fn handle_select_input(&mut self, ui: &egui::Ui, ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        // An Escape-cancelled gesture stays suppressed while the
        // pointer button is held; release re-arms dragging.
        if !ui.input(|i| i.pointer.primary_down()) {
            self.suppress_drag_regrab = false;
        }
        // Double-click on dimension to edit value (multi_clicked: a
        // preceding click would otherwise turn the pair into a
        // "triple" and swallow it)
        if Self::multi_clicked(response) {
            let mut edited = false;
            for dim in self.sketch.dimensions.iter() {
                let (ts, te) = self.dim_text_segment(dim);
                let d = Self::screen_point_to_segment_dist(mouse_screen, ts, te);
                if d < 15.0 {
                    self.dim_input = Self::dim_edit_string(dim);
                    self.dim_kind = Some(dim.kind);
                    self.dim_offset = dim.offset;
                    self.dim_edit_did = Some(dim.did);
                    self.dim_editing = true;
                    self.dim_select_all = true;
                    self.dim_placing = false;
                    self.dim_derived = dim.derived;
                    self.dim_derived_prev = dim.derived;
                    self.dim_input_backup.clear();
                    self.tool = Tool::Dimension;
                    self.selection.clear();
                    self.selection.push(Selection::Dimension(dim.did));
                    edited = true;
                    break;
                }
            }
            if !edited
                && let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold)
            {
                // Double-click on a line or arc: extend
                // selection with the whole connected chain
                // of lines + arcs, stopping at branches
                // and dead ends. Shift extends the existing
                // selection; otherwise we replace.
                if matches!(sel, Selection::Line(_) | Selection::Arc(_)) {
                    let additive = ui.input(|i| i.modifiers.shift);
                    if !additive { self.selection.clear(); }
                    self.select_chain(sel);
                }
            }
        }

        // Drag: geometry or dimension
        if response.dragged_by(egui::PointerButton::Primary) {
            if self.grab.is_none() && self.drag_dimension.is_none()
                && self.box_select_start.is_none() && !self.suppress_drag_regrab {
                // First drag frame: hit-test against the PRESS ORIGIN,
                // not the current cursor. egui's dragged_by only fires
                // after the pointer has moved past its internal
                // drag-start threshold (max_click_dist, a few px), so
                // the cursor has drifted outside the hover zone by the
                // time we get here. pointer.press_origin() returns the
                // screen position where the mouse button actually went
                // down -- what the user aimed at.
                let press_origin = ui.ctx().input(|i| i.pointer.press_origin());
                let press_sketch = press_origin
                    .map(|p| self.to_sketch(p))
                    .unwrap_or(mouse_sketch);
                let mut grabbed_dim = false;
                if let Some(sel) = self.hit_test_selection(press_sketch, hit_threshold)
                    && let Selection::Dimension(i) = sel {
                        self.drag_dimension = Some(i);
                        grabbed_dim = true;
                    }
                if !grabbed_dim
                    && let Some(target) = self.hit_test(press_sketch, hit_threshold) {
                        self.start_drag(target, press_sketch);
                    }
                // Nothing grabbable at the press origin --
                // enter classical box-select. The start
                // position lives in screen coords so the
                // rect tracks the viewport if the user pans
                // mid-drag (it doesn't, since Select drag
                // doesn't pan, but this also shields the
                // stored point from zoom changes).
                if !grabbed_dim && self.grab.is_none()
                    && let Some(p) = press_origin
                {
                    self.box_select_start = Some(p);
                }
            }
            if let Some(drag_did) = self.drag_dimension {
                // Update dimension offset and text_along from mouse;
                // locked placement keeps the dim's axis / sector family.
                if let Some(dim_idx) = self.sketch.dimension_index_by_did(drag_did) {
                    let kind = self.sketch.dimensions[dim_idx].kind;
                    if let Some(p) = self.dim_placement_from_mouse(&kind, mouse_sketch, true) {
                        self.sketch.mutate_values(|s| {
                            s.dimensions[dim_idx].offset = p.offset;
                            if let Some(along) = p.text_along {
                                s.dimensions[dim_idx].text_along = along;
                            }
                        });
                    }
                }
                ctx.request_repaint();
            }
            if self.grab.is_some() {
                self.update_drag(mouse_sketch, hit_threshold);
                ctx.request_repaint();
            }
            if self.box_select_start.is_some() {
                ctx.request_repaint();
            }
        }

        // End drag
        if response.drag_stopped_by(egui::PointerButton::Primary) {
            if self.grab.is_some() {
                self.end_drag(hit_threshold);
            }
            // A label drag mutated placement live; commit it
            // as one undoable MoveDimension.
            if let Some(did) = self.drag_dimension
                && let Some(i) = self.sketch.dimension_index_by_did(did) {
                    let offset = self.sketch.dimensions[i].offset;
                    let text_along = self.sketch.dimensions[i].text_along;
                    self.begin_group();
                    self.exec(Action::MoveDimension { did, offset, text_along });
            }
            self.drag_dimension = None;
            // Box-select completion: compute the rect in
            // sketch coords and add every entity that sits
            // inside or crosses it to the selection. Shift
            // extends the existing selection; without it
            // we replace. Empty rects (barely-moved mouse)
            // are ignored so a stray click doesn't wipe
            // the selection.
            if let Some(start) = self.box_select_start.take() {
                let end = mouse_screen;
                let dx = (end.x - start.x).abs();
                let dy = (end.y - start.y).abs();
                if dx >= 2.0 || dy >= 2.0 {
                    let additive = ui.input(|i| i.modifiers.shift);
                    self.apply_box_select(start, end, additive);
                }
            }
        }

        // Click (no drag): paste into command prompt if focused, else select/deselect.
        // egui fires `clicked_by` together with `double_clicked_by` on the
        // second release of a double-click -- skip the single-click path
        // in that case so the double-click chain-select isn't immediately
        // toggled off by the same event.
        if response.clicked_by(egui::PointerButton::Primary)
            && !Self::multi_clicked(response)
        {
            if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                if self.show_command && self.command_has_focus {
                    if let Some(name) = self.selection_command_name(&sel) {
                        // Insert at cursor position, not at end
                        let cmd_id = egui::Id::new("command_input");
                        let cursor_pos = egui::TextEdit::load_state(ui.ctx(), cmd_id)
                            .and_then(|s| s.cursor.char_range())
                            .map(|r| r.primary.index)
                            .unwrap_or(self.command_input.len());
                        let pos = cursor_pos.min(self.command_input.len());
                        // Add space before if needed
                        let need_space = pos > 0
                            && !self.command_input[..pos].ends_with(' ');
                        let insert = if need_space { format!(" {}", name) } else { name.clone() };
                        self.command_input.insert_str(pos, &insert);
                        // Move cursor after inserted text
                        let new_pos = pos + insert.len();
                        if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), cmd_id) {
                            let ccursor = egui::text::CCursor::new(new_pos);
                            state.cursor.set_char_range(Some(egui::text::CCursorRange::one(ccursor)));
                            egui::TextEdit::store_state(ui.ctx(), cmd_id, state);
                        }
                        self.command_focus = true; // re-focus after paste
                    }
                } else {
                    self.toggle_selection(sel);
                }
            } else {
                if !(self.show_command && self.command_has_focus) {
                    self.selection.clear();
                }
            }
        }
    }

    /// Canvas input for `Tool::DrawPoint`.
    pub(crate) fn handle_draw_point(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, _mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if response.clicked_by(egui::PointerButton::Primary) {
            self.begin_group();
            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
            let pos = snap.map_or(mouse_sketch, |(p, _)| p);
            let action = Action::AddPoint { pos };
            let new_point = self.exec(action).point();
            if let (Some((_, snap_target)), Some(new_point)) = (snap, new_point) {
                self.apply_snap_coincident_point(snap_target, new_point);
            }
        }
    }

    /// Canvas input for `Tool::DrawLine`.
    pub(crate) fn handle_draw_line(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, _mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        // Double-click terminates the line chain without adding a segment.
        if Self::multi_clicked(response) {
            self.line_draw = None;
        } else if response.clicked_by(egui::PointerButton::Primary) {
            self.begin_group();
            if let Some(state) = self.line_draw.take() {
                // Second click: finish line
                // Snap end point to nearby entity
                let end_snap = self.find_snap_target(mouse_sketch, hit_threshold);
                let mut end_pos = end_snap.map_or(mouse_sketch, |(pos, _)| pos);

                // Auto-perpendicular. Allowed alongside an
                // end-line-body snap (both project to lines,
                // so the unique intersection determines the
                // endpoint and BOTH constraints fire).
                // Suppressed when the end snap is point-like
                // or arc-like (position already fixed).
                let perp_eligible = matches!(end_snap, None | Some((_, SnapTarget::Line(_))));
                let perp_host = if perp_eligible {
                    self.find_best_perp_host_at(state.start.pos, mouse_sketch, crate::PERP_SNAP_PX, None)
                } else { None };
                // If both end-line snap and start-perp fire, place
                // the endpoint at their intersection.
                let mut combined_used = false;
                if let (Some((_, SnapTarget::Line(other))), Some((host, _))) = (end_snap, perp_host) {
                    let hl = &self.sketch.lines[host];
                    let hdx = hl.p2.value.x - hl.p1.value.x;
                    let hdy = hl.p2.value.y - hl.p1.value.y;
                    let ol = &self.sketch.lines[other];
                    let odx = ol.p2.value.x - ol.p1.value.x;
                    let ody = ol.p2.value.y - ol.p1.value.y;
                    let cross = (-hdy) * ody - hdx * odx;
                    if cross.abs() >= 1e-9 {
                        let perp_p2 = vect2d::new(state.start.pos.x - hdy, state.start.pos.y + hdx);
                        end_pos = line_line_intersection(state.start.pos, perp_p2, ol.p1.value, ol.p2.value);
                        combined_used = true;
                    }
                } else if let Some((_, p)) = perp_host {
                    end_pos = p;
                }
                // End-side perpendicular: snap end onto the
                // foot-of-perpendicular from start onto the
                // target line. Skipped when start-perp combine
                // already decided the position.
                let end_perp_target: Option<Ref<Line>> = if !combined_used {
                    match end_snap {
                        Some((p_on_target, SnapTarget::Line(target))) => {
                            let host_ref = state.start.snap.and_then(EditorApp::perp_host_from_snap);
                            if host_ref == Some(target) {
                                None
                            } else {
                                let tl = &self.sketch.lines[target];
                                self.try_perp_end_snap(state.start.pos, tl.p1.value, tl.p2.value, p_on_target, crate::PERP_SNAP_PX)
                                    .map(|foot| { end_pos = foot; target })
                            }
                        }
                        _ => None,
                    }
                } else { None };

                // Auto-collinear: pull end onto a host line
                // whose infinite extension passes through
                // start, when the cursor is close to that
                // line. Priority over auto-H/V because it's
                // the stronger structural relation.
                let collinear_host = if !self.snap_disabled
                    && !combined_used
                    && perp_host.is_none()
                    && end_perp_target.is_none()
                    && end_snap.is_none()
                {
                    self.find_best_collinear_host_at(state.start.pos, end_pos, crate::PERP_SNAP_PX, None)
                } else { None };
                if let Some((_, p)) = collinear_host { end_pos = p; }

                // Auto-tangent: a line started on an arc pulls onto
                // the arc's tangent there when the cursor is close
                // to it. Below collinear (a line-line relation is
                // the simpler statement of the same direction),
                // above H/V. Not with a point-like end snap (position
                // already fixed); an end snap on a line body meets
                // the tangent line at their intersection.
                let tangent_host = if !combined_used
                    && perp_host.is_none()
                    && end_perp_target.is_none()
                    && collinear_host.is_none()
                    && matches!(end_snap, None | Some((_, SnapTarget::Line(_))))
                {
                    self.line_tangent_snap(state.start.snap, state.start.pos, mouse_sketch, crate::PERP_SNAP_PX)
                } else { None };
                if let Some((_, p)) = tangent_host {
                    end_pos = match end_snap {
                        Some((_, SnapTarget::Line(other))) => {
                            let ol = &self.sketch.lines[other];
                            let (odx, ody) = (ol.p2.value.x - ol.p1.value.x, ol.p2.value.y - ol.p1.value.y);
                            let (tdx, tdy) = (p.x - state.start.pos.x, p.y - state.start.pos.y);
                            if (tdx * ody - tdy * odx).abs() >= 1e-9 {
                                line_line_intersection(state.start.pos, p, ol.p1.value, ol.p2.value)
                            } else { p }
                        }
                        _ => p,
                    };
                }

                // Auto-horizontal/vertical snap: only when no
                // stronger placement constraint has fired
                // (end-snap position, start-perp combined,
                // start-perp free, or end-perp, or collinear, or
                // tangent). Otherwise the drawn line already has
                // its angle determined by the perp/snap and H/V
                // would conflict.
                let hv = if !self.snap_disabled
                    && !combined_used
                    && perp_host.is_none()
                    && end_perp_target.is_none()
                    && end_snap.is_none()
                    && collinear_host.is_none()
                    && tangent_host.is_none()
                {
                    hv_snap_from(state.start.pos, end_pos, self.scale, crate::PERP_SNAP_PX)
                } else { None };
                if let Some((_, p)) = hv { end_pos = p; }

                // Reject zero-length lines
                let dx = end_pos.x - state.start.pos.x;
                let dy = end_pos.y - state.start.pos.y;
                if dx * dx + dy * dy < 1e-6 {
                    // Put state back, ignore this click
                    self.line_draw = Some(state);
                } else {

                let action = Action::AddLine { p1: state.start.pos, p2: end_pos };
                // A rejected creation ends the gesture but
                // must not return early: the rest of the
                // frame still renders (the rejection is in
                // status_error).
                if let Some(new_line) = self.exec(action).line() {

                // Auto-coincident for start snap
                if let Some(snap) = state.start.snap {
                    self.apply_snap_coincident(snap, new_line, true);
                }
                // Auto-coincident for end snap
                if let Some((_, snap)) = end_snap {
                    self.apply_snap_coincident(snap, new_line, false);
                }
                // Auto-perpendicular constraint, same undo group.
                if let Some((host, _)) = perp_host {
                    let action = Action::ApplyPerpendicular { a: new_line, b: host };
                    if arael_sketch_backend::conflicts::validate_action(&self.sketch, &action).is_none() {
                        self.exec(action);
                    }
                }
                if let Some(target) = end_perp_target {
                    let action = Action::ApplyPerpendicular { a: new_line, b: target };
                    if arael_sketch_backend::conflicts::validate_action(&self.sketch, &action).is_none() {
                        self.exec(action);
                    }
                }
                // Auto-collinear constraint emission.
                if let Some((host, _)) = collinear_host {
                    let action = Action::ApplyCollinear { a: new_line, b: host };
                    if arael_sketch_backend::conflicts::validate_action(&self.sketch, &action).is_none() {
                        self.exec(action);
                    }
                }
                // Auto-tangent constraint emission.
                if let Some((arc, _)) = tangent_host {
                    let action = Action::ApplyTangentLA { line: new_line, arc };
                    if arael_sketch_backend::conflicts::validate_action(&self.sketch, &action).is_none() {
                        self.exec(action);
                    }
                }
                // Auto-H/V constraint emission in the same
                // undo group as the AddLine.
                if let Some((horizontal, _)) = hv {
                    let action = if horizontal {
                        Action::ApplyHorizontal { lines: vec![new_line] }
                    } else {
                        Action::ApplyVertical { lines: vec![new_line] }
                    };
                    if arael_sketch_backend::conflicts::validate_action(&self.sketch, &action).is_none() {
                        self.exec(action);
                    }
                }

                // Chain: start next line from end of this one
                self.line_draw = Some(LineDrawState {
                    start: PlacedPoint { pos: end_pos, snap: Some(SnapTarget::LineP2(new_line)) },
                    chained: true,
                });
                } // end if let (line created)
                } // end else (non-zero length)
            } else {
                // First click: start line, snap to nearby entity
                let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                let start_pos = snap.map_or(mouse_sketch, |(pos, _)| pos);
                self.line_draw = Some(LineDrawState {
                    start: PlacedPoint { pos: start_pos, snap: snap.map(|(_, t)| t) },
                    chained: false,
                });
            }
        }
    }

    /// Canvas input for `Tool::DrawCircle`.
    pub(crate) fn handle_draw_circle(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        // Live tracking: the radius follows the mouse (or a rim snap)
        // until typed; the input mirrors it fully selected. A snap is
        // offered only while the radius is not typed, so a snapped
        // point is always one the rim passes through.
        if self.circle_draw.is_some() {
            let (c, typed) = {
                let s = self.circle_draw.as_ref().unwrap();
                (s.center.pos, s.typed_r.is_some())
            };
            let snap = if typed { None } else {
                self.find_snap_target(mouse_sketch, hit_threshold).filter(|(p, _)| {
                    let dx = p.x - c.x;
                    let dy = p.y - c.y;
                    dx * dx + dy * dy > 1e-12
                })
            };
            let state = self.circle_draw.as_mut().unwrap();
            state.cursor = mouse_screen;
            state.live_snap = None;
            let edge = snap.map_or(mouse_sketch, |(p, _)| p);
            let dx = edge.x - c.x;
            let dy = edge.y - c.y;
            let len = (dx * dx + dy * dy).sqrt();
            if len > 1e-9 {
                state.dir = vect2d::new(dx / len, dy / len);
                if !typed {
                    state.r = len;
                    state.live_snap = snap;
                    self.dim_input = format!("{:.4}", len);
                    self.dim_select_all = true;
                }
            }
        }

        if !response.clicked_by(egui::PointerButton::Primary) {
            return;
        }
        if self.circle_draw.is_none() {
            // First click: center. Opens the value overlay.
            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
            let center = snap.map_or(mouse_sketch, |(p, _)| p);
            self.circle_draw = Some(CircleDrawState {
                center: PlacedPoint { pos: center, snap: snap.map(|(_, t)| t) },
                r: 0.0,
                typed_r: None,
                dir: vect2d::new(1.0, 0.0),
                live_snap: None,
                cursor: mouse_screen,
            });
            self.open_creation_input();
        } else if self.circle_draw.as_ref().unwrap().typed_r.is_some()
            || self.circle_draw.as_ref().unwrap().r > 1e-9
        {
            // Second click: edge point. Typed text always goes
            // through complete_circle so a broken value is reported.
            self.complete_circle();
        }
    }

    /// Create the circle from the session values, add a radius dim
    /// for a typed radius, and end the session. Called by the edge
    /// click and by Enter with a typed radius. Typed text that does
    /// not evaluate keeps the session open with a status error.
    pub(crate) fn complete_circle(&mut self) -> bool {
        let Some(s) = self.circle_draw.as_ref() else { return false };
        let typed = match s.typed_r.clone() {
            None => None,
            Some(raw) => match self.parse_axis_input(&raw) {
                Some(parsed) => Some(parsed),
                None => {
                    self.status_error = Some(format!("Radius: invalid value or expression: {}", raw));
                    return false;
                }
            },
        };
        let state = self.circle_draw.take().unwrap();
        let center = state.center.pos;
        let edge = vect2d::new(center.x + state.dir.x * state.r, center.y + state.dir.y * state.r);
        self.begin_group();
        // Rejected creation (zero radius) ends the gesture.
        let Some(arc) = self.exec(Action::AddCircle { center, edge }).arc() else {
            self.close_creation_input();
            return false;
        };
        if let Some(sn) = state.center.snap {
            self.apply_snap_coincident_arc(sn, arc, ArcPoint::Center, center);
        }
        // Rim snap: helper on the circle, tied to the target.
        if let Some((pos, target)) = state.live_snap
            && let Some(helper) = self.exec(Action::AddHelperPoint { pos }).point()
        {
            self.exec(Action::ApplyPointOnArc { point: helper, arc });
            self.apply_snap_coincident_point(target, helper);
        }
        if let Some((value, expr)) = typed {
            let value = if expr.is_some() { 0.0 } else { value };
            self.exec(Action::AddDimension {
                kind: DimensionKind::ArcRadius(arc), value, expr, derived: false, range: None,
            });
        }
        self.close_creation_input();
        true
    }

    /// Drop an in-progress circle gesture and close its value overlay.
    pub(crate) fn cancel_circle_session(&mut self) {
        if self.circle_draw.take().is_some() {
            self.close_creation_input();
        }
    }

    /// Open the value overlay for a creation session (circle,
    /// ellipse, rect): it live-tracks the first value from the mouse.
    fn open_creation_input(&mut self) {
        self.dim_editing = true;
        self.dim_kind = None;
        self.dim_edit_did = None;
        self.dim_derived = false;
        self.dim_input.clear();
        self.dim_select_all = true;
    }

    fn close_creation_input(&mut self) {
        self.dim_editing = false;
        self.dim_kind = None;
        self.dim_edit_did = None;
        self.dim_input.clear();
    }

    /// Whether a creation session (circle, ellipse, rect) owns the
    /// value overlay, and where its cursor last was.
    pub(crate) fn creation_session_cursor(&self) -> Option<egui::Pos2> {
        self.circle_draw.as_ref().map(|s| s.cursor)
            .or(self.ellipse_draw.as_ref().map(|s| s.cursor))
            .or(self.rect_draw.as_ref().map(|s| s.cursor))
            // The arc's radius input exists once the chord does.
            .or(self.arc_draw.as_ref().filter(|s| s.end.is_some()).map(|s| s.cursor))
    }

    /// Canvas input for `Tool::DrawEllipse`: click 1 center, click 2
    /// major-axis direction (H/V snapped unless disabled), click 3
    /// minor extent. The value overlay is live from the center click:
    /// it tracks the length under the mouse until the user types,
    /// which fixes that axis. Typed lengths become driving dims on
    /// commit (end_ellipse_session).
    pub(crate) fn handle_draw_ellipse(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        // Per-frame live tracking: direction and untyped lengths
        // follow the mouse; the input mirrors the active length and
        // stays fully selected so a keystroke replaces it. A snap
        // target under the mouse (points, endpoints, line/arc bodies)
        // takes over the aimed rim point -- offered only while that
        // axis length is not typed, so a snapped point is always one
        // the rim passes through and a constraint can honour.
        if self.ellipse_draw.is_some() {
            let (c, axis_fixed, dir, rx, typed_rx, typed_ry, typed_angle) = {
                let s = self.ellipse_draw.as_ref().unwrap();
                (s.center.pos, s.axis_fixed, s.dir, s.rx, s.typed_rx.is_some(),
                 s.typed_ry.is_some(), s.typed_angle.is_some())
            };
            // A typed angle fixes the direction, so a snapped point
            // could not be on the rim either: no snap then.
            let snap_ok = if axis_fixed { !typed_ry } else { !typed_rx && !typed_angle };
            let snap = if snap_ok {
                self.find_snap_target(mouse_sketch, hit_threshold).filter(|(p, _)| {
                    let dx = p.x - c.x;
                    let dy = p.y - c.y;
                    dx * dx + dy * dy > 1e-12
                })
            } else { None };
            let hv = if !axis_fixed && !typed_angle && snap.is_none() && !self.snap_disabled {
                hv_snap_from(c, mouse_sketch, self.scale, crate::PERP_SNAP_PX)
            } else { None };
            let scale = self.scale;
            let state = self.ellipse_draw.as_mut().unwrap();
            state.cursor = mouse_screen;
            state.live_snap = None;
            if !axis_fixed && typed_angle {
                // Direction is typed: the mouse only picks the length,
                // by projection onto the fixed axis.
                state.hv = None;
                let proj = ((mouse_sketch.x - c.x) * dir.x + (mouse_sketch.y - c.y) * dir.y).abs();
                if !typed_rx && proj > 1e-6 {
                    state.rx = proj;
                    self.dim_input = format!("{:.4}", proj);
                    self.dim_select_all = true;
                }
            } else if !axis_fixed {
                let end = snap.map(|(p, _)| p).or(hv.map(|(_, p)| p)).unwrap_or(mouse_sketch);
                let dx = end.x - c.x;
                let dy = end.y - c.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len > 1e-6 {
                    state.dir = vect2d::new(dx / len, dy / len);
                    state.hv = hv.map(|(h, _)| h);
                    state.angle_text = format!("{:.2}", dy.atan2(dx).to_degrees());
                    if !typed_rx {
                        state.rx = len;
                        state.live_snap = snap;
                        self.dim_input = format!("{:.4}", len);
                        self.dim_select_all = true;
                    }
                }
            } else {
                // Rim snap: the semi-minor that puts the ellipse
                // through the snapped point (needs it strictly inside
                // the major span); otherwise the perpendicular
                // distance of the mouse.
                let frame = |p: vect2d| (
                    (p.x - c.x) * dir.x + (p.y - c.y) * dir.y,
                    (p.x - c.x) * (-dir.y) + (p.y - c.y) * dir.x,
                );
                let snapped = snap.and_then(|(p, t)| {
                    let (u, v) = frame(p);
                    let k = 1.0 - (u / rx) * (u / rx);
                    (k > 1e-9 && v.abs() > 1e-9).then(|| (v.abs() / k.sqrt(), v.signum(), (p, t)))
                });
                let (ry, sign, live) = match snapped {
                    Some((ry, sign, s)) => (Some(ry), sign, Some(s)),
                    None => {
                        let (_, v) = frame(mouse_sketch);
                        if v.abs() > 1e-6 { (Some(v.abs()), v.signum(), None) } else { (None, state.ry_sign, None) }
                    }
                };
                let _ = scale;
                state.ry_sign = sign;
                if let Some(ry) = ry && !typed_ry {
                    state.ry = ry;
                    state.live_snap = live;
                    self.dim_input = format!("{:.4}", ry);
                    self.dim_select_all = true;
                }
            }
        }

        if !response.clicked_by(egui::PointerButton::Primary) {
            return;
        }
        enum Phase { Center, Axis, Minor }
        let phase = match &self.ellipse_draw {
            None => Phase::Center,
            Some(s) if !s.axis_fixed => Phase::Axis,
            Some(_) => Phase::Minor,
        };
        match phase {
            Phase::Center => self.start_ellipse(mouse_sketch, hit_threshold),
            Phase::Axis => {
                // Direction needs the mouse off-center unless the
                // angle is typed; the live tracking above already
                // refreshed dir/rx this frame.
                let state = self.ellipse_draw.as_ref().unwrap();
                let c = state.center.pos;
                let dx = mouse_sketch.x - c.x;
                let dy = mouse_sketch.y - c.y;
                if dx * dx + dy * dy < 1e-12 && state.typed_angle.is_none() {
                    return;
                }
                self.fix_ellipse_axis();
            }
            Phase::Minor => {
                // The minor was already live-editable while aiming,
                // so the click completes the session outright. A
                // click on the axis line itself (no size yet) waits;
                // typed text always goes through complete_ellipse so
                // a broken value is reported, not silently ignored.
                let s = self.ellipse_draw.as_mut().unwrap();
                if s.typed_ry.is_some() || s.ry >= 1e-6 {
                    if let Some(snap) = s.live_snap.take() {
                        s.rim_snaps.push(snap);
                    }
                    self.complete_ellipse();
                }
            }
        }
    }

    /// Fix the major axis from the current session values (second
    /// click, or Enter with both length and angle typed). Typed text
    /// that does not evaluate keeps the session on the axis with a
    /// status error; a zero length keeps waiting. Returns whether the
    /// axis was fixed.
    pub(crate) fn fix_ellipse_axis(&mut self) -> bool {
        let Some(s) = self.ellipse_draw.as_ref() else { return false };
        for (raw, what, angle) in [(s.typed_rx.clone(), "Semi-major", false), (s.typed_angle.clone(), "Angle", true)] {
            if let Some(raw) = raw && self.parse_typed_input(&raw, !angle).is_none() {
                self.status_error = Some(format!("{}: invalid value or expression: {}", what, raw));
                return false;
            }
        }
        let state = self.ellipse_draw.as_mut().unwrap();
        if state.rx < 1e-6 { return false; }
        state.axis_fixed = true;
        if let Some(snap) = state.live_snap.take() {
            state.rim_snaps.push(snap);
        }
        // Overlay switches to the semi-minor length; the next frame's
        // live tracking fills it.
        self.dim_input.clear();
        self.dim_select_all = true;
        true
    }

    fn start_ellipse(&mut self, mouse_sketch: vect2d, hit_threshold: f64) {
        let snap = self.find_snap_target(mouse_sketch, hit_threshold);
        let center = snap.map_or(mouse_sketch, |(p, _)| p);
        self.ellipse_draw = Some(EllipseDrawState {
            center: PlacedPoint { pos: center, snap: snap.map(|(_, t)| t) },
            dir: vect2d::new(1.0, 0.0),
            hv: None,
            axis_fixed: false,
            rx: 0.0,
            typed_rx: None,
            angle_text: String::new(),
            typed_angle: None,
            ry: 0.0,
            ry_sign: 1.0,
            typed_ry: None,
            cursor: egui::Pos2::ZERO,
            live_snap: None,
            rim_snaps: Vec::new(),
        });
        self.open_creation_input();
    }

    /// Parse a typed ellipse value: a plain number or `=expr` snapshot
    /// gives (value, None); a live expression gives (value, Some(expr)).
    /// None while the text does not (yet) evaluate to a finite value
    /// -- positive when `positive` (lengths), any sign otherwise
    /// (the axis angle, degrees).
    pub(crate) fn parse_typed_input(&self, raw: &str, positive: bool) -> Option<(f64, Option<String>)> {
        let raw = raw.trim();
        if raw.is_empty() { return None; }
        let (text, live) = match raw.strip_prefix('=') {
            Some(rest) => (rest.trim(), false),
            None => (raw, raw.parse::<f64>().is_err()),
        };
        let v = arael_sketch_backend::commands::eval_expr(&self.sketch, text).ok()?;
        if !v.is_finite() || (positive && v <= 0.0) { return None; }
        Some((v, live.then(|| text.to_string())))
    }

    /// Typed axis length (semi-major / semi-minor): see parse_typed_input.
    pub(crate) fn parse_axis_input(&self, raw: &str) -> Option<(f64, Option<String>)> {
        self.parse_typed_input(raw, true)
    }

    /// Create the ellipse from the current session values (axis
    /// fixed, ry set), add dims for the typed values in the same undo
    /// group -- radius / radius_b for typed lengths, xangle for a
    /// typed axis angle -- and end the session. Called by the minor
    /// click and by Enter with a typed minor. Typed text that does
    /// not evaluate keeps the session open with a status error.
    /// Returns false when nothing was created.
    pub(crate) fn complete_ellipse(&mut self) -> bool {
        let Some(s) = self.ellipse_draw.as_ref() else { return false };
        type KindOf = fn(Ref<Arc>) -> DimensionKind;
        let typed: [(Option<String>, &str, KindOf, bool); 3] = [
            (s.typed_rx.clone(), "Semi-major", DimensionKind::ArcRadius, true),
            (s.typed_ry.clone(), "Semi-minor", DimensionKind::ArcRadiusB, true),
            (s.typed_angle.clone(), "Angle", DimensionKind::ArcRotation, false),
        ];
        let mut dims = Vec::new();
        for (raw, what, kind_of, positive) in typed {
            let Some(raw) = raw else { continue };
            match self.parse_typed_input(&raw, positive) {
                Some(parsed) => dims.push((kind_of, parsed)),
                None => {
                    self.status_error = Some(format!("{}: invalid value or expression: {}", what, raw));
                    return false;
                }
            }
        }
        let state = self.ellipse_draw.take().unwrap();
        let center = state.center.pos;
        let rotation = state.dir.y.atan2(state.dir.x);
        self.begin_group();
        let created = self.exec(Action::AddEllipse {
            center, rx: state.rx, ry: state.ry, rotation,
        }).arc();
        let Some(arc) = created else {
            // Rejected creation ends the gesture.
            self.close_creation_input();
            return false;
        };
        if let Some(s) = state.center.snap {
            self.apply_snap_coincident_arc(s, arc, ArcPoint::Center, center);
        }
        // Snapped rim points: helper on the ellipse, tied to the
        // target (circle-tool edge-snap pattern).
        for (pos, target) in state.rim_snaps {
            if let Some(helper) = self.exec(Action::AddHelperPoint { pos }).point() {
                self.exec(Action::ApplyPointOnArc { point: helper, arc });
                self.apply_snap_coincident_point(target, helper);
            }
        }
        for (kind_of, (value, expr)) in dims {
            let value = if expr.is_some() { 0.0 } else { value };
            self.exec(Action::AddDimension { kind: kind_of(arc), value, expr, derived: false, range: None });
        }
        self.close_creation_input();
        true
    }

    /// Drop an in-progress ellipse gesture and close its value
    /// overlay. Nothing is committed -- creation happens only in
    /// complete_ellipse. Safe to call in any state.
    pub(crate) fn cancel_ellipse_session(&mut self) {
        if self.ellipse_draw.take().is_some() {
            self.close_creation_input();
        }
    }

    /// Canvas input for `Tool::DrawArc`.
    pub(crate) fn handle_draw_arc(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        // Live tracking once the chord exists: resolve the third
        // point in priority -- typed radius (mouse picks side and
        // minor/major), point snap, tangent snap at a connected end,
        // plain mouse -- and mirror the radius in the input.
        if let Some(true) = self.arc_draw.as_ref().map(|s| s.end.is_some()) {
            let (s, e, start_snap, end_snap, typed, prev_side) = {
                let st = self.arc_draw.as_ref().unwrap();
                let end = st.end.unwrap();
                (st.start.pos, end.pos, st.start.snap, end.snap, st.typed_r.is_some(), st.side)
            };
            let snap = if typed { None } else {
                self.find_snap_target(mouse_sketch, hit_threshold).filter(|(p, _)| {
                    (p.x - s.x).powi(2) + (p.y - s.y).powi(2) > 1e-12
                        && (p.x - e.x).powi(2) + (p.y - e.y).powi(2) > 1e-12
                })
            };
            let tangent = if !typed && snap.is_none() {
                self.arc_tangent_snap(s, start_snap, e, end_snap, mouse_sketch, crate::PERP_SNAP_PX)
            } else { None };
            let chord = vect2d::new(e.x - s.x, e.y - s.y);
            let cross = chord.x * (mouse_sketch.y - s.y) - chord.y * (mouse_sketch.x - s.x);
            let side = if cross.abs() > 1e-12 { cross.signum() } else { prev_side };
            let state = self.arc_draw.as_mut().unwrap();
            state.cursor = mouse_screen;
            state.side = side;
            state.live_snap = None;
            state.tangent = None;
            if typed {
                // Radius fixed: bulge toward the mouse's side; the
                // minor arc unless the mouse is farther than r from
                // the chord. Below half the chord there is no arc --
                // leave the third point on the chord (no preview,
                // completion reports it).
                let len = (chord.x * chord.x + chord.y * chord.y).sqrt();
                let mid_c = vect2d::new((s.x + e.x) * 0.5, (s.y + e.y) * 0.5);
                if len > 1e-12 && state.r >= len * 0.5 {
                    let n = vect2d::new(-chord.y / len * side, chord.x / len * side);
                    let d = (state.r * state.r - (len * 0.5) * (len * 0.5)).max(0.0).sqrt();
                    let h = ((mouse_sketch.x - s.x) * n.x + (mouse_sketch.y - s.y) * n.y).abs();
                    let sag = if h > state.r { state.r + d } else { state.r - d };
                    state.mid = vect2d::new(mid_c.x + n.x * sag, mid_c.y + n.y * sag);
                } else {
                    state.mid = mid_c;
                }
            } else {
                state.mid = snap.map(|(p, _)| p)
                    .or(tangent.map(|(_, _, m)| m))
                    .unwrap_or(mouse_sketch);
                state.live_snap = snap;
                state.tangent = tangent.map(|(h, a, _)| (h, a));
                if let Some((_, r, _, _, _)) = circumscribed_arc(s, e, state.mid) {
                    state.r = r;
                    self.dim_input = format!("{:.4}", r);
                    self.dim_select_all = true;
                }
            }
        }

        if !response.clicked_by(egui::PointerButton::Primary) {
            return;
        }
        let snap = self.find_snap_target(mouse_sketch, hit_threshold);
        let pos = snap.map_or(mouse_sketch, |(p, _)| p);
        let snap_target = snap.map(|(_, t)| t);
        match self.arc_draw.as_mut() {
            None => {
                // First click: start point.
                self.arc_draw = Some(ArcDrawState {
                    start: PlacedPoint { pos, snap: snap_target },
                    end: None,
                    mid: pos,
                    r: 0.0,
                    typed_r: None,
                    live_snap: None,
                    tangent: None,
                    side: 1.0,
                    cursor: mouse_screen,
                });
            }
            Some(state) if state.end.is_none() => {
                // Second click: end point. Opens the radius input.
                state.end = Some(PlacedPoint { pos, snap: snap_target });
                self.open_creation_input();
            }
            Some(_) => {
                // Third click: point on the arc (resolved above).
                self.complete_arc();
            }
        }
    }

    /// Arc tool tangent snap: the arc through both chord ends that is
    /// tangent to what an end is connected to (line direction, or the
    /// host arc's tangent there). Offered while the cursor is within
    /// `threshold_px` of that circle; the nearer of the start-side and
    /// end-side candidates wins. Returns the host, the connected end,
    /// and the cursor's projection onto the tangent circle (a third
    /// point that yields exactly that circle).
    fn arc_tangent_snap(
        &self,
        s: vect2d, start_snap: Option<SnapTarget>,
        e: vect2d, end_snap: Option<SnapTarget>,
        cursor: vect2d, threshold_px: f32,
    ) -> Option<(TangentHost, vect2d, vect2d)> {
        if self.snap_disabled { return None; }
        let mut best: Option<(f32, TangentHost, vect2d, vect2d)> = None;
        for (anchor_snap, anchor, other) in [(start_snap, s, e), (end_snap, e, s)] {
            let Some(sn) = anchor_snap else { continue };
            let Some((host, t)) = self.tangent_host_at_snap(sn, anchor) else { continue };
            let Some((c, r)) = circle_tangent_through(anchor, other, t) else { continue };
            let dx = cursor.x - c.x;
            let dy = cursor.y - c.y;
            let dm = (dx * dx + dy * dy).sqrt();
            if dm < 1e-9 { continue; }
            let dist_px = ((dm - r).abs() as f32) * self.scale;
            if dist_px < threshold_px && best.as_ref().map_or(true, |b| dist_px < b.0) {
                let mid = vect2d::new(c.x + dx / dm * r, c.y + dy / dm * r);
                best = Some((dist_px, host, anchor, mid));
            }
        }
        best.map(|(_, h, a, m)| (h, a, m))
    }

    /// Create the arc from the session values, tie its ends, snapped
    /// third point and tangent host, add a radius dim for a typed
    /// radius, and end the session. Called by the third click and by
    /// Enter with a typed radius. Typed text that does not evaluate,
    /// or a radius below half the chord, keeps the session open with
    /// a status error.
    pub(crate) fn complete_arc(&mut self) -> bool {
        let Some(st) = self.arc_draw.as_ref() else { return false };
        let Some(end) = st.end else { return false };
        let typed = match st.typed_r.clone() {
            None => None,
            Some(raw) => match self.parse_axis_input(&raw) {
                Some(parsed) => Some(parsed),
                None => {
                    self.status_error = Some(format!("Radius: invalid value or expression: {}", raw));
                    return false;
                }
            },
        };
        let s = st.start.pos;
        let half = ((end.pos.x - s.x).powi(2) + (end.pos.y - s.y).powi(2)).sqrt() * 0.5;
        if typed.is_some() && st.r < half - 1e-9 {
            self.status_error = Some(format!("Radius: {:.4} is below half the chord ({:.4})", st.r, half));
            return false;
        }
        let state = self.arc_draw.take().unwrap();
        self.begin_group();
        // Rejected creation (collinear points) ends the gesture.
        let Some(arc) = self.exec(Action::AddArc { start: s, end: end.pos, mid: state.mid }).arc() else {
            self.close_creation_input();
            return false;
        };
        // Arc start_angle always corresponds to the start click,
        // end_angle to the end click (direction stored in ccw flag).
        if let Some(sn) = state.start.snap {
            self.apply_snap_coincident_arc(sn, arc, ArcPoint::Start, s);
        }
        if let Some(sn) = end.snap {
            self.apply_snap_coincident_arc(sn, arc, ArcPoint::End, end.pos);
        }
        // Third point snapped: helper on the arc, tied to the target.
        if let Some((pos, target)) = state.live_snap
            && let Some(helper) = self.exec(Action::AddHelperPoint { pos }).point()
        {
            self.exec(Action::ApplyPointOnArc { point: helper, arc });
            self.apply_snap_coincident_point(target, helper);
        }
        // Tangent snap: the constraint, when the gate accepts it.
        if let Some((host, _)) = state.tangent {
            let action = match host {
                TangentHost::Line(line) => Action::ApplyTangentLA { line, arc },
                TangentHost::Arc(other) => Action::ApplyTangentAA { a: arc, b: other },
            };
            if arael_sketch_backend::conflicts::validate_action(&self.sketch, &action).is_none() {
                self.exec(action);
            }
        }
        if let Some((value, expr)) = typed {
            let value = if expr.is_some() { 0.0 } else { value };
            self.exec(Action::AddDimension {
                kind: DimensionKind::ArcRadius(arc), value, expr, derived: false, range: None,
            });
        }
        self.close_creation_input();
        true
    }

    /// Drop an in-progress arc gesture and close its value overlay.
    pub(crate) fn cancel_arc_session(&mut self) {
        if self.arc_draw.take().is_some_and(|s| s.end.is_some()) {
            self.close_creation_input();
        }
    }

    /// Canvas input for `Tool::DrawRect`: click the first corner, then
    /// the opposite one. The value overlay is live from the first
    /// click with width and height (Tab between them); typed sides
    /// are fixed -- the mouse then only picks the quadrant -- and
    /// become driving length dims on completion.
    pub(crate) fn handle_draw_rect(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        // Live tracking: untyped sides follow the mouse (or a corner
        // snap, offered only while neither side is typed -- the
        // corner must land exactly on the target for the coincidence
        // to be honest); the quadrant always follows the mouse.
        if self.rect_draw.is_some() {
            let (c, typed_w, typed_h) = {
                let s = self.rect_draw.as_ref().unwrap();
                (s.corner.pos, s.typed_w.is_some(), s.typed_h.is_some())
            };
            let snap = if typed_w || typed_h { None } else {
                self.find_snap_target(mouse_sketch, hit_threshold).filter(|(p, _)| {
                    (p.x - c.x).abs() > 1e-6 && (p.y - c.y).abs() > 1e-6
                })
            };
            let state = self.rect_draw.as_mut().unwrap();
            state.cursor = mouse_screen;
            state.live_snap = None;
            let p = snap.map_or(mouse_sketch, |(p, _)| p);
            let dx = p.x - c.x;
            let dy = p.y - c.y;
            if dx.abs() > 1e-9 { state.sx = dx.signum(); }
            if dy.abs() > 1e-9 { state.sy = dy.signum(); }
            if !typed_w && dx.abs() > 1e-9 {
                state.w = dx.abs();
                self.dim_input = format!("{:.4}", dx.abs());
                self.dim_select_all = true;
            }
            if !typed_h && dy.abs() > 1e-9 {
                state.h = dy.abs();
                state.height_text = format!("{:.4}", dy.abs());
            }
            if !typed_w && !typed_h {
                state.live_snap = snap;
            }
        }

        if !response.clicked_by(egui::PointerButton::Primary) {
            return;
        }
        if self.rect_draw.is_none() {
            // First click: corner, snap to nearby entity. Opens the
            // value overlay.
            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
            let corner = snap.map_or(mouse_sketch, |(p, _)| p);
            self.rect_draw = Some(RectDrawState {
                corner: PlacedPoint { pos: corner, snap: snap.map(|(_, t)| t) },
                w: 0.0,
                typed_w: None,
                h: 0.0,
                height_text: String::new(),
                typed_h: None,
                sx: 1.0,
                sy: 1.0,
                live_snap: None,
                cursor: mouse_screen,
            });
            self.open_creation_input();
        } else {
            // Second click: opposite corner. A zero-area rect waits;
            // typed text always goes through complete_rect so a
            // broken value is reported.
            let s = self.rect_draw.as_ref().unwrap();
            if s.typed_w.is_some() || s.typed_h.is_some() || (s.w > 1e-6 && s.h > 1e-6) {
                self.complete_rect();
            }
        }
    }

    /// Build the rect from the session values -- four lines, corner
    /// coincidents, H/V, snaps, and length dims for typed sides -- as
    /// one undo group, and end the session. Called by the corner
    /// click and by Enter with both sides typed. Typed text that does
    /// not evaluate keeps the session open with a status error.
    pub(crate) fn complete_rect(&mut self) -> bool {
        let Some(s) = self.rect_draw.as_ref() else { return false };
        let mut typed = [None, None];
        for (i, (raw, what)) in [(s.typed_w.clone(), "Width"), (s.typed_h.clone(), "Height")].into_iter().enumerate() {
            let Some(raw) = raw else { continue };
            match self.parse_axis_input(&raw) {
                Some(parsed) => typed[i] = Some(parsed),
                None => {
                    self.status_error = Some(format!("{}: invalid value or expression: {}", what, raw));
                    return false;
                }
            }
        }
        let s = self.rect_draw.as_ref().unwrap();
        if s.w < 1e-6 || s.h < 1e-6 { return false; }
        let state = self.rect_draw.take().unwrap();
        let bl = state.corner.pos;
        let tr = vect2d::new(bl.x + state.sx * state.w, bl.y + state.sy * state.h);
        let br = vect2d::new(tr.x, bl.y);
        let tl = vect2d::new(bl.x, tr.y);
        let corners = [bl, br, tr, tl];

        self.begin_group();
        let mut lines: std::vec::Vec<Ref<Line>> = std::vec::Vec::with_capacity(4);
        for i in 0..4 {
            let a = corners[i];
            let b = corners[(i + 1) % 4];
            match self.exec(Action::AddLine { p1: a, p2: b }).line() {
                Some(r) => lines.push(r),
                None => break,
            }
        }
        if lines.len() < 4 {
            // Atomic: a failed side rolls the partial rect back (the
            // sides so far are the current undo group).
            let rejection = self.status_error.take();
            if !lines.is_empty()
                && let Some((restored, cur)) = self.history.undo() {
                    self.sketch = restored.into();
                    self.command_cursor = cur.pos;
                    self.command_cursor_tangent = cur.tangent;
                    self.refresh_dof();
            }
            self.status_error = rejection;
            self.close_creation_input();
            return false;
        }

        // Corner coincidents: L(i).p2 = L(i+1).p1
        for i in 0..4 {
            self.exec(Action::ApplyCoincidentLL21 { a: lines[i], b: lines[(i + 1) % 4] });
        }
        // Axis-aligned: top/bottom horizontal, sides vertical.
        self.exec(Action::ApplyHorizontal { lines: vec![lines[0], lines[2]] });
        self.exec(Action::ApplyVertical { lines: vec![lines[1], lines[3]] });
        // External snap for the first corner (L0.p1) and the opposite
        // one (L1.p2).
        if let Some(sn) = state.corner.snap {
            self.apply_snap_coincident(sn, lines[0], true);
        }
        if let Some((_, sn)) = state.live_snap {
            self.apply_snap_coincident(sn, lines[1], false);
        }
        // Typed sides: length dims on the first horizontal / vertical
        // side.
        for (i, line) in [(0, lines[0]), (1, lines[1])] {
            if let Some((value, expr)) = typed[i].take() {
                let value = if expr.is_some() { 0.0 } else { value };
                self.exec(Action::AddDimension {
                    kind: DimensionKind::LineLength(line), value, expr, derived: false, range: None,
                });
            }
        }
        self.close_creation_input();
        true
    }

    /// Drop an in-progress rect gesture and close its value overlay.
    pub(crate) fn cancel_rect_session(&mut self) {
        if self.rect_draw.take().is_some() {
            self.close_creation_input();
        }
    }

    /// Canvas input for `Tool::Fillet | Tool::Chamfer`.
    pub(crate) fn handle_fillet_chamfer(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, _mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if response.clicked_by(egui::PointerButton::Primary) {
            // Derive a typed corner from this click: an endpoint snap
            // yields Endpoint{line, is_p1}; a line-body click
            // accumulates into self.selection and, once two lines are
            // in, yields Lines(a, b). Returns the corner plus the
            // shortest line length (used for the 10 % starting
            // radius when this is the first corner of the session).
            use arael_sketch_backend::corner_ops::CornerSpec;
            #[derive(Clone)]
            enum Picked { Corner(CornerSpec, f64), Nothing }
            let pre_len = |app: &Self, r: Ref<Line>| -> f64 {
                let ln = &app.sketch.lines[r];
                let dx = ln.p2.value.x - ln.p1.value.x;
                let dy = ln.p2.value.y - ln.p1.value.y;
                (dx * dx + dy * dy).sqrt()
            };
            let picked = match self.find_snap_target(mouse_sketch, hit_threshold) {
                Some((_, SnapTarget::LineP1(l))) => {
                    Picked::Corner(CornerSpec::Endpoint { line: l, is_p1: true }, pre_len(self, l))
                }
                Some((_, SnapTarget::LineP2(l))) => {
                    Picked::Corner(CornerSpec::Endpoint { line: l, is_p1: false }, pre_len(self, l))
                }
                _ => if let Some(Selection::Line(r)) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                    if self.selection.iter().any(|s| matches!(s, Selection::Line(rr) if *rr == r)) {
                        self.selection.retain(|s| !matches!(s, Selection::Line(rr) if *rr == r));
                    } else {
                        self.selection.push(Selection::Line(r));
                    }
                    let lines: Vec<Ref<Line>> = self.selection.iter().filter_map(|s| {
                        if let Selection::Line(r) = s { Some(*r) } else { None }
                    }).collect();
                    if lines.len() == 2 {
                        let shortest = pre_len(self, lines[0]).min(pre_len(self, lines[1]));
                        self.selection.clear();
                        Picked::Corner(CornerSpec::Lines(lines[0], lines[1]), shortest)
                    } else { Picked::Nothing }
                } else { Picked::Nothing },
            };
            match picked {
                Picked::Corner(spec, shortest) => {
                    if self.fillet_pending.is_some() {
                        self.toggle_fillet_corner(spec);
                    } else if self.tool == Tool::Chamfer {
                        self.try_start_gui_chamfer(spec, shortest);
                    } else {
                        self.try_start_gui_fillet(spec, shortest);
                    }
                }
                Picked::Nothing => {}
            }
        }
    }

    /// Canvas input for `Tool::Split | Tool::Trim`.
    pub(crate) fn handle_split_trim(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, _mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if response.clicked_by(egui::PointerButton::Primary) {
            let trim = self.tool == Tool::Trim;
            self.gui_split_trim(mouse_sketch, hit_threshold, trim);
        }
    }

    /// The line/arc under the cursor, as a split target. Endpoint hits
    /// resolve to their entity so a near-endpoint click still picks
    /// the curve (the cut itself is refused if degenerate).
    pub(crate) fn split_trim_target_at(&self, mouse_sketch: vect2d, hit_threshold: f64)
        -> Option<arael_sketch_backend::split::SplitTarget>
    {
        use arael_sketch_backend::split::SplitTarget;
        match self.hit_test_selection(mouse_sketch, hit_threshold)? {
            Selection::Line(r) | Selection::LineP1(r) | Selection::LineP2(r) => {
                Some(SplitTarget::Line(r))
            }
            Selection::Arc(r) | Selection::ArcStart(r) | Selection::ArcEnd(r) => {
                Some(SplitTarget::Arc(r))
            }
            _ => None,
        }
    }

    /// Hover preview for the split/trim tools: the target under the
    /// cursor and the parameter span a click would isolate. A `None`
    /// span means the whole entity (trim would delete it entirely).
    pub(crate) fn split_trim_preview(&self, mouse_sketch: vect2d, hit_threshold: f64)
        -> Option<(arael_sketch_backend::split::SplitTarget, Option<(f64, f64)>)>
    {
        use arael_sketch_backend::split;
        let target = self.split_trim_target_at(mouse_sketch, hit_threshold)?;
        let (t, _) = split::target_param_near(&self.sketch, target, mouse_sketch);
        let (all_cuts, _) = split::find_cuts(&self.sketch, target, None);
        Some((target, split::preview_span(&self.sketch, target, &all_cuts, t)))
    }

    /// One split/trim click: bracket the cuts around the click, run
    /// the plan plus its gated follow-ups as one undo group, and push
    /// the id-rich report to the command panel.
    pub(crate) fn gui_split_trim(&mut self, mouse_sketch: vect2d, hit_threshold: f64, trim: bool) {
        use arael_sketch_backend::split::{self, SplitPlan, SplitTarget};
        let Some(target) = self.split_trim_target_at(mouse_sketch, hit_threshold) else {
            return;
        };
        let (closed, tname) = match target {
            SplitTarget::Line(r) => (false, self.sketch.lines[r].name.clone()),
            SplitTarget::Arc(r) => (self.sketch.arcs[r].closed, self.sketch.arcs[r].name.clone()),
        };
        let (t, _) = split::target_param_near(&self.sketch, target, mouse_sketch);
        let (all_cuts, _) = split::find_cuts(&self.sketch, target, None);
        if all_cuts.is_empty() || (closed && all_cuts.len() < 2) {
            if trim {
                // Nothing to cut at: trim deletes the whole entity.
                self.begin_group();
                match target {
                    SplitTarget::Line(r) => { self.exec(Action::DeleteLine { line: r }); }
                    SplitTarget::Arc(r) => { self.exec(Action::DeleteArc { arc: r }); }
                }
                self.command_output.push((
                    format!("Trimmed (no intersections): deleted {}", tname),
                    false, false,
                ));
            } else {
                self.status_error = Some(format!("no intersections on {} to split at", tname));
            }
            return;
        }
        let (cuts, clicked) = split::bracket_cuts(&self.sketch, target, &all_cuts, t, closed);
        let n = split::piece_count(closed, cuts.len());
        let mut keep = vec![true; n];
        if trim {
            keep[clicked] = false;
        }
        let plan = SplitPlan { target, cuts, keep };
        self.begin_group();
        let outcome = match self.exec_split(plan.clone()) {
            Ok(o) => o,
            Err(e) => {
                self.status_error = Some(e);
                return;
            }
        };
        // Gated follow-ups; a rejection here means "already implied".
        let mark = self.sketch.next_constraint_id;
        for action in split::post_split_actions(&plan, &outcome.pieces, true) {
            self.exec(action);
            self.status_error = None;
        }
        let mut added: Vec<String> = Vec::new();
        self.sketch.for_each_constraint_collection_ref(|_, meta, coll| {
            if meta.dimension_backed {
                return;
            }
            for i in 0..coll.len() {
                let c = coll.item(i);
                if c.nid() >= mark {
                    added.push(format!("C{} {}", c.nid(), c.describe(&self.sketch)));
                }
            }
        });
        let kept: Vec<String> = outcome.piece_names.iter().flatten().cloned().collect();
        let verb = if trim { "Trimmed" } else { "Split" };
        let mut lines = vec![format!("{} {} -> {}", verb, tname, kept.join(" "))];
        let section = |label: &str, items: &[String], lines: &mut Vec<String>| {
            if !items.is_empty() {
                lines.push(format!("  {}: {}", label, items.join("; ")));
            }
        };
        section("added", &added, &mut lines);
        section("moved", &outcome.moved, &mut lines);
        section("copied", &outcome.copied, &mut lines);
        section("dropped", &outcome.dropped, &mut lines);
        section("expressions", &outcome.expr_report, &mut lines);
        self.command_output.push((lines.join("\n"), false, false));
    }

    /// Canvas input for `Tool::Scale`: single click toggles an entity
    /// into the scale set (lines, arcs and points alike); double-click
    /// on a point-like target sets the center. Once both exist, the
    /// value overlay opens with a live preview.
    pub(crate) fn handle_scale_input(&mut self, ui: &egui::Ui, ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        // Drag: box-select. The scale tool has no geometry drag, so
        // every drag is a marquee. Always additive -- the set is
        // built incrementally and clicks toggle entities back out.
        if response.dragged_by(egui::PointerButton::Primary) {
            if self.box_select_start.is_none() {
                self.box_select_start = ui.ctx().input(|i| i.pointer.press_origin());
            }
            ctx.request_repaint();
        }
        if response.drag_stopped_by(egui::PointerButton::Primary)
            && let Some(start) = self.box_select_start.take()
        {
            let end = mouse_screen;
            if (end.x - start.x).abs() >= 2.0 || (end.y - start.y).abs() >= 2.0 {
                self.apply_box_select(start, end, true);
                self.try_start_scale_session();
            }
        }

        if Self::multi_clicked(response) {
            let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) else { return };
            let pos = match sel {
                Selection::Point(r) => Some(self.sketch.points[r].pos.value),
                Selection::LineP1(r) => Some(self.sketch.lines[r].p1.value),
                Selection::LineP2(r) => Some(self.sketch.lines[r].p2.value),
                Selection::ArcCenter(r) => Some(self.sketch.arcs[r].center.value),
                Selection::ArcStart(r) => Some(arc_start_pos(&self.sketch.arcs[r])),
                Selection::ArcEnd(r) => Some(arc_end_pos(&self.sketch.arcs[r])),
                _ => None,
            };
            let Some(pos) = pos else { return };
            // The pair's first click already toggled this target into
            // the set; toggling again reverts it.
            if let Some(norm) = Self::scale_set_member(sel) {
                self.toggle_selection(norm);
            }
            self.scale_center = Some(pos);
            self.try_start_scale_session();
        } else if response.clicked_by(egui::PointerButton::Primary)
            && !Self::multi_clicked(response)
        {
            let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) else { return };
            if let Some(norm) = Self::scale_set_member(sel) {
                self.toggle_selection(norm);
                self.try_start_scale_session();
            }
        }
    }

    /// Double-click detection that survives a nearby preceding click.
    /// egui's click count is time-based only: a release within the
    /// double window of the last click counts 2, but within the wider
    /// triple window of the click before THAT it counts 3 -- so a
    /// stray click anywhere shortly before a double-click turns it
    /// into a "triple" and `double_clicked` stays false. Any count
    /// above one means the user double-clicked as far as the tools
    /// here care.
    pub(crate) fn multi_clicked(response: &egui::Response) -> bool {
        response.double_clicked_by(egui::PointerButton::Primary)
            || response.triple_clicked_by(egui::PointerButton::Primary)
    }

    /// Whole-entity membership for a scale-set click: endpoint hits
    /// resolve to their entity.
    fn scale_set_member(sel: Selection) -> Option<Selection> {
        match sel {
            Selection::Line(r) | Selection::LineP1(r) | Selection::LineP2(r) => {
                Some(Selection::Line(r))
            }
            Selection::Arc(r) | Selection::ArcCenter(r)
            | Selection::ArcStart(r) | Selection::ArcEnd(r) => Some(Selection::Arc(r)),
            Selection::Point(r) => Some(Selection::Point(r)),
            _ => None,
        }
    }

    /// Entering the Scale tool: keep an existing selection as the
    /// initial scale set. Endpoint selections resolve to their
    /// entity; constraints, dimensions and duplicates drop out.
    pub(crate) fn adopt_selection_for_scale(&mut self) {
        let mut kept: Vec<Selection> = Vec::new();
        for sel in self.selection.drain(..) {
            if let Some(norm) = Self::scale_set_member(sel) {
                if !kept.contains(&norm) {
                    kept.push(norm);
                }
            }
        }
        self.selection = kept;
    }

    /// The scale sets, read from the live selection.
    fn scale_sets(&self) -> (Vec<Ref<Line>>, Vec<Ref<Arc>>, Vec<Ref<Point>>) {
        let mut lines = Vec::new();
        let mut arcs = Vec::new();
        let mut points = Vec::new();
        for sel in &self.selection {
            match sel {
                Selection::Line(r) => if !lines.contains(r) { lines.push(*r); },
                Selection::Arc(r) => if !arcs.contains(r) { arcs.push(*r); },
                Selection::Point(r) => if !points.contains(r) { points.push(*r); },
                _ => {}
            }
        }
        (lines, arcs, points)
    }

    /// Open the value overlay once a center and at least one entity
    /// exist; re-preview if a session is already live.
    fn try_start_scale_session(&mut self) {
        if self.scale_pending.is_some() {
            self.reapply_scale();
            return;
        }
        if self.scale_center.is_none() {
            return;
        }
        let (l, a, p) = self.scale_sets();
        if l.is_empty() && a.is_empty() && p.is_empty() {
            return;
        }
        let Ok(pre_snapshot) = bincode::serialize(&self.sketch) else { return };
        self.scale_pending = Some(ScalePending {
            pre_snapshot,
            history_cursor_before: self.history.cursor,
            last_valid_factor: "1".into(),
            last_applied_sig: String::new(),
        });
        self.dim_input = "1".into();
        self.dim_editing = true;
        self.dim_edit_did = None;
        self.dim_kind = None;
        self.dim_placing = false;
        self.dim_select_all = true;
        self.dim_derived = false;
        self.dim_derived_prev = false;
        self.dim_input_backup.clear();
        self.reapply_scale();
    }

    /// Current factor token: the typed input when it evaluates to a
    /// positive number, otherwise the last valid one.
    fn scale_effective_factor(&self) -> Option<String> {
        let p = self.scale_pending.as_ref()?;
        let typed = self.dim_input.trim();
        if !typed.is_empty()
            && arael_sketch_backend::commands::eval_expr(&self.sketch, typed)
                .map(|v| v.is_finite() && v > 0.0)
                .unwrap_or(false)
        {
            return Some(typed.to_string());
        }
        Some(p.last_valid_factor.clone())
    }

    /// Restore the pre-scale sketch and re-apply with the current
    /// factor, sets and center. No-op when nothing changed.
    pub(crate) fn reapply_scale(&mut self) {
        let Some(p) = self.scale_pending.as_ref() else { return };
        let Some(center) = self.scale_center else { return };
        let Some(factor_tok) = self.scale_effective_factor() else { return };
        let (lines, arcs, points) = self.scale_sets();
        let sig = format!("{}|{:?}|{:?}|{:?}|{:?}", factor_tok, (center.x, center.y), lines, arcs, points);
        if sig == p.last_applied_sig {
            return;
        }
        let pre_snapshot = p.pre_snapshot.clone();
        let history_cursor_before = p.history_cursor_before;
        if let Ok(s) = bincode::deserialize::<Sketch>(&pre_snapshot) {
            self.sketch = s.into();
        }
        self.history.actions.truncate(history_cursor_before);
        self.history.snapshots.truncate(history_cursor_before);
        self.history.cursors.truncate(history_cursor_before);
        self.history.groups.truncate(history_cursor_before);
        self.history.cursor = history_cursor_before;
        self.status_error = None;
        let factor = arael_sketch_backend::commands::eval_expr(&self.sketch, &factor_tok)
            .unwrap_or(1.0);
        self.begin_group();
        self.exec(Action::Scale { lines, arcs, points, center, factor });
        let ok = self.status_error.is_none();
        if let Some(pending) = self.scale_pending.as_mut() {
            if ok {
                pending.last_valid_factor = factor_tok;
            }
            pending.last_applied_sig = sig;
        }
    }

    /// Enter: keep the applied scale, report, and end the session.
    pub(crate) fn commit_pending_scale(&mut self) {
        self.reapply_scale();
        // Report like the command does.
        let (lines, arcs, points) = self.scale_sets();
        let (_, report) = arael_sketch_backend::scale::classify_scale_dims(
            &self.sketch, &lines, &arcs, &points);
        let mut msg = "Scaled".to_string();
        for r in &lines { msg += &format!(" {}", self.sketch.lines[*r].name); }
        for r in &arcs { msg += &format!(" {}", self.sketch.arcs[*r].name); }
        for r in &points { msg += &format!(" {}", self.sketch.points[*r].name); }
        if let Some(p) = self.scale_pending.as_ref() {
            msg += &format!(" x{}", p.last_valid_factor);
        }
        if !report.scaled.is_empty() {
            msg += &format!("\n  dims scaled: {}", report.scaled.join(" "));
        }
        if !report.left.is_empty() {
            let left: Vec<String> = report.left.iter()
                .map(|(n, why)| format!("{} ({})", n, why)).collect();
            msg += &format!("\n  dims left: {}", left.join("; "));
        }
        self.command_output.push((msg, false, false));
        self.scale_pending = None;
        self.scale_center = None;
        self.selection.clear();
        self.dim_editing = false;
        self.dim_edit_did = None;
        self.dim_kind = None;
        self.dim_input.clear();
    }

    /// Escape: restore the pre-scale sketch and end the session.
    pub(crate) fn cancel_pending_scale(&mut self) {
        let Some(p) = self.scale_pending.take() else { return };
        if let Ok(s) = bincode::deserialize::<Sketch>(&p.pre_snapshot) {
            self.sketch = s.into();
        }
        self.history.actions.truncate(p.history_cursor_before);
        self.history.snapshots.truncate(p.history_cursor_before);
        self.history.cursors.truncate(p.history_cursor_before);
        self.history.groups.truncate(p.history_cursor_before);
        self.history.cursor = p.history_cursor_before;
        self.scale_center = None;
        self.dim_editing = false;
        self.dim_edit_did = None;
        self.dim_kind = None;
        self.dim_input.clear();
        self.status_error = None;
        self.refresh_dof();
    }

    /// Canvas input for `Tool::ConstraintMode(ct)`.
    pub(crate) fn handle_constraint_mode(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, _mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64, ct: ConstraintType) {
        if response.clicked_by(egui::PointerButton::Primary) {
            // Find what was clicked
            if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                // Only accept valid entities for this constraint
                if Self::is_valid_for_constraint(ct, &sel) {
                    self.toggle_selection(sel);
                    // Check if we can now apply
                    if self.can_apply_constraint(ct) {
                        match ct {
                            ConstraintType::Horizontal => self.apply_horizontal(),
                            ConstraintType::Vertical => self.apply_vertical(),
                            ConstraintType::Coincident => self.apply_coincident(),
                            ConstraintType::Parallel => self.apply_parallel(),
                            ConstraintType::Perpendicular => self.apply_perpendicular(),
                            ConstraintType::EqualLength => self.apply_equal_length(),
                            ConstraintType::Tangent => self.apply_tangent(),
                            ConstraintType::Collinear => self.apply_collinear(),
                            ConstraintType::Midpoint => self.apply_midpoint(),
                            ConstraintType::OnNormal => self.apply_on_normal(),
                            ConstraintType::Symmetry => self.apply_symmetry(),
                            ConstraintType::Lock => self.apply_lock(),
                            ConstraintType::ToggleConstruction => self.apply_toggle_construction(),
                        }
                        self.selection.clear();
                        // Stay in constraint mode for more
                    }
                }
            } else {
                self.selection.clear();
            }
        }
    }

    /// Canvas input for `Tool::Dimension`.
    pub(crate) fn handle_dimension_tool(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if self.dim_placing {
            // Dynamic kind switching: while the user drags the
            // preview, re-pick between PointPointDistance /
            // LineLength / HDistance / VDistance based on the
            // mouse's zone relative to the selected entities.
            // Line-based shapes (LineLength, or H/V between
            // the line's own endpoints) are checked first so
            // we don't degrade them to a generic point-pair.
            if let Some(current_kind) = self.dim_kind {
                if let Some(line_ref) = Self::line_dim_base_line(&current_kind) {
                    let new_kind = self.pick_line_dim_kind(line_ref, Some(mouse_sketch));
                    if new_kind != current_kind {
                        self.dim_kind = Some(new_kind);
                        let measured = self.measure_dimension(&new_kind);
                        self.dim_input = format!("{:.4}", measured);
                    }
                } else {
                    let endpoints = match current_kind {
                        DimensionKind::PointPointDistance(a, b)
                        | DimensionKind::HDistance(a, b)
                        | DimensionKind::VDistance(a, b) => Some((a, b)),
                        _ => None,
                    };
                    if let Some((a, b)) = endpoints {
                        let new_kind = self.pick_point_pair_dim_kind(a, b, Some(mouse_sketch));
                        if new_kind != current_kind {
                            self.dim_kind = Some(new_kind);
                            let measured = self.measure_dimension(&new_kind);
                            self.dim_input = format!("{:.4}", measured);
                        }
                    }
                }
            }
            // Phase 2: positioning with mouse, click to confirm.
            // Unlocked placement may switch the radius axis or angle
            // sector; a kind change re-measures and refreshes the
            // input text.
            if let Some(kind) = self.dim_kind
                && let Some(p) = self.dim_placement_from_mouse(&kind, mouse_sketch, false) {
                    if let Some(nk) = p.new_kind
                        && self.dim_kind != Some(nk) {
                            self.dim_kind = Some(nk);
                            let measured = self.measure_dimension(&nk);
                            self.dim_input = format!("{:.4}", measured);
                        }
                    self.dim_offset = p.offset;
                    self.dim_text_along = p.text_along.unwrap_or(0.0);
                }
            if response.clicked_by(egui::PointerButton::Primary) {
                // If clicking on a geometry entity, cancel placing and
                // add it to selection (to switch dimension type)
                let hit = self.hit_test_selection(mouse_sketch, hit_threshold);
                let hit_geometry = hit.as_ref().is_some_and(|s| matches!(s,
                    Selection::Line(_) | Selection::Arc(_) | Selection::Point(_)
                    | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_)));
                if hit_geometry {
                    self.dim_placing = false;
                    self.toggle_selection(hit.unwrap());
                    if let Some(kind) = self.selection_to_dim_kind(Some(mouse_sketch)) {
                        let measured = self.measure_dimension(&kind);
                        self.dim_input = format!("{:.4}", measured);
                        self.dim_kind = Some(kind);
                        self.dim_placing = true;
                        self.dim_offset = vect2d::new(0.0, 1.0);
                        self.dim_text_along = 0.0;
                    }
                } else {
                    // Confirm position, enter text input
                    self.dim_placing = false;
                    self.dim_editing = true;
                    self.dim_select_all = true;
                    self.dim_derived = false;
                    self.dim_derived_prev = false;
                    self.dim_input_backup.clear();
                }
            }
        } else if !self.dim_editing {
            // Phase 1: selecting entities
            if response.clicked_by(egui::PointerButton::Primary) {
                if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                    match &sel {
                        Selection::Line(_) | Selection::Arc(_)
                        | Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                        | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_) => {
                            self.toggle_selection(sel);
                        }
                        _ => {}
                    }
                    // Check if we can form a dimension
                    if let Some(kind) = self.selection_to_dim_kind(Some(mouse_sketch)) {
                        let measured = self.measure_dimension(&kind);
                        self.dim_input = format!("{:.4}", measured);
                        self.dim_kind = Some(kind);
                        self.dim_placing = true;
                        self.dim_offset = vect2d::new(0.0, 1.0);
                        self.dim_text_along = 0.0;
                    }
                } else {
                    self.selection.clear();
                }
            }
        }
        // Double-click on existing dimension to edit
        if response.double_clicked_by(egui::PointerButton::Primary) {
            // Check if clicking on a dimension
            for dim in self.sketch.dimensions.iter() {
                let (ts, te) = self.dim_text_segment(dim);
                let d = Self::screen_point_to_segment_dist(mouse_screen, ts, te);
                if d < 15.0 {
                    self.dim_input = Self::dim_edit_string(dim);
                    self.dim_kind = Some(dim.kind);
                    self.dim_offset = dim.offset;
                    self.dim_edit_did = Some(dim.did);
                    self.dim_editing = true;
                    self.dim_select_all = true;
                    self.dim_placing = false;
                    self.dim_derived = dim.derived;
                    self.dim_derived_prev = dim.derived;
                    self.dim_input_backup.clear();
                    break;
                }
            }
        }
    }

}
