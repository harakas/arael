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
                let dist = ((mouse.x - cx).powi(2) + (mouse.y - cy).powi(2)).sqrt();
                let offset_y = dist - a.radius.value;
                let mouse_angle = (mouse.y - cy).atan2(mouse.x - cx);
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
        // Double-click on dimension to edit value
        if response.double_clicked_by(egui::PointerButton::Primary) {
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
            && !response.double_clicked_by(egui::PointerButton::Primary)
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
        if response.double_clicked_by(egui::PointerButton::Primary) {
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

                // Auto-horizontal/vertical snap: only when no
                // stronger placement constraint has fired
                // (end-snap position, start-perp combined,
                // start-perp free, or end-perp, or collinear).
                // Otherwise the drawn line already has its
                // angle determined by the perp/snap and H/V
                // would conflict.
                let hv = if !self.snap_disabled
                    && !combined_used
                    && perp_host.is_none()
                    && end_perp_target.is_none()
                    && end_snap.is_none()
                    && collinear_host.is_none()
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
    pub(crate) fn handle_draw_circle(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, _mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if response.clicked_by(egui::PointerButton::Primary) {
            self.begin_group();
            if let Some(state) = self.circle_draw.take() {
                // Second click: edge point
                let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                let edge = snap.map_or(mouse_sketch, |(p, _)| p);
                // Rejected creation (zero radius): gesture
                // ends, frame still renders.
                if let Some(new_arc) = self.exec(Action::AddCircle { center: state.center.pos, edge }).arc() {
                    // Auto-coincident for center
                    if let Some(s) = state.center.snap {
                        self.apply_snap_coincident_arc(s, new_arc, ArcPoint::Center, state.center.pos);
                    }
                    // Auto-coincident for edge (point on circle)
                    if let Some((_, s)) = snap {
                        if let Some(helper) = self.exec(Action::AddHelperPoint { pos: edge }).point() {
                            self.exec(Action::ApplyPointOnArc { point: helper, arc: new_arc });
                            self.apply_snap_coincident_point(s, helper);
                        }
                    }
                }
            } else {
                // First click: center
                let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                let center = snap.map_or(mouse_sketch, |(p, _)| p);
                self.circle_draw = Some(CircleDrawState {
                    center: PlacedPoint { pos: center, snap: snap.map(|(_, t)| t) },
                });
            }
        }
    }

    /// Canvas input for `Tool::DrawArc`.
    pub(crate) fn handle_draw_arc(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, _mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if response.clicked_by(egui::PointerButton::Primary) {
            self.begin_group();
            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
            let pos = snap.map_or(mouse_sketch, |(p, _)| p);
            let snap_target = snap.map(|(_, t)| t);

            if let Some(state) = self.arc_draw.take() {
                if let Some(PlacedPoint { pos: end, snap: snap_end }) = state.end {
                    // Third click: mid point on arc, create it.
                    // Rejected creation (collinear points):
                    // gesture ends, frame still renders.
                    if let Some(new_arc) = self.exec(Action::AddArc { start: state.start.pos, end, mid: pos }).arc() {
                        // Arc start_angle always corresponds to start click,
                        // end_angle to end click (direction stored in ccw flag)
                        let (start_ap, end_ap) = (ArcPoint::Start, ArcPoint::End);

                        // Auto-coincident for start click
                        if let Some(s) = state.start.snap {
                            self.apply_snap_coincident_arc(s, new_arc, start_ap, state.start.pos);
                        }
                        // Auto-coincident for end click
                        if let Some(s) = snap_end {
                            self.apply_snap_coincident_arc(s, new_arc, end_ap, end);
                        }
                        // Auto-coincident for mid (point on arc) - needs helper point
                        if let Some(s) = snap_target
                            && let Some(helper) = self.exec(Action::AddHelperPoint { pos }).point() {
                                self.exec(Action::ApplyPointOnArc { point: helper, arc: new_arc });
                                self.apply_snap_coincident_point(s, helper);
                        }
                    }
                } else {
                    // Second click: end point
                    self.arc_draw = Some(ArcDrawState {
                        start: state.start,
                        end: Some(PlacedPoint { pos, snap: snap_target }),
                    });
                }
            } else {
                // First click: start point
                self.arc_draw = Some(ArcDrawState {
                    start: PlacedPoint { pos, snap: snap_target },
                    end: None,
                });
            }
        }
    }

    /// Canvas input for `Tool::DrawRect`.
    pub(crate) fn handle_draw_rect(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, _mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if response.clicked_by(egui::PointerButton::Primary) {
            self.begin_group();
            if let Some(state) = self.rect_draw.take() {
                // Second click: opposite corner. Snap it, reject
                // zero-area rects.
                let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                let p2 = snap.map_or(mouse_sketch, |(p, _)| p);
                let dx = p2.x - state.corner.pos.x;
                let dy = p2.y - state.corner.pos.y;
                if dx.abs() < 1e-6 || dy.abs() < 1e-6 {
                    self.rect_draw = Some(state);
                } else {
                    let bl = state.corner.pos;
                    let br = vect2d::new(p2.x, state.corner.pos.y);
                    let tr = p2;
                    let tl = vect2d::new(state.corner.pos.x, p2.y);
                    let corners = [bl, br, tr, tl];

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
                        // Atomic: a failed side rolls the
                        // partial rect back (the sides so
                        // far are the current undo group).
                        let rejection = self.status_error.take();
                        if !lines.is_empty()
                            && let Some((restored, cur)) = self.history.undo() {
                                self.sketch = restored.into();
                                self.command_cursor = cur.pos;
                                self.command_cursor_tangent = cur.tangent;
                                self.refresh_dof();
                        }
                        self.status_error = rejection;
                    } else {

                    // Corner coincidents: L(i).p2 = L(i+1).p1
                    for i in 0..4 {
                        self.exec(Action::ApplyCoincidentLL21 {
                            a: lines[i],
                            b: lines[(i + 1) % 4],
                        });
                    }

                    // Axis-aligned: top/bottom horizontal, sides vertical.
                    self.exec(Action::ApplyHorizontal { lines: vec![lines[0], lines[2]] });
                    self.exec(Action::ApplyVertical { lines: vec![lines[1], lines[3]] });

                    // External snap for bl (L0.p1) and tr (L1.p2).
                    if let Some(s) = state.corner.snap {
                        self.apply_snap_coincident(s, lines[0], true);
                    }
                    if let Some((_, s)) = snap {
                        self.apply_snap_coincident(s, lines[1], false);
                    }
                    } // end else (all four sides created)
                }
            } else {
                // First click: opposite corner, snap to nearby entity.
                let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                let corner = snap.map_or(mouse_sketch, |(p, _)| p);
                self.rect_draw = Some(RectDrawState {
                    corner: PlacedPoint { pos: corner, snap: snap.map(|(_, t)| t) },
                });
            }
        }
    }

    /// Canvas input for `Tool::Fillet | Tool::Chamfer`.
    pub(crate) fn handle_fillet_chamfer(&mut self, _ui: &egui::Ui, _ctx: &egui::Context, response: &egui::Response, _mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if response.clicked_by(egui::PointerButton::Primary) {
            // Derive a "corner arg" from this click: an
            // endpoint snap yields "<line>.pN"; a
            // line-body click accumulates into
            // self.selection and, once two lines are in,
            // yields "L0 L1". Returns the arg plus the
            // shortest line length (used for the 10 %
            // starting radius when this is the first
            // corner of the session).
            #[derive(Clone)]
            enum Picked { Corner(String, f64), Nothing }
            let pre_len = |app: &Self, r: Ref<Line>| -> f64 {
                let ln = &app.sketch.lines[r];
                let dx = ln.p2.value.x - ln.p1.value.x;
                let dy = ln.p2.value.y - ln.p1.value.y;
                (dx * dx + dy * dy).sqrt()
            };
            let picked = match self.find_snap_target(mouse_sketch, hit_threshold) {
                Some((_, SnapTarget::LineP1(l))) => {
                    Picked::Corner(format!("{}.p1", self.sketch.lines[l].name), pre_len(self, l))
                }
                Some((_, SnapTarget::LineP2(l))) => {
                    Picked::Corner(format!("{}.p2", self.sketch.lines[l].name), pre_len(self, l))
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
                        let arg = format!("{} {}", self.sketch.lines[lines[0]].name, self.sketch.lines[lines[1]].name);
                        self.selection.clear();
                        Picked::Corner(arg, shortest)
                    } else { Picked::Nothing }
                } else { Picked::Nothing },
            };
            match picked {
                Picked::Corner(arg, shortest) => {
                    if self.fillet_pending.is_some() {
                        self.toggle_fillet_corner(&arg);
                    } else if self.tool == Tool::Chamfer {
                        self.try_start_gui_chamfer(&arg, shortest);
                    } else {
                        self.try_start_gui_fillet(&arg, shortest);
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
