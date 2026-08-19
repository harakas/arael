//! The Offset tool: its window (type, distances, flip, pin, apply), the
//! canvas input (click / double-click / marquee build the sequence, the
//! mouse picks the side), the live preview, and edit mode for an existing
//! offset. The engine is `arael_sketch_backend::offset`.

use eframe::egui;
use arael::vect::vect2d;
use arael_sketch_solver::*;
use arael_sketch_backend::Selection;
use arael_sketch_backend::chain::{self, Sequence};
use arael_sketch_backend::offset::{self, OffsetParams, OffsetPlan};
use crate::EditorApp;
use crate::tools::Tool;

/// What the tool window shows and what the canvas built.
pub struct OffsetToolState {
    pub kind: OffsetKind,
    /// The distance fields as typed (number or expression).
    pub distance: String,
    pub distance2: String,
    /// +1 left of the chain direction, -1 right.
    pub side: f64,
    /// Once Flip was pressed the side no longer follows the mouse.
    pub side_fixed: bool,
    pub pinned: bool,
    /// Round convex corners with an arc of the distance.
    pub round: bool,
    /// Close the ends of an open two-sided result.
    pub caps: CapKind,
    /// Editing an existing offset: its meta id.
    pub edit: Option<u32>,
    /// The sequence the selection orders into (the side is judged on it
    /// even when the plan fails).
    pub seq: Option<Sequence>,
    /// The current plan (for the preview) or why there is none.
    pub plan: Option<OffsetPlan>,
    pub error: Option<String>,
    /// Give the distance field the focus on the next frame, its text
    /// selected so typing replaces it.
    pub focus_distance: bool,
    /// Edit mode: a typed distance not yet applied (Enter applies).
    pub pending_text: bool,
}

impl Default for OffsetToolState {
    fn default() -> Self {
        OffsetToolState {
            kind: OffsetKind::OneSide,
            distance: "1".into(),
            distance2: "1".into(),
            side: 1.0,
            side_fixed: false,
            pinned: true,
            round: false,
            caps: CapKind::None,
            edit: None,
            seq: None,
            plan: None,
            error: None,
            focus_distance: true,
            pending_text: false,
        }
    }
}

impl EditorApp {
    /// Enter the tool: keep a selection of lines and arcs as the sequence;
    /// a selected offset result opens its offset for editing.
    pub fn enter_offset_tool(&mut self) {
        self.tool = Tool::Offset;
        let mut state = OffsetToolState::default();
        let edit = self.selected_offset_meta();
        let mut kept: Vec<Selection> = Vec::new();
        for sel in self.selection.drain(..) {
            if let Some(norm) = Self::offset_set_member(sel) {
                if !kept.contains(&norm) {
                    kept.push(norm);
                }
            }
        }
        self.selection = kept;
        if let Some(mid) = edit {
            self.load_offset_for_edit(&mut state, mid);
        }
        self.offset_tool = Some(state);
        self.refresh_offset_plan();
    }

    pub fn leave_offset_tool(&mut self) {
        self.offset_tool = None;
    }

    /// Whole-entity membership: endpoint hits resolve to their line / arc;
    /// points, constraints and dimensions drop out.
    fn offset_set_member(sel: Selection) -> Option<Selection> {
        match sel {
            Selection::Line(r) | Selection::LineP1(r) | Selection::LineP2(r) => Some(Selection::Line(r)),
            Selection::Arc(r) | Selection::ArcCenter(r) | Selection::ArcStart(r) | Selection::ArcEnd(r) => {
                Some(Selection::Arc(r))
            }
            _ => None,
        }
    }

    fn selection_entities(&self) -> Vec<OffsetEntity> {
        self.selection
            .iter()
            .filter_map(|s| match s {
                Selection::Line(r) => Some(OffsetEntity::Line(*r)),
                Selection::Arc(r) => Some(OffsetEntity::Arc(*r)),
                _ => None,
            })
            .collect()
    }

    /// Open a meta-constraint in its tool for editing (the Offset tool
    /// in edit mode, for an offset).
    pub(crate) fn open_meta_edit(&mut self, mid: u32) {
        let Some(i) = self.sketch.meta_index(mid) else { return };
        self.selection.clear();
        self.selection.push(Selection::Meta(mid));
        match &self.sketch.metas[i].kind {
            MetaKind::Offset(_) => self.enter_offset_tool(),
            MetaKind::Pattern(_) => self.enter_pattern_tool(),
        }
    }

    /// The offset meta selected by its marker, or one of the selected
    /// entities is a result of.
    fn selected_offset_meta(&self) -> Option<u32> {
        for s in &self.selection {
            if let Selection::Meta(mid) = s
                && self.sketch.meta_index(*mid).is_some_and(|i| self.sketch.metas[i].as_offset().is_some())
            {
                return Some(*mid);
            }
        }
        for e in self.selection_entities() {
            if let Some(m) = arael_sketch_backend::meta::owner_of(&self.sketch, e) {
                if m.as_offset().is_some() {
                    return Some(m.mid);
                }
            }
        }
        None
    }

    fn load_offset_for_edit(&mut self, state: &mut OffsetToolState, mid: u32) {
        let Some(i) = self.sketch.meta_index(mid) else { return };
        let Some(o) = self.sketch.metas[i].as_offset() else { return };
        let p = offset::params_of(o);
        state.kind = p.kind;
        state.distance = p.distance.expr.clone().unwrap_or_else(|| trim_num(p.distance.value));
        state.distance2 = p
            .distance2
            .as_ref()
            .map(|v| v.expr.clone().unwrap_or_else(|| trim_num(v.value)))
            .unwrap_or_else(|| state.distance.clone());
        state.side = p.side;
        state.side_fixed = true;
        state.pinned = p.pinned;
        state.round = p.round;
        state.caps = p.caps;
        state.edit = Some(mid);
        self.select_offset_result(mid);
    }

    /// Show an offset's result as the selection.
    fn select_offset_result(&mut self, mid: u32) {
        let Some(i) = self.sketch.meta_index(mid) else { return };
        let Some(o) = self.sketch.metas[i].as_offset() else { return };
        self.selection = o
            .result_entities()
            .map(|e| match e {
                OffsetEntity::Line(l) => Selection::Line(l),
                OffsetEntity::Arc(a) => Selection::Arc(a),
            })
            .collect();
    }

    /// The parameters as typed, or why they do not parse.
    fn offset_params(&self, state: &OffsetToolState) -> Result<OffsetParams, String> {
        let distance = offset::parse_value(&self.sketch, &state.distance)?;
        let distance2 = if state.kind == OffsetKind::TwoSides {
            Some(offset::parse_value(&self.sketch, &state.distance2)?)
        } else {
            None
        };
        Ok(OffsetParams {
            kind: state.kind,
            distance,
            distance2,
            side: state.side,
            pinned: state.pinned,
            round: state.round,
            caps: state.caps,
        })
    }

    /// The sequence under edit, or the selection ordered into one.
    fn offset_sequence(&self, state: &OffsetToolState) -> Result<Sequence, String> {
        if let Some(mid) = state.edit
            && let Some(i) = self.sketch.meta_index(mid)
            && let Some(o) = self.sketch.metas[i].as_offset()
        {
            return Ok(offset::sequence_of(o));
        }
        let set = self.selection_entities();
        if set.is_empty() {
            return Err("click or box-select the lines and arcs to offset; double-click walks a sequence".into());
        }
        chain::order(&self.sketch, &set)
    }

    /// Recompute the preview plan from the selection and the window.
    pub fn refresh_offset_plan(&mut self) {
        let Some(mut state) = self.offset_tool.take() else { return };
        let seq = self.offset_sequence(&state);
        state.seq = seq.as_ref().ok().cloned();
        let result = seq
            .and_then(|seq| self.offset_params(&state).map(|p| (seq, p)))
            .and_then(|(seq, p)| offset::plan(&self.sketch, &seq, &p));
        match result {
            Ok(plan) => {
                state.plan = Some(plan);
                state.error = None;
            }
            Err(e) => {
                state.plan = None;
                state.error = Some(e);
            }
        }
        self.offset_tool = Some(state);
    }

    /// Canvas input for `Tool::Offset`: click toggles a line / arc into
    /// the set, double-click walks the sequence through it, a marquee
    /// adds the boxed entities; the mouse picks the side until Flip.
    pub(crate) fn handle_offset_input(&mut self, ui: &egui::Ui, ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if self.offset_tool.is_none() {
            self.offset_tool = Some(OffsetToolState::default());
        }
        let mut changed = false;

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
                self.leave_offset_edit();
                self.apply_box_select(start, end, true);
                self.selection.retain(|s| matches!(s, Selection::Line(_) | Selection::Arc(_)));
                changed = true;
            }
        }

        if Self::multi_clicked(response) {
            if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold)
                && let Some(norm) = Self::offset_set_member(sel)
            {
                let seed = match norm {
                    Selection::Line(r) => OffsetEntity::Line(r),
                    Selection::Arc(r) => OffsetEntity::Arc(r),
                    _ => unreachable!(),
                };
                if !self.open_offset_edit_for(seed) {
                    let seq = chain::walk(&self.sketch, seed);
                    self.selection = seq
                        .entities()
                        .map(|e| match e {
                            OffsetEntity::Line(l) => Selection::Line(l),
                            OffsetEntity::Arc(a) => Selection::Arc(a),
                        })
                        .collect();
                }
                changed = true;
            }
        } else if response.clicked_by(egui::PointerButton::Primary) {
            let hit = self.hit_test_selection(mouse_sketch, hit_threshold);
            match (hit.and_then(Self::offset_set_member), hit) {
                (Some(norm), _) => {
                    let e = match norm {
                        Selection::Line(r) => OffsetEntity::Line(r),
                        Selection::Arc(r) => OffsetEntity::Arc(r),
                        _ => unreachable!(),
                    };
                    if !self.open_offset_edit_for(e) {
                        self.leave_offset_edit();
                        self.toggle_selection(norm);
                    }
                }
                // A meta-constraint's marker opens it.
                (None, Some(Selection::Meta(mid))) => {
                    if !self.open_offset_edit_mid(mid) {
                        self.leave_offset_edit();
                        self.selection.clear();
                    }
                }
                (None, _) => {
                    self.leave_offset_edit();
                    self.selection.clear();
                }
            }
            changed = true;
        }

        // A changed set: the distance field takes the keyboard again, its
        // text selected, so the distance can be typed right away.
        if changed
            && let Some(state) = self.offset_tool.as_mut()
            && state.edit.is_none()
        {
            state.focus_distance = true;
        }
        // The side follows the mouse until it is fixed.
        if let Some(state) = self.offset_tool.as_mut()
            && !state.side_fixed
            && state.edit.is_none()
            && let Some(seq) = state.seq.as_ref()
        {
            let side = offset::side_of_point(&self.sketch, seq, mouse_sketch);
            if side != state.side {
                state.side = side;
                changed = true;
            }
        }
        if changed || self.offset_tool.as_ref().is_some_and(|s| s.plan.is_none() && s.error.is_none()) {
            self.refresh_offset_plan();
            ctx.request_repaint();
        }
    }

    /// Clicking a result of an existing offset opens that offset for
    /// editing. Returns whether it did.
    fn open_offset_edit_for(&mut self, e: OffsetEntity) -> bool {
        let Some(mid) = arael_sketch_backend::meta::owner_of(&self.sketch, e)
            .filter(|m| m.as_offset().is_some())
            .map(|m| m.mid)
        else {
            return false;
        };
        self.open_offset_edit_mid(mid)
    }

    /// Switch the tool to editing the offset `mid`; false when it is not
    /// an offset.
    fn open_offset_edit_mid(&mut self, mid: u32) -> bool {
        if !self.sketch.meta_index(mid).is_some_and(|i| self.sketch.metas[i].as_offset().is_some()) {
            return false;
        }
        let mut state = self.offset_tool.take().unwrap_or_default();
        if state.edit != Some(mid) {
            state = OffsetToolState::default();
            self.load_offset_for_edit(&mut state, mid);
        }
        self.offset_tool = Some(state);
        true
    }

    fn leave_offset_edit(&mut self) {
        if let Some(state) = self.offset_tool.as_mut()
            && state.edit.is_some()
        {
            let mut fresh = OffsetToolState::default();
            fresh.distance = state.distance.clone();
            fresh.distance2 = state.distance2.clone();
            fresh.kind = state.kind;
            fresh.pinned = state.pinned;
            fresh.round = state.round;
            fresh.caps = state.caps;
            *state = fresh;
            self.selection.clear();
        }
    }

    /// Create the planned offset (creation mode).
    pub fn apply_offset(&mut self) {
        let Some(state) = self.offset_tool.as_ref() else { return };
        if state.edit.is_some() {
            return;
        }
        let Some(plan) = state.plan.clone() else {
            if let Some(e) = state.error.clone() {
                self.status_error = Some(e);
            }
            return;
        };
        self.status_error = None;
        match offset::apply(self, &plan) {
            Ok(out) => {
                // Done: the tool closes on the new result, selected; its
                // marker opens it again for editing.
                self.select_offset_result(out.mid);
                self.leave_offset_tool();
                self.tool = Tool::Select;
            }
            Err(e) => self.status_error = Some(e),
        }
    }

    /// Apply the window's parameters to the offset under edit.
    pub(crate) fn update_offset(&mut self) {
        let Some(state) = self.offset_tool.as_ref() else { return };
        let Some(mid) = state.edit else { return };
        let params = match self.offset_params(state) {
            Ok(p) => p,
            Err(e) => { self.status_error = Some(e); return; }
        };
        self.status_error = None;
        match offset::update(self, mid, &params) {
            Ok(_) => {
                if let Some(state) = self.offset_tool.as_mut() {
                    state.pending_text = false;
                }
                // The result set may have changed (a side added or removed).
                let mut state = self.offset_tool.take().unwrap_or_default();
                self.load_offset_for_edit(&mut state, mid);
                self.offset_tool = Some(state);
                self.refresh_offset_plan();
            }
            Err(e) => {
                // Rolled back: the record and its result are as before;
                // the window keeps what was typed, with the error.
                self.status_error = Some(e);
                self.select_offset_result(mid);
            }
        }
    }

    /// The tool window.
    pub fn render_offset_window(&mut self, ctx: &egui::Context) {
        if self.tool != Tool::Offset {
            return;
        }
        let Some(mut state) = self.offset_tool.take() else { return };
        let title = match state.edit.and_then(|mid| self.sketch.meta_index(mid)) {
            Some(i) => format!("Offset {}", self.sketch.metas[i].name),
            None => "Offset".to_string(),
        };
        let mut changed = false;
        let mut apply = false;
        let mut close = false;
        let mut flip = false;
        let mut text_committed = false;
        let editing = state.edit.is_some();
        egui::Window::new(title)
            .id(egui::Id::new("offset_tool_window"))
            .default_pos(egui::pos2(self.canvas_rect.left() + 12.0, self.canvas_rect.top() + 12.0))
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    changed |= ui.selectable_value(&mut state.kind, OffsetKind::OneSide, "One side").changed();
                    changed |= ui.selectable_value(&mut state.kind, OffsetKind::Symmetric, "Symmetric").changed();
                    changed |= ui.selectable_value(&mut state.kind, OffsetKind::TwoSides, "Two sides").changed();
                });
                ui.horizontal(|ui| {
                    ui.label("Distance");
                    let r = ui.add(egui::TextEdit::singleline(&mut state.distance).desired_width(80.0).id(egui::Id::new("offset_distance")));
                    if state.focus_distance {
                        r.request_focus();
                        let mut ts = egui::TextEdit::load_state(ui.ctx(), r.id).unwrap_or_default();
                        ts.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                            egui::text::CCursor::new(0),
                            egui::text::CCursor::new(state.distance.chars().count()),
                        )));
                        egui::TextEdit::store_state(ui.ctx(), r.id, ts);
                        state.focus_distance = false;
                    }
                    if r.changed() {
                        changed = true;
                        if editing { state.pending_text = true; }
                    }
                    if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                        text_committed = true;
                    }
                    if state.kind == OffsetKind::TwoSides {
                        ui.label("and");
                        let r2 = ui.add(egui::TextEdit::singleline(&mut state.distance2).desired_width(80.0).id(egui::Id::new("offset_distance2")));
                        if r2.changed() {
                            changed = true;
                            if editing { state.pending_text = true; }
                        }
                        if r2.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                            text_committed = true;
                        }
                    }
                });
                ui.horizontal(|ui| {
                    if ui.button("Flip").on_hover_text("The other side").clicked() {
                        flip = true;
                    }
                    ui.label(if state.side > 0.0 { "left of the chain" } else { "right of the chain" });
                });
                // Caps close the ends of an open two-sided result.
                let open = state.seq.as_ref().is_none_or(|s| !s.closed);
                if open && state.kind != OffsetKind::OneSide {
                    ui.horizontal(|ui| {
                        ui.label("Caps");
                        changed |= ui.selectable_value(&mut state.caps, CapKind::None, "None").changed();
                        changed |= ui.selectable_value(&mut state.caps, CapKind::Line, "Line").on_hover_text("A line across each end").changed();
                        if state.kind == OffsetKind::Symmetric {
                            changed |= ui.selectable_value(&mut state.caps, CapKind::Round, "Round").on_hover_text("A half circle of the distance around each end").changed();
                        } else if state.caps == CapKind::Round {
                            state.caps = CapKind::Line;
                            changed = true;
                        }
                    });
                } else if state.caps != CapKind::None {
                    state.caps = CapKind::None;
                    changed = true;
                }
                ui.horizontal(|ui| {
                    changed |= ui.checkbox(&mut state.round, "Round corners").on_hover_text("Convex corners get an arc of the distance around the source corner instead of a sharp corner.").changed();
                    changed |= ui.checkbox(&mut state.pinned, "Pin ends").on_hover_text("Hold the free ends and tangent joints on the source's normals (on_normal). Off: they stay free.").changed();
                });
                if let Some(plan) = &state.plan {
                    if plan.approximate {
                        ui.colored_label(self.colors.notice_text, "approximate: an ellipse's offset is the concentric ellipse with both semi-axes moved");
                    }
                    let gone = offset::dropped_names(&self.sketch, plan);
                    if !gone.is_empty() {
                        ui.colored_label(self.colors.notice_text, format!("{} vanish at this distance (offset inward past the radius); the neighbours meet directly", gone.join(" ")));
                    }
                } else if let Some(e) = &state.error {
                    ui.colored_label(self.colors.error_text, e.as_str());
                }
                ui.horizontal(|ui| {
                    if editing {
                        ui.label("changes apply as you make them");
                        if state.pending_text && ui.button("Apply distance").clicked() {
                            text_committed = true;
                        }
                        if ui.button("Done").clicked() {
                            close = true;
                        }
                    } else {
                        if ui.add_enabled(state.plan.is_some(), egui::Button::new("Create")).clicked() {
                            apply = true;
                        }
                        if ui.button("Cancel").clicked() {
                            close = true;
                        }
                    }
                });
            });
        if flip {
            state.side = -state.side;
            state.side_fixed = true;
            changed = true;
        }
        // Enter in a distance field applies / creates.
        if text_committed {
            if editing { changed = true; } else { apply = true; }
        }
        self.offset_tool = Some(state);
        if changed {
            self.refresh_offset_plan();
            if editing && (text_committed || !self.offset_tool.as_ref().is_some_and(|s| s.pending_text)) {
                self.update_offset();
            }
        }
        if apply {
            self.apply_offset();
        }
        if close {
            self.leave_offset_edit();
            self.selection.clear();
            self.tool = Tool::Select;
            self.leave_offset_tool();
        }
    }

    /// The preview: the planned result, dashed. In edit mode only while a
    /// typed distance is not yet applied (the other changes apply at once),
    /// so the new place shows before Enter / Apply.
    pub fn draw_offset_preview(&self, painter: &egui::Painter) {
        let Some(state) = self.offset_tool.as_ref() else { return };
        if state.edit.is_some() && !state.pending_text {
            return;
        }
        let Some(plan) = state.plan.as_ref() else { return };
        let stroke = egui::Stroke::new(1.5, self.colors.offset_preview);
        for poly in offset::preview_polylines(plan, 48) {
            let pts: Vec<egui::Pos2> = poly.into_iter().map(|p| self.to_screen(p)).collect();
            crate::drawing::draw_styled_polyline(painter, &pts, stroke, LineStyle::Dashed);
        }
    }
}

fn trim_num(v: f64) -> String {
    let s = format!("{:.4}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() { "0".into() } else { s.to_string() }
}
