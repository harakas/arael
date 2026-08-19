// The Pattern tool: circular / rectangular patterns of the selected
// entities, with its own window (docs/dev/PATTERN.md). The set is the
// selection (click, box, double-click walk); the center / direction
// line are picked with the Pick buttons; the window holds the numbers.
// The engine (arael_sketch_backend::pattern) plans, previews, creates
// and edits; this file is the egui side of it.

use eframe::egui;
use arael::vect::vect2d;
use arael_sketch_solver::*;
use arael_sketch_backend::Selection;
use arael_sketch_backend::pattern::{self, PatternParams, PatternPlan, PatternSpec};
use crate::EditorApp;
use crate::tools::Tool;

/// Which pattern the window makes.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PatternToolKind {
    Circular,
    Rectangular,
}

/// What the next canvas click picks instead of adding to the set.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Pick {
    Center,
    Frame,
}

/// The tool's session: the window's fields, the edit target, the plan.
pub struct PatternToolState {
    pub kind: PatternToolKind,
    // Circular
    pub center: Option<CenterRef>,
    pub distribution: Distribution,
    pub quantity: String,
    pub angle: String,
    // Rectangular
    pub frame: Option<arael::refs::Ref<Line>>,
    pub extent: bool,
    pub quantity1: String,
    pub distance1: String,
    pub symmetric1: bool,
    pub quantity2: String,
    pub distance2: String,
    pub symmetric2: bool,
    /// A Pick button was pressed: the next click sets the center / frame.
    pub picking: Option<Pick>,
    /// Editing an existing pattern: its meta id.
    pub edit: Option<u32>,
    /// The current plan (for the preview) or why there is none.
    pub plan: Option<PatternPlan>,
    pub error: Option<String>,
    /// Edit mode: a typed number not yet applied (Enter applies).
    pub pending_text: bool,
}

impl Default for PatternToolState {
    fn default() -> Self {
        PatternToolState {
            kind: PatternToolKind::Circular,
            center: None,
            distribution: Distribution::Full,
            quantity: "4".into(),
            angle: "90".into(),
            frame: None,
            extent: false,
            quantity1: "3".into(),
            distance1: "1".into(),
            symmetric1: false,
            quantity2: "1".into(),
            distance2: "1".into(),
            symmetric2: false,
            picking: None,
            edit: None,
            plan: None,
            error: None,
            pending_text: false,
        }
    }
}

impl EditorApp {
    /// Enter the tool: the selection (lines, arcs, points) is the set; a
    /// selected pattern (its marker, or one of its copies) is opened
    /// for editing instead.
    pub fn enter_pattern_tool(&mut self) {
        self.tool = Tool::Pattern;
        let mut state = PatternToolState::default();
        let edit = self.selected_pattern_meta();
        let mut kept: Vec<Selection> = Vec::new();
        for sel in self.selection.drain(..) {
            if let Some(norm) = Self::pattern_set_member(sel)
                && !kept.contains(&norm)
            {
                kept.push(norm);
            }
        }
        self.selection = kept;
        if let Some(mid) = edit {
            self.load_pattern_for_edit(&mut state, mid);
        }
        self.pattern_tool = Some(state);
        self.refresh_pattern_plan();
    }

    pub fn leave_pattern_tool(&mut self) {
        self.pattern_tool = None;
    }

    /// Whole-entity membership: endpoint hits resolve to their entity;
    /// constraints, dimensions and metas drop out.
    fn pattern_set_member(sel: Selection) -> Option<Selection> {
        match sel {
            Selection::Line(r) | Selection::LineP1(r) | Selection::LineP2(r) => Some(Selection::Line(r)),
            Selection::Arc(r) | Selection::ArcCenter(r) | Selection::ArcStart(r) | Selection::ArcEnd(r) => Some(Selection::Arc(r)),
            Selection::Point(r) => Some(Selection::Point(r)),
            _ => None,
        }
    }

    fn pattern_sources(&self) -> Vec<MetaEntity> {
        self.selection
            .iter()
            .filter_map(|s| match s {
                Selection::Line(r) => Some(MetaEntity::Line(*r)),
                Selection::Arc(r) => Some(MetaEntity::Arc(*r)),
                Selection::Point(r) => Some(MetaEntity::Point(*r)),
                _ => None,
            })
            .collect()
    }

    /// The pattern selected by its marker, or one of the selected
    /// entities is a copy of.
    fn selected_pattern_meta(&self) -> Option<u32> {
        for s in &self.selection {
            if let Selection::Meta(mid) = s
                && self.sketch.meta_index(*mid).is_some_and(|i| self.sketch.metas[i].as_pattern().is_some())
            {
                return Some(*mid);
            }
        }
        for e in self.pattern_sources() {
            if let Some(m) = arael_sketch_backend::meta::owner_of(&self.sketch, e)
                && m.as_pattern().is_some()
            {
                return Some(m.mid);
            }
        }
        None
    }

    fn load_pattern_for_edit(&mut self, state: &mut PatternToolState, mid: u32) {
        let Some(i) = self.sketch.meta_index(mid) else { return };
        let Some(p) = self.sketch.metas[i].as_pattern() else { return };
        let text = |v: &MetaValue| v.expr.clone().unwrap_or_else(|| trim_num(v.value));
        match &p.kind {
            PatternKind::Circular { center, distribution, angle, quantity, .. } => {
                state.kind = PatternToolKind::Circular;
                state.center = Some(*center);
                state.distribution = *distribution;
                state.angle = text(angle);
                state.quantity = quantity.to_string();
            }
            PatternKind::Rectangular { frame, extent, axis1, axis2 } => {
                state.kind = PatternToolKind::Rectangular;
                state.frame = *frame;
                state.extent = *extent;
                state.quantity1 = axis1.quantity.to_string();
                state.distance1 = text(&axis1.distance);
                state.symmetric1 = axis1.symmetric;
                state.quantity2 = axis2.quantity.to_string();
                state.distance2 = text(&axis2.distance);
                state.symmetric2 = axis2.symmetric;
            }
        }
        state.edit = Some(mid);
        self.select_pattern_copies(mid);
    }

    /// Show a pattern's copies as the selection.
    fn select_pattern_copies(&mut self, mid: u32) {
        let Some(i) = self.sketch.meta_index(mid) else { return };
        let Some(p) = self.sketch.metas[i].as_pattern() else { return };
        self.selection = p
            .copies
            .iter()
            .flat_map(|c| c.entities.iter())
            .map(|e| match e {
                MetaEntity::Line(l) => Selection::Line(*l),
                MetaEntity::Arc(a) => Selection::Arc(*a),
                MetaEntity::Point(p) => Selection::Point(*p),
            })
            .collect();
    }

    /// The parameters as typed, or why they do not parse.
    fn pattern_params(&self, state: &PatternToolState) -> Result<PatternParams, String> {
        let quantity = |s: &str| s.trim().parse::<u32>().map_err(|_| format!("quantity must be a whole number, got {}", s));
        let kind = match state.kind {
            PatternToolKind::Circular => {
                let center = state.center.ok_or("pick the center: a point, an endpoint or an arc center")?;
                let angle = if state.distribution == Distribution::Full {
                    MetaValue { value: 0.0, expr: None }
                } else {
                    pattern::parse_value(&self.sketch, &state.angle)?
                };
                PatternSpec::Circular { center, distribution: state.distribution, angle, quantity: quantity(&state.quantity)? }
            }
            PatternToolKind::Rectangular => {
                let axis = |q: &str, d: &str, sym: bool| -> Result<PatternAxis, String> {
                    let quantity = quantity(q)?;
                    let distance = if quantity >= 2 {
                        pattern::parse_value(&self.sketch, d)?
                    } else {
                        MetaValue { value: 0.0, expr: None }
                    };
                    Ok(PatternAxis { quantity, distance, symmetric: sym })
                };
                PatternSpec::Rectangular {
                    frame: state.frame,
                    extent: state.extent,
                    axis1: axis(&state.quantity1, &state.distance1, state.symmetric1)?,
                    axis2: axis(&state.quantity2, &state.distance2, state.symmetric2)?,
                }
            }
        };
        Ok(PatternParams { kind })
    }

    /// The sources under edit, or the selection.
    fn pattern_plan_sources(&self, state: &PatternToolState) -> Result<Vec<MetaEntity>, String> {
        if let Some(mid) = state.edit
            && let Some(i) = self.sketch.meta_index(mid)
            && let Some(p) = self.sketch.metas[i].as_pattern()
        {
            return Ok(p.sources.clone());
        }
        let set = self.pattern_sources();
        if set.is_empty() {
            return Err("click or box-select the lines, arcs and points to pattern; double-click walks a sequence".into());
        }
        Ok(set)
    }

    /// Recompute the preview plan from the selection and the window.
    pub fn refresh_pattern_plan(&mut self) {
        let Some(mut state) = self.pattern_tool.take() else { return };
        let result = self
            .pattern_plan_sources(&state)
            .and_then(|set| self.pattern_params(&state).map(|p| (set, p)))
            .and_then(|(set, p)| pattern::plan(&self.sketch, &set, &p));
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
        self.pattern_tool = Some(state);
    }

    /// Canvas input for `Tool::Pattern`: click toggles an entity into the
    /// set, double-click walks a sequence, a marquee adds; after a Pick
    /// button the next click sets the center / direction line; a marker
    /// click opens that pattern.
    pub(crate) fn handle_pattern_input(&mut self, ui: &egui::Ui, ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if self.pattern_tool.is_none() {
            self.pattern_tool = Some(PatternToolState::default());
        }
        let mut changed = false;
        let picking = self.pattern_tool.as_ref().and_then(|s| s.picking);

        if picking.is_none() {
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
                    self.leave_pattern_edit();
                    self.apply_box_select(start, end, true);
                    self.selection.retain(|s| matches!(s, Selection::Line(_) | Selection::Arc(_) | Selection::Point(_)));
                    changed = true;
                }
            }
        }

        if let Some(pick) = picking {
            if response.clicked_by(egui::PointerButton::Primary) {
                let hit = self.hit_test_selection(mouse_sketch, hit_threshold);
                let picked = match (pick, hit) {
                    (Pick::Center, Some(Selection::Point(p))) => Some(CenterRef::Point(p)),
                    (Pick::Center, Some(Selection::LineP1(l))) => Some(CenterRef::Endpoint(DimensionEndpoint::LineP1(l))),
                    (Pick::Center, Some(Selection::LineP2(l))) => Some(CenterRef::Endpoint(DimensionEndpoint::LineP2(l))),
                    (Pick::Center, Some(Selection::ArcCenter(a))) => Some(CenterRef::Endpoint(DimensionEndpoint::ArcCenter(a))),
                    (Pick::Center, Some(Selection::ArcStart(a))) => Some(CenterRef::Endpoint(DimensionEndpoint::ArcStart(a))),
                    (Pick::Center, Some(Selection::ArcEnd(a))) => Some(CenterRef::Endpoint(DimensionEndpoint::ArcEnd(a))),
                    _ => None,
                };
                let st = self.pattern_tool.as_mut().unwrap();
                match (pick, hit) {
                    (Pick::Center, _) if picked.is_some() => {
                        st.center = picked;
                        st.picking = None;
                        changed = true;
                    }
                    (Pick::Frame, Some(Selection::Line(l) | Selection::LineP1(l) | Selection::LineP2(l))) => {
                        st.frame = Some(l);
                        st.picking = None;
                        changed = true;
                    }
                    _ => {
                        self.status_error = Some(match pick {
                            Pick::Center => "pick a point, an endpoint or an arc center".into(),
                            Pick::Frame => "pick a line".into(),
                        });
                    }
                }
            }
        } else if Self::multi_clicked(response) {
            if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                if let Selection::Meta(mid) = sel {
                    self.open_pattern_edit_mid(mid);
                } else if let Some(norm) = Self::pattern_set_member(sel)
                    && let Selection::Line(_) | Selection::Arc(_) = norm
                {
                    let seed = match norm {
                        Selection::Line(r) => OffsetEntity::Line(r),
                        Selection::Arc(r) => OffsetEntity::Arc(r),
                        _ => unreachable!(),
                    };
                    self.leave_pattern_edit();
                    let seq = arael_sketch_backend::chain::walk(&self.sketch, seed);
                    for e in seq.entities() {
                        let s = match e {
                            OffsetEntity::Line(l) => Selection::Line(l),
                            OffsetEntity::Arc(a) => Selection::Arc(a),
                        };
                        if !self.selection.contains(&s) {
                            self.selection.push(s);
                        }
                    }
                }
                changed = true;
            }
        } else if response.clicked_by(egui::PointerButton::Primary) {
            let hit = self.hit_test_selection(mouse_sketch, hit_threshold);
            match (hit.and_then(Self::pattern_set_member), hit) {
                (Some(norm), _) => {
                    let e = match norm {
                        Selection::Line(r) => MetaEntity::Line(r),
                        Selection::Arc(r) => MetaEntity::Arc(r),
                        Selection::Point(r) => MetaEntity::Point(r),
                        _ => unreachable!(),
                    };
                    let owner = arael_sketch_backend::meta::owner_of(&self.sketch, e).filter(|m| m.as_pattern().is_some()).map(|m| m.mid);
                    match owner {
                        Some(mid) => { self.open_pattern_edit_mid(mid); }
                        None => {
                            self.leave_pattern_edit();
                            self.toggle_selection(norm);
                        }
                    }
                }
                (None, Some(Selection::Meta(mid))) => {
                    if !self.open_pattern_edit_mid(mid) {
                        self.leave_pattern_edit();
                        self.selection.clear();
                    }
                }
                (None, _) => {
                    self.leave_pattern_edit();
                    self.selection.clear();
                }
            }
            changed = true;
        }
        if changed {
            self.refresh_pattern_plan();
        }
    }

    /// Open a pattern in the tool for editing; false when `mid` is not a
    /// pattern.
    pub(crate) fn open_pattern_edit_mid(&mut self, mid: u32) -> bool {
        if !self.sketch.meta_index(mid).is_some_and(|i| self.sketch.metas[i].as_pattern().is_some()) {
            return false;
        }
        let mut state = self.pattern_tool.take().unwrap_or_default();
        if state.edit != Some(mid) {
            state = PatternToolState::default();
            self.load_pattern_for_edit(&mut state, mid);
        }
        self.pattern_tool = Some(state);
        true
    }

    fn leave_pattern_edit(&mut self) {
        if let Some(state) = self.pattern_tool.as_mut()
            && state.edit.is_some()
        {
            let mut fresh = PatternToolState::default();
            fresh.kind = state.kind;
            *state = fresh;
            self.selection.clear();
        }
    }

    /// Create the planned pattern (creation mode): the tool closes on the
    /// new copies, selected.
    pub fn apply_pattern(&mut self) {
        let Some(state) = self.pattern_tool.as_ref() else { return };
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
        match pattern::apply(self, &plan) {
            Ok(out) => {
                self.select_pattern_copies(out.mid);
                self.leave_pattern_tool();
                self.tool = Tool::Select;
            }
            Err(e) => self.status_error = Some(e),
        }
    }

    /// Apply the window's parameters to the pattern under edit.
    pub(crate) fn update_pattern(&mut self) {
        let Some(state) = self.pattern_tool.as_ref() else { return };
        let Some(mid) = state.edit else { return };
        let params = match self.pattern_params(state) {
            Ok(p) => p,
            Err(e) => { self.status_error = Some(e); return; }
        };
        self.status_error = None;
        match pattern::update(self, mid, &params) {
            Ok(_) => {
                if let Some(state) = self.pattern_tool.as_mut() {
                    state.pending_text = false;
                }
                let mut state = self.pattern_tool.take().unwrap_or_default();
                self.load_pattern_for_edit(&mut state, mid);
                self.pattern_tool = Some(state);
                self.refresh_pattern_plan();
            }
            Err(e) => {
                self.status_error = Some(e);
                self.select_pattern_copies(mid);
            }
        }
    }

    /// The tool window.
    pub fn render_pattern_window(&mut self, ctx: &egui::Context) {
        if self.tool != Tool::Pattern {
            return;
        }
        let Some(mut state) = self.pattern_tool.take() else { return };
        let title = match state.edit.and_then(|mid| self.sketch.meta_index(mid)) {
            Some(i) => format!("Pattern {}", self.sketch.metas[i].name),
            None => "Pattern".to_string(),
        };
        let mut changed = false;
        let mut apply = false;
        let mut close = false;
        let mut text_committed = false;
        let editing = state.edit.is_some();
        let center_label = match &state.center {
            Some(c) => pattern::center_name(&self.sketch, c),
            None => "(pick)".to_string(),
        };
        let frame_label = match state.frame {
            Some(l) => self.sketch.lines.get(l).map(|l| l.name.clone()).unwrap_or("?".into()),
            None => "none (right / up)".to_string(),
        };
        // A text field: returns (changed, committed by Enter).
        let text_field = |ui: &mut egui::Ui, text: &mut String, id: &str, width: f32| -> (bool, bool) {
            let r = ui.add(egui::TextEdit::singleline(text).desired_width(width).id(egui::Id::new(id)));
            (r.changed(), r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)))
        };
        egui::Window::new(title)
            .id(egui::Id::new("pattern_tool_window"))
            .default_pos(egui::pos2(self.canvas_rect.left() + 12.0, self.canvas_rect.top() + 12.0))
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.add_enabled_ui(!editing, |ui| {
                    ui.horizontal(|ui| {
                        changed |= ui.selectable_value(&mut state.kind, PatternToolKind::Circular, "Circular").changed();
                        changed |= ui.selectable_value(&mut state.kind, PatternToolKind::Rectangular, "Rectangular").changed();
                    });
                });
                match state.kind {
                    PatternToolKind::Circular => {
                        ui.horizontal(|ui| {
                            ui.label("Center");
                            ui.label(&center_label);
                            let picking = state.picking == Some(Pick::Center);
                            if ui.selectable_label(picking, "Pick").on_hover_text("Then click a point, an endpoint or an arc center").clicked() {
                                state.picking = if picking { None } else { Some(Pick::Center) };
                            }
                        });
                        ui.horizontal(|ui| {
                            changed |= ui.selectable_value(&mut state.distribution, Distribution::Full, "Full").changed();
                            changed |= ui.selectable_value(&mut state.distribution, Distribution::Partial, "Partial").changed();
                            changed |= ui.selectable_value(&mut state.distribution, Distribution::Symmetric, "Symmetric").changed();
                        });
                        ui.horizontal(|ui| {
                            ui.label("Quantity");
                            let (c, e) = text_field(ui, &mut state.quantity, "pattern_quantity", 50.0);
                            if c { changed = true; if editing { state.pending_text = true; } }
                            text_committed |= e;
                            if state.distribution != Distribution::Full {
                                ui.label("Angle");
                                let (c, e) = text_field(ui, &mut state.angle, "pattern_angle", 60.0);
                                if c { changed = true; if editing { state.pending_text = true; } }
                                text_committed |= e;
                                ui.label("deg");
                            }
                        });
                    }
                    PatternToolKind::Rectangular => {
                        ui.horizontal(|ui| {
                            ui.label("Direction");
                            ui.label(&frame_label);
                            let picking = state.picking == Some(Pick::Frame);
                            if ui.selectable_label(picking, "Pick").on_hover_text("Then click a line: axis 1 runs along it, axis 2 across it").clicked() {
                                state.picking = if picking { None } else { Some(Pick::Frame) };
                            }
                            if state.frame.is_some() && ui.button("None").clicked() {
                                state.frame = None;
                                changed = true;
                            }
                        });
                        ui.horizontal(|ui| {
                            changed |= ui.selectable_value(&mut state.extent, false, "Spacing").on_hover_text("The distance is between consecutive instances").changed();
                            changed |= ui.selectable_value(&mut state.extent, true, "Extent").on_hover_text("The distance is from the first instance to the last").changed();
                        });
                        for (label, q, d, sym, idq, idd) in [
                            ("Axis 1", &mut state.quantity1, &mut state.distance1, &mut state.symmetric1, "pattern_q1", "pattern_d1"),
                            ("Axis 2", &mut state.quantity2, &mut state.distance2, &mut state.symmetric2, "pattern_q2", "pattern_d2"),
                        ] {
                            ui.horizontal(|ui| {
                                ui.label(label);
                                ui.label("Quantity");
                                let (c, e) = text_field(ui, q, idq, 40.0);
                                if c { changed = true; if editing { state.pending_text = true; } }
                                text_committed |= e;
                                ui.label("Distance");
                                let (c, e) = text_field(ui, d, idd, 60.0);
                                if c { changed = true; if editing { state.pending_text = true; } }
                                text_committed |= e;
                                changed |= ui.selectable_value(sym, false, "One").changed();
                                changed |= ui.selectable_value(sym, true, "Symmetric").on_hover_text("Instances on both sides of the source").changed();
                            });
                        }
                    }
                }
                if state.plan.is_none()
                    && let Some(e) = &state.error
                {
                    ui.colored_label(self.colors.error_text, e.as_str());
                }
                ui.horizontal(|ui| {
                    if editing {
                        ui.label("changes apply as you make them");
                        if state.pending_text && ui.button("Apply").clicked() {
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
        // Enter in a number field applies / creates.
        if text_committed {
            if editing { changed = true; } else { apply = true; }
        }
        self.pattern_tool = Some(state);
        if changed {
            self.refresh_pattern_plan();
            if editing && (text_committed || !self.pattern_tool.as_ref().is_some_and(|s| s.pending_text)) {
                self.update_pattern();
            }
        }
        if apply {
            self.apply_pattern();
        }
        if close {
            self.leave_pattern_edit();
            self.selection.clear();
            self.tool = Tool::Select;
            self.leave_pattern_tool();
        }
    }

    /// The preview: every planned copy, dashed. In edit mode only while a
    /// typed number is not yet applied.
    pub fn draw_pattern_preview(&self, painter: &egui::Painter) {
        let Some(state) = self.pattern_tool.as_ref() else { return };
        if state.edit.is_some() && !state.pending_text {
            return;
        }
        let Some(plan) = state.plan.as_ref() else { return };
        let stroke = egui::Stroke::new(1.5, self.colors.offset_preview);
        for poly in pattern::preview_polylines(plan, 48) {
            let pts: Vec<egui::Pos2> = poly.into_iter().map(|p| self.to_screen(p)).collect();
            if pts.len() == 1 {
                painter.circle_stroke(pts[0], 3.0, stroke);
            } else {
                crate::drawing::draw_styled_polyline(painter, &pts, stroke, LineStyle::Dashed);
            }
        }
    }
}

fn trim_num(v: f64) -> String {
    let s = format!("{:.4}", v);
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() { "0".into() } else { s.to_string() }
}
