// The Mirror tool: mirrored copies of the selected entities across an
// axis line. The set is the selection (click, box, double-click walk);
// the axis is picked with the Pick button; Create runs the engine
// (arael_sketch_backend::mirror), which holds the copies to their
// sources with symmetry constraints. Fire-and-forget: no meta record,
// undo removes the whole mirror.

use eframe::egui;
use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;
use arael_sketch_backend::Selection;
use arael_sketch_backend::mirror::{self, MirrorParams, MirrorPlan};
use crate::EditorApp;
use crate::tools::Tool;

/// The tool's session: the axis, the options, the plan.
pub struct MirrorToolState {
    pub axis: Option<Ref<Line>>,
    /// The Pick button was pressed: the next click sets the axis.
    pub picking: bool,
    /// Recreate coincidences and add symmetry constraints (default).
    pub constraints: bool,
    /// The current plan (for the preview) or why there is none.
    pub plan: Option<MirrorPlan>,
    pub error: Option<String>,
}

impl Default for MirrorToolState {
    fn default() -> Self {
        MirrorToolState { axis: None, picking: false, constraints: true, plan: None, error: None }
    }
}

impl EditorApp {
    /// Enter the tool: the selection (lines, arcs, points) is the set.
    pub fn enter_mirror_tool(&mut self) {
        self.tool = Tool::Mirror;
        let mut kept: Vec<Selection> = Vec::new();
        for sel in self.selection.drain(..) {
            if let Some(norm) = Self::pattern_set_member(sel)
                && !kept.contains(&norm)
            {
                kept.push(norm);
            }
        }
        self.selection = kept;
        self.mirror_tool = Some(MirrorToolState::default());
        self.refresh_mirror_plan();
    }

    pub fn leave_mirror_tool(&mut self) {
        self.mirror_tool = None;
    }

    /// Recompute the preview plan from the selection and the window.
    pub fn refresh_mirror_plan(&mut self) {
        let Some(mut state) = self.mirror_tool.take() else { return };
        let result = (|| {
            let Some(axis) = state.axis else {
                return Err("pick the axis line to mirror about".to_string());
            };
            let set = self.pattern_sources();
            if set.is_empty() {
                return Err("click or box-select the lines, arcs and points to mirror; double-click walks a sequence".into());
            }
            let params = MirrorParams { axis, noconstraint: !state.constraints, strict: false };
            mirror::plan(&self.sketch, &set, &params)
        })();
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
        self.mirror_tool = Some(state);
    }

    /// Canvas input for `Tool::Mirror`: click toggles an entity into
    /// the set, double-click walks a sequence, a marquee adds; after
    /// Pick the next click sets the axis line.
    pub(crate) fn handle_mirror_input(&mut self, ui: &egui::Ui, ctx: &egui::Context, response: &egui::Response, mouse_screen: egui::Pos2, mouse_sketch: vect2d, hit_threshold: f64) {
        if self.mirror_tool.is_none() {
            self.mirror_tool = Some(MirrorToolState::default());
        }
        let mut changed = false;
        let picking = self.mirror_tool.as_ref().is_some_and(|s| s.picking);

        if !picking {
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
                    self.selection.retain(|s| matches!(s, Selection::Line(_) | Selection::Arc(_) | Selection::Point(_)));
                    changed = true;
                }
            }
        }

        if picking {
            if response.clicked_by(egui::PointerButton::Primary) {
                let hit = self.hit_test_selection(mouse_sketch, hit_threshold);
                if let Some(Selection::Line(l) | Selection::LineP1(l) | Selection::LineP2(l)) = hit {
                    let st = self.mirror_tool.as_mut().unwrap();
                    st.axis = Some(l);
                    st.picking = false;
                    changed = true;
                } else {
                    self.status_error = Some("pick a line to mirror about".into());
                }
            }
        } else if Self::multi_clicked(response) {
            if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                if let Some(norm) = Self::pattern_set_member(sel)
                    && let Selection::Line(_) | Selection::Arc(_) = norm
                {
                    let seed = match norm {
                        Selection::Line(r) => OffsetEntity::Line(r),
                        Selection::Arc(r) => OffsetEntity::Arc(r),
                        _ => unreachable!(),
                    };
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
            match self.hit_test_selection(mouse_sketch, hit_threshold).and_then(Self::pattern_set_member) {
                Some(norm) => self.toggle_selection(norm),
                None => self.selection.clear(),
            }
            changed = true;
        }
        if changed {
            self.refresh_mirror_plan();
        }
    }

    /// Create the planned mirror, select the copies, back to Select.
    pub fn apply_mirror(&mut self) {
        let Some(state) = self.mirror_tool.as_ref() else { return };
        let Some(plan) = state.plan.clone() else {
            if let Some(e) = state.error.clone() {
                self.status_error = Some(e);
            }
            return;
        };
        self.status_error = None;
        match mirror::apply(self, &plan) {
            Ok(out) => {
                self.selection = out.copies.iter().filter_map(|e| match e {
                    MetaEntity::Line(r) => Some(Selection::Line(*r)),
                    MetaEntity::Arc(r) => Some(Selection::Arc(*r)),
                    MetaEntity::Point(r) => Some(Selection::Point(*r)),
                }).collect();
                if !out.warnings.is_empty() {
                    self.status_notice = Some(out.warnings.join("; "));
                }
                self.leave_mirror_tool();
                self.tool = Tool::Select;
            }
            Err(e) => self.status_error = Some(e),
        }
    }

    /// The tool window.
    pub fn render_mirror_window(&mut self, ctx: &egui::Context) {
        if self.tool != Tool::Mirror {
            return;
        }
        let Some(mut state) = self.mirror_tool.take() else { return };
        let mut changed = false;
        let mut apply = false;
        let mut close = false;
        let axis_label = match state.axis {
            Some(l) => self.sketch.lines.get(l).map(|l| l.name.clone()).unwrap_or("?".into()),
            None => "(pick)".to_string(),
        };
        egui::Window::new("Mirror")
            .id(egui::Id::new("mirror_tool_window"))
            .default_pos(egui::pos2(self.canvas_rect.left() + 12.0, self.canvas_rect.top() + 12.0))
            .resizable(false)
            .collapsible(false)
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Axis");
                    ui.label(&axis_label);
                    if ui.selectable_label(state.picking, "Pick").on_hover_text("Then click the line to mirror about").clicked() {
                        state.picking = !state.picking;
                    }
                });
                ui.horizontal(|ui| {
                    changed |= ui.selectable_value(&mut state.constraints, true, "Constraints")
                        .on_hover_text("Recreate the set's coincidences among the copies and hold every copy to its source with symmetry constraints").changed();
                    changed |= ui.selectable_value(&mut state.constraints, false, "Free copies")
                        .on_hover_text("Bare geometry, no constraints").changed();
                });
                if state.plan.is_none()
                    && let Some(e) = &state.error
                {
                    ui.colored_label(self.colors.error_text, e.as_str());
                }
                ui.horizontal(|ui| {
                    if ui.add_enabled(state.plan.is_some(), egui::Button::new("Create")).clicked() {
                        apply = true;
                    }
                    if ui.button("Cancel").clicked() {
                        close = true;
                    }
                });
            });
        self.mirror_tool = Some(state);
        if changed {
            self.refresh_mirror_plan();
        }
        if apply {
            self.apply_mirror();
        }
        if close {
            self.selection.clear();
            self.tool = Tool::Select;
            self.leave_mirror_tool();
        }
    }

    /// The preview: every planned copy, dashed.
    pub fn draw_mirror_preview(&self, painter: &egui::Painter) {
        let Some(state) = self.mirror_tool.as_ref() else { return };
        let Some(plan) = state.plan.as_ref() else { return };
        let stroke = egui::Stroke::new(1.5, self.colors.offset_preview);
        for poly in mirror::preview_polylines(&self.sketch, plan, 48) {
            let pts: Vec<egui::Pos2> = poly.into_iter().map(|p| self.to_screen(p)).collect();
            if pts.len() == 1 {
                painter.circle_stroke(pts[0], 3.0, stroke);
            } else {
                crate::drawing::draw_styled_polyline(painter, &pts, stroke, LineStyle::Dashed);
            }
        }
    }
}
