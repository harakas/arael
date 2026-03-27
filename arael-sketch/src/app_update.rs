// eframe::App implementation for EditorApp.

use eframe::egui;
use arael::utils::rad2rad;
use arael::vect::vect2d;
use arael::refs::Ref;
use arael_sketch_solver::*;
use crate::colors::ColorScheme;
use crate::tools::*;
use crate::actions::Action;
use crate::geometry::*;
use crate::{EditorApp, spawn_async};

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll background DOF computation
        self.poll_dof();

        // Check for pending file load from async dialog
        let pending_json = self.pending_load.lock().unwrap().take();
        if let Some(json) = pending_json {
            self.load_from_json(&json);
        }

        // Apply egui visuals for widgets (side panel, buttons, etc.)
        ctx.set_visuals(if self.dark_mode { egui::Visuals::dark() } else { egui::Visuals::light() });

        // Side panel: toolbar
        egui::SidePanel::left("toolbar").min_width(140.0).show(ctx, |ui| {
            // Theme toggle
            ui.horizontal(|ui| {
                let theme_label = if self.dark_mode { "Light" } else { "Dark" };
                if ui.button(theme_label).clicked() {
                    self.dark_mode = !self.dark_mode;
                    self.colors = if self.dark_mode { ColorScheme::dark() } else { ColorScheme::light() };
                }
                let constr_label = if self.show_constraints { "Hide Cstr" } else { "Show Cstr" };
                if ui.button(constr_label).clicked() {
                    self.show_constraints = !self.show_constraints;
                }
            });
            ui.separator();

            ui.heading("Tools");
            ui.separator();
            if ui.selectable_label(self.tool == Tool::Select, "Select (S)").clicked() {
                self.tool = Tool::Select;
            }
            if ui.selectable_label(self.tool == Tool::DrawPoint, "Point (P)").clicked() {
                self.tool = Tool::DrawPoint;
            }
            if ui.selectable_label(self.tool == Tool::DrawLine, "Line (L)").clicked() {
                self.tool = Tool::DrawLine;
                self.line_draw = None;
            }
            if ui.selectable_label(self.tool == Tool::DrawCircle, "Circle (O)").clicked() {
                self.tool = Tool::DrawCircle;
                self.circle_draw = None;
            }
            if ui.selectable_label(self.tool == Tool::DrawArc, "Arc (A)").clicked() {
                self.tool = Tool::DrawArc;
                self.arc_draw = None;
            }

            ui.separator();
            ui.heading("Constraints");
            ui.separator();

            let constraint_btn = |ui: &mut egui::Ui, this: &mut EditorApp, ct: ConstraintType, label: &str| {
                let active = matches!(this.tool, Tool::ConstraintMode(t) if t == ct);
                let can_apply = this.can_apply_constraint(ct);
                let can_enter = this.could_enter_constraint_mode(ct);
                let enabled = can_apply || can_enter;
                let btn = egui::Button::new(label).selected(active);
                if ui.add_enabled(enabled, btn).clicked() {
                    this.try_apply_or_enter_mode(ct);
                }
            };
            constraint_btn(ui, self, ConstraintType::Horizontal, "Horizontal (H)");
            constraint_btn(ui, self, ConstraintType::Vertical, "Vertical (V)");
            constraint_btn(ui, self, ConstraintType::Coincident, "Coincident (C)");
            constraint_btn(ui, self, ConstraintType::Parallel, "Parallel");
            constraint_btn(ui, self, ConstraintType::Perpendicular, "Perpendicular");
            constraint_btn(ui, self, ConstraintType::EqualLength, "Equal (=)");
            constraint_btn(ui, self, ConstraintType::Tangent, "Tangent (T)");
            constraint_btn(ui, self, ConstraintType::Collinear, "Collinear");
            constraint_btn(ui, self, ConstraintType::Midpoint, "Midpoint (M)");
            constraint_btn(ui, self, ConstraintType::Symmetry, "Symmetry (S)");
            constraint_btn(ui, self, ConstraintType::Lock, "Lock (K)");
            constraint_btn(ui, self, ConstraintType::ToggleStyle, "Style (X)");

            ui.separator();
            let dim_active = matches!(self.tool, Tool::Dimension);
            let dim_btn = egui::Button::new("Dimension (D)").selected(dim_active);
            if ui.add(dim_btn).clicked() {
                self.tool = Tool::Dimension;
                self.dim_editing = false;
                self.dim_kind = None;
            }

            // Dimension value input
            if self.dim_editing {
                ui.separator();
                let label = if self.dim_edit_index.is_some() { "Edit dimension:" } else { "New dimension:" };
                ui.label(label);
                let response = ui.text_edit_singleline(&mut self.dim_input);
                // Select all text when entering edit mode (one-shot flag)
                if self.dim_select_all && response.has_focus() {
                    self.dim_select_all = false;
                    let mut state = egui::TextEdit::load_state(ui.ctx(), response.id).unwrap_or_default();
                    state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                        egui::text::CCursor::new(0),
                        egui::text::CCursor::new(self.dim_input.len()),
                    )));
                    egui::TextEdit::store_state(ui.ctx(), response.id, state);
                }
                let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
                if enter_pressed || (response.lost_focus() && enter_pressed) {
                    let input = self.dim_input.trim().to_string();
                    let is_numeric = input.parse::<f64>().is_ok();
                    let is_expr = !is_numeric && arael_sym::parse(&input).is_ok();

                    if is_numeric || is_expr {
                        self.begin_group();
                        if let Some(edit_idx) = self.dim_edit_index.take() {
                            // Editing existing: update in place (preserves name)
                            if is_numeric {
                                let value = input.parse::<f64>().unwrap();
                                self.exec(Action::UpdateDimension { index: edit_idx, value, expr: None });
                            } else {
                                self.exec(Action::UpdateDimension {
                                    index: edit_idx, value: 0.0,
                                    expr: Some(input.clone()),
                                });
                            }
                        } else if let Some(kind) = self.dim_kind.take() {
                            let is_dup = self.sketch.dimensions.iter().any(|d| d.kind == kind);
                            if is_dup {
                                self.status_error = Some("Dimension already exists".into());
                            } else if is_numeric {
                                let value = input.parse::<f64>().unwrap();
                                let n_dims_before = self.sketch.dimensions.len();
                                self.exec(Action::AddDimension { kind, value });
                                if self.sketch.dimensions.len() > n_dims_before {
                                    if let Some(d) = self.sketch.dimensions.last_mut() {
                                        d.offset = self.dim_offset;
                                        d.text_along = self.dim_text_along;
                                    }
                                }
                            } else {
                                if let Err(e) = self.sketch.add_expr_dimension(
                                    kind, &input, self.dim_offset, self.dim_text_along) {
                                    self.status_error = Some(format!("Expression error: {}", e));
                                } else {
                                    self.sketch.solve();
                                    self.last_cost = 0.0; // will be updated
                                }
                            }
                        }
                    } else if !input.is_empty() {
                        self.status_error = Some(format!("Invalid value or expression: {}", input));
                    }
                    self.dim_editing = false;
                    self.dim_placing = false;
                    self.dim_edit_index = None;
                    self.selection.clear();
                } else if !response.has_focus() && self.dim_editing {
                    response.request_focus();
                }
            }

            ui.separator();
            ui.heading("File");
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    if let Ok(json) = serde_json::to_string_pretty(&self.sketch) {
                        let json_bytes = json.into_bytes();
                        spawn_async(async move {
                            if let Some(handle) = rfd::AsyncFileDialog::new()
                                .add_filter("Sketch JSON", &["json"])
                                .set_file_name("sketch.json")
                                .save_file().await
                            {
                                let _ = handle.write(&json_bytes).await;
                            }
                        });
                    }
                }
                if ui.button("Open").clicked() {
                    let pending = self.pending_load.clone();
                    spawn_async(async move {
                        if let Some(file) = rfd::AsyncFileDialog::new()
                            .add_filter("Sketch JSON", &["json"])
                            .pick_file().await
                        {
                            let data = file.read().await;
                            if let Ok(json) = String::from_utf8(data) {
                                *pending.lock().unwrap() = Some(json);
                            }
                        }
                    });
                }
            });

            ui.separator();
            ui.heading("History");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.add_enabled(self.history.can_undo(), egui::Button::new("Undo")).clicked() {
                    if let Some(restored) = self.history.undo() {
                        self.sketch = restored;
                        self.selection.clear();
                        self.update_cost();
                        self.compute_dof_async();
                    }
                }
                if ui.add_enabled(self.history.can_redo(), egui::Button::new("Redo")).clicked() {
                    if let Some(restored) = self.history.redo() {
                        self.sketch = restored;
                        self.selection.clear();
                        self.update_cost();
                        self.compute_dof_async();
                    }
                }
            });
            ui.label(format!("Actions: {}/{}", self.history.cursor, self.history.actions.len()));

            ui.separator();
            ui.label(format!("Points: {}  Lines: {}  Arcs: {}",
                self.sketch.points.len(), self.sketch.lines.len(), self.sketch.arcs.len()));
            if !self.selection.is_empty() {
                let names: Vec<String> = self.selection.iter().filter_map(|s| {
                    match *s {
                        Selection::Point(r) => Some(self.sketch.points[r].name.clone()),
                        Selection::Line(r) => Some(self.sketch.lines[r].name.clone()),
                        Selection::LineP1(r) => Some(format!("{}.p1", self.sketch.lines[r].name)),
                        Selection::LineP2(r) => Some(format!("{}.p2", self.sketch.lines[r].name)),
                        Selection::Arc(r) => Some(self.sketch.arcs[r].name.clone()),
                        Selection::ArcCenter(r) => Some(format!("{}.c", self.sketch.arcs[r].name)),
                        Selection::ArcStart(r) => Some(format!("{}.s", self.sketch.arcs[r].name)),
                        Selection::ArcEnd(r) => Some(format!("{}.e", self.sketch.arcs[r].name)),
                        Selection::Constraint(id) => Some(self.describe_constraint(id)),
                        Selection::Dimension(i) => {
                            if i < self.sketch.dimensions.len() {
                                let d = &self.sketch.dimensions[i];
                                Some(format!("{} = {:.2}", d.name, d.value))
                            } else { Some("dim?".to_string()) }
                        }
                    }
                }).collect();
                ui.label(format!("Selected: {}", names.join(", ")));
            }

            // Constraint conflict error message
            if let Some(ref err) = self.status_error {
                ui.separator();
                ui.colored_label(egui::Color32::from_rgb(255, 80, 80), err.as_str());
            }
        });

        // Central panel: canvas
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::click_and_drag(),
            );
            let rect = response.rect;

            // Auto-fit after file load
            if self.pending_fit {
                self.fit_all(rect);
                self.pending_fit = false;
            }

            // Keyboard shortcuts (skip when editing dimension text)
            if !self.dim_editing {
            if ui.input(|i| i.key_pressed(egui::Key::S) && !i.modifiers.ctrl && !i.modifiers.mac_cmd) { self.tool = Tool::Select; }
            if ui.input(|i| i.key_pressed(egui::Key::P)) { self.tool = Tool::DrawPoint; }
            if ui.input(|i| i.key_pressed(egui::Key::L)) {
                self.tool = Tool::DrawLine;
                self.line_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::O) && !i.modifiers.ctrl && !i.modifiers.mac_cmd) {
                self.tool = Tool::DrawCircle;
                self.circle_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::A)) {
                self.tool = Tool::DrawArc;
                self.arc_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::H)) { self.try_apply_or_enter_mode(ConstraintType::Horizontal); }
            if ui.input(|i| i.key_pressed(egui::Key::V)) { self.try_apply_or_enter_mode(ConstraintType::Vertical); }
            if ui.input(|i| i.key_pressed(egui::Key::C)) { self.try_apply_or_enter_mode(ConstraintType::Coincident); }
            if ui.input(|i| i.key_pressed(egui::Key::K)) { self.try_apply_or_enter_mode(ConstraintType::Lock); }
            if ui.input(|i| i.key_pressed(egui::Key::T)) { self.try_apply_or_enter_mode(ConstraintType::Tangent); }
            if ui.input(|i| i.key_pressed(egui::Key::M)) { self.try_apply_or_enter_mode(ConstraintType::Midpoint); }
            if ui.input(|i| i.key_pressed(egui::Key::S)) { self.try_apply_or_enter_mode(ConstraintType::Symmetry); }
            if ui.input(|i| i.key_pressed(egui::Key::X)) { self.try_apply_or_enter_mode(ConstraintType::ToggleStyle); }
            if ui.input(|i| i.key_pressed(egui::Key::D) && !i.modifiers.ctrl && !i.modifiers.mac_cmd) {
                self.tool = Tool::Dimension;
                self.dim_editing = false;
                self.dim_kind = None;
            }
            } // !self.dim_editing
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.selection.clear();
                self.line_draw = None;
                self.circle_draw = None;
                self.arc_draw = None;
                self.dim_editing = false;
                self.dim_kind = None;
                self.dim_placing = false;
                self.dim_edit_index = None;
                self.status_error = None;
                self.tool = Tool::Select;
            }

            // Delete selected entities/constraints with Backspace/Delete
            // (skip when editing dimension text — Backspace edits the text field)
            if !self.dim_editing && ui.input(|i| i.key_pressed(egui::Key::Backspace) || i.key_pressed(egui::Key::Delete)) {
                let sel = self.selection.clone();
                if !sel.is_empty() {
                    self.begin_group();
                    for s in &sel {
                        match *s {
                            Selection::Point(r) => { self.exec(Action::DeletePoint { point: r }); }
                            Selection::Line(r) => { self.exec(Action::DeleteLine { line: r }); }
                            Selection::Arc(r) => { self.exec(Action::DeleteArc { arc: r }); }
                            Selection::Constraint(id) => { self.delete_constraint(id); }
                            Selection::Dimension(i) => { self.exec(Action::RemoveDimension { index: i }); }
                            _ => {} // endpoints aren't deletable on their own
                        }
                    }
                    self.selection.clear();
                }
            }

            // Undo/redo keyboard shortcuts
            let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.mac_cmd);
            let shift = ui.input(|i| i.modifiers.shift);
            if ctrl && shift && ui.input(|i| i.key_pressed(egui::Key::Z)) {
                if let Some(restored) = self.history.redo() {
                    self.sketch = restored;
                    self.selection.clear();
                    self.update_cost();
                    self.compute_dof_async();
                }
            } else if ctrl && ui.input(|i| i.key_pressed(egui::Key::Z)) {
                if let Some(restored) = self.history.undo() {
                    self.sketch = restored;
                    self.selection.clear();
                    self.update_cost();
                    self.compute_dof_async();
                }
            }
            if ctrl && ui.input(|i| i.key_pressed(egui::Key::S)) {
                if let Ok(json) = serde_json::to_string_pretty(&self.sketch) {
                    let json_bytes = json.into_bytes();
                    spawn_async(async move {
                        if let Some(handle) = rfd::AsyncFileDialog::new()
                            .add_filter("Sketch JSON", &["json"])
                            .set_file_name("sketch.json")
                            .save_file().await
                        {
                            let _ = handle.write(&json_bytes).await;
                        }
                    });
                }
            }
            if ctrl && ui.input(|i| i.key_pressed(egui::Key::O)) {
                let pending = self.pending_load.clone();
                spawn_async(async move {
                    if let Some(file) = rfd::AsyncFileDialog::new()
                        .add_filter("Sketch JSON", &["json"])
                        .pick_file().await
                    {
                        let data = file.read().await;
                        if let Ok(json) = String::from_utf8(data) {
                            *pending.lock().unwrap() = Some(json);
                        }
                    }
                });
            }

            // Zoom (scroll wheel)
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                let zoom_factor = if scroll > 0.0 { 1.1 } else { 1.0 / 1.1 };
                if let Some(mouse) = ui.input(|i| i.pointer.hover_pos()) {
                    // Zoom toward mouse position
                    let before = self.to_sketch(mouse);
                    self.scale *= zoom_factor;
                    self.scale = self.scale.clamp(1e-4, 1e7);
                    let after = self.to_screen(before);
                    self.offset += mouse - after;
                }
            }

            // Pan (middle mouse drag) / Fit all (middle double-click)
            if response.double_clicked_by(egui::PointerButton::Middle) {
                self.fit_all(rect);
            } else if response.dragged_by(egui::PointerButton::Middle) {
                self.offset += response.drag_delta();
            }

            // Get mouse position in sketch coords
            let mouse_screen = response.hover_pos().unwrap_or(egui::Pos2::ZERO);
            let mouse_sketch = self.to_sketch(mouse_screen);
            let hit_threshold = 15.0 / self.scale as f64;  // screen pixels -> sketch units

            // Tool-specific input handling
            match self.tool {
                Tool::Select => {
                    // Double-click on dimension to edit value
                    if response.double_clicked_by(egui::PointerButton::Primary) {
                        let mut edited = false;
                        for (i, dim) in self.sketch.dimensions.iter().enumerate() {
                            let (ts, te) = self.dim_text_segment(dim);
                            let d = Self::screen_point_to_segment_dist(mouse_screen, ts, te);
                            if d < 15.0 {
                                self.dim_input = if let Some(ref expr) = dim.expr_str {
                                    expr.clone()
                                } else {
                                    format!("{:.4}", dim.value)
                                };
                                self.dim_kind = Some(dim.kind.clone());
                                self.dim_offset = dim.offset;
                                self.dim_edit_index = Some(i);
                                self.dim_editing = true;
                                self.dim_select_all = true;
                                self.dim_placing = false;
                                self.tool = Tool::Dimension;
                                self.selection.clear();
                                self.selection.push(Selection::Dimension(i));
                                edited = true;
                                break;
                            }
                        }
                        if !edited {
                            // Normal double-click behavior (if any)
                        }
                    }

                    // Drag: geometry or dimension
                    if response.dragged_by(egui::PointerButton::Primary) {
                        if self.grab.is_none() && self.drag_dimension.is_none() {
                            // First drag frame: try to grab a dimension first, then geometry
                            let mut grabbed_dim = false;
                            if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                                if let Selection::Dimension(i) = sel {
                                    self.drag_dimension = Some(i);
                                    grabbed_dim = true;
                                }
                            }
                            if !grabbed_dim {
                                if let Some(target) = self.hit_test(mouse_sketch, hit_threshold) {
                                    self.start_drag(target, mouse_sketch);
                                }
                            }
                        }
                        if let Some(dim_idx) = self.drag_dimension {
                            // Update dimension offset and text_along from mouse
                            if dim_idx < self.sketch.dimensions.len() {
                                let kind = self.sketch.dimensions[dim_idx].kind.clone();
                                let is_radius = matches!(kind, DimensionKind::ArcRadius(_));
                                if is_radius {
                                    if let DimensionKind::ArcRadius(r) = kind {
                                        let a = &self.sketch.arcs[r];
                                        let angle = (mouse_sketch.y - a.center.value.y)
                                            .atan2(mouse_sketch.x - a.center.value.x);
                                        self.sketch.dimensions[dim_idx].offset = vect2d::new(angle, 0.0);
                                    }
                                } else if let DimensionKind::Angle(a, b, sup) = kind {
                                    // Drag existing: lock to 2 opposing sectors (same supplement)
                                    let la = &self.sketch.lines[a];
                                    let lb = &self.sketch.lines[b];
                                    let ix = line_line_intersection(
                                        la.p1.value, la.p2.value, lb.p1.value, lb.p2.value);
                                    let dist = ((mouse_sketch.x - ix.x).powi(2)
                                        + (mouse_sketch.y - ix.y).powi(2)).sqrt();
                                    let mouse_angle = (mouse_sketch.y - ix.y).atan2(mouse_sketch.x - ix.x);
                                    let sector_mid = self.angle_dim_opposing_sector(a, b, sup, mouse_angle);
                                    let new_offset = vect2d::new(sector_mid, dist.max(0.3));
                                    // Compute text_along from mouse angle relative to sector
                                    let (_ix, start, sweep) = self.angle_dim_sector(a, b, sup, new_offset);
                                    let delta = rad2rad(mouse_angle - start);
                                    let along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                                    self.sketch.dimensions[dim_idx].offset = new_offset;
                                    self.sketch.dimensions[dim_idx].text_along = along;
                                } else {
                                    // Decompose mouse into perpendicular offset and along-line position
                                    let (p1, p2) = self.dim_endpoints(&kind);
                                    let ddx = p2.x - p1.x;
                                    let ddy = p2.y - p1.y;
                                    let len = (ddx * ddx + ddy * ddy).sqrt().max(1e-12);
                                    let ux = ddx / len;
                                    let uy = ddy / len;
                                    let nx = -ddy / len;
                                    let ny = ddx / len;
                                    let mx = (p1.x + p2.x) / 2.0;
                                    let my = (p1.y + p2.y) / 2.0;
                                    let rel_x = mouse_sketch.x - mx;
                                    let rel_y = mouse_sketch.y - my;
                                    let perp = rel_x * nx + rel_y * ny;
                                    let along = (rel_x * ux + rel_y * uy) / len;
                                    self.sketch.dimensions[dim_idx].offset = vect2d::new(0.0, perp);
                                    self.sketch.dimensions[dim_idx].text_along = along;
                                }
                            }
                            ctx.request_repaint();
                        }
                        if self.grab.is_some() {
                            self.update_drag(mouse_sketch);
                            ctx.request_repaint();
                        }
                    }

                    // End drag
                    if response.drag_stopped_by(egui::PointerButton::Primary) {
                        if self.grab.is_some() {
                            self.end_drag(hit_threshold);
                        }
                        self.drag_dimension = None;
                    }

                    // Click (no drag): select/deselect
                    if response.clicked_by(egui::PointerButton::Primary) {
                        if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                            self.toggle_selection(sel);
                        } else {
                            self.selection.clear();
                        }
                    }
                }

                Tool::DrawPoint => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.begin_group();
                        let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                        let pos = snap.map_or(mouse_sketch, |(p, _)| p);
                        let action = Action::AddPoint { pos };
                        self.exec(action);
                        let new_point = Ref::new(self.sketch.points.slot_count() as u32 - 1);
                        if let Some((_, snap_target)) = snap {
                            self.apply_snap_coincident_point(snap_target, new_point);
                        }
                    }
                }

                Tool::DrawLine => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.begin_group();
                        if let Some(state) = self.line_draw.take() {
                            // Second click: finish line
                            // Snap end point to nearby entity
                            let end_snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let end_pos = end_snap.map_or(mouse_sketch, |(pos, _)| pos);

                            let action = Action::AddLine { p1: state.start, p2: end_pos };
                            self.exec(action);
                            let new_line = Ref::new(self.sketch.lines.slot_count() as u32 - 1);

                            // Auto-coincident for start snap
                            if let Some(snap) = state.snap_start {
                                self.apply_snap_coincident(snap, new_line, true);
                            }
                            // Auto-coincident for end snap
                            if let Some((_, snap)) = end_snap {
                                self.apply_snap_coincident(snap, new_line, false);
                            }

                            // Chain: start next line from end of this one
                            self.line_draw = Some(LineDrawState {
                                start: end_pos,
                                snap_start: Some(SnapTarget::LineP2(new_line)),
                            });
                        } else {
                            // First click: start line, snap to nearby entity
                            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let start_pos = snap.map_or(mouse_sketch, |(pos, _)| pos);
                            self.line_draw = Some(LineDrawState {
                                start: start_pos,
                                snap_start: snap.map(|(_, t)| t),
                            });
                        }
                    }
                }

                Tool::DrawCircle => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.begin_group();
                        if let Some(state) = self.circle_draw.take() {
                            // Second click: edge point
                            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let edge = snap.map_or(mouse_sketch, |(p, _)| p);
                            self.exec(Action::AddCircle { center: state.center, edge });
                            let new_arc = Ref::new(self.sketch.arcs.slot_count() as u32 - 1);

                            // Auto-coincident for center
                            if let Some(s) = state.snap_center {
                                self.apply_snap_coincident_arc(s, new_arc, ArcPoint::Center, state.center);
                            }
                            // Auto-coincident for edge (point on circle)
                            if let Some((_, s)) = snap {
                                // Edge point is "on arc" - use a helper point for PointOnArc
                                self.exec(Action::AddHelperPoint { pos: edge });
                                let helper = Ref::new(self.sketch.points.slot_count() as u32 - 1);
                                self.exec(Action::ApplyPointOnArc { point: helper, arc: new_arc });
                                self.apply_snap_coincident_point(s, helper);
                            }
                        } else {
                            // First click: center
                            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let center = snap.map_or(mouse_sketch, |(p, _)| p);
                            self.circle_draw = Some(CircleDrawState {
                                center,
                                snap_center: snap.map(|(_, t)| t),
                            });
                        }
                    }
                }

                Tool::DrawArc => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.begin_group();
                        let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                        let pos = snap.map_or(mouse_sketch, |(p, _)| p);
                        let snap_target = snap.map(|(_, t)| t);

                        if let Some(state) = self.arc_draw.take() {
                            if let Some((end, snap_end)) = state.end {
                                // Third click: mid point on arc, create it
                                let swapped = circumscribed_arc(state.start, end, pos)
                                    .map_or(false, |(_, _, _, _, s)| s);
                                self.exec(Action::AddArc { start: state.start, end, mid: pos, swapped });
                                let new_arc = Ref::new(self.sketch.arcs.slot_count() as u32 - 1);

                                // When swapped, arc.start_angle corresponds to `end` click
                                // and arc.end_angle corresponds to `start` click
                                let (start_ap, end_ap) = if swapped {
                                    (ArcPoint::End, ArcPoint::Start)
                                } else {
                                    (ArcPoint::Start, ArcPoint::End)
                                };

                                // Auto-coincident for start click
                                if let Some(s) = state.snap_start {
                                    self.apply_snap_coincident_arc(s, new_arc, start_ap, state.start);
                                }
                                // Auto-coincident for end click
                                if let Some(s) = snap_end {
                                    self.apply_snap_coincident_arc(s, new_arc, end_ap, end);
                                }
                                // Auto-coincident for mid (point on arc) - needs helper point
                                if let Some(s) = snap_target {
                                    self.exec(Action::AddHelperPoint { pos });
                                    let helper = Ref::new(self.sketch.points.slot_count() as u32 - 1);
                                    self.exec(Action::ApplyPointOnArc { point: helper, arc: new_arc });
                                    self.apply_snap_coincident_point(s, helper);
                                }
                            } else {
                                // Second click: end point
                                self.arc_draw = Some(ArcDrawState {
                                    start: state.start,
                                    snap_start: state.snap_start,
                                    end: Some((pos, snap_target)),
                                });
                            }
                        } else {
                            // First click: start point
                            self.arc_draw = Some(ArcDrawState {
                                start: pos,
                                snap_start: snap_target,
                                end: None,
                            });
                        }
                    }
                }

                Tool::ConstraintMode(ct) => {
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
                                        ConstraintType::ToggleStyle => self.apply_toggle_style(),
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

                Tool::Dimension => {
                    if self.dim_placing {
                        // Phase 2: positioning with mouse, click to confirm
                        if let Some(ref kind) = self.dim_kind {
                            if matches!(kind, DimensionKind::ArcRadius(r) if self.sketch.arcs.contains(*r)) {
                                if let DimensionKind::ArcRadius(r) = kind {
                                    let a = &self.sketch.arcs[*r];
                                    let angle = (mouse_sketch.y - a.center.value.y)
                                        .atan2(mouse_sketch.x - a.center.value.x);
                                    self.dim_offset = vect2d::new(angle, 0.0);
                                    self.dim_text_along = 0.0;
                                }
                            } else if let DimensionKind::Angle(a, b, _) = kind {
                                // Mouse position determines arc radius and which of 4 sectors
                                let la = &self.sketch.lines[*a];
                                let lb = &self.sketch.lines[*b];
                                let ix = line_line_intersection(
                                    la.p1.value, la.p2.value, lb.p1.value, lb.p2.value);
                                let dist = ((mouse_sketch.x - ix.x).powi(2)
                                    + (mouse_sketch.y - ix.y).powi(2)).sqrt();
                                let mouse_angle = (mouse_sketch.y - ix.y).atan2(mouse_sketch.x - ix.x);
                                let (sector_mid, sup) = self.angle_dim_sector_from_mouse(*a, *b, mouse_angle);
                                let new_offset = vect2d::new(sector_mid, dist.max(0.3));
                                // Compute text_along from mouse angle
                                let (_ix, start, sweep) = self.angle_dim_sector(*a, *b, sup, new_offset);
                                let delta = rad2rad(mouse_angle - start);
                                let along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                                self.dim_offset = new_offset;
                                self.dim_text_along = along.clamp(-0.5, 0.5); // During creation, clamp to sector
                                // Update supplement flag and measured value
                                if let Some(DimensionKind::Angle(_, _, ref mut s)) = self.dim_kind {
                                    if *s != sup {
                                        *s = sup;
                                        let measured = self.measure_dimension(&self.dim_kind.clone().unwrap());
                                        self.dim_input = format!("{:.4}", measured);
                                    }
                                }
                            } else {
                                // Decompose mouse into perpendicular and along
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
                                let rel_x = mouse_sketch.x - mx;
                                let rel_y = mouse_sketch.y - my;
                                let perp = rel_x * nx + rel_y * ny;
                                let along = (rel_x * ux + rel_y * uy) / len;
                                self.dim_offset = vect2d::new(0.0, perp);
                                self.dim_text_along = along;
                            }
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
                                if let Some(kind) = self.selection_to_dim_kind() {
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
                                if let Some(kind) = self.selection_to_dim_kind() {
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
                        for (i, dim) in self.sketch.dimensions.iter().enumerate() {
                            let (ts, te) = self.dim_text_segment(dim);
                            let d = Self::screen_point_to_segment_dist(mouse_screen, ts, te);
                            if d < 15.0 {
                                self.dim_input = if let Some(ref expr) = dim.expr_str {
                                    expr.clone()
                                } else {
                                    format!("{:.4}", dim.value)
                                };
                                self.dim_kind = Some(dim.kind.clone());
                                self.dim_offset = dim.offset;
                                self.dim_edit_index = Some(i);
                                self.dim_editing = true;
                                self.dim_select_all = true;
                                self.dim_placing = false;
                                break;
                            }
                        }
                    }
                }
            }

            // Build constraint markers and draw canvas
            if self.show_constraints {
                self.build_constraint_markers();
            } else {
                self.constraint_markers.clear();
            }
            self.draw_canvas(&painter, rect, mouse_screen);

            // Dimension preview while placing (not when editing an existing dimension)
            if (self.dim_placing || (self.dim_editing && self.dim_edit_index.is_none())) && self.dim_kind.is_some() {
                let kind = self.dim_kind.clone().unwrap();
                let measured = self.measure_dimension(&kind);
                let is_radius = matches!(kind, DimensionKind::ArcRadius(_));
                let preview_color = egui::Color32::from_rgba_premultiplied(200, 100, 50, 180);
                self.draw_dimension(&painter, &kind, measured, self.dim_offset, self.dim_text_along, preview_color, is_radius, false);
            }

            // Draw overlays ON TOP of canvas: preview line and cursor crosshair
            if let Some(ref state) = self.line_draw {
                let p1 = self.to_screen(state.start);
                painter.line_segment([p1, mouse_screen],
                    egui::Stroke::new(1.5, self.colors.preview_line));
                painter.circle_filled(p1, 4.0, self.colors.endpoint);
            }

            // Circle preview
            if let Some(ref state) = self.circle_draw {
                let center = self.to_screen(state.center);
                let radius_px = ((mouse_sketch.x - state.center.x).powi(2)
                    + (mouse_sketch.y - state.center.y).powi(2)).sqrt() as f32 * self.scale;
                painter.circle_stroke(center, radius_px,
                    egui::Stroke::new(1.5, self.colors.preview_line));
                painter.circle_filled(center, 4.0, self.colors.endpoint);
            }

            // Arc preview
            if let Some(ref state) = self.arc_draw {
                let start_screen = self.to_screen(state.start);
                painter.circle_filled(start_screen, 4.0, self.colors.endpoint);
                if let Some((end, _)) = state.end {
                    let end_screen = self.to_screen(end);
                    painter.circle_filled(end_screen, 4.0, self.colors.endpoint);
                    // Preview arc through start, end, and mouse
                    if let Some((c, r, sa, ea, _)) = circumscribed_arc(state.start, end, mouse_sketch) {
                        let norm = |v: f64| -> f64 { let rv = v % std::f64::consts::TAU; if rv < 0.0 { rv + std::f64::consts::TAU } else { rv } };
                        let span = norm(ea - sa);
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
                } else {
                    // Only start placed; draw line to mouse as hint
                    painter.line_segment([start_screen, mouse_screen],
                        egui::Stroke::new(1.0, self.colors.preview_line));
                }
            }

            // Draw cursor crosshair when drawing
            if self.tool != Tool::Select {
                painter.line_segment(
                    [egui::Pos2::new(mouse_screen.x, rect.top()),
                     egui::Pos2::new(mouse_screen.x, rect.bottom())],
                    egui::Stroke::new(0.5, self.colors.cursor_crosshair));
                painter.line_segment(
                    [egui::Pos2::new(rect.left(), mouse_screen.y),
                     egui::Pos2::new(rect.right(), mouse_screen.y)],
                    egui::Stroke::new(0.5, self.colors.cursor_crosshair));
            }

            // Status bar at bottom
            let status = match self.tool {
                Tool::Select => "Select: click to select/deselect, drag to move. Ctrl+Z undo, Ctrl+Shift+Z redo.",
                Tool::DrawPoint => "Point: click to place.",
                Tool::DrawLine => if self.line_draw.is_some() {
                    "Line: click to place end point (chains next line). Escape to finish."
                } else {
                    "Line: click to place start point. Snaps to nearby points/endpoints."
                },
                Tool::DrawCircle => if self.circle_draw.is_some() {
                    "Circle: click to set radius."
                } else {
                    "Circle: click to place center."
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

            // DOF + cost + version at bottom-right
            let dof_str = match self.dof_display {
                Some(0) => "DOF: 0 (fully constrained)".to_string(),
                Some(d) => format!("DOF: {}", d),
                None => "DOF: ...".to_string(),
            };
            let info = format!("{}  |  cost: {:.6}  |  arael v{}", dof_str, self.last_cost, env!("CARGO_PKG_VERSION"));
            painter.text(
                egui::Pos2::new(rect.right() - 10.0, rect.bottom() - 20.0),
                egui::Align2::RIGHT_CENTER,
                info,
                egui::FontId::proportional(11.0),
                self.colors.status_text,
            );
        });
    }
}
