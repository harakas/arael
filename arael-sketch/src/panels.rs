// Side panel, parameters panel, and command panel for the sketch
// editor, plus the file-dialog and history-restore helpers they share
// with the keyboard shortcuts.

use eframe::egui;
use arael_sketch_backend::actions::Action;
use arael_sketch_backend::history::CursorState;
use arael_sketch_solver::*;
use crate::colors::ColorScheme;
use crate::tools::*;
use crate::{EditorApp, spawn_async};

impl EditorApp {
    /// Restore a history snapshot: swap in the sketch, move the command
    /// cursor with it, and refresh the derived UI state.
    pub(crate) fn apply_history_restore(&mut self, restored: Sketch, cur: CursorState) {
        self.sketch = restored.into();
        self.command_cursor = cur.pos;
        self.command_cursor_tangent = cur.tangent;
        self.selection.clear();
        self.update_cost();
        self.refresh_dof();
    }

    /// Open the async save-file dialog for the current sketch.
    pub(crate) fn spawn_save_dialog(&self) {
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

    /// Open the async open-file dialog; the JSON lands in pending_load.
    pub(crate) fn spawn_open_dialog(&self) {
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
}

impl EditorApp {
    /// Left toolbar: view toggles, tools, constraint buttons, file,
    /// history, stats, and help.
    pub(crate) fn side_panel(&mut self, ctx: &egui::Context) {
        // Side panel: toolbar
        egui::SidePanel::left("toolbar").min_width(50.0).default_width(50.0).show(ctx, |ui| {
            // Toggle buttons (2x3 grid)
            egui::Grid::new("toggle_grid").num_columns(2).show(ui, |ui| {
                if ui.selectable_label(self.dark_mode, "Dark").clicked() {
                    self.dark_mode = !self.dark_mode;
                    self.colors = if self.dark_mode { ColorScheme::dark() } else { ColorScheme::light() };
                }
                if ui.selectable_label(self.show_constraints, "Cstr").clicked() {
                    self.show_constraints = !self.show_constraints;
                }
                ui.end_row();
                if ui.selectable_label(self.show_params, "Param").clicked() {
                    self.show_params = !self.show_params;
                }
                if ui.selectable_label(self.show_command, "/Cmd").clicked() {
                    self.show_command = !self.show_command;
                    if self.show_command {
                        self.command_focus = true;
                    } else {
                        self.command_has_focus = false;
                    }
                }
                ui.end_row();
                if ui.selectable_label(self.show_dimensions, "Dims").clicked() {
                    self.show_dimensions = !self.show_dimensions;
                }
                if ui.selectable_label(self.show_points, "Pts").clicked() {
                    self.show_points = !self.show_points;
                }
                ui.end_row();
            });
            ui.separator();

            // Tools (two columns)
            ui.heading("Tools");
            ui.separator();
            egui::Grid::new("tools_grid").num_columns(2).show(ui, |ui| {
                if ui.selectable_label(self.tool == Tool::Select, "Select (Esc)").clicked() {
                    self.tool = Tool::Select;
                }
                if ui.selectable_label(self.tool == Tool::DrawPoint, "Point (P)").clicked() {
                    self.tool = Tool::DrawPoint;
                }
                ui.end_row();
                if ui.selectable_label(self.tool == Tool::DrawLine, "Line (L)").clicked() {
                    self.tool = Tool::DrawLine;
                    self.line_draw = None;
                }
                if ui.selectable_label(self.tool == Tool::DrawCircle, "Circle (O)").clicked() {
                    self.tool = Tool::DrawCircle;
                    self.circle_draw = None;
                }
                ui.end_row();
                if ui.selectable_label(self.tool == Tool::DrawArc, "Arc (A)").clicked() {
                    self.tool = Tool::DrawArc;
                    self.arc_draw = None;
                }
                if ui.selectable_label(self.tool == Tool::DrawRect, "Rect (R)").clicked() {
                    self.tool = Tool::DrawRect;
                    self.rect_draw = None;
                }
                ui.end_row();
                if ui.selectable_label(self.tool == Tool::Fillet, "Fillet (F)").clicked() {
                    self.tool = Tool::Fillet;
                    self.selection.clear();
                }
                if ui.selectable_label(self.tool == Tool::Chamfer, "Chamfer").clicked() {
                    self.tool = Tool::Chamfer;
                    self.selection.clear();
                }
                ui.end_row();
                if ui.selectable_label(self.tool == Tool::Dimension, "Dims (D)").clicked() {
                    self.tool = Tool::Dimension;
                    self.dim_editing = false;
                    self.dim_kind = None;
                }
                ui.end_row();
            });

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
            constraint_btn(ui, self, ConstraintType::ToggleConstruction, "Constr (X)");

            // Dimension tool now lives under Tools (see "Dims (D)" in
            // the tools grid above). Dimension value input renders as
            // a floating overlay near the dim label via
            // `render_dim_input`.

            ui.separator();
            ui.heading("File");
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    self.spawn_save_dialog();
                }
                if ui.button("Open").clicked() {
                    self.spawn_open_dialog();
                }
            });

            ui.separator();
            ui.heading("History");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.add_enabled(self.history.can_undo(), egui::Button::new("Undo")).clicked()
                    && let Some((restored, cur)) = self.history.undo() {
                        self.apply_history_restore(restored, cur);
                    }
                if ui.add_enabled(self.history.can_redo(), egui::Button::new("Redo")).clicked()
                    && let Some((restored, cur)) = self.history.redo() {
                        self.apply_history_restore(restored, cur);
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
                        Selection::Dimension(did) => {
                            match self.sketch.dimension_index_by_did(did) {
                                Some(i) => {
                                    let d = &self.sketch.dimensions[i];
                                    Some(format!("{} = {:.2}", d.name, d.value))
                                }
                                None => Some("dim?".to_string()),
                            }
                        }
                    }
                }).collect();
                // For a single line / circle / arc selection, append
                // live measurements (length, radius, minor radius,
                // sweep-based arc length) so the user can read them
                // without running `info`.
                let measurement = if self.selection.len() == 1 {
                    match self.selection[0] {
                        Selection::Line(r) => {
                            let l = &self.sketch.lines[r];
                            let dx = l.p2.value.x - l.p1.value.x;
                            let dy = l.p2.value.y - l.p1.value.y;
                            Some(format!("length={:.4}", (dx * dx + dy * dy).sqrt()))
                        }
                        Selection::Arc(r) => {
                            let a = &self.sketch.arcs[r];
                            // Sweep: closed arcs are 2 pi by construction; open
                            // arcs use |end - start|. Arc length = sweep * r
                            // (for a circle) or the ellipse perimeter when
                            // is_ellipse, approximated via Ramanujan's 2nd
                            // formula.
                            let sweep = if a.closed { std::f64::consts::TAU }
                                else { (a.end_angle.value - a.start_angle.value).abs() };
                            if a.is_ellipse {
                                // No length -- the perimeter / arc
                                // length needs incomplete elliptic
                                // integrals; don't surface an
                                // approximation the user might trust.
                                let _ = sweep;
                                Some(format!("rx={:.4} ry={:.4}",
                                    a.radius.value, a.radius_b.value))
                            } else {
                                let r = a.radius.value;
                                Some(format!("r={:.4} length={:.4}", r, sweep * r))
                            }
                        }
                        _ => None,
                    }
                } else { None };
                let label = match measurement {
                    Some(m) => format!("Selected: {} ({})", names.join(", "), m),
                    None => format!("Selected: {}", names.join(", ")),
                };
                ui.label(label);
            }

            // Constraint conflict error message
            if let Some(ref err) = self.status_error {
                ui.separator();
                ui.colored_label(self.colors.error_text, err.as_str());
            }

            // Help button + debug flags at the bottom
            ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                ui.horizontal(|ui| {
                    if ui.button("Help").clicked() {
                        self.show_command = true;
                        self.command_focus = true;
                        self.help_expand = true;
                        self.help_scroll_top = true;
                        self.command_output.clear();
                        let results = self.run_commands("help full");
                        for result in &results {
                            if !result.output.is_empty() {
                                self.command_output.push((result.output.clone(), result.is_error, result.markdown));
                            }
                        }
                    }
                    // Debug: command input state flags
                    let flags = format!("[{}{}{}{}]",
                        if self.command_has_focus { "F" } else { "." },
                        if self.command_focus { "R" } else { "." },
                        if !self.completions.is_empty() { "C" } else { "." },
                        if self.completion_suppressed { "S" } else { "." },
                    );
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        ui.label(egui::RichText::new(flags).monospace().small().weak());
                    });
                });
            });

        });
    }
}

impl EditorApp {
    /// Top parameters table (toggled).
    pub(crate) fn params_panel(&mut self, ctx: &egui::Context) {
        // Parameters panel (top, toggled)
        if self.show_params {
            egui::TopBottomPanel::top("parameters").show(ctx, |ui| {
                use egui_extras::{TableBuilder, Column};
                ui.horizontal(|ui| {
                    ui.heading("Parameters");
                });
                let broken_color = self.colors.dimension_broken;
                let normal_color = ui.visuals().text_color();
                let row_height = 20.0;
                let mut remove_idx = None;
                let mut update_action = None;
                let mut start_edit: Option<(usize, bool)> = None; // (row, focus_expr)
                let mut add_new = false;

                TableBuilder::new(ui)
                    .striped(true)
                    .resizable(true)
                    .cell_layout(egui::Layout::left_to_right(egui::Align::Center))
                    .column(Column::initial(80.0).at_least(40.0).resizable(true))
                    .column(Column::initial(140.0).at_least(60.0).resizable(true).clip(true))
                    .column(Column::initial(80.0).at_least(40.0).resizable(true))
                    .column(Column::auto())
                    .header(row_height, |mut header| {
                        header.col(|ui| { ui.strong("Name"); });
                        header.col(|ui| { ui.strong("Expression"); });
                        header.col(|ui| { ui.strong("Value"); });
                        header.col(|_ui| {});
                    })
                    .body(|mut body| {
                        // Existing params
                        for i in 0..self.sketch.user_params.len() {
                            body.row(row_height, |mut row| {
                                let p = &self.sketch.user_params[i];
                                let color = if p.broken { broken_color } else { normal_color };
                                let editing = self.param_edit_index == Some(i);

                                row.col(|ui| {
                                    if editing {
                                        let r = ui.add(egui::TextEdit::singleline(&mut self.param_edit_name)
                                            .desired_width(ui.available_width()));
                                        if self.param_focus_field == Some(false) {
                                            r.request_focus();
                                            self.param_focus_field = None;
                                        }
                                    } else {
                                        ui.colored_label(color, &p.name);
                                        if ui.interact(ui.max_rect(), ui.id().with(("name", i)), egui::Sense::click()).clicked() {
                                            start_edit = Some((i, false));
                                        }
                                    }
                                });
                                row.col(|ui| {
                                    if editing {
                                        let r = ui.add(egui::TextEdit::singleline(&mut self.param_edit_expr)
                                            .desired_width(ui.available_width()));
                                        if self.param_focus_field == Some(true) {
                                            r.request_focus();
                                            self.param_focus_field = None;
                                        }
                                    } else {
                                        ui.colored_label(color, &p.expr_str);
                                        if ui.interact(ui.max_rect(), ui.id().with(("expr", i)), egui::Sense::click()).clicked() {
                                            start_edit = Some((i, true));
                                        }
                                    }
                                });
                                row.col(|ui| {
                                    ui.colored_label(color, format!("{:.4}", p.value));
                                });
                                row.col(|ui| {
                                    if ui.small_button("x").clicked() {
                                        remove_idx = Some(i);
                                    }
                                    if editing {
                                        let enter = ui.input(|inp| inp.key_pressed(egui::Key::Enter));
                                        let escape = ui.input(|inp| inp.key_pressed(egui::Key::Escape));
                                        if enter {
                                            let n = self.param_edit_name.trim().to_string();
                                            let e = self.param_edit_expr.trim().to_string();
                                            if !n.is_empty() && !e.is_empty() {
                                                update_action = Some((i, n, e));
                                            }
                                            self.param_edit_index = None;
                                        } else if escape {
                                            self.param_edit_index = None;
                                        }
                                    }
                                });
                            });
                        }
                        // Add-new row (aligned to same columns)
                        body.row(row_height, |mut row| {
                            row.col(|ui| {
                                let r = ui.add(egui::TextEdit::singleline(&mut self.param_new_name)
                                    .desired_width(ui.available_width()).hint_text("name"));
                                if self.param_focus_new {
                                    r.request_focus();
                                    self.param_focus_new = false;
                                }
                            });
                            row.col(|ui| {
                                let r = ui.add(egui::TextEdit::singleline(&mut self.param_new_expr)
                                    .desired_width(ui.available_width()).hint_text("expression"));
                                if r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                                    add_new = true;
                                }
                            });
                            row.col(|_ui| {});
                            row.col(|ui| {
                                if ui.small_button("+").clicked() {
                                    add_new = true;
                                }
                            });
                        });
                    });

                // Deferred actions (must be outside table closure)
                if let Some((i, focus_expr)) = start_edit {
                    // If already editing a different row, save it first
                    if let Some(prev) = self.param_edit_index
                        && prev != i {
                            let n = self.param_edit_name.trim().to_string();
                            let e = self.param_edit_expr.trim().to_string();
                            if !n.is_empty() && !e.is_empty() && prev < self.sketch.user_params.len() {
                                update_action = Some((prev, n, e));
                            }
                        }
                    if i < self.sketch.user_params.len() {
                        self.param_edit_index = Some(i);
                        self.param_edit_name = self.sketch.user_params[i].name.clone();
                        self.param_edit_expr = self.sketch.user_params[i].expr_str.clone();
                        self.param_focus_field = Some(focus_expr);
                    }
                }
                if let Some((idx, new_name, new_expr)) = update_action {
                    if let Err(e) = self.sketch.validate_param_name(&new_name, Some(idx)) {
                        self.status_error = Some(e);
                    } else {
                        self.apply_param_change(Action::UpdateUserParam { index: idx, name: new_name, expr_str: new_expr });
                    }
                }
                if let Some(idx) = remove_idx {
                    self.begin_group();
                    self.exec(Action::RemoveUserParam { index: idx });
                }
                if add_new {
                    let name = self.param_new_name.trim().to_string();
                    let expr = self.param_new_expr.trim().to_string();
                    if !name.is_empty() && !expr.is_empty() {
                        if let Err(e) = self.sketch.validate_param_name(&name, None) {
                            self.status_error = Some(e);
                        } else {
                            self.apply_param_change(Action::AddUserParam { name, expr_str: expr });
                            self.param_new_name.clear();
                            self.param_new_expr.clear();
                            self.param_focus_new = true;
                        }
                    }
                }
            });
        }
    }
}

impl EditorApp {
    /// Bottom command panel (toggled with /).
    pub(crate) fn command_panel(&mut self, ctx: &egui::Context) {
        // Command panel (bottom, toggled with /)
        if self.show_command {
            let default_h = if self.help_expand {
                self.help_expand = false;
                ctx.screen_rect().height() * 0.5
            } else {
                150.0
            };
            egui::TopBottomPanel::bottom("command_panel")
                .resizable(true)
                .min_height(60.0)
                .default_height(default_h)
                .show(ctx, |ui| {
                // Layout: input pinned to bottom, scroll area fills the rest above
                ui.with_layout(egui::Layout::bottom_up(egui::Align::LEFT), |ui| {
                    // Input area (at the bottom, fixed height)
                    // Multiline: Shift+Enter adds newline, Enter executes all lines
                    let cmd_id = egui::Id::new("command_input");
                    let line_height = ui.text_style_height(&egui::TextStyle::Monospace);
                    let num_rows = self.command_input.lines().count().max(1).min(8);
                    let input_h = line_height * num_rows as f32 + 10.0;
                    let input_w = ui.available_width();

                    // Intercept keys BEFORE TextEdit sees them (when completions are showing)
                    let (tab_accepted, arrow_up, arrow_down) = if !self.completions.is_empty() {
                        ui.input_mut(|i| {
                            let mut accepted = false;
                            let mut up = false;
                            let mut down = false;
                            i.events.retain(|e| {
                                match e {
                                    egui::Event::Key { key: egui::Key::Tab, pressed: true, .. } => {
                                        accepted = true;
                                        false
                                    }
                                    egui::Event::Key { key: egui::Key::Enter, pressed: true, modifiers, .. }
                                        if !modifiers.shift => {
                                        accepted = true;
                                        false
                                    }
                                    egui::Event::Key { key: egui::Key::ArrowUp, pressed: true, .. } => {
                                        up = true;
                                        false
                                    }
                                    egui::Event::Key { key: egui::Key::ArrowDown, pressed: true, .. } => {
                                        down = true;
                                        false
                                    }
                                    _ => true,
                                }
                            });
                            (accepted, up, down)
                        })
                    } else { (false, false, false) };

                    let r = ui.add_sized(
                        egui::vec2(input_w, input_h),
                        egui::TextEdit::multiline(&mut self.command_input)
                            .id(cmd_id)
                            .return_key(egui::KeyboardShortcut::new(
                                egui::Modifiers::SHIFT, egui::Key::Enter))
                            .desired_rows(num_rows)
                            .lock_focus(true)
                            .hint_text("type command, Shift+Enter for newline")
                            .font(egui::TextStyle::Monospace),
                    );
                    if self.command_focus {
                        r.request_focus();
                        self.command_focus = false;
                    }
                    // When completions are showing, tell egui's focus manager
                    // that we handle Escape (so it won't unfocus the TextEdit).
                    if !self.completions.is_empty() {
                        ui.memory_mut(|mem| {
                            mem.set_focus_lock_filter(cmd_id, egui::EventFilter {
                                tab: true,
                                escape: true,
                                horizontal_arrows: true,
                                vertical_arrows: true,
                            });
                        });
                    }
                    // Enter (without Shift) executes all lines
                    let enter_pressed = r.has_focus() && ui.input(|i|
                        i.key_pressed(egui::Key::Enter) && !i.modifiers.shift);
                    if enter_pressed {
                        let input = self.command_input.trim().to_string();
                        if !input.is_empty() {
                            self.command_history.push(input.clone());
                            self.command_history_pos = self.command_history.len();
                            for line in input.lines() {
                                let line = line.trim();
                                if line.is_empty() || line.starts_with('#') { continue; }
                                self.command_output.push((format!("> {}", line), false, false));
                                let results = self.run_commands(line);
                                for result in results {
                                    if !result.output.is_empty() {
                                        self.command_output.push((result.output, result.is_error, result.markdown));
                                    }
                                }
                            }
                            self.command_input.clear();
                            self.command_scroll_to_bottom = true;
                        }
                        self.command_focus = true;
                    }
                    // command_has_focus = "user is in command-entry mode".
                    if r.has_focus() {
                        self.command_has_focus = true;
                    }

                    // History navigation (only for single-line input, not when completions showing)
                    if r.has_focus() && !self.command_input.contains('\n') && self.completions.is_empty() {
                        let mut history_changed = false;
                        if ui.input(|i| i.key_pressed(egui::Key::ArrowUp)) && !ui.input(|i| i.modifiers.shift)
                            && self.command_history_pos > 0 {
                                self.command_history_pos -= 1;
                                self.command_input = self.command_history[self.command_history_pos].clone();
                                history_changed = true;
                            }
                        if ui.input(|i| i.key_pressed(egui::Key::ArrowDown)) && !ui.input(|i| i.modifiers.shift)
                            && self.command_history_pos < self.command_history.len() {
                                self.command_history_pos += 1;
                                if self.command_history_pos < self.command_history.len() {
                                    self.command_input = self.command_history[self.command_history_pos].clone();
                                } else {
                                    self.command_input.clear();
                                }
                                history_changed = true;
                            }
                        if history_changed
                            && let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), cmd_id) {
                                let end = egui::text::CCursor::new(self.command_input.len());
                                state.cursor.set_char_range(Some(egui::text::CCursorRange::one(end)));
                                egui::TextEdit::store_state(ui.ctx(), cmd_id, state);
                            }
                    }

                    // Autocomplete (skip when suppressed by Escape)
                    if r.has_focus() && !self.completion_suppressed {
                        let cursor_pos = egui::TextEdit::load_state(ui.ctx(), cmd_id)
                            .and_then(|s| s.cursor.char_range())
                            .map(|cr| cr.primary.index)
                            .unwrap_or(self.command_input.len());
                        self.completions = arael_sketch_backend::commands::complete(
                            &self.sketch, &self.session_names,
                            &self.command_input, cursor_pos);
                        if self.completions.is_empty() {
                            self.completion_idx = 0;
                        } else {
                            self.completion_idx = self.completion_idx.min(self.completions.len() - 1);
                        }

                        // Tab or Enter: accept selected completion (events consumed above)
                        if !self.completions.is_empty() && tab_accepted {
                            let completion = self.completions[self.completion_idx].clone();
                            // Find current word boundaries to replace
                            let input_before_cursor = &self.command_input[..cursor_pos.min(self.command_input.len())];
                            let current_line = input_before_cursor.lines().last().unwrap_or("");
                            let line_start = input_before_cursor.len() - current_line.len();
                            let word_start = line_start + current_line.rfind(|c: char| c.is_whitespace())
                                .map(|i| i + 1).unwrap_or(0);
                            let word_end = cursor_pos.min(self.command_input.len());
                            // Replace current word with completion
                            self.command_input.replace_range(word_start..word_end, &completion);
                            let new_pos = word_start + completion.len();
                            // Add space after command names (no dot in completion)
                            if !completion.contains('.') {
                                self.command_input.insert(new_pos, ' ');
                            }
                            if let Some(mut state) = egui::TextEdit::load_state(ui.ctx(), cmd_id) {
                                let cursor = egui::text::CCursor::new(
                                    if completion.contains('.') { new_pos } else { new_pos + 1 });
                                state.cursor.set_char_range(Some(egui::text::CCursorRange::one(cursor)));
                                egui::TextEdit::store_state(ui.ctx(), cmd_id, state);
                            }
                            self.completions.clear();
                        }

                        // Up/Down: navigate completion list (events consumed above)
                        if !self.completions.is_empty() {
                            if arrow_up && self.completion_idx > 0 {
                                self.completion_idx -= 1;
                            }
                            if arrow_down && self.completion_idx + 1 < self.completions.len() {
                                self.completion_idx += 1;
                            }
                        }
                    } else {
                        self.completions.clear();
                    }

                    // Show completion popup above the input, at cursor position
                    let mut clicked_completion: Option<String> = None;
                    if !self.completions.is_empty() {
                        let num_shown = self.completions.len().min(8);
                        let item_h = 18.0;
                        let popup_h = num_shown as f32 * item_h + 12.0;

                        // Get cursor screen X by measuring text up to cursor
                        let cursor_pos = egui::TextEdit::load_state(ui.ctx(), cmd_id)
                            .and_then(|s| s.cursor.char_range())
                            .map(|cr| cr.primary.index)
                            .unwrap_or(self.command_input.len());
                        let text_before_cursor = &self.command_input[..cursor_pos.min(self.command_input.len())];
                        let current_line_text = text_before_cursor.lines().last().unwrap_or("");
                        let font_id = egui::TextStyle::Monospace.resolve(ui.style());
                        let text_width = ui.fonts(|f| f.layout_no_wrap(
                            current_line_text.to_string(), font_id.clone(), egui::Color32::WHITE)).size().x;
                        let cursor_x = r.rect.left() + 4.0 + text_width;
                        let popup_x = cursor_x.min(r.rect.right() - 150.0).max(r.rect.left());
                        let popup_pos = egui::pos2(popup_x, r.rect.top() - popup_h);

                        // Compute min width from longest suggestion
                        let max_len = self.completions.iter().take(8).map(|s| s.len()).max().unwrap_or(10);
                        let char_w = ui.fonts(|f| f.layout_no_wrap(
                            "M".to_string(), font_id.clone(), egui::Color32::WHITE)).size().x;
                        let min_w = (max_len as f32 * char_w + 20.0).max(120.0);

                        egui::Area::new(egui::Id::new("cmd_completions"))
                            .fixed_pos(popup_pos)
                            .order(egui::Order::Foreground)
                            .show(ui.ctx(), |ui| {
                                egui::Frame::popup(ui.style()).show(ui, |ui| {
                                    ui.set_min_width(min_w);
                                    for (i, suggestion) in self.completions.iter().enumerate().take(8) {
                                        let selected = i == self.completion_idx;
                                        if ui.add(egui::SelectableLabel::new(selected, suggestion)).clicked() {
                                            clicked_completion = Some(suggestion.clone());
                                        }
                                    }
                                });
                            });
                    }
                    if let Some(completion) = clicked_completion {
                        let cursor_pos = egui::TextEdit::load_state(ui.ctx(), cmd_id)
                            .and_then(|s| s.cursor.char_range())
                            .map(|cr| cr.primary.index)
                            .unwrap_or(self.command_input.len());
                        let input_before = &self.command_input[..cursor_pos.min(self.command_input.len())];
                        let cl = input_before.lines().last().unwrap_or("");
                        let ls = input_before.len() - cl.len();
                        let ws = ls + cl.rfind(|c: char| c.is_whitespace()).map(|i| i + 1).unwrap_or(0);
                        let we = cursor_pos.min(self.command_input.len());
                        self.command_input.replace_range(ws..we, &completion);
                        let np = ws + completion.len();
                        if !completion.contains('.') {
                            self.command_input.insert(np, ' ');
                        }
                        self.command_focus = true;
                        self.completions.clear();
                    }

                    // Scroll area fills remaining space above input
                    let scroll_h = ui.available_height();
                    ui.with_layout(egui::Layout::top_down(egui::Align::LEFT), |ui| {
                        let scroll_top = self.help_scroll_top;
                        if scroll_top { self.help_scroll_top = false; }
                        let scroll_bottom = self.command_scroll_to_bottom;
                        if scroll_bottom { self.command_scroll_to_bottom = false; }
                        // Reset scroll state to force position
                        if scroll_top || scroll_bottom {
                            let scroll_id = ui.make_persistent_id(egui::Id::new("scroll_area"));
                            let mut state = egui::scroll_area::State::default();
                            if scroll_top {
                                state.offset.y = 0.0;
                            }
                            // scroll_stuck_to_end defaults to TRUE, so stick_to_bottom
                            // will snap to end on next frame for scroll_bottom case
                            state.store(ui.ctx(), scroll_id);
                        }
                        let scroll_id = ui.make_persistent_id(egui::Id::new("scroll_area"));
                        // Keyboard scrolling when command input is not focused
                        let mut kbd_scrolling_up = false;
                        if !self.command_has_focus {
                            let line_h = 20.0;
                            let page_h = scroll_h - line_h;
                            let mut dy = 0.0;
                            let mut jump = None;
                            let mut jump_end = false;
                            ui.input(|i| {
                                if i.key_pressed(egui::Key::ArrowUp) { dy -= line_h; }
                                if i.key_pressed(egui::Key::ArrowDown) { dy += line_h; }
                                if i.key_pressed(egui::Key::PageUp) { dy -= page_h; }
                                if i.key_pressed(egui::Key::PageDown) { dy += page_h; }
                                if i.key_pressed(egui::Key::Home) { jump = Some(0.0_f32); }
                                if i.key_pressed(egui::Key::End) { jump_end = true; }
                            });
                            if jump_end {
                                let state = egui::scroll_area::State::default();
                                state.store(ui.ctx(), scroll_id);
                            } else if dy != 0.0 || jump.is_some() {
                                if let Some(mut state) = egui::scroll_area::State::load(ui.ctx(), scroll_id) {
                                    if let Some(target) = jump {
                                        state.offset.y = target;
                                    } else {
                                        state.offset.y = (state.offset.y + dy).max(0.0);
                                    }
                                    state.store(ui.ctx(), scroll_id);
                                }
                                if dy < 0.0 || jump == Some(0.0) {
                                    kbd_scrolling_up = true;
                                }
                            }
                        }
                        let scroll = egui::ScrollArea::vertical()
                            .max_height(scroll_h)
                            .stick_to_bottom(!scroll_top && !kbd_scrolling_up);
                        scroll.show(ui, |ui| {
                            ui.set_min_width(ui.available_width());
                            let mut md_cache = egui_commonmark::CommonMarkCache::default();
                            for (text, is_err, is_md) in self.command_output.iter() {
                                if *is_md {
                                    egui_commonmark::CommonMarkViewer::new()
                                        .show(ui, &mut md_cache, text);
                                } else {
                                    let color = if *is_err {
                                        self.colors.error_text
                                    } else {
                                        ui.visuals().text_color()
                                    };
                                    for line in text.lines() {
                                        ui.colored_label(color, line);
                                    }
                                }
                            }
                        });
                    });
                });
            });
        }
    }
}

