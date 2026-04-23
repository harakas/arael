// eframe::App implementation for EditorApp.

use eframe::egui;
use arael::utils::rad2rad;
use arael::vect::vect2d;
use arael::refs::Ref;
use arael_sketch_solver::*;
use crate::colors::ColorScheme;
use crate::tools::*;
use arael_sketch_backend::actions::Action;
use arael_sketch_backend::geometry::*;
use crate::{EditorApp, spawn_async};

// Small right-angle corner marker placed at the corner where the drawn
// line meets the host line. Two short segments form the open "L".
//
// The host-arm direction is chosen to lie along the portion of the host
// line that actually exists: if the corner sits at host.P1, the arm
// points toward P2; if at host.P2, toward P1. For a mid-body corner both
// sides are valid, so we fall back to the quadrant that also sits along
// the drawn-line direction -- keeping the marker visually "hugging" the
// drawn line's side.
/// Auto-horizontal / auto-vertical snap for a line defined by `start`
/// and `end`. Returns `(horizontal, snapped_end)` when the line's
/// perpendicular-to-axis deviation (in screen pixels) falls below
/// `threshold_px`: the end is pulled onto the axis. Requires the line
/// to be at least a minimum length so short jitter doesn't fire the
/// hint. Mirrors PERP_SNAP_PX in spirit -- same tolerance scale.
pub fn hv_snap_from(start: arael::vect::vect2d, end: arael::vect::vect2d, scale: f32, threshold_px: f32) -> Option<(bool, arael::vect::vect2d)> {
    let dx = end.x - start.x;
    let dy = end.y - start.y;
    let dx_px = (dx.abs() as f32) * scale;
    let dy_px = (dy.abs() as f32) * scale;
    // Below this segment length the line is too short for the H/V
    // hint to be meaningful; let the user position freely.
    let min_len_px = threshold_px * 3.0;
    if dx_px < min_len_px && dy_px < min_len_px {
        return None;
    }
    if dy_px < threshold_px && dx_px >= dy_px {
        return Some((true, arael::vect::vect2d::new(end.x, start.y)));
    }
    if dx_px < threshold_px && dy_px >= dx_px {
        return Some((false, arael::vect::vect2d::new(start.x, end.y)));
    }
    None
}

/// Draw the Collinear glyph (diagonal line with a gap in the middle)
/// at the given screen position. Mirrors ConstraintSymbol::Collinear
/// in drawing.rs so live hints look the same as committed markers.
fn draw_collinear_marker(painter: &egui::Painter, pos: egui::Pos2, color: egui::Color32) {
    let s = 7.0f32;
    let stroke = egui::Stroke::new(2.0, color);
    painter.line_segment([
        egui::Pos2::new(pos.x - s * 0.7, pos.y + s * 0.7),
        egui::Pos2::new(pos.x - s * 0.1, pos.y + s * 0.1),
    ], stroke);
    painter.line_segment([
        egui::Pos2::new(pos.x + s * 0.1, pos.y - s * 0.1),
        egui::Pos2::new(pos.x + s * 0.7, pos.y - s * 0.7),
    ], stroke);
}

/// Draw an H or V constraint symbol on the sketch canvas at the given
/// screen position. Mirrors the ConstraintSymbol::H / V glyphs from
/// drawing.rs so a live hint during line-draw or drag looks the same
/// as a committed marker.
fn draw_hv_marker(painter: &egui::Painter, pos: egui::Pos2, horizontal: bool, color: egui::Color32) {
    let s = 7.0f32;
    let stroke = egui::Stroke::new(2.0, color);
    if horizontal {
        let g = s * 0.45;
        painter.line_segment([egui::Pos2::new(pos.x - g, pos.y - s), egui::Pos2::new(pos.x - g, pos.y + s)], stroke);
        painter.line_segment([egui::Pos2::new(pos.x + g, pos.y - s), egui::Pos2::new(pos.x + g, pos.y + s)], stroke);
        painter.line_segment([egui::Pos2::new(pos.x - g, pos.y), egui::Pos2::new(pos.x + g, pos.y)], stroke);
    } else {
        painter.line_segment([egui::Pos2::new(pos.x - s, pos.y - s), egui::Pos2::new(pos.x, pos.y + s)], stroke);
        painter.line_segment([egui::Pos2::new(pos.x + s, pos.y - s), egui::Pos2::new(pos.x, pos.y + s)], stroke);
    }
}

fn draw_perp_corner_marker(
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

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Handle exit request
        if self.exit_requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        // Poll background DOF computation
        self.poll_dof();

        // Keep repainting while a constraint-conflict flash is active
        // (3 flashes at 3 Hz = 1 s total). Without continuous repaint
        // the canvas would freeze between input events.
        if let Some(start) = self.flash_start {
            let elapsed = start.elapsed().as_secs_f64();
            if elapsed > 1.0 {
                self.flash_start = None;
                self.flash_names.clear();
            } else {
                ctx.request_repaint_after(std::time::Duration::from_millis(16));
            }
        }

        // Hold-to-disable snapping and auto-constraints. Modifier keys
        // (Shift / Ctrl / Alt / Command) are unusable under Parallels +
        // GNOME: the host captures the hold and only leaks a brief
        // release event to the guest, so `i.modifiers.*` is effectively
        // always false while held. A regular letter key goes through
        // egui's `keys_down` set, which is maintained from full press
        // and release events -- reliable on every platform.
        // `modifiers.command` = Cmd on macOS, Ctrl on Windows/Linux.
        // Either works natively; Q is the Parallels-safe fallback since
        // modifier events don't leak cleanly through the VM layer.
        self.snap_disabled = ctx.input(|i| i.key_down(egui::Key::Q) || i.modifiers.command);

        // Give MCP server access to egui context for waking the GUI
        #[cfg(not(target_arch = "wasm32"))]
        if self.egui_ctx.lock().unwrap().is_none() {
            *self.egui_ctx.lock().unwrap() = Some(ctx.clone());
        }

        // Check for pending file load from async dialog
        let pending_json = self.pending_load.lock().unwrap().take();
        if let Some(json) = pending_json {
            self.load_from_json(&json);
        }

        // Global key handling (before any widgets process input)
        // Escape: if completions showing, consume the event, close popup, suppress until space/dot.
        // If completions are showing, Escape dismisses them (but keeps focus).
        // If completions were just dismissed (suppressed), next Escape exits command mode.
        if !self.completions.is_empty() {
            // Consume Escape so TextEdit doesn't unfocus — just dismiss completions
            let esc = ctx.input_mut(|i| {
                let mut found = false;
                i.events.retain(|e| {
                    if matches!(e, egui::Event::Key { key: egui::Key::Escape, pressed: true, .. }) {
                        found = true;
                        false // consume
                    } else { true }
                });
                found
            });
            if esc {
                self.completions.clear();
                self.completion_suppressed = true;
            }
        } else if self.command_has_focus && ctx.input(|i| i.key_pressed(egui::Key::Escape)) {
            self.command_has_focus = false;
            self.completion_suppressed = false;
        }
        // Reset suppression when user types a separator (space, dot, semicolon, #, newline)
        if self.completion_suppressed {
            let reset = ctx.input(|i| {
                i.events.iter().any(|e| match e {
                    egui::Event::Text(t) => t.chars().any(|c| !c.is_alphanumeric() && c != '_'),
                    egui::Event::Key { key: egui::Key::Enter, pressed: true, modifiers, .. }
                        if modifiers.shift => true, // Shift+Enter = new line
                    _ => false,
                })
            });
            if reset { self.completion_suppressed = false; }
        }

        // Apply egui visuals for widgets (side panel, buttons, etc.)
        ctx.set_visuals(if self.dark_mode { egui::Visuals::dark() } else { egui::Visuals::light() });

        // Process pending MCP requests
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut pending = Vec::new();
            if let Some(ref mut mcp_rx) = self.mcp_rx {
                while let Ok(req) = mcp_rx.try_recv() {
                    pending.push(req);
                }
            }
            for req in pending {
                // Show MCP commands in command history (unless it's a msg command)
                let is_msg = req.command.starts_with("msg ");
                if !is_msg {
                    self.command_output.push((format!("**MCP>** {}", req.command), false, true));
                }
                let results = self.run_commands_with_blocked(&req.command, req.blocked_commands);
                let output = results.iter()
                    .filter(|r| !r.output.is_empty())
                    .map(|r| if r.is_error { format!("ERROR: {}", r.output) } else { r.output.clone() })
                    .collect::<Vec<_>>()
                    .join("\n");
                // Show results in command history
                for result in &results {
                    if !result.output.is_empty() {
                        self.command_output.push((result.output.clone(), result.is_error, result.markdown));
                    }
                }
                let _ = req.response_tx.send(output);
            }
        }

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
                if ui.selectable_label(self.tool == Tool::Select, "Select (S)").clicked() {
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
                if ui.button("Save").clicked()
                    && let Ok(json) = serde_json::to_string_pretty(&self.sketch) {
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
                if ui.add_enabled(self.history.can_undo(), egui::Button::new("Undo")).clicked()
                    && let Some((restored, cur)) = self.history.undo() {
                        self.sketch = restored;
                        self.command_cursor = cur.pos;
                        self.command_cursor_tangent = cur.tangent;
                        self.selection.clear();
                        self.update_cost();
                        self.compute_dof_async();
                    }
                if ui.add_enabled(self.history.can_redo(), egui::Button::new("Redo")).clicked()
                    && let Some((restored, cur)) = self.history.redo() {
                        self.sketch = restored;
                        self.command_cursor = cur.pos;
                        self.command_cursor_tangent = cur.tangent;
                        self.selection.clear();
                        self.update_cost();
                        self.compute_dof_async();
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

            // Keyboard shortcuts (skip when any text field has focus)
            if !ctx.wants_keyboard_input() {
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
            if ui.input(|i| i.key_pressed(egui::Key::R) && !i.modifiers.ctrl && !i.modifiers.mac_cmd) {
                self.tool = Tool::DrawRect;
                self.rect_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::F) && !i.modifiers.ctrl && !i.modifiers.mac_cmd) {
                self.tool = Tool::Fillet;
                self.selection.clear();
            }
            if ui.input(|i| i.key_pressed(egui::Key::H)) { self.try_apply_or_enter_mode(ConstraintType::Horizontal); }
            if ui.input(|i| i.key_pressed(egui::Key::V)) { self.try_apply_or_enter_mode(ConstraintType::Vertical); }
            if ui.input(|i| i.key_pressed(egui::Key::C)) { self.try_apply_or_enter_mode(ConstraintType::Coincident); }
            if ui.input(|i| i.key_pressed(egui::Key::K)) { self.try_apply_or_enter_mode(ConstraintType::Lock); }
            if ui.input(|i| i.key_pressed(egui::Key::T)) { self.try_apply_or_enter_mode(ConstraintType::Tangent); }
            if ui.input(|i| i.key_pressed(egui::Key::M)) { self.try_apply_or_enter_mode(ConstraintType::Midpoint); }
            if ui.input(|i| i.key_pressed(egui::Key::S)) { self.try_apply_or_enter_mode(ConstraintType::Symmetry); }
            if ui.input(|i| i.key_pressed(egui::Key::X)) { self.try_apply_or_enter_mode(ConstraintType::ToggleConstruction); }
            if ui.input(|i| i.key_pressed(egui::Key::D) && !i.modifiers.ctrl && !i.modifiers.mac_cmd) {
                self.tool = Tool::Dimension;
                self.dim_editing = false;
                self.dim_kind = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::Slash)) {
                self.show_command = true;
                self.command_focus = true;
            }
            } // !wants_keyboard_input
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                // Fillet-in-flight gets first crack: roll back the
                // applied fillet before the generic Escape clears
                // dim-editing state and flips the tool.
                if self.fillet_pending.is_some() {
                    self.cancel_pending_fillet();
                }
                self.selection.clear();
                self.line_draw = None;
                self.circle_draw = None;
                self.arc_draw = None;
                self.rect_draw = None;
                self.dim_editing = false;
                self.dim_kind = None;
                self.dim_placing = false;
                self.dim_edit_index = None;
                self.status_error = None;
                self.tool = Tool::Select;
            }

            // Delete selected entities/constraints with Backspace/Delete
            // (skip when editing dimension text — Backspace edits the text field)
            if !ctx.wants_keyboard_input() && ui.input(|i| i.key_pressed(egui::Key::Backspace) || i.key_pressed(egui::Key::Delete)) {
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
                if let Some((restored, cur)) = self.history.redo() {
                    self.sketch = restored;
                    self.command_cursor = cur.pos;
                    self.command_cursor_tangent = cur.tangent;
                    self.selection.clear();
                    self.update_cost();
                    self.compute_dof_async();
                }
            } else if ctrl && ui.input(|i| i.key_pressed(egui::Key::Z))
                && let Some((restored, cur)) = self.history.undo() {
                    self.sketch = restored;
                    self.command_cursor = cur.pos;
                    self.command_cursor_tangent = cur.tangent;
                    self.selection.clear();
                    self.update_cost();
                    self.compute_dof_async();
                }
            if ctrl && ui.input(|i| i.key_pressed(egui::Key::S))
                && let Ok(json) = serde_json::to_string_pretty(&self.sketch) {
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

            // Zoom (scroll wheel, only when mouse is over the canvas)
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 && response.hovered() {
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

            // Hover detection (frozen during geometry drag)
            let new_hover = if self.grab.is_some() {
                // During drag, keep hover on the dragged entity
                match self.grab {
                    Some(GrabTarget::LineDrag(r)) => Some(Selection::Line(r)),
                    Some(GrabTarget::ArcDrag(r)) => Some(Selection::Arc(r)),
                    Some(GrabTarget::LineP1(r)) => Some(Selection::LineP1(r)),
                    Some(GrabTarget::LineP2(r)) => Some(Selection::LineP2(r)),
                    Some(GrabTarget::ArcCenter(r)) => Some(Selection::ArcCenter(r)),
                    Some(GrabTarget::ArcStart(r)) => Some(Selection::ArcStart(r)),
                    Some(GrabTarget::ArcEnd(r)) => Some(Selection::ArcEnd(r)),
                    Some(GrabTarget::Point(r)) => Some(Selection::Point(r)),
                    None => None,
                }
            } else if response.hovered() {
                self.hit_test_selection(mouse_sketch, hit_threshold)
            } else {
                None
            };
            if new_hover != self.hovered {
                self.hovered = new_hover;
                ctx.request_repaint();
            }

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
                                self.dim_input = Self::dim_edit_string(dim);
                                self.dim_kind = Some(dim.kind);
                                self.dim_offset = dim.offset;
                                self.dim_edit_index = Some(i);
                                self.dim_editing = true;
                                self.dim_select_all = true;
                                self.dim_placing = false;
                                self.dim_derived = dim.derived;
                                self.dim_derived_prev = dim.derived;
                                self.dim_input_backup.clear();
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
                            // First drag frame: hit-test against the PRESS ORIGIN,
                            // not the current cursor. egui's dragged_by only fires
                            // after the pointer has moved past its internal
                            // drag-start threshold (max_click_dist, a few px), so
                            // the cursor has drifted outside the hover zone by the
                            // time we get here. pointer.press_origin() returns the
                            // screen position where the mouse button actually went
                            // down -- what the user aimed at.
                            let press_sketch = ui.ctx().input(|i| i.pointer.press_origin())
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
                        }
                        if let Some(dim_idx) = self.drag_dimension {
                            // Update dimension offset and text_along from mouse
                            if dim_idx < self.sketch.dimensions.len() {
                                let kind = self.sketch.dimensions[dim_idx].kind;
                                let is_radius = matches!(kind, DimensionKind::ArcRadius(_) | DimensionKind::ArcRadiusB(_));
                                if is_radius {
                                    let (arc_ref, is_b) = match kind {
                                        DimensionKind::ArcRadius(r) => (Some(r), false),
                                        DimensionKind::ArcRadiusB(r) => (Some(r), true),
                                        _ => (None, false),
                                    };
                                    if let Some(r) = arc_ref {
                                        let a = &self.sketch.arcs[r];
                                        if a.is_ellipse {
                                            // Project mouse onto the relevant axis to determine side
                                            let dx = mouse_sketch.x - a.center.value.x;
                                            let dy = mouse_sketch.y - a.center.value.y;
                                            let axis_angle = if is_b {
                                                a.rotation.value + std::f64::consts::FRAC_PI_2
                                            } else {
                                                a.rotation.value
                                            };
                                            // Dot product with axis direction gives sign
                                            let proj = dx * axis_angle.cos() + dy * axis_angle.sin();
                                            let sign = if proj >= 0.0 { 1.0 } else { -1.0 };
                                            self.sketch.dimensions[dim_idx].offset = vect2d::new(sign, 0.0);
                                        } else {
                                            // Circle: free angle
                                            let abs_angle = (mouse_sketch.y - a.center.value.y)
                                                .atan2(mouse_sketch.x - a.center.value.x);
                                            let rel_angle = abs_angle - a.start_angle.value;
                                            self.sketch.dimensions[dim_idx].offset = vect2d::new(rel_angle, 0.0);
                                        }
                                    }
                                } else if let DimensionKind::ArcSweep(r) = kind {
                                    let a = &self.sketch.arcs[r];
                                    let cx = a.center.value.x;
                                    let cy = a.center.value.y;
                                    let dist = ((mouse_sketch.x - cx).powi(2) + (mouse_sketch.y - cy).powi(2)).sqrt();
                                    let offset_y = dist - a.radius.value;
                                    let mouse_angle = (mouse_sketch.y - cy).atan2(mouse_sketch.x - cx);
                                    let sa = a.start_angle.value;
                                    let sweep = a.end_angle.value - sa;
                                    let delta = rad2rad(mouse_angle - sa);
                                    let along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                                    self.sketch.dimensions[dim_idx].offset = vect2d::new(0.0, offset_y);
                                    self.sketch.dimensions[dim_idx].text_along = along;
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
                                } else if let DimensionKind::LineAngle(r) = kind {
                                    let p1 = self.sketch.lines[r].p1.value;
                                    let line_angle = {
                                        let l = &self.sketch.lines[r];
                                        (l.p2.value.y - l.p1.value.y).atan2(l.p2.value.x - l.p1.value.x)
                                    };
                                    let dist = ((mouse_sketch.x - p1.x).powi(2)
                                        + (mouse_sketch.y - p1.y).powi(2)).sqrt();
                                    let mouse_angle = (mouse_sketch.y - p1.y).atan2(mouse_sketch.x - p1.x);
                                    let sweep = line_angle;
                                    let delta = rad2rad(mouse_angle);
                                    let along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                                    self.sketch.dimensions[dim_idx].offset = vect2d::new(0.0, dist.max(0.3));
                                    self.sketch.dimensions[dim_idx].text_along = along;
                                } else if let DimensionKind::ArcRotation(r) = kind {
                                    let a = &self.sketch.arcs[r];
                                    let center = a.center.value;
                                    let rotation = a.rotation.value;
                                    let dist = ((mouse_sketch.x - center.x).powi(2)
                                        + (mouse_sketch.y - center.y).powi(2)).sqrt();
                                    let mouse_angle = (mouse_sketch.y - center.y).atan2(mouse_sketch.x - center.x);
                                    let sweep = rotation;
                                    let delta = rad2rad(mouse_angle);
                                    let along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                                    self.sketch.dimensions[dim_idx].offset = vect2d::new(0.0, dist.max(0.3));
                                    self.sketch.dimensions[dim_idx].text_along = along;
                                } else if matches!(kind, DimensionKind::HDistance(..) | DimensionKind::VDistance(..)) {
                                    let horizontal = matches!(kind, DimensionKind::HDistance(..));
                                    let (p1, p2) = self.dim_endpoints(&kind);
                                    let mid = vect2d::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
                                    let offset_val = if horizontal {
                                        mouse_sketch.y - mid.y
                                    } else {
                                        mouse_sketch.x - mid.x
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
                                    let along = ((mouse_sketch.x - qmx) * ddx + (mouse_sketch.y - qmy) * ddy) / (dlen * dlen);
                                    self.sketch.dimensions[dim_idx].offset = vect2d::new(0.0, offset_val);
                                    self.sketch.dimensions[dim_idx].text_along = along;
                                } else if let DimensionKind::ConcentricDistance(a_ref, _b_ref) = kind {
                                    // 2D text anchor in world coords relative
                                    // to the shared center. The leader rotates
                                    // to follow the text; text renders parallel
                                    // to the leader; an extension line connects
                                    // the outer arrow tip to the text when the
                                    // anchor sits past the outer radius.
                                    let center = self.sketch.arcs[a_ref].center.value;
                                    self.sketch.dimensions[dim_idx].offset =
                                        vect2d::new(mouse_sketch.x - center.x,
                                                    mouse_sketch.y - center.y);
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
                            self.update_drag(mouse_sketch, hit_threshold);
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

                    // Click (no drag): paste into command prompt if focused, else select/deselect
                    if response.clicked_by(egui::PointerButton::Primary) {
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
                                self.find_best_perp_host_at(state.start, mouse_sketch, crate::PERP_SNAP_PX, None)
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
                                    let perp_p2 = vect2d::new(state.start.x - hdy, state.start.y + hdx);
                                    end_pos = line_line_intersection(state.start, perp_p2, ol.p1.value, ol.p2.value);
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
                                        let host_ref = state.snap_start.and_then(EditorApp::perp_host_from_snap);
                                        if host_ref == Some(target) {
                                            None
                                        } else {
                                            let tl = &self.sketch.lines[target];
                                            self.try_perp_end_snap(state.start, tl.p1.value, tl.p2.value, p_on_target, crate::PERP_SNAP_PX)
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
                                self.find_best_collinear_host_at(state.start, end_pos, crate::PERP_SNAP_PX, None)
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
                                hv_snap_from(state.start, end_pos, self.scale, crate::PERP_SNAP_PX)
                            } else { None };
                            if let Some((_, p)) = hv { end_pos = p; }

                            // Reject zero-length lines
                            let dx = end_pos.x - state.start.x;
                            let dy = end_pos.y - state.start.y;
                            if dx * dx + dy * dy < 1e-6 {
                                // Put state back, ignore this click
                                self.line_draw = Some(state);
                            } else {

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
                            // Auto-perpendicular constraint, same undo group.
                            if let Some((host, _)) = perp_host {
                                if !self.has_perp_conflict(new_line, host) {
                                    self.exec(Action::ApplyPerpendicular { a: new_line, b: host });
                                }
                            }
                            if let Some(target) = end_perp_target {
                                if !self.has_perp_conflict(new_line, target) {
                                    self.exec(Action::ApplyPerpendicular { a: new_line, b: target });
                                }
                            }
                            // Auto-collinear constraint emission.
                            if let Some((host, _)) = collinear_host {
                                let action = Action::ApplyCollinear { a: new_line, b: host };
                                if arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action).is_none() {
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
                                if arael_sketch_backend::conflicts::check_constraint_conflict(&self.sketch, &action).is_none() {
                                    self.exec(action);
                                }
                            }

                            // Chain: start next line from end of this one
                            self.line_draw = Some(LineDrawState {
                                start: end_pos,
                                snap_start: Some(SnapTarget::LineP2(new_line)),
                                chained: true,
                            });
                            } // end else (non-zero length)
                        } else {
                            // First click: start line, snap to nearby entity
                            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let start_pos = snap.map_or(mouse_sketch, |(pos, _)| pos);
                            self.line_draw = Some(LineDrawState {
                                start: start_pos,
                                snap_start: snap.map(|(_, t)| t),
                                chained: false,
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
                                let helper = self.sketch.add_helper_point(edge);
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
                                self.exec(Action::AddArc { start: state.start, end, mid: pos });
                                let new_arc = Ref::new(self.sketch.arcs.slot_count() as u32 - 1);

                                // Arc start_angle always corresponds to start click,
                                // end_angle to end click (direction stored in ccw flag)
                                let (start_ap, end_ap) = (ArcPoint::Start, ArcPoint::End);

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
                                    let helper = self.sketch.add_helper_point(pos);
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

                Tool::DrawRect => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.begin_group();
                        if let Some(state) = self.rect_draw.take() {
                            // Second click: opposite corner. Snap it, reject
                            // zero-area rects.
                            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let p2 = snap.map_or(mouse_sketch, |(p, _)| p);
                            let dx = p2.x - state.corner.x;
                            let dy = p2.y - state.corner.y;
                            if dx.abs() < 1e-6 || dy.abs() < 1e-6 {
                                self.rect_draw = Some(state);
                            } else {
                                let bl = state.corner;
                                let br = vect2d::new(p2.x, state.corner.y);
                                let tr = p2;
                                let tl = vect2d::new(state.corner.x, p2.y);
                                let corners = [bl, br, tr, tl];

                                let mut lines = [Ref::<Line>::new(0); 4];
                                for i in 0..4 {
                                    let a = corners[i];
                                    let b = corners[(i + 1) % 4];
                                    self.exec(Action::AddLine { p1: a, p2: b });
                                    lines[i] = Ref::new(self.sketch.lines.slot_count() as u32 - 1);
                                }

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
                                if let Some(s) = state.snap_corner {
                                    self.apply_snap_coincident(s, lines[0], true);
                                }
                                if let Some((_, s)) = snap {
                                    self.apply_snap_coincident(s, lines[1], false);
                                }
                            }
                        } else {
                            // First click: opposite corner, snap to nearby entity.
                            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let corner = snap.map_or(mouse_sketch, |(p, _)| p);
                            self.rect_draw = Some(RectDrawState {
                                corner,
                                snap_corner: snap.map(|(_, t)| t),
                            });
                        }
                    }
                }

                Tool::Fillet | Tool::Chamfer => {
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

                Tool::Dimension => {
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
                        // Phase 2: positioning with mouse, click to confirm
                        if let Some(ref kind) = self.dim_kind {
                            if matches!(kind, DimensionKind::ArcRadius(_) | DimensionKind::ArcRadiusB(_)) {
                                let arc_ref = match kind {
                                    DimensionKind::ArcRadius(r) | DimensionKind::ArcRadiusB(r) => Some(*r),
                                    _ => None,
                                };
                                if let Some(r) = arc_ref
                                    && self.sketch.arcs.contains(r) {
                                        let a = &self.sketch.arcs[r];
                                        if a.is_ellipse {
                                            let dx = mouse_sketch.x - a.center.value.x;
                                            let dy = mouse_sketch.y - a.center.value.y;
                                            let rot = a.rotation.value;
                                            let major = (dx * rot.cos() + dy * rot.sin()).abs();
                                            let minor = (-dx * rot.sin() + dy * rot.cos()).abs();
                                            let is_b = minor > major;
                                            // Switch dimension kind dynamically
                                            self.dim_kind = Some(if is_b {
                                                DimensionKind::ArcRadiusB(r)
                                            } else {
                                                DimensionKind::ArcRadius(r)
                                            });
                                            let measured = self.measure_dimension(self.dim_kind.as_ref().unwrap());
                                            self.dim_input = format!("{:.4}", measured);
                                            // Pick side based on projection onto the chosen axis
                                            let axis_angle = if is_b { rot + std::f64::consts::FRAC_PI_2 } else { rot };
                                            let proj = dx * axis_angle.cos() + dy * axis_angle.sin();
                                            self.dim_offset = vect2d::new(if proj >= 0.0 { 1.0 } else { -1.0 }, 0.0);
                                        } else {
                                            let abs_angle = (mouse_sketch.y - a.center.value.y)
                                                .atan2(mouse_sketch.x - a.center.value.x);
                                            let rel_angle = abs_angle - a.start_angle.value;
                                            self.dim_offset = vect2d::new(rel_angle, 0.0);
                                        }
                                        self.dim_text_along = 0.0;
                                    }
                            } else if let DimensionKind::ArcSweep(r) = kind {
                                let a = &self.sketch.arcs[*r];
                                let cx = a.center.value.x;
                                let cy = a.center.value.y;
                                let dist = ((mouse_sketch.x - cx).powi(2) + (mouse_sketch.y - cy).powi(2)).sqrt();
                                let offset_y = dist - a.radius.value;
                                let mouse_angle = (mouse_sketch.y - cy).atan2(mouse_sketch.x - cx);
                                let sa = a.start_angle.value;
                                let sweep = a.end_angle.value - sa;
                                let delta = rad2rad(mouse_angle - sa);
                                let along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                                self.dim_offset = vect2d::new(0.0, offset_y);
                                self.dim_text_along = along;
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
                                if let Some(DimensionKind::Angle(_, _, ref mut s)) = self.dim_kind
                                    && *s != sup {
                                        *s = sup;
                                        let measured = self.measure_dimension(&self.dim_kind.unwrap());
                                        self.dim_input = format!("{:.4}", measured);
                                    }
                            } else if let DimensionKind::LineAngle(r) = kind {
                                let p1 = self.sketch.lines[*r].p1.value;
                                let line_angle = {
                                    let l = &self.sketch.lines[*r];
                                    (l.p2.value.y - l.p1.value.y).atan2(l.p2.value.x - l.p1.value.x)
                                };
                                let dist = ((mouse_sketch.x - p1.x).powi(2)
                                    + (mouse_sketch.y - p1.y).powi(2)).sqrt();
                                let mouse_angle = (mouse_sketch.y - p1.y).atan2(mouse_sketch.x - p1.x);
                                let sweep = line_angle;
                                let delta = rad2rad(mouse_angle);
                                let along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                                self.dim_offset = vect2d::new(0.0, dist.max(0.3));
                                self.dim_text_along = along.clamp(-0.5, 0.5);
                            } else if let DimensionKind::ArcRotation(r) = kind {
                                // Same sector placement as LineAngle, anchored
                                // at the ellipse center and sweeping to the
                                // current rotation angle of the major axis.
                                let a = &self.sketch.arcs[*r];
                                let center = a.center.value;
                                let rotation = a.rotation.value;
                                let dist = ((mouse_sketch.x - center.x).powi(2)
                                    + (mouse_sketch.y - center.y).powi(2)).sqrt();
                                let mouse_angle = (mouse_sketch.y - center.y).atan2(mouse_sketch.x - center.x);
                                let sweep = rotation;
                                let delta = rad2rad(mouse_angle);
                                let along = if sweep.abs() > 1e-6 { delta / sweep - 0.5 } else { 0.0 };
                                self.dim_offset = vect2d::new(0.0, dist.max(0.3));
                                self.dim_text_along = along.clamp(-0.5, 0.5);
                            } else if let DimensionKind::ConcentricDistance(a_ref, _b_ref) = kind {
                                // Same encoding as the drag path: 2D text
                                // anchor in world coords relative to the
                                // shared center. Lets the user position the
                                // text anywhere before clicking to commit.
                                let center = self.sketch.arcs[*a_ref].center.value;
                                self.dim_offset = vect2d::new(mouse_sketch.x - center.x,
                                                               mouse_sketch.y - center.y);
                                self.dim_text_along = 0.0;
                            } else if matches!(kind, DimensionKind::HDistance(..) | DimensionKind::VDistance(..)) {
                                let horizontal = matches!(kind, DimensionKind::HDistance(..));
                                let (p1, p2) = self.dim_endpoints(kind);
                                let mid = vect2d::new((p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0);
                                let offset_val = if horizontal {
                                    mouse_sketch.y - mid.y
                                } else {
                                    mouse_sketch.x - mid.x
                                };
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
                                let along = ((mouse_sketch.x - qmx) * ddx + (mouse_sketch.y - qmy) * ddy) / (dlen * dlen);
                                self.dim_offset = vect2d::new(0.0, offset_val);
                                self.dim_text_along = along;
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
                        for (i, dim) in self.sketch.dimensions.iter().enumerate() {
                            let (ts, te) = self.dim_text_segment(dim);
                            let d = Self::screen_point_to_segment_dist(mouse_screen, ts, te);
                            if d < 15.0 {
                                self.dim_input = Self::dim_edit_string(dim);
                                self.dim_kind = Some(dim.kind);
                                self.dim_offset = dim.offset;
                                self.dim_edit_index = Some(i);
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

            // Build constraint markers and draw canvas
            if self.show_constraints {
                self.build_constraint_markers();
            } else {
                self.constraint_markers.clear();
            }
            self.draw_canvas(&painter, rect, mouse_screen);

            // Dimension preview while placing (not when editing an existing dimension)
            if (self.dim_placing || (self.dim_editing && self.dim_edit_index.is_none())) && self.dim_kind.is_some() {
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
                let p1 = self.to_screen(state.start);
                // Honor end-point snap in the preview so the user sees
                // exactly where the endpoint will land. Suppress any snap
                // whose target position coincides with the segment's start
                // -- a zero-length line is rejected anyway, and snapping
                // the end back to the start just paints a misleading box.
                let end_snap = self.find_snap_target(mouse_sketch, hit_threshold)
                    .filter(|(p, _)| {
                        let dx = p.x - state.start.x;
                        let dy = p.y - state.start.y;
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
                // Any line passing through state.start is a candidate --
                // the chained previous segment, a line whose body the
                // start was placed on, a coincident endpoint elsewhere.
                // Pick the host that gives the closest-to-exact 90.
                let perp = if perp_eligible {
                    self.find_best_perp_host_at(state.start, mouse_sketch, crate::PERP_SNAP_PX, None)
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
                            let perp_p2 = vect2d::new(state.start.x - hdy, state.start.y + hdx);
                            Some(line_line_intersection(state.start, perp_p2, ol.p1.value, ol.p2.value))
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
                            let host_ref = state.snap_start.and_then(EditorApp::perp_host_from_snap);
                            if host_ref == Some(target) {
                                None
                            } else if self.has_perp_conflict(target, target) {
                                None
                            } else {
                                let tl = &self.sketch.lines[target];
                                self.try_perp_end_snap(state.start, tl.p1.value, tl.p2.value, p_on_target, crate::PERP_SNAP_PX)
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
                    self.find_best_collinear_host_at(state.start, mouse_sketch, crate::PERP_SNAP_PX, None)
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
                    hv_snap_from(state.start, mouse_sketch, self.scale, crate::PERP_SNAP_PX)
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
                        state.start, drawn_free,
                        self.scale, self.colors.constraint_marker_selected,
                    );
                }
                if let Some((target, foot)) = end_perp {
                    let tl = &self.sketch.lines[target];
                    draw_perp_corner_marker(
                        &painter, self.to_screen(foot),
                        tl.p1.value, tl.p2.value,
                        foot, state.start,
                        self.scale, self.colors.constraint_marker_selected,
                    );
                }
            } else if matches!(self.tool,
                Tool::DrawLine | Tool::DrawPoint
                | Tool::DrawCircle | Tool::DrawArc | Tool::DrawRect)
                && self.circle_draw.is_none() && self.arc_draw.is_none()
                && self.rect_draw.is_none()
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

            // Circle preview
            if let Some(ref state) = self.circle_draw {
                let center = self.to_screen(state.center);
                // Live snap lookup on the edge-point click.
                let edge_snap = self.find_snap_target(mouse_sketch, hit_threshold)
                    .filter(|(p, _)| {
                        let dx = p.x - state.center.x;
                        let dy = p.y - state.center.y;
                        dx * dx + dy * dy > 1e-12
                    });
                let edge_sketch = edge_snap.map_or(mouse_sketch, |(p, _)| p);
                let edge_pt = self.to_screen(edge_sketch);
                let radius_px = ((edge_sketch.x - state.center.x).powi(2)
                    + (edge_sketch.y - state.center.y).powi(2)).sqrt() as f32 * self.scale;
                painter.circle_stroke(center, radius_px,
                    egui::Stroke::new(1.5, self.colors.preview_line));
                painter.circle_filled(center, 4.0, self.colors.endpoint);

                if let Some(ref t) = state.snap_center { draw_snap_hint(center, t); }
                if let Some((_, t)) = &edge_snap { draw_snap_hint(edge_pt, t); }
                if edge_snap.is_none() { draw_cursor_cross(edge_pt); }
            }

            // Rect preview
            if let Some(ref state) = self.rect_draw {
                let corner_screen = self.to_screen(state.corner);
                painter.circle_filled(corner_screen, 4.0, self.colors.endpoint);
                if let Some(ref t) = state.snap_corner { draw_snap_hint(corner_screen, t); }

                let end_snap = self.find_snap_target(mouse_sketch, hit_threshold)
                    .filter(|(p, _)| {
                        (p.x - state.corner.x).abs() > 1e-6
                            && (p.y - state.corner.y).abs() > 1e-6
                    });
                let end_sketch = end_snap.map_or(mouse_sketch, |(p, _)| p);
                let end_pt = self.to_screen(end_sketch);
                let c1 = self.to_screen(state.corner);
                let c2 = egui::Pos2::new(end_pt.x, c1.y);
                let c3 = end_pt;
                let c4 = egui::Pos2::new(c1.x, end_pt.y);
                let stroke = egui::Stroke::new(1.5, self.colors.preview_line);
                painter.line_segment([c1, c2], stroke);
                painter.line_segment([c2, c3], stroke);
                painter.line_segment([c3, c4], stroke);
                painter.line_segment([c4, c1], stroke);
                if let Some((_, t)) = &end_snap { draw_snap_hint(end_pt, t); }
                if end_snap.is_none() { draw_cursor_cross(end_pt); }
            }

            // Arc preview
            if let Some(ref state) = self.arc_draw {
                let start_screen = self.to_screen(state.start);
                painter.circle_filled(start_screen, 4.0, self.colors.endpoint);
                if let Some(ref t) = state.snap_start { draw_snap_hint(start_screen, t); }
                if let Some((end, ref snap_end)) = state.end {
                    let end_screen = self.to_screen(end);
                    painter.circle_filled(end_screen, 4.0, self.colors.endpoint);
                    if let Some(t) = snap_end { draw_snap_hint(end_screen, t); }
                    // Live snap for the mid-point click.
                    let mid_snap = self.find_snap_target(mouse_sketch, hit_threshold)
                        .filter(|(p, _)| {
                            let d_s = (p.x - state.start.x).powi(2) + (p.y - state.start.y).powi(2);
                            let d_e = (p.x - end.x).powi(2) + (p.y - end.y).powi(2);
                            d_s > 1e-12 && d_e > 1e-12
                        });
                    let mid_sketch = mid_snap.map_or(mouse_sketch, |(p, _)| p);
                    // Preview arc through start, end, and mid (or mouse).
                    if let Some((c, r, sa, ea, ccw)) = circumscribed_arc(state.start, end, mid_sketch) {
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
                            let dx = p.x - state.start.x;
                            let dy = p.y - state.start.y;
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
                Tool::DrawRect => if self.rect_draw.is_some() {
                    "Rect: click to place opposite corner."
                } else {
                    "Rect: click to place first corner."
                },
                Tool::Fillet => if self.fillet_pending.is_some() {
                    "Fillet: type radius and press Enter. Escape to cancel."
                } else {
                    "Fillet: click a connecting endpoint, or select two lines. Escape to cancel."
                },
                Tool::Chamfer => if self.fillet_pending.is_some() {
                    "Chamfer: type corner-to-end distance and press Enter. Escape to cancel."
                } else {
                    "Chamfer: click a connecting endpoint, or select two lines. Escape to cancel."
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

            // DOF + cost + version at bottom-right
            let dof_str = match self.dof_display {
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
        });

        // Dimension-value input overlay: floats over the canvas at the
        // dim label position so the user types where they're already
        // looking, instead of shuttling attention to the side panel.
        if self.dim_editing
            && let Some(anchor) = self.dim_input_anchor_screen()
        {
            // Offset down-right of the label so the input doesn't cover
            // the dimension text itself.
            let offset = egui::Vec2::new(12.0, 12.0);
            egui::Area::new("dim_input_overlay".into())
                .order(egui::Order::Foreground)
                .fixed_pos(anchor + offset)
                .show(ctx, |ui| {
                    egui::Frame::popup(ui.style()).show(ui, |ui| {
                        ui.set_min_width(180.0);
                        self.render_dim_input(ui);
                    });
                });
        }
    }
}

impl EditorApp {
    /// Screen-space anchor for the dimension-value input overlay:
    /// midpoint of the dim text segment. For an in-progress placement
    /// we synthesise a temporary `Dimension` with the current kind /
    /// offset / text_along / measured value so `dim_text_segment`
    /// computes the same text position the preview is showing.
    fn dim_input_anchor_screen(&self) -> Option<egui::Pos2> {
        let dim_ref = if let Some(idx) = self.dim_edit_index {
            self.sketch.dimensions.get(idx).cloned()
        } else if let Some(kind) = self.dim_kind {
            let value = self.measure_dimension(&kind);
            Some(arael_sketch_solver::Dimension {
                kind,
                value,
                offset: self.dim_offset,
                text_along: self.dim_text_along,
                name: String::new(),
                expr_str: None,
                broken: false,
                derived: false,
                range: None,
            })
        } else {
            None
        };
        let dim = dim_ref?;
        let (ts, te) = self.dim_text_segment(&dim);
        Some(egui::Pos2::new((ts.x + te.x) * 0.5, (ts.y + te.y) * 0.5))
    }

    /// Render the dimension-value input (derived checkbox + text
    /// edit + submit handling). Extracted from the side panel so it
    /// can be hosted in a floating `egui::Area` over the canvas,
    /// near the dimension label where the user is looking.
    fn render_dim_input(&mut self, ui: &mut egui::Ui) {
        // Derived checkbox is meaningless during a fillet edit --
        // the radius dim is always driven (it's the control the user
        // is actively setting), and its value flows from the typed
        // expression or the fallback literal.
        if self.fillet_pending.is_none() {
            ui.checkbox(&mut self.dim_derived, "Derived");
        }
        // Edge-detect the checkbox: on false -> true, back up what
        // the user had typed and swap in the measured value; on
        // true -> false, restore the backup so they see their
        // expression / number unchanged. Keeps the derived toggle
        // non-destructive.
        if self.dim_derived && !self.dim_derived_prev {
            self.dim_input_backup = self.dim_input.clone();
            let measured = if let Some(idx) = self.dim_edit_index {
                self.sketch.dimensions.get(idx).map(|d| d.value).unwrap_or(0.0)
            } else if let Some(kind) = self.dim_kind {
                self.measure_dimension(&kind)
            } else {
                0.0
            };
            self.dim_input = format!("{:.4}", measured);
        } else if !self.dim_derived && self.dim_derived_prev {
            self.dim_input = std::mem::take(&mut self.dim_input_backup);
            // After restoring, select all so the user can immediately
            // overtype if they wish.
            self.dim_select_all_on_uncheck = true;
        }
        self.dim_derived_prev = self.dim_derived;
        // When derived is checked the value is locked to what the
        // sketch currently measures; show it read-only so the user
        // can't mistakenly type a number that won't be used.
        let response = ui.add(
            egui::TextEdit::singleline(&mut self.dim_input)
                .interactive(!self.dim_derived),
        );
        // Live fillet preview: whenever the text or corner set
        // changes, restore the pre-fillet sketch and reapply every
        // pending corner. reapply_fillets is a no-op when nothing
        // meaningful changed (signature match).
        if self.fillet_pending.is_some() && response.changed() {
            self.reapply_fillets();
        }
        // Select all text when entering edit mode (one-shot flag)
        if (self.dim_select_all || self.dim_select_all_on_uncheck) && response.has_focus() {
            self.dim_select_all = false;
            self.dim_select_all_on_uncheck = false;
            let mut state = egui::TextEdit::load_state(ui.ctx(), response.id).unwrap_or_default();
            state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                egui::text::CCursor::new(0),
                egui::text::CCursor::new(self.dim_input.len()),
            )));
            egui::TextEdit::store_state(ui.ctx(), response.id, state);
        }
        let enter_pressed = ui.input(|i| i.key_pressed(egui::Key::Enter));
        if enter_pressed && self.fillet_pending.is_some() {
            // Fillet commit: reapply already baked the current input
            // into the sketch, so Enter just finalises the session.
            // No UpdateDimension is queued -- the single fillet undo
            // group built by reapply stays as the canonical record.
            self.reapply_fillets();
            self.fillet_pending = None;
            self.dim_editing = false;
            self.dim_edit_index = None;
            self.dim_kind = None;
            self.selection.clear();
            return;
        }
        if enter_pressed || (response.lost_focus() && enter_pressed) {
            let mut input = self.dim_input.trim().to_string();
            // Range syntax: `>= V`, `<= V`, `LO to HI`. If the
            // input matches, short-circuit the numeric / expr
            // paths and build a ranged dimension.
            let range_result = arael_sketch_backend::commands::parse_range_input(&self.sketch, &input);
            let is_range = matches!(range_result, Ok(Some(_)));
            // Snapshot prefix `=expr`: evaluate now, rewrite `input`
            // as the resulting literal so the numeric branch below
            // handles it. Failure falls through to is_expr, which
            // will catch the parse error with a clearer message.
            if !is_range
                && let Some(expr) = input.strip_prefix('=')
                && let Ok(v) = arael_sketch_backend::commands::eval_expr(&self.sketch, expr.trim())
            {
                input = format!("{}", v);
            }
            let is_numeric = !is_range && input.parse::<f64>().is_ok();
            let is_expr = !is_range && !is_numeric && arael_sym::parse(&input).is_ok();

            let mut success = false;
            if let Err(e) = &range_result {
                self.status_error = Some(format!("Range parse: {}", e));
            } else if is_range {
                let rb = range_result.unwrap().unwrap();
                if self.dim_derived {
                    self.status_error = Some("Range dimensions are not compatible with `derived`".into());
                } else if let Some(edit_idx) = self.dim_edit_index.take() {
                    // Editing existing dim -> re-bind as range.
                    let measured = self.sketch.dimensions.get(edit_idx)
                        .map(|d| d.value).unwrap_or(0.0);
                    self.begin_group();
                    self.exec(Action::UpdateDimension {
                        index: edit_idx, value: measured, expr: None, range: Some(rb),
                    });
                    success = true;
                } else if let Some(kind) = self.dim_kind {
                    self.begin_group();
                    let measured = self.measure_dimension(&kind);
                    // LineLineDistance still needs its paired
                    // Parallel constraint before the range dim.
                    if let DimensionKind::LineLineDistance(a, b) = kind {
                        let has_parallel = self.sketch.parallel.iter().any(|p|
                            (p.a == a && p.b == b) || (p.a == b && p.b == a));
                        if !has_parallel {
                            self.exec(Action::ApplyParallel { a, b });
                        }
                    }
                    // ConcentricDistance is self-contained but we emit a
                    // paired `Concentric` for visibility in `list`.
                    if let DimensionKind::ConcentricDistance(a, b) = kind {
                        let has_concentric = self.sketch.concentric.iter().any(|c|
                            (c.a == a && c.b == b) || (c.a == b && c.b == a));
                        if !has_concentric {
                            self.exec(Action::ApplyConcentric { a, b });
                        }
                    }
                    let n_dims_before = self.sketch.dimensions.len();
                    self.exec(Action::AddDimension {
                        kind, value: measured, expr: None, derived: false, range: Some(rb),
                    });
                    if self.sketch.dimensions.len() > n_dims_before
                        && let Some(d) = self.sketch.dimensions.last_mut() {
                            d.offset = self.dim_offset;
                            d.text_along = self.dim_text_along;
                        }
                    success = true;
                }
            } else if is_numeric || is_expr || (input.is_empty() && self.dim_derived) {
                self.begin_group();
                if let Some(edit_idx) = self.dim_edit_index.take() {
                    // Editing existing: update in place (preserves name)
                    if self.dim_derived != self.sketch.dimensions[edit_idx].derived {
                        // Toggle derived status
                        let name = self.sketch.dimensions[edit_idx].name.clone();
                        if self.dim_derived {
                            self.run_commands(&format!("set_derived {}", name));
                        } else {
                            let val = if is_numeric { input.parse::<f64>().unwrap() } else { self.sketch.dimensions[edit_idx].value };
                            self.run_commands(&format!("set_driven {} {}", name, val));
                        }
                        success = true;
                    } else if is_numeric {
                        let value = input.parse::<f64>().unwrap();
                        self.exec(Action::UpdateDimension { index: edit_idx, value, expr: None, range: None });
                        success = true;
                    } else if let Err(e) = self.sketch.validate_expr(&input) {
                        self.status_error = Some(format!("Expression error: {}", e));
                        self.dim_edit_index = Some(edit_idx); // restore
                    } else {
                        self.exec(Action::UpdateDimension {
                            index: edit_idx, value: 0.0,
                            expr: Some(input.clone()), range: None,
                        });
                        success = true;
                    }
                } else if let Some(kind) = self.dim_kind {
                    let is_dup = self.sketch.dimensions.iter().any(|d| d.kind == kind);
                    if is_dup {
                        self.status_error = Some("Dimension already exists".into());
                    } else if is_numeric || (input.is_empty() && self.dim_derived) {
                        let value = input.parse::<f64>().unwrap_or(0.0);
                        let n_dims_before = self.sketch.dimensions.len();
                        // LineLineDistance requires a paired Parallel
                        // constraint; emit it first if not already present.
                        if let DimensionKind::LineLineDistance(a, b) = kind {
                            let has_parallel = self.sketch.parallel.iter().any(|p|
                                (p.a == a && p.b == b) || (p.a == b && p.b == a));
                            if !has_parallel {
                                self.exec(Action::ApplyParallel { a, b });
                            }
                        }
                        // ConcentricDistance: also emit Concentric for
                        // list-visibility (the dim enforces concentricity
                        // itself, so it would work without this, but
                        // users expect to see the pairing).
                        if let DimensionKind::ConcentricDistance(a, b) = kind {
                            let has_concentric = self.sketch.concentric.iter().any(|c|
                                (c.a == a && c.b == b) || (c.a == b && c.b == a));
                            if !has_concentric {
                                self.exec(Action::ApplyConcentric { a, b });
                            }
                        }
                        self.exec(Action::AddDimension { kind, value, expr: None, derived: self.dim_derived, range: None });
                        if self.sketch.dimensions.len() > n_dims_before
                            && let Some(d) = self.sketch.dimensions.last_mut() {
                                d.offset = self.dim_offset;
                                d.text_along = self.dim_text_along;
                            }
                        success = true;
                    } else {
                        if let Err(e) = self.sketch.validate_expr(&input) {
                            self.status_error = Some(format!("Expression error: {}", e));
                        } else {
                            let n_dims_before = self.sketch.dimensions.len();
                            if let DimensionKind::LineLineDistance(a, b) = kind {
                                let has_parallel = self.sketch.parallel.iter().any(|p|
                                    (p.a == a && p.b == b) || (p.a == b && p.b == a));
                                if !has_parallel {
                                    self.exec(Action::ApplyParallel { a, b });
                                }
                            }
                            if let DimensionKind::ConcentricDistance(a, b) = kind {
                                let has_concentric = self.sketch.concentric.iter().any(|c|
                                    (c.a == a && c.b == b) || (c.a == b && c.b == a));
                                if !has_concentric {
                                    self.exec(Action::ApplyConcentric { a, b });
                                }
                            }
                            self.exec(Action::AddDimension {
                                kind, value: 0.0, expr: Some(input.clone()), derived: self.dim_derived, range: None,
                            });
                            if self.sketch.dimensions.len() > n_dims_before
                                && let Some(d) = self.sketch.dimensions.last_mut() {
                                    d.offset = self.dim_offset;
                                    d.text_along = self.dim_text_along;
                                }
                            success = true;
                        }
                    }
                }
            } else if !input.is_empty() {
                self.status_error = Some(format!("Invalid value or expression: {}", input));
            }
            if success {
                self.dim_editing = false;
                self.dim_placing = false;
                self.dim_edit_index = None;
                self.dim_kind = None;
                self.selection.clear();
                // Fillet-in-flight: the dim commit finalises the
                // fillet, so drop the pre-fillet snapshot. A later
                // Escape would otherwise roll the whole fillet back.
                self.fillet_pending = None;
            }
        } else if !response.has_focus() && self.dim_editing {
            response.request_focus();
        }
    }
}
