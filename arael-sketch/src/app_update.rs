// eframe::App implementation for EditorApp.

use eframe::egui;
use crate::tools::*;
use arael_sketch_backend::actions::Action;
use crate::EditorApp;

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

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.update_app(ctx);
    }
}

impl EditorApp {
    /// One frame of the application: panels, canvas input, drawing,
    /// overlays. Separated from the eframe::App impl so the GUI test
    /// harness can drive frames headlessly (eframe::Frame is not
    /// constructible outside eframe).
    pub(crate) fn update_app(&mut self, ctx: &egui::Context) {
        // Handle exit request
        if self.exit_requested {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }

        // Poll background DOF computation

        // A tool switch mid-drag (keyboard shortcut or toolbar click)
        // leaves the Select arm before the drag can end; the apparatus
        // would stay in the sketch forever. Cancel the gesture first.
        if self.grab.is_some() && !matches!(self.tool, crate::tools::Tool::Select) {
            self.cancel_drag();
        }

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
            self.cancel_drag();
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

        self.side_panel(ctx);
        self.params_panel(ctx);
        self.command_panel(ctx);

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

            // Keyboard shortcuts (skip when any text field has focus).
            // One ctrl/cmd gate for the whole block: Ctrl+S/C/V/X/A and
            // friends belong to the system, not the tool palette.
            if !ctx.wants_keyboard_input()
                && !ui.input(|i| i.modifiers.ctrl || i.modifiers.mac_cmd) {
            // Select has no key: Escape returns to it.
            if ui.input(|i| i.key_pressed(egui::Key::P)) { self.tool = Tool::DrawPoint; }
            if ui.input(|i| i.key_pressed(egui::Key::L)) {
                self.tool = Tool::DrawLine;
                self.line_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::O)) {
                self.tool = Tool::DrawCircle;
                self.circle_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::A)) {
                self.tool = Tool::DrawArc;
                self.arc_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::R)) {
                self.tool = Tool::DrawRect;
                self.rect_draw = None;
            }
            // F cycles Fillet <-> Chamfer.
            if ui.input(|i| i.key_pressed(egui::Key::F)) {
                self.tool = if self.tool == Tool::Fillet { Tool::Chamfer } else { Tool::Fillet };
                self.selection.clear();
            }
            // B cycles Break (Split) <-> Trim.
            if ui.input(|i| i.key_pressed(egui::Key::B)) {
                self.tool = if self.tool == Tool::Split { Tool::Trim } else { Tool::Split };
                self.selection.clear();
            }
            if ui.input(|i| i.key_pressed(egui::Key::H)) { self.try_apply_or_enter_mode(ConstraintType::Horizontal); }
            if ui.input(|i| i.key_pressed(egui::Key::V)) { self.try_apply_or_enter_mode(ConstraintType::Vertical); }
            if ui.input(|i| i.key_pressed(egui::Key::C)) { self.try_apply_or_enter_mode(ConstraintType::Coincident); }
            if ui.input(|i| i.key_pressed(egui::Key::K)) { self.try_apply_or_enter_mode(ConstraintType::Lock); }
            if ui.input(|i| i.key_pressed(egui::Key::T)) { self.try_apply_or_enter_mode(ConstraintType::Tangent); }
            if ui.input(|i| i.key_pressed(egui::Key::M)) { self.try_apply_or_enter_mode(ConstraintType::Midpoint); }
            if ui.input(|i| i.key_pressed(egui::Key::S)) { self.try_apply_or_enter_mode(ConstraintType::Symmetry); }
            if ui.input(|i| i.key_pressed(egui::Key::Equals)) { self.try_apply_or_enter_mode(ConstraintType::EqualLength); }
            if ui.input(|i| i.key_pressed(egui::Key::X)) { self.try_apply_or_enter_mode(ConstraintType::ToggleConstruction); }
            if ui.input(|i| i.key_pressed(egui::Key::D)) {
                self.tool = Tool::Dimension;
                self.dim_editing = false;
                self.dim_kind = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::Slash)) {
                self.show_command = true;
                self.command_focus = true;
            }
            // Escape resets tool/selection/gestures -- gated on
            // keyboard focus like every other shortcut, so typing in
            // the command panel or an overlay never flips the tool
            // (those fields handle their own Escape).
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
                self.box_select_start = None;
                self.dim_editing = false;
                self.dim_kind = None;
                self.dim_placing = false;
                self.dim_edit_did = None;
                self.status_error = None;
                self.tool = Tool::Select;
                // Cancel any live grab and keep the still-held pointer
                // from re-grabbing at the press origin next frame.
                self.suppress_drag_regrab = true;
                self.cancel_drag();
            }
            } // !wants_keyboard_input && no ctrl/cmd

            // Delete selected entities/constraints with Backspace/Delete
            // (skip when editing dimension text — Backspace edits the text field)
            if self.grab.is_none() && !ctx.wants_keyboard_input() && ui.input(|i| i.key_pressed(egui::Key::Backspace) || i.key_pressed(egui::Key::Delete)) {
                let sel = self.selection.clone();
                if !sel.is_empty() {
                    self.begin_group();
                    for s in &sel {
                        match *s {
                            Selection::Point(r) => { self.exec(Action::DeletePoint { point: r }); }
                            Selection::Line(r) => { self.exec(Action::DeleteLine { line: r }); }
                            Selection::Arc(r) => { self.exec(Action::DeleteArc { arc: r }); }
                            Selection::Constraint(id) => { self.delete_constraint(id); }
                            Selection::Dimension(did) => { self.exec(Action::RemoveDimension { did }); }
                            _ => {} // endpoints aren't deletable on their own
                        }
                    }
                    self.selection.clear();
                }
            }

            // Undo/redo keyboard shortcuts
            let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.mac_cmd);
            let shift = ui.input(|i| i.modifiers.shift);
            if self.grab.is_some() && ctrl && ui.input(|i| i.key_pressed(egui::Key::Z)) {
                // Undo/redo mid-drag would restore a sketch without the
                // drag apparatus and panic on the next drag frame.
            } else if ctrl && shift && ui.input(|i| i.key_pressed(egui::Key::Z)) {
                if let Some((restored, cur)) = self.history.redo() {
                    self.apply_history_restore(restored, cur);
                }
            } else if ctrl && ui.input(|i| i.key_pressed(egui::Key::Z))
                && let Some((restored, cur)) = self.history.undo() {
                    self.apply_history_restore(restored, cur);
                }
            if ctrl && ui.input(|i| i.key_pressed(egui::Key::S)) {
                self.spawn_save_dialog();
            }
            if ctrl && ui.input(|i| i.key_pressed(egui::Key::O)) {
                self.spawn_open_dialog();
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
                Tool::Select => self.handle_select_input(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold),
                Tool::DrawPoint => self.handle_draw_point(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold),
                Tool::DrawLine => self.handle_draw_line(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold),
                Tool::DrawCircle => self.handle_draw_circle(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold),
                Tool::DrawArc => self.handle_draw_arc(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold),
                Tool::DrawRect => self.handle_draw_rect(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold),
                Tool::Fillet | Tool::Chamfer => self.handle_fillet_chamfer(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold),
                Tool::Split | Tool::Trim => self.handle_split_trim(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold),
                Tool::ConstraintMode(ct) => self.handle_constraint_mode(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold, ct),
                Tool::Dimension => self.handle_dimension_tool(ui, ctx, &response, mouse_screen, mouse_sketch, hit_threshold),
            }

            // Build constraint markers and draw canvas
            if self.show_constraints {
                self.build_constraint_markers();
            } else {
                self.constraint_markers.clear();
            }
            self.draw_canvas(&painter, rect, mouse_screen);

            self.draw_canvas_overlays(ui, &painter, rect, mouse_screen, mouse_sketch, hit_threshold);
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
