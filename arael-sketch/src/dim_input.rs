// Dimension-value input overlay: the floating editor that appears at
// the dimension label during placement and double-click editing.

use eframe::egui;
use arael_sketch_solver::*;
use arael_sketch_backend::actions::Action;
use crate::EditorApp;

impl EditorApp {
    /// Screen-space anchor for the dimension-value input overlay:
    /// midpoint of the dim text segment. For an in-progress placement
    /// we synthesise a temporary `Dimension` with the current kind /
    /// offset / text_along / measured value so `dim_text_segment`
    /// computes the same text position the preview is showing.
    pub(crate) fn dim_input_anchor_screen(&self) -> Option<egui::Pos2> {
        // Scale session: the factor input floats at the scale center.
        if self.scale_pending.is_some()
            && let Some(c) = self.scale_center
        {
            return Some(self.to_screen(c));
        }
        // Creation session (circle / ellipse / rect): the value input
        // trails the cursor below-right so it never sits under the
        // next click.
        if let Some(cursor) = self.creation_session_cursor() {
            return Some(egui::Pos2::new(cursor.x + 18.0, cursor.y + 18.0));
        }
        let dim_ref = if let Some(idx) = self.dim_edit_did.and_then(|d| self.sketch.dimension_index_by_did(d)) {
            self.sketch.dimensions.get(idx).cloned()
        } else if let Some(kind) = self.dim_kind {
            let value = self.measure_dimension(&kind);
            Some(arael_sketch_solver::Dimension {
                did: 0,
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
    pub(crate) fn render_dim_input(&mut self, ui: &mut egui::Ui) {
        // Derived checkbox is meaningless during a fillet, scale or
        // creation-session edit -- the value being typed is the
        // control itself (radius / factor / axis length / side), not
        // a dimension that could go derived.
        let creation = self.creation_session_cursor().is_some();
        if self.fillet_pending.is_none() && self.scale_pending.is_none() && !creation {
            ui.checkbox(&mut self.dim_derived, "Derived");
        }
        // Edge-detect the checkbox: on false -> true, back up what
        // the user had typed and swap in the measured value; on
        // true -> false, restore the backup so they see their
        // expression / number unchanged. Keeps the derived toggle
        // non-destructive.
        if self.dim_derived && !self.dim_derived_prev {
            self.dim_input_backup = self.dim_input.clone();
            let measured = if let Some(idx) = self.dim_edit_did.and_then(|d| self.sketch.dimension_index_by_did(d)) {
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
        //
        // Two-field sessions: the ellipse axis phase (semi-major
        // length + absolute axis angle in degrees) and the rect (width
        // + height). Tab moves between the fields (egui focus
        // traversal); the second field's text lives in the session
        // state and is swapped in for rendering.
        #[derive(PartialEq, Clone, Copy)]
        enum Second { None, EllipseAngle, RectHeight }
        let second = if self.ellipse_draw.as_ref().is_some_and(|s| !s.axis_fixed) {
            Second::EllipseAngle
        } else if self.rect_draw.is_some() {
            Second::RectHeight
        } else {
            Second::None
        };
        let mut second_response = None;
        let response = if second != Second::None {
            let mut second_text = match second {
                Second::EllipseAngle => std::mem::take(&mut self.ellipse_draw.as_mut().unwrap().angle_text),
                _ => std::mem::take(&mut self.rect_draw.as_mut().unwrap().height_text),
            };
            let label = if second == Second::EllipseAngle { "deg" } else { "h" };
            let mut first_response = None;
            ui.horizontal(|ui| {
                first_response = Some(ui.add(egui::TextEdit::singleline(&mut self.dim_input).desired_width(72.0)));
                second_response = Some(ui.add(egui::TextEdit::singleline(&mut second_text).desired_width(72.0)));
                ui.label(label);
            });
            match second {
                Second::EllipseAngle => self.ellipse_draw.as_mut().unwrap().angle_text = second_text,
                _ => self.rect_draw.as_mut().unwrap().height_text = second_text,
            }
            first_response.unwrap()
        } else {
            ui.add(
                egui::TextEdit::singleline(&mut self.dim_input)
                    .interactive(!self.dim_derived),
            )
        };
        // Live fillet preview: whenever the text or corner set
        // changes, restore the pre-fillet sketch and reapply every
        // pending corner. reapply_fillets is a no-op when nothing
        // meaningful changed (signature match).
        if self.fillet_pending.is_some() && response.changed() {
            self.reapply_fillets();
        }
        // Live scale preview, same shape.
        if self.scale_pending.is_some() && response.changed() {
            self.reapply_scale();
        }
        // Live ellipse axis length: any typed text takes the active
        // length over from mouse tracking (major while aiming the
        // axis, minor after); the length itself updates only once the
        // text evaluates, so a half-typed "." or "sqrt(" keeps the
        // last good value instead of being clobbered by the mouse.
        // Cleared input hands control back to the mouse (no dim on
        // commit). Typing also cancels the select-all this frame's
        // tracking queued -- it would select the fresh text and the
        // next keystroke would replace it.
        if response.changed() && creation {
            let raw = self.dim_input.trim().to_string();
            let parsed = self.parse_axis_input(&raw);
            self.dim_select_all = false;
            let typed = if raw.is_empty() { None } else { Some(raw) };
            if let Some(state) = self.ellipse_draw.as_mut() {
                if state.axis_fixed {
                    state.typed_ry = typed;
                    if let Some((v, _)) = parsed { state.ry = v; }
                } else {
                    state.typed_rx = typed;
                    if let Some((v, _)) = parsed { state.rx = v; }
                }
            } else if let Some(state) = self.circle_draw.as_mut() {
                state.typed_r = typed;
                if let Some((v, _)) = parsed { state.r = v; }
                if state.typed_r.is_some() { state.live_snap = None; }
            } else if let Some(state) = self.rect_draw.as_mut() {
                state.typed_w = typed;
                if let Some((v, _)) = parsed { state.w = v; }
                if state.typed_w.is_some() { state.live_snap = None; }
            } else if let Some(state) = self.arc_draw.as_mut() {
                state.typed_r = typed;
                if let Some((v, _)) = parsed { state.r = v; }
                if state.typed_r.is_some() {
                    state.live_snap = None;
                    state.tangent = None;
                }
            }
        }
        // Second field (ellipse angle / rect height): typing takes it
        // over from the mouse (the value follows once the text
        // evaluates); an emptied field hands it back. While it still
        // mirrors the mouse, keep it fully selected under focus so a
        // keystroke replaces the live value.
        if let Some(sr) = &second_response {
            if sr.changed() {
                match second {
                    Second::EllipseAngle => {
                        let raw = self.ellipse_draw.as_ref().unwrap().angle_text.trim().to_string();
                        let parsed = self.parse_typed_input(&raw, false);
                        let state = self.ellipse_draw.as_mut().unwrap();
                        state.typed_angle = if raw.is_empty() { None } else { Some(raw) };
                        if state.typed_angle.is_some() && let Some((deg, _)) = parsed {
                            let (s, c) = deg.to_radians().sin_cos();
                            state.dir = arael::vect::vect2d::new(c, s);
                            state.hv = None;
                            state.live_snap = None;
                        }
                    }
                    _ => {
                        let raw = self.rect_draw.as_ref().unwrap().height_text.trim().to_string();
                        let parsed = self.parse_axis_input(&raw);
                        let state = self.rect_draw.as_mut().unwrap();
                        state.typed_h = if raw.is_empty() { None } else { Some(raw) };
                        if let Some((v, _)) = parsed { state.h = v; }
                        if state.typed_h.is_some() { state.live_snap = None; }
                    }
                }
            }
            let (untyped, len) = match second {
                Second::EllipseAngle => {
                    let s = self.ellipse_draw.as_ref().unwrap();
                    (s.typed_angle.is_none(), s.angle_text.len())
                }
                _ => {
                    let s = self.rect_draw.as_ref().unwrap();
                    (s.typed_h.is_none(), s.height_text.len())
                }
            };
            if sr.has_focus() && untyped {
                let mut st = egui::TextEdit::load_state(ui.ctx(), sr.id).unwrap_or_default();
                st.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                    egui::text::CCursor::new(0),
                    egui::text::CCursor::new(len),
                )));
                egui::TextEdit::store_state(ui.ctx(), sr.id, st);
            }
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
        // Escape cancels the overlay edit only -- the tool and
        // selection stay (the generic Escape handler is gated on
        // keyboard focus and does not fire while typing here).
        if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
            if self.fillet_pending.is_some() {
                self.cancel_pending_fillet();
            } else if self.scale_pending.is_some() {
                self.cancel_pending_scale();
            } else if self.ellipse_draw.is_some() {
                self.cancel_ellipse_session();
            } else if self.circle_draw.is_some() {
                self.cancel_circle_session();
            } else if self.rect_draw.is_some() {
                self.cancel_rect_session();
            } else if self.arc_draw.is_some() {
                self.cancel_arc_session();
            } else {
                self.dim_editing = false;
                self.dim_kind = None;
                self.dim_placing = false;
                self.dim_edit_did = None;
                self.dim_input.clear();
            }
            return;
        }
        if enter_pressed && let Some(state) = self.ellipse_draw.as_ref() {
            if state.axis_fixed && state.typed_ry.is_some() && state.ry > 1e-6 {
                // Minor typed: Enter completes without the third click.
                self.complete_ellipse();
            } else if !state.axis_fixed && state.typed_rx.is_some() && state.typed_angle.is_some() {
                // Length and angle typed: Enter fixes the axis without
                // the second click.
                self.fix_ellipse_axis();
            }
            // Otherwise Enter keeps the typed value; the next click
            // is still needed for direction / minor side.
            return;
        }
        if enter_pressed && self.circle_draw.is_some() {
            // Radius typed: Enter completes without the edge click.
            if self.circle_draw.as_ref().unwrap().typed_r.is_some() {
                self.complete_circle();
            }
            return;
        }
        if enter_pressed && let Some(state) = self.rect_draw.as_ref() {
            // Both sides typed: Enter completes without the corner
            // click (quadrant from the mouse's side).
            if state.typed_w.is_some() && state.typed_h.is_some() {
                self.complete_rect();
            }
            return;
        }
        if enter_pressed && self.arc_draw.is_some() {
            // Radius typed: Enter completes without the third click
            // (side and minor/major from the mouse).
            if self.arc_draw.as_ref().unwrap().typed_r.is_some() {
                self.complete_arc();
            }
            return;
        }
        if enter_pressed && self.scale_pending.is_some() {
            // Scale commit: reapply already baked the typed factor in;
            // Enter finalises and reports.
            self.commit_pending_scale();
            return;
        }
        if enter_pressed && self.fillet_pending.is_some() {
            // Fillet commit: reapply already baked the current input
            // into the sketch, so Enter just finalises the session.
            // No UpdateDimension is queued -- the single fillet undo
            // group built by reapply stays as the canonical record.
            self.reapply_fillets();
            self.fillet_pending = None;
            self.dim_editing = false;
            self.dim_edit_did = None;
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
                } else if let Some(edit_did) = self.dim_edit_did.take() {
                    // Editing existing dim -> re-bind as range.
                    let measured = self.sketch.dimension_index_by_did(edit_did)
                        .and_then(|i| self.sketch.dimensions.get(i))
                        .map(|d| d.value).unwrap_or(0.0);
                    self.begin_group();
                    self.exec(Action::UpdateDimension {
                        did: edit_did, value: measured, expr: None, range: Some(rb),
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
                        && let Some(did) = self.sketch.dimensions.last().map(|d| d.did) {
                            self.exec(Action::MoveDimension {
                                did, offset: self.dim_offset, text_along: self.dim_text_along,
                            });
                        }
                    success = true;
                }
            } else if is_numeric || is_expr || (input.is_empty() && self.dim_derived) {
                self.begin_group();
                if let Some((edit_did, edit_idx)) = self.dim_edit_did.take()
                    .and_then(|d| self.sketch.dimension_index_by_did(d).map(|i| (d, i))) {
                    // Editing existing: update in place (preserves name)
                    if self.dim_derived != self.sketch.dimensions[edit_idx].derived {
                        // Toggle derived status in place.
                        let value = if self.dim_derived {
                            None
                        } else if is_numeric {
                            Some(input.parse::<f64>().unwrap())
                        } else {
                            Some(self.sketch.dimensions[edit_idx].value)
                        };
                        self.exec(Action::ConvertDimension {
                            did: edit_did, derived: self.dim_derived, value,
                        });
                        success = true;
                    } else if is_numeric {
                        let value = input.parse::<f64>().unwrap();
                        self.exec(Action::UpdateDimension { did: edit_did, value, expr: None, range: None });
                        success = true;
                    } else if let Err(e) = self.sketch.validate_expr(&input) {
                        self.status_error = Some(format!("Expression error: {}", e));
                        self.dim_edit_did = Some(edit_did); // restore
                    } else {
                        self.exec(Action::UpdateDimension {
                            did: edit_did, value: 0.0,
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
                            && let Some(did) = self.sketch.dimensions.last().map(|d| d.did) {
                                self.exec(Action::MoveDimension {
                                    did, offset: self.dim_offset, text_along: self.dim_text_along,
                                });
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
                                && let Some(did) = self.sketch.dimensions.last().map(|d| d.did) {
                                    self.exec(Action::MoveDimension {
                                        did, offset: self.dim_offset, text_along: self.dim_text_along,
                                    });
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
                self.dim_edit_did = None;
                self.dim_kind = None;
                self.selection.clear();
                // Fillet-in-flight: the dim commit finalises the
                // fillet, so drop the pre-fillet snapshot. A later
                // Escape would otherwise roll the whole fillet back.
                self.fillet_pending = None;
            }
        } else if !response.has_focus() && self.dim_editing
            && !second_response.as_ref().is_some_and(|r| r.has_focus())
        {
            // Keep the value field focused -- unless the user tabbed
            // into the ellipse angle field.
            response.request_focus();
        }
    }
}
