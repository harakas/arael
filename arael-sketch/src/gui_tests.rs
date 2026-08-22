// GUI gesture tests: real input events through the real update loop,
// assertions on sketch/app state. No pixels. See test_harness.rs.

use eframe::egui;
use crate::test_harness::{Gui, v, near};
use crate::tools::{ConstraintSymbol, ConstraintType, Tool};
use arael_sketch_solver::{CapKind, CenterRef, DimensionKind, OffsetKind};
use arael_sketch_backend::Selection;

// -- Tool selection ---------------------------------------------------

#[test]
fn test_tool_keys() {
    let mut gui = Gui::new();
    assert_eq!(gui.app.tool, Tool::Select);
    gui.key(egui::Key::L);
    assert_eq!(gui.app.tool, Tool::DrawLine);
    gui.key(egui::Key::O);
    assert_eq!(gui.app.tool, Tool::DrawCircle);
    gui.key(egui::Key::A);
    assert_eq!(gui.app.tool, Tool::DrawArc);
    gui.key(egui::Key::R);
    assert_eq!(gui.app.tool, Tool::DrawRect);
    gui.key(egui::Key::P);
    assert_eq!(gui.app.tool, Tool::DrawPoint);
    gui.key(egui::Key::D);
    assert_eq!(gui.app.tool, Tool::Dimension);
    gui.key(egui::Key::Escape);
    assert_eq!(gui.app.tool, Tool::Select);
}

// -- Creation gestures ------------------------------------------------

#[test]
fn test_line_tool_click_click() {
    let mut gui = Gui::new();
    gui.key(egui::Key::L);
    gui.click(v(0.0, 0.0));
    assert_eq!(gui.line_count(), 0, "first click only starts the line");
    assert!(gui.app.line_draw.is_some());
    gui.click(v(2.0, 1.0));
    assert_eq!(gui.line_count(), 1);
    let (p1, p2) = gui.line(0);
    assert!(near(p1, v(0.0, 0.0), 0.05), "p1 = {:?}", p1);
    assert!(near(p2, v(2.0, 1.0), 0.05), "p2 = {:?}", p2);
}

#[test]
fn test_line_tool_chains() {
    let mut gui = Gui::new();
    gui.key(egui::Key::L);
    gui.click(v(0.0, 0.0));
    gui.click(v(2.0, 0.5));
    // Chained second segment continues from the first line's end.
    gui.click(v(3.0, -1.0));
    assert_eq!(gui.line_count(), 2);
    let (p1, _) = gui.line(1);
    assert!(near(p1, v(2.0, 0.5), 0.05));
    // The chain junction carries a line-line coincident constraint.
    let total_ll = gui.sketch().coincident_ll11.len()
        + gui.sketch().coincident_ll12.len()
        + gui.sketch().coincident_ll21.len()
        + gui.sketch().coincident_ll22.len();
    assert_eq!(total_ll, 1);
    // Escape ends the chain without touching geometry.
    gui.key(egui::Key::Escape);
    assert!(gui.app.line_draw.is_none());
    assert_eq!(gui.line_count(), 2);
}

#[test]
fn test_zero_length_line_rejected() {
    let mut gui = Gui::new();
    gui.key(egui::Key::L);
    gui.click(v(1.0, 1.0));
    gui.click(v(1.0, 1.0));
    // Rejected by the validation gate; the gesture ends, no panic,
    // later frames still run (W21 regression).
    assert_eq!(gui.line_count(), 0);
    gui.frames(2);
}

#[test]
fn test_circle_tool() {
    let mut gui = Gui::new();
    gui.key(egui::Key::O);
    gui.click(v(0.0, 0.0));
    gui.click(v(1.5, 0.0));
    assert_eq!(gui.arc_count(), 1);
    let r = gui.sketch().arcs.refs().next().unwrap();
    let a = &gui.sketch().arcs[r];
    assert!(a.closed);
    assert!((a.radius.value - 1.5).abs() < 0.05, "radius = {}", a.radius.value);
    assert!(near(a.center.value, v(0.0, 0.0), 0.05));
}

#[test]
fn test_rect_tool() {
    let mut gui = Gui::new();
    gui.key(egui::Key::R);
    gui.click(v(0.0, 0.0));
    gui.click(v(2.0, 1.0));
    assert_eq!(gui.line_count(), 4);
    assert_eq!(gui.sketch().coincident_ll21.len(), 4, "four corner coincidents");
    let hcount = gui.sketch().lines.iter().filter(|l| l.constraints.horizontal).count();
    let vcount = gui.sketch().lines.iter().filter(|l| l.constraints.vertical).count();
    assert_eq!((hcount, vcount), (2, 2));
    // The whole rect is one undo group.
    gui.cmd("undo");
    assert_eq!(gui.line_count(), 0);
    assert_eq!(gui.sketch().coincident_ll21.len(), 0);
}

#[test]
fn test_arc_tool_three_clicks() {
    let mut gui = Gui::new();
    gui.key(egui::Key::A);
    gui.click(v(-1.0, 0.0));
    gui.click(v(1.0, 0.0));
    assert_eq!(gui.arc_count(), 0, "two clicks only stage start/end");
    gui.click(v(0.0, 1.0));
    assert_eq!(gui.arc_count(), 1);
    let r = gui.sketch().arcs.refs().next().unwrap();
    let a = &gui.sketch().arcs[r];
    assert!(!a.closed);
    // Circle through (-1,0),(1,0),(0,1) is the unit circle.
    assert!(near(a.center.value, v(0.0, 0.0), 0.05), "center = {:?}", a.center.value);
    assert!((a.radius.value - 1.0).abs() < 0.05);
}

#[test]
fn test_line_snap_emits_coincident() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.key(egui::Key::L);
    // Start exactly on L0.p2: snap fires, completion emits the
    // coincident.
    gui.click(v(2.0, 0.0));
    gui.click(v(3.0, 1.0));
    assert_eq!(gui.line_count(), 2);
    let total_ll = gui.sketch().coincident_ll11.len()
        + gui.sketch().coincident_ll12.len()
        + gui.sketch().coincident_ll21.len()
        + gui.sketch().coincident_ll22.len();
    assert_eq!(total_ll, 1, "snap should emit exactly one line-line coincident");
}

// -- Selection --------------------------------------------------------

#[test]
fn test_click_select_escape_clear() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.click(v(1.0, 0.0));
    assert_eq!(gui.app.selection.len(), 1);
    assert!(matches!(gui.app.selection[0], Selection::Line(_)));
    gui.key(egui::Key::Escape);
    assert!(gui.app.selection.is_empty());
}

#[test]
fn test_box_select() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 1,0; add_line 0,1 1,1");
    // Drag over empty space around both lines.
    gui.drag(v(-0.5, -0.5), v(1.5, 1.5));
    let lines = gui.app.selection.iter()
        .filter(|s| matches!(s, Selection::Line(_)))
        .count();
    assert_eq!(lines, 2, "selection = {:?}", gui.app.selection);
}

#[test]
fn test_double_click_chain_select() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0 4,1");
    gui.double_click(v(1.0, 0.0));
    let lines = gui.app.selection.iter()
        .filter(|s| matches!(s, Selection::Line(_)))
        .count();
    assert_eq!(lines, 2, "chain select should take both segments: {:?}", gui.app.selection);
}

// -- Drag -------------------------------------------------------------

#[test]
fn test_drag_endpoint_moves_and_commits() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.drag(v(2.0, 0.0), v(3.0, 1.0));
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(3.0, 1.0), 0.15), "p2 after drag = {:?}", p2);
    assert!(gui.app.grab.is_none());
    // One committed history entry: undo restores the start position.
    gui.cmd("undo");
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(2.0, 0.0), 0.05), "p2 after undo = {:?}", p2);
}

#[test]
fn test_escape_cancels_drag_cleanly() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    let actions0 = gui.app.history.actions.len();
    gui.drag_moves(v(2.0, 0.0), v(2.5, 0.8));
    assert!(gui.app.grab.is_some(), "drag should be active");
    gui.key(egui::Key::Escape);
    // Cancelled: geometry restored, no apparatus left (the frame
    // invariant also checks drag_apparatus), no helper points.
    assert!(gui.app.grab.is_none());
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(2.0, 0.0), 0.05), "p2 after cancel = {:?}", p2);
    assert_eq!(gui.sketch().points.len(), 0, "no leftover helper points");
    // The button is still held: further motion must NOT re-grab the
    // endpoint at the press origin and resume the drag.
    gui.move_to(v(1.0, 1.5));
    gui.move_to(v(0.5, -1.0));
    assert!(gui.app.grab.is_none(), "held pointer re-grabbed after Escape");
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(2.0, 0.0), 0.05), "p2 moved after cancel: {:?}", p2);
    // Release commits nothing.
    gui.release(v(0.5, -1.0));
    assert_eq!(gui.app.history.actions.len(), actions0, "cancelled drag must not commit");
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(2.0, 0.0), 0.05));
    // A fresh drag after release works again.
    gui.drag(v(2.0, 0.0), v(3.0, 1.0));
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(3.0, 1.0), 0.15), "fresh drag should work: {:?}", p2);
}

// -- Constraints via keyboard ------------------------------------------

#[test]
fn test_horizontal_key_on_selection() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0.3");
    gui.click(v(1.0, 0.15));
    assert_eq!(gui.app.selection.len(), 1);
    gui.key(egui::Key::H);
    let l = gui.sketch().lines.iter().next().unwrap();
    assert!(l.constraints.horizontal);
    let (p1, p2) = gui.line(0);
    assert!((p1.y - p2.y).abs() < 1e-6, "line should be horizontal after solve");
}

#[test]
fn test_duplicate_horizontal_rejected() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0; horizontal L0");
    gui.click(v(1.0, 0.0));
    gui.key(egui::Key::H);
    // Rejected as duplicate; state undamaged, later frames fine.
    let l = gui.sketch().lines.iter().next().unwrap();
    assert!(l.constraints.horizontal);
    gui.frames(2);
}

// -- Dimensions --------------------------------------------------------

#[test]
fn test_dimension_placement_flow() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.key(egui::Key::D);
    // Phase 1: pick the line.
    gui.click(v(1.0, 0.0));
    assert!(gui.app.dim_placing, "clicking a line should enter placement");
    assert!(matches!(gui.app.dim_kind, Some(DimensionKind::LineLength(_))));
    // Phase 2: position the label below the line, confirm.
    gui.move_to(v(1.0, -1.0));
    assert!(gui.app.dim_offset.y < 0.0, "offset should follow the mouse side");
    gui.click(v(1.0, -1.0));
    assert!(gui.app.dim_editing, "confirm click should open the value input");
    // The overlay focuses itself; type a value and commit.
    gui.frame();
    gui.type_text("5");
    gui.key(egui::Key::Enter);
    assert_eq!(gui.sketch().dimensions.len(), 1);
    let d = &gui.sketch().dimensions[0];
    assert!((d.value - 5.0).abs() < 1e-9, "dimension value = {}", d.value);
    let (p1, p2) = gui.line(0);
    let len = ((p2.x - p1.x).powi(2) + (p2.y - p1.y).powi(2)).sqrt();
    assert!((len - 5.0).abs() < 0.01, "line length after solve = {}", len);
}

#[test]
fn test_dimension_escape_cancels() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.key(egui::Key::D);
    gui.click(v(1.0, 0.0));
    gui.click(v(1.0, -1.0));
    assert!(gui.app.dim_editing);
    gui.frame();
    gui.key(egui::Key::Escape);
    assert!(!gui.app.dim_editing);
    assert_eq!(gui.sketch().dimensions.len(), 0);
}

#[test]
fn test_ellipse_radius_axis_flip() {
    let mut gui = Gui::new();
    gui.cmd("add_ellipse 0,0 2 1 0");
    gui.key(egui::Key::D);
    // Pick near the minor axis: kind starts as ArcRadiusB.
    gui.click(v(0.0, 1.0));
    assert!(matches!(gui.app.dim_kind, Some(DimensionKind::ArcRadiusB(_))),
        "kind = {:?}", gui.app.dim_kind);
    // Placement preview flips to the major axis as the mouse moves.
    gui.move_to(v(2.2, 0.1));
    assert!(matches!(gui.app.dim_kind, Some(DimensionKind::ArcRadius(_))),
        "kind after move = {:?}", gui.app.dim_kind);
    gui.move_to(v(0.1, 1.2));
    assert!(matches!(gui.app.dim_kind, Some(DimensionKind::ArcRadiusB(_))),
        "kind after move back = {:?}", gui.app.dim_kind);
}

#[test]
fn test_dimension_label_drag_commits_move() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0; length L0 2");
    let offset0 = gui.sketch().dimensions[0].offset;
    let actions0 = gui.app.history.actions.len();
    // The label sits at the dimension line, offset.y above the line
    // midpoint; grab it there and pull it further up.
    gui.drag(v(1.0, offset0.y), v(1.0, offset0.y + 1.0));
    let offset1 = gui.sketch().dimensions[0].offset;
    assert!((offset1.y - offset0.y).abs() > 0.5,
        "offset should move: {} -> {}", offset0.y, offset1.y);
    assert_eq!(gui.app.history.actions.len(), actions0 + 1,
        "label drag should commit exactly one history entry");
    assert_eq!(gui.line_count(), 1, "geometry untouched");
}

#[test]
fn test_dof_display_recovers_after_drag() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    let dof_before = gui.sketch().cached_dof();
    assert!(dof_before.is_some(), "DOF known before drag");

    // Mid-gesture: the cell's cache is retired by the apparatus, but
    // the display falls back to the pre-drag rank's nullity -- the
    // corner shows the pre-drag DOF, not "...".
    gui.drag_moves(v(2.0, 0.0), v(3.0, 1.0));
    let displayed = gui.sketch().cached_dof()
        .or_else(|| gui.app.drag_rank.as_ref().map(|r| r.nullity));
    assert_eq!(displayed, dof_before, "mid-drag display should show pre-drag DOF");
    gui.release(v(3.0, 1.0));

    // After the gesture the cache must be a number again.
    gui.frames(3);
    assert_eq!(gui.sketch().cached_dof(), dof_before,
        "DOF display must recover after drag");
}

// -- Undo/redo keyboard -------------------------------------------------

#[test]
fn test_undo_redo_keys() {
    let mut gui = Gui::new();
    gui.key(egui::Key::L);
    gui.click(v(0.0, 0.0));
    gui.click(v(2.0, 0.0));
    gui.key(egui::Key::Escape);
    assert_eq!(gui.line_count(), 1);
    gui.key_with(egui::Key::Z, egui::Modifiers::CTRL);
    assert_eq!(gui.line_count(), 0, "Ctrl+Z should remove the line");
    gui.key_with(egui::Key::Z, egui::Modifiers::CTRL | egui::Modifiers::SHIFT);
    assert_eq!(gui.line_count(), 1, "Ctrl+Shift+Z should restore it");
}

// -- Split / Trim tools -----------------------------------------------

/// A 10-long horizontal line crossed by a vertical cutter at x=4,
/// drawn via commands so the geometry is exact.
fn split_gui() -> Gui {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 10,0");
    gui.cmd("add_line 4,-2 4,2 noconnect");
    gui
}

#[test]
fn test_split_trim_key_cycles() {
    let mut gui = Gui::new();
    gui.key(egui::Key::B);
    assert_eq!(gui.app.tool, Tool::Split);
    gui.key(egui::Key::B);
    assert_eq!(gui.app.tool, Tool::Trim);
    gui.key(egui::Key::B);
    assert_eq!(gui.app.tool, Tool::Split);
    gui.key(egui::Key::Escape);
    assert_eq!(gui.app.tool, Tool::Select);
}

#[test]
fn test_split_tool_click() {
    let mut gui = split_gui();
    assert_eq!(gui.line_count(), 2);
    gui.key(egui::Key::B);
    gui.click(v(2.0, 0.0));
    assert_eq!(gui.line_count(), 3, "target replaced by two pieces");
    // Pieces joined at the cut and pinned to the cutter.
    assert_eq!(gui.sketch().coincident_ll21.len(), 1);
    assert_eq!(gui.sketch().line_p2_on_line.len(), 1);
    // Report landed in the command panel.
    assert!(gui.app.command_output.iter().any(|(t, _, _)| t.contains("Split L0")),
        "command_output: {:?}", gui.app.command_output.last());
}

#[test]
fn test_split_preview_picks_clicked_span() {
    let mut gui = split_gui();
    gui.key(egui::Key::B);
    // Hover left of the cutter: the preview span is 0..4.
    let (_, span) = gui.app.split_trim_preview(v(2.0, 0.0), 0.2).unwrap();
    let (t0, t1) = span.unwrap();
    assert!(near_f(t0, 0.0) && near_f(t1, 0.4), "span {:?}", (t0, t1));
    // Hover right of the cutter: 4..10.
    let (_, span) = gui.app.split_trim_preview(v(8.0, 0.0), 0.2).unwrap();
    let (t0, t1) = span.unwrap();
    assert!(near_f(t0, 0.4) && near_f(t1, 1.0), "span {:?}", (t0, t1));
}

fn near_f(a: f64, b: f64) -> bool { (a - b).abs() < 1e-6 }

#[test]
fn test_trim_tool_click_removes_span() {
    let mut gui = split_gui();
    gui.key(egui::Key::B);
    gui.key(egui::Key::B); // cycle to Trim
    assert_eq!(gui.app.tool, Tool::Trim);
    gui.click(v(2.0, 0.0));
    // The clicked left span is gone; one piece + cutter remain.
    assert_eq!(gui.line_count(), 2);
    let survivor = gui.sketch().lines.iter()
        .find(|l| l.name != "L1").unwrap();
    assert!(near(survivor.p1.value, v(4.0, 0.0), 0.05),
        "surviving piece starts at the cut: {:?}", survivor.p1.value);
}

#[test]
fn test_trim_no_cuts_deletes_entity() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 10,0");
    gui.key(egui::Key::B);
    gui.key(egui::Key::B);
    gui.click(v(5.0, 0.0));
    assert_eq!(gui.line_count(), 0);
}

#[test]
fn test_split_undo_one_group() {
    let mut gui = split_gui();
    gui.key(egui::Key::B);
    gui.click(v(2.0, 0.0));
    assert_eq!(gui.line_count(), 3);
    // One Ctrl+Z restores the target, its pieces, and the follow-up
    // constraints as a single group.
    gui.key_with(egui::Key::Z, egui::Modifiers::CTRL);
    assert_eq!(gui.line_count(), 2);
    assert!(gui.sketch().lines.iter().any(|l| l.name == "L0"));
    assert!(gui.sketch().coincident_ll21.is_empty());
}

#[test]
fn test_fillet_chamfer_key_cycles() {
    let mut gui = Gui::new();
    gui.key(egui::Key::F);
    assert_eq!(gui.app.tool, Tool::Fillet);
    gui.key(egui::Key::F);
    assert_eq!(gui.app.tool, Tool::Chamfer);
    gui.key(egui::Key::F);
    assert_eq!(gui.app.tool, Tool::Fillet);
    gui.key(egui::Key::Escape);
    assert_eq!(gui.app.tool, Tool::Select);
}

#[test]
fn test_tool_switch_mid_drag_cancels() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    let actions0 = gui.app.history.actions.len();
    gui.drag_moves(v(2.0, 0.0), v(2.5, 0.8));
    assert!(gui.app.grab.is_some(), "drag should be active");
    // A tool shortcut mid-gesture switches the tool; the frame-top
    // check must cancel the drag: apparatus removed, geometry
    // restored, nothing committed.
    gui.key(egui::Key::L);
    gui.frames(1);
    assert_eq!(gui.app.tool, Tool::DrawLine);
    assert!(gui.app.grab.is_none(), "tool switch must cancel the grab");
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(2.0, 0.0), 0.05), "p2 after tool-switch cancel = {:?}", p2);
    assert_eq!(gui.sketch().points.len(), 0, "no leftover helper points");
    gui.release(v(2.5, 0.8));
    assert_eq!(gui.app.history.actions.len(), actions0,
        "cancelled drag must not commit");
}

#[test]
fn test_chain_select_through_arc() {
    let mut gui = Gui::new();
    // line -> arc -> line, joined by the auto-connect coincidences.
    gui.cmd("add_line 0,0 2,0");
    gui.cmd("add_arc 2,0 4,0 3,1");
    gui.cmd("add_line 4,0 6,0");
    assert_eq!(gui.line_count(), 2);
    assert_eq!(gui.arc_count(), 1);
    gui.double_click(v(1.0, 0.0));
    let lines = gui.app.selection.iter()
        .filter(|s| matches!(s, Selection::Line(_))).count();
    let arcs = gui.app.selection.iter()
        .filter(|s| matches!(s, Selection::Arc(_))).count();
    assert_eq!((lines, arcs), (2, 1),
        "chain crosses the arc: {:?}", gui.app.selection);
}

#[test]
fn test_chain_select_stops_at_construction() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.cmd("add_line 2,0 4,1 constr");
    gui.double_click(v(1.0, 0.0));
    let lines = gui.app.selection.iter()
        .filter(|s| matches!(s, Selection::Line(_))).count();
    assert_eq!(lines, 1,
        "construction geometry stays out of the chain: {:?}", gui.app.selection);
}

// -- Fillet / Chamfer gestures ----------------------------------------

#[test]
fn test_fillet_tool_gesture() {
    let mut gui = Gui::new();
    // Two lines meeting at a right-angle corner at (2,0).
    gui.cmd("add_line 0,0 2,0");
    gui.cmd("add_line 2,0 2,2");
    gui.key(egui::Key::F);
    assert_eq!(gui.app.tool, Tool::Fillet);
    // Click the shared corner endpoint: the session starts with a
    // 10 % mock radius already applied.
    gui.click(v(2.0, 0.0));
    assert!(gui.app.fillet_pending.is_some(), "corner click starts the session");
    assert_eq!(gui.arc_count(), 1, "mock fillet applied immediately");
    assert!(gui.app.dim_editing, "radius edit opens");
    // Type a radius and commit.
    gui.frame();
    gui.type_text("0.5");
    gui.key(egui::Key::Enter);
    assert!(gui.app.fillet_pending.is_none(), "Enter finalises the session");
    assert_eq!(gui.arc_count(), 1);
    let arc = gui.sketch().arcs.iter().next().unwrap();
    assert!((arc.radius.value - 0.5).abs() < 1e-6, "radius = {}", arc.radius.value);
    // Radius dim + tangents landed.
    assert_eq!(gui.sketch().dimensions.len(), 1);
    assert_eq!(gui.sketch().tangent_la.len(), 2);
    // One Ctrl+Z undoes the whole fillet.
    gui.key_with(egui::Key::Z, egui::Modifiers::CTRL);
    assert_eq!(gui.arc_count(), 0);
    assert_eq!(gui.sketch().dimensions.len(), 0);
}

#[test]
fn test_fillet_escape_restores() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.cmd("add_line 2,0 2,2");
    let actions0 = gui.app.history.actions.len();
    gui.key(egui::Key::F);
    gui.click(v(2.0, 0.0));
    assert_eq!(gui.arc_count(), 1);
    // Escape rolls the whole session back.
    gui.frame();
    gui.key(egui::Key::Escape);
    assert!(gui.app.fillet_pending.is_none());
    assert_eq!(gui.arc_count(), 0);
    assert_eq!(gui.app.history.actions.len(), actions0);
    // The trimmed endpoints are restored.
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(2.0, 0.0), 0.01), "p2 restored: {:?}", p2);
}

#[test]
fn test_chamfer_tool_gesture() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.cmd("add_line 2,0 2,2");
    gui.key(egui::Key::F);
    gui.key(egui::Key::F); // cycle to Chamfer
    assert_eq!(gui.app.tool, Tool::Chamfer);
    gui.click(v(2.0, 0.0));
    assert!(gui.app.fillet_pending.is_some());
    assert_eq!(gui.line_count(), 3, "bevel line added");
    gui.frame();
    gui.type_text("0.4");
    gui.key(egui::Key::Enter);
    assert!(gui.app.fillet_pending.is_none());
    // Two leg dims, the corner anchor point, both point-on constraints.
    assert_eq!(gui.sketch().dimensions.len(), 2);
    assert_eq!(gui.sketch().point_on_line.len(), 2);
    let bevel = gui.line(2);
    let len = ((bevel.1.x - bevel.0.x).powi(2) + (bevel.1.y - bevel.0.y).powi(2)).sqrt();
    let expect = 0.4_f64 * std::f64::consts::SQRT_2;
    assert!((len - expect).abs() < 0.01, "bevel length = {} vs {}", len, expect);
}

#[test]
fn test_dim_placement_hv_switch() {
    let mut gui = Gui::new();
    gui.cmd("add_point 0,0");
    gui.cmd("add_point 2,1");
    gui.key(egui::Key::D);
    gui.click(v(0.0, 0.0));
    gui.click(v(2.0, 1.0));
    assert!(gui.app.dim_placing, "two points enter placement");
    // Above the pair's box: horizontal distance.
    gui.move_to(v(1.0, 3.0));
    assert!(matches!(gui.app.dim_kind, Some(DimensionKind::HDistance(..))),
        "above -> HDistance, got {:?}", gui.app.dim_kind);
    // Right of the box: vertical distance.
    gui.move_to(v(4.0, 0.5));
    assert!(matches!(gui.app.dim_kind, Some(DimensionKind::VDistance(..))),
        "right -> VDistance, got {:?}", gui.app.dim_kind);
    // Near the diagonal: plain point-point distance.
    gui.move_to(v(1.0, 0.5));
    assert!(matches!(gui.app.dim_kind, Some(DimensionKind::PointPointDistance(..))),
        "inside -> PointPointDistance, got {:?}", gui.app.dim_kind);
}

#[test]
fn test_earc_sweep_label_selectable() {
    let mut gui = Gui::new();
    // The n5_check shape: rotated elliptic arc with a derived sweep dim.
    gui.cmd("add_earc_center 2,4 3 1.2 25 0 210");
    gui.cmd("sweep EA0 derived");
    gui.cmd("center 2,4");
    assert_eq!(gui.sketch().dimensions.len(), 1);
    let dim = gui.sketch().dimensions[0].clone();
    let did = dim.did;
    // The hit segment must sit where draw_sweep_dimension puts the
    // text: on the rotated annotation ellipse, not on an unrotated
    // circle of radius r + offset. Compare in screen pixels (the
    // segment is offset from the anchor by text height).
    let a = gui.sketch().arcs.iter().next().unwrap();
    let (sa, ea) = (a.start_angle.value, a.end_angle.value);
    let text_angle = sa + (ea - sa) * (0.5 + dim.text_along);
    let expected = arael_sketch_solver::ellipse_point(
        a.center.value,
        (a.radius.value + dim.offset.y).max(0.1),
        (a.radius_b.value + dim.offset.y).max(0.1),
        a.rotation.value,
        text_angle,
    );
    let expected_screen = gui.app.to_screen(expected);
    let (ts, te) = gui.app.dim_text_segment(&dim);
    let mid = egui::Pos2::new((ts.x + te.x) / 2.0, (ts.y + te.y) / 2.0);
    let d = ((mid.x - expected_screen.x).powi(2) + (mid.y - expected_screen.y).powi(2)).sqrt();
    assert!(d < 20.0,
        "label hit segment {} px from the drawn text (at {:?}, drawn {:?})",
        d, (mid.x, mid.y), (expected_screen.x, expected_screen.y));
    // Clicking the drawn label selects the dimension.
    gui.click(expected);
    assert!(gui.app.selection.contains(&Selection::Dimension(did)),
        "click on the sweep label must select it: {:?}", gui.app.selection);
    // Dragging the label outward from the center grows offset.y (the
    // placement math follows the ellipse frame, not a raw circle).
    let c = gui.sketch().arcs.iter().next().unwrap().center.value;
    let dir = v(expected.x - c.x, expected.y - c.y);
    let dlen = (dir.x * dir.x + dir.y * dir.y).sqrt();
    let target = v(expected.x + dir.x / dlen, expected.y + dir.y / dlen);
    gui.drag(expected, target);
    let off = gui.sketch().dimensions[0].offset.y;
    assert!(off > 1.2, "outward label drag must grow offset.y, got {}", off);
}

// -- All dimension kinds: select / drag / edit ------------------------

/// One scene per DimensionKind: setup commands (the last creates the
/// dimension), the value to type in the edit test, and the expected
/// stored value after Enter.
const DIM_SCENES: &[(&[&str], &str, f64)] = &[
    // LineLength
    (&["add_line 0,0 3,0", "length L0 3"], "3.5", 3.5),
    // PointPointDistance
    (&["add_point 0,0", "add_point 2,1", "distance P0 P1 2.5"], "2.8", 2.8),
    // PointLineDistance
    (&["add_line 0,0 3,0", "add_point 1,1", "distance P0 L0 1"], "1.2", 1.2),
    // ArcRadius
    (&["add_circle 0,0 1.5", "radius A0 1.5"], "1.8", 1.8),
    // ArcRadiusB
    (&["add_ellipse 0,0 2 1 0", "radius_b EA0 1"], "0.8", 0.8),
    // ArcSweep, circular
    (&["add_arc 1,0 -1,0 0,1", "sweep A0 180"], "170", 170.0),
    // ArcSweep on a rotated elliptic arc (the label used to hit-test
    // on an unrotated circle and was unreachable)
    (&["add_earc_center 2,4 3 1.2 25 0 210", "sweep EA0 210", "center 2,4"], "190", 190.0),
    // ArcRotation
    (&["add_ellipse 0,0 2 1 30", "xangle EA0 30"], "40", 40.0),
    // Angle
    (&["add_line 0,0 2,0", "add_line 0,0 0,2 noconnect", "angle L0 L1 90"], "80", 80.0),
    // HDistance
    (&["add_point 0,0", "add_point 2,1", "hdistance P0 P1 2"], "2.4", 2.4),
    // VDistance
    (&["add_point 0,0", "add_point 2,1", "vdistance P0 P1 1"], "1.3", 1.3),
    // LineAngle
    (&["add_line 0,0 2,1", "xangle L0 26.5651"], "35", 35.0),
    // ConcentricDistance
    (&["add_circle 0,0 1", "add_circle 0,0 2 noconnect", "distance A0 A1 1"], "1.2", 1.2),
    // LineLineDistance
    (&["add_line 0,0 3,0", "add_line 0,1 3,1 noconnect", "distance L0 L1 1"], "1.4", 1.4),
];

fn dim_scene(cmds: &[&str]) -> (Gui, u32) {
    let mut gui = Gui::new();
    for c in cmds {
        gui.cmd(c);
    }
    assert_eq!(gui.sketch().dimensions.len(), 1, "scene {:?}", cmds);
    let did = gui.sketch().dimensions[0].did;
    (gui, did)
}

/// Sketch-space position of the dimension's drawn label.
fn dim_label_pos(gui: &Gui, did: u32) -> arael::vect::vect2d {
    let i = gui.app.sketch.dimension_index_by_did(did).unwrap();
    let dim = gui.app.sketch.dimensions[i].clone();
    let (ts, te) = gui.app.dim_text_segment(&dim);
    gui.app.to_sketch(egui::Pos2::new((ts.x + te.x) / 2.0, (ts.y + te.y) / 2.0))
}

#[test]
fn test_all_dims_selectable() {
    for (cmds, _, _) in DIM_SCENES {
        let (mut gui, did) = dim_scene(cmds);
        let pos = dim_label_pos(&gui, did);
        gui.click(pos);
        assert!(gui.app.selection.contains(&Selection::Dimension(did)),
            "clicking the label must select the dim: scene {:?}, selection {:?}",
            cmds, gui.app.selection);
    }
}

#[test]
fn test_all_dims_label_drag() {
    for (cmds, _, _) in DIM_SCENES {
        let (mut gui, did) = dim_scene(cmds);
        let actions0 = gui.app.history.actions.len();
        let pos = dim_label_pos(&gui, did);
        gui.drag(pos, v(pos.x + 0.4, pos.y + 0.4));
        // The drag commits exactly one MoveDimension.
        assert_eq!(gui.app.history.actions.len(), actions0 + 1,
            "label drag must commit one action: scene {:?}", cmds);
        let desc = gui.app.history.actions.last().unwrap().describe();
        assert_eq!(desc, "Move dimension",
            "label drag committed '{}' instead: scene {:?}", desc, cmds);
    }
}

#[test]
fn test_all_dims_editable() {
    for (cmds, typed, expected) in DIM_SCENES {
        let (mut gui, did) = dim_scene(cmds);
        let pos = dim_label_pos(&gui, did);
        gui.double_click(pos);
        assert!(gui.app.dim_editing && gui.app.dim_edit_did == Some(did),
            "double-click must open the value edit: scene {:?}", cmds);
        gui.frame();
        gui.type_text(typed);
        gui.key(egui::Key::Enter);
        let i = gui.app.sketch.dimension_index_by_did(did).unwrap();
        let v = gui.app.sketch.dimensions[i].value;
        assert!((v - expected).abs() < 1e-6,
            "edited value should be {}, got {}: scene {:?}", expected, v, cmds);
    }
}

// -- Scale tool -------------------------------------------------------

#[test]
fn test_scale_tool_flow() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.cmd("add_line 0,0 0,1 noconnect");
    gui.cmd("add_point 0,0");
    gui.app.tool = Tool::Scale;
    gui.click(v(1.0, 0.0));
    gui.click(v(0.0, 0.5));
    assert_eq!(gui.app.selection.len(), 2);
    assert!(gui.app.scale_pending.is_none(), "no session before a center exists");
    // Double-click the point: center set, session + value input open.
    gui.double_click(v(0.0, 0.0));
    assert!(gui.app.scale_center.is_some());
    assert!(gui.app.scale_pending.is_some());
    assert!(gui.app.dim_editing);
    // Live preview: typing updates the canvas before Enter.
    gui.frame();
    gui.type_text("3");
    gui.frame();
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(6.0, 0.0), 0.05), "live preview at x3: {:?}", p2);
    // Commit.
    gui.key(egui::Key::Enter);
    assert!(gui.app.scale_pending.is_none());
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(6.0, 0.0), 0.05));
    let (_, q2) = gui.line(1);
    assert!(near(q2, v(0.0, 3.0), 0.05), "second line scaled too: {:?}", q2);
    assert!(gui.app.command_output.iter().any(|(t, _, _)| t.contains("Scaled")),
        "report in command panel");
    // One undo restores the pre-scale geometry.
    gui.cmd("undo");
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(2.0, 0.0), 0.05), "after undo: {:?}", p2);
}

#[test]
fn test_scale_escape_restores() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.cmd("add_point 0,0");
    let actions0 = gui.app.history.actions.len();
    gui.app.tool = Tool::Scale;
    gui.click(v(1.0, 0.0));
    gui.double_click(v(0.0, 0.0));
    gui.frame();
    gui.type_text("4");
    gui.frame();
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(8.0, 0.0), 0.05), "preview applied: {:?}", p2);
    gui.key(egui::Key::Escape);
    assert!(gui.app.scale_pending.is_none());
    assert!(gui.app.scale_center.is_none());
    let (_, p2) = gui.line(0);
    assert!(near(p2, v(2.0, 0.0), 0.05), "Escape restores: {:?}", p2);
    assert_eq!(gui.app.history.actions.len(), actions0, "nothing committed");
}

#[test]
fn test_scale_point_in_set() {
    let mut gui = Gui::new();
    gui.cmd("add_point 0,0");
    gui.cmd("add_point 2,1");
    gui.app.tool = Tool::Scale;
    // P1 into the set, P0 as center.
    gui.click(v(2.0, 1.0));
    gui.double_click(v(0.0, 0.0));
    assert!(gui.app.scale_pending.is_some());
    gui.frame();
    gui.type_text("2");
    gui.key(egui::Key::Enter);
    let p1 = gui.sketch().points.iter().find(|p| p.name == "P1").unwrap().pos.value;
    assert!(near(p1, v(4.0, 2.0), 0.05), "P1 scaled: {:?}", p1);
    let p0 = gui.sketch().points.iter().find(|p| p.name == "P0").unwrap().pos.value;
    assert!(near(p0, v(0.0, 0.0), 0.05), "center point untouched");
}

#[test]
fn test_scale_adopts_prior_selection() {
    let mut gui = Gui::new();
    gui.cmd("add_line 1,0 2,0");
    gui.cmd("add_line 1,1 2,1");
    gui.cmd("add_point 0,0");
    // Select a line and an endpoint with the Select tool...
    gui.click(v(1.5, 0.0));
    gui.click(v(1.0, 1.0));
    assert_eq!(gui.app.selection.len(), 2, "sel = {:?}", gui.app.selection);
    // ...then enter Scale the way the toolbar button does.
    gui.app.tool = Tool::Scale;
    gui.app.adopt_selection_for_scale();
    gui.app.scale_center = None;
    assert!(
        gui.app.selection.iter().all(|s| matches!(s, Selection::Line(_)))
            && gui.app.selection.len() == 2,
        "selection carried over as whole lines: {:?}", gui.app.selection
    );
    gui.double_click(v(0.0, 0.0));
    assert!(gui.app.scale_pending.is_some());
    gui.frame();
    gui.type_text("2");
    gui.key(egui::Key::Enter);
    let l1 = gui.sketch().lines.iter().find(|l| l.name == "L1").unwrap();
    assert!(near(l1.p1.value, v(2.0, 2.0), 0.05), "L1.p1 scaled: {:?}", l1.p1.value);
    assert!(near(l1.p2.value, v(4.0, 2.0), 0.05), "L1.p2 scaled: {:?}", l1.p2.value);
    let l0 = gui.sketch().lines.iter().find(|l| l.name == "L0").unwrap();
    assert!(near(l0.p2.value, v(4.0, 0.0), 0.05), "L0.p2 scaled: {:?}", l0.p2.value);
}

// -- Circle / rect typed inputs ---------------------------------------

#[test]
fn test_circle_typed_radius() {
    let mut gui = Gui::new();
    gui.key(egui::Key::O);
    gui.click(v(0.0, 0.0));
    assert!(gui.app.dim_editing, "radius input live from the center click");
    gui.frame();
    gui.type_text("2.5");
    gui.move_to(v(1.0, 0.0)); // mouse no longer changes the radius
    assert!((gui.app.circle_draw.as_ref().unwrap().r - 2.5).abs() < 1e-9);
    gui.key(egui::Key::Enter); // completes without the edge click
    assert_eq!(gui.arc_count(), 1);
    let a = gui.sketch().arcs.iter().next().unwrap().clone();
    assert!(a.closed && (a.radius.value - 2.5).abs() < 1e-9, "r {}", a.radius.value);
    let dims = &gui.sketch().dimensions;
    assert_eq!(dims.len(), 1);
    assert!(matches!(dims[0].kind, DimensionKind::ArcRadius(_)) && (dims[0].value - 2.5).abs() < 1e-9);
    assert!(gui.app.circle_draw.is_none() && !gui.app.dim_editing);
    // Click-only circle: no dim (test_circle_tool covers geometry).
    gui.click(v(5.0, 0.0));
    gui.click(v(6.0, 0.0));
    assert_eq!(gui.arc_count(), 2);
    assert_eq!(gui.sketch().dimensions.len(), 1, "click-only circle adds no dim");
}

#[test]
fn test_circle_typed_radius_ignores_snap_and_reports_invalid() {
    let mut gui = Gui::new();
    gui.cmd("add_point 2,0");
    gui.key(egui::Key::O);
    gui.click(v(0.0, 0.0));
    gui.move_to(v(1.98, 0.02));
    assert!(gui.app.circle_draw.as_ref().unwrap().live_snap.is_some(), "rim snap offered while live");
    gui.frame();
    gui.type_text("sqrt(");
    gui.move_to(v(1.98, 0.02));
    assert!(gui.app.circle_draw.as_ref().unwrap().live_snap.is_none(), "typed radius: no snap");
    gui.click(v(1.98, 0.02));
    assert_eq!(gui.arc_count(), 0, "broken text blocks completion");
    assert!(gui.app.status_error.as_deref().is_some_and(|e| e.contains("Radius")), "{:?}", gui.app.status_error);
    gui.type_text("2)");
    gui.click(v(1.98, 0.02));
    let a = gui.sketch().arcs.iter().next().expect("created after fix").clone();
    assert!((a.radius.value - 2f64.sqrt()).abs() < 1e-9);
    assert_eq!(gui.sketch().point_on_arc.len(), 0, "no rim tie with a typed radius");
    assert!(gui.sketch().dimensions[0].expr_str.as_deref() == Some("sqrt(2)"));
}

#[test]
fn test_rect_typed_sides() {
    let mut gui = Gui::new();
    gui.key(egui::Key::R);
    gui.click(v(1.0, 1.0));
    assert!(gui.app.dim_editing, "width/height inputs live from the first corner");
    gui.frame();
    gui.type_text("4");
    gui.key(egui::Key::Tab);
    gui.frame();
    gui.type_text("2");
    // Quadrant follows the mouse; sides stay typed.
    gui.move_to(v(-1.0, 2.0));
    let s = gui.app.rect_draw.as_ref().unwrap();
    assert!((s.w - 4.0).abs() < 1e-9 && (s.h - 2.0).abs() < 1e-9, "w {} h {}", s.w, s.h);
    assert!(s.sx < 0.0 && s.sy > 0.0, "quadrant from the mouse");
    gui.key(egui::Key::Enter); // both typed: completes without the click
    assert_eq!(gui.line_count(), 4);
    let (p1, p2) = gui.line(0);
    assert!(near(p1, v(1.0, 1.0), 1e-9) && near(p2, v(-3.0, 1.0), 1e-9), "bottom side {:?} {:?}", p1, p2);
    let (_, p2) = gui.line(1);
    assert!(near(p2, v(-3.0, 3.0), 1e-9), "opposite corner {:?}", p2);
    let dims = &gui.sketch().dimensions;
    assert_eq!(dims.len(), 2, "width and height dims");
    assert!(dims.iter().any(|d| matches!(d.kind, DimensionKind::LineLength(_)) && (d.value - 4.0).abs() < 1e-9));
    assert!(dims.iter().any(|d| matches!(d.kind, DimensionKind::LineLength(_)) && (d.value - 2.0).abs() < 1e-9));
    // One undo step for the whole rect incl. dims.
    gui.cmd("undo");
    assert_eq!(gui.line_count(), 0);
    assert!(gui.sketch().dimensions.is_empty());
}

#[test]
fn test_rect_typed_width_only() {
    let mut gui = Gui::new();
    gui.key(egui::Key::R);
    gui.click(v(0.0, 0.0));
    gui.frame();
    gui.type_text("3");
    gui.click(v(1.0, 1.5)); // height from the mouse, width fixed at 3
    assert_eq!(gui.line_count(), 4);
    let (_, p2) = gui.line(1);
    assert!(near(p2, v(3.0, 1.5), 1e-9), "opposite corner {:?}", p2);
    let dims = &gui.sketch().dimensions;
    assert_eq!(dims.len(), 1, "only the typed width became a dim");
    assert!((dims[0].value - 3.0).abs() < 1e-9);
}

// -- Arc tool: radius input and tangent snaps -------------------------

#[test]
fn test_arc_typed_radius() {
    let mut gui = Gui::new();
    gui.key(egui::Key::A);
    gui.click(v(-1.0, 0.0));
    assert!(!gui.app.dim_editing, "no radius input before the chord exists");
    gui.click(v(1.0, 0.0));
    assert!(gui.app.dim_editing, "radius input live once the chord exists");
    gui.frame();
    gui.type_text("2");
    // Mouse above the chord, nearer than r: minor arc bulging up.
    gui.move_to(v(0.0, 0.5));
    gui.key(egui::Key::Enter);
    assert_eq!(gui.arc_count(), 1);
    let a = gui.sketch().arcs.iter().next().unwrap().clone();
    assert!((a.radius.value - 2.0).abs() < 1e-9, "r {}", a.radius.value);
    // Minor arc: center below the chord (bulge up), sweep < 180.
    assert!(a.center.value.y < 0.0, "center {:?}", a.center.value);
    let sweep = (a.end_angle.value - a.start_angle.value).abs() % std::f64::consts::TAU;
    assert!(sweep < std::f64::consts::PI, "minor arc: sweep {}", sweep);
    let dims = &gui.sketch().dimensions;
    assert_eq!(dims.len(), 1);
    assert!(matches!(dims[0].kind, DimensionKind::ArcRadius(_)) && (dims[0].value - 2.0).abs() < 1e-9);
    // Same again, mouse far above (> r): the major arc.
    gui.click(v(-1.0, -2.0));
    gui.click(v(1.0, -2.0));
    gui.frame();
    gui.type_text("2");
    gui.move_to(v(0.0, 1.0)); // 3 above the chord > r
    gui.key(egui::Key::Enter);
    let a = gui.sketch().arcs.iter().nth(1).unwrap().clone();
    let sweep = (a.end_angle.value - a.start_angle.value).abs() % std::f64::consts::TAU;
    assert!(sweep > std::f64::consts::PI, "major arc: sweep {}", sweep);
    assert!(a.center.value.y > -2.0, "center on the bulge side for the major arc");
}

#[test]
fn test_arc_typed_radius_below_half_chord_reports() {
    let mut gui = Gui::new();
    gui.key(egui::Key::A);
    gui.click(v(-1.0, 0.0));
    gui.click(v(1.0, 0.0));
    gui.frame();
    gui.type_text("0.5");
    gui.click(v(0.0, 1.0));
    assert_eq!(gui.arc_count(), 0);
    assert!(gui.app.arc_draw.is_some(), "session stays open");
    assert!(gui.app.status_error.as_deref().is_some_and(|e| e.contains("half the chord")), "{:?}", gui.app.status_error);
    for _ in 0..3 { gui.key(egui::Key::Backspace); }
    gui.type_text("2");
    gui.click(v(0.0, 1.0));
    assert_eq!(gui.arc_count(), 1, "fixed value creates the arc");
}

#[test]
fn test_arc_click_only_no_dim() {
    let mut gui = Gui::new();
    gui.key(egui::Key::A);
    gui.click(v(-1.0, 0.0));
    gui.click(v(1.0, 0.0));
    gui.click(v(0.0, 1.0));
    assert_eq!(gui.arc_count(), 1);
    assert!(gui.sketch().dimensions.is_empty(), "click-only arc adds no dim");
    assert!(gui.app.arc_draw.is_none() && !gui.app.dim_editing);
}

#[test]
fn test_arc_tangent_snap_off_line_end() {
    // Line along x; the arc starts at its end. Aiming near the
    // tangent circle pulls the arc tangent to the line.
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.key(egui::Key::A);
    gui.click(v(2.0, 0.0)); // start on L0.p2
    gui.click(v(4.0, 2.0)); // end
    // The tangent circle: tangent to x-axis at (2,0), through (4,2):
    // center (2,2), r 2. Aim slightly off it, on the bulge side.
    let on_circle = v(2.0 + 2.0 * (-std::f64::consts::FRAC_PI_4).cos(), 2.0 + 2.0 * (-std::f64::consts::FRAC_PI_4).sin());
    gui.move_to(v(on_circle.x + 0.05, on_circle.y - 0.05));
    let s = gui.app.arc_draw.as_ref().unwrap();
    assert!(matches!(s.tangent, Some((crate::tools::TangentHost::Line(_), _))), "tangent snap offered: {:?}", s.tangent.map(|t| t.0));
    assert!((s.r - 2.0).abs() < 1e-6, "snapped to the tangent circle: r {}", s.r);
    gui.click(v(on_circle.x + 0.05, on_circle.y - 0.05));
    let a = gui.sketch().arcs.iter().next().expect("created").clone();
    assert!((a.radius.value - 2.0).abs() < 1e-6);
    assert!(near(a.center.value, v(2.0, 2.0), 1e-6), "center {:?}", a.center.value);
    assert_eq!(gui.sketch().tangent_la.len(), 1, "tangent constraint emitted");
}

#[test]
fn test_arc_tangent_snap_off_arc_end() {
    // Unit circle arc from (1,0) to (0,1) ccw; a new arc starts at
    // its end (0,1) where the tangent is (-1,0).
    let mut gui = Gui::new();
    gui.cmd("add_arc 1,0 0,1 0.70710678118654752,0.70710678118654752");
    gui.key(egui::Key::A);
    gui.click(v(0.0, 1.0)); // start on A0.end
    gui.click(v(-2.0, 1.0)); // end: on the tangent line -> no tangent circle
    gui.move_to(v(-1.0, 1.5));
    assert!(gui.app.arc_draw.as_ref().unwrap().tangent.is_none(), "end on the tangent line: no circle");
    gui.key(egui::Key::Escape);
    gui.key(egui::Key::A);
    gui.click(v(0.0, 1.0));
    gui.click(v(-2.0, 3.0)); // circle tangent at (0,1) to (-1,0) through (-2,3): center (0,3), r 2
    let ang = -0.75 * std::f64::consts::PI; // a point on that circle toward the lower-left
    let on = v(2.0 * ang.cos(), 3.0 + 2.0 * ang.sin());
    gui.move_to(v(on.x + 0.04, on.y + 0.04));
    let s = gui.app.arc_draw.as_ref().unwrap();
    assert!(matches!(s.tangent, Some((crate::tools::TangentHost::Arc(_), _))), "arc-arc tangent snap: {:?}", s.tangent.map(|t| t.0));
    gui.click(v(on.x + 0.04, on.y + 0.04));
    let a = gui.sketch().arcs.iter().nth(1).expect("created").clone();
    assert!(near(a.center.value, v(0.0, 3.0), 1e-6), "center {:?}", a.center.value);
    assert_eq!(gui.sketch().tangent_aa.len(), 1, "arc-arc tangent emitted");
}

#[test]
fn test_line_tangent_snap_off_arc() {
    // Unit circle arc ending at (0,1) with tangent (-1,0): a line
    // started there and aimed roughly along the tangent snaps onto it.
    let mut gui = Gui::new();
    gui.cmd("add_arc 1,0 0,1 0.70710678118654752,0.70710678118654752");
    gui.key(egui::Key::L);
    gui.click(v(0.0, 1.0)); // start on A0.end
    gui.click(v(-2.0, 1.06)); // within the pixel threshold of the tangent line y = 1
    assert_eq!(gui.line_count(), 1);
    let (p1, p2) = gui.line(0);
    assert!(near(p1, v(0.0, 1.0), 1e-9));
    assert!((p2.y - 1.0).abs() < 1e-9 && (p2.x + 2.0).abs() < 1e-6, "end pulled onto the tangent: {:?}", p2);
    assert_eq!(gui.sketch().tangent_la.len(), 1, "line-arc tangent emitted");
    // Aimed clearly off the tangent: no snap, no constraint.
    gui.key(egui::Key::Escape);
    gui.key(egui::Key::L);
    gui.click(v(0.0, 1.0));
    gui.click(v(-2.0, 2.0));
    assert_eq!(gui.line_count(), 2);
    assert_eq!(gui.sketch().tangent_la.len(), 1, "no second tangent");
}

// -- Ellipse tool -----------------------------------------------------

#[test]
fn test_ellipse_three_clicks() {
    let mut gui = Gui::new();
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    assert!(gui.app.dim_editing, "length input live from the center click");
    gui.click(v(3.0, 0.02)); // snaps horizontal -> rx exactly 3
    assert!(gui.app.ellipse_draw.as_ref().is_some_and(|s| s.axis_fixed));
    gui.click(v(1.0, 2.0)); // minor extent: perpendicular distance 2, completes
    let arc = gui.sketch().arcs.iter().next().expect("ellipse created").clone();
    assert!((arc.radius.value - 3.0).abs() < 1e-6, "rx {}", arc.radius.value);
    assert!((arc.radius_b.value - 2.0).abs() < 1e-6, "ry {}", arc.radius_b.value);
    assert!(arc.rotation.value.abs() < 1e-9, "rotation {}", arc.rotation.value);
    assert!(gui.sketch().dimensions.is_empty(), "click-only: no dims");
    assert!(gui.app.ellipse_draw.is_none(), "session ends at the third click");
    assert!(!gui.app.dim_editing);
}

#[test]
fn test_ellipse_hv_snap_and_q_disable() {
    let mut gui = Gui::new();
    gui.app.tool = Tool::DrawEllipse;
    // Near-vertical axis snaps to exactly pi/2.
    gui.click(v(0.0, 0.0));
    gui.click(v(0.02, 2.5));
    gui.click(v(1.0, 1.0));
    let arc = gui.sketch().arcs.iter().next().unwrap().clone();
    let quarter = std::f64::consts::FRAC_PI_2;
    assert!((arc.rotation.value.abs() - quarter).abs() < 1e-9,
        "snapped vertical: {}", arc.rotation.value);
    // Q held: same gesture stays unsnapped.
    gui.cmd("clear");
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.hold_key(egui::Key::Q);
    gui.click(v(0.02, 2.5));
    gui.release_key(egui::Key::Q);
    gui.click(v(1.0, 1.0));
    let arc = gui.sketch().arcs.iter().next().unwrap().clone();
    assert!((arc.rotation.value.abs() - quarter).abs() > 1e-4,
        "unsnapped: {}", arc.rotation.value);
}

#[test]
fn test_ellipse_typed_axes_make_dims() {
    let mut gui = Gui::new();
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.frame(); // input takes focus; select-all applies next frame
    gui.type_text("5"); // fix the semi-major while aiming
    gui.click(v(3.0, 0.0)); // direction only; length stays 5
    gui.frame();
    gui.type_text("1.5"); // fix the semi-minor
    gui.key(egui::Key::Enter); // completes without a minor click
    let arc = gui.sketch().arcs.iter().next().unwrap().clone();
    assert!((arc.radius.value - 5.0).abs() < 1e-6, "typed rx {}", arc.radius.value);
    assert!((arc.radius_b.value - 1.5).abs() < 1e-6, "typed ry {}", arc.radius_b.value);
    let dims = &gui.sketch().dimensions;
    assert_eq!(dims.len(), 2, "both typed values became dims: {:?}",
        dims.iter().map(|d| &d.name).collect::<Vec<_>>());
    assert!(dims.iter().any(|d| (d.value - 5.0).abs() < 1e-9 && !d.derived));
    assert!(dims.iter().any(|d| (d.value - 1.5).abs() < 1e-9 && !d.derived));
    // Whole creation (ellipse + dims) is one undo step.
    gui.cmd("undo");
    assert!(gui.sketch().arcs.iter().next().is_none(), "undo removed the ellipse");
    assert!(gui.sketch().dimensions.is_empty(), "undo removed the dims");
}

#[test]
fn test_ellipse_typed_minor_only() {
    let mut gui = Gui::new();
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.click(v(3.0, 0.0));
    gui.frame();
    gui.type_text("1.25"); // fix the minor while aiming it
    gui.click(v(1.0, 2.0)); // picks the side only; completes
    let arc = gui.sketch().arcs.iter().next().unwrap().clone();
    assert!((arc.radius_b.value - 1.25).abs() < 1e-6, "ry {}", arc.radius_b.value);
    let dims = &gui.sketch().dimensions;
    assert_eq!(dims.len(), 1, "only the typed minor became a dim");
    assert!((dims[0].value - 1.25).abs() < 1e-9);
}

#[test]
fn test_ellipse_typing_char_by_char() {
    // Regression: a half-typed value ("." or "s...") must take over
    // from mouse tracking instead of being clobbered by it, and the
    // first typed char must not be select-all'd away by the next.
    let mut gui = Gui::new();
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.frame();
    for ch in [".", "5"] { gui.type_text(ch); }
    assert_eq!(gui.app.dim_input, ".5", "input kept across frames");
    let rx = gui.app.ellipse_draw.as_ref().unwrap().rx;
    assert!((rx - 0.5).abs() < 1e-9, "rx follows the typed value: {}", rx);
    gui.click(v(3.0, 0.0)); // direction only
    gui.frame();
    for ch in ["s", "q", "r", "t", "(", "2", ")"] { gui.type_text(ch); }
    assert_eq!(gui.app.dim_input, "sqrt(2)");
    gui.key(egui::Key::Enter);
    let arc = gui.sketch().arcs.iter().next().expect("ellipse created").clone();
    assert!((arc.radius.value - 0.5).abs() < 1e-6, "rx {}", arc.radius.value);
    assert!((arc.radius_b.value - 2f64.sqrt()).abs() < 1e-6, "ry {}", arc.radius_b.value);
    let dims = &gui.sketch().dimensions;
    assert_eq!(dims.len(), 2);
    assert!(dims.iter().any(|d| d.expr_str.as_deref() == Some("sqrt(2)")),
        "expression stays live: {:?}", dims.iter().map(|d| &d.expr_str).collect::<Vec<_>>());
}

#[test]
fn test_ellipse_invalid_typed_blocks_completion() {
    let mut gui = Gui::new();
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.click(v(3.0, 0.0));
    gui.frame();
    gui.type_text("sqrt(");
    gui.click(v(1.0, 2.0)); // would complete, but the minor text is broken
    assert!(gui.sketch().arcs.iter().next().is_none(), "nothing created");
    assert!(gui.app.ellipse_draw.is_some(), "session stays open");
    assert!(gui.app.status_error.as_deref().is_some_and(|e| e.contains("Semi-minor")),
        "status error names the field: {:?}", gui.app.status_error);
    // Fixing the text lets the click through.
    gui.type_text("2)");
    gui.click(v(1.0, 2.0));
    let arc = gui.sketch().arcs.iter().next().expect("created after fix").clone();
    assert!((arc.radius_b.value - 2f64.sqrt()).abs() < 1e-6);
}

#[test]
fn test_ellipse_axis_end_snaps_to_point() {
    let mut gui = Gui::new();
    gui.cmd("add_point 3,0.3");
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    // Aim just off the point: the snap wins over the H/V pull.
    gui.move_to(v(2.97, 0.32));
    let s = gui.app.ellipse_draw.as_ref().unwrap();
    assert!(s.live_snap.is_some(), "snap offered while aiming the axis");
    assert!(s.hv.is_none(), "point snap suppresses H/V");
    gui.click(v(2.97, 0.32));
    gui.click(v(1.0, 2.0));
    let arc = gui.sketch().arcs.iter().next().expect("created").clone();
    let want = (3f64 * 3.0 + 0.3 * 0.3).sqrt();
    assert!((arc.radius.value - want).abs() < 1e-6, "rx from the snapped point: {}", arc.radius.value);
    assert!((arc.rotation.value - 0.3f64.atan2(3.0)).abs() < 1e-9, "axis points at it: {}", arc.rotation.value);
    // Rim tie: helper on the ellipse, coincident with P0.
    assert_eq!(gui.sketch().point_on_arc.len(), 1, "helper on the ellipse");
    assert_eq!(gui.sketch().coincident_pp.len(), 1, "helper tied to the point");
}

#[test]
fn test_ellipse_minor_snaps_through_point() {
    let mut gui = Gui::new();
    gui.cmd("add_point 1,1.5");
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.click(v(3.0, 0.0));
    gui.move_to(v(1.02, 1.48));
    let s = gui.app.ellipse_draw.as_ref().unwrap();
    assert!(s.live_snap.is_some(), "snap offered while aiming the minor");
    // Semi-minor that puts the rim through (1, 1.5) with rx = 3.
    let want = 1.5 / (1.0 - (1.0f64 / 3.0).powi(2)).sqrt();
    assert!((s.ry - want).abs() < 1e-6, "ry solves the rim through the point: {} vs {}", s.ry, want);
    gui.click(v(1.02, 1.48));
    let arc = gui.sketch().arcs.iter().next().expect("created").clone();
    assert!((arc.radius_b.value - want).abs() < 1e-6);
    assert_eq!(gui.sketch().point_on_arc.len(), 1);
    assert_eq!(gui.sketch().coincident_pp.len(), 1);
    // The tie holds through a solve: drag P0 and the rim follows it.
    gui.app.tool = Tool::Select;
    gui.drag(v(1.0, 1.5), v(1.0, 2.0));
    let p0 = gui.sketch().points.iter().find(|p| p.name == "P0").unwrap().pos.value;
    let arc = gui.sketch().arcs.iter().next().unwrap().clone();
    let (dx, dy) = (p0.x - arc.center.value.x, p0.y - arc.center.value.y);
    let (co, si) = (arc.rotation.value.cos(), arc.rotation.value.sin());
    let (u, w) = (dx * co + dy * si, -dx * si + dy * co);
    let r = (u / arc.radius.value).powi(2) + (w / arc.radius_b.value).powi(2);
    assert!((r - 1.0).abs() < 1e-3, "P0 still on the rim after the drag: {}", r);
}

#[test]
fn test_ellipse_typed_length_ignores_snap() {
    let mut gui = Gui::new();
    gui.cmd("add_point 3,0.3");
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.frame();
    gui.type_text("2");
    gui.move_to(v(2.97, 0.32));
    let s = gui.app.ellipse_draw.as_ref().unwrap();
    assert!(s.live_snap.is_none(), "typed length: rim would miss the point, no snap");
    assert!((s.rx - 2.0).abs() < 1e-9);
    gui.click(v(2.97, 0.32));
    gui.click(v(1.0, 2.0));
    assert_eq!(gui.sketch().point_on_arc.len(), 0, "no rim tie");
    assert_eq!(gui.sketch().dimensions.len(), 1, "typed major became a dim");
}

#[test]
fn test_ellipse_typed_angle_via_tab() {
    let mut gui = Gui::new();
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.frame();
    gui.type_text("2"); // semi-major
    gui.key(egui::Key::Tab); // into the angle field
    gui.frame();
    gui.type_text("30");
    let s = gui.app.ellipse_draw.as_ref().unwrap();
    assert_eq!(s.typed_angle.as_deref(), Some("30"), "angle field took the text: {:?}", s.angle_text);
    let want = 30f64.to_radians();
    assert!((s.dir.y.atan2(s.dir.x) - want).abs() < 1e-9, "dir follows the typed angle");
    // Mouse no longer steers the direction.
    gui.move_to(v(-1.0, 3.0));
    let s = gui.app.ellipse_draw.as_ref().unwrap();
    assert!((s.dir.y.atan2(s.dir.x) - want).abs() < 1e-9, "typed angle holds against the mouse");
    assert!((s.rx - 2.0).abs() < 1e-9, "typed length holds too");
    gui.key(egui::Key::Enter); // both typed: fixes the axis without a click
    assert!(gui.app.ellipse_draw.as_ref().unwrap().axis_fixed);
    gui.click(v(0.0, 1.5)); // minor side
    let arc = gui.sketch().arcs.iter().next().expect("created").clone();
    assert!((arc.rotation.value - want).abs() < 1e-9, "rotation {}", arc.rotation.value);
    assert!((arc.radius.value - 2.0).abs() < 1e-9);
    // Dims: radius (typed) + xangle (typed); the minor was clicked.
    let dims = &gui.sketch().dimensions;
    assert_eq!(dims.len(), 2, "{:?}", dims.iter().map(|d| (&d.name, d.value)).collect::<Vec<_>>());
    assert!(dims.iter().any(|d| matches!(d.kind, DimensionKind::ArcRotation(_)) && (d.value - 30.0).abs() < 1e-9),
        "xangle dim at 30 deg");
    assert!(arc.constraints.has_target_rotation, "rotation pinned by the dim");
}

#[test]
fn test_ellipse_typed_angle_mouse_picks_length() {
    let mut gui = Gui::new();
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.key(egui::Key::Tab);
    gui.frame();
    gui.type_text("90");
    // Length from the projection of the mouse onto the vertical axis.
    gui.move_to(v(0.7, 2.5));
    let s = gui.app.ellipse_draw.as_ref().unwrap();
    assert!((s.rx - 2.5).abs() < 1e-9, "rx = projection: {}", s.rx);
    assert!(s.typed_rx.is_none(), "length still live");
    gui.click(v(0.7, 2.5));
    gui.click(v(1.0, 0.0));
    let arc = gui.sketch().arcs.iter().next().expect("created").clone();
    assert!((arc.rotation.value.abs() - std::f64::consts::FRAC_PI_2).abs() < 1e-9);
    assert!((arc.radius.value - 2.5).abs() < 1e-9);
    assert!((arc.radius_b.value - 1.0).abs() < 1e-9);
    assert_eq!(gui.sketch().dimensions.len(), 1, "only the angle was typed");
}

#[test]
fn test_ellipse_escape_cancels_gesture() {
    let mut gui = Gui::new();
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(0.0, 0.0));
    gui.click(v(3.0, 0.0));
    assert!(gui.app.dim_editing);
    gui.key(egui::Key::Escape);
    assert!(gui.app.ellipse_draw.is_none(), "gesture dropped");
    assert!(!gui.app.dim_editing);
    assert!(gui.sketch().arcs.iter().next().is_none(), "no entity created");
}

#[test]
fn test_ellipse_o_toggle_and_center_snap() {
    let mut gui = Gui::new();
    assert_eq!(gui.app.tool, Tool::Select);
    gui.key(egui::Key::O);
    assert_eq!(gui.app.tool, Tool::DrawCircle);
    gui.key(egui::Key::O);
    assert_eq!(gui.app.tool, Tool::DrawEllipse);
    gui.key(egui::Key::O);
    assert_eq!(gui.app.tool, Tool::DrawCircle);
    // Center snapped onto an existing point ties the ellipse center.
    gui.cmd("add_point 1,1");
    let n_points = gui.sketch().points.len();
    gui.app.tool = Tool::DrawEllipse;
    gui.click(v(1.0, 1.0));
    gui.click(v(3.5, 1.0));
    gui.click(v(2.0, 2.0));
    let arc = gui.sketch().arcs.iter().next().unwrap().clone();
    assert!(near(arc.center.value, v(1.0, 1.0), 1e-6), "center on P0");
    assert_eq!(gui.sketch().points.len(), n_points + 1,
        "helper point bridges the center coincident");
}

#[test]
fn test_scale_box_select() {
    let mut gui = Gui::new();
    gui.cmd("add_line 1,0 2,0; add_line 1,1 2,1");
    gui.cmd("add_point 0,0");
    gui.app.tool = Tool::Scale;
    // Marquee over both lines (center point stays outside).
    gui.drag(v(0.5, -0.5), v(2.5, 1.5));
    let lines = gui.app.selection.iter()
        .filter(|s| matches!(s, Selection::Line(_)))
        .count();
    assert_eq!(lines, 2, "box selected both lines: {:?}", gui.app.selection);
    gui.double_click(v(0.0, 0.0));
    assert!(gui.app.scale_pending.is_some());
    gui.frame();
    gui.type_text("2");
    gui.key(egui::Key::Enter);
    let l1 = gui.sketch().lines.iter().find(|l| l.name == "L1").unwrap();
    assert!(near(l1.p1.value, v(2.0, 2.0), 0.05), "L1.p1 scaled: {:?}", l1.p1.value);
    assert!(near(l1.p2.value, v(4.0, 2.0), 0.05), "L1.p2 scaled: {:?}", l1.p2.value);
}

// -- Offset tool ----------------------------------------------------------

/// The id of the first meta-constraint.
fn o_mid(gui: &Gui) -> u32 {
    gui.sketch().metas[0].mid
}

/// Clicks build the sequence, the preview appears, the typed distance and
/// Enter create the offset, and the result is then under edit.
#[test]
fn test_offset_tool_click_type_enter() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 3,0 3,2");
    gui.app.enter_offset_tool();
    gui.frames(2); // the window settles before the canvas takes clicks
    assert!(gui.app.offset_tool.is_some());
    gui.click(v(1.5, 0.0));
    gui.click(v(3.0, 1.0));
    assert_eq!(gui.app.selection.len(), 2);
    gui.frame();
    let st = gui.app.offset_tool.as_ref().unwrap();
    assert!(st.plan.is_some(), "a preview plan for two connected lines: {:?}", st.error);
    assert!(st.edit.is_none());
    // The distance field has the focus with its text selected: typing
    // replaces it, Enter applies.
    gui.type_text("0.5");
    gui.frame();
    assert_eq!(gui.app.offset_tool.as_ref().unwrap().distance, "0.5");
    gui.key(egui::Key::Enter);
    gui.frame();
    assert_eq!(gui.sketch().metas.len(), 1, "the offset was created");
    assert_eq!(gui.line_count(), 4);
    let m = &gui.sketch().metas[0];
    let o = m.as_offset().unwrap();
    assert!((o.distance.value - 0.5).abs() < 1e-9);
    // Created: the tool closes, back in Select with the result selected.
    assert!(gui.app.offset_tool.is_none());
    assert_eq!(gui.app.tool, Tool::Select);
    assert_eq!(gui.app.selection.len(), 2);
}

/// Double-click walks the sequence; a marquee adds; the side follows the
/// mouse until Flip; Escape leaves the tool.
#[test]
fn test_offset_tool_walk_box_side_escape() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0 2,2 0,2");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.double_click(v(1.0, 0.0));
    assert_eq!(gui.app.selection.len(), 3, "the walk selected the whole open sequence: {:?}", gui.app.selection);
    // The mouse below the bottom line: right of the chain (0,0)->(2,0).
    gui.move_to(v(1.0, -1.0));
    assert_eq!(gui.app.offset_tool.as_ref().unwrap().side, -1.0);
    gui.move_to(v(1.0, 0.5));
    assert_eq!(gui.app.offset_tool.as_ref().unwrap().side, 1.0);
    // Flip fixes the side.
    {
        let st = gui.app.offset_tool.as_mut().unwrap();
        st.side = -st.side;
        st.side_fixed = true;
    }
    gui.move_to(v(1.0, 0.8));
    assert_eq!(gui.app.offset_tool.as_ref().unwrap().side, -1.0, "fixed");
    gui.key(egui::Key::Escape);
    assert_eq!(gui.app.tool, Tool::Select);
    assert!(gui.app.offset_tool.is_none());

    // A marquee over two lines of a fresh tool session.
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0 2,2");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.drag(v(-0.5, -0.5), v(2.5, 2.5));
    assert_eq!(gui.app.selection.len(), 2, "{:?}", gui.app.selection);
    gui.frame();
    assert!(gui.app.offset_tool.as_ref().unwrap().plan.is_some());
}

/// A fillet offset inward past its radius vanishes from the plan (the
/// window names it) and the created offset has no arc for it.
#[test]
fn test_offset_tool_vanishing_arc() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,3 0,0 4,0; fillet L0 L1 1");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.double_click(v(2.0, 0.0));
    assert_eq!(gui.app.selection.len(), 3, "{:?}", gui.app.selection);
    {
        let st = gui.app.offset_tool.as_mut().unwrap();
        st.distance = "2".into();
        st.side = 1.0; // inner side of the corner
        st.side_fixed = true;
    }
    gui.app.refresh_offset_plan();
    let plan = gui.app.offset_tool.as_ref().unwrap().plan.clone().expect("plan");
    assert_eq!(arael_sketch_backend::offset::dropped_names(gui.sketch(), &plan), vec!["A0"]);
    assert_eq!(plan.sides[0].sources, vec![0, 2]);
    gui.app.apply_offset();
    assert_eq!(gui.app.sketch.arcs.len(), 1, "no result arc");
    assert_eq!(gui.sketch().metas[0].as_offset().unwrap().sides[0].dropped, vec![1]);
}

/// Editing a vanished-arc offset back to a distance where the arcs
/// return rebuilds the side under the tool (the selection and the
/// markers follow the new entities).
#[test]
fn test_offset_tool_edit_rebuild_after_vanish() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,3 0,0 4,0; fillet L0 L1 1");
    // Created in the tool: walk the chain, inward by 2 (the fillet vanishes).
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.double_click(v(2.0, 0.0));
    assert_eq!(gui.app.selection.len(), 3, "{:?}", gui.app.selection);
    {
        let st = gui.app.offset_tool.as_mut().unwrap();
        st.distance = "2".into();
        st.side = 1.0;
        st.side_fixed = true;
    }
    gui.app.refresh_offset_plan();
    gui.app.apply_offset();
    assert_eq!(gui.sketch().metas.len(), 1);
    assert_eq!(gui.app.sketch.arcs.len(), 1, "no result arc");
    assert_eq!(gui.app.tool, Tool::Select, "the tool closed");
    gui.frame();
    let mid = gui.sketch().metas[0].mid;
    let p = gui.app.to_sketch(gui.app.meta_markers[0].pos);
    gui.double_click(p);
    assert_eq!(gui.app.offset_tool.as_ref().unwrap().edit, Some(mid));
    {
        let st = gui.app.offset_tool.as_mut().unwrap();
        st.distance = "0.5".into();
        st.pending_text = true;
    }
    gui.app.refresh_offset_plan();
    assert!(gui.app.offset_tool.as_ref().unwrap().plan.is_some());
    gui.app.update_offset();
    assert!(gui.app.status_error.is_none(), "{:?}", gui.app.status_error);
    gui.frames(2);
    assert_eq!(gui.app.sketch.arcs.len(), 2);
    assert!(gui.sketch().metas[0].as_offset().unwrap().sides[0].dropped.is_empty());
    assert_eq!(gui.app.selection.len(), 3, "the new result is selected");
}

/// An invalid set shows why and has no plan; a rejected plan likewise.
#[test]
fn test_offset_tool_errors() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0; add_line 5,-2 6,-3");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.click(v(1.0, 0.0));
    gui.click(v(5.5, -2.5));
    gui.frame();
    let st = gui.app.offset_tool.as_ref().unwrap();
    assert!(st.plan.is_none());
    assert!(st.error.as_deref().is_some_and(|e| e.contains("not one connected sequence")), "{:?}", st.error);
    // A bad distance.
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.click(v(1.0, 0.0));
    gui.app.offset_tool.as_mut().unwrap().distance = "nonsense(".into();
    gui.app.refresh_offset_plan();
    let st = gui.app.offset_tool.as_ref().unwrap();
    assert!(st.plan.is_none() && st.error.is_some());
}

/// Clicking a result of an existing offset opens it for editing, and a
/// change in the window updates the geometry.
#[test]
fn test_offset_tool_edit_existing() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 4,0; offset L0 1");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.click(v(2.0, 1.0)); // L1, the result
    let st = gui.app.offset_tool.as_ref().unwrap();
    let mid = gui.sketch().metas[0].mid;
    assert_eq!(st.edit, Some(mid));
    assert_eq!(st.distance, "1");
    // Symmetric: a second side appears.
    gui.app.offset_tool.as_mut().unwrap().kind = OffsetKind::Symmetric;
    gui.app.refresh_offset_plan();
    gui.app.update_offset();
    assert_eq!(gui.line_count(), 3);
    assert_eq!(gui.sketch().metas[0].as_offset().unwrap().sides.len(), 2);
    // A new distance moves the geometry.
    gui.app.offset_tool.as_mut().unwrap().distance = "2".into();
    gui.app.refresh_offset_plan();
    gui.app.update_offset();
    let ys: Vec<f64> = gui.sketch().lines.iter().filter(|l| l.name != "L0").map(|l| l.p1.value.y).collect();
    assert!(ys.iter().all(|y| (y.abs() - 2.0).abs() < 1e-6), "{:?}", ys);
    assert_eq!(gui.sketch().metas.len(), 1);
    // Deleting a result line drops the offset with a notice on the status line.
    gui.key(egui::Key::Escape);
    gui.cmd("select L1");
    gui.key(egui::Key::Delete);
    assert!(gui.sketch().metas.is_empty());
    assert!(gui.app.status_notice.as_deref().is_some_and(|n| n.contains("dropped")), "{:?}", gui.app.status_notice);
}

/// Round corners from the tool: the option is in the plan and the
/// created offset; toggling it in edit mode rebuilds the result; the
/// option is loaded back when an existing round offset is opened.
#[test]
fn test_offset_tool_round_corners() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 3,0 3,2");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.click(v(1.5, 0.0));
    gui.click(v(3.0, 1.0));
    gui.move_to(v(1.5, -1.0)); // below: the convex side of the corner
    gui.frame();
    let dof = gui.app.sketch.dof().unwrap();
    {
        let st = gui.app.offset_tool.as_mut().unwrap();
        assert!(!st.round);
        st.round = true;
        st.side_fixed = true;
        st.side = -1.0;
    }
    gui.app.refresh_offset_plan();
    let st = gui.app.offset_tool.as_ref().unwrap();
    let plan = st.plan.as_ref().expect("plan");
    assert!(plan.params.round);
    assert_eq!(plan.sides[0].joints.iter().filter(|j| j.round.is_some()).count(), 1);
    gui.app.apply_offset();
    assert_eq!(gui.sketch().metas.len(), 1);
    let o = gui.sketch().metas[0].as_offset().unwrap().clone();
    assert!(o.round && o.sides[0].corners.len() == 1);
    assert_eq!(gui.sketch().arcs.len(), 1, "the corner arc");
    assert_eq!(gui.app.sketch.dof().unwrap(), dof);
    // Edit mode (reopened from the marker): sharp again removes the arc.
    gui.app.open_meta_edit(o_mid(&gui));
    assert!(gui.app.offset_tool.as_ref().unwrap().edit.is_some());
    gui.app.offset_tool.as_mut().unwrap().round = false;
    gui.app.refresh_offset_plan();
    gui.app.update_offset();
    assert_eq!(gui.sketch().arcs.len(), 0);
    assert!(!gui.sketch().metas[0].as_offset().unwrap().round);
    assert_eq!(gui.app.sketch.dof().unwrap(), dof);

    // Opening a round offset loads the option.
    let mut gui = Gui::new();
    gui.cmd("add_rect 0,0 4,3; offset L0 L1 L2 L3 1 outward round");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.click(v(2.0, -1.0)); // the bottom result line
    let st = gui.app.offset_tool.as_ref().unwrap();
    assert!(st.edit.is_some() && st.round, "{:?} {}", st.edit, st.round);
}

/// Caps from the tool: planned and previewed for a symmetric open
/// sequence, created with the offset, editable; the option is loaded
/// back for an existing capped offset and drops with one side.
#[test]
fn test_offset_tool_caps() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 3,0 3,2");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.click(v(1.5, 0.0));
    gui.click(v(3.0, 1.0));
    gui.frame();
    let dof = gui.app.sketch.dof().unwrap();
    {
        let st = gui.app.offset_tool.as_mut().unwrap();
        st.kind = OffsetKind::Symmetric;
        st.caps = CapKind::Round;
    }
    gui.app.refresh_offset_plan();
    let plan = gui.app.offset_tool.as_ref().unwrap().plan.clone().expect("plan");
    assert_eq!(plan.caps.len(), 2);
    // The preview shows the results, and the two caps.
    assert_eq!(arael_sketch_backend::offset::preview_polylines(&plan, 8).len(), 6);
    gui.app.apply_offset();
    let o = gui.sketch().metas[0].as_offset().unwrap().clone();
    assert_eq!(o.caps.kind, CapKind::Round);
    assert_eq!(o.caps.entities.len(), 2);
    assert_eq!(gui.sketch().arcs.len(), 2);
    assert_eq!(gui.app.sketch.dof().unwrap(), dof);
    // Edit (reopened from the marker): line caps.
    gui.app.open_meta_edit(o_mid(&gui));
    gui.app.offset_tool.as_mut().unwrap().caps = CapKind::Line;
    gui.app.refresh_offset_plan();
    gui.app.update_offset();
    assert_eq!(gui.sketch().arcs.len(), 0);
    assert_eq!(gui.sketch().metas[0].as_offset().unwrap().caps.kind, CapKind::Line);
    assert_eq!(gui.app.sketch.dof().unwrap(), dof);
    // One side: the window drops the caps (the state is fixed up when
    // the window renders), and the update goes through.
    gui.app.offset_tool.as_mut().unwrap().kind = OffsetKind::OneSide;
    gui.frame();
    assert_eq!(gui.app.offset_tool.as_ref().unwrap().caps, CapKind::None);

    // Opening a capped offset loads the option.
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 4,0; offset L0 1 symmetric caps line");
    gui.app.enter_offset_tool();
    gui.frames(2);
    gui.click(v(2.0, 1.0));
    let st = gui.app.offset_tool.as_ref().unwrap();
    assert!(st.edit.is_some());
    assert_eq!(st.caps, CapKind::Line);
}

/// The meta-constraint marker: drawn on the first result, a click
/// selects it (its parts highlight, the status shows its description),
/// double-click opens the offset for editing, Delete dissolves it; it
/// hides with the constraint markers.
#[test]
fn test_meta_marker_select_open_delete() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 3,0 3,2; offset L0 L1 1");
    gui.frame();
    assert_eq!(gui.app.meta_markers.len(), 1);
    let mid = gui.sketch().metas[0].mid;
    assert_eq!(gui.app.meta_markers[0].mid, mid);
    let marker_at = |gui: &Gui| gui.app.to_sketch(gui.app.meta_markers[0].pos);
    // Click: selected, exclusive.
    let p = marker_at(&gui);
    gui.click(p);
    assert_eq!(gui.app.selection, vec![Selection::Meta(mid)]);
    assert!(gui.app.hit_test_selection(p, 0.2) == Some(Selection::Meta(mid)));
    // Double-click: the Offset tool opens on it.
    let p = marker_at(&gui);
    gui.double_click(p);
    assert_eq!(gui.app.tool, Tool::Offset);
    let st = gui.app.offset_tool.as_ref().expect("tool state");
    assert_eq!(st.edit, Some(mid));
    gui.key(egui::Key::Escape);
    assert_eq!(gui.app.tool, Tool::Select);
    // Delete on the selected marker dissolves: geometry stays.
    gui.frame();
    let p = marker_at(&gui);
    gui.click(p);
    assert_eq!(gui.app.selection, vec![Selection::Meta(mid)]);
    gui.key(egui::Key::Delete);
    assert!(gui.sketch().metas.is_empty());
    assert_eq!(gui.line_count(), 4);
    assert!(gui.app.selection.is_empty());
    gui.frame();
    assert!(gui.app.meta_markers.is_empty());
    // Undo brings the meta and its marker back; hiding constraints
    // hides the marker too.
    gui.key_with(egui::Key::Z, egui::Modifiers::CTRL);
    gui.frame();
    assert_eq!(gui.app.meta_markers.len(), 1);
    gui.app.show_constraints = false;
    gui.frame();
    assert!(gui.app.meta_markers.is_empty());

    // A two-sided offset has a marker on each side, both for the one meta.
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 4,0; offset L0 1 symmetric");
    gui.frame();
    let mid = gui.sketch().metas[0].mid;
    assert_eq!(gui.app.meta_markers.len(), 2);
    assert!(gui.app.meta_markers.iter().all(|m| m.mid == mid));
    let ys: Vec<f64> = gui.app.meta_markers.iter().map(|m| gui.app.to_sketch(m.pos).y).collect();
    assert!(ys.iter().any(|y| *y > 0.5) && ys.iter().any(|y| *y < -0.5), "{:?}", ys);
    let p = gui.app.to_sketch(gui.app.meta_markers[1].pos);
    gui.click(p);
    assert_eq!(gui.app.selection, vec![Selection::Meta(mid)]);
}

/// The Offset tool's edit mode previews a typed distance before it is
/// applied, and a marker click inside the tool opens that offset.
#[test]
fn test_offset_tool_edit_preview_and_marker() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 3,0 3,2; offset L0 L1 1");
    gui.app.enter_offset_tool();
    gui.frames(2);
    let mid = gui.sketch().metas[0].mid;
    let p = gui.app.to_sketch(gui.app.meta_markers[0].pos);
    gui.click(p);
    let st = gui.app.offset_tool.as_ref().unwrap();
    assert_eq!(st.edit, Some(mid));
    assert!(!st.pending_text);
    // A typed distance: planned (previewed) but not applied until Enter.
    {
        let st = gui.app.offset_tool.as_mut().unwrap();
        st.distance = "1.5".into();
        st.pending_text = true;
    }
    gui.app.refresh_offset_plan();
    let st = gui.app.offset_tool.as_ref().unwrap();
    let plan = st.plan.as_ref().unwrap_or_else(|| panic!("plan: {:?}", st.error));
    assert!((plan.params.distance.value - 1.5).abs() < 1e-9);
    assert!((gui.sketch().metas[0].as_offset().unwrap().distance.value - 1.0).abs() < 1e-9, "not applied yet");
    assert!(st.edit.is_some() && st.pending_text, "the preview condition holds");
    gui.app.update_offset();
    assert!((gui.sketch().metas[0].as_offset().unwrap().distance.value - 1.5).abs() < 1e-9);
    assert!(!gui.app.offset_tool.as_ref().unwrap().pending_text);
}

// -- Pattern tool ---------------------------------------------------------

/// Lock over a selection runs as one batch: a single history entry,
/// every vertex locked, DOF cache fresh. A second apply unlocks; the
/// shared (coincident) corner selected through both lines must not
/// produce duplicate unlocks.
#[test]
fn test_lock_selection_batched() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0; add_line 2,0 2,2");
    let lines: Vec<_> = gui.sketch().lines.refs().collect();
    gui.app.selection = lines.iter()
        .flat_map(|&l| [Selection::LineP1(l), Selection::LineP2(l)])
        .collect();
    let actions0 = gui.app.history.actions.len();
    gui.app.apply_lock();
    assert_eq!(gui.app.history.actions.len(), actions0 + 1, "one entry for the lock");
    for &l in &lines {
        assert!(!gui.sketch().lines[l].p1.optimize, "p1 locked");
        assert!(!gui.sketch().lines[l].p2.optimize, "p2 locked");
    }
    assert_eq!(gui.sketch().cached_dof(), Some(0), "DOF display refreshed");
    gui.app.apply_lock();
    assert_eq!(gui.app.history.actions.len(), actions0 + 2, "one entry for the unlock");
    for &l in &lines {
        assert!(gui.sketch().lines[l].p1.optimize, "p1 unlocked");
        assert!(gui.sketch().lines[l].p2.optimize, "p2 unlocked");
    }
    gui.frames(2);
}

/// Construction toggle over a selection runs as one batch with a
/// single history entry; a second toggle restores.
#[test]
fn test_toggle_construction_batched() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 2,0; add_line 2,0 2,2; add_circle 5,0 1");
    gui.cmd("select all");
    let actions0 = gui.app.history.actions.len();
    gui.app.apply_toggle_construction();
    assert_eq!(gui.app.history.actions.len(), actions0 + 1, "one entry");
    assert!(gui.sketch().lines.iter().all(|l| l.construction));
    assert!(gui.sketch().arcs.iter().all(|a| a.construction));
    gui.app.apply_toggle_construction();
    assert!(gui.sketch().lines.iter().all(|l| !l.construction));
    assert!(gui.sketch().arcs.iter().all(|a| !a.construction));
    gui.frames(2);
}

/// Backspace deletes the whole selection as one batch: a single
/// history entry, stale refs inside the batch tolerated (the line's
/// cascade removes the selected constraint first), one undo restores
/// everything, and the DOF cache is left fresh.
#[test]
fn test_delete_selection_batched() {
    let mut gui = Gui::new();
    // The second add_line auto-connects to the first: coincident C1.
    gui.cmd("add_line 0,0 2,0; add_line 2,0 2,2; add_point 4,4; add_circle 6,0 1");
    gui.cmd("horizontal L0");
    let actions0 = gui.app.history.actions.len();
    gui.cmd("select L0 L1 P0 A0");
    // Constraints aren't select-command addressable; push them the way
    // a glyph click does.
    use arael_sketch_backend::ids::ConstraintId;
    let nid = gui.sketch().coincident_ll12.first().expect("auto coincident").nid;
    let l0 = gui.sketch().lines.refs().next().unwrap();
    gui.app.selection.push(Selection::Constraint(ConstraintId::Numbered(nid)));
    gui.app.selection.push(Selection::Constraint(ConstraintId::Horizontal(l0)));
    gui.key(egui::Key::Backspace);
    assert_eq!(gui.line_count(), 0);
    assert_eq!(gui.arc_count(), 0);
    assert_eq!(gui.sketch().points.refs().count(), 0);
    assert!(gui.app.selection.is_empty());
    assert_eq!(gui.app.history.actions.len(), actions0 + 1,
        "the whole delete is one history entry");
    assert!(gui.sketch().cached_dof().is_some(), "DOF display refreshed");
    gui.key_with(egui::Key::Z, egui::Modifiers::CTRL);
    assert_eq!(gui.line_count(), 2, "one undo restores everything");
    assert_eq!(gui.arc_count(), 1);
    assert_eq!(gui.sketch().points.refs().count(), 1);
    assert_eq!(gui.sketch().coincident_ll12.len(), 1);
    assert!(gui.sketch().lines.iter().next().unwrap().constraints.horizontal);
}

/// Rect pattern from the tool: the batched engine run defers the DOF
/// refresh to its end_group, which must leave the display cache fresh
/// (a stale cache reads as None through cached_dof).
#[test]
fn test_pattern_tool_apply_leaves_dof_fresh() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 1,0");
    gui.cmd("select all");
    gui.app.enter_pattern_tool();
    {
        let st = gui.app.pattern_tool.as_mut().unwrap();
        st.kind = crate::pattern_tool::PatternToolKind::Rectangular;
        st.quantity1 = "3".into();
        st.distance1 = "2".into();
        st.quantity2 = "3".into();
        st.distance2 = "2".into();
    }
    gui.app.refresh_pattern_plan();
    gui.app.apply_pattern();
    assert!(gui.app.status_error.is_none(), "{:?}", gui.app.status_error);
    assert_eq!(gui.sketch().metas.len(), 1);
    assert_eq!(gui.line_count(), 9);
    assert!(!gui.app.dof_hold, "the hold ends with the operation");
    assert_eq!(gui.sketch().cached_dof(), Some(4),
        "a free line keeps 4 DOF, rigid copies add none; None means end_group did not refresh");
    gui.frames(2);
}

/// Circular pattern from the tool: the set by clicks, the center by
/// Pick + click, the numbers in the window; Create makes it and closes
/// the tool; every copy gets a marker; the marker opens it for editing.
#[test]
fn test_pattern_tool_circular() {
    let mut gui = Gui::new();
    gui.cmd("add_line 1,0 3,0; add_point 0,0");
    gui.app.enter_pattern_tool();
    gui.frames(2);
    gui.click(v(2.0, 0.0));
    assert_eq!(gui.app.selection, vec![Selection::Line(gui.sketch().lines.refs().next().unwrap())]);
    // Pick the center: the next click goes to the center, not the set.
    gui.app.pattern_tool.as_mut().unwrap().picking = Some(crate::pattern_tool::Pick::Center);
    gui.frame();
    gui.click(v(0.0, 0.0));
    let st = gui.app.pattern_tool.as_ref().unwrap();
    assert!(matches!(st.center, Some(CenterRef::Point(_))), "{:?}", st.center);
    assert!(st.picking.is_none());
    assert_eq!(gui.app.selection.len(), 1, "the set is unchanged");
    gui.app.pattern_tool.as_mut().unwrap().quantity = "4".into();
    gui.app.refresh_pattern_plan();
    let plan = gui.app.pattern_tool.as_ref().unwrap().plan.clone().expect("plan");
    assert_eq!(plan.copies.len(), 3);
    assert_eq!(arael_sketch_backend::pattern::preview_polylines(&plan, 8).len(), 3);
    let dof = gui.app.sketch.dof().unwrap();
    gui.app.apply_pattern();
    assert_eq!(gui.app.tool, Tool::Select);
    assert!(gui.app.pattern_tool.is_none());
    assert_eq!(gui.sketch().metas.len(), 1);
    assert_eq!(gui.line_count(), 4);
    assert_eq!(gui.app.sketch.dof().unwrap(), dof);
    assert_eq!(gui.app.selection.len(), 3, "the copies are selected");
    gui.frame();
    assert_eq!(gui.app.meta_markers.len(), 3, "one marker per copy");
    // Double-click a marker: the Pattern tool opens in edit mode. The
    // marker keeps its place while the selection (and with it the image
    // markers on the copies) changes.
    let p = gui.app.to_sketch(gui.app.meta_markers[1].pos);
    gui.double_click(p);
    assert_eq!(gui.app.tool, Tool::Pattern);
    let st = gui.app.pattern_tool.as_ref().unwrap();
    assert_eq!(st.edit, Some(gui.sketch().metas[0].mid));
    assert_eq!(st.quantity, "4");
    // Image markers show while a copy is selected, and stay while the
    // image constraint itself is the selection (a click on the marker).
    gui.frame();
    let marker = gui.app.constraint_markers.iter().find(|m| m.symbol == ConstraintSymbol::Image).map(|m| (m.pos, m.id)).expect("image marker");
    gui.key(egui::Key::Escape);
    assert_eq!(gui.app.tool, Tool::Select);
    gui.cmd("select L1");
    gui.frame();
    let marker = gui.app.constraint_markers.iter().find(|m| m.symbol == ConstraintSymbol::Image).map(|m| (m.pos, m.id)).unwrap_or(marker);
    gui.click(gui.app.to_sketch(marker.0));
    assert_eq!(gui.app.selection, vec![Selection::Constraint(marker.1)]);
    gui.frame();
    assert!(gui.app.constraint_markers.iter().any(|m| m.symbol == ConstraintSymbol::Image && m.id == marker.1), "the selected image marker is drawn");
}

/// Rectangular pattern from the tool, then an in-place distance edit
/// with a preview until Enter, and a quantity edit that rebuilds.
#[test]
fn test_pattern_tool_rect_edit() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 1,0");
    gui.app.enter_pattern_tool();
    gui.frames(2);
    gui.click(v(0.5, 0.0));
    {
        let st = gui.app.pattern_tool.as_mut().unwrap();
        st.kind = crate::pattern_tool::PatternToolKind::Rectangular;
        st.quantity1 = "3".into();
        st.distance1 = "2".into();
    }
    gui.app.refresh_pattern_plan();
    assert_eq!(gui.app.pattern_tool.as_ref().unwrap().plan.as_ref().map(|p| p.copies.len()), Some(2));
    gui.app.apply_pattern();
    assert_eq!(gui.line_count(), 3);
    let mid = gui.sketch().metas[0].mid;
    gui.app.open_meta_edit(mid);
    assert_eq!(gui.app.tool, Tool::Pattern);
    // A typed distance previews until applied.
    {
        let st = gui.app.pattern_tool.as_mut().unwrap();
        st.distance1 = "3".into();
        st.pending_text = true;
    }
    gui.app.refresh_pattern_plan();
    let st = gui.app.pattern_tool.as_ref().unwrap();
    assert!(st.plan.is_some() && st.pending_text);
    let names_before: Vec<String> = gui.sketch().lines.iter().map(|l| l.name.clone()).collect();
    gui.app.update_pattern();
    let names_after: Vec<String> = gui.sketch().lines.iter().map(|l| l.name.clone()).collect();
    assert_eq!(names_before, names_after, "moved in place");
    let last = gui.sketch().lines.iter().last().unwrap();
    assert!(near(last.p1.value, v(6.0, 0.0), 1e-6), "{:?}", last.p1.value);
    // Quantity rebuilds.
    gui.app.pattern_tool.as_mut().unwrap().quantity1 = "2".into();
    gui.app.refresh_pattern_plan();
    gui.app.update_pattern();
    assert_eq!(gui.line_count(), 2);
    assert!(gui.app.status_error.is_none(), "{:?}", gui.app.status_error);
}

// -- On-normal constraint mode ------------------------------------------

/// Two endpoint clicks in the mode, placed first: the constraint is
/// applied, its marker is drawn, and the mode stays for the next pair.
#[test]
fn test_on_normal_mode_two_endpoints() {
    let mut gui = Gui::new();
    gui.cmd("add_line 0,0 3,0; add_line 4,1 6,1");
    gui.app.tool = Tool::ConstraintMode(ConstraintType::OnNormal);
    gui.click(v(4.0, 1.0)); // L1.p1, the endpoint to place
    assert_eq!(gui.app.selection.len(), 1);
    gui.click(v(3.0, 0.0)); // L0.p2, the reference
    assert_eq!(gui.sketch().on_normal_ll.len(), 1, "constraint applied");
    let c = &gui.sketch().on_normal_ll[0];
    assert!(!c.placed_end && c.ref_end, "L1.p1 placed on L0's normal at p2");
    assert!(gui.app.selection.is_empty(), "selection cleared for the next pair");
    assert!(matches!(gui.app.tool, Tool::ConstraintMode(ConstraintType::OnNormal)));
    gui.frame();
    assert!(gui.app.constraint_markers.iter().any(|m| m.symbol == ConstraintSymbol::OnNormal), "marker drawn");
    // A body click is not a valid pick in this mode.
    gui.click(v(1.5, 0.0));
    assert!(gui.app.selection.is_empty());
}
