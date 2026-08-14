// GUI gesture tests: real input events through the real update loop,
// assertions on sketch/app state. No pixels. See test_harness.rs.

use eframe::egui;
use crate::test_harness::{Gui, v, near};
use crate::tools::Tool;
use arael_sketch_solver::DimensionKind;
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
