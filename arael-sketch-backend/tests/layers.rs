// Unit coverage for the previously untested layers: ids.rs name
// round-trips (including the dimension-backed and helper-bridge
// addressability rules), geometry.rs circumscribed_arc, and
// history.rs group mechanics.

use arael::model::CrossBlock;
use arael::vect::vect2d;
use arael_sketch_backend::actions::Action;
use arael_sketch_backend::history::{CursorState, History};
use arael_sketch_backend::ids::{constraint_id_name, find_constraint_by_name, ConstraintId};
use arael_sketch_backend::geometry::circumscribed_arc;
use arael_sketch_solver::{CoincidentLP1, DistancePP, Parallel, Sketch};

// ---------------------------------------------------------------------------
// ids
// ---------------------------------------------------------------------------

#[test]
fn constraint_names_round_trip() {
    let mut s = Sketch::new();
    let l0 = s.add_line(vect2d::new(0.0, 0.0), vect2d::new(4.0, 0.0));
    let l1 = s.add_line(vect2d::new(0.0, 2.0), vect2d::new(4.0, 2.0));
    s.lines[l0].constraints.horizontal = true;
    s.parallel.push(Parallel { a: l0, b: l1, nid: 0, cid: 0, hb: CrossBlock::new() });
    s.assign_constraint_names();
    let nid = s.parallel[0].nid;

    // Numbered: name -> id -> name.
    let id = find_constraint_by_name(&s, &format!("C{}", nid)).unwrap();
    assert!(matches!(id, ConstraintId::Numbered(n) if n == nid));
    assert_eq!(constraint_id_name(&s, id).unwrap(), format!("C{}", nid));

    // Flag: only resolves while the flag is set.
    let id = find_constraint_by_name(&s, "CL0H").unwrap();
    assert!(matches!(id, ConstraintId::Horizontal(r) if r == l0));
    assert_eq!(constraint_id_name(&s, id).unwrap(), "CL0H");
    assert!(find_constraint_by_name(&s, "CL0V").is_none());
    s.lines[l0].constraints.horizontal = false;
    assert!(find_constraint_by_name(&s, "CL0H").is_none());

    // Unknown and sentinel names resolve to nothing.
    assert!(find_constraint_by_name(&s, "C0").is_none());
    assert!(find_constraint_by_name(&s, "C99999").is_none());
    assert!(find_constraint_by_name(&s, "nonsense").is_none());
}

#[test]
fn dimension_backed_and_bridge_constraints_are_not_addressable() {
    let mut s = Sketch::new();
    let p0 = s.add_point(vect2d::new(0.0, 0.0));
    let p1 = s.add_point(vect2d::new(3.0, 0.0));
    s.distance_pp.push(DistancePP {
        a: p0, b: p1, distance: 3.0, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    let l = s.add_line(vect2d::new(0.0, 2.0), vect2d::new(4.0, 2.0));
    let h = s.add_helper_point(vect2d::new(0.0, 2.0));
    s.coincident_lp1.push(CoincidentLP1 {
        line: l, point: h, nid: 0, cid: 0, hb: CrossBlock::new(),
    });
    s.assign_constraint_names();

    // A dimension-backed constraint deletes through its d<n>, not C<n>.
    let dist_nid = s.distance_pp[0].nid;
    assert!(find_constraint_by_name(&s, &format!("C{}", dist_nid)).is_none(),
        "dimension-backed constraint must not be addressable by C-name");

    // A helper bridge is internal plumbing.
    let bridge_nid = s.coincident_lp1[0].nid;
    assert!(find_constraint_by_name(&s, &format!("C{}", bridge_nid)).is_none(),
        "helper bridge must not be addressable by C-name");
    assert!(constraint_id_name(&s, ConstraintId::HelperBridge(h)).is_none());
}

// ---------------------------------------------------------------------------
// geometry
// ---------------------------------------------------------------------------

#[test]
fn circumscribed_arc_geometry() {
    // Circle through three points of the unit circle around (2, 1).
    let (c, r, _, _, _) = circumscribed_arc(
        vect2d::new(3.0, 1.0), vect2d::new(1.0, 1.0), vect2d::new(2.0, 2.0)).unwrap();
    assert!((c.x - 2.0).abs() < 1e-9 && (c.y - 1.0).abs() < 1e-9, "center {:?}", (c.x, c.y));
    assert!((r - 1.0).abs() < 1e-9, "radius {}", r);

    // CCW flag follows the winding of start -> mid -> end.
    let (_, _, _, _, ccw_up) = circumscribed_arc(
        vect2d::new(3.0, 1.0), vect2d::new(1.0, 1.0), vect2d::new(2.0, 2.0)).unwrap();
    let (_, _, _, _, ccw_down) = circumscribed_arc(
        vect2d::new(3.0, 1.0), vect2d::new(1.0, 1.0), vect2d::new(2.0, 0.0)).unwrap();
    assert_ne!(ccw_up, ccw_down);

    // Collinear and coincident points have no circumscribed circle.
    assert!(circumscribed_arc(
        vect2d::new(0.0, 0.0), vect2d::new(2.0, 0.0), vect2d::new(1.0, 0.0)).is_none());
    assert!(circumscribed_arc(
        vect2d::new(1.0, 1.0), vect2d::new(1.0, 1.0), vect2d::new(2.0, 0.0)).is_none());
}

// ---------------------------------------------------------------------------
// history
// ---------------------------------------------------------------------------

#[test]
fn history_groups_undo_and_goto() {
    let mut s = Sketch::new();
    let mut h = History::new(&s);
    let cur = CursorState::default();

    // Group 1: one line.
    h.begin_group();
    let a1 = Action::AddLine { p1: vect2d::new(0.0, 0.0), p2: vect2d::new(4.0, 0.0) };
    a1.apply(&mut s);
    h.push(a1, &s, cur.clone());

    // Group 2: two points in one group -- one undo unit.
    h.begin_group();
    for pos in [vect2d::new(1.0, 1.0), vect2d::new(2.0, 2.0)] {
        let a = Action::AddPoint { pos };
        a.apply(&mut s);
        h.push(a, &s, cur.clone());
    }
    assert_eq!(h.group_list().len(), 2);
    assert_eq!(h.cursor, 3);

    // Undo removes the whole two-action group.
    let (restored, _) = h.undo().unwrap();
    assert_eq!(restored.points.refs().count(), 0);
    assert_eq!(restored.lines.refs().count(), 1);
    assert_eq!(h.cursor, 1);
    assert!(h.can_undo() && h.can_redo());

    // Redo brings the whole group back.
    let (restored, _) = h.redo().unwrap();
    assert_eq!(restored.points.refs().count(), 2);
    assert_eq!(h.cursor, 3);
    assert!(!h.can_redo());

    // goto jumps to an absolute cursor; 0 is the initial state.
    let (restored, _) = h.goto(1).unwrap();
    assert_eq!(restored.lines.refs().count(), 1);
    assert_eq!(restored.points.refs().count(), 0);
    let (restored, _) = h.goto(0).unwrap();
    assert_eq!(restored.lines.refs().count(), 0);
    let (restored, _) = h.goto(3).unwrap();
    assert_eq!(restored.points.refs().count(), 2);
}
