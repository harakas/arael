// Constraint identification and selection types used by commands/actions
// and the GUI. Carved out of the old arael-sketch/src/tools.rs so the
// backend crate can reference them without pulling in egui.

use arael::refs::Ref;
use arael_sketch_solver::*;

// Selection -- what entity is selected for constraint application.
// Held by CommandContext so command output and the GUI can share state.
#[derive(Clone, Copy, PartialEq)]
pub enum Selection {
    Point(Ref<Point>),
    Line(Ref<Line>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    Arc(Ref<Arc>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
    Constraint(ConstraintId),
    Dimension(usize),
}

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum CoincidentKind {
    PP, LP1, LP2,
    LL11, LL12, LL21, LL22,
    PointOnLine, PointOnArc,
    LP1OnLine, LP2OnLine,
    LP1OnArc, LP2OnArc,
    ArcCenter, ArcStart, ArcEnd,
    LP1ArcCenter, LP2ArcCenter,
    LP1ArcStart, LP2ArcStart,
    LP1ArcEnd, LP2ArcEnd,
    ArcCenterStart, ArcCenterEnd,
    ArcStartCenter, ArcEndCenter,
    ArcStartStart, ArcStartEnd,
    ArcEndStart, ArcEndEnd,
}

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum MidpointKind {
    Point,
    LP1,
    LP2,
    ArcStart,
    ArcEnd,
    ArcPoint,
    LP1Arc,
    LP2Arc,
    ArcStartArc,
    ArcEndArc,
}

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum ConstraintId {
    Horizontal(Ref<Line>),
    Vertical(Ref<Line>),
    Parallel(usize),
    Perpendicular(usize),
    EqualLength(usize),
    EqualRadius(usize),
    TangentLA(usize),
    TangentAA(usize),
    Collinear(usize),
    Symmetry(usize),
    SymmetryPP(usize),
    SymmetryAA(usize),
    Concentric(usize),
    Coincident(CoincidentKind, usize),
    Midpoint(MidpointKind, usize),
    HelperBridge(Ref<Point>),
}

/// Look up a constraint by its auto-assigned name.
///
/// Numeric names "C<n>" scan every Vec-stored constraint that has a
/// ConstraintId variant and return the first match. Synthetic flag
/// names like "CL0H" / "CL3V" parse the entity reference and validate
/// the flag is currently set on that line.
pub fn find_constraint_by_name(sketch: &Sketch, name: &str) -> Option<ConstraintId> {
    if let Some(rest) = name.strip_prefix('C')
        && let Ok(nid) = rest.parse::<u32>() {
        if nid == 0 { return None; }
        for (i, c) in sketch.coincident_pp.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::PP, i)); } }
        for (i, c) in sketch.coincident_lp1.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP1, i)); } }
        for (i, c) in sketch.coincident_lp2.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP2, i)); } }
        for (i, c) in sketch.coincident_ll11.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LL11, i)); } }
        for (i, c) in sketch.coincident_ll12.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LL12, i)); } }
        for (i, c) in sketch.coincident_ll21.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LL21, i)); } }
        for (i, c) in sketch.coincident_ll22.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LL22, i)); } }
        for (i, c) in sketch.point_on_line.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::PointOnLine, i)); } }
        for (i, c) in sketch.midpoint.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::Point, i)); } }
        for (i, c) in sketch.midpoint_lp1.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::LP1, i)); } }
        for (i, c) in sketch.midpoint_lp2.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::LP2, i)); } }
        for (i, c) in sketch.midpoint_arc_start.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::ArcStart, i)); } }
        for (i, c) in sketch.midpoint_arc_end.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::ArcEnd, i)); } }
        for (i, c) in sketch.midpoint_arc_point.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::ArcPoint, i)); } }
        for (i, c) in sketch.midpoint_lp1_arc.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::LP1Arc, i)); } }
        for (i, c) in sketch.midpoint_lp2_arc.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::LP2Arc, i)); } }
        for (i, c) in sketch.midpoint_arc_start_arc.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::ArcStartArc, i)); } }
        for (i, c) in sketch.midpoint_arc_end_arc.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Midpoint(MidpointKind::ArcEndArc, i)); } }
        for (i, c) in sketch.point_on_arc.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::PointOnArc, i)); } }
        for (i, c) in sketch.parallel.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Parallel(i)); } }
        for (i, c) in sketch.perpendicular.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Perpendicular(i)); } }
        for (i, c) in sketch.collinear.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Collinear(i)); } }
        for (i, c) in sketch.equal_length.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::EqualLength(i)); } }
        for (i, c) in sketch.tangent_la.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::TangentLA(i)); } }
        for (i, c) in sketch.concentric.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Concentric(i)); } }
        for (i, c) in sketch.equal_radius.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::EqualRadius(i)); } }
        for (i, c) in sketch.tangent_aa.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::TangentAA(i)); } }
        for (i, c) in sketch.symmetry_ll.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Symmetry(i)); } }
        for (i, c) in sketch.symmetry_pp.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::SymmetryPP(i)); } }
        for (i, c) in sketch.symmetry_aa.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::SymmetryAA(i)); } }
        for (i, c) in sketch.line_p1_on_line.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP1OnLine, i)); } }
        for (i, c) in sketch.line_p2_on_line.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP2OnLine, i)); } }
        for (i, c) in sketch.coincident_arc_center.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcCenter, i)); } }
        for (i, c) in sketch.coincident_arc_start.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcStart, i)); } }
        for (i, c) in sketch.coincident_arc_end.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcEnd, i)); } }
        for (i, c) in sketch.coincident_lp1_arc_center.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP1ArcCenter, i)); } }
        for (i, c) in sketch.coincident_lp2_arc_center.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP2ArcCenter, i)); } }
        for (i, c) in sketch.coincident_lp1_arc_start.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP1ArcStart, i)); } }
        for (i, c) in sketch.coincident_lp2_arc_start.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP2ArcStart, i)); } }
        for (i, c) in sketch.coincident_lp1_arc_end.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP1ArcEnd, i)); } }
        for (i, c) in sketch.coincident_lp2_arc_end.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP2ArcEnd, i)); } }
        for (i, c) in sketch.coincident_arc_center_start.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcCenterStart, i)); } }
        for (i, c) in sketch.coincident_arc_center_end.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcCenterEnd, i)); } }
        for (i, c) in sketch.coincident_arc_start_center.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcStartCenter, i)); } }
        for (i, c) in sketch.coincident_arc_end_center.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcEndCenter, i)); } }
        for (i, c) in sketch.coincident_arc_start_start.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcStartStart, i)); } }
        for (i, c) in sketch.coincident_arc_start_end.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcStartEnd, i)); } }
        for (i, c) in sketch.coincident_arc_end_start.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcEndStart, i)); } }
        for (i, c) in sketch.coincident_arc_end_end.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::ArcEndEnd, i)); } }
        for (i, c) in sketch.line_p1_on_arc.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP1OnArc, i)); } }
        for (i, c) in sketch.line_p2_on_arc.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::Coincident(CoincidentKind::LP2OnArc, i)); } }
        return None;
    }

    if let Some((entity, flag)) = arael_sketch_solver::parse_flag_name(name) {
        for r in sketch.lines.refs() {
            if sketch.lines[r].name == entity {
                let l = &sketch.lines[r];
                return match flag {
                    'H' if l.constraints.horizontal => Some(ConstraintId::Horizontal(r)),
                    'V' if l.constraints.vertical => Some(ConstraintId::Vertical(r)),
                    _ => None,
                };
            }
        }
    }
    None
}
