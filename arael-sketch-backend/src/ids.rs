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
    ArcLineParallel(usize),
    ArcArcParallel(usize),
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

/// Format a ConstraintId as its user-visible name: `CL0H` /
/// `CL0V` for Horizontal/Vertical line flags, `C<nid>` for every
/// numbered constraint collection. Mirrors the name shown by `list`,
/// accepted by `delete`, and returned from `find_constraint_by_name`.
/// Command handlers use this to surface the IDs of constraints they
/// create or delete in their success messages.
pub fn constraint_id_name(sketch: &Sketch, id: ConstraintId) -> Option<String> {
    use arael_sketch_solver::format_flag_name;
    match id {
        ConstraintId::Horizontal(r) => Some(format_flag_name(&sketch.lines[r].name, 'H')),
        ConstraintId::Vertical(r) => Some(format_flag_name(&sketch.lines[r].name, 'V')),
        ConstraintId::Parallel(i) => Some(format!("C{}", sketch.parallel.get(i)?.nid)),
        ConstraintId::ArcLineParallel(i) => Some(format!("C{}", sketch.arc_line_parallel.get(i)?.nid)),
        ConstraintId::ArcArcParallel(i) => Some(format!("C{}", sketch.arc_arc_parallel.get(i)?.nid)),
        ConstraintId::Perpendicular(i) => Some(format!("C{}", sketch.perpendicular.get(i)?.nid)),
        ConstraintId::EqualLength(i) => Some(format!("C{}", sketch.equal_length.get(i)?.nid)),
        ConstraintId::EqualRadius(i) => Some(format!("C{}", sketch.equal_radius.get(i)?.nid)),
        ConstraintId::Concentric(i) => Some(format!("C{}", sketch.concentric.get(i)?.nid)),
        ConstraintId::TangentLA(i) => Some(format!("C{}", sketch.tangent_la.get(i)?.nid)),
        ConstraintId::TangentAA(i) => Some(format!("C{}", sketch.tangent_aa.get(i)?.nid)),
        ConstraintId::Collinear(i) => Some(format!("C{}", sketch.collinear.get(i)?.nid)),
        ConstraintId::Symmetry(i) => Some(format!("C{}", sketch.symmetry_ll.get(i)?.nid)),
        ConstraintId::SymmetryPP(i) => Some(format!("C{}", sketch.symmetry_pp.get(i)?.nid)),
        ConstraintId::SymmetryAA(i) => Some(format!("C{}", sketch.symmetry_aa.get(i)?.nid)),
        ConstraintId::Midpoint(kind, i) => {
            let nid = match kind {
                MidpointKind::Point => sketch.midpoint.get(i)?.nid,
                MidpointKind::LP1 => sketch.midpoint_lp1.get(i)?.nid,
                MidpointKind::LP2 => sketch.midpoint_lp2.get(i)?.nid,
                MidpointKind::ArcStart => sketch.midpoint_arc_start.get(i)?.nid,
                MidpointKind::ArcEnd => sketch.midpoint_arc_end.get(i)?.nid,
                MidpointKind::ArcPoint => sketch.midpoint_arc_point.get(i)?.nid,
                MidpointKind::LP1Arc => sketch.midpoint_lp1_arc.get(i)?.nid,
                MidpointKind::LP2Arc => sketch.midpoint_lp2_arc.get(i)?.nid,
                MidpointKind::ArcStartArc => sketch.midpoint_arc_start_arc.get(i)?.nid,
                MidpointKind::ArcEndArc => sketch.midpoint_arc_end_arc.get(i)?.nid,
            };
            Some(format!("C{}", nid))
        }
        ConstraintId::Coincident(kind, i) => {
            let nid = match kind {
                CoincidentKind::PP => sketch.coincident_pp.get(i)?.nid,
                CoincidentKind::LP1 => sketch.coincident_lp1.get(i)?.nid,
                CoincidentKind::LP2 => sketch.coincident_lp2.get(i)?.nid,
                CoincidentKind::LL11 => sketch.coincident_ll11.get(i)?.nid,
                CoincidentKind::LL12 => sketch.coincident_ll12.get(i)?.nid,
                CoincidentKind::LL21 => sketch.coincident_ll21.get(i)?.nid,
                CoincidentKind::LL22 => sketch.coincident_ll22.get(i)?.nid,
                CoincidentKind::PointOnLine => sketch.point_on_line.get(i)?.nid,
                CoincidentKind::PointOnArc => sketch.point_on_arc.get(i)?.nid,
                CoincidentKind::LP1OnLine => sketch.line_p1_on_line.get(i)?.nid,
                CoincidentKind::LP2OnLine => sketch.line_p2_on_line.get(i)?.nid,
                CoincidentKind::LP1OnArc => sketch.line_p1_on_arc.get(i)?.nid,
                CoincidentKind::LP2OnArc => sketch.line_p2_on_arc.get(i)?.nid,
                CoincidentKind::ArcCenter => sketch.coincident_arc_center.get(i)?.nid,
                CoincidentKind::ArcStart => sketch.coincident_arc_start.get(i)?.nid,
                CoincidentKind::ArcEnd => sketch.coincident_arc_end.get(i)?.nid,
                CoincidentKind::LP1ArcCenter => sketch.coincident_lp1_arc_center.get(i)?.nid,
                CoincidentKind::LP2ArcCenter => sketch.coincident_lp2_arc_center.get(i)?.nid,
                CoincidentKind::LP1ArcStart => sketch.coincident_lp1_arc_start.get(i)?.nid,
                CoincidentKind::LP2ArcStart => sketch.coincident_lp2_arc_start.get(i)?.nid,
                CoincidentKind::LP1ArcEnd => sketch.coincident_lp1_arc_end.get(i)?.nid,
                CoincidentKind::LP2ArcEnd => sketch.coincident_lp2_arc_end.get(i)?.nid,
                CoincidentKind::ArcCenterStart => sketch.coincident_arc_center_start.get(i)?.nid,
                CoincidentKind::ArcCenterEnd => sketch.coincident_arc_center_end.get(i)?.nid,
                CoincidentKind::ArcStartCenter => sketch.coincident_arc_start_center.get(i)?.nid,
                CoincidentKind::ArcEndCenter => sketch.coincident_arc_end_center.get(i)?.nid,
                CoincidentKind::ArcStartStart => sketch.coincident_arc_start_start.get(i)?.nid,
                CoincidentKind::ArcStartEnd => sketch.coincident_arc_start_end.get(i)?.nid,
                CoincidentKind::ArcEndStart => sketch.coincident_arc_end_start.get(i)?.nid,
                CoincidentKind::ArcEndEnd => sketch.coincident_arc_end_end.get(i)?.nid,
            };
            Some(format!("C{}", nid))
        }
        // Helper bridges are internal plumbing -- no user-visible name.
        ConstraintId::HelperBridge(_) => None,
    }
}

/// Look up a constraint by its auto-assigned name.
///
/// Numeric names `C<n>` scan every Vec-stored constraint that has a
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
        for (i, c) in sketch.arc_line_parallel.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::ArcLineParallel(i)); } }
        for (i, c) in sketch.arc_arc_parallel.iter().enumerate() { if c.nid == nid { return Some(ConstraintId::ArcArcParallel(i)); } }
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
