// Tool, selection, and constraint type enums for the sketch editor.

use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;

// What the user can grab and drag
#[derive(Clone, Copy, PartialEq)]
pub enum GrabTarget {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
    LineDrag(Ref<Line>),
    ArcDrag(Ref<Arc>),
}

// Selection -- what entity is selected for constraint application
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

// Constraint type for constraint mode
#[derive(Clone, Copy, PartialEq)]
pub enum ConstraintType {
    Horizontal,
    Vertical,
    Coincident,
    Parallel,
    Perpendicular,
    EqualLength,
    Tangent,
    Collinear,
    Midpoint,
    Symmetry,
    Lock,
    ToggleConstruction,
}

impl ConstraintType {
    #[allow(dead_code)]
    pub fn name(self) -> &'static str {
        match self {
            ConstraintType::Horizontal => "Horizontal",
            ConstraintType::Vertical => "Vertical",
            ConstraintType::Coincident => "Coincident",
            ConstraintType::Parallel => "Parallel",
            ConstraintType::Perpendicular => "Perpendicular",
            ConstraintType::EqualLength => "Equal",
            ConstraintType::Tangent => "Tangent",
            ConstraintType::Collinear => "Collinear",
            ConstraintType::Midpoint => "Midpoint",
            ConstraintType::Symmetry => "Symmetry",
            ConstraintType::Lock => "Lock",
            ConstraintType::ToggleConstruction => "Construction",
        }
    }
}

// Active tool
#[derive(Clone, Copy, PartialEq)]
pub enum Tool {
    Select,
    DrawPoint,
    DrawLine,
    DrawCircle,
    DrawArc,
    ConstraintMode(ConstraintType),
    Dimension,
}

// Delete target
#[derive(Clone, Copy)]
#[allow(dead_code)]
pub enum DeleteTarget {
    Point(Ref<Point>),
    Line(Ref<Line>),
    Arc(Ref<Arc>),
}

// In-progress line drawing state
pub struct LineDrawState {
    pub start: vect2d,
    // What the start point snapped to (for auto-coincident on completion)
    pub snap_start: Option<SnapTarget>,
    // True when this segment starts from the end of a just-placed segment
    // (line-tool chaining). Suppresses the start-snap marker in the preview
    // since that endpoint was just confirmed by the user's last click.
    pub chained: bool,
}

pub struct CircleDrawState {
    pub center: vect2d,
    pub snap_center: Option<SnapTarget>,
}

pub struct ArcDrawState {
    pub start: vect2d,
    pub snap_start: Option<SnapTarget>,
    pub end: Option<(vect2d, Option<SnapTarget>)>,  // None until second click
}

// Constraint identification (for selection and deletion)
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
    Point,          // Point at midpoint of line
    LP1,            // Line P1 at midpoint of another line
    LP2,            // Line P2 at midpoint of another line
    ArcStart,       // Arc start at midpoint of line
    ArcEnd,         // Arc end at midpoint of line
    ArcPoint,       // Point at angular midpoint of arc
    LP1Arc,         // Line P1 at angular midpoint of arc
    LP2Arc,         // Line P2 at angular midpoint of arc
    ArcStartArc,    // Arc start at angular midpoint of another arc
    ArcEndArc,      // Arc end at angular midpoint of another arc
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
    HelperBridge(Ref<Point>),  // helper point bridging two constraints
}

/// Look up a constraint by its auto-assigned name.
///
/// Numeric names "C<n>" scan every Vec-stored constraint that has a
/// ConstraintId variant and return the first match. Synthetic flag
/// names like "CL0H" / "CL3V" parse the entity reference and validate
/// the flag is currently set on that line.
///
/// Returns None for names that don't match either form, for numeric
/// names whose nid belongs to a constraint without a ConstraintId
/// variant (e.g., dimension-managed distance constraints — users reach
/// those through the backing dimension name), and for flag names whose
/// entity doesn't exist or doesn't have that flag set.
pub fn find_constraint_by_name(sketch: &Sketch, name: &str) -> Option<ConstraintId> {
    // Numeric form: C<nid>.
    if let Some(rest) = name.strip_prefix('C')
        && let Ok(nid) = rest.parse::<u32>() {
        if nid == 0 { return None; }
        // Scan ConstraintId-covered vectors in the same canonical order
        // as assign_constraint_names walks them.
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

    // Synthetic flag form: C<entity><flag>.
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

// Constraint symbol types (drawn with painter, not text)
#[derive(Clone, Copy)]
pub enum ConstraintSymbol {
    H,           // Horizontal
    V,           // Vertical
    Parallel,    // ||
    Perpendicular, // upside-down T
    Equal,       // =
    Tangent,     // T
    Collinear,   // diagonal line with gap
    Midpoint,    // triangle
    Symmetry,    // three parallel vertical lines |||
    Coincident,  // corner with dot
}

// A drawn constraint marker with screen position
pub struct ConstraintMarker {
    pub pos: eframe::egui::Pos2,
    pub symbol: ConstraintSymbol,
    pub id: ConstraintId,
}

// Which point on an arc we're referring to
#[derive(Clone, Copy)]
pub enum ArcPoint { Center, Start, End }

// What a point/endpoint snapped to
#[derive(Clone, Copy)]
pub enum SnapTarget {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    LineMidpoint(Ref<Line>),  // midpoint of line body; applies a MidpointLP1/LP2 constraint
    Line(Ref<Line>),  // on line body (not endpoint)
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
    ArcMidpoint(Ref<Arc>),  // midpoint of arc curve; applies a MidpointLP1/LP2Arc constraint
    ArcBody(Ref<Arc>),  // on arc/circle curve
}
