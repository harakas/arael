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
