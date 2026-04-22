// Tool, drag and draw-state enums for the GUI. Constraint identity types
// (ConstraintId, CoincidentKind, MidpointKind, Selection,
// find_constraint_by_name) live in arael-sketch-backend::ids; this file
// holds the pieces that the egui rendering and input code own.

use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;
// Re-export backend identity types so existing `crate::tools::*` imports
// across main.rs/drawing.rs/app_update.rs continue to resolve after the
// split into arael-sketch-backend.
pub use arael_sketch_backend::ids::{ConstraintId, CoincidentKind, MidpointKind, Selection};
// find_constraint_by_name is used via arael_sketch_backend::find_constraint_by_name in main.rs
#[allow(unused_imports)]
pub use arael_sketch_backend::ids::find_constraint_by_name;

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
    DrawRect,
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

// Rectangle drawing: user clicks two opposite corners, we build an
// axis-aligned rect as four lines with corner coincidents and H/V
// constraints on the sides.
pub struct RectDrawState {
    pub corner: vect2d,
    pub snap_corner: Option<SnapTarget>,
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
