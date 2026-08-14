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
pub use arael_sketch_backend::ids::{ConstraintId, Selection};
// find_constraint_by_name is used via arael_sketch_backend::find_constraint_by_name in main.rs
#[allow(unused_imports)]
pub use arael_sketch_backend::ids::find_constraint_by_name;

// What the user can grab and drag
#[derive(Clone, Copy, PartialEq, Debug)]
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
#[derive(Clone, Copy, PartialEq, Debug)]
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
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Tool {
    Select,
    DrawPoint,
    DrawLine,
    DrawCircle,
    DrawArc,
    DrawRect,
    Fillet,
    Chamfer,
    /// Break the hovered line/arc at the intersections bracketing the
    /// click; all pieces survive.
    Split,
    /// Same bracketing, but the clicked span is deleted.
    Trim,
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

// A clicked anchor during tool drawing: the position and what it
// snapped to (for auto-coincident on completion).
#[derive(Clone, Copy)]
pub struct PlacedPoint {
    pub pos: vect2d,
    pub snap: Option<SnapTarget>,
}

// In-progress line drawing state
pub struct LineDrawState {
    pub start: PlacedPoint,
    // True when this segment starts from the end of a just-placed segment
    // (line-tool chaining). Suppresses the start-snap marker in the preview
    // since that endpoint was just confirmed by the user's last click.
    pub chained: bool,
}

pub struct CircleDrawState {
    pub center: PlacedPoint,
}

pub struct ArcDrawState {
    pub start: PlacedPoint,
    pub end: Option<PlacedPoint>,  // None until second click
}

// Rectangle drawing: user clicks two opposite corners, we build an
// axis-aligned rect as four lines with corner coincidents and H/V
// constraints on the sides.
pub struct RectDrawState {
    pub corner: PlacedPoint,
}

// In-flight corner-op set (fillet or chamfer), created when the tool
// fires but still editable via the dim-input overlay until the user
// commits with Enter or reverts with Escape. The pre-op sketch
// snapshot lets Escape fully restore the sketch -- trim, new arc or
// chamfer line, coincidents, tangents, and dim(s) all undone -- and
// `history_cursor_before` matches the dropped actions so undo lines
// up. Every live edit restores to this snapshot and re-applies all
// pending corners; keeping state declarative lets the radius edit
// and corner add/remove ripple through cleanly.
pub struct CornerOpPending {
    /// Backend command name: "fillet" or "chamfer". The reapply loop
    /// runs `<command> <corner_arg> <radius>` per corner, so the
    /// two tools share the same live-preview plumbing.
    pub command: &'static str,
    pub pre_snapshot: std::vec::Vec<u8>,
    pub history_cursor_before: usize,
    /// Corner argument strings, one per corner: `"L0 L1"` or
    /// `"L0.p2"`. First entry is the primary -- its created dimension
    /// receives the user-typed value. Subsequent entries reference
    /// the primary dim by name (`d<N>`) so every corner tracks it.
    pub corners: std::vec::Vec<String>,
    /// Last radius/distance token whose reapply actually produced
    /// a result. Empty / 0 / unparseable input falls back to this so
    /// the canvas keeps the most recent valid state while the user
    /// is mid-edit.
    pub last_valid_radius: String,
    /// Signature of the last successful reapply: radius token plus
    /// corner count. Used to decide whether another reapply pass is
    /// needed when dim_input or `corners` changes.
    pub last_applied_sig: String,
}

/// Back-compat type alias while the rest of the code still uses
/// FilletPending. Cheap rename target.
pub type FilletPending = CornerOpPending;

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
