// Tool, drag and draw-state enums for the GUI. Constraint identity types
// (ConstraintId, CoincidentKind, MidpointKind, Selection,
// find_constraint_by_name) live in arael-sketch-backend::ids; this file
// holds the pieces that the egui rendering and input code own.

use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;
use eframe::egui;
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
    /// First pick the endpoint to place, then the reference endpoint:
    /// the first lies on the normal of the second's curve there.
    OnNormal,
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
            ConstraintType::OnNormal => "On normal",
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
    /// Three-click ellipse: center, end of the major axis (H/V
    /// snapped unless disabled), minor extent. Typed axis lengths
    /// become driving dimensions.
    DrawEllipse,
    DrawArc,
    DrawRect,
    Fillet,
    Chamfer,
    /// Break the hovered line/arc at the intersections bracketing the
    /// click; all pieces survive.
    Split,
    /// Same bracketing, but the clicked span is deleted.
    Trim,
    /// Uniform scale of clicked entities about a double-clicked
    /// center point, factor typed in the value overlay.
    Scale,
    /// Offset a sequence of lines and arcs: the set from clicks,
    /// double-click (walk) or a marquee, the parameters in the tool's
    /// own window (see offset_tool.rs).
    Offset,
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

// In-progress circle drawing. The value overlay opens with the center
// click and live-tracks the radius under the mouse (or a rim snap);
// typing takes it over. A typed radius becomes a driving dimension.
pub struct CircleDrawState {
    pub center: PlacedPoint,
    /// Radius: live until typed, then fixed.
    pub r: f64,
    pub typed_r: Option<String>,
    /// Unit direction center -> mouse; places the edge point of a
    /// typed radius.
    pub dir: vect2d,
    /// Snap target the rim passes through; offered while the radius
    /// is not typed.
    pub live_snap: Option<(vect2d, SnapTarget)>,
    /// Last cursor position (screen); the value input trails it.
    pub cursor: egui::Pos2,
}

// In-progress ellipse drawing. The value overlay opens with the
// center click and live-tracks the length under the mouse (semi-major
// while aiming the axis, semi-minor after the axis click); typing
// takes the value over and fixes it. Only typed values become
// driving dimensions on commit.
pub struct EllipseDrawState {
    pub center: PlacedPoint,
    /// Unit direction of the major axis; tracks the (H/V snapped)
    /// mouse until the second click fixes it.
    pub dir: vect2d,
    /// H/V snap in effect for the axis (drives the preview marker).
    pub hv: Option<bool>,
    /// True once the second click fixed the axis direction.
    pub axis_fixed: bool,
    /// Semi-major: live from the mouse until typed, then fixed.
    pub rx: f64,
    pub typed_rx: Option<String>,
    /// Absolute axis angle field (degrees): mirrors `dir` from the
    /// mouse until typed, then fixes `dir` and becomes an xangle dim.
    pub angle_text: String,
    pub typed_angle: Option<String>,
    /// Semi-minor: live once the axis is fixed, until typed.
    pub ry: f64,
    /// Which side of the axis the minor preview points (mouse side).
    pub ry_sign: f64,
    pub typed_ry: Option<String>,
    /// Last cursor position (screen). The value input trails it at
    /// an offset while aiming so it never sits under the click.
    pub cursor: egui::Pos2,
    /// Snap target under the mouse for the point being aimed (axis
    /// end, then minor extent); the rim passes through it. Not
    /// offered while that axis length is typed.
    pub live_snap: Option<(vect2d, SnapTarget)>,
    /// Snapped rim points confirmed by clicks; each becomes a helper
    /// point on the ellipse tied to its target at completion.
    pub rim_snaps: Vec<(vect2d, SnapTarget)>,
}

/// Entity a live tangent snap is measured against.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TangentHost {
    Line(Ref<Line>),
    Arc(Ref<Arc>),
}

// Three-click arc: start, end, then a point on the arc. Once the end
// is placed the value overlay live-tracks the radius; typing fixes it
// (the mouse then only picks the side and minor/major) and a typed
// radius becomes a driving dimension. The third point snaps to points
// (the arc passes through them) and, when an end is connected to a
// line or arc, to the arc tangent to it there.
pub struct ArcDrawState {
    pub start: PlacedPoint,
    pub end: Option<PlacedPoint>,  // None until second click
    /// Third point resolved this frame: mouse, point snap, tangent
    /// snap, or built from a typed radius.
    pub mid: vect2d,
    /// Radius: circumradius through start/end/mid until typed.
    pub r: f64,
    pub typed_r: Option<String>,
    /// Point snap for the third click; offered while r is not typed.
    pub live_snap: Option<(vect2d, SnapTarget)>,
    /// Tangent snap in effect: host and the connected end it applies at.
    pub tangent: Option<(TangentHost, vect2d)>,
    /// Which side of the chord the arc bulges to (typed radius): the
    /// mouse's last off-chord side.
    pub side: f64,
    /// Last cursor position (screen); the value input trails it.
    pub cursor: egui::Pos2,
}

// Rectangle drawing: user clicks two opposite corners, we build an
// axis-aligned rect as four lines with corner coincidents and H/V
// constraints on the sides. The value overlay opens with the first
// corner and live-tracks width and height (Tab between them); typed
// sides become driving length dims on the first horizontal and
// vertical side.
pub struct RectDrawState {
    pub corner: PlacedPoint,
    /// Width / height: live from the mouse until typed, then fixed.
    pub w: f64,
    pub typed_w: Option<String>,
    pub h: f64,
    /// Second field's text (the first is `dim_input`).
    pub height_text: String,
    pub typed_h: Option<String>,
    /// Quadrant of the opposite corner: signs of (dx, dy) from the
    /// mouse, so typed sides still follow the mouse's side.
    pub sx: f64,
    pub sy: f64,
    /// Snap target for the opposite corner; offered while neither
    /// side is typed.
    pub live_snap: Option<(vect2d, SnapTarget)>,
    /// Last cursor position (screen); the value input trails it.
    pub cursor: egui::Pos2,
}

// In-flight corner-op set (fillet or chamfer), created when the tool
// fires but still editable via the dim-input overlay until the user
// commits with Enter or reverts with Escape. The pre-op sketch
// snapshot lets Escape fully restore the sketch -- trim, new arc or
// chamfer line, coincidents, tangents, and dim(s) all undone -- and
// `history_cursor_before` matches the dropped actions so undo lines
// up. Every live edit restores to this snapshot and re-applies all
// pending corners through the typed backend engine
// (corner_ops::apply_corner_ops); keeping state declarative lets the
// radius edit and corner add/remove ripple through cleanly.
pub struct CornerOpPending {
    pub kind: arael_sketch_backend::corner_ops::CornerKind,
    pub pre_snapshot: std::vec::Vec<u8>,
    pub history_cursor_before: usize,
    /// Typed corners, one per corner op. First entry is the primary --
    /// its created dimension receives the user-typed value; the engine
    /// makes subsequent corners reference it by dim name so every
    /// corner tracks a single source.
    pub corners: std::vec::Vec<arael_sketch_backend::corner_ops::CornerSpec>,
    /// The primary corner's dimension after the last reapply; the
    /// dim-input overlay edits it live.
    pub primary_dim_did: Option<u32>,
    /// Last radius/distance token whose reapply actually produced
    /// a result. Empty / 0 / unparseable input falls back to this so
    /// the canvas keeps the most recent valid state while the user
    /// is mid-edit.
    pub last_valid_radius: String,
    /// Signature of the last reapply: radius token plus corners. Used
    /// to decide whether another reapply pass is needed when
    /// dim_input or `corners` changes.
    pub last_applied_sig: String,
}

/// Back-compat type alias while the rest of the code still uses
/// FilletPending. Cheap rename target.
pub type FilletPending = CornerOpPending;

/// In-flight scale session: entities come from the live selection and
/// the center from EditorApp::scale_center; this holds what a live
/// re-preview needs to restore and re-apply deterministically.
pub struct ScalePending {
    pub pre_snapshot: std::vec::Vec<u8>,
    pub history_cursor_before: usize,
    /// Last factor token whose reapply succeeded; empty/invalid input
    /// falls back to it so the canvas keeps the latest valid state.
    pub last_valid_factor: String,
    /// Signature (factor + sets + center) of the last reapply.
    pub last_applied_sig: String,
}

// Constraint symbol types (drawn with painter, not text)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
    OnNormal,    // base with a tick rising from its end, dot on top
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
