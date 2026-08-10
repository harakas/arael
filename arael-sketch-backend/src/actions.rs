//! Action enum and `apply()` for undo/redo in the sketch editor.
//!
//! # Three layers, one `Action` alphabet
//!
//! The sketch is always mutated through an [`Action`]. Three
//! independent layers build sequences of actions, but all of them
//! emit the same variants of this enum, which is why undo/redo,
//! serialization, and MCP replay work uniformly regardless of how
//! a sketch was built:
//!
//! 1. **Raw `Action::apply`.** The lowest layer. Each variant does
//!    one thing on the underlying [`Sketch`] and nothing else.
//!    `Action::AddLine { p1, p2 }` is literally
//!    `sketch.add_line(p1, p2)` -- no endpoint snapping, no
//!    coincidence, no solve. Stable semantics for programmatic
//!    callers (the `rectangle_actions` example is the canonical
//!    use). Actions are deliberately dumb; anything cleverer
//!    belongs in a higher layer.
//!
//! 2. **Command parser** (`commands::cmd_add_line` and siblings in
//!    [`crate::commands`]). Walks the parsed text arguments, snaps
//!    endpoints against existing entities within a tolerance, and
//!    emits extra coincidence actions (`ApplyCoincidentPP`,
//!    `ApplyCoincidentLL21`, `ApplyCoincidentArcStart`, etc.)
//!    alongside the bare `AddLine`. The "Coincident constraint
//!    already exists" dedup messages and the
//!    `[connected: L1.p1=L0.p2]` output seen in the
//!    `rectangle_commands` example come from this layer.
//!
//! 3. **GUI tools** (`EditorApp::apply_snap_coincident` and
//!    `apply_snap_coincident_arc` in the `arael-sketch` crate). The
//!    mouse resolves to a richer `SnapTarget` enum than a parser can
//!    concisely spell -- line body, line midpoint, arc body, arc
//!    midpoint, arc start/end/center -- so the GUI dispatches the
//!    full ten-variant snap taxonomy into the matching actions
//!    (`ApplyLineP1OnLine`, `ApplyMidpointLP1`, `ApplyMidpointLP1Arc`,
//!    `ApplyLineP1OnArc`, ...). It also layers in auto-perpendicular
//!    (`ApplyPerpendicular`) when the drawn line crosses a host line
//!    at a right angle, gated by `has_perp_conflict` so a redundant
//!    perp is never pushed. All of it goes into one `begin_group()`
//!    frame so a single Ctrl+Z undoes the line *and* its auto-snaps
//!    *and* any auto-perpendiculars as one unit.
//!
//! The command parser and the GUI are independent pipelines and do
//! duplicate the "add line then coincident-where-needed" shape,
//! but they converge on the same low-level `Action` alphabet. That
//! convergence is what makes [`History`](crate::History)'s
//! bincode-snapshot model work uniformly across GUI edits,
//! scripted batches, and MCP tool calls.

use arael::model::{Param, CrossBlock};
use arael::refs::Ref;
use arael::utils::{deg2rad, rad2deg, rad2rad};

/// Normalise a user-typed numeric dimension value into the canonical
/// range for its kind. Angular kinds (`ArcRotation`, `LineAngle`)
/// fold into `[-180, 180]` so repeated edits and save/load cycles do
/// not accumulate +/- 360 offsets -- `xangle L0 190` stores as -170,
/// matching the signed-angle convention of `atan2` that already
/// drives the displayed value. Other kinds pass through unchanged.
fn canonicalise_dim_value(kind: &DimensionKind, value: f64) -> f64 {
    match kind {
        DimensionKind::ArcRotation(_) | DimensionKind::LineAngle(_) => {
            rad2deg(rad2rad(deg2rad(value)))
        }
        _ => value,
    }
}
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::geometry::circumscribed_arc;

// Action log for undo/redo
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum Action {
    AddPoint { pos: vect2d },
    /// A hidden junction point for composite gestures: two or more
    /// constraints exec'd in the same group tie it down. Invisible to
    /// the user like every helper.
    AddHelperPoint { pos: vect2d },
    AddLine { p1: vect2d, p2: vect2d },
    ApplyHorizontal { lines: Vec<Ref<Line>> },
    ApplyVertical { lines: Vec<Ref<Line>> },
    ApplyCoincidentPP { a: Ref<Point>, b: Ref<Point> },
    ApplyCoincidentLL11 { a: Ref<Line>, b: Ref<Line> },
    ApplyCoincidentLL12 { a: Ref<Line>, b: Ref<Line> },
    ApplyCoincidentLL21 { a: Ref<Line>, b: Ref<Line> },
    ApplyCoincidentLL22 { a: Ref<Line>, b: Ref<Line> },
    ApplyCoincidentLP1 { line: Ref<Line>, point: Ref<Point> },
    ApplyCoincidentLP2 { line: Ref<Line>, point: Ref<Point> },
    ApplyParallel { a: Ref<Line>, b: Ref<Line> },
    ApplyArcLineParallel { arc: Ref<Arc>, line: Ref<Line> },
    ApplyArcArcParallel { a: Ref<Arc>, b: Ref<Arc> },
    ApplyPerpendicular { a: Ref<Line>, b: Ref<Line> },
    ApplyEqualLength { a: Ref<Line>, b: Ref<Line> },
    AddCircle { center: vect2d, edge: vect2d },
    AddEllipse { center: vect2d, rx: f64, ry: f64, rotation: f64 },
    AddArc { start: vect2d, end: vect2d, mid: vect2d },
    AddEllipticArc { center: vect2d, rx: f64, ry: f64, rotation: f64,
        start: f64, end: f64, ccw: bool },
    ApplyCoincidentArcCenter { point: Ref<Point>, arc: Ref<Arc> },
    ApplyCoincidentArcStart { point: Ref<Point>, arc: Ref<Arc> },
    ApplyCoincidentArcEnd { point: Ref<Point>, arc: Ref<Arc> },
    ApplyConcentric { a: Ref<Arc>, b: Ref<Arc> },
    // Line endpoint <-> Arc point (direct, no helper)
    ApplyCoincidentLP1ArcCenter { line: Ref<Line>, arc: Ref<Arc> },
    ApplyCoincidentLP2ArcCenter { line: Ref<Line>, arc: Ref<Arc> },
    ApplyCoincidentLP1ArcStart { line: Ref<Line>, arc: Ref<Arc> },
    ApplyCoincidentLP2ArcStart { line: Ref<Line>, arc: Ref<Arc> },
    ApplyCoincidentLP1ArcEnd { line: Ref<Line>, arc: Ref<Arc> },
    ApplyCoincidentLP2ArcEnd { line: Ref<Line>, arc: Ref<Arc> },
    // Arc-Arc endpoint (direct, no helper)
    ApplyCoincidentArcCenterStart { a: Ref<Arc>, b: Ref<Arc> },
    ApplyCoincidentArcCenterEnd { a: Ref<Arc>, b: Ref<Arc> },
    ApplyCoincidentArcStartCenter { a: Ref<Arc>, b: Ref<Arc> },
    ApplyCoincidentArcEndCenter { a: Ref<Arc>, b: Ref<Arc> },
    ApplyCoincidentArcStartStart { a: Ref<Arc>, b: Ref<Arc> },
    ApplyCoincidentArcStartEnd { a: Ref<Arc>, b: Ref<Arc> },
    ApplyCoincidentArcEndStart { a: Ref<Arc>, b: Ref<Arc> },
    ApplyCoincidentArcEndEnd { a: Ref<Arc>, b: Ref<Arc> },
    ApplyLineP1OnArc { line: Ref<Line>, arc: Ref<Arc> },
    ApplyLineP2OnArc { line: Ref<Line>, arc: Ref<Arc> },
    ApplyEqualRadius { a: Ref<Arc>, b: Ref<Arc> },
    ApplyTangentLA { line: Ref<Line>, arc: Ref<Arc> },
    ApplyTangentAA { a: Ref<Arc>, b: Ref<Arc> },
    ApplyPointOnLine { point: Ref<Point>, line: Ref<Line> },
    ApplyPointOnArc { point: Ref<Point>, arc: Ref<Arc> },
    /// Constrain any endpoint (including arc center/start/end) on a line.
    /// Creates helper point + bridge constraint internally if needed.
    ApplyEndpointOnLine { endpoint: DimensionEndpoint, line: Ref<Line> },
    /// Constrain any endpoint (including arc center/start/end) on an arc.
    /// Creates helper point + bridge constraint internally if needed.
    ApplyEndpointOnArc { endpoint: DimensionEndpoint, arc: Ref<Arc> },
    ApplyCollinear { a: Ref<Line>, b: Ref<Line> },
    ApplySymmetryLL { a: Ref<Line>, b: Ref<Line>, c: Ref<Line> },
    /// Endpoints, not points: helper bridges are created inside
    /// apply(), like every dimension endpoint.
    ApplySymmetryPP { a: DimensionEndpoint, line: Ref<Line>, c: DimensionEndpoint },
    ApplySymmetryAA { a: Ref<Arc>, line: Ref<Line>, c: Ref<Arc> },
    ApplyMidpoint { point: Ref<Point>, line: Ref<Line> },
    ApplyMidpointLP1 { line: Ref<Line>, target: Ref<Line> },
    ApplyMidpointLP2 { line: Ref<Line>, target: Ref<Line> },
    ApplyMidpointArcStart { arc: Ref<Arc>, line: Ref<Line> },
    ApplyMidpointArcEnd { arc: Ref<Arc>, line: Ref<Line> },
    ApplyMidpointArcPoint { point: Ref<Point>, arc: Ref<Arc> },
    ApplyMidpointLP1Arc { line: Ref<Line>, arc: Ref<Arc> },
    ApplyMidpointLP2Arc { line: Ref<Line>, arc: Ref<Arc> },
    ApplyMidpointArcStartArc { a: Ref<Arc>, b: Ref<Arc> },
    ApplyMidpointArcEndArc { a: Ref<Arc>, b: Ref<Arc> },
    ApplyLineP1OnLine { a: Ref<Line>, b: Ref<Line> },
    ApplyLineP2OnLine { a: Ref<Line>, b: Ref<Line> },
    LockPoint { point: Ref<Point>, pos: vect2d },
    UnlockPoint { point: Ref<Point> },
    LockLineP1 { line: Ref<Line>, pos: vect2d },
    UnlockLineP1 { line: Ref<Line> },
    LockLineP2 { line: Ref<Line>, pos: vect2d },
    UnlockLineP2 { line: Ref<Line> },
    LockArcCenter { arc: Ref<Arc>, pos: vect2d },
    UnlockArcCenter { arc: Ref<Arc> },
    DeletePoint { point: Ref<Point> },
    DeleteLine { line: Ref<Line> },
    ToggleConstructionLine { line: Ref<Line> },
    ToggleConstructionArc { arc: Ref<Arc> },
    SetStyleLine { line: Ref<Line>, style: LineStyle },
    SetStyleArc { arc: Ref<Arc>, style: LineStyle },
    SetQuietPoint { point: Ref<Point>, on: bool },
    SetQuietLine { line: Ref<Line>, on: bool },
    SetQuietArc { arc: Ref<Arc>, on: bool },
    SetConstructionLine { line: Ref<Line>, on: bool },
    SetConstructionArc { arc: Ref<Arc>, on: bool },
    DeleteArc { arc: Ref<Arc> },
    AddDimension {
        kind: DimensionKind,
        value: f64,
        expr: Option<String>,
        derived: bool,
        /// Inequality bound. When set, the dimension is a range dim:
        /// its residual is a barrier (zero inside the feasible region)
        /// and `value` tracks the current measured reading for display.
        /// Mutually exclusive with `expr` and `derived`.
        #[serde(default)]
        range: Option<RangeBound>,
    },
    UpdateDimension {
        did: u32,
        value: f64,
        expr: Option<String>,
        #[serde(default)]
        range: Option<RangeBound>,
    },
    RemoveDimension { did: u32 },
    /// Switch a dimension between driving and derived (reference) in
    /// place: name, did and placement survive, only the backing
    /// constraint comes or goes.
    ConvertDimension { did: u32, derived: bool, value: Option<f64> },
    MoveDimension { did: u32, offset: arael::vect::vect2d, text_along: f64 },
    AddUserParam { name: String, expr_str: String },
    UpdateUserParam { index: usize, name: String, expr_str: String },
    RemoveUserParam { index: usize },
    DeleteConstraint { id: crate::ids::ConstraintId },
    // Drag is non-deterministic; store full state after drag completes
    Drag { snapshot: Vec<u8> },
}

impl Action {
    /// Human-readable description of this action.
    pub fn describe(&self) -> String {
        match self {
            Action::AddPoint { .. } => "Add point".into(),
            Action::AddHelperPoint { .. } => "Add helper point".into(),
            Action::AddLine { .. } => "Add line".into(),
            Action::AddCircle { .. } => "Add circle".into(),
            Action::AddEllipse { .. } => "Add ellipse".into(),
            Action::AddArc { .. } => "Add arc".into(),
            Action::AddEllipticArc { .. } => "Add elliptic arc".into(),
            Action::ApplyHorizontal { lines } => format!("Horizontal ({})", lines.len()),
            Action::ApplyVertical { lines } => format!("Vertical ({})", lines.len()),
            Action::ApplyCoincidentPP { .. } => "Coincident PP".into(),
            Action::ApplyCoincidentLL11 { .. } | Action::ApplyCoincidentLL12 { .. } |
            Action::ApplyCoincidentLL21 { .. } | Action::ApplyCoincidentLL22 { .. } => "Coincident LL".into(),
            Action::ApplyCoincidentLP1 { .. } | Action::ApplyCoincidentLP2 { .. } => "Coincident LP".into(),
            Action::ApplyParallel { .. } => "Parallel".into(),
            Action::ApplyArcLineParallel { .. } => "Arc-Line parallel".into(),
            Action::ApplyArcArcParallel { .. } => "Arc-Arc parallel".into(),
            Action::ApplyPerpendicular { .. } => "Perpendicular".into(),
            Action::ApplyEqualLength { .. } => "Equal length".into(),
            Action::ApplyCollinear { .. } => "Collinear".into(),
            Action::ApplySymmetryLL { .. } | Action::ApplySymmetryPP { .. } | Action::ApplySymmetryAA { .. } => "Symmetry".into(),
            Action::ApplyMidpoint { .. } | Action::ApplyMidpointLP1 { .. } |
            Action::ApplyMidpointLP2 { .. } | Action::ApplyMidpointArcStart { .. } |
            Action::ApplyMidpointArcEnd { .. } |
            Action::ApplyMidpointArcPoint { .. } | Action::ApplyMidpointLP1Arc { .. } |
            Action::ApplyMidpointLP2Arc { .. } | Action::ApplyMidpointArcStartArc { .. } |
            Action::ApplyMidpointArcEndArc { .. } => "Midpoint".into(),
            Action::ApplyPointOnLine { .. } | Action::ApplyEndpointOnLine { .. } => "Point on line".into(),
            Action::ApplyPointOnArc { .. } | Action::ApplyEndpointOnArc { .. } => "Point on arc".into(),
            Action::ApplyLineP1OnLine { .. } | Action::ApplyLineP2OnLine { .. } => "Endpoint on line".into(),
            Action::ApplyLineP1OnArc { .. } | Action::ApplyLineP2OnArc { .. } => "Endpoint on arc".into(),
            Action::ApplyTangentLA { .. } => "Tangent LA".into(),
            Action::ApplyTangentAA { .. } => "Tangent AA".into(),
            Action::ApplyConcentric { .. } => "Concentric".into(),
            Action::ApplyEqualRadius { .. } => "Equal radius".into(),
            Action::ApplyCoincidentArcCenter { .. } | Action::ApplyCoincidentArcStart { .. } |
            Action::ApplyCoincidentArcEnd { .. } => "Coincident arc point".into(),
            Action::ApplyCoincidentLP1ArcCenter { .. } | Action::ApplyCoincidentLP2ArcCenter { .. } |
            Action::ApplyCoincidentLP1ArcStart { .. } | Action::ApplyCoincidentLP2ArcStart { .. } |
            Action::ApplyCoincidentLP1ArcEnd { .. } | Action::ApplyCoincidentLP2ArcEnd { .. } => "Coincident line-arc".into(),
            Action::ApplyCoincidentArcCenterStart { .. } | Action::ApplyCoincidentArcCenterEnd { .. } |
            Action::ApplyCoincidentArcStartCenter { .. } | Action::ApplyCoincidentArcEndCenter { .. } |
            Action::ApplyCoincidentArcStartStart { .. } | Action::ApplyCoincidentArcStartEnd { .. } |
            Action::ApplyCoincidentArcEndStart { .. } | Action::ApplyCoincidentArcEndEnd { .. } => "Coincident arc-arc".into(),
            Action::LockPoint { .. } => "Lock point".into(),
            Action::UnlockPoint { .. } => "Unlock point".into(),
            Action::LockLineP1 { .. } | Action::LockLineP2 { .. } => "Lock endpoint".into(),
            Action::UnlockLineP1 { .. } | Action::UnlockLineP2 { .. } => "Unlock endpoint".into(),
            Action::LockArcCenter { .. } => "Lock arc center".into(),
            Action::UnlockArcCenter { .. } => "Unlock arc center".into(),
            Action::DeletePoint { .. } => "Delete point".into(),
            Action::DeleteLine { .. } => "Delete line".into(),
            Action::DeleteArc { .. } => "Delete arc".into(),
            Action::ToggleConstructionLine { .. } | Action::ToggleConstructionArc { .. } => "Toggle construction".into(),
            Action::SetStyleLine { .. } | Action::SetStyleArc { .. } => "Set style".into(),
            Action::SetQuietPoint { .. } | Action::SetQuietLine { .. } | Action::SetQuietArc { .. } => "Set quiet".into(),
            Action::SetConstructionLine { .. } | Action::SetConstructionArc { .. } => "Set construction".into(),
            Action::AddDimension { kind, expr, .. } => {
                let kind_str = match kind {
                    DimensionKind::LineLength(_) => "length",
                    DimensionKind::ArcRadius(_) => "radius",
                    DimensionKind::ArcRadiusB(_) => "radius_b",
                    DimensionKind::ArcSweep(_) => "sweep",
                    DimensionKind::ArcRotation(_) => "xangle",
                    DimensionKind::PointPointDistance(_, _) => "distance",
                    DimensionKind::PointLineDistance(_, _) => "distance",
                    DimensionKind::Angle(_, _, _) => "angle",
                    DimensionKind::HDistance(_, _) => "hdistance",
                    DimensionKind::VDistance(_, _) => "vdistance",
                    DimensionKind::LineAngle(_) => "xangle",
                    DimensionKind::ConcentricDistance(_, _) => "concentric_distance",
                    DimensionKind::LineLineDistance(_, _) => "distance",
                };
                if expr.is_some() { format!("Add {} (expr)", kind_str) }
                else { format!("Add {}", kind_str) }
            }
            Action::UpdateDimension { .. } => "Update dimension".into(),
            Action::RemoveDimension { .. } => "Remove dimension".into(),
            Action::MoveDimension { .. } => "Move dimension".into(),
            Action::ConvertDimension { derived: true, .. } => "Set dimension derived".into(),
            Action::ConvertDimension { derived: false, .. } => "Set dimension driven".into(),
            Action::AddUserParam { name, .. } => format!("Add param {}", name),
            Action::UpdateUserParam { name, .. } => format!("Update param {}", name),
            Action::RemoveUserParam { .. } => "Remove param".into(),
            Action::DeleteConstraint { .. } => "Delete constraint".into(),
            Action::Drag { .. } => "Drag".into(),
        }
    }

    /// Returns true for constraint-adding actions that should be validated
    /// by the solver (cost check after application).
    pub fn is_constraint_action(&self) -> bool {
        matches!(self,
            Action::ApplyHorizontal { .. } | Action::ApplyVertical { .. } |
            Action::ApplyCoincidentPP { .. } |
            Action::ApplyCoincidentLL11 { .. } | Action::ApplyCoincidentLL12 { .. } |
            Action::ApplyCoincidentLL21 { .. } | Action::ApplyCoincidentLL22 { .. } |
            Action::ApplyCoincidentLP1 { .. } | Action::ApplyCoincidentLP2 { .. } |
            Action::ApplyParallel { .. } | Action::ApplyPerpendicular { .. } |
            Action::ApplyArcLineParallel { .. } | Action::ApplyArcArcParallel { .. } |
            Action::ApplyEqualLength { .. } |
            Action::ApplyCoincidentArcCenter { .. } | Action::ApplyCoincidentArcStart { .. } |
            Action::ApplyCoincidentArcEnd { .. } |
            Action::ApplyConcentric { .. } |
            Action::ApplyCoincidentLP1ArcCenter { .. } | Action::ApplyCoincidentLP2ArcCenter { .. } |
            Action::ApplyCoincidentLP1ArcStart { .. } | Action::ApplyCoincidentLP2ArcStart { .. } |
            Action::ApplyCoincidentLP1ArcEnd { .. } | Action::ApplyCoincidentLP2ArcEnd { .. } |
            Action::ApplyCoincidentArcCenterStart { .. } | Action::ApplyCoincidentArcCenterEnd { .. } |
            Action::ApplyCoincidentArcStartCenter { .. } | Action::ApplyCoincidentArcEndCenter { .. } |
            Action::ApplyCoincidentArcStartStart { .. } | Action::ApplyCoincidentArcStartEnd { .. } |
            Action::ApplyCoincidentArcEndStart { .. } | Action::ApplyCoincidentArcEndEnd { .. } |
            Action::ApplyLineP1OnArc { .. } | Action::ApplyLineP2OnArc { .. } |
            Action::ApplyEqualRadius { .. } |
            Action::ApplyTangentLA { .. } | Action::ApplyTangentAA { .. } |
            Action::ApplyCollinear { .. } | Action::ApplySymmetryLL { .. } | Action::ApplySymmetryPP { .. } | Action::ApplySymmetryAA { .. } |
            Action::ApplyMidpoint { .. } | Action::ApplyMidpointLP1 { .. } |
            Action::ApplyMidpointLP2 { .. } | Action::ApplyMidpointArcStart { .. } |
            Action::ApplyMidpointArcEnd { .. } |
            Action::ApplyMidpointArcPoint { .. } | Action::ApplyMidpointLP1Arc { .. } |
            Action::ApplyMidpointLP2Arc { .. } | Action::ApplyMidpointArcStartArc { .. } |
            Action::ApplyMidpointArcEndArc { .. } |
            Action::ApplyPointOnLine { .. } | Action::ApplyPointOnArc { .. } |
            Action::ApplyEndpointOnLine { .. } | Action::ApplyEndpointOnArc { .. } |
            Action::ApplyLineP1OnLine { .. } | Action::ApplyLineP2OnLine { .. } |
            Action::AddDimension { .. } | Action::UpdateDimension { .. }
        )
    }
}

// Resolve a DimensionEndpoint to a Ref<Point>, reusing an existing helper point
// if one is already constrained to the same entity endpoint, otherwise creating one.
pub fn resolve_dim_endpoint(sketch: &mut Sketch, ep: &DimensionEndpoint) -> Ref<Point> {
    match *ep {
        DimensionEndpoint::Point(r) => r,
        DimensionEndpoint::LineP1(r) => {
            if let Some(hp) = sketch.coincident_lp1.iter().find(|c| c.line == r).map(|c| c.point) {
                return hp;
            }
            let pos = sketch.lines[r].p1.value;
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_lp1.push(CoincidentLP1 { line: r, point: hp, nid: 0, cid: 0, hb: CrossBlock::new() });
            hp
        }
        DimensionEndpoint::LineP2(r) => {
            if let Some(hp) = sketch.coincident_lp2.iter().find(|c| c.line == r).map(|c| c.point) {
                return hp;
            }
            let pos = sketch.lines[r].p2.value;
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_lp2.push(CoincidentLP2 { line: r, point: hp, nid: 0, cid: 0, hb: CrossBlock::new() });
            hp
        }
        DimensionEndpoint::ArcCenter(r) => {
            if let Some(hp) = sketch.coincident_arc_center.iter().find(|c| c.arc == r).map(|c| c.point) {
                return hp;
            }
            let pos = sketch.arcs[r].center.value;
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_arc_center.push(CoincidentArcCenter { point: hp, arc: r, nid: 0, cid: 0, hb: CrossBlock::new() });
            hp
        }
        DimensionEndpoint::ArcStart(r) => {
            if let Some(hp) = sketch.coincident_arc_start.iter().find(|c| c.arc == r).map(|c| c.point) {
                return hp;
            }
            let pos = arc_start_pos_sketch(sketch, r);
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_arc_start.push(CoincidentArcStart { point: hp, arc: r, nid: 0, cid: 0, hb: CrossBlock::new() });
            hp
        }
        DimensionEndpoint::ArcEnd(r) => {
            if let Some(hp) = sketch.coincident_arc_end.iter().find(|c| c.arc == r).map(|c| c.point) {
                return hp;
            }
            let pos = arc_end_pos_sketch(sketch, r);
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_arc_end.push(CoincidentArcEnd { point: hp, arc: r, nid: 0, cid: 0, hb: CrossBlock::new() });
            hp
        }
    }
}

pub fn arc_start_pos_sketch(sketch: &Sketch, r: Ref<Arc>) -> vect2d {
    let a = &sketch.arcs[r];
    vect2d::new(
        a.center.value.x + a.radius.value * a.start_angle.value.cos(),
        a.center.value.y + a.radius.value * a.start_angle.value.sin(),
    )
}

pub fn arc_end_pos_sketch(sketch: &Sketch, r: Ref<Arc>) -> vect2d {
    let a = &sketch.arcs[r];
    vect2d::new(
        a.center.value.x + a.radius.value * a.end_angle.value.cos(),
        a.center.value.y + a.radius.value * a.end_angle.value.sin(),
    )
}

/// Get position of a DimensionEndpoint from a Sketch (read-only).
pub fn dim_endpoint_pos_sketch(sketch: &Sketch, ep: &DimensionEndpoint) -> vect2d {
    match *ep {
        DimensionEndpoint::Point(r) => sketch.points[r].pos.value,
        DimensionEndpoint::LineP1(r) => sketch.lines[r].p1.value,
        DimensionEndpoint::LineP2(r) => sketch.lines[r].p2.value,
        DimensionEndpoint::ArcCenter(r) => sketch.arcs[r].center.value,
        DimensionEndpoint::ArcStart(r) => arc_start_pos_sketch(sketch, r),
        DimensionEndpoint::ArcEnd(r) => arc_end_pos_sketch(sketch, r),
    }
}

/// Push the correct euclidean distance constraint for the given endpoint pair.
fn push_distance(sketch: &mut Sketch, a: &DimensionEndpoint, b: &DimensionEndpoint, distance: f64) {
    use DimensionEndpoint::*;
    match (a, b) {
        (Point(pa), Point(pb)) => { sketch.distance_pp.push(DistancePP { a: *pa, b: *pb, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        // Line-Line
        (LineP1(la), LineP1(lb)) => { sketch.distance_ll11.push(DistanceLL11 { a: *la, b: *lb, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP1(la), LineP2(lb)) => { sketch.distance_ll12.push(DistanceLL12 { a: *la, b: *lb, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP2(la), LineP1(lb)) => { sketch.distance_ll21.push(DistanceLL21 { a: *la, b: *lb, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP2(la), LineP2(lb)) => { sketch.distance_ll22.push(DistanceLL22 { a: *la, b: *lb, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        // Line-Point
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => { sketch.distance_lp1.push(DistanceLP1 { line: *l, point: *p, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => { sketch.distance_lp2.push(DistanceLP2 { line: *l, point: *p, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        // Arc-Point
        (ArcCenter(ar), Point(p)) | (Point(p), ArcCenter(ar)) => { sketch.distance_arc_center_p.push(DistanceArcCenterP { arc: *ar, point: *p, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(ar), Point(p)) | (Point(p), ArcStart(ar)) => { sketch.distance_arc_start_p.push(DistanceArcStartP { arc: *ar, point: *p, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(ar), Point(p)) | (Point(p), ArcEnd(ar)) => { sketch.distance_arc_end_p.push(DistanceArcEndP { arc: *ar, point: *p, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        // Arc-Line
        (ArcCenter(ar), LineP1(l)) | (LineP1(l), ArcCenter(ar)) => { sketch.distance_arc_center_l1.push(DistanceArcCenterL1 { arc: *ar, line: *l, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcCenter(ar), LineP2(l)) | (LineP2(l), ArcCenter(ar)) => { sketch.distance_arc_center_l2.push(DistanceArcCenterL2 { arc: *ar, line: *l, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(ar), LineP1(l)) | (LineP1(l), ArcStart(ar)) => { sketch.distance_arc_start_l1.push(DistanceArcStartL1 { arc: *ar, line: *l, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(ar), LineP2(l)) | (LineP2(l), ArcStart(ar)) => { sketch.distance_arc_start_l2.push(DistanceArcStartL2 { arc: *ar, line: *l, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(ar), LineP1(l)) | (LineP1(l), ArcEnd(ar)) => { sketch.distance_arc_end_l1.push(DistanceArcEndL1 { arc: *ar, line: *l, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(ar), LineP2(l)) | (LineP2(l), ArcEnd(ar)) => { sketch.distance_arc_end_l2.push(DistanceArcEndL2 { arc: *ar, line: *l, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        // Arc-Arc
        (ArcCenter(a), ArcCenter(b)) => { sketch.distance_aa_ce_ce.push(DistanceAACeCe { a: *a, b: *b, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcCenter(a), ArcStart(b)) => { sketch.distance_aa_ce_s.push(DistanceAACeS { a: *a, b: *b, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcCenter(a), ArcEnd(b)) => { sketch.distance_aa_ce_e.push(DistanceAACeE { a: *a, b: *b, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcCenter(b)) => { sketch.distance_aa_s_ce.push(DistanceAASCe { a: *a, b: *b, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcStart(b)) => { sketch.distance_aa_s_s.push(DistanceAASS { a: *a, b: *b, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcEnd(b)) => { sketch.distance_aa_s_e.push(DistanceAASE { a: *a, b: *b, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcCenter(b)) => { sketch.distance_aa_e_ce.push(DistanceAAECe { a: *a, b: *b, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcStart(b)) => { sketch.distance_aa_e_s.push(DistanceAAES { a: *a, b: *b, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcEnd(b)) => { sketch.distance_aa_e_e.push(DistanceAAEE { a: *a, b: *b, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
    }
}

/// Remove euclidean distance constraint matching the given endpoint pair.
fn remove_distance(sketch: &mut Sketch, a: &DimensionEndpoint, b: &DimensionEndpoint) {
    use DimensionEndpoint::*;
    match (a, b) {
        (Point(pa), Point(pb)) => { sketch.distance_pp.retain(|c| !(c.a == *pa && c.b == *pb)); }
        (LineP1(la), LineP1(lb)) => { sketch.distance_ll11.retain(|c| !(c.a == *la && c.b == *lb)); }
        (LineP1(la), LineP2(lb)) => { sketch.distance_ll12.retain(|c| !(c.a == *la && c.b == *lb)); }
        (LineP2(la), LineP1(lb)) => { sketch.distance_ll21.retain(|c| !(c.a == *la && c.b == *lb)); }
        (LineP2(la), LineP2(lb)) => { sketch.distance_ll22.retain(|c| !(c.a == *la && c.b == *lb)); }
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => { sketch.distance_lp1.retain(|c| !(c.line == *l && c.point == *p)); }
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => { sketch.distance_lp2.retain(|c| !(c.line == *l && c.point == *p)); }
        (ArcCenter(ar), Point(p)) | (Point(p), ArcCenter(ar)) => { sketch.distance_arc_center_p.retain(|c| !(c.arc == *ar && c.point == *p)); }
        (ArcStart(ar), Point(p)) | (Point(p), ArcStart(ar)) => { sketch.distance_arc_start_p.retain(|c| !(c.arc == *ar && c.point == *p)); }
        (ArcEnd(ar), Point(p)) | (Point(p), ArcEnd(ar)) => { sketch.distance_arc_end_p.retain(|c| !(c.arc == *ar && c.point == *p)); }
        (ArcCenter(ar), LineP1(l)) | (LineP1(l), ArcCenter(ar)) => { sketch.distance_arc_center_l1.retain(|c| !(c.arc == *ar && c.line == *l)); }
        (ArcCenter(ar), LineP2(l)) | (LineP2(l), ArcCenter(ar)) => { sketch.distance_arc_center_l2.retain(|c| !(c.arc == *ar && c.line == *l)); }
        (ArcStart(ar), LineP1(l)) | (LineP1(l), ArcStart(ar)) => { sketch.distance_arc_start_l1.retain(|c| !(c.arc == *ar && c.line == *l)); }
        (ArcStart(ar), LineP2(l)) | (LineP2(l), ArcStart(ar)) => { sketch.distance_arc_start_l2.retain(|c| !(c.arc == *ar && c.line == *l)); }
        (ArcEnd(ar), LineP1(l)) | (LineP1(l), ArcEnd(ar)) => { sketch.distance_arc_end_l1.retain(|c| !(c.arc == *ar && c.line == *l)); }
        (ArcEnd(ar), LineP2(l)) | (LineP2(l), ArcEnd(ar)) => { sketch.distance_arc_end_l2.retain(|c| !(c.arc == *ar && c.line == *l)); }
        (ArcCenter(a), ArcCenter(b)) => { sketch.distance_aa_ce_ce.retain(|c| !(c.a == *a && c.b == *b)); }
        (ArcCenter(a), ArcStart(b)) => { sketch.distance_aa_ce_s.retain(|c| !(c.a == *a && c.b == *b)); }
        (ArcCenter(a), ArcEnd(b)) => { sketch.distance_aa_ce_e.retain(|c| !(c.a == *a && c.b == *b)); }
        (ArcStart(a), ArcCenter(b)) => { sketch.distance_aa_s_ce.retain(|c| !(c.a == *a && c.b == *b)); }
        (ArcStart(a), ArcStart(b)) => { sketch.distance_aa_s_s.retain(|c| !(c.a == *a && c.b == *b)); }
        (ArcStart(a), ArcEnd(b)) => { sketch.distance_aa_s_e.retain(|c| !(c.a == *a && c.b == *b)); }
        (ArcEnd(a), ArcCenter(b)) => { sketch.distance_aa_e_ce.retain(|c| !(c.a == *a && c.b == *b)); }
        (ArcEnd(a), ArcStart(b)) => { sketch.distance_aa_e_s.retain(|c| !(c.a == *a && c.b == *b)); }
        (ArcEnd(a), ArcEnd(b)) => { sketch.distance_aa_e_e.retain(|c| !(c.a == *a && c.b == *b)); }
    }
}

/// Compute signed perpendicular distance from a point to a line.
fn compute_signed_pl(sketch: &Sketch, pt_pos: vect2d, line: Ref<Line>, value: f64) -> f64 {
    let l = &sketch.lines[line];
    let ldx = l.p2.value.x - l.p1.value.x;
    let ldy = l.p2.value.y - l.p1.value.y;
    let len = (ldx * ldx + ldy * ldy).sqrt();
    if len < 1e-12 { return value; }
    let sign = ((pt_pos.x - l.p1.value.x) * ldy - (pt_pos.y - l.p1.value.y) * ldx) / len;
    if sign >= 0.0 { value } else { -value }
}

/// Push the correct point-to-line distance constraint for the given endpoint.
fn push_distance_pl(sketch: &mut Sketch, pt: &DimensionEndpoint, line: Ref<Line>, distance: f64) {
    use DimensionEndpoint::*;
    match pt {
        Point(p) => { sketch.distance_pl.push(DistancePL { point: *p, line, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        LineP1(l) => { sketch.distance_lp1l.push(DistanceLP1L { a: *l, b: line, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        LineP2(l) => { sketch.distance_lp2l.push(DistanceLP2L { a: *l, b: line, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        ArcCenter(ar) => { sketch.distance_arc_center_l.push(DistanceArcCenterL { arc: *ar, line, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        ArcStart(ar) => { sketch.distance_arc_start_l.push(DistanceArcStartL { arc: *ar, line, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        ArcEnd(ar) => { sketch.distance_arc_end_l.push(DistanceArcEndL { arc: *ar, line, distance, nid: 0, cid: 0, hb: CrossBlock::new() }); }
    }
}

/// Remove point-to-line distance constraint matching the given endpoint.
fn remove_distance_pl(sketch: &mut Sketch, pt: &DimensionEndpoint, line: Ref<Line>) {
    use DimensionEndpoint::*;
    match pt {
        Point(p) => { sketch.distance_pl.retain(|c| !(c.point == *p && c.line == line)); }
        LineP1(l) => { sketch.distance_lp1l.retain(|c| !(c.a == *l && c.b == line)); }
        LineP2(l) => { sketch.distance_lp2l.retain(|c| !(c.a == *l && c.b == line)); }
        ArcCenter(ar) => { sketch.distance_arc_center_l.retain(|c| !(c.arc == *ar && c.line == line)); }
        ArcStart(ar) => { sketch.distance_arc_start_l.retain(|c| !(c.arc == *ar && c.line == line)); }
        ArcEnd(ar) => { sketch.distance_arc_end_l.retain(|c| !(c.arc == *ar && c.line == line)); }
    }
}

/// Install the numeric equality constraint backing a driving
/// dimension. Returns false when the kind's validation rejects the
/// geometry (missing arcs, non-parallel lines) -- the caller must not
/// create the dimension then.
fn push_numeric_dim_constraint(sketch: &mut Sketch, kind: &DimensionKind, value: &f64) -> bool {
    match kind {
        DimensionKind::LineLength(line) => {
            sketch.lines[*line].constraints.has_length = true;
            sketch.lines[*line].constraints.length = *value;
        }
        DimensionKind::PointPointDistance(a, b) => {
            push_distance(sketch, a, b, *value);
        }
        DimensionKind::PointLineDistance(pt, line) => {
            let pt_pos = dim_endpoint_pos_sketch(sketch, pt);
            let signed = compute_signed_pl(sketch, pt_pos, *line, *value);
            push_distance_pl(sketch, pt, *line, signed);
        }
        DimensionKind::ArcRadius(arc) => {
            sketch.arcs[*arc].constraints.has_target_radius = true;
            sketch.arcs[*arc].constraints.target_radius = *value;
        }
        DimensionKind::ArcRadiusB(arc) => {
            sketch.arcs[*arc].constraints.has_target_radius_b = true;
            sketch.arcs[*arc].constraints.target_radius_b = *value;
        }
        DimensionKind::ArcSweep(arc) => {
            sketch.arcs[*arc].constraints.sweep_sign = if sketch.arcs[*arc].ccw { 1.0 } else { -1.0 };
            sketch.arcs[*arc].constraints.has_target_sweep = true;
            sketch.arcs[*arc].constraints.target_sweep = deg2rad(*value);
        }
        DimensionKind::ArcRotation(arc) => {
            // Normalise user input into (-pi, pi] so repeated
            // `xangle EA0 200` edits don't accumulate value
            // drift across save/load and undo/redo cycles.
            let target = arael::utils::rad2rad(deg2rad(*value));
            sketch.arcs[*arc].constraints.has_target_rotation = true;
            sketch.arcs[*arc].constraints.target_rotation = target;
        }
        DimensionKind::Angle(a, b, supplement) => {
            // value is in degrees, constraint uses radians.
            // Compute target angle closest to current atan2 value
            // so the solver doesn't have to cross a discontinuity.
            let la = &sketch.lines[*a];
            let lb = &sketch.lines[*b];
            let dx1 = la.p2.value.x - la.p1.value.x;
            let dy1 = la.p2.value.y - la.p1.value.y;
            let dx2 = lb.p2.value.x - lb.p1.value.x;
            let dy2 = lb.p2.value.y - lb.p1.value.y;
            let current = (dx1 * dy2 - dy1 * dx2).atan2(dx1 * dx2 + dy1 * dy2);
            let mut target = deg2rad(*value);
            if *supplement { target = std::f64::consts::PI - target; }
            // Match sign to current atan2
            if current < 0.0 { target = -target; }
            sketch.angle.push(AngleConstraint {
                a: *a, b: *b, angle: target, nid: 0, cid: 0, hb: CrossBlock::new(),
            });
        }
        DimensionKind::HDistance(a, b) | DimensionKind::VDistance(a, b) => {
            let horizontal = matches!(kind, DimensionKind::HDistance(..));
            let pa_pos = dim_endpoint_pos_sketch(sketch, a);
            let pb_pos = dim_endpoint_pos_sketch(sketch, b);
            let current = if horizontal { pa_pos.x - pb_pos.x } else { pa_pos.y - pb_pos.y };
            let signed = if current >= 0.0 { *value } else { -*value };
            push_axis_distance(sketch, a, b, signed, horizontal);
        }
        DimensionKind::LineAngle(line) => {
            let target = deg2rad(*value);
            sketch.lines[*line].constraints.has_angle = true;
            sketch.lines[*line].constraints.target_angle = target;
        }
        DimensionKind::ConcentricDistance(a, b) => {
            // Validate: both arcs exist and are circular.
            // `DistanceConcentric`'s residual is
            // self-contained (it enforces center-coincidence
            // itself), so we no longer require a paired
            // `Concentric` -- the caller (cmd_distance / GUI
            // tool) still emits `ApplyConcentric` up front
            // for list-output visibility, but the action
            // doesn't depend on it.
            let valid_arcs = sketch.arcs.get(*a).is_some()
                && sketch.arcs.get(*b).is_some()
                && !sketch.arcs[*a].is_ellipse
                && !sketch.arcs[*b].is_ellipse;
            if !valid_arcs {
                return false;
            }
            let init_diff = sketch.arcs[*b].radius.value
                          - sketch.arcs[*a].radius.value;
            let sign = if init_diff >= 0.0 { 1.0 } else { -1.0 };
            sketch.distance_concentric.push(DistanceConcentric {
                a: *a, b: *b,
                sign,
                distance: value.abs(),
                nid: 0, cid: 0,
                hb: CrossBlock::new(),
            });
        }
        DimensionKind::LineLineDistance(a, b) => {
            // Validate: both lines exist and are currently
            // geometrically parallel. The dim's residual is a
            // point-to-line perpendicular distance; if the
            // lines ever rotate off-parallel its meaning
            // decays, but we don't enforce the pairing here --
            // the caller (cmd_distance / GUI) emits Parallel
            // explicitly when it matters. Keeping the action
            //'s guard purely geometric means the dim works
            // the instant you ask for it, without caring
            // whether parallelism is currently expressed via
            // an explicit Parallel, twin H/V flags, or a
            // chain of other constraints.
            let valid_lines = sketch.lines.get(*a).is_some()
                && sketch.lines.get(*b).is_some();
            let parallel_present = if !valid_lines { false }
            else {
                let la_line = &sketch.lines[*a];
                let lb_line = &sketch.lines[*b];
                let ax = la_line.p2.value.x - la_line.p1.value.x;
                let ay = la_line.p2.value.y - la_line.p1.value.y;
                let bx = lb_line.p2.value.x - lb_line.p1.value.x;
                let by = lb_line.p2.value.y - lb_line.p1.value.y;
                let alen = (ax * ax + ay * ay).sqrt();
                let blen = (bx * bx + by * by).sqrt();
                if alen < 1e-12 || blen < 1e-12 { false }
                else { (ax * by - ay * bx).abs() / (alen * blen) < 1e-6 }
            };
            if !valid_lines || !parallel_present {
                return false;
            }
            // Reuse the point-to-line distance constraint:
            // anchor line B's p1 to line A at the target gap.
            let pt = DimensionEndpoint::LineP1(*b);
            let pt_pos = dim_endpoint_pos_sketch(sketch, &pt);
            let signed = compute_signed_pl(sketch, pt_pos, *a, *value);
            push_distance_pl(sketch, &pt, *a, signed);
        }
    }
    true
}

/// Remove the underlying numeric equality constraint installed for a
/// dimension kind. Called when a dimension is removed, updated with a
/// new value, or switched to a range / expression (where the equality
/// constraint must go so the barrier / expr residual can drive the
/// parameter unopposed).
fn remove_numeric_dim_constraint(sketch: &mut Sketch, kind: &DimensionKind) {
    match *kind {
        DimensionKind::LineLength(line) => {
            if let Some(l) = sketch.lines.get_mut(line) {
                l.constraints.has_length = false;
            }
        }
        DimensionKind::ArcRadius(arc) => {
            if let Some(a) = sketch.arcs.get_mut(arc) {
                a.constraints.has_target_radius = false;
            }
        }
        DimensionKind::ArcRadiusB(arc) => {
            if let Some(a) = sketch.arcs.get_mut(arc) {
                a.constraints.has_target_radius_b = false;
            }
        }
        DimensionKind::ArcSweep(arc) => {
            if let Some(a) = sketch.arcs.get_mut(arc) {
                a.constraints.has_target_sweep = false;
            }
        }
        DimensionKind::ArcRotation(arc) => {
            if let Some(a) = sketch.arcs.get_mut(arc) {
                a.constraints.has_target_rotation = false;
            }
        }
        DimensionKind::PointPointDistance(ref a, ref b) => {
            remove_distance(sketch, a, b);
        }
        DimensionKind::PointLineDistance(ref pt, line) => {
            remove_distance_pl(sketch, pt, line);
        }
        DimensionKind::Angle(a, b, _) => {
            sketch.angle.retain(|c| !(c.a == a && c.b == b));
        }
        DimensionKind::HDistance(ref a, ref b) => {
            remove_axis_distance(sketch, a, b, true);
        }
        DimensionKind::VDistance(ref a, ref b) => {
            remove_axis_distance(sketch, a, b, false);
        }
        DimensionKind::LineAngle(line) => {
            if let Some(l) = sketch.lines.get_mut(line) {
                l.constraints.has_angle = false;
            }
        }
        DimensionKind::ConcentricDistance(a, b) => {
            sketch.distance_concentric.retain(|c|
                !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
        }
        DimensionKind::LineLineDistance(a, b) => {
            let pt = DimensionEndpoint::LineP1(b);
            remove_distance_pl(sketch, &pt, a);
        }
    }
}

/// Push the correct AxisDistance constraint for the given endpoint pair.
fn push_axis_distance(sketch: &mut Sketch, a: &DimensionEndpoint, b: &DimensionEndpoint, distance: f64, horizontal: bool) {
    use DimensionEndpoint::*;
    match (a, b) {
        (Point(pa), Point(pb)) => {
            if horizontal {
                sketch.hdistance_pp.push(HorizontalDistancePP { a: *pa, b: *pb, distance, nid: 0, cid: 0, hb: CrossBlock::new() });
            } else {
                sketch.vdistance_pp.push(VerticalDistancePP { a: *pa, b: *pb, distance, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
        }
        // Line-Line
        (LineP1(la), LineP1(lb)) => { sketch.axis_distance_ll11.push(AxisDistanceLL11 { a: *la, b: *lb, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP1(la), LineP2(lb)) => { sketch.axis_distance_ll12.push(AxisDistanceLL12 { a: *la, b: *lb, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP2(la), LineP1(lb)) => { sketch.axis_distance_ll21.push(AxisDistanceLL21 { a: *la, b: *lb, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP2(la), LineP2(lb)) => { sketch.axis_distance_ll22.push(AxisDistanceLL22 { a: *la, b: *lb, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        // Line-Point (constraint has line first, so negate distance if point is first arg)
        (LineP1(l), Point(p)) => { sketch.axis_distance_lp1.push(AxisDistanceLP1 { line: *l, point: *p, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (Point(p), LineP1(l)) => { sketch.axis_distance_lp1.push(AxisDistanceLP1 { line: *l, point: *p, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP2(l), Point(p)) => { sketch.axis_distance_lp2.push(AxisDistanceLP2 { line: *l, point: *p, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (Point(p), LineP2(l)) => { sketch.axis_distance_lp2.push(AxisDistanceLP2 { line: *l, point: *p, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        // Arc-Point (constraint has arc first)
        (ArcCenter(ar), Point(p)) => { sketch.axis_distance_arc_center_p.push(AxisDistanceArcCenterP { arc: *ar, point: *p, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (Point(p), ArcCenter(ar)) => { sketch.axis_distance_arc_center_p.push(AxisDistanceArcCenterP { arc: *ar, point: *p, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(ar), Point(p)) => { sketch.axis_distance_arc_start_p.push(AxisDistanceArcStartP { arc: *ar, point: *p, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (Point(p), ArcStart(ar)) => { sketch.axis_distance_arc_start_p.push(AxisDistanceArcStartP { arc: *ar, point: *p, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(ar), Point(p)) => { sketch.axis_distance_arc_end_p.push(AxisDistanceArcEndP { arc: *ar, point: *p, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (Point(p), ArcEnd(ar)) => { sketch.axis_distance_arc_end_p.push(AxisDistanceArcEndP { arc: *ar, point: *p, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        // Arc-Line (constraint has arc first)
        (ArcCenter(ar), LineP1(l)) => { sketch.axis_distance_arc_center_l1.push(AxisDistanceArcCenterL1 { arc: *ar, line: *l, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP1(l), ArcCenter(ar)) => { sketch.axis_distance_arc_center_l1.push(AxisDistanceArcCenterL1 { arc: *ar, line: *l, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcCenter(ar), LineP2(l)) => { sketch.axis_distance_arc_center_l2.push(AxisDistanceArcCenterL2 { arc: *ar, line: *l, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP2(l), ArcCenter(ar)) => { sketch.axis_distance_arc_center_l2.push(AxisDistanceArcCenterL2 { arc: *ar, line: *l, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(ar), LineP1(l)) => { sketch.axis_distance_arc_start_l1.push(AxisDistanceArcStartL1 { arc: *ar, line: *l, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP1(l), ArcStart(ar)) => { sketch.axis_distance_arc_start_l1.push(AxisDistanceArcStartL1 { arc: *ar, line: *l, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(ar), LineP2(l)) => { sketch.axis_distance_arc_start_l2.push(AxisDistanceArcStartL2 { arc: *ar, line: *l, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP2(l), ArcStart(ar)) => { sketch.axis_distance_arc_start_l2.push(AxisDistanceArcStartL2 { arc: *ar, line: *l, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(ar), LineP1(l)) => { sketch.axis_distance_arc_end_l1.push(AxisDistanceArcEndL1 { arc: *ar, line: *l, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP1(l), ArcEnd(ar)) => { sketch.axis_distance_arc_end_l1.push(AxisDistanceArcEndL1 { arc: *ar, line: *l, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(ar), LineP2(l)) => { sketch.axis_distance_arc_end_l2.push(AxisDistanceArcEndL2 { arc: *ar, line: *l, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (LineP2(l), ArcEnd(ar)) => { sketch.axis_distance_arc_end_l2.push(AxisDistanceArcEndL2 { arc: *ar, line: *l, distance: -distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        // Arc-Arc
        (ArcCenter(a), ArcCenter(b)) => { sketch.axis_distance_aa_ce_ce.push(AxisDistanceAACeCe { a: *a, b: *b, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcCenter(a), ArcStart(b)) => { sketch.axis_distance_aa_ce_s.push(AxisDistanceAACeS { a: *a, b: *b, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcCenter(a), ArcEnd(b)) => { sketch.axis_distance_aa_ce_e.push(AxisDistanceAACeE { a: *a, b: *b, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcCenter(b)) => { sketch.axis_distance_aa_s_ce.push(AxisDistanceAASCe { a: *a, b: *b, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcStart(b)) => { sketch.axis_distance_aa_s_s.push(AxisDistanceAASS { a: *a, b: *b, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcEnd(b)) => { sketch.axis_distance_aa_s_e.push(AxisDistanceAASE { a: *a, b: *b, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcCenter(b)) => { sketch.axis_distance_aa_e_ce.push(AxisDistanceAAECe { a: *a, b: *b, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcStart(b)) => { sketch.axis_distance_aa_e_s.push(AxisDistanceAAES { a: *a, b: *b, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcEnd(b)) => { sketch.axis_distance_aa_e_e.push(AxisDistanceAAEE { a: *a, b: *b, distance, horizontal, nid: 0, cid: 0, hb: CrossBlock::new() }); }
    }
}

/// Remove all axis distance constraints matching the given endpoint pair.
fn remove_axis_distance(sketch: &mut Sketch, a: &DimensionEndpoint, b: &DimensionEndpoint, horizontal: bool) {
    use DimensionEndpoint::*;
    match (a, b) {
        (Point(pa), Point(pb)) => {
            if horizontal {
                sketch.hdistance_pp.retain(|c| !(c.a == *pa && c.b == *pb));
            } else {
                sketch.vdistance_pp.retain(|c| !(c.a == *pa && c.b == *pb));
            }
        }
        (LineP1(la), LineP1(lb)) => { sketch.axis_distance_ll11.retain(|c| !(c.a == *la && c.b == *lb && c.horizontal == horizontal)); }
        (LineP1(la), LineP2(lb)) => { sketch.axis_distance_ll12.retain(|c| !(c.a == *la && c.b == *lb && c.horizontal == horizontal)); }
        (LineP2(la), LineP1(lb)) => { sketch.axis_distance_ll21.retain(|c| !(c.a == *la && c.b == *lb && c.horizontal == horizontal)); }
        (LineP2(la), LineP2(lb)) => { sketch.axis_distance_ll22.retain(|c| !(c.a == *la && c.b == *lb && c.horizontal == horizontal)); }
        (LineP1(l), Point(p)) => { sketch.axis_distance_lp1.retain(|c| !(c.line == *l && c.point == *p && c.horizontal == horizontal)); }
        (Point(p), LineP1(l)) => { sketch.axis_distance_lp1.retain(|c| !(c.line == *l && c.point == *p && c.horizontal == horizontal)); }
        (LineP2(l), Point(p)) => { sketch.axis_distance_lp2.retain(|c| !(c.line == *l && c.point == *p && c.horizontal == horizontal)); }
        (Point(p), LineP2(l)) => { sketch.axis_distance_lp2.retain(|c| !(c.line == *l && c.point == *p && c.horizontal == horizontal)); }
        (ArcCenter(ar), Point(p)) | (Point(p), ArcCenter(ar)) => { sketch.axis_distance_arc_center_p.retain(|c| !(c.arc == *ar && c.point == *p && c.horizontal == horizontal)); }
        (ArcStart(ar), Point(p)) | (Point(p), ArcStart(ar)) => { sketch.axis_distance_arc_start_p.retain(|c| !(c.arc == *ar && c.point == *p && c.horizontal == horizontal)); }
        (ArcEnd(ar), Point(p)) | (Point(p), ArcEnd(ar)) => { sketch.axis_distance_arc_end_p.retain(|c| !(c.arc == *ar && c.point == *p && c.horizontal == horizontal)); }
        (ArcCenter(ar), LineP1(l)) | (LineP1(l), ArcCenter(ar)) => { sketch.axis_distance_arc_center_l1.retain(|c| !(c.arc == *ar && c.line == *l && c.horizontal == horizontal)); }
        (ArcCenter(ar), LineP2(l)) | (LineP2(l), ArcCenter(ar)) => { sketch.axis_distance_arc_center_l2.retain(|c| !(c.arc == *ar && c.line == *l && c.horizontal == horizontal)); }
        (ArcStart(ar), LineP1(l)) | (LineP1(l), ArcStart(ar)) => { sketch.axis_distance_arc_start_l1.retain(|c| !(c.arc == *ar && c.line == *l && c.horizontal == horizontal)); }
        (ArcStart(ar), LineP2(l)) | (LineP2(l), ArcStart(ar)) => { sketch.axis_distance_arc_start_l2.retain(|c| !(c.arc == *ar && c.line == *l && c.horizontal == horizontal)); }
        (ArcEnd(ar), LineP1(l)) | (LineP1(l), ArcEnd(ar)) => { sketch.axis_distance_arc_end_l1.retain(|c| !(c.arc == *ar && c.line == *l && c.horizontal == horizontal)); }
        (ArcEnd(ar), LineP2(l)) | (LineP2(l), ArcEnd(ar)) => { sketch.axis_distance_arc_end_l2.retain(|c| !(c.arc == *ar && c.line == *l && c.horizontal == horizontal)); }
        (ArcCenter(a), ArcCenter(b)) => { sketch.axis_distance_aa_ce_ce.retain(|c| !(c.a == *a && c.b == *b && c.horizontal == horizontal)); }
        (ArcCenter(a), ArcStart(b)) => { sketch.axis_distance_aa_ce_s.retain(|c| !(c.a == *a && c.b == *b && c.horizontal == horizontal)); }
        (ArcCenter(a), ArcEnd(b)) => { sketch.axis_distance_aa_ce_e.retain(|c| !(c.a == *a && c.b == *b && c.horizontal == horizontal)); }
        (ArcStart(a), ArcCenter(b)) => { sketch.axis_distance_aa_s_ce.retain(|c| !(c.a == *a && c.b == *b && c.horizontal == horizontal)); }
        (ArcStart(a), ArcStart(b)) => { sketch.axis_distance_aa_s_s.retain(|c| !(c.a == *a && c.b == *b && c.horizontal == horizontal)); }
        (ArcStart(a), ArcEnd(b)) => { sketch.axis_distance_aa_s_e.retain(|c| !(c.a == *a && c.b == *b && c.horizontal == horizontal)); }
        (ArcEnd(a), ArcCenter(b)) => { sketch.axis_distance_aa_e_ce.retain(|c| !(c.a == *a && c.b == *b && c.horizontal == horizontal)); }
        (ArcEnd(a), ArcStart(b)) => { sketch.axis_distance_aa_e_s.retain(|c| !(c.a == *a && c.b == *b && c.horizontal == horizontal)); }
        (ArcEnd(a), ArcEnd(b)) => { sketch.axis_distance_aa_e_e.retain(|c| !(c.a == *a && c.b == *b && c.horizontal == horizontal)); }
    }
}

/// The entity an action added, for a caller that has to act on it next --
/// an auto-coincident against whatever the new point snapped to, say. The
/// arena chooses the slot, so a freed slot gets refilled and the new entity
/// is not the last one; only the value `add_*` returned identifies it.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Created {
    #[default]
    Nothing,
    Point(Ref<Point>),
    Line(Ref<Line>),
    Arc(Ref<Arc>),
}

impl Created {
    /// The added point, if this action added one.
    pub fn point(self) -> Option<Ref<Point>> {
        if let Created::Point(r) = self { Some(r) } else { None }
    }
    /// The added line, if this action added one.
    pub fn line(self) -> Option<Ref<Line>> {
        if let Created::Line(r) = self { Some(r) } else { None }
    }
    /// The added arc, if this action added one.
    pub fn arc(self) -> Option<Ref<Arc>> {
        if let Created::Arc(r) = self { Some(r) } else { None }
    }
}

impl Action {
    /// Apply the action, then solve. Returns what it added.
    pub fn apply(&self, sketch: &mut Sketch) -> Created {
        let (needs_expr_update, created) = self.apply_without_solve(sketch);
        sketch.assign_constraint_names();
        sketch.solve();
        if needs_expr_update {
            sketch.update_expr_dim_values();
        }
        created
    }

    /// Apply the action's mutation without solving. Returns whether
    /// `update_expr_dim_values` should be called after solving, and what the
    /// action added.
    pub fn apply_without_solve(&self, sketch: &mut Sketch) -> (bool, Created) {
        sketch.clear_cached_dof();
        let mut created = Created::Nothing;
        match self {
            Action::AddPoint { pos } => { created = Created::Point(sketch.add_point(*pos)); }
            Action::AddHelperPoint { pos } => { created = Created::Point(sketch.add_helper_point(*pos)); }
            Action::AddLine { p1, p2 } => { created = Created::Line(sketch.add_line(*p1, *p2)); }
            Action::AddCircle { center, edge } => {
                let r = ((edge.x - center.x).powi(2) + (edge.y - center.y).powi(2)).sqrt();
                created = Created::Arc(sketch.add_arc(*center, r, 0.0, std::f64::consts::TAU, true));
            }
            Action::AddEllipse { center, rx, ry, rotation } => {
                created = Created::Arc(sketch.add_ellipse(*center, *rx, *ry, *rotation, true));
            }
            Action::AddArc { start, end, mid, .. } => {
                if let Some((c, r, sa, ea, ccw)) = circumscribed_arc(*start, *end, *mid) {
                    created = Created::Arc(sketch.add_arc_with_dir(c, r, sa, ea, false, ccw));
                }
            }
            Action::AddEllipticArc { center, rx, ry, rotation, start, end, ccw } => {
                created = Created::Arc(sketch.add_elliptic_arc(*center, *rx, *ry, *rotation, *start, *end, *ccw));
            }
            Action::ApplyHorizontal { lines } => {
                for r in lines {
                    // Capture current (p2.x - p1.x) sign so the
                    // direction-preserving heaviside barrier knows which
                    // way the user oriented the line. See horizontal_dir
                    // on Line in arael-sketch-solver/src/entities.rs.
                    let dx = sketch.lines[*r].p2.value.x - sketch.lines[*r].p1.value.x;
                    sketch.lines[*r].constraints.h_dir_sign = if dx >= 0.0 { 1.0 } else { -1.0 };
                    sketch.lines[*r].constraints.horizontal = true;
                }
            }
            Action::ApplyVertical { lines } => {
                for r in lines {
                    let dy = sketch.lines[*r].p2.value.y - sketch.lines[*r].p1.value.y;
                    sketch.lines[*r].constraints.v_dir_sign = if dy >= 0.0 { 1.0 } else { -1.0 };
                    sketch.lines[*r].constraints.vertical = true;
                }
            }
            Action::ApplyCoincidentPP { a, b } => {
                sketch.coincident_pp.push(CoincidentPP { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLL11 { a, b } => {
                sketch.coincident_ll11.push(CoincidentLL11 { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLL12 { a, b } => {
                sketch.coincident_ll12.push(CoincidentLL12 { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLL21 { a, b } => {
                sketch.coincident_ll21.push(CoincidentLL21 { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLL22 { a, b } => {
                sketch.coincident_ll22.push(CoincidentLL22 { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP1 { line, point } => {
                sketch.coincident_lp1.push(CoincidentLP1 { line: *line, point: *point, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP2 { line, point } => {
                sketch.coincident_lp2.push(CoincidentLP2 { line: *line, point: *point, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyParallel { a, b } => {
                sketch.parallel.push(Parallel { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyArcLineParallel { arc, line } => {
                sketch.arc_line_parallel.push(ArcLineParallel {
                    arc: *arc, line: *line, nid: 0, cid: 0, hb: CrossBlock::new()
                });
            }
            Action::ApplyArcArcParallel { a, b } => {
                sketch.arc_arc_parallel.push(ArcArcParallel {
                    a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new()
                });
            }
            Action::ApplyPerpendicular { a, b } => {
                let la = &sketch.lines[*a];
                let lb = &sketch.lines[*b];
                let dx1 = la.p2.value.x - la.p1.value.x;
                let dy1 = la.p2.value.y - la.p1.value.y;
                let dx2 = lb.p2.value.x - lb.p1.value.x;
                let dy2 = lb.p2.value.y - lb.p1.value.y;
                let cross = dx1 * dy2 - dy1 * dx2;
                let dir_sign = if cross >= 0.0 { 1.0 } else { -1.0 };
                sketch.perpendicular.push(Perpendicular { a: *a, b: *b, dir_sign, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyEqualLength { a, b } => {
                sketch.equal_length.push(EqualLength { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcCenter { point, arc } => {
                sketch.coincident_arc_center.push(CoincidentArcCenter { point: *point, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcStart { point, arc } => {
                sketch.coincident_arc_start.push(CoincidentArcStart { point: *point, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcEnd { point, arc } => {
                sketch.coincident_arc_end.push(CoincidentArcEnd { point: *point, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyConcentric { a, b } => {
                sketch.concentric.push(Concentric { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP1ArcCenter { line, arc } => {
                sketch.coincident_lp1_arc_center.push(CoincidentLP1ArcCenter { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP2ArcCenter { line, arc } => {
                sketch.coincident_lp2_arc_center.push(CoincidentLP2ArcCenter { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP1ArcStart { line, arc } => {
                sketch.coincident_lp1_arc_start.push(CoincidentLP1ArcStart { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP2ArcStart { line, arc } => {
                sketch.coincident_lp2_arc_start.push(CoincidentLP2ArcStart { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP1ArcEnd { line, arc } => {
                sketch.coincident_lp1_arc_end.push(CoincidentLP1ArcEnd { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP2ArcEnd { line, arc } => {
                sketch.coincident_lp2_arc_end.push(CoincidentLP2ArcEnd { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcCenterStart { a, b } => {
                sketch.coincident_arc_center_start.push(CoincidentArcCenterStart { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcCenterEnd { a, b } => {
                sketch.coincident_arc_center_end.push(CoincidentArcCenterEnd { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcStartCenter { a, b } => {
                sketch.coincident_arc_start_center.push(CoincidentArcStartCenter { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcEndCenter { a, b } => {
                sketch.coincident_arc_end_center.push(CoincidentArcEndCenter { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcStartStart { a, b } => {
                sketch.coincident_arc_start_start.push(CoincidentArcStartStart { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcStartEnd { a, b } => {
                sketch.coincident_arc_start_end.push(CoincidentArcStartEnd { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcEndStart { a, b } => {
                sketch.coincident_arc_end_start.push(CoincidentArcEndStart { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcEndEnd { a, b } => {
                sketch.coincident_arc_end_end.push(CoincidentArcEndEnd { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyLineP1OnArc { line, arc } => {
                sketch.line_p1_on_arc.push(LineP1OnArc { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyLineP2OnArc { line, arc } => {
                sketch.line_p2_on_arc.push(LineP2OnArc { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyEqualRadius { a, b } => {
                sketch.equal_radius.push(EqualRadius { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyTangentLA { line, arc } => {
                let l = &sketch.lines[*line];
                let a = &sketch.arcs[*arc];
                // Detect which line endpoint meets which arc endpoint
                let snap = 1e-3;
                let sp = crate::geometry::arc_start_pos(a);
                let ep = crate::geometry::arc_end_pos(a);
                let near = |p: arael::vect::vect2d, q: arael::vect::vect2d|
                    (p.x - q.x).abs() < snap && (p.y - q.y).abs() < snap;
                let p1_arc_start = near(l.p1.value, sp);
                let p1_arc_end = near(l.p1.value, ep);
                let p2_arc_start = near(l.p2.value, sp);
                let p2_arc_end = near(l.p2.value, ep);
                // Compute sign for no-shared-endpoint formula
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                let dist = ((a.center.value.x - l.p1.value.x) * dy
                          - (a.center.value.y - l.p1.value.y) * dx) / len;
                let sign = if dist >= 0.0 { 1.0 } else { -1.0 };
                // Compute dir_sign from arc tangent vector at shared endpoint
                let shared = p1_arc_start || p1_arc_end || p2_arc_start || p2_arc_end;
                let dir_sign = if shared {
                    let angle = if p1_arc_start || p2_arc_start { a.start_angle.value } else { a.end_angle.value };
                    let cr = a.rotation.value.cos();
                    let sr = a.rotation.value.sin();
                    let ct = angle.cos();
                    let st = angle.sin();
                    let ax = a.center.value.x + a.radius.value * ct * cr - a.radius_b.value * st * sr;
                    let ay = a.center.value.y + a.radius.value * ct * sr + a.radius_b.value * st * cr;
                    let tx = -a.radius.value * st * cr - a.radius_b.value * ct * sr;
                    let ty = -a.radius.value * st * sr + a.radius_b.value * ct * cr;
                    let (ldx, ldy) = if p1_arc_start || p1_arc_end {
                        (l.p2.value.x - ax, l.p2.value.y - ay)
                    } else {
                        (l.p1.value.x - ax, l.p1.value.y - ay)
                    };
                    let dot = ldx * tx + ldy * ty;
                    if dot >= 0.0 { 1.0 } else { -1.0 }
                } else {
                    f64::NAN
                };
                sketch.tangent_la.push(TangentLA {
                    line: *line, arc: *arc, sign,
                    p1_arc_start, p1_arc_end, p2_arc_start, p2_arc_end,
                    dir_sign, nid: 0, cid: 0, hb: CrossBlock::new(),
                });
            }
            Action::ApplyTangentAA { a, b } => {
                let snap = 1e-3;
                let arc_a = &sketch.arcs[*a];
                let arc_b = &sketch.arcs[*b];
                let a_sp = crate::geometry::arc_start_pos(arc_a);
                let a_ep = crate::geometry::arc_end_pos(arc_a);
                let b_sp = crate::geometry::arc_start_pos(arc_b);
                let b_ep = crate::geometry::arc_end_pos(arc_b);
                let near = |p: vect2d, q: vect2d| (p.x - q.x).abs() < snap && (p.y - q.y).abs() < snap;
                let shared = if near(a_sp, b_sp) { SharedEndpoint::StartStart }
                    else if near(a_sp, b_ep) { SharedEndpoint::StartEnd }
                    else if near(a_ep, b_sp) { SharedEndpoint::EndStart }
                    else if near(a_ep, b_ep) { SharedEndpoint::EndEnd }
                    else { SharedEndpoint::None };
                sketch.tangent_aa.push(TangentAA { a: *a, b: *b, shared, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyPointOnLine { point, line } => {
                sketch.point_on_line.push(PointOnLine { point: *point, line: *line, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyPointOnArc { point, arc } => {
                sketch.point_on_arc.push(PointOnArc { point: *point, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyEndpointOnLine { endpoint, line } => {
                let point = resolve_dim_endpoint(sketch, endpoint);
                sketch.point_on_line.push(PointOnLine { point, line: *line, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyEndpointOnArc { endpoint, arc } => {
                let point = resolve_dim_endpoint(sketch, endpoint);
                sketch.point_on_arc.push(PointOnArc { point, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyCollinear { a, b } => {
                sketch.collinear.push(Collinear { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplySymmetryLL { a, b, c } => {
                sketch.symmetry_ll.push(SymmetryLL {
                    a: *a, b: *b, c: *c, nid: 0, cid: 0,
                    hb_ab: CrossBlock::new(), hb_ac: CrossBlock::new(), hb_bc: CrossBlock::new(),
                });
            }
            Action::ApplySymmetryPP { a, line, c } => {
                let pa = resolve_dim_endpoint(sketch, a);
                let pc = resolve_dim_endpoint(sketch, c);
                sketch.symmetry_pp.push(SymmetryPP {
                    a: pa, c: pc, line: *line, nid: 0, cid: 0,
                    hb_ac: CrossBlock::new(), hb_al: CrossBlock::new(), hb_cl: CrossBlock::new(),
                });
            }
            Action::ApplySymmetryAA { a, line, c } => {
                sketch.symmetry_aa.push(SymmetryAA {
                    a: *a, c: *c, line: *line, nid: 0, cid: 0,
                    hb_ac: CrossBlock::new(), hb_al: CrossBlock::new(), hb_cl: CrossBlock::new(),
                });
            }
            Action::ApplyMidpoint { point, line } => {
                sketch.midpoint.push(MidpointConstraint { point: *point, line: *line, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointLP1 { line, target } => {
                sketch.midpoint_lp1.push(MidpointLP1 { line: *line, target: *target, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointLP2 { line, target } => {
                sketch.midpoint_lp2.push(MidpointLP2 { line: *line, target: *target, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcStart { arc, line } => {
                sketch.midpoint_arc_start.push(MidpointArcStart { arc: *arc, line: *line, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcEnd { arc, line } => {
                sketch.midpoint_arc_end.push(MidpointArcEnd { arc: *arc, line: *line, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcPoint { point, arc } => {
                sketch.midpoint_arc_point.push(MidpointArcPoint { point: *point, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointLP1Arc { line, arc } => {
                sketch.midpoint_lp1_arc.push(MidpointLP1Arc { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointLP2Arc { line, arc } => {
                sketch.midpoint_lp2_arc.push(MidpointLP2Arc { line: *line, arc: *arc, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcStartArc { a, b } => {
                sketch.midpoint_arc_start_arc.push(MidpointArcStartArc { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcEndArc { a, b } => {
                sketch.midpoint_arc_end_arc.push(MidpointArcEndArc { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyLineP1OnLine { a, b } => {
                sketch.line_p1_on_line.push(LineP1OnLine { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::ApplyLineP2OnLine { a, b } => {
                sketch.line_p2_on_line.push(LineP2OnLine { a: *a, b: *b, nid: 0, cid: 0, hb: CrossBlock::new() });
            }
            Action::LockPoint { point, pos } => {
                let p = &mut sketch.points[*point];
                p.constraints.has_fix_x = true;
                p.constraints.fix_x = pos.x;
                p.constraints.has_fix_y = true;
                p.constraints.fix_y = pos.y;
            }
            Action::UnlockPoint { point } => {
                let p = &mut sketch.points[*point];
                p.constraints.has_fix_x = false;
                p.constraints.has_fix_y = false;
            }
            Action::LockLineP1 { line, pos } => {
                sketch.lines[*line].p1 = Param::fixed(*pos);
            }
            Action::UnlockLineP1 { line } => {
                let val = sketch.lines[*line].p1.value;
                sketch.lines[*line].p1 = Param::new(val);
            }
            Action::LockLineP2 { line, pos } => {
                sketch.lines[*line].p2 = Param::fixed(*pos);
            }
            Action::UnlockLineP2 { line } => {
                let val = sketch.lines[*line].p2.value;
                sketch.lines[*line].p2 = Param::new(val);
            }
            Action::LockArcCenter { arc, pos } => {
                sketch.arcs[*arc].center = Param::fixed(*pos);
            }
            Action::UnlockArcCenter { arc } => {
                let val = sketch.arcs[*arc].center.value;
                sketch.arcs[*arc].center = Param::new(val);
            }
            Action::DeletePoint { point } => {
                sketch.delete_point(*point);
            }
            Action::DeleteLine { line } => {
                sketch.delete_line(*line);
            }
            Action::ToggleConstructionLine { line } => {
                let l = &mut sketch.lines[*line];
                l.construction = !l.construction;
                l.style = if l.construction { LineStyle::DashDot } else { LineStyle::Solid };
            }
            Action::ToggleConstructionArc { arc } => {
                let a = &mut sketch.arcs[*arc];
                a.construction = !a.construction;
                a.style = if a.construction { LineStyle::DashDot } else { LineStyle::Solid };
            }
            Action::SetStyleLine { line, style } => {
                sketch.lines[*line].style = *style;
            }
            Action::SetStyleArc { arc, style } => {
                sketch.arcs[*arc].style = *style;
            }
            Action::SetQuietPoint { point, on } => {
                sketch.points[*point].quiet = *on;
            }
            Action::SetQuietLine { line, on } => {
                sketch.lines[*line].quiet = *on;
            }
            Action::SetQuietArc { arc, on } => {
                sketch.arcs[*arc].quiet = *on;
            }
            Action::SetConstructionLine { line, on } => {
                let l = &mut sketch.lines[*line];
                l.construction = *on;
                l.style = if *on { LineStyle::DashDot } else { LineStyle::Solid };
            }
            Action::SetConstructionArc { arc, on } => {
                let a = &mut sketch.arcs[*arc];
                a.construction = *on;
                a.style = if *on { LineStyle::DashDot } else { LineStyle::Solid };
            }
            Action::DeleteArc { arc } => {
                sketch.delete_arc(*arc);
            }
            Action::AddDimension { kind, value, expr, derived, range } => {
                // Normalise user input early so every branch below sees a
                // canonical angle for ArcRotation (range-, derived-,
                // expression- and numeric-dim paths alike). All other dim
                // kinds pass through unchanged.
                let value_ref = value;
                let normed_value = canonicalise_dim_value(kind, *value_ref);
                let value = &normed_value;
                // Range dimension: reject illegal combinations, then store
                // `kind + range` on the dimension. No underlying-constraint
                // push: `rebuild_expr_constraints` synthesises the barrier
                // residual on every solve from `dim.range`.
                if let Some(rb) = range {
                    if expr.is_some() || *derived { return (false, created); }
                    let name = format!("d{}", sketch.next_dimension_id);
                    sketch.next_dimension_id += 1;
                    sketch.dimensions.push(Dimension {
                        did: 0, // minted by assign_dimension_ids
                        kind: *kind, value: *value,
                        offset: vect2d::new(0.0, 1.0),
                        text_along: 0.0,
                        name,
                        expr_str: None,
                        broken: false,
                        derived: false,
                        range: Some(rb.clone()),
                    });
                    return (false, created);
                }
                // Expression dimension: delegate to add_expr_dimension
                if let Some(expr_str) = expr {
                    let _ = sketch.add_expr_dimension(*kind, expr_str,
                        vect2d::new(0.0, 1.0), 0.0);
                    if *derived
                        && let Some(d) = sketch.dimensions.last_mut() { d.derived = true; }
                    return (false, created);
                }
                let name = format!("d{}", sketch.next_dimension_id);
                sketch.next_dimension_id += 1;
                // Apply the numeric constraint (skip for derived dims)
                if !derived && !push_numeric_dim_constraint(sketch, kind, value) {
                    return (false, created);
                }
                sketch.dimensions.push(Dimension {
                    did: 0, // minted by assign_dimension_ids
                    kind: *kind, value: *value,
                    offset: vect2d::new(0.0, 1.0),
                    text_along: 0.0,
                    name,
                    expr_str: None,
                    broken: false, derived: *derived,
                    range: None,
                });
            }
            Action::UpdateDimension { did, value, expr, range } => {
                let Some(index) = sketch.dimension_index_by_did(*did) else { return (false, created); };
                // Same normalisation as AddDimension -- keeps the stored
                // value and the effective target in canonical range when
                // the user overwrites an xangle numeric dim.
                let normed_value = sketch.dimensions.get(index)
                    .map(|d| canonicalise_dim_value(&d.kind, *value))
                    .unwrap_or(*value);
                let value = &normed_value;
                // Range-dim update: rewrite the bound and re-measure.
                // No underlying-constraint bookkeeping (none was pushed).
                if let Some(rb) = range {
                    // Numeric -> range transition: drop any existing
                    // per-kind equality constraint so the barrier
                    // residual drives the parameter on its own. No-op
                    // when the dim was already range-typed (nothing
                    // was pushed) or derived.
                    let was_numeric_non_derived = sketch.dimensions.get(index)
                        .is_some_and(|d| d.expr_str.is_none() && !d.derived && d.range.is_none());
                    if was_numeric_non_derived {
                        let kind_copy = sketch.dimensions[index].kind;
                        remove_numeric_dim_constraint(sketch, &kind_copy);
                    }
                    if let Some(dim) = sketch.dimensions.get_mut(index) {
                        dim.range = Some(rb.clone());
                        dim.value = *value;
                        dim.expr_str = None;
                        dim.broken = false;
                    }
                    return (true, created);
                }
                // Range -> numeric (or expression) transition: clear the
                // `range` marker so `rebuild_expr_constraints` stops
                // synthesising the barrier residual. The numeric /
                // expression constraint is (re)built below.
                if let Some(dim) = sketch.dimensions.get_mut(index) {
                    dim.range = None;
                }
                // Remove old underlying constraint (only for numeric, non-derived dims)
                {
                    let dim = &sketch.dimensions[index];
                    let dim_kind = dim.kind;
                    let is_numeric_non_derived = dim.expr_str.is_none() && !dim.derived;
                    if is_numeric_non_derived {
                        match dim_kind {
                            DimensionKind::LineLength(line) => {
                                if let Some(l) = sketch.lines.get_mut(line) {
                                    l.constraints.has_length = false;
                                }
                            }
                            DimensionKind::ArcRadius(arc) => {
                                if let Some(a) = sketch.arcs.get_mut(arc) {
                                    a.constraints.has_target_radius = false;
                                }
                            }
                            DimensionKind::ArcRadiusB(arc) => {
                                if let Some(a) = sketch.arcs.get_mut(arc) {
                                    a.constraints.has_target_radius_b = false;
                                }
                            }
                            DimensionKind::ArcSweep(arc) => {
                                if let Some(a) = sketch.arcs.get_mut(arc) {
                                    a.constraints.has_target_sweep = false;
                                }
                            }
                            DimensionKind::ArcRotation(arc) => {
                                if let Some(a) = sketch.arcs.get_mut(arc) {
                                    a.constraints.has_target_rotation = false;
                                }
                            }
                            DimensionKind::PointPointDistance(ref a, ref b) => {
                                remove_distance(sketch, a, b);
                            }
                            DimensionKind::PointLineDistance(ref pt, line) => {
                                remove_distance_pl(sketch, pt, line);
                            }
                            DimensionKind::Angle(a, b, _) => {
                                sketch.angle.retain(|c| !(c.a == a && c.b == b));
                            }
                            DimensionKind::HDistance(ref a, ref b) => {
                                remove_axis_distance(sketch, a, b, true);
                            }
                            DimensionKind::VDistance(ref a, ref b) => {
                                remove_axis_distance(sketch, a, b, false);
                            }
                            DimensionKind::LineAngle(line) => {
                                if let Some(l) = sketch.lines.get_mut(line) {
                                    l.constraints.has_angle = false;
                                }
                            }
                            DimensionKind::ConcentricDistance(a, b) => {
                                sketch.distance_concentric.retain(|c|
                                    !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
                            }
                            DimensionKind::LineLineDistance(a, b) => {
                                // Drop the DistancePL anchored at line B's p1
                                // against line A (the shape we pushed on add).
                                let pt = DimensionEndpoint::LineP1(b);
                                remove_distance_pl(sketch, &pt, a);
                            }
                        }
                    }
                }
                // Update dimension in place (keeps name, kind, offset, text_along)
                let dim = &mut sketch.dimensions[index];
                dim.value = *value;
                dim.expr_str = expr.clone();
                // Add new underlying constraint (only for numeric, non-derived dims)
                if expr.is_none() && !dim.derived {
                    let kind = dim.kind;
                    let value = *value;
                    match kind {
                        DimensionKind::LineLength(line) => {
                            sketch.lines[line].constraints.has_length = true;
                            sketch.lines[line].constraints.length = value;
                        }
                        DimensionKind::ArcRadius(arc) => {
                            sketch.arcs[arc].constraints.has_target_radius = true;
                            sketch.arcs[arc].constraints.target_radius = value;
                        }
                        DimensionKind::ArcRadiusB(arc) => {
                            sketch.arcs[arc].constraints.has_target_radius_b = true;
                            sketch.arcs[arc].constraints.target_radius_b = value;
                        }
                        DimensionKind::ArcSweep(arc) => {
                            sketch.arcs[arc].constraints.sweep_sign = if sketch.arcs[arc].ccw { 1.0 } else { -1.0 };
                            sketch.arcs[arc].constraints.has_target_sweep = true;
                            sketch.arcs[arc].constraints.target_sweep = deg2rad(value);
                        }
                        DimensionKind::ArcRotation(arc) => {
                            let target = arael::utils::rad2rad(deg2rad(value));
                            sketch.arcs[arc].constraints.has_target_rotation = true;
                            sketch.arcs[arc].constraints.target_rotation = target;
                        }
                        DimensionKind::PointPointDistance(a, b) => {
                            push_distance(sketch, &a, &b, value);
                        }
                        DimensionKind::PointLineDistance(pt, line) => {
                            let pt_pos = dim_endpoint_pos_sketch(sketch, &pt);
                            let signed = compute_signed_pl(sketch, pt_pos, line, value);
                            push_distance_pl(sketch, &pt, line, signed);
                        }
                        DimensionKind::Angle(a, b, supplement) => {
                            let la = &sketch.lines[a];
                            let lb = &sketch.lines[b];
                            let dx1 = la.p2.value.x - la.p1.value.x;
                            let dy1 = la.p2.value.y - la.p1.value.y;
                            let dx2 = lb.p2.value.x - lb.p1.value.x;
                            let dy2 = lb.p2.value.y - lb.p1.value.y;
                            let current = (dx1 * dy2 - dy1 * dx2).atan2(dx1 * dx2 + dy1 * dy2);
                            let mut target = deg2rad(value);
                            if supplement { target = std::f64::consts::PI - target; }
                            if current < 0.0 { target = -target; }
                            sketch.angle.push(AngleConstraint {
                                a, b, angle: target, nid: 0, cid: 0, hb: CrossBlock::new(),
                            });
                        }
                        DimensionKind::HDistance(a, b) | DimensionKind::VDistance(a, b) => {
                            let horizontal = matches!(kind, DimensionKind::HDistance(..));
                            let pa_pos = dim_endpoint_pos_sketch(sketch, &a);
                            let pb_pos = dim_endpoint_pos_sketch(sketch, &b);
                            let current = if horizontal { pa_pos.x - pb_pos.x } else { pa_pos.y - pb_pos.y };
                            let signed = if current >= 0.0 { value } else { -value };
                            push_axis_distance(sketch, &a, &b, signed, horizontal);
                        }
                        DimensionKind::LineAngle(line) => {
                            let target = deg2rad(value);
                            sketch.lines[line].constraints.has_angle = true;
                            sketch.lines[line].constraints.target_angle = target;
                        }
                        DimensionKind::ConcentricDistance(a, b) => {
                            // Re-create with sign captured from current geometry
                            // (UpdateDimension is the user-visible "set new value"
                            // path; the sign tracks whichever arc is currently outer).
                            let init_diff = sketch.arcs[b].radius.value
                                          - sketch.arcs[a].radius.value;
                            let sign = if init_diff >= 0.0 { 1.0 } else { -1.0 };
                            sketch.distance_concentric.push(DistanceConcentric {
                                a, b,
                                sign,
                                distance: value.abs(),
                                nid: 0, cid: 0,
                                hb: CrossBlock::new(),
                            });
                        }
                        DimensionKind::LineLineDistance(a, b) => {
                            // Re-push the DistancePL with sign captured from
                            // current geometry, same shape as the add path.
                            let pt = DimensionEndpoint::LineP1(b);
                            let pt_pos = dim_endpoint_pos_sketch(sketch, &pt);
                            let signed = compute_signed_pl(sketch, pt_pos, a, value);
                            push_distance_pl(sketch, &pt, a, signed);
                        }
                    }
                }
                // Expression dims: rebuild_expr_constraints() in solve() handles it
            }
            Action::ConvertDimension { did, derived, value } => {
                let Some(index) = sketch.dimension_index_by_did(*did) else { return (false, created); };
                if sketch.dimensions[index].derived == *derived { return (false, created); }
                let kind = sketch.dimensions[index].kind;
                let is_expr = sketch.dimensions[index].expr_str.is_some();
                if *derived {
                    // Driving -> derived: the backing equality goes.
                    // Expression residuals are rebuilt from the flags on
                    // the next solve, so the flag flip covers them; a
                    // range cannot be derived and is dropped.
                    if !is_expr && sketch.dimensions[index].range.is_none() {
                        remove_numeric_dim_constraint(sketch, &kind);
                    }
                    let dim = &mut sketch.dimensions[index];
                    dim.derived = true;
                    dim.range = None;
                } else {
                    // Derived -> driving at the given (or measured) value.
                    {
                        let dim = &mut sketch.dimensions[index];
                        dim.derived = false;
                        if let Some(v) = value {
                            dim.value = canonicalise_dim_value(&kind, *v);
                        }
                    }
                    let v = sketch.dimensions[index].value;
                    if !is_expr && !push_numeric_dim_constraint(sketch, &kind, &v) {
                        // Geometry rejected the constraint: stay derived.
                        sketch.dimensions[index].derived = true;
                        return (false, created);
                    }
                }
                return (is_expr, created);
            }
            Action::MoveDimension { did, offset, text_along } => {
                if let Some(dim) = sketch.dimension_index_by_did(*did)
                    .and_then(|i| sketch.dimensions.get_mut(i)) {
                    dim.offset = *offset;
                    dim.text_along = *text_along;
                }
            }
            Action::RemoveDimension { did } => {
                if let Some(index) = sketch.dimension_index_by_did(*did) {
                    let dim = sketch.dimensions.remove(index);
                    // Expression dimension: remove the ExpressionConstraint
                    if dim.expr_str.is_some() {
                        let desc_prefix = format!("{} = ", dim.name);
                        sketch.expr_constraints.retain(|ec| !ec.description.starts_with(&desc_prefix));
                        return (false, created);  // skip normal constraint removal
                    }
                    // Remove the underlying constraint
                    match dim.kind {
                        DimensionKind::LineLength(line) => {
                            if let Some(l) = sketch.lines.get_mut(line) {
                                l.constraints.has_length = false;
                            }
                        }
                        DimensionKind::ArcRadius(arc) => {
                            if let Some(a) = sketch.arcs.get_mut(arc) {
                                a.constraints.has_target_radius = false;
                            }
                        }
                        DimensionKind::ArcRadiusB(arc) => {
                            if let Some(a) = sketch.arcs.get_mut(arc) {
                                a.constraints.has_target_radius_b = false;
                            }
                        }
                        DimensionKind::ArcSweep(arc) => {
                            if let Some(a) = sketch.arcs.get_mut(arc) {
                                a.constraints.has_target_sweep = false;
                            }
                        }
                        DimensionKind::ArcRotation(arc) => {
                            if let Some(a) = sketch.arcs.get_mut(arc) {
                                a.constraints.has_target_rotation = false;
                            }
                        }
                        DimensionKind::PointPointDistance(a, b) => {
                            remove_distance(sketch, &a, &b);
                        }
                        DimensionKind::PointLineDistance(pt, line) => {
                            remove_distance_pl(sketch, &pt, line);
                        }
                        DimensionKind::Angle(a, b, _) => {
                            sketch.angle.retain(|c| !(c.a == a && c.b == b));
                        }
                        DimensionKind::HDistance(a, b) => {
                            remove_axis_distance(sketch, &a, &b, true);
                        }
                        DimensionKind::VDistance(a, b) => {
                            remove_axis_distance(sketch, &a, &b, false);
                        }
                        DimensionKind::LineAngle(line) => {
                            if let Some(l) = sketch.lines.get_mut(line) {
                                l.constraints.has_angle = false;
                            }
                        }
                        DimensionKind::ConcentricDistance(a, b) => {
                            sketch.distance_concentric.retain(|c|
                                !((c.a == a && c.b == b) || (c.a == b && c.b == a)));
                        }
                        DimensionKind::LineLineDistance(a, b) => {
                            let pt = DimensionEndpoint::LineP1(b);
                            remove_distance_pl(sketch, &pt, a);
                        }
                    }
                    sketch.cleanup_helper_points();
                }
            }
            Action::AddUserParam { name, expr_str } => {
                let value = expr_str.trim().parse::<f64>().unwrap_or(0.0);
                sketch.user_params.push(UserParam {
                    name: name.clone(), expr_str: expr_str.clone(),
                    value, broken: false,
                });
                return (true, created);
            }
            Action::UpdateUserParam { index, name, expr_str } => {
                if *index < sketch.user_params.len() {
                    let old_name = sketch.user_params[*index].name.clone();
                    sketch.user_params[*index].name = name.clone();
                    sketch.user_params[*index].expr_str = expr_str.clone();
                    if let Ok(v) = expr_str.trim().parse::<f64>() {
                        sketch.user_params[*index].value = v;
                    }
                    // Propagate name change to expressions that reference the old name
                    if old_name != *name {
                        for p in &mut sketch.user_params {
                            if let Ok(parsed) = arael_sym::parse(&p.expr_str)
                                && parsed.symbols().contains(&old_name) {
                                    let replaced = parsed.subs(&old_name, &arael_sym::symbol(name));
                                    p.expr_str = format!("{}", replaced);
                                }
                        }
                        for d in &mut sketch.dimensions {
                            if let Some(ref es) = d.expr_str
                                && let Ok(parsed) = arael_sym::parse(es)
                                    && parsed.symbols().contains(&old_name) {
                                        let replaced = parsed.subs(&old_name, &arael_sym::symbol(name));
                                        d.expr_str = Some(format!("{}", replaced));
                                    }
                        }
                    }
                    return (true, created);
                }
            }
            Action::RemoveUserParam { index } => {
                if *index < sketch.user_params.len() {
                    sketch.user_params.remove(*index);
                    return (true, created);
                }
            }
            Action::DeleteConstraint { id } => {
                use crate::ids::ConstraintId;
                match id {
                    ConstraintId::Horizontal(r) => {
                        if let Some(l) = sketch.lines.get_mut(*r) { l.constraints.horizontal = false; }
                    }
                    ConstraintId::Vertical(r) => {
                        if let Some(l) = sketch.lines.get_mut(*r) { l.constraints.vertical = false; }
                    }
                    ConstraintId::Numbered(nid) => {
                        // Parallel carries a cascade: a LineLineDistance
                        // dimension is backed by it (plus a DistancePL),
                        // and goes with it.
                        let parallel_pair = sketch.parallel.iter()
                            .find(|c| c.nid == *nid).map(|c| (c.a, c.b));
                        if sketch.remove_constraint_by_nid(*nid)
                            && let Some((a, b)) = parallel_pair {
                                let pt = DimensionEndpoint::LineP1(b);
                                remove_distance_pl(sketch, &pt, a);
                                let pt = DimensionEndpoint::LineP1(a);
                                remove_distance_pl(sketch, &pt, b);
                                sketch.dimensions.retain(|d| !d.kind.references_parallel_pair(a, b));
                        }
                        sketch.cleanup_helper_points();
                    }
                    ConstraintId::HelperBridge(pt) => {
                        // Only ever a helper; a stale or repurposed ref
                        // must not delete a real point.
                        if sketch.points.get(*pt).is_some_and(|p| p.helper) {
                            sketch.delete_point(*pt);
                        }
                    }
                }
            }
            Action::Drag { snapshot } => {
                *sketch = bincode::deserialize(snapshot).unwrap();
            }
        }
        (false, created)
    }
}
