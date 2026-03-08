//! 2D parametric constraint-based sketch solver and editor.
//!
//! Work in progress -- a parametric CAD sketching tool built on the arael
//! optimization framework. Draw geometry, apply constraints, and the
//! solver keeps everything consistent in real time.
//!
//! # Entities
//!
//! Three entity types, each owning its own parameters:
//!
//! - [`Point`] -- 2D position (x, y)
//! - [`Line`] -- two endpoints p1, p2 (4 params), plus optional
//!   horizontal/vertical/length constraints
//! - [`Arc`] -- center, radius, start/end angle (5 params), optionally
//!   closed (full circle)
//!
//! Shared geometry (e.g. two lines meeting at a point) is enforced via
//! coincident constraints, not shared references.
//!
//! # Constraints
//!
//! Over 40 constraint types including: coincident (point-point, line-line,
//! point-on-line, point-on-arc), parallel, perpendicular, tangent,
//! equal length/radius, distance, horizontal/vertical distance, and more.
//! All constraints are symbolically differentiated at compile time.
//!
//! # Solving
//!
//! The [`Sketch::solve()`] method runs Levenberg-Marquardt optimization with
//! drift regularization. Locking: use `Param::fixed(value)` to pin
//! individual parameters. The solver skips fixed params entirely.
//!
//! # Editor
//!
//! An interactive editor (see `examples/editor.rs`) provides drawing tools,
//! constraint application, dimension annotations, undo/redo, and file I/O.
//! Runs natively and in the browser via WebAssembly.
//!
//! [![Sketch Editor](https://raw.githubusercontent.com/harakas/arael/refs/heads/master/docs/sketch.png)](https://sketch.mare.ee/)
//!
//! [Try it in the browser](https://sketch.mare.ee/) |
//! [Source](https://github.com/harakas/arael/tree/master/arael-sketch)
//!
//! ```ignore
//! // Native:
//! cargo run -r -p arael-sketch --example editor --features editor
//!
//! // Browser (WASM):
//! cd arael-sketch
//! trunk build --release --example editor --features editor
//! python3 -m http.server -d dist 8080
//! ```

use arael::model::{Model, Param, SelfBlock, CrossBlock};
use arael::vect::vect2d;
use arael::refs::{Ref, Arena};

// ---------------------------------------------------------------------------
// Line/arc visual style
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Default, serde::Serialize, serde::Deserialize)]
pub enum LineStyle {
    #[default]
    Solid,
    Dashed,
    DashDot,
}

impl LineStyle {
    pub fn next(self) -> Self {
        match self {
            LineStyle::Solid => LineStyle::Dashed,
            LineStyle::Dashed => LineStyle::DashDot,
            LineStyle::DashDot => LineStyle::Solid,
        }
    }
}

// ---------------------------------------------------------------------------
// Dimension annotations (constraint + visual)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DimensionEndpoint {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
}

#[derive(Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum DimensionKind {
    LineLength(Ref<Line>),
    PointPointDistance(DimensionEndpoint, DimensionEndpoint),
    PointLineDistance(DimensionEndpoint, Ref<Line>),
    ArcRadius(Ref<Arc>),
}

impl DimensionEndpoint {
    pub fn references_point(&self, r: Ref<Point>) -> bool {
        matches!(self, DimensionEndpoint::Point(p) if *p == r)
    }
    pub fn references_line(&self, r: Ref<Line>) -> bool {
        matches!(self, DimensionEndpoint::LineP1(l) | DimensionEndpoint::LineP2(l) if *l == r)
    }
    pub fn references_arc(&self, r: Ref<Arc>) -> bool {
        matches!(self, DimensionEndpoint::ArcCenter(a) | DimensionEndpoint::ArcStart(a) | DimensionEndpoint::ArcEnd(a) if *a == r)
    }
}

impl DimensionKind {
    pub fn references_point(&self, r: Ref<Point>) -> bool {
        match self {
            DimensionKind::PointPointDistance(a, b) => a.references_point(r) || b.references_point(r),
            DimensionKind::PointLineDistance(a, _) => a.references_point(r),
            _ => false,
        }
    }
    pub fn references_line(&self, r: Ref<Line>) -> bool {
        match self {
            DimensionKind::LineLength(l) => *l == r,
            DimensionKind::PointPointDistance(a, b) => a.references_line(r) || b.references_line(r),
            DimensionKind::PointLineDistance(a, l) => a.references_line(r) || *l == r,
            _ => false,
        }
    }
    pub fn references_arc(&self, r: Ref<Arc>) -> bool {
        match self {
            DimensionKind::ArcRadius(a) => *a == r,
            DimensionKind::PointPointDistance(a, b) => a.references_arc(r) || b.references_arc(r),
            DimensionKind::PointLineDistance(a, _) => a.references_arc(r),
            _ => false,
        }
    }
}

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct Dimension {
    pub kind: DimensionKind,
    pub value: f64,
    pub offset: vect2d,      // visual offset (y = perpendicular distance)
    pub text_along: f64,     // text position along the line: 0=center, -0.5..0.5=within arrows, outside=extend
    pub name: String,
}

// ---------------------------------------------------------------------------
// Constraint data stored on entities (for guarded self-constraints)
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
pub struct PointConstraints {
    #[arael(skip)]
    pub has_fix_x: bool,
    pub fix_x: f64,
    #[arael(skip)]
    pub has_fix_y: bool,
    pub fix_y: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
pub struct LineConstraints {
    #[arael(skip)]
    pub horizontal: bool,
    #[arael(skip)]
    pub vertical: bool,
    #[arael(skip)]
    pub has_length: bool,
    pub length: f64,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
pub struct ArcConstraints {
    #[arael(skip)]
    pub has_target_radius: bool,
    pub target_radius: f64,
}

// ---------------------------------------------------------------------------
// Entities
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Drift: weak regularizer
#[arael(constraint(hb, {
    let d = point.pos - point.pos_value;
    [d.x * sketch.drift_isigma, d.y * sketch.drift_isigma]
}))]
// Fix X coordinate
#[arael(constraint(hb, guard = self.constraints.has_fix_x, {
    [(point.pos.x - point.constraints.fix_x) * sketch.constraint_isigma]
}))]
// Fix Y coordinate
#[arael(constraint(hb, guard = self.constraints.has_fix_y, {
    [(point.pos.y - point.constraints.fix_y) * sketch.constraint_isigma]
}))]
pub struct Point {
    pub pos: Param<vect2d>,
    pub constraints: PointConstraints,
    #[arael(skip)]
    pub helper: bool,
    #[arael(skip)]
    pub name: String,
    #[serde(skip)]
    pub hb: SelfBlock<Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Drift: weak regularizer on both endpoints
#[arael(constraint(hb, {
    let d1 = line.p1 - line.p1_value;
    let d2 = line.p2 - line.p2_value;
    [d1.x * sketch.drift_isigma, d1.y * sketch.drift_isigma,
     d2.x * sketch.drift_isigma, d2.y * sketch.drift_isigma]
}))]
// Drift: weak regularizer on length
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let dx0 = line.p2_value.x - line.p1_value.x;
    let dy0 = line.p2_value.y - line.p1_value.y;
    [(sqrt(dx * dx + dy * dy) - sqrt(dx0 * dx0 + dy0 * dy0)) * sketch.drift_isigma]
}))]
// Horizontal: p1.y == p2.y
#[arael(constraint(hb, guard = self.constraints.horizontal, {
    [(line.p1.y - line.p2.y) * sketch.constraint_isigma]
}))]
// Vertical: p1.x == p2.x
#[arael(constraint(hb, guard = self.constraints.vertical, {
    [(line.p1.x - line.p2.x) * sketch.constraint_isigma]
}))]
// Length
#[arael(constraint(hb, guard = self.constraints.has_length, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    [(sqrt(dx * dx + dy * dy) - line.constraints.length) * sketch.constraint_isigma]
}))]
pub struct Line {
    pub p1: Param<vect2d>,
    pub p2: Param<vect2d>,
    pub constraints: LineConstraints,
    #[arael(skip)]
    pub style: LineStyle,
    #[arael(skip)]
    pub name: String,
    #[serde(skip)]
    pub hb: SelfBlock<Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
// Drift: weak regularizer on center, radius, angles
#[arael(constraint(hb, {
    let dc = arc.center - arc.center_value;
    let dr = arc.radius - arc.radius_value;
    let dsa = arc.start_angle - arc.start_angle_value;
    let dea = arc.end_angle - arc.end_angle_value;
    [dc.x * sketch.drift_isigma, dc.y * sketch.drift_isigma,
     dr * sketch.drift_isigma,
     dsa * sketch.drift_isigma, dea * sketch.drift_isigma]
}))]
// Target radius
#[arael(constraint(hb, guard = self.constraints.has_target_radius, {
    [(arc.radius - arc.constraints.target_radius) * sketch.constraint_isigma]
}))]
pub struct Arc {
    pub center: Param<vect2d>,
    pub radius: Param<f64>,
    pub start_angle: Param<f64>,
    pub end_angle: Param<f64>,
    #[arael(skip)]
    pub closed: bool,
    #[arael(skip)]
    pub style: LineStyle,
    #[arael(skip)]
    pub name: String,
    pub constraints: ArcConstraints,
    #[serde(skip)]
    pub hb: SelfBlock<Arc>,
}

// ---------------------------------------------------------------------------
// Cross-constraints (stored in root collections)
// ---------------------------------------------------------------------------

// -- Point-Point --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [a.pos.x - b.pos.x, a.pos.y - b.pos.y]
}))]
pub struct CoincidentPP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub b: Ref<Point>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.pos.x - b.pos.x;
    let dy = a.pos.y - b.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancepp.distance) * sketch.constraint_isigma]
}))]
pub struct DistancePP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub b: Ref<Point>,
    pub distance: f64,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

// -- Line-Line endpoint distance --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p1.x - b.p1.x; let dy = a.p1.y - b.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancell11.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL11 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p1.x - b.p2.x; let dy = a.p1.y - b.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancell12.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL12 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p2.x - b.p1.x; let dy = a.p2.y - b.p1.y;
    [(sqrt(dx * dx + dy * dy) - distancell21.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL21 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p2.x - b.p2.x; let dy = a.p2.y - b.p2.y;
    [(sqrt(dx * dx + dy * dy) - distancell22.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLL22 {
    #[arael(ref = root.lines)] pub a: Ref<Line>,
    #[arael(ref = root.lines)] pub b: Ref<Line>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Line>,
}

// -- Line endpoint to Point distance --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p1.x - point.pos.x; let dy = line.p1.y - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancelp1.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLP1 {
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - point.pos.x; let dy = line.p2.y - point.pos.y;
    [(sqrt(dx * dx + dy * dy) - distancelp2.distance) * sketch.constraint_isigma]
}))]
pub struct DistanceLP2 {
    #[arael(ref = root.lines)] pub line: Ref<Line>,
    #[arael(ref = root.points)] pub point: Ref<Point>,
    pub distance: f64,
    #[serde(skip)] pub hb: CrossBlock<Line, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.pos.x - b.pos.x - horizontaldistancepp.distance) * sketch.constraint_isigma]
}))]
pub struct HorizontalDistancePP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub b: Ref<Point>,
    pub distance: f64,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.pos.y - b.pos.y - verticaldistancepp.distance) * sketch.constraint_isigma]
}))]
pub struct VerticalDistancePP {
    #[arael(ref = root.points)]
    pub a: Ref<Point>,
    #[arael(ref = root.points)]
    pub b: Ref<Point>,
    pub distance: f64,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Point>,
}

// -- Point-Line --

// Point lies on infinite line through p1-p2 (cross product = 0)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    [((point.pos.x - line.p1.x) * dy - (point.pos.y - line.p1.y) * dx) / len
     * sketch.constraint_isigma]
}))]
pub struct PointOnLine {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Line>,
}

// Point at midpoint of line segment
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let mx = (line.p1.x + line.p2.x) * 0.5;
    let my = (line.p1.y + line.p2.y) * 0.5;
    [(point.pos.x - mx) * sketch.constraint_isigma,
     (point.pos.y - my) * sketch.constraint_isigma]
}))]
pub struct MidpointConstraint {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Line>,
}

// -- Point-Arc --

// Point lies on circle defined by arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = point.pos.x - arc.center.x;
    let dy = point.pos.y - arc.center.y;
    [(sqrt(dx * dx + dy * dy) - arc.radius) * sketch.constraint_isigma]
}))]
pub struct PointOnArc {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Point coincides with arc center
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(point.pos.x - arc.center.x) * sketch.constraint_isigma,
     (point.pos.y - arc.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcCenter {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Point coincides with arc start endpoint (center + radius * [cos(sa), sin(sa)])
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc.center.x + arc.radius * cos(arc.start_angle);
    let sy = arc.center.y + arc.radius * sin(arc.start_angle);
    [(point.pos.x - sx) * sketch.constraint_isigma,
     (point.pos.y - sy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStart {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// Point coincides with arc end endpoint (center + radius * [cos(ea), sin(ea)])
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc.center.x + arc.radius * cos(arc.end_angle);
    let ey = arc.center.y + arc.radius * sin(arc.end_angle);
    [(point.pos.x - ex) * sketch.constraint_isigma,
     (point.pos.y - ey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEnd {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Point, Arc>,
}

// -- Line-Line --

// Parallel: cross product of direction vectors = 0
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    let len1 = sqrt(dx1 * dx1 + dy1 * dy1);
    let len2 = sqrt(dx2 * dx2 + dy2 * dy2);
    [(dx1 * dy2 - dy1 * dx2) / (len1 * len2) * sketch.constraint_isigma]
}))]
pub struct Parallel {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Perpendicular: dot product of direction vectors = 0
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    let len1 = sqrt(dx1 * dx1 + dy1 * dy1);
    let len2 = sqrt(dx2 * dx2 + dy2 * dy2);
    [(dx1 * dx2 + dy1 * dy2) / (len1 * len2) * sketch.constraint_isigma]
}))]
pub struct Perpendicular {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Collinear: line2 endpoints both lie on infinite line of line1
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.p2.x - a.p1.x;
    let dy = a.p2.y - a.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let cross1 = ((b.p1.x - a.p1.x) * dy - (b.p1.y - a.p1.y) * dx) / len;
    let cross2 = ((b.p2.x - a.p1.x) * dy - (b.p2.y - a.p1.y) * dx) / len;
    [cross1 * sketch.constraint_isigma, cross2 * sketch.constraint_isigma]
}))]
pub struct Collinear {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Equal length
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    [(sqrt(dx1*dx1 + dy1*dy1) - sqrt(dx2*dx2 + dy2*dy2)) * sketch.constraint_isigma]
}))]
pub struct EqualLength {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Angle between lines
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx1 = a.p2.x - a.p1.x;
    let dy1 = a.p2.y - a.p1.y;
    let dx2 = b.p2.x - b.p1.x;
    let dy2 = b.p2.y - b.p1.y;
    [(atan2(dx1 * dy2 - dy1 * dx2, dx1 * dx2 + dy1 * dy2) - angleconstraint.angle)
     * sketch.constraint_isigma]
}))]
pub struct AngleConstraint {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    pub angle: f64,  // target angle in radians
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// -- Line-Point (endpoint coincidence) --

// Line p1 coincides with standalone point
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(line.p1.x - point.pos.x) * sketch.constraint_isigma,
     (line.p1.y - point.pos.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1 {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Point>,
}

// Line p2 coincides with standalone point
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(line.p2.x - point.pos.x) * sketch.constraint_isigma,
     (line.p2.y - point.pos.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2 {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Point>,
}

// -- Line-Line endpoint coincidence (4 variants for endpoint combos) --

// a.p1 == b.p1
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.p1.x - b.p1.x) * sketch.constraint_isigma,
     (a.p1.y - b.p1.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLL11 {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// a.p1 == b.p2
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.p1.x - b.p2.x) * sketch.constraint_isigma,
     (a.p1.y - b.p2.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLL12 {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// a.p2 == b.p1  (most common: end of line a -> start of line b)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.p2.x - b.p1.x) * sketch.constraint_isigma,
     (a.p2.y - b.p1.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLL21 {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// a.p2 == b.p2
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.p2.x - b.p2.x) * sketch.constraint_isigma,
     (a.p2.y - b.p2.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLL22 {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// -- Line-Arc --

// Line tangent to arc (distance from center to infinite line = radius)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let dist = ((arc.center.x - line.p1.x) * dy - (arc.center.y - line.p1.y) * dx) / len;
    [(dist * dist - arc.radius * arc.radius) * sketch.constraint_isigma]
}))]
pub struct TangentLA {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// -- Arc-Arc --

// Concentric: centers coincide
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.center.x - b.center.x) * sketch.constraint_isigma,
     (a.center.y - b.center.y) * sketch.constraint_isigma]
}))]
pub struct Concentric {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Equal radius
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(a.radius - b.radius) * sketch.constraint_isigma]
}))]
pub struct EqualRadius {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// Tangent arc-arc (external: dist between centers = r1 + r2)
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = a.center.x - b.center.x;
    let dy = a.center.y - b.center.y;
    let dist = sqrt(dx * dx + dy * dy);
    let target = a.radius + b.radius;
    [(dist - target) * sketch.constraint_isigma]
}))]
pub struct TangentAA {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// -- Line endpoint <-> Arc point coincidence --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(line.p1.x - arc.center.x) * sketch.constraint_isigma,
     (line.p1.y - arc.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1ArcCenter {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    [(line.p2.x - arc.center.x) * sketch.constraint_isigma,
     (line.p2.y - arc.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2ArcCenter {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc.center.x + arc.radius * cos(arc.start_angle);
    let sy = arc.center.y + arc.radius * sin(arc.start_angle);
    [(line.p1.x - sx) * sketch.constraint_isigma,
     (line.p1.y - sy) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1ArcStart {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let sx = arc.center.x + arc.radius * cos(arc.start_angle);
    let sy = arc.center.y + arc.radius * sin(arc.start_angle);
    [(line.p2.x - sx) * sketch.constraint_isigma,
     (line.p2.y - sy) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2ArcStart {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc.center.x + arc.radius * cos(arc.end_angle);
    let ey = arc.center.y + arc.radius * sin(arc.end_angle);
    [(line.p1.x - ex) * sketch.constraint_isigma,
     (line.p1.y - ey) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP1ArcEnd {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let ex = arc.center.x + arc.radius * cos(arc.end_angle);
    let ey = arc.center.y + arc.radius * sin(arc.end_angle);
    [(line.p2.x - ex) * sketch.constraint_isigma,
     (line.p2.y - ey) * sketch.constraint_isigma]
}))]
pub struct CoincidentLP2ArcEnd {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// -- Arc-Arc endpoint coincidence --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let bsx = b.center.x + b.radius * cos(b.start_angle);
    let bsy = b.center.y + b.radius * sin(b.start_angle);
    [(a.center.x - bsx) * sketch.constraint_isigma,
     (a.center.y - bsy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcCenterStart {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let bex = b.center.x + b.radius * cos(b.end_angle);
    let bey = b.center.y + b.radius * sin(b.end_angle);
    [(a.center.x - bex) * sketch.constraint_isigma,
     (a.center.y - bey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcCenterEnd {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = a.center.x + a.radius * cos(a.start_angle);
    let asy = a.center.y + a.radius * sin(a.start_angle);
    [(asx - b.center.x) * sketch.constraint_isigma,
     (asy - b.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStartCenter {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = a.center.x + a.radius * cos(a.end_angle);
    let aey = a.center.y + a.radius * sin(a.end_angle);
    [(aex - b.center.x) * sketch.constraint_isigma,
     (aey - b.center.y) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEndCenter {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = a.center.x + a.radius * cos(a.start_angle);
    let asy = a.center.y + a.radius * sin(a.start_angle);
    let bsx = b.center.x + b.radius * cos(b.start_angle);
    let bsy = b.center.y + b.radius * sin(b.start_angle);
    [(asx - bsx) * sketch.constraint_isigma,
     (asy - bsy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStartStart {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let asx = a.center.x + a.radius * cos(a.start_angle);
    let asy = a.center.y + a.radius * sin(a.start_angle);
    let bex = b.center.x + b.radius * cos(b.end_angle);
    let bey = b.center.y + b.radius * sin(b.end_angle);
    [(asx - bex) * sketch.constraint_isigma,
     (asy - bey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcStartEnd {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = a.center.x + a.radius * cos(a.end_angle);
    let aey = a.center.y + a.radius * sin(a.end_angle);
    let bsx = b.center.x + b.radius * cos(b.start_angle);
    let bsy = b.center.y + b.radius * sin(b.start_angle);
    [(aex - bsx) * sketch.constraint_isigma,
     (aey - bsy) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEndStart {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let aex = a.center.x + a.radius * cos(a.end_angle);
    let aey = a.center.y + a.radius * sin(a.end_angle);
    let bex = b.center.x + b.radius * cos(b.end_angle);
    let bey = b.center.y + b.radius * sin(b.end_angle);
    [(aex - bex) * sketch.constraint_isigma,
     (aey - bey) * sketch.constraint_isigma]
}))]
pub struct CoincidentArcEndEnd {
    #[arael(ref = root.arcs)]
    pub a: Ref<Arc>,
    #[arael(ref = root.arcs)]
    pub b: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Arc, Arc>,
}

// -- Line endpoint on line --

// Line a's p1 lies on infinite line through b's p1-p2
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = b.p2.x - b.p1.x;
    let dy = b.p2.y - b.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    [((a.p1.x - b.p1.x) * dy - (a.p1.y - b.p1.y) * dx) / len
     * sketch.constraint_isigma]
}))]
pub struct LineP1OnLine {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// Line a's p2 lies on infinite line through b's p1-p2
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = b.p2.x - b.p1.x;
    let dy = b.p2.y - b.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    [((a.p2.x - b.p1.x) * dy - (a.p2.y - b.p1.y) * dx) / len
     * sketch.constraint_isigma]
}))]
pub struct LineP2OnLine {
    #[arael(ref = root.lines)]
    pub a: Ref<Line>,
    #[arael(ref = root.lines)]
    pub b: Ref<Line>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Line>,
}

// -- Line endpoint on arc --

// Line p1 lies on circle defined by arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p1.x - arc.center.x;
    let dy = line.p1.y - arc.center.y;
    [(sqrt(dx * dx + dy * dy) - arc.radius) * sketch.constraint_isigma]
}))]
pub struct LineP1OnArc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// Line p2 lies on circle defined by arc
#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - arc.center.x;
    let dy = line.p2.y - arc.center.y;
    [(sqrt(dx * dx + dy * dy) - arc.radius) * sketch.constraint_isigma]
}))]
pub struct LineP2OnArc {
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    #[arael(ref = root.arcs)]
    pub arc: Ref<Arc>,
    #[serde(skip)]
    pub hb: CrossBlock<Line, Arc>,
}

// -- Distance Point-Line --

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(constraint(hb, {
    let dx = line.p2.x - line.p1.x;
    let dy = line.p2.y - line.p1.y;
    let len = sqrt(dx * dx + dy * dy);
    let dist = ((point.pos.x - line.p1.x) * dy - (point.pos.y - line.p1.y) * dx) / len;
    [(dist - distancepl.distance) * sketch.constraint_isigma]
}))]
pub struct DistancePL {
    #[arael(ref = root.points)]
    pub point: Ref<Point>,
    #[arael(ref = root.lines)]
    pub line: Ref<Line>,
    pub distance: f64,  // signed distance (positive = left of line direction)
    #[serde(skip)]
    pub hb: CrossBlock<Point, Line>,
}

// ---------------------------------------------------------------------------
// Root
// ---------------------------------------------------------------------------

#[derive(serde::Serialize, serde::Deserialize)]
#[arael::model]
#[arael(root)]
pub struct Sketch {
    pub points: Arena<Point>,
    pub lines: Arena<Line>,
    pub arcs: Arena<Arc>,
    // Solver parameters
    pub drift_isigma: f64,
    pub constraint_isigma: f64,
    // Auto-naming counters
    #[arael(skip)]
    pub next_point_id: u32,
    #[arael(skip)]
    pub next_line_id: u32,
    #[arael(skip)]
    pub next_arc_id: u32,
    // Cross-constraint collections
    pub coincident_pp: std::vec::Vec<CoincidentPP>,
    pub coincident_lp1: std::vec::Vec<CoincidentLP1>,
    pub coincident_lp2: std::vec::Vec<CoincidentLP2>,
    pub coincident_ll11: std::vec::Vec<CoincidentLL11>,
    pub coincident_ll12: std::vec::Vec<CoincidentLL12>,
    pub coincident_ll21: std::vec::Vec<CoincidentLL21>,
    pub coincident_ll22: std::vec::Vec<CoincidentLL22>,
    pub distance_pp: std::vec::Vec<DistancePP>,
    pub hdistance_pp: std::vec::Vec<HorizontalDistancePP>,
    pub vdistance_pp: std::vec::Vec<VerticalDistancePP>,
    pub point_on_line: std::vec::Vec<PointOnLine>,
    pub midpoint: std::vec::Vec<MidpointConstraint>,
    pub point_on_arc: std::vec::Vec<PointOnArc>,
    pub parallel: std::vec::Vec<Parallel>,
    pub perpendicular: std::vec::Vec<Perpendicular>,
    pub collinear: std::vec::Vec<Collinear>,
    pub equal_length: std::vec::Vec<EqualLength>,
    pub angle: std::vec::Vec<AngleConstraint>,
    pub tangent_la: std::vec::Vec<TangentLA>,
    pub concentric: std::vec::Vec<Concentric>,
    pub equal_radius: std::vec::Vec<EqualRadius>,
    pub tangent_aa: std::vec::Vec<TangentAA>,
    pub distance_pl: std::vec::Vec<DistancePL>,
    pub line_p1_on_line: std::vec::Vec<LineP1OnLine>,
    pub line_p2_on_line: std::vec::Vec<LineP2OnLine>,
    pub coincident_arc_center: std::vec::Vec<CoincidentArcCenter>,
    pub coincident_arc_start: std::vec::Vec<CoincidentArcStart>,
    pub coincident_arc_end: std::vec::Vec<CoincidentArcEnd>,
    // Line endpoint <-> Arc point
    pub coincident_lp1_arc_center: std::vec::Vec<CoincidentLP1ArcCenter>,
    pub coincident_lp2_arc_center: std::vec::Vec<CoincidentLP2ArcCenter>,
    pub coincident_lp1_arc_start: std::vec::Vec<CoincidentLP1ArcStart>,
    pub coincident_lp2_arc_start: std::vec::Vec<CoincidentLP2ArcStart>,
    pub coincident_lp1_arc_end: std::vec::Vec<CoincidentLP1ArcEnd>,
    pub coincident_lp2_arc_end: std::vec::Vec<CoincidentLP2ArcEnd>,
    // Arc-Arc endpoint
    pub coincident_arc_center_start: std::vec::Vec<CoincidentArcCenterStart>,
    pub coincident_arc_center_end: std::vec::Vec<CoincidentArcCenterEnd>,
    pub coincident_arc_start_center: std::vec::Vec<CoincidentArcStartCenter>,
    pub coincident_arc_end_center: std::vec::Vec<CoincidentArcEndCenter>,
    pub coincident_arc_start_start: std::vec::Vec<CoincidentArcStartStart>,
    pub coincident_arc_start_end: std::vec::Vec<CoincidentArcStartEnd>,
    pub coincident_arc_end_start: std::vec::Vec<CoincidentArcEndStart>,
    pub coincident_arc_end_end: std::vec::Vec<CoincidentArcEndEnd>,
    pub line_p1_on_arc: std::vec::Vec<LineP1OnArc>,
    pub line_p2_on_arc: std::vec::Vec<LineP2OnArc>,
    pub distance_ll11: std::vec::Vec<DistanceLL11>,
    pub distance_ll12: std::vec::Vec<DistanceLL12>,
    pub distance_ll21: std::vec::Vec<DistanceLL21>,
    pub distance_ll22: std::vec::Vec<DistanceLL22>,
    pub distance_lp1: std::vec::Vec<DistanceLP1>,
    pub distance_lp2: std::vec::Vec<DistanceLP2>,
    // Dimension annotations
    #[arael(skip)]
    pub dimensions: std::vec::Vec<Dimension>,
    #[arael(skip)]
    pub next_dimension_id: u32,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

impl Sketch {
    pub fn new() -> Self {
        let drift_sigma = 1000.0_f64;
        Sketch {
            points: Arena::new(),
            lines: Arena::new(),
            arcs: Arena::new(),
            drift_isigma: 1.0 / drift_sigma,
            constraint_isigma: 1000.0, // tight constraints
            next_point_id: 0,
            next_line_id: 0,
            next_arc_id: 0,
            coincident_pp: Vec::new(),
            coincident_lp1: Vec::new(),
            coincident_lp2: Vec::new(),
            coincident_ll11: Vec::new(),
            coincident_ll12: Vec::new(),
            coincident_ll21: Vec::new(),
            coincident_ll22: Vec::new(),
            distance_pp: Vec::new(),
            hdistance_pp: Vec::new(),
            vdistance_pp: Vec::new(),
            point_on_line: Vec::new(),
            midpoint: Vec::new(),
            point_on_arc: Vec::new(),
            parallel: Vec::new(),
            perpendicular: Vec::new(),
            collinear: Vec::new(),
            equal_length: Vec::new(),
            angle: Vec::new(),
            tangent_la: Vec::new(),
            concentric: Vec::new(),
            equal_radius: Vec::new(),
            tangent_aa: Vec::new(),
            distance_pl: Vec::new(),
            line_p1_on_line: Vec::new(),
            line_p2_on_line: Vec::new(),
            coincident_arc_center: Vec::new(),
            coincident_arc_start: Vec::new(),
            coincident_arc_end: Vec::new(),
            coincident_lp1_arc_center: Vec::new(),
            coincident_lp2_arc_center: Vec::new(),
            coincident_lp1_arc_start: Vec::new(),
            coincident_lp2_arc_start: Vec::new(),
            coincident_lp1_arc_end: Vec::new(),
            coincident_lp2_arc_end: Vec::new(),
            coincident_arc_center_start: Vec::new(),
            coincident_arc_center_end: Vec::new(),
            coincident_arc_start_center: Vec::new(),
            coincident_arc_end_center: Vec::new(),
            coincident_arc_start_start: Vec::new(),
            coincident_arc_start_end: Vec::new(),
            coincident_arc_end_start: Vec::new(),
            coincident_arc_end_end: Vec::new(),
            line_p1_on_arc: Vec::new(),
            line_p2_on_arc: Vec::new(),
            distance_ll11: Vec::new(),
            distance_ll12: Vec::new(),
            distance_ll21: Vec::new(),
            distance_ll22: Vec::new(),
            distance_lp1: Vec::new(),
            distance_lp2: Vec::new(),
            dimensions: Vec::new(),
            next_dimension_id: 0,
        }
    }

    pub fn add_point(&mut self, pos: vect2d) -> Ref<Point> {
        let name = format!("P{}", self.next_point_id);
        self.next_point_id += 1;
        self.points.push(Point {
            pos: Param::new(pos),
            constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
            helper: false, name,
            hb: SelfBlock::new(),
        })
    }

    pub fn add_point_fixed(&mut self, pos: vect2d) -> Ref<Point> {
        let name = format!("P{}", self.next_point_id);
        self.next_point_id += 1;
        self.points.push(Point {
            pos: Param::fixed(pos),
            constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
            helper: false, name,
            hb: SelfBlock::new(),
        })
    }

    pub fn add_helper_point(&mut self, pos: vect2d) -> Ref<Point> {
        let name = format!("Pc{}", self.next_point_id);
        self.next_point_id += 1;
        self.points.push(Point {
            pos: Param::new(pos),
            constraints: PointConstraints { has_fix_x: false, fix_x: 0.0, has_fix_y: false, fix_y: 0.0 },
            helper: true, name,
            hb: SelfBlock::new(),
        })
    }

    pub fn add_line(&mut self, p1: vect2d, p2: vect2d) -> Ref<Line> {
        let name = format!("L{}", self.next_line_id);
        self.next_line_id += 1;
        self.lines.push(Line {
            p1: Param::new(p1),
            p2: Param::new(p2),
            constraints: LineConstraints { horizontal: false, vertical: false, has_length: false, length: 0.0 },
            style: LineStyle::Solid, name,
            hb: SelfBlock::new(),
        })
    }

    pub fn add_arc(&mut self, center: vect2d, radius: f64, start: f64, end: f64, closed: bool) -> Ref<Arc> {
        let name = format!("A{}", self.next_arc_id);
        self.next_arc_id += 1;
        self.arcs.push(Arc {
            center: Param::new(center),
            radius: Param::new(radius),
            start_angle: Param::new(start),
            end_angle: Param::new(end),
            closed,
            style: LineStyle::Solid, name,
            constraints: ArcConstraints { has_target_radius: false, target_radius: 0.0 },
            hb: SelfBlock::new(),
        })
    }

    /// Remove a point and all constraints referencing it.
    pub fn delete_point(&mut self, r: Ref<Point>) {
        self.dimensions.retain(|d| !d.kind.references_point(r));
        self.points.remove(r);
        self.coincident_pp.retain(|c| c.a != r && c.b != r);
        self.coincident_lp1.retain(|c| c.point != r);
        self.coincident_lp2.retain(|c| c.point != r);
        self.distance_pp.retain(|c| c.a != r && c.b != r);
        self.hdistance_pp.retain(|c| c.a != r && c.b != r);
        self.vdistance_pp.retain(|c| c.a != r && c.b != r);
        self.point_on_line.retain(|c| c.point != r);
        self.midpoint.retain(|c| c.point != r);
        self.point_on_arc.retain(|c| c.point != r);
        self.distance_pl.retain(|c| c.point != r);
        self.coincident_arc_center.retain(|c| c.point != r);
        self.coincident_arc_start.retain(|c| c.point != r);
        self.coincident_arc_end.retain(|c| c.point != r);
        self.distance_lp1.retain(|c| c.point != r);
        self.distance_lp2.retain(|c| c.point != r);
    }

    /// Remove a line and all constraints referencing it.
    pub fn delete_line(&mut self, r: Ref<Line>) {
        self.dimensions.retain(|d| !d.kind.references_line(r));
        self.lines.remove(r);
        self.coincident_lp1.retain(|c| c.line != r);
        self.coincident_lp2.retain(|c| c.line != r);
        self.coincident_ll11.retain(|c| c.a != r && c.b != r);
        self.coincident_ll12.retain(|c| c.a != r && c.b != r);
        self.coincident_ll21.retain(|c| c.a != r && c.b != r);
        self.coincident_ll22.retain(|c| c.a != r && c.b != r);
        self.point_on_line.retain(|c| c.line != r);
        self.midpoint.retain(|c| c.line != r);
        self.parallel.retain(|c| c.a != r && c.b != r);
        self.perpendicular.retain(|c| c.a != r && c.b != r);
        self.collinear.retain(|c| c.a != r && c.b != r);
        self.equal_length.retain(|c| c.a != r && c.b != r);
        self.angle.retain(|c| c.a != r && c.b != r);
        self.tangent_la.retain(|c| c.line != r);
        self.distance_pl.retain(|c| c.line != r);
        self.line_p1_on_line.retain(|c| c.a != r && c.b != r);
        self.line_p2_on_line.retain(|c| c.a != r && c.b != r);
        self.coincident_lp1_arc_center.retain(|c| c.line != r);
        self.coincident_lp2_arc_center.retain(|c| c.line != r);
        self.coincident_lp1_arc_start.retain(|c| c.line != r);
        self.coincident_lp2_arc_start.retain(|c| c.line != r);
        self.coincident_lp1_arc_end.retain(|c| c.line != r);
        self.coincident_lp2_arc_end.retain(|c| c.line != r);
        self.line_p1_on_arc.retain(|c| c.line != r);
        self.line_p2_on_arc.retain(|c| c.line != r);
        self.distance_ll11.retain(|c| c.a != r && c.b != r);
        self.distance_ll12.retain(|c| c.a != r && c.b != r);
        self.distance_ll21.retain(|c| c.a != r && c.b != r);
        self.distance_ll22.retain(|c| c.a != r && c.b != r);
        self.distance_lp1.retain(|c| c.line != r);
        self.distance_lp2.retain(|c| c.line != r);
        self.cleanup_helper_points();
    }

    /// Remove an arc and all constraints referencing it.
    pub fn delete_arc(&mut self, r: Ref<Arc>) {
        self.dimensions.retain(|d| !d.kind.references_arc(r));
        self.arcs.remove(r);
        self.point_on_arc.retain(|c| c.arc != r);
        self.line_p1_on_arc.retain(|c| c.arc != r);
        self.line_p2_on_arc.retain(|c| c.arc != r);
        self.tangent_la.retain(|c| c.arc != r);
        self.concentric.retain(|c| c.a != r && c.b != r);
        self.equal_radius.retain(|c| c.a != r && c.b != r);
        self.tangent_aa.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_center.retain(|c| c.arc != r);
        self.coincident_arc_start.retain(|c| c.arc != r);
        self.coincident_arc_end.retain(|c| c.arc != r);
        self.coincident_lp1_arc_center.retain(|c| c.arc != r);
        self.coincident_lp2_arc_center.retain(|c| c.arc != r);
        self.coincident_lp1_arc_start.retain(|c| c.arc != r);
        self.coincident_lp2_arc_start.retain(|c| c.arc != r);
        self.coincident_lp1_arc_end.retain(|c| c.arc != r);
        self.coincident_lp2_arc_end.retain(|c| c.arc != r);
        self.coincident_arc_center_start.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_center_end.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_start_center.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_end_center.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_start_start.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_start_end.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_end_start.retain(|c| c.a != r && c.b != r);
        self.coincident_arc_end_end.retain(|c| c.a != r && c.b != r);
        self.cleanup_helper_points();
    }

    /// Remove helper points that have no remaining constraints referencing them.
    pub fn cleanup_helper_points(&mut self) {
        // Collect all point refs that appear in any constraint
        let mut referenced: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut add_pt = |r: Ref<Point>| { referenced.insert(r.index()); };
        for c in &self.coincident_pp { add_pt(c.a); add_pt(c.b); }
        for c in &self.coincident_lp1 { add_pt(c.point); }
        for c in &self.coincident_lp2 { add_pt(c.point); }
        for c in &self.distance_pp { add_pt(c.a); add_pt(c.b); }
        for c in &self.hdistance_pp { add_pt(c.a); add_pt(c.b); }
        for c in &self.vdistance_pp { add_pt(c.a); add_pt(c.b); }
        for c in &self.point_on_line { add_pt(c.point); }
        for c in &self.midpoint { add_pt(c.point); }
        for c in &self.point_on_arc { add_pt(c.point); }
        for c in &self.distance_pl { add_pt(c.point); }
        for c in &self.coincident_arc_center { add_pt(c.point); }
        for c in &self.coincident_arc_start { add_pt(c.point); }
        for c in &self.coincident_arc_end { add_pt(c.point); }

        // Remove unreferenced helper points
        let to_remove: std::vec::Vec<Ref<Point>> = self.points.refs()
            .filter(|r| {
                let p = &self.points[*r];
                p.helper && !referenced.contains(&r.index())
            })
            .collect();
        for r in to_remove {
            self.points.remove(r);
        }
    }

    /// Remove duplicate constraints from all collections. Prints a warning if any are found.
    pub fn dedup_constraints(&mut self) {
        let mut total_removed = 0usize;
        macro_rules! dedup_ab {
            ($coll:expr, $name:expr, $container_a:expr, $container_b:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.a.index(), c.b.index())) {
                        let na = $container_a.get(c.a).map(|e| e.name.as_str()).unwrap_or("?");
                        let nb = $container_b.get(c.b).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: a={}, b={}", $name, na, nb);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.a.index(), c.b.index())));
            };
        }
        macro_rules! dedup_lp {
            ($coll:expr, $name:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.line.index(), c.point.index())) {
                        let nl = self.lines.get(c.line).map(|e| e.name.as_str()).unwrap_or("?");
                        let np = self.points.get(c.point).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: line={}, point={}", $name, nl, np);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.line.index(), c.point.index())));
            };
        }
        macro_rules! dedup_la {
            ($coll:expr, $name:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.line.index(), c.arc.index())) {
                        let nl = self.lines.get(c.line).map(|e| e.name.as_str()).unwrap_or("?");
                        let na = self.arcs.get(c.arc).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: line={}, arc={}", $name, nl, na);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.line.index(), c.arc.index())));
            };
        }
        macro_rules! dedup_pa {
            ($coll:expr, $name:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.point.index(), c.arc.index())) {
                        let np = self.points.get(c.point).map(|e| e.name.as_str()).unwrap_or("?");
                        let na = self.arcs.get(c.arc).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: point={}, arc={}", $name, np, na);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.point.index(), c.arc.index())));
            };
        }
        macro_rules! dedup_pl {
            ($coll:expr, $name:expr) => {
                let mut seen = std::collections::HashSet::new();
                for c in $coll.iter() {
                    if !seen.insert((c.point.index(), c.line.index())) {
                        let np = self.points.get(c.point).map(|e| e.name.as_str()).unwrap_or("?");
                        let nl = self.lines.get(c.line).map(|e| e.name.as_str()).unwrap_or("?");
                        eprintln!("BUG: duplicate {} constraint: point={}, line={}", $name, np, nl);
                        eprintln!("{}", std::backtrace::Backtrace::force_capture());
                        total_removed += 1;
                    }
                }
                seen.clear();
                $coll.retain(|c| seen.insert((c.point.index(), c.line.index())));
            };
        }
        dedup_ab!(self.coincident_pp, "coincident_pp", self.points, self.points);
        // PP is symmetric: (a,b) == (b,a)
        {
            let before = self.coincident_pp.len();
            let mut seen = std::collections::HashSet::new();
            self.coincident_pp.retain(|c| {
                let (a, b) = (c.a.index().min(c.b.index()), c.a.index().max(c.b.index()));
                seen.insert((a, b))
            });
            let removed = before - self.coincident_pp.len();
            if removed > 0 { eprintln!("BUG: removed {} cross-duplicate coincident_pp constraints", removed); total_removed += removed; }
        }
        dedup_lp!(self.coincident_lp1, "coincident_lp1");
        dedup_lp!(self.coincident_lp2, "coincident_lp2");
        dedup_ab!(self.coincident_ll11, "coincident_ll11", self.lines, self.lines);
        dedup_ab!(self.coincident_ll12, "coincident_ll12", self.lines, self.lines);
        dedup_ab!(self.coincident_ll21, "coincident_ll21", self.lines, self.lines);
        dedup_ab!(self.coincident_ll22, "coincident_ll22", self.lines, self.lines);
        // Cross-Vec dedup for LL: ll11(a,b)==ll11(b,a), ll22(a,b)==ll22(b,a), ll12(a,b)==ll21(b,a)
        {
            let mut seen = std::collections::HashSet::new();
            // Normalize: represent each endpoint pair as (min_id, max_id) where id encodes line+endpoint
            let ep_id = |line: u32, is_p2: bool| -> u64 { (line as u64) << 1 | (is_p2 as u64) };
            let mut add = |line_a: u32, p2_a: bool, line_b: u32, p2_b: bool| -> bool {
                let a = ep_id(line_a, p2_a);
                let b = ep_id(line_b, p2_b);
                let key = (a.min(b), a.max(b));
                seen.insert(key)
            };
            let before = self.coincident_ll11.len() + self.coincident_ll12.len()
                + self.coincident_ll21.len() + self.coincident_ll22.len();
            self.coincident_ll11.retain(|c| add(c.a.index(), false, c.b.index(), false));
            self.coincident_ll12.retain(|c| add(c.a.index(), false, c.b.index(), true));
            self.coincident_ll21.retain(|c| add(c.a.index(), true, c.b.index(), false));
            self.coincident_ll22.retain(|c| add(c.a.index(), true, c.b.index(), true));
            let after = self.coincident_ll11.len() + self.coincident_ll12.len()
                + self.coincident_ll21.len() + self.coincident_ll22.len();
            let removed = before - after;
            if removed > 0 { eprintln!("BUG: removed {} cross-duplicate LL coincident constraints", removed); total_removed += removed; }
        }
        dedup_ab!(self.distance_pp, "distance_pp", self.points, self.points);
        dedup_ab!(self.hdistance_pp, "hdistance_pp", self.points, self.points);
        dedup_ab!(self.vdistance_pp, "vdistance_pp", self.points, self.points);
        dedup_pl!(self.point_on_line, "point_on_line");
        dedup_pl!(self.midpoint, "midpoint");
        dedup_pa!(self.point_on_arc, "point_on_arc");
        dedup_ab!(self.parallel, "parallel", self.lines, self.lines);
        dedup_ab!(self.perpendicular, "perpendicular", self.lines, self.lines);
        dedup_ab!(self.collinear, "collinear", self.lines, self.lines);
        dedup_ab!(self.equal_length, "equal_length", self.lines, self.lines);
        dedup_ab!(self.angle, "angle", self.lines, self.lines);
        dedup_la!(self.tangent_la, "tangent_la");
        dedup_la!(self.line_p1_on_arc, "line_p1_on_arc");
        dedup_la!(self.line_p2_on_arc, "line_p2_on_arc");
        dedup_ab!(self.concentric, "concentric", self.arcs, self.arcs);
        dedup_ab!(self.equal_radius, "equal_radius", self.arcs, self.arcs);
        dedup_ab!(self.tangent_aa, "tangent_aa", self.arcs, self.arcs);
        dedup_pl!(self.distance_pl, "distance_pl");
        dedup_ab!(self.line_p1_on_line, "line_p1_on_line", self.lines, self.lines);
        dedup_ab!(self.line_p2_on_line, "line_p2_on_line", self.lines, self.lines);
        dedup_pa!(self.coincident_arc_center, "coincident_arc_center");
        dedup_pa!(self.coincident_arc_start, "coincident_arc_start");
        dedup_pa!(self.coincident_arc_end, "coincident_arc_end");
        dedup_la!(self.coincident_lp1_arc_center, "coincident_lp1_arc_center");
        dedup_la!(self.coincident_lp2_arc_center, "coincident_lp2_arc_center");
        dedup_la!(self.coincident_lp1_arc_start, "coincident_lp1_arc_start");
        dedup_la!(self.coincident_lp2_arc_start, "coincident_lp2_arc_start");
        dedup_la!(self.coincident_lp1_arc_end, "coincident_lp1_arc_end");
        dedup_la!(self.coincident_lp2_arc_end, "coincident_lp2_arc_end");
        dedup_ab!(self.coincident_arc_center_start, "coincident_arc_center_start", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_center_end, "coincident_arc_center_end", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_start_center, "coincident_arc_start_center", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_end_center, "coincident_arc_end_center", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_start_start, "coincident_arc_start_start", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_start_end, "coincident_arc_start_end", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_end_start, "coincident_arc_end_start", self.arcs, self.arcs);
        dedup_ab!(self.coincident_arc_end_end, "coincident_arc_end_end", self.arcs, self.arcs);
        let _ = total_removed;
    }

    /// Merge duplicate helper points at the same position and consolidate
    /// helper-point-bridged constraints into direct constraints.
    pub fn consolidate_helper_constraints(&mut self) {
        // Phase 1: Merge duplicate helper points at the same position.
        // If two helper points are at the same position, rewrite all constraints
        // referencing the second to reference the first, then remove the second.
        let helper_refs: std::vec::Vec<Ref<Point>> = self.points.refs()
            .filter(|r| self.points[*r].helper)
            .collect();
        let mut merged = std::collections::HashMap::<u32, Ref<Point>>::new(); // old -> canonical
        for i in 0..helper_refs.len() {
            let ri = helper_refs[i];
            if merged.contains_key(&ri.index()) { continue; }
            let pi = self.points[ri].pos.value;
            for j in (i+1)..helper_refs.len() {
                let rj = helper_refs[j];
                if merged.contains_key(&rj.index()) { continue; }
                let pj = self.points[rj].pos.value;
                if (pi.x - pj.x).abs() < 1e-9 && (pi.y - pj.y).abs() < 1e-9 {
                    merged.insert(rj.index(), ri);
                    eprintln!("INFO: merging duplicate helper point {} into {}", rj.index(), ri.index());
                }
            }
        }
        if !merged.is_empty() {
            // Rewrite all point refs in constraints
            let remap = |r: &mut Ref<Point>| {
                if let Some(canonical) = merged.get(&r.index()) { *r = *canonical; }
            };
            for c in &mut self.coincident_pp { remap(&mut c.a); remap(&mut c.b); }
            for c in &mut self.coincident_lp1 { remap(&mut c.point); }
            for c in &mut self.coincident_lp2 { remap(&mut c.point); }
            for c in &mut self.distance_pp { remap(&mut c.a); remap(&mut c.b); }
            for c in &mut self.hdistance_pp { remap(&mut c.a); remap(&mut c.b); }
            for c in &mut self.vdistance_pp { remap(&mut c.a); remap(&mut c.b); }
            for c in &mut self.point_on_line { remap(&mut c.point); }
            for c in &mut self.midpoint { remap(&mut c.point); }
            for c in &mut self.point_on_arc { remap(&mut c.point); }
            for c in &mut self.distance_pl { remap(&mut c.point); }
            for c in &mut self.coincident_arc_center { remap(&mut c.point); }
            for c in &mut self.coincident_arc_start { remap(&mut c.point); }
            for c in &mut self.coincident_arc_end { remap(&mut c.point); }
            // Remove merged points
            for (old, _) in &merged { self.points.remove(Ref::new(*old)); }
            // Dedup again after remapping
            self.dedup_constraints();
        }

        // Phase 2: Replace helper-point bridges with direct constraints
        let helper_refs: std::vec::Vec<Ref<Point>> = self.points.refs()
            .filter(|r| self.points[*r].helper)
            .collect();
        for hr in &helper_refs {
            let hr = *hr;
            let lp1: Option<Ref<Line>> = self.coincident_lp1.iter().find(|c| c.point == hr).map(|c| c.line);
            let lp2: Option<Ref<Line>> = self.coincident_lp2.iter().find(|c| c.point == hr).map(|c| c.line);
            let ac: Option<Ref<Arc>> = self.coincident_arc_center.iter().find(|c| c.point == hr).map(|c| c.arc);
            let a_start: Option<Ref<Arc>> = self.coincident_arc_start.iter().find(|c| c.point == hr).map(|c| c.arc);
            let a_end: Option<Ref<Arc>> = self.coincident_arc_end.iter().find(|c| c.point == hr).map(|c| c.arc);

            macro_rules! consolidate {
                ($line_opt:expr, $arc_opt:expr, $lp_coll:ident, $arc_coll:ident, $direct_coll:ident, $DirectType:ident, $label:expr) => {
                    if let (Some(line), Some(arc)) = ($line_opt, $arc_opt) {
                        self.$direct_coll.push($DirectType { line, arc, hb: CrossBlock::new() });
                        self.$lp_coll.retain(|c| !(c.line == line && c.point == hr));
                        self.$arc_coll.retain(|c| !(c.point == hr && c.arc == arc));
                        eprintln!("INFO: consolidated helper {} -> {}", hr.index(), $label);
                    }
                };
            }
            consolidate!(lp1, ac, coincident_lp1, coincident_arc_center, coincident_lp1_arc_center, CoincidentLP1ArcCenter, "LP1ArcCenter");
            consolidate!(lp2, ac, coincident_lp2, coincident_arc_center, coincident_lp2_arc_center, CoincidentLP2ArcCenter, "LP2ArcCenter");
            consolidate!(lp1, a_start, coincident_lp1, coincident_arc_start, coincident_lp1_arc_start, CoincidentLP1ArcStart, "LP1ArcStart");
            consolidate!(lp2, a_start, coincident_lp2, coincident_arc_start, coincident_lp2_arc_start, CoincidentLP2ArcStart, "LP2ArcStart");
            consolidate!(lp1, a_end, coincident_lp1, coincident_arc_end, coincident_lp1_arc_end, CoincidentLP1ArcEnd, "LP1ArcEnd");
            consolidate!(lp2, a_end, coincident_lp2, coincident_arc_end, coincident_lp2_arc_end, CoincidentLP2ArcEnd, "LP2ArcEnd");
        }
        self.cleanup_helper_points();
        self.dedup_constraints();
    }

    /// Solve the sketch constraints using Levenberg-Marquardt.
    /// Uses sparse faer Cholesky for n > 64 params, dense Cholesky otherwise.
    pub fn solve(&mut self) -> arael::simple_lm::LmResult<f64> {
        let mut params64: std::vec::Vec<f64> = std::vec::Vec::new();
        self.serialize64(&mut params64);
        let n = params64.len();

        let config = arael::simple_lm::LmConfig::<f64> {
            verbose: false,
            ..Default::default()
        };
        let result = if n >= 64 {
            arael::simple_lm::solve_sparse_faer(&params64, self, &config)
        } else {
            arael::simple_lm::solve(&params64, self, &config)
        };
        self.deserialize64(&result.x);
        result
    }
}
