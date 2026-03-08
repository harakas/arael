// Interactive 2D sketch editor with real-time constraint solving.
//
// Tools: Select (drag to solve), Point, Line
// Constraints: Horizontal, Vertical, Coincident (via toolbar)
// Navigation: scroll wheel = zoom, middle mouse drag = pan

use eframe::egui;
use arael::model::{Param, CrossBlock};
use arael::vect::vect2d;
use arael::refs::Ref;
use arael_sketch::*;
use arael_sketch::{Dimension, DimensionKind, DimensionEndpoint};

/// Spawn an async task without blocking the UI thread.
/// On WASM: uses wasm_bindgen_futures. On native: uses std::thread.
#[cfg(target_arch = "wasm32")]
fn spawn_async<F: std::future::Future<Output = ()> + 'static>(f: F) {
    wasm_bindgen_futures::spawn_local(f);
}

#[cfg(not(target_arch = "wasm32"))]
fn spawn_async<F: std::future::Future<Output = ()> + Send + 'static>(f: F) {
    std::thread::spawn(move || pollster::block_on(f));
}

// ---------------------------------------------------------------------------
// Color scheme
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct ColorScheme {
    background: egui::Color32,
    grid: egui::Color32,
    origin: egui::Color32,
    line: egui::Color32,
    line_selected: egui::Color32,
    endpoint: egui::Color32,
    endpoint_selected: egui::Color32,
    endpoint_line_selected: egui::Color32,
    point: egui::Color32,
    point_selected: egui::Color32,
    point_locked: egui::Color32,
    arc: egui::Color32,
    constraint_marker: egui::Color32,
    preview_line: egui::Color32,
    cursor_crosshair: egui::Color32,
    status_text: egui::Color32,
}

impl ColorScheme {
    fn light() -> Self {
        ColorScheme {
            background: egui::Color32::from_gray(255),
            grid: egui::Color32::from_gray(220),
            origin: egui::Color32::from_rgb(180, 180, 180),
            line: egui::Color32::from_rgb(40, 40, 40),
            line_selected: egui::Color32::from_rgb(255, 140, 0),
            endpoint: egui::Color32::from_rgb(100, 180, 255),
            endpoint_selected: egui::Color32::from_rgb(255, 140, 0),
            endpoint_line_selected: egui::Color32::from_rgb(255, 220, 100),
            point: egui::Color32::from_rgb(100, 180, 255),
            point_selected: egui::Color32::from_rgb(255, 140, 0),
            point_locked: egui::Color32::from_rgb(0, 180, 0),
            arc: egui::Color32::from_rgb(40, 40, 40),
            constraint_marker: egui::Color32::from_rgb(0, 160, 0),
            preview_line: egui::Color32::from_rgba_premultiplied(40, 40, 40, 128),
            cursor_crosshair: egui::Color32::from_rgba_premultiplied(0, 0, 0, 40),
            status_text: egui::Color32::from_gray(80),
        }
    }

    fn dark() -> Self {
        ColorScheme {
            background: egui::Color32::from_gray(30),
            grid: egui::Color32::from_gray(45),
            origin: egui::Color32::from_rgb(80, 80, 80),
            line: egui::Color32::from_rgb(200, 200, 200),
            line_selected: egui::Color32::from_rgb(255, 180, 50),
            endpoint: egui::Color32::from_rgb(100, 180, 255),
            endpoint_selected: egui::Color32::from_rgb(255, 180, 50),
            endpoint_line_selected: egui::Color32::from_rgb(255, 220, 100),
            point: egui::Color32::from_rgb(100, 180, 255),
            point_selected: egui::Color32::from_rgb(255, 180, 50),
            point_locked: egui::Color32::from_rgb(0, 200, 0),
            arc: egui::Color32::from_rgb(200, 200, 200),
            constraint_marker: egui::Color32::from_rgb(100, 255, 100),
            preview_line: egui::Color32::from_rgba_premultiplied(200, 200, 200, 128),
            cursor_crosshair: egui::Color32::from_rgba_premultiplied(255, 255, 255, 60),
            status_text: egui::Color32::from_gray(150),
        }
    }
}

// ---------------------------------------------------------------------------
// What the user can grab and drag
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum GrabTarget {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
}

// ---------------------------------------------------------------------------
// Selection -- what entity is selected for constraint application
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Selection {
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

// ---------------------------------------------------------------------------
// Constraint type for constraint mode
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum ConstraintType {
    Horizontal,
    Vertical,
    Coincident,
    Parallel,
    Perpendicular,
    EqualLength,
    Tangent,
    Lock,
    ToggleStyle,
}

impl ConstraintType {
    #[allow(dead_code)]
    fn name(self) -> &'static str {
        match self {
            ConstraintType::Horizontal => "Horizontal",
            ConstraintType::Vertical => "Vertical",
            ConstraintType::Coincident => "Coincident",
            ConstraintType::Parallel => "Parallel",
            ConstraintType::Perpendicular => "Perpendicular",
            ConstraintType::EqualLength => "Equal",
            ConstraintType::Tangent => "Tangent",
            ConstraintType::Lock => "Lock",
            ConstraintType::ToggleStyle => "Style",
        }
    }
}

// ---------------------------------------------------------------------------
// Active tool
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Tool {
    Select,
    DrawPoint,
    DrawLine,
    DrawCircle,
    DrawArc,
    ConstraintMode(ConstraintType),
    Dimension,
}

// ---------------------------------------------------------------------------
// Delete target
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
#[allow(dead_code)]
enum DeleteTarget {
    Point(Ref<Point>),
    Line(Ref<Line>),
    Arc(Ref<Arc>),
}

// ---------------------------------------------------------------------------
// In-progress line drawing state
// ---------------------------------------------------------------------------

struct LineDrawState {
    start: vect2d,
    // What the start point snapped to (for auto-coincident on completion)
    snap_start: Option<SnapTarget>,
}

struct CircleDrawState {
    center: vect2d,
    snap_center: Option<SnapTarget>,
}

struct ArcDrawState {
    start: vect2d,
    snap_start: Option<SnapTarget>,
    end: Option<(vect2d, Option<SnapTarget>)>,  // None until second click
}

// ---------------------------------------------------------------------------
// Constraint identification (for selection and deletion)
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum CoincidentKind {
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

#[derive(Clone, Copy, PartialEq)]
enum ConstraintId {
    Horizontal(Ref<Line>),
    Vertical(Ref<Line>),
    Parallel(usize),
    Perpendicular(usize),
    EqualLength(usize),
    EqualRadius(usize),
    TangentLA(usize),
    TangentAA(usize),
    Coincident(CoincidentKind, usize),
    HelperBridge(Ref<Point>),  // helper point bridging two constraints
}

// Constraint symbol types (drawn with painter, not text)
#[derive(Clone, Copy)]
enum ConstraintSymbol {
    H,           // Horizontal
    V,           // Vertical
    Parallel,    // ||
    Perpendicular, // upside-down T
    Equal,       // =
    Tangent,     // T
    Coincident,  // corner with dot
}

// A drawn constraint marker with screen position
struct ConstraintMarker {
    pos: egui::Pos2,
    symbol: ConstraintSymbol,
    id: ConstraintId,
}

// Which point on an arc we're referring to
#[derive(Clone, Copy)]
enum ArcPoint { Center, Start, End }

// What a point/endpoint snapped to
#[derive(Clone, Copy)]
enum SnapTarget {
    Point(Ref<Point>),
    LineP1(Ref<Line>),
    LineP2(Ref<Line>),
    Line(Ref<Line>),  // on line body (not endpoint)
    ArcCenter(Ref<Arc>),
    ArcStart(Ref<Arc>),
    ArcEnd(Ref<Arc>),
    ArcBody(Ref<Arc>),  // on arc/circle curve
}

// ---------------------------------------------------------------------------
// Action log for undo/redo
// ---------------------------------------------------------------------------

#[derive(Clone, serde::Serialize, serde::Deserialize)]
enum Action {
    AddPoint { pos: vect2d },
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
    ApplyPerpendicular { a: Ref<Line>, b: Ref<Line> },
    ApplyEqualLength { a: Ref<Line>, b: Ref<Line> },
    AddCircle { center: vect2d, edge: vect2d },
    AddArc { start: vect2d, end: vect2d, mid: vect2d, swapped: bool },
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
    ToggleStyleLine { line: Ref<Line> },
    ToggleStyleArc { arc: Ref<Arc> },
    DeleteArc { arc: Ref<Arc> },
    AddDimension { kind: DimensionKind, value: f64 },
    RemoveDimension { index: usize },
    // Drag is non-deterministic; store full state after drag completes
    Drag { snapshot: Vec<u8> },
}

// Resolve a DimensionEndpoint to a Ref<Point>, creating helper point + constraint if needed.
fn resolve_dim_endpoint(sketch: &mut Sketch, ep: &DimensionEndpoint) -> Ref<Point> {
    match *ep {
        DimensionEndpoint::Point(r) => r,
        DimensionEndpoint::LineP1(r) => {
            let pos = sketch.lines[r].p1.value;
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_lp1.push(CoincidentLP1 { line: r, point: hp, hb: CrossBlock::new() });
            hp
        }
        DimensionEndpoint::LineP2(r) => {
            let pos = sketch.lines[r].p2.value;
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_lp2.push(CoincidentLP2 { line: r, point: hp, hb: CrossBlock::new() });
            hp
        }
        DimensionEndpoint::ArcCenter(r) => {
            let pos = sketch.arcs[r].center.value;
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_arc_center.push(CoincidentArcCenter { point: hp, arc: r, hb: CrossBlock::new() });
            hp
        }
        DimensionEndpoint::ArcStart(r) => {
            let pos = arc_start_pos_sketch(sketch, r);
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_arc_start.push(CoincidentArcStart { point: hp, arc: r, hb: CrossBlock::new() });
            hp
        }
        DimensionEndpoint::ArcEnd(r) => {
            let pos = arc_end_pos_sketch(sketch, r);
            let hp = sketch.add_helper_point(pos);
            sketch.coincident_arc_end.push(CoincidentArcEnd { point: hp, arc: r, hb: CrossBlock::new() });
            hp
        }
    }
}

fn arc_start_pos_sketch(sketch: &Sketch, r: Ref<Arc>) -> vect2d {
    let a = &sketch.arcs[r];
    vect2d::new(
        a.center.value.x + a.radius.value * a.start_angle.value.cos(),
        a.center.value.y + a.radius.value * a.start_angle.value.sin(),
    )
}

fn arc_end_pos_sketch(sketch: &Sketch, r: Ref<Arc>) -> vect2d {
    let a = &sketch.arcs[r];
    vect2d::new(
        a.center.value.x + a.radius.value * a.end_angle.value.cos(),
        a.center.value.y + a.radius.value * a.end_angle.value.sin(),
    )
}

impl Action {
    fn apply(&self, sketch: &mut Sketch) {
        match self {
            Action::AddPoint { pos } => { sketch.add_point(*pos); }
            Action::AddHelperPoint { pos } => { sketch.add_helper_point(*pos); }
            Action::AddLine { p1, p2 } => { sketch.add_line(*p1, *p2); }
            Action::AddCircle { center, edge } => {
                let r = ((edge.x - center.x).powi(2) + (edge.y - center.y).powi(2)).sqrt();
                sketch.add_arc(*center, r, 0.0, std::f64::consts::TAU, true);
                sketch.solve(); // anchor drift before constraints are added
            }
            Action::AddArc { start, end, mid, .. } => {
                if let Some((c, r, sa, ea, _)) = circumscribed_arc(*start, *end, *mid) {
                    sketch.add_arc(c, r, sa, ea, false);
                    sketch.solve(); // anchor drift before constraints are added
                }
            }
            Action::ApplyHorizontal { lines } => {
                for r in lines { sketch.lines[*r].constraints.horizontal = true; }
                sketch.solve();
            }
            Action::ApplyVertical { lines } => {
                for r in lines { sketch.lines[*r].constraints.vertical = true; }
                sketch.solve();
            }
            Action::ApplyCoincidentPP { a, b } => {
                sketch.coincident_pp.push(CoincidentPP { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentLL11 { a, b } => {
                sketch.coincident_ll11.push(CoincidentLL11 { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentLL12 { a, b } => {
                sketch.coincident_ll12.push(CoincidentLL12 { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentLL21 { a, b } => {
                sketch.coincident_ll21.push(CoincidentLL21 { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentLL22 { a, b } => {
                sketch.coincident_ll22.push(CoincidentLL22 { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentLP1 { line, point } => {
                sketch.coincident_lp1.push(CoincidentLP1 { line: *line, point: *point, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentLP2 { line, point } => {
                sketch.coincident_lp2.push(CoincidentLP2 { line: *line, point: *point, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyParallel { a, b } => {
                sketch.parallel.push(Parallel { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyPerpendicular { a, b } => {
                sketch.perpendicular.push(Perpendicular { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyEqualLength { a, b } => {
                sketch.equal_length.push(EqualLength { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentArcCenter { point, arc } => {
                sketch.coincident_arc_center.push(CoincidentArcCenter { point: *point, arc: *arc, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentArcStart { point, arc } => {
                sketch.coincident_arc_start.push(CoincidentArcStart { point: *point, arc: *arc, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentArcEnd { point, arc } => {
                sketch.coincident_arc_end.push(CoincidentArcEnd { point: *point, arc: *arc, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyConcentric { a, b } => {
                sketch.concentric.push(Concentric { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyCoincidentLP1ArcCenter { line, arc } => {
                sketch.coincident_lp1_arc_center.push(CoincidentLP1ArcCenter { line: *line, arc: *arc, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentLP2ArcCenter { line, arc } => {
                sketch.coincident_lp2_arc_center.push(CoincidentLP2ArcCenter { line: *line, arc: *arc, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentLP1ArcStart { line, arc } => {
                sketch.coincident_lp1_arc_start.push(CoincidentLP1ArcStart { line: *line, arc: *arc, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentLP2ArcStart { line, arc } => {
                sketch.coincident_lp2_arc_start.push(CoincidentLP2ArcStart { line: *line, arc: *arc, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentLP1ArcEnd { line, arc } => {
                sketch.coincident_lp1_arc_end.push(CoincidentLP1ArcEnd { line: *line, arc: *arc, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentLP2ArcEnd { line, arc } => {
                sketch.coincident_lp2_arc_end.push(CoincidentLP2ArcEnd { line: *line, arc: *arc, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentArcCenterStart { a, b } => {
                sketch.coincident_arc_center_start.push(CoincidentArcCenterStart { a: *a, b: *b, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentArcCenterEnd { a, b } => {
                sketch.coincident_arc_center_end.push(CoincidentArcCenterEnd { a: *a, b: *b, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentArcStartCenter { a, b } => {
                sketch.coincident_arc_start_center.push(CoincidentArcStartCenter { a: *a, b: *b, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentArcEndCenter { a, b } => {
                sketch.coincident_arc_end_center.push(CoincidentArcEndCenter { a: *a, b: *b, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentArcStartStart { a, b } => {
                sketch.coincident_arc_start_start.push(CoincidentArcStartStart { a: *a, b: *b, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentArcStartEnd { a, b } => {
                sketch.coincident_arc_start_end.push(CoincidentArcStartEnd { a: *a, b: *b, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentArcEndStart { a, b } => {
                sketch.coincident_arc_end_start.push(CoincidentArcEndStart { a: *a, b: *b, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyCoincidentArcEndEnd { a, b } => {
                sketch.coincident_arc_end_end.push(CoincidentArcEndEnd { a: *a, b: *b, hb: CrossBlock::new() }); sketch.solve();
            }
            Action::ApplyLineP1OnArc { line, arc } => {
                sketch.line_p1_on_arc.push(LineP1OnArc { line: *line, arc: *arc, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyLineP2OnArc { line, arc } => {
                sketch.line_p2_on_arc.push(LineP2OnArc { line: *line, arc: *arc, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyEqualRadius { a, b } => {
                sketch.equal_radius.push(EqualRadius { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyTangentLA { line, arc } => {
                sketch.tangent_la.push(TangentLA { line: *line, arc: *arc, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyTangentAA { a, b } => {
                sketch.tangent_aa.push(TangentAA { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyPointOnLine { point, line } => {
                sketch.point_on_line.push(PointOnLine { point: *point, line: *line, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyPointOnArc { point, arc } => {
                sketch.point_on_arc.push(PointOnArc { point: *point, arc: *arc, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyLineP1OnLine { a, b } => {
                sketch.line_p1_on_line.push(LineP1OnLine { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyLineP2OnLine { a, b } => {
                sketch.line_p2_on_line.push(LineP2OnLine { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::LockPoint { point, pos } => {
                let p = &mut sketch.points[*point];
                p.constraints.has_fix_x = true;
                p.constraints.fix_x = pos.x;
                p.constraints.has_fix_y = true;
                p.constraints.fix_y = pos.y;
                sketch.solve();
            }
            Action::UnlockPoint { point } => {
                let p = &mut sketch.points[*point];
                p.constraints.has_fix_x = false;
                p.constraints.has_fix_y = false;
                sketch.solve();
            }
            Action::LockLineP1 { line, pos } => {
                sketch.lines[*line].p1 = Param::fixed(*pos);
                sketch.solve();
            }
            Action::UnlockLineP1 { line } => {
                let val = sketch.lines[*line].p1.value;
                sketch.lines[*line].p1 = Param::new(val);
                sketch.solve();
            }
            Action::LockLineP2 { line, pos } => {
                sketch.lines[*line].p2 = Param::fixed(*pos);
                sketch.solve();
            }
            Action::UnlockLineP2 { line } => {
                let val = sketch.lines[*line].p2.value;
                sketch.lines[*line].p2 = Param::new(val);
                sketch.solve();
            }
            Action::LockArcCenter { arc, pos } => {
                sketch.arcs[*arc].center = Param::fixed(*pos);
                sketch.solve();
            }
            Action::UnlockArcCenter { arc } => {
                let val = sketch.arcs[*arc].center.value;
                sketch.arcs[*arc].center = Param::new(val);
                sketch.solve();
            }
            Action::DeletePoint { point } => {
                sketch.delete_point(*point);
                sketch.solve();
            }
            Action::DeleteLine { line } => {
                sketch.delete_line(*line);
                sketch.solve();
            }
            Action::ToggleStyleLine { line } => {
                sketch.lines[*line].style = sketch.lines[*line].style.next();
            }
            Action::ToggleStyleArc { arc } => {
                sketch.arcs[*arc].style = sketch.arcs[*arc].style.next();
            }
            Action::DeleteArc { arc } => {
                sketch.delete_arc(*arc);
                sketch.solve();
            }
            Action::AddDimension { kind, value } => {
                let name = format!("D{}", sketch.next_dimension_id);
                sketch.next_dimension_id += 1;
                // Apply the constraint
                match kind {
                    DimensionKind::LineLength(line) => {
                        sketch.lines[*line].constraints.has_length = true;
                        sketch.lines[*line].constraints.length = *value;
                    }
                    DimensionKind::PointPointDistance(a, b) => {
                        match (a, b) {
                            (DimensionEndpoint::Point(pa), DimensionEndpoint::Point(pb)) => {
                                sketch.distance_pp.push(DistancePP { a: *pa, b: *pb, distance: *value, hb: CrossBlock::new() });
                            }
                            // Line-Line combinations
                            (DimensionEndpoint::LineP1(la), DimensionEndpoint::LineP1(lb)) => {
                                sketch.distance_ll11.push(DistanceLL11 { a: *la, b: *lb, distance: *value, hb: CrossBlock::new() });
                            }
                            (DimensionEndpoint::LineP1(la), DimensionEndpoint::LineP2(lb)) => {
                                sketch.distance_ll12.push(DistanceLL12 { a: *la, b: *lb, distance: *value, hb: CrossBlock::new() });
                            }
                            (DimensionEndpoint::LineP2(la), DimensionEndpoint::LineP1(lb)) => {
                                sketch.distance_ll21.push(DistanceLL21 { a: *la, b: *lb, distance: *value, hb: CrossBlock::new() });
                            }
                            (DimensionEndpoint::LineP2(la), DimensionEndpoint::LineP2(lb)) => {
                                sketch.distance_ll22.push(DistanceLL22 { a: *la, b: *lb, distance: *value, hb: CrossBlock::new() });
                            }
                            // Line-Point combinations
                            (DimensionEndpoint::LineP1(l), DimensionEndpoint::Point(p))
                            | (DimensionEndpoint::Point(p), DimensionEndpoint::LineP1(l)) => {
                                sketch.distance_lp1.push(DistanceLP1 { line: *l, point: *p, distance: *value, hb: CrossBlock::new() });
                            }
                            (DimensionEndpoint::LineP2(l), DimensionEndpoint::Point(p))
                            | (DimensionEndpoint::Point(p), DimensionEndpoint::LineP2(l)) => {
                                sketch.distance_lp2.push(DistanceLP2 { line: *l, point: *p, distance: *value, hb: CrossBlock::new() });
                            }
                            // Fallback: use helper points for arc endpoints and other combos
                            _ => {
                                let pa = resolve_dim_endpoint(sketch, a);
                                let pb = resolve_dim_endpoint(sketch, b);
                                sketch.distance_pp.push(DistancePP { a: pa, b: pb, distance: *value, hb: CrossBlock::new() });
                            }
                        }
                    }
                    DimensionKind::PointLineDistance(pt, line) => {
                        // Compute signed distance to preserve side
                        let compute_signed = |sketch: &Sketch, pt_pos: vect2d, line: Ref<Line>| -> f64 {
                            let l = &sketch.lines[line];
                            let ldx = l.p2.value.x - l.p1.value.x;
                            let ldy = l.p2.value.y - l.p1.value.y;
                            let len = (ldx * ldx + ldy * ldy).sqrt();
                            if len < 1e-12 { return *value; }
                            let sign = ((pt_pos.x - l.p1.value.x) * ldy - (pt_pos.y - l.p1.value.y) * ldx) / len;
                            if sign >= 0.0 { *value } else { -*value }
                        };
                        match pt {
                            DimensionEndpoint::Point(p) => {
                                let signed = compute_signed(sketch, sketch.points[*p].pos.value, *line);
                                sketch.distance_pl.push(DistancePL { point: *p, line: *line, distance: signed, hb: CrossBlock::new() });
                            }
                            _ => {
                                let p = resolve_dim_endpoint(sketch, pt);
                                let signed = compute_signed(sketch, sketch.points[p].pos.value, *line);
                                sketch.distance_pl.push(DistancePL { point: p, line: *line, distance: signed, hb: CrossBlock::new() });
                            }
                        }
                    }
                    DimensionKind::ArcRadius(arc) => {
                        sketch.arcs[*arc].constraints.has_target_radius = true;
                        sketch.arcs[*arc].constraints.target_radius = *value;
                    }
                }
                sketch.dimensions.push(Dimension {
                    kind: *kind, value: *value,
                    offset: vect2d::new(0.0, 1.0),
                    text_along: 0.0,
                    name,
                });
                sketch.solve();
            }
            Action::RemoveDimension { index } => {
                if *index < sketch.dimensions.len() {
                    let dim = sketch.dimensions.remove(*index);
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
                        DimensionKind::PointPointDistance(a, b) => {
                            let val = dim.value;
                            match (a, b) {
                                (DimensionEndpoint::Point(pa), DimensionEndpoint::Point(pb)) => {
                                    sketch.distance_pp.retain(|c| !(c.a == pa && c.b == pb && (c.distance - val).abs() < 1e-9));
                                }
                                (DimensionEndpoint::LineP1(la), DimensionEndpoint::LineP1(lb)) => {
                                    sketch.distance_ll11.retain(|c| !(c.a == la && c.b == lb && (c.distance - val).abs() < 1e-9));
                                }
                                (DimensionEndpoint::LineP1(la), DimensionEndpoint::LineP2(lb)) => {
                                    sketch.distance_ll12.retain(|c| !(c.a == la && c.b == lb && (c.distance - val).abs() < 1e-9));
                                }
                                (DimensionEndpoint::LineP2(la), DimensionEndpoint::LineP1(lb)) => {
                                    sketch.distance_ll21.retain(|c| !(c.a == la && c.b == lb && (c.distance - val).abs() < 1e-9));
                                }
                                (DimensionEndpoint::LineP2(la), DimensionEndpoint::LineP2(lb)) => {
                                    sketch.distance_ll22.retain(|c| !(c.a == la && c.b == lb && (c.distance - val).abs() < 1e-9));
                                }
                                (DimensionEndpoint::LineP1(l), DimensionEndpoint::Point(p))
                                | (DimensionEndpoint::Point(p), DimensionEndpoint::LineP1(l)) => {
                                    sketch.distance_lp1.retain(|c| !(c.line == l && c.point == p && (c.distance - val).abs() < 1e-9));
                                }
                                (DimensionEndpoint::LineP2(l), DimensionEndpoint::Point(p))
                                | (DimensionEndpoint::Point(p), DimensionEndpoint::LineP2(l)) => {
                                    sketch.distance_lp2.retain(|c| !(c.line == l && c.point == p && (c.distance - val).abs() < 1e-9));
                                }
                                _ => {
                                    // Helper point fallback: find by distance value
                                    if let Some(idx) = sketch.distance_pp.iter().position(|c| (c.distance - val).abs() < 1e-9) {
                                        sketch.distance_pp.remove(idx);
                                    }
                                }
                            }
                        }
                        DimensionKind::PointLineDistance(_, _) => {
                            if let Some(idx) = sketch.distance_pl.iter().position(|c| (c.distance.abs() - dim.value.abs()).abs() < 1e-9) {
                                sketch.distance_pl.remove(idx);
                            }
                        }
                    }
                    sketch.cleanup_helper_points();
                    sketch.solve();
                }
            }
            Action::Drag { snapshot } => {
                *sketch = bincode::deserialize(snapshot).unwrap();
                sketch.solve();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// History for undo/redo
// ---------------------------------------------------------------------------

struct History {
    actions: Vec<Action>,
    snapshots: Vec<Vec<u8>>,  // bincode-serialized Sketch after each action
    groups: Vec<u32>,         // group id for each action
    cursor: usize,            // number of applied actions (0 = empty sketch)
    next_group: u32,
    current_group: u32,
}

impl History {
    fn new() -> Self {
        History {
            actions: Vec::new(), snapshots: Vec::new(), groups: Vec::new(),
            cursor: 0, next_group: 0, current_group: 0,
        }
    }

    fn begin_group(&mut self) {
        self.current_group = self.next_group;
        self.next_group += 1;
    }

    fn push(&mut self, action: Action, sketch: &Sketch) {
        // Truncate any redo tail
        self.actions.truncate(self.cursor);
        self.snapshots.truncate(self.cursor);
        self.groups.truncate(self.cursor);
        // Push new
        self.actions.push(action);
        self.snapshots.push(bincode::serialize(sketch).unwrap());
        self.groups.push(self.current_group);
        self.cursor += 1;
    }

    fn can_undo(&self) -> bool { self.cursor > 0 }
    fn can_redo(&self) -> bool { self.cursor < self.actions.len() }

    fn undo(&mut self) -> Option<Sketch> {
        if self.cursor == 0 { return None; }
        // Find the start of the current group
        let group = self.groups[self.cursor - 1];
        while self.cursor > 0 && self.groups[self.cursor - 1] == group {
            self.cursor -= 1;
        }
        if self.cursor == 0 {
            Some(Sketch::new())
        } else {
            let mut sketch: Sketch = bincode::deserialize(&self.snapshots[self.cursor - 1]).unwrap();
            sketch.solve();
            Some(sketch)
        }
    }

    fn redo(&mut self) -> Option<Sketch> {
        if self.cursor >= self.actions.len() { return None; }
        // Find the end of the next group
        let group = self.groups[self.cursor];
        while self.cursor < self.actions.len() && self.groups[self.cursor] == group {
            self.cursor += 1;
        }
        let mut sketch: Sketch = bincode::deserialize(&self.snapshots[self.cursor - 1]).unwrap();
        sketch.solve();
        Some(sketch)
    }
}

// ---------------------------------------------------------------------------
// Editor state
// ---------------------------------------------------------------------------

struct EditorApp {
    sketch: Sketch,
    // View transform
    offset: egui::Vec2,  // pan offset in screen pixels
    scale: f32,          // pixels per sketch unit

    // Tools
    tool: Tool,
    line_draw: Option<LineDrawState>,
    circle_draw: Option<CircleDrawState>,
    arc_draw: Option<ArcDrawState>,

    // Selection
    selection: Vec<Selection>,

    // Drag state
    grab: Option<GrabTarget>,
    drag_point: Option<Ref<Point>>,  // temporary drag point
    drag_dimension: Option<usize>,   // index of dimension being dragged

    // Undo/redo
    history: History,

    // Constraint markers (rebuilt each frame for hit testing)
    constraint_markers: Vec<ConstraintMarker>,

    // File
    pending_load: std::sync::Arc<std::sync::Mutex<Option<String>>>,
    pending_fit: bool,

    // Dimension input
    dim_input: String,          // text being typed for dimension value
    dim_editing: bool,          // true when text input is active
    dim_kind: Option<DimensionKind>, // what dimension is being created
    dim_placing: bool,          // true when positioning the dimension with mouse
    dim_offset: vect2d,         // current offset being placed
    dim_text_along: f64,        // text position along line during creation
    dim_edit_index: Option<usize>, // index of dimension being edited (for double-click edit)

    // Display
    show_constraints: bool,

    // Theme
    dark_mode: bool,
    colors: ColorScheme,
}

impl EditorApp {
    fn demo() -> Self {
        let mut sketch = Sketch::new();
        let history = History::new();

        // Isoceles triangle with a circle at the apex
        // L0: apex (0,0) -> bottom-left (-1.5,-5)
        // L1: bottom-left -> bottom-right (1.5,-5), length=3
        // L2: bottom-right -> apex
        let l0 = sketch.add_line(vect2d::new(0.0, 0.0), vect2d::new(-1.5, -5.0));
        let l1 = sketch.add_line(vect2d::new(-1.5, -5.0), vect2d::new(1.5, -5.0));
        let l2 = sketch.add_line(vect2d::new(1.5, -5.0), vect2d::new(0.0, 0.0));

        // Connect corners: L0.p2=L1.p1, L1.p2=L2.p1, L2.p2=L0.p1
        sketch.coincident_ll21.push(CoincidentLL21 { a: l0, b: l1, hb: CrossBlock::new() });
        sketch.coincident_ll21.push(CoincidentLL21 { a: l1, b: l2, hb: CrossBlock::new() });
        sketch.coincident_ll21.push(CoincidentLL21 { a: l2, b: l0, hb: CrossBlock::new() });



        // Point at apex + circle (full arc, r=1.5) centered there
        let p0 = sketch.add_point(vect2d::new(0.0, 0.0));
        let a0 = sketch.add_arc(vect2d::new(0.0, 0.0), 1.5, 0.0, std::f64::consts::TAU, true);

        // Equal length: L0 = L2 (isoceles)
        sketch.equal_length.push(EqualLength { a: l2, b: l0, hb: CrossBlock::new() });

        // Arc center = L0.p1 (apex), point = L0.p1
        sketch.coincident_lp1.push(CoincidentLP1 { line: l0, point: p0, hb: CrossBlock::new() });
        sketch.coincident_lp1_arc_center.push(CoincidentLP1ArcCenter { line: l0, arc: a0, hb: CrossBlock::new() });

        // Dimensions (via Action, then adjust offsets for nice layout)
        Action::AddDimension { kind: DimensionKind::LineLength(l1), value: 3.0 }.apply(&mut sketch);
        sketch.dimensions.last_mut().unwrap().offset = vect2d::new(0.0, -0.32);
        sketch.dimensions.last_mut().unwrap().text_along = -0.27;

        Action::AddDimension {
            kind: DimensionKind::PointLineDistance(DimensionEndpoint::LineP1(l0), l1), value: 5.0,
        }.apply(&mut sketch);
        sketch.dimensions.last_mut().unwrap().offset = vect2d::new(0.0, 1.72);
        sketch.dimensions.last_mut().unwrap().text_along = 0.20;

        Action::AddDimension { kind: DimensionKind::ArcRadius(a0), value: 1.5 }.apply(&mut sketch);
        sketch.dimensions.last_mut().unwrap().offset = vect2d::new(0.91, 0.0);

        sketch.solve();

        EditorApp {
            sketch,
            offset: egui::Vec2::new(400.0, 300.0),
            scale: 80.0,
            tool: Tool::Select,
            line_draw: None,
            circle_draw: None,
            arc_draw: None,
            selection: Vec::new(),
            grab: None,
            drag_point: None,
            drag_dimension: None,
            history,
            constraint_markers: Vec::new(),
            pending_load: std::sync::Arc::new(std::sync::Mutex::new(None)),
            pending_fit: true,
            dim_input: String::new(),
            dim_editing: false,
            dim_kind: None,
            dim_placing: false,
            dim_offset: vect2d::new(0.0, 1.0),
            dim_text_along: 0.0,
            dim_edit_index: None,
            show_constraints: true,
            dark_mode: cfg!(target_arch = "wasm32"),
            colors: if cfg!(target_arch = "wasm32") { ColorScheme::dark() } else { ColorScheme::light() },
        }
    }
}

impl Default for EditorApp {
    fn default() -> Self {
        Self::demo()
    }
}

impl EditorApp {
    // Sketch coords -> screen coords
    fn to_screen(&self, p: vect2d) -> egui::Pos2 {
        egui::Pos2::new(
            p.x as f32 * self.scale + self.offset.x,
            -p.y as f32 * self.scale + self.offset.y, // y flipped
        )
    }

    // Screen coords -> sketch coords
    fn to_sketch(&self, p: egui::Pos2) -> vect2d {
        vect2d::new(
            ((p.x - self.offset.x) / self.scale) as f64,
            (-(p.y - self.offset.y) / self.scale) as f64,
        )
    }

    // Hit test: find nearest grabbable target within threshold
    fn hit_test(&self, sketch_pos: vect2d, threshold: f64) -> Option<GrabTarget> {
        let mut best: Option<(f64, GrabTarget)> = None;

        let mut check = |dist: f64, target: GrabTarget| {
            if dist < threshold {
                if best.is_none() || dist < best.unwrap().0 {
                    best = Some((dist, target));
                }
            }
        };

        // Points (skip helpers)
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let d = ((p.pos.value.x - sketch_pos.x).powi(2)
                   + (p.pos.value.y - sketch_pos.y).powi(2)).sqrt();
            check(d, GrabTarget::Point(r));
        }

        // Line endpoints (priority over line body)
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];

            let d1 = ((l.p1.value.x - sketch_pos.x).powi(2)
                    + (l.p1.value.y - sketch_pos.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - sketch_pos.x).powi(2)
                    + (l.p2.value.y - sketch_pos.y).powi(2)).sqrt();
            check(d1, GrabTarget::LineP1(r));
            check(d2, GrabTarget::LineP2(r));
        }

        // Arc centers and endpoints
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let dc = ((a.center.value.x - sketch_pos.x).powi(2)
                    + (a.center.value.y - sketch_pos.y).powi(2)).sqrt();
            check(dc, GrabTarget::ArcCenter(r));

            if !a.closed {
                let sp = arc_start_pos(a);
                let ep = arc_end_pos(a);
                let ds = ((sp.x - sketch_pos.x).powi(2) + (sp.y - sketch_pos.y).powi(2)).sqrt();
                let de = ((ep.x - sketch_pos.x).powi(2) + (ep.y - sketch_pos.y).powi(2)).sqrt();
                check(ds, GrabTarget::ArcStart(r));
                check(de, GrabTarget::ArcEnd(r));
            }
        }

        best.map(|(_, t)| t)
    }

    // Find selection target (entity near mouse)
    // Priority: Constraints > Points > Endpoints > Lines/Arcs
    fn hit_test_selection(&self, sketch_pos: vect2d, threshold: f64) -> Option<Selection> {
        // Constraint markers first (screen-space, pick closest)
        let screen_pos = self.to_screen(sketch_pos);
        let mut best_constraint: Option<(f32, ConstraintId)> = None;
        for marker in &self.constraint_markers {
            let dx = screen_pos.x - marker.pos.x;
            let dy = screen_pos.y - marker.pos.y;
            let d = (dx * dx + dy * dy).sqrt();
            if d < 10.0 {
                if best_constraint.is_none() || d < best_constraint.unwrap().0 {
                    best_constraint = Some((d, marker.id));
                }
            }
        }
        if let Some((_, id)) = best_constraint {
            return Some(Selection::Constraint(id));
        }

        // Then standalone points (skip helpers)
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let d = ((p.pos.value.x - sketch_pos.x).powi(2)
                   + (p.pos.value.y - sketch_pos.y).powi(2)).sqrt();
            if d < threshold { return Some(Selection::Point(r)); }
        }

        // Then line endpoints (pseudo-points)
        let mut best_ep: Option<(f64, Selection)> = None;
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];

            let d1 = ((l.p1.value.x - sketch_pos.x).powi(2)
                    + (l.p1.value.y - sketch_pos.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - sketch_pos.x).powi(2)
                    + (l.p2.value.y - sketch_pos.y).powi(2)).sqrt();
            if d1 < threshold {
                if best_ep.is_none() || d1 < best_ep.unwrap().0 {
                    best_ep = Some((d1, Selection::LineP1(r)));
                }
            }
            if d2 < threshold {
                if best_ep.is_none() || d2 < best_ep.unwrap().0 {
                    best_ep = Some((d2, Selection::LineP2(r)));
                }
            }
        }
        // Arc centers and endpoints (same priority as line endpoints)
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let dc = ((a.center.value.x - sketch_pos.x).powi(2)
                    + (a.center.value.y - sketch_pos.y).powi(2)).sqrt();
            if dc < threshold {
                if best_ep.is_none() || dc < best_ep.unwrap().0 {
                    best_ep = Some((dc, Selection::ArcCenter(r)));
                }
            }
            if !a.closed {
                let sp = arc_start_pos(a);
                let ep = arc_end_pos(a);
                let ds = ((sp.x - sketch_pos.x).powi(2) + (sp.y - sketch_pos.y).powi(2)).sqrt();
                let de = ((ep.x - sketch_pos.x).powi(2) + (ep.y - sketch_pos.y).powi(2)).sqrt();
                if ds < threshold {
                    if best_ep.is_none() || ds < best_ep.unwrap().0 {
                        best_ep = Some((ds, Selection::ArcStart(r)));
                    }
                }
                if de < threshold {
                    if best_ep.is_none() || de < best_ep.unwrap().0 {
                        best_ep = Some((de, Selection::ArcEnd(r)));
                    }
                }
            }
        }

        if let Some((_, sel)) = best_ep { return Some(sel); }

        // Dimension annotations (higher priority than line/arc bodies)
        // Check both the text segment AND the dimension arrow line
        let screen_pos = self.to_screen(sketch_pos);
        for (i, dim) in self.sketch.dimensions.iter().enumerate() {
            // Text segment
            let (ts, te) = self.dim_text_segment(dim);
            let dt = Self::screen_point_to_segment_dist(screen_pos, ts, te);
            // Arrow line segment
            let da = if matches!(dim.kind, DimensionKind::ArcRadius(_)) {
                dt // for radius, text check is enough
            } else {
                let (p1, p2) = self.dim_endpoints(&dim.kind);
                let dx = p2.x - p1.x;
                let dy = p2.y - p1.y;
                let len = (dx * dx + dy * dy).sqrt().max(1e-12);
                let nx = -dy / len;
                let ny = dx / len;
                let off = dim.offset.y;
                let q1 = vect2d::new(p1.x + nx * off, p1.y + ny * off);
                let q2 = vect2d::new(p2.x + nx * off, p2.y + ny * off);
                let sq1 = self.to_screen(q1);
                let sq2 = self.to_screen(q2);
                Self::screen_point_to_segment_dist(screen_pos, sq1, sq2)
            };
            if dt < 12.0 || da < 8.0 {
                return Some(Selection::Dimension(i));
            }
        }

        // Then lines (distance to segment)
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            let d = point_to_segment_dist(sketch_pos, l.p1.value, l.p2.value);
            if d < threshold { return Some(Selection::Line(r)); }
        }

        // Then arc/circle curves
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let (d, _) = point_to_arc_dist(sketch_pos, a);
            if d < threshold { return Some(Selection::Arc(r)); }
        }

        None
    }

    // Start dragging: create a temporary fixed point and coincident constraint
    fn start_drag(&mut self, target: GrabTarget, mouse_pos: vect2d) {
        // Create a fixed point at mouse position
        let drag_pt = self.sketch.add_point_fixed(mouse_pos);
        self.drag_point = Some(drag_pt);

        // Add coincident constraint between drag point and the grabbed target
        match target {
            GrabTarget::Point(r) => {
                self.sketch.coincident_pp.push(CoincidentPP {
                    a: drag_pt, b: r, hb: CrossBlock::new(),
                });
            }
            GrabTarget::LineP1(r) => {
                self.sketch.coincident_lp1.push(CoincidentLP1 {
                    line: r, point: drag_pt, hb: CrossBlock::new(),
                });
            }
            GrabTarget::LineP2(r) => {
                self.sketch.coincident_lp2.push(CoincidentLP2 {
                    line: r, point: drag_pt, hb: CrossBlock::new(),
                });
            }
            GrabTarget::ArcCenter(r) => {
                self.sketch.coincident_arc_center.push(CoincidentArcCenter {
                    point: drag_pt, arc: r, hb: CrossBlock::new(),
                });
            }
            GrabTarget::ArcStart(r) => {
                self.sketch.coincident_arc_start.push(CoincidentArcStart {
                    point: drag_pt, arc: r, hb: CrossBlock::new(),
                });
            }
            GrabTarget::ArcEnd(r) => {
                self.sketch.coincident_arc_end.push(CoincidentArcEnd {
                    point: drag_pt, arc: r, hb: CrossBlock::new(),
                });
            }
        }
        self.grab = Some(target);
    }

    // Update drag position and re-solve
    fn update_drag(&mut self, mouse_pos: vect2d) {
        if let Some(drag_pt) = self.drag_point {
            self.sketch.points[drag_pt].pos = Param::fixed(mouse_pos);
            self.sketch.solve();
        }
    }

    // End drag: remove temporary point and constraint, auto-snap, record action
    fn end_drag(&mut self, hit_threshold: f64) {
        self.begin_group();
        if let Some(drag_pt) = self.drag_point.take() {
            // Get final position of the dragged endpoint for snap detection
            let _drag_pos = self.sketch.points[drag_pt].pos.value;
            let grab = self.grab;

            // Remove the last coincident constraint (the drag one)
            match grab {
                Some(GrabTarget::Point(_)) => { self.sketch.coincident_pp.pop(); }
                Some(GrabTarget::LineP1(_)) => { self.sketch.coincident_lp1.pop(); }
                Some(GrabTarget::LineP2(_)) => { self.sketch.coincident_lp2.pop(); }
                Some(GrabTarget::ArcCenter(_)) => { self.sketch.coincident_arc_center.pop(); }
                Some(GrabTarget::ArcStart(_)) => { self.sketch.coincident_arc_start.pop(); }
                Some(GrabTarget::ArcEnd(_)) => { self.sketch.coincident_arc_end.pop(); }
                None => {}
            }
            // Remove the drag point and re-solve without the drag constraint
            self.sketch.points.remove(drag_pt);
            self.sketch.solve();

            // Record drag as a non-deterministic action with full state snapshot
            let snapshot = bincode::serialize(&self.sketch).unwrap();
            let action = Action::Drag { snapshot };
            self.history.push(action, &self.sketch);

            // Auto-snap: if a point-like entity was dragged near another,
            // create a coincident constraint
            match grab {
                Some(GrabTarget::LineP1(line) | GrabTarget::LineP2(line)) => {
                    let is_p1 = matches!(grab, Some(GrabTarget::LineP1(_)));
                    let ep_pos = if is_p1 {
                        self.sketch.lines[line].p1.value
                    } else {
                        self.sketch.lines[line].p2.value
                    };
                    if let Some((_, snap)) = self.find_snap_target_excluding(ep_pos, hit_threshold, Some(line)) {
                        self.apply_snap_coincident(snap, line, is_p1);
                    }
                }
                Some(GrabTarget::ArcCenter(arc)) => {
                    let pos = self.sketch.arcs[arc].center.value;
                    if let Some((_, snap)) = self.find_snap_target_ex(pos, hit_threshold, None, Some(arc)) {
                        self.apply_snap_coincident_arc(snap, arc, ArcPoint::Center, pos);
                    }
                }
                Some(GrabTarget::ArcStart(arc)) => {
                    let pos = arc_start_pos(&self.sketch.arcs[arc]);
                    if let Some((_, snap)) = self.find_snap_target_ex(pos, hit_threshold, None, Some(arc)) {
                        self.apply_snap_coincident_arc(snap, arc, ArcPoint::Start, pos);
                    }
                }
                Some(GrabTarget::ArcEnd(arc)) => {
                    let pos = arc_end_pos(&self.sketch.arcs[arc]);
                    if let Some((_, snap)) = self.find_snap_target_ex(pos, hit_threshold, None, Some(arc)) {
                        self.apply_snap_coincident_arc(snap, arc, ArcPoint::End, pos);
                    }
                }
                _ => {}
            }
        }
        self.grab = None;
    }

    // Toggle selection
    fn toggle_selection(&mut self, sel: Selection) {
        if let Some(idx) = self.selection.iter().position(|s| *s == sel) {
            self.selection.remove(idx);
        } else {
            // Constraints are exclusive - clear everything else when selecting one
            if matches!(sel, Selection::Constraint(_)) {
                self.selection.clear();
            } else {
                // Clear any constraint selections when selecting non-constraints
                self.selection.retain(|s| !matches!(s, Selection::Constraint(_)));
            }
            self.selection.push(sel);
        }
    }

    fn load_from_json(&mut self, json: &str) {
        match serde_json::from_str::<Sketch>(json) {
            Ok(mut sketch) => {
                sketch.dedup_constraints();
                sketch.consolidate_helper_constraints();
                sketch.solve();
                self.sketch = sketch;
                self.selection.clear();
                self.history = History::new();
                self.line_draw = None;
                self.circle_draw = None;
                self.arc_draw = None;
                self.pending_fit = true;
            }
            Err(e) => eprintln!("Failed to parse sketch: {}", e),
        }
    }


    // Execute an action: apply to sketch and record in history
    fn exec(&mut self, action: Action) {
        action.apply(&mut self.sketch);
        self.sketch.dedup_constraints();
        self.history.push(action, &self.sketch);
    }

    // Apply constraint to current selection
    // Check if the current selection exactly satisfies a constraint for immediate application
    fn can_apply_constraint(&self, ct: ConstraintType) -> bool {
        let sel = &self.selection;
        match ct {
            ConstraintType::Horizontal | ConstraintType::Vertical => {
                !sel.is_empty() && sel.iter().all(|s| matches!(s, Selection::Line(_)))
            }
            ConstraintType::Parallel | ConstraintType::Perpendicular => {
                sel.len() == 2 && sel.iter().all(|s| matches!(s, Selection::Line(_)))
            }
            ConstraintType::EqualLength => {
                sel.len() == 2 && (
                    sel.iter().all(|s| matches!(s, Selection::Line(_)))
                    || sel.iter().all(|s| matches!(s, Selection::Arc(_)))
                )
            }
            ConstraintType::Tangent => {
                sel.len() == 2 && {
                    let lines = sel.iter().filter(|s| matches!(s, Selection::Line(_))).count();
                    let arcs = sel.iter().filter(|s| matches!(s, Selection::Arc(_))).count();
                    (lines == 1 && arcs == 1) || arcs == 2
                }
            }
            ConstraintType::Coincident => {
                sel.len() == 2 && {
                    let point_like = |s: &Selection| matches!(s,
                        Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                        | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_));
                    let body = |s: &Selection| matches!(s, Selection::Line(_) | Selection::Arc(_));
                    // Two point-like, or one point-like + one body, or two bodies
                    (sel.iter().all(|s| point_like(s)))
                    || (sel.iter().filter(|s| point_like(s)).count() == 1 && sel.iter().filter(|s| body(s)).count() == 1)
                    || (sel.iter().all(|s| matches!(s, Selection::Line(_))))
                }
            }
            ConstraintType::Lock => {
                !sel.is_empty() && sel.iter().all(|s| matches!(s,
                    Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_)))
            }
            ConstraintType::ToggleStyle => {
                !sel.is_empty() && sel.iter().all(|s| matches!(s, Selection::Line(_) | Selection::Arc(_)))
            }
        }
    }

    // Check if entering constraint mode makes sense given current selection.
    // True if the selection could still lead to a valid constraint with more clicks.
    fn could_enter_constraint_mode(&self, ct: ConstraintType) -> bool {
        let sel = &self.selection;
        if sel.is_empty() { return true; }

        // Filter to only valid entities for this constraint
        let valid: Vec<&Selection> = sel.iter().filter(|s| Self::is_valid_for_constraint(ct, s)).collect();
        if valid.is_empty() { return false; }

        let needed = match ct {
            ConstraintType::Horizontal | ConstraintType::Vertical
            | ConstraintType::Lock | ConstraintType::ToggleStyle => 1,
            _ => 2,
        };

        // If we already have enough, it must be directly applicable (handled by can_apply)
        if valid.len() >= needed { return false; }

        // For tangent with 1 selection: check it's a valid partial (line or arc)
        if ct == ConstraintType::Tangent && valid.len() == 1 {
            return matches!(valid[0], Selection::Line(_) | Selection::Arc(_));
        }

        true
    }

    // Check if a selection entity is valid to add in constraint mode
    fn is_valid_for_constraint(ct: ConstraintType, sel: &Selection) -> bool {
        match ct {
            ConstraintType::Horizontal | ConstraintType::Vertical
            | ConstraintType::Parallel | ConstraintType::Perpendicular => {
                matches!(sel, Selection::Line(_))
            }
            ConstraintType::EqualLength => {
                matches!(sel, Selection::Line(_) | Selection::Arc(_))
            }
            ConstraintType::Tangent => {
                matches!(sel, Selection::Line(_) | Selection::Arc(_))
            }
            ConstraintType::Coincident => {
                matches!(sel, Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_)
                    | Selection::Line(_) | Selection::Arc(_))
            }
            ConstraintType::Lock => {
                matches!(sel, Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_))
            }
            ConstraintType::ToggleStyle => {
                matches!(sel, Selection::Line(_) | Selection::Arc(_))
            }
        }
    }

    // Try to apply constraint if selection is valid, otherwise enter constraint mode
    fn try_apply_or_enter_mode(&mut self, ct: ConstraintType) {
        if self.can_apply_constraint(ct) {
            match ct {
                ConstraintType::Horizontal => self.apply_horizontal(),
                ConstraintType::Vertical => self.apply_vertical(),
                ConstraintType::Coincident => self.apply_coincident(),
                ConstraintType::Parallel => self.apply_parallel(),
                ConstraintType::Perpendicular => self.apply_perpendicular(),
                ConstraintType::EqualLength => self.apply_equal_length(),
                ConstraintType::Tangent => self.apply_tangent(),
                ConstraintType::Lock => self.apply_lock(),
                ConstraintType::ToggleStyle => self.apply_toggle_style(),
            }
        } else {
            // Enter constraint mode, keep current selection (filter invalid ones)
            self.selection.retain(|s| Self::is_valid_for_constraint(ct, s));
            self.tool = Tool::ConstraintMode(ct);
        }
    }

    // Check if two "vertices" (point-like entities) are already transitively coincident.
    // Uses the same union-find as compute_locked_sets.
    fn are_transitively_coincident(&self, sel_a: Selection, sel_b: Selection) -> bool {
        let np = self.sketch.points.slot_count();
        let nl = self.sketch.lines.slot_count();
        let na = self.sketch.arcs.slot_count();
        let total = np + 2 * nl + 3 * na;
        let mut parent: Vec<usize> = (0..total).collect();
        let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
            while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; } x
        };
        let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (find(parent, a), find(parent, b));
            if ra != rb { parent[ra] = rb; }
        };
        let pt_id = |r: Ref<Point>| r.index() as usize;
        let lp1_id = |r: Ref<Line>| np + r.index() as usize;
        let lp2_id = |r: Ref<Line>| np + nl + r.index() as usize;
        let ac_id = |r: Ref<Arc>| np + 2 * nl + r.index() as usize;
        let as_id = |r: Ref<Arc>| np + 2 * nl + na + r.index() as usize;
        let ae_id = |r: Ref<Arc>| np + 2 * nl + 2 * na + r.index() as usize;

        // Build unions (same as compute_locked_sets)
        for c in &self.sketch.coincident_pp { union(&mut parent, pt_id(c.a), pt_id(c.b)); }
        for c in &self.sketch.coincident_lp1 { union(&mut parent, lp1_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_lp2 { union(&mut parent, lp2_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_ll11 { union(&mut parent, lp1_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll12 { union(&mut parent, lp1_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_ll21 { union(&mut parent, lp2_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll22 { union(&mut parent, lp2_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_arc_center { union(&mut parent, pt_id(c.point), ac_id(c.arc)); }
        for c in &self.sketch.coincident_arc_start { union(&mut parent, pt_id(c.point), as_id(c.arc)); }
        for c in &self.sketch.coincident_arc_end { union(&mut parent, pt_id(c.point), ae_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_center { union(&mut parent, lp1_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_center { union(&mut parent, lp2_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_start { union(&mut parent, lp1_id(c.line), as_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_start { union(&mut parent, lp2_id(c.line), as_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_end { union(&mut parent, lp1_id(c.line), ae_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_end { union(&mut parent, lp2_id(c.line), ae_id(c.arc)); }
        for c in &self.sketch.concentric { union(&mut parent, ac_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_center_start { union(&mut parent, ac_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_center_end { union(&mut parent, ac_id(c.a), ae_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_center { union(&mut parent, as_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_center { union(&mut parent, ae_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_start { union(&mut parent, as_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_end { union(&mut parent, as_id(c.a), ae_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_start { union(&mut parent, ae_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_end { union(&mut parent, ae_id(c.a), ae_id(c.b)); }

        let sel_to_id = |s: Selection| -> Option<usize> {
            match s {
                Selection::Point(r) => Some(pt_id(r)),
                Selection::LineP1(r) => Some(lp1_id(r)),
                Selection::LineP2(r) => Some(lp2_id(r)),
                Selection::ArcCenter(r) => Some(ac_id(r)),
                Selection::ArcStart(r) => Some(as_id(r)),
                Selection::ArcEnd(r) => Some(ae_id(r)),
                _ => None,
            }
        };

        if let (Some(id_a), Some(id_b)) = (sel_to_id(sel_a), sel_to_id(sel_b)) {
            find(&mut parent, id_a) == find(&mut parent, id_b)
        } else {
            false
        }
    }

    // Convert a line endpoint + snap target to Selection pairs for transitive check
    fn snap_to_selection(snap: SnapTarget) -> Option<Selection> {
        match snap {
            SnapTarget::Point(r) => Some(Selection::Point(r)),
            SnapTarget::LineP1(r) => Some(Selection::LineP1(r)),
            SnapTarget::LineP2(r) => Some(Selection::LineP2(r)),
            SnapTarget::ArcCenter(r) => Some(Selection::ArcCenter(r)),
            SnapTarget::ArcStart(r) => Some(Selection::ArcStart(r)),
            SnapTarget::ArcEnd(r) => Some(Selection::ArcEnd(r)),
            SnapTarget::Line(_) | SnapTarget::ArcBody(_) => None,
        }
    }

    // Check if a coincident constraint already exists (direct or transitive)
    fn has_existing_coincident_line(&self, line: Ref<Line>, is_p1: bool, snap: SnapTarget) -> bool {
        if let Some(snap_sel) = Self::snap_to_selection(snap) {
            let line_sel = if is_p1 { Selection::LineP1(line) } else { Selection::LineP2(line) };
            self.are_transitively_coincident(line_sel, snap_sel)
        } else {
            // Check direct constraints for Line/ArcBody snap targets
            match (snap, is_p1) {
                (SnapTarget::ArcBody(arc), true) => self.sketch.line_p1_on_arc.iter().any(|c| c.line == line && c.arc == arc),
                (SnapTarget::ArcBody(arc), false) => self.sketch.line_p2_on_arc.iter().any(|c| c.line == line && c.arc == arc),
                (SnapTarget::Line(other), true) => self.sketch.line_p1_on_line.iter().any(|c| c.a == line && c.b == other),
                (SnapTarget::Line(other), false) => self.sketch.line_p2_on_line.iter().any(|c| c.a == line && c.b == other),
                _ => false,
            }
        }
    }

    // Get position of a DimensionEndpoint
    fn dim_endpoint_pos(&self, ep: &DimensionEndpoint) -> vect2d {
        match *ep {
            DimensionEndpoint::Point(r) => self.sketch.points[r].pos.value,
            DimensionEndpoint::LineP1(r) => self.sketch.lines[r].p1.value,
            DimensionEndpoint::LineP2(r) => self.sketch.lines[r].p2.value,
            DimensionEndpoint::ArcCenter(r) => self.sketch.arcs[r].center.value,
            DimensionEndpoint::ArcStart(r) => arc_start_pos(&self.sketch.arcs[r]),
            DimensionEndpoint::ArcEnd(r) => arc_end_pos(&self.sketch.arcs[r]),
        }
    }

    // Measure the current value for a dimension kind
    fn measure_dimension(&self, kind: &DimensionKind) -> f64 {
        match kind {
            DimensionKind::LineLength(r) => {
                let l = &self.sketch.lines[*r];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                (dx * dx + dy * dy).sqrt()
            }
            DimensionKind::PointPointDistance(a, b) => {
                let pa = self.dim_endpoint_pos(a);
                let pb = self.dim_endpoint_pos(b);
                let dx = pb.x - pa.x;
                let dy = pb.y - pa.y;
                (dx * dx + dy * dy).sqrt()
            }
            DimensionKind::PointLineDistance(pt, line) => {
                let p = self.dim_endpoint_pos(pt);
                let l = &self.sketch.lines[*line];
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len = (dx * dx + dy * dy).sqrt();
                if len < 1e-12 { return 0.0; }
                (((p.x - l.p1.value.x) * dy - (p.y - l.p1.value.y) * dx) / len).abs()
            }
            DimensionKind::ArcRadius(r) => {
                self.sketch.arcs[*r].radius.value
            }
        }
    }

    // Convert a Selection to a DimensionEndpoint (for point-like selections)
    fn selection_to_dim_endpoint(sel: &Selection) -> Option<DimensionEndpoint> {
        match *sel {
            Selection::Point(r) => Some(DimensionEndpoint::Point(r)),
            Selection::LineP1(r) => Some(DimensionEndpoint::LineP1(r)),
            Selection::LineP2(r) => Some(DimensionEndpoint::LineP2(r)),
            Selection::ArcCenter(r) => Some(DimensionEndpoint::ArcCenter(r)),
            Selection::ArcStart(r) => Some(DimensionEndpoint::ArcStart(r)),
            Selection::ArcEnd(r) => Some(DimensionEndpoint::ArcEnd(r)),
            _ => None,
        }
    }

    // Try to determine DimensionKind from current selection
    fn selection_to_dim_kind(&self) -> Option<DimensionKind> {
        let sel = &self.selection;
        if sel.len() == 1 {
            match sel[0] {
                Selection::Line(r) => return Some(DimensionKind::LineLength(r)),
                Selection::Arc(r) => return Some(DimensionKind::ArcRadius(r)),
                _ => {}
            }
        }
        if sel.len() == 2 {
            // Point + Line -> point-line distance
            let point_ep = sel.iter().find_map(|s| Self::selection_to_dim_endpoint(s));
            let line_ref = sel.iter().find_map(|s| if let Selection::Line(r) = s { Some(*r) } else { None });
            if let (Some(ep), Some(line)) = (point_ep, line_ref) {
                return Some(DimensionKind::PointLineDistance(ep, line));
            }
            // Two point-like -> point-point distance
            let ep_a = Self::selection_to_dim_endpoint(&sel[0]);
            let ep_b = Self::selection_to_dim_endpoint(&sel[1]);
            if let (Some(a), Some(b)) = (ep_a, ep_b) {
                return Some(DimensionKind::PointPointDistance(a, b));
            }
        }
        None
    }

    // Compute perpendicular offset from a sketch point to the measurement line of a dimension kind
    #[allow(dead_code)]
    fn compute_dim_offset(&self, kind: &DimensionKind, mouse_sketch: vect2d) -> vect2d {
        let (p1, p2) = self.dim_endpoints(kind);
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let len = (dx * dx + dy * dy).sqrt().max(1e-12);
        let nx = -dy / len;
        let ny = dx / len;
        // Project mouse onto the normal direction
        let offset_perp = (mouse_sketch.x - (p1.x + p2.x) / 2.0) * nx
                        + (mouse_sketch.y - (p1.y + p2.y) / 2.0) * ny;
        vect2d::new(0.0, offset_perp)
    }

    // Get the two sketch-space endpoints for a dimension kind
    fn dim_endpoints(&self, kind: &DimensionKind) -> (vect2d, vect2d) {
        match kind {
            DimensionKind::LineLength(r) => {
                let l = &self.sketch.lines[*r];
                (l.p1.value, l.p2.value)
            }
            DimensionKind::PointPointDistance(a, b) => {
                (self.dim_endpoint_pos(a), self.dim_endpoint_pos(b))
            }
            DimensionKind::PointLineDistance(pt, line) => {
                let p = self.dim_endpoint_pos(pt);
                let l = &self.sketch.lines[*line];
                // Project onto infinite line (not clamped to segment)
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len2 = dx * dx + dy * dy;
                let foot = if len2 < 1e-12 { l.p1.value } else {
                    let t = ((p.x - l.p1.value.x) * dx + (p.y - l.p1.value.y) * dy) / len2;
                    vect2d::new(l.p1.value.x + t * dx, l.p1.value.y + t * dy)
                };
                (p, foot)
            }
            DimensionKind::ArcRadius(r) => {
                let a = &self.sketch.arcs[*r];
                let edge = vect2d::new(a.center.value.x + a.radius.value, a.center.value.y);
                (a.center.value, edge)
            }
        }
    }

    // Draw rotated text using egui's native TextShape.angle support.
    // `center` is where text should be centered, `dir_x/dir_y` is the unit direction along the text.
    // Returns the bounding segment (text_start, text_end) in screen coords.
    fn draw_rotated_text(&self, painter: &egui::Painter, center: egui::Pos2,
                          dir_x: f32, dir_y: f32, text: &str,
                          font: egui::FontId, color: egui::Color32) -> (egui::Pos2, egui::Pos2) {
        // Ensure text reads left-to-right: flip direction if pointing left
        let (dx, dy) = if dir_x < 0.0 { (-dir_x, -dir_y) } else { (dir_x, dir_y) };
        let angle = dy.atan2(dx); // rotation angle in radians

        // Layout the text to get its size
        let galley = painter.layout_no_wrap(text.to_string(), font, color);
        let text_width = galley.rect.width();
        let text_height = galley.rect.height();

        // Position: pivot is top-left of the galley, rotated around that point.
        // We want the text centered at `center`, offset perpendicular to read above the line.
        // Compute the top-left position before rotation such that after rotation the text is centered.
        let half_w = text_width / 2.0;
        let half_h = text_height / 2.0;
        // Center of unrotated text at (pos.x + half_w, pos.y + half_h).
        // After rotating by `angle` around pos, center moves to:
        //   pos + rotate(half_w, half_h, angle)
        // We want that to equal `center - normal * offset` (slightly above the line)
        let nx = -dy;
        let ny = dx;
        let target_x = center.x - nx * (half_h + 2.0);
        let target_y = center.y - ny * (half_h + 2.0);
        let cos_a = angle.cos();
        let sin_a = angle.sin();
        let rotated_cx = half_w * cos_a - half_h * sin_a;
        let rotated_cy = half_w * sin_a + half_h * cos_a;
        let pos = egui::Pos2::new(target_x - rotated_cx, target_y - rotated_cy);

        let shape = egui::epaint::TextShape::new(pos, galley, color)
            .with_angle(angle);
        painter.add(shape);

        // Return text extent segment for hit testing
        let ts = egui::Pos2::new(center.x - dx * half_w, center.y - dy * half_w);
        let te = egui::Pos2::new(center.x + dx * half_w, center.y + dy * half_w);
        (ts, te)
    }

    // Draw a dimension annotation. Returns (text_start, text_end) screen segment for hit testing.
    fn draw_dimension(&self, painter: &egui::Painter, kind: &DimensionKind, value: f64,
                       offset: vect2d, text_along: f64, color: egui::Color32, is_radius: bool) -> (egui::Pos2, egui::Pos2) {
        if is_radius {
            if let DimensionKind::ArcRadius(r) = kind {
                let a = &self.sketch.arcs[*r];
                let angle = offset.x;
                let edge = vect2d::new(
                    a.center.value.x + a.radius.value * angle.cos(),
                    a.center.value.y + a.radius.value * angle.sin(),
                );
                let arrow_len = a.radius.value * 0.6;
                let inner = vect2d::new(
                    edge.x - arrow_len * angle.cos(),
                    edge.y - arrow_len * angle.sin(),
                );
                let se = self.to_screen(edge);
                let si = self.to_screen(inner);
                let stroke = egui::Stroke::new(1.0, color);
                painter.line_segment([se, si], stroke);
                // Arrowhead at edge
                let adx = si.x - se.x;
                let ady = si.y - se.y;
                let alen = (adx * adx + ady * ady).sqrt().max(1.0);
                let ax = adx / alen;
                let ay = ady / alen;
                let asz = 6.0;
                painter.line_segment([se, egui::Pos2::new(se.x + ax * asz + ay * asz * 0.4, se.y + ay * asz - ax * asz * 0.4)], stroke);
                painter.line_segment([se, egui::Pos2::new(se.x + ax * asz - ay * asz * 0.4, se.y + ay * asz + ax * asz * 0.4)], stroke);
                // Text along arrow
                let mid = egui::Pos2::new((se.x + si.x) / 2.0, (se.y + si.y) / 2.0);
                let text = format!("R{:.2}", value);
                return self.draw_rotated_text(painter, mid, ax, ay, &text,
                    egui::FontId::proportional(12.0), color);
            }
        }

        let (p1_sketch, p2_sketch) = self.dim_endpoints(kind);
        let dx = p2_sketch.x - p1_sketch.x;
        let dy = p2_sketch.y - p1_sketch.y;
        let len = (dx * dx + dy * dy).sqrt().max(1e-12);
        let nx = -dy / len;
        let ny = dx / len;
        let off = offset.y;

        let q1 = vect2d::new(p1_sketch.x + nx * off, p1_sketch.y + ny * off);
        let q2 = vect2d::new(p2_sketch.x + nx * off, p2_sketch.y + ny * off);

        let sq1 = self.to_screen(q1);
        let sq2 = self.to_screen(q2);
        let sp1 = self.to_screen(p1_sketch);
        let sp2 = self.to_screen(p2_sketch);

        let stroke = egui::Stroke::new(1.0, color);

        // Extension lines
        // For point-line distance, the line-side extension goes from the nearest
        // line endpoint to the dimension arrow position (not from the foot projection)
        if let DimensionKind::PointLineDistance(_, line_ref) = kind {
            let l = &self.sketch.lines[*line_ref];
            // Find which endpoint is closer to the foot (p2_sketch)
            let d1 = ((l.p1.value.x - p2_sketch.x).powi(2) + (l.p1.value.y - p2_sketch.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - p2_sketch.x).powi(2) + (l.p2.value.y - p2_sketch.y).powi(2)).sqrt();
            let nearest = if d1 < d2 { self.to_screen(l.p1.value) } else { self.to_screen(l.p2.value) };
            painter.line_segment([sp1, sq1], egui::Stroke::new(0.5, color));
            painter.line_segment([nearest, sq2], egui::Stroke::new(0.5, color));
        } else {
            painter.line_segment([sp1, sq1], egui::Stroke::new(0.5, color));
            painter.line_segment([sp2, sq2], egui::Stroke::new(0.5, color));
        }

        // Arrowheads and dimension line
        let adx = sq2.x - sq1.x;
        let ady = sq2.y - sq1.y;
        let alen = (adx * adx + ady * ady).sqrt().max(1.0);
        let ax = adx / alen;
        let ay = ady / alen;
        let asz = 6.0;

        // Text position along the line
        let text = format!("{:.2}", value);
        let char_width = 12.0 * 0.6;
        let text_half_w = text.len() as f32 * char_width / 2.0;
        let text_center = egui::Pos2::new(
            (sq1.x + sq2.x) / 2.0 + ax * (text_along as f32) * alen,
            (sq1.y + sq2.y) / 2.0 + ay * (text_along as f32) * alen,
        );

        // Dimension line: extend beyond endpoints if text is outside
        let line_start;
        let line_end;
        let text_left = text_center.x - ax * text_half_w;
        let text_left_y = text_center.y - ay * text_half_w;
        let text_right = text_center.x + ax * text_half_w;
        let text_right_y = text_center.y + ay * text_half_w;

        // Project text edges onto the line to find extension
        let proj_left = (text_left - sq1.x) * ax + (text_left_y - sq1.y) * ay;
        let proj_right = (text_right - sq1.x) * ax + (text_right_y - sq1.y) * ay;
        let margin = 4.0;

        let min_proj = proj_left.min(proj_right) - margin;
        let max_proj = proj_right.max(proj_left) + margin;

        line_start = if min_proj < 0.0 {
            egui::Pos2::new(sq1.x + ax * min_proj, sq1.y + ay * min_proj)
        } else { sq1 };
        line_end = if max_proj > alen {
            egui::Pos2::new(sq1.x + ax * max_proj, sq1.y + ay * max_proj)
        } else { sq2 };

        // Draw dimension line (possibly extended)
        painter.line_segment([line_start, line_end], stroke);

        // Arrowheads at original endpoints (sq1, sq2)
        painter.line_segment([sq1, egui::Pos2::new(sq1.x + ax * asz + ay * asz * 0.4, sq1.y + ay * asz - ax * asz * 0.4)], stroke);
        painter.line_segment([sq1, egui::Pos2::new(sq1.x + ax * asz - ay * asz * 0.4, sq1.y + ay * asz + ax * asz * 0.4)], stroke);
        painter.line_segment([sq2, egui::Pos2::new(sq2.x - ax * asz + ay * asz * 0.4, sq2.y - ay * asz - ax * asz * 0.4)], stroke);
        painter.line_segment([sq2, egui::Pos2::new(sq2.x - ax * asz - ay * asz * 0.4, sq2.y - ay * asz + ax * asz * 0.4)], stroke);

        // Value text rotated along the dimension line
        self.draw_rotated_text(painter, text_center, ax, ay, &text,
            egui::FontId::proportional(12.0), color)
    }

    // Compute the screen-space text segment for a dimension (for hit testing without drawing)
    fn dim_text_segment(&self, dim: &Dimension) -> (egui::Pos2, egui::Pos2) {
        let is_radius = matches!(dim.kind, DimensionKind::ArcRadius(_));
        let text = if is_radius { format!("R{:.2}", dim.value) } else { format!("{:.2}", dim.value) };
        let char_width = 12.0 * 0.6;
        let total_width = text.len() as f32 * char_width;

        if is_radius {
            if let DimensionKind::ArcRadius(r) = dim.kind {
                let a = &self.sketch.arcs[r];
                let angle = dim.offset.x;
                let edge = vect2d::new(
                    a.center.value.x + a.radius.value * angle.cos(),
                    a.center.value.y + a.radius.value * angle.sin(),
                );
                let arrow_len = a.radius.value * 0.6;
                let inner = vect2d::new(
                    edge.x - arrow_len * angle.cos(),
                    edge.y - arrow_len * angle.sin(),
                );
                let se = self.to_screen(edge);
                let si = self.to_screen(inner);
                let adx = si.x - se.x;
                let ady = si.y - se.y;
                let alen = (adx * adx + ady * ady).sqrt().max(1.0);
                let dx = if adx / alen < 0.0 { -adx / alen } else { adx / alen };
                let dy = if adx / alen < 0.0 { -ady / alen } else { ady / alen };
                let text_offset = 8.0;
                let tnx = -dy;
                let tny = dx;
                let mid = egui::Pos2::new(
                    (se.x + si.x) / 2.0 - tnx * text_offset,
                    (se.y + si.y) / 2.0 - tny * text_offset,
                );
                return (
                    egui::Pos2::new(mid.x - dx * total_width / 2.0, mid.y - dy * total_width / 2.0),
                    egui::Pos2::new(mid.x + dx * total_width / 2.0, mid.y + dy * total_width / 2.0),
                );
            }
        }

        let (p1, p2) = self.dim_endpoints(&dim.kind);
        let ddx = p2.x - p1.x;
        let ddy = p2.y - p1.y;
        let len = (ddx * ddx + ddy * ddy).sqrt().max(1e-12);
        let nx = -ddy / len;
        let ny = ddx / len;
        let off = dim.offset.y;
        let q1 = vect2d::new(p1.x + nx * off, p1.y + ny * off);
        let q2 = vect2d::new(p2.x + nx * off, p2.y + ny * off);
        let sq1 = self.to_screen(q1);
        let sq2 = self.to_screen(q2);
        let adx = sq2.x - sq1.x;
        let ady = sq2.y - sq1.y;
        let alen = (adx * adx + ady * ady).sqrt().max(1.0);
        let dx = if adx / alen < 0.0 { -adx / alen } else { adx / alen };
        let dy = if adx / alen < 0.0 { -ady / alen } else { ady / alen };
        // Apply same perpendicular offset as draw_rotated_text (text is above the line)
        let text_offset = 8.0;
        let tnx = -dy;
        let tny = dx;
        // Apply text_along offset
        let along_offset = dim.text_along as f32 * alen;
        let mid = egui::Pos2::new(
            (sq1.x + sq2.x) / 2.0 + (adx / alen) * along_offset - tnx * text_offset,
            (sq1.y + sq2.y) / 2.0 + (ady / alen) * along_offset - tny * text_offset,
        );
        (
            egui::Pos2::new(mid.x - dx * total_width / 2.0, mid.y - dy * total_width / 2.0),
            egui::Pos2::new(mid.x + dx * total_width / 2.0, mid.y + dy * total_width / 2.0),
        )
    }

    // Distance from a screen point to a screen-space line segment
    fn screen_point_to_segment_dist(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let len2 = dx * dx + dy * dy;
        if len2 < 1.0 {
            return ((p.x - a.x).powi(2) + (p.y - a.y).powi(2)).sqrt();
        }
        let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0);
        let proj_x = a.x + t * dx;
        let proj_y = a.y + t * dy;
        ((p.x - proj_x).powi(2) + (p.y - proj_y).powi(2)).sqrt()
    }

    // Start a new undo group. All exec() calls until the next begin_group share the same group.
    fn begin_group(&mut self) {
        self.history.begin_group();
    }

    fn apply_horizontal(&mut self) {
        self.begin_group();
        let lines: Vec<Ref<Line>> = self.selection.iter().filter_map(|s| {
            if let Selection::Line(r) = s { Some(*r) } else { None }
        }).collect();
        if !lines.is_empty() {
            self.exec(Action::ApplyHorizontal { lines });
        }
    }

    fn apply_vertical(&mut self) {
        self.begin_group();
        let lines: Vec<Ref<Line>> = self.selection.iter().filter_map(|s| {
            if let Selection::Line(r) = s { Some(*r) } else { None }
        }).collect();
        if !lines.is_empty() {
            self.exec(Action::ApplyVertical { lines });
        }
    }

    // Get position of a point-like selection
    fn selection_pos(&self, sel: Selection) -> Option<vect2d> {
        match sel {
            Selection::Point(r) => Some(self.sketch.points[r].pos.value),
            Selection::LineP1(r) => Some(self.sketch.lines[r].p1.value),
            Selection::LineP2(r) => Some(self.sketch.lines[r].p2.value),
            Selection::ArcCenter(r) => Some(self.sketch.arcs[r].center.value),
            Selection::ArcStart(r) => Some(arc_start_pos(&self.sketch.arcs[r])),
            Selection::ArcEnd(r) => Some(arc_end_pos(&self.sketch.arcs[r])),
            Selection::Line(_) | Selection::Arc(_) | Selection::Constraint(_) | Selection::Dimension(_) => None,
        }
    }

    // Create a coincident constraint between a point-like selection and a helper point.
    // Returns the Action, or None if the selection is not a point-like entity.
    fn coincident_action_to_point(sel: Selection, point: Ref<Point>) -> Option<Action> {
        match sel {
            Selection::Point(r) => Some(Action::ApplyCoincidentPP { a: point, b: r }),
            Selection::LineP1(r) => Some(Action::ApplyCoincidentLP1 { line: r, point }),
            Selection::LineP2(r) => Some(Action::ApplyCoincidentLP2 { line: r, point }),
            Selection::ArcCenter(r) => Some(Action::ApplyCoincidentArcCenter { point, arc: r }),
            Selection::ArcStart(r) => Some(Action::ApplyCoincidentArcStart { point, arc: r }),
            Selection::ArcEnd(r) => Some(Action::ApplyCoincidentArcEnd { point, arc: r }),
            Selection::Line(_) | Selection::Arc(_) | Selection::Constraint(_) | Selection::Dimension(_) => None,
        }
    }

    fn apply_coincident(&mut self) {
        self.begin_group();
        if self.selection.len() != 2 { return; }
        let (s0, s1) = (self.selection[0], self.selection[1]);

        // Point-on-line / endpoint-on-line: special cases (different constraint type)
        match (s0, s1) {
            (Selection::Point(point), Selection::Line(line))
            | (Selection::Line(line), Selection::Point(point)) => {
                self.exec(Action::ApplyPointOnLine { point, line });
                return;
            }
            (Selection::LineP1(a), Selection::Line(b))
            | (Selection::Line(b), Selection::LineP1(a)) => {
                self.exec(Action::ApplyLineP1OnLine { a, b });
                return;
            }
            (Selection::LineP2(a), Selection::Line(b))
            | (Selection::Line(b), Selection::LineP2(a)) => {
                self.exec(Action::ApplyLineP2OnLine { a, b });
                return;
            }
            // Point on arc/circle
            (Selection::Point(point), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::Point(point)) => {
                self.exec(Action::ApplyPointOnArc { point, arc });
                return;
            }
            // Line endpoint on arc/circle (direct constraint)
            (Selection::LineP1(line), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::LineP1(line)) => {
                self.exec(Action::ApplyLineP1OnArc { line, arc });
                return;
            }
            (Selection::LineP2(line), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::LineP2(line)) => {
                self.exec(Action::ApplyLineP2OnArc { line, arc });
                return;
            }
            // Arc endpoint on arc/circle (via helper point - PointOnArc needs a Point)
            (Selection::ArcCenter(src_arc), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::ArcCenter(src_arc)) => {
                let pos = self.sketch.arcs[src_arc].center.value;
                self.exec(Action::AddHelperPoint { pos });
                let helper = Ref::new(self.sketch.points.slot_count() as u32 - 1);
                self.exec(Action::ApplyCoincidentArcCenter { point: helper, arc: src_arc });
                self.exec(Action::ApplyPointOnArc { point: helper, arc });
                return;
            }
            (Selection::ArcStart(src_arc), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::ArcStart(src_arc)) => {
                let pos = arc_start_pos(&self.sketch.arcs[src_arc]);
                self.exec(Action::AddHelperPoint { pos });
                let helper = Ref::new(self.sketch.points.slot_count() as u32 - 1);
                self.exec(Action::ApplyCoincidentArcStart { point: helper, arc: src_arc });
                self.exec(Action::ApplyPointOnArc { point: helper, arc });
                return;
            }
            (Selection::ArcEnd(src_arc), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::ArcEnd(src_arc)) => {
                let pos = arc_end_pos(&self.sketch.arcs[src_arc]);
                self.exec(Action::AddHelperPoint { pos });
                let helper = Ref::new(self.sketch.points.slot_count() as u32 - 1);
                self.exec(Action::ApplyCoincidentArcEnd { point: helper, arc: src_arc });
                self.exec(Action::ApplyPointOnArc { point: helper, arc });
                return;
            }
            // Line-to-line (default: a.p2 == b.p1)
            (Selection::Line(a), Selection::Line(b)) => {
                self.exec(Action::ApplyCoincidentLL21 { a, b });
                return;
            }
            _ => {}
        }

        // For two point-like selections: try direct constraint first, fall back to helper point
        let pos = match self.selection_pos(s0) {
            Some(p) => p,
            None => return,
        };
        if self.selection_pos(s1).is_none() { return; }

        // Direct constraints for common cases (no helper point needed)
        match (s0, s1) {
            (Selection::Point(a), Selection::Point(b)) => {
                self.exec(Action::ApplyCoincidentPP { a, b });
                return;
            }
            (Selection::LineP1(a), Selection::LineP1(b)) => {
                self.exec(Action::ApplyCoincidentLL11 { a, b }); return;
            }
            (Selection::LineP1(a), Selection::LineP2(b)) => {
                self.exec(Action::ApplyCoincidentLL12 { a, b }); return;
            }
            (Selection::LineP2(a), Selection::LineP1(b)) => {
                self.exec(Action::ApplyCoincidentLL21 { a, b }); return;
            }
            (Selection::LineP2(a), Selection::LineP2(b)) => {
                self.exec(Action::ApplyCoincidentLL22 { a, b }); return;
            }
            (Selection::LineP1(line), Selection::Point(point))
            | (Selection::Point(point), Selection::LineP1(line)) => {
                self.exec(Action::ApplyCoincidentLP1 { line, point }); return;
            }
            (Selection::LineP2(line), Selection::Point(point))
            | (Selection::Point(point), Selection::LineP2(line)) => {
                self.exec(Action::ApplyCoincidentLP2 { line, point }); return;
            }
            (Selection::Point(point), Selection::ArcCenter(arc))
            | (Selection::ArcCenter(arc), Selection::Point(point)) => {
                self.exec(Action::ApplyCoincidentArcCenter { point, arc }); return;
            }
            (Selection::Point(point), Selection::ArcStart(arc))
            | (Selection::ArcStart(arc), Selection::Point(point)) => {
                self.exec(Action::ApplyCoincidentArcStart { point, arc }); return;
            }
            (Selection::Point(point), Selection::ArcEnd(arc))
            | (Selection::ArcEnd(arc), Selection::Point(point)) => {
                self.exec(Action::ApplyCoincidentArcEnd { point, arc }); return;
            }
            (Selection::ArcCenter(a), Selection::ArcCenter(b)) => {
                self.exec(Action::ApplyConcentric { a, b }); return;
            }
            // Line endpoint <-> Arc point (direct)
            (Selection::LineP1(line), Selection::ArcCenter(arc))
            | (Selection::ArcCenter(arc), Selection::LineP1(line)) => {
                self.exec(Action::ApplyCoincidentLP1ArcCenter { line, arc }); return;
            }
            (Selection::LineP2(line), Selection::ArcCenter(arc))
            | (Selection::ArcCenter(arc), Selection::LineP2(line)) => {
                self.exec(Action::ApplyCoincidentLP2ArcCenter { line, arc }); return;
            }
            (Selection::LineP1(line), Selection::ArcStart(arc))
            | (Selection::ArcStart(arc), Selection::LineP1(line)) => {
                self.exec(Action::ApplyCoincidentLP1ArcStart { line, arc }); return;
            }
            (Selection::LineP2(line), Selection::ArcStart(arc))
            | (Selection::ArcStart(arc), Selection::LineP2(line)) => {
                self.exec(Action::ApplyCoincidentLP2ArcStart { line, arc }); return;
            }
            (Selection::LineP1(line), Selection::ArcEnd(arc))
            | (Selection::ArcEnd(arc), Selection::LineP1(line)) => {
                self.exec(Action::ApplyCoincidentLP1ArcEnd { line, arc }); return;
            }
            (Selection::LineP2(line), Selection::ArcEnd(arc))
            | (Selection::ArcEnd(arc), Selection::LineP2(line)) => {
                self.exec(Action::ApplyCoincidentLP2ArcEnd { line, arc }); return;
            }
            // Arc-Arc endpoint (direct)
            (Selection::ArcCenter(a), Selection::ArcStart(b)) => {
                self.exec(Action::ApplyCoincidentArcCenterStart { a, b }); return;
            }
            (Selection::ArcStart(b), Selection::ArcCenter(a)) => {
                self.exec(Action::ApplyCoincidentArcCenterStart { a, b }); return;
            }
            (Selection::ArcCenter(a), Selection::ArcEnd(b)) => {
                self.exec(Action::ApplyCoincidentArcCenterEnd { a, b }); return;
            }
            (Selection::ArcEnd(b), Selection::ArcCenter(a)) => {
                self.exec(Action::ApplyCoincidentArcCenterEnd { a, b }); return;
            }
            (Selection::ArcStart(a), Selection::ArcStart(b)) => {
                self.exec(Action::ApplyCoincidentArcStartStart { a, b }); return;
            }
            (Selection::ArcStart(a), Selection::ArcEnd(b)) => {
                self.exec(Action::ApplyCoincidentArcStartEnd { a, b }); return;
            }
            (Selection::ArcEnd(a), Selection::ArcStart(b)) => {
                self.exec(Action::ApplyCoincidentArcEndStart { a, b }); return;
            }
            (Selection::ArcEnd(a), Selection::ArcEnd(b)) => {
                self.exec(Action::ApplyCoincidentArcEndEnd { a, b }); return;
            }
            _ => {}
        }

        // General case: create a helper point and constrain both selections to it
        self.exec(Action::AddHelperPoint { pos });
        let helper = Ref::new(self.sketch.points.slot_count() as u32 - 1);
        if let Some(action) = Self::coincident_action_to_point(s0, helper) {
            self.exec(action);
        }
        if let Some(action) = Self::coincident_action_to_point(s1, helper) {
            self.exec(action);
        }
    }

    fn apply_parallel(&mut self) {
        self.begin_group();
        if self.selection.len() == 2 {
            if let (Selection::Line(a), Selection::Line(b)) = (self.selection[0], self.selection[1]) {
                self.exec(Action::ApplyParallel { a, b });
            }
        }
    }

    fn apply_perpendicular(&mut self) {
        self.begin_group();
        if self.selection.len() == 2 {
            if let (Selection::Line(a), Selection::Line(b)) = (self.selection[0], self.selection[1]) {
                self.exec(Action::ApplyPerpendicular { a, b });
            }
        }
    }

    fn apply_toggle_style(&mut self) {
        self.begin_group();
        for sel in &self.selection.clone() {
            match *sel {
                Selection::Line(r) => { self.exec(Action::ToggleStyleLine { line: r }); }
                Selection::Arc(r) => { self.exec(Action::ToggleStyleArc { arc: r }); }
                _ => {}
            }
        }
    }

    // Find the directly locked vertex in the same transitive group as `sel`.
    // Returns unlock actions for all directly locked vertices in the group.
    fn find_direct_locks_in_group(&self, sel: Selection) -> Vec<Action> {
        let (_pt_locked, _l_p1_locked, _l_p2_locked, _arc_c_locked) = self.compute_locked_sets();

        // Build union-find (same as compute_locked_sets)
        let np = self.sketch.points.slot_count();
        let nl = self.sketch.lines.slot_count();
        let na = self.sketch.arcs.slot_count();
        let total = np + 2 * nl + 3 * na;
        let mut parent: Vec<usize> = (0..total).collect();
        let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
            while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; } x
        };
        let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (find(parent, a), find(parent, b));
            if ra != rb { parent[ra] = rb; }
        };
        let pt_id = |r: Ref<Point>| r.index() as usize;
        let lp1_id = |r: Ref<Line>| np + r.index() as usize;
        let lp2_id = |r: Ref<Line>| np + nl + r.index() as usize;
        let ac_id = |r: Ref<Arc>| np + 2 * nl + r.index() as usize;

        // Build unions (abbreviated - same constraint list as compute_locked_sets)
        for c in &self.sketch.coincident_pp { union(&mut parent, pt_id(c.a), pt_id(c.b)); }
        for c in &self.sketch.coincident_lp1 { union(&mut parent, lp1_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_lp2 { union(&mut parent, lp2_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_ll11 { union(&mut parent, lp1_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll12 { union(&mut parent, lp1_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_ll21 { union(&mut parent, lp2_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll22 { union(&mut parent, lp2_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_arc_center { union(&mut parent, pt_id(c.point), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_center { union(&mut parent, lp1_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_center { union(&mut parent, lp2_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.concentric { union(&mut parent, ac_id(c.a), ac_id(c.b)); }
        // (other arc constraint unions omitted for brevity - they follow the same pattern as compute_locked_sets)
        for c in &self.sketch.coincident_arc_start { union(&mut parent, pt_id(c.point), np + 2*nl + na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_arc_end { union(&mut parent, pt_id(c.point), np + 2*nl + 2*na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_lp1_arc_start { union(&mut parent, lp1_id(c.line), np + 2*nl + na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_lp2_arc_start { union(&mut parent, lp2_id(c.line), np + 2*nl + na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_lp1_arc_end { union(&mut parent, lp1_id(c.line), np + 2*nl + 2*na + c.arc.index() as usize); }
        for c in &self.sketch.coincident_lp2_arc_end { union(&mut parent, lp2_id(c.line), np + 2*nl + 2*na + c.arc.index() as usize); }

        let sel_id = match sel {
            Selection::Point(r) => Some(pt_id(r)),
            Selection::LineP1(r) => Some(lp1_id(r)),
            Selection::LineP2(r) => Some(lp2_id(r)),
            Selection::ArcCenter(r) => Some(ac_id(r)),
            _ => None,
        };
        let sel_id = match sel_id { Some(id) => id, None => return Vec::new() };
        let sel_root = find(&mut parent, sel_id);

        let mut actions = Vec::new();
        // Find all directly locked vertices in the same group
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.constraints.has_fix_x && p.constraints.has_fix_y && find(&mut parent, pt_id(r)) == sel_root {
                actions.push(Action::UnlockPoint { point: r });
            }
        }
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            if !l.p1.optimize && find(&mut parent, lp1_id(r)) == sel_root {
                actions.push(Action::UnlockLineP1 { line: r });
            }
            if !l.p2.optimize && find(&mut parent, lp2_id(r)) == sel_root {
                actions.push(Action::UnlockLineP2 { line: r });
            }
        }
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            if !a.center.optimize && find(&mut parent, ac_id(r)) == sel_root {
                actions.push(Action::UnlockArcCenter { arc: r });
            }
        }
        actions
    }

    fn apply_lock(&mut self) {
        self.begin_group();
        let (pt_locked, l_p1_locked, l_p2_locked, arc_c_locked) = self.compute_locked_sets();
        for sel in &self.selection.clone() {
            let is_locked = match *sel {
                Selection::Point(r) => pt_locked.contains(&r.index()),
                Selection::LineP1(r) => l_p1_locked.contains(&r.index()),
                Selection::LineP2(r) => l_p2_locked.contains(&r.index()),
                Selection::ArcCenter(r) => arc_c_locked.contains(&r.index()),
                _ => false,
            };
            if is_locked {
                // Unlock all directly locked vertices in the transitive group
                let unlock_actions = self.find_direct_locks_in_group(*sel);
                for action in unlock_actions {
                    self.exec(action);
                }
            } else {
                match *sel {
                    Selection::Point(r) => {
                        let pos = self.sketch.points[r].pos.value;
                        self.exec(Action::LockPoint { point: r, pos });
                    }
                    Selection::LineP1(r) => {
                        let pos = self.sketch.lines[r].p1.value;
                        self.exec(Action::LockLineP1 { line: r, pos });
                    }
                    Selection::LineP2(r) => {
                        let pos = self.sketch.lines[r].p2.value;
                        self.exec(Action::LockLineP2 { line: r, pos });
                    }
                    Selection::ArcCenter(r) => {
                        let pos = self.sketch.arcs[r].center.value;
                        self.exec(Action::LockArcCenter { arc: r, pos });
                    }
                    _ => {}
                }
            }
        }
    }

    fn apply_tangent(&mut self) {
        self.begin_group();
        if self.selection.len() != 2 { return; }
        let (s0, s1) = (self.selection[0], self.selection[1]);
        match (s0, s1) {
            (Selection::Line(line), Selection::Arc(arc))
            | (Selection::Arc(arc), Selection::Line(line)) => {
                self.exec(Action::ApplyTangentLA { line, arc });
            }
            (Selection::Arc(a), Selection::Arc(b)) => {
                self.exec(Action::ApplyTangentAA { a, b });
            }
            _ => {}
        }
    }

    fn apply_equal_length(&mut self) {
        self.begin_group();
        if self.selection.len() == 2 {
            match (self.selection[0], self.selection[1]) {
                (Selection::Line(a), Selection::Line(b)) => {
                    self.exec(Action::ApplyEqualLength { a, b });
                }
                (Selection::Arc(a), Selection::Arc(b)) => {
                    self.exec(Action::ApplyEqualRadius { a, b });
                }
                _ => {}
            }
        }
    }

    // Find a snap target near a position (for line drawing auto-coincident)
    fn find_snap_target(&self, sketch_pos: vect2d, threshold: f64) -> Option<(vect2d, SnapTarget)> {
        self.find_snap_target_ex(sketch_pos, threshold, None, None)
    }

    fn find_snap_target_excluding(&self, sketch_pos: vect2d, threshold: f64, exclude_line: Option<Ref<Line>>) -> Option<(vect2d, SnapTarget)> {
        self.find_snap_target_ex(sketch_pos, threshold, exclude_line, None)
    }

    fn find_snap_target_ex(&self, sketch_pos: vect2d, threshold: f64, exclude_line: Option<Ref<Line>>, exclude_arc: Option<Ref<Arc>>) -> Option<(vect2d, SnapTarget)> {
        // First pass: check points and line endpoints (high priority)
        let mut best: Option<(f64, vect2d, SnapTarget)> = None;

        let mut check = |dist: f64, pos: vect2d, target: SnapTarget| {
            if dist < threshold {
                if best.is_none() || dist < best.unwrap().0 {
                    best = Some((dist, pos, target));
                }
            }
        };

        // Standalone points (skip helpers)
        for r in self.sketch.points.refs() {
            if self.drag_point == Some(r) { continue; }
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let d = ((p.pos.value.x - sketch_pos.x).powi(2)
                   + (p.pos.value.y - sketch_pos.y).powi(2)).sqrt();
            check(d, p.pos.value, SnapTarget::Point(r));
        }

        // Line endpoints
        for r in self.sketch.lines.refs() {
            if exclude_line == Some(r) { continue; }
            let l = &self.sketch.lines[r];

            let d1 = ((l.p1.value.x - sketch_pos.x).powi(2)
                    + (l.p1.value.y - sketch_pos.y).powi(2)).sqrt();
            let d2 = ((l.p2.value.x - sketch_pos.x).powi(2)
                    + (l.p2.value.y - sketch_pos.y).powi(2)).sqrt();
            check(d1, l.p1.value, SnapTarget::LineP1(r));
            check(d2, l.p2.value, SnapTarget::LineP2(r));
        }

        // Arc centers and endpoints (same priority as points/line endpoints)
        for r in self.sketch.arcs.refs() {
            if exclude_arc == Some(r) { continue; }
            let a = &self.sketch.arcs[r];
            let dc = ((a.center.value.x - sketch_pos.x).powi(2)
                    + (a.center.value.y - sketch_pos.y).powi(2)).sqrt();
            check(dc, a.center.value, SnapTarget::ArcCenter(r));
            if !a.closed {
                let sp = arc_start_pos(a);
                let ep = arc_end_pos(a);
                let ds = ((sp.x - sketch_pos.x).powi(2) + (sp.y - sketch_pos.y).powi(2)).sqrt();
                let de = ((ep.x - sketch_pos.x).powi(2) + (ep.y - sketch_pos.y).powi(2)).sqrt();
                check(ds, sp, SnapTarget::ArcStart(r));
                check(de, ep, SnapTarget::ArcEnd(r));
            }
        }

        // If we found a point/endpoint, prefer it over line body
        if best.is_some() {
            return best.map(|(_, pos, target)| (pos, target));
        }

        // Second pass: check line bodies and arc/circle curves (lower priority)
        for r in self.sketch.lines.refs() {
            if exclude_line == Some(r) { continue; }
            let l = &self.sketch.lines[r];
            let d = point_to_segment_dist(sketch_pos, l.p1.value, l.p2.value);
            if d < threshold {
                if best.is_none() || d < best.unwrap().0 {
                    let proj = project_onto_segment(sketch_pos, l.p1.value, l.p2.value);
                    best = Some((d, proj, SnapTarget::Line(r)));
                }
            }
        }

        for r in self.sketch.arcs.refs() {
            if exclude_arc == Some(r) { continue; }
            let a = &self.sketch.arcs[r];
            let (d, proj) = point_to_arc_dist(sketch_pos, a);
            if d < threshold {
                if best.is_none() || d < best.unwrap().0 {
                    best = Some((d, proj, SnapTarget::ArcBody(r)));
                }
            }
        }

        best.map(|(_, pos, target)| (pos, target))
    }

    // Apply a coincident constraint between a snap target and a line endpoint
    // Apply a coincident/on-line constraint between a snap target and a line endpoint.
    // For arc snap targets that lack a direct line-arc constraint, uses a helper point.
    fn apply_snap_coincident(&mut self, snap: SnapTarget, line: Ref<Line>, is_p1: bool) {
        if self.has_existing_coincident_line(line, is_p1, snap) { return; }
        match (snap, is_p1) {
            (SnapTarget::Point(p), true) => { self.exec(Action::ApplyCoincidentLP1 { line, point: p }); }
            (SnapTarget::Point(p), false) => { self.exec(Action::ApplyCoincidentLP2 { line, point: p }); }
            (SnapTarget::LineP1(other), true) => { self.exec(Action::ApplyCoincidentLL11 { a: line, b: other }); }
            (SnapTarget::LineP1(other), false) => { self.exec(Action::ApplyCoincidentLL21 { a: line, b: other }); }
            (SnapTarget::LineP2(other), true) => { self.exec(Action::ApplyCoincidentLL12 { a: line, b: other }); }
            (SnapTarget::LineP2(other), false) => { self.exec(Action::ApplyCoincidentLL22 { a: line, b: other }); }
            (SnapTarget::Line(other), true) => { self.exec(Action::ApplyLineP1OnLine { a: line, b: other }); }
            (SnapTarget::Line(other), false) => { self.exec(Action::ApplyLineP2OnLine { a: line, b: other }); }
            // Direct line-arc constraints
            (SnapTarget::ArcCenter(arc), true) => { self.exec(Action::ApplyCoincidentLP1ArcCenter { line, arc }); }
            (SnapTarget::ArcCenter(arc), false) => { self.exec(Action::ApplyCoincidentLP2ArcCenter { line, arc }); }
            (SnapTarget::ArcStart(arc), true) => { self.exec(Action::ApplyCoincidentLP1ArcStart { line, arc }); }
            (SnapTarget::ArcStart(arc), false) => { self.exec(Action::ApplyCoincidentLP2ArcStart { line, arc }); }
            (SnapTarget::ArcEnd(arc), true) => { self.exec(Action::ApplyCoincidentLP1ArcEnd { line, arc }); }
            (SnapTarget::ArcEnd(arc), false) => { self.exec(Action::ApplyCoincidentLP2ArcEnd { line, arc }); }
            // Line endpoint on arc body (direct constraint)
            (SnapTarget::ArcBody(arc), true) => { self.exec(Action::ApplyLineP1OnArc { line, arc }); }
            (SnapTarget::ArcBody(arc), false) => { self.exec(Action::ApplyLineP2OnArc { line, arc }); }
        }
    }

    // Apply a coincident/on-line constraint between a snap target and a standalone point
    fn apply_snap_coincident_point(&mut self, snap: SnapTarget, point: Ref<Point>) {
        if let Some(snap_sel) = Self::snap_to_selection(snap) {
            if self.are_transitively_coincident(Selection::Point(point), snap_sel) { return; }
        }
        let action = match snap {
            SnapTarget::Point(other) => Action::ApplyCoincidentPP { a: point, b: other },
            SnapTarget::LineP1(line) => Action::ApplyCoincidentLP1 { line, point },
            SnapTarget::LineP2(line) => Action::ApplyCoincidentLP2 { line, point },
            SnapTarget::Line(line) => Action::ApplyPointOnLine { point, line },
            SnapTarget::ArcCenter(arc) => Action::ApplyCoincidentArcCenter { point, arc },
            SnapTarget::ArcStart(arc) => Action::ApplyCoincidentArcStart { point, arc },
            SnapTarget::ArcEnd(arc) => Action::ApplyCoincidentArcEnd { point, arc },
            SnapTarget::ArcBody(arc) => Action::ApplyPointOnArc { point, arc },
        };
        self.exec(action);
    }


    // Apply a snap constraint between a snap target and an arc point.
    // Uses direct constraints where possible; helper point for Line/ArcBody.
    fn apply_snap_coincident_arc(&mut self, snap: SnapTarget, arc: Ref<Arc>, which: ArcPoint, pos: vect2d) {
        // Check transitive coincidence
        if let Some(snap_sel) = Self::snap_to_selection(snap) {
            let arc_sel = match which {
                ArcPoint::Center => Selection::ArcCenter(arc),
                ArcPoint::Start => Selection::ArcStart(arc),
                ArcPoint::End => Selection::ArcEnd(arc),
            };
            if self.are_transitively_coincident(arc_sel, snap_sel) { return; }
        }
        // Direct Line endpoint <-> Arc point constraints
        match (&which, snap) {
            (ArcPoint::Center, SnapTarget::LineP1(line)) | (ArcPoint::Center, SnapTarget::LineP2(line)) => {
                let is_p1 = matches!(snap, SnapTarget::LineP1(_));
                if is_p1 { self.exec(Action::ApplyCoincidentLP1ArcCenter { line, arc }); }
                else { self.exec(Action::ApplyCoincidentLP2ArcCenter { line, arc }); }
                return;
            }
            (ArcPoint::Start, SnapTarget::LineP1(line)) | (ArcPoint::Start, SnapTarget::LineP2(line)) => {
                let is_p1 = matches!(snap, SnapTarget::LineP1(_));
                if is_p1 { self.exec(Action::ApplyCoincidentLP1ArcStart { line, arc }); }
                else { self.exec(Action::ApplyCoincidentLP2ArcStart { line, arc }); }
                return;
            }
            (ArcPoint::End, SnapTarget::LineP1(line)) | (ArcPoint::End, SnapTarget::LineP2(line)) => {
                let is_p1 = matches!(snap, SnapTarget::LineP1(_));
                if is_p1 { self.exec(Action::ApplyCoincidentLP1ArcEnd { line, arc }); }
                else { self.exec(Action::ApplyCoincidentLP2ArcEnd { line, arc }); }
                return;
            }
            // Direct Arc <-> Arc point constraints
            (ArcPoint::Center, SnapTarget::ArcCenter(other)) => { self.exec(Action::ApplyConcentric { a: arc, b: other }); return; }
            (ArcPoint::Center, SnapTarget::ArcStart(other)) => { self.exec(Action::ApplyCoincidentArcCenterStart { a: arc, b: other }); return; }
            (ArcPoint::Center, SnapTarget::ArcEnd(other)) => { self.exec(Action::ApplyCoincidentArcCenterEnd { a: arc, b: other }); return; }
            (ArcPoint::Start, SnapTarget::ArcCenter(other)) => { self.exec(Action::ApplyCoincidentArcStartCenter { a: arc, b: other }); return; }
            (ArcPoint::Start, SnapTarget::ArcStart(other)) => { self.exec(Action::ApplyCoincidentArcStartStart { a: arc, b: other }); return; }
            (ArcPoint::Start, SnapTarget::ArcEnd(other)) => { self.exec(Action::ApplyCoincidentArcStartEnd { a: arc, b: other }); return; }
            (ArcPoint::End, SnapTarget::ArcCenter(other)) => { self.exec(Action::ApplyCoincidentArcEndCenter { a: arc, b: other }); return; }
            (ArcPoint::End, SnapTarget::ArcStart(other)) => { self.exec(Action::ApplyCoincidentArcEndStart { a: arc, b: other }); return; }
            (ArcPoint::End, SnapTarget::ArcEnd(other)) => { self.exec(Action::ApplyCoincidentArcEndEnd { a: arc, b: other }); return; }
            _ => {}
        }
        // Point, Line body, ArcBody: need a helper point
        let arc_constraint: fn(Ref<Point>, Ref<Arc>) -> Action = match which {
            ArcPoint::Center => |p, a| Action::ApplyCoincidentArcCenter { point: p, arc: a },
            ArcPoint::Start => |p, a| Action::ApplyCoincidentArcStart { point: p, arc: a },
            ArcPoint::End => |p, a| Action::ApplyCoincidentArcEnd { point: p, arc: a },
        };
        self.exec(Action::AddHelperPoint { pos });
        let helper = Ref::new(self.sketch.points.slot_count() as u32 - 1);
        self.exec(arc_constraint(helper, arc));
        self.apply_snap_coincident_point(snap, helper);
    }

    fn describe_constraint(&self, id: ConstraintId) -> String {
        let ln = |r: Ref<Line>| self.sketch.lines[r].name.clone();
        let an = |r: Ref<Arc>| self.sketch.arcs[r].name.clone();
        let pn = |r: Ref<Point>| self.sketch.points[r].name.clone();
        match id {
            ConstraintId::Horizontal(r) => format!("H({})", ln(r)),
            ConstraintId::Vertical(r) => format!("V({})", ln(r)),
            ConstraintId::Parallel(i) => { let c = &self.sketch.parallel[i]; format!("Parallel({}, {})", ln(c.a), ln(c.b)) }
            ConstraintId::Perpendicular(i) => { let c = &self.sketch.perpendicular[i]; format!("Perp({}, {})", ln(c.a), ln(c.b)) }
            ConstraintId::EqualLength(i) => { let c = &self.sketch.equal_length[i]; format!("Equal({}, {})", ln(c.a), ln(c.b)) }
            ConstraintId::EqualRadius(i) => { let c = &self.sketch.equal_radius[i]; format!("EqualR({}, {})", an(c.a), an(c.b)) }
            ConstraintId::TangentLA(i) => { let c = &self.sketch.tangent_la[i]; format!("Tangent({}, {})", ln(c.line), an(c.arc)) }
            ConstraintId::TangentAA(i) => { let c = &self.sketch.tangent_aa[i]; format!("Tangent({}, {})", an(c.a), an(c.b)) }
            ConstraintId::Coincident(kind, i) => {
                let desc = match kind {
                    CoincidentKind::PP => { let c = &self.sketch.coincident_pp[i]; format!("{} = {}", pn(c.a), pn(c.b)) }
                    CoincidentKind::LP1 => { let c = &self.sketch.coincident_lp1[i]; format!("{}.p1 = {}", ln(c.line), pn(c.point)) }
                    CoincidentKind::LP2 => { let c = &self.sketch.coincident_lp2[i]; format!("{}.p2 = {}", ln(c.line), pn(c.point)) }
                    CoincidentKind::LL11 => { let c = &self.sketch.coincident_ll11[i]; format!("{}.p1 = {}.p1", ln(c.a), ln(c.b)) }
                    CoincidentKind::LL12 => { let c = &self.sketch.coincident_ll12[i]; format!("{}.p1 = {}.p2", ln(c.a), ln(c.b)) }
                    CoincidentKind::LL21 => { let c = &self.sketch.coincident_ll21[i]; format!("{}.p2 = {}.p1", ln(c.a), ln(c.b)) }
                    CoincidentKind::LL22 => { let c = &self.sketch.coincident_ll22[i]; format!("{}.p2 = {}.p2", ln(c.a), ln(c.b)) }
                    CoincidentKind::PointOnLine => { let c = &self.sketch.point_on_line[i]; format!("{} on {}", pn(c.point), ln(c.line)) }
                    CoincidentKind::PointOnArc => { let c = &self.sketch.point_on_arc[i]; format!("{} on {}", pn(c.point), an(c.arc)) }
                    CoincidentKind::LP1OnLine => { let c = &self.sketch.line_p1_on_line[i]; format!("{}.p1 on {}", ln(c.a), ln(c.b)) }
                    CoincidentKind::LP2OnLine => { let c = &self.sketch.line_p2_on_line[i]; format!("{}.p2 on {}", ln(c.a), ln(c.b)) }
                    CoincidentKind::LP1OnArc => { let c = &self.sketch.line_p1_on_arc[i]; format!("{}.p1 on {}", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP2OnArc => { let c = &self.sketch.line_p2_on_arc[i]; format!("{}.p2 on {}", ln(c.line), an(c.arc)) }
                    CoincidentKind::ArcCenter => { let c = &self.sketch.coincident_arc_center[i]; format!("{} = {}.c", pn(c.point), an(c.arc)) }
                    CoincidentKind::ArcStart => { let c = &self.sketch.coincident_arc_start[i]; format!("{} = {}.s", pn(c.point), an(c.arc)) }
                    CoincidentKind::ArcEnd => { let c = &self.sketch.coincident_arc_end[i]; format!("{} = {}.e", pn(c.point), an(c.arc)) }
                    CoincidentKind::LP1ArcCenter => { let c = &self.sketch.coincident_lp1_arc_center[i]; format!("{}.p1 = {}.c", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP2ArcCenter => { let c = &self.sketch.coincident_lp2_arc_center[i]; format!("{}.p2 = {}.c", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP1ArcStart => { let c = &self.sketch.coincident_lp1_arc_start[i]; format!("{}.p1 = {}.s", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP2ArcStart => { let c = &self.sketch.coincident_lp2_arc_start[i]; format!("{}.p2 = {}.s", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP1ArcEnd => { let c = &self.sketch.coincident_lp1_arc_end[i]; format!("{}.p1 = {}.e", ln(c.line), an(c.arc)) }
                    CoincidentKind::LP2ArcEnd => { let c = &self.sketch.coincident_lp2_arc_end[i]; format!("{}.p2 = {}.e", ln(c.line), an(c.arc)) }
                    CoincidentKind::ArcCenterStart => { let c = &self.sketch.coincident_arc_center_start[i]; format!("{}.c = {}.s", an(c.a), an(c.b)) }
                    CoincidentKind::ArcCenterEnd => { let c = &self.sketch.coincident_arc_center_end[i]; format!("{}.c = {}.e", an(c.a), an(c.b)) }
                    CoincidentKind::ArcStartCenter => { let c = &self.sketch.coincident_arc_start_center[i]; format!("{}.s = {}.c", an(c.a), an(c.b)) }
                    CoincidentKind::ArcEndCenter => { let c = &self.sketch.coincident_arc_end_center[i]; format!("{}.e = {}.c", an(c.a), an(c.b)) }
                    CoincidentKind::ArcStartStart => { let c = &self.sketch.coincident_arc_start_start[i]; format!("{}.s = {}.s", an(c.a), an(c.b)) }
                    CoincidentKind::ArcStartEnd => { let c = &self.sketch.coincident_arc_start_end[i]; format!("{}.s = {}.e", an(c.a), an(c.b)) }
                    CoincidentKind::ArcEndStart => { let c = &self.sketch.coincident_arc_end_start[i]; format!("{}.e = {}.s", an(c.a), an(c.b)) }
                    CoincidentKind::ArcEndEnd => { let c = &self.sketch.coincident_arc_end_end[i]; format!("{}.e = {}.e", an(c.a), an(c.b)) }
                };
                format!("Coinc({})", desc)
            }
            ConstraintId::HelperBridge(pt) => {
                let mut parts = Vec::new();
                for c in &self.sketch.coincident_lp1 { if c.point == pt { parts.push(format!("{}.p1", ln(c.line))); } }
                for c in &self.sketch.coincident_lp2 { if c.point == pt { parts.push(format!("{}.p2", ln(c.line))); } }
                for c in &self.sketch.coincident_pp { if c.a == pt { parts.push(pn(c.b)); } if c.b == pt { parts.push(pn(c.a)); } }
                for c in &self.sketch.point_on_line { if c.point == pt { parts.push(format!("on {}", ln(c.line))); } }
                for c in &self.sketch.point_on_arc { if c.point == pt { parts.push(format!("on {}", an(c.arc))); } }
                for c in &self.sketch.coincident_arc_center { if c.point == pt { parts.push(format!("{}.c", an(c.arc))); } }
                for c in &self.sketch.coincident_arc_start { if c.point == pt { parts.push(format!("{}.s", an(c.arc))); } }
                for c in &self.sketch.coincident_arc_end { if c.point == pt { parts.push(format!("{}.e", an(c.arc))); } }
                format!("Bridge({})", parts.join(" = "))
            }
        }
    }

    // Get the line/arc refs involved in a constraint (for highlighting)
    fn constraint_entities(&self, id: ConstraintId) -> (Vec<Ref<Line>>, Vec<Ref<Arc>>) {
        let mut lines = Vec::new();
        let mut arcs = Vec::new();
        match id {
            ConstraintId::Horizontal(r) | ConstraintId::Vertical(r) => { lines.push(r); }
            ConstraintId::Parallel(i) => {
                let c = &self.sketch.parallel[i];
                lines.push(c.a); lines.push(c.b);
            }
            ConstraintId::Perpendicular(i) => {
                let c = &self.sketch.perpendicular[i];
                lines.push(c.a); lines.push(c.b);
            }
            ConstraintId::EqualLength(i) => {
                let c = &self.sketch.equal_length[i];
                lines.push(c.a); lines.push(c.b);
            }
            ConstraintId::EqualRadius(i) => {
                let c = &self.sketch.equal_radius[i];
                arcs.push(c.a); arcs.push(c.b);
            }
            ConstraintId::TangentLA(i) => {
                let c = &self.sketch.tangent_la[i];
                lines.push(c.line); arcs.push(c.arc);
            }
            ConstraintId::TangentAA(i) => {
                let c = &self.sketch.tangent_aa[i];
                arcs.push(c.a); arcs.push(c.b);
            }
            ConstraintId::Coincident(kind, i) => {
                match kind {
                    CoincidentKind::LP1 => { let c = &self.sketch.coincident_lp1[i]; lines.push(c.line); }
                    CoincidentKind::LP2 => { let c = &self.sketch.coincident_lp2[i]; lines.push(c.line); }
                    CoincidentKind::LL11 | CoincidentKind::LL12 | CoincidentKind::LL21 | CoincidentKind::LL22 => {
                        let (a, b) = match kind {
                            CoincidentKind::LL11 => { let c = &self.sketch.coincident_ll11[i]; (c.a, c.b) }
                            CoincidentKind::LL12 => { let c = &self.sketch.coincident_ll12[i]; (c.a, c.b) }
                            CoincidentKind::LL21 => { let c = &self.sketch.coincident_ll21[i]; (c.a, c.b) }
                            CoincidentKind::LL22 => { let c = &self.sketch.coincident_ll22[i]; (c.a, c.b) }
                            _ => unreachable!(),
                        };
                        lines.push(a); lines.push(b);
                    }
                    CoincidentKind::PointOnLine => { let c = &self.sketch.point_on_line[i]; lines.push(c.line); }
                    CoincidentKind::PointOnArc => { let c = &self.sketch.point_on_arc[i]; arcs.push(c.arc); }
                    CoincidentKind::LP1OnLine => { let c = &self.sketch.line_p1_on_line[i]; lines.push(c.a); lines.push(c.b); }
                    CoincidentKind::LP2OnLine => { let c = &self.sketch.line_p2_on_line[i]; lines.push(c.a); lines.push(c.b); }
                    CoincidentKind::LP1OnArc => { let c = &self.sketch.line_p1_on_arc[i]; lines.push(c.line); arcs.push(c.arc); }
                    CoincidentKind::LP2OnArc => { let c = &self.sketch.line_p2_on_arc[i]; lines.push(c.line); arcs.push(c.arc); }
                    CoincidentKind::ArcCenter => { let c = &self.sketch.coincident_arc_center[i]; arcs.push(c.arc); }
                    CoincidentKind::ArcStart => { let c = &self.sketch.coincident_arc_start[i]; arcs.push(c.arc); }
                    CoincidentKind::ArcEnd => { let c = &self.sketch.coincident_arc_end[i]; arcs.push(c.arc); }
                    CoincidentKind::LP1ArcCenter | CoincidentKind::LP1ArcStart | CoincidentKind::LP1ArcEnd => {
                        match kind {
                            CoincidentKind::LP1ArcCenter => { let c = &self.sketch.coincident_lp1_arc_center[i]; lines.push(c.line); arcs.push(c.arc); }
                            CoincidentKind::LP1ArcStart => { let c = &self.sketch.coincident_lp1_arc_start[i]; lines.push(c.line); arcs.push(c.arc); }
                            CoincidentKind::LP1ArcEnd => { let c = &self.sketch.coincident_lp1_arc_end[i]; lines.push(c.line); arcs.push(c.arc); }
                            _ => unreachable!(),
                        }
                    }
                    CoincidentKind::LP2ArcCenter | CoincidentKind::LP2ArcStart | CoincidentKind::LP2ArcEnd => {
                        match kind {
                            CoincidentKind::LP2ArcCenter => { let c = &self.sketch.coincident_lp2_arc_center[i]; lines.push(c.line); arcs.push(c.arc); }
                            CoincidentKind::LP2ArcStart => { let c = &self.sketch.coincident_lp2_arc_start[i]; lines.push(c.line); arcs.push(c.arc); }
                            CoincidentKind::LP2ArcEnd => { let c = &self.sketch.coincident_lp2_arc_end[i]; lines.push(c.line); arcs.push(c.arc); }
                            _ => unreachable!(),
                        }
                    }
                    CoincidentKind::ArcCenterStart | CoincidentKind::ArcCenterEnd
                    | CoincidentKind::ArcStartCenter | CoincidentKind::ArcEndCenter
                    | CoincidentKind::ArcStartStart | CoincidentKind::ArcStartEnd
                    | CoincidentKind::ArcEndStart | CoincidentKind::ArcEndEnd => {
                        // These are all Arc-Arc; get both arcs
                        let (a, b) = match kind {
                            CoincidentKind::ArcCenterStart => { let c = &self.sketch.coincident_arc_center_start[i]; (c.a, c.b) }
                            CoincidentKind::ArcCenterEnd => { let c = &self.sketch.coincident_arc_center_end[i]; (c.a, c.b) }
                            CoincidentKind::ArcStartCenter => { let c = &self.sketch.coincident_arc_start_center[i]; (c.a, c.b) }
                            CoincidentKind::ArcEndCenter => { let c = &self.sketch.coincident_arc_end_center[i]; (c.a, c.b) }
                            CoincidentKind::ArcStartStart => { let c = &self.sketch.coincident_arc_start_start[i]; (c.a, c.b) }
                            CoincidentKind::ArcStartEnd => { let c = &self.sketch.coincident_arc_start_end[i]; (c.a, c.b) }
                            CoincidentKind::ArcEndStart => { let c = &self.sketch.coincident_arc_end_start[i]; (c.a, c.b) }
                            CoincidentKind::ArcEndEnd => { let c = &self.sketch.coincident_arc_end_end[i]; (c.a, c.b) }
                            _ => unreachable!(),
                        };
                        arcs.push(a); arcs.push(b);
                    }
                    CoincidentKind::PP => {} // point-point: no lines/arcs to highlight
                }
            }
            ConstraintId::HelperBridge(pt) => {
                // Find all lines/arcs connected through this helper point
                for c in &self.sketch.coincident_lp1 { if c.point == pt { lines.push(c.line); } }
                for c in &self.sketch.coincident_lp2 { if c.point == pt { lines.push(c.line); } }
                for c in &self.sketch.point_on_line { if c.point == pt { lines.push(c.line); } }
                for c in &self.sketch.point_on_arc { if c.point == pt { arcs.push(c.arc); } }
                for c in &self.sketch.coincident_arc_center { if c.point == pt { arcs.push(c.arc); } }
                for c in &self.sketch.coincident_arc_start { if c.point == pt { arcs.push(c.arc); } }
                for c in &self.sketch.coincident_arc_end { if c.point == pt { arcs.push(c.arc); } }
            }
        }
        (lines, arcs)
    }

    // Delete a constraint by id
    fn delete_constraint(&mut self, id: ConstraintId) {
        self.begin_group();
        // We need an action for this. Use a Drag-style snapshot since constraint
        // deletion is structural and we want clean undo.
        match id {
            ConstraintId::Horizontal(r) => {
                self.sketch.lines[r].constraints.horizontal = false;
            }
            ConstraintId::Vertical(r) => {
                self.sketch.lines[r].constraints.vertical = false;
            }
            ConstraintId::Parallel(i) => { self.sketch.parallel.remove(i); }
            ConstraintId::Perpendicular(i) => { self.sketch.perpendicular.remove(i); }
            ConstraintId::EqualLength(i) => { self.sketch.equal_length.remove(i); }
            ConstraintId::EqualRadius(i) => { self.sketch.equal_radius.remove(i); }
            ConstraintId::TangentLA(i) => { self.sketch.tangent_la.remove(i); }
            ConstraintId::TangentAA(i) => { self.sketch.tangent_aa.remove(i); }
            ConstraintId::Coincident(kind, i) => {
                match kind {
                    CoincidentKind::PP => { self.sketch.coincident_pp.remove(i); }
                    CoincidentKind::LP1 => { self.sketch.coincident_lp1.remove(i); }
                    CoincidentKind::LP2 => { self.sketch.coincident_lp2.remove(i); }
                    CoincidentKind::LL11 => { self.sketch.coincident_ll11.remove(i); }
                    CoincidentKind::LL12 => { self.sketch.coincident_ll12.remove(i); }
                    CoincidentKind::LL21 => { self.sketch.coincident_ll21.remove(i); }
                    CoincidentKind::LL22 => { self.sketch.coincident_ll22.remove(i); }
                    CoincidentKind::PointOnLine => { self.sketch.point_on_line.remove(i); }
                    CoincidentKind::PointOnArc => { self.sketch.point_on_arc.remove(i); }
                    CoincidentKind::LP1OnLine => { self.sketch.line_p1_on_line.remove(i); }
                    CoincidentKind::LP2OnLine => { self.sketch.line_p2_on_line.remove(i); }
                    CoincidentKind::LP1OnArc => { self.sketch.line_p1_on_arc.remove(i); }
                    CoincidentKind::LP2OnArc => { self.sketch.line_p2_on_arc.remove(i); }
                    CoincidentKind::ArcCenter => { self.sketch.coincident_arc_center.remove(i); }
                    CoincidentKind::ArcStart => { self.sketch.coincident_arc_start.remove(i); }
                    CoincidentKind::ArcEnd => { self.sketch.coincident_arc_end.remove(i); }
                    CoincidentKind::LP1ArcCenter => { self.sketch.coincident_lp1_arc_center.remove(i); }
                    CoincidentKind::LP2ArcCenter => { self.sketch.coincident_lp2_arc_center.remove(i); }
                    CoincidentKind::LP1ArcStart => { self.sketch.coincident_lp1_arc_start.remove(i); }
                    CoincidentKind::LP2ArcStart => { self.sketch.coincident_lp2_arc_start.remove(i); }
                    CoincidentKind::LP1ArcEnd => { self.sketch.coincident_lp1_arc_end.remove(i); }
                    CoincidentKind::LP2ArcEnd => { self.sketch.coincident_lp2_arc_end.remove(i); }
                    CoincidentKind::ArcCenterStart => { self.sketch.coincident_arc_center_start.remove(i); }
                    CoincidentKind::ArcCenterEnd => { self.sketch.coincident_arc_center_end.remove(i); }
                    CoincidentKind::ArcStartCenter => { self.sketch.coincident_arc_start_center.remove(i); }
                    CoincidentKind::ArcEndCenter => { self.sketch.coincident_arc_end_center.remove(i); }
                    CoincidentKind::ArcStartStart => { self.sketch.coincident_arc_start_start.remove(i); }
                    CoincidentKind::ArcStartEnd => { self.sketch.coincident_arc_start_end.remove(i); }
                    CoincidentKind::ArcEndStart => { self.sketch.coincident_arc_end_start.remove(i); }
                    CoincidentKind::ArcEndEnd => { self.sketch.coincident_arc_end_end.remove(i); }
                }
                self.sketch.cleanup_helper_points();
            }
            ConstraintId::HelperBridge(pt) => {
                self.sketch.delete_point(pt);
            }
        }
        self.sketch.solve();
        let snapshot = bincode::serialize(&self.sketch).unwrap();
        let action = Action::Drag { snapshot };
        self.history.push(action, &self.sketch);
        self.selection.clear();
    }

    // Compute the marker position for a line, offset perpendicular by `offset_px` screen pixels.
    // `along` shifts along the line direction (for stacking multiple markers).
    fn line_marker_pos(&self, line_ref: Ref<Line>, offset_px: f32, along: f32) -> egui::Pos2 {
        let l = &self.sketch.lines[line_ref];
        let p1 = self.to_screen(l.p1.value);
        let p2 = self.to_screen(l.p2.value);
        let mx = (p1.x + p2.x) / 2.0;
        let my = (p1.y + p2.y) / 2.0;
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let len = (dx * dx + dy * dy).sqrt().max(1.0);
        // Normal (perpendicular), always point "up" (negative y in screen)
        let nx = -dy / len;
        let ny = dx / len;
        let sign = if ny > 0.0 { -1.0 } else { 1.0 };
        let ux = dx / len;
        let uy = dy / len;
        egui::Pos2::new(
            mx + nx * offset_px * sign + ux * along,
            my + ny * offset_px * sign + uy * along,
        )
    }

    // Compute marker position for an arc (at the midpoint of the arc curve).
    // Position a constraint marker inside the arc curve, spread along it by index.
    fn arc_marker_pos(&self, arc_ref: Ref<Arc>, idx: i32) -> egui::Pos2 {
        let a = &self.sketch.arcs[arc_ref];
        let sa = a.start_angle.value;
        let ea = a.end_angle.value;
        let norm = |v: f64| -> f64 { let r = v % std::f64::consts::TAU; if r < 0.0 { r + std::f64::consts::TAU } else { r } };
        let span = if a.closed { std::f64::consts::TAU } else { norm(ea - sa) };
        let mid_angle = sa + span / 2.0;
        // Spread markers along the arc near the midpoint
        let angle_offset = idx as f64 * 12.0 / (a.radius.value * self.scale as f64).max(1.0);
        let angle = mid_angle + angle_offset;
        // Place inside the curve (negative offset from radius)
        let r = a.radius.value - 10.0 / self.scale as f64;
        let pos = vect2d::new(
            a.center.value.x + r * angle.cos(),
            a.center.value.y + r * angle.sin(),
        );
        self.to_screen(pos)
    }

    // Build constraint markers for the current frame
    fn build_constraint_markers(&mut self) {
        self.constraint_markers.clear();

        // Track how many markers each line/arc already has (for stacking)
        let mut line_marker_count: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();
        let mut arc_marker_count: std::collections::HashMap<u32, i32> = std::collections::HashMap::new();

        let add_line_marker = |this: &EditorApp, markers: &mut Vec<ConstraintMarker>,
                                    line: Ref<Line>, symbol: ConstraintSymbol, id: ConstraintId,
                                    counts: &mut std::collections::HashMap<u32, i32>| {
            let idx = *counts.get(&line.index()).unwrap_or(&0);
            *counts.entry(line.index()).or_insert(0) += 1;
            let along = (idx as f32 - 0.5) * 14.0; // spread along the line
            let pos = this.line_marker_pos(line, 10.0, along);
            markers.push(ConstraintMarker { pos, symbol, id });
        };

        let add_arc_marker = |this: &EditorApp, markers: &mut Vec<ConstraintMarker>,
                                    arc: Ref<Arc>, symbol: ConstraintSymbol, id: ConstraintId,
                                    counts: &mut std::collections::HashMap<u32, i32>| {
            let idx = *counts.get(&arc.index()).unwrap_or(&0);
            *counts.entry(arc.index()).or_insert(0) += 1;
            let pos = this.arc_marker_pos(arc, idx);
            markers.push(ConstraintMarker { pos, symbol, id });
        };

        // Collect markers into a temporary vec, then assign
        let mut markers = Vec::new();

        // Self-constraints on lines
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            if l.constraints.horizontal {
                add_line_marker(self, &mut markers, r, ConstraintSymbol::H, ConstraintId::Horizontal(r), &mut line_marker_count);
            }
            if l.constraints.vertical {
                add_line_marker(self, &mut markers, r, ConstraintSymbol::V, ConstraintId::Vertical(r), &mut line_marker_count);
            }
        }

        // Shared constraints
        for (i, c) in self.sketch.parallel.iter().enumerate() {
            let id = ConstraintId::Parallel(i);
            add_line_marker(self, &mut markers, c.a, ConstraintSymbol::Parallel, id, &mut line_marker_count);
            add_line_marker(self, &mut markers, c.b, ConstraintSymbol::Parallel, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.perpendicular.iter().enumerate() {
            let id = ConstraintId::Perpendicular(i);
            add_line_marker(self, &mut markers, c.a, ConstraintSymbol::Perpendicular, id, &mut line_marker_count);
            add_line_marker(self, &mut markers, c.b, ConstraintSymbol::Perpendicular, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.equal_length.iter().enumerate() {
            let id = ConstraintId::EqualLength(i);
            add_line_marker(self, &mut markers, c.a, ConstraintSymbol::Equal, id, &mut line_marker_count);
            add_line_marker(self, &mut markers, c.b, ConstraintSymbol::Equal, id, &mut line_marker_count);
        }
        for (i, c) in self.sketch.equal_radius.iter().enumerate() {
            let id = ConstraintId::EqualRadius(i);
            add_arc_marker(self, &mut markers, c.a, ConstraintSymbol::Equal, id, &mut arc_marker_count);
            add_arc_marker(self, &mut markers, c.b, ConstraintSymbol::Equal, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.tangent_la.iter().enumerate() {
            let id = ConstraintId::TangentLA(i);
            add_line_marker(self, &mut markers, c.line, ConstraintSymbol::Tangent, id, &mut line_marker_count);
            add_arc_marker(self, &mut markers, c.arc, ConstraintSymbol::Tangent, id, &mut arc_marker_count);
        }
        for (i, c) in self.sketch.tangent_aa.iter().enumerate() {
            let id = ConstraintId::TangentAA(i);
            add_arc_marker(self, &mut markers, c.a, ConstraintSymbol::Tangent, id, &mut arc_marker_count);
            add_arc_marker(self, &mut markers, c.b, ConstraintSymbol::Tangent, id, &mut arc_marker_count);
        }

        // Coincident display setup
        let sel = &self.selection;
        let pt_sel = |r: Ref<Point>| sel.contains(&Selection::Point(r));
        let lp1_sel = |r: Ref<Line>| sel.contains(&Selection::LineP1(r));
        let lp2_sel = |r: Ref<Line>| sel.contains(&Selection::LineP2(r));
        let ac_sel = |r: Ref<Arc>| sel.contains(&Selection::ArcCenter(r));
        let as_sel = |r: Ref<Arc>| sel.contains(&Selection::ArcStart(r));
        let ae_sel = |r: Ref<Arc>| sel.contains(&Selection::ArcEnd(r));

        // Helper point bridges: show as single markers
        let mut helper_point_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut coinc_count: std::collections::HashMap<u64, i32> = std::collections::HashMap::new();
        let pos_key = |p: egui::Pos2| -> u64 { ((p.x * 100.0) as u64) << 32 | ((p.y * 100.0) as u64) };
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if !p.helper { continue; }
            helper_point_ids.insert(r.index());

            let bridge_id = ConstraintId::HelperBridge(r);
            let bridge_selected = sel.contains(&Selection::Constraint(bridge_id));
            let mut visible = bridge_selected;
            if !visible {
                for c in &self.sketch.coincident_lp1 { if c.point == r { visible |= lp1_sel(c.line); } }
                for c in &self.sketch.coincident_lp2 { if c.point == r { visible |= lp2_sel(c.line); } }
                for c in &self.sketch.coincident_arc_center { if c.point == r { visible |= ac_sel(c.arc); } }
                for c in &self.sketch.coincident_arc_start { if c.point == r { visible |= as_sel(c.arc); } }
                for c in &self.sketch.coincident_arc_end { if c.point == r { visible |= ae_sel(c.arc); } }
                for c in &self.sketch.coincident_pp { if c.a == r { visible |= pt_sel(c.b); } if c.b == r { visible |= pt_sel(c.a); } }
            }
            if visible {
                let pos = self.to_screen(p.pos.value);
                let key = pos_key(pos);
                let idx = *coinc_count.get(&key).unwrap_or(&0);
                *coinc_count.entry(key).or_insert(0) += 1;
                let offset = egui::Vec2::new(8.0 + idx as f32 * 12.0, -8.0);
                markers.push(ConstraintMarker { pos: pos + offset, symbol: ConstraintSymbol::Coincident, id: bridge_id });
            }
        }

        // Coincident constraints - collect, skip those involving helper points
        // Phase 1: collect all coincident markers with their base position and visibility flag
        struct CoincidentEntry {
            base_pos: egui::Pos2,
            id: ConstraintId,
            vertex_selected: bool,
        }
        let mut coinc_entries: Vec<CoincidentEntry> = Vec::new();

        let mut add_coinc_entry = |_markers: &mut Vec<ConstraintMarker>, pos: egui::Pos2, id: ConstraintId, visible: bool| {
            coinc_entries.push(CoincidentEntry { base_pos: pos, id, vertex_selected: visible });
        };

        // Skip constraints that reference helper points (those are shown as HelperBridge markers)
        let skip_if_helper_pp = |c: &CoincidentPP| -> bool {
            helper_point_ids.contains(&c.a.index()) || helper_point_ids.contains(&c.b.index())
        };
        let skip_if_helper_pt = |pt: Ref<Point>| -> bool {
            helper_point_ids.contains(&pt.index())
        };

        macro_rules! coinc {
            ($markers:expr, $coll:expr, $kind:expr, $pos_expr:expr, $vis_expr:expr) => {
                for (i, c) in $coll.iter().enumerate() {
                    let id = ConstraintId::Coincident($kind, i);
                    let pos = $pos_expr(c);
                    let vis = $vis_expr(c);
                    add_coinc_entry(&mut $markers, pos, id, vis);
                }
            };
            ($markers:expr, $coll:expr, $kind:expr, $pos_expr:expr, $vis_expr:expr, skip_helper: $skip:expr) => {
                for (i, c) in $coll.iter().enumerate() {
                    if $skip(c) { continue; }
                    let id = ConstraintId::Coincident($kind, i);
                    let pos = $pos_expr(c);
                    let vis = $vis_expr(c);
                    add_coinc_entry(&mut $markers, pos, id, vis);
                }
            };
        }

        // Point-Point
        coinc!(markers, self.sketch.coincident_pp, CoincidentKind::PP,
            |c: &CoincidentPP| self.to_screen(self.sketch.points[c.a].pos.value),
            |c: &CoincidentPP| pt_sel(c.a) || pt_sel(c.b),
            skip_helper: |c: &CoincidentPP| skip_if_helper_pp(c));
        // Line-Point
        coinc!(markers, self.sketch.coincident_lp1, CoincidentKind::LP1,
            |c: &CoincidentLP1| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &CoincidentLP1| lp1_sel(c.line) || pt_sel(c.point),
            skip_helper: |c: &CoincidentLP1| skip_if_helper_pt(c.point));
        coinc!(markers, self.sketch.coincident_lp2, CoincidentKind::LP2,
            |c: &CoincidentLP2| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &CoincidentLP2| lp2_sel(c.line) || pt_sel(c.point),
            skip_helper: |c: &CoincidentLP2| skip_if_helper_pt(c.point));
        // Line-Line
        coinc!(markers, self.sketch.coincident_ll11, CoincidentKind::LL11,
            |c: &CoincidentLL11| self.to_screen(self.sketch.lines[c.a].p1.value),
            |c: &CoincidentLL11| lp1_sel(c.a) || lp1_sel(c.b));
        coinc!(markers, self.sketch.coincident_ll12, CoincidentKind::LL12,
            |c: &CoincidentLL12| self.to_screen(self.sketch.lines[c.a].p1.value),
            |c: &CoincidentLL12| lp1_sel(c.a) || lp2_sel(c.b));
        coinc!(markers, self.sketch.coincident_ll21, CoincidentKind::LL21,
            |c: &CoincidentLL21| self.to_screen(self.sketch.lines[c.a].p2.value),
            |c: &CoincidentLL21| lp2_sel(c.a) || lp1_sel(c.b));
        coinc!(markers, self.sketch.coincident_ll22, CoincidentKind::LL22,
            |c: &CoincidentLL22| self.to_screen(self.sketch.lines[c.a].p2.value),
            |c: &CoincidentLL22| lp2_sel(c.a) || lp2_sel(c.b));
        // Point on line/arc
        coinc!(markers, self.sketch.point_on_line, CoincidentKind::PointOnLine,
            |c: &PointOnLine| self.to_screen(self.sketch.points[c.point].pos.value),
            |c: &PointOnLine| pt_sel(c.point),
            skip_helper: |c: &PointOnLine| skip_if_helper_pt(c.point));
        coinc!(markers, self.sketch.point_on_arc, CoincidentKind::PointOnArc,
            |c: &PointOnArc| self.to_screen(self.sketch.points[c.point].pos.value),
            |c: &PointOnArc| pt_sel(c.point),
            skip_helper: |c: &PointOnArc| skip_if_helper_pt(c.point));
        // Line endpoint on line
        coinc!(markers, self.sketch.line_p1_on_line, CoincidentKind::LP1OnLine,
            |c: &LineP1OnLine| self.to_screen(self.sketch.lines[c.a].p1.value),
            |c: &LineP1OnLine| lp1_sel(c.a));
        coinc!(markers, self.sketch.line_p1_on_arc, CoincidentKind::LP1OnArc,
            |c: &LineP1OnArc| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &LineP1OnArc| lp1_sel(c.line));
        coinc!(markers, self.sketch.line_p2_on_arc, CoincidentKind::LP2OnArc,
            |c: &LineP2OnArc| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &LineP2OnArc| lp2_sel(c.line));
        coinc!(markers, self.sketch.line_p2_on_line, CoincidentKind::LP2OnLine,
            |c: &LineP2OnLine| self.to_screen(self.sketch.lines[c.a].p2.value),
            |c: &LineP2OnLine| lp2_sel(c.a));
        // Point-Arc
        coinc!(markers, self.sketch.coincident_arc_center, CoincidentKind::ArcCenter,
            |c: &CoincidentArcCenter| self.to_screen(self.sketch.arcs[c.arc].center.value),
            |c: &CoincidentArcCenter| pt_sel(c.point) || ac_sel(c.arc),
            skip_helper: |c: &CoincidentArcCenter| skip_if_helper_pt(c.point));
        coinc!(markers, self.sketch.coincident_arc_start, CoincidentKind::ArcStart,
            |c: &CoincidentArcStart| self.to_screen(arc_start_pos(&self.sketch.arcs[c.arc])),
            |c: &CoincidentArcStart| pt_sel(c.point) || as_sel(c.arc),
            skip_helper: |c: &CoincidentArcStart| skip_if_helper_pt(c.point));
        coinc!(markers, self.sketch.coincident_arc_end, CoincidentKind::ArcEnd,
            |c: &CoincidentArcEnd| self.to_screen(arc_end_pos(&self.sketch.arcs[c.arc])),
            |c: &CoincidentArcEnd| pt_sel(c.point) || ae_sel(c.arc),
            skip_helper: |c: &CoincidentArcEnd| skip_if_helper_pt(c.point));
        // Line-Arc
        coinc!(markers, self.sketch.coincident_lp1_arc_center, CoincidentKind::LP1ArcCenter,
            |c: &CoincidentLP1ArcCenter| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &CoincidentLP1ArcCenter| lp1_sel(c.line) || ac_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp2_arc_center, CoincidentKind::LP2ArcCenter,
            |c: &CoincidentLP2ArcCenter| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &CoincidentLP2ArcCenter| lp2_sel(c.line) || ac_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp1_arc_start, CoincidentKind::LP1ArcStart,
            |c: &CoincidentLP1ArcStart| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &CoincidentLP1ArcStart| lp1_sel(c.line) || as_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp2_arc_start, CoincidentKind::LP2ArcStart,
            |c: &CoincidentLP2ArcStart| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &CoincidentLP2ArcStart| lp2_sel(c.line) || as_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp1_arc_end, CoincidentKind::LP1ArcEnd,
            |c: &CoincidentLP1ArcEnd| self.to_screen(self.sketch.lines[c.line].p1.value),
            |c: &CoincidentLP1ArcEnd| lp1_sel(c.line) || ae_sel(c.arc));
        coinc!(markers, self.sketch.coincident_lp2_arc_end, CoincidentKind::LP2ArcEnd,
            |c: &CoincidentLP2ArcEnd| self.to_screen(self.sketch.lines[c.line].p2.value),
            |c: &CoincidentLP2ArcEnd| lp2_sel(c.line) || ae_sel(c.arc));
        // Arc-Arc
        coinc!(markers, self.sketch.concentric, CoincidentKind::ArcCenterStart, // reuse for concentric
            |c: &Concentric| self.to_screen(self.sketch.arcs[c.a].center.value),
            |c: &Concentric| ac_sel(c.a) || ac_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_center_start, CoincidentKind::ArcCenterStart,
            |c: &CoincidentArcCenterStart| self.to_screen(self.sketch.arcs[c.a].center.value),
            |c: &CoincidentArcCenterStart| ac_sel(c.a) || as_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_center_end, CoincidentKind::ArcCenterEnd,
            |c: &CoincidentArcCenterEnd| self.to_screen(self.sketch.arcs[c.a].center.value),
            |c: &CoincidentArcCenterEnd| ac_sel(c.a) || ae_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_start_center, CoincidentKind::ArcStartCenter,
            |c: &CoincidentArcStartCenter| self.to_screen(arc_start_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcStartCenter| as_sel(c.a) || ac_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_end_center, CoincidentKind::ArcEndCenter,
            |c: &CoincidentArcEndCenter| self.to_screen(arc_end_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcEndCenter| ae_sel(c.a) || ac_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_start_start, CoincidentKind::ArcStartStart,
            |c: &CoincidentArcStartStart| self.to_screen(arc_start_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcStartStart| as_sel(c.a) || as_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_start_end, CoincidentKind::ArcStartEnd,
            |c: &CoincidentArcStartEnd| self.to_screen(arc_start_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcStartEnd| as_sel(c.a) || ae_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_end_start, CoincidentKind::ArcEndStart,
            |c: &CoincidentArcEndStart| self.to_screen(arc_end_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcEndStart| ae_sel(c.a) || as_sel(c.b));
        coinc!(markers, self.sketch.coincident_arc_end_end, CoincidentKind::ArcEndEnd,
            |c: &CoincidentArcEndEnd| self.to_screen(arc_end_pos(&self.sketch.arcs[c.a])),
            |c: &CoincidentArcEndEnd| ae_sel(c.a) || ae_sel(c.b));

        // Phase 2: determine which positions should show markers.
        // A position is visible if any entry there has vertex_selected=true OR any entry there is selected as a constraint.
        let mut pos_visible: std::collections::HashSet<u64> = std::collections::HashSet::new();
        let pos_key = |p: egui::Pos2| -> u64 { ((p.x * 100.0) as u64) << 32 | ((p.y * 100.0) as u64) };
        for e in &coinc_entries {
            let key = pos_key(e.base_pos);
            if e.vertex_selected || sel.contains(&Selection::Constraint(e.id)) {
                pos_visible.insert(key);
            }
        }

        // Phase 3: add visible coincident markers with stacking
        let mut coinc_count: std::collections::HashMap<u64, i32> = std::collections::HashMap::new();
        for e in &coinc_entries {
            let key = pos_key(e.base_pos);
            if !pos_visible.contains(&key) { continue; }
            let idx = *coinc_count.get(&key).unwrap_or(&0);
            *coinc_count.entry(key).or_insert(0) += 1;
            let offset = egui::Vec2::new(8.0 + idx as f32 * 12.0, -8.0);
            markers.push(ConstraintMarker {
                pos: e.base_pos + offset,
                symbol: ConstraintSymbol::Coincident,
                id: e.id,
            });
        }

        self.constraint_markers = markers;
    }

    // Hit test for delete: returns target only if exactly one entity is in range.
    // Only standalone points and lines (body, not endpoints) are delete targets.
    #[allow(dead_code)]
    fn hit_test_delete(&self, sketch_pos: vect2d, threshold: f64) -> Option<DeleteTarget> {
        let mut targets: Vec<DeleteTarget> = Vec::new();

        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let d = ((p.pos.value.x - sketch_pos.x).powi(2)
                   + (p.pos.value.y - sketch_pos.y).powi(2)).sqrt();
            if d < threshold { targets.push(DeleteTarget::Point(r)); }
        }

        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            let d = point_to_segment_dist(sketch_pos, l.p1.value, l.p2.value);
            if d < threshold { targets.push(DeleteTarget::Line(r)); }
        }

        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let (d, _) = point_to_arc_dist(sketch_pos, a);
            if d < threshold { targets.push(DeleteTarget::Arc(r)); }
        }

        if targets.len() == 1 { Some(targets[0]) } else { None }
    }

    // Fit all entities into view with margin
    fn fit_all(&mut self, rect: egui::Rect) {
        let mut min_x = f64::MAX;
        let mut min_y = f64::MAX;
        let mut max_x = f64::MIN;
        let mut max_y = f64::MIN;

        let mut has_any = false;
        let mut extend = |x: f64, y: f64| {
            has_any = true;
            if x < min_x { min_x = x; }
            if x > max_x { max_x = x; }
            if y < min_y { min_y = y; }
            if y > max_y { max_y = y; }
        };

        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            extend(p.pos.value.x, p.pos.value.y);
        }
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            extend(l.p1.value.x, l.p1.value.y);
            extend(l.p2.value.x, l.p2.value.y);
        }
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let r_val = a.radius.value;
            extend(a.center.value.x - r_val, a.center.value.y - r_val);
            extend(a.center.value.x + r_val, a.center.value.y + r_val);
        }

        if !has_any { return; }

        // Add small padding if all points are coincident
        if max_x - min_x < 1e-6 { min_x -= 1.0; max_x += 1.0; }
        if max_y - min_y < 1e-6 { min_y -= 1.0; max_y += 1.0; }

        let margin = 0.08; // 8% margin on each side
        let w = rect.width();
        let h = rect.height();
        let span_x = (max_x - min_x) as f32;
        let span_y = (max_y - min_y) as f32;

        // Scale to fit with margin
        let usable_w = w * (1.0 - 2.0 * margin);
        let usable_h = h * (1.0 - 2.0 * margin);
        self.scale = (usable_w / span_x).min(usable_h / span_y).clamp(1e-4, 1e7);

        // Center: sketch center -> screen center
        let cx = (min_x + max_x) as f32 / 2.0;
        let cy = (min_y + max_y) as f32 / 2.0;
        self.offset.x = rect.center().x - cx * self.scale;
        self.offset.y = rect.center().y + cy * self.scale; // y flipped
    }

    // Compute which points/endpoints are transitively locked via coincident chains.
    // Returns (point_locked, line_p1_locked, line_p2_locked) as HashSets of Refs.
    fn compute_locked_sets(&self) -> (
        std::collections::HashSet<u32>,  // locked point indices
        std::collections::HashSet<u32>,  // locked line indices (p1)
        std::collections::HashSet<u32>,  // locked line indices (p2)
        std::collections::HashSet<u32>,  // locked arc indices (center)
    ) {
        // Flat IDs: points, line p1s, line p2s, arc centers, arc starts, arc ends
        let np = self.sketch.points.slot_count();
        let nl = self.sketch.lines.slot_count();
        let na = self.sketch.arcs.slot_count();
        let total = np + 2 * nl + 3 * na;

        let mut parent: Vec<usize> = (0..total).collect();
        let find = |parent: &mut Vec<usize>, mut x: usize| -> usize {
            while parent[x] != x { parent[x] = parent[parent[x]]; x = parent[x]; }
            x
        };
        let union = |parent: &mut Vec<usize>, a: usize, b: usize| {
            let (ra, rb) = (find(parent, a), find(parent, b));
            if ra != rb { parent[ra] = rb; }
        };

        let pt_id = |r: Ref<Point>| r.index() as usize;
        let lp1_id = |r: Ref<Line>| np + r.index() as usize;
        let lp2_id = |r: Ref<Line>| np + nl + r.index() as usize;
        let ac_id = |r: Ref<Arc>| np + 2 * nl + r.index() as usize;
        let as_id = |r: Ref<Arc>| np + 2 * nl + na + r.index() as usize;
        let ae_id = |r: Ref<Arc>| np + 2 * nl + 2 * na + r.index() as usize;

        // Point-Point, Line-Point, Line-Line
        for c in &self.sketch.coincident_pp { union(&mut parent, pt_id(c.a), pt_id(c.b)); }
        for c in &self.sketch.coincident_lp1 { union(&mut parent, lp1_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_lp2 { union(&mut parent, lp2_id(c.line), pt_id(c.point)); }
        for c in &self.sketch.coincident_ll11 { union(&mut parent, lp1_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll12 { union(&mut parent, lp1_id(c.a), lp2_id(c.b)); }
        for c in &self.sketch.coincident_ll21 { union(&mut parent, lp2_id(c.a), lp1_id(c.b)); }
        for c in &self.sketch.coincident_ll22 { union(&mut parent, lp2_id(c.a), lp2_id(c.b)); }
        // Point-Arc
        for c in &self.sketch.coincident_arc_center { union(&mut parent, pt_id(c.point), ac_id(c.arc)); }
        for c in &self.sketch.coincident_arc_start { union(&mut parent, pt_id(c.point), as_id(c.arc)); }
        for c in &self.sketch.coincident_arc_end { union(&mut parent, pt_id(c.point), ae_id(c.arc)); }
        // Line-Arc
        for c in &self.sketch.coincident_lp1_arc_center { union(&mut parent, lp1_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_center { union(&mut parent, lp2_id(c.line), ac_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_start { union(&mut parent, lp1_id(c.line), as_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_start { union(&mut parent, lp2_id(c.line), as_id(c.arc)); }
        for c in &self.sketch.coincident_lp1_arc_end { union(&mut parent, lp1_id(c.line), ae_id(c.arc)); }
        for c in &self.sketch.coincident_lp2_arc_end { union(&mut parent, lp2_id(c.line), ae_id(c.arc)); }
        // Arc-Arc
        for c in &self.sketch.concentric { union(&mut parent, ac_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_center_start { union(&mut parent, ac_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_center_end { union(&mut parent, ac_id(c.a), ae_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_center { union(&mut parent, as_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_center { union(&mut parent, ae_id(c.a), ac_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_start { union(&mut parent, as_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_start_end { union(&mut parent, as_id(c.a), ae_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_start { union(&mut parent, ae_id(c.a), as_id(c.b)); }
        for c in &self.sketch.coincident_arc_end_end { union(&mut parent, ae_id(c.a), ae_id(c.b)); }

        // Find locked roots
        let mut locked_roots: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.constraints.has_fix_x && p.constraints.has_fix_y {
                locked_roots.insert(find(&mut parent, pt_id(r)));
            }
        }
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            if !l.p1.optimize { locked_roots.insert(find(&mut parent, lp1_id(r))); }
            if !l.p2.optimize { locked_roots.insert(find(&mut parent, lp2_id(r))); }
        }
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            if !a.center.optimize { locked_roots.insert(find(&mut parent, ac_id(r))); }
        }

        // Collect locked vertices
        let mut pt_locked = std::collections::HashSet::new();
        let mut l_p1_locked = std::collections::HashSet::new();
        let mut l_p2_locked = std::collections::HashSet::new();
        let mut arc_c_locked = std::collections::HashSet::new();

        for r in self.sketch.points.refs() {
            if locked_roots.contains(&find(&mut parent, pt_id(r))) {
                pt_locked.insert(r.index());
            }
        }
        for r in self.sketch.lines.refs() {
            if locked_roots.contains(&find(&mut parent, lp1_id(r))) {
                l_p1_locked.insert(r.index());
            }
            if locked_roots.contains(&find(&mut parent, lp2_id(r))) {
                l_p2_locked.insert(r.index());
            }
        }
        for r in self.sketch.arcs.refs() {
            if locked_roots.contains(&find(&mut parent, ac_id(r))) {
                arc_c_locked.insert(r.index());
            }
        }

        (pt_locked, l_p1_locked, l_p2_locked, arc_c_locked)
    }

    // Check if a specific line endpoint is selected
    fn is_endpoint_selected(&self, line_ref: Ref<Line>, is_p1: bool) -> bool {
        self.selection.iter().any(|s| {
            if is_p1 {
                *s == Selection::LineP1(line_ref)
            } else {
                *s == Selection::LineP2(line_ref)
            }
        })
    }

    // Draw the canvas
    fn draw_canvas(&self, painter: &egui::Painter, rect: egui::Rect, mouse_screen: egui::Pos2) {
        let c = &self.colors;
        let empty_set = std::collections::HashSet::new();
        let (pt_locked, l_p1_locked, l_p2_locked, arc_c_locked) = if self.show_constraints {
            self.compute_locked_sets()
        } else {
            (empty_set.clone(), empty_set.clone(), empty_set.clone(), empty_set.clone())
        };

        // Compute which line/arc endpoints are coincident-connected (to hide blue dots)
        let mut connected_lp1: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut connected_lp2: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut connected_arc_s: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut connected_arc_e: std::collections::HashSet<u32> = std::collections::HashSet::new();
        // LL coincidences
        for c in &self.sketch.coincident_ll11 { connected_lp1.insert(c.a.index()); connected_lp1.insert(c.b.index()); }
        for c in &self.sketch.coincident_ll12 { connected_lp1.insert(c.a.index()); connected_lp2.insert(c.b.index()); }
        for c in &self.sketch.coincident_ll21 { connected_lp2.insert(c.a.index()); connected_lp1.insert(c.b.index()); }
        for c in &self.sketch.coincident_ll22 { connected_lp2.insert(c.a.index()); connected_lp2.insert(c.b.index()); }
        // Line-Arc
        for c in &self.sketch.coincident_lp1_arc_center { connected_lp1.insert(c.line.index()); }
        for c in &self.sketch.coincident_lp2_arc_center { connected_lp2.insert(c.line.index()); }
        for c in &self.sketch.coincident_lp1_arc_start { connected_lp1.insert(c.line.index()); connected_arc_s.insert(c.arc.index()); }
        for c in &self.sketch.coincident_lp2_arc_start { connected_lp2.insert(c.line.index()); connected_arc_s.insert(c.arc.index()); }
        for c in &self.sketch.coincident_lp1_arc_end { connected_lp1.insert(c.line.index()); connected_arc_e.insert(c.arc.index()); }
        for c in &self.sketch.coincident_lp2_arc_end { connected_lp2.insert(c.line.index()); connected_arc_e.insert(c.arc.index()); }
        // Arc-Arc
        for c in &self.sketch.coincident_arc_start_start { connected_arc_s.insert(c.a.index()); connected_arc_s.insert(c.b.index()); }
        for c in &self.sketch.coincident_arc_start_end { connected_arc_s.insert(c.a.index()); connected_arc_e.insert(c.b.index()); }
        for c in &self.sketch.coincident_arc_end_start { connected_arc_e.insert(c.a.index()); connected_arc_s.insert(c.b.index()); }
        for c in &self.sketch.coincident_arc_end_end { connected_arc_e.insert(c.a.index()); connected_arc_e.insert(c.b.index()); }

        // Compute constraint-highlighted entities
        let mut highlight_lines: std::collections::HashSet<u32> = std::collections::HashSet::new();
        let mut highlight_arcs: std::collections::HashSet<u32> = std::collections::HashSet::new();
        for sel in &self.selection {
            if let Selection::Constraint(id) = sel {
                let (lines, arcs) = self.constraint_entities(*id);
                for l in lines { highlight_lines.insert(l.index()); }
                for a in arcs { highlight_arcs.insert(a.index()); }
            }
        }
        let highlight_color = egui::Color32::from_rgb(255, 120, 180); // pink

        // Background
        painter.rect_filled(rect, 0.0, c.background);

        // Grid
        self.draw_grid(painter, rect);

        // Lines
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            let p1 = self.to_screen(l.p1.value);
            let p2 = self.to_screen(l.p2.value);

            let selected = self.selection.contains(&Selection::Line(r));
            let color = if selected { c.line_selected }
                else if highlight_lines.contains(&r.index()) { highlight_color }
                else { c.line };
            let width = if l.style == LineStyle::Solid { 2.0 } else { 1.0 };
            draw_styled_polyline(&painter, &[p1, p2], egui::Stroke::new(width, color), l.style);

            // Endpoints -- highlight individually if selected
            let p1_selected = self.is_endpoint_selected(r, true);
            let p2_selected = self.is_endpoint_selected(r, false);

            let ep1_color = if p1_selected { c.endpoint_selected }
                else if selected { c.endpoint_line_selected }
                else if l_p1_locked.contains(&r.index()) { c.point_locked }
                else { c.endpoint };
            let ep2_color = if p2_selected { c.endpoint_selected }
                else if selected { c.endpoint_line_selected }
                else if l_p2_locked.contains(&r.index()) { c.point_locked }
                else { c.endpoint };

            let ep1_radius = if p1_selected { 6.0 } else { 4.0 };
            let ep2_radius = if p2_selected { 6.0 } else { 4.0 };

            // Hide endpoint dot if coincident-connected, unless selected or locked
            let near_p1 = (mouse_screen.x - p1.x).powi(2) + (mouse_screen.y - p1.y).powi(2) < 225.0; // 15px
            let near_p2 = (mouse_screen.x - p2.x).powi(2) + (mouse_screen.y - p2.y).powi(2) < 225.0;
            let show_p1 = p1_selected || selected
                || l_p1_locked.contains(&r.index())
                || !connected_lp1.contains(&r.index())
                || near_p1;
            let show_p2 = p2_selected || selected
                || l_p2_locked.contains(&r.index())
                || !connected_lp2.contains(&r.index())
                || near_p2;
            if show_p1 { painter.circle_filled(p1, ep1_radius, ep1_color); }
            if show_p2 { painter.circle_filled(p2, ep2_radius, ep2_color); }

        }

        // Points (skip helper points)
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            if self.drag_point == Some(r) { continue; }
            let sp = self.to_screen(p.pos.value);
            let selected = self.selection.contains(&Selection::Point(r));
            let color = if selected { c.point_selected }
                else if pt_locked.contains(&r.index()) { c.point_locked }
                else { c.point };
            painter.circle_filled(sp, 5.0, color);
        }

        // Arcs
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let center = self.to_screen(a.center.value);
            let radius_px = a.radius.value as f32 * self.scale;
            let arc_selected = self.selection.contains(&Selection::Arc(r));
            let arc_color = if arc_selected { c.line_selected }
                else if highlight_arcs.contains(&r.index()) { highlight_color }
                else { c.arc };
            let arc_width = if a.style == LineStyle::Solid { 2.0 } else { 1.0 };
            let stroke = egui::Stroke::new(arc_width, arc_color);

            // Tessellate arc/circle
            let (sa, span) = if a.closed {
                (0.0, std::f64::consts::TAU)
            } else {
                let sa = a.start_angle.value;
                let ea = a.end_angle.value;
                let norm = |v: f64| -> f64 { let rv = v % std::f64::consts::TAU; if rv < 0.0 { rv + std::f64::consts::TAU } else { rv } };
                (sa, norm(ea - sa))
            };
            let n_segs = ((span * radius_px as f64 / 4.0).ceil() as usize).clamp(8, 256);
            let points: Vec<egui::Pos2> = (0..=n_segs).map(|i| {
                let t = sa + span * (i as f64 / n_segs as f64);
                self.to_screen(vect2d::new(
                    a.center.value.x + a.radius.value * t.cos(),
                    a.center.value.y + a.radius.value * t.sin(),
                ))
            }).collect();
            draw_styled_polyline(&painter, &points, stroke, a.style);

            if !a.closed {
                // Draw arc endpoints (hide if coincident-connected, unless selected)
                let start_sel = self.selection.contains(&Selection::ArcStart(r));
                let end_sel = self.selection.contains(&Selection::ArcEnd(r));
                let start_color = if start_sel { c.endpoint_selected } else { c.endpoint };
                let end_color = if end_sel { c.endpoint_selected } else { c.endpoint };
                let sp = points[0];
                let ep = *points.last().unwrap();
                let near_start = (mouse_screen.x - sp.x).powi(2) + (mouse_screen.y - sp.y).powi(2) < 225.0;
                let near_end = (mouse_screen.x - ep.x).powi(2) + (mouse_screen.y - ep.y).powi(2) < 225.0;
                let show_start = start_sel || arc_selected || !connected_arc_s.contains(&r.index()) || near_start;
                let show_end = end_sel || arc_selected || !connected_arc_e.contains(&r.index()) || near_end;
                if show_start { painter.circle_filled(points[0], if start_sel { 6.0 } else { 4.0 }, start_color); }
                if show_end { painter.circle_filled(*points.last().unwrap(), if end_sel { 6.0 } else { 4.0 }, end_color); }
            }
            // Center point
            let center_sel = self.selection.contains(&Selection::ArcCenter(r));
            let center_locked = arc_c_locked.contains(&r.index());
            let center_color = if center_sel { c.endpoint_selected }
                else if center_locked { c.point_locked }
                else { c.endpoint };
            painter.circle_filled(center, if center_sel { 5.0 } else { 3.0 }, center_color);
        }

        // Origin crosshair
        let origin = self.to_screen(vect2d::new(0.0, 0.0));
        let sz = 10.0;
        painter.line_segment(
            [egui::Pos2::new(origin.x - sz, origin.y), egui::Pos2::new(origin.x + sz, origin.y)],
            egui::Stroke::new(1.0, c.origin));
        painter.line_segment(
            [egui::Pos2::new(origin.x, origin.y - sz), egui::Pos2::new(origin.x, origin.y + sz)],
            egui::Stroke::new(1.0, c.origin));

        // Constraint markers (drawn with painter lines)
        for marker in &self.constraint_markers {
            let selected = self.selection.contains(&Selection::Constraint(marker.id));
            let color = if selected {
                egui::Color32::from_rgb(220, 40, 40)
            } else {
                c.constraint_marker
            };
            let w = if selected { 2.0 } else { 1.5 };
            let s = if selected { 7.0 } else { 5.0 }; // half-size
            let p = marker.pos;
            let stroke = egui::Stroke::new(w, color);
            match marker.symbol {
                ConstraintSymbol::H => {
                    // H shape: two verticals close together + horizontal crossbar
                    let g = s * 0.45;
                    painter.line_segment([egui::Pos2::new(p.x - g, p.y - s), egui::Pos2::new(p.x - g, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x + g, p.y - s), egui::Pos2::new(p.x + g, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x - g, p.y), egui::Pos2::new(p.x + g, p.y)], stroke);
                }
                ConstraintSymbol::V => {
                    // V shape: two diagonals meeting at bottom
                    painter.line_segment([egui::Pos2::new(p.x - s, p.y - s), egui::Pos2::new(p.x, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x + s, p.y - s), egui::Pos2::new(p.x, p.y + s)], stroke);
                }
                ConstraintSymbol::Parallel => {
                    // Two vertical parallel lines
                    let g = s * 0.35;
                    painter.line_segment([egui::Pos2::new(p.x - g, p.y - s), egui::Pos2::new(p.x - g, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x + g, p.y - s), egui::Pos2::new(p.x + g, p.y + s)], stroke);
                }
                ConstraintSymbol::Perpendicular => {
                    // T shape: horizontal line on bottom, vertical up from center
                    painter.line_segment([egui::Pos2::new(p.x - s, p.y + s), egui::Pos2::new(p.x + s, p.y + s)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x, p.y + s), egui::Pos2::new(p.x, p.y - s)], stroke);
                }
                ConstraintSymbol::Equal => {
                    // Two horizontal parallel lines
                    let g = s * 0.3;
                    painter.line_segment([egui::Pos2::new(p.x - s, p.y - g), egui::Pos2::new(p.x + s, p.y - g)], stroke);
                    painter.line_segment([egui::Pos2::new(p.x - s, p.y + g), egui::Pos2::new(p.x + s, p.y + g)], stroke);
                }
                ConstraintSymbol::Tangent => {
                    // Small circle with a diagonal line tangent at top-right
                    let r = s * 0.45;
                    let cx = p.x - s * 0.15;
                    let cy = p.y + s * 0.15;
                    painter.circle_stroke(egui::Pos2::new(cx, cy), r, stroke);
                    // Touch point at 45 deg, nudged outward by stroke width
                    let k = std::f32::consts::FRAC_1_SQRT_2;
                    let ro = r + w;
                    let tx = cx + ro * k;
                    let ty = cy - ro * k;
                    // Tangent direction is perpendicular to radius.
                    // Radius direction at 45 deg: (k, -k). Perpendicular: (k, k).
                    let half = s * 0.9;
                    painter.line_segment([
                        egui::Pos2::new(tx - k * half, ty - k * half),
                        egui::Pos2::new(tx + k * half, ty + k * half),
                    ], stroke);
                }
                ConstraintSymbol::Coincident => {
                    // Corner with dot: small filled square + lines going right and up
                    let d = s * 0.25;
                    painter.rect_filled(
                        egui::Rect::from_center_size(
                            egui::Pos2::new(p.x - s * 0.3, p.y + s * 0.3),
                            egui::Vec2::splat(d * 2.0),
                        ), 0.0, color);
                    // Line going right
                    painter.line_segment([
                        egui::Pos2::new(p.x - s * 0.3, p.y + s * 0.3),
                        egui::Pos2::new(p.x + s * 0.7, p.y + s * 0.3),
                    ], stroke);
                    // Line going up
                    painter.line_segment([
                        egui::Pos2::new(p.x - s * 0.3, p.y + s * 0.3),
                        egui::Pos2::new(p.x - s * 0.3, p.y - s * 0.7),
                    ], stroke);
                }
            }
        }

        // Dimension annotations
        let dim_color = egui::Color32::from_rgb(200, 100, 50);
        let dim_sel_color = egui::Color32::from_rgb(220, 40, 40);
        for (i, dim) in self.sketch.dimensions.iter().enumerate() {
            let selected = self.selection.contains(&Selection::Dimension(i));
            let color = if selected { dim_sel_color } else { dim_color };
            let is_radius = matches!(dim.kind, DimensionKind::ArcRadius(_));
            self.draw_dimension(&painter, &dim.kind, dim.value, dim.offset, dim.text_along, color, is_radius);
        }

        // Redraw selected and locked points/endpoints on top so they're not obscured
        for r in self.sketch.points.refs() {
            let p = &self.sketch.points[r];
            if p.helper { continue; }
            let selected = self.selection.contains(&Selection::Point(r));
            let locked = pt_locked.contains(&r.index());
            if selected || locked {
                let sp = self.to_screen(p.pos.value);
                let color = if selected { c.point_selected } else { c.point_locked };
                painter.circle_filled(sp, if selected { 6.0 } else { 5.0 }, color);
            }
        }
        for r in self.sketch.lines.refs() {
            let l = &self.sketch.lines[r];
            let p1s = self.is_endpoint_selected(r, true);
            let p2s = self.is_endpoint_selected(r, false);
            let p1l = l_p1_locked.contains(&r.index());
            let p2l = l_p2_locked.contains(&r.index());
            if p1s || p1l {
                let p1 = self.to_screen(l.p1.value);
                // Selected on top of locked: draw locked first, then selected ring
                if p1l { painter.circle_filled(p1, 5.0, c.point_locked); }
                if p1s { painter.circle_filled(p1, 6.0, c.endpoint_selected); }
                // If both, draw a green dot inside the orange
                if p1s && p1l { painter.circle_filled(p1, 3.0, c.point_locked); }
            }
            if p2s || p2l {
                let p2 = self.to_screen(l.p2.value);
                if p2l { painter.circle_filled(p2, 5.0, c.point_locked); }
                if p2s { painter.circle_filled(p2, 6.0, c.endpoint_selected); }
                if p2s && p2l { painter.circle_filled(p2, 3.0, c.point_locked); }
            }
        }
        for r in self.sketch.arcs.refs() {
            let a = &self.sketch.arcs[r];
            let cs = self.selection.contains(&Selection::ArcCenter(r));
            let cl = arc_c_locked.contains(&r.index());
            if cs || cl {
                let center = self.to_screen(a.center.value);
                if cl { painter.circle_filled(center, 4.0, c.point_locked); }
                if cs { painter.circle_filled(center, 5.0, c.endpoint_selected); }
                if cs && cl { painter.circle_filled(center, 2.5, c.point_locked); }
            }
            if !a.closed {
                if self.selection.contains(&Selection::ArcStart(r)) {
                    let sp = self.to_screen(arc_start_pos(a));
                    painter.circle_filled(sp, 6.0, c.endpoint_selected);
                }
                if self.selection.contains(&Selection::ArcEnd(r)) {
                    let ep = self.to_screen(arc_end_pos(a));
                    painter.circle_filled(ep, 6.0, c.endpoint_selected);
                }
            }
        }
    }

    fn draw_grid(&self, painter: &egui::Painter, rect: egui::Rect) {
        let grid_color = self.colors.grid;

        // Determine grid spacing based on zoom
        let mut spacing = 1.0_f32;
        while spacing * self.scale < 30.0 { spacing *= 5.0; }
        while spacing * self.scale > 150.0 { spacing /= 5.0; }

        let tl = self.to_sketch(rect.left_top());
        let br = self.to_sketch(rect.right_bottom());

        let x_start = (tl.x.min(br.x) as f32 / spacing).floor() * spacing;
        let x_end = (tl.x.max(br.x) as f32 / spacing).ceil() * spacing;
        let y_start = (tl.y.min(br.y) as f32 / spacing).floor() * spacing;
        let y_end = (tl.y.max(br.y) as f32 / spacing).ceil() * spacing;

        let mut x = x_start;
        while x <= x_end {
            let sx = self.to_screen(vect2d::new(x as f64, 0.0)).x;
            painter.line_segment(
                [egui::Pos2::new(sx, rect.top()), egui::Pos2::new(sx, rect.bottom())],
                egui::Stroke::new(0.5, grid_color));
            x += spacing;
        }
        let mut y = y_start;
        while y <= y_end {
            let sy = self.to_screen(vect2d::new(0.0, y as f64)).y;
            painter.line_segment(
                [egui::Pos2::new(rect.left(), sy), egui::Pos2::new(rect.right(), sy)],
                egui::Stroke::new(0.5, grid_color));
            y += spacing;
        }
    }
}

fn point_to_segment_dist(p: vect2d, a: vect2d, b: vect2d) -> f64 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-12 {
        return ((p.x - a.x).powi(2) + (p.y - a.y).powi(2)).sqrt();
    }
    let t = ((p.x - a.x) * dx + (p.y - a.y) * dy) / len2;
    let t = t.clamp(0.0, 1.0);
    let proj_x = a.x + t * dx;
    let proj_y = a.y + t * dy;
    ((p.x - proj_x).powi(2) + (p.y - proj_y).powi(2)).sqrt()
}

// Compute circumscribed circle arc from 3 points (start, end, mid on arc).
// Returns (center, radius, start_angle, end_angle, swapped) or None if collinear.
// `swapped` is true if start/end angles were swapped (arc goes the other way).
fn circumscribed_arc(p1: vect2d, p2: vect2d, p3: vect2d) -> Option<(vect2d, f64, f64, f64, bool)> {
    let ax = p1.x; let ay = p1.y;
    let bx = p2.x; let by = p2.y;
    let cx = p3.x; let cy = p3.y;
    let d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by));
    if d.abs() < 1e-12 { return None; } // collinear
    let aa = ax * ax + ay * ay;
    let bb = bx * bx + by * by;
    let cc = cx * cx + cy * cy;
    let ux = (aa * (by - cy) + bb * (cy - ay) + cc * (ay - by)) / d;
    let uy = (aa * (cx - bx) + bb * (ax - cx) + cc * (bx - ax)) / d;
    let center = vect2d::new(ux, uy);
    let radius = ((ax - ux).powi(2) + (ay - uy).powi(2)).sqrt();

    // Angles from center to start (p1) and end (p2)
    let sa = (ay - uy).atan2(ax - ux);
    let ea = (by - uy).atan2(bx - ux);

    // Check if mid point (p3) is on the arc going sa->ea counterclockwise.
    // If not, swap to go the other way.
    let ma = (cy - uy).atan2(cx - ux);

    // Normalize angle difference to [0, 2*PI)
    let norm = |a: f64| -> f64 { let r = a % std::f64::consts::TAU; if r < 0.0 { r + std::f64::consts::TAU } else { r } };
    let span_ccw = norm(ea - sa);
    let mid_ccw = norm(ma - sa);

    if mid_ccw < span_ccw {
        // Mid is on the CCW arc from sa to ea
        Some((center, radius, sa, ea, false))
    } else {
        // Mid is on the other side; swap start/end
        Some((center, radius, ea, sa, true))
    }
}

// Distance from point to arc curve. Returns (distance, nearest point on arc).
fn point_to_arc_dist(p: vect2d, a: &Arc) -> (f64, vect2d) {
    let dx = p.x - a.center.value.x;
    let dy = p.y - a.center.value.y;
    let dist_to_center = (dx * dx + dy * dy).sqrt();
    let angle = dy.atan2(dx);
    let r = a.radius.value;

    if a.closed {
        // Full circle: nearest point is projection onto circle
        if dist_to_center < 1e-12 {
            return (r, vect2d::new(a.center.value.x + r, a.center.value.y));
        }
        let proj = vect2d::new(
            a.center.value.x + r * dx / dist_to_center,
            a.center.value.y + r * dy / dist_to_center,
        );
        ((dist_to_center - r).abs(), proj)
    } else {
        // Partial arc: check if angle falls within arc range
        let sa = a.start_angle.value;
        let ea = a.end_angle.value;
        let norm = |v: f64| -> f64 { let rv = v % std::f64::consts::TAU; if rv < 0.0 { rv + std::f64::consts::TAU } else { rv } };
        let span = norm(ea - sa);
        let a_norm = norm(angle - sa);

        if a_norm <= span {
            // Angle is within arc range
            if dist_to_center < 1e-12 {
                let proj = vect2d::new(
                    a.center.value.x + r * angle.cos(),
                    a.center.value.y + r * angle.sin(),
                );
                return (r, proj);
            }
            let proj = vect2d::new(
                a.center.value.x + r * dx / dist_to_center,
                a.center.value.y + r * dy / dist_to_center,
            );
            ((dist_to_center - r).abs(), proj)
        } else {
            // Outside arc range: nearest is one of the endpoints
            let sp = arc_start_pos(a);
            let ep = arc_end_pos(a);
            let ds = ((p.x - sp.x).powi(2) + (p.y - sp.y).powi(2)).sqrt();
            let de = ((p.x - ep.x).powi(2) + (p.y - ep.y).powi(2)).sqrt();
            if ds < de { (ds, sp) } else { (de, ep) }
        }
    }
}

// Draw a polyline with the given style (solid, dashed, dash-dot).
// `points` is a slice of screen-space positions.
fn draw_styled_polyline(painter: &egui::Painter, points: &[egui::Pos2], stroke: egui::Stroke, style: LineStyle) {
    match style {
        LineStyle::Solid => {
            for w in points.windows(2) {
                painter.line_segment([w[0], w[1]], stroke);
            }
        }
        LineStyle::Dashed => {
            draw_pattern_polyline(painter, points, stroke, &[10.0, 6.0]);
        }
        LineStyle::DashDot => {
            draw_pattern_polyline(painter, points, stroke, &[10.0, 4.0, 2.0, 4.0]);
        }
    }
}

// Draw a polyline with a repeating dash pattern (lengths in pixels).
// Even indices are drawn, odd indices are gaps.
fn draw_pattern_polyline(painter: &egui::Painter, points: &[egui::Pos2], stroke: egui::Stroke, pattern: &[f32]) {
    if points.len() < 2 || pattern.is_empty() { return; }
    let mut pat_idx = 0;
    let mut pat_remaining = pattern[0];
    let mut drawing = true; // even indices = draw, odd = gap
    let mut seg_start = points[0];

    for w in points.windows(2) {
        let (a, b) = (w[0], w[1]);
        let dx = b.x - a.x;
        let dy = b.y - a.y;
        let seg_len = (dx * dx + dy * dy).sqrt();
        if seg_len < 1e-6 { continue; }
        let ux = dx / seg_len;
        let uy = dy / seg_len;
        let mut consumed = 0.0f32;

        // Walk along this segment
        while consumed < seg_len - 0.01 {
            let remaining_seg = seg_len - consumed;
            if pat_remaining <= remaining_seg {
                // Pattern element ends within this segment
                let end_x = a.x + ux * (consumed + pat_remaining);
                let end_y = a.y + uy * (consumed + pat_remaining);
                let end = egui::Pos2::new(end_x, end_y);
                if drawing {
                    painter.line_segment([seg_start, end], stroke);
                }
                consumed += pat_remaining;
                seg_start = end;
                // Advance pattern
                drawing = !drawing;
                pat_idx = (pat_idx + 1) % pattern.len();
                pat_remaining = pattern[pat_idx];
            } else {
                // Segment ends before pattern element
                pat_remaining -= remaining_seg;
                if drawing {
                    painter.line_segment([seg_start, b], stroke);
                }
                seg_start = b;
                consumed = seg_len;
            }
        }
    }
}

fn arc_start_pos(a: &Arc) -> vect2d {
    vect2d::new(
        a.center.value.x + a.radius.value * a.start_angle.value.cos(),
        a.center.value.y + a.radius.value * a.start_angle.value.sin(),
    )
}

fn arc_end_pos(a: &Arc) -> vect2d {
    vect2d::new(
        a.center.value.x + a.radius.value * a.end_angle.value.cos(),
        a.center.value.y + a.radius.value * a.end_angle.value.sin(),
    )
}

fn project_onto_segment(p: vect2d, a: vect2d, b: vect2d) -> vect2d {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-12 { return a; }
    let t = (((p.x - a.x) * dx + (p.y - a.y) * dy) / len2).clamp(0.0, 1.0);
    vect2d::new(a.x + t * dx, a.y + t * dy)
}

// ---------------------------------------------------------------------------
// eframe App impl
// ---------------------------------------------------------------------------

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Check for pending file load from async dialog
        let pending_json = self.pending_load.lock().unwrap().take();
        if let Some(json) = pending_json {
            self.load_from_json(&json);
        }

        // Apply egui visuals for widgets (side panel, buttons, etc.)
        ctx.set_visuals(if self.dark_mode { egui::Visuals::dark() } else { egui::Visuals::light() });

        // Side panel: toolbar
        egui::SidePanel::left("toolbar").min_width(140.0).show(ctx, |ui| {
            // Theme toggle
            ui.horizontal(|ui| {
                let theme_label = if self.dark_mode { "Light" } else { "Dark" };
                if ui.button(theme_label).clicked() {
                    self.dark_mode = !self.dark_mode;
                    self.colors = if self.dark_mode { ColorScheme::dark() } else { ColorScheme::light() };
                }
                let constr_label = if self.show_constraints { "Hide Cstr" } else { "Show Cstr" };
                if ui.button(constr_label).clicked() {
                    self.show_constraints = !self.show_constraints;
                }
            });
            ui.separator();

            ui.heading("Tools");
            ui.separator();
            if ui.selectable_label(self.tool == Tool::Select, "Select (S)").clicked() {
                self.tool = Tool::Select;
            }
            if ui.selectable_label(self.tool == Tool::DrawPoint, "Point (P)").clicked() {
                self.tool = Tool::DrawPoint;
            }
            if ui.selectable_label(self.tool == Tool::DrawLine, "Line (L)").clicked() {
                self.tool = Tool::DrawLine;
                self.line_draw = None;
            }
            if ui.selectable_label(self.tool == Tool::DrawCircle, "Circle (O)").clicked() {
                self.tool = Tool::DrawCircle;
                self.circle_draw = None;
            }
            if ui.selectable_label(self.tool == Tool::DrawArc, "Arc (A)").clicked() {
                self.tool = Tool::DrawArc;
                self.arc_draw = None;
            }

            ui.separator();
            ui.heading("Constraints");
            ui.separator();

            let constraint_btn = |ui: &mut egui::Ui, this: &mut EditorApp, ct: ConstraintType, label: &str| {
                let active = matches!(this.tool, Tool::ConstraintMode(t) if t == ct);
                let can_apply = this.can_apply_constraint(ct);
                let can_enter = this.could_enter_constraint_mode(ct);
                let enabled = can_apply || can_enter;
                let btn = egui::Button::new(label).selected(active);
                if ui.add_enabled(enabled, btn).clicked() {
                    this.try_apply_or_enter_mode(ct);
                }
            };
            constraint_btn(ui, self, ConstraintType::Horizontal, "Horizontal (H)");
            constraint_btn(ui, self, ConstraintType::Vertical, "Vertical (V)");
            constraint_btn(ui, self, ConstraintType::Coincident, "Coincident (C)");
            constraint_btn(ui, self, ConstraintType::Parallel, "Parallel");
            constraint_btn(ui, self, ConstraintType::Perpendicular, "Perpendicular");
            constraint_btn(ui, self, ConstraintType::EqualLength, "Equal (=)");
            constraint_btn(ui, self, ConstraintType::Tangent, "Tangent (T)");
            constraint_btn(ui, self, ConstraintType::Lock, "Lock (K)");
            constraint_btn(ui, self, ConstraintType::ToggleStyle, "Style (X)");

            ui.separator();
            let dim_active = matches!(self.tool, Tool::Dimension);
            let dim_btn = egui::Button::new("Dimension (D)").selected(dim_active);
            if ui.add(dim_btn).clicked() {
                self.tool = Tool::Dimension;
                self.dim_editing = false;
                self.dim_kind = None;
            }

            // Dimension value input
            if self.dim_editing {
                ui.separator();
                let label = if self.dim_edit_index.is_some() { "Edit dimension:" } else { "New dimension:" };
                ui.label(label);
                let response = ui.text_edit_singleline(&mut self.dim_input);
                // Auto-focus and select all text when first shown
                if response.gained_focus() {
                    let mut state = egui::TextEdit::load_state(ui.ctx(), response.id).unwrap_or_default();
                    state.cursor.set_char_range(Some(egui::text::CCursorRange::two(
                        egui::text::CCursor::new(0),
                        egui::text::CCursor::new(self.dim_input.len()),
                    )));
                    egui::TextEdit::store_state(ui.ctx(), response.id, state);
                }
                if response.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    if let Ok(value) = self.dim_input.parse::<f64>() {
                        self.begin_group();
                        if let Some(edit_idx) = self.dim_edit_index.take() {
                            // Editing existing: remove old, add new with same offset
                            let offset = self.dim_offset;
                            let kind = self.dim_kind.take().unwrap();
                            self.exec(Action::RemoveDimension { index: edit_idx });
                            self.exec(Action::AddDimension { kind, value });
                            // Update offset and text_along on the newly created dimension
                            if let Some(d) = self.sketch.dimensions.last_mut() {
                                d.offset = offset;
                                d.text_along = self.dim_text_along;
                            }
                        } else if let Some(kind) = self.dim_kind.take() {
                            // New dimension
                            self.exec(Action::AddDimension { kind, value });
                            if let Some(d) = self.sketch.dimensions.last_mut() {
                                d.offset = self.dim_offset;
                                d.text_along = self.dim_text_along;
                            }
                        }
                    }
                    self.dim_editing = false;
                    self.dim_placing = false;
                    self.dim_edit_index = None;
                    self.selection.clear();
                } else if !response.has_focus() && self.dim_editing {
                    response.request_focus();
                }
            }

            ui.separator();
            ui.heading("File");
            ui.separator();
            ui.horizontal(|ui| {
                if ui.button("Save").clicked() {
                    if let Ok(json) = serde_json::to_string_pretty(&self.sketch) {
                        let json_bytes = json.into_bytes();
                        spawn_async(async move {
                            if let Some(handle) = rfd::AsyncFileDialog::new()
                                .add_filter("Sketch JSON", &["json"])
                                .set_file_name("sketch.json")
                                .save_file().await
                            {
                                let _ = handle.write(&json_bytes).await;
                            }
                        });
                    }
                }
                if ui.button("Open").clicked() {
                    let pending = self.pending_load.clone();
                    spawn_async(async move {
                        if let Some(file) = rfd::AsyncFileDialog::new()
                            .add_filter("Sketch JSON", &["json"])
                            .pick_file().await
                        {
                            let data = file.read().await;
                            if let Ok(json) = String::from_utf8(data) {
                                *pending.lock().unwrap() = Some(json);
                            }
                        }
                    });
                }
            });

            ui.separator();
            ui.heading("History");
            ui.separator();

            ui.horizontal(|ui| {
                if ui.add_enabled(self.history.can_undo(), egui::Button::new("Undo")).clicked() {
                    if let Some(restored) = self.history.undo() {
                        self.sketch = restored;
                        self.selection.clear();
                    }
                }
                if ui.add_enabled(self.history.can_redo(), egui::Button::new("Redo")).clicked() {
                    if let Some(restored) = self.history.redo() {
                        self.sketch = restored;
                        self.selection.clear();
                    }
                }
            });
            ui.label(format!("Actions: {}/{}", self.history.cursor, self.history.actions.len()));

            ui.separator();
            ui.label(format!("Points: {}  Lines: {}  Arcs: {}",
                self.sketch.points.len(), self.sketch.lines.len(), self.sketch.arcs.len()));
            if !self.selection.is_empty() {
                let names: Vec<String> = self.selection.iter().filter_map(|s| {
                    match *s {
                        Selection::Point(r) => Some(self.sketch.points[r].name.clone()),
                        Selection::Line(r) => Some(self.sketch.lines[r].name.clone()),
                        Selection::LineP1(r) => Some(format!("{}.p1", self.sketch.lines[r].name)),
                        Selection::LineP2(r) => Some(format!("{}.p2", self.sketch.lines[r].name)),
                        Selection::Arc(r) => Some(self.sketch.arcs[r].name.clone()),
                        Selection::ArcCenter(r) => Some(format!("{}.c", self.sketch.arcs[r].name)),
                        Selection::ArcStart(r) => Some(format!("{}.s", self.sketch.arcs[r].name)),
                        Selection::ArcEnd(r) => Some(format!("{}.e", self.sketch.arcs[r].name)),
                        Selection::Constraint(id) => Some(self.describe_constraint(id)),
                        Selection::Dimension(i) => {
                            if i < self.sketch.dimensions.len() {
                                let d = &self.sketch.dimensions[i];
                                Some(format!("{} = {:.2}", d.name, d.value))
                            } else { Some("dim?".to_string()) }
                        }
                    }
                }).collect();
                ui.label(format!("Selected: {}", names.join(", ")));
            }
        });

        // Central panel: canvas
        egui::CentralPanel::default().show(ctx, |ui| {
            let (response, painter) = ui.allocate_painter(
                ui.available_size(),
                egui::Sense::click_and_drag(),
            );
            let rect = response.rect;

            // Auto-fit after file load
            if self.pending_fit {
                self.fit_all(rect);
                self.pending_fit = false;
            }

            // Keyboard shortcuts
            if ui.input(|i| i.key_pressed(egui::Key::S) && !i.modifiers.ctrl && !i.modifiers.mac_cmd) { self.tool = Tool::Select; }
            if ui.input(|i| i.key_pressed(egui::Key::P)) { self.tool = Tool::DrawPoint; }
            if ui.input(|i| i.key_pressed(egui::Key::L)) {
                self.tool = Tool::DrawLine;
                self.line_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::O) && !i.modifiers.ctrl && !i.modifiers.mac_cmd) {
                self.tool = Tool::DrawCircle;
                self.circle_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::A)) {
                self.tool = Tool::DrawArc;
                self.arc_draw = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::H)) { self.try_apply_or_enter_mode(ConstraintType::Horizontal); }
            if ui.input(|i| i.key_pressed(egui::Key::V)) { self.try_apply_or_enter_mode(ConstraintType::Vertical); }
            if ui.input(|i| i.key_pressed(egui::Key::C)) { self.try_apply_or_enter_mode(ConstraintType::Coincident); }
            if ui.input(|i| i.key_pressed(egui::Key::K)) { self.try_apply_or_enter_mode(ConstraintType::Lock); }
            if ui.input(|i| i.key_pressed(egui::Key::T)) { self.try_apply_or_enter_mode(ConstraintType::Tangent); }
            if ui.input(|i| i.key_pressed(egui::Key::X)) { self.try_apply_or_enter_mode(ConstraintType::ToggleStyle); }
            if ui.input(|i| i.key_pressed(egui::Key::D) && !i.modifiers.ctrl && !i.modifiers.mac_cmd) {
                self.tool = Tool::Dimension;
                self.dim_editing = false;
                self.dim_kind = None;
            }
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                self.selection.clear();
                self.line_draw = None;
                self.circle_draw = None;
                self.arc_draw = None;
                self.dim_editing = false;
                self.dim_kind = None;
                self.dim_placing = false;
                self.dim_edit_index = None;
                self.tool = Tool::Select;
            }

            // Delete selected entities/constraints with Backspace/Delete
            if ui.input(|i| i.key_pressed(egui::Key::Backspace) || i.key_pressed(egui::Key::Delete)) {
                let sel = self.selection.clone();
                if !sel.is_empty() {
                    self.begin_group();
                    for s in &sel {
                        match *s {
                            Selection::Point(r) => { self.exec(Action::DeletePoint { point: r }); }
                            Selection::Line(r) => { self.exec(Action::DeleteLine { line: r }); }
                            Selection::Arc(r) => { self.exec(Action::DeleteArc { arc: r }); }
                            Selection::Constraint(id) => { self.delete_constraint(id); }
                            Selection::Dimension(i) => { self.exec(Action::RemoveDimension { index: i }); }
                            _ => {} // endpoints aren't deletable on their own
                        }
                    }
                    self.selection.clear();
                }
            }

            // Undo/redo keyboard shortcuts
            let ctrl = ui.input(|i| i.modifiers.ctrl || i.modifiers.mac_cmd);
            let shift = ui.input(|i| i.modifiers.shift);
            if ctrl && shift && ui.input(|i| i.key_pressed(egui::Key::Z)) {
                if let Some(restored) = self.history.redo() {
                    self.sketch = restored;
                    self.selection.clear();
                }
            } else if ctrl && ui.input(|i| i.key_pressed(egui::Key::Z)) {
                if let Some(restored) = self.history.undo() {
                    self.sketch = restored;
                    self.selection.clear();
                }
            }
            if ctrl && ui.input(|i| i.key_pressed(egui::Key::S)) {
                if let Ok(json) = serde_json::to_string_pretty(&self.sketch) {
                    let json_bytes = json.into_bytes();
                    spawn_async(async move {
                        if let Some(handle) = rfd::AsyncFileDialog::new()
                            .add_filter("Sketch JSON", &["json"])
                            .set_file_name("sketch.json")
                            .save_file().await
                        {
                            let _ = handle.write(&json_bytes).await;
                        }
                    });
                }
            }
            if ctrl && ui.input(|i| i.key_pressed(egui::Key::O)) {
                let pending = self.pending_load.clone();
                spawn_async(async move {
                    if let Some(file) = rfd::AsyncFileDialog::new()
                        .add_filter("Sketch JSON", &["json"])
                        .pick_file().await
                    {
                        let data = file.read().await;
                        if let Ok(json) = String::from_utf8(data) {
                            *pending.lock().unwrap() = Some(json);
                        }
                    }
                });
            }

            // Zoom (scroll wheel)
            let scroll = ui.input(|i| i.raw_scroll_delta.y);
            if scroll != 0.0 {
                let zoom_factor = if scroll > 0.0 { 1.1 } else { 1.0 / 1.1 };
                if let Some(mouse) = ui.input(|i| i.pointer.hover_pos()) {
                    // Zoom toward mouse position
                    let before = self.to_sketch(mouse);
                    self.scale *= zoom_factor;
                    self.scale = self.scale.clamp(1e-4, 1e7);
                    let after = self.to_screen(before);
                    self.offset += mouse - after;
                }
            }

            // Pan (middle mouse drag) / Fit all (middle double-click)
            if response.double_clicked_by(egui::PointerButton::Middle) {
                self.fit_all(rect);
            } else if response.dragged_by(egui::PointerButton::Middle) {
                self.offset += response.drag_delta();
            }

            // Get mouse position in sketch coords
            let mouse_screen = response.hover_pos().unwrap_or(egui::Pos2::ZERO);
            let mouse_sketch = self.to_sketch(mouse_screen);
            let hit_threshold = 15.0 / self.scale as f64;  // screen pixels -> sketch units

            // Tool-specific input handling
            match self.tool {
                Tool::Select => {
                    // Double-click on dimension to edit value
                    if response.double_clicked_by(egui::PointerButton::Primary) {
                        let mut edited = false;
                        for (i, dim) in self.sketch.dimensions.iter().enumerate() {
                            let (ts, te) = self.dim_text_segment(dim);
                            let d = Self::screen_point_to_segment_dist(mouse_screen, ts, te);
                            if d < 15.0 {
                                self.dim_input = format!("{:.4}", dim.value);
                                self.dim_kind = Some(dim.kind.clone());
                                self.dim_offset = dim.offset;
                                self.dim_edit_index = Some(i);
                                self.dim_editing = true;
                                self.dim_placing = false;
                                self.tool = Tool::Dimension;
                                self.selection.clear();
                                self.selection.push(Selection::Dimension(i));
                                edited = true;
                                break;
                            }
                        }
                        if !edited {
                            // Normal double-click behavior (if any)
                        }
                    }

                    // Drag: geometry or dimension
                    if response.dragged_by(egui::PointerButton::Primary) {
                        if self.grab.is_none() && self.drag_dimension.is_none() {
                            // First drag frame: try to grab a dimension first, then geometry
                            let mut grabbed_dim = false;
                            if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                                if let Selection::Dimension(i) = sel {
                                    self.drag_dimension = Some(i);
                                    grabbed_dim = true;
                                }
                            }
                            if !grabbed_dim {
                                if let Some(target) = self.hit_test(mouse_sketch, hit_threshold) {
                                    self.start_drag(target, mouse_sketch);
                                }
                            }
                        }
                        if let Some(dim_idx) = self.drag_dimension {
                            // Update dimension offset and text_along from mouse
                            if dim_idx < self.sketch.dimensions.len() {
                                let kind = self.sketch.dimensions[dim_idx].kind.clone();
                                let is_radius = matches!(kind, DimensionKind::ArcRadius(_));
                                if is_radius {
                                    if let DimensionKind::ArcRadius(r) = kind {
                                        let a = &self.sketch.arcs[r];
                                        let angle = (mouse_sketch.y - a.center.value.y)
                                            .atan2(mouse_sketch.x - a.center.value.x);
                                        self.sketch.dimensions[dim_idx].offset = vect2d::new(angle, 0.0);
                                    }
                                } else {
                                    // Decompose mouse into perpendicular offset and along-line position
                                    let (p1, p2) = self.dim_endpoints(&kind);
                                    let ddx = p2.x - p1.x;
                                    let ddy = p2.y - p1.y;
                                    let len = (ddx * ddx + ddy * ddy).sqrt().max(1e-12);
                                    let ux = ddx / len;
                                    let uy = ddy / len;
                                    let nx = -ddy / len;
                                    let ny = ddx / len;
                                    let mx = (p1.x + p2.x) / 2.0;
                                    let my = (p1.y + p2.y) / 2.0;
                                    let rel_x = mouse_sketch.x - mx;
                                    let rel_y = mouse_sketch.y - my;
                                    let perp = rel_x * nx + rel_y * ny;
                                    let along = (rel_x * ux + rel_y * uy) / len;
                                    self.sketch.dimensions[dim_idx].offset = vect2d::new(0.0, perp);
                                    self.sketch.dimensions[dim_idx].text_along = along;
                                }
                            }
                            ctx.request_repaint();
                        }
                        if self.grab.is_some() {
                            self.update_drag(mouse_sketch);
                            ctx.request_repaint();
                        }
                    }

                    // End drag
                    if response.drag_stopped_by(egui::PointerButton::Primary) {
                        if self.grab.is_some() {
                            self.end_drag(hit_threshold);
                        }
                        self.drag_dimension = None;
                    }

                    // Click (no drag): select/deselect
                    if response.clicked_by(egui::PointerButton::Primary) {
                        if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                            self.toggle_selection(sel);
                        } else {
                            self.selection.clear();
                        }
                    }
                }

                Tool::DrawPoint => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.begin_group();
                        let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                        let pos = snap.map_or(mouse_sketch, |(p, _)| p);
                        let action = Action::AddPoint { pos };
                        self.exec(action);
                        let new_point = Ref::new(self.sketch.points.slot_count() as u32 - 1);
                        if let Some((_, snap_target)) = snap {
                            self.apply_snap_coincident_point(snap_target, new_point);
                        }
                    }
                }

                Tool::DrawLine => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.begin_group();
                        if let Some(state) = self.line_draw.take() {
                            // Second click: finish line
                            // Snap end point to nearby entity
                            let end_snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let end_pos = end_snap.map_or(mouse_sketch, |(pos, _)| pos);

                            let action = Action::AddLine { p1: state.start, p2: end_pos };
                            self.exec(action);
                            let new_line = Ref::new(self.sketch.lines.slot_count() as u32 - 1);

                            // Auto-coincident for start snap
                            if let Some(snap) = state.snap_start {
                                self.apply_snap_coincident(snap, new_line, true);
                            }
                            // Auto-coincident for end snap
                            if let Some((_, snap)) = end_snap {
                                self.apply_snap_coincident(snap, new_line, false);
                            }

                            // Chain: start next line from end of this one
                            self.line_draw = Some(LineDrawState {
                                start: end_pos,
                                snap_start: Some(SnapTarget::LineP2(new_line)),
                            });
                        } else {
                            // First click: start line, snap to nearby entity
                            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let start_pos = snap.map_or(mouse_sketch, |(pos, _)| pos);
                            self.line_draw = Some(LineDrawState {
                                start: start_pos,
                                snap_start: snap.map(|(_, t)| t),
                            });
                        }
                    }
                }

                Tool::DrawCircle => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.begin_group();
                        if let Some(state) = self.circle_draw.take() {
                            // Second click: edge point
                            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let edge = snap.map_or(mouse_sketch, |(p, _)| p);
                            self.exec(Action::AddCircle { center: state.center, edge });
                            let new_arc = Ref::new(self.sketch.arcs.slot_count() as u32 - 1);

                            // Auto-coincident for center
                            if let Some(s) = state.snap_center {
                                self.apply_snap_coincident_arc(s, new_arc, ArcPoint::Center, state.center);
                            }
                            // Auto-coincident for edge (point on circle)
                            if let Some((_, s)) = snap {
                                // Edge point is "on arc" - use a helper point for PointOnArc
                                self.exec(Action::AddHelperPoint { pos: edge });
                                let helper = Ref::new(self.sketch.points.slot_count() as u32 - 1);
                                self.exec(Action::ApplyPointOnArc { point: helper, arc: new_arc });
                                self.apply_snap_coincident_point(s, helper);
                            }
                        } else {
                            // First click: center
                            let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                            let center = snap.map_or(mouse_sketch, |(p, _)| p);
                            self.circle_draw = Some(CircleDrawState {
                                center,
                                snap_center: snap.map(|(_, t)| t),
                            });
                        }
                    }
                }

                Tool::DrawArc => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        self.begin_group();
                        let snap = self.find_snap_target(mouse_sketch, hit_threshold);
                        let pos = snap.map_or(mouse_sketch, |(p, _)| p);
                        let snap_target = snap.map(|(_, t)| t);

                        if let Some(state) = self.arc_draw.take() {
                            if let Some((end, snap_end)) = state.end {
                                // Third click: mid point on arc, create it
                                let swapped = circumscribed_arc(state.start, end, pos)
                                    .map_or(false, |(_, _, _, _, s)| s);
                                self.exec(Action::AddArc { start: state.start, end, mid: pos, swapped });
                                let new_arc = Ref::new(self.sketch.arcs.slot_count() as u32 - 1);

                                // When swapped, arc.start_angle corresponds to `end` click
                                // and arc.end_angle corresponds to `start` click
                                let (start_ap, end_ap) = if swapped {
                                    (ArcPoint::End, ArcPoint::Start)
                                } else {
                                    (ArcPoint::Start, ArcPoint::End)
                                };

                                // Auto-coincident for start click
                                if let Some(s) = state.snap_start {
                                    self.apply_snap_coincident_arc(s, new_arc, start_ap, state.start);
                                }
                                // Auto-coincident for end click
                                if let Some(s) = snap_end {
                                    self.apply_snap_coincident_arc(s, new_arc, end_ap, end);
                                }
                                // Auto-coincident for mid (point on arc) - needs helper point
                                if let Some(s) = snap_target {
                                    self.exec(Action::AddHelperPoint { pos });
                                    let helper = Ref::new(self.sketch.points.slot_count() as u32 - 1);
                                    self.exec(Action::ApplyPointOnArc { point: helper, arc: new_arc });
                                    self.apply_snap_coincident_point(s, helper);
                                }
                            } else {
                                // Second click: end point
                                self.arc_draw = Some(ArcDrawState {
                                    start: state.start,
                                    snap_start: state.snap_start,
                                    end: Some((pos, snap_target)),
                                });
                            }
                        } else {
                            // First click: start point
                            self.arc_draw = Some(ArcDrawState {
                                start: pos,
                                snap_start: snap_target,
                                end: None,
                            });
                        }
                    }
                }

                Tool::ConstraintMode(ct) => {
                    if response.clicked_by(egui::PointerButton::Primary) {
                        // Find what was clicked
                        if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                            // Only accept valid entities for this constraint
                            if Self::is_valid_for_constraint(ct, &sel) {
                                self.toggle_selection(sel);
                                // Check if we can now apply
                                if self.can_apply_constraint(ct) {
                                    match ct {
                                        ConstraintType::Horizontal => self.apply_horizontal(),
                                        ConstraintType::Vertical => self.apply_vertical(),
                                        ConstraintType::Coincident => self.apply_coincident(),
                                        ConstraintType::Parallel => self.apply_parallel(),
                                        ConstraintType::Perpendicular => self.apply_perpendicular(),
                                        ConstraintType::EqualLength => self.apply_equal_length(),
                                        ConstraintType::Tangent => self.apply_tangent(),
                                        ConstraintType::Lock => self.apply_lock(),
                                        ConstraintType::ToggleStyle => self.apply_toggle_style(),
                                    }
                                    self.selection.clear();
                                    // Stay in constraint mode for more
                                }
                            }
                        } else {
                            self.selection.clear();
                        }
                    }
                }

                Tool::Dimension => {
                    if self.dim_placing {
                        // Phase 2: positioning with mouse, click to confirm
                        if let Some(ref kind) = self.dim_kind {
                            if matches!(kind, DimensionKind::ArcRadius(r) if self.sketch.arcs.contains(*r)) {
                                if let DimensionKind::ArcRadius(r) = kind {
                                    let a = &self.sketch.arcs[*r];
                                    let angle = (mouse_sketch.y - a.center.value.y)
                                        .atan2(mouse_sketch.x - a.center.value.x);
                                    self.dim_offset = vect2d::new(angle, 0.0);
                                    self.dim_text_along = 0.0;
                                }
                            } else {
                                // Decompose mouse into perpendicular and along
                                let (p1, p2) = self.dim_endpoints(kind);
                                let ddx = p2.x - p1.x;
                                let ddy = p2.y - p1.y;
                                let len = (ddx * ddx + ddy * ddy).sqrt().max(1e-12);
                                let ux = ddx / len;
                                let uy = ddy / len;
                                let nx = -ddy / len;
                                let ny = ddx / len;
                                let mx = (p1.x + p2.x) / 2.0;
                                let my = (p1.y + p2.y) / 2.0;
                                let rel_x = mouse_sketch.x - mx;
                                let rel_y = mouse_sketch.y - my;
                                let perp = rel_x * nx + rel_y * ny;
                                let along = (rel_x * ux + rel_y * uy) / len;
                                self.dim_offset = vect2d::new(0.0, perp);
                                self.dim_text_along = along;
                            }
                        }
                        if response.clicked_by(egui::PointerButton::Primary) {
                            // Confirm position, enter text input
                            self.dim_placing = false;
                            self.dim_editing = true;
                        }
                    } else if !self.dim_editing {
                        // Phase 1: selecting entities
                        if response.clicked_by(egui::PointerButton::Primary) {
                            if let Some(sel) = self.hit_test_selection(mouse_sketch, hit_threshold) {
                                match &sel {
                                    Selection::Line(_) | Selection::Arc(_)
                                    | Selection::Point(_) | Selection::LineP1(_) | Selection::LineP2(_)
                                    | Selection::ArcCenter(_) | Selection::ArcStart(_) | Selection::ArcEnd(_) => {
                                        self.toggle_selection(sel);
                                    }
                                    _ => {}
                                }
                                // Check if we can form a dimension
                                if let Some(kind) = self.selection_to_dim_kind() {
                                    let measured = self.measure_dimension(&kind);
                                    self.dim_input = format!("{:.4}", measured);
                                    self.dim_kind = Some(kind);
                                    self.dim_placing = true;
                                    self.dim_offset = vect2d::new(0.0, 1.0);
                                    self.dim_text_along = 0.0;
                                }
                            } else {
                                self.selection.clear();
                            }
                        }
                    }
                    // Double-click on existing dimension to edit
                    if response.double_clicked_by(egui::PointerButton::Primary) {
                        // Check if clicking on a dimension
                        for (i, dim) in self.sketch.dimensions.iter().enumerate() {
                            let (ts, te) = self.dim_text_segment(dim);
                            let d = Self::screen_point_to_segment_dist(mouse_screen, ts, te);
                            if d < 15.0 {
                                self.dim_input = format!("{:.4}", dim.value);
                                self.dim_kind = Some(dim.kind.clone());
                                self.dim_offset = dim.offset;
                                self.dim_edit_index = Some(i);
                                self.dim_editing = true;
                                self.dim_placing = false;
                                break;
                            }
                        }
                    }
                }
            }

            // Build constraint markers and draw canvas
            if self.show_constraints {
                self.build_constraint_markers();
            } else {
                self.constraint_markers.clear();
            }
            self.draw_canvas(&painter, rect, mouse_screen);

            // Dimension preview while placing (not when editing an existing dimension)
            if (self.dim_placing || (self.dim_editing && self.dim_edit_index.is_none())) && self.dim_kind.is_some() {
                let kind = self.dim_kind.clone().unwrap();
                let measured = self.measure_dimension(&kind);
                let is_radius = matches!(kind, DimensionKind::ArcRadius(_));
                let preview_color = egui::Color32::from_rgba_premultiplied(200, 100, 50, 180);
                self.draw_dimension(&painter, &kind, measured, self.dim_offset, self.dim_text_along, preview_color, is_radius);
            }

            // Draw overlays ON TOP of canvas: preview line and cursor crosshair
            if let Some(ref state) = self.line_draw {
                let p1 = self.to_screen(state.start);
                painter.line_segment([p1, mouse_screen],
                    egui::Stroke::new(1.5, self.colors.preview_line));
                painter.circle_filled(p1, 4.0, self.colors.endpoint);
            }

            // Circle preview
            if let Some(ref state) = self.circle_draw {
                let center = self.to_screen(state.center);
                let radius_px = ((mouse_sketch.x - state.center.x).powi(2)
                    + (mouse_sketch.y - state.center.y).powi(2)).sqrt() as f32 * self.scale;
                painter.circle_stroke(center, radius_px,
                    egui::Stroke::new(1.5, self.colors.preview_line));
                painter.circle_filled(center, 4.0, self.colors.endpoint);
            }

            // Arc preview
            if let Some(ref state) = self.arc_draw {
                let start_screen = self.to_screen(state.start);
                painter.circle_filled(start_screen, 4.0, self.colors.endpoint);
                if let Some((end, _)) = state.end {
                    let end_screen = self.to_screen(end);
                    painter.circle_filled(end_screen, 4.0, self.colors.endpoint);
                    // Preview arc through start, end, and mouse
                    if let Some((c, r, sa, ea, _)) = circumscribed_arc(state.start, end, mouse_sketch) {
                        let norm = |v: f64| -> f64 { let rv = v % std::f64::consts::TAU; if rv < 0.0 { rv + std::f64::consts::TAU } else { rv } };
                        let span = norm(ea - sa);
                        let n_segs = 64usize;
                        let points: Vec<egui::Pos2> = (0..=n_segs).map(|i| {
                            let t = sa + span * (i as f64 / n_segs as f64);
                            self.to_screen(vect2d::new(c.x + r * t.cos(), c.y + r * t.sin()))
                        }).collect();
                        for w in points.windows(2) {
                            painter.line_segment([w[0], w[1]],
                                egui::Stroke::new(1.5, self.colors.preview_line));
                        }
                    }
                } else {
                    // Only start placed; draw line to mouse as hint
                    painter.line_segment([start_screen, mouse_screen],
                        egui::Stroke::new(1.0, self.colors.preview_line));
                }
            }

            // Draw cursor crosshair when drawing
            if self.tool != Tool::Select {
                painter.line_segment(
                    [egui::Pos2::new(mouse_screen.x, rect.top()),
                     egui::Pos2::new(mouse_screen.x, rect.bottom())],
                    egui::Stroke::new(0.5, self.colors.cursor_crosshair));
                painter.line_segment(
                    [egui::Pos2::new(rect.left(), mouse_screen.y),
                     egui::Pos2::new(rect.right(), mouse_screen.y)],
                    egui::Stroke::new(0.5, self.colors.cursor_crosshair));
            }

            // Status bar at bottom
            let status = match self.tool {
                Tool::Select => "Select: click to select/deselect, drag to move. Ctrl+Z undo, Ctrl+Shift+Z redo.",
                Tool::DrawPoint => "Point: click to place.",
                Tool::DrawLine => if self.line_draw.is_some() {
                    "Line: click to place end point (chains next line). Escape to finish."
                } else {
                    "Line: click to place start point. Snaps to nearby points/endpoints."
                },
                Tool::DrawCircle => if self.circle_draw.is_some() {
                    "Circle: click to set radius."
                } else {
                    "Circle: click to place center."
                },
                Tool::DrawArc => if let Some(ref s) = self.arc_draw {
                    if s.end.is_some() {
                        "Arc: click a point on the arc."
                    } else {
                        "Arc: click to place end point."
                    }
                } else {
                    "Arc: click to place start point."
                },
                Tool::ConstraintMode(_) => "Constraint: click entities to apply. Escape to cancel.",
                Tool::Dimension => if self.dim_editing {
                    "Dimension: type value and press Enter. Escape to cancel."
                } else {
                    "Dimension: click a line/arc, or two points. Escape to cancel."
                },
            };
            painter.text(
                egui::Pos2::new(rect.left() + 10.0, rect.bottom() - 20.0),
                egui::Align2::LEFT_CENTER,
                status,
                egui::FontId::proportional(12.0),
                self.colors.status_text,
            );
        });
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1024.0, 768.0])
            .with_title("Arael Sketch Editor")
            .with_app_id("arael-sketch-editor"),
        ..Default::default()
    };
    eframe::run_native(
        "Arael Sketch Editor",
        options,
        Box::new(|_cc| Ok(Box::new(EditorApp::default()))),
    )
}

#[cfg(target_arch = "wasm32")]
fn main() {
    use eframe::wasm_bindgen::JsCast;
    let web_options = eframe::WebOptions::default();
    wasm_bindgen_futures::spawn_local(async {
        let document = web_sys::window().unwrap().document().unwrap();
        let canvas = document.get_element_by_id("arael_canvas").unwrap();
        let canvas: web_sys::HtmlCanvasElement = canvas
            .dyn_into::<web_sys::HtmlCanvasElement>()
            .unwrap();
        eframe::WebRunner::new()
            .start(
                canvas,
                web_options,
                Box::new(|_cc| Ok(Box::new(EditorApp::default()))),
            )
            .await
            .expect("failed to start eframe");
    });
}
