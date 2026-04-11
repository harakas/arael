// Action enum and apply() for undo/redo in the sketch editor.

use arael::model::{Param, CrossBlock, TripletBlock};
use arael::refs::Ref;
use arael::utils::deg2rad;
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::geometry::circumscribed_arc;

// Action log for undo/redo
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub enum Action {
    AddPoint { pos: vect2d },
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
    ApplySymmetryPP { a: Ref<Point>, line: Ref<Line>, c: Ref<Point> },
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
    DeleteArc { arc: Ref<Arc> },
    AddDimension { kind: DimensionKind, value: f64, expr: Option<String>, derived: bool },
    UpdateDimension { index: usize, value: f64, expr: Option<String> },
    RemoveDimension { index: usize },
    AddUserParam { name: String, expr_str: String },
    UpdateUserParam { index: usize, name: String, expr_str: String },
    RemoveUserParam { index: usize },
    DeleteConstraint { id: crate::tools::ConstraintId },
    // Drag is non-deterministic; store full state after drag completes
    Drag { snapshot: Vec<u8> },
}

impl Action {
    /// Human-readable description of this action.
    pub fn describe(&self) -> String {
        match self {
            Action::AddPoint { .. } => "Add point".into(),
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
            Action::AddDimension { kind, expr, .. } => {
                let kind_str = match kind {
                    DimensionKind::LineLength(_) => "length",
                    DimensionKind::ArcRadius(_) => "radius",
                    DimensionKind::ArcRadiusB(_) => "radius_b",
                    DimensionKind::ArcSweep(_) => "sweep",
                    DimensionKind::PointPointDistance(_, _) => "distance",
                    DimensionKind::PointLineDistance(_, _) => "distance",
                    DimensionKind::Angle(_, _, _) => "angle",
                    DimensionKind::HDistance(_, _) => "hdistance",
                    DimensionKind::VDistance(_, _) => "vdistance",
                    DimensionKind::LineAngle(_) => "xangle",
                };
                if expr.is_some() { format!("Add {} (expr)", kind_str) }
                else { format!("Add {}", kind_str) }
            }
            Action::UpdateDimension { .. } => "Update dimension".into(),
            Action::RemoveDimension { .. } => "Remove dimension".into(),
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

// Resolve a DimensionEndpoint to a Ref<Point>, creating helper point + constraint if needed.
pub fn resolve_dim_endpoint(sketch: &mut Sketch, ep: &DimensionEndpoint) -> Ref<Point> {
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
        (Point(pa), Point(pb)) => { sketch.distance_pp.push(DistancePP { a: *pa, b: *pb, distance, hb: CrossBlock::new() }); }
        // Line-Line
        (LineP1(la), LineP1(lb)) => { sketch.distance_ll11.push(DistanceLL11 { a: *la, b: *lb, distance, hb: CrossBlock::new() }); }
        (LineP1(la), LineP2(lb)) => { sketch.distance_ll12.push(DistanceLL12 { a: *la, b: *lb, distance, hb: CrossBlock::new() }); }
        (LineP2(la), LineP1(lb)) => { sketch.distance_ll21.push(DistanceLL21 { a: *la, b: *lb, distance, hb: CrossBlock::new() }); }
        (LineP2(la), LineP2(lb)) => { sketch.distance_ll22.push(DistanceLL22 { a: *la, b: *lb, distance, hb: CrossBlock::new() }); }
        // Line-Point
        (LineP1(l), Point(p)) | (Point(p), LineP1(l)) => { sketch.distance_lp1.push(DistanceLP1 { line: *l, point: *p, distance, hb: CrossBlock::new() }); }
        (LineP2(l), Point(p)) | (Point(p), LineP2(l)) => { sketch.distance_lp2.push(DistanceLP2 { line: *l, point: *p, distance, hb: CrossBlock::new() }); }
        // Arc-Point
        (ArcCenter(ar), Point(p)) | (Point(p), ArcCenter(ar)) => { sketch.distance_arc_center_p.push(DistanceArcCenterP { arc: *ar, point: *p, distance, hb: CrossBlock::new() }); }
        (ArcStart(ar), Point(p)) | (Point(p), ArcStart(ar)) => { sketch.distance_arc_start_p.push(DistanceArcStartP { arc: *ar, point: *p, distance, hb: CrossBlock::new() }); }
        (ArcEnd(ar), Point(p)) | (Point(p), ArcEnd(ar)) => { sketch.distance_arc_end_p.push(DistanceArcEndP { arc: *ar, point: *p, distance, hb: CrossBlock::new() }); }
        // Arc-Line
        (ArcCenter(ar), LineP1(l)) | (LineP1(l), ArcCenter(ar)) => { sketch.distance_arc_center_l1.push(DistanceArcCenterL1 { arc: *ar, line: *l, distance, hb: CrossBlock::new() }); }
        (ArcCenter(ar), LineP2(l)) | (LineP2(l), ArcCenter(ar)) => { sketch.distance_arc_center_l2.push(DistanceArcCenterL2 { arc: *ar, line: *l, distance, hb: CrossBlock::new() }); }
        (ArcStart(ar), LineP1(l)) | (LineP1(l), ArcStart(ar)) => { sketch.distance_arc_start_l1.push(DistanceArcStartL1 { arc: *ar, line: *l, distance, hb: CrossBlock::new() }); }
        (ArcStart(ar), LineP2(l)) | (LineP2(l), ArcStart(ar)) => { sketch.distance_arc_start_l2.push(DistanceArcStartL2 { arc: *ar, line: *l, distance, hb: CrossBlock::new() }); }
        (ArcEnd(ar), LineP1(l)) | (LineP1(l), ArcEnd(ar)) => { sketch.distance_arc_end_l1.push(DistanceArcEndL1 { arc: *ar, line: *l, distance, hb: CrossBlock::new() }); }
        (ArcEnd(ar), LineP2(l)) | (LineP2(l), ArcEnd(ar)) => { sketch.distance_arc_end_l2.push(DistanceArcEndL2 { arc: *ar, line: *l, distance, hb: CrossBlock::new() }); }
        // Arc-Arc
        (ArcCenter(a), ArcCenter(b)) => { sketch.distance_aa_ce_ce.push(DistanceAACeCe { a: *a, b: *b, distance, hb: CrossBlock::new() }); }
        (ArcCenter(a), ArcStart(b)) => { sketch.distance_aa_ce_s.push(DistanceAACeS { a: *a, b: *b, distance, hb: CrossBlock::new() }); }
        (ArcCenter(a), ArcEnd(b)) => { sketch.distance_aa_ce_e.push(DistanceAACeE { a: *a, b: *b, distance, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcCenter(b)) => { sketch.distance_aa_s_ce.push(DistanceAASCe { a: *a, b: *b, distance, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcStart(b)) => { sketch.distance_aa_s_s.push(DistanceAASS { a: *a, b: *b, distance, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcEnd(b)) => { sketch.distance_aa_s_e.push(DistanceAASE { a: *a, b: *b, distance, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcCenter(b)) => { sketch.distance_aa_e_ce.push(DistanceAAECe { a: *a, b: *b, distance, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcStart(b)) => { sketch.distance_aa_e_s.push(DistanceAAES { a: *a, b: *b, distance, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcEnd(b)) => { sketch.distance_aa_e_e.push(DistanceAAEE { a: *a, b: *b, distance, hb: CrossBlock::new() }); }
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
        Point(p) => { sketch.distance_pl.push(DistancePL { point: *p, line, distance, hb: CrossBlock::new() }); }
        LineP1(l) => { sketch.distance_lp1l.push(DistanceLP1L { a: *l, b: line, distance, hb: CrossBlock::new() }); }
        LineP2(l) => { sketch.distance_lp2l.push(DistanceLP2L { a: *l, b: line, distance, hb: CrossBlock::new() }); }
        ArcCenter(ar) => { sketch.distance_arc_center_l.push(DistanceArcCenterL { arc: *ar, line, distance, hb: CrossBlock::new() }); }
        ArcStart(ar) => { sketch.distance_arc_start_l.push(DistanceArcStartL { arc: *ar, line, distance, hb: CrossBlock::new() }); }
        ArcEnd(ar) => { sketch.distance_arc_end_l.push(DistanceArcEndL { arc: *ar, line, distance, hb: CrossBlock::new() }); }
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

/// Push the correct AxisDistance constraint for the given endpoint pair.
fn push_axis_distance(sketch: &mut Sketch, a: &DimensionEndpoint, b: &DimensionEndpoint, distance: f64, horizontal: bool) {
    use DimensionEndpoint::*;
    match (a, b) {
        (Point(pa), Point(pb)) => {
            if horizontal {
                sketch.hdistance_pp.push(HorizontalDistancePP { a: *pa, b: *pb, distance, hb: CrossBlock::new() });
            } else {
                sketch.vdistance_pp.push(VerticalDistancePP { a: *pa, b: *pb, distance, hb: CrossBlock::new() });
            }
        }
        // Line-Line
        (LineP1(la), LineP1(lb)) => { sketch.axis_distance_ll11.push(AxisDistanceLL11 { a: *la, b: *lb, distance, horizontal, hb: CrossBlock::new() }); }
        (LineP1(la), LineP2(lb)) => { sketch.axis_distance_ll12.push(AxisDistanceLL12 { a: *la, b: *lb, distance, horizontal, hb: CrossBlock::new() }); }
        (LineP2(la), LineP1(lb)) => { sketch.axis_distance_ll21.push(AxisDistanceLL21 { a: *la, b: *lb, distance, horizontal, hb: CrossBlock::new() }); }
        (LineP2(la), LineP2(lb)) => { sketch.axis_distance_ll22.push(AxisDistanceLL22 { a: *la, b: *lb, distance, horizontal, hb: CrossBlock::new() }); }
        // Line-Point (constraint has line first, so negate distance if point is first arg)
        (LineP1(l), Point(p)) => { sketch.axis_distance_lp1.push(AxisDistanceLP1 { line: *l, point: *p, distance, horizontal, hb: CrossBlock::new() }); }
        (Point(p), LineP1(l)) => { sketch.axis_distance_lp1.push(AxisDistanceLP1 { line: *l, point: *p, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        (LineP2(l), Point(p)) => { sketch.axis_distance_lp2.push(AxisDistanceLP2 { line: *l, point: *p, distance, horizontal, hb: CrossBlock::new() }); }
        (Point(p), LineP2(l)) => { sketch.axis_distance_lp2.push(AxisDistanceLP2 { line: *l, point: *p, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        // Arc-Point (constraint has arc first)
        (ArcCenter(ar), Point(p)) => { sketch.axis_distance_arc_center_p.push(AxisDistanceArcCenterP { arc: *ar, point: *p, distance, horizontal, hb: CrossBlock::new() }); }
        (Point(p), ArcCenter(ar)) => { sketch.axis_distance_arc_center_p.push(AxisDistanceArcCenterP { arc: *ar, point: *p, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        (ArcStart(ar), Point(p)) => { sketch.axis_distance_arc_start_p.push(AxisDistanceArcStartP { arc: *ar, point: *p, distance, horizontal, hb: CrossBlock::new() }); }
        (Point(p), ArcStart(ar)) => { sketch.axis_distance_arc_start_p.push(AxisDistanceArcStartP { arc: *ar, point: *p, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        (ArcEnd(ar), Point(p)) => { sketch.axis_distance_arc_end_p.push(AxisDistanceArcEndP { arc: *ar, point: *p, distance, horizontal, hb: CrossBlock::new() }); }
        (Point(p), ArcEnd(ar)) => { sketch.axis_distance_arc_end_p.push(AxisDistanceArcEndP { arc: *ar, point: *p, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        // Arc-Line (constraint has arc first)
        (ArcCenter(ar), LineP1(l)) => { sketch.axis_distance_arc_center_l1.push(AxisDistanceArcCenterL1 { arc: *ar, line: *l, distance, horizontal, hb: CrossBlock::new() }); }
        (LineP1(l), ArcCenter(ar)) => { sketch.axis_distance_arc_center_l1.push(AxisDistanceArcCenterL1 { arc: *ar, line: *l, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        (ArcCenter(ar), LineP2(l)) => { sketch.axis_distance_arc_center_l2.push(AxisDistanceArcCenterL2 { arc: *ar, line: *l, distance, horizontal, hb: CrossBlock::new() }); }
        (LineP2(l), ArcCenter(ar)) => { sketch.axis_distance_arc_center_l2.push(AxisDistanceArcCenterL2 { arc: *ar, line: *l, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        (ArcStart(ar), LineP1(l)) => { sketch.axis_distance_arc_start_l1.push(AxisDistanceArcStartL1 { arc: *ar, line: *l, distance, horizontal, hb: CrossBlock::new() }); }
        (LineP1(l), ArcStart(ar)) => { sketch.axis_distance_arc_start_l1.push(AxisDistanceArcStartL1 { arc: *ar, line: *l, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        (ArcStart(ar), LineP2(l)) => { sketch.axis_distance_arc_start_l2.push(AxisDistanceArcStartL2 { arc: *ar, line: *l, distance, horizontal, hb: CrossBlock::new() }); }
        (LineP2(l), ArcStart(ar)) => { sketch.axis_distance_arc_start_l2.push(AxisDistanceArcStartL2 { arc: *ar, line: *l, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        (ArcEnd(ar), LineP1(l)) => { sketch.axis_distance_arc_end_l1.push(AxisDistanceArcEndL1 { arc: *ar, line: *l, distance, horizontal, hb: CrossBlock::new() }); }
        (LineP1(l), ArcEnd(ar)) => { sketch.axis_distance_arc_end_l1.push(AxisDistanceArcEndL1 { arc: *ar, line: *l, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        (ArcEnd(ar), LineP2(l)) => { sketch.axis_distance_arc_end_l2.push(AxisDistanceArcEndL2 { arc: *ar, line: *l, distance, horizontal, hb: CrossBlock::new() }); }
        (LineP2(l), ArcEnd(ar)) => { sketch.axis_distance_arc_end_l2.push(AxisDistanceArcEndL2 { arc: *ar, line: *l, distance: -distance, horizontal, hb: CrossBlock::new() }); }
        // Arc-Arc
        (ArcCenter(a), ArcCenter(b)) => { sketch.axis_distance_aa_ce_ce.push(AxisDistanceAACeCe { a: *a, b: *b, distance, horizontal, hb: CrossBlock::new() }); }
        (ArcCenter(a), ArcStart(b)) => { sketch.axis_distance_aa_ce_s.push(AxisDistanceAACeS { a: *a, b: *b, distance, horizontal, hb: CrossBlock::new() }); }
        (ArcCenter(a), ArcEnd(b)) => { sketch.axis_distance_aa_ce_e.push(AxisDistanceAACeE { a: *a, b: *b, distance, horizontal, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcCenter(b)) => { sketch.axis_distance_aa_s_ce.push(AxisDistanceAASCe { a: *a, b: *b, distance, horizontal, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcStart(b)) => { sketch.axis_distance_aa_s_s.push(AxisDistanceAASS { a: *a, b: *b, distance, horizontal, hb: CrossBlock::new() }); }
        (ArcStart(a), ArcEnd(b)) => { sketch.axis_distance_aa_s_e.push(AxisDistanceAASE { a: *a, b: *b, distance, horizontal, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcCenter(b)) => { sketch.axis_distance_aa_e_ce.push(AxisDistanceAAECe { a: *a, b: *b, distance, horizontal, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcStart(b)) => { sketch.axis_distance_aa_e_s.push(AxisDistanceAAES { a: *a, b: *b, distance, horizontal, hb: CrossBlock::new() }); }
        (ArcEnd(a), ArcEnd(b)) => { sketch.axis_distance_aa_e_e.push(AxisDistanceAAEE { a: *a, b: *b, distance, horizontal, hb: CrossBlock::new() }); }
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

impl Action {
    /// Apply the action, then solve.
    pub fn apply(&self, sketch: &mut Sketch) {
        let needs_expr_update = self.apply_without_solve(sketch);
        sketch.solve();
        if needs_expr_update {
            sketch.update_expr_dim_values();
        }
    }

    /// Apply the action's mutation without solving.
    /// Returns true if `update_expr_dim_values` should be called after solving.
    pub fn apply_without_solve(&self, sketch: &mut Sketch) -> bool {
        sketch.cached_dof = None;
        match self {
            Action::AddPoint { pos } => { sketch.add_point(*pos); }
            Action::AddLine { p1, p2 } => { sketch.add_line(*p1, *p2); }
            Action::AddCircle { center, edge } => {
                let r = ((edge.x - center.x).powi(2) + (edge.y - center.y).powi(2)).sqrt();
                sketch.add_arc(*center, r, 0.0, std::f64::consts::TAU, true);
            }
            Action::AddEllipse { center, rx, ry, rotation } => {
                sketch.add_ellipse(*center, *rx, *ry, *rotation, true);
            }
            Action::AddArc { start, end, mid, .. } => {
                if let Some((c, r, sa, ea, ccw)) = circumscribed_arc(*start, *end, *mid) {
                    sketch.add_arc_with_dir(c, r, sa, ea, false, ccw);
                }
            }
            Action::AddEllipticArc { center, rx, ry, rotation, start, end, ccw } => {
                sketch.add_elliptic_arc(*center, *rx, *ry, *rotation, *start, *end, *ccw);
            }
            Action::ApplyHorizontal { lines } => {
                for r in lines { sketch.lines[*r].constraints.horizontal = true; }
            }
            Action::ApplyVertical { lines } => {
                for r in lines { sketch.lines[*r].constraints.vertical = true; }
            }
            Action::ApplyCoincidentPP { a, b } => {
                sketch.coincident_pp.push(CoincidentPP { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLL11 { a, b } => {
                sketch.coincident_ll11.push(CoincidentLL11 { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLL12 { a, b } => {
                sketch.coincident_ll12.push(CoincidentLL12 { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLL21 { a, b } => {
                sketch.coincident_ll21.push(CoincidentLL21 { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLL22 { a, b } => {
                sketch.coincident_ll22.push(CoincidentLL22 { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP1 { line, point } => {
                sketch.coincident_lp1.push(CoincidentLP1 { line: *line, point: *point, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP2 { line, point } => {
                sketch.coincident_lp2.push(CoincidentLP2 { line: *line, point: *point, hb: CrossBlock::new() });
            }
            Action::ApplyParallel { a, b } => {
                sketch.parallel.push(Parallel { a: *a, b: *b, hb: CrossBlock::new() });
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
                sketch.perpendicular.push(Perpendicular { a: *a, b: *b, dir_sign, hb: CrossBlock::new() });
            }
            Action::ApplyEqualLength { a, b } => {
                sketch.equal_length.push(EqualLength { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcCenter { point, arc } => {
                sketch.coincident_arc_center.push(CoincidentArcCenter { point: *point, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcStart { point, arc } => {
                sketch.coincident_arc_start.push(CoincidentArcStart { point: *point, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcEnd { point, arc } => {
                sketch.coincident_arc_end.push(CoincidentArcEnd { point: *point, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyConcentric { a, b } => {
                sketch.concentric.push(Concentric { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP1ArcCenter { line, arc } => {
                sketch.coincident_lp1_arc_center.push(CoincidentLP1ArcCenter { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP2ArcCenter { line, arc } => {
                sketch.coincident_lp2_arc_center.push(CoincidentLP2ArcCenter { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP1ArcStart { line, arc } => {
                sketch.coincident_lp1_arc_start.push(CoincidentLP1ArcStart { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP2ArcStart { line, arc } => {
                sketch.coincident_lp2_arc_start.push(CoincidentLP2ArcStart { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP1ArcEnd { line, arc } => {
                sketch.coincident_lp1_arc_end.push(CoincidentLP1ArcEnd { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentLP2ArcEnd { line, arc } => {
                sketch.coincident_lp2_arc_end.push(CoincidentLP2ArcEnd { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcCenterStart { a, b } => {
                sketch.coincident_arc_center_start.push(CoincidentArcCenterStart { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcCenterEnd { a, b } => {
                sketch.coincident_arc_center_end.push(CoincidentArcCenterEnd { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcStartCenter { a, b } => {
                sketch.coincident_arc_start_center.push(CoincidentArcStartCenter { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcEndCenter { a, b } => {
                sketch.coincident_arc_end_center.push(CoincidentArcEndCenter { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcStartStart { a, b } => {
                sketch.coincident_arc_start_start.push(CoincidentArcStartStart { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcStartEnd { a, b } => {
                sketch.coincident_arc_start_end.push(CoincidentArcStartEnd { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcEndStart { a, b } => {
                sketch.coincident_arc_end_start.push(CoincidentArcEndStart { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyCoincidentArcEndEnd { a, b } => {
                sketch.coincident_arc_end_end.push(CoincidentArcEndEnd { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyLineP1OnArc { line, arc } => {
                sketch.line_p1_on_arc.push(LineP1OnArc { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyLineP2OnArc { line, arc } => {
                sketch.line_p2_on_arc.push(LineP2OnArc { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyEqualRadius { a, b } => {
                sketch.equal_radius.push(EqualRadius { a: *a, b: *b, hb: CrossBlock::new() });
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
                    dir_sign, hb: CrossBlock::new(),
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
                sketch.tangent_aa.push(TangentAA { a: *a, b: *b, shared, hb: CrossBlock::new() });
            }
            Action::ApplyPointOnLine { point, line } => {
                sketch.point_on_line.push(PointOnLine { point: *point, line: *line, hb: CrossBlock::new() });
            }
            Action::ApplyPointOnArc { point, arc } => {
                sketch.point_on_arc.push(PointOnArc { point: *point, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyEndpointOnLine { endpoint, line } => {
                let point = resolve_dim_endpoint(sketch, endpoint);
                sketch.point_on_line.push(PointOnLine { point, line: *line, hb: CrossBlock::new() });
            }
            Action::ApplyEndpointOnArc { endpoint, arc } => {
                let point = resolve_dim_endpoint(sketch, endpoint);
                sketch.point_on_arc.push(PointOnArc { point, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyCollinear { a, b } => {
                sketch.collinear.push(Collinear { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplySymmetryLL { a, b, c } => {
                sketch.symmetry_ll.push(SymmetryLL { a: *a, b: *b, c: *c, hb: TripletBlock::new() });
            }
            Action::ApplySymmetryPP { a, line, c } => {
                sketch.symmetry_pp.push(SymmetryPP { a: *a, c: *c, line: *line, hb: TripletBlock::new() });
            }
            Action::ApplySymmetryAA { a, line, c } => {
                sketch.symmetry_aa.push(SymmetryAA { a: *a, c: *c, line: *line, hb: TripletBlock::new() });
            }
            Action::ApplyMidpoint { point, line } => {
                sketch.midpoint.push(MidpointConstraint { point: *point, line: *line, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointLP1 { line, target } => {
                sketch.midpoint_lp1.push(MidpointLP1 { line: *line, target: *target, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointLP2 { line, target } => {
                sketch.midpoint_lp2.push(MidpointLP2 { line: *line, target: *target, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcStart { arc, line } => {
                sketch.midpoint_arc_start.push(MidpointArcStart { arc: *arc, line: *line, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcEnd { arc, line } => {
                sketch.midpoint_arc_end.push(MidpointArcEnd { arc: *arc, line: *line, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcPoint { point, arc } => {
                sketch.midpoint_arc_point.push(MidpointArcPoint { point: *point, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointLP1Arc { line, arc } => {
                sketch.midpoint_lp1_arc.push(MidpointLP1Arc { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointLP2Arc { line, arc } => {
                sketch.midpoint_lp2_arc.push(MidpointLP2Arc { line: *line, arc: *arc, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcStartArc { a, b } => {
                sketch.midpoint_arc_start_arc.push(MidpointArcStartArc { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyMidpointArcEndArc { a, b } => {
                sketch.midpoint_arc_end_arc.push(MidpointArcEndArc { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyLineP1OnLine { a, b } => {
                sketch.line_p1_on_line.push(LineP1OnLine { a: *a, b: *b, hb: CrossBlock::new() });
            }
            Action::ApplyLineP2OnLine { a, b } => {
                sketch.line_p2_on_line.push(LineP2OnLine { a: *a, b: *b, hb: CrossBlock::new() });
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
            Action::DeleteArc { arc } => {
                sketch.delete_arc(*arc);
            }
            Action::AddDimension { kind, value, expr, derived } => {
                // Expression dimension: delegate to add_expr_dimension
                if let Some(expr_str) = expr {
                    let _ = sketch.add_expr_dimension(*kind, expr_str,
                        vect2d::new(0.0, 1.0), 0.0);
                    if *derived {
                        if let Some(d) = sketch.dimensions.last_mut() { d.derived = true; }
                    }
                    return false;
                }
                let name = format!("d{}", sketch.next_dimension_id);
                sketch.next_dimension_id += 1;
                // Apply the numeric constraint (skip for derived dims)
                if !derived { match kind {
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
                            a: *a, b: *b, angle: target, hb: CrossBlock::new(),
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
                }
                } // end if !derived
                sketch.dimensions.push(Dimension {
                    kind: *kind, value: *value,
                    offset: vect2d::new(0.0, 1.0),
                    text_along: 0.0,
                    name,
                    expr_str: None,
                    broken: false, derived: *derived,
                });
            }
            Action::UpdateDimension { index, value, expr } => {
                if *index >= sketch.dimensions.len() { return false; }
                // Remove old underlying constraint (only for numeric, non-derived dims)
                {
                    let dim = &sketch.dimensions[*index];
                    let dim_kind = dim.kind;
                    let dim_value = dim.value;
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
                        }
                    }
                }
                // Update dimension in place (keeps name, kind, offset, text_along)
                let dim = &mut sketch.dimensions[*index];
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
                                a, b, angle: target, hb: CrossBlock::new(),
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
                    }
                }
                // Expression dims: rebuild_expr_constraints() in solve() handles it
            }
            Action::RemoveDimension { index } => {
                if *index < sketch.dimensions.len() {
                    let dim = sketch.dimensions.remove(*index);
                    // Expression dimension: remove the ExpressionConstraint
                    if dim.expr_str.is_some() {
                        let desc_prefix = format!("{} = ", dim.name);
                        sketch.expr_constraints.retain(|ec| !ec.description.starts_with(&desc_prefix));
                        return false;  // skip normal constraint removal
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
                return true;
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
                            if let Ok(parsed) = arael_sym::parse(&p.expr_str) {
                                if parsed.symbols().contains(&old_name) {
                                    let replaced = parsed.subs(&old_name, &arael_sym::symbol(name));
                                    p.expr_str = format!("{}", replaced);
                                }
                            }
                        }
                        for d in &mut sketch.dimensions {
                            if let Some(ref es) = d.expr_str {
                                if let Ok(parsed) = arael_sym::parse(es) {
                                    if parsed.symbols().contains(&old_name) {
                                        let replaced = parsed.subs(&old_name, &arael_sym::symbol(name));
                                        d.expr_str = Some(format!("{}", replaced));
                                    }
                                }
                            }
                        }
                    }
                    return true;
                }
            }
            Action::RemoveUserParam { index } => {
                if *index < sketch.user_params.len() {
                    sketch.user_params.remove(*index);
                    return true;
                }
            }
            Action::DeleteConstraint { id } => {
                use crate::tools::{ConstraintId, CoincidentKind, MidpointKind};
                match id {
                    ConstraintId::Horizontal(r) => { sketch.lines[*r].constraints.horizontal = false; }
                    ConstraintId::Vertical(r) => { sketch.lines[*r].constraints.vertical = false; }
                    ConstraintId::Parallel(i) => { sketch.parallel.remove(*i); }
                    ConstraintId::Perpendicular(i) => { sketch.perpendicular.remove(*i); }
                    ConstraintId::EqualLength(i) => { sketch.equal_length.remove(*i); }
                    ConstraintId::EqualRadius(i) => { sketch.equal_radius.remove(*i); }
                    ConstraintId::Concentric(i) => { sketch.concentric.remove(*i); }
                    ConstraintId::TangentLA(i) => { sketch.tangent_la.remove(*i); }
                    ConstraintId::TangentAA(i) => { sketch.tangent_aa.remove(*i); }
                    ConstraintId::Collinear(i) => { sketch.collinear.remove(*i); }
                    ConstraintId::Symmetry(i) => { sketch.symmetry_ll.remove(*i); }
                    ConstraintId::SymmetryPP(i) => { sketch.symmetry_pp.remove(*i); }
                    ConstraintId::SymmetryAA(i) => { sketch.symmetry_aa.remove(*i); }
                    ConstraintId::Midpoint(kind, i) => {
                        match kind {
                            MidpointKind::Point => { sketch.midpoint.remove(*i); }
                            MidpointKind::LP1 => { sketch.midpoint_lp1.remove(*i); }
                            MidpointKind::LP2 => { sketch.midpoint_lp2.remove(*i); }
                            MidpointKind::ArcStart => { sketch.midpoint_arc_start.remove(*i); }
                            MidpointKind::ArcEnd => { sketch.midpoint_arc_end.remove(*i); }
                            MidpointKind::ArcPoint => { sketch.midpoint_arc_point.remove(*i); }
                            MidpointKind::LP1Arc => { sketch.midpoint_lp1_arc.remove(*i); }
                            MidpointKind::LP2Arc => { sketch.midpoint_lp2_arc.remove(*i); }
                            MidpointKind::ArcStartArc => { sketch.midpoint_arc_start_arc.remove(*i); }
                            MidpointKind::ArcEndArc => { sketch.midpoint_arc_end_arc.remove(*i); }
                        }
                    }
                    ConstraintId::Coincident(kind, i) => {
                        match kind {
                            CoincidentKind::PP => { sketch.coincident_pp.remove(*i); }
                            CoincidentKind::LP1 => { sketch.coincident_lp1.remove(*i); }
                            CoincidentKind::LP2 => { sketch.coincident_lp2.remove(*i); }
                            CoincidentKind::LL11 => { sketch.coincident_ll11.remove(*i); }
                            CoincidentKind::LL12 => { sketch.coincident_ll12.remove(*i); }
                            CoincidentKind::LL21 => { sketch.coincident_ll21.remove(*i); }
                            CoincidentKind::LL22 => { sketch.coincident_ll22.remove(*i); }
                            CoincidentKind::PointOnLine => { sketch.point_on_line.remove(*i); }
                            CoincidentKind::PointOnArc => { sketch.point_on_arc.remove(*i); }
                            CoincidentKind::LP1OnLine => { sketch.line_p1_on_line.remove(*i); }
                            CoincidentKind::LP2OnLine => { sketch.line_p2_on_line.remove(*i); }
                            CoincidentKind::LP1OnArc => { sketch.line_p1_on_arc.remove(*i); }
                            CoincidentKind::LP2OnArc => { sketch.line_p2_on_arc.remove(*i); }
                            CoincidentKind::ArcCenter => { sketch.coincident_arc_center.remove(*i); }
                            CoincidentKind::ArcStart => { sketch.coincident_arc_start.remove(*i); }
                            CoincidentKind::ArcEnd => { sketch.coincident_arc_end.remove(*i); }
                            CoincidentKind::LP1ArcCenter => { sketch.coincident_lp1_arc_center.remove(*i); }
                            CoincidentKind::LP2ArcCenter => { sketch.coincident_lp2_arc_center.remove(*i); }
                            CoincidentKind::LP1ArcStart => { sketch.coincident_lp1_arc_start.remove(*i); }
                            CoincidentKind::LP2ArcStart => { sketch.coincident_lp2_arc_start.remove(*i); }
                            CoincidentKind::LP1ArcEnd => { sketch.coincident_lp1_arc_end.remove(*i); }
                            CoincidentKind::LP2ArcEnd => { sketch.coincident_lp2_arc_end.remove(*i); }
                            CoincidentKind::ArcCenterStart => { sketch.coincident_arc_center_start.remove(*i); }
                            CoincidentKind::ArcCenterEnd => { sketch.coincident_arc_center_end.remove(*i); }
                            CoincidentKind::ArcStartCenter => { sketch.coincident_arc_start_center.remove(*i); }
                            CoincidentKind::ArcEndCenter => { sketch.coincident_arc_end_center.remove(*i); }
                            CoincidentKind::ArcStartStart => { sketch.coincident_arc_start_start.remove(*i); }
                            CoincidentKind::ArcStartEnd => { sketch.coincident_arc_start_end.remove(*i); }
                            CoincidentKind::ArcEndStart => { sketch.coincident_arc_end_start.remove(*i); }
                            CoincidentKind::ArcEndEnd => { sketch.coincident_arc_end_end.remove(*i); }
                        }
                        sketch.cleanup_helper_points();
                    }
                    ConstraintId::HelperBridge(pt) => {
                        sketch.delete_point(*pt);
                    }
                }
            }
            Action::Drag { snapshot } => {
                *sketch = bincode::deserialize(snapshot).unwrap();
            }
        }
        false
    }
}
