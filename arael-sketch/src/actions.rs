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
    ApplyCollinear { a: Ref<Line>, b: Ref<Line> },
    ApplySymmetryLL { a: Ref<Line>, b: Ref<Line>, c: Ref<Line> },
    ApplyMidpoint { point: Ref<Point>, line: Ref<Line> },
    ApplyMidpointLP1 { line: Ref<Line>, target: Ref<Line> },
    ApplyMidpointLP2 { line: Ref<Line>, target: Ref<Line> },
    ApplyMidpointArcStart { arc: Ref<Arc>, line: Ref<Line> },
    ApplyMidpointArcEnd { arc: Ref<Arc>, line: Ref<Line> },
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

impl Action {
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
            Action::ApplyCollinear { .. } | Action::ApplySymmetryLL { .. } |
            Action::ApplyMidpoint { .. } | Action::ApplyMidpointLP1 { .. } |
            Action::ApplyMidpointLP2 { .. } | Action::ApplyMidpointArcStart { .. } |
            Action::ApplyMidpointArcEnd { .. } |
            Action::ApplyPointOnLine { .. } | Action::ApplyPointOnArc { .. } |
            Action::ApplyLineP1OnLine { .. } | Action::ApplyLineP2OnLine { .. } |
            Action::AddDimension { .. }
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

impl Action {
    pub fn apply(&self, sketch: &mut Sketch) {
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
            Action::ApplyCollinear { a, b } => {
                sketch.collinear.push(Collinear { a: *a, b: *b, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplySymmetryLL { a, b, c } => {
                sketch.symmetry_ll.push(SymmetryLL { a: *a, b: *b, c: *c, hb: TripletBlock::new() });
                sketch.solve();
            }
            Action::ApplyMidpoint { point, line } => {
                sketch.midpoint.push(MidpointConstraint { point: *point, line: *line, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyMidpointLP1 { line, target } => {
                sketch.midpoint_lp1.push(MidpointLP1 { line: *line, target: *target, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyMidpointLP2 { line, target } => {
                sketch.midpoint_lp2.push(MidpointLP2 { line: *line, target: *target, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyMidpointArcStart { arc, line } => {
                sketch.midpoint_arc_start.push(MidpointArcStart { arc: *arc, line: *line, hb: CrossBlock::new() });
                sketch.solve();
            }
            Action::ApplyMidpointArcEnd { arc, line } => {
                sketch.midpoint_arc_end.push(MidpointArcEnd { arc: *arc, line: *line, hb: CrossBlock::new() });
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
                let name = format!("d{}", sketch.next_dimension_id);
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
                }
                sketch.dimensions.push(Dimension {
                    kind: *kind, value: *value,
                    offset: vect2d::new(0.0, 1.0),
                    text_along: 0.0,
                    name,
                    expr_str: None,
                });
                sketch.solve();
            }
            Action::RemoveDimension { index } => {
                if *index < sketch.dimensions.len() {
                    let dim = sketch.dimensions.remove(*index);
                    // Expression dimension: remove the ExpressionConstraint
                    if dim.expr_str.is_some() {
                        let desc_prefix = format!("{} = ", dim.name);
                        sketch.expr_constraints.retain(|ec| !ec.description.starts_with(&desc_prefix));
                        sketch.solve();
                        return;  // skip normal constraint removal
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
                        DimensionKind::Angle(a, b, _) => {
                            sketch.angle.retain(|c| !(c.a == a && c.b == b));
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
