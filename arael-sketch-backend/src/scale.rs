//! Uniform scaling of sketch entities about a center point, plus the
//! dimension bookkeeping that makes it mean what a user expects: a
//! driving linear dimension whose referents are all inside the scaled
//! set scales with the geometry, everything else is left alone and
//! reported (see docs/dev/SCALETOOL.md).

use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::actions::{push_numeric_dim_constraint, remove_numeric_dim_constraint};

/// What the dimension pass did, for the caller's output.
#[derive(Default)]
pub struct ScaleDimReport {
    /// Dimension names whose values (and backing constraints) scaled.
    pub scaled: Vec<String>,
    /// `(name, reason)` for linear dims touching the set that were
    /// deliberately not scaled -- these may resist the scale on the
    /// next solve.
    pub left: Vec<(String, &'static str)>,
}

/// True when the endpoint's entity is in the scaled set.
fn ep_inside(
    ep: &DimensionEndpoint,
    lines: &[Ref<Line>],
    arcs: &[Ref<Arc>],
    points: &[Ref<Point>],
) -> bool {
    match *ep {
        DimensionEndpoint::Point(p) => points.contains(&p),
        DimensionEndpoint::LineP1(l) | DimensionEndpoint::LineP2(l) => lines.contains(&l),
        DimensionEndpoint::ArcCenter(a)
        | DimensionEndpoint::ArcStart(a)
        | DimensionEndpoint::ArcEnd(a) => arcs.contains(&a),
    }
}

/// Whether every referent of a linear dimension kind is inside the
/// scaled set. `None` for angular kinds (scale-invariant, never
/// touched or reported).
fn linear_kind_inside(
    kind: &DimensionKind,
    lines: &[Ref<Line>],
    arcs: &[Ref<Arc>],
    points: &[Ref<Point>],
) -> Option<bool> {
    match kind {
        DimensionKind::LineLength(l) => Some(lines.contains(l)),
        DimensionKind::ArcRadius(a) | DimensionKind::ArcRadiusB(a) => Some(arcs.contains(a)),
        DimensionKind::PointPointDistance(a, b)
        | DimensionKind::HDistance(a, b)
        | DimensionKind::VDistance(a, b) => {
            Some(ep_inside(a, lines, arcs, points) && ep_inside(b, lines, arcs, points))
        }
        DimensionKind::PointLineDistance(pt, l) => {
            Some(ep_inside(pt, lines, arcs, points) && lines.contains(l))
        }
        DimensionKind::ConcentricDistance(a, b) => Some(arcs.contains(a) && arcs.contains(b)),
        DimensionKind::LineLineDistance(a, b) => Some(lines.contains(a) && lines.contains(b)),
        DimensionKind::Angle(..)
        | DimensionKind::LineAngle(_)
        | DimensionKind::ArcSweep(_)
        | DimensionKind::ArcRotation(_) => None,
    }
}

/// Does the kind reference ANY entity of the set? (Reporting filter:
/// dims entirely elsewhere are irrelevant.)
fn kind_touches(
    kind: &DimensionKind,
    lines: &[Ref<Line>],
    arcs: &[Ref<Arc>],
    points: &[Ref<Point>],
) -> bool {
    lines.iter().any(|r| kind.references_line(*r))
        || arcs.iter().any(|r| kind.references_arc(*r))
        || points.iter().any(|r| kind.references_point(*r))
}

/// Classify the sketch's dimensions against a scaled set: which dids
/// scale with the geometry, and which are left (with reasons).
/// Pure -- the command layer calls it for the report, apply_scale for
/// the work, and both see the same answer.
pub fn classify_scale_dims(
    sketch: &Sketch,
    lines: &[Ref<Line>],
    arcs: &[Ref<Arc>],
    points: &[Ref<Point>],
) -> (Vec<u32>, ScaleDimReport) {
    let mut scale_dids = Vec::new();
    let mut report = ScaleDimReport::default();
    for dim in &sketch.dimensions {
        let Some(inside) = linear_kind_inside(&dim.kind, lines, arcs, points) else {
            continue; // angular: scale-invariant
        };
        if !kind_touches(&dim.kind, lines, arcs, points) {
            continue; // entirely elsewhere
        }
        if dim.derived {
            continue; // re-measures on its own
        }
        if dim.expr_str.is_some() {
            report.left.push((dim.name.clone(), "expression"));
            continue;
        }
        if let Some(range) = &dim.range {
            let live = match range {
                RangeBound::Min(v) | RangeBound::Max(v) => matches!(v, RangeValue::Live(_)),
                RangeBound::Between(lo, hi) => {
                    matches!(lo, RangeValue::Live(_)) || matches!(hi, RangeValue::Live(_))
                }
            };
            if live {
                report.left.push((dim.name.clone(), "live range bound"));
                continue;
            }
        }
        if !inside {
            report.left.push((dim.name.clone(), "spans unscaled geometry"));
            continue;
        }
        scale_dids.push(dim.did);
        report.scaled.push(dim.name.clone());
    }
    (scale_dids, report)
}

fn scale_literal(v: &mut RangeValue, factor: f64) {
    if let RangeValue::Literal(x) = v {
        *x *= factor;
    }
}

/// Apply the scale: transform the geometry values, scale the
/// fully-inside driving dims and their backing constraints. The
/// caller (Action::apply) solves afterwards.
pub fn apply_scale(
    sketch: &mut Sketch,
    lines: &[Ref<Line>],
    arcs: &[Ref<Arc>],
    points: &[Ref<Point>],
    center: vect2d,
    factor: f64,
) -> ScaleDimReport {
    let t = |p: vect2d| -> vect2d {
        vect2d::new(
            center.x + factor * (p.x - center.x),
            center.y + factor * (p.y - center.y),
        )
    };
    let (scale_dids, report) = classify_scale_dims(sketch, lines, arcs, points);

    // 1. Drop the backing constraints of the dims that scale; they
    // are re-pushed after the transform so their sign captures read
    // the scaled geometry.
    let mut driving: Vec<(u32, DimensionKind)> = Vec::new();
    for dim in &sketch.dimensions {
        if scale_dids.contains(&dim.did) && dim.range.is_none() {
            driving.push((dim.did, dim.kind));
        }
    }
    for (_, kind) in &driving {
        remove_numeric_dim_constraint(sketch, kind);
    }

    // 2. Geometry. Value assignment keeps each Param's locked state;
    // point fix targets scale so locks move with their points.
    for &r in lines {
        if let Some(l) = sketch.lines.get_mut(r) {
            l.p1.value = t(l.p1.value);
            l.p2.value = t(l.p2.value);
        }
    }
    for &r in points {
        if let Some(p) = sketch.points.get_mut(r) {
            p.pos.value = t(p.pos.value);
            if p.constraints.has_fix_x || p.constraints.has_fix_y {
                let fixed = t(vect2d::new(p.constraints.fix_x, p.constraints.fix_y));
                p.constraints.fix_x = fixed.x;
                p.constraints.fix_y = fixed.y;
            }
        }
    }
    for &r in arcs {
        if let Some(a) = sketch.arcs.get_mut(r) {
            a.center.value = t(a.center.value);
            a.radius.value *= factor;
            a.radius_b.value *= factor;
        }
    }

    // 3. Dimension values and literal range bounds.
    for dim in &mut sketch.dimensions {
        if !scale_dids.contains(&dim.did) {
            continue;
        }
        dim.value *= factor;
        if let Some(range) = &mut dim.range {
            match range {
                RangeBound::Min(v) | RangeBound::Max(v) => scale_literal(v, factor),
                RangeBound::Between(lo, hi) => {
                    scale_literal(lo, factor);
                    scale_literal(hi, factor);
                }
            }
        }
    }

    // 4. Re-push the driving constraints at the scaled values.
    for (did, kind) in &driving {
        if let Some(i) = sketch.dimension_index_by_did(*did) {
            let value = sketch.dimensions[i].value;
            push_numeric_dim_constraint(sketch, kind, &value);
        }
    }
    report
}

#[cfg(test)]
mod scale_tests {
    use super::*;

    fn v(x: f64, y: f64) -> vect2d { vect2d::new(x, y) }
    fn near(a: f64, b: f64) -> bool { (a - b).abs() < 1e-9 }
    fn near_v(a: vect2d, b: vect2d) -> bool { near(a.x, b.x) && near(a.y, b.y) }

    #[test]
    fn test_geometry_transform() {
        let mut s = Sketch::new();
        let l = s.add_line(v(1.0, 0.0), v(3.0, 0.0));
        let a = s.add_arc(v(2.0, 2.0), 1.0, 0.0, 1.0, false);
        let p = s.add_point(v(0.0, 1.0));
        let report = apply_scale(&mut s, &[l], &[a], &[p], v(1.0, 0.0), 2.0);
        assert!(near_v(s.lines[l].p1.value, v(1.0, 0.0)), "center endpoint stays");
        assert!(near_v(s.lines[l].p2.value, v(5.0, 0.0)));
        assert!(near_v(s.arcs[a].center.value, v(3.0, 4.0)));
        assert!(near(s.arcs[a].radius.value, 2.0));
        assert!(near(s.arcs[a].radius_b.value, 2.0));
        assert!(near(s.arcs[a].start_angle.value, 0.0), "angles unchanged");
        assert!(near_v(s.points[p].pos.value, v(-1.0, 2.0)));
        assert!(report.scaled.is_empty() && report.left.is_empty());
    }

    #[test]
    fn test_locked_point_target_scales() {
        let mut s = Sketch::new();
        let p = s.add_point(v(2.0, 0.0));
        s.points[p].constraints.has_fix_x = true;
        s.points[p].constraints.fix_x = 2.0;
        s.points[p].constraints.has_fix_y = true;
        s.points[p].constraints.fix_y = 0.0;
        apply_scale(&mut s, &[], &[], &[p], v(0.0, 0.0), 3.0);
        assert!(near(s.points[p].constraints.fix_x, 6.0));
        assert!(near(s.points[p].constraints.fix_y, 0.0));
    }

    #[test]
    fn test_inside_driving_dim_scales() {
        let mut s = Sketch::new();
        let l = s.add_line(v(0.0, 0.0), v(3.0, 0.0));
        s.lines[l].constraints.has_length = true;
        s.lines[l].constraints.length = 3.0;
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::LineLength(l), value: 3.0,
            offset: v(0.0, 1.0), text_along: 0.0, name: "d0".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.assign_constraint_names();
        let report = apply_scale(&mut s, &[l], &[], &[], v(0.0, 0.0), 2.0);
        assert_eq!(report.scaled, vec!["d0".to_string()]);
        assert!(near(s.dimensions[0].value, 6.0));
        assert!(near(s.lines[l].constraints.length, 6.0), "backing target scaled");
        // The scaled state satisfies the constraint: solve stays put.
        s.solve();
        let dx = s.lines[l].p2.value.x - s.lines[l].p1.value.x;
        assert!((dx - 6.0).abs() < 1e-6, "length after solve = {}", dx);
    }

    #[test]
    fn test_boundary_dim_left() {
        let mut s = Sketch::new();
        let l0 = s.add_line(v(0.0, 0.0), v(3.0, 0.0));
        let l1 = s.add_line(v(0.0, 2.0), v(3.0, 2.0));
        // Distance between the two lines' endpoints; only l0 scales.
        s.distance_ll11.push(DistanceLL11 {
            a: l0, b: l1, distance: 2.0, nid: 0, cid: 0,
            hb: arael::model::CrossBlock::new(),
        });
        s.dimensions.push(Dimension {
            did: 0,
            kind: DimensionKind::PointPointDistance(
                DimensionEndpoint::LineP1(l0), DimensionEndpoint::LineP1(l1)),
            value: 2.0, offset: v(0.0, 1.0), text_along: 0.0, name: "d0".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.assign_constraint_names();
        let report = apply_scale(&mut s, &[l0], &[], &[], v(0.0, 0.0), 2.0);
        assert!(report.scaled.is_empty());
        assert_eq!(report.left, vec![("d0".to_string(), "spans unscaled geometry")]);
        assert!(near(s.dimensions[0].value, 2.0), "boundary dim value untouched");
    }

    #[test]
    fn test_expression_dim_left_angular_ignored() {
        let mut s = Sketch::new();
        let l = s.add_line(v(0.0, 0.0), v(3.0, 0.0));
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::LineLength(l), value: 3.0,
            offset: v(0.0, 1.0), text_along: 0.0, name: "d0".into(),
            expr_str: Some("w * 2".into()), broken: false, derived: false, range: None,
        });
        // An angular dim on the same line: invariant, not reported.
        s.lines[l].constraints.has_angle = true;
        s.lines[l].constraints.target_angle = 0.0;
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::LineAngle(l), value: 0.0,
            offset: v(0.0, 1.0), text_along: 0.0, name: "d1".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.assign_constraint_names();
        let report = apply_scale(&mut s, &[l], &[], &[], v(0.0, 0.0), 2.0);
        assert!(report.scaled.is_empty());
        assert_eq!(report.left, vec![("d0".to_string(), "expression")]);
        assert!(near(s.dimensions[1].value, 0.0));
        assert!(near(s.lines[l].constraints.target_angle, 0.0), "angle target untouched");
    }

    #[test]
    fn test_literal_range_bounds_scale() {
        let mut s = Sketch::new();
        let l = s.add_line(v(0.0, 0.0), v(3.0, 0.0));
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::LineLength(l), value: 3.0,
            offset: v(0.0, 1.0), text_along: 0.0, name: "d0".into(),
            expr_str: None, broken: false, derived: false,
            range: Some(RangeBound::Between(RangeValue::Literal(2.0), RangeValue::Literal(4.0))),
        });
        s.assign_constraint_names();
        let report = apply_scale(&mut s, &[l], &[], &[], v(0.0, 0.0), 2.0);
        assert_eq!(report.scaled, vec!["d0".to_string()]);
        let Some(RangeBound::Between(RangeValue::Literal(lo), RangeValue::Literal(hi))) =
            s.dimensions[0].range.clone() else { panic!("range shape changed") };
        assert!(near(lo, 4.0) && near(hi, 8.0), "bounds scaled: {} {}", lo, hi);
    }
}
