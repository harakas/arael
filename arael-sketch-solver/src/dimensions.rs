use arael::refs::Ref;
use arael::vect::vect2d;

use crate::{Point, Line, Arc};

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
    /// Angle between two lines. The bool is `supplement`: when true,
    /// constrains the supplementary angle (pi - angle) instead.
    Angle(Ref<Line>, Ref<Line>, bool),
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
            DimensionKind::Angle(a, b, _) => *a == r || *b == r,
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
    /// Expression string for parametric dimensions (e.g. "d0 * 2").
    /// None = constant numeric value.
    #[serde(default)]
    pub expr_str: Option<String>,
}

impl Dimension {
    /// Build a symbolic expression for the measured property of this
    /// dimension kind, using entity names from the sketch.
    pub fn measured_symbol(&self, sketch: &super::Sketch) -> arael_sym::E {
        use arael_sym::symbol;
        match &self.kind {
            DimensionKind::LineLength(r) => {
                let name = &sketch.lines[*r].name;
                symbol(&format!("{}.length", name))
            }
            DimensionKind::ArcRadius(r) => {
                let name = &sketch.arcs[*r].name;
                symbol(&format!("{}.radius", name))
            }
            DimensionKind::PointPointDistance(a, b) => {
                let pa = dim_endpoint_symbol(a, sketch);
                let pb = dim_endpoint_symbol(b, sketch);
                let dx = pa.0 - pb.0;
                let dy = pa.1 - pb.1;
                arael_sym::sqrt(dx.clone() * dx + dy.clone() * dy)
            }
            DimensionKind::PointLineDistance(pt, line) => {
                let (px, py) = dim_endpoint_symbol(pt, sketch);
                let ln = &sketch.lines[*line].name;
                let p1x = symbol(&format!("{}.p1.x", ln));
                let p1y = symbol(&format!("{}.p1.y", ln));
                let p2x = symbol(&format!("{}.p2.x", ln));
                let p2y = symbol(&format!("{}.p2.y", ln));
                let dx = p2x.clone() - p1x.clone();
                let dy = p2y.clone() - p1y.clone();
                let len = arael_sym::sqrt(dx.clone() * dx.clone() + dy.clone() * dy.clone());
                let signed = ((px - p1x) * dy.clone() - (py - p1y) * dx.clone()) / len;
                // Determine sign from current geometry so expression matches
                // the positive value the user sees. Negate if point is on
                // the negative side of the line.
                let pt_pos = dim_endpoint_pos(pt, sketch);
                let l = &sketch.lines[*line];
                let ldx = l.p2.value.x - l.p1.value.x;
                let ldy = l.p2.value.y - l.p1.value.y;
                let cross = (pt_pos.x - l.p1.value.x) * ldy - (pt_pos.y - l.p1.value.y) * ldx;
                if cross >= 0.0 { signed } else { -signed }
            }
            DimensionKind::Angle(a, b, supplement) => {
                let la = &sketch.lines[*a].name;
                let lb = &sketch.lines[*b].name;
                let dx1 = symbol(&format!("{}.p2.x", la)) - symbol(&format!("{}.p1.x", la));
                let dy1 = symbol(&format!("{}.p2.y", la)) - symbol(&format!("{}.p1.y", la));
                let dx2 = symbol(&format!("{}.p2.x", lb)) - symbol(&format!("{}.p1.x", lb));
                let dy2 = symbol(&format!("{}.p2.y", lb)) - symbol(&format!("{}.p1.y", lb));
                let cross = dx1.clone() * dy2.clone() - dy1.clone() * dx2.clone();
                let dot = dx1 * dx2 + dy1 * dy2;
                let angle = arael_sym::atan2(cross, dot);
                // Determine sign from current geometry
                let la_line = &sketch.lines[*a];
                let lb_line = &sketch.lines[*b];
                let cur_dx1 = la_line.p2.value.x - la_line.p1.value.x;
                let cur_dy1 = la_line.p2.value.y - la_line.p1.value.y;
                let cur_dx2 = lb_line.p2.value.x - lb_line.p1.value.x;
                let cur_dy2 = lb_line.p2.value.y - lb_line.p1.value.y;
                let cur_cross = cur_dx1 * cur_dy2 - cur_dy1 * cur_dx2;
                let cur_dot = cur_dx1 * cur_dx2 + cur_dy1 * cur_dy2;
                let cur_angle = cur_cross.atan2(cur_dot);
                // Convert to degrees, matching sign of current angle
                let deg_factor = arael_sym::constant(180.0 / std::f64::consts::PI);
                let signed_deg = angle * deg_factor;
                if *supplement {
                    let sup_sign = if cur_angle >= 0.0 { 1.0 } else { -1.0 };
                    arael_sym::constant(sup_sign * 180.0) - signed_deg
                } else {
                    if cur_angle >= 0.0 { signed_deg } else { -signed_deg }
                }
            }
        }
    }
}

/// Get the current numeric position of a DimensionEndpoint.
fn dim_endpoint_pos(ep: &DimensionEndpoint, sketch: &super::Sketch) -> vect2d {
    match ep {
        DimensionEndpoint::Point(r) => sketch.points[*r].pos.value,
        DimensionEndpoint::LineP1(r) => sketch.lines[*r].p1.value,
        DimensionEndpoint::LineP2(r) => sketch.lines[*r].p2.value,
        DimensionEndpoint::ArcCenter(r) => sketch.arcs[*r].center.value,
        DimensionEndpoint::ArcStart(r) => {
            let a = &sketch.arcs[*r];
            vect2d::new(a.center.value.x + a.radius.value * a.start_angle.value.cos(),
                        a.center.value.y + a.radius.value * a.start_angle.value.sin())
        }
        DimensionEndpoint::ArcEnd(r) => {
            let a = &sketch.arcs[*r];
            vect2d::new(a.center.value.x + a.radius.value * a.end_angle.value.cos(),
                        a.center.value.y + a.radius.value * a.end_angle.value.sin())
        }
    }
}

/// Build symbolic (x, y) expressions for a DimensionEndpoint.
fn dim_endpoint_symbol(ep: &DimensionEndpoint, sketch: &super::Sketch) -> (arael_sym::E, arael_sym::E) {
    use arael_sym::symbol;
    match ep {
        DimensionEndpoint::Point(r) => {
            let n = &sketch.points[*r].name;
            (symbol(&format!("{}.pos.x", n)), symbol(&format!("{}.pos.y", n)))
        }
        DimensionEndpoint::LineP1(r) => {
            let n = &sketch.lines[*r].name;
            (symbol(&format!("{}.p1.x", n)), symbol(&format!("{}.p1.y", n)))
        }
        DimensionEndpoint::LineP2(r) => {
            let n = &sketch.lines[*r].name;
            (symbol(&format!("{}.p2.x", n)), symbol(&format!("{}.p2.y", n)))
        }
        DimensionEndpoint::ArcCenter(r) => {
            let n = &sketch.arcs[*r].name;
            (symbol(&format!("{}.center.x", n)), symbol(&format!("{}.center.y", n)))
        }
        DimensionEndpoint::ArcStart(r) => {
            let n = &sketch.arcs[*r].name;
            let cx = symbol(&format!("{}.center.x", n));
            let cy = symbol(&format!("{}.center.y", n));
            let r = symbol(&format!("{}.radius", n));
            let sa = symbol(&format!("{}.start_angle", n));
            (cx + r.clone() * arael_sym::cos(sa.clone()), cy + r * arael_sym::sin(sa))
        }
        DimensionEndpoint::ArcEnd(r) => {
            let n = &sketch.arcs[*r].name;
            let cx = symbol(&format!("{}.center.x", n));
            let cy = symbol(&format!("{}.center.y", n));
            let r = symbol(&format!("{}.radius", n));
            let ea = symbol(&format!("{}.end_angle", n));
            (cx + r.clone() * arael_sym::cos(ea.clone()), cy + r * arael_sym::sin(ea))
        }
    }
}
