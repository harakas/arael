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
