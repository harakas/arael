//! Meta-constraints: a recorded operation (an offset, a pattern) that
//! owns the entities, constraints and dimensions it created and is
//! edited as one thing. Created and kept consistent by the backend; this
//! is the record the sketch carries. Every kind reports what it owns
//! through [`Meta`], so one reconcile serves them all.

use arael::refs::Ref;

use crate::{Arc, DimensionEndpoint, Line, Point};

/// Any entity a meta-constraint can own or be made from.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, serde::Serialize, serde::Deserialize)]
pub enum MetaEntity {
    Line(Ref<Line>),
    Arc(Ref<Arc>),
    Point(Ref<Point>),
}

/// A line or an arc, as a segment of a sequence or a result.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum OffsetEntity {
    Line(Ref<Line>),
    Arc(Ref<Arc>),
}

impl From<OffsetEntity> for MetaEntity {
    fn from(e: OffsetEntity) -> Self {
        match e {
            OffsetEntity::Line(l) => MetaEntity::Line(l),
            OffsetEntity::Arc(a) => MetaEntity::Arc(a),
        }
    }
}

impl MetaEntity {
    /// The offset's view of the entity: lines and arcs only.
    pub fn as_offset_entity(self) -> Option<OffsetEntity> {
        match self {
            MetaEntity::Line(l) => Some(OffsetEntity::Line(l)),
            MetaEntity::Arc(a) => Some(OffsetEntity::Arc(a)),
            MetaEntity::Point(_) => None,
        }
    }
}

/// One segment of a source sequence, in chain order; `reversed` when
/// the chain traverses it p2 -> p1 (end -> start).
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub struct OffsetSource {
    pub entity: OffsetEntity,
    pub reversed: bool,
}

/// How many sides an offset has, and with which distances.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum OffsetKind {
    /// One chain at `distance`, on the side `side` names.
    OneSide,
    /// Two chains, `distance` on both sides.
    Symmetric,
    /// Two chains: `distance` on the side `side` names, `distance2` on the other.
    TwoSides,
}

/// A value as typed: the number, and the live expression it came from
/// when it was one. Written into owned dimensions verbatim.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct MetaValue {
    pub value: f64,
    pub expr: Option<String>,
}

/// An owned dimension and what the operation wrote into it; the
/// reconcile drops the meta when the dimension no longer says that.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct OffsetDim {
    pub did: u32,
    pub expect: MetaValue,
}

/// An offset's result on one side.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct OffsetSideResult {
    /// +1 left of the chain direction, -1 right.
    pub sign: f64,
    /// The result entities, in chain order: one per source segment
    /// except the `dropped` ones.
    pub segs: Vec<OffsetEntity>,
    /// Indices (into the offset's `source`) of the segments whose offset
    /// vanished on this side (an arc offset inward past its radius);
    /// their neighbours meet directly.
    #[serde(default)]
    pub dropped: Vec<usize>,
    /// Round-corner arcs (arcs of the distance around convex source
    /// corners), in joint order.
    #[serde(default)]
    pub corners: Vec<OffsetEntity>,
    /// Owned constraints (nids): the per-segment relations and the
    /// joints. Deleting one drops the offset.
    pub constraints: Vec<u32>,
    /// Soft-owned pins (nids): regenerated and cleaned up with the
    /// offset, but deleting one by hand keeps it.
    pub pins: Vec<u32>,
    /// Owned dimensions: the per-segment distances (and the corner
    /// arcs' radii).
    pub dims: Vec<OffsetDim>,
}

/// How the ends of an open two-sided offset are closed.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub enum CapKind {
    #[default]
    None,
    /// A line across each end.
    Line,
    /// A half circle around each source end, tangent to both results
    /// (symmetric offsets only).
    Round,
}

/// The caps of an open two-sided offset: what was made and what holds it.
#[derive(Clone, PartialEq, Debug, Default, serde::Serialize, serde::Deserialize)]
pub struct OffsetCaps {
    pub kind: CapKind,
    /// The cap entities: the start cap, then the end cap.
    pub entities: Vec<OffsetEntity>,
    /// Owned constraints (nids).
    pub constraints: Vec<u32>,
}

/// The offset operation's record.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Offset {
    pub source: Vec<OffsetSource>,
    pub closed: bool,
    pub kind: OffsetKind,
    pub distance: MetaValue,
    pub distance2: Option<MetaValue>,
    /// The side `distance` is on: +1 left of the chain direction, -1 right.
    pub side: f64,
    /// Whether the free ends and tangent joints were pinned (`on_normal`).
    pub pinned: bool,
    /// Convex corners rounded with an arc of the distance instead of
    /// extended to a sharp corner.
    #[serde(default)]
    pub round: bool,
    pub sides: Vec<OffsetSideResult>,
    #[serde(default)]
    pub caps: OffsetCaps,
}

impl OffsetSideResult {
    /// The source index of every result segment, in order: all sources
    /// but the dropped ones.
    pub fn sources(&self, source_count: usize) -> Vec<usize> {
        (0..source_count).filter(|i| !self.dropped.contains(i)).collect()
    }
}

impl Offset {
    /// Every result entity: both sides, corner arcs and caps included.
    pub fn result_entities(&self) -> impl Iterator<Item = OffsetEntity> + '_ {
        self.sides
            .iter()
            .flat_map(|s| s.segs.iter().copied().chain(s.corners.iter().copied()))
            .chain(self.caps.entities.iter().copied())
    }
}

/// How a circular pattern spreads its instances.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Distribution {
    /// Over the full circle.
    Full,
    /// Over `angle`, starting at the source.
    Partial,
    /// Over `angle`, centered on the source.
    Symmetric,
}

/// The center of a circular pattern: a point, or an endpoint / arc
/// center that a hidden helper point follows.
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub enum CenterRef {
    Point(Ref<Point>),
    Endpoint(DimensionEndpoint),
}

/// One axis of a rectangular pattern.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct PatternAxis {
    /// Instances along this axis, the source included (1 = off).
    pub quantity: u32,
    /// Between consecutive instances, or first to last (`extent`);
    /// negative reverses the axis.
    pub distance: MetaValue,
    /// Instances on both sides of the source (one less on the backward
    /// side for an even quantity).
    pub symmetric: bool,
}

/// What a pattern does to its sources.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub enum PatternKind {
    Circular {
        center: CenterRef,
        /// The hidden helper point standing in for an endpoint center.
        helper: Option<Ref<Point>>,
        distribution: Distribution,
        /// Degrees; unused for `Full`.
        angle: MetaValue,
        /// Instances, the source included.
        quantity: u32,
    },
    Rectangular {
        /// Axis 1 runs along this line, axis 2 across it to the left;
        /// none: +x and +y.
        frame: Option<Ref<Line>>,
        /// The distances span first to last instead of one step.
        extent: bool,
        axis1: PatternAxis,
        axis2: PatternAxis,
    },
}

/// One instance of a pattern.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct PatternCopy {
    /// Grid position (axis 1, axis 2) for a rectangular pattern, (step,
    /// 0) for a circular one; the source is (0, 0).
    pub index: (i32, i32),
    /// The copy entities, parallel to the pattern's `sources`.
    pub entities: Vec<MetaEntity>,
    /// Owned constraints (nids): the image constraints and the
    /// coincidences recreated among the copy.
    pub constraints: Vec<u32>,
}

/// The pattern operation's record.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Pattern {
    pub sources: Vec<MetaEntity>,
    pub kind: PatternKind,
    pub copies: Vec<PatternCopy>,
    /// Owned constraints of the pattern as a whole (the helper point's
    /// coincidence with the center endpoint).
    #[serde(default)]
    pub constraints: Vec<u32>,
}

impl Pattern {
    /// The entities the pattern depends on beyond its sources: the
    /// center point / frame line.
    pub fn reference_entities(&self) -> Vec<MetaEntity> {
        match &self.kind {
            PatternKind::Circular { center, .. } => match center {
                CenterRef::Point(p) => vec![MetaEntity::Point(*p)],
                CenterRef::Endpoint(e) => match e {
                    DimensionEndpoint::Point(p) => vec![MetaEntity::Point(*p)],
                    DimensionEndpoint::LineP1(l) | DimensionEndpoint::LineP2(l) => vec![MetaEntity::Line(*l)],
                    DimensionEndpoint::ArcCenter(a) | DimensionEndpoint::ArcStart(a) | DimensionEndpoint::ArcEnd(a) => {
                        vec![MetaEntity::Arc(*a)]
                    }
                },
            },
            PatternKind::Rectangular { frame, .. } => frame.iter().map(|l| MetaEntity::Line(*l)).collect(),
        }
    }

    /// The hidden helper point of a circular pattern about an endpoint.
    pub fn helper(&self) -> Option<Ref<Point>> {
        match &self.kind {
            PatternKind::Circular { helper, .. } => *helper,
            PatternKind::Rectangular { .. } => None,
        }
    }
}

/// What kind of operation a meta-constraint records.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub enum MetaKind {
    Offset(Offset),
    Pattern(Pattern),
}

/// A meta-constraint: the record of one operation, named `M<n>`.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct Meta {
    /// Permanent id; the name is `M<mid>`.
    pub mid: u32,
    pub name: String,
    pub kind: MetaKind,
}

impl Meta {
    /// The entities this operation created (a pattern's helper point
    /// included).
    pub fn owned_entities(&self) -> Vec<MetaEntity> {
        match &self.kind {
            MetaKind::Offset(o) => o.result_entities().map(MetaEntity::from).collect(),
            MetaKind::Pattern(p) => p
                .copies
                .iter()
                .flat_map(|c| c.entities.iter().copied())
                .chain(p.helper().map(MetaEntity::Point))
                .collect(),
        }
    }

    /// The result entities grouped: one group per side of an offset
    /// (segments first, then its corner arcs), one per copy of a
    /// pattern. Each group gets a marker.
    pub fn result_groups(&self) -> Vec<Vec<MetaEntity>> {
        match &self.kind {
            MetaKind::Offset(o) => o
                .sides
                .iter()
                .map(|s| s.segs.iter().chain(s.corners.iter()).map(|e| MetaEntity::from(*e)).collect())
                .collect(),
            MetaKind::Pattern(p) => p.copies.iter().map(|c| c.entities.clone()).collect(),
        }
    }

    /// The entities this operation was made from (a pattern's center /
    /// frame included).
    pub fn source_entities(&self) -> Vec<MetaEntity> {
        match &self.kind {
            MetaKind::Offset(o) => o.source.iter().map(|s| MetaEntity::from(s.entity)).collect(),
            MetaKind::Pattern(p) => p.sources.iter().copied().chain(p.reference_entities()).collect(),
        }
    }

    /// Owned constraints (nids); deleting one drops the meta.
    pub fn owned_constraints(&self) -> Vec<u32> {
        match &self.kind {
            MetaKind::Offset(o) => o
                .sides
                .iter()
                .flat_map(|s| s.constraints.iter().copied())
                .chain(o.caps.constraints.iter().copied())
                .collect(),
            MetaKind::Pattern(p) => p
                .copies
                .iter()
                .flat_map(|c| c.constraints.iter().copied())
                .chain(p.constraints.iter().copied())
                .collect(),
        }
    }

    /// Soft-owned constraints (nids): cleaned up with the meta, but the
    /// meta survives their deletion.
    pub fn soft_owned_constraints(&self) -> Vec<u32> {
        match &self.kind {
            MetaKind::Offset(o) => o.sides.iter().flat_map(|s| s.pins.iter().copied()).collect(),
            MetaKind::Pattern(_) => Vec::new(),
        }
    }

    /// Owned dimensions with what was written into them.
    pub fn owned_dims(&self) -> Vec<OffsetDim> {
        match &self.kind {
            MetaKind::Offset(o) => o.sides.iter().flat_map(|s| s.dims.iter().cloned()).collect(),
            MetaKind::Pattern(_) => Vec::new(),
        }
    }

    /// True when `e` is one of the result entities.
    pub fn owns_entity(&self, e: MetaEntity) -> bool {
        self.owned_entities().contains(&e)
    }

    /// The offset record, for a meta of that kind.
    pub fn as_offset(&self) -> Option<&Offset> {
        match &self.kind {
            MetaKind::Offset(o) => Some(o),
            MetaKind::Pattern(_) => None,
        }
    }

    pub fn as_offset_mut(&mut self) -> Option<&mut Offset> {
        match &mut self.kind {
            MetaKind::Offset(o) => Some(o),
            MetaKind::Pattern(_) => None,
        }
    }

    /// The pattern record, for a meta of that kind.
    pub fn as_pattern(&self) -> Option<&Pattern> {
        match &self.kind {
            MetaKind::Pattern(p) => Some(p),
            MetaKind::Offset(_) => None,
        }
    }

    pub fn as_pattern_mut(&mut self) -> Option<&mut Pattern> {
        match &mut self.kind {
            MetaKind::Pattern(p) => Some(p),
            MetaKind::Offset(_) => None,
        }
    }

    /// Short kind word for listings.
    pub fn kind_name(&self) -> &'static str {
        match &self.kind {
            MetaKind::Offset(_) => "offset",
            MetaKind::Pattern(_) => "pattern",
        }
    }
}
