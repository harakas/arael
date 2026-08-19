//! Meta-constraints: a recorded operation (an offset, later patterns)
//! that owns the entities, constraints and dimensions it created and is
//! edited as one thing. Created and kept consistent by the backend; this
//! is the record the sketch carries. Every kind reports what it owns
//! through [`Meta`], so one reconcile serves them all.

use arael::refs::Ref;

use crate::{Arc, Line};

/// A line or an arc, as a segment of a sequence or a result.
#[derive(Clone, Copy, PartialEq, Eq, Debug, serde::Serialize, serde::Deserialize)]
pub enum OffsetEntity {
    Line(Ref<Line>),
    Arc(Ref<Arc>),
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
pub struct OffsetValue {
    pub value: f64,
    pub expr: Option<String>,
}

/// An owned dimension and what the operation wrote into it; the
/// reconcile drops the meta when the dimension no longer says that.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub struct OffsetDim {
    pub did: u32,
    pub expect: OffsetValue,
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
    pub distance: OffsetValue,
    pub distance2: Option<OffsetValue>,
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

/// What kind of operation a meta-constraint records.
#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub enum MetaKind {
    Offset(Offset),
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
    /// The entities this operation created.
    pub fn owned_entities(&self) -> Vec<OffsetEntity> {
        match &self.kind {
            MetaKind::Offset(o) => o.result_entities().collect(),
        }
    }

    /// The result entities grouped: one group per side of an offset
    /// (segments first, then its corner arcs). Each group gets a marker.
    pub fn result_groups(&self) -> Vec<Vec<OffsetEntity>> {
        match &self.kind {
            MetaKind::Offset(o) => o
                .sides
                .iter()
                .map(|s| s.segs.iter().chain(s.corners.iter()).copied().collect())
                .collect(),
        }
    }

    /// The entities this operation was made from.
    pub fn source_entities(&self) -> Vec<OffsetEntity> {
        match &self.kind {
            MetaKind::Offset(o) => o.source.iter().map(|s| s.entity).collect(),
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
        }
    }

    /// Soft-owned constraints (nids): cleaned up with the meta, but the
    /// meta survives their deletion.
    pub fn soft_owned_constraints(&self) -> Vec<u32> {
        match &self.kind {
            MetaKind::Offset(o) => o.sides.iter().flat_map(|s| s.pins.iter().copied()).collect(),
        }
    }

    /// Owned dimensions with what was written into them.
    pub fn owned_dims(&self) -> Vec<OffsetDim> {
        match &self.kind {
            MetaKind::Offset(o) => o.sides.iter().flat_map(|s| s.dims.iter().cloned()).collect(),
        }
    }

    /// True when `e` is one of the result entities.
    pub fn owns_entity(&self, e: OffsetEntity) -> bool {
        self.owned_entities().contains(&e)
    }

    /// The offset record, for a meta of that kind.
    pub fn as_offset(&self) -> Option<&Offset> {
        match &self.kind {
            MetaKind::Offset(o) => Some(o),
        }
    }

    pub fn as_offset_mut(&mut self) -> Option<&mut Offset> {
        match &mut self.kind {
            MetaKind::Offset(o) => Some(o),
        }
    }

    /// Short kind word for listings.
    pub fn kind_name(&self) -> &'static str {
        match &self.kind {
            MetaKind::Offset(_) => "offset",
        }
    }
}
