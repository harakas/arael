//! Split/trim engine: cut a line or arc at resolved parameters,
//! transfer every reference from the target onto the pieces, and
//! delete the target. See docs/dev/TRIMSPLIT.md for the design.
//!
//! The transfer is driven by the per-field [`RefRole`] annotations in
//! the solver's constraint registry: Endpoint roles follow the piece
//! owning that endpoint, Host follows the piece nearest the
//! constraint's other referents, Whole replicates onto every kept
//! piece, Contact follows the tangency point, Extent is dropped.
//! Dimensions retarget (one `did`, one target); their backing numeric
//! constraints are swapped through the same push/remove helpers the
//! dimension actions use. Expression strings are rewritten token-wise;
//! expressions left referencing the target unresolvably mark their
//! owner broken.

use arael::model::Param;
use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;
use std::collections::HashMap;

use crate::actions::{push_numeric_dim_constraint, remove_numeric_dim_constraint, Action};
use crate::geometry::{nearest_arc_param, project_onto_segment};

// ---------------------------------------------------------------------------
// Plan
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub enum SplitTarget {
    Line(Ref<Line>),
    Arc(Ref<Arc>),
}

/// Entity a cut was found on, kept so the cut endpoint can be pinned
/// onto it afterwards.
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize)]
pub enum Cutter {
    Line(Ref<Line>),
    Arc(Ref<Arc>),
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SplitCut {
    /// Parameter on the target: t in [0,1] for a line, the in-span
    /// parametric angle for an arc.
    pub param: f64,
    pub pos: vect2d,
    pub cutter: Option<Cutter>,
}

/// A fully resolved split: where to cut and which pieces survive.
/// `cuts` are sorted along the target's own direction; `keep` has one
/// entry per piece (`cuts.len() + 1` pieces for an open target,
/// `cuts.len()` for a closed one).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct SplitPlan {
    pub target: SplitTarget,
    pub cuts: Vec<SplitCut>,
    pub keep: Vec<bool>,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum PieceRef {
    Line(Ref<Line>),
    Arc(Ref<Arc>),
}

impl PieceRef {
    pub fn line(self) -> Option<Ref<Line>> {
        if let PieceRef::Line(r) = self { Some(r) } else { None }
    }
    pub fn arc(self) -> Option<Ref<Arc>> {
        if let PieceRef::Arc(r) = self { Some(r) } else { None }
    }
}

/// What a split did, for the caller's report. Piece slots align with
/// the plan's piece indices; trimmed pieces are `None`.
pub struct SplitOutcome {
    pub pieces: Vec<Option<PieceRef>>,
    pub piece_names: Vec<Option<String>>,
    /// Constraints/dimensions whose references were retargeted.
    pub moved: Vec<String>,
    /// Whole-role constraint copies and replicated H/V flags.
    pub copied: Vec<String>,
    /// Constraints/dimensions dropped because their meaning has no
    /// successor (Extent roles, trimmed-away endpoints, arc sweep).
    pub dropped: Vec<String>,
    /// Expression rewrites and broken markings.
    pub expr_report: Vec<String>,
}

pub fn piece_count(closed: bool, ncuts: usize) -> usize {
    if closed { ncuts } else { ncuts + 1 }
}

// ---------------------------------------------------------------------------
// Target geometry snapshot
// ---------------------------------------------------------------------------

struct TargetGeom {
    name: String,
    // Line data
    p1: Param<vect2d>,
    p2: Param<vect2d>,
    horizontal: bool,
    vertical: bool,
    h_dir_sign: f64,
    v_dir_sign: f64,
    // Arc data
    center: Param<vect2d>,
    radius: Param<f64>,
    radius_b: Param<f64>,
    rotation: Param<f64>,
    sa: f64,
    ea: f64,
    ccw: bool,
    closed: bool,
    is_ellipse: bool,
    // Shared
    style: LineStyle,
    construction: bool,
    quiet: bool,
}

fn capture_geom(sketch: &Sketch, target: SplitTarget) -> Result<TargetGeom, String> {
    match target {
        SplitTarget::Line(r) => {
            let l = sketch.lines.get(r).ok_or("split: line no longer exists")?;
            Ok(TargetGeom {
                name: l.name.clone(),
                p1: l.p1.clone(),
                p2: l.p2.clone(),
                horizontal: l.constraints.horizontal,
                vertical: l.constraints.vertical,
                h_dir_sign: l.constraints.h_dir_sign,
                v_dir_sign: l.constraints.v_dir_sign,
                center: Param::fixed(vect2d::new(0.0, 0.0)),
                radius: Param::fixed(0.0),
                radius_b: Param::fixed(0.0),
                rotation: Param::fixed(0.0),
                sa: 0.0,
                ea: 0.0,
                ccw: true,
                closed: false,
                is_ellipse: false,
                style: l.style,
                construction: l.construction,
                quiet: l.quiet,
            })
        }
        SplitTarget::Arc(r) => {
            let a = sketch.arcs.get(r).ok_or("split: arc no longer exists")?;
            Ok(TargetGeom {
                name: a.name.clone(),
                p1: Param::fixed(vect2d::new(0.0, 0.0)),
                p2: Param::fixed(vect2d::new(0.0, 0.0)),
                horizontal: false,
                vertical: false,
                h_dir_sign: f64::NAN,
                v_dir_sign: f64::NAN,
                center: a.center.clone(),
                radius: a.radius.clone(),
                radius_b: a.radius_b.clone(),
                rotation: a.rotation.clone(),
                sa: a.start_angle.value,
                ea: a.end_angle.value,
                ccw: a.ccw,
                closed: a.closed,
                is_ellipse: a.is_ellipse,
                style: a.style,
                construction: a.construction,
                quiet: a.quiet,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Piece geometry
// ---------------------------------------------------------------------------

/// Distance-from-start in parameter units, monotonic along the
/// target's own direction. Closed arcs use ascending-from-start span
/// coordinates (their stored ccw flag is meaningless for a full
/// circle, and `arc_param_in_span` already maps every angle that way).
fn s_of(geom: &TargetGeom, is_line: bool, t: f64) -> f64 {
    if is_line {
        t
    } else if geom.closed || geom.ccw {
        t - geom.sa
    } else {
        geom.sa - t
    }
}

/// Piece boundaries in target parameter coordinates, one more entry
/// than pieces. Open targets bracket the sorted cuts with the ends;
/// closed targets wrap the last piece back to the first cut + TAU.
fn boundaries(geom: &TargetGeom, is_line: bool, cuts: &[SplitCut]) -> Vec<f64> {
    let mut b: Vec<f64> = Vec::with_capacity(cuts.len() + 2);
    if geom.closed && !is_line {
        for c in cuts {
            b.push(c.param);
        }
        b.push(cuts[0].param + std::f64::consts::TAU);
    } else {
        b.push(if is_line { 0.0 } else { geom.sa });
        for c in cuts {
            b.push(c.param);
        }
        b.push(if is_line { 1.0 } else { geom.ea });
    }
    b
}

/// Piece index whose span contains parameter `t`.
fn piece_of_param(geom: &TargetGeom, is_line: bool, bounds: &[f64], t: f64) -> usize {
    let n = bounds.len() - 1;
    let mut s = s_of(geom, is_line, t);
    if geom.closed && !is_line {
        let s0 = s_of(geom, is_line, bounds[0]);
        let mut rel = s - s0;
        while rel < 0.0 {
            rel += std::f64::consts::TAU;
        }
        s = s0 + rel;
    }
    for i in 0..n {
        if s <= s_of(geom, is_line, bounds[i + 1]) {
            return i;
        }
    }
    n - 1
}

fn line_point_at(geom: &TargetGeom, t: f64) -> vect2d {
    vect2d::new(
        geom.p1.value.x + t * (geom.p2.value.x - geom.p1.value.x),
        geom.p1.value.y + t * (geom.p2.value.y - geom.p1.value.y),
    )
}

/// Create the kept pieces. Returns refs aligned to piece indices.
fn create_pieces(
    sketch: &mut Sketch,
    geom: &TargetGeom,
    is_line: bool,
    plan: &SplitPlan,
    bounds: &[f64],
) -> Vec<Option<PieceRef>> {
    let n = plan.keep.len();
    let mut pieces: Vec<Option<PieceRef>> = vec![None; n];
    for i in 0..n {
        if !plan.keep[i] {
            continue;
        }
        if is_line {
            let a = line_point_at(geom, bounds[i]);
            let b = line_point_at(geom, bounds[i + 1]);
            let r = sketch.add_line(a, b);
            {
                let l = &mut sketch.lines[r];
                l.constraints.horizontal = geom.horizontal;
                l.constraints.vertical = geom.vertical;
                l.constraints.h_dir_sign = geom.h_dir_sign;
                l.constraints.v_dir_sign = geom.v_dir_sign;
                l.style = geom.style;
                l.construction = geom.construction;
                l.quiet = geom.quiet;
                // Outer endpoints inherit the original Param (value
                // and locked state); interior cut endpoints stay free.
                if i == 0 {
                    l.p1 = geom.p1.clone();
                }
                if i == n - 1 {
                    l.p2 = geom.p2.clone();
                }
            }
            pieces[i] = Some(PieceRef::Line(r));
        } else {
            // Closed targets emit ascending (ccw) pieces; open pieces
            // inherit the target's direction.
            let ccw = if geom.closed { true } else { geom.ccw };
            let r = if geom.is_ellipse {
                sketch.add_elliptic_arc(
                    geom.center.value,
                    geom.radius.value,
                    geom.radius_b.value,
                    geom.rotation.value,
                    bounds[i],
                    bounds[i + 1],
                    ccw,
                )
            } else {
                sketch.add_arc_with_dir(
                    geom.center.value,
                    geom.radius.value,
                    bounds[i],
                    bounds[i + 1],
                    false,
                    ccw,
                )
            };
            {
                let a = &mut sketch.arcs[r];
                // Exact Param copies preserve values and any locked
                // state on center/radius/rotation.
                a.center = geom.center.clone();
                a.radius = geom.radius.clone();
                a.radius_b = geom.radius_b.clone();
                a.rotation = geom.rotation.clone();
                a.style = geom.style;
                a.construction = geom.construction;
                a.quiet = geom.quiet;
            }
            pieces[i] = Some(PieceRef::Arc(r));
        }
    }
    pieces
}

// ---------------------------------------------------------------------------
// Piece lookups
// ---------------------------------------------------------------------------

struct PieceMap<'a> {
    pieces: &'a [Option<PieceRef>],
}

impl<'a> PieceMap<'a> {
    fn first_kept(&self) -> Option<PieceRef> {
        self.pieces.iter().flatten().next().copied()
    }
    fn kept(&self) -> impl Iterator<Item = PieceRef> + '_ {
        self.pieces.iter().flatten().copied()
    }
    /// Piece owning the original start (p1 / start_angle end).
    fn start_owner(&self) -> Option<PieceRef> {
        self.pieces[0]
    }
    /// Piece owning the original end (p2 / end_angle end).
    fn end_owner(&self) -> Option<PieceRef> {
        self.pieces[self.pieces.len() - 1]
    }
    fn at(&self, i: usize) -> Option<PieceRef> {
        self.pieces[i]
    }
    /// Piece at `i` if kept, else the nearest kept piece by index.
    fn at_or_nearest(&self, i: usize) -> Option<PieceRef> {
        if let Some(p) = self.pieces[i] {
            return Some(p);
        }
        for d in 1..self.pieces.len() {
            if i >= d
                && let Some(p) = self.pieces[i - d]
            {
                return Some(p);
            }
            if i + d < self.pieces.len()
                && let Some(p) = self.pieces[i + d]
            {
                return Some(p);
            }
        }
        None
    }
    fn all_kept(&self) -> bool {
        self.pieces.iter().all(|p| p.is_some())
    }
}

// ---------------------------------------------------------------------------
// Dimension transfer
// ---------------------------------------------------------------------------

fn map_endpoint(
    ep: &DimensionEndpoint,
    target: SplitTarget,
    pm: &PieceMap,
) -> Result<DimensionEndpoint, ()> {
    match (*ep, target) {
        (DimensionEndpoint::LineP1(l), SplitTarget::Line(t)) if l == t => pm
            .start_owner()
            .and_then(PieceRef::line)
            .map(DimensionEndpoint::LineP1)
            .ok_or(()),
        (DimensionEndpoint::LineP2(l), SplitTarget::Line(t)) if l == t => pm
            .end_owner()
            .and_then(PieceRef::line)
            .map(DimensionEndpoint::LineP2)
            .ok_or(()),
        (DimensionEndpoint::ArcCenter(a), SplitTarget::Arc(t)) if a == t => pm
            .first_kept()
            .and_then(PieceRef::arc)
            .map(DimensionEndpoint::ArcCenter)
            .ok_or(()),
        (DimensionEndpoint::ArcStart(a), SplitTarget::Arc(t)) if a == t => pm
            .start_owner()
            .and_then(PieceRef::arc)
            .map(DimensionEndpoint::ArcStart)
            .ok_or(()),
        (DimensionEndpoint::ArcEnd(a), SplitTarget::Arc(t)) if a == t => pm
            .end_owner()
            .and_then(PieceRef::arc)
            .map(DimensionEndpoint::ArcEnd)
            .ok_or(()),
        _ => Ok(*ep),
    }
}

/// New kind for a dimension referencing the target, or Err(reason
/// label) when the dimension has no successor and must drop.
fn map_dim_kind(
    kind: &DimensionKind,
    target: SplitTarget,
    geom: &TargetGeom,
    pm: &PieceMap,
    bounds: &[f64],
    sketch: &Sketch,
) -> Result<DimensionKind, &'static str> {
    let is_line = matches!(target, SplitTarget::Line(_));
    match *kind {
        DimensionKind::LineLength(l) => {
            if let SplitTarget::Line(t) = target
                && l == t
            {
                // Both original endpoints must survive; the length of
                // the whole becomes the end-to-end distance.
                let first = pm.start_owner().and_then(PieceRef::line).ok_or("p1 trimmed away")?;
                let last = pm.end_owner().and_then(PieceRef::line).ok_or("p2 trimmed away")?;
                return Ok(DimensionKind::PointPointDistance(
                    DimensionEndpoint::LineP1(first),
                    DimensionEndpoint::LineP2(last),
                ));
            }
            Ok(*kind)
        }
        DimensionKind::ArcRadius(a) => match target {
            SplitTarget::Arc(t) if a == t => pm
                .first_kept()
                .and_then(PieceRef::arc)
                .map(DimensionKind::ArcRadius)
                .ok_or("no surviving piece"),
            _ => Ok(*kind),
        },
        DimensionKind::ArcRadiusB(a) => match target {
            SplitTarget::Arc(t) if a == t => pm
                .first_kept()
                .and_then(PieceRef::arc)
                .map(DimensionKind::ArcRadiusB)
                .ok_or("no surviving piece"),
            _ => Ok(*kind),
        },
        DimensionKind::ArcRotation(a) => match target {
            SplitTarget::Arc(t) if a == t => pm
                .first_kept()
                .and_then(PieceRef::arc)
                .map(DimensionKind::ArcRotation)
                .ok_or("no surviving piece"),
            _ => Ok(*kind),
        },
        DimensionKind::ArcSweep(a) => match target {
            SplitTarget::Arc(t) if a == t => Err("sweep of the whole arc has no successor"),
            _ => Ok(*kind),
        },
        DimensionKind::LineAngle(l) => match target {
            SplitTarget::Line(t) if l == t => pm
                .first_kept()
                .and_then(PieceRef::line)
                .map(DimensionKind::LineAngle)
                .ok_or("no surviving piece"),
            _ => Ok(*kind),
        },
        DimensionKind::Angle(a, b, sup) => {
            if let SplitTarget::Line(t) = target {
                let first = pm.first_kept().and_then(PieceRef::line);
                let na = if a == t { first.ok_or("no surviving piece")? } else { a };
                let nb = if b == t { first.ok_or("no surviving piece")? } else { b };
                return Ok(DimensionKind::Angle(na, nb, sup));
            }
            Ok(*kind)
        }
        DimensionKind::PointPointDistance(ref a, ref b) => {
            let na = map_endpoint(a, target, pm).map_err(|_| "endpoint trimmed away")?;
            let nb = map_endpoint(b, target, pm).map_err(|_| "endpoint trimmed away")?;
            Ok(DimensionKind::PointPointDistance(na, nb))
        }
        DimensionKind::HDistance(ref a, ref b) => {
            let na = map_endpoint(a, target, pm).map_err(|_| "endpoint trimmed away")?;
            let nb = map_endpoint(b, target, pm).map_err(|_| "endpoint trimmed away")?;
            Ok(DimensionKind::HDistance(na, nb))
        }
        DimensionKind::VDistance(ref a, ref b) => {
            let na = map_endpoint(a, target, pm).map_err(|_| "endpoint trimmed away")?;
            let nb = map_endpoint(b, target, pm).map_err(|_| "endpoint trimmed away")?;
            Ok(DimensionKind::VDistance(na, nb))
        }
        DimensionKind::PointLineDistance(ref pt, l) => {
            let npt = map_endpoint(pt, target, pm).map_err(|_| "endpoint trimmed away")?;
            let nl = if let SplitTarget::Line(t) = target
                && l == t
            {
                // Host: the piece nearest the measured point.
                let pos = crate::actions::dim_endpoint_pos_sketch(sketch, pt);
                let proj = project_onto_segment(pos, geom.p1.value, geom.p2.value);
                let t_par = line_param_of(geom, proj);
                let idx = piece_of_param(geom, is_line, bounds, t_par);
                pm.at_or_nearest(idx)
                    .and_then(PieceRef::line)
                    .ok_or("no surviving piece")?
            } else {
                l
            };
            Ok(DimensionKind::PointLineDistance(npt, nl))
        }
        DimensionKind::ConcentricDistance(a, b) => {
            if let SplitTarget::Arc(t) = target {
                let first = pm.first_kept().and_then(PieceRef::arc);
                let na = if a == t { first.ok_or("no surviving piece")? } else { a };
                let nb = if b == t { first.ok_or("no surviving piece")? } else { b };
                return Ok(DimensionKind::ConcentricDistance(na, nb));
            }
            Ok(*kind)
        }
        DimensionKind::LineLineDistance(a, b) => {
            if let SplitTarget::Line(t) = target {
                let first = pm.first_kept().and_then(PieceRef::line);
                let na = if a == t { first.ok_or("no surviving piece")? } else { a };
                let nb = if b == t { first.ok_or("no surviving piece")? } else { b };
                return Ok(DimensionKind::LineLineDistance(na, nb));
            }
            Ok(*kind)
        }
    }
}

fn line_param_of(geom: &TargetGeom, p: vect2d) -> f64 {
    let dx = geom.p2.value.x - geom.p1.value.x;
    let dy = geom.p2.value.y - geom.p1.value.y;
    let len2 = dx * dx + dy * dy;
    if len2 < 1e-24 {
        return 0.0;
    }
    (((p.x - geom.p1.value.x) * dx + (p.y - geom.p1.value.y) * dy) / len2).clamp(0.0, 1.0)
}

fn dim_kind_label(kind: &DimensionKind) -> &'static str {
    match kind {
        DimensionKind::LineLength(_) => "length",
        DimensionKind::ArcRadius(_) => "radius",
        DimensionKind::ArcRadiusB(_) => "radius_b",
        DimensionKind::ArcSweep(_) => "sweep",
        DimensionKind::ArcRotation(_) | DimensionKind::LineAngle(_) => "xangle",
        DimensionKind::Angle(..) => "angle",
        DimensionKind::HDistance(..) => "hdistance",
        DimensionKind::VDistance(..) => "vdistance",
        _ => "distance",
    }
}

fn transfer_dimensions(
    sketch: &mut Sketch,
    target: SplitTarget,
    geom: &TargetGeom,
    pm: &PieceMap,
    bounds: &[f64],
    out: &mut SplitOutcome,
) {
    let references = |kind: &DimensionKind| match target {
        SplitTarget::Line(t) => kind.references_line(t),
        SplitTarget::Arc(t) => kind.references_arc(t),
    };
    // Plan first (immutable), then apply.
    enum DimOp {
        Retarget { index: usize, new_kind: DimensionKind },
        Drop { index: usize, reason: &'static str },
    }
    let mut ops: Vec<DimOp> = Vec::new();
    for (i, dim) in sketch.dimensions.iter().enumerate() {
        if !references(&dim.kind) {
            continue;
        }
        match map_dim_kind(&dim.kind, target, geom, pm, bounds, sketch) {
            Ok(new_kind) => ops.push(DimOp::Retarget { index: i, new_kind }),
            Err(reason) => ops.push(DimOp::Drop { index: i, reason }),
        }
    }
    let mut dropped_indices: Vec<usize> = Vec::new();
    for op in &ops {
        match op {
            DimOp::Retarget { index, new_kind } => {
                let (old_kind, value, name, driving) = {
                    let d = &sketch.dimensions[*index];
                    (d.kind, d.value, d.name.clone(),
                     d.expr_str.is_none() && !d.derived && d.range.is_none())
                };
                if driving {
                    remove_numeric_dim_constraint(sketch, &old_kind);
                }
                sketch.dimensions[*index].kind = *new_kind;
                if driving && !push_numeric_dim_constraint(sketch, new_kind, &value) {
                    // Validation refused the retargeted constraint:
                    // the dimension has no working successor.
                    dropped_indices.push(*index);
                    out.dropped
                        .push(format!("{} ({} -- retarget refused)", name, dim_kind_label(&old_kind)));
                    continue;
                }
                out.moved.push(format!(
                    "{} -> {} {}",
                    name,
                    dim_kind_label(new_kind),
                    dim_subject(sketch, new_kind)
                ));
                // Whole-arc dims (radius, radius_b, rotation) hold
                // for every piece, like whole-role constraints: the
                // first piece keeps the original, every other kept
                // piece gets a copy (new did/name, same value /
                // expression / driven state).
                for piece in pm.kept().skip(1) {
                    let Some(kind) = whole_arc_dim_kind(new_kind, piece) else { break };
                    let copy_name = format!("d{}", sketch.next_dimension_id);
                    if driving && !push_numeric_dim_constraint(sketch, &kind, &value) {
                        out.dropped.push(format!(
                            "{} copy for {} ({} -- refused)",
                            name, dim_subject(sketch, &kind), dim_kind_label(&kind)));
                        continue;
                    }
                    sketch.next_dimension_id += 1;
                    let mut copy = sketch.dimensions[*index].clone();
                    copy.did = 0; // minted by assign_dimension_ids
                    copy.name = copy_name.clone();
                    copy.kind = kind;
                    sketch.dimensions.push(copy);
                    out.copied.push(format!(
                        "{} -> {} ({} {})",
                        name, copy_name, dim_kind_label(&kind), dim_subject(sketch, &kind)));
                }
            }
            DimOp::Drop { index, reason } => {
                let (old_kind, name, driving) = {
                    let d = &sketch.dimensions[*index];
                    (d.kind, d.name.clone(),
                     d.expr_str.is_none() && !d.derived && d.range.is_none())
                };
                if driving {
                    remove_numeric_dim_constraint(sketch, &old_kind);
                }
                dropped_indices.push(*index);
                out.dropped
                    .push(format!("{} ({} -- {})", name, dim_kind_label(&old_kind), reason));
            }
        }
    }
    dropped_indices.sort_unstable();
    for &i in dropped_indices.iter().rev() {
        sketch.dimensions.remove(i);
    }
}

/// The same whole-arc dimension kind on another piece, for the kinds
/// that hold for every piece of a split arc. None for every other
/// kind (or a line piece).
fn whole_arc_dim_kind(kind: &DimensionKind, piece: PieceRef) -> Option<DimensionKind> {
    let arc = piece.arc()?;
    match kind {
        DimensionKind::ArcRadius(_) => Some(DimensionKind::ArcRadius(arc)),
        DimensionKind::ArcRadiusB(_) => Some(DimensionKind::ArcRadiusB(arc)),
        DimensionKind::ArcRotation(_) => Some(DimensionKind::ArcRotation(arc)),
        _ => None,
    }
}

/// Short subject text for a retargeted dimension's report line.
fn dim_subject(sketch: &Sketch, kind: &DimensionKind) -> String {
    let ep = |e: &DimensionEndpoint| -> String {
        match e {
            DimensionEndpoint::Point(p) => sketch.point_display_name(*p),
            DimensionEndpoint::LineP1(l) => format!("{}.p1", sketch.lines[*l].name),
            DimensionEndpoint::LineP2(l) => format!("{}.p2", sketch.lines[*l].name),
            DimensionEndpoint::ArcCenter(a) => format!("{}.center", sketch.arcs[*a].name),
            DimensionEndpoint::ArcStart(a) => format!("{}.start", sketch.arcs[*a].name),
            DimensionEndpoint::ArcEnd(a) => format!("{}.end", sketch.arcs[*a].name),
        }
    };
    match kind {
        DimensionKind::LineLength(l) | DimensionKind::LineAngle(l) => sketch.lines[*l].name.clone(),
        DimensionKind::ArcRadius(a)
        | DimensionKind::ArcRadiusB(a)
        | DimensionKind::ArcSweep(a)
        | DimensionKind::ArcRotation(a) => sketch.arcs[*a].name.clone(),
        DimensionKind::Angle(a, b, _) => {
            format!("{} {}", sketch.lines[*a].name, sketch.lines[*b].name)
        }
        DimensionKind::PointPointDistance(a, b)
        | DimensionKind::HDistance(a, b)
        | DimensionKind::VDistance(a, b) => format!("{} {}", ep(a), ep(b)),
        DimensionKind::PointLineDistance(a, l) => {
            format!("{} {}", ep(a), sketch.lines[*l].name)
        }
        DimensionKind::ConcentricDistance(a, b) => {
            format!("{} {}", sketch.arcs[*a].name, sketch.arcs[*b].name)
        }
        DimensionKind::LineLineDistance(a, b) => {
            format!("{} {}", sketch.lines[*a].name, sketch.lines[*b].name)
        }
    }
}

// ---------------------------------------------------------------------------
// Constraint transfer
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
enum NewRef {
    L(Ref<Line>),
    A(Ref<Arc>),
}

impl NewRef {
    fn from_piece(p: PieceRef) -> NewRef {
        match p {
            PieceRef::Line(r) => NewRef::L(r),
            PieceRef::Arc(r) => NewRef::A(r),
        }
    }
}

struct COp {
    coll: &'static str,
    index: usize,
    /// (is_line_field, slot, new ref) rewrites applied in place.
    sets: Vec<(bool, usize, NewRef)>,
    /// Whole-role slots: one clone per kept piece beyond the first.
    rep_slots: Vec<(bool, usize)>,
    drop: bool,
    desc_before: String,
    nid: u32,
}

fn plan_constraint_ops(
    sketch: &Sketch,
    target: SplitTarget,
    geom: &TargetGeom,
    pm: &PieceMap,
    bounds: &[f64],
) -> Vec<COp> {
    let is_line = matches!(target, SplitTarget::Line(_));
    let mut ops: Vec<COp> = Vec::new();
    sketch.for_each_constraint_collection_ref(|arenas, meta, coll| {
        if meta.dimension_backed {
            // Owned by a dimension; the dimension transfer swaps them.
            return;
        }
        for i in 0..coll.len() {
            let c = coll.item(i);
            let references = match target {
                SplitTarget::Line(t) => c.references_line(t),
                SplitTarget::Arc(t) => c.references_arc(t),
            };
            if !references {
                continue;
            }
            let mut op = COp {
                coll: meta.name,
                index: i,
                sets: Vec::new(),
                rep_slots: Vec::new(),
                drop: false,
                desc_before: c.describe(sketch),
                nid: c.nid(),
            };
            // Anchor positions of the constraint's other referents,
            // for Host resolution.
            let mut anchors: Vec<vect2d> = Vec::new();
            c.each_point_ref(&mut |p| {
                if let Some(pt) = arenas.points.get(p) {
                    anchors.push(pt.pos.value);
                }
            });
            c.each_line_field(&mut |_, r, _| {
                if !matches!(target, SplitTarget::Line(t) if t == r)
                    && let Some(l) = arenas.lines.get(r)
                {
                    anchors.push(vect2d::new(
                        (l.p1.value.x + l.p2.value.x) * 0.5,
                        (l.p1.value.y + l.p2.value.y) * 0.5,
                    ));
                }
            });
            c.each_arc_field(&mut |_, r, _| {
                if !matches!(target, SplitTarget::Arc(t) if t == r)
                    && let Some(a) = arenas.arcs.get(r)
                {
                    anchors.push(a.center.value);
                }
            });
            // Contact param for tangency constraints: the point on the
            // target nearest the other curve's defining anchor.
            let contact_param = || -> Option<f64> {
                let mut other_anchor: Option<vect2d> = None;
                c.each_arc_field(&mut |_, r, _| {
                    if !matches!(target, SplitTarget::Arc(t) if t == r)
                        && let Some(a) = arenas.arcs.get(r)
                    {
                        other_anchor = Some(a.center.value);
                    }
                });
                if other_anchor.is_none() {
                    // Line-arc tangency with the target being the arc:
                    // the contact sits nearest the line's foot of the
                    // target's center.
                    c.each_line_field(&mut |_, r, _| {
                        if !matches!(target, SplitTarget::Line(t) if t == r)
                            && let Some(l) = arenas.lines.get(r)
                        {
                            other_anchor = Some(project_onto_segment(
                                geom.center.value,
                                l.p1.value,
                                l.p2.value,
                            ));
                        }
                    });
                }
                let anchor = other_anchor?;
                Some(if is_line {
                    line_param_of(geom, project_onto_segment(anchor, geom.p1.value, geom.p2.value))
                } else if let SplitTarget::Arc(t) = target {
                    nearest_arc_param(&arenas.arcs[t], anchor)
                } else {
                    unreachable!()
                })
            };
            let host_param = || -> f64 {
                if anchors.is_empty() {
                    return if is_line { 0.0 } else { geom.sa };
                }
                let n = anchors.len() as f64;
                let avg = vect2d::new(
                    anchors.iter().map(|p| p.x).sum::<f64>() / n,
                    anchors.iter().map(|p| p.y).sum::<f64>() / n,
                );
                if is_line {
                    line_param_of(geom, project_onto_segment(avg, geom.p1.value, geom.p2.value))
                } else if let SplitTarget::Arc(t) = target {
                    nearest_arc_param(&arenas.arcs[t], avg)
                } else {
                    unreachable!()
                }
            };
            // Decide per matching field.
            let decide = |slot: usize, field_is_line: bool, role: RefRole, op: &mut COp| {
                let piece: Option<PieceRef> = match role {
                    RefRole::Start => pm.start_owner(),
                    RefRole::End => pm.end_owner(),
                    RefRole::Center => pm.first_kept(),
                    RefRole::Host => {
                        let idx = piece_of_param(geom, is_line, bounds, host_param());
                        pm.at_or_nearest(idx)
                    }
                    RefRole::Whole => {
                        op.rep_slots.push((field_is_line, slot));
                        pm.first_kept()
                    }
                    RefRole::Contact => match contact_param() {
                        Some(t) => pm.at(piece_of_param(geom, is_line, bounds, t)),
                        None => pm.first_kept(),
                    },
                    RefRole::Extent => None,
                };
                match piece {
                    Some(p) => op.sets.push((field_is_line, slot, NewRef::from_piece(p))),
                    None => op.drop = true,
                }
            };
            match target {
                SplitTarget::Line(t) => {
                    let mut decisions: Vec<(usize, RefRole)> = Vec::new();
                    c.each_line_field(&mut |slot, r, role| {
                        if r == t {
                            decisions.push((slot, role));
                        }
                    });
                    for (slot, role) in decisions {
                        decide(slot, true, role, &mut op);
                    }
                }
                SplitTarget::Arc(t) => {
                    let mut decisions: Vec<(usize, RefRole)> = Vec::new();
                    c.each_arc_field(&mut |slot, r, role| {
                        if r == t {
                            decisions.push((slot, role));
                        }
                    });
                    for (slot, role) in decisions {
                        decide(slot, false, role, &mut op);
                    }
                }
            }
            if op.drop {
                op.sets.clear();
                op.rep_slots.clear();
            }
            ops.push(op);
        }
    });
    ops
}

fn apply_constraint_ops(sketch: &mut Sketch, ops: &[COp], pm_pieces: &[Option<PieceRef>]) {
    let extra_pieces: Vec<PieceRef> = pm_pieces.iter().flatten().copied().skip(1).collect();
    // Group ops per collection.
    let mut by_coll: HashMap<&'static str, Vec<&COp>> = HashMap::new();
    for op in ops {
        by_coll.entry(op.coll).or_default().push(op);
    }
    sketch.for_each_constraint_collection(|_, meta, coll| {
        let Some(coll_ops) = by_coll.get(meta.name) else {
            return;
        };
        let original_len = coll.len();
        // 1. In-place rewrites.
        for op in coll_ops {
            if op.drop {
                continue;
            }
            for &(field_is_line, slot, new_ref) in &op.sets {
                let item = coll.item_mut(op.index);
                match (field_is_line, new_ref) {
                    (true, NewRef::L(r)) => item.set_line_field(slot, r),
                    (false, NewRef::A(r)) => item.set_arc_field(slot, r),
                    // A line field cannot take an arc piece or vice
                    // versa: pieces are the same entity kind as the
                    // target, and the field held the target.
                    _ => unreachable!("piece kind mismatch in split transfer"),
                }
            }
        }
        // 2. Whole-role clones, one per kept piece beyond the first.
        for op in coll_ops {
            if op.drop || op.rep_slots.is_empty() {
                continue;
            }
            for piece in &extra_pieces {
                let copy = coll.clone_push_blank(op.index);
                for &(field_is_line, slot) in &op.rep_slots {
                    let item = coll.item_mut(copy);
                    match (field_is_line, NewRef::from_piece(*piece)) {
                        (true, NewRef::L(r)) => item.set_line_field(slot, r),
                        (false, NewRef::A(r)) => item.set_arc_field(slot, r),
                        _ => unreachable!("piece kind mismatch in split replication"),
                    }
                }
            }
        }
        // 3. Drops (original items only; clones sit past original_len).
        let to_drop: std::collections::HashSet<usize> =
            coll_ops.iter().filter(|op| op.drop).map(|op| op.index).collect();
        if !to_drop.is_empty() {
            let mut idx = 0;
            coll.retain_constraints(&mut |_| {
                let keep = idx >= original_len || !to_drop.contains(&idx);
                idx += 1;
                keep
            });
        }
    });
}

// ---------------------------------------------------------------------------
// Tangency fixups after retarget
// ---------------------------------------------------------------------------

/// Reset direction memory on retargeted tangents and re-detect
/// shared-endpoint flags from the (already re-pointed) coincidence
/// collections and current geometry. TangentLA's flags refresh on the
/// next solve; TangentAA's `shared` is re-detected here by proximity,
/// like `ApplyTangentAA` does at creation.
fn fixup_tangents(sketch: &mut Sketch, touched_la: &[u32], touched_aa: &[u32]) {
    for t in &mut sketch.tangent_la {
        if touched_la.contains(&t.nid) {
            t.dir_sign = f64::NAN;
        }
    }
    let snap = 1e-3;
    let near = |p: vect2d, q: vect2d| (p.x - q.x).abs() < snap && (p.y - q.y).abs() < snap;
    let mut updates: Vec<(usize, SharedEndpoint)> = Vec::new();
    for (i, t) in sketch.tangent_aa.iter().enumerate() {
        if !touched_aa.contains(&t.nid) {
            continue;
        }
        let (Some(arc_a), Some(arc_b)) = (sketch.arcs.get(t.a), sketch.arcs.get(t.b)) else {
            continue;
        };
        let a_sp = arc_a.start_pos();
        let a_ep = arc_a.end_pos();
        let b_sp = arc_b.start_pos();
        let b_ep = arc_b.end_pos();
        let shared = if near(a_sp, b_sp) {
            SharedEndpoint::StartStart
        } else if near(a_sp, b_ep) {
            SharedEndpoint::StartEnd
        } else if near(a_ep, b_sp) {
            SharedEndpoint::EndStart
        } else if near(a_ep, b_ep) {
            SharedEndpoint::EndEnd
        } else {
            SharedEndpoint::None
        };
        updates.push((i, shared));
    }
    for (i, shared) in updates {
        sketch.tangent_aa[i].shared = shared;
    }
    sketch.fixup_tangent_signs();
}

// ---------------------------------------------------------------------------
// Expression map
// ---------------------------------------------------------------------------

fn expression_map(
    sketch: &Sketch,
    geom: &TargetGeom,
    is_line: bool,
    pm: &PieceMap,
) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let name = |p: PieceRef| -> String {
        match p {
            PieceRef::Line(r) => sketch.lines[r].name.clone(),
            PieceRef::Arc(r) => sketch.arcs[r].name.clone(),
        }
    };
    let n = &geom.name;
    if let Some(first) = pm.first_kept() {
        let f = name(first);
        if is_line {
            map.insert(format!("{}.angle", n), format!("{}.angle", f));
        } else {
            for prop in ["center.x", "center.y", "radius", "radius_b", "rotation", "diameter"] {
                map.insert(format!("{}.{}", n, prop), format!("{}.{}", f, prop));
            }
        }
    }
    if let Some(start) = pm.start_owner() {
        let s = name(start);
        if is_line {
            map.insert(format!("{}.p1.x", n), format!("{}.p1.x", s));
            map.insert(format!("{}.p1.y", n), format!("{}.p1.y", s));
        } else {
            map.insert(format!("{}.start.x", n), format!("{}.start.x", s));
            map.insert(format!("{}.start.y", n), format!("{}.start.y", s));
            map.insert(format!("{}.start_angle", n), format!("{}.start_angle", s));
        }
    }
    if let Some(end) = pm.end_owner() {
        let e = name(end);
        if is_line {
            map.insert(format!("{}.p2.x", n), format!("{}.p2.x", e));
            map.insert(format!("{}.p2.y", n), format!("{}.p2.y", e));
        } else {
            map.insert(format!("{}.end.x", n), format!("{}.end.x", e));
            map.insert(format!("{}.end.y", n), format!("{}.end.y", e));
            map.insert(format!("{}.end_angle", n), format!("{}.end_angle", e));
        }
    }
    if pm.all_kept() {
        let sum = |prop: &str| -> String {
            let parts: Vec<String> = pm.kept().map(|p| format!("{}.{}", name(p), prop)).collect();
            format!("({})", parts.join(" + "))
        };
        if is_line {
            map.insert(format!("{}.length", n), sum("length"));
        } else {
            map.insert(format!("{}.sweep", n), sum("sweep"));
        }
    }
    map
}

// ---------------------------------------------------------------------------
// The engine
// ---------------------------------------------------------------------------

/// Apply a resolved split plan: create pieces, transfer references,
/// rewrite expressions, delete the target. Does not solve; callers run
/// `assign_constraint_names` + `solve` afterwards (the Action wrapper
/// does both).
pub fn apply_split(sketch: &mut Sketch, plan: &SplitPlan) -> Result<SplitOutcome, String> {
    let geom = capture_geom(sketch, plan.target)?;
    let is_line = matches!(plan.target, SplitTarget::Line(_));
    let n_pieces = piece_count(geom.closed && !is_line, plan.cuts.len());
    if plan.keep.len() != n_pieces {
        return Err(format!(
            "split: keep mask has {} entries for {} pieces",
            plan.keep.len(),
            n_pieces
        ));
    }
    if !plan.keep.iter().any(|&k| k) {
        return Err("split: no piece kept (use delete instead)".into());
    }
    if geom.closed && !is_line && plan.cuts.len() < 2 {
        return Err("split: a closed circle/ellipse needs at least two cuts".into());
    }
    if plan.cuts.is_empty() {
        return Err("split: no cuts".into());
    }

    let bounds = boundaries(&geom, is_line, &plan.cuts);
    let pre_next_nid = sketch.next_constraint_id;

    // 1. Pieces.
    let pieces = create_pieces(sketch, &geom, is_line, plan, &bounds);
    let pm = PieceMap { pieces: &pieces };

    let mut out = SplitOutcome {
        pieces: pieces.clone(),
        piece_names: pieces
            .iter()
            .map(|p| {
                p.map(|p| match p {
                    PieceRef::Line(r) => sketch.lines[r].name.clone(),
                    PieceRef::Arc(r) => sketch.arcs[r].name.clone(),
                })
            })
            .collect(),
        moved: Vec::new(),
        copied: Vec::new(),
        dropped: Vec::new(),
        expr_report: Vec::new(),
    };

    // Replicated H/V flags are copies of the original's CL<n>H/V.
    if is_line {
        for name in out.piece_names.iter().flatten() {
            if geom.horizontal {
                out.copied.push(format_flag_name(name, 'H'));
            }
            if geom.vertical {
                out.copied.push(format_flag_name(name, 'V'));
            }
        }
    }

    // 2. Dimensions (their backing constraints ride along).
    transfer_dimensions(sketch, plan.target, &geom, &pm, &bounds, &mut out);

    // 3. Constraints, role-driven.
    let ops = plan_constraint_ops(sketch, plan.target, &geom, &pm, &bounds);
    let mut touched_la: Vec<u32> = Vec::new();
    let mut touched_aa: Vec<u32> = Vec::new();
    let mut moved_nids: Vec<u32> = Vec::new();
    for op in &ops {
        if op.drop {
            out.dropped.push(format!("C{} {}", op.nid, op.desc_before));
        } else if !op.sets.is_empty() {
            moved_nids.push(op.nid);
            match op.coll {
                "tangent_la" => touched_la.push(op.nid),
                "tangent_aa" => touched_aa.push(op.nid),
                _ => {}
            }
        }
    }
    apply_constraint_ops(sketch, &ops, &pieces);
    fixup_tangents(sketch, &touched_la, &touched_aa);

    // 4. Expressions.
    let map = expression_map(sketch, &geom, is_line, &pm);
    out.expr_report = sketch.rewrite_expression_symbols(&map, &geom.name);

    // 5. Delete the target. Everything was re-pointed; the cascade in
    // delete_* only backstops what was deliberately dropped.
    match plan.target {
        SplitTarget::Line(r) => sketch.delete_line(r),
        SplitTarget::Arc(r) => sketch.delete_arc(r),
    }

    // 6. Mint ids, then describe moved and copied constraints with
    // their final names.
    sketch.assign_constraint_names();
    let mut moved_desc: Vec<(u32, String)> = Vec::new();
    let mut copied_desc: Vec<(u32, String)> = Vec::new();
    sketch.for_each_constraint_collection_ref(|_, meta, coll| {
        for i in 0..coll.len() {
            let c = coll.item(i);
            if moved_nids.contains(&c.nid()) {
                moved_desc.push((c.nid(), c.describe(sketch)));
            } else if !meta.dimension_backed && c.nid() >= pre_next_nid {
                copied_desc.push((c.nid(), c.describe(sketch)));
            }
        }
    });
    moved_desc.sort_by_key(|(nid, _)| *nid);
    copied_desc.sort_by_key(|(nid, _)| *nid);
    for (nid, d) in moved_desc {
        out.moved.push(format!("C{} {}", nid, d));
    }
    for (nid, d) in copied_desc {
        out.copied.push(format!("C{} {}", nid, d));
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Follow-up constraint actions (gated)
// ---------------------------------------------------------------------------

/// Build the gated follow-up actions for an applied split: cut-point
/// coincidences between adjacent kept pieces, endpoint pins onto the
/// cutters, and concentricity between adjacent arc pieces. Each is an
/// ordinary constraint action, so the redundancy gate decides it; a
/// rejection means "already implied" and is consumed by the caller.
pub fn post_split_actions(plan: &SplitPlan, pieces: &[Option<PieceRef>], pin: bool) -> Vec<Action> {
    let closed = pieces.len() == plan.cuts.len();
    let mut actions = Vec::new();
    let mut concentric_done: Vec<(usize, usize)> = Vec::new();
    for (ci, cut) in plan.cuts.iter().enumerate() {
        // Pieces meeting at this cut.
        let (left_idx, right_idx) = if closed {
            ((ci + pieces.len() - 1) % pieces.len(), ci)
        } else {
            (ci, ci + 1)
        };
        let left = pieces[left_idx];
        let right = pieces[right_idx];
        if left_idx == right_idx {
            continue; // single-piece wrap; nothing to join
        }
        if let (Some(l), Some(r)) = (left, right) {
            match (l, r) {
                (PieceRef::Line(a), PieceRef::Line(b)) => {
                    actions.push(Action::ApplyCoincidentLL21 { a, b });
                }
                (PieceRef::Arc(a), PieceRef::Arc(b)) => {
                    actions.push(Action::ApplyCoincidentArcEndStart { a, b });
                    let key = (left_idx.min(right_idx), left_idx.max(right_idx));
                    if !concentric_done.contains(&key) {
                        concentric_done.push(key);
                        actions.push(Action::ApplyConcentric { a, b });
                    }
                }
                _ => {}
            }
        }
        if pin && let Some(cutter) = cut.cutter {
            // Pin one surviving endpoint at the cut onto the cutter.
            let pinned: Option<Action> = match (left, right, cutter) {
                (Some(PieceRef::Line(a)), _, Cutter::Line(b)) => {
                    Some(Action::ApplyLineP2OnLine { a, b })
                }
                (Some(PieceRef::Line(line)), _, Cutter::Arc(arc)) => {
                    Some(Action::ApplyLineP2OnArc { line, arc })
                }
                (Some(PieceRef::Arc(a)), _, Cutter::Line(line)) => {
                    Some(Action::ApplyEndpointOnLine {
                        endpoint: DimensionEndpoint::ArcEnd(a),
                        line,
                    })
                }
                (Some(PieceRef::Arc(a)), _, Cutter::Arc(arc)) => {
                    Some(Action::ApplyEndpointOnArc {
                        endpoint: DimensionEndpoint::ArcEnd(a),
                        arc,
                    })
                }
                (None, Some(PieceRef::Line(a)), Cutter::Line(b)) => {
                    Some(Action::ApplyLineP1OnLine { a, b })
                }
                (None, Some(PieceRef::Line(line)), Cutter::Arc(arc)) => {
                    Some(Action::ApplyLineP1OnArc { line, arc })
                }
                (None, Some(PieceRef::Arc(a)), Cutter::Line(line)) => {
                    Some(Action::ApplyEndpointOnLine {
                        endpoint: DimensionEndpoint::ArcStart(a),
                        line,
                    })
                }
                (None, Some(PieceRef::Arc(a)), Cutter::Arc(arc)) => {
                    Some(Action::ApplyEndpointOnArc {
                        endpoint: DimensionEndpoint::ArcStart(a),
                        arc,
                    })
                }
                (None, None, _) => None,
            };
            if let Some(a) = pinned {
                actions.push(a);
            }
        }
    }
    actions
}

// ---------------------------------------------------------------------------
// Cut discovery
// ---------------------------------------------------------------------------

/// All intersections of `target` with the given cutters (or with every
/// other line/arc in the sketch when `cutters` is None), as sorted,
/// deduplicated cuts. Cuts closer than `min_length` to an open
/// target's endpoint or to a neighbouring cut are refused (degenerate
/// pieces) and reported in the second return.
pub fn find_cuts(
    sketch: &Sketch,
    target: SplitTarget,
    cutters: Option<&[Cutter]>,
) -> (Vec<SplitCut>, Vec<String>) {
    let mut raw: Vec<SplitCut> = Vec::new();
    let mut probe = |cutter: Cutter| {
        match (target, cutter) {
            (SplitTarget::Line(t), Cutter::Line(c)) => {
                if t == c {
                    return;
                }
                let lt = &sketch.lines[t];
                let lc = &sketch.lines[c];
                if let Some(h) = crate::geometry::intersect_segments(
                    lt.p1.value,
                    lt.p2.value,
                    lc.p1.value,
                    lc.p2.value,
                ) {
                    raw.push(SplitCut { param: h.t_a, pos: h.pos, cutter: Some(cutter) });
                }
            }
            (SplitTarget::Line(t), Cutter::Arc(c)) => {
                let lt = &sketch.lines[t];
                for h in crate::geometry::intersect_segment_arc(
                    lt.p1.value,
                    lt.p2.value,
                    &sketch.arcs[c],
                ) {
                    raw.push(SplitCut { param: h.t_a, pos: h.pos, cutter: Some(cutter) });
                }
            }
            (SplitTarget::Arc(t), Cutter::Line(c)) => {
                let lc = &sketch.lines[c];
                for h in crate::geometry::intersect_segment_arc(
                    lc.p1.value,
                    lc.p2.value,
                    &sketch.arcs[t],
                ) {
                    raw.push(SplitCut { param: h.t_b, pos: h.pos, cutter: Some(cutter) });
                }
            }
            (SplitTarget::Arc(t), Cutter::Arc(c)) => {
                if t == c {
                    return;
                }
                for h in crate::geometry::intersect_arcs(&sketch.arcs[t], &sketch.arcs[c]) {
                    raw.push(SplitCut { param: h.t_a, pos: h.pos, cutter: Some(cutter) });
                }
            }
        }
    };
    match cutters {
        Some(list) => {
            for c in list {
                probe(*c);
            }
        }
        None => {
            for r in sketch.lines.refs() {
                probe(Cutter::Line(r));
            }
            for r in sketch.arcs.refs() {
                if sketch.arcs[r].closed || !matches!(target, SplitTarget::Arc(t) if t == r) {
                    probe(Cutter::Arc(r));
                }
            }
        }
    }
    sort_and_dedup_cuts(sketch, target, raw)
}

/// Sort cuts along the target's direction, merge near-duplicates, and
/// refuse cuts that would create pieces shorter than `min_length`.
fn sort_and_dedup_cuts(
    sketch: &Sketch,
    target: SplitTarget,
    mut raw: Vec<SplitCut>,
) -> (Vec<SplitCut>, Vec<String>) {
    let geom = match capture_geom(sketch, target) {
        Ok(g) => g,
        Err(_) => return (Vec::new(), Vec::new()),
    };
    let is_line = matches!(target, SplitTarget::Line(_));
    // Minimum piece extent in parameter units.
    let min_len = sketch.min_length.max(1e-9);
    let param_tol = if is_line {
        let dx = geom.p2.value.x - geom.p1.value.x;
        let dy = geom.p2.value.y - geom.p1.value.y;
        let len = (dx * dx + dy * dy).sqrt();
        if len < 1e-12 { 1.0 } else { min_len / len }
    } else {
        min_len / geom.radius.value.max(geom.radius_b.value).max(1e-9)
    };
    raw.sort_by(|a, b| {
        s_of(&geom, is_line, a.param)
            .partial_cmp(&s_of(&geom, is_line, b.param))
            .unwrap()
    });
    let mut cuts: Vec<SplitCut> = Vec::new();
    let mut refused: Vec<String> = Vec::new();
    let span_end = if is_line {
        1.0
    } else if geom.closed {
        std::f64::consts::TAU
    } else {
        s_of(&geom, is_line, geom.ea)
    };
    for c in raw {
        let s = s_of(&geom, is_line, c.param);
        // Near an open target's endpoint.
        if !geom.closed && (s < param_tol || s > span_end - param_tol) {
            refused.push(format!(
                "cut at ({:.3},{:.3}) too close to an endpoint",
                c.pos.x, c.pos.y
            ));
            continue;
        }
        // Near the previous cut (including closed wrap-around against
        // the first cut).
        if let Some(prev) = cuts.last()
            && (s - s_of(&geom, is_line, prev.param)).abs() < param_tol
        {
            refused.push(format!(
                "cut at ({:.3},{:.3}) merged with a neighbouring cut",
                c.pos.x, c.pos.y
            ));
            continue;
        }
        cuts.push(c);
    }
    if geom.closed && cuts.len() >= 2 {
        let first_s = s_of(&geom, is_line, cuts[0].param);
        let last_s = s_of(&geom, is_line, cuts[cuts.len() - 1].param);
        if (first_s + std::f64::consts::TAU - last_s).abs() < param_tol {
            let c = cuts.pop().unwrap();
            refused.push(format!(
                "cut at ({:.3},{:.3}) merged with a neighbouring cut",
                c.pos.x, c.pos.y
            ));
        }
    }
    (cuts, refused)
}

/// Parameter of the position on the target nearest to `p`, plus the
/// distance to it: the click/coordinate resolution for `split`/`trim`.
pub fn target_param_near(sketch: &Sketch, target: SplitTarget, p: vect2d) -> (f64, f64) {
    match target {
        SplitTarget::Line(t) => {
            let l = &sketch.lines[t];
            let proj = project_onto_segment(p, l.p1.value, l.p2.value);
            let geom_t = {
                let dx = l.p2.value.x - l.p1.value.x;
                let dy = l.p2.value.y - l.p1.value.y;
                let len2 = dx * dx + dy * dy;
                if len2 < 1e-24 {
                    0.0
                } else {
                    (((proj.x - l.p1.value.x) * dx + (proj.y - l.p1.value.y) * dy) / len2)
                        .clamp(0.0, 1.0)
                }
            };
            let d = ((p.x - proj.x).powi(2) + (p.y - proj.y).powi(2)).sqrt();
            (geom_t, d)
        }
        SplitTarget::Arc(t) => {
            let arc = &sketch.arcs[t];
            let param = nearest_arc_param(arc, p);
            let q = arc.point_at(param);
            let d = ((p.x - q.x).powi(2) + (p.y - q.y).powi(2)).sqrt();
            (param, d)
        }
    }
}

/// Predict the names the kept pieces will get, in piece order,
/// without mutating anything: names come off the same monotonic
/// counters `add_line` / `add_arc` / `add_ellipse` use.
pub fn apply_split_target_names(sketch: &Sketch, plan: &SplitPlan) -> Vec<String> {
    let mut names = Vec::new();
    let mut n = match plan.target {
        SplitTarget::Line(_) => sketch.next_line_id,
        SplitTarget::Arc(_) => sketch.next_arc_id,
    };
    let prefix = match plan.target {
        SplitTarget::Line(_) => "L",
        SplitTarget::Arc(r) => {
            if sketch.arcs[r].is_ellipse { "EA" } else { "A" }
        }
    };
    for &k in &plan.keep {
        if k {
            names.push(format!("{}{}", prefix, n));
            n += 1;
        }
    }
    names
}

/// The piece index that contains the parameter `t` -- which span of
/// the target a click selects, given the cuts.
pub fn clicked_piece(sketch: &Sketch, target: SplitTarget, cuts: &[SplitCut], t: f64) -> usize {
    let geom = capture_geom(sketch, target).expect("target exists");
    let is_line = matches!(target, SplitTarget::Line(_));
    let bounds = boundaries(&geom, is_line, cuts);
    piece_of_param(&geom, is_line, &bounds, t)
}

/// The bracketing cuts around the piece parameter `t` falls in, given
/// ALL cuts of the target. Returns the subset of cuts a bracketing
/// plan will use plus the index (in the subset's piece numbering) of
/// the clicked piece.
pub fn bracket_cuts(
    sketch: &Sketch,
    target: SplitTarget,
    all_cuts: &[SplitCut],
    t: f64,
    closed: bool,
) -> (Vec<SplitCut>, usize) {
    let piece = clicked_piece(sketch, target, all_cuts, t);
    if closed {
        // Piece i of a closed target spans cut i .. cut i+1 (wrap).
        let k = all_cuts.len();
        if k == 2 {
            return (all_cuts.to_vec(), piece);
        }
        let a = piece;
        let b = (piece + 1) % k;
        let mut second = all_cuts[b].clone();
        if b < a {
            // Wrapped bracket: lift the second cut by a full turn so
            // the subset's parameters ascend from the first cut, as
            // the plan's boundary builder requires. Piece 0 of the
            // subset is then the clicked span, piece 1 the rest.
            second.param += std::f64::consts::TAU;
        }
        (vec![all_cuts[a].clone(), second], 0)
    } else {
        // Open target: piece i is bounded by cut i-1 on the left (if
        // any) and cut i on the right (if any).
        let mut cuts = Vec::new();
        let mut clicked = 0;
        if piece > 0 {
            cuts.push(all_cuts[piece - 1].clone());
            clicked = 1;
        }
        if piece < all_cuts.len() {
            cuts.push(all_cuts[piece].clone());
        }
        (cuts, clicked)
    }
}

/// Parameter range `[from, to]` of the span of the target that a
/// bracketing plan at `t` would isolate (the hover-preview span).
/// `None` when there are no usable cuts -- the whole entity is the
/// span. Arc ranges ascend along the target's own direction and may
/// exceed the [start, start+TAU) window for wrapped closed spans.
pub fn preview_span(
    sketch: &Sketch,
    target: SplitTarget,
    all_cuts: &[SplitCut],
    t: f64,
) -> Option<(f64, f64)> {
    let geom = capture_geom(sketch, target).ok()?;
    let is_line = matches!(target, SplitTarget::Line(_));
    if all_cuts.is_empty() || (geom.closed && !is_line && all_cuts.len() < 2) {
        return None;
    }
    let (cuts, clicked) = bracket_cuts(sketch, target, all_cuts, t, geom.closed && !is_line);
    let bounds = boundaries(&geom, is_line, &cuts);
    Some((bounds[clicked], bounds[clicked + 1]))
}

#[cfg(test)]
mod split_tests {
    use super::*;
    use arael::model::CrossBlock;

    fn v(x: f64, y: f64) -> vect2d { vect2d::new(x, y) }
    fn near_v(a: vect2d, b: vect2d, tol: f64) -> bool {
        (a.x - b.x).abs() < tol && (a.y - b.y).abs() < tol
    }

    /// L0 (0,0)-(10,0), a vertical cutter at x=4, a reference line L2.
    fn line_fixture() -> (Sketch, Ref<Line>, Ref<Line>, Ref<Line>) {
        let mut s = Sketch::new();
        let l0 = s.add_line(v(0.0, 0.0), v(10.0, 0.0));
        let l1 = s.add_line(v(4.0, -2.0), v(4.0, 2.0));
        let l2 = s.add_line(v(0.0, 5.0), v(0.0, 9.0));
        s.assign_constraint_names();
        (s, l0, l1, l2)
    }

    fn line_split_plan(s: &Sketch, l0: Ref<Line>, l1: Ref<Line>, keep: Vec<bool>) -> SplitPlan {
        let (cuts, refused) = find_cuts(s, SplitTarget::Line(l0), Some(&[Cutter::Line(l1)]));
        assert!(refused.is_empty(), "{:?}", refused);
        assert_eq!(cuts.len(), 1);
        SplitPlan { target: SplitTarget::Line(l0), cuts, keep }
    }

    #[test]
    fn test_line_split_geometry_and_identity() {
        let (mut s, l0, l1, _l2) = line_fixture();
        let plan = line_split_plan(&s, l0, l1, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        assert!(s.lines.get(l0).is_none(), "target deleted");
        let a = out.pieces[0].unwrap().line().unwrap();
        let b = out.pieces[1].unwrap().line().unwrap();
        assert!(near_v(s.lines[a].p1.value, v(0.0, 0.0), 1e-9));
        assert!(near_v(s.lines[a].p2.value, v(4.0, 0.0), 1e-9));
        assert!(near_v(s.lines[b].p1.value, v(4.0, 0.0), 1e-9));
        assert!(near_v(s.lines[b].p2.value, v(10.0, 0.0), 1e-9));
        // Fresh names, target's name retired.
        assert_ne!(s.lines[a].name, "L0");
        assert_ne!(s.lines[b].name, "L0");
    }

    #[test]
    fn test_whole_replicates_endpoint_follows() {
        let (mut s, l0, l1, l2) = line_fixture();
        // Whole-role: perpendicular to L2. Endpoint-role: L0.p2
        // coincident with L2.p1.
        s.perpendicular.push(Perpendicular {
            a: l0, b: l2, dir_sign: 1.0, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.coincident_ll21.push(CoincidentLL21 {
            a: l0, b: l2, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.assign_constraint_names();
        let plan = line_split_plan(&s, l0, l1, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        let a = out.pieces[0].unwrap().line().unwrap();
        let b = out.pieces[1].unwrap().line().unwrap();
        // Perpendicular replicated: one per piece, same partner.
        assert_eq!(s.perpendicular.len(), 2);
        assert!(s.perpendicular.iter().any(|c| c.a == a && c.b == l2));
        assert!(s.perpendicular.iter().any(|c| c.a == b && c.b == l2));
        // The p2 coincidence followed the piece owning p2.
        assert_eq!(s.coincident_ll21.len(), 1);
        assert_eq!(s.coincident_ll21[0].a, b);
        assert_eq!(out.copied.len(), 1, "one perpendicular copy: {:?}", out.copied);
        assert!(out.moved.iter().any(|m| m.contains("perpendicular")), "{:?}", out.moved);
    }

    #[test]
    fn test_trim_drops_endpoint_side_and_extent() {
        let (mut s, l0, l1, l2) = line_fixture();
        s.coincident_ll21.push(CoincidentLL21 {
            a: l0, b: l2, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.equal_length.push(EqualLength {
            a: l0, b: l2, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.assign_constraint_names();
        // Trim away the p2-side piece.
        let plan = line_split_plan(&s, l0, l1, vec![true, false]);
        let out = apply_split(&mut s, &plan).unwrap();
        assert!(out.pieces[1].is_none());
        // The p2 coincidence and the equal-length are both gone.
        assert!(s.coincident_ll21.is_empty());
        assert!(s.equal_length.is_empty());
        assert_eq!(out.dropped.len(), 2, "{:?}", out.dropped);
    }

    #[test]
    fn test_length_dim_becomes_distance_same_did() {
        let (mut s, l0, l1, _l2) = line_fixture();
        s.lines[l0].constraints.has_length = true;
        s.lines[l0].constraints.length = 10.0;
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::LineLength(l0), value: 10.0,
            offset: v(0.0, 1.0), text_along: 0.25, name: "d0".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.assign_constraint_names();
        let did = s.dimensions[0].did;
        assert_ne!(did, 0);
        let plan = line_split_plan(&s, l0, l1, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        let a = out.pieces[0].unwrap().line().unwrap();
        let b = out.pieces[1].unwrap().line().unwrap();
        assert_eq!(s.dimensions.len(), 1);
        let d = &s.dimensions[0];
        assert_eq!(d.did, did, "did survives");
        assert_eq!(d.name, "d0");
        assert_eq!(d.value, 10.0);
        assert_eq!(d.text_along, 0.25);
        assert_eq!(
            d.kind,
            DimensionKind::PointPointDistance(
                DimensionEndpoint::LineP1(a),
                DimensionEndpoint::LineP2(b)
            )
        );
        // Backing constraint swapped: line flag gone, distance_ll12 in.
        assert_eq!(s.distance_ll12.len(), 1);
        assert_eq!(s.distance_ll12[0].a, a);
        assert_eq!(s.distance_ll12[0].b, b);
        assert_eq!(s.distance_ll12[0].distance, 10.0);
        assert!(out.moved.iter().any(|m| m.starts_with("d0 ->")), "{:?}", out.moved);
    }

    #[test]
    fn test_length_dim_drops_when_end_trimmed() {
        let (mut s, l0, l1, _l2) = line_fixture();
        s.lines[l0].constraints.has_length = true;
        s.lines[l0].constraints.length = 10.0;
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::LineLength(l0), value: 10.0,
            offset: v(0.0, 1.0), text_along: 0.0, name: "d0".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.assign_constraint_names();
        let plan = line_split_plan(&s, l0, l1, vec![true, false]);
        let out = apply_split(&mut s, &plan).unwrap();
        assert!(s.dimensions.is_empty());
        assert!(out.dropped.iter().any(|d| d.starts_with("d0")), "{:?}", out.dropped);
    }

    #[test]
    fn test_expression_rewrite_and_broken() {
        let (mut s, l0, l1, _l2) = line_fixture();
        s.user_params.push(UserParam {
            name: "w".into(), expr_str: "L0.length / 2".into(), value: 5.0, broken: false,
        });
        s.user_params.push(UserParam {
            name: "x1".into(), expr_str: "L0.p1.x + 1".into(), value: 1.0, broken: false,
        });
        s.assign_constraint_names();
        // Full split: length becomes the sum.
        let plan = line_split_plan(&s, l0, l1, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        let a_name = out.piece_names[0].clone().unwrap();
        let b_name = out.piece_names[1].clone().unwrap();
        assert_eq!(
            s.user_params[0].expr_str,
            format!("({}.length + {}.length) / 2", a_name, b_name)
        );
        assert!(!s.user_params[0].broken);
        assert_eq!(s.user_params[1].expr_str, format!("{}.p1.x + 1", a_name));
    }

    #[test]
    fn test_expression_broken_on_trim() {
        let (mut s, l0, l1, _l2) = line_fixture();
        s.user_params.push(UserParam {
            name: "w".into(), expr_str: "L0.length / 2".into(), value: 5.0, broken: false,
        });
        s.assign_constraint_names();
        let plan = line_split_plan(&s, l0, l1, vec![true, false]);
        let out = apply_split(&mut s, &plan).unwrap();
        assert!(s.user_params[0].broken, "length sum impossible after trim");
        assert!(out.expr_report.iter().any(|l| l.contains("broken")), "{:?}", out.expr_report);
    }

    #[test]
    fn test_hv_flags_replicate() {
        let (mut s, l0, l1, _l2) = line_fixture();
        s.lines[l0].constraints.horizontal = true;
        s.lines[l0].constraints.h_dir_sign = 1.0;
        s.assign_constraint_names();
        let plan = line_split_plan(&s, l0, l1, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        for p in out.pieces.iter().flatten() {
            let r = p.line().unwrap();
            assert!(s.lines[r].constraints.horizontal);
            assert_eq!(s.lines[r].constraints.h_dir_sign, 1.0);
        }
        assert_eq!(out.copied.len(), 2, "two flag copies reported: {:?}", out.copied);
    }

    #[test]
    fn test_locked_endpoint_survives() {
        let (mut s, l0, l1, _l2) = line_fixture();
        s.lines[l0].p1 = Param::fixed(v(0.0, 0.0));
        s.assign_constraint_names();
        let plan = line_split_plan(&s, l0, l1, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        let a = out.pieces[0].unwrap().line().unwrap();
        assert!(!s.lines[a].p1.optimize, "locked p1 stays locked on piece 0");
        assert!(s.lines[a].p2.optimize);
    }

    #[test]
    fn test_circle_split_two_pieces() {
        let mut s = Sketch::new();
        let c = s.add_arc(v(0.0, 0.0), 2.0, 0.0, std::f64::consts::TAU, true);
        let cut_line = s.add_line(v(0.0, -3.0), v(0.0, 3.0));
        s.assign_constraint_names();
        let (cuts, refused) =
            find_cuts(&s, SplitTarget::Arc(c), Some(&[Cutter::Line(cut_line)]));
        assert!(refused.is_empty(), "{:?}", refused);
        assert_eq!(cuts.len(), 2, "vertical line cuts the circle twice");
        let plan = SplitPlan { target: SplitTarget::Arc(c), cuts, keep: vec![true, true] };
        let out = apply_split(&mut s, &plan).unwrap();
        assert!(s.arcs.get(c).is_none());
        let a = out.pieces[0].unwrap().arc().unwrap();
        let b = out.pieces[1].unwrap().arc().unwrap();
        for r in [a, b] {
            let arc = &s.arcs[r];
            assert!(!arc.closed, "pieces are open arcs");
            assert!(arc.start_angle.optimize && arc.end_angle.optimize,
                "piece angles are free params");
            assert_eq!(arc.radius.value, 2.0);
            assert!(near_v(arc.center.value, v(0.0, 0.0), 1e-9));
        }
        // The two pieces cover the circle: sweeps sum to TAU.
        let sweep_a = s.arcs[a].end_angle.value - s.arcs[a].start_angle.value;
        let sweep_b = s.arcs[b].end_angle.value - s.arcs[b].start_angle.value;
        assert!((sweep_a + sweep_b - std::f64::consts::TAU).abs() < 1e-9);
        // Piece boundaries sit on the cut line (x = 0).
        assert!(s.arcs[a].start_pos().x.abs() < 1e-9);
        assert!(s.arcs[a].end_pos().x.abs() < 1e-9);
    }

    #[test]
    fn test_arc_radius_dim_replicates_to_every_piece() {
        let mut s = Sketch::new();
        let c = s.add_arc(v(0.0, 0.0), 2.0, 0.0, std::f64::consts::TAU, true);
        let cut_line = s.add_line(v(0.0, -3.0), v(0.0, 3.0));
        s.arcs[c].constraints.has_target_radius = true;
        s.arcs[c].constraints.target_radius = 2.0;
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::ArcRadius(c), value: 2.0,
            offset: v(0.0, 1.0), text_along: 0.0, name: "d0".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.next_dimension_id = 1;
        s.assign_constraint_names();
        let did0 = s.dimensions[0].did;
        let (cuts, _) = find_cuts(&s, SplitTarget::Arc(c), Some(&[Cutter::Line(cut_line)]));
        let plan = SplitPlan { target: SplitTarget::Arc(c), cuts, keep: vec![true, true] };
        let out = apply_split(&mut s, &plan).unwrap();
        s.assign_constraint_names();
        let a = out.pieces[0].unwrap().arc().unwrap();
        let b = out.pieces[1].unwrap().arc().unwrap();
        // Original stays on the first piece with its did; the second
        // piece gets a copy under a new name.
        assert_eq!(s.dimensions.len(), 2);
        assert_eq!(s.dimensions[0].kind, DimensionKind::ArcRadius(a));
        assert_eq!(s.dimensions[0].did, did0);
        assert_eq!(s.dimensions[1].kind, DimensionKind::ArcRadius(b));
        assert_eq!(s.dimensions[1].name, "d1");
        assert_ne!(s.dimensions[1].did, did0);
        assert_eq!(s.dimensions[1].value, 2.0);
        assert!(s.arcs[a].constraints.has_target_radius, "backing flag on the first piece");
        assert!(s.arcs[b].constraints.has_target_radius, "backing flag on the copy's piece");
        assert!(out.copied.iter().any(|c| c.starts_with("d0 -> d1")), "copy reported: {:?}", out.copied);
    }

    #[test]
    fn test_ellipse_dims_replicate_to_every_piece() {
        // radius, radius_b and rotation each hold for both halves;
        // an expression dim copies as an expression.
        let mut s = Sketch::new();
        let e = s.add_ellipse(v(0.0, 0.0), 3.0, 1.0, 0.5, true);
        let cut_line = s.add_line(v(0.0, -5.0), v(0.0, 5.0));
        s.arcs[e].constraints.has_target_radius = true;
        s.arcs[e].constraints.target_radius = 3.0;
        s.arcs[e].constraints.has_target_rotation = true;
        s.arcs[e].constraints.target_rotation = 0.5;
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::ArcRadius(e), value: 3.0,
            offset: v(0.0, 1.0), text_along: 0.0, name: "d0".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.next_dimension_id = 1;
        s.add_expr_dimension(DimensionKind::ArcRadiusB(e), "1", v(0.0, 1.0), 0.0).unwrap(); // d1
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::ArcRotation(e), value: 0.5f64.to_degrees(),
            offset: v(0.0, 1.0), text_along: 0.0, name: "d2".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.next_dimension_id = 3;
        s.assign_constraint_names();
        let (cuts, _) = find_cuts(&s, SplitTarget::Arc(e), Some(&[Cutter::Line(cut_line)]));
        assert_eq!(cuts.len(), 2);
        let plan = SplitPlan { target: SplitTarget::Arc(e), cuts, keep: vec![true, true] };
        let out = apply_split(&mut s, &plan).unwrap();
        s.assign_constraint_names();
        let a = out.pieces[0].unwrap().arc().unwrap();
        let b = out.pieces[1].unwrap().arc().unwrap();
        assert_eq!(s.dimensions.len(), 6, "{:?}", s.dimensions.iter().map(|d| &d.name).collect::<Vec<_>>());
        for arc in [a, b] {
            assert!(s.dimensions.iter().any(|d| d.kind == DimensionKind::ArcRadius(arc)), "radius on each piece");
            assert!(s.dimensions.iter().any(|d| d.kind == DimensionKind::ArcRadiusB(arc)
                && d.expr_str.as_deref() == Some("1")), "radius_b expression on each piece");
            assert!(s.dimensions.iter().any(|d| d.kind == DimensionKind::ArcRotation(arc)), "rotation on each piece");
            assert!(s.arcs[arc].constraints.has_target_radius);
            assert!(s.arcs[arc].constraints.has_target_rotation);
        }
        assert_eq!(out.copied.len(), 3, "{:?}", out.copied);
        // Names are fresh and unique.
        let mut names: Vec<_> = s.dimensions.iter().map(|d| d.name.clone()).collect();
        names.sort();
        names.dedup();
        assert_eq!(names.len(), 6);
    }

    #[test]
    fn test_sweep_dim_drops() {
        let mut s = Sketch::new();
        let c = s.add_arc(v(0.0, 0.0), 2.0, 0.0, std::f64::consts::PI, false);
        let cut_line = s.add_line(v(0.0, -3.0), v(0.0, 3.0));
        s.arcs[c].constraints.has_target_sweep = true;
        s.arcs[c].constraints.target_sweep = std::f64::consts::PI;
        s.dimensions.push(Dimension {
            did: 0, kind: DimensionKind::ArcSweep(c), value: 180.0,
            offset: v(0.0, 1.0), text_along: 0.0, name: "d0".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.assign_constraint_names();
        let (cuts, _) = find_cuts(&s, SplitTarget::Arc(c), Some(&[Cutter::Line(cut_line)]));
        assert_eq!(cuts.len(), 1, "half circle crossed once by the vertical line");
        let plan = SplitPlan { target: SplitTarget::Arc(c), cuts, keep: vec![true, true] };
        let out = apply_split(&mut s, &plan).unwrap();
        assert!(s.dimensions.is_empty());
        assert!(out.dropped.iter().any(|d| d.contains("sweep")), "{:?}", out.dropped);
    }

    #[test]
    fn test_tangent_follows_contact_piece() {
        let mut s = Sketch::new();
        // L0 along x from (0,0) to (10,0); circle tangent from above at (7,0).
        let l0 = s.add_line(v(0.0, 0.0), v(10.0, 0.0));
        let cut = s.add_line(v(2.0, -1.0), v(2.0, 1.0));
        let arc = s.add_arc(v(7.0, 1.0), 1.0, 0.0, std::f64::consts::TAU, true);
        s.tangent_la.push(TangentLA {
            line: l0, arc, sign: 1.0,
            p1_arc_start: false, p1_arc_end: false,
            p2_arc_start: false, p2_arc_end: false,
            dir_sign: f64::NAN, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.assign_constraint_names();
        let plan = line_split_plan(&s, l0, cut, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        let b = out.pieces[1].unwrap().line().unwrap();
        // Contact at x=7 is on the second piece (2..10).
        assert_eq!(s.tangent_la.len(), 1);
        assert_eq!(s.tangent_la[0].line, b);
    }

    #[test]
    fn test_point_on_line_host_follows_nearest() {
        let (mut s, l0, l1, _l2) = line_fixture();
        let p = s.add_point(v(8.0, 0.0));
        s.point_on_line.push(PointOnLine {
            point: p, line: l0, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.assign_constraint_names();
        let plan = line_split_plan(&s, l0, l1, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        let b = out.pieces[1].unwrap().line().unwrap();
        assert_eq!(s.point_on_line.len(), 1);
        assert_eq!(s.point_on_line[0].line, b, "host follows the piece under the point");
    }

    #[test]
    fn test_helper_bridge_survives() {
        let (mut s, l0, l1, l2) = line_fixture();
        // A distance dim from L0.p1 to L2.p1 creates a helper bridged
        // to L0.p1 through resolve_dim_endpoint.
        let ep_a = DimensionEndpoint::LineP1(l0);
        let ep_b = DimensionEndpoint::LineP1(l2);
        let pa = crate::actions::resolve_dim_endpoint(&mut s, &ep_a);
        assert!(s.points[pa].helper);
        let pb = crate::actions::resolve_dim_endpoint(&mut s, &ep_b);
        s.distance_pp.push(DistancePP {
            a: pa, b: pb,
            distance: 5.0, nid: 0, cid: 0, hb: CrossBlock::new(),
        });
        s.dimensions.push(Dimension {
            did: 0,
            kind: DimensionKind::PointPointDistance(
                DimensionEndpoint::Point(pa), ep_b),
            value: 5.0, offset: v(0.0, 1.0), text_along: 0.0, name: "d0".into(),
            expr_str: None, broken: false, derived: false, range: None,
        });
        s.assign_constraint_names();
        let plan = line_split_plan(&s, l0, l1, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        let a = out.pieces[0].unwrap().line().unwrap();
        // Helper still present, its bridge re-pointed to the p1 piece
        // (the second lp1 entry is l2's own bridge, untouched).
        assert!(s.points.get(pa).is_some(), "helper survived the split");
        let bridge = s.coincident_lp1.iter().find(|c| c.point == pa).unwrap();
        assert_eq!(bridge.line, a);
        assert_eq!(s.dimensions.len(), 1);
        assert_eq!(s.distance_pp.len(), 1);
    }

    #[test]
    fn test_post_split_actions_shape() {
        let (mut s, l0, l1, _l2) = line_fixture();
        s.assign_constraint_names();
        let plan = line_split_plan(&s, l0, l1, vec![true, true]);
        let out = apply_split(&mut s, &plan).unwrap();
        let actions = post_split_actions(&plan, &out.pieces, true);
        assert_eq!(actions.len(), 2, "coincidence + pin");
        assert!(matches!(actions[0], Action::ApplyCoincidentLL21 { .. }));
        assert!(matches!(actions[1], Action::ApplyLineP2OnLine { .. }));
        // Trim: no coincidence, pin lands on the surviving side.
        let (mut s2, l0b, l1b, _) = line_fixture();
        s2.assign_constraint_names();
        let plan2 = line_split_plan(&s2, l0b, l1b, vec![false, true]);
        let out2 = apply_split(&mut s2, &plan2).unwrap();
        let actions2 = post_split_actions(&plan2, &out2.pieces, true);
        assert_eq!(actions2.len(), 1);
        assert!(matches!(actions2[0], Action::ApplyLineP1OnLine { .. }));
    }

    #[test]
    fn test_arc_split_post_actions_concentric() {
        let mut s = Sketch::new();
        let c = s.add_arc(v(0.0, 0.0), 2.0, 0.0, std::f64::consts::TAU, true);
        let cut_line = s.add_line(v(0.0, -3.0), v(0.0, 3.0));
        s.assign_constraint_names();
        let (cuts, _) = find_cuts(&s, SplitTarget::Arc(c), Some(&[Cutter::Line(cut_line)]));
        let plan = SplitPlan { target: SplitTarget::Arc(c), cuts, keep: vec![true, true] };
        let out = apply_split(&mut s, &plan).unwrap();
        let actions = post_split_actions(&plan, &out.pieces, true);
        let coincidences = actions.iter()
            .filter(|a| matches!(a, Action::ApplyCoincidentArcEndStart { .. })).count();
        let concentrics = actions.iter()
            .filter(|a| matches!(a, Action::ApplyConcentric { .. })).count();
        let pins = actions.iter()
            .filter(|a| matches!(a, Action::ApplyEndpointOnLine { .. })).count();
        assert_eq!(coincidences, 2, "both cut points joined");
        assert_eq!(concentrics, 1, "one concentric per piece pair");
        assert_eq!(pins, 2, "both cuts pinned to the cutter");
    }

    #[test]
    fn test_find_cuts_refuses_near_endpoint() {
        let mut s = Sketch::new();
        let l0 = s.add_line(v(0.0, 0.0), v(10.0, 0.0));
        // Cutter passing within min_length of L0's start.
        let l1 = s.add_line(v(0.00005, -1.0), v(0.00005, 1.0));
        s.assign_constraint_names();
        let (cuts, refused) = find_cuts(&s, SplitTarget::Line(l0), Some(&[Cutter::Line(l1)]));
        assert!(cuts.is_empty());
        assert_eq!(refused.len(), 1, "{:?}", refused);
    }
}
