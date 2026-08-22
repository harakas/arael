//! The offset engine (docs/dev/OFFSET.md): plan the result of offsetting
//! a sequence of lines and arcs, create it through an [`ActionRunner`]
//! together with the constraints and dimensions that hold it at the
//! distance, record the offset meta-constraint, keep that record honest
//! after every action, and edit it.
//!
//! Per source segment the result is a parallel line (`Parallel` +
//! `LineLineDistance`), a concentric arc (`Concentric` +
//! `ConcentricDistance`), or for an ellipse a concentric ellipse with
//! both semi-axes moved by the distance (`Concentric` + `ArcArcParallel`
//! + `ConcentricDistance`) -- an approximation, exact at the axis ends.
//! Consecutive results meet at sharp corners (extended / trimmed to their
//! intersection) or, where the sources are tangent, at the offset of the
//! source joint; with `round`, a convex corner is a fillet of the
//! distance (arc + coincidences + tangents + radius dimension). Tangent
//! joints and free ends are pinned with `on_normal` so the result has no
//! slide left and its joints are well conditioned.

use std::f64::consts::{PI, TAU};

use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::actions::Action;
use crate::chain::{self, Sequence};
use crate::corner_ops::ActionRunner;

/// What the user asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct OffsetParams {
    pub kind: OffsetKind,
    pub distance: MetaValue,
    pub distance2: Option<MetaValue>,
    /// +1: `distance` goes left of the chain direction, -1: right.
    pub side: f64,
    pub pinned: bool,
    /// Round the convex corners with an arc of the distance.
    pub round: bool,
    /// Close the ends of an open two-sided offset.
    pub caps: CapKind,
}

impl OffsetParams {
    /// The (sign, distance) of every side this kind produces, the
    /// `distance` side first.
    pub fn sides(&self) -> Vec<(f64, MetaValue)> {
        match self.kind {
            OffsetKind::OneSide => vec![(self.side, self.distance.clone())],
            OffsetKind::Symmetric => vec![(self.side, self.distance.clone()), (-self.side, self.distance.clone())],
            OffsetKind::TwoSides => vec![
                (self.side, self.distance.clone()),
                (-self.side, self.distance2.clone().unwrap_or_else(|| self.distance.clone())),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Source geometry, in chain direction
// ---------------------------------------------------------------------------

/// A source segment as the chain traverses it.
#[derive(Clone, Copy, Debug)]
enum SegGeom {
    /// From `a` to `b`.
    Line { a: vect2d, b: vect2d },
    /// From parameter `t0` to `t1`; `ccw_travel` when the parameter
    /// increases along the chain.
    Arc { center: vect2d, ra: f64, rb: f64, rot: f64, t0: f64, t1: f64, ccw_travel: bool, is_ellipse: bool, closed: bool },
}

#[derive(Clone, Copy, Debug)]
struct Seg {
    src: OffsetSource,
    geom: SegGeom,
}

fn seg_of(sketch: &Sketch, src: OffsetSource) -> Seg {
    let geom = match src.entity {
        OffsetEntity::Line(l) => {
            let l = &sketch.lines[l];
            let (a, b) = if src.reversed { (l.p2.value, l.p1.value) } else { (l.p1.value, l.p2.value) };
            SegGeom::Line { a, b }
        }
        OffsetEntity::Arc(r) => {
            let a = &sketch.arcs[r];
            // The end parameter continues from the start by the signed
            // sweep (the stored angles may wrap): positive for a ccw arc.
            let sa = a.start_angle.value;
            let ea = if a.closed {
                sa + TAU
            } else {
                let norm = |v: f64| v.rem_euclid(TAU);
                if a.ccw { sa + norm(a.end_angle.value - sa) } else { sa - norm(sa - a.end_angle.value) }
            };
            let (t0, t1) = if src.reversed { (ea, sa) } else { (sa, ea) };
            SegGeom::Arc {
                center: a.center.value,
                ra: a.radius.value,
                rb: if a.is_ellipse { a.radius_b.value } else { a.radius.value },
                rot: if a.is_ellipse { a.rotation.value } else { 0.0 },
                t0,
                t1,
                ccw_travel: a.ccw != src.reversed,
                is_ellipse: a.is_ellipse,
                closed: a.closed,
            }
        }
    };
    Seg { src, geom }
}

fn rot90(v: vect2d) -> vect2d {
    vect2d::new(-v.y, v.x)
}
fn unit(v: vect2d) -> vect2d {
    let n = (v.x * v.x + v.y * v.y).sqrt();
    if n < 1e-300 { v } else { vect2d::new(v.x / n, v.y / n) }
}
fn dot(a: vect2d, b: vect2d) -> f64 {
    a.x * b.x + a.y * b.y
}
fn cross(a: vect2d, b: vect2d) -> f64 {
    a.x * b.y - a.y * b.x
}
fn dist(a: vect2d, b: vect2d) -> f64 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

/// Ellipse / circle point at parameter `t`.
fn arc_point(center: vect2d, ra: f64, rb: f64, rot: f64, t: f64) -> vect2d {
    let (c, s) = (rot.cos(), rot.sin());
    let (x, y) = (ra * t.cos(), rb * t.sin());
    vect2d::new(center.x + x * c - y * s, center.y + x * s + y * c)
}

/// Unit tangent of travel at the arc's parameter `t`.
fn arc_travel_tangent(ra: f64, rb: f64, rot: f64, t: f64, ccw_travel: bool) -> vect2d {
    let (c, s) = (rot.cos(), rot.sin());
    let (tx, ty) = (-ra * t.sin(), rb * t.cos());
    let v = unit(vect2d::new(tx * c - ty * s, tx * s + ty * c));
    if ccw_travel { v } else { vect2d::new(-v.x, -v.y) }
}

/// Parameter of a point on (or near) the ellipse / circle.
fn arc_param_of(center: vect2d, ra: f64, rb: f64, rot: f64, p: vect2d) -> f64 {
    let (c, s) = (rot.cos(), rot.sin());
    let (dx, dy) = (p.x - center.x, p.y - center.y);
    let (xl, yl) = (c * dx + s * dy, -s * dx + c * dy);
    (yl / rb).atan2(xl / ra)
}

impl Seg {
    fn start(&self) -> vect2d {
        match self.geom {
            SegGeom::Line { a, .. } => a,
            SegGeom::Arc { center, ra, rb, rot, t0, .. } => arc_point(center, ra, rb, rot, t0),
        }
    }
    fn end(&self) -> vect2d {
        match self.geom {
            SegGeom::Line { b, .. } => b,
            SegGeom::Arc { center, ra, rb, rot, t1, .. } => arc_point(center, ra, rb, rot, t1),
        }
    }
    /// Unit tangent of travel at the start / end.
    fn tangent_at_start(&self) -> vect2d {
        match self.geom {
            SegGeom::Line { a, b } => unit(b - a),
            SegGeom::Arc { ra, rb, rot, t0, ccw_travel, .. } => arc_travel_tangent(ra, rb, rot, t0, ccw_travel),
        }
    }
    fn tangent_at_end(&self) -> vect2d {
        match self.geom {
            SegGeom::Line { a, b } => unit(b - a),
            SegGeom::Arc { ra, rb, rot, t1, ccw_travel, .. } => arc_travel_tangent(ra, rb, rot, t1, ccw_travel),
        }
    }
    fn is_ellipse(&self) -> bool {
        matches!(self.geom, SegGeom::Arc { is_ellipse: true, .. })
    }
    /// Left unit normal of travel at the start / end (the tangent rotated
    /// a quarter turn counter-clockwise).
    fn left_normal_at_start(&self) -> vect2d {
        rot90(self.tangent_at_start())
    }
    fn left_normal_at_end(&self) -> vect2d {
        rot90(self.tangent_at_end())
    }
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

/// One result entity's geometry, in the SOURCE entity's orientation
/// (p1 <-> p1, start <-> start), ready to create.
#[derive(Clone, Copy, Debug)]
pub enum ResultGeom {
    Line { p1: vect2d, p2: vect2d },
    /// `start`/`end` are parameter angles in the source's own direction
    /// (`end - start` has the source's sweep sign); `ccw` is the source's.
    Arc { center: vect2d, radius: f64, start: f64, end: f64, ccw: bool, closed: bool },
    Ellipse { center: vect2d, rx: f64, ry: f64, rotation: f64, start: f64, end: f64, ccw: bool, closed: bool },
}

/// A round corner: an arc of the distance around the source joint,
/// from the earlier result's end to the later result's start, through
/// `mid`.
#[derive(Clone, Copy, Debug)]
pub struct RoundCorner {
    pub start: vect2d,
    pub end: vect2d,
    pub mid: vect2d,
}

/// A joint between consecutive results.
#[derive(Clone, Copy, Debug)]
pub struct JointPlan {
    /// The sources are tangent there: pin the earlier result's end on the
    /// source's normal, then coincide the later result's start with it.
    pub tangent: bool,
    /// The closing joint of a loop whose joints are all tangent: both
    /// ends are pinned, nothing is coincided (a loop of coincidences is
    /// one equation redundant; the ends meet by geometry).
    pub closure: bool,
    /// Where the results meet (a sharp or tangent joint); for a round
    /// corner the earlier result's end.
    pub point: vect2d,
    /// The later result's start: the same as `point` except at a round
    /// corner.
    pub next_start: vect2d,
    /// A convex corner rounded with an arc.
    pub round: Option<RoundCorner>,
}

#[derive(Clone, Debug)]
pub struct SidePlan {
    pub sign: f64,
    pub distance: MetaValue,
    /// The source index of every result, in chain order: every segment
    /// but the dropped ones.
    pub sources: Vec<usize>,
    /// Segments whose offset vanishes on this side (an arc offset inward
    /// past its radius): no result, the neighbours meet directly.
    pub dropped: Vec<usize>,
    /// One per entry of `sources`.
    pub results: Vec<ResultGeom>,
    /// Per result: whether it carries a distance dimension. The distance
    /// carries through a tangent joint (its coincidence fixes the next
    /// result's offset / radius), so only the first result of each run of
    /// tangent joints has one; a result after a sharp corner has its own.
    pub dims: Vec<bool>,
    /// `joints[i]` joins result i and i+1; a closed sequence has one more,
    /// joining the last and the first.
    pub joints: Vec<JointPlan>,
}

/// A cap across one end of an open two-sided offset, from the
/// `distance` side's end to the other side's.
#[derive(Clone, Copy, Debug)]
pub enum CapGeom {
    Line { a: vect2d, b: vect2d },
    /// A half circle around the source end, bulging away from the chain.
    Arc { start: vect2d, end: vect2d, mid: vect2d },
}

#[derive(Clone, Copy, Debug)]
pub struct CapPlan {
    /// At the sequence's end (true) or start (false).
    pub at_end: bool,
    pub geom: CapGeom,
}

#[derive(Clone, Debug)]
pub struct OffsetPlan {
    pub seq: Sequence,
    pub params: OffsetParams,
    pub sides: Vec<SidePlan>,
    /// The start cap, then the end cap; empty without caps.
    pub caps: Vec<CapPlan>,
    /// An elliptic segment is in the sequence: the result is approximate.
    pub approximate: bool,
}

/// The offset curve of a segment on one side, as an unbounded curve.
#[derive(Clone, Copy, Debug)]
enum Curve {
    Line { p: vect2d, dir: vect2d },
    Circle { center: vect2d, r: f64 },
    Ellipse { center: vect2d, ra: f64, rb: f64, rot: f64 },
}

/// The offset curve (unbounded) of `seg` at signed distance `s * d`,
/// and the radial change applied to an arc (`None` for a line).
fn offset_curve(seg: &Seg, s: f64, d: f64) -> (Curve, Option<f64>) {
    match seg.geom {
        SegGeom::Line { a, b } => {
            let n = rot90(unit(b - a));
            (Curve::Line { p: a + n * (s * d), dir: unit(b - a) }, None)
        }
        SegGeom::Arc { center, ra, rb, rot, ccw_travel, is_ellipse, .. } => {
            // Travelling counter-clockwise, left is toward the center.
            let dr = if ccw_travel { -s * d } else { s * d };
            if is_ellipse {
                (Curve::Ellipse { center, ra: ra + dr, rb: rb + dr, rot }, Some(dr))
            } else {
                (Curve::Circle { center, r: ra + dr }, Some(dr))
            }
        }
    }
}

/// Signed "inside/outside" function of a curve, zero on it.
fn implicit(c: &Curve, p: vect2d) -> f64 {
    match *c {
        Curve::Line { p: q, dir } => cross(dir, p - q),
        Curve::Circle { center, r } => dist(p, center) - r,
        Curve::Ellipse { center, ra, rb, rot } => {
            let (cs, sn) = (rot.cos(), rot.sin());
            let (dx, dy) = (p.x - center.x, p.y - center.y);
            let xl = (cs * dx + sn * dy) / ra;
            let yl = (-sn * dx + cs * dy) / rb;
            xl * xl + yl * yl - 1.0
        }
    }
}

/// Every intersection of two unbounded offset curves.
fn intersections(a: &Curve, b: &Curve) -> Vec<vect2d> {
    match (a, b) {
        (Curve::Line { p: p1, dir: d1 }, Curve::Line { p: p2, dir: d2 }) => {
            let den = cross(*d1, *d2);
            if den.abs() < 1e-12 { return Vec::new(); }
            let t = cross(*p2 - *p1, *d2) / den;
            vec![*p1 + *d1 * t]
        }
        (Curve::Line { p, dir }, Curve::Circle { center, r }) | (Curve::Circle { center, r }, Curve::Line { p, dir }) => {
            // Foot of the center on the line, then the chord half-length.
            let t = dot(*center - *p, *dir);
            let foot = *p + *dir * t;
            let h2 = r * r - dist(foot, *center).powi(2);
            if h2 < -1e-12 * r.max(1.0) { return Vec::new(); }
            let h = h2.max(0.0).sqrt();
            if h < 1e-12 { vec![foot] } else { vec![foot + *dir * h, foot - *dir * h] }
        }
        (Curve::Circle { center: c1, r: r1 }, Curve::Circle { center: c2, r: r2 }) => {
            let d = dist(*c1, *c2);
            if d < 1e-12 { return Vec::new(); }
            let tol = 1e-9 * (r1 + r2 + d);
            if d > r1 + r2 + tol || d < (r1 - r2).abs() - tol { return Vec::new(); }
            let along = (r1 * r1 - r2 * r2 + d * d) / (2.0 * d);
            let h2 = r1 * r1 - along * along;
            let h = if h2 > 0.0 { h2.sqrt() } else { 0.0 };
            let u = unit(*c2 - *c1);
            let base = *c1 + u * along;
            if h < tol { vec![base] } else { vec![base + rot90(u) * h, base - rot90(u) * h] }
        }
        // Anything with an ellipse: march the ellipse, bisect the other
        // curve's implicit function across sign changes.
        (Curve::Ellipse { center, ra, rb, rot }, other) | (other, Curve::Ellipse { center, ra, rb, rot }) => {
            let n = 720;
            let mut out = Vec::new();
            let at = |t: f64| arc_point(*center, *ra, *rb, *rot, t);
            let mut prev_t = 0.0;
            let mut prev_g = implicit(other, at(prev_t));
            for i in 1..=n {
                let t = TAU * i as f64 / n as f64;
                let g = implicit(other, at(t));
                if prev_g == 0.0 {
                    out.push(at(prev_t));
                } else if prev_g * g < 0.0 {
                    let (mut lo, mut hi, mut g_lo) = (prev_t, t, prev_g);
                    for _ in 0..60 {
                        let mid = 0.5 * (lo + hi);
                        let gm = implicit(other, at(mid));
                        if g_lo * gm <= 0.0 { hi = mid; } else { lo = mid; g_lo = gm; }
                    }
                    out.push(at(0.5 * (lo + hi)));
                }
                prev_t = t;
                prev_g = g;
            }
            out
        }
    }
}

/// Whether two consecutive segments meet tangentially at their joint:
/// a tangent constraint between them, or tangents of travel that agree to
/// a microradian.
fn joint_is_tangent(sketch: &Sketch, a: &Seg, b: &Seg) -> bool {
    let ta = a.tangent_at_end();
    let tb = b.tangent_at_start();
    if dot(ta, tb) > 0.0 && cross(ta, tb).abs() < 1e-6 {
        return true;
    }
    match (a.src.entity, b.src.entity) {
        (OffsetEntity::Line(l), OffsetEntity::Arc(r)) | (OffsetEntity::Arc(r), OffsetEntity::Line(l)) => {
            sketch.tangent_la.iter().any(|c| c.line == l && c.arc == r)
        }
        (OffsetEntity::Arc(x), OffsetEntity::Arc(y)) => {
            sketch.tangent_aa.iter().any(|c| (c.a == x && c.b == y) || (c.a == y && c.b == x))
        }
        _ => false,
    }
}

/// Plan the offset of `seq` under `params`: every result's geometry and
/// every joint, or why it cannot be done.
pub fn plan(sketch: &Sketch, seq: &Sequence, params: &OffsetParams) -> Result<OffsetPlan, String> {
    if seq.segs.is_empty() {
        return Err("nothing to offset".into());
    }
    for (sign, d) in params.sides() {
        if !(d.value > 0.0) || !d.value.is_finite() {
            return Err(format!("offset distance must be positive, got {}", d.value));
        }
        let _ = sign;
    }
    let segs: Vec<Seg> = seq.segs.iter().map(|s| seg_of(sketch, *s)).collect();
    let n = segs.len();
    let approximate = segs.iter().any(|s| s.is_ellipse());
    let min_len = sketch.min_length.max(1e-9);
    if params.caps != CapKind::None {
        if seq.closed {
            return Err("caps: a closed sequence has no ends".into());
        }
        if params.kind == OffsetKind::OneSide {
            return Err("caps need both sides (symmetric or two distances)".into());
        }
        if params.caps == CapKind::Round && params.kind != OffsetKind::Symmetric {
            return Err("round caps need a symmetric offset".into());
        }
    }

    // Joints of the source: tangent or corner. A tangent joint at an
    // elliptic segment has no consistent result (the approximate ellipse
    // misses the neighbour's offset endpoint), so it is refused. A closed
    // single entity (circle, ellipse) has no joint at all.
    let joint_count = match (seq.closed, n) {
        (true, 1) => 0,
        (true, n) => n,
        (false, n) => n - 1,
    };
    let mut tangent_joint = Vec::with_capacity(joint_count);
    for i in 0..joint_count {
        let (a, b) = (&segs[i], &segs[(i + 1) % n]);
        let t = joint_is_tangent(sketch, a, b);
        if t && (a.is_ellipse() || b.is_ellipse()) {
            return Err(format!(
                "{} and {} meet tangentially; an elliptic arc's offset is approximate and cannot keep a tangent joint",
                chain::entity_name(sketch, a.src.entity),
                chain::entity_name(sketch, b.src.entity)
            ));
        }
        // A reversal (the chain doubles back on itself) has no offset corner.
        if !t && dot(a.tangent_at_end(), b.tangent_at_start()) < -1.0 + 1e-9 {
            return Err(format!(
                "{} and {} double back on each other; no offset corner exists there",
                chain::entity_name(sketch, a.src.entity),
                chain::entity_name(sketch, b.src.entity)
            ));
        }
        tangent_joint.push(t);
    }

    let mut sides = Vec::new();
    for (sign, d) in params.sides() {
        let dv = d.value;
        // The unbounded offset curves of the surviving segments, with the
        // radial changes. An arc offset inward past its radius has no
        // offset: it is dropped and its neighbours meet directly.
        let mut sources: Vec<usize> = Vec::with_capacity(n);
        let mut dropped: Vec<usize> = Vec::new();
        let mut curves = Vec::with_capacity(n);
        let mut vanished_radius = 0.0;
        for (i, seg) in segs.iter().enumerate() {
            let (c, dr) = offset_curve(seg, sign, dv);
            if let (Some(dr), SegGeom::Arc { ra, rb, is_ellipse, .. }) = (dr, seg.geom) {
                let rmin = if is_ellipse { ra.min(rb) } else { ra };
                if rmin + dr <= min_len {
                    dropped.push(i);
                    vanished_radius = rmin;
                    continue;
                }
            }
            sources.push(i);
            curves.push(c);
        }
        let m = sources.len();
        // Nothing left, or a loop that cannot close on one segment.
        if m == 0 || (seq.closed && m == 1 && n > 1) {
            let first = dropped[0];
            return Err(format!(
                "{} (radius {:.4}) cannot be offset inward by {:.4}: nothing remains",
                chain::entity_name(sketch, segs[first].src.entity),
                vanished_radius,
                dv
            ));
        }
        let joint_count = match (seq.closed, m) {
            (true, 1) => 0,
            (true, m) => m,
            (false, m) => m - 1,
        };
        // Joint points between consecutive results: the offset of the
        // source joint where the sources are adjacent and tangent; for
        // corners the nearest intersection of the two result curves, or
        // with `round` on the convex side an arc of the distance around
        // the source joint from the one result's end to the other's start.
        let mut joints = Vec::with_capacity(joint_count);
        for k in 0..joint_count {
            let (sa, sb) = (sources[k], sources[(k + 1) % m]);
            let (a, b) = (&segs[sa], &segs[sb]);
            let adjacent = sb == (sa + 1) % n;
            let j = (a.end() + b.start()) * 0.5;
            if adjacent && tangent_joint[sa] {
                let p = j + a.left_normal_at_end() * (sign * dv);
                joints.push(JointPlan { tangent: true, closure: false, point: p, next_start: p, round: None });
                continue;
            }
            if !adjacent && dot(a.tangent_at_end(), b.tangent_at_start()) < -1.0 + 1e-9 {
                return Err(format!(
                    "{} and {} double back on each other once {} is gone; no offset corner exists there",
                    chain::entity_name(sketch, a.src.entity),
                    chain::entity_name(sketch, b.src.entity),
                    chain::entity_name(sketch, segs[(sa + 1) % n].src.entity)
                ));
            }
            // Convex on this side: the chain turns away from it.
            let turn = cross(a.tangent_at_end(), b.tangent_at_start());
            if params.round && sign * turn < 0.0 {
                let p1 = j + a.left_normal_at_end() * (sign * dv);
                let p2 = j + b.left_normal_at_start() * (sign * dv);
                let mid = j + unit((p1 - j) + (p2 - j)) * dv;
                joints.push(JointPlan {
                    tangent: false,
                    closure: false,
                    point: p1,
                    next_start: p2,
                    round: Some(RoundCorner { start: p1, end: p2, mid }),
                });
                continue;
            }
            let cands = intersections(&curves[k], &curves[(k + 1) % m]);
            let best = cands
                .into_iter()
                .min_by(|p, q| dist(*p, j).partial_cmp(&dist(*q, j)).unwrap());
            match best {
                Some(p) => joints.push(JointPlan { tangent: false, closure: false, point: p, next_start: p, round: None }),
                None => {
                    return Err(format!(
                        "the offsets of {} and {} do not meet at distance {:.4}",
                        chain::entity_name(sketch, a.src.entity),
                        chain::entity_name(sketch, b.src.entity),
                        dv
                    ))
                }
            }
        }
        // Result geometry per surviving segment: ends from the joints, free
        // ends at the source ends' offsets.
        let mut results = Vec::with_capacity(m);
        for (k, &si) in sources.iter().enumerate() {
            let seg = &segs[si];
            // A closed single entity has no joints and no ends to place.
            let lone_closed = seq.closed && m == 1;
            let start_pt = if lone_closed {
                seg.start()
            } else if k == 0 && !seq.closed {
                seg.start() + seg.left_normal_at_start() * (sign * dv)
            } else {
                joints[(k + m - 1) % m].next_start
            };
            let end_pt = if lone_closed {
                seg.end()
            } else if k + 1 == m && !seq.closed {
                seg.end() + seg.left_normal_at_end() * (sign * dv)
            } else {
                joints[k].point
            };
            let name = || chain::entity_name(sketch, seg.src.entity);
            let geom = match (seg.geom, curves[k]) {
                (SegGeom::Line { a, b }, _) => {
                    // Gone, or turned around by its neighbours' corners
                    // (an inner offset past the segment's reach).
                    if dist(start_pt, end_pt) <= min_len || dot(end_pt - start_pt, b - a) <= 0.0 {
                        return Err(format!("{} collapses at distance {:.4}", name(), dv));
                    }
                    let (p1, p2) = if seg.src.reversed { (end_pt, start_pt) } else { (start_pt, end_pt) };
                    ResultGeom::Line { p1, p2 }
                }
                (SegGeom::Arc { t0, t1, ccw_travel, closed, is_ellipse, .. }, curve) => {
                    let (center, ra2, rb2, rot) = match curve {
                        Curve::Circle { center, r } => (center, r, r, 0.0),
                        Curve::Ellipse { center, ra, rb, rot } => (center, ra, rb, rot),
                        Curve::Line { .. } => unreachable!("an arc's offset curve is never a line"),
                    };
                    if closed {
                        if is_ellipse {
                            ResultGeom::Ellipse { center, rx: ra2, ry: rb2, rotation: rot, start: 0.0, end: TAU, ccw: true, closed: true }
                        } else {
                            ResultGeom::Arc { center, radius: ra2, start: 0.0, end: TAU, ccw: true, closed: true }
                        }
                    } else {
                        // Parameters of the new ends; keep the sweep on the
                        // travel direction's side of the start.
                        let (u0, mut u1) = (
                            arc_param_of(center, ra2, rb2, rot, start_pt),
                            arc_param_of(center, ra2, rb2, rot, end_pt),
                        );
                        // Free / tangent ends keep the source parameter
                        // exactly (their offset point has that parameter).
                        let src_sweep = t1 - t0;
                        if ccw_travel {
                            while u1 <= u0 { u1 += TAU; }
                        } else {
                            while u1 >= u0 { u1 -= TAU; }
                        }
                        // Sharp inner corners can trim the arc past its far
                        // end: then the new sweep turns over or vanishes.
                        let new_sweep = u1 - u0;
                        if new_sweep.abs() <= 1e-9 || new_sweep.abs() > src_sweep.abs() + PI {
                            return Err(format!("{} collapses at distance {:.4}", name(), dv));
                        }
                        // Nearly full circles: the wrap above can add a turn
                        // to a joint that landed a hair past the start.
                        if (new_sweep.abs() - src_sweep.abs()).abs() > PI {
                            if ccw_travel { u1 -= TAU; } else { u1 += TAU; }
                        }
                        if (u1 - u0).abs() <= 1e-9 {
                            return Err(format!("{} collapses at distance {:.4}", name(), dv));
                        }
                        // Source orientation: start <-> start.
                        let (start, end) = if seg.src.reversed { (u1, u0) } else { (u0, u1) };
                        let ccw = ccw_travel != seg.src.reversed;
                        if is_ellipse {
                            ResultGeom::Ellipse { center, rx: ra2, ry: rb2, rotation: rot, start, end, ccw, closed: false }
                        } else {
                            ResultGeom::Arc { center, radius: ra2, start, end, ccw, closed: false }
                        }
                    }
                }
            };
            results.push(geom);
        }
        // Dimensions: the first result of each run of tangent joints (the
        // first of an open sequence, every result after a sharp corner).
        // A loop whose joints are all tangent has one run and no start:
        // the first result takes the dimension and the closing joint is
        // made by pins, not a coincidence (a loop of coincidences is one
        // equation redundant).
        let mut dims: Vec<bool> = (0..m)
            .map(|k| {
                if seq.closed {
                    joint_count > 0 && !joints[(k + m - 1) % m].tangent
                } else {
                    k == 0 || !joints[k - 1].tangent
                }
            })
            .collect();
        if !dims.iter().any(|&d| d) {
            dims[0] = true;
            if joint_count > 0 {
                joints[m - 1].closure = true;
            }
        }
        sides.push(SidePlan { sign, distance: d, sources, dropped, results, dims, joints });
    }
    // Caps: from the `distance` side's free end to the other side's. A
    // round cap is a half circle around the source end, which needs the
    // same end segment on both sides.
    let mut caps = Vec::new();
    if params.caps != CapKind::None && sides.len() == 2 {
        let (sa, da) = (sides[0].sign, sides[0].distance.value);
        let (sb, db) = (sides[1].sign, sides[1].distance.value);
        for at_end in [false, true] {
            let end_source = |side: &SidePlan| if at_end { *side.sources.last().unwrap() } else { side.sources[0] };
            let free_end = |side: &SidePlan, sign: f64, dv: f64| {
                let seg = &segs[end_source(side)];
                if at_end {
                    seg.end() + seg.left_normal_at_end() * (sign * dv)
                } else {
                    seg.start() + seg.left_normal_at_start() * (sign * dv)
                }
            };
            let a = free_end(&sides[0], sa, da);
            let b = free_end(&sides[1], sb, db);
            let geom = match params.caps {
                CapKind::Line => CapGeom::Line { a, b },
                CapKind::Round => {
                    let (ea, eb) = (end_source(&sides[0]), end_source(&sides[1]));
                    if ea != eb {
                        let gone = if at_end { ea.max(eb) } else { ea.min(eb) };
                        return Err(format!(
                            "round caps need the same end segment on both sides; {} vanishes on one side",
                            chain::entity_name(sketch, segs[gone].src.entity)
                        ));
                    }
                    let seg = &segs[ea];
                    let (p, t) = if at_end { (seg.end(), seg.tangent_at_end()) } else { (seg.start(), seg.tangent_at_start()) };
                    let away = if at_end { t } else { vect2d::new(-t.x, -t.y) };
                    CapGeom::Arc { start: a, end: b, mid: p + away * da }
                }
                CapKind::None => unreachable!(),
            };
            caps.push(CapPlan { at_end, geom });
        }
    }
    Ok(OffsetPlan { seq: seq.clone(), params: params.clone(), sides, caps, approximate })
}

// ---------------------------------------------------------------------------
// Applying a plan
// ---------------------------------------------------------------------------

/// What an offset created, for the command output.
#[derive(Clone, Debug, Default)]
pub struct OffsetOutcome {
    pub name: String,
    pub mid: u32,
    /// Per side: result entity names.
    pub entities: Vec<Vec<String>>,
    pub constraints: Vec<String>,
    pub dims: Vec<String>,
    /// Source segments whose offset vanished on some side (names, one
    /// entry per side it vanished on).
    pub dropped: Vec<String>,
    pub approximate: bool,
}

fn endpoint_of(e: OffsetEntity, is_end: bool) -> DimensionEndpoint {
    match (e, is_end) {
        (OffsetEntity::Line(l), false) => DimensionEndpoint::LineP1(l),
        (OffsetEntity::Line(l), true) => DimensionEndpoint::LineP2(l),
        (OffsetEntity::Arc(a), false) => DimensionEndpoint::ArcStart(a),
        (OffsetEntity::Arc(a), true) => DimensionEndpoint::ArcEnd(a),
    }
}

/// The source-orientation endpoint the chain exits / enters a segment by.
fn exit_is_end(src: &OffsetSource) -> bool {
    !src.reversed
}

/// Tangency between a result and a round-corner arc that shares an end
/// with it.
fn tangent_action(e: OffsetEntity, arc: Ref<Arc>) -> Action {
    match e {
        OffsetEntity::Line(l) => Action::ApplyTangentLA { line: l, arc },
        OffsetEntity::Arc(a) => Action::ApplyTangentAA { a, b: arc },
    }
}

fn run_checked(runner: &mut dyn ActionRunner, action: Action, what: &str) -> Result<(), String> {
    let before = runner.sketch().next_constraint_id;
    let _ = runner.run(action);
    if let Some(e) = runner.take_error() {
        return Err(format!("{}: {}", what, e));
    }
    if runner.sketch().next_constraint_id == before {
        return Err(format!("{}: the constraint was not applied", what));
    }
    Ok(())
}

fn last_nid(runner: &dyn ActionRunner) -> u32 {
    runner.sketch().next_constraint_id.saturating_sub(1)
}

fn name_of(runner: &dyn ActionRunner, e: OffsetEntity) -> String {
    chain::entity_name(runner.sketch(), e)
}

/// The creation action for one result entity.
fn create_action(g: &ResultGeom) -> Action {
    match *g {
        ResultGeom::Line { p1, p2 } => Action::AddLine { p1, p2 },
        ResultGeom::Arc { center, radius, start, end, ccw, closed } => {
            if closed {
                Action::AddCircle { center, edge: vect2d::new(center.x + radius, center.y) }
            } else {
                let at = |t: f64| vect2d::new(center.x + radius * t.cos(), center.y + radius * t.sin());
                let _ = ccw;
                Action::AddArc { start: at(start), end: at(end), mid: at(0.5 * (start + end)) }
            }
        }
        ResultGeom::Ellipse { center, rx, ry, rotation, start, end, ccw, closed } => {
            if closed {
                Action::AddEllipse { center, rx, ry, rotation }
            } else {
                Action::AddEllipticArc { center, rx, ry, rotation, start, end, ccw }
            }
        }
    }
}

/// The actions matching the source's construction flag and style onto a
/// result entity.
fn flag_actions(sketch: &Sketch, src: OffsetEntity, res: OffsetEntity, acts: &mut Vec<Action>) {
    let (construction, style) = match src {
        OffsetEntity::Line(l) => (sketch.lines[l].construction, sketch.lines[l].style),
        OffsetEntity::Arc(a) => (sketch.arcs[a].construction, sketch.arcs[a].style),
    };
    if construction {
        match res {
            OffsetEntity::Line(l) => acts.push(Action::SetConstructionLine { line: l, on: true }),
            OffsetEntity::Arc(a) => acts.push(Action::SetConstructionArc { arc: a, on: true }),
        }
    }
    if style != LineStyle::Solid {
        match res {
            OffsetEntity::Line(l) => acts.push(Action::SetStyleLine { line: l, style }),
            OffsetEntity::Arc(a) => acts.push(Action::SetStyleArc { arc: a, style }),
        }
    }
}

/// Run a batch and report its error.
fn run_batch(runner: &mut dyn ActionRunner, label: &str, actions: Vec<Action>) -> Result<crate::actions::Created, String> {
    let created = runner.run(Action::Batch { label: label.to_string(), actions });
    match runner.take_error() {
        Some(e) => Err(format!("{}: {}", label, e)),
        None => Ok(created),
    }
}

/// Create entities in one batch, returning them in order.
fn create_entities(runner: &mut dyn ActionRunner, label: &str, creates: Vec<Action>) -> Result<Vec<OffsetEntity>, String> {
    let count = creates.len();
    let crate::actions::Created::Many(list) = run_batch(runner, label, creates)? else {
        return Err(format!("{}: nothing was added", label));
    };
    let entities: Vec<OffsetEntity> = list
        .into_iter()
        .filter_map(|c| match c {
            crate::actions::Created::Line(l) => Some(OffsetEntity::Line(l)),
            crate::actions::Created::Arc(a) => Some(OffsetEntity::Arc(a)),
            _ => None,
        })
        .collect();
    if entities.len() != count {
        return Err(format!("{}: not every entity was added", label));
    }
    Ok(entities)
}

/// Collects one constraint batch: constraint-pushing actions first (their
/// nids predicted chronologically, see `Action::Batch`), then the
/// dimensions, then flag actions.
struct ConstraintBatch {
    acts: Vec<Action>,
    next_nid: u32,
    dims: Vec<Action>,
    expects: Vec<MetaValue>,
    flags: Vec<Action>,
}

impl ConstraintBatch {
    fn new(runner: &dyn ActionRunner) -> Self {
        ConstraintBatch {
            acts: Vec::new(),
            next_nid: runner.sketch().next_constraint_id,
            dims: Vec::new(),
            expects: Vec::new(),
            flags: Vec::new(),
        }
    }

    /// Queue a constraint; returns the nid it will get.
    fn constraint(&mut self, a: Action) -> u32 {
        self.acts.push(a);
        let n = self.next_nid;
        self.next_nid += 1;
        n
    }

    /// Queue a dimension holding `expect`.
    fn dim(&mut self, kind: DimensionKind, expect: &MetaValue) {
        self.dims.push(Action::AddDimension {
            kind,
            value: expect.value,
            expr: expect.expr.clone(),
            derived: false,
            range: None,
        });
        self.expects.push(expect.clone());
    }

    /// Run everything as one batch; returns the dims with their dids.
    fn run(self, runner: &mut dyn ActionRunner, label: &str) -> Result<Vec<OffsetDim>, String> {
        let dim0 = runner.sketch().dimensions.len();
        let predicted = self.next_nid;
        let mut acts = self.acts;
        acts.extend(self.dims);
        acts.extend(self.flags);
        run_batch(runner, label, acts)?;
        if runner.sketch().next_constraint_id < predicted {
            return Err(format!("{}: a constraint was not applied", label));
        }
        let sketch = runner.sketch();
        if sketch.dimensions.len() != dim0 + self.expects.len() {
            return Err(format!("{}: a dimension was not added", label));
        }
        Ok(sketch.dimensions[dim0..]
            .iter()
            .zip(self.expects)
            .map(|(d, expect)| OffsetDim { did: d.did, expect })
            .collect())
    }
}

/// A side's result entities with their sources: `segs[k]` is the offset
/// of `seq.segs[sources[k]]`.
#[derive(Clone, Copy)]
struct SideSegs<'a> {
    segs: &'a [OffsetEntity],
    sources: &'a [usize],
}

impl SideSegs<'_> {
    fn len(&self) -> usize {
        self.segs.len()
    }
    fn source<'s>(&self, seq: &'s Sequence, k: usize) -> &'s OffsetSource {
        &seq.segs[self.sources[k]]
    }
}

/// The actions pinning the free ends of an open sequence's result on
/// the source ends' normals.
fn free_end_pin_actions(sketch: &Sketch, seq: &Sequence, side: SideSegs) -> Vec<Action> {
    let m = side.len();
    if seq.closed || m == 0 {
        return Vec::new();
    }
    let first = side.source(seq, 0);
    if matches!(first.entity, OffsetEntity::Arc(a) if sketch.arcs[a].closed) {
        return Vec::new();
    }
    let last = side.source(seq, m - 1);
    vec![
        Action::ApplyOnNormal {
            placed: endpoint_of(side.segs[0], !exit_is_end(first)),
            reference: endpoint_of(first.entity, !exit_is_end(first)),
        },
        Action::ApplyOnNormal {
            placed: endpoint_of(side.segs[m - 1], exit_is_end(last)),
            reference: endpoint_of(last.entity, exit_is_end(last)),
        },
    ]
}

/// Pin the free ends of an open sequence's result on the source ends'
/// normals. Returns the nids.
fn pin_free_ends(runner: &mut dyn ActionRunner, seq: &Sequence, side: SideSegs) -> Result<Vec<u32>, String> {
    let mut pins = Vec::new();
    for a in free_end_pin_actions(runner.sketch(), seq, side) {
        run_checked(runner, a, "on_normal")?;
        pins.push(last_nid(runner));
    }
    Ok(pins)
}

/// The optional pins of an existing side, all at once: the tangent
/// joints' exit ends and (unless round caps hold them) the free ends.
/// For turning `pin` on afterwards.
fn pin_side(
    runner: &mut dyn ActionRunner,
    seq: &Sequence,
    side: SideSegs,
    joints: &[JointPlan],
    free_ends: bool,
) -> Result<Vec<u32>, String> {
    let mut pins = Vec::new();
    let m = side.len();
    for (k, joint) in joints.iter().enumerate() {
        if !joint.tangent {
            continue;
        }
        let a_src = side.source(seq, k);
        let placed = endpoint_of(side.segs[k], exit_is_end(a_src));
        let reference = endpoint_of(a_src.entity, exit_is_end(a_src));
        run_checked(runner, Action::ApplyOnNormal { placed, reference }, "on_normal")?;
        pins.push(last_nid(runner));
        if joint.closure {
            let b_src = side.source(seq, (k + 1) % m);
            let placed = endpoint_of(side.segs[(k + 1) % m], !exit_is_end(b_src));
            let reference = endpoint_of(b_src.entity, !exit_is_end(b_src));
            run_checked(runner, Action::ApplyOnNormal { placed, reference }, "on_normal")?;
            pins.push(last_nid(runner));
        }
    }
    if free_ends {
        pins.extend(pin_free_ends(runner, seq, side)?);
    }
    Ok(pins)
}

/// The free end of a side's result at the sequence's start / end.
fn free_end_of(seq: &Sequence, side: SideSegs, at_end: bool) -> DimensionEndpoint {
    let m = side.len();
    if at_end {
        endpoint_of(side.segs[m - 1], exit_is_end(side.source(seq, m - 1)))
    } else {
        endpoint_of(side.segs[0], !exit_is_end(side.source(seq, 0)))
    }
}

/// Create the caps of a two-sided open offset, in two batches: a line
/// across each end, joined to both results; or a half circle around the
/// source end, joined and tangent to both results (which makes its
/// radius the distance) and held in place by a pin of the `distance`
/// side's end on the source end's normal.
fn apply_caps(
    runner: &mut dyn ActionRunner,
    plan: &OffsetPlan,
    sides: &[OffsetSideResult],
    out: &mut OffsetOutcome,
) -> Result<OffsetCaps, String> {
    let mut caps = OffsetCaps { kind: plan.params.caps, ..Default::default() };
    if plan.caps.is_empty() {
        return Ok(caps);
    }
    let seq = &plan.seq;
    let n = seq.segs.len();
    let side_a = sides.iter().find(|s| s.sign == plan.sides[0].sign).ok_or("caps: the first side is missing")?;
    let side_b = sides.iter().find(|s| s.sign == plan.sides[1].sign).ok_or("caps: the second side is missing")?;
    let (src_a, src_b) = (side_a.sources(n), side_b.sources(n));
    let (segs_a, segs_b) = (
        SideSegs { segs: &side_a.segs, sources: &src_a },
        SideSegs { segs: &side_b.segs, sources: &src_b },
    );
    // Batch 1: the cap entities.
    let creates: Vec<Action> = plan
        .caps
        .iter()
        .map(|cap| match cap.geom {
            CapGeom::Line { a, b } => Action::AddLine { p1: a, p2: b },
            CapGeom::Arc { start, end, mid } => Action::AddArc { start, end, mid },
        })
        .collect();
    caps.entities = create_entities(runner, "Offset caps", creates)?;
    // Batch 2: the joints, tangents and the round cap's pin.
    let mut batch = ConstraintBatch::new(runner);
    let mut names = Vec::new();
    for (cap, &entity) in plan.caps.iter().zip(&caps.entities) {
        let end_a = free_end_of(seq, segs_a, cap.at_end);
        let end_b = free_end_of(seq, segs_b, cap.at_end);
        let res_a = if cap.at_end { *side_a.segs.last().unwrap() } else { side_a.segs[0] };
        let res_b = if cap.at_end { *side_b.segs.last().unwrap() } else { side_b.segs[0] };
        let (own_start, own_end) = match entity {
            OffsetEntity::Line(l) => (DimensionEndpoint::LineP1(l), DimensionEndpoint::LineP2(l)),
            OffsetEntity::Arc(a) => (DimensionEndpoint::ArcStart(a), DimensionEndpoint::ArcEnd(a)),
        };
        names.push(name_of(runner, entity));
        let c1 = Action::coincident(own_start, end_a).expect("endpoint pair");
        caps.constraints.push(batch.constraint(c1));
        let c2 = Action::coincident(own_end, end_b).expect("endpoint pair");
        caps.constraints.push(batch.constraint(c2));
        if let (CapGeom::Arc { .. }, OffsetEntity::Arc(arc)) = (cap.geom, entity) {
            caps.constraints.push(batch.constraint(tangent_action(res_a, arc)));
            caps.constraints.push(batch.constraint(tangent_action(res_b, arc)));
            // The plan made sure both sides end on the same segment.
            let src = segs_a.source(seq, if cap.at_end { segs_a.len() - 1 } else { 0 });
            let reference = endpoint_of(src.entity, cap.at_end == exit_is_end(src));
            caps.constraints.push(batch.constraint(Action::ApplyOnNormal { placed: end_a, reference }));
        }
    }
    batch.run(runner, "Offset cap joints")?;
    out.entities.push(names);
    out.constraints.extend(caps.constraints.iter().map(|n| format!("C{}", n)));
    Ok(caps)
}

/// Create one side's result with its relations, joints and pins, in two
/// batches: the entities, then the constraints and dimensions (which are
/// consistent by construction -- the plan already refused anything
/// degenerate -- so they skip the per-constraint gate).
fn apply_side(
    runner: &mut dyn ActionRunner,
    plan: &OffsetPlan,
    side: &SidePlan,
    out: &mut OffsetOutcome,
) -> Result<OffsetSideResult, String> {
    let seq = &plan.seq;
    let m = side.sources.len();
    // Batch 1: the result entities, then the round-corner arcs.
    let mut creates: Vec<Action> = side.results.iter().map(create_action).collect();
    let corner_joints: Vec<usize> = side
        .joints
        .iter()
        .enumerate()
        .filter_map(|(k, j)| j.round.as_ref().map(|_| k))
        .collect();
    for &k in &corner_joints {
        let rc = side.joints[k].round.as_ref().expect("round joint");
        creates.push(Action::AddArc { start: rc.start, end: rc.end, mid: rc.mid });
    }
    let created = create_entities(runner, "Offset result", creates)?;
    let segs: Vec<OffsetEntity> = created[..m].to_vec();
    let corners: Vec<OffsetEntity> = created[m..].to_vec();
    let corner_of = |k: usize| -> OffsetEntity {
        corners[corner_joints.iter().position(|&j| j == k).expect("a corner joint")]
    };
    let names: Vec<String> = created.iter().map(|&e| name_of(runner, e)).collect();

    // Batch 2: relations, joints, pins, dims, flags.
    let mut batch = ConstraintBatch::new(runner);
    let mut constraints = Vec::new();
    let mut pins = Vec::new();
    for (k, &si) in side.sources.iter().enumerate() {
        let (src, res) = (&seq.segs[si], segs[k]);
        flag_actions(runner.sketch(), src.entity, res, &mut batch.flags);
        let kind = match (src.entity, res) {
            (OffsetEntity::Line(s), OffsetEntity::Line(r)) => {
                constraints.push(batch.constraint(Action::ApplyParallel { a: s, b: r }));
                DimensionKind::LineLineDistance(s, r)
            }
            (OffsetEntity::Arc(s), OffsetEntity::Arc(r)) => {
                constraints.push(batch.constraint(Action::ApplyConcentric { a: s, b: r }));
                if runner.sketch().arcs[s].is_ellipse {
                    constraints.push(batch.constraint(Action::ApplyArcArcParallel { a: s, b: r }));
                }
                DimensionKind::ConcentricDistance(s, r)
            }
            _ => unreachable!("a result has its source's kind"),
        };
        if side.dims[k] {
            batch.dim(kind, &side.distance);
        }
    }
    // Joints, in chain order: a pin on the earlier result's exit end where
    // the sources are tangent, then the coincidence. A round corner is a
    // fillet between the two results: an arc of the distance joined to
    // both ends, tangent to both, its radius a dimension; the geometry
    // puts its center on the source joint.
    let side_segs = SideSegs { segs: &segs, sources: &side.sources };
    for (k, joint) in side.joints.iter().enumerate() {
        let (a_src, b_src) = (side_segs.source(seq, k), side_segs.source(seq, (k + 1) % m));
        let (a_res, b_res) = (segs[k], segs[(k + 1) % m]);
        let a_end = endpoint_of(a_res, exit_is_end(a_src));
        let b_start = endpoint_of(b_res, !exit_is_end(b_src));
        let a_ref = endpoint_of(a_src.entity, exit_is_end(a_src));
        if joint.round.is_some() {
            let OffsetEntity::Arc(arc) = corner_of(k) else { unreachable!("a corner is an arc") };
            let c1 = Action::coincident(DimensionEndpoint::ArcStart(arc), a_end).expect("endpoint pair");
            constraints.push(batch.constraint(c1));
            let c2 = Action::coincident(DimensionEndpoint::ArcEnd(arc), b_start).expect("endpoint pair");
            constraints.push(batch.constraint(c2));
            constraints.push(batch.constraint(tangent_action(a_res, arc)));
            constraints.push(batch.constraint(tangent_action(b_res, arc)));
            batch.dim(DimensionKind::ArcRadius(arc), &side.distance);
            continue;
        }
        if joint.tangent && plan.params.pinned {
            pins.push(batch.constraint(Action::ApplyOnNormal { placed: a_end, reference: a_ref }));
        }
        if joint.closure {
            // Closing an all-tangent loop: the later result's start is
            // pinned too; the ends meet by geometry.
            if plan.params.pinned {
                let b_ref = endpoint_of(b_src.entity, !exit_is_end(b_src));
                pins.push(batch.constraint(Action::ApplyOnNormal { placed: b_start, reference: b_ref }));
            }
            continue;
        }
        let action = Action::coincident(a_end, b_start).expect("endpoint pair");
        constraints.push(batch.constraint(action));
    }
    // Round caps hold the free ends themselves (tangent to a half circle
    // around the source end): no pins there.
    if plan.params.pinned && plan.params.caps != CapKind::Round {
        for a in free_end_pin_actions(runner.sketch(), seq, side_segs) {
            pins.push(batch.constraint(a));
        }
    }
    let dims = batch.run(runner, "Offset relations")?;
    out.entities.push(names);
    out.constraints.extend(constraints.iter().chain(pins.iter()).map(|n| format!("C{}", n)));
    let sketch = runner.sketch();
    for d in &dims {
        if let Some(i) = sketch.dimension_index_by_did(d.did) {
            out.dims.push(sketch.dimensions[i].name.clone());
        }
    }
    Ok(OffsetSideResult {
        sign: side.sign,
        segs,
        dropped: side.dropped.clone(),
        corners,
        constraints,
        pins,
        dims,
    })
}

/// Create the planned offset and register its meta-constraint, as one
/// undo group; a failure half-way rolls it back, leaving nothing behind.
pub fn apply(runner: &mut dyn ActionRunner, plan: &OffsetPlan) -> Result<OffsetOutcome, String> {
    runner.begin_group();
    let r = apply_inner(runner, plan);
    if r.is_err() {
        runner.rollback_group();
    } else {
        runner.end_group();
    }
    r
}

fn apply_inner(runner: &mut dyn ActionRunner, plan: &OffsetPlan) -> Result<OffsetOutcome, String> {
    let mut out = OffsetOutcome {
        approximate: plan.approximate,
        dropped: dropped_names(runner.sketch(), plan),
        ..Default::default()
    };
    let mut sides = Vec::with_capacity(plan.sides.len());
    for side in &plan.sides {
        sides.push(apply_side(runner, plan, side, &mut out)?);
    }
    let caps = apply_caps(runner, plan, &sides, &mut out)?;
    let offset = Offset {
        source: plan.seq.segs.clone(),
        closed: plan.seq.closed,
        kind: plan.params.kind,
        distance: plan.params.distance.clone(),
        distance2: plan.params.distance2.clone(),
        side: plan.params.side,
        pinned: plan.params.pinned,
        round: plan.params.round,
        sides,
        caps,
    };
    runner.run(Action::RegisterMeta {
        meta: Meta { mid: 0, name: String::new(), kind: MetaKind::Offset(offset) },
    });
    if let Some(e) = runner.take_error() {
        return Err(format!("registering the offset: {}", e));
    }
    let m = runner.sketch().metas.last().expect("just registered");
    out.name = m.name.clone();
    out.mid = m.mid;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Editing
// ---------------------------------------------------------------------------

/// The offset record of meta `mid`, with the meta's name.
fn offset_record(sketch: &Sketch, mid: u32) -> Result<(String, Offset), String> {
    let i = sketch.meta_index(mid).ok_or_else(|| format!("no meta-constraint M{}", mid))?;
    let m = &sketch.metas[i];
    let o = m.as_offset().ok_or_else(|| format!("{} is not an offset", m.name))?;
    Ok((m.name.clone(), o.clone()))
}

/// The current parameters of an offset.
pub fn params_of(o: &Offset) -> OffsetParams {
    OffsetParams {
        kind: o.kind,
        distance: o.distance.clone(),
        distance2: o.distance2.clone(),
        side: o.side,
        pinned: o.pinned,
        round: o.round,
        caps: o.caps.kind,
    }
}

/// The sequence an offset was made of.
pub fn sequence_of(o: &Offset) -> Sequence {
    Sequence { segs: o.source.clone(), closed: o.closed }
}

/// Write planned geometry into an existing result entity's values.
fn reseed(sketch: &mut Sketch, e: OffsetEntity, g: &ResultGeom) {
    match (e, *g) {
        (OffsetEntity::Line(l), ResultGeom::Line { p1, p2 }) => {
            let line = &mut sketch.lines[l];
            line.p1.value = p1;
            line.p2.value = p2;
        }
        (OffsetEntity::Arc(a), ResultGeom::Arc { center, radius, start, end, .. }) => {
            let arc = &mut sketch.arcs[a];
            arc.center.value = center;
            arc.radius.value = radius;
            arc.radius_b.value = radius;
            if !arc.closed {
                arc.start_angle.value = start;
                arc.end_angle.value = end;
            }
        }
        (OffsetEntity::Arc(a), ResultGeom::Ellipse { center, rx, ry, rotation, start, end, .. }) => {
            let arc = &mut sketch.arcs[a];
            arc.center.value = center;
            arc.radius.value = rx;
            arc.radius_b.value = ry;
            arc.rotation.value = rotation;
            if !arc.closed {
                arc.start_angle.value = start;
                arc.end_angle.value = end;
            }
        }
        _ => {}
    }
}

/// Re-point an existing side's geometry at a new plan for that side: the
/// result values are re-seeded and the concentric-distance signs follow
/// the new radii, so the solver settles on the new side.
fn reseed_side(sketch: &mut Sketch, o: &Offset, side: &OffsetSideResult, plan_side: &SidePlan) {
    for (k, &e) in side.segs.iter().enumerate() {
        reseed(sketch, e, &plan_side.results[k]);
    }
    for (k, &si) in side.sources(o.source.len()).iter().enumerate() {
        if let (OffsetEntity::Arc(s), OffsetEntity::Arc(r)) = (o.source[si].entity, side.segs[k]) {
            let gap = sketch.arcs[r].radius.value - sketch.arcs[s].radius.value;
            for c in sketch.distance_concentric.iter_mut() {
                if c.a == s && c.b == r {
                    c.sign = if gap >= 0.0 { 1.0 } else { -1.0 };
                }
            }
        }
    }
}

/// Change an offset's parameters. Distances are rewritten into the owned
/// dimensions; a side change re-seeds the geometry on the new side; a
/// kind change creates or removes a side chain. Returns the outcome for
/// any side it created. Runs as one undo group; a failure half-way rolls
/// it back, so the offset is either edited or untouched.
pub fn update(runner: &mut dyn ActionRunner, mid: u32, params: &OffsetParams) -> Result<OffsetOutcome, String> {
    let (name, o) = offset_record(runner.sketch(), mid)?;
    let seq = sequence_of(&o);
    let new_plan = plan(runner.sketch(), &seq, params)?;
    let mut out = OffsetOutcome {
        name,
        mid,
        approximate: new_plan.approximate,
        dropped: dropped_names(runner.sketch(), &new_plan),
        ..Default::default()
    };
    runner.begin_group();
    let r = update_inner(runner, mid, params, &o, &seq, &new_plan, &mut out);
    if let Err(e) = r {
        runner.rollback_group();
        return Err(e);
    }
    runner.end_group();
    Ok(out)
}

fn update_inner(
    runner: &mut dyn ActionRunner,
    mid: u32,
    params: &OffsetParams,
    o: &Offset,
    seq: &Sequence,
    new_plan: &OffsetPlan,
    out: &mut OffsetOutcome,
) -> Result<(), String> {
    let register = |runner: &mut dyn ActionRunner, o: Offset| -> Result<(), String> {
        let name = runner.sketch().metas[runner.sketch().meta_index(mid).expect("registered")].name.clone();
        runner.run(Action::RegisterMeta { meta: Meta { mid, name, kind: MetaKind::Offset(o) } });
        runner.take_error().map_or(Ok(()), Err)
    };

    // Pair the existing sides with the wanted ones: by sign where a sign
    // survives, else an existing side takes over a wanted sign (a flip
    // moves the one side across), and whatever is left over is deleted
    // or created. A side survives only with the structure the new plan
    // wants there: the same dropped segments and round corners (both
    // depend on the side and the distance); otherwise it is rebuilt.
    let wanted: Vec<(f64, MetaValue)> = params.sides();
    let same_structure = |s: &OffsetSideResult, sign: f64| -> bool {
        new_plan.sides.iter().find(|p| p.sign == sign).is_some_and(|p| {
            p.dropped == s.dropped && p.joints.iter().filter(|j| j.round.is_some()).count() == s.corners.len()
        })
    };
    let mut remaining = wanted.clone();
    let mut kept: Vec<OffsetSideResult> = Vec::new();
    let mut unmatched: Vec<OffsetSideResult> = Vec::new();
    for s in &o.sides {
        match remaining.iter().position(|(sign, _)| *sign == s.sign) {
            Some(pos) if same_structure(s, s.sign) => {
                remaining.remove(pos);
                kept.push(s.clone());
            }
            _ => unmatched.push(s.clone()),
        }
    }
    let mut doomed: Vec<OffsetEntity> = Vec::new();
    for mut s in unmatched {
        // Across the chain the same dropped set is the same structure
        // only without round corners (those sit at different joints).
        let movable = !remaining.is_empty() && s.corners.is_empty() && same_structure(&s, remaining[0].0);
        if movable {
            s.sign = remaining.remove(0).0;
            kept.push(s);
        } else {
            doomed.extend(s.segs.iter().copied());
            doomed.extend(s.corners.iter().copied());
        }
    }
    // The caps are rebuilt when their kind changes or any side is; a
    // plain distance edit keeps them (they follow their coincidences and
    // dims).
    let sides_change = !doomed.is_empty() || kept.len() != wanted.len();
    let rebuild_caps = params.caps != o.caps.kind || sides_change;
    let mut doomed_caps: Vec<OffsetEntity> = Vec::new();
    let mut doomed_cap_constraints: Vec<u32> = Vec::new();
    let mut new_record = o.clone();
    new_record.kind = params.kind;
    new_record.distance = params.distance.clone();
    new_record.distance2 = params.distance2.clone();
    new_record.side = params.side;
    new_record.pinned = params.pinned;
    new_record.round = params.round;
    new_record.sides = kept;
    if rebuild_caps {
        doomed_caps = std::mem::take(&mut new_record.caps.entities);
        doomed_cap_constraints = std::mem::take(&mut new_record.caps.constraints);
        new_record.caps = OffsetCaps { kind: params.caps, ..Default::default() };
    }
    // The record is written first so the deletions below do not read as
    // tampering.
    register(runner, new_record.clone())?;
    let mut deletes: Vec<Action> = doomed
        .into_iter()
        .chain(doomed_caps)
        .map(|e| match e {
            OffsetEntity::Line(l) => Action::DeleteLine { line: l },
            OffsetEntity::Arc(a) => Action::DeleteArc { arc: a },
        })
        .collect();
    // A cap's pin is between results and sources: it does not go with
    // the cap entity.
    deletes.extend(
        doomed_cap_constraints
            .into_iter()
            .filter(|&nid| crate::meta::nid_exists(runner.sketch(), nid))
            .map(|nid| Action::DeleteConstraint { id: crate::ids::ConstraintId::Numbered(nid) }),
    );
    if !deletes.is_empty() {
        run_batch(runner, "Delete offset result", deletes)?;
    }
    // Surviving sides: re-seed on the (possibly new) side, then write the
    // distances and the record's expectations in one step.
    for (sign, _) in &wanted {
        if let (Some(side), Some(plan_side)) = (
            new_record.sides.iter().find(|s| s.sign == *sign),
            new_plan.sides.iter().find(|s| s.sign == *sign),
        ) {
            reseed_side(runner.sketch_mut(), &new_record, side, plan_side);
        }
    }
    runner.run(Action::SetOffsetDistances { mid, distance: params.distance.clone(), distance2: params.distance2.clone() });
    if let Some(e) = runner.take_error() { return Err(e); }
    // Pins on the surviving sides: redone when `pin` / `nopin` changes or
    // round caps come or go (they hold the free ends themselves). The old
    // pins are deleted (the record rewritten first so the deletion is not
    // tampering), then the wanted ones are added.
    let round_caps = |k: CapKind| k == CapKind::Round;
    if params.pinned != o.pinned || round_caps(params.caps) != round_caps(o.caps.kind) {
        let (_, mut rec) = offset_record(runner.sketch(), mid)?;
        let old_pins: Vec<u32> = rec.sides.iter().flat_map(|s| s.pins.iter().copied()).collect();
        for s in rec.sides.iter_mut() {
            s.pins.clear();
        }
        register(runner, rec.clone())?;
        if !old_pins.is_empty() {
            let deletes = old_pins
                .into_iter()
                .map(|nid| Action::DeleteConstraint { id: crate::ids::ConstraintId::Numbered(nid) })
                .collect();
            run_batch(runner, "Delete offset pins", deletes)?;
        }
        if params.pinned {
            for s in rec.sides.iter_mut() {
                let plan_side = new_plan.sides.iter().find(|p| p.sign == s.sign).expect("planned");
                let side_segs = SideSegs { segs: &s.segs, sources: &plan_side.sources };
                s.pins = pin_side(runner, &seq, side_segs, &plan_side.joints, !round_caps(params.caps))?;
                out.constraints.extend(s.pins.iter().map(|n| format!("C{}", n)));
            }
            register(runner, rec)?;
        }
    }
    // New sides.
    let mut created_sides: Vec<OffsetSideResult> = Vec::new();
    for (sign, _) in &wanted {
        if new_record.sides.iter().any(|s| s.sign == *sign) { continue; }
        let plan_side = new_plan.sides.iter().find(|s| s.sign == *sign).expect("planned");
        created_sides.push(apply_side(runner, new_plan, plan_side, out)?);
    }
    if !created_sides.is_empty() {
        let (_, mut rec) = offset_record(runner.sketch(), mid)?;
        rec.sides.extend(created_sides);
        register(runner, rec)?;
    }
    // New caps, across the final sides.
    if rebuild_caps && !new_plan.caps.is_empty() {
        let (_, mut rec) = offset_record(runner.sketch(), mid)?;
        rec.caps = apply_caps(runner, new_plan, &rec.sides, out)?;
        register(runner, rec)?;
    }
    Ok(())
}

/// Which side of the sequence a point is on: +1 left of the chain
/// direction, -1 right, judged at the point of the sequence nearest to
/// `p` (sampled). For the tool's "the side follows the mouse".
pub fn side_of_point(sketch: &Sketch, seq: &Sequence, p: vect2d) -> f64 {
    let mut best: Option<(f64, f64)> = None; // (distance, side)
    for src in &seq.segs {
        let seg = seg_of(sketch, *src);
        let n = 32;
        for i in 0..=n {
            let t = i as f64 / n as f64;
            let (q, tangent) = match seg.geom {
                SegGeom::Line { a, b } => (a + (b - a) * t, unit(b - a)),
                SegGeom::Arc { center, ra, rb, rot, t0, t1, ccw_travel, .. } => {
                    let u = t0 + (t1 - t0) * t;
                    (arc_point(center, ra, rb, rot, u), arc_travel_tangent(ra, rb, rot, u, ccw_travel))
                }
            };
            let d = dist(p, q);
            if best.map_or(true, |(bd, _)| d < bd) {
                let s = cross(tangent, p - q);
                best = Some((d, if s >= 0.0 { 1.0 } else { -1.0 }));
            }
        }
    }
    best.map_or(1.0, |(_, s)| s)
}

/// Sample a planned result for drawing: points along it, in order.
pub fn sample_result(g: &ResultGeom, n: usize) -> Vec<vect2d> {
    match *g {
        ResultGeom::Line { p1, p2 } => vec![p1, p2],
        ResultGeom::Arc { center, radius, start, end, closed, .. } => {
            let (s, e) = if closed { (0.0, TAU) } else { (start, end) };
            (0..=n).map(|i| arc_point(center, radius, radius, 0.0, s + (e - s) * i as f64 / n as f64)).collect()
        }
        ResultGeom::Ellipse { center, rx, ry, rotation, start, end, closed, .. } => {
            let (s, e) = if closed { (0.0, TAU) } else { (start, end) };
            (0..=n).map(|i| arc_point(center, rx, ry, rotation, s + (e - s) * i as f64 / n as f64)).collect()
        }
    }
}

/// Points along a three-point arc (start, end, through `mid`), for a
/// preview.
fn sample_three_point_arc(start: vect2d, end: vect2d, mid: vect2d, n: usize) -> Vec<vect2d> {
    let Some((c, r, sa, ea, ccw)) = crate::geometry::circumscribed_arc(start, end, mid) else {
        return vec![start, mid, end];
    };
    let mut sweep = ea - sa;
    if ccw {
        while sweep < 0.0 { sweep += TAU; }
    } else {
        while sweep > 0.0 { sweep -= TAU; }
    }
    (0..=n).map(|i| arc_point(c, r, r, 0.0, sa + sweep * i as f64 / n as f64)).collect()
}

/// Names of the source segments whose offset vanishes on some side of
/// the plan, each once, in chain order.
pub fn dropped_names(sketch: &Sketch, plan: &OffsetPlan) -> Vec<String> {
    let mut gone: Vec<usize> = plan.sides.iter().flat_map(|s| s.dropped.iter().copied()).collect();
    gone.sort_unstable();
    gone.dedup();
    gone.into_iter().map(|i| chain::entity_name(sketch, plan.seq.segs[i].entity)).collect()
}

/// Every polyline of a plan's result: the segments, the round corners
/// and the caps. For the preview.
pub fn preview_polylines(plan: &OffsetPlan, n: usize) -> Vec<Vec<vect2d>> {
    let mut out = Vec::new();
    for side in &plan.sides {
        for g in &side.results {
            out.push(sample_result(g, n));
        }
        for j in &side.joints {
            if let Some(rc) = &j.round {
                out.push(sample_three_point_arc(rc.start, rc.end, rc.mid, n / 2));
            }
        }
    }
    for cap in &plan.caps {
        match cap.geom {
            CapGeom::Line { a, b } => out.push(vec![a, b]),
            CapGeom::Arc { start, end, mid } => out.push(sample_three_point_arc(start, end, mid, n / 2)),
        }
    }
    out
}

/// One line describing an offset, after the meta's name.
pub fn describe(sketch: &Sketch, o: &Offset) -> String {
    let src: Vec<String> = o.source.iter().map(|s| chain::entity_name(sketch, s.entity)).collect();
    let kind = match o.kind {
        OffsetKind::OneSide => format!("{} {}", fmt_value(&o.distance), side_name(o.side)),
        OffsetKind::Symmetric => format!("{} symmetric", fmt_value(&o.distance)),
        OffsetKind::TwoSides => format!(
            "{} {} / {} {}",
            fmt_value(&o.distance),
            side_name(o.side),
            o.distance2.as_ref().map(fmt_value).unwrap_or_default(),
            side_name(-o.side)
        ),
    };
    let names_of = |es: &[OffsetEntity]| -> String {
        es.iter()
            .filter(|e| crate::meta::entity_exists(sketch, **e))
            .map(|e| chain::entity_name(sketch, *e))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let mut results: Vec<String> = o
        .sides
        .iter()
        .map(|s| {
            let es: Vec<OffsetEntity> = s.segs.iter().chain(s.corners.iter()).copied().collect();
            let gone: Vec<OffsetEntity> = s.dropped.iter().filter_map(|&i| o.source.get(i)).map(|src| src.entity).collect();
            if gone.is_empty() {
                format!("{}: {}", side_name(s.sign), names_of(&es))
            } else {
                format!("{}: {} ({} vanished)", side_name(s.sign), names_of(&es), names_of(&gone))
            }
        })
        .collect();
    if o.caps.kind != CapKind::None {
        results.push(format!("{} caps: {}", cap_name(o.caps.kind), names_of(&o.caps.entities)));
    }
    format!(
        "offset of {}{} by {} -> {}{}{}",
        src.join(" "),
        if o.closed { " (closed)" } else { "" },
        kind,
        results.join("; "),
        if o.round { " [round]" } else { "" },
        if o.pinned { "" } else { " [nopin]" }
    )
}

pub fn cap_name(k: CapKind) -> &'static str {
    match k {
        CapKind::None => "no",
        CapKind::Line => "line",
        CapKind::Round => "round",
    }
}

fn fmt_value(v: &MetaValue) -> String {
    match &v.expr {
        Some(e) => format!("{} ({:.4})", e, v.value),
        None => format!("{}", v.value),
    }
}

pub fn side_name(sign: f64) -> &'static str {
    if sign > 0.0 { "left" } else { "right" }
}

/// Parse a distance token the way dimension values are typed: a number,
/// a live expression (re-evaluated every solve), or `=expr` evaluated
/// once. The value is evaluated now either way: the plan needs it.
pub fn parse_value(sketch: &Sketch, token: &str) -> Result<MetaValue, String> {
    let token = token.trim().trim_matches('"');
    if let Ok(v) = token.parse::<f64>() {
        return Ok(MetaValue { value: v, expr: None });
    }
    let (src, live) = match token.strip_prefix('=') {
        Some(rest) => (rest.trim(), false),
        None => (token, true),
    };
    let value = crate::commands::eval_expr(sketch, src)
        .map_err(|e| format!("cannot evaluate '{}': {}", src, e))?;
    Ok(MetaValue { value, expr: if live { Some(src.to_string()) } else { None } })
}

/// Reference helpers for callers holding a `Ref`.
pub fn line_entity(r: Ref<Line>) -> OffsetEntity { OffsetEntity::Line(r) }
pub fn arc_entity(r: Ref<Arc>) -> OffsetEntity { OffsetEntity::Arc(r) }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::Created;
    use crate::commands::CommandContext;

    fn ctx(script: &str) -> CommandContext {
        let mut ctx = CommandContext::new();
        for r in crate::commands::execute(&mut ctx, script) {
            assert!(!r.is_error, "{}", r.output);
        }
        ctx
    }

    /// A runner that refuses the n-th batch (the engine's actions all run
    /// in batches), to fail an operation half-way.
    struct FailAt<'a> {
        inner: &'a mut CommandContext,
        fail_at: usize,
        seen: usize,
        failed: bool,
    }

    impl ActionRunner for FailAt<'_> {
        fn sketch(&self) -> &Sketch { self.inner.sketch() }
        fn sketch_mut(&mut self) -> &mut Sketch { self.inner.sketch_mut() }
        fn run(&mut self, action: Action) -> Created {
            if matches!(action, Action::Batch { .. }) {
                self.seen += 1;
                if self.seen == self.fail_at {
                    self.failed = true;
                    return Created::Nothing;
                }
            }
            self.inner.run(action)
        }
        fn run_unchecked(&mut self, action: Action) -> Created { self.inner.run_unchecked(action) }
        fn take_error(&mut self) -> Option<String> {
            if std::mem::take(&mut self.failed) { Some("refused for the test".into()) } else { self.inner.take_error() }
        }
        fn begin_group(&mut self) { self.inner.begin_group() }
        fn end_group(&mut self) { self.inner.end_group() }
        fn rollback_group(&mut self) { self.inner.rollback_group() }
    }

    fn counts(s: &Sketch) -> (usize, usize, usize, usize, usize) {
        (s.lines.refs().count(), s.arcs.refs().count(), s.dimensions.len(), s.metas.len(), s.parallel.len())
    }

    fn params(d: f64) -> OffsetParams {
        OffsetParams {
            kind: OffsetKind::OneSide,
            distance: MetaValue { value: d, expr: None },
            distance2: None,
            side: 1.0,
            pinned: true,
            round: false,
            caps: CapKind::None,
        }
    }

    /// A failure while creating leaves nothing: no entities, relations,
    /// dims or record, and no history entry to undo.
    #[test]
    fn a_failed_apply_leaves_nothing_behind() {
        let mut c = ctx("add_line 0,0 4,0 4,3");
        let before = counts(&c.sketch);
        let history_len = c.history.actions.len();
        let seq = chain::walk(&c.sketch, OffsetEntity::Line(crate::commands::resolve_line(&c.sketch, "L0").unwrap()));
        let plan = plan(&c.sketch, &seq, &params(1.0)).unwrap();
        // Batch 1 creates the entities; batch 2 (the relations) fails.
        let mut runner = FailAt { inner: &mut c, fail_at: 2, seen: 0, failed: false };
        let e = apply(&mut runner, &plan).unwrap_err();
        assert!(e.contains("refused for the test"), "{}", e);
        assert_eq!(counts(&c.sketch), before);
        assert_eq!(c.history.actions.len(), history_len);
        assert_eq!(c.history.cursor, history_len);
    }

    /// A failure while editing leaves the offset as it was.
    #[test]
    fn a_failed_update_keeps_the_offset() {
        let mut c = ctx("add_line 0,3 0,0 4,0; fillet L0 L1 1; offset L0 A0 L1 2");
        let before = counts(&c.sketch);
        let record = c.sketch.metas[0].clone();
        let history_len = c.history.actions.len();
        // 0.5 brings the fillet back: the side is rebuilt -- the old side
        // is deleted (batch 1), the new entities created (batch 2), and
        // the relation batch (3) fails.
        let mut runner = FailAt { inner: &mut c, fail_at: 3, seen: 0, failed: false };
        let e = update(&mut runner, 0, &params(0.5)).unwrap_err();
        assert!(e.contains("refused for the test"), "{}", e);
        assert_eq!(counts(&c.sketch), before);
        assert_eq!(c.sketch.metas[0], record);
        assert_eq!(c.history.actions.len(), history_len);
        // And it still edits afterwards.
        let out = update(&mut c, 0, &params(0.5)).unwrap();
        assert!(out.dropped.is_empty());
        assert_eq!(c.sketch.arcs.refs().count(), 2);
    }
}
