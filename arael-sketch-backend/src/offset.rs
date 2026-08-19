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
    pub distance: OffsetValue,
    pub distance2: Option<OffsetValue>,
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
    pub fn sides(&self) -> Vec<(f64, OffsetValue)> {
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
            let (sa, ea) = (a.start_angle.value, if a.closed { a.start_angle.value + TAU } else { a.end_angle.value });
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
    pub distance: OffsetValue,
    /// Parallel to the sequence.
    pub results: Vec<ResultGeom>,
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
fn offset_curve(seg: &Seg, s: f64, d: f64) -> Result<(Curve, Option<f64>), String> {
    match seg.geom {
        SegGeom::Line { a, b } => {
            let n = rot90(unit(b - a));
            Ok((Curve::Line { p: a + n * (s * d), dir: unit(b - a) }, None))
        }
        SegGeom::Arc { center, ra, rb, rot, ccw_travel, is_ellipse, .. } => {
            // Travelling counter-clockwise, left is toward the center.
            let dr = if ccw_travel { -s * d } else { s * d };
            if is_ellipse {
                Ok((Curve::Ellipse { center, ra: ra + dr, rb: rb + dr, rot }, Some(dr)))
            } else {
                Ok((Curve::Circle { center, r: ra + dr }, Some(dr)))
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
        // The unbounded offset curves, with the radial changes.
        let mut curves = Vec::with_capacity(n);
        for seg in &segs {
            let (c, dr) = offset_curve(seg, sign, dv)?;
            if let (Some(dr), SegGeom::Arc { ra, rb, is_ellipse, .. }) = (dr, seg.geom) {
                let rmin = if is_ellipse { ra.min(rb) } else { ra };
                if rmin + dr <= min_len {
                    return Err(format!(
                        "{} (radius {:.4}) cannot be offset inward by {:.4}",
                        chain::entity_name(sketch, seg.src.entity),
                        rmin,
                        dv
                    ));
                }
            }
            curves.push(c);
        }
        // Joint points: the offset of the source joint for tangent joints;
        // for corners the nearest intersection of the two result curves, or
        // with `round` on the convex side an arc of the distance around
        // the source joint from the one result's end to the other's start.
        let mut joints = Vec::with_capacity(joint_count);
        for i in 0..joint_count {
            let (a, b) = (&segs[i], &segs[(i + 1) % n]);
            let j = (a.end() + b.start()) * 0.5;
            if tangent_joint[i] {
                let p = j + a.left_normal_at_end() * (sign * dv);
                joints.push(JointPlan { tangent: true, point: p, next_start: p, round: None });
                continue;
            }
            // Convex on this side: the chain turns away from it.
            let turn = cross(a.tangent_at_end(), b.tangent_at_start());
            if params.round && sign * turn < 0.0 {
                let p1 = j + a.left_normal_at_end() * (sign * dv);
                let p2 = j + b.left_normal_at_start() * (sign * dv);
                let mid = j + unit((p1 - j) + (p2 - j)) * dv;
                joints.push(JointPlan {
                    tangent: false,
                    point: p1,
                    next_start: p2,
                    round: Some(RoundCorner { start: p1, end: p2, mid }),
                });
                continue;
            }
            let cands = intersections(&curves[i], &curves[(i + 1) % n]);
            let best = cands
                .into_iter()
                .min_by(|p, q| dist(*p, j).partial_cmp(&dist(*q, j)).unwrap());
            match best {
                Some(p) => joints.push(JointPlan { tangent: false, point: p, next_start: p, round: None }),
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
        // Result geometry per segment: ends from the joints, free ends at
        // the source ends' offsets.
        let mut results = Vec::with_capacity(n);
        for (i, seg) in segs.iter().enumerate() {
            // A closed single entity has no joints and no ends to place.
            let lone_closed = seq.closed && n == 1;
            let start_pt = if lone_closed {
                seg.start()
            } else if i == 0 && !seq.closed {
                seg.start() + seg.left_normal_at_start() * (sign * dv)
            } else {
                joints[(i + n - 1) % n].next_start
            };
            let end_pt = if lone_closed {
                seg.end()
            } else if i + 1 == n && !seq.closed {
                seg.end() + seg.left_normal_at_end() * (sign * dv)
            } else {
                joints[i].point
            };
            let name = || chain::entity_name(sketch, seg.src.entity);
            let geom = match (seg.geom, curves[i]) {
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
        sides.push(SidePlan { sign, distance: d, results, joints });
    }
    // Caps: from the `distance` side's free end to the other side's,
    // around the source end.
    let mut caps = Vec::new();
    if params.caps != CapKind::None && sides.len() == 2 {
        let (sa, da) = (sides[0].sign, sides[0].distance.value);
        let (sb, db) = (sides[1].sign, sides[1].distance.value);
        for at_end in [false, true] {
            let seg = if at_end { &segs[n - 1] } else { &segs[0] };
            let (p, nrm, t) = if at_end {
                (seg.end(), seg.left_normal_at_end(), seg.tangent_at_end())
            } else {
                (seg.start(), seg.left_normal_at_start(), seg.tangent_at_start())
            };
            let a = p + nrm * (sa * da);
            let b = p + nrm * (sb * db);
            let geom = match params.caps {
                CapKind::Line => CapGeom::Line { a, b },
                CapKind::Round => {
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

fn last_did(runner: &dyn ActionRunner) -> Result<u32, String> {
    runner.sketch().dimensions.last().map(|d| d.did).ok_or_else(|| "the dimension was not added".to_string())
}

fn name_of(runner: &dyn ActionRunner, e: OffsetEntity) -> String {
    chain::entity_name(runner.sketch(), e)
}

/// Create one result entity in place.
fn create_result(runner: &mut dyn ActionRunner, g: &ResultGeom, src: OffsetEntity) -> Result<OffsetEntity, String> {
    let sketch = runner.sketch();
    let (construction, style) = match src {
        OffsetEntity::Line(l) => (sketch.lines[l].construction, sketch.lines[l].style),
        OffsetEntity::Arc(a) => (sketch.arcs[a].construction, sketch.arcs[a].style),
    };
    let created = match *g {
        ResultGeom::Line { p1, p2 } => runner.run(Action::AddLine { p1, p2 }),
        ResultGeom::Arc { center, radius, start, end, ccw, closed } => {
            if closed {
                runner.run(Action::AddCircle { center, edge: vect2d::new(center.x + radius, center.y) })
            } else {
                let at = |t: f64| vect2d::new(center.x + radius * t.cos(), center.y + radius * t.sin());
                let _ = ccw;
                runner.run(Action::AddArc { start: at(start), end: at(end), mid: at(0.5 * (start + end)) })
            }
        }
        ResultGeom::Ellipse { center, rx, ry, rotation, start, end, ccw, closed } => {
            if closed {
                runner.run(Action::AddEllipse { center, rx, ry, rotation })
            } else {
                runner.run(Action::AddEllipticArc { center, rx, ry, rotation, start, end, ccw })
            }
        }
    };
    if let Some(e) = runner.take_error() {
        return Err(format!("creating the offset of {}: {}", name_of(runner, src), e));
    }
    let entity = match created {
        crate::actions::Created::Line(l) => OffsetEntity::Line(l),
        crate::actions::Created::Arc(a) => OffsetEntity::Arc(a),
        _ => return Err(format!("creating the offset of {}: nothing was added", name_of(runner, src))),
    };
    if construction {
        match entity {
            OffsetEntity::Line(l) => { runner.run(Action::SetConstructionLine { line: l, on: true }); }
            OffsetEntity::Arc(a) => { runner.run(Action::SetConstructionArc { arc: a, on: true }); }
        }
    }
    if style != LineStyle::Solid {
        match entity {
            OffsetEntity::Line(l) => { runner.run(Action::SetStyleLine { line: l, style }); }
            OffsetEntity::Arc(a) => { runner.run(Action::SetStyleArc { arc: a, style }); }
        }
    }
    Ok(entity)
}

/// Pin the free ends of an open sequence's result on the source ends'
/// normals. Returns the nids.
fn pin_free_ends(runner: &mut dyn ActionRunner, seq: &Sequence, segs: &[OffsetEntity]) -> Result<Vec<u32>, String> {
    let n = seq.segs.len();
    let mut pins = Vec::new();
    if seq.closed || n == 0 {
        return Ok(pins);
    }
    let first = &seq.segs[0];
    if matches!(first.entity, OffsetEntity::Arc(a) if runner.sketch().arcs[a].closed) {
        return Ok(pins);
    }
    let placed = endpoint_of(segs[0], !exit_is_end(first));
    let reference = endpoint_of(first.entity, !exit_is_end(first));
    run_checked(runner, Action::ApplyOnNormal { placed, reference }, "on_normal")?;
    pins.push(last_nid(runner));
    let last = &seq.segs[n - 1];
    let placed = endpoint_of(segs[n - 1], exit_is_end(last));
    let reference = endpoint_of(last.entity, exit_is_end(last));
    run_checked(runner, Action::ApplyOnNormal { placed, reference }, "on_normal")?;
    pins.push(last_nid(runner));
    Ok(pins)
}

/// The optional pins of an existing side, all at once: the tangent
/// joints' exit ends and (unless round caps hold them) the free ends.
/// For turning `pin` on afterwards.
fn pin_side(
    runner: &mut dyn ActionRunner,
    seq: &Sequence,
    segs: &[OffsetEntity],
    joints: &[JointPlan],
    free_ends: bool,
) -> Result<Vec<u32>, String> {
    let mut pins = Vec::new();
    for (i, joint) in joints.iter().enumerate() {
        if !joint.tangent {
            continue;
        }
        let a_src = &seq.segs[i];
        let placed = endpoint_of(segs[i], exit_is_end(a_src));
        let reference = endpoint_of(a_src.entity, exit_is_end(a_src));
        run_checked(runner, Action::ApplyOnNormal { placed, reference }, "on_normal")?;
        pins.push(last_nid(runner));
    }
    if free_ends {
        pins.extend(pin_free_ends(runner, seq, segs)?);
    }
    Ok(pins)
}

/// The free end of a side's result at the sequence's start / end.
fn free_end_of(seq: &Sequence, segs: &[OffsetEntity], at_end: bool) -> DimensionEndpoint {
    let n = seq.segs.len();
    if at_end {
        endpoint_of(segs[n - 1], exit_is_end(&seq.segs[n - 1]))
    } else {
        endpoint_of(segs[0], !exit_is_end(&seq.segs[0]))
    }
}

/// Create the caps of a two-sided open offset: a line across each end,
/// joined to both results; or a half circle around the source end,
/// joined and tangent to both results (which makes its radius the
/// distance) and held in place by a pin of the `distance` side's end on
/// the source end's normal.
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
    let side_a = sides.iter().find(|s| s.sign == plan.sides[0].sign).ok_or("caps: the first side is missing")?;
    let side_b = sides.iter().find(|s| s.sign == plan.sides[1].sign).ok_or("caps: the second side is missing")?;
    let mut names = Vec::new();
    for cap in &plan.caps {
        let end_a = free_end_of(seq, &side_a.segs, cap.at_end);
        let end_b = free_end_of(seq, &side_b.segs, cap.at_end);
        let res_a = if cap.at_end { *side_a.segs.last().unwrap() } else { side_a.segs[0] };
        let res_b = if cap.at_end { *side_b.segs.last().unwrap() } else { side_b.segs[0] };
        let (entity, own_start, own_end) = match cap.geom {
            CapGeom::Line { a, b } => {
                let created = runner.run(Action::AddLine { p1: a, p2: b });
                if let Some(e) = runner.take_error() {
                    return Err(format!("cap: {}", e));
                }
                let crate::actions::Created::Line(l) = created else {
                    return Err("cap: the line was not added".into());
                };
                (OffsetEntity::Line(l), DimensionEndpoint::LineP1(l), DimensionEndpoint::LineP2(l))
            }
            CapGeom::Arc { start, end, mid } => {
                let created = runner.run(Action::AddArc { start, end, mid });
                if let Some(e) = runner.take_error() {
                    return Err(format!("cap: {}", e));
                }
                let crate::actions::Created::Arc(a) = created else {
                    return Err("cap: the arc was not added".into());
                };
                (OffsetEntity::Arc(a), DimensionEndpoint::ArcStart(a), DimensionEndpoint::ArcEnd(a))
            }
        };
        caps.entities.push(entity);
        names.push(name_of(runner, entity));
        let c1 = Action::coincident(own_start, end_a).expect("endpoint pair");
        run_checked(runner, c1, "coincident")?;
        caps.constraints.push(last_nid(runner));
        let c2 = Action::coincident(own_end, end_b).expect("endpoint pair");
        run_checked(runner, c2, "coincident")?;
        caps.constraints.push(last_nid(runner));
        if let (CapGeom::Arc { .. }, OffsetEntity::Arc(arc)) = (cap.geom, entity) {
            run_checked(runner, tangent_action(res_a, arc), "tangent")?;
            caps.constraints.push(last_nid(runner));
            run_checked(runner, tangent_action(res_b, arc), "tangent")?;
            caps.constraints.push(last_nid(runner));
            let n = seq.segs.len();
            let src = if cap.at_end { &seq.segs[n - 1] } else { &seq.segs[0] };
            let reference = endpoint_of(src.entity, cap.at_end == exit_is_end(src));
            run_checked(runner, Action::ApplyOnNormal { placed: end_a, reference }, "on_normal")?;
            caps.constraints.push(last_nid(runner));
        }
    }
    out.entities.push(names);
    out.constraints.extend(caps.constraints.iter().map(|n| format!("C{}", n)));
    Ok(caps)
}

/// Create one side's result with its relations, joints and pins.
fn apply_side(
    runner: &mut dyn ActionRunner,
    plan: &OffsetPlan,
    side: &SidePlan,
    out: &mut OffsetOutcome,
) -> Result<OffsetSideResult, String> {
    let seq = &plan.seq;
    let n = seq.segs.len();
    let mut segs = Vec::with_capacity(n);
    let mut constraints = Vec::new();
    let mut pins = Vec::new();
    let mut dims = Vec::new();
    let mut names = Vec::with_capacity(n);

    // Entities first, then the per-segment relations.
    for (i, g) in side.results.iter().enumerate() {
        let e = create_result(runner, g, seq.segs[i].entity)?;
        names.push(name_of(runner, e));
        segs.push(e);
    }
    for (i, src) in seq.segs.iter().enumerate() {
        let res = segs[i];
        match (src.entity, res) {
            (OffsetEntity::Line(s), OffsetEntity::Line(r)) => {
                run_checked(runner, Action::ApplyParallel { a: s, b: r }, "parallel")?;
                constraints.push(last_nid(runner));
                runner.run(Action::AddDimension {
                    kind: DimensionKind::LineLineDistance(s, r),
                    value: side.distance.value,
                    expr: side.distance.expr.clone(),
                    derived: false,
                    range: None,
                });
                if let Some(e) = runner.take_error() { return Err(format!("distance: {}", e)); }
                let did = last_did(runner)?;
                dims.push(OffsetDim { did, expect: side.distance.clone() });
            }
            (OffsetEntity::Arc(s), OffsetEntity::Arc(r)) => {
                run_checked(runner, Action::ApplyConcentric { a: s, b: r }, "concentric")?;
                constraints.push(last_nid(runner));
                if runner.sketch().arcs[s].is_ellipse {
                    run_checked(runner, Action::ApplyArcArcParallel { a: s, b: r }, "parallel")?;
                    constraints.push(last_nid(runner));
                }
                runner.run(Action::AddDimension {
                    kind: DimensionKind::ConcentricDistance(s, r),
                    value: side.distance.value,
                    expr: side.distance.expr.clone(),
                    derived: false,
                    range: None,
                });
                if let Some(e) = runner.take_error() { return Err(format!("concentric distance: {}", e)); }
                let did = last_did(runner)?;
                dims.push(OffsetDim { did, expect: side.distance.clone() });
            }
            _ => unreachable!("a result has its source's kind"),
        }
    }
    // Joints, in chain order: a pin on the earlier result's exit end where
    // the sources are tangent, then the coincidence. A round corner is a
    // fillet between the two results: an arc of the distance joined to
    // both ends, tangent to both, its radius a dimension; the geometry
    // puts its center on the source joint.
    let mut corners = Vec::new();
    for (i, joint) in side.joints.iter().enumerate() {
        let (a_src, b_src) = (&seq.segs[i], &seq.segs[(i + 1) % n]);
        let (a_res, b_res) = (segs[i], segs[(i + 1) % n]);
        let a_end = endpoint_of(a_res, exit_is_end(a_src));
        let b_start = endpoint_of(b_res, !exit_is_end(b_src));
        let a_ref = endpoint_of(a_src.entity, exit_is_end(a_src));
        if let Some(rc) = &joint.round {
            let created = runner.run(Action::AddArc { start: rc.start, end: rc.end, mid: rc.mid });
            if let Some(e) = runner.take_error() {
                return Err(format!("round corner: {}", e));
            }
            let crate::actions::Created::Arc(arc) = created else {
                return Err("round corner: the arc was not added".into());
            };
            corners.push(OffsetEntity::Arc(arc));
            names.push(name_of(runner, OffsetEntity::Arc(arc)));
            let c1 = Action::coincident(DimensionEndpoint::ArcStart(arc), a_end).expect("endpoint pair");
            run_checked(runner, c1, "coincident")?;
            constraints.push(last_nid(runner));
            let c2 = Action::coincident(DimensionEndpoint::ArcEnd(arc), b_start).expect("endpoint pair");
            run_checked(runner, c2, "coincident")?;
            constraints.push(last_nid(runner));
            run_checked(runner, tangent_action(a_res, arc), "tangent")?;
            constraints.push(last_nid(runner));
            run_checked(runner, tangent_action(b_res, arc), "tangent")?;
            constraints.push(last_nid(runner));
            runner.run(Action::AddDimension {
                kind: DimensionKind::ArcRadius(arc),
                value: side.distance.value,
                expr: side.distance.expr.clone(),
                derived: false,
                range: None,
            });
            if let Some(e) = runner.take_error() { return Err(format!("radius: {}", e)); }
            let did = last_did(runner)?;
            dims.push(OffsetDim { did, expect: side.distance.clone() });
            continue;
        }
        if joint.tangent && plan.params.pinned {
            run_checked(runner, Action::ApplyOnNormal { placed: a_end, reference: a_ref }, "on_normal")?;
            pins.push(last_nid(runner));
        }
        let action = Action::coincident(a_end, b_start).expect("endpoint pair");
        run_checked(runner, action, "coincident")?;
        constraints.push(last_nid(runner));
    }
    // Round caps hold the free ends themselves (tangent to a half circle
    // around the source end): no pins there.
    if plan.params.pinned && plan.params.caps != CapKind::Round {
        pins.extend(pin_free_ends(runner, seq, &segs)?);
    }
    out.entities.push(names);
    out.constraints.extend(constraints.iter().chain(pins.iter()).map(|n| format!("C{}", n)));
    let sketch = runner.sketch();
    for d in &dims {
        if let Some(i) = sketch.dimension_index_by_did(d.did) {
            out.dims.push(sketch.dimensions[i].name.clone());
        }
    }
    Ok(OffsetSideResult { sign: side.sign, segs, corners, constraints, pins, dims })
}

/// Create the planned offset and register its meta-constraint. Everything
/// runs inside the runner's current undo group.
pub fn apply(runner: &mut dyn ActionRunner, plan: &OffsetPlan) -> Result<OffsetOutcome, String> {
    let mut out = OffsetOutcome { approximate: plan.approximate, ..Default::default() };
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
    for (i, &e) in side.segs.iter().enumerate() {
        reseed(sketch, e, &plan_side.results[i]);
    }
    for (i, src) in o.source.iter().enumerate() {
        if let (OffsetEntity::Arc(s), OffsetEntity::Arc(r)) = (src.entity, side.segs[i]) {
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
/// any side it created.
pub fn update(runner: &mut dyn ActionRunner, mid: u32, params: &OffsetParams) -> Result<OffsetOutcome, String> {
    let (name, o) = offset_record(runner.sketch(), mid)?;
    let seq = sequence_of(&o);
    let new_plan = plan(runner.sketch(), &seq, params)?;
    let mut out = OffsetOutcome { name, mid, approximate: new_plan.approximate, ..Default::default() };
    runner.begin_group();

    let register = |runner: &mut dyn ActionRunner, o: Offset| -> Result<(), String> {
        let name = runner.sketch().metas[runner.sketch().meta_index(mid).expect("registered")].name.clone();
        runner.run(Action::RegisterMeta { meta: Meta { mid, name, kind: MetaKind::Offset(o) } });
        runner.take_error().map_or(Ok(()), Err)
    };

    // Pair the existing sides with the wanted ones: by sign where a sign
    // survives, else an existing side takes over a wanted sign (a flip
    // moves the one side across), and whatever is left over is deleted
    // or created. A side survives only when its round corners are the
    // ones the new plan wants there (which joints get an arc depends on
    // the side and on `round`); otherwise it is rebuilt.
    let wanted: Vec<(f64, OffsetValue)> = params.sides();
    let planned_corners = |sign: f64| -> usize {
        new_plan
            .sides
            .iter()
            .find(|p| p.sign == sign)
            .map_or(0, |p| p.joints.iter().filter(|j| j.round.is_some()).count())
    };
    let mut remaining = wanted.clone();
    let mut kept: Vec<OffsetSideResult> = Vec::new();
    let mut unmatched: Vec<OffsetSideResult> = Vec::new();
    for s in &o.sides {
        match remaining.iter().position(|(sign, _)| *sign == s.sign) {
            Some(pos) if s.corners.len() == planned_corners(s.sign) => {
                remaining.remove(pos);
                kept.push(s.clone());
            }
            _ => unmatched.push(s.clone()),
        }
    }
    let mut doomed: Vec<OffsetEntity> = Vec::new();
    for mut s in unmatched {
        let movable = !remaining.is_empty() && s.corners.is_empty() && planned_corners(remaining[0].0) == 0;
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
    for e in doomed.into_iter().chain(doomed_caps) {
        match e {
            OffsetEntity::Line(l) => { runner.run(Action::DeleteLine { line: l }); }
            OffsetEntity::Arc(a) => { runner.run(Action::DeleteArc { arc: a }); }
        }
        if let Some(e) = runner.take_error() { return Err(e); }
    }
    // A cap's pin is between results and sources: it does not go with
    // the cap entity.
    for nid in doomed_cap_constraints {
        if !crate::meta::nid_exists(runner.sketch(), nid) { continue; }
        runner.run(Action::DeleteConstraint { id: crate::ids::ConstraintId::Numbered(nid) });
        if let Some(e) = runner.take_error() { return Err(e); }
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
        for nid in old_pins {
            runner.run(Action::DeleteConstraint { id: crate::ids::ConstraintId::Numbered(nid) });
            if let Some(e) = runner.take_error() { return Err(e); }
        }
        if params.pinned {
            for s in rec.sides.iter_mut() {
                let plan_side = new_plan.sides.iter().find(|p| p.sign == s.sign).expect("planned");
                s.pins = pin_side(runner, &seq, &s.segs, &plan_side.joints, !round_caps(params.caps))?;
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
        created_sides.push(apply_side(runner, &new_plan, plan_side, &mut out)?);
    }
    if !created_sides.is_empty() {
        let (_, mut rec) = offset_record(runner.sketch(), mid)?;
        rec.sides.extend(created_sides);
        register(runner, rec)?;
    }
    // New caps, across the final sides.
    if rebuild_caps && !new_plan.caps.is_empty() {
        let (_, mut rec) = offset_record(runner.sketch(), mid)?;
        rec.caps = apply_caps(runner, &new_plan, &rec.sides, &mut out)?;
        register(runner, rec)?;
    }
    Ok(out)
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
            format!("{}: {}", side_name(s.sign), names_of(&es))
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

fn fmt_value(v: &OffsetValue) -> String {
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
pub fn parse_value(sketch: &Sketch, token: &str) -> Result<OffsetValue, String> {
    let token = token.trim().trim_matches('"');
    if let Ok(v) = token.parse::<f64>() {
        return Ok(OffsetValue { value: v, expr: None });
    }
    let (src, live) = match token.strip_prefix('=') {
        Some(rest) => (rest.trim(), false),
        None => (token, true),
    };
    let value = crate::commands::eval_expr(sketch, src)
        .map_err(|e| format!("cannot evaluate '{}': {}", src, e))?;
    Ok(OffsetValue { value, expr: if live { Some(src.to_string()) } else { None } })
}

/// Reference helpers for callers holding a `Ref`.
pub fn line_entity(r: Ref<Line>) -> OffsetEntity { OffsetEntity::Line(r) }
pub fn arc_entity(r: Ref<Arc>) -> OffsetEntity { OffsetEntity::Arc(r) }
