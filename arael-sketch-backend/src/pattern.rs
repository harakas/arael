//! The pattern engine (docs/dev/PATTERN.md): plan circular / rectangular
//! copies of a set of entities, create them through an [`ActionRunner`]
//! with the image constraints and coincidences that hold every copy as a
//! rigid image of its source, record the pattern meta-constraint, and
//! edit it.
//!
//! A copy has no freedom of its own: every parameter of a copy entity is
//! determined exactly once, by an image row (`ImageLine*` / `ImageArc*` /
//! `ImagePoint*`, masked to the rows needed) or by a coincidence
//! recreated from the source (so a copy stays a connected shape). Which
//! rows a copy entity images is decided structurally from the source's
//! coincidence graph (`tie_plan`), never by a rank test.

use std::collections::HashMap;
use std::f64::consts::TAU;

use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::actions::{Action, Created, Xf};
use crate::coincide::CoincidenceGroups;
use crate::corner_ops::ActionRunner;
use crate::meta::{entity_exists, entity_name};

/// What the user asked for.
#[derive(Clone, Debug, PartialEq)]
pub struct PatternParams {
    pub kind: PatternSpec,
}

/// The pattern's kind and numbers (the record's `PatternKind` without
/// the helper it creates).
#[derive(Clone, Debug, PartialEq)]
pub enum PatternSpec {
    Circular { center: CenterRef, distribution: Distribution, angle: MetaValue, quantity: u32 },
    Rectangular { frame: Option<Ref<Line>>, extent: bool, axis1: PatternAxis, axis2: PatternAxis },
}

/// Instance offsets along one axis: `k` from `-floor((q-1)/2)` to
/// `ceil((q-1)/2)` when symmetric (one less on the backward side for an
/// even `q`), `0..q-1` otherwise; the source is 0.
pub fn axis_indices(quantity: u32, symmetric: bool) -> Vec<i32> {
    let q = quantity.max(1) as i32;
    if symmetric {
        let back = (q - 1) / 2;
        let fwd = q - 1 - back;
        (-back..=fwd).collect()
    } else {
        (0..q).collect()
    }
}

/// The step between consecutive instances.
fn step_of(distance: f64, quantity: u32, extent: bool) -> f64 {
    if extent && quantity > 1 { distance / (quantity - 1) as f64 } else { distance }
}

// ---------------------------------------------------------------------------
// Geometry of one copy
// ---------------------------------------------------------------------------

/// A rigid transform with its numbers resolved against the current
/// geometry (the frame line's direction, the center's position).
#[derive(Clone, Copy, Debug)]
pub struct Motion {
    /// Translation of a position, after the rotation.
    pub tx: f64,
    pub ty: f64,
    /// Rotation angle (radians, counter-clockwise); zero for translations.
    pub angle: f64,
    /// Rotation center.
    pub cx: f64,
    pub cy: f64,
}

impl Motion {
    pub fn point(&self, p: vect2d) -> vect2d {
        if self.angle == 0.0 {
            return vect2d::new(p.x + self.tx, p.y + self.ty);
        }
        let (s, c) = self.angle.sin_cos();
        let (ux, uy) = (p.x - self.cx, p.y - self.cy);
        vect2d::new(self.cx + c * ux - s * uy + self.tx, self.cy + s * ux + c * uy + self.ty)
    }
}

/// One copy entity's geometry, ready to create.
#[derive(Clone, Copy, Debug)]
pub enum CopyGeom {
    Line { p1: vect2d, p2: vect2d },
    Arc { center: vect2d, radius: f64, radius_b: f64, rotation: f64, start: f64, end: f64, ccw: bool, closed: bool, is_ellipse: bool },
    Point { pos: vect2d },
}

/// The geometry of the source entity moved by `m`.
pub fn moved(sketch: &Sketch, e: MetaEntity, m: &Motion) -> CopyGeom {
    match e {
        MetaEntity::Line(l) => {
            let l = &sketch.lines[l];
            CopyGeom::Line { p1: m.point(l.p1.value), p2: m.point(l.p2.value) }
        }
        MetaEntity::Arc(a) => {
            let a = &sketch.arcs[a];
            CopyGeom::Arc {
                center: m.point(a.center.value),
                radius: a.radius.value,
                radius_b: if a.is_ellipse { a.radius_b.value } else { a.radius.value },
                rotation: if a.is_ellipse { a.rotation.value + m.angle } else { 0.0 },
                start: a.start_angle.value + m.angle,
                end: a.end_angle.value + m.angle,
                ccw: a.ccw,
                closed: a.closed,
                is_ellipse: a.is_ellipse,
            }
        }
        MetaEntity::Point(p) => CopyGeom::Point { pos: m.point(sketch.points[p].pos.value) },
    }
}

/// Points along a copy's geometry, for the preview.
pub fn sample(g: &CopyGeom, n: usize) -> Vec<vect2d> {
    match *g {
        CopyGeom::Line { p1, p2 } => vec![p1, p2],
        CopyGeom::Point { pos } => vec![pos],
        CopyGeom::Arc { center, radius, radius_b, rotation, start, end, closed, ccw, .. } => {
            let (s, e) = if closed {
                (0.0, TAU)
            } else {
                // Sweep in the travel direction (the stored angles may wrap).
                let sweep = if ccw { (end - start).rem_euclid(TAU) } else { -((start - end).rem_euclid(TAU)) };
                (start, start + sweep)
            };
            let (cr, sr) = (rotation.cos(), rotation.sin());
            (0..=n)
                .map(|i| {
                    let t = s + (e - s) * i as f64 / n as f64;
                    let (x, y) = (radius * t.cos(), radius_b * t.sin());
                    vect2d::new(center.x + x * cr - y * sr, center.y + x * sr + y * cr)
                })
                .collect()
        }
    }
}

// ---------------------------------------------------------------------------
// The plan
// ---------------------------------------------------------------------------

/// How one source entity is held in every copy: the image rows, and the
/// coincidences recreated (this entity's point, the source point already
/// placed that it is coincident with).
#[derive(Clone, Debug)]
pub struct TiePlan {
    pub mask: u8,
    pub ties: Vec<(DimensionEndpoint, DimensionEndpoint)>,
}

#[derive(Clone, Debug)]
pub struct CopyPlan {
    /// (axis 1, axis 2), or (step, 0) for a circular pattern.
    pub index: (i32, i32),
    pub motion: Motion,
    /// One per source.
    pub geoms: Vec<CopyGeom>,
}

#[derive(Clone, Debug)]
pub struct PatternPlan {
    pub sources: Vec<MetaEntity>,
    pub params: PatternParams,
    /// Position of the rotation center (circular).
    pub center: Option<vect2d>,
    pub ties: Vec<TiePlan>,
    pub copies: Vec<CopyPlan>,
}

/// The position of a point-like reference.
fn endpoint_pos(sketch: &Sketch, e: DimensionEndpoint) -> vect2d {
    crate::actions::dim_endpoint_pos_sketch(sketch, &e)
}

pub(crate) fn endpoint_entity(e: DimensionEndpoint) -> MetaEntity {
    match e {
        DimensionEndpoint::Point(p) => MetaEntity::Point(p),
        DimensionEndpoint::LineP1(l) | DimensionEndpoint::LineP2(l) => MetaEntity::Line(l),
        DimensionEndpoint::ArcCenter(a) | DimensionEndpoint::ArcStart(a) | DimensionEndpoint::ArcEnd(a) => MetaEntity::Arc(a),
    }
}

/// The frame's unit vector along it and across it (left).
fn frame_axes(sketch: &Sketch, frame: Option<Ref<Line>>) -> Result<(vect2d, vect2d), String> {
    match frame {
        None => Ok((vect2d::new(1.0, 0.0), vect2d::new(0.0, 1.0))),
        Some(l) => {
            let l = &sketch.lines[l];
            let d = l.p2.value - l.p1.value;
            let len = (d.x * d.x + d.y * d.y).sqrt();
            if len < 1e-12 {
                return Err("the direction line has no length".into());
            }
            let u = vect2d::new(d.x / len, d.y / len);
            Ok((u, vect2d::new(-u.y, u.x)))
        }
    }
}

/// The structural decision: which rows each source entity images in a
/// copy and which coincidences the copy gets, walking the set in order
/// over the source's coincidence graph.
fn tie_plan(sketch: &Sketch, sources: &[MetaEntity]) -> Vec<TiePlan> {
    use image_rows::*;
    let mut groups = CoincidenceGroups::build(sketch);
    // group root -> a source point already placed in the copy
    let mut placed: HashMap<usize, DimensionEndpoint> = HashMap::new();
    let mut plans = Vec::with_capacity(sources.len());
    for &e in sources {
        let mut ties = Vec::new();
        let mut root_of = |ep: DimensionEndpoint| -> usize {
            let slot = match ep {
                DimensionEndpoint::Point(p) => groups.pt(p),
                DimensionEndpoint::LineP1(l) => groups.lp1(l),
                DimensionEndpoint::LineP2(l) => groups.lp2(l),
                DimensionEndpoint::ArcCenter(a) => groups.arc_center(a),
                DimensionEndpoint::ArcStart(a) => groups.arc_start(a),
                DimensionEndpoint::ArcEnd(a) => groups.arc_end(a),
            };
            groups.find(slot)
        };
        let mask = match e {
            MetaEntity::Line(l) => {
                let mut mask = 0u8;
                for (ep, bit) in [(DimensionEndpoint::LineP1(l), P1), (DimensionEndpoint::LineP2(l), P2)] {
                    let root = root_of(ep);
                    match placed.get(&root) {
                        Some(&other) => ties.push((ep, other)),
                        None => {
                            mask |= bit;
                            placed.insert(root, ep);
                        }
                    }
                }
                mask
            }
            MetaEntity::Point(p) => {
                let ep = DimensionEndpoint::Point(p);
                let root = root_of(ep);
                match placed.get(&root) {
                    Some(&other) => {
                        ties.push((ep, other));
                        0
                    }
                    None => {
                        placed.insert(root, ep);
                        P1
                    }
                }
            }
            MetaEntity::Arc(a) => {
                let arc = &sketch.arcs[a];
                let ellipse = arc.is_ellipse;
                let shape = if ellipse { RADIUS | RADIUS_B | ROTATION } else { RADIUS };
                let c_ep = DimensionEndpoint::ArcCenter(a);
                let c_root = root_of(c_ep);
                let c = placed.get(&c_root).copied();
                let (s, e_) = if arc.closed {
                    (None, None)
                } else {
                    (
                        placed.get(&root_of(DimensionEndpoint::ArcStart(a))).copied(),
                        placed.get(&root_of(DimensionEndpoint::ArcEnd(a))).copied(),
                    )
                };
                let mask = if arc.closed {
                    if c.is_some() { shape } else { CENTER | shape }
                } else {
                    match (c.is_some(), s.is_some(), e_.is_some()) {
                        (false, false, false) => CENTER | shape | START | END,
                        (true, false, false) | (false, true, false) | (false, false, true) => shape | START | END,
                        (false, true, true) => shape,
                        // With the center and one end held, the radius (and
                        // the held angle) follow; the other angle is imaged.
                        // An ellipse keeps radius_b and rotation imaged.
                        (true, true, false) | (true, true, true) => if ellipse { RADIUS_B | ROTATION | END } else { END },
                        (true, false, true) => if ellipse { RADIUS_B | ROTATION | START } else { START },
                    }
                };
                if let Some(other) = c {
                    ties.push((c_ep, other));
                } else {
                    placed.insert(c_root, c_ep);
                }
                if !arc.closed {
                    let s_ep = DimensionEndpoint::ArcStart(a);
                    let e_ep = DimensionEndpoint::ArcEnd(a);
                    match s {
                        Some(other) => ties.push((s_ep, other)),
                        None => { placed.insert(root_of(s_ep), s_ep); }
                    }
                    match e_ {
                        // A third tie would be redundant: the end meets by geometry.
                        Some(other) if !(c.is_some() && s.is_some()) => ties.push((e_ep, other)),
                        Some(_) => {}
                        None => { placed.insert(root_of(e_ep), e_ep); }
                    }
                }
                mask
            }
        };
        plans.push(TiePlan { mask, ties });
    }
    plans
}

/// Check the parameters and lay out every copy.
pub fn plan(sketch: &Sketch, sources: &[MetaEntity], params: &PatternParams) -> Result<PatternPlan, String> {
    if sources.is_empty() {
        return Err("nothing to pattern".into());
    }
    for &e in sources {
        if !entity_exists(sketch, e) {
            return Err("a source entity no longer exists".into());
        }
    }
    let mut copies = Vec::new();
    let mut center_pos = None;
    match &params.kind {
        PatternSpec::Circular { center, distribution, angle, quantity } => {
            let q = *quantity;
            if q < 2 {
                return Err("a circular pattern needs a quantity of at least 2".into());
            }
            let (cpos, center_entity) = match center {
                CenterRef::Point(p) => {
                    if sketch.points.get(*p).is_none() {
                        return Err("the center point no longer exists".into());
                    }
                    (sketch.points[*p].pos.value, MetaEntity::Point(*p))
                }
                CenterRef::Endpoint(ep) => {
                    if !entity_exists(sketch, endpoint_entity(*ep)) {
                        return Err("the center's entity no longer exists".into());
                    }
                    (endpoint_pos(sketch, *ep), endpoint_entity(*ep))
                }
            };
            if let CenterRef::Point(_) = center
                && sources.contains(&center_entity)
            {
                return Err("the center point cannot be in the pattern set".into());
            }
            center_pos = Some(cpos);
            let step = match distribution {
                Distribution::Full => TAU / q as f64,
                Distribution::Partial | Distribution::Symmetric => {
                    if angle.value == 0.0 || !angle.value.is_finite() {
                        return Err("the pattern angle must not be zero".into());
                    }
                    angle.value.to_radians() / (q - 1) as f64
                }
            };
            for k in axis_indices(q, *distribution == Distribution::Symmetric) {
                if k == 0 {
                    continue;
                }
                let motion = Motion { tx: 0.0, ty: 0.0, angle: k as f64 * step, cx: cpos.x, cy: cpos.y };
                let geoms = sources.iter().map(|&e| moved(sketch, e, &motion)).collect();
                copies.push(CopyPlan { index: (k, 0), motion, geoms });
            }
        }
        PatternSpec::Rectangular { frame, extent, axis1, axis2 } => {
            if let Some(l) = frame
                && sources.contains(&MetaEntity::Line(*l))
            {
                return Err("the direction line cannot be in the pattern set".into());
            }
            if axis1.quantity < 2 && axis2.quantity < 2 {
                return Err("a rectangular pattern needs a quantity of at least 2 on an axis".into());
            }
            for (ax, name) in [(axis1, "axis 1"), (axis2, "axis 2")] {
                if ax.quantity >= 2 && (ax.distance.value == 0.0 || !ax.distance.value.is_finite()) {
                    return Err(format!("the distance on {} must not be zero", name));
                }
            }
            let (u, v) = frame_axes(sketch, *frame)?;
            let (s1, s2) = (step_of(axis1.distance.value, axis1.quantity, *extent), step_of(axis2.distance.value, axis2.quantity, *extent));
            for i in axis_indices(axis1.quantity, axis1.symmetric) {
                for j in axis_indices(axis2.quantity, axis2.symmetric) {
                    if i == 0 && j == 0 {
                        continue;
                    }
                    let (dx, dy) = (i as f64 * s1, j as f64 * s2);
                    let motion = Motion { tx: dx * u.x + dy * v.x, ty: dx * u.y + dy * v.y, angle: 0.0, cx: 0.0, cy: 0.0 };
                    let geoms = sources.iter().map(|&e| moved(sketch, e, &motion)).collect();
                    copies.push(CopyPlan { index: (i, j), motion, geoms });
                }
            }
        }
    }
    if copies.is_empty() {
        return Err("the pattern makes no copies".into());
    }
    Ok(PatternPlan { sources: sources.to_vec(), params: params.clone(), center: center_pos, ties: tie_plan(sketch, sources), copies })
}

/// Every polyline of the plan, for the preview.
pub fn preview_polylines(plan: &PatternPlan, n: usize) -> Vec<Vec<vect2d>> {
    plan.copies.iter().flat_map(|c| c.geoms.iter().map(|g| sample(g, n))).collect()
}

// ---------------------------------------------------------------------------
// Applying a plan
// ---------------------------------------------------------------------------

/// What a pattern created, for the command output.
#[derive(Clone, Debug, Default)]
pub struct PatternOutcome {
    pub name: String,
    pub mid: u32,
    /// Per copy: entity names.
    pub entities: Vec<Vec<String>>,
    pub constraints: Vec<String>,
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

/// The creation action for one copy entity.
fn create_action(g: &CopyGeom) -> Action {
    match *g {
        CopyGeom::Line { p1, p2 } => Action::AddLine { p1, p2 },
        CopyGeom::Point { pos } => Action::AddPoint { pos },
        CopyGeom::Arc { center, radius, radius_b, rotation, start, end, ccw, closed, is_ellipse } => {
            if is_ellipse {
                if closed {
                    Action::AddEllipse { center, rx: radius, ry: radius_b, rotation }
                } else {
                    Action::AddEllipticArc { center, rx: radius, ry: radius_b, rotation, start, end, ccw }
                }
            } else if closed {
                Action::AddCircle { center, edge: vect2d::new(center.x + radius, center.y) }
            } else {
                Action::AddArcAngles { center, radius, start, end, ccw }
            }
        }
    }
}

/// The copy's counterpart of a source point reference.
pub(crate) fn map_endpoint(ep: DimensionEndpoint, sources: &[MetaEntity], copy: &[MetaEntity]) -> Option<DimensionEndpoint> {
    let idx = sources.iter().position(|&s| s == endpoint_entity(ep))?;
    Some(match (ep, copy[idx]) {
        (DimensionEndpoint::Point(_), MetaEntity::Point(p)) => DimensionEndpoint::Point(p),
        (DimensionEndpoint::LineP1(_), MetaEntity::Line(l)) => DimensionEndpoint::LineP1(l),
        (DimensionEndpoint::LineP2(_), MetaEntity::Line(l)) => DimensionEndpoint::LineP2(l),
        (DimensionEndpoint::ArcCenter(_), MetaEntity::Arc(a)) => DimensionEndpoint::ArcCenter(a),
        (DimensionEndpoint::ArcStart(_), MetaEntity::Arc(a)) => DimensionEndpoint::ArcStart(a),
        (DimensionEndpoint::ArcEnd(_), MetaEntity::Arc(a)) => DimensionEndpoint::ArcEnd(a),
        _ => return None,
    })
}

/// The transform of a copy as the image constraints carry it.
fn xf_of(plan: &PatternPlan, copy: &CopyPlan, center: Option<Ref<Point>>) -> Xf {
    match &plan.params.kind {
        PatternSpec::Circular { .. } => Xf::Rotate { center: center.expect("a circular pattern has a center point"), angle: copy.motion.angle },
        PatternSpec::Rectangular { frame, extent, axis1, axis2 } => {
            let (s1, s2) = (step_of(axis1.distance.value, axis1.quantity, *extent), step_of(axis2.distance.value, axis2.quantity, *extent));
            let (dx, dy) = (copy.index.0 as f64 * s1, copy.index.1 as f64 * s2);
            match frame {
                Some(f) => Xf::TranslateAlong { frame: *f, dx, dy },
                None => Xf::Translate { dx, dy },
            }
        }
    }
}

/// Run a batch and report its error.
fn run_batch(runner: &mut dyn ActionRunner, label: &str, actions: Vec<Action>) -> Result<Created, String> {
    let created = runner.run(Action::Batch { label: label.to_string(), actions });
    match runner.take_error() {
        Some(e) => Err(format!("{}: {}", label, e)),
        None => Ok(created),
    }
}

/// Every copy's entities and constraints, in two batches: all the
/// entities (one solve), then all the image constraints, the recreated
/// coincidences and the construction / style flags (one solve). The
/// constraints are known to be consistent by construction, so they skip
/// the per-constraint gate; a pattern of hundreds of entities is a
/// handful of history entries.
fn make_copies(
    runner: &mut dyn ActionRunner,
    plan: &PatternPlan,
    center: Option<Ref<Point>>,
    out: &mut PatternOutcome,
) -> Result<Vec<PatternCopy>, String> {
    let n = plan.sources.len();
    // Phase 1: the entities.
    let creates: Vec<Action> = plan.copies.iter().flat_map(|c| c.geoms.iter().map(create_action)).collect();
    let created = run_batch(runner, "Pattern copies", creates)?;
    let Created::Many(list) = created else {
        return Err("pattern copies: nothing was added".into());
    };
    if list.len() != plan.copies.len() * n {
        return Err("pattern copies: not every entity was added".into());
    }
    let mut entities_per_copy: Vec<Vec<MetaEntity>> = Vec::with_capacity(plan.copies.len());
    let mut it = list.into_iter();
    for copy in &plan.copies {
        let mut entities = Vec::with_capacity(n);
        for (k, _) in copy.geoms.iter().enumerate() {
            let e = match it.next() {
                Some(Created::Line(l)) => MetaEntity::Line(l),
                Some(Created::Arc(a)) => MetaEntity::Arc(a),
                Some(Created::Point(p)) => MetaEntity::Point(p),
                _ => return Err(format!("copying {}: nothing was added", entity_name(runner.sketch(), plan.sources[k]))),
            };
            entities.push(e);
        }
        entities_per_copy.push(entities);
    }
    // Phase 2: images, coincidences, flags.
    let n0 = runner.sketch().next_constraint_id;
    let mut acts: Vec<Action> = Vec::new();
    for (copy, entities) in plan.copies.iter().zip(&entities_per_copy) {
        let xf = xf_of(plan, copy, center);
        for (k, tie) in plan.ties.iter().enumerate() {
            if tie.mask != 0 {
                acts.push(match (plan.sources[k], entities[k]) {
                    (MetaEntity::Line(a), MetaEntity::Line(b)) => Action::ApplyImageLine { a, b, xf, mask: tie.mask },
                    (MetaEntity::Arc(a), MetaEntity::Arc(b)) => Action::ApplyImageArc { a, b, xf, mask: tie.mask },
                    (MetaEntity::Point(a), MetaEntity::Point(b)) => Action::ApplyImagePoint { a, b, xf },
                    _ => unreachable!("a copy has its source's kind"),
                });
            }
            for &(own, other) in &tie.ties {
                if let (Some(a), Some(b)) = (map_endpoint(own, &plan.sources, entities), map_endpoint(other, &plan.sources, entities))
                    && let Some(action) = Action::coincident(a, b)
                {
                    acts.push(action);
                }
            }
        }
        for (k, &e) in entities.iter().enumerate() {
            let sketch = runner.sketch();
            let (construction, style) = match plan.sources[k] {
                MetaEntity::Line(l) => (sketch.lines[l].construction, sketch.lines[l].style),
                MetaEntity::Arc(a) => (sketch.arcs[a].construction, sketch.arcs[a].style),
                MetaEntity::Point(_) => (false, LineStyle::Solid),
            };
            if construction {
                match e {
                    MetaEntity::Line(l) => acts.push(Action::SetConstructionLine { line: l, on: true }),
                    MetaEntity::Arc(a) => acts.push(Action::SetConstructionArc { arc: a, on: true }),
                    MetaEntity::Point(_) => {}
                }
            }
            if style != LineStyle::Solid {
                match e {
                    MetaEntity::Line(l) => acts.push(Action::SetStyleLine { line: l, style }),
                    MetaEntity::Arc(a) => acts.push(Action::SetStyleArc { arc: a, style }),
                    MetaEntity::Point(_) => {}
                }
            }
        }
    }
    run_batch(runner, "Pattern constraints", acts)?;
    // The new constraints, by the copy they reference.
    let mut owner: HashMap<MetaEntity, usize> = HashMap::new();
    for (ci, entities) in entities_per_copy.iter().enumerate() {
        for &e in entities {
            owner.insert(e, ci);
        }
    }
    let mut per_copy: Vec<Vec<u32>> = vec![Vec::new(); plan.copies.len()];
    runner.sketch().for_each_constraint_collection_ref(|_, _, coll| {
        for i in 0..coll.len() {
            let c = coll.item(i);
            if c.nid() < n0 {
                continue;
            }
            let mut hit: Option<usize> = None;
            c.each_line_ref(&mut |l| if let Some(&ci) = owner.get(&MetaEntity::Line(l)) { hit = Some(ci); });
            c.each_arc_ref(&mut |a| if let Some(&ci) = owner.get(&MetaEntity::Arc(a)) { hit = Some(ci); });
            c.each_point_ref(&mut |p| if let Some(&ci) = owner.get(&MetaEntity::Point(p)) { hit = Some(ci); });
            if let Some(ci) = hit {
                per_copy[ci].push(c.nid());
            }
        }
    });
    let mut copies = Vec::with_capacity(plan.copies.len());
    for ((copy, entities), mut constraints) in plan.copies.iter().zip(entities_per_copy).zip(per_copy) {
        constraints.sort_unstable();
        out.entities.push(entities.iter().map(|e| entity_name(runner.sketch(), *e)).collect());
        out.constraints.extend(constraints.iter().map(|n| format!("C{}", n)));
        copies.push(PatternCopy { index: copy.index, entities, constraints });
    }
    Ok(copies)
}

/// The center point of a circular pattern: the point itself, or a hidden
/// helper made coincident with the endpoint.
/// The center point of a circular pattern: the point itself, or a hidden
/// helper made coincident with the endpoint. Returns (center point,
/// helper, helper coincidence nid).
fn center_point(runner: &mut dyn ActionRunner, plan: &PatternPlan) -> Result<(Option<Ref<Point>>, Option<Ref<Point>>, Vec<u32>), String> {
    let PatternSpec::Circular { center, .. } = &plan.params.kind else {
        return Ok((None, None, Vec::new()));
    };
    match center {
        CenterRef::Point(p) => Ok((Some(*p), None, Vec::new())),
        CenterRef::Endpoint(ep) => {
            let pos = plan.center.expect("planned");
            let created = runner.run(Action::AddHelperPoint { pos });
            if let Some(e) = runner.take_error() {
                return Err(format!("center: {}", e));
            }
            let Created::Point(h) = created else {
                return Err("center: the helper point was not added".into());
            };
            let action = Action::coincident(DimensionEndpoint::Point(h), *ep).expect("point pair");
            run_checked(runner, action, "center")?;
            Ok((Some(h), Some(h), vec![last_nid(runner)]))
        }
    }
}

fn record_kind(plan: &PatternPlan, helper: Option<Ref<Point>>) -> PatternKind {
    match &plan.params.kind {
        PatternSpec::Circular { center, distribution, angle, quantity } => PatternKind::Circular {
            center: *center,
            helper,
            distribution: *distribution,
            angle: angle.clone(),
            quantity: *quantity,
        },
        PatternSpec::Rectangular { frame, extent, axis1, axis2 } => PatternKind::Rectangular {
            frame: *frame,
            extent: *extent,
            axis1: axis1.clone(),
            axis2: axis2.clone(),
        },
    }
}

/// Create the planned pattern and register its meta-constraint, as one
/// undo group; a failure half-way rolls it back.
pub fn apply(runner: &mut dyn ActionRunner, plan: &PatternPlan) -> Result<PatternOutcome, String> {
    runner.begin_group();
    let r = apply_inner(runner, plan);
    if r.is_err() {
        runner.rollback_group();
    } else {
        runner.end_group();
    }
    r
}

fn apply_inner(runner: &mut dyn ActionRunner, plan: &PatternPlan) -> Result<PatternOutcome, String> {
    let mut out = PatternOutcome::default();
    let (center, helper, constraints) = center_point(runner, plan)?;
    let copies = make_copies(runner, plan, center, &mut out)?;
    let pattern = Pattern { sources: plan.sources.clone(), kind: record_kind(plan, helper), copies, constraints };
    runner.run(Action::RegisterMeta { meta: Meta { mid: 0, name: String::new(), kind: MetaKind::Pattern(pattern) } });
    if let Some(e) = runner.take_error() {
        return Err(format!("registering the pattern: {}", e));
    }
    let m = runner.sketch().metas.last().expect("just registered");
    out.name = m.name.clone();
    out.mid = m.mid;
    Ok(out)
}

// ---------------------------------------------------------------------------
// Editing
// ---------------------------------------------------------------------------

/// The parameters a record was made with.
pub fn params_of(p: &Pattern) -> PatternParams {
    let kind = match &p.kind {
        PatternKind::Circular { center, distribution, angle, quantity, .. } => PatternSpec::Circular {
            center: *center,
            distribution: *distribution,
            angle: angle.clone(),
            quantity: *quantity,
        },
        PatternKind::Rectangular { frame, extent, axis1, axis2 } => PatternSpec::Rectangular {
            frame: *frame,
            extent: *extent,
            axis1: axis1.clone(),
            axis2: axis2.clone(),
        },
    };
    PatternParams { kind }
}

fn pattern_record(sketch: &Sketch, mid: u32) -> Result<(String, Pattern), String> {
    let i = sketch.meta_index(mid).ok_or_else(|| format!("no meta-constraint M{}", mid))?;
    let m = &sketch.metas[i];
    match m.as_pattern() {
        Some(p) => Ok((m.name.clone(), p.clone())),
        None => Err(format!("{} is not a pattern", m.name)),
    }
}

/// Only the numbers changed (distance / angle): the same copies, moved.
fn same_structure(a: &PatternParams, b: &PatternParams) -> bool {
    match (&a.kind, &b.kind) {
        (
            PatternSpec::Circular { center: c1, distribution: d1, quantity: q1, .. },
            PatternSpec::Circular { center: c2, distribution: d2, quantity: q2, .. },
        ) => c1 == c2 && d1 == d2 && q1 == q2,
        (
            PatternSpec::Rectangular { frame: f1, extent: _, axis1: a1, axis2: b1 },
            PatternSpec::Rectangular { frame: f2, extent: _, axis1: a2, axis2: b2 },
        ) => f1 == f2 && a1.quantity == a2.quantity && a1.symmetric == a2.symmetric && b1.quantity == b2.quantity && b1.symmetric == b2.symmetric,
        _ => false,
    }
}

/// Set a copy's geometry to the plan's, so the solver settles there.
fn reseed(sketch: &mut Sketch, e: MetaEntity, g: &CopyGeom) {
    match (e, *g) {
        (MetaEntity::Line(l), CopyGeom::Line { p1, p2 }) => {
            let line = &mut sketch.lines[l];
            line.p1.value = p1;
            line.p2.value = p2;
        }
        (MetaEntity::Point(p), CopyGeom::Point { pos }) => sketch.points[p].pos.value = pos,
        (MetaEntity::Arc(a), CopyGeom::Arc { center, radius, radius_b, rotation, start, end, .. }) => {
            let arc = &mut sketch.arcs[a];
            arc.center.value = center;
            arc.radius.value = radius;
            arc.radius_b.value = radius_b;
            if arc.is_ellipse {
                arc.rotation.value = rotation;
            }
            if !arc.closed {
                arc.start_angle.value = start;
                arc.end_angle.value = end;
            }
        }
        _ => {}
    }
}

/// Change a pattern's parameters: new numbers move the copies in place
/// (the image constraints are rewritten), anything else rebuilds them.
/// One undo group, rolled back on a failure half-way.
pub fn update(runner: &mut dyn ActionRunner, mid: u32, params: &PatternParams) -> Result<PatternOutcome, String> {
    let (name, p) = pattern_record(runner.sketch(), mid)?;
    let new_plan = plan(runner.sketch(), &p.sources, params)?;
    runner.begin_group();
    let r = update_inner(runner, mid, &p, &new_plan);
    match r {
        Ok(mut out) => {
            runner.end_group();
            out.name = name;
            out.mid = mid;
            Ok(out)
        }
        Err(e) => {
            runner.rollback_group();
            Err(e)
        }
    }
}

fn update_inner(runner: &mut dyn ActionRunner, mid: u32, p: &Pattern, new_plan: &PatternPlan) -> Result<PatternOutcome, String> {
    let mut out = PatternOutcome::default();
    let old = params_of(p);
    if same_structure(&old, &new_plan.params) && p.copies.len() == new_plan.copies.len() {
        // In place: rewrite every image constraint, re-seed, record.
        let center = p.helper().or(match &p.kind {
            PatternKind::Circular { center: CenterRef::Point(pt), .. } => Some(*pt),
            _ => None,
        });
        let mut updates = Vec::new();
        for (copy, cplan) in p.copies.iter().zip(&new_plan.copies) {
            let xf = xf_of(new_plan, cplan, center);
            updates.extend(copy.constraints.iter().map(|&nid| (nid, xf)));
        }
        let sketch = runner.sketch_mut();
        for (copy, cplan) in p.copies.iter().zip(&new_plan.copies) {
            for (e, g) in copy.entities.iter().zip(&cplan.geoms) {
                reseed(sketch, *e, g);
            }
        }
        runner.run(Action::SetImageTransforms { updates });
        if let Some(e) = runner.take_error() { return Err(e); }
        let mut rec = p.clone();
        rec.kind = record_kind(new_plan, p.helper());
        register(runner, mid, rec)?;
        return Ok(out);
    }
    // Rebuild: forget the copies (record first, so the deletions are not
    // tampering), delete them and the helper, make the new ones.
    let mut rec = p.clone();
    let doomed: Vec<MetaEntity> = rec.copies.iter().flat_map(|c| c.entities.iter().copied()).collect();
    let old_helper = rec.helper();
    rec.copies.clear();
    rec.constraints.clear();
    rec.kind = record_kind(new_plan, None);
    register(runner, mid, rec)?;
    let mut deletes: Vec<Action> = doomed
        .into_iter()
        .filter(|e| entity_exists(runner.sketch(), *e))
        .map(crate::meta::delete_action)
        .collect();
    if let Some(h) = old_helper
        && runner.sketch().points.get(h).is_some()
    {
        deletes.push(Action::DeletePoint { point: h });
    }
    if !deletes.is_empty() {
        run_batch(runner, "Delete pattern copies", deletes)?;
    }
    let (center, helper, constraints) = center_point(runner, new_plan)?;
    let copies = make_copies(runner, new_plan, center, &mut out)?;
    let rec = Pattern { sources: p.sources.clone(), kind: record_kind(new_plan, helper), copies, constraints };
    register(runner, mid, rec)?;
    Ok(out)
}

fn register(runner: &mut dyn ActionRunner, mid: u32, p: Pattern) -> Result<(), String> {
    let name = runner.sketch().metas[runner.sketch().meta_index(mid).expect("registered")].name.clone();
    runner.run(Action::RegisterMeta { meta: Meta { mid, name, kind: MetaKind::Pattern(p) } });
    runner.take_error().map_or(Ok(()), Err)
}

// ---------------------------------------------------------------------------
// Listing
// ---------------------------------------------------------------------------

fn fmt_value(v: &MetaValue) -> String {
    match &v.expr {
        Some(e) => format!("{} ({:.4})", e, v.value),
        None => format!("{}", v.value),
    }
}

/// The center as the user named it.
pub fn center_name(sketch: &Sketch, c: &CenterRef) -> String {
    match c {
        CenterRef::Point(p) => sketch.points.get(*p).map(|p| p.name.clone()).unwrap_or("?".into()),
        CenterRef::Endpoint(e) => crate::meta::endpoint_name(sketch, e),
    }
}

/// One line describing a pattern, after the meta's name.
pub fn describe(sketch: &Sketch, p: &Pattern) -> String {
    let names = |es: &[MetaEntity]| -> String {
        es.iter()
            .filter(|e| entity_exists(sketch, **e))
            .map(|e| entity_name(sketch, *e))
            .collect::<Vec<_>>()
            .join(" ")
    };
    let what = match &p.kind {
        PatternKind::Circular { center, distribution, angle, quantity, .. } => {
            let dist = match distribution {
                Distribution::Full => "full".to_string(),
                Distribution::Partial => format!("partial {} deg", fmt_value(angle)),
                Distribution::Symmetric => format!("symmetric {} deg", fmt_value(angle)),
            };
            format!("circular pattern of {} about {}, {} {}", names(&p.sources), center_name(sketch, center), quantity, dist)
        }
        PatternKind::Rectangular { frame, extent, axis1, axis2 } => {
            let axis = |a: &PatternAxis| {
                format!(
                    "{} {} {}{}",
                    a.quantity,
                    if *extent { "over" } else { "every" },
                    fmt_value(&a.distance),
                    if a.symmetric { " symmetric" } else { "" }
                )
            };
            let mut s = format!("rectangular pattern of {}, {}", names(&p.sources), axis(axis1));
            if axis2.quantity > 1 {
                s += &format!(" x {}", axis(axis2));
            }
            if let Some(f) = frame {
                s += &format!(" along {}", sketch.lines.get(*f).map(|l| l.name.as_str()).unwrap_or("?"));
            }
            s
        }
    };
    let copies: Vec<String> = p
        .copies
        .iter()
        .map(|c| {
            let idx = match &p.kind {
                PatternKind::Circular { .. } => format!("#{}", c.index.0),
                PatternKind::Rectangular { .. } => format!("#{},{}", c.index.0, c.index.1),
            };
            format!("{}: {}", idx, names(&c.entities))
        })
        .collect();
    format!("{} -> {}", what, copies.join("; "))
}

/// Parse a distance / angle value as typed: a number or an expression.
pub fn parse_value(sketch: &Sketch, text: &str) -> Result<MetaValue, String> {
    crate::offset::parse_value(sketch, text)
}

/// Angle step between instances, for display.
pub fn angle_step_deg(distribution: Distribution, angle: f64, quantity: u32) -> f64 {
    match distribution {
        Distribution::Full => 360.0 / quantity.max(1) as f64,
        _ => if quantity > 1 { angle / (quantity - 1) as f64 } else { angle },
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::actions::Created;
    use crate::commands::CommandContext;
    use crate::corner_ops::ActionRunner;

    /// Counts group lifecycle calls; optionally refuses the n-th batch
    /// so an apply fails half-way.
    struct Recorder<'a> {
        inner: &'a mut CommandContext,
        begins: usize,
        ends: usize,
        rollbacks: usize,
        fail_batch: Option<usize>,
        seen: usize,
        failed: bool,
    }

    impl ActionRunner for Recorder<'_> {
        fn sketch(&self) -> &Sketch { self.inner.sketch() }
        fn sketch_mut(&mut self) -> &mut Sketch { self.inner.sketch_mut() }
        fn run(&mut self, action: Action) -> Created {
            if matches!(action, Action::Batch { .. }) {
                self.seen += 1;
                if Some(self.seen) == self.fail_batch {
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
        fn begin_group(&mut self) { self.begins += 1; self.inner.begin_group() }
        fn end_group(&mut self) { self.ends += 1; self.inner.end_group() }
        fn rollback_group(&mut self) { self.rollbacks += 1; self.inner.rollback_group() }
    }

    fn line_plan(ctx: &CommandContext) -> PatternPlan {
        let sources = vec![MetaEntity::Line(ctx.sketch.lines.refs().next().unwrap())];
        let params = PatternParams {
            kind: PatternSpec::Rectangular {
                frame: None,
                extent: false,
                axis1: PatternAxis { quantity: 3, distance: MetaValue { value: 10.0, expr: None }, symmetric: false },
                axis2: PatternAxis { quantity: 1, distance: MetaValue { value: 0.0, expr: None }, symmetric: false },
            },
        };
        plan(&ctx.sketch, &sources, &params).unwrap()
    }

    fn ctx() -> CommandContext {
        let mut ctx = CommandContext::new();
        for r in crate::commands::execute(&mut ctx, "add_line 0,0 1,0") {
            assert!(!r.is_error, "{}", r.output);
        }
        ctx
    }

    #[test]
    fn apply_ends_the_group_on_success() {
        let mut ctx = ctx();
        let p = line_plan(&ctx);
        let mut rec = Recorder { inner: &mut ctx, begins: 0, ends: 0, rollbacks: 0, fail_batch: None, seen: 0, failed: false };
        apply(&mut rec, &p).unwrap();
        assert_eq!((rec.begins, rec.ends, rec.rollbacks), (1, 1, 0));
    }

    #[test]
    fn failed_apply_rolls_back_without_end() {
        let mut ctx = ctx();
        let p = line_plan(&ctx);
        let mut rec = Recorder { inner: &mut ctx, begins: 0, ends: 0, rollbacks: 0, fail_batch: Some(1), seen: 0, failed: false };
        assert!(apply(&mut rec, &p).is_err());
        assert_eq!((rec.begins, rec.ends, rec.rollbacks), (1, 0, 1));
        assert_eq!(rec.inner.sketch.lines.refs().count(), 1, "rolled back to the source line");
    }
}
