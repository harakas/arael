//! Fillet/chamfer engine, typed and front-end neutral: the `fillet` /
//! `chamfer` commands and the GUI's corner tools all call
//! [`apply_corner_ops`] through the [`ActionRunner`] abstraction, so
//! the corner math and the action sequence exist once.

use arael::refs::Ref;
use arael::vect::vect2d;
use arael_sketch_solver::*;

use crate::actions::{Action, Created};
use crate::geometry::{arc_end_pos, arc_start_pos};
use crate::ids::ConstraintId;

/// Executor abstraction over the two front ends. CommandContext and
/// the GUI's EditorApp both run actions through their own gate
/// (validate, history, solve); composite operations program against
/// this trait so they behave identically from either side.
pub trait ActionRunner {
    fn sketch(&self) -> &Sketch;
    /// Structural mutation access (the endpoint trims of a corner op).
    fn sketch_mut(&mut self) -> &mut Sketch;
    fn run(&mut self, action: Action) -> Created;
    /// Run with the DOF-redundancy gate suppressed -- for coincidents
    /// that re-tie just-created geometry.
    fn run_unchecked(&mut self, action: Action) -> Created;
    /// Take the rejection message of the last `run`, if any.
    fn take_error(&mut self) -> Option<String>;
    fn begin_group(&mut self);
    /// The operation since `begin_group` completed: the runner may now
    /// run once-per-operation work it deferred during the group (the
    /// GUI refreshes its DOF display here). Every `begin_group` must be
    /// paired with `end_group` on success or `rollback_group` on
    /// failure.
    fn end_group(&mut self);
    /// Undo and forget everything run since `begin_group`: an operation
    /// that failed half-way leaves nothing behind.
    fn rollback_group(&mut self);
}

/// A corner named by what the user pointed at: one line endpoint, or
/// two lines whose shared corner is looked up at apply time.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CornerSpec {
    Endpoint { line: Ref<Line>, is_p1: bool },
    Lines(Ref<Line>, Ref<Line>),
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum CornerKind {
    Fillet,
    Chamfer,
}

pub struct CornerOpConfig {
    pub kind: CornerKind,
    pub radius: f64,
    /// Live expression for the primary corner's dimension; secondary
    /// corners reference the primary dim by name regardless.
    pub radius_expr: Option<String>,
    pub notangent: bool,
    pub noradius: bool,
}

/// What one corner produced, for the caller's report.
#[derive(Default)]
pub struct CornerOutcome {
    /// The created arc (fillet) or bevel line (chamfer). Empty when
    /// the corner failed.
    pub entity_name: String,
    /// Chamfer's corner anchor point.
    pub point_name: Option<String>,
    /// User-visible name of the deleted corner coincident.
    pub removed: Option<String>,
    /// Radius dim (fillet) / first leg dim (chamfer).
    pub primary_dim: Option<String>,
    /// Chamfer's second leg dim.
    pub secondary_dim: Option<String>,
    /// Added constraint ids and skip notes, in application order.
    pub added: Vec<String>,
    /// Failure reason; when set, the corner left no marks.
    pub error: Option<String>,
}

impl CornerOutcome {
    fn failed(e: String) -> Self {
        CornerOutcome { error: Some(e), ..Default::default() }
    }
}

pub struct CornerOpsResult {
    /// One entry per requested corner, in order.
    pub outcomes: Vec<CornerOutcome>,
    /// The first corner's dimension: later corners track it and the
    /// GUI's radius overlay edits it live.
    pub primary_dim_name: Option<String>,
    pub primary_dim_did: Option<u32>,
}

/// Refs resolved from a corner spec: the two lines that meet at the
/// corner, which endpoint of each is at the corner, and the coincident
/// constraint that ties them there.
pub struct CornerRefs {
    pub line_a: Ref<Line>,
    pub is_p1_a: bool,
    pub line_b: Ref<Line>,
    pub is_p1_b: bool,
    pub coincident_id: ConstraintId,
}

/// Scan the four line-line coincident collections for a constraint
/// involving `(line, is_p1)`. Returns the partner line, which of its
/// endpoints is the shared corner, and the ConstraintId so the caller
/// can delete the coincidence during a fillet or similar topology
/// edit. Returns None if the endpoint isn't coincident with any other
/// line's endpoint directly -- shared-via-helper-point corners are
/// rejected on purpose, keeping the corner-op scope tractable.
pub fn find_ll_coincident_partner(
    sketch: &Sketch,
    line: Ref<Line>,
    is_p1: bool,
) -> Option<(Ref<Line>, bool, ConstraintId)> {
    // LL11: c.a.p1 = c.b.p1
    for c in sketch.coincident_ll11.iter() {
        if c.a == line && is_p1 { return Some((c.b, true, ConstraintId::Numbered(c.nid))); }
        if c.b == line && is_p1 { return Some((c.a, true, ConstraintId::Numbered(c.nid))); }
    }
    // LL12: c.a.p1 = c.b.p2
    for c in sketch.coincident_ll12.iter() {
        if c.a == line && is_p1 { return Some((c.b, false, ConstraintId::Numbered(c.nid))); }
        if c.b == line && !is_p1 { return Some((c.a, true, ConstraintId::Numbered(c.nid))); }
    }
    // LL21: c.a.p2 = c.b.p1
    for c in sketch.coincident_ll21.iter() {
        if c.a == line && !is_p1 { return Some((c.b, true, ConstraintId::Numbered(c.nid))); }
        if c.b == line && is_p1 { return Some((c.a, false, ConstraintId::Numbered(c.nid))); }
    }
    // LL22: c.a.p2 = c.b.p2
    for c in sketch.coincident_ll22.iter() {
        if c.a == line && !is_p1 { return Some((c.b, false, ConstraintId::Numbered(c.nid))); }
        if c.b == line && !is_p1 { return Some((c.a, false, ConstraintId::Numbered(c.nid))); }
    }
    None
}

/// Resolve a corner spec against the current sketch. Re-run per
/// corner during a multi-corner apply: every fillet deletes one LL
/// coincident, shifting what a later corner resolves to.
pub fn resolve_corner_spec(sketch: &Sketch, spec: &CornerSpec) -> Result<CornerRefs, String> {
    match *spec {
        CornerSpec::Endpoint { line, is_p1 } => {
            let name = sketch
                .lines
                .get(line)
                .map(|l| format!("{}.p{}", l.name, if is_p1 { 1 } else { 2 }))
                .ok_or("corner line no longer exists")?;
            match find_ll_coincident_partner(sketch, line, is_p1) {
                Some((lb, is_p1_b, cid)) => Ok(CornerRefs {
                    line_a: line, is_p1_a: is_p1, line_b: lb, is_p1_b, coincident_id: cid,
                }),
                None => Err(format!("{} isn't coincident with another line endpoint", name)),
            }
        }
        CornerSpec::Lines(la, lb) => {
            if la == lb {
                return Err("two-line corner needs two different lines".into());
            }
            let (na, nb) = match (sketch.lines.get(la), sketch.lines.get(lb)) {
                (Some(a), Some(b)) => (a.name.clone(), b.name.clone()),
                _ => return Err("corner line no longer exists".into()),
            };
            for probe in [true, false] {
                if let Some((p, is_p1_b, cid)) = find_ll_coincident_partner(sketch, la, probe)
                    && p == lb
                {
                    return Ok(CornerRefs {
                        line_a: la, is_p1_a: probe, line_b: lb, is_p1_b, coincident_id: cid,
                    });
                }
            }
            Err(format!("{} and {} are not connected at an endpoint", na, nb))
        }
    }
}

/// Apply fillets or chamfers at the given corners as one undo group.
/// The first corner's dimension carries `radius` / `radius_expr`;
/// later corners reference it by name so every corner tracks a single
/// source. A corner that fails is reported in its outcome and does
/// not abort the others.
pub fn apply_corner_ops(
    runner: &mut dyn ActionRunner,
    cfg: &CornerOpConfig,
    corners: &[CornerSpec],
) -> CornerOpsResult {
    runner.begin_group();
    let mut outcomes: Vec<CornerOutcome> = Vec::new();
    let mut primary_dim_name: Option<String> = None;
    for (idx, spec) in corners.iter().enumerate() {
        let refs = match resolve_corner_spec(runner.sketch(), spec) {
            Ok(r) => r,
            Err(e) => {
                outcomes.push(CornerOutcome::failed(e));
                continue;
            }
        };
        let (value, expr) = if idx == 0 {
            (cfg.radius, cfg.radius_expr.clone())
        } else if let Some(name) = &primary_dim_name {
            let v = crate::commands::eval_expr(runner.sketch(), name).unwrap_or(cfg.radius);
            (v, Some(name.clone()))
        } else {
            (cfg.radius, None)
        };
        let out = match cfg.kind {
            CornerKind::Fillet => {
                apply_one_fillet(runner, &refs, value, expr, cfg.notangent, cfg.noradius)
            }
            CornerKind::Chamfer => apply_one_chamfer(runner, &refs, value, expr),
        };
        match out {
            Ok(o) => {
                if idx == 0 {
                    primary_dim_name = o.primary_dim.clone();
                }
                outcomes.push(o);
            }
            Err(e) => outcomes.push(CornerOutcome::failed(e)),
        }
    }
    runner.end_group();
    let primary_dim_did = primary_dim_name.as_ref().and_then(|n| {
        runner.sketch().dimensions.iter().find(|d| &d.name == n).map(|d| d.did)
    });
    CornerOpsResult { outcomes, primary_dim_name, primary_dim_did }
}

fn last_dim_name(runner: &dyn ActionRunner) -> String {
    runner.sketch().dimensions.last().map(|d| d.name.clone()).unwrap_or_default()
}

/// Apply one fillet at the resolved corner. Returns the ids of every
/// new constraint and the deleted coincident so the caller can
/// surface them. Err (without mutating) for geometry errors: too
/// short, zero-angle, collinear.
fn apply_one_fillet(
    runner: &mut dyn ActionRunner,
    refs: &CornerRefs,
    radius: f64,
    radius_expr: Option<String>,
    notangent: bool,
    noradius: bool,
) -> Result<CornerOutcome, String> {
    let CornerRefs { line_a, is_p1_a, line_b, is_p1_b, coincident_id } = *refs;
    if radius <= 1e-9 {
        return Err("fillet radius must be positive".into());
    }

    // Geometry: unit direction vectors from the shared corner toward
    // the far endpoint of each line.
    let la_ref = &runner.sketch().lines[line_a];
    let lb_ref = &runner.sketch().lines[line_b];
    let (corner_a, far_a) = if is_p1_a { (la_ref.p1.value, la_ref.p2.value) } else { (la_ref.p2.value, la_ref.p1.value) };
    let (corner_b, far_b) = if is_p1_b { (lb_ref.p1.value, lb_ref.p2.value) } else { (lb_ref.p2.value, lb_ref.p1.value) };
    // Use the corners' mean so small solver residuals don't bias
    // the geometry.
    let corner = vect2d::new((corner_a.x + corner_b.x) * 0.5, (corner_a.y + corner_b.y) * 0.5);
    let dx_a = far_a.x - corner.x;
    let dy_a = far_a.y - corner.y;
    let len_a = (dx_a * dx_a + dy_a * dy_a).sqrt();
    let dx_b = far_b.x - corner.x;
    let dy_b = far_b.y - corner.y;
    let len_b = (dx_b * dx_b + dy_b * dy_b).sqrt();
    if len_a < 1e-9 || len_b < 1e-9 {
        return Err("one of the lines has zero length".into());
    }
    let ua = vect2d::new(dx_a / len_a, dy_a / len_a);
    let ub = vect2d::new(dx_b / len_b, dy_b / len_b);
    let cos_theta = ua.x * ub.x + ua.y * ub.y;
    if cos_theta >= 1.0 - 1e-6 {
        return Err("lines overlap at the corner (zero angle)".into());
    }
    if cos_theta <= -1.0 + 1e-6 {
        return Err("lines are collinear at the corner (no fillet possible)".into());
    }
    let half_theta = (cos_theta.acos()) * 0.5;
    let tan_half = half_theta.tan();
    let sin_half = half_theta.sin();
    let trim_dist = radius / tan_half;
    if trim_dist + 1e-9 >= len_a || trim_dist + 1e-9 >= len_b {
        return Err(format!(
            "lines too short for radius {} (need {:.4} on each side; have {:.4} and {:.4})",
            radius, trim_dist, len_a, len_b));
    }
    let t_a = vect2d::new(corner.x + ua.x * trim_dist, corner.y + ua.y * trim_dist);
    let t_b = vect2d::new(corner.x + ub.x * trim_dist, corner.y + ub.y * trim_dist);
    let bis_x = ua.x + ub.x;
    let bis_y = ua.y + ub.y;
    let bis_len = (bis_x * bis_x + bis_y * bis_y).sqrt();
    let bis = vect2d::new(bis_x / bis_len, bis_y / bis_len);
    let center_dist = radius / sin_half;
    let arc_center = vect2d::new(corner.x + bis.x * center_dist, corner.y + bis.y * center_dist);
    let mid = vect2d::new(arc_center.x - bis.x * radius, arc_center.y - bis.y * radius);

    let mut added: Vec<String> = Vec::new();

    // Capture the deleted coincident's user-visible name BEFORE the
    // delete so we can surface it alongside the new ids.
    let removed = crate::ids::constraint_id_name(runner.sketch(), coincident_id);
    runner.run(Action::DeleteConstraint { id: coincident_id });

    if is_p1_a { runner.sketch_mut().lines[line_a].p1.value = t_a; }
    else { runner.sketch_mut().lines[line_a].p2.value = t_a; }
    if is_p1_b { runner.sketch_mut().lines[line_b].p1.value = t_b; }
    else { runner.sketch_mut().lines[line_b].p2.value = t_b; }

    let Some(arc_ref) = runner.run(Action::AddArc { start: t_a, end: t_b, mid }).arc() else {
        return Err("Cannot fillet: degenerate corner geometry".into());
    };
    let arc_name = runner.sketch().arcs[arc_ref].name.clone();

    let arc_start = arc_start_pos(&runner.sketch().arcs[arc_ref]);
    let arc_end = arc_end_pos(&runner.sketch().arcs[arc_ref]);
    let a_to_start = (t_a.x - arc_start.x).powi(2) + (t_a.y - arc_start.y).powi(2);
    let a_to_end = (t_a.x - arc_end.x).powi(2) + (t_a.y - arc_end.y).powi(2);
    let a_matches_start = a_to_start <= a_to_end;

    let coincide_a = match (is_p1_a, a_matches_start) {
        (true, true) => Action::ApplyCoincidentLP1ArcStart { line: line_a, arc: arc_ref },
        (true, false) => Action::ApplyCoincidentLP1ArcEnd { line: line_a, arc: arc_ref },
        (false, true) => Action::ApplyCoincidentLP2ArcStart { line: line_a, arc: arc_ref },
        (false, false) => Action::ApplyCoincidentLP2ArcEnd { line: line_a, arc: arc_ref },
    };
    let coincide_b = match (is_p1_b, a_matches_start) {
        (true, true) => Action::ApplyCoincidentLP1ArcEnd { line: line_b, arc: arc_ref },
        (true, false) => Action::ApplyCoincidentLP1ArcStart { line: line_b, arc: arc_ref },
        (false, true) => Action::ApplyCoincidentLP2ArcEnd { line: line_b, arc: arc_ref },
        (false, false) => Action::ApplyCoincidentLP2ArcStart { line: line_b, arc: arc_ref },
    };
    // Look up the most recently added constraint in the collection
    // whose kind matches `action` and format its id.
    let bridge_name = |runner: &dyn ActionRunner, action: &Action| -> Option<String> {
        let s = runner.sketch();
        match action {
            Action::ApplyCoincidentLP1ArcStart { .. } => s.coincident_lp1_arc_start.last().map(|c| format!("C{}", c.nid)),
            Action::ApplyCoincidentLP1ArcEnd { .. } => s.coincident_lp1_arc_end.last().map(|c| format!("C{}", c.nid)),
            Action::ApplyCoincidentLP2ArcStart { .. } => s.coincident_lp2_arc_start.last().map(|c| format!("C{}", c.nid)),
            Action::ApplyCoincidentLP2ArcEnd { .. } => s.coincident_lp2_arc_end.last().map(|c| format!("C{}", c.nid)),
            _ => None,
        }
    };
    let ca = coincide_a.clone();
    runner.run_unchecked(coincide_a);
    if let Some(n) = bridge_name(runner, &ca) { added.push(n); }
    let cb = coincide_b.clone();
    runner.run_unchecked(coincide_b);
    if let Some(n) = bridge_name(runner, &cb) { added.push(n); }

    if !notangent {
        // A rejected tangent is reported in the corner's outcome, not
        // silently dropped.
        for line in [line_a, line_b] {
            runner.run(Action::ApplyTangentLA { line, arc: arc_ref });
            match runner.take_error() {
                None => {
                    if let Some(c) = runner.sketch().tangent_la.last() {
                        added.push(format!("C{}", c.nid));
                    }
                }
                Some(e) => added.push(format!(
                    "tangent {} skipped ({})", runner.sketch().lines[line].name, e)),
            }
        }
    }

    let mut dim_name: Option<String> = None;
    if !noradius {
        runner.run(Action::AddDimension {
            kind: DimensionKind::ArcRadius(arc_ref),
            value: radius, expr: radius_expr, derived: false, range: None,
        });
        match runner.take_error() {
            None => dim_name = Some(last_dim_name(runner)),
            Some(e) => added.push(format!("radius dim skipped ({})", e)),
        }
    }

    Ok(CornerOutcome {
        entity_name: arc_name,
        point_name: None,
        removed,
        primary_dim: dim_name,
        secondary_dim: None,
        added,
        error: None,
    })
}

/// Apply one chamfer at the resolved corner: trim both lines back by
/// `distance`, span a bevel line, anchor a point at the old corner
/// via PointOnLine on both lines, and dim both legs (the second leg
/// tracks the first by expression).
fn apply_one_chamfer(
    runner: &mut dyn ActionRunner,
    refs: &CornerRefs,
    distance: f64,
    dist_expr: Option<String>,
) -> Result<CornerOutcome, String> {
    let CornerRefs { line_a, is_p1_a, line_b, is_p1_b, coincident_id } = *refs;
    if distance <= 1e-9 {
        return Err("chamfer distance must be positive".into());
    }
    let la_ref = &runner.sketch().lines[line_a];
    let lb_ref = &runner.sketch().lines[line_b];
    let (corner_a, far_a) = if is_p1_a { (la_ref.p1.value, la_ref.p2.value) } else { (la_ref.p2.value, la_ref.p1.value) };
    let (corner_b, far_b) = if is_p1_b { (lb_ref.p1.value, lb_ref.p2.value) } else { (lb_ref.p2.value, lb_ref.p1.value) };
    let corner = vect2d::new((corner_a.x + corner_b.x) * 0.5, (corner_a.y + corner_b.y) * 0.5);
    let dx_a = far_a.x - corner.x;
    let dy_a = far_a.y - corner.y;
    let len_a = (dx_a * dx_a + dy_a * dy_a).sqrt();
    let dx_b = far_b.x - corner.x;
    let dy_b = far_b.y - corner.y;
    let len_b = (dx_b * dx_b + dy_b * dy_b).sqrt();
    if len_a < 1e-9 || len_b < 1e-9 {
        return Err("one of the lines has zero length".into());
    }
    let ua = vect2d::new(dx_a / len_a, dy_a / len_a);
    let ub = vect2d::new(dx_b / len_b, dy_b / len_b);
    let cos_theta = ua.x * ub.x + ua.y * ub.y;
    if cos_theta >= 1.0 - 1e-6 {
        return Err("lines overlap at the corner (zero angle)".into());
    }
    if cos_theta <= -1.0 + 1e-6 {
        return Err("lines are collinear at the corner (no chamfer possible)".into());
    }
    if distance + 1e-9 >= len_a || distance + 1e-9 >= len_b {
        return Err(format!(
            "lines too short for distance {} (have {:.4} and {:.4})",
            distance, len_a, len_b));
    }
    let t_a = vect2d::new(corner.x + ua.x * distance, corner.y + ua.y * distance);
    let t_b = vect2d::new(corner.x + ub.x * distance, corner.y + ub.y * distance);

    let mut added: Vec<String> = Vec::new();

    let removed = crate::ids::constraint_id_name(runner.sketch(), coincident_id);
    runner.run(Action::DeleteConstraint { id: coincident_id });

    if is_p1_a { runner.sketch_mut().lines[line_a].p1.value = t_a; }
    else { runner.sketch_mut().lines[line_a].p2.value = t_a; }
    if is_p1_b { runner.sketch_mut().lines[line_b].p1.value = t_b; }
    else { runner.sketch_mut().lines[line_b].p2.value = t_b; }

    let Some(point_ref) = runner.run(Action::AddPoint { pos: corner }).point() else {
        return Err(runner.take_error().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    let point_name = runner.sketch().points[point_ref].name.clone();

    let Some(new_line_ref) = runner.run(Action::AddLine { p1: t_a, p2: t_b }).line() else {
        return Err(runner.take_error().unwrap_or_else(|| "Internal: creation action added no entity".into()));
    };
    let new_line_name = runner.sketch().lines[new_line_ref].name.clone();

    let coincide_a = if is_p1_a {
        Action::ApplyCoincidentLL11 { a: line_a, b: new_line_ref }
    } else {
        Action::ApplyCoincidentLL21 { a: line_a, b: new_line_ref }
    };
    let coincide_b = if is_p1_b {
        Action::ApplyCoincidentLL12 { a: line_b, b: new_line_ref }
    } else {
        Action::ApplyCoincidentLL22 { a: line_b, b: new_line_ref }
    };
    let ca = coincide_a.clone();
    runner.run_unchecked(coincide_a);
    let ca_id = match ca {
        Action::ApplyCoincidentLL11 { .. } => runner.sketch().coincident_ll11.last().map(|c| format!("C{}", c.nid)),
        Action::ApplyCoincidentLL21 { .. } => runner.sketch().coincident_ll21.last().map(|c| format!("C{}", c.nid)),
        _ => None,
    };
    if let Some(n) = ca_id { added.push(n); }
    let cb = coincide_b.clone();
    runner.run_unchecked(coincide_b);
    let cb_id = match cb {
        Action::ApplyCoincidentLL12 { .. } => runner.sketch().coincident_ll12.last().map(|c| format!("C{}", c.nid)),
        Action::ApplyCoincidentLL22 { .. } => runner.sketch().coincident_ll22.last().map(|c| format!("C{}", c.nid)),
        _ => None,
    };
    if let Some(n) = cb_id { added.push(n); }
    runner.run_unchecked(Action::ApplyPointOnLine { point: point_ref, line: line_a });
    if let Some(c) = runner.sketch().point_on_line.last() { added.push(format!("C{}", c.nid)); }
    runner.run_unchecked(Action::ApplyPointOnLine { point: point_ref, line: line_b });
    if let Some(c) = runner.sketch().point_on_line.last() { added.push(format!("C{}", c.nid)); }

    let ep_a = if is_p1_a { DimensionEndpoint::LineP1(line_a) } else { DimensionEndpoint::LineP2(line_a) };
    let ep_b = if is_p1_b { DimensionEndpoint::LineP1(line_b) } else { DimensionEndpoint::LineP2(line_b) };
    runner.run(Action::AddDimension {
        kind: DimensionKind::PointPointDistance(DimensionEndpoint::Point(point_ref), ep_a),
        value: distance, expr: dist_expr, derived: false, range: None,
    });
    let primary_dim = Some(last_dim_name(runner));

    runner.run(Action::AddDimension {
        kind: DimensionKind::PointPointDistance(DimensionEndpoint::Point(point_ref), ep_b),
        value: distance, expr: primary_dim.clone(), derived: false, range: None,
    });
    let secondary_dim = Some(last_dim_name(runner));

    Ok(CornerOutcome {
        entity_name: new_line_name,
        point_name: Some(point_name),
        removed,
        primary_dim,
        secondary_dim,
        added,
        error: None,
    })
}

// ---------------------------------------------------------------------------
// Runner impls
// ---------------------------------------------------------------------------

impl ActionRunner for crate::commands::CommandContext {
    fn sketch(&self) -> &Sketch {
        &self.sketch
    }
    fn sketch_mut(&mut self) -> &mut Sketch {
        self.sketch.get_mut()
    }
    fn run(&mut self, action: Action) -> Created {
        self.exec(action)
    }
    fn run_unchecked(&mut self, action: Action) -> Created {
        let saved = self.skip_dof_check;
        self.skip_dof_check = true;
        let created = self.exec(action);
        self.skip_dof_check = saved;
        created
    }
    fn take_error(&mut self) -> Option<String> {
        self.status_error.take()
    }
    fn begin_group(&mut self) {
        crate::commands::CommandContext::begin_group(self)
    }
    fn end_group(&mut self) {}
    fn rollback_group(&mut self) {
        if let Some(s) = self.history.discard_current_group() {
            self.sketch = s.into();
            crate::commands::prune_selection(self);
        }
    }
}
